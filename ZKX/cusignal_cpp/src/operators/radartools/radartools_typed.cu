#include <cusignal/operators/radartools/radartools_typed.h>
#include "radartools_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cusignal {
namespace {
int next_power_of_two(int n){int p=1;while(p<n)p<<=1;return p;}

DeviceArray<float>& ca_cfar_cumulative_workspace(std::size_t required_elements)
{
    // CA-CFAR only supports the default stream.  A host thread can therefore
    // safely reuse its integral-image storage across ordered calls without
    // changing kernel order or exposing a new public workspace contract.
    thread_local DeviceArray<float> workspace;
    if (workspace.size() < required_elements) {
        workspace.reset(required_elements);
    }
    return workspace;
}

}  // namespace


namespace detail {

template<class T>
void ambgfun_fp32_compute_device(
    const DeviceArray<TypedComplex<T>>& x,
    const DeviceArray<TypedComplex<T>>& y,
    int nx,
    int ny,
    int cut,
    float fs,
    float cut_value,
    DeviceArray<float>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (x.empty() || y.empty() || nx < 1 || ny < 1 ||
        nx > static_cast<int>(x.size()) || ny > static_cast<int>(y.size()) ||
        cut < 0 || cut > 2)
        throw std::invalid_argument("ambgfun compute parameters are invalid");
    const int rows = nx + ny - 1;
    const int nfreq = next_power_of_two(rows);
    const std::size_t expected = cut == 0
        ? static_cast<std::size_t>(rows) * static_cast<std::size_t>(nfreq)
        : static_cast<std::size_t>(cut == 1 ? nfreq : rows);
    if (output.size() != expected)
        throw std::invalid_argument("ambgfun compute output shape mismatch");

    DeviceArray<float> x_norm(1);
    DeviceArray<float> y_norm(1);
    cuda_utils::launch_1d_kernel(
        radar_detail::ambiguity_norm_kernel<T>, 1,
        x.data(), static_cast<int>(x.size()), x_norm.data());
    cuda_utils::launch_1d_kernel(
        radar_detail::ambiguity_norm_kernel<T>, 1,
        y.data(), static_cast<int>(y.size()), y_norm.data());

    if (cut == 0) {
        DeviceArray<ComplexFloat> delay_product(expected);
        DeviceArray<ComplexFloat> transformed(expected);
        cuda_utils::launch_1d_kernel(
            radar_detail::ambiguity_build_delay_kernel<T>, expected,
            x.data(), nx, y.data(), ny, rows, nfreq,
            x_norm.data(), y_norm.data(), delay_product.data());
        FFTInterface fft(nfreq, FFTInterface::BatchOnly{});
        fft.ifft_batch_device(delay_product, transformed, rows);
        cuda_utils::launch_1d_kernel(
            radar_detail::ambiguity_magnitude_shift_kernel, expected,
            transformed.data(), rows, nfreq, output.data());
        return;
    }

    if (cut == 1) {
        DeviceArray<ComplexFloat> x_time(nfreq);
        DeviceArray<ComplexFloat> x_frequency(nfreq);
        DeviceArray<ComplexFloat> product(nfreq);
        DeviceArray<ComplexFloat> transformed(nfreq);
        cuda_utils::launch_1d_kernel(
            radar_detail::ambiguity_pack_normalized_kernel<T>, nfreq,
            x.data(), nx, nfreq, x_norm.data(), 0.0F, fs, false, x_time.data());
        FFTInterface fft(nfreq, FFTInterface::BatchOnly{});
        fft.fft_batch_device(x_time, x_frequency, 1);
        cuda_utils::launch_1d_kernel(
            radar_detail::ambiguity_delay_phase_kernel, nfreq,
            x_frequency.data(), nfreq, fs, cut_value);
        fft.ifft_batch_device(x_frequency, transformed, 1);
        cuda_utils::launch_1d_kernel(
            radar_detail::ambiguity_delay_product_kernel<T>, nfreq,
            y.data(), ny, y_norm.data(), transformed.data(), nfreq,
            product.data());
        fft.ifft_batch_device(product, transformed, 1);
        cuda_utils::launch_1d_kernel(
            radar_detail::ambiguity_cut_magnitude_kernel, nfreq,
            transformed.data(), nfreq, true,
            static_cast<float>(nfreq), output.data());
        return;
    }

    DeviceArray<ComplexFloat> x_time(rows);
    DeviceArray<ComplexFloat> y_time(rows);
    DeviceArray<ComplexFloat> x_frequency(rows);
    DeviceArray<ComplexFloat> y_frequency(rows);
    DeviceArray<ComplexFloat> product(rows);
    DeviceArray<ComplexFloat> transformed(rows);
    cuda_utils::launch_1d_kernel(
        radar_detail::ambiguity_pack_normalized_kernel<T>, rows,
        x.data(), nx, rows, x_norm.data(), cut_value, fs, true, x_time.data());
    cuda_utils::launch_1d_kernel(
        radar_detail::ambiguity_pack_normalized_kernel<T>, rows,
        y.data(), ny, rows, y_norm.data(), 0.0F, fs, false, y_time.data());
    FFTInterface fft(rows, FFTInterface::BatchOnly{});
    fft.fft_batch_device(x_time, x_frequency, 1);
    fft.fft_batch_device(y_time, y_frequency, 1);
    cuda_utils::launch_1d_kernel(
        radar_detail::ambiguity_frequency_product_kernel, rows,
        y_frequency.data(), x_frequency.data(), rows, product.data());
    fft.ifft_batch_device(product, transformed, 1);
    cuda_utils::launch_1d_kernel(
        radar_detail::ambiguity_cut_magnitude_kernel, rows,
        transformed.data(), rows, false, 1.0F, output.data());
}

template<class T>
void ca_cfar_fp32_compute_device(
    const DeviceArray<T>& input,
    int rank,
    int rows,
    int columns,
    int guard_rows,
    int guard_columns,
    int reference_rows,
    int reference_columns,
    float alpha,
    DeviceArray<float>& threshold,
    DeviceArray<CfarDetection>& detections)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (rank < 0 || rank > 2 || rows < 0 || columns < 0 ||
        input.size() != threshold.size() || input.size() != detections.size() ||
        input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ca_cfar compute shape mismatch");
    }
    if (input.empty()) return;
    DeviceArray<float>& cumulative = ca_cfar_cumulative_workspace(input.size());
    if (rank == 1) {
        cuda_utils::launch_1d_kernel(
            radar_detail::ca_cfar_column_cumsum_kernel<T>, 1,
            input.data(), rows, 1, cumulative.data());
    } else if (rank == 2) {
        cuda_utils::launch_1d_kernel(
            radar_detail::ca_cfar_column_cumsum_kernel<T>,
            static_cast<std::size_t>(columns),
            input.data(), rows, columns, cumulative.data());
        cuda_utils::launch_1d_kernel(
            radar_detail::ca_cfar_row_cumsum_kernel,
            static_cast<std::size_t>(rows),
            cumulative.data(), rows, columns);
    }
    cuda_utils::launch_1d_kernel(
        radar_detail::ca_cfar_integral_threshold_kernel<T>, input.size(),
        input.data(), cumulative.data(), static_cast<int>(input.size()),
        rank, rows, columns,
        guard_rows, guard_columns, reference_rows, reference_columns,
        alpha, threshold.data(), detections.data());
}

template <class T>
void cfar_alpha_fp32_compute_device(
    T pfa, int reference, DeviceArray<float>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (output.size() != 1)
        throw std::invalid_argument("cfar_alpha compute output must be scalar");
    cuda_utils::launch_1d_kernel(
        radar_detail::cfar_alpha_kernel<T>, 1,
        pfa, reference, output.data());
}

template <class T>
void pulse_compression_fp32_compute_device(
    const DeviceArray<TypedComplex<T>>& input,
    const DeviceArray<TypedComplex<T>>& pulse_template,
    const PulseCompressionOptions& options,
    const DeviceArray<float>* explicit_window,
    DeviceArray<ComplexFloat>& output)
{
    const int nfft = options.nfft == -1 ? options.samples_per_pulse : options.nfft;
    const int window_kind = static_cast<int>(options.window);
    const float* window = explicit_window == nullptr ? nullptr : explicit_window->data();
    DeviceArray<float> norm(1);
    DeviceArray<ComplexFloat> template_time(nfft), template_frequency(nfft);
    DeviceArray<ComplexFloat> input_time(output.size()), input_frequency(output.size());
    DeviceArray<ComplexFloat> product_frequency(output.size());
    if (options.normalize) {
        cuda_utils::launch_1d_kernel(
            radar_detail::pulse_compression_norm_kernel<T>, 1,
            pulse_template.data(), static_cast<int>(pulse_template.size()),
            window_kind, window, norm.data());
    }
    cuda_utils::launch_1d_kernel(
        radar_detail::pulse_compression_prepare_template_kernel<T>,
        template_time.size(), pulse_template.data(),
        static_cast<int>(pulse_template.size()), nfft, options.normalize,
        window_kind, window, norm.data(), template_time.data());
    cuda_utils::launch_1d_kernel(
        radar_detail::pulse_compression_pack_input_kernel<T>, input_time.size(),
        input.data(), options.num_pulses, options.samples_per_pulse,
        nfft, input_time.data());
    FFTInterface fft(nfft, FFTInterface::BatchOnly{});
    fft.fft_batch_device(template_time, template_frequency, 1);
    fft.fft_batch_device(input_time, input_frequency, options.num_pulses);
    cuda_utils::launch_1d_kernel(
        radar_detail::pulse_compression_multiply_kernel,
        product_frequency.size(), input_frequency.data(),
        template_frequency.data(), static_cast<int>(product_frequency.size()),
        nfft, product_frequency.data());
    fft.ifft_batch_device(product_frequency, output, options.num_pulses);
}

template <class T>
void pulse_doppler_fp32_compute_device(
    const DeviceArray<TypedComplex<T>>& input,
    const PulseDopplerOptions& options,
    const DeviceArray<float>* explicit_window,
    DeviceArray<ComplexFloat>& output)
{
    const int nfft = options.nfft == -1 ? options.num_pulses : options.nfft;
    const float* window = explicit_window == nullptr ? nullptr : explicit_window->data();
    DeviceArray<ComplexFloat> fft_input(output.size()), fft_output(output.size());
    cuda_utils::launch_1d_kernel(
        radar_detail::pulse_doppler_prepare_input_kernel<T>, fft_input.size(),
        input.data(), options.num_pulses, options.samples_per_pulse,
        nfft, static_cast<int>(options.window), window, fft_input.data());
    FFTInterface fft(nfft, FFTInterface::BatchOnly{});
    fft.fft_batch_device(fft_input, fft_output, options.samples_per_pulse);
    cuda_utils::launch_1d_kernel(
        radar_detail::pulse_doppler_transpose_output_kernel, output.size(),
        fft_output.data(), options.samples_per_pulse, nfft, output.data());
}

}  // namespace detail

#define INST(T) template void detail::ambgfun_fp32_compute_device(const DeviceArray<TypedComplex<T>>&,const DeviceArray<TypedComplex<T>>&,int,int,int,float,float,DeviceArray<float>&);template void detail::ca_cfar_fp32_compute_device(const DeviceArray<T>&,int,int,int,int,int,int,int,float,DeviceArray<float>&,DeviceArray<CfarDetection>&);template void detail::cfar_alpha_fp32_compute_device(T,int,DeviceArray<float>&);template void detail::pulse_compression_fp32_compute_device(const DeviceArray<TypedComplex<T>>&,const DeviceArray<TypedComplex<T>>&,const PulseCompressionOptions&,const DeviceArray<float>*,DeviceArray<ComplexFloat>&);template void detail::pulse_doppler_fp32_compute_device(const DeviceArray<TypedComplex<T>>&,const PulseDopplerOptions&,const DeviceArray<float>*,DeviceArray<ComplexFloat>&)
INST(float);INST(__half);INST(std::int32_t);INST(std::int16_t);INST(std::int8_t);
#undef INST
} // namespace cusignal
