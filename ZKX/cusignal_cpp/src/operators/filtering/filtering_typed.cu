#include <cusignal/operators/filtering/filtering_typed.h>
#include "filtering_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cusignal {
namespace {
constexpr float pi = 3.14159265358979323846F;

}

namespace detail {
void channelize_poly_fft_device(
    const DeviceArray<ComplexFloat>& polyphase,
    int channel_count,
    int point_count,
    DeviceArray<ComplexFloat>& output)
{
    const std::size_t expected = static_cast<std::size_t>(channel_count)
        * static_cast<std::size_t>(point_count);
    if (channel_count <= 0 || point_count < 0 ||
        polyphase.size() != expected || output.size() != expected)
        throw std::invalid_argument("channelize_poly FFT shape");
    if (expected == 0) return;
    DeviceArray<ComplexFloat> transformed(expected);
    FFTInterface fft(channel_count, FFTInterface::BatchOnly{});
    fft.fft_batch_device(polyphase, transformed, point_count);
    cuda_utils::launch_1d_kernel(
        channelize_poly_finalize_float_kernel, expected,
        transformed.data(), channel_count, point_count, output.data());
}
}

template <class T>
void channelize_poly_device(
    const DeviceArray<T>& x, const DeviceArray<T>& h, int n_chans,
    ChannelizePolyDeviceResult& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_channelize_poly(
        x.size(), h.size(), n_chans);
    output = ChannelizePolyDeviceResult{};
    output.shape = {plan.channel_count, plan.point_count};
    output.values = DeviceArray<ComplexFloat>(plan.output_size);
    if (plan.output_size == 0) return;
    DeviceArray<ComplexFloat> polyphase(plan.output_size);
    cuda_utils::launch_1d_kernel(
        detail::channelize_poly_accumulate_kernel<T>, polyphase.size(),
        x.data(), static_cast<int>(x.size()), h.data(), plan.tap_count,
        plan.channel_count, plan.point_count, polyphase.data());
    detail::channelize_poly_fft_device(
        polyphase, plan.channel_count, plan.point_count, output.values);
}

template <class T>
void channelize_poly_device(
    const DeviceArray<T>& x, const DeviceArray<T>& h, int n_chans,
    const std::vector<int>& input_shape,
    const std::vector<int>& filter_shape,
    ChannelizePolyDeviceResult& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_channelize_poly(
        x.size(), h.size(), n_chans, input_shape, filter_shape);
    output = ChannelizePolyDeviceResult{};
    output.shape = {plan.channel_count, plan.point_count};
    output.values = DeviceArray<ComplexFloat>(plan.output_size);
    if (plan.output_size == 0) return;
    DeviceArray<ComplexFloat> polyphase(plan.output_size);
    cuda_utils::launch_1d_kernel(
        detail::channelize_poly_accumulate_kernel<T>, polyphase.size(),
        x.data(), plan.input_count, h.data(), plan.tap_count,
        plan.channel_count, plan.point_count, polyphase.data());
    detail::channelize_poly_fft_device(
        polyphase, plan.channel_count, plan.point_count, output.values);
}
template <class T>
void detrend_fp32_compute_device(
    const DeviceArray<T>& x,
    int sample_count,
    int inner_count,
    const std::vector<int>& breakpoints,
    DetrendMode mode,
    DeviceArray<float>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (output.size() != x.size() || sample_count < 0 || inner_count < 0)
        throw std::invalid_argument("detrend compute shape");
    if (x.empty()) return;
    if (inner_count == 0)
        throw std::invalid_argument("detrend compute inner stride");
    int previous = 0;
    for (const int breakpoint : breakpoints) {
        if (breakpoint <= previous || breakpoint >= sample_count)
            throw std::invalid_argument("detrend compute breakpoint");
        previous = breakpoint;
    }
    if (sample_count <= 0 || x.size() % static_cast<std::size_t>(sample_count) != 0)
        throw std::invalid_argument("detrend compute axis");
    DeviceArray<float> input(x.size());
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, x.size(),
        x.data(), input.data(), x.size());
    auto device_breakpoints = DeviceArray<int>::from_host(breakpoints);
    const int batch_count = static_cast<int>(x.size() / sample_count);
    cuda_utils::launch_1d_kernel(
        detail::detrend_batch_kernel, static_cast<std::size_t>(batch_count),
        input.data(), sample_count, device_breakpoints.data(),
        static_cast<int>(device_breakpoints.size()),
        mode == DetrendMode::linear ? 1 : 0, output.data(), batch_count,
        inner_count);
}
template <class T>
void firfilter_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    FirfilterDeviceResult& output, const FirfilterOptions& options,
    const DeviceArray<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_firfilter_layout(
        x.size(), b.size(), options, zi != nullptr, zi == nullptr ? 0 : zi->size());
    output = FirfilterDeviceResult{};
    output.has_zf = zi != nullptr;
    output.y_shape = plan.input_shape;
    output.y = DeviceArray<float>(x.size());
    if (output.has_zf) {
        output.zf_shape = plan.state_shape;
        output.zf = DeviceArray<float>(
            static_cast<std::size_t>(plan.outer_count)
            * plan.state_count * plan.inner_count);
    }

    DeviceArray<float> coefficients(b.size());
    DeviceArray<float> input(x.size());
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, b.size(),
        b.data(), coefficients.data(), b.size());
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, x.size(),
        x.data(), input.data(), x.size());

    DeviceArray<float> initial_state;
    if (zi != nullptr) {
        initial_state.reset(zi->size());
        if (!zi->empty())
            cuda_utils::launch_1d_kernel(
                detail::convert_signal_kernel<T,float>, zi->size(),
                zi->data(), initial_state.data(), zi->size());
    }
    auto line_offsets = DeviceArray<int>::from_host(plan.zi_line_offsets);
    cuda_utils::launch_1d_kernel(
        detail::firfilter_axis_output_kernel, output.y.size(),
        coefficients.data(), static_cast<int>(coefficients.size()), input.data(),
        static_cast<int>(input.size()), plan.sample_count, plan.inner_count,
        initial_state.data(), line_offsets.data(), plan.zi_state_stride,
        zi != nullptr, output.y.data());
    if (!output.zf.empty())
        cuda_utils::launch_1d_kernel(
            detail::firfilter_axis_state_kernel, output.zf.size(),
            coefficients.data(), static_cast<int>(coefficients.size()), input.data(),
            plan.sample_count, plan.inner_count, initial_state.data(),
            line_offsets.data(), plan.zi_state_stride, output.zf.data(),
            static_cast<int>(output.zf.size()));
}

void firfilter_fp32_resident_device(
    const DeviceArray<float>& coefficients,
    const DeviceArray<float>& input,
    DeviceArray<float>& output)
{
    if (coefficients.empty() || output.size() != input.size())
        throw std::invalid_argument("firfilter FP32 resident shape");
    cuda_utils::launch_1d_kernel(
        detail::firfilter_axis_output_kernel, output.size(),
        coefficients.data(), static_cast<int>(coefficients.size()),
        input.data(), static_cast<int>(input.size()),
        static_cast<int>(input.size()), 1,
        static_cast<const float*>(nullptr), static_cast<const int*>(nullptr),
        0, false, output.data());
}
template <class T>
void firfilter2_fp32_compute_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    DeviceArray<float>& computed, const Firfilter2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (computed.size() != x.size())
        throw std::invalid_argument("firfilter2 compute output shape");
    const auto plan = detail::prepare_firfilter2_layout(
        x.size(), b.size(), options);
    constexpr std::size_t fused_coefficient_limit = 16;
    if (b.size() <= fused_coefficient_limit) {
        int pad_mode = 2;
        if (options.padtype == Firfilter2PadType::odd) pad_mode = 0;
        if (options.padtype == Firfilter2PadType::even) pad_mode = 1;
        cuda_utils::launch_1d_kernel(
            detail::firfilter2_small_fused_axis_kernel<T>, computed.size(),
            b.data(), static_cast<int>(b.size()), x.data(), plan.sample_count,
            plan.inner_count, plan.outer_count, plan.edge, pad_mode,
            computed.data());
        return;
    }
    const std::size_t extended_size = static_cast<std::size_t>(plan.outer_count)
        * plan.extended_sample_count * plan.inner_count;
    const std::size_t state_size = static_cast<std::size_t>(plan.outer_count)
        * plan.state_count * plan.inner_count;

    FirfilterOptions internal_options;
    internal_options.shape = plan.extended_shape;
    internal_options.axis = plan.axis;
    internal_options.zi_shape = plan.extended_shape;
    internal_options.zi_shape[plan.axis] = plan.state_count;
    const auto internal_plan = detail::prepare_firfilter_layout(
        extended_size, b.size(), internal_options, true, state_size);

    DeviceArray<float> coefficients(b.size());
    DeviceArray<float> padded(extended_size);
    DeviceArray<float> base_state(plan.state_count);
    DeviceArray<float> forward_state(state_size);
    DeviceArray<float> reverse_state(state_size);
    DeviceArray<float> forward(extended_size);
    DeviceArray<float> forward_reversed(extended_size);
    DeviceArray<float> reverse_filtered(extended_size);
    DeviceArray<float> full(extended_size);
    auto line_offsets = DeviceArray<int>::from_host(
        internal_plan.zi_line_offsets);

    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, b.size(),
        b.data(), coefficients.data(), b.size());
    int pad_mode = 2;
    if (options.padtype == Firfilter2PadType::odd) pad_mode = 0;
    if (options.padtype == Firfilter2PadType::even) pad_mode = 1;
    cuda_utils::launch_1d_kernel(
        detail::firfilter2_pad_axis_kernel<T>, padded.size(),
        x.data(), plan.sample_count, plan.inner_count, plan.outer_count,
        plan.edge, pad_mode, padded.data(), plan.extended_sample_count);

    if (plan.state_count > 0) {
        cuda_utils::launch_1d_kernel(
            detail::firfilter2_base_state_kernel, base_state.size(),
            coefficients.data(), static_cast<int>(coefficients.size()),
            base_state.data());
        cuda_utils::launch_1d_kernel(
            detail::firfilter2_scale_state_axis_kernel, forward_state.size(),
            base_state.data(), plan.state_count, padded.data(),
            plan.extended_sample_count, plan.inner_count, plan.outer_count,
            false, forward_state.data());
    }
    cuda_utils::launch_1d_kernel(
        detail::firfilter_axis_output_kernel, forward.size(),
        coefficients.data(), static_cast<int>(coefficients.size()), padded.data(),
        static_cast<int>(padded.size()), plan.extended_sample_count,
        plan.inner_count, forward_state.data(), line_offsets.data(),
        internal_plan.zi_state_stride, true, forward.data());
    cuda_utils::launch_1d_kernel(
        detail::firfilter2_reverse_axis_kernel, forward_reversed.size(),
        forward.data(), plan.extended_sample_count, plan.inner_count,
        plan.outer_count, forward_reversed.data());
    if (plan.state_count > 0)
        cuda_utils::launch_1d_kernel(
            detail::firfilter2_scale_state_axis_kernel, reverse_state.size(),
            base_state.data(), plan.state_count, forward.data(),
            plan.extended_sample_count, plan.inner_count, plan.outer_count,
            true, reverse_state.data());
    cuda_utils::launch_1d_kernel(
        detail::firfilter_axis_output_kernel, reverse_filtered.size(),
        coefficients.data(), static_cast<int>(coefficients.size()),
        forward_reversed.data(), static_cast<int>(forward_reversed.size()),
        plan.extended_sample_count, plan.inner_count, reverse_state.data(),
        line_offsets.data(), internal_plan.zi_state_stride, true,
        reverse_filtered.data());
    cuda_utils::launch_1d_kernel(
        detail::firfilter2_reverse_axis_kernel, full.size(),
        reverse_filtered.data(), plan.extended_sample_count, plan.inner_count,
        plan.outer_count, full.data());
    cuda_utils::launch_1d_kernel(
        detail::firfilter2_crop_axis_kernel, computed.size(),
        full.data(), plan.extended_sample_count, plan.sample_count,
        plan.inner_count, plan.outer_count, plan.edge, computed.data());

}
template <class T>
void firfilter2_fp64_storage_compute_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    std::uint64_t* output, std::size_t output_size,
    const Firfilter2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (output_size != x.size())
        throw std::invalid_argument("firfilter2 FP64 storage output shape");
    const auto plan = detail::prepare_firfilter2_layout(
        x.size(), b.size(), options);
    constexpr std::size_t fused_coefficient_limit = 16;
    if (b.size() > fused_coefficient_limit)
        throw std::invalid_argument(
            "firfilter2 FP64 storage fast path coefficient count");
    int pad_mode = 2;
    if (options.padtype == Firfilter2PadType::odd) pad_mode = 0;
    if (options.padtype == Firfilter2PadType::even) pad_mode = 1;
    cuda_utils::launch_1d_kernel(
        detail::firfilter2_small_fused_axis_fp64_storage_kernel<T>,
        output_size, b.data(), static_cast<int>(b.size()), x.data(),
        plan.sample_count, plan.inner_count, plan.outer_count, plan.edge,
        pad_mode, output);
}
template <class Input>
void freq_shift_fp32_compute_device(
    const DeviceArray<Input>& input,
    DeviceArray<ComplexFloat>& output,
    float frequency,
    float sample_rate)
{
    static_assert(detail::is_freq_shift_input_v<Input>);
    if (output.size() != input.size())
        throw std::invalid_argument("freq_shift compute shape");
    if (input.empty()) return;
    cuda_utils::launch_1d_kernel(
        detail::freq_shift_kernel<Input,ComplexFloat,float>, output.size(),
        input.data(), output.data(), input.size(), frequency, sample_rate);
}
template <class T>
void hilbert_fp32_compute_device(
    const DeviceArray<T>& x, DeviceArray<ComplexFloat>& computed,
    const HilbertOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert_layout(x.size(), options);
    if (computed.size() != plan.output_size)
        throw std::invalid_argument("hilbert compute output shape");
    if (plan.output_size == 0) return;
    DeviceArray<ComplexFloat> packed(plan.output_size);
    DeviceArray<ComplexFloat> spectrum(plan.output_size);
    cuda_utils::launch_1d_kernel(
        detail::hilbert_pack_axis_kernel<T>, packed.size(), x.data(),
        plan.input_axis_length, plan.fft_length, plan.inner_count,
        plan.line_count, packed.data());
    FFTInterface fft(plan.fft_length, FFTInterface::BatchOnly{});
    fft.fft_batch_device(packed, spectrum, plan.line_count);
    cuda_utils::launch_1d_kernel(
        detail::hilbert_mask_kernel, spectrum.size(), spectrum.data(),
        plan.fft_length, static_cast<int>(spectrum.size()));
    fft.ifft_batch_device(spectrum, packed, plan.line_count);
    cuda_utils::launch_1d_kernel(
        detail::hilbert_unpack_axis_kernel, computed.size(), packed.data(),
        plan.fft_length, plan.inner_count, plan.line_count, computed.data());
}
template <class T>
void hilbert2_fp32_compute_device(
    const DeviceArray<T>& x, DeviceArray<ComplexFloat>& computed,
    const Hilbert2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert2_layout(x.size(), options);
    if (computed.size() != plan.output_size)
        throw std::invalid_argument("hilbert2 compute output shape");
    DeviceArray<ComplexFloat> time(plan.fft_size);
    DeviceArray<ComplexFloat> first_spectrum(plan.fft_size);
    cuda_utils::launch_1d_kernel(
        detail::real_matrix_to_complex_float_kernel<T>, time.size(),
        x.data(), plan.input_rows, plan.input_cols, time.data(),
        plan.fft_rows, plan.fft_cols);
    FFTInterface2D first_fft(
        plan.fft_rows, plan.fft_cols, FFTInterface::BatchOnly{});
    first_fft.fft2d_device(time, first_spectrum);
    if (plan.fft_rows == plan.fft_cols) {
        cuda_utils::launch_1d_kernel(
            detail::hilbert2_mask_kernel, first_spectrum.size(),
            first_spectrum.data(), plan.fft_cols);
        first_fft.ifft2d_device(first_spectrum, computed);
        return;
    }
    DeviceArray<ComplexFloat> square_spectrum(plan.output_size);
    cuda_utils::launch_1d_kernel(
        detail::hilbert2_broadcast_row_mask_kernel, square_spectrum.size(),
        first_spectrum.data(), plan.fft_cols, square_spectrum.data());
    FFTInterface2D square_fft(
        plan.output_rows, plan.output_cols, FFTInterface::BatchOnly{});
    square_fft.ifft2d_device(square_spectrum, computed);
}
template <class T>
void lfilter_zi_fp32_compute_device(
    const DeviceArray<T>& b, const DeviceArray<T>& a,
    const detail::LfilterZiLayoutPlan& plan,
    DeviceArray<float>& computed, DeviceArray<int>& status)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (computed.size() != static_cast<std::size_t>(plan.output_count) ||
        status.size() != 1)
        throw std::invalid_argument("lfilter_zi compute output shape");
    DeviceArray<float> numerator(b.size());
    DeviceArray<float> denominator(plan.denominator_count);
    if (!b.empty()) {
        cuda_utils::launch_1d_kernel(
            detail::convert_signal_kernel<T,float>, b.size(), b.data(),
            numerator.data(), b.size());
    }
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, denominator.size(),
        a.data() + plan.denominator_offset, denominator.data(),
        denominator.size());
    DeviceArray<float> normalized_numerator(plan.common_count);
    DeviceArray<float> normalized_denominator(plan.common_count);
    DeviceArray<float> augmented(plan.augmented_count);
    DeviceArray<float> solution(plan.output_count);
    detail::lfilter_zi_fp32_kernel<<<1,1>>>(
        numerator.data(), static_cast<int>(numerator.size()),
        denominator.data(), plan.denominator_count,
        normalized_numerator.data(), normalized_denominator.data(),
        augmented.data(), solution.data(), computed.data(), status.data());
    CUDA_KERNEL_CHECK();
}
template <class T>
void sosfilt_device(
    const DeviceArray<T>& sos, int sections, const DeviceArray<T>& x,
    SosfiltDeviceResult& output, const SosfiltOptions& options,
    const DeviceArray<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const bool has_zi = zi != nullptr;
    const auto plan = detail::prepare_sosfilt_layout(
        x.size(), sos.size(), sections, options, has_zi,
        has_zi ? zi->size() : 0U);

    const auto host_sos = sos.to_host();
    std::vector<float> coefficients(host_sos.size());
    for (std::size_t index = 0; index < host_sos.size(); ++index)
        coefficients[index] = detail::SimpleSignalTypePolicy<T>::load(host_sos[index]);
    detail::validate_sos_coefficients(coefficients, sections);

    output = SosfiltDeviceResult{};
    output.y = DeviceArray<float>(x.size());
    output.y_shape = plan.input_shape;
    output.has_zf = has_zi;

    DeviceArray<float> zero_state;
    float* state = nullptr;
    if (has_zi) {
        output.zf = DeviceArray<float>(plan.state_size);
        output.zf_shape = plan.state_shape;
        if (plan.state_size != 0) {
            cuda_utils::launch_1d_kernel(
                detail::convert_signal_kernel<T,float>, plan.state_size,
                zi->data(), output.zf.data(), plan.state_size);
        }
        state = output.zf.data();
    } else {
        zero_state = DeviceArray<float>(plan.state_size);
        cuda_utils::zero_device(zero_state.data(), zero_state.size());
        state = zero_state.data();
    }
    if (plan.line_count == 0) return;

    cuda_utils::launch_1d_kernel(
        detail::sosfilt_axis_kernel<T>,
        static_cast<std::size_t>(plan.line_count),
        sos.data(), sections, x.data(), plan.sample_count,
        plan.inner_count, plan.outer_count, state, output.y.data());
    cuda_utils::synchronize_stream();
}
template <class T>
void wiener_fp32_compute_device(
    const DeviceArray<T>& x, const detail::WienerLayoutPlan& plan,
    bool has_explicit_noise, float explicit_noise,
    DeviceArray<float>& computed)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (computed.size() != x.size())
        throw std::invalid_argument("wiener compute output shape");
    if (x.empty()) return;

    auto device_shape = DeviceArray<int>::from_host(plan.input_shape);
    auto device_strides = DeviceArray<int>::from_host(plan.input_strides);
    auto device_window = DeviceArray<int>::from_host(plan.window_shape);
    DeviceArray<float> local_mean(x.size());
    DeviceArray<float> local_variance(x.size());
    DeviceArray<float> noise(1);
    cuda_utils::launch_1d_kernel(
        detail::wiener_nd_stats_kernel<T>, x.size(), x.data(),
        plan.element_count, device_shape.data(), device_strides.data(),
        device_window.data(), plan.rank, plan.window_volume,
        local_mean.data(), local_variance.data());
    if (has_explicit_noise) {
        cuda_utils::copy_value_to_device(noise.data(), explicit_noise);
    } else {
        detail::wiener_mean_noise_kernel<<<1, 1>>>(
            local_variance.data(), plan.element_count, noise.data());
        CUDA_KERNEL_CHECK();
    }
    cuda_utils::launch_1d_kernel(
        detail::wiener_result_kernel<T>, x.size(), x.data(),
        local_mean.data(), local_variance.data(), plan.element_count,
        noise.data(), computed.data());
}
template <class T>
void decimate_coefficients_fp32_device(
    const DeviceArray<T>& coefficients, DeviceArray<float>& fp32)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (coefficients.empty() || fp32.size() != coefficients.size())
        throw std::invalid_argument("decimate coefficient conversion shape");
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, coefficients.size(),
        coefficients.data(), fp32.data(), coefficients.size());
}

template <class T>
void decimate_fp32_compute_device(
    const DeviceArray<T>& x, const DeviceArray<float>& taps,
    const detail::DecimateLayoutPlan& plan, int q, bool zero_phase,
    DeviceArray<float>& computed)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (q < 1 || taps.empty())
        throw std::invalid_argument("decimate compute parameters");
    if (computed.size() != plan.output_size)
        throw std::invalid_argument("decimate compute output shape");
    if (computed.empty()) return;
    if (plan.inner_count <= 0 || plan.line_count <= 0)
        throw std::invalid_argument("decimate compute line shape");
    const int tap_count = static_cast<int>(taps.size());
    const int half_length = (tap_count - 1) / 2;
    const int prefix = zero_phase ? q - half_length % q : 0;
    if (tap_count > std::numeric_limits<int>::max() - prefix)
        throw std::invalid_argument("decimate padded coefficient count");
    const int output_start = zero_phase
        ? (half_length + prefix) / q
        : 0;
    cuda_utils::launch_1d_kernel(
        detail::decimate_axis_kernel<T>, computed.size(),
        taps.data(), tap_count, x.data(), plan.sample_count, q,
        prefix, output_start, computed.data(), plan.output_sample_count,
        plan.inner_count, plan.line_count);
}
template <class T>
void resample_fp32_compute_device(
    const DeviceArray<T>& x, const detail::ResampleLayoutPlan& plan,
    ResampleDomain domain, const DeviceArray<float>* window,
    DeviceArray<float>& computed)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (computed.size() != plan.output_size)
        throw std::invalid_argument("resample compute output shape");
    if (window != nullptr &&
        window->size() != static_cast<std::size_t>(plan.input_sample_count))
        throw std::invalid_argument("resample compute window shape");
    if (computed.empty()) return;
    if (plan.line_count <= 0 || plan.inner_count <= 0)
        throw std::invalid_argument("resample compute line shape");

    DeviceArray<ComplexFloat> packed(plan.input_workspace_size);
    cuda_utils::launch_1d_kernel(
        detail::resample_pack_axis_kernel<T>, packed.size(),
        x.data(), plan.input_sample_count, plan.inner_count,
        plan.line_count, domain == ResampleDomain::frequency,
        window == nullptr ? nullptr : window->data(), window != nullptr,
        packed.data());

    DeviceArray<ComplexFloat> spectrum;
    if (domain == ResampleDomain::time) {
        spectrum = DeviceArray<ComplexFloat>(plan.input_workspace_size);
        FFTInterface forward(
            plan.input_sample_count, FFTInterface::BatchOnly{});
        forward.fft_batch_device(packed, spectrum, plan.line_count);
        if (window != nullptr)
            cuda_utils::launch_1d_kernel(
                detail::resample_apply_window_kernel, spectrum.size(),
                spectrum.data(), plan.input_sample_count, plan.line_count,
                window->data());
    } else {
        spectrum = std::move(packed);
    }

    DeviceArray<ComplexFloat> resized(plan.output_size);
    cuda_utils::launch_1d_kernel(
        detail::resample_copy_spectrum_kernel<T>, resized.size(),
        spectrum.data(), plan.input_sample_count, resized.data(),
        plan.output_sample_count, plan.line_count,
        domain == ResampleDomain::frequency);
    DeviceArray<ComplexFloat> transformed(plan.output_size);
    FFTInterface inverse(
        plan.output_sample_count, FFTInterface::BatchOnly{});
    inverse.ifft_batch_device(resized, transformed, plan.line_count);
    cuda_utils::launch_1d_kernel(
        detail::resample_store_axis_kernel, computed.size(),
        transformed.data(), plan.output_sample_count, plan.inner_count,
        plan.line_count,
        static_cast<float>(plan.output_sample_count) /
            static_cast<float>(plan.input_sample_count),
        computed.data());
}
template <typename T>
void upfirdn_device(
    const DeviceArray<T>& h, const DeviceArray<T>& x,
    int up, int down, UpfirdnDeviceResult& output,
    const UpfirdnOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_upfirdn_layout(
        x.size(), h.size(), up, down, options);
    output = UpfirdnDeviceResult{};
    output.dtype = std::is_same_v<T, std::int32_t>
        ? UpfirdnOutputDtype::fp64
        : UpfirdnOutputDtype::fp32;
    output.shape = plan.output_shape;

    DeviceArray<float> input(x.size());
    DeviceArray<float> taps(h.size());
    if (!x.empty())
        cuda_utils::launch_1d_kernel(
            detail::convert_signal_kernel<T, float>, x.size(),
            x.data(), input.data(), x.size());
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T, float>, h.size(),
        h.data(), taps.data(), h.size());
    DeviceArray<float> computed(plan.output_size);
    if (!computed.empty()) {
        const bool contiguous_axis = plan.inner_count == 1;
        const int input_batch_stride = contiguous_axis
            ? plan.input_sample_count : 1;
        const int input_sample_stride = contiguous_axis
            ? 1 : plan.inner_count;
        const int output_batch_stride = contiguous_axis
            ? plan.output_sample_count : 1;
        const int output_sample_stride = contiguous_axis
            ? 1 : plan.inner_count;
        cuda_utils::launch_1d_kernel(
            detail::upfirdn_kernel<float>, computed.size(),
            taps.data(), plan.coefficient_count,
            input.data(), plan.input_sample_count,
            plan.up, plan.down, computed.data(), plan.output_sample_count,
            plan.line_count, input_batch_stride, input_sample_stride,
            output_batch_stride, output_sample_stride);
    }
    if (output.dtype == UpfirdnOutputDtype::fp32) {
        cuda_utils::synchronize_stream();
        output.fp32 = std::move(computed);
    } else {
        output.fp64 = detail::finalize_upfirdn_fp64_output(computed);
    }
}
template <typename T>
void resample_poly_custom_coefficients_fp32_device(
    const DeviceArray<T>& coefficients, int up, DeviceArray<float>& scaled)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (up < 1 || scaled.size() != coefficients.size())
        throw std::invalid_argument("resample_poly coefficient shape");
    if (scaled.empty()) return;
    cuda_utils::launch_1d_kernel(
        detail::resample_poly_custom_coefficients_kernel<T>, scaled.size(),
        coefficients.data(), static_cast<int>(coefficients.size()), up,
        scaled.data());
}

template <typename T>
void resample_poly_fp32_compute_device(
    const DeviceArray<T>& x, const DeviceArray<float>& coefficients,
    const detail::ResamplePolyLayoutPlan& plan, DeviceArray<float>& computed)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (x.size() != static_cast<std::size_t>(plan.input_sample_count) *
            plan.line_count ||
        coefficients.size() !=
            static_cast<std::size_t>(plan.coefficient_count) ||
        computed.size() != plan.output_size || plan.up < 1 || plan.down < 1)
        throw std::invalid_argument("resample_poly compute shape");
    if (computed.empty()) return;
    if (plan.identity || plan.inner_count <= 0 || plan.line_count <= 0)
        throw std::invalid_argument("resample_poly compute layout");
    cuda_utils::launch_1d_kernel(
        detail::resample_poly_axis_kernel<T>, computed.size(),
        x.data(), plan.input_sample_count, plan.inner_count, plan.line_count,
        coefficients.data(), plan.coefficient_count, plan.up, plan.down,
        plan.prefix_zeros, plan.output_start, plan.output_sample_count,
        computed.data());
}

#define INST(T) template void channelize_poly_device(const DeviceArray<T>&,const DeviceArray<T>&,int,ChannelizePolyDeviceResult&);template void channelize_poly_device(const DeviceArray<T>&,const DeviceArray<T>&,int,const std::vector<int>&,const std::vector<int>&,ChannelizePolyDeviceResult&);template void detrend_fp32_compute_device(const DeviceArray<T>&,int,int,const std::vector<int>&,DetrendMode,DeviceArray<float>&);template void firfilter_device(const DeviceArray<T>&,const DeviceArray<T>&,FirfilterDeviceResult&,const FirfilterOptions&,const DeviceArray<T>*);template void firfilter2_fp32_compute_device(const DeviceArray<T>&,const DeviceArray<T>&,DeviceArray<float>&,const Firfilter2Options&);template void hilbert_fp32_compute_device(const DeviceArray<T>&,DeviceArray<ComplexFloat>&,const HilbertOptions&);template void hilbert2_fp32_compute_device(const DeviceArray<T>&,DeviceArray<ComplexFloat>&,const Hilbert2Options&);template void lfilter_zi_fp32_compute_device(const DeviceArray<T>&,const DeviceArray<T>&,const detail::LfilterZiLayoutPlan&,DeviceArray<float>&,DeviceArray<int>&);template void sosfilt_device(const DeviceArray<T>&,int,const DeviceArray<T>&,SosfiltDeviceResult&,const SosfiltOptions&,const DeviceArray<T>*);template void wiener_fp32_compute_device(const DeviceArray<T>&,const detail::WienerLayoutPlan&,bool,float,DeviceArray<float>&);template void decimate_coefficients_fp32_device(const DeviceArray<T>&,DeviceArray<float>&);template void decimate_fp32_compute_device(const DeviceArray<T>&,const DeviceArray<float>&,const detail::DecimateLayoutPlan&,int,bool,DeviceArray<float>&);template void resample_fp32_compute_device(const DeviceArray<T>&,const detail::ResampleLayoutPlan&,ResampleDomain,const DeviceArray<float>*,DeviceArray<float>&);template void upfirdn_device(const DeviceArray<T>&,const DeviceArray<T>&,int,int,UpfirdnDeviceResult&,const UpfirdnOptions&);template void resample_poly_custom_coefficients_fp32_device(const DeviceArray<T>&,int,DeviceArray<float>&);template void resample_poly_fp32_compute_device(const DeviceArray<T>&,const DeviceArray<float>&,const detail::ResamplePolyLayoutPlan&,DeviceArray<float>&)
INST(float);INST(__half);INST(std::int32_t);INST(std::int16_t);INST(std::int8_t);
#undef INST
#define INST_FIRFILTER2_FP64_STORAGE(T) template void firfilter2_fp64_storage_compute_device(const DeviceArray<T>&,const DeviceArray<T>&,std::uint64_t*,std::size_t,const Firfilter2Options&)
INST_FIRFILTER2_FP64_STORAGE(float);INST_FIRFILTER2_FP64_STORAGE(__half);INST_FIRFILTER2_FP64_STORAGE(std::int32_t);INST_FIRFILTER2_FP64_STORAGE(std::int16_t);INST_FIRFILTER2_FP64_STORAGE(std::int8_t);
#undef INST_FIRFILTER2_FP64_STORAGE
#define INST_FREQ(T) template void freq_shift_fp32_compute_device(const DeviceArray<T>&,DeviceArray<ComplexFloat>&,float,float);template void freq_shift_fp32_compute_device(const DeviceArray<TypedComplex<T>>&,DeviceArray<ComplexFloat>&,float,float)
INST_FREQ(float);INST_FREQ(__half);INST_FREQ(std::int32_t);INST_FREQ(std::int16_t);INST_FREQ(std::int8_t);
#undef INST_FREQ
} // namespace cusignal
