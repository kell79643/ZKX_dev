#include <cusignal/operators/filter_design/filter_design_typed.h>
#include "filter_design_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>

#include <cmath>
#include <stdexcept>

namespace cusignal {
void firwin_fp32_compute_device(
    int n,
    const DeviceArray<float>& bands,
    const DeviceArray<float>& window,
    DeviceArray<float>& normalization,
    DeviceArray<float>& out,
    bool scale)
{
    if (n <= 0 || bands.empty() || bands.size() % 2 != 0 ||
        window.size() != static_cast<std::size_t>(n) ||
        normalization.size() != 1 ||
        out.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument("firwin compute shape");
    if (scale) {
        cuda_utils::launch_1d_kernel(
            filter_design_detail::firwin_prepared_normalization_kernel,
            256U, n, bands.data(), static_cast<int>(bands.size() / 2),
            window.data(), normalization.data());
    }
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_prepared_kernel, out.size(), n,
        bands.data(), static_cast<int>(bands.size() / 2), window.data(),
        normalization.data(), out.data(), scale);
}

void firwin_fp64_storage_compute_device(
    int n,
    const DeviceArray<float>& bands,
    const DeviceArray<float>& window,
    DeviceArray<float>& normalization,
    void* out_storage,
    std::size_t out_size,
    bool scale)
{
    if (n <= 0 || bands.empty() || bands.size() % 2 != 0 ||
        window.size() != static_cast<std::size_t>(n) ||
        normalization.size() != 1 ||
        out_size != static_cast<std::size_t>(n) || out_storage == nullptr)
        throw std::invalid_argument("firwin storage compute shape");
    if (scale) {
        cuda_utils::launch_1d_kernel(
            filter_design_detail::firwin_prepared_normalization_kernel,
            256U, n, bands.data(), static_cast<int>(bands.size() / 2),
            window.data(), normalization.data());
    }
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_prepared_fp64_storage_kernel,
        out_size, n, bands.data(), static_cast<int>(bands.size() / 2),
        window.data(), normalization.data(),
        static_cast<std::uint64_t*>(out_storage), scale);
}

void firwin_fused_fp64_storage_compute_device(
    int n,
    const DeviceArray<float>& bands,
    const DeviceArray<float>& window,
    void* out_storage,
    std::size_t out_size,
    bool scale)
{
    if (n <= 0 || n > filter_design_detail::firwin_fused_max_taps ||
        bands.empty() || bands.size() % 2 != 0 ||
        window.size() != static_cast<std::size_t>(n) ||
        out_size != static_cast<std::size_t>(n) || out_storage == nullptr)
        throw std::invalid_argument("firwin fused storage compute shape");
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_prepared_fused_fp64_storage_kernel,
        256U, n, bands.data(), static_cast<int>(bands.size() / 2),
        window.data(), static_cast<std::uint64_t*>(out_storage), scale);
}

template <typename T>
void firwin_lowpass_explicit_window_resident_device(
    int numtaps, const DeviceArray<T>& cutoff,
    const DeviceArray<float>& window, DeviceArray<float>& output,
    float fs, bool scale)
{
    static_assert(detail::is_simple_signal_input_v<T>,
        "unsupported firwin resident cutoff dtype");
    if (numtaps <= 0 ||
        numtaps > filter_design_detail::firwin_fused_max_taps ||
        cutoff.size() != 1 ||
        window.size() != static_cast<std::size_t>(numtaps) ||
        output.size() != static_cast<std::size_t>(numtaps) ||
        !std::isfinite(fs) || fs <= 0.0F) {
        throw std::invalid_argument("firwin lowpass resident shape or fs");
    }
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_lowpass_explicit_window_fused_fp32_kernel<T>,
        256, numtaps, cutoff.data(), window.data(), fs, output.data(), scale);
}
void firwin2_fp32_compute_device(
    int n,
    const DeviceArray<float>& freq,
    const DeviceArray<float>& gain,
    const DeviceArray<float>& window,
    DeviceArray<float>& out,
    int nfreqs,
    bool antisymmetric,
    float nyquist)
{
    if (n <= 0 || freq.size() < 2 || freq.size() != gain.size() ||
        window.size() != static_cast<std::size_t>(n) ||
        out.size() != static_cast<std::size_t>(n) || nfreqs <= n ||
        nyquist <= 0.0F) {
        throw std::invalid_argument("firwin2 compute shape");
    }
    const std::size_t spectrum_size =
        static_cast<std::size_t>(2 * (nfreqs - 1));
    DeviceArray<ComplexFloat> spectrum(spectrum_size), time(spectrum_size);
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin2_build_spectrum_kernel<float,float>,
        spectrum_size, freq.data(), gain.data(), static_cast<int>(freq.size()),
        nfreqs, n, antisymmetric, nyquist, spectrum.data());
    // ZQ500 dlfft 在反复创建/销毁单次计划时会保留少量宿主侧运行时元数据。
    // 复用 FFTInterface 的有界 batch-plan 池（batch=1），避免长生命周期进程中
    // 每次 firwin2 调用都让 CPU 活跃堆继续增长；池中计划在线程退出时统一销毁。
    FFTInterface fft(static_cast<int>(spectrum_size), FFTInterface::BatchOnly{});
    fft.ifft_batch_device(spectrum, time, 1);
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin2_apply_window_kernel<float,float>,
        out.size(), time.data(), window.data(), out.data(), n,
        antisymmetric && (n % 2 != 0));
}

#define INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(T) \
    template void firwin_lowpass_explicit_window_resident_device<T>( \
        int, const DeviceArray<T>&, const DeviceArray<float>&, \
        DeviceArray<float>&, float, bool)

INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(float);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(__half);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int32_t);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int16_t);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int8_t);

#undef INSTANTIATE_FIRWIN_LOWPASS_RESIDENT

}  // namespace cusignal
