#include <cusignal/operators/windows/windows_typed.h>
#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN
#include <cusignal/backends/fft/fft_interface.h>
#endif
#include "windows_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>

#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace cusignal {

template <class T>
void chebwin_fp32_compute_device(
    int n, T attenuation, DeviceArray<float>& output, bool symmetric)
{
    if (n < 1 || output.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "chebwin_device: n must be nonnegative and output size must equal n");
    }
    const int effective = symmetric ? n : n + 1;
#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN
#if defined(USE_DLFFT)
    // ZQ500官方dlfft只接受2的整数次幂。补零后截断并不等价于原长度DFT，
    // 因而非2次幂chebwin使用严格等价的两级混合基数DFT保持标准窗语义；
    // 不使用Bluestein，也不切换到fft_thrust。2次幂仍进入真实dlfft路径。
    if ((effective & (effective - 1)) != 0) {
        int radix1 = 1;
        for (int candidate = 2; candidate <= effective / candidate; ++candidate) {
            if (effective % candidate == 0) radix1 = candidate;
        }
        const int radix2 = effective / radix1;
        DeviceArray<ComplexFloat> stage(effective);
        DeviceArray<ComplexFloat> transformed(effective);
        DeviceArray<float> reordered(effective);
        cuda_utils::launch_1d_kernel(
            windows_detail::chebwin_mixed_radix_stage1_kernel<T>,
            stage.size(),
            effective,
            radix1,
            radix2,
            attenuation,
            stage.data());
        cuda_utils::launch_1d_kernel(
            windows_detail::chebwin_mixed_radix_stage2_kernel,
            transformed.size(),
            stage.data(),
            effective,
            radix1,
            radix2,
            transformed.data());
        cuda_utils::launch_1d_kernel(
            windows_detail::chebwin_reorder_kernel,
            reordered.size(),
            transformed.data(),
            effective,
            reordered.data());
        thrust::device_ptr<float> reordered_pointer(reordered.data());
        const float peak = *thrust::max_element(
            reordered_pointer,
            reordered_pointer + effective);
        cuda_utils::launch_1d_kernel(
            windows_detail::normalize_window_kernel<float, float>,
            output.size(),
            reordered.data(),
            n,
            peak,
            output.data());
        return;
    }
#endif
    DeviceArray<ComplexFloat> spectrum(effective);
    DeviceArray<ComplexFloat> transformed(effective);
    DeviceArray<float> reordered(effective);
    cuda_utils::launch_1d_kernel(
        windows_detail::chebwin_spectrum_kernel<T>,
        spectrum.size(),
        effective,
        attenuation,
        spectrum.data());
    FFTInterface fft(effective, FFTInterface::BatchOnly{});
    fft.fft_batch_device(spectrum, transformed, 1);
    cuda_utils::launch_1d_kernel(
        windows_detail::chebwin_reorder_kernel,
        reordered.size(),
        transformed.data(),
        effective,
        reordered.data());
    thrust::device_ptr<float> reordered_pointer(reordered.data());
    const float peak = *thrust::max_element(
        reordered_pointer,
        reordered_pointer + effective);
    cuda_utils::launch_1d_kernel(
        windows_detail::normalize_window_kernel<float, float>,
        output.size(),
        reordered.data(),
        n,
        peak,
        output.data());
#else
    DeviceArray<float> raw(effective);
    cuda_utils::launch_1d_kernel(
        windows_detail::chebwin_raw_kernel<T, float>,
        raw.size(),
        effective,
        attenuation,
        raw.data());
    thrust::device_ptr<float> raw_pointer(raw.data());
    const float peak = *thrust::max_element(
        raw_pointer,
        raw_pointer + effective);
    cuda_utils::launch_1d_kernel(
        windows_detail::normalize_window_kernel<float, float>,
        output.size(),
        raw.data(),
        n,
        peak,
        output.data());
#endif
}
template <class T> void general_cosine_fp32_compute_device(int n,const DeviceArray<T>&a,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<=1||o.size()!=size_t(n))throw std::invalid_argument("general_cosine_device: compute output size must equal n and n must exceed one");cuda_utils::launch_1d_kernel(windows_detail::general_cosine_kernel<T,float>,o.size(),n,a.data(),(int)a.size(),o.data(),sym?n:n+1);}
template <class T> void general_cosine_fp32_compute_device(int n,const DeviceArray<T>&a,int coefficient_count,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<=1||coefficient_count<0||static_cast<std::size_t>(coefficient_count)>a.size()||o.size()!=size_t(n))throw std::invalid_argument("general_cosine_device: coefficient shape or output size mismatch");cuda_utils::launch_1d_kernel(windows_detail::general_cosine_kernel<T,float>,o.size(),n,a.data(),coefficient_count,o.data(),sym?n:n+1);}
template <class T> void general_gaussian_fp32_compute_device(int n,T p,T w,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<=1||o.size()!=size_t(n))throw std::invalid_argument("general_gaussian_device: compute output size must equal n and n must exceed one");cuda_utils::launch_1d_kernel(windows_detail::general_gaussian_kernel<float,T>,o.size(),n,p,w,o.data(),sym?n:n+1);}
template <class T> void hamming_fp32_compute_device(int n,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("hamming_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::hamming_kernel<float>,o.size(),n,o.data(),(!sym&&n%2==0)?n+1:n);}
template <class T> void kaiser_fp32_compute_device(int n,T b,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("kaiser_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::kaiser_kernel<float,T>,o.size(),n,b,o.data(),(!sym&&n%2==0)?n+1:n);}
template <class T> void parzen_fp32_compute_device(int n,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("parzen_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::parzen_kernel<float>,o.size(),n,o.data(),sym?n:n+1);}
template <class T>
void taylor_fp32_compute_device(
    int n,
    int sidelobe_count,
    T sidelobe_level,
    DeviceArray<float>& output,
    bool normalize,
    bool symmetric)
{
    if (n <= 1 || sidelobe_count < 1 ||
        output.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "taylor_device: n, nbar and output size are inconsistent");
    }
    const int effective = symmetric ? n : n + 1;
    float scale = 1.0F;
    const std::vector<float> coefficients =
        windows_detail::build_taylor_coefficients(
            effective,
            sidelobe_count,
            detail::SimpleSignalTypePolicy<T>::load(sidelobe_level),
            normalize,
            scale);
    DeviceArray<float> device_coefficients =
        DeviceArray<float>::from_host(coefficients);
    cuda_utils::launch_1d_kernel(
        windows_detail::taylor_kernel<float, float>,
        output.size(),
        n,
        device_coefficients.data(),
        static_cast<int>(coefficients.size()),
        scale,
        output.data(),
        effective);
}
template <class T> void triang_fp32_compute_device(int n,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("triang_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::triang_kernel<float>,o.size(),n,o.data(),sym?n:n+1);}

#define INSTANTIATE(T) \
 template void chebwin_fp32_compute_device(int,T,DeviceArray<float>&,bool); \
 template void general_cosine_fp32_compute_device(int,const DeviceArray<T>&,DeviceArray<float>&,bool); \
 template void general_cosine_fp32_compute_device(int,const DeviceArray<T>&,int,DeviceArray<float>&,bool); \
 template void general_gaussian_fp32_compute_device(int,T,T,DeviceArray<float>&,bool); \
 template void hamming_fp32_compute_device<T>(int,DeviceArray<float>&,bool); \
 template void kaiser_fp32_compute_device(int,T,DeviceArray<float>&,bool); \
 template void parzen_fp32_compute_device<T>(int,DeviceArray<float>&,bool); \
 template void taylor_fp32_compute_device(int,int,T,DeviceArray<float>&,bool,bool); \
 template void triang_fp32_compute_device<T>(int,DeviceArray<float>&,bool)
INSTANTIATE(float); INSTANTIATE(__half); INSTANTIATE(std::int32_t); INSTANTIATE(std::int16_t); INSTANTIATE(std::int8_t);
#undef INSTANTIATE
}  // namespace cusignal
