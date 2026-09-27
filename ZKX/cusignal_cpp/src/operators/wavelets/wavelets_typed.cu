#include <cusignal/operators/wavelets/wavelets_typed.h>
#include "wavelets_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>

#include <stdexcept>

namespace cusignal {

namespace wavelets_detail {

template <>
struct WaveletComplexTraits<ComplexFloat> {
    template <typename Scalar>
    __device__ static void store(ComplexFloat* output, Scalar real, Scalar imag)
    {
        output->re = static_cast<float>(real);
        output->im = static_cast<float>(imag);
    }

    template <typename Scalar>
    __device__ static void load(
        const ComplexFloat* input,
        Scalar& real,
        Scalar& imag)
    {
        real = static_cast<Scalar>(input->re);
        imag = static_cast<Scalar>(input->im);
    }
};

}  // namespace wavelets_detail

namespace {

template <typename T>
__global__ void morlet_output_kernel(
    int count,
    T frequency_input,
    T scale_input,
    bool complete,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float frequency = detail::SimpleSignalTypePolicy<T>::load(frequency_input);
    const float scale = detail::SimpleSignalTypePolicy<T>::load(scale_input);
    float real = 0.0F;
    float imag = 0.0F;
    wavelets_detail::morlet_value(
        index, count, frequency, scale, complete, real, imag);
    output[index] = ComplexFloat{real, imag};
}

template <typename T>
__global__ void morlet2_output_kernel(
    int count,
    T scale_input,
    T frequency_input,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float scale = detail::SimpleSignalTypePolicy<T>::load(scale_input);
    const float frequency = detail::SimpleSignalTypePolicy<T>::load(frequency_input);
    float real = 0.0F;
    float imag = 0.0F;
    wavelets_detail::morlet2_value(
        index, count, scale, frequency, real, imag);
    output[index] = ComplexFloat{real, imag};
}

}  // namespace

template <typename T>
void morlet_complex_fp32_compute_device(
    int count,
    T frequency,
    T scale,
    bool complete,
    DeviceArray<ComplexFloat>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (count < 1 || output.size() != static_cast<std::size_t>(count) ||
        detail::SimpleSignalTypePolicy<T>::load(scale) <= 0.0F) {
        throw std::invalid_argument("morlet_device: n and scale must be positive and output size must equal n");
    }
    cuda_utils::launch_1d_kernel(
        morlet_output_kernel<T>,
        output.size(),
        count,
        frequency,
        scale,
        complete,
        output.data());
}

template <typename T>
void morlet2_complex_fp32_compute_device(
    int count,
    T scale,
    T frequency,
    DeviceArray<ComplexFloat>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (count < 1 || output.size() != static_cast<std::size_t>(count) ||
        detail::SimpleSignalTypePolicy<T>::load(scale) <= 0.0F) {
        throw std::invalid_argument("morlet2_device: n and scale must be positive and output size must equal n");
    }
    cuda_utils::launch_1d_kernel(
        morlet2_output_kernel<T>,
        output.size(),
        count,
        scale,
        frequency,
        output.data());
}

template <typename T>
void ricker_fp32_compute_device(int count, T width, DeviceArray<float>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (count < 1 || output.size() != static_cast<std::size_t>(count) ||
        detail::SimpleSignalTypePolicy<T>::load(width) <= 0.0F) {
        throw std::invalid_argument("ricker_device: n and width must be positive and output size must equal n");
    }
    cuda_utils::launch_1d_kernel(
        wavelets_detail::ricker_kernel<float, T>,
        output.size(),
        output.data(),
        count,
        width);
}

template <typename T>
void cwt_complex_fp32_compute_device(
    const DeviceArray<T>& input,
    const CwtDeviceWorkspace& workspace,
    DeviceArray<ComplexFloat>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (workspace.data_count != static_cast<int>(input.size()) ||
        workspace.width_count < 0 ||
        workspace.max_wavelet_length != workspace.data_count ||
        workspace.lengths.size() != static_cast<std::size_t>(workspace.width_count) ||
        workspace.wavelets.size() !=
            static_cast<std::size_t>(workspace.width_count) * input.size() ||
        output.size() != workspace.output_size()) {
        throw std::invalid_argument("cwt_device: input, workspace and output shapes are inconsistent");
    }
    if (output.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::cwt_convolution_kernel<T, ComplexFloat, float>,
        output.size(),
        input.data(),
        workspace.data_count,
        workspace.wavelets.data(),
        workspace.lengths.data(),
        workspace.max_wavelet_length,
        workspace.width_count,
        output.data());
}

void cwt_extract_real_fp32_device(
    const DeviceArray<ComplexFloat>& input,
    DeviceArray<float>& output)
{
    if (input.size() != output.size())
        throw std::invalid_argument("cwt_device: real extraction size mismatch");
    if (output.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::cwt_extract_real_kernel,
        output.size(), input.data(), output.data(), output.size());
}

template <typename T>
void qmf_device(const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported qmf input dtype");
    if (hk.size() != out.size()) throw std::invalid_argument("qmf_device size mismatch");
    if (hk.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::qmf_kernel, hk.size(), out.data(), hk.size());
}

template <typename T>
void qmf_device(
    const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out,
    const std::vector<int>& shape)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported qmf input dtype");
    if (shape.empty()) throw std::invalid_argument("qmf shape must not be empty");
    std::size_t input_size = 1;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument("qmf shape");
        if (extent != 0 && input_size > hk.size() / static_cast<std::size_t>(extent))
            throw std::invalid_argument("qmf data/shape mismatch");
        input_size *= static_cast<std::size_t>(extent);
    }
    if (input_size != hk.size()
        || out.size() != static_cast<std::size_t>(shape.front()))
        throw std::invalid_argument("qmf data/output shape mismatch");
    if (out.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::qmf_kernel, out.size(), out.data(), out.size());
}

#define INSTANTIATE_WAVELET_GPU(T) \
    template void morlet_complex_fp32_compute_device(int, T, T, bool, DeviceArray<ComplexFloat>&); \
    template void morlet2_complex_fp32_compute_device(int, T, T, DeviceArray<ComplexFloat>&); \
    template void ricker_fp32_compute_device(int, T, DeviceArray<float>&); \
    template void cwt_complex_fp32_compute_device( \
        const DeviceArray<T>&, const CwtDeviceWorkspace&, DeviceArray<ComplexFloat>&); \
    template void qmf_device(const DeviceArray<T>&, DeviceArray<std::int64_t>&); \
    template void qmf_device( \
        const DeviceArray<T>&, DeviceArray<std::int64_t>&, const std::vector<int>&)

INSTANTIATE_WAVELET_GPU(float);
INSTANTIATE_WAVELET_GPU(__half);
INSTANTIATE_WAVELET_GPU(std::int32_t);
INSTANTIATE_WAVELET_GPU(std::int16_t);
INSTANTIATE_WAVELET_GPU(std::int8_t);

#undef INSTANTIATE_WAVELET_GPU

}  // namespace cusignal
