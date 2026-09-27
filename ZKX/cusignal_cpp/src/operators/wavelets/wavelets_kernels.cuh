#pragma once

#include "simple_signal_typed.h"

#include <cmath>
#include <cstddef>

namespace cusignal::wavelets_detail {

template <typename Complex>
struct WaveletComplexTraits;

template <typename T>
__host__ __device__ inline auto wavelet_load(T value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::load(value);
    } else {
        return value;
    }
}

template <typename T, typename Scalar>
__host__ __device__ inline T wavelet_store(Scalar value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
    } else {
        return static_cast<T>(value);
    }
}

__global__ void qmf_kernel(std::int64_t* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::int64_t sign = (index & 1U) ? -1 : 1;
    output[index] = static_cast<std::int64_t>(count - (index + 1)) * sign;
}

template <typename Scalar>
__device__ inline void morlet_value(
    int index,
    int count,
    Scalar frequency,
    Scalar scale,
    bool complete,
    Scalar& real,
    Scalar& imag)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar two_pi = static_cast<Scalar>(2) * pi;
    const Scalar start = -scale * two_pi;
    const Scalar delta = count == 1
        ? Scalar{0}
        : (static_cast<Scalar>(2) * scale * two_pi) /
            static_cast<Scalar>(count - 1);
    const Scalar x = start + delta * static_cast<Scalar>(index);
    const Scalar correction = complete
        ? exp(static_cast<Scalar>(-0.5F) * frequency * frequency)
        : Scalar{0};
    const Scalar envelope =
        exp(static_cast<Scalar>(-0.5F) * x * x) *
        pow(pi, static_cast<Scalar>(-0.25F));
    real = (cos(frequency * x) - correction) * envelope;
    imag = sin(frequency * x) * envelope;
}

template <typename Scalar>
__device__ inline void morlet2_value(
    int index,
    int count,
    Scalar scale,
    Scalar frequency,
    Scalar& real,
    Scalar& imag)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar x =
        (static_cast<Scalar>(index) -
            static_cast<Scalar>(0.5F) * static_cast<Scalar>(count - 1)) /
        scale;
    const Scalar envelope =
        sqrt(static_cast<Scalar>(1) / scale) *
        exp(static_cast<Scalar>(-0.5F) * x * x) *
        pow(pi, static_cast<Scalar>(-0.25F));
    real = cos(frequency * x) * envelope;
    imag = sin(frequency * x) * envelope;
}

template <typename Scalar>
__device__ inline Scalar ricker_value(int index, int count, Scalar width)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar position =
        static_cast<Scalar>(index) -
        static_cast<Scalar>(0.5F) * static_cast<Scalar>(count - 1);
    const Scalar width_squared = width * width;
    const Scalar position_squared = position * position;
    const Scalar amplitude =
        static_cast<Scalar>(2) /
        (sqrt(static_cast<Scalar>(3) * width) *
            pow(pi, static_cast<Scalar>(0.25F)));
    return amplitude *
        (static_cast<Scalar>(1) - position_squared / width_squared) *
        exp(-position_squared / (static_cast<Scalar>(2) * width_squared));
}

template <typename T, typename Parameter>
__global__ void ricker_kernel(
    T* output,
    int count,
    Parameter width_input)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    using Scalar = decltype(wavelet_load(width_input));
    const Scalar width = static_cast<Scalar>(wavelet_load(width_input));
    output[index] = wavelet_store<T>(ricker_value(index, count, width));
}

template <typename Input, typename Complex, typename Scalar>
__global__ void cwt_convolution_kernel(
    const Input* data,
    int data_count,
    const Complex* wavelets,
    const int* lengths,
    int max_wavelet_length,
    int width_count,
    Complex* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = width_count * data_count;
    if (index >= total) {
        return;
    }
    const int width_index = index / data_count;
    const int sample_index = index % data_count;
    const int length = lengths[width_index];
    const int start = (length - 1) / 2;
    const int full_index = start + sample_index;
    const int first_candidate = full_index - (length - 1);
    const int input_min = first_candidate > 0 ? first_candidate : 0;
    const int input_max = full_index < data_count - 1
        ? full_index
        : data_count - 1;
    compute_policy::FmaFloatAccumulator real_sum;
    compute_policy::FmaFloatAccumulator imag_sum;
    const Complex* wavelet =
        wavelets + static_cast<long long>(width_index) * max_wavelet_length;
    for (int input_index = input_min; input_index <= input_max; ++input_index) {
        const int coefficient = full_index - input_index;
        Scalar coefficient_real = Scalar{0};
        Scalar coefficient_imag = Scalar{0};
        WaveletComplexTraits<Complex>::load(
            wavelet + length - 1 - coefficient,
            coefficient_real,
            coefficient_imag);
        const Scalar input_value =
            static_cast<Scalar>(wavelet_load(data[input_index]));
        real_sum.add_product(input_value, coefficient_real);
        imag_sum.add_product(-input_value, coefficient_imag);
    }
    WaveletComplexTraits<Complex>::store(
        output + index, real_sum.value(), imag_sum.value());
}

__global__ void cwt_extract_real_kernel(
    const ComplexFloat* input,
    float* output,
    std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index].re;
}

}  // namespace cusignal::wavelets_detail
