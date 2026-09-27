#pragma once

#include <cusignal/operators/filter_design/filter_design_typed.h>
#include <cusignal/backends/fft/fft_interface.h>

#include <cmath>
#include <cstdint>
namespace cusignal {
namespace filter_design_detail {

template <class T>
using Accumulator = float;

template <class T>
__device__ Accumulator<T> load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <class T, class Value>
__device__ T store(Value value)
{
    return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
}

template <class Value>
__device__ Value sinc(Value value)
{
    const Value epsilon = static_cast<Value>(1.0e-7);
    const Value pi = static_cast<Value>(3.14159265358979323846);
    return fabs(value) < epsilon
        ? static_cast<Value>(1)
        : sin(pi * value) / (pi * value);
}

template <class Value>
__device__ Value firwin_prepared_ideal(
    const float* bands, int band_count, Value sample_offset)
{
    Value result = static_cast<Value>(0);
    for (int band = 0; band < band_count; ++band) {
        const Value left = static_cast<Value>(bands[2 * band]);
        const Value right = static_cast<Value>(bands[2 * band + 1]);
        result += right * sinc(right * sample_offset)
            - left * sinc(left * sample_offset);
    }
    return result;
}

__device__ std::uint64_t fp32_to_fp64_storage_bits(float value)
{
    union FloatStorageBits {
        float value;
        std::uint32_t bits;
    } source{value};
    const std::uint64_t sign =
        static_cast<std::uint64_t>(source.bits >> 31U) << 63U;
    const std::uint32_t exponent = (source.bits >> 23U) & 0xffU;
    std::uint32_t fraction = source.bits & 0x7fffffU;
    if (exponent == 0U) {
        if (fraction == 0U) return sign;
        int shift = 0;
        while ((fraction & 0x400000U) == 0U) {
            fraction <<= 1U;
            ++shift;
        }
        const std::uint64_t widened_exponent =
            static_cast<std::uint64_t>(896 - shift) << 52U;
        const std::uint64_t widened_fraction =
            static_cast<std::uint64_t>(fraction & 0x3fffffU) << 30U;
        return sign | widened_exponent | widened_fraction;
    }
    if (exponent == 0xffU) {
        return sign | (UINT64_C(0x7ff) << 52U) |
            (static_cast<std::uint64_t>(fraction) << 29U);
    }
    const std::uint64_t widened_exponent =
        static_cast<std::uint64_t>(exponent + 896U) << 52U;
    const std::uint64_t widened_fraction =
        static_cast<std::uint64_t>(fraction) << 29U;
    return sign | widened_exponent | widened_fraction;
}

__device__ float firwin_prepared_unscaled_value(
    int tap, int numtaps, const float* bands, int band_count,
    const float* window)
{
    const float midpoint = static_cast<float>(0.5) * (numtaps - 1);
    const float offset = tap - midpoint;
    return firwin_prepared_ideal(
        bands, band_count, offset) * window[tap];
}

__global__ void firwin_prepared_normalization_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    float* normalization)
{
    __shared__ float partial[256];
    const int lane = static_cast<int>(threadIdx.x);
    const float midpoint = static_cast<float>(0.5) * (numtaps - 1);
    const float pi = static_cast<float>(3.14159265358979323846);
    const float left = bands[0];
    const float right = bands[1];
    const float scale_frequency = left == static_cast<float>(0)
        ? static_cast<float>(0)
        : (right == static_cast<float>(1)
            ? static_cast<float>(1)
            : static_cast<float>(0.5) * (left + right));
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = lane; index < numtaps; index += 256) {
        const float sample_offset = index - midpoint;
        sum.add(firwin_prepared_unscaled_value(
            index, numtaps, bands, band_count, window)
            * cos(pi * sample_offset * scale_frequency));
    }
    partial[lane] = sum.value();
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) normalization[0] = partial[0];
}

__global__ void firwin_prepared_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    const float* normalization,
    float* output,
    bool scale)
{
    const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tap >= numtaps) return;
    const float divisor = scale ? normalization[0] : static_cast<float>(1);
    output[tap] = firwin_prepared_unscaled_value(
        tap, numtaps, bands, band_count, window) / divisor;
}

__global__ void firwin_prepared_fp64_storage_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    const float* normalization,
    std::uint64_t* output,
    bool scale)
{
    const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tap >= numtaps) return;
    const float divisor = scale ? normalization[0] : static_cast<float>(1);
    output[tap] = fp32_to_fp64_storage_bits(
        firwin_prepared_unscaled_value(
            tap, numtaps, bands, band_count, window) / divisor);
}

inline constexpr int firwin_fused_max_taps = 1024;

__global__ void firwin_prepared_fused_fp64_storage_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    std::uint64_t* output,
    bool scale)
{
    __shared__ float unscaled[firwin_fused_max_taps];
    __shared__ float partial[256];
    const int lane = static_cast<int>(threadIdx.x);
    const float midpoint = static_cast<float>(0.5) * (numtaps - 1);
    const float pi = static_cast<float>(3.14159265358979323846);
    const float left = bands[0];
    const float right = bands[1];
    const float scale_frequency = left == static_cast<float>(0)
        ? static_cast<float>(0)
        : (right == static_cast<float>(1)
            ? static_cast<float>(1)
            : static_cast<float>(0.5) * (left + right));
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = lane; index < numtaps; index += 256) {
        const float value = firwin_prepared_unscaled_value(
            index, numtaps, bands, band_count, window);
        unscaled[index] = value;
        if (scale)
            sum.add(value * cos(pi * (index - midpoint) * scale_frequency));
    }
    partial[lane] = sum.value();
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float divisor = scale ? partial[0] : static_cast<float>(1);
    for (int index = lane; index < numtaps; index += 256)
        output[index] = fp32_to_fp64_storage_bits(unscaled[index] / divisor);
}

template <class T>
__global__ void firwin_lowpass_explicit_window_fused_fp32_kernel(
    int numtaps,
    const T* cutoff,
    const float* window,
    float fs,
    float* output,
    bool scale)
{
    __shared__ float unscaled[firwin_fused_max_taps];
    __shared__ float partial[256];
    const int lane = static_cast<int>(threadIdx.x);
    const float normalized_cutoff = load(cutoff[0]) / (0.5F * fs);
    const float midpoint = 0.5F * (numtaps - 1);
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = lane; index < numtaps; index += 256) {
        const float offset = index - midpoint;
        const float value = normalized_cutoff * sinc(normalized_cutoff * offset)
            * window[index];
        unscaled[index] = value;
        if (scale) sum.add(value);
    }
    partial[lane] = sum.value();
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float divisor = scale ? partial[0] : 1.0F;
    for (int index = lane; index < numtaps; index += 256)
        output[index] = unscaled[index] / divisor;
}

template <class T, class Value>
__device__ ComplexFloat firwin2_positive_spectrum(
    int frequency_index,
    int frequency_count,
    const T* frequencies,
    const T* gains,
    int point_count,
    int numtaps,
    bool antisymmetric,
    Value nyquist)
{
    const Value pi = static_cast<Value>(3.14159265358979323846);
    const Value spacing = nyquist / static_cast<Value>(frequency_count - 1);
    const Value frequency = frequency_index * spacing;
    int segment = 0;
    for (int index = point_count - 2; index >= 0; --index) {
        if (frequency >= static_cast<Value>(load(frequencies[index]))) {
            segment = index;
            break;
        }
    }
    const Value left_frequency = static_cast<Value>(load(frequencies[segment]));
    const Value right_frequency = static_cast<Value>(load(frequencies[segment + 1]));
    const Value denominator = right_frequency - left_frequency;
    const Value ratio = fabs(denominator) > static_cast<Value>(1.0e-30)
        ? (frequency - left_frequency) / denominator
        : static_cast<Value>(0);
    const Value gain = static_cast<Value>(load(gains[segment]))
            * (static_cast<Value>(1) - ratio)
        + static_cast<Value>(load(gains[segment + 1])) * ratio;
    const int filter_type = antisymmetric
        ? ((numtaps % 2 == 0) ? 4 : 3)
        : ((numtaps % 2 == 0) ? 2 : 1);
    const Value phase = -static_cast<Value>(numtaps - 1)
        / static_cast<Value>(2) * pi * frequency / nyquist;
    const Value cosine = cos(phase);
    const Value sine = sin(phase);
    return filter_type > 2
        ? ComplexFloat{
            static_cast<float>(-gain * sine),
            static_cast<float>(gain * cosine)}
        : ComplexFloat{
            static_cast<float>(gain * cosine),
            static_cast<float>(gain * sine)};
}

template <class T, class Value>
__global__ void firwin2_build_spectrum_kernel(
    const T* frequencies,
    const T* gains,
    int point_count,
    int frequency_count,
    int numtaps,
    bool antisymmetric,
    Value nyquist,
    ComplexFloat* spectrum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = 2 * (frequency_count - 1);
    if (index >= total) {
        return;
    }
    if (index < frequency_count) {
        spectrum[index] = firwin2_positive_spectrum(
            index, frequency_count, frequencies, gains, point_count,
            numtaps, antisymmetric, nyquist);
    } else {
        const ComplexFloat value = firwin2_positive_spectrum(
            total - index, frequency_count, frequencies, gains, point_count,
            numtaps, antisymmetric, nyquist);
        spectrum[index] = ComplexFloat{value.re, -value.im};
    }
}

template <class Window, class Output>
__global__ void firwin2_apply_window_kernel(
    const ComplexFloat* input,
    const Window* window,
    Output* output,
    int numtaps,
    bool zero_center)
{
    const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tap >= numtaps) {
        return;
    }
    using Value = Accumulator<Output>;
    Value result = static_cast<Value>(input[tap].re)
        * static_cast<Value>(load(window[tap]));
    if (zero_center && tap == numtaps / 2) {
        result = static_cast<Value>(0);
    }
    output[tap] = store<Output>(result);
}

} // namespace filter_design_detail
} // namespace cusignal
