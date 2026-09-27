#pragma once

#include <cusignal/operators/convolution/convolution_typed.h>

namespace cusignal {
namespace convolution_detail {

template <class T>
using Accumulator = float;

template <class T>
using CorrelateDirectAccumulator =
    std::conditional_t<std::is_integral_v<T>,
        compute_policy::modular_unsigned_t<T>,
        compute_policy::FmaFloatAccumulator>;

template <class T>
__device__ Accumulator<T> load(T value)
{
    return detail::ConvolutionTypePolicy<T>::load(value);
}

template <class T>
__device__ T signed_from_bits_device(std::make_unsigned_t<T> raw)
{
    return compute_policy::signed_from_modular_bits<T>(raw);
}

template <class T>
__device__ T rounded_modular_store_device(float value)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
    using U = std::make_unsigned_t<T>;
    constexpr int bits = std::numeric_limits<U>::digits;
    const float modulus = ldexpf(1.0F, bits);
    float wrapped = fmodf(rintf(value), modulus);
    if constexpr (std::is_same_v<T, std::int32_t>) {
        // Adding 2^32 to a small negative FP32 value destroys its low bits
        // (for example -3 becomes exactly 2^32). Keep the value in the signed
        // interval before converting, so the FP32 FFT path preserves INT32
        // modular boundary semantics without device FP64.
        const float sign = ldexpf(1.0F, bits - 1);
        if (wrapped >= sign) wrapped -= modulus;
        if (wrapped < -sign) wrapped += modulus;
        return static_cast<T>(wrapped);
    } else {
        if (wrapped < 0.0F) wrapped += modulus;
        return signed_from_bits_device<T>(static_cast<U>(wrapped));
    }
}

template <class T>
__device__ void correlate_direct_accumulate(
    CorrelateDirectAccumulator<T>& sum,
    T left,
    T right)
{
    if constexpr (std::is_integral_v<T>) {
        sum = compute_policy::modular_multiply_add<T>(sum, left, right);
    } else {
        sum.add_product(load(left), load(right));
    }
}

template <class T>
__device__ T correlate_direct_store(CorrelateDirectAccumulator<T> value)
{
    if constexpr (std::is_integral_v<T>) {
        return compute_policy::signed_from_modular_bits<T>(value);
    } else {
        return detail::ConvolutionTypePolicy<T>::store(value.value());
    }
}

template <class T>
__device__ T correlate_fft_store(float value)
{
    if constexpr (std::is_integral_v<T>) {
        return rounded_modular_store_device<T>(value);
    } else {
        return detail::ConvolutionTypePolicy<T>::store(value);
    }
}

template <class T>
__global__ void correlate_real_direct_kernel(
    const T* in1,
    int len1,
    const T* in2,
    int len2,
    T* out,
    int out_len,
    int start)
{
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= out_len) {
        return;
    }
    const int full_index = start + output_index;
    const int begin = max(0, len2 - 1 - full_index);
    const int end = min(len2 - 1, len1 + len2 - 2 - full_index);
    CorrelateDirectAccumulator<T> sum{};
    for (int kernel_index = begin; kernel_index <= end; ++kernel_index) {
        const int input_index = full_index - (len2 - 1 - kernel_index);
        correlate_direct_accumulate<T>(sum, in1[input_index], in2[kernel_index]);
    }
    out[output_index] = correlate_direct_store<T>(sum);
}

template <class T>
__global__ void pack_1d_kernel(
    const T* input,
    int length,
    ComplexFloat* output,
    int fft_length,
    bool reverse)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= fft_length) {
        return;
    }
    if (index < length) {
        const int source = reverse ? length - 1 - index : index;
        output[index] = ComplexFloat{static_cast<float>(load(input[source])), 0.0F};
    } else {
        output[index] = ComplexFloat{0.0F, 0.0F};
    }
}

template <class = void>
__global__ void multiply_kernel(
    const ComplexFloat* left,
    const ComplexFloat* right,
    ComplexFloat* output,
    int total)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {
        return;
    }
    const ComplexFloat x = left[index];
    const ComplexFloat y = right[index];
    output[index] = ComplexFloat{
        x.re * y.re - x.im * y.im,
        x.re * y.im + x.im * y.re};
}

template <class T>
__global__ void crop_1d_kernel(
    const ComplexFloat* input, T* output, int output_length, int start)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < output_length) {
        output[index] = correlate_fft_store<T>(input[start + index].re);
    }
}

template <class T>
__global__ void correlate_scalar_kernel(const T* in1, const T* in2, T* output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = correlate_fft_store<T>(load(in1[0]) * load(in2[0]));
    }
}

template <class T>
__global__ void pack_nd_real_kernel(
    const T* input,
    int input_elements,
    const int* shape,
    const int* input_strides,
    const int* fft_strides,
    int rank,
    bool reverse,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= input_elements) return;
    int remainder = index;
    long long target = 0;
    for (int axis = 0; axis < rank; ++axis) {
        int coordinate = remainder / input_strides[axis];
        remainder %= input_strides[axis];
        if (reverse) coordinate = shape[axis] - 1 - coordinate;
        target += static_cast<long long>(coordinate) * fft_strides[axis];
    }
    output[target] = ComplexFloat{load(input[index]), 0.0F};
}

static __global__ void pack_nd_axis_kernel(
    const ComplexFloat* input,
    int axis_length,
    int inner_size,
    int total,
    ComplexFloat* packed)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) return;
    const int line = index / axis_length;
    const int sample = index % axis_length;
    const int outer = line / inner_size;
    const int inner = line % inner_size;
    const long long source = static_cast<long long>(outer) * axis_length * inner_size +
                             static_cast<long long>(sample) * inner_size + inner;
    packed[index] = input[source];
}

static __global__ void unpack_nd_axis_kernel(
    const ComplexFloat* packed,
    int axis_length,
    int inner_size,
    int total,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) return;
    const int line = index / axis_length;
    const int sample = index % axis_length;
    const int outer = line / inner_size;
    const int inner = line % inner_size;
    const long long target = static_cast<long long>(outer) * axis_length * inner_size +
                             static_cast<long long>(sample) * inner_size + inner;
    output[target] = packed[index];
}

template <class T>
__global__ void crop_nd_kernel(
    const ComplexFloat* input,
    T* output,
    int output_elements,
    const int* output_strides,
    const int* fft_strides,
    const int* crop_start,
    int rank)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_elements) return;
    int remainder = index;
    long long source = 0;
    for (int axis = 0; axis < rank; ++axis) {
        const int coordinate = remainder / output_strides[axis];
        remainder %= output_strides[axis];
        source += static_cast<long long>(crop_start[axis] + coordinate) * fft_strides[axis];
    }
    output[index] = correlate_fft_store<T>(input[source].re);
}

__device__ inline int correlate2d_map_index(
    std::int64_t index,
    int size,
    int boundary_code)
{
    if (index >= 0 && index < size) return static_cast<int>(index);
    if (boundary_code == 0) return -1;
    if (boundary_code == 1) {
        std::int64_t wrapped = index % size;
        if (wrapped < 0) wrapped += size;
        return static_cast<int>(wrapped);
    }
    const std::int64_t period = static_cast<std::int64_t>(size) * 2;
    std::int64_t wrapped = index % period;
    if (wrapped < 0) wrapped += period;
    return static_cast<int>(wrapped < size ? wrapped : period - 1 - wrapped);
}

template <class T>
__global__ void correlate2d_boundary_direct_kernel(
    const T* in1,
    int rows1,
    int cols1,
    const T* in2,
    int rows2,
    int cols2,
    T* output,
    int output_rows,
    int output_cols,
    int row_start,
    int col_start,
    int boundary_code,
    T fillvalue)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = output_rows * output_cols;
    if (index >= total) {
        return;
    }
    const int output_row = index / output_cols;
    const int output_col = index % output_cols;
    const int full_row = row_start + output_row;
    const int full_col = col_start + output_col;
    CorrelateDirectAccumulator<T> sum{};
    for (int kernel_row = 0; kernel_row < rows2; ++kernel_row) {
        const std::int64_t source_row = static_cast<std::int64_t>(full_row) -
            (rows2 - 1 - kernel_row);
        const int input_row = correlate2d_map_index(source_row, rows1, boundary_code);
        for (int kernel_col = 0; kernel_col < cols2; ++kernel_col) {
            const std::int64_t source_col = static_cast<std::int64_t>(full_col) -
                (cols2 - 1 - kernel_col);
            const int input_col = correlate2d_map_index(source_col, cols1, boundary_code);
            const T sample = input_row < 0 || input_col < 0
                ? fillvalue
                : in1[input_row * cols1 + input_col];
            correlate_direct_accumulate<T>(
                sum, sample, in2[kernel_row * cols2 + kernel_col]);
        }
    }
    output[index] = correlate_direct_store<T>(sum);
}

} // namespace convolution_detail
} // namespace cusignal
