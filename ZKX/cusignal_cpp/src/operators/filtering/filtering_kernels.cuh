#pragma once

#include <cusignal/operators/filtering/filtering_typed.h>
#include "simple_signal_typed.h"

namespace cusignal::detail {

template <typename T>
__device__ inline void filtering_complex_load(
    const TypedComplex<T>& value,
    float& real,
    float& imag)
{
    real = SimpleSignalTypePolicy<T>::load(value.real);
    imag = SimpleSignalTypePolicy<T>::load(value.imag);
}

template <typename T>
__device__ inline std::enable_if_t<is_simple_signal_input_v<T>>
filtering_complex_load(const T& value, float& real, float& imag)
{
    real = SimpleSignalTypePolicy<T>::load(value);
    imag = 0.0F;
}

__device__ inline void filtering_complex_store(
    ComplexFloat& output,
    float real,
    float imag)
{
    output = ComplexFloat{real, imag};
}

template <typename Input, typename Output, typename Scalar>
__global__ void freq_shift_kernel(
    const Input* input,
    Output* output,
    std::size_t count,
    Scalar frequency,
    Scalar sample_rate)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    Scalar real = Scalar{0};
    Scalar imag = Scalar{0};
    filtering_complex_load(input[index], real, imag);
    const Scalar phase = static_cast<Scalar>(-6.28318530717958647692) * frequency *
        static_cast<Scalar>(index) / sample_rate;
    const Scalar cosine = cos(phase);
    const Scalar sine = sin(phase);
    filtering_complex_store(
        output[index],
        real * cosine - imag * sine,
        real * sine + imag * cosine);
}

template <typename T>
__device__ inline auto filtering_load(T value)
{
    return SimpleSignalTypePolicy<T>::load(value);
}

template <typename T>
__global__ void real_to_complex_float_kernel(
    const T* input,
    std::size_t input_size,
    ComplexFloat* output,
    std::size_t output_size)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_size) {
        return;
    }
    output[index].re = (index < input_size)
        ? static_cast<float>(filtering_load(input[index]))
        : 0.0F;
    output[index].im = 0.0F;
}

template <typename T>
__global__ void real_matrix_to_complex_float_kernel(
    const T* input,
    int input_rows,
    int input_cols,
    ComplexFloat* output,
    int output_rows,
    int output_cols)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = output_rows * output_cols;
    if (linear_index >= total) {
        return;
    }

    const int row = linear_index / output_cols;
    const int column = linear_index % output_cols;
    output[linear_index].re = (row < input_rows && column < input_cols)
        ? static_cast<float>(filtering_load(
            input[static_cast<std::size_t>(row) * input_cols + column]))
        : 0.0F;
    output[linear_index].im = 0.0F;
}

template <typename T, typename Value>
__device__ inline T filtering_store(Value value)
{
    return SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
}

template <typename T>
__global__ void complex_float_real_store_kernel(
    const ComplexFloat* input,
    T* output,
    std::size_t count,
    float scale)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = filtering_store<T>(input[index].re * scale);
}

template <typename T>
__device__ inline T resample_integral_wrap_device(
    std::make_unsigned_t<T> raw)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
    constexpr int bits = static_cast<int>(sizeof(T) * 8);
    const unsigned long long encoded = static_cast<unsigned long long>(raw);
    const unsigned long long sign = 1ULL << (bits - 1);
    const unsigned long long modulus = 1ULL << bits;
    const long long decoded = encoded >= sign
        ? static_cast<long long>(encoded) - static_cast<long long>(modulus)
        : static_cast<long long>(encoded);
    return static_cast<T>(decoded);
}

template <typename T>
__device__ inline float resample_frequency_multiply_value(
    T value, float window)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const T typed_window = static_cast<T>(window);
        const Unsigned product = static_cast<Unsigned>(
            static_cast<unsigned long long>(static_cast<Unsigned>(value)) *
            static_cast<unsigned long long>(
                static_cast<Unsigned>(typed_window)));
        return static_cast<float>(resample_integral_wrap_device<T>(product));
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half typed_window = filtering_store<T>(window);
        return filtering_load(filtering_store<T>(
            filtering_load(value) * filtering_load(typed_window)));
    } else {
        return static_cast<float>(value * static_cast<T>(window));
    }
}

template <typename T>
__device__ inline float resample_frequency_add_value(float left, float right)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned sum = static_cast<Unsigned>(
            static_cast<unsigned long long>(
                static_cast<Unsigned>(static_cast<T>(left))) +
            static_cast<unsigned long long>(
                static_cast<Unsigned>(static_cast<T>(right))));
        return static_cast<float>(resample_integral_wrap_device<T>(sum));
    } else if constexpr (std::is_same_v<T, __half>) {
        return filtering_load(filtering_store<T>(left + right));
    } else {
        return static_cast<float>(static_cast<T>(left + right));
    }
}

template <typename T>
__global__ void resample_pack_axis_kernel(
    const T* input,
    int sample_count,
    int inner_count,
    int line_count,
    bool frequency_domain,
    const float* window,
    bool has_window,
    ComplexFloat* packed)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * sample_count;
    if (index >= total) return;
    const int line = index / sample_count;
    const int sample = index % sample_count;
    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long source =
        (static_cast<long long>(outer) * sample_count * inner_count) +
        static_cast<long long>(sample) * inner_count + inner;
    float value = filtering_load(input[source]);
    if (frequency_domain && has_window)
        value = resample_frequency_multiply_value(input[source], window[sample]);
    packed[index] = ComplexFloat{value, 0.0F};
}

__global__ void resample_apply_window_kernel(
    ComplexFloat* spectrum,
    int sample_count,
    int line_count,
    const float* window)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * sample_count;
    if (index >= total) return;
    const float factor = window[index % sample_count];
    spectrum[index].re *= factor;
    spectrum[index].im *= factor;
}

template <typename T>
__global__ void resample_copy_spectrum_kernel(
    const ComplexFloat* input,
    int input_count,
    ComplexFloat* output,
    int output_count,
    int line_count,
    bool frequency_domain)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * output_count;
    if (index >= total) return;
    const int line = index / output_count;
    const int destination = index % output_count;
    const int copied = input_count < output_count ? input_count : output_count;
    const int nyquist_end = copied / 2 + 1;
    int source = -1;
    if (destination < nyquist_end) {
        source = destination;
    } else if (copied > 2 &&
               destination >= output_count + nyquist_end - copied) {
        source = input_count + destination - output_count;
    }
    ComplexFloat value{0.0F, 0.0F};
    if (source >= 0)
        value = input[static_cast<long long>(line) * input_count + source];
    if (copied % 2 == 0) {
        if (output_count < input_count &&
            destination == output_count - copied / 2) {
            const ComplexFloat negative = input[
                static_cast<long long>(line) * input_count +
                input_count - copied / 2];
            if (frequency_domain) {
                value.re = resample_frequency_add_value<T>(
                    value.re, negative.re);
                value.im = 0.0F;
            } else {
                value.re += negative.re;
                value.im += negative.im;
            }
        } else if (input_count < output_count &&
                   destination == copied / 2) {
            if (frequency_domain) {
                value.re = resample_frequency_multiply_value<T>(
                    filtering_store<T>(value.re), 0.5F);
                value.im = 0.0F;
            } else {
                value.re *= 0.5F;
                value.im *= 0.5F;
            }
        } else if (input_count < output_count &&
                   destination == output_count - copied / 2) {
            value = input[static_cast<long long>(line) * input_count +
                          copied / 2];
            if (frequency_domain) {
                value.re = resample_frequency_multiply_value<T>(
                    filtering_store<T>(value.re), 0.5F);
                value.im = 0.0F;
            } else {
                value.re *= 0.5F;
                value.im *= 0.5F;
            }
        }
    }
    output[index] = value;
}

__global__ void resample_store_axis_kernel(
    const ComplexFloat* packed,
    int output_count,
    int inner_count,
    int line_count,
    float scale,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * output_count;
    if (index >= total) return;
    const int line = index / output_count;
    const int sample = index % output_count;
    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long destination =
        (static_cast<long long>(outer) * output_count * inner_count) +
        static_cast<long long>(sample) * inner_count + inner;
    output[destination] = packed[index].re * scale;
}

template <typename T>
__global__ void channelize_poly_accumulate_kernel(
    const T* input,
    int input_length,
    const T* taps,
    int taps_per_phase,
    int channel_count,
    int point_count,
    ComplexFloat* polyphase)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = point_count * channel_count;
    if (linear_index >= total) {
        return;
    }

    const int point = linear_index / channel_count;
    const int channel = linear_index % channel_count;
    compute_policy::FmaFloatAccumulator sum;
    for (int tap = 0; tap < taps_per_phase; ++tap) {
        const int input_index = point * channel_count + (channel_count - 1 - channel) -
            tap * channel_count;
        if (input_index >= 0 && input_index < input_length) {
            sum.add_product(
                filtering_load(taps[channel + tap * channel_count]),
                filtering_load(input[input_index]));
        }
    }
    polyphase[linear_index] = ComplexFloat{sum.value(), 0.0F};
}

static __global__ void channelize_poly_finalize_float_kernel(
    const ComplexFloat* fft_output,
    int channel_count,
    int point_count,
    ComplexFloat* output)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = point_count * channel_count;
    if (linear_index >= total) {
        return;
    }
    const int point = linear_index / channel_count;
    const int channel = linear_index % channel_count;
    const ComplexFloat value = fft_output[linear_index];
    output[channel * point_count + point] = ComplexFloat{value.re, -value.im};
}

void channelize_poly_fft_device(
    const DeviceArray<ComplexFloat>& polyphase,
    int channel_count,
    int point_count,
    DeviceArray<ComplexFloat>& output);

template <typename T>
__global__ void upfirdn_kernel(
    const T* taps,
    int tap_count,
    const T* input,
    int input_length,
    int up,
    int down,
    T* output,
    int output_length,
    int batch_count,
    int input_batch_stride,
    int input_sample_stride,
    int output_batch_stride,
    int output_sample_stride)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_count * output_length;
    if (linear_index >= total) {
        return;
    }

    const int batch = linear_index / output_length;
    const int output_index = linear_index % output_length;
    const long long upsampled_index = static_cast<long long>(output_index) * down;
    const T* input_batch = input_length == 0
        ? input
        : input + static_cast<long long>(batch) * input_batch_stride;
    T* output_batch = output + static_cast<long long>(batch) * output_batch_stride;
    compute_policy::FmaFloatAccumulator sum;

    for (int tap = 0; tap < tap_count; ++tap) {
        const long long candidate = upsampled_index - tap;
        if (candidate % up != 0) {
            continue;
        }
        const long long input_index = candidate / up;
        if (input_index >= 0 && input_index < input_length) {
            sum.add_product(
                filtering_load(taps[tap]),
                filtering_load(input_batch[input_index * input_sample_stride]));
        }
    }
    output_batch[static_cast<long long>(output_index) * output_sample_stride] =
        filtering_store<T>(sum.value());
}

template <typename Input>
__global__ void decimate_axis_kernel(
    const float* taps,
    int tap_count,
    const Input* input,
    int input_length,
    int q,
    int prefix_zeros,
    int output_start,
    float* output,
    int output_length,
    int inner_count,
    int line_count)
{
    const int linear_index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * output_length;
    if (linear_index >= total) return;

    const int line = linear_index / output_length;
    const int output_index = linear_index % output_length;
    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long input_base =
        (static_cast<long long>(outer) * input_length * inner_count) + inner;
    const long long output_base =
        (static_cast<long long>(outer) * output_length * inner_count) + inner;
    const long long filtered_index =
        (static_cast<long long>(output_index) + output_start) * q;
    compute_policy::FmaFloatAccumulator sum;
    for (int tap = 0; tap < tap_count; ++tap) {
        const long long source = filtered_index - prefix_zeros - tap;
        if (source < 0 || source >= input_length) continue;
        sum.add_product(taps[tap], filtering_load(
            input[input_base + source * inner_count]));
    }
    output[output_base + static_cast<long long>(output_index) * inner_count] =
        sum.value();
}

template <typename T>
__global__ void copy_signal_kernel(const T* input, T* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}

template <typename Output, typename Input>
__device__ inline float resample_poly_scaled_integral_device(
    Input value, int up)
{
    static_assert(std::is_integral_v<Output> && std::is_signed_v<Output>);
    using Unsigned = std::make_unsigned_t<Output>;
    const Output promoted = static_cast<Output>(value);
    const Unsigned product = static_cast<Unsigned>(
        static_cast<unsigned long long>(static_cast<Unsigned>(promoted)) *
        static_cast<unsigned long long>(up));
    return static_cast<float>(
        resample_integral_wrap_device<Output>(product));
}

template <typename T>
__device__ inline float resample_poly_scaled_custom_value_device(
    T value, int up)
{
    if constexpr (std::is_same_v<T, float>) {
        return value * static_cast<float>(up);
    } else if constexpr (std::is_same_v<T, __half>) {
        const float loaded = filtering_load(value);
        return up <= 255
            ? filtering_load(filtering_store<T>(loaded * static_cast<float>(up)))
            : loaded * static_cast<float>(up);
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return resample_poly_scaled_integral_device<std::int32_t>(value, up);
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return up < 32768
            ? resample_poly_scaled_integral_device<std::int16_t>(value, up)
            : resample_poly_scaled_integral_device<std::int32_t>(value, up);
    } else {
        if (up <= 127)
            return resample_poly_scaled_integral_device<std::int8_t>(value, up);
        return up < 32768
            ? resample_poly_scaled_integral_device<std::int16_t>(value, up)
            : resample_poly_scaled_integral_device<std::int32_t>(value, up);
    }
}

template <typename T>
__global__ void resample_poly_custom_coefficients_kernel(
    const T* coefficients, int coefficient_count, int up, float* scaled)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < coefficient_count)
        scaled[index] = resample_poly_scaled_custom_value_device(
            coefficients[index], up);
}

template <typename T>
__global__ void resample_poly_axis_kernel(
    const T* input,
    int input_sample_count,
    int inner_count,
    int line_count,
    const float* coefficients,
    int coefficient_count,
    int up,
    int down,
    int prefix_zeros,
    int output_start,
    int output_sample_count,
    float* output)
{
    const int linear_index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * output_sample_count;
    if (linear_index >= total) return;

    const int line = linear_index / output_sample_count;
    const int output_index = linear_index % output_sample_count;
    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long input_base =
        static_cast<long long>(outer) * input_sample_count * inner_count + inner;
    const long long output_base =
        static_cast<long long>(outer) * output_sample_count * inner_count + inner;
    const long long filtered =
        (static_cast<long long>(output_index) + output_start) * down;
    compute_policy::FmaFloatAccumulator sum;
    for (int tap = 0; tap < coefficient_count; ++tap) {
        const long long candidate = filtered - prefix_zeros - tap;
        if (candidate % up != 0) continue;
        const long long source = candidate / up;
        if (source < 0 || source >= input_sample_count) continue;
        sum.add_product(coefficients[tap], filtering_load(
            input[input_base + source * inner_count]));
    }
    output[output_base +
        static_cast<long long>(output_index) * inner_count] = sum.value();
}

template <typename T>
__device__ inline float wiener_squared_input(T value)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned raw = static_cast<Unsigned>(value);
        const Unsigned squared = static_cast<Unsigned>(
            static_cast<unsigned long long>(raw) *
            static_cast<unsigned long long>(raw));
        constexpr int bits = static_cast<int>(sizeof(T) * 8);
        const unsigned long long encoded =
            static_cast<unsigned long long>(squared);
        const unsigned long long sign = 1ULL << (bits - 1);
        const unsigned long long modulus = 1ULL << bits;
        const long long signed_value = encoded >= sign
            ? static_cast<long long>(encoded) - static_cast<long long>(modulus)
            : static_cast<long long>(encoded);
        return static_cast<float>(signed_value);
    } else if constexpr (std::is_same_v<T, __half>) {
        const float loaded = filtering_load(value);
        return filtering_load(filtering_store<T>(loaded * loaded));
    } else {
        const float loaded = filtering_load(value);
        return loaded * loaded;
    }
}

template <typename T>
__global__ void wiener_nd_stats_kernel(
    const T* input,
    int element_count,
    const int* input_shape,
    const int* input_strides,
    const int* window_shape,
    int rank,
    int window_volume,
    float* local_mean,
    float* local_variance)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= element_count) return;
    compute_policy::CompensatedFloatAccumulator sum;
    compute_policy::CompensatedFloatAccumulator square_sum;
    for (int window_index = 0; window_index < window_volume; ++window_index) {
        int remaining = window_index;
        int source_index = 0;
        bool inside = true;
        for (int dimension = rank - 1; dimension >= 0; --dimension) {
            const int output_coordinate =
                (index / input_strides[dimension]) % input_shape[dimension];
            const int window_coordinate = remaining % window_shape[dimension];
            remaining /= window_shape[dimension];
            const int source_coordinate = output_coordinate + window_coordinate
                - window_shape[dimension] / 2;
            if (source_coordinate < 0 ||
                source_coordinate >= input_shape[dimension]) {
                inside = false;
                break;
            }
            source_index += source_coordinate * input_strides[dimension];
        }
        if (!inside) continue;
        sum.add(filtering_load(input[source_index]));
        square_sum.add(wiener_squared_input(input[source_index]));
    }
    const float mean = sum.value() / static_cast<float>(window_volume);
    local_mean[index] = mean;
    local_variance[index] = fmaxf(
        square_sum.value() / static_cast<float>(window_volume) - mean * mean,
        0.0F);
}

static __global__ void wiener_mean_noise_kernel(
    const float* local_variance, int count, float* noise)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = 0; index < count; ++index)
        sum.add(local_variance[index]);
    noise[0] = sum.value() / static_cast<float>(count);
}

template <typename T>
__global__ void wiener_result_kernel(
    const T* input,
    const float* local_mean,
    const float* local_variance,
    int count,
    const float* noise,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float variance = local_variance[index];
    const float filtered = (filtering_load(input[index]) - local_mean[index])
        * (1.0F - noise[0] / variance) + local_mean[index];
    output[index] = variance < noise[0] ? local_mean[index] : filtered;
}

__global__ void detrend_batch_kernel(
    const float* input,
    int sample_count,
    const int* breakpoints,
    int breakpoint_count,
    int mode,
    float* output,
    int batch_count,
    int inner_count)
{
    const int batch = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (batch >= batch_count || sample_count <= 0) {
        return;
    }
    using Accumulator = float;
    const int outer = batch / inner_count;
    const int inner = batch % inner_count;
    const long long base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;

    if (mode == 0) {
        compute_policy::CompensatedFloatAccumulator mean_sum;
        for (int index = 0; index < sample_count; ++index) {
            mean_sum.add(input[base + static_cast<long long>(index) * inner_count]);
        }
        const Accumulator mean = mean_sum.value() /
            static_cast<Accumulator>(sample_count);
        for (int index = 0; index < sample_count; ++index) {
            const long long offset = base
                + static_cast<long long>(index) * inner_count;
            output[offset] = input[offset] - mean;
        }
        return;
    }

    int segment_start = 0;
    for (int segment = 0; segment <= breakpoint_count; ++segment) {
        const int segment_end =
            (segment < breakpoint_count) ? breakpoints[segment] : sample_count;
        const int segment_length = segment_end - segment_start;
        if (segment_length <= 0) {
            segment_start = segment_end;
            continue;
        }
        compute_policy::CompensatedFloatAccumulator sum_x_acc;
        compute_policy::CompensatedFloatAccumulator sum_y_acc;
        compute_policy::CompensatedFloatAccumulator sum_xx_acc;
        compute_policy::CompensatedFloatAccumulator sum_xy_acc;
        for (int index = segment_start; index < segment_end; ++index) {
            const Accumulator coordinate = static_cast<Accumulator>(index - segment_start);
            const Accumulator sample = input[
                base + static_cast<long long>(index) * inner_count];
            sum_x_acc.add(coordinate);
            sum_y_acc.add(sample);
            sum_xx_acc.add(coordinate * coordinate);
            sum_xy_acc.add(fmaf(coordinate, sample, 0.0F));
        }
        const Accumulator sum_x = sum_x_acc.value();
        const Accumulator sum_y = sum_y_acc.value();
        const Accumulator sum_xx = sum_xx_acc.value();
        const Accumulator sum_xy = sum_xy_acc.value();
        const Accumulator length = static_cast<Accumulator>(segment_length);
        const Accumulator denominator = length * sum_xx - sum_x * sum_x;
        const Accumulator epsilon = 1.0e-7F;
        const bool valid = fabs(denominator) > epsilon;
        const Accumulator slope = valid
            ? (length * sum_xy - sum_x * sum_y) / denominator
            : Accumulator{0};
        const Accumulator intercept = valid
            ? (sum_y - slope * sum_x) / length
            : sum_y / length;
        for (int index = segment_start; index < segment_end; ++index) {
            const Accumulator coordinate = static_cast<Accumulator>(index - segment_start);
            const long long offset = base
                + static_cast<long long>(index) * inner_count;
            output[offset] = input[offset] - (slope * coordinate + intercept);
        }
        segment_start = segment_end;
    }
}

__global__ void lfilter_zi_fp32_kernel(
    const float* numerator,
    int numerator_count,
    const float* denominator,
    int denominator_count,
    float* normalized_numerator,
    float* normalized_denominator,
    float* augmented,
    float* solution,
    float* output,
    int* status)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    *status = 0;
    const int order_plus_one =
        (numerator_count > denominator_count) ? numerator_count : denominator_count;
    const int order = order_plus_one - 1;
    if (order <= 0) {
        return;
    }
    const float a0 = denominator[0];
    for (int index = 0; index < order_plus_one; ++index) {
        normalized_numerator[index] = (index < numerator_count)
            ? numerator[index] / a0 : 0.0F;
        normalized_denominator[index] = (index < denominator_count)
            ? denominator[index] / a0 : 0.0F;
    }

    for (int row = 0; row < order; ++row) {
        for (int column = 0; column < order; ++column) {
            float companion = 0.0F;
            if (column == 0) {
                companion = -normalized_denominator[row + 1];
            } else if (column == row + 1) {
                companion = 1.0F;
            }
            augmented[row * (order + 1) + column] =
                (row == column ? 1.0F : 0.0F) - companion;
        }
        augmented[row * (order + 1) + order] = normalized_numerator[row + 1] -
            normalized_denominator[row + 1] * normalized_numerator[0];
    }

    const float epsilon = 1.0e-7F;
    for (int pivot = 0; pivot < order; ++pivot) {
        int pivot_row = pivot;
        float pivot_magnitude = fabsf(augmented[pivot * (order + 1) + pivot]);
        for (int row = pivot + 1; row < order; ++row) {
            const float candidate = fabsf(augmented[row * (order + 1) + pivot]);
            if (candidate > pivot_magnitude) {
                pivot_magnitude = candidate;
                pivot_row = row;
            }
        }
        if (pivot_row != pivot) {
            for (int column = pivot; column <= order; ++column) {
                const int first = pivot * (order + 1) + column;
                const int second = pivot_row * (order + 1) + column;
                const float temporary = augmented[first];
                augmented[first] = augmented[second];
                augmented[second] = temporary;
            }
        }
        const float pivot_value = augmented[pivot * (order + 1) + pivot];
        if (isfinite(pivot_value) && fabsf(pivot_value) < epsilon) {
            *status = 1;
            return;
        }
        for (int row = pivot + 1; row < order; ++row) {
            const float factor = augmented[row * (order + 1) + pivot] / pivot_value;
            if (factor == 0.0F) {
                continue;
            }
            for (int column = pivot; column <= order; ++column) {
                augmented[row * (order + 1) + column] -=
                    factor * augmented[pivot * (order + 1) + column];
            }
        }
    }
    for (int row = order - 1; row >= 0; --row) {
        float value = augmented[row * (order + 1) + order];
        for (int column = row + 1; column < order; ++column) {
            value -= augmented[row * (order + 1) + column] * solution[column];
        }
        solution[row] = value / augmented[row * (order + 1) + row];
    }
    for (int index = 0; index < order; ++index) {
        output[index] = solution[index];
    }
}

template <typename T>
__global__ void pad_signal_kernel(
    const T* input,
    int sample_count,
    int edge,
    int mode,
    T* output,
    int extended_length,
    int batch_count,
    int input_batch_stride,
    int input_sample_stride,
    int output_batch_stride,
    int output_sample_stride)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_count * extended_length;
    if (linear_index >= total) {
        return;
    }
    using Accumulator = float;
    const int batch = linear_index / extended_length;
    const int output_index = linear_index % extended_length;
    const T* input_batch = input + static_cast<long long>(batch) * input_batch_stride;
    T* output_batch = output + static_cast<long long>(batch) * output_batch_stride;
    const Accumulator first = static_cast<Accumulator>(filtering_load(input_batch[0]));
    const Accumulator last = static_cast<Accumulator>(filtering_load(
        input_batch[static_cast<long long>(sample_count - 1) * input_sample_stride]));
    Accumulator value = Accumulator{0};
    if (output_index < edge) {
        const Accumulator reflected = static_cast<Accumulator>(filtering_load(
            input_batch[static_cast<long long>(edge - output_index) * input_sample_stride]));
        value = (mode == 0) ? Accumulator{2} * first - reflected
            : ((mode == 1) ? reflected : first);
    } else if (output_index < edge + sample_count) {
        value = static_cast<Accumulator>(filtering_load(
            input_batch[static_cast<long long>(output_index - edge) * input_sample_stride]));
    } else {
        const int right_index = output_index - edge - sample_count;
        const int reflected_index = sample_count - 2 - right_index;
        const Accumulator reflected = static_cast<Accumulator>(filtering_load(
            input_batch[static_cast<long long>(reflected_index) * input_sample_stride]));
        value = (mode == 0) ? Accumulator{2} * last - reflected
            : ((mode == 1) ? reflected : last);
    }
    output_batch[static_cast<long long>(output_index) * output_sample_stride] =
        filtering_store<T>(value);
}

template <typename T>
__device__ inline float firfilter2_odd_pad_value(T endpoint, T reflected)
{
    if constexpr (std::is_integral_v<T>) {
        constexpr int bits = static_cast<int>(sizeof(T) * 8);
        constexpr long long modulus = 1LL << bits;
        constexpr long long sign = modulus / 2;
        long long value = 2LL * static_cast<long long>(endpoint)
            - static_cast<long long>(reflected);
        value %= modulus;
        if (value < 0) value += modulus;
        if (value >= sign) value -= modulus;
        return static_cast<float>(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        const T doubled = SimpleSignalTypePolicy<T>::store(
            2.0F * SimpleSignalTypePolicy<T>::load(endpoint));
        const T value = SimpleSignalTypePolicy<T>::store(
            SimpleSignalTypePolicy<T>::load(doubled)
            - SimpleSignalTypePolicy<T>::load(reflected));
        return SimpleSignalTypePolicy<T>::load(value);
    } else {
        const T doubled = static_cast<T>(2.0F * endpoint);
        const T value = static_cast<T>(doubled - reflected);
        return static_cast<float>(value);
    }
}

template <typename T>
__global__ void firfilter2_pad_axis_kernel(
    const T* input,
    int sample_count,
    int inner_count,
    int outer_count,
    int edge,
    int mode,
    float* output,
    int extended_sample_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer_count * extended_sample_count * inner_count;
    if (index >= total) return;
    const int inner = index % inner_count;
    const int sample = (index / inner_count) % extended_sample_count;
    const int outer = index / (extended_sample_count * inner_count);
    const long long input_base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
    float value = 0.0F;
    if (sample < edge) {
        const T endpoint = input[input_base];
        const T reflected = input[input_base
            + static_cast<long long>(edge - sample) * inner_count];
        value = mode == 0 ? firfilter2_odd_pad_value(endpoint, reflected)
            : (mode == 1 ? SimpleSignalTypePolicy<T>::load(reflected)
                         : SimpleSignalTypePolicy<T>::load(endpoint));
    } else if (sample < edge + sample_count) {
        value = SimpleSignalTypePolicy<T>::load(input[input_base
            + static_cast<long long>(sample - edge) * inner_count]);
    } else {
        const int right = sample - edge - sample_count;
        const T endpoint = input[input_base
            + static_cast<long long>(sample_count - 1) * inner_count];
        const T reflected = input[input_base
            + static_cast<long long>(sample_count - 2 - right) * inner_count];
        value = mode == 0 ? firfilter2_odd_pad_value(endpoint, reflected)
            : (mode == 1 ? SimpleSignalTypePolicy<T>::load(reflected)
                         : SimpleSignalTypePolicy<T>::load(endpoint));
    }
    output[index] = value;
}

template <typename T>
__device__ inline float firfilter2_padded_axis_value(
    const T* input,
    int sample_count,
    int inner_count,
    int outer,
    int inner,
    int edge,
    int mode,
    int padded_sample)
{
    const long long input_base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
    if (padded_sample < edge) {
        const T endpoint = input[input_base];
        const T reflected = input[input_base
            + static_cast<long long>(edge - padded_sample) * inner_count];
        return mode == 0 ? firfilter2_odd_pad_value(endpoint, reflected)
            : (mode == 1 ? SimpleSignalTypePolicy<T>::load(reflected)
                         : SimpleSignalTypePolicy<T>::load(endpoint));
    }
    if (padded_sample < edge + sample_count) {
        return SimpleSignalTypePolicy<T>::load(input[input_base
            + static_cast<long long>(padded_sample - edge) * inner_count]);
    }
    const int right = padded_sample - edge - sample_count;
    const T endpoint = input[input_base
        + static_cast<long long>(sample_count - 1) * inner_count];
    const T reflected = input[input_base
        + static_cast<long long>(sample_count - 2 - right) * inner_count];
    return mode == 0 ? firfilter2_odd_pad_value(endpoint, reflected)
        : (mode == 1 ? SimpleSignalTypePolicy<T>::load(reflected)
                     : SimpleSignalTypePolicy<T>::load(endpoint));
}

template <typename T>
__device__ inline float firfilter2_small_base_state(
    const T* coefficients, int coefficient_count, int state)
{
    float value = 0.0F;
    for (int tap = state + 1; tap < coefficient_count; ++tap)
        value += SimpleSignalTypePolicy<T>::load(coefficients[tap]);
    return value;
}

template <typename T>
__device__ inline float firfilter2_small_forward_value(
    const T* coefficients,
    int coefficient_count,
    const T* input,
    int sample_count,
    int inner_count,
    int outer,
    int inner,
    int edge,
    int mode,
    int padded_sample)
{
    float value = 0.0F;
    const int maximum_tap = padded_sample < coefficient_count - 1
        ? padded_sample : coefficient_count - 1;
    for (int tap = 0; tap <= maximum_tap; ++tap) {
        value += SimpleSignalTypePolicy<T>::load(coefficients[tap])
            * firfilter2_padded_axis_value(
                input, sample_count, inner_count, outer, inner, edge, mode,
                padded_sample - tap);
    }
    if (padded_sample < coefficient_count - 1) {
        value += firfilter2_small_base_state(
            coefficients, coefficient_count, padded_sample)
            * firfilter2_padded_axis_value(
                input, sample_count, inner_count, outer, inner, edge, mode, 0);
    }
    return value;
}

template <typename T>
__device__ inline float firfilter2_small_fused_axis_value(
    const T* coefficients,
    int coefficient_count,
    const T* input,
    int sample_count,
    int inner_count,
    int outer_count,
    int edge,
    int mode,
    int index)
{
    const int inner = index % inner_count;
    const int sample = (index / inner_count) % sample_count;
    const int outer = index / (sample_count * inner_count);
    const int extended_sample_count = sample_count + 2 * edge;
    const int reverse_sample = extended_sample_count - 1 - (edge + sample);
    const int maximum_tap = reverse_sample < coefficient_count - 1
        ? reverse_sample : coefficient_count - 1;
    float value = 0.0F;
    for (int tap = 0; tap <= maximum_tap; ++tap) {
        const int forward_sample = extended_sample_count - 1
            - (reverse_sample - tap);
        value += SimpleSignalTypePolicy<T>::load(coefficients[tap])
            * firfilter2_small_forward_value(
                coefficients, coefficient_count, input, sample_count,
                inner_count, outer, inner, edge, mode, forward_sample);
    }
    if (reverse_sample < coefficient_count - 1) {
        value += firfilter2_small_base_state(
            coefficients, coefficient_count, reverse_sample)
            * firfilter2_small_forward_value(
                coefficients, coefficient_count, input, sample_count,
                inner_count, outer, inner, edge, mode,
                extended_sample_count - 1);
    }
    return value;
}

__device__ inline std::uint64_t firfilter2_fp32_to_fp64_storage_bits(
    float value)
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

template <typename T>
__global__ void firfilter2_small_fused_axis_kernel(
    const T* coefficients, int coefficient_count, const T* input,
    int sample_count, int inner_count, int outer_count, int edge, int mode,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer_count * sample_count * inner_count;
    if (index >= total) return;
    output[index] = firfilter2_small_fused_axis_value(
        coefficients, coefficient_count, input, sample_count, inner_count,
        outer_count, edge, mode, index);
}

template <typename T>
__global__ void firfilter2_small_fused_axis_fp64_storage_kernel(
    const T* coefficients, int coefficient_count, const T* input,
    int sample_count, int inner_count, int outer_count, int edge, int mode,
    std::uint64_t* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer_count * sample_count * inner_count;
    if (index >= total) return;
    const float value = firfilter2_small_fused_axis_value(
        coefficients, coefficient_count, input, sample_count, inner_count,
        outer_count, edge, mode, index);
    output[index] = firfilter2_fp32_to_fp64_storage_bits(value);
}

static __global__ void firfilter2_base_state_kernel(
    const float* coefficients,
    int coefficient_count,
    float* base_state)
{
    const int state = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (state >= coefficient_count - 1) return;
    float value = 0.0F;
    for (int tap = state + 1; tap < coefficient_count; ++tap)
        value += coefficients[tap];
    base_state[state] = value;
}

static __global__ void firfilter2_scale_state_axis_kernel(
    const float* base_state,
    int state_count,
    const float* signal,
    int sample_count,
    int inner_count,
    int outer_count,
    bool use_last,
    float* output_state)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer_count * state_count * inner_count;
    if (index >= total) return;
    const int inner = index % inner_count;
    const int state = (index / inner_count) % state_count;
    const int outer = index / (state_count * inner_count);
    const int sample = use_last ? sample_count - 1 : 0;
    const long long signal_index = static_cast<long long>(outer)
        * sample_count * inner_count + static_cast<long long>(sample) * inner_count
        + inner;
    output_state[index] = base_state[state] * signal[signal_index];
}

static __global__ void firfilter2_reverse_axis_kernel(
    const float* input,
    int sample_count,
    int inner_count,
    int outer_count,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer_count * sample_count * inner_count;
    if (index >= total) return;
    const int inner = index % inner_count;
    const int sample = (index / inner_count) % sample_count;
    const int outer = index / (sample_count * inner_count);
    const long long base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
    output[index] = input[base
        + static_cast<long long>(sample_count - 1 - sample) * inner_count];
}

static __global__ void firfilter2_crop_axis_kernel(
    const float* input,
    int input_sample_count,
    int output_sample_count,
    int inner_count,
    int outer_count,
    int edge,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer_count * output_sample_count * inner_count;
    if (index >= total) return;
    const int inner = index % inner_count;
    const int sample = (index / inner_count) % output_sample_count;
    const int outer = index / (output_sample_count * inner_count);
    const long long input_base = static_cast<long long>(outer)
        * input_sample_count * inner_count + inner;
    output[index] = input[input_base
        + static_cast<long long>(edge + sample) * inner_count];
}

template <typename T>
__global__ void reverse_batch_kernel(
    const T* input,
    int sample_count,
    T* output,
    int batch_count,
    int input_batch_stride,
    int input_sample_stride,
    int output_batch_stride,
    int output_sample_stride)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_count * sample_count;
    if (linear_index >= total) {
        return;
    }
    const int batch = linear_index / sample_count;
    const int output_index = linear_index % sample_count;
    const T* input_batch = input + static_cast<long long>(batch) * input_batch_stride;
    T* output_batch = output + static_cast<long long>(batch) * output_batch_stride;
    output_batch[static_cast<long long>(output_index) * output_sample_stride] =
        input_batch[static_cast<long long>(sample_count - 1 - output_index) * input_sample_stride];
}

template <typename T>
__global__ void crop_batch_kernel(
    const T* input,
    int edge,
    int output_count,
    T* output,
    int batch_count,
    int input_batch_stride,
    int input_sample_stride,
    int output_batch_stride,
    int output_sample_stride)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_count * output_count;
    if (linear_index >= total) {
        return;
    }
    const int batch = linear_index / output_count;
    const int output_index = linear_index % output_count;
    const T* input_batch = input + static_cast<long long>(batch) * input_batch_stride;
    T* output_batch = output + static_cast<long long>(batch) * output_batch_stride;
    output_batch[static_cast<long long>(output_index) * output_sample_stride] =
        input_batch[static_cast<long long>(edge + output_index) * input_sample_stride];
}

template <typename T>
__global__ void scale_zi_from_signal_kernel(
    const T* base_state,
    int state_count,
    const T* signal,
    int signal_length,
    bool use_last,
    T* output_state,
    int batch_count,
    int signal_batch_stride,
    int signal_sample_stride)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_count * state_count;
    if (linear_index >= total) {
        return;
    }
    using Accumulator = float;
    const int batch = linear_index / state_count;
    const int state_index = linear_index % state_count;
    const T* signal_batch = signal + static_cast<long long>(batch) * signal_batch_stride;
    const int sample_index = use_last ? signal_length - 1 : 0;
    const Accumulator scale = static_cast<Accumulator>(filtering_load(
        signal_batch[static_cast<long long>(sample_index) * signal_sample_stride]));
    output_state[static_cast<long long>(batch) * state_count + state_index] =
        filtering_store<T>(
            static_cast<Accumulator>(filtering_load(base_state[state_index])) * scale);
}

template <typename Input, typename Output>
__global__ void convert_signal_kernel(
    const Input* input,
    Output* output,
    std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = filtering_store<Output>(filtering_load(input[index]));
    }
}

template <typename Accumulator>
__device__ __forceinline__ Accumulator sosfilt_apply_section(
    Accumulator sample,
    Accumulator b0,
    Accumulator b1,
    Accumulator b2,
    Accumulator a1,
    Accumulator a2,
    Accumulator& state0,
    Accumulator& state1)
{
    const Accumulator output = b0 * sample + state0;
    const Accumulator next_state0 = b1 * sample - a1 * output + state1;
    const Accumulator next_state1 = b2 * sample - a2 * output;
    state0 = next_state0;
    state1 = next_state1;
    return output;
}

template <typename T>
__global__ void sosfilt_axis_kernel(
    const T* sos,
    int sections,
    const T* input,
    int sample_count,
    int inner_count,
    int outer_count,
    float* state,
    float* output)
{
    const int line = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int line_count = outer_count * inner_count;
    if (line >= line_count) return;

    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long signal_base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
    for (int sample = 0; sample < sample_count; ++sample) {
        const long long signal_index = signal_base
            + static_cast<long long>(sample) * inner_count;
        float value = filtering_load(input[signal_index]);
        for (int section = 0; section < sections; ++section) {
            const T* coefficient = sos + static_cast<long long>(section) * 6;
            const long long state0 =
                ((static_cast<long long>(section) * outer_count + outer) * 2)
                * inner_count + inner;
            const long long state1 = state0 + inner_count;
            value = sosfilt_apply_section(
                value,
                filtering_load(coefficient[0]),
                filtering_load(coefficient[1]),
                filtering_load(coefficient[2]),
                filtering_load(coefficient[4]),
                filtering_load(coefficient[5]),
                state[state0],
                state[state1]);
        }
        output[signal_index] = value;
    }
}

// 正式五类型无状态 FIR 核心，统一使用 FP32 累加和可选初态语义。
template <typename T>
__global__ void firfilter_direct_kernel(
    const T* b,
    int nb,
    const T* x,
    int n_samples,
    const T* zi,
    int zi_batch_stride,
    bool has_zi,
    T* y,
    int batch_count,
    int input_batch_stride,
    int input_sample_stride,
    int output_batch_stride,
    int output_sample_stride)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_count * n_samples;
    if (linear_index >= total) {
        return;
    }

    using Accumulator = float;
    const int batch = linear_index / n_samples;
    const int sample_index = linear_index % n_samples;
    const T* x_batch = x + static_cast<long long>(batch) * input_batch_stride;
    T* y_batch = y + static_cast<long long>(batch) * output_batch_stride;

    Accumulator output = Accumulator{0};
    const int maximum_tap = (nb - 1 < sample_index) ? nb - 1 : sample_index;
    for (int tap = 0; tap <= maximum_tap; ++tap) {
        output += static_cast<Accumulator>(filtering_load(b[tap])) *
            static_cast<Accumulator>(filtering_load(
                x_batch[static_cast<long long>(sample_index - tap) * input_sample_stride]));
    }
    if (has_zi && sample_index < nb - 1) {
        const T* zi_batch = zi + static_cast<long long>(batch) * zi_batch_stride;
        output += static_cast<Accumulator>(filtering_load(zi_batch[sample_index]));
    }
    y_batch[static_cast<long long>(sample_index) * output_sample_stride] =
        filtering_store<T>(output);
}

__device__ inline float firfilter_full_value(
    const float* coefficients,
    int coefficient_count,
    const float* input,
    int sample_count,
    int inner_count,
    int outer,
    int inner,
    int full_index,
    const float* zi,
    const int* zi_line_offsets,
    int zi_state_stride,
    bool has_zi)
{
    const long long input_base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
    float value = 0.0F;
    for (int tap = 0; tap < coefficient_count; ++tap) {
        const int sample = full_index - tap;
        if (sample >= 0 && sample < sample_count)
            value += coefficients[tap]
                * input[input_base + static_cast<long long>(sample) * inner_count];
    }
    const int state_count = coefficient_count - 1;
    if (has_zi && full_index < state_count) {
        const int line = outer * inner_count + inner;
        value += zi[zi_line_offsets[line] + full_index * zi_state_stride];
    }
    return value;
}

static __global__ void firfilter_axis_output_kernel(
    const float* coefficients,
    int coefficient_count,
    const float* input,
    int input_size,
    int sample_count,
    int inner_count,
    const float* zi,
    const int* zi_line_offsets,
    int zi_state_stride,
    bool has_zi,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= input_size) return;
    const int inner = index % inner_count;
    const int sample = (index / inner_count) % sample_count;
    const int outer = index / (sample_count * inner_count);
    output[index] = firfilter_full_value(
        coefficients, coefficient_count, input, sample_count, inner_count,
        outer, inner, sample, zi, zi_line_offsets, zi_state_stride, has_zi);
}

static __global__ void firfilter_axis_state_kernel(
    const float* coefficients,
    int coefficient_count,
    const float* input,
    int sample_count,
    int inner_count,
    const float* zi,
    const int* zi_line_offsets,
    int zi_state_stride,
    float* final_state,
    int final_state_size)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= final_state_size) return;
    const int state_count = coefficient_count - 1;
    const int inner = index % inner_count;
    const int state = (index / inner_count) % state_count;
    const int outer = index / (state_count * inner_count);
    final_state[index] = firfilter_full_value(
        coefficients, coefficient_count, input, sample_count, inner_count,
        outer, inner, sample_count + state, zi, zi_line_offsets,
        zi_state_stride, true);
}

template <typename T>
__global__ void hilbert_pack_axis_kernel(
    const T* input,
    int input_axis_length,
    int fft_length,
    int inner_count,
    int line_count,
    ComplexFloat* packed)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * fft_length;
    if (index >= total) return;
    const int line = index / fft_length;
    const int sample = index % fft_length;
    float value = 0.0F;
    if (sample < input_axis_length) {
        const int outer = line / inner_count;
        const int inner = line % inner_count;
        const long long input_index = static_cast<long long>(outer)
            * input_axis_length * inner_count
            + static_cast<long long>(sample) * inner_count + inner;
        value = SimpleSignalTypePolicy<T>::load(input[input_index]);
    }
    packed[index] = ComplexFloat{value, 0.0F};
}

static __global__ void hilbert_mask_kernel(
    ComplexFloat* spectrum, int fft_length, int total)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) return;
    const int frequency = index % fft_length;
    if (frequency == 0) return;

    const int half = fft_length / 2;
    if ((fft_length % 2) == 0) {
        if (frequency < half) {
            spectrum[index].re *= 2.0F;
            spectrum[index].im *= 2.0F;
        } else if (frequency > half) {
            spectrum[index] = ComplexFloat{0.0F, 0.0F};
        }
    } else if (frequency <= half) {
        spectrum[index].re *= 2.0F;
        spectrum[index].im *= 2.0F;
    } else {
        spectrum[index] = ComplexFloat{0.0F, 0.0F};
    }
}

static __global__ void hilbert_unpack_axis_kernel(
    const ComplexFloat* packed,
    int fft_length,
    int inner_count,
    int line_count,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * fft_length;
    if (index >= total) return;
    const int line = index / fft_length;
    const int sample = index % fft_length;
    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long output_index = static_cast<long long>(outer)
        * fft_length * inner_count
        + static_cast<long long>(sample) * inner_count + inner;
    output[output_index] = packed[index];
}

__device__ inline float hilbert2_mask_weight(int index, int length)
{
    if (index == 0 || ((length % 2) == 0 && index == length / 2))
        return 1.0F;
    return index < (length + 1) / 2 ? 2.0F : 0.0F;
}

static __global__ void hilbert2_mask_kernel(
    ComplexFloat* spectrum, int length)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = length * length;
    if (index >= total) return;
    const int row = index / length;
    const int col = index % length;
    const float scale = hilbert2_mask_weight(row, length)
        * hilbert2_mask_weight(col, length);
    spectrum[index].re *= scale;
    spectrum[index].im *= scale;
}

static __global__ void hilbert2_broadcast_row_mask_kernel(
    const ComplexFloat* row_spectrum,
    int length,
    ComplexFloat* square_spectrum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = length * length;
    if (index >= total) return;
    const int row = index / length;
    const int col = index % length;
    const float scale = hilbert2_mask_weight(row, length)
        * hilbert2_mask_weight(col, length);
    const ComplexFloat value = row_spectrum[col];
    square_spectrum[index] = ComplexFloat{
        value.re * scale, value.im * scale};
}

}  // namespace cusignal::detail
