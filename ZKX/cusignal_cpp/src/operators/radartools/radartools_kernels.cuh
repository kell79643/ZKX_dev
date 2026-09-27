#pragma once

#include <cusignal/operators/filtering/filtering_typed.h>
#include "simple_signal_typed.h"

#include <limits>
#include <type_traits>

namespace cusignal::radar_detail {

constexpr float radar_pi = 3.14159265358979323846F;

template <typename T>
__host__ __device__ inline auto radar_load(T value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::load(value);
    } else {
        return value;
    }
}

template <typename Input>
__device__ inline bool ca_cfar_detect(Input input, float threshold)
{
    if (isnan(threshold)) return false;
    if constexpr (std::is_integral_v<Input>) {
        if (isinf(threshold)) return threshold < 0.0F;
        const float low = static_cast<float>(std::numeric_limits<Input>::lowest());
        const float high = static_cast<float>(std::numeric_limits<Input>::max());
        if (threshold < low) return true;
        if (threshold >= high) return false;
        return input > static_cast<Input>(floorf(threshold));
    } else {
        return static_cast<float>(radar_load(input)) > threshold;
    }
}

template <typename Input>
__global__ void ca_cfar_column_cumsum_kernel(
    const Input* input, int rows, int columns, float* cumulative)
{
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= columns) return;
    float running = 0.0F;
    for (int row = 0; row < rows; ++row) {
        const int index = row * columns + column;
        running += static_cast<float>(radar_load(input[index]));
        cumulative[index] = running;
    }
}

__global__ void ca_cfar_row_cumsum_kernel(
    float* cumulative, int rows, int columns)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    float running = 0.0F;
    for (int column = 0; column < columns; ++column) {
        const int index = row * columns + column;
        running += cumulative[index];
        cumulative[index] = running;
    }
}

template <typename Input>
__global__ void ca_cfar_integral_threshold_kernel(
    const Input* input,
    const float* cumulative,
    int count,
    int rank,
    int rows,
    int columns,
    int guard_rows,
    int guard_columns,
    int reference_rows,
    int reference_columns,
    float alpha,
    float* threshold_output,
    CfarDetection* detections)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    float threshold = 0.0F;
    if (rank == 1) {
        const int margin = guard_rows + reference_rows;
        if (index >= margin && index < rows - margin) {
            const int outer_right = index + margin;
            const int outer_left = index - margin - 1;
            const int inner_right = index + guard_rows;
            const int inner_left = index - guard_rows - 1;
            const float outer = outer_left >= 0
                ? cumulative[outer_right] - cumulative[outer_left]
                : cumulative[outer_right];
            const float inner = cumulative[inner_right] - cumulative[inner_left];
            threshold = (alpha / static_cast<float>(2 * reference_rows)) *
                (outer - inner);
        }
    } else if (rank == 2) {
        const int row = index / columns;
        const int column = index % columns;
        const int row_margin = guard_rows + reference_rows;
        const int column_margin = guard_columns + reference_columns;
        if (row >= row_margin && row < rows - row_margin &&
            column >= column_margin && column < columns - column_margin) {
            const int outer_top = row + row_margin;
            const int outer_left = row - row_margin - 1;
            const int outer_right_column = column + column_margin;
            const int outer_left_column = column - column_margin - 1;
            float outer = cumulative[outer_top * columns + outer_right_column];
            if (outer_left >= 0)
                outer -= cumulative[outer_left * columns + outer_right_column];
            if (outer_left_column >= 0)
                outer -= cumulative[outer_top * columns + outer_left_column];
            if (outer_left >= 0 && outer_left_column >= 0)
                outer += cumulative[outer_left * columns + outer_left_column];
            const float inner =
                cumulative[(row + guard_rows) * columns + column + guard_columns] -
                cumulative[(row - guard_rows - 1) * columns + column + guard_columns] -
                cumulative[(row + guard_rows) * columns + column - guard_columns - 1] +
                cumulative[(row - guard_rows - 1) * columns +
                    column - guard_columns - 1];
            const int reference_count =
                2 * reference_rows *
                    (2 * reference_columns + 2 * guard_columns + 1) +
                2 * (2 * guard_rows + 1) * reference_columns;
            threshold = (alpha / static_cast<float>(reference_count)) *
                (outer - inner);
        }
    }
    threshold_output[index] = threshold;
    detections[index] = ca_cfar_detect(input[index], threshold)
        ? CfarDetection::yes : CfarDetection::no;
}

template <typename Input>
__global__ void cfar_alpha_kernel(
    Input pfa,
    int reference_count,
    float* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const float probability = static_cast<float>(radar_load(pfa));
    output[0] = static_cast<float>(reference_count) *
        (powf(probability, -1.0F / static_cast<float>(reference_count)) - 1.0F);
}

__device__ inline float radar_window_value(
    int kind, int index, int length, const float* explicit_window)
{
    if (kind == 3 || kind == 4) return explicit_window[index];
    if (length <= 1 || kind == 0) return 1.0F;
    const float phase = 6.28318530717958647692F * static_cast<float>(index) /
        static_cast<float>(length - 1);
    if (kind == 1) return 0.5F - 0.5F * cosf(phase);
    if (kind == 2) return 0.54F - 0.46F * cosf(phase);
    return 1.0F;
}

template <typename T>
__global__ void pulse_compression_norm_kernel(
    const TypedComplex<T>* input,
    int count,
    int window_kind,
    const float* explicit_window,
    float* norm)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = 0; index < count; ++index) {
        const float window = radar_window_value(
            window_kind, index, count, explicit_window);
        const float real = static_cast<float>(radar_load(input[index].real)) * window;
        const float imag = static_cast<float>(radar_load(input[index].imag)) * window;
        sum.add(fmaf(real, real, imag * imag));
    }
    norm[0] = sqrtf(sum.value());
}

template <typename T>
__global__ void pulse_compression_prepare_template_kernel(
    const TypedComplex<T>* input,
    int input_count,
    int fft_length,
    bool normalize,
    int window_kind,
    const float* explicit_window,
    const float* norm,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= fft_length) {
        return;
    }
    float real = 0.0F;
    float imag = 0.0F;
    if (index < input_count) {
        float scale = 1.0F;
        if (normalize) {
            scale = 1.0F / norm[0];
        }
        const float window = radar_window_value(
            window_kind, index, input_count, explicit_window);
        real = static_cast<float>(radar_load(input[index].real)) * window * scale;
        imag = static_cast<float>(radar_load(input[index].imag)) * window * scale;
    }
    output[index] = ComplexFloat{real, imag};
}

template <typename T>
__global__ void pulse_compression_pack_input_kernel(
    const TypedComplex<T>* input,
    int pulse_count,
    int samples_per_pulse,
    int fft_length,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = pulse_count * fft_length;
    if (index >= total) {
        return;
    }
    const int pulse = index / fft_length;
    const int sample = index % fft_length;
    if (sample < samples_per_pulse) {
        const int source = pulse * samples_per_pulse + sample;
        output[index] = ComplexFloat{
            static_cast<float>(radar_load(input[source].real)),
            static_cast<float>(radar_load(input[source].imag))};
    } else {
        output[index] = ComplexFloat{0.0F, 0.0F};
    }
}

static __global__ void pulse_compression_multiply_kernel(
    const ComplexFloat* input_frequency,
    const ComplexFloat* template_frequency,
    int count,
    int fft_length,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    const ComplexFloat input = input_frequency[index];
    const ComplexFloat templ = template_frequency[index % fft_length];
    output[index] = ComplexFloat{
        input.re * templ.re + input.im * templ.im,
        input.im * templ.re - input.re * templ.im};
}

__device__ inline bool ambiguity_norm_update(
    float value, float& scale, float& scaled_sum)
{
    value = fabsf(value);
    if (isnan(value)) {
        scale = value;
        return false;
    }
    if (isinf(value)) {
        scale = value;
        scaled_sum = 1.0F;
        return true;
    }
    if (value == 0.0F || isinf(scale)) return true;
    if (scale < value) {
        const float ratio = scale / value;
        scaled_sum = 1.0F + scaled_sum * ratio * ratio;
        scale = value;
    } else {
        const float ratio = value / scale;
        scaled_sum += ratio * ratio;
    }
    return true;
}

template <typename T>
__global__ void ambiguity_norm_kernel(
    const TypedComplex<T>* input,
    int count,
    float* norm)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float scale = 0.0F;
    float scaled_sum = 1.0F;
    for (int index = 0; index < count; ++index) {
        if (!ambiguity_norm_update(
                static_cast<float>(radar_load(input[index].real)), scale, scaled_sum) ||
            !ambiguity_norm_update(
                static_cast<float>(radar_load(input[index].imag)), scale, scaled_sum)) {
            norm[0] = scale;
            return;
        }
    }
    norm[0] = scale == 0.0F ? 0.0F : scale * sqrtf(scaled_sum);
}

template <typename T>
__global__ void ambiguity_build_delay_kernel(
    const TypedComplex<T>* x,
    int x_count,
    const TypedComplex<T>* y,
    int y_count,
    int row_count,
    int fft_length,
    const float* x_norm,
    const float* y_norm,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = row_count * fft_length;
    if (index >= total) {
        return;
    }
    const int row = index / fft_length;
    const int column = index % fft_length;
    float real = 0.0F;
    float imag = 0.0F;
    const int x_column = column - (x_count - 1) + row;
    const float denominator = x_norm[0] * y_norm[0];
    if (column < y_count && x_column >= 0 && x_column < x_count) {
        const float y_real = static_cast<float>(radar_load(y[column].real));
        const float y_imag = static_cast<float>(radar_load(y[column].imag));
        const float x_real = static_cast<float>(radar_load(x[x_column].real));
        const float x_imag = static_cast<float>(radar_load(x[x_column].imag));
        real = (y_real * x_real + y_imag * x_imag) / denominator;
        imag = (y_imag * x_real - y_real * x_imag) / denominator;
    }
    output[index] = ComplexFloat{real, imag};
}

template <typename T>
__global__ void ambiguity_pack_normalized_kernel(
    const TypedComplex<T>* input,
    int input_count,
    int output_count,
    const float* norm,
    float modulation_frequency,
    float sample_rate,
    bool modulate,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (index >= input_count) {
        output[index] = ComplexFloat{0.0F, 0.0F};
        return;
    }
    float real = static_cast<float>(radar_load(input[index].real)) / norm[0];
    float imag = static_cast<float>(radar_load(input[index].imag)) / norm[0];
    if (modulate) {
        const float angle = 2.0F * radar_pi * modulation_frequency *
            static_cast<float>(index) / sample_rate;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float rotated_real = real * cosine - imag * sine;
        imag = real * sine + imag * cosine;
        real = rotated_real;
    }
    output[index] = ComplexFloat{real, imag};
}

__global__ void ambiguity_delay_phase_kernel(
    ComplexFloat* spectrum,
    int count,
    float sample_rate,
    float cut_value)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float frequency = -0.5F * sample_rate +
        sample_rate * static_cast<float>(index) / static_cast<float>(count);
    const float angle = 2.0F * radar_pi * frequency * cut_value;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const ComplexFloat value = spectrum[index];
    spectrum[index] = ComplexFloat{
        value.re * cosine - value.im * sine,
        value.re * sine + value.im * cosine};
}

template <typename T>
__global__ void ambiguity_delay_product_kernel(
    const TypedComplex<T>* y,
    int y_count,
    const float* y_norm,
    const ComplexFloat* shifted_x,
    int count,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    if (index >= y_count) {
        output[index] = ComplexFloat{0.0F, 0.0F};
        return;
    }
    const float y_real = static_cast<float>(radar_load(y[index].real)) / y_norm[0];
    const float y_imag = static_cast<float>(radar_load(y[index].imag)) / y_norm[0];
    const ComplexFloat shifted = shifted_x[index];
    output[index] = ComplexFloat{
        y_real * shifted.re + y_imag * shifted.im,
        y_imag * shifted.re - y_real * shifted.im};
}

__global__ void ambiguity_frequency_product_kernel(
    const ComplexFloat* left,
    const ComplexFloat* right,
    int count,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const ComplexFloat a = left[index];
    const ComplexFloat b = right[index];
    output[index] = ComplexFloat{
        a.re * b.re + a.im * b.im,
        a.im * b.re - a.re * b.im};
}

__global__ void ambiguity_cut_magnitude_kernel(
    const ComplexFloat* input,
    int count,
    bool inverse_shift,
    float scale,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const int source = inverse_shift
        ? (index + count / 2) % count
        : (index - count / 2 + count) % count;
    const ComplexFloat value = input[source];
    output[index] = scale * hypotf(value.re, value.im);
}

static __global__ void ambiguity_magnitude_shift_kernel(
    const ComplexFloat* input,
    int row_count,
    int fft_length,
    float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = row_count * fft_length;
    if (index >= total) {
        return;
    }
    const int row = index / fft_length;
    const int column = index % fft_length;
    const int shifted_column = (column + fft_length / 2) % fft_length;
    const ComplexFloat value = input[index];
    output[row * fft_length + shifted_column] =
        static_cast<float>(fft_length) * hypotf(value.re, value.im);
}

template <typename T>
__global__ void pulse_doppler_prepare_input_kernel(
    const TypedComplex<T>* input,
    int pulse_count,
    int range_bin_count,
    int fft_length,
    int window_kind,
    const float* explicit_window,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = range_bin_count * fft_length;
    if (index >= total) {
        return;
    }
    const int range_bin = index / fft_length;
    const int pulse = index % fft_length;
    if (pulse >= pulse_count) {
        output[index] = ComplexFloat{0.0F, 0.0F};
        return;
    }
    const float window = radar_window_value(
        window_kind, pulse, pulse_count, explicit_window);
    const TypedComplex<T> value = input[pulse * range_bin_count + range_bin];
    output[index] = ComplexFloat{
        static_cast<float>(radar_load(value.real)) * window,
        static_cast<float>(radar_load(value.imag)) * window};
}

static __global__ void pulse_doppler_transpose_output_kernel(
    const ComplexFloat* input,
    int range_bin_count,
    int fft_length,
    ComplexFloat* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = range_bin_count * fft_length;
    if (index >= total) {
        return;
    }
    const int doppler = index / range_bin_count;
    const int range_bin = index % range_bin_count;
    output[index] = input[range_bin * fft_length + doppler];
}

}  // namespace cusignal::radar_detail
