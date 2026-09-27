#pragma once

// ---- CSD kernels ----

#include "operator_compute_policy.h"
#include <cusignal/backends/fft/fft_interface.h>

#include <cstddef>
#include <type_traits>

namespace cusignal {
namespace spectral_detail {

template <class Output, class Value>
__device__ void csd_store(Output* output, int index, Value real, Value imag)
{
    static_assert(std::is_same_v<Output, ComplexFloat>);
    output[index] = ComplexFloat{
        static_cast<float>(real), static_cast<float>(imag)};
}

template <class Accumulator, class Output>
__global__ void csd_reduce_kernel(
    const ComplexFloat* x_fft,
    const ComplexFloat* y_fft,
    int fft_length,
    int frequency_count,
    int frame_count,
    int line_count,
    int output_inner_count,
    Accumulator scale,
    bool return_onesided,
    Output* output)
{
    const int work_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * frequency_count;
    if (work_index >= total) {
        return;
    }
    const int line = work_index / frequency_count;
    const int frequency = work_index % frequency_count;
    compute_policy::CompensatedFloatAccumulator real_sum_acc;
    compute_policy::CompensatedFloatAccumulator imag_sum_acc;
    for (int frame = 0; frame < frame_count; ++frame) {
        const int fft_index = (line * frame_count + frame) * fft_length + frequency;
        const ComplexFloat x = x_fft[fft_index];
        const ComplexFloat y = y_fft[fft_index];
        Accumulator real = static_cast<Accumulator>(x.re)
                * static_cast<Accumulator>(y.re)
            + static_cast<Accumulator>(x.im)
                * static_cast<Accumulator>(y.im);
        Accumulator imag = static_cast<Accumulator>(x.re)
                * static_cast<Accumulator>(y.im)
            - static_cast<Accumulator>(x.im)
                * static_cast<Accumulator>(y.re);
        if (return_onesided) {
            const bool double_bin = ((fft_length % 2) == 0)
                ? (frequency > 0 && frequency < frequency_count - 1)
                : (frequency > 0);
            if (double_bin) {
                real *= static_cast<Accumulator>(2);
                imag *= static_cast<Accumulator>(2);
            }
        }
        real_sum_acc.add(fmaf(real, scale, 0.0F));
        imag_sum_acc.add(fmaf(imag, scale, 0.0F));
    }
    Accumulator real_sum = static_cast<Accumulator>(real_sum_acc.value());
    Accumulator imag_sum = static_cast<Accumulator>(imag_sum_acc.value());
    if (frame_count > 0) {
        real_sum /= static_cast<Accumulator>(frame_count);
        imag_sum /= static_cast<Accumulator>(frame_count);
    }
    const int outer = line / output_inner_count;
    const int inner = line % output_inner_count;
    const int output_index =
        (outer * frequency_count + frequency) * output_inner_count + inner;
    csd_store(output, output_index, real_sum, imag_sum);
}

__global__ void csd_median_materialize_kernel(
    const ComplexFloat* x_fft,
    const ComplexFloat* y_fft,
    int fft_length,
    int frequency_count,
    int frame_count,
    int line_count,
    int output_inner_count,
    float scale,
    bool return_onesided,
    float* real_values,
    float* imag_values)
{
    const int work_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_count = line_count * frequency_count;
    const int total = output_count * frame_count;
    if (work_index >= total) {
        return;
    }
    const int frame = work_index % frame_count;
    const int output_index = work_index / frame_count;
    const int inner = output_index % output_inner_count;
    const int frequency = (output_index / output_inner_count) % frequency_count;
    const int outer = output_index / (frequency_count * output_inner_count);
    const int line = outer * output_inner_count + inner;
    const int fft_index = (line * frame_count + frame) * fft_length + frequency;
    const ComplexFloat x = x_fft[fft_index];
    const ComplexFloat y = y_fft[fft_index];
    float frequency_scale = scale;
    if (return_onesided) {
        const bool double_bin = ((fft_length % 2) == 0)
            ? (frequency > 0 && frequency < frequency_count - 1)
            : (frequency > 0);
        if (double_bin) {
            frequency_scale *= 2.0F;
        }
    }
    const float real = fmaf(x.re, y.re, x.im * y.im);
    const float imag = fmaf(x.re, y.im, -x.im * y.re);
    real_values[work_index] = fmaf(real, frequency_scale, 0.0F);
    imag_values[work_index] = fmaf(imag, frequency_scale, 0.0F);
}

__device__ float csd_quickselect(float* values, int count, int kth)
{
    int left = 0;
    int right = count - 1;
    while (left < right) {
        const float pivot = values[left + (right - left) / 2];
        int low = left;
        int high = right;
        while (low <= high) {
            while (values[low] < pivot) ++low;
            while (values[high] > pivot) --high;
            if (low <= high) {
                const float temporary = values[low];
                values[low] = values[high];
                values[high] = temporary;
                ++low;
                --high;
            }
        }
        if (kth <= high) {
            right = high;
        } else if (kth >= low) {
            left = low;
        } else {
            break;
        }
    }
    return values[kth];
}

__device__ float csd_median_in_place(float* values, int count)
{
    const int middle = count / 2;
    const float upper = csd_quickselect(values, count, middle);
    if ((count % 2) != 0) {
        return upper;
    }
    float lower = values[0];
    for (int index = 1; index < middle; ++index) {
        if (values[index] > lower) {
            lower = values[index];
        }
    }
    return 0.5F * (lower + upper);
}

__global__ void csd_median_reduce_kernel(
    float* real_values,
    float* imag_values,
    int frame_count,
    int output_count,
    float bias,
    ComplexFloat* output)
{
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_count) {
        return;
    }
    float* real = real_values + static_cast<std::size_t>(output_index) * frame_count;
    float* imag = imag_values + static_cast<std::size_t>(output_index) * frame_count;
    output[output_index] = ComplexFloat{
        csd_median_in_place(real, frame_count) / bias,
        csd_median_in_place(imag, frame_count) / bias};
}

} // namespace spectral_detail
} // namespace cusignal

// ---- ISTFT kernels ----

#include "simple_signal_typed.h"
#include <cusignal/backends/fft/fft_interface.h>

#include <cmath>
#include <type_traits>

namespace cusignal {
namespace spectral_detail {

template <class T>
__device__ ComplexFloat istft_load_complex(T value)
{
    return ComplexFloat{
        detail::SimpleSignalTypePolicy<T>::load(value), 0.0F};
}

template <class T>
__device__ ComplexFloat istft_load_complex(TypedComplex<T> value)
{
    return ComplexFloat{
        detail::SimpleSignalTypePolicy<T>::load(value.real),
        detail::SimpleSignalTypePolicy<T>::load(value.imag)};
}

template <class Input>
__global__ void istft_build_spectrum_kernel(
    const Input* input,
    int rows,
    int cols,
    int frequency_count,
    int frame_count,
    int fft_length,
    bool input_onesided,
    bool transposed_input,
    ComplexFloat* spectrum)
{
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = frame_count * fft_length;
    if (output_index >= total) {
        return;
    }
    const int frame = output_index / fft_length;
    const int frequency = output_index % fft_length;
    ComplexFloat output{0.0F, 0.0F};
    if (input_onesided) {
        const int copy_count = min(frequency_count, fft_length / 2 + 1);
        if (frequency < copy_count) {
            const int offset = transposed_input
                ? frame * cols + frequency
                : frequency * cols + frame;
            output = istft_load_complex(input[offset]);
        } else {
            const int source = fft_length - frequency;
            const int maximum_source = ((fft_length % 2) == 0)
                ? copy_count - 2
                : copy_count - 1;
            if (source >= 1 && source <= maximum_source) {
                const int offset = transposed_input
                    ? frame * cols + source
                    : source * cols + frame;
                output = istft_load_complex(input[offset]);
                output.im = -output.im;
            }
        }
    } else if (frequency < frequency_count) {
        const int offset = transposed_input
            ? frame * cols + frequency
            : frequency * cols + frame;
        output = istft_load_complex(input[offset]);
    }
    spectrum[output_index] = output;
}

template <class Output, class Accumulator>
__device__ Output istft_store(Accumulator value)
{
    if constexpr (std::is_integral_v<Output>) {
        const Output truncated = detail::SimpleSignalTypePolicy<Output>::store(
            static_cast<float>(value));
        const float base = detail::SimpleSignalTypePolicy<Output>::load(truncated);
        const float adjacent = base + (value >= static_cast<Accumulator>(0) ? 1.0F : -1.0F);
        return fabsf(static_cast<float>(value) - adjacent) < 1.0e-4F
            ? detail::SimpleSignalTypePolicy<Output>::store(adjacent)
            : truncated;
    } else {
        return detail::SimpleSignalTypePolicy<Output>::store(static_cast<float>(value));
    }
}

template <class Window, class Accumulator, class Output>
__global__ void istft_finalize_output_kernel(
    const ComplexFloat* segments,
    const Window* window,
    int segment_length,
    int overlap,
    int fft_length,
    int frame_count,
    bool input_onesided,
    bool boundary,
    Accumulator window_sum,
    int expected_signal_length,
    Output* output)
{
    const int half_segment = segment_length / 2;
    const int trim = boundary ? half_segment : 0;
    const int final_length = expected_signal_length - 2 * trim;
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= final_length) {
        return;
    }
    const int signal_index = output_index + trim;
    static_assert(std::is_same_v<Accumulator, float>);
    compute_policy::FmaFloatAccumulator reconstructed_acc;
    compute_policy::FmaFloatAccumulator normalization_acc;
    (void)input_onesided;
    const int step = segment_length - overlap;
    for (int frame = 0; frame < frame_count; ++frame) {
        const int start = frame * step;
        if (signal_index >= start && signal_index < start + segment_length) {
            const int offset = signal_index - start;
            const Accumulator weight = static_cast<Accumulator>(window[offset]);
            const Accumulator weighted_segment = static_cast<Accumulator>(
                segments[frame * fft_length + offset].re) * window_sum;
            reconstructed_acc.add_product(weighted_segment, weight);
            normalization_acc.add_product(weight, weight);
        }
    }
    const Accumulator reconstructed = reconstructed_acc.value();
    const Accumulator normalization = normalization_acc.value();
    const Accumulator epsilon = static_cast<Accumulator>(1.0e-10);
    output[output_index] = istft_store<Output>(normalization > epsilon
        ? reconstructed / normalization
        : static_cast<Accumulator>(0));
}

} // namespace spectral_detail
} // namespace cusignal

// ---- Lomb-Scargle kernels ----

#include "simple_signal_typed.h"

#include <cmath>
#include <type_traits>

namespace cusignal {
namespace spectral_detail {

template <class Accumulator, class T>
__device__ Accumulator lombscargle_load(T value)
{
    return static_cast<Accumulator>(detail::SimpleSignalTypePolicy<T>::load(value));
}

template <class Input, class Frequency, class Accumulator, class Output>
__global__ void lombscargle_kernel(
    const Input* time,
    const Input* signal,
    int sample_count,
    const Frequency* frequencies,
    int frequency_count,
    bool precenter,
    bool normalize,
    Output* out)
{
    const int frequency_index = static_cast<int>(blockIdx.x);
    if (frequency_index >= frequency_count || threadIdx.x != 0) {
        return;
    }

    const Accumulator omega = lombscargle_load<Accumulator>(frequencies[frequency_index]);
    static_assert(std::is_same_v<Accumulator, float>);
    Accumulator signal_mean = static_cast<Accumulator>(0);
    if (precenter) {
        compute_policy::CompensatedFloatAccumulator signal_sum;
        for (int index = 0; index < sample_count; ++index) {
            signal_sum.add(lombscargle_load<Accumulator>(signal[index]));
        }
        signal_mean = signal_sum.value() / static_cast<Accumulator>(sample_count);
    }
    Accumulator y_dot = static_cast<Accumulator>(0);
    if (normalize) {
        compute_policy::CompensatedFloatAccumulator y_dot_acc;
        for (int index = 0; index < sample_count; ++index) {
            const Accumulator value = lombscargle_load<Accumulator>(signal[index]);
            y_dot_acc.add_product(value, value);
        }
        y_dot = y_dot_acc.value();
    }
    const Accumulator normalization = y_dot == static_cast<Accumulator>(0)
        ? static_cast<Accumulator>(1)
        : static_cast<Accumulator>(2) / y_dot;
    compute_policy::CompensatedFloatAccumulator xc_acc;
    compute_policy::CompensatedFloatAccumulator xs_acc;
    compute_policy::CompensatedFloatAccumulator cc_acc;
    compute_policy::CompensatedFloatAccumulator ss_acc;
    compute_policy::CompensatedFloatAccumulator cs_acc;
    for (int index = 0; index < sample_count; ++index) {
        const Accumulator phase = omega * lombscargle_load<Accumulator>(time[index]);
        const Accumulator cosine = cos(phase);
        const Accumulator sine = sin(phase);
        const Accumulator centered = lombscargle_load<Accumulator>(signal[index]) - signal_mean;
        xc_acc.add_product(centered, cosine);
        xs_acc.add_product(centered, sine);
        cc_acc.add_product(cosine, cosine);
        ss_acc.add_product(sine, sine);
        cs_acc.add_product(cosine, sine);
    }
    const Accumulator xc = xc_acc.value();
    const Accumulator xs = xs_acc.value();
    const Accumulator cc = cc_acc.value();
    const Accumulator ss = ss_acc.value();
    const Accumulator cs = cs_acc.value();
    const Accumulator tau = atan2(static_cast<Accumulator>(2) * cs, cc - ss) /
        (static_cast<Accumulator>(2) * omega);
    const Accumulator c_tau = cos(omega * tau);
    const Accumulator s_tau = sin(omega * tau);
    const Accumulator c_tau2 = c_tau * c_tau;
    const Accumulator s_tau2 = s_tau * s_tau;
    const Accumulator cs_tau = static_cast<Accumulator>(2) * c_tau * s_tau;
    const Accumulator first = c_tau * xc + s_tau * xs;
    const Accumulator second = c_tau * xs - s_tau * xc;
    out[frequency_index] = static_cast<Output>(
        static_cast<Accumulator>(0.5) *
        (first * first / (c_tau2 * cc + cs_tau * cs + s_tau2 * ss) +
         second * second / (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)) *
        normalization);
}

} // namespace spectral_detail
} // namespace cusignal

// ---- Spectrogram kernels ----

#include <cusignal/backends/fft/fft_interface.h>

#include <cmath>

namespace cusignal {
namespace spectral_detail {

template <class Accumulator, class Output>
__global__ void spectrogram_real_post_kernel(
    const ComplexFloat* fft_output,
    int fft_length,
    int frequency_count,
    int frame_count,
    int line_count,
    int inner_count,
    Accumulator scale,
    bool return_onesided,
    int mode,
    Output* output)
{
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * frequency_count * frame_count;
    if (output_index >= total) {
        return;
    }
    const int frame = output_index % frame_count;
    const int frequency = (output_index / frame_count) % frequency_count;
    const int packed_line = output_index / (frame_count * frequency_count);
    const int outer = packed_line / inner_count;
    const int inner = packed_line % inner_count;
    const int line = outer * inner_count + inner;
    const ComplexFloat value = fft_output[
        (line * frame_count + frame) * fft_length + frequency];
    const Accumulator real = static_cast<Accumulator>(value.re);
    const Accumulator imag = static_cast<Accumulator>(value.im);
    Accumulator result = static_cast<Accumulator>(0);
    if (mode == 0) {
        result = (real * real + imag * imag) * scale;
        if (return_onesided) {
            const bool double_bin = ((fft_length % 2) == 0)
                ? (frequency > 0 && frequency < frequency_count - 1)
                : (frequency > 0);
            if (double_bin) {
                result *= static_cast<Accumulator>(2);
            }
        }
    } else if (mode == 1) {
        result = sqrt(real * real + imag * imag);
    } else {
        result = atan2(imag, real);
    }
    const int destination =
        ((outer * frequency_count + frequency) * inner_count + inner)
            * frame_count + frame;
    output[destination] = static_cast<Output>(result);
}

__global__ void spectrogram_phase_unwrap_kernel(
    float* phase,
    int frequency_count,
    int frame_count,
    int line_count,
    int inner_count)
{
    const int line_time = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * frame_count;
    if (line_time >= total || frequency_count <= 1) {
        return;
    }
    const int time = line_time % frame_count;
    const int packed_line = line_time / frame_count;
    const int outer = packed_line / inner_count;
    const int inner = packed_line % inner_count;
    constexpr float pi = 3.14159265358979323846F;
    constexpr float two_pi = 6.28318530717958647692F;
    const int first = (outer * frequency_count * inner_count + inner)
        * frame_count + time;
    float previous_raw = phase[first];
    float correction = 0.0F;
    for (int frequency = 1; frequency < frequency_count; ++frequency) {
        const int index = ((outer * frequency_count + frequency)
            * inner_count + inner) * frame_count + time;
        const float current_raw = phase[index];
        const float delta = current_raw - previous_raw;
        if (delta > pi) {
            correction -= two_pi;
        } else if (delta < -pi) {
            correction += two_pi;
        }
        phase[index] = current_raw + correction;
        previous_raw = current_raw;
    }
}

} // namespace spectral_detail
} // namespace cusignal

// ---- STFT kernels ----

#include "simple_signal_typed.h"
#include <cusignal/backends/fft/fft_interface.h>

#include <cuComplex.h>
#include <type_traits>

namespace cusignal {
namespace spectral_detail {

template <class Accumulator, class T>
__device__ Accumulator stft_load(T value)
{
    return static_cast<Accumulator>(detail::SimpleSignalTypePolicy<T>::load(value));
}

template <class Accumulator, class Input>
__device__ Accumulator stft_real_sample_at(
    const Input* input, int length, int source_index, int boundary_mode)
{
    if (source_index >= 0 && source_index < length) {
        return stft_load<Accumulator>(input[source_index]);
    }
    if (boundary_mode == 0) {
        return static_cast<Accumulator>(0);
    }
    if (boundary_mode == 1) {
        return stft_load<Accumulator>(input[source_index < 0 ? 0 : length - 1]);
    }
    const int reflected = source_index < 0
        ? -source_index
        : 2 * length - 2 - source_index;
    const Accumulator reflected_value = stft_load<Accumulator>(input[reflected]);
    if (boundary_mode == 2) {
        return reflected_value;
    }
    const Accumulator edge = stft_load<Accumulator>(
        input[source_index < 0 ? 0 : length - 1]);
    return static_cast<Accumulator>(2) * edge - reflected_value;
}

template <class Accumulator, class Input>
__device__ Accumulator stft_axis_sample_at(
    const Input* input, int line_base, int axis_length, int axis_stride,
    int source_index, int boundary_mode)
{
    if (source_index >= 0 && source_index < axis_length) {
        return stft_load<Accumulator>(
            input[line_base + source_index * axis_stride]);
    }
    if (boundary_mode == 0) return static_cast<Accumulator>(0);
    if (boundary_mode == 1) {
        const int sample = source_index < 0 ? 0 : axis_length - 1;
        return stft_load<Accumulator>(input[line_base + sample * axis_stride]);
    }
    const int reflected = source_index < 0
        ? -source_index : 2 * axis_length - 2 - source_index;
    const Accumulator reflected_value = stft_load<Accumulator>(
        input[line_base + reflected * axis_stride]);
    if (boundary_mode == 2) return reflected_value;
    const int edge_sample = source_index < 0 ? 0 : axis_length - 1;
    const Accumulator edge = stft_load<Accumulator>(
        input[line_base + edge_sample * axis_stride]);
    return static_cast<Accumulator>(2) * edge - reflected_value;
}

template <class Input, class Window, class Accumulator>
__global__ void stft_prepare_real_kernel(
    const Input* input,
    const int* line_bases,
    int axis_length,
    int axis_stride,
    int line_count,
    const Window* window,
    int segment_length,
    int fft_length,
    int step,
    int frame_count,
    int boundary_left,
    int boundary_mode,
    bool detrend_constant,
    Accumulator scale,
    ComplexFloat* fft_input)
{
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * frame_count * fft_length;
    if (output_index >= total) {
        return;
    }
    const int batch = output_index / fft_length;
    const int line = batch / frame_count;
    const int frame = batch % frame_count;
    const int local_index = output_index % fft_length;
    const int start = frame * step - boundary_left;
    if (local_index >= segment_length) {
        fft_input[output_index] = ComplexFloat{0.0F, 0.0F};
        return;
    }

    Accumulator mean = static_cast<Accumulator>(0);
    if (detrend_constant) {
        static_assert(std::is_same_v<Accumulator, float>);
        compute_policy::CompensatedFloatAccumulator mean_acc;
        for (int index = 0; index < segment_length; ++index) {
            mean_acc.add(stft_axis_sample_at<Accumulator>(
                input, line_bases[line], axis_length, axis_stride,
                start + index, boundary_mode));
        }
        mean = mean_acc.value() / static_cast<Accumulator>(segment_length);
    }
    const Accumulator value = (
        stft_axis_sample_at<Accumulator>(
            input, line_bases[line], axis_length, axis_stride,
            start + local_index, boundary_mode) - mean)
        * stft_load<Accumulator>(window[local_index]) * scale;
    fft_input[output_index] = ComplexFloat{static_cast<float>(value), 0.0F};
}

template <class Output>
__device__ void stft_store_complex(Output* output, int index, ComplexFloat value)
{
    static_assert(std::is_same_v<Output, ComplexFloat>);
    output[index] = value;
}

template <class Output>
__global__ void stft_pack_output_kernel(
    const ComplexFloat* fft_output,
    int fft_length,
    int frequency_count,
    int frame_count,
    int line_count,
    int inner_count,
    Output* output)
{
    const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = line_count * frequency_count * frame_count;
    if (output_index >= total) {
        return;
    }
    const int frame = output_index % frame_count;
    const int frequency = (output_index / frame_count) % frequency_count;
    const int line = output_index / (frame_count * frequency_count);
    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const int destination =
        ((outer * frequency_count + frequency) * inner_count + inner)
            * frame_count + frame;
    stft_store_complex(
        output, destination,
        fft_output[(line * frame_count + frame) * fft_length + frequency]);
}

} // namespace spectral_detail
} // namespace cusignal

// ---- Vector strength kernels ----

#include "simple_signal_typed.h"

#include <cmath>
#include <type_traits>

namespace cusignal {
namespace spectral_detail {

template <class Accumulator, class T>
__device__ Accumulator vectorstrength_load(T value)
{
    return static_cast<Accumulator>(detail::SimpleSignalTypePolicy<T>::load(value));
}

template <class Input, class Period, class Accumulator, class Output>
__global__ void vectorstrength_single_kernel(
    const Input* events,
    int event_count,
    Period period,
    Output* out)
{
    __shared__ Accumulator sine_sums[256];
    __shared__ Accumulator cosine_sums[256];
    const int thread_index = threadIdx.x;
    const Accumulator two_pi = static_cast<Accumulator>(6.28318530717958647692);
    const Accumulator omega = two_pi
        / vectorstrength_load<Accumulator>(period);
    static_assert(std::is_same_v<Accumulator, float>);
    compute_policy::CompensatedFloatAccumulator sine_sum_acc;
    compute_policy::CompensatedFloatAccumulator cosine_sum_acc;
    for (int index = thread_index; index < event_count; index += blockDim.x) {
        const Accumulator angle = vectorstrength_load<Accumulator>(events[index]) * omega;
        sine_sum_acc.add(sin(angle));
        cosine_sum_acc.add(cos(angle));
    }
    sine_sums[thread_index] = sine_sum_acc.value();
    cosine_sums[thread_index] = cosine_sum_acc.value();
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            sine_sums[thread_index] += sine_sums[thread_index + stride];
            cosine_sums[thread_index] += cosine_sums[thread_index + stride];
        }
        __syncthreads();
    }
    if (thread_index == 0) {
        const Accumulator magnitude = sqrt(
            sine_sums[0] * sine_sums[0]
            + cosine_sums[0] * cosine_sums[0]);
        out[0] = static_cast<Output>(event_count > 0
            ? magnitude / static_cast<Accumulator>(event_count)
            : static_cast<Accumulator>(0));
    }
}

template <class Input, class Period, class Accumulator, class Output>
__global__ void vectorstrength_multi_kernel(
    const Input* events,
    int event_count,
    const Period* periods,
    int period_count,
    Output* strength,
    Output* phase)
{
    __shared__ Accumulator sine_sums[256];
    __shared__ Accumulator cosine_sums[256];
    const int period_index = static_cast<int>(blockIdx.x);
    const int thread_index = threadIdx.x;
    if (period_index >= period_count) {
        return;
    }
    const Accumulator two_pi = static_cast<Accumulator>(6.28318530717958647692);
    const Accumulator omega = two_pi
        / vectorstrength_load<Accumulator>(periods[period_index]);
    static_assert(std::is_same_v<Accumulator, float>);
    compute_policy::CompensatedFloatAccumulator sine_sum_acc;
    compute_policy::CompensatedFloatAccumulator cosine_sum_acc;
    for (int index = thread_index; index < event_count; index += blockDim.x) {
        const Accumulator angle = vectorstrength_load<Accumulator>(events[index]) * omega;
        sine_sum_acc.add(sin(angle));
        cosine_sum_acc.add(cos(angle));
    }
    sine_sums[thread_index] = sine_sum_acc.value();
    cosine_sums[thread_index] = cosine_sum_acc.value();
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            sine_sums[thread_index] += sine_sums[thread_index + stride];
            cosine_sums[thread_index] += cosine_sums[thread_index + stride];
        }
        __syncthreads();
    }
    if (thread_index == 0) {
        if (event_count == 0) {
            strength[period_index] = static_cast<Output>(nanf(""));
            phase[period_index] = static_cast<Output>(nanf(""));
            return;
        }
        const Accumulator magnitude = sqrt(
            sine_sums[0] * sine_sums[0]
            + cosine_sums[0] * cosine_sums[0]);
        strength[period_index] = static_cast<Output>(
            magnitude / static_cast<Accumulator>(event_count));
        phase[period_index] = static_cast<Output>(
            atan2(sine_sums[0], cosine_sums[0]));
    }
}

} // namespace spectral_detail
} // namespace cusignal
