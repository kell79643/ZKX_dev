#pragma once

#include "simple_signal_typed.h"

#include <cmath>
#include <type_traits>
#include <vector>

namespace cusignal::windows_detail {

template <typename T>
__host__ __device__ inline auto window_load(T value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::load(value);
    } else {
        return value;
    }
}

template <typename T, typename Scalar>
__host__ __device__ inline T window_store(Scalar value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
    } else {
        return static_cast<T>(value);
    }
}

template <typename Scalar>
__device__ inline Scalar scaled_bessel_i0(Scalar value)
{
    static_assert(std::is_same_v<Scalar, float>);
    const Scalar absolute = fabsf(value);
    if (absolute < static_cast<Scalar>(3.75F)) {
        Scalar ratio = value / static_cast<Scalar>(3.75F);
        const Scalar y = ratio * ratio;
        const Scalar polynomial = static_cast<Scalar>(1) + y *
            (static_cast<Scalar>(3.5156229F) + y *
            (static_cast<Scalar>(3.0899424F) + y *
            (static_cast<Scalar>(1.2067492F) + y *
            (static_cast<Scalar>(0.2659732F) + y *
            (static_cast<Scalar>(0.0360768F) + y *
             static_cast<Scalar>(0.0045813F))))));
        return polynomial * expf(-absolute);
    }
    const Scalar y = static_cast<Scalar>(3.75F) / absolute;
    return static_cast<Scalar>(1) / sqrtf(absolute) *
        (static_cast<Scalar>(0.39894228F) + y *
        (static_cast<Scalar>(0.01328592F) + y *
        (static_cast<Scalar>(0.00225319F) + y *
        (static_cast<Scalar>(-0.00157565F) + y *
        (static_cast<Scalar>(0.00916281F) + y *
        (static_cast<Scalar>(-0.02057706F) + y *
        (static_cast<Scalar>(0.02635537F) + y *
        (static_cast<Scalar>(-0.01647633F) + y *
         static_cast<Scalar>(0.00392377F)))))))));
}

template <typename Scalar>
__device__ inline Scalar stable_bessel_i0_ratio(
    Scalar numerator_argument,
    Scalar denominator_argument)
{
    static_assert(std::is_same_v<Scalar, float>);
    const Scalar numerator_absolute = fabsf(numerator_argument);
    const Scalar denominator_absolute = fabsf(denominator_argument);
    return expf(numerator_absolute - denominator_absolute) *
        scaled_bessel_i0(numerator_absolute) /
        scaled_bessel_i0(denominator_absolute);
}

template <typename Input, typename Output = Input>
__global__ void general_cosine_kernel(
    int output_count,
    const Input* coefficients,
    int coefficient_count,
    Output* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<Output>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(coefficients[0]));
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar phase = -pi + static_cast<Scalar>(2) * pi *
        static_cast<Scalar>(index) / static_cast<Scalar>(effective_count - 1);
    Scalar value = Scalar{0};
    for (int order = 0; order < coefficient_count; ++order) {
        value += window_load(coefficients[order]) *
            cos(static_cast<Scalar>(order) * phase);
    }
    output[index] = window_store<Output>(value);
}

template <typename T, typename Parameter>
__global__ void general_gaussian_kernel(
    int output_count,
    Parameter power_input,
    Parameter width_input,
    T* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(power_input));
    const Scalar power = window_load(power_input);
    const Scalar width = window_load(width_input);
    const Scalar position =
        (static_cast<Scalar>(index) - static_cast<Scalar>(0.5F) *
            static_cast<Scalar>(effective_count - 1)) / width;
    output[index] = window_store<T>(exp(
        static_cast<Scalar>(-0.5F) * pow(
            fabs(position), static_cast<Scalar>(2) * power)));
}

template <typename T>
__global__ void hamming_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    using Scalar = decltype(window_load(output[index]));
    Scalar value = static_cast<Scalar>(1);
    if (effective_count != 1) {
        const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
        value = static_cast<Scalar>(0.54F) - static_cast<Scalar>(0.46F) *
            cos(static_cast<Scalar>(2) * pi * static_cast<Scalar>(index) /
                static_cast<Scalar>(effective_count - 1));
    }
    output[index] = window_store<T>(value);
}

template <typename T, typename Parameter>
__global__ void kaiser_kernel(
    int output_count,
    Parameter beta_input,
    T* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(beta_input));
    const Scalar beta = window_load(beta_input);
    const Scalar alpha = static_cast<Scalar>(0.5F) *
        static_cast<Scalar>(effective_count - 1);
    const Scalar position =
        (static_cast<Scalar>(index) - alpha) / alpha;
    const Scalar radicand = static_cast<Scalar>(1) - position * position;
    const Scalar bounded = radicand > Scalar{0} ? radicand : Scalar{0};
    output[index] = window_store<T>(stable_bessel_i0_ratio(
        beta * sqrtf(bounded), beta));
}

template <typename T>
__global__ void parzen_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(output[index]));
    const Scalar position = fabs(
        static_cast<Scalar>(index) - static_cast<Scalar>(0.5F) *
            static_cast<Scalar>(effective_count - 1)) /
        (static_cast<Scalar>(0.5F) * static_cast<Scalar>(effective_count));
    Scalar value;
    if (position <= static_cast<Scalar>(0.5F)) {
        value = static_cast<Scalar>(1) - static_cast<Scalar>(6) *
            position * position + static_cast<Scalar>(6) *
            position * position * position;
    } else {
        const Scalar tail = static_cast<Scalar>(1) - position;
        value = static_cast<Scalar>(2) * tail * tail * tail;
    }
    output[index] = window_store<T>(value);
}

template <typename T>
__global__ void triang_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    const int midpoint = effective_count / 2;
    const int rank = index < midpoint ? index + 1 : effective_count - index;
    using Scalar = decltype(window_load(output[index]));
    const Scalar value = (effective_count & 1)
        ? static_cast<Scalar>(2 * rank) /
            static_cast<Scalar>(effective_count + 1)
        : static_cast<Scalar>(2 * rank - 1) /
            static_cast<Scalar>(effective_count);
    output[index] = window_store<T>(value);
}

template <typename Scalar>
__device__ inline Scalar chebyshev_spectrum_real(
    int frequency,
    int length,
    Scalar beta)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar angle = pi * static_cast<Scalar>(frequency) /
        static_cast<Scalar>(length);
    const Scalar x = beta * cos(angle);
    const int order = length - 1;
    if (order <= 0) return static_cast<Scalar>(1);
    Scalar previous = static_cast<Scalar>(1);
    Scalar current = x;
    for (int index = 2; index <= order; ++index) {
        const Scalar next = static_cast<Scalar>(2) * x * current - previous;
        previous = current;
        current = next;
    }
    return current;
}

__device__ inline int chebyshev_raw_index(int output_index, int length)
{
    if (length & 1) {
        const int center = (length + 1) / 2 - 1;
        return output_index <= center
            ? center - output_index
            : output_index - center;
    }
    const int half = length / 2;
    return output_index < half
        ? half - output_index
        : output_index - half + 1;
}

template <typename Scalar>
__device__ inline Scalar chebyshev_raw_value(
    int output_index,
    int length,
    Scalar beta)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar two_pi = static_cast<Scalar>(2) * pi;
    const int source = chebyshev_raw_index(output_index, length);
    const bool odd = (length & 1) != 0;
    static_assert(std::is_same_v<Scalar, float>);
    compute_policy::CompensatedFloatAccumulator sum;
    for (int frequency = 0; frequency < length; ++frequency) {
        const Scalar real = chebyshev_spectrum_real(frequency, length, beta);
        const Scalar phase = pi * static_cast<Scalar>(frequency) /
            static_cast<Scalar>(length);
        const Scalar spectrum_real = odd ? real : real * cos(phase);
        const Scalar spectrum_imag = odd ? Scalar{0} : real * sin(phase);
        const Scalar angle = -two_pi * static_cast<Scalar>(source) *
            static_cast<Scalar>(frequency) / static_cast<Scalar>(length);
        sum.add(fmaf(
            spectrum_real,
            cos(angle),
            -spectrum_imag * sin(angle)));
    }
    return sum.value();
}

template <typename Parameter, typename Scalar>
__global__ void chebwin_raw_kernel(
    int effective_count,
    Parameter attenuation_input,
    Scalar* raw)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= effective_count) return;
    if (effective_count == 1) {
        raw[index] = static_cast<Scalar>(1);
        return;
    }
    const Scalar attenuation = static_cast<Scalar>(window_load(attenuation_input));
    const Scalar beta = cosh(acosh(pow(
        static_cast<Scalar>(10),
        fabs(attenuation) / static_cast<Scalar>(20))) /
        static_cast<Scalar>(effective_count - 1));
    raw[index] = chebyshev_raw_value(index, effective_count, beta);
}

#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN
template <typename Parameter>
__device__ inline ComplexFloat chebwin_spectrum_value(
    int index,
    int effective_count,
    Parameter attenuation_input)
{
    const float pi = 3.14159265358979323846F;
    const float attenuation = static_cast<float>(window_load(attenuation_input));
    const int integer_order = effective_count - 1;
    const float order = static_cast<float>(integer_order);
    const float angle = pi * static_cast<float>(index) /
        static_cast<float>(effective_count);

    // beta and |cos(angle)| are both very close to one for the bins that
    // determine a long high-attenuation window.  Forming their product in
    // FP32 first loses the small difference from one, and multiplying the
    // resulting phase by order amplifies that rounding error.  The half-angle
    // form computes |x|-1 directly without cancellation:
    //   cosh(u) cos(v) - 1
    //     = 2(sinh(u/2)^2 - sin(v/2)^2)
    //       - 4 sinh(u/2)^2 sin(v/2)^2.
    const float growth = acoshf(powf(
        10.0F, fabsf(attenuation) / 20.0F)) / order;
    const float growth_half_sinh = sinhf(0.5F * growth);
    const float growth_square = growth_half_sinh * growth_half_sinh;
    const int folded_index = index <= effective_count / 2
        ? index
        : effective_count - index;
    const float folded_angle = pi * static_cast<float>(folded_index) /
        static_cast<float>(effective_count);
    const float angle_half_sine = sinf(0.5F * folded_angle);
    const float angle_square = angle_half_sine * angle_half_sine;
    const float delta = 2.0F * (growth_square - angle_square) -
        4.0F * growth_square * angle_square;
    float real;
    if (delta > 0.0F) {
        const float inverse_hyperbolic = 2.0F * asinhf(sqrtf(0.5F * delta));
        real = coshf(order * inverse_hyperbolic);
    } else {
        const float inverse_cosine = 2.0F * asinf(sqrtf(-0.5F * delta));
        real = cosf(order * inverse_cosine);
    }
    const bool negative_x = index > effective_count / 2;
    if (negative_x && (integer_order & 1)) real = -real;

    return (effective_count & 1)
        ? ComplexFloat{real, 0.0F}
        : ComplexFloat{real * cosf(angle), real * sinf(angle)};
}

template <typename Parameter>
__global__ void chebwin_spectrum_kernel(
    int effective_count,
    Parameter attenuation_input,
    ComplexFloat* spectrum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= effective_count) return;
    spectrum[index] = chebwin_spectrum_value(
        index, effective_count, attenuation_input);
}

// Exact mixed-radix DFT for the non-power-of-two chebwin boundary.  With
// N=radix1*radix2 and n=n1+radix1*n2, the first stage transforms n2.  The
// second stage applies the twiddle and transforms n1.  This is ordinary
// Cooley-Tukey factorization, not zero-padding, truncation or Bluestein.
template <typename Parameter>
__global__ void chebwin_mixed_radix_stage1_kernel(
    int effective_count,
    int radix1,
    int radix2,
    Parameter attenuation_input,
    ComplexFloat* stage)
{
    const int output_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= effective_count) return;
    const int n1 = output_index / radix2;
    const int k2 = output_index - n1 * radix2;
    const float two_pi = 6.28318530717958647692F;
    compute_policy::CompensatedFloatAccumulator real_sum;
    compute_policy::CompensatedFloatAccumulator imag_sum;
    for (int n2 = 0; n2 < radix2; ++n2) {
        const int source_index = n1 + radix1 * n2;
        const ComplexFloat value = chebwin_spectrum_value(
            source_index, effective_count, attenuation_input);
        const float angle = -two_pi * static_cast<float>(n2 * k2) /
            static_cast<float>(radix2);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        real_sum.add(fmaf(value.re, cosine, -value.im * sine));
        imag_sum.add(fmaf(value.re, sine, value.im * cosine));
    }
    stage[output_index] = ComplexFloat{real_sum.value(), imag_sum.value()};
}

__global__ void chebwin_mixed_radix_stage2_kernel(
    const ComplexFloat* stage,
    int effective_count,
    int radix1,
    int radix2,
    ComplexFloat* transformed)
{
    const int output_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= effective_count) return;
    const int k2 = output_index % radix2;
    const float two_pi = 6.28318530717958647692F;
    compute_policy::CompensatedFloatAccumulator real_sum;
    compute_policy::CompensatedFloatAccumulator imag_sum;
    for (int n1 = 0; n1 < radix1; ++n1) {
        const ComplexFloat value = stage[n1 * radix2 + k2];
        const float angle = -two_pi * static_cast<float>(n1 * output_index) /
            static_cast<float>(effective_count);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        real_sum.add(fmaf(value.re, cosine, -value.im * sine));
        imag_sum.add(fmaf(value.re, sine, value.im * cosine));
    }
    transformed[output_index] =
        ComplexFloat{real_sum.value(), imag_sum.value()};
}

__global__ void chebwin_reorder_kernel(
    const ComplexFloat* transformed,
    int effective_count,
    float* reordered)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= effective_count) return;

    int source;
    if (effective_count & 1) {
        const int center = (effective_count - 1) / 2;
        source = index <= center ? center - index : index - center;
    } else {
        const int half = effective_count / 2;
        source = index < half ? half - index : index - half + 1;
    }
    reordered[index] = transformed[source].re;
}
#endif

template <typename Scalar>
__global__ void absolute_kernel(const Scalar* input, Scalar* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) output[index] = fabs(input[index]);
}

template <typename Scalar, typename T>
__global__ void normalize_window_kernel(
    const Scalar* raw,
    int output_count,
    Scalar peak,
    T* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    const Scalar value = peak > Scalar{0} ? raw[index] / peak : raw[index];
    output[index] = window_store<T>(value);
}

template <typename Scalar>
inline std::vector<Scalar> build_taylor_coefficients(
    int effective_count,
    int sidelobe_count,
    Scalar sidelobe_level,
    bool normalize,
    Scalar& scale)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar amplitude = pow(
        static_cast<Scalar>(10),
        sidelobe_level / static_cast<Scalar>(20));
    const Scalar shape = acosh(amplitude) / pi;
    const Scalar offset = static_cast<Scalar>(sidelobe_count) -
        static_cast<Scalar>(0.5F);
    const Scalar denominator_scale =
        static_cast<Scalar>(sidelobe_count * sidelobe_count) /
        (shape * shape + offset * offset);
    const int coefficient_count = sidelobe_count > 1
        ? sidelobe_count - 1
        : 0;
    std::vector<Scalar> coefficients(coefficient_count);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
        const int order = coefficient_index + 1;
        const Scalar order_squared = static_cast<Scalar>(order * order);
        Scalar numerator = (coefficient_index & 1)
            ? static_cast<Scalar>(-1)
            : static_cast<Scalar>(1);
        for (int index = 1; index < sidelobe_count; ++index) {
            const Scalar index_offset =
                static_cast<Scalar>(index) - static_cast<Scalar>(0.5F);
            numerator *= static_cast<Scalar>(1) - order_squared /
                denominator_scale /
                (shape * shape + index_offset * index_offset);
        }
        Scalar denominator = static_cast<Scalar>(2);
        for (int index = 1; index < sidelobe_count; ++index) {
            if (index != order) {
                denominator *= static_cast<Scalar>(1) - order_squared /
                    static_cast<Scalar>(index * index);
            }
        }
        coefficients[coefficient_index] = numerator / denominator;
    }
    scale = static_cast<Scalar>(1);
    if (normalize) {
        const Scalar phase_scale = static_cast<Scalar>(2) * pi /
            static_cast<Scalar>(effective_count);
        const Scalar center_phase = phase_scale *
            ((static_cast<Scalar>(effective_count) - static_cast<Scalar>(1)) /
                static_cast<Scalar>(2) -
             static_cast<Scalar>(effective_count) / static_cast<Scalar>(2) +
             static_cast<Scalar>(0.5F));
        Scalar center = Scalar{0};
        for (int order = 1; order < sidelobe_count; ++order) {
            center += coefficients[order - 1] *
                cos(center_phase * static_cast<Scalar>(order));
        }
        scale = static_cast<Scalar>(1) /
            (static_cast<Scalar>(1) + static_cast<Scalar>(2) * center);
    }
    return coefficients;
}

template <typename T, typename Scalar>
__global__ void taylor_kernel(
    int output_count,
    const Scalar* coefficients,
    int coefficient_count,
    Scalar scale,
    T* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar phase = static_cast<Scalar>(2) * pi /
        static_cast<Scalar>(effective_count) *
        (static_cast<Scalar>(index) -
         static_cast<Scalar>(effective_count) / static_cast<Scalar>(2) +
         static_cast<Scalar>(0.5F));
    static_assert(std::is_same_v<Scalar, float>);
    compute_policy::CompensatedFloatAccumulator sum;
    for (int order = 1; order <= coefficient_count; ++order) {
        sum.add(fmaf(
            coefficients[order - 1],
            cos(phase * static_cast<Scalar>(order)),
            0.0F));
    }
    output[index] = window_store<T>(
        (static_cast<Scalar>(1) + static_cast<Scalar>(2) * sum.value()) * scale);
}

}  // namespace cusignal::windows_detail
