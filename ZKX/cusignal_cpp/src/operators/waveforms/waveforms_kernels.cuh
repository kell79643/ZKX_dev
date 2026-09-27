#pragma once

#include "waveform_math.h"
#include <cusignal/runtime/device_complex.h>
#include "simple_signal_typed.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cusignal::waveforms_detail {

template <typename T>
__host__ __device__ inline auto waveform_load(T value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::load(value);
    } else {
        return value;
    }
}

template <typename T, typename Scalar>
__host__ __device__ inline T waveform_store(Scalar value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
    } else {
        return static_cast<T>(value);
    }
}

template <typename Scalar>
__device__ inline Scalar wrap_two_pi(Scalar value)
{
    const Scalar two_pi = static_cast<Scalar>(6.28318530717958647692F);
    Scalar wrapped = fmod(value, two_pi);
    if (wrapped < Scalar{0}) {
        wrapped += two_pi;
    }
    return wrapped;
}

template <typename T, typename Scalar>
__global__ void chirp_real_kernel(
    const T* t,
    T* output,
    std::size_t count,
    Scalar f0,
    Scalar t1,
    Scalar f1,
    int method,
    Scalar phase_offset,
    bool vertex_zero)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float time = static_cast<float>(waveform_load(t[index]));
    const float phase = stable_chirp_phase(
        time, static_cast<float>(f0), static_cast<float>(t1),
        static_cast<float>(f1), static_cast<ChirpMethod>(method), vertex_zero);
    output[index] = waveform_store<T>(
        cosf(phase + static_cast<float>(phase_offset)));
}

template <typename T, typename Scalar>
__global__ void chirp_complex_kernel(
    const T* t,
    ComplexFloat* output,
    std::size_t count,
    Scalar f0,
    Scalar t1,
    Scalar f1,
    Scalar phase_offset)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float time = static_cast<float>(waveform_load(t[index]));
    const float phase = stable_chirp_phase(
        time, static_cast<float>(f0), static_cast<float>(t1),
        static_cast<float>(f1), ChirpMethod::Linear, true) +
        static_cast<float>(phase_offset);
    output[index] = ComplexFloat{cosf(phase), sinf(phase)};
}

template <typename T, typename Scalar>
__global__ void gausspulse_kernel(
    const T* t,
    T* in_phase,
    T* quadrature,
    T* envelope,
    std::size_t count,
    Scalar carrier_frequency,
    Scalar attenuation)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float time = static_cast<float>(waveform_load(t[index]));
    const float env = expf(-static_cast<float>(attenuation) * time * time);
    const float phase = kTwoPi * static_cast<float>(carrier_frequency) * time;
    if (envelope != nullptr) {
        envelope[index] = waveform_store<T>(env);
    }
    if (quadrature != nullptr) {
        quadrature[index] = waveform_store<T>(env * sinf(phase));
    }
    if (in_phase != nullptr) {
        in_phase[index] = waveform_store<T>(env * cosf(phase));
    }
}

template <typename Input, typename Parameter, typename Output = Input>
__global__ void sawtooth_fixed_kernel(
    const Input* t,
    Output* output,
    std::size_t count,
    Parameter width_input)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    using Scalar = decltype(waveform_load(t[index]));
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar two_pi = static_cast<Scalar>(2) * pi;
    const Scalar width = static_cast<Scalar>(waveform_load(width_input));
    if (width < static_cast<Scalar>(0) || width > static_cast<Scalar>(1)) {
        output[index] = waveform_store<Output>(NAN);
        return;
    }
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    const Scalar value = phase < width * two_pi
        ? phase / (pi * width) - static_cast<Scalar>(1)
        : (pi * (width + static_cast<Scalar>(1)) - phase) /
            (pi * (static_cast<Scalar>(1) - width));
    output[index] = waveform_store<Output>(value);
}

template <typename Input, typename Parameter, typename Output = float>
__global__ void sawtooth_variable_kernel(
    const Input* t,
    const Parameter* width_input,
    Output* output,
    std::size_t count,
    std::size_t width_count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    using Scalar = decltype(waveform_load(t[index]));
    const std::size_t width_index = width_count == 1 ? 0 : index;
    const Scalar width = static_cast<Scalar>(waveform_load(width_input[width_index]));
    if (width < static_cast<Scalar>(0) || width > static_cast<Scalar>(1)) {
        output[index] = waveform_store<Output>(NAN);
        return;
    }
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar two_pi = static_cast<Scalar>(2) * pi;
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    const Scalar value = phase < width * two_pi
        ? phase / (pi * width) - static_cast<Scalar>(1)
        : (pi * (width + static_cast<Scalar>(1)) - phase) /
            (pi * (static_cast<Scalar>(1) - width));
    output[index] = waveform_store<Output>(value);
}

template <typename Input, typename Parameter, typename Output = float>
__global__ void sawtooth_broadcast_kernel(
    const Input* t,
    const Parameter* width_input,
    const std::size_t* t_indices,
    const std::size_t* width_indices,
    Output* output,
    std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    using Scalar = decltype(waveform_load(t[t_indices[index]]));
    const Scalar width = static_cast<Scalar>(
        waveform_load(width_input[width_indices[index]]));
    if (width < static_cast<Scalar>(0) || width > static_cast<Scalar>(1)) {
        output[index] = waveform_store<Output>(NAN);
        return;
    }
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar two_pi = static_cast<Scalar>(2) * pi;
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[t_indices[index]])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    const Scalar value = phase < width * two_pi
        ? phase / (pi * width) - static_cast<Scalar>(1)
        : (pi * (width + static_cast<Scalar>(1)) - phase) /
            (pi * (static_cast<Scalar>(1) - width));
    output[index] = waveform_store<Output>(value);
}

template <typename Input, typename Parameter, typename Output = Input>
__global__ void square_fixed_kernel(
    const Input* t,
    Output* output,
    std::size_t count,
    Parameter duty_input)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    using Scalar = decltype(waveform_load(t[index]));
    const Scalar duty = static_cast<Scalar>(waveform_load(duty_input));
    const Scalar two_pi = static_cast<Scalar>(6.28318530717958647692F);
    if (duty < static_cast<Scalar>(0) || duty > static_cast<Scalar>(1)) {
        output[index] = waveform_store<Output>(NAN);
        return;
    }
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    output[index] = waveform_store<Output>(
        phase < duty * two_pi ? static_cast<Scalar>(1) : static_cast<Scalar>(-1));
}

template <typename Input, typename Parameter, typename Output = float>
__global__ void square_variable_kernel(
    const Input* t,
    const Parameter* duty_input,
    Output* output,
    std::size_t count,
    std::size_t duty_count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    using Scalar = decltype(waveform_load(t[index]));
    const std::size_t duty_index = duty_count == 1 ? 0 : index;
    const Scalar duty = static_cast<Scalar>(waveform_load(duty_input[duty_index]));
    const Scalar two_pi = static_cast<Scalar>(6.28318530717958647692F);
    if (duty < static_cast<Scalar>(0) || duty > static_cast<Scalar>(1)) {
        output[index] = waveform_store<Output>(NAN);
        return;
    }
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    output[index] = waveform_store<Output>(
        phase < duty * two_pi ? static_cast<Scalar>(1) : static_cast<Scalar>(-1));
}

template <typename Input, typename Parameter>
__global__ void square_fixed_fp64_storage_kernel(
    const Input* t,
    std::uint64_t* output,
    std::size_t count,
    Parameter duty_input)
{
    constexpr std::uint64_t positive_one_bits = UINT64_C(0x3ff0000000000000);
    constexpr std::uint64_t negative_one_bits = UINT64_C(0xbff0000000000000);
    constexpr std::uint64_t quiet_nan_bits = UINT64_C(0x7ff8000000000000);
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    using Scalar = decltype(waveform_load(t[index]));
    const Scalar duty = static_cast<Scalar>(waveform_load(duty_input));
    const Scalar two_pi = static_cast<Scalar>(6.28318530717958647692F);
    if (duty < static_cast<Scalar>(0) || duty > static_cast<Scalar>(1)) {
        output[index] = quiet_nan_bits;
        return;
    }
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    output[index] = phase < duty * two_pi
        ? positive_one_bits
        : negative_one_bits;
}

template <typename Input, typename Parameter>
__global__ void square_variable_fp64_storage_kernel(
    const Input* t,
    const Parameter* duty_input,
    std::uint64_t* output,
    std::size_t count,
    std::size_t duty_count)
{
    constexpr std::uint64_t positive_one_bits = UINT64_C(0x3ff0000000000000);
    constexpr std::uint64_t negative_one_bits = UINT64_C(0xbff0000000000000);
    constexpr std::uint64_t quiet_nan_bits = UINT64_C(0x7ff8000000000000);
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    using Scalar = decltype(waveform_load(t[index]));
    const std::size_t duty_index = duty_count == 1 ? 0 : index;
    const Scalar duty = static_cast<Scalar>(waveform_load(duty_input[duty_index]));
    const Scalar two_pi = static_cast<Scalar>(6.28318530717958647692F);
    if (duty < static_cast<Scalar>(0) || duty > static_cast<Scalar>(1)) {
        output[index] = quiet_nan_bits;
        return;
    }
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    output[index] = phase < duty * two_pi
        ? positive_one_bits
        : negative_one_bits;
}

template <typename Input, typename Parameter>
__global__ void square_broadcast_fp64_storage_kernel(
    const Input* t,
    const Parameter* duty_input,
    const std::size_t* t_indices,
    const std::size_t* duty_indices,
    std::uint64_t* output,
    std::size_t count)
{
    constexpr std::uint64_t positive_one_bits = UINT64_C(0x3ff0000000000000);
    constexpr std::uint64_t negative_one_bits = UINT64_C(0xbff0000000000000);
    constexpr std::uint64_t quiet_nan_bits = UINT64_C(0x7ff8000000000000);
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    using Scalar = decltype(waveform_load(t[t_indices[index]]));
    const Scalar duty = static_cast<Scalar>(
        waveform_load(duty_input[duty_indices[index]]));
    const Scalar two_pi = static_cast<Scalar>(6.28318530717958647692F);
    if (duty < static_cast<Scalar>(0) || duty > static_cast<Scalar>(1)) {
        output[index] = quiet_nan_bits;
        return;
    }
    Scalar phase = fmodf(
        static_cast<Scalar>(waveform_load(t[t_indices[index]])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    output[index] = phase < duty * two_pi
        ? positive_one_bits
        : negative_one_bits;
}

template <typename T>
__global__ void unit_impulse_kernel(T* output, std::size_t count, int impulse_index)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = static_cast<T>(
            static_cast<int>(index) == impulse_index ? 1 : 0);
    }
}

__global__ void unit_impulse_fp64_storage_kernel(
    std::uint64_t* output, std::size_t count, int impulse_index)
{
    constexpr std::uint64_t fp64_one_bits = UINT64_C(0x3ff0000000000000);
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = static_cast<int>(index) == impulse_index
            ? fp64_one_bits
            : UINT64_C(0);
    }
}

template <typename T>
__global__ void unit_impulse_2d_kernel(
    T* output,
    int rows,
    int columns,
    int impulse_row,
    int impulse_column)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    if (index >= count) {
        return;
    }
    const int row = static_cast<int>(index / columns);
    const int column = static_cast<int>(index % columns);
    output[index] = waveform_store<T>(
        row == impulse_row && column == impulse_column ? 1.0F : 0.0F);
}

}  // namespace cusignal::waveforms_detail
