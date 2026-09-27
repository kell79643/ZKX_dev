#pragma once

#include <cusignal/operators/demod/demod_typed.h>

#include <cmath>

namespace cusignal {
namespace demod_detail {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 6.28318530717958647692F;

template <class Complex>
__device__ float phase(Complex value)
{
    const float real = detail::DemodTypePolicy<decltype(value.re)>::load(value.re);
    const float imag = detail::DemodTypePolicy<decltype(value.im)>::load(value.im);
    return atan2f(imag, real);
}

__device__ inline float unwrap_local_delta(float previous_phase, float current_phase)
{
    const float delta = current_phase - previous_phase;
    if (fabsf(delta) < kPi) return delta;

    float corrected = fmodf(delta + kPi, kTwoPi);
    if (corrected < 0.0F) corrected += kTwoPi;
    corrected -= kPi;
    if (corrected == -kPi && delta > 0.0F) corrected = kPi;
    return corrected;
}

template <class Complex>
__global__ void fm_demod_nd_kernel(
    const Complex* input,
    float* output,
    int output_count,
    int axis_length,
    int output_axis_length,
    int inner_size)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;

    const int inner_index = index % inner_size;
    const int axis_index = (index / inner_size) % output_axis_length;
    const int outer_index = index / (inner_size * output_axis_length);
    const int previous =
        (outer_index * axis_length + axis_index) * inner_size + inner_index;
    const int current = previous + inner_size;
    output[index] = unwrap_local_delta(phase(input[previous]), phase(input[current]));
}

}  // namespace demod_detail
}  // namespace cusignal
