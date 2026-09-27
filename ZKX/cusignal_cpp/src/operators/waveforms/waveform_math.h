#pragma once

#include <cmath>

namespace cusignal::waveforms_detail {

enum class ChirpMethod : int {
    Linear = 0,
    Quadratic = 1,
    Logarithmic = 2,
    Hyperbolic = 3
};

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 6.28318530717958647692F;

// The public inputs and outputs retain their cuSignal dtype contract, while the
// ZQ500 device path intentionally performs continuous phase math in FP32.
// These forms avoid the largest cancellation sites without requiring FP64.
__host__ __device__ inline float stable_chirp_phase(
    float time,
    float f0,
    float t1,
    float f1,
    ChirpMethod method,
    bool vertex_zero)
{
    switch (method) {
        case ChirpMethod::Linear: {
            const float beta = (f1 - f0) / t1;
            const float cycles = fmaf(0.5F * beta * time, time, f0 * time);
            return kTwoPi * cycles;
        }
        case ChirpMethod::Quadratic: {
            const float beta = (f1 - f0) / (t1 * t1);
            if (vertex_zero) {
                const float cubic_term = (beta / 3.0F) * time * time * time;
                return kTwoPi * fmaf(f0, time, cubic_term);
            }
            // (t1-time)^3 - t1^3 is cancellation-prone near time=0.  The
            // factored polynomial is algebraically identical and retains the
            // small phase increment in FP32.
            const float cube_delta = time *
                fmaf(time, 3.0F * t1 - time, -3.0F * t1 * t1);
            const float cycles = fmaf(beta / 3.0F, cube_delta, f1 * time);
            return kTwoPi * cycles;
        }
        case ChirpMethod::Logarithmic: {
            if (f0 == f1) return kTwoPi * f0 * time;
            const float log_ratio = logf(f1 / f0);
            const float relative_time = time / t1;
            const float growth = expm1f(log_ratio * relative_time);
            return kTwoPi * ((t1 * f0 / log_ratio) * growth);
        }
        case ChirpMethod::Hyperbolic: {
            if (f0 == f1) return kTwoPi * f0 * time;
            const float singular = -f1 * t1 / (f0 - f1);
            const float relative_time = time / singular;
            const float one_minus = 1.0F - relative_time;
            const float logarithm = one_minus > 0.0F
                ? log1pf(-relative_time)
                : logf(fabsf(one_minus));
            return kTwoPi * ((-singular * f0) * logarithm);
        }
    }
    return 0.0F;
}

__host__ __device__ inline float gausspulse_attenuation_fp32(
    float carrier_frequency,
    float fractional_bandwidth,
    float reference_level_db)
{
    constexpr float natural_log_ten = 2.30258509299404568402F;
    const float log_reference = (reference_level_db / 20.0F) * natural_log_ten;
    const float scaled_frequency = carrier_frequency * fractional_bandwidth;
    return -(kPi * kPi * scaled_frequency * scaled_frequency) /
        (4.0F * log_reference);
}

}  // namespace cusignal::waveforms_detail
