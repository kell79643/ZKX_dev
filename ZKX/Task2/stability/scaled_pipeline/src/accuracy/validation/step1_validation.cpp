#include "accuracy/validation/step1_validation.h"

#include <cusignal/operators/waveforms/waveforms_typed.h>

#include <cmath>

namespace task2::accuracy {
namespace {

std::vector<float> synthesize_cpu(
    const std::vector<float>& chirp, const std::vector<float>& gaussian,
    const std::vector<float>& sawtooth, const std::vector<float>& square,
    const std::vector<float>& saved_noise, float target_weight, float gausspulse_weight,
    float sawtooth_weight, float square_weight, std::vector<float>* target,
    std::vector<float>* interference, std::vector<float>* noise)
{
    std::vector<float> echo(chirp.size());
    require(saved_noise.size() == chirp.size(),
            "CPU reference saved noise shape mismatch");
    target->assign(chirp.size(), 0.0F);
    interference->assign(chirp.size(), 0.0F);
    noise->assign(chirp.size(), 0.0F);
    for (std::size_t index = 0; index < chirp.size(); ++index) {
        (*target)[index] = index >= static_cast<std::size_t>(kTargetDelay)
            ? target_weight * chirp[index - kTargetDelay] : 0.0F;
        (*interference)[index] = gausspulse_weight * gaussian[index] +
            sawtooth_weight * sawtooth[index] + square_weight * square[index];
        (*noise)[index] = saved_noise[index];
        echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];
    }
    return echo;
}

}  // namespace

Step1ValidationData collect_step1_validation(const PipelineState& state)
{
    Step1ValidationData data;
    std::vector<float> pulse_time(kSamples), phase(kSamples), square_phase(kSamples);
    for (int index = 0; index < kSamples; ++index) {
        pulse_time[index] = state.time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / kSampleRate, 1.0);
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / kSampleRate, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    data.chirp_cpu = cusignal::chirp_typed_cpu(
        state.time, 20.0F,
        1.0F, 70.0F, 0.0F);
    const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(
        pulse_time, 170.0F, 0.20F);
    const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::sawtooth_typed_cpu(phase, 0.65));
    const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::square_typed_cpu(square_phase, 0.35));
    data.echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, kStabilityNoise,
        kChirpWeight, kGausspulseWeight, kSawtoothWeight, kSquareWeight,
        &data.target_cpu, &data.interference_cpu, &data.noise_cpu);
    std::vector<float> changed_target, changed_interference, changed_noise;
    data.changed_echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, kStabilityNoise,
        kChirpWeight * 0.95F, kGausspulseWeight, kSawtoothWeight, kSquareWeight,
        &changed_target, &changed_interference, &changed_noise);
    return data;
}

}  // namespace task2::accuracy
