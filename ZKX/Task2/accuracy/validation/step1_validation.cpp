#include "accuracy/validation/step1_validation.h"

#include <cusignal/operators/waveforms/waveforms_typed.h>

#include <cmath>

namespace task2::accuracy {
namespace {

std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    value *= 3266489917U;
    return value ^ (value >> 16);
}

float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * amplitude;
}

std::vector<float> synthesize_cpu(
    const std::vector<float>& chirp, const std::vector<float>& gaussian,
    const std::vector<float>& sawtooth, const std::vector<float>& square,
    std::uint32_t seed, float target_weight, std::vector<float>* target,
    std::vector<float>* interference, std::vector<float>* noise,
    const TaskConfig& config)
{
    std::vector<float> echo(chirp.size());
    target->assign(chirp.size(), 0.0F);
    interference->assign(chirp.size(), 0.0F);
    noise->assign(chirp.size(), 0.0F);
    for (std::size_t index = 0; index < chirp.size(); ++index) {
        (*target)[index] = index >= static_cast<std::size_t>(config.target_delay_samples)
            ? target_weight * chirp[index - config.target_delay_samples] : 0.0F;
        (*interference)[index] = config.gaussian_weight * gaussian[index] +
            config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index];
        (*noise)[index] = noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude);
        echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];
    }
    return echo;
}

}  // namespace

Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)
{
    Step1ValidationData data;
    std::vector<float> time(config.samples), pulse_time(config.samples);
    std::vector<float> phase(config.samples), square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        time[index] = static_cast<float>(index) / config.sample_rate_hz;
        pulse_time[index] = time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    data.chirp_cpu = cusignal::chirp_typed_cpu(
        time, 20.0F,
        std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),
        70.0F, 0.0F);
    const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(
        pulse_time, 170.0F, 0.20F);
    const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::sawtooth_typed_cpu(phase, 0.65));
    const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::square_typed_cpu(square_phase, 0.35));
    data.echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed,
        config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config);
    std::vector<float> changed_target, changed_interference, changed_noise;
    data.changed_echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U,
        std::max(0.0F, config.target_weight - 0.05F),
        &changed_target, &changed_interference, &changed_noise, config);
    return data;
}

Step1ValidationData collect_step1_validation(const PipelineState& state)
{
    return generate_step1_cpu_reference(state.config);
}

}  // namespace task2::accuracy
