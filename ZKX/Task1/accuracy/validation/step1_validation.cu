#include "accuracy/validation/step1_validation.h"

namespace task1::accuracy {
namespace {

std::uint32_t cpu_noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    value *= 3266489917U;
    return value ^ (value >> 16);
}

float cpu_uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)
{
    return (static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) /
            32767.5F - 1.0F) * noise_std;
}

std::vector<cusignal::TypedComplex<float>> waveform_cpu(const TaskConfig& config)
{
    std::vector<cusignal::TypedComplex<float>> output(config.pulse_samples);
    const double pulse_width = static_cast<double>(config.pulse_samples) / config.sample_rate_hz;
    const double slope = static_cast<double>(config.bandwidth_hz) / pulse_width;
    for (int index = 0; index < config.pulse_samples; ++index) {
        const double time = static_cast<double>(index) / config.sample_rate_hz - 0.5 * pulse_width;
        const double phase = 3.14159265358979323846 * slope * time * time;
        output[index] = {static_cast<float>(std::cos(phase)),
                         static_cast<float>(std::sin(phase))};
    }
    return output;
}

std::vector<cusignal::TypedComplex<float>> echo_cpu(
    const std::vector<cusignal::TypedComplex<float>>& waveform,
    int delay, float doppler_hz, std::uint32_t seed,
    const TaskConfig& config,
    std::vector<cusignal::TypedComplex<float>>* noiseless,
    std::vector<cusignal::TypedComplex<float>>* noise)
{
    const int count = config.num_pulses * config.samples_per_pulse;
    noiseless->assign(count, {0.0F, 0.0F});
    noise->assign(count, {0.0F, 0.0F});
    std::vector<cusignal::TypedComplex<float>> output(count);
    for (int index = 0; index < count; ++index) {
        const int pulse = index / config.samples_per_pulse;
        const int sample = index % config.samples_per_pulse;
        const int source = sample - delay;
        if (source >= 0 && source < config.pulse_samples) {
            const double time = static_cast<double>(pulse) / config.prf_hz +
                static_cast<double>(sample) / config.sample_rate_hz;
            const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            (*noiseless)[index] = {
                static_cast<float>(config.target_amplitude *
                    (waveform[source].real * cosine - waveform[source].imag * sine)),
                static_cast<float>(config.target_amplitude *
                    (waveform[source].real * sine + waveform[source].imag * cosine))};
        }
        const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);
        (*noise)[index] = {
            cpu_uniform_noise(base_index, seed, config.noise_std),
            cpu_uniform_noise(base_index + 1U, seed, config.noise_std)};
        output[index] = {
            (*noiseless)[index].real + (*noise)[index].real,
            (*noiseless)[index].imag + (*noise)[index].imag};
    }
    return output;
}

}  // namespace

Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)
{
    Step1ValidationData data;
    data.waveform_cpu = waveform_cpu(config);
    const float doppler_hz = config.doppler_hz();
    data.echo_cpu = echo_cpu(
        data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed, config,
        &data.noiseless_cpu, &data.noise_cpu);
    return data;
}

Step1ValidationData collect_step1_validation(const PipelineState& state)
{
    const auto& config = state.config;
    Step1ValidationData data = generate_step1_cpu_reference(config);
    const float doppler_hz = config.doppler_hz();
    std::vector<cusignal::TypedComplex<float>> temporary_noiseless, temporary_noise;
    data.changed_delay_cpu = echo_cpu(
        data.waveform_cpu, config.target_delay_samples + 3, doppler_hz, config.noise_seed, config,
        &temporary_noiseless, &temporary_noise);
    data.changed_doppler_cpu = echo_cpu(
        data.waveform_cpu, config.target_delay_samples,
        doppler_hz + config.prf_hz / config.num_pulses, config.noise_seed, config,
        &temporary_noiseless, &temporary_noise);
    data.changed_seed_cpu = echo_cpu(
        data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed + 1U, config,
        &temporary_noiseless, &temporary_noise);
    return data;
}

}  // namespace task1::accuracy
