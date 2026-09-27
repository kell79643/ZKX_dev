#include "accuracy/validation/step1_validation.h"

namespace task1::accuracy {
namespace {

std::vector<cusignal::TypedComplex<float>> waveform_cpu()
{
    std::vector<cusignal::TypedComplex<float>> output(kPulseSamples);
    const double pulse_width = static_cast<double>(kPulseSamples) / kSampleRate;
    const double slope = static_cast<double>(kBandwidth) / pulse_width;
    for (int index = 0; index < kPulseSamples; ++index) {
        const double time = static_cast<double>(index) / kSampleRate - 0.5 * pulse_width;
        const double phase = 3.14159265358979323846 * slope * time * time;
        output[index] = {static_cast<float>(std::cos(phase)),
                         static_cast<float>(std::sin(phase))};
    }
    return output;
}

std::vector<cusignal::TypedComplex<float>> echo_cpu(
    const std::vector<cusignal::TypedComplex<float>>& waveform,
    int delay, float doppler_hz,
    const std::vector<cusignal::TypedComplex<float>>& saved_noise,
    std::vector<cusignal::TypedComplex<float>>* noiseless,
    std::vector<cusignal::TypedComplex<float>>* noise)
{
    const int count = kNumPulses * kSamplesPerPulse;
    require(saved_noise.size() == static_cast<std::size_t>(count),
            "CPU reference saved noise shape mismatch");
    noiseless->assign(count, {0.0F, 0.0F});
    noise->assign(count, {0.0F, 0.0F});
    std::vector<cusignal::TypedComplex<float>> output(count);
    for (int index = 0; index < count; ++index) {
        const int pulse = index / kSamplesPerPulse;
        const int sample = index % kSamplesPerPulse;
        const int source = sample - delay;
        if (source >= 0 && source < kPulseSamples) {
            const double time = static_cast<double>(pulse) / kPrf +
                static_cast<double>(sample) / kSampleRate;
            const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            (*noiseless)[index] = {
                static_cast<float>(kTargetAmplitude *
                    (waveform[source].real * cosine - waveform[source].imag * sine)),
                static_cast<float>(kTargetAmplitude *
                    (waveform[source].real * sine + waveform[source].imag * cosine))};
        }
        (*noise)[index] = saved_noise[index];
        output[index] = {
            (*noiseless)[index].real + (*noise)[index].real,
            (*noiseless)[index].imag + (*noise)[index].imag};
    }
    return output;
}

}  // namespace

Step1ValidationData collect_step1_validation(const PipelineState&)
{
    Step1ValidationData data;
    data.waveform_cpu = waveform_cpu();
    const float doppler_hz = kDopplerFrequency;
    data.echo_cpu = echo_cpu(
        data.waveform_cpu, kTargetDelay, doppler_hz, kStabilityNoise,
        &data.noiseless_cpu, &data.noise_cpu);
    std::vector<cusignal::TypedComplex<float>> temporary_noiseless, temporary_noise;
    data.changed_delay_cpu = echo_cpu(
        data.waveform_cpu, kTargetDelay + 3, doppler_hz, kStabilityNoise,
        &temporary_noiseless, &temporary_noise);
    data.changed_doppler_cpu = echo_cpu(
        data.waveform_cpu, kTargetDelay, doppler_hz + kPrf / kNumPulses, kStabilityNoise,
        &temporary_noiseless, &temporary_noise);
    data.changed_seed_cpu = data.echo_cpu;
    return data;
}

}  // namespace task1::accuracy
