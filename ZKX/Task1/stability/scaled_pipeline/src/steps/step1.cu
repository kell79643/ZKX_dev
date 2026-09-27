#include "steps/step1.h"

#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/kernel_launch.h>

#include <numeric>

namespace task1 {
namespace {

__global__ void generate_lfm_kernel(
    cusignal::TypedComplex<float>* waveform, int pulse_samples)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pulse_samples) return;
    const float pulse_width = static_cast<float>(pulse_samples) / kSampleRate;
    const float time = static_cast<float>(index) / kSampleRate - 0.5F * pulse_width;
    const float phase = kPi * (kBandwidth / pulse_width) * time * time;
    waveform[index] = {cosf(phase), sinf(phase)};
}

__global__ void simulate_echo_kernel(
    const cusignal::TypedComplex<float>* waveform,
    const cusignal::TypedComplex<float>* saved_noise,
    cusignal::TypedComplex<float>* noiseless,
    cusignal::TypedComplex<float>* noise,
    cusignal::TypedComplex<float>* echo,
    int count, int delay, float doppler_hz, float target_amplitude)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const int pulse = index / kSamplesPerPulse;
    const int sample = index % kSamplesPerPulse;
    float target_real = 0.0F;
    float target_imag = 0.0F;
    const int source = sample - delay;
    if (source >= 0 && source < kPulseSamples) {
        const auto base = waveform[source];
        const float time = static_cast<float>(pulse) / kPrf +
            static_cast<float>(sample) / kSampleRate;
        const float phase = 2.0F * kPi * doppler_hz * time;
        const float cosine = cosf(phase);
        const float sine = sinf(phase);
        target_real = target_amplitude * (base.real * cosine - base.imag * sine);
        target_imag = target_amplitude * (base.real * sine + base.imag * cosine);
    }
    const float noise_real = saved_noise[index].real;
    const float noise_imag = saved_noise[index].imag;
    noiseless[index] = {target_real, target_imag};
    noise[index] = {noise_real, noise_imag};
    echo[index] = {target_real + noise_real, target_imag + noise_imag};
}

}  // namespace

StepEvidence run_step1(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step1";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    const float doppler_hz = kDopplerFrequency;
    const int count = kNumPulses * kSamplesPerPulse;
    require(kStabilityNoise.size() == static_cast<std::size_t>(count),
            "step1 saved noise shape mismatch");
    evidence.prep_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_waveform(kPulseSamples);
    evidence.operator_ms["gpu_lfm_generation"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            generate_lfm_kernel, static_cast<std::size_t>(kPulseSamples),
            d_waveform.data(), kPulseSamples);
    });
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noise(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_echo(count);
    auto d_saved_noise =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(kStabilityNoise);
    evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            simulate_echo_kernel, static_cast<std::size_t>(count),
            d_waveform.data(), d_saved_noise.data(), d_noiseless.data(),
            d_noise.data(), d_echo.data(), count, kTargetDelay, doppler_hz,
            kTargetAmplitude);
    });
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
    begin = Clock::now();
    state.waveform = d_waveform.to_host();
    state.noiseless_echo = d_noiseless.to_host();
    state.noise = d_noise.to_host();
    state.echo = d_echo.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
