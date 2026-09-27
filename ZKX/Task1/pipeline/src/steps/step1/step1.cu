#include <task1/steps/step1.h>

#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/kernel_launch.h>

#include <numeric>

namespace task1 {
namespace {

__device__ std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    value *= 3266489917U;
    return value ^ (value >> 16);
}

__device__ float uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * noise_std;
}

__global__ void generate_lfm_kernel(
    cusignal::TypedComplex<float>* waveform, int pulse_samples,
    float sample_rate_hz, float bandwidth_hz)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pulse_samples) return;
    const float pulse_width = static_cast<float>(pulse_samples) / sample_rate_hz;
    const float time = static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width;
    const float phase = kPi * (bandwidth_hz / pulse_width) * time * time;
    waveform[index] = {cosf(phase), sinf(phase)};
}

__global__ void simulate_echo_kernel(
    const cusignal::TypedComplex<float>* waveform,
    cusignal::TypedComplex<float>* noiseless,
    cusignal::TypedComplex<float>* noise,
    cusignal::TypedComplex<float>* echo,
    int count, int delay, float doppler_hz, std::uint32_t seed,
    int samples_per_pulse, int pulse_samples, float prf_hz,
    float sample_rate_hz, float target_amplitude, float noise_std)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const int pulse = index / samples_per_pulse;
    const int sample = index % samples_per_pulse;
    float target_real = 0.0F;
    float target_imag = 0.0F;
    const int source = sample - delay;
    if (source >= 0 && source < pulse_samples) {
        const auto base = waveform[source];
        const float time = static_cast<float>(pulse) / prf_hz +
            static_cast<float>(sample) / sample_rate_hz;
        const float phase = 2.0F * kPi * doppler_hz * time;
        const float cosine = cosf(phase);
        const float sine = sinf(phase);
        target_real = target_amplitude * (base.real * cosine - base.imag * sine);
        target_imag = target_amplitude * (base.real * sine + base.imag * cosine);
    }
    const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);
    const float noise_real = uniform_noise(base_index, seed, noise_std);
    const float noise_imag = uniform_noise(base_index + 1U, seed, noise_std);
    noiseless[index] = {target_real, target_imag};
    noise[index] = {noise_real, noise_imag};
    echo[index] = {target_real + noise_real, target_imag + noise_imag};
}

}  // namespace

StepEvidence run_step1(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step1";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    const float doppler_hz = config.doppler_hz();
    const int count = config.num_pulses * config.samples_per_pulse;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_waveform(config.pulse_samples);
    evidence.operator_ms["gpu_lfm_generation"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            generate_lfm_kernel, static_cast<std::size_t>(config.pulse_samples),
            d_waveform.data(), config.pulse_samples,
            config.sample_rate_hz, config.bandwidth_hz);
    });
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noise(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_echo(count);
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            simulate_echo_kernel, static_cast<std::size_t>(count),
            d_waveform.data(), d_noiseless.data(), d_noise.data(), d_echo.data(),
            count, config.target_delay_samples, doppler_hz, config.noise_seed,
            config.samples_per_pulse, config.pulse_samples, config.prf_hz,
            config.sample_rate_hz, config.target_amplitude, config.noise_std);
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
