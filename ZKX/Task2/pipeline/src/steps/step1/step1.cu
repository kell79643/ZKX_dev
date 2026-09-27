#include <task2/steps/step1.h>

#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/kernel_launch.h>
#include <cusignal/operators/waveforms/waveforms_typed.h>

#include <cmath>
#include <numeric>

namespace task2 {
namespace {

__global__ void mix_echo_kernel(
    const float* chirp, const float* gaussian, const float* sawtooth,
    const float* square,
    float* target, float* interference, float* noise, float* echo,
    int count, int delay, std::uint32_t seed, float target_weight,
    float gaussian_weight, float sawtooth_weight, float square_weight,
    float noise_amplitude)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    std::uint32_t value = seed ^
        (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    value *= 3266489917U;
    value ^= value >> 16;
    const float noise_value =
        (static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;
    const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;
    const float interference_value =
        gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] +
        square_weight * square[index];
    target[index] = target_value;
    interference[index] = interference_value;
    noise[index] = noise_value;
    echo[index] = target_value + interference_value + noise_value;
}

}  // namespace

StepEvidence run_step1(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step1";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    state.time.resize(config.samples);
    std::vector<float> pulse_time(config.samples);
    std::vector<float> phase(config.samples);
    std::vector<float> square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        state.time[index] = static_cast<float>(index) / config.sample_rate_hz;
        pulse_time[index] = state.time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    evidence.prep_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    auto d_time = cusignal::DeviceArray<float>::from_host(state.time);
    auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);
    auto d_phase = cusignal::DeviceArray<float>::from_host(phase);
    auto d_square_phase = cusignal::DeviceArray<float>::from_host(square_phase);
    evidence.h2d_ms += milliseconds(begin, Clock::now());

    cusignal::DeviceArray<float> d_chirp(config.samples);
    cusignal::DeviceArray<float> d_gaussian(config.samples);
    cusignal::DeviceArray<double> d_sawtooth(config.samples);
    cusignal::DeviceArray<double> d_square(config.samples);
    evidence.operator_ms["chirp"] = time_gpu([&] {
        cusignal::chirp_device(
            d_time, d_chirp, 20.0F,
            std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),
            70.0F, 0.0F);
    });
    evidence.operator_ms["gausspulse"] = time_gpu([&] {
        cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F);
    });
    evidence.operator_ms["sawtooth"] = time_gpu([&] {
        cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65);
    });
    evidence.operator_ms["square"] = time_gpu([&] {
        cusignal::square_device(d_square_phase, d_square, 0.35);
    });

    begin = Clock::now();
    state.target_waveform = d_chirp.to_host();
    const auto gaussian = d_gaussian.to_host();
    const auto sawtooth_double = d_sawtooth.to_host();
    const auto square_double = d_square.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    const auto sawtooth =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double);
    const auto square =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double);
    begin = Clock::now();
    auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);
    auto d_square_float = cusignal::DeviceArray<float>::from_host(square);
    evidence.h2d_ms += milliseconds(begin, Clock::now());

    cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);
    cusignal::DeviceArray<float> d_noise(config.samples), d_echo(config.samples);
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    evidence.operator_ms["gpu_echo_superposition"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            mix_echo_kernel, static_cast<std::size_t>(config.samples),
            d_chirp.data(), d_gaussian.data(), d_saw_float.data(), d_square_float.data(),
            d_target.data(), d_interference.data(), d_noise.data(), d_echo.data(),
            config.samples, config.target_delay_samples, config.noise_seed,
            config.target_weight, config.gaussian_weight, config.sawtooth_weight,
            config.square_weight, config.noise_amplitude);
    });
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });

    begin = Clock::now();
    state.target_component = d_target.to_host();
    state.interference_component = d_interference.to_host();
    state.noise_component = d_noise.to_host();
    state.echo = d_echo.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled()) {
        state.gaussian_waveform = gaussian;
        state.sawtooth_waveform = sawtooth;
        state.square_waveform = square;
    }
    return evidence;
}

}  // namespace task2
