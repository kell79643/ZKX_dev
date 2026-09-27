#include <task1/steps/step2.h>

#include <cusignal/runtime/device_array.h>

namespace task1 {

StepEvidence run_step2(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step2";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    require_task1_upstream(state.echo.size() == static_cast<std::size_t>(
                config.num_pulses * config.samples_per_pulse),
            "Step2", "Step1.echo");
    require_task1_upstream(
            state.waveform.size() == static_cast<std::size_t>(config.pulse_samples),
            "Step2", "Step1.waveform");
    cusignal::PulseCompressionOptions options;
    options.num_pulses = config.num_pulses;
    options.samples_per_pulse = config.samples_per_pulse;
    options.normalize = true;
    options.nfft = config.samples_per_pulse;
    options.window = cusignal::RadarWindowKind::none;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    auto d_echo = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.echo);
    auto d_waveform =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::RadarComplexDeviceResult gpu_result;
    evidence.operator_ms["pulse_compression"] = time_gpu([&] {
        gpu_result = cusignal::pulse_compression_device<float>(
            d_echo, d_waveform, options);
    });
    evidence.compute_ms = evidence.operator_ms["pulse_compression"];
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    begin = Clock::now();
    state.compressed = radar_result_to_host(gpu_result);
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
