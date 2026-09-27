#include <task1/steps/step3.h>

#include <cusignal/runtime/device_array.h>

namespace task1 {

StepEvidence run_step3(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step3";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    require_task1_upstream(state.compressed.size() == static_cast<std::size_t>(
                config.num_pulses * config.samples_per_pulse),
            "Step3", "Step2.compressed");
    cusignal::PulseDopplerOptions options;
    options.num_pulses = config.num_pulses;
    options.samples_per_pulse = config.samples_per_pulse;
    options.nfft = config.num_pulses;
    options.window = cusignal::RadarWindowKind::hamming;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    auto d_compressed =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    const std::size_t output_count = static_cast<std::size_t>(config.num_pulses) *
        static_cast<std::size_t>(config.samples_per_pulse);
    if (state.range_doppler_device.size() != output_count) {
        state.range_doppler_device.reset(output_count);
    }
    evidence.operator_ms["pulse_doppler"] = time_gpu([&] {
        cusignal::pulse_doppler_device<float>(
            d_compressed, options, state.range_doppler_device);
    });
    evidence.compute_ms = evidence.operator_ms["pulse_doppler"];
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    begin = Clock::now();
    const auto device_values = state.range_doppler_device.to_host();
    state.range_doppler.resize(device_values.size());
    for (std::size_t index = 0; index < device_values.size(); ++index) {
        state.range_doppler[index] = {
            device_values[index].re, device_values[index].im};
    }
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
