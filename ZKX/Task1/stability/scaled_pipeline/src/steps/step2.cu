#include "steps/step2.h"

#include <cusignal/runtime/device_array.h>

namespace task1 {

StepEvidence run_step2(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step2";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    require(state.echo.size() == static_cast<std::size_t>(
                kNumPulses * kSamplesPerPulse),
            "step2 did not receive the step1 echo");
    require(state.waveform.size() == kPulseSamples,
            "step2 did not receive the step1 waveform");
    cusignal::PulseCompressionOptions options;
    options.num_pulses = kNumPulses;
    options.samples_per_pulse = kSamplesPerPulse;
    options.normalize = true;
    options.nfft = kSamplesPerPulse;
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
    begin = Clock::now();
    state.compressed = radar_result_to_host(gpu_result);
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
