#include "steps/step3.h"

#include <cusignal/runtime/device_array.h>

namespace task1 {

StepEvidence run_step3(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step3";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    require(state.compressed.size() == static_cast<std::size_t>(
                kNumPulses * kSamplesPerPulse),
            "step3 did not receive the step2 compressed signal");
    cusignal::PulseDopplerOptions options;
    options.num_pulses = kNumPulses;
    options.samples_per_pulse = kSamplesPerPulse;
    options.nfft = kNumPulses;
    options.window = cusignal::RadarWindowKind::hamming;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    auto d_compressed =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::RadarComplexDeviceResult gpu_result;
    evidence.operator_ms["pulse_doppler"] = time_gpu([&] {
        gpu_result = cusignal::pulse_doppler_device<float>(d_compressed, options);
    });
    evidence.compute_ms = evidence.operator_ms["pulse_doppler"];
    begin = Clock::now();
    state.range_doppler = radar_result_to_host(gpu_result);
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
