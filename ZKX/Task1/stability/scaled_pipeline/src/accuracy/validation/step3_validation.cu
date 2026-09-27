#include "accuracy/validation/step3_validation.h"

namespace task1::accuracy {

Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data;
    cusignal::PulseDopplerOptions options;
    options.num_pulses = kNumPulses;
    options.samples_per_pulse = kSamplesPerPulse;
    options.nfft = kNumPulses;
    options.window = cusignal::RadarWindowKind::hamming;
    data.range_doppler_cpu = radar_cpu_to_host(
        cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));
    options.window = cusignal::RadarWindowKind::none;
    data.no_window_cpu = radar_cpu_to_host(
        cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));
    return data;
}

}  // namespace task1::accuracy
