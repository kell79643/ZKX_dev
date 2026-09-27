#include "accuracy/validation/step3_validation.h"

namespace task1::accuracy {

Step3ValidationData generate_step3_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
    Step3ValidationData data;
    cusignal::PulseDopplerOptions options;
    options.num_pulses = config.num_pulses;
    options.samples_per_pulse = config.samples_per_pulse;
    options.nfft = config.num_pulses;
    options.window = cusignal::RadarWindowKind::hamming;
    data.range_doppler_cpu = radar_cpu_to_host(
        cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));
    return data;
}

Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data = generate_step3_cpu_reference(state);
    cusignal::PulseDopplerOptions options;
    options.num_pulses = state.config.num_pulses;
    options.samples_per_pulse = state.config.samples_per_pulse;
    options.nfft = state.config.num_pulses;
    options.window = cusignal::RadarWindowKind::none;
    data.no_window_cpu = radar_cpu_to_host(
        cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));
    return data;
}

}  // namespace task1::accuracy
