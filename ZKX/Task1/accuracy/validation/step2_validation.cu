#include "accuracy/validation/step2_validation.h"

namespace task1::accuracy {

Step2ValidationData generate_step2_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
    Step2ValidationData data;
    cusignal::PulseCompressionOptions options;
    options.num_pulses = config.num_pulses;
    options.samples_per_pulse = config.samples_per_pulse;
    options.normalize = true;
    options.nfft = config.samples_per_pulse;
    options.window = cusignal::RadarWindowKind::none;
    data.compressed_cpu = radar_cpu_to_host(
        cusignal::pulse_compression_typed_cpu<float>(
            state.echo, state.waveform, options));
    return data;
}

Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data = generate_step2_cpu_reference(state);
    cusignal::PulseCompressionOptions options;
    options.num_pulses = state.config.num_pulses;
    options.samples_per_pulse = state.config.samples_per_pulse;
    options.normalize = true;
    options.nfft = state.config.samples_per_pulse;
    options.window = cusignal::RadarWindowKind::none;
    auto altered_waveform = state.waveform;
    altered_waveform[0].real *= -1.0F;
    altered_waveform[0].imag *= -1.0F;
    data.altered_template_cpu = radar_cpu_to_host(
        cusignal::pulse_compression_typed_cpu<float>(
            state.echo, altered_waveform, options));
    return data;
}

}  // namespace task1::accuracy
