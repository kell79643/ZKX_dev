#include "accuracy/validation/step2_validation.h"

namespace task1::accuracy {

Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data;
    cusignal::PulseCompressionOptions options;
    options.num_pulses = kNumPulses;
    options.samples_per_pulse = kSamplesPerPulse;
    options.normalize = true;
    options.nfft = kSamplesPerPulse;
    options.window = cusignal::RadarWindowKind::none;
    data.compressed_cpu = radar_cpu_to_host(
        cusignal::pulse_compression_typed_cpu<float>(
            state.echo, state.waveform, options));
    auto altered_waveform = state.waveform;
    altered_waveform[0].real *= -1.0F;
    data.altered_template_cpu = radar_cpu_to_host(
        cusignal::pulse_compression_typed_cpu<float>(
            state.echo, altered_waveform, options));
    return data;
}

}  // namespace task1::accuracy
