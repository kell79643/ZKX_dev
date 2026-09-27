#include "accuracy/validation/step4_validation.h"

namespace task1::accuracy {

Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    Step4ValidationData data;
    const int count = kNumPulses * kSamplesPerPulse;
    cusignal::CaCfarOptions options;
    options.guard_cells = {1, 2};
    options.reference_cells = {2, 6};
    options.pfa = kPfa;
    constexpr int reference_count = 104;
    data.power_cpu.resize(count);
    for (int index = 0; index < count; ++index) {
        const auto value = state.range_doppler[index];
        data.power_cpu[index] = value.real * value.real + value.imag * value.imag;
    }
    const auto cpu = cusignal::ca_cfar_typed_cpu<float>(
        data.power_cpu, {kNumPulses, kSamplesPerPulse}, options);
    data.threshold_cpu = cpu.threshold;
    data.detections_cpu = cpu.detections;
    data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(kPfa, reference_count);

    return data;
}

}  // namespace task1::accuracy
