#include "accuracy/validation/step3_validation.h"

#include <cusignal/operators/bsplines/bsplines_typed.h>
#include <cusignal/operators/filtering/filtering_typed.h>

#include <numeric>

namespace task2::accuracy {

Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data;
    data.cropped_input.assign(
        state.filtered.begin() + state.config.step3_crop_each,
        state.filtered.end() - state.config.step3_crop_each);
    data.reference_for_correlation_cpu.assign(
        state.target_waveform.begin() + state.config.step3_crop_each,
        state.target_waveform.end() - state.config.step3_crop_each);
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    auto weights_cpu = cusignal::cubic_cpu(offsets);
    const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);
    for (float& value : weights_cpu) value /= sum;
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(data.cropped_input.size())};
    options.axis = -1;
    data.smoothed_cpu = cusignal::firfilter_typed_cpu(
        weights_cpu, data.cropped_input, options).y;
    return data;
}

}  // namespace task2::accuracy
