#include "accuracy/validation/step2_validation.h"

#include <cusignal/operators/filter_design/filter_design_typed.h>
#include <cusignal/operators/filtering/filtering_typed.h>
#include <cusignal/operators/windows/windows_typed.h>

namespace task2::accuracy {

Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data;
    constexpr int taps_count = kFilterTaps;
    const std::vector<float> cutoff{kFilterCutoffHz};
    const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);
    cusignal::FirwinOptions design_options;
    design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;
    design_options.window = window_cpu;
    design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;
    design_options.scale = true;
    design_options.fs = kSampleRate;
    const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));
    cusignal::FirfilterOptions filter_options;
    filter_options.shape = {kSamples};
    filter_options.axis = -1;
    data.filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.echo, filter_options).y;
    data.target_filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.target_component, filter_options).y;
    return data;
}

}  // namespace task2::accuracy
