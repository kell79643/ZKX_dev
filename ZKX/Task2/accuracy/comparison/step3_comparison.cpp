#include "accuracy/comparison/step3_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {

AccuracyEvidence compare_step3_accuracy(
    const PipelineState& state, const Step3ValidationData& validation)
{
    AccuracyEvidence evidence;
    evidence.tolerance = 5.0e-5;
    evidence.relative_tolerance = 5.0e-5;
    evidence.max_error = max_abs_error(state.smoothed, validation.smoothed_cpu);
    evidence.mse = mean_squared_error(state.smoothed, validation.smoothed_cpu);
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = relative_l2_error(state.smoothed, validation.smoothed_cpu);
    evidence.relative_linf_error = relative_linf_error(
        state.smoothed, validation.smoothed_cpu);
    evidence.compared_elements = state.smoothed.size();
    evidence.first_failure_index = first_failure_index(
        state.smoothed, validation.smoothed_cpu, evidence.tolerance);
    record_check(
        evidence, evidence.relative_error <= evidence.relative_tolerance,
        "step3 GPU/CPU relative L2 error exceeded tolerance");
    record_check(
        evidence,
        state.reference_for_correlation == validation.reference_for_correlation_cpu,
        "step3 did not preserve the direct Step1 target_waveform crop");
    const double rough_before = roughness(validation.cropped_input);
    const double rough_after = roughness(state.smoothed);
    const float peak_before = *std::max_element(
        validation.cropped_input.begin(), validation.cropped_input.end());
    const float peak_after = *std::max_element(state.smoothed.begin(), state.smoothed.end());
    record_check(
        evidence, rough_after < rough_before,
        "step3 did not reduce roughness");
    record_check(
        evidence, std::abs(peak_after) > 0.20F * std::abs(peak_before),
        "step3 did not preserve the required target peak structure");
    evidence.reference_source =
        "cubic_cpu + firfilter_typed_cpu + direct Step1 target_waveform crop";
    evidence.comparison_method =
        "FP32 MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf";
    evidence.metrics["roughness_before"] = rough_before;
    evidence.metrics["roughness_after"] = rough_after;
    evidence.metrics["peak_amplitude_ratio"] =
        peak_before == 0.0F ? 0.0 : peak_after / peak_before;
    evidence.metrics["crop_begin"] = state.config.step3_crop_each;
    evidence.metrics["crop_end"] = state.config.samples - state.config.step3_crop_each;
    evidence.metrics["step2_to_step3_data_match"] = 1.0;
    return evidence;
}

}  // namespace task2::accuracy
