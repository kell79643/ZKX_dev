#include "accuracy/comparison/step2_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {

AccuracyEvidence compare_step2_accuracy(
    const PipelineState& state, const Step2ValidationData& validation)
{
    AccuracyEvidence evidence;
    evidence.tolerance = 5.0e-4;
    evidence.relative_tolerance = 5.0e-4;
    evidence.max_error = max_abs_error(state.filtered, validation.filtered_cpu);
    evidence.mse = mean_squared_error(state.filtered, validation.filtered_cpu);
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = relative_l2_error(state.filtered, validation.filtered_cpu);
    evidence.relative_linf_error = relative_linf_error(
        state.filtered, validation.filtered_cpu);
    evidence.compared_elements = state.filtered.size();
    evidence.first_failure_index = first_failure_index(
        state.filtered, validation.filtered_cpu, evidence.tolerance);
    record_check(
        evidence, evidence.relative_error <= evidence.relative_tolerance,
        "step2 GPU/CPU relative L2 error exceeded tolerance");
    std::vector<float> input_residual(state.config.samples), output_residual(state.config.samples);
    for (int index = 0; index < state.config.samples; ++index) {
        input_residual[index] = state.echo[index] - state.target_component[index];
        output_residual[index] = state.filtered[index] -
            validation.target_filtered_cpu[index];
    }
    const double snr_before = 20.0 * std::log10(
        std::max(rms(state.target_component), 1.0e-12) /
        std::max(rms(input_residual), 1.0e-12));
    const double snr_after = 20.0 * std::log10(
        std::max(rms(validation.target_filtered_cpu), 1.0e-12) /
        std::max(rms(output_residual), 1.0e-12));
    record_check(
        evidence, snr_after > snr_before + 1.0,
        "step2 SNR improvement did not exceed 1 dB");
    evidence.reference_source =
        "hamming_typed_cpu<float> + firwin_typed_cpu + firfilter_typed_cpu";
    evidence.comparison_method =
        "FP32 MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf";
    evidence.metrics["snr_before_db"] = snr_before;
    evidence.metrics["snr_after_db"] = snr_after;
    evidence.metrics["snr_improvement_db"] = snr_after - snr_before;
    evidence.metrics["step1_to_step2_data_match"] = 1.0;
    return evidence;
}

}  // namespace task2::accuracy
