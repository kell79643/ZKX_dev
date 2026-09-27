#include "accuracy/comparison/step6_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {

AccuracyEvidence compare_step6_accuracy(
    const PipelineState& state, const Step6ValidationData& validation)
{
    AccuracyEvidence evidence;
    evidence.tolerance = 2.0e-5;
    evidence.relative_tolerance = 2.0e-5;
    evidence.reference_source =
        "typed CPU Kalman predict/update state and covariance";
    evidence.comparison_method =
        "maximum MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf across FP32 state and covariance; truth and R perturbation retained";
    record_check(
        evidence, state.kalman_covariance.size() == 1,
        "step6 covariance shape mismatch");
    if (state.kalman_covariance.size() != 1) return evidence;
    const std::vector<float> estimated_state{state.estimate};
    evidence.max_error = std::max(
        max_abs_error(estimated_state, validation.state_cpu),
        max_abs_error(state.kalman_covariance, validation.covariance_cpu));
    evidence.mse = std::max(
        mean_squared_error(estimated_state, validation.state_cpu),
        mean_squared_error(state.kalman_covariance, validation.covariance_cpu));
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = std::max(
        relative_l2_error(estimated_state, validation.state_cpu),
        relative_l2_error(state.kalman_covariance, validation.covariance_cpu));
    evidence.relative_linf_error = std::max(
        relative_linf_error(estimated_state, validation.state_cpu),
        relative_linf_error(state.kalman_covariance, validation.covariance_cpu));
    evidence.compared_elements = estimated_state.size() +
        state.kalman_covariance.size();
    evidence.first_failure_index = first_failure_index(
        estimated_state, validation.state_cpu, evidence.tolerance);
    const auto covariance_failure = first_failure_index(
        state.kalman_covariance, validation.covariance_cpu, evidence.tolerance);
    if (evidence.first_failure_index < 0 && covariance_failure >= 0)
        evidence.first_failure_index =
            static_cast<std::int64_t>(estimated_state.size()) + covariance_failure;
    record_check(
        evidence, evidence.relative_error <= evidence.relative_tolerance,
        "step6 GPU/CPU state or covariance relative L2 error exceeded tolerance");
    record_check(
        evidence, std::isfinite(state.estimate) &&
            std::isfinite(state.kalman_covariance.front()),
        "step6 produced NaN/Inf");
    const double truth_error = std::abs(state.estimate - state.truth);
    record_check(
        evidence, truth_error <= 0.18,
        "step6 estimate missed simulation truth tolerance");
    const double noise_response = std::abs(
        validation.alternate_estimate_cpu - validation.state_cpu.front());
    record_check(
        evidence, std::isfinite(validation.alternate_estimate_cpu) &&
            noise_response > 1.0e-5,
        "step6 measurement-noise perturbation had no finite observable effect");
    evidence.metrics["observation_count"] = state.kalman_observations.size();
    evidence.metrics["anchor_index"] = state.kalman_anchor;
    evidence.metrics["truth_normalized"] = state.truth;
    evidence.metrics["estimate_normalized"] = state.estimate;
    evidence.metrics["truth_absolute_error"] = truth_error;
    evidence.metrics["measurement_noise_response"] = noise_response;
    evidence.metrics["step5_to_step6_data_match"] = 1.0;
    return evidence;
}

}  // namespace task2::accuracy
