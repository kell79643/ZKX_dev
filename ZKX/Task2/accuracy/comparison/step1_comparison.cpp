#include "accuracy/comparison/step1_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {

AccuracyEvidence compare_step1_accuracy(
    const PipelineState& state, const Step1ValidationData& validation)
{
    AccuracyEvidence evidence;
    evidence.tolerance = 3.0e-5;
    evidence.relative_tolerance = 3.0e-5;
    evidence.max_error = std::max({
        max_abs_error(state.target_waveform, validation.chirp_cpu),
        max_abs_error(state.target_component, validation.target_cpu),
        max_abs_error(state.interference_component, validation.interference_cpu),
        max_abs_error(state.noise_component, validation.noise_cpu),
        max_abs_error(state.echo, validation.echo_cpu)});
    evidence.mse = std::max({
        mean_squared_error(state.target_waveform, validation.chirp_cpu),
        mean_squared_error(state.target_component, validation.target_cpu),
        mean_squared_error(state.interference_component, validation.interference_cpu),
        mean_squared_error(state.noise_component, validation.noise_cpu),
        mean_squared_error(state.echo, validation.echo_cpu)});
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = std::max({
        relative_l2_error(state.target_waveform, validation.chirp_cpu),
        relative_l2_error(state.target_component, validation.target_cpu),
        relative_l2_error(state.interference_component, validation.interference_cpu),
        relative_l2_error(state.noise_component, validation.noise_cpu),
        relative_l2_error(state.echo, validation.echo_cpu)});
    evidence.relative_linf_error = std::max({
        relative_linf_error(state.target_waveform, validation.chirp_cpu),
        relative_linf_error(state.target_component, validation.target_cpu),
        relative_linf_error(state.interference_component, validation.interference_cpu),
        relative_linf_error(state.noise_component, validation.noise_cpu),
        relative_linf_error(state.echo, validation.echo_cpu)});
    evidence.compared_elements = state.target_waveform.size() +
        state.target_component.size() + state.interference_component.size() +
        state.noise_component.size() + state.echo.size();
    std::size_t offset = 0;
    auto update_failure = [&](const std::vector<float>& actual,
                              const std::vector<float>& expected) {
        const auto local = first_failure_index(actual, expected, evidence.tolerance);
        if (evidence.first_failure_index < 0 && local >= 0)
            evidence.first_failure_index = static_cast<std::int64_t>(offset) + local;
        offset += actual.size();
    };
    update_failure(state.target_waveform, validation.chirp_cpu);
    update_failure(state.target_component, validation.target_cpu);
    update_failure(state.interference_component, validation.interference_cpu);
    update_failure(state.noise_component, validation.noise_cpu);
    update_failure(state.echo, validation.echo_cpu);
    record_check(
        evidence, evidence.relative_error <= evidence.relative_tolerance,
        "step1 GPU/CPU relative L2 error exceeded tolerance");
    const double perturbation_delta = max_abs_error(
        validation.changed_echo_cpu, validation.echo_cpu);
    record_check(
        evidence, perturbation_delta > 1.0e-3,
        "step1 seed/target-weight perturbation did not change output");
    evidence.reference_source =
        "typed CPU waveform APIs plus independent Host echo/noise formulas";
    evidence.comparison_method =
        "maximum MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf across five FP32 arrays";
    evidence.metrics["waveform_functions_selected"] = 4.0;
    evidence.metrics["target_rms"] = rms(state.target_component);
    evidence.metrics["interference_rms"] = rms(state.interference_component);
    evidence.metrics["noise_rms"] = rms(state.noise_component);
    evidence.metrics["perturbation_delta"] = perturbation_delta;
    return evidence;
}

}  // namespace task2::accuracy
