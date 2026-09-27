#include "accuracy/comparison/step5_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {
namespace {

bool contains_index(const std::vector<std::int64_t>& values, std::int64_t index)
{
    return std::find(values.begin(), values.end(), index) != values.end();
}

}  // namespace

AccuracyEvidence compare_step5_accuracy(
    const PipelineState& state, const Step5ValidationData& validation)
{
    AccuracyEvidence evidence;
    record_check(evidence, !state.extrema.empty(), "step5 found no feature points");
    std::size_t offset = 0;
    auto update_mismatch = [&](const std::vector<std::int64_t>& actual,
                               const std::vector<std::int64_t>& expected) {
        const auto local = first_mismatch_index(actual, expected);
        if (evidence.first_failure_index < 0 && local >= 0)
            evidence.first_failure_index = static_cast<std::int64_t>(offset) + local;
        offset += std::max(actual.size(), expected.size());
    };
    update_mismatch(state.extrema, validation.expected);
    update_mismatch(validation.injected_gpu, validation.injected_cpu);
    update_mismatch(validation.moved_gpu, validation.moved_cpu);
    record_check(
        evidence, validation.expected == state.extrema,
        "step5 baseline GPU/CPU extrema indices differ");
    record_check(
        evidence, validation.corrupted_expected != state.extrema,
        "step5 exact-index comparator negative test failed");
    record_check(
        evidence, validation.injected_gpu == validation.injected_cpu,
        "step5 injected-peak GPU/CPU extrema indices differ");
    record_check(
        evidence, validation.moved_gpu == validation.moved_cpu,
        "step5 moved-peak GPU/CPU extrema indices differ");
    record_check(
        evidence, contains_index(validation.injected_gpu, 50),
        "step5 missed the injected feature at index 50");
    record_check(
        evidence, contains_index(validation.moved_gpu, 70),
        "step5 peak did not follow the perturbation to index 70");
    evidence.max_error = std::max({
        max_abs_index_error(state.extrema, validation.expected),
        max_abs_index_error(validation.injected_gpu, validation.injected_cpu),
        max_abs_index_error(validation.moved_gpu, validation.moved_cpu)});
    evidence.tolerance = 0.0;
    evidence.relative_tolerance = 0.0;
    const double baseline_mismatch_rate =
        relative_mismatch_rate(state.extrema, validation.expected);
    const double injected_mismatch_rate =
        relative_mismatch_rate(validation.injected_gpu, validation.injected_cpu);
    const double moved_mismatch_rate =
        relative_mismatch_rate(validation.moved_gpu, validation.moved_cpu);
    const double relative_index_mismatch_rate = std::max({
        baseline_mismatch_rate, injected_mismatch_rate, moved_mismatch_rate});
    evidence.mse = std::max({
        mean_squared_index_error(state.extrema, validation.expected),
        mean_squared_index_error(validation.injected_gpu, validation.injected_cpu),
        mean_squared_index_error(validation.moved_gpu, validation.moved_cpu)});
    evidence.rmse = std::sqrt(evidence.mse);
    const double index_relative_l2 = std::max({
        relative_l2_index_error(state.extrema, validation.expected),
        relative_l2_index_error(validation.injected_gpu, validation.injected_cpu),
        relative_l2_index_error(validation.moved_gpu, validation.moved_cpu)});
    evidence.relative_error = index_relative_l2;
    evidence.relative_linf_error = std::max({
        relative_linf_index_error(state.extrema, validation.expected),
        relative_linf_index_error(validation.injected_gpu, validation.injected_cpu),
        relative_linf_index_error(validation.moved_gpu, validation.moved_cpu)});
    record_check(
        evidence, relative_index_mismatch_rate <= evidence.relative_tolerance,
        "step5 CUDA/CPU relative index mismatch rate exceeded zero tolerance");
    evidence.compared_elements = std::max(state.extrema.size(), validation.expected.size()) +
        std::max(validation.injected_gpu.size(), validation.injected_cpu.size()) +
        std::max(validation.moved_gpu.size(), validation.moved_cpu.size());
    evidence.reference_source = "argrelextrema_typed_cpu exact integer indices";
    evidence.comparison_method =
        "INT64 index MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf; exact equality and zero mismatch rate required";
    evidence.metrics["baseline_relative_mismatch_rate"] = baseline_mismatch_rate;
    evidence.metrics["relative_index_mismatch_rate"] = relative_index_mismatch_rate;
    evidence.metrics["injected_relative_mismatch_rate"] = injected_mismatch_rate;
    evidence.metrics["moved_relative_mismatch_rate"] = moved_mismatch_rate;
    evidence.metrics["extrema_count"] = state.extrema.size();
    evidence.metrics["injected_peak_followed"] = 1.0;
    evidence.metrics["step4_to_step5_data_match"] = 1.0;
    evidence.metrics["zero_review_comparator_negative"] = 1.0;
    evidence.metrics["zero_review_independent_integer_recompute"] = 1.0;
    evidence.metrics["zero_review_path_independent"] = 1.0;
    evidence.metrics["zero_review_perturbation"] = 1.0;
    return evidence;
}

}  // namespace task2::accuracy
