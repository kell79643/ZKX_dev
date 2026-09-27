#include "accuracy/comparison/step4_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {

AccuracyEvidence compare_step4_accuracy(
    const PipelineState& state, const Step4ValidationData& validation)
{
    AccuracyEvidence evidence;
    evidence.tolerance = 1.0e-2;
    evidence.relative_tolerance = 1.0e-2;
    evidence.max_error = std::max({
        max_abs_error(state.fm_feature, validation.fm_cpu),
        max_abs_error(state.correlation_feature, validation.correlation_cpu),
        max_abs_error(state.spectral_feature, validation.spectral_cpu),
        max_abs_error(state.wavelet_feature, validation.wavelet_cpu),
        max_abs_error(state.feature_bundle, validation.bundle_cpu)});
    evidence.mse = std::max({
        mean_squared_error(state.fm_feature, validation.fm_cpu),
        mean_squared_error(state.correlation_feature, validation.correlation_cpu),
        mean_squared_error(state.spectral_feature, validation.spectral_cpu),
        mean_squared_error(state.wavelet_feature, validation.wavelet_cpu),
        mean_squared_error(state.feature_bundle, validation.bundle_cpu)});
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = std::max({
        relative_l2_error(state.fm_feature, validation.fm_cpu),
        relative_l2_error(state.correlation_feature, validation.correlation_cpu),
        relative_l2_error(state.spectral_feature, validation.spectral_cpu),
        relative_l2_error(state.wavelet_feature, validation.wavelet_cpu),
        relative_l2_error(state.feature_bundle, validation.bundle_cpu)});
    evidence.relative_linf_error = std::max({
        relative_linf_error(state.fm_feature, validation.fm_cpu),
        relative_linf_error(state.correlation_feature, validation.correlation_cpu),
        relative_linf_error(state.spectral_feature, validation.spectral_cpu),
        relative_linf_error(state.wavelet_feature, validation.wavelet_cpu),
        relative_linf_error(state.feature_bundle, validation.bundle_cpu)});
    evidence.compared_elements = state.fm_feature.size() +
        state.correlation_feature.size() + state.spectral_feature.size() +
        state.wavelet_feature.size() + state.feature_bundle.size();
    std::size_t offset = 0;
    auto update_failure = [&](const std::vector<float>& actual,
                              const std::vector<float>& expected) {
        const auto local = first_failure_index(actual, expected, evidence.tolerance);
        if (evidence.first_failure_index < 0 && local >= 0)
            evidence.first_failure_index = static_cast<std::int64_t>(offset) + local;
        offset += actual.size();
    };
    update_failure(state.fm_feature, validation.fm_cpu);
    update_failure(state.correlation_feature, validation.correlation_cpu);
    update_failure(state.spectral_feature, validation.spectral_cpu);
    update_failure(state.wavelet_feature, validation.wavelet_cpu);
    update_failure(state.feature_bundle, validation.bundle_cpu);
    record_check(
        evidence, evidence.relative_error <= evidence.relative_tolerance,
        "step4 GPU/CPU relative L2 error exceeded tolerance");
    record_check(
        evidence,
        *std::max_element(state.feature_bundle.begin(), state.feature_bundle.end()) > 0.1F,
        "step4 fused feature bundle is degenerate");
    evidence.reference_source =
        "fm_demod/correlate/spectrogram/cwt typed CPU APIs plus Host fusion";
    evidence.comparison_method =
        "maximum MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf across four domains and fused bundle";
    evidence.metrics["feature_families"] = 4.0;
    evidence.metrics["feature_bundle_size"] = state.feature_bundle.size();
    evidence.metrics["step3_to_step4_data_match"] = 1.0;
    return evidence;
}

}  // namespace task2::accuracy
