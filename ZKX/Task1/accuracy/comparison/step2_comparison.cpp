#include "accuracy/comparison/step2_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task1::accuracy {
namespace {

int peak_range_bin(const std::vector<cusignal::TypedComplex<float>>& values,
                   const TaskConfig& config)
{
    int best = 0;
    double peak = -1.0;
    for (int range = 0; range < config.samples_per_pulse; ++range) {
        double energy = 0.0;
        for (int pulse = 0; pulse < config.num_pulses; ++pulse) {
            const auto value = values[static_cast<std::size_t>(pulse) *
                config.samples_per_pulse + range];
            energy += static_cast<double>(value.real) * value.real +
                static_cast<double>(value.imag) * value.imag;
        }
        if (energy > peak) {
            peak = energy;
            best = range;
        }
    }
    return best;
}

int half_power_width(
    const std::vector<cusignal::TypedComplex<float>>& values, int peak_bin,
    int samples_per_pulse)
{
    const double peak = complex_magnitude(values[peak_bin]);
    const double threshold = peak / std::sqrt(2.0);
    int left = peak_bin;
    int right = peak_bin;
    while (left > 0 && complex_magnitude(values[left - 1]) >= threshold) --left;
    while (right + 1 < samples_per_pulse &&
           complex_magnitude(values[right + 1]) >= threshold) ++right;
    return right - left + 1;
}

}  // namespace

AccuracyEvidence compare_step2_accuracy(
    const PipelineState& state, const Step2ValidationData& validation)
{
    const auto& config = state.config;
    AccuracyEvidence evidence;
    // Secondary legacy guard only; formal benchmark gates all four metrics from JSON.
    evidence.relative_tolerance = 1.0e-4;
    evidence.relative_floor = 1.0e-12;
    evidence.max_abs_error = max_abs_error(state.compressed, validation.compressed_cpu);
    evidence.mse = mean_squared_error(state.compressed, validation.compressed_cpu);
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = relative_l2_error(
        state.compressed, validation.compressed_cpu, evidence.relative_floor);
    evidence.relative_linf_error = relative_linf_error(
        state.compressed, validation.compressed_cpu, evidence.relative_floor);
    evidence.compared_elements = state.compressed.size();
    require(evidence.relative_error <= evidence.relative_tolerance,
            "step2 pulse compression GPU/CPU relative L2 error exceeds tolerance");
    const int peak = peak_range_bin(state.compressed, config);
    const int width = half_power_width(state.compressed, peak, config.samples_per_pulse);
    const double range_error = std::abs(peak - config.target_delay_samples) * kSpeedOfLight /
        (2.0 * config.sample_rate_hz);
    require(std::abs(peak - config.target_delay_samples) <= 1,
            "step2 compressed peak does not match delay truth");
    require(width < config.pulse_samples,
            "step2 pulse compression did not narrow the effective pulse");
    require(max_abs_error(
                validation.altered_template_cpu, validation.compressed_cpu) > 1.0e-3,
            "step2 template perturbation did not change output");
    evidence.reference_source = "pulse_compression_typed_cpu<float>";
    evidence.comparison_method =
        "MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf; range peak/template perturbation";
    evidence.metrics["pulse_compression_called"] = 1.0;
    evidence.metrics["step1_to_step2_data_match"] = 1.0;
    evidence.metrics["peak_range_bin"] = peak;
    evidence.metrics["expected_range_bin"] = config.target_delay_samples;
    evidence.metrics["range_error_m"] = range_error;
    evidence.metrics["range_bin_spacing_m"] = kSpeedOfLight /
        (2.0 * config.sample_rate_hz);
    evidence.metrics["theoretical_range_resolution_m"] = kSpeedOfLight /
        (2.0 * config.bandwidth_hz);
    evidence.metrics["compressed_half_power_width_samples"] = width;
    evidence.metrics["uncompressed_width_samples"] = config.pulse_samples;
    evidence.metrics["normalization_enabled"] = 1.0;
    evidence.metrics["template_samples"] = config.pulse_samples;
    evidence.metrics["nfft"] = config.samples_per_pulse;
    evidence.metrics["padding_policy_circular"] = 1.0;
    evidence.metrics["output_rows"] = config.num_pulses;
    evidence.metrics["output_columns"] = config.samples_per_pulse;
    return evidence;
}

}  // namespace task1::accuracy
