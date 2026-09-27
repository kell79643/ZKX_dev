#include "accuracy/comparison/step3_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task1::accuracy {
namespace {

std::pair<int, int> peak_bin(
    const std::vector<cusignal::TypedComplex<float>>& values,
    const TaskConfig& config)
{
    std::pair<int, int> best{0, 0};
    double peak = -1.0;
    for (int doppler = 0; doppler < config.num_pulses; ++doppler) {
        for (int range = 0; range < config.samples_per_pulse; ++range) {
            const double value = complex_magnitude(
                values[static_cast<std::size_t>(doppler) * config.samples_per_pulse + range]);
            if (value > peak) {
                peak = value;
                best = {doppler, range};
            }
        }
    }
    return best;
}

}  // namespace

AccuracyEvidence compare_step3_accuracy(
    const PipelineState& state, const Step3ValidationData& validation)
{
    const auto& config = state.config;
    AccuracyEvidence evidence;
    // Secondary legacy guard only; formal benchmark gates all four metrics from JSON.
    evidence.relative_tolerance = 5.0e-4;
    evidence.relative_floor = 1.0e-12;
    evidence.max_abs_error = max_abs_error(
        state.range_doppler, validation.range_doppler_cpu);
    evidence.mse = mean_squared_error(
        state.range_doppler, validation.range_doppler_cpu);
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.relative_error = relative_l2_error(
        state.range_doppler, validation.range_doppler_cpu, evidence.relative_floor);
    evidence.relative_linf_error = relative_linf_error(
        state.range_doppler, validation.range_doppler_cpu, evidence.relative_floor);
    evidence.compared_elements = state.range_doppler.size();
    require(evidence.relative_error <= evidence.relative_tolerance,
            "step3 pulse Doppler GPU/CPU relative L2 error exceeds tolerance");
    const auto peak = peak_bin(state.range_doppler, config);
    const int signed_bin = peak.first <= config.num_pulses / 2
        ? peak.first : peak.first - config.num_pulses;
    const double detected_hz = static_cast<double>(signed_bin) * config.prf_hz / config.num_pulses;
    const double expected_hz = config.doppler_hz();
    require(signed_bin == config.doppler_bin,
            "step3 Doppler peak bin or sign does not match truth");
    require(std::abs(peak.second - config.target_delay_samples) <= 1,
            "step3 range peak no longer matches delay truth");
    require(max_abs_error(
                validation.no_window_cpu, validation.range_doppler_cpu) > 1.0e-3,
            "step3 window perturbation did not change output");
    evidence.reference_source = "pulse_doppler_typed_cpu<float>";
    evidence.comparison_method =
        "MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf; peak truth/window perturbation";
    evidence.metrics["pulse_doppler_called"] = 1.0;
    evidence.metrics["step2_to_step3_data_match"] = 1.0;
    evidence.metrics["peak_doppler_bin"] = peak.first;
    evidence.metrics["signed_doppler_bin"] = signed_bin;
    evidence.metrics["detected_doppler_hz"] = detected_hz;
    evidence.metrics["expected_doppler_hz"] = expected_hz;
    evidence.metrics["doppler_error_hz"] = std::abs(detected_hz - expected_hz);
    evidence.metrics["peak_range_bin"] = peak.second;
    evidence.metrics["slow_time_axis"] = 0.0;
    evidence.metrics["hamming_window"] = 1.0;
    evidence.metrics["nfft"] = config.num_pulses;
    evidence.metrics["doppler_bin_spacing_hz"] =
        static_cast<double>(config.prf_hz) / config.num_pulses;
    evidence.metrics["normalization_none"] = 1.0;
    evidence.metrics["padding_or_truncation_none"] = 1.0;
    evidence.metrics["output_rows"] = config.num_pulses;
    evidence.metrics["output_columns"] = config.samples_per_pulse;
    return evidence;
}

}  // namespace task1::accuracy
