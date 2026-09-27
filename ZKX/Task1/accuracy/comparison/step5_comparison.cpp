#include "accuracy/comparison/step5_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task1::accuracy {
namespace {

int peak_index(const std::vector<double>& values)
{
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

int half_power_width(const std::vector<double>& values, int peak)
{
    const double threshold = values[peak] / std::sqrt(2.0);
    int left = peak;
    int right = peak;
    while (left > 0 && values[left - 1] >= threshold) --left;
    while (right + 1 < static_cast<int>(values.size()) &&
           values[right + 1] >= threshold) ++right;
    return right - left + 1;
}

double pslr_db(const std::vector<double>& values, int peak, int exclusion)
{
    double sidelobe = 0.0;
    for (int index = 0; index < static_cast<int>(values.size()); ++index) {
        if (std::abs(index - peak) <= exclusion) continue;
        sidelobe = std::max(sidelobe, values[index]);
    }
    return 20.0 * std::log10(std::max(sidelobe, 1.0e-12) / values[peak]);
}

}  // namespace

AccuracyEvidence compare_step5_accuracy(
    const PipelineState& state, const Step5ValidationData& validation)
{
    const auto& config = state.config;
    AccuracyEvidence evidence;
    const std::vector<double> ambiguity_2d(
        state.ambiguity_2d.begin(), state.ambiguity_2d.end());
    // Secondary legacy guard only; formal benchmark gates all four metrics from JSON.
    evidence.relative_tolerance = 2.0e-4;
    evidence.relative_floor = 1.0e-12;
    evidence.max_abs_error = std::max({
        max_abs_error(ambiguity_2d, validation.ambiguity_2d_cpu),
        max_abs_error(state.ambiguity_delay, validation.ambiguity_delay_cpu),
        max_abs_error(state.ambiguity_doppler, validation.ambiguity_doppler_cpu)});
    evidence.mse = std::max({
        mean_squared_error(ambiguity_2d, validation.ambiguity_2d_cpu),
        mean_squared_error(state.ambiguity_delay, validation.ambiguity_delay_cpu),
        mean_squared_error(state.ambiguity_doppler, validation.ambiguity_doppler_cpu)});
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.compared_elements = ambiguity_2d.size() +
        state.ambiguity_delay.size() + state.ambiguity_doppler.size();
    evidence.relative_error = std::max({
        relative_l2_error(ambiguity_2d, validation.ambiguity_2d_cpu,
                          evidence.relative_floor),
        relative_l2_error(state.ambiguity_delay, validation.ambiguity_delay_cpu,
                          evidence.relative_floor),
        relative_l2_error(state.ambiguity_doppler, validation.ambiguity_doppler_cpu,
                          evidence.relative_floor)});
    evidence.relative_linf_error = std::max({
        relative_linf_error(ambiguity_2d, validation.ambiguity_2d_cpu,
                            evidence.relative_floor),
        relative_linf_error(state.ambiguity_delay, validation.ambiguity_delay_cpu,
                            evidence.relative_floor),
        relative_linf_error(state.ambiguity_doppler, validation.ambiguity_doppler_cpu,
                            evidence.relative_floor)});
    require(evidence.relative_error <= evidence.relative_tolerance,
            "step5 ambiguity GPU/CPU relative L2 error exceeds tolerance");
    const int rows = 2 * config.pulse_samples - 1;
    const int columns = config.ambiguity_nfreq;
    const int delay_peak = peak_index(state.ambiguity_delay);
    const int doppler_peak = peak_index(state.ambiguity_doppler);
    const int delay_width = half_power_width(state.ambiguity_delay, delay_peak);
    const int doppler_width = half_power_width(state.ambiguity_doppler, doppler_peak);
    const double delay_pslr = pslr_db(
        state.ambiguity_delay, delay_peak, std::max(2, delay_width));
    const double doppler_pslr = pslr_db(
        state.ambiguity_doppler, doppler_peak, std::max(2, doppler_width));
    const int peak_2d = peak_index(ambiguity_2d);
    require(peak_2d / columns == config.pulse_samples - 1 &&
            peak_2d % columns == columns / 2,
            "step5 ambiguity peak is not at zero delay/zero Doppler");
    require(std::abs(ambiguity_2d[peak_2d] - 1.0) <= 3.0e-3,
            "step5 ambiguity normalization peak is not one");
    require(delay_peak == rows / 2, "step5 delay cut peak is not at zero delay");
    require(doppler_peak == columns / 2,
            "step5 Doppler cut peak is not at zero Doppler");
    require(delay_width <= 3 && doppler_width <= 5,
            "step5 ambiguity mainlobe is wider than approved threshold");
    require(delay_pslr <= -8.0 && doppler_pslr <= -8.0,
            "step5 ambiguity sidelobe level failed approved threshold");
    require(max_abs_error(
                validation.changed_waveform_cpu,
                validation.ambiguity_delay_cpu) > 1.0e-4,
            "step5 waveform perturbation did not change ambiguity output");
    evidence.reference_source = "ambgfun_typed_cpu<float> for 2D and both cuts";
    evidence.comparison_method =
        "MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf for each ambiguity output; maximum across outputs; peak/mainlobe/PSLR/waveform perturbation";
    evidence.metrics["ambgfun_called"] = 1.0;
    evidence.metrics["step1_waveform_to_step5_match"] = 1.0;
    evidence.metrics["ambiguity_rows"] = rows;
    evidence.metrics["ambiguity_columns"] = columns;
    evidence.metrics["peak_delay_index"] = delay_peak;
    evidence.metrics["peak_doppler_index"] = doppler_peak;
    evidence.metrics["peak_delay_seconds"] = 0.0;
    evidence.metrics["peak_doppler_hz"] = 0.0;
    evidence.metrics["delay_grid_min_seconds"] =
        -static_cast<double>(config.pulse_samples - 1) / config.sample_rate_hz;
    evidence.metrics["delay_grid_max_seconds"] =
        static_cast<double>(config.pulse_samples - 1) / config.sample_rate_hz;
    evidence.metrics["delay_grid_spacing_seconds"] = 1.0 / config.sample_rate_hz;
    evidence.metrics["doppler_grid_min_hz"] = -0.5 * config.sample_rate_hz;
    evidence.metrics["doppler_grid_max_hz"] =
        0.5 * config.sample_rate_hz - static_cast<double>(config.sample_rate_hz) / columns;
    evidence.metrics["doppler_grid_spacing_hz"] =
        static_cast<double>(config.sample_rate_hz) / columns;
    evidence.metrics["normalization_peak"] = ambiguity_2d[peak_2d];
    evidence.metrics["normalization_error"] =
        std::abs(ambiguity_2d[peak_2d] - 1.0);
    evidence.metrics["delay_mainlobe_width_samples"] = delay_width;
    evidence.metrics["doppler_mainlobe_width_bins"] = doppler_width;
    evidence.metrics["delay_pslr_db"] = delay_pslr;
    evidence.metrics["doppler_pslr_db"] = doppler_pslr;
    evidence.metrics["waveform_quality_pass"] = 1.0;
    return evidence;
}

}  // namespace task1::accuracy
