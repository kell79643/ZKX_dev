#include "accuracy/comparison/step1_comparison.h"

#include "accuracy/accuracy_utils.h"

namespace task1::accuracy {

AccuracyEvidence compare_step1_accuracy(
    const PipelineState& state, const Step1ValidationData& validation)
{
    const auto& config = state.config;
    AccuracyEvidence evidence;
    evidence.relative_tolerance = 2.0e-5;
    evidence.relative_floor = 1.0e-12;
    evidence.max_abs_error = std::max({
        max_abs_error(state.waveform, validation.waveform_cpu),
        max_abs_error(state.noiseless_echo, validation.noiseless_cpu),
        max_abs_error(state.noise, validation.noise_cpu),
        max_abs_error(state.echo, validation.echo_cpu)});
    evidence.mse = std::max({
        mean_squared_error(state.waveform, validation.waveform_cpu),
        mean_squared_error(state.noiseless_echo, validation.noiseless_cpu),
        mean_squared_error(state.noise, validation.noise_cpu),
        mean_squared_error(state.echo, validation.echo_cpu)});
    evidence.rmse = std::sqrt(evidence.mse);
    evidence.compared_elements = state.waveform.size() + state.noiseless_echo.size() +
        state.noise.size() + state.echo.size();
    evidence.relative_error = std::max({
        relative_l2_error(state.waveform, validation.waveform_cpu, evidence.relative_floor),
        relative_l2_error(state.noiseless_echo, validation.noiseless_cpu, evidence.relative_floor),
        relative_l2_error(state.noise, validation.noise_cpu, evidence.relative_floor),
        relative_l2_error(state.echo, validation.echo_cpu, evidence.relative_floor)});
    evidence.relative_linf_error = std::max({
        relative_linf_error(state.waveform, validation.waveform_cpu, evidence.relative_floor),
        relative_linf_error(state.noiseless_echo, validation.noiseless_cpu, evidence.relative_floor),
        relative_linf_error(state.noise, validation.noise_cpu, evidence.relative_floor),
        relative_linf_error(state.echo, validation.echo_cpu, evidence.relative_floor)});
    require(evidence.relative_error <= evidence.relative_tolerance,
            "step1 GPU/CPU relative L2 error exceeds tolerance");
    require(max_abs_error(validation.changed_delay_cpu, validation.echo_cpu) > 0.1,
            "step1 delay perturbation did not change output");
    require(max_abs_error(validation.changed_doppler_cpu, validation.echo_cpu) > 0.1,
            "step1 Doppler perturbation did not change output");
    const double seed_perturbation_delta =
        max_abs_error(validation.changed_seed_cpu, validation.echo_cpu);
    const bool noise_enabled = config.noise_std > 0.0F;
    if (noise_enabled) {
        require(seed_perturbation_delta > 1.0e-3,
                "step1 seed perturbation did not change noisy output");
    } else {
        require(seed_perturbation_delta <= evidence.relative_floor,
                "step1 seed perturbation changed noiseless output");
    }
    const double target_range = static_cast<double>(config.target_delay_samples) * kSpeedOfLight /
        (2.0 * config.sample_rate_hz);
    const double noise_rms = rms_complex(state.noise);
    const double signal_rms = rms_complex(state.noiseless_echo);
    evidence.reference_source = "independent Host LFM/echo/noise formulas";
    evidence.comparison_method =
        "MSE, RMSE, relative L2 and CPU-amplitude-normalized relative L_inf for each complex output; maximum across outputs; delay/Doppler perturbations; seed sensitivity when noise is enabled and seed invariance when noise_std is zero";
    evidence.metrics["target_echo_present"] = signal_rms > 0.0 ? 1.0 : 0.0;
    evidence.metrics["delay_ground_truth_recorded"] = 1.0;
    evidence.metrics["doppler_ground_truth_recorded"] = 1.0;
    evidence.metrics["noise_added"] = noise_enabled ? 1.0 : 0.0;
    evidence.metrics["noise_free_case"] = noise_enabled ? 0.0 : 1.0;
    evidence.metrics["sample_rate_hz"] = config.sample_rate_hz;
    evidence.metrics["num_pulses"] = config.num_pulses;
    evidence.metrics["samples_per_pulse"] = config.samples_per_pulse;
    evidence.metrics["pulse_samples"] = config.pulse_samples;
    evidence.metrics["pulse_width_seconds"] =
        static_cast<double>(config.pulse_samples) / config.sample_rate_hz;
    evidence.metrics["bandwidth_hz"] = config.bandwidth_hz;
    evidence.metrics["prf_hz"] = config.prf_hz;
    evidence.metrics["pri_seconds"] = 1.0 / config.prf_hz;
    evidence.metrics["target_amplitude"] = config.target_amplitude;
    evidence.metrics["noise_std"] = config.noise_std;
    evidence.metrics["delay_ground_truth_samples"] = config.target_delay_samples;
    evidence.metrics["target_range_m"] = target_range;
    evidence.metrics["doppler_ground_truth_hz"] =
        config.doppler_hz();
    evidence.metrics["noise_seed"] = config.noise_seed;
    evidence.metrics["noise_rms"] = noise_rms;
    evidence.metrics["seed_perturbation_delta"] = seed_perturbation_delta;
    evidence.metrics["seed_perturbation_expected_change"] = noise_enabled ? 1.0 : 0.0;
    evidence.metrics["snr_db_defined"] = noise_rms > 0.0 ? 1.0 : 0.0;
    if (noise_rms > 0.0)
        evidence.metrics["snr_db"] = 20.0 * std::log10(signal_rms / noise_rms);
    evidence.metrics["output_rows"] = config.num_pulses;
    evidence.metrics["output_columns"] = config.samples_per_pulse;
    evidence.metrics["perturbation_checks_pass"] = 3.0;
    return evidence;
}

}  // namespace task1::accuracy
