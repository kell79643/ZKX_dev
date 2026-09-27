#include "pipeline_visualization_export.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>

namespace task2 {
namespace {

#ifndef TASK2_EVIDENCE_GIT_COMMIT
#define TASK2_EVIDENCE_GIT_COMMIT "unknown"
#endif

std::ofstream open_csv(const std::string& directory, const std::string& name)
{
    std::ofstream output(directory + "/" + name);
    require(static_cast<bool>(output), "cannot write Task2 visualization CSV: " + name);
    output << std::setprecision(17);
    return output;
}

void write_feature(
    std::ofstream& output, const char* family, const std::vector<float>& values)
{
    const double denominator = values.size() > 1 ? values.size() - 1.0 : 1.0;
    for (std::size_t index = 0; index < values.size(); ++index)
        output << family << ',' << index << ',' << index / denominator << ','
               << values[index] << '\n';
}

}  // namespace

void export_visualization_data(const PipelineState& state, int completed_steps)
{
    const char* raw_directory = std::getenv("ZKX_VISUALIZATION_DIR");
    if (raw_directory == nullptr || raw_directory[0] == '\0') return;
    const std::string directory(raw_directory);
    const auto& config = state.config;

    {
        auto output = open_csv(directory, "task2_manifest.csv");
        output << "key,value\n"
               << "task,Task2\n"
               << "source,cusignal_cpp_zq500_actual_pipeline\n"
               << "git_commit," << TASK2_EVIDENCE_GIT_COMMIT << "\n"
               << "completed_steps," << completed_steps << "\n"
               << "config_path," << config.path.string() << "\n"
               << "config_sha256," << config.sha256 << "\n"
               << "samples," << config.samples << "\n"
               << "filter_taps," << config.filter_taps << "\n"
               << "step3_crop_each," << config.step3_crop_each << "\n"
               << "step3_output_samples," << config.step3_samples() << "\n"
               << "kalman_observations," << config.kalman_observation_count << "\n"
               << "sample_rate_hz," << config.sample_rate_hz << "\n"
               << "target_delay_samples," << config.target_delay_samples << "\n"
               << "noise_seed," << config.noise_seed << "\n"
               << "step2_input,step1.echo\n"
               << "step3_input,step2.filtered\n"
               << "step4_input,step3.smoothed+step3.reference\n"
               << "step5_input,step4.feature_bundle\n"
               << "step6_input,step5.extrema+step4.feature_bundle\n";
    }

    if (completed_steps >= 1) {
        require(state.time.size() == static_cast<std::size_t>(config.samples) && state.target_waveform.size() == static_cast<std::size_t>(config.samples) &&
                    state.gaussian_waveform.size() == static_cast<std::size_t>(config.samples) &&
                    state.sawtooth_waveform.size() == static_cast<std::size_t>(config.samples) &&
                    state.square_waveform.size() == static_cast<std::size_t>(config.samples),
                "Task2 visualization base-waveform shape mismatch");
        auto waveforms = open_csv(directory, "task2_step1_waveforms.csv");
        waveforms << "sample,time_seconds,chirp,gausspulse,sawtooth,square\n";
        for (int index = 0; index < config.samples; ++index)
            waveforms << index << ',' << state.time[index] << ','
                      << state.target_waveform[index] << ','
                      << state.gaussian_waveform[index] << ','
                      << state.sawtooth_waveform[index] << ','
                      << state.square_waveform[index] << '\n';

        require(state.target_component.size() == static_cast<std::size_t>(config.samples) &&
                    state.interference_component.size() == static_cast<std::size_t>(config.samples) &&
                    state.noise_component.size() == static_cast<std::size_t>(config.samples) && state.echo.size() == static_cast<std::size_t>(config.samples),
                "Task2 visualization echo shape mismatch");
        auto echo = open_csv(directory, "task2_step1_echo.csv");
        echo << "sample,time_seconds,target,interference,noise,echo\n";
        for (int index = 0; index < config.samples; ++index)
            echo << index << ',' << state.time[index] << ','
                 << state.target_component[index] << ','
                 << state.interference_component[index] << ','
                 << state.noise_component[index] << ',' << state.echo[index] << '\n';
    }

    if (completed_steps >= 2) {
        require(state.echo.size() == static_cast<std::size_t>(config.samples) && state.filtered.size() == static_cast<std::size_t>(config.samples) &&
                    state.filter_taps.size() == static_cast<std::size_t>(config.filter_taps),
                "Task2 visualization filter shape mismatch");
        auto signal = open_csv(directory, "task2_step2_filter.csv");
        signal << "sample,time_seconds,input_echo,filtered\n";
        for (int index = 0; index < config.samples; ++index)
            signal << index << ',' << state.time[index] << ',' << state.echo[index]
                   << ',' << state.filtered[index] << '\n';
        auto taps = open_csv(directory, "task2_step2_taps.csv");
        taps << "tap,relative_sample,value\n";
        for (std::size_t index = 0; index < state.filter_taps.size(); ++index)
            taps << index << ',' << static_cast<int>(index) - (config.filter_taps - 1) / 2 << ','
                 << state.filter_taps[index] << '\n';
    }

    if (completed_steps >= 3) {
        require(state.filtered.size() == static_cast<std::size_t>(config.samples) && state.smoothed.size() == static_cast<std::size_t>(config.step3_samples()) &&
                    state.spline_weights.size() == 9,
                "Task2 visualization smoothing shape mismatch");
        auto signal = open_csv(directory, "task2_step3_smoothing.csv");
        signal << "sample,source_sample,time_seconds,cropped_input,smoothed\n";
        for (std::size_t index = 0; index < state.smoothed.size(); ++index) {
            const std::size_t source = index + config.step3_crop_each;
            signal << index << ',' << source << ',' << state.time[source] << ','
                   << state.filtered[source] << ',' << state.smoothed[index] << '\n';
        }
        auto weights = open_csv(directory, "task2_step3_spline_weights.csv");
        weights << "tap,offset,value\n";
        for (std::size_t index = 0; index < state.spline_weights.size(); ++index)
            weights << index << ',' << -2.0 + 0.5 * index << ','
                    << state.spline_weights[index] << '\n';
    }

    if (completed_steps >= 4) {
        require(!state.fm_feature.empty() && !state.correlation_feature.empty() &&
                    !state.spectral_feature.empty() && !state.wavelet_feature.empty() &&
                    state.feature_bundle.size() == static_cast<std::size_t>(config.step3_samples()),
                "Task2 visualization feature vectors are incomplete");
        auto features = open_csv(directory, "task2_step4_features.csv");
        features << "family,index,normalized_position,value\n";
        write_feature(features, "fm_demod", state.fm_feature);
        write_feature(features, "correlation", state.correlation_feature);
        write_feature(features, "spectral_envelope", state.spectral_feature);
        write_feature(features, "wavelet_envelope", state.wavelet_feature);
        write_feature(features, "fused_bundle", state.feature_bundle);

        require(state.spectrogram_frequency_bins > 0 && state.spectrogram_frames > 0 &&
                    state.spectrogram_grid.size() == static_cast<std::size_t>(
                        state.spectrogram_frequency_bins * state.spectrogram_frames),
                "Task2 visualization spectrogram shape mismatch");
        auto spectrogram = open_csv(directory, "task2_step4_spectrogram.csv");
        spectrogram << "frequency_index,frequency_hz,frame_index,time_seconds,value\n";
        for (int frequency = 0; frequency < state.spectrogram_frequency_bins; ++frequency) {
            const int signed_bin = frequency <= state.spectrogram_frequency_bins / 2
                ? frequency : frequency - state.spectrogram_frequency_bins;
            const double frequency_hz = static_cast<double>(signed_bin) * config.sample_rate_hz /
                state.spectrogram_frequency_bins;
            for (int frame = 0; frame < state.spectrogram_frames; ++frame) {
                const double time_seconds = static_cast<double>(
                    config.step3_crop_each + frame * 32 + 32) /
                    config.sample_rate_hz;
                spectrogram << frequency << ',' << frequency_hz << ',' << frame << ','
                            << time_seconds << ',' << state.spectrogram_grid[
                                static_cast<std::size_t>(frequency) *
                                state.spectrogram_frames + frame] << '\n';
            }
        }

        require(state.cwt_width_count == 4 && state.cwt_grid.size() ==
                    static_cast<std::size_t>(state.cwt_width_count) * state.smoothed.size(),
                "Task2 visualization CWT shape mismatch");
        const float widths[] = {2.0F, 4.0F, 8.0F, 12.0F};
        auto cwt = open_csv(directory, "task2_step4_cwt.csv");
        cwt << "width_index,width,sample,time_seconds,value\n";
        for (int width = 0; width < state.cwt_width_count; ++width) {
            for (std::size_t sample = 0; sample < state.smoothed.size(); ++sample) {
                cwt << width << ',' << widths[width] << ',' << sample << ','
                    << state.time[sample + config.step3_crop_each] << ',' << state.cwt_grid[
                        static_cast<std::size_t>(width) * state.smoothed.size() + sample]
                    << '\n';
            }
        }
    }

    if (completed_steps >= 5) {
        require(!state.feature_bundle.empty() && !state.extrema.empty(),
                "Task2 visualization extrema are incomplete");
        auto output = open_csv(directory, "task2_step5_extrema.csv");
        output << "order,index,normalized_position,value\n";
        for (std::size_t order = 0; order < state.extrema.size(); ++order) {
            const auto index = state.extrema[order];
            require(index >= 0 && index < static_cast<std::int64_t>(state.feature_bundle.size()),
                    "Task2 visualization extrema index out of range");
            output << order << ',' << index << ','
                   << static_cast<double>(index) / (state.feature_bundle.size() - 1) << ','
                   << state.feature_bundle[static_cast<std::size_t>(index)] << '\n';
        }
    }

    if (completed_steps >= 6) {
        require(state.kalman_observations.size() == static_cast<std::size_t>(config.kalman_observation_count) &&
                    state.kalman_covariance.size() == 1,
                "Task2 visualization Kalman result is incomplete");
        auto output = open_csv(directory, "task2_step6_kalman.csv");
        output << "observation_index,observation,anchor_index,estimate,truth,covariance\n";
        for (std::size_t index = 0; index < state.kalman_observations.size(); ++index)
            output << index << ',' << state.kalman_observations[index] << ','
                   << state.kalman_anchor << ',' << state.estimate << ',' << state.truth
                   << ',' << state.kalman_covariance.front() << '\n';
    }
}

}  // namespace task2
