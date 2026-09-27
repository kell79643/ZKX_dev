#include "pipeline_visualization_export.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>

namespace task1 {
namespace {

#ifndef TASK1_EVIDENCE_GIT_COMMIT
#define TASK1_EVIDENCE_GIT_COMMIT "unknown"
#endif

std::ofstream open_csv(const std::string& directory, const std::string& name)
{
    std::ofstream output(directory + "/" + name);
    require(static_cast<bool>(output), "cannot write Task1 visualization CSV: " + name);
    output << std::setprecision(17);
    return output;
}

double magnitude(const cusignal::TypedComplex<float>& value)
{
    return std::hypot(static_cast<double>(value.real),
                      static_cast<double>(value.imag));
}

double power_db(double value)
{
    return 10.0 * std::log10(std::max(value, 1.0e-20));
}

}  // namespace

void export_visualization_data(const PipelineState& state, int completed_steps)
{
    const char* raw_directory = std::getenv("ZKX_VISUALIZATION_DIR");
    if (raw_directory == nullptr || raw_directory[0] == '\0') return;
    const std::string directory(raw_directory);
    const auto& config = state.config;

    {
        auto output = open_csv(directory, "task1_manifest.csv");
        output << "key,value\n"
               << "task,Task1\n"
               << "source,cusignal_cpp_zq500_actual_pipeline\n"
               << "git_commit," << TASK1_EVIDENCE_GIT_COMMIT << "\n"
               << "completed_steps," << completed_steps << "\n"
               << "config_path," << config.path.string() << "\n"
               << "config_sha256," << config.sha256 << "\n"
               << "num_pulses," << config.num_pulses << "\n"
               << "samples_per_pulse," << config.samples_per_pulse << "\n"
               << "pulse_samples," << config.pulse_samples << "\n"
               << "sample_rate_hz," << config.sample_rate_hz << "\n"
               << "prf_hz," << config.prf_hz << "\n"
               << "target_delay_samples," << config.target_delay_samples << "\n"
               << "target_doppler_bin," << config.doppler_bin << "\n"
               << "target_doppler_hz," << config.doppler_hz() << "\n"
               << "bandwidth_hz," << config.bandwidth_hz << "\n"
               << "carrier_frequency_hz," << config.carrier_frequency_hz << "\n"
               << "noise_std," << config.noise_std << "\n"
               << "noise_seed," << config.noise_seed << "\n"
               << "step2_input,step1.echo+step1.waveform\n"
               << "step3_input,step2.compressed\n"
               << "step4_input,step3.range_doppler\n"
               << "step5_input,step1.waveform\n";
    }

    if (completed_steps >= 1) {
        require(state.waveform.size() == static_cast<std::size_t>(config.pulse_samples),
                "Task1 visualization waveform shape mismatch");
        auto waveform = open_csv(directory, "task1_step1_waveform.csv");
        waveform << "sample,time_seconds,real,imag,magnitude,phase_rad\n";
        for (int sample = 0; sample < config.pulse_samples; ++sample) {
            const auto value = state.waveform[static_cast<std::size_t>(sample)];
            waveform << sample << ',' << static_cast<double>(sample) / config.sample_rate_hz << ','
                     << value.real << ',' << value.imag << ',' << magnitude(value) << ','
                     << std::atan2(value.imag, value.real) << '\n';
        }

        const std::size_t count = static_cast<std::size_t>(config.num_pulses) * config.samples_per_pulse;
        require(state.noiseless_echo.size() == count && state.noise.size() == count &&
                    state.echo.size() == count,
                "Task1 visualization echo shape mismatch");
        auto echo = open_csv(directory, "task1_step1_echo.csv");
        echo << "pulse,sample,fast_time_seconds,slow_time_seconds,noiseless_real,noiseless_imag,"
                "noise_real,noise_imag,echo_real,echo_imag,echo_magnitude\n";
        for (int pulse = 0; pulse < config.num_pulses; ++pulse) {
            for (int sample = 0; sample < config.samples_per_pulse; ++sample) {
                const std::size_t index = static_cast<std::size_t>(pulse) * config.samples_per_pulse + sample;
                const auto clean = state.noiseless_echo[index];
                const auto noise = state.noise[index];
                const auto received = state.echo[index];
                echo << pulse << ',' << sample << ','
                     << static_cast<double>(sample) / config.sample_rate_hz << ','
                     << static_cast<double>(pulse) / config.prf_hz << ','
                     << clean.real << ',' << clean.imag << ','
                     << noise.real << ',' << noise.imag << ','
                     << received.real << ',' << received.imag << ','
                     << magnitude(received) << '\n';
            }
        }
    }

    if (completed_steps >= 2) {
        require(state.compressed.size() == static_cast<std::size_t>(config.num_pulses) * config.samples_per_pulse,
                "Task1 visualization pulse-compression shape mismatch");
        auto output = open_csv(directory, "task1_step2_compressed.csv");
        output << "pulse,range_bin,range_meters,real,imag,magnitude,magnitude_db\n";
        for (int pulse = 0; pulse < config.num_pulses; ++pulse) {
            for (int range = 0; range < config.samples_per_pulse; ++range) {
                const auto value = state.compressed[
                    static_cast<std::size_t>(pulse) * config.samples_per_pulse + range];
                const double amplitude = magnitude(value);
                output << pulse << ',' << range << ','
                       << range * kSpeedOfLight / (2.0 * config.sample_rate_hz) << ','
                       << value.real << ',' << value.imag << ',' << amplitude << ','
                       << 20.0 * std::log10(std::max(amplitude, 1.0e-10)) << '\n';
            }
        }
    }

    if (completed_steps >= 3) {
        require(state.range_doppler.size() ==
                    static_cast<std::size_t>(config.num_pulses) * config.samples_per_pulse,
                "Task1 visualization range-Doppler shape mismatch");
        auto output = open_csv(directory, "task1_step3_range_doppler.csv");
        output << "doppler_index,signed_doppler_bin,doppler_hz,range_bin,range_meters,"
                  "real,imag,magnitude,power_db\n";
        for (int doppler = 0; doppler < config.num_pulses; ++doppler) {
            const int signed_bin = doppler <= config.num_pulses / 2
                ? doppler : doppler - config.num_pulses;
            const double doppler_hz = static_cast<double>(signed_bin) * config.prf_hz / config.num_pulses;
            for (int range = 0; range < config.samples_per_pulse; ++range) {
                const auto value = state.range_doppler[
                    static_cast<std::size_t>(doppler) * config.samples_per_pulse + range];
                const double amplitude = magnitude(value);
                output << doppler << ',' << signed_bin << ',' << doppler_hz << ','
                       << range << ',' << range * kSpeedOfLight / (2.0 * config.sample_rate_hz) << ','
                       << value.real << ',' << value.imag << ',' << amplitude << ','
                       << power_db(amplitude * amplitude) << '\n';
            }
        }
    }

    if (completed_steps >= 4) {
        const std::size_t count = static_cast<std::size_t>(config.num_pulses) * config.samples_per_pulse;
        require(state.power.size() == count && state.cfar_threshold.size() == count &&
                    state.detections.size() == count,
                "Task1 visualization CFAR shape mismatch");
        auto output = open_csv(directory, "task1_step4_cfar.csv");
        output << "doppler_index,signed_doppler_bin,doppler_hz,range_bin,range_meters,"
                  "power,threshold,detection\n";
        for (int doppler = 0; doppler < config.num_pulses; ++doppler) {
            const int signed_bin = doppler <= config.num_pulses / 2
                ? doppler : doppler - config.num_pulses;
            const double doppler_hz = static_cast<double>(signed_bin) * config.prf_hz / config.num_pulses;
            for (int range = 0; range < config.samples_per_pulse; ++range) {
                const std::size_t index = static_cast<std::size_t>(doppler) * config.samples_per_pulse + range;
                output << doppler << ',' << signed_bin << ',' << doppler_hz << ','
                       << range << ',' << range * kSpeedOfLight / (2.0 * config.sample_rate_hz) << ','
                       << state.power[index] << ',' << state.cfar_threshold[index] << ','
                       << (state.detections[index] == cusignal::CfarDetection::yes ? 1 : 0)
                       << '\n';
            }
        }
    }

    if (completed_steps >= 5) {
        const int rows = 2 * config.pulse_samples - 1;
        const int columns = static_cast<int>(state.ambiguity_doppler.size());
        require(columns > 0, "Task1 visualization ambiguity Doppler axis is empty");
        require(state.ambiguity_2d.size() == static_cast<std::size_t>(rows * columns) &&
                    state.ambiguity_delay.size() == rows &&
                    state.ambiguity_doppler.size() == columns,
                "Task1 visualization ambiguity shape mismatch");
        auto map = open_csv(directory, "task1_step5_ambiguity.csv");
        map << "delay_index,delay_seconds,doppler_index,doppler_hz,value\n";
        for (int delay = 0; delay < rows; ++delay) {
            const double delay_seconds = static_cast<double>(delay - (config.pulse_samples - 1)) / config.sample_rate_hz;
            for (int doppler = 0; doppler < columns; ++doppler) {
                const double doppler_hz = -0.5 * config.sample_rate_hz +
                    static_cast<double>(doppler) * config.sample_rate_hz / columns;
                map << delay << ',' << delay_seconds << ',' << doppler << ',' << doppler_hz
                    << ',' << state.ambiguity_2d[
                        static_cast<std::size_t>(delay) * columns + doppler] << '\n';
            }
        }
        auto delay_cut = open_csv(directory, "task1_step5_delay_cut.csv");
        delay_cut << "delay_index,delay_seconds,value\n";
        for (int index = 0; index < rows; ++index)
            delay_cut << index << ','
                      << static_cast<double>(index - (config.pulse_samples - 1)) / config.sample_rate_hz
                      << ',' << state.ambiguity_delay[static_cast<std::size_t>(index)] << '\n';
        auto doppler_cut = open_csv(directory, "task1_step5_doppler_cut.csv");
        doppler_cut << "doppler_index,doppler_hz,value\n";
        for (int index = 0; index < columns; ++index)
            doppler_cut << index << ','
                        << -0.5 * config.sample_rate_hz + static_cast<double>(index) * config.sample_rate_hz / columns
                        << ',' << state.ambiguity_doppler[static_cast<std::size_t>(index)] << '\n';
    }
}

}  // namespace task1
