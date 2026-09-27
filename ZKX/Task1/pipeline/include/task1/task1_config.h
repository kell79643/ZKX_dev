#pragma once

#include <demo/config_file.h>
#include <task1/task1_abnormal_validation.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace task1 {

struct TaskConfig {
    std::filesystem::path path;
    std::string sha256;
    int num_pulses;
    int samples_per_pulse;
    int pulse_samples;
    float sample_rate_hz;
    float prf_hz;
    float bandwidth_hz;
    double carrier_frequency_hz;
    int target_delay_samples;
    int doppler_bin;
    float target_amplitude;
    float noise_std;
    std::uint32_t noise_seed;
    float pfa;
    int cfar_guard_doppler;
    int cfar_guard_range;
    int cfar_reference_doppler;
    int cfar_reference_range;
    int ambiguity_nfreq;

    void validate() const
    {
        if (!demo_config::is_power_of_two(num_pulses) || !demo_config::is_power_of_two(samples_per_pulse))
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1 FFT维度必须为2的整数次幂");
        (void)checked_task1_input_elements(num_pulses, samples_per_pulse);
        if (pulse_samples < 1 || pulse_samples > samples_per_pulse)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1 pulse_samples必须落在接收窗口内");
        if (target_delay_samples < 0 || target_delay_samples + pulse_samples > samples_per_pulse)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1目标延迟使脉冲越出接收窗口");
        if (std::abs(doppler_bin) >= num_pulses / 2)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1 doppler_bin超出无混叠范围");
        if (!(sample_rate_hz > 0 && prf_hz > 0 && bandwidth_hz > 0 && carrier_frequency_hz > 0 &&
              target_amplitude > 0 && noise_std >= 0))
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1采样率、幅度或噪声参数违反物理约束");
        if (!(pfa > 0 && pfa < 1))
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1 pfa必须位于开区间(0,1)");
        if (cfar_guard_doppler < 0 || cfar_guard_range < 0 || cfar_reference_doppler < 1 || cfar_reference_range < 1)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1 CFAR保护单元或参考单元参数无效");
        int expected_nfreq = 1;
        while (expected_nfreq < 2 * pulse_samples - 1) expected_nfreq <<= 1;
        if (ambiguity_nfreq != expected_nfreq)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task1 ambiguity_nfreq必须等于next_power_of_two(2*pulse_samples-1)");
    }

    static TaskConfig load(const std::filesystem::path& path)
    {
        try {
            const std::set<std::string> fields{
                "num_pulses", "samples_per_pulse", "pulse_samples", "sample_rate_hz",
                "prf_hz", "bandwidth_hz", "carrier_frequency_hz", "target_delay_samples",
                "doppler_bin", "target_amplitude", "noise_std", "noise_seed", "pfa",
                "cfar_guard_doppler", "cfar_guard_range", "cfar_reference_doppler",
                "cfar_reference_range", "ambiguity_nfreq"};
            const auto file = demo_config::ConfigFile::load(path, fields);
            TaskConfig config{
                file.path, file.sha256,
                file.integer("num_pulses"), file.integer("samples_per_pulse"),
                file.integer("pulse_samples"), static_cast<float>(file.number("sample_rate_hz")),
                static_cast<float>(file.number("prf_hz")), static_cast<float>(file.number("bandwidth_hz")),
                file.number("carrier_frequency_hz"), file.integer("target_delay_samples"),
                file.integer("doppler_bin"), static_cast<float>(file.number("target_amplitude")),
                static_cast<float>(file.number("noise_std")), file.unsigned_integer("noise_seed"),
                static_cast<float>(file.number("pfa")), file.integer("cfar_guard_doppler"),
                file.integer("cfar_guard_range"), file.integer("cfar_reference_doppler"),
                file.integer("cfar_reference_range"), file.integer("ambiguity_nfreq")};
            config.validate();
            return config;
        } catch (const cusignal::ProjectError&) {
            throw;
        } catch (const std::exception& error) {
            cusignal::throw_error(cusignal::ErrorCode::CONFIG_ERROR,
                "Task1配置缺失或损坏：" + std::string(error.what()));
        }
    }

    float doppler_hz() const
    {
        return static_cast<float>(doppler_bin) * prf_hz / static_cast<float>(num_pulses);
    }
};

}  // namespace task1
