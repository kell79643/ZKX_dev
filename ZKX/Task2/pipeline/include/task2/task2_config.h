#pragma once

#include <demo/config_file.h>
#include <task2/task2_abnormal_validation.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace task2 {

struct TaskConfig {
    std::filesystem::path path;
    std::string sha256;
    int samples;
    float sample_rate_hz;
    int target_delay_samples;
    std::uint32_t noise_seed;
    float target_weight;
    float gaussian_weight;
    float sawtooth_weight;
    float square_weight;
    float noise_amplitude;
    int filter_taps;
    float filter_cutoff_hz;
    int step3_crop_each;
    float fusion_weight_fm;
    float fusion_weight_correlation;
    float fusion_weight_spectral;
    float fusion_weight_wavelet;
    int argrelextrema_order;
    int kalman_observation_count;
    float kalman_f;
    float kalman_q;
    float kalman_h;
    float kalman_r;
    float kalman_measurement_noise_variant;

    void validate() const
    {
        if (!demo_config::is_power_of_two(samples) || samples < 256)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 samples必须是大于等于256的2的整数次幂");
        (void)checked_task2_input_elements(samples);
        if (target_delay_samples < 0 || target_delay_samples >= samples)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 target_delay_samples超出信号范围");
        if (!(sample_rate_hz > 0 && filter_cutoff_hz > 0 &&
              filter_cutoff_hz < sample_rate_hz / 2))
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2采样率或滤波截止频率违反Nyquist约束");
        if (filter_taps < 3 || filter_taps % 2 == 0)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 filter_taps必须为大于等于3的奇数");
        if (step3_crop_each < 0 || 2 * step3_crop_each >= samples - 64)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 Step3裁剪后输出长度不足");
        const float weights[] = {target_weight, gaussian_weight,
            sawtooth_weight, square_weight, noise_amplitude, fusion_weight_fm,
            fusion_weight_correlation, fusion_weight_spectral,
            fusion_weight_wavelet, kalman_q, kalman_r,
            kalman_measurement_noise_variant};
        for (const float value : weights)
            if (value < 0)
                cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                    "Task2权重、噪声或协方差参数必须非负");
        const float fusion_sum = fusion_weight_fm + fusion_weight_correlation +
            fusion_weight_spectral + fusion_weight_wavelet;
        if (std::abs(fusion_sum - 1.0F) > 1.0e-6F)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2融合权重之和必须为1");
        if (argrelextrema_order < 1 || 2 * argrelextrema_order >= samples)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 argrelextrema_order无效");
        if (kalman_observation_count < 1 || kalman_observation_count > samples)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 kalman_observation_count无效");
        if (kalman_f == 0 || kalman_h == 0 || kalman_r == 0)
            cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
                "Task2 Kalman F/H/R必须非零");
    }

    static TaskConfig load(const std::filesystem::path& path)
    {
        try {
            const std::set<std::string> fields{
                "samples", "sample_rate_hz", "target_delay_samples", "noise_seed",
                "target_weight", "gaussian_weight", "sawtooth_weight", "square_weight",
                "noise_amplitude", "filter_taps", "filter_cutoff_hz", "step3_crop_each",
                "fusion_weight_fm", "fusion_weight_correlation", "fusion_weight_spectral",
                "fusion_weight_wavelet", "argrelextrema_order", "kalman_observation_count",
                "kalman_F", "kalman_Q", "kalman_H", "kalman_R",
                "kalman_measurement_noise_variant"};
            const auto file = demo_config::ConfigFile::load(path, fields);
            TaskConfig config{
                file.path, file.sha256, file.integer("samples"),
                static_cast<float>(file.number("sample_rate_hz")),
                file.integer("target_delay_samples"), file.unsigned_integer("noise_seed"),
                static_cast<float>(file.number("target_weight")),
                static_cast<float>(file.number("gaussian_weight")),
                static_cast<float>(file.number("sawtooth_weight")),
                static_cast<float>(file.number("square_weight")),
                static_cast<float>(file.number("noise_amplitude")), file.integer("filter_taps"),
                static_cast<float>(file.number("filter_cutoff_hz")), file.integer("step3_crop_each"),
                static_cast<float>(file.number("fusion_weight_fm")),
                static_cast<float>(file.number("fusion_weight_correlation")),
                static_cast<float>(file.number("fusion_weight_spectral")),
                static_cast<float>(file.number("fusion_weight_wavelet")),
                file.integer("argrelextrema_order"), file.integer("kalman_observation_count"),
                static_cast<float>(file.number("kalman_F")),
                static_cast<float>(file.number("kalman_Q")),
                static_cast<float>(file.number("kalman_H")),
                static_cast<float>(file.number("kalman_R")),
                static_cast<float>(file.number("kalman_measurement_noise_variant"))};
            config.validate();
            return config;
        } catch (const cusignal::ProjectError&) {
            throw;
        } catch (const std::exception& error) {
            cusignal::throw_error(cusignal::ErrorCode::CONFIG_ERROR,
                "Task2配置缺失或损坏：" + std::string(error.what()));
        }
    }

    int step3_samples() const { return samples - 2 * step3_crop_each; }
};

}  // namespace task2
