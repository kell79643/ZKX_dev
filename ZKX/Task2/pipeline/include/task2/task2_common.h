#pragma once

#include <cusignal/runtime/cuda_error.h>
#include <cusignal/runtime/host_output_finalize.h>
#include <cusignal/runtime/runtime_utils.h>
#include <task2/task2_config.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace task2 {

using Clock = std::chrono::steady_clock;
// 正式多规模CSV中，16384使连续Step1/2/3同时取得各自最高speedup。
inline constexpr float kPi = 3.14159265358979323846F;

struct StepEvidence {
    std::string name;
    double prep_ms = 0.0;
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
    double formal_execution_ms = 0.0;
    bool accuracy_pass = false;
    double relative_error = 0.0;
    double relative_linf_error = 0.0;
    double mse = 0.0;
    double rmse = 0.0;
    double relative_tolerance = 0.0;
    double max_error = 0.0;
    std::size_t compared_elements = 0;
    double tolerance = 0.0;
    std::int64_t first_failure_index = -1;
    std::string reference_source;
    std::string comparison_method;
    std::string failure_reason;
    std::string input_shape;
    std::string input_dtype;
    std::string failure_action;
    std::map<std::string, float> operator_ms;
    std::map<std::string, double> metrics;
};

struct PipelineState {
    explicit PipelineState(TaskConfig selected_config)
        : config(std::move(selected_config)) {}
    TaskConfig config;
    std::vector<float> time;
    std::vector<float> target_waveform;
    std::vector<float> gaussian_waveform;
    std::vector<float> sawtooth_waveform;
    std::vector<float> square_waveform;
    std::vector<float> target_component;
    std::vector<float> interference_component;
    std::vector<float> noise_component;
    std::vector<float> echo;
    std::vector<float> filtered;
    std::vector<float> filter_taps;
    std::vector<float> smoothed;
    std::vector<float> spline_weights;
    std::vector<float> reference_for_correlation;
    std::vector<float> fm_feature;
    std::vector<float> correlation_feature;
    std::vector<float> spectral_feature;
    std::vector<float> wavelet_feature;
    std::vector<float> feature_bundle;
    std::vector<float> spectrogram_grid;
    std::vector<double> cwt_grid;
    int spectrogram_frequency_bins = 0;
    int spectrogram_frames = 0;
    int cwt_width_count = 0;
    std::vector<std::int64_t> extrema;
    std::vector<float> kalman_observations;
    std::vector<float> kalman_covariance;
    std::int64_t kalman_anchor = 0;
    float estimate = 0.0F;
    float truth = 0.0F;
};

inline bool visualization_capture_enabled()
{
    const char* directory = std::getenv("ZKX_VISUALIZATION_DIR");
    return directory != nullptr && directory[0] != '\0';
}

double milliseconds(Clock::time_point begin, Clock::time_point end);
void require(bool condition, const std::string& message);
double rms(const std::vector<float>& values);
double roughness(const std::vector<float>& values);
std::vector<float> normalize_abs(const std::vector<float>& input);
std::vector<float> resample_linear(
    const std::vector<float>& input, std::size_t count);

template <class Function>
float time_gpu(Function&& function)
{
    cusignal::cuda_utils::EventTimer timer;
    timer.record_start();
    function();
    timer.record_stop();
    return timer.elapsed_ms();
}

}  // namespace task2
