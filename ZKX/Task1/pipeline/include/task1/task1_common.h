#pragma once

#include <cusignal/runtime/cuda_error.h>
#include <cusignal/runtime/runtime_utils.h>
#include <cusignal/operators/radartools/radartools_typed.h>
#include <task1/task1_config.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace task1 {

using Clock = std::chrono::steady_clock;
inline constexpr float kPi = 3.14159265358979323846F;
inline constexpr double kSpeedOfLight = 299792458.0;

struct StepEvidence {
    std::string name;
    double prep_ms = 0.0;
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
    double formal_execution_ms = 0.0;
    double max_abs_error = 0.0;
    double mse = 0.0;
    double rmse = 0.0;
    double relative_error = 0.0;
    double relative_linf_error = 0.0;
    std::size_t compared_elements = 0;
    double relative_tolerance = 0.0;
    double relative_floor = 0.0;
    std::int64_t first_failure_index = -1;
    std::string reference_source;
    std::string comparison_method;
    std::map<std::string, float> operator_ms;
    std::map<std::string, double> metrics;
};

struct PipelineState {
    explicit PipelineState(TaskConfig selected_config)
        : config(std::move(selected_config)) {}
    TaskConfig config;
    std::vector<cusignal::TypedComplex<float>> waveform;
    std::vector<cusignal::TypedComplex<float>> noiseless_echo;
    std::vector<cusignal::TypedComplex<float>> noise;
    std::vector<cusignal::TypedComplex<float>> echo;
    std::vector<cusignal::TypedComplex<float>> compressed;
    std::vector<cusignal::TypedComplex<float>> range_doppler;
    cusignal::DeviceArray<ComplexFloat> range_doppler_device;
    std::vector<float> power;
    double cfar_alpha = 0.0;
    std::vector<float> cfar_threshold;
    std::vector<cusignal::CfarDetection> detections;
    std::vector<float> ambiguity_2d;
    std::vector<double> ambiguity_delay;
    std::vector<double> ambiguity_doppler;
};

double milliseconds(Clock::time_point begin, Clock::time_point end);
void require(bool condition, const std::string& message);
std::vector<cusignal::TypedComplex<float>> radar_result_to_host(
    const cusignal::RadarComplexDeviceResult& result);
std::vector<cusignal::TypedComplex<float>> radar_cpu_to_host(
    const cusignal::RadarComplexCpuResult& result);

template <class Function>
float time_gpu(Function&& function)
{
    cusignal::cuda_utils::EventTimer timer;
    timer.record_start();
    function();
    timer.record_stop();
    return timer.elapsed_ms();
}

}  // namespace task1
