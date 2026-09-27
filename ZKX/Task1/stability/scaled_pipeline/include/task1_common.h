#pragma once

#include <cusignal/runtime/cuda_error.h>
#include <cusignal/runtime/runtime_utils.h>
#include <cusignal/operators/radartools/radartools_typed.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace task1 {

using Clock = std::chrono::steady_clock;
#ifndef TASK1_STABILITY_NUM_PULSES
#define TASK1_STABILITY_NUM_PULSES 32
#endif
#ifndef TASK1_STABILITY_SAMPLES_PER_PULSE
#define TASK1_STABILITY_SAMPLES_PER_PULSE 256
#endif
inline constexpr int kNumPulses = TASK1_STABILITY_NUM_PULSES;
inline constexpr int kSamplesPerPulse = TASK1_STABILITY_SAMPLES_PER_PULSE;
inline constexpr int kPulseSamples = 128;
inline int kTargetDelay = 80;
inline int kDopplerBin = 5;
inline constexpr float kSampleRate = 2.56e6F;
inline constexpr float kPrf = 8.0e3F;
inline constexpr float kBandwidth = 6.4e5F;
inline constexpr float kCarrierFrequency = 10.0e9F;
inline float kTargetAmplitude = 1.0F;
inline float kDopplerFrequency = 1250.0F;
inline constexpr float kNoiseStd = 0.035F;
inline constexpr float kPfa = 1.0e-3F;
inline std::vector<cusignal::TypedComplex<float>> kStabilityNoise;
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
    std::vector<cusignal::TypedComplex<float>> waveform;
    std::vector<cusignal::TypedComplex<float>> noiseless_echo;
    std::vector<cusignal::TypedComplex<float>> noise;
    std::vector<cusignal::TypedComplex<float>> echo;
    std::vector<cusignal::TypedComplex<float>> compressed;
    std::vector<cusignal::TypedComplex<float>> range_doppler;
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
    function();
    return 0.0F;
}

}  // namespace task1
