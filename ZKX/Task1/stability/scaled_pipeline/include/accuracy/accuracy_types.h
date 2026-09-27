#pragma once

#include "task1_common.h"

namespace task1::accuracy {

struct AccuracyEvidence {
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
    std::map<std::string, double> metrics;
};

struct Step1ValidationData {
    std::vector<cusignal::TypedComplex<float>> waveform_cpu;
    std::vector<cusignal::TypedComplex<float>> noiseless_cpu;
    std::vector<cusignal::TypedComplex<float>> noise_cpu;
    std::vector<cusignal::TypedComplex<float>> echo_cpu;
    std::vector<cusignal::TypedComplex<float>> changed_delay_cpu;
    std::vector<cusignal::TypedComplex<float>> changed_doppler_cpu;
    std::vector<cusignal::TypedComplex<float>> changed_seed_cpu;
};

struct Step2ValidationData {
    std::vector<cusignal::TypedComplex<float>> compressed_cpu;
    std::vector<cusignal::TypedComplex<float>> altered_template_cpu;
};

struct Step3ValidationData {
    std::vector<cusignal::TypedComplex<float>> range_doppler_cpu;
    std::vector<cusignal::TypedComplex<float>> no_window_cpu;
};

struct Step4ValidationData {
    std::vector<float> power_cpu;
    std::vector<float> threshold_cpu;
    std::vector<cusignal::CfarDetection> detections_cpu;
    double alpha_cpu = 0.0;
    std::vector<cusignal::CfarDetection> noise_detections_gpu;
    std::vector<cusignal::CfarDetection> interference_detections_gpu;
    double stricter_target_threshold_cpu = 0.0;
};

struct Step5ValidationData {
    std::vector<double> ambiguity_2d_cpu;
    std::vector<double> ambiguity_delay_cpu;
    std::vector<double> ambiguity_doppler_cpu;
    std::vector<double> changed_waveform_cpu;
};

}  // namespace task1::accuracy
