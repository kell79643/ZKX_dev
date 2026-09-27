#pragma once

#include <task2/task2_common.h>

namespace task2::accuracy {

struct AccuracyEvidence {
    bool passed = true;
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
    std::map<std::string, double> metrics;
};

struct Step1ValidationData {
    std::vector<float> chirp_cpu;
    std::vector<float> target_cpu;
    std::vector<float> interference_cpu;
    std::vector<float> noise_cpu;
    std::vector<float> echo_cpu;
    std::vector<float> changed_echo_cpu;
};

struct Step2ValidationData {
    std::vector<float> filtered_cpu;
    std::vector<float> target_filtered_cpu;
};

struct Step3ValidationData {
    std::vector<float> cropped_input;
    std::vector<float> smoothed_cpu;
    std::vector<float> reference_for_correlation_cpu;
};

struct Step4ValidationData {
    std::vector<float> fm_cpu;
    std::vector<float> correlation_cpu;
    std::vector<float> spectral_cpu;
    std::vector<float> wavelet_cpu;
    std::vector<float> bundle_cpu;
};

struct Step5ValidationData {
    std::vector<std::int64_t> expected;
    std::vector<std::int64_t> corrupted_expected;
    std::vector<std::int64_t> injected_gpu;
    std::vector<std::int64_t> injected_cpu;
    std::vector<std::int64_t> moved_gpu;
    std::vector<std::int64_t> moved_cpu;
};

struct Step6ValidationData {
    std::vector<float> state_cpu;
    std::vector<float> covariance_cpu;
    float alternate_estimate_cpu = 0.0F;
};

}  // namespace task2::accuracy
