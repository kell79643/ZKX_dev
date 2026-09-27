#include "accuracy/accuracy_utils.h"

namespace task2::accuracy {

double max_abs_error(
    const std::vector<float>& actual, const std::vector<float>& expected)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index)
        maximum = std::max(maximum, std::abs(
            static_cast<double>(actual[index]) - expected[index]));
    return maximum;
}

double relative_l2_error(
    const std::vector<float>& actual, const std::vector<float>& expected)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = static_cast<double>(actual[index]) - expected[index];
        squared_error += difference * difference;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
    constexpr double kReferenceFloor = 1.0e-24;
    return std::sqrt(squared_error / std::max(squared_reference, kReferenceFloor));
}

double mean_squared_error(
    const std::vector<float>& actual, const std::vector<float>& expected)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    if (actual.empty()) return 0.0;
    double squared_error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = static_cast<double>(actual[index]) - expected[index];
        squared_error += difference * difference;
    }
    return squared_error / static_cast<double>(actual.size());
}

double relative_linf_error(
    const std::vector<float>& actual, const std::vector<float>& expected)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(maximum_error, std::abs(
            static_cast<double>(actual[index]) - expected[index]));
        maximum_reference = std::max(maximum_reference, std::abs(
            static_cast<double>(expected[index])));
    }
    constexpr double kReferenceFloor = 1.0e-12;
    return maximum_error / std::max(maximum_reference, kReferenceFloor);
}

double mean_squared_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected)
{
    require(actual.size() == expected.size(), "index comparison size mismatch");
    if (actual.empty()) return 0.0;
    double squared_error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = static_cast<double>(actual[index]) - expected[index];
        squared_error += difference * difference;
    }
    return squared_error / static_cast<double>(actual.size());
}

double relative_l2_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected)
{
    require(actual.size() == expected.size(), "index comparison size mismatch");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = static_cast<double>(actual[index]) - expected[index];
        squared_error += difference * difference;
        const double reference = static_cast<double>(expected[index]);
        squared_reference += reference * reference;
    }
    constexpr double kReferenceFloor = 1.0e-24;
    return std::sqrt(squared_error / std::max(squared_reference, kReferenceFloor));
}

double relative_linf_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected)
{
    require(actual.size() == expected.size(), "index comparison size mismatch");
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(maximum_error, std::abs(
            static_cast<double>(actual[index]) - expected[index]));
        maximum_reference = std::max(maximum_reference, std::abs(
            static_cast<double>(expected[index])));
    }
    constexpr double kReferenceFloor = 1.0e-12;
    return maximum_error / std::max(maximum_reference, kReferenceFloor);
}

double relative_mismatch_rate(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected)
{
    const std::size_t count = std::max(actual.size(), expected.size());
    if (count == 0) return 0.0;
    const std::size_t common = std::min(actual.size(), expected.size());
    std::size_t mismatches = count - common;
    for (std::size_t index = 0; index < common; ++index)
        if (actual[index] != expected[index]) ++mismatches;
    return static_cast<double>(mismatches) / static_cast<double>(count);
}

double max_abs_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected)
{
    const std::size_t common = std::min(actual.size(), expected.size());
    double maximum = actual.size() == expected.size() ? 0.0 : 1.0;
    for (std::size_t index = 0; index < common; ++index)
        maximum = std::max(maximum, std::abs(
            static_cast<double>(actual[index]) - expected[index]));
    return maximum;
}

std::int64_t first_failure_index(
    const std::vector<float>& actual, const std::vector<float>& expected,
    double tolerance)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index)
        if (std::abs(static_cast<double>(actual[index]) - expected[index]) > tolerance)
            return static_cast<std::int64_t>(index);
    return -1;
}

std::int64_t first_mismatch_index(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected)
{
    const std::size_t common = std::min(actual.size(), expected.size());
    for (std::size_t index = 0; index < common; ++index)
        if (actual[index] != expected[index])
            return static_cast<std::int64_t>(index);
    return actual.size() == expected.size() ? -1 : static_cast<std::int64_t>(common);
}

void record_check(
    AccuracyEvidence& evidence, bool condition, const std::string& failure_reason)
{
    if (condition) return;
    evidence.passed = false;
    if (!evidence.failure_reason.empty()) evidence.failure_reason += "; ";
    evidence.failure_reason += failure_reason;
}

void merge_accuracy(StepEvidence& step, AccuracyEvidence accuracy)
{
    step.accuracy_pass = accuracy.passed;
    step.relative_error = accuracy.relative_error;
    step.relative_linf_error = accuracy.relative_linf_error;
    step.mse = accuracy.mse;
    step.rmse = accuracy.rmse;
    step.relative_tolerance = accuracy.relative_tolerance;
    step.max_error = accuracy.max_error;
    step.compared_elements = accuracy.compared_elements;
    step.tolerance = accuracy.tolerance;
    step.first_failure_index = accuracy.first_failure_index;
    step.reference_source = std::move(accuracy.reference_source);
    step.comparison_method = std::move(accuracy.comparison_method);
    step.failure_reason = std::move(accuracy.failure_reason);
    for (auto& item : accuracy.metrics)
        step.metrics[item.first] = std::move(item.second);
}

}  // namespace task2::accuracy
