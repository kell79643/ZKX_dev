#include "accuracy/accuracy_utils.h"

#include <utility>

namespace task1::accuracy {
namespace {

template <class Actual, class Expected, class Difference>
double maximum_difference(
    const Actual& actual, const Expected& expected, Difference difference)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index)
        maximum = std::max(maximum, difference(actual[index], expected[index]));
    return maximum;
}

template <class Actual, class Expected, class Difference, class Magnitude>
double relative_l2_difference(
    const Actual& actual, const Expected& expected, double relative_floor,
    Difference difference, Magnitude magnitude)
{
    require(relative_floor > 0.0, "relative error floor must be positive");
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = difference(actual[index], expected[index]);
        const double reference = magnitude(expected[index]);
        squared_error += error * error;
        squared_reference += reference * reference;
    }
    return std::sqrt(squared_error /
        std::max(squared_reference, relative_floor * relative_floor));
}

template <class Actual, class Expected, class Difference>
double mean_squared_difference(
    const Actual& actual, const Expected& expected, Difference difference)
{
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    if (actual.empty()) return 0.0;
    double squared_error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = difference(actual[index], expected[index]);
        squared_error += error * error;
    }
    return squared_error / static_cast<double>(actual.size());
}

template <class Actual, class Expected, class Difference, class Magnitude>
double relative_linf_difference(
    const Actual& actual, const Expected& expected, double relative_floor,
    Difference difference, Magnitude magnitude)
{
    require(relative_floor > 0.0, "relative error floor must be positive");
    require(actual.size() == expected.size(), "accuracy comparison size mismatch");
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(
            maximum_error, difference(actual[index], expected[index]));
        maximum_reference = std::max(maximum_reference, magnitude(expected[index]));
    }
    return maximum_error / std::max(maximum_reference, relative_floor);
}

double complex_difference(
    cusignal::TypedComplex<float> actual,
    cusignal::TypedComplex<float> expected)
{
    return std::hypot(
        static_cast<double>(actual.real) - expected.real,
        static_cast<double>(actual.imag) - expected.imag);
}

}  // namespace

double complex_magnitude(cusignal::TypedComplex<float> value)
{
    return std::hypot(static_cast<double>(value.real), static_cast<double>(value.imag));
}

double max_abs_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected)
{
    return maximum_difference(actual, expected, complex_difference);
}

double mean_squared_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected)
{
    return mean_squared_difference(actual, expected, complex_difference);
}

double mean_squared_error(
    const std::vector<float>& actual, const std::vector<float>& expected)
{
    return mean_squared_difference(actual, expected, [](float lhs, float rhs) {
        return std::abs(static_cast<double>(lhs) - rhs);
    });
}

double mean_squared_error(
    const std::vector<double>& actual, const std::vector<double>& expected)
{
    return mean_squared_difference(actual, expected, [](double lhs, double rhs) {
        return std::abs(lhs - rhs);
    });
}

double max_abs_error(
    const std::vector<float>& actual, const std::vector<float>& expected)
{
    return maximum_difference(actual, expected, [](float lhs, float rhs) {
        return std::abs(static_cast<double>(lhs) - rhs);
    });
}

double max_abs_error(
    const std::vector<double>& actual, const std::vector<double>& expected)
{
    return maximum_difference(actual, expected, [](double lhs, double rhs) {
        return std::abs(lhs - rhs);
    });
}

double relative_l2_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected,
    double relative_floor)
{
    return relative_l2_difference(
        actual, expected, relative_floor, complex_difference, complex_magnitude);
}

double relative_linf_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected,
    double relative_floor)
{
    return relative_linf_difference(
        actual, expected, relative_floor, complex_difference, complex_magnitude);
}

double relative_linf_error(
    const std::vector<float>& actual, const std::vector<float>& expected,
    double relative_floor)
{
    return relative_linf_difference(
        actual, expected, relative_floor,
        [](float lhs, float rhs) { return std::abs(static_cast<double>(lhs) - rhs); },
        [](float value) { return std::abs(static_cast<double>(value)); });
}

double relative_linf_error(
    const std::vector<double>& actual, const std::vector<double>& expected,
    double relative_floor)
{
    return relative_linf_difference(
        actual, expected, relative_floor,
        [](double lhs, double rhs) { return std::abs(lhs - rhs); },
        [](double value) { return std::abs(value); });
}

double relative_l2_error(
    const std::vector<float>& actual, const std::vector<float>& expected,
    double relative_floor)
{
    return relative_l2_difference(
        actual, expected, relative_floor,
        [](float lhs, float rhs) { return std::abs(static_cast<double>(lhs) - rhs); },
        [](float value) { return std::abs(static_cast<double>(value)); });
}

double relative_l2_error(
    const std::vector<double>& actual, const std::vector<double>& expected,
    double relative_floor)
{
    return relative_l2_difference(
        actual, expected, relative_floor,
        [](double lhs, double rhs) { return std::abs(lhs - rhs); },
        [](double value) { return std::abs(value); });
}

double relative_error(double actual, double expected, double relative_floor)
{
    require(relative_floor > 0.0, "relative error floor must be positive");
    return std::abs(actual - expected) /
        std::max(std::abs(expected), relative_floor);
}

double rms_complex(const std::vector<cusignal::TypedComplex<float>>& values)
{
    double sum = 0.0;
    for (const auto& value : values)
        sum += static_cast<double>(value.real) * value.real +
            static_cast<double>(value.imag) * value.imag;
    return std::sqrt(sum / std::max<std::size_t>(values.size(), 1));
}

void merge_accuracy(StepEvidence& step, AccuracyEvidence accuracy)
{
    step.max_abs_error = accuracy.max_abs_error;
    step.mse = accuracy.mse;
    step.rmse = accuracy.rmse;
    step.relative_error = accuracy.relative_error;
    step.relative_linf_error = accuracy.relative_linf_error;
    step.compared_elements = accuracy.compared_elements;
    step.relative_tolerance = accuracy.relative_tolerance;
    step.relative_floor = accuracy.relative_floor;
    step.first_failure_index = accuracy.first_failure_index;
    step.reference_source = std::move(accuracy.reference_source);
    step.comparison_method = std::move(accuracy.comparison_method);
    step.metrics = std::move(accuracy.metrics);
}

}  // namespace task1::accuracy
