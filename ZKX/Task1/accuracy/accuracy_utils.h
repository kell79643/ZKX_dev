#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

double complex_magnitude(cusignal::TypedComplex<float> value);
double max_abs_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected);
double max_abs_error(
    const std::vector<float>& actual, const std::vector<float>& expected);
double max_abs_error(
    const std::vector<double>& actual, const std::vector<double>& expected);
double mean_squared_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected);
double mean_squared_error(
    const std::vector<float>& actual, const std::vector<float>& expected);
double mean_squared_error(
    const std::vector<double>& actual, const std::vector<double>& expected);
double relative_l2_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected,
    double relative_floor);
double relative_l2_error(
    const std::vector<float>& actual, const std::vector<float>& expected,
    double relative_floor);
double relative_l2_error(
    const std::vector<double>& actual, const std::vector<double>& expected,
    double relative_floor);
double relative_linf_error(
    const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected,
    double relative_floor);
double relative_linf_error(
    const std::vector<float>& actual, const std::vector<float>& expected,
    double relative_floor);
double relative_linf_error(
    const std::vector<double>& actual, const std::vector<double>& expected,
    double relative_floor);
double relative_error(double actual, double expected, double relative_floor);
double rms_complex(const std::vector<cusignal::TypedComplex<float>>& values);
void merge_accuracy(StepEvidence& step, AccuracyEvidence accuracy);

}  // namespace task1::accuracy
