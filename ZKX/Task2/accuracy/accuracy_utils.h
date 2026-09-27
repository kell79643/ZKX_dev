#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {

double max_abs_error(
    const std::vector<float>& actual, const std::vector<float>& expected);
double relative_l2_error(
    const std::vector<float>& actual, const std::vector<float>& expected);
double mean_squared_error(
    const std::vector<float>& actual, const std::vector<float>& expected);
double relative_linf_error(
    const std::vector<float>& actual, const std::vector<float>& expected);
double mean_squared_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected);
double relative_l2_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected);
double relative_linf_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected);
double relative_mismatch_rate(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected);
double max_abs_index_error(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected);
std::int64_t first_failure_index(
    const std::vector<float>& actual, const std::vector<float>& expected,
    double tolerance);
std::int64_t first_mismatch_index(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected);
void record_check(
    AccuracyEvidence& evidence, bool condition, const std::string& failure_reason);
void merge_accuracy(StepEvidence& step, AccuracyEvidence accuracy);

}  // namespace task2::accuracy
