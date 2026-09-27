#include "accuracy/formal_accuracy.h"

#include "accuracy/comparison/step1_comparison.h"
#include "accuracy/comparison/step2_comparison.h"
#include "accuracy/comparison/step3_comparison.h"
#include "accuracy/comparison/step4_comparison.h"
#include "accuracy/comparison/step5_comparison.h"
#include "accuracy/comparison/step6_comparison.h"
#include "accuracy/validation/step1_validation.h"
#include "accuracy/validation/step2_validation.h"
#include "accuracy/validation/step3_validation.h"
#include "accuracy/validation/step4_validation.h"
#include "accuracy/validation/step5_validation.h"
#include "accuracy/validation/step6_validation.h"
#include <demo/config_file.h>
#include "test_all/support/config/json_value.h"
#include "test_all/support/metrics/accuracy.h"
#include "test_all/support/metrics/csv_writer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace task2::accuracy {
namespace {

using zkx::config::JsonValue;

const JsonValue& member(const JsonValue& value, const char* name)
{
    const auto* result = value.find(name);
    if (!result)
        throw std::invalid_argument(std::string("accuracy threshold missing field: ") + name);
    return *result;
}

std::string key(
    const std::string& target, const std::string& dtype,
    const std::string& output_name)
{
    return target + "|" + dtype + "|" + output_name;
}

double nonnegative(const JsonValue& value, const char* name)
{
    const double result = member(value, name).as_number();
    if (!std::isfinite(result) || result < 0)
        throw std::invalid_argument(std::string("invalid accuracy threshold: ") + name);
    return result;
}

std::vector<double> f64(const std::vector<float>& input)
{
    return std::vector<double>(input.begin(), input.end());
}

OutputAccuracyResult numeric_result(
    const std::vector<float>& actual, const std::vector<float>& expected,
    const FormalThresholdSet& thresholds, const std::string& target,
    const std::string& output_name, const std::string& semantic)
{
    const auto& threshold = thresholds.require(target, "FP32", output_name);
    if (!threshold.numeric)
        throw std::invalid_argument(target + " must use numeric thresholds");
    const auto accuracy = zkx::common::compare(
        f64(expected), f64(actual), threshold.relative_floor);
    OutputAccuracyResult result;
    result.threshold = threshold;
    result.mse = accuracy.mse;
    result.rmse = accuracy.rmse;
    result.relative_l2 = accuracy.relative_l2;
    result.relative_linf = accuracy.relative_linf;
    result.semantic_check = threshold.semantic_rule + ":" + semantic;
    result.pass = std::isfinite(result.mse) && std::isfinite(result.rmse) &&
        std::isfinite(result.relative_l2) && std::isfinite(result.relative_linf) &&
        result.mse <= threshold.mse_max && result.rmse <= threshold.rmse_max &&
        result.relative_l2 <= threshold.relative_l2_max &&
        result.relative_linf <= threshold.relative_linf_max && semantic == "PASS";
    return result;
}

OutputAccuracyResult discrete_result(
    const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected,
    const FormalThresholdSet& thresholds, const std::string& target,
    const std::string& output_name, const std::string& semantic)
{
    const auto& threshold = thresholds.require(target, "FP32", output_name);
    if (threshold.numeric)
        throw std::invalid_argument(target + " must use discrete thresholds");
    OutputAccuracyResult result;
    result.threshold = threshold;
    const std::size_t common = std::min(actual.size(), expected.size());
    for (std::size_t index = 0; index < common; ++index)
        if (actual[index] != expected[index]) ++result.mismatch_count;
    result.mismatch_count +=
        std::max(actual.size(), expected.size()) - common;
    result.exact_match = result.mismatch_count == 0;
    result.semantic_check = threshold.semantic_rule + ":" + semantic;
    result.pass = (!threshold.require_exact_match || result.exact_match) &&
        result.mismatch_count <= threshold.max_mismatch_count && semantic == "PASS";
    return result;
}

double worst(const std::map<std::string, OutputAccuracyResult>& values,
             double OutputAccuracyResult::*field)
{
    double result = 0.0;
    for (const auto& item : values) result = std::max(result, item.second.*field);
    return result;
}

std::string number(double value)
{
    std::ostringstream output;
    output.precision(17);
    output << value;
    return output.str();
}

}  // namespace

FormalThresholdSet FormalThresholdSet::load(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open formal accuracy thresholds: " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string text = contents.str();
    const std::vector<unsigned char> bytes(text.begin(), text.end());
    const JsonValue document = JsonValue::parse(text);
    if (member(document, "schema_version").as_number() != 1.0)
        throw std::invalid_argument("accuracy threshold schema_version must be 1");
    FormalThresholdSet set;
    set.id_ = member(document, "threshold_set_id").as_string();
    set.sha256_ = demo_config::sha256(bytes);
    for (const JsonValue& item : member(document, "entries").as_array()) {
        FormalThreshold threshold;
        threshold.target = member(item, "target").as_string();
        threshold.dtype = member(item, "dtype").as_string();
        threshold.output_name = member(item, "output_name").as_string();
        threshold.output_kind = member(item, "output_kind").as_string();
        threshold.numeric = threshold.output_kind != "discrete";
        threshold.require_exact_match = member(item, "require_exact_match").as_boolean();
        if (threshold.numeric) {
            threshold.mse_max = nonnegative(item, "mse_max");
            threshold.rmse_max = nonnegative(item, "rmse_max");
            threshold.relative_l2_max = nonnegative(item, "relative_l2_max");
            threshold.relative_linf_max = nonnegative(item, "relative_linf_max");
            threshold.relative_floor = nonnegative(item, "relative_floor");
            if (!member(item, "max_mismatch_count").is_null())
                throw std::invalid_argument("numeric threshold max_mismatch_count must be null");
        } else {
            for (const char* name : {"mse_max", "rmse_max", "relative_l2_max",
                                     "relative_linf_max", "relative_floor"})
                if (!member(item, name).is_null())
                    throw std::invalid_argument(
                        std::string("discrete threshold must use null: ") + name);
            const double count = member(item, "max_mismatch_count").as_number();
            if (count < 0.0 || std::floor(count) != count)
                throw std::invalid_argument(
                    "max_mismatch_count must be a nonnegative integer");
            threshold.max_mismatch_count = static_cast<std::size_t>(count);
        }
        threshold.semantic_rule = member(item, "semantic_rule").as_string();
        const auto entry_key = key(
            threshold.target, threshold.dtype, threshold.output_name);
        if (!set.entries_.emplace(entry_key, threshold).second)
            throw std::invalid_argument("duplicate accuracy threshold key: " + threshold.target);
    }
    return set;
}

const FormalThreshold& FormalThresholdSet::require(
    const std::string& target, const std::string& dtype,
    const std::string& output_name) const
{
    const auto found = entries_.find(key(target, dtype, output_name));
    if (found == entries_.end())
        throw std::invalid_argument(
            "missing exact accuracy threshold: " + target + "/" + dtype + "/" + output_name);
    return found->second;
}

AccuracyResults evaluate_step1_formal(
    const PipelineState& state, const Step1ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    const auto semantic = compare_step1_accuracy(state, validation);
    const std::string status = semantic.passed ? "PASS" : "FAIL:" + semantic.failure_reason;
    return {
        numeric_result(state.target_waveform, validation.chirp_cpu, thresholds,
            "Task2.step1.target_waveform", "target_waveform", status),
        numeric_result(state.target_component, validation.target_cpu, thresholds,
            "Task2.step1.target_component", "target_component", status),
        numeric_result(state.interference_component, validation.interference_cpu, thresholds,
            "Task2.step1.interference_component", "interference_component", status),
        numeric_result(state.noise_component, validation.noise_cpu, thresholds,
            "Task2.step1.noise_component", "noise_component", status),
        numeric_result(state.echo, validation.echo_cpu, thresholds,
            "Task2.step1.echo", "echo", status)};
}

AccuracyResults evaluate_step2_formal(
    const PipelineState& state, const Step2ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    const auto semantic = compare_step2_accuracy(state, validation);
    const std::string status = semantic.passed ? "PASS" : "FAIL:" + semantic.failure_reason;
    return {numeric_result(state.filtered, validation.filtered_cpu, thresholds,
        "Task2.step2.filtered", "filtered", status)};
}

AccuracyResults evaluate_step3_formal(
    const PipelineState& state, const Step3ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    const auto semantic = compare_step3_accuracy(state, validation);
    const std::string status = semantic.passed ? "PASS" : "FAIL:" + semantic.failure_reason;
    return {
        numeric_result(state.smoothed, validation.smoothed_cpu, thresholds,
            "Task2.step3.smoothed", "smoothed", status),
        numeric_result(state.reference_for_correlation,
            validation.reference_for_correlation_cpu, thresholds,
            "Task2.step3.reference_for_correlation", "reference_for_correlation", status)};
}

AccuracyResults evaluate_step4_formal(
    const PipelineState& state, const Step4ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    const auto semantic = compare_step4_accuracy(state, validation);
    const std::string status = semantic.passed ? "PASS" : "FAIL:" + semantic.failure_reason;
    return {
        numeric_result(state.fm_feature, validation.fm_cpu, thresholds,
            "Task2.step4.fm_feature", "fm_feature", status),
        numeric_result(state.correlation_feature, validation.correlation_cpu, thresholds,
            "Task2.step4.correlation_feature", "correlation_feature", status),
        numeric_result(state.spectral_feature, validation.spectral_cpu, thresholds,
            "Task2.step4.spectral_feature", "spectral_feature", status),
        numeric_result(state.wavelet_feature, validation.wavelet_cpu, thresholds,
            "Task2.step4.wavelet_feature", "wavelet_feature", status),
        numeric_result(state.feature_bundle, validation.bundle_cpu, thresholds,
            "Task2.step4.feature_bundle", "feature_bundle", status)};
}

AccuracyResults evaluate_step5_formal(
    const PipelineState& state, const Step5ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    Step5ValidationData full = collect_step5_cpu_validation(state);
    if (full.expected != validation.expected)
        throw std::runtime_error("timed Step5 CPU reference changed outside primary timing");
    const auto cuda = collect_step5_cuda_validation(state);
    full.injected_gpu = cuda.injected_gpu;
    full.moved_gpu = cuda.moved_gpu;
    const auto semantic = compare_step5_accuracy(state, full);
    const std::string status = semantic.passed ? "PASS" : "FAIL:" + semantic.failure_reason;
    return {discrete_result(state.extrema, full.expected, thresholds,
        "Task2.step5.extrema", "extrema", status)};
}

AccuracyResults evaluate_step6_formal(
    const PipelineState& state, const Step6ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    Step6ValidationData full = collect_step6_validation(state);
    if (full.state_cpu != validation.state_cpu ||
        full.covariance_cpu != validation.covariance_cpu)
        throw std::runtime_error("timed Step6 CPU reference changed outside primary timing");
    const auto semantic = compare_step6_accuracy(state, full);
    const std::string status = semantic.passed ? "PASS" : "FAIL:" + semantic.failure_reason;
    return {
        numeric_result(std::vector<float>{state.estimate}, full.state_cpu, thresholds,
            "Task2.step6.estimate", "estimate", status),
        numeric_result(state.kalman_covariance, full.covariance_cpu, thresholds,
            "Task2.step6.covariance", "covariance", status)};
}

AccuracyResults evaluate_pipeline_formal(
    const PipelineState& gpu, const PipelineState& cpu,
    const FormalThresholdSet& thresholds)
{
    AccuracyResults results;
    auto append = [&](AccuracyResults values) {
        results.insert(results.end(), values.begin(), values.end());
    };

    auto step1 = collect_step1_validation(gpu);
    step1.chirp_cpu = cpu.target_waveform;
    step1.target_cpu = cpu.target_component;
    step1.interference_cpu = cpu.interference_component;
    step1.noise_cpu = cpu.noise_component;
    step1.echo_cpu = cpu.echo;
    append(evaluate_step1_formal(gpu, step1, thresholds));

    auto step2 = collect_step2_validation(gpu);
    step2.filtered_cpu = cpu.filtered;
    append(evaluate_step2_formal(gpu, step2, thresholds));

    auto step3 = collect_step3_validation(gpu);
    step3.smoothed_cpu = cpu.smoothed;
    // reference_for_correlation is a direct crop/view mapping of this pipeline's
    // Step1 target waveform, so its frozen zero-error gate is same-input by design.
    // The independently generated CPU pipeline remains the reference for smoothed.
    append(evaluate_step3_formal(gpu, step3, thresholds));

    auto step4 = collect_step4_validation(gpu);
    step4.fm_cpu = cpu.fm_feature;
    step4.correlation_cpu = cpu.correlation_feature;
    step4.spectral_cpu = cpu.spectral_feature;
    step4.wavelet_cpu = cpu.wavelet_feature;
    step4.bundle_cpu = cpu.feature_bundle;
    append(evaluate_step4_formal(gpu, step4, thresholds));

    Step5ValidationData step5 = collect_step5_cpu_validation(gpu);
    const auto step5_cuda = collect_step5_cuda_validation(gpu);
    step5.injected_gpu = step5_cuda.injected_gpu;
    step5.moved_gpu = step5_cuda.moved_gpu;
    const auto step5_semantic = compare_step5_accuracy(gpu, step5);
    const std::string step5_status = step5_semantic.passed
        ? "PASS" : "FAIL:" + step5_semantic.failure_reason;
    results.push_back(discrete_result(gpu.extrema, cpu.extrema, thresholds,
        "Task2.step5.extrema", "extrema", step5_status));

    const auto step6 = collect_step6_validation(gpu);
    const auto step6_semantic = compare_step6_accuracy(gpu, step6);
    const std::string step6_status = step6_semantic.passed
        ? "PASS" : "FAIL:" + step6_semantic.failure_reason;
    results.push_back(numeric_result(std::vector<float>{gpu.estimate},
        std::vector<float>{cpu.estimate}, thresholds,
        "Task2.step6.estimate", "estimate", step6_status));
    results.push_back(numeric_result(gpu.kalman_covariance,
        cpu.kalman_covariance, thresholds,
        "Task2.step6.covariance", "covariance", step6_status));
    return results;
}

void merge_worst_accuracy(
    std::map<std::string, OutputAccuracyResult>& aggregate,
    const AccuracyResults& sample)
{
    for (const auto& result : sample) {
        auto inserted = aggregate.emplace(result.threshold.target, result);
        if (inserted.second) continue;
        auto& current = inserted.first->second;
        current.mse = std::max(current.mse, result.mse);
        current.rmse = std::max(current.rmse, result.rmse);
        current.relative_l2 = std::max(current.relative_l2, result.relative_l2);
        current.relative_linf = std::max(current.relative_linf, result.relative_linf);
        current.exact_match = current.exact_match && result.exact_match;
        current.mismatch_count = std::max(current.mismatch_count, result.mismatch_count);
        current.pass = current.pass && result.pass;
        if (!result.pass) current.semantic_check = result.semantic_check;
    }
}

double worst_mse(const std::map<std::string, OutputAccuracyResult>& values)
{ return worst(values, &OutputAccuracyResult::mse); }
double worst_rmse(const std::map<std::string, OutputAccuracyResult>& values)
{ return worst(values, &OutputAccuracyResult::rmse); }
double worst_relative_l2(const std::map<std::string, OutputAccuracyResult>& values)
{ return worst(values, &OutputAccuracyResult::relative_l2); }
double worst_relative_linf(const std::map<std::string, OutputAccuracyResult>& values)
{ return worst(values, &OutputAccuracyResult::relative_linf); }

bool all_accuracy_pass(const AccuracyResults& values)
{
    return std::all_of(values.begin(), values.end(),
        [](const OutputAccuracyResult& value) { return value.pass; });
}

std::string accuracy_failure_message(const AccuracyResults& values)
{
    std::ostringstream message;
    for (const auto& result : values) {
        if (result.pass) continue;
        if (message.tellp() != std::streampos(0)) message << "; ";
        message << result.threshold.target;
        if (result.threshold.numeric)
            message << " mse=" << result.mse << "/" << result.threshold.mse_max
                    << " rmse=" << result.rmse << "/" << result.threshold.rmse_max
                    << " relative_l2=" << result.relative_l2 << "/"
                    << result.threshold.relative_l2_max << " relative_linf="
                    << result.relative_linf << "/" << result.threshold.relative_linf_max;
        else
            message << " mismatch_count=" << result.mismatch_count << "/"
                    << result.threshold.max_mismatch_count;
        message << " semantic=" << result.semantic_check;
    }
    return message.str().empty() ? "all accuracy outputs passed" : message.str();
}

void write_formal_accuracy_csv(
    const std::filesystem::path& path, const AccuracyCsvContext& context,
    const FormalThresholdSet& thresholds,
    const std::map<std::string, OutputAccuracyResult>& aggregate)
{
    zkx::common::CsvWriter csv(path, {
        "schema_version", "run_id", "run_type", "git_commit", "case_id", "scale_id",
        "backend", "measured_runs", "threshold_set_id", "thresholds_sha256", "target",
        "output_name", "dtype", "output_kind", "mse", "mse_max", "rmse", "rmse_max",
        "relative_l2", "relative_l2_max", "relative_linf", "relative_linf_max",
        "relative_floor", "exact_match", "mismatch_count", "max_mismatch_count",
        "semantic_rule", "semantic_check", "accuracy_status"});
    for (const auto& item : aggregate) {
        const auto& result = item.second;
        const auto& threshold = result.threshold;
        csv.append({"1", context.run_id, context.run_type, context.git_commit,
            context.case_id, context.scale_id, context.backend,
            std::to_string(context.measured_runs), thresholds.id(), thresholds.sha256(),
            threshold.target, threshold.output_name, threshold.dtype,
            threshold.output_kind,
            threshold.numeric ? number(result.mse) : "NA",
            threshold.numeric ? number(threshold.mse_max) : "NA",
            threshold.numeric ? number(result.rmse) : "NA",
            threshold.numeric ? number(threshold.rmse_max) : "NA",
            threshold.numeric ? number(result.relative_l2) : "NA",
            threshold.numeric ? number(threshold.relative_l2_max) : "NA",
            threshold.numeric ? number(result.relative_linf) : "NA",
            threshold.numeric ? number(threshold.relative_linf_max) : "NA",
            threshold.numeric ? number(threshold.relative_floor) : "NA",
            threshold.numeric ? "NA" : (result.exact_match ? "true" : "false"),
            threshold.numeric ? "NA" : std::to_string(result.mismatch_count),
            threshold.numeric ? "NA" : std::to_string(threshold.max_mismatch_count),
            threshold.semantic_rule, result.semantic_check,
            result.pass ? "PASS" : "FAIL"});
    }
}

}  // namespace task2::accuracy
