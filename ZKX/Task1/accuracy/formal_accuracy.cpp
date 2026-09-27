#include "accuracy/formal_accuracy.h"

#include "accuracy/accuracy_utils.h"
#include "accuracy/comparison/step1_comparison.h"
#include "accuracy/comparison/step2_comparison.h"
#include "accuracy/comparison/step3_comparison.h"
#include "accuracy/comparison/step4_comparison.h"
#include "accuracy/comparison/step5_comparison.h"
#include "accuracy/validation/step4_validation.h"
#include <demo/config_file.h>
#include "test_all/support/config/json_value.h"
#include "test_all/support/metrics/csv_writer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace task1::accuracy {
namespace {

using zkx::config::JsonValue;

std::string key(const std::string& target, const std::string& dtype,
                const std::string& output_name)
{
    return target + "\n" + dtype + "\n" + output_name;
}

const JsonValue& member(const JsonValue& object, const char* name)
{
    const JsonValue* value = object.find(name);
    if (value == nullptr) throw std::invalid_argument(
        std::string("accuracy threshold missing field: ") + name);
    return *value;
}

double nonnegative_number(const JsonValue& object, const char* name)
{
    const double value = member(object, name).as_number();
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string("invalid accuracy threshold: ") + name);
    return value;
}

void reject_unknown(const JsonValue& object, const std::vector<std::string>& allowed,
                    const std::string& where)
{
    if (!object.is_object()) throw std::invalid_argument(where + " must be an object");
    for (const auto& item : object.as_object())
        if (std::find(allowed.begin(), allowed.end(), item.first) == allowed.end())
            throw std::invalid_argument(where + " has unknown field: " + item.first);
}

template <class Vector>
OutputAccuracyResult numeric_result(
    const Vector& actual, const Vector& expected, const FormalThresholdSet& thresholds,
    const std::string& target, const std::string& dtype,
    const std::string& output_name, const std::string& semantic)
{
    const auto& threshold = thresholds.require(target, dtype, output_name);
    if (!threshold.numeric) throw std::invalid_argument(target + " must use numeric thresholds");
    OutputAccuracyResult result;
    result.threshold = threshold;
    result.mse = mean_squared_error(actual, expected);
    result.rmse = std::sqrt(result.mse);
    result.relative_l2 = relative_l2_error(actual, expected, threshold.relative_floor);
    result.relative_linf = relative_linf_error(actual, expected, threshold.relative_floor);
    result.exact_match = false;
    result.semantic_check = threshold.semantic_rule + ":" + semantic;
    result.pass = std::isfinite(result.mse) && std::isfinite(result.rmse) &&
        std::isfinite(result.relative_l2) && std::isfinite(result.relative_linf) &&
        result.mse <= threshold.mse_max && result.rmse <= threshold.rmse_max &&
        result.relative_l2 <= threshold.relative_l2_max &&
        result.relative_linf <= threshold.relative_linf_max && semantic == "PASS";
    return result;
}

OutputAccuracyResult scalar_result(
    double actual, double expected, const FormalThresholdSet& thresholds,
    const std::string& target, const std::string& dtype,
    const std::string& output_name, const std::string& semantic)
{
    return numeric_result(std::vector<double>{actual}, std::vector<double>{expected},
        thresholds, target, dtype, output_name, semantic);
}

OutputAccuracyResult detection_result(
    const std::vector<cusignal::CfarDetection>& actual,
    const std::vector<cusignal::CfarDetection>& expected,
    const FormalThresholdSet& thresholds, const std::string& semantic)
{
    const auto& threshold = thresholds.require(
        "Task1.step4.detections", "FP32", "detections");
    if (threshold.numeric) throw std::invalid_argument("detections must use discrete thresholds");
    if (actual.size() != expected.size())
        throw std::runtime_error("Task1.step4.detections shape mismatch");
    OutputAccuracyResult result;
    result.threshold = threshold;
    for (std::size_t index = 0; index < actual.size(); ++index)
        if (actual[index] != expected[index]) ++result.mismatch_count;
    result.exact_match = result.mismatch_count == 0;
    result.semantic_check = threshold.semantic_rule + ":" + semantic;
    result.pass = (!threshold.require_exact_match || result.exact_match) &&
        result.mismatch_count <= threshold.max_mismatch_count && semantic == "PASS";
    return result;
}

template <class Semantic>
AccuracyResults with_semantic(AccuracyResults results, Semantic semantic)
{
    try { semantic(); }
    catch (const std::exception& error) {
        for (auto& result : results) {
            result.semantic_check = result.threshold.semantic_rule + ":FAIL:" + error.what();
            result.pass = false;
        }
    }
    return results;
}

AccuracyResults step1_results(
    const PipelineState& actual, const Step1ValidationData& expected,
    const FormalThresholdSet& thresholds)
{
    return {
        numeric_result(actual.waveform, expected.waveform_cpu, thresholds,
            "Task1.step1.waveform", "FP32", "waveform", "PASS"),
        numeric_result(actual.noiseless_echo, expected.noiseless_cpu, thresholds,
            "Task1.step1.noiseless_echo", "FP32", "noiseless_echo", "PASS"),
        numeric_result(actual.noise, expected.noise_cpu, thresholds,
            "Task1.step1.noise", "FP32", "noise", "PASS"),
        numeric_result(actual.echo, expected.echo_cpu, thresholds,
            "Task1.step1.echo", "FP32", "echo", "PASS")};
}

AccuracyResults step2_results(
    const PipelineState& actual, const Step2ValidationData& expected,
    const FormalThresholdSet& thresholds)
{
    return {numeric_result(actual.compressed, expected.compressed_cpu, thresholds,
        "Task1.step2.compressed", "FP32", "compressed", "PASS")};
}

AccuracyResults step3_results(
    const PipelineState& actual, const Step3ValidationData& expected,
    const FormalThresholdSet& thresholds)
{
    return {numeric_result(actual.range_doppler, expected.range_doppler_cpu, thresholds,
        "Task1.step3.range_doppler", "FP32", "range_doppler", "PASS")};
}

AccuracyResults step4_results(
    const PipelineState& actual, const Step4ValidationData& expected,
    const FormalThresholdSet& thresholds)
{
    return {
        numeric_result(actual.power, expected.power_cpu, thresholds,
            "Task1.step4.power", "FP32", "power", "PASS"),
        numeric_result(actual.cfar_threshold, expected.threshold_cpu, thresholds,
            "Task1.step4.cfar_threshold", "FP32", "cfar_threshold", "PASS"),
        scalar_result(actual.cfar_alpha, expected.alpha_cpu, thresholds,
            "Task1.step4.cfar_alpha", "FP32", "cfar_alpha", "PASS"),
        detection_result(actual.detections, expected.detections_same_power_cpu,
            thresholds, "PASS")};
}

AccuracyResults step5_results(
    const PipelineState& actual, const Step5ValidationData& expected,
    const FormalThresholdSet& thresholds)
{
    const std::vector<double> ambiguity_2d(actual.ambiguity_2d.begin(), actual.ambiguity_2d.end());
    return {
        numeric_result(ambiguity_2d, expected.ambiguity_2d_cpu, thresholds,
            "Task1.step5.ambiguity_2d", "FP32", "ambiguity_2d", "PASS"),
        numeric_result(actual.ambiguity_delay, expected.ambiguity_delay_cpu, thresholds,
            "Task1.step5.ambiguity_delay", "FP32", "ambiguity_delay", "PASS"),
        numeric_result(actual.ambiguity_doppler, expected.ambiguity_doppler_cpu, thresholds,
            "Task1.step5.ambiguity_doppler", "FP32", "ambiguity_doppler", "PASS")};
}

template <class T>
void append(std::vector<T>& target, std::vector<T> source)
{
    target.insert(target.end(), std::make_move_iterator(source.begin()),
                  std::make_move_iterator(source.end()));
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
    reject_unknown(document, {"schema_version", "threshold_set_id", "entries"},
                   "accuracy threshold document");
    if (member(document, "schema_version").as_number() != 1.0)
        throw std::invalid_argument("accuracy threshold schema_version must be 1");
    FormalThresholdSet set;
    set.id_ = member(document, "threshold_set_id").as_string();
    set.sha256_ = demo_config::sha256(bytes);
    const auto& entries = member(document, "entries").as_array();
    if (entries.empty()) throw std::invalid_argument("accuracy threshold entries must not be empty");
    const std::vector<std::string> allowed{
        "target", "dtype", "output_name", "output_kind", "mse_max", "rmse_max",
        "relative_l2_max", "relative_linf_max", "relative_floor", "require_exact_match",
        "max_mismatch_count", "semantic_rule"};
    for (const JsonValue& item : entries) {
        reject_unknown(item, allowed, "accuracy threshold entry");
        FormalThreshold threshold;
        threshold.target = member(item, "target").as_string();
        threshold.dtype = member(item, "dtype").as_string();
        threshold.output_name = member(item, "output_name").as_string();
        threshold.output_kind = member(item, "output_kind").as_string();
        threshold.semantic_rule = member(item, "semantic_rule").as_string();
        const std::vector<std::string> dtypes{
            "FP32", "FP16", "INT32", "INT16", "INT8"};
        const std::vector<std::string> output_kinds{"floating", "complex", "discrete"};
        if (std::find(dtypes.begin(), dtypes.end(), threshold.dtype) == dtypes.end())
            throw std::invalid_argument("unsupported accuracy threshold dtype: " + threshold.dtype);
        if (std::find(output_kinds.begin(), output_kinds.end(), threshold.output_kind) == output_kinds.end())
            throw std::invalid_argument("unsupported accuracy output_kind: " + threshold.output_kind);
        if (threshold.target.empty() || threshold.output_name.empty() ||
            threshold.semantic_rule.empty())
            throw std::invalid_argument("accuracy threshold identity and semantic_rule must be nonempty");
        threshold.require_exact_match = member(item, "require_exact_match").as_boolean();
        threshold.numeric = threshold.output_kind != "discrete";
        if (threshold.numeric) {
            threshold.mse_max = nonnegative_number(item, "mse_max");
            threshold.rmse_max = nonnegative_number(item, "rmse_max");
            threshold.relative_l2_max = nonnegative_number(item, "relative_l2_max");
            threshold.relative_linf_max = nonnegative_number(item, "relative_linf_max");
            threshold.relative_floor = nonnegative_number(item, "relative_floor");
            if (threshold.relative_floor <= 0.0)
                throw std::invalid_argument("relative_floor must be positive");
            if (!member(item, "max_mismatch_count").is_null())
                throw std::invalid_argument("numeric threshold max_mismatch_count must be null");
        } else {
            for (const char* name : {"mse_max", "rmse_max", "relative_l2_max",
                                     "relative_linf_max", "relative_floor"})
                if (!member(item, name).is_null())
                    throw std::invalid_argument(std::string("discrete threshold must use null: ") + name);
            const double count = member(item, "max_mismatch_count").as_number();
            if (count < 0.0 || std::floor(count) != count)
                throw std::invalid_argument("max_mismatch_count must be a nonnegative integer");
            threshold.max_mismatch_count = static_cast<std::size_t>(count);
        }
        const std::string entry_key = key(
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
    if (found == entries_.end()) throw std::invalid_argument(
        "missing exact accuracy threshold: " + target + "/" + dtype + "/" + output_name);
    return found->second;
}

AccuracyResults evaluate_step1_formal(
    const PipelineState& state, const Step1ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    return with_semantic(step1_results(state, validation, thresholds),
        [&] { (void)compare_step1_accuracy(state, validation); });
}
AccuracyResults evaluate_step2_formal(
    const PipelineState& state, const Step2ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    return with_semantic(step2_results(state, validation, thresholds),
        [&] { (void)compare_step2_accuracy(state, validation); });
}
AccuracyResults evaluate_step3_formal(
    const PipelineState& state, const Step3ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    return with_semantic(step3_results(state, validation, thresholds),
        [&] { (void)compare_step3_accuracy(state, validation); });
}
AccuracyResults evaluate_step4_formal(
    const PipelineState& state, const Step4ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    return with_semantic(step4_results(state, validation, thresholds),
        [&] { (void)compare_step4_accuracy(state, validation); });
}
AccuracyResults evaluate_step5_formal(
    const PipelineState& state, const Step5ValidationData& validation,
    const FormalThresholdSet& thresholds)
{
    return with_semantic(step5_results(state, validation, thresholds),
        [&] { (void)compare_step5_accuracy(state, validation); });
}

AccuracyResults evaluate_pipeline_formal(
    const PipelineState& gpu, const PipelineState& cpu, const FormalThresholdSet& thresholds)
{
    AccuracyResults results;
    Step1ValidationData step1;
    step1.waveform_cpu = cpu.waveform;
    step1.noiseless_cpu = cpu.noiseless_echo;
    step1.noise_cpu = cpu.noise;
    step1.echo_cpu = cpu.echo;
    append(results, step1_results(gpu, step1, thresholds));
    Step2ValidationData step2; step2.compressed_cpu = cpu.compressed;
    append(results, step2_results(gpu, step2, thresholds));
    Step3ValidationData step3; step3.range_doppler_cpu = cpu.range_doppler;
    append(results, step3_results(gpu, step3, thresholds));
    Step4ValidationData step4;
    step4.power_cpu = cpu.power;
    step4.threshold_cpu = cpu.cfar_threshold;
    step4.alpha_cpu = cpu.cfar_alpha;
    // cuSignal defines detections as a discontinuous, full-array comparison
    // against its threshold mask.  Verify that operator result against the CPU
    // reference fed by the same GPU-produced power array; independent pipeline
    // mask differences remain separate end-to-end diagnostics.
    step4.detections_cpu = cpu.detections;
    step4.detections_same_power_cpu =
        generate_step4_same_power_detections_cpu(gpu);
    append(results, step4_results(gpu, step4, thresholds));
    Step5ValidationData step5;
    step5.ambiguity_2d_cpu.assign(cpu.ambiguity_2d.begin(), cpu.ambiguity_2d.end());
    step5.ambiguity_delay_cpu = cpu.ambiguity_delay;
    step5.ambiguity_doppler_cpu = cpu.ambiguity_doppler;
    append(results, step5_results(gpu, step5, thresholds));
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
    }
}

double worst_mse(const std::map<std::string, OutputAccuracyResult>& values)
{ double v=0.0; for(const auto& x:values) if(x.second.threshold.numeric) v=std::max(v,x.second.mse); return v; }
double worst_rmse(const std::map<std::string, OutputAccuracyResult>& values)
{ double v=0.0; for(const auto& x:values) if(x.second.threshold.numeric) v=std::max(v,x.second.rmse); return v; }
double worst_relative_l2(const std::map<std::string, OutputAccuracyResult>& values)
{ double v=0.0; for(const auto& x:values) if(x.second.threshold.numeric) v=std::max(v,x.second.relative_l2); return v; }
double worst_relative_linf(const std::map<std::string, OutputAccuracyResult>& values)
{ double v=0.0; for(const auto& x:values) if(x.second.threshold.numeric) v=std::max(v,x.second.relative_linf); return v; }

bool all_accuracy_pass(const AccuracyResults& values)
{
    return std::all_of(values.begin(), values.end(),
        [](const OutputAccuracyResult& value) { return value.pass; });
}

std::string accuracy_failure_message(const AccuracyResults& values)
{
    std::ostringstream message;
    bool first = true;
    for (const auto& result : values) {
        if (result.pass) continue;
        if (!first) message << "; ";
        first = false;
        message << result.threshold.target;
        if (result.threshold.numeric)
            message << " mse=" << result.mse << "/" << result.threshold.mse_max
                    << " rmse=" << result.rmse << "/" << result.threshold.rmse_max
                    << " relative_l2=" << result.relative_l2 << "/"
                    << result.threshold.relative_l2_max
                    << " relative_linf=" << result.relative_linf << "/"
                    << result.threshold.relative_linf_max;
        else message << " mismatch_count=" << result.mismatch_count << "/"
                     << result.threshold.max_mismatch_count;
        message << " semantic=" << result.semantic_check;
    }
    return first ? "all accuracy outputs passed" : message.str();
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
    auto number = [](double value) {
        std::ostringstream output; output.precision(17); output << value; return output.str();
    };
    for (const auto& item : aggregate) {
        const auto& result = item.second;
        const auto& threshold = result.threshold;
        csv.append({"1", context.run_id, context.run_type, context.git_commit,
            context.case_id, context.scale_id, context.backend,
            std::to_string(context.measured_runs), thresholds.id(), thresholds.sha256(),
            threshold.target, threshold.output_name, threshold.dtype, threshold.output_kind,
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
            threshold.semantic_rule, result.semantic_check, result.pass ? "PASS" : "FAIL"});
    }
}

}  // namespace task1::accuracy
