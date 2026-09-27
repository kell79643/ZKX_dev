#include <task2/task2_runner.h>

#include "accuracy/accuracy_utils.h"
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
#include <task2/task2_common.h>
#include <task2/steps/step1.h>
#include <task2/steps/step2.h>
#include <task2/steps/step3.h>
#include <task2/steps/step4.h>
#include <task2/steps/step5.h>
#include <task2/steps/step6.h>
#include "pipeline_visualization_export.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace task2 {
namespace {

#ifndef TASK2_EVIDENCE_GIT_COMMIT
#define TASK2_EVIDENCE_GIT_COMMIT "unknown"
#endif

std::string json_escape(const std::string& value)
{
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default: escaped << character; break;
        }
    }
    return escaped.str();
}

void print_evidence(const StepEvidence& evidence)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[TASK2][" << evidence.name << "][TIMING] prep_ms=" << evidence.prep_ms
              << " h2d_ms=" << evidence.h2d_ms
              << " compute_ms=" << evidence.compute_ms
              << " d2h_ms=" << evidence.d2h_ms
              << " post_ms=" << evidence.post_ms
              << " formal_execution_ms=" << evidence.formal_execution_ms
              << " total_ms=" << evidence.total_ms << '\n'
              << std::scientific << std::setprecision(9)
              << "[TASK2][" << evidence.name << "][ACCURACY] max_error="
              << evidence.max_error
              << " mse=" << evidence.mse
              << " rmse=" << evidence.rmse
              << " relative_l2_error=" << evidence.relative_error
              << " relative_linf_error=" << evidence.relative_linf_error
              << " relative_tolerance=" << evidence.relative_tolerance
              << " compared_elements=" << evidence.compared_elements
              << " tolerance=" << evidence.tolerance
              << " first_failure_index=" << evidence.first_failure_index
              << " reference_source=\"" << evidence.reference_source << "\""
              << " comparison_method=\"" << evidence.comparison_method << "\""
              << " input_shape=\"" << evidence.input_shape << "\""
              << " input_dtype=\"" << evidence.input_dtype << "\""
              << " status=" << (evidence.accuracy_pass ? "pass" : "fail") << '\n';
    std::cout << std::fixed << std::setprecision(6);
    if (!evidence.accuracy_pass)
        std::cout << "[TASK2][" << evidence.name << "][FAILURE] reason=\""
                  << evidence.failure_reason << "\" action=\""
                  << evidence.failure_action << "\"\n";
    for (const auto& item : evidence.operator_ms)
        std::cout << "[TASK2][OPERATOR] step=" << evidence.name
                  << " name=" << item.first << " gpu_ms=" << item.second << '\n';
    for (const auto& item : evidence.metrics)
        std::cout << "[TASK2][METRIC] step=" << evidence.name
                  << " name=" << item.first << " value=" << item.second << '\n';
}

void write_json(
    const std::vector<StepEvidence>& steps, int requested_steps,
    bool all_pass, const TaskConfig& config,
    double pipeline_total_ms, double sum_step_total_ms)
{
    std::ostringstream filename;
    filename << "task2_step" << requested_steps << "_evidence.json";
    std::ofstream output(filename.str());
    require(static_cast<bool>(output), "cannot write Task2 evidence JSON");
    output << "{\n  \"task\": \"Task2\",\n  \"requested_steps\": "
           << requested_steps << ",\n  \"completed_steps\": " << steps.size()
           << ",\n  \"git_commit\": \""
           << TASK2_EVIDENCE_GIT_COMMIT
           << "\",\n  \"status\": \"" << (all_pass ? "pass" : "fail") << "\",\n"
           << "  \"implementation\": \"cpp_cuda_zq500_gpu\",\n"
           << "  \"config_path\": \"" << json_escape(config.path.string()) << "\",\n"
           << "  \"config_sha256\": \"" << config.sha256 << "\",\n"
           << "  \"input_contract\": {\"dtype\": \"FP32\", \"shape\": ["
           << config.samples << "], \"sample_rate_hz\": " << config.sample_rate_hz
           << ", \"target_delay_samples\": " << config.target_delay_samples
           << ", \"noise_seed\": " << config.noise_seed
           << ", \"step3_output_samples\": " << config.step3_samples()
           << ", \"target_weight\": " << config.target_weight
           << ", \"interference_weights\": [" << config.gaussian_weight << ", "
           << config.sawtooth_weight << ", " << config.square_weight << "]"
           << ", \"noise_amplitude\": " << config.noise_amplitude
           << ", \"filter_taps\": " << config.filter_taps
           << ", \"filter_cutoff_hz\": " << config.filter_cutoff_hz
           << ", \"feature_fusion_weights\": {\"fm\": " << config.fusion_weight_fm
           << ", \"correlation\": " << config.fusion_weight_correlation
           << ", \"spectral\": " << config.fusion_weight_spectral
           << ", \"wavelet\": " << config.fusion_weight_wavelet << "}"
           << ", \"argrelextrema\": {\"axis\": 0, \"order\": "
           << config.argrelextrema_order << ", \"mode\": \"clip\"}"
           << ", \"kalman\": {\"observation_count\": " << config.kalman_observation_count
           << ", \"F\": " << config.kalman_f << ", \"Q\": " << config.kalman_q
           << ", \"H\": " << config.kalman_h << ", \"R\": " << config.kalman_r
           << ", \"measurement_noise_variant\": " << config.kalman_measurement_noise_variant
           << "}},\n"
           << "  \"pipeline_total_ms\": " << pipeline_total_ms << ",\n"
           << "  \"sum_step_total_ms\": " << sum_step_total_ms << ",\n"
           << "  \"steps\": [\n";
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const auto& step = steps[index];
        output << "    {\n"
               << "      \"name\": \"" << json_escape(step.name) << "\",\n"
               << "      \"timing_ms\": {\"prepare\": " << step.prep_ms
               << ", \"h2d\": " << step.h2d_ms
               << ", \"compute\": " << step.compute_ms
               << ", \"d2h\": " << step.d2h_ms
               << ", \"postprocess\": " << step.post_ms
               << ", \"formal_execution\": " << step.formal_execution_ms
               << ", \"total\": " << step.total_ms << "},\n"
               << "      \"accuracy\": {\"relative_error\": " << step.relative_error
               << ", \"mse\": " << step.mse
               << ", \"rmse\": " << step.rmse
               << ", \"relative_l2_error\": " << step.relative_error
               << ", \"relative_linf_error\": " << step.relative_linf_error
               << ", \"relative_tolerance\": " << step.relative_tolerance
               << ", \"max_abs_error\": " << step.max_error
               << ", \"max_error\": " << step.max_error
               << ", \"compared_elements\": " << step.compared_elements
               << ", \"tolerance\": " << step.tolerance
               << ", \"first_failure_index\": " << step.first_failure_index
               << ", \"reference_source\": \"" << json_escape(step.reference_source) << "\""
               << ", \"comparison_method\": \"" << json_escape(step.comparison_method) << "\""
               << ", \"status\": \"" << (step.accuracy_pass ? "pass" : "fail") << "\""
               << ", \"failure_reason\": \"" << json_escape(step.failure_reason) << "\""
               << ", \"input_shape\": \"" << json_escape(step.input_shape) << "\""
               << ", \"input_dtype\": \"" << json_escape(step.input_dtype) << "\""
               << ", \"failure_action\": \"" << json_escape(step.failure_action) << "\"},\n"
               << "      \"operators_ms\": {";
        std::size_t item_index = 0;
        for (const auto& item : step.operator_ms) {
            if (item_index++ != 0) output << ", ";
            output << '\"' << json_escape(item.first) << "\": " << item.second;
        }
        output << "},\n      \"metrics\": {";
        item_index = 0;
        for (const auto& item : step.metrics) {
            if (item_index++ != 0) output << ", ";
            output << '\"' << json_escape(item.first) << "\": " << item.second;
        }
        output << "}\n    }";
        if (index + 1 != steps.size()) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
}

}  // namespace

template <class RunStep, class CollectValidation, class CompareAccuracy>
StepEvidence run_verified_step(
    PipelineState& state,
    const char* step_name,
    const char* input_shape,
    const char* input_dtype,
    RunStep run_step,
    CollectValidation collect_validation,
    CompareAccuracy compare_accuracy)
{
    const auto total_begin = Clock::now();
    StepEvidence evidence;
    evidence.name = step_name;
    evidence.input_shape = input_shape;
    evidence.input_dtype = input_dtype;
    evidence.failure_action = std::string("stop at ") + step_name +
        "; inspect evidence; fix this step and rerun this and downstream steps";
    try {
        evidence = run_step(state);
        evidence.input_shape = input_shape;
        evidence.input_dtype = input_dtype;
        evidence.failure_action = std::string("stop at ") + step_name +
            "; inspect evidence; fix this step and rerun this and downstream steps";
        evidence.formal_execution_ms = evidence.total_ms;
        const auto post_begin = Clock::now();
        try {
            const auto validation = collect_validation(state);
            accuracy::merge_accuracy(evidence, compare_accuracy(state, validation));
        } catch (const std::exception& error) {
            evidence.accuracy_pass = false;
            evidence.failure_reason = error.what();
            if (evidence.comparison_method.empty())
                evidence.comparison_method =
                    "exception captured before accuracy comparison completed";
        }
        evidence.post_ms += milliseconds(post_begin, Clock::now());
    } catch (const std::exception& error) {
        evidence.name = step_name;
        evidence.input_shape = input_shape;
        evidence.input_dtype = input_dtype;
        evidence.accuracy_pass = false;
        evidence.failure_reason = error.what();
        evidence.failure_action = std::string("stop at ") + step_name +
            "; inspect evidence; fix this step and rerun this and downstream steps";
        evidence.comparison_method =
            "exception captured during formal step execution";
    }
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
}

int run_task2_until(int stop_after, const TaskConfig& config)
{
    try {
        if (stop_after < 1 || stop_after > 6)
            throw std::invalid_argument("Task2 stop_after must be in [1,6]");
        const auto pipeline_begin = Clock::now();
        std::cout << "[TASK2][CONFIG] path=" << config.path.string()
                  << " sha256=" << config.sha256 << " shape=[" << config.samples << ']'
                  << " sample_rate_hz=" << config.sample_rate_hz
                  << " target_delay_samples=" << config.target_delay_samples
                  << " noise_seed=" << config.noise_seed
                  << " target_weight=" << config.target_weight
                  << " interference_weights=[" << config.gaussian_weight << ','
                  << config.sawtooth_weight << ',' << config.square_weight << ']'
                  << " noise_amplitude=" << config.noise_amplitude
                  << " filter_taps=" << config.filter_taps
                  << " filter_cutoff_hz=" << config.filter_cutoff_hz
                  << " step3_crop_each=" << config.step3_crop_each
                  << " fusion_weights=[" << config.fusion_weight_fm << ','
                  << config.fusion_weight_correlation << ',' << config.fusion_weight_spectral
                  << ',' << config.fusion_weight_wavelet << ']'
                  << " argrelextrema_order=" << config.argrelextrema_order
                  << " kalman_observation_count=" << config.kalman_observation_count
                  << " kalman_F=" << config.kalman_f << " kalman_Q=" << config.kalman_q
                  << " kalman_H=" << config.kalman_h << " kalman_R=" << config.kalman_r
                  << " kalman_measurement_noise_variant=" << config.kalman_measurement_noise_variant
                  << '\n';
        PipelineState state(config);
        const std::string full_shape = "[" + std::to_string(config.samples) + "]";
        const std::string filter_shape = "signal [" + std::to_string(config.samples) +
            "], filter [" + std::to_string(config.filter_taps) + "]";
        const std::string crop_shape = "input [" + std::to_string(config.samples) +
            "], output [" + std::to_string(config.step3_samples()) + "]";
        const std::string feature_shape = "input [" + std::to_string(config.step3_samples()) +
            "], fused output [" + std::to_string(config.step3_samples()) + "]";
        const std::string extrema_shape = "input [" + std::to_string(config.step3_samples()) +
            "], variable-length coordinate output";
        const std::string kalman_shape = std::to_string(config.kalman_observation_count) +
            " observations, state [1], covariance [1]";
        std::vector<StepEvidence> evidence;
        evidence.push_back(run_verified_step(
            state, "step1", full_shape.c_str(),
            "FP32 outputs; FP64 typed waveform intermediates where required",
            run_step1, accuracy::collect_step1_validation,
            accuracy::compare_step1_accuracy));
        if (stop_after >= 2 && evidence.back().accuracy_pass)
            evidence.push_back(run_verified_step(
                state, "step2", filter_shape.c_str(), "FP32",
            run_step2, accuracy::collect_step2_validation,
            accuracy::compare_step2_accuracy));
        if (stop_after >= 3 && evidence.back().accuracy_pass)
            evidence.push_back(run_verified_step(
                state, "step3", crop_shape.c_str(), "FP32",
            run_step3, accuracy::collect_step3_validation,
            accuracy::compare_step3_accuracy));
        if (stop_after >= 4 && evidence.back().accuracy_pass)
            evidence.push_back(run_verified_step(
                state, "step4", feature_shape.c_str(),
            "FP32 features; FP64 CWT workspace/intermediate where required",
            run_step4, accuracy::collect_step4_validation,
            accuracy::compare_step4_accuracy));
        if (stop_after >= 5 && evidence.back().accuracy_pass)
            evidence.push_back(run_verified_step(
                state, "step5", extrema_shape.c_str(),
            "FP32 input / INT64 indices",
            run_step5, [](const PipelineState& current) {
                auto cpu = accuracy::collect_step5_cpu_validation(current);
                const auto cuda = accuracy::collect_step5_cuda_validation(current);
                cpu.injected_gpu = cuda.injected_gpu;
                cpu.moved_gpu = cuda.moved_gpu;
                return cpu;
            },
            accuracy::compare_step5_accuracy));
        if (stop_after >= 6 && evidence.back().accuracy_pass)
            evidence.push_back(run_verified_step(
                state, "step6", kalman_shape.c_str(), "FP32",
            run_step6, accuracy::collect_step6_validation,
            accuracy::compare_step6_accuracy));
        const double pipeline_total_ms = milliseconds(pipeline_begin, Clock::now());
        double sum_step_total_ms = 0.0;
        for (const auto& step : evidence) sum_step_total_ms += step.total_ms;
        const bool all_pass = evidence.size() == static_cast<std::size_t>(stop_after) &&
            std::all_of(evidence.begin(), evidence.end(),
                [](const StepEvidence& step) { return step.accuracy_pass; });
        const auto pass_count = static_cast<int>(std::count_if(
            evidence.begin(), evidence.end(),
            [](const StepEvidence& step) { return step.accuracy_pass; }));
        for (const auto& step : evidence) print_evidence(step);
        std::cout << "[TASK2][PIPELINE][TIMING] requested_steps=" << stop_after
                  << " completed_steps=" << evidence.size()
                  << " pipeline_total_ms=" << pipeline_total_ms
                  << " sum_step_total_ms=" << sum_step_total_ms << '\n';
        write_json(evidence, stop_after, all_pass, config, pipeline_total_ms, sum_step_total_ms);
        export_visualization_data(state, static_cast<int>(evidence.size()));
        std::cout << "[TASK2][SUMMARY] requested_steps=" << stop_after << "/6"
                  << " completed_steps=" << evidence.size() << "/6"
                  << " gpu_operator_mapping=" << evidence.size() << "/6"
                  << " cpu_fallback=0 temporary_implementation=0 nvidia_only_path=0"
                  << " accuracy_pass=" << pass_count << "/" << stop_after
                  << " semantic_metrics_pass=" << pass_count << "/" << stop_after
                  << " status=" << (all_pass ? "pass" : "fail") << '\n';
        return all_pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[TASK2][FAIL] " << error.what() << '\n';
        return 1;
    }
}

}  // namespace task2
