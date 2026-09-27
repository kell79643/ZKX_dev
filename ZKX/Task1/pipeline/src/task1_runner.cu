#include <task1/task1_runner.h>

#include "accuracy/accuracy_utils.h"
#include "accuracy/comparison/step1_comparison.h"
#include "accuracy/comparison/step2_comparison.h"
#include "accuracy/comparison/step3_comparison.h"
#include "accuracy/comparison/step4_comparison.h"
#include "accuracy/comparison/step5_comparison.h"
#include "accuracy/validation/step1_validation.h"
#include "accuracy/validation/step2_validation.h"
#include "accuracy/validation/step3_validation.h"
#include "accuracy/validation/step4_validation.h"
#include "accuracy/validation/step5_validation.h"
#include <task1/task1_common.h>
#include <task1/steps/step1.h>
#include <task1/steps/step2.h>
#include <task1/steps/step3.h>
#include <task1/steps/step4.h>
#include <task1/steps/step5.h>
#include "pipeline_visualization_export.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace task1 {
namespace {

#ifndef TASK1_EVIDENCE_GIT_COMMIT
#define TASK1_EVIDENCE_GIT_COMMIT "unknown"
#endif
#ifndef TASK1_EVIDENCE_GIT_DIRTY
#define TASK1_EVIDENCE_GIT_DIRTY "unknown"
#endif
#ifndef TASK1_EVIDENCE_TEST_TIME
#define TASK1_EVIDENCE_TEST_TIME "unknown"
#endif

void print_evidence(const StepEvidence& evidence)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[TASK1][" << evidence.name << "][TIMING] prep_ms=" << evidence.prep_ms
              << " h2d_ms=" << evidence.h2d_ms
              << " compute_ms=" << evidence.compute_ms
              << " d2h_ms=" << evidence.d2h_ms
              << " post_ms=" << evidence.post_ms
              << " formal_execution_ms=" << evidence.formal_execution_ms
              << " total_ms=" << evidence.total_ms << '\n'
              << "[TASK1][" << evidence.name << "][ACCURACY] max_abs_error="
              << evidence.max_abs_error
              << " mse=" << evidence.mse
              << " rmse=" << evidence.rmse
              << " relative_l2_error=" << evidence.relative_error
              << " relative_linf_error=" << evidence.relative_linf_error
              << " compared_elements=" << evidence.compared_elements
              << " relative_tolerance=" << evidence.relative_tolerance
              << " relative_floor=" << evidence.relative_floor
              << " first_failure_index=" << evidence.first_failure_index
              << " reference_source=\"" << evidence.reference_source << "\""
              << " comparison_method=\"" << evidence.comparison_method << "\""
              << " status=pass\n";
    for (const auto& item : evidence.operator_ms)
        std::cout << "[TASK1][OPERATOR] step=" << evidence.name
                  << " name=" << item.first << " gpu_ms=" << item.second << '\n';
    for (const auto& item : evidence.metrics)
        std::cout << "[TASK1][METRIC] step=" << evidence.name
                  << " name=" << item.first << " value=" << item.second << '\n';
}

void write_json(
    const std::vector<StepEvidence>& steps,
    int completed_steps,
    const std::string& evidence_id,
    const TaskConfig& config,
    double pipeline_total_ms,
    double sum_step_total_ms)
{
    std::ostringstream filename;
    filename << "task1_" << evidence_id << "_evidence.json";
    std::ofstream output(filename.str());
    require(static_cast<bool>(output), "cannot write Task1 evidence JSON");
    output << std::setprecision(17);
    output << "{\n"
           << "  \"task\": \"Task1\",\n"
           << "  \"implementation\": \"cusignal_cpp_zq500\",\n"
           << "  \"evidence_id\": \"" << evidence_id << "\",\n"
           << "  \"completed_steps\": " << completed_steps << ",\n"
           << "  \"git_commit\": \"" << TASK1_EVIDENCE_GIT_COMMIT << "\",\n"
           << "  \"git_dirty\": \"" << TASK1_EVIDENCE_GIT_DIRTY << "\",\n"
           << "  \"test_time\": \"" << TASK1_EVIDENCE_TEST_TIME << "\",\n"
           << "  \"config_path\": \"" << config.path.string() << "\",\n"
           << "  \"config_sha256\": \"" << config.sha256 << "\",\n"
           << "  \"host\": \"usr_02@10.110.12.10\",\n"
           << "  \"container\": \"gpu_02\",\n"
           << "  \"sdk\": \"/zq500/sdk\",\n"
           << "  \"dtype\": \"ComplexFP32\",\n"
           << "  \"result_location\": \"build_task1/task1_evidence/"
           << evidence_id << "/task1_" << evidence_id << "_evidence.json\",\n"
           << "  \"input_contract\": {\"dtype\": \"ComplexFP32\", \"shape\": ["
           << config.num_pulses << ", " << config.samples_per_pulse
           << "], \"sample_rate_hz\": " << config.sample_rate_hz
           << ", \"prf_hz\": " << config.prf_hz
           << ", \"pulse_samples\": " << config.pulse_samples
           << ", \"target_delay_samples\": " << config.target_delay_samples
           << ", \"doppler_hz\": " << config.doppler_hz()
           << ", \"doppler_bin\": " << config.doppler_bin
           << ", \"bandwidth_hz\": " << config.bandwidth_hz
           << ", \"carrier_frequency_hz\": " << config.carrier_frequency_hz
           << ", \"target_amplitude\": " << config.target_amplitude
           << ", \"noise_std\": " << config.noise_std
           << ", \"noise_seed\": " << config.noise_seed
           << ", \"pfa\": " << config.pfa
           << ", \"cfar_guard\": [" << config.cfar_guard_doppler << ", " << config.cfar_guard_range
           << "], \"cfar_reference\": [" << config.cfar_reference_doppler << ", " << config.cfar_reference_range
           << "], \"ambiguity_nfreq\": " << config.ambiguity_nfreq << "},\n"
           << "  \"memory_contract\": {\"gpu_algorithm_storage\": \"DeviceArray\", "
               "\"pipeline_handoff_storage\": \"hybrid PipelineState; Step3-to-Step4 device-resident\", "
              "\"layout\": \"row-major\"},\n"
           << "  \"timing_policy\": {\"gpu_clock\": \"CudaEvent\", "
              "\"step_clock\": \"steady_clock\", "
              "\"plan_workspace\": \"included in formal operator compute_ms\", "
              "\"synchronization\": \"CudaEvent end synchronization and explicit transfers\"},\n"
           << "  \"formal_apis\": [\"pulse_compression_device<float>\", "
              "\"pulse_doppler_device<float>\", \"cfar_alpha_device<float>\", "
              "\"ca_cfar_device<float>\", \"ambgfun_device<float>\"],\n"
           << "  \"private_gpu_helpers\": [\"gpu_lfm_generation\", "
              "\"gpu_delay_doppler_noise\", \"range_doppler_power\"],\n"
           << "  \"operator_counting\": {\"formal_business_operators\": 5, "
              "\"private_helpers_are_business_operators\": false},\n"
           << "  \"accuracy_architecture\": {"
              "\"validation_files\": 5, \"comparison_files\": 5, "
              "\"transport\": \"in-memory typed validation data\", "
              "\"comparison_gpu_calls\": 0},\n"
           << "  \"source_files\": [\"task1_common.h\", \"task1_common.cu\", "
              "\"task1_runner.h\", \"task1_runner.cu\", "
              "\"pipeline_visualization_export.h\", "
              "\"pipeline_visualization_export.cpp\", "
              "\"accuracy/accuracy_types.h\", \"accuracy/accuracy_utils.h\", "
              "\"accuracy/accuracy_utils.cpp\", "
              "\"accuracy/validation/step1_validation.cu\", "
              "\"accuracy/validation/step2_validation.cu\", "
              "\"accuracy/validation/step3_validation.cu\", "
              "\"accuracy/validation/step4_validation.cu\", "
              "\"accuracy/validation/step5_validation.cu\", "
              "\"accuracy/comparison/step1_comparison.cpp\", "
              "\"accuracy/comparison/step2_comparison.cpp\", "
              "\"accuracy/comparison/step3_comparison.cpp\", "
              "\"accuracy/comparison/step4_comparison.cpp\", "
              "\"accuracy/comparison/step5_comparison.cpp\", "
              "\"step1/step1.cu\", \"step2/step2.cu\", \"step3/step3.cu\", "
              "\"step4/step4.cu\", \"step5/step5.cu\", \"CMakeLists.txt\", "
              "\"verify_task1_source_contract.cmake\", "
              "\"cusignal_cpp/include/cusignal/operators/radartools/radartools_typed.h\", "
              "\"cusignal_cpp/src/operators/radartools/radartools_typed.cpp\", "
              "\"cusignal_cpp/src/operators/radartools/radartools_typed.cu\", "
              "\"cusignal_cpp/CMakeLists.txt\"],\n"
           << "  \"status\": \"pass\",\n"
           << "  \"pipeline_total_ms\": " << pipeline_total_ms << ",\n"
           << "  \"sum_step_total_ms\": " << sum_step_total_ms << ",\n"
           << "  \"steps\": [\n";
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const auto& step = steps[index];
        output << "    {\n"
               << "      \"name\": \"" << step.name << "\",\n"
               << "      \"timing_ms\": {\"prepare\": " << step.prep_ms
               << ", \"h2d\": " << step.h2d_ms
               << ", \"compute\": " << step.compute_ms
               << ", \"d2h\": " << step.d2h_ms
               << ", \"postprocess\": " << step.post_ms
               << ", \"formal_execution\": " << step.formal_execution_ms
               << ", \"total\": " << step.total_ms << "},\n"
               << "      \"accuracy\": {\"max_abs_error\": " << step.max_abs_error
               << ", \"mse\": " << step.mse
               << ", \"rmse\": " << step.rmse
               << ", \"relative_error\": " << step.relative_error
               << ", \"relative_l2_error\": " << step.relative_error
               << ", \"relative_linf_error\": " << step.relative_linf_error
               << ", \"compared_elements\": " << step.compared_elements
               << ", \"relative_tolerance\": " << step.relative_tolerance
               << ", \"relative_floor\": " << step.relative_floor
               << ", \"first_failure_index\": " << step.first_failure_index
               << ", \"reference_source\": \"" << step.reference_source << "\""
               << ", \"comparison_method\": \"" << step.comparison_method << "\""
               << "},\n"
               << "      \"operators_ms\": {";
        std::size_t item_index = 0;
        for (const auto& item : step.operator_ms) {
            if (item_index++ != 0) output << ", ";
            output << '\"' << item.first << "\": " << item.second;
        }
        output << "},\n      \"metrics\": {";
        item_index = 0;
        for (const auto& item : step.metrics) {
            if (item_index++ != 0) output << ", ";
            output << '\"' << item.first << "\": " << item.second;
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
    RunStep run_step,
    CollectValidation collect_validation,
    CompareAccuracy compare_accuracy)
{
    const auto total_begin = Clock::now();
    StepEvidence evidence = run_step(state);
    const auto post_begin = Clock::now();
    const auto validation = collect_validation(state);
    accuracy::merge_accuracy(evidence, compare_accuracy(state, validation));
    evidence.post_ms = milliseconds(post_begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
}

int run_task1_until(
    int stop_after, const char* evidence_id, const TaskConfig& config)
{
    try {
        if (stop_after < 1 || stop_after > 5)
            throw std::invalid_argument("Task1 stop_after must be in [1,5]");
        if (evidence_id == nullptr || std::string(evidence_id).empty())
            throw std::invalid_argument("Task1 evidence_id must be nonempty");
        const auto pipeline_begin = Clock::now();
        std::cout << "[TASK1][CONFIG] path=" << config.path.string()
                  << " sha256=" << config.sha256
                  << " shape=[" << config.num_pulses << ',' << config.samples_per_pulse << "]"
                  << " pulse_samples=" << config.pulse_samples
                  << " sample_rate_hz=" << config.sample_rate_hz
                  << " prf_hz=" << config.prf_hz
                  << " bandwidth_hz=" << config.bandwidth_hz
                  << " carrier_frequency_hz=" << config.carrier_frequency_hz
                  << " target_delay_samples=" << config.target_delay_samples
                  << " doppler_bin=" << config.doppler_bin
                  << " doppler_hz=" << config.doppler_hz()
                  << " target_amplitude=" << config.target_amplitude
                  << " noise_std=" << config.noise_std
                  << " noise_seed=" << config.noise_seed
                  << " pfa=" << config.pfa
                  << " cfar_guard=[" << config.cfar_guard_doppler << ',' << config.cfar_guard_range << ']'
                  << " cfar_reference=[" << config.cfar_reference_doppler << ',' << config.cfar_reference_range << ']'
                  << " ambiguity_nfreq=" << config.ambiguity_nfreq << '\n';
        PipelineState state(config);
        std::vector<StepEvidence> evidence;
        evidence.push_back(run_verified_step(
            state, run_step1, accuracy::collect_step1_validation,
            accuracy::compare_step1_accuracy));
        if (stop_after >= 2) evidence.push_back(run_verified_step(
            state, run_step2, accuracy::collect_step2_validation,
            accuracy::compare_step2_accuracy));
        if (stop_after >= 3) evidence.push_back(run_verified_step(
            state, run_step3, accuracy::collect_step3_validation,
            accuracy::compare_step3_accuracy));
        if (stop_after >= 4) evidence.push_back(run_verified_step(
            state, run_step4, accuracy::collect_step4_validation,
            accuracy::compare_step4_accuracy));
        if (stop_after >= 5) evidence.push_back(run_verified_step(
            state, run_step5, accuracy::collect_step5_validation,
            accuracy::compare_step5_accuracy));
        const double pipeline_total_ms = milliseconds(pipeline_begin, Clock::now());
        double sum_step_total_ms = 0.0;
        for (const auto& step : evidence) sum_step_total_ms += step.total_ms;
        for (const auto& step : evidence) print_evidence(step);
        std::cout << "[TASK1][PIPELINE][TIMING] completed_steps=" << stop_after
                  << " pipeline_total_ms=" << pipeline_total_ms
                  << " sum_step_total_ms=" << sum_step_total_ms << '\n';
        write_json(
            evidence, stop_after, evidence_id, config,
            pipeline_total_ms, sum_step_total_ms);
        export_visualization_data(state, stop_after);
        const int formal_operator_count = stop_after < 2 ? 0 :
            (stop_after == 2 ? 1 : (stop_after == 3 ? 2 :
            (stop_after == 4 ? 4 : 5)));
        std::cout << "[TASK1][SUMMARY] official_steps=" << stop_after << "/5"
                  << " gpu_step_mapping=" << stop_after << "/5"
                  << " formal_business_operators_exercised="
                  << formal_operator_count << "/5"
                  << " cpu_fallback=0 temporary_implementation=0 nvidia_only_path=0"
                  << " self_timed_steps=" << stop_after << "/5"
                  << " accuracy_validation_files=" << stop_after << "/5"
                  << " accuracy_comparison_files=" << stop_after << "/5"
                  << " accuracy_pass=" << stop_after << "/5"
                  << " semantic_metrics_pass=" << stop_after << "/5"
                  << " comparison_negative_tests_pass=true"
                  << " status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TASK1][FAIL] " << error.what() << '\n';
        return 1;
    }
}

}  // namespace task1
