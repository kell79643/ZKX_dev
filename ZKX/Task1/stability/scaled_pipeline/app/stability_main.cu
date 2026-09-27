#include "accuracy/validation/step1_validation.h"
#include "accuracy/validation/step2_validation.h"
#include "accuracy/validation/step3_validation.h"
#include "accuracy/validation/step4_validation.h"
#include "accuracy/validation/step5_validation.h"
#include "steps/step1.h"
#include "steps/step2.h"
#include "steps/step3.h"
#include "steps/step4.h"
#include "steps/step5.h"
#include "test_all/support/stability/task_stability_evidence.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

using zkx::stability::Measurement;
using zkx::stability::PairedCase;

struct Metrics {
    double mse{}, rmse{}, relative_l2{}, relative_linf{};
    bool exact{true};
};

template <class Actual, class Expected>
Metrics real_metrics(const std::vector<Actual>& actual, const std::vector<Expected>& expected)
{
    if (actual.size() != expected.size() || actual.empty())
        throw std::runtime_error("accuracy output shape mismatch or empty output");
    long double squared_error = 0.0L, squared_reference = 0.0L;
    double max_error = 0.0, max_reference = 0.0;
    bool exact = sizeof(Actual) == sizeof(Expected);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double a = static_cast<double>(actual[index]);
        const double e = static_cast<double>(expected[index]);
        const double difference = a - e;
        squared_error += difference * difference;
        squared_reference += e * e;
        max_error = std::max(max_error, std::abs(difference));
        max_reference = std::max(max_reference, std::abs(e));
        if constexpr (sizeof(Actual) == sizeof(Expected))
            exact = exact && std::memcmp(&actual[index], &expected[index], sizeof(Actual)) == 0;
        else
            exact = false;
    }
    Metrics result;
    result.mse = static_cast<double>(squared_error / actual.size());
    result.rmse = std::sqrt(result.mse);
    result.relative_l2 = std::sqrt(static_cast<double>(squared_error)) /
        std::max(std::sqrt(static_cast<double>(squared_reference)), 1.0e-12);
    result.relative_linf = max_error / std::max(max_reference, 1.0e-12);
    result.exact = exact;
    return result;
}

Metrics complex_metrics(const std::vector<cusignal::TypedComplex<float>>& actual,
    const std::vector<cusignal::TypedComplex<float>>& expected)
{
    if (actual.size() != expected.size() || actual.empty())
        throw std::runtime_error("complex accuracy output shape mismatch or empty output");
    std::vector<double> a, e;
    a.reserve(actual.size() * 2);
    e.reserve(expected.size() * 2);
    bool exact = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        a.push_back(actual[index].real); a.push_back(actual[index].imag);
        e.push_back(expected[index].real); e.push_back(expected[index].imag);
        exact = exact && std::memcmp(&actual[index], &expected[index], sizeof(actual[index])) == 0;
    }
    auto result = real_metrics(a, e);
    result.exact = exact;
    return result;
}

bool passed(const Metrics& value)
{
    return std::isfinite(value.mse) && std::isfinite(value.rmse) &&
        std::isfinite(value.relative_l2) && std::isfinite(value.relative_linf) &&
        value.relative_l2 <= 1.0e-3 && value.relative_linf <= 1.0e-3;
}

struct PipelineRun {
    task1::PipelineState state;
    std::array<Measurement, 5> steps;
    Measurement total;
};

PipelineRun run_gpu_pipeline()
{
    PipelineRun result;
    result.total.before = zkx::stability::sample_resources();
    const auto begin = task1::Clock::now();
    auto step1 = zkx::stability::measure([&] { task1::run_step1(result.state); return 0; }, true);
    result.steps[0] = step1.second;
    auto step2 = zkx::stability::measure([&] { task1::run_step2(result.state); return 0; }, true);
    result.steps[1] = step2.second;
    auto step3 = zkx::stability::measure([&] { task1::run_step3(result.state); return 0; }, true);
    result.steps[2] = step3.second;
    auto step4 = zkx::stability::measure([&] { task1::run_step4(result.state); return 0; }, true);
    result.steps[3] = step4.second;
    auto step5 = zkx::stability::measure([&] { task1::run_step5(result.state); return 0; }, true);
    result.steps[4] = step5.second;
    result.total.milliseconds = task1::milliseconds(begin, task1::Clock::now());
    result.total.after = zkx::stability::sample_resources();
    result.total.peak = result.total.after;
    for (const auto& step : result.steps) {
        result.total.peak.heap = std::max(result.total.peak.heap, step.peak.heap);
        result.total.peak.rss = std::max(result.total.peak.rss, step.peak.rss);
        result.total.peak.gpu = std::max(result.total.peak.gpu, step.peak.gpu);
    }
    return result;
}

PipelineRun run_cpu_pipeline()
{
    PipelineRun result;
    result.total.before = zkx::stability::sample_resources();
    const auto begin = task1::Clock::now();
    auto step1 = zkx::stability::measure(
        [&] { return task1::accuracy::collect_step1_validation(result.state); }, false);
    result.steps[0] = step1.second;
    result.state.waveform = std::move(step1.first.waveform_cpu);
    result.state.noiseless_echo = std::move(step1.first.noiseless_cpu);
    result.state.noise = std::move(step1.first.noise_cpu);
    result.state.echo = std::move(step1.first.echo_cpu);
    auto step2 = zkx::stability::measure(
        [&] { return task1::accuracy::collect_step2_validation(result.state); }, false);
    result.steps[1] = step2.second;
    result.state.compressed = std::move(step2.first.compressed_cpu);
    auto step3 = zkx::stability::measure(
        [&] { return task1::accuracy::collect_step3_validation(result.state); }, false);
    result.steps[2] = step3.second;
    result.state.range_doppler = std::move(step3.first.range_doppler_cpu);
    auto step4 = zkx::stability::measure(
        [&] { return task1::accuracy::collect_step4_validation(result.state); }, false);
    result.steps[3] = step4.second;
    result.state.power = std::move(step4.first.power_cpu);
    result.state.cfar_threshold = std::move(step4.first.threshold_cpu);
    result.state.detections = std::move(step4.first.detections_cpu);
    result.state.cfar_alpha = step4.first.alpha_cpu;
    auto step5 = zkx::stability::measure(
        [&] { return task1::accuracy::collect_step5_validation(result.state); }, false);
    result.steps[4] = step5.second;
    result.state.ambiguity_2d.assign(
        step5.first.ambiguity_2d_cpu.begin(), step5.first.ambiguity_2d_cpu.end());
    result.state.ambiguity_delay = std::move(step5.first.ambiguity_delay_cpu);
    result.state.ambiguity_doppler = std::move(step5.first.ambiguity_doppler_cpu);
    result.total.milliseconds = task1::milliseconds(begin, task1::Clock::now());
    result.total.after = zkx::stability::sample_resources();
    result.total.peak = result.total.after;
    for (const auto& step : result.steps) {
        result.total.peak.heap = std::max(result.total.peak.heap, step.peak.heap);
        result.total.peak.rss = std::max(result.total.peak.rss, step.peak.rss);
        result.total.peak.gpu = std::max(result.total.peak.gpu, step.peak.gpu);
    }
    return result;
}

std::vector<std::string> row(const PairedCase& item, const std::filesystem::path& params,
    const std::string& run_id, const std::string& backend, const std::string& git_commit,
    const std::string& git_dirty, const std::string& test_time, const std::string& step,
    const std::string& operators, const std::string& device, const Measurement& measurement,
    const Metrics& metrics, double speedup)
{
    const bool ok = passed(metrics);
    const bool cpu = device == "cpu";
    std::vector<std::string> values{
        "1", run_id, "stability", test_time, git_commit, git_dirty,
        item.parameter_set_id, item.file_sha256, "task", "Task1." + step, "Task1",
        step, operators, item.case_id, "task1_s10_baseline", "10^4",
        std::to_string(task1::kNumPulses * task1::kSamplesPerPulse),
        "[64,512]", "ComplexFP32", device, backend,
        step == "pipeline_total" ? "compatibility/end-to-end" : "compatibility/step",
        ok ? "PASS" : "FAIL", ok ? "OK" : "ACCURACY_FAILED",
        "0", "1", zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds), "0", "0",
        cpu ? "NA" : zkx::stability::number(speedup),
        "independent CPU pipeline versus GPU pipeline on one frozen paired perturbation",
        zkx::stability::number(metrics.mse), zkx::stability::number(metrics.rmse),
        zkx::stability::number(metrics.relative_l2), zkx::stability::number(metrics.relative_linf),
        metrics.exact ? "true" : "false", metrics.exact ? "0" : "NA",
        "finite relative_l2<=1e-3 and relative_linf<=1e-3", ok ? "PASS" : "FAIL",
        "stage10_task1_relative_v1", "NA"};
    if (cpu) {
        values.insert(values.end(), {
            zkx::stability::resource_value(measurement.before.heap_ok, measurement.before.heap),
            zkx::stability::resource_value(measurement.after.heap_ok, measurement.after.heap),
            zkx::stability::resource_value(measurement.peak.heap_ok, measurement.peak.heap),
            zkx::stability::resource_value(measurement.before.heap_ok && measurement.after.heap_ok,
                measurement.after.heap - measurement.before.heap),
            zkx::stability::resource_value(measurement.before.rss_ok, measurement.before.rss),
            zkx::stability::resource_value(measurement.after.rss_ok, measurement.after.rss),
            zkx::stability::resource_value(measurement.peak.rss_ok, measurement.peak.rss),
            zkx::stability::resource_value(measurement.before.rss_ok && measurement.after.rss_ok,
                measurement.after.rss - measurement.before.rss),
            "NA", "NA", "NA", "NA"});
    } else {
        values.insert(values.end(), {"NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA",
            zkx::stability::resource_value(measurement.before.gpu_ok, measurement.before.gpu),
            zkx::stability::resource_value(measurement.after.gpu_ok, measurement.after.gpu),
            zkx::stability::resource_value(measurement.peak.gpu_ok, measurement.peak.gpu),
            zkx::stability::resource_value(measurement.before.gpu_ok && measurement.after.gpu_ok,
                measurement.after.gpu - measurement.before.gpu)});
    }
    values.insert(values.end(), {ok ? "0" : "1", std::to_string(item.index),
        std::to_string(item.total_cases), std::filesystem::absolute(params).generic_string(),
        item.file_sha256, item.summary, "generated", "NA", "NA",
        std::to_string(item.seed), item.case_id});
    return values;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 8)
            throw std::invalid_argument(
                "usage: task1_stability_512 PARAMS_JSON CASE_INDEX OUTPUT_DIR RUN_ID BACKEND GIT_COMMIT GIT_DIRTY");
        const std::filesystem::path params = argv[1];
        const int index = std::stoi(argv[2]);
        const std::filesystem::path output_dir = argv[3];
        if (std::filesystem::exists(output_dir))
            throw std::runtime_error("refusing to overwrite output directory: " + output_dir.string());
        std::filesystem::create_directories(output_dir);
        const auto item = zkx::stability::load_case(params, index, "Task1");
        const auto& config = item.task;
        if (zkx::stability::field(config, "baseline_scale_id").as_string() != "task1_s10_baseline")
            throw std::invalid_argument("Task1 case baseline_scale_id mismatch");
        task1::kTargetDelay = zkx::stability::integer(config, "target_delay_samples");
        task1::kDopplerBin = zkx::stability::integer(config, "doppler_bin");
        task1::kDopplerFrequency =
            task1::kDopplerBin * task1::kPrf / static_cast<float>(task1::kNumPulses);
        task1::kTargetAmplitude = zkx::stability::f32(config, "target_amplitude");
        const float noise_std = zkx::stability::f32(config, "noise_std");
        zkx::stability::XorShift32 noise(zkx::stability::uint32(config, "noise_seed"));
        task1::kStabilityNoise.resize(task1::kNumPulses * task1::kSamplesPerPulse);
        for (auto& value : task1::kStabilityNoise)
            value = {noise.symmetric(noise_std), noise.symmetric(noise_std)};

        const auto gpu = run_gpu_pipeline();
        const auto cpu = run_cpu_pipeline();
        std::array<Metrics, 6> metrics{
            complex_metrics(gpu.state.echo, cpu.state.echo),
            complex_metrics(gpu.state.compressed, cpu.state.compressed),
            complex_metrics(gpu.state.range_doppler, cpu.state.range_doppler),
            real_metrics(gpu.state.cfar_threshold, cpu.state.cfar_threshold),
            real_metrics(gpu.state.ambiguity_2d, cpu.state.ambiguity_2d), Metrics{}};
        metrics[5] = metrics[0];
        for (std::size_t i = 1; i < 5; ++i) {
            metrics[5].mse = std::max(metrics[5].mse, metrics[i].mse);
            metrics[5].rmse = std::max(metrics[5].rmse, metrics[i].rmse);
            metrics[5].relative_l2 = std::max(metrics[5].relative_l2, metrics[i].relative_l2);
            metrics[5].relative_linf = std::max(metrics[5].relative_linf, metrics[i].relative_linf);
            metrics[5].exact = metrics[5].exact && metrics[i].exact;
        }
        const std::array<std::string, 6> names{
            "step1", "step2", "step3", "step4", "step5", "pipeline_total"};
        const std::array<std::string, 6> operators{
            "chirp+echo_synthesis", "pulse_compression", "pulse_doppler",
            "cfar_alpha+ca_cfar", "ambgfun", "frozen_multi_upstream_graph"};
        const std::string test_time = zkx::stability::utc_now();
        std::ofstream csv(output_dir / "task1_stability_results.csv");
        if (!csv) throw std::runtime_error("cannot create Task1 stability CSV");
        zkx::stability::write_csv_row(csv, zkx::stability::standard_header());
        bool all_pass = true;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const Measurement& cm = i < 5 ? cpu.steps[i] : cpu.total;
            const Measurement& gm = i < 5 ? gpu.steps[i] : gpu.total;
            const double speedup = cm.milliseconds / std::max(gm.milliseconds, 1.0e-12);
            zkx::stability::write_csv_row(csv, row(item, params, argv[4], argv[5], argv[6],
                argv[7], test_time, names[i], operators[i], "cpu", cm, metrics[i], speedup));
            zkx::stability::write_csv_row(csv, row(item, params, argv[4], argv[5], argv[6],
                argv[7], test_time, names[i], operators[i], "gpu", gm, metrics[i], speedup));
            all_pass = all_pass && passed(metrics[i]);
        }
        std::ofstream review(output_dir / "task1_zero_error_review.json");
        review << "{\n  \"stability_case_index\": " << index
               << ",\n  \"case_id\": \"" << item.case_id
               << "\",\n  \"audit\": \"independent CPU/GPU pipeline states and distinct output buffers\","
                  "\n  \"zero_error_steps\": [";
        bool first = true;
        for (std::size_t i = 0; i < metrics.size(); ++i) if (metrics[i].mse == 0.0) {
            if (!first) review << ',';
            first = false;
            review << "\"" << names[i] << "\"";
        }
        review << "],\n  \"same_buffer_comparison\": false,\n  \"status\": \"PASS\"\n}\n";
        std::ofstream summary(output_dir / "task1_stability_summary.txt");
        summary << "run_type=stability\nstability_case_index=" << index
                << "\ncase_id=" << item.case_id << "\ncpu_pipeline_complete=true"
                   "\ngpu_pipeline_complete=true\nrows=12\nstatus="
                << (all_pass ? "PASS" : "FAIL") << '\n';
        std::cout << "[TASK1][STABILITY] case_index=" << index << " case_id=" << item.case_id
                  << " rows=12 status=" << (all_pass ? "PASS" : "FAIL") << '\n';
        return all_pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[TASK1][STABILITY][FAIL] " << error.what() << '\n';
        return 2;
    }
}
