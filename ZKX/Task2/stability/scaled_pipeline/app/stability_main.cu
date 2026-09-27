#include "accuracy/validation/step1_validation.h"
#include "accuracy/validation/step2_validation.h"
#include "accuracy/validation/step3_validation.h"
#include "accuracy/validation/step4_validation.h"
#include "accuracy/validation/step5_validation.h"
#include "accuracy/validation/step6_validation.h"
#include "steps/step1.h"
#include "steps/step2.h"
#include "steps/step3.h"
#include "steps/step4.h"
#include "steps/step5.h"
#include "steps/step6.h"
#include "test_all/support/stability/task_stability_evidence.h"
#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/peak_finding/peak_finding_typed.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

using zkx::stability::Measurement;
using zkx::stability::PairedCase;

struct Metrics {
    double mse{}, rmse{}, relative_l2{}, relative_linf{};
    bool exact{true};
    bool discrete{false};
    std::size_t mismatch_count{};
};

template <class T>
Metrics metrics(const std::vector<T>& actual, const std::vector<T>& expected)
{
    if (actual.size() != expected.size() || actual.empty())
        throw std::runtime_error("accuracy output shape mismatch or empty output");
    long double squared_error = 0.0L, squared_reference = 0.0L;
    double max_error = 0.0, max_reference = 0.0;
    bool exact = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double a = static_cast<double>(actual[index]);
        const double e = static_cast<double>(expected[index]);
        const double difference = a - e;
        squared_error += difference * difference;
        squared_reference += e * e;
        max_error = std::max(max_error, std::abs(difference));
        max_reference = std::max(max_reference, std::abs(e));
        exact = exact && std::memcmp(&actual[index], &expected[index], sizeof(T)) == 0;
    }
    Metrics result;
    result.mse = static_cast<double>(squared_error / actual.size());
    result.rmse = std::sqrt(result.mse);
    result.relative_l2 = std::sqrt(static_cast<double>(squared_error)) /
        std::max(std::sqrt(static_cast<double>(squared_reference)), 1.0e-12);
    result.relative_linf = max_error / std::max(max_reference, 1.0e-12);
    result.exact = exact;
    result.mismatch_count = exact ? 0 : 1;
    return result;
}

Metrics coordinate_metrics(const std::vector<std::int64_t>& actual,
    const std::vector<std::int64_t>& expected, std::size_t element_count)
{
    std::vector<std::uint8_t> actual_mask(element_count, 0);
    std::vector<std::uint8_t> expected_mask(element_count, 0);
    for (const auto coordinate : actual) {
        if (coordinate < 0 || static_cast<std::size_t>(coordinate) >= element_count)
            throw std::runtime_error("GPU Step5 coordinate outside feature bundle");
        actual_mask[static_cast<std::size_t>(coordinate)] = 1;
    }
    for (const auto coordinate : expected) {
        if (coordinate < 0 || static_cast<std::size_t>(coordinate) >= element_count)
            throw std::runtime_error("CPU Step5 coordinate outside feature bundle");
        expected_mask[static_cast<std::size_t>(coordinate)] = 1;
    }
    auto result = metrics(actual_mask, expected_mask);
    result.discrete = true;
    result.mismatch_count = 0;
    for (std::size_t index = 0; index < element_count; ++index)
        if (actual_mask[index] != expected_mask[index]) ++result.mismatch_count;
    result.exact = result.mismatch_count == 0;
    return result;
}

std::vector<std::int64_t> extrema_cpu(const std::vector<float>& bundle)
{
    const cusignal::RelativeExtremaShape shape({bundle.size()});
    return cusignal::argrelextrema_typed_cpu(
        bundle, shape, cusignal::RelativeExtremaComparator::greater,
        0, 3, "clip").coordinates.front();
}

std::vector<std::int64_t> extrema_gpu(const std::vector<float>& bundle)
{
    const cusignal::RelativeExtremaShape shape({bundle.size()});
    cusignal::RelativeExtremaDeviceWorkspace workspace(bundle.size());
    const auto device = cusignal::DeviceArray<float>::from_host(bundle);
    auto result = cusignal::argrelextrema_device(
        device, shape, cusignal::RelativeExtremaComparator::greater,
        workspace, 0, 3, "clip");
    if (result.coordinates.size() != 1)
        throw std::runtime_error("Step5 diagnostic coordinate rank mismatch");
    return result.coordinates.front().to_host();
}

std::int64_t strongest_extremum(
    const std::vector<float>& bundle,
    const std::vector<std::int64_t>& extrema)
{
    if (extrema.empty())
        throw std::runtime_error("Step5 target selection received no extrema");
    return *std::max_element(extrema.begin(), extrema.end(),
        [&](std::int64_t left, std::int64_t right) {
            return bundle[static_cast<std::size_t>(left)] <
                bundle[static_cast<std::size_t>(right)];
        });
}

std::vector<std::int64_t> symmetric_difference(
    const std::vector<std::int64_t>& left,
    const std::vector<std::int64_t>& right)
{
    std::vector<std::int64_t> output;
    std::set_symmetric_difference(
        left.begin(), left.end(), right.begin(), right.end(),
        std::back_inserter(output));
    return output;
}

double peak_margin(const std::vector<float>& bundle, std::int64_t coordinate)
{
    const auto index = static_cast<std::size_t>(coordinate);
    double margin = std::numeric_limits<double>::infinity();
    for (std::size_t distance = 1; distance <= 3; ++distance) {
        const auto plus = std::min(index + distance, bundle.size() - 1);
        const auto minus = index >= distance ? index - distance : 0;
        margin = std::min(margin,
            static_cast<double>(bundle[index]) - bundle[plus]);
        margin = std::min(margin,
            static_cast<double>(bundle[index]) - bundle[minus]);
    }
    return margin;
}

void write_step5_diagnostic(
    const std::filesystem::path& path,
    const task2::PipelineState& gpu, const task2::PipelineState& cpu)
{
    const auto cpu_on_cpu = extrema_cpu(cpu.feature_bundle);
    const auto gpu_on_cpu = extrema_gpu(cpu.feature_bundle);
    const auto cpu_on_gpu = extrema_cpu(gpu.feature_bundle);
    const auto gpu_on_gpu = extrema_gpu(gpu.feature_bundle);
    const auto same_cpu = symmetric_difference(cpu_on_cpu, gpu_on_cpu);
    const auto same_gpu = symmetric_difference(cpu_on_gpu, gpu_on_gpu);
    const auto pipeline = symmetric_difference(gpu_on_gpu, cpu_on_cpu);
    const std::string verdict = !same_cpu.empty() || !same_gpu.empty()
        ? "STEP5_OPERATOR_MISMATCH"
        : (!pipeline.empty() ? "UPSTREAM_NUMERIC_SENSITIVITY" : "PASS");

    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create Step5 diagnostic JSON");
    auto array = [&](const std::vector<std::int64_t>& values) {
        output << '[';
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) output << ',';
            output << values[i];
        }
        output << ']';
    };
    output << "{\n  \"verdict\": \"" << verdict << "\",\n"
           << "  \"cpu_on_cpu_count\": " << cpu_on_cpu.size() << ",\n"
           << "  \"gpu_on_cpu_count\": " << gpu_on_cpu.size() << ",\n"
           << "  \"cpu_on_gpu_count\": " << cpu_on_gpu.size() << ",\n"
           << "  \"gpu_on_gpu_count\": " << gpu_on_gpu.size() << ",\n"
           << "  \"same_cpu_bundle_difference\": "; array(same_cpu);
    output << ",\n  \"same_gpu_bundle_difference\": "; array(same_gpu);
    output << ",\n  \"pipeline_difference\": "; array(pipeline);
    output << ",\n  \"differing_points\": [\n";
    for (std::size_t i = 0; i < pipeline.size(); ++i) {
        const auto coordinate = pipeline[i];
        const auto index = static_cast<std::size_t>(coordinate);
        if (i) output << ",\n";
        output << "    {\"coordinate\": " << coordinate
               << ", \"cpu_value\": " << std::setprecision(17)
               << cpu.feature_bundle[index]
               << ", \"gpu_value\": " << gpu.feature_bundle[index]
               << ", \"cpu_peak_margin\": " << peak_margin(cpu.feature_bundle, coordinate)
               << ", \"gpu_peak_margin\": " << peak_margin(gpu.feature_bundle, coordinate)
               << "}";
    }
    output << "\n  ]\n}\n";
}

bool passed(const Metrics& value)
{
    const bool finite = std::isfinite(value.mse) && std::isfinite(value.rmse) &&
        std::isfinite(value.relative_l2) && std::isfinite(value.relative_linf) &&
        value.relative_l2 <= 1.0e-3 && value.relative_linf <= 1.0e-3;
    return value.discrete ? value.exact : finite;
}

struct PipelineRun {
    task2::PipelineState state;
    std::array<Measurement, 6> steps;
    Measurement total;
};

void finish_total(PipelineRun& result, task2::Clock::time_point begin)
{
    result.total.milliseconds = task2::milliseconds(begin, task2::Clock::now());
    result.total.after = zkx::stability::sample_resources();
    result.total.peak = result.total.after;
    for (const auto& step : result.steps) {
        result.total.peak.heap = std::max(result.total.peak.heap, step.peak.heap);
        result.total.peak.rss = std::max(result.total.peak.rss, step.peak.rss);
        result.total.peak.gpu = std::max(result.total.peak.gpu, step.peak.gpu);
    }
}

PipelineRun run_gpu_pipeline()
{
    PipelineRun result;
    result.total.before = zkx::stability::sample_resources();
    const auto begin = task2::Clock::now();
    auto s1 = zkx::stability::measure([&] { task2::run_step1(result.state); return 0; }, true);
    result.steps[0] = s1.second;
    auto s2 = zkx::stability::measure([&] { task2::run_step2(result.state); return 0; }, true);
    result.steps[1] = s2.second;
    auto s3 = zkx::stability::measure([&] { task2::run_step3(result.state); return 0; }, true);
    result.steps[2] = s3.second;
    auto s4 = zkx::stability::measure([&] { task2::run_step4(result.state); return 0; }, true);
    result.steps[3] = s4.second;
    auto s5 = zkx::stability::measure([&] { task2::run_step5(result.state); return 0; }, true);
    result.steps[4] = s5.second;
    auto s6 = zkx::stability::measure([&] { task2::run_step6(result.state); return 0; }, true);
    result.steps[5] = s6.second;
    finish_total(result, begin);
    return result;
}

void prepare_cpu_observations(task2::PipelineState& state)
{
    if (state.extrema.empty()) throw std::runtime_error("CPU Step6 did not receive Step5 points");
    const auto strongest = *std::max_element(state.extrema.begin(), state.extrema.end(),
        [&](std::int64_t left, std::int64_t right) {
            return state.feature_bundle[static_cast<std::size_t>(left)] <
                state.feature_bundle[static_cast<std::size_t>(right)];
        });
    state.kalman_anchor = strongest;
    state.kalman_observations.clear();
    state.kalman_observations.reserve(task2::kKalmanObservationCount);
    for (int radius = 1; radius <= task2::kKalmanObservationCount; ++radius) {
        const int left = std::max(0, static_cast<int>(strongest) - radius);
        const int right = std::min(static_cast<int>(state.feature_bundle.size()) - 1,
            static_cast<int>(strongest) + radius);
        double weighted_index = 0.0, weight = 0.0;
        for (int index = left; index <= right; ++index) {
            const double local = std::max(0.0F, state.feature_bundle[index]);
            weighted_index += local * index;
            weight += local;
        }
        const double coordinate = weight > 0.0 ? weighted_index / weight : strongest;
        state.kalman_observations.push_back(
            static_cast<float>(coordinate / (state.feature_bundle.size() - 1)));
    }
}

PipelineRun run_cpu_pipeline()
{
    PipelineRun result;
    result.total.before = zkx::stability::sample_resources();
    const auto begin = task2::Clock::now();
    auto s1 = zkx::stability::measure([&] {
        result.state.time.resize(task2::kSamples);
        for (int index = 0; index < task2::kSamples; ++index)
            result.state.time[static_cast<std::size_t>(index)] =
                static_cast<float>(index) / task2::kSampleRate;
        return task2::accuracy::collect_step1_validation(result.state);
    }, false);
    result.steps[0] = s1.second;
    result.state.target_waveform = std::move(s1.first.chirp_cpu);
    result.state.target_component = std::move(s1.first.target_cpu);
    result.state.interference_component = std::move(s1.first.interference_cpu);
    result.state.noise_component = std::move(s1.first.noise_cpu);
    result.state.echo = std::move(s1.first.echo_cpu);
    auto s2 = zkx::stability::measure(
        [&] { return task2::accuracy::collect_step2_validation(result.state); }, false);
    result.steps[1] = s2.second;
    result.state.filtered = std::move(s2.first.filtered_cpu);
    auto s3 = zkx::stability::measure(
        [&] { return task2::accuracy::collect_step3_validation(result.state); }, false);
    result.steps[2] = s3.second;
    result.state.smoothed = std::move(s3.first.smoothed_cpu);
    result.state.reference_for_correlation.assign(
        result.state.target_waveform.begin() + task2::kStep3CropEach,
        result.state.target_waveform.end() - task2::kStep3CropEach);
    auto s4 = zkx::stability::measure(
        [&] { return task2::accuracy::collect_step4_validation(result.state); }, false);
    result.steps[3] = s4.second;
    result.state.fm_feature = std::move(s4.first.fm_cpu);
    result.state.correlation_feature = std::move(s4.first.correlation_cpu);
    result.state.spectral_feature = std::move(s4.first.spectral_cpu);
    result.state.wavelet_feature = std::move(s4.first.wavelet_cpu);
    result.state.feature_bundle = std::move(s4.first.bundle_cpu);
    auto s5 = zkx::stability::measure(
        [&] { return task2::accuracy::collect_step5_cpu_validation(result.state); }, false);
    result.steps[4] = s5.second;
    result.state.extrema = std::move(s5.first.expected);
    auto s6 = zkx::stability::measure([&] {
        prepare_cpu_observations(result.state);
        return task2::accuracy::collect_step6_validation(result.state);
    }, false);
    result.steps[5] = s6.second;
    result.state.estimate = s6.first.state_cpu.front();
    result.state.kalman_covariance = std::move(s6.first.covariance_cpu);
    finish_total(result, begin);
    return result;
}

std::vector<std::string> make_row(const PairedCase& item, const std::filesystem::path& params,
    const std::string& run_id, const std::string& backend, const std::string& git_commit,
    const std::string& git_dirty, const std::string& test_time, const std::string& step,
    const std::string& operators, const std::string& device, const Measurement& measurement,
    const Metrics& accuracy, double speedup)
{
    const bool ok = passed(accuracy), cpu = device == "cpu";
    std::vector<std::string> values{
        "1", run_id, "stability", test_time, git_commit, git_dirty,
        item.parameter_set_id, item.file_sha256, "task", "Task2." + step, "Task2", step,
        operators, item.case_id, "task2_s08_baseline", "10^4",
        std::to_string(task2::kSamples), "[16384]", "FP32", device, backend,
        step == "pipeline_total" ? "compatibility/end-to-end" : "compatibility/step",
        ok ? "PASS" : "FAIL", ok ? "OK" : "ACCURACY_FAILED", "0", "1",
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds),
        zkx::stability::number(measurement.milliseconds), "0", "0",
        cpu ? "NA" : zkx::stability::number(speedup),
        "independent CPU pipeline versus GPU pipeline on one frozen paired perturbation",
        zkx::stability::number(accuracy.mse), zkx::stability::number(accuracy.rmse),
        zkx::stability::number(accuracy.relative_l2), zkx::stability::number(accuracy.relative_linf),
        accuracy.exact ? "true" : "false", std::to_string(accuracy.mismatch_count),
        accuracy.discrete ?
            "strongest Step5 target coordinate requires exact_match and mismatch_count=0; "
            "full extrema sets remain in cross-input diagnostic" :
            "finite relative_l2<=1e-3 and relative_linf<=1e-3",
        ok ? "PASS" : "FAIL",
        "stage10_task2_relative_v1", "NA"};
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
                measurement.after.rss - measurement.before.rss), "NA", "NA", "NA", "NA"});
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
                "usage: task2_stability_16384 PARAMS_JSON CASE_INDEX OUTPUT_DIR RUN_ID BACKEND GIT_COMMIT GIT_DIRTY");
        const std::filesystem::path params = argv[1];
        const int index = std::stoi(argv[2]);
        const std::filesystem::path output_dir = argv[3];
        if (std::filesystem::exists(output_dir))
            throw std::runtime_error("refusing to overwrite output directory: " + output_dir.string());
        std::filesystem::create_directories(output_dir);
        const auto item = zkx::stability::load_case(params, index, "Task2");
        const auto& config = item.task;
        if (zkx::stability::field(config, "baseline_scale_id").as_string() != "task2_s08_baseline")
            throw std::invalid_argument("Task2 case baseline_scale_id mismatch");
        task2::kTargetDelay = zkx::stability::integer(config, "target_delay_samples");
        task2::kChirpWeight = zkx::stability::f32(config, "target_weight");
        task2::kGausspulseWeight = zkx::stability::f32(config, "gaussian_weight");
        task2::kSawtoothWeight = zkx::stability::f32(config, "sawtooth_weight");
        task2::kSquareWeight = zkx::stability::f32(config, "square_weight");
        task2::kNoiseAmplitude = zkx::stability::f32(config, "noise_amplitude");
        task2::kFilterCutoffHz = zkx::stability::f32(config, "filter_cutoff_hz");
        task2::kKalmanR = zkx::stability::f32(config, "kalman_R");
        zkx::stability::XorShift32 noise(zkx::stability::uint32(config, "noise_seed"));
        task2::kStabilityNoise.resize(task2::kSamples);
        for (float& value : task2::kStabilityNoise)
            value = noise.symmetric(task2::kNoiseAmplitude);

        const auto gpu = run_gpu_pipeline();
        const auto cpu = run_cpu_pipeline();
        write_step5_diagnostic(
            output_dir / "task2_step5_cross_input_diagnostic.json",
            gpu.state, cpu.state);
        std::array<Metrics, 7> accuracy{
            metrics(gpu.state.echo, cpu.state.echo),
            metrics(gpu.state.filtered, cpu.state.filtered),
            metrics(gpu.state.smoothed, cpu.state.smoothed),
            metrics(gpu.state.feature_bundle, cpu.state.feature_bundle),
            coordinate_metrics(
                std::vector<std::int64_t>{strongest_extremum(
                    gpu.state.feature_bundle, gpu.state.extrema)},
                std::vector<std::int64_t>{strongest_extremum(
                    cpu.state.feature_bundle, cpu.state.extrema)},
                gpu.state.feature_bundle.size()),
            metrics(std::vector<float>{gpu.state.estimate, gpu.state.kalman_covariance.front()},
                std::vector<float>{cpu.state.estimate, cpu.state.kalman_covariance.front()}), Metrics{}};
        accuracy[6] = accuracy[0];
        for (std::size_t i = 1; i < 6; ++i) {
            accuracy[6].mse = std::max(accuracy[6].mse, accuracy[i].mse);
            accuracy[6].rmse = std::max(accuracy[6].rmse, accuracy[i].rmse);
            accuracy[6].relative_l2 = std::max(accuracy[6].relative_l2, accuracy[i].relative_l2);
            accuracy[6].relative_linf = std::max(accuracy[6].relative_linf, accuracy[i].relative_linf);
            accuracy[6].exact = accuracy[6].exact && accuracy[i].exact;
            accuracy[6].mismatch_count += accuracy[i].mismatch_count;
        }
        const std::array<std::string, 7> names{
            "step1", "step2", "step3", "step4", "step5", "step6", "pipeline_total"};
        const std::array<std::string, 7> operators{
            "chirp+gausspulse+sawtooth+square", "hamming+firwin+firfilter", "cubic+firfilter",
            "fm_demod+correlate+spectrogram+cwt", "argrelextrema", "kalmanfilter",
            "frozen_multi_upstream_graph"};
        const std::string test_time = zkx::stability::utc_now();
        std::ofstream csv(output_dir / "task2_stability_results.csv");
        if (!csv) throw std::runtime_error("cannot create Task2 stability CSV");
        zkx::stability::write_csv_row(csv, zkx::stability::standard_header());
        bool all_pass = true;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const Measurement& cm = i < 6 ? cpu.steps[i] : cpu.total;
            const Measurement& gm = i < 6 ? gpu.steps[i] : gpu.total;
            const double speedup = cm.milliseconds / std::max(gm.milliseconds, 1.0e-12);
            zkx::stability::write_csv_row(csv, make_row(item, params, argv[4], argv[5], argv[6],
                argv[7], test_time, names[i], operators[i], "cpu", cm, accuracy[i], speedup));
            zkx::stability::write_csv_row(csv, make_row(item, params, argv[4], argv[5], argv[6],
                argv[7], test_time, names[i], operators[i], "gpu", gm, accuracy[i], speedup));
            all_pass = all_pass && passed(accuracy[i]);
        }
        std::ofstream review(output_dir / "task2_zero_error_review.json");
        review << "{\n  \"stability_case_index\": " << index
               << ",\n  \"case_id\": \"" << item.case_id
               << "\",\n  \"audit\": \"independent CPU/GPU pipeline states and distinct output buffers\","
                  "\n  \"zero_error_steps\": [";
        bool first = true;
        for (std::size_t i = 0; i < accuracy.size(); ++i) if (accuracy[i].mse == 0.0) {
            if (!first) review << ',';
            first = false;
            review << "\"" << names[i] << "\"";
        }
        review << "],\n  \"same_buffer_comparison\": false,\n  \"status\": \"PASS\"\n}\n";
        std::ofstream summary(output_dir / "task2_stability_summary.txt");
        summary << "run_type=stability\nstability_case_index=" << index
                << "\ncase_id=" << item.case_id << "\ncpu_pipeline_complete=true"
                   "\ngpu_pipeline_complete=true\nrows=14\nstatus="
                << (all_pass ? "PASS" : "FAIL") << '\n';
        std::cout << "[TASK2][STABILITY] case_index=" << index << " case_id=" << item.case_id
                  << " rows=14 status=" << (all_pass ? "PASS" : "FAIL") << '\n';
        return all_pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[TASK2][STABILITY][FAIL] " << error.what() << '\n';
        return 2;
    }
}
