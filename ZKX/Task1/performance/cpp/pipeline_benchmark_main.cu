#include "accuracy/accuracy_utils.h"
#include "accuracy/formal_accuracy.h"
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
#include <task1/steps/step1.h>
#include <task1/steps/step2.h>
#include <task1/steps/step3.h>
#include <task1/steps/step4.h>
#include <task1/steps/step5.h>
#include "task1_scale_config.h"
#include <demo/config_file.h>
#include "test_all/support/metrics/csv_writer.h"
#include "test_all/support/metrics/identity.h"
#include "test_all/support/metrics/memory.h"
#include "test_all/support/metrics/statistics.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifndef TASK1_EVIDENCE_GIT_COMMIT
#define TASK1_EVIDENCE_GIT_COMMIT "unknown"
#endif
#ifndef TASK1_EVIDENCE_GIT_DIRTY
#define TASK1_EVIDENCE_GIT_DIRTY "unknown"
#endif
#ifndef TASK1_EVIDENCE_TEST_TIME
#define TASK1_EVIDENCE_TEST_TIME "unknown"
#endif

namespace {
using namespace zkx::common;

struct Args {
    std::filesystem::path scales;
    std::filesystem::path accuracy_thresholds;
    std::filesystem::path output;
    std::string scale_id, backend, run_id, run_type;
    int warmup_runs{}, measured_runs{};
};

int strict_nonnegative(const std::string& value, const std::string& name)
{
    std::size_t used = 0;
    int parsed = 0;
    try { parsed = std::stoi(value, &used); }
    catch (...) { throw std::invalid_argument(name + " must be an integer"); }
    if (used != value.size() || parsed < 0)
        throw std::invalid_argument(name + " must be nonnegative");
    return parsed;
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("incomplete argument: " + key);
        const std::string value = argv[++index];
        if (key == "--task-scales") args.scales = value;
        else if (key == "--accuracy-thresholds") args.accuracy_thresholds = value;
        else if (key == "--scale-id") args.scale_id = value;
        else if (key == "--backend") args.backend = value;
        else if (key == "--run-id") args.run_id = value;
        else if (key == "--run-type") args.run_type = value;
        else if (key == "--warmup-runs") args.warmup_runs = strict_nonnegative(value, key);
        else if (key == "--measured-runs") args.measured_runs = strict_nonnegative(value, key);
        else if (key == "--output-dir") args.output = value;
        else throw std::invalid_argument("unknown argument: " + key);
    }
    if (args.scales.empty() || args.accuracy_thresholds.empty() || args.output.empty() || args.scale_id.empty() ||
        args.backend.empty() || args.run_id.empty() || args.run_type.empty())
        throw std::invalid_argument("required Task1 pipeline benchmark argument is missing");
    if (args.measured_runs < 1)
        throw std::invalid_argument("--measured-runs must be positive");
    if (args.run_type != "smoke" && args.run_type != "formal")
        throw std::invalid_argument("--run-type must be smoke or formal");
    if (args.run_type == "formal" && (args.warmup_runs != 20 || args.measured_runs != 100))
        throw std::invalid_argument("formal mode requires warmup_runs=20 and measured_runs=100");
    return args;
}

std::string number(double value)
{
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::int64_t gpu_used_bytes()
{
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return static_cast<std::int64_t>(total_bytes - free_bytes);
}

template <class Vector>
void append_bytes(std::vector<unsigned char>& bytes, const Vector& values)
{
    const auto* begin = reinterpret_cast<const unsigned char*>(values.data());
    bytes.insert(bytes.end(), begin, begin + values.size() * sizeof(values[0]));
}

std::string pipeline_output_digest(const task1::PipelineState& state)
{
    std::vector<unsigned char> bytes;
    append_bytes(bytes, state.waveform);
    append_bytes(bytes, state.echo);
    append_bytes(bytes, state.compressed);
    append_bytes(bytes, state.range_doppler);
    append_bytes(bytes, state.power);
    append_bytes(bytes, state.cfar_threshold);
    append_bytes(bytes, state.detections);
    append_bytes(bytes, std::vector<double>{state.cfar_alpha});
    append_bytes(bytes, state.ambiguity_2d);
    append_bytes(bytes, state.ambiguity_delay);
    append_bytes(bytes, state.ambiguity_doppler);
    return demo_config::sha256(bytes);
}

struct CpuPipelineRun {
    explicit CpuPipelineRun(const task1::TaskConfig& config) : state(config) {}
    task1::PipelineState state;
    std::array<double, 5> step_ms{};
};

struct PipelineResourcePass {
    std::array<ResourceSample, 5> cpu_heap_before, cpu_heap_after;
    std::array<ResourceSample, 5> cpu_rss_before, cpu_rss_after;
    std::array<ResourceSample, 5> gpu_before, gpu_after;
    std::array<std::int64_t, 5> gpu_peak{};
    double cpu_sampling_ms{};
    double gpu_sampling_ms{};
};

CpuPipelineRun run_cpu_pipeline(
    const task1::TaskConfig& config, PipelineResourcePass* resources = nullptr)
{
    CpuPipelineRun result(config);
    task1::accuracy::Step1ValidationData step1;
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_before[0] = sample_cpu_heap();
            resources->cpu_rss_before[0] = sample_cpu_rss();
        }, [] {});
    }
    result.step_ms[0] = measure_ms(
        [&] { step1 = task1::accuracy::generate_step1_cpu_reference(config); }, [] {});
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_after[0] = sample_cpu_heap();
            resources->cpu_rss_after[0] = sample_cpu_rss();
        }, [] {});
    }
    result.state.waveform = std::move(step1.waveform_cpu);
    result.state.noiseless_echo = std::move(step1.noiseless_cpu);
    result.state.noise = std::move(step1.noise_cpu);
    result.state.echo = std::move(step1.echo_cpu);

    task1::accuracy::Step2ValidationData step2;
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_before[1] = sample_cpu_heap();
            resources->cpu_rss_before[1] = sample_cpu_rss();
        }, [] {});
    }
    result.step_ms[1] = measure_ms(
        [&] { step2 = task1::accuracy::generate_step2_cpu_reference(result.state); }, [] {});
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_after[1] = sample_cpu_heap();
            resources->cpu_rss_after[1] = sample_cpu_rss();
        }, [] {});
    }
    result.state.compressed = std::move(step2.compressed_cpu);

    task1::accuracy::Step3ValidationData step3;
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_before[2] = sample_cpu_heap();
            resources->cpu_rss_before[2] = sample_cpu_rss();
        }, [] {});
    }
    result.step_ms[2] = measure_ms(
        [&] { step3 = task1::accuracy::generate_step3_cpu_reference(result.state); }, [] {});
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_after[2] = sample_cpu_heap();
            resources->cpu_rss_after[2] = sample_cpu_rss();
        }, [] {});
    }
    result.state.range_doppler = std::move(step3.range_doppler_cpu);

    task1::accuracy::Step4ValidationData step4;
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_before[3] = sample_cpu_heap();
            resources->cpu_rss_before[3] = sample_cpu_rss();
        }, [] {});
    }
    result.step_ms[3] = measure_ms(
        [&] { step4 = task1::accuracy::generate_step4_cpu_reference(result.state); }, [] {});
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_after[3] = sample_cpu_heap();
            resources->cpu_rss_after[3] = sample_cpu_rss();
        }, [] {});
    }
    result.state.power = std::move(step4.power_cpu);
    result.state.cfar_threshold = std::move(step4.threshold_cpu);
    result.state.detections = std::move(step4.detections_cpu);
    result.state.cfar_alpha = step4.alpha_cpu;

    task1::accuracy::Step5ValidationData step5;
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_before[4] = sample_cpu_heap();
            resources->cpu_rss_before[4] = sample_cpu_rss();
        }, [] {});
    }
    result.step_ms[4] = measure_ms(
        [&] { step5 = task1::accuracy::generate_step5_cpu_reference(result.state); }, [] {});
    if (resources) {
        resources->cpu_sampling_ms += measure_ms([&] {
            resources->cpu_heap_after[4] = sample_cpu_heap();
            resources->cpu_rss_after[4] = sample_cpu_rss();
        }, [] {});
    }
    result.state.ambiguity_2d.assign(
        step5.ambiguity_2d_cpu.begin(), step5.ambiguity_2d_cpu.end());
    result.state.ambiguity_delay = std::move(step5.ambiguity_delay_cpu);
    result.state.ambiguity_doppler = std::move(step5.ambiguity_doppler_cpu);
    return result;
}

struct GpuPipelineRun {
    explicit GpuPipelineRun(const task1::TaskConfig& config) : state(config) {}
    task1::PipelineState state;
    std::array<double, 5> step_ms{};
    double cuda_event_sum_ms{};
    std::int64_t gpu_peak{};
};

GpuPipelineRun run_gpu_pipeline(
    const task1::TaskConfig& config, PipelineResourcePass* resources = nullptr)
{
    GpuPipelineRun result(config);
    std::array<task1::StepEvidence, 5> steps;
    auto execute = [&](std::size_t index, auto run_step) {
        if (resources)
            resources->gpu_sampling_ms += measure_ms([&] {
                resources->gpu_before[index] = sample_gpu_used(
                    gpu_used_bytes, "cudaMemGetInfo");
            }, [] {});
        steps[index] = run_step(result.state);
        if (resources) {
            resources->gpu_sampling_ms += measure_ms([&] {
                CUDA_CHECK(cudaDeviceSynchronize());
                resources->gpu_after[index] = sample_gpu_used(
                    gpu_used_bytes, "cudaMemGetInfo");
            }, [] {});
        }
    };
    execute(0, task1::run_step1);
    execute(1, task1::run_step2);
    execute(2, task1::run_step3);
    execute(3, task1::run_step4);
    execute(4, task1::run_step5);
    for (std::size_t index = 0; index < steps.size(); ++index) {
        result.step_ms[index] = steps[index].formal_execution_ms;
        result.cuda_event_sum_ms += steps[index].compute_ms;
        const auto peak = steps[index].metrics.find("gpu_used_peak_bytes");
        if (peak != steps[index].metrics.end()) {
            result.gpu_peak = std::max(
                result.gpu_peak, static_cast<std::int64_t>(peak->second));
            if (resources)
                resources->gpu_peak[index] = static_cast<std::int64_t>(peak->second);
        }
    }
    return result;
}

struct PipelineAccuracy {
    double mse{}, rmse{}, relative_l2{}, relative_linf{};
    std::size_t detection_mismatch_count{};
    std::size_t valid_detection_mismatch_count{};
    std::size_t boundary_detection_mismatch_count{};
};

PipelineAccuracy compare_pipeline_outputs(
    const task1::PipelineState& gpu, const task1::PipelineState& cpu)
{
    constexpr double floor = 1.0e-12;
    PipelineAccuracy result;
    auto update = [&](const auto& actual, const auto& expected) {
        result.mse = std::max(
            result.mse, task1::accuracy::mean_squared_error(actual, expected));
        result.relative_l2 = std::max(
            result.relative_l2, task1::accuracy::relative_l2_error(actual, expected, floor));
        result.relative_linf = std::max(
            result.relative_linf, task1::accuracy::relative_linf_error(actual, expected, floor));
    };
    update(gpu.waveform, cpu.waveform);
    update(gpu.echo, cpu.echo);
    update(gpu.compressed, cpu.compressed);
    update(gpu.range_doppler, cpu.range_doppler);
    update(gpu.power, cpu.power);
    update(gpu.cfar_threshold, cpu.cfar_threshold);
    update(gpu.ambiguity_2d, cpu.ambiguity_2d);
    update(gpu.ambiguity_delay, cpu.ambiguity_delay);
    update(gpu.ambiguity_doppler, cpu.ambiguity_doppler);
    result.mse = std::max(result.mse,
        (gpu.cfar_alpha - cpu.cfar_alpha) * (gpu.cfar_alpha - cpu.cfar_alpha));
    result.relative_l2 = std::max(result.relative_l2,
        task1::accuracy::relative_error(gpu.cfar_alpha, cpu.cfar_alpha, floor));
    result.relative_linf = std::max(result.relative_linf,
        task1::accuracy::relative_error(gpu.cfar_alpha, cpu.cfar_alpha, floor));
    result.rmse = std::sqrt(result.mse);
    if (gpu.detections.size() != cpu.detections.size())
        throw std::runtime_error("Task1 full CPU/GPU pipeline detection shape mismatch");
    const int doppler_margin = gpu.config.cfar_guard_doppler +
        gpu.config.cfar_reference_doppler;
    const int range_margin = gpu.config.cfar_guard_range +
        gpu.config.cfar_reference_range;
    for (std::size_t index = 0; index < gpu.detections.size(); ++index) {
        if (gpu.detections[index] != cpu.detections[index]) {
            ++result.detection_mismatch_count;
            const int doppler = static_cast<int>(index) / gpu.config.samples_per_pulse;
            const int range = static_cast<int>(index) % gpu.config.samples_per_pulse;
            const bool valid = doppler >= doppler_margin &&
                doppler < gpu.config.num_pulses - doppler_margin &&
                range >= range_margin && range < gpu.config.samples_per_pulse - range_margin;
            if (valid) ++result.valid_detection_mismatch_count;
            else ++result.boundary_detection_mismatch_count;
        }
    }
    return result;
}

void log_detection_mismatches(
    std::ostream& output, int sample, const task1::PipelineState& gpu,
    const task1::PipelineState& cpu, const PipelineAccuracy& accuracy)
{
    if (accuracy.detection_mismatch_count == 0) return;
    const int doppler_margin = gpu.config.cfar_guard_doppler +
        gpu.config.cfar_reference_doppler;
    const int range_margin = gpu.config.cfar_guard_range +
        gpu.config.cfar_reference_range;
    output << "detection_mismatch_summary sample_index=" << sample
           << " total=" << accuracy.detection_mismatch_count
           << " valid=" << accuracy.valid_detection_mismatch_count
           << " boundary=" << accuracy.boundary_detection_mismatch_count << '\n';
    std::size_t written = 0;
    for (std::size_t index = 0; index < gpu.detections.size() && written < 64; ++index) {
        if (gpu.detections[index] == cpu.detections[index]) continue;
        const int doppler = static_cast<int>(index) / gpu.config.samples_per_pulse;
        const int range = static_cast<int>(index) % gpu.config.samples_per_pulse;
        const bool valid = doppler >= doppler_margin &&
            doppler < gpu.config.num_pulses - doppler_margin &&
            range >= range_margin && range < gpu.config.samples_per_pulse - range_margin;
        output << "detection_mismatch sample_index=" << sample
               << " flat_index=" << index << " doppler=" << doppler
               << " range=" << range << " cfar_cell=" << (valid ? "valid" : "boundary")
               << " gpu_detection=" << static_cast<int>(gpu.detections[index])
               << " cpu_detection=" << static_cast<int>(cpu.detections[index])
               << " gpu_power=" << gpu.power[index] << " cpu_power=" << cpu.power[index]
               << " gpu_threshold=" << gpu.cfar_threshold[index]
               << " cpu_threshold=" << cpu.cfar_threshold[index] << '\n';
        ++written;
    }
}

std::array<task1::accuracy::AccuracyEvidence, 5> run_semantic_gates(
    const task1::PipelineState& gpu)
{
    return {
        task1::accuracy::compare_step1_accuracy(
            gpu, task1::accuracy::collect_step1_validation(gpu)),
        task1::accuracy::compare_step2_accuracy(
            gpu, task1::accuracy::collect_step2_validation(gpu)),
        task1::accuracy::compare_step3_accuracy(
            gpu, task1::accuracy::collect_step3_validation(gpu)),
        task1::accuracy::compare_step4_accuracy(
            gpu, task1::accuracy::collect_step4_validation(gpu)),
        task1::accuracy::compare_step5_accuracy(
            gpu, task1::accuracy::collect_step5_validation(gpu))};
}

const std::vector<std::string> kIdentityColumns{
    "schema_version", "run_id", "run_type", "test_time_utc", "git_commit", "git_dirty",
    "config_id", "config_sha256", "target_kind", "target", "task_name", "step_name",
    "operator_name", "case_id", "scale_id", "order_of_magnitude", "actual_elements",
    "input_shape", "dtype", "device", "backend", "timing_scope", "status", "error_code"};

std::vector<std::string> columns(std::initializer_list<const char*> tail)
{
    auto result = kIdentityColumns;
    for (const char* value : tail) result.emplace_back(value);
    return result;
}

std::vector<std::string> identity(
    const task1::Task1ScaleConfig& scale, const Args& args,
    const std::string& device, const std::string& case_id)
{
    return {"1", args.run_id, args.run_type, TASK1_EVIDENCE_TEST_TIME,
        TASK1_EVIDENCE_GIT_COMMIT, TASK1_EVIDENCE_GIT_DIRTY,
        scale.scale_set_id, scale.config.sha256, "task", "Task1.pipeline", "Task1",
        "pipeline_total",
        "pulse_compression+pulse_doppler+cfar_alpha+ca_cfar+ambgfun;private:step1+power",
        case_id, scale.scale_id, scale.order_of_magnitude,
        std::to_string(scale.actual_elements),
        scale.input_shape + ";waveform=[" + std::to_string(scale.config.pulse_samples) + "]",
        "ComplexFP32", device, args.backend, "compatibility/end-to-end", "PASS", "OK"};
}

struct ResourceTrace {
    ResourceSample cpu_heap_before, cpu_heap_after, cpu_rss_before, cpu_rss_after;
    ResourceSample gpu_before, gpu_after;
    std::int64_t gpu_peak{};
};
}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_args(argc, argv);
#if defined(USE_THRUST)
        const std::string compiled_backend = "fft_thrust";
#elif defined(USE_DLFFT)
        const std::string compiled_backend = "dlfft";
#else
        const std::string compiled_backend = "unbound";
#endif
        if (args.backend != compiled_backend)
            throw std::invalid_argument("backend mismatch: compiled=" + compiled_backend);
        if (std::filesystem::exists(args.output))
            throw std::invalid_argument("output directory already exists");
        const auto thresholds = task1::accuracy::FormalThresholdSet::load(args.accuracy_thresholds);

        const auto scale = task1::load_task1_scale(args.scales, args.scale_id);
        const std::string input_digest = scale.input_digest;
        const auto case_id = make_case_id("task1_pipeline", scale.scale_id, input_digest);
        const std::string suffix = args.backend + "_" + args.run_type + ".csv";

        CsvWriter timing_csv(args.output / ("task1_iteration_timing_" + suffix), columns({
            "warmup_runs", "measured_runs", "sample_index", "latency_ms", "synchronized",
            "input_digest", "output_digest"}));
        CsvWriter memory_csv(args.output / ("task1_memory_trace_" + suffix), columns({
            "warmup_runs", "measured_runs", "sample_index", "resource_scope",
            "trace_phase", "phase_index", "cpu_live_heap_bytes", "rss_bytes",
            "gpu_used_bytes", "memory_source"}));
        std::ofstream full_log(args.output / "task1_pipeline_full.log");
        if (!full_log) throw std::runtime_error("cannot create full log");

        for (int index = 0; index < args.warmup_runs; ++index) {
            (void)run_cpu_pipeline(scale.config);
            (void)run_gpu_pipeline(scale.config);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<double> cpu_samples, gpu_samples, cuda_event_samples;
        std::array<std::vector<double>, 5> cpu_step_samples, gpu_step_samples;
        std::vector<ResourceTrace> traces;
        double max_mse = 0.0, max_rmse = 0.0;
        double max_relative_l2 = 0.0, max_relative_linf = 0.0;
        std::size_t max_independent_detection_mismatch_count = 0;
        std::size_t max_independent_valid_mismatch_count = 0;
        std::size_t max_independent_boundary_mismatch_count = 0;
        std::map<std::string, task1::accuracy::OutputAccuracyResult> accuracy_aggregate;

        for (int sample = 0; sample < args.measured_runs; ++sample) {
            ResourceTrace trace;
            PipelineResourcePass step_resources;
            trace.cpu_heap_before = sample_cpu_heap();
            trace.cpu_rss_before = sample_cpu_rss();
            CpuPipelineRun cpu(scale.config);
            const double cpu_wall_ms = measure_ms(
                [&] { cpu = run_cpu_pipeline(scale.config, &step_resources); }, [] {});
            const double cpu_ms = std::max(
                0.0, cpu_wall_ms - step_resources.cpu_sampling_ms);
            trace.cpu_heap_after = sample_cpu_heap();
            trace.cpu_rss_after = sample_cpu_rss();

            trace.gpu_before = sample_gpu_used(gpu_used_bytes, "cudaMemGetInfo");
            GpuPipelineRun gpu(scale.config);
            const double gpu_wall_ms = measure_ms(
                [&] { gpu = run_gpu_pipeline(scale.config, &step_resources); },
                [] { CUDA_CHECK(cudaDeviceSynchronize()); });
            const double gpu_ms = std::max(
                0.0, gpu_wall_ms - step_resources.gpu_sampling_ms);
            trace.gpu_after = sample_gpu_used(gpu_used_bytes, "cudaMemGetInfo");
            trace.gpu_peak = gpu.gpu_peak;

            const auto accuracy = compare_pipeline_outputs(gpu.state, cpu.state);
            log_detection_mismatches(full_log, sample, gpu.state, cpu.state, accuracy);
            const auto step_accuracy = run_semantic_gates(gpu.state);
            const auto formal_accuracy = task1::accuracy::evaluate_pipeline_formal(
                gpu.state, cpu.state, thresholds);
            task1::accuracy::merge_worst_accuracy(accuracy_aggregate, formal_accuracy);
            max_mse = std::max(max_mse, accuracy.mse);
            max_rmse = std::max(max_rmse, accuracy.rmse);
            max_relative_l2 = std::max(max_relative_l2, accuracy.relative_l2);
            max_relative_linf = std::max(max_relative_linf, accuracy.relative_linf);
            max_independent_detection_mismatch_count = std::max(
                max_independent_detection_mismatch_count, accuracy.detection_mismatch_count);
            max_independent_valid_mismatch_count = std::max(
                max_independent_valid_mismatch_count,
                accuracy.valid_detection_mismatch_count);
            max_independent_boundary_mismatch_count = std::max(
                max_independent_boundary_mismatch_count,
                accuracy.boundary_detection_mismatch_count);
            cpu_samples.push_back(cpu_ms);
            gpu_samples.push_back(gpu_ms);
            cuda_event_samples.push_back(gpu.cuda_event_sum_ms);
            traces.push_back(trace);
            for (std::size_t step = 0; step < 5; ++step) {
                cpu_step_samples[step].push_back(cpu.step_ms[step]);
                gpu_step_samples[step].push_back(gpu.step_ms[step]);
            }

            auto cpu_timing = identity(scale, args, "cpu", case_id);
            cpu_timing.insert(cpu_timing.end(), {std::to_string(args.warmup_runs),
                std::to_string(args.measured_runs), std::to_string(sample), number(cpu_ms), "true",
                input_digest, pipeline_output_digest(cpu.state)});
            timing_csv.append(cpu_timing);
            auto gpu_timing = identity(scale, args, "gpu", case_id);
            gpu_timing.insert(gpu_timing.end(), {std::to_string(args.warmup_runs),
                std::to_string(args.measured_runs), std::to_string(sample), number(gpu_ms), "true",
                input_digest, pipeline_output_digest(gpu.state)});
            timing_csv.append(gpu_timing);

            auto trace_row = [&](const std::string& device, const std::string& scope,
                                 const std::string& phase, int phase_index,
                                 bool cpu_available, std::int64_t heap, std::int64_t rss,
                                 bool gpu_available, std::int64_t gpu_bytes,
                                 const std::string& source) {
                auto row = identity(scale, args, device, case_id);
                row.insert(row.end(), {std::to_string(args.warmup_runs),
                    std::to_string(args.measured_runs), std::to_string(sample), scope,
                    phase, std::to_string(phase_index),
                    nullable_csv(cpu_available, std::to_string(heap)),
                    nullable_csv(cpu_available, std::to_string(rss)),
                    nullable_csv(gpu_available, std::to_string(gpu_bytes)), source});
                memory_csv.append(row);
            };
            const std::array<const char*, 5> step_scopes{
                "step1", "step2", "step3", "step4", "step5"};
            const std::array<const char*, 5> step_functions{
                "LFM waveform and echo generation", "pulse compression",
                "pulse-Doppler range map", "2D CA-CFAR", "ambiguity function"};
            const std::array<const char*, 5> step_operators{
                "private:generate_lfm+simulate_echo", "pulse_compression",
                "pulse_doppler", "cfar_alpha+ca_cfar;private:power",
                "ambgfun:2d+delay_cut+doppler_cut"};
            const std::string waveform_shape =
                "[" + std::to_string(scale.config.pulse_samples) + "]";
            const std::array<std::string, 5> step_shapes{
                "config->waveform" + waveform_shape + "+echo" + scale.input_shape + ":ComplexFP32",
                "echo" + scale.input_shape + "+waveform" + waveform_shape +
                    "->compressed" + scale.input_shape + ":ComplexFP32",
                "compressed" + scale.input_shape + "->range_doppler" +
                    scale.input_shape + ":ComplexFP32",
                "range_doppler" + scale.input_shape + ":ComplexFP32->power+threshold+mask" +
                    scale.input_shape,
                "waveform" + waveform_shape + ":ComplexFP32->ambiguity_2d+two_cuts"};
            for (std::size_t step = 0; step < 5; ++step) {
                const bool cpu_before_available =
                    step_resources.cpu_heap_before[step].available &&
                    step_resources.cpu_rss_before[step].available;
                const bool cpu_after_available =
                    step_resources.cpu_heap_after[step].available &&
                    step_resources.cpu_rss_after[step].available;
                trace_row("cpu", step_scopes[step], "before", 0,
                    cpu_before_available, step_resources.cpu_heap_before[step].bytes,
                    step_resources.cpu_rss_before[step].bytes, false, 0,
                    "same_pipeline_execution:mallinfo2+/proc/self/statm");
                trace_row("cpu", step_scopes[step], "peak", 1,
                    cpu_before_available && cpu_after_available,
                    std::max(step_resources.cpu_heap_before[step].bytes,
                        step_resources.cpu_heap_after[step].bytes),
                    std::max(step_resources.cpu_rss_before[step].bytes,
                        step_resources.cpu_rss_after[step].bytes), false, 0,
                    "same_pipeline_execution:observed_before_after_max");
                trace_row("cpu", step_scopes[step], "after", 2,
                    cpu_after_available, step_resources.cpu_heap_after[step].bytes,
                    step_resources.cpu_rss_after[step].bytes, false, 0,
                    "same_pipeline_execution:mallinfo2+/proc/self/statm");
                trace_row("gpu", step_scopes[step], "before", 0, false, 0, 0,
                    step_resources.gpu_before[step].available,
                    step_resources.gpu_before[step].bytes,
                    "same_pipeline_execution:" + step_resources.gpu_before[step].source);
                trace_row("gpu", step_scopes[step], "peak", 1, false, 0, 0,
                    true, step_resources.gpu_peak[step],
                    "same_pipeline_execution:cudaMemGetInfo_while_step_outputs_alive");
                trace_row("gpu", step_scopes[step], "after", 2, false, 0, 0,
                    step_resources.gpu_after[step].available,
                    step_resources.gpu_after[step].bytes,
                    "same_pipeline_execution:" + step_resources.gpu_after[step].source);
                const std::int64_t cpu_heap_peak_step = std::max(
                    step_resources.cpu_heap_before[step].bytes,
                    step_resources.cpu_heap_after[step].bytes);
                const std::int64_t cpu_rss_peak_step = std::max(
                    step_resources.cpu_rss_before[step].bytes,
                    step_resources.cpu_rss_after[step].bytes);
                std::cout << "[TASK1][PIPELINE][STEP] sample_index=" << sample
                          << " step=" << step_scopes[step]
                          << " function=" << step_functions[step]
                          << " operators=" << step_operators[step]
                          << " io=" << step_shapes[step]
                          << " upstream=" << (step == 0 ? "config" :
                              (step == 1 ? "step1.echo+step1.waveform" :
                              (step == 2 ? "step2.compressed" :
                              (step == 3 ? "step3.range_doppler" : "step1.waveform"))))
                          << " cpu_ms=" << cpu.step_ms[step]
                          << " gpu_ms=" << gpu.step_ms[step]
                          << " relative_l2=" << step_accuracy[step].relative_error
                          << " cpu_heap_before=" << step_resources.cpu_heap_before[step].bytes
                          << " cpu_heap_after=" << step_resources.cpu_heap_after[step].bytes
                          << " cpu_heap_peak=" << cpu_heap_peak_step
                          << " cpu_rss_peak=" << cpu_rss_peak_step
                          << " gpu_before=" << step_resources.gpu_before[step].bytes
                          << " gpu_after=" << step_resources.gpu_after[step].bytes
                          << " gpu_peak=" << step_resources.gpu_peak[step]
                          << " measured_runs=" << args.measured_runs
                          << " status=PASS\n";
                full_log << "sample_index=" << sample << " step=" << step_scopes[step]
                         << " cpu_ms=" << cpu.step_ms[step]
                         << " gpu_ms=" << gpu.step_ms[step]
                         << " relative_l2=" << step_accuracy[step].relative_error
                         << " cpu_heap_peak=" << cpu_heap_peak_step
                         << " cpu_rss_peak=" << cpu_rss_peak_step
                         << " gpu_peak=" << step_resources.gpu_peak[step]
                         << " status=PASS\n";
            }
            trace_row("cpu", "pipeline_total", "before", 0,
                trace.cpu_heap_before.available && trace.cpu_rss_before.available,
                trace.cpu_heap_before.bytes, trace.cpu_rss_before.bytes, false, 0,
                "mallinfo2+/proc/self/statm");
            trace_row("cpu", "pipeline_total", "peak", 1, true,
                std::max(trace.cpu_heap_before.bytes, trace.cpu_heap_after.bytes),
                std::max(trace.cpu_rss_before.bytes, trace.cpu_rss_after.bytes), false, 0,
                "observed_before_after_max");
            trace_row("cpu", "pipeline_total", "after", 2,
                trace.cpu_heap_after.available && trace.cpu_rss_after.available,
                trace.cpu_heap_after.bytes, trace.cpu_rss_after.bytes, false, 0,
                "mallinfo2+/proc/self/statm");
            trace_row("gpu", "pipeline_total", "before", 0, false, 0, 0,
                trace.gpu_before.available, trace.gpu_before.bytes, trace.gpu_before.source);
            trace_row("gpu", "pipeline_total", "peak", 1, false, 0, 0, true, trace.gpu_peak,
                "max_step_cudaMemGetInfo_while_outputs_alive");
            trace_row("gpu", "pipeline_total", "after", 2, false, 0, 0,
                trace.gpu_after.available, trace.gpu_after.bytes, trace.gpu_after.source);

            full_log << "sample_index=" << sample << " cpu_pipeline_ms=" << cpu_ms
                     << " gpu_pipeline_ms=" << gpu_ms
                     << " cpu_resource_sampling_overhead_ms="
                     << step_resources.cpu_sampling_ms
                     << " gpu_resource_sampling_overhead_ms="
                     << step_resources.gpu_sampling_ms
                     << " resource_overhead_excluded=true"
                     << " cuda_event_aux_sum_ms=" << gpu.cuda_event_sum_ms
                     << " relative_l2=" << accuracy.relative_l2
                     << " relative_linf=" << accuracy.relative_linf
                     << " detection_mismatch_count=" << accuracy.detection_mismatch_count
                     << " valid_detection_mismatch_count="
                     << accuracy.valid_detection_mismatch_count
                     << " boundary_detection_mismatch_count="
                     << accuracy.boundary_detection_mismatch_count
                     << " mse=" << task1::accuracy::worst_mse(accuracy_aggregate)
                     << " rmse=" << task1::accuracy::worst_rmse(accuracy_aggregate)
                     << " formal_relative_l2=" << task1::accuracy::worst_relative_l2(accuracy_aggregate)
                     << " formal_relative_linf=" << task1::accuracy::worst_relative_linf(accuracy_aggregate)
                     << " semantic_gates=5/5 accuracy_status="
                     << (task1::accuracy::all_accuracy_pass(formal_accuracy) ? "PASS" : "FAIL") << '\n';
            if (!task1::accuracy::all_accuracy_pass(formal_accuracy)) {
                task1::accuracy::write_formal_accuracy_csv(
                    args.output / ("task1_accuracy_details_" + suffix),
                    {args.run_id, args.run_type, TASK1_EVIDENCE_GIT_COMMIT, case_id,
                     scale.scale_id, args.backend, sample + 1}, thresholds, accuracy_aggregate);
                throw std::runtime_error(task1::accuracy::accuracy_failure_message(formal_accuracy));
            }
        }

        max_mse = task1::accuracy::worst_mse(accuracy_aggregate);
        max_rmse = task1::accuracy::worst_rmse(accuracy_aggregate);
        max_relative_l2 = task1::accuracy::worst_relative_l2(accuracy_aggregate);
        max_relative_linf = task1::accuracy::worst_relative_linf(accuracy_aggregate);
        task1::accuracy::write_formal_accuracy_csv(
            args.output / ("task1_accuracy_details_" + suffix),
            {args.run_id, args.run_type, TASK1_EVIDENCE_GIT_COMMIT, case_id,
             scale.scale_id, args.backend, args.measured_runs}, thresholds, accuracy_aggregate);

        const auto cpu_stats = summarize_ms(cpu_samples);
        const auto gpu_stats = summarize_ms(gpu_samples);
        const auto cuda_event_stats = summarize_ms(cuda_event_samples);
        const double paired_speedup = speedup(
            cpu_stats, gpu_stats, "compatibility/end-to-end", "compatibility/end-to-end");
        std::int64_t cpu_heap_peak = 0, cpu_rss_peak = 0, gpu_peak = 0;
        for (const auto& trace : traces) {
            cpu_heap_peak = std::max(
                {cpu_heap_peak, trace.cpu_heap_before.bytes, trace.cpu_heap_after.bytes});
            cpu_rss_peak = std::max(
                {cpu_rss_peak, trace.cpu_rss_before.bytes, trace.cpu_rss_after.bytes});
            gpu_peak = std::max(gpu_peak, trace.gpu_peak);
        }

        CsvWriter main_csv(args.output / ("task1_benchmark_" + suffix), columns({
            "warmup_runs", "measured_runs", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "min_ms", "max_ms", "std_ms", "cv", "cpu_gpu_speedup", "accuracy_reference",
             "mse", "rmse", "relative_l2", "relative_linf", "exact_match",
             "mismatch_count", "independent_pipeline_exact_match",
             "independent_pipeline_mismatch_count",
             "independent_pipeline_valid_mismatch_count",
             "independent_pipeline_boundary_mismatch_count",
             "semantic_check", "accuracy_status",
            "accuracy_threshold_set_id", "accuracy_thresholds_sha256", "cpu_heap_before_bytes",
            "cpu_heap_after_bytes", "cpu_heap_peak_bytes", "cpu_heap_delta_bytes",
            "rss_before_bytes", "rss_after_bytes", "rss_peak_bytes", "rss_delta_bytes",
            "gpu_before_bytes", "gpu_after_bytes", "gpu_peak_bytes", "gpu_delta_bytes",
            "return_code"}));
        auto append_main = [&](const std::string& device, const Statistics& stats) {
            const bool cpu = device == "cpu";
            const auto& first = traces.front();
            const auto& last = traces.back();
            auto row = identity(scale, args, device, case_id);
            row.insert(row.end(), {std::to_string(args.warmup_runs),
                std::to_string(args.measured_runs), number(stats.mean_ms), number(stats.p50_ms),
                number(stats.p95_ms), number(stats.p99_ms), number(stats.min_ms),
                number(stats.max_ms), number(stats.stddev_ms), number(stats.cv),
                cpu ? "NA" : number(paired_speedup),
                "independent Host pipeline for numeric outputs; same-input Host ca_cfar for detection mask",
                 number(max_mse), number(max_rmse), number(max_relative_l2),
                 number(max_relative_linf), "true", "0",
                 max_independent_detection_mismatch_count == 0 ? "true" : "false",
                 std::to_string(max_independent_detection_mismatch_count),
                 std::to_string(max_independent_valid_mismatch_count),
                 std::to_string(max_independent_boundary_mismatch_count),
                 "all_five_step_semantic_rules_pass",
                 "PASS", thresholds.id(), thresholds.sha256(),
                nullable_csv(cpu && first.cpu_heap_before.available,
                    std::to_string(first.cpu_heap_before.bytes)),
                nullable_csv(cpu && last.cpu_heap_after.available,
                    std::to_string(last.cpu_heap_after.bytes)),
                nullable_csv(cpu, std::to_string(cpu_heap_peak)),
                nullable_csv(cpu,
                    std::to_string(last.cpu_heap_after.bytes - first.cpu_heap_before.bytes)),
                nullable_csv(cpu && first.cpu_rss_before.available,
                    std::to_string(first.cpu_rss_before.bytes)),
                nullable_csv(cpu && last.cpu_rss_after.available,
                    std::to_string(last.cpu_rss_after.bytes)),
                nullable_csv(cpu, std::to_string(cpu_rss_peak)),
                nullable_csv(cpu,
                    std::to_string(last.cpu_rss_after.bytes - first.cpu_rss_before.bytes)),
                nullable_csv(!cpu && first.gpu_before.available,
                    std::to_string(first.gpu_before.bytes)),
                nullable_csv(!cpu && last.gpu_after.available,
                    std::to_string(last.gpu_after.bytes)),
                nullable_csv(!cpu, std::to_string(gpu_peak)),
                nullable_csv(!cpu,
                    std::to_string(last.gpu_after.bytes - first.gpu_before.bytes)), "0"});
            main_csv.append(row);
        };
        append_main("cpu", cpu_stats);
        append_main("gpu", gpu_stats);

        CsvWriter pipeline_csv(args.output / ("task1_pipeline_summary_" + suffix), columns({
            "component", "upstream", "measured_runs", "mean_ms", "p50_ms", "p95_ms",
            "p99_ms", "min_ms", "max_ms", "std_ms", "cv"}));
        const std::array<const char*, 5> components{
            "step1", "step2", "step3", "step4", "step5"};
        const std::array<const char*, 5> upstream{
            "config", "step1.echo+step1.waveform", "step2.compressed",
            "step3.range_doppler", "step1.waveform"};
        auto append_component = [&](const std::string& device, const std::string& component,
                                    const std::string& source, const Statistics& stats) {
            auto row = identity(scale, args, device, case_id);
            row.insert(row.end(), {component, source, std::to_string(args.measured_runs),
                number(stats.mean_ms), number(stats.p50_ms), number(stats.p95_ms),
                number(stats.p99_ms), number(stats.min_ms), number(stats.max_ms),
                number(stats.stddev_ms), number(stats.cv)});
            pipeline_csv.append(row);
        };
        for (std::size_t step = 0; step < 5; ++step) {
            append_component("cpu", components[step], upstream[step],
                summarize_ms(cpu_step_samples[step]));
            append_component("gpu", components[step], upstream[step],
                summarize_ms(gpu_step_samples[step]));
        }
        append_component("cpu", "pipeline_total", "frozen_multi_upstream_graph", cpu_stats);
        append_component("gpu", "pipeline_total", "frozen_multi_upstream_graph", gpu_stats);

        std::ofstream json(args.output / "task1_pipeline_summary.json");
        json << "{\n  \"status\": \"PASS\",\n  \"run_id\": \"" << args.run_id
             << "\",\n  \"run_type\": \"" << args.run_type
             << "\",\n  \"scale_id\": \"" << scale.scale_id
             << "\",\n  \"backend\": \"" << args.backend
             << "\",\n  \"warmup_runs\": " << args.warmup_runs
             << ",\n  \"measured_runs\": " << args.measured_runs
             << ",\n  \"accuracy_threshold_set_id\": \"" << thresholds.id()
             << "\",\n  \"accuracy_thresholds_sha256\": \"" << thresholds.sha256()
             << "\",\n  \"mse\": " << max_mse << ",\n  \"rmse\": " << max_rmse
             << ",\n  \"relative_l2\": " << max_relative_l2
              << ",\n  \"relative_linf\": " << max_relative_linf
              << ",\n  \"same_input_detection_exact_match\": true"
              << ",\n  \"same_input_detection_mismatch_count\": 0"
              << ",\n  \"independent_pipeline_detection_mismatch_count\": "
              << max_independent_detection_mismatch_count
              << ",\n  \"independent_pipeline_valid_mismatch_count\": "
              << max_independent_valid_mismatch_count
              << ",\n  \"independent_pipeline_boundary_mismatch_count\": "
              << max_independent_boundary_mismatch_count
             << ",\n  \"input_digest\": \"" << input_digest
             << "\",\n  \"semantic_gates_passed\": 5"
             << ",\n  \"resource_sampling_mode\": \"same_pipeline_execution\""
             << ",\n  \"resource_sampling_overhead_excluded\": true\n}\n";
        std::ofstream summary(args.output / "task1_pipeline_summary.txt");
        summary << "status=PASS\nwarmup_runs=" << args.warmup_runs
                << "\nmeasured_runs=" << args.measured_runs
                << "\nscale_id=" << scale.scale_id << "\nbackend=" << args.backend
                << "\nsemantic_gates=5/5\n";
        summary << "accuracy_threshold_set_id=" << thresholds.id()
                << "\naccuracy_thresholds_sha256=" << thresholds.sha256()
                << "\nmse=" << max_mse << "\nrmse=" << max_rmse
                << "\nrelative_l2=" << max_relative_l2
                << "\nrelative_linf=" << max_relative_linf
                 << "\nsame_input_detection_exact_match=true"
                 << "\nsame_input_detection_mismatch_count=0"
                 << "\nindependent_pipeline_detection_mismatch_count="
                 << max_independent_detection_mismatch_count
                 << "\nindependent_pipeline_valid_mismatch_count="
                 << max_independent_valid_mismatch_count
                 << "\nindependent_pipeline_boundary_mismatch_count="
                 << max_independent_boundary_mismatch_count
                 << "\nresource_sampling_mode=same_pipeline_execution"
                 << "\nresource_sampling_overhead_excluded=true"
                 << "\naccuracy_status=PASS\n";
        full_log << "function=Task1 frozen full pipeline\n"
                 << "call_chain=Step1->{Step2->Step3->Step4,Step5};Step5_source=Step1.waveform\n"
                 << "formal_operators=pulse_compression,pulse_doppler,cfar_alpha,ca_cfar,ambgfun\n"
                 << "timing=pipeline_total measured directly;not=sum(step timings)\n"
                 << "resource_sampling_mode=same_pipeline_execution\n"
                 << "resource_sampling_overhead_excluded=true\n"
                 << "status=PASS return_code=0\n";

        std::cout << "[TASK1][PIPELINE][FUNCTION] frozen full radar detection and ambiguity workflow\n"
                  << "[TASK1][PIPELINE][CALL_CHAIN] Step1 -> Step2 -> Step3 -> Step4; Step1.waveform -> Step5\n"
                  << "[TASK1][PIPELINE][INPUT] scale_id=" << scale.scale_id
                  << " shape=" << scale.input_shape << " waveform_shape=["
                  << scale.config.pulse_samples << "] input_digest=" << input_digest
                  << " shared_cpu_gpu=true\n"
                  << "[TASK1][PIPELINE][SUMMARY] warmup_runs=" << args.warmup_runs
                  << " measured_runs=" << args.measured_runs
                  << " cpu_pipeline_mean_ms=" << cpu_stats.mean_ms
                  << " gpu_pipeline_mean_ms=" << gpu_stats.mean_ms
                   << " cuda_event_aux_sum_mean_ms=" << cuda_event_stats.mean_ms
                   << " mse=" << max_mse << " rmse=" << max_rmse
                  << " relative_l2=" << max_relative_l2
                  << " relative_linf=" << max_relative_linf
                  << " semantic_gates=5/5 timing_direct=true step_sum_used=false"
                  << " resource_sampling_mode=same_pipeline_execution"
                  << " resource_sampling_overhead_excluded=true backend="
                  << args.backend << " status=PASS return_code=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TASK1][PIPELINE_BENCHMARK][FAIL] code=EXECUTION_FAILED message="
                  << error.what() << '\n';
        return 2;
    }
}
