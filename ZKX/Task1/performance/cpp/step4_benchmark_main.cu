#include "accuracy/comparison/step4_comparison.h"
#include "accuracy/formal_accuracy.h"
#include "accuracy/validation/step4_validation.h"
#include <task1/steps/step1.h>
#include <task1/steps/step2.h>
#include <task1/steps/step3.h>
#include <task1/steps/step4.h>
#include "task1_scale_config.h"
#include <demo/config_file.h>
#include "test_all/support/metrics/csv_writer.h"
#include "test_all/support/metrics/identity.h"
#include "test_all/support/metrics/memory.h"
#include "test_all/support/metrics/statistics.h"

#include <algorithm>
#include <cstring>
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
    std::string scale_id;
    std::string backend;
    std::string run_id;
    std::string run_type;
    int warmup_runs{};
    int measured_runs{};
};

int strict_nonnegative(const std::string& value, const std::string& name)
{
    std::size_t used = 0;
    int parsed = 0;
    try { parsed = std::stoi(value, &used); }
    catch (...) { throw std::invalid_argument(name + " must be an integer"); }
    if (used != value.size() || parsed < 0) throw std::invalid_argument(name + " must be nonnegative");
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
        throw std::invalid_argument("required Step4 benchmark argument is missing");
    if (args.measured_runs < 1) throw std::invalid_argument("--measured-runs must be positive");
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

std::string cpu_output_digest(const task1::accuracy::Step4ValidationData& data)
{
    std::vector<unsigned char> bytes;
    append_bytes(bytes, data.power_cpu);
    append_bytes(bytes, data.threshold_cpu);
    append_bytes(bytes, data.detections_cpu);
    append_bytes(bytes, std::vector<double>{data.alpha_cpu});
    return demo_config::sha256(bytes);
}

std::string gpu_output_digest(const task1::PipelineState& state)
{
    std::vector<unsigned char> bytes;
    append_bytes(bytes, state.power);
    append_bytes(bytes, state.cfar_threshold);
    append_bytes(bytes, state.detections);
    append_bytes(bytes, std::vector<double>{state.cfar_alpha});
    return demo_config::sha256(bytes);
}

std::string step4_input_digest(const task1::PipelineState& state)
{
    std::vector<unsigned char> bytes;
    append_bytes(bytes, state.range_doppler);
    return demo_config::sha256(bytes);
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
        scale.scale_set_id, scale.config.sha256, "task", "Task1.step4", "Task1",
        "step4", "cfar_alpha+ca_cfar;private:range_doppler_power", case_id,
        scale.scale_id, scale.order_of_magnitude, std::to_string(scale.actual_elements),
        scale.input_shape, "ComplexFP32", device, args.backend, "compatibility/end-to-end",
        "PASS", "OK"};
}

struct ResourceTrace {
    ResourceSample cpu_heap_before, cpu_heap_after, cpu_rss_before, cpu_rss_after;
    ResourceSample gpu_before, gpu_after;
    std::int64_t gpu_peak{};
};

bool valid_cfar_cell(int doppler, int range, const task1::TaskConfig& config)
{
    const int doppler_margin =
        config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int range_margin =
        config.cfar_guard_range + config.cfar_reference_range;
    return doppler >= doppler_margin &&
        doppler < config.num_pulses - doppler_margin &&
        range >= range_margin &&
        range < config.samples_per_pulse - range_margin;
}

struct Step4DetectionDiagnostics {
    std::size_t power_bitwise_mismatch_count{};
    std::size_t threshold_bitwise_mismatch_count{};
    std::size_t independent_detection_mismatch_count{};
    std::size_t independent_valid_mismatch_count{};
    std::size_t independent_boundary_mismatch_count{};
    std::size_t same_power_detection_mismatch_count{};
    std::size_t same_power_valid_mismatch_count{};
    std::size_t same_power_boundary_mismatch_count{};
    std::size_t gpu_predicate_mismatch_count{};
    std::int64_t first_independent_mismatch_index{-1};
    std::int64_t first_same_power_mismatch_index{-1};
};

Step4DetectionDiagnostics diagnose_step4_detections(
    const task1::PipelineState& state,
    const task1::accuracy::Step4ValidationData& validation)
{
    const auto& config = state.config;
    const std::size_t expected_count =
        static_cast<std::size_t>(config.num_pulses) * config.samples_per_pulse;
    task1::require(state.power.size() == expected_count &&
            state.cfar_threshold.size() == expected_count &&
            state.detections.size() == expected_count &&
            validation.power_cpu.size() == expected_count &&
            validation.threshold_cpu.size() == expected_count &&
            validation.detections_cpu.size() == expected_count &&
            validation.detections_same_power_cpu.size() == expected_count,
        "Step4 detection diagnostics shape mismatch");

    Step4DetectionDiagnostics diagnostics;
    for (std::size_t index = 0; index < expected_count; ++index) {
        if (std::memcmp(&state.power[index], &validation.power_cpu[index],
                sizeof(float)) != 0)
            ++diagnostics.power_bitwise_mismatch_count;
        if (std::memcmp(&state.cfar_threshold[index],
                &validation.threshold_cpu[index], sizeof(float)) != 0)
            ++diagnostics.threshold_bitwise_mismatch_count;

        const int doppler = static_cast<int>(index) / config.samples_per_pulse;
        const int range = static_cast<int>(index) % config.samples_per_pulse;
        const bool valid = valid_cfar_cell(doppler, range, config);
        if (state.detections[index] != validation.detections_cpu[index]) {
            if (diagnostics.first_independent_mismatch_index < 0)
                diagnostics.first_independent_mismatch_index =
                    static_cast<std::int64_t>(index);
            ++diagnostics.independent_detection_mismatch_count;
            if (valid) ++diagnostics.independent_valid_mismatch_count;
            else ++diagnostics.independent_boundary_mismatch_count;
        }
        if (state.detections[index] !=
                validation.detections_same_power_cpu[index]) {
            if (diagnostics.first_same_power_mismatch_index < 0)
                diagnostics.first_same_power_mismatch_index =
                    static_cast<std::int64_t>(index);
            ++diagnostics.same_power_detection_mismatch_count;
            if (valid) ++diagnostics.same_power_valid_mismatch_count;
            else ++diagnostics.same_power_boundary_mismatch_count;
        }
        const auto predicate = state.power[index] > state.cfar_threshold[index]
            ? cusignal::CfarDetection::yes : cusignal::CfarDetection::no;
        if (state.detections[index] != predicate)
            ++diagnostics.gpu_predicate_mismatch_count;
    }
    return diagnostics;
}

void write_step4_failure_diagnostics(
    const std::filesystem::path& output, int sample,
    const task1::TaskConfig& config,
    const Step4DetectionDiagnostics& diagnostics)
{
    const auto first_row = [&](std::int64_t index) {
        return index < 0 ? -1 :
            static_cast<int>(index) / config.samples_per_pulse;
    };
    const auto first_column = [&](std::int64_t index) {
        return index < 0 ? -1 :
            static_cast<int>(index) % config.samples_per_pulse;
    };
    std::ofstream text(output / "task1_step4_failure_diagnostics.txt");
    std::ofstream json(output / "task1_step4_failure_diagnostics.json");
    if (!text || !json)
        throw std::runtime_error("cannot create Step4 failure diagnostics");

    text << "status=FAIL\nsample_index=" << sample
         << "\npower_bitwise_mismatch_count="
         << diagnostics.power_bitwise_mismatch_count
         << "\nthreshold_bitwise_mismatch_count="
         << diagnostics.threshold_bitwise_mismatch_count
         << "\nindependent_detection_mismatch_count="
         << diagnostics.independent_detection_mismatch_count
         << "\nindependent_valid_mismatch_count="
         << diagnostics.independent_valid_mismatch_count
         << "\nindependent_boundary_mismatch_count="
         << diagnostics.independent_boundary_mismatch_count
         << "\nsame_power_detection_mismatch_count="
         << diagnostics.same_power_detection_mismatch_count
         << "\nsame_power_valid_mismatch_count="
         << diagnostics.same_power_valid_mismatch_count
         << "\nsame_power_boundary_mismatch_count="
         << diagnostics.same_power_boundary_mismatch_count
         << "\ngpu_predicate_mismatch_count="
         << diagnostics.gpu_predicate_mismatch_count
         << "\nfirst_independent_mismatch_index="
         << diagnostics.first_independent_mismatch_index
         << "\nfirst_independent_mismatch_doppler="
         << first_row(diagnostics.first_independent_mismatch_index)
         << "\nfirst_independent_mismatch_range="
         << first_column(diagnostics.first_independent_mismatch_index)
         << "\nfirst_same_power_mismatch_index="
         << diagnostics.first_same_power_mismatch_index
         << "\nfirst_same_power_mismatch_doppler="
         << first_row(diagnostics.first_same_power_mismatch_index)
         << "\nfirst_same_power_mismatch_range="
         << first_column(diagnostics.first_same_power_mismatch_index) << '\n';

    json << "{\n  \"status\": \"FAIL\",\n  \"sample_index\": " << sample
         << ",\n  \"power_bitwise_mismatch_count\": "
         << diagnostics.power_bitwise_mismatch_count
         << ",\n  \"threshold_bitwise_mismatch_count\": "
         << diagnostics.threshold_bitwise_mismatch_count
         << ",\n  \"independent_detection_mismatch_count\": "
         << diagnostics.independent_detection_mismatch_count
         << ",\n  \"independent_valid_mismatch_count\": "
         << diagnostics.independent_valid_mismatch_count
         << ",\n  \"independent_boundary_mismatch_count\": "
         << diagnostics.independent_boundary_mismatch_count
         << ",\n  \"same_power_detection_mismatch_count\": "
         << diagnostics.same_power_detection_mismatch_count
         << ",\n  \"same_power_valid_mismatch_count\": "
         << diagnostics.same_power_valid_mismatch_count
         << ",\n  \"same_power_boundary_mismatch_count\": "
         << diagnostics.same_power_boundary_mismatch_count
         << ",\n  \"gpu_predicate_mismatch_count\": "
         << diagnostics.gpu_predicate_mismatch_count
         << ",\n  \"first_independent_mismatch_index\": "
         << diagnostics.first_independent_mismatch_index
         << ",\n  \"first_independent_mismatch_doppler\": "
         << first_row(diagnostics.first_independent_mismatch_index)
         << ",\n  \"first_independent_mismatch_range\": "
         << first_column(diagnostics.first_independent_mismatch_index)
         << ",\n  \"first_same_power_mismatch_index\": "
         << diagnostics.first_same_power_mismatch_index
         << ",\n  \"first_same_power_mismatch_doppler\": "
         << first_row(diagnostics.first_same_power_mismatch_index)
         << ",\n  \"first_same_power_mismatch_range\": "
         << first_column(diagnostics.first_same_power_mismatch_index) << "\n}\n";
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        using namespace zkx::common;
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
        task1::PipelineState identity_state(scale.config);
        (void)task1::run_step1(identity_state);
        (void)task1::run_step2(identity_state);
        (void)task1::run_step3(identity_state);
        const std::string input_digest = step4_input_digest(identity_state);
        const auto case_id = make_case_id("task1_step4", scale.scale_id, input_digest);
        const std::string suffix = args.backend + "_" + args.run_type + ".csv";

        CsvWriter timing_csv(args.output / ("task1_iteration_timing_" + suffix), columns({
            "warmup_runs", "measured_runs", "sample_index", "latency_ms", "synchronized",
            "input_digest", "output_digest"}));
        CsvWriter memory_csv(args.output / ("task1_memory_trace_" + suffix), columns({
            "warmup_runs", "measured_runs", "sample_index", "trace_phase", "phase_index",
            "cpu_live_heap_bytes", "rss_bytes", "gpu_used_bytes", "memory_source"}));
        std::ofstream full_log(args.output / "task1_step4_full.log");
        if (!full_log) throw std::runtime_error("cannot create full log");

        for (int index = 0; index < args.warmup_runs; ++index) {
            task1::PipelineState state(scale.config);
            (void)task1::run_step1(state);
            (void)task1::run_step2(state);
            (void)task1::run_step3(state);
            task1::require(step4_input_digest(state) == input_digest, "Step3 upstream input digest changed");
            (void)task1::accuracy::generate_step4_cpu_reference(state);
            (void)task1::run_step4(state);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<double> cpu_samples, gpu_samples, cuda_event_samples;
        std::vector<ResourceTrace> traces;
        cpu_samples.reserve(args.measured_runs);
        gpu_samples.reserve(args.measured_runs);
        traces.reserve(args.measured_runs);
        std::map<std::string, task1::accuracy::OutputAccuracyResult> accuracy_aggregate;

        for (int sample = 0; sample < args.measured_runs; ++sample) {
            task1::PipelineState state(scale.config);
            (void)task1::run_step1(state);
            (void)task1::run_step2(state);
            (void)task1::run_step3(state);
            task1::require(step4_input_digest(state) == input_digest, "Step3 upstream input digest changed");
            ResourceTrace trace;
            trace.cpu_heap_before = sample_cpu_heap();
            trace.cpu_rss_before = sample_cpu_rss();
            task1::accuracy::Step4ValidationData cpu;
            const double cpu_ms = measure_ms(
                [&] { cpu = task1::accuracy::generate_step4_cpu_reference(state); }, [] {});
            trace.cpu_heap_after = sample_cpu_heap();
            trace.cpu_rss_after = sample_cpu_rss();

            trace.gpu_before = sample_gpu_used(gpu_used_bytes, "cudaMemGetInfo");
            task1::StepEvidence gpu_evidence;
            const double gpu_ms = measure_ms(
                [&] { gpu_evidence = task1::run_step4(state); },
                [] { CUDA_CHECK(cudaDeviceSynchronize()); });
            trace.gpu_after = sample_gpu_used(gpu_used_bytes, "cudaMemGetInfo");
            trace.gpu_peak = static_cast<std::int64_t>(gpu_evidence.metrics.at("gpu_used_peak_bytes"));

            const auto validation = task1::accuracy::collect_step4_validation(state);
            const auto accuracy = task1::accuracy::evaluate_step4_formal(state, validation, thresholds);
            const auto diagnostics = diagnose_step4_detections(state, validation);
            task1::accuracy::merge_worst_accuracy(accuracy_aggregate, accuracy);
            cpu_samples.push_back(cpu_ms);
            gpu_samples.push_back(gpu_ms);
            cuda_event_samples.push_back(gpu_evidence.compute_ms);
            traces.push_back(trace);

            auto cpu_timing = identity(scale, args, "cpu", case_id);
            cpu_timing.insert(cpu_timing.end(), {std::to_string(args.warmup_runs),
                std::to_string(args.measured_runs), std::to_string(sample), number(cpu_ms), "true",
                input_digest, cpu_output_digest(cpu)});
            timing_csv.append(cpu_timing);
            auto gpu_timing = identity(scale, args, "gpu", case_id);
            gpu_timing.insert(gpu_timing.end(), {std::to_string(args.warmup_runs),
                std::to_string(args.measured_runs), std::to_string(sample), number(gpu_ms), "true",
                input_digest, gpu_output_digest(state)});
            timing_csv.append(gpu_timing);

            auto trace_row = [&](const std::string& device, const std::string& phase, int phase_index,
                                 bool cpu_available, std::int64_t heap, std::int64_t rss,
                                 bool gpu_available, std::int64_t gpu, const std::string& source) {
                auto row = identity(scale, args, device, case_id);
                row.insert(row.end(), {std::to_string(args.warmup_runs),
                    std::to_string(args.measured_runs), std::to_string(sample), phase,
                    std::to_string(phase_index), nullable_csv(cpu_available, std::to_string(heap)),
                    nullable_csv(cpu_available, std::to_string(rss)),
                    nullable_csv(gpu_available, std::to_string(gpu)), source});
                memory_csv.append(row);
            };
            trace_row("cpu", "before", 0,
                trace.cpu_heap_before.available && trace.cpu_rss_before.available,
                trace.cpu_heap_before.bytes, trace.cpu_rss_before.bytes, false, 0,
                "mallinfo2+/proc/self/statm");
            trace_row("cpu", "peak", 1, true,
                std::max(trace.cpu_heap_before.bytes, trace.cpu_heap_after.bytes),
                std::max(trace.cpu_rss_before.bytes, trace.cpu_rss_after.bytes), false, 0,
                "observed_before_after_max");
            trace_row("cpu", "after", 2,
                trace.cpu_heap_after.available && trace.cpu_rss_after.available,
                trace.cpu_heap_after.bytes, trace.cpu_rss_after.bytes, false, 0,
                "mallinfo2+/proc/self/statm");
            trace_row("gpu", "before", 0, false, 0, 0,
                trace.gpu_before.available, trace.gpu_before.bytes, trace.gpu_before.source);
            trace_row("gpu", "peak", 1, false, 0, 0, true, trace.gpu_peak,
                "cudaMemGetInfo_after_device_allocations");
            trace_row("gpu", "after", 2, false, 0, 0,
                trace.gpu_after.available, trace.gpu_after.bytes, trace.gpu_after.source);

            const auto detection = std::find_if(accuracy.begin(), accuracy.end(),
                [](const auto& result) {
                    return result.threshold.output_name == "detections";
                });
            task1::require(detection != accuracy.end(),
                "Step4 detection accuracy result is missing");
            full_log << "sample_index=" << sample << " cpu_ms=" << cpu_ms
                     << " gpu_ms=" << gpu_ms << " h2d_ms=" << gpu_evidence.h2d_ms
                     << " d2h_ms=" << gpu_evidence.d2h_ms
                     << " resident_input_reused="
                     << gpu_evidence.metrics.at("resident_input_reused")
                     << " cuda_event_aux_ms=" << gpu_evidence.compute_ms
                     << " mse=" << task1::accuracy::worst_mse(accuracy_aggregate)
                     << " rmse=" << task1::accuracy::worst_rmse(accuracy_aggregate)
                     << " relative_l2=" << task1::accuracy::worst_relative_l2(accuracy_aggregate)
                     << " relative_linf=" << task1::accuracy::worst_relative_linf(accuracy_aggregate)
                     << " exact_match=" << (detection->exact_match ? "true" : "false")
                     << " mismatch_count=" << detection->mismatch_count
                     << " independent_valid_mismatch_count="
                     << diagnostics.independent_valid_mismatch_count
                     << " independent_boundary_mismatch_count="
                     << diagnostics.independent_boundary_mismatch_count
                     << " same_power_mismatch_count="
                     << diagnostics.same_power_detection_mismatch_count
                     << " gpu_predicate_mismatch_count="
                     << diagnostics.gpu_predicate_mismatch_count << " accuracy_status="
                     << (task1::accuracy::all_accuracy_pass(accuracy) ? "PASS" : "FAIL") << '\n';
            if (!task1::accuracy::all_accuracy_pass(accuracy)) {
                write_step4_failure_diagnostics(
                    args.output, sample, state.config, diagnostics);
                task1::accuracy::write_formal_accuracy_csv(
                    args.output / ("task1_accuracy_details_" + suffix),
                    {args.run_id, args.run_type, TASK1_EVIDENCE_GIT_COMMIT, case_id,
                     scale.scale_id, args.backend, sample + 1}, thresholds, accuracy_aggregate);
                throw std::runtime_error(task1::accuracy::accuracy_failure_message(accuracy));
            }
        }

        const double max_mse = task1::accuracy::worst_mse(accuracy_aggregate);
        const double max_rmse = task1::accuracy::worst_rmse(accuracy_aggregate);
        const double max_relative_l2 = task1::accuracy::worst_relative_l2(accuracy_aggregate);
        const double max_relative_linf = task1::accuracy::worst_relative_linf(accuracy_aggregate);
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
            cpu_heap_peak = std::max({cpu_heap_peak, trace.cpu_heap_before.bytes, trace.cpu_heap_after.bytes});
            cpu_rss_peak = std::max({cpu_rss_peak, trace.cpu_rss_before.bytes, trace.cpu_rss_after.bytes});
            gpu_peak = std::max(gpu_peak, trace.gpu_peak);
        }

        CsvWriter main_csv(args.output / ("task1_benchmark_" + suffix), columns({
            "warmup_runs", "measured_runs", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "min_ms", "max_ms", "std_ms", "cv", "cpu_gpu_speedup", "accuracy_reference",
            "mse", "rmse", "relative_l2", "relative_linf", "exact_match",
            "mismatch_count", "semantic_check", "accuracy_status",
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
            row.insert(row.end(), {std::to_string(args.warmup_runs), std::to_string(args.measured_runs),
                number(stats.mean_ms), number(stats.p50_ms), number(stats.p95_ms), number(stats.p99_ms),
                number(stats.min_ms), number(stats.max_ms), number(stats.stddev_ms), number(stats.cv),
                cpu ? "NA" : number(paired_speedup), "Host power+cfar_alpha/ca_cfar_typed_cpu<float>", number(max_mse),
                number(max_rmse), number(max_relative_l2), number(max_relative_linf),
                "true", "0", "target_noise_interference_pfa_and_boundary_rules_pass", "PASS",
                thresholds.id(), thresholds.sha256(),
                nullable_csv(cpu && first.cpu_heap_before.available, std::to_string(first.cpu_heap_before.bytes)),
                nullable_csv(cpu && last.cpu_heap_after.available, std::to_string(last.cpu_heap_after.bytes)),
                nullable_csv(cpu, std::to_string(cpu_heap_peak)),
                nullable_csv(cpu, std::to_string(last.cpu_heap_after.bytes - first.cpu_heap_before.bytes)),
                nullable_csv(cpu && first.cpu_rss_before.available, std::to_string(first.cpu_rss_before.bytes)),
                nullable_csv(cpu && last.cpu_rss_after.available, std::to_string(last.cpu_rss_after.bytes)),
                nullable_csv(cpu, std::to_string(cpu_rss_peak)),
                nullable_csv(cpu, std::to_string(last.cpu_rss_after.bytes - first.cpu_rss_before.bytes)),
                nullable_csv(!cpu && first.gpu_before.available, std::to_string(first.gpu_before.bytes)),
                nullable_csv(!cpu && last.gpu_after.available, std::to_string(last.gpu_after.bytes)),
                nullable_csv(!cpu, std::to_string(gpu_peak)),
                nullable_csv(!cpu, std::to_string(last.gpu_after.bytes - first.gpu_before.bytes)), "0"});
            main_csv.append(row);
        };
        append_main("cpu", cpu_stats);
        append_main("gpu", gpu_stats);

        std::ofstream json(args.output / "task1_step4_summary.json");
        json << "{\n  \"status\": \"PASS\",\n  \"run_id\": \"" << args.run_id
             << "\",\n  \"run_type\": \"" << args.run_type << "\",\n  \"scale_id\": \""
             << scale.scale_id << "\",\n  \"backend\": \"" << args.backend
             << "\",\n  \"warmup_runs\": " << args.warmup_runs
             << ",\n  \"measured_runs\": " << args.measured_runs
             << ",\n  \"accuracy_threshold_set_id\": \"" << thresholds.id()
             << "\",\n  \"accuracy_thresholds_sha256\": \"" << thresholds.sha256()
             << "\",\n  \"mse\": " << max_mse << ",\n  \"rmse\": " << max_rmse
             << ",\n  \"relative_l2\": " << max_relative_l2
             << ",\n  \"relative_linf\": " << max_relative_linf
             << ",\n  \"exact_match\": true,\n  \"mismatch_count\": 0"
             << ",\n  \"input_digest\": \"" << input_digest << "\"\n}\n";
        std::ofstream summary(args.output / "task1_step4_summary.txt");
        summary << "status=PASS\nwarmup_runs=" << args.warmup_runs
                << "\nmeasured_runs=" << args.measured_runs << "\nscale_id=" << scale.scale_id
                << "\nbackend=" << args.backend << "\n";
        summary << "accuracy_threshold_set_id=" << thresholds.id()
                << "\naccuracy_thresholds_sha256=" << thresholds.sha256()
                << "\nmse=" << max_mse << "\nrmse=" << max_rmse
                << "\nrelative_l2=" << max_relative_l2
                << "\nrelative_linf=" << max_relative_linf
                << "\nexact_match=true\nmismatch_count=0\naccuracy_status=PASS\n";
        full_log << "function=range-Doppler power and two-dimensional CA-CFAR detection\n"
                 << "input_source=Step3 range_doppler;config_scale=" << scale.scale_id << '\n'
                 << "operators=cfar_alpha_device<float>,ca_cfar_device<float>;private=power_kernel\n"
                 << "shape=" << scale.input_shape << " dtype=ComplexFP32\nstatus=PASS return_code=0\n";

        std::cout << "[TASK1][STEP4][FUNCTION] range-Doppler power and two-dimensional CA-CFAR detection\n"
                  << "[TASK1][STEP4][CALL_CHAIN] Step3.range_doppler -> power_kernel + cfar_alpha_device<float> + ca_cfar_device<float>\n"
                  << "[TASK1][STEP4][INPUT] source=Step3.range_doppler scale_id="
                  << scale.scale_id << " shape=" << scale.input_shape
                  << " dtype=ComplexFP32 input_digest=" << input_digest
                  << " shared_cpu_gpu=true\n"
                  << "[TASK1][STEP4][SUMMARY] warmup_runs=" << args.warmup_runs
                  << " measured_runs=" << args.measured_runs << " cpu_mean_ms=" << cpu_stats.mean_ms
                  << " gpu_mean_ms=" << gpu_stats.mean_ms
                  << " cuda_event_aux_mean_ms=" << cuda_event_stats.mean_ms
                  << " mse=" << max_mse << " rmse=" << max_rmse
                  << " relative_l2=" << max_relative_l2
                  << " relative_linf=" << max_relative_linf << " backend=" << args.backend
                  << " status=PASS return_code=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TASK1][STEP4_BENCHMARK][FAIL] code=EXECUTION_FAILED message="
                  << error.what() << '\n';
        return 2;
    }
}
