#include "accuracy/formal_accuracy.h"
#include "accuracy/validation/step1_validation.h"
#include "accuracy/validation/step2_validation.h"
#include "accuracy/validation/step3_validation.h"
#include "accuracy/validation/step4_validation.h"
#include "accuracy/validation/step5_validation.h"
#include "accuracy/validation/step6_validation.h"
#include <task2/steps/step1.h>
#include <task2/steps/step2.h>
#include <task2/steps/step3.h>
#include <task2/steps/step4.h>
#include <task2/steps/step5.h>
#include <task2/steps/step6.h>
#include "task2_scale_config.h"
#include <demo/config_file.h>
#include "test_all/support/metrics/csv_writer.h"
#include "test_all/support/metrics/identity.h"
#include "test_all/support/metrics/memory.h"
#include "test_all/support/metrics/statistics.h"
#include <cusignal/operators/convolution/convolution_typed.h>
#include <cusignal/operators/estimation/estimation_typed.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <sstream>
#include <thread>

#ifndef TASK2_EVIDENCE_GIT_COMMIT
#define TASK2_EVIDENCE_GIT_COMMIT "unknown"
#endif
#ifndef TASK2_EVIDENCE_GIT_DIRTY
#define TASK2_EVIDENCE_GIT_DIRTY "unknown"
#endif
#ifndef TASK2_EVIDENCE_TEST_TIME
#define TASK2_EVIDENCE_TEST_TIME "unknown"
#endif

namespace {
using namespace zkx::common;

struct Args {
    std::filesystem::path scales, thresholds, output;
    std::string scale_id, backend, run_id, run_type;
    int warmup{}, measured{};
};

int count_arg(const std::string& value, const std::string& name)
{
    std::size_t used{};
    int result{};
    try { result = std::stoi(value, &used); }
    catch (...) { throw std::invalid_argument(name + " must be an integer"); }
    if (used != value.size() || result < 0)
        throw std::invalid_argument(name + " must be nonnegative");
    return result;
}

Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("incomplete argument: " + key);
        const std::string value = argv[++i];
        if (key == "--task-scales") a.scales = value;
        else if (key == "--accuracy-thresholds") a.thresholds = value;
        else if (key == "--scale-id") a.scale_id = value;
        else if (key == "--backend") a.backend = value;
        else if (key == "--run-id") a.run_id = value;
        else if (key == "--run-type") a.run_type = value;
        else if (key == "--warmup-runs") a.warmup = count_arg(value, key);
        else if (key == "--measured-runs") a.measured = count_arg(value, key);
        else if (key == "--output-dir") a.output = value;
        else throw std::invalid_argument("unknown argument: " + key);
    }
    if (a.scales.empty() || a.thresholds.empty() || a.output.empty() ||
        a.scale_id.empty() || a.backend.empty() || a.run_id.empty() || a.run_type.empty())
        throw std::invalid_argument("required Task2 pipeline benchmark argument is missing");
    if (a.measured < 1) throw std::invalid_argument("--measured-runs must be positive");
    if (a.run_type != "smoke" && a.run_type != "formal")
        throw std::invalid_argument("--run-type must be smoke or formal");
    if (a.run_type == "formal" && (a.warmup != 20 || a.measured != 100))
        throw std::invalid_argument("formal mode requires warmup_runs=20 and measured_runs=100");
    return a;
}

std::string number(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::int64_t gpu_used_bytes()
{
    std::size_t free_bytes{}, total_bytes{};
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return static_cast<std::int64_t>(total_bytes - free_bytes);
}

template <class Vector>
void append_bytes(std::vector<unsigned char>& bytes, const Vector& values)
{
    const auto* first = reinterpret_cast<const unsigned char*>(values.data());
    bytes.insert(bytes.end(), first, first + values.size() * sizeof(values[0]));
}

template <class Value>
void append_value(std::vector<unsigned char>& bytes, const Value& value)
{
    const auto* first = reinterpret_cast<const unsigned char*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(value));
}

std::string output_digest(const task2::PipelineState& s)
{
    std::vector<unsigned char> bytes;
    append_bytes(bytes, s.target_waveform); append_bytes(bytes, s.echo);
    append_bytes(bytes, s.filtered); append_bytes(bytes, s.smoothed);
    append_bytes(bytes, s.reference_for_correlation); append_bytes(bytes, s.fm_feature);
    append_bytes(bytes, s.correlation_feature); append_bytes(bytes, s.spectral_feature);
    append_bytes(bytes, s.wavelet_feature); append_bytes(bytes, s.feature_bundle);
    append_bytes(bytes, s.extrema); append_bytes(bytes, s.kalman_observations);
    append_value(bytes, s.kalman_anchor); append_value(bytes, s.estimate);
    append_bytes(bytes, s.kalman_covariance);
    return demo_config::sha256(bytes);
}

struct Resources {
    std::array<ResourceSample, 6> heap_before, heap_after, rss_before, rss_after;
    std::array<ResourceSample, 6> gpu_before, gpu_after;
    std::array<std::int64_t, 6> gpu_peak{};
    double cpu_sampling_ms{};
    double gpu_sampling_ms{};
};

struct CpuRun {
    explicit CpuRun(const task2::TaskConfig& c) : state(c) {}
    task2::PipelineState state;
    std::array<double, 6> step_ms{};
    task2::accuracy::Step1ValidationData s1;
    task2::accuracy::Step2ValidationData s2;
    task2::accuracy::Step3ValidationData s3;
    task2::accuracy::Step4ValidationData s4;
    task2::accuracy::Step5ValidationData s5;
    task2::accuracy::Step6ValidationData s6;
};

task2::accuracy::Step6ValidationData run_cpu_step6(task2::PipelineState& state)
{
    task2::prepare_step6_host_inputs(state);
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F}, initial_p{1.0F};
    const std::vector<float> f{state.config.kalman_f}, q{state.config.kalman_q};
    const std::vector<float> alpha{1.0F}, h{state.config.kalman_h}, r{state.config.kalman_r};
    auto kalman = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
    for (float observation : state.kalman_observations) {
        cusignal::kalman_predict_typed_cpu(kalman, f, q, alpha);
        cusignal::kalman_update_typed_cpu(
            kalman, h, r, std::vector<float>{observation});
    }
    task2::accuracy::Step6ValidationData result;
    result.state_cpu = kalman.x;
    result.covariance_cpu = kalman.p;
    return result;
}

std::vector<float> correlate_same_direct_parallel_exact(
    const std::vector<float>& in1, const std::vector<float>& in2)
{
    if (in1.empty() || in2.empty()) return {};
    if (in1.size() != in2.size())
        throw std::invalid_argument(
            "Task2 pipeline same correlation requires equal input sizes");
    if (in1.size() > std::numeric_limits<std::size_t>::max() - in2.size() + 1)
        throw std::invalid_argument("Task2 pipeline correlation size overflow");

    const std::size_t full_size = in1.size() + in2.size() - 1;
    const std::size_t output_size = in1.size();
    const std::size_t start = (full_size - output_size) / 2;
    std::vector<float> output(output_size);

    const unsigned detected = std::thread::hardware_concurrency();
    const std::size_t worker_count = std::min<std::size_t>(
        output_size, std::max<unsigned>(1U, detected));
    const std::size_t block_size =
        (output_size + worker_count - 1) / worker_count;

    auto compute = [&](std::size_t first, std::size_t last) {
        for (std::size_t output_index = first; output_index < last; ++output_index) {
            const std::size_t full_index = start + output_index;
            float accumulator = 0.0F;
            for (std::size_t j = 0; j < in2.size(); ++j) {
                const auto shifted = static_cast<std::ptrdiff_t>(full_index) -
                    static_cast<std::ptrdiff_t>(in2.size() - 1 - j);
                if (shifted >= 0 &&
                    shifted < static_cast<std::ptrdiff_t>(in1.size())) {
                    accumulator = std::fma(
                        in1[static_cast<std::size_t>(shifted)], in2[j], accumulator);
                }
            }
            output[output_index] = accumulator;
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
    for (std::size_t worker = 1; worker < worker_count; ++worker) {
        const std::size_t first = worker * block_size;
        if (first >= output_size) break;
        workers.emplace_back(compute, first, std::min(output_size, first + block_size));
    }
    compute(0, std::min(output_size, block_size));
    for (auto& worker : workers) worker.join();
    return output;
}

void verify_parallel_correlation_contract()
{
    const std::vector<float> left{
        0.25F, -1.0F, 2.5F, 0.0F, -0.75F, 3.0F, 1.25F, -2.0F};
    const std::vector<float> right{
        -0.5F, 1.5F, 0.25F, -2.25F, 0.75F, 1.0F, -1.25F, 0.5F};
    const auto expected = cusignal::correlate_typed_cpu(
        left, right, "same", cusignal::CorrelateMethod::direct);
    const auto actual = correlate_same_direct_parallel_exact(left, right);
    if (actual != expected)
        throw std::runtime_error(
            "Task2 pipeline parallel direct correlation exact-match gate failed");
}

CpuRun run_cpu(const task2::TaskConfig& config, Resources* resources = nullptr)
{
    CpuRun x(config);
    auto sample_before = [&](int step) {
        if (resources) {
            resources->cpu_sampling_ms += measure_ms([&] {
                resources->heap_before[step] = sample_cpu_heap();
                resources->rss_before[step] = sample_cpu_rss();
            }, [] {});
        }
    };
    auto sample_after = [&](int step) {
        if (resources) {
            resources->cpu_sampling_ms += measure_ms([&] {
                resources->heap_after[step] = sample_cpu_heap();
                resources->rss_after[step] = sample_cpu_rss();
            }, [] {});
        }
    };
    sample_before(0);
    x.step_ms[0] = measure_ms([&] { x.s1 = task2::accuracy::generate_step1_cpu_reference(config); }, [] {});
    x.state.target_waveform = x.s1.chirp_cpu; x.state.target_component = x.s1.target_cpu;
    x.state.interference_component = x.s1.interference_cpu; x.state.noise_component = x.s1.noise_cpu;
    x.state.echo = x.s1.echo_cpu; sample_after(0); sample_before(1);
    x.step_ms[1] = measure_ms([&] { x.s2 = task2::accuracy::collect_step2_validation(x.state); }, [] {});
    x.state.filtered = x.s2.filtered_cpu; sample_after(1); sample_before(2);
    x.step_ms[2] = measure_ms([&] { x.s3 = task2::accuracy::collect_step3_validation(x.state); }, [] {});
    x.state.smoothed = x.s3.smoothed_cpu; x.state.reference_for_correlation = x.s3.reference_for_correlation_cpu;
    sample_after(2); sample_before(3);
    x.step_ms[3] = measure_ms([&] {
        auto correlation = correlate_same_direct_parallel_exact(
            x.state.smoothed, x.state.reference_for_correlation);
        x.s4 = task2::accuracy::collect_step4_validation_with_precomputed_correlation(
            x.state, std::move(correlation));
    }, [] {});
    x.state.fm_feature = x.s4.fm_cpu; x.state.correlation_feature = x.s4.correlation_cpu;
    x.state.spectral_feature = x.s4.spectral_cpu; x.state.wavelet_feature = x.s4.wavelet_cpu;
    x.state.feature_bundle = x.s4.bundle_cpu; sample_after(3); sample_before(4);
    x.step_ms[4] = measure_ms([&] { x.s5 = task2::accuracy::collect_step5_cpu_validation(x.state); }, [] {});
    x.state.extrema = x.s5.expected; sample_after(4); sample_before(5);
    x.step_ms[5] = measure_ms([&] { x.s6 = run_cpu_step6(x.state); }, [] {});
    x.state.estimate = x.s6.state_cpu.front(); x.state.kalman_covariance = x.s6.covariance_cpu;
    sample_after(5);
    return x;
}

struct GpuRun {
    explicit GpuRun(const task2::TaskConfig& c) : state(c) {}
    task2::PipelineState state;
    std::array<double, 6> step_ms{};
    double event_sum{};
    std::int64_t peak{};
};

GpuRun run_gpu(const task2::TaskConfig& config, Resources* resources = nullptr)
{
    GpuRun x(config);
    std::array<task2::StepEvidence, 6> evidence;
    auto execute = [&](int i, auto fn) {
        if (resources)
            resources->gpu_sampling_ms += measure_ms([&] {
                resources->gpu_before[i] = sample_gpu_used(
                    gpu_used_bytes, "cudaMemGetInfo");
            }, [] {});
        evidence[i] = fn(x.state);
        if (resources) {
            resources->gpu_sampling_ms += measure_ms([&] {
                CUDA_CHECK(cudaDeviceSynchronize());
                resources->gpu_after[i] = sample_gpu_used(
                    gpu_used_bytes, "cudaMemGetInfo");
            }, [] {});
        }
    };
    execute(0, task2::run_step1); execute(1, task2::run_step2); execute(2, task2::run_step3);
    execute(3, task2::run_step4); execute(4, task2::run_step5); execute(5, task2::run_step6);
    for (int i = 0; i < 6; ++i) {
        x.step_ms[i] = evidence[i].total_ms; x.event_sum += evidence[i].compute_ms;
        const auto found = evidence[i].metrics.find("gpu_used_peak_bytes");
        if (found != evidence[i].metrics.end()) {
            const auto peak = static_cast<std::int64_t>(found->second);
            x.peak = std::max(x.peak, peak);
            if (resources) resources->gpu_peak[i] = peak;
        }
    }
    return x;
}

const std::vector<std::string> kIdentity{
    "schema_version", "run_id", "run_type", "test_time_utc", "git_commit", "git_dirty",
    "config_id", "config_sha256", "target_kind", "target", "task_name", "step_name",
    "operator_name", "case_id", "scale_id", "order_of_magnitude", "actual_elements",
    "input_shape", "dtype", "device", "backend", "timing_scope", "status", "error_code"};

std::vector<std::string> columns(std::initializer_list<const char*> tail)
{
    auto result = kIdentity; for (const char* name : tail) result.emplace_back(name); return result;
}

std::vector<std::string> identity(const task2::Task2ScaleConfig& scale, const Args& a,
                                  const std::string& device, const std::string& case_id)
{
    return {"1", a.run_id, a.run_type, TASK2_EVIDENCE_TEST_TIME, TASK2_EVIDENCE_GIT_COMMIT,
        TASK2_EVIDENCE_GIT_DIRTY, scale.scale_set_id, scale.config.sha256, "task",
        "Task2.pipeline", "Task2", "pipeline_total",
        "chirp+gausspulse+sawtooth+square+hamming+firwin+firfilter+cubic+fm_demod+correlate+spectrogram+cwt+argrelextrema+kalman_filter;host:ricker+fusion+observation",
        case_id, scale.scale_id, scale.order_of_magnitude, std::to_string(scale.actual_elements),
        scale.input_shape, "FP32", device, a.backend, "compatibility/end-to-end", "PASS", "OK"};
}

struct Trace { ResourceSample heap0, heap1, rss0, rss1, gpu0, gpu1; std::int64_t peak{}; };
}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args a = parse_args(argc, argv);
        verify_parallel_correlation_contract();
#if defined(USE_THRUST)
        const std::string compiled = "fft_thrust";
#elif defined(USE_DLFFT)
        const std::string compiled = "dlfft";
#else
        const std::string compiled = "unbound";
#endif
        if (a.backend != compiled) throw std::invalid_argument("backend mismatch: compiled=" + compiled);
        if (std::filesystem::exists(a.output)) throw std::invalid_argument("output directory already exists");
        const auto scale = task2::load_task2_scale(a.scales, a.scale_id);
        const auto thresholds = task2::accuracy::FormalThresholdSet::load(a.thresholds);
        const std::string case_id = make_case_id("task2_pipeline", scale.scale_id, scale.input_digest);
        const std::string suffix = a.backend + "_" + a.run_type + ".csv";
        CsvWriter timing(a.output / ("task2_iteration_timing_" + suffix), columns({
            "warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"}));
        CsvWriter memory(a.output / ("task2_memory_trace_" + suffix), columns({
            "warmup_runs","measured_runs","sample_index","resource_scope","trace_phase","phase_index",
            "cpu_live_heap_bytes","rss_bytes","gpu_used_bytes","memory_source"}));
        std::ofstream log(a.output / "task2_pipeline_full.log");
        if (!log) throw std::runtime_error("cannot create full log");
        for (int i = 0; i < a.warmup; ++i) { (void)run_cpu(scale.config); (void)run_gpu(scale.config); CUDA_CHECK(cudaDeviceSynchronize()); }
        std::vector<double> cpu_samples, gpu_samples, event_samples;
        std::vector<double> cpu_resource_overhead_samples, gpu_resource_overhead_samples;
        std::array<std::vector<double>, 6> cpu_steps, gpu_steps;
        std::vector<Trace> traces;
        std::map<std::string, task2::accuracy::OutputAccuracyResult> aggregate;
        for (int sample = 0; sample < a.measured; ++sample) {
            Trace trace;
            Resources resources;
            trace.heap0 = sample_cpu_heap(); trace.rss0 = sample_cpu_rss();
            CpuRun cpu(scale.config); const double cpu_wall_ms = measure_ms([&] { cpu = run_cpu(scale.config, &resources); }, [] {});
            const double cpu_ms = std::max(0.0, cpu_wall_ms - resources.cpu_sampling_ms);
            trace.heap1 = sample_cpu_heap(); trace.rss1 = sample_cpu_rss();
            trace.gpu0 = sample_gpu_used(gpu_used_bytes, "cudaMemGetInfo");
            GpuRun gpu(scale.config); const double gpu_wall_ms = measure_ms([&] { gpu = run_gpu(scale.config, &resources); }, [] { CUDA_CHECK(cudaDeviceSynchronize()); });
            const double gpu_ms = std::max(0.0, gpu_wall_ms - resources.gpu_sampling_ms);
            trace.gpu1 = sample_gpu_used(gpu_used_bytes, "cudaMemGetInfo"); trace.peak = gpu.peak;
            const auto accuracy = task2::accuracy::evaluate_pipeline_formal(
                gpu.state, cpu.state, thresholds);
            task2::accuracy::merge_worst_accuracy(aggregate, accuracy);
            if (!task2::accuracy::all_accuracy_pass(accuracy))
                throw std::runtime_error(task2::accuracy::accuracy_failure_message(accuracy));
            cpu_samples.push_back(cpu_ms); gpu_samples.push_back(gpu_ms); event_samples.push_back(gpu.event_sum); traces.push_back(trace);
            cpu_resource_overhead_samples.push_back(resources.cpu_sampling_ms);
            gpu_resource_overhead_samples.push_back(resources.gpu_sampling_ms);
            for (int i = 0; i < 6; ++i) { cpu_steps[i].push_back(cpu.step_ms[i]); gpu_steps[i].push_back(gpu.step_ms[i]); }
            auto add_timing = [&](const char* device, double ms, const std::string& digest) {
                auto row = identity(scale,a,device,case_id); row.insert(row.end(),{
                    std::to_string(a.warmup),std::to_string(a.measured),std::to_string(sample),number(ms),"true",scale.input_digest,digest}); timing.append(row); };
            add_timing("cpu",cpu_ms,output_digest(cpu.state)); add_timing("gpu",gpu_ms,output_digest(gpu.state));
            auto add_memory = [&](const char* device,const char* scope,const char* phase,int phase_index,
                bool cpu_ok,std::int64_t heap,std::int64_t rss,bool gpu_ok,std::int64_t gpu_bytes,const std::string& source) {
                auto row=identity(scale,a,device,case_id); row.insert(row.end(),{std::to_string(a.warmup),std::to_string(a.measured),
                    std::to_string(sample),scope,phase,std::to_string(phase_index),nullable_csv(cpu_ok,std::to_string(heap)),
                    nullable_csv(cpu_ok,std::to_string(rss)),nullable_csv(gpu_ok,std::to_string(gpu_bytes)),source}); memory.append(row); };
            const std::array<const char*,6> scopes{"step1","step2","step3","step4","step5","step6"};
            for (int i=0;i<6;++i) {
                add_memory("cpu",scopes[i],"before",0,resources.heap_before[i].available&&resources.rss_before[i].available,resources.heap_before[i].bytes,resources.rss_before[i].bytes,false,0,"same_pipeline_execution:mallinfo2+/proc/self/statm");
                add_memory("cpu",scopes[i],"peak",1,true,std::max(resources.heap_before[i].bytes,resources.heap_after[i].bytes),std::max(resources.rss_before[i].bytes,resources.rss_after[i].bytes),false,0,"same_pipeline_execution:observed_before_after_max");
                add_memory("cpu",scopes[i],"after",2,resources.heap_after[i].available&&resources.rss_after[i].available,resources.heap_after[i].bytes,resources.rss_after[i].bytes,false,0,"same_pipeline_execution:mallinfo2+/proc/self/statm");
                add_memory("gpu",scopes[i],"before",0,false,0,0,resources.gpu_before[i].available,resources.gpu_before[i].bytes,"same_pipeline_execution:cudaMemGetInfo");
                add_memory("gpu",scopes[i],"peak",1,false,0,0,true,resources.gpu_peak[i],"same_pipeline_execution:cudaMemGetInfo_while_outputs_alive");
                add_memory("gpu",scopes[i],"after",2,false,0,0,resources.gpu_after[i].available,resources.gpu_after[i].bytes,"same_pipeline_execution:cudaMemGetInfo");
            }
            add_memory("cpu","pipeline_total","before",0,trace.heap0.available&&trace.rss0.available,trace.heap0.bytes,trace.rss0.bytes,false,0,"mallinfo2+/proc/self/statm");
            add_memory("cpu","pipeline_total","peak",1,true,std::max(trace.heap0.bytes,trace.heap1.bytes),std::max(trace.rss0.bytes,trace.rss1.bytes),false,0,"observed_before_after_max");
            add_memory("cpu","pipeline_total","after",2,trace.heap1.available&&trace.rss1.available,trace.heap1.bytes,trace.rss1.bytes,false,0,"mallinfo2+/proc/self/statm");
            add_memory("gpu","pipeline_total","before",0,false,0,0,trace.gpu0.available,trace.gpu0.bytes,"cudaMemGetInfo");
            add_memory("gpu","pipeline_total","peak",1,false,0,0,true,trace.peak,"max_step_cudaMemGetInfo_while_outputs_alive");
            add_memory("gpu","pipeline_total","after",2,false,0,0,trace.gpu1.available,trace.gpu1.bytes,"cudaMemGetInfo");
            log << "sample_index="<<sample<<" cpu_pipeline_ms="<<cpu_ms<<" gpu_pipeline_ms="<<gpu_ms
                <<" cpu_resource_sampling_overhead_ms="<<resources.cpu_sampling_ms
                <<" gpu_resource_sampling_overhead_ms="<<resources.gpu_sampling_ms
                <<" semantic_gates=16/16 timing_direct=true resource_overhead_excluded=true step_sum_used=false status=PASS\n";
        }
        task2::accuracy::write_formal_accuracy_csv(a.output/("task2_accuracy_details_"+suffix),
            {a.run_id,a.run_type,TASK2_EVIDENCE_GIT_COMMIT,case_id,scale.scale_id,a.backend,a.measured},thresholds,aggregate);
        const auto cpu_stats=summarize_ms(cpu_samples), gpu_stats=summarize_ms(gpu_samples), event_stats=summarize_ms(event_samples);
        const auto cpu_resource_stats=summarize_ms(cpu_resource_overhead_samples), gpu_resource_stats=summarize_ms(gpu_resource_overhead_samples);
        const double speed=speedup(cpu_stats,gpu_stats,"compatibility/end-to-end","compatibility/end-to-end");
        std::int64_t heap_peak{},rss_peak{},gpu_peak{}; for(const auto& t:traces){heap_peak=std::max({heap_peak,t.heap0.bytes,t.heap1.bytes});rss_peak=std::max({rss_peak,t.rss0.bytes,t.rss1.bytes});gpu_peak=std::max(gpu_peak,t.peak);}
        CsvWriter main_csv(a.output/("task2_benchmark_"+suffix),columns({"warmup_runs","measured_runs","mean_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms","std_ms","cv","cpu_gpu_speedup","accuracy_reference","mse","rmse","relative_l2","relative_linf","exact_match","mismatch_count","semantic_check","accuracy_status","accuracy_threshold_set_id","accuracy_thresholds_sha256","cpu_heap_before_bytes","cpu_heap_after_bytes","cpu_heap_peak_bytes","cpu_heap_delta_bytes","rss_before_bytes","rss_after_bytes","rss_peak_bytes","rss_delta_bytes","gpu_before_bytes","gpu_after_bytes","gpu_peak_bytes","gpu_delta_bytes","return_code"}));
        auto add_main=[&](const char* device,const Statistics& s){const bool cpu=std::string(device)=="cpu";const auto& f=traces.front();const auto& l=traces.back();auto row=identity(scale,a,device,case_id);row.insert(row.end(),{std::to_string(a.warmup),std::to_string(a.measured),number(s.mean_ms),number(s.p50_ms),number(s.p95_ms),number(s.p99_ms),number(s.min_ms),number(s.max_ms),number(s.stddev_ms),number(s.cv),cpu?"NA":number(speed),"independent typed CPU six-step pipeline",number(task2::accuracy::worst_mse(aggregate)),number(task2::accuracy::worst_rmse(aggregate)),number(task2::accuracy::worst_relative_l2(aggregate)),number(task2::accuracy::worst_relative_linf(aggregate)),"NA","NA","all_six_step_semantic_rules_pass","PASS",thresholds.id(),thresholds.sha256(),nullable_csv(cpu&&f.heap0.available,std::to_string(f.heap0.bytes)),nullable_csv(cpu&&l.heap1.available,std::to_string(l.heap1.bytes)),nullable_csv(cpu,std::to_string(heap_peak)),nullable_csv(cpu,std::to_string(l.heap1.bytes-f.heap0.bytes)),nullable_csv(cpu&&f.rss0.available,std::to_string(f.rss0.bytes)),nullable_csv(cpu&&l.rss1.available,std::to_string(l.rss1.bytes)),nullable_csv(cpu,std::to_string(rss_peak)),nullable_csv(cpu,std::to_string(l.rss1.bytes-f.rss0.bytes)),nullable_csv(!cpu&&f.gpu0.available,std::to_string(f.gpu0.bytes)),nullable_csv(!cpu&&l.gpu1.available,std::to_string(l.gpu1.bytes)),nullable_csv(!cpu,std::to_string(gpu_peak)),nullable_csv(!cpu,std::to_string(l.gpu1.bytes-f.gpu0.bytes)),"0"});main_csv.append(row);};
        add_main("cpu",cpu_stats);add_main("gpu",gpu_stats);
        CsvWriter components(a.output/("task2_pipeline_summary_"+suffix),columns({"component","upstream","measured_runs","mean_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms","std_ms","cv"}));
        const std::array<const char*,6> names{"step1","step2","step3","step4","step5","step6"};
        const std::array<const char*,6> upstream{"config","step1.echo","step2.filtered+step1.target_waveform","step3.smoothed+step3.reference","step4.feature_bundle","step5.extrema+step4.feature_bundle+Kalman/delay"};
        auto add_component=[&](const char* device,const char* name,const char* source,const Statistics& s){auto row=identity(scale,a,device,case_id);row.insert(row.end(),{name,source,std::to_string(a.measured),number(s.mean_ms),number(s.p50_ms),number(s.p95_ms),number(s.p99_ms),number(s.min_ms),number(s.max_ms),number(s.stddev_ms),number(s.cv)});components.append(row);};
        for(int i=0;i<6;++i){add_component("cpu",names[i],upstream[i],summarize_ms(cpu_steps[i]));add_component("gpu",names[i],upstream[i],summarize_ms(gpu_steps[i]));}
        add_component("cpu","pipeline_total","frozen_multi_upstream_graph",cpu_stats);add_component("gpu","pipeline_total","frozen_multi_upstream_graph",gpu_stats);
        const std::size_t correlation_workers = std::min<std::size_t>(
            scale.config.step3_samples(),
            std::max<unsigned>(1U, std::thread::hardware_concurrency()));
        std::ofstream json(a.output/"task2_pipeline_summary.json");json<<"{\n  \"status\": \"PASS\",\n  \"run_id\": \""<<a.run_id<<"\",\n  \"run_type\": \""<<a.run_type<<"\",\n  \"scale_id\": \""<<scale.scale_id<<"\",\n  \"backend\": \""<<a.backend<<"\",\n  \"warmup_runs\": "<<a.warmup<<",\n  \"measured_runs\": "<<a.measured<<",\n  \"accuracy_threshold_set_id\": \""<<thresholds.id()<<"\",\n  \"accuracy_thresholds_sha256\": \""<<thresholds.sha256()<<"\",\n  \"semantic_gates_passed\": 16,\n  \"pipeline_total_direct\": true,\n  \"cpu_correlation_method\": \"direct_parallel_exact\",\n  \"cpu_correlation_workers\": "<<correlation_workers<<",\n  \"resource_sampling_mode\": \"same_pipeline_execution\",\n  \"input_digest\": \""<<scale.input_digest<<"\"\n}\n";
        std::ofstream summary(a.output/"task2_pipeline_summary.txt");summary<<"status=PASS\nwarmup_runs="<<a.warmup<<"\nmeasured_runs="<<a.measured<<"\nscale_id="<<scale.scale_id<<"\nbackend="<<a.backend<<"\nsemantic_gates=16/16\naccuracy_threshold_set_id="<<thresholds.id()<<"\naccuracy_thresholds_sha256="<<thresholds.sha256()<<"\npipeline_total_direct=true\ncpu_correlation_method=direct_parallel_exact\ncpu_correlation_workers="<<correlation_workers<<"\nresource_sampling_mode=same_pipeline_execution\naccuracy_status=PASS\n";
        log<<"function=Task2 frozen full six-step pipeline\ncall_chain=Step1->Step2;Step2+Step1->Step3;Step3 dual outputs->Step4;Step4->Step5;Step5+Step4+Kalman/delay->Step6\ntiming=pipeline_total measured directly;not=sum(step timings)\ncpu_correlation_method=direct_parallel_exact workers="<<correlation_workers<<"\nresource_sampling_mode=same_pipeline_execution\nstatus=PASS return_code=0\n";
        std::cout<<"[TASK2][PIPELINE][SUMMARY] warmup_runs="<<a.warmup<<" measured_runs="<<a.measured<<" cpu_pipeline_mean_ms="<<cpu_stats.mean_ms<<" gpu_pipeline_mean_ms="<<gpu_stats.mean_ms<<" cuda_event_aux_sum_mean_ms="<<event_stats.mean_ms<<" semantic_gates=16/16 timing_direct=true step_sum_used=false cpu_correlation_method=direct_parallel_exact cpu_correlation_workers="<<correlation_workers<<" resource_sampling_mode=same_pipeline_execution backend="<<a.backend<<" status=PASS return_code=0\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<"[TASK2][PIPELINE_BENCHMARK][FAIL] code=EXECUTION_FAILED message="<<e.what()<<'\n';return 2;}
}
