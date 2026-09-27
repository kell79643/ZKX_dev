#include "accuracy.h"
#include <cusignal/operators/bsplines/bsplines_typed.h>
#include <cusignal/runtime/copy_utils.h>
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/estimation/estimation_typed.h>
#include "memory.h"
#include <cusignal/operators/peak_finding/peak_finding_typed.h>
#include "sha256.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef ZKX_TASK2_PIPELINE_OPERATOR
#error "ZKX_TASK2_PIPELINE_OPERATOR required"
#endif

namespace {
namespace fs = std::filesystem;
constexpr const char* kOperator = ZKX_TASK2_PIPELINE_OPERATOR;

struct Config {
    std::string dtype;
    int count;
    int order;
    std::size_t warmup;
    std::size_t measured;
    fs::path output;
};

struct Phase {
    std::string name;
    int index;
    zkx::common::ResourceSample heap;
    zkx::common::ResourceSample rss;
    zkx::common::ResourceSample gpu;
};

struct Evidence {
    std::vector<double> cpu_output;
    std::vector<double> gpu_output;
    std::vector<double> cpu_samples;
    std::vector<double> gpu_samples;
    std::vector<Phase> cpu_trace;
    std::vector<Phase> gpu_trace;
    std::string manifest;
    std::string typed_digest;
    std::string input_digest;
    std::string cpu_output_digest;
    std::string gpu_output_digest;
    std::string preview;
    std::string output_dtype;
    std::size_t mismatch_count{0};
    bool discrete{false};
};

Config parse_cli(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("CLI");
        values.emplace(std::string(argv[index]).substr(2), argv[index + 1]);
    }
    auto get = [&](const char* key, const char* fallback = nullptr) {
        const auto found = values.find(key);
        if (found != values.end()) return found->second;
        if (fallback) return std::string(fallback);
        throw std::invalid_argument(key);
    };
    return {get("dtype"), std::stoi(get("count")), std::stoi(get("order", "3")),
            std::stoull(get("warmup")), std::stoull(get("measured")), get("output")};
}

zkx::common::ResourceSample sample_gpu()
{
    return zkx::common::sample_gpu_used([] {
        const auto info = cusignal::cuda_utils::current_memory_info();
        return static_cast<std::int64_t>(info.total_bytes - info.free_bytes);
    }, "cudaMemGetInfo.total_minus_free");
}

Phase phase(const char* name, int index, bool gpu)
{
    return {name, index, zkx::common::sample_cpu_heap(), zkx::common::sample_cpu_rss(),
            gpu ? sample_gpu() : zkx::common::ResourceSample{false, 0, "NA", "CPU"}};
}

template <class T> T convert(double value) { return static_cast<T>(value); }
template <> __half convert<__half>(double value)
{
    return __float2half_rn(static_cast<float>(value));
}
template <class T> double load(T value) { return static_cast<double>(value); }
template <> double load<__half>(__half value) { return __half2float(value); }

template <class T> std::string digest(const std::vector<T>& values)
{
    return zkx::remaining_operator_case::sha256_hex(std::string(
        reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T)));
}

template <class T> std::vector<double> flatten(const std::vector<T>& values)
{
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(load(value));
    return result;
}

template <class T> std::vector<T> cubic_input(int count)
{
    std::vector<T> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        double value = -2.5 + 5.0 * index / std::max(1, count - 1);
        if constexpr (std::is_integral_v<T>) value = std::round(value);
        result[static_cast<std::size_t>(index)] = convert<T>(value);
    }
    return result;
}

template <class T> std::vector<T> feature_bundle(int count)
{
    std::vector<T> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        double value = 6.0 * std::sin(0.037 * index) +
                       2.0 * std::cos(0.013 * index) +
                       ((index * 11) % 17 - 8) * 0.0625;
        if (index % 97 == 43) value += 12.0;
        if constexpr (std::is_integral_v<T>) value = std::round(value);
        result[static_cast<std::size_t>(index)] = convert<T>(value);
    }
    return result;
}

template <class T> std::vector<T> observations(int count)
{
    std::vector<T> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        double value = 1.0 + 0.25 * std::sin(0.021 * index) +
                       ((index * 7) % 9 - 4) * 0.015625;
        if constexpr (std::is_integral_v<T>) value = std::round(value);
        result[static_cast<std::size_t>(index)] = convert<T>(value);
    }
    return result;
}

template <class T> std::string preview(const std::vector<T>& values)
{
    std::vector<std::size_t> indices;
    if (values.size() <= 6) {
        for (std::size_t index = 0; index < values.size(); ++index) indices.push_back(index);
    } else {
        indices = {0, 1, 2, 3, values.size() - 2, values.size() - 1};
    }
    std::ostringstream output;
    output << std::setprecision(9);
    for (std::size_t item = 0; item < indices.size(); ++item) {
        if (item) output << ';';
        output << "i=" << indices[item] << ':' << load(values[indices[item]]);
    }
    return output.str();
}

template <class Function>
std::vector<double> measure(std::size_t warmup, std::size_t measured, Function function)
{
    for (std::size_t run = 0; run < warmup; ++run) function();
    std::vector<double> result;
    result.reserve(measured);
    for (std::size_t run = 0; run < measured; ++run) {
        const auto begin = std::chrono::steady_clock::now();
        function();
        const auto end = std::chrono::steady_clock::now();
        result.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return result;
}

template <class T> std::vector<float> as_float(const std::vector<T>& values)
{
    std::vector<float> result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(static_cast<float>(load(value)));
    return result;
}

template <class T> Evidence run(const Config& config)
{
    Evidence evidence;
    const std::string op = kOperator;
    if (config.count <= 0) throw std::invalid_argument("count");

    if (op == "cubic") {
        const auto input = cubic_input<T>(config.count);
        std::vector<T> cpu;
        std::vector<T> gpu;
        auto cpu_call = [&] { cpu = cusignal::cubic_cpu(input); };
        auto gpu_call = [&] {
            auto device_input = cusignal::DeviceArray<T>::from_host(input);
            cusignal::DeviceArray<T> device_output(input.size());
            cusignal::cubic_device(device_input, device_output);
            gpu = device_output.to_host();
        };
        evidence.cpu_samples = measure(config.warmup, config.measured, cpu_call);
        evidence.gpu_samples = measure(config.warmup, config.measured, gpu_call);

        evidence.cpu_trace.push_back(phase("before", 0, false));
        auto cpu_input = input;
        evidence.cpu_trace.push_back(phase("allocate", 1, false));
        evidence.cpu_trace.push_back(phase("h2d", 2, false));
        cpu = cusignal::cubic_cpu(cpu_input);
        evidence.cpu_trace.push_back(phase("execute_sync", 3, false));
        evidence.cpu_trace.push_back(phase("d2h", 4, false));
        cpu_input.clear();
        evidence.cpu_trace.push_back(phase("release_sync", 5, false));
        evidence.cpu_trace.push_back(phase("after", 6, false));

        evidence.gpu_trace.push_back(phase("before", 0, true));
        cusignal::DeviceArray<T> device_input(input.size());
        cusignal::DeviceArray<T> device_output(input.size());
        evidence.gpu_trace.push_back(phase("allocate", 1, true));
        cusignal::cuda_utils::copy_to_device(device_input.data(), input.data(), input.size());
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("h2d", 2, true));
        cusignal::cubic_device(device_input, device_output);
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("execute_sync", 3, true));
        gpu = device_output.to_host();
        evidence.gpu_trace.push_back(phase("d2h", 4, true));
        device_input = {};
        device_output = {};
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("release_sync", 5, true));
        evidence.gpu_trace.push_back(phase("after", 6, true));

        evidence.cpu_output = flatten(cpu);
        evidence.gpu_output = flatten(gpu);
        evidence.cpu_output_digest = digest(cpu);
        evidence.gpu_output_digest = digest(gpu);
        evidence.typed_digest = digest(input);
        evidence.manifest = "x:" + config.dtype + ":[" + std::to_string(input.size()) +
                            "]:sha256=" + evidence.typed_digest;
        evidence.preview = preview(input);
        evidence.output_dtype = config.dtype;
    } else if (op == "argrelextrema") {
        const auto input = feature_bundle<T>(config.count);
        const cusignal::RelativeExtremaShape shape({input.size()});
        std::vector<std::int64_t> cpu;
        std::vector<std::int64_t> gpu;
        auto cpu_call = [&] {
            cpu = cusignal::argrelextrema_typed_cpu(
                input, shape, cusignal::RelativeExtremaComparator::greater,
                0, config.order, "clip").coordinates.front();
        };
        auto gpu_call = [&] {
            auto device_input = cusignal::DeviceArray<T>::from_host(input);
            cusignal::RelativeExtremaDeviceWorkspace workspace(input.size());
            auto result = cusignal::argrelextrema_device(
                device_input, shape, cusignal::RelativeExtremaComparator::greater,
                workspace, 0, config.order, "clip");
            gpu = result.coordinates.front().to_host();
        };
        evidence.cpu_samples = measure(config.warmup, config.measured, cpu_call);
        evidence.gpu_samples = measure(config.warmup, config.measured, gpu_call);

        evidence.cpu_trace.push_back(phase("before", 0, false));
        auto cpu_input = input;
        evidence.cpu_trace.push_back(phase("allocate", 1, false));
        evidence.cpu_trace.push_back(phase("h2d", 2, false));
        cpu = cusignal::argrelextrema_typed_cpu(
            cpu_input, shape, cusignal::RelativeExtremaComparator::greater,
            0, config.order, "clip").coordinates.front();
        evidence.cpu_trace.push_back(phase("execute_sync", 3, false));
        evidence.cpu_trace.push_back(phase("d2h", 4, false));
        cpu_input.clear();
        evidence.cpu_trace.push_back(phase("release_sync", 5, false));
        evidence.cpu_trace.push_back(phase("after", 6, false));

        evidence.gpu_trace.push_back(phase("before", 0, true));
        cusignal::DeviceArray<T> device_input(input.size());
        cusignal::RelativeExtremaDeviceWorkspace workspace(input.size());
        evidence.gpu_trace.push_back(phase("allocate", 1, true));
        cusignal::cuda_utils::copy_to_device(device_input.data(), input.data(), input.size());
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("h2d", 2, true));
        auto result = cusignal::argrelextrema_device(
            device_input, shape, cusignal::RelativeExtremaComparator::greater,
            workspace, 0, config.order, "clip");
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("execute_sync", 3, true));
        gpu = result.coordinates.front().to_host();
        evidence.gpu_trace.push_back(phase("d2h", 4, true));
        result = {};
        workspace = {};
        device_input = {};
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("release_sync", 5, true));
        evidence.gpu_trace.push_back(phase("after", 6, true));

        evidence.cpu_output = flatten(cpu);
        evidence.gpu_output = flatten(gpu);
        evidence.cpu_output_digest = digest(cpu);
        evidence.gpu_output_digest = digest(gpu);
        evidence.typed_digest = digest(input);
        evidence.manifest = "feature_bundle:" + config.dtype + ":[" +
            std::to_string(input.size()) + "]:sha256=" + evidence.typed_digest +
            "|order=" + std::to_string(config.order) + "|mode=clip";
        evidence.preview = preview(input);
        evidence.output_dtype = "INT64_COORDINATES";
        evidence.discrete = true;
        evidence.mismatch_count = cpu.size() > gpu.size() ? cpu.size() - gpu.size() : gpu.size() - cpu.size();
        const std::size_t shared = std::min(cpu.size(), gpu.size());
        for (std::size_t index = 0; index < shared; ++index)
            if (cpu[index] != gpu[index]) ++evidence.mismatch_count;
    } else if (op == "kalman_filter") {
        const auto typed_observations = observations<T>(config.count);
        const std::vector<T> initial_x{convert<T>(0)}, initial_p{convert<T>(1)};
        const std::vector<T> f{convert<T>(1)}, q{convert<T>(1)}, alpha{convert<T>(1)};
        const std::vector<T> h{convert<T>(1)}, r{convert<T>(2)};
        const cusignal::KalmanLayout layout(1, 1, 1);
        std::vector<float> cpu;
        std::vector<float> gpu;
        auto cpu_call = [&] {
            auto state = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
            for (const auto observation : typed_observations) {
                cusignal::kalman_predict_typed_cpu(state, f, q, alpha);
                cusignal::kalman_update_typed_cpu(state, h, r, std::vector<T>{observation});
            }
            cpu = {state.x.front(), state.p.front()};
        };
        const auto x_fp = as_float(initial_x), p_fp = as_float(initial_p);
        const auto operator_fp = as_float(f), q_fp = as_float(q), alpha_fp = as_float(alpha);
        const auto h_fp = as_float(h), r_fp = as_float(r);
        const auto observations_fp = as_float(typed_observations);
        auto gpu_call = [&] {
            auto dx = cusignal::DeviceArray<float>::from_host(x_fp);
            auto dp = cusignal::DeviceArray<float>::from_host(p_fp);
            auto df = cusignal::DeviceArray<float>::from_host(operator_fp);
            auto dq = cusignal::DeviceArray<float>::from_host(q_fp);
            auto da = cusignal::DeviceArray<float>::from_host(alpha_fp);
            auto dh = cusignal::DeviceArray<float>::from_host(h_fp);
            auto dr = cusignal::DeviceArray<float>::from_host(r_fp);
            auto dz = cusignal::DeviceArray<float>::from_host(observations_fp);
            cusignal::KalmanDeviceState state(layout);
            cusignal::initialize_kalman_device_state(dx, dp, state);
            cusignal::kalman_predict_update_scalar_sequence_device(
                state, df, dq, da, dh, dr, dz);
            const auto gx = state.x.to_host();
            const auto gp = state.p.to_host();
            gpu = {gx.front(), gp.front()};
        };
        evidence.cpu_samples = measure(config.warmup, config.measured, cpu_call);
        evidence.gpu_samples = measure(config.warmup, config.measured, gpu_call);

        evidence.cpu_trace.push_back(phase("before", 0, false));
        auto cpu_input = typed_observations;
        evidence.cpu_trace.push_back(phase("allocate", 1, false));
        evidence.cpu_trace.push_back(phase("h2d", 2, false));
        cpu_call();
        evidence.cpu_trace.push_back(phase("execute_sync", 3, false));
        evidence.cpu_trace.push_back(phase("d2h", 4, false));
        cpu_input.clear();
        evidence.cpu_trace.push_back(phase("release_sync", 5, false));
        evidence.cpu_trace.push_back(phase("after", 6, false));

        evidence.gpu_trace.push_back(phase("before", 0, true));
        auto dx = cusignal::DeviceArray<float>::from_host(x_fp);
        auto dp = cusignal::DeviceArray<float>::from_host(p_fp);
        auto df = cusignal::DeviceArray<float>::from_host(operator_fp);
        auto dq = cusignal::DeviceArray<float>::from_host(q_fp);
        auto da = cusignal::DeviceArray<float>::from_host(alpha_fp);
        auto dh = cusignal::DeviceArray<float>::from_host(h_fp);
        auto dr = cusignal::DeviceArray<float>::from_host(r_fp);
        auto dz = cusignal::DeviceArray<float>::from_host(observations_fp);
        cusignal::KalmanDeviceState state(layout);
        evidence.gpu_trace.push_back(phase("allocate", 1, true));
        evidence.gpu_trace.push_back(phase("h2d", 2, true));
        cusignal::initialize_kalman_device_state(dx, dp, state);
        cusignal::kalman_predict_update_scalar_sequence_device(
            state, df, dq, da, dh, dr, dz);
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("execute_sync", 3, true));
        const auto gx = state.x.to_host();
        const auto gp = state.p.to_host();
        gpu = {gx.front(), gp.front()};
        evidence.gpu_trace.push_back(phase("d2h", 4, true));
        state = {};
        dx = {}; dp = {}; df = {}; dq = {}; da = {}; dh = {}; dr = {}; dz = {};
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("release_sync", 5, true));
        evidence.gpu_trace.push_back(phase("after", 6, true));

        evidence.cpu_output = flatten(cpu);
        evidence.gpu_output = flatten(gpu);
        evidence.cpu_output_digest = digest(cpu);
        evidence.gpu_output_digest = digest(gpu);
        evidence.typed_digest = digest(typed_observations);
        evidence.manifest = "observations:" + config.dtype + ":[" +
            std::to_string(typed_observations.size()) + "]:sha256=" + evidence.typed_digest +
            "|execution_dtype=FP32_TASK_SEQUENCE|f=1|q=1|alpha=1|h=1|r=2";
        evidence.preview = preview(typed_observations);
        evidence.output_dtype = "FP32_STATE_COVARIANCE";
    } else {
        throw std::invalid_argument("operator");
    }
    evidence.input_digest = zkx::remaining_operator_case::sha256_hex(evidence.manifest);
    return evidence;
}

void write_samples(std::ostream& output, const std::vector<double>& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << std::setprecision(17) << values[index];
    }
    output << ']';
}

void write_resource(std::ostream& output, const zkx::common::ResourceSample& sample)
{
    output << "{\"available\":" << (sample.available ? "true" : "false")
           << ",\"bytes\":" << (sample.available ? std::to_string(sample.bytes) : "null") << '}';
}

void write_trace(std::ostream& output, const std::vector<Phase>& phases)
{
    output << '[';
    for (std::size_t index = 0; index < phases.size(); ++index) {
        if (index) output << ',';
        output << "{\"phase\":\"" << phases[index].name
               << "\",\"phase_index\":" << phases[index].index << ",\"heap\":";
        write_resource(output, phases[index].heap);
        output << ",\"rss\":"; write_resource(output, phases[index].rss);
        output << ",\"gpu\":"; write_resource(output, phases[index].gpu);
        output << '}';
    }
    output << ']';
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_cli(argc, argv);
        Evidence evidence;
        if (config.dtype == "FP32") evidence = run<float>(config);
        else if (config.dtype == "FP16") evidence = run<__half>(config);
        else if (config.dtype == "INT32") evidence = run<std::int32_t>(config);
        else if (config.dtype == "INT16") evidence = run<std::int16_t>(config);
        else if (config.dtype == "INT8") evidence = run<std::int8_t>(config);
        else throw std::invalid_argument("dtype");

        zkx::common::Accuracy accuracy{};
        if (!evidence.discrete)
            accuracy = zkx::common::compare(evidence.cpu_output, evidence.gpu_output, 1.0e-12);
        if (fs::exists(config.output)) throw std::runtime_error("output exists");
        std::ofstream output(config.output);
        output << "{\"operator\":\"" << kOperator
               << "\",\"requested_dtype\":\"" << config.dtype
               << "\",\"output_dtype\":\"" << evidence.output_dtype
               << "\",\"backend\":\"not_applicable\""
               << ",\"output_count\":" << evidence.cpu_output.size()
               << ",\"discrete\":" << (evidence.discrete ? "true" : "false")
               << ",\"typed_input_manifest\":\"" << evidence.manifest
               << "\",\"typed_array_digest\":\"" << evidence.typed_digest
               << "\",\"actual_input_digest\":\"" << evidence.input_digest
               << "\",\"preview0\":\"" << evidence.preview
               << "\",\"cpu_output_digest\":\"" << evidence.cpu_output_digest
               << "\",\"gpu_output_digest\":\"" << evidence.gpu_output_digest
               << "\",\"mismatch_count\":" << evidence.mismatch_count
               << ",\"exact_match\":" << (evidence.mismatch_count == 0 ? "true" : "false")
               << ",\"mse\":" << accuracy.mse << ",\"rmse\":" << accuracy.rmse
               << ",\"relative_l2\":" << accuracy.relative_l2
               << ",\"relative_linf\":" << accuracy.relative_linf
               << ",\"cpu_samples_ms\":";
        write_samples(output, evidence.cpu_samples);
        output << ",\"gpu_samples_ms\":"; write_samples(output, evidence.gpu_samples);
        output << ",\"cpu_trace\":"; write_trace(output, evidence.cpu_trace);
        output << ",\"gpu_trace\":"; write_trace(output, evidence.gpu_trace);
        output << "}\n";
        std::cout << "OPERATOR_CASE_OPERATOR_BENCHMARKINAL_PROBE PASS operator=" << kOperator
                  << " dtype=" << config.dtype << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
