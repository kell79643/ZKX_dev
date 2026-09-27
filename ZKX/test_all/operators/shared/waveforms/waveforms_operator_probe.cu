#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include "memory.h"
#include "sha256.h"
#include <cusignal/operators/waveforms/waveforms_typed.h>

#include <cuda_fp16.h>
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

#ifndef ZKX_OPERATOR_CASE_WAVEFORM_OPERATOR
#error "ZKX_OPERATOR_CASE_WAVEFORM_OPERATOR is required"
#endif

namespace {
namespace fs = std::filesystem;
constexpr const char* kOperator = ZKX_OPERATOR_CASE_WAVEFORM_OPERATOR;

struct Cli {
    std::string dtype;
    std::size_t count;
    double p0, p1, p2, p3;
    std::size_t warmup, measured;
    fs::path output;
};
struct Phase {
    std::string name;
    int index;
    zkx::common::ResourceSample heap, rss, gpu;
};
struct Evidence {
    std::vector<double> cpu_values, gpu_values, cpu_ms, gpu_ms;
    std::vector<Phase> cpu_trace, gpu_trace;
    std::string typed_manifest, typed_array_digest, input_digest;
    std::string cpu_digest, gpu_digest, output_dtype, generator, preview;
};

Cli parse_cli(int argc, char** argv) {
    std::map<std::string, std::string> values;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("missing CLI value");
        values.emplace(std::string(argv[i]).substr(2), argv[i + 1]);
    }
    auto get = [&](const char* key, const char* fallback = nullptr) {
        const auto it = values.find(key);
        if (it != values.end()) return it->second;
        if (fallback) return std::string(fallback);
        throw std::invalid_argument(std::string("missing --") + key);
    };
    return {get("dtype"), std::stoull(get("count")), std::stod(get("p0", "0")),
            std::stod(get("p1", "0")), std::stod(get("p2", "0")),
            std::stod(get("p3", "0")), std::stoull(get("warmup")),
            std::stoull(get("measured")), get("output")};
}

zkx::common::ResourceSample gpu_sample() {
    return zkx::common::sample_gpu_used([] {
        const auto info = cusignal::cuda_utils::current_memory_info();
        return static_cast<std::int64_t>(info.total_bytes - info.free_bytes);
    }, "cudaMemGetInfo.total_minus_free");
}
Phase sample_phase(const char* name, int index, bool gpu) {
    return {name, index, zkx::common::sample_cpu_heap(), zkx::common::sample_cpu_rss(),
            gpu ? gpu_sample() : zkx::common::ResourceSample{false, 0, "NA", "CPU phase"}};
}

template<class T> T typed_value(double value) { return static_cast<T>(value); }
template<> __half typed_value<__half>(double value) { return __float2half_rn(static_cast<float>(value)); }
template<class T> double numeric_value(T value) { return static_cast<double>(value); }
template<> double numeric_value<__half>(__half value) { return __half2float(value); }

template<class T>
std::vector<T> make_input(std::size_t count, const std::string& op, std::string& generator) {
    std::vector<T> input(count);
    constexpr bool integral = std::is_integral_v<T>;
    if (op == "chirp") generator = integral ? "quantized_uniform_time_0_to_31(floor(31*i/(N-1)))" : "uniform_time_0_to_31(31*i/(N-1))";
    else if (op == "gausspulse") generator = integral ? "quantized_centered_time_minus16_to_16(round(32*i/(N-1)-16))" : "centered_time_minus16_to_16(32*i/(N-1)-16)";
    else generator = integral ? "quantized_periodic_phase_8cycles(floor(16*pi*i/(N-1)))" : "periodic_phase_8cycles(16*pi*i/(N-1))";
    for (std::size_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (op == "chirp") {
            const double raw = count == 1 ? 0.0 : 31.0 * static_cast<double>(i) / (count - 1);
            value = integral ? std::floor(raw) : raw;
        } else if (op == "gausspulse") {
            const double raw = count == 1 ? 0.0 : 32.0 * static_cast<double>(i) / (count - 1) - 16.0;
            value = integral ? std::round(raw) : raw;
        } else {
            const double raw = count == 1 ? 0.0 : 16.0 * 3.14159265358979323846 * static_cast<double>(i) / (count - 1);
            value = integral ? std::floor(raw) : raw;
        }
        input[i] = typed_value<T>(value);
    }
    return input;
}

template<class T> std::vector<double> as_double(const std::vector<T>& input) {
    std::vector<double> result;
    result.reserve(input.size());
    for (const auto value : input) result.push_back(numeric_value(value));
    return result;
}
template<class T> std::string bytes_digest(const std::vector<T>& values) {
    return zkx::remaining_operator_case::sha256_hex(std::string(
        reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T)));
}
std::string preview(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::setprecision(9);
    std::vector<std::size_t> indices;
    if (values.size() <= 6) for (std::size_t i = 0; i < values.size(); ++i) indices.push_back(i);
    else indices = {0, 1, 2, 3, values.size() - 2, values.size() - 1};
    for (std::size_t j = 0; j < indices.size(); ++j) {
        if (j) out << ';';
        out << "i=" << indices[j] << ':' << values[indices[j]];
    }
    return out.str();
}

template<class Function>
std::vector<double> measure(std::size_t warmup, std::size_t measured, Function function) {
    for (std::size_t i = 0; i < warmup; ++i) function();
    std::vector<double> samples;
    samples.reserve(measured);
    for (std::size_t i = 0; i < measured; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return samples;
}

template<class T>
Evidence run(const Cli& cli) {
    Evidence evidence;
    const std::string op = kOperator;
    const auto input = make_input<T>(cli.count, op, evidence.generator);
    evidence.preview = preview(as_double(input));
    evidence.typed_array_digest = bytes_digest(input);
    {
        std::ostringstream manifest;
        manifest << "input:" << cli.dtype << ":count=" << input.size()
                 << ":bytes=" << input.size() * sizeof(T)
                 << ":sha256=" << evidence.typed_array_digest
                 << "|p0=" << std::setprecision(17) << cli.p0 << "|p1=" << cli.p1
                 << "|p2=" << cli.p2 << "|p3=" << cli.p3;
        evidence.typed_manifest = manifest.str();
        evidence.input_digest = zkx::remaining_operator_case::sha256_hex(evidence.typed_manifest);
    }

    std::vector<T> cpu_typed, gpu_typed;
    std::vector<double> cpu_fp64, gpu_fp64;
    auto cpu_call = [&] {
        if (op == "chirp") cpu_typed = cusignal::chirp_typed_cpu(input, static_cast<float>(cli.p0), static_cast<float>(cli.p1), static_cast<float>(cli.p2), static_cast<float>(cli.p3));
        else if (op == "gausspulse") cpu_typed = cusignal::gausspulse_typed_cpu(input, static_cast<float>(cli.p0), static_cast<float>(cli.p1));
        else if (op == "sawtooth") cpu_fp64 = cusignal::sawtooth_typed_cpu(input, cli.p0);
        else if (op == "square") cpu_fp64 = cusignal::square_typed_cpu(input, cli.p0);
        else throw std::invalid_argument("unsupported compiled operator");
    };
    auto gpu_call = [&] {
        auto device_input = cusignal::DeviceArray<T>::from_host(input);
        if (op == "chirp" || op == "gausspulse") {
            cusignal::DeviceArray<T> device_output(input.size());
            if (op == "chirp") cusignal::chirp_device(device_input, device_output, static_cast<float>(cli.p0), static_cast<float>(cli.p1), static_cast<float>(cli.p2), static_cast<float>(cli.p3));
            else cusignal::gausspulse_device(device_input, device_output, static_cast<float>(cli.p0), static_cast<float>(cli.p1));
            gpu_typed = device_output.to_host();
        } else {
            cusignal::DeviceArray<double> device_output(input.size());
            if (op == "sawtooth") cusignal::sawtooth_device(device_input, device_output, cli.p0);
            else cusignal::square_device(device_input, device_output, cli.p0);
            gpu_fp64 = device_output.to_host();
        }
        cusignal::cuda_utils::synchronize_stream();
    };
    evidence.cpu_ms = measure(cli.warmup, cli.measured, cpu_call);
    evidence.gpu_ms = measure(cli.warmup, cli.measured, gpu_call);

    evidence.cpu_trace.push_back(sample_phase("before", 0, false));
    std::vector<T> cpu_input = input;
    evidence.cpu_trace.push_back(sample_phase("allocate", 1, false));
    evidence.cpu_trace.push_back(sample_phase("h2d", 2, false));
    if (op == "chirp") cpu_typed = cusignal::chirp_typed_cpu(cpu_input, static_cast<float>(cli.p0), static_cast<float>(cli.p1), static_cast<float>(cli.p2), static_cast<float>(cli.p3));
    else if (op == "gausspulse") cpu_typed = cusignal::gausspulse_typed_cpu(cpu_input, static_cast<float>(cli.p0), static_cast<float>(cli.p1));
    else if (op == "sawtooth") cpu_fp64 = cusignal::sawtooth_typed_cpu(cpu_input, cli.p0);
    else cpu_fp64 = cusignal::square_typed_cpu(cpu_input, cli.p0);
    evidence.cpu_trace.push_back(sample_phase("execute_sync", 3, false));
    evidence.cpu_values = (op == "chirp" || op == "gausspulse") ? as_double(cpu_typed) : cpu_fp64;
    evidence.cpu_digest = (op == "chirp" || op == "gausspulse")
        ? bytes_digest(cpu_typed) : bytes_digest(cpu_fp64);
    evidence.cpu_trace.push_back(sample_phase("d2h", 4, false));
    cpu_input.clear(); cpu_typed.clear(); cpu_fp64.clear();
    evidence.cpu_trace.push_back(sample_phase("release_sync", 5, false));
    evidence.cpu_trace.push_back(sample_phase("after", 6, false));

    evidence.gpu_trace.push_back(sample_phase("before", 0, true));
    cusignal::DeviceArray<T> device_input;
    cusignal::DeviceArray<T> device_output_t;
    cusignal::DeviceArray<double> device_output_d;
    if (op == "chirp" || op == "gausspulse") device_output_t = cusignal::DeviceArray<T>(input.size());
    else device_output_d = cusignal::DeviceArray<double>(input.size());
    evidence.gpu_trace.push_back(sample_phase("allocate", 1, true));
    device_input = cusignal::DeviceArray<T>::from_host(input);
    evidence.gpu_trace.push_back(sample_phase("h2d", 2, true));
    if (op == "chirp") cusignal::chirp_device(device_input, device_output_t, static_cast<float>(cli.p0), static_cast<float>(cli.p1), static_cast<float>(cli.p2), static_cast<float>(cli.p3));
    else if (op == "gausspulse") cusignal::gausspulse_device(device_input, device_output_t, static_cast<float>(cli.p0), static_cast<float>(cli.p1));
    else if (op == "sawtooth") cusignal::sawtooth_device(device_input, device_output_d, cli.p0);
    else cusignal::square_device(device_input, device_output_d, cli.p0);
    cusignal::cuda_utils::synchronize_stream();
    evidence.gpu_trace.push_back(sample_phase("execute_sync", 3, true));
    if (op == "chirp" || op == "gausspulse") {
        gpu_typed = device_output_t.to_host();
        evidence.gpu_values = as_double(gpu_typed);
    } else {
        gpu_fp64 = device_output_d.to_host();
        evidence.gpu_values = gpu_fp64;
    }
    evidence.gpu_trace.push_back(sample_phase("d2h", 4, true));
    device_input = {}; device_output_t = {}; device_output_d = {};
    cusignal::cuda_utils::synchronize_stream();
    evidence.gpu_trace.push_back(sample_phase("release_sync", 5, true));
    evidence.gpu_trace.push_back(sample_phase("after", 6, true));

    if (op == "chirp" || op == "gausspulse") {
        evidence.output_dtype = cli.dtype;
        evidence.gpu_digest = bytes_digest(gpu_typed);
    } else {
        evidence.output_dtype = "FP64";
        evidence.gpu_digest = bytes_digest(evidence.gpu_values);
    }
    return evidence;
}

void emit_samples(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << std::setprecision(17) << values[i];
    }
    out << ']';
}
void emit_resource(std::ostream& out, const zkx::common::ResourceSample& sample) {
    out << "{\"available\":" << (sample.available ? "true" : "false")
        << ",\"bytes\":" << (sample.available ? std::to_string(sample.bytes) : "null") << '}';
}
void emit_trace(std::ostream& out, const std::vector<Phase>& trace) {
    out << '[';
    for (std::size_t i = 0; i < trace.size(); ++i) {
        if (i) out << ',';
        out << "{\"phase\":\"" << trace[i].name << "\",\"phase_index\":" << trace[i].index << ",\"heap\":";
        emit_resource(out, trace[i].heap);
        out << ",\"rss\":"; emit_resource(out, trace[i].rss);
        out << ",\"gpu\":"; emit_resource(out, trace[i].gpu);
        out << '}';
    }
    out << ']';
}
int dispatch(const Cli& cli, Evidence& evidence) {
    if (cli.dtype == "FP32") evidence = run<float>(cli);
    else if (cli.dtype == "FP16") evidence = run<__half>(cli);
    else if (cli.dtype == "INT32") evidence = run<std::int32_t>(cli);
    else if (cli.dtype == "INT16") evidence = run<std::int16_t>(cli);
    else if (cli.dtype == "INT8") evidence = run<std::int8_t>(cli);
    else return 2;
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        const Cli cli = parse_cli(argc, argv);
        Evidence evidence;
        if (dispatch(cli, evidence) != 0) return 2;
        const auto accuracy = zkx::common::compare(evidence.cpu_values, evidence.gpu_values, 1e-12);
        if (fs::exists(cli.output)) throw std::runtime_error("output exists");
        std::ofstream out(cli.output);
        out << "{\"operator\":\"" << kOperator << "\",\"requested_dtype\":\"" << cli.dtype
            << "\",\"output_dtype\":\"" << evidence.output_dtype
            << "\",\"typed_input_manifest\":\"" << evidence.typed_manifest
            << "\",\"typed_array_digest\":\"" << evidence.typed_array_digest
            << "\",\"actual_input_digest\":\"" << evidence.input_digest
            << "\",\"input_generator\":\"" << evidence.generator
            << "\",\"input_preview_heacuda_api_tail2\":\"" << evidence.preview
            << "\",\"cpu_output_digest\":\"" << evidence.cpu_digest
            << "\",\"gpu_output_digest\":\"" << evidence.gpu_digest
            << "\",\"mse\":" << accuracy.mse << ",\"rmse\":" << accuracy.rmse
            << ",\"relative_l2\":" << accuracy.relative_l2
            << ",\"relative_linf\":" << accuracy.relative_linf
            << ",\"cpu_samples_ms\":"; emit_samples(out, evidence.cpu_ms);
        out << ",\"gpu_samples_ms\":"; emit_samples(out, evidence.gpu_ms);
        out << ",\"cpu_trace\":"; emit_trace(out, evidence.cpu_trace);
        out << ",\"gpu_trace\":"; emit_trace(out, evidence.gpu_trace);
        out << "}\n";
        std::cout << "OPERATOR_CASE_WAVEFORM_PROBE PASS operator=" << kOperator
                  << " dtype=" << cli.dtype << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
