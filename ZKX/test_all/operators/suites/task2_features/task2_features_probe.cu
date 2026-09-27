#include "accuracy.h"
#include <cusignal/operators/convolution/convolution_typed.h>
#include <cusignal/runtime/copy_utils.h>
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/demod/demod_typed.h>
#include "memory.h"
#include "sha256.h"
#include <cusignal/operators/spectral_analysis/spectral_analysis_typed.h>

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

#ifndef ZKX_TASK2_FEATURE_OPERATOR
#error "ZKX_TASK2_FEATURE_OPERATOR required"
#endif

namespace {
namespace fs = std::filesystem;
constexpr const char* kOperator = ZKX_TASK2_FEATURE_OPERATOR;

struct Config {
    std::string dtype;
    int count;
    int reference_count;
    int frame;
    int hop;
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
    std::string preview0;
    std::string preview1;
    std::string output_dtype;
    std::string backend;
    std::size_t mismatch_count{0};
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
    return {get("dtype"), std::stoi(get("count")),
            std::stoi(get("reference-count", "31")),
            std::stoi(get("frame", "64")), std::stoi(get("hop", "32")),
            std::stoull(get("warmup")), std::stoull(get("measured")),
            get("output")};
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
    return {name, index, zkx::common::sample_cpu_heap(),
            zkx::common::sample_cpu_rss(),
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

template <class T> std::vector<T> signal(int count, int salt)
{
    std::vector<T> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        double value = 4.0 * std::sin(0.031 * index) +
                       2.0 * std::cos(0.071 * index) +
                       ((index * 17 + salt) % 13 - 6) * 0.125;
        if constexpr (std::is_integral_v<T>) value = std::round(value);
        result[static_cast<std::size_t>(index)] = convert<T>(value);
    }
    return result;
}

template <class T> std::vector<cusignal::DemodComplex<T>> analytic(int count)
{
    std::vector<cusignal::DemodComplex<T>> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        const double phase_value = 0.025 * index + 0.000002 * index * index;
        double real = 20.0 * std::cos(phase_value);
        double imag = 20.0 * std::sin(phase_value);
        if constexpr (std::is_integral_v<T>) {
            real = std::round(real);
            imag = std::round(imag);
        }
        result[static_cast<std::size_t>(index)] =
            {convert<T>(real), convert<T>(imag)};
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

template <class T>
std::string complex_preview(const std::vector<cusignal::DemodComplex<T>>& values)
{
    const std::vector<std::size_t> indices = values.size() <= 4
        ? std::vector<std::size_t>{0, values.size() - 1}
        : std::vector<std::size_t>{0, 1, 2, 3, values.size() - 2, values.size() - 1};
    std::ostringstream output;
    output << std::setprecision(9);
    for (std::size_t item = 0; item < indices.size(); ++item) {
        if (item) output << ';';
        const auto& value = values[indices[item]];
        output << "i=" << indices[item] << ":(" << load(value.re)
               << ',' << load(value.im) << ')';
    }
    return output.str();
}

template <class Function>
std::vector<double> measure(std::size_t warmup, std::size_t measured, Function function)
{
    for (std::size_t run = 0; run < warmup; ++run) function();
    std::vector<double> samples;
    samples.reserve(measured);
    for (std::size_t run = 0; run < measured; ++run) {
        const auto begin = std::chrono::steady_clock::now();
        function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return samples;
}

template <class T> Evidence run(const Config& config)
{
    Evidence evidence;
    const std::string op = kOperator;
    if (config.count <= 64) throw std::invalid_argument("Task2 Step4 count");

    if (op == "fm_demod") {
        const auto input = analytic<T>(config.count);
        cusignal::FmDemodWorkspace workspace(cusignal::FmDemodOptions{{config.count}, -1});
        std::vector<float> cpu;
        std::vector<float> gpu;
        auto cpu_call = [&] { cpu = cusignal::fm_demod_typed_cpu(input, workspace); };
        auto gpu_call = [&] {
            auto device_input = cusignal::DeviceArray<cusignal::DemodComplex<T>>::from_host(input);
            cusignal::DeviceArray<float> device_output(workspace.output_size());
            cusignal::fm_demod_device(device_input, device_output, workspace);
            gpu = device_output.to_host();
            cusignal::cuda_utils::synchronize_stream();
        };
        evidence.cpu_samples = measure(config.warmup, config.measured, cpu_call);
        evidence.gpu_samples = measure(config.warmup, config.measured, gpu_call);

        evidence.cpu_trace.push_back(phase("before", 0, false));
        auto cpu_input = input;
        evidence.cpu_trace.push_back(phase("allocate", 1, false));
        evidence.cpu_trace.push_back(phase("h2d", 2, false));
        cpu = cusignal::fm_demod_typed_cpu(cpu_input, workspace);
        evidence.cpu_trace.push_back(phase("execute_sync", 3, false));
        evidence.cpu_trace.push_back(phase("d2h", 4, false));
        cpu_input.clear();
        evidence.cpu_trace.push_back(phase("release_sync", 5, false));
        evidence.cpu_trace.push_back(phase("after", 6, false));

        evidence.gpu_trace.push_back(phase("before", 0, true));
        cusignal::DeviceArray<cusignal::DemodComplex<T>> device_input(input.size());
        cusignal::DeviceArray<float> device_output(workspace.output_size());
        evidence.gpu_trace.push_back(phase("allocate", 1, true));
        cusignal::cuda_utils::copy_to_device(
            device_input.data(), input.data(), input.size());
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("h2d", 2, true));
        cusignal::fm_demod_device(device_input, device_output, workspace);
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
        evidence.manifest = "analytic:Complex" + config.dtype + ":[" +
            std::to_string(input.size()) + "]:sha256=" + evidence.typed_digest;
        evidence.preview0 = complex_preview(input);
        evidence.preview1 = "NA";
        evidence.output_dtype = "FP32";
        evidence.backend = "not_applicable";
    } else if (op == "correlate") {
        const auto input = signal<T>(config.count, 41);
        const auto reference = signal<T>(config.reference_count, 7);
        std::vector<T> cpu;
        std::vector<T> gpu;
        auto cpu_call = [&] {
            cpu = cusignal::correlate_typed_cpu(
                input, reference, "same", cusignal::CorrelateMethod::direct);
        };
        auto gpu_call = [&] {
            auto device_input = cusignal::DeviceArray<T>::from_host(input);
            auto device_reference = cusignal::DeviceArray<T>::from_host(reference);
            cusignal::DeviceArray<T> device_output(input.size());
            cusignal::correlate_device(
                device_input, device_reference, device_output, "same");
            gpu = device_output.to_host();
            cusignal::cuda_utils::synchronize_stream();
        };
        evidence.cpu_samples = measure(config.warmup, config.measured, cpu_call);
        evidence.gpu_samples = measure(config.warmup, config.measured, gpu_call);

        evidence.cpu_trace.push_back(phase("before", 0, false));
        auto cpu_input = input;
        auto cpu_reference = reference;
        evidence.cpu_trace.push_back(phase("allocate", 1, false));
        evidence.cpu_trace.push_back(phase("h2d", 2, false));
        cpu = cusignal::correlate_typed_cpu(
            cpu_input, cpu_reference, "same", cusignal::CorrelateMethod::direct);
        evidence.cpu_trace.push_back(phase("execute_sync", 3, false));
        evidence.cpu_trace.push_back(phase("d2h", 4, false));
        cpu_input.clear();
        cpu_reference.clear();
        evidence.cpu_trace.push_back(phase("release_sync", 5, false));
        evidence.cpu_trace.push_back(phase("after", 6, false));

        evidence.gpu_trace.push_back(phase("before", 0, true));
        cusignal::DeviceArray<T> device_input(input.size());
        cusignal::DeviceArray<T> device_reference(reference.size());
        cusignal::DeviceArray<T> device_output(input.size());
        evidence.gpu_trace.push_back(phase("allocate", 1, true));
        cusignal::cuda_utils::copy_to_device(
            device_input.data(), input.data(), input.size());
        cusignal::cuda_utils::copy_to_device(
            device_reference.data(), reference.data(), reference.size());
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("h2d", 2, true));
        cusignal::correlate_device(
            device_input, device_reference, device_output, "same");
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("execute_sync", 3, true));
        gpu = device_output.to_host();
        evidence.gpu_trace.push_back(phase("d2h", 4, true));
        device_input = {};
        device_reference = {};
        device_output = {};
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("release_sync", 5, true));
        evidence.gpu_trace.push_back(phase("after", 6, true));

        evidence.cpu_output = flatten(cpu);
        evidence.gpu_output = flatten(gpu);
        evidence.cpu_output_digest = digest(cpu);
        evidence.gpu_output_digest = digest(gpu);
        evidence.typed_digest = digest(input);
        evidence.manifest = "signal:" + config.dtype + ":[" +
            std::to_string(input.size()) + "]:sha256=" + digest(input) +
            "|reference:" + config.dtype + ":[" +
            std::to_string(reference.size()) + "]:sha256=" + digest(reference);
        evidence.preview0 = preview(input);
        evidence.preview1 = preview(reference);
        evidence.output_dtype = config.dtype;
        evidence.backend = "not_applicable";
        for (std::size_t index = 0; index < cpu.size(); ++index)
            if (load(cpu[index]) != load(gpu[index])) ++evidence.mismatch_count;
    } else if (op == "spectrogram") {
        const auto input = signal<T>(config.count, 19);
        const int frames = 1 + (config.count - config.frame) / config.hop;
        std::vector<float> cpu;
        std::vector<float> gpu;
        auto cpu_call = [&] {
            cpu = cusignal::spectrogram_typed_cpu(input, config.frame, config.hop);
        };
        auto gpu_call = [&] {
            auto device_input = cusignal::DeviceArray<T>::from_host(input);
            cusignal::DeviceArray<float> device_output(
                static_cast<std::size_t>(config.frame) * frames);
            cusignal::spectrogram_device(
                device_input, config.frame, config.hop, device_output);
            gpu = device_output.to_host();
            cusignal::cuda_utils::synchronize_stream();
        };
        evidence.cpu_samples = measure(config.warmup, config.measured, cpu_call);
        evidence.gpu_samples = measure(config.warmup, config.measured, gpu_call);

        evidence.cpu_trace.push_back(phase("before", 0, false));
        auto cpu_input = input;
        evidence.cpu_trace.push_back(phase("allocate", 1, false));
        evidence.cpu_trace.push_back(phase("h2d", 2, false));
        cpu = cusignal::spectrogram_typed_cpu(cpu_input, config.frame, config.hop);
        evidence.cpu_trace.push_back(phase("execute_sync", 3, false));
        evidence.cpu_trace.push_back(phase("d2h", 4, false));
        cpu_input.clear();
        evidence.cpu_trace.push_back(phase("release_sync", 5, false));
        evidence.cpu_trace.push_back(phase("after", 6, false));

        evidence.gpu_trace.push_back(phase("before", 0, true));
        cusignal::DeviceArray<T> device_input(input.size());
        cusignal::DeviceArray<float> device_output(
            static_cast<std::size_t>(config.frame) * frames);
        evidence.gpu_trace.push_back(phase("allocate", 1, true));
        cusignal::cuda_utils::copy_to_device(
            device_input.data(), input.data(), input.size());
        cusignal::cuda_utils::synchronize_stream();
        evidence.gpu_trace.push_back(phase("h2d", 2, true));
        cusignal::spectrogram_device(
            device_input, config.frame, config.hop, device_output);
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
        evidence.manifest = "signal:" + config.dtype + ":[" +
            std::to_string(input.size()) + "]:sha256=" + evidence.typed_digest +
            "|frame=" + std::to_string(config.frame) +
            "|hop=" + std::to_string(config.hop);
        evidence.preview0 = preview(input);
        evidence.preview1 = "frame=" + std::to_string(config.frame) +
                            ",hop=" + std::to_string(config.hop) +
                            ",frames=" + std::to_string(frames);
        evidence.output_dtype = "FP32";
        evidence.backend = ZKX_SELECTED_FFT_BACKEND_NAME;
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
           << ",\"bytes\":" << (sample.available ? std::to_string(sample.bytes) : "null")
           << '}';
}

void write_trace(std::ostream& output, const std::vector<Phase>& phases)
{
    output << '[';
    for (std::size_t index = 0; index < phases.size(); ++index) {
        if (index) output << ',';
        output << "{\"phase\":\"" << phases[index].name
               << "\",\"phase_index\":" << phases[index].index << ",\"heap\":";
        write_resource(output, phases[index].heap);
        output << ",\"rss\":";
        write_resource(output, phases[index].rss);
        output << ",\"gpu\":";
        write_resource(output, phases[index].gpu);
        output << '}';
    }
    output << ']';
}

template <class T> int dispatch(const Config& config, Evidence& evidence)
{
    evidence = run<T>(config);
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_cli(argc, argv);
        Evidence evidence;
        if (config.dtype == "FP32") dispatch<float>(config, evidence);
        else if (config.dtype == "FP16") dispatch<__half>(config, evidence);
        else if (config.dtype == "INT32") dispatch<std::int32_t>(config, evidence);
        else if (config.dtype == "INT16") dispatch<std::int16_t>(config, evidence);
        else if (config.dtype == "INT8") dispatch<std::int8_t>(config, evidence);
        else throw std::invalid_argument("dtype");

        const auto accuracy = zkx::common::compare(
            evidence.cpu_output, evidence.gpu_output, 1.0e-12);
        if (fs::exists(config.output)) throw std::runtime_error("output exists");
        std::ofstream output(config.output);
        output << "{\"operator\":\"" << kOperator
               << "\",\"requested_dtype\":\"" << config.dtype
               << "\",\"output_dtype\":\"" << evidence.output_dtype
               << "\",\"backend\":\"" << evidence.backend
               << "\",\"output_count\":" << evidence.cpu_output.size()
               << ",\"reference_count\":" << config.reference_count
               << ",\"frame\":" << config.frame
               << ",\"hop\":" << config.hop
               << ",\"typed_input_manifest\":\"" << evidence.manifest
               << "\",\"typed_array_digest\":\"" << evidence.typed_digest
               << "\",\"actual_input_digest\":\"" << evidence.input_digest
               << "\",\"preview0\":\"" << evidence.preview0
               << "\",\"preview1\":\"" << evidence.preview1
               << "\",\"cpu_output_digest\":\"" << evidence.cpu_output_digest
               << "\",\"gpu_output_digest\":\"" << evidence.gpu_output_digest
               << "\",\"mismatch_count\":" << evidence.mismatch_count
               << ",\"exact_match\":" << (evidence.mismatch_count == 0 ? "true" : "false")
               << ",\"mse\":" << accuracy.mse
               << ",\"rmse\":" << accuracy.rmse
               << ",\"relative_l2\":" << accuracy.relative_l2
               << ",\"relative_linf\":" << accuracy.relative_linf
               << ",\"cpu_samples_ms\":";
        write_samples(output, evidence.cpu_samples);
        output << ",\"gpu_samples_ms\":";
        write_samples(output, evidence.gpu_samples);
        output << ",\"cpu_trace\":";
        write_trace(output, evidence.cpu_trace);
        output << ",\"gpu_trace\":";
        write_trace(output, evidence.gpu_trace);
        output << "}\n";
        std::cout << "OPERATOR_CASE_OPERATOR_BENCHMARKEATURE_PROBE PASS operator=" << kOperator
                  << " dtype=" << config.dtype << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
