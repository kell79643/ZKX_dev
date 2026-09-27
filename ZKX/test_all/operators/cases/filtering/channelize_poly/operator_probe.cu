#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/filtering/filtering_typed.h>
#include "memory.h"
#include "sha256.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef ZKX_REMAINING_OPERATOR_CASE_CHANNELIZE_FFT_BACKEND
#error "channelize_poly probe requires an explicitly selected FFT backend"
#endif
#ifndef ZKX_SELECTED_FFT_BACKEND_NAME
#error "ZKX_SELECTED_FFT_BACKEND_NAME is required"
#endif

namespace {
namespace fs = std::filesystem;

struct Cli {
    std::string dtype;
    std::size_t x_count{};
    std::size_t h_count{};
    int n_chans{};
    std::uint64_t seed{};
    std::size_t warmup{};
    std::size_t measured{};
    fs::path output;
};

struct Phase {
    std::string name;
    int index{};
    zkx::common::ResourceSample heap;
    zkx::common::ResourceSample rss;
    zkx::common::ResourceSample gpu;
};

std::string json_escape(const std::string& value)
{
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
                              << std::setfill('0') << static_cast<int>(c);
            else out << static_cast<char>(c);
        }
    }
    return out.str();
}

Cli parse_cli(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0)
            throw std::invalid_argument("expected --key value pairs");
        values.emplace(std::string(argv[i]).substr(2), argv[i + 1]);
    }
    const auto get = [&](const char* key) -> const std::string& {
        const auto it = values.find(key);
        if (it == values.end()) throw std::invalid_argument(std::string("missing --") + key);
        return it->second;
    };
    Cli cli;
    cli.dtype = get("dtype");
    cli.x_count = std::stoull(get("x-count"));
    cli.h_count = std::stoull(get("h-count"));
    cli.n_chans = std::stoi(get("n-chans"));
    cli.seed = std::stoull(get("seed"));
    cli.warmup = std::stoull(get("warmup"));
    cli.measured = std::stoull(get("measured"));
    cli.output = get("output");
    if (values.size() != 8 || cli.x_count == 0 || cli.h_count == 0 ||
        cli.n_chans <= 0 || cli.x_count % static_cast<std::size_t>(cli.n_chans) != 0 ||
        cli.h_count % static_cast<std::size_t>(cli.n_chans) != 0 ||
        cli.h_count / static_cast<std::size_t>(cli.n_chans) > 32 ||
        cli.warmup == 0 || cli.measured == 0)
        throw std::invalid_argument("invalid channelize_poly probe arguments");
    return cli;
}

zkx::common::ResourceSample gpu_sample()
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
            gpu ? gpu_sample() : zkx::common::ResourceSample{false, 0, "NA", "CPU path"}};
}

template <class T> T typed(float value) { return static_cast<T>(value); }
template <> __half typed<__half>(float value) { return __float2half_rn(value); }

template <class T>
std::vector<T> make_input(std::size_t count, std::uint64_t seed, bool taps)
{
    std::vector<T> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int raw = taps
            ? static_cast<int>((i * 7 + seed * 3) % 9) - 4
            : static_cast<int>((i * 17 + seed) % 23) - 11;
        const float value = std::is_integral_v<T>
            ? static_cast<float>(raw)
            : static_cast<float>(raw) * (taps ? 0.125f : 0.25f);
        out.push_back(typed<T>(value));
    }
    return out;
}

template <class T>
std::string raw_digest(const std::vector<T>& values)
{
    return zkx::remaining_operator_case::sha256_hex(std::string(
        reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T)));
}

std::vector<std::complex<double>> complex_values(
    const std::vector<ComplexFloat>& values)
{
    std::vector<std::complex<double>> out;
    out.reserve(values.size());
    for (const auto& value : values) out.emplace_back(value.re, value.im);
    return out;
}

template <class Callable>
std::vector<double> measure(std::size_t warmup, std::size_t measured, Callable&& call)
{
    for (std::size_t i = 0; i < warmup; ++i) call();
    std::vector<double> samples;
    samples.reserve(measured);
    for (std::size_t i = 0; i < measured; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        call();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return samples;
}

void write_samples(std::ostream& out, const std::vector<double>& values)
{
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << std::setprecision(17) << values[i];
    }
    out << ']';
}

void write_sample(std::ostream& out, const zkx::common::ResourceSample& sample)
{
    out << "{\"available\":" << (sample.available ? "true" : "false")
        << ",\"bytes\":" << (sample.available ? std::to_string(sample.bytes) : "null")
        << ",\"source\":\"" << json_escape(sample.source)
        << "\",\"error\":\"" << json_escape(sample.error) << "\"}";
}

void write_phases(std::ostream& out, const std::vector<Phase>& phases)
{
    out << '[';
    for (std::size_t i = 0; i < phases.size(); ++i) {
        if (i) out << ',';
        out << "{\"phase\":\"" << phases[i].name << "\",\"phase_index\":"
            << phases[i].index << ",\"heap\":";
        write_sample(out, phases[i].heap);
        out << ",\"rss\":";
        write_sample(out, phases[i].rss);
        out << ",\"gpu\":";
        write_sample(out, phases[i].gpu);
        out << '}';
    }
    out << ']';
}

template <class T>
int execute(const Cli& cli)
{
    const std::vector<T> x = make_input<T>(cli.x_count, cli.seed, false);
    const std::vector<T> h = make_input<T>(cli.h_count, cli.seed + 1, true);
    const std::string x_digest = raw_digest(x);
    const std::string h_digest = raw_digest(h);
    std::ostringstream manifest;
    manifest << "x:" << cli.dtype << ":[" << x.size() << "]:bytes="
             << x.size() * sizeof(T) << ":sha256=" << x_digest
             << "|h:" << cli.dtype << ":[" << h.size() << "]:bytes="
             << h.size() * sizeof(T) << ":sha256=" << h_digest;
    const std::string input_digest = zkx::remaining_operator_case::sha256_hex(manifest.str());

    cusignal::ChannelizePolyCpuResult cpu_result;
    cusignal::ChannelizePolyDeviceResult gpu_result;
    std::vector<ComplexFloat> gpu_host;
    const auto cpu_samples = measure(cli.warmup, cli.measured, [&] {
        cpu_result = cusignal::channelize_poly_typed_cpu(x, h, cli.n_chans);
    });
    const auto gpu_samples = measure(cli.warmup, cli.measured, [&] {
        auto dx = cusignal::DeviceArray<T>::from_host(x);
        auto dh = cusignal::DeviceArray<T>::from_host(h);
        cusignal::channelize_poly_device(dx, dh, cli.n_chans, gpu_result);
        gpu_host = gpu_result.values.to_host();
    });

    std::vector<Phase> cpu_trace;
    cpu_trace.push_back(phase("before", 0, false));
    std::vector<T> cx = x, ch = h;
    cpu_trace.push_back(phase("allocate", 1, false));
    cpu_trace.push_back(phase("h2d", 2, false));
    auto traced_cpu = cusignal::channelize_poly_typed_cpu(cx, ch, cli.n_chans);
    cpu_trace.push_back(phase("execute_sync", 3, false));
    cpu_trace.push_back(phase("d2h", 4, false));
    traced_cpu.values.clear(); cx.clear(); ch.clear();
    cpu_trace.push_back(phase("release_sync", 5, false));
    cpu_trace.push_back(phase("after", 6, false));

    std::vector<Phase> gpu_trace;
    cusignal::cuda_utils::synchronize_stream();
    gpu_trace.push_back(phase("before", 0, true));
    std::optional<cusignal::DeviceArray<T>> dx, dh;
    cusignal::ChannelizePolyDeviceResult traced_gpu;
    gpu_trace.push_back(phase("allocate", 1, true));
    dx.emplace(cusignal::DeviceArray<T>::from_host(x));
    dh.emplace(cusignal::DeviceArray<T>::from_host(h));
    gpu_trace.push_back(phase("h2d", 2, true));
    cusignal::channelize_poly_device(*dx, *dh, cli.n_chans, traced_gpu);
    cusignal::cuda_utils::synchronize_stream();
    gpu_trace.push_back(phase("execute_sync", 3, true));
    auto traced_host = traced_gpu.values.to_host();
    gpu_trace.push_back(phase("d2h", 4, true));
    traced_host.clear(); traced_gpu.values.reset(); dx.reset(); dh.reset();
    cusignal::cuda_utils::synchronize_stream();
    gpu_trace.push_back(phase("release_sync", 5, true));
    gpu_trace.push_back(phase("after", 6, true));

    if (cpu_result.shape != gpu_result.shape ||
        cpu_result.shape != std::vector<int>{cli.n_chans,
            static_cast<int>(cli.x_count / static_cast<std::size_t>(cli.n_chans))})
        throw std::runtime_error("channelize_poly output shape mismatch");
    const auto accuracy = zkx::common::compare(
        complex_values(cpu_result.values), complex_values(gpu_host), 1e-12);
    const std::string cpu_output_digest = raw_digest(cpu_result.values);
    const std::string gpu_output_digest = raw_digest(gpu_host);

    if (fs::exists(cli.output)) throw std::runtime_error("raw output already exists");
    std::ofstream out(cli.output, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create raw output");
    out << "{\n\"schema_version\":1,\"operator_name\":\"channelize_poly\""
        << ",\"backend\":\"" << ZKX_SELECTED_FFT_BACKEND_NAME
        << "\",\"requested_dtype\":\"" << cli.dtype
        << "\",\"compute_dtype\":\"FP32\",\"fft_dtype\":\"ComplexFP32\""
        << ",\"output_dtype\":\"ComplexFP32\",\"typed_input_count\":2"
        << ",\"typed_input_manifest\":\"" << manifest.str()
        << "\",\"actual_input_digest\":\"" << input_digest << "\""
        << ",\"cpu_output_digest\":\"" << cpu_output_digest
        << "\",\"gpu_output_digest\":\"" << gpu_output_digest << "\""
        << ",\"output_shape\":[" << gpu_result.shape[0] << ',' << gpu_result.shape[1] << ']'
        << ",\"mse\":" << std::setprecision(17) << accuracy.mse
        << ",\"rmse\":" << accuracy.rmse
        << ",\"relative_l2\":" << accuracy.relative_l2
        << ",\"relative_linf\":" << accuracy.relative_linf
        << ",\"cpu_samples_ms\":";
    write_samples(out, cpu_samples);
    out << ",\"gpu_samples_ms\":";
    write_samples(out, gpu_samples);
    out << ",\"cpu_trace\":";
    write_phases(out, cpu_trace);
    out << ",\"gpu_trace\":";
    write_phases(out, gpu_trace);
    out << "\n}\n";
    if (!out) throw std::runtime_error("cannot write raw output");
    std::cout << "REMAINING_OPERATOR_CASE_CHANNELIZE_PROBE PASS dtype=" << cli.dtype
              << " x=" << cli.x_count << " h=" << cli.h_count
              << " n_chans=" << cli.n_chans
              << " backend=" << ZKX_SELECTED_FFT_BACKEND_NAME << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Cli cli = parse_cli(argc, argv);
        if (cli.dtype == "FP32") return execute<float>(cli);
        if (cli.dtype == "FP16") return execute<__half>(cli);
        if (cli.dtype == "INT32") return execute<std::int32_t>(cli);
        if (cli.dtype == "INT16") return execute<std::int16_t>(cli);
        if (cli.dtype == "INT8") return execute<std::int8_t>(cli);
        throw std::invalid_argument("unsupported dtype");
    } catch (const std::exception& error) {
        std::cerr << "REMAINING_OPERATOR_CASE_CHANNELIZE_PROBE FAIL error=" << error.what() << '\n';
        return 2;
    }
}
