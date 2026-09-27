#include <cusignal/runtime/runtime_utils.h>
#include <cusignal/backends/fft/fft_interface.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarmup = 5;
constexpr int kIterations = 20;
constexpr int kBatches = 5;

struct Case {
    const char* name;
    int n;
    int batch;
};

void require_cuda(cudaError_t status, const char* step)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(step) + ": " + cudaGetErrorString(status));
    }
}

void require_fft(FFTResult status, const char* step)
{
    if (status != FFT_SUCCESS) {
        throw std::runtime_error(
            std::string(step) + ": FFT status " + std::to_string(static_cast<int>(status)));
    }
}

const char* backend_name()
{
#if defined(USE_DLFFT)
    return "dlfft";
#elif defined(USE_THRUST)
    return "thrust";
#else
    return "unknown";
#endif
}

double percentile(std::vector<float> values, double q)
{
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(q * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

double mean(const std::vector<float>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
}

double standard_deviation(const std::vector<float>& values, double average)
{
    double sum = 0.0;
    for (float value : values) {
        const double delta = static_cast<double>(value) - average;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size()));
}

std::vector<ComplexFloat> make_input(std::size_t count)
{
    std::vector<ComplexFloat> input(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i % 257U);
        input[i] = ComplexFloat{
            std::sin(0.03125F * x) + 0.25F * std::cos(0.0078125F * x),
            std::cos(0.0234375F * x) - 0.125F * std::sin(0.015625F * x)};
    }
    return input;
}

void run_case(const Case& test_case)
{
    const std::size_t elements =
        static_cast<std::size_t>(test_case.n) * static_cast<std::size_t>(test_case.batch);
    const std::size_t bytes = elements * sizeof(ComplexFloat);
    const auto host_input = make_input(elements);

    ComplexFloat* device_input = nullptr;
    ComplexFloat* device_output = nullptr;
    ComplexFloat* device_roundtrip = nullptr;
    FFTPlanHandle forward = nullptr;
    FFTPlanHandle inverse = nullptr;

    require_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc input");
    require_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc output");
    require_cuda(cudaMalloc(&device_roundtrip, bytes), "cudaMalloc roundtrip");
    require_cuda(cudaMemcpy(
        device_input, host_input.data(), bytes, cudaMemcpyHostToDevice), "copy input");

    const auto plan_begin = std::chrono::steady_clock::now();
    require_fft(fft_plan_create_1d(
        &forward, test_case.n, test_case.batch, FFT_TYPE_C2C, FFT_FORWARD),
        "create forward plan");
    require_fft(fft_plan_create_1d(
        &inverse, test_case.n, test_case.batch, FFT_TYPE_C2C, FFT_INVERSE),
        "create inverse plan");
    const auto plan_end = std::chrono::steady_clock::now();
    const double plan_create_ms = std::chrono::duration<double, std::milli>(
        plan_end - plan_begin).count();

    for (int i = 0; i < kWarmup; ++i) {
        require_fft(fft_execute(forward, device_output, device_input), "warmup execute");
    }
    require_cuda(cudaDeviceSynchronize(), "warmup synchronize");

    const std::size_t cpu_before = mallinfo2().uordblks;
    std::size_t gpu_free_before = 0;
    std::size_t gpu_total = 0;
    require_cuda(cudaMemGetInfo(&gpu_free_before, &gpu_total), "memory before");

    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(kIterations * kBatches));
    std::vector<float> batch_means;
    batch_means.reserve(kBatches);
    cusignal::cuda_utils::EventTimer timer;
    for (int batch_index = 0; batch_index < kBatches; ++batch_index) {
        const std::size_t first = samples.size();
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            timer.record_start();
            require_fft(fft_execute(forward, device_output, device_input), "timed execute");
            timer.record_stop();
            samples.push_back(timer.elapsed_ms());
        }
        batch_means.push_back(mean(std::vector<float>(
            samples.begin() + static_cast<std::ptrdiff_t>(first), samples.end())));
    }

    require_cuda(cudaDeviceSynchronize(), "final synchronize");
    const std::size_t cpu_after = mallinfo2().uordblks;
    std::size_t gpu_free_after = 0;
    require_cuda(cudaMemGetInfo(&gpu_free_after, &gpu_total), "memory after");

    require_fft(fft_execute(forward, device_output, device_input), "correctness forward");
    require_fft(fft_execute(inverse, device_roundtrip, device_output), "correctness inverse");
    require_cuda(cudaDeviceSynchronize(), "correctness synchronize");
    std::vector<ComplexFloat> host_roundtrip(elements);
    require_cuda(cudaMemcpy(
        host_roundtrip.data(), device_roundtrip, bytes, cudaMemcpyDeviceToHost),
        "copy roundtrip");

    double error_l2 = 0.0;
    double reference_l2 = 0.0;
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < elements; ++i) {
        const double dre = static_cast<double>(host_roundtrip[i].re) - host_input[i].re;
        const double dim = static_cast<double>(host_roundtrip[i].im) - host_input[i].im;
        const double error = std::sqrt(dre * dre + dim * dim);
        error_l2 += error * error;
        reference_l2 += static_cast<double>(host_input[i].re) * host_input[i].re +
            static_cast<double>(host_input[i].im) * host_input[i].im;
        max_abs_error = std::max(max_abs_error, error);
    }
    const double relative_l2 = std::sqrt(error_l2 / std::max(reference_l2, 1.0e-30));
    const bool correct = relative_l2 <= 5.0e-3 && max_abs_error <= 5.0e-2;

    const double average = mean(samples);
    const double stddev = standard_deviation(samples, average);
    std::sort(batch_means.begin(), batch_means.end());
    const double center_mean = batch_means[batch_means.size() / 2];
    const long long cpu_growth = static_cast<long long>(cpu_after) -
        static_cast<long long>(cpu_before);
    const long long gpu_growth = static_cast<long long>(gpu_free_before) -
        static_cast<long long>(gpu_free_after);

    std::cout << std::fixed << std::setprecision(6)
              << "[FFT-PERF] backend=" << backend_name()
              << " case=" << test_case.name
              << " n=" << test_case.n
              << " batch=" << test_case.batch
              << " elements=" << elements
              << " scope=device_resident_api"
              << " external_allocation=0 h2d=0 d2h=0 plan_create=0 postprocess=0"
              << " backend_internal_work=included"
              << " warmup=" << kWarmup
              << " iterations=" << kIterations
              << " batches=" << kBatches
              << " center_mean_ms=" << center_mean
              << " pooled_mean_ms=" << average
              << " p50_ms=" << percentile(samples, 0.50)
              << " p95_ms=" << percentile(samples, 0.95)
              << " p99_ms=" << percentile(samples, 0.99)
              << " min_ms=" << *std::min_element(samples.begin(), samples.end())
              << " max_ms=" << *std::max_element(samples.begin(), samples.end())
              << " cv=" << (average == 0.0 ? 0.0 : stddev / average)
              << " plan_create_ms=" << plan_create_ms
              << " cpu_execute_growth_bytes=" << cpu_growth
              << " gpu_execute_growth_bytes=" << gpu_growth
              << " roundtrip_relative_l2=" << relative_l2
              << " roundtrip_max_abs=" << max_abs_error
              << " correctness=" << (correct ? "pass" : "fail")
              << '\n';

    require_fft(fft_plan_destroy(inverse), "destroy inverse plan");
    require_fft(fft_plan_destroy(forward), "destroy forward plan");
    require_cuda(cudaFree(device_roundtrip), "free roundtrip");
    require_cuda(cudaFree(device_output), "free output");
    require_cuda(cudaFree(device_input), "free input");

    if (!correct) {
        throw std::runtime_error(std::string(test_case.name) + ": correctness failed");
    }
}

}  // namespace

int main()
{
    try {
        const std::vector<Case> cases{
            {"task1_doppler", 32, 256},
            {"task2_spectrogram", 64, 15},
            {"task1_ambiguity_2d", 128, 127},
            {"task1_pulse_compression", 256, 32},
            {"non_power_127", 127, 1},
            {"throughput_1024", 1024, 16},
            {"large_4096", 4096, 1},
        };
        for (const Case& test_case : cases) {
            run_case(test_case);
        }
        std::cout << "[FFT-PERF][SUMMARY] backend=" << backend_name()
                  << " cases=" << cases.size() << "/" << cases.size()
                  << " correctness=passed status=passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FFT-PERF][FAIL] " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
