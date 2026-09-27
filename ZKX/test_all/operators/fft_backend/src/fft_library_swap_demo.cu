/**
 * 同一源码动态库平替验证：只通过 Makefile 的 FFT_LIBRARY 选择 dlfft 或
 * zkx_fft_thrust，业务源码、输入、API 调用和判定逻辑完全相同。
 */
#include <cufftXt.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupRuns = 1;
constexpr int kMeasuredRuns = 1;
constexpr double kAbsoluteTolerance = 2.0e-4;
constexpr double kRelativeTolerance = 2.0e-4;
using Complex = std::complex<double>;

struct Metrics {
    double mse = 0.0;
    double rmse = 0.0;
    double relative_l2 = 0.0;
    double relative_linf = 0.0;
};

struct CaseResult {
    std::string name;
    std::string shape;
    double gpu_ms = 0.0;
    double cpu_reference_ms = 0.0;
    Metrics metrics;
    std::vector<Complex> output;
    bool pass = false;
};

bool cuda_ok(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess) return true;
    std::cerr << "[FFT_SWAP][ERROR] operation=" << operation
              << " cuda_status=" << static_cast<int>(status) << '\n';
    return false;
}

bool cufft_ok(cufftResult status, const char* operation)
{
    if (status == CUFFT_SUCCESS) return true;
    std::cerr << "[FFT_SWAP][ERROR] operation=" << operation
              << " cufft_status=" << static_cast<int>(status) << '\n';
    return false;
}

std::vector<Complex> dft_1d(
    const std::vector<Complex>& input, int length, int batch, bool inverse)
{
    std::vector<Complex> output(input.size());
    const double sign = inverse ? 1.0 : -1.0;
    const double pi = std::acos(-1.0);
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < length; ++k) {
            Complex sum(0.0, 0.0);
            for (int n = 0; n < length; ++n) {
                const double angle = sign * 2.0 * pi * k * n / length;
                sum += input[b * length + n] * Complex(std::cos(angle), std::sin(angle));
            }
            output[b * length + k] = sum;
        }
    }
    return output;
}

std::vector<Complex> dft_2d(
    const std::vector<Complex>& input, int rows, int columns, bool inverse)
{
    std::vector<Complex> output(input.size());
    const double sign = inverse ? 1.0 : -1.0;
    const double pi = std::acos(-1.0);
    for (int ky = 0; ky < rows; ++ky) {
        for (int kx = 0; kx < columns; ++kx) {
            Complex sum(0.0, 0.0);
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < columns; ++x) {
                    const double phase = static_cast<double>(ky * y) / rows
                                       + static_cast<double>(kx * x) / columns;
                    const double angle = sign * 2.0 * pi * phase;
                    sum += input[y * columns + x] * Complex(std::cos(angle), std::sin(angle));
                }
            }
            output[ky * columns + kx] = sum;
        }
    }
    return output;
}

Metrics calculate_metrics(const std::vector<Complex>& actual, const std::vector<Complex>& reference)
{
    Metrics result;
    double error_l2 = 0.0;
    double reference_l2 = 0.0;
    double error_linf = 0.0;
    double reference_linf = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - reference[i]);
        error_l2 += error * error;
        reference_l2 += std::norm(reference[i]);
        error_linf = std::max(error_linf, error);
        reference_linf = std::max(reference_linf, std::abs(reference[i]));
    }
    result.mse = actual.empty() ? std::numeric_limits<double>::infinity()
                                : error_l2 / actual.size();
    result.rmse = std::sqrt(result.mse);
    result.relative_l2 = std::sqrt(error_l2) / std::max(std::sqrt(reference_l2), 1.0e-30);
    result.relative_linf = error_linf / std::max(reference_linf, 1.0e-30);
    return result;
}

bool finite_output(const std::vector<Complex>& values)
{
    if (values.empty()) return false;
    for (const auto& value : values) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) return false;
    }
    return true;
}

std::string encode_output(const std::vector<Complex>& values)
{
    std::ostringstream stream;
    stream << std::setprecision(9);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) stream << ';';
        stream << values[i].real() << ',' << values[i].imag();
    }
    return stream.str();
}

std::vector<Complex> deterministic_complex_input(int count)
{
    std::vector<Complex> values(count);
    for (int i = 0; i < count; ++i) {
        values[i] = Complex(0.125 * (i + 1), ((i * 5) % 11 - 5) * 0.0625);
    }
    return values;
}

bool metrics_pass(const Metrics& metrics)
{
    return std::isfinite(metrics.mse) && std::isfinite(metrics.rmse)
        && std::isfinite(metrics.relative_l2) && std::isfinite(metrics.relative_linf)
        && metrics.rmse <= kAbsoluteTolerance
        && metrics.relative_l2 <= kRelativeTolerance
        && metrics.relative_linf <= kRelativeTolerance;
}

CaseResult run_c2c_1d(const std::string& name, int length, int batch, int direction, bool inplace)
{
    CaseResult result{name, "[" + std::to_string(batch) + "," + std::to_string(length) + "]"};
    const auto input = deterministic_complex_input(length * batch);
    const auto cpu_started = std::chrono::steady_clock::now();
    const auto reference = dft_1d(input, length, batch, direction == CUFFT_INVERSE);
    result.cpu_reference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_started).count();
    std::vector<cufftComplex> host_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        host_input[i].x = static_cast<float>(input[i].real());
        host_input[i].y = static_cast<float>(input[i].imag());
    }
    cufftComplex* device_input = nullptr;
    cufftComplex* device_output = nullptr;
    cufftHandle plan = 0;
    bool ok = cufft_ok(cufftPlan1d(&plan, length, CUFFT_C2C, batch), "cufftPlan1d(C2C)")
        && cuda_ok(cudaMalloc(&device_input, host_input.size() * sizeof(cufftComplex)), "cudaMalloc(input)");
    if (ok && !inplace) ok = cuda_ok(cudaMalloc(&device_output, host_input.size() * sizeof(cufftComplex)), "cudaMalloc(output)");
    if (inplace) device_output = device_input;
    for (int run = 0; ok && run < kWarmupRuns + kMeasuredRuns; ++run) {
        ok = cuda_ok(cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
        if (!ok) break;
        const auto started = std::chrono::steady_clock::now();
        ok = cufft_ok(cufftExecC2C(plan, device_input, device_output, direction), "cufftExecC2C")
            && cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (ok && run >= kWarmupRuns) result.gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }
    std::vector<cufftComplex> host_output(input.size());
    if (ok) ok = cuda_ok(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
    if (ok) {
        result.output.resize(host_output.size());
        for (std::size_t i = 0; i < host_output.size(); ++i) result.output[i] = Complex(host_output[i].x, host_output[i].y);
        result.metrics = calculate_metrics(result.output, reference);
        result.pass = finite_output(result.output) && metrics_pass(result.metrics);
    }
    if (plan) ok = cufft_ok(cufftDestroy(plan), "cufftDestroy") && ok;
    if (!inplace && device_output) cudaFree(device_output);
    if (device_input) cudaFree(device_input);
    result.pass = result.pass && ok;
    return result;
}

CaseResult run_c2c_2d(const std::string& name, int rows, int columns, int direction)
{
    CaseResult result{name, "[" + std::to_string(rows) + "," + std::to_string(columns) + "]"};
    const auto input = deterministic_complex_input(rows * columns);
    const auto cpu_started = std::chrono::steady_clock::now();
    const auto reference = dft_2d(input, rows, columns, direction == CUFFT_INVERSE);
    result.cpu_reference_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_started).count();
    std::vector<cufftComplex> host_input(input.size()), host_output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) host_input[i] = {static_cast<float>(input[i].real()), static_cast<float>(input[i].imag())};
    cufftComplex* device_input = nullptr; cufftComplex* device_output = nullptr; cufftHandle plan = 0;
    bool ok = cufft_ok(cufftPlan2d(&plan, rows, columns, CUFFT_C2C), "cufftPlan2d")
        && cuda_ok(cudaMalloc(&device_input, host_input.size() * sizeof(cufftComplex)), "cudaMalloc(input)")
        && cuda_ok(cudaMalloc(&device_output, host_output.size() * sizeof(cufftComplex)), "cudaMalloc(output)");
    for (int run = 0; ok && run < kWarmupRuns + kMeasuredRuns; ++run) {
        ok = cuda_ok(cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
        const auto started = std::chrono::steady_clock::now();
        if (ok) ok = cufft_ok(cufftExecC2C(plan, device_input, device_output, direction), "cufftExecC2C(2D)") && cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (ok && run >= kWarmupRuns) result.gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }
    if (ok) ok = cuda_ok(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
    if (ok) {
        result.output.resize(host_output.size());
        for (std::size_t i = 0; i < host_output.size(); ++i) result.output[i] = Complex(host_output[i].x, host_output[i].y);
        result.metrics = calculate_metrics(result.output, reference); result.pass = finite_output(result.output) && metrics_pass(result.metrics);
    }
    if (plan) ok = cufft_ok(cufftDestroy(plan), "cufftDestroy") && ok;
    if (device_output) cudaFree(device_output); if (device_input) cudaFree(device_input);
    result.pass = result.pass && ok; return result;
}

CaseResult run_r2c(int length, int batch)
{
    CaseResult result{"r2c_1d_batch", "[" + std::to_string(batch) + "," + std::to_string(length) + "]->[" + std::to_string(batch) + "," + std::to_string(length) + "]"};
    std::vector<float> input(length * batch); for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(0.25 * std::sin(0.3 * i) + 0.0625 * i);
    std::vector<Complex> complex_input(input.size()); for (std::size_t i = 0; i < input.size(); ++i) complex_input[i] = Complex(input[i], 0.0);
    const auto cpu_started = std::chrono::steady_clock::now(); const auto reference = dft_1d(complex_input, length, batch, false);
    result.cpu_reference_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_started).count();
    std::vector<cufftComplex> host_output(reference.size()); cufftReal* device_input = nullptr; cufftComplex* device_output = nullptr; cufftHandle plan = 0;
    bool ok = cufft_ok(cufftPlan1d(&plan, length, CUFFT_R2C, batch), "cufftPlan1d(R2C)")
        && cuda_ok(cudaMalloc(&device_input, input.size() * sizeof(float)), "cudaMalloc(input)")
        && cuda_ok(cudaMalloc(&device_output, host_output.size() * sizeof(cufftComplex)), "cudaMalloc(output)");
    for (int run = 0; ok && run < kWarmupRuns + kMeasuredRuns; ++run) {
        ok = cuda_ok(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)"); const auto started = std::chrono::steady_clock::now();
        if (ok) ok = cufft_ok(cufftExecR2C(plan, device_input, device_output), "cufftExecR2C") && cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (ok && run >= kWarmupRuns) result.gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }
    if (ok) ok = cuda_ok(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
    if (ok) { result.output.resize(host_output.size()); for (std::size_t i = 0; i < host_output.size(); ++i) result.output[i] = Complex(host_output[i].x, host_output[i].y); result.metrics = calculate_metrics(result.output, reference); result.pass = finite_output(result.output) && metrics_pass(result.metrics); }
    if (plan) ok = cufft_ok(cufftDestroy(plan), "cufftDestroy") && ok; if (device_output) cudaFree(device_output); if (device_input) cudaFree(device_input); result.pass = result.pass && ok; return result;
}

CaseResult run_c2r(int length, int batch)
{
    CaseResult result{"c2r_1d_batch", "[" + std::to_string(batch) + "," + std::to_string(length) + "]->[" + std::to_string(batch) + "," + std::to_string(length) + "]"};
    std::vector<Complex> full_input; std::vector<Complex> reference;
    const auto cpu_started = std::chrono::steady_clock::now();
    for (int b = 0; b < batch; ++b) {
        std::vector<Complex> full(length); full[0] = Complex(1.0 + b, 0.0); full[length / 2] = Complex(0.5, 0.0);
        for (int k = 1; k < length / 2; ++k) { full[k] = Complex(0.125 * (b + k), -0.0625 * k); full[length - k] = std::conj(full[k]); }
        full_input.insert(full_input.end(), full.begin(), full.end()); const auto transformed = dft_1d(full, length, 1, true); reference.insert(reference.end(), transformed.begin(), transformed.end());
    }
    result.cpu_reference_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_started).count();
    std::vector<cufftComplex> host_input(full_input.size()); for (std::size_t i = 0; i < full_input.size(); ++i) host_input[i] = {static_cast<float>(full_input[i].real()), static_cast<float>(full_input[i].imag())};
    std::vector<float> host_output(reference.size()); cufftComplex* device_input = nullptr; cufftReal* device_output = nullptr; cufftHandle plan = 0;
    bool ok = cufft_ok(cufftPlan1d(&plan, length, CUFFT_C2R, batch), "cufftPlan1d(C2R)")
        && cuda_ok(cudaMalloc(&device_input, host_input.size() * sizeof(cufftComplex)), "cudaMalloc(input)")
        && cuda_ok(cudaMalloc(&device_output, host_output.size() * sizeof(float)), "cudaMalloc(output)");
    for (int run = 0; ok && run < kWarmupRuns + kMeasuredRuns; ++run) {
        ok = cuda_ok(cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)"); const auto started = std::chrono::steady_clock::now();
        if (ok) ok = cufft_ok(cufftExecC2R(plan, device_input, device_output), "cufftExecC2R") && cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (ok && run >= kWarmupRuns) result.gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }
    if (ok) ok = cuda_ok(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
    if (ok) { result.output.resize(host_output.size()); for (std::size_t i = 0; i < host_output.size(); ++i) result.output[i] = Complex(host_output[i], 0.0); result.metrics = calculate_metrics(result.output, reference); result.pass = finite_output(result.output) && metrics_pass(result.metrics); }
    if (plan) ok = cufft_ok(cufftDestroy(plan), "cufftDestroy") && ok; if (device_output) cudaFree(device_output); if (device_input) cudaFree(device_input); result.pass = result.pass && ok; return result;
}

void print_case(const CaseResult& result)
{
    std::cout << std::setprecision(12) << "[FFT_SWAP][CASE]"
              << "|name=" << result.name << "|shape=" << result.shape
              << "|warmup_runs=" << kWarmupRuns << "|measured_runs=" << kMeasuredRuns
              << "|cpu_reference_ms=" << result.cpu_reference_ms << "|gpu_ms=" << result.gpu_ms
              << "|mse=" << result.metrics.mse << "|rmse=" << result.metrics.rmse
              << "|relative_l2=" << result.metrics.relative_l2
              << "|relative_linf=" << result.metrics.relative_linf
              << "|status=" << (result.pass ? "PASS" : "FAIL")
              << "|output=" << encode_output(result.output) << '\n';
}

}  // namespace

int main()
{
    Dl_info information{};
    const bool path_ok = dladdr(reinterpret_cast<void*>(&cufftPlan1d), &information) != 0
        && information.dli_fname != nullptr;
    const std::string library_path = path_ok ? information.dli_fname : "UNKNOWN";
    const std::string backend_identity = library_path.find("libzkx_fft_thrust.so") != std::string::npos
        ? "FFT_THRUST" : (library_path.find("libdlfft.so") != std::string::npos ? "DLFFT" : "UNKNOWN");
    int version = 0;
    const cufftResult version_status = cufftGetVersion(&version);
    std::cout << "[FFT_SWAP][IDENTITY]|backend_identity=" << backend_identity
              << "|library_path=" << library_path
              << "|version_status=" << (version_status == CUFFT_SUCCESS ? "CUFFT_SUCCESS" : "ERROR")
              << "|version=" << version << "|source_contract=identical_source_only_library_name_changes\n";

    std::vector<CaseResult> cases;
    cases.push_back(run_c2c_1d("c2c_1d_forward_batch", 8, 2, CUFFT_FORWARD, false));
    cases.push_back(run_c2c_1d("c2c_1d_inverse", 8, 1, CUFFT_INVERSE, false));
    cases.push_back(run_c2c_1d("c2c_1d_inplace", 8, 1, CUFFT_FORWARD, true));
    cases.push_back(run_r2c(8, 2));
    cases.push_back(run_c2r(8, 2));
    cases.push_back(run_c2c_2d("c2c_2d_forward", 4, 4, CUFFT_FORWARD));
    cases.push_back(run_c2c_2d("c2c_2d_inverse", 4, 4, CUFFT_INVERSE));
    bool pass = path_ok && backend_identity != "UNKNOWN"
        && version_status == CUFFT_SUCCESS && version > 0;
    for (const auto& item : cases) { print_case(item); pass = pass && item.pass; }
    std::cout << "[FFT_SWAP][SUMMARY]|cases=" << cases.size()
              << "|status=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}
