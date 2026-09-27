/**
 * @file fft_interface.cpp
 * @brief FFT 接口 C++ 封装类实现
 *
 * 实现 FFTInterface 和 FFTInterface2D 类，提供便捷的 C++ 接口。
 * 内部调用 C 接口的 FFT 函数。
 *
 * @author cusignal_cpp
 * @date 2024-2026
 */

#include <cusignal/backends/fft/fft_interface.h>
#include <cusignal/runtime/cuda_error.h>
#include <cusignal/runtime/device_allocation_tracker.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cusignal {

constexpr double PI = 3.14159265358979323846;

namespace {

struct BatchFFTCache {
    int n = 0;
    int batch = 0;
    FFTDirection direction = FFT_FORWARD;
    FFTPlanHandle plan = nullptr;
    void* d_input = nullptr;
    void* d_output = nullptr;

    ~BatchFFTCache() { reset(); }

    void reset() {
        if (plan) {
            fft_plan_destroy(plan);
            plan = nullptr;
        }
        if (d_input) {
            cuda_utils::tracked_cuda_free(d_input);
            d_input = nullptr;
        }
        if (d_output) {
            cuda_utils::tracked_cuda_free(d_output);
            d_output = nullptr;
        }
        n = 0;
        batch = 0;
        direction = FFT_FORWARD;
    }

    void ensure(int new_n, int new_batch, FFTDirection new_direction) {
        if (plan && d_input && d_output && n == new_n && batch == new_batch && direction == new_direction) {
            return;
        }
        reset();
        try {
            n = new_n;
            batch = new_batch;
            direction = new_direction;
            if (fft_plan_create_1d(&plan, n, batch, FFT_TYPE_C2C, direction) != FFT_SUCCESS) {
                throw std::runtime_error("BatchFFTCache: fft_plan_create_1d failed");
            }
            CUDA_CHECK(cuda_utils::tracked_cuda_malloc(
                &d_input, static_cast<std::size_t>(n) * batch * sizeof(ComplexFloat)));
            CUDA_CHECK(cuda_utils::tracked_cuda_malloc(
                &d_output, static_cast<std::size_t>(n) * batch * sizeof(ComplexFloat)));
        } catch (...) {
            reset();
            throw;
        }
    }
};

struct BatchFFTPlanCache {
    int n = 0;
    int batch = 0;
    FFTDirection direction = FFT_FORWARD;
    FFTPlanHandle plan = nullptr;

    ~BatchFFTPlanCache() { reset(); }

    void reset() {
        if (plan) {
            fft_plan_destroy(plan);
            plan = nullptr;
        }
        n = 0;
        batch = 0;
        direction = FFT_FORWARD;
    }

    void ensure(int new_n, int new_batch, FFTDirection new_direction) {
        if (plan && n == new_n && batch == new_batch && direction == new_direction) {
            return;
        }
        reset();
        n = new_n;
        batch = new_batch;
        direction = new_direction;
        FFTResult result = fft_plan_create_1d(&plan, n, batch, FFT_TYPE_C2C, direction);
        if (result != FFT_SUCCESS) {
            throw std::runtime_error("BatchFFTPlanCache: fft_plan_create_1d failed");
        }
    }
};

struct BatchFFTPlanPool {
    static constexpr std::size_t capacity = 8;
    std::array<BatchFFTPlanCache, capacity> entries{};
    std::size_t size = 0;
    std::size_t next_replacement = 0;

    FFTPlanHandle get(int n, int batch, FFTDirection direction) {
        for (std::size_t i = 0; i < size; ++i) {
            auto& entry = entries[i];
            if (entry.plan && entry.n == n && entry.batch == batch &&
                entry.direction == direction) {
                return entry.plan;
            }
        }
        std::size_t index = 0;
        if (size < capacity) {
            index = size++;
        } else {
            index = next_replacement;
            next_replacement = (next_replacement + 1) % capacity;
        }
        entries[index].ensure(n, batch, direction);
        return entries[index].plan;
    }
};

struct FFT2DPlanCache {
    int nx = 0;
    int ny = 0;
    FFTDirection direction = FFT_FORWARD;
    FFTPlanHandle plan = nullptr;

    ~FFT2DPlanCache() {
        if (plan) fft_plan_destroy(plan);
    }

    void ensure(int new_nx, int new_ny, FFTDirection new_direction) {
        if (plan && nx == new_nx && ny == new_ny && direction == new_direction) return;
        if (plan) {
            fft_plan_destroy(plan);
            plan = nullptr;
        }
        nx = new_nx;
        ny = new_ny;
        direction = new_direction;
        if (fft_plan_create_2d(&plan, ny, nx, FFT_TYPE_C2C, direction) != FFT_SUCCESS) {
            throw std::runtime_error("FFT2DPlanCache: fft_plan_create_2d failed");
        }
    }
};

struct FFT2DPlanPool {
    static constexpr std::size_t capacity = 8;
    std::array<FFT2DPlanCache, capacity> entries{};
    std::size_t size = 0;
    std::size_t next_replacement = 0;

    FFTPlanHandle get(int nx, int ny, FFTDirection direction) {
        for (std::size_t i = 0; i < size; ++i) {
            auto& entry = entries[i];
            if (entry.plan && entry.nx == nx && entry.ny == ny &&
                entry.direction == direction) return entry.plan;
        }
        std::size_t index = 0;
        if (size < capacity) {
            index = size++;
        } else {
            index = next_replacement;
            next_replacement = (next_replacement + 1) % capacity;
        }
        entries[index].ensure(nx, ny, direction);
        return entries[index].plan;
    }
};

} // namespace

/**
 * @brief 第一类修正 Bessel 函数 I0
 *
 * 用于 Kaiser 窗计算。
 * 使用多项式逼近，在 |x| < 3.75 时使用小值近似，否则使用大值近似。
 */
static double _bessel_i0(double x) {
    double ax = std::abs(x);
    if (ax < 3.75) {
        double y = x / 3.75;
        y = y * y;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    } else {
        double y = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + y * (0.01328592
                + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
                + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
    }
}

/**
 * @brief Sinc 函数
 *
 * sinc(x) = sin(πx) / (πx)
 * 用于 FIR 滤波器设计。
 */
static double sinc(double x) {
    if (std::abs(x) < 1e-10) return 1.0;
    return std::sin(PI * x) / (PI * x);
}

/**
 * @brief FFTInterface 构造函数
 *
 * 创建 1D FFT 接口对象，分配 GPU 内存并创建 FFT 计划。
 */
FFTInterface::FFTInterface(int n)
    : FFTInterface(n, false)
{
}

FFTInterface::FFTInterface(int n, BatchOnly)
    : FFTInterface(n, true)
{
}

FFTInterface::FFTInterface(int n, bool batch_only) : n_(n), plan_forward_(nullptr), plan_inverse_(nullptr),
    plan_r2c_(nullptr), plan_c2r_(nullptr), d_input_(nullptr), d_output_(nullptr) {
    if (n_ <= 0) {
        throw std::invalid_argument("FFTInterface: length must be positive");
    }
    // Batch API 由固定容量的 thread_local plan pool 按 n/batch/direction 复用 plan，
    // 不需要再为临时接口对象创建两套单 FFT plan 和输入/输出缓冲。
    if (batch_only) {
        return;
    }
    try {
        if (fft_plan_create_1d(&plan_forward_, n_, 1, FFT_TYPE_C2C, FFT_FORWARD) != FFT_SUCCESS) {
            throw std::runtime_error("FFTInterface: forward plan creation failed");
        }
        if (fft_plan_create_1d(&plan_inverse_, n_, 1, FFT_TYPE_C2C, FFT_INVERSE) != FFT_SUCCESS) {
            throw std::runtime_error("FFTInterface: inverse plan creation failed");
        }
        CUDA_CHECK(cuda_utils::tracked_cuda_malloc(&d_input_, static_cast<std::size_t>(n_) * sizeof(ComplexFloat)));
        CUDA_CHECK(cuda_utils::tracked_cuda_malloc(&d_output_, static_cast<std::size_t>(n_) * sizeof(ComplexFloat)));
    } catch (...) {
        if (plan_forward_) fft_plan_destroy(plan_forward_);
        if (plan_inverse_) fft_plan_destroy(plan_inverse_);
        if (d_input_) cuda_utils::tracked_cuda_free(d_input_);
        if (d_output_) cuda_utils::tracked_cuda_free(d_output_);
        plan_forward_ = nullptr;
        plan_inverse_ = nullptr;
        d_input_ = nullptr;
        d_output_ = nullptr;
        throw;
    }
}

/**
 * @brief FFTInterface 析构函数
 *
 * 释放 FFT 计划和 GPU 内存。
 */
FFTInterface::~FFTInterface() noexcept {
    if (plan_forward_) fft_plan_destroy(plan_forward_);
    if (plan_inverse_) fft_plan_destroy(plan_inverse_);
    if (plan_r2c_) fft_plan_destroy(plan_r2c_);
    if (plan_c2r_) fft_plan_destroy(plan_c2r_);
    if (d_input_) cuda_utils::tracked_cuda_free(d_input_);
    if (d_output_) cuda_utils::tracked_cuda_free(d_output_);
}

/**
 * @brief 执行正向 FFT
 *
 * 将输入数据拷贝到 GPU，执行 FFT，将结果拷贝回主机。
 */
std::vector<complexd> FFTInterface::fft(const std::vector<complexd>& input) {
    std::vector<ComplexFloat> host_input(n_);
    for (int i = 0; i < n_ && i < (int)input.size(); i++) {
        host_input[i].re = static_cast<float>(input[i].real());
        host_input[i].im = static_cast<float>(input[i].imag());
    }
    cudaMemcpy(d_input_, host_input.data(), n_ * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(plan_forward_, d_output_, d_input_);
    std::vector<ComplexFloat> host_output(n_);
    cudaMemcpy(host_output.data(), d_output_, n_ * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);
    std::vector<complexd> result(n_);
    for (int i = 0; i < n_; i++) {
        result[i] = complexd(host_output[i].re, host_output[i].im);
    }
    return result;
}

/**
 * @brief 执行逆向 FFT
 *
 * 将输入数据拷贝到 GPU，执行 IFFT，将结果拷贝回主机。
 */
std::vector<complexd> FFTInterface::ifft(const std::vector<complexd>& input) {
    std::vector<ComplexFloat> host_input(n_);
    for (int i = 0; i < n_ && i < (int)input.size(); i++) {
        host_input[i].re = static_cast<float>(input[i].real());
        host_input[i].im = static_cast<float>(input[i].imag());
    }
    cudaMemcpy(d_input_, host_input.data(), n_ * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(plan_inverse_, d_output_, d_input_);
    std::vector<ComplexFloat> host_output(n_);
    cudaMemcpy(host_output.data(), d_output_, n_ * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);
    std::vector<complexd> result(n_);
    for (int i = 0; i < n_; i++) {
        result[i] = complexd(host_output[i].re, host_output[i].im);
    }
    return result;
}

void FFTInterface::fft_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output) {
    if (input.size() != static_cast<std::size_t>(n_) || output.size() != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("FFTInterface::fft_device: input/output sizes must match FFT length");
    }
    FFTResult result = fft_execute(plan_forward_, output.data(), input.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::fft_device: fft_execute failed");
    }
}

void FFTInterface::fft_device(const ComplexFloat* input, ComplexFloat* output, std::size_t count) {
    if (count != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("FFTInterface::fft_device: input/output sizes must match FFT length");
    }
    FFTResult result = fft_execute(plan_forward_, output, input);
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::fft_device: fft_execute failed");
    }
}

void FFTInterface::ifft_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output) {
    if (input.size() != static_cast<std::size_t>(n_) || output.size() != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("FFTInterface::ifft_device: input/output sizes must match FFT length");
    }
    FFTResult result = fft_execute(plan_inverse_, output.data(), input.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::ifft_device: fft_execute failed");
    }
}

void FFTInterface::ifft_device(const ComplexFloat* input, ComplexFloat* output, std::size_t count) {
    if (count != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("FFTInterface::ifft_device: input/output sizes must match FFT length");
    }
    FFTResult result = fft_execute(plan_inverse_, output, input);
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::ifft_device: fft_execute failed");
    }
}

void FFTInterface::fft_inplace_device(DeviceArray<ComplexFloat>& data) {
    if (data.size() != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("FFTInterface::fft_inplace_device: data size must match FFT length");
    }
    FFTResult result = fft_execute_inplace(plan_forward_, data.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::fft_inplace_device: fft_execute_inplace failed");
    }
}

void FFTInterface::ifft_inplace_device(DeviceArray<ComplexFloat>& data) {
    if (data.size() != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("FFTInterface::ifft_inplace_device: data size must match FFT length");
    }
    FFTResult result = fft_execute_inplace(plan_inverse_, data.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::ifft_inplace_device: fft_execute_inplace failed");
    }
}

void FFTInterface::fft_batch_device(
    const DeviceArray<ComplexFloat>& input,
    DeviceArray<ComplexFloat>& output,
    int batch)
{
    if (batch <= 0) {
        throw std::invalid_argument("FFTInterface::fft_batch_device: batch must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(n_) * static_cast<std::size_t>(batch);
    if (input.size() != expected || output.size() != expected) {
        throw std::invalid_argument("FFTInterface::fft_batch_device: input/output sizes must match n*batch");
    }

    static thread_local BatchFFTPlanPool cache;
    const FFTPlanHandle plan = cache.get(n_, batch, FFT_FORWARD);
    FFTResult result = fft_execute(plan, output.data(), input.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::fft_batch_device: fft_execute failed");
    }
}

void FFTInterface::fft_batch_device(
    const ComplexFloat* input,
    ComplexFloat* output,
    int batch,
    std::size_t count)
{
    if (batch <= 0) {
        throw std::invalid_argument("FFTInterface::fft_batch_device: batch must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(n_) * static_cast<std::size_t>(batch);
    if (count != expected) {
        throw std::invalid_argument("FFTInterface::fft_batch_device: input/output sizes must match n*batch");
    }

    static thread_local BatchFFTPlanPool cache;
    const FFTPlanHandle plan = cache.get(n_, batch, FFT_FORWARD);
    FFTResult result = fft_execute(plan, output, input);
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::fft_batch_device: fft_execute failed");
    }
}

void FFTInterface::ifft_batch_device(
    const DeviceArray<ComplexFloat>& input,
    DeviceArray<ComplexFloat>& output,
    int batch)
{
    if (batch <= 0) {
        throw std::invalid_argument("FFTInterface::ifft_batch_device: batch must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(n_) * static_cast<std::size_t>(batch);
    if (input.size() != expected || output.size() != expected) {
        throw std::invalid_argument("FFTInterface::ifft_batch_device: input/output sizes must match n*batch");
    }

    static thread_local BatchFFTPlanPool cache;
    const FFTPlanHandle plan = cache.get(n_, batch, FFT_INVERSE);
    FFTResult result = fft_execute(plan, output.data(), input.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::ifft_batch_device: fft_execute failed");
    }
}

void FFTInterface::ifft_batch_device(
    const ComplexFloat* input,
    ComplexFloat* output,
    int batch,
    std::size_t count)
{
    if (batch <= 0) {
        throw std::invalid_argument("FFTInterface::ifft_batch_device: batch must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(n_) * static_cast<std::size_t>(batch);
    if (count != expected) {
        throw std::invalid_argument("FFTInterface::ifft_batch_device: input/output sizes must match n*batch");
    }

    static thread_local BatchFFTPlanPool cache;
    const FFTPlanHandle plan = cache.get(n_, batch, FFT_INVERSE);
    FFTResult result = fft_execute(plan, output, input);
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface::ifft_batch_device: fft_execute failed");
    }
}

std::vector<std::vector<complexd>> FFTInterface::fft_batch(const std::vector<std::vector<complexd>>& input) {
    int batch = static_cast<int>(input.size());
    if (batch == 0) {
        return {};
    }

    static thread_local BatchFFTCache cache;
    cache.ensure(n_, batch, FFT_FORWARD);

    std::vector<ComplexFloat> host_input(static_cast<std::size_t>(n_) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n_; ++i) {
            std::size_t idx = static_cast<std::size_t>(b) * n_ + i;
            if (i < static_cast<int>(input[b].size())) {
                host_input[idx].re = static_cast<float>(input[b][i].real());
                host_input[idx].im = static_cast<float>(input[b][i].imag());
            } else {
                host_input[idx].re = 0.0f;
                host_input[idx].im = 0.0f;
            }
        }
    }

    cudaMemcpy(cache.d_input, host_input.data(), host_input.size() * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(cache.plan, cache.d_output, cache.d_input);

    std::vector<ComplexFloat> host_output(static_cast<std::size_t>(n_) * batch);
    cudaMemcpy(host_output.data(), cache.d_output, host_output.size() * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);

    std::vector<std::vector<complexd>> result(batch, std::vector<complexd>(n_));
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n_; ++i) {
            std::size_t idx = static_cast<std::size_t>(b) * n_ + i;
            result[b][i] = complexd(host_output[idx].re, host_output[idx].im);
        }
    }

    return result;
}

std::vector<std::vector<complexd>> FFTInterface::ifft_batch(const std::vector<std::vector<complexd>>& input) {
    int batch = static_cast<int>(input.size());
    if (batch == 0) {
        return {};
    }

    static thread_local BatchFFTCache cache;
    cache.ensure(n_, batch, FFT_INVERSE);

    std::vector<ComplexFloat> host_input(static_cast<std::size_t>(n_) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n_; ++i) {
            std::size_t idx = static_cast<std::size_t>(b) * n_ + i;
            if (i < static_cast<int>(input[b].size())) {
                host_input[idx].re = static_cast<float>(input[b][i].real());
                host_input[idx].im = static_cast<float>(input[b][i].imag());
            } else {
                host_input[idx].re = 0.0f;
                host_input[idx].im = 0.0f;
            }
        }
    }

    cudaMemcpy(cache.d_input, host_input.data(), host_input.size() * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(cache.plan, cache.d_output, cache.d_input);

    std::vector<ComplexFloat> host_output(static_cast<std::size_t>(n_) * batch);
    cudaMemcpy(host_output.data(), cache.d_output, host_output.size() * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);

    std::vector<std::vector<complexd>> result(batch, std::vector<complexd>(n_));
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n_; ++i) {
            std::size_t idx = static_cast<std::size_t>(b) * n_ + i;
            result[b][i] = complexd(host_output[idx].re, host_output[idx].im);
        }
    }

    return result;
}

/**
 * @brief 实数 FFT
 *
 * 执行实数到复数的 FFT，返回前 N/2+1 个点。
 */
std::vector<ComplexFloat> FFTInterface::ifft_batch_float(const std::vector<ComplexFloat>& input, int batch) {
    if (batch <= 0) {
        return {};
    }

    static thread_local BatchFFTCache cache;
    cache.ensure(n_, batch, FFT_INVERSE);

    std::size_t total = static_cast<std::size_t>(n_) * batch;
    std::vector<ComplexFloat> host_input(total);
    std::size_t copy_count = std::min(total, input.size());
    std::copy(input.begin(), input.begin() + copy_count, host_input.begin());

    cudaMemcpy(cache.d_input, host_input.data(), total * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(cache.plan, cache.d_output, cache.d_input);

    std::vector<ComplexFloat> host_output(total);
    cudaMemcpy(host_output.data(), cache.d_output, total * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);
    return host_output;
}

std::vector<complexd> FFTInterface::rfft(const std::vector<complexd>& input) {
    auto full_fft = fft(input);
    int n = input.size();
    int output_len = n / 2 + 1;

    std::vector<complexd> result(output_len);
    for (int i = 0; i < output_len && i < (int)full_fft.size(); i++) {
        result[i] = full_fft[i];
    }
    return result;
}

/**
 * @brief 实数 IFFT
 *
 * 执行复数到实数的 IFFT，输入为前 N/2+1 个点。
 */
std::vector<complexd> FFTInterface::irfft(const std::vector<complexd>& input, int output_len) {
    int n = input.size();
    if (output_len <= 0) {
        output_len = 2 * (n - 1);
    }

    std::vector<complexd> full_spectrum(output_len);
    for (int i = 0; i < n && i < output_len; i++) {
        full_spectrum[i] = input[i];
    }
    for (int i = 1; i < n - 1 && i < output_len; i++) {
        int target_idx = output_len - i;
        if (target_idx >= n && target_idx < output_len) {
            full_spectrum[target_idx] = std::conj(input[i]);
        }
    }

    FFTInterface fft_iface(output_len);
    auto result = fft_iface.ifft(full_spectrum);

    std::vector<complexd> real_result(output_len);
    for (int i = 0; i < output_len; i++) {
        real_result[i] = complexd(result[i].real(), 0.0);
    }
    return real_result;
}

/**
 * @brief FFTInterface2D 构造函数
 *
 * 创建 2D FFT 接口对象，分配 GPU 内存并创建 2D FFT 计划。
 */
FFTInterface2D::FFTInterface2D(int nx, int ny)
    : FFTInterface2D(nx, ny, false) {}

FFTInterface2D::FFTInterface2D(int nx, int ny, FFTInterface::BatchOnly)
    : FFTInterface2D(nx, ny, true) {}

FFTInterface2D::FFTInterface2D(int nx, int ny, bool device_only) : nx_(nx), ny_(ny),
    plan_forward_(nullptr), plan_inverse_(nullptr), d_input_(nullptr), d_output_(nullptr) {
    if (nx_ <= 0 || ny_ <= 0) {
        throw std::invalid_argument("FFTInterface2D: dimensions must be positive");
    }
    if (device_only) return;
    const std::size_t count = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    try {
        if (fft_plan_create_2d(&plan_forward_, ny_, nx_, FFT_TYPE_C2C, FFT_FORWARD) != FFT_SUCCESS) {
            throw std::runtime_error("FFTInterface2D: forward plan creation failed");
        }
        if (fft_plan_create_2d(&plan_inverse_, ny_, nx_, FFT_TYPE_C2C, FFT_INVERSE) != FFT_SUCCESS) {
            throw std::runtime_error("FFTInterface2D: inverse plan creation failed");
        }
        CUDA_CHECK(cuda_utils::tracked_cuda_malloc(&d_input_, count * sizeof(ComplexFloat)));
        CUDA_CHECK(cuda_utils::tracked_cuda_malloc(&d_output_, count * sizeof(ComplexFloat)));
    } catch (...) {
        if (plan_forward_) fft_plan_destroy(plan_forward_);
        if (plan_inverse_) fft_plan_destroy(plan_inverse_);
        if (d_input_) cuda_utils::tracked_cuda_free(d_input_);
        if (d_output_) cuda_utils::tracked_cuda_free(d_output_);
        plan_forward_ = nullptr;
        plan_inverse_ = nullptr;
        d_input_ = nullptr;
        d_output_ = nullptr;
        throw;
    }
}

/**
 * @brief FFTInterface2D 析构函数
 *
 * 释放 2D FFT 计划和 GPU 内存。
 */
FFTInterface2D::~FFTInterface2D() noexcept {
    if (plan_forward_) fft_plan_destroy(plan_forward_);
    if (plan_inverse_) fft_plan_destroy(plan_inverse_);
    if (d_input_) cuda_utils::tracked_cuda_free(d_input_);
    if (d_output_) cuda_utils::tracked_cuda_free(d_output_);
}

/**
 * @brief 执行 2D 正向 FFT
 *
 * 将输入矩阵拷贝到 GPU，执行 2D FFT，将结果拷贝回主机。
 */
std::vector<std::vector<complexd>> FFTInterface2D::fft2d(const std::vector<std::vector<complexd>>& input) {
    std::vector<ComplexFloat> host_input(nx_ * ny_);
    for (int i = 0; i < nx_; i++) {
        for (int j = 0; j < ny_; j++) {
            int idx = i * ny_ + j;
            if (i < (int)input.size() && j < (int)input[i].size()) {
                host_input[idx].re = static_cast<float>(input[i][j].real());
                host_input[idx].im = static_cast<float>(input[i][j].imag());
            } else {
                host_input[idx].re = 0.0f;
                host_input[idx].im = 0.0f;
            }
        }
    }
    cudaMemcpy(d_input_, host_input.data(), nx_ * ny_ * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(plan_forward_, d_output_, d_input_);
    std::vector<ComplexFloat> host_output(nx_ * ny_);
    cudaMemcpy(host_output.data(), d_output_, nx_ * ny_ * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);
    std::vector<std::vector<complexd>> result(nx_, std::vector<complexd>(ny_));
    for (int i = 0; i < nx_; i++) {
        for (int j = 0; j < ny_; j++) {
            int idx = i * ny_ + j;
            result[i][j] = complexd(host_output[idx].re, host_output[idx].im);
        }
    }
    return result;
}

/**
 * @brief 执行 2D 逆向 FFT
 *
 * 将输入矩阵拷贝到 GPU，执行 2D IFFT，将结果拷贝回主机。
 */
std::vector<std::vector<complexd>> FFTInterface2D::ifft2d(const std::vector<std::vector<complexd>>& input) {
    std::vector<ComplexFloat> host_input(nx_ * ny_);
    for (int i = 0; i < nx_; i++) {
        for (int j = 0; j < ny_; j++) {
            int idx = i * ny_ + j;
            if (i < (int)input.size() && j < (int)input[i].size()) {
                host_input[idx].re = static_cast<float>(input[i][j].real());
                host_input[idx].im = static_cast<float>(input[i][j].imag());
            } else {
                host_input[idx].re = 0.0f;
                host_input[idx].im = 0.0f;
            }
        }
    }
    cudaMemcpy(d_input_, host_input.data(), nx_ * ny_ * sizeof(ComplexFloat), cudaMemcpyHostToDevice);
    fft_execute(plan_inverse_, d_output_, d_input_);
    std::vector<ComplexFloat> host_output(nx_ * ny_);
    cudaMemcpy(host_output.data(), d_output_, nx_ * ny_ * sizeof(ComplexFloat), cudaMemcpyDeviceToHost);
    std::vector<std::vector<complexd>> result(nx_, std::vector<complexd>(ny_));
    for (int i = 0; i < nx_; i++) {
        for (int j = 0; j < ny_; j++) {
            int idx = i * ny_ + j;
            result[i][j] = complexd(host_output[idx].re, host_output[idx].im);
        }
    }
    return result;
}

void FFTInterface2D::fft2d_device(
    const DeviceArray<ComplexFloat>& input,
    DeviceArray<ComplexFloat>& output)
{
    const std::size_t expected = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    if (input.size() != expected || output.size() != expected) {
        throw std::invalid_argument("FFTInterface2D::fft2d_device: input/output sizes must match nx*ny");
    }
    static thread_local FFT2DPlanPool cache;
    const FFTPlanHandle plan = plan_forward_ ? plan_forward_
        : cache.get(nx_, ny_, FFT_FORWARD);
    FFTResult result = fft_execute(plan, output.data(), input.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface2D::fft2d_device: fft_execute failed");
    }
}

void FFTInterface2D::ifft2d_device(
    const DeviceArray<ComplexFloat>& input,
    DeviceArray<ComplexFloat>& output)
{
    const std::size_t expected = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    if (input.size() != expected || output.size() != expected) {
        throw std::invalid_argument("FFTInterface2D::ifft2d_device: input/output sizes must match nx*ny");
    }
    static thread_local FFT2DPlanPool cache;
    const FFTPlanHandle plan = plan_inverse_ ? plan_inverse_
        : cache.get(nx_, ny_, FFT_INVERSE);
    FFTResult result = fft_execute(plan, output.data(), input.data());
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFTInterface2D::ifft2d_device: fft_execute failed");
    }
}

} // namespace cusignal
