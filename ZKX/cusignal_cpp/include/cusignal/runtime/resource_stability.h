#pragma once

#include <cusignal/runtime/cuda_error.h>

#include <cstddef>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace cusignal::cuda_utils {

#ifndef CUSIGNAL_RESOURCE_WARMUP_RUNS
#define CUSIGNAL_RESOURCE_WARMUP_RUNS 1024
#endif
#ifndef CUSIGNAL_RESOURCE_MEASURED_RUNS
#define CUSIGNAL_RESOURCE_MEASURED_RUNS 1024
#endif

inline constexpr int kResourceWarmupRuns = CUSIGNAL_RESOURCE_WARMUP_RUNS;
inline constexpr int kResourceMeasuredRuns = CUSIGNAL_RESOURCE_MEASURED_RUNS;

inline int resource_run_count(const char* environment_name, int fallback)
{
    const char* value = std::getenv(environment_name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 1000000L) {
        throw std::invalid_argument(
            std::string(environment_name) + " must be an integer in [1, 1000000]");
    }
    return static_cast<int>(parsed);
}

inline int resource_warmup_runs()
{
    return resource_run_count(
        "CUSIGNAL_RESOURCE_WARMUP_RUNS", kResourceWarmupRuns);
}

inline int resource_measured_runs()
{
    return resource_run_count(
        "CUSIGNAL_RESOURCE_MEASURED_RUNS", kResourceMeasuredRuns);
}

struct ResourceSample {
    std::size_t cpu_rss_bytes = 0;
    std::size_t cpu_live_heap_bytes = 0;
    std::size_t gpu_used_bytes = 0;
};

class DevicePeakMemoryMonitor {
public:
    explicit DevicePeakMemoryMonitor(
        std::chrono::milliseconds interval = std::chrono::milliseconds(1))
        : interval_(interval)
    {
        CUDA_CHECK(cudaGetDevice(&device_));
        baseline_used_bytes_ = current_used_bytes();
        peak_used_bytes_.store(baseline_used_bytes_);
        worker_ = std::thread([this] { poll(); });
    }

    DevicePeakMemoryMonitor(const DevicePeakMemoryMonitor&) = delete;
    DevicePeakMemoryMonitor& operator=(const DevicePeakMemoryMonitor&) = delete;

    ~DevicePeakMemoryMonitor()
    {
        if (worker_.joinable()) {
            stop_.store(true);
            worker_.join();
        }
    }

    void stop()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        sample_once();
        stop_.store(true);
        if (worker_.joinable()) worker_.join();
        if (cuda_error_.load() != cudaSuccess) {
            throw std::runtime_error("device peak memory monitor cudaMemGetInfo failed");
        }
    }

    std::size_t baseline_used_bytes() const { return baseline_used_bytes_; }
    std::size_t peak_used_bytes() const { return peak_used_bytes_.load(); }
    std::size_t delta_peak_bytes() const
    {
        const auto peak = peak_used_bytes();
        return peak > baseline_used_bytes_ ? peak - baseline_used_bytes_ : 0;
    }

private:
    std::size_t current_used_bytes()
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        return total_bytes - free_bytes;
    }

    void sample_once()
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            cuda_error_.store(status);
            return;
        }
        const std::size_t used = total_bytes - free_bytes;
        auto peak = peak_used_bytes_.load();
        while (used > peak && !peak_used_bytes_.compare_exchange_weak(peak, used)) {
        }
    }

    void poll()
    {
        if (cudaSetDevice(device_) != cudaSuccess) {
            cuda_error_.store(cudaErrorInvalidDevice);
            return;
        }
        while (!stop_.load()) {
            sample_once();
            std::this_thread::sleep_for(interval_);
        }
        sample_once();
    }

    int device_ = 0;
    std::chrono::milliseconds interval_;
    std::size_t baseline_used_bytes_ = 0;
    std::atomic<std::size_t> peak_used_bytes_{0};
    std::atomic<bool> stop_{false};
    std::atomic<cudaError_t> cuda_error_{cudaSuccess};
    std::thread worker_;
};

inline std::size_t current_cpu_rss_bytes()
{
    std::ifstream statm("/proc/self/statm");
    std::size_t total_pages = 0;
    std::size_t resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) {
        throw std::runtime_error("cannot read /proc/self/statm");
    }
    (void)total_pages;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    return resident_pages * static_cast<std::size_t>(page_size);
}

inline ResourceSample sample_resources()
{
    CUDA_CHECK(cudaDeviceSynchronize());
#if defined(__GLIBC__)
    // 测的是仍被程序持有的 CPU 内存，而不是 glibc 已 free 但暂留在 arena 中的页。
    // trim 只用于稳定性门禁采样，不进入正式算子或任务性能路径。
    (void)malloc_trim(0);
#endif
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
#if defined(__GLIBC__)
    const auto heap = mallinfo2();
    const std::size_t live_heap_bytes = heap.uordblks;
#else
    // 非 glibc 平台没有可移植的 live heap 计数，退化为 RSS 趋势。
    const std::size_t live_heap_bytes = current_cpu_rss_bytes();
#endif
    return {
        current_cpu_rss_bytes(), live_heap_bytes, total_bytes - free_bytes};
}

struct ResourceTrend {
    std::size_t first = 0;
    std::size_t last = 0;
    std::size_t growth = 0;
    std::size_t positive_steps = 0;
    std::size_t negative_steps = 0;
    std::size_t trailing_positive_steps = 0;
    std::size_t tail_growth = 0;
    std::size_t tail_positive_steps = 0;
    std::size_t tail_negative_steps = 0;
    bool stable = true;
};

template <class Selector>
ResourceTrend evaluate_resource_trend(
    const std::vector<ResourceSample>& samples,
    Selector selector,
    std::size_t allowed_growth)
{
    if (samples.size() < 2) {
        throw std::invalid_argument("resource trend requires at least two samples");
    }
    ResourceTrend trend{};
    trend.first = selector(samples.front());
    trend.last = selector(samples.back());
    trend.growth = trend.last > trend.first ? trend.last - trend.first : 0;
    for (std::size_t i = 1; i < samples.size(); ++i) {
        if (selector(samples[i]) > selector(samples[i - 1])) {
            ++trend.positive_steps;
            ++trend.trailing_positive_steps;
        } else if (selector(samples[i]) < selector(samples[i - 1])) {
            ++trend.negative_steps;
            trend.trailing_positive_steps = 0;
        }
    }
    const std::size_t tail_begin = samples.size() * 3 / 4;
    const std::size_t tail_first = selector(samples[tail_begin]);
    const std::size_t tail_last = selector(samples.back());
    trend.tail_growth = tail_last > tail_first ? tail_last - tail_first : 0;
    for (std::size_t i = tail_begin + 1; i < samples.size(); ++i) {
        if (selector(samples[i]) > selector(samples[i - 1])) {
            ++trend.tail_positive_steps;
        } else if (selector(samples[i]) < selector(samples[i - 1])) {
            ++trend.tail_negative_steps;
        }
    }
    // 慢速泄漏同样不可接受：全窗或最后四分之一窗只要存在正净增长，且上涨步数
    // 至少是回落步数的两倍，就判为方向性持续增长。使用尾窗统计而不是“最后恰好
    // 两点上涨”，避免把 glibc arena 的几个字节级端点抖动误判为泄漏。
    const bool directional_growth =
        trend.growth > 0 && trend.positive_steps >= 2 &&
        trend.positive_steps >= 2 * trend.negative_steps;
    const bool tail_directional_growth =
        trend.tail_growth > 0 && trend.tail_positive_steps >= 2 &&
        trend.tail_positive_steps >= 2 * trend.tail_negative_steps;
    const bool sustained_growth = directional_growth || tail_directional_growth;
    trend.stable = trend.growth <= allowed_growth && !sustained_growth;
    return trend;
}

inline bool report_resource_stability(
    const std::string& label,
    const std::vector<ResourceSample>& samples,
    bool* cpu_stable_out = nullptr,
    bool* gpu_stable_out = nullptr,
    // 1024 点正式窗将 CPU 有界分配器抖动限制在 64 KiB；GPU 要求逐字节净增长为 0。
    // 方向判据仍独立生效，因此低于绝对阈值的持续慢增长也不能通过。
    std::size_t allowed_cpu_growth = 64U * 1024U,
    std::size_t allowed_gpu_growth = 0)
{
    for (std::size_t i = 0; i < samples.size(); ++i) {
        std::cout << "[MEMORY][SAMPLE] label=" << label
                  << " iteration=" << (i + 1)
                  << " cpu_rss_bytes=" << samples[i].cpu_rss_bytes
                  << " cpu_live_heap_bytes=" << samples[i].cpu_live_heap_bytes
                  << " gpu_used_bytes=" << samples[i].gpu_used_bytes << "\n";
    }
    const auto cpu = evaluate_resource_trend(
        samples, [](const ResourceSample& sample) {
            return sample.cpu_live_heap_bytes;
        },
        allowed_cpu_growth);
    const auto gpu = evaluate_resource_trend(
        samples, [](const ResourceSample& sample) { return sample.gpu_used_bytes; },
        allowed_gpu_growth);
    if (cpu_stable_out != nullptr) *cpu_stable_out = cpu.stable;
    if (gpu_stable_out != nullptr) *gpu_stable_out = gpu.stable;
    std::cout << "[MEMORY][SUMMARY] label=" << label
              << " cpu_growth_bytes=" << cpu.growth
              << " cpu_positive_steps=" << cpu.positive_steps
              << " cpu_negative_steps=" << cpu.negative_steps
              << " cpu_trailing_positive_steps=" << cpu.trailing_positive_steps
              << " cpu_tail_growth_bytes=" << cpu.tail_growth
              << " cpu_tail_positive_steps=" << cpu.tail_positive_steps
              << " cpu_tail_negative_steps=" << cpu.tail_negative_steps
              << " gpu_growth_bytes=" << gpu.growth
              << " gpu_positive_steps=" << gpu.positive_steps
              << " gpu_negative_steps=" << gpu.negative_steps
              << " gpu_trailing_positive_steps=" << gpu.trailing_positive_steps
              << " gpu_tail_growth_bytes=" << gpu.tail_growth
              << " gpu_tail_positive_steps=" << gpu.tail_positive_steps
              << " gpu_tail_negative_steps=" << gpu.tail_negative_steps
              << " cpu_status=" << (cpu.stable ? "pass" : "fail")
              << " gpu_status=" << (gpu.stable ? "pass" : "fail")
              << " status=" << (cpu.stable && gpu.stable ? "passed" : "failed")
              << "\n";
    return cpu.stable && gpu.stable;
}

}  // namespace cusignal::cuda_utils
