#include <task2/task2_runner.h>
#include <task2/task2_config.h>

#include <cusignal/runtime/resource_stability.h>
#include <cusignal/runtime/device_allocation_tracker.h>
#include <cusignal/runtime/cuda_error.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    try {
        if (argc != 3 || std::string(argv[1]) != "--config")
            throw std::invalid_argument("usage: task2_memory_stability --config <path>");
        const auto config = task2::TaskConfig::load(argv[2]);
        const int warmup_runs = cusignal::cuda_utils::resource_warmup_runs();
        const int measured_runs = cusignal::cuda_utils::resource_measured_runs();
        for (int i = 0; i < warmup_runs; ++i) {
            if (task2::run_task2_until(6, config) != 0) {
                return 1;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::size_t allocator_baseline =
            cusignal::cuda_utils::tracked_device_current_bytes();
        cusignal::cuda_utils::reset_tracked_device_peak();
        cusignal::cuda_utils::DevicePeakMemoryMonitor peak_monitor;
        std::vector<cusignal::cuda_utils::ResourceSample> samples;
        samples.reserve(static_cast<std::size_t>(measured_runs));
        for (int i = 0; i < measured_runs; ++i) {
            if (task2::run_task2_until(6, config) != 0) {
                return 1;
            }
            samples.push_back(cusignal::cuda_utils::sample_resources());
        }
        peak_monitor.stop();
        const std::size_t allocator_peak =
            cusignal::cuda_utils::tracked_device_peak_bytes();
        const std::size_t allocator_final =
            cusignal::cuda_utils::tracked_device_current_bytes();
        const bool allocator_stable = allocator_final <= allocator_baseline;
        bool cpu_stable = false;
        bool gpu_stable = false;
        const bool stable = cusignal::cuda_utils::report_resource_stability(
            "task2-pipeline", samples, &cpu_stable, &gpu_stable);
        std::cout << "[TASK][MEMORY] task=Task2 implementation=cpp_cuda"
                  << " iterations=" << measured_runs
                  << " baseline_used_bytes=" << peak_monitor.baseline_used_bytes()
                  << " peak_used_bytes=" << peak_monitor.peak_used_bytes()
                  << " delta_peak_bytes=" << peak_monitor.delta_peak_bytes()
                  << " allocator_baseline_bytes=" << allocator_baseline
                  << " allocator_peak_live_bytes=" << allocator_peak
                  << " allocator_delta_peak_bytes=" << (allocator_peak - allocator_baseline)
                  << " allocator_final_bytes=" << allocator_final
                  << " sampling_interval_ms=1 scope=task-allocator-live-peak"
                  << " device_scope=device-global-post-warmup-diagnostic"
                  << " allocator_status=" << (allocator_stable ? "pass" : "fail")
                  << " cpu_status=" << (cpu_stable ? "pass" : "fail")
                  << " gpu_status=" << (gpu_stable ? "pass" : "fail")
                  << " status=" << (stable ? "pass" : "fail") << "\n";
        return stable ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[MEMORY][FAIL] label=task2-pipeline reason=" << error.what() << "\n";
        return 1;
    }
}
