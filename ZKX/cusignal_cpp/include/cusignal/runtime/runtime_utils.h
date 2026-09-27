#pragma once

#include <cusignal/runtime/cuda_error.h>

#include <cstddef>

namespace cusignal::cuda_utils {

struct DeviceInfo {
    int ordinal = 0;
    cudaDeviceProp properties{};
};

struct MemoryInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

inline DeviceInfo current_device_info()
{
    DeviceInfo info{};
    CUDA_CHECK(cudaGetDevice(&info.ordinal));
    CUDA_CHECK(cudaGetDeviceProperties(&info.properties, info.ordinal));
    return info;
}

inline int device_count()
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

inline MemoryInfo current_memory_info()
{
    MemoryInfo info{};
    CUDA_CHECK(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
    return info;
}

class EventTimer {
public:
    EventTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        try {
            CUDA_CHECK(cudaEventCreate(&stop_));
        } catch (...) {
            (void)cudaEventDestroy(start_);
            start_ = nullptr;
            throw;
        }
    }

    ~EventTimer() noexcept
    {
        if (start_ != nullptr) {
            (void)cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            (void)cudaEventDestroy(stop_);
        }
    }

    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;
    EventTimer(EventTimer&&) = delete;
    EventTimer& operator=(EventTimer&&) = delete;

    void record_start(cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void record_stop(cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    float elapsed_ms()
    {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace cusignal::cuda_utils
