#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace cusignal::cuda_utils {

namespace detail {

struct DeviceAllocationState {
    std::mutex mutex;
    std::unordered_map<void*, std::size_t> live_allocations;
    std::size_t current_bytes = 0;
    std::size_t peak_bytes = 0;
};

inline DeviceAllocationState& device_allocation_state()
{
    static DeviceAllocationState state;
    return state;
}

}  // namespace detail

// These wrappers measure requested bytes owned by this application. Unlike a
// cudaMemGetInfo polling thread, allocation-event accounting cannot miss short
// lived buffers and is not contaminated by other processes on the device.
inline cudaError_t tracked_cuda_malloc(void** pointer, std::size_t bytes)
{
    const cudaError_t status = cudaMalloc(pointer, bytes);
    if (status == cudaSuccess && pointer != nullptr && *pointer != nullptr) {
        auto& state = detail::device_allocation_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.live_allocations[*pointer] = bytes;
        state.current_bytes += bytes;
        state.peak_bytes = std::max(state.peak_bytes, state.current_bytes);
    }
    return status;
}

template <typename T>
inline cudaError_t tracked_cuda_malloc(T** pointer, std::size_t bytes)
{
    return tracked_cuda_malloc(reinterpret_cast<void**>(pointer), bytes);
}

inline cudaError_t tracked_cuda_free(void* pointer)
{
    const cudaError_t status = cudaFree(pointer);
    if (status == cudaSuccess && pointer != nullptr) {
        auto& state = detail::device_allocation_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        const auto found = state.live_allocations.find(pointer);
        if (found != state.live_allocations.end()) {
            state.current_bytes -= found->second;
            state.live_allocations.erase(found);
        }
    }
    return status;
}

inline void reset_tracked_device_peak()
{
    auto& state = detail::device_allocation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.peak_bytes = state.current_bytes;
}

inline std::size_t tracked_device_current_bytes()
{
    auto& state = detail::device_allocation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.current_bytes;
}

inline std::size_t tracked_device_peak_bytes()
{
    auto& state = detail::device_allocation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.peak_bytes;
}

}  // namespace cusignal::cuda_utils
