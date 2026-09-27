#pragma once

#include <cusignal/runtime/cuda_error.h>

#include <cstddef>

namespace cusignal::cuda_utils {

template <typename T>
inline void copy_to_device(
    T* device_dst,
    const T* host_src,
    std::size_t count,
    cudaStream_t stream = nullptr)
{
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        device_dst, host_src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T>
inline void copy_to_host(
    T* host_dst,
    const T* device_src,
    std::size_t count,
    cudaStream_t stream = nullptr)
{
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        host_dst, device_src, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template <typename T>
inline void copy_device_to_device(
    T* device_dst,
    const T* device_src,
    std::size_t count,
    cudaStream_t stream = nullptr)
{
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        device_dst, device_src, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
inline void copy_value_to_device(
    T* device_dst,
    const T& host_value,
    cudaStream_t stream = nullptr)
{
    copy_to_device(device_dst, &host_value, 1, stream);
}

inline void memset_device(
    void* device_dst,
    int value,
    std::size_t bytes,
    cudaStream_t stream = nullptr)
{
    if (bytes == 0) {
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(device_dst, value, bytes, stream));
}

template <typename T>
inline void zero_device(T* device_dst, std::size_t count, cudaStream_t stream = nullptr)
{
    memset_device(device_dst, 0, count * sizeof(T), stream);
}

inline void synchronize_stream(cudaStream_t stream = nullptr)
{
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

}  // namespace cusignal::cuda_utils
