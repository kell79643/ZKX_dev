#pragma once

#include <cusignal/runtime/copy_utils.h>
#include <cusignal/runtime/cuda_error.h>
#include <cusignal/runtime/device_allocation_tracker.h>

#include <cstddef>
#include <vector>

namespace cusignal {

/**
 * @brief Minimal 1D contiguous device-resident array.
 *
 * DeviceArray is intentionally move-only. It owns CUDA device memory and never
 * performs implicit host/device transfers. Use from_host() and to_host()
 * explicitly at API boundaries.
 */
template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;

    explicit DeviceArray(std::size_t count)
    {
        allocate(count);
    }

    ~DeviceArray() noexcept
    {
        reset_noexcept();
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this != &other) {
            reset_noexcept();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    static DeviceArray from_host(const std::vector<T>& host)
    {
        DeviceArray out(host.size());
        if (!host.empty()) {
            cuda_utils::copy_to_device(out.data_, host.data(), host.size());
            cuda_utils::synchronize_stream();
        }
        return out;
    }

    std::vector<T> to_host() const
    {
        std::vector<T> host(size_);
        if (size_ != 0) {
            cuda_utils::copy_to_host(host.data(), data_, size_);
            cuda_utils::synchronize_stream();
        }
        return host;
    }

    void reset()
    {
        if (data_ != nullptr) {
            cudaError_t free_status = cuda_utils::tracked_cuda_free(data_);
            data_ = nullptr;
            size_ = 0;
            if (free_status != cudaSuccess) {
                cuda_utils::throw_cuda_error(free_status, "cudaFree(data_)", __FILE__, __LINE__);
            }
        }
    }

    void reset(std::size_t count)
    {
        allocate(count);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

private:
    void allocate(std::size_t count)
    {
        reset();
        if (count == 0) {
            return;
        }
        CUDA_CHECK(cuda_utils::tracked_cuda_malloc(
            reinterpret_cast<void**>(&data_), count * sizeof(T)));
        size_ = count;
    }

    void reset_noexcept() noexcept
    {
        if (data_ != nullptr) {
            (void)cuda_utils::tracked_cuda_free(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace cusignal
