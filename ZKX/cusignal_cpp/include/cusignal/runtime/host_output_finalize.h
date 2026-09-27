#pragma once

#include <cusignal/runtime/device_array.h>
#include "fp64_storage_finalize.h"
#include <cusignal/backends/fft/fft_interface.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cusignal::cuda_utils {

// FP64 is an interface/storage dtype only.  These helpers deliberately perform
// every widening conversion on the host after the FP32 GPU computation has
// completed.  DeviceArray<double>/DeviceArray<ComplexDouble> are then populated
// only by H2D byte transfer; no CUDA kernel performs FP64 arithmetic.
inline std::vector<double> finalize_fp64_on_host(
    const std::vector<float>& computed)
{
    std::vector<double> output(computed.size());
    for (std::size_t index = 0; index < computed.size(); ++index) {
        output[index] = static_cast<double>(computed[index]);
    }
    return output;
}

inline std::vector<ComplexDouble> finalize_complex_fp64_on_host(
    const std::vector<ComplexFloat>& computed)
{
    std::vector<ComplexDouble> output(computed.size());
    for (std::size_t index = 0; index < computed.size(); ++index) {
        output[index].re = static_cast<double>(computed[index].re);
        output[index].im = static_cast<double>(computed[index].im);
    }
    return output;
}

inline DeviceArray<double> finalize_fp64_device_storage(
    const DeviceArray<float>& computed)
{
    DeviceArray<double> output(computed.size());
    widen_fp32_storage_device(
        computed.data(), output.data(), computed.size());
    return output;
}

inline DeviceArray<ComplexDouble> finalize_complex_fp64_device_storage(
    const DeviceArray<ComplexFloat>& computed)
{
    static_assert(sizeof(ComplexDouble) == 2 * sizeof(double));
    DeviceArray<ComplexDouble> output(computed.size());
    widen_complex_fp32_storage_device(
        computed.data(), output.data(), computed.size());
    return output;
}

// An FP64 result may feed another formal operator only through an explicit
// host boundary.  These helpers make that precision-changing boundary visible
// in source and prevent a DeviceArray<double> from being consumed by a GPU
// kernel.  NaN and infinities retain their IEEE category; a finite value that
// cannot be represented by FP32 is rejected instead of being silently clipped.
inline float narrow_fp64_value_to_fp32_on_host(double value)
{
    constexpr double fp32_low =
        static_cast<double>(std::numeric_limits<float>::lowest());
    constexpr double fp32_high =
        static_cast<double>(std::numeric_limits<float>::max());
    if (std::isfinite(value) && (value < fp32_low || value > fp32_high)) {
        throw std::overflow_error(
            "FP64-to-FP32 host boundary received a finite value outside the FP32 range");
    }
    return static_cast<float>(value);
}

inline std::vector<float> narrow_fp64_to_fp32_on_host(
    const std::vector<double>& output_storage)
{
    std::vector<float> input(output_storage.size());
    for (std::size_t index = 0; index < output_storage.size(); ++index) {
        input[index] = narrow_fp64_value_to_fp32_on_host(output_storage[index]);
    }
    return input;
}

inline std::vector<ComplexFloat> narrow_complex_fp64_to_fp32_on_host(
    const std::vector<ComplexDouble>& output_storage)
{
    std::vector<ComplexFloat> input(output_storage.size());
    for (std::size_t index = 0; index < output_storage.size(); ++index) {
        input[index].re = narrow_fp64_value_to_fp32_on_host(output_storage[index].re);
        input[index].im = narrow_fp64_value_to_fp32_on_host(output_storage[index].im);
    }
    return input;
}

inline DeviceArray<float> narrow_fp64_device_storage_to_fp32_on_host(
    const DeviceArray<double>& output_storage)
{
    return DeviceArray<float>::from_host(
        narrow_fp64_to_fp32_on_host(output_storage.to_host()));
}

inline DeviceArray<ComplexFloat>
narrow_complex_fp64_device_storage_to_fp32_on_host(
    const DeviceArray<ComplexDouble>& output_storage)
{
    return DeviceArray<ComplexFloat>::from_host(
        narrow_complex_fp64_to_fp32_on_host(output_storage.to_host()));
}

}  // namespace cusignal::cuda_utils
