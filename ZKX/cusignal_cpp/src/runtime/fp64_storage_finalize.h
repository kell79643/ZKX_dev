#pragma once

#include <cstddef>

namespace cusignal::cuda_utils {

// ZQ500 does not execute FP64 arithmetic in formal kernels. These entry points
// widen already-computed FP32 values by writing IEEE storage bits only.
void widen_fp32_storage_device(
    const float* input, void* output_storage, std::size_t count);

void widen_complex_fp32_storage_device(
    const void* input, void* output_storage, std::size_t count);

}  // namespace cusignal::cuda_utils
