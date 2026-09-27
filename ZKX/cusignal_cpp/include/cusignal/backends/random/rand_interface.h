#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace cusignal {

// ZQ500 RAND interface. The dlrand backend currently supports FP32 buffers.
// Integer and FP16 callers should generate FP32 random values first, then convert
// or scale in the owning operator according to its numeric policy.
void rand_generate_uniform_device(
    float* d_output,
    std::size_t count,
    unsigned long long seed,
    cudaStream_t stream = nullptr);

void rand_generate_normal_device(
    float* d_output,
    std::size_t count,
    float mean,
    float stddev,
    unsigned long long seed,
    cudaStream_t stream = nullptr);

}  // namespace cusignal
