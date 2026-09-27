#pragma once

#include <cusignal/runtime/errors.h>

#include <cuda_runtime.h>

#include <string>

namespace cusignal::cuda_utils {

inline void throw_cuda_error(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line)
{
    const char* name = cudaGetErrorName(error);
    const char* message = cudaGetErrorString(error);
    BackendError backend{std::to_string(static_cast<int>(error)),
        name ? name : "UNKNOWN_BACKEND_ERROR",
        message ? message : "backend error message unavailable"};
    const ErrorCode code = backend.name.find("MemoryAllocation") != std::string::npos ||
            backend.name.find("OUT_OF_MEMORY") != std::string::npos
        ? ErrorCode::OUT_OF_MEMORY : ErrorCode::CUDA_RUNTIME_ERROR;
    throw_error(code,
        std::string("CUDA Runtime API调用失败：") + expression +
            "，位置=" + file + ':' + std::to_string(line) +
            "，后端=" + backend.name + "（" + backend.message + "）",
        std::move(backend), true);
}

inline void check_cuda(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line)
{
    if (error != cudaSuccess) {
        throw_cuda_error(error, expression, file, line);
    }
}

inline void check_last_kernel_error(
    const char* file,
    int line)
{
    check_cuda(cudaGetLastError(), "cudaGetLastError()", file, line);
}

}  // namespace cusignal::cuda_utils

#define CUDA_CHECK(expr) \
    ::cusignal::cuda_utils::check_cuda((expr), #expr, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK() \
    ::cusignal::cuda_utils::check_last_kernel_error(__FILE__, __LINE__)
