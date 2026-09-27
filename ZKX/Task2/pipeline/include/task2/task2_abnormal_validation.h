#pragma once

#include <cusignal/runtime/errors.h>

#include <cstdint>
#include <string>

namespace task2 {

inline constexpr std::uint64_t kTask2MaxInputElements = 1ULL << 26U;

inline void require_task2_upstream(bool condition, const std::string& step_name,
                                   const std::string& required_input)
{
    if (condition) return;
    cusignal::throw_error(cusignal::ErrorCode::UPSTREAM_STEP_FAILED,
        "Task2 " + step_name + "缺少上游输入" + required_input +
        "，pipeline已停止且未消费下游输出");
}

inline void validate_task2_dtype(const std::string& dtype)
{
    if (dtype == "FP32") return;
    cusignal::throw_error(cusignal::ErrorCode::INVALID_DTYPE,
        "Task2 pipeline不支持输入dtype：" + dtype + "；当前任务合同仅允许FP32");
}

inline void validate_task2_backend(const std::string& requested_backend,
                                   const std::string& compiled_backend)
{
    if ((compiled_backend == "dlfft" || compiled_backend == "fft_thrust") &&
        requested_backend == compiled_backend) return;
    cusignal::throw_error(cusignal::ErrorCode::UNSUPPORTED_OPERATION,
        "Task2请求的FFT backend不可用：requested=" + requested_backend +
        "，compiled=" + compiled_backend + "；禁止静默回退");
}

inline std::uint64_t checked_task2_input_elements(int samples)
{
    if (samples <= 0)
        cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
            "Task2 samples必须为正数");
    const auto elements = static_cast<std::uint64_t>(samples);
    if (elements > kTask2MaxInputElements)
        cusignal::throw_error(cusignal::ErrorCode::OUT_OF_MEMORY,
            "Task2输入规模超过受控资源上限：elements=" +
            std::to_string(elements) + "，limit=" +
            std::to_string(kTask2MaxInputElements) + "；未执行内存分配");
    return elements;
}

}  // namespace task2
