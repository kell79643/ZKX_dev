#pragma once

#include <cusignal/runtime/errors.h>

#include <cstdint>
#include <limits>
#include <string>

namespace task1 {

inline constexpr std::uint64_t kTask1MaxInputElements = 1ULL << 26U;

inline void require_task1_upstream(bool condition, const std::string& step_name,
                                   const std::string& required_input)
{
    if (condition) return;
    cusignal::throw_error(cusignal::ErrorCode::UPSTREAM_STEP_FAILED,
        "Task1 " + step_name + "缺少上游输入" + required_input +
        "，pipeline已停止且未消费下游输出");
}

inline void validate_task1_dtype(const std::string& dtype)
{
    if (dtype == "ComplexFP32") return;
    cusignal::throw_error(cusignal::ErrorCode::INVALID_DTYPE,
        "Task1 pipeline不支持dtype：" + dtype + "；当前任务合同仅允许ComplexFP32");
}

inline void validate_task1_backend(const std::string& requested_backend,
                                   const std::string& compiled_backend)
{
    if ((compiled_backend == "dlfft" || compiled_backend == "fft_thrust") &&
        requested_backend == compiled_backend) return;
    cusignal::throw_error(cusignal::ErrorCode::UNSUPPORTED_OPERATION,
        "Task1请求的FFT backend不可用：requested=" + requested_backend +
        "，compiled=" + compiled_backend + "；禁止静默回退");
}

inline std::uint64_t checked_task1_input_elements(int num_pulses,
                                                  int samples_per_pulse)
{
    if (num_pulses <= 0 || samples_per_pulse <= 0)
        cusignal::throw_error(cusignal::ErrorCode::INVALID_ARGUMENT,
            "Task1输入维度必须为正数");
    const auto pulses = static_cast<std::uint64_t>(num_pulses);
    const auto samples = static_cast<std::uint64_t>(samples_per_pulse);
    if (pulses > std::numeric_limits<std::uint64_t>::max() / samples)
        cusignal::throw_error(cusignal::ErrorCode::OUT_OF_MEMORY,
            "Task1输入元素数溢出，已在分配前安全拒绝");
    const std::uint64_t elements = pulses * samples;
    if (elements > kTask1MaxInputElements)
        cusignal::throw_error(cusignal::ErrorCode::OUT_OF_MEMORY,
            "Task1输入规模超过受控资源上限：elements=" +
            std::to_string(elements) + "，limit=" +
            std::to_string(kTask1MaxInputElements) + "；未执行内存分配");
    return elements;
}

}  // namespace task1
