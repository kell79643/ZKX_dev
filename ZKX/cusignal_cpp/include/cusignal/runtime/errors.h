#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#if __has_include(<cuda.h>)
#include <cuda.h>
#else
using CUresult = int;
#endif

namespace cusignal {

// 显式数值属于算子库公开合同；后端原始编号不得直接充当项目错误码。
enum class ErrorCode : int {
    SUCCESS = 0,
    INVALID_ARGUMENT = 1001,
    INVALID_SHAPE = 1002,
    INVALID_DTYPE = 1003,
    UNSUPPORTED_OPERATION = 1004,
    CONFIG_ERROR = 1005,
    CUDA_DRIVER_ERROR = 2001,
    CUDA_RUNTIME_ERROR = 2002,
    OUT_OF_MEMORY = 2003,
    BACKEND_ERROR = 2099,
    EXECUTION_FAILED = 2100,
    SYNC_FAILED = 2101,
    ACCURACY_FAILED = 3001,
    RESOURCE_FAILED = 3002,
    IO_ERROR = 4001,
    INTERNAL_ERROR = 9001,
    UPSTREAM_STEP_FAILED = 9002,

    // 仅用于兼容阶段03/04已有调用；新结果写规范名称。
    OK = SUCCESS,
    INVALID_CONFIG = CONFIG_ERROR,
    UNSUPPORTED = UNSUPPORTED_OPERATION,
    ALLOCATION_FAILED = OUT_OF_MEMORY,
};

inline std::string error_code_name(ErrorCode code)
{
    switch (code) {
    case ErrorCode::SUCCESS: return "SUCCESS";
    case ErrorCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case ErrorCode::INVALID_SHAPE: return "INVALID_SHAPE";
    case ErrorCode::INVALID_DTYPE: return "INVALID_DTYPE";
    case ErrorCode::UNSUPPORTED_OPERATION: return "UNSUPPORTED_OPERATION";
    case ErrorCode::CONFIG_ERROR: return "CONFIG_ERROR";
    case ErrorCode::CUDA_DRIVER_ERROR: return "CUDA_DRIVER_ERROR";
    case ErrorCode::CUDA_RUNTIME_ERROR: return "CUDA_RUNTIME_ERROR";
    case ErrorCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case ErrorCode::BACKEND_ERROR: return "BACKEND_ERROR";
    case ErrorCode::EXECUTION_FAILED: return "EXECUTION_FAILED";
    case ErrorCode::SYNC_FAILED: return "SYNC_FAILED";
    case ErrorCode::ACCURACY_FAILED: return "ACCURACY_FAILED";
    case ErrorCode::RESOURCE_FAILED: return "RESOURCE_FAILED";
    case ErrorCode::IO_ERROR: return "IO_ERROR";
    case ErrorCode::INTERNAL_ERROR: return "INTERNAL_ERROR";
    case ErrorCode::UPSTREAM_STEP_FAILED: return "UPSTREAM_STEP_FAILED";
    }
    return "INTERNAL_ERROR";
}

inline int error_code_value(ErrorCode code) noexcept { return static_cast<int>(code); }

struct BackendError {
    std::string code{"NA"};
    std::string name{"NA"};
    std::string message{"NA"};
};

struct Error {
    ErrorCode code{ErrorCode::SUCCESS};
    std::string message;
    BackendError backend;
    bool safe_exit{true};
};

class ProjectError final : public std::runtime_error {
public:
    explicit ProjectError(Error error)
        : std::runtime_error(error.message), error_(std::move(error))
    {
    }

    const Error& detail() const noexcept { return error_; }

private:
    Error error_;
};

[[noreturn]] inline void throw_error(ErrorCode code, std::string message,
                                     BackendError backend = {}, bool safe_exit = true)
{
    if (message.empty()) message = "项目操作失败，但未提供错误描述";
    throw ProjectError({code, std::move(message), std::move(backend), safe_exit});
}

inline ErrorCode parse_error_code(const std::string& name)
{
    if (name == "SUCCESS") return ErrorCode::SUCCESS;
    if (name == "INVALID_ARGUMENT") return ErrorCode::INVALID_ARGUMENT;
    if (name == "INVALID_SHAPE") return ErrorCode::INVALID_SHAPE;
    if (name == "INVALID_DTYPE") return ErrorCode::INVALID_DTYPE;
    if (name == "UNSUPPORTED_OPERATION") return ErrorCode::UNSUPPORTED_OPERATION;
    if (name == "CONFIG_ERROR") return ErrorCode::CONFIG_ERROR;
    if (name == "CUDA_DRIVER_ERROR") return ErrorCode::CUDA_DRIVER_ERROR;
    if (name == "CUDA_RUNTIME_ERROR") return ErrorCode::CUDA_RUNTIME_ERROR;
    if (name == "OUT_OF_MEMORY") return ErrorCode::OUT_OF_MEMORY;
    if (name == "BACKEND_ERROR") return ErrorCode::BACKEND_ERROR;
    if (name == "EXECUTION_FAILED") return ErrorCode::EXECUTION_FAILED;
    if (name == "SYNC_FAILED") return ErrorCode::SYNC_FAILED;
    if (name == "ACCURACY_FAILED") return ErrorCode::ACCURACY_FAILED;
    if (name == "RESOURCE_FAILED") return ErrorCode::RESOURCE_FAILED;
    if (name == "IO_ERROR") return ErrorCode::IO_ERROR;
    if (name == "INTERNAL_ERROR") return ErrorCode::INTERNAL_ERROR;
    if (name == "UPSTREAM_STEP_FAILED") return ErrorCode::UPSTREAM_STEP_FAILED;
    throw_error(ErrorCode::CONFIG_ERROR, "未知项目错误码名称：" + name);
}

inline void require_business_dtype(const std::string& dtype)
{
    if (dtype == "FP32" || dtype == "FP16" || dtype == "INT32" ||
        dtype == "INT16" || dtype == "INT8") return;
    throw_error(ErrorCode::INVALID_DTYPE,
        "输入dtype不受支持：" + dtype + "；允许FP32/FP16/INT32/INT16/INT8");
}

inline BackendError map_backend_error(int raw_code,
    const std::function<const char*(int)>& name_resolver,
    const std::function<const char*(int)>& message_resolver)
{
    const char* resolved_name = name_resolver ? name_resolver(raw_code) : nullptr;
    const char* resolved_message = message_resolver ? message_resolver(raw_code) : nullptr;
    return {std::to_string(raw_code),
        resolved_name ? resolved_name : "UNKNOWN_BACKEND_ERROR",
        resolved_message ? resolved_message : "backend error message unavailable"};
}

using CuErrorResolver = CUresult (*)(CUresult, const char**);

struct MappedCudaError {
    ErrorCode project_code{ErrorCode::BACKEND_ERROR};
    BackendError backend;
};

inline MappedCudaError map_cu_result(
    CUresult result, CuErrorResolver get_name, CuErrorResolver get_string)
{
    if (static_cast<int>(result) == 0) return {ErrorCode::SUCCESS, {}};
    const char* name = nullptr;
    const char* message = nullptr;
    const bool name_ok = get_name && get_name(result, &name) == 0 && name;
    const bool message_ok = get_string && get_string(result, &message) == 0 && message;
    BackendError backend{std::to_string(static_cast<int>(result)),
        name_ok ? name : "UNKNOWN_BACKEND_ERROR",
        message_ok ? message : "backend error message unavailable"};
    const ErrorCode code = backend.name.find("OUT_OF_MEMORY") != std::string::npos
        ? ErrorCode::OUT_OF_MEMORY : ErrorCode::CUDA_DRIVER_ERROR;
    return {code, std::move(backend)};
}

inline Error run_checked_cu_driver_call(const std::function<CUresult()>& backend_call,
    const std::function<void()>& consume_valid_output,
    CuErrorResolver get_name, CuErrorResolver get_string)
{
    if (!backend_call) {
        return {ErrorCode::INTERNAL_ERROR, "CUDA Driver API调用器为空，未执行后端调用", {}, true};
    }
    const CUresult result = backend_call();
    if (static_cast<int>(result) != 0) {
        auto mapped = map_cu_result(result, get_name, get_string);
        return {mapped.project_code,
            "CUDA Driver API调用失败：" + mapped.backend.name + "（" + mapped.backend.message + "）",
            std::move(mapped.backend), true};
    }
    if (consume_valid_output) consume_valid_output();
    return {ErrorCode::SUCCESS, "CUDA Driver API调用成功", {}, true};
}

struct AbnormalCaseResult {
    std::string test_object;
    std::string case_id;
    std::string abnormal_type;
    std::string input_summary;
    ErrorCode expected_code{ErrorCode::INTERNAL_ERROR};
    ErrorCode actual_code{ErrorCode::INTERNAL_ERROR};
    std::string error_message;
    std::string backend_error_code{"NA"};
    std::string backend_error_name{"NA"};
    std::string backend_error_message{"NA"};
    bool safe_exit{};
};

inline AbnormalCaseResult make_abnormal_case_result(std::string test_object,
    std::string case_id, std::string abnormal_type, std::string input_summary,
    ErrorCode expected_code, const Error& actual_error)
{
    return {std::move(test_object), std::move(case_id), std::move(abnormal_type),
        std::move(input_summary), expected_code, actual_error.code, actual_error.message,
        actual_error.backend.code, actual_error.backend.name, actual_error.backend.message,
        actual_error.safe_exit};
}

inline bool abnormal_case_passed(const AbnormalCaseResult& result) noexcept
{
    return result.actual_code == result.expected_code &&
        result.actual_code != ErrorCode::SUCCESS && !result.error_message.empty() &&
        !result.backend_error_code.empty() && !result.backend_error_name.empty() &&
        !result.backend_error_message.empty() && result.safe_exit;
}

inline std::string format_abnormal_case(const AbnormalCaseResult& result)
{
    std::ostringstream output;
    output << "test_object=" << result.test_object
           << ",case_id=" << result.case_id
           << ",abnormal_type=" << result.abnormal_type
           << ",input_summary=" << result.input_summary
           << ",expected_code=" << error_code_name(result.expected_code)
           << ",actual_code=" << error_code_name(result.actual_code)
           << ",error_message=" << result.error_message
           << ",backend_error_code=" << result.backend_error_code
           << ",backend_error_name=" << result.backend_error_name
           << ",backend_error_message=" << result.backend_error_message
           << ",safe_exit=" << (result.safe_exit ? "true" : "false")
           << ',' << (abnormal_case_passed(result) ? "PASS" : "FAIL");
    return output.str();
}

}  // namespace cusignal
