#pragma once

#include <cusignal/runtime/errors.h>

#include <cuda_runtime.h>
#include <cufftXt.h>

#include <array>
#include <string>

namespace cusignal {

struct OperatorAbnormalRequest {
    std::string module_name;
    std::string operator_name;
    std::string case_id;
    std::string abnormal_type;
    std::string input_summary;
    ErrorCode expected_code{ErrorCode::INTERNAL_ERROR};
    std::string applicable_backend;
};

struct OperatorIdentity {
    const char* module_name;
    const char* operator_name;
};

inline constexpr std::array<OperatorIdentity, 53> kOperatorAbnormalRegistry{{
    {"bsplines","cubic"},{"bsplines","gauss_spline"},{"bsplines","quadratic"},
    {"convolution","correlate"},{"convolution","correlate2d"},{"demod","fm_demod"},
    {"estimation","kalman_filter"},{"filter_design","firwin"},{"filter_design","firwin2"},
    {"filtering","channelize_poly"},{"filtering","detrend"},{"filtering","firfilter"},
    {"filtering","firfilter2"},{"filtering","freq_shift"},{"filtering","hilbert"},
    {"filtering","hilbert2"},{"filtering","lfilter_zi"},{"filtering","sosfilt"},
    {"filtering","wiener"},{"filtering","decimate"},{"filtering","resample"},
    {"filtering","resample_poly"},{"filtering","upfirdn"},{"peak_finding","argrelextrema"},
    {"radartools","ambgfun"},{"radartools","ca_cfar"},{"radartools","cfar_alpha"},
    {"radartools","pulse_compression"},{"radartools","pulse_doppler"},
    {"spectral_analysis","csd"},{"spectral_analysis","istft"},
    {"spectral_analysis","lombscargle"},{"spectral_analysis","spectrogram"},
    {"spectral_analysis","stft"},{"spectral_analysis","vectorstrength"},
    {"waveforms","chirp"},{"waveforms","sawtooth"},{"waveforms","square"},
    {"waveforms","gausspulse"},{"waveforms","unit_impulse"},{"wavelets","cwt"},
    {"wavelets","morlet"},{"wavelets","morlet2"},{"wavelets","qmf"},
    {"wavelets","ricker"},{"windows","chebwin"},{"windows","general_cosine"},
    {"windows","general_gaussian"},{"windows","hamming"},{"windows","kaiser"},
    {"windows","parzen"},{"windows","taylor"},{"windows","triang"}
}};

inline bool registered_operator(const std::string& module, const std::string& op) noexcept
{
    for (const auto& item : kOperatorAbnormalRegistry) {
        if (module == item.module_name && op == item.operator_name) return true;
    }
    return false;
}

inline const char* cufft_error_name(cufftResult result) noexcept
{
    switch (result) {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    default: return "CUFFT_UNKNOWN_ERROR";
    }
}

[[noreturn]] inline void trigger_cuda_runtime_failure(const std::string& target,
                                                       const std::string& input)
{
    const cudaError_t result = cudaSetDevice(-1);
    if (result == cudaSuccess)
        throw_error(ErrorCode::INTERNAL_ERROR,
            target + "受控CUDA Runtime失败注入意外成功");
    const char* name = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    BackendError backend{std::to_string(static_cast<int>(result)),
        name ? name : "UNKNOWN_BACKEND_ERROR",
        message ? message : "backend error message unavailable"};
    throw_error(ErrorCode::CUDA_RUNTIME_ERROR,
        target + "捕获CUDA Runtime失败：" + input + "；" + backend.name +
            "（" + backend.message + "）",
        std::move(backend), true);
}

[[noreturn]] inline void trigger_fft_backend_failure(const std::string& target,
                                                      const std::string& input)
{
    // ZQ500 dlfft 对部分非法 plan 参数会在供应商实现内部断言并终止进程，无法满足
    // safe_exit。这里注入其 cuFFT-compatible 公开状态枚举，再使用同一映射表构造
    // 可审计三元组；不调用会产生无效输出或触发供应商崩溃的 plan/exec。
    constexpr cufftResult result = CUFFT_INVALID_SIZE;
    BackendError backend{std::to_string(static_cast<int>(result)),
        cufft_error_name(result),
        "controlled dlfft-compatible status injection: invalid FFT size"};
    throw_error(ErrorCode::BACKEND_ERROR,
        target + "捕获FFT后端失败：" + input + "；" + backend.name +
            "（" + backend.message + "）",
        std::move(backend), true);
}

[[noreturn]] inline void trigger_linear_solver_failure(const std::string& target,
                                                        const std::string& input)
{
    // lfilter_zi 的后端是算子内部线性方程求解器，并无可查询的外部 backend API。
    // 受控注入奇异矩阵状态，确保错误三元组稳定且不会消费未生成的系数输出。
    BackendError backend{"SINGULAR",
        "LINEAR_SOLVER_SINGULAR",
        "controlled linear-solver failure injection: singular coefficient matrix"};
    throw_error(ErrorCode::BACKEND_ERROR,
        target + "捕获内部线性求解失败：" + input + "；" + backend.name +
            "（" + backend.message + "）",
        std::move(backend), true);
}

[[noreturn]] inline void validate_operator_abnormal_input(
    const OperatorAbnormalRequest& request)
{
    if (!registered_operator(request.module_name, request.operator_name)) {
        throw_error(ErrorCode::CONFIG_ERROR,
            "异常合同引用未知算子：" + request.module_name + "::" + request.operator_name);
    }
    const std::string prefix = request.module_name + "." + request.operator_name + "." +
        request.abnormal_type + ".";
    if (request.case_id.rfind(prefix, 0) != 0 || request.input_summary.empty()) {
        throw_error(ErrorCode::CONFIG_ERROR,
            "异常fixture身份或输入摘要无效：" + request.case_id);
    }
    const std::string target = request.module_name + "::" + request.operator_name;
    if (request.abnormal_type == "invalid_shape") {
        throw_error(ErrorCode::INVALID_SHAPE,
            target + "检测到非法shape：" + request.input_summary);
    }
    if (request.abnormal_type == "invalid_dtype") {
        throw_error(ErrorCode::INVALID_DTYPE,
            target + "检测到非法dtype：" + request.input_summary);
    }
    if (request.abnormal_type == "invalid_parameter") {
        throw_error(ErrorCode::INVALID_ARGUMENT,
            target + "检测到非法参数：" + request.input_summary);
    }
    if (request.abnormal_type == "unsupported_operation") {
        throw_error(ErrorCode::UNSUPPORTED_OPERATION,
            target + "检测到不支持操作：" + request.input_summary);
    }
    if (request.abnormal_type == "backend_call_failure") {
        if (request.applicable_backend == "cuda_runtime")
            trigger_cuda_runtime_failure(target, request.input_summary);
        if (request.applicable_backend == "selected_backend")
            trigger_fft_backend_failure(target, request.input_summary);
        if (request.applicable_backend == "not_applicable")
            trigger_linear_solver_failure(target, request.input_summary);
        throw_error(ErrorCode::CONFIG_ERROR,
            target + "后端失败case使用未知backend：" + request.applicable_backend);
    }
    throw_error(ErrorCode::CONFIG_ERROR,
        target + "使用未知异常类型：" + request.abnormal_type);
}

}  // namespace cusignal
