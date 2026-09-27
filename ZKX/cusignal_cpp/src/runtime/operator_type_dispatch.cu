#include <cusignal/runtime/operator_type_dispatch.h>

#include <cusignal/runtime/cuda_error.h>
#include <cusignal/runtime/kernel_launch.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cusignal::type_dispatch {
namespace {

constexpr const char* kOperatorNames[] = {
    "cubic", "gauss_spline", "quadratic", "correlate", "correlate2d",
    "fm_demod", "KalmanFilter", "firwin", "firwin2", "channelize_poly",
    "detrend", "firfilter", "firfilter2", "freq_shift", "hilbert", "hilbert2",
    "lfilter_zi", "sosfilt", "wiener", "decimate", "resample", "resample_poly",
    "upfirdn", "argrelextrema", "ambgfun", "ca_cfar", "cfar_alpha",
    "pulse_compression", "pulse_doppler", "csd", "istft", "lombscargle",
    "spectrogram", "stft", "vectorstrength", "chirp", "sawtooth", "square",
    "gausspulse", "unit_impulse", "cwt", "morlet", "morlet2", "qmf", "ricker",
    "chebwin", "general_cosine", "general_gaussian", "hamming", "kaiser", "parzen",
    "taylor", "triang",
};

constexpr bool is_fft_dispatch(OperatorId op)
{
    switch (op) {
    case OperatorId::correlate:
    case OperatorId::correlate2d:
    case OperatorId::firwin2:
    case OperatorId::channelize_poly:
    case OperatorId::hilbert:
    case OperatorId::hilbert2:
    case OperatorId::resample:
    case OperatorId::ambgfun:
    case OperatorId::pulse_compression:
    case OperatorId::pulse_doppler:
    case OperatorId::csd:
    case OperatorId::istft:
    case OperatorId::spectrogram:
    case OperatorId::stft:
        return true;
    default:
        return false;
    }
}

constexpr bool has_fixed_output_dtype(OperatorId op)
{
    switch (op) {
    case OperatorId::fm_demod:
    case OperatorId::channelize_poly:
    case OperatorId::freq_shift:
    case OperatorId::hilbert:
    case OperatorId::hilbert2:
    case OperatorId::argrelextrema:
    case OperatorId::ambgfun:
    case OperatorId::ca_cfar:
    case OperatorId::cfar_alpha:
    case OperatorId::pulse_compression:
    case OperatorId::pulse_doppler:
    case OperatorId::csd:
    case OperatorId::lombscargle:
    case OperatorId::spectrogram:
    case OperatorId::stft:
    case OperatorId::vectorstrength:
    case OperatorId::cwt:
    case OperatorId::morlet:
    case OperatorId::morlet2:
        return true;
    default:
        return false;
    }
}

template <typename InputT>
__device__ float load_as_float(InputT value)
{
    return static_cast<float>(value);
}

template <>
__device__ float load_as_float(__half value)
{
    return __half2float(value);
}

template <typename InputT>
__global__ void to_float_kernel(const InputT* input, float* output, std::size_t count)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = load_as_float(input[i]);
    }
}

template <typename OutputT>
__global__ void from_float_kernel(const float* input, OutputT* output, std::size_t count)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = static_cast<OutputT>(input[i]);
    }
}

template <>
__global__ void from_float_kernel(const float* input, __half* output, std::size_t count)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = __float2half_rn(input[i]);
    }
}

template <typename OutputT>
__global__ void quantize_saturate_kernel(
    const float* input,
    OutputT* output,
    std::size_t count,
    float scale,
    std::int32_t zero_point)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    float quantized = input[i] / scale + static_cast<float>(zero_point);
    const float low = static_cast<float>(std::numeric_limits<OutputT>::min());
    const float high = static_cast<float>(std::numeric_limits<OutputT>::max());
    // ZQ500 LLVM 15 cannot legalize the device-side nearbyint libcall. Clamp
    // first, then implement round-half-away-from-zero using only arithmetic
    // and the CUDA integer conversion; the cast truncates toward zero.
    quantized = quantized < low ? low : (quantized > high ? high : quantized);
    quantized += quantized >= 0.0 ? 0.5 : -0.5;
    output[i] = static_cast<OutputT>(quantized);
}

template <typename InputT>
DeviceArray<float> convert_to_float(const DeviceArray<InputT>& input)
{
    DeviceArray<float> output(input.size());
    if (!input.empty()) {
        cuda_utils::launch_1d_kernel(
            to_float_kernel<InputT>, input.size(), input.data(), output.data(), input.size());
        cuda_utils::synchronize_stream();
    }
    return output;
}

template <typename OutputT>
DeviceArray<OutputT> convert_from_float(const DeviceArray<float>& input)
{
    DeviceArray<OutputT> output(input.size());
    if (!input.empty()) {
        cuda_utils::launch_1d_kernel(
            from_float_kernel<OutputT>, input.size(), input.data(), output.data(), input.size());
        cuda_utils::synchronize_stream();
    }
    return output;
}

template <typename OutputT>
DeviceArray<OutputT> quantize_from_float(
    const DeviceArray<float>& input,
    IntegerOutputPolicy policy)
{
    if (!(policy.scale > 0.0) || !std::isfinite(policy.scale)) {
        throw std::invalid_argument("integer output scale must be finite and greater than zero");
    }
    DeviceArray<OutputT> output(input.size());
    if (!input.empty()) {
        cuda_utils::launch_1d_kernel(
            quantize_saturate_kernel<OutputT>,
            input.size(),
            input.data(),
            output.data(),
            input.size(),
            policy.scale,
            policy.zero_point);
        cuda_utils::synchronize_stream();
    }
    return output;
}

}  // namespace

static_assert(sizeof(kOperatorNames) / sizeof(kOperatorNames[0]) == operator_count(),
              "E3 operator registry must contain all 53 operators");

const char* operator_name(OperatorId op) noexcept
{
    const auto index = static_cast<std::size_t>(op);
    return index < operator_count() ? kOperatorNames[index] : "unknown";
}

const char* target_type_name(TargetType type) noexcept
{
    switch (type) {
    case TargetType::fp32: return "FP32";
    case TargetType::fp16: return "FP16";
    case TargetType::int32: return "INT32";
    case TargetType::int16: return "INT16";
    case TargetType::int8: return "INT8";
    }
    return "unknown";
}

DispatchPlan dispatch_plan(OperatorId op, TargetType type)
{
    if (static_cast<std::size_t>(op) >= operator_count()) {
        throw std::invalid_argument("operator id is outside the E3 registry");
    }
    if (type == TargetType::fp32 && op == OperatorId::kalman_filter) {
        return {ComputePath::native_fp32, true, "existing float GPU specialization"};
    }
    if (type == TargetType::fp32) {
        return {is_fft_dispatch(op) ? ComputePath::promote_to_fp32_fft
                                    : ComputePath::promote_to_fp32,
                !has_fixed_output_dtype(op),
                is_fft_dispatch(op) ? "typed input uses the FP32 complex FFT backend boundary"
                                    : "typed input uses the FP32 ZQ500 compute path"};
    }
    if (type == TargetType::fp16) {
        return {is_fft_dispatch(op) ? ComputePath::promote_to_fp32_fft
                                    : ComputePath::promote_to_fp32,
                !has_fixed_output_dtype(op),
                is_fft_dispatch(op) ? "the selected FFT backend uses FP32 complex input; promote FP16"
                                    : "load FP16 on device and accumulate in the verified compute path"};
    }
    if (has_fixed_output_dtype(op)) {
        return {is_fft_dispatch(op) ? ComputePath::promote_to_fp32_fft
                                    : ComputePath::promote_to_fp32,
                false,
                "integer business input is promoted to FP32; output dtype is fixed by cuSignal semantics"};
    }
    return {is_fft_dispatch(op) ? ComputePath::promote_to_fp32_fft
                                : ComputePath::quantized_via_fp32,
            true,
            is_fft_dispatch(op)
                ? "promote input to FP32 complex and store the cuSignal same-type output"
                : "compute in FP32 and apply the typed wrapper's saturating same-type store"};
}

DeviceArray<float> to_fp32_compute(const DeviceArray<float>& input) { return convert_to_float(input); }
DeviceArray<float> to_fp32_compute(const DeviceArray<__half>& input) { return convert_to_float(input); }
DeviceArray<float> to_fp32_compute(const DeviceArray<std::int32_t>& input) { return convert_to_float(input); }
DeviceArray<float> to_fp32_compute(const DeviceArray<std::int16_t>& input) { return convert_to_float(input); }
DeviceArray<float> to_fp32_compute(const DeviceArray<std::int8_t>& input) { return convert_to_float(input); }

DeviceArray<float> from_fp32_fp32(const DeviceArray<float>& input) { return convert_from_float<float>(input); }
DeviceArray<__half> from_fp32_fp16(const DeviceArray<float>& input) { return convert_from_float<__half>(input); }
DeviceArray<std::int32_t> from_fp32_int32(const DeviceArray<float>& input, IntegerOutputPolicy policy)
{
    return quantize_from_float<std::int32_t>(input, policy);
}
DeviceArray<std::int16_t> from_fp32_int16(const DeviceArray<float>& input, IntegerOutputPolicy policy)
{
    return quantize_from_float<std::int16_t>(input, policy);
}
DeviceArray<std::int8_t> from_fp32_int8(const DeviceArray<float>& input, IntegerOutputPolicy policy)
{
    return quantize_from_float<std::int8_t>(input, policy);
}

}  // namespace cusignal::type_dispatch
