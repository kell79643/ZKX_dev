#pragma once

#include <cusignal/runtime/device_array.h>

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace cusignal::type_dispatch {

enum class TargetType : std::uint8_t {
    fp32,
    fp16,
    int32,
    int16,
    int8,
};

enum class ComputePath : std::uint8_t {
    native_fp32,
    promote_to_fp32,
    promote_to_fp32_fft,
    quantized_via_fp32,
};

enum class OperatorId : std::uint8_t {
    cubic,
    gauss_spline,
    quadratic,
    correlate,
    correlate2d,
    fm_demod,
    kalman_filter,
    firwin,
    firwin2,
    channelize_poly,
    detrend,
    firfilter,
    firfilter2,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter_zi,
    sosfilt,
    wiener,
    decimate,
    resample,
    resample_poly,
    upfirdn,
    argrelextrema,
    ambgfun,
    ca_cfar,
    cfar_alpha,
    pulse_compression,
    pulse_doppler,
    csd,
    istft,
    lombscargle,
    spectrogram,
    stft,
    vectorstrength,
    chirp,
    sawtooth,
    square,
    gausspulse,
    unit_impulse,
    cwt,
    morlet,
    morlet2,
    qmf,
    ricker,
    chebwin,
    general_cosine,
    general_gaussian,
    hamming,
    kaiser,
    parzen,
    taylor,
    triang,
    count,
};

struct DispatchPlan {
    ComputePath path;
    bool output_dtype_follows_input;
    const char* reason;
};

struct IntegerOutputPolicy {
    float scale;
    std::int32_t zero_point;
};

constexpr std::size_t operator_count()
{
    return static_cast<std::size_t>(OperatorId::count);
}

const char* operator_name(OperatorId op) noexcept;
const char* target_type_name(TargetType type) noexcept;
DispatchPlan dispatch_plan(OperatorId op, TargetType type);

DeviceArray<float> to_fp32_compute(const DeviceArray<float>& input);
DeviceArray<float> to_fp32_compute(const DeviceArray<__half>& input);
DeviceArray<float> to_fp32_compute(const DeviceArray<std::int32_t>& input);
DeviceArray<float> to_fp32_compute(const DeviceArray<std::int16_t>& input);
DeviceArray<float> to_fp32_compute(const DeviceArray<std::int8_t>& input);
// FP64 is result storage, not a sixth formal GPU compute input.  Call
// cuda_utils::narrow_fp64_device_storage_to_fp32_on_host explicitly when an
// FP64 result must feed another operator.
DeviceArray<float> to_fp32_compute(const DeviceArray<double>& input) = delete;

DeviceArray<float> from_fp32_fp32(const DeviceArray<float>& input);
DeviceArray<__half> from_fp32_fp16(const DeviceArray<float>& input);
DeviceArray<std::int32_t> from_fp32_int32(
    const DeviceArray<float>& input,
    IntegerOutputPolicy policy);
DeviceArray<std::int16_t> from_fp32_int16(
    const DeviceArray<float>& input,
    IntegerOutputPolicy policy);
DeviceArray<std::int8_t> from_fp32_int8(
    const DeviceArray<float>& input,
    IntegerOutputPolicy policy);

}  // namespace cusignal::type_dispatch
