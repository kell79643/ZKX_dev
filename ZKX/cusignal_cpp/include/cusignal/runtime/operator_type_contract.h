#pragma once

#include <cusignal/runtime/operator_type_dispatch.h>

#include <array>

namespace cusignal::type_contract {

enum class OutputKind {
    same_as_input,
    same_as_input_or_float64_scalar,
    same_as_input_or_complex_float32,
    float64_or_complex_float64,
    detrend_mode_dependent,
    decimate_filter_dependent,
    resample_poly_configuration_dependent,
    int32_float64_else_float32,
    int32_complex64_else_complex32,
    spectrogram_mode_dependent,
    float_input_float32_integer_float64,
    float_input_complex32_integer_complex64,
    float32,
    float64,
    complex_float32,
    complex_float64,
    int32,
    int64,
    float32_and_bool
};
enum class ToleranceClass { exact, scalar, pointwise, filtering, fft };

struct OperatorContract {
    type_dispatch::OperatorId id;
    OutputKind output;
    ToleranceClass tolerance;
    const char* compute;
};

struct ComparisonPolicy {
    float absolute;
    float relative;
    bool exact;
    bool require_finite;
    bool zero_metric_requires_three_reviews;
};

inline constexpr std::array<OperatorContract, 53> operator_contracts{{
    {type_dispatch::OperatorId::cubic, OutputKind::same_as_input, ToleranceClass::pointwise, "fp32"},
    {type_dispatch::OperatorId::gauss_spline, OutputKind::same_as_input, ToleranceClass::pointwise, "fp32"},
    {type_dispatch::OperatorId::quadratic, OutputKind::same_as_input, ToleranceClass::pointwise, "fp32"},
    {type_dispatch::OperatorId::correlate, OutputKind::same_as_input, ToleranceClass::fft, "int32-modular-or-fp32-direct/dlfft+same-width-wrap"},
    {type_dispatch::OperatorId::correlate2d, OutputKind::same_as_input, ToleranceClass::filtering, "boundary-aware-int32-modular-or-fp32-direct+same-width-wrap"},
    {type_dispatch::OperatorId::fm_demod, OutputKind::float32, ToleranceClass::pointwise, "complex-components-to-fp32-angle+unwrap-local-delta"},
    {type_dispatch::OperatorId::kalman_filter, OutputKind::float32, ToleranceClass::filtering, "five-input-to-fp32-state+fixed-inverse+joseph-update"},
    {type_dispatch::OperatorId::firwin, OutputKind::float64, ToleranceClass::filtering, "fp32+direct-ieee754-storage-bits-without-fp64-arithmetic"},
    {type_dispatch::OperatorId::firwin2, OutputKind::float64, ToleranceClass::filtering, "fp32/dlfft+host-fp64-finalize"},
    {type_dispatch::OperatorId::channelize_poly, OutputKind::complex_float32, ToleranceClass::fft, "fp32"},
    {type_dispatch::OperatorId::detrend, OutputKind::detrend_mode_dependent, ToleranceClass::filtering, "fp32+boundary-convert"},
    {type_dispatch::OperatorId::firfilter, OutputKind::float32, ToleranceClass::filtering, "fp32-axis+zi+zf"},
    {type_dispatch::OperatorId::firfilter2, OutputKind::int32_float64_else_float32, ToleranceClass::filtering, "fp32-pad+zi+bidirectional+conditional-device-int32-fp64-storage"},
    {type_dispatch::OperatorId::freq_shift, OutputKind::complex_float64, ToleranceClass::filtering, "complex-fp32+host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::hilbert, OutputKind::float_input_complex32_integer_complex64, ToleranceClass::fft, "complex-fp32+integer-host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::hilbert2, OutputKind::float_input_complex32_integer_complex64, ToleranceClass::fft, "complex-fp32-square-mask+integer-host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::lfilter_zi, OutputKind::float64, ToleranceClass::filtering, "fp32-linear-solve+host-fp64-finalize"},
    {type_dispatch::OperatorId::sosfilt, OutputKind::float32, ToleranceClass::filtering, "fp32"},
    {type_dispatch::OperatorId::wiener, OutputKind::float64, ToleranceClass::filtering, "fp32_host_widen"},
    {type_dispatch::OperatorId::decimate, OutputKind::decimate_filter_dependent, ToleranceClass::filtering, "fp32-axis-upfirdn+conditional-host-fp64-finalize"},
    {type_dispatch::OperatorId::resample, OutputKind::float_input_float32_integer_float64, ToleranceClass::fft, "complex-fp32-axis-dlfft+integer-host-fp64-finalize"},
    {type_dispatch::OperatorId::resample_poly, OutputKind::resample_poly_configuration_dependent, ToleranceClass::filtering, "fp32-axis-upfirdn+conditional-host-fp64-finalize"},
    {type_dispatch::OperatorId::upfirdn, OutputKind::int32_float64_else_float32, ToleranceClass::filtering, "fp32-axis-polyphase+host-int32-fp64-finalize"},
    {type_dispatch::OperatorId::argrelextrema, OutputKind::int64, ToleranceClass::exact, "native-fp32/int32-or-exact-fp32/int32-extension+rank-coordinate-tuple"},
    {type_dispatch::OperatorId::ambgfun, OutputKind::float_input_float32_integer_float64, ToleranceClass::fft, "complex-components-to-fp32-normalize+dlfft+cut-dependent-host-fp64-finalize"},
    {type_dispatch::OperatorId::ca_cfar, OutputKind::float32_and_bool, ToleranceClass::filtering, "fp32-threshold+bool-detection"},
    {type_dispatch::OperatorId::cfar_alpha, OutputKind::float64, ToleranceClass::scalar, "fp32-device+host-fp64-finalize"},
    {type_dispatch::OperatorId::pulse_compression, OutputKind::float_input_complex32_integer_complex64, ToleranceClass::fft, "complex-fp32-dlfft+conditional-host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::pulse_doppler, OutputKind::float_input_complex32_integer_complex64, ToleranceClass::fft, "complex-fp32-dlfft+conditional-host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::csd, OutputKind::int32_complex64_else_complex32, ToleranceClass::fft, "complex-fp32/dlfft+int32-host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::istft, OutputKind::int32_float64_else_float32, ToleranceClass::fft, "complex-fp32/dlfft+int32-host-fp64-finalize"},
    {type_dispatch::OperatorId::lombscargle, OutputKind::float64, ToleranceClass::fft, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::spectrogram, OutputKind::spectrogram_mode_dependent, ToleranceClass::fft, "complex-fp32/dlfft+mode-postprocess+int32-host-fp64-finalize"},
    {type_dispatch::OperatorId::stft, OutputKind::int32_complex64_else_complex32, ToleranceClass::fft, "complex-fp32/dlfft+int32-host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::vectorstrength, OutputKind::float64, ToleranceClass::scalar, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::chirp, OutputKind::same_as_input_or_complex_float32, ToleranceClass::pointwise, "fp32-real-or-complex"},
    {type_dispatch::OperatorId::sawtooth, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::square, OutputKind::float64, ToleranceClass::exact, "fp32-predicate+direct-ieee754-storage-bits-without-fp64-conversion-or-arithmetic"},
    {type_dispatch::OperatorId::gausspulse, OutputKind::same_as_input_or_float64_scalar, ToleranceClass::pointwise, "fp32-arrays+host-fp64-cutoff"},
    {type_dispatch::OperatorId::unit_impulse, OutputKind::float64, ToleranceClass::exact, "no-business-input+direct-ieee754-storage-bits-without-fp64-conversion-or-arithmetic"},
    {type_dispatch::OperatorId::cwt, OutputKind::float64_or_complex_float64, ToleranceClass::filtering, "fp32-convolution+host-fp64-finalize"},
    {type_dispatch::OperatorId::morlet, OutputKind::complex_float64, ToleranceClass::pointwise, "complex-fp32+host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::morlet2, OutputKind::complex_float64, ToleranceClass::pointwise, "complex-fp32+host-complex-fp64-finalize"},
    {type_dispatch::OperatorId::qmf, OutputKind::int64, ToleranceClass::exact, "integer-index-kernel"},
    {type_dispatch::OperatorId::ricker, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::chebwin, OutputKind::float64, ToleranceClass::fft, "fp32-custom+host-fp64-finalize"},
    {type_dispatch::OperatorId::general_cosine, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::general_gaussian, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::hamming, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::kaiser, OutputKind::float64, ToleranceClass::pointwise, "fp32-custom-bessel+host-fp64-finalize"},
    {type_dispatch::OperatorId::parzen, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::taylor, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
    {type_dispatch::OperatorId::triang, OutputKind::float64, ToleranceClass::pointwise, "fp32+host-fp64-finalize"},
}};

inline constexpr ComparisonPolicy comparison_policy(
    const OperatorContract& contract, type_dispatch::TargetType input)
{
    const bool integral = input == type_dispatch::TargetType::int32 ||
        input == type_dispatch::TargetType::int16 || input == type_dispatch::TargetType::int8;
    if (contract.tolerance == ToleranceClass::exact ||
        (integral && (contract.output == OutputKind::same_as_input ||
            contract.output == OutputKind::same_as_input_or_float64_scalar))) {
        return {0.0F, 0.0F, true, false,
            contract.id == type_dispatch::OperatorId::square ||
            contract.id == type_dispatch::OperatorId::unit_impulse};
    }
    const float absolute_scale = input == type_dispatch::TargetType::fp16
        ? 1.5F
        : 1.0F;
    const float relative = input == type_dispatch::TargetType::fp16
        ? 1.0e-3F
        : 1.0e-4F;
    float absolute = 0.0F;
    switch (contract.tolerance) {
    case ToleranceClass::scalar: absolute = 1.0e-4F; break;
    case ToleranceClass::pointwise: absolute = 3.0e-2F; break;
    case ToleranceClass::filtering: absolute = 4.0e-2F; break;
    case ToleranceClass::fft: absolute = 6.0e-2F; break;
    case ToleranceClass::exact: break;
    }
    return {
        absolute * absolute_scale,
        relative,
        false,
        true,
        true};
}

inline constexpr const char* input_quantization_name(type_dispatch::TargetType input)
{
    switch (input) {
    case type_dispatch::TargetType::fp32: return "identity-fp32";
    case type_dispatch::TargetType::fp16: return "fp16-load-to-fp32";
    case type_dispatch::TargetType::int32: return "int32-exact-load-to-fp32";
    case type_dispatch::TargetType::int16: return "int16-exact-load-to-fp32";
    case type_dispatch::TargetType::int8: return "int8-exact-load-to-fp32";
    }
    return "unknown";
}

inline constexpr const char* input_quantization_name(
    const OperatorContract& contract,
    type_dispatch::TargetType input)
{
    if (contract.id == type_dispatch::OperatorId::unit_impulse)
        return "none-no-business-input";
    if (contract.id == type_dispatch::OperatorId::argrelextrema) {
        switch (input) {
        case type_dispatch::TargetType::fp32: return "identity-fp32-compare";
        case type_dispatch::TargetType::fp16: return "fp16-exact-to-fp32-compare";
        case type_dispatch::TargetType::int32: return "identity-int32-compare";
        case type_dispatch::TargetType::int16: return "int16-exact-to-int32-compare";
        case type_dispatch::TargetType::int8: return "int8-exact-to-int32-compare";
        }
    }
    if (contract.id == type_dispatch::OperatorId::ambgfun) {
        switch (input) {
        case type_dispatch::TargetType::fp32: return "complex-fp32-native";
        case type_dispatch::TargetType::fp16: return "complex-fp16-components-to-complex-fp32";
        case type_dispatch::TargetType::int32: return "complex-int32-components-to-complex-fp32";
        case type_dispatch::TargetType::int16: return "complex-int16-components-to-complex-fp32";
        case type_dispatch::TargetType::int8: return "complex-int8-components-to-complex-fp32";
        }
    }
    return input_quantization_name(input);
}

inline constexpr const char* output_conversion_name(
    OutputKind output,
    type_dispatch::TargetType input)
{
    if (output == OutputKind::float32) return "fp32";
    if (output == OutputKind::float64) return "host-widen-fp64";
    if (output == OutputKind::complex_float32) return "complex-fp32";
    if (output == OutputKind::complex_float64) return "host-widen-complex-fp64";
    if (output == OutputKind::int32) return "exact-int32";
    if (output == OutputKind::int64) return "exact-int64";
    if (output == OutputKind::float32_and_bool) return "fp32+exact-bool-detection";
    if (output == OutputKind::same_as_input_or_float64_scalar)
        return "same-as-input-array-or-host-fp64-scalar";
    if (output == OutputKind::same_as_input_or_complex_float32)
        return "same-as-input-real-or-complex-fp32";
    if (output == OutputKind::float64_or_complex_float64)
        return "host-widen-fp64-or-complex-fp64";
    if (output == OutputKind::detrend_mode_dependent)
        return "fp16-constant-or-fp32-or-host-widen-fp64";
    if (output == OutputKind::decimate_filter_dependent)
        return "default-host-widen-fp64-or-custom-fp32/int32-fp64";
    if (output == OutputKind::resample_poly_configuration_dependent)
        return "identity-same-as-input-or-default/custom-fp32/fp64";
    if (output == OutputKind::int32_float64_else_float32)
        return input == type_dispatch::TargetType::int32
            ? "host-widen-fp64" : "fp32";
    if (output == OutputKind::int32_complex64_else_complex32)
        return input == type_dispatch::TargetType::int32
            ? "host-widen-complex-fp64" : "complex-fp32";
    if (output == OutputKind::spectrogram_mode_dependent)
        return input == type_dispatch::TargetType::int32
            ? "host-widen-fp64-or-complex-fp64"
            : "fp32-or-complex-fp32";
    if (output == OutputKind::float_input_float32_integer_float64)
        return input == type_dispatch::TargetType::fp32 ||
               input == type_dispatch::TargetType::fp16
            ? "fp32" : "host-widen-fp64";
    if (output == OutputKind::float_input_complex32_integer_complex64)
        return input == type_dispatch::TargetType::fp32 ||
               input == type_dispatch::TargetType::fp16
            ? "complex-fp32" : "host-widen-complex-fp64";
    switch (input) {
    case type_dispatch::TargetType::fp32: return "fp32";
    case type_dispatch::TargetType::fp16: return "round-to-fp16";
    case type_dispatch::TargetType::int32: return "round-saturate-int32";
    case type_dispatch::TargetType::int16: return "round-saturate-int16";
    case type_dispatch::TargetType::int8: return "round-saturate-int8";
    }
    return "unknown";
}

inline constexpr const char* output_conversion_name(
    const OperatorContract& contract, OutputKind actual_output,
    type_dispatch::TargetType input)
{
    if (contract.id == type_dispatch::OperatorId::firwin ||
        contract.id == type_dispatch::OperatorId::square ||
        contract.id == type_dispatch::OperatorId::unit_impulse) {
        return "direct-ieee754-storage-bits";
    }
    if (contract.id == type_dispatch::OperatorId::firfilter2 &&
        input == type_dispatch::TargetType::int32) {
        return "conditional-device-ieee754-storage-bits";
    }
    return output_conversion_name(actual_output, input);
}

inline constexpr const char* shape_policy_name(const OperatorContract& contract)
{
    if (contract.id == type_dispatch::OperatorId::argrelextrema)
        return "value-dependent-length-rank-int64-coordinate-tuple";
    if (contract.id == type_dispatch::OperatorId::ambgfun)
        return "mode-and-input-shape-before-value-2d-or-delay/doppler-vector";
    if (contract.id == type_dispatch::OperatorId::ca_cfar)
        return "input-shape-before-value-rank0/rank1/rank2-threshold+bool";
    if (contract.output == OutputKind::int32) return "indices-shape-before-value";
    if (contract.output == OutputKind::float32_and_bool)
        return "dual-output-shapes-before-value";
    if (contract.id == type_dispatch::OperatorId::cfar_alpha ||
        contract.id == type_dispatch::OperatorId::vectorstrength)
        return "scalar-shape-before-value";
    return "operator-defined-shape-before-value";
}

inline constexpr const char* output_kind_name(OutputKind output)
{
    switch (output) {
    case OutputKind::same_as_input: return "same-as-input";
    case OutputKind::same_as_input_or_float64_scalar:
        return "same-as-input-array-or-fp64-scalar";
    case OutputKind::same_as_input_or_complex_float32:
        return "same-as-input-real-or-complex-fp32";
    case OutputKind::float64_or_complex_float64:
        return "fp64-or-complex-fp64";
    case OutputKind::detrend_mode_dependent:
        return "detrend-mode-dependent-fp16-fp32-fp64";
    case OutputKind::decimate_filter_dependent:
        return "decimate-filter-dependent-fp32-fp64";
    case OutputKind::resample_poly_configuration_dependent:
        return "resample-poly-identity-or-configuration-dependent";
    case OutputKind::int32_float64_else_float32:
        return "int32-fp64-else-fp32";
    case OutputKind::int32_complex64_else_complex32:
        return "int32-complex-fp64-else-complex-fp32";
    case OutputKind::spectrogram_mode_dependent:
        return "spectrogram-mode-and-input-dependent";
    case OutputKind::float_input_float32_integer_float64:
        return "float-input-fp32-integer-fp64";
    case OutputKind::float_input_complex32_integer_complex64:
        return "float-input-complex-fp32-integer-complex-fp64";
    case OutputKind::float32: return "fp32";
    case OutputKind::float64: return "fp64";
    case OutputKind::complex_float32: return "complex-fp32";
    case OutputKind::complex_float64: return "complex-fp64";
    case OutputKind::int32: return "int32";
    case OutputKind::int64: return "int64";
    case OutputKind::float32_and_bool: return "fp32+bool";
    }
    return "unknown";
}

}  // namespace cusignal::type_contract
