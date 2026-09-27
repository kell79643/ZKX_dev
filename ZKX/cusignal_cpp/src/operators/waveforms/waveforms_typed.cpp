#include <cusignal/operators/waveforms/waveforms_typed.h>
#include "waveform_math.h"
#include <cusignal/runtime/host_output_finalize.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cusignal {
namespace {
constexpr float pi = 3.14159265358979323846F;

float wrap_two_pi(float value)
{
    constexpr float two_pi = 2.0F * pi;
    float wrapped = std::fmod(value, two_pi);
    if (wrapped < 0.0F) wrapped += two_pi;
    return wrapped;
}
}

template <class T>
void sawtooth_fp32_compute_device(
    const DeviceArray<T>&, DeviceArray<float>&, float);
template <class T>
void sawtooth_fp32_compute_device(
    const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<float>&);
template <class T>
void sawtooth_broadcast_fp32_compute_device(
    const DeviceArray<T>&, const DeviceArray<T>&,
    const DeviceArray<std::size_t>&, const DeviceArray<std::size_t>&,
    DeviceArray<float>&);
template <class T>
void square_fp64_storage_device(
    const DeviceArray<T>&, void*, std::size_t, float);
template <class T>
void square_fp64_storage_device(
    const DeviceArray<T>&, const DeviceArray<T>&, void*, std::size_t);
template <class T>
void square_broadcast_fp64_storage_device(
    const DeviceArray<T>&, const DeviceArray<T>&,
    const DeviceArray<std::size_t>&, const DeviceArray<std::size_t>&,
    void*, std::size_t);
void unit_impulse_fp64_storage_device(void*, std::size_t, int);

namespace {
constexpr double waveform_pi64 = 3.141592653589793238462643383279502884;
constexpr double waveform_two_pi64 = 2.0 * waveform_pi64;

double sawtooth_value(double phase_input, double width)
{
    if (width < 0.0 || width > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double phase = std::fmod(phase_input, waveform_two_pi64);
    if (phase < 0.0) phase += waveform_two_pi64;
    if (width == 0.0) return 1.0 - phase / waveform_pi64;
    if (width == 1.0) return phase / waveform_pi64 - 1.0;
    if (phase < width * waveform_two_pi64) {
        return phase / (waveform_pi64 * width) - 1.0;
    }
    return (waveform_pi64 * (width + 1.0) - phase) /
        (waveform_pi64 * (1.0 - width));
}

double square_value(double phase_input, double duty)
{
    if (duty < 0.0 || duty > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double phase = std::fmod(phase_input, waveform_two_pi64);
    if (phase < 0.0) phase += waveform_two_pi64;
    return phase < duty * waveform_two_pi64 ? 1.0 : -1.0;
}

struct BroadcastPlan {
    std::vector<int> shape;
    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
};

std::size_t checked_shape_size(const std::vector<int>& shape)
{
    std::size_t size = 1;
    bool zero = false;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument("waveform broadcast shape");
        if (extent == 0) {
            zero = true;
            continue;
        }
        if (zero) continue;
        const auto value = static_cast<std::size_t>(extent);
        if (size > std::numeric_limits<std::size_t>::max() / value)
            throw std::invalid_argument("waveform broadcast shape overflow");
        size *= value;
    }
    return zero ? 0 : size;
}

BroadcastPlan make_broadcast_plan(
    const std::vector<int>& left_shape,
    const std::vector<int>& right_shape,
    std::size_t left_size,
    std::size_t right_size)
{
    if (checked_shape_size(left_shape) != left_size
        || checked_shape_size(right_shape) != right_size)
        throw std::invalid_argument("waveform broadcast data/shape mismatch");

    BroadcastPlan plan;
    const std::size_t rank = std::max(left_shape.size(), right_shape.size());
    plan.shape.assign(rank, 1);
    std::vector<int> left(rank, 1), right(rank, 1);
    std::copy(left_shape.begin(), left_shape.end(), left.end() - left_shape.size());
    std::copy(right_shape.begin(), right_shape.end(), right.end() - right_shape.size());
    for (std::size_t axis = 0; axis < rank; ++axis) {
        const int a = left[axis], b = right[axis];
        if (a != b && a != 1 && b != 1)
            throw std::invalid_argument("waveform inputs are not broadcastable");
        plan.shape[axis] = a == 1 ? b : a;
    }

    const std::size_t output_size = checked_shape_size(plan.shape);
    plan.left_indices.resize(output_size);
    plan.right_indices.resize(output_size);
    std::vector<std::size_t> left_strides(rank, 1), right_strides(rank, 1);
    for (std::size_t axis = rank; axis-- > 1;) {
        left_strides[axis - 1] = left_strides[axis] * static_cast<std::size_t>(left[axis]);
        right_strides[axis - 1] = right_strides[axis] * static_cast<std::size_t>(right[axis]);
    }
    for (std::size_t linear = 0; linear < output_size; ++linear) {
        std::size_t remainder = linear, left_index = 0, right_index = 0;
        for (std::size_t axis = rank; axis-- > 0;) {
            const auto extent = static_cast<std::size_t>(plan.shape[axis]);
            const std::size_t coordinate = remainder % extent;
            remainder /= extent;
            if (left[axis] != 1) left_index += coordinate * left_strides[axis];
            if (right[axis] != 1) right_index += coordinate * right_strides[axis];
        }
        plan.left_indices[linear] = left_index;
        plan.right_indices[linear] = right_index;
    }
    return plan;
}

float gausspulse_attenuation(float fc, float bw, float bwr)
{
    if (fc < 0.0F || bw <= 0.0F || bwr >= 0.0F) {
        throw std::invalid_argument("gausspulse parameters");
    }
    return waveforms_detail::gausspulse_attenuation_fp32(fc, bw, bwr);
}

waveforms_detail::ChirpMethod parse_cpu_chirp_method(const std::string& method)
{
    if (method == "linear" || method == "lin" || method == "li")
        return waveforms_detail::ChirpMethod::Linear;
    if (method == "quadratic" || method == "quad" || method == "q")
        return waveforms_detail::ChirpMethod::Quadratic;
    if (method == "logarithmic" || method == "log" || method == "lo")
        return waveforms_detail::ChirpMethod::Logarithmic;
    if (method == "hyperbolic" || method == "hyp")
        return waveforms_detail::ChirpMethod::Hyperbolic;
    throw std::invalid_argument("chirp: method must be linear, quadratic, logarithmic, or hyperbolic");
}

float cpu_chirp_phase(
    float time, float f0, float t1, float f1,
    waveforms_detail::ChirpMethod method, bool vertex_zero)
{
    return waveforms_detail::stable_chirp_phase(
        time, f0, t1, f1, method, vertex_zero);
}
}

template <class T>
std::vector<T> chirp_typed_cpu(
    const std::vector<T>& time, float f0, float t1, float f1, float phase)
{
    return chirp_typed_cpu(
        time, f0, t1, f1, std::string("linear"), phase, true);
}

template <class T>
std::vector<T> chirp_typed_cpu(
    const std::vector<T>& time, float f0, float t1, float f1,
    const std::string& method, float phase, bool vertex_zero)
{
    const auto parsed = parse_cpu_chirp_method(method);
    if (parsed == waveforms_detail::ChirpMethod::Logarithmic && f0 * f1 <= 0.0F)
        throw std::invalid_argument("logarithmic chirp frequencies");
    if (parsed == waveforms_detail::ChirpMethod::Hyperbolic && (f0 == 0.0F || f1 == 0.0F))
        throw std::invalid_argument("hyperbolic chirp frequencies");
    std::vector<T> out(time.size());
    const float phase_offset = phase * pi / 180.0F;
    for (std::size_t i = 0; i < time.size(); ++i) {
        const float x = detail::SimpleSignalTypePolicy<T>::load(time[i]);
        const float angle = cpu_chirp_phase(
            x, f0, t1, f1, parsed, vertex_zero) + phase_offset;
        out[i] = detail::SimpleSignalTypePolicy<T>::store(std::cos(angle));
    }
    return out;
}

template <class T>
std::vector<ComplexFloat> chirp_complex_typed_cpu(
    const std::vector<T>& time, float f0, float t1, float f1, float phase)
{
    std::vector<ComplexFloat> out(time.size());
    const float phase_offset = phase * pi / 180.0F;
    for (std::size_t i = 0; i < time.size(); ++i) {
        const float x = detail::SimpleSignalTypePolicy<T>::load(time[i]);
        const float angle = cpu_chirp_phase(
            x, f0, t1, f1, waveforms_detail::ChirpMethod::Linear, true) + phase_offset;
        out[i] = ComplexFloat{std::cos(angle), std::sin(angle)};
    }
    return out;
}

template <class T>
std::vector<T> gausspulse_typed_cpu(const std::vector<T>& time, float fc, float bw)
{
    return gausspulse_typed_cpu(time, fc, bw, -6.0F, false, false).in_phase;
}

template <class T>
GausspulseTypedResult<T> gausspulse_typed_cpu(
    const std::vector<T>& time, float fc, float bw, float bwr,
    bool retquad, bool retenv)
{
    const float attenuation = gausspulse_attenuation(fc, bw, bwr);
    GausspulseTypedResult<T> result;
    result.in_phase.resize(time.size());
    if (retquad) result.quadrature.resize(time.size());
    if (retenv) result.envelope.resize(time.size());
    for (std::size_t i = 0; i < time.size(); ++i) {
        const float x = detail::SimpleSignalTypePolicy<T>::load(time[i]);
        const float envelope_value = std::exp(-attenuation * x * x);
        const float phase = 2.0F * pi * fc * x;
        result.in_phase[i] = detail::SimpleSignalTypePolicy<T>::store(
            envelope_value * std::cos(phase));
        if (retquad) {
            result.quadrature[i] = detail::SimpleSignalTypePolicy<T>::store(
                envelope_value * std::sin(phase));
        }
        if (retenv) {
            result.envelope[i] = detail::SimpleSignalTypePolicy<T>::store(
                envelope_value);
        }
    }
    return result;
}

double gausspulse_cutoff(double fc, double bw, double bwr, double tpr)
{
    if (fc < 0.0 || bw <= 0.0 || bwr >= 0.0 || tpr >= 0.0) {
        throw std::invalid_argument("gausspulse cutoff parameters");
    }
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    const double reference = std::pow(10.0, bwr / 20.0);
    const double attenuation = -(pi64 * pi64 * fc * fc * bw * bw) /
        (4.0 * std::log(reference));
    const double time_reference = std::pow(10.0, tpr / 20.0);
    return std::sqrt(-std::log(time_reference) / attenuation);
}

template <typename T>
std::vector<double> sawtooth_typed_cpu(const std::vector<T>& t, double width)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported sawtooth input dtype");
    std::vector<double> y(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        y[i] = sawtooth_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[i])),
            width);
    }
    return y;
}

template <typename T>
std::vector<double> sawtooth_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& width)
{
    if (width.size() != 1 && width.size() != t.size()) {
        throw std::invalid_argument("sawtooth width is not broadcastable to t");
    }
    std::vector<double> output(t.size());
    for (std::size_t index = 0; index < t.size(); ++index) {
        const double current = static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(
            width.size() == 1 ? width.front() : width[index]));
        output[index] = sawtooth_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[index])),
            current);
    }
    return output;
}

template <typename T>
WaveformBroadcastCpuResult sawtooth_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& width,
    const WaveformBroadcastOptions& options)
{
    const auto plan = make_broadcast_plan(
        options.t_shape, options.control_shape, t.size(), width.size());
    WaveformBroadcastCpuResult result;
    result.shape = plan.shape;
    result.values.resize(plan.left_indices.size());
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] = sawtooth_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[plan.left_indices[i]])),
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(width[plan.right_indices[i]])));
    }
    return result;
}

template <typename T>
std::vector<double> square_typed_cpu(const std::vector<T>& t, double duty)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported square input dtype");
    std::vector<double> y(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        y[i] = square_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[i])),
            duty);
    }
    return y;
}

template <typename T>
std::vector<double> square_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& duty)
{
    if (duty.size() != 1 && duty.size() != t.size()) {
        throw std::invalid_argument("square duty is not broadcastable to t");
    }
    std::vector<double> output(t.size());
    for (std::size_t index = 0; index < t.size(); ++index) {
        const double current = static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(
            duty.size() == 1 ? duty.front() : duty[index]));
        output[index] = square_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[index])),
            current);
    }
    return output;
}

template <typename T>
WaveformBroadcastCpuResult square_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& duty,
    const WaveformBroadcastOptions& options)
{
    const auto plan = make_broadcast_plan(
        options.t_shape, options.control_shape, t.size(), duty.size());
    WaveformBroadcastCpuResult result;
    result.shape = plan.shape;
    result.values.resize(plan.left_indices.size());
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] = square_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[plan.left_indices[i]])),
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(duty[plan.right_indices[i]])));
    }
    return result;
}

template <typename T>
void sawtooth_device(
    const DeviceArray<T>& t, DeviceArray<double>& out, double width)
{
    if (out.size() != t.size()) throw std::invalid_argument("sawtooth_device size mismatch");
    DeviceArray<float> computed(t.size());
    sawtooth_fp32_compute_device(t, computed, static_cast<float>(width));
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <typename T>
void sawtooth_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    DeviceArray<double>& out)
{
    if (out.size() != t.size()) throw std::invalid_argument("sawtooth_device size mismatch");
    DeviceArray<float> computed(t.size());
    sawtooth_fp32_compute_device(t, width, computed);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <typename T>
WaveformBroadcastDeviceResult sawtooth_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    const WaveformBroadcastOptions& options)
{
    const auto plan = make_broadcast_plan(
        options.t_shape, options.control_shape, t.size(), width.size());
    auto left_indices = DeviceArray<std::size_t>::from_host(plan.left_indices);
    auto right_indices = DeviceArray<std::size_t>::from_host(plan.right_indices);
    DeviceArray<float> computed(plan.left_indices.size());
    sawtooth_broadcast_fp32_compute_device(
        t, width, left_indices, right_indices, computed);
    WaveformBroadcastDeviceResult result;
    result.values = cuda_utils::finalize_fp64_device_storage(computed);
    result.shape = plan.shape;
    return result;
}

template <typename T>
void square_device(
    const DeviceArray<T>& t, DeviceArray<double>& out, double duty)
{
    if (out.size() != t.size()) throw std::invalid_argument("square_device size mismatch");
    square_fp64_storage_device(
        t, static_cast<void*>(out.data()), out.size(), static_cast<float>(duty));
}

template <typename T>
void square_device(
    const DeviceArray<T>& t, const DeviceArray<T>& duty,
    DeviceArray<double>& out)
{
    if (out.size() != t.size()) throw std::invalid_argument("square_device size mismatch");
    square_fp64_storage_device(
        t, duty, static_cast<void*>(out.data()), out.size());
}

template <typename T>
WaveformBroadcastDeviceResult square_device(
    const DeviceArray<T>& t, const DeviceArray<T>& duty,
    const WaveformBroadcastOptions& options)
{
    const auto plan = make_broadcast_plan(
        options.t_shape, options.control_shape, t.size(), duty.size());
    auto left_indices = DeviceArray<std::size_t>::from_host(plan.left_indices);
    auto right_indices = DeviceArray<std::size_t>::from_host(plan.right_indices);
    WaveformBroadcastDeviceResult result;
    result.values.reset(plan.left_indices.size());
    square_broadcast_fp64_storage_device(
        t, duty, left_indices, right_indices,
        static_cast<void*>(result.values.data()), result.values.size());
    result.shape = plan.shape;
    return result;
}

template <typename T>
std::vector<double> unit_impulse_typed_cpu(int shape, int idx)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported unit_impulse output dtype");
    std::vector<double> y(shape > 0 ? static_cast<std::size_t>(shape) : 0, 0.0);
    if (idx >= 0 && idx < shape) {
        y[static_cast<std::size_t>(idx)] = 1.0;
    }
    return y;
}

namespace {
int unit_impulse_first_extent(const std::vector<int>& shape)
{
    if (shape.empty()) throw std::invalid_argument("unit_impulse shape must not be empty");
    if (shape.front() < 0) throw std::invalid_argument("unit_impulse shape");
    return shape.front();
}

int unit_impulse_first_index(const std::vector<int>& idx)
{
    return idx.empty() ? 0 : idx.front();
}

int unit_impulse_named_index(const std::vector<int>& shape, const std::string& idx)
{
    if (idx != "mid") throw std::invalid_argument("unit_impulse idx must be mid");
    return unit_impulse_first_extent(shape) / 2;
}
}

template <typename T>
std::vector<double> unit_impulse_typed_cpu(
    const std::vector<int>& shape, const std::vector<int>& idx)
{
    return unit_impulse_typed_cpu<T>(
        unit_impulse_first_extent(shape), unit_impulse_first_index(idx));
}

template <typename T>
std::vector<double> unit_impulse_typed_cpu(
    const std::vector<int>& shape, const std::string& idx)
{
    return unit_impulse_typed_cpu<T>(
        unit_impulse_first_extent(shape), unit_impulse_named_index(shape, idx));
}

template <typename T>
void unit_impulse_device(DeviceArray<double>& out, int idx)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported unit_impulse dtype");
    static_assert(sizeof(double) == sizeof(std::uint64_t), "FP64 storage must be 64-bit");
    static_assert(std::numeric_limits<double>::is_iec559, "FP64 storage must use IEEE-754");
    unit_impulse_fp64_storage_device(out.data(), out.size(), idx);
}

template <typename T>
void unit_impulse_device(
    DeviceArray<double>& out, const std::vector<int>& shape,
    const std::vector<int>& idx)
{
    const int extent = unit_impulse_first_extent(shape);
    if (extent < 0 || out.size() != static_cast<std::size_t>(extent))
        throw std::invalid_argument("unit_impulse output/shape mismatch");
    unit_impulse_device<T>(out, unit_impulse_first_index(idx));
}

template <typename T>
void unit_impulse_device(
    DeviceArray<double>& out, const std::vector<int>& shape,
    const std::string& idx)
{
    const int extent = unit_impulse_first_extent(shape);
    if (extent < 0 || out.size() != static_cast<std::size_t>(extent))
        throw std::invalid_argument("unit_impulse output/shape mismatch");
    unit_impulse_device<T>(out, unit_impulse_named_index(shape, idx));
}

#define INSTANTIATE_CPU(T) \
    template std::vector<T> chirp_typed_cpu(const std::vector<T>&, float, float, float, float); \
    template std::vector<T> chirp_typed_cpu( \
        const std::vector<T>&, float, float, float, const std::string&, float, bool); \
    template std::vector<ComplexFloat> chirp_complex_typed_cpu( \
        const std::vector<T>&, float, float, float, float); \
    template std::vector<T> gausspulse_typed_cpu(const std::vector<T>&, float, float); \
    template GausspulseTypedResult<T> gausspulse_typed_cpu( \
        const std::vector<T>&, float, float, float, bool, bool); \
    template std::vector<double> sawtooth_typed_cpu(const std::vector<T>&, double); \
    template std::vector<double> sawtooth_typed_cpu(const std::vector<T>&, const std::vector<T>&); \
    template WaveformBroadcastCpuResult sawtooth_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const WaveformBroadcastOptions&); \
    template std::vector<double> square_typed_cpu(const std::vector<T>&, double); \
    template std::vector<double> square_typed_cpu(const std::vector<T>&, const std::vector<T>&); \
    template WaveformBroadcastCpuResult square_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const WaveformBroadcastOptions&); \
    template void sawtooth_device(const DeviceArray<T>&, DeviceArray<double>&, double); \
    template void sawtooth_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<double>&); \
    template void square_device(const DeviceArray<T>&, DeviceArray<double>&, double); \
    template void square_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<double>&); \
    template WaveformBroadcastDeviceResult sawtooth_device( \
        const DeviceArray<T>&, const DeviceArray<T>&, const WaveformBroadcastOptions&); \
    template WaveformBroadcastDeviceResult square_device( \
        const DeviceArray<T>&, const DeviceArray<T>&, const WaveformBroadcastOptions&); \
    template std::vector<double> unit_impulse_typed_cpu<T>(int, int); \
    template std::vector<double> unit_impulse_typed_cpu<T>( \
        const std::vector<int>&, const std::vector<int>&); \
    template std::vector<double> unit_impulse_typed_cpu<T>( \
        const std::vector<int>&, const std::string&); \
    template void unit_impulse_device<T>(DeviceArray<double>&, int); \
    template void unit_impulse_device<T>( \
        DeviceArray<double>&, const std::vector<int>&, const std::vector<int>&); \
    template void unit_impulse_device<T>( \
        DeviceArray<double>&, const std::vector<int>&, const std::string&)
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
#undef INSTANTIATE_CPU

}  // namespace cusignal
