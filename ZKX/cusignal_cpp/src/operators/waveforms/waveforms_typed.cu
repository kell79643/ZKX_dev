#include <cusignal/operators/waveforms/waveforms_typed.h>
#include "waveforms_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>
#include <cmath>
#include <stdexcept>
namespace cusignal {
namespace {
waveforms_detail::ChirpMethod parse_typed_chirp_method(const std::string& method)
{
    if (method == "linear" || method == "lin" || method == "li")
        return waveforms_detail::ChirpMethod::Linear;
    if (method == "quadratic" || method == "quad" || method == "q")
        return waveforms_detail::ChirpMethod::Quadratic;
    if (method == "logarithmic" || method == "log" || method == "lo")
        return waveforms_detail::ChirpMethod::Logarithmic;
    if (method == "hyperbolic" || method == "hyp")
        return waveforms_detail::ChirpMethod::Hyperbolic;
    throw std::invalid_argument("chirp_device: method must be linear, quadratic, logarithmic, or hyperbolic");
}

void validate_chirp_frequencies(
    waveforms_detail::ChirpMethod method, float f0, float f1)
{
    if (method == waveforms_detail::ChirpMethod::Logarithmic &&
        f0 * f1 <= 0.0F)
        throw std::invalid_argument("logarithmic chirp frequencies");
    if (method == waveforms_detail::ChirpMethod::Hyperbolic &&
        (f0 == 0.0F || f1 == 0.0F))
        throw std::invalid_argument("hyperbolic chirp frequencies");
}
}

template<class T>
void chirp_device(
    const DeviceArray<T>& t, DeviceArray<T>& output,
    float f0, float t1, float f1, float phase)
{
    chirp_device(t, output, f0, t1, f1, std::string("linear"), phase, true);
}

template<class T>
void chirp_device(
    const DeviceArray<T>& t, DeviceArray<T>& output,
    float f0, float t1, float f1, const std::string& method,
    float phase, bool vertex_zero)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (output.size() != t.size()) throw std::invalid_argument("chirp shape");
    const auto parsed = parse_typed_chirp_method(method);
    validate_chirp_frequencies(parsed, f0, f1);
    if (t.empty()) return;
    constexpr float degrees_to_radians = 3.14159265358979323846F / 180.0F;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::chirp_real_kernel<T,float>, output.size(),
        t.data(), output.data(), t.size(), f0, t1, f1,
        static_cast<int>(parsed), phase * degrees_to_radians, vertex_zero);
}

template<class T>
void chirp_complex_device(
    const DeviceArray<T>& t, DeviceArray<ComplexFloat>& output,
    float f0, float t1, float f1, float phase)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (output.size() != t.size()) throw std::invalid_argument("chirp shape");
    if (t.empty()) return;
    constexpr float degrees_to_radians = 3.14159265358979323846F / 180.0F;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::chirp_complex_kernel<T,float>, output.size(),
        t.data(), output.data(), t.size(), f0, t1, f1,
        phase * degrees_to_radians);
}

template<class T>
void gausspulse_device(
    const DeviceArray<T>& t, DeviceArray<T>& in_phase,
    DeviceArray<T>* quadrature, DeviceArray<T>* envelope,
    float fc, float bw, float bwr)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (fc < 0.0F || bw <= 0.0F || bwr >= 0.0F ||
        in_phase.size() != t.size() ||
        (quadrature != nullptr && quadrature->size() != t.size()) ||
        (envelope != nullptr && envelope->size() != t.size())) {
        throw std::invalid_argument("gausspulse parameters or shape");
    }
    const float attenuation =
        waveforms_detail::gausspulse_attenuation_fp32(fc, bw, bwr);
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::gausspulse_kernel<T,float>, in_phase.size(),
        t.data(), in_phase.data(),
        quadrature == nullptr ? nullptr : quadrature->data(),
        envelope == nullptr ? nullptr : envelope->data(),
        t.size(), fc, attenuation);
}

template<class T>
void gausspulse_device(
    const DeviceArray<T>& t, DeviceArray<T>& output, float fc, float bw)
{
    gausspulse_device<T>(
        t, output, static_cast<DeviceArray<T>*>(nullptr),
        static_cast<DeviceArray<T>*>(nullptr), fc, bw, -6.0F);
}

template<class T>
void gausspulse_device(
    const DeviceArray<T>& t, DeviceArray<T>& in_phase,
    DeviceArray<T>& quadrature, DeviceArray<T>& envelope,
    float fc, float bw, float bwr)
{
    gausspulse_device(
        t, in_phase, &quadrature, &envelope, fc, bw, bwr);
}

template <typename T>
void sawtooth_fp32_compute_device(
    const DeviceArray<T>& t, DeviceArray<float>& y, float width)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported sawtooth input dtype");
    if (t.size() != y.size()) throw std::invalid_argument("sawtooth_device size mismatch");
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::sawtooth_fixed_kernel<T, float, float>,
        t.size(), t.data(), y.data(), t.size(), width);
}

template <typename T>
void sawtooth_fp32_compute_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    DeviceArray<float>& y)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported sawtooth input dtype");
    if (t.size() != y.size() || (width.size() != 1 && width.size() != t.size())) {
        throw std::invalid_argument("sawtooth width is not broadcastable to t");
    }
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::sawtooth_variable_kernel<T, T, float>,
        t.size(), t.data(), width.data(), y.data(), t.size(), width.size());
}

template <typename T>
void sawtooth_broadcast_fp32_compute_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    const DeviceArray<std::size_t>& t_indices,
    const DeviceArray<std::size_t>& width_indices,
    DeviceArray<float>& y)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported sawtooth input dtype");
    if (t_indices.size() != y.size() || width_indices.size() != y.size())
        throw std::invalid_argument("sawtooth broadcast index shape");
    if (y.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::sawtooth_broadcast_kernel<T, T, float>,
        y.size(), t.data(), width.data(), t_indices.data(), width_indices.data(),
        y.data(), y.size());
}

template <typename T>
void square_fp64_storage_device(
    const DeviceArray<T>& t, void* output_storage, std::size_t count, float duty)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported square input dtype");
    if (t.size() != count) throw std::invalid_argument("square_device size mismatch");
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::square_fixed_fp64_storage_kernel<T, float>,
        t.size(), t.data(), static_cast<std::uint64_t*>(output_storage),
        t.size(), duty);
}

template <typename T>
void square_fp64_storage_device(
    const DeviceArray<T>& t, const DeviceArray<T>& duty,
    void* output_storage, std::size_t count)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported square input dtype");
    if (t.size() != count || (duty.size() != 1 && duty.size() != t.size())) {
        throw std::invalid_argument("square duty is not broadcastable to t");
    }
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::square_variable_fp64_storage_kernel<T, T>,
        t.size(), t.data(), duty.data(),
        static_cast<std::uint64_t*>(output_storage), t.size(), duty.size());
}

template <typename T>
void square_broadcast_fp64_storage_device(
    const DeviceArray<T>& t, const DeviceArray<T>& duty,
    const DeviceArray<std::size_t>& t_indices,
    const DeviceArray<std::size_t>& duty_indices,
    void* output_storage, std::size_t count)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported square input dtype");
    if (t_indices.size() != count || duty_indices.size() != count)
        throw std::invalid_argument("square broadcast index shape");
    if (count == 0) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::square_broadcast_fp64_storage_kernel<T, T>,
        count, t.data(), duty.data(), t_indices.data(), duty_indices.data(),
        static_cast<std::uint64_t*>(output_storage), count);
}

void unit_impulse_fp64_storage_device(
    void* output_storage, std::size_t count, int idx)
{
    if (count == 0) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::unit_impulse_fp64_storage_kernel, count,
        static_cast<std::uint64_t*>(output_storage), count, idx);
}

#define I(T) \
    template void chirp_device(const DeviceArray<T>&,DeviceArray<T>&,float,float,float,float); \
    template void chirp_device(const DeviceArray<T>&,DeviceArray<T>&,float,float,float,const std::string&,float,bool); \
    template void chirp_complex_device(const DeviceArray<T>&,DeviceArray<ComplexFloat>&,float,float,float,float); \
    template void gausspulse_device(const DeviceArray<T>&,DeviceArray<T>&,float,float); \
    template void gausspulse_device(const DeviceArray<T>&,DeviceArray<T>&,DeviceArray<T>*,DeviceArray<T>*,float,float,float); \
    template void gausspulse_device(const DeviceArray<T>&,DeviceArray<T>&,DeviceArray<T>&,DeviceArray<T>&,float,float,float); \
    template void sawtooth_fp32_compute_device(const DeviceArray<T>&, DeviceArray<float>&, float); \
    template void sawtooth_fp32_compute_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<float>&); \
    template void sawtooth_broadcast_fp32_compute_device(const DeviceArray<T>&, const DeviceArray<T>&, const DeviceArray<std::size_t>&, const DeviceArray<std::size_t>&, DeviceArray<float>&); \
    template void square_fp64_storage_device(const DeviceArray<T>&, void*, std::size_t, float); \
    template void square_fp64_storage_device(const DeviceArray<T>&, const DeviceArray<T>&, void*, std::size_t); \
    template void square_broadcast_fp64_storage_device(const DeviceArray<T>&, const DeviceArray<T>&, const DeviceArray<std::size_t>&, const DeviceArray<std::size_t>&, void*, std::size_t)
I(float);I(__half);I(int32_t);I(int16_t);I(int8_t);
#undef I
}
