#include <cusignal/operators/windows/windows_typed.h>
#include <cusignal/runtime/errors.h>
#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN
#include <cusignal/backends/fft/fft_interface_cpu.h>
#endif
#include <cusignal/runtime/host_output_finalize.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cusignal {
namespace {

constexpr float kPi = 3.14159265358979323846F;

template <class T>
float host_load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <class T>
T host_store(float value)
{
    return detail::SimpleSignalTypePolicy<T>::store(value);
}

// CPU reference 使用独立的 host 数学 helper，不包含或调用 GPU kernel helper。
double host_bessel_i0_fp64(double x)
{
    const double absolute_x = std::fabs(x);
    if (absolute_x < 3.75) {
        const double ratio = x / 3.75;
        const double y = ratio * ratio;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    const double y = 3.75 / absolute_x;
    return std::exp(absolute_x) / std::sqrt(absolute_x)
        * (0.39894228 + y * (0.01328592
        + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
        + y * (-0.02057706 + y * (0.02635537
        + y * (-0.01647633 + y * 0.00392377))))))));
}

template <typename Scalar>
void host_chebyshev_polynomial(
    int frequency, int length, Scalar beta, Scalar& real, Scalar& imag)
{
    const Scalar pi = static_cast<Scalar>(3.141592653589793238462643383279502884);
    const Scalar order = static_cast<Scalar>(length - 1);
    const Scalar angle = pi * static_cast<Scalar>(frequency) /
        static_cast<Scalar>(length);
    const Scalar x = beta * std::cos(angle);
    Scalar value = static_cast<Scalar>(0);
    if (x > static_cast<Scalar>(1)) {
        value = std::cosh(order * std::acosh(x));
    } else if (x < static_cast<Scalar>(-1)) {
        value = static_cast<Scalar>((length & 1) ? 1 : -1)
            * std::cosh(order * std::acosh(-x));
    } else {
        value = std::cos(order * std::acos(x));
    }
    if (length & 1) {
        real = value;
        imag = static_cast<Scalar>(0);
    } else {
        real = value * std::cos(angle);
        imag = value * std::sin(angle);
    }
}

template <typename Scalar>
Scalar host_chebyshev_raw(int index, int length, Scalar beta)
{
    const Scalar pi = static_cast<Scalar>(3.141592653589793238462643383279502884);
    Scalar sum = static_cast<Scalar>(0);
    for (int frequency = 0; frequency < length; ++frequency) {
        Scalar real = static_cast<Scalar>(0);
        Scalar imag = static_cast<Scalar>(0);
        host_chebyshev_polynomial(frequency, length, beta, real, imag);
        const Scalar angle = static_cast<Scalar>(-2) * pi *
            static_cast<Scalar>(index) * static_cast<Scalar>(frequency) /
            static_cast<Scalar>(length);
        sum += real * std::cos(angle) - imag * std::sin(angle);
    }
    return sum;
}

int host_chebyshev_raw_index(int output_index, int length)
{
    if (length & 1) {
        const int center = (length + 1) / 2 - 1;
        return output_index <= center
            ? center - output_index
            : output_index - center;
    }
    const int half = length / 2;
    return output_index < half
        ? half - output_index
        : output_index - half + 1;
}

template <typename Scalar>
Scalar host_taylor_value(
    Scalar position, int effective, int nbar, Scalar a, Scalar s2)
{
    const Scalar pi = static_cast<Scalar>(3.141592653589793238462643383279502884);
    Scalar value = static_cast<Scalar>(1);
    for (int order = 1; order < nbar; ++order) {
        const Scalar order_squared = static_cast<Scalar>(order * order);
        Scalar numerator = static_cast<Scalar>(
            ((order - 1) & 1) ? -1 : 1);
        Scalar denominator = static_cast<Scalar>(2);
        for (int index = 1; index < nbar; ++index) {
            const Scalar offset = static_cast<Scalar>(index) -
                static_cast<Scalar>(0.5);
            numerator *= static_cast<Scalar>(1) - order_squared
                / s2 / (a * a + offset * offset);
        }
        for (int index = 1; index < nbar; ++index) {
            if (index != order) {
                denominator *= static_cast<Scalar>(1)
                    - order_squared / static_cast<Scalar>(index * index);
            }
        }
        value += static_cast<Scalar>(2) * (numerator / denominator)
            * std::cos(static_cast<Scalar>(2) * pi *
                static_cast<Scalar>(order) *
                (position - static_cast<Scalar>(effective) /
                    static_cast<Scalar>(2) + static_cast<Scalar>(0.5)) /
                static_cast<Scalar>(effective));
    }
    return value;
}

} // namespace

template <class T>
void hamming_fp32_compute_device(int, DeviceArray<float>&, bool);
template <class T>
void chebwin_fp32_compute_device(int, T, DeviceArray<float>&, bool);
template <class T>
void general_cosine_fp32_compute_device(
    int, const DeviceArray<T>&, DeviceArray<float>&, bool);
template <class T>
void general_cosine_fp32_compute_device(
    int, const DeviceArray<T>&, int, DeviceArray<float>&, bool);
template <class T>
void general_gaussian_fp32_compute_device(
    int, T, T, DeviceArray<float>&, bool);
template <class T>
void kaiser_fp32_compute_device(int, T, DeviceArray<float>&, bool);
template <class T>
void parzen_fp32_compute_device(int, DeviceArray<float>&, bool);
template <class T>
void taylor_fp32_compute_device(
    int, int, T, DeviceArray<float>&, bool, bool);
template <class T>
void triang_fp32_compute_device(int, DeviceArray<float>&, bool);

template <class T>
std::vector<double> chebwin_typed_cpu(int length, T attenuation, bool symmetric)
{
    if (length < 0) {
        throw_error(ErrorCode::INVALID_SHAPE,
            "chebwin输入shape无效：length必须非负，实际为" + std::to_string(length));
    }
    if (length == 0) return {};
    if (length == 1) return {1.0};
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    const double attenuation_value =
        static_cast<double>(host_load(attenuation));
    const double beta = std::cosh(std::acosh(std::pow(
        10.0, std::fabs(attenuation_value) / 20.0)) / (effective - 1));
#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN
    std::vector<complexd> spectrum(static_cast<std::size_t>(effective));
    const double pi = 3.141592653589793238462643383279502884;
    const double order = static_cast<double>(effective - 1);
    for (int index = 0; index < effective; ++index) {
        const double angle = pi * static_cast<double>(index) /
            static_cast<double>(effective);
        const double x = beta * std::cos(angle);
        double real;
        if (x > 1.0) {
            real = std::cosh(order * std::acosh(x));
        } else if (x < -1.0) {
            const double sign = (effective & 1) ? 1.0 : -1.0;
            real = sign * std::cosh(order * std::acosh(-x));
        } else {
            real = std::cos(order * std::acos(x));
        }
        spectrum[static_cast<std::size_t>(index)] = (effective & 1)
            ? complexd(real, 0.0)
            : complexd(real * std::cos(angle), real * std::sin(angle));
    }
    const auto transformed = FFTInterface_cpu(effective).fft(spectrum);
    std::vector<double> full(static_cast<std::size_t>(effective));
    for (int index = 0; index < effective; ++index) {
        int source;
        if (effective & 1) {
            const int center = (effective - 1) / 2;
            source = index <= center ? center - index : index - center;
        } else {
            const int half = effective / 2;
            source = index < half ? half - index : index - half + 1;
        }
        full[static_cast<std::size_t>(index)] =
            transformed[static_cast<std::size_t>(source)].real();
    }
    const double peak = *std::max_element(full.begin(), full.end());
    for (int index = 0; index < length; ++index) {
        out[static_cast<std::size_t>(index)] =
            full[static_cast<std::size_t>(index)] / peak;
    }
    return out;
#else
    double peak = std::numeric_limits<double>::lowest();
    for (int index = 0; index < effective; ++index) {
        peak = std::max(peak, host_chebyshev_raw(
            host_chebyshev_raw_index(index, effective), effective, beta));
    }
    for (int index = 0; index < length; ++index) {
        out[index] = host_chebyshev_raw(
            host_chebyshev_raw_index(index, effective), effective, beta) / peak;
    }
    return out;
#endif
}

template <class T>
void chebwin_device(
    int length, T attenuation, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("chebwin_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
        chebwin_fp32_compute_device(length, attenuation, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
std::vector<double> general_cosine_typed_cpu(
    int length, const std::vector<T>& coefficients, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("general_cosine: length must be nonnegative");
    if (length == 0) return {};
    if (length == 1) return {1.0};
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    for (int index = 0; index < length; ++index) {
        const double phase = -pi64 + 2.0 * pi64 * index / (effective - 1);
        double value = 0.0;
        for (int order = 0; order < static_cast<int>(coefficients.size()); ++order) {
            value += static_cast<double>(host_load(coefficients[order]))
                * std::cos(order * phase);
        }
        out[index] = value;
    }
    return out;
}

namespace {
std::size_t general_cosine_coefficient_count(
    const std::vector<int>& shape, std::size_t data_size)
{
    if (shape.empty()) throw std::invalid_argument("general_cosine coefficient shape");
    std::size_t product = 1;
    bool zero = false;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument("general_cosine coefficient shape");
        if (extent == 0) {
            zero = true;
        } else if (!zero) {
            if (product > data_size / static_cast<std::size_t>(extent))
                throw std::invalid_argument("general_cosine coefficient data/shape mismatch");
            product *= static_cast<std::size_t>(extent);
        }
    }
    if ((zero ? 0 : product) != data_size)
        throw std::invalid_argument("general_cosine coefficient data/shape mismatch");
    return static_cast<std::size_t>(shape.front());
}
}

template <class T>
std::vector<double> general_cosine_typed_cpu(
    int length, const std::vector<T>& coefficients,
    const std::vector<int>& coefficient_shape, bool symmetric)
{
    const auto count = general_cosine_coefficient_count(
        coefficient_shape, coefficients.size());
    return general_cosine_typed_cpu(
        length, std::vector<T>(coefficients.begin(), coefficients.begin() + count),
        symmetric);
}

template <class T>
void general_cosine_device(
    int length, const DeviceArray<T>& coefficients,
    DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("general_cosine_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
        general_cosine_fp32_compute_device(
            length, coefficients, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
void general_cosine_device(
    int length, const DeviceArray<T>& coefficients,
    const std::vector<int>& coefficient_shape,
    DeviceArray<double>& out, bool symmetric)
{
    const auto count = general_cosine_coefficient_count(
        coefficient_shape, coefficients.size());
    if (length < 0 || out.size() != static_cast<std::size_t>(length))
        throw std::invalid_argument("general_cosine_device: output size must equal requested length");
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
        general_cosine_fp32_compute_device(
            length, coefficients, static_cast<int>(count), computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
std::vector<double> general_gaussian_typed_cpu(
    int length, T power, T width, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("general_gaussian: length must be nonnegative");
    if (length == 0) return {};
    if (length == 1) return {1.0};
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    const double power_value = static_cast<double>(host_load(power));
    const double width_value = static_cast<double>(host_load(width));
    for (int index = 0; index < length; ++index) {
        const double x = (index - 0.5 * (effective - 1)) / width_value;
        out[index] = std::exp(
            -0.5 * std::pow(std::fabs(x), 2.0 * power_value));
    }
    return out;
}

template <class T>
void general_gaussian_device(
    int length, T power, T width, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("general_gaussian_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
        general_gaussian_fp32_compute_device(
            length, power, width, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
std::vector<double> hamming_typed_cpu(int length, bool symmetric)
{
    if (length < 1) return {};
    const int effective = (!symmetric && length % 2 == 0) ? length + 1 : length;
    std::vector<double> out(length);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    for (int index = 0; index < length; ++index) {
        out[index] = effective == 1
            ? 1.0
            : 0.54 - 0.46
                * std::cos(2.0 * pi64 * index / (effective - 1));
    }
    return out;
}

template <class T>
void hamming_device(int length, DeviceArray<double>& out, bool symmetric)
{
    const std::size_t expected = length > 0
        ? static_cast<std::size_t>(length)
        : 0;
    if (out.size() != expected)
        throw std::invalid_argument("hamming_device: output size must equal max(length, 0)");
    DeviceArray<float> computed(expected);
    if (expected != 0) {
        hamming_fp32_compute_device<T>(length, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
void hamming_resident_device(
    int length, DeviceArray<float>& out, bool symmetric)
{
    static_assert(detail::is_simple_signal_input_v<T>,
        "unsupported hamming dtype");
    hamming_fp32_compute_device<T>(length, out, symmetric);
}

template <class T>
std::vector<double> kaiser_typed_cpu(int length, T beta, bool symmetric)
{
    if (length < 1) return {};
    const int effective = (!symmetric && length % 2 == 0) ? length + 1 : length;
    std::vector<double> out(length);
    const double beta_value = static_cast<double>(host_load(beta));
    const double denominator = host_bessel_i0_fp64(beta_value);
    for (int index = 0; index < length; ++index) {
        if (effective == 1) {
            out[index] = 1.0;
            continue;
        }
        const double alpha = 0.5 * (effective - 1);
        const double x = (index - alpha) / alpha;
        const double numerator = host_bessel_i0_fp64(beta_value
            * std::sqrt(std::max(0.0, 1.0 - x * x)));
        out[index] = numerator / denominator;
    }
    return out;
}

template <class T>
void kaiser_device(
    int length, T beta, DeviceArray<double>& out, bool symmetric)
{
    const std::size_t expected = length > 0
        ? static_cast<std::size_t>(length)
        : 0;
    if (out.size() != expected)
        throw std::invalid_argument("kaiser_device: output size must equal max(length, 0)");
    DeviceArray<float> computed(expected);
    if (expected != 0) {
        kaiser_fp32_compute_device(length, beta, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
std::vector<double> parzen_typed_cpu(int length, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("parzen: length must be nonnegative");
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    for (int index = 0; index < length; ++index) {
        if (effective == 1) {
            out[index] = 1.0;
            continue;
        }
        const double x = std::fabs(index - 0.5 * (effective - 1))
            / (0.5 * effective);
        out[index] = x <= 0.5
            ? 1.0 - 6.0 * x * x + 6.0 * x * x * x
            : 2.0 * std::pow(1.0 - x, 3.0);
    }
    return out;
}

template <class T>
void parzen_device(int length, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("parzen_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length != 0) parzen_fp32_compute_device<T>(length, computed, symmetric);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
std::vector<double> taylor_typed_cpu(
    int length, int nbar, T sidelobe, bool normalize, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("taylor: length must be nonnegative");
    if (length == 0) return {};
    if (length == 1) return {1.0};
    if (nbar < 1) throw std::invalid_argument("taylor: nbar must be at least one");
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    const double sidelobe_value = static_cast<double>(host_load(sidelobe));
    const double a = std::acosh(std::pow(10.0, sidelobe_value / 20.0)) / pi64;
    const double nbar_offset = nbar - 0.5;
    const double s2 = static_cast<double>(nbar * nbar)
        / (a * a + nbar_offset * nbar_offset);
    const double peak = host_taylor_value(
        0.5 * (effective - 1), effective, nbar, a, s2);
    for (int index = 0; index < length; ++index) {
        const double value = host_taylor_value(
            static_cast<double>(index), effective, nbar, a, s2);
        out[index] = normalize ? value / peak : value;
    }
    return out;
}

template <class T>
void taylor_device(
    int length, int nbar, T sidelobe, DeviceArray<double>& out,
    bool normalize, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("taylor_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
    if (nbar < 1) throw std::invalid_argument("taylor_device: nbar must be at least one");
        taylor_fp32_compute_device(
            length, nbar, sidelobe, computed, normalize, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
std::vector<double> triang_typed_cpu(int length, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("triang: length must be nonnegative");
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    for (int index = 0; index < length; ++index) {
        if (effective == 1) {
            out[index] = 1.0;
            continue;
        }
        const int midpoint = effective / 2;
        const int rank = index < midpoint ? index + 1 : effective - index;
        out[index] = (effective & 1)
            ? 2.0 * rank / (effective + 1)
            : (2.0 * rank - 1.0) / effective;
    }
    return out;
}

template <class T>
void triang_device(int length, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("triang_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length != 0) triang_fp32_compute_device<T>(length, computed, symmetric);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

#define INSTANTIATE_WINDOWS_CPU(T) \
    template std::vector<double> chebwin_typed_cpu(int, T, bool); \
    template void chebwin_device(int, T, DeviceArray<double>&, bool); \
    template std::vector<double> general_cosine_typed_cpu( \
        int, const std::vector<T>&, bool); \
    template std::vector<double> general_cosine_typed_cpu( \
        int, const std::vector<T>&, const std::vector<int>&, bool); \
    template void general_cosine_device( \
        int, const DeviceArray<T>&, DeviceArray<double>&, bool); \
    template void general_cosine_device( \
        int, const DeviceArray<T>&, const std::vector<int>&, DeviceArray<double>&, bool); \
    template std::vector<double> general_gaussian_typed_cpu(int, T, T, bool); \
    template void general_gaussian_device( \
        int, T, T, DeviceArray<double>&, bool); \
    template std::vector<double> hamming_typed_cpu<T>(int, bool); \
    template void hamming_device<T>(int, DeviceArray<double>&, bool); \
    template void hamming_resident_device<T>(int, DeviceArray<float>&, bool); \
    template std::vector<double> kaiser_typed_cpu(int, T, bool); \
    template void kaiser_device(int, T, DeviceArray<double>&, bool); \
    template std::vector<double> parzen_typed_cpu<T>(int, bool); \
    template void parzen_device<T>(int, DeviceArray<double>&, bool); \
    template std::vector<double> taylor_typed_cpu(int, int, T, bool, bool); \
    template void taylor_device( \
        int, int, T, DeviceArray<double>&, bool, bool); \
    template std::vector<double> triang_typed_cpu<T>(int, bool); \
    template void triang_device<T>(int, DeviceArray<double>&, bool)

INSTANTIATE_WINDOWS_CPU(float);
INSTANTIATE_WINDOWS_CPU(__half);
INSTANTIATE_WINDOWS_CPU(std::int32_t);
INSTANTIATE_WINDOWS_CPU(std::int16_t);
INSTANTIATE_WINDOWS_CPU(std::int8_t);

#undef INSTANTIATE_WINDOWS_CPU

} // namespace cusignal
