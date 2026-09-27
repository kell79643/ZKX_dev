#include <cusignal/operators/wavelets/wavelets_typed.h>
#include <cusignal/runtime/host_output_finalize.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cusignal {
namespace {

template <typename T>
float load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <class T>
std::vector<int> cwt_lengths(
    int data_count, const std::vector<T>& widths)
{
    if (data_count < 0)
        throw std::invalid_argument("prepare_cwt_workspace: data_count must be nonnegative");
    if (widths.empty()) return {};
    std::vector<int> lengths(widths.size());
    for (std::size_t index = 0; index < widths.size(); ++index) {
        const int width = static_cast<int>(load(widths[index]));
        const int length = std::min(10 * width, data_count);
        if (length <= 0)
            throw std::invalid_argument("prepare_cwt_workspace: every width must produce a positive wavelet length");
        lengths[index] = length;
    }
    return lengths;
}

}  // namespace

template <typename T>
void ricker_fp32_compute_device(int, T, DeviceArray<float>&);
template <typename T>
void morlet_complex_fp32_compute_device(
    int, T, T, bool, DeviceArray<ComplexFloat>&);
template <typename T>
void morlet2_complex_fp32_compute_device(
    int, T, T, DeviceArray<ComplexFloat>&);
template <typename T>
void cwt_complex_fp32_compute_device(
    const DeviceArray<T>&, const CwtDeviceWorkspace&,
    DeviceArray<ComplexFloat>&);
void cwt_extract_real_fp32_device(
    const DeviceArray<ComplexFloat>&, DeviceArray<float>&);

template <class T>
std::vector<ComplexDouble> morlet_typed_cpu(int n, T frequency, T scale, bool complete)
{
    const double s = static_cast<double>(load(scale));
    const double w = static_cast<double>(load(frequency));
    if (n < 1 || s <= 0)
        throw std::invalid_argument("morlet: n and scale must both be positive");
    std::vector<ComplexDouble> out(n);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    for (int i = 0; i < n; ++i) {
        const double x = n == 1 ? -2 * pi64 * s
            : -2 * pi64 * s + 4 * pi64 * s * i / (n - 1);
        const double envelope = std::exp(-.5 * x * x) * std::pow(pi64, -.25);
        const double correction = complete ? std::exp(-.5 * w * w) : 0;
        out[i] = {envelope * (std::cos(w * x) - correction), envelope * std::sin(w * x)};
    }
    return out;
}

template <class T>
std::vector<ComplexDouble> morlet2_typed_cpu(int n, T scale, T frequency)
{
    const double s = static_cast<double>(load(scale));
    const double w = static_cast<double>(load(frequency));
    if (n < 1 || s <= 0)
        throw std::invalid_argument("morlet2: n and scale must both be positive");
    std::vector<ComplexDouble> out(n);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    for (int i = 0; i < n; ++i) {
        const double x = (i - .5 * (n - 1)) / s;
        const double envelope = std::exp(-.5 * x * x) * std::pow(pi64, -.25) / std::sqrt(s);
        out[i] = {envelope * std::cos(w * x), envelope * std::sin(w * x)};
    }
    return out;
}

template <typename T>
void morlet_device(
    int n, T frequency, T scale, bool complete,
    DeviceArray<ComplexDouble>& out)
{
    if (n < 1 || out.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("morlet_device: n must be positive and output size must equal n");
    }
    DeviceArray<ComplexFloat> computed(out.size());
    morlet_complex_fp32_compute_device(n, frequency, scale, complete, computed);
    out = cuda_utils::finalize_complex_fp64_device_storage(computed);
}

template <typename T>
void morlet2_device(
    int n, T scale, T frequency, DeviceArray<ComplexDouble>& out)
{
    if (n < 1 || out.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("morlet2_device: n must be positive and output size must equal n");
    }
    DeviceArray<ComplexFloat> computed(out.size());
    morlet2_complex_fp32_compute_device(n, scale, frequency, computed);
    out = cuda_utils::finalize_complex_fp64_device_storage(computed);
}

template <class T>
std::vector<double> ricker_typed_cpu(int n, T width)
{
    const double a = static_cast<double>(load(width));
    if (n < 1 || a <= 0)
        throw std::invalid_argument("ricker: n and width must both be positive");
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) {
        const double x = i - .5 * (n - 1);
        const double q = x * x / (a * a);
        const double value = 2 / (std::sqrt(3 * a) *
            std::pow(3.141592653589793238462643383279502884, .25)) *
            (1 - q) * std::exp(-.5 * q);
        out[i] = value;
    }
    return out;
}

template <typename T>
void ricker_device(int n, T width, DeviceArray<double>& out)
{
    if (n < 1 || out.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("ricker_device: n must be positive and output size must equal n");
    }
    DeviceArray<float> computed(out.size());
    ricker_fp32_compute_device(n, width, computed);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("prepare_cwt_workspace: real wavelet callable is empty");
    const auto lengths_host = cwt_lengths(data_count, widths);
    std::vector<ComplexFloat> bank(
        static_cast<std::size_t>(data_count) * widths.size(), {0.0F, 0.0F});
    for (std::size_t row = 0; row < widths.size(); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths_host[row], width);
        if (values.size() != static_cast<std::size_t>(lengths_host[row]))
            throw std::invalid_argument("prepare_cwt_workspace: real wavelet callable returned an unexpected length");
        for (int index = 0; index < lengths_host[row]; ++index)
            bank[row * data_count + index] = {
                static_cast<float>(values[index]), 0.0F};
    }
    CwtDeviceWorkspace workspace;
    workspace.data_count = data_count;
    workspace.width_count = static_cast<int>(widths.size());
    workspace.max_wavelet_length = data_count;
    workspace.complex_output = false;
    workspace.lengths = DeviceArray<int>::from_host(lengths_host);
    workspace.wavelets = DeviceArray<ComplexFloat>::from_host(bank);
    return workspace;
}

template <class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("prepare_cwt_workspace: complex wavelet callable is empty");
    const auto lengths_host = cwt_lengths(data_count, widths);
    std::vector<ComplexFloat> bank(
        static_cast<std::size_t>(data_count) * widths.size(), {0.0F, 0.0F});
    for (std::size_t row = 0; row < widths.size(); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths_host[row], width);
        if (values.size() != static_cast<std::size_t>(lengths_host[row]))
            throw std::invalid_argument("prepare_cwt_workspace: complex wavelet callable returned an unexpected length");
        for (int index = 0; index < lengths_host[row]; ++index)
            bank[row * data_count + index] = {
                static_cast<float>(values[index].re),
                static_cast<float>(values[index].im)};
    }
    CwtDeviceWorkspace workspace;
    workspace.data_count = data_count;
    workspace.width_count = static_cast<int>(widths.size());
    workspace.max_wavelet_length = data_count;
    workspace.complex_output = true;
    workspace.lengths = DeviceArray<int>::from_host(lengths_host);
    workspace.wavelets = DeviceArray<ComplexFloat>::from_host(bank);
    return workspace;
}

template <class T>
std::vector<double> cwt_typed_cpu(
    const std::vector<T>& signal, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("cwt_typed_cpu: real wavelet callable is empty");
    const int n = static_cast<int>(signal.size());
    const auto lengths = cwt_lengths(n, widths);
    std::vector<double> out(signal.size() * widths.size());
    for (int row = 0; row < static_cast<int>(widths.size()); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths[row], width);
        if (values.size() != static_cast<std::size_t>(lengths[row]))
            throw std::invalid_argument("cwt_typed_cpu: real wavelet callable returned an unexpected length");
        for (int time = 0; time < n; ++time) {
            const int full = (lengths[row] - 1) / 2 + time;
            const int first = std::max(0, full - (lengths[row] - 1));
            const int last = std::min(n - 1, full);
            double sum = 0.0;
            for (int input = first; input <= last; ++input) {
                const int coefficient = lengths[row] - 1 - (full - input);
                sum += static_cast<double>(load(signal[input])) * values[coefficient];
            }
            out[row * n + time] = sum;
        }
    }
    return out;
}

template <class T>
std::vector<ComplexDouble> cwt_typed_cpu(
    const std::vector<T>& signal, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("cwt_typed_cpu: complex wavelet callable is empty");
    const int n = static_cast<int>(signal.size());
    const auto lengths = cwt_lengths(n, widths);
    std::vector<ComplexDouble> out(signal.size() * widths.size());
    for (int row = 0; row < static_cast<int>(widths.size()); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths[row], width);
        if (values.size() != static_cast<std::size_t>(lengths[row]))
            throw std::invalid_argument("cwt_typed_cpu: complex wavelet callable returned an unexpected length");
        for (int time = 0; time < n; ++time) {
            const int full = (lengths[row] - 1) / 2 + time;
            const int first = std::max(0, full - (lengths[row] - 1));
            const int last = std::min(n - 1, full);
            ComplexDouble sum{0.0, 0.0};
            for (int input = first; input <= last; ++input) {
                const int coefficient = lengths[row] - 1 - (full - input);
                const double value = static_cast<double>(load(signal[input]));
                sum.re += value * values[coefficient].re;
                sum.im -= value * values[coefficient].im;
            }
            out[row * n + time] = sum;
        }
    }
    return out;
}

template <class T>
void cwt_device(
    const DeviceArray<T>& signal, const CwtDeviceWorkspace& workspace,
    DeviceArray<double>& out)
{
    if (workspace.complex_output || out.size() != workspace.output_size())
        throw std::invalid_argument("cwt_device: real output requested with complex workspace or wrong output size");
    DeviceArray<ComplexFloat> computed(workspace.output_size());
    cwt_complex_fp32_compute_device(signal, workspace, computed);
    DeviceArray<float> real(computed.size());
    cwt_extract_real_fp32_device(computed, real);
    out = cuda_utils::finalize_fp64_device_storage(real);
}

template <class T>
void cwt_device(
    const DeviceArray<T>& signal, const CwtDeviceWorkspace& workspace,
    DeviceArray<ComplexDouble>& out)
{
    if (!workspace.complex_output || out.size() != workspace.output_size())
        throw std::invalid_argument("cwt_device: complex output requested with real workspace or wrong output size");
    DeviceArray<ComplexFloat> computed(workspace.output_size());
    cwt_complex_fp32_compute_device(signal, workspace, computed);
    out = cuda_utils::finalize_complex_fp64_device_storage(computed);
}

template <typename T>
std::vector<std::int64_t> qmf_typed_cpu(const std::vector<T>& hk)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported qmf input dtype");
    std::vector<std::int64_t> out(hk.size());
    for (std::size_t i = 0; i < hk.size(); ++i) {
        const std::int64_t sign = (i & 1U) ? -1 : 1;
        out[i] = static_cast<std::int64_t>(hk.size() - (i + 1)) * sign;
    }
    return out;
}

template <typename T>
std::vector<std::int64_t> qmf_typed_cpu(
    const std::vector<T>& hk, const std::vector<int>& shape)
{
    if (shape.empty()) throw std::invalid_argument("qmf shape must not be empty");
    std::size_t input_size = 1;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument("qmf shape");
        if (extent != 0 && input_size > hk.size() / static_cast<std::size_t>(extent))
            throw std::invalid_argument("qmf data/shape mismatch");
        input_size *= static_cast<std::size_t>(extent);
    }
    if (input_size != hk.size()) throw std::invalid_argument("qmf data/shape mismatch");
    const std::size_t count = static_cast<std::size_t>(shape.front());
    std::vector<std::int64_t> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::int64_t sign = (i & 1U) ? -1 : 1;
        out[i] = static_cast<std::int64_t>(count - (i + 1)) * sign;
    }
    return out;
}

#define INSTANTIATE_CPU(T) \
    template std::vector<ComplexDouble> morlet_typed_cpu(int, T, T, bool); \
    template std::vector<ComplexDouble> morlet2_typed_cpu(int, T, T); \
    template void morlet_device(int, T, T, bool, DeviceArray<ComplexDouble>&); \
    template void morlet2_device(int, T, T, DeviceArray<ComplexDouble>&); \
    template std::vector<double> ricker_typed_cpu(int, T); \
    template void ricker_device(int, T, DeviceArray<double>&); \
    template CwtDeviceWorkspace prepare_cwt_workspace( \
        int, const std::vector<T>&, const CwtRealWaveletCallable&); \
    template CwtDeviceWorkspace prepare_cwt_workspace( \
        int, const std::vector<T>&, const CwtComplexWaveletCallable&); \
    template std::vector<double> cwt_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const CwtRealWaveletCallable&); \
    template std::vector<ComplexDouble> cwt_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const CwtComplexWaveletCallable&); \
    template void cwt_device( \
        const DeviceArray<T>&, const CwtDeviceWorkspace&, DeviceArray<double>&); \
    template void cwt_device( \
        const DeviceArray<T>&, const CwtDeviceWorkspace&, DeviceArray<ComplexDouble>&); \
    template std::vector<std::int64_t> qmf_typed_cpu(const std::vector<T>&); \
    template std::vector<std::int64_t> qmf_typed_cpu( \
        const std::vector<T>&, const std::vector<int>&)
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
#undef INSTANTIATE_CPU

}  // namespace cusignal
