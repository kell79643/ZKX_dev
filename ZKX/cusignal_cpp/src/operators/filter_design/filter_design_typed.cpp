#include <cusignal/operators/filter_design/filter_design_typed.h>
#include <cusignal/runtime/host_output_finalize.h>
#include <cusignal/operators/windows/windows_typed.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cusignal {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
template <typename T> double load(T v) { return static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(v)); }
double sinc(double x) { return std::fabs(x) < 1.0e-15 ? 1.0 : std::sin(pi * x) / (pi * x); }

struct FirwinPrepared {
    std::vector<double> bands;
    std::vector<double> window;
    bool scale{true};
};

template <typename T>
FirwinPrepared prepare_firwin(
    int n, const std::vector<T>& cutoff, const FirwinOptions& options)
{
    if (n <= 0 || cutoff.empty() || !std::isfinite(options.fs) ||
        options.fs <= 0.0 ||
        options.fs > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::invalid_argument("firwin shape or fs");
    const double nyquist = options.fs * 0.5;
    std::vector<double> normalized(cutoff.size());
    double previous = 0.0;
    for (std::size_t index = 0; index < cutoff.size(); ++index) {
        const double value = load(cutoff[index]) / nyquist;
        if (!(value > 0.0 && value < 1.0) ||
            (index > 0 && !(value > previous)))
            throw std::invalid_argument("firwin cutoff");
        normalized[index] = value;
        previous = value;
    }
    bool pass_zero = true;
    switch (options.pass_zero) {
    case FirwinPassZero::boolean_true: pass_zero = true; break;
    case FirwinPassZero::boolean_false: pass_zero = false; break;
    case FirwinPassZero::lowpass:
        if (cutoff.size() != 1) throw std::invalid_argument("firwin lowpass cutoff");
        pass_zero = true;
        break;
    case FirwinPassZero::highpass:
        if (cutoff.size() != 1) throw std::invalid_argument("firwin highpass cutoff");
        pass_zero = false;
        break;
    case FirwinPassZero::bandpass:
        if (cutoff.size() <= 1) throw std::invalid_argument("firwin bandpass cutoff");
        pass_zero = false;
        break;
    case FirwinPassZero::bandstop:
        if (cutoff.size() <= 1) throw std::invalid_argument("firwin bandstop cutoff");
        pass_zero = true;
        break;
    }
    const bool pass_nyquist = static_cast<bool>(cutoff.size() & 1U) ^ pass_zero;
    if (pass_nyquist && n % 2 == 0)
        throw std::invalid_argument("even numtaps require zero response at Nyquist");
    std::vector<double> edges;
    if (pass_zero) edges.push_back(0.0);
    edges.insert(edges.end(), normalized.begin(), normalized.end());
    if (pass_nyquist) edges.push_back(1.0);
    if (edges.size() % 2 != 0)
        throw std::invalid_argument("firwin passband pairing");
    FirwinPrepared prepared;
    prepared.bands = std::move(edges);
    prepared.scale = options.scale;
    prepared.window.resize(n, 1.0);
    if (options.use_width) {
        const double attenuation =
            2.285 * (n - 1) * pi * (options.width / nyquist) + 7.95;
        const double beta = attenuation > 50.0
            ? 0.1102 * (attenuation - 8.7)
            : (attenuation > 21.0
                ? 0.5842 * std::pow(attenuation - 21.0, 0.4)
                    + 0.07886 * (attenuation - 21.0)
                : 0.0);
        const auto kaiser = kaiser_typed_cpu<float>(
            n, static_cast<float>(beta), true);
        for (int index = 0; index < n; ++index)
            prepared.window[index] = kaiser[index];
    } else if (options.window_mode == FirwinWindowMode::hamming) {
        for (int index = 0; index < n; ++index)
            prepared.window[index] = n == 1 ? 1.0
                : 0.54 - 0.46 * std::cos(2.0 * pi * index / (n - 1));
    } else if (options.window_mode == FirwinWindowMode::explicit_values) {
        if (options.window.size() != static_cast<std::size_t>(n))
            throw std::invalid_argument("firwin explicit window length");
        prepared.window = options.window;
    } else if (options.window_mode != FirwinWindowMode::none) {
        throw std::invalid_argument("firwin window mode");
    }
    return prepared;
}

struct Firwin2Prepared {
    int nfreqs{0};
    int filter_type{1};
    float nyquist{1.0F};
    std::vector<float> freq;
    std::vector<float> gain;
    std::vector<float> window;
};

template <typename T>
Firwin2Prepared prepare_firwin2(
    int n, const std::vector<T>& freq, const std::vector<T>& gain,
    const Firwin2Options& options)
{
    if (n <= 0 || freq.size() < 2 || freq.size() != gain.size() ||
        !std::isfinite(options.fs) || options.fs <= 0.0 ||
        options.fs > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::invalid_argument("firwin2 shape or fs");
    Firwin2Prepared prepared;
    prepared.nyquist = static_cast<float>(options.fs * 0.5);
    if (!(prepared.nyquist > 0.0F))
        throw std::invalid_argument("firwin2 fs is not representable in FP32");
    prepared.freq.resize(freq.size());
    prepared.gain.resize(gain.size());
    for (std::size_t index = 0; index < freq.size(); ++index) {
        prepared.freq[index] = static_cast<float>(load(freq[index]));
        prepared.gain[index] = static_cast<float>(load(gain[index]));
    }
    if (prepared.freq.front() != 0.0F ||
        prepared.freq.back() != prepared.nyquist)
        throw std::invalid_argument("firwin2 frequency endpoints");
    std::vector<float> differences(freq.size() - 1);
    for (std::size_t index = 0; index + 1 < freq.size(); ++index) {
        differences[index] = prepared.freq[index + 1] - prepared.freq[index];
        if (differences[index] < 0.0F)
            throw std::invalid_argument("firwin2 frequencies nondecreasing");
    }
    for (std::size_t index = 0; index + 1 < differences.size(); ++index)
        if (differences[index] == 0.0F && differences[index + 1] == 0.0F)
            throw std::invalid_argument("firwin2 frequency repeated more than twice");
    if (prepared.freq[1] == 0.0F ||
        prepared.freq[prepared.freq.size() - 2] == prepared.nyquist)
        throw std::invalid_argument("firwin2 endpoint repeated");
    prepared.filter_type = options.antisymmetric
        ? ((n % 2 == 0) ? 4 : 3)
        : ((n % 2 == 0) ? 2 : 1);
    if ((prepared.filter_type == 2 && prepared.gain.back() != 0.0F) ||
        (prepared.filter_type == 3 &&
            (prepared.gain.front() != 0.0F || prepared.gain.back() != 0.0F)) ||
        (prepared.filter_type == 4 && prepared.gain.front() != 0.0F))
        throw std::invalid_argument("firwin2 filter type endpoint gain");
    prepared.nfreqs = options.nfreqs;
    if (prepared.nfreqs == 0) {
        prepared.nfreqs = 1;
        while (prepared.nfreqs < n) {
            if (prepared.nfreqs > std::numeric_limits<int>::max() / 2)
                throw std::invalid_argument("firwin2 numtaps too large");
            prepared.nfreqs <<= 1;
        }
        ++prepared.nfreqs;
    }
    if (prepared.nfreqs <= n ||
        prepared.nfreqs > (std::numeric_limits<int>::max() / 2 + 1))
        throw std::invalid_argument("firwin2 nfreqs must exceed numtaps");
    for (std::size_t index = 0; index + 1 < prepared.freq.size(); ++index) {
        if (prepared.freq[index] == prepared.freq[index + 1]) {
            const float repeated = prepared.freq[index];
            prepared.freq[index] = std::nextafter(
                repeated, -std::numeric_limits<float>::infinity());
            prepared.freq[index + 1] = std::nextafter(
                repeated, std::numeric_limits<float>::infinity());
        }
    }
    for (std::size_t index = 1; index < prepared.freq.size(); ++index)
        if (!(prepared.freq[index] > prepared.freq[index - 1]))
            throw std::invalid_argument("firwin2 repeated frequency too close");
    prepared.window.resize(n, 1.0F);
    if (options.window_mode == Firwin2WindowMode::hamming) {
        for (int tap = 0; tap < n; ++tap)
            prepared.window[tap] = n == 1 ? 1.0F
                : 0.54F - 0.46F * std::cos(
                    2.0F * static_cast<float>(pi) * tap / (n - 1));
    } else if (options.window_mode == Firwin2WindowMode::explicit_values) {
        if (options.window.size() != static_cast<std::size_t>(n))
            throw std::invalid_argument("firwin2 explicit window length");
        prepared.window = options.window;
    } else if (options.window_mode != Firwin2WindowMode::none) {
        throw std::invalid_argument("firwin2 window mode");
    }
    return prepared;
}
}

void firwin_fp32_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    DeviceArray<float>&, DeviceArray<float>&, bool);
void firwin_fp64_storage_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    DeviceArray<float>&, void*, std::size_t, bool);
void firwin_fused_fp64_storage_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    void*, std::size_t, bool);
void firwin2_fp32_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    const DeviceArray<float>&, DeviceArray<float>&, int, bool, float);

#define DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(T) \
FirwinDeviceWorkspace::FirwinDeviceWorkspace( \
    int n, const std::vector<T>& cutoff, const FirwinOptions& options) \
    : numtaps_(n) \
{ \
    const auto prepared = prepare_firwin(n, cutoff, options); \
    scale_ = prepared.scale; \
    bands_ = DeviceArray<float>::from_host(std::vector<float>( \
        prepared.bands.begin(), prepared.bands.end())); \
    window_ = DeviceArray<float>::from_host(std::vector<float>( \
        prepared.window.begin(), prepared.window.end())); \
    normalization_.reset(1); \
}
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(float)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(__half)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int32_t)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int16_t)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int8_t)
#undef DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR

void firwin_resident_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<float>& out)
{
    if (out.size() != static_cast<std::size_t>(workspace.numtaps_))
        throw std::invalid_argument("firwin resident output shape");
    firwin_fp32_compute_device(
        workspace.numtaps_, workspace.bands_, workspace.window_,
        workspace.normalization_, out, workspace.scale_);
}

void firwin_resident_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out)
{
    if (out.size() != static_cast<std::size_t>(workspace.numtaps_))
        throw std::invalid_argument("firwin resident FP64 storage output shape");
    static_assert(sizeof(double) == sizeof(std::uint64_t),
        "FP64 storage must be 64-bit");
    if (workspace.numtaps_ <= 1024) {
        firwin_fused_fp64_storage_compute_device(
            workspace.numtaps_, workspace.bands_, workspace.window_,
            out.data(), out.size(), workspace.scale_);
    } else {
        firwin_fp64_storage_compute_device(
            workspace.numtaps_, workspace.bands_, workspace.window_,
            workspace.normalization_, out.data(), out.size(), workspace.scale_);
    }
}

void firwin_resident_stage_c_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out)
{
    if (out.size() != static_cast<std::size_t>(workspace.numtaps_))
        throw std::invalid_argument("firwin stage C FP64 storage output shape");
    firwin_fp64_storage_compute_device(
        workspace.numtaps_, workspace.bands_, workspace.window_,
        workspace.normalization_, out.data(), out.size(), workspace.scale_);
}

template <typename T>
std::vector<double> firwin_typed_cpu(
    int n, const std::vector<T>& cutoff, const FirwinOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin dtype");
    const auto prepared = prepare_firwin(n, cutoff, options);
    const auto coefficient = [&](double m) {
        double value = 0.0;
        for (std::size_t band = 0; band < prepared.bands.size(); band += 2) {
            const double left = prepared.bands[band];
            const double right = prepared.bands[band + 1];
            value += right * sinc(right * m) - left * sinc(left * m);
        }
        return value;
    };
    std::vector<double> work(n); const double mid = 0.5 * (n - 1);
    for (int i = 0; i < n; ++i) {
        const double m = i - mid;
        work[i] = coefficient(m) * prepared.window[i];
    }
    if (prepared.scale) {
        double normalization = 0.0;
        const double left = prepared.bands[0];
        const double right = prepared.bands[1];
        const double scale_frequency = left == 0.0 ? 0.0
            : (right == 1.0 ? 1.0 : 0.5 * (left + right));
        for (int i = 0; i < n; ++i) {
            normalization += work[i] * std::cos(pi * (i - mid) * scale_frequency);
        }
        for (double& value : work) value /= normalization;
    }
    return work;
}

template <typename T>
void firwin_device(
    int n, const DeviceArray<T>& cutoff, DeviceArray<double>& out,
    const FirwinOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin dtype");
    if (out.size() != static_cast<std::size_t>(n > 0 ? n : 0))
        throw std::invalid_argument("firwin output shape");
    const auto prepared = prepare_firwin(n, cutoff.to_host(), options);
    std::vector<float> fp32_bands(
        prepared.bands.begin(), prepared.bands.end());
    std::vector<float> fp32_window(
        prepared.window.begin(), prepared.window.end());
    auto bands = DeviceArray<float>::from_host(fp32_bands);
    auto window = DeviceArray<float>::from_host(fp32_window);
    DeviceArray<float> normalization;
    if (n > 1024) normalization.reset(1);
    static_assert(sizeof(double) == sizeof(std::uint64_t),
        "FP64 storage must be 64-bit");
    if (n <= 1024) {
        firwin_fused_fp64_storage_compute_device(
            n, bands, window, out.data(), out.size(), prepared.scale);
    } else {
        firwin_fp64_storage_compute_device(
            n, bands, window, normalization, out.data(), out.size(), prepared.scale);
    }
}

template <typename T>
std::vector<double> firwin2_typed_cpu(
    int n, const std::vector<T>& freq, const std::vector<T>& gain,
    const Firwin2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin2 dtype");
    const auto prepared = prepare_firwin2(n, freq, gain, options);
    const int total = 2 * (prepared.nfreqs - 1);
    const double spacing = prepared.nyquist / (prepared.nfreqs - 1);
    std::vector<std::complex<double>> spectrum(total);
    for (int index = 0; index < prepared.nfreqs; ++index) {
        const double frequency = index * spacing;
        std::size_t segment = 0;
        for (std::size_t candidate = prepared.freq.size() - 1; candidate-- > 0;) {
            if (frequency >= prepared.freq[candidate]) {
                segment = candidate;
                break;
            }
        }
        if (segment + 1 >= prepared.freq.size()) segment = prepared.freq.size() - 2;
        const double left = prepared.freq[segment];
        const double right = prepared.freq[segment + 1];
        const double ratio = std::fabs(right - left) > 1.0e-30
            ? (frequency - left) / (right - left)
            : 0.0;
        const double response = prepared.gain[segment] * (1.0 - ratio)
            + prepared.gain[segment + 1] * ratio;
        const double phase = -0.5 * (n - 1) * pi * frequency / prepared.nyquist;
        const std::complex<double> shift = prepared.filter_type > 2
            ? std::complex<double>(-std::sin(phase), std::cos(phase))
            : std::complex<double>(std::cos(phase), std::sin(phase));
        spectrum[index] = response * shift;
    }
    for (int index = 1; index < prepared.nfreqs - 1; ++index) {
        spectrum[total - index] = std::conj(spectrum[index]);
    }
    std::vector<double> out(n);
    for (int tap = 0; tap < n; ++tap) {
        std::complex<double> time_value(0.0, 0.0);
        for (int frequency = 0; frequency < total; ++frequency) {
            const double angle = 2.0 * pi * tap * frequency / total;
            time_value += spectrum[frequency]
                * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        time_value /= static_cast<double>(total);
        double value = time_value.real() * prepared.window[tap];
        if (prepared.filter_type == 3 && tap == n / 2) value = 0.0;
        out[tap] = value;
    }
    return out;
}

template <typename T>
void firwin2_device(
    int n, const DeviceArray<T>& freq, const DeviceArray<T>& gain,
    DeviceArray<double>& out, const Firwin2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin2 dtype");
    if (out.size() != static_cast<std::size_t>(n > 0 ? n : 0))
        throw std::invalid_argument("firwin2 output shape");
    const auto prepared = prepare_firwin2(
        n, freq.to_host(), gain.to_host(), options);
    auto device_freq = DeviceArray<float>::from_host(prepared.freq);
    auto device_gain = DeviceArray<float>::from_host(prepared.gain);
    auto device_window = DeviceArray<float>::from_host(prepared.window);
    DeviceArray<float> computed(static_cast<std::size_t>(n));
    firwin2_fp32_compute_device(
        n, device_freq, device_gain, device_window, computed,
        prepared.nfreqs, options.antisymmetric, prepared.nyquist);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}

#define INST(T) \
 template std::vector<double> firwin_typed_cpu( \
    int, const std::vector<T>&, const FirwinOptions&); \
 template void firwin_device( \
    int, const DeviceArray<T>&, DeviceArray<double>&, const FirwinOptions&); \
 template std::vector<double> firwin2_typed_cpu( \
    int, const std::vector<T>&, const std::vector<T>&, const Firwin2Options&); \
 template void firwin2_device( \
    int, const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<double>&, \
    const Firwin2Options&)
INST(float); INST(__half); INST(std::int32_t); INST(std::int16_t); INST(std::int8_t);
#undef INST
}  // namespace cusignal
