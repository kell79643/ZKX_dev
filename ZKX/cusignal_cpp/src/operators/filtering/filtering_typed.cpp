#include <cusignal/operators/filtering/filtering_typed.h>
#include <cusignal/runtime/host_output_finalize.h>
#include <cusignal/runtime/operator_type_dispatch.h>
#include <cusignal/operators/filter_design/filter_design_typed.h>
#include <cusignal/backends/fft/fft_interface_cpu.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cusignal {
namespace {
constexpr float pi = 3.14159265358979323846F;
template <typename T> float ld(T v){return detail::SimpleSignalTypePolicy<T>::load(v);}
template <typename T> T st(float v){return detail::SimpleSignalTypePolicy<T>::store(v);}
template <typename T> double wiener_host_square(T value)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned raw = static_cast<Unsigned>(value);
        const Unsigned squared = static_cast<Unsigned>(
            static_cast<unsigned long long>(raw) *
            static_cast<unsigned long long>(raw));
        constexpr int bits = static_cast<int>(sizeof(T) * 8);
        const unsigned long long encoded =
            static_cast<unsigned long long>(squared);
        const unsigned long long sign = 1ULL << (bits - 1);
        const unsigned long long modulus = 1ULL << bits;
        const long long signed_value = encoded >= sign
            ? static_cast<long long>(encoded) - static_cast<long long>(modulus)
            : static_cast<long long>(encoded);
        return static_cast<double>(signed_value);
    } else if constexpr (std::is_same_v<T, __half>) {
        const float loaded = __half2float(value);
        return static_cast<double>(
            __half2float(__float2half_rn(loaded * loaded)));
    } else {
        const float loaded = static_cast<float>(value);
        return static_cast<double>(loaded * loaded);
    }
}
template <typename T>
constexpr DecimateOutputDtype decimate_output_dtype(bool custom_coefficients)
{
    return !custom_coefficients || std::is_same_v<T, std::int32_t>
        ? DecimateOutputDtype::fp64
        : DecimateOutputDtype::fp32;
}
template <typename T>
constexpr ResampleOutputDtype resample_output_dtype()
{
    return std::is_integral_v<T>
        ? ResampleOutputDtype::fp64
        : ResampleOutputDtype::fp32;
}

template <typename T>
constexpr ResamplePolyOutputDtype resample_poly_identity_dtype()
{
    if constexpr (std::is_same_v<T, float>)
        return ResamplePolyOutputDtype::fp32;
    if constexpr (std::is_same_v<T, __half>)
        return ResamplePolyOutputDtype::fp16;
    if constexpr (std::is_same_v<T, std::int32_t>)
        return ResamplePolyOutputDtype::int32;
    if constexpr (std::is_same_v<T, std::int16_t>)
        return ResamplePolyOutputDtype::int16;
    return ResamplePolyOutputDtype::int8;
}

template <typename T>
constexpr UpfirdnOutputDtype upfirdn_output_dtype()
{
    return std::is_same_v<T, std::int32_t>
        ? UpfirdnOutputDtype::fp64
        : UpfirdnOutputDtype::fp32;
}

template <typename T>
constexpr ResamplePolyOutputDtype resample_poly_output_dtype(
    bool identity, bool custom_coefficients, int reduced_up)
{
    if (identity) return resample_poly_identity_dtype<T>();
    if (!custom_coefficients) return ResamplePolyOutputDtype::fp64;
    if constexpr (std::is_same_v<T, std::int32_t>)
        return ResamplePolyOutputDtype::fp64;
    if constexpr (std::is_same_v<T, std::int16_t> ||
                  std::is_same_v<T, std::int8_t>)
        return reduced_up < 32768
            ? ResamplePolyOutputDtype::fp32
            : ResamplePolyOutputDtype::fp64;
    return reduced_up < 65536
        ? ResamplePolyOutputDtype::fp32
        : ResamplePolyOutputDtype::fp64;
}

template <typename T>
T resample_integral_wrap(std::make_unsigned_t<T> raw)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
    constexpr int bits = static_cast<int>(sizeof(T) * 8);
    const unsigned long long encoded = static_cast<unsigned long long>(raw);
    const unsigned long long sign = 1ULL << (bits - 1);
    const unsigned long long modulus = 1ULL << bits;
    const long long decoded = encoded >= sign
        ? static_cast<long long>(encoded) - static_cast<long long>(modulus)
        : static_cast<long long>(encoded);
    return static_cast<T>(decoded);
}

template <typename Output, typename Input>
double resample_poly_scaled_integral(Input value, int up)
{
    static_assert(std::is_integral_v<Output> && std::is_signed_v<Output>);
    using Unsigned = std::make_unsigned_t<Output>;
    const Output promoted = static_cast<Output>(value);
    const Unsigned product = static_cast<Unsigned>(
        static_cast<unsigned long long>(static_cast<Unsigned>(promoted)) *
        static_cast<unsigned long long>(up));
    return static_cast<double>(resample_integral_wrap<Output>(product));
}

template <typename T>
double resample_poly_scaled_custom_value(T value, int up)
{
    if constexpr (std::is_same_v<T, float>) {
        return up < 65536
            ? static_cast<double>(static_cast<float>(value) * up)
            : static_cast<double>(value) * static_cast<double>(up);
    } else if constexpr (std::is_same_v<T, __half>) {
        const float loaded = __half2float(value);
        if (up <= 255)
            return static_cast<double>(__half2float(__float2half_rn(
                loaded * static_cast<float>(up))));
        return up < 65536
            ? static_cast<double>(loaded * static_cast<float>(up))
            : static_cast<double>(loaded) * static_cast<double>(up);
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return resample_poly_scaled_integral<std::int32_t>(value, up);
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return up < 32768
            ? resample_poly_scaled_integral<std::int16_t>(value, up)
            : resample_poly_scaled_integral<std::int32_t>(value, up);
    } else {
        if (up <= 127)
            return resample_poly_scaled_integral<std::int8_t>(value, up);
        return up < 32768
            ? resample_poly_scaled_integral<std::int16_t>(value, up)
            : resample_poly_scaled_integral<std::int32_t>(value, up);
    }
}

double resample_poly_bessel_i0(double value)
{
    const double absolute = std::fabs(value);
    if (absolute < 3.75) {
        const double ratio = value / 3.75;
        const double y = ratio * ratio;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y *
            (1.2067492 + y * (0.2659732 + y *
            (0.0360768 + y * 0.0045813)))));
    }
    const double y = 3.75 / absolute;
    return std::exp(absolute) / std::sqrt(absolute) *
        (0.39894228 + y * (0.01328592 + y * (0.00225319 + y *
        (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y *
        (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
}

std::vector<double> design_resample_poly_coefficients(
    const detail::ResamplePolyLayoutPlan& plan,
    const ResamplePolyOptions& options)
{
    const int count = plan.coefficient_count;
    const int half = plan.half_length;
    std::vector<double> window(static_cast<std::size_t>(count));
    switch (options.window_mode) {
    case ResamplePolyWindowMode::kaiser: {
        const double denominator = resample_poly_bessel_i0(
            options.kaiser_beta);
        for (int index = 0; index < count; ++index) {
            const double ratio = static_cast<double>(index - half) / half;
            const double bounded = std::max(0.0, 1.0 - ratio * ratio);
            window[index] = resample_poly_bessel_i0(
                options.kaiser_beta * std::sqrt(bounded)) / denominator;
        }
        break;
    }
    case ResamplePolyWindowMode::hamming:
        for (int index = 0; index < count; ++index)
            window[index] = 0.54 - 0.46 * std::cos(
                6.283185307179586476925286766559 * index / (count - 1));
        break;
    case ResamplePolyWindowMode::explicit_values:
        if (options.design_window.size() != static_cast<std::size_t>(count))
            throw std::invalid_argument("resample_poly design window shape");
        window = options.design_window;
        break;
    default:
        throw std::invalid_argument("resample_poly window mode");
    }
    const double cutoff = 1.0 / std::max(plan.up, plan.down);
    std::vector<double> coefficients(static_cast<std::size_t>(count));
    double scale = 0.0;
    for (int index = 0; index < count; ++index) {
        const int offset = index - half;
        const double sinc = offset == 0
            ? cutoff
            : std::sin(3.1415926535897932384626433832795 * cutoff * offset) /
                (3.1415926535897932384626433832795 * offset);
        coefficients[index] = sinc * window[index];
        scale += coefficients[index];
    }
    for (double& coefficient : coefficients)
        coefficient = coefficient / scale * plan.up;
    return coefficients;
}

template <typename T>
T resample_frequency_multiply(T value, double window)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const T typed_window = static_cast<T>(window);
        const Unsigned product = static_cast<Unsigned>(
            static_cast<unsigned long long>(static_cast<Unsigned>(value)) *
            static_cast<unsigned long long>(
                static_cast<Unsigned>(typed_window)));
        return resample_integral_wrap<T>(product);
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half typed_window = __float2half_rn(static_cast<float>(window));
        return __float2half_rn(
            __half2float(value) * __half2float(typed_window));
    } else {
        return static_cast<T>(value * static_cast<T>(window));
    }
}

template <typename T>
double resample_frequency_add(double left, double right)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const T lhs = static_cast<T>(left);
        const T rhs = static_cast<T>(right);
        const Unsigned sum = static_cast<Unsigned>(
            static_cast<unsigned long long>(static_cast<Unsigned>(lhs)) +
            static_cast<unsigned long long>(static_cast<Unsigned>(rhs)));
        return static_cast<double>(resample_integral_wrap<T>(sum));
    } else if constexpr (std::is_same_v<T, __half>) {
        return static_cast<double>(__half2float(__float2half_rn(
            static_cast<float>(left) + static_cast<float>(right))));
    } else {
        return static_cast<double>(
            static_cast<T>(static_cast<T>(left) + static_cast<T>(right)));
    }
}

template <typename T>
double resample_frequency_half(double value)
{
    if constexpr (std::is_same_v<T, __half>) {
        return static_cast<double>(__half2float(resample_frequency_multiply(
            __float2half_rn(static_cast<float>(value)), 0.5)));
    } else {
        return static_cast<double>(resample_frequency_multiply(
            static_cast<T>(value), 0.5));
    }
}

template <typename T>
double resample_time_delta(T first, T second)
{
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned difference = static_cast<Unsigned>(second) -
            static_cast<Unsigned>(first);
        return static_cast<double>(resample_integral_wrap<T>(difference));
    } else if constexpr (std::is_same_v<T, __half>) {
        return static_cast<double>(__half2float(__float2half_rn(
            __half2float(second) - __half2float(first))));
    } else {
        return static_cast<double>(second - first);
    }
}

template <typename T>
void validate_resample_frequency_integer_split(
    const detail::ResampleLayoutPlan& plan,
    const ResampleOptions& options)
{
    if constexpr (std::is_integral_v<T>) {
        const int copied = std::min(
            plan.input_sample_count, plan.output_sample_count);
        if (options.domain == ResampleDomain::frequency &&
            plan.input_sample_count < plan.output_sample_count &&
            copied % 2 == 0) {
            throw std::invalid_argument(
                "resample integer frequency-domain Nyquist half");
        }
    }
}

void validate_decimate_filter_selection(
    const DecimateOptions& options, bool custom_coefficients)
{
    if (custom_coefficients && options.filter_order.has_value())
        throw std::invalid_argument(
            "decimate filter_order and coefficients are mutually exclusive");
}

std::vector<double> design_decimate_filter(
    int q, const DecimateOptions& options)
{
    if (q < 1) throw std::invalid_argument("decimate q");
    int order = 0;
    if (options.filter_order.has_value()) {
        order = *options.filter_order;
    } else {
        if (q > std::numeric_limits<int>::max() / 20)
            throw std::invalid_argument("decimate default filter order overflow");
        order = 20 * q;
    }
    if (order < 0 || order == std::numeric_limits<int>::max())
        throw std::invalid_argument("decimate filter order");
    const std::vector<float> cutoff{1.0F / static_cast<float>(q)};
    FirwinOptions firwin_options;
    firwin_options.window_mode = FirwinWindowMode::hamming;
    return firwin_typed_cpu<float>(order + 1, cutoff, firwin_options);
}
template <typename T> double detrend_host_load(T value)
{
    if constexpr (std::is_same_v<T, __half>)
        return static_cast<double>(__half2float(value));
    else
        return static_cast<double>(value);
}
template <typename T>
std::enable_if_t<detail::is_simple_signal_input_v<T>> freq_shift_host_load(
    T value, double& real, double& imag)
{
    real = detrend_host_load(value);
    imag = 0.0;
}
template <typename T>
void freq_shift_host_load(
    const TypedComplex<T>& value, double& real, double& imag)
{
    real = detrend_host_load(value.real);
    imag = detrend_host_load(value.imag);
}

struct DetrendLayout {
    int sample_count{0};
    int inner_count{1};
    int batch_count{0};
    std::vector<int> breakpoints;
};

DetrendLayout prepare_detrend_layout(
    std::size_t element_count, const DetrendOptions& options)
{
    if (options.mode != DetrendMode::linear &&
        options.mode != DetrendMode::constant)
        throw std::invalid_argument("detrend mode");
    std::vector<int> shape = options.shape;
    if (shape.empty()) {
        if (element_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("detrend input exceeds int indexing");
        shape.push_back(static_cast<int>(element_count));
    }
    long long product = 1;
    for (const int extent : shape) {
        if (extent < 0 || product > std::numeric_limits<int>::max())
            throw std::invalid_argument("detrend shape");
        product *= extent;
    }
    if (product != static_cast<long long>(element_count) ||
        product > std::numeric_limits<int>::max())
        throw std::invalid_argument("detrend shape product");
    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(shape.size());
    if (axis < 0 || axis >= static_cast<int>(shape.size()))
        throw std::invalid_argument("detrend axis");
    DetrendLayout layout;
    layout.sample_count = shape[axis];
    long long inner = 1;
    for (std::size_t index = static_cast<std::size_t>(axis + 1);
         index < shape.size(); ++index) {
        inner *= shape[index];
    }
    if (inner > std::numeric_limits<int>::max())
        throw std::invalid_argument("detrend inner stride");
    layout.inner_count = static_cast<int>(inner);
    long long batches = 1;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (static_cast<int>(index) != axis) batches *= shape[index];
    }
    if (batches > std::numeric_limits<int>::max())
        throw std::invalid_argument("detrend batch count");
    layout.batch_count = static_cast<int>(batches);
    if (options.mode == DetrendMode::linear) {
        if (layout.sample_count <= 0)
            throw std::invalid_argument("detrend linear axis is empty");
        layout.breakpoints = options.breakpoints;
        layout.breakpoints.push_back(0);
        layout.breakpoints.push_back(layout.sample_count);
        std::sort(layout.breakpoints.begin(), layout.breakpoints.end());
        layout.breakpoints.erase(
            std::unique(layout.breakpoints.begin(), layout.breakpoints.end()),
            layout.breakpoints.end());
        if (layout.breakpoints.front() < 0 ||
            layout.breakpoints.back() > layout.sample_count)
            throw std::invalid_argument("detrend breakpoint");
        layout.breakpoints.erase(layout.breakpoints.begin());
        layout.breakpoints.pop_back();
    }
    return layout;
}

template <typename T>
DetrendOutputDtype detrend_output_dtype(DetrendMode mode)
{
    if constexpr (std::is_same_v<T, float>) {
        return DetrendOutputDtype::fp32;
    } else if constexpr (std::is_same_v<T, __half>) {
        return mode == DetrendMode::constant
            ? DetrendOutputDtype::fp16
            : DetrendOutputDtype::fp64;
    } else {
        return DetrendOutputDtype::fp64;
    }
}
}

namespace detail {
ChannelizePolyPlan prepare_channelize_poly(
    std::size_t input_size, std::size_t filter_size, int n_chans)
{
    if (n_chans <= 0)
        throw std::invalid_argument("channelize_poly n_chans");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        filter_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("channelize_poly input exceeds int indexing");
    ChannelizePolyPlan plan;
    plan.channel_count = n_chans;
    plan.input_count = static_cast<int>(input_size);
    plan.tap_count = static_cast<int>(filter_size /
        static_cast<std::size_t>(n_chans));
    if (plan.tap_count > 32)
        throw std::invalid_argument("channelize_poly taps per phase exceed 32");
    plan.point_count = static_cast<int>(input_size /
        static_cast<std::size_t>(n_chans));
    plan.output_size = static_cast<std::size_t>(plan.channel_count)
        * static_cast<std::size_t>(plan.point_count);
    if (plan.output_size > static_cast<std::size_t>(
            std::numeric_limits<int>::max()))
        throw std::invalid_argument("channelize_poly output exceeds int indexing");
    return plan;
}


ChannelizePolyPlan prepare_channelize_poly(
    std::size_t input_size, std::size_t filter_size, int n_chans,
    const std::vector<int>& input_shape,
    const std::vector<int>& filter_shape)
{
    auto first_extent = [](const std::vector<int>& shape,
                           std::size_t data_size) -> std::size_t {
        if (shape.empty()) throw std::invalid_argument("channelize_poly shape");
        std::size_t product = 1;
        bool zero = false;
        for (int extent : shape) {
            if (extent < 0) throw std::invalid_argument("channelize_poly shape");
            if (extent == 0) zero = true;
            else if (!zero) {
                if (product > data_size / static_cast<std::size_t>(extent))
                    throw std::invalid_argument("channelize_poly data/shape mismatch");
                product *= static_cast<std::size_t>(extent);
            }
        }
        if ((zero ? 0 : product) != data_size)
            throw std::invalid_argument("channelize_poly data/shape mismatch");
        const auto first = static_cast<std::size_t>(shape.front());
        if (first > data_size)
            throw std::invalid_argument("channelize_poly first dimension exceeds storage");
        return first;
    };
    return prepare_channelize_poly(
        first_extent(input_shape, input_size),
        first_extent(filter_shape, filter_size), n_chans);
}

HilbertLayoutPlan prepare_hilbert_layout(
    std::size_t input_size, const HilbertOptions& options)
{
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("hilbert input exceeds int indexing");
    HilbertLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("hilbert input shape");
        if (product != 0 && extent != 0 &&
            product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("hilbert input shape overflow");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("hilbert input shape product");
    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("hilbert axis");
    plan.axis = axis;
    plan.input_axis_length = plan.input_shape[axis];
    plan.fft_length = options.fft_length.has_value()
        ? *options.fft_length : plan.input_axis_length;
    if (plan.fft_length <= 0)
        throw std::invalid_argument("hilbert N must be positive");

    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension) {
        const int extent = plan.input_shape[dimension];
        if (outer != 0 && extent != 0 &&
            outer > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("hilbert outer shape overflow");
        outer *= plan.input_shape[dimension];
    }
    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension) {
        const int extent = plan.input_shape[dimension];
        if (inner != 0 && extent != 0 &&
            inner > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("hilbert inner shape overflow");
        inner *= extent;
    }
    if (outer > std::numeric_limits<int>::max() ||
        inner > std::numeric_limits<int>::max() ||
        outer * inner > std::numeric_limits<int>::max())
        throw std::invalid_argument("hilbert line indexing overflow");
    plan.outer_count = static_cast<int>(outer);
    plan.inner_count = static_cast<int>(inner);
    plan.line_count = static_cast<int>(outer * inner);
    const long long output_size = static_cast<long long>(plan.line_count)
        * plan.fft_length;
    if (output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("hilbert output exceeds int indexing");
    plan.output_size = static_cast<std::size_t>(output_size);
    plan.output_shape = plan.input_shape;
    plan.output_shape[axis] = plan.fft_length;
    return plan;
}

Hilbert2LayoutPlan prepare_hilbert2_layout(
    std::size_t input_size, const Hilbert2Options& options)
{
    if (options.shape.size() != 2)
        throw std::invalid_argument("hilbert2 input must be 2-D");
    Hilbert2LayoutPlan plan;
    plan.input_rows = options.shape[0];
    plan.input_cols = options.shape[1];
    if (plan.input_rows < 2 || plan.input_cols < 0)
        throw std::invalid_argument("hilbert2 fixed cuSignal path requires two rows");
    const long long input_product = static_cast<long long>(plan.input_rows)
        * plan.input_cols;
    if (input_product > std::numeric_limits<int>::max() ||
        input_product != static_cast<long long>(input_size))
        throw std::invalid_argument("hilbert2 input shape product");

    if (options.fft_shape.empty()) {
        plan.fft_rows = plan.input_rows;
        plan.fft_cols = plan.input_cols;
    } else if (options.fft_shape.size() == 1) {
        plan.fft_rows = options.fft_shape[0];
        plan.fft_cols = options.fft_shape[0];
    } else if (options.fft_shape.size() == 2) {
        plan.fft_rows = options.fft_shape[0];
        plan.fft_cols = options.fft_shape[1];
    } else {
        throw std::invalid_argument("hilbert2 N must be scalar or pair");
    }
    if (plan.fft_rows <= 0 || plan.fft_cols <= 0)
        throw std::invalid_argument("hilbert2 N must contain positive values");
    if (plan.fft_rows != plan.fft_cols && plan.fft_rows != 1)
        throw std::invalid_argument("hilbert2 fixed square mask cannot broadcast");
    plan.output_rows = plan.fft_rows == 1
        ? plan.fft_cols : plan.fft_rows;
    plan.output_cols = plan.fft_cols;
    const long long fft_size = static_cast<long long>(plan.fft_rows)
        * plan.fft_cols;
    const long long output_size = static_cast<long long>(plan.output_rows)
        * plan.output_cols;
    if (fft_size > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("hilbert2 FFT shape exceeds int indexing");
    plan.fft_size = static_cast<std::size_t>(fft_size);
    plan.output_size = static_cast<std::size_t>(output_size);
    return plan;
}

LfilterZiLayoutPlan prepare_lfilter_zi_layout(
    std::size_t numerator_count, std::size_t denominator_count,
    std::size_t denominator_offset)
{
    if (denominator_count == 0 || denominator_offset >= denominator_count)
        throw std::invalid_argument("lfilter_zi denominator");
    if (numerator_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        denominator_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("lfilter_zi coefficient count exceeds int indexing");

    LfilterZiLayoutPlan plan;
    plan.denominator_offset = static_cast<int>(denominator_offset);
    plan.denominator_count = static_cast<int>(denominator_count - denominator_offset);
    plan.common_count = std::max(
        static_cast<int>(numerator_count), plan.denominator_count);
    if (plan.common_count < 2)
        throw std::invalid_argument("lfilter_zi companion requires two coefficients");
    plan.output_count = plan.common_count - 1;
    const unsigned long long augmented_count =
        static_cast<unsigned long long>(plan.output_count) * plan.common_count;
    if (augmented_count >
        static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("lfilter_zi workspace exceeds int indexing");
    plan.augmented_count = static_cast<std::size_t>(augmented_count);
    return plan;
}

SosfiltLayoutPlan prepare_sosfilt_layout(
    std::size_t input_size, std::size_t sos_size, int sections,
    const SosfiltOptions& options, bool has_zi, std::size_t zi_size)
{
    if (sections < 1 || sections > 512)
        throw std::invalid_argument("sosfilt sections");
    if (sos_size != static_cast<std::size_t>(sections) * 6U)
        throw std::invalid_argument("sosfilt sos shape");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("sosfilt input exceeds int indexing");

    SosfiltLayoutPlan plan;
    plan.sections = sections;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));

    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("sosfilt input shape");
        if (product != 0 && extent != 0 &&
            product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("sosfilt input shape overflow");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("sosfilt input shape product");

    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("sosfilt axis");
    plan.axis = axis;
    plan.sample_count = plan.input_shape[axis];
    if (sections > plan.sample_count)
        throw std::invalid_argument("sosfilt sections exceed samples");

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        outer *= plan.input_shape[dimension];
    const long long lines = outer * inner;
    if (inner > std::numeric_limits<int>::max() ||
        outer > std::numeric_limits<int>::max() ||
        lines > std::numeric_limits<int>::max())
        throw std::invalid_argument("sosfilt layout exceeds int indexing");
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(outer);
    plan.line_count = static_cast<int>(lines);

    plan.state_shape.reserve(plan.input_shape.size() + 1);
    plan.state_shape.push_back(sections);
    plan.state_shape.insert(
        plan.state_shape.end(), plan.input_shape.begin(), plan.input_shape.end());
    plan.state_shape[static_cast<std::size_t>(axis + 1)] = 2;
    const unsigned long long state_size =
        static_cast<unsigned long long>(sections) *
        static_cast<unsigned long long>(plan.line_count) * 2ULL;
    if (state_size >
        static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("sosfilt state exceeds int indexing");
    plan.state_size = static_cast<std::size_t>(state_size);

    if (!has_zi) {
        if (!options.zi_shape.empty() || zi_size != 0)
            throw std::invalid_argument("sosfilt zi metadata without zi");
    } else {
        if (options.zi_shape != plan.state_shape)
            throw std::invalid_argument("sosfilt zi shape");
        if (zi_size != plan.state_size)
            throw std::invalid_argument("sosfilt zi shape product");
    }
    return plan;
}

void validate_sos_coefficients(
    const std::vector<float>& coefficients, int sections)
{
    if (sections < 1 ||
        coefficients.size() != static_cast<std::size_t>(sections) * 6U)
        throw std::invalid_argument("sosfilt sos shape");
    for (int section = 0; section < sections; ++section) {
        if (coefficients[static_cast<std::size_t>(section) * 6U + 3U] != 1.0F)
            throw std::invalid_argument("sosfilt sos a0 must equal one");
    }
}

WienerLayoutPlan prepare_wiener_layout(
    std::size_t input_size, const WienerOptions& options)
{
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("wiener input exceeds int indexing");
    if (options.scalar_window.has_value() && !options.window_shape.empty())
        throw std::invalid_argument("wiener window forms are mutually exclusive");

    WienerLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    plan.rank = static_cast<int>(plan.input_shape.size());
    if (plan.rank <= 0)
        throw std::invalid_argument("wiener input rank");

    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("wiener input shape");
        if (product != 0 && extent != 0 &&
            product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("wiener input shape overflow");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("wiener input shape product");
    plan.element_count = static_cast<int>(input_size);

    if (options.scalar_window.has_value()) {
        plan.window_shape.assign(plan.rank, *options.scalar_window);
    } else if (options.window_shape.empty()) {
        plan.window_shape.assign(plan.rank, 3);
    } else {
        if (options.window_shape.size() != plan.input_shape.size())
            throw std::invalid_argument("wiener window rank");
        plan.window_shape = options.window_shape;
    }
    long long window_volume = 1;
    for (const int extent : plan.window_shape) {
        if (extent <= 0 ||
            window_volume > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("wiener window shape");
        window_volume *= extent;
    }
    plan.window_volume = static_cast<int>(window_volume);

    plan.input_strides.resize(plan.rank);
    long long stride = 1;
    for (int dimension = plan.rank - 1; dimension >= 0; --dimension) {
        if (stride > std::numeric_limits<int>::max())
            throw std::invalid_argument("wiener input stride");
        plan.input_strides[dimension] = static_cast<int>(stride);
        stride *= plan.input_shape[dimension];
    }
    return plan;
}

DecimateLayoutPlan prepare_decimate_layout(
    std::size_t input_size, int q, const DecimateOptions& options)
{
    if (q < 1) throw std::invalid_argument("decimate q");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("decimate input exceeds int indexing");

    DecimateLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    if (plan.input_shape.empty())
        throw std::invalid_argument("decimate input rank");

    long long input_product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("decimate input shape");
        if (input_product != 0 && extent != 0 &&
            input_product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("decimate input shape overflow");
        input_product *= extent;
    }
    if (input_product != static_cast<long long>(input_size))
        throw std::invalid_argument("decimate input shape product");

    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("decimate axis");
    plan.axis = axis;
    plan.sample_count = plan.input_shape[axis];
    plan.output_sample_count = plan.sample_count == 0
        ? 0
        : 1 + (plan.sample_count - 1) / q;
    plan.output_shape = plan.input_shape;
    plan.output_shape[axis] = plan.output_sample_count;

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension) {
        const int extent = plan.input_shape[dimension];
        if (inner != 0 && extent != 0 &&
            inner > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("decimate inner shape overflow");
        inner *= extent;
    }
    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension) {
        const int extent = plan.input_shape[dimension];
        if (outer != 0 && extent != 0 &&
            outer > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("decimate outer shape overflow");
        outer *= extent;
    }
    const long long lines = inner * outer;
    const long long output_size = lines * plan.output_sample_count;
    if (inner > std::numeric_limits<int>::max() ||
        outer > std::numeric_limits<int>::max() ||
        lines > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("decimate output shape overflow");
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(outer);
    plan.line_count = static_cast<int>(lines);
    plan.output_size = static_cast<std::size_t>(output_size);
    return plan;
}

ResampleLayoutPlan prepare_resample_layout(
    std::size_t input_size, int output_sample_count,
    const ResampleOptions& options)
{
    if (output_sample_count <= 0)
        throw std::invalid_argument("resample output sample count");
    if (options.domain != ResampleDomain::time &&
        options.domain != ResampleDomain::frequency)
        throw std::invalid_argument("resample domain");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("resample input exceeds int indexing");

    ResampleLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    long long input_product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("resample input shape");
        if (input_product != 0 && extent != 0 &&
            input_product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("resample input shape overflow");
        input_product *= extent;
    }
    if (input_product != static_cast<long long>(input_size))
        throw std::invalid_argument("resample input shape product");

    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("resample axis");
    plan.axis = axis;
    plan.input_sample_count = plan.input_shape[axis];
    if (plan.input_sample_count <= 0)
        throw std::invalid_argument("resample selected axis is empty");
    plan.output_sample_count = output_sample_count;
    if (!options.frequency_window.empty() &&
        options.frequency_window.size() !=
            static_cast<std::size_t>(plan.input_sample_count))
        throw std::invalid_argument("resample window shape");

    plan.output_shape = plan.input_shape;
    plan.output_shape[axis] = output_sample_count;
    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension) {
        const int extent = plan.input_shape[dimension];
        if (inner != 0 && extent != 0 &&
            inner > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("resample inner shape overflow");
        inner *= extent;
    }
    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension) {
        const int extent = plan.input_shape[dimension];
        if (outer != 0 && extent != 0 &&
            outer > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("resample outer shape overflow");
        outer *= extent;
    }
    const long long lines = inner * outer;
    const long long input_workspace = lines * plan.input_sample_count;
    const long long output_size = lines * output_sample_count;
    if (lines > std::numeric_limits<int>::max() ||
        input_workspace > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("resample workspace shape overflow");
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(outer);
    plan.line_count = static_cast<int>(lines);
    plan.input_workspace_size = static_cast<std::size_t>(input_workspace);
    plan.output_size = static_cast<std::size_t>(output_size);
    return plan;
}

ResamplePolyLayoutPlan prepare_resample_poly_layout(
    std::size_t input_size, int up, int down,
    const ResamplePolyOptions& options,
    bool custom_coefficients, std::size_t coefficient_count)
{
    if (up < 1 || down < 1)
        throw std::invalid_argument("resample_poly factors");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        coefficient_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("resample_poly input indexing");

    ResamplePolyLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    long long input_product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("resample_poly input shape");
        if (input_product != 0 && extent != 0 &&
            input_product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("resample_poly input shape overflow");
        input_product *= extent;
    }
    if (input_product != static_cast<long long>(input_size))
        throw std::invalid_argument("resample_poly input shape product");

    const int divisor = std::gcd(up, down);
    plan.up = up / divisor;
    plan.down = down / divisor;
    plan.identity = plan.up == 1 && plan.down == 1;
    plan.custom_coefficients = custom_coefficients;
    plan.output_shape = plan.input_shape;
    plan.output_size = input_size;
    if (plan.identity) return plan;

    if (plan.input_shape.size() < 1 || plan.input_shape.size() > 2)
        throw std::invalid_argument("resample_poly nonidentity rank");
    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("resample_poly axis");
    plan.axis = axis;
    plan.input_sample_count = plan.input_shape[axis];

    const long long scaled = static_cast<long long>(plan.input_sample_count) *
        plan.up;
    const long long output_count = (scaled + plan.down - 1) / plan.down;
    if (output_count > std::numeric_limits<int>::max())
        throw std::invalid_argument("resample_poly output sample count");
    plan.output_sample_count = static_cast<int>(output_count);
    plan.output_shape[axis] = plan.output_sample_count;

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        outer *= plan.input_shape[dimension];
    const long long lines = inner * outer;
    const long long output_size = lines * output_count;
    if (inner > std::numeric_limits<int>::max() ||
        outer > std::numeric_limits<int>::max() ||
        lines > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("resample_poly workspace shape");
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(outer);
    plan.line_count = static_cast<int>(lines);
    plan.output_size = static_cast<std::size_t>(output_size);

    if (custom_coefficients) {
        plan.coefficient_count = static_cast<int>(coefficient_count);
        plan.half_length = plan.coefficient_count == 0
            ? -1
            : (plan.coefficient_count - 1) / 2;
    } else {
        const int maximum_rate = std::max(plan.up, plan.down);
        if (maximum_rate >
            (std::numeric_limits<int>::max() - 1) / 20)
            throw std::invalid_argument("resample_poly designed filter length");
        plan.half_length = 10 * maximum_rate;
        plan.coefficient_count = 2 * plan.half_length + 1;
        if (options.window_mode == ResamplePolyWindowMode::explicit_values &&
            options.design_window.size() !=
                static_cast<std::size_t>(plan.coefficient_count))
            throw std::invalid_argument("resample_poly design window shape");
        if (options.window_mode != ResamplePolyWindowMode::kaiser &&
            options.window_mode != ResamplePolyWindowMode::hamming &&
            options.window_mode != ResamplePolyWindowMode::explicit_values)
            throw std::invalid_argument("resample_poly window mode");
    }

    const int remainder = ((plan.half_length % plan.down) + plan.down) %
        plan.down;
    plan.prefix_zeros = plan.down - remainder;
    plan.output_start = (plan.half_length + plan.prefix_zeros) / plan.down;
    const long long target = static_cast<long long>(plan.output_start) +
        plan.output_sample_count;
    const long long minimum_padded_count = (target - 1) * plan.down -
        (static_cast<long long>(plan.input_sample_count) - 1) * plan.up + 1;
    const long long base_padded_count = static_cast<long long>(
        plan.prefix_zeros) + plan.coefficient_count;
    const long long suffix = std::max(0LL,
        minimum_padded_count - base_padded_count);
    if (suffix > std::numeric_limits<int>::max())
        throw std::invalid_argument("resample_poly suffix padding");
    plan.suffix_zeros = static_cast<int>(suffix);
    return plan;
}

UpfirdnLayoutPlan prepare_upfirdn_layout(
    std::size_t input_size, std::size_t coefficient_count,
    int up, int down, const UpfirdnOptions& options)
{
    if (coefficient_count == 0)
        throw std::invalid_argument("upfirdn coefficients");
    if (up < 1 || down < 1)
        throw std::invalid_argument("upfirdn factors");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        coefficient_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("upfirdn indexing");

    UpfirdnLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    if (plan.input_shape.size() < 1 || plan.input_shape.size() > 2)
        throw std::invalid_argument("upfirdn rank");
    long long input_product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("upfirdn input shape");
        if (input_product != 0 && extent != 0 &&
            input_product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("upfirdn input shape overflow");
        input_product *= extent;
    }
    if (input_product != static_cast<long long>(input_size))
        throw std::invalid_argument("upfirdn input shape product");

    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("upfirdn axis");
    plan.axis = axis;
    plan.up = up;
    plan.down = down;
    plan.coefficient_count = static_cast<int>(coefficient_count);
    plan.input_sample_count = plan.input_shape[axis];
    const long long numerator =
        (static_cast<long long>(plan.input_sample_count) - 1) * up +
        plan.coefficient_count - 1;
    long long quotient = numerator / down;
    if (numerator % down < 0) --quotient;
    const long long output_count = quotient + 1;
    if (output_count < 0 || output_count > std::numeric_limits<int>::max())
        throw std::invalid_argument("upfirdn output sample count");
    plan.output_sample_count = static_cast<int>(output_count);
    plan.output_shape = plan.input_shape;
    plan.output_shape[axis] = plan.output_sample_count;

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        outer *= plan.input_shape[dimension];
    const long long lines = inner * outer;
    const long long output_size = lines * output_count;
    if (inner > std::numeric_limits<int>::max() ||
        outer > std::numeric_limits<int>::max() ||
        lines > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("upfirdn workspace shape");
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(outer);
    plan.line_count = static_cast<int>(lines);
    plan.output_size = static_cast<std::size_t>(output_size);
    return plan;
}

DeviceArray<double> finalize_upfirdn_fp64_output(
    const DeviceArray<float>& computed)
{
    return cuda_utils::finalize_fp64_device_storage(computed);
}

FirfilterLayoutPlan prepare_firfilter_layout(
    std::size_t input_size, std::size_t coefficient_count,
    const FirfilterOptions& options, bool has_zi, std::size_t zi_size)
{
    if (coefficient_count == 0 ||
        coefficient_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("firfilter coefficients");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("firfilter input exceeds int indexing");

    FirfilterLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent <= 0 || product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("firfilter input shape");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("firfilter input shape product");

    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("firfilter axis");
    plan.sample_count = plan.input_shape[axis];
    plan.state_count = static_cast<int>(coefficient_count) - 1;
    plan.state_shape = plan.input_shape;
    plan.state_shape[axis] = plan.state_count;

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(input_size /
        static_cast<std::size_t>(plan.sample_count * plan.inner_count));

    if (!has_zi) {
        if (!options.zi_shape.empty())
            throw std::invalid_argument("firfilter zi_shape without zi");
        if (zi_size != 0)
            throw std::invalid_argument("firfilter zi size without zi");
        return plan;
    }

    std::vector<int> zi_shape = options.zi_shape.empty()
        ? plan.state_shape : options.zi_shape;
    if (zi_shape.size() != plan.input_shape.size())
        throw std::invalid_argument("firfilter zi rank");
    long long zi_product = 1;
    for (std::size_t dimension = 0; dimension < zi_shape.size(); ++dimension) {
        const int extent = zi_shape[dimension];
        const int expected = plan.state_shape[dimension];
        if (extent < 0 ||
            (static_cast<int>(dimension) == axis
                ? extent != expected
                : extent != expected && extent != 1))
            throw std::invalid_argument("firfilter zi broadcast shape");
        if (zi_product != 0 && extent != 0 &&
            zi_product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("firfilter zi shape overflow");
        zi_product *= extent;
    }
    if (zi_product != static_cast<long long>(zi_size))
        throw std::invalid_argument("firfilter zi shape product");

    std::vector<int> zi_strides(zi_shape.size(), 1);
    for (int dimension = static_cast<int>(zi_shape.size()) - 2;
         dimension >= 0; --dimension)
        zi_strides[dimension] = zi_strides[dimension + 1]
            * zi_shape[dimension + 1];
    plan.zi_state_stride = zi_strides[axis];
    if (plan.state_count == 0) return plan;

    const int line_count = plan.outer_count * plan.inner_count;
    plan.zi_line_offsets.resize(line_count);
    for (int line = 0; line < line_count; ++line) {
        int outer = line / plan.inner_count;
        int inner_index = line % plan.inner_count;
        int offset = 0;
        for (int dimension = axis - 1; dimension >= 0; --dimension) {
            const int coordinate = outer % plan.input_shape[dimension];
            outer /= plan.input_shape[dimension];
            if (zi_shape[dimension] != 1)
                offset += coordinate * zi_strides[dimension];
        }
        for (int dimension = static_cast<int>(plan.input_shape.size()) - 1;
             dimension > axis; --dimension) {
            const int coordinate = inner_index % plan.input_shape[dimension];
            inner_index /= plan.input_shape[dimension];
            if (zi_shape[dimension] != 1)
                offset += coordinate * zi_strides[dimension];
        }
        plan.zi_line_offsets[line] = offset;
    }
    return plan;
}

Firfilter2LayoutPlan prepare_firfilter2_layout(
    std::size_t input_size, std::size_t coefficient_count,
    const Firfilter2Options& options)
{
    if (coefficient_count == 0 ||
        coefficient_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("firfilter2 coefficients");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("firfilter2 input exceeds int indexing");
    if (options.method != Firfilter2Method::pad &&
        options.method != Firfilter2Method::gust)
        throw std::invalid_argument("firfilter2 method");
    if (options.method == Firfilter2Method::gust)
        throw std::invalid_argument("firfilter2 gust is not supported by cuSignal 23.08.00");
    if (options.padtype != Firfilter2PadType::odd &&
        options.padtype != Firfilter2PadType::even &&
        options.padtype != Firfilter2PadType::constant &&
        options.padtype != Firfilter2PadType::none)
        throw std::invalid_argument("firfilter2 padtype");

    Firfilter2LayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent <= 0 || product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("firfilter2 input shape");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("firfilter2 input shape product");

    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("firfilter2 axis");
    plan.axis = axis;
    plan.sample_count = plan.input_shape[axis];
    plan.state_count = static_cast<int>(coefficient_count) - 1;

    int requested_edge = 0;
    if (options.padtype != Firfilter2PadType::none) {
        if (options.padlen.has_value()) {
            requested_edge = *options.padlen;
        } else {
            if (coefficient_count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max() / 3))
                throw std::invalid_argument("firfilter2 default padlen overflow");
            requested_edge = 3 * static_cast<int>(coefficient_count);
        }
    }
    if (plan.sample_count <= requested_edge)
        throw std::invalid_argument("firfilter2 input axis must exceed padlen");
    plan.edge = requested_edge > 0 ? requested_edge : 0;
    if (plan.edge > (std::numeric_limits<int>::max() - plan.sample_count) / 2)
        throw std::invalid_argument("firfilter2 extended axis overflow");
    plan.extended_sample_count = plan.sample_count + 2 * plan.edge;
    plan.extended_shape = plan.input_shape;
    plan.extended_shape[axis] = plan.extended_sample_count;

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(input_size /
        static_cast<std::size_t>(plan.sample_count * plan.inner_count));
    const long long extended_size = static_cast<long long>(plan.outer_count)
        * plan.extended_sample_count * plan.inner_count;
    if (extended_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("firfilter2 extended input exceeds int indexing");
    const long long state_size = static_cast<long long>(plan.outer_count)
        * plan.state_count * plan.inner_count;
    if (state_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("firfilter2 state exceeds int indexing");
    return plan;
}
}  // namespace detail

template <typename T>
void detrend_fp32_compute_device(
    const DeviceArray<T>&, int, int, const std::vector<int>&,
    DetrendMode, DeviceArray<float>&);
template <typename Input>
void freq_shift_fp32_compute_device(
    const DeviceArray<Input>&, DeviceArray<ComplexFloat>&, float, float);
template <typename T>
void firfilter2_fp32_compute_device(
    const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<float>&,
    const Firfilter2Options&);
template <typename T>
void firfilter2_fp64_storage_compute_device(
    const DeviceArray<T>&, const DeviceArray<T>&, std::uint64_t*, std::size_t,
    const Firfilter2Options&);
template <typename T>
void hilbert_fp32_compute_device(
    const DeviceArray<T>&, DeviceArray<ComplexFloat>&,
    const HilbertOptions&);
template <typename T>
void hilbert2_fp32_compute_device(
    const DeviceArray<T>&, DeviceArray<ComplexFloat>&,
    const Hilbert2Options&);
template <typename T>
void lfilter_zi_fp32_compute_device(
    const DeviceArray<T>&, const DeviceArray<T>&,
    const detail::LfilterZiLayoutPlan&, DeviceArray<float>&,
    DeviceArray<int>&);
template <typename T>
void wiener_fp32_compute_device(
    const DeviceArray<T>&, const detail::WienerLayoutPlan&,
    bool, float, DeviceArray<float>&);
template <typename T>
void decimate_coefficients_fp32_device(
    const DeviceArray<T>&, DeviceArray<float>&);
template <typename T>
void decimate_fp32_compute_device(
    const DeviceArray<T>&, const DeviceArray<float>&,
    const detail::DecimateLayoutPlan&, int, bool,
    DeviceArray<float>&);
template <typename T>
void resample_fp32_compute_device(
    const DeviceArray<T>&, const detail::ResampleLayoutPlan&,
    ResampleDomain, const DeviceArray<float>*, DeviceArray<float>&);
template <typename T>
void resample_poly_custom_coefficients_fp32_device(
    const DeviceArray<T>&, int, DeviceArray<float>&);
template <typename T>
void resample_poly_fp32_compute_device(
    const DeviceArray<T>&, const DeviceArray<float>&,
    const detail::ResamplePolyLayoutPlan&, DeviceArray<float>&);

template <typename T>
ChannelizePolyCpuResult channelize_poly_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& h, int n_chans)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_channelize_poly(
        x.size(), h.size(), n_chans);
    ChannelizePolyCpuResult result;
    result.shape = {plan.channel_count, plan.point_count};
    result.values.resize(plan.output_size);
    for (int channel = 0; channel < plan.channel_count; ++channel) {
        for (int point = 0; point < plan.point_count; ++point) {
            double real = 0.0;
            double imag = 0.0;
            for (int phase = 0; phase < plan.channel_count; ++phase) {
                double filtered = 0.0;
                for (int tap = 0; tap < plan.tap_count; ++tap) {
                    const int group = point - tap;
                    if (group >= 0) {
                        const std::size_t input_index =
                            static_cast<std::size_t>(group) * plan.channel_count
                            + (plan.channel_count - 1 - phase);
                        const std::size_t filter_index = phase
                            + static_cast<std::size_t>(tap) * plan.channel_count;
                        filtered += detrend_host_load(x[input_index])
                            * detrend_host_load(h[filter_index]);
                    }
                }
                const double angle = 6.283185307179586476925286766559
                    * static_cast<double>(channel) * phase / plan.channel_count;
                real += filtered * std::cos(angle);
                imag += filtered * std::sin(angle);
            }
            result.values[static_cast<std::size_t>(channel) * plan.point_count
                + point] = ComplexFloat{
                    static_cast<float>(real), static_cast<float>(imag)};
        }
    }
    return result;
}

template <typename T>
ChannelizePolyCpuResult channelize_poly_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& h, int n_chans,
    const std::vector<int>& input_shape,
    const std::vector<int>& filter_shape)
{
    const auto plan = detail::prepare_channelize_poly(
        x.size(), h.size(), n_chans, input_shape, filter_shape);
    const auto input_count = static_cast<std::size_t>(plan.input_count);
    const auto filter_count = static_cast<std::size_t>(filter_shape.front());
    return channelize_poly_typed_cpu(
        std::vector<T>(x.begin(), x.begin() + input_count),
        std::vector<T>(h.begin(), h.begin() + filter_count), n_chans);
}
template <typename T>
DetrendCpuResult detrend_typed_cpu(
    std::vector<T>& x, const DetrendOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto layout = prepare_detrend_layout(x.size(), options);
    DetrendCpuResult result;
    result.dtype = detrend_output_dtype<T>(options.mode);
    if (result.dtype == DetrendOutputDtype::fp16) result.fp16.resize(x.size());
    if (result.dtype == DetrendOutputDtype::fp32) result.fp32.resize(x.size());
    if (result.dtype == DetrendOutputDtype::fp64) result.fp64.resize(x.size());
    const auto store = [&](std::size_t index, double value) {
        if (result.dtype == DetrendOutputDtype::fp16)
            result.fp16[index] = detail::SimpleSignalTypePolicy<__half>::store(
                static_cast<float>(value));
        else if (result.dtype == DetrendOutputDtype::fp32)
            result.fp32[index] = static_cast<float>(value);
        else
            result.fp64[index] = value;
    };
    const auto load_value = [&](std::size_t index) {
        return detrend_host_load(x[index]);
    };
    for (int batch = 0; batch < layout.batch_count; ++batch) {
        const int outer = layout.inner_count == 0 ? 0 : batch / layout.inner_count;
        const int inner = layout.inner_count == 0 ? 0 : batch % layout.inner_count;
        const std::size_t base = static_cast<std::size_t>(outer)
            * layout.sample_count * layout.inner_count + inner;
        if (options.mode == DetrendMode::constant) {
            if (layout.sample_count == 0) continue;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
                float sum = 0.0F;
                for (int sample = 0; sample < layout.sample_count; ++sample)
                    sum += static_cast<float>(load_value(
                        base + static_cast<std::size_t>(sample)
                            * layout.inner_count));
                const float mean = sum / layout.sample_count;
                for (int sample = 0; sample < layout.sample_count; ++sample) {
                    const std::size_t index = base + static_cast<std::size_t>(sample)
                        * layout.inner_count;
                    store(index, static_cast<float>(load_value(index)) - mean);
                }
            } else {
                double sum = 0.0;
                for (int sample = 0; sample < layout.sample_count; ++sample)
                    sum += load_value(base + static_cast<std::size_t>(sample)
                        * layout.inner_count);
                const double mean = sum / layout.sample_count;
                for (int sample = 0; sample < layout.sample_count; ++sample) {
                    const std::size_t index = base + static_cast<std::size_t>(sample)
                        * layout.inner_count;
                    store(index, load_value(index) - mean);
                }
            }
            continue;
        }
        int segment_start = 0;
        for (std::size_t segment = 0;
             segment <= layout.breakpoints.size(); ++segment) {
            const int segment_end = segment < layout.breakpoints.size()
                ? layout.breakpoints[segment] : layout.sample_count;
            const int count = segment_end - segment_start;
            double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
            for (int offset = 0; offset < count; ++offset) {
                const double coordinate = static_cast<double>(offset);
                const std::size_t index = base
                    + static_cast<std::size_t>(segment_start + offset)
                        * layout.inner_count;
                const double value = load_value(index);
                sum_x += coordinate;
                sum_y += value;
                sum_xx += coordinate * coordinate;
                sum_xy += coordinate * value;
            }
            const double denominator = count * sum_xx - sum_x * sum_x;
            const double slope = denominator == 0.0
                ? 0.0 : (count * sum_xy - sum_x * sum_y) / denominator;
            const double intercept = denominator == 0.0
                ? sum_y / count : (sum_y - slope * sum_x) / count;
            for (int offset = 0; offset < count; ++offset) {
                const std::size_t index = base
                    + static_cast<std::size_t>(segment_start + offset)
                        * layout.inner_count;
                store(index, load_value(index)
                    - (slope * offset + intercept));
            }
            segment_start = segment_end;
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        if (options.mode == DetrendMode::linear && options.overwrite_data)
            x = result.fp32;
    }
    return result;
}

template <typename T>
void detrend_device(
    DeviceArray<T>& x, DetrendDeviceResult& result,
    const DetrendOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto layout = prepare_detrend_layout(x.size(), options);
    DeviceArray<float> computed(x.size());
    detrend_fp32_compute_device(
        x, layout.sample_count, layout.inner_count, layout.breakpoints,
        options.mode, computed);
    result = DetrendDeviceResult{};
    result.dtype = detrend_output_dtype<T>(options.mode);
    if (result.dtype == DetrendOutputDtype::fp32) {
        if constexpr (std::is_same_v<T, float>) {
            if (options.mode == DetrendMode::linear && options.overwrite_data &&
                !computed.empty())
                cuda_utils::copy_device_to_device(
                    x.data(), computed.data(), computed.size());
        }
        result.fp32 = std::move(computed);
    } else if (result.dtype == DetrendOutputDtype::fp16) {
        result.fp16 = type_dispatch::from_fp32_fp16(computed);
    } else {
        result.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    }
}
template <typename T>
FirfilterCpuResult firfilter_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& x,
    const FirfilterOptions& options, const std::vector<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_firfilter_layout(
        x.size(), b.size(), options, zi != nullptr, zi == nullptr ? 0 : zi->size());
    FirfilterCpuResult result;
    result.has_zf = zi != nullptr;
    result.y_shape = plan.input_shape;
    result.y.resize(x.size());
    if (result.has_zf) {
        result.zf_shape = plan.state_shape;
        result.zf.resize(static_cast<std::size_t>(plan.outer_count)
            * plan.state_count * plan.inner_count);
    }
    const int line_count = plan.outer_count * plan.inner_count;
    const auto full_value = [&](int line, int full_index) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t input_base = static_cast<std::size_t>(outer)
            * plan.sample_count * plan.inner_count + inner;
        double value = 0.0;
        for (int tap = 0; tap <= plan.state_count; ++tap) {
            const int sample = full_index - tap;
            if (sample >= 0 && sample < plan.sample_count)
                value += detrend_host_load(b[tap]) * detrend_host_load(
                    x[input_base + static_cast<std::size_t>(sample)
                        * plan.inner_count]);
        }
        if (zi != nullptr && full_index < plan.state_count) {
            const int zi_index = plan.zi_line_offsets[line]
                + full_index * plan.zi_state_stride;
            value += detrend_host_load((*zi)[zi_index]);
        }
        return static_cast<float>(value);
    };
    for (int line = 0; line < line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t output_base = static_cast<std::size_t>(outer)
            * plan.sample_count * plan.inner_count + inner;
        for (int sample = 0; sample < plan.sample_count; ++sample)
            result.y[output_base + static_cast<std::size_t>(sample)
                * plan.inner_count] = full_value(line, sample);
        if (result.has_zf) {
            const std::size_t state_base = static_cast<std::size_t>(outer)
                * plan.state_count * plan.inner_count + inner;
            for (int state = 0; state < plan.state_count; ++state)
                result.zf[state_base + static_cast<std::size_t>(state)
                    * plan.inner_count] = full_value(
                        line, plan.sample_count + state);
        }
    }
    return result;
}
template <typename T>
std::vector<double> lfilter_zi_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& a);
namespace {
template <typename T>
T firfilter2_host_odd_reflect(T endpoint, T reflected)
{
    if constexpr (std::is_integral_v<T>) {
        constexpr int bits = static_cast<int>(sizeof(T) * 8);
        constexpr long long modulus = 1LL << bits;
        constexpr long long sign = modulus / 2;
        long long value = 2LL * static_cast<long long>(endpoint)
            - static_cast<long long>(reflected);
        value %= modulus;
        if (value < 0) value += modulus;
        if (value >= sign) value -= modulus;
        return static_cast<T>(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        const T doubled = detail::SimpleSignalTypePolicy<T>::store(
            2.0F * detail::SimpleSignalTypePolicy<T>::load(endpoint));
        return detail::SimpleSignalTypePolicy<T>::store(
            detail::SimpleSignalTypePolicy<T>::load(doubled)
            - detail::SimpleSignalTypePolicy<T>::load(reflected));
    } else {
        const T doubled = static_cast<T>(2.0F * endpoint);
        return static_cast<T>(doubled - reflected);
    }
}

template <typename T>
Firfilter2OutputDtype firfilter2_output_dtype()
{
    return std::is_same_v<T, std::int32_t>
        ? Firfilter2OutputDtype::fp64 : Firfilter2OutputDtype::fp32;
}
}

template <typename T>
Firfilter2CpuResult firfilter2_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& x,
    const Firfilter2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_firfilter2_layout(
        x.size(), b.size(), options);
    Firfilter2CpuResult result;
    result.dtype = firfilter2_output_dtype<T>();
    result.shape = plan.input_shape;
    if (result.dtype == Firfilter2OutputDtype::fp32)
        result.fp32.resize(x.size());
    else
        result.fp64.resize(x.size());

    std::vector<double> coefficients(b.size());
    for (std::size_t index = 0; index < b.size(); ++index)
        coefficients[index] = detrend_host_load(b[index]);
    std::vector<double> base_state(plan.state_count);
    double suffix = 0.0;
    for (int state = plan.state_count - 1; state >= 0; --state) {
        suffix += coefficients[state + 1];
        base_state[state] = suffix;
    }
    const auto filter_line = [&](const std::vector<double>& input, double scale) {
        std::vector<double> output(input.size());
        for (int sample = 0; sample < static_cast<int>(input.size()); ++sample) {
            double value = sample < plan.state_count
                ? base_state[sample] * scale : 0.0;
            for (int tap = 0; tap < static_cast<int>(coefficients.size()); ++tap) {
                const int input_index = sample - tap;
                if (input_index >= 0)
                    value += coefficients[tap] * input[input_index];
            }
            output[sample] = value;
        }
        return output;
    };

    const int line_count = plan.outer_count * plan.inner_count;
    for (int line = 0; line < line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t input_base = static_cast<std::size_t>(outer)
            * plan.sample_count * plan.inner_count + inner;
        std::vector<T> input_line(plan.sample_count);
        for (int sample = 0; sample < plan.sample_count; ++sample)
            input_line[sample] = x[input_base
                + static_cast<std::size_t>(sample) * plan.inner_count];
        std::vector<T> padded_line(plan.extended_sample_count);
        for (int sample = 0; sample < plan.extended_sample_count; ++sample) {
            if (sample < plan.edge) {
                const T reflected = input_line[plan.edge - sample];
                if (options.padtype == Firfilter2PadType::odd)
                    padded_line[sample] = firfilter2_host_odd_reflect(
                        input_line.front(), reflected);
                else if (options.padtype == Firfilter2PadType::even)
                    padded_line[sample] = reflected;
                else
                    padded_line[sample] = input_line.front();
            } else if (sample < plan.edge + plan.sample_count) {
                padded_line[sample] = input_line[sample - plan.edge];
            } else {
                const int right = sample - plan.edge - plan.sample_count;
                const T reflected = input_line[plan.sample_count - 2 - right];
                if (options.padtype == Firfilter2PadType::odd)
                    padded_line[sample] = firfilter2_host_odd_reflect(
                        input_line.back(), reflected);
                else if (options.padtype == Firfilter2PadType::even)
                    padded_line[sample] = reflected;
                else
                    padded_line[sample] = input_line.back();
            }
        }
        std::vector<double> extended(plan.extended_sample_count);
        for (int sample = 0; sample < plan.extended_sample_count; ++sample)
            extended[sample] = detrend_host_load(padded_line[sample]);
        auto forward = filter_line(extended, extended.front());
        const double reverse_scale = forward.back();
        std::reverse(forward.begin(), forward.end());
        auto backward = filter_line(forward, reverse_scale);
        std::reverse(backward.begin(), backward.end());
        for (int sample = 0; sample < plan.sample_count; ++sample) {
            const std::size_t output_index = input_base
                + static_cast<std::size_t>(sample) * plan.inner_count;
            const double value = backward[plan.edge + sample];
            if (result.dtype == Firfilter2OutputDtype::fp32)
                result.fp32[output_index] = static_cast<float>(value);
            else
                result.fp64[output_index] = value;
        }
    }
    return result;
}

template <typename T>
void firfilter2_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    Firfilter2DeviceResult& output, const Firfilter2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_firfilter2_layout(
        x.size(), b.size(), options);
    output.shape = plan.input_shape;
    output.dtype = firfilter2_output_dtype<T>();
    if (output.dtype == Firfilter2OutputDtype::fp32) {
        if (!output.fp64.empty()) output.fp64.reset();
        if (output.fp32.size() != x.size()) output.fp32.reset(x.size());
        firfilter2_fp32_compute_device(b, x, output.fp32, options);
    } else {
        if (!output.fp32.empty()) output.fp32.reset();
        if (output.fp64.size() != x.size()) output.fp64.reset(x.size());
        if (b.size() <= 16) {
            firfilter2_fp64_storage_compute_device(
                b, x, reinterpret_cast<std::uint64_t*>(output.fp64.data()),
                output.fp64.size(), options);
        } else {
            DeviceArray<float> computed(x.size());
            firfilter2_fp32_compute_device(b, x, computed, options);
            output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
        }
    }
}

template <typename Input>
std::vector<ComplexDouble> freq_shift_typed_cpu(
    const std::vector<Input>& x, double freq, double fs)
{
    static_assert(detail::is_freq_shift_input_v<Input>);
    constexpr double two_pi = 6.283185307179586476925286766559;
    std::vector<ComplexDouble> output(x.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        double real = 0.0, imag = 0.0;
        freq_shift_host_load(x[index], real, imag);
        const double phase = -two_pi * freq * static_cast<double>(index) / fs;
        const double cosine = std::cos(phase);
        const double sine = std::sin(phase);
        output[index] = ComplexDouble{
            real * cosine - imag * sine,
            real * sine + imag * cosine};
    }
    return output;
}

template <typename Input>
void freq_shift_device(
    const DeviceArray<Input>& x, DeviceArray<ComplexDouble>& output,
    double freq, double fs)
{
    static_assert(detail::is_freq_shift_input_v<Input>);
    if (output.size() != x.size())
        throw std::invalid_argument("freq_shift output shape");
    DeviceArray<ComplexFloat> computed(x.size());
    if (!x.empty())
        freq_shift_fp32_compute_device(
            x, computed, static_cast<float>(freq), static_cast<float>(fs));
    output = cuda_utils::finalize_complex_fp64_device_storage(computed);
}
namespace {
template <typename T>
HilbertOutputDtype hilbert_output_dtype()
{
    return std::is_integral_v<T>
        ? HilbertOutputDtype::complex_fp64
        : HilbertOutputDtype::complex_fp32;
}

double hilbert_mask_weight(int index, int length)
{
    if (index == 0 || ((length % 2) == 0 && index == length / 2))
        return 1.0;
    return index < (length + 1) / 2 ? 2.0 : 0.0;
}

template <typename T>
std::size_t lfilter_zi_denominator_offset(const std::vector<T>& denominator)
{
    if (denominator.empty())
        throw std::invalid_argument("lfilter_zi denominator");
    std::size_t offset = 0;
    while (denominator.size() - offset > 1 &&
           detrend_host_load(denominator[offset]) == 0.0) {
        ++offset;
    }
    return offset;
}
}

template <typename T>
HilbertCpuResult hilbert_typed_cpu(
    const std::vector<T>& x, const HilbertOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert_layout(x.size(), options);
    HilbertCpuResult result;
    result.dtype = hilbert_output_dtype<T>();
    result.shape = plan.output_shape;
    if (result.dtype == HilbertOutputDtype::complex_fp32)
        result.complex_fp32.resize(plan.output_size);
    else
        result.complex_fp64.resize(plan.output_size);

    struct HostComplex { double real; double imag; };
    std::vector<HostComplex> spectrum(plan.fft_length);
    for (int line = 0; line < plan.line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t input_base = static_cast<std::size_t>(outer)
            * plan.input_axis_length * plan.inner_count + inner;
        for (int frequency = 0; frequency < plan.fft_length; ++frequency) {
            double real = 0.0;
            double imag = 0.0;
            for (int sample = 0; sample < plan.fft_length; ++sample) {
                const double value = sample < plan.input_axis_length
                    ? detrend_host_load(x[input_base
                        + static_cast<std::size_t>(sample) * plan.inner_count])
                    : 0.0;
                const double angle = -6.283185307179586476925286766559
                    * frequency * sample / plan.fft_length;
                real += value * std::cos(angle);
                imag += value * std::sin(angle);
            }
            const double mask = (frequency == 0 ||
                ((plan.fft_length % 2) == 0 &&
                 frequency == plan.fft_length / 2))
                ? 1.0
                : (frequency < (plan.fft_length + 1) / 2 ? 2.0 : 0.0);
            spectrum[frequency] = HostComplex{real * mask, imag * mask};
        }
        const std::size_t output_base = static_cast<std::size_t>(outer)
            * plan.fft_length * plan.inner_count + inner;
        for (int sample = 0; sample < plan.fft_length; ++sample) {
            double real = 0.0;
            double imag = 0.0;
            for (int frequency = 0; frequency < plan.fft_length; ++frequency) {
                const double angle = 6.283185307179586476925286766559
                    * frequency * sample / plan.fft_length;
                real += spectrum[frequency].real * std::cos(angle)
                    - spectrum[frequency].imag * std::sin(angle);
                imag += spectrum[frequency].real * std::sin(angle)
                    + spectrum[frequency].imag * std::cos(angle);
            }
            real /= plan.fft_length;
            imag /= plan.fft_length;
            const std::size_t output_index = output_base
                + static_cast<std::size_t>(sample) * plan.inner_count;
            if (result.dtype == HilbertOutputDtype::complex_fp32)
                result.complex_fp32[output_index] = ComplexFloat{
                    static_cast<float>(real), static_cast<float>(imag)};
            else
                result.complex_fp64[output_index] = ComplexDouble{real, imag};
        }
    }
    return result;
}

template <typename T>
void hilbert_device(
    const DeviceArray<T>& x, HilbertDeviceResult& output,
    const HilbertOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert_layout(x.size(), options);
    DeviceArray<ComplexFloat> computed(plan.output_size);
    hilbert_fp32_compute_device(x, computed, options);
    output = HilbertDeviceResult{};
    output.dtype = hilbert_output_dtype<T>();
    output.shape = plan.output_shape;
    if (output.dtype == HilbertOutputDtype::complex_fp32)
        output.complex_fp32 = std::move(computed);
    else
        output.complex_fp64 =
            cuda_utils::finalize_complex_fp64_device_storage(computed);
}
template <typename T>
Hilbert2CpuResult hilbert2_typed_cpu(
    const std::vector<T>& x, const Hilbert2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert2_layout(x.size(), options);
    std::vector<std::vector<complexd>> time(
        plan.fft_rows, std::vector<complexd>(plan.fft_cols));
    for (int row = 0; row < plan.fft_rows; ++row) {
        for (int col = 0; col < plan.fft_cols; ++col) {
            if (row < plan.input_rows && col < plan.input_cols) {
                time[row][col] = complexd{
                    detrend_host_load(x[static_cast<std::size_t>(row)
                        * plan.input_cols + col]),
                    0.0};
            }
        }
    }
    FFTInterface2D_cpu forward_fft(plan.fft_rows, plan.fft_cols);
    const auto first_spectrum = forward_fft.fft2d(time);
    std::vector<std::vector<complexd>> masked(
        plan.output_rows, std::vector<complexd>(plan.output_cols));
    for (int u = 0; u < plan.output_rows; ++u) {
        for (int v = 0; v < plan.output_cols; ++v) {
            const int source_u = plan.fft_rows == 1 ? 0 : u;
            const auto value = first_spectrum[source_u][v];
            const double scale = hilbert_mask_weight(u, plan.fft_cols)
                * hilbert_mask_weight(v, plan.fft_cols);
            masked[u][v] = value * scale;
        }
    }

    FFTInterface2D_cpu inverse_fft(plan.output_rows, plan.output_cols);
    const auto computed = inverse_fft.ifft2d(masked);

    Hilbert2CpuResult result;
    result.dtype = hilbert_output_dtype<T>();
    result.shape = {plan.output_rows, plan.output_cols};
    if (result.dtype == HilbertOutputDtype::complex_fp32)
        result.complex_fp32.resize(plan.output_size);
    else
        result.complex_fp64.resize(plan.output_size);
    for (int row = 0; row < plan.output_rows; ++row) {
        for (int col = 0; col < plan.output_cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row)
                * plan.output_cols + col;
            const double real = computed[row][col].real();
            const double imag = computed[row][col].imag();
            if (result.dtype == HilbertOutputDtype::complex_fp32)
                result.complex_fp32[index] = ComplexFloat{
                    static_cast<float>(real), static_cast<float>(imag)};
            else
                result.complex_fp64[index] = ComplexDouble{real, imag};
        }
    }
    return result;
}

template <typename T>
void hilbert2_device(
    const DeviceArray<T>& x, Hilbert2DeviceResult& output,
    const Hilbert2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert2_layout(x.size(), options);
    DeviceArray<ComplexFloat> computed(plan.output_size);
    hilbert2_fp32_compute_device(x, computed, options);
    output = Hilbert2DeviceResult{};
    output.dtype = hilbert_output_dtype<T>();
    output.shape = {plan.output_rows, plan.output_cols};
    if (output.dtype == HilbertOutputDtype::complex_fp32)
        output.complex_fp32 = std::move(computed);
    else
        output.complex_fp64 =
            cuda_utils::finalize_complex_fp64_device_storage(computed);
}
template <typename T>
std::vector<double> lfilter_zi_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& a)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const std::size_t denominator_offset =
        lfilter_zi_denominator_offset(a);
    const auto plan = detail::prepare_lfilter_zi_layout(
        b.size(), a.size(), denominator_offset);
    std::vector<double> numerator(plan.common_count, 0.0);
    std::vector<double> denominator(plan.common_count, 0.0);
    const double a0 = detrend_host_load(a[denominator_offset]);
    for (std::size_t index = 0; index < b.size(); ++index)
        numerator[index] = detrend_host_load(b[index]) / a0;
    for (int index = 0; index < plan.denominator_count; ++index)
        denominator[index] = detrend_host_load(
            a[denominator_offset + static_cast<std::size_t>(index)]) / a0;

    std::vector<double> augmented(plan.augmented_count, 0.0);
    for (int row = 0; row < plan.output_count; ++row) {
        for (int column = 0; column < plan.output_count; ++column) {
            double companion = 0.0;
            if (column == 0)
                companion = -denominator[row + 1];
            else if (column == row + 1)
                companion = 1.0;
            augmented[static_cast<std::size_t>(row) * plan.common_count + column] =
                (row == column ? 1.0 : 0.0) - companion;
        }
        augmented[static_cast<std::size_t>(row) * plan.common_count +
                  plan.output_count] =
            numerator[row + 1] - denominator[row + 1] * numerator[0];
    }
    for (int pivot = 0; pivot < plan.output_count; ++pivot) {
        int pivot_row = pivot;
        for (int row = pivot + 1; row < plan.output_count; ++row) {
            if (std::fabs(augmented[static_cast<std::size_t>(row) *
                                  plan.common_count + pivot]) >
                std::fabs(augmented[static_cast<std::size_t>(pivot_row) *
                                  plan.common_count + pivot])) {
                pivot_row = row;
            }
        }
        if (pivot_row != pivot) {
            for (int column = pivot; column < plan.common_count; ++column) {
                std::swap(
                    augmented[static_cast<std::size_t>(pivot) *
                              plan.common_count + column],
                    augmented[static_cast<std::size_t>(pivot_row) *
                              plan.common_count + column]);
            }
        }
        const double divisor = augmented[static_cast<std::size_t>(pivot) *
                                         plan.common_count + pivot];
        if (std::isfinite(divisor) && std::fabs(divisor) < 1.0e-14)
            throw std::runtime_error("singular lfilter_zi system");
        for (int row = pivot + 1; row < plan.output_count; ++row) {
            const double factor = augmented[static_cast<std::size_t>(row) *
                                            plan.common_count + pivot] / divisor;
            for (int column = pivot; column < plan.common_count; ++column) {
                augmented[static_cast<std::size_t>(row) * plan.common_count +
                          column] -= factor * augmented[
                    static_cast<std::size_t>(pivot) * plan.common_count + column];
            }
        }
    }
    std::vector<double> output(plan.output_count, 0.0);
    for (int row = plan.output_count - 1; row >= 0; --row) {
        double value = augmented[static_cast<std::size_t>(row) *
                                 plan.common_count + plan.output_count];
        for (int column = row + 1; column < plan.output_count; ++column) {
            value -= augmented[static_cast<std::size_t>(row) *
                               plan.common_count + column] * output[column];
        }
        output[row] = value / augmented[static_cast<std::size_t>(row) *
                                       plan.common_count + row];
    }
    return output;
}

template <typename T>
void lfilter_zi_device(
    const DeviceArray<T>& b, const DeviceArray<T>& a,
    LfilterZiDeviceResult& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto host_denominator = a.to_host();
    const std::size_t denominator_offset =
        lfilter_zi_denominator_offset(host_denominator);
    const auto plan = detail::prepare_lfilter_zi_layout(
        b.size(), a.size(), denominator_offset);
    DeviceArray<float> computed(plan.output_count);
    DeviceArray<int> status(1);
    lfilter_zi_fp32_compute_device(b, a, plan, computed, status);
    const auto host_status = status.to_host();
    if (host_status[0] != 0)
        throw std::runtime_error("singular lfilter_zi system");
    output = LfilterZiDeviceResult{};
    output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
}
template <typename T>
SosfiltCpuResult sosfilt_typed_cpu(
    const std::vector<T>& sos, int sections, const std::vector<T>& x,
    const SosfiltOptions& options, const std::vector<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const bool has_zi = zi != nullptr;
    const auto plan = detail::prepare_sosfilt_layout(
        x.size(), sos.size(), sections, options, has_zi,
        has_zi ? zi->size() : 0U);

    std::vector<float> coefficients(sos.size());
    for (std::size_t index = 0; index < sos.size(); ++index)
        coefficients[index] = ld(sos[index]);
    detail::validate_sos_coefficients(coefficients, sections);

    std::vector<double> state(plan.state_size, 0.0);
    if (has_zi) {
        for (std::size_t index = 0; index < plan.state_size; ++index)
            state[index] = static_cast<double>(ld((*zi)[index]));
    }

    SosfiltCpuResult output;
    output.y.resize(x.size());
    output.y_shape = plan.input_shape;
    output.has_zf = has_zi;
    if (has_zi) output.zf_shape = plan.state_shape;

    for (int line = 0; line < plan.line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        for (int sample = 0; sample < plan.sample_count; ++sample) {
            const std::size_t signal_index = static_cast<std::size_t>(outer) *
                static_cast<std::size_t>(plan.sample_count) * plan.inner_count +
                static_cast<std::size_t>(sample) * plan.inner_count + inner;
            double value = static_cast<double>(ld(x[signal_index]));
            for (int section = 0; section < sections; ++section) {
                const std::size_t coefficient =
                    static_cast<std::size_t>(section) * 6U;
                const std::size_t state0 =
                    ((static_cast<std::size_t>(section) * plan.outer_count + outer)
                        * 2U) * plan.inner_count + inner;
                const std::size_t state1 = state0 + plan.inner_count;
                const double filtered = coefficients[coefficient] * value + state[state0];
                const double next0 = coefficients[coefficient + 1U] * value -
                    coefficients[coefficient + 4U] * filtered + state[state1];
                const double next1 = coefficients[coefficient + 2U] * value -
                    coefficients[coefficient + 5U] * filtered;
                state[state0] = next0;
                state[state1] = next1;
                value = filtered;
            }
            output.y[signal_index] = static_cast<float>(value);
        }
    }
    if (has_zi) {
        output.zf.resize(plan.state_size);
        for (std::size_t index = 0; index < plan.state_size; ++index)
            output.zf[index] = static_cast<float>(state[index]);
    }
    return output;
}
template <typename T>
WienerCpuResult wiener_typed_cpu(
    const std::vector<T>& x, const WienerOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_wiener_layout(x.size(), options);
    std::vector<double> local_mean(x.size());
    std::vector<double> local_variance(x.size());
    std::vector<int> output_coordinates(plan.rank);

    for (int index = 0; index < plan.element_count; ++index) {
        for (int dimension = 0; dimension < plan.rank; ++dimension) {
            output_coordinates[dimension] =
                (index / plan.input_strides[dimension]) %
                plan.input_shape[dimension];
        }
        double sum = 0.0;
        double square_sum = 0.0;
        for (int window_index = 0;
             window_index < plan.window_volume; ++window_index) {
            int remaining = window_index;
            int source_index = 0;
            bool inside = true;
            for (int dimension = plan.rank - 1; dimension >= 0; --dimension) {
                const int window_coordinate =
                    remaining % plan.window_shape[dimension];
                remaining /= plan.window_shape[dimension];
                const int source_coordinate = output_coordinates[dimension] +
                    window_coordinate - plan.window_shape[dimension] / 2;
                if (source_coordinate < 0 ||
                    source_coordinate >= plan.input_shape[dimension]) {
                    inside = false;
                    break;
                }
                source_index += source_coordinate * plan.input_strides[dimension];
            }
            if (!inside) continue;
            sum += static_cast<double>(ld(x[source_index]));
            square_sum += wiener_host_square(x[source_index]);
        }
        const double mean = sum / static_cast<double>(plan.window_volume);
        local_mean[index] = mean;
        local_variance[index] = square_sum /
            static_cast<double>(plan.window_volume) - mean * mean;
    }

    double noise = 0.0;
    if (options.noise.has_value()) {
        noise = *options.noise;
    } else if (plan.element_count == 0) {
        noise = std::numeric_limits<double>::quiet_NaN();
    } else {
        for (const double variance : local_variance) noise += variance;
        noise /= static_cast<double>(plan.element_count);
    }

    WienerCpuResult output;
    output.fp64.resize(x.size());
    output.shape = plan.input_shape;
    for (int index = 0; index < plan.element_count; ++index) {
        const double input = static_cast<double>(ld(x[index]));
        const double variance = local_variance[index];
        const double filtered = (input - local_mean[index]) *
            (1.0 - noise / variance) + local_mean[index];
        output.fp64[index] = variance < noise
            ? local_mean[index] : filtered;
    }
    return output;
}
template <typename T>
void wiener_device(
    const DeviceArray<T>& x, WienerDeviceResult& output,
    const WienerOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_wiener_layout(x.size(), options);
    DeviceArray<float> computed(x.size());
    wiener_fp32_compute_device(
        x, plan, options.noise.has_value(),
        options.noise.has_value() ? static_cast<float>(*options.noise) : 0.0F,
        computed);
    output = WienerDeviceResult{};
    output.shape = plan.input_shape;
    output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
}
template <typename T>
DecimateCpuResult decimate_typed_cpu(
    const std::vector<T>& x, int q, const DecimateOptions& options,
    const std::vector<T>* coefficients)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_decimate_layout(x.size(), q, options);
    const bool custom = coefficients != nullptr;
    validate_decimate_filter_selection(options, custom);
    if (custom && coefficients->empty())
        throw std::invalid_argument("decimate coefficients");

    std::vector<double> taps;
    if (custom) {
        if (coefficients->size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("decimate coefficient count");
        taps.resize(coefficients->size());
        for (std::size_t index = 0; index < coefficients->size(); ++index)
            taps[index] = detrend_host_load((*coefficients)[index]);
    } else {
        taps = design_decimate_filter(q, options);
    }

    const int tap_count = static_cast<int>(taps.size());
    const int half_length = (tap_count - 1) / 2;
    const int prefix = options.zero_phase
        ? q - half_length % q
        : 0;
    if (tap_count > std::numeric_limits<int>::max() - prefix)
        throw std::invalid_argument("decimate padded coefficient count");
    const int output_start = options.zero_phase
        ? (half_length + prefix) / q
        : 0;

    DecimateCpuResult output;
    output.dtype = decimate_output_dtype<T>(custom);
    output.shape = plan.output_shape;
    if (output.dtype == DecimateOutputDtype::fp32)
        output.fp32.resize(plan.output_size);
    else
        output.fp64.resize(plan.output_size);

    for (int line = 0; line < plan.line_count; ++line) {
        const int outer = plan.inner_count == 0
            ? 0 : line / plan.inner_count;
        const int inner = plan.inner_count == 0
            ? 0 : line % plan.inner_count;
        const std::size_t input_base =
            (static_cast<std::size_t>(outer) * plan.sample_count *
             plan.inner_count) + inner;
        const std::size_t output_base =
            (static_cast<std::size_t>(outer) * plan.output_sample_count *
             plan.inner_count) + inner;
        for (int output_index = 0;
             output_index < plan.output_sample_count; ++output_index) {
            const long long filtered_index =
                (static_cast<long long>(output_index) + output_start) * q;
            const std::size_t destination = output_base +
                static_cast<std::size_t>(output_index) * plan.inner_count;
            if (output.dtype == DecimateOutputDtype::fp32) {
                float sum = 0.0F;
                for (int padded_tap = prefix;
                     padded_tap < prefix + tap_count; ++padded_tap) {
                    const long long source = filtered_index - padded_tap;
                    if (source < 0 || source >= plan.sample_count) continue;
                    const std::size_t source_index = input_base +
                        static_cast<std::size_t>(source) * plan.inner_count;
                    sum += static_cast<float>(taps[padded_tap - prefix]) *
                        static_cast<float>(detrend_host_load(x[source_index]));
                }
                output.fp32[destination] = sum;
            } else {
                double sum = 0.0;
                for (int padded_tap = prefix;
                     padded_tap < prefix + tap_count; ++padded_tap) {
                    const long long source = filtered_index - padded_tap;
                    if (source < 0 || source >= plan.sample_count) continue;
                    const std::size_t source_index = input_base +
                        static_cast<std::size_t>(source) * plan.inner_count;
                    sum += taps[padded_tap - prefix] *
                        detrend_host_load(x[source_index]);
                }
                output.fp64[destination] = sum;
            }
        }
    }
    return output;
}

template <typename T>
void decimate_device(
    const DeviceArray<T>& x, int q, DecimateDeviceResult& output,
    const DecimateOptions& options,
    const DeviceArray<T>* coefficients)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_decimate_layout(x.size(), q, options);
    const bool custom = coefficients != nullptr;
    validate_decimate_filter_selection(options, custom);
    if (custom && coefficients->empty())
        throw std::invalid_argument("decimate coefficients");

    DeviceArray<float> taps;
    if (custom) {
        if (coefficients->size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("decimate coefficient count");
        taps = DeviceArray<float>(coefficients->size());
        decimate_coefficients_fp32_device(*coefficients, taps);
    } else {
        const auto designed = design_decimate_filter(q, options);
        std::vector<float> fp32_taps(designed.size());
        for (std::size_t index = 0; index < designed.size(); ++index)
            fp32_taps[index] = static_cast<float>(designed[index]);
        taps = DeviceArray<float>::from_host(fp32_taps);
    }

    DeviceArray<float> computed(plan.output_size);
    decimate_fp32_compute_device(
        x, taps, plan, q, options.zero_phase, computed);
    output = DecimateDeviceResult{};
    output.dtype = decimate_output_dtype<T>(custom);
    output.shape = plan.output_shape;
    if (output.dtype == DecimateOutputDtype::fp32) {
        cuda_utils::synchronize_stream();
        output.fp32 = std::move(computed);
    } else {
        output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    }
}
template <typename T>
ResampleCpuResult resample_typed_cpu(
    const std::vector<T>& x, int output_sample_count,
    const ResampleOptions& options,
    const std::vector<T>* sample_positions)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_resample_layout(
        x.size(), output_sample_count, options);
    validate_resample_frequency_integer_split<T>(plan, options);
    if (sample_positions != nullptr && sample_positions->size() < 2)
        throw std::invalid_argument("resample sample positions");

    ResampleCpuResult output;
    output.dtype = resample_output_dtype<T>();
    output.shape = plan.output_shape;
    output.has_time = sample_positions != nullptr;
    if (output.dtype == ResampleOutputDtype::fp32)
        output.fp32.resize(plan.output_size);
    else
        output.fp64.resize(plan.output_size);

    const int nx = plan.input_sample_count;
    const int ny = plan.output_sample_count;
    const int copied = std::min(nx, ny);
    const int nyquist_end = copied / 2 + 1;
    std::vector<std::complex<double>> spectrum(nx);
    std::vector<std::complex<double>> resized(ny);
    for (int line = 0; line < plan.line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t input_base =
            (static_cast<std::size_t>(outer) * nx * plan.inner_count) + inner;
        if (options.domain == ResampleDomain::time) {
            for (int frequency = 0; frequency < nx; ++frequency) {
                std::complex<double> sum{0.0, 0.0};
                for (int sample = 0; sample < nx; ++sample) {
                    const double angle = -6.283185307179586476925286766559 *
                        frequency * sample / static_cast<double>(nx);
                    const double value = detrend_host_load(
                        x[input_base + static_cast<std::size_t>(sample) *
                          plan.inner_count]);
                    sum += value * std::complex<double>(
                        std::cos(angle), std::sin(angle));
                }
                if (!options.frequency_window.empty()) {
                    const double window = output.dtype == ResampleOutputDtype::fp32
                        ? static_cast<double>(static_cast<float>(
                            options.frequency_window[frequency]))
                        : options.frequency_window[frequency];
                    sum *= window;
                }
                spectrum[frequency] = sum;
            }
        } else {
            for (int frequency = 0; frequency < nx; ++frequency) {
                T staged = x[input_base +
                    static_cast<std::size_t>(frequency) * plan.inner_count];
                if (!options.frequency_window.empty())
                    staged = resample_frequency_multiply(
                        staged, options.frequency_window[frequency]);
                spectrum[frequency] = std::complex<double>(
                    detrend_host_load(staged), 0.0);
            }
        }

        std::fill(resized.begin(), resized.end(), std::complex<double>{0.0, 0.0});
        for (int destination = 0; destination < ny; ++destination) {
            int source = -1;
            if (destination < nyquist_end) {
                source = destination;
            } else if (copied > 2 &&
                       destination >= ny + nyquist_end - copied) {
                source = nx + destination - ny;
            }
            if (source >= 0) resized[destination] = spectrum[source];
        }
        if (copied % 2 == 0) {
            if (ny < nx) {
                const int destination = ny - copied / 2;
                const auto negative = spectrum[nx - copied / 2];
                if (options.domain == ResampleDomain::frequency) {
                    resized[destination] = std::complex<double>(
                        resample_frequency_add<T>(
                            resized[destination].real(), negative.real()),
                        0.0);
                } else {
                    resized[destination] += negative;
                }
            } else if (nx < ny) {
                if (options.domain == ResampleDomain::frequency) {
                    resized[copied / 2] = std::complex<double>(
                        resample_frequency_half<T>(
                            resized[copied / 2].real()), 0.0);
                } else {
                    resized[copied / 2] *= 0.5;
                }
                resized[ny - copied / 2] = resized[copied / 2];
            }
        }

        const std::size_t output_base =
            (static_cast<std::size_t>(outer) * ny * plan.inner_count) + inner;
        for (int sample = 0; sample < ny; ++sample) {
            std::complex<double> sum{0.0, 0.0};
            for (int frequency = 0; frequency < ny; ++frequency) {
                const double angle = 6.283185307179586476925286766559 *
                    frequency * sample / static_cast<double>(ny);
                sum += resized[frequency] * std::complex<double>(
                    std::cos(angle), std::sin(angle));
            }
            const double value = sum.real() / static_cast<double>(nx);
            const std::size_t destination = output_base +
                static_cast<std::size_t>(sample) * plan.inner_count;
            if (output.dtype == ResampleOutputDtype::fp32)
                output.fp32[destination] = static_cast<float>(value);
            else
                output.fp64[destination] = value;
        }
    }

    if (sample_positions != nullptr) {
        output.time.resize(static_cast<std::size_t>(ny));
        const double first = detrend_host_load((*sample_positions)[0]);
        const double spacing = resample_time_delta(
            (*sample_positions)[0], (*sample_positions)[1]);
        for (int index = 0; index < ny; ++index) {
            output.time[index] = static_cast<double>(index) * spacing * nx /
                static_cast<double>(ny) + first;
        }
    }
    return output;
}

template <typename T>
void resample_device(
    const DeviceArray<T>& x, int output_sample_count,
    ResampleDeviceResult& output, const ResampleOptions& options,
    const DeviceArray<T>* sample_positions)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_resample_layout(
        x.size(), output_sample_count, options);
    validate_resample_frequency_integer_split<T>(plan, options);
    std::vector<double> host_time;
    if (sample_positions != nullptr) {
        if (sample_positions->size() < 2)
            throw std::invalid_argument("resample sample positions");
        std::array<T, 2> host_positions{};
        cuda_utils::copy_to_host(
            host_positions.data(), sample_positions->data(), host_positions.size());
        cuda_utils::synchronize_stream();
        const double first = detrend_host_load(host_positions[0]);
        const double spacing = resample_time_delta(
            host_positions[0], host_positions[1]);
        host_time.resize(static_cast<std::size_t>(plan.output_sample_count));
        for (int index = 0; index < plan.output_sample_count; ++index) {
            host_time[index] = static_cast<double>(index) * spacing *
                plan.input_sample_count / plan.output_sample_count + first;
        }
    }

    DeviceArray<float> device_window;
    const DeviceArray<float>* window_pointer = nullptr;
    if (!options.frequency_window.empty()) {
        std::vector<float> fp32_window(options.frequency_window.size());
        for (std::size_t index = 0; index < fp32_window.size(); ++index)
            fp32_window[index] = static_cast<float>(
                options.frequency_window[index]);
        device_window = DeviceArray<float>::from_host(fp32_window);
        window_pointer = &device_window;
    }
    DeviceArray<float> computed(plan.output_size);
    resample_fp32_compute_device(
        x, plan, options.domain, window_pointer, computed);

    output = ResampleDeviceResult{};
    output.dtype = resample_output_dtype<T>();
    output.shape = plan.output_shape;
    output.has_time = sample_positions != nullptr;
    if (output.dtype == ResampleOutputDtype::fp32) {
        cuda_utils::synchronize_stream();
        output.fp32 = std::move(computed);
    } else {
        output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    }
    if (output.has_time)
        output.time = DeviceArray<double>::from_host(host_time);
}
template <typename T>
UpfirdnCpuResult upfirdn_typed_cpu(
    const std::vector<T>& h, const std::vector<T>& x,
    int up, int down, const UpfirdnOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_upfirdn_layout(
        x.size(), h.size(), up, down, options);
    UpfirdnCpuResult output;
    output.dtype = upfirdn_output_dtype<T>();
    output.shape = plan.output_shape;
    if constexpr (std::is_same_v<T, std::int32_t>) {
        output.fp64.resize(plan.output_size);
        for (int line = 0; line < plan.line_count; ++line) {
            const int outer = line / plan.inner_count;
            const int inner = line % plan.inner_count;
            const std::size_t input_base =
                static_cast<std::size_t>(outer) * plan.input_sample_count *
                plan.inner_count + inner;
            const std::size_t output_base =
                static_cast<std::size_t>(outer) * plan.output_sample_count *
                plan.inner_count + inner;
            for (int sample = 0; sample < plan.output_sample_count; ++sample) {
                const long long filtered =
                    static_cast<long long>(sample) * plan.down;
                double sum = 0.0;
                for (int tap = 0; tap < plan.coefficient_count; ++tap) {
                    const long long candidate = filtered - tap;
                    if (candidate % plan.up != 0) continue;
                    const long long source = candidate / plan.up;
                    if (source < 0 || source >= plan.input_sample_count) continue;
                    sum += static_cast<double>(h[tap]) * static_cast<double>(
                        x[input_base + static_cast<std::size_t>(source) *
                          plan.inner_count]);
                }
                output.fp64[output_base +
                    static_cast<std::size_t>(sample) * plan.inner_count] = sum;
            }
        }
    } else {
        output.fp32.resize(plan.output_size);
        for (int line = 0; line < plan.line_count; ++line) {
            const int outer = line / plan.inner_count;
            const int inner = line % plan.inner_count;
            const std::size_t input_base =
                static_cast<std::size_t>(outer) * plan.input_sample_count *
                plan.inner_count + inner;
            const std::size_t output_base =
                static_cast<std::size_t>(outer) * plan.output_sample_count *
                plan.inner_count + inner;
            for (int sample = 0; sample < plan.output_sample_count; ++sample) {
                const long long filtered =
                    static_cast<long long>(sample) * plan.down;
                float sum = 0.0F;
                for (int tap = 0; tap < plan.coefficient_count; ++tap) {
                    const long long candidate = filtered - tap;
                    if (candidate % plan.up != 0) continue;
                    const long long source = candidate / plan.up;
                    if (source < 0 || source >= plan.input_sample_count) continue;
                    sum += ld(h[tap]) * ld(x[input_base +
                        static_cast<std::size_t>(source) * plan.inner_count]);
                }
                output.fp32[output_base +
                    static_cast<std::size_t>(sample) * plan.inner_count] = sum;
            }
        }
    }
    return output;
}
template <typename T>
ResamplePolyCpuResult resample_poly_typed_cpu(
    const std::vector<T>& x, int up, int down,
    const ResamplePolyOptions& options,
    const std::vector<T>* coefficients)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const bool custom = coefficients != nullptr;
    const auto plan = detail::prepare_resample_poly_layout(
        x.size(), up, down, options, custom,
        custom ? coefficients->size() : 0);
    ResamplePolyCpuResult output;
    output.dtype = resample_poly_output_dtype<T>(
        plan.identity, custom, plan.up);
    output.shape = plan.output_shape;
    if (plan.identity) {
        if constexpr (std::is_same_v<T, float>) output.fp32 = x;
        else if constexpr (std::is_same_v<T, __half>) output.fp16 = x;
        else if constexpr (std::is_same_v<T, std::int32_t>) output.int32 = x;
        else if constexpr (std::is_same_v<T, std::int16_t>) output.int16 = x;
        else output.int8 = x;
        return output;
    }

    std::vector<double> scaled_coefficients;
    if (custom) {
        scaled_coefficients.resize(coefficients->size());
        for (std::size_t index = 0; index < coefficients->size(); ++index)
            scaled_coefficients[index] = resample_poly_scaled_custom_value(
                (*coefficients)[index], plan.up);
    } else {
        scaled_coefficients = design_resample_poly_coefficients(plan, options);
    }

    if (output.dtype == ResamplePolyOutputDtype::fp32) {
        std::vector<float> taps(scaled_coefficients.size());
        for (std::size_t index = 0; index < taps.size(); ++index)
            taps[index] = static_cast<float>(scaled_coefficients[index]);
        output.fp32.resize(plan.output_size);
        for (int line = 0; line < plan.line_count; ++line) {
            const int outer = line / plan.inner_count;
            const int inner = line % plan.inner_count;
            const std::size_t input_base =
                (static_cast<std::size_t>(outer) * plan.input_sample_count *
                 plan.inner_count) + inner;
            const std::size_t output_base =
                (static_cast<std::size_t>(outer) * plan.output_sample_count *
                 plan.inner_count) + inner;
            for (int sample = 0; sample < plan.output_sample_count; ++sample) {
                const long long filtered =
                    (static_cast<long long>(sample) + plan.output_start) *
                    plan.down;
                float sum = 0.0F;
                for (int tap = 0; tap < plan.coefficient_count; ++tap) {
                    const long long candidate = filtered -
                        plan.prefix_zeros - tap;
                    if (candidate % plan.up != 0) continue;
                    const long long source = candidate / plan.up;
                    if (source < 0 || source >= plan.input_sample_count) continue;
                    sum += taps[tap] * ld(x[input_base +
                        static_cast<std::size_t>(source) * plan.inner_count]);
                }
                output.fp32[output_base +
                    static_cast<std::size_t>(sample) * plan.inner_count] = sum;
            }
        }
    } else {
        output.fp64.resize(plan.output_size);
        for (int line = 0; line < plan.line_count; ++line) {
            const int outer = line / plan.inner_count;
            const int inner = line % plan.inner_count;
            const std::size_t input_base =
                (static_cast<std::size_t>(outer) * plan.input_sample_count *
                 plan.inner_count) + inner;
            const std::size_t output_base =
                (static_cast<std::size_t>(outer) * plan.output_sample_count *
                 plan.inner_count) + inner;
            for (int sample = 0; sample < plan.output_sample_count; ++sample) {
                const long long filtered =
                    (static_cast<long long>(sample) + plan.output_start) *
                    plan.down;
                double sum = 0.0;
                for (int tap = 0; tap < plan.coefficient_count; ++tap) {
                    const long long candidate = filtered -
                        plan.prefix_zeros - tap;
                    if (candidate % plan.up != 0) continue;
                    const long long source = candidate / plan.up;
                    if (source < 0 || source >= plan.input_sample_count) continue;
                    sum += scaled_coefficients[tap] * detrend_host_load(
                        x[input_base + static_cast<std::size_t>(source) *
                          plan.inner_count]);
                }
                output.fp64[output_base +
                    static_cast<std::size_t>(sample) * plan.inner_count] = sum;
            }
        }
    }
    return output;
}

template <typename T>
void resample_poly_device(
    const DeviceArray<T>& x, int up, int down,
    ResamplePolyDeviceResult& output,
    const ResamplePolyOptions& options,
    const DeviceArray<T>* coefficients)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const bool custom = coefficients != nullptr;
    const auto plan = detail::prepare_resample_poly_layout(
        x.size(), up, down, options, custom,
        custom ? coefficients->size() : 0);
    output = ResamplePolyDeviceResult{};
    output.dtype = resample_poly_output_dtype<T>(
        plan.identity, custom, plan.up);
    output.shape = plan.output_shape;
    if (plan.identity) {
        if constexpr (std::is_same_v<T, float>) {
            output.fp32 = DeviceArray<float>(x.size());
            if (!x.empty()) cuda_utils::copy_device_to_device(
                output.fp32.data(), x.data(), x.size());
        } else if constexpr (std::is_same_v<T, __half>) {
            output.fp16 = DeviceArray<__half>(x.size());
            if (!x.empty()) cuda_utils::copy_device_to_device(
                output.fp16.data(), x.data(), x.size());
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
            output.int32 = DeviceArray<std::int32_t>(x.size());
            if (!x.empty()) cuda_utils::copy_device_to_device(
                output.int32.data(), x.data(), x.size());
        } else if constexpr (std::is_same_v<T, std::int16_t>) {
            output.int16 = DeviceArray<std::int16_t>(x.size());
            if (!x.empty()) cuda_utils::copy_device_to_device(
                output.int16.data(), x.data(), x.size());
        } else {
            output.int8 = DeviceArray<std::int8_t>(x.size());
            if (!x.empty()) cuda_utils::copy_device_to_device(
                output.int8.data(), x.data(), x.size());
        }
        cuda_utils::synchronize_stream();
        return;
    }

    DeviceArray<float> taps;
    if (custom) {
        taps = DeviceArray<float>(coefficients->size());
        resample_poly_custom_coefficients_fp32_device(
            *coefficients, plan.up, taps);
    } else {
        const auto designed = design_resample_poly_coefficients(plan, options);
        std::vector<float> fp32_taps(designed.size());
        for (std::size_t index = 0; index < designed.size(); ++index)
            fp32_taps[index] = static_cast<float>(designed[index]);
        taps = DeviceArray<float>::from_host(fp32_taps);
    }
    DeviceArray<float> computed(plan.output_size);
    resample_poly_fp32_compute_device(x, taps, plan, computed);
    if (output.dtype == ResamplePolyOutputDtype::fp32) {
        cuda_utils::synchronize_stream();
        output.fp32 = std::move(computed);
    } else {
        output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    }
}

#define INST(T) template ChannelizePolyCpuResult channelize_poly_typed_cpu(const std::vector<T>&,const std::vector<T>&,int);template ChannelizePolyCpuResult channelize_poly_typed_cpu(const std::vector<T>&,const std::vector<T>&,int,const std::vector<int>&,const std::vector<int>&);template DetrendCpuResult detrend_typed_cpu(std::vector<T>&,const DetrendOptions&);template void detrend_device(DeviceArray<T>&,DetrendDeviceResult&,const DetrendOptions&);template FirfilterCpuResult firfilter_typed_cpu(const std::vector<T>&,const std::vector<T>&,const FirfilterOptions&,const std::vector<T>*);template Firfilter2CpuResult firfilter2_typed_cpu(const std::vector<T>&,const std::vector<T>&,const Firfilter2Options&);template void firfilter2_device(const DeviceArray<T>&,const DeviceArray<T>&,Firfilter2DeviceResult&,const Firfilter2Options&);template HilbertCpuResult hilbert_typed_cpu(const std::vector<T>&,const HilbertOptions&);template void hilbert_device(const DeviceArray<T>&,HilbertDeviceResult&,const HilbertOptions&);template Hilbert2CpuResult hilbert2_typed_cpu(const std::vector<T>&,const Hilbert2Options&);template void hilbert2_device(const DeviceArray<T>&,Hilbert2DeviceResult&,const Hilbert2Options&);template std::vector<double>lfilter_zi_typed_cpu(const std::vector<T>&,const std::vector<T>&);template void lfilter_zi_device(const DeviceArray<T>&,const DeviceArray<T>&,LfilterZiDeviceResult&);template SosfiltCpuResult sosfilt_typed_cpu(const std::vector<T>&,int,const std::vector<T>&,const SosfiltOptions&,const std::vector<T>*);template WienerCpuResult wiener_typed_cpu(const std::vector<T>&,const WienerOptions&);template void wiener_device(const DeviceArray<T>&,WienerDeviceResult&,const WienerOptions&);template DecimateCpuResult decimate_typed_cpu(const std::vector<T>&,int,const DecimateOptions&,const std::vector<T>*);template void decimate_device(const DeviceArray<T>&,int,DecimateDeviceResult&,const DecimateOptions&,const DeviceArray<T>*);template ResampleCpuResult resample_typed_cpu(const std::vector<T>&,int,const ResampleOptions&,const std::vector<T>*);template void resample_device(const DeviceArray<T>&,int,ResampleDeviceResult&,const ResampleOptions&,const DeviceArray<T>*);template UpfirdnCpuResult upfirdn_typed_cpu(const std::vector<T>&,const std::vector<T>&,int,int,const UpfirdnOptions&);template ResamplePolyCpuResult resample_poly_typed_cpu(const std::vector<T>&,int,int,const ResamplePolyOptions&,const std::vector<T>*);template void resample_poly_device(const DeviceArray<T>&,int,int,ResamplePolyDeviceResult&,const ResamplePolyOptions&,const DeviceArray<T>*)
INST(float);INST(__half);INST(std::int32_t);INST(std::int16_t);INST(std::int8_t);
#undef INST
#define INST_FREQ(T) template std::vector<ComplexDouble> freq_shift_typed_cpu(const std::vector<T>&,double,double);template std::vector<ComplexDouble> freq_shift_typed_cpu(const std::vector<TypedComplex<T>>&,double,double);template void freq_shift_device(const DeviceArray<T>&,DeviceArray<ComplexDouble>&,double,double);template void freq_shift_device(const DeviceArray<TypedComplex<T>>&,DeviceArray<ComplexDouble>&,double,double)
INST_FREQ(float);INST_FREQ(__half);INST_FREQ(std::int32_t);INST_FREQ(std::int16_t);INST_FREQ(std::int8_t);
#undef INST_FREQ
} // namespace cusignal
