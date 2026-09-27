#include <cusignal/operators/radartools/radartools_typed.h>
#include <cusignal/runtime/host_output_finalize.h>

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cusignal {
namespace {
constexpr float pi = 3.14159265358979323846F;
template<class T> float load(T value) { return detail::SimpleSignalTypePolicy<T>::load(value); }
int next_power_of_two(int n)
{
    int value = 1;
    while (value < n) {
        if (value > std::numeric_limits<int>::max() / 2)
            throw std::invalid_argument("ambgfun FFT length exceeds int capacity");
        value <<= 1;
    }
    return value;
}

using ComplexReference = std::complex<double>;
constexpr double reference_pi = 3.141592653589793238462643383279502884;

template <class T>
std::vector<ComplexReference> normalized_reference(
    const std::vector<TypedComplex<T>>& input)
{
    double sum = 0.0;
    for (const auto& value : input) {
        const double real = static_cast<double>(load(value.real));
        const double imag = static_cast<double>(load(value.imag));
        sum += real * real + imag * imag;
    }
    const double norm = std::sqrt(sum);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<ComplexReference> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (norm == 0.0) {
            output[index] = {nan, nan};
        } else {
            output[index] = ComplexReference(
                static_cast<double>(load(input[index].real)) / norm,
                static_cast<double>(load(input[index].imag)) / norm);
        }
    }
    return output;
}

std::vector<ComplexReference> direct_transform(
    const std::vector<ComplexReference>& input, int length, bool inverse)
{
    std::vector<ComplexReference> output(static_cast<std::size_t>(length));
    const double sign = inverse ? 1.0 : -1.0;
    const double scale = inverse ? 1.0 / static_cast<double>(length) : 1.0;
    for (int frequency = 0; frequency < length; ++frequency) {
        ComplexReference sum(0.0, 0.0);
        for (std::size_t sample = 0; sample < input.size() && sample < output.size(); ++sample) {
            const double angle = sign * 2.0 * reference_pi *
                static_cast<double>(frequency) * static_cast<double>(sample) /
                static_cast<double>(length);
            sum += input[sample] * ComplexReference(std::cos(angle), std::sin(angle));
        }
        output[static_cast<std::size_t>(frequency)] = sum * scale;
    }
    return output;
}

void validate_ambgfun_options(const AmbgfunOptions& options)
{
    switch (options.cut) {
    case AmbgfunCut::two_dimensional: return;
    case AmbgfunCut::delay:
    case AmbgfunCut::doppler:
        if (options.fs == 0.0)
            throw std::invalid_argument("ambgfun fs must be nonzero for one-dimensional cuts");
        return;
    }
    throw std::invalid_argument("ambgfun cut is invalid");
}

int ambgfun_first_extent(
    std::size_t count, const std::vector<int>& shape, const char* label)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string("ambgfun ") + label + " length exceeds int capacity");
    if (shape.empty()) {
        return static_cast<int>(count);
    }
    std::size_t product = 1;
    for (const int extent : shape) {
        if (extent < 1)
            throw std::invalid_argument(std::string("ambgfun ") + label + " shape");
        if (product > count / static_cast<std::size_t>(extent))
            throw std::invalid_argument(std::string("ambgfun ") + label + " shape product");
        product *= static_cast<std::size_t>(extent);
    }
    if (product != count)
        throw std::invalid_argument(std::string("ambgfun ") + label + " data/shape mismatch");
    return shape.front();
}

void validate_ambgfun_rank_contract(
    const AmbgfunOptions& options, bool has_reference)
{
    if (!has_reference && !options.reference_shape.empty())
        throw std::invalid_argument("ambgfun reference_shape requires y");
    const bool high_rank = options.input_shape.size() > 1 ||
        (has_reference && options.reference_shape.size() > 1);
    if (high_rank && options.cut != AmbgfunCut::two_dimensional)
        throw std::invalid_argument("ambgfun high-rank input is only defined for cut='2d'");
}

template <class T>
constexpr bool ambgfun_two_dimensional_fp32()
{
    return std::is_same_v<T, float> || std::is_same_v<T, __half>;
}

template <class T>
AmbgfunOutputDtype ambgfun_output_dtype(AmbgfunCut cut)
{
    return cut == AmbgfunCut::two_dimensional && ambgfun_two_dimensional_fp32<T>()
        ? AmbgfunOutputDtype::fp32 : AmbgfunOutputDtype::fp64;
}

AmbgfunCpuResult finish_ambgfun_cpu(
    std::vector<double> values, std::vector<int> shape, AmbgfunOutputDtype dtype)
{
    AmbgfunCpuResult result;
    result.dtype = dtype;
    result.shape = std::move(shape);
    if (dtype == AmbgfunOutputDtype::fp32) {
        result.fp32.resize(values.size());
        for (std::size_t index = 0; index < values.size(); ++index)
            result.fp32[index] = static_cast<float>(values[index]);
    } else {
        result.fp64 = std::move(values);
    }
    return result;
}

struct CaCfarLayout {
    int rank{0};
    int rows{1};
    int columns{1};
    int guard_rows{0};
    int guard_columns{0};
    int reference_rows{0};
    int reference_columns{0};
    int reference_count{0};
    int element_count{1};
    float alpha{0.0F};
};

CaCfarLayout validate_ca_cfar(
    std::size_t input_size,
    const std::vector<int>& shape,
    const CaCfarOptions& options)
{
    if (shape.size() > 2)
        throw std::invalid_argument("ca_cfar supports only scalar, 1D, or 2D input");
    CaCfarLayout layout;
    layout.rank = static_cast<int>(shape.size());
    if (shape.empty()) {
        if (input_size != 1)
            throw std::invalid_argument("ca_cfar scalar input must contain one element");
        return layout;
    }
    if (options.guard_cells.size() != shape.size() ||
        options.reference_cells.size() != shape.size()) {
        throw std::invalid_argument("ca_cfar cell parameter rank mismatch");
    }
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (shape[dimension] < 0)
            throw std::invalid_argument("ca_cfar shape dimensions must be nonnegative");
        if (options.guard_cells[dimension] < 0 ||
            options.reference_cells[dimension] < 1) {
            throw std::invalid_argument("ca_cfar guard/reference cells are invalid");
        }
        const std::int64_t margin = static_cast<std::int64_t>(
            options.guard_cells[dimension]) + options.reference_cells[dimension];
        if (static_cast<std::int64_t>(shape[dimension]) <= 2 * margin)
            throw std::invalid_argument("ca_cfar input dimension is too small");
    }
    if (!std::isfinite(options.pfa) || options.pfa <= 0.0 || options.pfa > 1.0)
        throw std::invalid_argument("ca_cfar pfa must be in (0,1]");
    layout.rows = shape[0];
    layout.columns = shape.size() == 2 ? shape[1] : 1;
    const std::int64_t count64 = static_cast<std::int64_t>(layout.rows) * layout.columns;
    if (count64 > std::numeric_limits<int>::max() ||
        input_size != static_cast<std::size_t>(count64)) {
        throw std::invalid_argument("ca_cfar input size or capacity is invalid");
    }
    layout.element_count = static_cast<int>(count64);
    layout.guard_rows = options.guard_cells[0];
    layout.reference_rows = options.reference_cells[0];
    if (shape.size() == 2) {
        layout.guard_columns = options.guard_cells[1];
        layout.reference_columns = options.reference_cells[1];
        const std::int64_t references =
            2LL * layout.reference_rows *
                (2LL * layout.reference_columns + 2LL * layout.guard_columns + 1LL) +
            2LL * (2LL * layout.guard_rows + 1LL) * layout.reference_columns;
        if (references > std::numeric_limits<int>::max())
            throw std::invalid_argument("ca_cfar reference count exceeds int capacity");
        layout.reference_count = static_cast<int>(references);
    } else {
        layout.reference_count = 2 * layout.reference_rows;
    }
    const double reference_count = static_cast<double>(layout.reference_count);
    layout.alpha = static_cast<float>(
        reference_count * (std::pow(options.pfa, -1.0 / reference_count) - 1.0));
    return layout;
}

template <class T>
bool ca_cfar_detect_host(T input, float threshold)
{
    if (std::isnan(threshold)) return false;
    if constexpr (std::is_integral_v<T>) {
        if (std::isinf(threshold)) return threshold < 0.0F;
        const float low = static_cast<float>(std::numeric_limits<T>::lowest());
        const float high = static_cast<float>(std::numeric_limits<T>::max());
        if (threshold < low) return true;
        if (threshold >= high) return false;
        return input > static_cast<T>(std::floor(threshold));
    } else {
        return load(input) > threshold;
    }
}

float radar_window_host(
    RadarWindowKind kind, int index, int length,
    const std::vector<float>* explicit_window)
{
    if (kind == RadarWindowKind::explicit_fp32 ||
        kind == RadarWindowKind::explicit_fp64)
        return (*explicit_window)[static_cast<std::size_t>(index)];
    if (kind == RadarWindowKind::none || length <= 1) return 1.0F;
    const float phase = 2.0F * pi * static_cast<float>(index) /
        static_cast<float>(length - 1);
    if (kind == RadarWindowKind::hann) return 0.5F - 0.5F * std::cos(phase);
    if (kind == RadarWindowKind::hamming) return 0.54F - 0.46F * std::cos(phase);
    throw std::invalid_argument("unsupported radar window kind");
}

void validate_radar_window(
    RadarWindowKind kind, std::size_t expected,
    const std::vector<float>* explicit_window)
{
    const int code = static_cast<int>(kind);
    if (code < static_cast<int>(RadarWindowKind::none) ||
        code > static_cast<int>(RadarWindowKind::explicit_fp64)) {
        throw std::invalid_argument("unsupported radar window kind");
    }
    if (kind == RadarWindowKind::explicit_fp32 ||
        kind == RadarWindowKind::explicit_fp64) {
        if (explicit_window == nullptr || explicit_window->size() != expected)
            throw std::invalid_argument("explicit radar window size mismatch");
    } else if (explicit_window != nullptr) {
        throw std::invalid_argument("explicit radar window requires explicit_fp32 or explicit_fp64 kind");
    }
}

template <class T>
bool radar_complex_result_is_fp64(RadarWindowKind window)
{
    if (window == RadarWindowKind::none) return std::is_integral_v<T>;
    if (window == RadarWindowKind::explicit_fp32)
        return std::is_same_v<T, std::int32_t>;
    return true;
}

std::size_t checked_radar_elements(int first, int second, const char* function)
{
    const std::int64_t count = static_cast<std::int64_t>(first) * second;
    if (first < 1 || second < 1 || count > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(function) + " shape exceeds kernel capacity");
    return static_cast<std::size_t>(count);
}
}

namespace detail {
template <class T>
void ambgfun_fp32_compute_device(
    const DeviceArray<TypedComplex<T>>& x,
    const DeviceArray<TypedComplex<T>>& y,
    int nx,
    int ny,
    int cut, float fs, float cut_value,
    DeviceArray<float>& output);

template <class T>
void ca_cfar_fp32_compute_device(
    const DeviceArray<T>& input,
    int rank, int rows, int columns,
    int guard_rows, int guard_columns,
    int reference_rows, int reference_columns,
    float alpha,
    DeviceArray<float>& threshold,
    DeviceArray<CfarDetection>& detections);

template <class T>
void cfar_alpha_fp32_compute_device(
    T pfa, int reference, DeviceArray<float>& output);

template <class T>
void pulse_compression_fp32_compute_device(
    const DeviceArray<TypedComplex<T>>& input,
    const DeviceArray<TypedComplex<T>>& pulse_template,
    const PulseCompressionOptions& options,
    const DeviceArray<float>* explicit_window,
    DeviceArray<ComplexFloat>& output);

template <class T>
void pulse_doppler_fp32_compute_device(
    const DeviceArray<TypedComplex<T>>& input,
    const PulseDopplerOptions& options,
    const DeviceArray<float>* explicit_window,
    DeviceArray<ComplexFloat>& output);
}

template <class T>
AmbgfunCpuResult ambgfun_typed_cpu(
    const std::vector<TypedComplex<T>>& x,
    const AmbgfunOptions& options,
    const std::vector<TypedComplex<T>>* y)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    validate_ambgfun_options(options);
    validate_ambgfun_rank_contract(options, y != nullptr);
    const auto& reference = y == nullptr ? x : *y;
    if (x.empty() || reference.empty())
        throw std::invalid_argument("ambgfun inputs must be nonempty");
    const int nx = ambgfun_first_extent(x.size(), options.input_shape, "input");
    const int ny = y == nullptr
        ? nx
        : ambgfun_first_extent(reference.size(), options.reference_shape, "reference");
    if (nx > std::numeric_limits<int>::max() - ny + 1)
        throw std::invalid_argument("ambgfun input length exceeds int capacity");
    const int rows = nx + ny - 1;
    const int nfreq = next_power_of_two(rows);
    const auto xnorm = normalized_reference(x);
    const auto ynorm = y == nullptr ? xnorm : normalized_reference(reference);
    std::vector<double> values;
    std::vector<int> shape;

    if (options.cut == AmbgfunCut::two_dimensional) {
        values.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(nfreq));
        shape = {rows, nfreq};
        for (int row = 0; row < rows; ++row) {
            for (int shifted = 0; shifted < nfreq; ++shifted) {
                const int frequency = (shifted - nfreq / 2 + nfreq) % nfreq;
                ComplexReference sum(0.0, 0.0);
                for (int column = 0; column < ny; ++column) {
                    const int x_column = column - (nx - 1) + row;
                    if (x_column < 0 || x_column >= nx) continue;
                    const double angle = 2.0 * reference_pi *
                        static_cast<double>(frequency) * static_cast<double>(column) /
                        static_cast<double>(nfreq);
                    sum += ynorm[static_cast<std::size_t>(column)] *
                           std::conj(xnorm[static_cast<std::size_t>(x_column)]) *
                           ComplexReference(std::cos(angle), std::sin(angle));
                }
                values[static_cast<std::size_t>(row) * nfreq + shifted] = std::abs(sum);
            }
        }
    } else if (options.cut == AmbgfunCut::delay) {
        shape = {nfreq};
        auto x_frequency = direct_transform(xnorm, nfreq, false);
        for (int index = 0; index < nfreq; ++index) {
            const double frequency = -options.fs / 2.0 +
                options.fs * static_cast<double>(index) / static_cast<double>(nfreq);
            const double angle = 2.0 * reference_pi * frequency * options.cut_value;
            x_frequency[static_cast<std::size_t>(index)] *=
                ComplexReference(std::cos(angle), std::sin(angle));
        }
        const auto x_shift = direct_transform(x_frequency, nfreq, true);
        std::vector<ComplexReference> product(static_cast<std::size_t>(nfreq));
        for (int index = 0; index < ny && index < nfreq; ++index)
            product[static_cast<std::size_t>(index)] =
                ynorm[static_cast<std::size_t>(index)] *
                std::conj(x_shift[static_cast<std::size_t>(index)]);
        const auto transformed = direct_transform(product, nfreq, true);
        values.resize(static_cast<std::size_t>(nfreq));
        for (int index = 0; index < nfreq; ++index) {
            const int source = (index + nfreq / 2) % nfreq;
            values[static_cast<std::size_t>(index)] =
                static_cast<double>(nfreq) * std::abs(transformed[static_cast<std::size_t>(source)]);
        }
    } else {
        shape = {rows};
        std::vector<ComplexReference> modulated = xnorm;
        for (int index = 0; index < nx; ++index) {
            const double time = static_cast<double>(index) / options.fs;
            const double angle = 2.0 * reference_pi * options.cut_value * time;
            modulated[static_cast<std::size_t>(index)] *=
                ComplexReference(std::cos(angle), std::sin(angle));
        }
        const auto y_frequency = direct_transform(ynorm, rows, false);
        const auto x_frequency = direct_transform(modulated, rows, false);
        std::vector<ComplexReference> product(static_cast<std::size_t>(rows));
        for (int index = 0; index < rows; ++index)
            product[static_cast<std::size_t>(index)] =
                y_frequency[static_cast<std::size_t>(index)] *
                std::conj(x_frequency[static_cast<std::size_t>(index)]);
        const auto transformed = direct_transform(product, rows, true);
        values.resize(static_cast<std::size_t>(rows));
        for (int index = 0; index < rows; ++index) {
            const int source = (index - rows / 2 + rows) % rows;
            values[static_cast<std::size_t>(index)] =
                std::abs(transformed[static_cast<std::size_t>(source)]);
        }
    }
    return finish_ambgfun_cpu(
        std::move(values), std::move(shape), ambgfun_output_dtype<T>(options.cut));
}

template <class T>
void ambgfun_device(
    const DeviceArray<TypedComplex<T>>& x,
    AmbgfunDeviceResult& output,
    const AmbgfunOptions& options,
    const DeviceArray<TypedComplex<T>>* y)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    validate_ambgfun_options(options);
    validate_ambgfun_rank_contract(options, y != nullptr);
    const auto& reference = y == nullptr ? x : *y;
    if (x.empty() || reference.empty())
        throw std::invalid_argument("ambgfun inputs must be nonempty");
    const int nx = ambgfun_first_extent(x.size(), options.input_shape, "input");
    const int ny = y == nullptr
        ? nx
        : ambgfun_first_extent(reference.size(), options.reference_shape, "reference");
    if (nx > std::numeric_limits<int>::max() - ny + 1)
        throw std::invalid_argument("ambgfun input length exceeds int capacity");
    const int rows = nx + ny - 1;
    const int nfreq = next_power_of_two(rows);
    const std::size_t count = options.cut == AmbgfunCut::two_dimensional
        ? static_cast<std::size_t>(rows) * static_cast<std::size_t>(nfreq)
        : static_cast<std::size_t>(options.cut == AmbgfunCut::delay ? nfreq : rows);
    DeviceArray<float> computed(count);
    detail::ambgfun_fp32_compute_device(
        x, reference, nx, ny, static_cast<int>(options.cut),
        static_cast<float>(options.fs), static_cast<float>(options.cut_value), computed);
    output = AmbgfunDeviceResult{};
    output.dtype = ambgfun_output_dtype<T>(options.cut);
    output.shape = options.cut == AmbgfunCut::two_dimensional
        ? std::vector<int>{rows, nfreq}
        : std::vector<int>{options.cut == AmbgfunCut::delay ? nfreq : rows};
    if (output.dtype == AmbgfunOutputDtype::fp32) {
        cuda_utils::synchronize_stream();
        output.fp32 = std::move(computed);
    } else {
        output.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    }
}
template <class T>
CaCfarCpuResult ca_cfar_typed_cpu(
    const std::vector<T>& input,
    const std::vector<int>& shape,
    const CaCfarOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const CaCfarLayout layout = validate_ca_cfar(input.size(), shape, options);
    CaCfarCpuResult result;
    result.shape = shape;
    result.threshold.assign(input.size(), 0.0F);
    result.detections.assign(input.size(), CfarDetection::no);
    if (layout.rank == 1) {
        std::vector<float> cumulative(input.size());
        float running = 0.0F;
        for (int index = 0; index < layout.rows; ++index) {
            running += load(input[static_cast<std::size_t>(index)]);
            cumulative[static_cast<std::size_t>(index)] = running;
        }
        const int margin = layout.guard_rows + layout.reference_rows;
        for (int index = margin; index < layout.rows - margin; ++index) {
            const int outer_right = index + margin;
            const int outer_left = index - margin - 1;
            const int inner_right = index + layout.guard_rows;
            const int inner_left = index - layout.guard_rows - 1;
            const float outer = outer_left >= 0
                ? cumulative[static_cast<std::size_t>(outer_right)] -
                    cumulative[static_cast<std::size_t>(outer_left)]
                : cumulative[static_cast<std::size_t>(outer_right)];
            const float inner = cumulative[static_cast<std::size_t>(inner_right)] -
                cumulative[static_cast<std::size_t>(inner_left)];
            result.threshold[static_cast<std::size_t>(index)] =
                (layout.alpha / static_cast<float>(layout.reference_count)) *
                (outer - inner);
        }
    } else if (layout.rank == 2) {
        std::vector<float> cumulative(input.size());
        for (int column = 0; column < layout.columns; ++column) {
            float running = 0.0F;
            for (int row = 0; row < layout.rows; ++row) {
                const std::size_t index = static_cast<std::size_t>(row) *
                    layout.columns + column;
                running += load(input[index]);
                cumulative[index] = running;
            }
        }
        for (int row = 0; row < layout.rows; ++row) {
            float running = 0.0F;
            for (int column = 0; column < layout.columns; ++column) {
                const std::size_t index = static_cast<std::size_t>(row) *
                    layout.columns + column;
                running += cumulative[index];
                cumulative[index] = running;
            }
        }
        const int row_margin = layout.guard_rows + layout.reference_rows;
        const int column_margin = layout.guard_columns + layout.reference_columns;
        for (int row = row_margin; row < layout.rows - row_margin; ++row) {
            for (int column = column_margin;
                 column < layout.columns - column_margin; ++column) {
                const auto at = [&](int r, int c) {
                    return cumulative[static_cast<std::size_t>(r) * layout.columns + c];
                };
                const int outer_top = row + row_margin;
                const int outer_left = row - row_margin - 1;
                const int outer_right_column = column + column_margin;
                const int outer_left_column = column - column_margin - 1;
                float outer = at(outer_top, outer_right_column);
                if (outer_left >= 0) outer -= at(outer_left, outer_right_column);
                if (outer_left_column >= 0) outer -= at(outer_top, outer_left_column);
                if (outer_left >= 0 && outer_left_column >= 0)
                    outer += at(outer_left, outer_left_column);
                const float inner =
                    at(row + layout.guard_rows, column + layout.guard_columns) -
                    at(row - layout.guard_rows - 1, column + layout.guard_columns) -
                    at(row + layout.guard_rows, column - layout.guard_columns - 1) +
                    at(row - layout.guard_rows - 1, column - layout.guard_columns - 1);
                const std::size_t index = static_cast<std::size_t>(row) *
                    layout.columns + column;
                result.threshold[index] =
                    (layout.alpha / static_cast<float>(layout.reference_count)) *
                    (outer - inner);
            }
        }
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
        result.detections[index] = ca_cfar_detect_host(input[index], result.threshold[index])
            ? CfarDetection::yes : CfarDetection::no;
    }
    return result;
}

template <class T>
void ca_cfar_device(
    const DeviceArray<T>& input,
    const std::vector<int>& shape,
    const CaCfarOptions& options,
    DeviceArray<float>& threshold,
    DeviceArray<CfarDetection>& detections)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const CaCfarLayout layout = validate_ca_cfar(input.size(), shape, options);
    if (threshold.size() != input.size() || detections.size() != input.size()) {
        throw std::invalid_argument(
            "ca_cfar preallocated output shape must match input");
    }
    detail::ca_cfar_fp32_compute_device(
        input, layout.rank, layout.rows, layout.columns,
        layout.guard_rows, layout.guard_columns,
        layout.reference_rows, layout.reference_columns,
        layout.alpha, threshold, detections);
}

template <class T>
CaCfarDeviceResult ca_cfar_device(
    const DeviceArray<T>& input,
    const std::vector<int>& shape,
    const CaCfarOptions& options)
{
    CaCfarDeviceResult result;
    result.shape = shape;
    result.threshold = DeviceArray<float>(input.size());
    result.detections = DeviceArray<CfarDetection>(input.size());
    ca_cfar_device(
        input, shape, options, result.threshold, result.detections);
    return result;
}
template <class T>
double cfar_alpha_typed_cpu(T pfa, int reference)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const double probability = static_cast<double>(load(pfa));
    if (reference < 1 || !std::isfinite(probability) ||
        probability <= 0.0 || probability > 1.0) {
        throw std::invalid_argument("cfar_alpha requires reference >= 1 and pfa in (0,1]");
    }
    return static_cast<double>(reference) *
        (std::pow(probability, -1.0 / static_cast<double>(reference)) - 1.0);
}

template <class T>
DeviceArray<double> cfar_alpha_device(T pfa, int reference)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const float probability = load(pfa);
    if (reference < 1 || !std::isfinite(probability) ||
        probability <= 0.0F || probability > 1.0F) {
        throw std::invalid_argument("cfar_alpha requires reference >= 1 and pfa in (0,1]");
    }
    DeviceArray<float> computed(1);
    detail::cfar_alpha_fp32_compute_device(pfa, reference, computed);
    return cuda_utils::finalize_fp64_device_storage(computed);
}
template <class T>
RadarComplexCpuResult pulse_compression_typed_cpu(
    const std::vector<TypedComplex<T>>& input,
    const std::vector<TypedComplex<T>>& pulse_template,
    const PulseCompressionOptions& options,
    const std::vector<float>* explicit_window)
{
    const int nfft = options.nfft == -1 ? options.samples_per_pulse : options.nfft;
    const std::size_t input_count = checked_radar_elements(
        options.num_pulses, options.samples_per_pulse, "pulse_compression");
    const std::size_t output_count = checked_radar_elements(
        options.num_pulses, nfft, "pulse_compression");
    if (options.num_pulses < 1 || options.samples_per_pulse < 1 || nfft < 1 ||
        input.size() != input_count || pulse_template.empty() ||
        pulse_template.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("pulse_compression shape is invalid");
    }
    validate_radar_window(options.window, pulse_template.size(), explicit_window);
    std::vector<double> template_real(pulse_template.size());
    std::vector<double> template_imag(pulse_template.size());
    double norm = 0.0;
    for (std::size_t index = 0; index < pulse_template.size(); ++index) {
        const double window = radar_window_host(
            options.window, static_cast<int>(index),
            static_cast<int>(pulse_template.size()), explicit_window);
        template_real[index] = static_cast<double>(load(pulse_template[index].real)) * window;
        template_imag[index] = static_cast<double>(load(pulse_template[index].imag)) * window;
        norm += template_real[index] * template_real[index] +
            template_imag[index] * template_imag[index];
    }
    norm = options.normalize ? std::sqrt(norm) : 1.0;
    RadarComplexCpuResult result;
    result.is_fp64 = radar_complex_result_is_fp64<T>(options.window);
    result.shape = {options.num_pulses, nfft};
    if (result.is_fp64) result.fp64.resize(output_count);
    else result.fp32.resize(output_count);
    for (int pulse = 0; pulse < options.num_pulses; ++pulse) {
        for (int output_index = 0; output_index < nfft; ++output_index) {
            double real = 0.0;
            double imag = 0.0;
            for (int tap = 0; tap < nfft && tap < static_cast<int>(pulse_template.size()); ++tap) {
                const int sample = (output_index + tap) % nfft;
                const std::size_t source = static_cast<std::size_t>(pulse) *
                    options.samples_per_pulse + sample;
                const double xr = sample < options.samples_per_pulse ? load(input[source].real) : 0.0;
                const double xi = sample < options.samples_per_pulse ? load(input[source].imag) : 0.0;
                const double hr = template_real[tap] / norm;
                const double hi = -template_imag[tap] / norm;
                real += xr * hr - xi * hi;
                imag += xr * hi + xi * hr;
            }
            const std::size_t target = static_cast<std::size_t>(pulse) * nfft + output_index;
            if (result.is_fp64) result.fp64[target] = ComplexDouble{real, imag};
            else result.fp32[target] = ComplexFloat{static_cast<float>(real), static_cast<float>(imag)};
        }
    }
    return result;
}

template <class T>
RadarComplexDeviceResult pulse_compression_device(
    const DeviceArray<TypedComplex<T>>& input,
    const DeviceArray<TypedComplex<T>>& pulse_template,
    const PulseCompressionOptions& options,
    const DeviceArray<float>* explicit_window)
{
    const int nfft = options.nfft == -1 ? options.samples_per_pulse : options.nfft;
    const std::size_t input_count = checked_radar_elements(
        options.num_pulses, options.samples_per_pulse, "pulse_compression");
    const std::size_t output_count = checked_radar_elements(
        options.num_pulses, nfft, "pulse_compression");
    if (options.num_pulses < 1 || options.samples_per_pulse < 1 || nfft < 1 ||
        input.size() != input_count || pulse_template.empty() ||
        pulse_template.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("pulse_compression shape is invalid");
    }
    const bool needs_window = options.window == RadarWindowKind::explicit_fp32 ||
        options.window == RadarWindowKind::explicit_fp64;
    if (needs_window != (explicit_window != nullptr) ||
        (explicit_window != nullptr && explicit_window->size() != pulse_template.size())) {
        throw std::invalid_argument("pulse_compression window is invalid");
    }
    const int window_code = static_cast<int>(options.window);
    if (window_code < 0 || window_code > 4)
        throw std::invalid_argument("unsupported pulse_compression window");
    DeviceArray<ComplexFloat> computed(output_count);
    detail::pulse_compression_fp32_compute_device(
        input, pulse_template, options, explicit_window, computed);
    RadarComplexDeviceResult result;
    result.shape = {options.num_pulses, nfft};
    result.is_fp64 = radar_complex_result_is_fp64<T>(options.window);
    if (result.is_fp64) result.fp64 = cuda_utils::finalize_complex_fp64_device_storage(computed);
    else result.fp32 = std::move(computed);
    return result;
}

template <class T>
RadarComplexCpuResult pulse_doppler_typed_cpu(
    const std::vector<TypedComplex<T>>& input,
    const PulseDopplerOptions& options,
    const std::vector<float>* explicit_window)
{
    const int nfft = options.nfft == -1 ? options.num_pulses : options.nfft;
    const std::size_t input_count = checked_radar_elements(
        options.num_pulses, options.samples_per_pulse, "pulse_doppler");
    const std::size_t output_count = checked_radar_elements(
        nfft, options.samples_per_pulse, "pulse_doppler");
    if (options.num_pulses < 1 || options.samples_per_pulse < 1 || nfft < 1 ||
        input.size() != input_count) {
        throw std::invalid_argument("pulse_doppler shape is invalid");
    }
    validate_radar_window(options.window, options.num_pulses, explicit_window);
    RadarComplexCpuResult result;
    result.is_fp64 = radar_complex_result_is_fp64<T>(options.window);
    result.shape = {nfft, options.samples_per_pulse};
    if (result.is_fp64) result.fp64.resize(output_count);
    else result.fp32.resize(output_count);
    for (int frequency = 0; frequency < nfft; ++frequency) {
        for (int sample = 0; sample < options.samples_per_pulse; ++sample) {
            double real = 0.0;
            double imag = 0.0;
            for (int pulse = 0; pulse < options.num_pulses && pulse < nfft; ++pulse) {
                const double window = radar_window_host(
                    options.window, pulse, options.num_pulses, explicit_window);
                const double angle = -2.0 * static_cast<double>(pi) * frequency * pulse / nfft;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const auto& value = input[static_cast<std::size_t>(pulse) *
                    options.samples_per_pulse + sample];
                const double xr = load(value.real) * window;
                const double xi = load(value.imag) * window;
                real += xr * cosine - xi * sine;
                imag += xr * sine + xi * cosine;
            }
            const std::size_t target = static_cast<std::size_t>(frequency) *
                options.samples_per_pulse + sample;
            if (result.is_fp64) result.fp64[target] = ComplexDouble{real, imag};
            else result.fp32[target] = ComplexFloat{static_cast<float>(real), static_cast<float>(imag)};
        }
    }
    return result;
}

template <class T>
void pulse_doppler_device(
    const DeviceArray<TypedComplex<T>>& input,
    const PulseDopplerOptions& options,
    DeviceArray<ComplexFloat>& output,
    const DeviceArray<float>* explicit_window)
{
    const int nfft = options.nfft == -1 ? options.num_pulses : options.nfft;
    const std::size_t input_count = checked_radar_elements(
        options.num_pulses, options.samples_per_pulse, "pulse_doppler");
    const std::size_t output_count = checked_radar_elements(
        nfft, options.samples_per_pulse, "pulse_doppler");
    if (options.num_pulses < 1 || options.samples_per_pulse < 1 || nfft < 1 ||
        input.size() != input_count) {
        throw std::invalid_argument("pulse_doppler shape is invalid");
    }
    const bool needs_window = options.window == RadarWindowKind::explicit_fp32 ||
        options.window == RadarWindowKind::explicit_fp64;
    if (needs_window != (explicit_window != nullptr) ||
        (explicit_window != nullptr && explicit_window->size() != static_cast<std::size_t>(options.num_pulses))) {
        throw std::invalid_argument("pulse_doppler window is invalid");
    }
    const int window_code = static_cast<int>(options.window);
    if (window_code < 0 || window_code > 4)
        throw std::invalid_argument("unsupported pulse_doppler window");
    if (output.size() != output_count)
        throw std::invalid_argument("pulse_doppler preallocated output shape mismatch");
    detail::pulse_doppler_fp32_compute_device(input, options, explicit_window, output);
}

template <class T>
RadarComplexDeviceResult pulse_doppler_device(
    const DeviceArray<TypedComplex<T>>& input,
    const PulseDopplerOptions& options,
    const DeviceArray<float>* explicit_window)
{
    const int nfft = options.nfft == -1 ? options.num_pulses : options.nfft;
    const std::size_t output_count = checked_radar_elements(
        nfft, options.samples_per_pulse, "pulse_doppler");
    DeviceArray<ComplexFloat> computed(output_count);
    pulse_doppler_device(input, options, computed, explicit_window);
    RadarComplexDeviceResult result;
    result.shape = {nfft, options.samples_per_pulse};
    result.is_fp64 = radar_complex_result_is_fp64<T>(options.window);
    if (result.is_fp64) result.fp64 = cuda_utils::finalize_complex_fp64_device_storage(computed);
    else result.fp32 = std::move(computed);
    return result;
}

#define INSTANTIATE_CPU(T) template AmbgfunCpuResult ambgfun_typed_cpu(const std::vector<TypedComplex<T>>&,const AmbgfunOptions&,const std::vector<TypedComplex<T>>*);template void ambgfun_device(const DeviceArray<TypedComplex<T>>&,AmbgfunDeviceResult&,const AmbgfunOptions&,const DeviceArray<TypedComplex<T>>*);template CaCfarCpuResult ca_cfar_typed_cpu(const std::vector<T>&,const std::vector<int>&,const CaCfarOptions&);template CaCfarDeviceResult ca_cfar_device(const DeviceArray<T>&,const std::vector<int>&,const CaCfarOptions&);template void ca_cfar_device(const DeviceArray<T>&,const std::vector<int>&,const CaCfarOptions&,DeviceArray<float>&,DeviceArray<CfarDetection>&);template double cfar_alpha_typed_cpu(T,int);template DeviceArray<double> cfar_alpha_device(T,int);template RadarComplexCpuResult pulse_compression_typed_cpu(const std::vector<TypedComplex<T>>&,const std::vector<TypedComplex<T>>&,const PulseCompressionOptions&,const std::vector<float>*);template RadarComplexDeviceResult pulse_compression_device(const DeviceArray<TypedComplex<T>>&,const DeviceArray<TypedComplex<T>>&,const PulseCompressionOptions&,const DeviceArray<float>*);template RadarComplexCpuResult pulse_doppler_typed_cpu(const std::vector<TypedComplex<T>>&,const PulseDopplerOptions&,const std::vector<float>*);template RadarComplexDeviceResult pulse_doppler_device(const DeviceArray<TypedComplex<T>>&,const PulseDopplerOptions&,const DeviceArray<float>*);template void pulse_doppler_device(const DeviceArray<TypedComplex<T>>&,const PulseDopplerOptions&,DeviceArray<ComplexFloat>&,const DeviceArray<float>*)
INSTANTIATE_CPU(float);INSTANTIATE_CPU(__half);INSTANTIATE_CPU(std::int32_t);INSTANTIATE_CPU(std::int16_t);INSTANTIATE_CPU(std::int8_t);
#undef INSTANTIATE_CPU

} // namespace cusignal
