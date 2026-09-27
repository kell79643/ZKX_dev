#include <cusignal/operators/spectral_analysis/spectral_analysis_typed.h>
#include <cusignal/runtime/host_output_finalize.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace cusignal {

void csd_median_fp32_device(
    const CsdDeviceWorkspace& workspace,
    DeviceArray<ComplexFloat>& output);

namespace {

constexpr float kPi = 3.14159265358979323846F;

template <class T>
float host_load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <class T>
double host_load_double(T value)
{
    if constexpr (std::is_same_v<T, __half>) {
        return static_cast<double>(__half2float(value));
    }
    return static_cast<double>(value);
}

// CPU reference 独立实现周期 Hann 窗，不包含也不调用 GPU kernel helper。
float host_periodic_hann(int index, int length)
{
    return length == 1
        ? 1.0F
        : 0.5F - 0.5F * std::cos(2.0F * kPi * index / length);
}

float host_hann_sum(int length)
{
    float sum = 0.0F;
    for (int index = 0; index < length; ++index) {
        sum += host_periodic_hann(index, length);
    }
    return sum;
}

} // namespace

namespace detail {

template <class T> void lombscargle_fp32_compute_device(
    const DeviceArray<T>& time, const DeviceArray<T>& value,
    const DeviceArray<T>& frequency, DeviceArray<float>& output,
    bool precenter, bool normalize);

template <class T> void vectorstrength_fp32_compute_device(
    const DeviceArray<T>& events, const DeviceArray<T>& periods,
    DeviceArray<float>& strength, DeviceArray<float>& phase);

} // namespace detail

namespace spectral_layout {
namespace {

std::vector<int> normalize_shape(
    std::size_t element_count, const std::vector<int>& requested,
    const char* label)
{
    std::vector<int> shape = requested;
    if (shape.empty()) {
        if (element_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument(std::string(label) + " exceeds int indexing");
        shape = {static_cast<int>(element_count)};
    }
    if (shape.empty()) throw std::invalid_argument(std::string(label) + " rank");
    std::size_t product = 1;
    for (const int extent : shape) {
        if (extent < 0 || (extent != 0
            && product > static_cast<std::size_t>(std::numeric_limits<int>::max())
                / static_cast<std::size_t>(extent)))
            throw std::invalid_argument(std::string(label) + " shape");
        product *= static_cast<std::size_t>(extent);
    }
    if (product != element_count)
        throw std::invalid_argument(std::string(label) + " shape product");
    return shape;
}

int normalize_axis(int axis, int rank, const char* label)
{
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw std::invalid_argument(std::string(label) + " axis");
    return axis;
}

int checked_product(
    const std::vector<int>& shape, std::size_t begin, std::size_t end,
    const char* label)
{
    long long product = 1;
    for (std::size_t index = begin; index < end; ++index) {
        product *= shape[index];
        if (product > std::numeric_limits<int>::max())
            throw std::invalid_argument(std::string(label) + " layout");
    }
    return static_cast<int>(product);
}

std::vector<int> row_major_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    for (int index = static_cast<int>(shape.size()) - 2; index >= 0; --index)
        strides[index] = strides[index + 1] * shape[index + 1];
    return strides;
}

std::vector<int> remove_axis(const std::vector<int>& shape, int axis)
{
    std::vector<int> result;
    result.reserve(shape.size() - 1);
    for (int index = 0; index < static_cast<int>(shape.size()); ++index)
        if (index != axis) result.push_back(shape[index]);
    return result;
}

int broadcast_line_base(
    int output_line, const std::vector<int>& output_outer,
    const std::vector<int>& input_shape, int input_axis)
{
    const auto input_outer = remove_axis(input_shape, input_axis);
    const auto input_strides = row_major_strides(input_shape);
    std::vector<int> output_coordinates(output_outer.size(), 0);
    int remaining = output_line;
    for (int index = static_cast<int>(output_outer.size()) - 1; index >= 0; --index) {
        const int extent = output_outer[index];
        output_coordinates[index] = extent == 0 ? 0 : remaining % extent;
        if (extent != 0) remaining /= extent;
    }
    const int shift = static_cast<int>(output_outer.size() - input_outer.size());
    int base = 0;
    int outer_index = 0;
    for (int dimension = 0; dimension < static_cast<int>(input_shape.size()); ++dimension) {
        if (dimension == input_axis) continue;
        const int coordinate = input_outer[outer_index] == 1
            ? 0 : output_coordinates[outer_index + shift];
        base += coordinate * input_strides[dimension];
        ++outer_index;
    }
    return base;
}

} // namespace

AxisPlan prepare_axis_plan(
    std::size_t element_count, const std::vector<int>& requested_shape,
    int requested_axis, int frequency_count, int time_count,
    bool append_time)
{
    AxisPlan plan;
    plan.input_shape = normalize_shape(element_count, requested_shape, "spectral input");
    plan.axis = normalize_axis(
        requested_axis, static_cast<int>(plan.input_shape.size()), "spectral");
    plan.axis_length = plan.input_shape[plan.axis];
    plan.inner_count = checked_product(
        plan.input_shape, static_cast<std::size_t>(plan.axis + 1),
        plan.input_shape.size(), "spectral");
    plan.axis_stride = plan.inner_count;
    plan.outer_count = checked_product(
        plan.input_shape, 0, static_cast<std::size_t>(plan.axis), "spectral");
    plan.line_count = plan.outer_count * plan.inner_count;
    plan.line_bases.resize(plan.line_count);
    for (int outer = 0; outer < plan.outer_count; ++outer)
        for (int inner = 0; inner < plan.inner_count; ++inner)
            plan.line_bases[outer * plan.inner_count + inner] =
                outer * plan.axis_length * plan.inner_count + inner;
    plan.output_shape = plan.input_shape;
    plan.output_shape[plan.axis] = frequency_count;
    if (append_time) plan.output_shape.push_back(time_count);
    return plan;
}

CsdPlan prepare_csd_plan(
    std::size_t x_element_count, std::size_t y_element_count,
    const std::vector<int>& requested_x_shape,
    const std::vector<int>& requested_y_shape,
    int requested_axis, int frequency_count)
{
    CsdPlan plan;
    plan.x_shape = normalize_shape(x_element_count, requested_x_shape, "csd x");
    plan.y_shape = normalize_shape(y_element_count, requested_y_shape, "csd y");
    const int x_axis = normalize_axis(
        requested_axis, static_cast<int>(plan.x_shape.size()), "csd x");
    const int y_axis = normalize_axis(
        requested_axis, static_cast<int>(plan.y_shape.size()), "csd y");
    plan.x_axis_length = plan.x_shape[x_axis];
    plan.y_axis_length = plan.y_shape[y_axis];
    plan.x_axis_stride = checked_product(
        plan.x_shape, static_cast<std::size_t>(x_axis + 1),
        plan.x_shape.size(), "csd x");
    plan.y_axis_stride = checked_product(
        plan.y_shape, static_cast<std::size_t>(y_axis + 1),
        plan.y_shape.size(), "csd y");
    const auto x_outer = remove_axis(plan.x_shape, x_axis);
    const auto y_outer = remove_axis(plan.y_shape, y_axis);
    const std::size_t outer_rank = std::max(x_outer.size(), y_outer.size());
    std::vector<int> output_outer(outer_rank, 1);
    for (std::size_t offset = 0; offset < outer_rank; ++offset) {
        const int x_index = static_cast<int>(x_outer.size()) - 1 - static_cast<int>(offset);
        const int y_index = static_cast<int>(y_outer.size()) - 1 - static_cast<int>(offset);
        const int x_extent = x_index >= 0 ? x_outer[x_index] : 1;
        const int y_extent = y_index >= 0 ? y_outer[y_index] : 1;
        if (x_extent != y_extent && x_extent != 1 && y_extent != 1)
            throw std::invalid_argument("x and y cannot be broadcast together");
        output_outer[outer_rank - 1 - offset] = std::max(x_extent, y_extent);
    }
    plan.line_count = checked_product(
        output_outer, 0, output_outer.size(), "csd broadcast");
    const int output_rank = static_cast<int>(outer_rank) + 1;
    int output_axis = requested_axis;
    if (output_axis < 0) output_axis += output_rank;
    if (output_axis < 0 || output_axis >= output_rank)
        throw std::invalid_argument("csd output axis");
    plan.output_shape = output_outer;
    plan.output_shape.insert(
        plan.output_shape.begin() + output_axis, frequency_count);
    plan.output_inner_count = checked_product(
        plan.output_shape, static_cast<std::size_t>(output_axis + 1),
        plan.output_shape.size(), "csd output");
    plan.x_line_bases.resize(plan.line_count);
    plan.y_line_bases.resize(plan.line_count);
    for (int line = 0; line < plan.line_count; ++line) {
        plan.x_line_bases[line] = broadcast_line_base(
            line, output_outer, plan.x_shape, x_axis);
        plan.y_line_bases[line] = broadcast_line_base(
            line, output_outer, plan.y_shape, y_axis);
    }
    return plan;
}

} // namespace spectral_layout

template <class T>
std::vector<ComplexFloat> csd_typed_cpu(const std::vector<T>& x, const std::vector<T>& y)
{
    if (x.size() < 2 || x.size() != y.size()) {
        throw std::invalid_argument("csd shape");
    }
    const int length = static_cast<int>(x.size());
    const float window_sum = host_hann_sum(length);
    std::vector<ComplexFloat> out(length);
    for (int frequency = 0; frequency < length; ++frequency) {
        float x_real = 0.0F;
        float x_imag = 0.0F;
        float y_real = 0.0F;
        float y_imag = 0.0F;
        for (int sample = 0; sample < length; ++sample) {
            const float angle = -2.0F * kPi * frequency * sample / length;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float window = host_periodic_hann(sample, length);
            x_real += host_load(x[sample]) * window * cosine;
            x_imag += host_load(x[sample]) * window * sine;
            y_real += host_load(y[sample]) * window * cosine;
            y_imag += host_load(y[sample]) * window * sine;
        }
        out[frequency] = {
            (x_real * y_real + x_imag * y_imag) / (window_sum * window_sum),
            (x_real * y_imag - x_imag * y_real) / (window_sum * window_sum)};
    }
    return out;
}

template <class T>
std::vector<ComplexFloat> stft_typed_cpu(const std::vector<T>& x, int frame, int hop)
{
    if (frame < 2 || hop < 1 || hop > frame) {
        throw std::invalid_argument("stft shape");
    }
    const int frames = x.size() < static_cast<std::size_t>(frame)
        ? 0
        : 1 + (static_cast<int>(x.size()) - frame) / hop;
    const float window_sum = host_hann_sum(frame);
    std::vector<ComplexFloat> out(frames * frame);
    for (int frequency = 0; frequency < frame; ++frequency) {
        for (int frame_index = 0; frame_index < frames; ++frame_index) {
            float real = 0.0F;
            float imag = 0.0F;
            for (int sample = 0; sample < frame; ++sample) {
                const int input_index = frame_index * hop + sample;
                if (input_index < static_cast<int>(x.size())) {
                    const float angle = -2.0F * kPi * frequency * sample / frame;
                    const float value = host_load(x[input_index])
                        * host_periodic_hann(sample, frame) / window_sum;
                    real += value * std::cos(angle);
                    imag += value * std::sin(angle);
                }
            }
            out[frequency * frames + frame_index] = {real, imag};
        }
    }
    return out;
}

template <class T>
std::vector<float> istft_typed_cpu(
    const std::vector<TypedComplex<T>>& z, int frames, int bins, int hop)
{
    if (frames < 1 || bins < 2 || hop < 1 || hop > bins
        || z.size() != static_cast<std::size_t>(frames * bins)) {
        throw std::invalid_argument("istft shape");
    }
    const int length = (frames - 1) * hop + bins;
    const float window_sum = host_hann_sum(bins);
    std::vector<float> out(length);
    for (int output_index = 0; output_index < length; ++output_index) {
        float sum = 0.0F;
        float normalization = 0.0F;
        for (int frame_index = 0; frame_index < frames; ++frame_index) {
            const int local = output_index - frame_index * hop;
            if (local >= 0 && local < bins) {
                float segment = 0.0F;
                for (int frequency = 0; frequency < bins; ++frequency) {
                    const float angle = 2.0F * kPi * frequency * local / bins;
                    const auto& value = z[frequency * frames + frame_index];
                    segment += host_load(value.real) * std::cos(angle)
                        - host_load(value.imag) * std::sin(angle);
                }
                const float window = host_periodic_hann(local, bins);
                sum += segment / bins * window_sum * window;
                normalization += window * window;
            }
        }
        out[output_index] = normalization > 1.0e-7F ? sum / normalization : 0.0F;
    }
    return out;
}

template <class T>
std::vector<float> spectrogram_typed_cpu(const std::vector<T>& x, int frame, int hop)
{
    if (frame < 2 || hop < 1 || hop > frame) {
        throw std::invalid_argument("spectrogram shape");
    }
    const int frames = x.size() < static_cast<std::size_t>(frame)
        ? 0
        : 1 + (static_cast<int>(x.size()) - frame) / hop;
    const float window_sum = host_hann_sum(frame);
    std::vector<float> out(frames * frame);
    for (int frequency = 0; frequency < frame; ++frequency) {
        for (int frame_index = 0; frame_index < frames; ++frame_index) {
            float real = 0.0F;
            float imag = 0.0F;
            for (int sample = 0; sample < frame; ++sample) {
                const int input_index = frame_index * hop + sample;
                if (input_index < static_cast<int>(x.size())) {
                    const float angle = -2.0F * kPi * frequency * sample / frame;
                    const float value = host_load(x[input_index])
                        * host_periodic_hann(sample, frame);
                    real += value * std::cos(angle);
                    imag += value * std::sin(angle);
                }
            }
            out[frequency * frames + frame_index]
                = (real * real + imag * imag) / (window_sum * window_sum);
        }
    }
    return out;
}

template <class T>
std::vector<double> lombscargle_typed_cpu(
    const std::vector<T>& time,
    const std::vector<T>& y,
    const std::vector<T>& frequency,
    bool precenter,
    bool normalize)
{
    if (time.empty() || time.size() != y.size()) {
        throw std::invalid_argument("lombscargle: time and signal must be nonempty with equal lengths");
    }
    const int length = static_cast<int>(time.size());
    double signal_mean = 0.0;
    if (precenter) {
        for (const T value : y) {
            signal_mean += host_load_double(value);
        }
        signal_mean /= length;
    }
    double y_dot = 0.0;
    if (normalize) {
        for (const T value : y) {
            const double original = host_load_double(value);
            y_dot += original * original;
        }
    }
    const double normalization = normalize && y_dot != 0.0 ? 2.0 / y_dot : 1.0;
    std::vector<double> out(frequency.size());
    for (int frequency_index = 0;
         frequency_index < static_cast<int>(frequency.size());
         ++frequency_index) {
        const double omega = host_load_double(frequency[frequency_index]);
        double xc = 0.0;
        double xs = 0.0;
        double cc = 0.0;
        double ss = 0.0;
        double cs = 0.0;
        for (int index = 0; index < length; ++index) {
            const double angle = omega * host_load_double(time[index]);
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const double value = host_load_double(y[index]) - signal_mean;
            xc += value * cosine;
            xs += value * sine;
            cc += cosine * cosine;
            ss += sine * sine;
            cs += cosine * sine;
        }
        const double tau = std::atan2(2.0 * cs, cc - ss) / (2.0 * omega);
        const double cosine_tau = std::cos(omega * tau);
        const double sine_tau = std::sin(omega * tau);
        const double first = cosine_tau * xc + sine_tau * xs;
        const double second = cosine_tau * xs - sine_tau * xc;
        const double first_denominator = cosine_tau * cosine_tau * cc
            + 2.0 * cosine_tau * sine_tau * cs + sine_tau * sine_tau * ss;
        const double second_denominator = cosine_tau * cosine_tau * ss
            - 2.0 * cosine_tau * sine_tau * cs + sine_tau * sine_tau * cc;
        out[frequency_index] = 0.5
            * (first * first / first_denominator
               + second * second / second_denominator)
            * normalization;
    }
    return out;
}

template <class T>
VectorStrengthCpuResult vectorstrength_typed_cpu(
    const std::vector<T>& events, const std::vector<T>& periods)
{
    VectorStrengthCpuResult result;
    result.strength.resize(periods.size());
    result.phase.resize(periods.size());
    constexpr double pi = 3.14159265358979323846;
    for (std::size_t period_index = 0; period_index < periods.size(); ++period_index) {
        const double period = host_load_double(periods[period_index]);
        if (period <= 0.0) {
            throw std::invalid_argument("periods must be positive");
        }
        if (events.empty()) {
            result.strength[period_index] = std::numeric_limits<double>::quiet_NaN();
            result.phase[period_index] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        double real = 0.0;
        double imag = 0.0;
        for (const auto& event : events) {
            const double angle = 2.0 * pi * host_load_double(event) / period;
            real += std::cos(angle);
            imag += std::sin(angle);
        }
        real /= static_cast<double>(events.size());
        imag /= static_cast<double>(events.size());
        result.strength[period_index] = std::sqrt(real * real + imag * imag);
        result.phase[period_index] = std::atan2(imag, real);
    }
    return result;
}

template <class T>
VectorStrengthCpuResult vectorstrength_typed_cpu(const std::vector<T>& events, T period)
{
    auto result = vectorstrength_typed_cpu(events, std::vector<T>{period});
    result.scalar_period = true;
    return result;
}

template <class T>
DeviceArray<double> lombscargle_device(
    const DeviceArray<T>& time, const DeviceArray<T>& value,
    const DeviceArray<T>& frequency, bool precenter, bool normalize)
{
    if (time.empty() || time.size() != value.size()) {
        throw std::invalid_argument("lombscargle_device: input and output shapes are inconsistent");
    }
    DeviceArray<float> computed(frequency.size());
    detail::lombscargle_fp32_compute_device(
        time, value, frequency, computed, precenter, normalize);
    return cuda_utils::finalize_fp64_device_storage(computed);
}

template <class T>
VectorStrengthDeviceResult vectorstrength_device(
    const DeviceArray<T>& events, const DeviceArray<T>& periods)
{
    for (const T period : periods.to_host()) {
        if (host_load_double(period) <= 0.0) {
            throw std::invalid_argument("periods must be positive");
        }
    }
    DeviceArray<float> strength(periods.size());
    DeviceArray<float> phase(periods.size());
    detail::vectorstrength_fp32_compute_device(events, periods, strength, phase);
    VectorStrengthDeviceResult result;
    result.strength = cuda_utils::finalize_fp64_device_storage(strength);
    result.phase = cuda_utils::finalize_fp64_device_storage(phase);
    return result;
}

template <class T>
VectorStrengthDeviceResult vectorstrength_device(const DeviceArray<T>& events, T period)
{
    if (host_load_double(period) <= 0.0) {
        throw std::invalid_argument("periods must be positive");
    }
    const DeviceArray<T> periods = DeviceArray<T>::from_host({period});
    auto result = vectorstrength_device(events, periods);
    result.scalar_period = true;
    return result;
}

namespace {

std::vector<double> spectral_frequencies(int nfft, int count, double fs)
{
    std::vector<double> frequencies(count);
    for (int index = 0; index < count; ++index) {
        const int signed_bin = count == nfft && index > (nfft - 1) / 2
            ? index - nfft
            : index;
        frequencies[index] = static_cast<double>(signed_bin) * fs / nfft;
    }
    return frequencies;
}

std::vector<double> spectral_times(
    int count, int nperseg, int noverlap, double fs,
    const std::string& boundary)
{
    std::vector<double> times(count);
    const double offset = (boundary.empty() || boundary == "None")
        ? static_cast<double>(nperseg) / 2.0
        : 0.0;
    const int step = nperseg - noverlap;
    for (int index = 0; index < count; ++index) {
        times[index] = (offset + static_cast<double>(index * step)) / fs;
    }
    return times;
}

double host_bessel_i0(double x)
{
    double sum = 1.0;
    double term = 1.0;
    const double quarter_x_squared = 0.25 * x * x;
    for (int order = 1; order < 32; ++order) {
        term *= quarter_x_squared / (order * order);
        sum += term;
        if (term < sum * 1.0e-15) break;
    }
    return sum;
}

std::vector<double> host_spectral_window(const WindowParams& params, int size)
{
    if (!params.custom_window.empty()) {
        if (static_cast<int>(params.custom_window.size()) != size) {
            throw std::invalid_argument("window length must equal nperseg");
        }
        return {params.custom_window.begin(), params.custom_window.end()};
    }
    constexpr double pi = 3.14159265358979323846;
    const std::string type = params.type.empty() ? "hann" : params.type;
    std::vector<double> window(size, 1.0);
    for (int index = 0; index < size; ++index) {
        const double angle = 2.0 * pi * index / size;
        if (type == "hann" || type == "hanning") {
            window[index] = 0.5 - 0.5 * std::cos(angle);
        } else if (type == "hamming") {
            window[index] = 0.54 - 0.46 * std::cos(angle);
        } else if (type == "blackman") {
            window[index] = 0.42 - 0.5 * std::cos(angle)
                + 0.08 * std::cos(2.0 * angle);
        } else if (type == "tukey") {
            const double alpha = params.param;
            const double edge = alpha * size / 2.0;
            if (alpha <= 0.0) {
                window[index] = 1.0;
            } else if (alpha >= 1.0) {
                window[index] = 0.5 - 0.5 * std::cos(angle);
            } else if (index < edge) {
                window[index] = 0.5 * (1.0 + std::cos(
                    pi * (-1.0 + 2.0 * index / (alpha * size))));
            } else if (index > size - edge) {
                window[index] = 0.5 * (1.0 + std::cos(pi * (
                    -2.0 / alpha + 1.0 + 2.0 * index / (alpha * size))));
            }
        } else if (type == "boxcar" || type == "rectangular") {
            window[index] = 1.0;
        } else if (type == "bartlett" || type == "triang") {
            const double fraction = static_cast<double>(index) / size;
            window[index] = 2.0 * std::min(fraction, 1.0 - fraction);
        } else if (type == "kaiser") {
            // CPU reference keeps the same FP32 coefficient generator as the GPU workspace.
            const double beta = params.param > 0.0F ? params.param : 5.0F;
            const double ratio = size == 1 ? 0.0
                : (2.0 * index / static_cast<double>(size - 1) - 1.0);
            window[index] = host_bessel_i0(beta * std::sqrt(1.0 - ratio * ratio))
                / host_bessel_i0(beta);
        } else {
            throw std::invalid_argument("unsupported window type");
        }
    }
    return window;
}

template <class T>
double host_boundary_sample(
    const std::vector<T>& input, int index, const std::string& boundary)
{
    const int length = static_cast<int>(input.size());
    if (index >= 0 && index < length) {
        return host_load_double(input[index]);
    }
    if (boundary.empty() || boundary == "None" || boundary == "zeros") {
        return 0.0;
    }
    if (boundary == "constant") {
        return host_load_double(input[index < 0 ? 0 : length - 1]);
    }
    const int reflected = index < 0 ? -index : 2 * length - 2 - index;
    const double reflected_value = host_load_double(input[reflected]);
    if (boundary == "even") {
        return reflected_value;
    }
    if (boundary == "odd") {
        const double edge = host_load_double(input[index < 0 ? 0 : length - 1]);
        return 2.0 * edge - reflected_value;
    }
    throw std::invalid_argument("unsupported boundary");
}

template <class T>
double host_axis_boundary_sample(
    const std::vector<T>& input, int line_base, int axis_stride,
    int axis_length, int index, const std::string& boundary)
{
    if (index >= 0 && index < axis_length)
        return host_load_double(input[line_base + index * axis_stride]);
    if (boundary.empty() || boundary == "None" || boundary == "zeros")
        return 0.0;
    if (boundary == "constant") {
        const int sample = index < 0 ? 0 : axis_length - 1;
        return host_load_double(input[line_base + sample * axis_stride]);
    }
    const int reflected = index < 0 ? -index : 2 * axis_length - 2 - index;
    const double reflected_value = host_load_double(
        input[line_base + reflected * axis_stride]);
    if (boundary == "even") return reflected_value;
    if (boundary == "odd") {
        const int edge_sample = index < 0 ? 0 : axis_length - 1;
        const double edge = host_load_double(
            input[line_base + edge_sample * axis_stride]);
        return 2.0 * edge - reflected_value;
    }
    throw std::invalid_argument("unsupported boundary");
}

template <class T>
constexpr bool spectral_requires_fp64_v = std::is_same_v<T, std::int32_t>;

template <class T>
void append_spectral_complex(
    SpectralComplexCpuResult& result, double real, double imag)
{
    if constexpr (spectral_requires_fp64_v<T>) {
        result.complex_fp64.push_back({real, imag});
    } else {
        result.complex_fp32.push_back(
            {static_cast<float>(real), static_cast<float>(imag)});
    }
}

// CPU reference only. The formal GPU median path uses the Device quickselect
// implementation and must not call either helper.
float spectral_median(std::vector<float>& values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5F * (values[middle - 1] + values[middle])
        : values[middle];
}

float spectral_median_bias(int count)
{
    float bias = 1.0F;
    for (int index = 1; index <= (count - 1) / 2; ++index) {
        const float even = static_cast<float>(2 * index);
        bias += 1.0F / (even + 1.0F) - 1.0F / even;
    }
    return bias;
}

template <class T>
void set_spectral_complex_output(
    DeviceArray<ComplexFloat>& computed,
    SpectralComplexDeviceResult& result)
{
    if constexpr (spectral_requires_fp64_v<T>) {
        result.dtype = SpectralComplexDtype::complex_fp64;
        result.complex_fp64 =
            cuda_utils::finalize_complex_fp64_device_storage(computed);
    } else {
        result.dtype = SpectralComplexDtype::complex_fp32;
        result.complex_fp32 = std::move(computed);
    }
}

} // namespace

template <class T>
SpectralComplexCpuResult stft_typed_cpu(
    const std::vector<T>& x, const StftDeviceParams& requested_params)
{
    SpectralComplexCpuResult result;
    if (x.empty()) {
        result.dtype = SpectralComplexDtype::fp64;
        return result;
    }
    result.dtype = spectral_requires_fp64_v<T>
        ? SpectralComplexDtype::complex_fp64
        : SpectralComplexDtype::complex_fp32;
    StftDeviceParams params = requested_params;
    const auto input_plan = spectral_layout::prepare_axis_plan(
        x.size(), params.shape, params.axis, 1, 0, false);
    const int input_axis_length = input_plan.axis_length;
    int nperseg = params.nperseg <= 0
        ? (params.window.custom_window.empty()
            ? 256 : static_cast<int>(params.window.custom_window.size()))
        : params.nperseg;
    if (!params.window.custom_window.empty()
        && (params.window.custom_window.size()
                > static_cast<std::size_t>(input_axis_length)
            || static_cast<int>(params.window.custom_window.size()) != nperseg))
        throw std::invalid_argument("window is longer than input or differs from nperseg");
    if (params.window.custom_window.empty() && nperseg > input_axis_length) {
        nperseg = input_axis_length;
    }
    const int noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    const int nfft = params.nfft <= 0 ? nperseg : params.nfft;
    if (params.fs <= 0.0F || nfft < nperseg || noverlap < 0 || noverlap >= nperseg
        || (params.detrend != "constant" && !params.detrend.empty())) {
        throw std::invalid_argument("stft parameters");
    }
    const int step = nperseg - noverlap;
    const bool extended = !params.boundary.empty() && params.boundary != "None";
    int effective_size = input_axis_length + (extended ? nperseg : 0);
    if (params.padded) {
        const int remainder = (effective_size - nperseg) % step;
        if (remainder != 0) {
            effective_size += step - remainder;
        }
    }
    const int frames = effective_size < nperseg
        ? 0 : 1 + (effective_size - nperseg) / step;
    const int nf = params.return_onesided ? nfft / 2 + 1 : nfft;
    const auto window = host_spectral_window(params.window, nperseg);
    double window_sum = 0.0;
    for (double value : window) window_sum += value;
    result.frequency_count = nf;
    result.time_count = frames;
    const auto layout = spectral_layout::prepare_axis_plan(
        x.size(), params.shape, params.axis, nf, frames, true);
    result.shape = layout.output_shape;
    result.frequencies = spectral_frequencies(nfft, nf, params.fs);
    result.times = spectral_times(frames, nperseg, noverlap, params.fs, params.boundary);
    const std::size_t output_count = static_cast<std::size_t>(layout.line_count)
        * static_cast<std::size_t>(nf) * static_cast<std::size_t>(frames);
    if constexpr (spectral_requires_fp64_v<T>) result.complex_fp64.reserve(output_count);
    else result.complex_fp32.reserve(output_count);
    constexpr double pi = 3.14159265358979323846;
    for (int outer = 0; outer < layout.outer_count; ++outer) {
      for (int frequency = 0; frequency < nf; ++frequency) {
       for (int inner = 0; inner < layout.inner_count; ++inner) {
        const int line = outer * layout.inner_count + inner;
        const int line_base = layout.line_bases[line];
        for (int frame = 0; frame < frames; ++frame) {
            const int start = frame * step - (extended ? nperseg / 2 : 0);
            double mean = 0.0;
            if (params.detrend == "constant") {
                for (int sample = 0; sample < nperseg; ++sample)
                    mean += host_axis_boundary_sample(
                        x, line_base, layout.axis_stride,
                        layout.axis_length, start + sample, params.boundary);
                mean /= nperseg;
            }
            double real = 0.0, imag = 0.0;
            for (int sample = 0; sample < nperseg; ++sample) {
                const double value = (host_axis_boundary_sample(
                    x, line_base, layout.axis_stride, layout.axis_length,
                    start + sample, params.boundary) - mean)
                    * window[sample] / window_sum;
                const double angle = -2.0 * pi * frequency * sample / nfft;
                real += value * std::cos(angle);
                imag += value * std::sin(angle);
            }
            append_spectral_complex<T>(result, real, imag);
        }
       }
      }
    }
    return result;
}

template <class T>
SpectralComplexCpuResult csd_typed_cpu(
    const std::vector<T>& x_input, const std::vector<T>& y_input,
    const CsdDeviceParams& params)
{
    SpectralComplexCpuResult result;
    if (x_input.empty() || y_input.empty()) {
        result.dtype = SpectralComplexDtype::fp64;
        return result;
    }
    result.dtype = spectral_requires_fp64_v<T>
        ? SpectralComplexDtype::complex_fp64
        : SpectralComplexDtype::complex_fp32;
    if (params.average != "mean" && params.average != "median")
        throw std::invalid_argument("average must be mean or median");
    const auto input_layout = spectral_layout::prepare_csd_plan(
        x_input.size(), y_input.size(), params.x_shape, params.y_shape,
        params.axis, 1);
    const int spectral_length = std::max(
        input_layout.x_axis_length, input_layout.y_axis_length);
    int nperseg = params.nperseg <= 0 ? 256 : params.nperseg;
    if (!params.window.custom_window.empty()) {
        if (params.nperseg <= 0) nperseg = static_cast<int>(params.window.custom_window.size());
        if (params.window.custom_window.size()
                > static_cast<std::size_t>(spectral_length)
            || static_cast<int>(params.window.custom_window.size()) != nperseg)
            throw std::invalid_argument("window is longer than input or differs from nperseg");
    }
    if (params.window.custom_window.empty() && nperseg > spectral_length)
        nperseg = spectral_length;
    const int noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    const int nfft = params.nfft <= 0 ? nperseg : params.nfft;
    if (params.fs <= 0.0F || nfft < nperseg || noverlap < 0 || noverlap >= nperseg)
        throw std::invalid_argument("csd parameters");
    const int frames = 1 + (spectral_length - nperseg) / (nperseg - noverlap);
    const int nf = params.return_onesided ? nfft / 2 + 1 : nfft;
    const auto layout = spectral_layout::prepare_csd_plan(
        x_input.size(), y_input.size(), params.x_shape, params.y_shape,
        params.axis, nf);
    const auto window = host_spectral_window(params.window, nperseg);
    double sum = 0.0, square_sum = 0.0;
    for (double value : window) { sum += value; square_sum += value * value; }
    const double scale = params.scaling == "density"
        ? 1.0 / (params.fs * square_sum)
        : params.scaling == "spectrum" ? 1.0 / (sum * sum)
        : throw std::invalid_argument("unsupported scaling");
    result.frequency_count = nf; result.time_count = 0;
    result.shape = layout.output_shape;
    result.frequencies = spectral_frequencies(nfft, nf, params.fs);
    constexpr double pi = 3.14159265358979323846;
    const int output_outer_count = layout.line_count / layout.output_inner_count;
    for (int outer = 0; outer < output_outer_count; ++outer) {
     for (int frequency = 0; frequency < nf; ++frequency) {
      for (int inner = 0; inner < layout.output_inner_count; ++inner) {
       const int line = outer * layout.output_inner_count + inner;
        std::vector<double> real_values(frames), imag_values(frames);
        for (int frame = 0; frame < frames; ++frame) {
            double xr=0,xi=0,yr=0,yi=0,xmean=0,ymean=0;
            const int start = frame * (nperseg - noverlap);
            if (params.detrend == "constant") {
                for (int sample=0;sample<nperseg;++sample){const int position=start+sample;if(position<input_layout.x_axis_length)xmean+=host_load_double(x_input[layout.x_line_bases[line]+position*input_layout.x_axis_stride]);if(position<input_layout.y_axis_length)ymean+=host_load_double(y_input[layout.y_line_bases[line]+position*input_layout.y_axis_stride]);}
                xmean/=nperseg;ymean/=nperseg;
            } else if (!params.detrend.empty()) throw std::invalid_argument("unsupported detrend");
            for (int sample=0;sample<nperseg;++sample){const int position=start+sample;const double angle=-2.0*pi*frequency*sample/nfft;const double c=std::cos(angle),s=std::sin(angle);const double xraw=position<input_layout.x_axis_length?host_load_double(x_input[layout.x_line_bases[line]+position*input_layout.x_axis_stride]):0.0;const double yraw=position<input_layout.y_axis_length?host_load_double(y_input[layout.y_line_bases[line]+position*input_layout.y_axis_stride]):0.0;const double xv=(xraw-xmean)*window[sample];const double yv=(yraw-ymean)*window[sample];xr+=xv*c;xi+=xv*s;yr+=yv*c;yi+=yv*s;}
            double bin_scale=scale;if(params.return_onesided&&((nfft%2==0)?(frequency>0&&frequency<nf-1):(frequency>0)))bin_scale*=2.0;
            real_values[frame]=(xr*yr+xi*yi)*bin_scale;imag_values[frame]=(xr*yi-xi*yr)*bin_scale;
        }
        double real=0,imag=0;
        if(params.average=="mean"){for(int frame=0;frame<frames;++frame){real+=real_values[frame];imag+=imag_values[frame];}real/=frames;imag/=frames;}
        else{std::vector<float> rf(real_values.begin(),real_values.end()),imf(imag_values.begin(),imag_values.end());const float bias=spectral_median_bias(frames);real=spectral_median(rf)/bias;imag=spectral_median(imf)/bias;}
        append_spectral_complex<T>(result,real,imag);
      }
     }
    }
    return result;
}

template <class T>
SpectrogramCpuResult spectrogram_typed_cpu(
    const std::vector<T>& x, const SpectrogramDeviceParams& params)
{
    SpectrogramCpuResult result;
    if (x.empty()) {
        result.dtype = SpectrogramOutputDtype::fp64;
        return result;
    }
    const auto input_layout = spectral_layout::prepare_axis_plan(
        x.size(), params.shape, params.axis, 1, 0, false);
    int actual_nperseg = params.nperseg <= 0
        ? (params.window.custom_window.empty()
            ? 256 : static_cast<int>(params.window.custom_window.size()))
        : params.nperseg;
    if (!params.window.custom_window.empty()
        && (params.window.custom_window.size()
                > static_cast<std::size_t>(input_layout.axis_length)
            || static_cast<int>(params.window.custom_window.size()) != actual_nperseg))
        throw std::invalid_argument("window is longer than input or differs from nperseg");
    if (params.window.custom_window.empty()
        && actual_nperseg > input_layout.axis_length)
        actual_nperseg = input_layout.axis_length;
    const int actual_nfft = params.nfft <= 0 ? actual_nperseg : params.nfft;
    StftDeviceParams stft_params;
    stft_params.fs = params.fs;
    stft_params.window = params.window;
    stft_params.nperseg = actual_nperseg;
    stft_params.noverlap = params.noverlap < 0
        ? actual_nperseg / 8
        : params.noverlap;
    stft_params.nfft = actual_nfft;
    stft_params.detrend = params.detrend;
    stft_params.return_onesided = params.return_onesided;
    stft_params.boundary = "";
    stft_params.padded = false;
    stft_params.axis = params.axis;
    stft_params.shape = params.shape;
    auto stft = stft_typed_cpu(x, stft_params);
    result.frequencies = stft.frequencies;
    result.times = stft.times;
    result.frequency_count = stft.frequency_count;
    result.time_count = stft.time_count;
    result.shape = stft.shape;
    const auto window = host_spectral_window(params.window, actual_nperseg);
    double sum = 0.0, square_sum = 0.0;
    for (double value : window) { sum += value; square_sum += value * value; }
    const double density_to_spectrum = sum * sum / (params.fs * square_sum);
    const double complex_factor = params.scaling == "density"
        ? std::sqrt(density_to_spectrum)
        : params.scaling == "spectrum" ? 1.0
        : throw std::invalid_argument("unsupported scaling");
    const bool complex_mode = params.mode == "complex";
    if (!complex_mode && params.mode != "psd" && params.mode != "magnitude"
        && params.mode != "angle" && params.mode != "phase")
        throw std::invalid_argument("unsupported spectrogram mode");
    const std::size_t count = spectral_requires_fp64_v<T>
        ? stft.complex_fp64.size() : stft.complex_fp32.size();
    std::vector<double> real(count), imag(count);
    for (std::size_t index = 0; index < count; ++index) {
        if constexpr (spectral_requires_fp64_v<T>) {
            real[index] = stft.complex_fp64[index].re * complex_factor;
            imag[index] = stft.complex_fp64[index].im * complex_factor;
        } else {
            real[index] = stft.complex_fp32[index].re * complex_factor;
            imag[index] = stft.complex_fp32[index].im * complex_factor;
        }
    }
    if (complex_mode) {
        result.dtype = spectral_requires_fp64_v<T>
            ? SpectrogramOutputDtype::complex_fp64
            : SpectrogramOutputDtype::complex_fp32;
        for (std::size_t index = 0; index < count; ++index) {
            if constexpr (spectral_requires_fp64_v<T>)
                result.complex_fp64.push_back({real[index], imag[index]});
            else result.complex_fp32.push_back(
                {static_cast<float>(real[index]), static_cast<float>(imag[index])});
        }
        return result;
    }
    std::vector<double> values(count);
    const auto output_layout = spectral_layout::prepare_axis_plan(
        x.size(), params.shape, params.axis, result.frequency_count,
        result.time_count, true);
    for (int outer = 0; outer < output_layout.outer_count; ++outer) {
     for (int frequency = 0; frequency < result.frequency_count; ++frequency) {
        const bool double_bin = params.return_onesided
            && ((actual_nfft % 2 == 0)
                ? frequency > 0 && frequency < result.frequency_count - 1
                : frequency > 0);
        for (int inner = 0; inner < output_layout.inner_count; ++inner) {
         for (int time = 0; time < result.time_count; ++time) {
            const int index = ((outer * result.frequency_count + frequency)
                * output_layout.inner_count + inner)
                * result.time_count + time;
            if (params.mode == "psd") {
                values[index] = (real[index] * real[index] + imag[index] * imag[index])
                    * (double_bin ? 2.0 : 1.0);
            } else if (params.mode == "magnitude") {
                values[index] = std::sqrt(real[index] * real[index] + imag[index] * imag[index]);
            } else {
                double angle_imag = imag[index];
                const double roundoff_limit = 64.0
                    * std::numeric_limits<double>::epsilon() * std::fabs(real[index]);
                if (real[index] != 0.0 && std::fabs(angle_imag) <= roundoff_limit) {
                    angle_imag = 0.0;
                }
                values[index] = std::atan2(angle_imag, real[index]);
            }
        }
        }
     }
    }
    if (params.mode == "phase") {
        constexpr double two_pi = 6.28318530717958647692;
        constexpr double pi = 3.14159265358979323846;
        for (int outer = 0; outer < output_layout.outer_count; ++outer) {
         for (int inner = 0; inner < output_layout.inner_count; ++inner) {
          for (int time = 0; time < result.time_count; ++time) {
            double correction = 0.0;
            const int first = (outer * result.frequency_count
                * output_layout.inner_count + inner) * result.time_count + time;
            double previous_raw = values[first];
            for (int frequency = 1; frequency < result.frequency_count; ++frequency) {
                const int index = ((outer * result.frequency_count + frequency)
                    * output_layout.inner_count + inner)
                    * result.time_count + time;
                const double current_raw = values[index];
                const double delta = current_raw - previous_raw;
                if (delta > pi) correction -= two_pi;
                else if (delta < -pi) correction += two_pi;
                values[index] = current_raw + correction;
                previous_raw = current_raw;
            }
          }
         }
        }
    }
    if constexpr (spectral_requires_fp64_v<T>) {
        result.dtype = SpectrogramOutputDtype::fp64;
        result.fp64 = std::move(values);
    } else {
        result.dtype = SpectrogramOutputDtype::fp32;
        result.fp32.assign(values.begin(), values.end());
    }
    return result;
}

template <class T>
IstftCpuResult istft_typed_cpu(
    const std::vector<TypedComplex<T>>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params)
{
    if (frequency_count < 1 || time_count < 1
        || z.size() != static_cast<std::size_t>(frequency_count * time_count)
        || params.fs <= 0.0F) throw std::invalid_argument("istft shape");
    const int n_default = params.input_onesided ? 2 * (frequency_count - 1) : frequency_count;
    const int nperseg = params.nperseg <= 0 ? n_default : params.nperseg;
    const int nfft = params.nfft <= 0
        ? (params.input_onesided && nperseg == n_default + 1 ? nperseg : n_default)
        : params.nfft;
    const int noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    if (nperseg < 1 || nfft < nperseg || noverlap < 0 || noverlap >= nperseg
        || frequency_count != (params.input_onesided ? nfft / 2 + 1 : nfft))
        throw std::invalid_argument("istft parameters");
    const int step = nperseg - noverlap;
    const int full_length = nperseg + (time_count - 1) * step;
    const int trim = params.boundary ? nperseg / 2 : 0;
    const int final_length = std::max(0, full_length - 2 * trim);
    const auto window = host_spectral_window(params.window, nperseg);
    double window_sum = 0.0; for (double value : window) window_sum += value;
    std::vector<double> reconstructed(full_length, 0.0), normalization(full_length, 0.0);
    constexpr double pi = 3.14159265358979323846;
    for (int frame = 0; frame < time_count; ++frame) {
        for (int sample = 0; sample < nperseg; ++sample) {
            double segment = 0.0;
            for (int frequency = 0; frequency < nfft; ++frequency) {
                double real = 0.0, imag = 0.0;
                if (frequency < frequency_count) {
                    const auto& value = z[frequency * time_count + frame];
                    real = host_load_double(value.real); imag = host_load_double(value.imag);
                } else if (params.input_onesided) {
                    const int source = nfft - frequency;
                    if (source > 0 && source < frequency_count) {
                        const auto& value = z[source * time_count + frame];
                        real = host_load_double(value.real); imag = -host_load_double(value.imag);
                    }
                }
                const double angle = 2.0 * pi * frequency * sample / nfft;
                segment += real * std::cos(angle) - imag * std::sin(angle);
            }
            segment /= nfft;
            const int output = frame * step + sample;
            reconstructed[output] += segment * window_sum * window[sample];
            normalization[output] += window[sample] * window[sample];
        }
    }
    IstftCpuResult result;
    result.is_fp64 = spectral_requires_fp64_v<T>;
    result.times.resize(final_length);
    for (int index = 0; index < final_length; ++index) {
        result.times[index] = static_cast<double>(index) / params.fs;
        const int source = index + trim;
        const double value = normalization[source] > 1.0e-10
            ? reconstructed[source] / normalization[source] : 0.0;
        if constexpr (spectral_requires_fp64_v<T>) result.fp64.push_back(value);
        else result.fp32.push_back(static_cast<float>(value));
    }
    return result;
}

template <class T>
IstftCpuResult istft_typed_cpu(
    const std::vector<T>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params)
{
    std::vector<TypedComplex<T>> complex(z.size());
    for (std::size_t index = 0; index < z.size(); ++index)
        complex[index] = {z[index], T{}};
    return istft_typed_cpu(complex, frequency_count, time_count, params);
}

template <class T>
SpectralComplexDeviceResult csd_device(
    const DeviceArray<T>& x, const DeviceArray<T>& y,
    const CsdDeviceParams& requested_params)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (x.empty() || y.empty()) {
        SpectralComplexDeviceResult result;
        result.dtype = SpectralComplexDtype::fp64;
        return result;
    }
    CsdDeviceParams params = requested_params;
    if (params.average != "mean" && params.average != "median") {
        throw std::invalid_argument("average must be mean or median");
    }
    const auto input_layout = spectral_layout::prepare_csd_plan(
        x.size(), y.size(), params.x_shape, params.y_shape,
        params.axis, 1);
    const int spectral_length = std::max(
        input_layout.x_axis_length, input_layout.y_axis_length);
    if (params.nperseg <= 0) {
        params.nperseg = params.window.custom_window.empty()
            ? 256
            : static_cast<int>(params.window.custom_window.size());
    }
    if (!params.window.custom_window.empty()
        && (params.window.custom_window.size()
                > static_cast<std::size_t>(spectral_length)
            || static_cast<int>(params.window.custom_window.size()) != params.nperseg))
        throw std::invalid_argument("window is longer than input or differs from nperseg");
    if (params.window.custom_window.empty()
        && params.nperseg > spectral_length) {
        params.nperseg = spectral_length;
    }

    CsdDeviceParams compute_params = params;
    compute_params.average = "mean";
    CsdDeviceWorkspace workspace(
        static_cast<int>(x.size()), static_cast<int>(y.size()),
        compute_params);
    DeviceArray<ComplexFloat> computed(workspace.output_size());
    csd_device(x, y, computed, workspace, compute_params);
    if (params.average == "median" && workspace.nframes > 0) {
        csd_median_fp32_device(workspace, computed);
    }

    SpectralComplexDeviceResult result;
    result.frequency_count = workspace.nf;
    result.time_count = 0;
    result.shape = workspace.output_shape;
    result.frequencies = DeviceArray<double>::from_host(
        spectral_frequencies(workspace.nfft, workspace.nf, params.fs));
    set_spectral_complex_output<T>(computed, result);
    return result;
}

template <class T>
SpectralComplexDeviceResult stft_device(
    const DeviceArray<T>& x, const StftDeviceParams& requested_params)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (x.empty()) {
        SpectralComplexDeviceResult result;
        result.dtype = SpectralComplexDtype::fp64;
        return result;
    }
    StftDeviceParams params = requested_params;
    const auto input_layout = spectral_layout::prepare_axis_plan(
        x.size(), params.shape, params.axis, 1, 0, false);
    if (params.nperseg <= 0) {
        params.nperseg = params.window.custom_window.empty()
            ? 256
            : static_cast<int>(params.window.custom_window.size());
    }
    if (!params.window.custom_window.empty()
        && (params.window.custom_window.size()
                > static_cast<std::size_t>(input_layout.axis_length)
            || static_cast<int>(params.window.custom_window.size()) != params.nperseg))
        throw std::invalid_argument("window is longer than input or differs from nperseg");
    if (params.window.custom_window.empty()
        && params.nperseg > input_layout.axis_length) {
        params.nperseg = input_layout.axis_length;
    }
    StftDeviceWorkspace workspace(static_cast<int>(x.size()), params, false);
    DeviceArray<ComplexFloat> computed(workspace.output_size());
    stft_device(x, computed, workspace, params);

    SpectralComplexDeviceResult result;
    result.frequency_count = workspace.nf;
    result.time_count = workspace.nframes;
    result.shape = workspace.output_shape;
    result.frequencies = DeviceArray<double>::from_host(
        spectral_frequencies(workspace.nfft, workspace.nf, params.fs));
    result.times = DeviceArray<double>::from_host(spectral_times(
        workspace.nframes, workspace.nperseg, workspace.noverlap,
        params.fs, workspace.boundary));
    set_spectral_complex_output<T>(computed, result);
    return result;
}

template <class T>
SpectrogramDeviceResult spectrogram_device(
    const DeviceArray<T>& x, const SpectrogramDeviceParams& requested_params)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    SpectrogramDeviceResult result;
    if (x.empty()) {
        result.dtype = SpectrogramOutputDtype::fp64;
        return result;
    }
    SpectrogramDeviceParams params = requested_params;
    const auto input_layout = spectral_layout::prepare_axis_plan(
        x.size(), params.shape, params.axis, 1, 0, false);
    if (params.nperseg <= 0) params.nperseg = params.window.custom_window.empty()
        ? 256 : static_cast<int>(params.window.custom_window.size());
    if (!params.window.custom_window.empty()
        && (params.window.custom_window.size()
                > static_cast<std::size_t>(input_layout.axis_length)
            || static_cast<int>(params.window.custom_window.size()) != params.nperseg))
        throw std::invalid_argument("window is longer than input or differs from nperseg");
    if (params.window.custom_window.empty()
        && params.nperseg > input_layout.axis_length)
        params.nperseg = input_layout.axis_length;
    if (params.noverlap < 0) params.noverlap = params.nperseg / 8;
    if (params.nfft <= 0) params.nfft = params.nperseg;
    SpectrogramDeviceWorkspace workspace(static_cast<int>(x.size()), params);
    result.frequency_count = workspace.stft.nf;
    result.time_count = workspace.stft.nframes;
    result.shape = workspace.stft.output_shape;
    result.frequencies = DeviceArray<double>::from_host(spectral_frequencies(
        workspace.stft.nfft, workspace.stft.nf, params.fs));
    result.times = DeviceArray<double>::from_host(spectral_times(
        workspace.stft.nframes, workspace.stft.nperseg,
        workspace.stft.noverlap, params.fs, ""));
    if (params.mode == "complex") {
        DeviceArray<ComplexFloat> computed(workspace.output_size());
        StftDeviceParams stft_params;
        stft_params.fs=params.fs;stft_params.window=params.window;
        stft_params.nperseg=params.nperseg;stft_params.noverlap=params.noverlap;
        stft_params.nfft=params.nfft;stft_params.detrend=params.detrend;
        stft_params.return_onesided=params.return_onesided;
        stft_params.boundary="";stft_params.padded=false;stft_params.axis=params.axis;
        stft_params.shape=params.shape;
        stft_device(x, computed, workspace.stft, stft_params);
        if constexpr (spectral_requires_fp64_v<T>) {
            result.dtype = SpectrogramOutputDtype::complex_fp64;
            result.complex_fp64 =
                cuda_utils::finalize_complex_fp64_device_storage(computed);
        } else {
            result.dtype = SpectrogramOutputDtype::complex_fp32;
            result.complex_fp32 = std::move(computed);
        }
        return result;
    }
    DeviceArray<float> computed(workspace.output_size());
    spectrogram_device(x, computed, workspace, params);
    if constexpr (spectral_requires_fp64_v<T>) {
        result.dtype = SpectrogramOutputDtype::fp64;
        result.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    } else {
        result.dtype = SpectrogramOutputDtype::fp32;
        result.fp32 = std::move(computed);
    }
    return result;
}

template <class T>
IstftDeviceResult istft_device(
    const DeviceArray<TypedComplex<T>>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (frequency_count < 1 || time_count < 1
        || z.size() != static_cast<std::size_t>(frequency_count * time_count)
        || params.fs <= 0.0F) throw std::invalid_argument("istft shape");
    const int n_default = params.input_onesided ? 2 * (frequency_count - 1) : frequency_count;
    const int nperseg = params.nperseg <= 0 ? n_default : params.nperseg;
    const int nfft = params.nfft <= 0
        ? (params.input_onesided && nperseg == n_default + 1 ? nperseg : n_default)
        : params.nfft;
    const int noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    IstftDeviceWorkspace workspace;
    workspace.reset(time_count, frequency_count, nperseg, noverlap, nfft,
        params.input_onesided, params.boundary, params.window);
    DeviceArray<float> computed(workspace.final_output_length);
    istft_device(z, computed, workspace);
    IstftDeviceResult result;
    std::vector<double> times(workspace.final_output_length);
    for (int index = 0; index < workspace.final_output_length; ++index)
        times[index] = static_cast<double>(index) / params.fs;
    result.times = DeviceArray<double>::from_host(times);
    if constexpr (spectral_requires_fp64_v<T>) {
        result.is_fp64 = true;
        result.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    } else result.fp32 = std::move(computed);
    return result;
}

template <class T>
IstftDeviceResult istft_device(
    const DeviceArray<T>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (frequency_count < 1 || time_count < 1
        || z.size() != static_cast<std::size_t>(frequency_count * time_count)
        || params.fs <= 0.0F) throw std::invalid_argument("istft shape");
    const int n_default = params.input_onesided ? 2 * (frequency_count - 1) : frequency_count;
    const int nperseg = params.nperseg <= 0 ? n_default : params.nperseg;
    const int nfft = params.nfft <= 0
        ? (params.input_onesided && nperseg == n_default + 1 ? nperseg : n_default)
        : params.nfft;
    const int noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    IstftDeviceWorkspace workspace;
    workspace.reset(time_count, frequency_count, nperseg, noverlap, nfft,
        params.input_onesided, params.boundary, params.window);
    DeviceArray<float> computed(workspace.final_output_length);
    istft_device(z, computed, workspace);
    IstftDeviceResult result;
    std::vector<double> times(workspace.final_output_length);
    for (int index = 0; index < workspace.final_output_length; ++index)
        times[index] = static_cast<double>(index) / params.fs;
    result.times = DeviceArray<double>::from_host(times);
    if constexpr (spectral_requires_fp64_v<T>) {
        result.is_fp64 = true;
        result.fp64 = cuda_utils::finalize_fp64_device_storage(computed);
    } else result.fp32 = std::move(computed);
    return result;
}

#define INSTANTIATE_SPECTRAL_CPU(T) \
    template std::vector<ComplexFloat> csd_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&); \
    template SpectralComplexCpuResult csd_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const CsdDeviceParams&); \
    template std::vector<ComplexFloat> stft_typed_cpu( \
        const std::vector<T>&, int, int); \
    template SpectralComplexCpuResult stft_typed_cpu( \
        const std::vector<T>&, const StftDeviceParams&); \
    template std::vector<float> istft_typed_cpu( \
        const std::vector<TypedComplex<T>>&, int, int, int); \
    template IstftCpuResult istft_typed_cpu( \
        const std::vector<TypedComplex<T>>&, int, int, const IstftDeviceParams&); \
    template IstftCpuResult istft_typed_cpu( \
        const std::vector<T>&, int, int, const IstftDeviceParams&); \
    template std::vector<float> spectrogram_typed_cpu( \
        const std::vector<T>&, int, int); \
    template SpectrogramCpuResult spectrogram_typed_cpu( \
        const std::vector<T>&, const SpectrogramDeviceParams&); \
    template std::vector<double> lombscargle_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, \
        bool, bool); \
    template DeviceArray<double> lombscargle_device( \
        const DeviceArray<T>&, const DeviceArray<T>&, const DeviceArray<T>&, \
        bool, bool); \
    template VectorStrengthCpuResult vectorstrength_typed_cpu( \
        const std::vector<T>&, T); \
    template VectorStrengthCpuResult vectorstrength_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&); \
    template VectorStrengthDeviceResult vectorstrength_device( \
        const DeviceArray<T>&, T); \
    template VectorStrengthDeviceResult vectorstrength_device( \
        const DeviceArray<T>&, const DeviceArray<T>&); \
    template SpectralComplexDeviceResult csd_device( \
        const DeviceArray<T>&, const DeviceArray<T>&, const CsdDeviceParams&); \
    template SpectralComplexDeviceResult stft_device( \
        const DeviceArray<T>&, const StftDeviceParams&); \
    template SpectrogramDeviceResult spectrogram_device( \
        const DeviceArray<T>&, const SpectrogramDeviceParams&); \
    template IstftDeviceResult istft_device( \
        const DeviceArray<TypedComplex<T>>&, int, int, const IstftDeviceParams&); \
    template IstftDeviceResult istft_device( \
        const DeviceArray<T>&, int, int, const IstftDeviceParams&)

INSTANTIATE_SPECTRAL_CPU(float);
INSTANTIATE_SPECTRAL_CPU(__half);
INSTANTIATE_SPECTRAL_CPU(std::int32_t);
INSTANTIATE_SPECTRAL_CPU(std::int16_t);
INSTANTIATE_SPECTRAL_CPU(std::int8_t);

#undef INSTANTIATE_SPECTRAL_CPU

} // namespace cusignal
