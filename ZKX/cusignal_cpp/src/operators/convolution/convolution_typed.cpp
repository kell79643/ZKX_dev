#include <cusignal/operators/convolution/convolution_typed.h>

#include <algorithm>
#include <stdexcept>

namespace cusignal {
namespace {

template <typename T>
T signed_from_bits(std::make_unsigned_t<T> raw)
{
    return compute_policy::signed_from_modular_bits<T>(raw);
}

template <typename T>
T rounded_modular_store(float value)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
    using U = std::make_unsigned_t<T>;
    constexpr int bits = std::numeric_limits<U>::digits;
    const double modulus = std::ldexp(1.0, bits);
    double wrapped = std::fmod(std::nearbyint(static_cast<double>(value)), modulus);
    if (wrapped < 0.0) wrapped += modulus;
    return signed_from_bits<T>(static_cast<U>(wrapped));
}

template <typename T>
T truncated_modular_store(double value)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
    using U = std::make_unsigned_t<T>;
    constexpr int bits = std::numeric_limits<U>::digits;
    const double modulus = std::ldexp(1.0, bits);
    double wrapped = std::fmod(std::trunc(value), modulus);
    if (wrapped < 0.0) wrapped += modulus;
    return signed_from_bits<T>(static_cast<U>(wrapped));
}

template <typename T>
T fp32_boundary_store(float value)
{
    if constexpr (std::is_integral_v<T>) {
        return rounded_modular_store<T>(value);
    } else {
        return detail::ConvolutionTypePolicy<T>::store(value);
    }
}

int correlate2d_boundary_code(const std::string& boundary)
{
    if (boundary == "fill") return 0;
    if (boundary == "wrap") return 1;
    if (boundary == "symm") return 2;
    throw std::invalid_argument("typed correlate2d boundary must be fill, wrap, or symm");
}

std::size_t checked_matrix_size(int rows, int cols, const char* label)
{
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument(std::string("typed correlate2d negative ") + label + " dimension");
    }
    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("typed correlate2d ") + label + " exceeds 32-bit indexing");
    }
    return size;
}

std::size_t checked_nd_size(const std::vector<int>& shape, const char* label)
{
    std::size_t total = 1;
    for (int dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument(
                std::string("typed correlate negative ") + label + " dimension");
        }
        if (dimension == 0) return 0;
        if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                        static_cast<std::size_t>(dimension)) {
            throw std::invalid_argument(
                std::string("typed correlate ") + label + " exceeds 32-bit indexing");
        }
        total *= static_cast<std::size_t>(dimension);
    }
    return total;
}

std::vector<int> contiguous_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        const long long stride = static_cast<long long>(strides[index]) * shape[index];
        if (stride > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("typed correlate stride exceeds 32-bit indexing");
        }
        strides[index - 1] = static_cast<int>(stride);
    }
    return strides;
}

int map_correlate2d_index(std::int64_t index, int size, int boundary_code)
{
    if (index >= 0 && index < size) return static_cast<int>(index);
    if (boundary_code == 0) return -1;
    if (boundary_code == 1) {
        std::int64_t wrapped = index % size;
        if (wrapped < 0) wrapped += size;
        return static_cast<int>(wrapped);
    }
    const std::int64_t period = static_cast<std::int64_t>(size) * 2;
    std::int64_t wrapped = index % period;
    if (wrapped < 0) wrapped += period;
    return static_cast<int>(wrapped < size ? wrapped : period - 1 - wrapped);
}

template <typename T>
T correlate2d_fillvalue(const Correlate2DOptions& options)
{
    if (options.boundary != "fill") return fp32_boundary_store<T>(0.0F);
    if constexpr (std::is_integral_v<T>) {
        if (!std::isfinite(options.fillvalue)) {
            throw std::invalid_argument("typed correlate2d integral fillvalue must be finite");
        }
        return truncated_modular_store<T>(options.fillvalue);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half_rn(static_cast<float>(options.fillvalue));
    } else {
        return static_cast<float>(options.fillvalue);
    }
}

}  // namespace

std::size_t correlate_output_size(
    std::size_t len1,
    std::size_t len2,
    const std::string& mode)
{
    if (mode != "full" && mode != "same" && mode != "valid") {
        throw std::invalid_argument("typed correlate mode must be full, same, or valid");
    }
    if (len1 == 0 || len2 == 0) return 0;
    if (mode == "full") {
        if (len1 > std::numeric_limits<std::size_t>::max() - len2 + 1) {
            throw std::invalid_argument("typed correlate output size overflow");
        }
        return len1 + len2 - 1;
    }
    if (mode == "same") return len1;
    return len1 >= len2 ? len1 - len2 + 1 : len2 - len1 + 1;
}

CorrelateNDOutputShape correlate_nd_output_shape(const CorrelateNDOptions& options)
{
    if (options.mode != "full" && options.mode != "same" && options.mode != "valid") {
        throw std::invalid_argument("typed correlate mode must be full, same, or valid");
    }
    if (options.method != CorrelateMethod::direct &&
        options.method != CorrelateMethod::fft &&
        options.method != CorrelateMethod::auto_select) {
        throw std::invalid_argument("typed correlate method must be direct, fft, or auto");
    }
    if (options.shape1.size() != options.shape2.size()) {
        throw std::invalid_argument("typed correlate inputs must have the same rank");
    }
    if (options.method == CorrelateMethod::direct && options.shape1.size() > 1) {
        throw std::invalid_argument("typed correlate direct method is only implemented for 1D");
    }

    const std::size_t size1 = checked_nd_size(options.shape1, "in1");
    const std::size_t size2 = checked_nd_size(options.shape2, "in2");
    if (size1 == 0 || size2 == 0) return {{0}, 0};
    if (options.shape1.empty()) return {{}, 1};

    CorrelateNDOutputShape output;
    output.dimensions.resize(options.shape1.size());
    bool in1_dominates = true;
    bool in2_dominates = true;
    for (std::size_t axis = 0; axis < options.shape1.size(); ++axis) {
        const int first = options.shape1[axis];
        const int second = options.shape2[axis];
        in1_dominates = in1_dominates && first >= second;
        in2_dominates = in2_dominates && second >= first;
        if (options.mode == "full") {
            if (first > std::numeric_limits<int>::max() - second + 1) {
                throw std::invalid_argument("typed correlate full shape exceeds 32-bit indexing");
            }
            output.dimensions[axis] = first + second - 1;
        } else if (options.mode == "same") {
            output.dimensions[axis] = first;
        } else {
            output.dimensions[axis] = std::abs(first - second) + 1;
        }
    }
    if (options.mode == "valid" && !in1_dominates && !in2_dominates) {
        throw std::invalid_argument(
            "typed correlate valid mode requires one input to dominate in every dimension");
    }
    output.elements = checked_nd_size(output.dimensions, "output");
    return output;
}

Correlate2DOutputShape correlate2d_output_shape(const Correlate2DOptions& options)
{
    if (options.mode != "full" && options.mode != "same" && options.mode != "valid") {
        throw std::invalid_argument("typed correlate2d mode must be full, same, or valid");
    }
    (void)correlate2d_boundary_code(options.boundary);
    (void)checked_matrix_size(options.rows1, options.cols1, "in1");
    (void)checked_matrix_size(options.rows2, options.cols2, "in2");
    if (options.rows1 == 0 || options.cols1 == 0 ||
        options.rows2 == 0 || options.cols2 == 0) {
        return {};
    }

    Correlate2DOutputShape shape;
    if (options.mode == "full") {
        if (options.rows1 > std::numeric_limits<int>::max() - options.rows2 + 1 ||
            options.cols1 > std::numeric_limits<int>::max() - options.cols2 + 1) {
            throw std::invalid_argument("typed correlate2d full shape exceeds 32-bit indexing");
        }
        shape.rows = options.rows1 + options.rows2 - 1;
        shape.cols = options.cols1 + options.cols2 - 1;
    } else if (options.mode == "same") {
        shape.rows = options.rows1;
        shape.cols = options.cols1;
    } else {
        const bool in1_dominates = options.rows1 >= options.rows2 && options.cols1 >= options.cols2;
        const bool in2_dominates = options.rows2 >= options.rows1 && options.cols2 >= options.cols1;
        if (!in1_dominates && !in2_dominates) {
            throw std::invalid_argument(
                "typed correlate2d valid mode requires one input to dominate in every dimension");
        }
        shape.rows = std::abs(options.rows1 - options.rows2) + 1;
        shape.cols = std::abs(options.cols1 - options.cols2) + 1;
    }
    (void)checked_matrix_size(shape.rows, shape.cols, "output");
    return shape;
}

void Correlate2DWorkspace::reset(const Correlate2DOptions& options)
{
    const Correlate2DOutputShape output = correlate2d_output_shape(options);
    rows1 = options.rows1;
    cols1 = options.cols1;
    rows2 = options.rows2;
    cols2 = options.cols2;
    mode = options.mode;
    boundary = options.boundary;
    fillvalue = options.fillvalue;
    boundary_code = mode == "valid" ? 0 : correlate2d_boundary_code(boundary);
    out_rows = output.rows;
    out_cols = output.cols;
    if (output.size() == 0) {
        row_start = 0;
        col_start = 0;
    } else {
        if (mode == "full") {
            row_start = 0;
            col_start = 0;
        } else if (mode == "same") {
            row_start = rows2 / 2;
            col_start = cols2 / 2;
        } else {
            row_start = std::min(rows1, rows2) - 1;
            col_start = std::min(cols1, cols2) - 1;
        }
    }

    fill_fp32 = mode == "valid" ? 0.0F : static_cast<float>(fillvalue);
    fill_fp16 = __float2half_rn(fill_fp32);
    integral_fill_valid = mode == "valid" || boundary != "fill" || std::isfinite(fillvalue);
    if (mode != "valid" && integral_fill_valid && boundary == "fill") {
        fill_int32 = truncated_modular_store<std::int32_t>(fillvalue);
        fill_int16 = truncated_modular_store<std::int16_t>(fillvalue);
        fill_int8 = truncated_modular_store<std::int8_t>(fillvalue);
    } else {
        fill_int32 = 0;
        fill_int16 = 0;
        fill_int8 = 0;
    }
}

template <typename T>
std::vector<T> correlate_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const std::string& mode,
    CorrelateMethod method)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate business input dtype");
    if (method == CorrelateMethod::auto_select) method = CorrelateMethod::direct;
    if (method != CorrelateMethod::direct && method != CorrelateMethod::fft) {
        throw std::invalid_argument("typed correlate method must be direct, fft, or auto");
    }
    const std::size_t out_len = correlate_output_size(in1.size(), in2.size(), mode);
    if (out_len == 0) return {};
    if (in1.size() > std::numeric_limits<std::size_t>::max() - in2.size() + 1) {
        throw std::invalid_argument("typed correlate full size overflow");
    }
    const std::size_t full_len = in1.size() + in2.size() - 1;
    const std::size_t start = (full_len - out_len) / 2;
    std::vector<T> out(out_len);
    for (std::size_t out_idx = 0; out_idx < out_len; ++out_idx) {
        const std::size_t full_idx = start + out_idx;
        using ModularAccumulator = compute_policy::modular_unsigned_t<T>;
        ModularAccumulator modular_acc = 0;
        float fp32_acc = 0.0F;
        for (std::size_t j = 0; j < in2.size(); ++j) {
            const auto shifted = static_cast<std::ptrdiff_t>(full_idx) -
                                 static_cast<std::ptrdiff_t>(in2.size() - 1 - j);
            if (shifted >= 0 && shifted < static_cast<std::ptrdiff_t>(in1.size())) {
                if constexpr (std::is_integral_v<T>) {
                    if (method == CorrelateMethod::direct) {
                        modular_acc = compute_policy::modular_multiply_add<T>(
                            modular_acc,
                            in1[static_cast<std::size_t>(shifted)], in2[j]);
                    } else {
                        fp32_acc += static_cast<float>(in1[static_cast<std::size_t>(shifted)]) *
                                    static_cast<float>(in2[j]);
                    }
                } else {
                    fp32_acc = std::fma(
                        detail::ConvolutionTypePolicy<T>::load(
                            in1[static_cast<std::size_t>(shifted)]),
                        detail::ConvolutionTypePolicy<T>::load(in2[j]),
                        fp32_acc);
                }
            }
        }
        if constexpr (std::is_integral_v<T>) {
            out[out_idx] = method == CorrelateMethod::direct
                ? signed_from_bits<T>(modular_acc)
                : rounded_modular_store<T>(fp32_acc);
        } else {
            out[out_idx] = fp32_boundary_store<T>(fp32_acc);
        }
    }
    return out;
}

template <typename T>
std::vector<T> correlate_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const CorrelateNDOptions& options)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate business input dtype");
    const CorrelateNDOutputShape output = correlate_nd_output_shape(options);
    const std::size_t size1 = checked_nd_size(options.shape1, "in1");
    const std::size_t size2 = checked_nd_size(options.shape2, "in2");
    if (in1.size() != size1 || in2.size() != size2) {
        throw std::invalid_argument("correlate_typed_cpu input shape mismatch");
    }
    if (output.size() == 0) return {};
    if (options.shape1.empty()) {
        const float product = detail::ConvolutionTypePolicy<T>::load(in1[0]) *
                              detail::ConvolutionTypePolicy<T>::load(in2[0]);
        return {fp32_boundary_store<T>(product)};
    }

    CorrelateMethod method = options.method == CorrelateMethod::auto_select
        ? (options.shape1.size() == 1 ? CorrelateMethod::direct : CorrelateMethod::fft)
        : options.method;
    if (options.shape1.size() == 1) {
        return correlate_typed_cpu(in1, in2, options.mode, method);
    }

    std::vector<int> full_shape(options.shape1.size());
    std::vector<int> crop_start(options.shape1.size());
    for (std::size_t axis = 0; axis < full_shape.size(); ++axis) {
        full_shape[axis] = options.shape1[axis] + options.shape2[axis] - 1;
        crop_start[axis] = (full_shape[axis] - output.dimensions[axis]) / 2;
    }
    const std::vector<int> stride1 = contiguous_strides(options.shape1);
    const std::vector<int> stride2 = contiguous_strides(options.shape2);
    const std::vector<int> output_stride = contiguous_strides(output.dimensions);
    std::vector<T> result(output.size());
    std::vector<int> output_coord(options.shape1.size());
    std::vector<int> kernel_coord(options.shape1.size());

    for (std::size_t output_index = 0; output_index < output.size(); ++output_index) {
        std::size_t remainder = output_index;
        for (std::size_t axis = 0; axis < output_coord.size(); ++axis) {
            output_coord[axis] = static_cast<int>(remainder / output_stride[axis]);
            remainder %= static_cast<std::size_t>(output_stride[axis]);
        }
        float sum = 0.0F;
        for (std::size_t kernel_index = 0; kernel_index < size2; ++kernel_index) {
            remainder = kernel_index;
            std::size_t input_index = 0;
            bool inside = true;
            for (std::size_t axis = 0; axis < kernel_coord.size(); ++axis) {
                kernel_coord[axis] = static_cast<int>(remainder / stride2[axis]);
                remainder %= static_cast<std::size_t>(stride2[axis]);
                const int full_coordinate = crop_start[axis] + output_coord[axis];
                const int source = full_coordinate -
                    (options.shape2[axis] - 1 - kernel_coord[axis]);
                if (source < 0 || source >= options.shape1[axis]) {
                    inside = false;
                    break;
                }
                input_index += static_cast<std::size_t>(source) * stride1[axis];
            }
            if (inside) {
                sum += detail::ConvolutionTypePolicy<T>::load(in1[input_index]) *
                       detail::ConvolutionTypePolicy<T>::load(in2[kernel_index]);
            }
        }
        result[output_index] = fp32_boundary_store<T>(sum);
    }
    return result;
}

template <typename T>
std::vector<T> correlate2d_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const Correlate2DOptions& options)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate2d business input dtype");
    const Correlate2DOutputShape output = correlate2d_output_shape(options);
    if (in1.size() != checked_matrix_size(options.rows1, options.cols1, "in1") ||
        in2.size() != checked_matrix_size(options.rows2, options.cols2, "in2")) {
        throw std::invalid_argument("correlate2d_typed_cpu shape mismatch");
    }
    if (output.size() == 0) return {};

    int row_start = 0;
    int col_start = 0;
    if (options.mode == "same") {
        row_start = options.rows2 / 2;
        col_start = options.cols2 / 2;
    } else if (options.mode == "valid") {
        row_start = std::min(options.rows1, options.rows2) - 1;
        col_start = std::min(options.cols1, options.cols2) - 1;
    }
    const int boundary_code = options.mode == "valid"
        ? 0
        : correlate2d_boundary_code(options.boundary);
    const T fill = options.mode == "valid"
        ? fp32_boundary_store<T>(0.0F)
        : correlate2d_fillvalue<T>(options);
    std::vector<T> out(output.size());
    for (int out_r = 0; out_r < output.rows; ++out_r) {
        for (int out_c = 0; out_c < output.cols; ++out_c) {
            const int full_r = row_start + out_r;
            const int full_c = col_start + out_c;
            using ModularAccumulator = compute_policy::modular_unsigned_t<T>;
            ModularAccumulator modular_acc = 0;
            float fp32_acc = 0.0F;
            for (int kr = 0; kr < options.rows2; ++kr) {
                const std::int64_t source_r = static_cast<std::int64_t>(full_r) -
                    (options.rows2 - 1 - kr);
                const int in_r = map_correlate2d_index(
                    source_r, options.rows1, boundary_code);
                for (int kc = 0; kc < options.cols2; ++kc) {
                    const std::int64_t source_c = static_cast<std::int64_t>(full_c) -
                        (options.cols2 - 1 - kc);
                    const int in_c = map_correlate2d_index(
                        source_c, options.cols1, boundary_code);
                    const T sample = in_r < 0 || in_c < 0
                        ? fill
                        : in1[static_cast<std::size_t>(in_r) * options.cols1 + in_c];
                    const T coefficient = in2[static_cast<std::size_t>(kr) * options.cols2 + kc];
                    if constexpr (std::is_integral_v<T>) {
                        modular_acc = compute_policy::modular_multiply_add<T>(
                            modular_acc, sample, coefficient);
                    } else {
                        fp32_acc = std::fma(
                            detail::ConvolutionTypePolicy<T>::load(sample),
                            detail::ConvolutionTypePolicy<T>::load(coefficient),
                            fp32_acc);
                    }
                }
            }
            const std::size_t index = static_cast<std::size_t>(out_r) * output.cols + out_c;
            if constexpr (std::is_integral_v<T>) {
                out[index] = signed_from_bits<T>(modular_acc);
            } else {
                out[index] = fp32_boundary_store<T>(fp32_acc);
            }
        }
    }
    return out;
}

#define INSTANTIATE_CONV_CPU(T) \
    template std::vector<T> correlate_typed_cpu(const std::vector<T>&, const std::vector<T>&, const std::string&, CorrelateMethod); \
    template std::vector<T> correlate_typed_cpu(const std::vector<T>&, const std::vector<T>&, const CorrelateNDOptions&); \
    template std::vector<T> correlate2d_typed_cpu(const std::vector<T>&, const std::vector<T>&, const Correlate2DOptions&)

INSTANTIATE_CONV_CPU(float);
INSTANTIATE_CONV_CPU(__half);
INSTANTIATE_CONV_CPU(std::int32_t);
INSTANTIATE_CONV_CPU(std::int16_t);
INSTANTIATE_CONV_CPU(std::int8_t);

#undef INSTANTIATE_CONV_CPU

}  // namespace cusignal
