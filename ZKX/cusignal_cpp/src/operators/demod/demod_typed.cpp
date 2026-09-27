#include <cusignal/operators/demod/demod_typed.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cusignal {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 6.28318530717958647692F;

std::size_t checked_nonzero_product(
    const std::vector<int>& shape,
    std::size_t begin,
    std::size_t end,
    const char* label)
{
    std::size_t product = 1;
    const std::size_t limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t i = begin; i < end; ++i) {
        const std::size_t dimension = static_cast<std::size_t>(shape[i]);
        if (dimension != 0 && product > limit / dimension) {
            throw std::invalid_argument(label);
        }
        product *= dimension;
    }
    return product;
}

bool same_workspace(
    const FmDemodWorkspace& left,
    const FmDemodWorkspace& right)
{
    return left.input_shape == right.input_shape &&
           left.output_shape == right.output_shape &&
           left.axis == right.axis &&
           left.axis_length == right.axis_length &&
           left.output_axis_length == right.output_axis_length &&
           left.inner_size == right.inner_size &&
           left.outer_size == right.outer_size &&
           left.input_count == right.input_count &&
           left.output_count == right.output_count;
}

FmDemodWorkspace validated_workspace(const FmDemodWorkspace& workspace)
{
    FmDemodOptions options;
    options.shape = workspace.input_shape;
    options.axis = workspace.axis;
    FmDemodWorkspace validated(options);
    if (!same_workspace(workspace, validated)) {
        throw std::invalid_argument("fm_demod workspace mismatch");
    }
    return validated;
}

FmDemodWorkspace one_dimensional_workspace(std::size_t count)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("fm_demod input exceeds 32-bit indexing");
    }
    FmDemodOptions options;
    options.shape = {static_cast<int>(count)};
    return FmDemodWorkspace(options);
}

template <typename T>
float phase(const DemodComplex<T>& value)
{
    const float real = detail::DemodTypePolicy<T>::load(value.re);
    const float imag = detail::DemodTypePolicy<T>::load(value.im);
    return std::atan2(imag, real);
}

float unwrap_local_delta(float previous_phase, float current_phase)
{
    const float delta = current_phase - previous_phase;
    if (std::fabs(delta) < kPi) return delta;

    float corrected = std::fmod(delta + kPi, kTwoPi);
    if (corrected < 0.0F) corrected += kTwoPi;
    corrected -= kPi;
    if (corrected == -kPi && delta > 0.0F) corrected = kPi;
    return corrected;
}

template <typename T>
float demod_sample(const DemodComplex<T>& previous, const DemodComplex<T>& current)
{
    return unwrap_local_delta(phase(previous), phase(current));
}

}  // namespace

void FmDemodWorkspace::reset(const FmDemodOptions& options)
{
    if (options.shape.empty()) {
        throw std::invalid_argument("fm_demod input must have at least one dimension");
    }
    if (options.shape.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("fm_demod rank exceeds 32-bit indexing");
    }
    for (int dimension : options.shape) {
        if (dimension < 0) {
            throw std::invalid_argument("fm_demod shape dimensions must be non-negative");
        }
    }

    const int rank = static_cast<int>(options.shape.size());
    if (options.axis < -rank || options.axis >= rank) {
        throw std::invalid_argument("fm_demod axis is out of range");
    }
    const int normalized_axis = options.axis < 0 ? options.axis + rank : options.axis;
    const bool has_zero = [&]() {
        for (int dimension : options.shape) {
            if (dimension == 0) return true;
        }
        return false;
    }();

    input_shape = options.shape;
    output_shape = options.shape;
    axis = normalized_axis;
    axis_length = input_shape[static_cast<std::size_t>(axis)];
    output_axis_length = axis_length > 0 ? axis_length - 1 : 0;
    output_shape[static_cast<std::size_t>(axis)] = output_axis_length;

    if (has_zero) {
        input_count = 0;
        output_count = 0;
        inner_size = 0;
        outer_size = 0;
        return;
    }

    input_count = checked_nonzero_product(
        input_shape, 0, input_shape.size(), "fm_demod input exceeds 32-bit indexing");
    const std::size_t inner = checked_nonzero_product(
        input_shape,
        static_cast<std::size_t>(axis + 1),
        input_shape.size(),
        "fm_demod inner layout exceeds 32-bit indexing");
    const std::size_t outer = checked_nonzero_product(
        input_shape,
        0,
        static_cast<std::size_t>(axis),
        "fm_demod outer layout exceeds 32-bit indexing");
    inner_size = static_cast<int>(inner);
    outer_size = static_cast<int>(outer);
    if (output_axis_length == 0) {
        output_count = 0;
    } else {
        output_count = checked_nonzero_product(
            output_shape, 0, output_shape.size(),
            "fm_demod output exceeds 32-bit indexing");
    }
}

template <typename T>
std::vector<float> fm_demod_typed_cpu(
    const std::vector<DemodComplex<T>>& x,
    const FmDemodWorkspace& workspace)
{
    static_assert(detail::is_demod_input_v<T>, "unsupported fm_demod business input dtype");
    const FmDemodWorkspace layout = validated_workspace(workspace);
    if (x.size() != layout.input_size()) {
        throw std::invalid_argument("fm_demod_typed_cpu input shape mismatch");
    }
    std::vector<float> output(layout.output_size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        const int linear = static_cast<int>(index);
        const int inner_index = linear % layout.inner_size;
        const int axis_index = (linear / layout.inner_size) % layout.output_axis_length;
        const int outer_index = linear / (layout.inner_size * layout.output_axis_length);
        const int previous =
            (outer_index * layout.axis_length + axis_index) * layout.inner_size + inner_index;
        const int current = previous + layout.inner_size;
        output[index] = demod_sample(x[static_cast<std::size_t>(previous)],
                                    x[static_cast<std::size_t>(current)]);
    }
    return output;
}

template <typename T>
std::vector<float> fm_demod_typed_cpu(const std::vector<DemodComplex<T>>& x)
{
    return fm_demod_typed_cpu(x, one_dimensional_workspace(x.size()));
}

template <typename T>
std::vector<float> fm_demod_typed_cpu(
    const std::vector<DemodComplex<T>>& x,
    int rows,
    int cols,
    int axis)
{
    FmDemodOptions options;
    options.shape = {rows, cols};
    options.axis = axis;
    return fm_demod_typed_cpu(x, FmDemodWorkspace(options));
}

#define INSTANTIATE_DEMOD_CPU(T) \
    template std::vector<float> fm_demod_typed_cpu(const std::vector<DemodComplex<T>>&, const FmDemodWorkspace&); \
    template std::vector<float> fm_demod_typed_cpu(const std::vector<DemodComplex<T>>&); \
    template std::vector<float> fm_demod_typed_cpu(const std::vector<DemodComplex<T>>&, int, int, int)

INSTANTIATE_DEMOD_CPU(float);
INSTANTIATE_DEMOD_CPU(__half);
INSTANTIATE_DEMOD_CPU(std::int32_t);
INSTANTIATE_DEMOD_CPU(std::int16_t);
INSTANTIATE_DEMOD_CPU(std::int8_t);

#undef INSTANTIATE_DEMOD_CPU

}  // namespace cusignal
