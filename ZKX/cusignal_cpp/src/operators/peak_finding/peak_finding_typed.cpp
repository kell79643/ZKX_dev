#include <cusignal/operators/peak_finding/peak_finding_typed.h>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cusignal {
namespace {

std::size_t checked_product(std::size_t left, std::size_t right)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument("relative extrema shape exceeds size_t capacity");
    }
    return left * right;
}

bool same_shape(const RelativeExtremaShape& left, const RelativeExtremaShape& right)
{
    return left.dimensions == right.dimensions && left.elements == right.elements;
}

RelativeExtremaShape validated_shape(const RelativeExtremaShape& shape)
{
    RelativeExtremaShape validated(shape.dimensions);
    if (!same_shape(shape, validated)) {
        throw std::invalid_argument("relative extrema shape was mutated");
    }
    return validated;
}

int normalized_axis(int axis, std::size_t rank)
{
    const auto signed_rank = static_cast<std::int64_t>(rank);
    std::int64_t normalized = axis;
    if (normalized < 0) normalized += signed_rank;
    if (normalized < 0 || normalized >= signed_rank) {
        throw std::invalid_argument("relative extrema axis is out of range");
    }
    return static_cast<int>(normalized);
}

bool clip_mode(const std::string& mode)
{
    if (mode == "raise") {
        throw std::logic_error("fixed cuSignal does not implement mode='raise'");
    }
    return mode == "clip";
}

void validate_comparator(RelativeExtremaComparator comparator)
{
    const int value = static_cast<int>(comparator);
    if (value < static_cast<int>(RelativeExtremaComparator::less) ||
        value > static_cast<int>(RelativeExtremaComparator::not_equal)) {
        throw std::invalid_argument("relative extrema comparator is invalid");
    }
}

template <typename T>
auto comparison_value(T value)
{
    return compute_policy::InputTraits<T>::load_comparison(value);
}

template <typename Value>
bool compare_values(Value left, Value right, RelativeExtremaComparator comparator)
{
    switch (comparator) {
    case RelativeExtremaComparator::less: return left < right;
    case RelativeExtremaComparator::greater: return left > right;
    case RelativeExtremaComparator::less_equal: return left <= right;
    case RelativeExtremaComparator::greater_equal: return left >= right;
    case RelativeExtremaComparator::equal: return left == right;
    case RelativeExtremaComparator::not_equal: return left != right;
    }
    return false;
}

std::size_t wrapped_plus(std::size_t coordinate, std::size_t distance, std::size_t length)
{
    const std::size_t offset = distance % length;
    return offset >= length - coordinate ? offset - (length - coordinate)
                                         : coordinate + offset;
}

std::size_t wrapped_minus(std::size_t coordinate, std::size_t distance, std::size_t length)
{
    const std::size_t offset = distance % length;
    return coordinate >= offset ? coordinate - offset : length - (offset - coordinate);
}

std::size_t clipped_plus(std::size_t coordinate, std::size_t distance, std::size_t length)
{
    return distance >= length - coordinate ? length - 1 : coordinate + distance;
}

std::size_t clipped_minus(std::size_t coordinate, std::size_t distance)
{
    return distance > coordinate ? 0 : coordinate - distance;
}

void append_coordinates(
    std::size_t flat,
    const RelativeExtremaShape& shape,
    RelativeExtremaHostResult& result)
{
    for (std::size_t dimension = shape.rank(); dimension-- > 0;) {
        const std::size_t length = shape.dimensions[dimension];
        result.coordinates[dimension].push_back(
            static_cast<std::int64_t>(flat % length));
        flat /= length;
    }
}

}  // namespace

void RelativeExtremaShape::reset(std::vector<std::size_t> dimensions_in)
{
    if (dimensions_in.empty()) {
        throw std::invalid_argument("relative extrema does not accept scalar input");
    }
    if (dimensions_in.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("relative extrema rank exceeds int axis capacity");
    }
    std::size_t count = 1;
    for (std::size_t length : dimensions_in) count = checked_product(count, length);
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("relative extrema flat index exceeds INT64 capacity");
    }
    dimensions = std::move(dimensions_in);
    elements = count;
}

void RelativeExtremaDeviceWorkspace::reset(std::size_t capacity)
{
    mask.reset(capacity);
    flat_indices.reset(capacity);
}

template <typename T>
RelativeExtremaHostResult argrelextrema_typed_cpu(
    const std::vector<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaComparator comparator,
    int axis,
    int order,
    const std::string& mode)
{
    static_assert(detail::is_simple_signal_input_v<T>,
                  "unsupported relative extrema business dtype");
    const RelativeExtremaShape validated = validated_shape(shape);
    if (data.size() != validated.elements) {
        throw std::invalid_argument("relative extrema input shape mismatch");
    }
    if (order < 1) throw std::invalid_argument("relative extrema order must be >= 1");
    validate_comparator(comparator);
    const int selected_axis = normalized_axis(axis, validated.rank());
    const bool clip = clip_mode(mode);

    RelativeExtremaHostResult result;
    result.coordinates.resize(validated.rank());
    if (validated.elements == 0) return result;

    std::size_t axis_stride = 1;
    for (std::size_t dimension = static_cast<std::size_t>(selected_axis) + 1;
         dimension < validated.rank(); ++dimension) {
        axis_stride *= validated.dimensions[dimension];
    }
    const std::size_t axis_length = validated.dimensions[selected_axis];
    for (std::size_t flat = 0; flat < validated.elements; ++flat) {
        const std::size_t coordinate = (flat / axis_stride) % axis_length;
        const auto center = comparison_value(data[flat]);
        bool extrema = true;
        for (int step = 1; extrema; ++step) {
            const std::size_t distance = static_cast<std::size_t>(step);
            const std::size_t plus_coordinate = clip
                ? clipped_plus(coordinate, distance, axis_length)
                : wrapped_plus(coordinate, distance, axis_length);
            const std::size_t minus_coordinate = clip
                ? clipped_minus(coordinate, distance)
                : wrapped_minus(coordinate, distance, axis_length);
            const std::size_t plus = plus_coordinate >= coordinate
                ? flat + (plus_coordinate - coordinate) * axis_stride
                : flat - (coordinate - plus_coordinate) * axis_stride;
            const std::size_t minus = minus_coordinate >= coordinate
                ? flat + (minus_coordinate - coordinate) * axis_stride
                : flat - (coordinate - minus_coordinate) * axis_stride;
            extrema = compare_values(center, comparison_value(data[plus]), comparator) &&
                      compare_values(center, comparison_value(data[minus]), comparator);
            if (step == order) break;
        }
        if (extrema) append_coordinates(flat, validated, result);
    }
    return result;
}

template <typename T>
RelativeExtremaHostResult argrelmin_typed_cpu(
    const std::vector<T>& data,
    const RelativeExtremaShape& shape,
    int axis,
    int order,
    const std::string& mode)
{
    return argrelextrema_typed_cpu(
        data, shape, RelativeExtremaComparator::less, axis, order, mode);
}

template <typename T>
RelativeExtremaHostResult argrelmax_typed_cpu(
    const std::vector<T>& data,
    const RelativeExtremaShape& shape,
    int axis,
    int order,
    const std::string& mode)
{
    return argrelextrema_typed_cpu(
        data, shape, RelativeExtremaComparator::greater, axis, order, mode);
}

#define INSTANTIATE_PEAK_CPU(T) \
    template RelativeExtremaHostResult argrelextrema_typed_cpu(const std::vector<T>&, const RelativeExtremaShape&, RelativeExtremaComparator, int, int, const std::string&); \
    template RelativeExtremaHostResult argrelmin_typed_cpu(const std::vector<T>&, const RelativeExtremaShape&, int, int, const std::string&); \
    template RelativeExtremaHostResult argrelmax_typed_cpu(const std::vector<T>&, const RelativeExtremaShape&, int, int, const std::string&)

INSTANTIATE_PEAK_CPU(float);
INSTANTIATE_PEAK_CPU(__half);
INSTANTIATE_PEAK_CPU(std::int32_t);
INSTANTIATE_PEAK_CPU(std::int16_t);
INSTANTIATE_PEAK_CPU(std::int8_t);

#undef INSTANTIATE_PEAK_CPU

}  // namespace cusignal
