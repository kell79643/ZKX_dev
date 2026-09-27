#include <cusignal/operators/peak_finding/peak_finding_typed.h>
#include "peak_finding_kernels.cuh"

#include <cusignal/runtime/cuda_utils.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace cusignal {
namespace {

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

int comparator_code(RelativeExtremaComparator comparator)
{
    const int value = static_cast<int>(comparator);
    if (value < static_cast<int>(RelativeExtremaComparator::less) ||
        value > static_cast<int>(RelativeExtremaComparator::not_equal)) {
        throw std::invalid_argument("relative extrema comparator is invalid");
    }
    return value;
}

void validate_workspace(
    const RelativeExtremaDeviceWorkspace& workspace,
    std::size_t elements)
{
    if (workspace.mask.size() != workspace.flat_indices.size() ||
        workspace.capacity() < elements) {
        throw std::invalid_argument("relative extrema workspace capacity mismatch");
    }
}

struct IsNonzero {
    __host__ __device__ bool operator()(int value) const { return value != 0; }
};

std::size_t compact_flat_indices(
    RelativeExtremaDeviceWorkspace& workspace,
    std::size_t elements)
{
    auto begin = thrust::make_counting_iterator<std::int64_t>(0);
    auto end = begin + static_cast<std::int64_t>(elements);
    thrust::device_ptr<const int> mask(workspace.mask.data());
    thrust::device_ptr<std::int64_t> output(workspace.flat_indices.data());
    const auto output_end = thrust::copy_if(
        thrust::device, begin, end, mask, output, IsNonzero{});
    cuda_utils::synchronize_stream();
    return static_cast<std::size_t>(output_end - output);
}

RelativeExtremaDeviceResult decode_result(
    const RelativeExtremaShape& shape,
    const RelativeExtremaDeviceWorkspace& workspace,
    std::size_t count)
{
    RelativeExtremaDeviceResult result;
    result.extrema_count = count;
    result.coordinates.reserve(shape.rank());
    if (count == 0) {
        for (std::size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            result.coordinates.emplace_back();
        }
        return result;
    }
    std::vector<std::size_t> strides(shape.rank(), 1);
    for (std::size_t dimension = shape.rank(); dimension-- > 1;) {
        strides[dimension - 1] = strides[dimension] * shape.dimensions[dimension];
    }
    for (std::size_t dimension = 0; dimension < shape.rank(); ++dimension) {
        DeviceArray<std::int64_t> coordinate(count);
        cuda_utils::launch_1d_kernel(
            peak_detail::decode_coordinate_kernel, count,
            workspace.flat_indices.data(), coordinate.data(), count,
            strides[dimension], shape.dimensions[dimension]);
        result.coordinates.emplace_back(std::move(coordinate));
    }
    return result;
}

}  // namespace

template <typename T>
RelativeExtremaDeviceResult argrelextrema_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaComparator comparator,
    RelativeExtremaDeviceWorkspace& workspace,
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
    const int comparison = comparator_code(comparator);
    const int selected_axis = normalized_axis(axis, validated.rank());
    const bool clip = clip_mode(mode);
    validate_workspace(workspace, validated.elements);
    if (validated.elements == 0) return decode_result(validated, workspace, 0);

    std::size_t axis_stride = 1;
    for (std::size_t dimension = static_cast<std::size_t>(selected_axis) + 1;
         dimension < validated.rank(); ++dimension) {
        axis_stride *= validated.dimensions[dimension];
    }
    const std::size_t axis_length = validated.dimensions[selected_axis];
    cuda_utils::launch_1d_kernel(
        peak_detail::argrelextrema_mask_kernel<T>, validated.elements,
        data.data(), validated.elements, axis_length, axis_stride,
        comparison, order, clip, workspace.mask.data());
    const std::size_t count = compact_flat_indices(workspace, validated.elements);
    return decode_result(validated, workspace, count);
}

template <typename T>
RelativeExtremaDeviceResult argrelextrema_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaComparator comparator,
    int axis,
    int order,
    const std::string& mode)
{
    RelativeExtremaDeviceWorkspace workspace(data.size());
    return argrelextrema_device(
        data, shape, comparator, workspace, axis, order, mode);
}

template <typename T>
RelativeExtremaDeviceResult argrelmin_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaDeviceWorkspace& workspace,
    int axis,
    int order,
    const std::string& mode)
{
    return argrelextrema_device(
        data, shape, RelativeExtremaComparator::less, workspace, axis, order, mode);
}

template <typename T>
RelativeExtremaDeviceResult argrelmin_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    int axis,
    int order,
    const std::string& mode)
{
    return argrelextrema_device(
        data, shape, RelativeExtremaComparator::less, axis, order, mode);
}

template <typename T>
RelativeExtremaDeviceResult argrelmax_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaDeviceWorkspace& workspace,
    int axis,
    int order,
    const std::string& mode)
{
    return argrelextrema_device(
        data, shape, RelativeExtremaComparator::greater, workspace, axis, order, mode);
}

template <typename T>
RelativeExtremaDeviceResult argrelmax_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    int axis,
    int order,
    const std::string& mode)
{
    return argrelextrema_device(
        data, shape, RelativeExtremaComparator::greater, axis, order, mode);
}

#define INSTANTIATE_PEAK_GPU(T) \
    template RelativeExtremaDeviceResult argrelextrema_device(const DeviceArray<T>&, const RelativeExtremaShape&, RelativeExtremaComparator, RelativeExtremaDeviceWorkspace&, int, int, const std::string&); \
    template RelativeExtremaDeviceResult argrelextrema_device(const DeviceArray<T>&, const RelativeExtremaShape&, RelativeExtremaComparator, int, int, const std::string&); \
    template RelativeExtremaDeviceResult argrelmin_device(const DeviceArray<T>&, const RelativeExtremaShape&, RelativeExtremaDeviceWorkspace&, int, int, const std::string&); \
    template RelativeExtremaDeviceResult argrelmin_device(const DeviceArray<T>&, const RelativeExtremaShape&, int, int, const std::string&); \
    template RelativeExtremaDeviceResult argrelmax_device(const DeviceArray<T>&, const RelativeExtremaShape&, RelativeExtremaDeviceWorkspace&, int, int, const std::string&); \
    template RelativeExtremaDeviceResult argrelmax_device(const DeviceArray<T>&, const RelativeExtremaShape&, int, int, const std::string&)

INSTANTIATE_PEAK_GPU(float);
INSTANTIATE_PEAK_GPU(__half);
INSTANTIATE_PEAK_GPU(std::int32_t);
INSTANTIATE_PEAK_GPU(std::int16_t);
INSTANTIATE_PEAK_GPU(std::int8_t);

#undef INSTANTIATE_PEAK_GPU

}  // namespace cusignal
