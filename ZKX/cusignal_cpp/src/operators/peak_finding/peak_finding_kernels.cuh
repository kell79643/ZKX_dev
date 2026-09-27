#pragma once

#include "simple_signal_typed.h"

#include <cstddef>
#include <cstdint>

namespace cusignal::peak_detail {

template <typename T>
struct ComparisonPolicy {
    using type = compute_policy::comparison_t<T>;
    __device__ static type load(T value)
    {
        return compute_policy::InputTraits<T>::load_comparison(value);
    }
};

template <typename Value>
__device__ bool compare(Value left, Value right, int comparator)
{
    switch (comparator) {
    case 0: return left < right;
    case 1: return left > right;
    case 2: return left <= right;
    case 3: return left >= right;
    case 4: return left == right;
    case 5: return left != right;
    default: return false;
    }
}

__device__ inline std::size_t wrapped_plus(
    std::size_t coordinate, std::size_t distance, std::size_t length)
{
    const std::size_t offset = distance % length;
    return offset >= length - coordinate ? offset - (length - coordinate)
                                         : coordinate + offset;
}

__device__ inline std::size_t wrapped_minus(
    std::size_t coordinate, std::size_t distance, std::size_t length)
{
    const std::size_t offset = distance % length;
    return coordinate >= offset ? coordinate - offset : length - (offset - coordinate);
}

template <typename T>
__global__ void argrelextrema_mask_kernel(
    const T* data,
    std::size_t elements,
    std::size_t axis_length,
    std::size_t axis_stride,
    int comparator,
    int order,
    bool clip,
    int* mask)
{
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= elements) return;
    const std::size_t coordinate = (flat / axis_stride) % axis_length;
    const auto center = ComparisonPolicy<T>::load(data[flat]);
    bool extrema = true;
    for (int step = 1; extrema; ++step) {
        const std::size_t distance = static_cast<std::size_t>(step);
        const std::size_t plus_coordinate = clip
            ? (distance >= axis_length - coordinate
                ? axis_length - 1 : coordinate + distance)
            : wrapped_plus(coordinate, distance, axis_length);
        const std::size_t minus_coordinate = clip
            ? (distance > coordinate ? 0 : coordinate - distance)
            : wrapped_minus(coordinate, distance, axis_length);
        const std::size_t plus = plus_coordinate >= coordinate
            ? flat + (plus_coordinate - coordinate) * axis_stride
            : flat - (coordinate - plus_coordinate) * axis_stride;
        const std::size_t minus = minus_coordinate >= coordinate
            ? flat + (minus_coordinate - coordinate) * axis_stride
            : flat - (coordinate - minus_coordinate) * axis_stride;
        extrema = compare(center, ComparisonPolicy<T>::load(data[plus]), comparator) &&
                  compare(center, ComparisonPolicy<T>::load(data[minus]), comparator);
        if (step == order) break;
    }
    mask[flat] = extrema ? 1 : 0;
}

__global__ void decode_coordinate_kernel(
    const std::int64_t* flat_indices,
    std::int64_t* coordinate,
    std::size_t count,
    std::size_t dimension_stride,
    std::size_t dimension_length)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t flat = static_cast<std::size_t>(flat_indices[index]);
    coordinate[index] = static_cast<std::int64_t>(
        (flat / dimension_stride) % dimension_length);
}

}  // namespace cusignal::peak_detail
