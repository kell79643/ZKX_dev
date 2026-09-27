#pragma once

#include <cusignal/operators/bsplines/bsplines_typed.h>

#include <cmath>
#include <cstddef>

namespace cusignal::bsplines_detail {

template <typename T>
__global__ void cubic_kernel(const T* x, T* y, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float value = detail::BsplineTypePolicy<T>::load(x[index]);
    const float absolute_value = fabsf(value);
    float result = 0.0F;
    if (absolute_value < 1.0F) {
        result = 2.0F / 3.0F -
            0.5F * absolute_value * absolute_value * (2.0F - absolute_value);
    } else if (absolute_value < 2.0F) {
        const float distance = 2.0F - absolute_value;
        result = distance * distance * distance / 6.0F;
    }
    y[index] = detail::BsplineTypePolicy<T>::store(result);
}

template <typename T>
__global__ void gauss_spline_kernel(
    const T* x,
    T* y,
    std::size_t count,
    float reciprocal_twice_variance,
    float normalization)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float value = detail::BsplineTypePolicy<T>::load(x[index]);
    y[index] = detail::BsplineTypePolicy<T>::store(
        normalization * expf(-(value * value) * reciprocal_twice_variance));
}

template <typename T>
__global__ void quadratic_kernel(const T* x, T* y, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float absolute_value = fabsf(detail::BsplineTypePolicy<T>::load(x[index]));
    float result = 0.0F;
    if (absolute_value < 0.5F) {
        result = 0.75F - absolute_value * absolute_value;
    } else if (absolute_value < 1.5F) {
        const float distance = absolute_value - 1.5F;
        result = 0.5F * distance * distance;
    }
    y[index] = detail::BsplineTypePolicy<T>::store(result);
}

}  // namespace cusignal::bsplines_detail
