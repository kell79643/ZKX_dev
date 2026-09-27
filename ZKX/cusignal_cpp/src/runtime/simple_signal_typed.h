#pragma once

#include <cusignal/runtime/device_array.h>
#include "operator_compute_policy.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace cusignal {
namespace detail {

template <typename T>
struct SimpleSignalTypePolicy {
    static constexpr bool supported = compute_policy::supported_input_v<T>;
    using ComputeT = compute_policy::continuous_compute_t<T>;
    using AccumulatorT = compute_policy::continuous_accumulator_t<T>;
    __host__ __device__ static ComputeT load(T value)
    {
        return compute_policy::InputTraits<T>::load_continuous(value);
    }
    __host__ __device__ static T store(float value)
    {
        if constexpr (std::is_integral_v<T>) {
            const float low = static_cast<float>(std::numeric_limits<T>::lowest());
            const float high = static_cast<float>(std::numeric_limits<T>::max());
            value = fminf(fmaxf(value, low), high);
        }
        return static_cast<T>(value);
    }
};

template <>
struct SimpleSignalTypePolicy<__half> {
    static constexpr bool supported = true;
    using ComputeT = compute_policy::continuous_compute_t<__half>;
    using AccumulatorT = compute_policy::continuous_accumulator_t<__half>;
    __host__ __device__ static ComputeT load(__half value)
    {
        return compute_policy::InputTraits<__half>::load_continuous(value);
    }
    __host__ __device__ static __half store(float value) { return __float2half_rn(value); }
};

template <typename T>
inline constexpr bool is_simple_signal_input_v = SimpleSignalTypePolicy<T>::supported;

}  // namespace detail

// Public interface predicate used by wrappers and callers.  FP64 remains a
// valid result-storage dtype, but it is deliberately absent from this set.
template <typename T>
inline constexpr bool is_formal_gpu_business_input_v =
    detail::is_simple_signal_input_v<T>;

template <typename T>
constexpr void require_formal_gpu_business_input()
{
    static_assert(is_formal_gpu_business_input_v<T>,
        "unsupported formal GPU business input: expected FP32, FP16, INT32, INT16 or INT8; FP64 output storage must cross an explicit Host narrowing boundary before reuse");
}

}  // namespace cusignal
