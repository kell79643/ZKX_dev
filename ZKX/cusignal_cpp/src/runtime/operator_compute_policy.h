#pragma once

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace cusignal::compute_policy {

template <typename T, bool = std::is_integral_v<T>>
struct ModularUnsigned {
    using type = std::uint32_t;
};

template <typename T>
struct ModularUnsigned<T, true> {
    using type = std::make_unsigned_t<T>;
};

enum class ArithmeticClass : std::uint8_t {
    continuous,
    long_reduction,
    exact_comparison,
    modular_integer,
    index_only,
};

template <typename T>
struct InputTraits {
    static constexpr bool supported =
        std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int8_t>;
    static constexpr bool integral = std::is_integral_v<T>;
    static constexpr bool fp16 = false;

    using ContinuousComputeT = float;
    using ContinuousAccumulatorT = float;
    using ComparisonT = std::conditional_t<integral, std::int32_t, float>;
    using ModularUnsignedT = typename ModularUnsigned<T>::type;

    __host__ __device__ static float load_continuous(T value)
    {
        return static_cast<float>(value);
    }

    __host__ __device__ static ComparisonT load_comparison(T value)
    {
        return static_cast<ComparisonT>(value);
    }
};

template <>
struct InputTraits<__half> {
    static constexpr bool supported = true;
    static constexpr bool integral = false;
    static constexpr bool fp16 = true;

    using ContinuousComputeT = float;
    using ContinuousAccumulatorT = float;
    using ComparisonT = float;
    using ModularUnsignedT = std::uint32_t;

    __host__ __device__ static float load_continuous(__half value)
    {
        return __half2float(value);
    }

    __host__ __device__ static float load_comparison(__half value)
    {
        return __half2float(value);
    }
};

template <typename T>
inline constexpr bool supported_input_v = InputTraits<T>::supported;

template <typename T>
using continuous_compute_t = typename InputTraits<T>::ContinuousComputeT;

template <typename T>
using continuous_accumulator_t = typename InputTraits<T>::ContinuousAccumulatorT;

template <typename T>
using comparison_t = typename InputTraits<T>::ComparisonT;

template <typename T>
using modular_unsigned_t = typename InputTraits<T>::ModularUnsignedT;

// Neumaier compensation keeps long FP32 reductions stable without requiring
// unsupported device FP64.  Call value() only after all terms have been added.
struct CompensatedFloatAccumulator {
    float sum{0.0F};
    float correction{0.0F};

    __host__ __device__ void add(float value)
    {
        const float next = sum + value;
        if (fabsf(sum) >= fabsf(value)) {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }

    __host__ __device__ void add_product(float left, float right)
    {
        add(fmaf(left, right, 0.0F));
    }

    __host__ __device__ float value() const
    {
        return sum + correction;
    }
};

// Explicit FMA removes one rounding from multiply-accumulate without adding
// a second storage lane.  Use this for per-output FIR/DFT loops where latency
// matters; use CompensatedFloatAccumulator for long scalar reductions.
struct FmaFloatAccumulator {
    float sum{0.0F};

    __host__ __device__ void add(float value)
    {
        sum += value;
    }

    __host__ __device__ void add_product(float left, float right)
    {
        sum = fmaf(left, right, sum);
    }

    __host__ __device__ float value() const
    {
        return sum;
    }
};

template <typename T>
__host__ __device__ modular_unsigned_t<T> modular_multiply_add(
    modular_unsigned_t<T> accumulator,
    T left,
    T right)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>,
        "modular multiply-add requires a signed integer input type");
    using U = modular_unsigned_t<T>;
    return static_cast<U>(accumulator +
        static_cast<U>(left) * static_cast<U>(right));
}

template <typename T>
__host__ __device__ T signed_from_modular_bits(modular_unsigned_t<T> raw)
{
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>,
        "modular signed conversion requires a signed integer input type");
    using U = modular_unsigned_t<T>;
    const U signed_max = static_cast<U>(~U{0}) >> 1U;
    if (raw <= signed_max) {
        return static_cast<T>(raw);
    }
    return static_cast<T>(-1 - static_cast<T>(static_cast<U>(~raw)));
}

static_assert(supported_input_v<float>);
static_assert(supported_input_v<__half>);
static_assert(supported_input_v<std::int32_t>);
static_assert(supported_input_v<std::int16_t>);
static_assert(supported_input_v<std::int8_t>);
static_assert(std::is_same_v<continuous_accumulator_t<__half>, float>);
static_assert(std::is_same_v<continuous_accumulator_t<std::int8_t>, float>);
static_assert(std::is_same_v<comparison_t<std::int16_t>, std::int32_t>);

}  // namespace cusignal::compute_policy
