/**
 * @file bsplines_typed.h
 * @brief B-spline 五类型 CPU/GPU 公开接口。
 *
 * 输入和输出遵循 cuSignal ElementwiseKernel 的同型契约：FP32、FP16、INT32、
 * INT16、INT8 均返回相同元素类型。所有 device 计算统一提升到 FP32；FP16 不直接
 * 调用超越函数，gauss_spline 使用 `__half2float -> expf/sqrtf -> __float2half_rn`。
 */

#pragma once

#include <cusignal/runtime/device_array.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cusignal {
namespace detail {

template <typename T>
struct BsplineTypePolicy {
    static constexpr bool supported =
        std::is_same_v<T, float> ||
        std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> ||
        std::is_same_v<T, std::int8_t>;
    using ComputeT = float;

    __host__ __device__ static ComputeT load(T value)
    {
        return static_cast<float>(value);
    }

    __host__ __device__ static T store(ComputeT value)
    {
        return static_cast<T>(value);
    }
};

template <>
struct BsplineTypePolicy<__half> {
    static constexpr bool supported = true;
    using ComputeT = float;

    __host__ __device__ static ComputeT load(__half value)
    {
        return __half2float(value);
    }

    __host__ __device__ static __half store(ComputeT value)
    {
        return __float2half_rn(value);
    }
};

template <typename T>
inline constexpr bool is_bspline_input_v = BsplineTypePolicy<T>::supported;

template <typename T>
inline float bspline_abs(T value)
{
    return std::fabs(BsplineTypePolicy<T>::load(value));
}

template <typename T>
inline T cubic_value(T value)
{
    const float x = BsplineTypePolicy<T>::load(value);
    const float ax = std::fabs(x);
    float result = 0.0F;
    if (ax < 1.0F) {
        result = 2.0F / 3.0F - 0.5F * ax * ax * (2.0F - ax);
    } else if (ax < 2.0F) {
        const float distance = 2.0F - ax;
        result = distance * distance * distance / 6.0F;
    }
    return BsplineTypePolicy<T>::store(result);
}

template <typename T>
inline T quadratic_value(T value)
{
    const float ax = bspline_abs(value);
    float result = 0.0F;
    if (ax < 0.5F) {
        result = 0.75F - ax * ax;
    } else if (ax < 1.5F) {
        const float distance = ax - 1.5F;
        result = 0.5F * distance * distance;
    }
    return BsplineTypePolicy<T>::store(result);
}

template <typename T>
inline T gauss_spline_value(T value, int n)
{
    const float signsq = static_cast<float>(n + 1) / 12.0F;
    const float r_signsq = 0.5F / signsq;
    constexpr float pi = 3.14159265358979323846F;
    const float norm = 1.0F / std::sqrt(2.0F * pi * signsq);
    const float x = BsplineTypePolicy<T>::load(value);
    return BsplineTypePolicy<T>::store(norm * std::exp(-(x * x) * r_signsq));
}

}  // namespace detail

/**
 * @brief 在 CPU 上计算三次 B-spline，作为正式 GPU 实现的数值 reference。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；返回元素类型与输入一致。
 * @param x 任意逻辑 rank 的连续 row-major host 输入；shape 由调用者持有，允许为空且不修改输入。
 * @return 与 `x` 同 shape、展平存储的 host vector；内部提升为 FP32，按 `|x| < 1`、`1 <= |x| < 2`
 * 两段多项式计算，`|x| >= 2` 输出零，再按 T 的转换规则写回。
 * @details 本函数不分配设备内存、不启动 kernel、不调用 CUDA API、dlfft 或 dlrand。
 * 整数输出按 C++ 浮点到整数转换截断，这与同类型 GPU 输出契约一致。
 * @throws std::bad_alloc host 输出分配失败。
 * @note 对齐 cuSignal cubic 逐元素公式；无 stream 或 workspace，整数同型输出是项目扩展。
 */
template <typename T>
std::vector<T> cubic_cpu(const std::vector<T>& x);

/**
 * @brief 已降级的 CPU 兼容别名；不属于任务 E 的正式 GPU 主路径。
 * @deprecated 新业务代码使用 `cubic_device`；CPU 对照显式使用 `cubic_cpu`。
 * @tparam T 与 `cubic_cpu` 相同的五类型集合。
 * @param x 一维 host 输入。
 * @return 直接返回 `cubic_cpu(x)`，不会隐式 H2D、启动 GPU 或调用科学计算库。
 */
template <typename T>
[[deprecated("CPU compatibility alias; use cubic_device for GPU execution or cubic_cpu for reference")]]
std::vector<T> cubic(const std::vector<T>& x);

/**
 * @brief 在 ZQ500 GPU 上计算三次 B-spline，是该算子的唯一正式五类型 GPU 入口。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；输入输出必须同型。
 * @param x 任意逻辑 rank、连续 row-major、设备驻留的只读输入。
 * @param y 调用者预分配的同 shape 展平设备输出，长度必须等于 `x.size()`；允许两个数组均为空。
 * @throws std::invalid_argument `x`、`y` 长度不一致时抛出。
 * @details 默认 stream 上启动 template custom kernel，使用 FP32 分段多项式并写回 T；
 * 不创建 workspace，不调用 dlfft/dlrand，不执行隐式 H2D/D2H。函数只检查 launch 错误，
 * 调用者在读取 `y` 前负责同步。
 * @note 对齐 cuSignal cubic 的逐元素实数公式；C++ 接口不接受广播或 Python array-like，
 * 整数输入按项目批准的同型量化扩展返回整数。
 */
template <typename T>
void cubic_device(const DeviceArray<T>& x, DeviceArray<T>& y);

/**
 * @brief 在 CPU 上计算 Gaussian spline，作为正式 GPU 实现的数值 reference。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；返回元素类型与输入一致。
 * @param x 任意逻辑 rank 的连续 row-major host 输入；shape 由调用者持有，允许为空。
 * @param n 非负 spline 阶数，方差固定为 `(n + 1) / 12`。
 * @return 与 `x` 同 shape、展平存储的 host vector；FP32 `sqrt/exp` 计算后写回 T。
 * @throws std::invalid_argument `n < 0` 时抛出，包括空输入场景。
 * @details 不调用 CUDA API、dlfft 或 dlrand；整数结果遵循与 GPU 相同的 T 写回规则。
 * @note 对齐 cuSignal gauss_spline 方差与逐元素公式；无 stream 或 workspace。
 */
template <typename T>
std::vector<T> gauss_spline_cpu(const std::vector<T>& x, int n);

/**
 * @brief 已降级的 Gaussian spline CPU 兼容别名；不属于正式 GPU 主路径。
 * @deprecated 新业务代码使用 `gauss_spline_device`；CPU 对照显式使用
 * `gauss_spline_cpu`。
 * @tparam T 与 `gauss_spline_cpu` 相同的五类型集合。
 * @param x 一维 host 输入。
 * @param n 非负 spline 阶数。
 * @return 直接返回 `gauss_spline_cpu(x, n)`，不会隐式调用 GPU。
 * @throws std::invalid_argument `n < 0` 时由 CPU reference 抛出。
 */
template <typename T>
[[deprecated("CPU compatibility alias; use gauss_spline_device for GPU execution or gauss_spline_cpu for reference")]]
std::vector<T> gauss_spline(const std::vector<T>& x, int n);

/**
 * @brief 在 ZQ500 GPU 上计算 Gaussian spline，是该算子的唯一正式五类型 GPU 入口。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；输入输出必须同型。
 * @param x 任意逻辑 rank、连续 row-major、设备驻留的只读输入。
 * @param y 调用者预分配的同 shape 展平设备输出，长度必须等于 `x.size()`。
 * @param n 非负 spline 阶数，方差为 `(n + 1) / 12`。
 * @throws std::invalid_argument 长度不一致或 `n < 0` 时抛出；空输入也不绕过 n 校验。
 * @details 默认 stream 上启动 template custom kernel，使用 FP32 `sqrtf/expf` 后写回 T；
 * FP16 经 `__half2float`/`__float2half_rn` 转换。无 workspace、dlfft、dlrand 或隐式传输，
 * 调用者读取 `y` 前负责同步。
 * @note 对齐 cuSignal gauss_spline 的 `(n+1)/12` 方差；本项目把 cuSignal 原本浮点输出
 * 扩展为五类型同型输出，并明确拒绝负 n。
 */
template <typename T>
void gauss_spline_device(const DeviceArray<T>& x, DeviceArray<T>& y, int n);

/**
 * @brief 在 CPU 上计算二次 B-spline，作为正式 GPU 实现的数值 reference。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；返回元素类型与输入一致。
 * @param x 任意逻辑 rank 的连续 row-major host 输入；shape 由调用者持有，允许为空。
 * @return 与 `x` 同 shape、展平存储的 host vector；内部提升为 FP32，按 `|x| < 0.5`、
 * `0.5 <= |x| < 1.5` 两段多项式计算，`|x| >= 1.5` 输出零，再写回 T。
 * @details 不分配设备资源、不调用 CUDA API、dlfft 或 dlrand。
 * @throws std::bad_alloc host 输出分配失败。
 * @note 对齐 cuSignal quadratic 逐元素公式；无 stream 或 workspace，整数同型输出是项目扩展。
 */
template <typename T>
std::vector<T> quadratic_cpu(const std::vector<T>& x);

/**
 * @brief 已降级的二次 B-spline CPU 兼容别名；不属于正式 GPU 主路径。
 * @deprecated 新业务代码使用 `quadratic_device`；CPU 对照显式使用 `quadratic_cpu`。
 * @tparam T 与 `quadratic_cpu` 相同的五类型集合。
 * @param x 一维 host 输入。
 * @return 直接返回 `quadratic_cpu(x)`，不会隐式调用 GPU 或科学计算库。
 */
template <typename T>
[[deprecated("CPU compatibility alias; use quadratic_device for GPU execution or quadratic_cpu for reference")]]
std::vector<T> quadratic(const std::vector<T>& x);

/**
 * @brief 在 ZQ500 GPU 上计算二次 B-spline，是该算子的唯一正式五类型 GPU 入口。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；输入输出必须同型。
 * @param x 任意逻辑 rank、连续 row-major、设备驻留的只读输入。
 * @param y 调用者预分配的同 shape 展平设备输出，长度必须等于 `x.size()`；允许均为空。
 * @throws std::invalid_argument `x`、`y` 长度不一致时抛出。
 * @details 默认 stream 上启动 template custom kernel，使用 FP32 分段多项式后写回 T；
 * 无 workspace、dlfft、dlrand 或隐式 H2D/D2H，调用者读取 `y` 前负责同步。
 * @note 对齐 cuSignal quadratic 的逐元素实数公式；整数输入属于项目五类型同型扩展，
 * 不是 cuSignal 的隐式浮点输出规则。
 */
template <typename T>
void quadratic_device(const DeviceArray<T>& x, DeviceArray<T>& y);

extern template std::vector<float> cubic_cpu(const std::vector<float>&);
extern template std::vector<__half> cubic_cpu(const std::vector<__half>&);
extern template std::vector<std::int32_t> cubic_cpu(const std::vector<std::int32_t>&);
extern template std::vector<std::int16_t> cubic_cpu(const std::vector<std::int16_t>&);
extern template std::vector<std::int8_t> cubic_cpu(const std::vector<std::int8_t>&);

extern template std::vector<float> gauss_spline_cpu(const std::vector<float>&, int);
extern template std::vector<__half> gauss_spline_cpu(const std::vector<__half>&, int);
extern template std::vector<std::int32_t> gauss_spline_cpu(const std::vector<std::int32_t>&, int);
extern template std::vector<std::int16_t> gauss_spline_cpu(const std::vector<std::int16_t>&, int);
extern template std::vector<std::int8_t> gauss_spline_cpu(const std::vector<std::int8_t>&, int);

extern template std::vector<float> quadratic_cpu(const std::vector<float>&);
extern template std::vector<__half> quadratic_cpu(const std::vector<__half>&);
extern template std::vector<std::int32_t> quadratic_cpu(const std::vector<std::int32_t>&);
extern template std::vector<std::int16_t> quadratic_cpu(const std::vector<std::int16_t>&);
extern template std::vector<std::int8_t> quadratic_cpu(const std::vector<std::int8_t>&);

extern template void cubic_device(const DeviceArray<float>&, DeviceArray<float>&);
extern template void cubic_device(const DeviceArray<__half>&, DeviceArray<__half>&);
extern template void cubic_device(const DeviceArray<std::int32_t>&, DeviceArray<std::int32_t>&);
extern template void cubic_device(const DeviceArray<std::int16_t>&, DeviceArray<std::int16_t>&);
extern template void cubic_device(const DeviceArray<std::int8_t>&, DeviceArray<std::int8_t>&);

extern template void gauss_spline_device(const DeviceArray<float>&, DeviceArray<float>&, int);
extern template void gauss_spline_device(const DeviceArray<__half>&, DeviceArray<__half>&, int);
extern template void gauss_spline_device(const DeviceArray<std::int32_t>&, DeviceArray<std::int32_t>&, int);
extern template void gauss_spline_device(const DeviceArray<std::int16_t>&, DeviceArray<std::int16_t>&, int);
extern template void gauss_spline_device(const DeviceArray<std::int8_t>&, DeviceArray<std::int8_t>&, int);

extern template void quadratic_device(const DeviceArray<float>&, DeviceArray<float>&);
extern template void quadratic_device(const DeviceArray<__half>&, DeviceArray<__half>&);
extern template void quadratic_device(const DeviceArray<std::int32_t>&, DeviceArray<std::int32_t>&);
extern template void quadratic_device(const DeviceArray<std::int16_t>&, DeviceArray<std::int16_t>&);
extern template void quadratic_device(const DeviceArray<std::int8_t>&, DeviceArray<std::int8_t>&);

}  // namespace cusignal
