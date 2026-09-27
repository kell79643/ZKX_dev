#pragma once

#include <cusignal/runtime/device_array.h>
#include <cusignal/backends/fft/fft_interface.h>

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cusignal {

/**
 * @brief 五种业务分量组成的交错复数输入。
 * @details `DemodComplex<float>`对应cuSignal原生ComplexFP32；FP16和三种整数分量
 * 通过批准的FP32合法路径转换后计算，不把复数拆成两个彼此独立的业务算子。
 */
template <typename T>
struct DemodComplex {
    T re;
    T im;
};

namespace detail {

template <typename T>
struct DemodTypePolicy {
    static constexpr bool supported =
        std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int8_t>;

    __host__ __device__ static float load(T value) { return static_cast<float>(value); }
};

template <>
struct DemodTypePolicy<__half> {
    static constexpr bool supported = true;
    __host__ __device__ static float load(__half value) { return __half2float(value); }
};

template <typename T>
inline constexpr bool is_demod_input_v = DemodTypePolicy<T>::supported;

}  // namespace detail

/**
 * @brief `fm_demod`公开shape与axis参数。
 * @param shape 非空、row-major的任意rank输入shape；各维非负。
 * @param axis cuSignal风格axis，范围`[-rank, rank-1]`，默认最后一维。
 */
struct FmDemodOptions {
    std::vector<int> shape;
    int axis = -1;
};

/**
 * @brief Host端验证后的`fm_demod`布局；不持有device scratch或科学库资源。
 * @details 输出保持输入rank和row-major布局，只把选中轴长度变为
 * `max(input_axis_length-1, 0)`。当前kernel使用32位线性索引，因此输入与输出总元素数
 * 均不得超过`INT_MAX`；该平台容量限制会显式抛出异常。
 */
struct FmDemodWorkspace {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int axis_length = 0;
    int output_axis_length = 0;
    int inner_size = 0;
    int outer_size = 0;
    std::size_t input_count = 0;
    std::size_t output_count = 0;

    FmDemodWorkspace() = default;
    explicit FmDemodWorkspace(const FmDemodOptions& options) { reset(options); }

    void reset(const FmDemodOptions& options);
    [[nodiscard]] std::size_t input_size() const noexcept { return input_count; }
    [[nodiscard]] std::size_t output_size() const noexcept { return output_count; }
};

/**
 * @brief 在CPU上执行完整cuSignal `angle -> unwrap -> diff` reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量；输出固定FP32。
 * @param x Host row-major交错复数输入，size必须等于workspace输入shape乘积。
 * @param workspace 已验证的任意rank布局。
 * @return 一个Host FP32数组，shape为`workspace.output_shape`。
 * @throws std::invalid_argument workspace或输入shape不匹配。
 * @details 每个输出先分别计算相邻复数样点的FP32 angle，再执行CuPy默认周期的局部
 * unwrap修正规则；精确`+pi`保留为`+pi`，精确`-pi`保留为`-pi`。该实现不调用GPU、
 * 不创建stream、不读取device结果，也不以`conj(prev)*curr`乘积替代angle，避免大分量
 * 乘积溢出改变语义。
 */
template <typename T>
std::vector<float> fm_demod_typed_cpu(
    const std::vector<DemodComplex<T>>& x,
    const FmDemodWorkspace& workspace);

/**
 * @brief cuSignal `fm_demod`一维CPU便利入口，五类型复数分量均输出FP32。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x Host一维交错复数输入。
 * @return Host FP32数组，长度为`max(x.size()-1, 0)`。
 * @throws std::invalid_argument 输入容量无法由正式workspace表达时抛出异常。
 * @details 等价于shape=`{x.size()}`、axis=-1并委托正式CPU reference；不创建GPU
 * stream、不读取device结果，输出shape与相位差语义和cuSignal保持一致。
 */
template <typename T>
std::vector<float> fm_demod_typed_cpu(const std::vector<DemodComplex<T>>& x);

/** @brief 二维便利重载；委托任意rank正式CPU reference。 */
template <typename T>
std::vector<float> fm_demod_typed_cpu(
    const std::vector<DemodComplex<T>>& x,
    int rows,
    int cols,
    int axis = -1);

/**
 * @brief 在GPU上执行任意rank完整cuSignal `fm_demod`正式路径。
 * @tparam T 五种真实业务复数分量；所有device中间角度、unwrap修正和输出均为FP32。
 * @param x 调用者持有的device row-major交错复数输入。
 * @param y 调用者按workspace预分配的device FP32单数组输出。
 * @param workspace Host-only shape/axis/索引布局；无device scratch。
 * @throws std::invalid_argument shape、axis、容量、输出size或workspace一致性非法。
 * @details 默认stream启动一个pointwise kernel，无隐式H2D/D2H、同步、FFT、RAND或
 * device FP64；每个输出独立计算CuPy修正后的局部相位差，调用者读取前负责同步。
 */
template <typename T>
void fm_demod_device(
    const DeviceArray<DemodComplex<T>>& x,
    DeviceArray<float>& y,
    const FmDemodWorkspace& workspace);

/**
 * @brief cuSignal `fm_demod`的`ComplexFloat`正式ABI，可直接承接FFT/Hilbert输出。
 * @param x 调用者持有的device ComplexFP32 row-major输入。
 * @param y 调用者按workspace预分配的device FP32输出。
 * @param workspace Host-only shape/axis/索引布局，不拥有device scratch。
 * @throws std::invalid_argument shape、axis、容量、输出size或workspace不一致时抛出异常。
 * @details 默认stream执行，与模板正式入口共享FP32 kernel；无隐式传输、同步或device
 * FP64，输出shape及`angle -> unwrap -> diff`语义与cuSignal保持一致。
 */
void fm_demod_device(
    const DeviceArray<ComplexFloat>& x,
    DeviceArray<float>& y,
    const FmDemodWorkspace& workspace);

/**
 * @brief cuSignal `fm_demod`五类型一维GPU便利入口，输出固定FP32。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x 调用者持有的device一维交错复数输入。
 * @param y 调用者预分配的device FP32输出，长度为`max(x.size()-1, 0)`。
 * @throws std::invalid_argument 输入或输出容量不符合自动构造的workspace时抛出异常。
 * @details 自动建立Host-only workspace并在默认stream委托任意rank正式入口；不执行隐式
 * 传输或同步，输出dtype、shape和相位差语义与cuSignal保持一致。
 */
template <typename T>
void fm_demod_device(
    const DeviceArray<DemodComplex<T>>& x,
    DeviceArray<float>& y);

/** @brief `ComplexFloat`正式ABI的一维便利重载。 */
void fm_demod_device(
    const DeviceArray<ComplexFloat>& x,
    DeviceArray<float>& y);

/** @brief 二维便利重载；委托任意rank正式GPU入口。 */
template <typename T>
void fm_demod_device(
    const DeviceArray<DemodComplex<T>>& x,
    DeviceArray<float>& y,
    int rows,
    int cols,
    int axis = -1);

}  // namespace cusignal
