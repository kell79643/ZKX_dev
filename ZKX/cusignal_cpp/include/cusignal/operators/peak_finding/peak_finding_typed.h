#pragma once

#include <cusignal/runtime/device_array.h>
#include "simple_signal_typed.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cusignal {

/** @brief 固定版cuSignal后端可稳定映射的六种relative-extrema比较器。 */
enum class RelativeExtremaComparator : std::uint8_t {
    less,
    greater,
    less_equal,
    greater_equal,
    equal,
    not_equal
};

/**
 * @brief row-major非标量输入shape。
 * @details `dimensions`至少包含一维；允许零长度维；`elements`由reset进行溢出检查后
 * 计算，调用者不得修改为与dimensions不一致的值。
 */
struct RelativeExtremaShape {
    std::vector<std::size_t> dimensions;
    std::size_t elements = 0;

    RelativeExtremaShape() = default;
    explicit RelativeExtremaShape(std::vector<std::size_t> dimensions_in)
    {
        reset(std::move(dimensions_in));
    }

    void reset(std::vector<std::size_t> dimensions_in);
    [[nodiscard]] std::size_t rank() const noexcept { return dimensions.size(); }
};

/**
 * @brief CPU reference返回的cuSignal式坐标tuple。
 * @details `coordinates.size()==input.rank()`；每个数组均为INT64、长度相同，并按
 * row-major mask的`nonzero`发现顺序排列。一维结果仍保留一个数组的tuple结构。
 */
struct RelativeExtremaHostResult {
    std::vector<std::vector<std::int64_t>> coordinates;

    [[nodiscard]] std::size_t count() const noexcept
    {
        return coordinates.empty() ? 0 : coordinates.front().size();
    }
};

/**
 * @brief GPU正式入口返回的cuSignal式device坐标tuple。
 * @details 每个`DeviceArray`均精确分配`extrema_count`个INT64元素；结果由调用者持有，
 * default stream上的坐标解码完成后才可读取。封装在返回前因compaction取得动态长度而
 * 发生一次Host同步；返回后不再隐式同步或D2H。
 */
struct RelativeExtremaDeviceResult {
    std::vector<DeviceArray<std::int64_t>> coordinates;
    std::size_t extrema_count = 0;
};

/**
 * @brief GPU调用者可复用workspace。
 * @details `mask`保存每个输入元素的0/1判定，`flat_indices`保存row-major compact结果；
 * 两者均由本对象持有，容量不足时正式入口抛出`std::invalid_argument`，不会静默重分配。
 * 同一workspace不得被并发调用复用。
 */
struct RelativeExtremaDeviceWorkspace {
    DeviceArray<int> mask;
    DeviceArray<std::int64_t> flat_indices;

    RelativeExtremaDeviceWorkspace() = default;
    explicit RelativeExtremaDeviceWorkspace(std::size_t capacity) { reset(capacity); }
    void reset(std::size_t capacity);
    [[nodiscard]] std::size_t capacity() const noexcept { return mask.size(); }
};

/**
 * @brief CPU上计算`cusignal.argrelextrema`的五类型独立reference。
 * @tparam T 真实FP32、FP16、INT32、INT16或INT8输入；FP32/INT32原生直接比较，
 * FP16精确提升FP32，INT16/INT8精确提升INT32，不发生舍入、饱和或设备FP64。
 * @param data row-major Host输入，元素数必须等于shape.elements。
 * @param shape 任意非标量rank及各维长度。
 * @param comparator less/greater/less_equal/greater_equal/equal/not_equal之一。
 * @param axis 选中轴；所有rank统一校验范围并按CuPy规则归一化有效负轴。
 * 固定版cuSignal二维backend对非零axis的误分支视为缺陷，不进入正式接口。
 * @param order 两侧各比较的距离数，必须>=1。
 * @param mode 精确`clip`使用端点截断，精确`raise`抛`std::logic_error`，其余字符串按
 * 固定版cuSignal的wrap分支执行完整模回绕。
 * @return rank个Host INT64坐标数组，顺序、shape与`cp.nonzero`一致。
 * @throws std::invalid_argument 标量shape、shape/输入不符、axis/order/comparator非法。
 * @details CPU reference不创建GPU stream或device workspace；输出坐标dtype、顺序及
 * 动态shape保持cuSignal/CuPy语义。
 */
template <typename T>
RelativeExtremaHostResult argrelextrema_typed_cpu(
    const std::vector<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaComparator comparator,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/** @brief CPU strict-less便利入口，对应`cusignal.argrelmin`，其余契约同上。 */
template <typename T>
RelativeExtremaHostResult argrelmin_typed_cpu(
    const std::vector<T>& data,
    const RelativeExtremaShape& shape,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/** @brief CPU strict-greater便利入口，对应`cusignal.argrelmax`，其余契约同上。 */
template <typename T>
RelativeExtremaHostResult argrelmax_typed_cpu(
    const std::vector<T>& data,
    const RelativeExtremaShape& shape,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/**
 * @brief GPU上计算`cusignal.argrelextrema`的五类型正式workspace入口。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实输入。
 * @param data 调用者持有的row-major device输入。
 * @param shape 任意非标量rank及各维长度。
 * @param comparator cuSignal支持的六种比较器之一。
 * @param workspace 调用者复用的mask与flat-index暂存。
 * @param axis cuSignal风格axis。
 * @param order 两侧比较距离数。
 * @param mode `clip`、`raise`或wrap分支字符串。
 * @throws std::invalid_argument shape、axis、order或workspace容量非法时抛出异常。
 * @details 调用者直接提供真实T device输入；GPU执行任意rank两侧比较、mask、row-major
 * compaction和坐标解码。INT32不经过FP32，所有device运算均无double/FP64。compaction
 * 为取得动态输出长度会同步一次；精确长度结果数组由返回对象持有，输入与workspace仍归
 * 调用者。默认stream执行；六比较器、axis/order/mode、空输入和异常与cuSignal及CPU
 * reference一致。
 */
template <typename T>
RelativeExtremaDeviceResult argrelextrema_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaComparator comparator,
    RelativeExtremaDeviceWorkspace& workspace,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/**
 * @brief GPU convenience入口。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实输入。
 * @param data 调用者持有的row-major device输入。
 * @param shape 任意非标量rank及各维长度。
 * @param comparator cuSignal支持的六种比较器之一。
 * @param axis cuSignal风格axis。
 * @param order 两侧比较距离数。
 * @param mode `clip`、`raise`或wrap分支字符串。
 * @throws std::invalid_argument shape、axis、order或容量非法时抛出异常。
 * @details 内部按data.size创建一次性workspace，数学和返回契约与正式workspace入口完全
 * 相同；默认stream执行FP32/整数kernel，性能测试应包含该分配，持续调用优先复用上方
 * workspace入口，输出保持cuSignal动态INT64坐标语义。
 */
template <typename T>
RelativeExtremaDeviceResult argrelextrema_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaComparator comparator,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/** @brief GPU strict-less workspace入口，对应`cusignal.argrelmin`。 */
template <typename T>
RelativeExtremaDeviceResult argrelmin_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaDeviceWorkspace& workspace,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/** @brief GPU strict-less convenience入口，对应`cusignal.argrelmin`。 */
template <typename T>
RelativeExtremaDeviceResult argrelmin_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/** @brief GPU strict-greater workspace入口，对应`cusignal.argrelmax`。 */
template <typename T>
RelativeExtremaDeviceResult argrelmax_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    RelativeExtremaDeviceWorkspace& workspace,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

/** @brief GPU strict-greater convenience入口，对应`cusignal.argrelmax`。 */
template <typename T>
RelativeExtremaDeviceResult argrelmax_device(
    const DeviceArray<T>& data,
    const RelativeExtremaShape& shape,
    int axis = 0,
    int order = 1,
    const std::string& mode = "clip");

}  // namespace cusignal
