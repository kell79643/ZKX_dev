# cusignal_cpp_argrelextrema 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：原始学习版本](#v1原始学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | 五类型、任意 rank、CPU reference、GPU mask/compaction/坐标解码正式接口 |

## V1：原始学习版本（fdd55ac）

### 1. 版本身份与只读边界

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交主题：`test(archive): 增加任务结果范围治理审计`
- 相关源码 dirty 状态：`cusignal_cpp/src/peak_finding/`、`cusignal_cpp/test/signal_processing/e3_peak_radar_type_smoke.cu`、`cusignal_cpp/contracts/operator_contracts.cmake` 均 clean；仓库整体 `git status --short` 为空。
- 读取方式：所有正式证据均通过 `git -C ZKX show fdd55ac8415d70379eb38a2c299047f90bcf0a41:<路径>` 读取；没有修改 `ZKX/cusignal_cpp`。
- 验证边界：本阶段没有构建、运行 CUDA 或连接 ZQ500；测试章节只解释该提交中已经存在的测试源码，不声称获得新的运行结果。仓库文档把最新正式路径标为 `source-changed-pending-zq500`。

### 2. 文件角色与准确定位

下列每条定位均绑定同一完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`。

| 层次 | 共享根目录 `ZKX_dev/` 下路径 | 符号 | 该提交行号 |
| --- | --- | --- | --- |
| 公共类型/API | `ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.h` | `RelativeExtremaComparator`、shape/result/workspace、CPU/GPU 声明 | 12-216 |
| CPU helper | `ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.cpp` | shape、axis、mode、比较、边界、坐标 helper | 8-140 |
| CPU 核心 | 同上 | `argrelextrema_typed_cpu` | 142-197 |
| CPU 便利入口/实例化 | 同上 | `argrelmin_typed_cpu`、`argrelmax_typed_cpu`、`INSTANTIATE_PEAK_CPU` | 199-236 |
| GPU device 核心 | `ZKX/cusignal_cpp/src/peak_finding/peak_finding_kernels.cuh` | `ComparisonPolicy`、`argrelextrema_mask_kernel`、`decode_coordinate_kernel` | 8-100 |
| GPU host helper | `ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.cu` | validation、Thrust compaction、`decode_result` | 15-118 |
| GPU 正式入口 | 同上 | `argrelextrema_device` workspace/convenience | 120-169 |
| GPU 便利入口/实例化 | 同上 | `argrelmin_device`、`argrelmax_device`、`INSTANTIATE_PEAK_GPU` | 171-237 |
| 直接测试 | `ZKX/cusignal_cpp/test/signal_processing/e3_peak_radar_type_smoke.cu` | 坐标比较与 `run_type<T>` 中 argrelextrema 切片 | 92-250、561-571 |

旧的 `peak_finding.h/.cpp/peak_finding_cuda.cu` 属于历史 flat/count 接口；当前公开契约和 E3 证据都指向 `peak_finding_typed.*`，因此本 V1 不把旧包装误写成正式核心实现。

### 3. 总调用链与职责

```text
CPU:
argrelextrema_typed_cpu<T>
  → shape/axis/order/comparator/mode 校验
  → 遍历 row-major flat 索引
  → 计算 axis coordinate 与 ±step 邻居
  → 六比较器之一逐值比较
  → append_coordinates 解码全部维坐标

GPU:
argrelextrema_device<T>
  → host 侧相同契约校验 + workspace 校验
  → argrelextrema_mask_kernel<T> 生成 int 0/1 mask
  → thrust::copy_if 压缩 row-major flat 索引
  → 一次 stream 同步取得动态 count
  → 每个维度启动 decode_coordinate_kernel
  → 返回 rank 个精确长度 INT64 DeviceArray
```

CPU reference 与 GPU 实现共享同一数学判据，但职责不同：CPU 直接在发现候选时追加坐标；GPU 先生成固定长度 mask，再压缩动态索引，最后逐维解码。

## 4. 公共接口逐行语义块

本章按该提交中 `peak_finding_typed.h` 的原顺序解释相关代码。每个代码块都给出完整 SHA、路径、符号和行号。

### 4.1 头文件保护、依赖与命名空间

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.h`，文件入口与 `namespace cusignal`，第 1-12 行。

```cpp
#pragma once

#include "cuda_utils/device_array.h"
#include "cuda_utils/simple_signal_typed.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cusignal {
```

- `#pragma once` 防止同一翻译单元重复包含头文件。
- 两个项目头分别提供 `DeviceArray` 和五类型计算策略；标准头提供 `size_t`、定宽整数、字符串、移动语义和动态数组。
- `namespace cusignal` 把后续公开类型与函数放入项目命名空间。

### 4.2 六比较器枚举

定位：同 SHA、同文件，`RelativeExtremaComparator`，第 14-22 行。

```cpp
/** @brief 固定版cuSignal后端可稳定映射的六种relative-extrema比较器。 */
enum class RelativeExtremaComparator : std::uint8_t {
    less,
    greater,
    less_equal,
    greater_equal,
    equal,
    not_equal
};
```

- Doxygen 注释明确枚举对应固定版 cuSignal 后端的六种稳定比较器。
- `enum class` 提供强类型作用域；底层 `uint8_t` 足以保存 0-5。
- 声明顺序有契约意义：GPU `comparator_code` 把枚举转为整数，device `switch` 用 0-5 解释为同一顺序。

### 4.3 row-major shape 对象

定位：同 SHA、同文件，`RelativeExtremaShape`，第 24-41 行。

```cpp
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
```

- 文档规定非标量、允许零长度维、`elements` 必须由 `reset` 计算。
- `dimensions` 保存各维长度，`elements` 保存乘积；两者是公开成员，所以 CPU/GPU 入口会重新构造并检查是否被调用者篡改。
- 默认构造暂不建立有效 shape；`explicit` 构造防止 vector 隐式转换，并用 `std::move` 把所有权交给 `reset`。
- `[[nodiscard]]` 提醒调用者不要忽略 `rank()` 返回值；`const noexcept` 表示不改对象且不抛异常。

### 4.4 CPU 坐标 tuple

定位：同 SHA、同文件，`RelativeExtremaHostResult`，第 43-55 行。

```cpp
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
```

- 外层 vector 模拟 Python tuple 的 rank 个坐标数组，内层固定 `int64_t`。
- `count()` 读取第一个坐标数组长度；若外层为空则返回 0。有效非标量结果通常外层不空，即使没有极值也保留 rank 个空数组。

### 4.5 GPU 结果与可复用 workspace

定位：同 SHA、同文件，`RelativeExtremaDeviceResult`、`RelativeExtremaDeviceWorkspace`，第 57-82 行。

```cpp
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
```

- GPU 结果仍是 rank 个 INT64 device 数组，并单独记录动态数量。
- compaction 必须知道输出尾指针，因此正式入口有一次 host 同步；坐标解码随后仍在 default stream。
- workspace 的 `mask` 是输入长度的 0/1 数组，`flat_indices` 是同容量的 INT64 压缩缓冲区。
- 默认构造容量为 0；显式容量构造调用 `reset`；`capacity()` 以 mask 长度为准，入口还会验证两个缓冲区长度一致。

### 4.6 CPU 正式入口契约与声明

定位：同 SHA、同文件，`argrelextrema_typed_cpu`，第 84-108 行。

```cpp
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
```

- 注释逐项冻结五类型计算策略、row-major 输入、六比较器、任意 rank、负轴、完整模回绕和 INT64 tuple 输出。
- `const std::vector<T>&` 与 `const RelativeExtremaShape&` 避免复制且禁止修改输入。
- 默认参数与 Python 一致；比较器没有默认值，调用者必须明确选择。
- `T` 只在五种业务输入类型上显式实例化；窄类型扩展是精确值转换，不改变比较结果。

### 4.7 CPU strict min/max 便利声明

定位：同 SHA、同文件，`argrelmin_typed_cpu`、`argrelmax_typed_cpu`，第 110-127 行。

```cpp
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
```

- 两个函数省略 comparator 参数，分别固定为 strict-less 和 strict-greater。
- 其他参数、默认值和返回类型保持一致；它们只是 API 便利层，不是新的核心算法。

### 4.8 GPU workspace 正式入口

定位：同 SHA、同文件，`argrelextrema_device` workspace 重载，第 129-154 行。

```cpp
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
```

- workspace 以非常量引用传入，因为 kernel 和 Thrust 会写入其中；输入和 shape 保持只读。
- GPU 工作顺序被接口注释明确为 mask → compaction → coordinate decode。
- `INT32` 保持整数比较，device 端不用 FP64；唯一明确同步点来自动态长度 compaction。

### 4.9 GPU convenience 重载

定位：同 SHA、同文件，`argrelextrema_device` convenience 重载，第 156-177 行。

```cpp
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
```

- 该重载不接收 workspace，会在实现内部按 `data.size()` 临时创建。
- 数学结果相同，但每次调用增加两个 device 缓冲区分配；持续调用应使用 workspace 重载。

### 4.10 GPU min/max 四个便利声明与命名空间结束

定位：同 SHA、同文件，`argrelmin_device`、`argrelmax_device` 与文件结尾，第 178-216 行。

```cpp
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
```

- `argrelmin_device` 和 `argrelmax_device` 各有 workspace 与 convenience 两个重载。
- 四个声明都保留 axis/order/mode，并分别固定 strict-less/strict-greater。
- 最后一行闭合第 12 行开始的 `cusignal` 命名空间；至此公共头文件每一行相关代码均已覆盖。

## 5. CPU reference 逐行语义块

### 5.1 依赖与匿名命名空间

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.cpp`，文件入口，第 1-9 行。

```cpp
#include "peak_finding_typed.h"

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cusignal {
namespace {
```

- 项目头提供所有公开声明；标准头分别提供数值上限、异常、类型特征和移动工具。
- 外层进入公开命名空间，内层匿名命名空间让 helper 只在本翻译单元可见。

### 5.2 shape 乘积溢出检查

定位：同 SHA、同文件，`checked_product`，第 11-17 行。

```cpp
std::size_t checked_product(std::size_t left, std::size_t right)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument("relative extrema shape exceeds size_t capacity");
    }
    return left * right;
}
```

- 先排除 `right==0`，避免除零；再用 `max/right` 预判乘法是否溢出。
- 溢出时抛参数异常，否则安全返回乘积。零长度维会把元素总数变为 0。

### 5.3 shape 一致性与防篡改验证

定位：同 SHA、同文件，`same_shape`、`validated_shape`，第 19-31 行。

```cpp
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
```

- `same_shape` 同时比较维度向量和缓存的元素数。
- `validated_shape` 从原始 dimensions 重建可信对象；若公开成员 `elements` 被改写，重建结果不一致并抛错。
- 返回新对象使后续计算只依赖重新验证过的 shape。

### 5.4 负轴归一化和范围检查

定位：同 SHA、同文件，`normalized_axis`，第 33-42 行。

```cpp
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
```

- rank 转成有符号 64 位，避免与负 axis 混合比较时发生无符号提升。
- 有效负轴加 rank，例如 rank 3 的 `-1` 变 2；仍越界则抛异常。
- 返回前转回 int，shape 构造已保证 rank 不超过 `INT_MAX`。

### 5.5 mode 与 comparator 校验

定位：同 SHA、同文件，`clip_mode`、`validate_comparator`，第 44-59 行。

```cpp
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
```

- `raise` 精确抛 `logic_error`；`clip` 返回 true；任何其他字符串返回 false并按固定版行为进入 wrap。
- 强类型枚举仍可通过强制转换制造非法值，因此显式检查闭区间 `[less,not_equal]`。

### 5.6 五类型比较值加载与六分支比较

定位：同 SHA、同文件，`comparison_value`、`compare_values`，第 61-79 行。

```cpp
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
```

- `comparison_value` 把真实输入加载为比较计算类型：FP16→FP32、INT16/INT8→INT32，FP32/INT32 原生。
- `compare_values` 的 `switch` 实现六种谓词；每个 case 直接返回，合法枚举不会落到末尾。
- 末尾 `false` 是防御性兜底，正式入口已在调用前验证 comparator。

### 5.7 完整模回绕与端点裁剪

定位：同 SHA、同文件，`wrapped_plus`、`wrapped_minus`、`clipped_plus`、`clipped_minus`，第 81-102 行。

```cpp
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
```

- wrap 先做 `distance % length`，因此 `order` 大于轴长时仍是真正周期回绕。
- 三元表达式规避无符号加减溢出：正向越过尾端时从 0 继续，负向越过首端时从尾端回退。
- clip 正向固定到 `length-1`，负向固定到 0。

### 5.8 flat 索引解码为 rank 个坐标

定位：同 SHA、同文件，`append_coordinates` 与匿名命名空间结束，第 104-117 行。

```cpp
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
```

- 多行签名传入 flat、shape 和可写结果。
- 逆序遍历维度，`flat % length` 得到当前维坐标，`flat /= length` 去掉该维。
- 每个维度的 vector 同步追加一次，所以坐标数组长度一致；匿名命名空间随后闭合。

### 5.9 shape reset 的全部约束

定位：同 SHA、同文件，`RelativeExtremaShape::reset`，第 119-134 行。

```cpp
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
```

- 空 dimensions 表示标量，正式契约拒绝。
- rank 必须能装入 axis 使用的 int；元素乘积逐维做溢出检查，并限制 flat 索引能装入 INT64。
- 所有检查通过后才移动 dimensions 并写入 elements，避免对象处于半更新状态。

### 5.10 workspace reset

定位：同 SHA、同文件，`RelativeExtremaDeviceWorkspace::reset`，第 136-140 行。

```cpp
void RelativeExtremaDeviceWorkspace::reset(std::size_t capacity)
{
    mask.reset(capacity);
    flat_indices.reset(capacity);
}
```

- 两个 device 缓冲区被重置为相同容量；后续 GPU 入口仍会再次验证一致性。

### 5.11 CPU 正式入口：签名与前置验证

定位：同 SHA、同文件，`argrelextrema_typed_cpu`，第 142-165 行。

```cpp
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
```

- 模板签名与头文件一致；`static_assert` 在编译期把 T 限制为五种简单信号输入。
- 依次验证 shape 防篡改、输入元素数、order、comparator、axis 和 mode。
- 结果先 resize 为 rank 个坐标数组；空输入立即返回 rank 个空数组，保持 Python tuple rank。

### 5.12 CPU 正式入口：axis stride

定位：同 SHA、同文件，`argrelextrema_typed_cpu`，第 167-173 行。

```cpp
    std::size_t axis_stride = 1;
    for (std::size_t dimension = static_cast<std::size_t>(selected_axis) + 1;
         dimension < validated.rank(); ++dimension) {
        axis_stride *= validated.dimensions[dimension];
    }
    const std::size_t axis_length = validated.dimensions[selected_axis];
```

- row-major 中沿选中轴移动一个坐标，需要跨过其右侧所有维度的乘积。
- 循环从 `selected_axis+1` 到最后一维计算 `axis_stride`；随后读取轴长。
- shape 已验证且 `elements>0`，所以所选轴长度不可能为 0，模运算安全。

### 5.13 CPU 正式入口：逐 flat 候选与邻域循环

定位：同 SHA、同文件，`argrelextrema_typed_cpu`，第 174-196 行。

```cpp
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
```

- 外循环严格按 row-major flat 顺序扫描，所以输出发现顺序与 `cp.nonzero` 一致。
- 轴坐标公式 `(flat/stride)%length` 去掉右侧维度再对轴长取模。
- 中心加载为精确比较类型，`extrema=true` 是全称逻辑与单位元。
- 内循环从 step 1 开始；条件中包含 `extrema`，一旦失败立即短路，不再检查更远邻居。
- `plus_coordinate/minus_coordinate` 根据 clip 选择端点截断或完整模回绕。
- 两个三元表达式把轴坐标差乘 stride，再对当前 flat 加减，其他维坐标不变。
- 两侧比较用 `&&` 合取；只有都为真才继续。到 `step==order` 时显式 break，避免无限循环。
- 候选最终为真时解码并追加所有维坐标；所有 flat 完成后返回。

### 5.14 CPU min/max 便利实现

定位：同 SHA、同文件，`argrelmin_typed_cpu`、`argrelmax_typed_cpu`，第 199-221 行。

```cpp
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
```

- 两个完整函数只转发参数，分别注入 `less` 和 `greater`；CPU 核心没有重复实现。

### 5.15 CPU 五类型显式实例化与文件结束

定位：同 SHA、同文件，`INSTANTIATE_PEAK_CPU`，第 223-236 行。

```cpp
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
```

- 反斜杠把宏定义续到下一物理行；宏一次生成通用、min、max 三个显式实例化声明。
- 五次调用覆盖 FP32、FP16、INT32、INT16、INT8，共 15 个 CPU 实例。
- `#undef` 防止宏泄漏，最后闭合 `cusignal`；CPU 文件相关代码到此最后一行全部覆盖。

## 6. GPU device kernel 逐行语义块

### 6.1 kernel 头文件入口

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/peak_finding/peak_finding_kernels.cuh`，文件入口，第 1-8 行。

```cpp
#pragma once

#include "cuda_utils/simple_signal_typed.h"

#include <cstddef>
#include <cstdint>

namespace cusignal::peak_detail {
```

- 头文件保护后引入五类型策略、size_t 和 INT64。
- C++17 嵌套命名空间把 device 细节放在 `cusignal::peak_detail`，与公开 API 隔离。

### 6.2 device 比较加载策略

定位：同 SHA、同文件，`ComparisonPolicy<T>`，第 10-17 行。

```cpp
template <typename T>
struct ComparisonPolicy {
    using type = compute_policy::comparison_t<T>;
    __device__ static type load(T value)
    {
        return compute_policy::InputTraits<T>::load_comparison(value);
    }
};
```

- `comparison_t<T>` 决定 device 比较类型；`using type` 建立别名。
- 静态 device 函数无需实例对象，按与 CPU 相同的 InputTraits 加载，保证五类型语义一致。

### 6.3 device 六比较器

定位：同 SHA、同文件，`compare<Value>`，第 19-31 行。

```cpp
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
```

- 整数操作码与公共枚举顺序完全一致。
- `default:false` 防御非法码；host wrapper 在 kernel 启动前已经验证范围。

### 6.4 device 完整模回绕

定位：同 SHA、同文件，`wrapped_plus`、`wrapped_minus`，第 33-46 行。

```cpp
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
```

- device 版本逐行对应 CPU 公式；`inline` 建议编译器内联。
- `distance % length` 是相较固定版一/二维 kernel 的关键修复，任意大 order 都保持周期索引。

### 6.5 mask kernel 签名与线程归属

定位：同 SHA、同文件，`argrelextrema_mask_kernel<T>`，第 48-63 行。

```cpp
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
```

- 每个 CUDA 线程负责一个 row-major flat 输入元素，而不是一个轴切片。
- 参数提供轴长与 stride，因此同一一维 grid 能处理任意 rank。
- 越界线程立即返回；有效线程计算轴坐标、加载中心并初始化候选标志。

### 6.6 mask kernel 邻域、地址与比较

定位：同 SHA、同文件，`argrelextrema_mask_kernel<T>`，第 64-84 行。

```cpp
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
```

- 循环条件含 `extrema`，失败后线程提前结束；distance 把正 step 转为无符号索引量。
- clip 的两个嵌套三元表达式分别复制末端与首端；wrap 调用完整模 helper。
- plus/minus flat 地址公式与 CPU 完全相同，只改变指定轴坐标。
- 两个 device 比较用 `&&` 合取；达到 order 后 break。
- 最终把 bool 显式写成 int 1/0，供 Thrust stencil 使用。

### 6.7 flat 坐标解码 kernel 与文件结束

定位：同 SHA、同文件，`decode_coordinate_kernel`，第 86-100 行。

```cpp
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
```

- 每个线程负责一个已压缩候选的某一维坐标。
- `flat_indices` 是 INT64；shape 构造已保证可安全转回 size_t 范围内的合法非负 flat。
- 坐标公式与 CPU 的逐维除模等价，结果显式转回 INT64。
- 最后一行闭合 detail 命名空间；kernel 文件全部相关行已覆盖。

## 7. GPU host wrapper 逐行语义块

### 7.1 依赖与匿名命名空间

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.cu`，文件入口，第 1-16 行。

```cpp
#include "peak_finding_typed.h"
#include "peak_finding_kernels.cuh"

#include "cuda_utils/cuda_utils.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace cusignal {
namespace {
```

- 两个 peak 头分别提供公共 API 和 device kernel；`cuda_utils` 提供统一 kernel 启动与同步。
- 四个 Thrust 头支持 `copy_if`、device 指针、device execution policy 和 INT64 counting iterator。
- 标准头提供异常、移动和 host vector；匿名命名空间限制 host helper 可见性。

### 7.2 GPU wrapper 的 shape 和 axis 验证

定位：同 SHA、同文件，`same_shape`、`validated_shape`、`normalized_axis`，第 18-41 行。

```cpp
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
```

- 这些 host helper 与 CPU 文件逐行同构，保证 CPU/GPU 对 shape 防篡改和负轴归一化作相同判断。
- 它们没有放进共享头，避免把内部实现暴露为公共 API。

### 7.3 GPU mode、比较器和 workspace 验证

定位：同 SHA、同文件，`clip_mode`、`comparator_code`、`validate_workspace`，第 43-69 行。

```cpp
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
```

- mode 语义与 CPU 完全一致。
- comparator 验证后直接返回 0-5 操作码，传入 device kernel。
- workspace 要求两个数组大小相等且容量至少覆盖 elements；大于输入的可复用容量允许，小容量或不一致立即抛错。

### 7.4 Thrust stencil 谓词

定位：同 SHA、同文件，`IsNonzero`，第 71-73 行。

```cpp
struct IsNonzero {
    __host__ __device__ bool operator()(int value) const { return value != 0; }
};
```

- 函数对象同时可在 host/device 编译；`operator()` 使实例可调用。
- Thrust `copy_if` 用它读取 mask stencil，保留值非零的 counting index。

### 7.5 row-major flat 索引压缩与同步

定位：同 SHA、同文件，`compact_flat_indices`，第 75-87 行。

```cpp
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
```

- counting iterator 懒生成 `[0,elements)` INT64 flat 索引，不额外存一份输入索引数组。
- mask 和 output 原始 device 指针包装成 Thrust 指针。
- `copy_if` 按输入顺序把 mask 非零位置的 counting value 写入 `flat_indices`，保持 row-major 发现顺序。
- 显式同步确保返回的迭代器差可用于 host 动态分配；差值就是 extrema count。

### 7.6 动态结果对象、空结果与 strides

定位：同 SHA、同文件，`decode_result`，第 89-107 行。

```cpp
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
```

- 结果记录 count，并只 reserve 外层容量，不提前构造坐标数组。
- count 为 0 时仍 emplace rank 个空 DeviceArray，保持 tuple rank，然后直接返回。
- 非空时创建全 1 strides；逆序循环计算每个维度右侧长度乘积。

### 7.7 每维坐标解码与匿名命名空间结束

定位：同 SHA、同文件，`decode_result`，第 108-118 行。

```cpp
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
```

- 每个维度精确分配 count 个 INT64，并启动一次一维解码 kernel。
- kernel 参数传 flat buffer、当前输出、count、该维 stride 和长度。
- `std::move` 把 DeviceArray 所有权移入外层 vector，不复制 device 数据。
- 返回前没有第二次显式同步；default stream 顺序保证之后读取时解码已排在 compaction 后。

### 7.8 GPU 正式 workspace 入口：签名与校验

定位：同 SHA、同文件，`argrelextrema_device<T>` workspace 重载，第 120-141 行。

```cpp
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
```

- 编译期和运行期检查顺序对应 CPU reference，并额外验证 workspace。
- comparator 转成 device 操作码；axis 归一化，mode 转成 clip bool。
- 空输入不启动 kernel，直接构造 rank 个空 device 坐标数组。

### 7.9 GPU 正式入口：stride、mask、压缩与解码

定位：同 SHA、同文件，`argrelextrema_device<T>` workspace 重载，第 143-155 行。

```cpp
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
```

- host 计算与 CPU 相同的 axis stride 和 length。
- `launch_1d_kernel` 的第二参数是工作元素数；后续参数按 mask kernel 签名传入。
- mask kernel 完成后，Thrust 压缩并同步取得 count；最后逐维解码并返回动态 tuple。

### 7.10 GPU convenience 重载

定位：同 SHA、同文件，`argrelextrema_device<T>` convenience 重载，第 157-169 行。

```cpp
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
```

- 按输入大小创建一次性 workspace，然后调用正式重载；局部 workspace 在返回后析构。
- 返回对象独立持有精确坐标数组，不依赖 workspace 生命周期。

### 7.11 GPU strict-min 两个重载

定位：同 SHA、同文件，`argrelmin_device<T>`，第 171-194 行。

```cpp
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
```

- workspace 与 convenience 版本都只注入 `less`，分别解析到相应通用重载。

### 7.12 GPU strict-max 两个重载

定位：同 SHA、同文件，`argrelmax_device<T>`，第 196-219 行。

```cpp
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
```

- 两个 max 包装固定 `greater`，没有复制任何 mask 或 compaction 逻辑。

### 7.13 GPU 五类型显式实例化与最后一行

定位：同 SHA、同文件，`INSTANTIATE_PEAK_GPU`，第 221-237 行。

```cpp
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
```

- 宏为每种 T 生成通用函数两个重载、min 两个重载、max 两个重载，共 6 个实例。
- 五类型合计 30 个 GPU 显式实例；`#undef` 清理宏。
- 最后一行闭合 `cusignal`；至此 GPU host 文件已解释到最后一行。

## 8. 直接测试逐行语义块

测试文件还覆盖 radar 等其他算子；本章只摘录 `argrelextrema` 直接相关的连续切片和五类型入口，不读取或解释无关测试主体。

### 8.1 GPU/CPU 坐标精确对照 helper

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/test/signal_processing/e3_peak_radar_type_smoke.cu`，`same_coordinates`、`exact_coordinates`，第 92-110 行。

```cpp
bool same_coordinates(
    const RelativeExtremaDeviceResult& actual,
    const RelativeExtremaHostResult& expected)
{
    if (actual.coordinates.size() != expected.coordinates.size() ||
        actual.extrema_count != expected.count()) return false;
    for (std::size_t dimension = 0; dimension < expected.coordinates.size(); ++dimension) {
        if (!test_evidence::f_observe_real(actual.coordinates[dimension].to_host(),
                expected.coordinates[dimension], 0.0)) return false;
    }
    return true;
}

bool exact_coordinates(
    const RelativeExtremaHostResult& actual,
    const std::vector<std::vector<std::int64_t>>& expected)
{
    return actual.coordinates == expected;
}
```

- `same_coordinates` 先比较 tuple rank 与动态 count，再逐维把 GPU 坐标 D2H，以容差 0 精确对照 CPU。
- 任一维失败立即返回 false；全部通过返回 true。
- `exact_coordinates` 用 vector 精确相等检查 CPU reference 对手写期望值。

### 8.2 五类型测试初始化和一维 strict max

定位：同 SHA、同测试，`run_type<T>` 的 argrelextrema 切片，第 112-126 行。

```cpp
template<class T>int run_type(const char*name)
{
    bool ok[6]{};
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::argrelextrema);
    const std::vector<double> host_peak{0, 1, 4, 1, 0};
    std::vector<T> peak;
    for (double v : host_peak) peak.push_back(convert<T>(static_cast<float>(v)));
    const RelativeExtremaShape peak_shape({5});
    auto d_peak = DeviceArray<T>::from_host(peak);
    RelativeExtremaDeviceWorkspace peak_workspace(peak.size());
    const auto extrema_ref = argrelmax_typed_cpu(peak, peak_shape);
    const auto extrema_gpu = argrelmax_device(
        d_peak, peak_shape, peak_workspace);
    bool peak_ok = exact_coordinates(extrema_ref, {{2}}) &&
                   same_coordinates(extrema_gpu, extrema_ref);
```

- `run_type<T>` 会由 main 对五种 T 调用；`ok[0]` 对应 argrelextrema。
- `[0,1,4,1,0]` 转为 T，同时创建 host/device 输入和复用 workspace。
- 默认 strict-max 应只返回索引 2；CPU 先对手写答案，GPU 再对 CPU。

### 8.3 二维 axis 与负轴

定位：同 SHA、同测试，`run_type<T>`，第 128-150 行。

```cpp
    const RelativeExtremaShape matrix_shape({3, 4});
    const std::vector<float> matrix_values{
        1, 3, 1, 2,
        5, 1, 4, 0,
        0, 2, 0, 2};
    std::vector<T> matrix;
    for (float value : matrix_values) matrix.push_back(convert<T>(value));
    auto d_matrix = DeviceArray<T>::from_host(matrix);
    RelativeExtremaDeviceWorkspace matrix_workspace(matrix.size());
    const auto matrix_ref = argrelextrema_typed_cpu(
        matrix, matrix_shape, RelativeExtremaComparator::greater, 1, 1, "clip");
    const auto matrix_gpu = argrelextrema_device(
        d_matrix, matrix_shape, RelativeExtremaComparator::greater,
        matrix_workspace, 1, 1, "clip");
    const auto matrix_negative_two_ref = argrelextrema_typed_cpu(
        matrix, matrix_shape, RelativeExtremaComparator::greater, -2, 1, "clip");
    const auto matrix_negative_two_gpu = argrelextrema_device(
        d_matrix, matrix_shape, RelativeExtremaComparator::greater,
        matrix_workspace, -2, 1, "clip");
    peak_ok = peak_ok && exact_coordinates(matrix_ref, {{0, 1, 2}, {1, 2, 1}}) &&
              same_coordinates(matrix_gpu, matrix_ref) &&
              exact_coordinates(matrix_negative_two_ref, {{1, 1}, {0, 2}}) &&
              same_coordinates(matrix_negative_two_gpu, matrix_negative_two_ref);
```

- shape 3×4，axis 1 检查每行；预期三个坐标 `(0,1),(1,2),(2,1)`。
- `-2` 在 rank 2 中归一化为 axis 0，预期两个坐标 `(1,0),(1,2)`。
- 两种轴都同时核对 CPU 手算答案与 GPU 等价性。

### 8.4 三维、非严格比较、wrap 与超轴长 order

定位：同 SHA、同测试，`run_type<T>`，第 152-167 行。

```cpp
    const RelativeExtremaShape cube_shape({2, 2, 3});
    const std::vector<float> cube_values{
        1, 5, 2, 4, 3, 1,
        2, 0, 6, 7, 1, 3};
    std::vector<T> cube;
    for (float value : cube_values) cube.push_back(convert<T>(value));
    auto d_cube = DeviceArray<T>::from_host(cube);
    RelativeExtremaDeviceWorkspace cube_workspace(cube.size());
    const auto cube_ref = argrelextrema_typed_cpu(
        cube, cube_shape, RelativeExtremaComparator::greater_equal, -1, 5, "wrap");
    const auto cube_gpu = argrelextrema_device(
        d_cube, cube_shape, RelativeExtremaComparator::greater_equal,
        cube_workspace, -1, 5, "wrap");
    peak_ok = peak_ok &&
        exact_coordinates(cube_ref, {{0, 0, 1, 1}, {0, 1, 0, 1}, {1, 0, 2, 0}}) &&
        same_coordinates(cube_gpu, cube_ref);
```

- 最后一轴长度只有 3，但 order 为 5，专门验证完整模回绕。
- `greater_equal` 覆盖非严格平台语义；输出是三个 INT64 坐标数组，每个长度 4。

### 8.5 六比较器与 strict-min 便利入口

定位：同 SHA、同测试，`run_type<T>`，第 169-186 行。

```cpp
    const RelativeExtremaComparator comparators[]{
        RelativeExtremaComparator::less,
        RelativeExtremaComparator::greater,
        RelativeExtremaComparator::less_equal,
        RelativeExtremaComparator::greater_equal,
        RelativeExtremaComparator::equal,
        RelativeExtremaComparator::not_equal};
    for (RelativeExtremaComparator comparator : comparators) {
        const auto cpu = argrelextrema_typed_cpu(
            peak, peak_shape, comparator, 0, 1, "clip");
        const auto gpu = argrelextrema_device(
            d_peak, peak_shape, comparator, peak_workspace, 0, 1, "clip");
        peak_ok = peak_ok && same_coordinates(gpu, cpu);
    }
    const auto minimum_ref = argrelmin_typed_cpu(peak, peak_shape);
    const auto minimum_gpu = argrelmin_device(d_peak, peak_shape, peak_workspace);
    peak_ok = peak_ok && exact_coordinates(minimum_ref, {{}}) &&
              same_coordinates(minimum_gpu, minimum_ref);
```

- 枚举数组逐一覆盖 0-5，要求 CPU/GPU 坐标精确一致。
- 对 `[0,1,4,1,0]` 的默认 clip strict-min，端点会与自身严格比较，内部没有谷，所以结果为空。

### 8.6 未知 mode 走 wrap

定位：同 SHA、同测试，`run_type<T>`，第 188-196 行。

```cpp
    const std::vector<T> unknown_mode_values{
        convert<T>(4), convert<T>(1), convert<T>(0), convert<T>(1), convert<T>(3)};
    auto d_unknown = DeviceArray<T>::from_host(unknown_mode_values);
    const auto unknown_ref = argrelmax_typed_cpu(
        unknown_mode_values, peak_shape, 0, 1, "unknown-fixed-version-wrap");
    const auto unknown_gpu = argrelmax_device(
        d_unknown, peak_shape, 0, 1, "unknown-fixed-version-wrap");
    peak_ok = peak_ok && exact_coordinates(unknown_ref, {{0}}) &&
              same_coordinates(unknown_gpu, unknown_ref);
```

- 非 `clip`、非 `raise` 字符串按固定版 cuSignal 进入 wrap；首元素 4 与末元素 3、次元素 1 比较后成为 strict max。

### 8.7 零长度维与 tuple rank

定位：同 SHA、同测试，`run_type<T>`，第 198-206 行。

```cpp
    const RelativeExtremaShape empty_shape({2, 0, 3});
    const std::vector<T> empty;
    DeviceArray<T> d_empty;
    RelativeExtremaDeviceWorkspace empty_workspace(0);
    const auto empty_ref = argrelmax_typed_cpu(empty, empty_shape, -1, 1, "clip");
    const auto empty_gpu = argrelmax_device(
        d_empty, empty_shape, empty_workspace, -1, 1, "clip");
    peak_ok = peak_ok && exact_coordinates(empty_ref, {{}, {}, {}}) &&
              same_coordinates(empty_gpu, empty_ref);
```

- shape rank 为 3、elements 为 0；返回仍必须是三个空坐标数组，而不是空外层 tuple。

### 8.8 异常边界

定位：同 SHA、同测试，`run_type<T>`，第 208-237 行。

```cpp
    bool rejects_raise_cpu = false;
    bool rejects_raise_gpu = false;
    bool rejects_order = false;
    bool rejects_axis = false;
    bool rejects_matrix_axis = false;
    bool rejects_matrix_axis_gpu = false;
    bool rejects_small_workspace = false;
    bool rejects_scalar_shape = false;
    try { (void)argrelmax_typed_cpu(peak, peak_shape, 0, 1, "raise"); }
    catch (const std::logic_error&) { rejects_raise_cpu = true; }
    try { (void)argrelmax_device(d_peak, peak_shape, peak_workspace, 0, 1, "raise"); }
    catch (const std::logic_error&) { rejects_raise_gpu = true; }
    try { (void)argrelmax_typed_cpu(peak, peak_shape, 0, 0, "clip"); }
    catch (const std::invalid_argument&) { rejects_order = true; }
    try { (void)argrelmax_device(d_peak, peak_shape, peak_workspace, 2, 1, "clip"); }
    catch (const std::invalid_argument&) { rejects_axis = true; }
    try { (void)argrelmax_typed_cpu(matrix, matrix_shape, 2, 1, "clip"); }
    catch (const std::invalid_argument&) { rejects_matrix_axis = true; }
    try { (void)argrelmax_device(d_matrix, matrix_shape, matrix_workspace, 2, 1, "clip"); }
    catch (const std::invalid_argument&) { rejects_matrix_axis_gpu = true; }
    try {
        RelativeExtremaDeviceWorkspace small_workspace(peak.size() - 1);
        (void)argrelmax_device(d_peak, peak_shape, small_workspace);
    } catch (const std::invalid_argument&) { rejects_small_workspace = true; }
    try { (void)RelativeExtremaShape(std::vector<std::size_t>{}); }
    catch (const std::invalid_argument&) { rejects_scalar_shape = true; }
    peak_ok = peak_ok && rejects_raise_cpu && rejects_raise_gpu &&
              rejects_order && rejects_axis && rejects_matrix_axis &&
              rejects_matrix_axis_gpu && rejects_small_workspace &&
              rejects_scalar_shape;
```

- 八个布尔标志分别验证 CPU/GPU raise、order 0、1D/2D 越界 axis、workspace 不足和标量 shape。
- `(void)` 明确丢弃返回值；每个 catch 只接受预期异常类型。
- 最后所有拒绝行为都必须为真，才能保留 `peak_ok`。

### 8.9 INT32 超过 $2^{24}$ 的精确比较

定位：同 SHA、同测试，`run_type<T>`，第 239-250 行。

```cpp
    if constexpr (std::is_same_v<T, std::int32_t>) {
        const std::vector<T> precise{
            static_cast<T>(16777216), static_cast<T>(16777217),
            static_cast<T>(16777216)};
        const RelativeExtremaShape precise_shape({3});
        auto d_precise = DeviceArray<T>::from_host(precise);
        const auto precise_ref = argrelmax_typed_cpu(precise, precise_shape);
        const auto precise_gpu = argrelmax_device(d_precise, precise_shape);
        peak_ok = peak_ok && exact_coordinates(precise_ref, {{1}}) &&
                  same_coordinates(precise_gpu, precise_ref);
    }
    ok[0] = peak_ok;
```

- `if constexpr` 只为 INT32 实例编译该块。
- `16777217` 无法由 FP32 精确表示；中点仍必须被识别，证明 INT32 没有错误转 FP32。
- 最终把本算子的累计结果写入 `ok[0]`；相关连续测试切片至此结束。

### 8.10 证据记录和五类型 main

定位：同 SHA、同测试，`run_type<T>` 证据尾部与 `main`，第 561-571 行；其中第 563-567 行是同一共享数组里的其他算子证据，仅为保持原始语义块完整而保留，不在本文展开。

```cpp
    bool evidence[6]{
        ev::record<T,int64_t>(td::OperatorId::argrelextrema,extrema_ref.count(),ok[0]),
        ambgfun_evidence,
        ev::record_mixed<T>(td::OperatorId::ca_cfar,cfar_ref.threshold.size(),ok[2]),
        ev::record<T,double>(td::OperatorId::cfar_alpha,1,ok[3]),
        pulse_compression_evidence,
        pulse_doppler_evidence};
    static const char*labels[6]={"argrelextrema","ambgfun","ca_cfar","cfar_alpha","pulse_compression","pulse_doppler"};int passed=0;std::cout<<"[E3][peak_radar]["<<name<<"]";for(int i=0;i<6;++i){passed+=evidence[i]?1:0;std::cout<<' '<<labels[i]<<'='<<evidence[i];}std::cout<<" pass="<<passed<<" fail="<<(6-passed)<<'\n';return passed;
}

int main(){int passed=0;passed+=run_type<float>("fp32");passed+=run_type<__half>("fp16");passed+=run_type<int32_t>("int32");passed+=run_type<int16_t>("int16");passed+=run_type<int8_t>("int8");std::cout<<"[E3][peak_radar] total=30 pass="<<passed<<" fail="<<(30-passed)<<'\n';return passed==30?0:1;}
```

- `evidence[0]` 记录 argrelextrema 的输入 T、固定 INT64 输出、数量和 `ok[0]`。
- 单行输出循环打印六个共享测试项并返回通过数；随后闭合 `run_type`。
- `main` 依次实例化五种业务类型；每种共享测试包含 6 项，总计 30。全部通过才返回进程码 0。
- 这里只说明测试意图；当前阶段没有实际运行，因此不能把源码中的 `total=30` 当作本次验证结果。

## 9. 数学、Python、CPU 与 GPU 四方映射

所有 C++ 定位均绑定 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $M[n]=\bigwedge_{k=1}^r(C(x[n],x[b(n+k)])\land C(x[n],x[b(n-k)]))$ | `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py`，`_boolrelextrema`，基准 62-72 | `ZKX/cusignal_cpp/src/peak_finding/peak_finding_typed.cpp`，`argrelextrema_typed_cpu`，174-193 | `peak_finding_kernels.cuh`，`argrelextrema_mask_kernel`，64-83 | 数学相同；C++ CPU/GPU 在候选失败后按元素短路 |
| 六比较器 | Python `_modedict` 与 CUDA 函数表 | `compare_values`，68-79 | `compare`，19-31；`comparator_code`，51-59 | C++ 用强类型枚举和显式范围校验 |
| axis 切片 | `cp.take(...,axis=axis)`；固定 2D kernel 对非零 axis 有缺陷 | `normalized_axis` 与 `axis_stride`，33-42、167-173 | host 同样归一化；kernel 用 stride，typed.cu 32-41、143-150 | 正式 C++ 有意按 CuPy 正确语义扩展所有 rank，不复制固定 2D 缺陷 |
| `clip` | 正负索引裁到端点 | `clipped_plus/minus`，94-102 | mask kernel 内嵌裁剪，66-74 | 等价实现 |
| `wrap` | 高维 CuPy 使用完整模；固定 1D/2D kernel 只加减一次轴长 | `wrapped_plus/minus`，81-92 | `wrapped_plus/minus`，33-46 | C++ 有意采用完整模，修复 `order>axis_length` 风险 |
| 布尔掩码 | helper 返回 bool array | CPU 不保留完整 mask，发现候选即追加 | `workspace.mask` 保存 int 0/1 | CPU 节省 mask；GPU 需要 stencil 供 compaction |
| `cp.nonzero` row-major tuple | `cp.nonzero(results)` | `append_coordinates`，104-115 | `thrust::copy_if` + `decode_coordinate_kernel` | 输出均为 rank 个 INT64 坐标数组；GPU 动态 count 需一次同步 |
| strict min/max | `argrelmin/argrelmax` 固定 `cp.less/cp.greater` | 199-221 | typed.cu 171-219 | 便利层只固定 comparator，核心不变 |

## 10. 数据结构、类型分发与内存布局

### 10.1 五类型比较策略

| 输入 T | 实际比较类型 | 精确性依据 |
| --- | --- | --- |
| `float` | FP32 | 原生加载与比较 |
| `__half` | FP32 | 所有 FP16 有限值可被 FP32 精确表示 |
| `int32_t` | INT32 | 不经过 FP32，保留超过 $2^{24}$ 的相邻整数差异 |
| `int16_t` | INT32 | 符号扩展精确 |
| `int8_t` | INT32 | 符号扩展精确 |

输出与输入 T 无关，坐标始终为 `int64_t`。源码没有 FP64 输入实例，也没有 device FP64 比较路径。

### 10.2 row-major 地址

设 shape 为 $(d_0,\ldots,d_{R-1})$，选择轴 $a$，则

$$
s_a=\prod_{j=a+1}^{R-1}d_j,
\qquad
c_a(n)=\left\lfloor\frac{n}{s_a}\right\rfloor\bmod d_a.
$$

邻居轴坐标变为 $c_a'$ 后，flat 地址按坐标差更新：

$$
n'=n+(c_a'-c_a)s_a.
$$

源码将正负差拆成加法或减法三元分支，以避免无符号负数。

### 10.3 GPU 内存与所有权

- 输入 `DeviceArray<T>` 由调用者持有，不复制。
- workspace 持有 `elements` 个 `int` mask 和至少 `elements` 个 INT64 flat buffer。
- 返回对象为每个维度精确分配 `count` 个 INT64，因此结果 device 存储为 $O(RK)$，$K$ 是极值数。
- convenience 重载额外临时分配 workspace；正式重载允许复用但禁止并发共享同一 workspace。

## 11. grid、block、线程范围与同步

- `cuda_utils::launch_1d_kernel` 隐藏具体 block 数；源码证据只能确认它按 `validated.elements` 启动一维 mask 工作域，本文不虚构具体 threads-per-block 数值。
- mask kernel：线程 flat 负责一个输入元素，并在线程内串行执行至多 `order` 个两侧比较。
- decode kernel：每个维度单独启动，线程 index 负责一个压缩候选的一个坐标。
- Thrust `copy_if(thrust::device,...)` 按 mask 压缩 flat 索引。
- `compact_flat_indices` 中 `cuda_utils::synchronize_stream()` 是明确的一次同步，用于取得动态 count；坐标解码依 default stream 顺序排在它之后。
- 源码没有显式共享内存、原子操作或 block 级同步。

## 12. 复杂度与可能瓶颈

设总元素数 $P$、每侧阶数 $r$、rank 为 $R$、极值数为 $K$：

- CPU：最坏 $O(Pr+KR)$ 时间；结果空间 $O(KR)$。候选提前失败会减少平均比较数。
- GPU mask：每线程最坏 $O(r)$ 串行工作，总比较量 $O(Pr)$。
- compaction：扫描 $P$ 个 mask，workspace 空间 $O(P)$。
- 解码：$R$ 次 kernel launch，总工作 $O(KR)$；高 rank 会增加启动次数。
- 输出空间 $O(KR)$，且动态 count 引入一次 host 同步。
- convenience 入口的重复 device 分配、Thrust compaction、动态同步和逐维 launch 是潜在工程瓶颈；本阶段没有运行性能实验，因此不声称哪一项在 ZQ500 上实际占主导。

## 13. 与 Python 实现相同、等价替换和有意不同的部分

| 类别 | 内容 |
| --- | --- |
| 相同 | 六比较器、两侧 `order` 全称比较、clip/非 clip 分支、raise 拒绝、按 axis、row-major `nonzero` 顺序、rank 个 INT64 坐标数组 |
| 等价替换 | Python bool mask → GPU int mask；`cp.nonzero` → Thrust flat compaction + coordinate decode；高维 `cp.take` → 通用 stride 地址公式 |
| 有意修复 | 所有 rank 统一校验/归一化 axis；wrap 对任意距离完整取模，不复制固定 1D/2D backend 缺陷 |
| 类型扩展 | 固定 kernel 原生仅四类型；正式 C++ 通过精确合法路径覆盖五种业务类型，同时明确不支持 FP64 |
| 工程新增 | 可复用 workspace、容量异常、shape 防篡改、标量拒绝、INT64 容量门禁 |

这些差异中，axis 和完整模属于“原理不变、缺陷修复/接口扩展”，没有改变局部极值的数学判据。

## 14. ZQ500 平台边界

- 当前源码使用 `DeviceArray`、项目 `cuda_utils`、Thrust 和 `__half`，实际可编译性与行为必须以 `gpu_02` 中 ZQ500 SDK 为准。
- 正式路径只有 FP32、FP16、INT32、INT16、INT8 显式实例；没有 DOUBLE。
- 仓库证据状态为 `source-changed-pending-zq500`，说明源码已变化、历史运行证据失效，不能宣称本提交已在当前 ZQ500 容器重新通过。
- 本学习任务按规则不构建、不运行、不连接远程服务器；若另开验证任务，必须遵循标准五阶段远程流程并只跑 E3 peak/radar 最小切片，不能启动全量测试。

## 15. 测试覆盖与剩余风险

### 15.1 源码中已有覆盖

- 五种 T；一维、二维、三维；正轴和负轴。
- strict/non-strict/equal/not-equal 六比较器。
- clip、wrap、未知字符串 wrap、`order` 超轴长。
- 空维仍返回 rank 个空数组。
- raise、order 0、axis 越界、workspace 不足、标量 shape 异常。
- INT32 `16777216/16777217` 精确区分。
- GPU 每维坐标与 CPU reference 以零容差对照。

### 15.2 仍未由本阶段新增证据覆盖

- 当前提交在 ZQ500 上的实际编译与运行状态。
- 极大 shape 的容量边界、所有零长度维组合及多次 workspace 复用/并发误用。
- NaN、正负零、无穷大在六比较器中的逐项期望。
- 高 rank、大 `order`、高候选密度的性能和显存峰值。
- 非 default stream 或跨线程并发调用行为；接口注释只承诺 default stream。

## 16. 自检问题与参考答案

### 16.1 为什么 V1 可以作为正式学习版本？

**参考答案 1：**相关源文件、直接测试和契约在 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 上均 clean，分支和提交时间已记录；代码证据全部通过 `git show <SHA>:<path>` 读取，因此文档能回到同一历史快照，不依赖未来工作区行号。

### 16.2 CPU 和 GPU 的核心职责如何区分？

**参考答案 2：**CPU 在 `peak_finding_typed.cpp:142-197` 直接遍历 flat、比较并追加坐标；GPU 在 `peak_finding_kernels.cuh:48-84` 生成 mask，由 `peak_finding_typed.cu:75-116` 压缩和解码。API 包装、CPU reference、GPU host 调度、device kernel 没有混写。

### 16.3 数学全称比较落在何处？

**参考答案 3：**CPU 的 `for (int step=1; extrema; ++step)` 和 GPU mask kernel 的同构循环分别计算正负邻居，并用 `compare(...) && compare(...)` 更新 `extrema`；到 `step==order` 才 break，所以候选为真等价于所有距离的两侧比较都为真。

### 16.4 五种输入类型为什么不会破坏比较？

**参考答案 4：**`InputTraits<T>::load_comparison` 让 FP32/INT32 原生、FP16 精确升 FP32、INT16/INT8 精确升 INT32。测试 `e3_peak_radar_type_smoke.cu:239-249` 用超过 $2^{24}$ 的相邻 INT32 验证没有经 FP32 丢失差异。

### 16.5 GPU 每个线程负责什么？

**参考答案 5：**mask kernel 的线程 flat 对应一个输入元素，在线程内串行比较至多 $2r$ 个邻居；decode kernel 的线程 index 对应一个已压缩极值，并只解码当前启动所负责的一个维度。

### 16.6 为什么 GPU 需要一次同步？

**参考答案 6：**坐标数组必须精确分配动态长度 K。`thrust::copy_if` 返回 output_end 后，host 要计算 `output_end-output` 才能创建 K 长度的 DeviceArray，因此 `compact_flat_indices` 在 `peak_finding_typed.cu:84` 调用 `synchronize_stream()`。

### 16.7 C++ wrap 与固定版一/二维 Python backend 有何不同？

**参考答案 7：**C++ CPU/GPU helper 先计算 `distance % axis_length`，对任意 order 都是真正模回绕；固定版一/二维 CUDA helper 只加减一次轴长，order 跨多周期可能仍越界。数学原理不变，C++ 是缺陷修复。

### 16.8 输出为什么是 INT64 tuple？

**参考答案 8：**Python `cp.nonzero` 返回每维一个坐标数组。C++ CPU 用 `vector<vector<int64_t>>`，GPU 用 `vector<DeviceArray<int64_t>>`，外层长度等于 rank，每个内层长度等于 K；shape reset 还保证 flat 索引不超过 INT64。

### 16.9 现有测试能否证明当前 ZQ500 已通过？

**参考答案 9：**不能。源码只表明测试意图与断言，仓库状态仍为 `source-changed-pending-zq500`；本阶段也没有运行。只有按远程流程在 `gpu_02` 执行最小 E3 切片并保存新结果，才能形成当前 ZQ500 运行证据。

### 16.10 主要剩余风险是什么？

**参考答案 10：**包括当前 ZQ500 编译/运行未复验、NaN/无穷大边界、极大 shape、并发 workspace 误用以及高 rank/大 order 性能。源码契约明确 default stream 和 workspace 不得并发复用，因此不能扩张成未验证保证。

## 版本差异与原理不变量

当前只有 V1，尚无 V2/V3 可比较。后续优化版本必须追加而不能覆盖本节。

| 维度 | V1 | 原理不变量 |
| --- | --- | --- |
| 算法结构 | CPU 直接追加；GPU mask→compaction→decode | 中心对两侧 `order` 邻居执行全称比较 |
| 复杂度 | 最坏 $O(Pr+KR)$ | 局部极值判据不变 |
| 内存访问 | row-major flat + axis stride | 仅选中轴坐标改变 |
| 并行方式 | 一个 mask 线程/输入元素，一个 decode 线程/候选/维 | 每个候选判断彼此独立 |
| 精度 | 五类型精确比较路径，INT64 坐标 | 比较器序关系和坐标语义不变 |
| 边界 | clip、完整模 wrap、raise 拒绝 | 边界模型是问题定义的一部分 |
| 运行证据 | `source-changed-pending-zq500` | 不把未执行测试写成通过 |
