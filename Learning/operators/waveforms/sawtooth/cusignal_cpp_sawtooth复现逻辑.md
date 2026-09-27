# cusignal_cpp_sawtooth 复现逻辑

## 版本索引

| 版本 | 短 SHA | 提交时间 | 分支 | 状态 |
| --- | --- | --- | --- | --- |
| V1 | `1ccd32e` | 2026-07-27 22:12:16 | `snapshot/all-current-20260716` | 当前实现 |

---

## V1：原始学习版本（1ccd32e）

### 一、版本信息

| 项目 | 内容 |
| --- | --- |
| 完整 Git SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支 | `snapshot/all-current-20260716` |
| sawtooth 相关文件 dirty 状态 | 全部干净（无修改） |
| 仓库整体 dirty | `demo/config/task1_demo.conf` 有修改，与 sawtooth 无关 |

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注“无直接原理对应”。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在“含义解释”列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §2.1 `waveforms_typed.h:225` CPU 标量 width 入口模板参数 | `template<class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，决定输入数组元素 dtype；支持 FP32/FP16/INT32/INT16/INT8 五种；输出固定为 `double` |
| `t` | §2.1 `waveforms_typed.h:225` CPU 标量 width 入口首参数 | `const std::vector<T>&`；GPU 侧为 `const DeviceArray<T>&` | 原理 1 §2.2 中的 $t$ | 输入时间数组（弧度）；kernel 内以 `const Input* t` 形式出现 |
| `width` | §2.1 `waveforms_typed.h:225` CPU 标量 width 入口第二参数 | `double`（标量入口）或 `const std::vector<T>&`（数组入口） | 原理 1 §2.2 中的 $w$ | 上升斜坡占比，约束 $w \in [0,1]$；$w=1$ 纯上升锯齿、$w=0$ 纯下降锯齿、$w=0.5$ 三角波；kernel 内经 `waveform_load` 加载为 `Scalar width` |
| `out` | §2.1 `waveforms_typed.h:249` GPU 标量 width 入口第二参数 | `DeviceArray<double>&` | 原理 1 §2.2 中的 $y$ | GPU 输出数组，固定 FP64 存储但 GPU 计算为 FP32，经 Host 拓宽 |
| `options` | §2.1 `waveforms_typed.h:234` CPU 广播入口第三参数 | `const WaveformBroadcastOptions&` | 无直接原理对应 | 工程广播配置，承载 `t_shape` 与 `control_shape` |
| `WaveformBroadcastOptions` | §2.1 `waveforms_typed.h:12-25` 辅助结构体 | 结构体，含 `t_shape`、`control_shape`（均为 `std::vector<int>`） | 无直接原理对应 | 工程广播形状描述，NumPy 风格广播的输入形状 |
| `t_shape` | §2.1 `waveforms_typed.h:12-25` `WaveformBroadcastOptions` 字段 | `std::vector<int>` | 无直接原理对应 | 工程变量，`t` 的逻辑形状 |
| `control_shape` | §2.1 `waveforms_typed.h:12-25` `WaveformBroadcastOptions` 字段 | `std::vector<int>` | 无直接原理对应 | 工程变量，`width` 的逻辑形状 |
| `WaveformBroadcastCpuResult` | §2.1 `waveforms_typed.h:12-25` 辅助结构体 | 结构体，含 `values`（`std::vector<double>`）、`shape`（`std::vector<int>`） | 无直接原理对应 | 工程广播 CPU 输出聚合 |
| `WaveformBroadcastDeviceResult` | §2.1 `waveforms_typed.h:12-25` 辅助结构体 | 结构体，含 `values`（`DeviceArray<double>`）、`shape`（`std::vector<int>`） | 无直接原理对应 | 工程广播 GPU 输出聚合 |
| `values` | §2.1 `waveforms_typed.h:12-25` 广播结果结构体字段 | `std::vector<double>`（CPU）或 `DeviceArray<double>`（GPU） | 原理 1 §2.2 中的 $y$ | 广播输出的一维展平值数组 |
| `shape` | §2.1 `waveforms_typed.h:12-25` 广播结果结构体字段 | `std::vector<int>` | 无直接原理对应 | 工程变量，广播输出的逻辑形状 |
| `phase_input` | §3.1 `waveforms_typed.cpp:51` `sawtooth_value` 首参数 | `double` | 原理 1 §2.2 中的 $t$ | 取模前的原始输入相位（时间），可能超出 $[0, 2\pi)$ |
| `waveform_pi64` | §3.1 `waveforms_typed.cpp:48` 命名常量 | `constexpr double waveform_pi64 = 3.141592653589793238462643383279502884` | 数学常数 $\pi$（FP64 精度） | 圆周率 FP64 字面量，CPU 全程使用以保证 FP64 精度 |
| `waveform_two_pi64` | §3.1 `waveforms_typed.cpp:49` 命名常量 | `constexpr double waveform_two_pi64 = 2.0 * waveform_pi64` | 原理 1 §2.2 中的 $T = 2\pi$ | 波形周期 FP64 字面量，CPU 取模基准 |
| `phase` | §3.1 `waveforms_typed.cpp:56` `sawtooth_value` 局部 | `double`，`std::fmod(phase_input, waveform_two_pi64)` 后经负相位修正 | 原理 1 §2.2 中的 $t_{\text{mod}}$ | 取模并修正到 $[0, 2\pi)$ 后的相位；kernel 内以 `Scalar phase`（FP32）同名出现 |
| `y` | §3.2 `waveforms_typed.cpp:306` CPU 标量入口返回值 | `std::vector<double>` | 原理 1 §2.2 中的 $y$ | CPU 标量 width 入口的输出数组；host wrapper 中以 `DeviceArray<float>& y` 作为 FP32 计算缓冲 |
| `i` | §3.2 `waveforms_typed.cpp:307` CPU 标量入口循环变量 | `std::size_t` | 无直接原理对应 | 工程变量，CPU 逐元素循环索引 |
| `output` | §3.3 `waveforms_typed.cpp:322` CPU 数组 width 入口返回值 | `std::vector<double>` | 原理 1 §2.2 中的 $y$ | CPU 数组 width 入口的输出数组（与 `y` 同义）；kernel 中以 `Output* output` 形式出现 |
| `index` | §3.3 `waveforms_typed.cpp:323` CPU 数组入口循环变量 | `std::size_t` | 无直接原理对应 | 工程变量，CPU 数组入口循环索引；kernel 内作为 `const std::size_t index` 表示 CUDA 线程全局索引 |
| `current` | §3.3 `waveforms_typed.cpp:324` CPU 数组入口局部 | `const double`，经 `SimpleSignalTypePolicy<T>::load` 加载的当前 width 值 | 原理 1 §2.2 中的 $w$（单个元素） | 数组 width 模式下当前元素对应的 width 值 |
| `plan` | §3.4 `waveforms_typed.cpp:338` CPU 广播入口局部 | `const auto plan`，`make_broadcast_plan(...)` 返回值 | 无直接原理对应 | 工程广播计划，包含输出 `shape`、`left_indices`、`right_indices` |
| `result` | §3.4 `waveforms_typed.cpp:340` CPU 广播入口返回值 | `WaveformBroadcastCpuResult` | 无直接原理对应 | 工程广播 CPU 输出聚合，承载 `values` 与 `shape` |
| `left_indices` | §3.4 `waveforms_typed.cpp:342` `plan` 字段 | `const std::vector<std::size_t>&`（CPU）或 `DeviceArray<std::size_t>`（GPU） | 无直接原理对应 | 工程变量，每个输出元素对应的 `t` 线性索引数组；GPU 侧经 H2D 搬运到 device，kernel 内以 `const std::size_t* t_indices` 出现 |
| `right_indices` | §3.4 `waveforms_typed.cpp:342` `plan` 字段 | `const std::vector<std::size_t>&`（CPU）或 `DeviceArray<std::size_t>`（GPU） | 无直接原理对应 | 工程变量，每个输出元素对应的 `width` 线性索引数组；GPU 侧经 H2D 搬运到 device，kernel 内以 `const std::size_t* width_indices` 出现 |
| `computed` | §4.1 `waveforms_typed.cpp:405` GPU 入口局部 | `DeviceArray<float>`，大小 `t.size()` | 无直接原理对应 | 工程 FP32 临时缓冲区，承载 GPU kernel 的 FP32 计算结果，随后经 `finalize_fp64_device_storage` 拓宽为 FP64 |
| `Input` | §5.1 `waveforms_kernels.cuh:176` kernel 模板参数 | `typename Input`，显式实例化为 `T` | 无直接原理对应 | 工程 kernel 输入元素类型 |
| `Parameter` | §5.1 `waveforms_kernels.cuh:176` kernel 模板参数 | `typename Parameter`，显式实例化为 `T` 或 `float` | 无直接原理对应 | 工程 kernel width 参数类型（标量入口为 `float`，数组入口为 `T`） |
| `Output` | §5.1 `waveforms_kernels.cuh:176` kernel 模板参数 | `typename Output = Input`，显式实例化为 `float` | 无直接原理对应 | 工程 kernel 输出元素类型，固定 `float`（FP32 计算缓冲） |
| `count` | §5.1 `waveforms_kernels.cuh:178` kernel 参数 | `std::size_t` | 无直接原理对应 | 工程变量，输入数组长度，用于线程越界检查 `if (index >= count) return` |
| `width_input` | §5.1 `waveforms_kernels.cuh:178` kernel 参数 | `Parameter width_input` | 原理 1 §2.2 中的 $w$ | kernel 接收的原始 width 参数（标量或数组首地址），经 `waveform_load` 加载为 `Scalar width` |
| `Scalar` | §5.1 `waveforms_kernels.cuh:182` kernel 局部类型别名 | `using Scalar = decltype(waveform_load(t[index]))`，对 `T=float` 得 `float` | 无直接原理对应 | 工程计算精度类型，决定 kernel 内中间量精度；FP32 路径下为 `float` |
| `pi` | §5.1 `waveforms_kernels.cuh:183` kernel 局部常量 | `const Scalar pi = static_cast<Scalar>(3.14159265358979323846F)` | 数学常数 $\pi$（FP32 精度） | 圆周率 FP32 字面量，kernel 全程使用 |
| `two_pi` | §5.1 `waveforms_kernels.cuh:184` kernel 局部常量 | `const Scalar two_pi = static_cast<Scalar>(2) * pi` | 原理 1 §2.2 中的 $T = 2\pi$ | 波形周期 FP32 字面量，kernel 取模基准 |
| `value` | §5.1 `waveforms_kernels.cuh:192` kernel 局部 | `const Scalar`，三元运算符计算的上升段或下降段结果 | 原理 1 §2.2 中的 $y$ | kernel 内当前元素的输出波形值 |
| `width_index` | §5.2 `waveforms_kernels.cuh:205-234` variable kernel 局部 | `const std::size_t`，`width_count == 1 ? 0 : index` | 无直接原理对应 | 工程变量，数组 width 模式下当前线程读取 width 数组的索引 |
| `width_count` | §5.2 `waveforms_kernels.cuh:205-234` variable kernel 参数 | `std::size_t` | 无直接原理对应 | 工程变量，width 数组长度，用于广播判断（`==1` 时标量广播） |
| `waveform_load` / `waveform_store` | §5.1 `waveforms_kernels.cuh:20-37` kernel 辅助函数 | 模板函数，对 FP32 直接透传，对 FP16/INT 经 `SimpleSignalTypePolicy<T>` 提升或降低 | 无直接原理对应 | 工程类型转换函数，kernel 内元素加载与存储的统一入口 |
| `SimpleSignalTypePolicy<T>` | §6.3 `simple_signal_typed.h` 类型策略模板 | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断） |

---

### 二、对外接口与声明

#### 2.1 头文件

**路径**：`cusignal_cpp/src/waveforms/waveforms_typed.h`

| 行号 | 声明 | 说明 |
| --- | --- | --- |
| 225–227 | `template<class T> std::vector<double> sawtooth_typed_cpu(const std::vector<T>& t, double width = 1.0)` | CPU 标量 width 入口 |
| 229–231 | `template<class T> std::vector<double> sawtooth_typed_cpu(const std::vector<T>& t, const std::vector<T>& width)` | CPU 数组 width 入口 |
| 234–237 | `template<class T> WaveformBroadcastCpuResult sawtooth_typed_cpu(const std::vector<T>& t, const std::vector<T>& width, const WaveformBroadcastOptions& options)` | CPU 广播入口 |
| 249–251 | `template<class T> void sawtooth_device(const DeviceArray<T>& t, DeviceArray<double>& out, double width = 1.0)` | GPU 标量 width 入口 |
| 253–256 | `template<class T> void sawtooth_device(const DeviceArray<T>& t, const DeviceArray<T>& width, DeviceArray<double>& out)` | GPU 数组 width 入口 |
| 259–262 | `template<class T> WaveformBroadcastDeviceResult sawtooth_device(const DeviceArray<T>& t, const DeviceArray<T>& width, const WaveformBroadcastOptions& options)` | GPU 广播入口 |

**辅助结构体**（同文件第 12–25 行）：

- `WaveformBroadcastOptions`：含 `t_shape` 和 `control_shape`，均为 `std::vector<int>`
- `WaveformBroadcastCpuResult`：含 `values`（`std::vector<double>`）和 `shape`（`std::vector<int>`）
- `WaveformBroadcastDeviceResult`：含 `values`（`DeviceArray<double>`）和 `shape`（`std::vector<int>`）

---

### 三、CPU 调用链与逐步算法

#### 3.1 CPU 核心计算函数 `sawtooth_value()`

**路径**：`cusignal_cpp/src/waveforms/waveforms_typed.cpp`，第 51–65 行

```cpp
double sawtooth_value(double phase_input, double width)
{
    if (width < 0.0 || width > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double phase = std::fmod(phase_input, waveform_two_pi64);
    if (phase < 0.0) phase += waveform_two_pi64;
    if (width == 0.0) return 1.0 - phase / waveform_pi64;
    if (width == 1.0) return phase / waveform_pi64 - 1.0;
    if (phase < width * waveform_two_pi64) {
        return phase / (waveform_pi64 * width) - 1.0;
    }
    return (waveform_pi64 * (width + 1.0) - phase) /
        (waveform_pi64 * (1.0 - width));
}
```

**逐步逻辑**：

1. **width 越界检查**：$w < 0$ 或 $w > 1$ → 返回 `quiet_NaN()`
2. **取模**：`phase = fmod(phase_input, 2π)`
3. **负相位修正**：若 `phase < 0`，则 `phase += 2π`（确保 `phase ∈ [0, 2π)`）
4. **width = 0 特例**：返回 $1 - \frac{\text{phase}}{\pi}$（纯下降锯齿）
5. **width = 1 特例**：返回 $\frac{\text{phase}}{\pi} - 1$（纯上升锯齿）
6. **上升段判断**：若 $\text{phase} < w \cdot 2\pi$，返回 $\frac{\text{phase}}{\pi w} - 1$
7. **下降段**：返回 $\frac{\pi(w+1) - \text{phase}}{\pi(1-w)}$

**关键常量**（第 48–49 行）：

```cpp
constexpr double waveform_pi64 = 3.141592653589793238462643383279502884;
constexpr double waveform_two_pi64 = 2.0 * waveform_pi64;
```

CPU 全程使用 **FP64**（`double`）精度。

#### 3.2 CPU 标量 width 入口

**路径**：`waveforms_typed.cpp`，第 302–313 行

```cpp
template <typename T>
std::vector<double> sawtooth_typed_cpu(const std::vector<T>& t, double width)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported sawtooth input dtype");
    std::vector<double> y(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        y[i] = sawtooth_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[i])),
            width);
    }
    return y;
}
```

**调用链**：

```
sawtooth_typed_cpu(t, width)
    │
    ├── static_assert: 检查 T 是否为合法输入类型
    ├── 分配输出 vector<double> y(t.size())
    │
    └── 逐元素循环 i = 0..N-1:
            │
            ├── SimpleSignalTypePolicy<T>::load(t[i])  → 提升为 FP32/FP64 值
            ├── static_cast<double>(...)                → 转为 FP64
            └── sawtooth_value(phase_input, width)      → FP64 核心计算
```

**数据类型策略**：输入 `T` 可为 `float`、`__half`、`int32_t`、`int16_t`、`int8_t`，均通过 `SimpleSignalTypePolicy<T>::load()` 提升后转为 `double`，以 FP64 精度完成计算。输出固定为 `vector<double>`。

#### 3.3 CPU 数组 width 入口

**路径**：`waveforms_typed.cpp`，第 315–331 行

```cpp
template <typename T>
std::vector<double> sawtooth_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& width)
{
    if (width.size() != 1 && width.size() != t.size()) {
        throw std::invalid_argument("sawtooth width is not broadcastable to t");
    }
    std::vector<double> output(t.size());
    for (std::size_t index = 0; index < t.size(); ++index) {
        const double current = static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(
            width.size() == 1 ? width.front() : width[index]));
        output[index] = sawtooth_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[index])),
            current);
    }
    return output;
}
```

**width 广播**：若 `width.size() == 1`，所有元素使用 `width.front()`；否则要求 `width.size() == t.size()`，逐元素配对。

#### 3.4 CPU 广播入口

**路径**：`waveforms_typed.cpp`，第 333–349 行

```cpp
template <typename T>
WaveformBroadcastCpuResult sawtooth_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& width,
    const WaveformBroadcastOptions& options)
{
    const auto plan = make_broadcast_plan(
        options.t_shape, options.control_shape, t.size(), width.size());
    WaveformBroadcastCpuResult result;
    result.shape = plan.shape;
    result.values.resize(plan.left_indices.size());
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] = sawtooth_value(
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(t[plan.left_indices[i]])),
            static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(width[plan.right_indices[i]])));
    }
    return result;
}
```

**广播机制**：`make_broadcast_plan()`（第 102–146 行）根据 `t_shape` 和 `control_shape` 计算 NumPy 风格的广播结果 shape、以及每个输出元素对应的 `t` 和 `width` 线性索引数组 `left_indices` / `right_indices`。CPU 广播入口直接使用这两个索引数组查表。

---

### 四、GPU 调用链、Host Wrapper 与 Kernel

#### 4.1 GPU 公开入口（标量 width）

**路径**：`waveforms_typed.cpp`，第 400–408 行

```cpp
template <typename T>
void sawtooth_device(
    const DeviceArray<T>& t, DeviceArray<double>& out, double width)
{
    if (out.size() != t.size()) throw std::invalid_argument("sawtooth_device size mismatch");
    DeviceArray<float> computed(t.size());
    sawtooth_fp32_compute_device(t, computed, static_cast<float>(width));
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

**调用链**：

```
sawtooth_device(t, out, width)
    │
    ├── 尺寸检查: out.size() != t.size() → 抛异常
    ├── 分配 DeviceArray<float> computed(t.size())   → GPU FP32 临时缓冲区
    │
    ├── sawtooth_fp32_compute_device(t, computed, width)
    │       │
    │       └── launch_1d_kernel(sawtooth_fixed_kernel, N, t.data, computed.data, N, width)
    │               │
    │               └── GPU kernel: 逐线程 FP32 计算
    │
    └── finalize_fp64_device_storage(computed)
            │
            ├── computed.to_host()              → D2H: GPU FP32 → Host vector<float>
            ├── finalize_fp64_on_host(...)      → Host: float → double 逐元素拓宽
            └── DeviceArray<double>::from_host  → H2D: Host vector<double> → GPU FP64
```

**精度路径总结**：GPU FP32 计算 → D2H → Host FP32→FP64 拓宽 → H2D。GPU 端**不执行任何 FP64 算术**，FP64 仅作存储/接口类型。

#### 4.2 GPU 公开入口（数组 width）

**路径**：`waveforms_typed.cpp`，第 410–419 行

```cpp
template <typename T>
void sawtooth_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    DeviceArray<double>& out)
{
    if (out.size() != t.size()) throw std::invalid_argument("sawtooth_device size mismatch");
    DeviceArray<float> computed(t.size());
    sawtooth_fp32_compute_device(t, width, computed);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

调用链与标量入口相同，仅中间调用 `sawtooth_fp32_compute_device(t, width, computed)`（数组 width 版本）。

#### 4.3 GPU 公开入口（广播）

**路径**：`waveforms_typed.cpp`，第 421–437 行

```cpp
template <typename T>
WaveformBroadcastDeviceResult sawtooth_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    const WaveformBroadcastOptions& options)
{
    const auto plan = make_broadcast_plan(
        options.t_shape, options.control_shape, t.size(), width.size());
    auto left_indices = DeviceArray<std::size_t>::from_host(plan.left_indices);
    auto right_indices = DeviceArray<std::size_t>::from_host(plan.right_indices);
    DeviceArray<float> computed(plan.left_indices.size());
    sawtooth_broadcast_fp32_compute_device(
        t, width, left_indices, right_indices, computed);
    WaveformBroadcastDeviceResult result;
    result.values = cuda_utils::finalize_fp64_device_storage(computed);
    result.shape = plan.shape;
    return result;
}
```

**广播步骤**：

1. Host 端 `make_broadcast_plan()` 计算广播索引
2. 索引数组 `left_indices` / `right_indices` 从 Host 拷贝到 Device（H2D）
3. GPU kernel 使用索引数组间接寻址
4. 结果经 FP32→FP64 拓宽后返回

#### 4.4 Host Wrapper（中间调度层）

**路径**：`cusignal_cpp/src/waveforms/waveforms_typed.cu`

**标量 width wrapper**（第 119–129 行）：

```cpp
template <typename T>
void sawtooth_fp32_compute_device(
    const DeviceArray<T>& t, DeviceArray<float>& y, float width)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported sawtooth input dtype");
    if (t.size() != y.size()) throw std::invalid_argument("sawtooth_device size mismatch");
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::sawtooth_fixed_kernel<T, float, float>,
        t.size(), t.data(), y.data(), t.size(), width);
}
```

**数组 width wrapper**（第 131–144 行）：

```cpp
template <typename T>
void sawtooth_fp32_compute_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    DeviceArray<float>& y)
{
    ...
    cuda_utils::launch_1d_kernel(
        waveforms_detail::sawtooth_variable_kernel<T, T, float>,
        t.size(), t.data(), width.data(), y.data(), t.size(), width.size());
}
```

**广播 wrapper**（第 146–161 行）：

```cpp
template <typename T>
void sawtooth_broadcast_fp32_compute_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    const DeviceArray<std::size_t>& t_indices,
    const DeviceArray<std::size_t>& width_indices,
    DeviceArray<float>& y)
{
    ...
    cuda_utils::launch_1d_kernel(
        waveforms_detail::sawtooth_broadcast_kernel<T, T, float>,
        y.size(), t.data(), width.data(), t_indices.data(), width_indices.data(),
        y.data(), y.size());
}
```

#### 4.5 Kernel Launch 配置

**路径**：`cusignal_cpp/src/cuda_utils/kernel_launch.h`

```cpp
template <typename Kernel, typename... Args>
inline void launch_1d_kernel(Kernel kernel, std::size_t elements, Args&&... args)
{
    launch_1d_kernel_with_config(kernel, elements, nullptr, 256, std::forward<Args>(args)...);
}
```

| 配置项 | 值 | 说明 |
| --- | --- | --- |
| block 大小 | 256 线程 | 一维 block，`dim3(256)` |
| grid 大小 | `ceil(N / 256)` | `dim3(div_up(elements, 256))` |
| stream | `nullptr`（默认流） | 同步提交 |
| 共享内存 | 0 字节 | 无共享内存使用 |

---

### 五、GPU Kernel 逐线程计算逻辑

#### 5.1 `sawtooth_fixed_kernel`（标量 width）

**路径**：`cusignal_cpp/src/waveforms/waveforms_kernels.cuh`，第 176–203 行

```cpp
template <typename Input, typename Parameter, typename Output = Input>
__global__ void sawtooth_fixed_kernel(
    const Input* t, Output* output, std::size_t count, Parameter width_input)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    using Scalar = decltype(waveform_load(t[index]));
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar two_pi = static_cast<Scalar>(2) * pi;
    const Scalar width = static_cast<Scalar>(waveform_load(width_input));
    if (width < static_cast<Scalar>(0) || width > static_cast<Scalar>(1)) {
        output[index] = waveform_store<Output>(NAN);
        return;
    }
    Scalar phase = fmodf(static_cast<Scalar>(waveform_load(t[index])), two_pi);
    if (phase < static_cast<Scalar>(0)) phase += two_pi;
    const Scalar value = phase < width * two_pi
        ? phase / (pi * width) - static_cast<Scalar>(1)
        : (pi * (width + static_cast<Scalar>(1)) - phase) /
            (pi * (static_cast<Scalar>(1) - width));
    output[index] = waveform_store<Output>(value);
}
```

**逐线程执行步骤**：

1. **线程索引**：`index = blockIdx.x * blockDim.x + threadIdx.x`，越界返回
2. **类型推导**：`Scalar = decltype(waveform_load(t[index]))`，对 `T=float` 得 `float`
3. **常量**：`pi` 和 `two_pi` 转为 `Scalar` 类型
4. **width 加载与检查**：通过 `waveform_load()` 提取值，越界写 `NAN` 并返回
5. **取模**：使用 `fmodf()`（单精度浮点取模），非 `fmod()`（双精度）
6. **负相位修正**：`if (phase < 0) phase += two_pi`，确保 `phase ∈ [0, 2π)`
7. **分支计算**：三元运算符选择上升段或下降段公式
8. **输出存储**：`waveform_store<Output>(value)`

**关键点**：kernel 中**无 `width == 0` / `width == 1` 的显式特例处理**。当 `width == 0` 时，`phase < 0 * two_pi = 0` 恒假（`phase ∈ [0, 2π)`），走下降段，分母 `pi * (1 - 0) = pi`，不除零；当 `width == 1` 时，`phase < 1 * two_pi` 恒真，走上段，分母 `pi * 1 = pi`，不除零。

#### 5.2 `sawtooth_variable_kernel`（数组 width）

**路径**：`waveforms_kernels.cuh`，第 205–234 行

与 fixed kernel 的唯一差异：

```cpp
const std::size_t width_index = width_count == 1 ? 0 : index;
const Scalar width = static_cast<Scalar>(waveform_load(width_input[width_index]));
```

**width 广播**：若 `width_count == 1`，所有线程使用 `width_input[0]`；否则使用 `width_input[index]`。

#### 5.3 `sawtooth_broadcast_kernel`（广播）

**路径**：`waveforms_kernels.cuh`，第 236–264 行

与 variable kernel 的差异：通过预计算索引数组间接寻址：

```cpp
const Scalar width = static_cast<Scalar>(waveform_load(width_input[width_indices[index]]));
...
Scalar phase = fmodf(static_cast<Scalar>(waveform_load(t[t_indices[index]])), two_pi);
```

---

### 六、Template、类型分发与显式实例化

#### 6.1 CPU 显式实例化

**路径**：`waveforms_typed.cpp`，第 554–594 行

```cpp
#define INSTANTIATE_CPU(T) \
    template std::vector<double> sawtooth_typed_cpu(const std::vector<T>&, double); \
    template std::vector<double> sawtooth_typed_cpu(const std::vector<T>&, const std::vector<T>&); \
    template WaveformBroadcastCpuResult sawtooth_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const WaveformBroadcastOptions&); \
    template void sawtooth_device(const DeviceArray<T>&, DeviceArray<double>&, double); \
    template void sawtooth_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<double>&); \
    template WaveformBroadcastDeviceResult sawtooth_device( \
        const DeviceArray<T>&, const DeviceArray<T>&, const WaveformBroadcastOptions&);
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
```

#### 6.2 GPU 显式实例化

**路径**：`waveforms_typed.cu`，第 218–232 行

```cpp
#define I(T) \
    template void sawtooth_fp32_compute_device(const DeviceArray<T>&, DeviceArray<float>&, float); \
    template void sawtooth_fp32_compute_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<float>&); \
    template void sawtooth_broadcast_fp32_compute_device( \
        const DeviceArray<T>&, const DeviceArray<T>&, \
        const DeviceArray<std::size_t>&, const DeviceArray<std::size_t>&, DeviceArray<float>&);
I(float);I(__half);I(int32_t);I(int16_t);I(int8_t);
```

#### 6.3 五种输入类型

| T | `SimpleSignalTypePolicy<T>::load()` 行为 | Scalar 类型 | 说明 |
| --- | --- | --- | --- |
| `float` | 恒等返回 | `float` | 原生 FP32 |
| `__half` | `__half2float()` 转为 FP32 | `float` | FP16 → FP32 提升 |
| `int32_t` | `static_cast<float>` | `float` | 整数 → FP32 |
| `int16_t` | `static_cast<float>` | `float` | 整数 → FP32 |
| `int8_t` | `static_cast<float>` | `float` | 整数 → FP32 |

---

### 七、Grid、Block、线程索引与数据范围

| 项目 | 说明 |
| --- | --- |
| 计算维度 | 一维（1D grid × 1D block） |
| 每个线程负责的数据范围 | 恰好 1 个元素：`t[index]` 和对应的 `width[index]`（或标量） |
| 线程总数 | $\lceil N / 256 \rceil \times 256$（可能超过 $N$，越界线程提前返回） |
| 有效线程数 | $N$（`index < count` 的线程） |
| 线程间同步 | 无（每个线程独立计算，无数据依赖） |
| 共享内存 | 无 |
| warp divergence 风险 | `phase < width * two_pi` 分支可能在一个 warp 内产生分歧，尤其是 width 为数组时 |

---

### 八、同步、临时缓冲区与数据搬运

#### 8.1 数据搬运路径

**标量/数组 width**：

```
Host t[N] ──H2D──→ Device t[N]
                         │
                    GPU kernel (FP32)
                         │
                    Device computed[N] (float)
                         │
              D2H → Host vector<float>
                         │
              Host float→double 拓宽
                         │
              H2D → Device out[N] (double)
```

**广播**：额外增加 `left_indices` 和 `right_indices` 的 H2D 搬运。

#### 8.2 临时缓冲区

| 缓冲区 | 类型 | 大小 | 生命周期 | 位置 |
| --- | --- | --- | --- | --- |
| `computed` | `DeviceArray<float>` | $N$ | 函数作用域 | GPU 显存 |
| `left_indices` | `DeviceArray<size_t>` | 输出元素数 | 函数作用域（仅广播入口） | GPU 显存 |
| `right_indices` | `DeviceArray<size_t>` | 输出元素数 | 函数作用域（仅广播入口） | GPU 显存 |
| Host 中间向量 | `vector<float>` → `vector<double>` | $N$ | `finalize` 函数作用域 | Host 内存 |

#### 8.3 同步

- Kernel 使用默认流（`stream = nullptr`），launch 后不显式同步
- `finalize_fp64_device_storage()` 内部：`to_host()` 隐含 `cudaMemcpy` 同步；`from_host()` 为又一次 H2D 同步
- **调用者读取 `out` 前无需额外同步**：因为 D2H → Host 计算 → H2D 的序列已确保数据就绪

---

### 九、数学/物理原理到 CPU 代码、GPU 代码的三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $t_{\text{mod}} = t \bmod 2\pi$ | `fmod(t, 2.0*M_PI)`（C `fmod`，负值结果为负） | `std::fmod(phase_input, waveform_two_pi64)` 后 `if (phase < 0) phase += 2π` | `fmodf(waveform_load(t[index]), two_pi)` 后 `if (phase < 0) phase += 2π` | CPU/GPU 均显式修正负相位，Python 未修正（已知差异） |
| width 越界 → NaN | `nan("0xfff8000000000000ULL")`（IEEE 754 硬编码位模式） | `std::numeric_limits<double>::quiet_NaN()` | `NAN` 宏 | 三者均产生静默 NaN，表示方式不同但语义一致 |
| $y = \frac{t_{\text{mod}}}{\pi w} - 1$（上升段） | `tmod / ( M_PI * w ) - 1`（FP64，在 ElementwiseKernel 中） | `phase / (waveform_pi64 * width) - 1.0`（FP64） | `phase / (pi * width) - Scalar(1)`（FP32） | GPU 以 FP32 计算，CPU/Python 以 FP64 计算 |
| $y = \frac{\pi(w+1) - t_{\text{mod}}}{\pi(1-w)}$（下降段） | `( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) )`（FP64） | `(waveform_pi64 * (width + 1.0) - phase) / (waveform_pi64 * (1.0 - width))`（FP64） | `(pi * (width + Scalar(1)) - phase) / (pi * (Scalar(1) - width))`（FP32） | 同上 |
| $w = 0$ 特例 | 依赖 `mask2` 条件 `tmod < 0` 恒假，走下降段 | `if (width == 0.0) return 1.0 - phase / waveform_pi64;` 显式处理 | 依赖 `phase < 0 * two_pi` 恒假，走下降段 | CPU 显式处理，Python/GPU 依赖分支条件自然退火 |
| $w = 1$ 特例 | 依赖 `mask2` 条件 `tmod < 2π` 恒真，走上升段 | `if (width == 1.0) return phase / waveform_pi64 - 1.0;` 显式处理 | 依赖 `phase < 1 * two_pi` 恒真，走上升段 | 同上 |
| width 数组广播 | CuPy ElementwiseKernel 自动广播 | `width.size() == 1 ? width.front() : width[index]` | `width_count == 1 ? 0 : index` 间接寻址 | Python 依赖 CuPy 运行时广播，C++ 手动处理 |
| 任意 rank 广播 | 不支持（cuSignal Python 仅支持标量或等长数组 width） | `make_broadcast_plan()` + 索引数组查表 | `sawtooth_broadcast_kernel` + GPU 索引数组间接寻址 | C++ 额外提供了 cuSignal Python 未覆盖的 NumPy 风格广播 |

---

### 十、与 cuSignal Python 实现的相同、等价替换与有意不同

#### 10.1 相同部分

| 方面 | 说明 |
| --- | --- |
| 分段线性公式 | 上升段和下降段公式完全一致 |
| width 越界行为 | $w < 0$ 或 $w > 1$ 均输出 NaN |
| 周期约定 | 默认周期 $2\pi$（弧度制） |
| 输出值域 | $[-1, 1]$（正常情况），NaN（越界） |

#### 10.2 等价替换

| Python | C++ | 等价原因 |
| --- | --- | --- |
| `cp.asarray()` 输入转换 | `SimpleSignalTypePolicy<T>::load()` + `static_cast<double/float>` | 均为类型提升到计算精度 |
| CuPy ElementwiseKernel 自动 GPU 并行 | `launch_1d_kernel()` + CUDA `__global__` kernel | 均为逐元素 GPU 并行 |
| CuPy broadcasting | `make_broadcast_plan()` + 索引数组 | 数学等价，实现方式不同 |
| `nan("0xfff8000000000000ULL")` | `std::numeric_limits<double>::quiet_NaN()` / `NAN` | 均为 IEEE 754 静默 NaN |

#### 10.3 有意不同

| 差异 | Python | C++ | 设计意图 |
| --- | --- | --- | --- |
| **负相位修正** | `fmod(t, 2π)` 无修正，负 `t` 可能产生超 $[-1,1]$ 的值 | `fmod` 后 `if (phase < 0) phase += 2π` | C++ 修正为数学正确的周期延拓，Python 行为是已知问题 |
| **width = 0/1 显式处理** | 依赖条件分支自然退火 | CPU 显式 `if (width == 0.0)` / `if (width == 1.0)` | 避免浮点精度边界下的微妙行为（如 `width` 极接近但不等于 0 或 1） |
| **计算精度** | ElementwiseKernel 使用类型参数 `T`（可能为 FP64） | GPU kernel 固定 FP32（`fmodf`），CPU 固定 FP64 | ZQ500 平台 FP64 性能远低于 FP32；GPU 端 FP32 + Host 拓宽是项目统一策略 |
| **输出 dtype** | 固定 `float64`（kernel 声明 `"float64 y"`） | 固定 `double`（`DeviceArray<double>`），但 GPU 计算为 FP32 | 接口 dtype 均为 FP64，但 C++ 的 FP64 是 Host 拓宽后的存储格式 |
| **任意 rank 广播** | 不支持 | `sawtooth_typed_cpu(t, width, options)` 支持 | C++ 扩展了 cuSignal 的功能边界 |

---

### 十一、精度、dtype、边界条件与 ZQ500 平台限制

#### 11.1 精度分析

| 路径 | 计算精度 | 取模函数 | $\pi$ 精度 |
| --- | --- | --- | --- |
| CPU | FP64（`double`） | `std::fmod`（双精度） | 36 位十进制 |
| GPU kernel | FP32（`float`） | `fmodf`（单精度） | 约 7 位十进制有效 |
| Host 拓宽 | FP32 → FP64 无损（FP32 值域内精确拓宽） | — | — |

**精度风险**：GPU FP32 的 `fmodf` 精度约 7 位有效数字，对大时间值（如 $t > 10^4$）可能丢失小数部分精度。CPU FP64 无此问题。

#### 11.2 dtype 策略

| 输入 T | GPU kernel 内 Scalar | GPU 输出 | 最终输出 |
| --- | --- | --- | --- |
| `float` | `float` | `float` | `double`（Host 拓宽） |
| `__half` | `float`（提升） | `float` | `double` |
| `int32_t` | `float`（转换） | `float` | `double` |
| `int16_t` | `float`（转换） | `float` | `double` |
| `int8_t` | `float`（转换） | `float` | `double` |

**ZQ500 平台限制**：

- FP16 输入经 `__half2float()` 提升，kernel 内全程 FP32，无 FP16 算术
- 整数输入经 `static_cast<float>` 转换，可能丢失大整数精度（`int32_t` 超过 $2^{24}$ 时）
- GPU 端不执行 FP64 算术，符合 ZQ500 DLI V2 的 FP32 计算策略

#### 11.3 边界条件处理

| 边界 | CPU 行为 | GPU 行为 |
| --- | --- | --- |
| $w < 0$ 或 $w > 1$ | `quiet_NaN()` | `NAN` 宏 |
| $w = 0$ | 显式返回 $1 - \text{phase}/\pi$ | `phase < 0` 恒假，走下降段公式 |
| $w = 1$ | 显式返回 $\text{phase}/\pi - 1$ | `phase < 2π` 恒真，走上升段公式 |
| 负时间 $t < 0$ | `fmod` + 负相位修正 → $\text{phase} \in [0, 2\pi)$ | `fmodf` + 负相位修正 → 同 CPU |
| NaN 输入 $t$ | `fmod(NaN, 2π)` 返回 NaN，后续行为不确定 | 同 CPU |
| 空输入 `t.size() == 0` | 返回空 vector | `if (t.empty()) return;` 不启动 kernel |

---

### 十二、测试验证

**测试文件**：`cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu`

#### 12.1 测试用例

| 测试项 | 行号 | 输入 | width | 验证方式 |
| --- | --- | --- | --- | --- |
| 标量 width 典型值 | 97 | 7 元素 `{-6, -3, -1, 0, 1, 3, 6}` | 0.5 | GPU vs CPU，容差 1e-6 |
| 数组 width | 101 | 同上 | `{0, 1, 0, 1, 0, 1, 0}` | GPU vs CPU |
| 无效 width | 104 | 同上 | -0.25 | 全部输出 NaN |
| 高 rank 时间数组 | 117 | 24 元素（7 元素重复填充） | 0.5 | GPU vs CPU |
| 广播 shape `{2,3,1}×{1,3,4}` | 127–131 | 6 元素 t | 12 元素 width | GPU vs CPU，验证 shape 和数值 |
| 标量时间广播 `{}×{1,3,4}` | 135–141 | 1 元素 t | 12 元素 width | GPU vs CPU，验证 shape 和数值 |
| 不兼容广播拒绝 | 146–152 | shape `{2,3}` / `{3,4}` | — | 抛 `std::invalid_argument` |

#### 12.2 测试框架

- 使用 `compare_real_vectors()` 计算最大绝对误差、最大相对误差、RMSE
- FP16 容差放宽至 `2e-3`，其余类型容差 `1e-6`
- NaN 对 NaN 视为匹配（`matching_nan`）
- 通过 `e_type_test_evidence.h` 的 `record()` 函数输出结构化 E6/F3 证据
- 每种输入类型（`float`、`__half`、`int32_t`、`int16_t`、`int8_t`）分别实例化测试

#### 12.3 未覆盖的风险

| 风险 | 说明 |
| --- | --- |
| 大时间值精度退化 | $t \gg 2\pi$ 时 FP32 `fmodf` 精度损失，未测试 |
| width 极接近 0 或 1 | 如 `width = 1e-7`，上升段斜率极大，未测试 |
| 极大数组 | 当前测试仅 7–24 元素，未验证大规模行为的正确性和性能 |
| 并发/多 stream | 测试使用默认流，未验证多 stream 并发安全性 |
| FP16 中间溢出 | FP16 输入经 FP32 提升后计算，理论上无溢出，但未测试 FP16 极端输入范围 |

---

### 十三、建议阅读顺序

按 **接口 → CPU → GPU wrapper → kernel → 测试** 顺序阅读：

1. **头文件** [`waveforms_typed.h`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/waveforms/waveforms_typed.h#L225-L262)：理解三个 CPU 重载和三个 GPU 重载的签名与 docstring
2. **CPU 核心函数** [`waveforms_typed.cpp:51–65`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp#L51-L65)：`sawtooth_value()` 的 FP64 分段线性计算，注意负相位修正和 width = 0/1 显式处理
3. **CPU 三个入口** [`waveforms_typed.cpp:302–349`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp#L302-L349)：标量、数组和广播三种调用方式
4. **GPU 公开入口** [`waveforms_typed.cpp:400–437`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp#L400-L437)：注意 FP32 计算缓冲区 → Host 拓宽 → FP64 存储的精度路径
5. **Host wrapper** [`waveforms_typed.cu:119–161`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cu#L119-L161)：三个 `sawtooth_*_device` 函数，理解 kernel 选择和参数传递
6. **CUDA kernel** [`waveforms_kernels.cuh:176–264`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh#L176-L264)：三个 kernel 的逐线程逻辑，注意 `fmodf`、负相位修正和 `waveform_load/store`
7. **辅助工具**：[`kernel_launch.h`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h)（launch 配置）、[`host_output_finalize.h`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h)（FP32→FP64 拓宽）
8. **测试** [`e3_simple_batch_type_smoke.cu:90–153`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu#L90-L153)：验证 GPU vs CPU 一致性

每段代码应关注的问题：

- CPU `sawtooth_value()` 的负相位修正与 cuSignal Python 的 `fmod` 行为差异
- GPU kernel 为什么使用 `fmodf`（FP32）而非 `fmod`（FP64）
- `finalize_fp64_device_storage()` 的 D2H → Host 拓宽 → H2D 路径是否有性能优化空间
- `sawtooth_fixed_kernel` 中 `width == 0` / `width == 1` 是否需要显式处理（当前依赖分支自然退火）
- 广播 kernel 的索引数组间接寻址对缓存友好性的影响

---

## 版本差异与原理不变量

（仅 V1 版本，暂无优化版本可供对比。以下记录当前 V1 的算法特征，供后续版本对比。）

| 特征 | V1 |
| --- | --- |
| 算法结构 | 逐元素分段线性，`fmod` + 条件分支 |
| CPU 计算精度 | FP64 |
| GPU 计算精度 | FP32（kernel）→ Host 拓宽 → FP64（存储） |
| 并行方式 | 1D grid × 256 线程/block，无线程间同步 |
| 内存访问 | 合并访问（coalesced），输入输出均为连续一维数组 |
| NaN 处理 | `quiet_NaN()`（CPU）/ `NAN`（GPU） |
| 负相位修正 | 有（`if (phase < 0) phase += 2π`） |
| width 特例处理 | CPU 显式处理（`width == 0` / `width == 1`），GPU 依赖分支自然退火 |
| 广播支持 | NumPy 风格多 rank 广播（CPU 和 GPU 均支持） |

**原理不变量**：无论版本如何优化，以下数学物理原理始终不变：

1. 分段线性公式（原理 1）：上升段和下降段的线性表达式不变
2. width 参数的变形语义（原理 3）：$w = 0$ 纯下降、$w = 1$ 纯上升、$w = 0.5$ 三角波
3. 周期约定：默认 $2\pi$ 弧度
4. 值域：$[-1, 1]$（width 越界时为 NaN）
