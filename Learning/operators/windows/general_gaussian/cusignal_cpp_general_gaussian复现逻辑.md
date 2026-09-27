# cusignal_cpp_general_gaussian 复现逻辑

## 版本索引

| 版本 | 短 SHA | 提交时间 | 分支 | 状态 |
| --- | --- | --- | --- | --- |
| V1 | `1ccd32ea` | 2026-07-27 22:12:16 +0800 | `snapshot/all-current-20260716` | 当前实现 |

- [V1：原始学习版本（1ccd32ea）](#v1原始学习版本1ccd32ea)
- [版本差异与原理不变量](#版本差异与原理不变量)

---

## V1：原始学习版本（1ccd32ea）

### 一、版本信息

| 项目 | 内容 |
| --- | --- |
| 完整 Git SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 短 SHA | `1ccd32ea` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支 | `snapshot/all-current-20260716` |
| `cusignal_cpp` dirty 状态 | 干净（无修改） |

`general_gaussian` 在本提交中涉及的文件均位于 `cusignal_cpp/` 下，路径以共享根目录 `ZKX_dev/` 起算：

| 相对路径 | 说明 |
| --- | --- |
| `cusignal_cpp/src/windows/windows_typed.h` | CPU/GPU 公开接口声明 |
| `cusignal_cpp/src/windows/windows_typed.cpp` | CPU reference 实现 + GPU host 包装层 |
| `cusignal_cpp/src/windows/windows_typed.cu` | GPU FP32 计算调度层 |
| `cusignal_cpp/src/windows/windows_kernels.cuh` | GPU device kernel 定义 |
| `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | 类型化 smoke 测试 |

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注"无直接原理对应"。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在"含义解释"列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §2.2 `windows_typed.h:102-104`（`general_gaussian_typed_cpu` 模板参数） | `template <class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，决定 `power` / `width` 输入 dtype；显式实例化为 `float`/`__half`/`std::int32_t`/`std::int16_t`/`std::int8_t` 五种；输出固定 `double`（CPU）/`DeviceArray<double>`（GPU） |
| `n` | §2.1 `windows_typed.h:119-121`（`general_gaussian_device` 首参数） | `int n` | 原理 1 §1 中的 $M$ | 窗长度（点数），约束 $M \geq 0$；CPU 入口记作 `length`（`windows_typed.h:102-104`） |
| `power` | §2.1 `windows_typed.h:119-121`（`general_gaussian_device` 第二参数） | `T power`，5 种 dtype 之一 | 原理 1 §1 中的 $p$（形状参数） | 形状参数，控制窗的峰度形态；$p=1$ 标准高斯、$p=0.5$ 拉普拉斯、$p>1$ 超高斯；接受负值但意义未定义（与 Python 一致，不校验） |
| `width` | §2.1 `windows_typed.h:119-121`（`general_gaussian_device` 第三参数） | `T width`，5 种 dtype 之一 | 原理 1 §1 中的 $\sigma$（尺度参数，代码记作 `width`） | 尺度参数（标准差），控制窗的宽度；与 Python `sig` 同义；$\sigma > 0$，但代码不校验（`width == 0` 时产生 `inf`/`nan`，与 Python 一致） |
| `out` | §2.1 `windows_typed.h:119-121`（`general_gaussian_device` 第四参数） | `DeviceArray<double>& out` | 原理 1 §1 中的 $w[n]$ | GPU 输出窗序列，FP64 存储但 device 计算 FP32，经 `finalize_fp64_device_storage` 拓宽；CPU 侧以 `std::vector<double> out` 形式出现 |
| `sym` | §2.1 `windows_typed.h:119-121`（`general_gaussian_device` 第五参数） | `bool sym = true` | cuSignal 算法 1 §2（步骤 2 周期扩展） | 工程旗标，`true` 对称窗（`effective = n`），`false` 周期窗（`effective = n+1` 再截断）；CPU 侧以 `symmetric` 形式出现 |
| `effective` | §3.1 `windows_typed.cpp:347`（`general_gaussian_typed_cpu` 局部） | `int`，`symmetric ? length : length + 1` | cuSignal 算法 1 §2（步骤 2 DFT-even 扩展） | 工程变量，对称中心化长度；GPU 侧以 `effective_count` 形式出现于 `general_gaussian_fp32_compute_device`（`windows_typed.cu:75`）与 kernel 参数 |
| `power_value` | §3.1 `windows_typed.cpp:349`（`general_gaussian_typed_cpu` 局部） | `double`，`static_cast<double>(host_load(power))` | 原理 1 §1 中的 $p$（FP64 标量） | 由 5 种输入类型统一转 FP64 的形状参数，CPU 路径专用 |
| `width_value` | §3.1 `windows_typed.cpp:350`（`general_gaussian_typed_cpu` 局部） | `double`，`static_cast<double>(host_load(width))` | 原理 1 §1 中的 $\sigma$（FP64 标量） | 由 5 种输入类型统一转 FP64 的尺度参数，CPU 路径专用 |
| `index` | §3.1 `windows_typed.cpp:351-356`（`general_gaussian_typed_cpu` 循环变量） | `int` | 原理 1 §1 中的 $i$（从 0 开始的索引） | 工程变量，逐点输出索引，对应 `out[index]` 的位置；kernel 侧以 `const int index` 形式出现（`windows_kernels.cuh:92`） |
| `x` | §3.1 `windows_typed.cpp:352`（`general_gaussian_typed_cpu` 局部） | `double`，`(index - 0.5 * (effective - 1)) / width_value` | 原理 1 §1 中的 $n/\sigma$（中心化偏移归一化） | 当前采样点的中心化位置除以尺度参数；kernel 侧以 `Scalar position` 形式出现（`windows_kernels.cuh:100-101`） |
| `computed` | §3.2 `windows_typed.cpp:367`（`general_gaussian_device` 局部） | `DeviceArray<float>`，长度 `out.size()` | 无直接原理对应 | 工程 FP32 临时缓冲区，承载 GPU kernel 的 FP32 计算结果，随后经 `finalize_fp64_device_storage` 拓宽为 FP64 |
| `p` | §4.3 `windows_typed.cu:75`（`general_gaussian_fp32_compute_device` 第二参数） | `T p` | 原理 1 §1 中的 $p$ | 工程变量，GPU 侧形状参数；与 CPU `power` 同义 |
| `w` | §4.3 `windows_typed.cu:75`（`general_gaussian_fp32_compute_device` 第三参数） | `T w` | 原理 1 §1 中的 $\sigma$ | 工程变量，GPU 侧尺度参数；与 CPU `width` 同义 |
| `o` | §4.3 `windows_typed.cu:75`（`general_gaussian_fp32_compute_device` 第四参数） | `DeviceArray<float>&` | 原理 1 §1 中的 $w[n]$（FP32 计算缓冲） | 工程变量，GPU FP32 输出缓冲区；通过 `o.data()` 传给 kernel 作为 `T* output` |
| `effective_count` | §4.3 `windows_typed.cu:75`（kernel 第五参数） | `int`，`sym ? n : n + 1` | cuSignal 算法 1 §2（步骤 2 DFT-even 扩展） | 工程变量，GPU 侧的对称中心化长度，与 CPU `effective` 同义 |
| `Parameter` | §2.4 `windows_kernels.cuh:84`（`general_gaussian_kernel` 模板参数） | `typename Parameter`，显式实例化为用户 `T` | 无直接原理对应 | 工程 kernel `power_input` / `width_input` 参数类型，与 host `T` 一致 |
| `output_count` | §4.4 `windows_kernels.cuh:86`（`general_gaussian_kernel` 首参数） | `int` | 原理 1 §1 中的 $M$（输出长度） | 工程变量，输出元素数，用于线程越界检查 `if (index >= output_count) return` |
| `power_input` | §4.4 `windows_kernels.cuh:87`（`general_gaussian_kernel` 第二参数） | `Parameter power_input` | 原理 1 §1 中的 $p$ | kernel 接收的原始形状参数，经 `window_load` 加载为 `Scalar power`（FP32） |
| `width_input` | §4.4 `windows_kernels.cuh:88`（`general_gaussian_kernel` 第三参数） | `Parameter width_input` | 原理 1 §1 中的 $\sigma$ | kernel 接收的原始尺度参数，经 `window_load` 加载为 `Scalar width`（FP32） |
| `output` | §4.4 `windows_kernels.cuh:89`（`general_gaussian_kernel` 第四参数） | `T*`（host 调度层固定 `T = float`） | 原理 1 §1 中的 $w[n]$ | kernel 输出指针，对应 `o.data()`，最终经 `finalize` 拓宽为 FP64 |
| `Scalar` | §4.4 `windows_kernels.cuh:97`（`general_gaussian_kernel` 局部类型别名） | `using Scalar = decltype(window_load(power_input))`，对 5 种 `Parameter` 得 `float` | 无直接原理对应 | 工程计算精度类型，决定 kernel 内中间量精度；FP32 路径下为 `float` |
| `power` (kernel) | §4.4 `windows_kernels.cuh:98`（`general_gaussian_kernel` 局部） | `const Scalar power`，`window_load(power_input)` | 原理 1 §1 中的 $p$（FP32 标量） | kernel 内形状参数（FP32） |
| `width` (kernel) | §4.4 `windows_kernels.cuh:99`（`general_gaussian_kernel` 局部） | `const Scalar width`，`window_load(width_input)` | 原理 1 §1 中的 $\sigma$（FP32 标量） | kernel 内尺度参数（FP32） |
| `position` | §4.4 `windows_kernels.cuh:100-101`（`general_gaussian_kernel` 局部） | `Scalar`，`(index - 0.5F * (effective_count - 1)) / width` | 原理 1 §1 中的 $n/\sigma$（中心化偏移归一化） | kernel 内当前线程的中心化位置；与 CPU `x` 数学等价 |
| `host_load<T>` | §3.1 `windows_typed.cpp:17-21`（host helper 函数模板） | `template <class T> float host_load(T value)` | 无直接原理对应 | 工程类型转换函数，调用 `SimpleSignalTypePolicy<T>::load` 将 5 种 dtype 统一转 `float`；CPU 再 `(double)` 拓宽为 FP64 |
| `window_load<T>` | §4.5 `windows_kernels.cuh:10-18`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::load`，5 种 dtype 统一转 `float` |
| `window_store<T, Scalar>` | §4.5 `windows_kernels.cuh:20-28`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::store`，把 FP32 `value` 转回 `T` |
| `SimpleSignalTypePolicy<T>` | §5.1 `cuda_utils/simple_signal_typed.h`（类型策略模板） | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断） |
| `is_simple_signal_input_v<T>` | §5.1 `cuda_utils/simple_signal_typed.h`（类型 trait） | `template <class T> constexpr bool is_simple_signal_input_v` | 无直接原理对应 | 工程类型 trait，编译期判断 `T` 是否属于 5 种简单信号类型，用于 `static_assert` |
| `launch_1d_kernel` | §4.3 `cuda_utils/kernel_launch.h`（kernel 启动工具） | 函数模板，参数 `(Kernel kernel, std::size_t elements, Args&&... args)` | 无直接原理对应 | 工程 kernel 启动工具，自动按 `elements` 计算 grid/block（默认 `block_size=256`，默认 stream） |
| `finalize_fp64_device_storage` | §3.2 `cuda_utils/host_output_finalize.h`（FP32→FP64 拓宽工具） | 函数，参数 `(DeviceArray<float> computed)` 返回 `DeviceArray<double>` | 无直接原理对应 | 工程 FP32→FP64 拓宽工具，内部走 D2H → host float→double → H2D；GPU 端不执行任何 FP64 算术 |

---

### 二、对外接口与代码定位

所有行号均针对提交 `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` 中的文件，按 `ZKX_dev/` 相对路径标记。

#### 2.1 API 包装层（GPU 公开入口）

| 符号 | 文件 | 行号 | 签名 |
| --- | --- | --- | --- |
| `general_gaussian_device<T>` | `cusignal_cpp/src/windows/windows_typed.h` | 119–121 | `template <class T> void general_gaussian_device(int n, T power, T width, DeviceArray<double>& out, bool sym = true);` |
| `general_gaussian_device<T>` 实现 | `cusignal_cpp/src/windows/windows_typed.cpp` | 360–375 | 见 3.2 节 |

#### 2.2 CPU/reference 核心计算

| 符号 | 文件 | 行号 | 签名 |
| --- | --- | --- | --- |
| `general_gaussian_typed_cpu<T>` | `cusignal_cpp/src/windows/windows_typed.h` | 102–104 | `template <class T> std::vector<double> general_gaussian_typed_cpu(int n, T power, T width, bool sym = true);` |
| `general_gaussian_typed_cpu<T>` 实现 | `cusignal_cpp/src/windows/windows_typed.cpp` | 341–358 | 见 3.1 节 |

#### 2.3 GPU host 调度层（FP32 计算入口）

| 符号 | 文件 | 行号 | 签名 |
| --- | --- | --- | --- |
| `general_gaussian_fp32_compute_device<T>` 前置声明 | `cusignal_cpp/src/windows/windows_typed.cpp` | 150–152 | `template <class T> void general_gaussian_fp32_compute_device(int, T, T, DeviceArray<float>&, bool);` |
| `general_gaussian_fp32_compute_device<T>` 实现 | `cusignal_cpp/src/windows/windows_typed.cu` | 75 | 见 4.3 节 |

#### 2.4 GPU device kernel 核心

| 符号 | 文件 | 行号 | 签名 |
| --- | --- | --- | --- |
| `general_gaussian_kernel<T, Parameter>` | `cusignal_cpp/src/windows/windows_kernels.cuh` | 84–107 | `template <typename T, typename Parameter> __global__ void general_gaussian_kernel(int output_count, Parameter power_input, Parameter width_input, T* output, int effective_count);` |

#### 2.5 测试入口

| 符号 | 文件 | 行号 | 说明 |
| --- | --- | --- | --- |
| `run_type<T>` 内 `ok[8]` | `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | 524–577 | `general_gaussian` 的 5 类型 × 多场景验证 |
| 测试标签数组 | 同上 | 743–745 | `labels[8] = "general_gaussian"` |
| 证据记录 | 同上 | 757 | `ev::record<T,double>(td::OperatorId::general_gaussian, ...)` |

---

### 三、CPU 调用链、数据结构与逐步算法

#### 3.1 CPU reference `general_gaussian_typed_cpu`

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/src/windows/windows_typed.cpp` / `general_gaussian_typed_cpu` / 第 341–358 行。

```cpp
template <class T>
std::vector<double> general_gaussian_typed_cpu(
    int length, T power, T width, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("general_gaussian: length must be nonnegative");
    if (length == 0) return {};
    if (length == 1) return {1.0};
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    const double power_value = static_cast<double>(host_load(power));
    const double width_value = static_cast<double>(host_load(width));
    for (int index = 0; index < length; ++index) {
        const double x = (index - 0.5 * (effective - 1)) / width_value;
        out[index] = std::exp(
            -0.5 * std::pow(std::fabs(x), 2.0 * power_value));
    }
    return out;
}
```

**逐步逻辑**：

1. **长度守卫**：`length < 0` 抛 `std::invalid_argument`；`length == 0` 返回空 `vector<double>`；`length == 1` 返回 `{1.0}`（单点退化）。
2. **对称/周期有效长度**：`effective = symmetric ? length : length + 1`。`sym=true`（对称）时 `effective = length`；`sym=false`（周期）时 `effective = length + 1`，但只输出前 `length` 个点（最后一点不写出，等价于 cuSignal Python 的 `_extend`/`_truncate` 截断）。
3. **参数提升**：`power_value = (double)host_load(power)` 和 `width_value = (double)host_load(width)`。`host_load`（`windows_typed.cpp:17-21`）调用 `detail::SimpleSignalTypePolicy<T>::load`，将 `__half` 经 `__half2float`、整数经 `static_cast<float>` 一律先提升为 `float`，再由 `static_cast<double>` 提升为 `double`。
4. **逐点公式**：对 `index = 0..length-1`，
   - 中心化位置 $x = \dfrac{\text{index} - 0.5 \cdot (\text{effective} - 1)}{\text{width\_value}}$
   - 窗值 $w[\text{index}] = \exp\!\left(-0.5 \cdot |x|^{2 \cdot \text{power\_value}}\right)$
5. **全程 FP64**：`std::exp`、`std::pow`、`std::fabs` 均为 `<cmath>` 双精度重载。
6. **不检查 `power` 或 `width` 合法性**：与 cuSignal Python 一致。当 `width == 0` 时 `x = inf` 或 `nan`，后续 `exp(-0.5 * pow(inf, ...))` 产生 `0` 或 `nan`（具体取决于 `power`），与 Python 行为一致。
7. **不做归一化**：广义高斯窗峰值自然为 1（中心点 $x=0$ 时 $|0|^{2p}=0$，$\exp(0)=1$），无需归一化步骤。与 Python 一致。

**数据结构**：

- 输入：`length`（int）、`power`（模板参数 T）、`width`（模板参数 T）、`symmetric`（bool）。
- 中间：`effective`（int）、`power_value`（double）、`width_value`（double）、`x`（double）。
- 输出：`std::vector<double> out(length)`，宿主内存。

#### 3.2 GPU 公开入口 `general_gaussian_device`（host 包装层）

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/src/windows/windows_typed.cpp` / `general_gaussian_device` / 第 360–375 行。

```cpp
template <class T>
void general_gaussian_device(
    int length, T power, T width, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("general_gaussian_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
        general_gaussian_fp32_compute_device(
            length, power, width, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

**逐步逻辑**：

1. **长度与输出尺寸校验**：`length < 0` 或 `out.size() != length` 抛 `std::invalid_argument`。
2. **分配 FP32 临时 device 缓冲区**：`DeviceArray<float> computed(out.size())`。这是函数内部 workspace，不对调用者暴露。
3. **退化分支**：`length == 1` 时直接以 `DeviceArray<float>::from_host({1.0F})` 写入单点 1.0F；`length == 0` 时跳过计算（`computed` 保持空），直接进入 finalize。
4. **正常分支**：`length > 1` 时调用 `general_gaussian_fp32_compute_device(length, power, width, computed, symmetric)`，由 GPU kernel 完成 FP32 计算。
5. **FP32 → FP64 拓宽**：`out = cuda_utils::finalize_fp64_device_storage(computed)`。该函数完成 D2H（device→host）→ Host 端 `float→double` 逐元素拓宽 → H2D（host→device）回写为 `DeviceArray<double>`。GPU 端不执行任何 FP64 算术。

---

### 四、GPU 调用链、host wrapper、kernel 启动与 device 计算逻辑

#### 4.1 GPU 调用链总览

```
general_gaussian_device(length, power, width, out, symmetric)     [windows_typed.cpp:360-375]
  │
  ├── 长度/尺寸校验
  ├── 分配 DeviceArray<float> computed
  ├── length == 1 → computed = {1.0F}（单点退化，不启动 kernel）
  ├── length == 0 → 跳过
  ├── length > 1 → general_gaussian_fp32_compute_device(...)       [windows_typed.cu:75]
  │       │
  │       ├── static_assert(is_simple_signal_input_v<T>)
  │       ├── n <= 1 || o.size() != n → 抛异常
  │       └── cuda_utils::launch_1d_kernel(
  │               windows_detail::general_gaussian_kernel<float, T>,
  │               o.size(),
  │               n, p, w,
  │               o.data(),
  │               sym ? n : n + 1)                                  // effective_count
  │
  └── out = finalize_fp64_device_storage(computed)
          │
          ├── computed.to_host()          → D2H: GPU FP32 → Host vector<float>
          ├── finalize_fp64_on_host(...)  → Host: float → double 逐元素拓宽
          └── DeviceArray<double>::from_host → H2D: Host vector<double> → GPU FP64
```

#### 4.2 host 包装层职责

`general_gaussian_device`（`windows_typed.cpp:360-375`）只负责：

- 长度/尺寸校验；
- 分配 FP32 临时缓冲区；
- 单点退化处理；
- 调用 FP32 计算调度层；
- 调用 `finalize_fp64_device_storage` 完成 FP32→FP64 拓宽与回写。

它**不直接** launch kernel，而是委托给 `general_gaussian_fp32_compute_device`。

#### 4.3 FP32 计算调度层 `general_gaussian_fp32_compute_device`

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/src/windows/windows_typed.cu` / `general_gaussian_fp32_compute_device` / 第 75 行（单行实现）。

展开后等价于：

```cpp
template <class T>
void general_gaussian_fp32_compute_device(
    int n, T p, T w, DeviceArray<float>& o, bool sym)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (n <= 1 || o.size() != size_t(n))
        throw std::invalid_argument(
            "general_gaussian_device: compute output size must equal n and n must exceed one");
    cuda_utils::launch_1d_kernel(
        windows_detail::general_gaussian_kernel<float, T>,
        o.size(),
        n, p, w,
        o.data(),
        sym ? n : n + 1);   // effective_count
}
```

**关键点**：

1. **`static_assert`**：编译期检查 `T` 属于 `{float, __half, int32_t, int16_t, int8_t}` 五种类型之一。
2. **运行时守卫**：`n <= 1` 或 `o.size() != n` 抛异常。注意此处 `n <= 1` 的检查在 host 包装层已经过滤（`length == 1` 走退化分支），此处是二级防御。
3. **kernel 实例化**：`windows_detail::general_gaussian_kernel<float, T>`，即 `T=float`（FP32 输出），`Parameter=T`（用户 dtype）。模板参数顺序为 `<typename T, typename Parameter>`，与 host 实例化对应。
4. **`effective_count` 传入**：`sym ? n : n + 1`，与 CPU 的 `effective = symmetric ? length : length + 1` 一致。
5. **`launch_1d_kernel` 参数**：第一个参数为 kernel 函数指针，第二个为元素数（grid 大小依据），其余为 kernel 参数包。

#### 4.4 device kernel `general_gaussian_kernel`

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/src/windows/windows_kernels.cuh` / `general_gaussian_kernel` / 第 84–107 行。

```cpp
template <typename T, typename Parameter>
__global__ void general_gaussian_kernel(
    int output_count,
    Parameter power_input,
    Parameter width_input,
    T* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(power_input));
    const Scalar power = window_load(power_input);
    const Scalar width = window_load(width_input);
    const Scalar position =
        (static_cast<Scalar>(index) - static_cast<Scalar>(0.5F) *
            static_cast<Scalar>(effective_count - 1)) / width;
    output[index] = window_store<T>(exp(
        static_cast<Scalar>(-0.5F) * pow(
            fabs(position), static_cast<Scalar>(2) * power)));
}
```

**逐线程执行步骤**：

1. **线程索引**：`index = blockIdx.x * blockDim.x + threadIdx.x`（一维）。
2. **边界检查**：`if (index >= output_count) return;`。
3. **单点退化**：`if (effective_count == 1)` 写 1.0F 并返回。该分支在 host 包装层已对 `length == 1` 做了短路，此处为 kernel 内二级防御。
4. **类型推导**：`using Scalar = decltype(window_load(power_input))`。`window_load`（`windows_kernels.cuh:10-18`）对 `is_simple_signal_input_v<T>` 的类型调用 `SimpleSignalTypePolicy<T>::load`，将 `__half`→`float`、整数→`float`、`float`→`float`。因此 `Scalar` 在所有 5 种 `Parameter` 下均为 `float`。
5. **参数加载**：`power = window_load(power_input)`、`width = window_load(width_input)`，均得到 `float`。
6. **位置计算**：`position = (index - 0.5F * (effective_count - 1)) / width`。与 CPU 的 $x = \dfrac{\text{index} - 0.5 \cdot (\text{effective} - 1)}{\text{width\_value}}$ 数学等价，但 cusignal_cpp 把 $n/\text{sig}$ 合并为单次 `position` 计算（CPU 也是一次计算 `x`），而 Python 先算 $n$ 再除以 $\text{sig}$。
7. **核心公式**：`output[index] = window_store<T>(exp(-0.5F * pow(fabs(position), 2.0F * power)))`。使用 FP32 的 `expf`/`powf`/`fabsf`（CUDA `__device__` 重载）。
8. **输出存储**：`window_store<T>`（`windows_kernels.cuh:20-28`）对 `is_simple_signal_input_v<T>` 类型调用 `SimpleSignalTypePolicy<T>::store(static_cast<float>(value))`。由于此处 `T=float`，`store` 为恒等返回。

**模板参数说明**：

- `T`：输出元素类型。host 调度层固定实例化为 `float`（FP32 计算缓冲）。
- `Parameter`：输入参数类型，等于用户请求的 dtype `T`（`float`/`__half`/`int32_t`/`int16_t`/`int8_t`）。`power_input` 和 `width_input` 按值传递，kernel 内通过 `window_load` 提升为 `float`。

---

### 五、template、类型分发、显式实例化、内存布局与数据搬运

#### 5.1 模板层级

```
template <class T>                          // 用户 dtype，5 种之一
void general_gaussian_device(int, T, T, DeviceArray<double>&, bool);
    │
    └── template <class T>
        void general_gaussian_fp32_compute_device(int, T, T, DeviceArray<float>&, bool);
            │
            └── template <typename T, typename Parameter>
                __global__ void general_gaussian_kernel(int, Parameter, Parameter, T*, int);
                │   实例化为 general_gaussian_kernel<float, T_user>
                │   T = float（FP32 输出）
                │   Parameter = T_user（用户 dtype）
                │
                └── Scalar = float（window_load 后统一为 FP32）
```

#### 5.2 CPU 显式实例化

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/src/windows/windows_typed.cpp` / `INSTANTIATE_WINDOWS_CPU` 宏 / 第 562–587 行。

宏展开中对 `general_gaussian` 的两行为：

```cpp
template std::vector<double> general_gaussian_typed_cpu(int, T, T, bool);
template void general_gaussian_device(int, T, T, DeviceArray<double>&, bool);
```

实例化列表（第 589–593 行）：

```cpp
INSTANTIATE_WINDOWS_CPU(float);
INSTANTIATE_WINDOWS_CPU(__half);
INSTANTIATE_WINDOWS_CPU(std::int32_t);
INSTANTIATE_WINDOWS_CPU(std::int16_t);
INSTANTIATE_WINDOWS_CPU(std::int8_t);
```

#### 5.3 GPU 显式实例化

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/src/windows/windows_typed.cu` / `INSTANTIATE` 宏 / 第 116–127 行。

宏展开中对 `general_gaussian` 的一行为：

```cpp
template void general_gaussian_fp32_compute_device(int, T, T, DeviceArray<float>&, bool);
```

实例化列表（第 126 行）：

```cpp
INSTANTIATE(float); INSTANTIATE(__half); INSTANTIATE(std::int32_t); INSTANTIATE(std::int16_t); INSTANTIATE(std::int8_t);
```

#### 5.4 五种输入类型的 `SimpleSignalTypePolicy` 行为

**路径**：`cusignal_cpp/src/cuda_utils/simple_signal_typed.h`

| `T` | `load(value)` 行为 | `store(float)` 行为 | kernel 内 `Scalar` |
| --- | --- | --- | --- |
| `float` | `static_cast<float>`（恒等） | `static_cast<float>`（恒等） | `float` |
| `__half` | `__half2float(value)` | `__float2half_rn(value)` | `float` |
| `std::int32_t` | `static_cast<float>` | 饱和截断到 `[INT32_MIN, INT32_MAX]` 后 `static_cast<int32_t>` | `float` |
| `std::int16_t` | `static_cast<float>` | 饱和截断到 `[INT16_MIN, INT16_MAX]` 后 `static_cast<int16_t>` | `float` |
| `std::int8_t` | `static_cast<float>` | 饱和截断到 `[INT8_MIN, INT8_MAX]` 后 `static_cast<int8_t>` | `float` |

**关键**：所有 5 种 `Parameter` 在 kernel 内经 `window_load` 后均为 `float`；`T` 固定为 `float`（host 调度层 `general_gaussian_kernel<float, T_user>`）。因此 kernel 内**全程 FP32**，不依赖用户 dtype 的原生算术。

#### 5.5 内存布局与数据搬运

```
Host: 用户构造 power (T), width (T)            → 按值传递给 general_gaussian_device
Host: DeviceArray<double> out(length)         → 调用者预分配，FP64 存储

GPU 路径（length > 1）:
  Host power/width (T) ──按值──→ kernel 参数 Parameter power_input/width_input
                                        │
                                   window_load → float
                                        │
  DeviceArray<float> computed(length) ← kernel 写入 FP32 结果
                                        │
  D2H: computed.to_host() → Host vector<float>
                                        │
  Host: float → double 逐元素拓宽
                                        │
  H2D: DeviceArray<double>::from_host → out (FP64)
```

**搬运次数**：1 次 H2D（finalize 阶段，将拓宽后的 FP64 写回 device）。`power` 和 `width` 为按值传递的标量，无独立 H2D。`computed` 的 D2H 在 `finalize_fp64_device_storage` 内部完成。

---

### 六、grid、block、线程索引与每线程数据范围

#### 6.1 launch 配置

`cuda_utils::launch_1d_kernel`（`cusignal_cpp/src/cuda_utils/kernel_launch.h`）使用：

| 配置项 | 值 | 说明 |
| --- | --- | --- |
| block 大小 | 256 线程 | 一维 block，`dim3(256)` |
| grid 大小 | $\lceil N / 256 \rceil$ | `dim3(div_up(elements, 256))`，`elements = o.size() = length` |
| stream | `nullptr`（默认流） | 同步提交，无显式 `cudaStreamSynchronize` |
| 共享内存 | 0 字节 | 无共享内存使用 |

#### 6.2 线程索引与数据范围

| 项目 | 说明 |
| --- | --- |
| 计算维度 | 一维（1D grid × 1D block） |
| 线程索引 | `index = blockIdx.x * blockDim.x + threadIdx.x` |
| 每个线程负责的数据范围 | 恰好 1 个输出元素 `output[index]` |
| 有效线程数 | $N = \text{length}$（`index < output_count` 的线程） |
| 总线程数 | $\lceil N / 256 \rceil \times 256$（可能超过 $N$，越界线程提前 `return`） |
| 线程间同步 | 无（每个线程独立计算，无数据依赖） |
| 共享内存 | 无 |
| warp divergence 风险 | `effective_count == 1` 单点退化分支在正常调用下不会触发（host 已短路）；`width == 0` 时所有线程走相同 `pow`/`exp` 路径，无 warp 分歧 |

---

### 七、同步、临时缓冲区、FFT/平台库调用与错误处理

#### 7.1 同步

- kernel 使用默认流（`stream = nullptr`），launch 后**不显式同步**。
- `finalize_fp64_device_storage(computed)` 内部：`computed.to_host()` 隐含 `cudaMemcpy` D2H 同步；`DeviceArray<double>::from_host` 为 H2D 同步。
- 调用者读取 `out` 前无需额外同步：因为 D2H → Host 拓宽 → H2D 的序列已确保数据就绪。

#### 7.2 临时缓冲区

| 缓冲区 | 类型 | 大小 | 生命周期 | 位置 |
| --- | --- | --- | --- | --- |
| `computed` | `DeviceArray<float>` | $N = \text{length}$ | `general_gaussian_device` 函数作用域 | GPU 显存 |
| Host 中间向量 | `vector<float>` → `vector<double>` | $N$ | `finalize_fp64_device_storage` 内部作用域 | Host 内存 |

无跨调用复用的 workspace，无 resident 缓冲区（与 `hamming_resident_device` 不同，`general_gaussian` 无 resident 接口）。

#### 7.3 FFT / 平台库调用

- **无 FFT**：`general_gaussian` 是时域解析公式，不涉及频域变换。
- **无 thrust reduction**：与 `chebwin` 不同，广义高斯窗无需求最大值归一化（峰值自然为 1）。
- **无归一化步骤**：公式 $w = \exp(-0.5 \cdot |x|^{2p})$ 在 $x=0$ 时自然为 1。
- **不调用 `loop_prep`**：所有常量（`power`、`width`、`position`）在 kernel 内逐线程计算。与 cuSignal Python 的 `_general_gaussian_kernel` 一致（Python 也没有 `loop_prep`）。
- **不调用 ZQ500 dlfft / DLI 库**：纯 elementwise 计算。

#### 7.4 错误处理

| 错误条件 | 抛出位置 | 异常类型 |
| --- | --- | --- |
| `length < 0` | `general_gaussian_typed_cpu`（`windows_typed.cpp:345`） | `std::invalid_argument("general_gaussian: length must be nonnegative")` |
| `length < 0` 或 `out.size() != length` | `general_gaussian_device`（`windows_typed.cpp:364-365`） | `std::invalid_argument("general_gaussian_device: output size must equal requested length")` |
| `n <= 1` 或 `o.size() != n` | `general_gaussian_fp32_compute_device`（`windows_typed.cu:75`） | `std::invalid_argument("general_gaussian_device: compute output size must equal n and n must exceed one")` |

**不检查的非法输入**（与 cuSignal Python 一致）：

- `power <= 0`：不抛异常，公式按数学结果传播（`pow(fabs(x), 2*power)` 当 `power < 0` 时可能产生 `inf` 或 `nan`）。
- `width == 0`：不抛异常，`position = (index - 0.5*(effective-1)) / 0` 产生 `inf` 或 `nan`，后续 `exp(-0.5 * pow(inf/nan, ...))` 产生 `nan`。测试 `gaussian_zero_width[2]` 期望为 `nan`。
- `width < 0`：不抛异常，`position` 符号翻转，但 `fabs(position)` 使结果与 `|width|` 一致。

CUDA launch 失败 / kernel 内错误走项目统一异常机制（不在 `general_gaussian` 代码内显式 `cudaGetLastError`）。

---

### 八、数学/物理原理到 CPU 代码、GPU 代码的三方映射

以下原理编号对应 `Learning/operators/windows/general_gaussian/数学物理原理.md`。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| **原理 1**：广义正态分布 PDF $f(x) \propto \exp(-|x/\alpha|^\beta)$，窗函数取 $-\frac{1}{2}|x/\sigma|^{2p}$ 形式 | `_general_gaussian_kernel` kernel body `w = exp(-0.5 * pow(abs(n/sig), 2.0 * p))`（`windows.py:1430` 基准 / `1484` 学习副本） | `out[index] = std::exp(-0.5 * std::pow(std::fabs(x), 2.0 * power_value))`（`windows_typed.cpp:354-355`） | `output[index] = window_store<T>(exp(-0.5F * pow(fabs(position), 2.0F * power)))`（`windows_kernels.cuh:104-106`） | 三者数学等价。Python 全程 FP64；CPU 全程 FP64；GPU 全程 FP32。Python 用 `abs`/`pow`/`exp`（CuPy ElementwiseKernel 内 C++ 代码，实际 FP64），C++ CPU 用 `std::fabs`/`std::pow`/`std::exp`，GPU 用 CUDA `fabsf`/`powf`/`expf`（FP32 重载） |
| **原理 2**：高斯窗时域定义 $w[n] = \exp(-0.5 |n/\sigma|^{2})$（$p=1$ 特例） | 同上 kernel，`p=1` 时 `2.0 * p = 2`，公式退化为标准高斯 | 同上，`power_value = 1` 时 `2.0 * power_value = 2` | 同上，`power = 1` 时 `2.0F * power = 2` | 三者均通过同一公式参数化覆盖 $p=1$，不单独为高斯窗写专用 kernel |
| **原理 3**：半功率点 $n_{\text{half}} = (2\ln 2)^{1/(2p)} \sigma$ | docstring 中 `.. math:: (2 \log(2))^{1/(2 p)} \sigma`（`windows.py:1469` 基准 / 学习副本对应行） | docstring 注释（`windows_typed.h:91-104` 的 `@brief`/`@details`）提及广义高斯窗但未显式写半功率公式 | 无显式半功率点代码，kernel 只算 $w[n]$ | 半功率点是窗的频域特性推导结论，不在生成代码中显式计算；三者均仅在 docstring 中提及 |
| **原理 4**：超高斯窗（$p>1$）平顶特性 | 同上 kernel，`p>1` 时指数 $2p > 2$，中心更平坦 | 同上公式 | 同上公式 | 三者通过同一参数化公式覆盖 $p>1$，无专用平顶分支 |
| **原理 5**：拉普拉斯形窗（$p=0.5$）重尾特性 | 同上 kernel，`p=0.5` 时 `2.0 * p = 1`，公式退化为 $\exp(-0.5 |n/\sigma|)$ | 同上公式 | 同上公式 | 三者通过同一参数化公式覆盖 $p=0.5$，无专用拉普拉斯分支 |
| **原理 6**：加窗原理——时域乘法对应频域卷积 | 不在 `general_gaussian` 生成代码内体现，属下游使用 | 不在生成代码内体现 | 不在生成代码内体现 | 加窗是窗函数的使用方式，不是窗函数生成逻辑的一部分 |
| **中心化位置** $n = i - (M-1)/2$ | `const double n { i - (_ind.size() - 1.0) * 0.5 }`（`windows.py:1429` 基准 / `1482` 学习副本） | `const double x = (index - 0.5 * (effective - 1)) / width_value`（`windows_typed.cpp:353`） | `const Scalar position = (index - 0.5F * (effective_count - 1)) / width`（`windows_kernels.cuh:101-103`） | Python 先算 $n$ 再除以 `sig`；C++ CPU 和 GPU 把 $n/\text{width}$ 合并为单次 `x`/`position` 计算。数学等价 |
| **对称/周期长度** `effective = sym ? M : M+1` | `_extend(M, sym)` 返回 `(M+1, True)` 或 `(M, False)`，再 `_truncate(w, needs_trunc)` 截断（`windows.py:1500/1504` 基准 / `1558/1564` 学习副本） | `const int effective = symmetric ? length : length + 1`（`windows_typed.cpp:348`），循环只写前 `length` 个点 | `sym ? n : n + 1` 作为 `effective_count` 传入 kernel（`windows_typed.cu:75`），kernel 输出 `length` 个点 | Python 用 extend/truncate 两步；C++ 用 `effective_count` 一步。数学等价 |
| **长度守卫** `M <= 0` | `_len_guards(M)` 返回 `True` → `cp.ones(M)` 返回空（`windows.py:1498` 基准 / `1555` 学习副本） | `length < 0` 抛异常；`length == 0` 返回 `{}`；`length == 1` 返回 `{1.0}`（`windows_typed.cpp:345-347`） | `length < 0` 或 `out.size() != length` 抛异常；`length == 1` 写 `{1.0F}`；`length == 0` 跳过（`windows_typed.cpp:364-370`） | Python 对 `M<=0` 返回空/全 1；C++ 对负数抛异常（更严格），对 0 返回空，对 1 返回 `{1.0}`。行为差异：Python `_len_guards` 还处理非整数 M，C++ 不处理（C++ 的 `int` 类型天然拒绝非整数） |
| **单点退化** $M=1 \Rightarrow w[0]=1$ | `_len_guards(1)` 返回 `True` → `cp.ones(1)` = `[1.0]` | `if (length == 1) return {1.0}`（`windows_typed.cpp:347`） | `if (length == 1) computed = DeviceArray<float>::from_host({1.0F})`（`windows_typed.cpp:368-369`）；kernel 内 `if (effective_count == 1)` 二级防御（`windows_kernels.cuh:94-96`） | 三者均退化到 1.0，但路径不同：Python 走 `_len_guards`，C++ 走显式分支 |
| **不检查 `power`/`width` 合法性** | 不检查（`windows.py:1493-1504` 无 guard） | 不检查（`windows_typed.cpp:341-358` 无 guard） | 不检查（`windows_typed.cu:75` 和 `windows_kernels.cuh:84-107` 无 guard） | 三者一致：`width == 0` 时产生 `nan`/`inf`，`power <= 0` 时产生 `inf`/`nan`，均按 IEEE 754 传播 |
| **不做归一化** | 不归一化（公式峰值自然为 1） | 不归一化 | 不归一化 | 三者一致：广义高斯窗 $w[0] = \exp(0) = 1$，无需除以最大值 |

---

### 九、与 cuSignal Python 实现的相同、等价替换与有意不同

#### 9.1 相同部分

| 方面 | 说明 |
| --- | --- |
| 核心公式 | $w[n] = \exp(-0.5 \cdot |n/\sigma|^{2p})$，三者逐字对应 |
| 形状参数 $p$ 语义 | $p=1$ 高斯、$p=0.5$ 拉普拉斯、$p>1$ 超高斯，三者通过同一参数化公式覆盖 |
| 对称/周期语义 | `sym=true` 用 $M$ 点，`sym=false` 用 $M+1$ 点内部长度再截前 $M$ 点 |
| 长度守卫 | `M<=0` 返回空，`M=1` 返回 `{1.0}` |
| 不检查 `power`/`width` | 三者均不检查，`width==0` 产生 `nan`，与 Python 测试 `gaussian_zero_width[2]` 期望一致 |
| 不做归一化 | 峰值自然为 1，无需归一化 |
| 无 `loop_prep` | Python 的 `_general_gaussian_kernel` 没有 `loop_prep`，C++ kernel 也在 kernel 内逐线程计算常量，两者一致 |

#### 9.2 等价替换

| Python | C++ | 等价原因 |
| --- | --- | --- |
| CuPy `ElementwiseKernel` 自动 GPU 并行 | `launch_1d_kernel()` + CUDA `__global__` kernel | 均为逐元素 GPU 并行 |
| `_extend(M, sym)` + `_truncate(w, needs_trunc)` | `effective = sym ? length : length + 1`，循环只写前 `length` 个点 | Python 两步 extend/truncate 等价于 C++ 一步 `effective_count` |
| `abs`/`pow`/`exp`（CuPy kernel 内 C++ 代码，FP64） | CPU `std::fabs`/`std::pow`/`std::exp`（FP64）；GPU `fabsf`/`powf`/`expf`（FP32） | 数学函数相同，精度不同 |
| `cp.asarray` 隐式类型转换 | `SimpleSignalTypePolicy<T>::load` + `static_cast<float/double>` | 均为类型提升到计算精度 |
| `_ind.size()` 在 kernel 内获取线程总数 | `effective_count` 作为参数传入 kernel | Python 从 CuPy 内置变量获取，C++ 显式传参 |

#### 9.3 有意不同

| 差异 | Python | C++ | 设计意图 |
| --- | --- | --- | --- |
| **计算精度** | `_general_gaussian_kernel` 声明 `float64 p, float64 sig, float64 w`，全程 FP64 | GPU kernel 固定 FP32（`expf`/`powf`/`fabsf`），CPU 固定 FP64 | ZQ500 平台 FP64 性能远低于 FP32；GPU 端 FP32 + Host 拓宽是项目统一策略（与 `hamming`、`kaiser`、`chebwin` 等窗一致） |
| **输出 dtype** | 固定 `float64`（kernel 声明 `"float64 w"`） | 固定 `double`（`DeviceArray<double>`），但 GPU 计算为 FP32，经 Host 拓宽为 FP64 | 接口 dtype 均为 FP64，但 C++ 的 FP64 是 Host 拓宽后的存储格式，不是 GPU 计算精度 |
| **负数 `length` 处理** | `_len_guards(M)` 对 `M<=0` 返回 `True`，进而 `cp.ones(M)` 返回空数组（不抛异常） | `length < 0` 抛 `std::invalid_argument` | C++ 更严格：负长度是调用者错误，应抛异常而非静默返回空 |
| **`gaussian` 算子专用 kernel** | Python 有独立的 `gaussian` 算子和 `_gaussian_kernel`（`windows.py:1351-1361` 基准 / `1399-1409` 学习副本），使用 `n*n/(2*sig2)` 而非 `pow` | cusignal_cpp **不提供** `gaussian` 的等价物；`general_gaussian` 的 `p=1` 即可替代 | 工程简化：避免维护两个数学等价的 kernel。Python 的 `gaussian` 用 `n*n` 代替 `pow(n, 2)` 是微优化，C++ 不做此优化 |
| **位置计算合并** | 先算 `n = i - (M-1)*0.5`，再 `n/sig` 在 `pow` 内部 | CPU 和 GPU 把 `n/width` 合并为单次 `x`/`position` 计算 | 减少一次中间变量，数学等价 |
| **`effective_count` 传参方式** | `_ind.size()` 在 kernel 内由 CuPy 运行时提供 | `effective_count` 作为 kernel 参数显式传入 | CuPy 的 `_ind.size()` 是其 ElementwiseKernel 的内置变量；C++ 需显式传参 |

---

### 十、精度、dtype、边界条件及 ZQ500 平台限制

#### 10.1 精度分析

| 路径 | 计算精度 | `fabs`/`pow`/`exp` | $\pi$ 精度 |
| --- | --- | --- | --- |
| CPU | FP64（`double`） | `std::fabs`/`std::pow`/`std::exp`（`<cmath>` 双精度重载） | 不涉及 $\pi$ |
| GPU kernel | FP32（`float`） | CUDA `fabsf`/`powf`/`expf`（FP32 设备重载） | 不涉及 $\pi$ |
| Host 拓宽 | FP32 → FP64 无损（FP32 值域内精确拓宽） | — | — |

**精度风险**：

- GPU FP32 的 `powf(fabsf(position), 2.0F * power)` 精度约 7 位有效数字。对 `position` 较大或 `power` 较大的情况，指数运算的相对误差可能放大。
- 测试容差 `matches_fp64` 使用 `4e-2`（`e3_wave_window_type_smoke.cu:83`），远大于 FP32 机器 epsilon（约 $1.2 \times 10^{-7}$），表明项目预期 GPU FP32 与 CPU FP64 之间存在可见的精度差异。
- `width == 0` 时 `position = inf`，`powf(inf, ...)` 和 `expf(...)` 的行为依赖 `power` 值：`power > 0` 时 `pow(inf, positive) = inf`，`exp(-0.5 * inf) = 0`；`power == 0` 时 `pow(inf, 0) = 1`，`exp(-0.5) ≈ 0.606`；`power < 0` 时 `pow(inf, negative) = 0`，`exp(0) = 1`。但 `width == 0` 时 `position` 的符号取决于 `index - 0.5*(effective-1)`，中心点 `position = 0/0 = nan`，`pow(nan, ...) = nan`，`exp(nan) = nan`。测试 `gaussian_zero_width[2]` 期望中心点为 `nan`。

#### 10.2 dtype 策略

| 输入 `T`（`Parameter`） | `window_load` 后 `Scalar` | kernel 输出 `T` | 最终输出 |
| --- | --- | --- | --- |
| `float` | `float` | `float` | `double`（Host 拓宽） |
| `__half` | `float`（`__half2float` 提升） | `float` | `double` |
| `int32_t` | `float`（`static_cast`） | `float` | `double` |
| `int16_t` | `float`（`static_cast`） | `float` | `double` |
| `int8_t` | `float`（`static_cast`） | `float` | `double` |

**ZQ500 平台限制**：

- FP16 输入经 `__half2float` 提升，kernel 内全程 FP32，无 FP16 算术。
- 整数输入经 `static_cast<float>` 转换，`int32_t` 超过 $2^{24} = 16777216$ 时丢失精度（FP32 尾数 24 位）。但 `power` 和 `width` 作为窗函数参数，典型值远小于此阈值。
- GPU 端不执行 FP64 算术，符合 ZQ500 DLI V2 的 FP32 计算策略。
- 不调用 ZQ500 dlfft / DLI 库，纯 elementwise CUDA 计算，无平台库依赖。

#### 10.3 边界条件处理

| 边界 | CPU 行为 | GPU 行为 | Python 行为 |
| --- | --- | --- | --- |
| `length < 0` | 抛 `std::invalid_argument` | 抛 `std::invalid_argument`（host 包装层） | `_len_guards` 返回 `True` → `cp.ones(M)` 返回空数组 |
| `length == 0` | 返回空 `vector<double>` | `computed` 为空，finalize 后 `out` 为空 | `_len_guards` 返回 `True` → `cp.ones(0)` 返回空数组 |
| `length == 1` | 返回 `{1.0}` | `computed = {1.0F}`，不启动 kernel | `_len_guards` 返回 `True` → `cp.ones(1)` = `[1.0]` |
| `width == 0` | `x = inf/nan`，`exp(-0.5 * pow(inf/nan, ...))` 产生 `0`/`nan` | 同 CPU（FP32） | 同 CPU（FP64） |
| `width < 0` | `x` 符号翻转，但 `fabs(x)` 使结果与 `|width|` 一致 | 同 CPU | 同 CPU |
| `power == 0` | `pow(fabs(x), 0) = 1`，`exp(-0.5) ≈ 0.606`（所有点相同） | 同 CPU（FP32） | 同 CPU |
| `power < 0` | `pow(fabs(x), negative)`：`fabs(x) > 0` 时为 `inf`，`exp(-inf) = 0`；`fabs(x) == 0`（中心点）时 `pow(0, negative) = inf`，`exp(-inf) = 0` | 同 CPU（FP32） | 同 CPU |
| NaN 输入 `power`/`width` | `window_load` 传播 NaN，后续计算产生 NaN | 同 CPU | 同 CPU |
| 空输出 `out.size() == 0` 且 `length == 0` | 不适用（CPU 返回空 vector） | 通过校验，跳过计算 | 不适用 |

---

### 十一、测试如何验证正确性，以及仍未覆盖的风险

#### 11.1 测试入口

**完整提交 SHA + 路径 + 行号**：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` / `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` / `run_type<T>` 内 `ok[8]` / 第 524–577 行。

测试在 `run_type<T>` 模板函数内，`ok[8]` 对应 `general_gaussian`。每种 `T`（`float`、`__half`、`int32_t`、`int16_t`、`int8_t`）各跑一遍，共 5 种类型。`main()`（第 773–783 行）调用 5 次 `run_type<T>`，总 14 个算子 × 5 类型 = 70 个检查点。

#### 11.2 测试用例

| 测试项 | 行号 | 输入 | 验证方式 |
| --- | --- | --- | --- |
| 标准对称窗 | 525–529 | `general_gaussian_device(5, convert<T>(1), convert<T>(2), fixed_fp64_window_out)` | GPU vs `general_gaussian_typed_cpu(5, convert<T>(1), convert<T>(2))`，容差 4e-2 |
| 负 width | 530–534 | `general_gaussian_device(5, convert<T>(1), convert<T>(-2), ...)` | GPU vs CPU，验证公式对负 width 的稳健性 |
| width=0（期望 nan） | 535–539 | `general_gaussian_device(5, convert<T>(1), convert<T>(0), ...)` | 提取 `gaussian_zero_width` 和 `gaussian_zero_width_reference`，后续验证 `std::isnan(gaussian_zero_width[2])` |
| 单点窗 | 540–541 | `general_gaussian_device<T>(1, convert<T>(1), convert<T>(0), singleton_fp64_window_out)` | 验证 `singleton_fp64_window_out.to_host() == {1.0}` |
| 空窗 | 542–543 | `general_gaussian_device<T>(0, convert<T>(1), convert<T>(0), empty_fp64_window_out)` | 验证 `empty_fp64_window_out.empty()` |
| 负长度拒绝（GPU） | 544–551 | `general_gaussian_device<T>(-1, convert<T>(1), convert<T>(2), empty_fp64_window_out)` | 期望抛 `std::invalid_argument` |
| 负长度拒绝（CPU） | 552–557 | `general_gaussian_typed_cpu<T>(-1, convert<T>(1), convert<T>(2))` | 期望抛 `std::invalid_argument` |
| 周期窗（sym=false） | 558–565 | `general_gaussian_device<T>(4, convert<T>(1), convert<T>(2), gaussian_periodic_out, false)` | GPU vs `general_gaussian_typed_cpu(4, convert<T>(1), convert<T>(2), false)`，容差 4e-2 |
| width=0 中心点 nan | 566–569 | 来自 535–539 的 `gaussian_zero_width` 和 `gaussian_zero_width_reference` | `gaussian_zero_width.size() == 5` && `gaussian_zero_width_reference.size() == 5` && `std::isnan(gaussian_zero_width[2])` && `std::isnan(gaussian_zero_width_reference[2])` |
| 单点 CPU | 570–572 | `general_gaussian_typed_cpu(1, convert<T>(1), convert<T>(0))` | 验证 `== {1.0}` |
| 空窗 CPU | 573–575 | `general_gaussian_typed_cpu(0, convert<T>(1), convert<T>(0))` | 验证 `.empty()` |

#### 11.3 测试框架

- 使用 `matches_fp64(actual, reference)`（第 81–84 行）调用 `test_evidence::f_observe_real(actual, reference, 4.0e-2)`，容差 4e-2。
- NaN 对 NaN：`std::isnan(gaussian_zero_width[2])` 直接判断，不通过容差比较。
- 通过 `test_evidence::f_begin_accuracy(type_dispatch::OperatorId::general_gaussian)`（第 524 行）标记算子边界。
- 通过 `ev::record<T, double>(td::OperatorId::general_gaussian, fixed_fp64_window_out.size(), ok[8])`（第 757 行）输出结构化 E3 证据。
- 5 种 dtype 各跑一遍：`run_type<float>("fp32")`、`run_type<__half>("fp16")`、`run_type<std::int32_t>("int32")`、`run_type<std::int16_t>("int16")`、`run_type<std::int8_t>("int8")`（第 776–780 行）。

#### 11.4 `ok[8]` 通过条件（第 561–577 行）

```cpp
ok[8] = gaussian_positive_matches && gaussian_negative_matches &&
    matches_fp64(gaussian_periodic_out.to_host(),
        general_gaussian_typed_cpu(4, convert<T>(1), convert<T>(2), false)) &&
    gaussian_zero_width.size() == 5 &&
    gaussian_zero_width_reference.size() == 5 &&
    std::isnan(gaussian_zero_width[2]) &&
    std::isnan(gaussian_zero_width_reference[2]) &&
    singleton_fp64_window_out.to_host() == std::vector<double>{1.0} &&
    general_gaussian_typed_cpu(1, convert<T>(1), convert<T>(0)) == std::vector<double>{1.0} &&
    empty_fp64_window_out.empty() &&
    general_gaussian_typed_cpu(0, convert<T>(1), convert<T>(0)).empty() &&
    gaussian_device_rejects_negative_length &&
    gaussian_cpu_rejects_negative_length;
```

#### 11.5 仍未覆盖的风险

| 风险 | 说明 |
| --- | --- |
| **大 `power` 值精度** | `power` 较大时 `powf(fabsf(position), 2.0F * power)` 的 FP32 精度损失未测试。测试仅用 `power=1`。 |
| **极大数组** | 测试仅用 4–5 元素，未验证大规模行为的正确性和性能。 |
| **`power < 0` 行为** | 测试未覆盖 `power < 0` 的输入。推测：`pow(fabs(x), negative)` 对 `fabs(x) > 0` 产生 `inf`，`exp(-inf) = 0`；中心点 `pow(0, negative) = inf`，`exp(-inf) = 0`。但此行为未测试。 |
| **`power` 非整数** | 测试仅用 `power=1`（整数）。非整数 `power`（如 `p=1.5`、`p=0.5`）的精度和正确性未测试，但数学公式支持。 |
| **FP16 极端输入** | FP16 输入经 `__half2float` 提升后计算，理论上无溢出，但未测试 FP16 极端输入范围（如 `power` 或 `width` 接近 FP16 上限）。 |
| **整数 `power`/`width` 截断** | `int32_t` 输入超过 $2^{24}$ 时 `static_cast<float>` 丢失精度，未测试。 |
| **并发/多 stream** | 测试使用默认流，未验证多 stream 并发安全性。 |
| **`width` 极接近 0** | 如 `width = 1e-30`，`position` 极大，`powf` 可能溢出为 `inf`，未测试。 |
| **与 Python `gaussian` 算子的等价性** | cusignal_cpp 不提供 `gaussian` 算子，`general_gaussian(p=1)` 是否与 Python `gaussian` 数值一致未在测试中验证（测试未对比 Python，仅对比 C++ CPU reference）。 |

---

### 十二、自检问题

1. `general_gaussian_typed_cpu` 为什么在 `length == 1` 时直接返回 `{1.0}` 而不走公式？——因为 `effective = 1` 时 `x = (0 - 0.5 * 0) / width = 0`，`exp(-0.5 * pow(0, 2p)) = exp(0) = 1`，公式结果也是 1，但显式返回避免 `width == 0` 时的 `0/0 = nan`。
2. GPU kernel 的模板参数顺序 `<typename T, typename Parameter>` 中，`T` 和 `Parameter` 分别是什么？——`T` 是输出类型（host 调度层固定为 `float`），`Parameter` 是输入参数类型（用户 dtype）。
3. 为什么 GPU 用 FP32 计算 + FP64 输出？——ZQ500 平台 FP64 性能远低于 FP32；FP32 计算后经 Host 拓宽为 FP64 存储，是项目统一策略。
4. `width == 0` 时 GPU 和 CPU 的行为是否一致？——一致，均产生 `nan`（中心点 `0/0`）或 `0`（非中心点 `inf` 经 `exp(-inf)`）。测试 `gaussian_zero_width[2]` 验证中心点为 `nan`。
5. cusignal_cpp 为什么不提供 `gaussian` 算子？——`general_gaussian` 的 `p=1` 即等价于 `gaussian`，工程上无需维护两个数学等价的 kernel。Python 中 `gaussian` 用 `n*n/(2*sig2)` 代替 `pow(n, 2)` 是微优化，cusignal_cpp 不做此优化。
6. `effective_count` 在 kernel 中如何使用？——`position = (index - 0.5F * (effective_count - 1)) / width`。`sym=true` 时 `effective_count = length`；`sym=false` 时 `effective_count = length + 1`，但 kernel 只写前 `length` 个点（`output_count = length`）。
7. `finalize_fp64_device_storage` 做了什么？——D2H（`computed.to_host()`）→ Host 端 `float→double` 逐元素拓宽 → H2D（`DeviceArray<double>::from_host`）回写。GPU 端不执行 FP64 算术。
8. 测试为什么用 4e-2 的容差？——GPU FP32 与 CPU FP64 之间的 `powf`/`expf` 精度差异可能达到 1e-2 量级，4e-2 是项目统一的窗函数测试容差（`matches_fp64` 用于所有窗的 FP64 输出比较）。

---

### 十三、建议阅读顺序

按 **接口 → CPU → GPU wrapper → kernel → 测试** 顺序阅读：

1. **头文件** [`windows_typed.h:102-104, 119-121`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_typed.h#L102-L121)：理解 `general_gaussian_typed_cpu` 和 `general_gaussian_device` 的签名与 docstring。
2. **CPU reference** [`windows_typed.cpp:341-358`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_typed.cpp#L341-L358)：`general_gaussian_typed_cpu` 的 FP64 逐点计算，注意长度守卫、单点退化、`effective` 对称/周期语义。
3. **GPU 公开入口** [`windows_typed.cpp:360-375`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_typed.cpp#L360-L375)：`general_gaussian_device` 的长度/尺寸校验、FP32 临时缓冲区、单点退化、`finalize_fp64_device_storage` 拓宽路径。
4. **Host 调度层** [`windows_typed.cu:75`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_typed.cu#L75)：`general_gaussian_fp32_compute_device` 的 `static_assert`、`n <= 1` 守卫、`launch_1d_kernel` 参数传递，注意 `general_gaussian_kernel<float, T>` 的模板实例化。
5. **CUDA kernel** [`windows_kernels.cuh:84-107`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_kernels.cuh#L84-L107)：`general_gaussian_kernel` 的逐线程逻辑，注意 `window_load`/`window_store`、`position` 计算和 `pow`/`exp` 的 FP32 重载。
6. **辅助工具**：[`simple_signal_typed.h`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/cuda_utils/simple_signal_typed.h)（`SimpleSignalTypePolicy` 的 `load`/`store` 行为）、[`host_output_finalize.h`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h)（`finalize_fp64_device_storage` 的 D2H→拓宽→H2D 路径）、[`kernel_launch.h`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h)（`launch_1d_kernel` 的 block=256 配置）。
7. **显式实例化** [`windows_typed.cpp:562-593`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_typed.cpp#L562-L593) 和 [`windows_typed.cu:116-127`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/src/windows/windows_typed.cu#L116-L127)：理解 5 种 dtype 的显式实例化。
8. **测试** [`e3_wave_window_type_smoke.cu:524-577`](file:///c:/Users/Lenovo/Desktop/ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu#L524-L577)：验证 GPU vs CPU 一致性、width=0 nan 行为、单点/空窗/负长度边界。

每段代码应关注的问题：

- CPU `general_gaussian_typed_cpu` 的 `length == 1` 显式返回 `{1.0}` 与走公式结果是否一致（当 `width == 0` 时不同）。
- GPU kernel 的模板参数顺序 `<typename T, typename Parameter>` 与 host 实例化 `general_gaussian_kernel<float, T>` 的对应关系。
- `finalize_fp64_device_storage` 的 D2H → Host 拓宽 → H2D 路径是否有性能优化空间。
- `width == 0` 时 GPU FP32 和 CPU FP64 的 `nan` 行为是否完全一致（`powf(nan, ...)` vs `std::pow(nan, ...)`）。
- cusignal_cpp 不提供 `gaussian` 算子是否会影响与 cuSignal Python 的兼容性。

---

## 版本差异与原理不变量

（仅 V1 版本，暂无优化版本可供对比。以下记录当前 V1 的算法特征，供后续版本对比。）

| 特征 | V1 |
| --- | --- |
| 算法结构 | 逐元素解析公式，`exp(-0.5 * pow(fabs(position), 2*power))` |
| CPU 计算精度 | FP64（`double`） |
| GPU 计算精度 | FP32（kernel）→ Host 拓宽 → FP64（存储） |
| 并行方式 | 1D grid × 256 线程/block，无线程间同步 |
| 内存访问 | 合并访问（coalesced），输入标量按值传递，输出连续一维数组 |
| NaN 处理 | 不检查 `power`/`width`，按 IEEE 754 传播（`width==0` 产生 `nan`） |
| 归一化 | 无（峰值自然为 1） |
| FFT/平台库 | 无（纯 elementwise CUDA 计算） |
| `loop_prep` | 无（常量在 kernel 内逐线程计算） |
| `gaussian` 算子 | 不提供（`general_gaussian(p=1)` 替代） |
| 显式实例化 | 5 种 dtype：`float`、`__half`、`int32_t`、`int16_t`、`int8_t` |

**原理不变量**：无论版本如何优化，以下数学物理原理始终不变：

1. **原理 1（广义正态分布）**：窗函数 PDF 形式 $\exp(-0.5 |x/\sigma|^{2p})$ 不变。
2. **原理 2（高斯窗时域定义）**：$p=1$ 退化为标准高斯窗 $e^{-(x/\sigma)^2/2}$ 不变。
3. **原理 3（半功率点）**：$n_{\text{half}} = (2\ln 2)^{1/(2p)} \sigma$ 是公式推导结论，不在生成代码中计算，不变。
4. **原理 4（超高斯平顶）**：$p>1$ 时窗顶部更平坦，由公式参数化覆盖，不变。
5. **原理 5（拉普拉斯重尾）**：$p=0.5$ 时尾部更重，由公式参数化覆盖，不变。
6. **原理 6（加窗原理）**：时域乘法对应频域卷积，是窗函数的使用方式，不是生成逻辑，不变。
7. **对称/周期语义**：`sym=true` 用 $M$ 点，`sym=false` 用 $M+1$ 点内部长度再截前 $M$ 点，不变。
8. **单点退化**：$M=1 \Rightarrow w[0]=1$，不变。
9. **不做归一化**：峰值自然为 1，不变。
10. **不检查 `power`/`width` 合法性**：与 cuSignal Python 一致，不变。
