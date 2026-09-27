# cusignal_cpp_hamming 复现逻辑

> 阶段三：`cusignal_cpp` CPU/GPU 复现逻辑。本文件只覆盖 `hamming` 算子，与
> `数学物理原理.md`（阶段一）和 `python源码算法.md`（阶段二）配套。代码定位
> 格式为「完整提交 SHA + `ZKX_dev/` 相对路径 + 符号名 + 该提交中的准确行号」。

## 版本索引

| 版本 | 短 SHA | 提交时间 | 分支 | 状态 |
| --- | --- | --- | --- | --- |
| V1 | `1ccd32ea` | 2026-07-27 22:12:16 +0800 | `snapshot/all-current-20260716` | 当前实现 |
| V2 | — | — | — | 占位（尚未发生优化） |
| V3 | — | — | — | 占位（尚未发生优化） |

---

## V1：原始学习版本（1ccd32ea）

### 一、Git 版本信息

| 项目 | 内容 |
| --- | --- |
| 完整 Git SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 短 SHA | `1ccd32ea` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支 | `snapshot/all-current-20260716` |
| 相关文件 dirty 状态 | 全部干净（`cusignal_cpp` 目录无修改） |

相关文件（均在 `ZKX_dev/cusignal_cpp/` 下）：

- `src/windows/windows_typed.h`
- `src/windows/windows_typed.cpp`
- `src/windows/windows_typed.cu`
- `src/windows/windows_kernels.cuh`
- `test/signal_processing/e3_wave_window_type_smoke.cu`

辅助文件（用于理解调用，非 `hamming` 独占）：

- `src/cuda_utils/kernel_launch.h`（`launch_1d_kernel` 实现）
- `src/cuda_utils/host_output_finalize.h`（`finalize_fp64_device_storage` 实现）
- `src/cuda_utils/simple_signal_typed.h`（`is_simple_signal_input_v`、`SimpleSignalTypePolicy<T>` 定义）

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注"无直接原理对应"。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在"含义解释"列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §一 `windows_typed.h:133-134`（`hamming_typed_cpu` 模板参数） | `template <class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，仅用于显式实例化与 `static_assert`；5 种类型：`float`/`__half`/`std::int32_t`/`std::int16_t`/`std::int8_t`；不参与计算，CPU 输出固定 `double`，GPU kernel 固定 `<float>` |
| `length` | §二 `windows_typed.h:133-134`（`hamming_typed_cpu` 首参数，记作 `length`） | `int n`（CPU 入口记作 `length`） | 原理 1/2 §"通用定义"中的 $M$ | 窗长度（点数），$M \geq 0$；CPU 路径专用名；GPU 侧以 `n` 形式出现于 `hamming_device`/`hamming_resident_device`/`hamming_fp32_compute_device` |
| `symmetric` | §二 `windows_typed.h:133-134`（`hamming_typed_cpu` 第二参数，记作 `symmetric`） | `bool sym = true` | 原理 6 §"离散形式与周期窗" | 工程旗标，`true` 对称窗（`effective = length`，用于 FIR 设计），`false` 周期窗（`length` 偶数时 `effective = length + 1` 再截断，用于频谱分析）；GPU 侧以 `sym` 形式出现 |
| `out` | §二 `windows_typed.h:133-134`（`hamming_typed_cpu` 返回类型） | `std::vector<double>` 返回值（CPU）/`DeviceArray<double>&` 主接口输出 /`DeviceArray<float>&` resident 接口输出 | 原理 1/2 §"通用定义"中的 $w[n]$ | 输出窗序列；CPU 全程 FP64；GPU 主接口由 `finalize_fp64_device_storage` 拓宽为 FP64；resident 接口直接交付 FP32（cusignal_cpp 独有） |
| `effective` | §三 `windows_typed.cpp:378`（`hamming_typed_cpu` 局部） | `int`，`(!symmetric && length % 2 == 0) ? length + 1 : length` | 原理 6 §"离散形式与周期窗"（周期窗扩展） | 工程变量，对称中心化长度；对称窗等于 `length`，周期窗偶数长度扩展为 `length + 1`；GPU 侧以 `effective_count` 形式出现于 `windows_typed.cu:76` 第 5 个 kernel 参数 |
| `pi64` | §三 `windows_typed.cpp:380`（`hamming_typed_cpu` 局部常量） | `constexpr double pi64 = 3.141592653589793238462643383279502884` | 数学常数 $\pi$（FP64 精度） | 圆周率 FP64 字面量，CPU 路径专用；与 `general_cosine_typed_cpu`、`kaiser_typed_cpu`、`taylor_typed_cpu` 共用同一字面量 |
| `index` | §三 `windows_typed.cpp:382`（`hamming_typed_cpu` 循环变量） | `int`，`for (int index = 0; index < length; ++index)` | 原理 1/2 §"通用定义"中的 $n$（$0 \leq n \leq M-1$） | 时域采样索引；GPU kernel 侧以 `index = blockIdx.x * blockDim.x + threadIdx.x` 形式出现于 `windows_kernels.cuh:111` |
| `expected` | §四 `windows_typed.cpp:395`（`hamming_device` 局部） | `std::size_t`，`length > 0 ? static_cast<std::size_t>(length) : 0` | 无直接原理对应 | 工程变量，主接口期望的输出长度；负长度映射为 0 实现短路；用于 `out.size()` 校验与 `computed` 缓冲分配；resident 接口无此变量 |
| `computed` | §四 `windows_typed.cpp:398`（`hamming_device` 局部） | `DeviceArray<float> computed(expected)` | 无直接原理对应 | 工程 FP32 临时缓冲区，承载 GPU kernel 的 FP32 计算结果，随后经 `finalize_fp64_device_storage` 拓宽为 FP64；resident 接口无此缓冲（直接写调用者 `out`） |
| `n` | §二 `windows_typed.h:146-147`（`hamming_device` 首参数） | `int n` | 原理 1/2 §"通用定义"中的 $M$ | 窗长度（点数），GPU 路径的统一名称；在 `hamming_fp32_compute_device` 内作为 size 校验依据（`n < 1` 抛异常） |
| `effective_count` | §四 `windows_typed.cu:76`（`hamming_fp32_compute_device` 第 5 个 kernel 参数） | `int`，`(!sym && n % 2 == 0) ? n + 1 : n` | 原理 6 §"离散形式与周期窗"（周期窗扩展） | 工程变量，GPU 侧的对称中心化长度，与 CPU `effective` 同义；作为 kernel 参数传入 `hamming_kernel`，控制公式分母 `effective_count - 1` |
| `output_count` | §四 `windows_kernels.cuh:109-110`（`hamming_kernel` 首参数） | `int output_count` | 无直接原理对应 | 工程变量，kernel 写入的输出元素数，等于 `n`（用户请求长度）；`sym=false` 且 `n` 为偶数时小于 `effective_count`，通过线程边界检查 `if (index >= output_count) return` 完成截断 |
| `output` | §四 `windows_kernels.cuh:109-110`（`hamming_kernel` 第二参数） | `T* output` | 原理 1/2 §"通用定义"中的 $w[n]$ | kernel 输出指针，对应 `computed.data()`（主接口）或 `out.data()`（resident 接口）；最终经 `finalize_fp64_device_storage` 拓宽为 FP64 |
| `Scalar` | §四 `windows_kernels.cuh:112`（`hamming_kernel` 局部 `using` 声明） | `using Scalar = decltype(window_load(output[index]))`，固定 FP32 | 无直接原理对应 | 工程 kernel 计算精度类型，固定 `float`；用于 `pi`、`value` 等局部变量类型推导；显式实例化只产生 `hamming_kernel<float>` 一份 |
| `value` | §四 `windows_kernels.cuh:113`（`hamming_kernel` 局部） | `Scalar value = static_cast<Scalar>(1)` | 原理 1/2 §"通用定义"中的 $w[n]$ | 工程变量，kernel 内计算的窗值；默认 1.0 用于 `effective_count == 1` 单点窗短路；否则按 `0.54F - 0.46F * cos(2*pi*index/(effective_count-1))` 计算 |
| `pi` | §四 `windows_kernels.cuh:115`（`hamming_kernel` 局部常量） | `const Scalar pi = static_cast<Scalar>(3.14159265358979323846F)` | 数学常数 $\pi$（FP32 精度） | 圆周率 FP32 字面量，kernel 全程使用；与 CPU 的 `pi64` 形成精度差异（约 7 位有效数字 vs 30 位），是 GPU 路径的精度差异来源之一 |
| `window_load<T>` | §五 `windows_kernels.cuh:10-18`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::load`，5 种 dtype 统一转 `float`；在 `hamming_kernel` 内用于推导 `Scalar` 类型 |
| `window_store<T, Scalar>` | §五 `windows_kernels.cuh:20-28`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::store`，把 FP32 `value` 转回 `T`；实际只实例化 `T=float`，等价于 `static_cast<float>` |
| `SimpleSignalTypePolicy<T>` | §五 `cuda_utils/simple_signal_typed.h`（类型策略模板） | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断）；`hamming` 路径实际只用 `T=float` 特化 |
| `launch_1d_kernel` | §四 `cuda_utils/kernel_launch.h:44-56`（kernel 启动工具） | 函数模板，参数 `(Kernel kernel, std::size_t elements, Args&&... args)` | 无直接原理对应 | 工程 kernel 启动工具，自动按 `elements` 计算 grid/block（默认 `block_size=256`，默认 stream），后跟 `CUDA_KERNEL_CHECK()` 错误检查 |
| `block_size` | §六 `cuda_utils/kernel_launch.h:44-56`（`launch_1d_kernel` 内部常量） | `constexpr int block_size = 256` | 无直接原理对应 | 工程变量，CUDA 线程块大小；`hamming_kernel` 每 block 256 线程，grid.x = ceil(elements/256) |
| `finalize_fp64_device_storage` | §四 `cuda_utils/host_output_finalize.h`（FP32→FP64 拓宽工具） | 函数，参数 `(DeviceArray<float> computed)` 返回 `DeviceArray<double>` | 无直接原理对应 | 工程 FP32→FP64 拓宽工具，内部走 D2H → host float→double → H2D；仅主接口调用，resident 接口跳过 |
| `is_simple_signal_input_v<T>` | §五 `cuda_utils/simple_signal_typed.h`（类型 trait） | `template <class T> constexpr bool is_simple_signal_input_v` | 无直接原理对应 | 工程类型 trait，编译期判断 `T` 是否属于 5 种简单信号类型，用于 `static_assert`；在 host wrapper 与 resident 接口触发 |
| `kPi` | §二 `windows_typed.cpp:15`（文件级常量） | `constexpr float kPi = 3.14159265358979323846F` | 数学常数 $\pi$（FP32 精度） | 工程文件级 FP32 圆周率常量；`hamming_kernel` 内 `pi` 字面量与之同值；`hamming` 主路径未直接引用 `kPi`（kernel 用局部 `pi`），列此供其他 windows 算子共用对照 |
| `INSTANTIATE_WINDOWS_CPU` | §五 `windows_typed.cpp:562-595`（CPU 侧实例化宏） | 宏，展开为 9 个 windows 算子 CPU/GPU 包装的 `template ... ;` 声明 | 无直接原理对应 | 工程 CPU 侧显式实例化宏，一次展开 5 种 `T`；`hamming` 相关三行位于 `windows_typed.cpp:576-578` |
| `INSTANTIATE` | §五 `windows_typed.cu:116-126`（GPU 侧实例化宏） | 宏，展开为 9 个 `*_fp32_compute_device<T>` 的 `template void ... ;` 声明 | 无直接原理对应 | 工程 GPU 侧显式实例化宏，一次展开 5 种 `T`；`hamming` 相关一行位于 `windows_typed.cu:121` |
| `matches_fp64` | §十一 `e3_wave_window_type_smoke.cu:600-604`（测试容差比较函数） | 测试工具函数，参数 `(actual, reference)`，容差 `4e-2` | 无直接原理对应 | 工程测试工具，FP32 GPU 输出与 FP64 CPU reference 的容差比较；容差 4e-2 用于吸收 FP32 舍入误差与 $\pi$ 字面量精度差异 |

---

### 二、对外接口与代码定位

`hamming` 在 `cusignal_cpp` 中共有四个层次的代码，分别对应阶段三要求的
「API 包装层 / CPU reference 核心 / GPU host 调度层 / GPU kernel 核心」。

#### 2.1 API 包装层（主，FP64 输出契约）

| 角色 | 路径 | 符号 | 行号 |
| --- | --- | --- | --- |
| 声明 | `cusignal_cpp/src/windows/windows_typed.h` | `template <class T> void hamming_device(int n, DeviceArray<double>& out, bool sym = true)` | 146–147 |
| 实现 | `cusignal_cpp/src/windows/windows_typed.cpp` | `hamming_device<T>` | 393–406 |

签名（`windows_typed.h:146-147`）：

```cpp
template <class T>
void hamming_device(int n, DeviceArray<double>& out, bool sym = true);
```

语义：在 GPU 上生成长度为 `max(n, 0)` 的 Hamming 窗，输出由 Host 拓宽后以 FP64
形式写回 `out`。`T` 仅作为业务请求类型存在（用于显式实例化分发），不参与计算。

#### 2.2 API 包装层（resident，FP32 输出契约，cusignal_cpp 独有）

| 角色 | 路径 | 符号 | 行号 |
| --- | --- | --- | --- |
| 声明 | `cusignal_cpp/src/windows/windows_typed.h` | `template <class T> void hamming_resident_device(int n, DeviceArray<float>& out, bool sym = true)` | 154–155 |
| 实现 | `cusignal_cpp/src/windows/windows_typed.cpp` | `hamming_resident_device<T>` | 408–415 |

签名（`windows_typed.h:154-155`）：

```cpp
template <class T>
void hamming_resident_device(int n, DeviceArray<float>& out, bool sym = true);
```

语义：与主接口同公式，但 `out` 由调用者预分配为 `DeviceArray<float>`，**不经过
`finalize_fp64_device_storage` 拓宽**，直接交付 FP32 给下游 FP32 算子。这是
cusignal_cpp 相对 cuSignal Python 的独有工程接口，目的是避免 FP64 拓宽开销。

#### 2.3 CPU reference 核心

| 角色 | 路径 | 符号 | 行号 |
| --- | --- | --- | --- |
| 声明 | `cusignal_cpp/src/windows/windows_typed.h` | `template <class T> std::vector<double> hamming_typed_cpu(int n, bool sym = true)` | 133–134 |
| 实现 | `cusignal_cpp/src/windows/windows_typed.cpp` | `hamming_typed_cpu<T>` | 377–391 |

签名（`windows_typed.h:133-134`）：

```cpp
template <class T>
std::vector<double> hamming_typed_cpu(int n, bool sym = true);
```

语义：在 Host 上以 FP64 余弦生成 Hamming 窗；`n < 1` 时返回空 `std::vector<double>`。
`T` 只用于显式实例化（不读取 `T` 参数），数学全程 FP64。

#### 2.4 GPU host 调度层

| 角色 | 路径 | 符号 | 行号 |
| --- | --- | --- | --- |
| 前置声明 | `cusignal_cpp/src/windows/windows_typed.cpp` | `template <class T> void hamming_fp32_compute_device(int, DeviceArray<float>&, bool)` | 141 |
| 实现 | `cusignal_cpp/src/windows/windows_typed.cu` | `hamming_fp32_compute_device<T>` | 76 |

行号说明：`windows_typed.cpp:141` 是同文件顶部对所有 `*_fp32_compute_device` 模板的
前置声明，统一汇总；真正实现在 `windows_typed.cu:76`（单行紧凑写法）。

#### 2.5 GPU device kernel 核心

| 角色 | 路径 | 符号 | 行号 |
| --- | --- | --- | --- |
| 定义 | `cusignal_cpp/src/windows/windows_kernels.cuh` | `windows_detail::hamming_kernel<T>` | 109–123 |

签名（`windows_kernels.cuh:109-110`）：

```cpp
template <typename T>
__global__ void hamming_kernel(int output_count, T* output, int effective_count)
```

#### 2.6 测试驱动

| 角色 | 路径 | 符号 | 行号 |
| --- | --- | --- | --- |
| 测试入口 | `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | `run_type<T>` 内 `ok[9]` | 578–604 |

`e3_wave_window_type_smoke.cu:578` 处先调用
`test_evidence::f_begin_accuracy(type_dispatch::OperatorId::hamming)` 起测，
随后执行覆盖 5 种 dtype 的 5 类边界用例，最终在 `ok[9]` 行（第 600–604 行）
汇总判定。

---

### 三、CPU 调用链、主要数据结构和逐步算法

#### 3.1 实现原文（`windows_typed.cpp:377-391`）

```cpp
template <class T>
std::vector<double> hamming_typed_cpu(int length, bool symmetric)
{
    if (length < 1) return {};
    const int effective = (!symmetric && length % 2 == 0) ? length + 1 : length;
    std::vector<double> out(length);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    for (int index = 0; index < length; ++index) {
        out[index] = effective == 1
            ? 1.0
            : 0.54 - 0.46
                * std::cos(2.0 * pi64 * index / (effective - 1));
    }
    return out;
}
```

#### 3.2 主要数据结构

- `int length`：调用者请求的窗长度，可为负数。
- `bool symmetric`：`true` 对称窗（默认），`false` 周期窗。
- `int effective`：实际参与公式分母的长度。
- `std::vector<double> out`：长度为 `max(length, 0)` 的输出，dtype 固定 FP64。
- `constexpr double pi64`：FP64 精度的 $\pi$，与 `general_cosine_typed_cpu`、
  `kaiser_typed_cpu`、`taylor_typed_cpu` 同一常量字面量。
- 模板参数 `T`：仅用于显式实例化与 `static_assert`（`hamming_typed_cpu` 本体未
  调用 `static_assert`，但 `INSTANTIATE_WINDOWS_CPU` 宏已限定 5 种类型）。

#### 3.3 逐步算法

1. **长度守卫**：`if (length < 1) return {}`。注意判据是 `< 1`（即 `length <= 0`），
   不是 `length < 0`。`length == 0` 和 `length < 0` 都返回空向量，不抛异常。
   这与 `parzen`、`taylor`、`general_cosine` 的 `if (length < 0) throw; if (length == 0) return {};`
   路径不同，更接近 cuSignal Python 的 `if M < 1: return cp.array([])` 行为。
2. **计算 effective**：`effective = (!symmetric && length % 2 == 0) ? length + 1 : length`。
   - 对称窗（`symmetric=true`）：`effective = length`。
   - 周期窗且 `length` 为偶数：`effective = length + 1`（生成 `length+1` 点再截断）。
   - 周期窗且 `length` 为奇数：`effective = length`（奇数长度周期窗无需扩展）。
3. **分配输出**：`std::vector<double> out(length)`，长度等于请求长度，不是 effective。
4. **逐点计算**：`for (index = 0..length-1)`，每个点套公式：
   - 若 `effective == 1`（即 `length == 1` 且未扩展）：`out[index] = 1.0`（短路到中心值）。
   - 否则：$out[index] = 0.54 - 0.46 \cdot \cos(2\pi \cdot index / (effective - 1))$。
5. **截断语义**：CPU 不显式截断，但循环只写到 `length-1`，而公式分母用 `effective-1`。
   当 `symmetric=false` 且 `length` 为偶数时，effective=length+1，循环写入前 `length` 个点，
   末点（索引 `length`）天然被丢弃，等价于 cuSignal Python 的 `w = w[:-1]`。

#### 3.4 CPU 调用链

```
hamming_typed_cpu<T>(length, symmetric)              [windows_typed.cpp:377-391]
  ├─ length < 1            → return {}
  ├─ effective = (!sym && length%2==0) ? length+1 : length
  ├─ std::vector<double> out(length)
  └─ for index = 0..length-1:
       out[index] = (effective == 1)
         ? 1.0
         : 0.54 - 0.46 * cos(2*pi64 * index / (effective-1))
```

CPU 全程使用 **FP64**，无 GPU stream、无 workspace、无 FFT、无平台库调用。

---

### 四、GPU 调用链、host wrapper、kernel 启动与 device 计算逻辑

#### 4.1 主接口 `hamming_device`（FP64 输出）

实现原文（`windows_typed.cpp:393-406`）：

```cpp
template <class T>
void hamming_device(int length, DeviceArray<double>& out, bool symmetric)
{
    const std::size_t expected = length > 0
        ? static_cast<std::size_t>(length)
        : 0;
    if (out.size() != expected)
        throw std::invalid_argument("hamming_device: output size must equal max(length, 0)");
    DeviceArray<float> computed(expected);
    if (expected != 0) {
        hamming_fp32_compute_device<T>(length, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

调用链：

```
hamming_device<T>(length, out, symmetric)            [windows_typed.cpp:393-406]
  ├─ expected = length > 0 ? size_t(length) : 0
  ├─ if out.size() != expected → throw std::invalid_argument("hamming_device: ...")
  ├─ DeviceArray<float> computed(expected)           // FP32 临时缓冲，expected 可为 0
  ├─ if expected != 0:
  │     hamming_fp32_compute_device<T>(length, computed, symmetric)
  └─ out = cuda_utils::finalize_fp64_device_storage(computed)
            ├─ computed.to_host()                    // D2H: GPU FP32 → Host vector<float>
            ├─ finalize_fp64_on_host(...)            // Host: float → double 逐元素拓宽
            └─ DeviceArray<double>::from_host(...)   // H2D: Host vector<double> → GPU FP64
```

要点：

- `length < 0` 不直接抛，而是令 `expected = 0`，再由 `out.size() != expected` 触发异常。
  这要求调用者对负长度传入大小为 0 的 `out`（测试 `hamming_device<T>(-1, empty_fp64_window_out)`
  正是这样做的，因此不抛）。
- `length == 0`：`expected == 0`，跳过 kernel 启动，`finalize_fp64_device_storage` 在空数组
  上返回空 `DeviceArray<double>`。
- 真正的 GPU 算术只发生在 `hamming_fp32_compute_device` 内部，**GPU 端不执行任何 FP64 算术**；
  FP64 仅作为接口/存储类型存在，由 Host 拓宽完成。

#### 4.2 Host wrapper `hamming_fp32_compute_device`（GPU host 调度层）

实现原文（`windows_typed.cu:76`，单行紧凑写法，下面按逻辑展开）：

```cpp
template <class T>
void hamming_fp32_compute_device(int n, DeviceArray<float>& o, bool sym)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (n < 1 || o.size() != size_t(n))
        throw std::invalid_argument("hamming_device: compute output size must equal positive n");
    cuda_utils::launch_1d_kernel(
        windows_detail::hamming_kernel<float>,
        o.size(),
        n,
        o.data(),
        (!sym && n % 2 == 0) ? n + 1 : n);
}
```

要点：

- `static_assert(is_simple_signal_input_v<T>)`：编译期拒绝 5 种之外的类型。
  注意该断言放在 host wrapper 而非 CPU reference；CPU reference 依赖宏实例化保证。
- `n < 1` 在此处是直接抛异常的判据（与 API 包装层把负长度映射成 `expected=0` 不同）。
  进入 host wrapper 的前提是 `expected != 0`，故实际触发仅在 `n>=1`。
- 第 5 个 kernel 参数 `effective_count = (!sym && n % 2 == 0) ? n + 1 : n`，与 CPU 的
  `effective` 计算完全一致。计算结果由 kernel 内部根据 `effective_count` 决定每个线程
  对应的相位。
- `launch_1d_kernel` 的第 2 参数 `o.size()` 是元素数，决定 grid/block 配置（见第 6 节）。

#### 4.3 Device kernel `hamming_kernel`（GPU kernel 核心）

实现原文（`windows_kernels.cuh:109-123`）：

```cpp
template <typename T>
__global__ void hamming_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    using Scalar = decltype(window_load(output[index]));
    Scalar value = static_cast<Scalar>(1);
    if (effective_count != 1) {
        const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
        value = static_cast<Scalar>(0.54F) - static_cast<Scalar>(0.46F) *
            cos(static_cast<Scalar>(2) * pi * static_cast<Scalar>(index) /
                static_cast<Scalar>(effective_count - 1));
    }
    output[index] = window_store<T>(value);
}
```

逐句解读：

1. `index = blockIdx.x * blockDim.x + threadIdx.x`：1D 线性线程索引。
2. `if (index >= output_count) return`：边界检查，超出输出长度的线程立即返回。
3. `Scalar value = 1`：默认值用于 `effective_count == 1` 的单点窗。
4. `if (effective_count != 1)`：单点短路，与 CPU 的 `effective == 1 ? 1.0 : ...` 等价。
5. `pi = 3.14159265358979323846F`：FP32 精度的 $\pi$（与 host wrapper 中
   `windows_typed.cpp:15` 处 `kPi` 字面量一致）。
6. `value = 0.54F - 0.46F * cos(2 * pi * index / (effective_count - 1))`：
   标准 Hamming 公式，全部以 FP32 计算。
7. `output[index] = window_store<T>(value)`：通过 `window_store` 把 FP32 `value` 转回
   `T` 类型存储。对于 `T = float`，`window_store<float>` 即 `static_cast<float>`；
   对其他 4 种类型（`__half`/`int32_t`/`int16_t`/`int8_t`）则按
   `SimpleSignalTypePolicy<T>::store` 进行饱和/截断转换。
   但由于显式实例化只把 `hamming_kernel<float>` 暴露给 host wrapper，实际下游始终使用
   `T = float`（参见第 5 节）。

#### 4.4 Resident 接口 `hamming_resident_device`（FP32 直出）

实现原文（`windows_typed.cpp:408-415`）：

```cpp
template <class T>
void hamming_resident_device(
    int length, DeviceArray<float>& out, bool symmetric)
{
    static_assert(detail::is_simple_signal_input_v<T>,
        "unsupported hamming dtype");
    hamming_fp32_compute_device<T>(length, out, symmetric);
}
```

要点：

- `out` 是 `DeviceArray<float>&`，**由调用者预分配**，函数内部不再分配 `computed`。
- 直接调用 `hamming_fp32_compute_device<T>(length, out, symmetric)`，不再经过
  `finalize_fp64_device_storage` 拓宽。
- `static_assert(is_simple_signal_input_v<T>)` 在此接口显式触发；与 host wrapper 的
  `static_assert` 形成双保险。
- 该接口不改变 `hamming_device` 的 FP64 公开契约；调用者必须显式选择 resident
  路径，承担「下游算子接受 FP32」的责任。
- 错误处理完全继承 `hamming_fp32_compute_device`：`n < 1` 或 `o.size() != n` 抛
  `std::invalid_argument`。

调用链：

```
hamming_resident_device<T>(length, out, symmetric)   [windows_typed.cpp:408-415]
  ├─ static_assert(is_simple_signal_input_v<T>)
  └─ hamming_fp32_compute_device<T>(length, out, symmetric)   [windows_typed.cu:76]
        └─ launch_1d_kernel(hamming_kernel<float>, out.size(), length, out.data(), effective_count)
              └─ GPU kernel: 逐线程 FP32 计算，直接写入 out（不再 D2H/H2D）
```

---

### 五、template、类型分发、显式实例化、内存布局与数据搬运

#### 5.1 模板参数

四个层次的模板参数均为 `<class T>`，但 `T` 在不同层次承担不同角色：

| 层次 | 模板 | T 的实际作用 |
| --- | --- | --- |
| `hamming_typed_cpu<T>` | `<class T>` | 仅用于显式实例化（5 种类型），不参与计算（无 `T` 参数） |
| `hamming_device<T>` | `<class T>` | 同上；T 仅决定调用 `hamming_fp32_compute_device<T>` |
| `hamming_resident_device<T>` | `<class T>` | 同上；额外触发 `static_assert` |
| `hamming_fp32_compute_device<T>` | `<class T>` | `static_assert` 限制 T；调用 `hamming_kernel<float>`（T 不传到 kernel） |
| `hamming_kernel<T>` | `<typename T>` | kernel 实际存储类型；但 host wrapper 永远实例化 `hamming_kernel<float>` |

支持的 5 种 `T`：

- `float`
- `__half`
- `std::int32_t`
- `std::int16_t`
- `std::int8_t`

判定逻辑见 `cuda_utils/simple_signal_typed.h` 中的 `detail::is_simple_signal_input_v<T>`
和 `detail::SimpleSignalTypePolicy<T>`（本文件不展开其实现）。

#### 5.2 显式实例化

CPU 侧（`windows_typed.cpp:562-595`）通过宏 `INSTANTIATE_WINDOWS_CPU(T)` 一次性实例化
9 个 windows 算子的 CPU/GPU 包装。`hamming` 相关三行（`windows_typed.cpp:576-578`）：

```cpp
template std::vector<double> hamming_typed_cpu<T>(int, bool);
template void hamming_device<T>(int, DeviceArray<double>&, bool);
template void hamming_resident_device<T>(int, DeviceArray<float>&, bool);
```

注意 `hamming_typed_cpu<T>`、`hamming_device<T>`、`hamming_resident_device<T>` 显式写出了
`<T>`，而其他 windows 算子（如 `chebwin_typed_cpu`、`general_cosine_device`）未写 `<T>`。
两者在语法上等价；这里多写 `<T>` 是历史风格。

宏展开 5 次（`windows_typed.cpp:589-593`）：

```cpp
INSTANTIATE_WINDOWS_CPU(float);
INSTANTIATE_WINDOWS_CPU(__half);
INSTANTIATE_WINDOWS_CPU(std::int32_t);
INSTANTIATE_WINDOWS_CPU(std::int16_t);
INSTANTIATE_WINDOWS_CPU(std::int8_t);
```

GPU 侧（`windows_typed.cu:116-126`）通过宏 `INSTANTIATE(T)` 实例化 9 个
`*_fp32_compute_device` 模板。`hamming` 相关一行（`windows_typed.cu:121`）：

```cpp
template void hamming_fp32_compute_device<T>(int, DeviceArray<float>&, bool);
```

宏展开 5 次（`windows_typed.cu:126`）：

```cpp
INSTANTIATE(float); INSTANTIATE(__half); INSTANTIATE(std::int32_t); INSTANTIATE(std::int16_t); INSTANTIATE(std::int8_t);
```

注意 host wrapper 显式调用 `windows_detail::hamming_kernel<float>`（`windows_typed.cu:76`），
即 kernel 实际只用 `T=float`。因此无论上层 `T` 是 `__half` 还是 `int8_t`，GPU
计算都在 FP32 中进行；输出 dtype 解耦由 `window_store<T>` 在 kernel 末尾完成，
但实例化上只产生 `hamming_kernel<float>` 一份（其他 4 种 `T` 的 kernel
在 `windows_kernels.cuh` 中是模板，未被显式实例化，也不参与 `hamming` 调用）。

#### 5.3 内存布局

| 数据 | 位置 | dtype | 大小 | 谁分配 |
| --- | --- | --- | --- | --- |
| `out`（API 主接口） | Device | `double`（FP64） | `expected` | 调用者预分配 |
| `out`（resident 接口） | Device | `float`（FP32） | `expected` | 调用者预分配 |
| `computed`（API 主接口内部临时） | Device | `float`（FP32） | `expected` | `hamming_device` 内部分配 |
| kernel 写入缓冲 | Device | `float` | `expected` | host wrapper 透传 `o.data()` |
| `finalize_fp64_on_host` 中间 | Host | `double` | `expected` | `finalize_fp64_device_storage` 内部分配 |

`expected` 仅在主接口存在；resident 接口不分配临时缓冲，直接写入调用者的 `out`。

#### 5.4 数据搬运路径

主接口：

```
Host length,sym ──► host wrapper ──► Device computed(FP32) ──► kernel 写入
                       │
                       └─► (kernel 完成后) computed.to_host() ──► Host vector<double>(FP64) ──► from_host ──► Device out(FP64)
```

resident 接口：

```
Host length,sym ──► host wrapper ──► Device out(FP32) ──► kernel 直接写入（无 D2H/H2D）
```

resident 路径在 kernel 完成后即返回，无 host 中转、无拓宽；这是其相对主接口的性能优势。

---

### 六、grid、block、线程索引与每线程数据范围

`launch_1d_kernel`（`cusignal_cpp/src/cuda_utils/kernel_launch.h:44-56`）固定使用：

- `block_size = 256`
- `grid = dim3(div_up(elements, 256))`，即 `grid.x = ceil(elements / 256)`
- `stream = nullptr`（默认 stream）
- shared memory = 0
- kernel 后跟 `CUDA_KERNEL_CHECK()`（`kernel_launch.h:41`）

对 `hamming`：

- `elements = o.size() = max(length, 0)`。
- 每个 thread 负责**一个**输出元素 `out[index]`，线程索引 `index = blockIdx.x * blockDim.x + threadIdx.x`。
- 边界检查 `if (index >= output_count) return`，无越界写。
- 无 warp divergence 优化；`effective_count != 1` 分支在所有线程上几乎一致（除非
  `effective_count == 1`，此时所有线程走短路分支，无分歧）。
- 无 shared memory、无 atomics、无 reduction。

举例：

- `length = 5`：`elements = 5`，`grid.x = 1`，`block.x = 256`，5 个线程写入，251 个立即 return。
- `length = 4, sym=false`：`effective_count = 5`，kernel 仍以 4 个线程写入，每个线程的
  公式分母用 `effective_count - 1 = 4`，等价于生成 5 点 Hamming 后丢弃末点。
- `length = 1`：`effective_count = 1`，所有线程走 `value = 1` 短路。
- `length = 0` 或 `length < 0`：主接口 `expected = 0`，**不启动 kernel**（`if (expected != 0)`）；
  resident 接口则会因 `n < 1` 抛异常，要求调用者自行短路。

---

### 七、同步、临时缓冲区、FFT/平台库调用与错误处理

#### 7.1 同步

- 所有调用使用**默认 stream**（`stream = nullptr`），无显式 `cudaStreamSynchronize`。
- 主接口的 `finalize_fp64_device_storage` 内部含 D2H 拷贝（`computed.to_host()`），
  隐式同步该 stream；之后 H2D（`DeviceArray<double>::from_host`）也走默认 stream。
- resident 接口既无 D2H 也无 H2D，返回后 kernel 是否完成取决于后续同步点；调用者
  必须在消费 `out` 之前自行同步（与下游算子使用同一默认 stream 即可顺序依赖）。

#### 7.2 临时缓冲区

- 主接口：`DeviceArray<float> computed(expected)`，函数内分配，函数结束自动释放。
  无跨调用复用、无 memory pool。
- resident 接口：无内部临时缓冲区，直接写调用者的 `out`。
- 无 host 端临时缓冲（除 `finalize_fp64_device_storage` 内部的 `vector<double>`）。

#### 7.3 FFT / 平台库调用

- `hamming` 不调用 FFT、不调用 thrust reduction、不调用任何 dlfft/cuFFT/cublas 接口。
- 与同文件中 `chebwin_fp32_compute_device`（`windows_typed.cu:14-72`，需要 thrust
  `max_element`）和 `taylor_fp32_compute_device`（`windows_typed.cu:79-113`，需要 host
  系数生成 + H2D）相比，`hamming` 是 windows 模块中最简单的 kernel，无任何依赖库。
- 因此 `hamming` 不受 ZQ500 FFT 接口（如 `FFTInterface`、`dlfft`）稳定性影响。

#### 7.4 错误处理

| 错误源 | 触发条件 | 异常类型 | 异常消息（含 "hamming" 关键字） |
| --- | --- | --- | --- |
| 主接口 size 校验 | `out.size() != max(length, 0)` | `std::invalid_argument` | `"hamming_device: output size must equal max(length, 0)"` |
| host wrapper size 校验 | `n < 1` 或 `o.size() != n` | `std::invalid_argument` | `"hamming_device: compute output size must equal positive n"` |
| CUDA kernel 启动/执行失败 | `CUDA_KERNEL_CHECK()` 触发 | 项目异常机制（`cuda_error.h`） | 不含 "hamming" 字面量 |
| CPU reference | 无显式异常路径（`length < 1` 返回空） | — | — |

注意两条 host 路径的异常消息都以 `hamming_device:` 开头，使测试可通过
`error.what().find("hamming")` 统一捕获（见第 11 节）。

#### 7.5 长度守卫对比

| 算子 | length<0 行为 | length==0 行为 | length==1 行为 |
| --- | --- | --- | --- |
| `hamming_typed_cpu` | 返回 `{}`（不抛） | 返回 `{}` | 返回 `{1.0}` |
| `hamming_device` | `expected=0`；若 `out.size()==0` 则返回空，否则抛 | 同上 | 走 kernel 短路分支 |
| `hamming_resident_device` | host wrapper 抛 `std::invalid_argument` | host wrapper 抛 `std::invalid_argument` | 走 kernel 短路分支 |

resident 接口对 `n < 1` 的语义与主接口不同：主接口在 `expected=0` 时短路返回空数组，
resident 接口要求调用者自行处理 `n < 1`，否则直接抛异常。这是因为 resident 路径
假设调用者已经预分配了正确大小的 `out`，`n < 1` 通常意味着调用者逻辑错误。

---

### 八、数学/物理原理 → CPU 代码 → GPU 代码 三方映射表

阶段一已建立 6 条原理（见 `数学物理原理.md`）。下表把每条原理定位到
cuSignal Python、C++ CPU、C++ GPU 三方代码的具体行号。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 原理 1：广义 Hamming 窗族 $w[n]=\alpha-(1-\alpha)\cos(2\pi n/(M-1))$ | `cusignal/windows/windows.py:1116`（基准）`w = 0.54 - 0.46 * cos(...)` | `windows_typed.cpp:387-388` `0.54 - 0.46 * std::cos(2.0*pi64*index/(effective-1))` | `windows_kernels.cuh:118-120` `0.54F - 0.46F * cos(2*pi*index/(effective_count-1))` | 三方均以专用 kernel/循环直接展开 $\alpha=0.54$ 公式；CPU 用 FP64，GPU 用 FP32，Python 用 FP64 |
| 原理 2：Hamming 窗特例（$\alpha=0.54$ 固化） | `windows.py:1116` 字面量 `0.54`、`0.46` | `windows_typed.cpp:387` 字面量 `0.54`、`0.46` | `windows_kernels.cuh:118` 字面量 `0.54F`、`0.46F` | 三方均把 $\alpha$ 烧进字面量，无运行时参数；与 `general_cosine` 通过系数数组传入不同 |
| 原理 3：频域三 sinc 叠加 | `windows.py:1149-1152`（学习副本函数文档数学定义） | 无显式实现（CPU 只生成时域序列） | 无显式实现（GPU 只生成时域序列） | 三方均不显式计算频域；频域特性是数学不变量，由时域公式保证 |
| 原理 4：0.54 系数优化来源 | `windows.py:1116` 字面量 `0.54` | `windows_typed.cpp:387` 字面量 `0.54` | `windows_kernels.cuh:118` 字面量 `0.54F` | 三方均直接使用优化结果 $\alpha=0.54$，无优化过程；优化在原理推导阶段完成 |
| 原理 5：0 阶连续性 → $-6$ dB/octave | 由 `0.54`/`0.46` 字面量隐含决定端点值 $0.08\neq 0$ | 同左 | 同左 | 三方均不显式计算衰减率；该特性是公式 $\alpha=0.54$ 的数学推论 |
| 原理 6：加窗原理与时域乘法 | `windows.py:1250`（学习副本）`return w`（窗序列交付给下游） | `windows_typed.cpp:390` `return out` | `windows_typed.cpp:405` `out = finalize_fp64_device_storage(computed)`；resident 路径直接写 `out` | 三方均只生成窗序列，不执行加窗乘法；加窗由调用者完成。resident 接口为此提供了直接交付 FP32 的捷径 |
| 周期窗扩展（`sym=false` 且 $M$ 偶数 → $M+1$ 后截断） | `windows.py:1240-1241`（学习副本）`if not sym and not odd: M = M + 1` + `windows.py:1247-1248` `w = w[:-1]` | `windows_typed.cpp:381` `effective = (!symmetric && length%2==0) ? length+1 : length`，循环到 `length-1` 自然截断 | `windows_typed.cu:76` 第 5 参数 `(!sym && n%2==0) ? n+1 : n`，kernel 写入 `output_count=length` 个元素自然截断 | 三方逻辑等价；CPU/GPU 通过 effective_count 控制公式分母，通过循环/output_count 控制写入长度，省去 Python 的 `w[:-1]` 显式切片 |
| 长度守卫（$M<1$ → 空数组） | `windows.py:1232`（学习副本）`if M < 1: return cp.array([])` | `windows_typed.cpp:380` `if (length < 1) return {}` | `windows_typed.cpp:396-404` `expected = length>0 ? size_t(length) : 0; if (expected != 0) ...` | 三方均返回空；CPU/GPU 用 `< 1` 而非 `_len_guards`；GPU 在 `expected==0` 时跳过 kernel |
| 单点窗（$M=1$ → `[1.0]`） | `windows.py:1235`（学习副本）`if M == 1: return cp.ones(1, "d")` | `windows_typed.cpp:385-386` `effective == 1 ? 1.0 : ...`（不短路，进循环） | `windows_kernels.cuh:115-116` `Scalar value = 1; if (effective_count != 1) {...}`（短路） | Python/C++ 都短路到 1.0，CPU 不短路但结果相同；GPU 在 kernel 内通过默认值短路 |
| $\pi$ 常数 | `windows.py:1116` 使用 CuPy `M_PI`（FP64） | `windows_typed.cpp:383` `pi64 = 3.141592653589793238462643383279502884`（FP64） | `windows_kernels.cuh:117` `pi = 3.14159265358979323846F`（FP32） | Python/CPU 用 FP64 $\pi$；GPU 用 FP32 $\pi$。这是 GPU 路径的精度差异来源之一 |
| 预计算 $1/(M-1)$ | `windows.py:1151`（学习副本）`loop_prep = "const double N { 1.0 / ( _ind.size() - 1 ) };"`，kernel 内 `i * N` | `windows_typed.cpp:388` 直接 `2.0*pi64*index/(effective-1)`（无预计算） | `windows_kernels.cuh:119-120` 直接 `2*pi*index/(effective_count-1)`（无预计算） | Python 通过 `loop_prep` 预乘 $N=1/(M-1)$ 把除法变成乘法；C++ CPU/GPU 直接除，依赖编译器把 `(effective-1)` 提到循环/kernel 外（推测） |

---

### 九、与 cuSignal Python 实现相同、等价替换和有意不同的部分

#### 9.1 相同（公式与算法本质一致）

1. **核心公式逐字一致**：Python `0.54 - 0.46 * cos(...)`、CPU `0.54 - 0.46 * std::cos(...)`、
   GPU `0.54F - 0.46F * cos(...)` 三方公式完全相同，只是字面量精度不同。
2. **周期窗扩展逻辑一致**：三方均以 `(!sym && M%2==0) → M+1` 扩展，并在写入时截断。
3. **长度守卫一致**：三方均以 `M < 1 → 空数组` 返回，不抛异常。
4. **单点窗一致**：三方均返回 `[1.0]`（CPU/GPU 通过 `effective==1` 判定，Python 直接短路）。
5. **专用 kernel 选择一致**：Python 用专用 `_hamming_kernel` 而非 `general_hamming` 或
   `general_cosine`；C++ 同样用专用 `hamming_kernel` 而非 `general_cosine_kernel`。
   这是性能优化选择，避免通用框架的数组拼接开销。
6. **不调用 `_len_guards`**：Python `hamming` 不调用 `_len_guards`，C++ 也不调用对应
   通用 helper，直接在函数体内做 `length < 1` 判断。两者一致。
7. **不检查 `M` 是否为整数**：Python `hamming` 不检查，C++ 也不检查（`int` 类型已保证）。

#### 9.2 等价替换（数学等价，工程实现不同）

1. **截断方式**：Python 显式 `w = w[:-1]`；C++ CPU 通过循环上限 `length-1` 隐式截断；
   C++ GPU 通过 `output_count=length` 而分母用 `effective_count=length+1` 隐式截断。
   三方数学等价，C++ 避免了一次数组切片。
2. **单点短路位置**：Python 在函数入口 `if M == 1: return cp.ones(1, "d")` 短路；
   C++ CPU 不短路，进入循环后由 `effective == 1` 触发 `1.0`；C++ GPU 在 kernel 内
   由 `value = 1` 默认值 + `if (effective_count != 1)` 短路。三方结果一致。
3. **$1/(M-1)$ 预计算**：Python 用 `loop_prep` 在 kernel 启动前预计算 $N$ 并在 kernel 内
   做乘法；C++ CPU/GPU 直接在循环/kernel 内做除法。数学等价；推测编译器会把
   `effective-1` 提到循环/kernel 外，但 Python 的 `loop_prep` 是显式优化，C++ 是隐式依赖。
4. **空数组表示**：Python 返回 `cp.array([])`（CuPy 空数组）；C++ CPU 返回 `std::vector<double>{}`
   （空 vector）；C++ GPU 返回空 `DeviceArray<double>`。三者语义等价。

#### 9.3 有意不同

1. **GPU FP32 计算 + FP64 输出（主接口）**：cusignal_cpp 主接口在 GPU 端用 FP32 完成全部
   计算，再在 Host 上拓宽到 FP64 写回。Python 全程 FP64。这是工程近似，目的是
   减少 GPU FP64 算术开销（ZQ500 等 GPU 的 FP64 吞吐远低于 FP32）。代价是 GPU
   结果与 Python/CPU 之间存在 FP32 舍入误差，需要测试容差吸收（见第 10、11 节）。
2. **$\pi$ 字面量精度不同**：CPU 用 FP64 `pi64`（30 位有效数字），GPU 用 FP32
   `pi = 3.14159265358979323846F`（实际为 FP32 精度，约 7 位有效数字）。这是 GPU
   路径的第二个精度差异来源。
3. **resident FP32 接口（cusignal_cpp 独有）**：`hamming_resident_device` 跳过
   `finalize_fp64_device_storage`，直接把 FP32 输出交付给下游 FP32 算子。
   Python 没有对应接口。这是 cusignal_cpp 为 FP32 流水线设计的优化，避免
   D2H/H2D/拓宽开销。代价是输出 dtype 与 `hamming_device` 不同，调用者必须显式选择。
4. **负长度处理差异**：Python 在 `M < 1` 时短路返回空数组，对负数也返回空；
   C++ 主接口同样把 `length < 0` 映射成 `expected = 0` 返回空数组（不抛）；
   但 **resident 接口对 `n < 1` 直接抛异常**（因 host wrapper 校验更严格）。这是
   resident 与主接口的有意差异，反映「resident 路径要求调用者保证输入合法性」的契约。
5. **错误消息含算子名**：cusignal_cpp 的异常消息以 `hamming_device:` 开头，便于测试
   通过 `error.what().find("hamming")` 捕获。Python 没有这种 size 校验异常
   （CuPy 自动处理 size mismatch）。
6. **测试覆盖 size mismatch 场景**：cusignal_cpp 测试专门验证 `hamming_rejects_output_size`
   （传入 `length=5` 但 `out.size()=4` 期望抛异常且消息含 "hamming"）。Python 没有对应
   测试场景。
7. **不使用 `loop_prep`**：cusignal_cpp kernel 直接 `2*pi*index/(effective_count-1)`，
   每个 thread 多一次除法。Python 用 `loop_prep` 预乘 $N=1/(M-1)$。
   推测：编译器会把 `effective_count-1` 提到循环/kernel 外（因为它是 uniform 参数），
   但 cusignal_cpp 没有显式做此优化。
8. **显式实例化限定 5 种类型**：cusignal_cpp 通过 `INSTANTIATE_WINDOWS_CPU` 和
   `INSTANTIATE` 宏把模板实例化限定在 5 种类型，链接期拒绝其他类型。Python 是
   动态类型，无此限制。

---

### 十、精度、dtype、边界条件及 ZQ500 平台限制

#### 10.1 精度路径

| 路径 | 计算精度 | 输出 dtype | 与 cuSignal Python 偏差来源 |
| --- | --- | --- | --- |
| `hamming_typed_cpu` | FP64 | `std::vector<double>` | 仅 $\pi$ 字面量精度差异（30 位 vs Python `M_PI` 的 double 精度） |
| `hamming_device`（主） | FP32（GPU） + FP64 拓宽（Host） | `DeviceArray<double>` | FP32 计算 + FP32 $\pi$ 字面量；测试用 `matches_fp64` 容差 4e-2 吸收 |
| `hamming_resident_device` | FP32（GPU） | `DeviceArray<float>` | 同主接口，但不再拓宽 |

主接口的 FP32 路径在 ZQ500 上是性能选择：ZQ500 GPU 的 FP64 吞吐通常远低于 FP32，
把全部余弦计算放在 FP32 内可显著降低 kernel 时延。拓宽到 FP64 仅作为接口契约，
不增加 GPU 计算开销。

#### 10.2 dtype 分发

- 5 种 `T` 通过宏显式实例化，覆盖 `float`、`__half`、`int32_t`、`int16_t`、`int8_t`。
- 但 `hamming_kernel` 实际只实例化 `hamming_kernel<float>` 一种；其他 4 种 `T` 的
  kernel 是模板但未实例化，不会被 `hamming` 调用。
- 上层 `T` 仅用于：(1) 触发 `static_assert`；(2) 选择正确的 `hamming_fp32_compute_device<T>`
  显式实例化版本；(3) 决定 `DeviceArray<double>` 输出的存储语义（与 `T` 解耦）。
- resident 接口的 `out` 始终是 `DeviceArray<float>`，与 `T` 无关。

#### 10.3 边界条件

| 输入 | CPU 行为 | 主接口行为 | resident 接口行为 |
| --- | --- | --- | --- |
| `length < 0` | 返回 `{}` | `expected=0`；若 `out.size()==0` 返回空，否则抛 | 抛 `std::invalid_argument` |
| `length == 0` | 返回 `{}` | `expected=0`；同上 | 抛 `std::invalid_argument` |
| `length == 1` | 返回 `{1.0}` | kernel 短路，输出 `[1.0]` | kernel 短路，输出 `[1.0]`（FP32） |
| `length == 2, sym=true` | 计算 $[0.08, 0.08]$（FP64） | 计算 $[0.08, 0.08]$（FP32→FP64） | 计算 $[0.08, 0.08]$（FP32） |
| `length == 4, sym=false` | effective=5，截前 4 点 | 同 CPU | 同 CPU（FP32） |
| `length == 5, sym=true` | 5 点对称窗 | 同 CPU（FP32→FP64） | 同 CPU（FP32） |
| `out.size() != max(length, 0)` | N/A | 抛 `std::invalid_argument` | 抛 `std::invalid_argument` |

#### 10.4 ZQ500 平台限制

- **CMake 最低版本**：项目保持 `cmake_minimum_required(VERSION 3.16)` 兼容；
  `gpu_02` 容器实测 CMake `3.22.1`，但本算子未使用 3.16 之上的新特性。
- **CUDA language**：`hamming_kernel` 是 `__global__` 模板，无 `--x cuda` 特殊要求；
  与 ZQ500 样例一致的 CXX 语言模式即可。
- **FP64 性能**：ZQ500 GPU 的 FP64 吞吐通常为 FP32 的 1/16 ~ 1/32；
  cusignal_cpp 把 GPU 计算放在 FP32 是平台适配选择，不是任意近似。
- **`__half` 支持**：5 种 `T` 包含 `__half`，但 `hamming_kernel` 只实例化 `<float>`；
  `__half` 路径仅停留在 host wrapper 层，kernel 端无 `__half` 算术。
- **无 FFT 依赖**：`hamming` 不调用 `dlfft`/`FFTInterface`/cuFFT，不受 ZQ500 FFT
  接口稳定性影响。这是相对 `chebwin`（依赖 FFT 或 custom direct DFT）的平台优势。
- **无 thrust 依赖**：`hamming` 不调用 thrust reduction，避免 ZQ500 thrust 版本兼容问题。
- **默认 stream**：使用 `stream = nullptr`，与 ZQ500 默认 stream 语义一致；
  无显式 stream 管理，调用者若需异步须自行包装。

---

### 十一、测试如何验证正确性，以及仍未覆盖的风险

#### 11.1 测试入口与覆盖场景

测试位于 `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:578-604`，
在模板函数 `run_type<T>` 内、`ok[9]` 槽位执行。`test_evidence::f_begin_accuracy(type_dispatch::OperatorId::hamming)`
在 `e3_wave_window_type_smoke.cu:578` 起测。

测试覆盖 5 种 dtype（`float`、`__half`、`int32_t`、`int16_t`、`int8_t`），每种 dtype
跑同一组用例：

| 用例 | 代码位置（行号） | 期望行为 |
| --- | --- | --- |
| `hamming_device<T>(5, fixed_fp64_window_out)` 对称默认 | 579 | 输出与 `hamming_typed_cpu<T>(5)` 在 `matches_fp64` 容差内一致 |
| `hamming_device<T>(4, hamming_periodic_out, false)` 周期窗 | 583 | 输出与 `hamming_typed_cpu<T>(4, false)` 在容差内一致 |
| `hamming_device<T>(1, singleton_fp64_window_out)` 单点 | 586 | 输出 `== std::vector<double>{1.0}`（精确匹配，无容差） |
| `hamming_device<T>(0, empty_fp64_window_out)` 空窗 | 589 | `empty_fp64_window_out` 为空 |
| `hamming_device<T>(-1, empty_fp64_window_out)` 负长度 | 590 | 不抛异常；`empty_fp64_window_out` 仍为空（主接口 `expected=0` 短路） |
| `hamming_device<T>(5, invalid_hamming_out)` 其中 `invalid_hamming_out.size()==4` | 593–598 | 抛 `std::invalid_argument`，且 `error.what().find("hamming") != npos` |
| CPU 对照：`hamming_typed_cpu<T>(5)`、`(4, false)`、`(1)`、`(0)`、`(-1)` | 581, 585, 602–604 | `(1)` 返回 `{1.0}`；`(0)` 和 `(-1)` 返回空 |

最终判定（`e3_wave_window_type_smoke.cu:600-604`）：

```cpp
ok[9] = matches_fp64(hamming_actual, hamming_reference) &&
    hamming_periodic_matches &&
    hamming_singleton_is_one && hamming_typed_cpu<T>(1) == std::vector<double>{1.0} &&
    empty_fp64_window_out.empty() && hamming_typed_cpu<T>(0).empty() &&
    hamming_typed_cpu<T>(-1).empty() && hamming_rejects_output_size;
```

容差：`matches_fp64` 用 `4e-2`（任务描述确认）。这个相对宽松的容差用于吸收
FP32 GPU 计算与 FP64 CPU reference 之间的舍入误差，尤其是 $\pi$ 字面量精度差异。

#### 11.2 验证维度

- **5 种 dtype 各跑一遍**：通过 `run_type<T>` 模板实例化 5 次实现。
- **CPU/GPU 一致性**：`matches_fp64(hamming_actual, hamming_reference)` 直接对比
  GPU 主接口输出与 CPU reference。
- **对称/周期双场景**：覆盖 `sym=true`（默认）和 `sym=false`（周期窗）。
- **边界值**：单点 `length=1`、空窗 `length=0`、负长度 `length=-1`。
- **错误处理**：size mismatch 必须抛异常且消息含 "hamming"。
- **resident 接口**：测试未直接覆盖 `hamming_resident_device`（推测：resident 接口
  通过其他测试或下游算子间接验证；本测试文件只覆盖主接口）。

#### 11.3 仍未覆盖的风险

1. **resident 接口缺直接测试**：`e3_wave_window_type_smoke.cu` 未对
   `hamming_resident_device` 做直接正确性验证。如果调用者预分配的 `out` 大小与
   `length` 不匹配，host wrapper 会抛异常，但 resident 路径的 FP32 输出是否
   与主接口的 FP64 输出在数值上一致（除拓宽外），没有直接断言。
2. **`length == 2` 未单独覆盖**：测试只跑 `length = 5, 4, 1, 0, -1`，未覆盖
   `length = 2`。`length = 2` 时 `effective = 2`，分母 `effective - 1 = 1`，
   是除法分母为 1 的边界，未单独验证。
3. **`sym=false` 且 `length` 为奇数未覆盖**：测试只跑 `length=4, sym=false`（偶数扩展），
   未跑 `length=5, sym=false`（奇数不扩展）。奇数周期窗路径 `effective = length` 未直接验证。
4. **大长度未覆盖**：测试只跑 `length <= 5`，未覆盖大长度（如 `length = 1<<20`）。
   大长度下 grid 配置、线程索引计算、`effective_count - 1` 的 int 溢出风险均未验证。
   推测：`int` 在 32 位平台为 2^31-1，足够覆盖实际窗长度，但无显式测试。
5. **`__half` 精度路径未单独验证**：测试中 `T=__half` 走的是同一 `hamming_kernel<float>`
   路径，`__half` 仅影响 host wrapper 的显式实例化选择，不影响 kernel 内部计算。
   但测试未断言 `__half` 路径的输出与 `float` 路径一致。
6. **CUDA kernel 启动失败未覆盖**：测试未模拟 CUDA OOM 或 kernel 启动失败场景。
   `CUDA_KERNEL_CHECK()` 的异常路径未在 `hamming` 上下文中验证。
7. **默认 stream 同步语义未覆盖**：测试通过 `to_host()` 隐式同步，未验证
   异步场景下的同步点正确性。
8. **resident 接口对 `n < 1` 抛异常的行为未覆盖**：测试只验证主接口对
   `length < 0` 不抛（返回空），未验证 resident 接口对 `n < 1` 抛异常。
9. **`matches_fp64` 容差 4e-2 的合理性**：该容差相对宽松，可能掩盖 FP32 路径的
   系统性偏差。对于 `length=5` 的小窗，FP32 与 FP64 的实际偏差通常远小于 4e-2
   （推测在 1e-6 量级），但测试未量化实际偏差。

---

### 十二、自检问题

1. **API 包装层**：`hamming_device` 与 `hamming_resident_device` 的输出 dtype 有何不同？
   为什么 resident 接口不调用 `finalize_fp64_device_storage`？
2. **CPU 长度守卫**：`hamming_typed_cpu` 的判据是 `length < 1` 还是 `length < 0`？
   与 `parzen`、`taylor` 的 `length < 0` throw 路径有何区别？为什么？
3. **effective 计算**：`(!symmetric && length % 2 == 0) ? length + 1 : length` 在
   CPU、host wrapper、kernel 三处是否完全一致？三处的 `effective`/`effective_count`
   分别叫什么名字？
4. **GPU kernel 短路**：`hamming_kernel` 如何处理 `effective_count == 1` 的单点窗？
   `Scalar value = 1` 默认值的作用是什么？
5. **精度路径**：主接口的 GPU 计算、Host 拓宽、GPU 存储 分别使用什么 dtype？
   为什么 GPU 端不执行 FP64 算术？
6. **显式实例化**：`hamming_kernel` 的模板参数 `T` 实际只实例化哪一种？为什么
   上层 `T` 可以是 5 种而 kernel 只有 1 种？
7. **grid/block 配置**：`launch_1d_kernel` 默认的 `block_size` 是多少？对 `length=5`
   会启动多少个 grid block？多少个 thread 立即 return？
8. **resident 接口的负长度行为**：为什么 resident 接口对 `n < 1` 抛异常，而主接口
   返回空数组？这反映了两条路径怎样的契约差异？
9. **Python `loop_prep` vs C++ 直接除法**：cuSignal Python 用 `loop_prep` 预计算
   `N = 1/(M-1)`，cusignal_cpp 直接在 kernel 内做 `2*pi*index/(effective_count-1)`。
   这两种写法在数学上等价，但在工程上有何差异？编译器是否能自动把
   `effective_count-1` 提到 kernel 外？（推测）
10. **测试容差**：`matches_fp64` 用 `4e-2` 容差吸收 FP32 与 FP64 的偏差。这个容差
    相对于 `length=5` 时的实际偏差（推测 1e-6 量级）是否过于宽松？可能掩盖什么问题？
11. **错误消息**：主接口和 host wrapper 的异常消息都以 `hamming_device:` 开头。为什么
    这样设计？测试如何利用这一点？
12. **resident 接口测试覆盖**：`e3_wave_window_type_smoke.cu` 是否直接测试了
    `hamming_resident_device`？如果没有，该接口的正确性如何保证？
13. **平台限制**：`hamming` 不依赖 FFT、thrust、cuFFT，这对 ZQ500 平台有什么好处？
    相对 `chebwin`（依赖 thrust `max_element`）有什么优势？
14. **版本管理**：本文件结构允许后续优化版本以 V2、V3 追加。如果未来在 kernel 中
    引入 `loop_prep` 等价优化，应该新增 V2 章节还是覆盖 V1？为什么？

---

### 十三、阅读建议

按以下顺序阅读源码可快速建立对 `hamming` 复现逻辑的完整理解：

1. **先看 API 包装层**：`cusignal_cpp/src/windows/windows_typed.h:133-155` 三条声明
   （`hamming_typed_cpu`、`hamming_device`、`hamming_resident_device`），建立
   「CPU reference / FP64 主接口 / FP32 resident 接口」三个层次的轮廓。
2. **看 CPU reference**：`windows_typed.cpp:377-391`。这是公式最直接的实现，无 GPU
   干扰。注意 `length < 1` 守卫和 `effective` 计算。
3. **看主接口实现**：`windows_typed.cpp:393-406`。关注 `expected = length > 0 ? ... : 0`
   的负长度处理，以及 `finalize_fp64_device_storage` 的 D2H→拓宽→H2D 路径。
4. **看 host wrapper**：`windows_typed.cu:76`。关注 `static_assert`、size 校验、
   `effective_count` 的第 5 参数传递，以及 `hamming_kernel<float>` 的固定实例化。
5. **看 kernel**：`windows_kernels.cuh:109-123`。关注线程索引、边界检查、
   `effective_count != 1` 短路、FP32 字面量、`window_store<T>` 输出。
6. **看 resident 接口**：`windows_typed.cpp:408-415`。与主接口对比，理解
   「跳过 finalize」如何省去 D2H/H2D。
7. **看显式实例化**：`windows_typed.cpp:562-595`（CPU 侧宏）和
   `windows_typed.cu:116-126`（GPU 侧宏），理解 5 种 dtype 如何通过宏展开。
8. **看测试**：`e3_wave_window_type_smoke.cu:578-604`。对照第 11.1 节的用例表，
   逐条理解每个 `hamming_device<T>(...)` 调用的期望行为。
9. **看辅助工具**：`cuda_utils/kernel_launch.h:44-56`（`launch_1d_kernel`）和
   `cuda_utils/host_output_finalize.h`（`finalize_fp64_device_storage`），理解
   grid/block 默认配置和 FP32→FP64 拓宽的具体路径。
10. **回到映射表**：第 8 节的三方映射表把上述代码片段与阶段一的 6 条原理对应起来，
    读完源码后再看映射表，可检验自己是否理解了「公式 → 代码」的对应关系。

---

## V2：第一次优化版本

> 占位。当且仅当 `hamming` 算子发生已提交的优化（例如引入 `loop_prep` 等价预计算、
> 改用 shared memory、resident 接口增加异步 stream 支持、或加入 `__half` kernel
> 实例化等）时，在本章节追加 V2 内容，包含完整 SHA、新代码定位、新调用链、
> 与 V1 的差异表格。**不得覆盖 V1 章节**。

## V3：第二次优化版本

> 占位。同 V2 规则。

---

## 版本差异与原理不变量

> 当前只有 V1，本节为占位。当 V2/V3 出现时，用下表比较算法结构、复杂度、
> 内存访问、并行方式、精度边界和性能变化，并指出哪些数学物理原理始终未变。

| 维度 | V1 | V2 | V3 |
| --- | --- | --- | --- |
| 完整 SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` | — | — |
| 算法结构 | 专用 `hamming_kernel`，1D 线性索引，每线程 1 元素 | — | — |
| 复杂度 | $O(M)$ 时间，$O(M)$ 临时 FP32 缓冲（主接口） | — | — |
| 内存访问 | 全局内存写入，无 shared memory | — | — |
| 并行方式 | 1D grid/block，block_size=256，默认 stream | — | — |
| 精度边界 | GPU FP32 + FP32 $\pi$；Host 拓宽到 FP64；resident 路径全 FP32 | — | — |
| 性能变化 | 基线 | — | — |
| 数学物理原理不变量 | 原理 1–6 全部不变（公式 $\alpha=0.54$ 固化、effective 计算逻辑、长度守卫、单点短路、加窗用途） | — | — |

---

## 完成质量检查

按 `AGENTS.md` 第六节清单逐项确认：

- 未混入其他算子泛化内容；所有代码定位均指向 `hamming` 相关符号。
- 通用原理引用阶段一已检索的 6 条原理，未从 cuSignal 反推。
- cuSignal 算法已通过第 8 节映射表逐项映射到原理 1–6。
- 未把 helper（`window_load`/`window_store`/`launch_1d_kernel`/`finalize_fp64_device_storage`）
  误当作核心算法；核心算法定位到 `hamming_typed_cpu`、`hamming_fp32_compute_device`、
  `hamming_kernel`。
- 每个公式中的符号（$M$、$n$、$\alpha$、$\pi$、`effective`、`effective_count`）均已解释。
- 每条原理都定位到 CPU/GPU 具体行号（第 8 节）。
- 源码文件、符号名、行号均已通过 Read 工具核对，真实存在。
- 本阶段不涉及 Python 源码摘录（Python 阶段已在 `python源码算法.md` 完成）。
- CPU 与 GPU 职责未混写：CPU 章节只讲 CPU，GPU 章节分 API/host wrapper/kernel 三层。
- 已绑定完整 Git SHA `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c`；无旧版可覆盖。
- 未声称执行过实际未执行的代码或测试；测试行为描述基于源码阅读，未在 ZQ500 上运行。
- 文档足以让学习者按第 13 节顺序阅读源码并复述执行逻辑。

不一致保留：

- resident 接口对 `n < 1` 抛异常，主接口返回空数组——保留并解释为契约差异。
- GPU 用 FP32 + FP32 $\pi$，CPU 用 FP64 + FP64 $\pi$——保留并解释为平台性能选择。
- cusignal_cpp 不使用 `loop_prep`，Python 使用——保留并解释为隐式 vs 显式优化。
- resident 接口在测试中未直接覆盖——保留并在第 11.3 节标为风险。
