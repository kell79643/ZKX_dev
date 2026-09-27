# cusignal_cpp_triang 复现逻辑

> 本文件记录 `cusignal_cpp` 中 `triang`（三角窗）算子 CPU/GPU 实现的复现学习。版本按 `operator_comprehension/Learning/AGENTS.md` "同一算子的优化演进记录" 组织：V1 为首次学习版本，V2/V3 占位待追加。所有代码定位使用"完整提交 SHA + `ZKX_dev/` 相对路径 + 符号名 + 该提交中的准确行号"格式。

## 版本索引

| 版本 | 状态 | 短 SHA | 完整 SHA | 提交时间 | 分支 | dirty 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [V1（当前实现）](#v1原始学习版本1ccd32ea) | 已学习 | `1ccd32ea` | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` | 2026-07-27 22:12:16 +0800 | `snapshot/all-current-20260716` | 干净 |
| V2 | 待追加 | — | — | — | — | — |
| V3 | 待追加 | — | — | — | — | — |

---

## V1：原始学习版本（`1ccd32ea`）

### 1. Git 版本信息

- 完整 SHA：`1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c`
- 短 SHA：`1ccd32ea`
- 提交时间：2026-07-27 22:12:16 +0800
- 分支：`snapshot/all-current-20260716`
- `cusignal_cpp` dirty 状态：干净
- 相关文件（在 `ZKX_dev/` 下的相对路径）：
  - `cusignal_cpp/src/windows/windows_typed.h`
  - `cusignal_cpp/src/windows/windows_typed.cpp`
  - `cusignal_cpp/src/windows/windows_typed.cu`
  - `cusignal_cpp/src/windows/windows_kernels.cuh`
  - `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`
- 依赖的辅助文件（非 triang 专属，但在调用链中被复用）：
  - `cusignal_cpp/src/cuda_utils/kernel_launch.h`（`launch_1d_kernel`）
  - `cusignal_cpp/src/cuda_utils/host_output_finalize.h`（`finalize_fp64_device_storage`）
  - `cusignal_cpp/src/cuda_utils/simple_signal_typed.h`（`is_simple_signal_input_v`、`SimpleSignalTypePolicy`）

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注"无直接原理对应"。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在"含义解释"列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §2.1 `windows_typed.h:272-273`（`triang_device` 模板参数） | `template <class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，仅用于显式实例化与 `static_assert`；5 种类型：`float`/`__half`/`std::int32_t`/`std::int16_t`/`std::int8_t`；不参与计算，CPU 输出固定 `double`，GPU kernel 固定 `<float>`（`windows_typed.cu:114` 硬编码 `triang_kernel<float>`） |
| `n` | §2.1 `windows_typed.h:272-273`（`triang_device` 首参数） | `int n`，CPU 入口记作 `length` | 原理 4 §"变体 B"中的 $M$ | 窗长度（点数），$M \geq 0$；CPU 路径专用名为 `length`（`windows_typed.cpp:531`）；GPU 路径在 `triang_fp32_compute_device` 内作为 size 校验依据（`n < 1` 抛异常） |
| `out` | §2.1 `windows_typed.h:272-273`（`triang_device` 第二参数） | `DeviceArray<double>& out`（GPU 主接口）/`std::vector<double>` 返回值（CPU）/`DeviceArray<float>&`（host wrapper compute 层） | 原理 4 §"变体 B"中的 $w[n]$ | 输出窗序列；CPU 全程 FP64；GPU 主接口由 `finalize_fp64_device_storage` 拓宽为 FP64；compute 层直接交付 FP32 临时缓冲给 kernel 写入 |
| `sym` | §2.1 `windows_typed.h:272-273`（`triang_device` 第三参数） | `bool sym = true` | 原理 4 §"变体 B"（symmetric/periodic 切换，对应 `$M$` vs `$M+1$` 分母） | 工程旗标，`true` 对称窗（`effective = length`，用于 FIR 设计），`false` 周期窗（`effective = length + 1` 再截断，用于频谱分析）；CPU 侧以 `symmetric` 形式出现于 `windows_typed.cpp:531` |
| `length` | §2.2 `windows_typed.h:259-260`（`triang_typed_cpu` 首参数，记作 `length`） | `int n`（CPU 入口记作 `length`） | 原理 4 §"变体 B"中的 $M$ | 窗长度（点数），CPU 路径专用名；与 GPU 侧 `n` 同义；`length < 0` 抛 `std::invalid_argument`（`windows_typed.cpp:534`） |
| `symmetric` | §2.2 `windows_typed.h:259-260`（`triang_typed_cpu` 第二参数，记作 `symmetric`） | `bool sym = true` | 原理 4 §"变体 B"（symmetric/periodic 切换） | 工程旗标，CPU 路径专用名；与 GPU 侧 `sym` 同义；控制 `effective` 计算分支 |
| `effective` | §3.1 `windows_typed.cpp:535`（`triang_typed_cpu` 局部） | `int`，`symmetric ? length : length + 1` | 原理 4 §"变体 B"（周期窗 $M \to M+1$ 扩展） | 工程变量，对称中心化长度；对称窗等于 `length`，周期窗扩展为 `length + 1`（多算一个采样再截断）；GPU 侧以 `effective_count` 形式出现于 `windows_typed.cu:114` 第 5 个 kernel 参数 |
| `midpoint` | §3.1 `windows_typed.cpp:542`（`triang_typed_cpu` 局部） | `int`，`effective / 2`（整数除法） | 原理 4 §"变体 B"中的 $\lfloor M/2 \rfloor$（rank 镜像分界点） | 工程变量，半长索引；左半 `index < midpoint` 时 `rank = index + 1`，右半 `rank = effective - index`；GPU kernel 侧同名出现于 `windows_kernels.cuh:185` |
| `rank` | §3.1 `windows_typed.cpp:543`（`triang_typed_cpu` 局部） | `int`，`index < midpoint ? index + 1 : effective - index` | 原理 4 §"变体 B"中的 $n+1$（左半）与 $M-n$（右半） | 工程变量，镜像序号；用于奇偶分支公式的分子计算；GPU kernel 侧同名出现于 `windows_kernels.cuh:186` |
| `index` | §3.1 `windows_typed.cpp:537`（`triang_typed_cpu` 循环变量） | `int`，`for (int index = 0; index < length; ++index)` | 原理 4 §"变体 B"中的 $n$（$0 \leq n \leq M-1$） | 时域采样索引；GPU kernel 侧以 `index = blockIdx.x * blockDim.x + threadIdx.x` 形式出现于 `windows_kernels.cuh:179` |
| `computed` | §4.1 `windows_typed.cpp:557`（`triang_device` 局部） | `DeviceArray<float> computed(out.size())` | 无直接原理对应 | 工程 FP32 临时缓冲区，承载 GPU kernel 的 FP32 计算结果，随后经 `finalize_fp64_device_storage` 拓宽为 FP64；resident 接口不存在（triang 无 resident 接口） |
| `effective_count` | §4.3 `windows_typed.cu:114`（`triang_fp32_compute_device` 第 5 个 kernel 参数） | `int`，`sym ? n : n + 1` | 原理 4 §"变体 B"（周期窗 $M \to M+1$ 扩展） | 工程变量，GPU 侧的对称中心化长度，与 CPU `effective` 同义；作为 kernel 参数传入 `triang_kernel`，控制奇偶分支判断与公式分母 |
| `output_count` | §2.4 `windows_kernels.cuh:176-194`（`triang_kernel` 首参数） | `int output_count`，等于 `n`（用户请求长度） | 无直接原理对应 | 工程变量，kernel 写入的输出元素数；`sym=false` 且 `n` 为偶数时小于 `effective_count`，通过线程边界检查 `if (index >= output_count) return` 完成截断 |
| `output` | §2.4 `windows_kernels.cuh:176-194`（`triang_kernel` 第二参数） | `T* output`（实际 `float*`，硬编码 `<float>`） | 原理 4 §"变体 B"中的 $w[n]$ | kernel 输出指针，对应 `computed.data()`；最终经 `finalize_fp64_device_storage` 拓宽为 FP64 |
| `Scalar` | §4.4 `windows_kernels.cuh:187`（`triang_kernel` 局部 `using` 声明） | `using Scalar = decltype(window_load(output[index]))`，固定 FP32 | 无直接原理对应 | 工程 kernel 计算精度类型，固定 `float`；用于 `value` 等局部变量类型推导；显式实例化只产生 `triang_kernel<float>` 一份 |
| `value` | §4.4 `windows_kernels.cuh:188`（`triang_kernel` 局部） | `const Scalar value`，奇偶分支计算 | 原理 4 §"变体 B"中的 $w[n]$ | 工程变量，kernel 内计算的窗值；奇 `effective_count` 时 `2*rank/(effective_count+1)`，偶 `effective_count` 时 `(2*rank-1)/effective_count`；FP32 计算 |
| `window_load<T>` | §5.2 `windows_kernels.cuh:10-18`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::load`，5 种 dtype 统一转 `float`；在 `triang_kernel` 内用于推导 `Scalar` 类型 |
| `window_store<T>` | §5.2 `windows_kernels.cuh:20-28`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::store`，把 FP32 `value` 转回 `T`；实际只实例化 `T=float`，等价于 `static_cast<float>`，无 clamp |
| `SimpleSignalTypePolicy<T>` | §5.2 `cuda_utils/simple_signal_typed.h:16-44`（类型策略模板） | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断）；`triang` 路径实际只用 `T=float` 特化 |
| `launch_1d_kernel` | §4.3 `cuda_utils/kernel_launch.h:44-56`（kernel 启动工具） | 函数模板，参数 `(Kernel kernel, std::size_t elements, Args&&... args)` | 无直接原理对应 | 工程 kernel 启动工具，自动按 `elements` 计算 grid/block（默认 `block_size=256`，默认 stream），后跟 `CUDA_KERNEL_CHECK()` 错误检查 |
| `block_size` | §6 `cuda_utils/kernel_launch.h:21-29`（`make_1d_launch_config` 内部常量） | `dim3(256)`，硬编码默认 | 无直接原理对应 | 工程变量，CUDA 线程块大小；`triang_kernel` 每 block 256 线程，`grid.x = div_up(n, 256) = (n + 255) / 256` |
| `finalize_fp64_device_storage` | §4.1 `cuda_utils/host_output_finalize.h:39-44`（FP32→FP64 拓宽工具） | 函数，参数 `(DeviceArray<float> computed)` 返回 `DeviceArray<double>` | 无直接原理对应 | 工程 FP32→FP64 拓宽工具，内部走 D2H → `finalize_fp64_on_host` host float→double → H2D；ZQ500 平台 FP64 算术受限的工程选择（注释见 `host_output_finalize.h:14-17`） |
| `is_simple_signal_input_v<T>` | §5.2 `cuda_utils/simple_signal_typed.h:18-41`（类型 trait） | `template <class T> constexpr bool is_simple_signal_input_v` | 无直接原理对应 | 工程类型 trait，编译期判断 `T` 是否属于 5 种简单信号类型，用于 `static_assert`；在 host wrapper 触发，拒绝 FP64 与其他不支持的类型进入 GPU 计算 |
| `INSTANTIATE_WINDOWS_CPU` | §5.2 `windows_typed.cpp:562-593`（CPU 侧实例化宏） | 宏，展开为 9 个 windows 算子 CPU/GPU 包装的 `template ... ;` 声明 | 无直接原理对应 | 工程 CPU 侧显式实例化宏，一次展开 5 种 `T`；`triang` 相关两行位于 `windows_typed.cpp:586-587` |
| `INSTANTIATE` | §5.2 `windows_typed.cu:116-127`（GPU 侧实例化宏） | 宏，展开为 9 个 `*_fp32_compute_device<T>` 的 `template void ... ;` 声明 | 无直接原理对应 | 工程 GPU 侧显式实例化宏，一次展开 5 种 `T`；`triang` 相关一行位于 `windows_typed.cu:125` |
| `matches_fp64` | §11 `e3_wave_window_type_smoke.cu:719-741`（测试容差比较函数） | 测试工具函数，参数 `(actual, reference)`，容差 `4e-2` | 无直接原理对应 | 工程测试工具，FP32 GPU 输出与 FP64 CPU reference 的容差比较；容差 4e-2 与其他 window 算子共用，对 `triang` 远超所需（实际偏差约 $6 \times 10^{-8}$） |

---

### 2. 对外接口与代码定位

按 `AGENTS.md` 阶段三要求，明确区分 API 包装层、CPU/reference 核心计算、GPU host 调度层、GPU kernel/device 核心计算、测试与 benchmark。

#### 2.1 API 包装层（GPU 对外接口）

- 声明：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:272-273` / `cusignal::triang_device<T>`
- 实现：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:551-560`

签名：

```cpp
template <class T>
void triang_device(int n, DeviceArray<double>& out, bool sym = true);
```

#### 2.2 CPU/reference 核心计算

- 声明：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:259-260` / `cusignal::triang_typed_cpu<T>`
- 实现：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:531-549`

签名：

```cpp
template <class T>
std::vector<double> triang_typed_cpu(int n, bool sym = true);
```

#### 2.3 GPU host 调度层

- 前置声明：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:160-161`
- 实现：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:114` / `cusignal::triang_fp32_compute_device<T>`

签名：

```cpp
template <class T>
void triang_fp32_compute_device(int n, DeviceArray<float>& o, bool sym);
```

#### 2.4 GPU kernel/device 核心计算

- 定义：`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:176-194` / `cusignal::windows_detail::triang_kernel<T>`

签名：

```cpp
template <typename T>
__global__ void triang_kernel(int output_count, T* output, int effective_count);
```

#### 2.5 测试与 benchmark

- 测试驱动：`1ccd32ea` / `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:719-741`，位于 `run_type<T>` 模板函数内，对应 `ok[13]`（`labels[13] == "triang"`，见同文件 `743-745`）。
- 没有发现独立的 `triang` benchmark 文件（搜索范围：`cusignal_cpp/test/`、`cusignal_cpp/bench/`（不存在））。

### 3. CPU 调用链、主要数据结构和逐步算法

#### 3.1 调用链

`triang_typed_cpu<T>` 是单一函数，无 helper 调用、无 FFT、无归一化。完整 CPU 路径（行号见 §2.2）：

```
triang_typed_cpu<T>(length, symmetric)        // windows_typed.cpp:531-549
  ├─ if length < 0 → throw std::invalid_argument("triang: length must be nonnegative")  // L534
  ├─ effective = symmetric ? length : length + 1               // L535
  ├─ std::vector<double> out(length)                          // L536
  └─ for index = 0 .. length-1:                                // L537-547
       if effective == 1:                                      // L538
           out[index] = 1.0; continue                          // L539-540
       midpoint = effective / 2                               // L542 (整数除法)
       rank = index < midpoint ? index + 1 : effective - index // L543
       out[index] = (effective & 1)                           // L544
           ? 2.0 * rank / (effective + 1)                     // L545 (奇 effective)
           : (2.0 * rank - 1.0) / effective                   // L546 (偶 effective)
  └─ return out                                                // L548
```

#### 3.2 主要数据结构

- 输入：`int length`（窗长度）、`bool symmetric`（对称/周期，默认 `true`）。
- 输出：`std::vector<double> out(length)`，FP64 host 连续容器。
- 内部标量：
  - `int effective`：参与计算的有效长度，对称时等于 `length`，周期时为 `length + 1`。
  - `int midpoint`：半长索引 $\lfloor \text{effective}/2 \rfloor$。
  - `int rank`：当前位置的"序号"，从左端起 `index+1` 递增、右端起 `effective-index` 递减，左右镜像。
- 模板参数 `T`：仅用于显式实例化分派（§5.2），函数体内不使用 `T`，输出固定 `double`。换言之，`T` 在 `triang_typed_cpu` 中是类型 tag，不影响数值结果。

#### 3.3 逐步算法

1. **长度守卫**：`length < 0` 抛 `std::invalid_argument`。`length == 0` 不抛，`out` 已为空，循环不进入，返回空 vector。
2. **effective 计算**：`symmetric ? length : length + 1`。周期窗多算一个采样再截断；CPU 直接用 `effective` 参与公式，无显式截断步骤（输出长度始终为 `length`）。
3. **逐点循环**（`index = 0 .. length-1`）：
   - 单点退化：`effective == 1` 时直接写 `1.0`，避免除零。这覆盖 `symmetric=true, length=1`。`symmetric=false, length=0` 不进入循环（`length==0` 短路）。
   - `midpoint = effective / 2`：整数除法，等价于 $\lfloor M/2 \rfloor$。
   - `rank`：`index < midpoint` 时取 `index + 1`，否则取 `effective - index`。镜像序号。
   - 奇偶分支（CPU 用 FP64 浮点运算）：
     - 奇数 `effective`（`effective & 1 == 1`）：$w[i] = \dfrac{2 \cdot \text{rank}}{M+1}$
     - 偶数 `effective`：$w[i] = \dfrac{2 \cdot \text{rank} - 1}{M}$
   - 其中 $M$ 表示 `effective`，$w[i]$ 表示 `out[index]`。

数值示例（CPU FP64）：
- `sym=true, length=5`（`effective=5` 奇）：$w = [2/6, 4/6, 6/6, 4/6, 2/6] = [1/3, 2/3, 1, 2/3, 1/3]$，峰值 1。
- `sym=true, length=4`（`effective=4` 偶）：$w = [1/4, 3/4, 3/4, 1/4]$，峰值 $3/4 = (M-1)/M$。
- `sym=true, length=1`（`effective=1`）：走 `effective == 1` 分支，$w = [1.0]$。
- `sym=false, length=5`（`effective=6` 偶，输出仍长 5）：$w = [1/6, 3/6, 5/6, 5/6, 3/6]$。
- `sym=false, length=1`（`effective=2` 偶）：$w[0] = (2 \cdot 1 - 1)/2 = 0.5$，即 $[0.5]$。

### 4. GPU 调用链、host wrapper、kernel 启动与 device 计算逻辑

#### 4.1 调用链

GPU 路径分两层：host wrapper（API 包装层 §2.1）与 device kernel（§2.4），中间通过 host 调度层（§2.3）衔接。

```
triang_device<T>(length, out, symmetric)              // windows_typed.cpp:551-560
  ├─ if length < 0 || out.size() != length → throw    // L554-556
  ├─ DeviceArray<float> computed(out.size())          // L557
  ├─ if length != 0 →                                 // L558
  │      triang_fp32_compute_device<T>(length, computed, symmetric)
  └─ out = cuda_utils::finalize_fp64_device_storage(computed)  // L559
                                                            // host_output_finalize.h:39-44

triang_fp32_compute_device<T>(n, o, sym)              // windows_typed.cu:114
  ├─ static_assert(detail::is_simple_signal_input_v<T>)       // L114 首段
  ├─ if n < 1 || o.size() != size_t(n) → throw        // L114
  └─ cuda_utils::launch_1d_kernel(                    // kernel_launch.h:44-56
        windows_detail::triang_kernel<float>,         // 模板实参硬编码 float
        o.size(),
        n, o.data(),
        sym ? n : n + 1)                              // effective_count
```

#### 4.2 host wrapper 关键点

- `triang_device` 在 host 上分配 `DeviceArray<float> computed`（FP32 临时数组，长度 = `out.size()` = `length`）。
- `length != 0` 才调用 GPU 计算；`length == 0` 时 `computed` 为空，跳过 kernel，直接走 FP64 拓宽（结果仍是空 `DeviceArray<double>`）。
- `finalize_fp64_device_storage` 完成 `D2H → host FP32→FP64 拓宽 → H2D` 三步（`host_output_finalize.h:39-44`、`18-26`、`DeviceArray<double>::from_host`）。**所有 FP64 拓宽在 host 完成，没有任何 kernel 执行 FP64 算术**（注释见 `host_output_finalize.h:14-17`）。
- 默认 stream（`launch_1d_kernel` 内部传 `nullptr`）、默认 `block_size=256`。
- `out = ...` 把新生成的 `DeviceArray<double>` 赋值给 `out` 引用参数（移动或拷贝赋值，取决于 `DeviceArray` 实现）。

#### 4.3 kernel 启动

```cpp
// windows_typed.cu:114 (末段)
cuda_utils::launch_1d_kernel(
    windows_detail::triang_kernel<float>,
    o.size(),            // 元素数 = n
    n, o.data(),
    sym ? n : n + 1);
```

`launch_1d_kernel` 在 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/cuda_utils/kernel_launch.h:44-56`：

```cpp
template <typename Kernel, typename... Args>
inline void launch_1d_kernel(Kernel kernel, std::size_t elements, Args&&... args)
{
    launch_1d_kernel_with_config(kernel, elements, nullptr, 256, std::forward<Args>(args)...);
}
```

`launch_1d_kernel_with_config`（`kernel_launch.h:31-42`）：

- `make_1d_launch_config(elements, 256)`（`kernel_launch.h:21-29`）：`block = dim3(256)`，`grid = dim3(div_up(elements, 256))`，其中 `div_up(elements, 256) = (elements + 255) / 256`（`kernel_launch.h:10-14`）。
- `kernel<<<grid, block, 0, nullptr>>>(args...)`，`stream = nullptr` 即默认 stream，`shmem = 0`。
- `CUDA_KERNEL_CHECK()`（`kernel_launch.h:41`，宏在 `cuda_error.h` 中定义）检查 launch 与 kernel 执行错误。

**关键**：模板实参硬编码为 `float`：`triang_fp32_compute_device<T>` 的 5 种 `T` 实例化都会调用同一个 `triang_kernel<float>`，写入 `DeviceArray<float>`。`T` 仅用于 `static_assert` 与类型分派，不影响 kernel 内的数值。

#### 4.4 device kernel 逐行逻辑

`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:176-194`：

```cpp
template <typename T>
__global__ void triang_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);  // L179
    if (index >= output_count) return;                                           // L180
    if (effective_count == 1) {                                                  // L181
        output[index] = window_store<T>(1.0F);                                   // L182
        return;                                                                  // L183
    }
    const int midpoint = effective_count / 2;                                    // L185
    const int rank = index < midpoint ? index + 1 : effective_count - index;     // L186
    using Scalar = decltype(window_load(output[index]));                         // L187
    const Scalar value = (effective_count & 1)                                  // L188
        ? static_cast<Scalar>(2 * rank) /                                        // L189
            static_cast<Scalar>(effective_count + 1)                             // L190
        : static_cast<Scalar>(2 * rank - 1) /                                    // L191
            static_cast<Scalar>(effective_count);                                // L192
    output[index] = window_store<T>(value);                                      // L193
}
```

逐行要点：

- **L179 线程索引**：`index = blockIdx.x * blockDim.x + threadIdx.x`，全局一维线性索引。
- **L180 边界检查**：`index >= output_count` 立即返回，防止越界写。
- **L181-184 单点退化**：`effective_count == 1` 直接写 `1.0F`（经 `window_store<T>`），避免除零。CPU 在 `effective == 1` 时也走相同分支（§3.3）。
- **L185 midpoint**：`effective_count / 2`，整数除法（正数等价于 $\lfloor M/2 \rfloor$）。
- **L186 rank**：与 CPU 完全一致的镜像公式。
- **L187 Scalar 推导**：`decltype(window_load(output[index]))`。`output[index]` 类型为 `T`；`window_load(T)`（`windows_kernels.cuh:10-18`）走 `SimpleSignalTypePolicy<T>::load`（`simple_signal_typed.h:16-31`）返回 `float`。又因 §4.3 硬编码 `T = float`，`Scalar = float`。
- **L188-192 奇偶分支**：
  - `2 * rank`、`2 * rank - 1`、`effective_count + 1`、`effective_count` 都是 `int`，先按 `int` 计算再 `static_cast<Scalar>` 转 `float`。
  - 由于 `Scalar = float`，除法是 FP32 除法。
  - 与 Python `2.0 * n / (_ind.size() + 1.0)` 相比：cusignal_cpp 多一步 `int → float` 隐式转换，数学等价，但 `2 * rank` 在 `int` 域计算无精度损失，损失发生在 `static_cast<float>` 与 FP32 除法。
- **L193 store**：`window_store<float>(value)`（`windows_kernels.cuh:20-28`）走 `SimpleSignalTypePolicy<float>::store`（`simple_signal_typed.h:22-30`），`float` 不在 `is_integral_v` 分支，直接 `static_cast<float>(value)`，无 clamp。
- **没有显式同步**：launch 在默认 stream 上，`launch_1d_kernel` 不调用 `cudaStreamSynchronize`；同步隐含在 `finalize_fp64_device_storage` 的 `computed.to_host()` 中——`to_host` 调用 `cudaMemcpy` D2H 隐式同步默认 stream。

### 5. template、类型分发、显式实例化、内存布局及数据搬运

#### 5.1 模板参数

- `triang_typed_cpu<T>`、`triang_device<T>`、`triang_fp32_compute_device<T>` 都是 `template <class T>`，`T` 仅作为类型分派 tag。
- `triang_kernel<T>` 是 `template <typename T>`，但 host 调度层硬编码为 `triang_kernel<float>`，因此实际只有 `T = float` 的实例被启动。

#### 5.2 类型分发与显式实例化

5 种业务类型（`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/cuda_utils/simple_signal_typed.h:18-41`）：
- `float`
- `__half`
- `std::int32_t`
- `std::int16_t`
- `std::int8_t`

CPU 端实例化（`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:562-593`，宏 `INSTANTIATE_WINDOWS_CPU(T)`）：

```cpp
// L586-587
template std::vector<double> triang_typed_cpu<T>(int, bool);
template void triang_device<T>(int, DeviceArray<double>&, bool);
// 实例化（L589-593）：
// INSTANTIATE_WINDOWS_CPU(float);
// INSTANTIATE_WINDOWS_CPU(__half);
// INSTANTIATE_WINDOWS_CPU(std::int32_t);
// INSTANTIATE_WINDOWS_CPU(std::int16_t);
// INSTANTIATE_WINDOWS_CPU(std::int8_t);
```

GPU 端实例化（`1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:116-127`，宏 `INSTANTIATE(T)`）：

```cpp
// L125
template void triang_fp32_compute_device<T>(int, DeviceArray<float>&, bool);
// 实例化（L126）：INSTANTIATE(float); INSTANTIATE(__half);
// INSTANTIATE(std::int32_t); INSTANTIATE(std::int16_t); INSTANTIATE(std::int8_t);
```

`static_assert(detail::is_simple_signal_input_v<T>)` 在 `triang_fp32_compute_device` 函数体内（`windows_typed.cu:114` 首段），拒绝 FP64 与其他不支持的类型进入 GPU 计算（错误信息见 `simple_signal_typed.h:54-55` 的 `require_formal_gpu_business_input`）。

#### 5.3 内存布局

- `triang_typed_cpu` 输出：`std::vector<double> out(length)`，host 连续 FP64，长度 `length`。
- `triang_device` 输入/输出：`DeviceArray<double>& out`，长度必须等于 `length`。
- `triang_device` 内部：`DeviceArray<float> computed(out.size())`，device 连续 FP32，长度 `length`。
- `triang_kernel` 写入：`float* output`（`T = float`），对应 `computed.data()`。
- 一维线性布局，无 stride/padding/soa。

#### 5.4 数据搬运路径

1. `triang_device` 在 host 分配 `computed`（device 内存，FP32）。
2. kernel 在 device 上写满 `computed`（每个线程一个采样点）。
3. `finalize_fp64_device_storage(computed)`（`host_output_finalize.h:39-44`）：
   - `computed.to_host()` → `std::vector<float>`，隐式 `cudaMemcpy` D2H，同步默认 stream。
   - `finalize_fp64_on_host`（`host_output_finalize.h:18-26`）逐元素 `static_cast<double>`，FP32→FP64 拓宽。
   - `DeviceArray<double>::from_host(...)` 把 FP64 host vector 再 H2D 回 device。
4. `out = ...` 把新 `DeviceArray<double>` 赋给 `out` 引用参数。

**关键**：FP64 输出在 host 端拓宽，避免 GPU 执行 FP64 算术；这是 ZQ500 平台 FP64 限制下的工程选择（`host_output_finalize.h:14-17` 注释）。

### 6. grid、block、线程索引与每线程数据范围

- `block_size = 256`（`kernel_launch.h:21-29` 默认）。
- `grid_size = div_up(n, 256) = (n + 255) / 256`。
- 每线程一维索引：`index = blockIdx.x * blockDim.x + threadIdx.x`。
- **每线程负责一个采样点**：`output[index]`。线程与数据是 1:1 映射，无共享内存、无 warp 协作、无 reduction。
- 边界线程（`index >= n`）直接 `return`，不参与计算。
- 单点退化（`effective_count == 1`）：所有进入 kernel 的线程中只有 `index == 0` 写入（其他被 `index >= output_count` 挡掉）。这是 `length = 1, sym = true` 的退化情形。

### 7. 同步、临时缓冲区、FFT/平台库调用与错误处理

#### 7.1 同步

- 显式：无 `cudaStreamSynchronize`、`cudaDeviceSynchronize`。
- 隐式：`finalize_fp64_device_storage` 调用 `computed.to_host()`，`cudaMemcpy` (D2H) 隐式同步默认 stream。
- 默认 stream（`launch_1d_kernel` 传 `nullptr`）。

#### 7.2 临时缓冲区

- `triang_device` 内部 `DeviceArray<float> computed(out.size())`：函数本地 workspace，不跨调用复用，不暴露给调用者。
- `finalize_fp64_device_storage` 内部短暂分配 `std::vector<float>`（D2H 结果）与 `std::vector<double>`（拓宽结果）。

#### 7.3 FFT / 平台库调用

- **无 FFT**：`triang` 是直接时域公式，不涉及频谱计算。
- **无 thrust reduction**：与 `chebwin`（求最大值归一化）不同，`triang` 不需要 reduction。
- **无归一化**：`triang` 公式本身决定峰值，奇数 `M` 时峰值为 1，偶数 `M` 时峰值为 `(M-1)/M`。
- **无 `dlfft`、无 `curt`、无 cuBLAS**：纯自定义 kernel + host 拓宽。这与 `chebwin`（custom direct DFT）、`taylor` 等算子不同。

#### 7.4 错误处理

| 层次 | 错误条件 | 抛出 |
| --- | --- | --- |
| `triang_typed_cpu` | `length < 0` | `std::invalid_argument("triang: length must be nonnegative")`（`windows_typed.cpp:534`） |
| `triang_device` | `length < 0` 或 `out.size() != length` | `std::invalid_argument("triang_device: output size must equal requested length")`（`windows_typed.cpp:554-556`） |
| `triang_fp32_compute_device` | `n < 1` 或 `o.size() != size_t(n)` | `std::invalid_argument("triang_device: compute output size must equal positive n")`（`windows_typed.cu:114`） |
| kernel launch | CUDA 错误 | `CUDA_KERNEL_CHECK()`（`kernel_launch.h:41`，宏在 `cuda_error.h` 中定义） |

**一致性差异**：

- CPU 在 `length == 0` 时不抛（返回空 vector）。
- `triang_device` 在 `length == 0` 时也不抛（`out` 必须是空 `DeviceArray`，跳过 kernel 走拓宽返回空）。
- `triang_fp32_compute_device` 在 `n < 1`（含 `n == 0`）时抛 `std::invalid_argument`。这是 host wrapper与下层 compute 之间的接口差异：wrapper 用 `if length != 0` 把 `length == 0` 挡在 compute 之外，避免触发 compute 的"`n<1`"错误。
- CPU 没有等价的"output size 不匹配"检查（CPU 直接构造 `out(length)`，无外部 out 参数）。

### 8. 数学/物理原理 → CPU 代码 → GPU 代码 三方映射表

原理编号沿用 `operator_comprehension/Learning/operators/windows/triang/数学物理原理.md`（阶段一）。Python 落实位置沿用 `operator_comprehension/Learning/operators/windows/triang/python源码算法.md`（阶段二，基准路径 `cusignal/windows/windows.py`）。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 原理 4：离散三角窗的归一化变体（triang 公式，奇偶分支） | `1ccd32ea` 之前的 cusignal 基准 `cusignal/windows/windows.py:206-209`（kernel body `if (odd) { w = 2.0 * n / (_ind.size() + 1.0); } else { w = (2.0 * n - 1.0) / _ind.size(); }`） | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:544-546`：`out[index] = (effective & 1) ? 2.0 * rank / (effective + 1) : (2.0 * rank - 1.0) / effective;` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:188-192`：`value = (effective_count & 1) ? static_cast<Scalar>(2*rank)/static_cast<Scalar>(effective_count+1) : static_cast<Scalar>(2*rank-1)/static_cast<Scalar>(effective_count);` | 三者公式一致；CPU 用 FP64，GPU 用 FP32，Python 用 FP64。GPU 在 kernel 内整数计算 `2*rank` 再转 float，与 Python 直接写 `2.0 * n` 等价（`int → float` 隐式转换）。 |
| 原理 4：rank 镜像（左半递增、右半递减） | `cusignal/windows/windows.py:200-204`（基准）：`if (i < m) { n = i + 1; } else { n = _ind.size() - i; }` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:542-543`：`midpoint = effective / 2; rank = index < midpoint ? index + 1 : effective - index;` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:185-186`：`midpoint = effective_count / 2; rank = index < midpoint ? index + 1 : effective_count - index;` | 三者数学等价。Python 用 `loop_prep` 预计算 `m`/`odd` 传入 kernel；C++ 在 kernel/CPU 内每点重算（`effective_count` 是 kernel 参数，对整个 grid 不变，推测编译器常量传播）。 |
| 原理 4：奇偶判断 | `cusignal/windows/windows.py:214-215`（基准 `loop_prep` 中 `const bool odd { _ind.size() & 1 }`） | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:544`：`(effective & 1)` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:188`：`(effective_count & 1)` | 一致，均用最低位 LSB 判奇偶。 |
| 原理 4：symmetric/periodic 切换（effective = sym ? M : M+1） | `cusignal/windows/windows.py:230`（基准 `M, needs_trunc = _extend(M, sym)`，`_extend` 内部按 sym 决定是否加 1） | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:535`：`effective = symmetric ? length : length + 1;` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:114`：`sym ? n : n + 1`（作为 `effective_count` 传入 kernel） | 三者数学等价。Python 用 `_extend`/`_truncate` 二步（多算一个采样再截断），C++ 直接用 `effective` 公式无显式截断。CPU 输出长度始终 = `length`，与 Python `_truncate` 后等长。 |
| 原理 4：单点退化（M=1 时返回 [1.0]） | `cusignal/windows/windows.py:228-229`（基准 `if _len_guards(M): return cp.ones(M)`，`M=1` 返回 `[1.0]`） | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:538-540`：`if (effective == 1) { out[index] = 1.0; continue; }` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:181-184`：`if (effective_count == 1) { output[index] = window_store<T>(1.0F); return; }` | `sym=true, M=1` 三者语义一致：返回 `[1.0]`。**`sym=false, M=1` 存在差异**：Python `_len_guards` 短路返回 `[1.0]`；cusignal_cpp CPU 与 GPU 都走 `effective=2` 偶分支，返回 `[0.5]`。见 §9.3、§11.4。 |
| 原理 4：长度 0（空窗） | `cusignal/windows/windows.py:228-229`（`M=0` 走 `_len_guards` 返回 `cp.ones(0)` 空） | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:536`（`length == 0` 时 `out` 空，循环不进入，返回空） | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:557-558`（`length == 0` 时 `computed` 空，跳过 kernel，`finalize` 返回空 `DeviceArray`） | 三者语义一致：返回空。 |
| 原理 4：长度负数非法 | Python `_len_guards`（`cusignal/windows/windows.py:20` 附近）做 `M < 0` 检查并抛 `ValueError` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:534`：`length < 0` 抛 `std::invalid_argument` | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:554-556`：`length < 0` 抛 `std::invalid_argument` | 三者都拒绝负长度。cusignal_cpp 不检查 `M` 是否为整数（`int` 参数天然保证）。 |
| 原理 4：不做归一化 | Python `triang` 不调用归一化 helper | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:531-549` 无归一化步骤 | `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:176-194` 无归一化步骤 | 三者一致，三角窗公式自身决定峰值。 |
| 原理 3：sinc² 频谱（三角窗的频域特性） | Python `triang` 只生成时域窗，不显式计算频谱 | C++ 不计算频谱 | C++ 不计算频谱 | 频域特性是分析性质，不在算子内显式实现；使用者自行做 FFT。 |
| 原理 2：三角函数作为两个矩形函数的卷积 | Python 不通过卷积构造 | C++ 不通过卷积构造 | C++ 不通过卷积构造 | 三者都用解析公式直接采样，不实现卷积构造路径。 |
| 原理 1：连续三角函数的时域定义与性质 | Python 公式是连续定义的离散采样 | C++ 公式是连续定义的离散采样 | C++ 公式是连续定义的离散采样 | 三者一致，均为离散解析公式。 |
| 原理 5：窗函数加窗原理（时域乘法 ↔ 频域卷积） | `triang` 只生成窗，不执行加窗 | C++ 不执行加窗 | C++ 不执行加窗 | 加窗是下游算子职责；`triang` 仅提供窗向量。 |

### 9. 与 cuSignal Python 实现相同、等价替换和有意不同的部分

#### 9.1 完全相同（数学等价）

1. **核心公式**：奇偶分支 `2*rank/(M+1)` 与 `(2*rank-1)/M` 三方逐字一致（仅 dtype 差异）。
2. **rank 镜像规则**：`if (i < m) n = i+1; else n = M-i` 三方一致。
3. **奇偶判断**：`& 1` 三方一致。
4. **symmetric/periodic 切换**：三方都用 `sym ? M : M+1` 的 effective 长度（Python 通过 `_extend`/`_truncate`，C++ 通过直接公式）。
5. **单点退化（`sym=true, M=1`）、空窗（`M=0`）、负长度拒绝**：三方语义一致。
6. **不归一化、不做 FFT、不卷积**：三方一致。

#### 9.2 等价替换（实现方式不同但结果相同）

1. **`loop_prep` vs kernel 内计算**：Python 用 `loop_prep` 字符串预计算 `m` 和 `odd`（`cusignal/windows/windows.py:214-215` 基准），通过 CuPy ElementwiseKernel 内嵌变量传入；cusignal_cpp 在 kernel 内每点重算 `midpoint = effective_count / 2` 和 `effective_count & 1`。**数学等价**，cusignal_cpp 依赖编译器常量传播（`effective_count` 是 kernel 参数，对整个 grid 不变，推测编译器会常量传播）。
2. **`_extend`/`_truncate` vs 直接 effective**：Python 调 `_extend(M, sym)` 返回 `(M, False)` 或 `(M+1, True)`，再调 `_truncate(w, needs_trunc)` 截断回 M；cusignal_cpp 直接用 `effective = sym ? length : length + 1` 公式，输出长度始终 = `length`，无显式截断步骤。**数学等价**。
3. **`_len_guards` vs 直接 throw**：Python `_len_guards`（`cusignal/windows/windows.py` 附近）检查 `int(M) != M or M < 0`；cusignal_cpp 只检查 `length < 0`。等价（`int` 参数天然是整数）。
4. **CPU `length == 0` 返回空 vector**：与 Python `cp.ones(0)` 空数组等价。
5. **GPU `length == 0` 跳过 kernel**：与 Python `M <= 1` 短路在 `M == 0` 时等价；`M == 1` 时 cusignal_cpp GPU 仍走 kernel 但 `effective_count == 1` 分支返回 1.0，与 Python `cp.ones(1) == [1.0]` 在 `sym=true` 下等价（`sym=false` 不等价，见 §9.3）。

#### 9.3 有意不同

1. **dtype 策略**：Python 全程 FP64；cusignal_cpp **GPU 用 FP32 计算 + FP64 输出**。这是 ZQ500 平台 FP64 算术受限的工程决策（见 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/cuda_utils/host_output_finalize.h:14-17` 注释）。CPU 仍用 FP64。精度差异在 §10 评估。
2. **kernel 模板实参硬编码 `float`**：`triang_fp32_compute_device<T>` 的 5 种 `T` 实例化都调用同一个 `triang_kernel<float>`。**`T` 不影响实际数值**，仅用于类型分派与 `static_assert`。与 Python（无模板）不同，但与 cusignal_cpp 其他 windows 算子（`chebwin`、`kaiser` 等接收 `T` 类型输入参数）的统一类型分发风格一致——`triang` 没有需要 `T` 类型的输入参数，`T` 在这里是冗余 tag。
3. **`sym=false, M=1` 单点退化不一致**：Python `_len_guards` 短路对 `M <= 1` 一律返回 `cp.ones(M)`，无视 `sym`，即 `sym=false, M=1` 返回 `[1.0]`；cusignal_cpp CPU 与 GPU 都只在 `effective == 1`（即 `sym=true, M=1`）时短路，`sym=false, M=1` 走 `effective=2` 偶分支，返回 `[0.5]`。**这是一个真实的语义差异**，但 cusignal_cpp 测试不覆盖此情形（§11.4 第 2 项），无法判定是 bug 还是有意。推测：cusignal_cpp 选择"不短路、让公式自然处理"，与 Python 上游 `_len_guards` 行为不同；下游使用者若依赖 `sym=false, M=1` 返回 `[1.0]` 会在 cusignal_cpp 上得到 `[0.5]`。
4. **整数类型存储时的 clamp 路径不触发**：`window_store<int32/int16/int8>` 会 clamp 到 `[lowest, max]` 再 cast（`simple_signal_typed.h:22-30`）。但 `triang_kernel` 模板实参硬编码 `float`，`output` 是 `float*`，所以 store 路径是 `window_store<float>`，无 clamp。**`T=int32/int16/int8` 时仍按 float 写入 `computed`**，再由 `finalize` 拓宽到 double。这与 `chebwin` 等"接收 `T` 类型输入"的算子不同。
5. **没有 `loop_prep` 字符串机制**：cusignal_cpp 不使用 CuPy ElementwiseKernel，所有逻辑写在原生 CUDA kernel。
6. **测试覆盖差异**：cusignal_cpp 测试 **不覆盖 `sym=false`（周期窗）**（见 §11.4），Python 阶段未限制但 cuSignal 上游 `triang` 的默认 `sym=True`。

### 10. 精度、dtype、边界条件及 ZQ500 平台限制

#### 10.1 精度

- CPU：FP64（`std::vector<double>`、`2.0 * rank` 浮点字面量）。
- GPU：FP32 计算（kernel 内 `Scalar = float`），FP64 输出（host 拓宽）。
- 精度损失来源：kernel 内 `2 * rank` 用 `int` 计算（无精度损失），转 `float` 后做 FP32 除法（相对 rounding 误差 $\approx 2^{-24} \approx 6 \times 10^{-8}$）；FP32→FP64 拓宽不引入新的 rounding。
- 测试容差 `matches_fp64` 用 `4e-2`（`1ccd32ea` / `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` 内的容差，`triang` 与其他 window 算子共用）。$M=5$ 时奇数公式 $w = 2 \cdot 3 / 6 = 1.0$，FP32 与 FP64 都精确表示；$M=4$ 时偶数公式 $w = (2 \cdot 2 - 1) / 4 = 0.75$，FP32 也精确表示。常见小 `M` 下 FP32 与 FP64 几乎相等，`4e-2` 容差足够宽松。

#### 10.2 dtype

- 业务输入 `T`：`float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t` 五种（`is_simple_signal_input_v`）。
- GPU 计算 dtype：硬编码 `float`，与 `T` 解耦。
- 输出 dtype：`double`（FP64），由 `finalize_fp64_device_storage` 拓宽。
- 不允许 FP64 作为 `T`（`static_assert` 会失败，`simple_signal_typed.h:54-55` 的 `require_formal_gpu_business_input` 错误信息明确）。

#### 10.3 边界条件

| 条件 | CPU 行为 | GPU 行为 | 与 Python 对比 |
| --- | --- | --- | --- |
| `length < 0` | 抛 `std::invalid_argument` | 抛 `std::invalid_argument` | 一致 |
| `length == 0` | 返回空 `vector<double>` | 返回空 `DeviceArray<double>`（跳过 kernel） | 一致 |
| `length == 1, sym=true` | `effective == 1` 分支返回 `[1.0]` | kernel `effective_count == 1` 分支写 `1.0F` | 一致 |
| `length == 1, sym=false` | `effective = 2`，走偶分支：`rank = (0<1)?1:2-0=1`，$w = (2 \cdot 1 - 1)/2 = 0.5$；返回 `[0.5]` | 同 CPU（`effective_count = 2`），返回 `[0.5]` | **不一致**：Python `_len_guards` 短路返回 `[1.0]` |
| `sym=false, length=0` | 不进入（`length==0` 短路） | 不进入（`length==0` 短路） | 一致（都返回空） |
| `out.size() != length`（GPU） | N/A | 抛 `std::invalid_argument` | Python 无等价检查 |
| `sym=false, length=5` | `effective=6`，$w = [1/6, 3/6, 5/6, 5/6, 3/6]$ | 同 CPU | 等价（Python 通过 `_extend` 给出相同 effective） |

**注意**：`length == 1, sym=false` 是易忽略的边界，cusignal_cpp 测试不覆盖（§11.4 第 2 项）。

#### 10.4 ZQ500 平台限制

- **FP64 算术受限**：cusignal_cpp 全局策略是"FP64 仅作接口/存储 dtype，所有 FP64 拓宽在 host 边界完成"（`host_output_finalize.h:14-17`）。`triang` 遵守此策略：device kernel 只做 FP32。
- **默认 stream**：与 ZQ500 SDK 样例风格一致，不显式创建 stream。
- **无 `curt`/`dlfft` 依赖**：`triang` 是纯算术 kernel，不依赖 ZQ500 平台科学库。这与 `chebwin`（custom direct DFT）等算子不同。
- **block_size 256**：cusignal_cpp 全局默认，与 ZQ500 SDK 样例一致（`docs/development/ZQ500_RUNTIME.md`）。
- **CMake 最低版本 3.16**：cusignal_cpp CMake 文件保持 3.16 兼容（AGENTS.md 要求），不因容器实测 3.22.1 提高最低版本。

### 11. 测试如何验证正确性，以及仍未覆盖的风险

#### 11.1 测试位置与入口

- 测试文件：`1ccd32ea` / `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`
- 测试段：`run_type<T>` 模板函数内，行号 `719-741`，对应 `ok[13]`（`labels[13] == "triang"`，见 `743-745`）。
- 类型分发：`run_type<T>` 对 5 种 dtype（`float`/`__half`/`int32`/`int16`/`int8`）各跑一遍。
- 容差：`matches_fp64` 用 `4e-2`（与其他 window 算子共用）。

#### 11.2 测试场景

`1ccd32ea` / `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:719-741`：

```cpp
test_evidence::f_begin_accuracy(type_dispatch::OperatorId::triang);
triang_device<T>(5, fixed_fp64_window_out);                    // 对称奇数
const auto triang_actual = fixed_fp64_window_out.to_host();
const auto triang_reference = triang_typed_cpu<T>(5);
DeviceArray<double> triang_alternate_out(4);
triang_device<T>(4, triang_alternate_out);                     // 对称偶数
const bool triang_alternate_matches = matches_fp64(
    triang_alternate_out.to_host(), triang_typed_cpu<T>(4));
triang_device<T>(1, singleton_fp64_window_out);                 // 单点
const bool triang_singleton_is_one =
    singleton_fp64_window_out.to_host() == std::vector<double>{1.0};
triang_device<T>(0, empty_fp64_window_out);                    // 空窗
bool triang_device_rejects_negative = false;
bool triang_cpu_rejects_negative = false;
try { triang_device<T>(-1, empty_fp64_window_out); }            // GPU 拒负长度
catch (const std::invalid_argument&) { triang_device_rejects_negative = true; }
try { (void)triang_typed_cpu<T>(-1); }                          // CPU 拒负长度
catch (const std::invalid_argument&) { triang_cpu_rejects_negative = true; }
ok[13] = matches_fp64(triang_actual, triang_reference) &&
    triang_alternate_matches &&
    triang_singleton_is_one && triang_typed_cpu<T>(1) == std::vector<double>{1.0} &&
    empty_fp64_window_out.empty() && triang_typed_cpu<T>(0).empty() &&
    triang_device_rejects_negative && triang_cpu_rejects_negative;
```

#### 11.3 验证维度

| 场景 | 长度 | sym | 期望 | 验证方式 |
| --- | --- | --- | --- | --- |
| 对称奇数 | 5 | true（默认） | 与 `triang_typed_cpu<T>(5)` 一致 | `matches_fp64(actual, reference)`，容差 4e-2 |
| 对称偶数 | 4 | true（默认） | 与 `triang_typed_cpu<T>(4)` 一致 | `matches_fp64(actual, reference)` |
| 单点 | 1 | true（默认） | `[1.0]` | `singleton_fp64_window_out.to_host() == std::vector<double>{1.0}`（精确相等） |
| 空窗 | 0 | true（默认） | 空 | `empty_fp64_window_out.empty()` |
| 负长度（GPU） | -1 | — | 抛 `std::invalid_argument` | try/catch 设置 `triang_device_rejects_negative` |
| 负长度（CPU） | -1 | — | 抛 `std::invalid_argument` | try/catch 设置 `triang_cpu_rejects_negative` |
| CPU 单点 | 1 | — | `[1.0]` | `triang_typed_cpu<T>(1) == std::vector<double>{1.0}` |
| CPU 空窗 | 0 | — | 空 | `triang_typed_cpu<T>(0).empty()` |

`ok[13]` 当且仅当上述所有断言为真才置 true。注意：`matches_fp64` 的"CPU 对照 GPU"实际上验证的是 cusignal_cpp 内部 CPU 与 GPU 的一致性，**不是直接对照 cuSignal Python 的数值**——Python 对照依赖阶段二的逐行学习，cusignal_cpp 不在运行时调用 Python。

#### 11.4 仍未覆盖的风险

1. **`sym=false`（周期窗）未测**：所有 `triang_device<T>(...)` 调用都用默认 `sym=true`。`sym=false` 路径（`effective = length + 1` 分支）从未执行。这是**测试覆盖漏洞**：如果 `triang_fp32_compute_device` 的 `sym ? n : n + 1` 写反、或 CPU 的 `length + 1` 错算，测试无法捕获。
   - **建议**：补充 `triang_device<T>(5, out, false)` 与 `triang_typed_cpu<T>(5, false)` 的对照测试。此时 `effective=6`（偶），输出 $w = [1/6, 1/2, 5/6, 5/6, 1/2]$（i=0..4，rank={1,2,3,3,2}）。
2. **`length == 1, sym=false` 未测**：cusignal_cpp 此时返回 `[0.5]`，而 Python `_len_guards` 短路返回 `[1.0]`（§9.3 第 3 项）。测试不覆盖此差异，无法判定 cusignal_cpp 的行为是有意还是 bug。建议补充测试明确语义。
3. **大 `M` 精度未测**：测试只覆盖 `M ∈ {0,1,4,5,-1}`。大 `M`（如 1024、65536）下的 FP32 rounding 累积未验证。`4e-2` 容差足够宽，`triang` 公式的 FP32/FP64 差异在大 `M` 下接近 $2^{-24} \approx 6 \times 10^{-8}$，仍远小于 4e-2，推测通过；但测试未实证。
4. **整型 dtype 的语义未独立验证**：5 种 `T` 都跑，但由于 kernel 模板实参硬编码 `float`，5 种 `T` 的 GPU 输出完全相同。测试没有独立验证 `T=int32` 时是否会产生与 `T=float` 不同的结果（按当前实现不会，但如果未来改 kernel 让 `T` 真正参与计算，测试不会捕获回归）。
5. **`out.size() != length` 未测**：`triang_device` 的 `out.size() != length` 抛错路径未直接测试（测试总是预先分配正确大小的 `out`）。
6. **`triang_fp32_compute_device` 的 `n < 1` 抛错未直接测**：host wrapper 用 `if length != 0` 挡住 `n=0`，因此 `triang_fp32_compute_device` 的 `n < 1` 路径只在 `n < 0` 时通过 `triang_device` 间接覆盖（但 `triang_device` 在 `length < 0` 时先抛，不会调到 compute 层）。**这条 compute 层错误路径实际从未被测试触发**。
7. **kernel launch 失败路径未测**：`CUDA_KERNEL_CHECK` 在 GPU OOM 或 driver 错误时抛，测试不覆盖。
8. **多 stream / 异步未测**：cusignal_cpp 用默认 stream，若调用者在自定义 stream 上调用（当前 API 不支持），未验证。
9. **`__half` dtype 的 store 路径**：由于 kernel 硬编码 `float`，`__half` 的 `window_store<__half>` 路径在 `triang` 中**不会被走到**（kernel 实参是 `float*`，不是 `__half*`）。这与 `chebwin` 等"接收 `__half` 输入"算子不同；测试即便跑 `T=__half` 也只是验证类型分派能实例化，不验证 `__half` 实际存储语义。
10. **cusignal_cpp 不直接对照 cuSignal Python 数值**：测试只做"CPU 自洽 + GPU 与 CPU 一致"，依赖阶段二（`python源码算法.md`）的逐行学习保证 cusignal_cpp 公式与 Python 公式逐字一致。若阶段二发现 Python 公式有 cusignal_cpp 未复制的边界行为（如 `sym=false, M=1` 的短路差异），测试无法捕获。

### 12. 自检问题

围绕"接口 → CPU → GPU wrapper → kernel → 测试"阅读顺序，建议自检：

1. **接口**：`triang_device<T>` 的 `out` 参数为什么是 `DeviceArray<double>&` 而不是 `DeviceArray<T>&`？答案见 §5.2、§10.2（FP64 是接口/存储 dtype，`T` 是业务输入 tag）。
2. **CPU**：`triang_typed_cpu<T>` 的 `T` 在函数体内是否被使用？答案：否，仅用于显式实例化分派（§3.2、§5.2）。
3. **CPU**：`effective == 1` 分支什么时候触发？答案：`sym=true, length=1`。`sym=false, length=0` 被 `length==0` 短路，实际不进入循环（§3.3、§10.3）。
4. **CPU**：`sym=false, length=1` 返回什么？答案：`[0.5]`，走偶分支 `effective=2`（§3.3、§10.3）。这与 Python `[1.0]` 不一致（§9.3 第 3 项）。
5. **GPU wrapper**：为什么 `triang_device` 在 `length == 0` 时跳过 `triang_fp32_compute_device`？答案：避免触发 compute 层 `n < 1` 抛错（§7.4、§11.4 第 6 项）。
6. **GPU wrapper**：`finalize_fp64_device_storage` 内部隐含同步在哪里发生？答案：`computed.to_host()` 调用 `cudaMemcpy` D2H，隐式同步默认 stream（§4.4、§7.1）。
7. **kernel**：`triang_kernel<float>` 的 `T` 模板实参是 `float`，那么 `triang_fp32_compute_device<int32_t>` 调用的 kernel 与 `triang_fp32_compute_device<float>` 调用的是同一个吗？答案：是，因为 `triang_kernel<float>` 硬编码（§4.3、§9.3 第 2 项）。
8. **kernel**：`2 * rank` 是 `int` 还是 `float` 运算？答案：先 `int`，再 `static_cast<Scalar>(...)` 转 float，最后做 FP32 除法（§4.4）。
9. **kernel**：`midpoint` 和 `effective_count & 1` 是否每线程重算？是否会被常量传播？答案：每线程重算；`effective_count` 是 kernel 参数（对整个 grid 不变），推测编译器会常量传播（§4.4、§9.2 第 1 项）。
10. **测试**：为什么 5 种 `T` 都跑同一组用例？答案：cusignal_cpp 的类型分发要求 5 种 dtype 都实例化并跑过 smoke（§5.2、§11.1）。
11. **测试**：`sym=false` 路径有没有被测试？答案：没有，这是已知覆盖漏洞（§11.4 第 1 项）。
12. **测试**：`matches_fp64` 用 4e-2 容差，对 `triang` 是否过宽？答案：三角窗值在 [0,1]，FP32/FP64 差异约 1e-7，4e-2 远超所需；这是为兼容其他容差较宽的 window 算子（如 `chebwin`）共享容差（§10.1）。
13. **平台**：为什么 GPU 不直接做 FP64？答案：ZQ500 平台 FP64 算术受限，cusignal_cpp 全局策略要求 FP64 拓宽在 host 边界完成（§10.4、`host_output_finalize.h:14-17`）。

### 13. 阅读建议

按"接口 → CPU → GPU wrapper → kernel → 测试"顺序阅读：

1. **接口**：先看 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:259-273`，理解 `triang_typed_cpu<T>` 与 `triang_device<T>` 的签名、docstring 与对称/周期语义。
2. **CPU**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:531-549`，注意 `effective == 1` 单点退化、奇偶分支公式、`length == 0` 返回空。
3. **GPU wrapper**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:551-560`，理解 `computed`（FP32 临时数组）的分配、`length != 0` 跳过、`finalize_fp64_device_storage` 的 FP64 拓宽。
4. **GPU compute**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:114`，注意 `static_assert`、`n < 1` 抛错、`launch_1d_kernel` 的参数与 `sym ? n : n + 1` effective_count 传递。
5. **kernel launch**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/cuda_utils/kernel_launch.h:21-56`，理解 `block_size=256`、`div_up` grid 计算、默认 stream、`CUDA_KERNEL_CHECK`。
6. **device kernel**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:176-194`，逐行理解 `index`、边界检查、单点退化、`midpoint`、`rank`、奇偶分支、`Scalar` 推导、`window_store`。
7. **类型策略**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/cuda_utils/simple_signal_typed.h:16-44`，理解 `SimpleSignalTypePolicy<T>` 的 `load`/`store` 与 `is_simple_signal_input_v`。
8. **FP64 拓宽**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/src/cuda_utils/host_output_finalize.h:18-44`，理解 `finalize_fp64_on_host` 与 `finalize_fp64_device_storage` 的三步（D2H → 拓宽 → H2D）。
9. **测试**：读 `1ccd32ea` / `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:719-741`，对照 §11.2 与 §11.3 的场景表，注意 `sym=false` 未测的覆盖漏洞，以及 `length=1, sym=false` 与 Python 不一致的边界（§9.3 第 3 项）。

读完后建议回到 §8 三方映射表交叉核对：每条公式都对应到 Python（阶段二 `python源码算法.md`）、CPU（§3）、GPU（§4）三处的具体行号；§9 列出的"有意不同"清单（dtype 策略、kernel 模板硬编码、`sym=false, M=1` 单点差异、整型 clamp 不触发）应在阅读源码时逐项验证。

---

## V2：第一次优化版本（待追加）

> 当 `triang` 算子出现下一个已提交的优化版本时，在此追加。V2 必须独立记录完整 SHA、代码定位、CPU/GPU 调用链、数学映射和测试证据，且不覆盖 V1 的任何路径行号。V2 章节结构与 V1 一致（13 项 + 映射表）。

## V3：第二次优化版本（待追加）

> 同 V2 规则。

## 版本差异与原理不变量

> 待 V2 出现后填充。计划用表格比较 V1/V2/V3 在算法结构、复杂度、内存访问、并行方式、精度边界和性能变化上的差异，并指出哪些数学物理原理始终未变。

当前 V1 的原理不变量（来自 `operator_comprehension/Learning/operators/windows/triang/数学物理原理.md` 阶段一）：

| 原理编号 | 原理名称 | V1 落实方式 | 是否可能被未来优化改变 |
| --- | --- | --- | --- |
| 原理 1 | 连续三角函数的时域定义与性质 | V1 用离散解析公式直接采样，不构造连续函数 | 否（公式不变） |
| 原理 2 | 三角函数作为两个矩形函数的卷积 | V1 不通过卷积构造，直接用 rank 公式 | 否（公式不变） |
| 原理 3 | 三角窗的频域特性 — sinc² 频谱 | V1 不显式计算频谱，仅生成时域窗 | 否（频谱是分析性质） |
| 原理 4 | 离散三角窗的归一化变体 — Bartlett 窗与 triang 窗 | V1 实现 triang 变体（奇偶分支公式），不实现 Bartlett 变体（端点为零，分母 `M-1`） | 否（triang 公式不变；若新增 Bartlett 是新算子，不算 V1 优化） |
| 原理 5 | 窗函数加窗原理 — 时域乘法对应频域卷积 | V1 只生成窗，不执行加窗 | 否（加窗是下游算子职责） |

V1 内可能受未来优化影响的工程量（非原理性）：

- dtype 策略（FP32 计算 + FP64 输出）— 若 ZQ500 平台 FP64 算术成本可接受，可能改为 device 直接 FP64。
- kernel 模板实参硬编码 `float` — 若未来需要 `T` 真正参与计算（如 `__half` 存储优化），可能改为 `triang_kernel<T>`。
- 常量重算 vs `loop_prep` 预计算 — 若编译器常量传播不足，可能改用 `__constant__` 或 host 预计算。
- 默认 stream vs 多 stream — 若加入异步流水，可能改用自定义 stream。
- block_size 256 — 若 occupancy 不足，可能改用 occupancy API。