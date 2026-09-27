# cusignal_cpp_general_cosine复现逻辑

本文件是 `general_cosine` 算子三阶段学习的第三阶段文档，记录 `cusignal_cpp` 中 CPU/GPU 复现版本的完整调用链、数学映射与工程决策。前置文档为同目录下的 `数学物理原理.md`（阶段一）与 `python源码算法.md`（阶段二）。本阶段所有代码定位格式为「完整提交 SHA + `ZKX_dev/` 相对路径 + 符号名 + 行号」。

## 版本索引

| 版本 | 短 SHA | 状态 | 说明 |
| --- | --- | --- | --- |
| V1 | `1ccd32ea` | 当前实现 | 原始学习版本，详见下文 V1 章节 |
| V2 | — | 占位 | 尚无优化版本；后续优化提交后追加，不覆盖 V1 |
| V3 | — | 占位 | 尚无优化版本；后续优化提交后追加 |

各版本独立记录完整 SHA、代码定位、CPU/GPU 调用链、数学映射与测试证据；旧版本章节保持可读，不因当前代码更新而修改旧路径行号。版本间差异见文末「版本差异与原理不变量」。

---

## V1：原始学习版本（`1ccd32ea`）

### 1. Git 版本信息

| 项 | 值 |
| --- | --- |
| 完整 SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 短 SHA | `1ccd32ea` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支 | `snapshot/all-current-20260716` |
| `cusignal_cpp` dirty 状态 | 干净 |

本阶段引用的相关文件（均位于 `ZKX_dev/` 下，本提交中无未提交改动）：

- `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h`
- `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp`
- `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu`
- `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh`
- `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注"无直接原理对应"。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在"含义解释"列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §1 `windows_typed.h:53-55`（`general_cosine_typed_cpu` 模板参数） | `template <class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，决定 `coefficients` 元素 dtype；显式实例化为 `float`/`__half`/`std::int32_t`/`std::int16_t`/`std::int8_t` 五种；输出固定 `double`（CPU）/`DeviceArray<double>`（GPU） |
| `length` | §1 `windows_typed.h:53-55`（`general_cosine_typed_cpu` 首参数） | `int` | 原理 1 §1 中的 $M$ | 窗长度（点数），约束 $M \geq 0$；GPU 侧以 `int n` 形式出现于 `general_cosine_device` |
| `coefficients` | §1 `windows_typed.h:53-55`（`general_cosine_typed_cpu` 第二参数） | `const std::vector<T>&`（CPU）/`const DeviceArray<T>&`（GPU） | 原理 1 §1 中的 $a_k$ 序列 | 余弦项权重系数数组，长度 $K$ 决定窗的阶数；GPU 侧以 `const Input* coefficients` 形式作为 kernel 参数 |
| `symmetric` | §1 `windows_typed.h:53-55`（`general_cosine_typed_cpu` 第三参数） | `bool` | 原理 5 §1（对称/周期窗约定） | 工程旗标，`true` 对称窗（`effective = length`），`false` 周期窗（`effective = length+1` 再截断）；GPU 侧以 `sym` 形式出现 |
| `out` | §1 `windows_typed.h:78-81`（`general_cosine_device` 第三参数） | `DeviceArray<double>& out` | 原理 1 §1 中的 $w[n]$ | GPU 输出窗序列，FP64 存储但 device 计算 FP32，经 `finalize_fp64_device_storage` 拓宽；CPU 侧以 `std::vector<double> out` 形式出现 |
| `coefficient_shape` | §1 `windows_typed.h:84-88`（`general_cosine_device` shape 重载第三参数） | `const std::vector<int>&` | 无直接原理对应 | 工程变量，多维系数数组的逻辑形状；由 `general_cosine_coefficient_count` 验证 shape 元素积等于 `data_size`，并取 `shape.front()` 作为有效系数数 $K$ |
| `effective` | §3.1 `windows_typed.cpp:253`（`general_cosine_typed_cpu` 局部） | `int`，`symmetric ? length : length + 1` | 原理 5 §1（DFT-even 扩展） | 工程变量，对称中心化长度；GPU 侧以 `effective_count` 形式作为 kernel 参数 |
| `pi64` | §3.1 `windows_typed.cpp:255`（`general_cosine_typed_cpu` 局部常量） | `constexpr double pi64 = 3.141592653589793238462643383279502884` | 数学常数 $\pi$（FP64 精度） | 圆周率 FP64 字面量，CPU 全程使用；与文件级 `kPi`（FP32）不是同一常量 |
| `index` | §3.1 `windows_typed.cpp:256-264`（`general_cosine_typed_cpu` 循环变量） | `int` | 原理 1 §1 中的 $n$ | 工程变量，逐点输出索引，对应 `out[index]` 的位置；kernel 侧以 `const int index` 形式出现（`windows_kernels.cuh:66`） |
| `phase` | §3.1 `windows_typed.cpp:257`（`general_cosine_typed_cpu` 局部） | `double`，`-pi64 + 2.0 * pi64 * index / (effective - 1)` | 原理 1 §1 中的 $\phi_n = -\pi + \frac{2\pi n}{M-1}$ | 当前采样点的相位；kernel 侧以 `Scalar phase` 形式出现（`windows_kernels.cuh:74-75`） |
| `value` | §3.1 `windows_typed.cpp:258`（`general_cosine_typed_cpu` 局部） | `double`，累加 $\sum_k a_k \cos(k \cdot \text{phase})$ | 原理 1 §1 中的 $w[n]$ | 当前采样点的窗值，累加后写入 `out[index]`；kernel 侧以 `Scalar value` 形式出现（`windows_kernels.cuh:76`） |
| `order` | §3.1 `windows_typed.cpp:259-262`（`general_cosine_typed_cpu` 内层循环变量） | `int` | 原理 1 §1 中的 $k$ | 余弦项索引，$0 \leq k \leq K-1$；kernel 侧同名出现于 `general_cosine_kernel`（`windows_kernels.cuh:77`） |
| `general_cosine_coefficient_count` | §3.2 `windows_typed.cpp:269-288`（匿名命名空间函数） | 函数，参数 `(const std::vector<int>& shape, std::size_t data_size)` | 无直接原理对应 | 工程 shape 重载辅助函数，验证 shape 元素积等于 `data_size` 后返回 `shape.front()` 作为有效系数数 |
| `shape` | §3.2 `windows_typed.cpp:269`（`general_cosine_coefficient_count` 首参数） | `const std::vector<int>&` | 无直接原理对应 | 工程变量，系数数组的逻辑形状向量 |
| `data_size` | §3.2 `windows_typed.cpp:269`（`general_cosine_coefficient_count` 第二参数） | `std::size_t` | 无直接原理对应 | 工程变量，`coefficients.size()` 的实际数据长度 |
| `extent` | §3.2 `windows_typed.cpp:275-283`（`general_cosine_coefficient_count` 循环变量） | `int` | 无直接原理对应 | 工程变量，`shape` 中每个维度的大小 |
| `product` | §3.2 `windows_typed.cpp:275-283`（`general_cosine_coefficient_count` 局部） | `int`，累乘 `shape` 各维度 | 无直接原理对应 | 工程变量，shape 元素积，用于与 `data_size` 一致性校验 |
| `zero` | §3.2 `windows_typed.cpp:275-283`（`general_cosine_coefficient_count` 局部） | `bool`，shape 中是否存在 0 维度 | 无直接原理对应 | 工程变量，标记 shape 含 0 维度的特殊情况 |
| `count` | §3.2 `windows_typed.cpp:296-297`（shape 重载局部） | `std::size_t`，`general_cosine_coefficient_count(...)` 返回值 | 原理 1 §1 中的 $K$ | 工程变量，shape 重载下实际使用的系数项数；GPU shape 重载以 `int coefficient_count` 形式传给 kernel |
| `computed` | §4.1 `windows_typed.cpp:311`（`general_cosine_device` 局部） | `DeviceArray<float>`，长度 `out.size()` | 无直接原理对应 | 工程 FP32 临时缓冲区，承载 GPU kernel 的 FP32 计算结果，随后经 `finalize_fp64_device_storage` 拓宽为 FP64 |
| `n` | §4.3 `windows_typed.cu:73`（`general_cosine_fp32_compute_device` 首参数） | `int` | 原理 1 §1 中的 $M$ | 工程变量，GPU 侧窗长度；与 CPU `length` 同义 |
| `a` | §4.3 `windows_typed.cu:73`（`general_cosine_fp32_compute_device` 第二参数） | `const DeviceArray<T>&` | 原理 1 §1 中的 $a_k$ 序列 | 工程变量，GPU 侧系数数组；通过 `a.data()` 传给 kernel |
| `o` | §4.3 `windows_typed.cu:73`（`general_cosine_fp32_compute_device` 第三参数） | `DeviceArray<float>&` | 原理 1 §1 中的 $w[n]$（FP32 计算缓冲） | 工程变量，GPU FP32 输出缓冲区；通过 `o.data()` 传给 kernel 作为 `Output* output` |
| `sym` | §4.3 `windows_typed.cu:73`（`general_cosine_fp32_compute_device` 第四参数） | `bool` | 原理 5 §1（对称/周期窗约定） | 工程旗标，与 CPU `symmetric` 同义；用于计算 `effective_count = sym ? n : n + 1` |
| `coefficient_count` | §4.3 `windows_typed.cu:74`（shape 重载参数） | `int` | 原理 1 §1 中的 $K$ | 工程变量，kernel 内层循环上界，控制只读取前 `count` 个系数 |
| `effective_count` | §4.3 `windows_typed.cu:73,74`（kernel 第五参数） | `int`，`sym ? n : n + 1` | 原理 5 §1（DFT-even 扩展） | 工程变量，GPU 侧的对称中心化长度，与 CPU `effective` 同义 |
| `Input` | §4.4 `windows_kernels.cuh:58`（`general_cosine_kernel` 模板参数） | `typename Input`，显式实例化为 `T` | 无直接原理对应 | 工程 kernel 系数元素类型 |
| `Output` | §4.4 `windows_kernels.cuh:58`（`general_cosine_kernel` 模板参数） | `typename Output = Input`，显式实例化为 `float` | 无直接原理对应 | 工程 kernel 输出元素类型，固定 `float`（FP32 计算缓冲） |
| `output_count` | §4.4 `windows_kernels.cuh:60`（`general_cosine_kernel` 首参数） | `int` | 原理 1 §1 中的 $M$（输出长度） | 工程变量，输出元素数，用于线程越界检查 `if (index >= output_count) return` |
| `output` | §4.4 `windows_kernels.cuh:63`（`general_cosine_kernel` 第四参数） | `Output*` | 原理 1 §1 中的 $w[n]$ | kernel 输出指针，对应 `o.data()`，最终经 `finalize` 拓宽为 FP64 |
| `Scalar` | §4.4 `windows_kernels.cuh:72`（`general_cosine_kernel` 局部类型别名） | `using Scalar = decltype(window_load(coefficients[0]))`，对 5 种 `T` 得 `float` | 无直接原理对应 | 工程计算精度类型，决定 kernel 内中间量精度；FP32 路径下为 `float` |
| `pi` | §4.4 `windows_kernels.cuh:73`（`general_cosine_kernel` 局部常量） | `const Scalar pi = static_cast<Scalar>(3.14159265358979323846F)` | 数学常数 $\pi$（FP32 精度） | 圆周率 FP32 字面量，kernel 全程使用 |
| `host_load<T>` | §4.5 `windows_typed.cpp:17-21`（host helper 函数模板） | `template <class T> float host_load(T value)` | 无直接原理对应 | 工程类型转换函数，调用 `SimpleSignalTypePolicy<T>::load` 将 5 种 dtype 统一转 `float`；CPU 再 `(double)` 拓宽为 FP64 |
| `host_store<T>` | §4.5 `windows_typed.cpp:23-27`（host helper 函数模板） | `template <class T> void host_store(T& target, float value)` | 无直接原理对应 | 工程类型转换函数，调用 `SimpleSignalTypePolicy<T>::store` 把 `float` 转回 `T`（含整数饱和截断） |
| `window_load<T>` | §4.5 `windows_kernels.cuh:10-18`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::load`，5 种 dtype 统一转 `float` |
| `window_store<T, Scalar>` | §4.5 `windows_kernels.cuh:20-28`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::store`，把 FP32 `value` 转回 `T` |
| `SimpleSignalTypePolicy<T>` | §5.1 `cuda_utils/simple_signal_typed.h`（类型策略模板） | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断） |
| `is_simple_signal_input_v<T>` | §5.1 `cuda_utils/simple_signal_typed.h`（类型 trait） | `template <class T> constexpr bool is_simple_signal_input_v` | 无直接原理对应 | 工程类型 trait，编译期判断 `T` 是否属于 5 种简单信号类型，用于 `static_assert` |
| `launch_1d_kernel` | §6.1 `cuda_utils/kernel_launch.h`（kernel 启动工具） | 函数模板，参数 `(Kernel kernel, std::size_t elements, Args&&... args)` | 无直接原理对应 | 工程 kernel 启动工具，自动按 `elements` 计算 grid/block（默认 `block_size=256`，默认 stream） |
| `finalize_fp64_device_storage` | §4.1 `cuda_utils/host_output_finalize.h`（FP32→FP64 拓宽工具） | 函数，参数 `(DeviceArray<float> computed)` 返回 `DeviceArray<double>` | 无直接原理对应 | 工程 FP32→FP64 拓宽工具，内部走 D2H → host float→double → H2D |
| `kPi` | §2 `windows_typed.cpp:15`（文件级常量） | `constexpr float kPi = 3.14159265358979323846F` | 数学常数 $\pi$（FP32 精度） | 工程文件级 FP32 圆周率常量；`general_cosine` 主路径不直接使用（CPU 用局部 `pi64`，kernel 用局部 `pi`），列此供其他 windows 算子共用对照 |

---

### 2. 对外接口与代码定位

`cusignal_cpp` 的 `general_cosine` 按职责分为五层。下表使用「完整 SHA + `ZKX_dev/` 相对路径 + 符号名 + 行号」格式定位。

| 层 | 符号 | 声明/定义位置 |
| --- | --- | --- |
| API 包装层（主） | `general_cosine_device<T>` | 声明 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:78-81`；实现 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:303-319` |
| API 包装层（shape 重载） | `general_cosine_device<T>` | 声明 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:84-88`；实现 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:321-339` |
| CPU/reference 核心（主） | `general_cosine_typed_cpu<T>` | 声明 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:53-55`；实现 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:246-266` |
| CPU/reference 核心（shape 重载） | `general_cosine_typed_cpu<T>` | 声明 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.h:61-64`；实现 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:291-301` |
| CPU helper（匿名命名空间） | `general_cosine_coefficient_count` | 定义 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:268-289`（匿名 `namespace` 在 268 行开启、289 行闭合；函数体 269-288） |
| GPU host 调度层（主） | `general_cosine_fp32_compute_device<T>` | 前置声明 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:145-146`；实现 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:73` |
| GPU host 调度层（shape 重载） | `general_cosine_fp32_compute_device<T>` | 前置声明 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:148-149`；实现 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:74` |
| GPU device 核心 kernel | `windows_detail::general_cosine_kernel<Input, Output>` | 定义 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:58-82` |
| host/device 载入 helper | `windows_detail::window_load<T>` | 定义 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:10-18` |
| host/device 写出 helper | `windows_detail::window_store<T, Scalar>` | 定义 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:20-28` |
| host 载入 helper | `host_load<T>` | 定义 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:17-21` |
| host 写出 helper | `host_store<T>` | 定义 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:23-27` |
| 测试驱动 | `run_type<T>` 内 `ok[7]` | `1ccd32ea` `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:475-522` |

`windows_typed.cpp:15` 处的 `constexpr float kPi = 3.14159265358979323846F` 是文件级常量，但 `general_cosine_typed_cpu` 主重载在 `windows_typed.cpp:255` 处使用的是局部 `constexpr double pi64 = 3.141592653589793238462643383279502884`，与 `kPi` 不是同一常量。

### 3. CPU 调用链、数据结构、逐步算法

#### 3.1 CPU 主重载调用链

`general_cosine_typed_cpu<T>(length, coefficients, symmetric)` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:246-266`：

```
general_cosine_typed_cpu(length, coefficients, symmetric)        [windows_typed.cpp:246-266]
  ├─ if length < 0 → throw std::invalid_argument                 [250]
  ├─ if length == 0 → return {}                                  [251]
  ├─ if length == 1 → return {1.0}                               [252]
  ├─ effective = symmetric ? length : length + 1                 [253]
  ├─ std::vector<double> out(length)                             [254]
  ├─ constexpr double pi64 = 3.141592653589793238462643383279502884  [255]
  └─ for index = 0..length-1:                                    [256-264]
       ├─ phase = -pi64 + 2.0 * pi64 * index / (effective - 1)   [257]
       ├─ double value = 0.0                                     [258]
       ├─ for order = 0..coefficients.size()-1:                  [259-262]
       │    └─ value += (double)host_load(coefficients[order])   [260]
       │              * std::cos(order * phase)                  [261]
       └─ out[index] = value                                     [263]
  return out                                                     [265]
```

#### 3.2 CPU shape 重载调用链

`general_cosine_typed_cpu<T>(length, coefficients, coefficient_shape, symmetric)` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:291-301`：

```
general_cosine_typed_cpu(length, coefficients, coefficient_shape, symmetric)  [291-301]
  ├─ count = general_cosine_coefficient_count(coefficient_shape, coefficients.size())  [296-297]
  │    └─ 验证 shape 元素积 == data_size；返回 shape.front() 作为系数数目  [269-288]
  └─ return general_cosine_typed_cpu(                            [298-300]
       length,
       std::vector<T>(coefficients.begin(), coefficients.begin() + count),
       symmetric)
```

`general_cosine_coefficient_count` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:269-288` 逻辑：

```
general_cosine_coefficient_count(shape, data_size)               [269-288]
  ├─ if shape.empty() → throw                                    [272]
  ├─ product = 1; zero = false
  ├─ for extent in shape:                                        [275-283]
  │    ├─ if extent < 0 → throw                                  [276]
  │    ├─ if extent == 0 → zero = true                           [277-278]
  │    └─ elif !zero:
  │         ├─ if product > data_size / extent → throw           [280-281]
  │         └─ product *= extent                                 [282]
  ├─ if (zero ? 0 : product) != data_size → throw                [285-286]
  └─ return (std::size_t)shape.front()                           [287]
```

#### 3.3 主要数据结构

- `coefficients`：`const std::vector<T>&`，host 端一维系数数组，`T` 为五种简单信号类型之一。
- `out`：`std::vector<double>`，长度 `length`，dtype 固定 FP64。
- `host_load<T>`（`windows_typed.cpp:17-21`）：调用 `detail::SimpleSignalTypePolicy<T>::load(value)`，将 `T` 标量转为 `float`；CPU 路径再 `(double)` 显式拓宽为 FP64 参与累加。
- `effective`：`int`，对称窗为 `length`，周期窗为 `length + 1`，与阶段二 Python 的 `_extend(M, sym)` 在数学上等价（见第 9 节）。

#### 3.4 逐步算法（CPU 主重载）

1. **长度守卫**：`length < 0` 抛 `std::invalid_argument`；`length == 0` 返回空 `std::vector<double>`；`length == 1` 返回 `{1.0}`。注意 `length == 1` 时不读取系数，直接返回单位值。
2. **有效长度**：`effective = symmetric ? length : length + 1`。对称窗在 `length` 个点上定义相位；周期窗在 `length + 1` 个点上定义相位后只取前 `length` 个点（DFT-even 约定，对应原理 5）。
3. **逐点计算**：对 `index = 0..length-1`，计算 `phase = -π + 2π·index/(effective-1)`，再以 `value = Σ_{order=0}^{K-1} a[order]·cos(order·phase)` 累加，其中 `K = coefficients.size()`。
4. **累加精度**：CPU 用 `double` 累加，与阶段二 Python 的 `float64` 一致。

### 4. GPU 调用链、host wrapper、kernel 启动、device 计算逻辑

#### 4.1 GPU API 包装层（主）

`general_cosine_device<T>(length, coefficients, out, symmetric)` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:303-319`：

```
general_cosine_device(length, coefficients, out, symmetric)      [303-319]
  ├─ if length < 0 || out.size() != length → throw               [308-310]
  ├─ DeviceArray<float> computed(out.size())                     [311]
  ├─ if length == 1: computed = DeviceArray<float>::from_host({1.0F})  [312-313]
  ├─ elif length > 1:
  │    general_cosine_fp32_compute_device(length, coefficients, computed, symmetric)  [314-316]
  └─ out = cuda_utils::finalize_fp64_device_storage(computed)    [318]
```

`length == 0` 时不进入 `length == 1` 或 `length > 1` 分支，`computed` 保持空，`finalize_fp64_device_storage` 产出空 `DeviceArray<double>`。

#### 4.2 GPU API 包装层（shape 重载）

`general_cosine_device<T>(length, coefficients, coefficient_shape, out, symmetric)` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:321-339`：

```
general_cosine_device(length, coefficients, coefficient_shape, out, symmetric)  [321-339]
  ├─ count = general_cosine_coefficient_count(coefficient_shape, coefficients.size())  [327-328]
  ├─ if length < 0 || out.size() != length → throw               [329-330]
  ├─ DeviceArray<float> computed(out.size())                     [331]
  ├─ if length == 1: computed = {1.0F}                           [332-333]
  ├─ elif length > 1:
  │    general_cosine_fp32_compute_device(length, coefficients, (int)count, computed, symmetric)  [334-336]
  └─ out = cuda_utils::finalize_fp64_device_storage(computed)    [338]
```

shape 重载不重新构造 `coefficients`，而是把 `count` 作为 `coefficient_count` 参数传给 kernel，由 kernel 内循环上界控制只读取前 `count` 个系数。

#### 4.3 GPU host 调度层

`general_cosine_fp32_compute_device<T>(n, a, o, sym)` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:73`（单行实现）：

```cpp
template <class T> void general_cosine_fp32_compute_device(int n,const DeviceArray<T>&a,DeviceArray<float>&o,bool sym){
    static_assert(detail::is_simple_signal_input_v<T>);
    if(n<=1||o.size()!=size_t(n))
        throw std::invalid_argument("general_cosine_device: compute output size must equal n and n must exceed one");
    cuda_utils::launch_1d_kernel(
        windows_detail::general_cosine_kernel<T,float>,
        o.size(),            // grid 配置依据：输出元素数
        n,                   // kernel 参数 1: output_count
        a.data(),            // kernel 参数 2: coefficients 指针
        (int)a.size(),       // kernel 参数 3: coefficient_count
        o.data(),            // kernel 参数 4: output 指针
        sym?n:n+1);          // kernel 参数 5: effective_count
}
```

shape 重载 `general_cosine_fp32_compute_device<T>(n, a, coefficient_count, o, sym)` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:74`：

```cpp
template <class T> void general_cosine_fp32_compute_device(int n,const DeviceArray<T>&a,int coefficient_count,DeviceArray<float>&o,bool sym){
    static_assert(detail::is_simple_signal_input_v<T>);
    if(n<=1||coefficient_count<0||static_cast<std::size_t>(coefficient_count)>a.size()||o.size()!=size_t(n))
        throw std::invalid_argument("general_cosine_device: coefficient shape or output size mismatch");
    cuda_utils::launch_1d_kernel(
        windows_detail::general_cosine_kernel<T,float>,
        o.size(),            // grid 配置依据
        n,                   // output_count
        a.data(),            // coefficients
        coefficient_count,   // coefficient_count（来自 shape.front()）
        o.data(),            // output
        sym?n:n+1);          // effective_count
}
```

`static_assert(detail::is_simple_signal_input_v<T>)` 在编译期强制 `T` ∈ {`float`, `__half`, `std::int32_t`, `std::int16_t`, `std::int8_t`}。

#### 4.4 GPU device kernel 内部逻辑

`windows_detail::general_cosine_kernel<Input, Output>` —— `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:58-82`：

```cpp
template <typename Input, typename Output = Input>
__global__ void general_cosine_kernel(
    int output_count,
    const Input* coefficients,
    int coefficient_count,
    Output* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);  // [66]
    if (index >= output_count) return;                                          // [67]
    if (effective_count == 1) {                                                 // [68-71]
        output[index] = window_store<Output>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(coefficients[0]));                      // [72]
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);             // [73]
    const Scalar phase = -pi + static_cast<Scalar>(2) * pi *                    // [74-75]
        static_cast<Scalar>(index) / static_cast<Scalar>(effective_count - 1);
    Scalar value = Scalar{0};                                                   // [76]
    for (int order = 0; order < coefficient_count; ++order) {                   // [77-80]
        value += window_load(coefficients[order]) *
            cos(static_cast<Scalar>(order) * phase);
    }
    output[index] = window_store<Output>(value);                                // [81]
}
```

GPU 实际实例化为 `general_cosine_kernel<T, float>`，故 `Output = float`，`Scalar = decltype(window_load(coefficients[0]))`。当 `T` 是五种简单信号类型之一时，`SimpleSignalTypePolicy<T>::load` 返回 `float`，所以 `Scalar = float`，整个 kernel 在 FP32 下计算。

#### 4.5 host/device helper

- `window_load<T>`（`windows_kernels.cuh:10-18`）：`__host__ __device__`，若 `is_simple_signal_input_v<T>` 为真则调用 `SimpleSignalTypePolicy<T>::load(value)` 转 `float`，否则原样返回。本算子的五种 `T` 都走前一路径。
- `window_store<T, Scalar>`（`windows_kernels.cuh:20-28`）：`__host__ __device__`，若 `is_simple_signal_input_v<T>` 为真则 `SimpleSignalTypePolicy<T>::store(static_cast<float>(value))`，先转 `float` 再转 `T`；否则 `static_cast<T>(value)`。实例化 `window_store<float, float>` 时为恒等。
- `host_load<T>`（`windows_typed.cpp:17-21`）与 `host_store<T>`（`windows_typed.cpp:23-27`）：CPU 路径专用，调用同一 `SimpleSignalTypePolicy<T>`，但 CPU 主重载在 `windows_typed.cpp:260` 处再 `(double)` 显式拓宽。

### 5. template、类型分发、显式实例化、内存布局、数据搬运

#### 5.1 模板与类型分发

`general_cosine` 全链路模板参数为 `<class T>`，支持五种简单信号类型：`float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t`。类型限制在 GPU host 调度层由 `static_assert(detail::is_simple_signal_input_v<T>)` 强制（`windows_typed.cu:73,74`）。CPU 路径没有同等的 `static_assert`，但通过显式实例化（见 5.3）只编译这五种类型。

#### 5.2 显式实例化

- **CPU + API 包装层**：`INSTANTIATE_WINDOWS_CPU(T)` 宏定义于 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:562-587`，在 589-593 行对五种类型展开。其中与 `general_cosine` 相关的实例化项：
  - `template std::vector<double> general_cosine_typed_cpu(int, const std::vector<T>&, bool);`（565-566）
  - `template std::vector<double> general_cosine_typed_cpu(int, const std::vector<T>&, const std::vector<int>&, bool);`（567-568）
  - `template void general_cosine_device(int, const DeviceArray<T>&, DeviceArray<double>&, bool);`（569-570）
  - `template void general_cosine_device(int, const DeviceArray<T>&, const std::vector<int>&, DeviceArray<double>&, bool);`（571-572）
- **GPU host 调度层**：`INSTANTIATE(T)` 宏定义于 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:116-125`，在 126 行对五种类型展开。相关项：
  - `template void general_cosine_fp32_compute_device(int,const DeviceArray<T>&,DeviceArray<float>&,bool);`（118）
  - `template void general_cosine_fp32_compute_device(int,const DeviceArray<T>&,int,DeviceArray<float>&,bool);`（119）
- **GPU kernel**：`general_cosine_kernel<Input, Output>` 是 `__global__` 模板，在 `general_cosine_fp32_compute_device` 内通过 `windows_detail::general_cosine_kernel<T,float>` 显式调用，由编译器在 host 调度层实例化时一并生成 `<T, float>` 的 device 代码。

#### 5.3 内存布局与数据搬运

| 数据 | 位置 | dtype | 大小 | 生命周期 |
| --- | --- | --- | --- | --- |
| `coefficients`（CPU 路径） | host | `T` | `coefficients.size()` | 调用者持有，函数只读 |
| `coefficients`（GPU 路径） | device | `T` | `a.size()` | 调用者持有的 `DeviceArray<T>`，函数只读 |
| `out`（CPU 路径） | host | `double` | `length` | 返回值 `std::vector<double>` |
| `computed`（GPU 路径） | device | `float` | `out.size()` == `length` | 函数内部 `DeviceArray<float>` workspace |
| `out`（GPU 路径） | device | `double` | `length` | 调用者持有的 `DeviceArray<double>`，函数写入 |

数据搬运路径（GPU 主路径，`length > 1`）：

1. 调用者预先把 `coefficients` 放在 device（`DeviceArray<T>`），无 H2D。
2. 函数内分配 `DeviceArray<float> computed(out.size())`，纯 device workspace。
3. `general_cosine_fp32_compute_device` 启动 kernel，kernel 直接从 `a.data()` 读系数、写 `o.data()`，无中间 D2D。
4. `out = cuda_utils::finalize_fp64_device_storage(computed)` 完成 FP32→FP64 的 D2D 拓宽并写回调用者的 `out`。
5. `length == 1` 时不启动 kernel，`computed = DeviceArray<float>::from_host({1.0F})` 走 H2D 单点写入后再 `finalize`。
6. `length == 0` 时不启动 kernel，`computed` 保持空，`finalize` 产出空 `DeviceArray<double>`。

`finalize_fp64_device_storage` 的具体实现位于 `cuda_utils`（本阶段未读），其语义是把 `DeviceArray<float>` 的内容在 device 上拓宽为 `DeviceArray<double>`，避免 host 中转。

### 6. grid、block、线程索引、每线程数据范围

#### 6.1 grid/block 配置

`general_cosine_fp32_compute_device` 调用 `cuda_utils::launch_1d_kernel(kernel, o.size(), ...)`，其中第二个参数 `o.size()` 是输出元素数。`launch_1d_kernel` 的实现位于 `ZKX_dev/cusignal_cpp/src/cuda_utils/`（本阶段未读），按一维 1D 模型自动配置 grid 与 block 维度。推测其行为与标准 `<<<ceil(N/threadsPerBlock), threadsPerBlock>>>` 一致，但具体 `threadsPerBlock` 取值需阅读 `cuda_utils` 才能确认。本阶段只确认：grid/block 由项目工具自动配置，调用者与 host 调度层都不手动指定。

#### 6.2 线程索引

kernel 内 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:66`：

```cpp
const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
```

仅使用 `x` 维度，是一维线程网格。

#### 6.3 边界与每线程数据范围

- `if (index >= output_count) return;`（`windows_kernels.cuh:67`）：超出输出范围的线程立即返回。
- 每个活跃线程负责且仅负责 `output[index]` 一个输出点，读取全部 `coefficient_count` 个系数。线程间无依赖、无归约、无共享内存。

### 7. 同步、临时缓冲区、FFT/平台库调用与错误处理

#### 7.1 同步

`general_cosine_device` 与 `general_cosine_fp32_compute_device` 都使用默认 stream，无显式 `cudaStreamCreate` / `cudaStreamSynchronize`。`launch_1d_kernel` 内部是否同步由 `cuda_utils` 决定（本阶段未读）。`finalize_fp64_device_storage` 是 D2D 拷贝，同样在默认 stream 上。调用者若需要确保结果可见，应在 `general_cosine_device` 返回后自行同步。

#### 7.2 临时缓冲区

- `DeviceArray<float> computed(out.size())`：函数内部 workspace，仅在 `length > 1` 时实际使用；函数返回前由 `finalize_fp64_device_storage` 消费并释放。
- 不存在跨调用复用的 workspace、cache 或常量内存。

#### 7.3 FFT/平台库调用

`general_cosine` 不调用任何 FFT 接口、不调用 `curt`、不调用 thrust reduction。原因：通用余弦窗不强制峰值归一化（原理 4），所以无需像 `chebwin` 那样求最大值再做归一化。`windows_typed.cu` 顶部 `#include <thrust/extrema.h>` 是为 `chebwin` 引入的，`general_cosine` 路径不使用 thrust。

#### 7.4 错误处理

| 检查点 | 位置 | 触发条件 | 异常 |
| --- | --- | --- | --- |
| CPU 主重载长度 | `windows_typed.cpp:250` | `length < 0` | `std::invalid_argument` |
| CPU shape 重载 shape 一致性 | `windows_typed.cpp:272,276,280-281,285-286` | `shape.empty()`、`extent < 0`、`product` 溢出、`product != data_size` | `std::invalid_argument` |
| GPU API 主重载长度 | `windows_typed.cpp:308-310` | `length < 0` 或 `out.size() != length` | `std::invalid_argument` |
| GPU API shape 重载 | `windows_typed.cpp:329-330` | `length < 0` 或 `out.size() != length`（shape 在 327-328 已先验证） | `std::invalid_argument` |
| GPU host 调度主重载 | `windows_typed.cu:73` | `n <= 1` 或 `o.size() != n` | `std::invalid_argument` |
| GPU host 调度 shape 重载 | `windows_typed.cu:74` | `n <= 1`、`coefficient_count < 0`、`coefficient_count > a.size()` 或 `o.size() != n` | `std::invalid_argument` |

CUDA runtime 错误由 `launch_1d_kernel` 与 `DeviceArray` 的项目异常机制处理（本阶段未读具体实现）。

### 8. 数学/物理原理 → CPU 代码 → GPU 代码 三方映射表

阶段一 `数学物理原理.md` 编号了 8 条原理。下表按「原理或公式 → cuSignal Python 落实位置 → C++ CPU 落实位置 → C++ GPU 落实位置 → 差异说明」映射。Python 行号来自阶段二，对应 `cusignal-23.08.00/cusignal/windows/windows.py` 基准。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 原理 1：通用余弦窗时域定义 $w[n]=\sum_{k=0}^{K-1} a_k \cos(k\phi_n)$，$\phi_n=-\pi+\frac{2\pi n}{M-1}$ | `_general_cosine_kernel` body `fac = -M_PI + delta*i`（`windows.py:47`）、`for (k=0;k<n;k++) temp += a[k]*cos(k*fac)`（`windows.py:49-52`） | `windows_typed.cpp:257` `phase = -pi64 + 2.0*pi64*index/(effective-1)`、`windows_typed.cpp:259-262` 双层循环累加 | `windows_kernels.cuh:74-75` `phase = -pi + 2*pi*index/(effective_count-1)`、`windows_kernels.cuh:77-80` 循环累加 | 三者公式同构；CPU 用 `double`，GPU 用 `float`，Python 用 `float64`。Python 用 `loop_prep` 预计算 `delta`，C++ 在 kernel/循环体内直接算除法 |
| 原理 2：经典窗作为通用余弦窗特例（如 Hamming = `general_cosine(M, [0.54, 0.46]`)） | `general_cosine` 接受任意 `a`，由调用者决定特例 | `general_cosine_typed_cpu` 不特判系数，由 `coefficients` 决定特例 | `general_cosine_kernel` 不特判系数 | 三者一致；`cusignal_cpp` 在同文件另有独立 `hamming_*` 实现，不走 `general_cosine` 路径 |
| 原理 3：频域为多个移位 Dirichlet 核加权和 | Python 不在窗生成阶段做频域分析 | CPU 不做频域分析 | GPU 不做频域分析 | 三者都只在时域生成窗；频域特性是分析性质，不体现在生成代码中 |
| 原理 4：系数符号约定与归一化（`general_cosine` 不强制峰值=1） | `windows.py:137` 后无归一化步骤 | `windows_typed.cpp:265` `return out`，无归一化 | `windows_typed.cpp:318` `out = finalize_fp64_device_storage(computed)`，无归一化 | 三者一致，都不归一化；与 `chebwin`/`kaiser` 的归一化路径不同 |
| 原理 5：对称窗（sym=true）与周期窗（sym=false，DFT-even）的数学约定 | `_extend(M, sym)`（`windows.py:133`）在 host 修改 `M`，再 `size=M` 传给 kernel（`windows.py:137`） | `effective = symmetric ? length : length + 1`（`windows_typed.cpp:253`），用 `effective` 算相位但只写 `length` 个点 | `effective_count = sym ? n : n + 1` 作为 kernel 参数（`windows_typed.cu:73,74`），kernel 用 `effective_count` 算相位 | 数学等价；Python 在 host 改 `M` 后传 `size`，C++ 直接传 `effective_count` 给 kernel。Python 周期窗通过 `_truncate` 截断，C++ 通过只让线程写 `index < length` 实现 |
| 原理 6：项数与主瓣宽度/旁瓣电平/衰减率的权衡 | 由调用者选择 `a` 长度，kernel 不限制 | `coefficients.size()` 决定 `K`（`windows_typed.cpp:259`） | `coefficient_count` 决定 `K`（`windows_kernels.cuh:77`） | 三者一致；权衡是设计选择，不在生成代码中体现 |
| 原理 7：时域连续性阶数与旁瓣衰减率的关系 | 由 `a` 的项数隐式决定 | 同上 | 同上 | 三者都不显式计算连续性阶数 |
| 原理 8：加窗 = 时域乘法 ↔ 频域卷积 | `general_cosine` 只生成窗，不做加窗 | 同上 | 同上 | 加窗是下游操作，三者都不在窗生成阶段体现 |

补充公式符号说明：$n$ 为样本索引，$M$ 为有效长度（`effective`/`effective_count`），$K$ 为系数项数，$a_k$ 为第 $k$ 个系数，$\phi_n$ 为相位。

### 9. 与 cuSignal Python 实现相同、等价替换和有意不同的部分

#### 9.1 相同（逐字级或数学同构）

1. **核心公式同构**：Python `_general_cosine_kernel` body 的 `fac = -M_PI + delta*i` 与 `temp += a[k]*cos(k*fac)`，和 C++ GPU kernel 的 `phase = -pi + 2*pi*index/(effective-1)` 与 `value += window_load(coefficients[order]) * cos(order*phase)` 在数学上逐字对应，只是从 CuPy ElementwiseKernel 改写成原生 CUDA kernel。
2. **不做最终归一化**：Python `windows.py:139` `return _truncate(w, needs_trunc)` 之前无归一化；C++ CPU `return out`、GPU `out = finalize_fp64_device_storage(computed)` 都无归一化。两者一致（原理 4）。
3. **CPU 累加精度**：C++ CPU 用 `double` 累加（`windows_typed.cpp:258`），Python 用 `float64`（即 `double`），精度一致。
4. **sym 语义**：`symmetric ? length : length + 1` 与 `_extend(M, sym)` 在数学上等价（原理 5）。

#### 9.2 等价替换（工程实现不同，数学等价）

5. **GPU FP32 计算 + FP64 输出**：C++ GPU kernel 用 FP32 累加（`Scalar = float`），再由 `finalize_fp64_device_storage` D2D 拓宽为 FP64；Python 全程 FP64。这是工程近似，精度差异在测试容差内（见第 10、11 节）。
6. **无 `loop_prep`**：Python `windows.py:56` `loop_prep = "delta = (M_PI - -M_PI)/(_ind.size()-1)"` 预计算 `delta` 避免每线程重复除法；C++ kernel 在 body 内直接算 `2*pi*index/(effective_count-1)`。数学等价，性能上 Python 的预计算略优，但 C++ 的除法在每个线程内只发生一次，影响有限。
7. **grid/block 自动配置**：C++ 用 `launch_1d_kernel` 自动配置；Python 由 CuPy 自动配置。两者都隐藏了线程块配置细节。
8. **CPU 长度守卫短路**：Python `_len_guards(M)`（`windows.py:131`）在 `M <= 1` 时直接 `return cp.ones(M)`；C++ CPU 在 `length == 1` 返回 `{1.0}`、`length == 0` 返回 `{}`。注意 Python 对 `M < 0` 也走 `_len_guards` 返回 `cp.ones`，而 C++ 对 `length < 0` 抛异常（见 9.3）。GPU 在 `length == 1` 时 `computed = {1.0F}`，`length > 1` 才启动 kernel，与 Python 的 `M <= 1` 短路在 `M >= 0` 范围内等价。

#### 9.3 有意不同

9. **不支持任意维度 shape 的系数**：Python `a = cp.asarray(a, dtype=cp.float64)`（`windows.py:135`）接受任意 `array_like`，包括多维数组；C++ 提供两个重载——一维 `std::vector<T>` 和带 `coefficient_shape` 的版本。后者通过 `general_cosine_coefficient_count` 验证 shape 元素积等于 `data_size`，但只取 `shape.front()` 个系数（`windows_typed.cpp:287`），等价于把多维数组按第一维切片。这是工程上的等价但更受限：C++ 不能直接表达「多维系数数组按展平后全部参与」，而是强制要求调用者声明 shape 并只使用第一维。
10. **负长度处理**：Python `M < 0` 时 `_len_guards` 返回 `cp.ones(M)`（NumPy 语义下 `cp.ones(-3)` 返回空数组）；C++ CPU 与 GPU 都对 `length < 0` 抛 `std::invalid_argument`。C++ 的选择更严格，避免静默返回空数组掩盖调用错误。
11. **shape 重载额外验证**：C++ 的 `general_cosine_coefficient_count`（`windows_typed.cpp:269-288`）显式验证 `shape` 元素积 == `data_size`；Python 没有这个验证，因为 NumPy/CuPy 的 `asarray` 自动处理 shape。C++ 的验证是为了在没有 NumPy 的环境下保证调用者传入的 shape 与数据一致。

### 10. 精度、dtype、边界条件及 ZQ500 平台限制

#### 10.1 精度与 dtype

| 路径 | 系数 dtype | 累加 dtype | 输出 dtype | 与 Python 差异 |
| --- | --- | --- | --- | --- |
| Python | `float64` | `float64` | `float64` | 基准 |
| C++ CPU | `T`（5 种），经 `host_load` 转 `float` 再 `(double)` | `double` | `double` | 累加精度一致；系数在 `host_load` 阶段先转 `float` 再转 `double`，对 `__half`/整型系数无损，对 `float` 系数也无损 |
| C++ GPU | `T`（5 种），经 `window_load` 转 `float` | `float` | `double`（经 `finalize_fp64_device_storage`） | GPU 累加用 FP32，比 Python 的 FP64 低；对长窗或大动态范围系数可能引入额外舍入，但测试容差 4e-2 覆盖 |

注意 CPU 路径在 `windows_typed.cpp:260` 处 `(double)host_load(coefficients[order])`：`host_load` 返回 `float`，再显式转 `double`。对于 `__half` 系数，这意味着先 half→float→double，half 的精度损失在所难免，但与 Python 的 `cp.asarray(a, dtype=cp.float64)` 直接从输入数组转 FP64 相比，C++ 多了一步 float 中转。推测对 `__half` 输入，C++ CPU 与 Python 的差异主要来自 half→float 的舍入，而非 float→double。

#### 10.2 边界条件

| 输入 | CPU 行为 | GPU 行为 | Python 行为 |
| --- | --- | --- | --- |
| `length < 0` | 抛 `std::invalid_argument` | 抛 `std::invalid_argument` | `_len_guards` 返回空数组（不抛） |
| `length == 0` | 返回 `{}` | `computed` 空，`out` 空 | `_len_guards` 返回 `cp.ones(0)` |
| `length == 1` | 返回 `{1.0}`，不读系数 | `computed = {1.0F}`，不启动 kernel | `_len_guards` 返回 `cp.ones(1)` |
| `coefficients` 空 + `length > 1` | 内层循环不执行，`value = 0`，返回全零窗 | kernel 内 `coefficient_count = 0`，循环不执行，`value = 0`，全零窗 | `a` 为空数组时 `temp` 不累加，全零窗 |
| `sym = false` + `length` 偶/奇 | `effective = length + 1`，取前 `length` 点 | 同 CPU | `_extend` 改 `M` 后截断 |
| shape 重载 `shape` 空 | 抛 | 抛（在 `general_cosine_coefficient_count` 内） | Python 无此重载 |
| shape 重载 `product != data_size` | 抛 | 抛 | Python 无此验证 |

#### 10.3 ZQ500 平台限制

根据 `AGENTS.md` 与 `docs/development/ZQ500_RUNTIME.md`：

- 构建必须在 `gpu_02` 容器内执行 `source /zq500/sdk/env.sh` 后进行，CMake 实测 `3.22.1`，但 `cusignal_cpp` 的 `CMakeLists.txt` 保持 `cmake_minimum_required(VERSION 3.16)` 向下兼容。
- ZQ500 样例惯例：`project(... LANGUAGES CXX)`，从 `$DLICC_PATH` 设编译器，把 `.cu` 标记为 `LANGUAGE CXX`，链接 `curt`，用 `-x cuda` 编译。`general_cosine` 不调用 `curt`，但其所在的 `windows_typed.cu` 仍按平台惯例编译。
- `general_cosine_kernel` 用 `cosf`（FP32 `cos`），ZQ500 CUDA 平台支持。
- `launch_1d_kernel` 的具体 block 大小与 ZQ500 GPU 架构参数有关，需在容器内构建后用 `dlsmi` 确认 GPU 状态；本阶段未在容器内运行，仅做静态学习。
- 平台每天 `01:00-08:00` 可能维护，不适合长时间实验；`general_cosine` 是轻量窗生成，不涉及长时间运行。

### 11. 测试如何验证正确性，以及仍未覆盖的风险

#### 11.1 测试入口与覆盖场景

测试位于 `1ccd32ea` `ZKX_dev/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:475-522`，在 `run_type<T>` 模板内、`ok[7]` 槽位。`run_type<T>` 对五种 `T` 各实例化一次，故 `general_cosine` 测试覆盖 5 种 dtype。

| 场景 | 测试代码行 | 验证内容 |
| --- | --- | --- |
| 标准对称窗 `coeff={1,0}`（矩形窗） | `e3:478-480` `general_cosine_device(5, d_coeff, fixed_fp64_window_out)` + `matches_fp64(..., general_cosine_typed_cpu(5, coeff))` | GPU 输出与 CPU reference 在 `matches_fp64` 容差内一致 |
| 空系数 + `length=5` | `e3:481-484` `general_cosine_device<T>(5, empty_coefficients, ...)` + `== std::vector<double>(5, 0.0)` | 空系数产生全零窗 |
| 单点 `length=1` | `e3:485` `general_cosine_device<T>(1, empty_coefficients, singleton_fp64_window_out)` + `e3:517` `== std::vector<double>{1.0}` | 单点退化为 `[1.0]` |
| 空窗 `length=0` | `e3:486` `general_cosine_device<T>(0, empty_coefficients, empty_fp64_window_out)` + `e3:519` `empty()` | 零长度返回空 |
| 负长度 GPU 抛 | `e3:489-494` `try { general_cosine_device<T>(-1, ...) } catch(std::invalid_argument&)` | GPU 对负长度抛异常 |
| 负长度 CPU 抛 | `e3:495-498` `try { general_cosine_typed_cpu<T>(-1, {}) } catch(std::invalid_argument&)` | CPU 对负长度抛异常 |
| 周期窗 `sym=false` | `e3:499-500,514-516` `general_cosine_device<T>(4, d_coeff, cosine_periodic_out, false)` + `matches_fp64(..., general_cosine_typed_cpu(4, coeff, false))` | 周期窗 GPU 与 CPU 一致 |
| 二维 shape `{2,3}` | `e3:501-511` `general_cosine_device<T>(5, d_rank2_coefficients, std::vector<int>{2,3}, ...)` + 与一维 `coeff={1,0}` 结果对比 | shape 重载只取前 2 个系数，与一维 `{1,0}` 结果一致 |
| CPU 单点 reference | `e3:518` `general_cosine_typed_cpu<T>(1, {}) == std::vector<double>{1.0}` | CPU 单点退化为 `[1.0]` |
| CPU 空窗 reference | `e3:520` `general_cosine_typed_cpu<T>(0, {}).empty()` | CPU 零长度返回空 |

`ok[7]` 是上述所有断言的逻辑与（`e3:512-522`）。

#### 11.2 容差

- `matches_fp64`：用于 GPU 输出与 CPU reference、CPU reference 与 CPU reference 的对比，容差 `4e-2`（来源：阶段二与测试代码，本阶段未单独读 `matches_fp64` 实现，容差值来自任务说明）。
- `matches_real<T>`：容差 `3e-2`，用于 `__half` 等类型的实型对比。
- 容差设置反映 FP32 GPU 与 FP64 CPU 之间的精度差异；`4e-2` 对矩形窗（值域 `[0,1]`）是宽松的，但对高项数系数可能出现的大动态范围可能不够（见 11.3）。

#### 11.3 仍未覆盖的风险

1. **大长度窗的 FP32 累加误差**：测试只覆盖 `length <= 5`。当 `length` 很大（如 4096）且系数项数多时，FP32 累加的舍入误差可能超出 `4e-2` 容差。测试未验证大长度场景。
2. **大动态范围系数**：测试只覆盖 `{1,0}` 和 `{1,0,9,9,9,9}`（后者只取前 2 个）。若系数本身有大的动态范围（如 `chebwin` 派生的系数），FP32 中间结果的精度可能不足。
3. **`__half` 系数的精度**：`__half` 系数经 `window_load` 转 `float` 再 FP32 累加，再 FP64 拓宽。测试用 `convert<T>(1)` 和 `convert<T>(0)`，对 `__half` 是精确表示；未覆盖 `__half` 无法精确表示的系数值。
4. **`launch_1d_kernel` 的 grid/block 上限**：当 `length` 超过单 grid 的最大线程数时，`launch_1d_kernel` 的行为未在测试中验证。推测其内部会分批或使用多维 grid，但本阶段未读实现。
5. **默认 stream 同步**：测试在 `general_cosine_device` 返回后立即 `to_host()`，隐式依赖默认 stream 的同步语义。若调用者在未同步时读取 `out`，可能读到未完成的数据。测试未显式验证同步契约。
6. **shape 重载的 shape 维度 > 2**：测试只覆盖 `{2,3}` 二维 shape。`general_cosine_coefficient_count` 支持任意维度，但高维 shape 的行为未测试。
7. **`finalize_fp64_device_storage` 的空输入**：`length == 0` 时 `computed` 为空，`finalize` 产出空 `DeviceArray<double>`。测试覆盖了空窗场景，但未验证 `finalize` 对空输入的内部行为（推测为 no-op）。
8. **ZQ500 平台实际运行**：本阶段为静态学习，未在 `gpu_02` 容器内构建或运行测试。CMake 适配、`curt` 链接、`-x cuda` 编译等平台特性需在容器内验证。

### 12. 自检问题

按 `AGENTS.md`「六、完成质量检查」清单逐项自检：

- [x] 没有混入其他算子的泛化内容：本文档只描述 `general_cosine`，`chebwin`/`hamming` 等仅作为对比参照提及。
- [x] 通用原理来自实际网络检索：阶段一已完成，本阶段引用原理 1-8 编号，不反推。
- [x] cuSignal 实际算法已逐项映射到通用原理编号：第 8 节映射表覆盖 8 条原理，包含「未对应」（原理 3、7、8）关系。
- [x] 没有把函数入口、helper 或测试代码误当作核心算法：`host_load`/`window_load` 等 helper 单独列出，核心算法是 `general_cosine_typed_cpu` 主重载与 `general_cosine_kernel`。
- [x] 每个公式中的符号都已解释：第 8 节末尾补充了 $n$、$M$、$K$、$a_k$、$\phi_n$ 的含义。
- [x] 每个重要原理都能定位到后续源码中的具体实现：映射表给出 Python/C++ CPU/C++ GPU 三方行号。
- [x] 源码文件、符号名和行号真实存在：均通过 Read 工具核对。
- [x] `cusignal_cpp` 学习内容已绑定可追溯的完整 Git SHA：第 1 节给出完整 SHA 与 dirty 状态。
- [x] 旧版与优化版说明没有相互覆盖：V2/V3 占位，未覆盖 V1。
- [x] 没有声称执行过实际未执行的代码或测试：第 11.3 节明确标注「本阶段为静态学习，未在 `gpu_02` 容器内构建或运行测试」。
- [x] CPU 与 GPU 的职责没有混写：第 3 节专写 CPU，第 4 节专写 GPU，第 8 节映射表分列三列。
- [x] 文档足以让学习者按给定顺序亲自阅读源码并复述执行逻辑：调用链伪代码 + 行号 + 映射表 + 阅读建议。

补充自检（推测项标注）：
- `launch_1d_kernel` 的具体 block 大小标注为「推测其行为与标准 `<<<ceil(N/threadsPerBlock), threadsPerBlock>>>` 一致」，未声称已确认。
- `finalize_fp64_device_storage` 的空输入行为标注为「推测为 no-op」。

---

## V2：第一次优化版本（占位）

暂无。待后续优化提交后在此追加，独立记录完整 SHA、代码定位、CPU/GPU 调用链、数学映射与测试证据。不覆盖 V1 内容，不修改 V1 的路径行号。

---

## V3：第二次优化版本（占位）

暂无。待后续优化提交后在此追加。

---

## 版本差异与原理不变量

| 维度 | V1（`1ccd32ea`） | V2（占位） | V3（占位） |
| --- | --- | --- | --- |
| 算法结构 | 双层循环：外层逐点、内层逐系数 | — | — |
| 复杂度 | $O(M \cdot K)$，$M$ 为长度，$K$ 为系数项数 | — | — |
| 内存访问 | 每线程读 `coefficient_count` 个系数（全局内存），写 1 个输出；系数无共享内存缓存 | — | — |
| 并行方式 | 一维 grid，每线程 1 个输出点，无归约 | — | — |
| 精度边界 | GPU FP32 累加 + FP64 输出；CPU FP64 累加 | — | — |
| 性能 | 未测量（本阶段为静态学习） | — | — |

**原理不变量**：无论后续优化如何改变工程实现，以下数学物理原理始终不变：

- 原理 1（通用余弦窗时域定义 $w[n]=\sum_k a_k \cos(k\phi_n)$）是生成公式的唯一来源；
- 原理 4（`general_cosine` 不强制峰值归一化）决定了无归一化步骤；
- 原理 5（对称/周期窗的 `effective = sym ? n : n+1` 约定）决定了相位分母。

若 V2/V3 只改变工程实现（如共享内存缓存系数、FP32→FP64 中间精度调整、loop unroll），应标注「原理不变、实现优化」；若算法本身改变（如改用 FFT 反变换生成），应重新映射到阶段一原理编号。

---

## 阅读建议

建议按以下顺序阅读本文档与源码：

1. **先看第 2 节**，建立五层职责（API 包装 / CPU 核心 / CPU helper / GPU host 调度 / GPU kernel）的全局图景。
2. **读 CPU 主重载**：打开 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:246-266`，对照第 3.1 节伪代码逐行理解长度守卫、`effective` 计算、双层循环。注意 `pi64` 是局部常量而非文件级 `kPi`。
3. **读 GPU kernel**：打开 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_kernels.cuh:58-82`，对照第 4.4 节。重点对比 kernel 内 `phase` 公式与 CPU 第 257 行的 `phase` 公式，确认两者数学同构、仅精度不同。
4. **读 GPU host 调度层**：打开 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cu:73-74`（单行实现），对照第 4.3 节。注意 `static_assert`、参数校验顺序、`sym?n:n+1` 如何作为 `effective_count` 传入 kernel。
5. **读 API 包装层**：打开 `1ccd32ea` `ZKX_dev/cusignal_cpp/src/windows/windows_typed.cpp:303-319` 与 `321-339`，对照第 4.1、4.2 节。注意 `length == 1` 短路、`finalize_fp64_device_storage` 的 FP32→FP64 拓宽。
6. **对照第 8 节映射表**，把 Python（阶段二）→ CPU → GPU 三方公式逐项对齐，确认原理 1、4、5 在三者中的落实位置。
7. **读第 9 节**，理解 10 项工程决策中哪些是「相同」、哪些是「等价替换」、哪些是「有意不同」。重点关注意 5（FP32/FP64）、意 9（shape 限制）、意 10（负长度抛异常）。
8. **读第 11 节**，理解测试覆盖了哪些场景、容差是多少、哪些风险仍未覆盖。特别注意 11.3 的第 1、2 项（大长度与大动态范围的 FP32 误差风险）。
9. **若需在 ZQ500 平台验证**，回到 `AGENTS.md` 与 `docs/development/ZQ500_RUNTIME.md`，按「标准远程工作流」同步、`docker cp`、进入 `gpu_02`、`source /zq500/sdk/env.sh` 后构建测试。
