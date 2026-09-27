# cusignal_cpp_chebwin复现逻辑

## 版本索引

| 版本 | 短 SHA | 完整 SHA | 提交时间 | 分支 | 状态 |
| --- | --- | --- | --- | --- | --- |
| V1 | `1ccd32ea` | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` | 2026-07-27 22:12:16 +0800 | `snapshot/all-current-20260716` | 当前实现 |
| V2 | — | — | — | — | 未发生,占位 |
| V3 | — | — | — | — | 未发生,占位 |

- 当前活动版本:**V1**,即 `1ccd32ea` 提交下的 cusignal_cpp chebwin 复现。
- 后续优化版本按 V2、V3 顺序追加到本文末尾,不覆盖 V1。
- 各版本差异见文末“版本差异与原理不变量”章节。

---

## V1:原始学习版本(`1ccd32ea`)

### 1. Git 版本信息

| 项 | 值 |
| --- | --- |
| 完整 SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 短 SHA | `1ccd32ea` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支 | `snapshot/all-current-20260716` |
| cusignal_cpp dirty 状态 | 干净(无未提交修改) |

**相关文件(`ZKX_dev/` 仓库相对路径)**:

- `cusignal_cpp/src/windows/windows_typed.h`
- `cusignal_cpp/src/windows/windows_typed.cpp`
- `cusignal_cpp/src/windows/windows_typed.cu`
- `cusignal_cpp/src/windows/windows_kernels.cuh`
- `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注"无直接原理对应"。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在"含义解释"列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §1 `windows_typed.h:21-22`（`chebwin_typed_cpu` 模板参数） | `template <class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，决定 `attenuation` 输入 dtype；显式实例化为 `float`/`__half`/`std::int32_t`/`std::int16_t`/`std::int8_t` 五种；输出固定 `double`（CPU）/`DeviceArray<double>`（GPU） |
| `length` | §1 `windows_typed.h:21-22`（`chebwin_typed_cpu` 首参数） | `int n`（CPU 入口记作 `length`） | 原理 1/2/3/5 中的 $M$ | 窗长度（点数），约束 $M \geq 0$；GPU 侧以 `int n` 形式出现于 `chebwin_device` |
| `attenuation` | §1 `windows_typed.h:38-40`（`chebwin_device` 第二参数） | `T attenuation`，5 种 dtype 之一 | 原理 5 §1 中的 $A$ | 旁瓣衰减（dB），$A$ 越大旁瓣越低；接受负值，由后续 `fabs` 取绝对值处理 |
| `out` | §1 `windows_typed.h:38-40`（`chebwin_device` 第三参数） | `DeviceArray<double>& out` | 原理 3 §1 中的 $w[n]$（归一化后） | GPU 输出窗序列，FP64 存储但 device 计算 FP32，经 `finalize_fp64_device_storage` 拓宽；CPU 侧以 `std::vector<double> out` 形式出现 |
| `sym` | §1 `windows_typed.h:38-40`（`chebwin_device` 第四参数） | `bool sym = true` | cuSignal 算法 1 §3（步骤 3 周期扩展） | 工程旗标，`true` 对称窗（`effective = n`），`false` 周期窗（`effective = n+1` 再截断）；CPU 侧以 `symmetric` 形式出现 |
| `effective` | §3.1 `windows_typed.cpp:169`（`chebwin_typed_cpu` 局部） | `int`，`symmetric ? length : length + 1` | cuSignal 算法 1 §3（步骤 3 DFT-even 扩展） | 工程变量，对称中心化长度；GPU 侧以 `effective_count` 形式出现于 `chebwin_fp32_compute_device`（`windows_typed.cu:22`）与 kernel 参数 |
| `attenuation_value` | §3.1 `windows_typed.cpp:171-172`（`chebwin_typed_cpu` 局部） | `double`，`host_load<T>(attenuation)` 后 `(double)` 拓宽 | 原理 5 §1 中的 $A$（FP64 标量） | 由 5 种输入类型统一转 FP64 的衰减值，CPU 路径专用 |
| `beta` | §3.1 `windows_typed.cpp:173-174`（`chebwin_typed_cpu` 局部） | `double`，`std::cosh(std::acosh(std::pow(10.0, std::fabs(attenuation_value)/20.0)) / (effective-1))` | 原理 5 §1 中的 $\beta$ | Chebyshev 增长参数，控制主瓣宽度与旁瓣电平；GPU 侧以 `Scalar beta` 形式在 `chebwin_raw_kernel` 内重算（`windows_kernels.cuh:269-272`） |
| `peak` | §3.1 `windows_typed.cpp:210, 217`（`chebwin_typed_cpu` 局部） | `double`（CPU）/`float`（GPU），`std::max_element` / `thrust::max_element` 返回 | cuSignal 算法 1 §8（归一化常数） | 时域窗最大值，用于归一化使峰值为 1；与原理 6（端点脉冲）相关 |
| `raw` | §3.1 `windows_typed.cpp:218-221`（`chebwin_typed_cpu` 循环局部）/ §4.2 `windows_typed.cu:52`（GPU `DeviceArray<float>`） | `double`（CPU 局部）/`Scalar*`（GPU 缓冲区指针） | 原理 3 §1 中的 $w[n]$（未归一化） | 工程临时变量，DP 路径的 IDFT 求和结果；GPU 侧作为 `chebwin_raw_kernel` 输出缓冲区 |
| `i` | §3.1 `windows_typed.cpp:211, 222`（`chebwin_typed_cpu` 循环变量） | `int` | 无直接原理对应 | 工程变量，CPU 写出循环索引，对应 `out[i]` 的位置 |
| `host_load<T>` | §3.4 `windows_typed.cpp:17-21`（host helper 函数模板） | `template <class T> float host_load(T value)` | 无直接原理对应 | 工程类型转换函数，调用 `SimpleSignalTypePolicy<T>::load` 将 5 种 dtype 统一转 `float`；CPU 再 `(double)` 拓宽为 FP64 |
| `host_chebyshev_polynomial` | §3.4 `windows_typed.cpp:47-72`（匿名命名空间函数） | 函数，参数 `(int frequency, int length, double beta, double& real, double& imag)` | 原理 1 §1（$T_n(x)$）+ 原理 2 §1（$W(k)$） | CPU 频域采样函数，按 $|x|$ 三分支解析求 $T_{M-1}(x)$ 并配奇偶长度相位因子；GPU 侧由 `chebyshev_spectrum_real` 等价替代（Clenshaw 递推） |
| `frequency` | §3.4 `windows_typed.cpp:47`（`host_chebyshev_polynomial` 首参数） | `int` | 原理 2/3 §1 中的 $k$ | 频域离散频率索引，$0 \leq k \leq M-1$；kernel 内作为内层循环变量 |
| `real` | §3.4 `windows_typed.cpp:47`（`host_chebyshev_polynomial` 输出参数） | `double&` | 原理 2 §1 中的 $\Re\{W(k)\}$ | 频域响应实部；kernel 侧以 `spectrum_real` 形式出现 |
| `imag` | §3.4 `windows_typed.cpp:47`（`host_chebyshev_polynomial` 输出参数） | `double&` | 原理 2 §1 中的 $\Im\{W(k)\}$ | 频域响应虚部（偶长度非零，奇长度为 0）；kernel 侧以 `spectrum_imag` 形式出现 |
| `x` | §3.4 `windows_typed.cpp:51`（`host_chebyshev_polynomial` 局部） | `double`，`beta * std::cos(pi * frequency / length)` | 原理 2 §1 中的 $x(k) = \beta\cos(\pi k/M)$ | Chebyshev 多项式自变量；kernel 侧在 `chebyshev_spectrum_real`（`windows_kernels.cuh:255`）以 `Scalar x` 形式出现 |
| `value` | §3.4 `windows_typed.cpp:55-66`（`host_chebyshev_polynomial` 局部） | `double`，按 $|x|$ 三分支解析计算 | 原理 1 §1 中的 $T_{M-1}(x)$ | Chebyshev 多项式求值结果（不归一化） |
| `angle` | §3.4 `windows_typed.cpp:84`（`host_chebyshev_raw` 局部） | `double`，`-2 * pi * index * frequency / length` | 原理 3 §1 中的 $2\pi nk/M$ | IDFT 相位因子；kernel 侧在 `chebyshev_raw_value`（`windows_kernels.cuh:238`）以 `Scalar angle` 形式出现 |
| `host_chebyshev_raw` | §3.4 `windows_typed.cpp:74-89`（匿名命名空间函数） | 函数，参数 `(int index, int length, double beta)` | 原理 3 §1（IDFT 实数求和） | CPU 时域采样函数，对 `frequency = 0..length-1` 累加 `real*cos(angle) - imag*sin(angle)`，等价于显式 IDFT 实部求和；GPU 侧由 `chebyshev_raw_value` 等价替代 |
| `index` | §3.4 `windows_typed.cpp:74`（`host_chebyshev_raw` 首参数） | `int` | 原理 3 §1 中的 $n$ | 时域输出索引；GPU 侧以 `output_index` 形式出现于 `chebyshev_raw_value` |
| `sum` | §3.4 `windows_typed.cpp:82`（`host_chebyshev_raw` 局部） | `double`，累加 `real*cos(angle) - imag*sin(angle)` | 原理 3 §1 中的 $\sum_k W(k) e^{j2\pi nk/M}$ | IDFT 实部求和累加器；kernel 侧以 `Scalar sum` 形式出现于 `chebyshev_raw_value` |
| `host_chebyshev_raw_index` | §3.4 `windows_typed.cpp:91-103`（匿名命名空间函数） | 函数，参数 `(int output_index, int length)` | cuSignal 算法 1 §7（对称重排） | CPU 对称中心化索引映射，把以中心为 0 的输出索引映射到 DFT 源索引；GPU 侧由 `chebyshev_raw_index` 完全等价替代 |
| `output_index` | §3.4 `windows_typed.cpp:91`（`host_chebyshev_raw_index` 首参数） | `int` | 原理 3 §1 中的 $n$ | 对称中心化的时域输出索引；kernel 侧同名出现于 `chebyshev_raw_value` |
| `center` | §3.4 `windows_typed.cpp:99`（`host_chebyshev_raw_index` 局部） | `int`，奇长度 `(length+1)/2 - 1` | 无直接原理对应 | 工程变量，奇长度对称中心索引 |
| `half` | §3.4 `windows_typed.cpp:101`（`host_chebyshev_raw_index` 局部） | `int`，偶长度 `length/2` | 无直接原理对应 | 工程变量，偶长度半窗宽度 |
| `computed` | §4.1 `windows_typed.cpp:237`（`chebwin_device` 局部） | `DeviceArray<float>`，长度 `out.size()` | 无直接原理对应 | 工程 FP32 临时缓冲区，承载 GPU kernel 的 FP32 计算结果，随后经 `finalize_fp64_device_storage` 拓宽为 FP64 |
| `effective_count` | §4.2 `windows_typed.cu:22`（`chebwin_fp32_compute_device` 局部） | `int`，`symmetric ? n : n + 1` | cuSignal 算法 1 §3（步骤 3 DFT-even 扩展） | 工程变量，GPU 侧的对称中心化长度，与 CPU `effective` 同义；作为 kernel 参数传入 `chebwin_raw_kernel` |
| `Parameter` | §4.4.1 `windows_kernels.cuh:256`（`chebwin_raw_kernel` 模板参数） | `typename Parameter`，显式实例化为 `T` | 无直接原理对应 | 工程 kernel `attenuation` 参数类型，与 host `T` 一致 |
| `Scalar` | §4.4.2 `windows_kernels.cuh:226`（`chebyshev_raw_value` 模板参数） | `typename Scalar`，显式实例化为 `float` | 无直接原理对应 | 工程 kernel 计算精度类型，固定 FP32（device 计算缓冲） |
| `pi` | §4.4.2 `windows_kernels.cuh:229`（`chebyshev_raw_value` 局部常量） | `const Scalar pi = 3.14159265358979323846F` | 数学常数 $\pi$（FP32 精度） | 圆周率 FP32 字面量，kernel 全程使用；CPU 侧以 `<cmath>` 标准常量或局部 `pi64` 形式出现 |
| `attenuation_input` | §4.4.1 `windows_kernels.cuh:207`（`chebwin_raw_kernel` 参数） | `Parameter attenuation_input` | 原理 5 §1 中的 $A$ | kernel 接收的原始衰减参数，经 `window_load` 加载为 `Scalar attenuation`（FP32） |
| `source` | §4.4.2 `windows_kernels.cuh:230`（`chebyshev_raw_value` 局部） | `int`，`chebyshev_raw_index(output_index, length)` 返回值 | cuSignal 算法 1 §7（对称重排） | 工程变量，对称中心化映射后的 DFT 源索引 |
| `odd` | §4.4.2 `windows_kernels.cuh:231`（`chebyshev_raw_value` 局部） | `bool`，`(length & 1) != 0` | 无直接原理对应 | 工程奇偶长度标志，决定 `spectrum_imag` 是否为 0 |
| `spectrum_real` | §4.4.2 `windows_kernels.cuh:236`（`chebyshev_raw_value` 局部） | `Scalar`，`odd ? real : real * cos(phase)` | 原理 2 §1 中的 $\Re\{W(k)\}$ | kernel 内的频域响应实部，已配奇偶长度相位因子 |
| `spectrum_imag` | §4.4.2 `windows_kernels.cuh:237`（`chebyshev_raw_value` 局部） | `Scalar`，`odd ? 0 : real * sin(phase)` | 原理 2 §1 中的 $\Im\{W(k)\}$ | kernel 内的频域响应虚部，奇长度为 0 |
| `phase` | §4.4.2 `windows_kernels.cuh:235`（`chebyshev_raw_value` 局部） | `Scalar`，`pi * frequency / length` | 原理 2 §1 中的 $\pi k/M$ | 工程变量，偶长度相位因子角，用于把实部虚部分配到 $\cos$/$\sin$ |
| `chebyshev_spectrum_real` | §4.4.3 `windows_kernels.cuh:196-216`（device inline 函数） | 函数，参数 `(int frequency, int length, Scalar beta)` | 原理 1 §1（$T_n(x)$ 递推） | GPU Chebyshev 多项式求值，用 Clenshaw 三项递推 $T_{k+1} = 2xT_k - T_{k-1}$ 统一求值，不分 $|x|>1$ 与 $|x|\le 1$ 分支 |
| `order` | §4.4.3 `windows_kernels.cuh:256`（`chebyshev_spectrum_real` 局部） | `int`，`length - 1` | 原理 1 §1 中的 $n$（多项式阶数） | Chebyshev 多项式阶数，等于 $M-1$ |
| `previous` | §4.4.3 `windows_kernels.cuh:258`（`chebyshev_spectrum_real` 局部） | `Scalar`，初始化为 `1` | 原理 1 §1 中的 $T_0(x) = 1$ | Clenshaw 递推的前一项，初值 $T_0$ |
| `current` | §4.4.3 `windows_kernels.cuh:259`（`chebyshev_spectrum_real` 局部） | `Scalar`，初始化为 `x` | 原理 1 §1 中的 $T_1(x) = x$ | Clenshaw 递推的当前项，初值 $T_1$ |
| `next` | §4.4.3 `windows_kernels.cuh:261`（`chebyshev_spectrum_real` 局部） | `Scalar`，`2 * x * current - previous` | 原理 1 §1 中的 $T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)$ | Clenshaw 递推的下一项 |
| `chebwin_raw_kernel` | §4.4.1 `windows_kernels.cuh:256-274`（GPU kernel） | `__global__` 模板，参数 `(int effective_count, Parameter attenuation_input, Scalar* raw)` | 无直接原理对应 | 工程 GPU kernel，每线程算一个 effective 长度上的采样点 |
| `chebyshev_raw_value` | §4.4.2 `windows_kernels.cuh:232-254`（device inline 函数） | 函数，参数 `(int output_index, int length, Scalar beta)` | 原理 3 §1（IDFT 实数求和） | GPU 时域采样函数，与 CPU `host_chebyshev_raw` 数学等价，差别仅在 FP32 精度与 Clenshaw 实现方式 |
| `chebyshev_raw_index` | §4.4.4 `windows_kernels.cuh:218-230`（device inline 函数） | 函数，参数 `(int output_index, int length)` | cuSignal 算法 1 §7（对称重排） | GPU 对称中心化索引映射，与 CPU `host_chebyshev_raw_index` 完全一致 |
| `normalize_window_kernel` | §4.4.5 `windows_kernels.cuh:359-370`（GPU kernel） | `__global__` 模板，参数 `(const Scalar* raw, int output_count, Scalar peak, T* output)` | cuSignal 算法 1 §8（归一化） | 工程 GPU kernel，对 `raw` 做 `value = peak > 0 ? raw[index] / peak : raw[index]` 归一化写回 `output` |
| `output_count` | §4.4.5 `windows_kernels.cuh:281`（`normalize_window_kernel` 参数） | `int`，等于 `n`（用户请求长度） | cuSignal 算法 1 §9（sym=false 截断） | 工程变量，输出长度，sym=false 时小于 `effective_count`，完成截断 |
| `output` | §4.4.5 `windows_kernels.cuh:281`（`normalize_window_kernel` 参数） | `T* output` | 原理 3 §1 中的 $w[n]$（归一化后） | kernel 输出指针，对应 `computed.data()`，最终经 `finalize` 拓宽为 FP64 |
| `window_load<T>` | §5.2 `windows_kernels.cuh:10-18`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::load`，5 种 dtype 统一转 `float` |
| `window_store<T, Scalar>` | §5.2 `windows_kernels.cuh:20-28`（device helper） | `__host__ __device__` 模板函数 | 无直接原理对应 | 工程类型转换函数，对 `is_simple_signal_input_v<T>` 调用 `SimpleSignalTypePolicy<T>::store`，把 FP32 `value` 转回 `T` |
| `SimpleSignalTypePolicy<T>` | §5.2 `cuda_utils/simple_signal_typed.h`（类型策略模板） | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断） |
| `launch_1d_kernel` | §4.3 `cuda_utils/kernel_launch.h`（kernel 启动工具） | 函数模板，参数 `(Kernel kernel, std::size_t elements, Args&&... args)` | 无直接原理对应 | 工程 kernel 启动工具，自动按 `elements` 计算 grid/block（默认 `block_size=256`，默认 stream） |
| `finalize_fp64_device_storage` | §4.1 `cuda_utils/host_output_finalize.h`（FP32→FP64 拓宽工具） | 函数，参数 `(DeviceArray<float> computed)` 返回 `DeviceArray<double>` | 无直接原理对应 | 工程 FP32→FP64 拓宽工具，内部走 D2H → host float→double → H2D |
| `is_simple_signal_input_v<T>` | §5.1 `cuda_utils/simple_signal_typed.h`（类型 trait） | `template <class T> constexpr bool is_simple_signal_input_v` | 无直接原理对应 | 工程类型 trait，编译期判断 `T` 是否属于 5 种简单信号类型，用于 `static_assert` |
| `spectrum` | §4.2 `windows_typed.cu:25`（FFT 路径临时缓冲，默认不启用） | `DeviceArray<ComplexFloat>`，长度 `effective` | 原理 2 §1 中的 $W(k)$（复数形式） | 工程 FFT 路径的频域序列缓冲区，仅当定义 `CUSIGNAL_WINDOWS_FFT_CHEBWIN` 时使用 |
| `transformed` | §4.2 `windows_typed.cu:26`（FFT 路径临时缓冲，默认不启用） | `DeviceArray<ComplexFloat>`，长度 `effective` | 原理 3 §1 中的 IDFT 结果 | 工程 FFT 路径的 IFFT 输出缓冲区 |
| `reordered` | §4.2 `windows_typed.cu:28`（FFT 路径临时缓冲，默认不启用） | `DeviceArray<float>`，长度 `effective` | 原理 3 §1 中的 $w[n]$（重排后实部） | 工程 FFT 路径重排后的实窗缓冲区 |
| `kPi` | §2 `windows_typed.cpp:15`（文件级常量） | `constexpr float kPi = 3.14159265358979323846F` | 数学常数 $\pi$（FP32 精度） | 工程文件级 FP32 圆周率常量；chebwin 主路径未直接引用（`host_chebyshev_*` 使用 `<cmath>` 标准常量），列此供其他 windows 算子共用对照 |

---

### 2. 对外接口与代码定位

所有行号针对 `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` 提交。

#### 2.1 API 包装层(GPU host 入口)

- 声明:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.h` / `chebwin_device` / `windows_typed.h:38-40`
  ```cpp
  template <class T>
  void chebwin_device(
      int n, T attenuation, DeviceArray<double>& out, bool sym = true);
  ```
- 实现:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cpp` / `chebwin_device` / `windows_typed.cpp:230-244`

#### 2.2 CPU / reference 核心计算

- 声明:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.h` / `chebwin_typed_cpu` / `windows_typed.h:21-22`
  ```cpp
  template <class T>
  std::vector<double> chebwin_typed_cpu(int n, T attenuation, bool sym = true);
  ```
- 实现:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cpp` / `chebwin_typed_cpu` / `windows_typed.cpp:163-228`

#### 2.3 GPU host 调度层

- 前置声明:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cpp` / `chebwin_fp32_compute_device` / `windows_typed.cpp:142-143`
- 实现:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cu` / `chebwin_fp32_compute_device` / `windows_typed.cu:14-72`

#### 2.4 GPU kernel / device 核心计算(全部位于 `windows_kernels.cuh`)

- `chebwin_raw_kernel`(默认 direct DFT 路径):`1ccd32ea` / `cusignal_cpp/src/windows/windows_kernels.cuh` / `chebwin_raw_kernel` / `windows_kernels.cuh:256-274`
- `chebyshev_raw_value`(device inline,单点 DFT 求和):`windows_kernels.cuh:232-254`
- `chebyshev_raw_index`(对称中心化索引重排):`windows_kernels.cuh:218-230`
- `chebyshev_spectrum_real`(Clenshaw 递推求 Chebyshev 多项式):`windows_kernels.cuh:196-216`
- `normalize_window_kernel`(归一化写回):`windows_kernels.cuh:359-370`
- `chebwin_spectrum_kernel`(FFT 路径,half-angle 稳定化,默认未启用):`windows_kernels.cuh:277-330`
- `chebwin_reorder_kernel`(FFT 路径重排,默认未启用):`windows_kernels.cuh:332-349`

#### 2.5 CPU 数学 helper(匿名命名空间)

- `host_chebyshev_polynomial`:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cpp` / `host_chebyshev_polynomial` / `windows_typed.cpp:47-72`
- `host_chebyshev_raw`:`windows_typed.cpp:74-89`
- `host_chebyshev_raw_index`:`windows_typed.cpp:91-103`

#### 2.6 测试

- 测试位置:`1ccd32ea` / `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` / `run_type<T>` 内 `ok[6]` / `e3_wave_window_type_smoke.cu:440-473`

### 3. CPU 调用链、数据结构与逐步算法

入口:`chebwin_typed_cpu(length, attenuation, symmetric)`(`windows_typed.cpp:163-228`)。

#### 3.1 调用链

```
chebwin_typed_cpu(length, attenuation, symmetric)
  ├─ 守卫:length < 0 → throw std::invalid_argument  (windows_typed.cpp:166)
  ├─ 守卫:length == 0 → return {}                   (windows_typed.cpp:167)
  ├─ 守卫:length == 1 → return {1.0}                (windows_typed.cpp:168)
  ├─ effective = symmetric ? length : length + 1      (windows_typed.cpp:169)
  ├─ out: std::vector<double>(length)                (windows_typed.cpp:170)
  ├─ attenuation_value = host_load(attenuation) 转 FP64 (windows_typed.cpp:171-172)
  ├─ beta = cosh(acosh(10^|at|/20) / (effective-1))  (windows_typed.cpp:173-174)
  ├─ #ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN → 走 FFT 路径(默认不启用)
  │     ├─ 显式构造频谱 spectrum[effective]        (windows_typed.cpp:176-195)
  │     ├─ FFTInterface_cpu(effective).fft(spectrum) (windows_typed.cpp:196)
  │     ├─ 按 host_chebyshev_raw_index 规则重排 → full[effective] (windows_typed.cpp:197-209)
  │     ├─ peak = std::max_element(full)             (windows_typed.cpp:210)
  │     └─ out[i] = full[i] / peak,i = 0..length-1  (windows_typed.cpp:211-214)
  └─ #else(默认 direct DFT 路径)
        ├─ peak = -inf                               (windows_typed.cpp:217)
        ├─ for i in 0..effective-1:
        │     raw = host_chebyshev_raw(host_chebyshev_raw_index(i, effective), effective, beta)
        │     peak = max(peak, raw)                  (windows_typed.cpp:218-221)
        └─ for i in 0..length-1:
              out[i] = host_chebyshev_raw(host_chebyshev_raw_index(i, effective), effective, beta) / peak
                                                    (windows_typed.cpp:222-225)
```

#### 3.2 主要数据结构

- `out`:`std::vector<double>` 长度 `length`,FP64 输出。
- `effective`:内部对称中心化长度,`symmetric ? length : length + 1`,与 Python `_extend`/`_truncate` 在 sym=false 时的扩展语义一致。
- `attenuation_value`:FP64 标量,由 `host_load<T>` 统一从 5 种输入类型转换。
- `beta`:频域 Chebyshev 增长参数,$\beta = \cosh\!\left(\dfrac{\operatorname{acosh}(10^{|at|/20})}{M-1}\right)$,其中 $M=$ `effective`。
- `peak`:DP 时域最大值,用于归一化使窗峰值为 1。

#### 3.3 逐步算法(默认 direct DFT 路径)

1. 长度守卫:`length < 0` 抛异常;`0` 返回空;`1` 返回 `{1.0}`。
2. effective 长度:`sym=false` 时按 `length+1` 内部计算,再截前 `length` 个点写入 `out`。
3. 参数转换:`host_load<T>(attenuation)` → `attenuation_value` (FP64),对负值由后续 `std::fabs` 处理。
4. 计算 $\beta$:`std::pow(10.0, std::fabs(attenuation_value)/20.0)` 给出线性旁瓣比 $R$;`std::acosh(R)/(effective-1)` 给出每点增长;`std::cosh` 得 $\beta$。
5. direct DFT 求时域采样:对 `index = 0..effective-1`,先 `host_chebyshev_raw_index(index, effective)` 把对称中心化输出索引映射到 DFT 源索引 `source`,再 `host_chebyshev_raw(source, effective, beta)` 求
   $$w[\text{source}] = \sum_{k=0}^{M-1} \left( \Re\{W[k]\}\cos\!\left(-\tfrac{2\pi\,\text{source}\,k}{M}\right) - \Im\{W[k]\}\sin\!\left(-\tfrac{2\pi\,\text{source}\,k}{M}\right) \right),$$
   其中 $W[k]$ 由 `host_chebyshev_polynomial` 给出,即 Chebyshev 多项式 $T_{M-1}(\beta\cos(\pi k/M))$ 配合奇偶长度相位因子。
6. peak 归一化:`out[i] = raw[i] / peak`,`peak = max(raw)`。
7. sym=false 截断:`out` 仅写入前 `length` 项,effective 多出来的最后一项不进入输出。

#### 3.4 host helper 语义

- `host_chebyshev_polynomial(frequency, length, beta, real, imag)`(`windows_typed.cpp:47-72`):按奇偶长度分两支求频域采样。
  - $x = \beta \cos(\pi f/M)$。
  - $x > 1$:`cosh((M-1)·acosh(x))`(典型高频旁瓣区域)。
  - $x < -1$:`sign·cosh((M-1)·acosh(-x))`,`sign = (M 奇 ? 1 : -1)`。
  - $|x| \le 1$:`cos((M-1)·acos(x))`(主瓣及过渡带)。
  - 奇长度:`real = value, imag = 0`;偶长度:`real = value·cos(angle), imag = value·sin(angle)`。
- `host_chebyshev_raw(index, length, beta)`(`windows_typed.cpp:74-89`):对 `frequency = 0..length-1` 累加 `real·cos(angle) - imag·sin(angle)`,`angle = -2π·index·frequency/length`。这是显式 IDFT 实部求和,**不调用任何 FFT 库**。
- `host_chebyshev_raw_index(output_index, length)`(`windows_typed.cpp:91-103`):把以中心为 0 的对称输出索引映射到 DFT 源索引。奇长度:`center = (M+1)/2 - 1`,左半 `center-i`、右半 `i-center`;偶长度:`half = M/2`,左半 `half-i`、右半 `i-half+1`。

### 4. GPU 调用链、host wrapper、kernel 启动与 device 计算逻辑

#### 4.1 顶层 host wrapper

入口:`chebwin_device(length, attenuation, out, symmetric)`(`windows_typed.cpp:230-244`)。

```
chebwin_device(length, attenuation, out, symmetric)
  ├─ 守卫:length < 0 || out.size() != length → throw std::invalid_argument
  │                                                  (windows_typed.cpp:234-236)
  ├─ computed: DeviceArray<float>(out.size())        (windows_typed.cpp:237)
  ├─ length == 1 → computed = DeviceArray<float>::from_host({1.0F})
  │                                                  (windows_typed.cpp:238-239)
  ├─ length > 1 → chebwin_fp32_compute_device(length, attenuation, computed, symmetric)
  │                                                  (windows_typed.cpp:240-241)
  └─ out = cuda_utils::finalize_fp64_device_storage(computed)  [FP32→FP64 拓宽]
                                                     (windows_typed.cpp:243)
```

- `finalize_fp64_device_storage` 由 `cuda_utils/host_output_finalize.h` 提供,在内部包含 D2D 拷贝(FP32→FP64)与必要的 D2H/H2D。具体行为以 `cuda_utils` 实现为准。
- `length == 0` 时 `computed` 为空,后续 finalize 写回空 `out`。

#### 4.2 GPU host 调度层

入口:`chebwin_fp32_compute_device(n, attenuation, output, symmetric)`(`windows_typed.cu:14-72`)。

```
chebwin_fp32_compute_device(n, attenuation, output, symmetric)
  ├─ 守卫:n < 1 || output.size() != n → throw std::invalid_argument  (windows_typed.cu:18-21)
  ├─ effective = symmetric ? n : n + 1                              (windows_typed.cu:22)
  ├─ #ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN → GPU FFT 路径(默认不启用)
  │     ├─ DeviceArray<ComplexFloat> spectrum, transformed; DeviceArray<float> reordered
  │     ├─ launch_1d_kernel(chebwin_spectrum_kernel<T>, effective, attenuation, spectrum)
  │     ├─ FFTInterface(effective, BatchOnly{}).fft_batch_device(spectrum, transformed, 1)
  │     ├─ launch_1d_kernel(chebwin_reorder_kernel, transformed, effective, reordered)
  │     ├─ thrust::max_element(reordered) → peak
  │     └─ launch_1d_kernel(normalize_window_kernel<float,float>, reordered, n, peak, output)
  └─ #else(默认 GPU direct DFT 路径)
        ├─ DeviceArray<float> raw(effective)
        ├─ launch_1d_kernel(chebwin_raw_kernel<T, float>, effective, attenuation, raw)
        │                                                  (windows_typed.cu:53-59)
        ├─ thrust::max_element(raw) → peak                (windows_typed.cu:60-63)
        └─ launch_1d_kernel(normalize_window_kernel<float, float>, raw, n, peak, output)
                                                          (windows_typed.cu:64-70)
```

#### 4.3 kernel 启动机制

- 所有 kernel 通过 `cuda_utils::launch_1d_kernel(...)` 启动,该工具由 `cuda_utils` 提供,自动按 `size` 计算 grid/block。
- 默认 stream 上启动,无显式 stream 参数,无显式 `cudaDeviceSynchronize`(依赖 Thrust reduction 与 finalize 内部的隐式同步语义)。
- `thrust::max_element` 在默认 stream 上做 reduction,返回标量 `peak` 拷回 host。

#### 4.4 device kernel 计算逻辑

##### 4.4.1 `chebwin_raw_kernel`(`windows_kernels.cuh:256-274`)

每线程算一个 effective 长度上的采样点。

```cpp
template <typename Parameter, typename Scalar>
__global__ void chebwin_raw_kernel(
    int effective_count, Parameter attenuation_input, Scalar* raw) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= effective_count) return;
    if (effective_count == 1) { raw[index] = 1; return; }
    const Scalar attenuation = window_load(attenuation_input);
    const Scalar beta = cosh(acosh(pow(10, fabs(attenuation)/20)) / (effective_count - 1));
    raw[index] = chebyshev_raw_value(index, effective_count, beta);
}
```

- `index` 即 effective 长度上的输出索引。
- `attenuation` 通过 `window_load` 从 5 种类型统一转 FP32 标量。
- `beta` 公式与 CPU 一致(但 FP32 精度)。
- 调用 `chebyshev_raw_value` 完成单点 IDFT 求和。

##### 4.4.2 `chebyshev_raw_value`(`windows_kernels.cuh:232-254`)

```cpp
template <typename Scalar>
__device__ inline Scalar chebyshev_raw_value(
    int output_index, int length, Scalar beta) {
    const Scalar pi = 3.14159265358979323846F;
    const int source = chebyshev_raw_index(output_index, length);  // 对称中心化映射
    const bool odd = (length & 1) != 0;
    Scalar sum = 0;
    for (int frequency = 0; frequency < length; ++frequency) {
        const Scalar real = chebyshev_spectrum_real(frequency, length, beta);
        const Scalar phase = pi * frequency / length;
        const Scalar spectrum_real = odd ? real : real * cos(phase);
        const Scalar spectrum_imag = odd ? 0 : real * sin(phase);
        const Scalar angle = -2 * pi * source * frequency / length;
        sum += spectrum_real * cos(angle) - spectrum_imag * sin(angle);
    }
    return sum;
}
```

- 与 CPU `host_chebyshev_raw` 数学等价,差别仅在:(a)Chebyshev 多项式用 `chebyshev_spectrum_real` 而非 `host_chebyshev_polynomial` 的解析分支;(b) FP32 而非 FP64。
- 内层 `frequency` 循环长度为 `length`,即每个线程串行求 `length` 项 DFT 和。复杂度 $O(M)$ 每线程,$O(M^2)$ 整体。

##### 4.4.3 `chebyshev_spectrum_real`(`windows_kernels.cuh:196-216`)

```cpp
template <typename Scalar>
__device__ inline Scalar chebyshev_spectrum_real(
    int frequency, int length, Scalar beta) {
    const Scalar angle = pi * frequency / length;
    const Scalar x = beta * cos(angle);
    const int order = length - 1;
    if (order <= 0) return 1;
    Scalar previous = 1;       // T_0(x)
    Scalar current = x;        // T_1(x)
    for (int index = 2; index <= order; ++index) {
        const Scalar next = 2 * x * current - previous;  // T_k = 2x T_{k-1} - T_{k-2}
        previous = current;
        current = next;
    }
    return current;
}
```

- **Clenshaw 递推**:用三项递推 $T_{k+1}(x) = 2x\,T_k(x) - T_{k-1}(x)$ 从 $T_0=1, T_1=x$ 求到 $T_{M-1}$。
- 与 Python `cusignal/windows/windows.py:1516, 1518-1519, 1521` 的 `cosh(order·acosh(x))` / `cos(order·acos(x))` 分支解析公式**数学等价**,但实现方式不同。Clenshaw 在 $|x|>1$ 与 $|x|\le 1$ 区间不分支,统一按递推求值,避免 `acosh`/`acos` 在临界点附近的浮点不稳定性,但累积舍入不同。

##### 4.4.4 `chebyshev_raw_index`(`windows_kernels.cuh:218-230`)

device inline,与 CPU `host_chebyshev_raw_index` 完全一致的对称中心化索引映射。

##### 4.4.5 `normalize_window_kernel`(`windows_kernels.cuh:359-370`)

```cpp
template <typename Scalar, typename T>
__global__ void normalize_window_kernel(
    const Scalar* raw, int output_count, Scalar peak, T* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const Scalar value = peak > 0 ? raw[index] / peak : raw[index];
    output[index] = window_store<T>(value);
}
```

- `output_count = n`(用户请求长度),`raw` 长度为 `effective`。sym=false 时只归一化前 `n` 项,丢弃 effective 末项,完成截断。
- `peak > 0` 守卫避免 0 除。

##### 4.4.6 FFT 路径 kernel(默认不启用)

- `chebwin_spectrum_kernel`(`windows_kernels.cuh:277-330`):用 half-angle 形式 $\cosh(u)\cos(v)-1 = 2(\sinh^2(u/2)-\sin^2(v/2)) - 4\sinh^2(u/2)\sin^2(v/2)$ 直接计算 $|x|-1$,避免在 $\beta\approx 1$、$\cos(v)\approx 1$ 时的 FP32 cancellation。仅当编译时定义 `CUSIGNAL_WINDOWS_FFT_CHEBWIN` 才使用。
- `chebwin_reorder_kernel`(`windows_kernels.cuh:332-349`):把 FFT 输出按奇偶长度重排到对称中心化顺序,取 `.re` 作为实窗。

### 5. template、类型分发、显式实例化、内存布局与数据搬运

#### 5.1 模板参数 `<class T>`

`T` 表示 attenuation 输入类型,在显式实例化处展开为 5 种:

1. `float`
2. `__half`
3. `std::int32_t`
4. `std::int16_t`
5. `std::int8_t`

#### 5.2 类型转换

- host 端:`host_load<T>(value)`(`windows_typed.cpp:17-21`)调用 `detail::SimpleSignalTypePolicy<T>::load(value)` 统一转 `float`(CPU 内部再 `static_cast<double>` 到 FP64)。
- device 端:`window_load<T>(value)`(`windows_kernels.cuh:10-18`)走同一 `SimpleSignalTypePolicy<T>::load`,在 kernel 内得到 FP32 标量。

#### 5.3 显式实例化

- CPU/GPU host 层:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cpp` / `INSTANTIATE_WINDOWS_CPU` 宏 / `windows_typed.cpp:562-595`,其中 5 种类型实例化在 `windows_typed.cpp:589-593`。
- GPU compute 层:`1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cu` / `INSTANTIATE` 宏 / `windows_typed.cu:116-127`,5 种类型实例化在 `windows_typed.cu:126`。

#### 5.4 device 计算与正式输出 dtype 解耦

- device 计算:**FP32**(`DeviceArray<float> computed`, `raw`, `spectrum`, `transformed`, `reordered`)。
- 正式输出:**FP64**(`DeviceArray<double> out`),由 `finalize_fp64_device_storage` 拓宽。
- 设计意图:device kernel 用 FP32 节省显存与算力;输出 FP64 对齐 cuSignal Python 的 `float64` 契约。精度差异在测试容差 4e-2 内可接受。

#### 5.5 内存布局与数据搬运

| 缓冲区 | 位置 | dtype | 长度 | 持有方 | 生命周期 |
| --- | --- | --- | --- | --- | --- |
| `out`(API 参数) | device | `double` | `length` | 调用者 | 跨调用 |
| `computed` | device | `float` | `length` | `chebwin_device` | 单次调用 |
| `raw`(direct 路径) | device | `float` | `effective` | `chebwin_fp32_compute_device` | 单次调用 |
| `spectrum` / `transformed` / `reordered`(FFT 路径) | device | `ComplexFloat`/`ComplexFloat`/`float` | `effective` | `chebwin_fp32_compute_device` | 单次调用 |
| `peak` | host scalar | `float` | 1 | `chebwin_fp32_compute_device` | 单次调用 |

- 所有 FP32 临时数组都是**函数内部 workspace**,不对调用者暴露,不跨调用复用。
- 数据搬运链:`chebwin_raw_kernel` 写 `raw` → `thrust::max_element` 在 device 上 reduction 得 `peak`(D2H 标量)→ `normalize_window_kernel` 读 `raw`+`peak` 写 `computed` → `finalize_fp64_device_storage(computed)` D2D 拷贝拓宽为 FP64 写回 `out`。

### 6. grid/block、线程索引与每线程数据范围

#### 6.1 启动模板

`cuda_utils::launch_1d_kernel(kernel, size, args...)` 由 `cuda_utils` 内部根据 `size` 自动计算 grid/block。源码未显式指定 `<<<grid, block>>>`,因此具体 block 大小以 `cuda_utils` 实现为准(本提交未展开该工具源码,标为推测)。

#### 6.2 线程索引约定

所有 chebwin 相关 kernel 统一使用:

```cpp
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= effective_count) return;   // 或 output_count
```

- `index` 即 1D 输出位置,与 effective/output 长度对齐。
- 没有线程内并行 reduction;每线程独立完成 1 个输出点的全部计算。

#### 6.3 每线程数据范围

| kernel | 输出长度 | 每线程负责 | 每线程串行工作量 |
| --- | --- | --- | --- |
| `chebwin_raw_kernel` | `effective` | 1 个 raw 采样点 | `length` 次 `chebyshev_spectrum_real` + 累加,$O(M)$ |
| `normalize_window_kernel` | `n` | 1 个归一化输出 | $O(1)$ |
| `chebwin_spectrum_kernel`(FFT 路径) | `effective` | 1 个频域 bin | $O(1)$(half-angle 解析式) |
| `chebwin_reorder_kernel`(FFT 路径) | `effective` | 1 个重排输出 | $O(1)$ |

- direct 路径整体复杂度:$O(M^2)$,与 CPU `host_chebyshev_raw` 双层循环一致。
- FFT 路径整体复杂度:$O(M\log M)$(FFT) + $O(M)$ 频谱构造与重排,但默认不启用。

### 7. 同步、临时缓冲区、FFT/平台库调用与错误处理

#### 7.1 同步

- 所有 kernel 在默认 stream 启动,无显式 stream 参数。
- `thrust::max_element` 在默认 stream 上 reduction;host 端读 `peak` 隐式等待 reduction 完成。
- `finalize_fp64_device_storage` 内部包含必要的同步(以 `cuda_utils` 实现为准)。
- 测试调用端(`e3_wave_window_type_smoke.cu:441-473`)在每次 `chebwin_device` 之后立即 `to_host()`,因此实测序列化等价于隐式同步。

#### 7.2 临时缓冲区

- direct 路径:`raw`(FP32, 长度 `effective`)、`computed`(FP32, 长度 `length`)、`out`(FP64, 长度 `length`,调用者持有)。
- FFT 路径:额外 `spectrum`、`transformed`(均为 `ComplexFloat`, 长度 `effective`)、`reordered`(FP32, 长度 `effective`)。
- 全部为函数内部 workspace,调用结束随 RAII 释放,不跨调用复用。

#### 7.3 FFT / 平台库调用

- **默认路径不调用任何 FFT 库**(不使用 `dlfft`、不使用 `cuFFT`)。direct DFT 显式求和。
- 仅当编译时定义 `CUSIGNAL_WINDOWS_FFT_CHEBWIN`:
  - CPU 走 `FFTInterface_cpu(effective).fft(spectrum)`(`windows_typed.cpp:196`)。
  - GPU 走 `FFTInterface(effective, FFTInterface::BatchOnly{}).fft_batch_device(spectrum, transformed, 1)`(`windows_typed.cu:33-34`)。
  - `FFTInterface` 是项目内封装,不直接等同于 cuSignal Python 的 `cp.fft.fft`,但数学等价。

#### 7.4 错误处理

| 错误条件 | 抛出/行为 | 位置 |
| --- | --- | --- |
| `length < 0`(`chebwin_device`) | `std::invalid_argument` | `windows_typed.cpp:234-236` |
| `out.size() != length`(`chebwin_device`) | `std::invalid_argument` | `windows_typed.cpp:234-236` |
| `length < 0`(`chebwin_typed_cpu`) | `std::invalid_argument` | `windows_typed.cpp:166` |
| `n < 1` 或 `output.size() != n`(`chebwin_fp32_compute_device`) | `std::invalid_argument` | `windows_typed.cu:18-21` |
| `length == 0` | 返回空,不抛 | `windows_typed.cpp:167` / `windows_typed.cpp:237-243` |
| `length == 1` | `computed = {1.0F}`,跳过 compute_device | `windows_typed.cpp:238-239` / `windows_typed.cpp:168` |
| CUDA kernel 启动失败 / Thrust 失败 | 走 `cuda_utils` 项目异常机制(本提交未展开具体异常类型,标为推测) | — |

### 8. 数学/物理原理 → CPU 代码 → GPU 代码 三方映射表

阶段一原理编号见 `operator_comprehension/Learning/operators/windows/chebwin/数学物理原理.md`。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 原理 1:Chebyshev 多项式(第一类)$T_n(x)$ 定义与递推 $T_{k+1}=2xT_k-T_{k-1}$ | `cusignal-23.08.00/cusignal/windows/windows.py:1516, 1518-1519, 1521`(`_chebwin_kernel` 内 `cosh(order*acosh(x))` / `cos(order*acos(x))` 分支) | `1ccd32ea` / `cusignal_cpp/src/windows/windows_typed.cpp` / `host_chebyshev_polynomial` / `windows_typed.cpp:47-72`(解析分支 `cosh`/`cos` + 奇偶长度相位因子) | `1ccd32ea` / `cusignal_cpp/src/windows/windows_kernels.cuh` / `chebyshev_spectrum_real` / `windows_kernels.cuh:196-216`(Clenshaw 递推 T_0=1, T_1=x, T_{k+1}=2xT_k-T_{k-1},不分 $|x|>1$ 与 $|x|\le 1$ 分支) | CPU 与 Python 数学等价(都按 $|x|$ 分支用解析式);GPU 改用 Clenshaw 递推统一求值,数学等价但累积舍入不同,FP32 精度低于 Python 的 FP64 |
| 原理 2:Dolph-Chebyshev 窗频域定义 $W[k] = T_{M-1}(\beta\cos(\pi k/M))$ 配相位因子 | `windows.py:1640-1646`(构造 spectrum,`beta = np.cosh(...)`) | `1ccd32ea` / `windows_typed.cpp` / `host_chebyshev_polynomial` + `host_chebyshev_raw` / `windows_typed.cpp:47-89`(显式 spectrum 实部/虚部 + IDFT 求和) | `1ccd32ea` / `windows_kernels.cuh` / `chebyshev_spectrum_real` + `chebyshev_raw_kernel` / `windows_kernels.cuh:196-274` | 三者数学等价;cusignal_cpp 默认不走 FFT,直接显式 DFT 求和 |
| 原理 3:从频域到时域的转换(IDFT)$w[n] = \frac{1}{M}\sum_{k} W[k] e^{j2\pi nk/M}$ | `windows.py:1646, 1651`(`cp.fft.fft(p)` 取实部,等价于逆变换的实数形式) | `1ccd32ea` / `windows_typed.cpp` / `host_chebyshev_raw` / `windows_typed.cpp:74-89`(`sum += real*cos(angle) - imag*sin(angle)`,`angle = -2π·index·frequency/length`) | `1ccd32ea` / `windows_kernels.cuh` / `chebyshev_raw_value` / `windows_kernels.cuh:232-254`(同样显式 DFT 求和) | cusignal_cpp **不调用 IFFT/FFT**,而是显式实数 DFT 公式;Python 直接 `cp.fft.fft`;数学等价,实现不同 |
| 原理 4:minimax 最优性(等纹波 Chebyshev 设计) | `windows.py:1640`(由 `beta = cosh(acosh(R)/(M-1))` 隐式保证等纹波性) | `1ccd32ea` / `windows_typed.cpp` / `chebwin_typed_cpu` 内 `beta = std::cosh(std::acosh(std::pow(10, std::fabs(at)/20)) / (effective-1))` / `windows_typed.cpp:173-174` | `1ccd32ea` / `windows_kernels.cuh` / `chebwin_raw_kernel` 内 `beta = cosh(acosh(pow(10, fabs(at)/20)) / (effective_count-1))` / `windows_kernels.cuh:269-272` | 三者公式一致,minimax 最优性由 $\beta$ 公式直接保证;无显式等纹波约束求解 |
| 原理 5:参数 $\beta$ 与旁瓣衰减的换算 $\beta = \cosh\!\left(\dfrac{\operatorname{acosh}(10^{|at|/20})}{M-1}\right)$ | `windows.py:1640`(基准) | `1ccd32ea` / `windows_typed.cpp` / `chebwin_typed_cpu` / `windows_typed.cpp:173-174` | `1ccd32ea` / `windows_kernels.cuh` / `chebwin_raw_kernel` / `windows_kernels.cuh:269-272` | 公式三处一致;CPU 用 FP64,GPU 用 FP32;Python 与 CPU 都对衰减取绝对值( cusignal_cpp 在 `fabs(attenuation_value)`,Python 在 `np.abs`) |
| 原理 6:频域等纹波导致时域端点脉冲(窗两端非零) | `windows.py:1655-1657`(归一化后 `_truncate`,端点由 spectrum 决定) | `1ccd32ea` / `windows_typed.cpp` / `chebwin_typed_cpu` 归一化 `out[i] = raw[i] / peak` / `windows_typed.cpp:211-214, 222-225` | `1ccd32ea` / `windows_kernels.cuh` / `normalize_window_kernel` / `windows_kernels.cuh:359-370` | 三者归一化方式数学等价;cusignal_cpp **不警告 `\|at\| < 45 dB`**(Python 在 `windows.py:1625-1633` 警告),这是工程差异,不改变数学 |
| 对称中心化索引重排(`_truncate` 等价的 sym=false 截断 + 对称重排) | `windows.py:1647-1653`(`cp.concatenate` 重排) / `windows.py:1657`(`_truncate`) | `1ccd32ea` / `windows_typed.cpp` / `host_chebyshev_raw_index` / `windows_typed.cpp:91-103`(对称中心化索引映射) + `effective = sym ? n : n+1` 截断 `windows_typed.cpp:169` | `1ccd32ea` / `windows_kernels.cuh` / `chebyshev_raw_index` / `windows_kernels.cuh:218-230` + `normalize_window_kernel` 输出长度 `n` 截断 | CPU/GPU 把对称重排内嵌到索引函数;Python 在 host 端 `cp.concatenate`;数学等价,实现位置不同 |
| 归一化使峰值=1 | `windows.py:1655`(`w = w / cp.max(w)`) | `1ccd32ea` / `windows_typed.cpp` / `chebwin_typed_cpu` / `windows_typed.cpp:210, 217-225`(`std::max_element` + 除法) | `1ccd32ea` / `windows_typed.cu` / `thrust::max_element` / `windows_typed.cu:60-63` + `normalize_window_kernel` / `windows_kernels.cuh:359-370` | CPU 用 `std::max_element`,GPU 用 `thrust::max_element`,Python 用 `cp.max`;三者数学等价 |
| sym=false 时的 `M+1` 扩展 + 截断 | `windows.py:1657`(`_truncate`,先 `_extend` 到 `M+1` 再截回 `M`) | `1ccd32ea` / `windows_typed.cpp` / `effective = symmetric ? length : length+1` + `out` 只写前 `length` 项 / `windows_typed.cpp:169, 211-214, 222-225` | `1ccd32ea` / `windows_typed.cu` / `effective = symmetric ? n : n+1` + `normalize_window_kernel` output_count=`n` / `windows_typed.cu:22, 64-70` | cusignal_cpp 不显式 `_extend`/`_truncate`,而是让 kernel 直接计算 `effective` 长度,再由输出长度截断;数学等价 |

### 9. 与 cuSignal Python 实现相同、等价替换和有意不同的部分

#### 9.1 完全相同

- $\beta$ 公式:`cosh(acosh(10^|at|/20)/(M-1))`,CPU/GPU/Python 三处一致。
- 长度守卫语义:`n<0` 抛异常,`n==0` 返回空,`n==1` 返回 `{1.0}`。
- sym=false 时 `effective = n+1` 再截前 `n` 个的语义。
- 输出归一化使峰值为 1。
- 对衰减取绝对值,接受负衰减输入。

#### 9.2 数学等价、实现不同

| 项 | Python | cusignal_cpp | 说明 |
| --- | --- | --- | --- |
| Chebyshev 多项式求值 | `cosh(order*acosh(x))` / `cos(order*acos(x))` 分支(`windows.py:1516, 1518-1519, 1521`) | CPU:同样解析分支(`host_chebyshev_polynomial`);GPU:Clenshaw 递推(`chebyshev_spectrum_real`) | GPU 改递推避免分支跳转与临界点不稳定,但 FP32 累积舍入与 Python FP64 不同 |
| 频域 → 时域转换 | `cp.fft.fft(p)`(`windows.py:1646, 1651`) | CPU/GPU:显式实数 DFT 求和 `sum += real*cos(angle) - imag*sin(angle)`(`host_chebyshev_raw` / `chebyshev_raw_value`) | cusignal_cpp 默认 **不调用 FFT 库**,直接 $O(M^2)$ 求和;数学等价 |
| 归一化 max | `cp.max(w)`(`windows.py:1655`) | CPU:`std::max_element`;GPU:`thrust::max_element` | reduction 实现不同,数学等价 |
| 对称重排 | host 端 `cp.concatenate`(`windows.py:1647-1653`) | kernel 内 `chebyshev_raw_index` / `host_chebyshev_raw_index` 索引映射 | 重排位置不同,数学等价 |
| sym=false 扩展+截断 | `_extend` + `_truncate`(`windows.py:1657`) | `effective = n+1` 后输出长度截断 | 不显式 extend/truncate,数学等价 |

#### 9.3 有意不同(工程决策)

1. **默认走 custom direct DFT,不使用 `dlfft`/`cuFFT`**:见 `chebwin_fp32_compute_device` 默认 `#else` 分支(`windows_typed.cu:52-71`)。注释说明(见 `windows_typed.h:33-36`):为 ZQ500 平台高衰减稳定性使用 custom direct DFT,而不是宣称通过 `dlfft`。`CUSIGNAL_WINDOWS_FFT_CHEBWIN` 是可选编译开关。
2. **GPU 用 FP32 计算 + FP64 输出,Python 全程 FP64**:`chebwin_device` 分配 `DeviceArray<float> computed`,`finalize_fp64_device_storage` 拓宽为 `DeviceArray<double> out`(`windows_typed.cpp:237, 243`)。精度差异由测试容差 4e-2 兜底。
3. **不警告 |at| < 45 dB**:Python 在 `cusignal-23.08.00/cusignal/windows/windows.py:1625-1633` 警告低衰减,cusignal_cpp 完全省略。工程差异,不改变数学。
4. **GPU Chebyshev 求值用 Clenshaw 递推**:见 `chebyshev_spectrum_real`(`windows_kernels.cuh:196-216`),三项递推统一求值,不区分 $|x|>1$ 与 $|x|\le 1$;Python 与 CPU `host_chebyshev_polynomial` 都按分支解析公式。数学等价,精度略不同。
5. **不显式 stream 参数**:GPU 全程默认 stream,无 stream 调度灵活性。
6. **临时缓冲区为函数内部 workspace**:`computed`、`raw`、`spectrum` 等都不对调用者暴露,不跨调用复用(见 `windows_typed.h:36` 注释)。

### 10. 精度、dtype、边界条件及 ZQ500 平台限制

#### 10.1 精度

| 路径 | 计算精度 | 输出精度 | 备注 |
| --- | --- | --- | --- |
| CPU `chebwin_typed_cpu` | FP64 | FP64 | 与 Python `float64` 同精度,作为 reference |
| GPU `chebwin_device`(direct) | FP32 | FP64 | device kernel 内 `chebwin_raw_kernel`、`chebyshev_spectrum_real`、`normalize_window_kernel` 均为 FP32;`finalize_fp64_device_storage` 拓宽 |
| GPU FFT 路径(默认未启用) | FP32 + half-angle 稳定化 | FP64 | `chebwin_spectrum_kernel` 用 half-angle 避免 $\beta\approx 1$ 时的 FP32 cancellation |

#### 10.2 dtype 分发

- 5 种 attenuation 输入类型:`float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t`。
- 输出固定 `DeviceArray<double>`(GPU)/ `std::vector<double>`(CPU)。
- 类型转换统一走 `detail::SimpleSignalTypePolicy<T>::load` / `::store`,不暴露给算子层。

#### 10.3 边界条件

| 条件 | CPU 行为 | GPU 行为 |
| --- | --- | --- |
| `length < 0` | throw `std::invalid_argument`(`windows_typed.cpp:166`) | throw `std::invalid_argument`(`windows_typed.cpp:234-236`) |
| `length == 0` | 返回 `{}`(`windows_typed.cpp:167`) | `computed` 空,finalize 写回空 `out`(`windows_typed.cpp:237-243`) |
| `length == 1` | 返回 `{1.0}`(`windows_typed.cpp:168`) | `computed = {1.0F}`,跳过 compute_device(`windows_typed.cpp:238-239`) |
| `length == 2` | `effective = 2`,`host_chebyshev_raw` 单层求和 | `chebwin_raw_kernel` 启动,每线程 2 项累加 |
| `attenuation == 0` | $\beta = \cosh(0/(M-1)) = 1$,退化为矩形窗类似形状 | 同上,FP32 |
| `attenuation < 0` | `fabs` 取绝对值,与正值同结果 | 同上 |
| `sym == false` | `effective = length+1`,输出截前 `length` 项 | 同上 |
| `out.size() != length`(GPU) | — | throw(`windows_typed.cpp:234-236`) |

#### 10.4 ZQ500 平台限制(推测 + 已知)

- 默认走 custom direct DFT,避免依赖 `dlfft`/`cuFFT` 的可用性与精度行为(见 `windows_typed.h:33-36` 注释)。
- GPU FP32 计算适合 ZQ500 的 FP32 算力;FP64 拓宽仅在 finalize 一次性发生,避免 FP64 kernel 全程开销。
- ZQ500 平台每天 01:00-08:00 可能维护,不适合长窗(>10^4)实验(推测,基于 `AGENTS.md` 平台维护说明)。
- 高衰减(>100 dB)在 FP32 + Clenshaw 递推下可能精度不足,FFT 路径(`CUSIGNAL_WINDOWS_FFT_CHEBWIN` + half-angle)是工程备选,但默认未启用(推测)。

### 11. 测试如何验证正确性,以及仍未覆盖的风险

#### 11.1 测试入口

- 文件:`1ccd32ea` / `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` / `run_type<T>` 内 `ok[6]` / `e3_wave_window_type_smoke.cu:440-473`。
- 在 `test_evidence::f_begin_accuracy(type_dispatch::OperatorId::chebwin)` 之后执行,5 种 dtype 各跑一遍。

#### 11.2 覆盖场景

| 行号 | 调用 | 期望 |
| --- | --- | --- |
| `e3_wave_window_type_smoke.cu:441-444` | `chebwin_device(5, convert<T>(60), fixed_fp64_window_out)` 与 `chebwin_typed_cpu(5, convert<T>(60))` 比对 | `matches_fp64` 通过 |
| `:445-448` | `chebwin_device(5, convert<T>(-60), ...)` 与 CPU 比对 | 测 `abs` 等价,通过 |
| `:449` | `chebwin_device<T>(1, convert<T>(0), singleton_fp64_window_out)` | 输出 `{1.0}` |
| `:450` | `chebwin_device<T>(0, convert<T>(0), empty_fp64_window_out)` | 输出空 |
| `:453-457` | `chebwin_device<T>(-1, convert<T>(60), ...)` | 抛 `std::invalid_argument` |
| `:458-461` | `chebwin_typed_cpu<T>(-1, convert<T>(60))` | 抛 `std::invalid_argument` |
| `:462-467` | `chebwin_device<T>(4, convert<T>(60), chebwin_periodic_out, false)` 与 `chebwin_typed_cpu(4, convert<T>(60), false)` 比对 | sym=false 周期窗一致 |
| `:468-469` | 单点 CPU reference | `{1.0}` |
| `:470-471` | 空窗 CPU reference | 空 |

#### 11.3 容差

- `matches_fp64` 使用 `test_evidence::f_observe_real(actual, reference, 4.0e-2)`,即 **4e-2 绝对误差**。
- 该容差为 GPU FP32 与 CPU FP64 之间留足余量,也吸收 Clenshaw 递推与解析公式的累积差异。

#### 11.4 CPU/GPU 对照

- 每个 GPU 调用同时调用对应 CPU reference,验证一致性。
- 验证维度:5 种 dtype(fp32/fp16/int32/int16/int8)各跑一遍,共 70 项(14 算子 × 5 类型),chebwin 贡献 1 项 × 5 类型 = 5 个 `ok[6]` 检查。

#### 11.5 未覆盖的风险

1. **高衰减(>100 dB)**:测试只覆盖 60 dB;FP32 + Clenshaw 递推在高衰减时 $\beta$ 极接近 1,$|x|-1$ 极小,FP32 cancellation 风险高,未验证。
2. **超长窗(>10^4)**:测试只覆盖 `n=5` 与 `n=4`;direct DFT $O(M^2)$ 在长窗下耗时与显存压力大,未验证。
3. **FFT 路径**:默认未编译 `CUSIGNAL_WINDOWS_FFT_CHEBWIN`,FFT 路径(`chebwin_spectrum_kernel`、`chebwin_reorder_kernel`、`FFTInterface` 调用)无测试覆盖。
4. **精度边界**:容差 4e-2 较宽,可能掩盖中等衰减(80-100 dB)下的精度退化。
5. **`__half`/`int8` 输入的衰减量化**:5 种 dtype 中 `__half` 与 `int8` 表示 60 dB 时精度差异未单独分析。
6. **sym=false 长窗截断**:仅 `n=4` 测试,未覆盖 sym=false 长窗下 `effective = n+1` 与输出长度 `n` 的边界对齐。
7. **kernel 启动失败路径**:`cuda_utils::launch_1d_kernel` 失败、`thrust::max_element` 失败的异常传播未在 chebwin 测试中显式触发。
8. **默认 stream 隐式同步**:依赖 `to_host()` 触发的隐式同步,多算子并发或自定义 stream 场景未覆盖。

### 12. 自检问题

参考阶段一/二风格,用于学习自测:

1. `chebwin_device` 在 `length == 1` 时为什么不调用 `chebwin_fp32_compute_device`?如果调用会发生什么?
2. `effective = symmetric ? length : length + 1` 这一行在 CPU 和 GPU 哪里出现?sym=false 时多出来的最后一项在哪里被丢弃?
3. `host_chebyshev_polynomial`(`windows_typed.cpp:47-72`)按 $|x|$ 分三支求值,而 GPU `chebyshev_spectrum_real`(`windows_kernels.cuh:196-216`)用 Clenshaw 递推不分支。这两种方式在 $x>1$ 与 $|x|\le 1$ 区间的数学结果分别是什么?为什么 GPU 选 Clenshaw?
4. `chebyshev_raw_value`(`windows_kernels.cuh:232-254`)的内层 `frequency` 循环长度是多少?整个 kernel 的复杂度是多少?为什么不拆成多个 kernel?
5. `normalize_window_kernel` 的 `output_count` 是 `n` 而不是 `effective`,这在 sym=false 时起到什么作用?
6. cusignal_cpp 默认不调用任何 FFT 库,但代码里有 `#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN` 分支。为什么默认走 direct DFT?ZQ500 平台因素是什么?
7. GPU 用 FP32 计算 + FP64 输出,Python 全程 FP64。容差 4e-2 是否足以覆盖 Clenshaw 递推 + FP32 在 60 dB 下的累积误差?在 >100 dB 时还够吗?
8. `thrust::max_element` 在 GPU 上做 reduction,返回 host 标量 `peak`。这一步会引入隐式同步吗?对后续 `normalize_window_kernel` 启动有什么影响?
9. `host_chebyshev_raw_index`(`windows_typed.cpp:91-103`)与 `chebyshev_raw_index`(`windows_kernels.cuh:218-230`)的代码是否完全一致?为什么 CPU 和 GPU 各写一份?
10. 阶段一原理 5 的 $\beta$ 公式在三处(Python `windows.py:1640`、CPU `windows_typed.cpp:173-174`、GPU `windows_kernels.cuh:269-272`)完全一致,但 Python 在 `windows.py:1625-1633` 警告 `|at| < 45`,cusignal_cpp 不警告。这是否影响数学等价性?是否影响工程可用性?
11. 5 种 dtype 中 `__half` 与 `std::int8_t` 在表示 60 dB 时精度差异多大?对 $\beta$ 计算的影响在测试容差内吗?
12. 如果未来启用 `CUSIGNAL_WINDOWS_FFT_CHEBWIN`,GPU 路径会调用 `FFTInterface(effective, BatchOnly{}).fft_batch_device`。这与 Python `cp.fft.fft(p)` 在数学上等价吗?half-angle 稳定化(`chebwin_spectrum_kernel` 注释)解决的是什么精度问题?

### 13. 阅读顺序建议

按以下顺序阅读源码,可在不丢失依赖的情况下完整理解 chebwin 复现:

1. **接口**:`cusignal_cpp/src/windows/windows_typed.h:21-40`(`chebwin_typed_cpu` 与 `chebwin_device` 声明,以及 docstring 中的 sym、attenuation、长度守卫语义)。
2. **CPU reference**:`cusignal_cpp/src/windows/windows_typed.cpp:163-228`(`chebwin_typed_cpu`),先读默认 `#else` direct DFT 路径,再读 `#ifdef CUSIGNAL_WINDOWS_FFT_CHEBWIN` 路径作对照;配套读匿名命名空间内的 `host_chebyshev_polynomial`(`:47-72`)、`host_chebyshev_raw`(`:74-89`)、`host_chebyshev_raw_index`(`:91-103`)。
3. **GPU host wrapper**:`cusignal_cpp/src/windows/windows_typed.cpp:230-244`(`chebwin_device`),关注 `computed` 的 FP32 分配与 `finalize_fp64_device_storage` 拓宽。
4. **GPU host 调度**:`cusignal_cpp/src/windows/windows_typed.cu:14-72`(`chebwin_fp32_compute_device`),先读默认 direct 路径,再读 FFT 路径。
5. **GPU kernel**:`cusignal_cpp/src/windows/windows_kernels.cuh`,按 `chebwin_raw_kernel`(`:256-274`)→ `chebyshev_raw_value`(`:232-254`)→ `chebyshev_raw_index`(`:218-230`)→ `chebyshev_spectrum_real`(`:196-216`)→ `normalize_window_kernel`(`:359-370`)顺序读;FFT 路径的 `chebwin_spectrum_kernel`(`:277-330`)与 `chebwin_reorder_kernel`(`:332-349`)作选读。
6. **测试**:`cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:440-473`(`run_type<T>` 内 `ok[6]`),对照 §11 的覆盖场景表。

---

## V2:第一次优化版本(占位)

> 当前未发生优化提交。当 chebwin 出现第一个已提交的优化版本时,在此追加 V2 章节,独立记录完整 SHA、代码定位、CPU/GPU 调用链、数学映射和测试证据;不覆盖 V1 内容。

---

## V3:第二次优化版本(占位)

> 当前未发生第二次优化提交。占位,规则同 V2。

---

## 版本差异与原理不变量

| 比较维度 | V1 | V2 | V3 | 始终未变的原理 |
| --- | --- | --- | --- | --- |
| 完整 SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` | — | — | — |
| 算法结构 | direct DFT 显式求和(默认)/ FFT(可选编译开关) | — | — | Dolph-Chebyshev 频域定义 + IDFT 等价性(原理 2、3) |
| 复杂度 | direct:$O(M^2)$;FFT 路径:$O(M\log M)$ | — | — | minimax 最优性(原理 4)由 $\beta$ 公式保证,与实现无关 |
| 内存访问 | 每线程串行 $O(M)$ 累加,临时 `raw` 长 `effective` | — | — | — |
| 并行方式 | 1D grid,effective 个线程,默认 stream | — | — | — |
| 精度边界 | GPU FP32 + Clenshaw;CPU FP64 解析公式 | — | — | $\beta$ 公式(原理 5)精度由 dtype 决定,数学定义不变 |
| 性能变化 | baseline | — | — | — |
| 数学原理不变量 | — | — | — | 原理 1(Chebyshev 多项式)、原理 2(频域定义)、原理 3(IDFT)、原理 4(minimax)、原理 5($\beta$ 换算)、原理 6(等纹波→端点脉冲) |

> 若 V2/V3 仅改变工程实现(如启用 FFT、改 FP64 计算、kernel 拆分),应在对应章节明确写"原理不变、实现优化";若算法本身改变(如改用 Taylor 近似、改用 IFFT 直接求),应重新映射到 `数学物理原理.md` 中的原理编号。
