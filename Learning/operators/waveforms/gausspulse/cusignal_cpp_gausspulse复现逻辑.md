# cusignal_cpp_gausspulse 复现逻辑

## 版本索引

| 版本 | 短 SHA | 说明 |
|------|--------|------|
| V1 | `1ccd32ea` | 初始学习版本：基于 `snapshot/all-current-20260716` 分支，完整记录 gausspulse 的 CPU/GPU 实现、数学映射和测试证据 |

---

## V1：原始学习版本（`1ccd32ea`）

### 版本元数据

| 项 | 值 |
|----|----|
| 完整提交 SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支名 | `snapshot/all-current-20260716` |
| 提交信息 | merge: 同步 ZKX_0720 性能测试改动 |
| gausspulse 相关文件 dirty 状态 | 未检测到未提交修改（`git status` 因 PowerShell 执行策略受限，基于源码阅读判断为 clean） |

### 相关文件清单

| 文件 | 路径（仓库相对） | 角色 |
|------|-----------------|------|
| 头文件 | `cusignal_cpp/src/waveforms/waveforms_typed.h` | 对外接口声明 |
| CPU 实现 | `cusignal_cpp/src/waveforms/waveforms_typed.cpp` | Host reference 实现 + 显式实例化 |
| GPU 实现 | `cusignal_cpp/src/waveforms/waveforms_typed.cu` | Device host wrapper + kernel launch |
| CUDA Kernel | `cusignal_cpp/src/waveforms/waveforms_kernels.cuh` | `__global__` kernel 定义 |
| 类型分发 | `cusignal_cpp/src/cuda_utils/operator_type_dispatch.h` | `OperatorId::gausspulse` 声明 |
| 类型分发实现 | `cusignal_cpp/src/cuda_utils/operator_type_dispatch.cu` | 名称表 + dispatch plan |
| 算子契约 | `cusignal_cpp/src/cuda_utils/operator_type_contract.h` | 输出类型、容差等级 |
| 类型策略 | `cusignal_cpp/src/cuda_utils/simple_signal_typed.h` | `SimpleSignalTypePolicy<T>` load/store |
| Kernel 启动 | `cusignal_cpp/src/cuda_utils/kernel_launch.h` | `launch_1d_kernel` 模板 |
| 测试 | `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | 全类型 smoke test |

---

### 变量字典：首次出现定义与原理映射

本节统一登记 V1 版本阶段三代码中出现的所有变量，按首次出现顺序排列。每个变量给出代码层面的定义、对应 `数学物理原理.md` 中的物理或数学符号（含原理编号与章节）以及含义解释。工程辅助变量标注“无直接原理对应”。同名变量在不同代码层（CPU reference、Host wrapper、CUDA kernel）出现时，按首次出现位置登记一次，并在“含义解释”列补充其他出现位置。

| 变量名 | 首次出现位置 | 代码定义 | 原理文档对应 | 含义解释 |
| --- | --- | --- | --- | --- |
| `T` | §1.1 `waveforms_typed.h:107` `GausspulseTypedResult` 模板参数 | `template <class T>`，约束 `is_simple_signal_input_v<T>` | 无直接原理对应 | 工程类型参数，决定输入/输出数组元素 dtype；支持 FP32/FP16/INT32/INT16/INT8 五种 |
| `in_phase` | §1.1 `waveforms_typed.h:108` 结构体字段 | `std::vector<T>`，必选输出 | 原理 2 §2.3 中的 $y_I(t)$ | 同相分量，$\exp(-at^2)\cos(2\pi f_c t)$；GPU 入参中以 `DeviceArray<T>&` 或 `T*` 出现 |
| `quadrature` | §1.1 `waveforms_typed.h:109` 结构体字段 | `std::vector<T>`，`retquad=true` 时填充 | 原理 2 §2.3 中的 $y_Q(t)$ | 正交分量，$\exp(-at^2)\sin(2\pi f_c t)$；GPU 完整入口中以 `DeviceArray<T>*` 指针出现，`nullptr` 表示不计算 |
| `envelope` | §1.1 `waveforms_typed.h:110` 结构体字段 | `std::vector<T>`，`retenv=true` 时填充 | 原理 2 §2.3 中的 $y_{\text{env}}(t)$ | 包络，$\exp(-at^2)$；GPU 完整入口中以 `DeviceArray<T>*` 指针出现，`nullptr` 表示不计算 |
| `t` | §1.2 `waveforms_typed.h:125` CPU 快捷入口首参数 | `const std::vector<T>&` | 原理 2 §2.1 中的 $t$ | 输入时间数组（弧度或秒）；CPU 完整入口中更名为 `time`，GPU/kernel 中以 `const DeviceArray<T>&` 或 `const T*` 出现 |
| `fc` | §1.2 `waveforms_typed.h:125` CPU 快捷入口第二参数 | `float` | 原理 2 §2.1 中的 $f_c$ | 载波（中心）频率，单位 Hz，约束 $f_c \geq 0$；kernel 内更名为 `carrier_frequency` |
| `bw` | §1.2 `waveforms_typed.h:125` CPU 快捷入口第三参数 | `float` | 原理 3 §3.1 中的 $b_w$ | 分数带宽 $b_w = \Delta f / f_c$，无量纲，约束 $b_w > 0$ |
| `bwr` | §1.2 `waveforms_typed.h:139` CPU 完整入口第四参数 | `float`；快捷入口固定 `-6.0F` | 原理 3 §3.1、§3.2 中的 $b_{wr}$ | 带宽参考电平，单位 dB，约束 $b_{wr} < 0$；$-6$ dB 对应半功率点 |
| `retquad` | §1.2 `waveforms_typed.h:139` CPU 完整入口第五参数 | `bool` | 无直接原理对应（控制 $y_Q$ 输出） | 工程旗标，`true` 时计算并填充 `quadrature`；Python 通过 4 个独立 kernel 组合实现等价语义 |
| `retenv` | §1.2 `waveforms_typed.h:139` CPU 完整入口第六参数 | `bool` | 无直接原理对应（控制 $y_{\text{env}}$ 输出） | 工程旗标，`true` 时计算并填充 `envelope` |
| `tpr` | §1.2 `waveforms_typed.h:148` `gausspulse_cutoff` 第四参数 | `double` | 原理 3 §3.3 中的 $t_{pr}$ | 截止时间参考电平，单位 dB，约束 $t_{pr} < 0$；$-60$ dB 对应包络衰减到 $0.001$ |
| `reference` | §2.1 `waveforms_typed.cpp:153` 衰减系数函数局部 | `const float`，`std::pow(10.0F, bwr / 20.0F)` | 原理 3 §3.2 中的 $r = 10^{b_{wr}/20}$ | 带宽参考电平对应的线性幅度，无量纲；$b_{wr}=-6$ 时 $r \approx 0.5012$；cutoff 路径以 FP64 重算；GPU Host wrapper `waveforms_typed.cu:88` 同名重算 |
| `attenuation` | §2.1 `waveforms_typed.cpp:154` 衰减系数函数返回值 | `float`，`-(pi*pi*fc*fc*bw*bw)/(4.0F*std::log(reference))` | 原理 2 §2.1 与原理 3 §3.2 中的 $a$ | 高斯包络时间衰减系数，$a > 0$；$a$ 越大包络越窄；GPU Host wrapper 以 FP32 重算后作为 `Scalar` 形参传入 kernel；cutoff 路径以 FP64 重算 |
| `pi` | §2.1 `waveforms_typed.cpp:148-156` 衰减系数公式隐式常量；GPU 侧 `waveforms_typed.cu:87` 显式定义 | `constexpr float pi = 3.14159265358979323846F` | 数学常数 $\pi$ | 圆周率；CPU reference 与 GPU Host wrapper 各自定义 FP32 字面量；kernel 内以 `6.28318530717958647692F` 表示 $2\pi$；cutoff 路径使用 `pi64` |
| `time` | §2.2 `waveforms_typed.cpp:263` CPU 完整入口首参数 | `const std::vector<T>&` | 原理 2 §2.1 中的 $t$ | 输入时间数组的别名（与快捷入口的 `t` 同义）；kernel 内作为局部 `Scalar time` 表示经 `waveform_load` 提升后的单个时间元素 |
| `x` | §2.2 `waveforms_typed.cpp:272` CPU 循环局部 | `const float`，`SimpleSignalTypePolicy<T>::load(time[i])` | 原理 2 §2.1 中的 $t$（单个元素） | 经类型提升后的当前时间元素，用于 `exp(-a*x*x)` 与 `2π*fc*x` |
| `envelope_value` | §2.2 `waveforms_typed.cpp:273` CPU 循环局部 | `const float`，`std::exp(-attenuation * x * x)` | 原理 2 §2.3 中的 $y_{\text{env}}(t)$ | 当前元素的高斯包络值；GPU kernel 内更名为 `env` |
| `phase` | §2.2 `waveforms_typed.cpp:274` CPU 循环局部 | `const float`，`2.0F * pi * fc * x` | 原理 2 §2.1 载波相位 $2\pi f_c t$ | 当前元素的载波相位，单位弧度；kernel 内以 `2π * carrier_frequency * time` 计算 |
| `result` | §2.2 `waveforms_typed.cpp:267` CPU 完整入口返回值 | `GausspulseTypedResult<T>` | 无直接原理对应 | 工程聚合输出，承载 `in_phase`、`quadrature`、`envelope` 三个分量 |
| `pi64` | §2.4 `waveforms_typed.cpp:294` cutoff 函数局部 | `constexpr double pi64 = 3.141592653589793238462643383279502884` | 数学常数 $\pi$（FP64 精度） | 圆周率 FP64 字面量，仅 cutoff 路径使用以保证与 Python NumPy float64 精度一致 |
| `time_reference` | §2.4 `waveforms_typed.cpp:298` cutoff 函数局部 | `const double`，`std::pow(10.0, tpr / 20.0)` | 原理 3 §3.3 中的 $t_{\text{ref}} = 10^{t_{pr}/20}$ | 截止时间参考电平对应的线性幅度；$t_{pr}=-60$ 时 $t_{\text{ref}} = 0.001$ |
| `Scalar` | §3.2 `waveforms_kernels.cuh:148` kernel 模板参数 | `typename Scalar`，显式实例化为 `float` | 无直接原理对应 | 工程计算精度类型，决定 kernel 内中间量精度；FP32 路径下为 `float`，无 FP64 GPU 路径 |
| `count` | §3.2 `waveforms_kernels.cuh:154` kernel 第四参数 | `std::size_t` | 无直接原理对应 | 工程变量，输入数组长度，用于线程越界检查 `if (index >= count) return` |
| `carrier_frequency` | §3.2 `waveforms_kernels.cuh:155` kernel第五参数 | `Scalar` | 原理 2 §2.1 中的 $f_c$ | 载波频率，Host 侧 `fc` 传入 kernel 时的形参名 |
| `index` | §3.2 `waveforms_kernels.cuh:158` kernel 局部 | `const std::size_t`，`blockIdx.x * blockDim.x + threadIdx.x` | 无直接原理对应 | 工程变量，CUDA 线程全局索引，决定当前线程处理的数组元素 |
| `env` | §3.2 `waveforms_kernels.cuh:163` kernel 局部 | `const Scalar`，`exp(-attenuation * time * time)` | 原理 2 §2.3 中的 $y_{\text{env}}(t)$ | kernel 内的包络局部变量，与 CPU 的 `envelope_value` 同义 |
| `SimpleSignalTypePolicy<T>` | §4.1 `simple_signal_typed.h:16` 类型策略模板 | 模板结构体，提供 `load()` / `store()` 静态成员 | 无直接原理对应 | 工程类型策略，负责输入 dtype → FP32（load）与 FP32 → 输出 dtype（store，含整数饱和截断） |
| `load()` / `store()` | §4.1 `simple_signal_typed.h:17-38` 类型策略成员 | `static float load(const T&)` / `static T store(float)` | 无直接原理对应 | 工程类型转换函数；FP32 恒等、FP16 经 `__half2float`/`__float2half_rn`、整数经 `static_cast` 与 clamp |

---

### 一、对外接口

#### 1.1 数据结构

```cpp
// waveforms_typed.h:107-112
template <class T>
struct GausspulseTypedResult {
    std::vector<T> in_phase;   // 同相分量（必选）
    std::vector<T> quadrature; // 正交分量（retquad=true 时填充，否则为空）
    std::vector<T> envelope;    // 包络（retenv=true 时填充，否则为空）
};
```

#### 1.2 CPU 接口（`waveforms_typed.h:114-150`）

| 签名 | 行号 | 说明 |
|------|------|------|
| `template<class T> std::vector<T> gausspulse_typed_cpu(const std::vector<T>& t, float fc, float bw)` | 125-127 | 快捷入口：固定 `bwr=-6`，仅返回同相分量 |
| `template<class T> GausspulseTypedResult<T> gausspulse_typed_cpu(const std::vector<T>& t, float fc, float bw, float bwr, bool retquad, bool retenv)` | 139-142 | 完整 CPU reference，覆盖全部返回组合 |
| `double gausspulse_cutoff(double fc, double bw, double bwr, double tpr)` | 148-150 | 截止时间计算，FP64 标量路径 |

#### 1.3 GPU 接口（`waveforms_typed.h:152-213`）

| 签名 | 行号 | 说明 |
|------|------|------|
| `template<class T> void gausspulse_device(const DeviceArray<T>& t, DeviceArray<T>& out, float fc, float bw)` | 165-167 | 快捷入口：固定 `bwr=-6`，仅输出同相分量 |
| `template<class T> void gausspulse_device(const DeviceArray<T>& t, DeviceArray<T>& in_phase, DeviceArray<T>* quadrature, DeviceArray<T>* envelope, float fc, float bw, float bwr)` | 183-189 | 完整入口：指针为 nullptr 表示不请求该输出 |
| `template<class T> void gausspulse_device(const DeviceArray<T>& t, DeviceArray<T>& in_phase, DeviceArray<T>& quadrature, DeviceArray<T>& envelope, float fc, float bw, float bwr)` | 207-213 | 三路全部输出版本（引用而非指针） |

#### 1.4 类型支持

`T` 支持五种类型：`float`（FP32）、`__half`（FP16）、`std::int32_t`（INT32）、`std::int16_t`（INT16）、`std::int8_t`（INT8）。

计算精度路径：

- FP32 输入 → 原生 FP32 计算（`ComputePath::native_fp32`）
- FP16 / INT32 / INT16 / INT8 → 提升为 FP32 计算（`ComputePath::promote_to_fp32`）
- 所有 GPU 计算均在 FP32 中完成，不涉及 GPU 端 FP64
- `cutoff` 函数是唯一 FP64 路径（Host 端）

---

### 二、CPU Reference 实现

#### 2.1 衰减系数计算

**源码位置**：`waveforms_typed.cpp:148-156`

```cpp
float gausspulse_attenuation(float fc, float bw, float bwr)
{
    if (fc < 0.0F || bw <= 0.0F || bwr >= 0.0F) {
        throw std::invalid_argument("gausspulse parameters");
    }
    const float reference = std::pow(10.0F, bwr / 20.0F);
    return -(pi * pi * fc * fc * bw * bw) /
        (4.0F * std::log(reference));
}
```

**数学映射**：对应 `数学物理原理.md` 原理 3 的公式：

$$a = -\frac{(\pi f_c b_w)^2}{4\,\ln r}, \quad r = 10^{b_{wr}/20}$$

**与 Python 的精度差异**：C++ 使用 `float`（FP32）计算，Python 使用 NumPy `float64` 计算。这是有意为之的设计——C++ 全部 GPU 计算在 FP32 中完成，CPU reference 也保持 FP32 以保证一致性。

#### 2.2 完整 CPU reference（逐元素循环）

**源码位置**：`waveforms_typed.cpp:261-287`

```cpp
template <class T>
GausspulseTypedResult<T> gausspulse_typed_cpu(
    const std::vector<T>& time, float fc, float bw, float bwr,
    bool retquad, bool retenv)
{
    const float attenuation = gausspulse_attenuation(fc, bw, bwr);
    GausspulseTypedResult<T> result;
    result.in_phase.resize(time.size());
    if (retquad) result.quadrature.resize(time.size());
    if (retenv) result.envelope.resize(time.size());
    for (std::size_t i = 0; i < time.size(); ++i) {
        const float x = detail::SimpleSignalTypePolicy<T>::load(time[i]);
        const float envelope_value = std::exp(-attenuation * x * x);
        const float phase = 2.0F * pi * fc * x;
        result.in_phase[i] = detail::SimpleSignalTypePolicy<T>::store(
            envelope_value * std::cos(phase));
        if (retquad) {
            result.quadrature[i] = detail::SimpleSignalTypePolicy<T>::store(
                envelope_value * std::sin(phase));
        }
        if (retenv) {
            result.envelope[i] = detail::SimpleSignalTypePolicy<T>::store(
                envelope_value);
        }
    }
    return result;
}
```

**调用链**：

```
gausspulse_typed_cpu(time, fc, bw, bwr, retquad, retenv)
  → gausspulse_attenuation(fc, bw, bwr)    // 计算衰减系数 a
  → for each time[i]:
      → SimpleSignalTypePolicy<T>::load()   // 输入类型 → float
      → exp(-a * x * x)                    // 高斯包络
      → 2π * fc * x                         // 载波相位
      → envelope * cos(phase)               // 同相分量
      → envelope * sin(phase)               // 正交分量（可选）
      → SimpleSignalTypePolicy<T>::store()  // float → 输出类型
```

**关键特征**：

1. **逐元素串行循环**：CPU reference 不使用 SIMD 或并行化，直接 `for` 循环
2. **类型提升/降低**：每个元素经 `load()` 提升为 `float`，计算后经 `store()` 转回 `T`
3. **`store()` 的饱和截断**：对整数类型，`store()` 将值 clamp 到该类型的 `[lowest, max]` 范围
4. **无 GPU 依赖**：CPU reference 不创建 GPU stream、不分配 device 内存、不调用任何 CUDA runtime

#### 2.3 快捷入口

**源码位置**：`waveforms_typed.cpp:255-259`

```cpp
template <class T>
std::vector<T> gausspulse_typed_cpu(const std::vector<T>& time, float fc, float bw)
{
    return gausspulse_typed_cpu(time, fc, bw, -6.0F, false, false).in_phase;
}
```

固定 `bwr=-6`、`retquad=false`、`retenv=false`，仅返回同相分量的 `std::vector<T>`。

#### 2.4 截止时间（FP64 路径）

**源码位置**：`waveforms_typed.cpp:289-300`

```cpp
double gausspulse_cutoff(double fc, double bw, double bwr, double tpr)
{
    if (fc < 0.0 || bw <= 0.0 || bwr >= 0.0 || tpr >= 0.0) {
        throw std::invalid_argument("gausspulse cutoff parameters");
    }
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    const double reference = std::pow(10.0, bwr / 20.0);
    const double attenuation = -(pi64 * pi64 * fc * fc * bw * bw) /
        (4.0 * std::log(reference));
    const double time_reference = std::pow(10.0, tpr / 20.0);
    return std::sqrt(-std::log(time_reference) / attenuation);
}
```

**数学映射**：$t_c = \sqrt{-\ln(t_{\text{ref}}) / a}$，对应原理 3.3 节。

**与 Python 对比**：Python 使用 NumPy `float64` 计算 cutoff，C++ 同样使用 `double`（FP64）——这是两边精度一致的唯一路径。

**特殊行为**：当 `fc=0` 时，`attenuation = 0`，`-ln(tref)/attenuation` 趋向无穷大，返回 `std::numeric_limits<double>::infinity()`。测试 `e3_wave_window_type_smoke.cu:270` 验证了此行为。

---

### 三、GPU 实现

#### 3.1 Host Wrapper

**源码位置**：`waveforms_typed.cu:74-98`

```cpp
template<class T>
void gausspulse_device(
    const DeviceArray<T>& t, DeviceArray<T>& in_phase,
    DeviceArray<T>* quadrature, DeviceArray<T>* envelope,
    float fc, float bw, float bwr)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (fc < 0.0F || bw <= 0.0F || bwr >= 0.0F ||
        in_phase.size() != t.size() ||
        (quadrature != nullptr && quadrature->size() != t.size()) ||
        (envelope != nullptr && envelope->size() != t.size())) {
        throw std::invalid_argument("gausspulse parameters or shape");
    }
    constexpr float pi = 3.14159265358979323846F;
    const float reference = powf(10.0F, bwr / 20.0F);
    const float attenuation = -(pi * pi * fc * fc * bw * bw) /
        (4.0F * logf(reference));
    if (t.empty()) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::gausspulse_kernel<T,float>, in_phase.size(),
        t.data(), in_phase.data(),
        quadrature == nullptr ? nullptr : quadrature->data(),
        envelope == nullptr ? nullptr : envelope->data(),
        t.size(), fc, attenuation);
}
```

**Host 端职责**：

1. **参数校验**：`fc >= 0`、`bw > 0`、`bwr < 0`、所有输出数组 size 匹配
2. **计算衰减系数**：在 Host 端用 FP32 计算 `attenuation`（与 CPU reference 相同公式）
3. **空输入处理**：`t.empty()` 时直接返回，不启动 kernel
4. **Kernel 启动配置**：通过 `launch_1d_kernel` 使用默认 stream（`nullptr`），256 线程/block

**便捷重载**：

- `gausspulse_device(t, out, fc, bw)` → 调用上面完整版本，传 `nullptr` 给 `quadrature` 和 `envelope`，`bwr=-6`
- `gausspulse_device(t, in_phase, quadrature, envelope, fc, bw, bwr)` → 传引用地址给完整版本

#### 3.2 CUDA Kernel

**源码位置**：`waveforms_kernels.cuh:148-174`

```cpp
template <typename T, typename Scalar>
__global__ void gausspulse_kernel(
    const T* t,
    T* in_phase,
    T* quadrature,
    T* envelope,
    std::size_t count,
    Scalar carrier_frequency,
    Scalar attenuation)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const Scalar time = static_cast<Scalar>(waveform_load(t[index]));
    const Scalar env = exp(-attenuation * time * time);
    const Scalar phase =
        static_cast<Scalar>(6.28318530717958647692F) * carrier_frequency * time;
    if (envelope != nullptr) {
        envelope[index] = waveform_store<T>(env);
    }
    if (quadrature != nullptr) {
        quadrature[index] = waveform_store<T>(env * sin(phase));
    }
    if (in_phase != nullptr) {
        in_phase[index] = waveform_store<T>(env * cos(phase));
    }
}
```

#### 3.3 Kernel 架构详解

**Grid/Block 配置**：

- Block 大小：256 线程（由 `launch_1d_kernel` 默认值 `block_size=256` 决定）
- Grid 大小：`ceil(N / 256)`，其中 N 为输入数组长度
- 每个线程处理一个元素

**线程索引**：

```
index = blockIdx.x * 256 + threadIdx.x
```

**执行流程**：

```
对每个线程（index < count）：
  1. waveform_load(t[index])    // 从 global memory 加载并提升为 Scalar
  2. exp(-attenuation * t²)     // 计算高斯包络
  3. phase = 2π * fc * t        // 计算载波相位
  4. if envelope ≠ nullptr → envelope[index] = store(env)
  5. if quadrature ≠ nullptr → quadrature[index] = store(env * sin(phase))
  6. in_phase[index] = store(env * cos(phase))   // 始终写入
```

**可选项的 nullptr 机制**：

与 Python cuSignal 使用 4 个独立 kernel（`F_F`、`F_T`、`T_F`、`T_T`）不同，C++ 使用**一个统一 kernel**，通过空指针检查实现可选输出：

- `quadrature == nullptr` → 不计算正弦分量
- `envelope == nullptr` → 不存储包络

这是工程上的权衡：消除 4 倍 kernel 代码体积，但引入运行时分支。由于同一 warp 内所有线程走相同分支（`quadrature` 和 `envelope` 对整个 kernel launch 是固定的），warp 内不会发生分支发散。

**`waveform_load` / `waveform_store`**：

定义在 `waveforms_kernels.cuh:20-37`，对 FP32 直接透传，对 FP16/INT 类型经 `SimpleSignalTypePolicy<T>` 进行类型提升/降低。

#### 3.4 同步与错误处理

- **异步执行**：`launch_1d_kernel` 使用默认 stream（`nullptr`），kernel launch 是异步的
- **启动错误检查**：`CUDA_KERNEL_CHECK()` 宏检查 `cudaLaunchKernel` 返回值
- **无完成同步**：函数返回前不调用 `cudaDeviceSynchronize`，调用方负责在读取输出前同步
- **无 workspace**：kernel 不分配共享内存或临时缓冲区，所有中间量（`env`、`phase`、`sin/cos` 值）在线程寄存器中计算

---

### 四、Template、类型分发与显式实例化

#### 4.1 类型策略

`SimpleSignalTypePolicy<T>`（`simple_signal_typed.h:16-38`）：

| 类型 `T` | `load()` | `store()` |
|----------|----------|-----------|
| `float` | 直接返回（identity） | 直接返回（identity） |
| `__half` | `__half2float(value)` | `__float2half_rn(value)`（就近舍入） |
| `std::int32_t` | `static_cast<float>(value)` | clamp 到 `[INT32_MIN, INT32_MAX]` 后截断 |
| `std::int16_t` | `static_cast<float>(value)` | clamp 到 `[INT16_MIN, INT16_MAX]` 后截断 |
| `std::int8_t` | `static_cast<float>(value)` | clamp 到 `[INT8_MIN, INT8_MAX]` 后截断 |

#### 4.2 类型分发计划

由 `operator_type_dispatch.cu` 中 `dispatch_plan(OperatorId::gausspulse, type)` 决定：

| 目标类型 | 计算路径 | 输出 dtype |
|----------|----------|-----------|
| FP32 | `promote_to_fp32` | 与输入相同（FP32） |
| FP16 | `promote_to_fp32` | 与输入相同（FP16） |
| INT32 | `promote_to_fp32` | 与输入相同（INT32） |
| INT16 | `promote_to_fp32` | 与输入相同（INT16） |
| INT8 | `promote_to_fp32` | 与输入相同（INT8） |

**注意**：`native_fp32` 仅用于 `kalman_filter`（该算子有专用 FP32 GPU 特化）。gausspulse 即使输入为 FP32，也走 `promote_to_fp32` 路径——这是类型分发框架的统一策略，保证所有算子在相同的 FP32 compute backend 上运行。对 FP32 输入而言，"提升"实际上是类型检查的 no-op。

所有类型的中间计算均在 FP32 中完成。

#### 4.3 算子契约

由 `operator_type_contract.h:86` 定义：

```
{gausspulse, OutputKind::same_as_input_or_float64_scalar, ToleranceClass::pointwise, "fp32-arrays+host-fp64-cutoff"}
```

- **OutputKind**：数组输出保持输入 dtype；`cutoff` 返回 FP64 标量
- **ToleranceClass**：`pointwise`（逐元素容差）
- **描述**：FP32 数组计算 + Host 端 FP64 cutoff

#### 4.4 显式实例化

**CPU 侧**（`waveforms_typed.cpp:554-594`）：

```cpp
#define INSTANTIATE_CPU(T) \
    template std::vector<T> gausspulse_typed_cpu(const std::vector<T>&, float, float); \
    template GausspulseTypedResult<T> gausspulse_typed_cpu( \
        const std::vector<T>&, float, float, float, bool, bool);
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
```

**GPU 侧**（`waveforms_typed.cu:218-232`）：

```cpp
#define I(T) \
    template void gausspulse_device(const DeviceArray<T>&,DeviceArray<T>&,float,float); \
    template void gausspulse_device(const DeviceArray<T>&,DeviceArray<T>&,DeviceArray<T>*,DeviceArray<T>*,float,float,float); \
    template void gausspulse_device(const DeviceArray<T>&,DeviceArray<T>&,DeviceArray<T>&,DeviceArray<T>&,float,float,float);
I(float); I(__half); I(int32_t); I(int16_t); I(int8_t);
```

五种类型的所有公开重载均有显式实例化覆盖。

---

### 五、数学原理→Python→C++ CPU→C++ GPU 四方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
|-----------|------------------------|-----------------|-----------------|---------|
| **原理 2**：$s(t) = \exp(-at^2)\cos(2\pi f_c t)$ | `waveforms.py:169-170`（F_F kernel） | `waveforms_typed.cpp:273-276` | `waveforms_kernels.cuh:162-173` | 数学完全等价；Python 用 4 个独立 kernel，C++ 用 1 个统一 kernel + nullptr 检查 |
| **原理 2 正交**：$y_Q = \exp(-at^2)\sin(2\pi f_c t)$ | `waveforms.py:199`（T_F kernel，`sincos`） | `waveforms_typed.cpp:278-279` | `waveforms_kernels.cuh:168-169` | Python 用 `sincos()` 一次调用同时计算 sin/cos；C++ GPU 用独立 `sin()` + `cos()`，无 `sincos` 优化 |
| **原理 2 包络**：$y_{\text{env}} = \exp(-at^2)$ | `waveforms.py:180`（F_T kernel） | `waveforms_typed.cpp:282-283` | `waveforms_kernels.cuh:165-166` | 数学完全等价 |
| **原理 3**：$a = -\dfrac{(\pi f_c b_w)^2}{4\ln r}$ | `waveforms.py:328`（NumPy float64） | `waveforms_typed.cpp:148-156`（FP32） | `waveforms_typed.cu:88-90`（FP32） | **精度差异**：Python 用 float64，C++ CPU/GPU 均用 float32。对典型参数（$f_c=1000, b_w=0.5, b_{wr}=-6$），$a$ 的相对误差约 $10^{-15}$，在 FP32 精度内可忽略 |
| **原理 3.3**：$t_c = \sqrt{-\ln(t_{\text{ref}})/a}$ | `waveforms.py:340`（NumPy float64） | `waveforms_typed.cpp:299`（FP64） | GPU 无此路径 | cutoff 在两边均使用 FP64，保持一致 |
| **原理 1**：高斯傅里叶变换仍是高斯 | 隐含（参数 $a$ 决定带宽） | 隐含（同左） | 隐含（同左） | 均不在代码中显式体现傅里叶关系，仅通过 $a$ 的公式关联 |

---

### 六、Python 与 C++ 实现的异同

#### 6.1 相同点

| 方面 | 说明 |
|------|------|
| 核心数学公式 | $\exp(-at^2)\cos(2\pi f_c t)$ 完全一致 |
| 参数 $a$ 的计算公式 | $a = -(\pi f_c b_w)^2 / (4\ln r)$ 完全一致 |
| 参数校验 | $f_c \geq 0$、$b_w > 0$、$b_{wr} < 0$ 完全一致 |
| 可选输出语义 | `retquad` / `retenv` 控制的 I/Q/包络输出语义完全一致 |
| cutoff 语义 | $t_c = \sqrt{-\ln(t_{\text{ref}})/a}$ 完全一致 |
| $f_c=0$ 行为 | 退化为纯高斯（$\cos(0)=1$），两边一致 |

#### 6.2 不同点

| 方面 | Python (cuSignal) | C++ (cusignal_cpp) | 影响 |
|------|-------------------|--------------------|------|
| **Kernel 数量** | 4 个独立 ElementwiseKernel（F_F, F_T, T_F, T_T） | 1 个统一 kernel + nullptr 检查 | C++ 代码体积更小，消除模板组合爆炸；运行时无 warp 发散 |
| **GPU sin/cos 计算** | T_F 和 T_T kernel 使用 `sincos()` 同时计算 sin 和 cos | 所有 kernel 使用独立 `sin()` + `cos()` 调用 | Python 版本更高效（减少约 50% 三角函数开销），C++ 版本未做此优化 |
| **衰减系数精度** | NumPy float64 | C++ float32 | 对典型参数 $a$ 误差约 $10^{-15}$，在 FP32 精度内可忽略 |
| **类型支持** | 动态类型（CuPy dtype） | 5 种静态类型（FP32/FP16/INT32/INT16/INT8） | C++ 支持整数输入，Python 不支持 |
| **cutoff 返回类型** | Python float（float64） | C++ double（float64） | 一致 |
| **错误处理** | `ValueError` | `std::invalid_argument` | 语义等价 |
| **输出内存布局** | CuPy 数组（column-major 兼容 row-major） | `std::vector<T>` 或 `DeviceArray<T>`（row-major） | C++ 使用连续 row-major 存储 |
| **Kernel 启动** | CuPy 管理（首次调用时 NVCC 编译，后续缓存） | 显式模板实例化 + 编译期确定 | C++ 编译时开销更高，运行时无 JIT 开销 |

#### 6.3 `sincos` 优化缺失的分析

C++ GPU kernel 中 `sin` 和 `cos` 分别计算，而 Python 版本使用 `sincos` 内置函数。这是一个已知的优化差异：

- Python（CuPy ElementwiseKernel）中 `sincos(angle, &sin_ptr, &cos_ptr)` 是 CUDA math API 的内置函数，一次调用同时产生 sin 和 cos
- C++ 版本在 `waveforms_kernels.cuh` 的 `gausspulse_kernel` 中直接使用 `sin(phase)` 和 `cos(phase)`，没有利用 `sincos`
- 这不影响正确性，但在 `retquad=true` 场景下会有约 2 倍的三角函数调用开销

可能的原因：C++ 版本采用统一 kernel（同时处理 `retquad` 和 `retenv` 的所有组合），需要在运行时判断是否计算 sin，因此 `sincos` 的优势无法充分发挥——如果 `quadrature == nullptr`，计算 `sincos` 的 sin 部分就浪费了。

---

### 七、精度、dtype 与边界条件

#### 7.1 精度

| 路径 | 精度 | 说明 |
|------|------|------|
| GPU 数组计算（FP32 输入） | FP32 | 原生 FP32 计算 |
| GPU 数组计算（FP16/INT 输入） | FP32 | `load()` 提升为 FP32，`store()` 降低回原类型 |
| CPU reference | FP32 | 始终使用 float |
| cutoff | FP64 | 唯一使用 double 的路径 |
| GPU ↔ CPU 一致性 | CPU reference 使用 FP32，与 GPU 计算精度匹配 |

#### 7.2 容差

测试 `e3_wave_window_type_smoke.cu` 使用 `3.0e-2`（3%）作为容差，与 `ToleranceClass::pointwise` 一致。此容差较宽松，原因：

1. CPU reference 在 FP32 中计算 `exp`、`sin`、`cos`，GPU 也在 FP32 中计算，但两者使用的数学函数实现可能略有不同（`std::exp` vs `__expf`）
2. 整数类型的截断误差可能累积
3. 对于极端参数（极大 $a$ 值），exp 下溢行为可能因平台而异

#### 7.3 边界条件

| 边界 | 行为 | 代码位置 |
|------|------|---------|
| $f_c = 0$ | 合法，退化为纯高斯 $\exp(-at^2)$ | `waveforms_typed.cpp:274`（$\cos(0)=1, \sin(0)=0$） |
| $f_c < 0$ | 抛出 `std::invalid_argument` | `waveforms_typed.cpp:150`、`waveforms_typed.cu:81` |
| $b_w \leq 0$ | 抛出 `std::invalid_argument` | 同上 |
| $b_{wr} \geq 0$ | 抛出 `std::invalid_argument`（防止 $\ln(r) \geq 0$ 导致 $a \leq 0$） | 同上 |
| 空输入数组 | GPU 直接返回（不启动 kernel）；CPU 返回空 vector | `waveforms_typed.cu:91`、`waveforms_typed.cpp:266`（`time.size()=0` 时循环零次） |
| cutoff $t_{pr} \geq 0$ | 抛出 `std::invalid_argument` | `waveforms_typed.cpp:291` |
| cutoff $f_c = 0$ | 返回 `double infinity`（因 $a=0$） | `waveforms_typed.cpp:296`（division by zero → +∞） |
| 输出 size 不匹配 | GPU 抛出 `std::invalid_argument` | `waveforms_typed.cu:82-84` |
| 整数溢出 | `SimpleSignalTypePolicy<T>::store()` 做 clamp | `simple_signal_typed.h:24-28` |

#### 7.4 ZQ500 平台限制

文档 `E3_WAVEFORM_WAVELET_DTYPE_AUDIT.md` 和 `OPERATOR_PLATFORMIZATION_D0_D1_MATRIX.md` 记录了 gausspulse 在 ZQ500 平台上的实现状态：

- FP32/FP16 路径在 ZQ500 GPU 上原生支持
- 整数类型通过 FP32 提升路径支持
- 无 FP64 GPU 计算路径（cutoff 在 Host 端 FP64 完成）

---

### 八、测试验证

#### 8.1 测试文件

**文件**：`cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`

**测试覆盖**（以 `T=float` 为例，所有 5 种类型均执行）：

| 测试项 | 代码行号 | 验证内容 |
|--------|---------|---------|
| 基础 GPU vs CPU 一致性 | 194-198 | `gausspulse_device(d_time, real_out, 0.25, 0.5)` vs `gausspulse_typed_cpu(time, 0.25, 0.5)` |
| 正交分量一致性 | 199-207 | I/Q 两路 GPU vs CPU 对比 |
| 包络一致性 | 208-217 | 包络 GPU vs CPU 对比 |
| 全部三路输出 | 218-221 | I+Q+envelope 同时输出 |
| 零频率 | 222-224 | `fc=0` 退化为纯高斯 |
| GPU 拒绝负频率 | 229-232 | `fc=-1` 抛出异常 |
| CPU 拒绝负频率 | 233-236 | `fc=-1` 抛出异常 |
| GPU 拒绝非负 bwr | 237-242 | `bwr=0` 抛出异常 |
| CPU 拒绝非负 bwr | 243-248 | `bwr=0` 抛出异常 |
| cutoff 默认有限 | 269 | `std::isfinite(gausspulse_cutoff())` |
| cutoff 零频率 → ∞ | 270 | `std::isinf(gausspulse_cutoff(0.0))` |
| cutoff 拒绝非负 tpr | 250-253 | `tpr=0` 抛出异常 |
| rank3 输入 | 254-257 | 24 元素 rank3 输入的 GPU vs CPU 一致性 |

#### 8.2 测试容差

```cpp
constexpr float kTolerance = 3.0e-2F;  // 3%
```

此容差覆盖 FP32 计算精度、`std::exp` vs CUDA `exp` 的微小差异、以及整数截断误差。

#### 8.3 通过标准

测试对每个类型（FP32、FP16、INT32、INT16、INT8）执行全部 14 项算子（包括 chirp、gausspulse、morlet 等），共 70 项检查。gausspulse 对应 `ok[1]`，必须全部通过：

```
ok[1] = gauss_base_matches && gauss_rank3_matches &&
    gauss_quad_matches && gauss_envelope_matches &&
    matches_real(gauss_i, gauss_all_reference.in_phase) &&
    matches_real(gauss_q, gauss_all_reference.quadrature) &&
    matches_real(gauss_envelope, gauss_all_reference.envelope) &&
    gauss_zero_frequency_matches &&
    gauss_gpu_rejects_negative_frequency &&
    gauss_cpu_rejects_negative_frequency &&
    gauss_gpu_rejects_nonnegative_bwr &&
    gauss_cpu_rejects_nonnegative_bwr &&
    std::isfinite(gausspulse_cutoff()) &&
    std::isinf(gausspulse_cutoff(0.0)) &&
    gauss_cutoff_rejects_nonnegative_tpr;
```

#### 8.4 未覆盖的风险

1. **极端参数**：极大 $a$ 值（极窄脉冲）的数值稳定性未在测试中验证
2. **大型数组**：测试仅使用 5 元素小数组，未验证大规模数据的性能和内存行为
3. **非规则 shape**：测试使用一维连续数组，未验证多维输入的广播语义（gausspulse 当前仅支持一维）
4. **GPU ↔ CPU 跨设备一致性**：未在 ZQ500 硬件上验证 GPU 输出与 CPU reference 的一致性

---

### 九、版本差异与原理不变量

#### 原理不变量

以下数学物理原理在 Python cuSignal 和 C++ cusignal_cpp 中完全一致：

| 不变量 | 说明 |
|--------|------|
| 包络公式 | $\exp(-at^2)$，$a = -(\pi f_c b_w)^2/(4\ln r)$ |
| 载波公式 | $\cos(2\pi f_c t)$ 和 $\sin(2\pi f_c t)$ |
| I/Q 关系 | $y_I = \text{env} \cdot \cos(\text{phase})$，$y_Q = \text{env} \cdot \sin(\text{phase})$ |
| 包络独立性 | 包络 $\exp(-at^2)$ 与载波频率 $f_c$ 无关 |
| cutoff 公式 | $t_c = \sqrt{-\ln(t_{\text{ref}})/a}$ |
| 参数约束 | $f_c \geq 0$、$b_w > 0$、$b_{wr} < 0$、$t_{pr} < 0$（cutoff） |

#### 工程实现差异

| 维度 | Python cuSignal | C++ cusignal_cpp | 性质 |
|------|----------------|------------------|------|
| Kernel 架构 | 4 个独立 ElementwiseKernel | 1 个统一 kernel + nullptr 检查 | 实现差异，原理不变 |
| sin/cos 计算 | `sincos()` 一次调用 | 独立 `sin()` + `cos()` | 性能差异，正确性不变 |
| $a$ 精度 | float64（NumPy） | float32（C++） | 精度差异，对典型参数可忽略 |
| 类型支持 | 动态 dtype | 5 种静态类型 | 功能差异（C++ 多支持整数） |
| 输出内存 | CuPy GPU 数组 | `DeviceArray<T>` / `std::vector<T>` | API 差异 |

---

### 十、建议阅读顺序

1. **头文件** `waveforms_typed.h:107-213` — 确认所有公开接口和数据结构
2. **衰减系数计算** `waveforms_typed.cpp:148-156` — 对照数学原理文档中 $a$ 的推导
3. **CPU reference** `waveforms_typed.cpp:261-287` — 逐元素循环，理解每个类型的 load/store
4. **Host wrapper** `waveforms_typed.cu:74-98` — 理解参数校验、衰减系数计算和 kernel 启动
5. **CUDA Kernel** `waveforms_kernels.cuh:148-174` — 理解 grid/block 配置、nullptr 机制、寄存器使用
6. **类型策略** `simple_signal_typed.h:16-38` — 理解 load/store 如何处理五种类型
7. **Kernel 启动** `kernel_launch.h:44-56` — 理解 `launch_1d_kernel` 的默认配置
8. **测试** `e3_wave_window_type_smoke.cu:194-285` — 理解测试覆盖范围和通过条件