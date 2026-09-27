# cusignal_cpp_chirp复现逻辑

## 版本索引

| 版本 | 状态 | Git 提交 | 说明 |
| --- | --- | --- | --- |
| [V1：当前正式学习版本](#v1当前正式学习版本55af451) | 当前 | `55af4512d44424a5309e3e9fe1faab484005a5d3` | 五 dtype、四种实值 method、linear complex 的 CPU/GPU 复现 |

当前只有 V1，尚无 V2。后续优化必须追加新版本章节，不覆盖本节。

## V1：当前正式学习版本（55af451）

### 1. 版本身份与可追溯性

| 项目 | 值 |
| --- | --- |
| 完整 SHA | `55af4512d44424a5309e3e9fe1faab484005a5d3` |
| 提交时间 | `2026-08-14T20:46:51+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交主题 | `docs(errors): 记录Task2代表异常与开发者交接` |
| 仓库状态 | `git status --short` 无输出，工作区 clean |
| chirp 相关文件状态 | 全部 clean，无未提交修改 |

本文行号均指上述提交。历史复核应使用：

```powershell
git -C ZKX show 55af4512d44424a5309e3e9fe1faab484005a5d3:cusignal_cpp/<仓库内相对路径>
```

本阶段只读源码，没有修改 `ZKX/cusignal_cpp`，没有本地或远程构建、运行及硬件测试。

### 2. 代码证据索引

| 层次 | SHA 中的路径、符号与行号 | 职责 |
| --- | --- | --- |
| 共享数学核心 | `cusignal_cpp/src/waveforms/waveform_math.h:7-67`，`ChirpMethod`、`stable_chirp_phase` | 四种 method 的 FP32 累计相位 |
| 类型装卸 | `cusignal_cpp/src/cuda_utils/simple_signal_typed.h:17-50`，`SimpleSignalTypePolicy` | 五种输入提升到连续计算类型，输出转换回 T |
| 复数布局 | `cusignal_cpp/src/fft_interface/fft_interface.h:95-107`，`ComplexFloat` | 两个 `float` 字段 `re/im` |
| 公共声明 | `cusignal_cpp/src/waveforms/waveforms_typed.h:27-105` | CPU/GPU real/complex 模板 API 契约 |
| CPU helper | `cusignal_cpp/src/waveforms/waveforms_typed.cpp:157-176`，`parse_cpu_chirp_method`、`cpu_chirp_phase` | method 解析和共享相位转发 |
| CPU real | 同文件 `:179-206`，`chirp_typed_cpu` | Host 循环、校验、余弦、dtype 写回 |
| CPU complex | 同文件 `:208-221`，`chirp_complex_typed_cpu` | linear complex reference |
| CPU 实例化 | 同文件 `:522-527,557-562`，`INSTANTIATE_CPU` | 五种 T 的显式实例化 |
| GPU wrapper | `cusignal_cpp/src/waveforms/waveforms_typed.cu:8-72` | method/频率/shape 校验、角度转换、launch |
| GPU kernel | `cusignal_cpp/src/waveforms/waveforms_kernels.cuh:13-31,44-84` | 装卸、线程索引、相位、real/complex 写回 |
| launch helper | `cusignal_cpp/src/cuda_utils/kernel_launch.h:10-56` | block=256、grid 向上取整、默认 stream、launch check |
| GPU 实例化 | `cusignal_cpp/src/waveforms/waveforms_typed.cu:216-230`，宏 `I` | 五种 T 的三个 GPU 入口实例化 |
| 类型注册 | `cusignal_cpp/src/cuda_utils/operator_type_dispatch.h:63`、`.cu:20` | 把 chirp 纳入 53 算子类型证据编号/名称 |
| 直接 smoke | `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:94-200`，`run_type<T>` chirp 段 | CPU/GPU、别名、complex、rank3、异常、空输入 |

### 3. 对外接口

#### 3.1 CPU

```cpp
template<class T>
std::vector<T> chirp_typed_cpu(
    const std::vector<T>& t, float f0, float t1, float f1, float phase = 0);

template<class T>
std::vector<T> chirp_typed_cpu(
    const std::vector<T>& t, float f0, float t1, float f1,
    const std::string& method, float phase = 0, bool vertex_zero = true);

template<class T>
std::vector<ComplexFloat> chirp_complex_typed_cpu(
    const std::vector<T>& t, float f0, float t1, float f1, float phase = 0);
```

第一个 real 重载固定转发为 `linear`；第二个支持四种 method 与别名；complex 固定为 linear，与 Python `type="complex"` 的范围一致。

#### 3.2 GPU

```cpp
template<class T>
void chirp_device(
    const DeviceArray<T>& t, DeviceArray<T>& out,
    float f0, float t1, float f1, float phase = 0);

template<class T>
void chirp_device(
    const DeviceArray<T>& t, DeviceArray<T>& out,
    float f0, float t1, float f1, const std::string& method,
    float phase = 0, bool vertex_zero = true);

template<class T>
void chirp_complex_device(
    const DeviceArray<T>& t, DeviceArray<ComplexFloat>& out,
    float f0, float t1, float f1, float phase = 0);
```

GPU API 不分配输入、输出或 workspace；调用者预分配 `DeviceArray`，保证尺寸一致并管理同步。逻辑 rank 由调用者保存，计算只看展平连续元素数。

### 4. 支持类型与数据语义

显式实例化的 T：

```text
float, __half, int32_t, int16_t, int8_t
```

- real：输入和输出均为 T，中间连续相位计算统一为 FP32。
- complex：五种真实时间输入统一返回 `ComplexFloat {float re, float im}`。
- `SimpleSignalTypePolicy<T>::load` 把 T 提升到连续计算类型。
- 普通整数 `store` 先限幅，再 `static_cast<T>`；源码没有调用 round，因此小数向零截断。头文件注释中的“舍入/饱和”应以实际代码为准：整数路径明确有饱和，但没有显式四舍五入。
- `__half` 使用 `__float2half_rn`，即 round-to-nearest half 转换。
- `ComplexFloat` 是两个相邻 float 字段，而不是 `std::complex<float>`。

### 5. 共享数学核心 `stable_chirp_phase`

函数同时标记 `__host__ __device__ inline`，所以 CPU reference 与 GPU kernel 调用同一份源码。它返回不含用户 `phase` 偏移的弧度累计相位。

#### 5.1 linear

```cpp
const float beta = (f1 - f0) / t1;
const float cycles = fmaf(0.5F * beta * time, time, f0 * time);
return kTwoPi * cycles;
```

$$
\theta(t)=2\pi\left(f_0t+\frac12\beta t^2\right).
$$

`fmaf(a,b,c)` 尽量用一次融合乘加计算 $ab+c$，减少一次中间舍入。

#### 5.2 quadratic，`vertex_zero=true`

```cpp
const float cubic_term = (beta / 3.0F) * time * time * time;
return kTwoPi * fmaf(f0, time, cubic_term);
```

对应 $2\pi(f_0t+\beta t^3/3)$，与通用原理一致。

#### 5.3 quadratic，`vertex_zero=false`

```cpp
const float cube_delta = time *
    fmaf(time, 3.0F * t1 - time, -3.0F * t1 * t1);
const float cycles = fmaf(beta / 3.0F, cube_delta, f1 * time);
```

展开：

$$
\begin{aligned}
\text{cube\_delta}
&=t[t(3t_1-t)-3t_1^2]\\
&=3t_1t^2-t^3-3t_1^2t\\
&=(t_1-t)^3-t_1^3.
\end{aligned}
$$

这与 Python 23.08.00 kernel 的实际表达式等价，避免了 $t\approx0$ 时直接相减两个接近的大立方数，但仍保留 Python 的符号。对相位求导得到

$$
f_{\mathrm{actual}}(t)=f_1-\beta(t_1-t)^2,
$$

而公开数学声明应为 $f_1+\beta(t_1-t)^2$。因此本版本是“对 Python 实际代码的数值稳定等价复现”，不是理论公式的修正。

#### 5.4 logarithmic

使用

```cpp
log_ratio = logf(f1 / f0);
growth = expm1f(log_ratio * time / t1);
```

`expm1f(x)` 直接计算 $e^x-1$，当 $x$ 接近 0 时比 `expf(x)-1` 更能保留小量。端点相等则直接退化为 $2\pi f_0t$。

#### 5.5 hyperbolic

奇点为

$$
t_s=-\frac{f_1t_1}{f_0-f_1}.
$$

当 $1-t/t_s>0$ 时用 `log1pf(-t/t_s)` 稳定计算 $\ln(1-t/t_s)$；否则按 Python 的绝对值语义使用 `logf(fabsf(...))`。奇点本身仍会产生非有限值，wrapper 未检查 `t` 是否命中或跨越奇点。

### 6. CPU 调用链

```text
chirp_typed_cpu(linear shortcut)
  → chirp_typed_cpu(method="linear", vertex_zero=true)
    → parse_cpu_chirp_method
    → log/hyp 频率约束
    → 分配 std::vector<T>
    → degree × π/180
    → 对每个 i：load(time[i])
      → stable_chirp_phase
      → cos(angle)
      → store<T>

chirp_complex_typed_cpu
  → 每个 i：load → stable_chirp_phase(Linear,true)
  → ComplexFloat{cos(angle), sin(angle)}
```

CPU real 为串行 $O(N)$ 循环；除输出 vector 外没有 workspace。CPU 和 GPU 共用相位核心，但各自使用 `std::cos` 与 `cosf/sinf`，并各自执行循环或线程调度。

### 7. GPU wrapper 调用链

```text
chirp_device(shortcut)
  → chirp_device(method="linear", vertex_zero=true)
    → static_assert(T 属于五种输入)
    → 检查 output.size == t.size
    → parse_typed_chirp_method
    → validate_chirp_frequencies
    → 空输入直接返回
    → degree × π/180
    → launch_1d_kernel(chirp_real_kernel, N, ...)

chirp_complex_device
  → 类型与尺寸检查 → 空输入返回 → 角度转换
  → launch_1d_kernel(chirp_complex_kernel, N, ...)
```

校验发生在 `t.empty()` 之前。因此空输入仍会拒绝非法 method、log/hyp 非法频率和 shape 不同；但 `t1=0` 没有公共拒绝，测试还专门记录了空输入 `t1=0` 不被拒绝的现状。

### 8. GPU kernel、grid/block 与线程职责

launch helper 固定：

$$
\text{block.x}=256,
\qquad
\text{grid.x}=\left\lceil\frac{N}{256}\right\rceil.
$$

默认 stream 参数是 `nullptr`。kernel 启动后调用 `CUDA_KERNEL_CHECK()` 检查 launch 错误，但 wrapper 不做完成同步。

每个线程计算：

```cpp
index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= count) return;
```

有效线程只负责一个展平元素：

1. 从 `t[index]` load 并转 FP32；
2. 调用共享相位函数；
3. real 写 `store<T>(cosf(...))`；complex 写 `{cosf, sinf}`；
4. 不访问其他线程数据，不需要 block 同步、归约或原子操作。

内存访问是连续、相邻线程访问相邻元素。没有临时全局缓冲区、FFT、科学库、H2D/D2H 或隐式分配。

### 9. Python—CPU—GPU 三方映射

| 原理或公式 | cuSignal Python | C++ CPU | C++ GPU | 差异 |
| --- | --- | --- | --- | --- |
| method/别名 | `waveforms.py:487-513`（基准） | `waveforms_typed.cpp:157-167` | `waveforms_typed.cu:8-18` | 名称集合一致 |
| degree→rad | Python 基准 `:485` | `.cpp:198,213` | `.cu:52,67` | 都乘 $\pi/180$ |
| linear 相位 | Python基准 `:325-326` | `stable_chirp_phase:29-32` | 同一共享函数 | C++ 用 FP32 `fmaf` |
| quadratic true | Python基准 `:352-354` | `:34-39` | 同一共享函数 | 数学一致 |
| quadratic false | Python基准 `:355-358` | `:40-46` | 同一共享函数 | C++ 因式分解但保留 Python 符号不一致 |
| logarithmic | Python基准 `:371-379` | `:48-54` | 同一共享函数 | C++ 用 `expm1f` 稳定小量 |
| hyperbolic | Python基准 `:389-397` | `:55-64` | 同一共享函数 | C++ 正区间用 `log1pf` |
| real 输出 | `cos(temp+phi)` | `std::cos` + store | `cosf` + store | C++ 输出可量化回 T |
| complex 输出 | 只 linear；Python dtype 随 t | `ComplexFloat` | `ComplexFloat` | C++ 固定 ComplexFP32 |
| 数组语义 | CuPy 广播/shape | 展平 vector | 展平 DeviceArray | C++ 控制参数仅标量，无 Python 数组广播 |

### 10. 显式实例化与链接

CPU `INSTANTIATE_CPU(T)` 和 GPU `I(T)` 都为五种 T 生成正式符号。模板定义放在 `.cpp/.cu`，显式实例化避免调用者必须看到实现，也保证 CMake 目标实际链接所支持的类型。

宏中还包含其他 waveforms 符号，它们不是 chirp 核心；chirp 对应每种 T 的三个符号：real shortcut、real method overload、complex linear。

### 11. 测试如何验证

`e3_wave_window_type_smoke.cu:94-200` 对每个 T：

1. 比较 linear GPU 与 CPU reference；
2. 遍历 quadratic/logarithmic/hyperbolic/linear 的所有短别名；
3. 比较 complex GPU 与 complex CPU；
4. 用 24 个展平元素模拟逻辑 rank3；
5. 检查 CPU/GPU 都拒绝非法 method；
6. 检查 CPU/GPU 都拒绝 log 零频率和 hyp 零频率；
7. 确认空输入配 `t1=0` 当前不公开拒绝；
8. 汇总到 `ok[0]` 和类型证据记录。

已确认的测试边界：

- CPU 和 GPU 调用同一 `stable_chirp_phase`，所以比较不能独立验证相位公式。
- 没有在该段直接与 cuSignal Python 输出或独立解析公式比较。
- 因而 `quadratic false` 的共同符号问题不会被 CPU/GPU 一致性测试发现。
- 本学习阶段没有实际运行测试；这里只说明提交中的测试代码意图。

### 12. 精度、边界与 ZQ500 限制

- 连续相位数学固定 FP32；即使接口未来有更宽存储，也不会自动获得 FP64 相位精度。
- ZQ500 正式输入集合为 FP32、FP16、INT32、INT16、INT8；FP64 不属于正式 GPU business input。
- integer 输出把 $[-1,1]$ 余弦量化/截断到整数，不能保持连续波形精度；typed smoke 主要验证契约一致性。
- `t1=0` 非空输入会进入除零或退化公式，当前 wrapper 没有统一拒绝。
- log 只接受端点非零同号；hyp 只检查端点非零，未检查时间样本奇点。
- 大相位的 `cosf` 参数约简、FP16/整数输入量化以及大 `time/frequency` 会降低精度。
- 默认 stream 异步：调用者必须保证输入输出生命周期，并在读取结果前同步。
- CMake、CUDA 和硬件行为必须以已初始化 ZQ500 `gpu_02` 为准；本阶段未据本地环境作运行结论。

### 13. 复杂度和性能结构

| 项目 | CPU | GPU |
| --- | --- | --- |
| 时间复杂度 | $O(N)$ 串行循环 | $O(N)$ 总工作量，最多 $N$ 个有效线程 |
| 输出空间 | $O(N)$ | 调用者预分配 $O(N)$ |
| 额外 workspace | 无 | 无 |
| 数据搬运 | CPU vector 内部 | wrapper 无隐式 H2D/D2H |
| 同步 | 函数返回即完成 CPU 计算 | 只检查 launch，不等待 kernel 完成 |
| 主要运算 | `cos`，部分 method 有 `log/expm1` | `cosf/sinf` 与 FP32 特殊函数 |

### 14. 阅读顺序

建议严格按以下顺序读 V1：

1. `waveforms_typed.h:27-105`：先理解接口和责任边界；
2. `waveforms_typed.cpp:157-221`：CPU reference；
3. `waveform_math.h:7-67`：逐分支验证相位；
4. `waveforms_typed.cu:8-72`：GPU host wrapper；
5. `kernel_launch.h:10-56`：grid/block 和异步启动；
6. `waveforms_kernels.cuh:44-84`：每线程计算；
7. `.cpp:522-527,557-562` 与 `.cu:216-230`：显式实例化；
8. smoke `:94-200`：理解验证了什么、没验证什么。

### 15. 自检问题与参考答案

#### 15.1 自检问题

1. V1 的正式源码版本是什么？为什么不能只记录当前工作区行号？
2. CPU 与 GPU 为什么能保证使用相同的四种相位公式？这种共享又带来什么测试盲区？
3. `chirp_device` shortcut 最终选择了什么 method 和 `vertex_zero`？
4. GPU wrapper、GPU kernel 和 launch helper 各自负责什么？
5. 一个 GPU 线程负责多少数据？grid 和 block 如何计算？
6. real 与 complex 输出的 dtype 契约有什么区别？
7. 普通整数与 `__half` 的 `store` 语义有什么差异？
8. C++ 对 logarithmic、hyperbolic 和 `t1=0` 分别做了哪些校验？
9. quadratic `vertex_zero=false` 的 C++ 公式是修复 Python，还是等价复现 Python？
10. 为什么 `expm1f` 和 `log1pf` 比直接的 `expf(x)-1`、`logf(1+x)` 更适合小量？
11. GPU wrapper 返回后是否代表计算已经完成？调用者还承担哪些责任？
12. 当前 typed smoke 能证明什么，不能证明什么？

#### 15.2 参考答案

1. V1 是完整 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`。活动源码会继续变化，裸行号无法指出历史内容；完整 SHA、路径、符号和该提交行号共同构成可复核证据。
2. `waveform_math.h:20-67` 的 `stable_chirp_phase` 同时标记 `__host__ __device__`，CPU helper 和 GPU kernel 都调用它。这保证两端实现一致，但 CPU/GPU 对比不能发现共享函数自身的公式错误，需要独立解析公式或 Python oracle。
3. `.cpp:183-184` 和 `.cu:38` 都转发到 `method="linear"`、`vertex_zero=true`，并保留调用者的 `phase`。
4. wrapper 负责模板约束、shape/method/频率校验、degree→rad 和组织参数；kernel 负责线程索引、load、相位、三角函数和 store；`kernel_launch.h:10-56` 负责 grid/block、默认 stream、实际 launch 与 launch error 检查。
5. 每个有效线程只处理 `index` 对应的一个展平元素。`block.x=256`，`grid.x=ceil(N/256)`；越界线程由 `if(index>=count)return` 退出。
6. real 输入输出都是同一 T，正式实例化为 FP32、FP16、INT32、INT16、INT8，中间相位 FP32；complex 接受同五种真实输入，但统一输出两个 float 字段组成的 `ComplexFloat`。
7. 普通整数先限幅到类型范围，再通过 `static_cast<T>` 向零截断；没有显式四舍五入。`__half` 使用 `__float2half_rn`，执行 round-to-nearest half 转换。
8. log 要求 $f_0f_1>0$；hyp 要求两个端点都非零。wrapper 没有统一拒绝 `t1=0`，空输入甚至在 smoke 中明确接受此现状；非空输入可能进入除零或非有限传播。
9. 它是数值稳定的等价复现，不是修复。`cube_delta` 因式分解后仍等于 $(t_1-t)^3-t_1^3$，求导仍得到 $f_1-\beta(t_1-t)^2$。
10. 当 $x$ 很小时，先计算 $e^x$ 或 $1+x$ 再与 1 相减/取对数会损失有效位；`expm1f(x)` 和 `log1pf(x)`直接针对这些小差量计算，减少消减误差。
11. 不代表完成。wrapper 使用默认 stream 异步 launch，只调用 `CUDA_KERNEL_CHECK()` 检查启动错误。调用者必须维持输入输出生命周期，并在读取、复制或释放相关数据前同步和检查执行错误。
12. smoke 能证明五种 T 下 CPU/GPU real、complex、别名、展平 rank3、部分异常和空输入契约一致；它不能独立证明共享相位公式正确，也没有直接对比 cuSignal Python 或解析 oracle，因此发现不了 CPU/GPU 共同的 quadratic false 符号问题。

## 版本差异与原理不变量

当前只有 V1，暂无版本间差异。以后追加 V2/V3 时，应比较相位公式、FP32 策略、类型装卸、grid/block、同步、独立 oracle、精度和性能。无论工程实现如何优化，下列原理应保持：瞬时频率是相位导数，生成波形前必须积分频率律，real 为余弦投影，linear complex 为 $\cos\psi+j\sin\psi$。
