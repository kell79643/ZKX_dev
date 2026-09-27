# cusignal_cpp_vectorstrength 复现逻辑

> 当前阶段：阶段三（`cusignal_cpp` CPU/GPU 复现逻辑）  
> 编写日期：2026-08-16  
> 学习对象：`vectorstrength`  
> 源码仓库：`ZKX`，本阶段始终只读

## 版本索引

| 版本 | 完整提交 SHA | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前学习版本](#v1当前学习版本fdd55ac8) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU FP64 reference；GPU FP32 补偿归约并生成 FP64 storage |

## V1：当前学习版本（fdd55ac8）

### 1. 版本身份与 dirty 门禁

| 项目 | 记录 |
| --- | --- |
| 完整 SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交说明 | `test(archive): 增加任务结果范围治理审计` |
| 仓库状态 | `git status --short` 无输出，整个 `ZKX` 仓库 clean |
| 相关文件 dirty 状态 | 下列接口、CPU、GPU、kernel、类型 helper、FP64 finalizer 与测试文件均无未提交修改 |

本章的每个代码定位都绑定上述完整 SHA。源码通过 `git show <SHA>:<path>` 核对；没有复制、恢复或修改 `ZKX/cusignal_cpp`。

### 2. 文件角色与调用链

| 层次 | 完整 SHA + 共享根目录 `ZKX_dev/` 下相对路径 | 关键符号 | 该提交行号 |
| --- | --- | --- | ---: |
| 公共结果与接口 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.h` | `VectorStrengthCpuResult`、`VectorStrengthDeviceResult`、`vectorstrength_typed_cpu`、`vectorstrength_device` | 710–753 |
| host 类型读取 | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp` | `host_load_double` | 26–33 |
| CPU 核心 | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp` | `vectorstrength_typed_cpu` 两个 overload | 441–480 |
| GPU host wrapper | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp` | `vectorstrength_device` 两个 overload | 496–524 |
| GPU detail 声明 | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp` | `detail::vectorstrength_fp32_compute_device` | 61–63 |
| GPU launch 实现 | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cu` | `detail::vectorstrength_fp32_compute_device` | 527 |
| GPU device 核心 | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh` | `vectorstrength_load`、`vectorstrength_multi_kernel` | 692–696、740–792 |
| 未接入 kernel | 同 SHA + 同一 `.cuh` | `vectorstrength_single_kernel` | 698–738 |
| 补偿累加 | 同 SHA + `ZKX/cusignal_cpp/src/cuda_utils/operator_compute_policy.h` | `CompensatedFloatAccumulator` | 90–116 |
| FP64 返回封装 | 同 SHA + `ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h` | `finalize_fp64_device_storage` | 40–47 |
| FP32→FP64 storage | 同 SHA + `ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu` | `widened_storage_bits`、`widen_real_storage_kernel`、`widen_fp32_storage_device` | 11–50、63–73 |
| 类型实例化 | 同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp` | `INSTANTIATE_SPECTRAL_CPU` 中的 vectorstrength 项 | 1374–1381、1393–1397 |
| 直接 typed smoke | 同 SHA + `ZKX/cusignal_cpp/test/signal_processing/e3_spectral_type_smoke.cu` | `run_type<T>` 中 vectorstrength case、`main` | 317、353–357、366、370 |

当前 GPU 调用链为：

```text
vectorstrength_device(events, scalar period)
  ├─ host 验证 scalar period
  ├─ H2D：构造单元素 DeviceArray<T> periods
  └─ vectorstrength_device(events, periods)
       ├─ D2H：periods.to_host() 做正周期验证
       ├─ 分配 FP32 strength/phase
       ├─ detail::vectorstrength_fp32_compute_device
       │    └─ vectorstrength_multi_kernel<<<P, 256>>>
       ├─ FP32 storage → FP64 storage 位编码 kernel
       └─ 返回 VectorStrengthDeviceResult
```

`vectorstrength_single_kernel` 不在这条链上。对该完整提交执行 `git grep vectorstrength_single_kernel` 只命中定义 `spectral_analysis_kernels.cuh:699`，没有 launch 或调用证据。

### 3. 公共接口、结果结构与模板职责

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.h`，符号 `VectorStrengthCpuResult`、`VectorStrengthDeviceResult`、四个公开 overload，第 710–753 行。

```cpp
struct VectorStrengthCpuResult {
    std::vector<double> strength;
    std::vector<double> phase;
    bool scalar_period{false};
};

struct VectorStrengthDeviceResult {
    DeviceArray<double> strength;
    DeviceArray<double> phase;
    bool scalar_period{false};
};
```

- CPU 结果把 strength 和 phase 存在 host `std::vector<double>` 中；GPU 结果保留在 device `DeviceArray<double>` 中。
- 两者都用 `scalar_period` 记录调用者传入的是标量周期还是周期数组。
- C++ 返回容器始终可表达 0、1 或多个元素；所谓“标量形态”是 bool 元数据，不像 Python 那样真的把长度 1 数组索引成标量。
- `{false}` 是类内成员初始化，数组 period overload 无需显式赋值就保持 false。

同一 SHA、同一路径中，CPU 声明位于符号 `vectorstrength_typed_cpu` 的第 731–735 行：

```cpp
template <class T> VectorStrengthCpuResult vectorstrength_typed_cpu(
    const std::vector<T>& events, T period);

template <class T> VectorStrengthCpuResult vectorstrength_typed_cpu(
    const std::vector<T>& events, const std::vector<T>& periods);
```

GPU 声明位于符号 `vectorstrength_device` 的第 749–753 行：

```cpp
template <class T> VectorStrengthDeviceResult vectorstrength_device(
    const DeviceArray<T>& events, T period);

template <class T> VectorStrengthDeviceResult vectorstrength_device(
    const DeviceArray<T>& events, const DeviceArray<T>& periods);
```

- `template <class T>` 让同一算法支持不同输入类型。
- CPU 输入通过 `const std::vector<T>&` 只读引用传入；GPU 输入通过 `const DeviceArray<T>&` 只读引用传入。
- 每侧各有标量 period 和数组 periods 两个 overload；函数名相同，由第二参数类型选择。
- 正式支持的 $T$ 为 `float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t`，由显式实例化固定，而不是允许任意 C++ 类型。

### 4. CPU reference：从输入类型转为 double

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `host_load_double`，第 26–33 行。

```cpp
template <class T>
double host_load_double(T value)
{
    if constexpr (std::is_same_v<T, __half>) {
        return static_cast<double>(__half2float(value));
    }
    return static_cast<double>(value);
}
```

- `if constexpr` 在模板实例化时选择分支；未选择分支不会参与该类型的编译。
- `__half` 先由 `__half2float` 解码，再转成 `double`；其他四种输入直接 `static_cast<double>`。
- 该 helper 只做 host 类型提升，不改变物理单位。
- CPU 核心后续全部用 `double` 三角函数和累加，因此它是独立 FP64 reference，不调用 GPU kernel。

### 5. CPU period 数组 overload：完整计算

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `vectorstrength_typed_cpu(const std::vector<T>&, const std::vector<T>&)`，第 441–472 行。

#### 5.1 初始化结果与逐周期循环

```cpp
template <class T>
VectorStrengthCpuResult vectorstrength_typed_cpu(
    const std::vector<T>& events, const std::vector<T>& periods)
{
    VectorStrengthCpuResult result;
    result.strength.resize(periods.size());
    result.phase.resize(periods.size());
    constexpr double pi = 3.14159265358979323846;
    for (std::size_t period_index = 0; period_index < periods.size(); ++period_index) {
```

- 函数返回 `VectorStrengthCpuResult`；参数保持只读。
- 两个输出 resize 到 $P=\texttt{periods.size()}$，确保逐周期一一对应。
- `constexpr double pi` 是编译期 double 常量。
- `for` 从 `period_index=0` 遍历到 $P-1$；空 periods 时循环不执行，返回两个空 vector。

#### 5.2 验证周期并处理空事件

```cpp
        const double period = host_load_double(periods[period_index]);
        if (period <= 0.0) {
            throw std::invalid_argument("periods must be positive");
        }
        if (events.empty()) {
            result.strength[period_index] = std::numeric_limits<double>::quiet_NaN();
            result.phase[period_index] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
```

- 当前 period 先提升为 double，再检查严格大于零；失败立即抛 `std::invalid_argument`。
- `events.empty()` 明确复现 cuSignal 空事件平均值的 NaN 语义；strength 和 phase 都写 quiet NaN。
- `continue` 跳到下一个 period，避免除以零。

#### 5.3 顺序累加正弦与余弦

```cpp
        double real = 0.0;
        double imag = 0.0;
        for (const auto& event : events) {
            const double angle = 2.0 * pi * host_load_double(event) / period;
            real += std::cos(angle);
            imag += std::sin(angle);
        }
```

- `real` 和 `imag` 分别累计 $\sum_i\cos\theta_i$ 与 $\sum_i\sin\theta_i$。
- range-for 的 `const auto& event` 逐项只读访问事件。
- `angle=2\pi t_i/T_p` 对应 Python 的正指数 $e^{j2\pi t_i/T_p}$。
- CPU 使用 `std::cos`、`std::sin` 的 double overload，按事件顺序串行累加。

#### 5.4 归一化、模、辐角与返回

```cpp
        real /= static_cast<double>(events.size());
        imag /= static_cast<double>(events.size());
        result.strength[period_index] = std::sqrt(real * real + imag * imag);
        result.phase[period_index] = std::atan2(imag, real);
    }
    return result;
}
```

- 两个分量除以 $N$ 得复均值 $\bar z=C+jS$。
- `sqrt(real*real+imag*imag)` 得 $R=|\bar z|$。
- `atan2(imag,real)` 得正指数约定下的平均相位。
- 循环闭合后返回结果；数组 overload 的 `scalar_period` 保持 false。

### 6. CPU 标量 period overload：复用数组算法

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `vectorstrength_typed_cpu(const std::vector<T>&, T)`，第 474–480 行。

```cpp
template <class T>
VectorStrengthCpuResult vectorstrength_typed_cpu(const std::vector<T>& events, T period)
{
    auto result = vectorstrength_typed_cpu(events, std::vector<T>{period});
    result.scalar_period = true;
    return result;
}
```

- `std::vector<T>{period}` 用初始化列表构造一个元素的周期数组。
- overload resolution 随后调用上一节的数组版本，不复制算法。
- 计算完成后只把 `scalar_period` 改为 true；strength/phase 容器仍各有一个元素。

### 7. GPU host wrapper：数组 period 路径

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `vectorstrength_device(const DeviceArray<T>&, const DeviceArray<T>&)`，第 496–512 行。

#### 7.1 D2H 周期验证

```cpp
template <class T>
VectorStrengthDeviceResult vectorstrength_device(
    const DeviceArray<T>& events, const DeviceArray<T>& periods)
{
    for (const T period : periods.to_host()) {
        if (host_load_double(period) <= 0.0) {
            throw std::invalid_argument("periods must be positive");
        }
    }
```

- `periods.to_host()` 把整个 period 数组从 device 复制到 host；这是真实 D2H 边界，通常还要求先前相关 device 工作完成。
- range-for 在 host 逐项验证正周期；任一非法值终止调用。
- 这一步使 period 数组验证成本为 $O(P)$ 并产生传输/同步开销；events 不复制回 host。

#### 7.2 FP32 中间结果、device 计算与 FP64 storage

```cpp
    DeviceArray<float> strength(periods.size());
    DeviceArray<float> phase(periods.size());
    detail::vectorstrength_fp32_compute_device(events, periods, strength, phase);
    VectorStrengthDeviceResult result;
    result.strength = cuda_utils::finalize_fp64_device_storage(strength);
    result.phase = cuda_utils::finalize_fp64_device_storage(phase);
    return result;
}
```

- 两个 FP32 device 临时数组长度均为 $P$。
- detail 函数启动真正的 vectorstrength kernel。
- `finalize_fp64_device_storage` 分别创建 FP64 storage 输出；它不提高已经完成的 FP32 科学计算精度。
- 未设置 `scalar_period`，因此数组路径保持 false。
- wrapper 没有提供外部 stream 或可复用 workspace；每次调用都分配临时和正式输出。

### 8. GPU host wrapper：标量 period 路径

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `vectorstrength_device(const DeviceArray<T>&, T)`，第 514–524 行。

```cpp
template <class T>
VectorStrengthDeviceResult vectorstrength_device(const DeviceArray<T>& events, T period)
{
    if (host_load_double(period) <= 0.0) {
        throw std::invalid_argument("periods must be positive");
    }
    const DeviceArray<T> periods = DeviceArray<T>::from_host({period});
    auto result = vectorstrength_device(events, periods);
    result.scalar_period = true;
    return result;
}
```

- 标量先在 host 验证，再由 `from_host({period})` 做一次单元素 H2D。
- 随后复用数组 wrapper；数组 wrapper 又执行 `periods.to_host()`，所以当前标量路径会再次 D2H 验证同一值。
- 当前标量路径仍启动 multi-period kernel，只是 $P=1$；未调用源码中的 single kernel。
- 最后设置 scalar marker 并返回长度 1 的两项 device 容器。

### 9. GPU detail launch：grid、block 与错误检查

声明定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `detail::vectorstrength_fp32_compute_device`，第 61–63 行。

```cpp
template <class T> void vectorstrength_fp32_compute_device(
    const DeviceArray<T>& events, const DeviceArray<T>& periods,
    DeviceArray<float>& strength, DeviceArray<float>& phase);
```

定义定位：同 SHA + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cu`，同符号，第 527 行。源码原本压缩在一个物理行中：

```cpp
template<class T>void vectorstrength_fp32_compute_device(const DeviceArray<T>&e,const DeviceArray<T>&p,DeviceArray<float>&s,DeviceArray<float>&phase){static_assert(is_simple_signal_input_v<T>);if(p.size()!=s.size()||p.size()!=phase.size())throw std::invalid_argument("vectorstrength_device: strength and phase output sizes must match period count");if(p.empty())return;constexpr int threads=256;spectral_detail::vectorstrength_multi_kernel<T,T,float,float><<<static_cast<unsigned int>(p.size()),threads>>>(e.data(),(int)e.size(),p.data(),(int)p.size(),s.data(),phase.data());CUDA_CHECK(cudaGetLastError());}
```

按源码从左到右解释：

1. `static_assert(is_simple_signal_input_v<T>)` 在编译期拒绝契约外输入类型。
2. strength、phase 的长度必须都等于 period 数；公开 wrapper 按 $P$ 分配，正常路径会满足。
3. $P=0$ 时直接返回，不启动 kernel。
4. block 固定 `threads=256`。
5. launch 配置为 `<<<P,256>>>`：一个 block 对应一个 period。
6. `e.size()`、`p.size()` 被转成 `int` 传入；源码没有在此检查超出 `int` 范围的极大输入。
7. 模板参数 `<T,T,float,float>` 表示 event/period 保持业务输入类型，累计与输出临时值固定 FP32。
8. `CUDA_CHECK(cudaGetLastError())` 检查 launch 提交阶段错误；这里没有显式 device synchronize。

### 10. GPU device kernel：类型加载

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh`，符号 `vectorstrength_load`，第 692–696 行。

```cpp
template <class Accumulator, class T>
__device__ Accumulator vectorstrength_load(T value)
{
    return static_cast<Accumulator>(detail::SimpleSignalTypePolicy<T>::load(value));
}
```

- `__device__` 表示只能从 device 代码调用。
- `SimpleSignalTypePolicy<T>::load` 把 FP16 或整数业务输入转换到连续计算类型。
- 外层 `static_cast<Accumulator>` 在当前实例中得到 float；因此五种输入都以 FP32 进入三角计算。

### 11. GPU device kernel：一个 block 负责一个 period

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh`，符号 `vectorstrength_multi_kernel`，第 740–792 行。

#### 11.1 签名、共享内存和二维职责映射

```cpp
template <class Input, class Period, class Accumulator, class Output>
__global__ void vectorstrength_multi_kernel(
    const Input* events,
    int event_count,
    const Period* periods,
    int period_count,
    Output* strength,
    Output* phase)
{
    __shared__ Accumulator sine_sums[256];
    __shared__ Accumulator cosine_sums[256];
    const int period_index = static_cast<int>(blockIdx.x);
    const int thread_index = threadIdx.x;
```

- `__global__` 是 CUDA kernel 入口；host 用 launch syntax 调用。
- 两个 shared 数组各有 256 个 float，与固定 block size 一一对应，总 shared memory 为 $2\times256\times4=2048$ bytes/block。
- `blockIdx.x` 选择 period，`threadIdx.x` 选择 block 内线程。

#### 11.2 period 防越界与角频率

```cpp
    if (period_index >= period_count) {
        return;
    }
    const Accumulator two_pi = static_cast<Accumulator>(6.28318530717958647692);
    const Accumulator omega = two_pi
        / vectorstrength_load<Accumulator>(periods[period_index]);
    static_assert(std::is_same_v<Accumulator, float>);
```

- 当前 grid 大小恰为 $P$，guard 通常不会触发，但保护了 period 索引。
- $\omega_p=2\pi/T_p$ 在 FP32 中计算。
- `static_assert` 明确禁止把当前 kernel 的累计类型改成非 float 而不触发编译错误。

#### 11.3 每线程跨步处理事件并做补偿累加

```cpp
    compute_policy::CompensatedFloatAccumulator sine_sum_acc;
    compute_policy::CompensatedFloatAccumulator cosine_sum_acc;
    for (int index = thread_index; index < event_count; index += blockDim.x) {
        const Accumulator angle = vectorstrength_load<Accumulator>(events[index]) * omega;
        sine_sum_acc.add(sin(angle));
        cosine_sum_acc.add(cos(angle));
    }
```

- 线程 $q$ 处理 $q,q+256,q+512,\ldots$ 的事件；每个事件恰由一个线程负责。
- `sin(angle)` 和 `cos(angle)` 对 float 实参执行 FP32 三角计算。
- 每个线程分别用 Neumaier 补偿器累计自己的正弦和余弦子序列，减少长 FP32 串行和的舍入误差。

#### 11.4 写 shared memory 并同步

```cpp
    sine_sums[thread_index] = sine_sum_acc.value();
    cosine_sums[thread_index] = cosine_sum_acc.value();
    __syncthreads();
```

- 每线程把补偿后的两个 partial sum 写到自己的 shared slot。
- `__syncthreads()` 保证所有 256 个 slot 完成后才进入树归约；即使线程没有事件，它的初值仍为 0。

#### 11.5 shared-memory 二叉树归约

```cpp
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            sine_sums[thread_index] += sine_sums[thread_index + stride];
            cosine_sums[thread_index] += cosine_sums[thread_index + stride];
        }
        __syncthreads();
    }
```

- stride 依次为 128、64、32、16、8、4、2、1。
- 前 stride 个线程把后一半 partial sum 合并到前一半。
- 每轮同步后才能安全进入下一层；最终 `sine_sums[0]`、`cosine_sums[0]` 保存全 block 总和。
- per-thread 局部和使用补偿，但 shared tree 的 `+=` 本身是普通 FP32 加法。

#### 11.6 空事件、strength、phase 与 kernel 结束

```cpp
    if (thread_index == 0) {
        if (event_count == 0) {
            strength[period_index] = static_cast<Output>(nanf(""));
            phase[period_index] = static_cast<Output>(nanf(""));
            return;
        }
        const Accumulator magnitude = sqrt(
            sine_sums[0] * sine_sums[0]
            + cosine_sums[0] * cosine_sums[0]);
        strength[period_index] = static_cast<Output>(
            magnitude / static_cast<Accumulator>(event_count));
        phase[period_index] = static_cast<Output>(
            atan2(sine_sums[0], cosine_sums[0]));
    }
}
```

- 只有 thread 0 写该 period 的两个输出，避免数据竞争。
- 空 events 时两个 FP32 输出都是 NaN，和 CPU reference、cuSignal Python 的空均值语义对齐。
- 非空时先算未归一化合成向量长度，再除以 $N$；这与“先分别除以 $N$ 再求模”代数等价。
- `atan2(sum_sin,sum_cos)` 不需要除以 $N$，因为正比例缩放不改变方向。
- 最后的花括号依次关闭 thread-0 分支和 kernel。

### 12. 补偿累加 helper

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/cuda_utils/operator_compute_policy.h`，符号 `CompensatedFloatAccumulator`，第 90–116 行。

```cpp
// Neumaier compensation keeps long FP32 reductions stable without requiring
// unsupported device FP64.  Call value() only after all terms have been added.
struct CompensatedFloatAccumulator {
    float sum{0.0F};
    float correction{0.0F};

    __host__ __device__ void add(float value)
    {
        const float next = sum + value;
        if (fabsf(sum) >= fabsf(value)) {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }

    __host__ __device__ void add_product(float left, float right)
    {
        add(fmaf(left, right, 0.0F));
    }

    __host__ __device__ float value() const
    {
        return sum + correction;
    }
};
```

- `sum` 保存主和，`correction` 保存浮点加法丢失的低位补偿。
- `next=sum+value` 先执行普通 FP32 加法；随后按较大绝对值项选择 Neumaier 误差恢复公式。
- vectorstrength 只调用 `add` 和 `value`，不调用 `add_product`；后者仍是该完整 helper 的组成。
- `__host__ __device__` 允许同一小型类型在 host/device 编译，但当前 vectorstrength kernel 在 device 使用。

### 13. FP32 结果到 FP64 storage 的收尾

#### 13.1 公开 finalizer

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`，符号 `finalize_fp64_device_storage`，第 40–47 行。

```cpp
inline DeviceArray<double> finalize_fp64_device_storage(
    const DeviceArray<float>& computed)
{
    DeviceArray<double> output(computed.size());
    widen_fp32_storage_device(
        computed.data(), output.data(), computed.size());
    return output;
}
```

- 先分配同元素数的 `DeviceArray<double>`。
- `widen_fp32_storage_device` 在 device 上把每个已计算 FP32 值转换成 IEEE-754 double storage bits。
- 返回类型是 double storage，但有效科学精度仍来自 FP32 kernel；不能把它描述成 GPU FP64 三角计算或 FP64 归约。

#### 13.2 storage 位编码和 real widening kernel

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu`，符号 `widened_storage_bits`、`widen_real_storage_kernel`、`widen_fp32_storage_device`，第 11–50、63–73 行。

```cpp
__device__ std::uint64_t widened_storage_bits(float value)
{
    union FloatStorageBits {
        float value;
        std::uint32_t bits;
    } source{value};
    const std::uint64_t sign =
        static_cast<std::uint64_t>(source.bits >> 31U) << 63U;
    const std::uint32_t exponent = (source.bits >> 23U) & 0xffU;
    std::uint32_t fraction = source.bits & 0x7fffffU;
```

- union 读取 FP32 的原始 32 位表示。
- sign、exponent、fraction 分别提取符号位、8 位指数和 23 位尾数。
- 后续分支分别处理 subnormal/zero、Inf/NaN 和 normal 数，并组合出 64 位 double 表示；这属于整数位运算，不执行 device FP64 算术。

```cpp
    if (exponent == 0U) {
        if (fraction == 0U) return sign;
        int shift = 0;
        while ((fraction & 0x400000U) == 0U) {
            fraction <<= 1U;
            ++shift;
        }
        const std::uint64_t widened_exponent =
            static_cast<std::uint64_t>(896 - shift) << 52U;
        const std::uint64_t widened_fraction =
            static_cast<std::uint64_t>(fraction & 0x3fffffU) << 30U;
        return sign | widened_exponent | widened_fraction;
    }
    if (exponent == 0xffU) {
        return sign | (UINT64_C(0x7ff) << 52U) |
            (static_cast<std::uint64_t>(fraction) << 29U);
    }
    const std::uint64_t widened_exponent =
        static_cast<std::uint64_t>(exponent + 896U) << 52U;
    const std::uint64_t widened_fraction =
        static_cast<std::uint64_t>(fraction) << 29U;
    return sign | widened_exponent | widened_fraction;
}
```

- exponent 0 且 fraction 0 是带符号零；subnormal 先把最高有效位移到规范位置。
- exponent 255 保持 Inf/NaN 类别并搬移 payload。
- normal FP32 的 exponent bias 127 转到 FP64 bias 1023，差值为 896；尾数左移补到 52 位。

定位：同一完整 SHA、同一 `.cu`，符号 `widen_real_storage_kernel`，第 45–50 行。

```cpp
__global__ void widen_real_storage_kernel(
    const float* input, std::uint64_t* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = widened_storage_bits(input[index]);
}
```

定位：同一完整 SHA、同一 `.cu`，符号 `widen_fp32_storage_device`，第 63–73 行。

```cpp
void widen_fp32_storage_device(
    const float* input, void* output_storage, std::size_t count)
{
    if (count == 0) return;
    launch_1d_kernel(
        widen_real_storage_kernel,
        count,
        input,
        static_cast<std::uint64_t*>(output_storage),
        count);
}
```

- widening kernel 采用普通一维全局索引，每个线程负责一个输出 storage 元素。
- `void*` 输出地址转为 `uint64_t*`，写入刚生成的 double 位模式。
- 空输出不 launch；非空通过公共 `launch_1d_kernel` 启动。
- vectorstrength 对 strength 和 phase 各调用一次，因此科学 kernel 之后还有两个 widening kernel。

### 14. 五类型分发与显式实例化

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，宏 `INSTANTIATE_SPECTRAL_CPU` 中的 vectorstrength 项，第 1374–1381 行；宏调用第 1393–1397 行。

```cpp
    template VectorStrengthCpuResult vectorstrength_typed_cpu( \
        const std::vector<T>&, T); \
    template VectorStrengthCpuResult vectorstrength_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&); \
    template VectorStrengthDeviceResult vectorstrength_device( \
        const DeviceArray<T>&, T); \
    template VectorStrengthDeviceResult vectorstrength_device( \
        const DeviceArray<T>&, const DeviceArray<T>&); \
```

反斜杠把这些声明连接进一个宏体，为同一 $T$ 显式实例化 CPU/GPU 的标量和数组 overload。宏随后对五种类型展开：

```cpp
INSTANTIATE_SPECTRAL_CPU(float);
INSTANTIATE_SPECTRAL_CPU(__half);
INSTANTIATE_SPECTRAL_CPU(std::int32_t);
INSTANTIATE_SPECTRAL_CPU(std::int16_t);
INSTANTIATE_SPECTRAL_CPU(std::int8_t);
```

GPU detail `.cu` 还在同 SHA 的 `spectral_analysis_typed.cu:530–531` 为 `vectorstrength_fp32_compute_device` 对同五类型显式实例化。由此可确认正式输入集合是 FP32、FP16、INT32、INT16、INT8；FP64 仅是结果 storage，不是 GPU 业务输入或 device 计算类型。

### 15. 数学—Python—CPU—GPU 三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $\omega_p=2\pi/T_p$ | 基准 `spectral.py:1539` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`spectral_analysis_typed.cpp:448,450,462` | 同 SHA，`spectral_analysis_kernels.cuh:756–758` | CPU double；GPU float |
| $\sum_i\cos(\omega_pt_i)$ | `cp.exp` 后复均值实部 | 同 SHA，`spectral_analysis_typed.cpp:459,463` | 同 SHA，`spectral_analysis_kernels.cuh:761–768,771–776` | Python 物化复矩阵；CPU 顺序 double；GPU 补偿 partial + shared tree |
| $\sum_i\sin(\omega_pt_i)$ | `cp.exp` 后复均值虚部 | 同 SHA，`spectral_analysis_typed.cpp:460,464` | 同 SHA，同一 kernel 行 | 正指数约定一致 |
| $R_p=\sqrt{C_p^2+S_p^2}$ | `spectral.py:1543–1544` | 同 SHA，`spectral_analysis_typed.cpp:466–468` | 同 SHA，`spectral_analysis_kernels.cuh:784–788` | GPU 先对总和求模再除 $N$，代数等价 |
| $\phi_p=\operatorname{atan2}(S_p,C_p)$ | `spectral.py:1545` | 同 SHA，`spectral_analysis_typed.cpp:469` | 同 SHA，`spectral_analysis_kernels.cuh:789–790` | 除以正数 $N$ 不改变相角 |
| 空事件返回 NaN | `cp.mean` 的空均值传播 | 同 SHA，`spectral_analysis_typed.cpp:454–457` | 同 SHA，`spectral_analysis_kernels.cuh:779–782` | C++ 显式实现，GPU 先写 FP32 NaN 再扩展 storage |
| 标量/数组形态 | Python 返回 scalar 或 `(P,)` | result vector + bool | DeviceArray + bool | C++ 用 marker 表达 scalar contract |

### 16. 与 cuSignal Python 相同、等价替换和有意不同之处

| 项目 | 关系 | 说明 |
| --- | --- | --- |
| 数学定义 | 相同 | 都用正相位 $2\pi t/T$，返回平均单位向量的模和角 |
| 多周期 | 相同 | 每个周期独立得到 strength/phase |
| 空 events | 相同语义 | 都得到 NaN strength/phase；C++ 显式分支 |
| 非正 period | 相同 | 都抛异常，错误消息均包含 `periods must be positive` |
| Python 数组实现 vs GPU kernel | 等价替换 | Python 物化 `(P,N)` 复矩阵；C++ GPU 每 period 一个 block，直接归约，不物化该矩阵 |
| CPU reference | 精度扩展 | CPU 用 double，而 Python dtype 取决于 CuPy promotion；它是独立 reference |
| GPU compute/output | 工程近似 | GPU 科学计算固定 FP32，再提供 FP64 storage；不等于 Python ComplexFP64 全精度计算 |
| scalar 返回 | 接口适配 | Python 真正拆成 scalar；C++ 保留长度 1 容器并设 bool marker |
| period 验证 | 工程实现差异 | GPU 数组 period 会 D2H 验证；Python 在 device 上执行 `(period<=0).any()` |

### 17. 内存布局、数据搬运、同步与错误处理

- `events`、`periods` 是连续 `DeviceArray<T>` 一维存储，kernel 通过裸指针线性读取。
- strength、phase 的 FP32 临时数组和 FP64 正式数组都为长度 $P$ 的一维连续存储。
- 数组 period 路径有一次 period D2H；标量路径先 H2D 单元素，再因复用数组 wrapper 发生一次 D2H。
- events 不发生 D2H；科学输出也不经 host，两个 widening kernel 在 device 完成 storage 转换。
- kernel 内同步只使用 block 范围的 `__syncthreads()`；不同 period blocks 互不通信。
- wrapper 没有显式 `cudaDeviceSynchronize`；launch 后用 `CUDA_CHECK(cudaGetLastError())` 检查提交错误。真正异步执行错误通常需后续同步边界才能暴露。
- 当前接口没有外部 stream、workspace 或输出复用参数；分配和 period 验证可能主导小输入时延。
- 不调用 FFT、插值、卷积或平台科学库；核心是自定义三角函数与 shared reduction。

### 18. 复杂度与性能结构

设事件数 $N$、period 数 $P$、block size $B=256$。

#### CPU

- 总工作：$O(PN)$；
- 输出空间：$O(P)$；
- 额外工作空间：$O(1)$（不计返回容器）；
- 每个 period 串行遍历全部事件。

#### GPU

- 总工作：$O(PN+PB)$；
- 每个 block 的主线程路径约处理 $\lceil N/B\rceil$ 个事件，再经历 $O(\log B)$ 轮树归约；
- grid 有 $P$ 个互相独立的 blocks；
- shared memory：2048 bytes/block；
- FP32 临时与 FP64 正式输出合计为 $O(P)$；不创建 Python 版本的 $O(PN)$ 复矩阵；
- 另有 period D2H、两次 output widening launch 和多次分配。小 $N,P$ 下这些固定成本可能超过核心归约。

以上是源码结构分析，不是本次任务新测得的性能结论。

### 19. 精度、边界条件与 ZQ500 限制

1. **CPU/GPU 精度不同。**CPU 使用 double 三角函数和顺序累加；GPU 使用 float 三角函数、per-thread Neumaier 补偿和普通 FP32 shared tree。两者只能按容差比较。
2. **FP64 是接口/storage。**GPU 返回 `DeviceArray<double>`，但结果数值只具有 FP32 科学计算精度；storage widening 不补回丢失位。
3. **近抵消 phase 不稳定。**当合成向量接近零，微小的累计顺序或三角约简误差会显著改变 `atan2`；这属于相位本身的条件不良。
4. **大相位约简风险。**GPU 直接对 float $2\pi t/T$ 调三角函数，没有先做高精度 modulo；极大 $|t/T|$ 会损失相位低位。
5. **固定 256 threads。**shared 数组长度也是 256；当前 launch 与 kernel 假设一致。若未来单独改变 block size，必须同步审查 shared 边界和二叉树归约。
6. **int size 转换。**event/period count 从 `std::size_t` 转为 `int`，没有显式超范围检查。
7. **空 periods。**CPU 返回空 vectors；GPU detail 在 $P=0$ 时不 launch，并返回空正式输出。
8. **空 events。**CPU 写 double NaN；GPU 写 float NaN 再保留 IEEE NaN 类别到 FP64 storage。
9. **period 数组验证有 host 边界。**它不适合被描述为完全异步、全 device pipeline。
10. **ZQ500 正式 kernel 不做 device FP64 算术。**当前源码通过 FP32 compute + integer storage-bit widening 遵守这一平台约束。

### 20. 未接入的 `vectorstrength_single_kernel`

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh`，符号 `vectorstrength_single_kernel`，第 698–738 行。

这个 kernel 与 multi kernel 共享 256-thread 正弦/余弦归约，但只输出 strength，不输出 phase；空事件分支写 0 而不是 NaN。完整提交内没有任何调用位置，因此：

- 不能把它列入当前标量 API 调用链；
- 不能用它的空事件行为解释公开 API；
- 当前标量 API 的真实语义来自“单元素 periods + multi kernel”。

保留该符号可能是历史实现或备用路径；在没有提交证据前，原因只能标为“未知”，不能推测成性能优化。

### 21. 直接测试如何验证，以及覆盖风险

定位：`fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/test/signal_processing/e3_spectral_type_smoke.cu`，`run_type<T>` 的 vectorstrength case，第 317 行。

```cpp
    const auto scalar_strength=vectorstrength_device(d_time,convert<T>(8));const auto scalar_reference=vectorstrength_typed_cpu(time,convert<T>(8));std::vector<T>periods{convert<T>(8),convert<T>(4)};auto d_periods=DeviceArray<T>::from_host(periods);const auto array_strength=vectorstrength_device(d_time,d_periods);const auto array_reference=vectorstrength_typed_cpu(time,periods);DeviceArray<T>empty_events;const auto empty_strength=vectorstrength_device(empty_events,convert<T>(8));bool vector_gpu_rejects_period=false,vector_cpu_rejects_period=false;try{(void)vectorstrength_device(d_time,convert<T>(0));}catch(const std::invalid_argument& error){vector_gpu_rejects_period=std::string(error.what()).find("period")!=std::string::npos;}try{(void)vectorstrength_typed_cpu(time,convert<T>(0));}catch(const std::invalid_argument& error){vector_cpu_rejects_period=std::string(error.what()).find("period")!=std::string::npos;}(void)vectorstrength_device(d_time,convert<T>(8));ok[5]=scalar_strength.scalar_period&&scalar_reference.scalar_period&&!array_strength.scalar_period&&!array_reference.scalar_period&&matches_double(scalar_strength.strength.to_host(),scalar_reference.strength)&&matches_double(scalar_strength.phase.to_host(),scalar_reference.phase)&&matches_double(array_strength.strength.to_host(),array_reference.strength)&&matches_double(array_strength.phase.to_host(),array_reference.phase)&&std::isnan(empty_strength.strength.to_host()[0])&&std::isnan(empty_strength.phase.to_host()[0])&&vector_gpu_rejects_period&&vector_cpu_rejects_period;
```

该物理行虽被压缩，但依次执行：

1. 用已有 `time=[0,1,2,4,7]` 构造 scalar period 8 的 GPU 与 CPU 结果；
2. 构造 periods `[8,4]`，比较数组 GPU/CPU；
3. 用空 `DeviceArray<T>` events 检查 strength/phase 都为 NaN；
4. 分别检查 GPU 和 CPU 对 period 0 抛 `std::invalid_argument` 且消息包含 `period`；
5. 检查 scalar marker 为 true、array marker 为 false；
6. 用 `matches_double` 比较 scalar/array 的 strength 和 phase。

同一完整 SHA、同一文件 `:353–357` 在 accuracy evidence 上下文再次比较四组数值：

```cpp
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::vectorstrength);
    (void)matches_double(scalar_strength.strength.to_host(),scalar_reference.strength);
    (void)matches_double(scalar_strength.phase.to_host(),scalar_reference.phase);
    (void)matches_double(array_strength.strength.to_host(),array_reference.strength);
    (void)matches_double(array_strength.phase.to_host(),array_reference.phase);
```

`:366` 把 `vectorstrength` 以 double 输出、2 个元素和 `ok[5]` 记录。测试文件最后的 `main` 位于 `:370`：

```cpp
int main(){int passed=0;passed+=run_type<float>("fp32");passed+=run_type<__half>("fp16");passed+=run_type<int32_t>("int32");passed+=run_type<int16_t>("int16");passed+=run_type<int8_t>("int8");std::cout<<"[E3][spectral] total=30 pass="<<passed<<" fail="<<(30-passed)<<'\n';return passed==30?0:1;}
```

该行按 FP32、FP16、INT32、INT16、INT8 顺序调用 `run_type`，累计通过数，输出总计 30 项并以 `passed==30` 决定进程退出码。这是直接测试代码的最后一行。

仓库文档 `ZKX/cusignal_cpp/docs/OPERATOR_PLATFORMIZATION_E3_GPU_TYPE_IMPLEMENTATION.md:241` 把五类型标为 `zq500-verified`，但其说明仍写“strength/phase 为 float”；当前源码已经增加 FP64 storage finalizer。`ZKX/cusignal_cpp/docs/OPERATOR_PLATFORMIZATION_D6_FP32_ACCURACY.md:111` 对 multi-period 精度状态仍为 `command-ready; not-run`。因此证据边界是：

- 历史 E3/D4 记录证明过 ZQ500 typed/reduction 路径；
- 当前提交的静态测试源码覆盖上述语义；
- 本学习任务没有重新同步、构建或运行当前 SHA，不能声称当前提交刚刚通过 ZQ500；
- 尚需特别关注 FP64 storage 收尾后的当前版本复验、极大相位、近零合成向量、超大计数和空 periods。

### 22. 阶段三自检问题与参考答案

1. **问题：V1 绑定哪个版本，相关源码是否 dirty？**  
   **答案：**完整 SHA 为 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，提交时间 `2026-08-15T23:45:58+08:00`，分支 `final-prep/benchmark-evidence-v1`；进入阶段三时整个 `ZKX` 的 `git status --short` 无输出，相关文件 clean。
2. **问题：CPU 和 GPU 的正式返回结构有什么不同？**  
   **答案：**同 SHA 的 `spectral_analysis_typed.h:710–720` 显示 CPU 用两个 `std::vector<double>`，GPU 用两个 `DeviceArray<double>`；两者都用 `scalar_period` marker 表示标量 period。
3. **问题：CPU 核心如何实现数学公式？**  
   **答案：**同 SHA 的 `spectral_analysis_typed.cpp:459–469` 用 double 累计 $\cos(2\pi t_i/T)$ 与 $\sin(2\pi t_i/T)$，除以 $N$ 后以 `sqrt` 得 strength、以 `atan2` 得 phase。
4. **问题：标量 GPU overload 是否调用 single kernel？**  
   **答案：**否。同 SHA 的 `spectral_analysis_typed.cpp:515–523` 把标量包装为单元素 `DeviceArray` 后调用数组 overload；`.cu:527` 只 launch `vectorstrength_multi_kernel`。`git grep` 对 single kernel 只命中 `kernels.cuh:699` 的定义。
5. **问题：grid、block 和线程职责是什么？**  
   **答案：**同 SHA 的 `.cu:527` 使用 `<<<P,256>>>`；`kernels.cuh:751–765` 表明 block $p$ 负责 period $p$，线程 $q$ 处理事件 $q,q+256,\ldots$。
6. **问题：为什么需要两类 `__syncthreads()` 位置？**  
   **答案：**同 SHA 的 `kernels.cuh:767–769` 写完全部 partial sum 后同步一次；`:771–776` 的每轮树归约后再同步，防止下一轮读取尚未完成的 shared 更新。
7. **问题：补偿累加覆盖了全部归约吗？**  
   **答案：**没有。每线程事件子序列用 `CompensatedFloatAccumulator`，但 shared tree 在同 SHA 的 `kernels.cuh:773–774` 使用普通 FP32 `+=`。
8. **问题：为何 GPU 返回 double 却不能称为 FP64 计算？**  
   **答案：**科学 kernel 模板在同 SHA 的 `.cu:527` 固定 `<T,T,float,float>`；`host_output_finalize.h:40–47` 只把已完成 FP32 值转为 double storage，`fp64_storage_finalize.cu:11–73` 用整数位编码生成 IEEE double 位模式。
9. **问题：类型集合如何固定？**  
   **答案：**同 SHA 的 `spectral_analysis_typed.cpp:1393–1397` 对 `float`、`__half`、`int32_t`、`int16_t`、`int8_t` 展开显式实例化；kernel 的 `static_assert` 再拒绝非正式类型。
10. **问题：GPU 有哪些数据搬运？**  
    **答案：**数组 period 在同 SHA wrapper `:500` 发生 D2H 验证；标量 period 在 `:520` 先 H2D，再进入同一 D2H 验证；events 不回 host；strength/phase 在 device 计算并在 device widening。
11. **问题：空 events 和非法 period 如何处理？**  
    **答案：**同 SHA CPU `:451–457` 对非正 period 抛异常、空 events 写 double NaN；GPU wrapper `:500–503,517–519` 验证 period，multi kernel `:779–782` 对空 events 写两个 float NaN。
12. **问题：CPU/GPU 为什么可能有数值差异？**  
    **答案：**CPU 是顺序 double 三角/累加；GPU 是 float 三角、分线程补偿 partial 和普通 float tree reduction。运算精度与加法顺序不同，尤其在合成向量近零或相位很大时差异会放大。
13. **问题：现有 typed smoke 覆盖什么？**  
    **答案：**同 SHA 的 `e3_spectral_type_smoke.cu:317,353–357,366,370` 对五类型覆盖标量/数组、CPU/GPU 数值、marker、空 events NaN 和 period 0 异常；不覆盖极大相位、超大计数、空 periods 和当前 FP64 finalizer 的独立深度精度验收。
14. **问题：算法复杂度和主要固定成本是什么？**  
    **答案：**总科学工作 $O(PN)$、输出/临时空间 $O(P)$、shared 2048 bytes/block；小输入的主要固定成本可能是 period D2H、分配、scientific kernel launch 和两次 widening launch。

### 23. V1 结论

V1 的数学原理与 cuSignal Python 保持不变：对每个 period 计算正弦/余弦平均的合成向量长度与方向。CPU 用独立 double reference；GPU 用“一 period 一 block、256 threads、每线程跨步事件、Neumaier partial、shared tree”的 FP32 直接归约，避免 Python 版本的 `(P,N)` 复矩阵。正式 GPU 输出是 FP64 storage，而非 device FP64 科学计算。

## 版本差异与原理不变量

当前只有 V1，没有可比较的后续优化版本。后续若新增 V2，必须追加而不能覆盖本章，并比较：

| 维度 | V1 | 后续版本需核对 |
| --- | --- | --- |
| 数学原理 | 一阶圆矩的模与角 | 是否仍为同一原理；若改变须回映阶段一编号 |
| CPU | double 顺序 reference | 是否保持独立性和边界语义 |
| GPU | FP32 compensated partial + shared tree | block、归约、精度是否变化 |
| 内存 | $O(P)$，无 `(P,N)` 矩阵 | workspace/分块/融合是否改变 |
| 数据搬运 | period D2H 验证；标量 H2D；device widening | 是否消除 host 边界或新增同步 |
| 输出 | FP64 storage，FP32 科学精度 | storage 与 compute dtype 是否变化 |
| 测试 | typed smoke 静态覆盖；本阶段未重跑 | 必须绑定新 SHA 和新证据 |
