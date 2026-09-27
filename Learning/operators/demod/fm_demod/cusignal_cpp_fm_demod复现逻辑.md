# `cusignal_cpp_fm_demod` 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前实现](#v1当前实现fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前 | typed CPU reference、任意 rank/axis GPU workspace 与单 kernel 实现。 |

## V1：当前实现（`fdd55ac`）

### 1. 版本身份与可追溯性

| 项目 | 值 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| 相关文件 dirty 状态 | clean；下列头文件、CPU、GPU、kernel 与 smoke test 均无未提交修改 |
| 本次运行状态 | 未构建、未运行；本阶段按规则只读源码，不能把历史记录冒充当前会话验证 |

本版本的每条定位都同时绑定上述完整 SHA。历史实现只能通过 `git -C ZKX show <SHA>:<path>` 读取；本文没有复制或修改 `ZKX/cusignal_cpp`。

### 2. 文件、符号与职责

以下路径均相对共享根目录 `ZKX_dev/`，行号属于提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`。

| 层次 | 路径、符号与行号 | 职责 |
| --- | --- | --- |
| 公共接口与类型 | `ZKX/cusignal_cpp/src/demod/demod_typed.h:20-176`；`DemodComplex`、`DemodTypePolicy`、`FmDemodOptions`、`FmDemodWorkspace`、`fm_demod_typed_cpu`、`fm_demod_device` | 定义交错复数、五类型加载、shape/axis workspace、CPU/GPU 正式入口及便利重载。 |
| CPU helper | `ZKX/cusignal_cpp/src/demod/demod_typed.cpp:10-92`；`checked_nonzero_product`、`validated_workspace`、`phase`、`unwrap_local_delta`、`demod_sample` | 验证容量与 workspace，计算 FP32 相位和 CuPy 兼容的局部 unwrap 差分。 |
| workspace | `ZKX/cusignal_cpp/src/demod/demod_typed.cpp:96-158`；`FmDemodWorkspace::reset` | 规范化 axis，计算输入/输出 shape、`inner_size`、`outer_size` 与元素数。 |
| CPU 核心 | `ZKX/cusignal_cpp/src/demod/demod_typed.cpp:160-215`；`fm_demod_typed_cpu` | 逐输出索引找到相邻输入样本并计算局部相位差；提供 1D/2D 重载及五类型显式实例化。 |
| GPU host 调度 | `ZKX/cusignal_cpp/src/demod/demod_typed.cu:12-141`；`validated_workspace`、`launch_fm_demod`、`fm_demod_device` | 检查 shape、输出和别名，启动 kernel；提供 `ComplexFloat` ABI、1D/2D 重载与实例化。 |
| GPU device 核心 | `ZKX/cusignal_cpp/src/demod/demod_kernels.cuh:10-52`；`phase`、`unwrap_local_delta`、`fm_demod_nd_kernel` | 每个线程计算一个输出元素。 |
| 直接测试 | `ZKX/cusignal_cpp/test/signal_processing/e3_demod_type_smoke.cu:18-249`；`run_device_case`、`run_type`、`main` | 覆盖五种分量类型、1D/2D/rank-3、正负 axis、精确 ±π、空轴、单元素轴、异常和 `ComplexFloat` ABI。 |
| 构建入口 | `ZKX/cusignal_cpp/CMakeLists.txt:178-180`；`e3_demod_type_smoke` | 创建单算子 smoke 可执行目标并链接 `signal_lib`、`fft_core`。 |

旧文档中出现的 `demod.cpp`、`demod_cuda.cu` 是历史布局线索；当前正式 typed 实现以上述文件为准。

### 3. 公共数据结构与类型策略

#### 3.1 交错复数

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.h:20-24`，符号 `DemodComplex<T>`：

```cpp
template <typename T>
struct DemodComplex {
    T re;
    T im;
};
```

`template <typename T>` 让实部、虚部共享同一种分量类型。结构体采用交错布局：一个复数对象内依次保存 `re` 和 `im`，不是两条独立数组。

#### 3.2 五类型加载与固定 FP32 计算

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.h:28-44`，符号 `DemodTypePolicy<T>`：

```cpp
template <typename T>
struct DemodTypePolicy {
    static constexpr bool supported =
        std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int8_t>;

    __host__ __device__ static float load(T value) { return static_cast<float>(value); }
};

template <>
struct DemodTypePolicy<__half> {
    static constexpr bool supported = true;
    __host__ __device__ static float load(__half value) { return __half2float(value); }
};

template <typename T>
inline constexpr bool is_demod_input_v = DemodTypePolicy<T>::supported;
```

普通模板批准 `float/int32/int16/int8`，统一用 `static_cast<float>` 进入 FP32。`__half` 需要模板特化并调用 `__half2float`。`__host__ __device__` 使同一加载策略能被 CPU 与 device 代码调用。输出没有回量化，固定为 `float`。

#### 3.3 shape、axis 与 workspace

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.h:53-80`，符号 `FmDemodOptions`、`FmDemodWorkspace`：

```cpp
struct FmDemodOptions {
    std::vector<int> shape;
    int axis = -1;
};

struct FmDemodWorkspace {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int axis_length = 0;
    int output_axis_length = 0;
    int inner_size = 0;
    int outer_size = 0;
    std::size_t input_count = 0;
    std::size_t output_count = 0;

    FmDemodWorkspace() = default;
    explicit FmDemodWorkspace(const FmDemodOptions& options) { reset(options); }

    void reset(const FmDemodOptions& options);
    [[nodiscard]] std::size_t input_size() const noexcept { return input_count; }
    [[nodiscard]] std::size_t output_size() const noexcept { return output_count; }
};
```

workspace 只保存 Host 端布局元数据，不拥有 device scratch。`inner_size` 是目标轴之后各维乘积，`outer_size` 是目标轴之前各维乘积。输出保持 rank，只把目标轴长度改为 `max(axis_length-1, 0)`。

### 4. workspace 构造与验证

#### 4.1 容量乘积检查

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cpp:13-29`，符号 `checked_nonzero_product`。

函数从 `shape[begin:end]` 累乘维度。乘法前用 `product > INT_MAX / dimension` 检查溢出；维度为零时跳过除法风险并使最终乘积变零。当前 kernel 使用 `int` 线性索引，因此超过 `INT_MAX` 会显式抛出 `invalid_argument`。

#### 4.2 防止伪造或修改 workspace

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cpp:31-66` 与 `ZKX/cusignal_cpp/src/demod/demod_typed.cu:12-47`，符号 `same_workspace`、`validated_workspace`、`one_dimensional_workspace`。

`validated_workspace` 从公开的 `input_shape` 和 `axis` 重新构造一份权威 workspace，再逐字段比较。调用者若修改了 `output_count`、步长或 shape 派生字段，比较失败并抛出异常。1D 便利入口把元素数包装成 shape `{count}`，也先检查能否放入 32 位索引。

#### 4.3 `reset` 的完整流程

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cpp:96-158`，符号 `FmDemodWorkspace::reset`：

1. 拒绝空 shape，即不接受标量 rank-0 输入。
2. 拒绝 rank 超出 `int`，并拒绝任何负维度。
3. 接受 axis 范围 `[-rank, rank-1]`；负 axis 用 `axis + rank` 规范化。
4. 扫描 shape 是否含零维。
5. 复制输入 shape，生成输出 shape；目标轴长度为 `axis_length > 0 ? axis_length - 1 : 0`。
6. 任一维为零时，把输入/输出计数和布局步长都置零并提前返回。
7. 非空布局分别计算输入总数、目标轴之后的 `inner_size`、之前的 `outer_size`。
8. 目标轴输出长度为零时输出计数为零，否则计算完整输出 shape 乘积。

以 shape `{2, 3, 4}`、axis `1` 为例：

```text
axis_length = 3
output_axis_length = 2
inner_size = 4
outer_size = 2
output_shape = {2, 2, 4}
```

### 5. CPU 调用链与核心算法

#### 5.1 相位与局部 unwrap

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cpp:68-92`，符号 `phase`、`unwrap_local_delta`、`demod_sample`：

```cpp
template <typename T>
float phase(const DemodComplex<T>& value)
{
    const float real = detail::DemodTypePolicy<T>::load(value.re);
    const float imag = detail::DemodTypePolicy<T>::load(value.im);
    return std::atan2(imag, real);
}

float unwrap_local_delta(float previous_phase, float current_phase)
{
    const float delta = current_phase - previous_phase;
    if (std::fabs(delta) < kPi) return delta;

    float corrected = std::fmod(delta + kPi, kTwoPi);
    if (corrected < 0.0F) corrected += kTwoPi;
    corrected -= kPi;
    if (corrected == -kPi && delta > 0.0F) corrected = kPi;
    return corrected;
}

template <typename T>
float demod_sample(const DemodComplex<T>& previous, const DemodComplex<T>& current)
{
    return unwrap_local_delta(phase(previous), phase(current));
}
```

执行顺序是分量转 FP32 → 两点分别 `atan2` → 当前相位减前一相位 → 映射到默认 $2\pi$ 区间。精确正向 `+π` 被保留为 `+π`；精确负向 `-π` 保留为 `-π`。它没有用 `conj(previous)*current`，避免极大分量相乘溢出改变 CuPy 的逐点 `angle` 语义。

#### 5.2 任意 rank CPU reference

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cpp:160-183`，符号 `fm_demod_typed_cpu`：

```cpp
template <typename T>
std::vector<float> fm_demod_typed_cpu(
    const std::vector<DemodComplex<T>>& x,
    const FmDemodWorkspace& workspace)
{
    static_assert(detail::is_demod_input_v<T>, "unsupported fm_demod business input dtype");
    const FmDemodWorkspace layout = validated_workspace(workspace);
    if (x.size() != layout.input_size()) {
        throw std::invalid_argument("fm_demod_typed_cpu input shape mismatch");
    }
    std::vector<float> output(layout.output_size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        const int linear = static_cast<int>(index);
        const int inner_index = linear % layout.inner_size;
        const int axis_index = (linear / layout.inner_size) % layout.output_axis_length;
        const int outer_index = linear / (layout.inner_size * layout.output_axis_length);
        const int previous =
            (outer_index * layout.axis_length + axis_index) * layout.inner_size + inner_index;
        const int current = previous + layout.inner_size;
        output[index] = demod_sample(x[static_cast<std::size_t>(previous)],
                                    x[static_cast<std::size_t>(current)]);
    }
    return output;
}
```

每个线性输出索引被拆成 `(outer_index, axis_index, inner_index)`。`previous` 指向目标轴位置 `axis_index`，`current = previous + inner_size` 指向同一 outer/inner 坐标下目标轴的下一点。循环彼此无状态，每个输出独立。

#### 5.3 CPU 便利重载与实例化

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cpp:185-215`，符号 `fm_demod_typed_cpu` 与 `INSTANTIATE_DEMOD_CPU`：

- 1D 重载自动建立 `{x.size()}` workspace。
- 2D 重载把 `rows/cols/axis` 包成 options，再委托正式任意 rank 入口。
- `INSTANTIATE_DEMOD_CPU` 显式实例化 `float/__half/int32/int16/int8` 的三种重载，确保模板定义在 `.cpp` 中仍产生可链接符号。

CPU 调用链为：

```text
1D/2D convenience overload
  → FmDemodWorkspace::reset
  → fm_demod_typed_cpu(x, workspace)
      → validated_workspace
      → output index decomposition
      → demod_sample
          → phase(previous), phase(current)
          → unwrap_local_delta
```

### 6. GPU host 调度

#### 6.1 正式启动 helper

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cu:49-77`，符号 `launch_fm_demod`：

```cpp
template <typename Complex>
void launch_fm_demod(
    const DeviceArray<Complex>& x,
    DeviceArray<float>& y,
    const FmDemodWorkspace& workspace)
{
    const FmDemodWorkspace layout = validated_workspace(workspace);
    if (x.size() != layout.input_size()) {
        throw std::invalid_argument("fm_demod_device input shape mismatch");
    }
    if (y.size() != layout.output_size()) {
        throw std::invalid_argument("fm_demod_device output size mismatch");
    }
    if (!y.empty() && reinterpret_cast<const void*>(x.data()) ==
                          reinterpret_cast<const void*>(y.data())) {
        throw std::invalid_argument("fm_demod_device does not permit input/output aliasing");
    }
    if (y.empty()) return;

    cuda_utils::launch_1d_kernel(
        demod_detail::fm_demod_nd_kernel<Complex>,
        y.size(),
        x.data(),
        y.data(),
        static_cast<int>(layout.output_size()),
        layout.axis_length,
        layout.output_axis_length,
        layout.inner_size);
}
```

Host 端先重验 workspace、输入大小、输出大小及输入输出是否同址。空输出不启动 kernel。非空时调用统一的 `launch_1d_kernel`；这里没有 H2D/D2H、同步、FFT、随机库或 device scratch 分配。

#### 6.2 GPU 对外入口

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_typed.cu:81-141`，符号 `fm_demod_device` 与 `INSTANTIATE_DEMOD_GPU`：

- `DeviceArray<DemodComplex<T>>` 正式入口用 `static_assert` 限制五种业务类型，再委托 `launch_fm_demod`。
- `DeviceArray<ComplexFloat>` 非模板入口共享同一个 kernel，可直接承接 FFT/Hilbert 的复 FP32 ABI。
- 1D 重载自动建立 workspace；2D 重载构造 `{rows, cols}` options。
- `INSTANTIATE_DEMOD_GPU` 为五种类型生成正式、1D、2D 三类可链接实例。

调用者必须预先分配 `y`，并在读取结果前自行同步。API 不隐藏传输与同步成本。

### 7. GPU kernel 与线程职责

#### 7.1 device 相位和 unwrap

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_kernels.cuh:13-31`，符号 `phase` 与 `unwrap_local_delta`：

device `phase` 使用同一 `DemodTypePolicy` 转 FP32，再调用 `atan2f`。`unwrap_local_delta` 用 `fabsf/fmodf` 实现与 CPU 相同的分支和精确 ±π 规则。

#### 7.2 kernel

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/demod/demod_kernels.cuh:33-52`，符号 `fm_demod_nd_kernel`：

```cpp
template <class Complex>
__global__ void fm_demod_nd_kernel(
    const Complex* input,
    float* output,
    int output_count,
    int axis_length,
    int output_axis_length,
    int inner_size)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;

    const int inner_index = index % inner_size;
    const int axis_index = (index / inner_size) % output_axis_length;
    const int outer_index = index / (inner_size * output_axis_length);
    const int previous =
        (outer_index * axis_length + axis_index) * inner_size + inner_index;
    const int current = previous + inner_size;
    output[index] = unwrap_local_delta(phase(input[previous]), phase(input[current]));
}
```

线程线性编号为 `blockIdx.x * blockDim.x + threadIdx.x`。越过 `output_count` 的尾部线程立即退出。每个有效线程只写 `output[index]`，读取两个沿目标轴相邻的输入元素；线程之间没有共享状态、原子操作或同步。grid/block 的具体 block 大小由 `cuda_utils::launch_1d_kernel` 统一决定，本算子只提供工作量 `y.size()`。

### 8. 数学—Python—CPU—GPU 映射

| 原理或公式 | cuSignal Python | C++ CPU | C++ GPU | 差异说明 |
| --- | --- | --- | --- | --- |
| $\theta_w=\operatorname{Arg}(x)$ | `cp.angle(x)`，基准 `demod.py:36` | `phase`，`demod_typed.cpp:68-74` | `phase`，`demod_kernels.cuh:13-19` | C++ 五类型先转 FP32；GPU 用 `atan2f`。 |
| 展开后局部差分 | `cp.unwrap(...); cp.diff(...)`，基准 `:36-37` | `unwrap_local_delta`，`:76-86` | `unwrap_local_delta`，kernel 头 `:21-31` | C++ 利用“unwrap 后立即 diff”只需局部修正，无需保存累计展开相位；输出数学等价。 |
| $y[n]=\theta_u[n+1]-\theta_u[n]$ | `cp.diff` | `demod_sample` 与索引循环 `:88-182` | 每线程一项，kernel `:42-51` | CPU 串行循环；GPU 输出元素并行。 |
| 目标轴缩短一项 | CuPy `diff` | workspace `output_axis_length`，`:125-127` | 使用同一 Host workspace | 完全对应，长度 0/1 均输出空轴。 |
| 输出 dtype | 由复 FP32 Python 路径得到 FP32 | 固定 `std::vector<float>` | 固定 `DeviceArray<float>` | FP16/整数复分量是 C++ 批准扩展，仍输出 FP32。 |
| 物理 Hz 标定 | 未实现 | 未实现 | 未实现 | 三者都只返回 rad/sample 型相位增量。 |

C++ 没有采用显式共轭乘积。它属于阶段一原理 1 的工程等价实现：利用 `unwrap` 后马上 `diff` 的局部性融合步骤，数学原理不变。

### 9. 内存、并行、同步与复杂度

| 项目 | CPU | GPU |
| --- | --- | --- |
| 输入布局 | Host `std::vector<DemodComplex<T>>`，row-major | device `DeviceArray<DemodComplex<T>>` 或 `DeviceArray<ComplexFloat>` |
| 输出 | Host `std::vector<float>` | 调用者预分配 `DeviceArray<float>` |
| 临时数据 | Host workspace；输出数组 | Host workspace；无 device scratch |
| 数据搬运 | CPU 路径无 device 搬运 | 正式 API 无隐式 H2D/D2H |
| 并行 | 一个普通循环 | 每个输出一个线程 |
| 同步 | 不涉及 device | kernel 启动后不隐式同步；调用者读取前负责同步 |
| 时间复杂度 | $O(M_{out})$ | 总工作 $O(M_{out})$，理想并行深度近似由 grid 调度决定 |
| 额外空间 | 输出 $O(M_{out})$，workspace $O(rank)$ | 输出由调用者提供；Host workspace $O(rank)$，device 额外 scratch $O(1)$ |

每个输出包含两次 FP32 `atan2` 和若干标量运算，计算成本高于单纯相减。GPU 输入相邻线程通常写连续输出；输入访问是否连续取决于 `inner_size`，但相同 `inner_index` 的目标轴相邻点相距 `inner_size`。

### 10. 精度、边界与平台限制

- 五种复分量均转 FP32 后计算；没有 FP64 device 路径。
- `float` 极大分量分别求 `atan2`，避免共轭乘积溢出。
- 精确 `+π/-π` 的边界方向按 CuPy 规则保留。
- `(0,0)` 的 `atan2(0,0)` 依赖 C/CUDA 数学函数行为；直接测试期望零相位差。
- shape 必须非空、各维非负，axis 必须合法。
- 任一零维令总输入与输出计数为零；目标轴长度 1 也产生零输出。
- 输入、输出及布局乘积不得超过 `INT_MAX`，这是当前 32 位线性索引限制。
- GPU 禁止输入输出别名，且要求输出容量精确匹配 workspace。
- 无噪声滤波、载频去除或 $f_s/(2\pi)$ 标定，和 Python 算子边界一致。
- 本文没有把通用 NVIDIA 环境结论写成 ZQ500 实测结论；本次未连接或运行 ZQ500。

### 11. 测试如何验证及未覆盖风险

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 的 `ZKX/cusignal_cpp/test/signal_processing/e3_demod_type_smoke.cu:18-249`，符号 `run_device_case`、`run_type`、`main`，设计了以下检查：

| 测试段行号 | 检查内容 |
| --- | --- |
| `61-70` | 1D 相位每步增加 $π/4$，CPU/GPU 都应输出四个 $π/4$。 |
| `72-87` | 2×3 输入分别沿 axis 1 和 axis 0，检查 shape/索引语义。 |
| `89-100` | rank-3、axis `-2` 规范化为 1，输出 shape `{2,1,3}`。 |
| `102-110` | 精确 `+π` 与 `-π` unwrap 边界。 |
| `112-127` | 零相位与 FP32 极大分量，后者验证不使用易溢出的复乘积。 |
| `129-148` | 零长度维和目标轴单元素，CPU/GPU 都返回空输出。 |
| `150-203` | 标量、非法 axis、负 shape、容量溢出、CPU shape、GPU 输出大小、篡改 workspace 的异常。 |
| `205-214` | `ComplexFloat` GPU ABI。 |
| `233-248` | 依次运行 FP32、FP16、INT32、INT16、INT8，五类全通过才返回 0。 |

本次没有运行该目标，因此只可称为“测试覆盖设计”，不能声称当前提交在本会话通过。仍有风险：

1. 未在本次 ZQ500 环境实际验证编译、kernel 启动和数值误差。
2. 没有噪声、NaN/Inf、超大 rank 或接近 `INT_MAX` 的实际内存用例。
3. 没有性能基准；本文不发布性能提升结论。
4. 没有用物理单位 Hz 的端到端 FM 信号验证，因为 API 本身不含采样率和灵敏度。

### 12. 阅读顺序

建议依次阅读：

1. `demod_typed.h`：接口、类型与 workspace 契约。
2. `demod_typed.cpp:96-158`：shape/axis 如何变成线性布局。
3. `demod_typed.cpp:68-92`：FP32 angle 与局部 unwrap。
4. `demod_typed.cpp:160-183`：CPU reference。
5. `demod_typed.cu:49-77`：GPU Host 检查和启动。
6. `demod_kernels.cuh:33-52`：线程索引与 device 计算。
7. `e3_demod_type_smoke.cu:56-248`：测试怎样覆盖正常、边界与异常。

### 13. 自检问题与参考答案

1. **问：V1 的可追溯身份是什么？**  
   **答：**完整 SHA 为 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，提交时间 `2026-08-15T23:45:58+08:00`，分支 `final-prep/benchmark-evidence-v1`；相关文件 clean。

2. **问：CPU reference 与 GPU wrapper/kernel 的职责如何区分？**  
   **答：**`demod_typed.cpp:160-183` 在 Host 循环中计算结果；`demod_typed.cu:49-77` 只验证并启动；`demod_kernels.cuh:33-52` 才是 device 核心计算。

3. **问：C++ 为什么无需保存完整 `unwrap` 相位数组？**  
   **答：**Python 在 `unwrap` 后立即 `diff`，累计的 $2π$ 修正经相邻相减只留下当前局部修正。CPU/GPU 的 `unwrap_local_delta` 直接计算该局部结果，数学上等价。

4. **问：五种输入类型如何统一？**  
   **答：**`DemodTypePolicy` 将 `float/__half/int32/int16/int8` 分量加载为 FP32；CPU 返回 `vector<float>`，GPU 输出 `DeviceArray<float>`，不回量化。

5. **问：每个 GPU 线程负责什么？**  
   **答：**kernel `demod_kernels.cuh:42-51` 的每个有效线程负责一个 `output[index]`，读取同一 outer/inner 坐标下沿目标轴相邻的 `previous/current` 两点。

6. **问：何处发生同步和数据搬运？**  
   **答：**正式 `fm_demod_device` 内两者都不发生；调用者持有 device 输入/输出，并在读取结果前同步。测试的 `from_host/to_host` 才显式触发搬运。

7. **问：精确 ±π 如何处理？**  
   **答：**`unwrap_local_delta` 先映射到 `[-π,π)`，若映射为 `-π` 且原 delta 为正，则改回 `+π`，所以正向 π 保持正、负向 π 保持负。

8. **问：当前主要平台容量限制是什么？**  
   **答：**kernel 使用 32 位线性索引；workspace 在 `demod_typed.cpp:13-29,137-156` 保证输入、输出和布局乘积不超过 `INT_MAX`。

9. **问：测试覆盖了哪些 shape 边界？**  
   **答：**测试 `:89-148` 覆盖 rank-3/负 axis、零长度维和目标轴长度 1；`:150-203` 覆盖标量、非法 axis、负 shape、容量及 workspace/size 不一致。

10. **问：本次能否宣称 ZQ500 测试通过？**  
    **答：**不能。本阶段未构建或运行；只能记录当前提交中的测试设计与历史文档线索，不能把它们冒充本次验证。

## 版本差异与原理不变量

当前只有 V1，尚无可追加的 V2。后续优化必须在新的已提交 SHA 下追加版本，不能覆盖 V1。

| 比较项 | V1 `fdd55ac` | 原理不变量 |
| --- | --- | --- |
| 算法结构 | 两点分别取 FP32 angle，再做 CuPy 兼容局部 unwrap 差分 | 始终估计相邻展开相位差。 |
| CPU | 任意 rank/axis 串行 reference | $y=\Delta\operatorname{unwrap}(\operatorname{Arg}x)$。 |
| GPU | 每输出一线程、单 kernel、无 device scratch | 与 CPU/Python 输出定义一致。 |
| dtype | 五种复分量进入 FP32，输出 FP32 | 输出是连续实数相位增量，不回量化整数。 |
| 物理标定 | 不含 $f_s/f_c/k_f$ | 只输出 rad/sample，调用方负责 Hz 与消息标度。 |
