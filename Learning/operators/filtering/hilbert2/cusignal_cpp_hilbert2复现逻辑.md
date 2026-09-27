+# cusignal_cpp_hilbert2复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前学习版本](#v1当前学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前 | 固定 cuSignal 23.08.00 方形 mask 语义的 CPU/GPU typed 实现 |

## V1：当前学习版本（fdd55ac）

### 1. 版本身份与工作区状态

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交标题：`test(archive): 增加任务结果范围治理审计`
- 相关文件 dirty 状态：对 `cusignal_cpp/src/filtering/`、相关头文件与 `e3_filtering_type_smoke.cu` 执行聚焦 `git status --short`，无输出；本版本可追溯。
- 本阶段只读 `ZKX/cusignal_cpp`，没有修改 C++ 源码，没有构建或运行测试，也没有连接 ZQ500。

下文每处 C++ 证据均绑定上述完整 SHA。路径均相对共享根目录 `ZKX_dev/`。

### 2. 文件职责与调用链

| 层次 | 路径与符号 | 职责 |
| --- | --- | --- |
| API/数据契约 | `ZKX/cusignal_cpp/src/filtering/filtering_typed.h`：`Hilbert2Options`、结果结构、`hilbert2_typed_cpu`、`hilbert2_device` | 定义 shape、dtype variant 和公开 typed 接口 |
| 公共布局计划 | `.../filtering_typed.cpp:565-610`：`prepare_hilbert2_layout` | 复现固定 Python 版本的 shape、`N`、方形 mask 与广播限制 |
| CPU/reference | `.../filtering_typed.cpp:1906-1961`：`hilbert2_typed_cpu` | Host 上独立执行二维 FFT、mask、IFFT 和 dtype 封装 |
| GPU API wrapper | `.../filtering_typed.cpp:1963-1980`：`hilbert2_device` | 分配结果、调用 GPU compute、封装 FP32/FP64 variant |
| GPU host 调度 | `.../filtering_typed.cu:353-384`：`hilbert2_fp32_compute_device` | 输入打包、二维 FFT、kernel 启动、二维 IFFT |
| GPU device 核心 | `.../filtering_kernels.cuh:1542-1578` | 计算一维权重、方阵原地 mask、单行广播到方阵并 mask |
| FFT 后端 | `.../fft_interface/fft_interface_cpu.cpp:240-295`；`.../fft_interface/fft_interface.cpp:739-769` | CPU 可分离二维变换；GPU `FFTInterface2D` 后端执行与错误检查 |
| 测试 | `.../test/signal_processing/e3_filtering_type_smoke.cu:146-157,919-1020` | CPU/GPU 比较、默认/标量/广播/拒绝/冲激/dtype/shape 覆盖 |

```text
hilbert2_typed_cpu
  → prepare_hilbert2_layout
  → FFTInterface2D_cpu::fft2d
  → hilbert_mask_weight × 两轴
  → FFTInterface2D_cpu::ifft2d
  → Hilbert2CpuResult

hilbert2_device
  → prepare_hilbert2_layout
  → hilbert2_fp32_compute_device
      → real_matrix_to_complex_float_kernel
      → FFTInterface2D::fft2d_device
      → hilbert2_mask_kernel
         或 hilbert2_broadcast_row_mask_kernel
      → FFTInterface2D::ifft2d_device
  → Hilbert2DeviceResult
  → 整数输入时 finalize_complex_fp64_device_storage
```

### 3. 对外数据结构与接口

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/filtering/filtering_typed.h`，符号 `Hilbert2Options`、`Hilbert2CpuResult`、`Hilbert2DeviceResult`，行 82–99：

```cpp
struct Hilbert2Options {
    std::vector<int> shape;
    std::vector<int> fft_shape;
};

struct Hilbert2CpuResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};

struct Hilbert2DeviceResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};
```

`shape={rows,cols}` 是扁平输入的逻辑二维 shape；`fft_shape` 为空、单元素或双元素，分别对应 Python 的 `N=None`、标量 `N`、二元 `N`。CPU 结果使用 Host `std::vector`，GPU 结果使用 device `DeviceArray`；`dtype` 指出哪个 variant 有效。

同 SHA、同文件，符号 `detail::Hilbert2LayoutPlan`，行 330–342：

```cpp
struct Hilbert2LayoutPlan {
    int input_rows = 0;
    int input_cols = 0;
    int fft_rows = 0;
    int fft_cols = 0;
    int output_rows = 0;
    int output_cols = 0;
    std::size_t fft_size = 0;
    std::size_t output_size = 0;
};

Hilbert2LayoutPlan prepare_hilbert2_layout(
    std::size_t input_size, const Hilbert2Options& options);
```

该结构把输入 shape、第一次 FFT shape、固定 Python 方形 mask 产生的最终 shape 以及两个扁平容量集中冻结，CPU 与 GPU 共用同一计划。

同 SHA、同文件，符号 `hilbert2_device`，行 625–628；符号 `hilbert2_typed_cpu`，行 879–881：

```cpp
template <typename T>
void hilbert2_device(
    const DeviceArray<T>& x, Hilbert2DeviceResult& out,
    const Hilbert2Options& options);

template <typename T>
Hilbert2CpuResult hilbert2_typed_cpu(
    const std::vector<T>& x, const Hilbert2Options& options);
```

支持显式实例化的 `T` 为 `float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t`。浮点输入输出 `ComplexFloat`；整数输入虽然计算仍经 ComplexFP32，却按固定 cuSignal dtype 边界拓宽为 `ComplexDouble` 存储。

### 4. 公共布局与错误检查

同 SHA，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp`，符号 `detail::prepare_hilbert2_layout`，行 565–610：

```cpp
Hilbert2LayoutPlan prepare_hilbert2_layout(
    std::size_t input_size, const Hilbert2Options& options)
{
    if (options.shape.size() != 2)
        throw std::invalid_argument("hilbert2 input must be 2-D");
    Hilbert2LayoutPlan plan;
    plan.input_rows = options.shape[0];
    plan.input_cols = options.shape[1];
    if (plan.input_rows < 2 || plan.input_cols < 0)
        throw std::invalid_argument("hilbert2 fixed cuSignal path requires two rows");
    const long long input_product = static_cast<long long>(plan.input_rows)
        * plan.input_cols;
    if (input_product > std::numeric_limits<int>::max() ||
        input_product != static_cast<long long>(input_size))
        throw std::invalid_argument("hilbert2 input shape product");

    if (options.fft_shape.empty()) {
        plan.fft_rows = plan.input_rows;
        plan.fft_cols = plan.input_cols;
    } else if (options.fft_shape.size() == 1) {
        plan.fft_rows = options.fft_shape[0];
        plan.fft_cols = options.fft_shape[0];
    } else if (options.fft_shape.size() == 2) {
        plan.fft_rows = options.fft_shape[0];
        plan.fft_cols = options.fft_shape[1];
    } else {
        throw std::invalid_argument("hilbert2 N must be scalar or pair");
    }
    if (plan.fft_rows <= 0 || plan.fft_cols <= 0)
        throw std::invalid_argument("hilbert2 N must contain positive values");
    if (plan.fft_rows != plan.fft_cols && plan.fft_rows != 1)
        throw std::invalid_argument("hilbert2 fixed square mask cannot broadcast");
    plan.output_rows = plan.fft_rows == 1
        ? plan.fft_cols : plan.fft_rows;
    plan.output_cols = plan.fft_cols;
    const long long fft_size = static_cast<long long>(plan.fft_rows)
        * plan.fft_cols;
    const long long output_size = static_cast<long long>(plan.output_rows)
        * plan.output_cols;
    if (fft_size > std::numeric_limits<int>::max() ||
        output_size > std::numeric_limits<int>::max())
        throw std::invalid_argument("hilbert2 FFT shape exceeds int indexing");
    plan.fft_size = static_cast<std::size_t>(fft_size);
    plan.output_size = static_cast<std::size_t>(output_size);
    return plan;
}
```

这里有意复现 Python 固定版本的实际行为，而不是修成理想矩形语义：Python 两个 mask 都按 `N[1]` 分配，外积必为 `N1×N1`。因此只有 `N0==N1` 可直接相乘，或 `N0==1` 时首行频谱可广播为 `N1×N1`；其他矩形在 Python 广播处失败，C++ 提前抛出明确异常。

`input_rows<2` 对应 Python `x[1].dtype` 的越界边界。`input_cols<0` 被拒绝，但零列会通过本层，后续 FFT 是否接受由 FFT 接口决定。

### 5. CPU/reference 实现

同 SHA，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp`，符号 `hilbert_output_dtype`、`hilbert_mask_weight`，行 1789–1803：

```cpp
namespace {
template <typename T>
HilbertOutputDtype hilbert_output_dtype()
{
    return std::is_integral_v<T>
        ? HilbertOutputDtype::complex_fp64
        : HilbertOutputDtype::complex_fp32;
}

double hilbert_mask_weight(int index, int length)
{
    if (index == 0 || ((length % 2) == 0 && index == length / 2))
        return 1.0;
    return index < (length + 1) / 2 ? 2.0 : 0.0;
}
```

权重规则与 Python kernel 一致：DC 与偶数 Nyquist 为 1，严格正频为 2，负频为 0。二维权重是两个一维权重乘积。

同 SHA、同文件，符号 `hilbert2_typed_cpu`，行 1905–1961：

```cpp
template <typename T>
Hilbert2CpuResult hilbert2_typed_cpu(
    const std::vector<T>& x, const Hilbert2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert2_layout(x.size(), options);
    std::vector<std::vector<complexd>> time(
        plan.fft_rows, std::vector<complexd>(plan.fft_cols));
    for (int row = 0; row < plan.fft_rows; ++row) {
        for (int col = 0; col < plan.fft_cols; ++col) {
            if (row < plan.input_rows && col < plan.input_cols) {
                time[row][col] = complexd{
                    detrend_host_load(x[static_cast<std::size_t>(row)
                        * plan.input_cols + col]),
                    0.0};
            }
        }
    }
    FFTInterface2D_cpu forward_fft(plan.fft_rows, plan.fft_cols);
    const auto first_spectrum = forward_fft.fft2d(time);
    std::vector<std::vector<complexd>> masked(
        plan.output_rows, std::vector<complexd>(plan.output_cols));
    for (int u = 0; u < plan.output_rows; ++u) {
        for (int v = 0; v < plan.output_cols; ++v) {
            const int source_u = plan.fft_rows == 1 ? 0 : u;
            const auto value = first_spectrum[source_u][v];
            const double scale = hilbert_mask_weight(u, plan.fft_cols)
                * hilbert_mask_weight(v, plan.fft_cols);
            masked[u][v] = value * scale;
        }
    }

    FFTInterface2D_cpu inverse_fft(plan.output_rows, plan.output_cols);
    const auto computed = inverse_fft.ifft2d(masked);

    Hilbert2CpuResult result;
    result.dtype = hilbert_output_dtype<T>();
    result.shape = {plan.output_rows, plan.output_cols};
    if (result.dtype == HilbertOutputDtype::complex_fp32)
        result.complex_fp32.resize(plan.output_size);
    else
        result.complex_fp64.resize(plan.output_size);
    for (int row = 0; row < plan.output_rows; ++row) {
        for (int col = 0; col < plan.output_cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row)
                * plan.output_cols + col;
            const double real = computed[row][col].real();
            const double imag = computed[row][col].imag();
            if (result.dtype == HilbertOutputDtype::complex_fp32)
                result.complex_fp32[index] = ComplexFloat{
                    static_cast<float>(real), static_cast<float>(imag)};
            else
                result.complex_fp64[index] = ComplexDouble{real, imag};
        }
    }
    return result;
}
```

CPU 数据为 row-major 扁平 `x[row*input_cols+col]`。`time` 默认零初始化，所以循环同时完成截断和补零。`FFTInterface2D_cpu` 先逐行再逐列执行一维变换，是二维 DFT 的可分离实现。`source_u=0` 实现 `N0=1` 时把唯一频谱行广播到方阵每一行。

CPU 核心计算使用 `complexd`（double complex）。最后仅按输出契约把浮点输入结果缩为 ComplexFP32，或把整数输入结果存入 ComplexFP64。CPU 是独立 reference，不调用 GPU kernel。

### 6. GPU 实现

同 SHA，`ZKX/cusignal_cpp/src/filtering/filtering_kernels.cuh`，符号 `real_matrix_to_complex_float_kernel`，行 82–104：

```cpp
template <typename T>
__global__ void real_matrix_to_complex_float_kernel(
    const T* input,
    int input_rows,
    int input_cols,
    ComplexFloat* output,
    int output_rows,
    int output_cols)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = output_rows * output_cols;
    if (linear_index >= total) {
        return;
    }

    const int row = linear_index / output_cols;
    const int column = linear_index % output_cols;
    output[linear_index].re = (row < input_rows && column < input_cols)
        ? static_cast<float>(filtering_load(
            input[static_cast<std::size_t>(row) * input_cols + column]))
        : 0.0F;
    output[linear_index].im = 0.0F;
}
```

每线程处理一个输出元素。线性索引反算 row/column；输入范围内读取并转 FP32，范围外写零，从而完成 FFT 截断/补零。

同 SHA、同文件，符号 `hilbert2_mask_weight`、`hilbert2_mask_kernel`、`hilbert2_broadcast_row_mask_kernel`，行 1542–1578：

```cpp
__device__ inline float hilbert2_mask_weight(int index, int length)
{
    if (index == 0 || ((length % 2) == 0 && index == length / 2))
        return 1.0F;
    return index < (length + 1) / 2 ? 2.0F : 0.0F;
}

static __global__ void hilbert2_mask_kernel(
    ComplexFloat* spectrum, int length)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = length * length;
    if (index >= total) return;
    const int row = index / length;
    const int col = index % length;
    const float scale = hilbert2_mask_weight(row, length)
        * hilbert2_mask_weight(col, length);
    spectrum[index].re *= scale;
    spectrum[index].im *= scale;
}

static __global__ void hilbert2_broadcast_row_mask_kernel(
    const ComplexFloat* row_spectrum,
    int length,
    ComplexFloat* square_spectrum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = length * length;
    if (index >= total) return;
    const int row = index / length;
    const int col = index % length;
    const float scale = hilbert2_mask_weight(row, length)
        * hilbert2_mask_weight(col, length);
    const ComplexFloat value = row_spectrum[col];
    square_spectrum[index] = ComplexFloat{
        value.re * scale, value.im * scale};
}
```

方阵 kernel 每线程负责一个 `(row,col)`，原地缩放一个复频谱元素。广播 kernel 以 `row_spectrum[col]` 把唯一频谱行复制到每个目标 row，再乘方形 mask。两者都没有共享内存、原子操作或块内同步。若 block 大小为 $B$，grid 约为 $\lceil L^2/B\rceil$；每线程处理一个元素，尾部线程越界返回。

同 SHA，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cu`，符号 `hilbert2_fp32_compute_device`，行 352–384：

```cpp
template <class T>
void hilbert2_fp32_compute_device(
    const DeviceArray<T>& x, DeviceArray<ComplexFloat>& computed,
    const Hilbert2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert2_layout(x.size(), options);
    if (computed.size() != plan.output_size)
        throw std::invalid_argument("hilbert2 compute output shape");
    DeviceArray<ComplexFloat> time(plan.fft_size);
    DeviceArray<ComplexFloat> first_spectrum(plan.fft_size);
    cuda_utils::launch_1d_kernel(
        detail::real_matrix_to_complex_float_kernel<T>, time.size(),
        x.data(), plan.input_rows, plan.input_cols, time.data(),
        plan.fft_rows, plan.fft_cols);
    FFTInterface2D first_fft(
        plan.fft_rows, plan.fft_cols, FFTInterface::BatchOnly{});
    first_fft.fft2d_device(time, first_spectrum);
    if (plan.fft_rows == plan.fft_cols) {
        cuda_utils::launch_1d_kernel(
            detail::hilbert2_mask_kernel, first_spectrum.size(),
            first_spectrum.data(), plan.fft_cols);
        first_fft.ifft2d_device(first_spectrum, computed);
        return;
    }
    DeviceArray<ComplexFloat> square_spectrum(plan.output_size);
    cuda_utils::launch_1d_kernel(
        detail::hilbert2_broadcast_row_mask_kernel, square_spectrum.size(),
        first_spectrum.data(), plan.fft_cols, square_spectrum.data());
    FFTInterface2D square_fft(
        plan.output_rows, plan.output_cols, FFTInterface::BatchOnly{});
    square_fft.ifft2d_device(square_spectrum, computed);
}
```

方阵路径复用 `first_spectrum` 原地 mask，并复用 `first_fft` 做 IFFT；`N0=1` 路径另分配方形频谱、广播后建立方阵 IFFT 计划。布局函数已拒绝其他矩形。调用顺序依赖默认 stream；这里没有算子层显式同步。

同 SHA，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp`，符号 `hilbert2_device`，行 1963–1980：

```cpp
template <typename T>
void hilbert2_device(
    const DeviceArray<T>& x, Hilbert2DeviceResult& output,
    const Hilbert2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert2_layout(x.size(), options);
    DeviceArray<ComplexFloat> computed(plan.output_size);
    hilbert2_fp32_compute_device(x, computed, options);
    output = Hilbert2DeviceResult{};
    output.dtype = hilbert_output_dtype<T>();
    output.shape = {plan.output_rows, plan.output_cols};
    if (output.dtype == HilbertOutputDtype::complex_fp32)
        output.complex_fp32 = std::move(computed);
    else
        output.complex_fp64 =
            cuda_utils::finalize_complex_fp64_device_storage(computed);
}
```

浮点输入用 `std::move` 转交 device buffer。整数输入把 ComplexFP32 结果拓宽为 ComplexFP64 存储；这不会恢复 FP32 FFT 已经损失的精度。

### 7. 显式实例化与类型分发

SHA 同上，`filtering_typed.cpp:2781-2783` 的 `INST(T)` 显式实例化 CPU 与公开 GPU wrapper；`filtering_typed.cu:683-685` 显式实例化 GPU compute。两处覆盖：

```cpp
INST(float);
INST(__half);
INST(std::int32_t);
INST(std::int16_t);
INST(std::int8_t);
```

模板可留在实现文件中，而链接器仍获得五种具体符号。`static_assert(detail::is_simple_signal_input_v<T>)` 是编译期类型门禁。

### 8. 数学—Python—CPU—GPU 映射

| 原理或公式 | cuSignal Python | C++ CPU | C++ GPU | 差异说明 |
| --- | --- | --- | --- | --- |
| $X=\operatorname{FFT2}(x,N)$ | `filtering.py:970` | `filtering_typed.cpp:1923-1924` | `filtering_typed.cu:367-369` | CPU 用 double complex，GPU 用 ComplexFP32 |
| $h_N[k]\in\{0,1,2\}$ | 基准 895–929 | `hilbert_mask_weight` 1798–1803 | `hilbert2_mask_weight` 1542–1547 | DC、Nyquist、正负频规则等价 |
| $H[u,v]=h_{N_1}[u]h_{N_1}[v]$ | 两个长度都使用 `N[1]` | 1931–1933 | 1557–1560 | C++有意复现固定方形 mask |
| `N0=1` 广播 | CuPy 广播 | `source_u=0`，1929 | broadcast kernel，1575 | 等价复现 |
| 其他矩形不兼容 | `Xf*h` 形状失败 | 595–596 提前抛错 | 共用布局检查 | 异常时机不同，均拒绝 |
| $x_a=\operatorname{IFFT2}(HX)$ | 基准 982 | 1937–1938 | 374 或 381–383 | 输出遵循固定方形 shape |

数学原理仍是阶段一“原理 2”的单象限二维解析信号；工程目标则保留固定 cuSignal 23.08.00 的 `N[1]×N[1]` 方形 mask 缺陷边界。

### 9. 内存、同步、复杂度与平台边界

- CPU workspace 为 `time`、`first_spectrum`、`masked`、`computed`，总体 $O(M)$。
- GPU 方阵路径主要有 `time`、`first_spectrum`、`computed`；广播路径增加 `square_spectrum`，整数输出增加 ComplexFP64 buffer。
- mask kernel 为 $O(L^2)$，共享内存为 0；FFT/IFFT 通常主导总时间 $O(M\log M)$。
- 输入、频谱与输出均为 row-major 扁平存储。
- 算子主体没有显式同步；默认 stream 顺序维持 kernel/FFT 依赖，错误检查由 wrapper 负责。
- ZQ500 构建必须在 `gpu_02` 中初始化 SDK。本阶段没有运行环境，不能声称本提交在当前硬件被本轮重新验证。
- ComplexFP64 整数输出仍源自 ComplexFP32 计算，是存储拓宽而非 FP64 FFT。

### 10. 测试证据与未覆盖风险

同 SHA，`ZKX/cusignal_cpp/test/signal_processing/e3_filtering_type_smoke.cu`：

- `matches_hilbert2`（146–157）比较 dtype、shape，再比较有效 variant。
- 919–929：默认 `2×2`，GPU 对 CPU。
- 930–936：标量 `N=3`，期望 `3×3`。
- 937–944：`N={1,3}` 广播，期望 `3×3`。
- 945–961：CPU/GPU 都拒绝 `{2,3}`。
- 962–982：CPU/GPU 都拒绝一行输入。
- 983–1003：单位冲激与已知 ComplexFP32/FP64 结果比较。
- 1004–1020：五类型 dtype、shape、拒绝行为、已知值和三条 CPU/GPU 一致性。

本轮未执行测试，以上仅是测试源码覆盖。仍需关注零列、所有奇偶/Nyquist 组合、大尺寸容量、不同 FFT 后端一致性，以及“ComplexFP64 只是 FP32 拓宽”的精度误解。

### 11. 自检问题与参考答案

#### 问题 1：本版本身份是什么，为什么可形成 V1？

答案：完整 SHA 为 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`，相关文件无 dirty 输出，因此行号和内容可由 Git 重现。

#### 问题 2：CPU、GPU wrapper、GPU host 调度和 device kernel 怎样区分？

答案：CPU reference 是 `filtering_typed.cpp:1906-1961`；公开 GPU wrapper 是 `:1963-1980`；GPU host 调度是 `filtering_typed.cu:353-384`；device 核心是 `filtering_kernels.cuh:1542-1578`。

#### 问题 3：为什么 mask 两轴都用 `fft_cols`？

答案：固定 Python 源码把两个 mask 都分配为 `N[1]`，所以外积为 `N1×N1`。C++ 为兼容实际源码，在 CPU 1931–1932 和 GPU kernel 中都使用 `fft_cols`。

#### 问题 4：`N={1,L}` 如何工作？

答案：第一次 FFT 得到 `1×L`。CPU 用 `source_u=0` 为每个目标行复用唯一频谱行；GPU 用 `row_spectrum[col]` 广播到 `L×L`，然后做方形 IFFT。

#### 问题 5：每个 GPU mask 线程负责什么？

答案：线程把线性索引转换为 `row=index/length`、`col=index%length`，计算 $h[row]h[col]$，缩放一个复频谱元素的实部和虚部。

#### 问题 6：整数输入是否执行 FP64 FFT？

答案：否。GPU 始终用 ComplexFP32 FFT和 mask，最后 `finalize_complex_fp64_device_storage` 只把存储拓宽为 ComplexFP64。

#### 问题 7：同步在哪里发生？

答案：算子主体没有显式同步。kernel 与 FFT 在默认 stream 上按顺序提交；具体后端检查和同步边界由 `launch_1d_kernel`、`FFTInterface2D` 及上层结果读取决定。

#### 问题 8：测试覆盖和空白是什么？

答案：源码覆盖默认、标量 N、单行广播、矩形拒绝、短行拒绝、冲激已知值、五 dtype 与 CPU/GPU 对比；零列、完整奇偶/Nyquist 组合、大尺寸和多后端交叉仍不足。

## 版本差异与原理不变量

当前只有 V1。后续优化必须追加 V2/V3，不得覆盖本节。应保持或明确重新定义的数学不变量包括：二维 FFT、每轴 `{0,1,2}` 解析掩膜、外积权重、二维 IFFT，以及兼容目标是否继续保留 `N[1]×N[1]` 方形 mask 边界。
