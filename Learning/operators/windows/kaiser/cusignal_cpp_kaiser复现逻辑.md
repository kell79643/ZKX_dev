# cusignal_cpp_kaiser 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前正式学习版本](#v1当前正式学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU FP64 reference；GPU FP32 稳定 Bessel 比值；结果按 FP64 存储契约拓宽 |

## V1：当前正式学习版本（fdd55ac）

### 1. 版本身份与源码状态

| 项目 | 记录 |
| --- | --- |
| 完整 SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15 23:45:58 +0800` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| 工作区状态 | `git status --short` 输出为空 |
| `kaiser` 相关文件 dirty 状态 | `windows_typed.h/.cpp/.cu`、`windows_kernels.cuh`、`e3_wave_window_type_smoke.cu` 均无未提交修改 |

本节全部源码证据都绑定上述 SHA。历史源码可用以下形式读取：

```powershell
git -C ZKX show fdd55ac8415d70379eb38a2c299047f90bcf0a41:cusignal_cpp/<相对路径>
```

本阶段只读 `ZKX/cusignal_cpp`，没有修改 C++/CUDA 源码，没有构建、运行测试或连接 ZQ500。

### 2. 代码定位与层次职责

| 层次 | 完整 SHA + 共享根目录相对路径 | 符号 | 提交内行号 | 职责 |
| --- | --- | --- | ---: | --- |
| 公共接口 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.h` | `kaiser_typed_cpu` | 158-169 | 声明 CPU reference，五种 `beta` 输入，固定 `std::vector<double>` 输出 |
| 公共接口 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.h` | `kaiser_device` | 172-183 | 声明 GPU 入口，调用者提供 `DeviceArray<double>` 输出 |
| CPU 数学 helper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.cpp` | `host_bessel_i0_fp64` | 31-47 | CPU FP64 分段多项式近似 $I_0$ |
| CPU 核心计算 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.cpp` | `kaiser_typed_cpu` | 422-442 | 长度/周期语义、逐点公式、FP64 输出 |
| GPU Host wrapper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.cpp` | `kaiser_device` | 444-458 | 检查输出、分配 FP32 中间量、调用 device 计算并拓宽存储 |
| GPU 计算桥 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.cu` | `kaiser_fp32_compute_device` | 77 | 类型门禁、参数检查、kernel 启动 |
| GPU Bessel helper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_kernels.cuh` | `scaled_bessel_i0` | 31-59 | 计算 $e^{-|x|}I_0(x)$ 的 FP32 分段近似 |
| GPU 稳定比值 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_kernels.cuh` | `stable_bessel_i0_ratio` | 61-72 | 避免先形成两个巨大 $I_0$ 再相除 |
| GPU device 核心 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_kernels.cuh` | `kaiser_kernel` | 141-164 | 每线程计算一个 Kaiser 样本 |
| 通用启动 helper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h` | `launch_1d_kernel` | 10-56 | 256-thread block、ceil grid、默认 stream、launch error check |
| FP64 存储封装 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h` | `finalize_fp64_device_storage` | 40-47 | 分配 `DeviceArray<double>` 并调用存储拓宽 |
| 存储拓宽 kernel | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu` | `widened_storage_bits` 等 | 11-49、63-73 | 用整数位操作把 FP32 值编码成等值 FP64 存储位，不执行 FP64 算术 |
| 类型实例化 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/windows/windows_typed.cpp/.cu` | `INSTANTIATE_WINDOWS_CPU`、`INSTANTIATE` | `.cpp:584-598`、`.cu:122-126` | 实例化 FP32、FP16、INT32、INT16、INT8 |
| 直接测试 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | `run_type<T>` 中的 Kaiser 检查 | 613-649 | CPU/GPU、边界、周期、高 beta、错误路径 |
| 测试类型入口 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | `main` | 788-799 | 对五种输入类型调用 `run_type<T>` |

### 3. 公共接口

#### 3.1 CPU reference 声明

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_typed.h`，符号 `kaiser_typed_cpu`，第 157-169 行：

```cpp
/**
 * @brief 在 CPU 上生成 Kaiser 窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8 beta输入类型；输出固定FP64。
 * @param n 输出长度；小于1时返回空数组。
 * @param beta 无量纲形状参数，当前实现接受任意有限 T 值。
 * @param sym true 使用 n 点；false 且 n 为偶数时用 n+1 点后截断，默认 true。
 * @return host上长度`max(n,0)`、dtype FP64的归一化实窗。
 * @throws std::bad_alloc Host输出分配失败；非正长度按cuSignal返回空数组而非异常。
 * @details CPU reference使用FP64 custom Bessel I0；无workspace/stream/科学计算库。
 * @note 对齐固定版cuSignal的float64输出、长度守卫和偶数周期窗扩展语义。
 */
template <class T>
std::vector<double> kaiser_typed_cpu(int n, T beta, bool sym = true);
```

`template <class T>` 把 `beta` 输入类型参数化；返回类型不随 `T` 变化，始终是 Host `std::vector<double>`。`sym` 默认 `true`。Doxygen 明确声明非正长度返回空数组以及“只有偶数周期窗扩展”的固定版 cuSignal 语义。

#### 3.2 GPU 公开入口声明

同一提交、同一文件，符号 `kaiser_device`，第 171-183 行：

```cpp
/**
 * @brief 在 GPU 上生成 Kaiser 窗。
 * @tparam T 五种beta输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；小于1时要求输出为空。
 * @param beta 无量纲形状参数。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 规则同 CPU reference，默认 true。
 * @throws std::invalid_argument 输出size与`max(n,0)`不匹配。
 * @details 默认stream上的custom kernel使用FP32 Bessel I0，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void kaiser_device(int n, T beta, DeviceArray<double>& out, bool sym = true);
```

调用者必须预先给出长度等于 `max(n,0)` 的 `DeviceArray<double>`。第 176、179 行描述了 Host 拓宽和 D2H/H2D，但当前提交的实际调用链使用 device 位拓宽 kernel；这是注释与实现不一致，后文按实现代码说明。

### 4. CPU reference：从输入到最后一行

#### 4.1 FP64 $I_0$ 分段近似 helper

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`，符号 `host_bessel_i0_fp64`，第 31-47 行：

```cpp
// CPU reference 使用独立的 host 数学 helper，不包含或调用 GPU kernel helper。
double host_bessel_i0_fp64(double x)
{
    const double absolute_x = std::fabs(x);
    if (absolute_x < 3.75) {
        const double ratio = x / 3.75;
        const double y = ratio * ratio;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    const double y = 3.75 / absolute_x;
    return std::exp(absolute_x) / std::sqrt(absolute_x)
        * (0.39894228 + y * (0.01328592
        + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
        + y * (-0.02057706 + y * (0.02635537
        + y * (-0.01647633 + y * 0.00392377))))))));
}
```

执行顺序如下：

1. `std::fabs(x)` 利用 $I_0(-x)=I_0(x)$，把后续分段统一到非负自变量。
2. 当 $|x|<3.75$，令 $y=(x/3.75)^2$，用 Horner 嵌套多项式近似 $I_0(x)$。
3. 当 $|x|\ge3.75$，令 $y=3.75/|x|$，用 $e^{|x|}/\sqrt{|x|}$ 乘渐近多项式。
4. helper 返回普通、未缩放的 `double` $I_0$ 近似。大到使 `exp(absolute_x)` 溢出的输入仍有风险。

CPU helper 与 GPU helper 是两个独立实现；测试用 CPU reference 对照 GPU 时，两者属于同类系数族但使用不同精度和稳定化策略。

#### 4.2 CPU 窗生成函数

同一提交、同一文件，符号 `kaiser_typed_cpu`，第 422-442 行：

```cpp
template <class T>
std::vector<double> kaiser_typed_cpu(int length, T beta, bool symmetric)
{
    if (length < 1) return {};
    const int effective = (!symmetric && length % 2 == 0) ? length + 1 : length;
    std::vector<double> out(length);
    const double beta_value = static_cast<double>(host_load(beta));
    const double denominator = host_bessel_i0_fp64(beta_value);
    for (int index = 0; index < length; ++index) {
        if (effective == 1) {
            out[index] = 1.0;
            continue;
        }
        const double alpha = 0.5 * (effective - 1);
        const double x = (index - alpha) / alpha;
        const double numerator = host_bessel_i0_fp64(beta_value
            * std::sqrt(std::max(0.0, 1.0 - x * x)));
        out[index] = numerator / denominator;
    }
    return out;
}
```

逐个语义块解释：

- 模板签名接收 `length`、类型为 `T` 的 `beta` 和 `symmetric`。
- `if (length < 1) return {};` 对 0 和负数都返回空 `std::vector<double>`。
- `effective` 完全复现 Python 固定版分支：仅当 `symmetric=false` 且 `length` 为偶数时使用 `length+1`，否则仍用原长度。
- `std::vector<double> out(length)` 在 Host 分配固定 FP64 输出。
- `host_load(beta)` 按五类型策略先把 `beta` 读入 continuous compute 值，再 `static_cast<double>`；Bessel 算术是 double，但输入已经先经过统一 load 语义。
- `denominator` 在循环外只计算一次。
- 循环只写 `index=0` 到 `length-1`。偶数周期窗无需物理构造 `length+1` 数组，只把 `effective` 用于坐标分母，与 Python 扩展后删末点数学等价。
- `effective==1` 时写 1 并 `continue`，避免 `alpha=0` 除零。
- `alpha=(effective-1)/2`，`x=(index-alpha)/alpha`，与 Python 的中心化坐标一致。
- `std::max(0.0,1.0-x*x)` 把微小负舍入误差夹到 0，避免 `sqrt` 产生 `NaN`。
- 最后计算 `numerator/denominator`，循环结束后 `return out`。

CPU 总工作量为 $O(M)$，输出空间为 $O(M)$；分母只求一次，每个元素求一次平方根和一次分子 Bessel。

### 5. GPU 调用链：Host wrapper 到 device kernel

#### 5.1 Host wrapper

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`，符号 `kaiser_device`，第 444-458 行：

```cpp
template <class T>
void kaiser_device(
    int length, T beta, DeviceArray<double>& out, bool symmetric)
{
    const std::size_t expected = length > 0
        ? static_cast<std::size_t>(length)
        : 0;
    if (out.size() != expected)
        throw std::invalid_argument("kaiser_device: output size must equal max(length, 0)");
    DeviceArray<float> computed(expected);
    if (expected != 0) {
        kaiser_fp32_compute_device(length, beta, computed, symmetric);
    }
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

执行到最后一行的职责：

1. 三元运算符把正长度转换为 `size_t`，非正长度统一为 0。
2. `out.size()` 必须精确等于 `expected`，否则抛 `std::invalid_argument`。
3. 分配同长度 `DeviceArray<float> computed`，它是实际 GPU 算术输出。
4. 空输出跳过 Kaiser kernel；非空才调用 `kaiser_fp32_compute_device`。
5. 最后一条赋值把 FP32 结果拓宽为 `DeviceArray<double>` 并移动赋给 `out`。数值精度仍来自 FP32，double 是公开存储 dtype。

#### 5.2 FP32 计算桥和 kernel 参数

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_typed.cu`，符号 `kaiser_fp32_compute_device`，第 77 行：

```cpp
template <class T> void kaiser_fp32_compute_device(int n,T b,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("kaiser_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::kaiser_kernel<float,T>,o.size(),n,b,o.data(),(!sym&&n%2==0)?n+1:n);}
```

虽然源码压成一行，语义顺序仍清楚：

1. `static_assert` 在编译期拒绝不属于正式五类型集合的 `T`。
2. 运行期要求 `n>=1` 且 FP32 中间数组长度等于 `n`。
3. `kaiser_kernel<float,T>` 表示输出计算类型为 `float`，`beta` 输入类型为 `T`。
4. `o.size()` 是逻辑元素数；业务参数依次为 `output_count=n`、`beta_input=b`、`output=o.data()`、`effective_count`。
5. `effective_count=(!sym&&n%2==0)?n+1:n` 与 CPU/Python 的偶数周期语义相同。

#### 5.3 通用一维 launch 配置

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h`，符号 `div_up`、`make_1d_launch_config`、`launch_1d_kernel_with_config`、`launch_1d_kernel`，第 10-56 行：

```cpp
template <typename IndexType>
constexpr IndexType div_up(IndexType value, IndexType divisor)
{
    return (value + divisor - 1) / divisor;
}

struct LaunchConfig1D {
    dim3 block;
    dim3 grid;
};

inline LaunchConfig1D make_1d_launch_config(
    std::size_t elements,
    unsigned int block_size = 256)
{
    return LaunchConfig1D{
        dim3(block_size),
        dim3(static_cast<unsigned int>(div_up(elements, static_cast<std::size_t>(block_size))))
    };
}

template <typename Kernel, typename... Args>
inline void launch_1d_kernel_with_config(
    Kernel kernel,
    std::size_t elements,
    cudaStream_t stream,
    unsigned int block_size,
    Args&&... args)
{
    LaunchConfig1D config = make_1d_launch_config(elements, block_size);
    kernel<<<config.grid, config.block, 0, stream>>>(std::forward<Args>(args)...);
    CUDA_KERNEL_CHECK();
}

template <typename Kernel, typename... Args>
inline void launch_1d_kernel(
    Kernel kernel,
    std::size_t elements,
    Args&&... args)
{
    launch_1d_kernel_with_config(
        kernel,
        elements,
        nullptr,
        256,
        std::forward<Args>(args)...);
}
```

Kaiser kernel 的 block 固定为 256 threads，grid 为 $\lceil M/256\rceil$ blocks，dynamic shared memory 为 0，stream 是 `nullptr`（默认 stream）。`CUDA_KERNEL_CHECK()` 检查 launch 错误，但 helper 没有显式设备同步。

#### 5.4 GPU scaled Bessel helper

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_kernels.cuh`，符号 `scaled_bessel_i0`，第 31-59 行：

```cpp
template <typename Scalar>
__device__ inline Scalar scaled_bessel_i0(Scalar value)
{
    static_assert(std::is_same_v<Scalar, float>);
    const Scalar absolute = fabsf(value);
    if (absolute < static_cast<Scalar>(3.75F)) {
        Scalar ratio = value / static_cast<Scalar>(3.75F);
        const Scalar y = ratio * ratio;
        const Scalar polynomial = static_cast<Scalar>(1) + y *
            (static_cast<Scalar>(3.5156229F) + y *
            (static_cast<Scalar>(3.0899424F) + y *
            (static_cast<Scalar>(1.2067492F) + y *
            (static_cast<Scalar>(0.2659732F) + y *
            (static_cast<Scalar>(0.0360768F) + y *
             static_cast<Scalar>(0.0045813F))))));
        return polynomial * expf(-absolute);
    }
    const Scalar y = static_cast<Scalar>(3.75F) / absolute;
    return static_cast<Scalar>(1) / sqrtf(absolute) *
        (static_cast<Scalar>(0.39894228F) + y *
        (static_cast<Scalar>(0.01328592F) + y *
        (static_cast<Scalar>(0.00225319F) + y *
        (static_cast<Scalar>(-0.00157565F) + y *
        (static_cast<Scalar>(0.00916281F) + y *
        (static_cast<Scalar>(-0.02057706F) + y *
        (static_cast<Scalar>(0.02635537F) + y *
        (static_cast<Scalar>(-0.01647633F) + y *
         static_cast<Scalar>(0.00392377F)))))))));
}
```

`__device__ inline` 表示只在 device 调用并允许内联；`static_assert` 把计算锁定为 FP32。它与 CPU helper 使用相同阈值和系数族，但返回

$$
I_{0e}(x)=e^{-|x|}I_0(x),
$$

而不是普通 $I_0(x)$：小自变量分支在多项式后乘 `expf(-absolute)`；大自变量渐近式只留下 $1/\sqrt{|x|}$ 与多项式。

#### 5.5 稳定 Bessel 比值

同一提交、同一文件，符号 `stable_bessel_i0_ratio`，第 61-72 行：

```cpp
template <typename Scalar>
__device__ inline Scalar stable_bessel_i0_ratio(
    Scalar numerator_argument,
    Scalar denominator_argument)
{
    static_assert(std::is_same_v<Scalar, float>);
    const Scalar numerator_absolute = fabsf(numerator_argument);
    const Scalar denominator_absolute = fabsf(denominator_argument);
    return expf(numerator_absolute - denominator_absolute) *
        scaled_bessel_i0(numerator_absolute) /
        scaled_bessel_i0(denominator_absolute);
}
```

由 $I_0(x)=e^{|x|}I_{0e}(x)$：

$$
\frac{I_0(a)}{I_0(b)}
=e^{|a|-|b|}\frac{I_{0e}(a)}{I_{0e}(b)}.
$$

代码正按这个恒等式计算，避免分子、分母先分别溢出为 `inf`。Kaiser 中 $|a|\le|b|$，指数项通常不大于 1。

#### 5.6 每线程 Kaiser 核心计算

同一提交、同一文件，符号 `kaiser_kernel`，第 141-164 行：

```cpp
template <typename T, typename Parameter>
__global__ void kaiser_kernel(
    int output_count,
    Parameter beta_input,
    T* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(beta_input));
    const Scalar beta = window_load(beta_input);
    const Scalar alpha = static_cast<Scalar>(0.5F) *
        static_cast<Scalar>(effective_count - 1);
    const Scalar position =
        (static_cast<Scalar>(index) - alpha) / alpha;
    const Scalar radicand = static_cast<Scalar>(1) - position * position;
    const Scalar bounded = radicand > Scalar{0} ? radicand : Scalar{0};
    output[index] = window_store<T>(stable_bessel_i0_ratio(
        beta * sqrtf(bounded), beta));
}
```

逐行职责：

1. `T=float` 是输出计算数组类型，`Parameter` 是五种 `beta` 输入之一。
2. `index=blockIdx.x*blockDim.x+threadIdx.x` 是全局一维线程索引。
3. 多余线程在 `index>=output_count` 时返回。
4. 单点窗写 1 后返回，避免除零。
5. `window_load(beta_input)` 把正式五类型输入加载到 FP32 continuous compute scalar。
6. `alpha=(effective_count-1)/2`，`position=(index-alpha)/alpha`，与 Python `temp`、CPU `x` 相同。
7. `bounded=max(1-position^2,0)` 防止负舍入误差进入平方根。
8. 最后一条跨行赋值计算 $I_0(\beta\sqrt{bounded})/I_0(\beta)$ 并写入 FP32 中间数组。

没有线程间依赖、共享内存、原子操作或显式同步；每个合法线程只负责一个 `output[index]`。

### 6. FP32 结果如何成为固定 FP64 存储

#### 6.1 分配公开输出存储

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`，符号 `finalize_fp64_device_storage`，第 40-47 行：

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

该 helper 分配等长 `DeviceArray<double>`，调用存储拓宽，再返回。它没有调用 `to_host()`，所以当前路径不是头文件注释所写的 D2H/Host/H2D。

#### 6.2 device 位拓宽

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu`，符号 `widened_storage_bits`、`widen_real_storage_kernel`、`widen_fp32_storage_device`，第 11-49、63-73 行：

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

__global__ void widen_real_storage_kernel(
    const float* input, std::uint64_t* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = widened_storage_bits(input[index]);
}

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

`widened_storage_bits` 拆出 FP32 的符号、指数和尾数，用整数位操作构造数值等价的 IEEE FP64 bit pattern：零/subnormal、`Inf/NaN`、normal 分别处理；normal 数把指数偏置从 127 调到 1023，差值为 896。`widen_real_storage_kernel` 每线程拓宽一个值；公开结果按 double 读取时与 `static_cast<double>(float_value)` 等值，但没有增加有效计算精度。

### 7. template、类型分发与显式实例化

#### 7.1 Host/device load 策略

`kaiser_typed_cpu` 的 `host_load(beta)` 和 kernel 的 `window_load(beta_input)` 最终依赖 `SimpleSignalTypePolicy<T>::load`。提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 的 `ZKX/cusignal_cpp/src/cuda_utils/simple_signal_typed.h:17-50` 把 FP32、FP16、INT32、INT16、INT8 加载为 continuous compute 类型；当前 Kaiser GPU 路径在 FP32 计算。

#### 7.2 CPU 与公开 GPU 入口实例化

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`，宏 `INSTANTIATE_WINDOWS_CPU` 的 Kaiser 项第 584-585 行、调用第 594-598 行：

```cpp
    template std::vector<double> kaiser_typed_cpu(int, T, bool); \
    template void kaiser_device(int, T, DeviceArray<double>&, bool); \
```

```cpp
INSTANTIATE_WINDOWS_CPU(float);
INSTANTIATE_WINDOWS_CPU(__half);
INSTANTIATE_WINDOWS_CPU(std::int32_t);
INSTANTIATE_WINDOWS_CPU(std::int16_t);
INSTANTIATE_WINDOWS_CPU(std::int8_t);
```

#### 7.3 GPU 计算桥实例化

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/windows/windows_typed.cu`，宏 `INSTANTIATE` 的 Kaiser 项第 122 行、调用第 126 行：

```cpp
 template void kaiser_fp32_compute_device(int,T,DeviceArray<float>&,bool); \
```

```cpp
INSTANTIATE(float); INSTANTIATE(__half); INSTANTIATE(std::int32_t); INSTANTIATE(std::int16_t); INSTANTIATE(std::int8_t);
```

五种 `beta` 输入共享同一个 FP32 输出 kernel 结构，没有 `double` 业务输入实例化。

### 8. CPU/GPU/Python 三方映射

下表所有 C++ 路径、符号和行号均绑定完整提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；Python 行号对应 cuSignal 23.08.00 只读基准。

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $M<1\to[]$ | 基准 `windows.py:1336-1337` | `windows_typed.cpp:425` | wrapper `:448-457` | 三者都得到长度 0；GPU要求调用者预分配空 `out` |
| $M=1\to[1]$ | `windows.py:1338-1339` | `windows_typed.cpp:431-434` | `windows_kernels.cuh:150-153` | CPU/GPU在一般函数内部处理，Python提前返回 |
| 偶数 `sym=false` 有效长度 $M+1$ | `windows.py:1340-1347` | `windows_typed.cpp:426` | `.cu:77` | C++不物理生成额外点，只修改 `effective_count`，数学等价 |
| $u=(i-\alpha)/\alpha$ | `windows.py:1219,1225` | `windows_typed.cpp:435-436` | `windows_kernels.cuh:156-159` | 公式一致 |
| $\sqrt{1-u^2}$ | `windows.py:1220` | `windows_typed.cpp:437-438` | `windows_kernels.cuh:160-163` | C++显式夹紧到非负，Python未显式夹紧 |
| $I_0(a)/I_0(\beta)$ | `windows.py:1220-1221` | FP64 普通比值 | FP32 scaled 稳定比值 | 数学相同，精度和溢出策略不同 |
| 输出 dtype | Python kernel 固定 FP64 | `vector<double>` | FP32 计算后按 FP64 存储位拓宽 | GPU公开 dtype 是 FP64，有效数值精度是 FP32 |
| 奇数 `sym=false` | 不扩展 | `effective=length` | `effective_count=n` | 三者共同保留固定版 cuSignal 的奇数周期差异 |

### 9. 内存、同步与错误处理

#### 9.1 内存布局和搬运

- CPU：连续 `std::vector<double>`，每元素 8 bytes。
- GPU `beta`：值参数传入 kernel，无参数数组。
- GPU 中间：连续 `DeviceArray<float>`，约 $4M$ bytes。
- GPU 公开输出：连续 `DeviceArray<double>`，约 $8M$ bytes。
- finalize 不做 D2H/H2D；第二个 device kernel直接编码 FP64 存储。
- 测试中的 `to_host()` 才把公开输出复制到 Host。

#### 9.2 stream 与同步

- Kaiser 计算和存储拓宽都使用默认 stream。
- 同一 stream 保证第二个 kernel 读取第一个 kernel 的结果。
- wrapper 没有显式 `cudaDeviceSynchronize`。
- `CUDA_KERNEL_CHECK()` 检查启动错误；同步拷贝或上层同步才会形成完成边界。

#### 9.3 异常与边界

| 条件 | CPU reference | GPU wrapper/bridge |
| --- | --- | --- |
| `length<1` | 返回空 vector | 要求 `out.size()==0`，跳过 Kaiser kernel |
| 输出长度错误 | 内部分配，无此输入 | wrapper 抛 `std::invalid_argument` |
| bridge 非正 `n` 或错误 FP32 size | 不适用 | bridge 抛 `std::invalid_argument` |
| 不支持的 `T` | 无实例/编译失败 | `static_assert` 和无显式实例阻止 |
| 大 `beta` | 普通 FP64 $I_0$ 仍可能溢出 | scaled ratio避免常见 `inf/inf` |
| 根号轻微负数 | `std::max(0.0,...)` | 三元表达式夹到 0 |

### 10. 直接测试如何验证

#### 10.1 Kaiser typed smoke 块

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`，`run_type<T>` 的 Kaiser 块，第 613-649 行：

```cpp
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::kaiser);
    kaiser_device(5, convert<T>(2), fixed_fp64_window_out);
    const auto kaiser_actual = fixed_fp64_window_out.to_host();
    const auto kaiser_reference = kaiser_typed_cpu(5, convert<T>(2));
    kaiser_device<T>(1, convert<T>(2), singleton_fp64_window_out);
    const bool kaiser_singleton_is_one =
        singleton_fp64_window_out.to_host() == std::vector<double>{1.0};
    kaiser_device<T>(0, convert<T>(2), empty_fp64_window_out);
    kaiser_device<T>(-1, convert<T>(2), empty_fp64_window_out);
    bool kaiser_rejects_output_size = false;
    try {
        DeviceArray<double> invalid_kaiser_out(4);
        kaiser_device<T>(5, convert<T>(2), invalid_kaiser_out);
    } catch (const std::invalid_argument& error) {
        kaiser_rejects_output_size = std::string(error.what()).find("kaiser") !=
            std::string::npos;
    }
    kaiser_device<T>(5, convert<T>(2), fixed_fp64_window_out);
    DeviceArray<double> kaiser_periodic_out(4);
    kaiser_device<T>(4, convert<T>(2), kaiser_periodic_out, false);
    kaiser_device(5, convert<T>(100), fixed_fp64_window_out);
    const auto kaiser_high_beta = fixed_fp64_window_out.to_host();
    const bool kaiser_high_beta_is_finite_and_matches =
        all_finite(kaiser_high_beta) && matches_fp64(
            kaiser_high_beta,
            kaiser_typed_cpu(5, convert<T>(100)));
    ok[10] = matches_fp64(kaiser_actual, kaiser_reference) &&
        matches_fp64(
            kaiser_periodic_out.to_host(),
            kaiser_typed_cpu<T>(4, convert<T>(2), false)) &&
        kaiser_singleton_is_one &&
        kaiser_typed_cpu<T>(1, convert<T>(2)) == std::vector<double>{1.0} &&
        empty_fp64_window_out.empty() &&
        kaiser_typed_cpu<T>(0, convert<T>(2)).empty() &&
        kaiser_typed_cpu<T>(-1, convert<T>(2)).empty() &&
        kaiser_rejects_output_size &&
        kaiser_high_beta_is_finite_and_matches;
```

它依次覆盖 $M=5,\beta=2$ 的 CPU/GPU 对照、单点、非正长度、错误输出 size、$M=4$ 偶数周期、$\beta=100$ 有限性，并把全部条件合并到 `ok[10]`。

#### 10.2 五类型入口

同一提交、同一测试文件，符号 `main`，第 788-799 行：

```cpp
int main()
{
    int passed = 0;
    passed += run_type<float>("fp32");
    passed += run_type<__half>("fp16");
    passed += run_type<std::int32_t>("int32");
    passed += run_type<std::int16_t>("int16");
    passed += run_type<std::int8_t>("int8");
    std::cout << "[E3][wave_window] total=70 pass=" << passed
              << " fail=" << (70 - passed) << '\n';
    return passed == 70 ? 0 : 1;
}
```

这里记录的是测试源码设计；本阶段没有实际运行，不能声称本次会话得到 PASS。

#### 10.3 尚未覆盖的风险

- CPU reference 与 GPU 使用同类近似系数族，可能有共同公式错误；没有直接外部 golden vector。
- 没有直接测试奇数 $M$ 的 `sym=false`。
- 没有直接覆盖负 `beta`、$\beta=0$、偶数 `sym=true` 峰值缺失。
- 高 beta 只测 100；CPU helper 对更大值仍可能 `exp` 溢出。
- 未在本阶段执行 ZQ500 硬件验证。

### 11. 精度、复杂度与 ZQ500 边界

| 项目 | CPU | GPU |
| --- | --- | --- |
| 输入类型 | 五种 `T`，按 continuous policy load | 五种 `T`，kernel 中 load 到 FP32 |
| Bessel 计算 | FP64 普通 $I_0$ 分段近似 | FP32 scaled $I_0$ 与稳定比值 |
| 输出类型 | `double` | FP32 计算后编码为 `DeviceArray<double>` |
| 总工作量 | $O(M)$ | 两个 $O(M)$ kernel：Kaiser + storage widen |
| 辅助空间 | 输出 $8M$ bytes | FP32 中间约 $4M$ + FP64 输出约 $8M$ bytes |
| 并行 | Host 循环 | 每线程一个元素，block 256 |
| 同步 | 不适用 | 默认 stream 顺序，无 wrapper 显式同步 |

ZQ500 契约不允许正式 kernel 执行 FP64 算术，所以 GPU 数学使用 FP32；FP64 仅作为接口/存储 dtype。当前结论来自源码，没有通过本阶段硬件运行重新确认。

### 12. 与 cuSignal Python 的相同、等价替换和有意不同

| 类别 | 内容 |
| --- | --- |
| 相同 | 非正长度为空、单点为 1、只在偶数周期窗使用 $M+1$、Kaiser 核心公式、固定 FP64 公开输出 |
| 等价替换 | Python 物理生成 $M+1$ 后切片；C++ 只保留 $M$ 个输出但使用 $M+1$ effective count |
| 有意不同 | CPU 用 FP64 custom Bessel；GPU 为 ZQ500 使用 FP32 stable scaled Bessel，再拓宽为 FP64 storage |
| 数值增强 | C++ CPU/GPU 都把 $1-u^2$ 夹到非负；GPU stable ratio避免高 beta 的 `inf/inf` |
| 工程扩展 | 五种 `beta` 输入、调用者管理 device 输出、size 验证、显式实例化和 typed smoke |
| 保留差异 | 奇数 `sym=false` 仍不扩展，继续对齐固定版 cuSignal |

### 13. 阶段三自检问题与参考答案

1. 当前 V1 的完整 SHA、分支和 dirty 状态是什么？

   **答案：**SHA 是 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`；进入阶段三时 `git status --short` 为空，相关实现/测试文件无未提交修改。
2. CPU 与 GPU 的核心数学公式是否改变？

   **答案：**没有。两者都计算 $I_0(\beta\sqrt{1-u^2})/I_0(\beta)$；CPU 在 `windows_typed.cpp:435-439`，GPU 在 `windows_kernels.cuh:156-163`。差别是 Bessel 精度和稳定求值方式。
3. CPU 的 `effective` 如何表达周期窗？

   **答案：**`windows_typed.cpp:426` 仅在 `!symmetric && length%2==0` 时令 `effective=length+1`。循环仍只写 `length` 个元素，等价于 Python 扩展后去末点。
4. GPU 每线程负责什么？grid/block 如何确定？

   **答案：**每个合法 `index` 写一个 `output[index]`；`kernel_launch.h:21-28,45-55` 使用 block 256、grid $\lceil M/256\rceil$，越界线程由 kernel 第 149 行返回。
5. GPU 高 `beta` 为什么更稳定？

   **答案：**`stable_bessel_i0_ratio` 使用 $e^{|a|-|b|}I_{0e}(a)/I_{0e}(b)$，避免分子、分母先分别溢出。
6. 为什么 GPU 返回 FP64，却不能说进行了 FP64 数学计算？

   **答案：**Kaiser kernel 输出 `DeviceArray<float>`；`fp64_storage_finalize.cu:11-49` 用整数位操作构造等值 double bit pattern。存储是 FP64，有效精度来自 FP32。
7. 实际数据路径是否符合头文件的 D2H/Host/H2D 描述？

   **答案：**不符合。`finalize_fp64_device_storage` 直接调用 device `widen_fp32_storage_device`，没有 `to_host()`；因此解释以实现为准，Doxygen 是滞后注释。
8. 五种输入类型如何获得链接实体？

   **答案：**`.cpp:584-598` 为 CPU/reference 和公开 GPU wrapper 显式实例化；`.cu:122,126` 为计算桥实例化 float、`__half`、INT32、INT16、INT8。
9. wrapper 有哪些错误处理和同步行为？

   **答案：**wrapper 检查 `out.size()==max(length,0)`，不匹配抛 `invalid_argument`；launch helper执行 `CUDA_KERNEL_CHECK()`。wrapper 没有显式同步，两个 kernel 依赖默认 stream 顺序。
10. 测试覆盖哪些关键边界，尚缺什么？

    **答案：**`e3_wave_window_type_smoke.cu:613-649` 覆盖常规对照、单点、非正长度、错误 size、偶数周期和 beta 100；`main:791-795` 对五类型运行。仍缺奇数周期、负 beta、beta 0、外部 golden 和本阶段实际 ZQ500 执行。

## 版本差异与原理不变量

当前只有 V1，尚无 V2/V3 可比较。

| 比较项 | V1（fdd55ac） | 原理不变量 |
| --- | --- | --- |
| CPU | FP64 普通 Bessel 分段近似 | Kaiser 比值公式不变 |
| GPU | FP32 scaled Bessel 稳定比值 + FP64 storage widen | 中心坐标、对称性和 $\beta$ 含义不变 |
| 周期语义 | 仅偶数 `sym=false` 使用 $M+1$ effective count | 对齐固定版 cuSignal 的离散分支 |
| 输出 | CPU/GPU公开固定 FP64 | 输出长度与归一化目标不变 |
| 测试 | 五类型 typed smoke，源码存在但本阶段未运行 | 正确性围绕公式、长度、周期和有限性 |
