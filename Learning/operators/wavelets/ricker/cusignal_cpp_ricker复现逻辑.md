# cusignal_cpp_ricker 复现逻辑

## 版本索引

| 版本 | Git 快照 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：原始学习版本](#v1原始学习版本fdd55ac8415d) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU FP64 reference；GPU FP32 计算后在设备上写 FP64 存储位 |

## V1：原始学习版本（fdd55ac8415d）

### 1. 版本身份与阅读边界

| 项目 | 记录 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15 23:45:58 +08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交说明 | `test(archive): 增加任务结果范围治理审计` |
| 仓库整体状态 | clean；`git status --short` 无输出 |
| `ricker` 相关文件状态 | 全部 clean，没有未提交修改 |
| 验证方式 | 仅本地只读 `git show <SHA>:<path>` 与源码核对；未构建、未运行、未连接 ZQ500 |

本阶段只读取以下与 `ricker` 直接有关的范围：

- `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`：公开 CPU/GPU 声明；
- `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`：CPU reference、公开 GPU wrapper、显式实例化；
- `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`：GPU 内部 host wrapper；
- `ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`：device 公式与 kernel；
- `ZKX/cusignal_cpp/src/cuda_utils/` 中被上述代码直接调用的类型策略、launch 和 FP64 存储拓宽 helper；
- `ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`：直接 typed smoke；
- `ZKX/cusignal_cpp/CMakeLists.txt`：该 smoke 的构建入口。

没有通读其他算子、整个测试树或全部 CMake。所有行号都指向上述完整 SHA 中的文件，不是
当前工作区会漂移的裸行号。

### 2. 总体架构与职责边界

```text
公开接口 wavelets_typed.h
    ├─ ricker_typed_cpu<T>(n, a) ──> CPU FP64 公式循环 ──> std::vector<double>
    └─ ricker_device<T>(n, a, out)
         ├─ 校验 n 与正式 FP64 输出 size
         ├─ 分配 DeviceArray<float> computed
         ├─ ricker_fp32_compute_device<T>
         │    ├─ 校验 T、width 与临时输出 size
         │    └─ launch_1d_kernel(ricker_kernel<float,T>)
         │          └─ 每线程：load width → ricker_value → 写一个 float
         └─ finalize_fp64_device_storage(computed)
              └─ 第二个 device kernel：FP32 IEEE 位模式 → FP64 IEEE 存储位
```

必须明确区分：

| 层次 | 符号 | 职责 |
| --- | --- | --- |
| API 声明 | `ricker_typed_cpu`、`ricker_device` | 规定模板参数、输入和返回/输出类型 |
| CPU/reference 核心 | `ricker_typed_cpu` | 用 `double` 逐点计算，用作 GPU 对照 |
| GPU 公开 host wrapper | `ricker_device` | 校验正式输出、分配 FP32 临时数组、调用计算与存储拓宽 |
| GPU 内部 host wrapper | `ricker_fp32_compute_device` | 校验宽度与临时数组，选择 kernel 模板并启动 |
| GPU device 公式 | `ricker_value` | 实现 Ricker 数学公式 |
| GPU kernel | `ricker_kernel` | 计算线程索引、越界保护、类型 load/store |
| FP64 存储 finalize | `widened_storage_bits`、`widen_real_storage_kernel` | 不做 FP64 浮点算术，按 IEEE 位规则写出 double 存储 |
| 测试 | `run_type<T>` 中的 ricker 块 | 五种输入类型下比较 GPU 与 CPU，并检查两类异常 |

### 3. 公开接口逐段解释

#### 3.1 CPU reference 声明

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，
`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`，符号 `ricker_typed_cpu`，第 73–83 行。

```cpp
/**
 * @brief 在 CPU 上生成 Ricker（Mexican hat）小波 reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 宽度参数；计算和固定输出为 FP32。
 * @param n 输出点数，必须至少为 1。
 * @param a 正宽度/尺度参数，dtype T。
 * @return host上`[n]`、dtype FP64的单个实数组。
 * @throws std::invalid_argument `n<1` 或 `a<=0`。
 * @details 与 cuSignal Ricker 点采样语义对应；无 workspace、stream 或科学计算库。
 */
template<class T>
std::vector<double> ricker_typed_cpu(int n, T a);
```

逐行解释：

1. `/**` 开始 Doxygen 文档块，`*/` 结束；这些行描述接口，不参与运行。
2. `@brief` 明确 CPU 函数既是可调用实现，也是 GPU 正确性 reference。
3. `@tparam T` 列出五种正式输入类型，但其中“计算和固定输出为 FP32”与本 SHA 的实现不符：
   返回类型是 `std::vector<double>`，函数体也用 `double` 计算。正式解释必须以实现为准。
4. `@param n` 和 `@param a` 给出有效域 $n\ge1$、$a>0$。
5. `@return` 与声明一致：host 上一维长度 $n$ 的 FP64 向量。
6. `@throws` 对应实现中的显式 `std::invalid_argument`。
7. `@details` 说明 CPU 路径不使用 GPU stream 或科学计算库；它仍会分配输出向量。
8. `template<class T>` 声明函数模板，`T` 只控制宽度参数的输入类型。
9. 最后一行声明返回 `std::vector<double>`，参数按值传入；分号结束声明。

#### 3.2 GPU 公开声明

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`，符号 `ricker_device`，
第 85–96 行。

```cpp
/**
 * @brief 在 GPU 上生成 Ricker 小波。
 * @tparam T FP32、FP16、INT32、INT16或INT8宽度参数；固定输出FP64。
 * @param n 输出点数，必须至少为 1。
 * @param a 正宽度/尺度参数。
 * @param out device输出`[n]`，由Host拓宽后作为FP64存储写回。
 * @throws std::invalid_argument n/a 非法或输出 size 不匹配。
 * @details 默认stream上device只进行FP32计算，随后D2H、Host转FP64并按需H2D；无设备
 * FP64算术或额外workspace，保持cuSignal Ricker语义。
 */
template<class T>
void ricker_device(int n, T a, DeviceArray<double>& out);
```

逐行解释：

1. 文档块声明这是 GPU 路径，正式输出接口固定为 `DeviceArray<double>`。
2. `T` 仍支持五种输入类型，输出不随 `T` 改变。
3. `n`、`a` 和调用者提供的 `out` 组成 API；`out` 是非常量引用，函数会替换其内容。
4. `@throws` 描述了两层校验的合并契约：公开 wrapper 检查 `n/out.size()`，内部 wrapper
   检查 `n/width/computed.size()`。
5. `@param out` 和 `@details` 声称 D2H→Host 拓宽→H2D，但本 SHA 的实际 finalize 走第二个
   device kernel，没有 D2H/H2D。这里是过期文档，不应覆盖代码事实。
6. “无额外 workspace”只能理解为“不要求调用者传 workspace”；实现确实分配了长度 $n$
   的 `DeviceArray<float> computed` 内部临时缓冲区。
7. `void` 表示结果通过 `out` 写回；`DeviceArray<double>&` 避免复制 wrapper 对象并允许重新赋值。

### 4. 输入类型策略

#### 4.1 CPU 使用的 `load` 包装

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，符号 `load`，第 11–15 行。

```cpp
template <typename T>
float load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}
```

逐行解释：

1. 这是匿名 namespace 中的函数模板，只在当前翻译单元可见。
2. 返回类型固定为 `float`；即使 CPU 核心随后转换到 `double`，宽度首先经过 FP32 表示。
3. 花括号形成函数体。
4. `SimpleSignalTypePolicy<T>::load` 统一处理 FP32、FP16 和整数输入。
5. 结束函数体。

#### 4.2 五种输入为何统一进入 FP32 连续计算域

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/operator_compute_policy.h`，
符号 `InputTraits` 与 `continuous_compute_t`，第 29–79 行。

```cpp
template <typename T>
struct InputTraits {
    static constexpr bool supported =
        std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int8_t>;
    static constexpr bool integral = std::is_integral_v<T>;
    static constexpr bool fp16 = false;

    using ContinuousComputeT = float;
    using ContinuousAccumulatorT = float;
    using ComparisonT = std::conditional_t<integral, std::int32_t, float>;
    using ModularUnsignedT = typename ModularUnsigned<T>::type;

    __host__ __device__ static float load_continuous(T value)
    {
        return static_cast<float>(value);
    }

    __host__ __device__ static ComparisonT load_comparison(T value)
    {
        return static_cast<ComparisonT>(value);
    }
};

template <>
struct InputTraits<__half> {
    static constexpr bool supported = true;
    static constexpr bool integral = false;
    static constexpr bool fp16 = true;

    using ContinuousComputeT = float;
    using ContinuousAccumulatorT = float;
    using ComparisonT = float;
    using ModularUnsignedT = std::uint32_t;

    __host__ __device__ static float load_continuous(__half value)
    {
        return __half2float(value);
    }

    __host__ __device__ static float load_comparison(__half value)
    {
        return __half2float(value);
    }
};

template <typename T>
inline constexpr bool supported_input_v = InputTraits<T>::supported;

template <typename T>
using continuous_compute_t = typename InputTraits<T>::ContinuousComputeT;
```

逐段解释：

- 通用 `InputTraits<T>` 只把 `float`、`int32_t`、`int16_t`、`int8_t` 标为支持；
  `supported` 是编译期布尔常量。
- `integral` 和 `fp16` 描述类型类别；普通模板不是 FP16。
- `ContinuousComputeT` 与累加类型都固定为 `float`。`ComparisonT` 对整数使用 `int32_t`，
  但 Ricker 走的是 continuous 路径。
- `load_continuous` 可在 host/device 调用，用 `static_cast<float>` 把普通输入载入 FP32。
- `load_comparison` 是同一策略的比较路径，Ricker 不直接调用它，但它属于该完整 traits 定义。
- `InputTraits<__half>` 是显式特化；`__half2float` 把 FP16 转为 FP32，其余连续类型也固定为
  `float`。
- `supported_input_v` 暴露支持性；`continuous_compute_t` 取出连续计算类型。后续
  `wavelet_load` 的返回类型因此对五种输入都是 `float`。

### 5. CPU/reference 完整执行逻辑

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，
符号 `ricker_typed_cpu`，第 113–129 行。

```cpp
template <class T>
std::vector<double> ricker_typed_cpu(int n, T width)
{
    const double a = static_cast<double>(load(width));
    if (n < 1 || a <= 0)
        throw std::invalid_argument("ricker: n and width must both be positive");
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) {
        const double x = i - .5 * (n - 1);
        const double q = x * x / (a * a);
        const double value = 2 / (std::sqrt(3 * a) *
            std::pow(3.141592653589793238462643383279502884, .25)) *
            (1 - q) * std::exp(-.5 * q);
        out[i] = value;
    }
    return out;
}
```

逐行解释：

1. `template <class T>` 与声明匹配，为五种宽度输入生成实例。
2. 返回值是 host `std::vector<double>`；`width` 按值传入。
3. `{` 开始函数体。
4. `load(width)` 先得到 FP32，再 `static_cast<double>` 进入后续 FP64 公式；因此 CPU 算术是
   FP64，但输入宽度已经受 `T` 和 FP32 load 的量化影响。
5. `if` 用短路或同时检查 $n<1$ 和 $a\le0$；下一行无花括号，只受控一条 `throw`。
6. `throw std::invalid_argument(...)` 终止非法调用并带上 `ricker` 名称。
7. `std::vector<double> out(n)` 分配并值初始化 $n$ 个 double。
8. `for` 令 `i` 从 0 递增到 $n-1$；每轮生成一个样本。
9. `x=i-0.5(n-1)` 把数组坐标中心移到 0；奇数 $n$ 含 0，偶数 $n$ 使用半整数中心。
10. `q=x*x/(a*a)` 对应无量纲比值 $q=x^2/a^2$，供两个公式因子复用。
11. `value` 的三行是一个跨行表达式：第一、二行计算
    $2/(\sqrt{3a}\pi^{1/4})$，第三行乘 $(1-q)e^{-q/2}$；这与 Python 公式完全同形。
12. `out[i]=value` 写入当前 host 元素。
13. `}` 结束循环，`return out` 返回完整向量，最后 `}` 结束函数。

CPU 复杂度为 $O(n)$ 时间和 $O(n)$ 输出空间；除输出外每轮只有常数个 `double` 局部量。

### 6. GPU host 调度链

#### 6.1 跨翻译单元前置声明

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，
符号 `ricker_fp32_compute_device` 前置声明，第 37–38 行。

```cpp
template <typename T>
void ricker_fp32_compute_device(int, T, DeviceArray<float>&);
```

第一行声明模板参数；第二行只写参数类型而省略参数名，这是合法的函数前置声明。
它让 `.cpp` 中的公开 wrapper 能调用定义在 `.cu` 翻译单元中的内部函数。正式输出是
`DeviceArray<double>`，但该内部边界明确使用 `DeviceArray<float>&`。

#### 6.2 公开 GPU wrapper

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，符号 `ricker_device`，
第 131–140 行。

```cpp
template <typename T>
void ricker_device(int n, T width, DeviceArray<double>& out)
{
    if (n < 1 || out.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("ricker_device: n must be positive and output size must equal n");
    }
    DeviceArray<float> computed(out.size());
    ricker_fp32_compute_device(n, width, computed);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

逐行解释：

1. 前两行定义 `T` 模板的公开 GPU 函数，返回 `void`，结果经 `out` 引用写回。
2. 函数体先检查 $n\ge1$，并用 `static_cast<std::size_t>(n)` 与无符号的 `out.size()` 比较；
   由于短路求值，$n<1$ 时不会依赖负数转换后的巨大无符号值来判错。
3. 条件为真时抛出 `std::invalid_argument`；花括号清楚界定异常分支。
4. `DeviceArray<float> computed(out.size())` 在设备上分配长度 $n$ 的内部 FP32 临时数组。
5. `ricker_fp32_compute_device` 把公式结果写入 `computed`；它还负责检查宽度 $>0$。
6. `finalize_fp64_device_storage(computed)` 新建 FP64 存储数组；赋值表达式用返回结果替换 `out`。
   源码没有把 `computed` 复制到 host。
7. 最后一行结束函数；`computed` 离开作用域并释放其设备存储。

这里的公开 `out` 虽由调用者预先分配并要求 size 正确，但计算不直接写入其原存储，而是在
最后用新返回的 `DeviceArray<double>` 重新赋值。源代码层面的峰值存储至少包含 FP32 临时数组
与新 FP64 结果；在赋值完成前还可能同时保有调用者原来的 FP64 分配，均为 $O(n)$。

#### 6.3 内部 FP32 wrapper

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，
符号 `ricker_fp32_compute_device`，第 118–132 行。

```cpp
template <typename T>
void ricker_fp32_compute_device(int count, T width, DeviceArray<float>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (count < 1 || output.size() != static_cast<std::size_t>(count) ||
        detail::SimpleSignalTypePolicy<T>::load(width) <= 0.0F) {
        throw std::invalid_argument("ricker_device: n and width must be positive and output size must equal n");
    }
    cuda_utils::launch_1d_kernel(
        wavelets_detail::ricker_kernel<float, T>,
        output.size(),
        output.data(),
        count,
        width);
}
```

逐行解释：

1. 模板定义接受原始宽度类型 `T`，输出明确为 `float` 设备数组。
2. `static_assert` 在编译期拒绝不属于五种正式输入集合的 `T`；失败时不会生成可执行实例。
3. 跨两行的 `if` 依次检查 `count`、临时输出长度和加载成 FP32 后的宽度；`||` 短路。
4. 宽度比较使用 `0.0F`，确认 GPU 连续计算域是 FP32。
5. 非法时抛出包含 `ricker_device` 的异常；这也是 GPU 路径实际检查 $a>0$ 的位置。
6. `launch_1d_kernel` 的跨行调用把 kernel 模板、逻辑元素数和 kernel 实参分开列出。
7. `ricker_kernel<float,T>` 的第一个模板参数固定输出/计算存储为 float，第二个保留输入类型。
8. `output.size()` 只用于计算 grid；随后 `output.data()`、`count`、`width` 按 kernel 签名传入。
9. 结束调用并结束函数；这里没有显式 `cudaDeviceSynchronize`。

### 7. grid、block 与 launch 错误检查

#### 7.1 一维 launch 配置

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h`，
符号 `div_up`、`make_1d_launch_config`、`launch_1d_kernel_with_config` 和 `launch_1d_kernel`，
第 10–56 行。

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

逐段解释：

- `div_up` 用整数公式 $\lceil value/divisor\rceil$ 计算需要多少 block。
- `LaunchConfig1D` 把 CUDA 的 `dim3 block` 与 `dim3 grid` 组合成返回对象。
- `make_1d_launch_config` 默认 `block_size=256`，返回
  $$block=(256,1,1),\qquad grid=(\lceil n/256\rceil,1,1).$$
- `launch_1d_kernel_with_config` 是可变参数模板；`Args&&...` 与
  `std::forward<Args>(args)...` 保留 kernel 实参的值类别。
- `kernel<<<grid,block,0,stream>>>` 启动 CUDA kernel，动态共享内存字节数为 0。
- `CUDA_KERNEL_CHECK()` 检查 launch 后的 last error。
- 简化版 `launch_1d_kernel` 固定默认 stream `nullptr` 和 256 线程 block，再转发给完整版本。

对于 `ricker`，每个线程最多负责一个输出索引；最后一个 block 中多出来的线程由 kernel
的 `index>=count` 判断返回。

#### 7.2 错误检查不是同步

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/cuda_error.h`，
符号 `check_last_kernel_error` 与 `CUDA_KERNEL_CHECK`，第 43–56 行。

```cpp
inline void check_last_kernel_error(
    const char* file,
    int line)
{
    check_cuda(cudaGetLastError(), "cudaGetLastError()", file, line);
}
```

函数接收调用文件和行号，把 `cudaGetLastError()` 交给统一的 `check_cuda`；非成功状态会进入
项目异常转换。它只读取 last error，不等待 kernel 完成。

```cpp
#define CUDA_CHECK(expr) \
    ::cusignal::cuda_utils::check_cuda((expr), #expr, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK() \
    ::cusignal::cuda_utils::check_last_kernel_error(__FILE__, __LINE__)
```

两个宏用 `__FILE__`、`__LINE__` 自动附上调用位置；`CUDA_KERNEL_CHECK()` 最终调用上面的
last-error helper。这里没有 `cudaDeviceSynchronize` 或 stream synchronize，因此运行期异步
错误通常要到后续同步边界才暴露。两个 Ricker kernel 都在默认 stream 上，CUDA stream 顺序
保证后启动的存储拓宽 kernel 在前一个公式 kernel 之后执行，无需两者之间做 host 同步。

### 8. GPU device 公式与线程逻辑

#### 8.1 输入 load 与输出 store

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，
符号 `wavelet_load`、`wavelet_store`，第 13–31 行。

```cpp
template <typename T>
__host__ __device__ inline auto wavelet_load(T value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::load(value);
    } else {
        return value;
    }
}

template <typename T, typename Scalar>
__host__ __device__ inline T wavelet_store(Scalar value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
    } else {
        return static_cast<T>(value);
    }
}
```

逐段解释：

- 两个函数都是模板、强制倾向内联，并可在 host/device 编译。
- `if constexpr` 在编译期选择分支；未选择的分支不会实例化。
- 对正式简单信号类型，`wavelet_load` 调用类型策略，把宽度变成连续 FP32；否则原样返回。
- `wavelet_store<T>` 对正式类型先把计算值变成 float，再交给策略存储；否则直接 cast。
- Ricker 实例为 `ricker_kernel<float,T>`，所以 `wavelet_store<float>` 最终写 float，不发生整数
  饱和或 FP16 存储。

#### 8.2 Ricker 数学 device 函数

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，
符号 `ricker_value`，第 93–109 行。

```cpp
template <typename Scalar>
__device__ inline Scalar ricker_value(int index, int count, Scalar width)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar position =
        static_cast<Scalar>(index) -
        static_cast<Scalar>(0.5F) * static_cast<Scalar>(count - 1);
    const Scalar width_squared = width * width;
    const Scalar position_squared = position * position;
    const Scalar amplitude =
        static_cast<Scalar>(2) /
        (sqrt(static_cast<Scalar>(3) * width) *
            pow(pi, static_cast<Scalar>(0.25F)));
    return amplitude *
        (static_cast<Scalar>(1) - position_squared / width_squared) *
        exp(-position_squared / (static_cast<Scalar>(2) * width_squared));
}
```

逐行解释：

1. `Scalar` 控制所有 device 算术；当前五种正式实例中它由 `wavelet_load` 推导为 `float`。
2. `__device__ inline` 表示只能从 device 代码调用，并建议编译器内联。
3. `pi` 的字面量带 `F`，原始精度已是 FP32，再 cast 到 `Scalar`；当前仍为 float。
4. `position` 的三行构成 $x_i=i-0.5(n-1)$，所有项显式转换到 `Scalar`。
5. `width_squared` 与 `position_squared` 分别缓存 $a^2$、$x_i^2$。
6. `amplitude` 的四行构成 $2/(\sqrt{3a}\pi^{1/4})$；`sqrt` 和 `pow` 在 FP32
   `Scalar` 上计算。
7. `return` 的三行乘上 $(1-x_i^2/a^2)e^{-x_i^2/(2a^2)}$，返回当前一个样本。
8. 最后 `}` 结束 helper；没有共享状态、循环或内存访问。

#### 8.3 每线程 kernel

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，
符号 `ricker_kernel`，第 111–124 行。

```cpp
template <typename T, typename Parameter>
__global__ void ricker_kernel(
    T* output,
    int count,
    Parameter width_input)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    using Scalar = decltype(wavelet_load(width_input));
    const Scalar width = static_cast<Scalar>(wavelet_load(width_input));
    output[index] = wavelet_store<T>(ricker_value(index, count, width));
}
```

逐行解释：

1. `T` 是输出元素类型，`Parameter` 是宽度输入类型；当前调用为 `<float,T>`。
2. `__global__` 定义 host 可启动的 CUDA kernel；跨行签名接收输出指针、元素数和标量宽度。
3. 全局一维索引为 `blockIdx.x*blockDim.x+threadIdx.x`，再 cast 到 `int`。
4. 越界线程立即 `return`，保证只写 `[0,count)`。
5. `decltype(wavelet_load(width_input))` 在编译期推导 `Scalar`；正式输入统一得到 float。
6. 下一行实际 load 宽度，并显式 cast 到已推导的 `Scalar`。
7. 最后一条执行链是：调用 `ricker_value` → `wavelet_store<float>` → 写 `output[index]`。
8. 结束 kernel；每个有效线程只负责一个样本，没有 block 内同步、原子操作或共享内存。

### 9. FP32 结果如何形成 FP64 device 存储

#### 9.1 finalize 包装

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`，
符号 `finalize_fp64_device_storage`，第 40–47 行。

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

逐行解释：

1. `inline` 的 helper 返回新的 `DeviceArray<double>`，输入是只读 FP32 device 数组引用。
2. 在设备上分配同长度的 double 存储；这一步只分配，不做数值公式。
3. 跨两行调用 `widen_fp32_storage_device`，把输入指针、输出指针和数量传下去。
4. `return output` 返回新的 device 数组；没有 `to_host()`、D2H 或 H2D。

文件名和该头文件 15–18 行的旧注释仍写“host widening”，但这个函数体是更直接的代码证据：
本 SHA 的正式路径在设备上执行存储拓宽。

#### 9.2 FP32 到 FP64 的 IEEE 位模式拓宽

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu`，
符号 `widened_storage_bits`，第 11–43 行。

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
```

逐段解释：

- 函数在 device 上接收一个 `float`，返回代表 IEEE-754 double 编码的 64 位无符号整数；
  它没有执行 double 浮点加减乘除。
- `union FloatStorageBits` 让同一 32 位存储既可看作 `float value`，也可看作 `uint32_t bits`；
  `source{value}` 初始化浮点成员。
- `sign` 抽取 float 第 31 位并移到 double 第 63 位；`exponent` 抽取 8 位指数，`fraction`
  抽取 23 位尾数。
- `exponent==0` 处理零和 subnormal：零直接返回符号位；非零 subnormal 左移尾数直到规格化，
  用 `896-shift` 形成 double 指数，并放置有效尾数。
- `exponent==0xff` 处理 infinity/NaN：double 指数设为 `0x7ff`，并把 float fraction 移到
  double fraction 区，保留类别与 payload 的高层信息。
- 普通规格化数把指数增加 $1023-127=896$，尾数左移 29 位对齐到 double 的 52 位 fraction。
- 最后一行用按位或组合 sign、exponent、fraction。数值等于把原 FP32 精确拓宽为 double，
  但没有增加原计算精度。

#### 9.3 存储拓宽 kernel 与 wrapper

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu`，
符号 `widen_real_storage_kernel`（第 45–50 行）和 `widen_fp32_storage_device`（第 63–73 行）。

```cpp
__global__ void widen_real_storage_kernel(
    const float* input, std::uint64_t* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = widened_storage_bits(input[index]);
}
```

这个 device kernel 接收 float 输入指针、64 位存储输出指针和数量；每个线程计算一维索引，
仅在范围内时拓宽并写一个槽。

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

public wrapper 接收 `void* output_storage`，避免在接口处表达 device double 算术；空数组直接
返回。非空时 `launch_1d_kernel` 仍使用默认 stream 和 256 线程 block，依次传入 real kernel、
逻辑元素数、输入指针、cast 后的 `uint64_t*` 输出和 count。两个相邻 kernel 同属默认 stream，
所以公式结果先写完，再按序读取并拓宽。

GPU 的真实数值路径因此是：

$$
T\xrightarrow{load}\text{FP32 width}
\xrightarrow{ricker\_value}\text{FP32 sample}
\xrightarrow{IEEE\ bit\ widening}\text{FP64 storage}.
$$

最终数组 dtype 是 FP64，但有效数值精度仍只有 FP32。

### 10. 显式实例化与类型分发

#### 10.1 `.cpp` 中的公开 API 实例

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，
宏 `INSTANTIATE_CPU` 的 Ricker 行 324–325，宏调用第 341–346 行。

```cpp
    template std::vector<double> ricker_typed_cpu(int, T); \
    template void ricker_device(int, T, DeviceArray<double>&); \
```

这两行位于共享宏内部：第一行要求编译器生成 CPU 函数实例，第二行生成公开 GPU wrapper
实例；行末反斜杠把宏继续到下一物理行。

```cpp
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
#undef INSTANTIATE_CPU
```

五次宏调用分别生成 FP32、FP16、INT32、INT16、INT8 宽度版本。`#undef` 删除宏，避免污染
后续预处理范围。这里没有 `double` 输入实例；FP64 只作为输出/reference 存储类型。

#### 10.2 `.cu` 中的内部 GPU 实例

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，
宏 `INSTANTIATE_WAVELET_GPU` 的 Ricker 行 210，调用第 217–223 行。

```cpp
    template void ricker_fp32_compute_device(int, T, DeviceArray<float>&); \
```

该宏行显式生成内部 FP32 wrapper，确保 `.cpp` 的前置声明能在链接时找到对应定义。

```cpp
INSTANTIATE_WAVELET_GPU(float);
INSTANTIATE_WAVELET_GPU(__half);
INSTANTIATE_WAVELET_GPU(std::int32_t);
INSTANTIATE_WAVELET_GPU(std::int16_t);
INSTANTIATE_WAVELET_GPU(std::int8_t);

#undef INSTANTIATE_WAVELET_GPU

}  // namespace cusignal
```

前五行生成五种内部实例；空行不执行；`#undef` 清理宏；最后一行是该 `.cu` 文件中
`namespace cusignal` 的闭合。就 Ricker GPU 类型实例化而言，这里是源码链的最后一行。

### 11. 直接测试如何验证正确性

#### 11.1 测试输入转换与 FP64 比较 helper

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`，
符号 `convert`（第 15 行）和 `matches_fp64`（第 81–84 行）。

```cpp
template <class T> T convert(float x) { return detail::SimpleSignalTypePolicy<T>::store(x); }
```

`convert<T>` 把测试字面值转换成当前模板输入类型；整数会按策略 cast，FP16 会舍入到 half。

```cpp
bool matches_fp64(const std::vector<double>& actual, const std::vector<double>& reference)
{
    return test_evidence::f_observe_real(actual, reference, 4.0e-2);
}
```

该 helper 接收两条 FP64 host 向量，但容差为 `4.0e-2`。它用于容纳 GPU FP32 计算与 CPU
FP64 reference 的差异；这不是逐位相等验证。

#### 11.2 Ricker typed smoke 主体

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`，
`run_type<T>` 内 Ricker 块，第 343–366 行。

```cpp
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::ricker);
    DeviceArray<double> ricker_out(5);
    ricker_device(5, convert<T>(2), ricker_out);
    ok[4] = matches_fp64(ricker_out.to_host(), ricker_typed_cpu(5, convert<T>(2)));
    ricker_device(5, convert<T>(3), ricker_out);
    ok[4] = ok[4] && matches_fp64(
        ricker_out.to_host(), ricker_typed_cpu(5, convert<T>(3)));
    bool ricker_device_rejects_output_size = false;
    bool ricker_cpu_rejects_nonpositive_width = false;
    try {
        DeviceArray<double> invalid_ricker_out(4);
        ricker_device(5, convert<T>(2), invalid_ricker_out);
    } catch (const std::invalid_argument& error) {
        ricker_device_rejects_output_size = std::string(error.what()).find("ricker") !=
            std::string::npos;
    }
    try { (void)ricker_typed_cpu(5, convert<T>(0)); }
    catch (const std::invalid_argument& error) {
        ricker_cpu_rejects_nonpositive_width =
            std::string(error.what()).find("ricker") != std::string::npos;
    }
    ricker_device(5, convert<T>(2), ricker_out);
    ok[4] = ok[4] && ricker_device_rejects_output_size &&
        ricker_cpu_rejects_nonpositive_width;
```

逐段解释：

1. `f_begin_accuracy` 开始记录 `OperatorId::ricker` 的正确性证据。
2. 分配 5 个 FP64 device 输出槽，先以 $a=2$ 调 GPU，再 `to_host()` 形成同步/传输边界，
   与同参数 CPU reference 按 0.04 容差比较。
3. $a=3$ 再做一次不同宽度比较，并用 `ok[4] &&` 保留前一次失败。
4. 两个 bool 初始为 false，分别记录 GPU 输出 size 异常与 CPU 非正宽度异常。
5. 第一个 `try` 故意只分配 4 个输出却传 `n=5`；只有捕获 `invalid_argument` 且消息包含
   `ricker` 才把检查记为 true。
6. 第二个 `try` 调 CPU 宽度 0；`(void)` 明确丢弃返回向量。catch 同样检查异常类型和名称。
7. 最后再次运行有效 GPU 调用，并要求数值比较、GPU size 异常、CPU width 异常全部成立。

该块没有直接测试 GPU 的 `width<=0`、CPU/GPU 的 `n<1`、偶数 $n$、极大/极小宽度、NaN、
infinity、与 Python cuSignal 的逐点直接对比，也没有单独验证第二个存储拓宽 kernel 的每种
IEEE 特殊值。

#### 11.3 五种类型与证据汇总

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，同一测试文件，`run_type<T>` 证据记录第 758–768 行，`main` 第 788–798 行。

```cpp
    static const char* labels[14] = {"chirp", "gausspulse", "morlet", "morlet2",
        "ricker", "cwt", "chebwin", "general_cosine", "general_gaussian", "hamming",
        "kaiser", "parzen", "taylor", "triang"};
```

这是共享结果标签数组；跨三行的初始化列表中第 5 个名称是 `ricker`，与 `ok[4]` 对应。

```cpp
    namespace ev = test_evidence;
    namespace td = type_dispatch;
```

两行分别建立 namespace 别名，让后续证据记录表达式更短。

```cpp
        ev::record<T,double>(td::OperatorId::ricker,ricker_out.size(),ok[4]),
```

这是 `evidence` 初始化列表中的 Ricker 项：模板参数记录输入类型 `T` 和输出类型 `double`，
实参记录算子 ID、输出长度和最终布尔结果。该行末尾逗号表示数组后面还有其他算子项；它们
不属于 Ricker 学习范围。

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

`main` 对五种 `T` 各调用一次 `run_type`；每个类型包含 14 个算子，所以总计 70 项。
最后输出 pass/fail，并仅在 70 项全通过时返回 0。这个共享可执行文件的成功不等于 Ricker
单项必然通过，必须结合 `ricker=` 证据字段或 `ok[4]` 追踪。

#### 11.4 构建入口

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/CMakeLists.txt`，目标 `e3_wave_window_type_smoke`，
第 202–204 行。

```cmake
add_executable(e3_wave_window_type_smoke test/signal_processing/e3_wave_window_type_smoke.cu)
configure_zq500_target(e3_wave_window_type_smoke)
target_link_libraries(e3_wave_window_type_smoke PRIVATE signal_lib fft_core)
```

第一行建立测试可执行文件；第二行应用项目的 ZQ500 目标配置；第三行私有链接实现库。
本学习任务没有执行该目标，因此本文只能说明“测试源码如何验证”，不能声称 V1 已在本轮
实测 PASS。

### 12. 数学、Python、CPU 与 GPU 三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 中心坐标 $x_i=i-(n-1)/2$ | `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py` 基准 125 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 121 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，`ricker_value` 97–99 行 | 三者数学相同；奇偶长度行为一致 |
| $q=x_i^2/a^2$ | 同一 Python 文件基准 126–128 行，拆成 `xsq/wsq` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 122 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，`ricker_value` 100–108 行 | CPU/Python double；C++ GPU float |
| $A=2/(\sqrt{3a}\pi^{1/4})$ | 同一 Python 文件基准 134 行 `loop_prep` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 123–125 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，`ricker_value` 102–105 行 | 公式相同，常量精度不同 |
| $y=A(1-q)e^{-q/2}$ | 同一 Python 文件基准 127–130 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 123–126 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，`ricker_value` 106–108 行 | 同一 Ricker 解析算法 |
| 有限一维输出 | 同一 Python 文件基准 177 行 `size=points` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 119、126、128 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_device` 137–139 行 | C++ GPU 多一次存储拓宽 pass |
| 输入尺度有效域 | Python 没有显式检查 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 117–118 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，`ricker_fp32_compute_device` 122–125 行 | C++ 扩展了清晰的异常契约 |
| 输出 dtype | Python kernel 声明 FP64 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 114、119、128 行 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，同一文件 `ricker_device` 132、139 行 | dtype 相同不代表算术精度相同 |

### 13. 与 cuSignal Python 相同、等价替换和有意不同之处

#### 13.1 相同

- 都使用 $i-(n-1)/2$ 的中心化坐标；
- 都使用单位能量归一化的正中心峰 Ricker 公式；
- 都生成一个长度为 `points/n` 的一维 FP64 接口数组；
- 都是逐点独立计算，没有 FFT、卷积、共享 workspace 或跨样本递推。

#### 13.2 数学等价、工程表达不同

- Python 把 $x^2$、$a^2$ 分成 `xsq/wsq`，CPU 用 $q=x^2/a^2$，GPU device helper分别缓存
  `position_squared/width_squared`；代数等价。
- Python 的 `ElementwiseKernel` 隐藏 grid/block；C++ 显式使用 256 线程 block 和
  $\lceil n/256\rceil$ 个 block。
- Python 在 `loop_prep` 对整次调用预计算 $A$、$a^2$；C++ CPU 每个循环重新计算 `a*a`
  和 amplitude，C++ GPU 每个线程也重新计算 width square 与 amplitude。数学相同，重复
  工作量不同。

#### 13.3 有意不同

- Python kernel 接收 `float64 a` 并以 `double` 计算；C++ GPU 为 ZQ500 正式类型策略使用
  FP32 公式，再只拓宽存储。
- C++ 支持 FP32、FP16、INT32、INT16、INT8 五种宽度输入；Python 公共签名只称 `scalar`，
  kernel 入口统一为 float64。
- C++ 明确拒绝 $n<1$、$a\le0$ 和输出 size 不匹配；Python `ricker` 没有对应 Python 检查。
- C++ GPU 需要调用者传入 `DeviceArray<double>& out`，但最终会重新赋值；Python 直接返回
  CuPy 自动分配的数组。

### 14. 精度、边界、同步与 ZQ500 限制

#### 14.1 精度

- CPU：宽度先经 `SimpleSignalTypePolicy<T>::load` 成为 float，再转 double；公式中的
  `sqrt/pow/exp` 和输出使用 double。
- GPU：宽度、圆周率、位置、幅度、指数和最终公式全为 float；第二个 kernel 只把已经得到
  的 FP32 值精确编码进 double 存储。
- 因此 GPU 输出虽然是 `double`，但不可能恢复 Python FP64 kernel 在公式计算阶段保留的
  精度。typed smoke 使用绝对/观察容差 `4.0e-2`，也表明它验证的是容差内等价。
- INT32 宽度若超过 float 的连续整数精确范围，CPU 与 GPU 都会先丢失低位；FP16 宽度先受
  half 表示量化。

#### 14.2 边界

- $n\ge1$、加载后的 $a>0$ 是 C++ 契约；
- $n=1$ 时位置为 0，输出为归一化峰值 $A$；
- 偶数 $n$ 时中心在两个样本之间；
- 极小正 $a$ 可能使 $a^2$ 下溢或比值溢出，极大 $a$ 会让波形在短数组内近似平坦；
- NaN 宽度不会满足 `a<=0`，可能穿过校验并生成 NaN；正 infinity 也会穿过校验，表达式中
  可能出现不定式。测试未覆盖这些情况。

#### 14.3 同步和错误

- 两次 GPU launch 都使用默认 stream，按 stream 顺序执行；
- wrapper 没有显式同步；`CUDA_KERNEL_CHECK` 只读取 `cudaGetLastError()`；
- 测试中的 `ricker_out.to_host()` 是实际的 host 观察边界，通常会迫使前序结果可见；
- 运行期异步错误在没有后续同步的调用者场景中可能延后报告。

#### 14.4 ZQ500 平台限制

- 正式 device 公式不使用 FP64 浮点算术；FP64 仅是接口/存储 dtype；
- device 使用 `__half`、自定义 kernel、默认 stream、`sqrt/pow/exp` 和整数位操作；
- 无 FFT、BLAS 或其他科学库依赖；
- 项目构建和硬件行为必须以 `gpu_02` 中初始化 `/zq500/sdk/env.sh` 后的环境为准，不能用
  本地 Windows 或通用 NVIDIA 环境覆盖；
- 本轮没有构建或实测，所以不能把源码可编译性、设备数学函数精度或 smoke PASS 写成
  2026-08-16 的新运行证据。

### 15. 复杂度、内存布局和性能边界

| 项目 | CPU | GPU |
| --- | --- | --- |
| 公式工作量 | $O(n)$，一个循环 | $O(n)$，第一个 kernel 一线程一元素 |
| finalize | 无 | 第二个 $O(n)$ 位拓宽 kernel |
| 输出 | 连续 `std::vector<double>` | 连续 `DeviceArray<double>` |
| 临时缓冲 | 除输出外 $O(1)$ 局部量 | `DeviceArray<float>` 长度 $n$，另分配新 double 结果 |
| launch 数 | 0 | 2 |
| 数据搬运 | 无设备搬运 | 核心路径无 D2H/H2D；测试观察时才 D2H |
| 线程配置 | 不适用 | block 256，grid $\lceil n/256\rceil$ |

可能的性能瓶颈：

1. `sqrt`、`pow`、`exp` 都在每个 GPU 线程重复执行；width 对所有线程相同，却未像 Python
   `loop_prep` 那样预计算 amplitude 和 $a^2$。
2. 正式输出需要第二次全数组读写；它避免 device FP64 算术，却增加一次 kernel launch 和
   内存流量。
3. 小 $n$ 时两次 launch 的固定开销可能超过公式本身。
4. public wrapper 先要求并持有一个已分配的 FP64 `out`，随后又以新数组替换，可能造成额外
   分配峰值。本文只描述源码结构，没有形成性能结论或进行优化。

### 16. 源码与文档不一致记录

| 位置 | 文档声称 | 代码事实 | 本文采用 |
| --- | --- | --- | --- |
| SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`，`ricker_typed_cpu` 文档 75 行 | CPU “计算和固定输出为 FP32” | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_typed_cpu` 113–129 行使用 double 公式并返回 `vector<double>` | CPU FP64 算术，输入先 load 为 float |
| SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`，`ricker_device` 文档 90–93 行 | Host 拓宽、D2H→H2D、无额外 workspace | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`ricker_device` 137–139 行分配 FP32 临时并调用 device finalize | 无 Host 往返；有内部 O(n) 临时数组 |
| SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`，文件注释 15–18 行 | 所有 widening 在 host | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，同一文件 `finalize_fp64_device_storage` 40–47 行调用 `widen_fp32_storage_device` | 以函数体与 `.cu` 实现为准 |

这些差异被保留为 V1 的真实状态。`Learning/` 阶段只读 `ZKX/cusignal_cpp`，没有擅自修改
活动源码或把注释“修成一致”。

### 17. 测试覆盖与剩余风险

已由测试源码覆盖的意图：

- 五种宽度输入类型；
- $n=5$、$a=2$ 与 $a=3$ 的 GPU/CPU 容差比较；
- GPU 输出 size 不匹配；
- CPU 宽度 0；
- 输出证据记录为 `double`。

仍未由当前 Ricker 块直接覆盖：

- GPU 宽度 0/负值与 CPU/GPU 的 $n<1$；
- $n=1$、偶数 $n$、很大 $n$；
- 极小/极大宽度、NaN、infinity；
- 与 cuSignal Python FP64 结果的直接黄金对比；
- 存储拓宽 helper 对 zero、subnormal、normal、infinity、NaN、正负号的独立测试；
- 异步 kernel 运行错误的强制同步观察；
- 内部分配失败与峰值显存行为。

### 18. 阶段三自检问题与参考答案

1. V1 为什么可以作为正式可追溯版本，而不是临时工作区说明？

   **答案：**`ZKX` 在记录时整体 clean，相关文件也 clean；所有证据通过
   `git show fdd55ac8415d70379eb38a2c299047f90bcf0a41:<path>` 读取，并记录了完整 SHA、
   时间与分支，因此源码内容和行号可以重现。

2. CPU reference 的真正计算精度是什么？头文件注释为什么不能直接采信？

   **答案：**SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 的
   `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，符号 `ricker_typed_cpu` 116 行先把
   float load 转为 double，121–126 行全部使用 double 和 `std::sqrt/pow/exp`，返回
   `vector<double>`。所以公式算术是 FP64；同 SHA 头文件该符号 75 行的说明与实现矛盾。

3. GPU 从 `T width` 到正式输出经历哪些 dtype？

   **答案：**`T` 经 `SimpleSignalTypePolicy<T>::load` 进入 FP32；
   `ricker_kernel<float,T>` 写 `DeviceArray<float>`；随后 `widen_real_storage_kernel` 把每个
   float 的 IEEE 表示编码为 double 存储，最终接口是 `DeviceArray<double>`。

4. GPU 的 grid、block 和每线程职责是什么？

   **答案：**`launch_1d_kernel` 固定 block 256，grid 为 $\lceil n/256\rceil$；
   `ricker_kernel:117` 计算一维全局 index，越界线程返回，每个有效线程调用一次
   `ricker_value` 并只写 `output[index]`。

5. CPU、GPU 与 Python 的核心数学是否相同？

   **答案：**相同，三者都使用 $x=i-(n-1)/2$ 和
   $2(1-x^2/a^2)e^{-x^2/(2a^2)}/(\sqrt{3a}\pi^{1/4})$。差别在计算精度、参数验证、
   内存分配和调度方式，不在 Ricker 原理。

6. `finalize_fp64_device_storage` 是否提高了数值精度？

   **答案：**没有。它把已经舍入到 FP32 的数值精确表示成 double；新增的尾数位都是由
   原 FP32 位推导的零/对齐位，不能恢复公式计算时丢失的信息。

7. 为什么说当前实现没有 Host 拓宽？

   **答案：**SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 的
   `ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`，符号
   `finalize_fp64_device_storage` 40–47 行直接调用 `widen_fp32_storage_device`；同 SHA
   `fp64_storage_finalize.cu` 的同名函数 63–73 行在默认 stream 启动 device kernel。

8. 两个 GPU kernel 之间如何保证先后，是否发生同步？

   **答案：**二者都由 `launch_1d_kernel` 发送到默认 stream，stream 内按提交顺序执行，
   所以拓宽 kernel 在公式 kernel 之后。wrapper 没有 host 同步；`cudaGetLastError` 只检查
   launch last error，真正 host 观察在测试的 `to_host()`。

9. 模板显式实例化覆盖哪些输入，为什么没有 FP64 输入？

   **答案：**`.cpp:341–345` 与 `.cu:217–221` 覆盖 `float`、`__half`、`int32_t`、
   `int16_t`、`int8_t`。类型策略把 FP64 定位为输出存储而非正式 GPU 业务输入，所以没有
   `double` 宽度实例。

10. typed smoke 对 Ricker 实际检查了什么？

    **答案：**对每个 `T`，用 $n=5$、$a=2,3$ 比较 GPU host 结果与 CPU reference，容差
    0.04；还要求 GPU 错误输出 size 与 CPU width 0 抛出含 `ricker` 的
    `invalid_argument`，最后通过 `record<T,double>` 记录。

11. 哪些临时内存与数据搬运容易被接口注释掩盖？

    **答案：**public wrapper 分配长度 $n$ 的 `DeviceArray<float> computed`，finalize 再分配
    新的 `DeviceArray<double>`；核心路径不做 D2H/H2D。测试的 `to_host()` 才是 D2H。

12. 当前最重要的剩余正确性风险是什么？

    **答案：**GPU 非正宽度、`n<1`、特殊浮点值、偶数/单点长度、Python 黄金结果和 IEEE
    存储拓宽特殊类别都缺少 Ricker 直接测试；此外 header 中两处精度/搬运说明已经与实现
    不一致，后续维护者可能据此误判。

### 19. 建议亲自阅读顺序

以下定位均绑定 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`：

1. `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`，`ricker_typed_cpu/ricker_device` 73–96 行：先分清 CPU 返回值和 GPU `out`；
2. `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，两个公开符号 113–140 行：CPU 公式，再看 GPU wrapper 的临时分配；
3. `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，`ricker_fp32_compute_device` 118–132 行：看 width 校验和 kernel 启动；
4. `ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，`ricker_value/ricker_kernel` 93–124 行：读 device 公式；
5. `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h`，一维 launch helpers 10–56 行：补齐 grid/block/default stream；
6. `ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h` 的 `finalize_fp64_device_storage` 40–47 行，以及 `fp64_storage_finalize.cu` 的 real widening helpers 11–73 行：确认真实存储拓宽路径；
7. `.cpp:324–345`、`.cu:210–223`：核对五种显式实例；
8. `e3_wave_window_type_smoke.cu:343–366`：最后检查测试到底覆盖了什么。

## 版本差异与原理不变量

当前只有 V1，尚无 V2/V3 可比较。

| 维度 | V1（当前） | 原理不变量 |
| --- | --- | --- |
| 算法结构 | CPU 一遍公式；GPU 公式 kernel + 存储拓宽 kernel | Ricker 解析公式不变 |
| 复杂度 | CPU/GPU 均 $O(n)$；GPU 两次线性 pass | 每个输出只依赖自身坐标和共同宽度 |
| 内存访问 | GPU 写 FP32 临时、再读 FP32 写 FP64 | 输出仍是一维中心化采样 |
| 并行方式 | 256 线程 block，一线程一元素 | 样本间无依赖 |
| 精度 | CPU FP64 公式；GPU FP32 公式、FP64 存储 | 正中心峰、零点和尺度关系的数学定义不变 |
| 测试 | 五类型 typed smoke，容差 0.04 | GPU 结果应与同参数 CPU reference 对应 |

若未来优化只预计算 amplitude/$a^2$、合并存储 pass 或调整分配而不改公式，应记录为
“原理不变、实现优化”；若更改归一化、坐标、峰值频率参数化或数值算法，必须新增版本并
重新映射到 `数学物理原理.md` 的原理编号。
