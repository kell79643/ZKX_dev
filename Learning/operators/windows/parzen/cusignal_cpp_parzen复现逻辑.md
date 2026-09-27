# cusignal_cpp_parzen 复现逻辑

## 版本索引

- [V1：当前学习版本（fdd55ac）](#v1当前学习版本fdd55ac)，当前实现。
- 后续优化版本尚无。
- [版本差异与原理不变量](#版本差异与原理不变量)。

## V1：当前学习版本（fdd55ac）

### 1. Git 身份与可追溯性门禁

| 项目 | 记录 |
| --- | --- |
| 完整提交 SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| 仓库状态 | `git status --short` 无输出，整个 `ZKX` 仓库 clean |
| 当前算子相关文件 | clean，可作为正式 V1 Git 快照 |

本文所有 C++/CUDA 证据均绑定上述完整 SHA。历史内容通过 `git -C ZKX show <SHA>:cusignal_cpp/<path>` 读取，没有复制、恢复或改写 `ZKX/cusignal_cpp`。

### 2. 阅读范围与职责分层

本阶段只读取 `parzen` 的直接相关文件与必要 helper：

| 层次 | SHA 绑定的路径 | 主要符号 |
| --- | --- | --- |
| 公共接口 | `cusignal_cpp/src/windows/windows_typed.h` | `parzen_typed_cpu<T>`、`parzen_device<T>` |
| CPU/reference | `cusignal_cpp/src/windows/windows_typed.cpp` | `parzen_typed_cpu<T>` |
| GPU host wrapper | `cusignal_cpp/src/windows/windows_typed.cpp` | `parzen_device<T>` |
| GPU compute launcher | `cusignal_cpp/src/windows/windows_typed.cu` | `parzen_fp32_compute_device<T>` |
| GPU device 核心 | `cusignal_cpp/src/windows/windows_kernels.cuh` | `parzen_kernel<float>` |
| 启动 helper | `cusignal_cpp/src/cuda_utils/kernel_launch.h` | `launch_1d_kernel` |
| FP64 存储收尾 | `cusignal_cpp/src/cuda_utils/host_output_finalize.h`、`fp64_storage_finalize.cu` | `finalize_fp64_device_storage`、`widen_fp32_storage_device` |
| 类型策略 | `cusignal_cpp/src/cuda_utils/simple_signal_typed.h` | `is_simple_signal_input_v<T>` |
| 直接测试 | `cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | `matches_fp64`、`run_type<T>` 中的 Parzen 切片 |

没有把合同注册、文档表格或 benchmark 标签误称为核心计算，也没有阅读其他算子的实现。

### 3. 公共接口

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.h`，符号 `parzen_typed_cpu<T>`，第 185-196 行：

```cpp
/**
 * @brief 在 CPU 上生成 Parzen 窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务请求类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`。
 * @details CPU reference使用FP64分段多项式；无workspace/stream/科学计算库。
 * @note 对齐固定版cuSignal的float64输出、长度守卫和sym截断语义。
 */
template <class T>
std::vector<double> parzen_typed_cpu(int n, bool sym = true);
```

这是 CPU reference 声明。`T` 表示五种业务请求维度，但 `parzen` 没有业务数据数组输入，因此 `T` 不进入数学公式；返回值始终是 host `std::vector<double>`。

同 SHA、同文件，符号 `parzen_device<T>`，第 198-209 行：

```cpp
/**
 * @brief 在 GPU 上生成 Parzen 窗。
 * @tparam T 五种业务请求类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配。
 * @details 默认stream上的custom kernel只执行FP32计算，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void parzen_device(int n, DeviceArray<double>& out, bool sym = true);
```

这是 GPU 对外 wrapper 声明。它要求调用者预先提供长度为 `n` 的 `DeviceArray<double>` 存储。

接口注释中的“D2H、Host 拓宽并按需 H2D”与当前实现不一致：当前实现使用 device kernel 按 IEEE 位模式把 FP32 精确表示成 binary64 存储，没有 D2H/Host/H2D 往返。后文保留此差异。

### 4. CPU/reference 实现

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cpp`，符号 `parzen_typed_cpu<T>`，第 460-478 行：

```cpp
template <class T>
std::vector<double> parzen_typed_cpu(int length, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("parzen: length must be nonnegative");
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    for (int index = 0; index < length; ++index) {
        if (effective == 1) {
            out[index] = 1.0;
            continue;
        }
        const double x = std::fabs(index - 0.5 * (effective - 1))
            / (0.5 * effective);
        out[index] = x <= 0.5
            ? 1.0 - 6.0 * x * x + 6.0 * x * x * x
            : 2.0 * std::pow(1.0 - x, 3.0);
    }
    return out;
}
```

#### 4.1 输入守卫与有效长度

- `length < 0` 抛出 `std::invalid_argument`；
- `symmetric=true` 时 `effective=length`；
- `symmetric=false` 时 `effective=length+1`，但循环仍只写前 `length` 点，相当于扩一位再截断；
- `std::vector<double> out(length)` 同时分配并值初始化输出。

#### 4.2 每个样本的坐标

对索引 $i=0,1,\ldots,length-1$，令 $L=effective$，源码计算

$$
x_i=\frac{\left|i-\frac{L-1}{2}\right|}{L/2}.
$$

`x` 是无量纲的中心归一化绝对距离。

#### 4.3 分段公式

当 $x\le 1/2$：

$$
w_i=1-6x_i^2+6x_i^3.
$$

否则：

$$
w_i=2(1-x_i)^3.
$$

CPU 直接使用 `double` 算术，和阶段一标准 Parzen 母窗一致。它没有 FFT、卷积库、workspace、stream 或临时数学库对象。

#### 4.4 CPU 边界

- `length=0`：构造空 vector，循环零次，返回空数组；
- `length=1, symmetric=true`：`effective==1`，返回 `{1.0}`；
- `length=1, symmetric=false`：`effective==2`，源码计算 $x=0.5$，返回 `{0.25}`；这与 Python `_len_guards` 在任何 `sym` 下都直接返回 `{1.0}` 不同；
- `length=5, symmetric=true`：得到标准对称序列 `[0.016, 0.424, 1, 0.424, 0.016]`，没有 Python kernel 的小奇数索引缺陷。

### 5. GPU 调用链

```text
parzen_device<T>
  → 校验 n 与 DeviceArray<double> 输出长度
  → 分配 DeviceArray<float> computed
  → parzen_fp32_compute_device<T>
      → static_assert 五类型契约
      → launch_1d_kernel(parzen_kernel<float>)
          → 每线程计算一个 FP32 Parzen 权重
  → finalize_fp64_device_storage
      → widen_fp32_storage_device
          → widen_real_storage_kernel
              → widened_storage_bits
  → DeviceArray<double> 仅作为 FP64 结果存储返回
```

这里有两个 GPU kernel：第一个计算 Parzen FP32 数值，第二个把每个 FP32 值转换成等值 binary64 位模式。两者都在默认 stream 上，依靠同一 stream 的顺序保证先计算、后写存储。

### 6. GPU host wrapper

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cpp`，符号 `parzen_device<T>`，第 480-489 行：

```cpp
template <class T>
void parzen_device(int length, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("parzen_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length != 0) parzen_fp32_compute_device<T>(length, computed, symmetric);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

职责按执行顺序为：

1. `length < 0` 先短路，避免把负数转换成巨大 `size_t`；
2. 输出大小必须精确等于请求长度；
3. 分配同长度 FP32 device 临时数组；
4. 长度非零时启动计算；
5. 把 FP32 结果转换成 FP64 存储并赋给 `out`。

wrapper 不计算分段公式，也不手写 grid/block；它负责契约、临时内存和调用编排。

### 7. GPU compute launcher

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cu`，符号 `parzen_fp32_compute_device<T>`，第 78 行：

```cpp
template <class T> void parzen_fp32_compute_device(int n,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("parzen_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::parzen_kernel<float>,o.size(),n,o.data(),sym?n:n+1);}
```

虽然源码压成一行，语义上包含四个动作：模板入口、五类型编译期断言、正长度/输出大小检查、启动 `parzen_kernel<float>`。传入参数依次是 `output_count=n`、输出指针和 `effective_count=sym?n:n+1`。

因此，`T` 不决定数学精度；无论请求维度是 FP32、FP16、INT32、INT16 还是 INT8，device 分段计算都固定为 FP32。

### 8. GPU device 核心

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_kernels.cuh`，符号 `parzen_kernel<T>`，第 166-190 行：

```cpp
template <typename T>
__global__ void parzen_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(output[index]));
    const Scalar position = fabs(
        static_cast<Scalar>(index) - static_cast<Scalar>(0.5F) *
            static_cast<Scalar>(effective_count - 1)) /
        (static_cast<Scalar>(0.5F) * static_cast<Scalar>(effective_count));
    Scalar value;
    if (position <= static_cast<Scalar>(0.5F)) {
        value = static_cast<Scalar>(1) - static_cast<Scalar>(6) *
            position * position + static_cast<Scalar>(6) *
            position * position * position;
    } else {
        const Scalar tail = static_cast<Scalar>(1) - position;
        value = static_cast<Scalar>(2) * tail * tail * tail;
    }
    output[index] = window_store<T>(value);
}
```

每个线程计算 $index=blockIdx.x\cdot blockDim.x+threadIdx.x$，越界即返回。`effective_count==1` 写 1 并避免除零。当前 launcher 固定调用 `parzen_kernel<float>`；`decltype` 是未求值语境，只推导 `Scalar`，不会读取未初始化的 `output[index]`。随后直接计算与 CPU 相同的坐标和两段公式，最后写 FP32。

### 9. grid、block、stream 与错误检查

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/cuda_utils/kernel_launch.h`，符号 `make_1d_launch_config`、`launch_1d_kernel_with_config`、`launch_1d_kernel`，第 21-55 行。

```cpp
inline LaunchConfig1D make_1d_launch_config(
    std::size_t elements,
    unsigned int block_size = 256)
{
    return LaunchConfig1D{
        dim3(block_size),
        dim3(static_cast<unsigned int>(div_up(elements, static_cast<std::size_t>(block_size))))
    };
}
```

```cpp
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

所以 block 固定为 256，grid 为 $\lceil n/256\rceil$，`stream=nullptr` 表示默认 stream，动态 shared memory 为 0。底层启动后调用 `CUDA_KERNEL_CHECK()` 检查 launch 错误，但不在这里显式同步 device。

### 10. FP32 计算到 FP64 存储

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/cuda_utils/host_output_finalize.h`，符号 `finalize_fp64_device_storage`，第 40-47 行：

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

它分配 `DeviceArray<double>`，然后调用 device 存储拓宽函数。函数体没有 `to_host()` 或 host vector。

完整 SHA 下的 `cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu` 中：`widened_storage_bits` 位于 11-43 行，`widen_real_storage_kernel` 位于 45-50 行，`widen_fp32_storage_device` 位于 63-73 行。关键 wrapper 是：

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

它把 double 存储地址视为 `uint64_t*` 写入，device 不执行 double 加减乘除。位函数读取 FP32 的符号、指数和尾数，用整数位运算构造等值 binary64。最终类型是 `double`，但数值有效精度仍只有前一阶段 FP32；拓宽不会恢复已经舍入的信息。

### 11. template 与显式实例化

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`：

- `cusignal_cpp/src/windows/windows_typed.cpp:586-587` 把 CPU 与 GPU wrapper 放入实例化宏；第 594-598 行对 `float`、`__half`、`int32_t`、`int16_t`、`int8_t` 展开；
- `cusignal_cpp/src/windows/windows_typed.cu:123` 把 compute launcher 放入实例化宏；第 126 行对同样五种类型展开。

相关原始项为：

```cpp
template std::vector<double> parzen_typed_cpu<T>(int, bool); \
template void parzen_device<T>(int, DeviceArray<double>&, bool); \
```

```cpp
template void parzen_fp32_compute_device<T>(int,DeviceArray<float>&,bool); \
```

`T` 是编译期请求契约维度，不是输出 dtype。五种实例都返回/存储 FP64，GPU 数学核心都调用 `parzen_kernel<float>`。

### 12. 内存布局、数据搬运与同步

| 阶段 | 存储 | 行为 |
| --- | --- | --- |
| 调用前 | `DeviceArray<double> out(n)` | 调用者提供正式输出存储 |
| GPU 计算 | `DeviceArray<float> computed(n)` | wrapper 分配连续 FP32 临时数组 |
| Parzen kernel | `computed[index]` | 每线程写一个连续 FP32 元素 |
| 存储拓宽 | `DeviceArray<double> output(n)` | 第二个 kernel 以 64 位整数位模式写入 |
| wrapper 返回 | `out = ...` | 赋值接收新的 FP64 device storage |

当前路径没有 workspace、FFT、科学计算库、shared memory、显式 D2H 或 H2D。没有 wrapper 级显式同步；同一默认 stream 保证 kernel 顺序，真正读取到 host 时由 `to_host()` 等后续操作形成必要同步/复制边界。临时空间为 $O(n)$ FP32 加 $O(n)$ FP64；两个 kernel 都是 $O(n)$。

### 13. 数学、Python、CPU 与 GPU 映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 有效长度 $L=sym?n:n+1$ | 基准 `windows.py:325-329` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cpp`，`parzen_typed_cpu<T>`，464-466 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cu`，`parzen_fp32_compute_device<T>`，78 | C++ 直接只写前 n 点 |
| $x=|i-(L-1)/2|/(L/2)$ | Python kernel 试图用三段索引重构 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cpp`，`parzen_typed_cpu<T>`，471-472 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_kernels.cuh`，`parzen_kernel<T>`，175-179 | C++ 直接按全局 index 计算，避免 Python 小奇数段长缺陷 |
| 内段 $1-6x^2+6x^3$ | 基准 `windows.py:303-304` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cpp`，`parzen_typed_cpu<T>`，473-475 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_kernels.cuh`，`parzen_kernel<T>`，181-184 | CPU double、GPU float |
| 外段 $2(1-x)^3$ | 基准 `windows.py:299-300,307-308` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_typed.cpp`，`parzen_typed_cpu<T>`，475 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/src/windows/windows_kernels.cuh`，`parzen_kernel<T>`，185-188 | C++ 直接坐标保持左右对称 |
| 固定 FP64 输出 | Python kernel 声明 `float64 w` | `std::vector<double>` | FP32 算后位模式拓宽至 `DeviceArray<double>` | ZQ500 GPU 不执行 FP64 算术 |
| 单点窗 | Python 对所有 sym 返回 1 | CPU/GPU 只在 `effective==1` 返回 1 | 同左 | `n=1,sym=false` C++ 返回 0.25 |

### 14. 与 cuSignal Python 的相同、替换和差异

- 相同：都接收长度与 `sym`；负长度非法、零长度为空；数学目标是同一 Parzen 分段式；正式输出 FP64。
- 等价替换：Python 使用 `ElementwiseKernel` 和三段索引，C++ 使用显式 `__global__` kernel 并从全局索引直接计算 $x$。
- 有意平台差异：Python 直接 FP64 device 计算；ZQ500 C++ GPU 固定 FP32 计算，再拓宽为 FP64 存储。
- 修复：C++ 直接坐标消除了 Python 对某些小奇数长度的非对称索引错误。
- 未对齐：Python 在 `_extend` 前对 `M<=1` 返回 ones；C++ 的 `n=1,sym=false` 返回 0.25。
- 注释不一致：公共头声明 Host 拓宽，当前实现实际为 device 位模式拓宽。

### 15. ZQ500 平台约束

同一完整 SHA 的 `docs/development/ZQ500_RUNTIME.md:457-485` 规定：正式业务输入维度覆盖 FP32、FP16、INT32、INT16、INT8；平台不支持 FP64 device 正式算术路径；CUDA 11.7 兼容不等于 NVIDIA CUDA，需以 ZQ500 SDK 为准。

Parzen 没有 FFT 后端依赖，不受 FFT 点数限制；它只用索引、绝对值、乘加与整数位运算。正式 GPU 计算固定 FP32，符合“不得依赖 FP64 fallback”。本文没有在 2026-08-16 新运行 ZQ500 测试；环境版本和历史 verified 状态来自仓库文档，不是本次实测。

### 16. 测试如何验证正确性

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`，`matches_fp64`，第 81-84 行：

```cpp
bool matches_fp64(const std::vector<double>& actual, const std::vector<double>& reference)
{
    return test_evidence::f_observe_real(actual, reference, 4.0e-2);
}
```

同文件 `run_type<T>` 的 Parzen 部分，第 650-672 行：

```cpp
test_evidence::f_begin_accuracy(type_dispatch::OperatorId::parzen);
parzen_device<T>(5, fixed_fp64_window_out);
const auto parzen_actual = fixed_fp64_window_out.to_host();
const auto parzen_reference = parzen_typed_cpu<T>(5);
DeviceArray<double> parzen_alternate_out(4);
parzen_device<T>(4, parzen_alternate_out);
const bool parzen_alternate_matches = matches_fp64(
    parzen_alternate_out.to_host(), parzen_typed_cpu<T>(4));
parzen_device<T>(1, singleton_fp64_window_out);
const bool parzen_singleton_is_one =
    singleton_fp64_window_out.to_host() == std::vector<double>{1.0};
parzen_device<T>(0, empty_fp64_window_out);
bool parzen_device_rejects_negative = false;
bool parzen_cpu_rejects_negative = false;
try { parzen_device<T>(-1, empty_fp64_window_out); }
catch (const std::invalid_argument&) { parzen_device_rejects_negative = true; }
try { (void)parzen_typed_cpu<T>(-1); }
catch (const std::invalid_argument&) { parzen_cpu_rejects_negative = true; }
ok[11] = matches_fp64(parzen_actual, parzen_reference) &&
    parzen_alternate_matches &&
    parzen_singleton_is_one && parzen_typed_cpu<T>(1) == std::vector<double>{1.0} &&
    empty_fp64_window_out.empty() && parzen_typed_cpu<T>(0).empty() &&
    parzen_device_rejects_negative && parzen_cpu_rejects_negative;
```

它覆盖默认对称的奇数 5、偶数 4、单点、空数组和负长度异常。`main` 位于第 788-798 行，对五种 `T` 分别调用 `run_type<T>`；第 775 行以 `record<T,double>` 记录请求类型和固定 double 输出。

### 17. 尚未覆盖的风险

1. GPU 只与同仓库 CPU reference 比较，没有直接比较固定版 cuSignal/SciPy；两端若同时错误可能一起通过。
2. `4.0e-2` 容差较宽，可能掩盖较小系统偏差。
3. 没有测试 `sym=false`，尤其漏掉 `n=1,sym=false` 的 0.25/1.0 差异。
4. 没有单独断言左右对称、非负、峰值与端点。
5. 没有测试输出 size 不匹配。
6. 本次没有实际构建或运行，不能把源码存在或历史状态写成本轮 PASS。
7. 公共头文件的 Host 拓宽说明与实现不符。

### 18. 复杂度与性能边界

| 部分 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| CPU reference | $O(n)$ | $O(n)$ | 每点常数次 double 算术 |
| GPU Parzen kernel | $O(n)$ 总工作量 | $O(n)$ FP32 临时输出 | 每线程一点 |
| FP64 存储拓宽 | $O(n)$ | $O(n)$ FP64 输出 | 第二次线性 kernel |
| 总 GPU wrapper | $O(n)$ 总工作量 | FP32 + FP64 两份线性存储 | 两次 kernel launch |

潜在瓶颈是两次 kernel 启动和两份 device 存储写入；对很小窗口，启动开销可能远大于算术。本文不是性能优化任务，没有给出未实测的速度结论。

### 19. 阶段三自检问题

1. V1 的完整 SHA、分支和 dirty 状态是什么？
2. `T` 在 Parzen 模板中代表什么，为什么不进入数学公式？
3. CPU 如何实现 `sym=false` 的扩一位再截断？
4. CPU 的归一化坐标公式是什么？
5. GPU wrapper、compute launcher 和 device kernel 各自负责什么？
6. GPU 的 grid、block 和每线程职责是什么？
7. 为什么 `decltype(window_load(output[index]))` 不会读取未初始化输出？
8. GPU 如何在不执行 FP64 算术的情况下产生 `DeviceArray<double>`？
9. “FP64 存储”是否意味着拥有 FP64 计算精度？
10. CPU/GPU 与 Python 小奇数实现有什么差异？
11. `n=1,sym=false` 的 Python 与 C++ 结果分别是什么？
12. 当前测试覆盖哪些边界，遗漏哪些关键条件？

### 20. 阶段三参考答案

1. SHA 是 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`；`git status --short` 无输出，V1 clean。
2. `T` 是 FP32、FP16、INT32、INT16、INT8 五种业务请求契约维度。Parzen 没有数据输入，kernel 固定为 float，所以 `T` 只参与编译期准入和证据记录。
3. `windows_typed.cpp:464-466` 令 `effective=length+1`，但 vector 和循环仍只有 `length` 点，直接只计算扩展窗前 n 点。
4. $x_i=|i-(L-1)/2|/(L/2)$，证据是 CPU 471-472 行和 GPU kernel 176-179 行。
5. wrapper 校验形状并管理临时/正式存储；launcher 做类型断言并启动 kernel；device kernel 计算分段公式。
6. block=256，grid=$\lceil n/256\rceil$，默认 stream；每个有效线程只写一个元素。
7. `decltype` 是未求值语境，只推导返回类型，不执行表达式。
8. 先 FP32 计算，再由 `widened_storage_bits` 以整数位运算构造等值 binary64 位模式。
9. 不是。输出能以 double 存储，但信息仍受 FP32 计算和舍入限制。
10. C++ 直接由全局索引计算 $x$，长度 5 得标准对称序列；Python 的三段索引对部分小奇数长度出错。
11. Python 对任意 `sym` 返回 `[1]`；C++ 的有效长度为 2，按内段得到 `[0.25]`。
12. 测试覆盖默认对称的长度 5、4、1、0、负长度和五种模板类型；遗漏 periodic、size mismatch、硬编码标准值和数学不变量。

### 21. V1 相关代码按源码顺序逐语义块解释

本节是阶段三代码完整陪读视图。空行不单独解释；每个非空 Parzen 专属源码行在一个语义块中出现一次，顺序从公共声明、CPU、GPU wrapper、launcher、kernel、显式实例化直到测试 `main` 的最后一个右花括号。

#### 21.1 CPU 公共声明

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`cusignal_cpp/src/windows/windows_typed.h:185-196`；`parzen_typed_cpu<T>`：

```cpp
/**
 * @brief 在 CPU 上生成 Parzen 窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务请求类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`。
 * @details CPU reference使用FP64分段多项式；无workspace/stream/科学计算库。
 * @note 对齐固定版cuSignal的float64输出、长度守卫和sym截断语义。
 */
template <class T>
std::vector<double> parzen_typed_cpu(int n, bool sym = true);
```

逐行对应说明：

1. `/**` 开始 Doxygen 文档块。
2. `@brief` 给出接口摘要，明确这是 CPU reference。
3. `@tparam` 列出五种请求类型并声明输出固定 FP64。
4. `@param n` 说明长度、零长度和负长度契约。
5. `@param sym` 说明对称/周期有效长度以及默认值。
6. `@return` 说明 host 一维 vector、长度 n、元素 double。
7. `@throws` 说明负长度异常类型。
8. `@details` 把 CPU double 多项式与 workspace/stream/科学库区分开。
9. `@note` 声明它试图对齐固定版 Python 语义；后文指出 singleton periodic 例外。
10. `*/` 结束文档块。
11. `template <class T>` 声明函数模板。
12. 最后一行声明返回 `std::vector<double>`，参数是 `int n` 与默认 `true` 的布尔值；分号表示只有声明、没有函数体。

#### 21.2 GPU 公共声明

同 SHA；`cusignal_cpp/src/windows/windows_typed.h:198-209`；`parzen_device<T>`：

```cpp
/**
 * @brief 在 GPU 上生成 Parzen 窗。
 * @tparam T 五种业务请求类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配。
 * @details 默认stream上的custom kernel只执行FP32计算，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void parzen_device(int n, DeviceArray<double>& out, bool sym = true);
```

逐行对应说明：

1. 文档块开始。
2. 摘要把该入口定位为 GPU 版本。
3. `@tparam` 说明请求类型与计算/输出 dtype 解耦。
4. `@param n` 给出长度边界。
5. `@param out` 声明调用者提供 device FP64 存储；“Host 拓宽”是已过期说明。
6. `@param sym` 复用 CPU 对称/周期契约。
7. `@throws` 同时覆盖负长度和 shape 不匹配。
8. `@details` 第一行说明默认 stream 和 FP32 kernel，但所述 D2H/Host/H2D 与当前实现不符。
9. 续行说明无 workspace，并宣称保持 cuSignal 语义。
10. 文档块结束。
11. 声明模板类型参数。
12. `void` 表示通过引用参数 `out` 交付结果；`DeviceArray<double>&` 是可修改左值引用；`sym` 默认 true；分号结束声明。

#### 21.3 CPU 输入、有效长度与输出分配

同 SHA；`cusignal_cpp/src/windows/windows_typed.cpp:460-465`；`parzen_typed_cpu<T>`：

```cpp
template <class T>
std::vector<double> parzen_typed_cpu(int length, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("parzen: length must be nonnegative");
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
```

逐行对应说明：

1. 定义函数模板。
2. 给出返回类型、函数名和两个形参；定义处不重复默认实参。
3. 左花括号开始函数体。
4. 单行 `if` 在负长度时构造并抛出 `invalid_argument`。
5. 条件运算符在对称模式选 `length`，周期模式选 `length+1`，结果绑定为不可修改的 `effective`。
6. 分配长度为 `length` 的 double vector；数值元素被值初始化。

#### 21.4 CPU 循环与单点分支

同 SHA；`windows_typed.cpp:466-470`：

```cpp
    for (int index = 0; index < length; ++index) {
        if (effective == 1) {
            out[index] = 1.0;
            continue;
        }
```

逐行对应说明：

1. `for` 把 `index` 从 0 递增到 `length-1`，左花括号开始循环体。
2. 有效长度为 1 时进入特殊分支，避免后续坐标除以零。
3. 当前输出元素写 double 常量 1。
4. `continue` 跳过本次循环余下语句，进入下一索引。
5. 右花括号结束 singleton 分支。

#### 21.5 CPU 坐标与分段公式

同 SHA；`windows_typed.cpp:471-475`：

```cpp
        const double x = std::fabs(index - 0.5 * (effective - 1))
            / (0.5 * effective);
        out[index] = x <= 0.5
            ? 1.0 - 6.0 * x * x + 6.0 * x * x * x
            : 2.0 * std::pow(1.0 - x, 3.0);
```

逐行对应说明：

1. 第一行开始计算中心绝对距离：索引减去 $(effective-1)/2$，`fabs` 取绝对值；表达式在下一行继续。
2. 除以 `effective/2` 得无量纲 $x$，分号结束初始化。
3. 以 `x<=0.5` 为条件开始三元表达式并选择内外段。
4. 问号分支计算内段 $1-6x^2+6x^3$。
5. 冒号分支用 `std::pow` 计算外段 $2(1-x)^3$，分号结束对 `out[index]` 的赋值。

#### 21.6 CPU 返回

同 SHA；`windows_typed.cpp:476-478`：

```cpp
    }
    return out;
}
```

逐行对应说明：第一行结束 `for` 循环；第二行按值返回 vector，编译器可用移动/返回值优化；第三行结束函数体。

#### 21.7 GPU host wrapper 的契约与临时存储

同 SHA；`windows_typed.cpp:480-489`；`parzen_device<T>`：

```cpp
template <class T>
void parzen_device(int length, DeviceArray<double>& out, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("parzen_device: output size must equal requested length");
    }
    DeviceArray<float> computed(out.size());
    if (length != 0) parzen_fp32_compute_device<T>(length, computed, symmetric);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

逐行对应说明：

1. 声明模板类型。
2. 定义返回 void 的 wrapper，结果写入 `out` 引用。
3. 开始函数体。
4. 逻辑或先检查负长度；短路保证负数时不执行右侧无符号转换。非负时再检查输出元素数。
5. 不满足契约时抛出 `invalid_argument`。
6. 结束异常分支。
7. 按正式输出长度分配 FP32 device 临时数组。
8. 非零长度时调用模板 launcher；零长度跳过，防止 launcher 的正长度守卫失败。
9. 调用收尾 helper 生成 FP64 存储，并赋回调用者输出。
10. 结束 wrapper。

#### 21.8 GPU compute launcher 的单行完整语义

同 SHA；`cusignal_cpp/src/windows/windows_typed.cu:78`；`parzen_fp32_compute_device<T>`：

```cpp
template <class T> void parzen_fp32_compute_device(int n,DeviceArray<float>&o,bool sym){static_assert(detail::is_simple_signal_input_v<T>);if(n<1||o.size()!=size_t(n))throw std::invalid_argument("parzen_device: compute output size must equal positive n");cuda_utils::launch_1d_kernel(windows_detail::parzen_kernel<float>,o.size(),n,o.data(),sym?n:n+1);}
```

这一物理行由分号分成连续动作：`template <class T>` 定义模板；函数接收长度、FP32 device 输出引用和 `sym`；`static_assert` 在编译期限制五种正式类型；`if` 检查正长度及输出大小；失败时抛异常；最后调用通用 1D launcher。launcher 的第一个参数固定为 `parzen_kernel<float>`，第二个参数 `o.size()` 是逻辑线程数，随后三个参数传给 kernel：`output_count=n`、`output=o.data()`、`effective_count=sym?n:n+1`；末尾右花括号结束函数。

#### 21.9 GPU kernel 的模板、索引与越界守卫

同 SHA；`cusignal_cpp/src/windows/windows_kernels.cuh:166-170`；`parzen_kernel<T>`：

```cpp
template <typename T>
__global__ void parzen_kernel(int output_count, T* output, int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
```

逐行对应说明：

1. 声明元素存储类型模板。
2. `__global__` 表示由 host 启动、在 device 执行；参数依次为输出逻辑长度、device 指针和有效长度。
3. 开始 kernel 体。
4. 由 block 和 thread 三个内建索引量计算一维全局索引，再显式转成 int。
5. 向上取整产生的尾部线程若越界立即返回。

#### 21.10 GPU kernel 的 singleton 与计算类型

同 SHA；`windows_kernels.cuh:171-175`：

```cpp
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    using Scalar = decltype(window_load(output[index]));
```

逐行对应说明：

1. 有效长度为 1 时进入特殊路径。
2. `window_store<T>` 把 FP32 常量按存储类型规则转换后写当前元素。
3. 当前线程返回，不再计算坐标。
4. 结束特殊分支。
5. `using` 创建类型别名；`decltype` 不求值，只由 `window_load` 返回类型推导 `Scalar`，因此不读取未初始化输出。

#### 21.11 GPU kernel 的归一化坐标

同 SHA；`windows_kernels.cuh:176-180`：

```cpp
    const Scalar position = fabs(
        static_cast<Scalar>(index) - static_cast<Scalar>(0.5F) *
            static_cast<Scalar>(effective_count - 1)) /
        (static_cast<Scalar>(0.5F) * static_cast<Scalar>(effective_count));
    Scalar value;
```

逐行对应说明：

1. 开始初始化不可修改的 `position`，`fabs` 将对完整差值取绝对值。
2. 把索引和 0.5 转成 `Scalar`，开始构造中心偏移。
3. 把 `effective_count-1` 转型，右括号结束 `fabs`，随后除法继续。
4. 分母是 `0.5*effective_count`；分号结束坐标计算。
5. 声明尚未赋值的 `value`，后续两个互斥分支都为其赋值。

#### 21.12 GPU kernel 的两段公式与写回

同 SHA；`windows_kernels.cuh:181-190`：

```cpp
    if (position <= static_cast<Scalar>(0.5F)) {
        value = static_cast<Scalar>(1) - static_cast<Scalar>(6) *
            position * position + static_cast<Scalar>(6) *
            position * position * position;
    } else {
        const Scalar tail = static_cast<Scalar>(1) - position;
        value = static_cast<Scalar>(2) * tail * tail * tail;
    }
    output[index] = window_store<T>(value);
}
```

逐行对应说明：

1. 中心距离不超过 0.5 时进入内段。
2. 开始计算 1 减 6 倍平方，表达式续行。
3. 加 6 倍立方，表达式仍续行。
4. 完成第三个 `position` 因子并以分号结束赋值。
5. 关闭内段并开始外段。
6. 计算到支撑端点的剩余距离 `tail=1-position`。
7. 用三次连乘得到 $2tail^3$。
8. 结束分支。
9. 把计算类型值转换成 `T` 后写回当前输出元素。
10. 结束 kernel。

#### 21.13 通用 1D 启动配置与错误检查

同 SHA；`cusignal_cpp/src/cuda_utils/kernel_launch.h:21-55`：

```cpp
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

逐行对应说明：`make_1d_launch_config` 的签名接收元素数和默认 256 的 block 大小；函数体返回聚合对象，第一项构造 block，第二项用 `div_up` 向上取整 grid，随后两个右花括号分别结束初始化和函数。带配置 launcher 的模板参数包含 kernel 类型及可变参数包；形参依次接收 kernel、元素数、stream、block 大小和完美转发参数；函数体先生成配置，再用 `<<<grid,block,0,stream>>>` 启动并转发参数，随后检查 launch 错误，最后闭合函数。简化 launcher 再声明同样的模板参数；它只接收 kernel、元素数和参数包，并调用带配置版本，依次传原 kernel、元素数、默认 stream 的 `nullptr`、固定 256、转发参数，最后闭合调用与函数。

#### 21.14 FP64 device 存储收尾 wrapper

同 SHA；`cusignal_cpp/src/cuda_utils/host_output_finalize.h:40-47`；`finalize_fp64_device_storage`：

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

逐行对应说明：第一行开始声明返回 double device 数组的 inline 函数；第二行接收不可修改的 FP32 device 数组引用并闭合签名；第三行开始函数体；第四行按相同元素数分配 double 存储；第五行开始调用拓宽函数；第六行传 FP32 输入指针、double 存储指针和元素数并闭合调用；第七行返回新数组；第八行结束函数。整个块没有 host copy。

#### 21.15 FP32 到 binary64 位模式拓宽

同 SHA；`cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu:11-50`。位转换函数与实数 kernel 为：

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
```

逐行对应说明：函数签名表明 device 输入 float、返回 64 位整数位模式；union 的两成员让同一 32 位存储可按 float 或 bits 观察，`source{value}` 初始化 float 成员。接着两行提取符号位并搬到 bit63；后两行提取 8 位指数和 23 位尾数。`exponent==0` 处理零与次正规数：尾数也为零时直接返回带符号零；否则循环左移尾数直到隐含最高位出现并累计 shift，再用偏置差 896 构造 11 位指数、把剩余尾数放到 binary64 尾数字段，按位或后返回。`exponent==0xff` 处理无穷/NaN：把 binary64 指数全置 1并迁移尾数。普通数路径把 FP32 指数加偏置差 896，把尾数左移 29 位，最后合并符号、指数、尾数。右花括号结束转换函数。随后 `__global__` 两行定义实数拓宽 kernel；函数体计算全局索引，仅对 `index<count` 的线程调用位转换并写一个 64 位槽，最后结束 kernel。

同文件第 63-73 行的启动 wrapper：

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

逐行对应说明：前两行定义函数并接收 FP32 输入、无类型输出地址和元素数；左花括号开始函数；零元素直接返回；随后开始通用 launcher 调用，依次传 kernel、逻辑线程数、输入指针、把输出地址转换成 `uint64_t*`、kernel 自身的 count，最后两行闭合调用与函数。

#### 21.16 显式实例化直到最后一种类型

同 SHA；`cusignal_cpp/src/windows/windows_typed.cpp:586-598` 中 Parzen 项与宏调用：

```cpp
    template std::vector<double> parzen_typed_cpu<T>(int, bool); \
    template void parzen_device<T>(int, DeviceArray<double>&, bool); \
INSTANTIATE_WINDOWS_CPU(float);
INSTANTIATE_WINDOWS_CPU(__half);
INSTANTIATE_WINDOWS_CPU(std::int32_t);
INSTANTIATE_WINDOWS_CPU(std::int16_t);
INSTANTIATE_WINDOWS_CPU(std::int8_t);
```

前两行是宏体中的显式实例化定义，反斜杠续接宏；后五行依次令 `T` 为 FP32、FP16、INT32、INT16、INT8，使 CPU reference 与 GPU wrapper 都有链接期实体。`.cu:123,126` 以同样五类型生成 compute launcher 实体。

#### 21.17 测试比较 helper

同 SHA；`cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu:81-84`；`matches_fp64`：

```cpp
bool matches_fp64(const std::vector<double>& actual, const std::vector<double>& reference)
{
    return test_evidence::f_observe_real(actual, reference, 4.0e-2);
}
```

第一行定义接收实际值和 reference 常量引用的布尔函数；第二行开始函数体；第三行把两数组和容差 0.04 交给证据比较函数并返回其结果；第四行结束函数。

#### 21.18 Parzen 测试主体直到 `ok[11]`

同 SHA；同测试文件 `run_type<T>`，第 650-672 行：

```cpp
test_evidence::f_begin_accuracy(type_dispatch::OperatorId::parzen);
parzen_device<T>(5, fixed_fp64_window_out);
const auto parzen_actual = fixed_fp64_window_out.to_host();
const auto parzen_reference = parzen_typed_cpu<T>(5);
DeviceArray<double> parzen_alternate_out(4);
parzen_device<T>(4, parzen_alternate_out);
const bool parzen_alternate_matches = matches_fp64(
    parzen_alternate_out.to_host(), parzen_typed_cpu<T>(4));
parzen_device<T>(1, singleton_fp64_window_out);
const bool parzen_singleton_is_one =
    singleton_fp64_window_out.to_host() == std::vector<double>{1.0};
parzen_device<T>(0, empty_fp64_window_out);
bool parzen_device_rejects_negative = false;
bool parzen_cpu_rejects_negative = false;
try { parzen_device<T>(-1, empty_fp64_window_out); }
catch (const std::invalid_argument&) { parzen_device_rejects_negative = true; }
try { (void)parzen_typed_cpu<T>(-1); }
catch (const std::invalid_argument&) { parzen_cpu_rejects_negative = true; }
ok[11] = matches_fp64(parzen_actual, parzen_reference) &&
    parzen_alternate_matches &&
    parzen_singleton_is_one && parzen_typed_cpu<T>(1) == std::vector<double>{1.0} &&
    empty_fp64_window_out.empty() && parzen_typed_cpu<T>(0).empty() &&
    parzen_device_rejects_negative && parzen_cpu_rejects_negative;
```

逐行对应说明：

1. 开始 Parzen 精度证据记录。
2. 生成默认对称长度 5 的 GPU 结果。
3. `to_host()` 复制/同步并以 `auto` 保存实际 vector。
4. 调用 CPU reference 生成长度 5 对照。
5. 分配长度 4 的另一份 device double 输出。
6. 生成长度 4 GPU 窗。
7. 开始布尔比较表达式。
8. 把长度 4 GPU host 结果与 CPU reference 比较并结束初始化。
9. 生成默认对称 singleton。
10. 开始 singleton 精确比较布尔值。
11. host vector 必须严格等于 `{1.0}`。
12. 调用零长度 GPU 路径，预期保持空输出。
13. 初始化 GPU 负长度拒绝标志为 false。
14. 初始化 CPU 负长度拒绝标志为 false。
15. `try` 调用负长度 GPU；若没有异常，标志仍 false。
16. 捕获 `invalid_argument` 时把 GPU 标志置 true。
17. `try` 调用负长度 CPU，并用 `(void)` 丢弃正常返回值。
18. 捕获同类异常时把 CPU 标志置 true。
19. 开始组合 `ok[11]`，第一项比较长度 5 GPU/CPU。
20. 逻辑与续接长度 4 比较。
21. 同时要求 GPU singleton 与 CPU singleton 为 1。
22. 同时要求 GPU/CPU 零长度输出为空。
23. 最后要求两个负长度拒绝标志都为真，分号结束整个测试判定。

#### 21.19 五类型测试入口直到文件最后一行

同 SHA；同测试文件 `main`，第 788-798 行：

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

逐行对应说明：

1. 定义程序入口，返回进程状态码 int。
2. 开始函数体。
3. 累计通过数从 0 开始。
4. 运行 FP32 请求类型套件并累加通过项。
5. 运行 FP16 套件。
6. 运行 INT32 套件。
7. 运行 INT16 套件。
8. 运行 INT8 套件。
9. 开始输出总项 70 和通过数，流表达式在下一行继续。
10. 输出失败数与换行，分号结束输出语句。
11. 三元表达式在 70 项全通过时返回 0，否则返回 1。
12. 最后一个右花括号结束 `main`，这也是本阶段所选直接测试源码的最后一行。

## 版本差异与原理不变量

目前只有 V1，没有可比较的 V2/V3。

| 版本 | 算法结构 | 计算精度 | 并行与内存 | 已知边界 | 数学原理不变量 |
| --- | --- | --- | --- | --- | --- |
| V1 `fdd55ac` | CPU/GPU 都直接计算中心归一化距离与两段三次式 | CPU FP64；GPU FP32 后拓宽为 FP64 存储 | 每点一线程；FP32 临时 + FP64 输出；第二个拓宽 kernel | `n=1,sym=false` 与 Python 不一致；测试无 periodic | Parzen 分段三次母窗、中心对称意图、周期有效长度 n+1 |

后续若只减少临时分配、合并 kernel 或调整 launch，且 $x_i$ 与两段公式不变，应标记“原理不变、实现优化”；若改变分段边界或采样坐标，则必须重新映射阶段一原理。
