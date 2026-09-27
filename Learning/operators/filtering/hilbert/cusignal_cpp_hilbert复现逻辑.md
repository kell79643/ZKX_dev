# cusignal_cpp_hilbert 复现逻辑

## 版本索引

- 当前实现：[V1：原始学习版本（`fdd55ac8415`）](#v1原始学习版本fdd55ac8415)
- 当前没有 V2/V3；后续优化只能追加，不能覆盖 V1。

## V1：原始学习版本（fdd55ac8415）

### 1. 版本身份

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 提交说明：`test(archive): 增加任务结果范围治理审计`
- 相关源文件、FFT/finalize 直接依赖、typed smoke 与独立 probe 的 dirty 状态均为空，因此可形成正式 V1。
- 所有证据均通过 `git show fdd55ac8415d70379eb38a2c299047f90bcf0a41:<path>` 读取；本阶段没有修改、构建或运行 `ZKX/cusignal_cpp`。

### 2. 代码定位与职责

| 职责 | 完整 SHA + 共享根目录相对路径 + 符号 + 提交内行号 |
| --- | --- |
| 选项与结果 ABI | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` + `ZKX/cusignal_cpp/src/filtering/filtering_typed.h` + `HilbertOutputDtype`/`HilbertOptions`/`HilbertCpuResult`/`HilbertDeviceResult` + 60-80 |
| 布局计划 | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.h` + `HilbertLayoutPlan`/`prepare_hilbert_layout` + 315-328 |
| CPU/GPU 声明 | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.h` + `hilbert_device` 600-613；`hilbert_typed_cpu` 855-867 |
| 布局校验 | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` + `prepare_hilbert_layout` + 500-563 |
| CPU reference | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` + `hilbert_typed_cpu` + 1819-1885 |
| GPU wrapper | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` + `hilbert_device` + 1887-1904 |
| GPU 调度 | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cu` + `hilbert_fp32_compute_device` + 326-351 |
| GPU kernels | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_kernels.cuh` + 三个 `hilbert_*_kernel` + 1472-1540 |
| batched FFT | 同 SHA + `ZKX/cusignal_cpp/src/fft_interface/fft_interface.cpp` + `fft_batch_device`/`ifft_batch_device` + 385-447 |
| 整数结果拓宽 | 同 SHA + `ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu` + `widen_complex_storage_kernel`/`widen_complex_fp32_storage_device` + 52-59、75-85 |
| 类型实例化 | 同 SHA + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` + `INST` + 2781-2783；`.cu` + `INST` + 683-685 |
| 测试 | 同 SHA + `ZKX/cusignal_cpp/test/signal_processing/e3_filtering_type_smoke.cu` + hilbert 段 + 809-917；`ZKX/test_all/operators/cases/filtering/hilbert/operator_probe.cu` + `run<T>` + 46-57 |

### 3. 接口和布局逐段解释

提交内 `filtering_typed.h:60-80`：

```cpp
enum class HilbertOutputDtype { complex_fp32, complex_fp64 };
struct HilbertOptions {
    std::vector<int> shape;
    std::optional<int> fft_length;
    int axis = -1;
};
struct HilbertCpuResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};
struct HilbertDeviceResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};
```

第一行给 tagged-result 的两种标签。`shape` 是连续 row-major 输入的逻辑形状；空 shape 表示一维。`fft_length` 对应 Python `N`，`nullopt` 表示取轴长；`axis=-1` 表示末轴。CPU/GPU 结果各保留两个候选容器，调用方依据 `dtype` 读取其中一个；最后的 shape 是目标轴替换为 N 后的形状。

提交内 `filtering_typed.h:315-328`：

```cpp
struct HilbertLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int input_axis_length = 0;
    int fft_length = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    std::size_t output_size = 0;
};
HilbertLayoutPlan prepare_hilbert_layout(
    std::size_t input_size, const HilbertOptions& options);
```

`axis` 已正规化为非负值；输入轴长与 N 分开才能表达截断/补零。`outer_count` 是轴前维度乘积，`inner_count` 是轴后维度乘积，`line_count=outer*inner` 是独立一维变换条数，`output_size=line_count*N`。最后两行声明 CPU/GPU 共用的布局 helper。

完整 SHA 同上，`filtering_typed.cpp::prepare_hilbert_layout:500-563` 的执行顺序是：

1. `input_size` 超过 int 索引范围即抛异常；复制 options shape，空 shape 补为 `{input_size}`。
2. 循环拒绝负 extent、检查乘法溢出，并要求 shape 乘积等于实际元素数。
3. 负 axis 加 rank，随后检查上下界；读取原轴长。
4. `fft_length.has_value() ? *fft_length : input_axis_length` 对应 Python `N` 默认值；N 必须为正。
5. 两个循环分别计算 outer 和 inner，并逐步检查溢出。
6. 计算 line/output size，复制输入 shape，只把目标轴改成 N，返回 plan。

这些是形状、索引和内存安全工程逻辑，不是 Hilbert 数学核心。

### 4. CPU reference：从签名到最后一行

完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp`，`hilbert_typed_cpu`，1819-1885。

```cpp
template <typename T>
HilbertCpuResult hilbert_typed_cpu(
    const std::vector<T>& x, const HilbertOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert_layout(x.size(), options);
    HilbertCpuResult result;
    result.dtype = hilbert_output_dtype<T>();
    result.shape = plan.output_shape;
    if (result.dtype == HilbertOutputDtype::complex_fp32)
        result.complex_fp32.resize(plan.output_size);
    else
        result.complex_fp64.resize(plan.output_size);
```

模板只允许正式实数类型。共享 helper 先完成边界检查。`hilbert_output_dtype<T>()` 对整数选 C64、float/half 选 C32；随后只 resize 标签对应的输出容器。

```cpp
    struct HostComplex { double real; double imag; };
    std::vector<HostComplex> spectrum(plan.fft_length);
    for (int line = 0; line < plan.line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t input_base = static_cast<std::size_t>(outer)
            * plan.input_axis_length * plan.inner_count + inner;
        for (int frequency = 0; frequency < plan.fft_length; ++frequency) {
            double real = 0.0, imag = 0.0;
            for (int sample = 0; sample < plan.fft_length; ++sample) {
                const double value = sample < plan.input_axis_length
                    ? detrend_host_load(x[input_base
                        + static_cast<std::size_t>(sample) * plan.inner_count])
                    : 0.0;
                const double angle = -6.283185307179586476925286766559
                    * frequency * sample / plan.fft_length;
                real += value * std::cos(angle);
                imag += value * std::sin(angle);
            }
            const double mask = (frequency == 0 ||
                ((plan.fft_length % 2) == 0 && frequency == plan.fft_length / 2))
                ? 1.0 : (frequency < (plan.fft_length + 1) / 2 ? 2.0 : 0.0);
            spectrum[frequency] = HostComplex{real * mask, imag * mask};
        }
```

临时频谱使用 double，与 GPU FFT 独立。`line` 经商/余数还原 outer/inner；沿轴每前进一个 sample，row-major 索引增加 inner。三重循环直接计算 $e^{-j2\pi kn/N}$ DFT；超出原轴长的 sample 读 0，N 较小时自然截断。mask 精确实现 DC/Nyquist=1、正频=2、负频=0。

```cpp
        const std::size_t output_base = static_cast<std::size_t>(outer)
            * plan.fft_length * plan.inner_count + inner;
        for (int sample = 0; sample < plan.fft_length; ++sample) {
            double real = 0.0, imag = 0.0;
            for (int frequency = 0; frequency < plan.fft_length; ++frequency) {
                const double angle = 6.283185307179586476925286766559
                    * frequency * sample / plan.fft_length;
                real += spectrum[frequency].real * std::cos(angle)
                    - spectrum[frequency].imag * std::sin(angle);
                imag += spectrum[frequency].real * std::sin(angle)
                    + spectrum[frequency].imag * std::cos(angle);
            }
            real /= plan.fft_length;
            imag /= plan.fft_length;
            const std::size_t output_index = output_base
                + static_cast<std::size_t>(sample) * plan.inner_count;
            if (result.dtype == HilbertOutputDtype::complex_fp32)
                result.complex_fp32[output_index] = ComplexFloat{
                    static_cast<float>(real), static_cast<float>(imag)};
            else
                result.complex_fp64[output_index] = ComplexDouble{real, imag};
        }
    }
    return result;
}
```

正号角度实现 IDFT，两条累加式展开复乘法；除以 N 完成归一化。输出索引使用新轴长。C32 分支将 double 收窄，整数对应 C64 分支保留 double。最后依次关闭 sample/line 循环、返回并关闭函数。复杂度 $O(LN^2)$，临时频谱 $O(N)$，输出 $O(LN)$。

### 5. GPU wrapper、调度与 kernels

完整 SHA 同上，`filtering_typed.cpp::hilbert_device:1887-1904`：

```cpp
template <typename T>
void hilbert_device(const DeviceArray<T>& x, HilbertDeviceResult& output,
                    const HilbertOptions& options) {
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert_layout(x.size(), options);
    DeviceArray<ComplexFloat> computed(plan.output_size);
    hilbert_fp32_compute_device(x, computed, options);
    output = HilbertDeviceResult{};
    output.dtype = hilbert_output_dtype<T>();
    output.shape = plan.output_shape;
    if (output.dtype == HilbertOutputDtype::complex_fp32)
        output.complex_fp32 = std::move(computed);
    else
        output.complex_fp64 =
            cuda_utils::finalize_complex_fp64_device_storage(computed);
}
```

wrapper 统一分配 C32 计算输出并调用 GPU 实现，然后清空旧 result、设置 dtype/shape。float/half 直接 move；整数把 C32 结果拓宽成 C64 存储。这里不是 FP64 FFT。

接口注释 `filtering_typed.h:606-608` 写“D2H 后 Host 拓宽”，但当前实现实际进入同 SHA `fp64_storage_finalize.cu:52-59,75-85` 的 device `widen_complex_storage_kernel`，用位操作生成 binary64 存储；没有 D2H/Host cast/H2D。它不做 FP64 浮点算术，但确实是 device widening。此处以实现为准并保留不一致。

完整 SHA 同上，`filtering_typed.cu::hilbert_fp32_compute_device:326-351`：

```cpp
template <class T>
void hilbert_fp32_compute_device(
    const DeviceArray<T>& x, DeviceArray<ComplexFloat>& computed,
    const HilbertOptions& options) {
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_hilbert_layout(x.size(), options);
    if (computed.size() != plan.output_size)
        throw std::invalid_argument("hilbert compute output shape");
    if (plan.output_size == 0) return;
    DeviceArray<ComplexFloat> packed(plan.output_size);
    DeviceArray<ComplexFloat> spectrum(plan.output_size);
    cuda_utils::launch_1d_kernel(detail::hilbert_pack_axis_kernel<T>,
        packed.size(), x.data(), plan.input_axis_length, plan.fft_length,
        plan.inner_count, plan.line_count, packed.data());
    FFTInterface fft(plan.fft_length, FFTInterface::BatchOnly{});
    fft.fft_batch_device(packed, spectrum, plan.line_count);
    cuda_utils::launch_1d_kernel(detail::hilbert_mask_kernel,
        spectrum.size(), spectrum.data(), plan.fft_length,
        static_cast<int>(spectrum.size()));
    fft.ifft_batch_device(spectrum, packed, plan.line_count);
    cuda_utils::launch_1d_kernel(detail::hilbert_unpack_axis_kernel,
        computed.size(), packed.data(), plan.fft_length, plan.inner_count,
        plan.line_count, computed.data());
}
```

顺序是类型约束→layout→容量检查→空输出早退→两个 workspace→pack→batched FFT→mask→batched IFFT→unpack。`FFTInterface` 在同 SHA `fft_interface.cpp:385-447` 检查 `batch>0` 和 `size=N*batch`，从 thread-local plan pool 取计划并调用 `fft_execute`，失败抛 `runtime_error`。算子没有显式全局同步，同 stream 顺序维持依赖。

三个 kernel 位于完整 SHA 同上、`filtering_kernels.cuh:1472-1540`：

- **pack**：线程线性索引 `blockIdx.x*blockDim.x+threadIdx.x` 对应一个 `(line,sample)`；默认 `value=0`，有效 sample 才按 outer/inner row-major 索引 gather，并通过 `SimpleSignalTypePolicy<T>::load` 转为 float，写 `ComplexFloat{value,0}`。因此同时完成 axis 搬运、类型转换、截断/补零。
- **mask**：每线程负责一个 `(line,frequency)`；`frequency=index%N`。DC 直接返回。偶数 N：`frequency<half` 乘 2，`>half` 置零，`==half` 保留；奇数 N：`<=floor(N/2)` 乘 2，其余置零。
- **unpack**：每线程负责一个 `(line,sample)`，用 outer/inner 公式 scatter 回目标轴长度为 N 的 row-major 输出。没有归约、共享内存或原子操作。

grid/block 由 `launch_1d_kernel` 统一决定，本算子没有硬编码 block 大小。`packed`、`spectrum`、`computed` 都是 $L\times N$ 个 C32，另有 FFT plan/workspace；空间量级 $O(LN)$，FFT 主复杂度 $O(LN\log N)$。

### 6. 模板、类型和数据移动

完整 SHA + `filtering_typed.cpp:2781-2783` 和 `.cu:683-685` 显式实例化 `float`、`__half`、`int32_t`、`int16_t`、`int8_t`。GPU 计算始终 C32；float/half 输出 C32，整数输出接口为 C64，但数值精度仍来自 C32 FFT。输入是调用者已有的连续 `DeviceArray<T>`；算子内部无输入 H2D，probe 才负责 H2D/D2H。

### 7. Python—CPU—GPU 三方映射

| 原理 | Python | C++ CPU | C++ GPU | 差异 |
| --- | --- | --- | --- | --- |
| N/axis | 基准 `filtering.py:877-882` | SHA + `.cpp:500-563` | 共用 layout | C++ 多 shape/溢出门禁 |
| 截断/补零 | `cp.fft.fft(x,N,axis)` | SHA + `.cpp:1843-1847` | SHA + kernels `:1486-1495` | C++ 显式 gather |
| DFT | CuPy FFT | SHA + `.cpp:1840-1852` double direct DFT | SHA + `.cu:342-343` batched C32 FFT | 数学等价，复杂度/精度不同 |
| mask | Python `_hilbert_kernel` | SHA + `.cpp:1853-1858` | SHA + kernels `:1498-1520` | GPU 原地修改 spectrum |
| IDFT/IFFT | `cp.fft.ifft` | SHA + `.cpp:1862-1874` | SHA + `.cu:347` | CPU 显式除 N |
| axis | Python广播 h | outer/inner line循环 | pack/batch/unpack | C++ 不创建广播 mask |
| dtype | CuPy后端决定 | double计算后按契约存储 | C32计算，整数仅拓宽存储 | C++固定五种业务类型 |

### 8. 测试证据与风险

完整 SHA + `e3_filtering_type_smoke.cu:809-917` 覆盖默认一维、二维 axis=0/N=3、三维 axis=1、截断、N=0/非法 axis 的 CPU/GPU 异常、空输入显式 N=4、长度 4 冲激已知结果、五种 T 的 dtype/shape 和 GPU—CPU reference 一致。

完整 SHA + `test_all/.../hilbert/operator_probe.cu:46-57` 构造二维 typed 输入，分别测量 CPU/GPU，显式同步/D2H，比较误差，并落盘 digest、dtype、shape、MSE/RMSE/相对误差和资源阶段。它是测试包装，不是核心实现。

本阶段没有运行测试。仍需警惕极大 N/FFT workspace、非连续 view、更多 backend 失败、极端动态范围、随机奇偶 N，以及接口注释与 device widening 实现不一致。

### 9. 自检问题与参考答案

1. **V1 身份？** 完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，相关文件 clean。
2. **CPU/GPU 职责？** CPU 是 double direct DFT/IDFT 独立 reference；GPU 是 pack+batched C32 FFT+mask+IFFT+unpack。
3. **outer/inner/line？** outer 为轴前维度乘积，inner 为轴后维度乘积，line=outer×inner；每条 line 做一次长度 N 变换。
4. **Nyquist？** 偶数 N 的 CPU mask 返回 1；GPU 分支既不乘 2 也不置零，等价保留 1。
5. **模板类型？** `float`、`__half`、`int32_t`、`int16_t`、`int8_t`，证据是同 SHA `.cpp:2781-2783`、`.cu:683-685`。
6. **整数是 FP64 计算吗？** 不是；FFT/mask/IFFT 为 C32，只把结果拓宽成 C64 存储。
7. **线程职责？** pack/unpack 每线程一个 `(line,sample)`；mask 每线程一个 `(line,frequency)`。
8. **同步？** 算子无显式全局同步；同 stream 排序，probe 在测量边界同步。
9. **代码/注释冲突？** 头文件称 Host 拓宽，实际为 device 位级 widening；以实现为准。
10. **测试边界？** smoke 覆盖 axis/N/shape/dtype/异常/冲激/CPU-GPU 一致，但不能证明所有规模、backend 和动态范围。

### 10. 阅读顺序

按 `Options/Result → LayoutPlan → prepare_hilbert_layout → hilbert_typed_cpu → hilbert_device → hilbert_fp32_compute_device → pack → mask → unpack → FFTInterface → typed smoke` 阅读。先手算 shape `{2,4}`、axis 0、N=3 的 outer/inner/line，再手算 N=4、N=5 mask。

## 版本差异与原理不变量

当前只有 V1。数学不变量是：实输入沿指定轴截断/补零到 N，DC/Nyquist 保持、严格正频加倍、负频置零，再做逆变换得到解析信号。CPU/GPU 的差别属于实现、复杂度和精度路径差别。
