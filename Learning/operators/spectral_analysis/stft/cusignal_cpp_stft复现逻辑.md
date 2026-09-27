# cusignal_cpp_stft 复现逻辑

## 版本索引

| 版本 | Git 提交 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| [V1](#v1原始学习版本fdd55ac8415d70379eb38a2c299047f90bcf0a41) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU direct DFT reference；GPU 预处理 kernel、batched `FFTInterface` 和输出 pack kernel |

## V1：原始学习版本（fdd55ac8415d70379eb38a2c299047f90bcf0a41）

### 1. 版本身份与阅读边界

- 完整 Git SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`；
- 分支：`final-prep/benchmark-evidence-v1`；
- 提交时间：`2026-08-15T23:45:58+08:00`；
- 提交主题：`test(archive): 增加任务结果范围治理审计`；
- 当前算子相关文件 dirty 状态：下列文件执行 `git status --short -- <paths>` 无输出，即均与该提交一致；
- 本阶段没有修改 `ZKX/cusignal_cpp`，也没有构建、运行或连接 ZQ500；测试结论只引用该提交已有源码和文档证据，不冒充本轮实测。

本阶段只读取：

1. `cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.h`：参数、workspace、返回结构和公开接口；
2. `cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`：CPU convenience、正式 CPU reference、正式 GPU 返回包装和显式实例化；
3. `cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cu`：窗口、workspace、GPU wrapper 和 launch；
4. `cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh`：STFT device helper 与两个 kernel；
5. `cusignal_cpp/src/fft_interface/fft_interface.h/.cpp`：STFT 直接调用的 batched FFT 接口；
6. `cusignal_cpp/test/signal_processing/e3_spectral_type_smoke.cu`：STFT 的直接类型、shape、空输入和 CPU/GPU 对照入口。

以下代码定位均绑定同一完整 SHA，不代表未来工作区的行号。

### 2. 对外参数、返回类型与声明

#### 2.1 `StftDeviceParams`

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，路径 `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.h`，符号 `StftDeviceParams`，该提交第 30-42 行：

```cpp
struct StftDeviceParams {
    float fs = 1.0F;
    WindowParams window;
    int nperseg = 256;
    int noverlap = -1;
    int nfft = 0;
    std::string detrend;
    bool return_onesided = true;
    std::string boundary = "zeros";
    bool padded = true;
    int axis = -1;
    std::vector<int> shape;
};
```

它把 Python `stft` 的主要参数面搬到 C++。两个哨兵值需要特别注意：`noverlap=-1` 表示稍后取 `nperseg/2`，`nfft=0` 表示稍后取 `nperseg`。空 `shape` 表示一维；非空 `shape` 配合 `axis` 描述 row-major 多维输入。

#### 2.2 workspace 的数据与容量

同一提交、同一路径，符号 `StftDeviceWorkspace`，第 83-143 行。其核心成员为：

```cpp
struct StftDeviceWorkspace {
    DeviceArray<float> window;
    DeviceArray<ComplexFloat> fft_input;
    DeviceArray<ComplexFloat> fft_output;
    std::unique_ptr<FFTInterface> fft;
    ComplexFloat* fft_input_ptr = nullptr;
    ComplexFloat* fft_output_ptr = nullptr;
    std::size_t fft_input_capacity = 0;
    std::size_t fft_output_capacity = 0;
    int input_size = 0;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    DeviceArray<int> line_bases;
    int axis = 0;
    int axis_length = 0;
    int axis_stride = 1;
    int inner_count = 1;
    int outer_count = 1;
    int line_count = 1;
    int nperseg = 0;
    int noverlap = 0;
    int nfft = 0;
    int nframes = 0;
    int nf = 0;
    bool return_onesided = true;
    bool complex_input = false;
    std::string boundary = "zeros";
    bool padded = true;
    std::string detrend = "constant";
    float scale = 1.0F;
```

`window` 保存 FP32 窗；`fft_input/fft_output` 是 `[line_count,nframes,nfft]` 的 ComplexFP32 scratch；`line_bases` 把多维 row-major 数组拆成沿目标轴的一组逻辑一维线。`axis_stride` 让 kernel 不必先转置整个输入。

容量函数位于第 133-142 行：

```cpp
[[nodiscard]] std::size_t fft_scratch_size() const noexcept
{
    return static_cast<std::size_t>(line_count)
        * static_cast<std::size_t>(nframes) * static_cast<std::size_t>(nfft);
}
[[nodiscard]] std::size_t output_size() const noexcept
{
    return static_cast<std::size_t>(line_count)
        * static_cast<std::size_t>(nf) * static_cast<std::size_t>(nframes);
}
```

因此 scratch 保存完整 `nfft` 个复频点，而正式输出在单边模式下只保存 `nf=nfft/2+1` 个频点。

#### 2.3 动态 dtype 三元组返回

同一提交、同一路径，符号 `SpectralComplexDtype`、`SpectralComplexCpuResult`、`SpectralComplexDeviceResult`，第 303-327 行：

```cpp
enum class SpectralComplexDtype { complex_fp32, complex_fp64, fp64 };

struct SpectralComplexCpuResult {
    SpectralComplexDtype dtype{SpectralComplexDtype::complex_fp32};
    std::vector<double> frequencies;
    std::vector<double> times;
    std::vector<double> fp64;
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    int frequency_count{0};
    int time_count{0};
    std::vector<int> shape;
};

struct SpectralComplexDeviceResult {
    SpectralComplexDtype dtype{SpectralComplexDtype::complex_fp32};
    DeviceArray<double> frequencies;
    DeviceArray<double> times;
    DeviceArray<double> fp64;
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    int frequency_count{0};
    int time_count{0};
    std::vector<int> shape;
};
```

不能只查看某个 vector 是否存在来猜 dtype，必须先读枚举：空输入用 `fp64` 实数空数组；INT32 非空结果使用 `complex_fp64` storage；其余四种业务输入使用 `complex_fp32`。

#### 2.4 四类入口

同一提交、同一路径，符号 `stft_typed_cpu`、`stft_device`，第 456-519 行：

```cpp
template <class T> std::vector<ComplexFloat> stft_typed_cpu(
    const std::vector<T>& x, int frame, int hop);

template <class T> SpectralComplexCpuResult stft_typed_cpu(
    const std::vector<T>& x, const StftDeviceParams& params);

template <class T> void stft_device(
    const DeviceArray<T>& x, int frame, int hop,
    DeviceArray<ComplexFloat>& out);

template <class T> void stft_device(
    const DeviceArray<T>& x, DeviceArray<ComplexFloat>& out,
    StftDeviceWorkspace& workspace, const StftDeviceParams& params);

template <class T> SpectralComplexDeviceResult stft_device(
    const DeviceArray<T>& x, const StftDeviceParams& params);
```

- `frame/hop` CPU 与 GPU 是固定 periodic-Hann、双边、无 boundary/padding/detrend 的 convenience 子集；
- 带 workspace 的 `void stft_device` 是可复用、无隐式整段传输的核心 GPU 调度层；
- 返回 `SpectralComplex*Result` 的 CPU/GPU overload 才是完整三元组正式入口。

### 3. CPU 调用链与逐步算法

#### 3.1 convenience CPU direct DFT

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，路径 `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `stft_typed_cpu(const std::vector<T>&,int,int)`，第 279-308 行：

```cpp
template <class T>
std::vector<ComplexFloat> stft_typed_cpu(const std::vector<T>& x, int frame, int hop)
{
    if (frame < 2 || hop < 1 || hop > frame) {
        throw std::invalid_argument("stft shape");
    }
    const int frames = x.size() < static_cast<std::size_t>(frame)
        ? 0
        : 1 + (static_cast<int>(x.size()) - frame) / hop;
    const float window_sum = host_hann_sum(frame);
    std::vector<ComplexFloat> out(frames * frame);
```

非法窗长/帧移立即拒绝；不做 padding，因此短输入返回零帧。输出布局固定为 frequency-major `[frame,frames]`。

```cpp
    for (int frequency = 0; frequency < frame; ++frequency) {
        for (int frame_index = 0; frame_index < frames; ++frame_index) {
            float real = 0.0F;
            float imag = 0.0F;
            for (int sample = 0; sample < frame; ++sample) {
                const int input_index = frame_index * hop + sample;
                if (input_index < static_cast<int>(x.size())) {
                    const float angle = -2.0F * kPi * frequency * sample / frame;
                    const float value = host_load(x[input_index])
                        * host_periodic_hann(sample, frame) / window_sum;
                    real += value * std::cos(angle);
                    imag += value * std::sin(angle);
                }
            }
            out[frequency * frames + frame_index] = {real, imag};
        }
    }
    return out;
}
```

三重循环直接实现

$$
Z[k,m]=\sum_{r=0}^{L-1}\frac{x[mH+r]w[r]}{\sum_qw[q]}
\left(\cos\frac{-2\pi kr}{L}+j\sin\frac{-2\pi kr}{L}\right).
$$

它不是 FFT，复杂度为 $O(TL^2)$，只作独立 reference。索引 `frequency * frames + frame_index` 对应 frequency-major。

#### 3.2 正式 CPU 参数归一化与 shape

同一提交、同一路径，符号 `stft_typed_cpu(const std::vector<T>&,const StftDeviceParams&)`，第 726-785 行。

```cpp
SpectralComplexCpuResult result;
if (x.empty()) {
    result.dtype = SpectralComplexDtype::fp64;
    return result;
}
result.dtype = spectral_requires_fp64_v<T>
    ? SpectralComplexDtype::complex_fp64
    : SpectralComplexDtype::complex_fp32;
```

空输入在正常 dtype 推导前早退，刻意复现 Python 无 dtype `cp.empty` 的 FP64 特例。`spectral_requires_fp64_v<T>` 对 INT32 为真。

```cpp
const auto input_plan = spectral_layout::prepare_axis_plan(
    x.size(), params.shape, params.axis, 1, 0, false);
const int input_axis_length = input_plan.axis_length;
int nperseg = params.nperseg <= 0
    ? (params.window.custom_window.empty()
        ? 256 : static_cast<int>(params.window.custom_window.size()))
    : params.nperseg;
```

`prepare_axis_plan` 先验证元素数、shape 和 axis，并得到轴长。自定义窗在 `nperseg<=0` 时决定窗长，否则默认 256。

```cpp
if (!params.window.custom_window.empty()
    && (params.window.custom_window.size()
            > static_cast<std::size_t>(input_axis_length)
        || static_cast<int>(params.window.custom_window.size()) != nperseg))
    throw std::invalid_argument("window is longer than input or differs from nperseg");
if (params.window.custom_window.empty() && nperseg > input_axis_length) {
    nperseg = input_axis_length;
}
const int noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
const int nfft = params.nfft <= 0 ? nperseg : params.nfft;
```

字符串窗遇到短输入会缩短，数组窗则必须严格匹配；这与 Python `_triage_segments` 的关键区别和对应关系一致。

```cpp
const int step = nperseg - noverlap;
const bool extended = !params.boundary.empty() && params.boundary != "None";
int effective_size = input_axis_length + (extended ? nperseg : 0);
if (params.padded) {
    const int remainder = (effective_size - nperseg) % step;
    if (remainder != 0) effective_size += step - remainder;
}
const int frames = effective_size < nperseg
    ? 0 : 1 + (effective_size - nperseg) / step;
const int nf = params.return_onesided ? nfft / 2 + 1 : nfft;
```

非空 boundary 在左右各补半窗，总长度增加一个 `nperseg`；尾部 padding 再补到整数帧。单边只改变 `nf`，不改变内部 DFT 长度。

#### 3.3 正式 CPU direct DFT 主循环

同一符号，第 786-817 行：

```cpp
for (int outer = 0; outer < layout.outer_count; ++outer) {
  for (int frequency = 0; frequency < nf; ++frequency) {
   for (int inner = 0; inner < layout.inner_count; ++inner) {
    const int line = outer * layout.inner_count + inner;
    const int line_base = layout.line_bases[line];
    for (int frame = 0; frame < frames; ++frame) {
        const int start = frame * step - (extended ? nperseg / 2 : 0);
```

`outer/inner` 分解 axis 前后的维度，`line_base` 指向该逻辑一维线的第一个元素；频率轴替换原 axis，时间轴最后追加。

```cpp
        double mean = 0.0;
        if (params.detrend == "constant") {
            for (int sample = 0; sample < nperseg; ++sample)
                mean += host_axis_boundary_sample(
                    x, line_base, layout.axis_stride,
                    layout.axis_length, start + sample, params.boundary);
            mean /= nperseg;
        }
```

`constant` detrend 先计算该帧均值；空字符串使 `mean` 保持 0。边界 helper 在取样时即时完成 zeros/constant/even/odd 延拓，没有先物化完整扩展数组。

```cpp
        double real = 0.0, imag = 0.0;
        for (int sample = 0; sample < nperseg; ++sample) {
            const double value = (host_axis_boundary_sample(
                x, line_base, layout.axis_stride, layout.axis_length,
                start + sample, params.boundary) - mean)
                * window[sample] / window_sum;
            const double angle = -2.0 * pi * frequency * sample / nfft;
            real += value * std::cos(angle);
            imag += value * std::sin(angle);
        }
        append_spectral_complex<T>(result, real, imag);
```

CPU 正式 reference 内部用 `double` 累加，再按模板类型把结果追加到 ComplexFP32 或 ComplexFP64 storage。它落实去均值、乘窗、窗和归一化和 $N$ 点 DFT；当 `nfft>nperseg` 时，未显式循环的尾部等价于补零。

### 4. GPU workspace 初始化

#### 4.1 窗、默认尺寸和 boundary 编码

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，路径 `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cu`：

- `make_stft_window`，第 34-106 行：生成 custom、Hamming、periodic Hann、Kaiser、boxcar、Blackman、Tukey 和 Bartlett/triang FP32 窗；其他类型抛异常；
- `normalize_stft_sizes`，第 108-120 行：落实 `nperseg`、半窗 overlap 和 `nfft=nperseg` 默认值；
- `boundary_mode`，第 128-135 行：`zeros/None/空→0`、`constant→1`、`even→2`、`odd→3`；
- `stft_frame_count`，第 137-151 行：计算 boundary 和 padded 后的帧数。

periodic Hann 使用

$$
w[r]=\frac12\left(1-\cos\frac{2\pi r}{L}\right),\quad 0\le r<L,
$$

因为代码令 `periodic_size=L+1`、`denominator=L`。

#### 4.2 `StftDeviceWorkspace::reset`

同一提交、同一路径，符号 `StftDeviceWorkspace::reset`，第 155-211 行。

初始化依次完成：

1. 拒绝负输入、未知 boundary、非 `constant`/空 detrend，以及复输入的单边请求；
2. `prepare_axis_plan` 计算 axis、stride、line count 和 line bases，并把 line bases 复制到 Device；
3. 归一化 `nperseg/noverlap/nfft`，验证 `nfft>=nperseg` 与 overlap；
4. 计算 `nframes`、`nf` 和完整输出 shape；
5. 在 Host 生成 FP32 窗，用补偿累加器求窗和并设置 `scale=1/window_sum`；
6. 分配两个大小为 `line_count*nframes*nfft` 的 ComplexFP32 scratch；
7. 有帧时创建 `FFTInterface(nfft,BatchOnly{})` plan。

窗口和接近零时当前 GPU workspace 退回 `scale=1`，而不是除零。这是防御性工程分支，与 Python 直接除以窗和的行为并非严格相同。

`bind_fft_scratch`（第 213-225 行）允许调用者复用外部 scratch；容量不足立即拒绝。`clear_fft_scratch_bindings`（第 227-233 行）恢复到 workspace 自有数组。它们只管理内存，不改变 STFT 数学公式。

### 5. GPU kernel 与每线程职责

#### 5.1 输入类型加载与边界取样

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，路径 `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh`，符号 `stft_load`、`stft_axis_sample_at`，第 536-590 行。

`stft_load` 通过 `SimpleSignalTypePolicy<T>::load` 把 FP32、FP16、INT32、INT16 或 INT8 输入提升到 `Accumulator=float`。`stft_axis_sample_at` 用

```cpp
input[line_base + source_index * axis_stride]
```

读取任意 axis；越界时按 mode 返回零、边缘常值、偶对称反射，或 `2*edge-reflected` 的奇对称延拓。

#### 5.2 预处理 kernel

同一提交、同一路径，符号 `stft_prepare_real_kernel`，第 592-642 行：

```cpp
const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
const int total = line_count * frame_count * fft_length;
if (output_index >= total) return;
const int batch = output_index / fft_length;
const int line = batch / frame_count;
const int frame = batch % frame_count;
const int local_index = output_index % fft_length;
const int start = frame * step - boundary_left;
```

一维 grid 中**每个线程负责一个 `[line,frame,local_index]` scratch 元素**。`batch=line*frame_count+frame` 与后续 batched FFT 完全一致。

```cpp
if (local_index >= segment_length) {
    fft_input[output_index] = ComplexFloat{0.0F, 0.0F};
    return;
}
```

当 `nfft>nperseg`，尾部线程直接写复零，从而实现 FFT 零填充。

```cpp
Accumulator mean = static_cast<Accumulator>(0);
if (detrend_constant) {
    compute_policy::CompensatedFloatAccumulator mean_acc;
    for (int index = 0; index < segment_length; ++index) {
        mean_acc.add(stft_axis_sample_at<Accumulator>(...));
    }
    mean = mean_acc.value() / static_cast<Accumulator>(segment_length);
}
```

开启 constant detrend 时，每个非补零线程都会重新遍历整帧求均值。它避免单独均值 kernel 和临时数组，但工作量增至 $O(TL^2)$，是 detrend 模式的潜在性能瓶颈；补偿累加降低 FP32 求和误差。

```cpp
const Accumulator value = (
    stft_axis_sample_at<Accumulator>(...) - mean)
    * stft_load<Accumulator>(window[local_index]) * scale;
fft_input[output_index] = ComplexFloat{static_cast<float>(value), 0.0F};
```

该线程完成边界取样、去均值、乘窗和 $1/\sum w$ 标度，最后把实数装成虚部为零的 ComplexFP32 FFT 输入。

#### 5.3 batched FFT

提交相同 SHA，路径 `ZKX/cusignal_cpp/src/fft_interface/fft_interface.cpp`，符号 `FFTInterface::fft_batch_device(const ComplexFloat*,...)`，第 406-426 行：

```cpp
if (batch <= 0) throw std::invalid_argument(...);
const std::size_t expected = static_cast<std::size_t>(n_)
    * static_cast<std::size_t>(batch);
if (count != expected) throw std::invalid_argument(...);
static thread_local BatchFFTPlanPool cache;
const FFTPlanHandle plan = cache.get(n_, batch, FFT_FORWARD);
FFTResult result = fft_execute(plan, output, input);
if (result != FFT_SUCCESS) throw std::runtime_error(...);
```

它验证 `count=nfft*batch`，从线程局部 plan pool 取得正向计划并执行构建时选择的 FFT 后端。STFT wrapper 传入 `batch=line_count*nframes`。这里的 FFT 是科学库/项目统一接口，不是自写 STFT kernel 内的 DFT 循环。

#### 5.4 输出 pack kernel

同一提交，路径 `spectral_analysis_kernels.cuh`，符号 `stft_pack_output_kernel`，第 651-677 行。

每个线程负责一个正式输出 `[line,frequency,frame]` 元素。它从 `output_index` 解出 frame、frequency 和 line，再把 line 分成 outer/inner：

```cpp
const int destination =
    ((outer * frequency_count + frequency) * inner_count + inner)
        * frame_count + frame;
```

这实现“原 axis 替换为 frequency，time 追加到最后”的 row-major layout。读取源为：

```cpp
fft_output[(line * frame_count + frame) * fft_length + frequency]
```

单边模式只是 pack 前 `frequency_count=nfft/2+1`，FFT scratch 仍是完整 Complex-to-Complex 双边结果。

### 6. GPU host wrapper、launch 与同步

#### 6.1 workspace 核心 wrapper

提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，路径 `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cu`，符号 `stft_device(DeviceArray<T>,DeviceArray<ComplexFloat>,StftDeviceWorkspace,StftDeviceParams)`，该提交第 471 行。源码被格式化成一个物理长行，其语义顺序为：

1. `static_assert` 限定五种简单实输入；
2. 重新推导默认 `pn/po/pf`，检查 workspace、参数和输出大小完全一致；
3. 空输入直接返回；有帧却无 FFT plan 则抛异常；
4. 计算 `step` 和左侧半窗 `boundary_left`；
5. 用 `launch_1d_kernel` 启动 `stft_prepare_real_kernel`，grid 元素数为 scratch size；
6. 调用 `fft_batch_device`，batch 为 `line_count*nframes`；
7. 用 `launch_1d_kernel` 启动 `stft_pack_output_kernel`，grid 元素数为正式输出大小。

代码没有显式 `cudaDeviceSynchronize`。三个操作在默认 stream 上按提交顺序排队；wrapper 也不做 D2H。调用者必须在 Host 读取结果前通过上层传输/同步语义等待完成。错误分成 launch/FFT 接口错误与参数异常；FFT 非成功码由 `FFTInterface` 转成 `std::runtime_error`。

#### 6.2 convenience wrapper

同文件符号 `stft_device(DeviceArray<T>,int,int,DeviceArray<ComplexFloat>)`，第 472 行：它验证 frame/hop，构造 periodic Hann、双边、无 boundary/padding/detrend、axis=-1 参数，临时创建 workspace，再调用核心 overload。反复调用会重复准备 workspace/plan，因此长期路径应复用 workspace。

#### 6.3 正式三元组 wrapper

提交相同 SHA，路径 `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp`，符号 `stft_device(const DeviceArray<T>&,const StftDeviceParams&)`，第 1175-1217 行。

它先处理空输入 FP64 特例和短输入窗长，再创建 workspace、分配 ComplexFP32 `computed` 并调用核心 GPU wrapper。随后：

```cpp
result.frequencies = DeviceArray<double>::from_host(
    spectral_frequencies(workspace.nfft, workspace.nf, params.fs));
result.times = DeviceArray<double>::from_host(spectral_times(
    workspace.nframes, workspace.nperseg, workspace.noverlap,
    params.fs, workspace.boundary));
set_spectral_complex_output<T>(computed, result);
```

频率与时间先在 Host 生成 double vector，再复制成 Device FP64 coordinates。`set_spectral_complex_output` 对 INT32 调用 `finalize_complex_fp64_device_storage` 形成 ComplexFP64 storage，其余类型移动 ComplexFP32 结果。主 STFT FFT 数值路径始终为 ComplexFP32。

### 7. template、类型分发、显式实例化与数据搬运

- `T` 覆盖 FP32、FP16、INT32、INT16、INT8；`SimpleSignalTypePolicy<T>::load` 在预处理 kernel 内提升为 float。
- `stft_prepare_real_kernel<T,float,float>` 的 `Input=T`、`Window=float`、`Accumulator=float`；设备 DFT 输入固定 `ComplexFloat`。
- 输出 dtype 契约：INT32 非空为 ComplexFP64 storage，其他四型 ComplexFP32，空输入实数 FP64 空结果。
- `spectral_analysis_typed.cpp` 末尾宏为五型显式实例化 CPU 正式入口和正式 GPU 返回入口；`spectral_analysis_typed.cu:530` 的宏显式实例化两个 GPU 核心 overload。
- 主要 H2D：workspace 构造时上传窗和 `line_bases`；正式 wrapper 上传 Host 生成的 frequency/time 坐标。
- 主输入、FFT scratch、正式谱输出始终驻留 Device；核心 wrapper 没有整段输入/输出隐式传输。
- CPU reference 使用 Host vector；GPU 正式入口只在元数据/坐标和窗上进行小规模 Host 参与。

### 8. 数学—Python—CPU—GPU 映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $H=L-noverlap$ | Python 基准 `spectral.py:1744-1750` | SHA `fdd55...` `spectral_analysis_typed.cpp:754-760` | `spectral_analysis_typed.cu:181-190,471` | 等价 |
| boundary 后再 padded | Python `:1752-1772` | `.cpp:761-770` 与 boundary sample helper | `.cu:137-150`、kernel `:543-590` | CPU/GPU 不物化完整扩展数组，按索引即时取样 |
| constant detrend | Python `:1774-1794,1904-1905` | `.cpp:794-801` | kernel `:625-635` | GPU 每个非补零线程重算帧均值 |
| $x_m[r]w[r]/\sum w$ | Python `:1799-1807,1907-1908` | `.cpp:803-807` | workspace `.cu:196-203` + kernel `:636-641` | GPU 窗和近零时退回 scale=1 |
| $N$ 点 DFT | Python `:1910-1916` 调 CuPy FFT | `.cpp:803-811` direct DFT | 预处理 → `FFTInterface::fft_batch_device` | CPU reference $O(TLN)$；GPU 科学库 batched FFT |
| 单边实信号 | Python `:1809-1829` 的 rFFT | CPU 只计算 `nf` 个频点 | GPU 做完整 C2C FFT 后只 pack `nf` | 数学输出等价，GPU 内部工作不等同 rFFT |
| axis 替换、time 末尾追加 | Python `:1861-1867` | layout `.cpp:777-783` | axis plan + pack kernel `:666-676` | 等价 row-major 输出布局 |
| FP64 坐标 | Python CuPy `fftfreq/arange` 默认精度 | `std::vector<double>` | Host double 生成后 H2D | 坐标不是 FFT kernel 计算 |

### 9. 与 Python 实现的相同、等价替换和有意差异

相同或等价：默认 Hann/256/半重叠/单边/zeros boundary/padded；边界扩展在 padding 前；窗和幅度归一化；时间标签按窗中心；输出频率轴替换原 axis、时间轴追加。

有意工程替换：

- Python 用 CuPy stride view；GPU C++ 用预处理 kernel 直接写连续 batched FFT scratch；
- Python 实输入单边走 `rfft`；C++ GPU 统一 C2C FFT 后裁剪正频率；
- Python FFT dtype 由 CuPy 推导；C++ 主设备计算固定 ComplexFP32，再按契约形成动态 storage；
- CPU reference 使用 direct DFT，目的在独立正确性对照，不追求 FFT 性能；
- C++ 只支持明确列出的窗口类型与 `constant`/空 detrend，未知 callable detrend 明确拒绝；
- GPU 窗和接近零时 scale 退回 1，与 Python 可能产生除零的行为不同。

### 10. ZQ500 平台限制与证据边界

根据 `ZKX/docs/development/ZQ500_RUNTIME.md`：

- ZQ500 正式路径不能依赖 FP64 device 计算；本实现主窗、预处理和 FFT 均为 FP32/ComplexFP32，FP64 主要作为坐标与最终 storage 契约；
- FFT 点数受平台 2 的整数次幂限制；上层若接受其他 `nfft`，必须由平台适配/补零策略保证后端可执行，不能从通用 C++ 参数检查推断任意长度在 ZQ500 都可运行；
- 构建必须显式二选一 `USE_DLFFT` 或 `USE_THRUST`，STFT 通过 `FFTInterface` 使用构建时选择的后端；
- DLFFT 与项目 Thrust FFT 都是正式可选后端，但长期同进程资源稳定性结论不同；本阶段不做性能评价；
- 当前提交文档把五类型 STFT 标为 ZQ500 verified，但本轮没有重新运行，因此只能引用既有证据，不能把它写成本轮测试结果。

### 11. 测试如何验证及未覆盖风险

提交相同 SHA，路径 `ZKX/cusignal_cpp/test/signal_processing/e3_spectral_type_smoke.cu`：

- 第 177 行验证非法 hop 的 CPU/GPU 异常、convenience CPU/GPU ComplexFloat 对照、正式 CPU/GPU 对照、动态 dtype、坐标长度与输出 size；
- 第 178-186 行验证五类型空输入都返回 `fp64` 且所有数组为空；
- 第 198-205 行验证 rank-3、`axis=-2` 的 shape `{2,4,3,3}` 及 CPU/GPU 数值一致；
- 第 324-328 行通过第二组输入触发精度证据采样，避免只验证一个数据模式；
- 第 362 行按输入类型记录 STFT 输出精度类型。

仍需保留的风险：窗口和接近零的 Python/C++ 差异；constant detrend 的重复均值计算；GPU 单边仍做完整 C2C FFT；平台非 2 次幂 FFT 限制；边界极短轴反射索引；workspace 不可并发共享；本轮未重新验证当前硬件、两个 FFT 后端和长期内存稳定性。

### 12. 阶段三自检问题与参考答案

1. **V1 绑定哪个版本，为什么可以形成正式章节？**

   **答案：**完整 SHA 是 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`，提交时间 `2026-08-15T23:45:58+08:00`。本节列出的七个 STFT 相关文件 `git status --short -- <paths>` 均无输出，因此代码与提交快照一致，可以用该 SHA 和提交内行号形成可追溯 V1。

2. **CPU convenience 入口是不是 FFT 实现？复杂度是什么？**

   **答案：**不是。`.cpp:290-305` 明确三重循环遍历 frequency、frame、sample，并用 `cos/sin` 累加 direct DFT。若每帧长度与频点数均为 $L$、帧数为 $T$，复杂度为 $O(TL^2)$。

3. **GPU 每个预处理线程负责什么？**

   **答案：**kernel `spectral_analysis_kernels.cuh:610-619` 把线性 `output_index` 解为 `line/frame/local_index`，所以一线程负责一个 `[line,frame,FFT输入位置]`。若位置超出 `nperseg`，第 620-623 行写零；否则第 625-641 行完成去趋势、边界取样、乘窗、标度和 ComplexFloat 写入。

4. **GPU 每个 pack 线程负责什么？**

   **答案：**`:661-676` 中一线程负责一个 `[line,frequency,frame]` 正式输出，计算 row-major destination，并从完整 FFT scratch 复制该频点。它同时落实单边裁剪和 axis/time 输出布局。

5. **FFT 的 batch 数和 scratch 大小如何推导？**

   **答案：**batch=`line_count*nframes`；每个 batch 长 `nfft`。因此 `fft_scratch_size=line_count*nframes*nfft`，见头文件第 133-136 行；wrapper 也以同一 batch 调 `fft_batch_device`。

6. **为什么说 CPU 与 GPU 职责没有混写？**

   **答案：**CPU `.cpp:727-817` 独立读取 Host vector 并 direct DFT；GPU `.cu:471` 调两个 device kernel 和 `FFTInterface`，主数据驻留 Device。正式 GPU wrapper 仅在 Host 生成少量坐标/窗/布局元数据，不把 CPU direct DFT 当作 GPU 核心计算。

7. **五种输入如何映射到计算 dtype 和输出 dtype？**

   **答案：**GPU 预处理统一 `Accumulator=float`、FFT 输入输出统一 ComplexFloat。FP32、FP16、INT16、INT8 非空返回 ComplexFP32；INT32 按 `spectral_requires_fp64_v<T>` 形成 ComplexFP64 storage；空输入在推导前早退为实数 FP64 空结果。证据为头文件 `303-327`、kernel `592-641` 和 `.cpp:709-721,1176-1216`。

8. **GPU 是否直接使用 rFFT 实现单边输出？**

   **答案：**不是。workspace scratch 长度始终包含完整 `nfft`，调用的是 ComplexFloat C2C `fft_batch_device`；单边时 `nf=nfft/2+1`，pack kernel 只复制前 `nf` 个频点。它与实信号 rFFT 输出数学等价，但内部计算量不同。

9. **代码在哪里处理同步和错误？**

   **答案：**STFT wrapper 没有显式全局同步；默认 stream 上预处理、FFT、pack 按序提交，调用者在读取前负责同步。参数/workspace 不一致抛 `invalid_argument`；`FFTInterface::fft_batch_device` 在 `.cpp:406-426` 检查 batch/count，后端返回非成功码时抛 `runtime_error`。

10. **当前实现相对 Python 有哪些明确差异？**

    **答案：**C++ GPU 用 kernel 物化连续 scratch 而非 stride view；单边仍做完整 C2C FFT；仅支持列举窗口与 constant/空 detrend；窗和近零时 scale=1；设备主计算固定 FP32/ComplexFP32。这些是工程替换或收窄，不能写成 Python 源码逐句相同。

11. **现有测试覆盖哪些边界，又缺什么？**

    **答案：**直接测试覆盖非法 hop、五类型 CPU/GPU 对照、动态 dtype、空输入、rank-3/负 axis、坐标与 shape。没有在本轮重新覆盖两个 FFT 后端、非 2 次幂平台限制、近零窗和、长期 workspace 并发/内存稳定性，因此这些仍是风险边界。

12. **ZQ500 上为什么不能直接声称任意 `nfft` 都可运行？**

    **答案：**通用参数检查只要求 `nfft>=nperseg`，但运行环境文档明确指出 ZQ500 FFT 点数限于 2 的整数次幂。不满足时需要平台适配或补零；没有实际容器构建运行证据时，不能用 C++ 接口的宽参数面覆盖硬件后端限制。

### 13. V1 阅读顺序

按以下顺序逐句读代码：

1. `spectral_analysis_typed.h:30-42,83-143,303-327,456-519`：参数、workspace、结果和接口；
2. `spectral_analysis_typed.cpp:279-308`：最小 CPU reference；
3. `spectral_analysis_typed.cpp:726-817`：完整 CPU 参数面和 direct DFT；
4. `spectral_analysis_typed.cu:34-151,155-233`：窗口、帧数和 workspace；
5. `spectral_analysis_kernels.cuh:536-677`：边界 helper、预处理和 pack；
6. `fft_interface.cpp:406-426`：batched FFT；
7. `spectral_analysis_typed.cu:471-472`：GPU wrapper 串联；
8. `spectral_analysis_typed.cpp:1175-1217`：正式三元组返回；
9. `e3_spectral_type_smoke.cu:177-205,324-328,362`：测试如何核对。

## 版本差异与原理不变量

当前只有 V1，尚无可比较的优化版本。后续版本必须追加，不能覆盖上述 SHA 和行号。无论工程实现如何优化，以下数学不变量都必须保持：移动窗局部化、帧移 $H=L-noverlap$、逐帧 DFT、实信号单边共轭冗余、窗和幅度标度、频率轴替换原 axis 且时间轴追加。
