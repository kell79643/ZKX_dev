# cusignal_cpp_spectrogram 复现逻辑

## 版本索引

| 版本 | Git 提交 | 提交时间 | 分支 | 状态 |
| --- | --- | --- | --- | --- |
| [V1：当前实现](#v1当前实现fdd55ac8) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 2026-08-15 23:45:58 +08:00 | `final-prep/benchmark-evidence-v1` | 当前学习版本 |

## V1：当前实现（fdd55ac8）

### 1. 版本身份与范围

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`；
- 提交标题：`test(archive): 增加任务结果范围治理审计`；
- 开始阶段三时 `ZKX` 工作树干净；当前算子相关文件没有未提交修改；
- 本文所有 C++ 定位都绑定上述 SHA，行号是该提交中的行号；
- 本阶段只读源码，没有修改 `ZKX/cusignal_cpp`，也没有运行本地、ZQ500 或远程测试。

当前算子所需文件：

| 职责 | 共享根目录 `ZKX_dev/` 下的仓库相对路径 | 主要符号 |
| --- | --- | --- |
| 参数、workspace、结果与接口声明 | `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.h` | `SpectrogramDeviceParams`、`SpectrogramDeviceWorkspace`、结果结构与 overload |
| CPU reference 与正式 GPU 返回包装 | `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cpp` | `spectrogram_typed_cpu`、`spectrogram_device` |
| GPU workspace 与调度 | `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_typed.cu` | `SpectrogramDeviceWorkspace::reset`、实值 `spectrogram_device` |
| GPU kernel | `ZKX/cusignal_cpp/src/spectral_analysis/spectral_analysis_kernels.cuh` | 帧准备、实值后处理、相位解缠 |
| 启动配置 | `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h` | `launch_1d_kernel` |
| FFT 抽象 | `ZKX/cusignal_cpp/src/fft_interface/fft_interface.h/.cpp` | `FFTInterface::fft_batch_device` |
| 类型与正确性 smoke | `ZKX/cusignal_cpp/test/signal_processing/e3_spectral_type_smoke.cu` | `run_type<T>` 中的 spectrogram 段 |

### 2. 参数、workspace 与结果类型

提交 `fdd55ac...`，`spectral_analysis_typed.h`，`SpectrogramDeviceParams`，第 145-157 行：

```cpp
struct SpectrogramDeviceParams {
    float fs = 1.0F;
    WindowParams window{"tukey", 0.25F};
    int nperseg = 0;
    int noverlap = -1;
    int nfft = 0;
    std::string detrend = "constant";
    bool return_onesided = true;
    std::string scaling = "density";
    int axis = -1;
    std::string mode = "psd";
    std::vector<int> shape;
};
```

这是 Python 签名的 C++ 参数对象：`0` 或 `-1` 表示采用默认值。`shape` 描述展平存储对应的逻辑维度；空 `shape` 表示一维。

同文件 `SpectrogramDeviceWorkspace`，第 159-172 行：

```cpp
struct SpectrogramDeviceWorkspace {
    StftDeviceWorkspace stft;
    float post_scale = 1.0F;

    SpectrogramDeviceWorkspace() = default;
    SpectrogramDeviceWorkspace(
        int input_size_in, const SpectrogramDeviceParams& params)
    {
        reset(input_size_in, params);
    }

    void reset(int input_size_in, const SpectrogramDeviceParams& params);
    [[nodiscard]] std::size_t output_size() const noexcept { return stft.output_size(); }
};
```

workspace 复用 STFT 的窗、FFT scratch、轴布局和 FFT plan，只额外保存后处理尺度。同一 workspace 不能并发共享。

同文件第 329-360 行定义 `SpectrogramOutputDtype`、CPU/GPU result。mode 为 `complex` 时写复容器，其余写实容器。INT32 非空结果为 FP64/ComplexFP64，其余四种输入为 FP32/ComplexFP32；任何输入类型的空结果固定为实 FP64 空数组。

### 3. 两组 API

提交 `fdd55ac...`，`spectral_analysis_typed.h:602-669` 声明：

1. convenience CPU/GPU：一维、periodic Hann、双边、`spectrum`、`psd`、无 detrend；
2. 正式 CPU/GPU：任意 rank/axis、单双边、两种 scaling、五种 mode，并返回 `(frequencies,times,Sxx)`。

convenience 入口只是简单 reference/smoke，不能代替完整接口。

### 4. CPU convenience：direct DFT

提交 `fdd55ac...`，`spectral_analysis_typed.cpp`，`spectrogram_typed_cpu(x,frame,hop)`，第 344-374 行：

```cpp
template <class T>
std::vector<float> spectrogram_typed_cpu(const std::vector<T>& x, int frame, int hop)
{
    if (frame < 2 || hop < 1 || hop > frame) {
        throw std::invalid_argument("spectrogram shape");
    }
    const int frames = x.size() < static_cast<std::size_t>(frame)
        ? 0
        : 1 + (static_cast<int>(x.size()) - frame) / hop;
    const float window_sum = host_hann_sum(frame);
    std::vector<float> out(frames * frame);
```

帧数为

$$
F=\begin{cases}0,&N_x<L,\\1+\left\lfloor\dfrac{N_x-L}{H}\right\rfloor,&N_x\ge L.\end{cases}
$$

第 355-373 行三重循环直接计算

$$
X[m,k]=\sum_{r=0}^{L-1}x[mH+r]w[r]e^{-j2\pi kr/L},
$$

并写入 `out[k*frames+m]=(real²+imag²)/(sum(w)²)`。复杂度 $O(FL^2)$，仅作独立 CPU reference。

### 5. CPU 正式入口

提交 `fdd55ac...`，`spectral_analysis_typed.cpp`，正式 `spectrogram_typed_cpu`，第 898-1042 行。

- 898-905：空输入返回 FP64 空结果；
- 906-940：axis plan 解析任意 rank/axis；默认 `nperseg` 为 256 或自定义窗长度，默认 overlap 为 `nperseg/8`，不扩边、不补尾；然后调用独立 `stft_typed_cpu`；
- 941-948：计算窗和、平方和及 STFT 的幅度尺度；
- 949-975：校验五种 mode，`complex` 直接写复结果并返回；
- 977-1009：PSD、magnitude、angle/phase 主值计算；
- 1010-1034：phase 沿频率轴解缠；
- 1035-1042：依据输入类型写 FP32 或 FP64 result。

尺度转换代码为：

```cpp
const double density_to_spectrum = sum * sum / (params.fs * square_sum);
const double complex_factor = params.scaling == "density"
    ? std::sqrt(density_to_spectrum)
    : params.scaling == "spectrum" ? 1.0
    : throw std::invalid_argument("unsupported scaling");
```

四种实值模式为：PSD $a^2+b^2$（单边时必要 bin 乘 2）、magnitude $\sqrt{a^2+b^2}$、angle `atan2(b,a)`、phase 在 angle 后解缠。CPU 还会把相对实部极小的舍入虚部归零，减少实轴附近相角抖动。

### 6. GPU workspace 与尺度分工

提交 `fdd55ac...`，`spectral_analysis_typed.cu`，`SpectrogramDeviceWorkspace::reset`，第 235-295 行。

它校验 mode 与正采样率，构造无边界、无补尾的 STFT workspace，并用 `CompensatedFloatAccumulator` 计算窗和及平方和：

| mode/scaling | `stft.scale` | `post_scale` |
| --- | ---: | ---: |
| PSD+density | 1 | $1/(f_s\sum w^2)$ |
| PSD+spectrum | 1 | $1/(\sum w)^2$ |
| 非 PSD+density | $\sqrt{1/(f_s\sum w^2)}$ | 1 |
| 非 PSD+spectrum | $1/\sum w$ | 1 |

这对应 Python 对复 STFT 使用功率尺度平方根的逻辑。

### 7. GPU wrapper 调用链

提交 `fdd55ac...`，`spectral_analysis_typed.cu`，实值 workspace overload `spectrogram_device`，第 476-523 行：

```text
参数/workspace/输出检查
→ stft_prepare_real_kernel
→ FFTInterface::fft_batch_device
→ spectrogram_real_post_kernel
→ phase 时再调用 spectrogram_phase_unwrap_kernel
```

`step=nperseg-noverlap`；batch 数为 `line_count*nframes`；mode 被映射为 PSD=0、magnitude=1、angle/phase=2。调用链没有隐式 host-device 数据搬运，也没有显式全设备同步；各操作走默认 stream，依赖同 stream 顺序，kernel 启动后执行错误检查。

### 8. 帧准备 kernel

提交 `fdd55ac...`，`spectral_analysis_kernels.cuh`，`stft_prepare_real_kernel`，第 592-642 行。

每线程负责一个 `[line,frame,local_fft_index]`。索引拆解：

```cpp
const int batch = output_index / fft_length;
const int line = batch / frame_count;
const int frame = batch % frame_count;
const int local_index = output_index % fft_length;
const int start = frame * step - boundary_left;
```

`local_index>=segment_length` 时写复零，实现 FFT 零填充。constant detrend 时每个有效线程顺序扫描整帧求均值，然后写

$$
((x[mH+r]-\bar{x}_m)w[r]\cdot scale)+j0.
$$

重复求均值是正确但可能昂贵的工程实现。

### 9. 实值后处理与 phase kernel

提交 `fdd55ac...`，`spectral_analysis_kernels.cuh`，`spectrogram_real_post_kernel`，第 435-483 行。

每线程负责一个 `[outer,frequency,inner,time]` 元素，并把 FFT scratch 的 `[line,time,frequency]` 重排到正式输出：

```cpp
if (mode == 0) {
    result = (real * real + imag * imag) * scale;
} else if (mode == 1) {
    result = sqrt(real * real + imag * imag);
} else {
    result = atan2(imag, real);
}
```

单边 PSD 中 DC 不翻倍；偶数 FFT 的 Nyquist 也不翻倍。

提交 `fdd55ac...`，同文件 `spectrogram_phase_unwrap_kernel`，第 485-523 行。每线程固定一条 `line×time`，在线程内部按 frequency 顺序循环；相邻差超过 $\pi$ 时累计减 $2\pi$，小于 $-\pi$ 时累计加 $2\pi$。频率链存在前后依赖，因此没有把同一链的频率点完全并行化。

### 10. grid、block、同步与 FFT

提交 `fdd55ac...`，`cuda_utils/kernel_launch.h:21-55`：默认 `block=256`，`grid=ceil(elements/256)`。

| kernel | elements | 每线程职责 | 线程内串行工作 |
| --- | ---: | --- | --- |
| 帧准备 | `line_count*nframes*nfft` | 一个 FFT 输入 | constant detrend 扫描 `nperseg` |
| 实值后处理 | `line_count*nf*nframes` | 一个谱图输出 | 常数工作 |
| phase unwrap | `line_count*nframes` | 一条频率链 | 扫描 `nf` |

FFT 经 `FFTInterface::fft_batch_device` 进入构建时选择的科学计算库后端；输入输出为 `ComplexFloat`，batch 为 `line_count*nframes`。

### 11. 正式 GPU 返回与 dtype finalization

提交 `fdd55ac...`，`spectral_analysis_typed.cpp`，正式 `spectrogram_device`，第 1220-1283 行：

1. 空输入返回实 FP64 空结果；
2. 解析默认帧参数并建 workspace；
3. FP64 frequency/time 坐标在 host 生成后复制到 device；
4. complex mode 调 `stft_device`；其他 mode 调实值 wrapper；
5. INT32 经 device storage finalization 形成 FP64/ComplexFP64，其余保留 FP32/ComplexFP32。

“设备端 FP32 计算”与“INT32 API 返回 FP64”不矛盾：后者是结果存储契约，不代表 FFT 以 FP64 计算。

### 12. template 与显式实例化

提交 `fdd55ac...`，`spectral_analysis_typed.cu:530-532` 和 `spectral_analysis_typed.cpp:1350-1390` 显式实例化 `float`、`__half`、`int32_t`、`int16_t`、`int8_t`。`static_assert(is_simple_signal_input_v<T>)` 编译期拒绝其他类型。模板改变输入加载与最终 dtype 分派；GPU FFT 主计算仍为 ComplexFP32。

### 13. Python—CPU—GPU 映射

| 原理 | Python | C++ CPU | C++ GPU | 关系 |
| --- | --- | --- | --- | --- |
| 默认窗/帧/overlap | `spectral.py:662-664,807-811` | `typed.cpp:908-935` | `typed.cpp:1232-1244` | 一致 |
| 分帧、去均值、窗乘 | `_fft_helper:1893-1908` | CPU STFT/direct DFT | prepare kernel | 等价实现 |
| DFT/FFT | CuPy FFT | direct DFT/CPU STFT | FFTInterface batch | 数学等价 |
| PSD/magnitude/angle | `spectral.py:813-848` | `typed.cpp:992-1005` | post kernel | 一致 |
| phase unwrap | `spectral.py:849-853` | `typed.cpp:1010-1034` | unwrap kernel | 沿频率轴一致 |
| 单边 PSD | `spectral.py:1842-1847` | `typed.cpp:983-994` | post kernel 466-473 | 端点规则一致 |
| dtype | CuPy result_type | INT32→FP64 | FP32 计算后 finalization | C++ 显式业务契约 |
| 坐标 | CuPy 生成 | host double | host 生成后传 device | 工程差异 |

### 14. 测试覆盖与风险

提交 `fdd55ac...`，`e3_spectral_type_smoke.cu:242-313,340-345,359-370` 的测试源码覆盖：

- convenience CPU/GPU 匹配与非法 hop；
- 正式 frequency/time 长度、dtype 和数值；
- rank-3、`axis=-2` 和输出 shape `{2,4,3,3}`；
- 五种 mode 与五种输入类型；
- 所有 mode 空输入均为实 FP64 空结果；
- accuracy evidence 记录。

本文未重新执行测试，只陈述测试源码覆盖。剩余风险包括：小规模 smoke 不能代替长窗和极端动态范围验证；低幅值相位仍不稳定；逐线程重复均值可能是长窗瓶颈；INT32 的 FP64 storage 不提高 FP32 FFT 精度；workspace 不可并发共享；坐标仍有 host→device 复制。

### 15. 阶段三自检问题与参考答案

#### 问题 1：V1 是哪个版本？

**答案 1：**`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，开始阶段三时相关工作树干净。

#### 问题 2：convenience 与正式接口差别？

**答案 2：**前者固定 Hann/双边/spectrum/PSD/一维；后者支持任意 rank/axis、完整参数、五 mode 并返回坐标。依据 `typed.h:602-669`。

#### 问题 3：CPU reference 是否调用 GPU？

**答案 3：**不调用。convenience CPU 在 `typed.cpp:344-374` direct DFT；正式 CPU 使用独立 host STFT 和 host 后处理。

#### 问题 4：GPU 主调用链？

**答案 4：**prepare kernel → FFTInterface batch → real-post kernel → phase 时 unwrap kernel，依据 `typed.cu:476-523`。

#### 问题 5：PSD density 在哪里落实？

**答案 5：**`typed.cu:267-284` 计算 `post_scale=1/(fs*sum(w²))`；`kernels.cuh:464-465` 计算 `(real²+imag²)*scale`。

#### 问题 6：phase 为什么每条频率链串行？

**答案 6：**当前 correction 依赖前一原始相位和累计修正；`kernels.cuh:507-522` 因而由一个线程顺序扫描。

#### 问题 7：三个 kernel 的职责？

**答案 7：**prepare 负责分帧/去均值/窗乘/补零；post 负责 PSD/幅度/相角和布局；unwrap 负责频率轴相位解缠。

#### 问题 8：INT32 返回 FP64 是否代表 FP64 FFT？

**答案 8：**不是。`typed.cpp:1275-1277` 只在计算完成后 finalization 为 FP64 storage；FFT scratch 是 ComplexFloat。

#### 问题 9：输出布局如何形成？

**答案 9：**axis plan 分为 outer/axis/inner；post kernel 写 `((outer*frequency+f)*inner+i)*time+t`，即原 axis 替换为频率并追加时间维。

#### 问题 10：单边翻倍端点规则？

**答案 10：**DC 不翻倍；偶数 nfft 的 Nyquist 不翻倍；其他正频率翻倍。CPU 见 `typed.cpp:983-994`，GPU 见 `kernels.cuh:466-473`。

#### 问题 11：workspace 的收益和限制？

**答案 11：**复用窗、scratch 和 FFT plan；但参数/shape 必须匹配，同一对象不可并发共享。

#### 问题 12：测试覆盖与缺口？

**答案 12：**覆盖五类型、五 mode、rank-3 非末轴、空输入、非法 hop 和 CPU/GPU匹配；未替代长窗、极端动态范围、并发 workspace 与大规模性能验证。

### 16. 阅读顺序

1. `typed.h:145-172,329-360,602-669`；
2. `typed.cpp:344-374`；
3. `typed.cpp:898-1042`；
4. `typed.cu:235-295`；
5. `kernels.cuh:592-642`；
6. `typed.cu:476-523`；
7. `kernels.cuh:435-523`；
8. `typed.cpp:1220-1283`；
9. `e3_spectral_type_smoke.cu:242-313`。

## 版本差异与原理不变量

当前只有 V1。未来追加 V2/V3 时保留本节行号与内容，并比较算法结构、复杂度、内存访问、并行方式、精度边界和性能。

数学不变量包括：滑窗局部化、逐帧 DFT/FFT、mode 与复 STFT 的派生关系、density/spectrum 归一化、实信号单边功率守恒和 phase 沿频率轴解缠。
