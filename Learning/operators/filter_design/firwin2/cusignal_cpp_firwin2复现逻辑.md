# cusignal_cpp_firwin2 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前实现](#v1当前实现fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前 | 首次正式学习版本；CPU 显式 IDFT，GPU FP32/dlfft IFFT，输出 FP64 存储 |

## V1：当前实现（fdd55ac）

### 1. 版本身份与可追溯性门禁

| 项目 | 记录 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 相关文件 dirty 状态 | `git status --short -- <相关文件>` 无输出，相关源码均为已提交版本 |
| 本阶段是否修改 `ZKX/cusignal_cpp` | 否；始终只读 |
| 本阶段是否构建或运行测试 | 否；按 Learning 规则只做静态源码学习 |

本版本所有代码证据均绑定上表完整 SHA。后文每个代码块仍重复给出 SHA、共享根目录 `ZKX_dev/` 下的相对路径、符号名和该提交中的准确行号，避免只靠当前工作区裸行号。

### 2. 阅读范围与分层

只读取 `firwin2` 直接相关内容：

| 层次 | 文件与符号 | 作用 |
| --- | --- | --- |
| API 声明 | `ZKX/cusignal_cpp/src/filter_design/filter_design_typed.h`：`Firwin2WindowMode`、`Firwin2Options`、`firwin2_typed_cpu`、`firwin2_device` | 对外类型与契约 |
| 公共参数准备 | `filter_design_typed.cpp`：`Firwin2Prepared`、`prepare_firwin2` | CPU/GPU 共用校验、类型选择、窗口准备 |
| CPU reference | `filter_design_typed.cpp`：`firwin2_typed_cpu` | Host 插值、频谱、显式 IDFT |
| GPU host wrapper | `filter_design_typed.cpp`：`firwin2_device` | D2H 校验、FP32 workspace、正式 FP64 输出 |
| GPU 调度 helper | `filter_design_typed.cu`：`firwin2_fp32_compute_device` | kernel → IFFT → kernel |
| GPU device 核心 | `filter_design_kernels.cuh`：三个 `firwin2_*` 符号 | 插值、相位、Hermitian 频谱、乘窗 |
| FFT 直接依赖 | `fft_interface.h/.cpp`：`FFTInterface::BatchOnly`、`ifft_batch_device` | ZQ500 `dlfft` IFFT 与 plan 池 |
| 输出拓宽依赖 | `host_output_finalize.h`、`fp64_storage_finalize.cu` | FP32 结果按 IEEE-754 位写成 FP64 存储 |
| 直接 smoke | `e3_estimation_filter_design_type_smoke.cu` | 五输入类型、窗口、四型和错误语义验证 |

注册表、合同表和文档索引不是核心计算，没有把它们误称为实现。`FFTInterface` 与 FP64 storage widening 只读取本次调用所需符号，没有扩展通读其余 FFT 或 CUDA utility。

### 3. 对外接口、选项与 dtype

#### 3.1 窗模式和选项结构

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_typed.h`；符号 `Firwin2WindowMode`、`Firwin2Options`；提交内第 155–163 行。

```cpp
enum class Firwin2WindowMode { hamming, none, explicit_values };

struct Firwin2Options {
    int nfreqs = 0;
    Firwin2WindowMode window_mode = Firwin2WindowMode::hamming;
    std::vector<float> window;
    bool antisymmetric = false;
    double fs = 2.0;
};
```

逐行说明：

- `enum class` 创建强类型枚举，三种值分别对应默认 Hamming、不加额外窗和调用者显式给窗；强类型枚举不会隐式转换为整数。
- `struct` 聚合全部可选设计参数。
- `nfreqs=0` 表示自动选择网格，而不是零点网格。
- 默认窗模式对齐 Python 的 `window="hamming"`。
- 显式窗系数固定用 `std::vector<float>` 保存，因此 GPU 窗运算是 FP32。
- `antisymmetric=false` 默认生成 I/II 型；设为真后生成 III/IV 型。
- `fs` 接口用 double 接收，但 `prepare_firwin2` 随后要求它可表示为 FP32。

与 Python 相比，C++ 没有接受任意窗口名称的 dispatcher；它支持 Hamming、None 和任意显式系数三种已批准契约。

#### 3.2 CPU 与 GPU 模板声明

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_typed.h`；符号 `firwin2_typed_cpu<T>`、`firwin2_device<T>`；提交内第 165–199 行。

```cpp
template <typename T>
std::vector<double> firwin2_typed_cpu(
    int numtaps, const std::vector<T>& freq, const std::vector<T>& gain,
    const Firwin2Options& options = {});

template <typename T>
void firwin2_device(
    int numtaps, const DeviceArray<T>& freq, const DeviceArray<T>& gain,
    DeviceArray<double>& out, const Firwin2Options& options = {});
```

逐行说明：

- `template <typename T>` 让同一接口接收五种业务输入类型。
- CPU 返回 `std::vector<double>`，shape 固定为 `[numtaps]`。
- `const std::vector<T>&` 避免复制 Host 输入并禁止修改调用者数据。
- `options={}` 使用上节结构的默认成员初始化。
- GPU 接收 `DeviceArray<T>`，调用者必须预分配 `DeviceArray<double> out`。
- GPU 函数返回 `void`，结果通过 `out` 写出。

显式实例化位于同一 SHA 的 `filter_design_typed.cpp:402-413`：`float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t` 五种输入都生成 CPU 和 GPU 符号；正式输出统一为 FP64。

### 4. CPU/GPU 共用参数准备

#### 4.1 准备结果的数据结构

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_typed.cpp`；符号 `Firwin2Prepared`；提交内第 104–111 行。

```cpp
struct Firwin2Prepared {
    int nfreqs{0};
    int filter_type{1};
    float nyquist{1.0F};
    std::vector<float> freq;
    std::vector<float> gain;
    std::vector<float> window;
};
```

这里把后续共同需要的所有元数据和数组正规化为 FP32。即使 CPU reference 后面用 `double` 做复数和 IDFT，控制点、Nyquist 与窗口已经先量化为 float；因此不能说 CPU 路径完整保留了 double 输入精度。

#### 4.2 shape、采样率与输入转换

定位：同一 SHA；`filter_design_typed.cpp`；符号 `prepare_firwin2<T>`；提交内第 113–134 行。

```cpp
template <typename T>
Firwin2Prepared prepare_firwin2(
    int n, const std::vector<T>& freq, const std::vector<T>& gain,
    const Firwin2Options& options)
{
    if (n <= 0 || freq.size() < 2 || freq.size() != gain.size() ||
        !std::isfinite(options.fs) || options.fs <= 0.0 ||
        options.fs > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::invalid_argument("firwin2 shape or fs");
    Firwin2Prepared prepared;
    prepared.nyquist = static_cast<float>(options.fs * 0.5);
    if (!(prepared.nyquist > 0.0F))
        throw std::invalid_argument("firwin2 fs is not representable in FP32");
    prepared.freq.resize(freq.size());
    prepared.gain.resize(gain.size());
    for (std::size_t index = 0; index < freq.size(); ++index) {
        prepared.freq[index] = static_cast<float>(load(freq[index]));
        prepared.gain[index] = static_cast<float>(load(gain[index]));
    }
    if (prepared.freq.front() != 0.0F ||
        prepared.freq.back() != prepared.nyquist)
        throw std::invalid_argument("firwin2 frequency endpoints");
```

执行顺序是：模板声明 → 组合 shape/fs 门禁 → 创建准备对象 → `fs/2` 转 FP32 → 再次防止下溢为非正 → 按输入长度分配两个向量 → `load` 统一读取 float/half/整数 → 转成 FP32 → 检查首尾端点。C++ 比 Python 多了正抽头、至少两个控制点、有限正 `fs` 和 FP32 可表示范围的显式检查。

#### 4.3 单调、重复点与四型约束

定位：同一 SHA；`filter_design_typed.cpp`；符号 `prepare_firwin2<T>`；提交内第 135–154 行。

```cpp
    std::vector<float> differences(freq.size() - 1);
    for (std::size_t index = 0; index + 1 < freq.size(); ++index) {
        differences[index] = prepared.freq[index + 1] - prepared.freq[index];
        if (differences[index] < 0.0F)
            throw std::invalid_argument("firwin2 frequencies nondecreasing");
    }
    for (std::size_t index = 0; index + 1 < differences.size(); ++index)
        if (differences[index] == 0.0F && differences[index + 1] == 0.0F)
            throw std::invalid_argument("firwin2 frequency repeated more than twice");
    if (prepared.freq[1] == 0.0F ||
        prepared.freq[prepared.freq.size() - 2] == prepared.nyquist)
        throw std::invalid_argument("firwin2 endpoint repeated");
    prepared.filter_type = options.antisymmetric
        ? ((n % 2 == 0) ? 4 : 3)
        : ((n % 2 == 0) ? 2 : 1);
    if ((prepared.filter_type == 2 && prepared.gain.back() != 0.0F) ||
        (prepared.filter_type == 3 &&
            (prepared.gain.front() != 0.0F || prepared.gain.back() != 0.0F)) ||
        (prepared.filter_type == 4 && prepared.gain.front() != 0.0F))
        throw std::invalid_argument("firwin2 filter type endpoint gain");
```

第一轮循环计算相邻差并立即拒绝降序；第二轮检测两个连续零差，即同一频率出现至少三次。随后拒绝端点重复，用嵌套条件运算符选择 I–IV 型，并把三类强制端点零值组合成一个布尔表达式。它和 Python 的数学约束相同，异常文本不要求逐字相同。

#### 4.4 默认网格与重复点扰动

定位：同一 SHA；`filter_design_typed.cpp`；符号 `prepare_firwin2<T>`；提交内第 155–179 行。

```cpp
    prepared.nfreqs = options.nfreqs;
    if (prepared.nfreqs == 0) {
        prepared.nfreqs = 1;
        while (prepared.nfreqs < n) {
            if (prepared.nfreqs > std::numeric_limits<int>::max() / 2)
                throw std::invalid_argument("firwin2 numtaps too large");
            prepared.nfreqs <<= 1;
        }
        ++prepared.nfreqs;
    }
    if (prepared.nfreqs <= n ||
        prepared.nfreqs > (std::numeric_limits<int>::max() / 2 + 1))
        throw std::invalid_argument("firwin2 nfreqs must exceed numtaps");
    for (std::size_t index = 0; index + 1 < prepared.freq.size(); ++index) {
        if (prepared.freq[index] == prepared.freq[index + 1]) {
            const float repeated = prepared.freq[index];
            prepared.freq[index] = std::nextafter(
                repeated, -std::numeric_limits<float>::infinity());
            prepared.freq[index + 1] = std::nextafter(
                repeated, std::numeric_limits<float>::infinity());
        }
    }
    for (std::size_t index = 1; index < prepared.freq.size(); ++index)
        if (!(prepared.freq[index] > prepared.freq[index - 1]))
            throw std::invalid_argument("firwin2 repeated frequency too close");
```

`<<=1` 每次把整数乘 2，直到得到不小于 `n` 的二次幂，随后 `++` 加一；溢出检查发生在移位前。显式或自动 `nfreqs` 都要求大于 `n`，并保证后续 `2*(nfreqs-1)` 可表示。重复点用 `nextafter` 移到最近的左右 FP32 数，而 Python 使用 `eps*nyq`；两者目标相同但不是逐位相同的扰动公式，所以关系是工程等价而非源码照抄。

#### 4.5 窗准备与返回

定位：同一 SHA；`filter_design_typed.cpp`；符号 `prepare_firwin2<T>`；提交内第 180–194 行。

```cpp
    prepared.window.resize(n, 1.0F);
    if (options.window_mode == Firwin2WindowMode::hamming) {
        for (int tap = 0; tap < n; ++tap)
            prepared.window[tap] = n == 1 ? 1.0F
                : 0.54F - 0.46F * std::cos(
                    2.0F * static_cast<float>(pi) * tap / (n - 1));
    } else if (options.window_mode == Firwin2WindowMode::explicit_values) {
        if (options.window.size() != static_cast<std::size_t>(n))
            throw std::invalid_argument("firwin2 explicit window length");
        prepared.window = options.window;
    } else if (options.window_mode != Firwin2WindowMode::none) {
        throw std::invalid_argument("firwin2 window mode");
    }
    return prepared;
}
```

先初始化全 1 窗，使 `none` 无需额外分支赋值。Hamming 使用对称分母 `n-1`，与 Python `fftbins=False` 的设计意图一致。显式窗必须与抽头数等长；未知枚举值被拒绝。最后按值返回准备对象，编译器可执行返回值优化或移动。

### 5. CPU reference 调用链与逐步算法

#### 5.1 频谱长度、插值和相位

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_typed.cpp`；符号 `firwin2_typed_cpu<T>`；提交内第 330–362 行。

```cpp
template <typename T>
std::vector<double> firwin2_typed_cpu(
    int n, const std::vector<T>& freq, const std::vector<T>& gain,
    const Firwin2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin2 dtype");
    const auto prepared = prepare_firwin2(n, freq, gain, options);
    const int total = 2 * (prepared.nfreqs - 1);
    const double spacing = prepared.nyquist / (prepared.nfreqs - 1);
    std::vector<std::complex<double>> spectrum(total);
    for (int index = 0; index < prepared.nfreqs; ++index) {
        const double frequency = index * spacing;
        std::size_t segment = 0;
        for (std::size_t candidate = prepared.freq.size() - 1; candidate-- > 0;) {
            if (frequency >= prepared.freq[candidate]) {
                segment = candidate;
                break;
            }
        }
        if (segment + 1 >= prepared.freq.size()) segment = prepared.freq.size() - 2;
        const double left = prepared.freq[segment];
        const double right = prepared.freq[segment + 1];
        const double ratio = std::fabs(right - left) > 1.0e-30
            ? (frequency - left) / (right - left)
            : 0.0;
        const double response = prepared.gain[segment] * (1.0 - ratio)
            + prepared.gain[segment + 1] * ratio;
        const double phase = -0.5 * (n - 1) * pi * frequency / prepared.nyquist;
        const std::complex<double> shift = prepared.filter_type > 2
            ? std::complex<double>(-std::sin(phase), std::cos(phase))
            : std::complex<double>(std::cos(phase), std::sin(phase));
        spectrum[index] = response * shift;
    }
```

`static_assert` 在编译期限制输入类型。`total=L=2(K-1)` 与 Python `irfft` 的隐含长度相同。外层循环只构造非负频率 $k=0,\ldots,K-1$；反向扫描控制点寻找分段，然后计算线性插值

$$
F_k=g_s(1-r)+g_{s+1}r.
$$

相位为 $-(n-1)\pi f/(2f_N)$。对称型构造 $(\cos\phi,\sin\phi)=e^{j\phi}$；反对称型构造 $(-\sin\phi,\cos\phi)=j e^{j\phi}$，和 Python `shift *= 1j` 等价。

#### 5.2 Hermitian 镜像和显式 IDFT

定位：同一 SHA；`filter_design_typed.cpp`；符号 `firwin2_typed_cpu<T>`；提交内第 363–380 行。

```cpp
    for (int index = 1; index < prepared.nfreqs - 1; ++index) {
        spectrum[total - index] = std::conj(spectrum[index]);
    }
    std::vector<double> out(n);
    for (int tap = 0; tap < n; ++tap) {
        std::complex<double> time_value(0.0, 0.0);
        for (int frequency = 0; frequency < total; ++frequency) {
            const double angle = 2.0 * pi * tap * frequency / total;
            time_value += spectrum[frequency]
                * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        time_value /= static_cast<double>(total);
        double value = time_value.real() * prepared.window[tap];
        if (prepared.filter_type == 3 && tap == n / 2) value = 0.0;
        out[tap] = value;
    }
    return out;
}
```

镜像循环跳过直流和 Nyquist，只给负频率写共轭值。然后对每个最终 tap 扫描全部 $L$ 个频率，直接计算

$$
g[t]=\frac1L\sum_{k=0}^{L-1}G[k]e^{j2\pi tk/L}.
$$

只计算前 `n` 个时域样本而不是完整 $L$ 点。取实部乘窗，III 型中心清零，再写入 FP64 输出。因此 CPU 时间复杂度是 $O(KP+ML)$，并非 FFT 的 $O(L\log L)$；它的职责是清晰、独立的 Host reference，而非高性能实现。

### 6. GPU host wrapper

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_typed.cpp`；符号 `firwin2_device<T>`；提交内第 382–400 行。

```cpp
template <typename T>
void firwin2_device(
    int n, const DeviceArray<T>& freq, const DeviceArray<T>& gain,
    DeviceArray<double>& out, const Firwin2Options& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin2 dtype");
    if (out.size() != static_cast<std::size_t>(n > 0 ? n : 0))
        throw std::invalid_argument("firwin2 output shape");
    const auto prepared = prepare_firwin2(
        n, freq.to_host(), gain.to_host(), options);
    auto device_freq = DeviceArray<float>::from_host(prepared.freq);
    auto device_gain = DeviceArray<float>::from_host(prepared.gain);
    auto device_window = DeviceArray<float>::from_host(prepared.window);
    DeviceArray<float> computed(static_cast<std::size_t>(n));
    firwin2_fp32_compute_device(
        n, device_freq, device_gain, device_window, computed,
        prepared.nfreqs, options.antisymmetric, prepared.nyquist);
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

调用链为：编译期 dtype 门禁 → 输出 shape 门禁 → `freq/gain.to_host()` 显式 D2H → 共用 Host 参数准备 → 三个 FP32 数组 H2D → 分配 FP32 结果 → GPU 计算 → device storage widening 得到 `DeviceArray<double>`。

这里存在一项必须保留的不一致：头文件 `filter_design_typed.h:189,192-194` 仍写“D2H、Host 拓宽、可选 H2D”，但本提交实际第 399 行调用 `finalize_fp64_device_storage`。该 helper 在 `host_output_finalize.h:40-46` 直接调用 device `widen_fp32_storage_device`；所以当前事实是**设备端位级拓宽，无结果 D2H/Host 算术/H2D**。这是声明注释陈旧，不是数学算法差异。

### 7. GPU 调度 helper

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_typed.cu`；符号 `firwin2_fp32_compute_device`；提交内第 100–132 行。

```cpp
void firwin2_fp32_compute_device(
    int n,
    const DeviceArray<float>& freq,
    const DeviceArray<float>& gain,
    const DeviceArray<float>& window,
    DeviceArray<float>& out,
    int nfreqs,
    bool antisymmetric,
    float nyquist)
{
    if (n <= 0 || freq.size() < 2 || freq.size() != gain.size() ||
        window.size() != static_cast<std::size_t>(n) ||
        out.size() != static_cast<std::size_t>(n) || nfreqs <= n ||
        nyquist <= 0.0F) {
        throw std::invalid_argument("firwin2 compute shape");
    }
    const std::size_t spectrum_size =
        static_cast<std::size_t>(2 * (nfreqs - 1));
    DeviceArray<ComplexFloat> spectrum(spectrum_size), time(spectrum_size);
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin2_build_spectrum_kernel<float,float>,
        spectrum_size, freq.data(), gain.data(), static_cast<int>(freq.size()),
        nfreqs, n, antisymmetric, nyquist, spectrum.data());
    FFTInterface fft(static_cast<int>(spectrum_size), FFTInterface::BatchOnly{});
    fft.ifft_batch_device(spectrum, time, 1);
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin2_apply_window_kernel<float,float>,
        out.size(), time.data(), window.data(), out.data(), n,
        antisymmetric && (n % 2 != 0));
}
```

shape 再校验后计算 $L=2(K-1)$，分配两个长度 $L$ 的 ComplexFP32 workspace。第一次 kernel 填完整频谱；`BatchOnly` FFT 对象不创建单 FFT plan，`ifft_batch_device(...,1)` 从 thread-local、按 `(n,batch,direction)` 键控的有界 plan 池取得逆变换计划；第二次 kernel 只处理前 `n` 个时域值并乘窗。函数不显式同步，kernel 与 FFT 均在默认 stream 顺序执行；调用者在读取或跨 stream 复用前负责同步。

### 8. GPU device 核心计算

#### 8.1 单个正频率样本

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/filter_design/filter_design_kernels.cuh`；符号 `firwin2_positive_spectrum<T,Value>`；提交内第 235–279 行。

该 `__device__` helper 每次负责一个非负频率索引。完整符号如下：

```cpp
template <class T, class Value>
__device__ ComplexFloat firwin2_positive_spectrum(
    int frequency_index,
    int frequency_count,
    const T* frequencies,
    const T* gains,
    int point_count,
    int numtaps,
    bool antisymmetric,
    Value nyquist)
{
    const Value pi = static_cast<Value>(3.14159265358979323846);
    const Value spacing = nyquist / static_cast<Value>(frequency_count - 1);
    const Value frequency = frequency_index * spacing;
    int segment = 0;
    for (int index = point_count - 2; index >= 0; --index) {
        if (frequency >= static_cast<Value>(load(frequencies[index]))) {
            segment = index;
            break;
        }
    }
    const Value left_frequency = static_cast<Value>(load(frequencies[segment]));
    const Value right_frequency = static_cast<Value>(load(frequencies[segment + 1]));
    const Value denominator = right_frequency - left_frequency;
    const Value ratio = fabs(denominator) > static_cast<Value>(1.0e-30)
        ? (frequency - left_frequency) / denominator
        : static_cast<Value>(0);
    const Value gain = static_cast<Value>(load(gains[segment]))
            * (static_cast<Value>(1) - ratio)
        + static_cast<Value>(load(gains[segment + 1])) * ratio;
    const int filter_type = antisymmetric
        ? ((numtaps % 2 == 0) ? 4 : 3)
        : ((numtaps % 2 == 0) ? 2 : 1);
    const Value phase = -static_cast<Value>(numtaps - 1)
        / static_cast<Value>(2) * pi * frequency / nyquist;
    const Value cosine = cos(phase);
    const Value sine = sin(phase);
    return filter_type > 2
        ? ComplexFloat{
            static_cast<float>(-gain * sine),
            static_cast<float>(gain * cosine)}
        : ComplexFloat{
            static_cast<float>(gain * cosine),
            static_cast<float>(gain * sine)};
}
```

按行连接起来就是：模板与设备函数签名声明参数类型；前三个常量得到 $\pi$、网格间距和当前频率；反向循环找到控制点区间；三行端点值和分母准备线性插值；条件表达式避免零分母；两项加权得到增益；嵌套条件表达式确定 I–IV 型；相位和正余弦建立复数响应；最后按对称或反对称形式返回实虚部。当前调用实例是 `<float,float>`，所以分段、插值、三角函数和返回值全部是 FP32。

#### 8.2 构造完整 Hermitian 频谱

定位：同一 SHA；`filter_design_kernels.cuh`；符号 `firwin2_build_spectrum_kernel<T,Value>`；提交内第 281–307 行。

```cpp
template <class T, class Value>
__global__ void firwin2_build_spectrum_kernel(
    const T* frequencies,
    const T* gains,
    int point_count,
    int frequency_count,
    int numtaps,
    bool antisymmetric,
    Value nyquist,
    ComplexFloat* spectrum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = 2 * (frequency_count - 1);
    if (index >= total) {
        return;
    }
    if (index < frequency_count) {
        spectrum[index] = firwin2_positive_spectrum(
            index, frequency_count, frequencies, gains, point_count,
            numtaps, antisymmetric, nyquist);
    } else {
        const ComplexFloat value = firwin2_positive_spectrum(
            total - index, frequency_count, frequencies, gains, point_count,
            numtaps, antisymmetric, nyquist);
        spectrum[index] = ComplexFloat{value.re, -value.im};
    }
}
```

线程全局索引为 `blockIdx.x*blockDim.x+threadIdx.x`。前 $K$ 个线程直接算直流到 Nyquist；其余线程把 `total-index` 映射回对应正频率并翻转虚部，形成共轭。越界线程直接返回。没有线程间共享数据或同步，每个线程只写 `spectrum[index]`。

#### 8.3 乘窗、截取和中心置零

定位：同一 SHA；`filter_design_kernels.cuh`；符号 `firwin2_apply_window_kernel<Window,Output>`；提交内第 309–328 行。

```cpp
const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
if (tap >= numtaps) return;
using Value = Accumulator<Output>;
Value result = static_cast<Value>(input[tap].re)
    * static_cast<Value>(load(window[tap]));
if (zero_center && tap == numtaps / 2) {
    result = static_cast<Value>(0);
}
output[tap] = store<Output>(result);
```

每个有效线程负责一个最终 tap，只读取 IFFT 实部和同位置窗值。当前实例 `<float,float>` 使累加与输出都是 FP32。`zero_center` 只在 `antisymmetric && odd` 时为真，恰好对应 III 型。最后通过类型策略 `store` 写出。

#### 8.4 grid、block 与线程范围

定位：同一 SHA；`ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h`；符号 `make_1d_launch_config`、`launch_1d_kernel`；提交内第 21–55 行。

默认 block 为 256 个线程，grid 为

$$
\left\lceil\frac{N}{256}\right\rceil.
$$

频谱 kernel 的 $N=L=2(K-1)$；乘窗 kernel 的 $N=M$。launcher 使用 `nullptr` stream，即默认 stream，并在启动后执行 `CUDA_KERNEL_CHECK()` 检查启动错误；kernel 本身不做显式同步。

### 9. FFT、拓宽、内存布局与数据搬运

#### 9.1 FFTInterface

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/fft_interface/fft_interface.cpp`；符号 `FFTInterface::FFTInterface(int,BatchOnly)`、`FFTInterface::ifft_batch_device`；提交内第 230–243、428–447 行。

`BatchOnly` 构造只保存长度并跳过单 FFT plan/缓冲创建。`ifft_batch_device` 验证 `batch>0` 和输入输出均为 `n*batch`，从 `thread_local BatchFFTPlanPool` 获取 inverse plan，调用 `fft_execute`；非成功状态抛 `runtime_error`。`firwin2` 的 batch 固定为 1，输入输出都是连续 `ComplexFloat[L]`。

#### 9.2 FP64 是输出存储契约，不是 GPU 数值精度

定位：同一 SHA；`ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`；符号 `finalize_fp64_device_storage`；提交内第 40–46 行；以及 `ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu`；符号 `widened_storage_bits`、`widen_real_storage_kernel`；提交内第 11–50、63–73 行。

GPU 最终 `computed` 是 `float[M]`。拓宽 kernel 读取每个 IEEE-754 binary32 的符号、指数和尾数，按整数位操作构造数值等价的 binary64 位模式，再写入 `DeviceArray<double>` 存储。它不重新计算滤波器，也不增加超过 FP32 已有的有效精度；其作用是满足正式输出 dtype 为 FP64 的接口契约，同时避免 ZQ500 device FP64 算术。

#### 9.3 主要数据搬运与 workspace

```text
调用者 DeviceArray<T> freq/gain
  └─ D2H：to_host，用于固定版本错误检查和参数准备
Host prepared：float freq/gain/window
  └─ H2D：三个 DeviceArray<float>
GPU：ComplexFloat spectrum[L] + time[L] + float computed[M]
  └─ device 位级拓宽
GPU：double out[M]
```

GPU 路径的临时空间为 $O(L+M)$；CPU 为 `complex<double> spectrum[L] + double out[M]`，同样是 $O(L+M)$。当前 API 每次调用重新分配这些 workspace，没有公开的 resident `firwin2` workspace。

### 10. 数学、Python、CPU 与 GPU 三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $K=1+2^{\lceil\log_2M\rceil}$ | `fir_filter_design.py:544-545` | `filter_design_typed.cpp:155-167` | 共用 Host prepare | C++ 用循环移位并做整数溢出门禁 |
| 重复频率表达跳变 | `:547-562` | `:168-179` | 共用 Host prepare 后 H2D | Python 用 `eps*nyq`；C++ 用 FP32 `nextafter` |
| I–IV 型 | `:522-542` | `:147-154` | kernel `:265-278` | 数学一致；GPU kernel 为独立局部计算类型号 |
| 分段线性插值 | `:564-569` | `:340-356` | `filter_design_kernels.cuh:246-264` | Python 用 `np.interp`；C++ 显式反向找分段 |
| $e^{-j(M-1)\omega/2}$ | `:571-577` | `:357-361` | `:268-278` | 反对称都等价乘 $j$ |
| Hermitian 负频 | `irfft` 隐式完成 | `typed.cpp:363-365` | spectrum kernel `:292-306` | C++ CPU/GPU 显式构造完整频谱 |
| IDFT/IFFT | `:579-580` | `typed.cpp:367-374` | `typed.cu:126-127` | CPU $O(ML)$ 显式 IDFT；GPU dlfft $O(L\log L)$ |
| 乘窗与截取 | `:586-594` | `typed.cpp:375-378` | window kernel `:317-327` | C++ 仅三类窗契约；GPU FP32 |
| III 型中心零 | `:596-597` | `typed.cpp:376` | window kernel `:324-326` | 完全对应 |
| 输出 dtype | Python 通常 float64 | `std::vector<double>` | FP32 计算后 FP64 storage bits | GPU FP64 不代表 FP64 计算精度 |

### 11. 相同、等价替换与有意不同

#### 相同

- 控制点、端点、非递减、单次重复、`nfreqs>M` 和 II–IV 型端点约束；
- 均匀网格、线性插值、线性相位、Hermitian 频谱、IDFT、对称窗和 III 型中心零；
- 默认 `fs=2`、默认 Hamming、默认对称。

#### 等价替换

- Python `irfft` 的隐式 Hermitian 补全被 C++ 显式写频谱替代；
- Python 复指数被 `sin/cos` 实部虚部直接构造替代；
- Python `eps` 扰动被 C++ `nextafter` 替代；
- CPU 直接 IDFT 与 GPU dlfft IFFT数学等价，但舍入顺序和精度不同。

#### 有意不同或扩展

- C++ 支持五种业务输入 dtype，全部输出 FP64；Python 接受普通 array-like，由 NumPy/CuPy dtype 传播。
- C++ GPU 正式数值路径固定 FP32/ComplexFP32，以适应 ZQ500，不执行 device FP64 算术。
- C++ `window` 不是任意名称 dispatcher，只提供 Hamming、None、显式数组。
- C++ 增加正 shape、有限正 `fs`、FP32 可表示性、整数溢出和预分配输出 shape 门禁。

### 12. 直接测试证据与未覆盖风险

#### 12.1 比较器与容差

定位：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu`；符号 `same<T>`；提交内第 22–26 行。

```cpp
template <typename T>
bool same(const std::vector<T>& a, const std::vector<T>& b)
{
    return test_evidence::f_observe_real(a, b, 2.0e-3);
}
```

GPU 输出与 CPU reference 通过项目观察器比较，显式容差为 `2e-3`。

#### 12.2 `firwin2` smoke 覆盖

定位：同一 SHA；同一测试文件；符号 `run_type<T>` 中的 `firwin2` 段；提交内第 438–562 行。

该段逐步覆盖：

1. 三点频率/增益、`fs=8`、`nfreqs=65` 的默认 Hamming CPU/GPU 对照；
2. `none` 窗；
3. 17 个值均为 0.5 的显式窗；
4. `nfreqs=0` 自动网格；
5. 17 tap III 型和 16 tap IV 型；
6. CPU/GPU 错误端点拒绝；
7. 内部重复频率 CPU/GPU 一致；
8. 三次重复拒绝；
9. II、III、IV 型非法端点增益分别拒绝；
10. `nfreqs==numtaps` 拒绝；
11. 将所有条件合并为 `firwin2_ok`，再记录 `OperatorId::firwin2` 证据。

定位：同一 SHA；同一文件；符号 `main`；提交内第 580–596 行。`main` 依次调用 `run_type<float>`、`run_type<__half>`、`run_type<int32_t>`、`run_type<int16_t>`、`run_type<int8_t>`，所以五种输入实例都进入同一 `firwin2` 测试段。

本阶段没有运行测试，因此这里只能表述为“源码存在这些验证”，不能声称本会话得到 PASS。

#### 12.3 仍未覆盖或需要警惕

- smoke 只用少量控制点和固定 17/16 tap，没有系统覆盖大 `nfreqs`、极窄跳变或多控制点性能；
- 没有在本段直接验证 NaN/Inf 控制点与 gain；`fs` 有显式 finite 门禁，但每个控制点没有独立 finite 检查；
- `nextafter` 与 Python `eps*nyq` 在极端相邻点上的拒绝边界可能不同；
- 头文件 GPU 拓宽注释已与实现不一致，后续应在独立开发任务中修正文档，Learning 阶段不能改活动源码；
- smoke 以 CPU C++ reference 为主要 oracle，不等同于逐例直接调用 Python cuSignal；合同记录提供固定版本对齐，但本会话未重新执行跨语言验收；
- 默认 stream 顺序足以保证本函数内部依赖，但调用者跨 stream 使用输出前必须显式协调。

### 13. 按“接口 → CPU → GPU wrapper → kernel → 测试”读到最后一行

1. **接口最后一行**：`filter_design_typed.h:199` 的 `options={}` 结束 GPU 声明；先确认输入驻留设备、输出预分配且类型固定 double。
2. **准备函数最后一行**：`filter_design_typed.cpp:193-194` 返回 `prepared` 并结束 helper；至此 CPU/GPU 的参数语义已经完全统一。
3. **CPU 最后一行**：`:379-380` 返回 `out` 并结束模板；在此之前每个 tap 已完成完整 IDFT、乘窗和中心修正。
4. **GPU wrapper 最后一行**：`:399-400` 完成 FP32 storage widening 后结束；没有结果 Host round-trip。
5. **GPU 调度最后一行**：`filter_design_typed.cu:131-132` 把 III 型布尔量交给乘窗 kernel 后结束；默认 stream 保证三段顺序。
6. **device 文件最后一个 `firwin2` 核心行**：`filter_design_kernels.cuh:327-328` 把 `result` 存入对应 tap 后闭合 kernel；每个 tap 只写一次。
7. **模板实例化最后一行**：`filter_design_typed.cpp:412-413` 实例化五种输入并取消宏定义，最终闭合 `cusignal` 命名空间。
8. **测试中 `firwin2` 段最后一行**：测试文件 `:570-571` 把 `firwin2_ok` 和输出长度记录为正式证据；`main:588` 完成第五种 `int8` 调用，随后汇总全部算子测试。

### 14. 自检问题与参考答案

#### 问题 1：V1 为什么可追溯？

**答案 1：**所有证据绑定完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，并记录提交时间、分支和相关文件 clean 状态；每个代码块同时给出 SHA、相对路径、符号和提交内行号。

#### 问题 2：CPU 与 GPU 的核心职责分别是什么？

**答案 2：**CPU `firwin2_typed_cpu` 是独立 reference，用 double complex 和嵌套循环显式计算前 $M$ 点 IDFT；GPU wrapper 负责 D2H 参数门禁和 workspace，device helper 负责频谱 kernel、dlfft IFFT、乘窗 kernel与输出拓宽。证据分别在 `filter_design_typed.cpp:330-400` 和 `filter_design_typed.cu:100-132`。

#### 问题 3：Python、CPU 和 GPU 怎样实现同一个线性相位公式？

**答案 3：**三者都实现 $G_k=F_ke^{-j(M-1)\omega_k/2}$；反对称型再乘 $j$。Python 是复指数，CPU 是 `(cos,sin)` 或 `(-sin,cos)`，GPU helper 同样直接构造实虚部，位置见映射表第 10 节。

#### 问题 4：模板和 dtype 契约是什么？

**答案 4：**输入显式实例化 float、half、int32、int16、int8；准备阶段统一转 FP32。CPU 最终返回 double，GPU 数学路径是 FP32/ComplexFP32，随后只按位拓宽成 FP64 存储。`static_assert` 和 `INST` 位于 `filter_design_typed.cpp:335,387,402-413`。

#### 问题 5：两个 kernel 的并行范围是什么？

**答案 5：**频谱 kernel 有 $L=2(K-1)$ 个逻辑元素，每线程写一个完整频谱 bin；乘窗 kernel 有 $M$ 个逻辑元素，每线程写一个 tap。默认 block 256，grid 分别为 $\lceil L/256\rceil$ 与 $\lceil M/256\rceil$。

#### 问题 6：本函数内部需要什么同步？

**答案 6：**两个 kernel 与 `ifft_batch_device` 都走默认 stream，依赖同一 stream 的顺序语义；源码没有显式 `cudaDeviceSynchronize`。launcher 检查启动错误，FFT 检查执行返回值。调用者读取或跨 stream 复用前负责同步。

#### 问题 7：FP64 输出是否意味着 GPU 做了 FP64 算术？

**答案 7：**否。GPU 先得到 `float[M]`，`widened_storage_bits` 用整数位操作构造等值 binary64 存储。它只改变表示格式，不恢复或增加 FP32 计算中已经丢失的精度。

#### 问题 8：测试覆盖了哪些边界，又缺少什么？

**答案 8：**覆盖五输入类型、三种窗模式、自动网格、III/IV 型、内部重复以及多类非法输入；未系统覆盖大规模、非有限控制点、极端近邻重复点和本会话跨 Python 的动态验收。源码证据为测试文件 `:438-562,580-596`。

## 版本差异与原理不变量

当前只有 V1，尚无 V2 可比较。后续优化版本必须追加而不能覆盖本节。无论实现将 CPU IDFT 改为 FFT、复用 workspace 或减少参数 D2H，只要接口仍是同一 `firwin2`，以下原理不变量都必须保留：控制点分段线性频响、I–IV 型约束、Hermitian 实序列频谱、群时延 $(M-1)/2$、有限长度乘窗以及 III 型中心零值。
