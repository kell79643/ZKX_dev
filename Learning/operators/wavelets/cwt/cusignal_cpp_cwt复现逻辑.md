# cusignal_cpp_cwt 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：原始学习版本](#v1原始学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | callable wavelet bank + CPU reference + 一线程一输出的 GPU 直接卷积 |

## V1：原始学习版本（fdd55ac）

### 1. 版本身份与 dirty 门禁

| 项目 | 记录 |
| --- | --- |
| 完整 SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| 相关文件 dirty 状态 | clean；`git status --short -- <相关文件>` 无输出 |
| 阅读方式 | 所有证据均通过 `git show/git grep fdd55ac...` 读取提交快照 |
| 执行状态 | 未构建、运行或连接 ZQ500；只说明测试源码覆盖，不声称本会话实际 PASS |

相关文件：`wavelets_typed.h/.cpp/.cu`、`wavelets_kernels.cuh`、`host_output_finalize.h`、`fp64_storage_finalize.cu`、`e3_wave_window_type_smoke.cu` 和 `operator_contracts.cmake`。

### 2. 分层职责与调用链

```text
Host wavelet callable
  → prepare_cwt_workspace
      → cwt_lengths：int(width)，L=min(10*width,N)
      → 逐 width 调 callable，检查长度
      → 缩窄并打包为 ComplexFloat bank → H2D
  → cwt_device
      → cwt_complex_fp32_compute_device
          → cwt_convolution_kernel：每线程一个 [width,time] 输出
      → real 路径提取实部
      → FP32 位级拓宽为 FP64/ComplexFP64 存储
```

CPU reference 独立执行 `cwt_typed_cpu → cwt_lengths → callable → 三重循环 same 卷积`，不创建 device、stream 或 workspace。

### 3. 接口、类型与 workspace 逐行解释

#### callable 类型与 workspace 数据结构

证据：SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`；`CwtRealWaveletCallable`、`CwtComplexWaveletCallable`、`CwtDeviceWorkspace`；第 98–117 行。

```cpp
using CwtRealWaveletCallable =
    std::function<std::vector<double>(int points, int width)>;
using CwtComplexWaveletCallable =
    std::function<std::vector<ComplexDouble>(int points, int width)>;

/** @brief 由任意C++ callable显式构造的CWT device wavelet bank。 */
struct CwtDeviceWorkspace {
    int data_count{0};
    int width_count{0};
    int max_wavelet_length{0};
    bool complex_output{false};
    DeviceArray<int> lengths;
    DeviceArray<ComplexFloat> wavelets;

    std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(data_count) *
            static_cast<std::size_t>(width_count);
    }
};
```

- 两组 `using` 分别规定 `(points,width)→vector<double>` 和 `(points,width)→vector<ComplexDouble>`，对应 Python 二参数 callable。
- `struct` 的前三个整数记录 $N$、$W$ 和 bank 行跨度；布尔值区分正式实/复输出；两个 `DeviceArray` 保存每行真实长度及统一 ComplexFP32 bank。
- `output_size()` 不修改对象且不抛异常。先转 `size_t` 再相乘，返回逻辑 `[W,N]` 的 $WN$ 个元素。

#### 对外声明

证据：同 SHA；同文件；`prepare_cwt_workspace`、`cwt_typed_cpu`、`cwt_device`；第 124–179 行。

```cpp
template<class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet);
template<class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet);

template<class T>
std::vector<double> cwt_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet);
template<class T>
std::vector<ComplexDouble> cwt_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet);

template<class T>
void cwt_device(
    const DeviceArray<T>& x, const CwtDeviceWorkspace& workspace,
    DeviceArray<double>& out);
template<class T>
void cwt_device(
    const DeviceArray<T>& x, const CwtDeviceWorkspace& workspace,
    DeviceArray<ComplexDouble>& out);
```

- 三对模板重载分别由 callable 类型或 `out` 类型区分 real/complex。
- `const &` 避免复制输入、widths、callable 与 workspace；非 const `out&` 允许替换正式 device storage。
- CPU 返回 FP64/ComplexFP64；GPU接口也呈现该存储类型，但 device 核心算术保持 FP32。

### 4. 公共长度 helper

证据：同 SHA；`ZKX_dev/ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`；`load`、`cwt_lengths`；第 11–33 行。

```cpp
template <typename T>
float load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <class T>
std::vector<int> cwt_lengths(
    int data_count, const std::vector<T>& widths)
{
    if (data_count < 0)
        throw std::invalid_argument("prepare_cwt_workspace: data_count must be nonnegative");
    if (widths.empty()) return {};
    std::vector<int> lengths(widths.size());
    for (std::size_t index = 0; index < widths.size(); ++index) {
        const int width = static_cast<int>(load(widths[index]));
        const int length = std::min(10 * width, data_count);
        if (length <= 0)
            throw std::invalid_argument("prepare_cwt_workspace: every width must produce a positive wavelet length");
        lengths[index] = length;
    }
    return lengths;
}
```

- `load` 通过统一策略把五类型标量变成 float；`static_cast<int>` 随后向零截断。
- helper 拒绝负 $N$，空 widths 返回空数组；循环计算 $L_j=\min(10\operatorname{int}(w_j),N)$ 并拒绝 $L_j\le0$。
- C++ 因而比 Python 多了明确的非正宽度/长度门禁。

### 5. workspace 准备逐行解释

#### 实 callable bank

证据：同 SHA；同 `.cpp`；实重载 `prepare_cwt_workspace`；第 143–168 行。

```cpp
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("prepare_cwt_workspace: real wavelet callable is empty");
    const auto lengths_host = cwt_lengths(data_count, widths);
    std::vector<ComplexFloat> bank(
        static_cast<std::size_t>(data_count) * widths.size(), {0.0F, 0.0F});
    for (std::size_t row = 0; row < widths.size(); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths_host[row], width);
        if (values.size() != static_cast<std::size_t>(lengths_host[row]))
            throw std::invalid_argument("prepare_cwt_workspace: real wavelet callable returned an unexpected length");
        for (int index = 0; index < lengths_host[row]; ++index)
            bank[row * data_count + index] = {
                static_cast<float>(values[index]), 0.0F};
    }
    CwtDeviceWorkspace workspace;
    workspace.data_count = data_count;
    workspace.width_count = static_cast<int>(widths.size());
    workspace.max_wavelet_length = data_count;
    workspace.complex_output = false;
    workspace.lengths = DeviceArray<int>::from_host(lengths_host);
    workspace.wavelets = DeviceArray<ComplexFloat>::from_host(bank);
    return workspace;
}
```

- 先验证 callable，再分配 $WN$ 个零初始化 `ComplexFloat`；每行固定跨度 $N$。
- row 循环按 `(L_j,int(width))` 调 callable并严格检查返回长度；内层把 double 缩窄为 float、虚部置零。
- 最后填写 metadata，并对 lengths/bank 各做一次 H2D。bank 此时尚未共轭反转。

#### 复 callable bank

证据：同 SHA；同 `.cpp`；复重载 `prepare_cwt_workspace`；第 170–197 行。

```cpp
template <class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("prepare_cwt_workspace: complex wavelet callable is empty");
    const auto lengths_host = cwt_lengths(data_count, widths);
    std::vector<ComplexFloat> bank(
        static_cast<std::size_t>(data_count) * widths.size(), {0.0F, 0.0F});
    for (std::size_t row = 0; row < widths.size(); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths_host[row], width);
        if (values.size() != static_cast<std::size_t>(lengths_host[row]))
            throw std::invalid_argument("prepare_cwt_workspace: complex wavelet callable returned an unexpected length");
        for (int index = 0; index < lengths_host[row]; ++index)
            bank[row * data_count + index] = {
                static_cast<float>(values[index].re),
                static_cast<float>(values[index].im)};
    }
    CwtDeviceWorkspace workspace;
    workspace.data_count = data_count;
    workspace.width_count = static_cast<int>(widths.size());
    workspace.max_wavelet_length = data_count;
    workspace.complex_output = true;
    workspace.lengths = DeviceArray<int>::from_host(lengths_host);
    workspace.wavelets = DeviceArray<ComplexFloat>::from_host(bank);
    return workspace;
}
```

- 与实重载逐行对应；差异是实虚部都从 `ComplexDouble` 缩窄，并设置 `complex_output=true`。
- 共轭和反转留到 CPU/GPU 累加时完成。

### 6. CPU/reference 核心计算逐行解释

#### 实小波 same 卷积

证据：同 SHA；同 `.cpp`；实重载 `cwt_typed_cpu`；第 199–226 行。

```cpp
template <class T>
std::vector<double> cwt_typed_cpu(
    const std::vector<T>& signal, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("cwt_typed_cpu: real wavelet callable is empty");
    const int n = static_cast<int>(signal.size());
    const auto lengths = cwt_lengths(n, widths);
    std::vector<double> out(signal.size() * widths.size());
    for (int row = 0; row < static_cast<int>(widths.size()); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths[row], width);
        if (values.size() != static_cast<std::size_t>(lengths[row]))
            throw std::invalid_argument("cwt_typed_cpu: real wavelet callable returned an unexpected length");
        for (int time = 0; time < n; ++time) {
            const int full = (lengths[row] - 1) / 2 + time;
            const int first = std::max(0, full - (lengths[row] - 1));
            const int last = std::min(n - 1, full);
            double sum = 0.0;
            for (int input = first; input <= last; ++input) {
                const int coefficient = lengths[row] - 1 - (full - input);
                sum += static_cast<double>(load(signal[input])) * values[coefficient];
            }
            out[row * n + time] = sum;
        }
    }
    return out;
}
```

- 验证 callable、取得 $N$、计算 lengths、分配 row-major $WN$ 输出。
- row 循环生成小波；time 循环计算每个 `same` 位置。`full/first/last` 将完整卷积索引夹到输入范围，范围外等价于补零。
- input 循环用反转 coefficient 做 double 累加，最后写 `out[row*n+time]`。

#### 复小波共轭 same 卷积

证据：同 SHA；同 `.cpp`；复重载 `cwt_typed_cpu`；第 228–257 行。

```cpp
template <class T>
std::vector<ComplexDouble> cwt_typed_cpu(
    const std::vector<T>& signal, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet)
{
    if (!wavelet) throw std::invalid_argument("cwt_typed_cpu: complex wavelet callable is empty");
    const int n = static_cast<int>(signal.size());
    const auto lengths = cwt_lengths(n, widths);
    std::vector<ComplexDouble> out(signal.size() * widths.size());
    for (int row = 0; row < static_cast<int>(widths.size()); ++row) {
        const int width = static_cast<int>(load(widths[row]));
        const auto values = wavelet(lengths[row], width);
        if (values.size() != static_cast<std::size_t>(lengths[row]))
            throw std::invalid_argument("cwt_typed_cpu: complex wavelet callable returned an unexpected length");
        for (int time = 0; time < n; ++time) {
            const int full = (lengths[row] - 1) / 2 + time;
            const int first = std::max(0, full - (lengths[row] - 1));
            const int last = std::min(n - 1, full);
            ComplexDouble sum{0.0, 0.0};
            for (int input = first; input <= last; ++input) {
                const int coefficient = lengths[row] - 1 - (full - input);
                const double value = static_cast<double>(load(signal[input]));
                sum.re += value * values[coefficient].re;
                sum.im -= value * values[coefficient].im;
            }
            out[row * n + time] = sum;
        }
    }
    return out;
}
```

- 结构与实重载相同；输出和 accumulator 改为 `ComplexDouble`。
- coefficient 索引完成反转，虚部减号完成共轭：$x(a+ib)^*=x(a-ib)$。

### 7. GPU host wrapper 与输出终结

#### real/complex wrapper

证据：同 SHA；同 `.cpp`；两组 `cwt_device`；第 259–283 行。

```cpp
template <class T>
void cwt_device(
    const DeviceArray<T>& signal, const CwtDeviceWorkspace& workspace,
    DeviceArray<double>& out)
{
    if (workspace.complex_output || out.size() != workspace.output_size())
        throw std::invalid_argument("cwt_device: real output requested with complex workspace or wrong output size");
    DeviceArray<ComplexFloat> computed(workspace.output_size());
    cwt_complex_fp32_compute_device(signal, workspace, computed);
    DeviceArray<float> real(computed.size());
    cwt_extract_real_fp32_device(computed, real);
    out = cuda_utils::finalize_fp64_device_storage(real);
}

template <class T>
void cwt_device(
    const DeviceArray<T>& signal, const CwtDeviceWorkspace& workspace,
    DeviceArray<ComplexDouble>& out)
{
    if (!workspace.complex_output || out.size() != workspace.output_size())
        throw std::invalid_argument("cwt_device: complex output requested with real workspace or wrong output size");
    DeviceArray<ComplexFloat> computed(workspace.output_size());
    cwt_complex_fp32_compute_device(signal, workspace, computed);
    out = cuda_utils::finalize_complex_fp64_device_storage(computed);
}
```

- 两个重载先检查 output kind 和 $WN$ 大小，再统一调用 ComplexFP32 核心。
- 实路径额外分配 float 数组并提取实部；复路径直接拓宽两个分量。最后的赋值替换 `out` storage。

#### FP64 存储 helper

证据：同 SHA；`ZKX_dev/ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`；两个 `finalize_*`；第 40–56 行。

```cpp
inline DeviceArray<double> finalize_fp64_device_storage(
    const DeviceArray<float>& computed)
{
    DeviceArray<double> output(computed.size());
    widen_fp32_storage_device(
        computed.data(), output.data(), computed.size());
    return output;
}

inline DeviceArray<ComplexDouble> finalize_complex_fp64_device_storage(
    const DeviceArray<ComplexFloat>& computed)
{
    static_assert(sizeof(ComplexDouble) == 2 * sizeof(double));
    DeviceArray<ComplexDouble> output(computed.size());
    widen_complex_fp32_storage_device(
        computed.data(), output.data(), computed.size());
    return output;
}
```

- helper 分配正式存储并启动位级拓宽；`static_assert` 保证复结构布局。
- `fp64_storage_finalize.cu:10–60` 将 FP32 符号、指数、尾数映射到 IEEE FP64 位域；没有 double 算术，数值仍等于 `double(float_result)`。

### 8. GPU 调度与 kernel 逐行解释

#### launch wrapper

证据：同 SHA；`ZKX_dev/ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`；`cwt_complex_fp32_compute_device`；第 134–161 行。

```cpp
template <typename T>
void cwt_complex_fp32_compute_device(
    const DeviceArray<T>& input,
    const CwtDeviceWorkspace& workspace,
    DeviceArray<ComplexFloat>& output)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (workspace.data_count != static_cast<int>(input.size()) ||
        workspace.width_count < 0 ||
        workspace.max_wavelet_length != workspace.data_count ||
        workspace.lengths.size() != static_cast<std::size_t>(workspace.width_count) ||
        workspace.wavelets.size() !=
            static_cast<std::size_t>(workspace.width_count) * input.size() ||
        output.size() != workspace.output_size()) {
        throw std::invalid_argument("cwt_device: input, workspace and output shapes are inconsistent");
    }
    if (output.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::cwt_convolution_kernel<T, ComplexFloat, float>,
        output.size(),
        input.data(),
        workspace.data_count,
        workspace.wavelets.data(),
        workspace.lengths.data(),
        workspace.max_wavelet_length,
        workspace.width_count,
        output.data());
}
```

- `static_assert` 限制五种 simple signal 类型；复合条件逐项验证 input/workspace/output shape。
- 空输出不 launch；非空时以 $WN$ 为线程总数，把全部 device 指针和 metadata 传入 custom kernel。没有 FFT plan、科学库或显式同步。

#### 核心 kernel：索引、边界与累加

证据：同 SHA；`ZKX_dev/ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`；`cwt_convolution_kernel`；第 126–170 行。

```cpp
template <typename Input, typename Complex, typename Scalar>
__global__ void cwt_convolution_kernel(
    const Input* data,
    int data_count,
    const Complex* wavelets,
    const int* lengths,
    int max_wavelet_length,
    int width_count,
    Complex* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = width_count * data_count;
    if (index >= total) {
        return;
    }
    const int width_index = index / data_count;
    const int sample_index = index % data_count;
    const int length = lengths[width_index];
    const int start = (length - 1) / 2;
    const int full_index = start + sample_index;
    const int first_candidate = full_index - (length - 1);
    const int input_min = first_candidate > 0 ? first_candidate : 0;
    const int input_max = full_index < data_count - 1
        ? full_index
        : data_count - 1;
    compute_policy::FmaFloatAccumulator real_sum;
    compute_policy::FmaFloatAccumulator imag_sum;
    const Complex* wavelet =
        wavelets + static_cast<long long>(width_index) * max_wavelet_length;
    for (int input_index = input_min; input_index <= input_max; ++input_index) {
        const int coefficient = full_index - input_index;
        Scalar coefficient_real = Scalar{0};
        Scalar coefficient_imag = Scalar{0};
        WaveletComplexTraits<Complex>::load(
            wavelet + length - 1 - coefficient,
            coefficient_real,
            coefficient_imag);
        const Scalar input_value =
            static_cast<Scalar>(wavelet_load(data[input_index]));
        real_sum.add_product(input_value, coefficient_real);
        imag_sum.add_product(-input_value, coefficient_imag);
    }
    WaveletComplexTraits<Complex>::store(
        output + index, real_sum.value(), imag_sum.value());
}
```

- 一维 index 映射为唯一 `[width_index,sample_index]`；越界线程返回。
- `start/full_index/input_min/input_max` 实现 `same` 居中和零扩展等价边界。
- 每线程拥有两个 FP32 FMA accumulator；row 指针以固定跨度 $N$ 定位。
- `length-1-coefficient` 完成反转；虚部负号完成共轭。线程循环结束只写自己的 `output[index]`，无需 atomic、barrier 或共享内存。

#### 实部提取 kernel

证据：同 SHA；同 `.cuh`；`cwt_extract_real_kernel`；第 172–179 行。

```cpp
__global__ void cwt_extract_real_kernel(
    const ComplexFloat* input, float* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index].re;
}
```

- 每线程按相同 index 复制实部；条件防止尾块越界。这是实输出 device 计算的最后一行核心代码。

### 9. 模板与显式实例化

证据：同 SHA；`ZKX_dev/ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`；`INSTANTIATE_CPU` 的 CWT 项与调用；第 326–345 行。

```cpp
    template CwtDeviceWorkspace prepare_cwt_workspace( \
        int, const std::vector<T>&, const CwtRealWaveletCallable&); \
    template CwtDeviceWorkspace prepare_cwt_workspace( \
        int, const std::vector<T>&, const CwtComplexWaveletCallable&); \
    template std::vector<double> cwt_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const CwtRealWaveletCallable&); \
    template std::vector<ComplexDouble> cwt_typed_cpu( \
        const std::vector<T>&, const std::vector<T>&, const CwtComplexWaveletCallable&); \
    template void cwt_device( \
        const DeviceArray<T>&, const CwtDeviceWorkspace&, DeviceArray<double>&); \
    template void cwt_device( \
        const DeviceArray<T>&, const CwtDeviceWorkspace&, DeviceArray<ComplexDouble>&); \

INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
```

- 六组显式实例化随宏参数展开；五次调用覆盖 FP32、FP16、INT32、INT16、INT8。
- `.cu:211–221` 对 GPU compute helper 同样实例化五种输入；输出由 real/complex overload 决定。

### 10. 测试代码逐行解释到 CWT 块最后一行

#### callable 与 CPU/GPU 对照

证据：同 SHA；`ZKX_dev/ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`；CWT 测试块；第 368–400 行。

```cpp
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::cwt);
    const std::vector<T> widths{convert<T>(2)};
    const CwtRealWaveletCallable real_wavelet = [](
        int points, int width) {
        std::vector<double> values(points);
        for (int index = 0; index < points; ++index) {
            const double centered = index - 0.5 * (points - 1);
            values[index] = centered / static_cast<double>(width);
        }
        return values;
    };
    const CwtComplexWaveletCallable complex_wavelet = [](
        int points, int width) {
        std::vector<ComplexDouble> values(points);
        for (int index = 0; index < points; ++index) {
            const double phase = static_cast<double>(index + 1) / width;
            values[index] = {std::cos(phase), std::sin(phase)};
        }
        return values;
    };
    auto cwt_real_workspace = prepare_cwt_workspace(
        static_cast<int>(time.size()), widths, real_wavelet);
    DeviceArray<double> cwt_real_out(time.size() * widths.size());
    cwt_device(d_time, cwt_real_workspace, cwt_real_out);
    const bool cwt_real_matches = matches_fp64(
        cwt_real_out.to_host(), cwt_typed_cpu(time, widths, real_wavelet));
    auto cwt_complex_workspace = prepare_cwt_workspace(
        static_cast<int>(time.size()), widths, complex_wavelet);
    DeviceArray<ComplexDouble> cwt_complex_out(time.size() * widths.size());
    cwt_device(d_time, cwt_complex_workspace, cwt_complex_out);
    const bool cwt_complex_matches = matches_complex(
        cwt_complex_out.to_host(),
        cwt_typed_cpu(time, widths, complex_wavelet));
```

- 测试用宽度 2 构造中心化实斜坡与单位圆复序列，验证任意 callable 而非固定 Ricker/Morlet。
- real/complex 都按“prepare → 分配 $WN$ → GPU → D2H → CPU reference 对比”执行。

#### 异常、空宽度和最终判据

证据：同 SHA；同测试文件；第 401–443 行。

```cpp
    bool cwt_rejects_bad_callable_length = false;
    const CwtRealWaveletCallable bad_wavelet = [](
        int, int) { return std::vector<double>{}; };
    try {
        (void)prepare_cwt_workspace(
            static_cast<int>(time.size()), widths, bad_wavelet);
    } catch (const std::invalid_argument&) {
        cwt_rejects_bad_callable_length = true;
    }
    bool cwt_rejects_output_kind_mismatch = false;
    try {
        cwt_device(d_time, cwt_complex_workspace, cwt_real_out);
    } catch (const std::invalid_argument&) {
        cwt_rejects_output_kind_mismatch = true;
    }
    const std::vector<T> empty_widths;
    auto empty_cwt_workspace = prepare_cwt_workspace(
        static_cast<int>(time.size()), empty_widths, real_wavelet);
    DeviceArray<double> empty_cwt_out;
    cwt_device(d_time, empty_cwt_workspace, empty_cwt_out);
    const bool cwt_empty_widths_match = empty_cwt_out.empty() &&
        cwt_typed_cpu(time, empty_widths, real_wavelet).empty();
    bool cwt_rejects_nonpositive_width = false;
    try {
        (void)prepare_cwt_workspace(
            static_cast<int>(time.size()),
            std::vector<T>{convert<T>(0)}, real_wavelet);
    } catch (const std::invalid_argument&) {
        cwt_rejects_nonpositive_width = true;
    }
    ok[5] = cwt_real_matches && cwt_complex_matches &&
        cwt_rejects_bad_callable_length &&
        cwt_rejects_output_kind_mismatch && cwt_empty_widths_match &&
        cwt_rejects_nonpositive_width;
    if (!ok[5]) {
        std::cout << "[E3][wave_window][" << name
                  << "][cwt][detail] real=" << cwt_real_matches
                  << " complex=" << cwt_complex_matches
                  << " reject_length=" << cwt_rejects_bad_callable_length
                  << " reject_kind=" << cwt_rejects_output_kind_mismatch
                  << " empty=" << cwt_empty_widths_match
                  << " reject_width=" << cwt_rejects_nonpositive_width << '\n';
    }
```

- 三个 try/catch 分别验证错误 callable 长度、输出种类不匹配和 width 0；空 widths 则要求 CPU/GPU 都返回空。
- `ok[5]` 要求六项全部为真；失败时逐项打印诊断。最后的 `}` 是 CWT 专属测试块的最后一行，第 445 行已转入其他算子。
- 未直接覆盖多 width 不同长度、大规模、奇偶长度全部组合、极端数值、并发 workspace 复用和性能。

### 11. 数学—Python—CPU—GPU 映射

| 原理或公式 | cuSignal Python | C++ CPU | C++ GPU | 差异说明 |
| --- | --- | --- | --- | --- |
| $L_j=\min(10\operatorname{int}(w_j),N)$ | `wavelets.py:318` | `.cpp:26–30` | workspace 共用 lengths | C++ 额外拒绝 $L_j\le0$ |
| callable 生成 $\psi_j$ | 每次调用 | 每 row 调 callable | Host prepare 后 H2D | GPU正式调用不执行 callable |
| 共轭反转 | `cp.conj(...)[::-1]` | 反转索引+虚部减号 | 反转索引+imag负号 | 数学等价 |
| same 卷积 | auto direct/FFT | direct 三重循环 | custom direct kernel | C++ GPU 不自动选 FFT |
| `[W,N]` | CuPy二维 | row-major vector | 线性 DeviceArray | 布局等价 |
| FP64输出 | 通常FP64运算 | double累加 | FP32算术后拓宽存储 | GPU精度有意不同 |

### 12. 并行、同步、内存与复杂度

- grid/block：`launch_1d_kernel` 按 $WN$ 生成一维配置；每线程负责一个 `(width,time)`。
- 同步：无 `__syncthreads()`、atomic 或显式 synchronize；同 stream 排序后续 kernel，调用者读取前负责同步。
- workspace bank 固定为 $W\times N$ ComplexFP32，真实每行仅前 $L_j$ 有效。
- CPU总时间 $O(\sum_j NL_j)$；GPU总工作量相同，单线程 $O(L_j)$。
- workspace/output 都是 $O(WN)$；实输出另有 ComplexFP32 与 FP32 两个临时数组。
- 主要风险：线程内长串行循环、重复全局内存读取、固定 stride=N 的 bank padding 浪费。

### 13. dtype、精度和 ZQ500 限制

- 显式支持 FP32、FP16、INT32、INT16、INT8 输入。
- callable 的 FP64/ComplexFP64 结果在 H2D 前缩窄为 ComplexFP32；kernel 使用 FP32 FMA。
- 正式 FP64/ComplexFP64 只是 FP32 结果的精确存储拓宽，不恢复已丢失精度。
- device 不依赖 FP64 算术，符合 ZQ500 不支持 FP64 的约束。
- 本路径不调用 dlfft，因而不受 FFT 点数必须为 2 的幂限制。
- 大 INT32 转 float 可能无法精确表示；CPU double reference 与 GPU FP32 应按 filtering 容差比较。

### 14. 相同、等价替换和有意不同

| 类别 | 内容 |
| --- | --- |
| 相同 | `int(width)`、十倍宽度截断、每 width 一行、same 边界、共轭反转、`[W,N]` |
| 等价替换 | Python `cp.conj()[::-1]` 由索引反转与虚部负号替代 |
| 有意不同 | GPU将 callable 提前物化为可复用 workspace |
| 有意不同 | Python auto 可 direct/FFT；C++ GPU固定 custom direct |
| 有意不同 | Python探测实/复；C++用 overload 显式决定 |
| 扩展 | C++增加 callable长度、width、shape、output kind 检查 |

### 15. 自检问题与参考答案

1. **版本身份与 dirty 状态？** 答：完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，提交时间 `2026-08-15T23:45:58+08:00`，分支 `final-prep/benchmark-evidence-v1`，相关文件 clean。
2. **CPU/GPU职责差异？** 答：CPU直接调 callable 并用 double/ComplexDouble 循环；GPU先Host准备 bank，再FP32 kernel计算并拓宽存储，证据 `.cpp:143–283`、`.cu:134–173`。
3. **共轭反转在哪里？** 答：`length-1-(full-input)` 反转；虚部减号共轭，证据 `.cpp:219–251`、`.cuh:155–166`。
4. **每线程职责？** 答：一个线性 index 对应一个 `[width,time]`，线程内部累加所有有效重叠输入，见 `.cuh:136–169`。
5. **为何无线程同步？** 答：线程只读共享输入但写唯一输出，无跨线程依赖；不同 kernel 由同 stream 排序。
6. **五类型如何进入？** 答：CPU/GPU宏分别显式实例化 `float,__half,int32_t,int16_t,int8_t`，见 `.cpp:326–345`、`.cu:211–221`。
7. **FP64输出是否FP64计算？** 答：否；核心是 ComplexFP32/FMAFloat，finalize仅位级拓宽，见 `.cuh:151–169`、`host_output_finalize.h:40–56`。
8. **测试边界？** 答：覆盖实/复对照、坏长度、kind错配、空 widths、width 0；未覆盖多width、大规模、极值、并发和性能，见测试 `:368–443`。
9. **为何workspace为 $O(WN)$？** 答：bank 每行固定 stride=N，分配 `data_count*widths.size()`，见 `.cpp:149–150,177–178`。
10. **与Python最大算法差异？** 答：Python卷积可auto direct/FFT；C++ GPU固定一线程一输出的direct kernel，并预物化callable，数学same相关不变。

## 版本差异与原理不变量

当前只有 V1。未来 V2/V3 必须追加，不能覆盖本版路径和行号。

| 比较项 | V1 | 原理不变量 |
| --- | --- | --- |
| 结构 | callable bank + direct convolution | 信号与共轭反转小波的局部相关 |
| 复杂度 | $O(\sum_j NL_j)$ | 每尺度每位置需求内积 |
| 内存 | bank/output $O(WN)$ | 输出逻辑 `[W,N]` |
| 并行 | 一线程一输出 | 各系数相互独立 |
| 精度 | device FP32、FP64存储 | 实/复输出语义保持 |
| 边界 | same 零扩展等价裁剪 | 只累加有效重叠区 |
