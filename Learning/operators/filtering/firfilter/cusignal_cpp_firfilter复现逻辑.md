# cusignal_cpp_firfilter 复现逻辑

## 版本索引

| 版本 | 快照 | 状态 |
| --- | --- | --- |
| [V1：首次正式学习版本](#v1首次正式学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 |

## V1：首次正式学习版本（fdd55ac）

### 1. 版本身份

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交主题：`test(archive): 增加任务结果范围治理审计`
- 进入阶段三时 `ZKX` 工作树 clean，`firfilter` 相关文件均无 dirty 修改。

本章所有行号均属于上述 SHA，可用 `git -C ZKX show <SHA>:cusignal_cpp/<path>` 重建，不是未绑定版本的裸行号。

### 2. 证据范围与分层

| 层 | `ZKX_dev/` 相对路径 | 符号与行号 |
| --- | --- | --- |
| 公开数据结构 | `ZKX/cusignal_cpp/src/filtering/filtering_typed.h` | `FirfilterOptions` 249-253；CPU/GPU result 255-269 |
| 布局规划 | 同上 | `FirfilterLayoutPlan` 468-477；`prepare_firfilter_layout` 479-481 |
| CPU 入口 | 同上 | `firfilter_typed_cpu<T>` 819-822 |
| GPU 入口 | 同上 | `firfilter_device<T>` 552-556；resident 563-565 |
| 布局实现 | `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` | `prepare_firfilter_layout` 1125-1224 |
| CPU 核心 | 同上 | `firfilter_typed_cpu<T>` 1536-1592 |
| GPU host wrapper | `ZKX/cusignal_cpp/src/filtering/filtering_typed.cu` | `firfilter_device<T>` 117-168；resident 170-184 |
| GPU device 核心 | `ZKX/cusignal_cpp/src/filtering/filtering_kernels.cuh` | `firfilter_full_value` 1394-1423；output kernel 1425-1446；state kernel 1448-1470 |
| 显式实例化 | `filtering_typed.cpp` / `filtering_typed.cu` | CPU 2781-2783；GPU 683-685 |
| 直接 smoke | `ZKX/cusignal_cpp/test/signal_processing/e3_filtering_type_smoke.cu` | 比较 helper 110-119；输入 370-379；测试 564-647 |

契约表、CMake 注册和审计文档不是算子核心。`firfilter2` 的 padding 和双向滤波也不在本章范围。

### 3. 公共接口和数据结构

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/filtering/filtering_typed.h`，249-269：

```cpp
struct FirfilterOptions {
    std::vector<int> shape;
    int axis = -1;
    std::vector<int> zi_shape;
};

struct FirfilterCpuResult {
    std::vector<float> y;
    std::vector<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};

struct FirfilterDeviceResult {
    DeviceArray<float> y;
    DeviceArray<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};
```

`shape` 描述 row-major N 维输入；空时按一维处理。`axis=-1` 对应 Python 最后一轴。`zi_shape` 保留非滤波轴的单例维广播契约。CPU/GPU 结果都固定 FP32；C++ 用 `has_zf` 表达 Python 的动态单返回/`(y,zf)` 双返回。

SHA 同上，同一文件，552-565、819-822：

```cpp
template <typename T>
void firfilter_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    FirfilterDeviceResult& out, const FirfilterOptions& options = {},
    const DeviceArray<T>* zi = nullptr);

void firfilter_fp32_resident_device(
    const DeviceArray<float>& b, const DeviceArray<float>& x,
    DeviceArray<float>& y);

template <typename T>
FirfilterCpuResult firfilter_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& x,
    const FirfilterOptions& options = {}, const std::vector<T>* zi = nullptr);
```

`firfilter_device<T>` 是完整 GPU 入口；resident 入口仅是一维、无初态、预分配 FP32 快速路径；`firfilter_typed_cpu<T>` 是不调用 GPU 的独立 Host reference。

### 4. 布局规划：N 维 axis 到多条一维 FIR

SHA 同上，`ZKX/cusignal_cpp/src/filtering/filtering_typed.h`，`FirfilterLayoutPlan`，468-477：

```cpp
struct FirfilterLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> state_shape;
    std::vector<int> zi_line_offsets;
    int sample_count = 0;
    int state_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int zi_state_stride = 0;
};
```

对 shape $(d_0,\ldots,d_{R-1})$ 和归一化 axis $a$：

$$
N=d_a,\quad S=L-1,\quad I=\prod_{j>a}d_j,\quad O=\prod_{j<a}d_j.
$$

数组被视为 $OI$ 条独立 FIR 线，每条长 $N$。`state_shape` 仅将 $d_a$ 替换为 $S$。

SHA 同上，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp`，`prepare_firfilter_layout`，1125-1171：

```cpp
FirfilterLayoutPlan prepare_firfilter_layout(
    std::size_t input_size, std::size_t coefficient_count,
    const FirfilterOptions& options, bool has_zi, std::size_t zi_size)
{
    if (coefficient_count == 0 ||
        coefficient_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("firfilter coefficients");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("firfilter input exceeds int indexing");
    FirfilterLayoutPlan plan;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));
    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent <= 0 || product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("firfilter input shape");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("firfilter input shape product");
    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("firfilter axis");
    plan.sample_count = plan.input_shape[axis];
    plan.state_count = static_cast<int>(coefficient_count) - 1;
    plan.state_shape = plan.input_shape;
    plan.state_shape[axis] = plan.state_count;
    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(input_size /
        static_cast<std::size_t>(plan.sample_count * plan.inner_count));
    if (!has_zi) {
        if (!options.zi_shape.empty())
            throw std::invalid_argument("firfilter zi_shape without zi");
        if (zi_size != 0)
            throw std::invalid_argument("firfilter zi size without zi");
        return plan;
    }
```

这一块检查系数、`int` 索引范围、shape 正值和乘积、axis，并计算 $O/N/I/S$。无 `zi` 时早返回，但拒绝矛盾的 `zi_shape`/`zi_size`。

SHA 同上，同一符号，1173-1224：

```cpp
    std::vector<int> zi_shape = options.zi_shape.empty()
        ? plan.state_shape : options.zi_shape;
    if (zi_shape.size() != plan.input_shape.size())
        throw std::invalid_argument("firfilter zi rank");
    long long zi_product = 1;
    for (std::size_t dimension = 0; dimension < zi_shape.size(); ++dimension) {
        const int extent = zi_shape[dimension];
        const int expected = plan.state_shape[dimension];
        if (extent < 0 ||
            (static_cast<int>(dimension) == axis
                ? extent != expected
                : extent != expected && extent != 1))
            throw std::invalid_argument("firfilter zi broadcast shape");
        if (zi_product != 0 && extent != 0 &&
            zi_product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("firfilter zi shape overflow");
        zi_product *= extent;
    }
    if (zi_product != static_cast<long long>(zi_size))
        throw std::invalid_argument("firfilter zi shape product");
    std::vector<int> zi_strides(zi_shape.size(), 1);
    for (int dimension = static_cast<int>(zi_shape.size()) - 2;
         dimension >= 0; --dimension)
        zi_strides[dimension] = zi_strides[dimension + 1]
            * zi_shape[dimension + 1];
    plan.zi_state_stride = zi_strides[axis];
    if (plan.state_count == 0) return plan;
    const int line_count = plan.outer_count * plan.inner_count;
    plan.zi_line_offsets.resize(line_count);
    for (int line = 0; line < line_count; ++line) {
        int outer = line / plan.inner_count;
        int inner_index = line % plan.inner_count;
        int offset = 0;
        for (int dimension = axis - 1; dimension >= 0; --dimension) {
            const int coordinate = outer % plan.input_shape[dimension];
            outer /= plan.input_shape[dimension];
            if (zi_shape[dimension] != 1)
                offset += coordinate * zi_strides[dimension];
        }
        for (int dimension = static_cast<int>(plan.input_shape.size()) - 1;
             dimension > axis; --dimension) {
            const int coordinate = inner_index % plan.input_shape[dimension];
            inner_index /= plan.input_shape[dimension];
            if (zi_shape[dimension] != 1)
                offset += coordinate * zi_strides[dimension];
        }
        plan.zi_line_offsets[line] = offset;
    }
    return plan;
}
```

`zi_strides` 是 row-major 步长。`zi_line_offsets[line]` 是每条 FIR 线的状态起点；某非 axis 维为 1 时不把该坐标加入 offset，等价于 Python stride-0 广播。

### 5. CPU/reference 核心

SHA 同上，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp`，`firfilter_typed_cpu<T>`，1536-1592：

```cpp
template <typename T>
FirfilterCpuResult firfilter_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& x,
    const FirfilterOptions& options, const std::vector<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_firfilter_layout(
        x.size(), b.size(), options, zi != nullptr, zi == nullptr ? 0 : zi->size());
    FirfilterCpuResult result;
    result.has_zf = zi != nullptr;
    result.y_shape = plan.input_shape;
    result.y.resize(x.size());
    if (result.has_zf) {
        result.zf_shape = plan.state_shape;
        result.zf.resize(static_cast<std::size_t>(plan.outer_count)
            * plan.state_count * plan.inner_count);
    }
    const int line_count = plan.outer_count * plan.inner_count;
    const auto full_value = [&](int line, int full_index) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t input_base = static_cast<std::size_t>(outer)
            * plan.sample_count * plan.inner_count + inner;
        double value = 0.0;
        for (int tap = 0; tap <= plan.state_count; ++tap) {
            const int sample = full_index - tap;
            if (sample >= 0 && sample < plan.sample_count)
                value += detrend_host_load(b[tap]) * detrend_host_load(
                    x[input_base + static_cast<std::size_t>(sample)
                        * plan.inner_count]);
        }
        if (zi != nullptr && full_index < plan.state_count) {
            const int zi_index = plan.zi_line_offsets[line]
                + full_index * plan.zi_state_stride;
            value += detrend_host_load((*zi)[zi_index]);
        }
        return static_cast<float>(value);
    };
    for (int line = 0; line < line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        const std::size_t output_base = static_cast<std::size_t>(outer)
            * plan.sample_count * plan.inner_count + inner;
        for (int sample = 0; sample < plan.sample_count; ++sample)
            result.y[output_base + static_cast<std::size_t>(sample)
                * plan.inner_count] = full_value(line, sample);
        if (result.has_zf) {
            const std::size_t state_base = static_cast<std::size_t>(outer)
                * plan.state_count * plan.inner_count + inner;
            for (int state = 0; state < plan.state_count; ++state)
                result.zf[state_base + static_cast<std::size_t>(state)
                    * plan.inner_count] = full_value(
                        line, plan.sample_count + state);
        }
    }
    return result;
}
```

`full_value(line,q)` 计算

$$
c[q]=\sum_{k=0}^{L-1}b[k]x[q-k],
$$

且只在 $0\le q-k<N$ 读取输入。$q<L-1$ 且有 `zi` 时再加 $z_i[q]$。CPU 用 `double` 累加后转 FP32。`y` 计算 $q=0,\ldots,N-1$；`zf` 计算 $q=N,\ldots,N+L-2$，与 Python full 卷积的前缀/尾部分割一致。

### 6. GPU host wrapper

SHA 同上，`ZKX/cusignal_cpp/src/filtering/filtering_typed.cu`，`firfilter_device<T>`，117-168：

```cpp
template <class T>
void firfilter_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    FirfilterDeviceResult& output, const FirfilterOptions& options,
    const DeviceArray<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const auto plan = detail::prepare_firfilter_layout(
        x.size(), b.size(), options, zi != nullptr, zi == nullptr ? 0 : zi->size());
    output = FirfilterDeviceResult{};
    output.has_zf = zi != nullptr;
    output.y_shape = plan.input_shape;
    output.y = DeviceArray<float>(x.size());
    if (output.has_zf) {
        output.zf_shape = plan.state_shape;
        output.zf = DeviceArray<float>(
            static_cast<std::size_t>(plan.outer_count)
            * plan.state_count * plan.inner_count);
    }
    DeviceArray<float> coefficients(b.size());
    DeviceArray<float> input(x.size());
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, b.size(),
        b.data(), coefficients.data(), b.size());
    cuda_utils::launch_1d_kernel(
        detail::convert_signal_kernel<T,float>, x.size(),
        x.data(), input.data(), x.size());
    DeviceArray<float> initial_state;
    if (zi != nullptr) {
        initial_state.reset(zi->size());
        if (!zi->empty())
            cuda_utils::launch_1d_kernel(
                detail::convert_signal_kernel<T,float>, zi->size(),
                zi->data(), initial_state.data(), zi->size());
    }
    auto line_offsets = DeviceArray<int>::from_host(plan.zi_line_offsets);
    cuda_utils::launch_1d_kernel(
        detail::firfilter_axis_output_kernel, output.y.size(),
        coefficients.data(), static_cast<int>(coefficients.size()), input.data(),
        static_cast<int>(input.size()), plan.sample_count, plan.inner_count,
        initial_state.data(), line_offsets.data(), plan.zi_state_stride,
        zi != nullptr, output.y.data());
    if (!output.zf.empty())
        cuda_utils::launch_1d_kernel(
            detail::firfilter_axis_state_kernel, output.zf.size(),
            coefficients.data(), static_cast<int>(coefficients.size()), input.data(),
            plan.sample_count, plan.inner_count, initial_state.data(),
            line_offsets.data(), plan.zi_state_stride, output.zf.data(),
            static_cast<int>(output.zf.size()));
}
```

五种 `T` 都先在 device 上转 FP32。Host 规划的 line-offset 表上传 device。主 kernel 启动 $|x|$ 个逻辑工作项；有 `zf` 时再启动 $OI(L-1)$ 个。wrapper 没有显式 `cudaDeviceSynchronize`/stream 参数，启动错误契约由 `launch_1d_kernel` 封装承担。

resident 入口（同文件 170-184）把 `sample_count=input.size()`、`inner_count=1`、`has_zi=false`，只启动 output kernel，不分配类型转换 workspace。

### 7. GPU device helper 和 kernel

SHA 同上，`ZKX/cusignal_cpp/src/filtering/filtering_kernels.cuh`，`firfilter_full_value`，1394-1423：

```cpp
__device__ inline float firfilter_full_value(
    const float* coefficients, int coefficient_count,
    const float* input, int sample_count, int inner_count,
    int outer, int inner, int full_index,
    const float* zi, const int* zi_line_offsets,
    int zi_state_stride, bool has_zi)
{
    const long long input_base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
    float value = 0.0F;
    for (int tap = 0; tap < coefficient_count; ++tap) {
        const int sample = full_index - tap;
        if (sample >= 0 && sample < sample_count)
            value += coefficients[tap]
                * input[input_base + static_cast<long long>(sample) * inner_count];
    }
    const int state_count = coefficient_count - 1;
    if (has_zi && full_index < state_count) {
        const int line = outer * inner_count + inner;
        value += zi[zi_line_offsets[line] + full_index * zi_state_stride];
    }
    return value;
}
```

它不存储完整 `out_full`，而按需直接计算一个 `full_index`，数学等价、存储策略不同。

SHA 同上，同一文件，output kernel 1425-1446：

```cpp
static __global__ void firfilter_axis_output_kernel(
    const float* coefficients, int coefficient_count,
    const float* input, int input_size, int sample_count, int inner_count,
    const float* zi, const int* zi_line_offsets,
    int zi_state_stride, bool has_zi, float* output)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= input_size) return;
    const int inner = index % inner_count;
    const int sample = (index / inner_count) % sample_count;
    const int outer = index / (sample_count * inner_count);
    output[index] = firfilter_full_value(
        coefficients, coefficient_count, input, sample_count, inner_count,
        outer, inner, sample, zi, zi_line_offsets, zi_state_stride, has_zi);
}
```

每线程负责一个 `y[index]`。线性索引还原为

$$
i=index\bmod I,quad n=\lfloor index/I\rfloor\bmod N,quad
o=\lfloor index/(NI)\rfloor.
$$

grid/block 由 `launch_1d_kernel` 决定；kernel 使用 `blockIdx.x*blockDim.x+threadIdx.x` 并自行越界返回。

SHA 同上，state kernel 1448-1470：

```cpp
static __global__ void firfilter_axis_state_kernel(
    const float* coefficients, int coefficient_count,
    const float* input, int sample_count, int inner_count,
    const float* zi, const int* zi_line_offsets, int zi_state_stride,
    float* final_state, int final_state_size)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= final_state_size) return;
    const int state_count = coefficient_count - 1;
    const int inner = index % inner_count;
    const int state = (index / inner_count) % state_count;
    const int outer = index / (state_count * inner_count);
    final_state[index] = firfilter_full_value(
        coefficients, coefficient_count, input, sample_count, inner_count,
        outer, inner, sample_count + state, zi, zi_line_offsets,
        zi_state_stride, true);
}
```

每线程负责一个 `zf`，状态 $s$ 映射到 full 索引 $N+s$。当 $N<L-1$ 时，初态仍可传播到 `zf`，与 Python 先加 `zi` 再切尾部一致。

`firfilter_direct_kernel<T>` 定义在同文件 1350-1392，但 SHA `fdd55ac...` 的正式 wrapper 实际启动 axis-output/state kernel，resident 也启动 axis-output kernel。因而 direct templated kernel 是未接入当前公开调用链的备选/历史 helper，不能写成当前核心。

### 8. template、dtype、实例化与 ZQ500 边界

- SHA 同上，`filtering_typed.cpp:2781-2783` 实例化 CPU，`filtering_typed.cu:683-685` 实例化 GPU。
- 类型均为 `float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t`。
- `T` 是业务输入/系数/初态类型；CPU/GPU `y`/`zf` 均为 FP32。GPU wrapper 先将 `b/x/zi` 转 FP32。
- 这是项目对 Python 契约的扩展：Python 纯整数 `result_type` 会被 `dtype.char` 门禁拒绝，C++ 则给四种非 FP32 业务类型提供 FP32 合法路径。
- 复数 typed FIR 入口不存在，不得从 Python 支持推断 C++ 已支持。
- shape 乘积、系数数量和输入数量受 `int` 索引范围限制。
- 实现不使用 FFT/dlfft、shared memory 或 device FP64。是否可在 ZQ500 构建/运行必须以远程 `gpu_02` 为准；本学习任务未连接服务器、未构建、未运行测试。

### 9. 内存、同步与复杂度

| 对象 | CPU | GPU |
| --- | --- | --- |
| `b/x/zi` | Host `vector<T>`，卷积时 load 到 double | 已驻留 device，wrapper 转临时 FP32 |
| 广播映射 | Host `zi_line_offsets` | Host 生成后 H2D |
| `y/zf` | Host FP32 | device FP32 |
| `out_full` | 不分配 | 不分配 |

正式 GPU workspace 是 FP32 `coefficients/input/initial_state` 和 INT32 offsets。wrapper 不 D2H 读回结果，不显式同步。

设独立 FIR 线数 $B=OI$、轴长 $N$、taps $L$：主输出总算术量 $O(BNL)$；有尾部时最坏额外 $O(BL^2)$。GPU 主输出逻辑线程数 $BN$，状态线程数 $B(L-1)$，每线程最多遍历 $L$ taps。空间为 $O(BN+BL)$ 输出加 FP32 转换 workspace 和 $O(B)$ offset 表。大 $L$ 时每线程串行 taps 循环是主瓶颈。

### 10. Python—CPU—GPU 核心映射

| 原理 | Python | CPU | GPU | 差异 |
| --- | --- | --- | --- | --- |
| $c[q]=\sum b[k]x[q-k]$ | 基准 `filtering.py:186` | SHA 同上 `full_value` 1554-1573 | `firfilter_full_value` 1394-1423 | Python 生成 full 数组；C++ 按需位置计算 |
| 越界输入为 0 | `cp.convolve` linear full | `sample` 边界 | 同左 | C++ 显式判断 |
| 前缀加 `zi` | Python 188-191 | CPU 1567-1571 | helper 1417-1421 | offsets 替代 stride-0 视图 |
| $y=c'[0:N]$ | Python 193-197 | CPU 1574-1581 | output kernel | GPU 一线程一 `y` |
| $zf=c'[N:N+L-1]$ | Python 199-201 | CPU 1582-1589 | state kernel | 只有 `zi` 才生成 |
| N 维 axis | `apply_along_axis` | `outer/sample/inner` | `index` 还原三坐标 | 都是多条一维 FIR |

### 11. 相同、等价替换和有意不同

- **相同**：指定 axis direct linear convolution；状态轴 $L-1$；非 axis 单例广播；有 `zi` 才有 `zf`；`y` 与 `x` 同 shape。
- **等价替换**：`apply_along_axis` 换成 `outer/sample/inner`；stride-0 视图换成 offsets；完整 `out_full` 换成按需 full-index 计算。
- **有意不同**：C++ 扩展四种非 FP32 业务输入并固定 FP32 输出；CPU double 累加而 GPU float 累加；异常文本不追求与 Python 逐字符一致。

### 12. 直接测试证据和未覆盖风险

SHA 同上，`ZKX/cusignal_cpp/test/signal_processing/e3_filtering_type_smoke.cu`，`run_type<T>` 370-379 定义 $x=[1,2,3,4,5,6,7,8]$、$b=[1,1]$ 并转五种 `T`。`matches_firfilter` 110-119 比较 `has_zf`、两个 shape 和 `y/zf` 数值。

`firfilter` 测试 564-647 覆盖：CPU/GPU 空系数拒绝；`shape={2,4},axis=1`；无状态；`zi_shape={1,1}` 广播；有 `zf`；越界 axis；错误 `zi_shape`；手写 `y=[1,3,5,7,5,11,13,15]`；有初态时两条线的首元素加 `0.5`；`zf=[4,8]`；GPU 与 CPU reference 对照。

未直接覆盖：复数、rank-3 中间轴手写值、$N<L-1$ 初态传播到 `zf`、大 taps 精度/性能压力。本轮未运行测试，只声称已提交测试源码存在上述检查。

### 13. 阅读顺序

1. `FirfilterOptions` 和 CPU/GPU result。
2. `prepare_firfilter_layout`，手算 `{2,4}`, axis 1, $L=2$ 时 $O=2,N=4,I=1,S=1$。
3. CPU `full_value`，再看 `y/zf` 两段索引。
4. GPU wrapper 的类型转换、workspace、offset 上传和两次 launch。
5. device helper 及 output/state kernel 的线性索引。
6. smoke 的手写期望值。

### 14. 自检问题与参考答案

#### 问题 1：V1 为什么可追溯？

**参考答案 1：** 它绑定完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`、提交时间和分支，且阶段三入口时工作树 clean，可用 `git show` 重建每条证据。

#### 问题 2：CPU、GPU wrapper 和 GPU kernel 各在哪里？

**参考答案 2：** CPU 是 `filtering_typed.cpp:1536-1592`；wrapper 是 `filtering_typed.cu:117-168`；device helper/output/state 是 `filtering_kernels.cuh:1394-1470`，均属于 V1 SHA。

#### 问题 3：N 维 axis 如何转为多条 FIR？

**参考答案 3：** $O=\prod_{j<a}d_j$，$N=d_a$，$I=\prod_{j>a}d_j$；固定 $(outer,inner)$ 就得到一条长 $N$ 的 FIR。证据是 layout 1157-1163 和 output kernel 1440-1445。

#### 问题 4：C++ 如何复现 stride-0 `zi` 广播？

**参考答案 4：** layout 1173-1222 对每条线生成 `zi_line_offsets`；单例维不把坐标加入 offset，多条线因而复用同一状态，与 stride 0 等价。

#### 问题 5：CPU/GPU 卷积公式有什么精度差异？

**参考答案 5：** 两者均令 `sample=full_index-tap` 并在 $0\le sample<N$ 累加；CPU 1559-1572 用 double 累加后转 float，GPU helper 1410-1422 用 float 累加，因此只保证容差内一致。

#### 问题 6：GPU 每线程负责什么？

**参考答案 6：** output kernel 一线程一 `y[index]`，state kernel 一线程一 `zf[index]`。两者使用 `blockIdx.x*blockDim.x+threadIdx.x`；grid/block 由 `launch_1d_kernel` 决定。

#### 问题 7：哪些类型被实例化，输出为什么仍是 FP32？

**参考答案 7：** CPU/GPU `INST` 均展开 `float/__half/int32/int16/int8`。result 结构的 `y/zf` 固定 float，GPU wrapper 也先把 `b/x/zi` 转 float，所以 `T` 只是输入分发类型。

#### 问题 8：是否使用 FFT、shared memory 或显式同步？

**参考答案 8：** 否。正式路径是每输出遍历 taps 的 direct FIR，没有 FFT/dlfft/共享缓存。wrapper 没有显式同步，启动契约交给 `launch_1d_kernel`。

#### 问题 9：smoke 覆盖什么，还缺什么？

**参考答案 9：** 564-647 覆盖空系数、二维 axis、无/有 `zi`、广播、`zf`、非法 axis/shape、手写值和 CPU/GPU 对照。缺复数、rank-3 手写值、$N<L-1$ 状态传播和大 taps 压力。本轮未实际运行测试。

#### 问题 10：`firfilter_direct_kernel<T>` 是当前核心吗？

**参考答案 10：** 不是。它虽定义在 1350-1392，但当前 wrapper 155-167 实际启动 axis-output/state kernel，resident 也启动 axis-output kernel，所以只能标为未接入当前调用链的 helper。

## 版本差异与原理不变量

| 项目 | V1 `fdd55ac` | 后续版本 | 原理不变量 |
| --- | --- | --- | --- |
| 算法 | 指定 axis direct FIR | 待追加 | $y[n]=\sum b[k]x[n-k]$ |
| 状态 | 前缀加 `zi`，尾部产生 `zf` | 待追加 | 分块续接语义 |
| CPU/GPU | double reference / FP32 一元素一线程 | 待追加 | full-index 到 `y/zf` 的分割 |
| dtype | 五实数输入固定 FP32 输出 | 待追加 | dtype 不改变 FIR 数学 |
| 复杂度 | $O(BNL)$，尾部最坏 $O(BL^2)$ | 待追加 | 结果必须等价于线性卷积 |

当前只有 V1，没有可声称的优化前后性能差异。后续若只改索引/缓存/并行，应记为“原理不变、实现优化”；若改为 FFT overlap-save，则需重新映射卷积等价原理。
