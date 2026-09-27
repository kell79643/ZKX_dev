# cusignal_cpp_sosfilt 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前实现](#v1当前实现fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前 | CPU 独立 double reference；GPU 每条信号一个线程、FP32 DF-II-T 动态状态 |

## V1：当前实现（fdd55ac）

### 1. 版本身份与读取边界

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 分支：`final-prep/benchmark-evidence-v1`
- 相关文件 dirty 状态：全部干净；整个 `ZKX` 工作树也无未提交修改。
- 读取方式：以 `git -C ZKX show <SHA>:<path>` 读取该提交快照，没有用当前工作区裸行号代替版本证据。
- 范围：接口/结果结构、layout 验证、CPU reference、GPU host wrapper、GPU kernel、显式实例化位置和直接 smoke test；未扩读其他算子。

### 2. 文件、符号与职责

| 职责 | SHA + `ZKX_dev/` 相对路径 | 符号 | 该提交行号 |
| --- | --- | --- | ---: |
| 选项/结果结构 | `fdd55ac...` + `ZKX/cusignal_cpp/src/filtering/filtering_typed.h` | `SosfiltOptions`、`SosfiltCpuResult`、`SosfiltDeviceResult` | 105-125 |
| 公共布局计划 | 同上 | `detail::SosfiltLayoutPlan` | 356-370 |
| GPU 对外声明 | 同上 | `sosfilt_device<T>` | 645-667 |
| CPU 对外声明 | 同上 | `sosfilt_typed_cpu<T>` | 897-910 |
| 公共校验/布局实现 | `fdd55ac...` + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` | `prepare_sosfilt_layout`、`validate_sos_coefficients` | 639-728 |
| CPU reference | 同上 | `sosfilt_typed_cpu<T>` | 2081-2142 |
| GPU host wrapper | `fdd55ac...` + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cu` | `sosfilt_device<T>` | 417-464 |
| GPU device 数学 helper | `fdd55ac...` + `ZKX/cusignal_cpp/src/filtering/filtering_kernels.cuh` | `sosfilt_apply_section` | 1288-1305 |
| GPU kernel | 同上 | `sosfilt_axis_kernel<T>` | 1307-1348 |
| GPU 显式实例化 | `fdd55ac...` + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cu` | `INST(T)` 中的 `sosfilt_device` | 683 |
| CPU 显式实例化 | `fdd55ac...` + `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` | `INST(T)` 中的 `sosfilt_typed_cpu` | 2781 |
| 直接 smoke test | `fdd55ac...` + `ZKX/cusignal_cpp/test/signal_processing/e3_filtering_type_smoke.cu` | `run_filtering_smoke<T>` 中 `sosfilt` 段 | 1094-1266 |

### 3. 接口、选项与结果结构逐行解释

#### 3.1 输入元数据

`fdd55ac...`，`ZKX/cusignal_cpp/src/filtering/filtering_typed.h:105-109`：

```cpp
struct SosfiltOptions {
    std::vector<int> shape;
    int axis = -1;
    std::vector<int> zi_shape;
};
```

- 第 105 行定义只保存 Host 元数据的结构体。
- 第 106 行保存 row-major 输入 shape；空 vector 代表按 `x.size()` 推导一维。
- 第 107 行默认沿末轴滤波，与 Python `axis=-1` 对应。
- 第 108 行保存可选初态 shape，不保存初态数据本身。
- 第 109 行结束结构体；分号是 C++ 类型定义必需结束符。

#### 3.2 CPU 与 GPU 的动态返回结构

`filtering_typed.h:111-125`：

```cpp
struct SosfiltCpuResult {
    std::vector<float> y;
    std::vector<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};

struct SosfiltDeviceResult {
    DeviceArray<float> y;
    DeviceArray<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};
```

- 两个结构逐字段对应；CPU 数据在 `std::vector<float>`，GPU 数据在 `DeviceArray<float>`。
- `y` 始终存在；`zf` 只有调用者传入 `zi` 时才有意义。
- shape 始终留在 Host，避免为仅用于契约描述的元数据分配 device 内存。
- `has_zf=false` 明确复现 Python 的动态返回：无 `zi` 对应只返回 `y`，有 `zi` 对应返回 `(y,zf)`。
- 输出固定 FP32；五种业务输入不是五种同 dtype 输出。

#### 3.3 布局计划

`filtering_typed.h:356-370`：

```cpp
struct SosfiltLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> state_shape;
    int axis = 0;
    int sections = 0;
    int sample_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    std::size_t state_size = 0;
};

SosfiltLayoutPlan prepare_sosfilt_layout(
    std::size_t input_size, std::size_t sos_size, int sections,
    const SosfiltOptions& options, bool has_zi, std::size_t zi_size);
```

- 两个 vector 保存规范化后的输入/状态 shape。
- `axis` 是已转正的轴；`sections` 是二阶节数；`sample_count` 是该轴长度。
- `inner_count` 是 axis 后各维乘积，`outer_count` 是 axis 前各维乘积。
- `line_count=outer_count*inner_count`，表示彼此独立的滤波序列数。
- `state_size=sections*line_count*2`，每节每条线两个 DF-II-T 状态。
- 后四行声明唯一公共 layout helper，CPU/GPU 调用同一验证逻辑，避免异常语义漂移。

#### 3.4 GPU 正式入口声明

`filtering_typed.h:645-667`：

```cpp
/**
 * @brief 在GPU上按固定cuSignal语义执行多维级联二阶节IIR滤波。
 * @tparam T `sos`、`x`和可选`zi`共同采用的FP32、FP16、INT32、INT16或INT8真实业务dtype。
 * FP32是cuSignal原生路径；FP16和整数为已批准的FP32合法路径扩展。
 * @param sos 调用者持有的连续device系数矩阵`[sections,6]`，列顺序为
 * `b0,b1,b2,a0,a1,a2`；每个`a0`必须精确等于1。
 * @param sections 正section数，公开限制不超过512，且不能超过所选axis样本数。
 * @param x 调用者持有的连续row-major device输入；元素数必须等于`options.shape`乘积。
 * @param out 函数重建的FP32结果；`y_shape`等于输入shape，传入`zi`时额外填充`zf`。
 * @param options 输入shape、axis及`zi_shape`；shape为空表示一维，`zi_shape`必须精确等于
 * `[sections] + x.shape`并把所选axis长度替换为2，未传`zi`时必须为空。
 * @param zi 可选device初态；非空指针保持cuSignal的双返回结构，且输入不会被修改。
 * @throws std::invalid_argument shape乘积、axis、section数、SOS宽度、`a0`、样本数或zi非法。
 * @details 默认stream上每条信号由一个GPU线程按Direct Form II Transposed顺序递推；状态为
 * 动态FP32全局workspace，不再使用固定32节局部数组。为验证`a0`会显式D2H读取系数；其余主计算
 * 均在GPU完成。无device FP64、FFT或RAND；函数返回前同步默认stream，`y/zf`随后可读取；
 * 输出分配、D2H校验和该同步均属于正式封装性能范围。
 */
template <typename T>
void sosfilt_device(
    const DeviceArray<T>& sos, int sections, const DeviceArray<T>& x,
    SosfiltDeviceResult& out, const SosfiltOptions& options = {},
    const DeviceArray<T>* zi = nullptr);
```

这个完整 Doxygen 块逐项规定 dtype、shape、异常、stream 和性能计入范围。`const DeviceArray<T>&` 表示不复制且不修改输入对象；`out` 是非常量引用，函数负责重建；`options={}` 与 `zi=nullptr` 复现默认一维、末轴、零初态。模板声明只定义接口，实际计算在 `.cu`。

#### 3.5 CPU reference 声明

`filtering_typed.h:897-910`：

```cpp
/**
 * @brief `sosfilt_device`逐参数对应的独立C++ CPU reference。
 * @tparam T 与GPU正式入口相同的五种真实业务dtype；所有值先进入FP32合法输入边界。
 * @param sos Host端连续`[sections,6]`系数，`a0`必须精确等于1。
 * @param sections、x、options、zi 与GPU入口具有相同shape、axis、状态和异常语义。
 * @return FP32 `y`，以及仅在传入`zi`时存在的FP32 `zf`和对应shape。
 * @details CPU使用独立的double Host workspace和状态递推作为数值对照，最终边界收敛为FP32；
 * 不调用GPU kernel、不读取GPU结果，也不复用GPU数学核心。无stream，输入、系数和状态均不被
 * 修改，并保持cuSignal的axis、zi/zf、shape和动态返回结构。
 */
template <typename T>
SosfiltCpuResult sosfilt_typed_cpu(
    const std::vector<T>& sos, int sections, const std::vector<T>& x,
    const SosfiltOptions& options = {}, const std::vector<T>* zi = nullptr);
```

注释明确 CPU 是独立 reference，不读取 GPU 结果；内部 double、边界 FP32。最后四行声明模板函数：Host vector 输入，返回值直接携带结果结构，可选 `zi` 用空指针表达缺省。

### 4. 公共布局与验证逐行解释

#### 4.1 基本大小、默认 shape 与 shape 乘积

`filtering_typed.cpp:639-666`：

```cpp
SosfiltLayoutPlan prepare_sosfilt_layout(
    std::size_t input_size, std::size_t sos_size, int sections,
    const SosfiltOptions& options, bool has_zi, std::size_t zi_size)
{
    if (sections < 1 || sections > 512)
        throw std::invalid_argument("sosfilt sections");
    if (sos_size != static_cast<std::size_t>(sections) * 6U)
        throw std::invalid_argument("sosfilt sos shape");
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("sosfilt input exceeds int indexing");

    SosfiltLayoutPlan plan;
    plan.sections = sections;
    plan.input_shape = options.shape;
    if (plan.input_shape.empty())
        plan.input_shape.push_back(static_cast<int>(input_size));

    long long product = 1;
    for (const int extent : plan.input_shape) {
        if (extent < 0)
            throw std::invalid_argument("sosfilt input shape");
        if (product != 0 && extent != 0 &&
            product > std::numeric_limits<int>::max() / extent)
            throw std::invalid_argument("sosfilt input shape overflow");
        product *= extent;
    }
    if (product != static_cast<long long>(input_size))
        throw std::invalid_argument("sosfilt input shape product");
```

函数签名收集数组元素数、节数和元数据。前三个 `if` 分别限制 `1..512` 节、严格六列和 int 索引容量。默认 shape 是 `{input_size}`。range-for 逐维拒绝负 extent，并在乘法前用除法做溢出门禁；最后要求 shape 乘积恰等于真实元素数。空维允许乘积变 0，但后续仍保持一致性。

#### 4.2 axis、独立信号布局与索引容量

`filtering_typed.cpp:668-691`：

```cpp
    int axis = options.axis;
    if (axis < 0) axis += static_cast<int>(plan.input_shape.size());
    if (axis < 0 || axis >= static_cast<int>(plan.input_shape.size()))
        throw std::invalid_argument("sosfilt axis");
    plan.axis = axis;
    plan.sample_count = plan.input_shape[axis];
    if (sections > plan.sample_count)
        throw std::invalid_argument("sosfilt sections exceed samples");

    long long inner = 1;
    for (std::size_t dimension = static_cast<std::size_t>(axis + 1);
         dimension < plan.input_shape.size(); ++dimension)
        inner *= plan.input_shape[dimension];
    long long outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        outer *= plan.input_shape[dimension];
    const long long lines = outer * inner;
    if (inner > std::numeric_limits<int>::max() ||
        outer > std::numeric_limits<int>::max() ||
        lines > std::numeric_limits<int>::max())
        throw std::invalid_argument("sosfilt layout exceeds int indexing");
    plan.inner_count = static_cast<int>(inner);
    plan.outer_count = static_cast<int>(outer);
    plan.line_count = static_cast<int>(lines);
```

负 axis 加 rank 转正，再做双边界验证。`sample_count` 是递推长度，并保持 cuSignal 的 `sections <= samples` 限制。两个循环分别求 axis 后/前维乘积；`lines` 是独立序列数。检查后才安全缩窄到 `int`，供 kernel 索引使用。

#### 4.3 构造状态 shape 并验证可选 `zi`

`filtering_typed.cpp:693-716`：

```cpp
    plan.state_shape.reserve(plan.input_shape.size() + 1);
    plan.state_shape.push_back(sections);
    plan.state_shape.insert(
        plan.state_shape.end(), plan.input_shape.begin(), plan.input_shape.end());
    plan.state_shape[static_cast<std::size_t>(axis + 1)] = 2;
    const unsigned long long state_size =
        static_cast<unsigned long long>(sections) *
        static_cast<unsigned long long>(plan.line_count) * 2ULL;
    if (state_size >
        static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("sosfilt state exceeds int indexing");
    plan.state_size = static_cast<std::size_t>(state_size);

    if (!has_zi) {
        if (!options.zi_shape.empty() || zi_size != 0)
            throw std::invalid_argument("sosfilt zi metadata without zi");
    } else {
        if (options.zi_shape != plan.state_shape)
            throw std::invalid_argument("sosfilt zi shape");
        if (zi_size != plan.state_size)
            throw std::invalid_argument("sosfilt zi shape product");
    }
    return plan;
}
```

`reserve` 只预留容量，`push_back` 放节数，`insert` 接上完整输入 shape；因为前插了一维，原 axis 在状态 shape 中变为 `axis+1`，再改为 2。状态元素数用无符号 64 位中间量计算并门禁。无指针时不允许伪造 `zi_shape/zi_size`；有指针时 shape 与元素数都必须精确匹配。最后返回完整计划并闭合函数。

#### 4.4 检查六列与 $a_0=1$

`filtering_typed.cpp:718-728`：

```cpp
void validate_sos_coefficients(
    const std::vector<float>& coefficients, int sections)
{
    if (sections < 1 ||
        coefficients.size() != static_cast<std::size_t>(sections) * 6U)
        throw std::invalid_argument("sosfilt sos shape");
    for (int section = 0; section < sections; ++section) {
        if (coefficients[static_cast<std::size_t>(section) * 6U + 3U] != 1.0F)
            throw std::invalid_argument("sosfilt sos a0 must equal one");
    }
}
```

该 helper 对已经转换到 FP32 的 Host 系数复核 shape，然后逐节访问偏移 `section*6+3`，即 $a_0$。使用精确 `!=1.0F`，不自动归一化、不使用容差；循环和函数的两个右花括号依次结束相应作用域。

### 5. CPU reference：从入口到最后一行

#### 5.1 类型、布局、系数与初态

`filtering_typed.cpp:2081-2101`：

```cpp
template <typename T>
SosfiltCpuResult sosfilt_typed_cpu(
    const std::vector<T>& sos, int sections, const std::vector<T>& x,
    const SosfiltOptions& options, const std::vector<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const bool has_zi = zi != nullptr;
    const auto plan = detail::prepare_sosfilt_layout(
        x.size(), sos.size(), sections, options, has_zi,
        has_zi ? zi->size() : 0U);

    std::vector<float> coefficients(sos.size());
    for (std::size_t index = 0; index < sos.size(); ++index)
        coefficients[index] = ld(sos[index]);
    detail::validate_sos_coefficients(coefficients, sections);

    std::vector<double> state(plan.state_size, 0.0);
    if (has_zi) {
        for (std::size_t index = 0; index < plan.state_size; ++index)
            state[index] = static_cast<double>(ld((*zi)[index]));
    }
```

模板和签名覆盖五种 `T`；`static_assert` 在编译期拒绝不在正式简单实数集合中的类型。指针非空决定动态返回。公共 plan 同时完成所有 shape/异常验证。`ld` 按类型策略把五型输入收敛到 FP32 合法边界，系数随后验证。状态 workspace 始终用 double 并零初始化；有初态时逐元素先经 FP32 边界再拓宽到 double。

#### 5.2 构造返回元数据

`filtering_typed.cpp:2103-2107`：

```cpp
    SosfiltCpuResult output;
    output.y.resize(x.size());
    output.y_shape = plan.input_shape;
    output.has_zf = has_zi;
    if (has_zi) output.zf_shape = plan.state_shape;
```

默认构造结果，按输入元素数分配 FP32 `y`，复制输入 shape。`has_zf` 与 `zi` 指针一致；单行 `if` 只在需要时保存状态 shape。

#### 5.3 N 维 row-major 索引和 DF-II-T 核心

`filtering_typed.cpp:2109-2135`：

```cpp
    for (int line = 0; line < plan.line_count; ++line) {
        const int outer = line / plan.inner_count;
        const int inner = line % plan.inner_count;
        for (int sample = 0; sample < plan.sample_count; ++sample) {
            const std::size_t signal_index = static_cast<std::size_t>(outer) *
                static_cast<std::size_t>(plan.sample_count) * plan.inner_count +
                static_cast<std::size_t>(sample) * plan.inner_count + inner;
            double value = static_cast<double>(ld(x[signal_index]));
            for (int section = 0; section < sections; ++section) {
                const std::size_t coefficient =
                    static_cast<std::size_t>(section) * 6U;
                const std::size_t state0 =
                    ((static_cast<std::size_t>(section) * plan.outer_count + outer)
                        * 2U) * plan.inner_count + inner;
                const std::size_t state1 = state0 + plan.inner_count;
                const double filtered = coefficients[coefficient] * value + state[state0];
                const double next0 = coefficients[coefficient + 1U] * value -
                    coefficients[coefficient + 4U] * filtered + state[state1];
                const double next1 = coefficients[coefficient + 2U] * value -
                    coefficients[coefficient + 5U] * filtered;
                state[state0] = next0;
                state[state1] = next1;
                value = filtered;
            }
            output.y[signal_index] = static_cast<float>(value);
        }
    }
```

最外层遍历所有独立线，商/余数恢复 axis 前后的组合索引。样本循环构造 row-major 线性索引；`value` 是当前节输入。节循环以 `section*6` 定位系数，以 `[section,outer,state,inner]` 的展平布局定位两个状态。

三式严格对应：

$$
v=b_0u+d_0,
$$

$$
d'_0=b_1u-a_1v+d_1,
$$

$$
d'_1=b_2u-a_2v.
$$

先计算两个 next，后覆盖 state，避免第二式读取已更新值；`value=filtered` 把本节输出传给下一节。所有节结束后缩窄到 FP32 输出。三个右花括号依次结束 sample、line 内部和 line 循环。

#### 5.4 返回 `zf` 并结束函数

`filtering_typed.cpp:2136-2142`：

```cpp
    if (has_zi) {
        output.zf.resize(plan.state_size);
        for (std::size_t index = 0; index < plan.state_size; ++index)
            output.zf[index] = static_cast<float>(state[index]);
    }
    return output;
}
```

有初态才分配并逐元素把 double 最终状态收敛为 FP32；这修复了 Python cuSignal kernel 未写回 `zf` 的缺口。最后按值返回结果并结束函数。

### 6. GPU host wrapper：从入口到最后一行

#### 6.1 模板、布局和 Host 端 $a_0$ 验证

`filtering_typed.cu:417-433`：

```cpp
template <class T>
void sosfilt_device(
    const DeviceArray<T>& sos, int sections, const DeviceArray<T>& x,
    SosfiltDeviceResult& output, const SosfiltOptions& options,
    const DeviceArray<T>* zi)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    const bool has_zi = zi != nullptr;
    const auto plan = detail::prepare_sosfilt_layout(
        x.size(), sos.size(), sections, options, has_zi,
        has_zi ? zi->size() : 0U);

    const auto host_sos = sos.to_host();
    std::vector<float> coefficients(host_sos.size());
    for (std::size_t index = 0; index < host_sos.size(); ++index)
        coefficients[index] = detail::SimpleSignalTypePolicy<T>::load(host_sos[index]);
    detail::validate_sos_coefficients(coefficients, sections);
```

签名与 CPU 参数逐项一致，只把容器换成 device 类型并通过输出引用返回。公共 plan 先验证纯元数据。随后 `to_host()` 显式 D2H 复制全部 SOS，仅为精确检查每个 $a_0$；类型策略把五型统一到 FP32。这段传输属于正式 wrapper 成本。

#### 6.2 输出、零初态或可写 `zf` 状态

`filtering_typed.cu:435-455`：

```cpp
    output = SosfiltDeviceResult{};
    output.y = DeviceArray<float>(x.size());
    output.y_shape = plan.input_shape;
    output.has_zf = has_zi;

    DeviceArray<float> zero_state;
    float* state = nullptr;
    if (has_zi) {
        output.zf = DeviceArray<float>(plan.state_size);
        output.zf_shape = plan.state_shape;
        if (plan.state_size != 0) {
            cuda_utils::launch_1d_kernel(
                detail::convert_signal_kernel<T,float>, plan.state_size,
                zi->data(), output.zf.data(), plan.state_size);
        }
        state = output.zf.data();
    } else {
        zero_state = DeviceArray<float>(plan.state_size);
        cuda_utils::zero_device(zero_state.data(), zero_state.size());
        state = zero_state.data();
    }
```

先用值初始化清掉调用者旧结果，再分配 FP32 `y`。若有 `zi`，直接把输出 `zf` 用作可写状态 workspace：转换 kernel 将五型初态转成 FP32，主 kernel 就地更新，最终自然得到 `zf`。若无 `zi`，建立零状态临时数组且不暴露。`state` 统一指向两条分支的实际 workspace。

#### 6.3 launch、同步和函数末行

`filtering_typed.cu:456-464`：

```cpp
    if (plan.line_count == 0) return;

    cuda_utils::launch_1d_kernel(
        detail::sosfilt_axis_kernel<T>,
        static_cast<std::size_t>(plan.line_count),
        sos.data(), sections, x.data(), plan.sample_count,
        plan.inner_count, plan.outer_count, state, output.y.data());
    cuda_utils::synchronize_stream();
}
```

无独立信号时提前返回，已构造的空结果仍合法。通用 1D launcher 按 `line_count` 个逻辑任务启动 kernel；参数完整传入系数、shape 分解、状态和输出。每个逻辑任务是一整条递推序列。显式同步保证返回后 `y/zf` 可立即读取；最后右花括号结束正式 GPU wrapper。

### 7. GPU device 代码：从第一行到最后一行

#### 7.1 单节 DF-II-T helper

`filtering_kernels.cuh:1288-1305`：

```cpp
template <typename Accumulator>
__device__ __forceinline__ Accumulator sosfilt_apply_section(
    Accumulator sample,
    Accumulator b0,
    Accumulator b1,
    Accumulator b2,
    Accumulator a1,
    Accumulator a2,
    Accumulator& state0,
    Accumulator& state1)
{
    const Accumulator output = b0 * sample + state0;
    const Accumulator next_state0 = b1 * sample - a1 * output + state1;
    const Accumulator next_state1 = b2 * sample - a2 * output;
    state0 = next_state0;
    state1 = next_state1;
    return output;
}
```

模板参数是累加类型；当前调用为 `float`。`__device__` 限定 GPU 内调用，`__forceinline__` 请求内联。六个值参数是样本和五个有效系数，两个引用参数允许原地更新状态。函数先用旧状态计算 output 和两个 next，再依次覆盖引用，最后返回本节输出；最后一行闭合函数。$a_0$ 不传入，因为 wrapper 已强制它为 1。

#### 7.2 kernel 参数、线程职责和边界

`filtering_kernels.cuh:1307-1325`：

```cpp
template <typename T>
__global__ void sosfilt_axis_kernel(
    const T* sos,
    int sections,
    const T* input,
    int sample_count,
    int inner_count,
    int outer_count,
    float* state,
    float* output)
{
    const int line = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int line_count = outer_count * inner_count;
    if (line >= line_count) return;

    const int outer = line / inner_count;
    const int inner = line % inner_count;
    const long long signal_base = static_cast<long long>(outer)
        * sample_count * inner_count + inner;
```

`T` 是输入/系数 dtype，状态与输出固定 float。`__global__` 表示 Host 可启动。标准一维全局线程索引映射到独立信号 line，越界线程立即返回。商余数恢复 outer/inner；`signal_base` 用 64 位计算这一条轴线在 row-major 数组中的首元素位置。

#### 7.3 每线程顺序处理整条信号

`filtering_kernels.cuh:1326-1348`：

```cpp
    for (int sample = 0; sample < sample_count; ++sample) {
        const long long signal_index = signal_base
            + static_cast<long long>(sample) * inner_count;
        float value = filtering_load(input[signal_index]);
        for (int section = 0; section < sections; ++section) {
            const T* coefficient = sos + static_cast<long long>(section) * 6;
            const long long state0 =
                ((static_cast<long long>(section) * outer_count + outer) * 2)
                * inner_count + inner;
            const long long state1 = state0 + inner_count;
            value = sosfilt_apply_section(
                value,
                filtering_load(coefficient[0]),
                filtering_load(coefficient[1]),
                filtering_load(coefficient[2]),
                filtering_load(coefficient[4]),
                filtering_load(coefficient[5]),
                state[state0],
                state[state1]);
        }
        output[signal_index] = value;
    }
}
```

外循环必须按时间顺序运行，因为 IIR 当前样本依赖上个样本状态；不同 line 才能并行。`filtering_load` 将五型输入提升为 FP32。内循环按节顺序级联，跳过系数索引 3 的 $a_0$。状态布局与 CPU 完全相同，两个引用把更新直接写回全局 `state`。节循环结束后写 FP32 `y`；两个右花括号依次结束 sample 循环和 kernel，这是阶段三核心代码最后一行。

### 8. 显式实例化与类型分发

- `filtering_typed.cu:683` 的共享 `INST(T)` 宏包含 `template void sosfilt_device(...)`；该宏随后针对项目五种业务类型展开。由于整行还包含其他 filtering 算子，本学习文档只记录当前声明，不复制无关符号。
- `filtering_typed.cpp:2781` 的共享 `INST(T)` 宏同理包含 `template SosfiltCpuResult sosfilt_typed_cpu(...)`。
- `static_assert(detail::is_simple_signal_input_v<T>)` 再次在实例内部限制类型集合。
- 五种输入：FP32 原生；FP16、INT32、INT16、INT8 经 `filtering_load`/类型策略进入 FP32 合法路径；CPU 内部用 double 作为独立数值 reference，最终 `y/zf` 都收敛到 FP32。

### 9. CPU/GPU/测试职责边界

| 层 | 做什么 | 不做什么 |
| --- | --- | --- |
| API 结构/声明 | 定义 shape、动态返回、dtype 与异常契约 | 不计算样本 |
| 公共 layout | 校验 shape/axis/section/zi，推导索引计划 | 不访问 device 样本 |
| CPU reference | 独立 double 状态递推，输出 FP32 | 不调用 GPU、不复用 device 数学 helper |
| GPU host wrapper | D2H 验证 $a_0$、分配/转换状态、launch、同步 | 不在 Host 计算滤波结果 |
| GPU helper | 计算一个样本通过一个二阶节的三式 | 不处理 shape 或线程索引 |
| GPU kernel | 每线程顺序处理一条 line 的所有样本和节 | 不跨线程共享状态、不使用 FFT/科学库 |
| smoke test | 比较已知答案、CPU/GPU、shape 和异常 | 不是核心实现或 benchmark |

### 10. 直接测试如何验证正确性

`e3_filtering_type_smoke.cu:1094-1266` 对每个模板业务类型执行同一组检查：

1. 1094-1100：构造两个 SOS 并搬到 device。
2. 1102-1116：分别验证二维 `axis=1` 与 `axis=0`，GPU 对照 CPU。
3. 1118-1130：传入非零 `zi`，检查 `y/zf` 双返回。
4. 1131-1140：验证 rank-3、中间轴和四维 `zf_shape`。
5. 1142-1155：33 节 identity SOS，证明已越过旧固定 32 节限制。
6. 1157-1219：CPU/GPU 都必须拒绝 $a_0\ne1$、错误 `zi_shape`、节数超过样本数、513 节。
7. 1220-1227：给出 axis1、axis0、含初态输出和最终状态的手写已知答案。
8. 1228-1266：回读 GPU，联合检查动态返回、shape、已知答案、CPU/GPU 一致及所有异常布尔量。

这些是源码中的测试意图；本学习阶段没有实际运行它们。没有读取到单独 benchmark 本算子的必要证据，因此不声称获得性能数字。

### 11. 原理—Python—CPU—GPU 三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $H(z)=\prod H_k(z)$ | `_sosfilt.cu` 中节流水 | `filtering_typed.cpp:2117-2132` | `filtering_kernels.cuh:1330-1345` | 都按节顺序级联；C++ GPU 是每 line 单线程，不是每节一线程 |
| $v=b_0u+d_0$ | Python kernel 69/86/105 | 2124 | helper 1299 | 数学相同 |
| $d'_0=b_1u-a_1v+d_1$ | Python kernel 70/87/106 | 2125-2126 | 1300 | 数学相同 |
| $d'_1=b_2u-a_2v$ | Python kernel 71/88/107 | 2127-2128 | 1301 | 数学相同 |
| $a_0=1$ | `_validate_sos:25-26` | 公共 validate 724-726 | wrapper D2H 后同一 validate | C++ 保持精确拒绝，不归一化 |
| 状态 shape | Python `x_zi_shape` | plan 693-704 | 同一 plan | C++ 与 Python契约一致 |
| 最终状态 | Python kernel 未写回 | 2136-2139 | `output.zf` 直接作为可写 state | C++ 有意修复 Python 实现缺口 |

### 12. 与 cuSignal Python 相同、等价替换和有意不同

- **相同**：六列顺序、$a_0=1$、多维任意 axis、`zi/zf` shape、无 `zi` 时零初态、动态返回结构、节数最多 512、节数不能超过样本数、DF-II-T 三式。
- **等价替换**：Python GPU 用“每节一个线程”的 block 内流水；C++ GPU 用“每条独立信号一个线程”，在线程内顺序遍历样本和节。输出数学等价，但并行结构、同步和内存访问不同。
- **有意不同**：C++ 支持五种业务输入并统一输出 FP32；Python fatbin 原生只有 float32/float64。ZQ500 正式路径不使用 device FP64。
- **有意修复**：C++ 将最终状态写回 `zf`；Python 23.08.00 kernel 的 `zi` 是 const，未写回。
- **实现优化**：动态全局状态取消旧 32 节局部数组上限；33 节测试锁定这一点。数学原理不变。

### 13. 并行、同步、内存和复杂度

- CPU：$O(LNK)$ 时间，$O(LK)$ 状态，$L=line_count$、$N=sample_count$、$K=sections$。
- GPU 总工作量同为 $O(LNK)$；并行度主要是 $L$ 条独立信号，单条 line 内因 IIR 状态依赖保持顺序。
- GPU 状态 `2LK` 个 FP32，全局内存；输出 `LN` 个 FP32。
- 每个样本每节读取 5 个有效系数并读写两个状态；系数未缓存到 shared/constant memory。
- kernel 内无 `__syncthreads()`，因为线程之间完全独立；wrapper 末尾有 stream 同步保证返回可读。
- 当 $L$ 很小而 $N,K$ 很大时 GPU 并行度有限；当 $L$ 大时可跨 line 扩展。
- wrapper 为验证 $a_0$ D2H 复制所有 SOS，且输出/状态分配、转换和同步都计入正式封装成本。

### 14. 精度、dtype、边界与 ZQ500 限制

- FP32 是正式原生计算/输出边界；FP16 和整数先提升为 FP32，不进行整数递推。
- CPU double workspace 是参考计算策略，不代表 GPU 使用 FP64；最终 CPU 参考也转 FP32。
- 极点接近单位圆时仍可能对 FP32 舍入敏感；SOS 降低风险但不保证无限精度。
- `a0` 必须精确为 1，调用者应在设计/转换阶段完成归一化。
- section 范围 `1..512`，且 `sections <= sample_count`。
- shape、state、line 相关值受 int 索引上限保护。
- 正式路径无 FFT、RAND 或科学库调用；ZQ500 不依赖 device double。

### 15. 仍未覆盖的风险

- smoke 覆盖 33 节但未展示 512 节正确性和资源/耗时压力。
- 未看到专门针对稳定极点、近单位圆极点、NaN/Inf、长序列误差积累的测试。
- 当前对系数 D2H 验证会成为小输入性能固定开销。
- 每 line 一线程在 line_count 很小时可能利用率低；无本阶段实测性能证据。
- smoke 是共享大测试中的一段，本阶段未执行，不能把源码中的 `ok[8]` 条件写成已通过的本次测试结果。

### 16. 阶段三自检问题与参考答案

#### 问题 1：V1 绑定哪个版本，为什么相关 dirty 状态重要？

**参考答案：**绑定完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，提交时间 `2026-08-15T23:45:58+08:00`，分支 `final-prep/benchmark-evidence-v1`。相关文件全部干净，所以文档中的行号和逻辑能由 `git show <SHA>:<path>` 重现；若相关文件 dirty，就不能把未提交内容冒充该 SHA 的正式版本。

#### 问题 2：CPU 与 GPU 的职责怎样区分？

**参考答案：**CPU `sosfilt_typed_cpu` 用独立 double Host 状态递推产生 FP32 reference；GPU wrapper 做元数据验证、D2H $a_0$ 检查、分配/转换、launch 和同步；GPU kernel 才处理 device 样本。依据分别是 `filtering_typed.cpp:2081-2142`、`filtering_typed.cu:417-464`、`filtering_kernels.cuh:1288-1348`。

#### 问题 3：C++ 中 DF-II-T 三式在哪里？

**参考答案：**CPU 在 2124-2128；GPU helper 在 1299-1301：$v=b_0u+d_0$，$d'_0=b_1u-a_1v+d_1$，$d'_1=b_2u-a_2v$。CPU/GPU 更新 state 后把 `v` 作为下一节输入。

#### 问题 4：为什么 GPU 是“一条信号一个线程”而不是“一个样本一个线程”？

**参考答案：**同一条信号的 $d_0,d_1$ 会从样本 $n$ 传到 $n+1$，时间维存在真依赖，不能无条件并行。`sosfilt_axis_kernel:1326-1347` 由每个线程顺序循环全部样本；不同 line 状态独立，才在 1318 行按线程并行。

#### 问题 5：五种 dtype 如何落实？

**参考答案：**模板 `T` 接受 FP32、FP16、INT32、INT16、INT8，由 `static_assert` 限定；`filtering_load`/`SimpleSignalTypePolicy` 转入 FP32，状态和输出固定 float。CPU 为独立参考在 FP32 输入边界后用 double workspace，最终也转 FP32。

#### 问题 6：`zi_shape` 为什么在 axis 位置使用 `axis+1`？

**参考答案：**状态 shape 先在输入 shape 前插入 `sections`，所以原输入第 `axis` 维右移一位。`prepare_sosfilt_layout:693-697` 先 push sections、再 insert 输入 shape，最后把 `[axis+1]` 改成 2。

#### 问题 7：C++ 如何修复 Python 的 `zf` 缺口？

**参考答案：**有 `zi` 时 GPU wrapper 将 `output.zf` 分配并初始化，然后让 `state` 指向它（443-450）；kernel 通过引用更新 `state[state0/state1]`（1343-1344），返回后它就是最终状态。CPU 也在 2136-2139 显式复制最终 state。Python kernel 的 `zi` 是 const 且没有写回。

#### 问题 8：有哪些同步？为什么 kernel 内没有 block 同步？

**参考答案：**线程各自拥有完整 line 和独立状态，不交换数据，所以无需 `__syncthreads()`。wrapper 463 行执行 stream 同步，保证函数返回后输出可读取；状态转换 kernel 与主 kernel 都在默认 stream，按 stream 顺序执行。

#### 问题 9：测试锁定了哪些边界？

**参考答案：**直接 smoke 覆盖 axis0/axis1、rank3 中间轴、非零 `zi/zf`、33 节、已知答案和 CPU/GPU 一致；同时要求 CPU/GPU 拒绝 $a_0\ne1$、错误 `zi_shape`、section 超过样本数和 513 节。证据为 `e3_filtering_type_smoke.cu:1094-1266`。

#### 问题 10：哪些结论不能从本阶段声称？

**参考答案：**不能声称本次已在 ZQ500 运行或测试通过，因为阶段三只读源码，没有构建、同步、连接服务器或执行测试；也不能给出性能数字。可以声称的是提交快照中存在这些实现与测试条件。

## 版本差异与原理不变量

当前只有 V1，暂无已提交的 V2 可比较。V1 相对 Python cuSignal 的主要工程差异是每 line 一线程、五类型转 FP32、动态全局状态和正确写回 `zf`；始终不变的数学原理是 SOS 级联与每节 DF-II-T 三式。
