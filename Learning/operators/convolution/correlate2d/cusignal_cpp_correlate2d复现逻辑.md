# `cusignal_cpp_correlate2d` 复现逻辑

## 版本索引

- [V1：当前正式 direct 实现（fdd55ac8415d）](#v1当前正式-direct-实现fdd55ac8415d)
- 当前实现：V1。
- 本文件尚无 V2/V3；后续只追加已提交的新版本，不覆盖 V1。

## V1：当前正式 direct 实现（fdd55ac8415d）

### 1. 版本身份与可追溯性门禁

| 项目 | 值 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| 相关源码 dirty 状态 | `clean` |
| 历史复核 | `git -C ZKX show <SHA>:<path>` |

相关实现文件为 `convolution_typed.h/.cpp/.cu` 和 `convolution_kernels.cuh`；直接验证包括 `e3_convolution_type_smoke.cu` 与 `test_all/operators/cases/convolution/correlate2d/`。本阶段只读 `ZKX/cusignal_cpp`，没有修改、构建、同步或运行测试。

### 2. 职责分层与调用链

```text
调用者
  ├─ Correlate2DOptions
  │    └─ correlate2d_output_shape(options)
  ├─ CPU reference
  │    └─ correlate2d_typed_cpu<T>
  │         ├─ map_correlate2d_index
  │         ├─ correlate2d_fillvalue<T>
  │         └─ 独立四重循环
  └─ GPU formal path
       ├─ Correlate2DWorkspace(options)
       │    └─ reset：shape/start/boundary/fill 预计算
       └─ correlate2d_device<T>
            ├─ workspace、shape、alias 门禁
            └─ launch_1d_kernel
                 └─ correlate2d_boundary_direct_kernel<T>
                      ├─ correlate2d_map_index
                      ├─ correlate_direct_accumulate<T>
                      └─ correlate_direct_store<T>
```

- API/参数层：`Correlate2DOptions`、`Correlate2DOutputShape`。
- Host workspace 层：保存预计算 shape、裁剪起点、边界码和五种 fill 表示。
- CPU/reference 核心：独立 Host 循环，不调用 GPU，也不读取 GPU 结果。
- GPU host wrapper：验证设备数组和 workspace，启动 kernel。
- GPU kernel：每线程一个输出元素，直接空间域乘加。
- 测试/benchmark：调用 CPU 与 GPU 并比较，不是核心实现。

### 3. 公共类型与接口逐行解释

以下定位均绑定完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`。

#### 3.1 `Correlate2DOptions`

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.h:193`，符号 `Correlate2DOptions`：

```cpp
struct Correlate2DOptions {
    int rows1 = 0;
    int cols1 = 0;
    int rows2 = 0;
    int cols2 = 0;
    std::string mode = "same";
    std::string boundary = "fill";
    double fillvalue = 0.0;
};
```

- 第 193 行定义聚合结构体。
- 第 194–197 行保存两个 row-major 矩阵的行列数。
- 第 198 行默认 `same`；Python `correlate2d` 默认 `full`，这是接口默认值差异。
- 第 199 行默认常数填充。
- 第 200 行以 Host double 接受填充值，后续按 T 转换。
- 第 201 行结束类型定义。

#### 3.2 输出 shape

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.h:203-218`，符号 `Correlate2DOutputShape`、`correlate2d_output_shape`：

```cpp
struct Correlate2DOutputShape {
    int rows = 0;
    int cols = 0;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    }
};

Correlate2DOutputShape correlate2d_output_shape(const Correlate2DOptions& options);
```

- 行列默认 0；`size()` 不改对象、不抛异常，并把行列转为 `size_t` 后相乘。
- `[[nodiscard]]` 提醒调用者不要忽略元素数。
- shape 函数按常量引用读 options，返回值对象。

#### 3.3 workspace

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.h:220-248`，符号 `Correlate2DWorkspace`：

```cpp
struct Correlate2DWorkspace {
    int rows1 = 0;
    int cols1 = 0;
    int rows2 = 0;
    int cols2 = 0;
    int out_rows = 0;
    int out_cols = 0;
    int row_start = 0;
    int col_start = 0;
    int boundary_code = 0;
    std::string mode = "same";
    std::string boundary = "fill";
    double fillvalue = 0.0;
    __half fill_fp16{};
    float fill_fp32 = 0.0F;
    std::int32_t fill_int32 = 0;
    std::int16_t fill_int16 = 0;
    std::int8_t fill_int8 = 0;
    bool integral_fill_valid = true;

    Correlate2DWorkspace() = default;
    explicit Correlate2DWorkspace(const Correlate2DOptions& options) { reset(options); }

    void reset(const Correlate2DOptions& options);
    [[nodiscard]] std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(out_rows) * static_cast<std::size_t>(out_cols);
    }
};
```

- 第 220–232 行缓存输入/输出 shape、full 坐标起点、边界码和公开参数。
- 第 233–237 行预转换五种 dtype 的 fill，避免 launch 时做不受控转换。
- 第 238 行记录整数 fill 是否有限。
- 第 240 行允许默认构造；第 241 行 `explicit` 构造后立即 `reset`。
- 第 243–247 行声明重置并计算输出元素数。

#### 3.4 CPU/GPU 声明

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.h:354-379`，符号 `correlate2d_typed_cpu<T>`、`correlate2d_device<T>`：

```cpp
template <typename T>
std::vector<T> correlate2d_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const Correlate2DOptions& options);

template <typename T>
void correlate2d_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    Correlate2DWorkspace& workspace);
```

- CPU 输入为只读 Host vector，新建同 T vector 返回。
- GPU 输入为只读 device array，结果写入调用者预分配的 `out`。
- GPU 返回 `void`，不隐式 D2H 或同步。

### 4. 类型策略与累加规则

#### 4.1 支持类型

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.h:21-54`，符号 `ConvolutionTypePolicy`：

```cpp
template <typename T>
struct ConvolutionTypePolicy {
    static constexpr bool supported =
        std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int8_t>;

    __host__ __device__ static float load(T value)
    {
        return static_cast<float>(value);
    }

    __host__ __device__ static T store(float value)
    {
        return static_cast<T>(value);
    }
};

template <>
struct ConvolutionTypePolicy<__half> {
    static constexpr bool supported = true;
    __host__ __device__ static float load(__half value) { return __half2float(value); }
    __host__ __device__ static __half store(float value) { return __float2half_rn(value); }
};

template <typename T>
inline constexpr bool is_convolution_input_v = ConvolutionTypePolicy<T>::supported;
```

- 通用模板允许 FP32、INT32、INT16、INT8；half 用显式特化加入。
- 普通 `load` 转 FP32、`store` 转回 T；half 用 CUDA 明确转换并 round-to-nearest。
- `is_convolution_input_v<T>` 给 CPU/GPU 的 `static_assert` 提供编译期门禁。
- 不支持 complex、float64、int64；这是 C++ 五种实数业务边界，不等于 Python cuSignal 的完整 dtype 集合。

#### 4.2 device 累加

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_kernels.cuh:11-15,52-73`，符号 `CorrelateDirectAccumulator`、`correlate_direct_accumulate`、`correlate_direct_store`：

```cpp
template <class T>
using CorrelateDirectAccumulator =
    std::conditional_t<std::is_integral_v<T>,
        compute_policy::modular_unsigned_t<T>,
        compute_policy::FmaFloatAccumulator>;

template <class T>
__device__ void correlate_direct_accumulate(
    CorrelateDirectAccumulator<T>& sum, T left, T right)
{
    if constexpr (std::is_integral_v<T>) {
        sum = compute_policy::modular_multiply_add<T>(sum, left, right);
    } else {
        sum.add_product(load(left), load(right));
    }
}

template <class T>
__device__ T correlate_direct_store(CorrelateDirectAccumulator<T> value)
{
    if constexpr (std::is_integral_v<T>) {
        return compute_policy::signed_from_modular_bits<T>(value);
    } else {
        return detail::ConvolutionTypePolicy<T>::store(value.value());
    }
}
```

- 整数 T 选择同宽无符号模累加器，定义二补码环绕而不是依赖有符号溢出。
- 非整数选择 FP32 FMA accumulator。
- `if constexpr` 在编译期删掉另一条分支。
- 整数最后把模位模式解释回同宽有符号 T；浮点/half 按类型策略写回。

### 5. 参数、shape、边界与 workspace

#### 5.1 边界字符串和容量

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:49-67`，符号 `correlate2d_boundary_code`、`checked_matrix_size`：

```cpp
int correlate2d_boundary_code(const std::string& boundary)
{
    if (boundary == "fill") return 0;
    if (boundary == "wrap") return 1;
    if (boundary == "symm") return 2;
    throw std::invalid_argument("typed correlate2d boundary must be fill, wrap, or symm");
}

std::size_t checked_matrix_size(int rows, int cols, const char* label)
{
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument(std::string("typed correlate2d negative ") + label + " dimension");
    }
    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("typed correlate2d ") + label + " exceeds 32-bit indexing");
    }
    return size;
}
```

- 三个 `if` 映射 0/1/2；包括 `reflect` 在内的其他字符串都拒绝。
- 容量 helper 先拒绝负维，再用 `size_t` 相乘；总元素超过 `INT_MAX` 时拒绝，因为 kernel 索引为 int。

#### 5.2 CPU 边界映射

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:101-114`，符号 `map_correlate2d_index`：

```cpp
int map_correlate2d_index(std::int64_t index, int size, int boundary_code)
{
    if (index >= 0 && index < size) return static_cast<int>(index);
    if (boundary_code == 0) return -1;
    if (boundary_code == 1) {
        std::int64_t wrapped = index % size;
        if (wrapped < 0) wrapped += size;
        return static_cast<int>(wrapped);
    }
    const std::int64_t period = static_cast<std::int64_t>(size) * 2;
    std::int64_t wrapped = index % period;
    if (wrapped < 0) wrapped += period;
    return static_cast<int>(wrapped < size ? wrapped : period - 1 - wrapped);
}
```

- 域内索引原样返回。
- fill 返回哨兵 -1，调用者据此使用 fillvalue。
- wrap 对 size 取模并修正负余数。
- symm 以 $2N$ 为周期，后半周期用 $2N-1-w$ 折返，是包含端点的 symmetric 延拓。

#### 5.3 fillvalue 转换

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:116-130`，符号 `correlate2d_fillvalue<T>`：

```cpp
template <typename T>
T correlate2d_fillvalue(const Correlate2DOptions& options)
{
    if (options.boundary != "fill") return fp32_boundary_store<T>(0.0F);
    if constexpr (std::is_integral_v<T>) {
        if (!std::isfinite(options.fillvalue)) {
            throw std::invalid_argument("typed correlate2d integral fillvalue must be finite");
        }
        return truncated_modular_store<T>(options.fillvalue);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half_rn(static_cast<float>(options.fillvalue));
    } else {
        return static_cast<float>(options.fillvalue);
    }
}
```

- 非 fill 返回 T 类型零，用户 fill 不影响 wrap/symm。
- 整数要求有限，截断后做同宽模转换，不饱和。
- half 先缩到 FP32，再转 half；float 也只保留 FP32。

#### 5.4 输出 shape

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:203-239`，符号 `correlate2d_output_shape`：

```cpp
Correlate2DOutputShape correlate2d_output_shape(const Correlate2DOptions& options)
{
    if (options.mode != "full" && options.mode != "same" && options.mode != "valid") {
        throw std::invalid_argument("typed correlate2d mode must be full, same, or valid");
    }
    (void)correlate2d_boundary_code(options.boundary);
    (void)checked_matrix_size(options.rows1, options.cols1, "in1");
    (void)checked_matrix_size(options.rows2, options.cols2, "in2");
    if (options.rows1 == 0 || options.cols1 == 0 ||
        options.rows2 == 0 || options.cols2 == 0) {
        return {};
    }

    Correlate2DOutputShape shape;
    if (options.mode == "full") {
        if (options.rows1 > std::numeric_limits<int>::max() - options.rows2 + 1 ||
            options.cols1 > std::numeric_limits<int>::max() - options.cols2 + 1) {
            throw std::invalid_argument("typed correlate2d full shape exceeds 32-bit indexing");
        }
        shape.rows = options.rows1 + options.rows2 - 1;
        shape.cols = options.cols1 + options.cols2 - 1;
    } else if (options.mode == "same") {
        shape.rows = options.rows1;
        shape.cols = options.cols1;
    } else {
        const bool in1_dominates = options.rows1 >= options.rows2 && options.cols1 >= options.cols2;
        const bool in2_dominates = options.rows2 >= options.rows1 && options.cols2 >= options.cols1;
        if (!in1_dominates && !in2_dominates) {
            throw std::invalid_argument(
                "typed correlate2d valid mode requires one input to dominate in every dimension");
        }
        shape.rows = std::abs(options.rows1 - options.rows2) + 1;
        shape.cols = std::abs(options.cols1 - options.cols2) + 1;
    }
    (void)checked_matrix_size(shape.rows, shape.cols, "output");
    return shape;
}
```

- mode、boundary、输入容量即使面对空输入也先校验。
- 任一维为零返回 `{0,0}`。
- full 先做 int 加法上界门禁，再算两维和减一。
- same 始终保持原 in1 shape。
- valid 要求一个输入逐维支配，交叉 shape 拒绝；输出为绝对差加一，因此无须真的交换数组。
- 返回前再次检查输出适合 32 位 device 索引。

#### 5.5 workspace reset

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:241-282`，符号 `Correlate2DWorkspace::reset`：

```cpp
void Correlate2DWorkspace::reset(const Correlate2DOptions& options)
{
    const Correlate2DOutputShape output = correlate2d_output_shape(options);
    rows1 = options.rows1;
    cols1 = options.cols1;
    rows2 = options.rows2;
    cols2 = options.cols2;
    mode = options.mode;
    boundary = options.boundary;
    fillvalue = options.fillvalue;
    boundary_code = mode == "valid" ? 0 : correlate2d_boundary_code(boundary);
    out_rows = output.rows;
    out_cols = output.cols;
    if (output.size() == 0) {
        row_start = 0;
        col_start = 0;
    } else if (mode == "full") {
        row_start = 0;
        col_start = 0;
    } else if (mode == "same") {
        row_start = rows2 / 2;
        col_start = cols2 / 2;
    } else {
        row_start = std::min(rows1, rows2) - 1;
        col_start = std::min(cols1, cols2) - 1;
    }

    fill_fp32 = mode == "valid" ? 0.0F : static_cast<float>(fillvalue);
    fill_fp16 = __float2half_rn(fill_fp32);
    integral_fill_valid = mode == "valid" || boundary != "fill" || std::isfinite(fillvalue);
    if (mode != "valid" && integral_fill_valid && boundary == "fill") {
        fill_int32 = truncated_modular_store<std::int32_t>(fillvalue);
        fill_int16 = truncated_modular_store<std::int16_t>(fillvalue);
        fill_int8 = truncated_modular_store<std::int8_t>(fillvalue);
    } else {
        fill_int32 = 0;
        fill_int16 = 0;
        fill_int8 = 0;
    }
}
```

- 先复用统一 shape 门禁，再复制公开参数。
- valid 强制边界码为 0，因为合法窗口不访问域外。
- `row_start/col_start` 是裁剪结果在 full 相关坐标中的起点：full/空为 0，same 为 `floor(K/2)`，valid 为较小维减一。
- same 的整数除法落实偶数 kernel 固定偏移。
- fill 被预转换到五种 dtype；整数 NaN/Inf 只记录 invalid，GPU入口随后抛错。
- workspace 不含 device scratch 或 FFT plan。

### 6. CPU reference 核心逐行解释

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:415-486`，符号 `correlate2d_typed_cpu<T>`。

#### 6.1 签名、shape 与边界准备

```cpp
template <typename T>
std::vector<T> correlate2d_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const Correlate2DOptions& options)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate2d business input dtype");
    const Correlate2DOutputShape output = correlate2d_output_shape(options);
    if (in1.size() != checked_matrix_size(options.rows1, options.cols1, "in1") ||
        in2.size() != checked_matrix_size(options.rows2, options.cols2, "in2")) {
        throw std::invalid_argument("correlate2d_typed_cpu shape mismatch");
    }
    if (output.size() == 0) return {};

    int row_start = 0;
    int col_start = 0;
    if (options.mode == "same") {
        row_start = options.rows2 / 2;
        col_start = options.cols2 / 2;
    } else if (options.mode == "valid") {
        row_start = std::min(options.rows1, options.rows2) - 1;
        col_start = std::min(options.cols1, options.cols2) - 1;
    }
    const int boundary_code = options.mode == "valid"
        ? 0
        : correlate2d_boundary_code(options.boundary);
    const T fill = options.mode == "valid"
        ? fp32_boundary_store<T>(0.0F)
        : correlate2d_fillvalue<T>(options);
```

- 模板函数返回同 T Host vector，`static_assert` 编译期拒绝其他类型。
- 统一 helper 校验参数和输出；两个 vector 元素数必须与二维 shape 一致。
- 空输出提前返回。
- full 保持 start 0；same 取模板半尺寸；valid 取每轴较小维减一。
- valid 固定 boundary 0 和 fill 0，但合法 valid 索引不会真正命中 fill。

#### 6.2 四重循环

```cpp
    std::vector<T> out(output.size());
    for (int out_r = 0; out_r < output.rows; ++out_r) {
        for (int out_c = 0; out_c < output.cols; ++out_c) {
            const int full_r = row_start + out_r;
            const int full_c = col_start + out_c;
            using ModularAccumulator = compute_policy::modular_unsigned_t<T>;
            ModularAccumulator modular_acc = 0;
            float fp32_acc = 0.0F;
            for (int kr = 0; kr < options.rows2; ++kr) {
                const std::int64_t source_r = static_cast<std::int64_t>(full_r) -
                    (options.rows2 - 1 - kr);
                const int in_r = map_correlate2d_index(
                    source_r, options.rows1, boundary_code);
                for (int kc = 0; kc < options.cols2; ++kc) {
                    const std::int64_t source_c = static_cast<std::int64_t>(full_c) -
                        (options.cols2 - 1 - kc);
                    const int in_c = map_correlate2d_index(
                        source_c, options.cols1, boundary_code);
                    const T sample = in_r < 0 || in_c < 0
                        ? fill
                        : in1[static_cast<std::size_t>(in_r) * options.cols1 + in_c];
                    const T coefficient = in2[static_cast<std::size_t>(kr) * options.cols2 + kc];
                    if constexpr (std::is_integral_v<T>) {
                        modular_acc = compute_policy::modular_multiply_add<T>(
                            modular_acc, sample, coefficient);
                    } else {
                        fp32_acc = std::fma(
                            detail::ConvolutionTypePolicy<T>::load(sample),
                            detail::ConvolutionTypePolicy<T>::load(coefficient),
                            fp32_acc);
                    }
                }
            }
```

- 外两层枚举输出，内两层枚举模板，总复杂度 $O(O_rO_cR_2C_2)$。
- full 坐标等于裁剪起点加输出坐标。
- `full_r-(rows2-1-kr)` 和列方向同式承担相关方向；模板本身正序读取。
- 64 位临时索引避免减法中间溢出。
- 映射返回 -1 时使用 fill，否则按 `row*cols+col` 读取 row-major in1。
- 整数走模乘加；float/half 先 load 到 FP32，用 `std::fma` 累加。

#### 6.3 写回和实例化

```cpp
            const std::size_t index = static_cast<std::size_t>(out_r) * output.cols + out_c;
            if constexpr (std::is_integral_v<T>) {
                out[index] = signed_from_bits<T>(modular_acc);
            } else {
                out[index] = fp32_boundary_store<T>(fp32_acc);
            }
        }
    }
    return out;
}

#define INSTANTIATE_CONV_CPU(T) \
    template std::vector<T> correlate_typed_cpu(const std::vector<T>&, const std::vector<T>&, const std::string&, CorrelateMethod); \
    template std::vector<T> correlate_typed_cpu(const std::vector<T>&, const std::vector<T>&, const CorrelateNDOptions&); \
    template std::vector<T> correlate2d_typed_cpu(const std::vector<T>&, const std::vector<T>&, const Correlate2DOptions&)

INSTANTIATE_CONV_CPU(float);
INSTANTIATE_CONV_CPU(__half);
INSTANTIATE_CONV_CPU(std::int32_t);
INSTANTIATE_CONV_CPU(std::int16_t);
INSTANTIATE_CONV_CPU(std::int8_t);

#undef INSTANTIATE_CONV_CPU
```

- 输出仍按 row-major 展平。
- 整数恢复同宽有符号位模式；浮点/half 按类型策略写回。
- 宏第三条是当前算子，五次调用生成五种链接实体；`#undef` 防止宏泄漏。

### 7. GPU host wrapper 逐行解释

#### 7.1 workspace fill 选择

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cu:116-130`，符号 `correlate2d_workspace_fill<T>`：

```cpp
template <typename T>
T correlate2d_workspace_fill(const Correlate2DWorkspace& workspace)
{
    if constexpr (std::is_same_v<T, __half>) {
        return workspace.fill_fp16;
    } else if constexpr (std::is_same_v<T, float>) {
        return workspace.fill_fp32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return workspace.fill_int32;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return workspace.fill_int16;
    } else {
        return workspace.fill_int8;
    }
}
```

- 编译期分支从 workspace 取与 T 完全一致的预转换标量。
- 最后 `else` 在类型门禁下只可能是 int8。
- 标量作为 kernel 参数按值传递，不分配 device scratch。

#### 7.2 正式 GPU 入口

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cu:382-425`，符号 `correlate2d_device<T>`：

```cpp
template <typename T>
void correlate2d_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    Correlate2DWorkspace& workspace)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate2d business input dtype");
    Correlate2DOptions options;
    options.rows1 = workspace.rows1;
    options.cols1 = workspace.cols1;
    options.rows2 = workspace.rows2;
    options.cols2 = workspace.cols2;
    options.mode = workspace.mode;
    options.boundary = workspace.boundary;
    options.fillvalue = workspace.fillvalue;
    const Correlate2DOutputShape expected = correlate2d_output_shape(options);
    Correlate2DWorkspace validated(options);
```

- 从 workspace 重建公开 options，再重新计算 expected 和 validated，避免盲信被调用者修改过的派生字段。

```cpp
    if (workspace.out_rows != validated.out_rows || workspace.out_cols != validated.out_cols ||
        workspace.row_start != validated.row_start || workspace.col_start != validated.col_start ||
        workspace.boundary_code != validated.boundary_code ||
        in1.size() != static_cast<std::size_t>(workspace.rows1) * workspace.cols1 ||
        in2.size() != static_cast<std::size_t>(workspace.rows2) * workspace.cols2 ||
        out.size() != expected.size()) {
        throw std::invalid_argument("typed correlate2d workspace mismatch");
    }
```

- 复合条件核对输出 shape、裁剪起点、边界码、两输入容量和输出容量。
- 任一不一致都在 launch 前拒绝，避免越界或旧 workspace 误用。

```cpp
    if constexpr (std::is_integral_v<T>) {
        if (!validated.integral_fill_valid) {
            throw std::invalid_argument("typed correlate2d integral fillvalue must be finite");
        }
    }
    if (!out.empty() && (out.data() == in1.data() || out.data() == in2.data())) {
        throw std::invalid_argument("typed correlate2d does not permit input/output aliasing");
    }
    if (out.empty()) return;
```

- 只有整数实例编译有限性检查。
- 非空输出不得与任一输入首地址相同，算法不承诺 in-place。
- 空输出完成门禁后直接返回，不启动 kernel。

```cpp
    cuda_utils::launch_1d_kernel(
        convolution_detail::correlate2d_boundary_direct_kernel<T>,
        expected.size(),
        in1.data(), validated.rows1, validated.cols1,
        in2.data(), validated.rows2, validated.cols2,
        out.data(), validated.out_rows, validated.out_cols,
        validated.row_start, validated.col_start, validated.boundary_code,
        correlate2d_workspace_fill<T>(validated));
}
```

- 通用 launcher 按输出元素数决定一维 grid/block。
- 实参依次传两输入、输出、裁剪起点、边界码和 T 类型 fill。
- 没有 FFT、workspace device 指针、H2D/D2H 或同步。
- SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 的 `ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cu:427-439`，符号 `INSTANTIATE_CONV_GPU`，为五种 dtype 生成 GPU 实体。

### 8. GPU kernel 逐行解释

#### 8.1 device 边界映射

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_kernels.cuh:248-264`，符号 `correlate2d_map_index`：

```cpp
__device__ inline int correlate2d_map_index(
    std::int64_t index,
    int size,
    int boundary_code)
{
    if (index >= 0 && index < size) return static_cast<int>(index);
    if (boundary_code == 0) return -1;
    if (boundary_code == 1) {
        std::int64_t wrapped = index % size;
        if (wrapped < 0) wrapped += size;
        return static_cast<int>(wrapped);
    }
    const std::int64_t period = static_cast<std::int64_t>(size) * 2;
    std::int64_t wrapped = index % period;
    if (wrapped < 0) wrapped += period;
    return static_cast<int>(wrapped < size ? wrapped : period - 1 - wrapped);
}
```

它逐行复制 CPU helper 的公式，但带 `__device__ inline`，可内联进最内层 kernel 循环。CPU/GPU 是两个独立实现，不是 CPU 调用 GPU。

#### 8.2 boundary-aware direct kernel

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_kernels.cuh:266-308`，符号 `correlate2d_boundary_direct_kernel<T>`：

```cpp
template <class T>
__global__ void correlate2d_boundary_direct_kernel(
    const T* in1, int rows1, int cols1,
    const T* in2, int rows2, int cols2,
    T* output, int output_rows, int output_cols,
    int row_start, int col_start, int boundary_code, T fillvalue)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = output_rows * output_cols;
    if (index >= total) return;
    const int output_row = index / output_cols;
    const int output_col = index % output_cols;
    const int full_row = row_start + output_row;
    const int full_col = col_start + output_col;
    CorrelateDirectAccumulator<T> sum{};
```

- 两输入只读、输出可写；shape/start/boundary/fill 按值传递。
- 一维全局线程 index 对应一个展平输出元素，尾部线程返回。
- 除法与取模恢复 row-major 行列，再加 start 得到 full 坐标。
- 累加器按 T 选择模整数或 FP32 FMA，并以零初始化。

```cpp
    for (int kernel_row = 0; kernel_row < rows2; ++kernel_row) {
        const std::int64_t source_row = static_cast<std::int64_t>(full_row) -
            (rows2 - 1 - kernel_row);
        const int input_row = correlate2d_map_index(source_row, rows1, boundary_code);
        for (int kernel_col = 0; kernel_col < cols2; ++kernel_col) {
            const std::int64_t source_col = static_cast<std::int64_t>(full_col) -
                (cols2 - 1 - kernel_col);
            const int input_col = correlate2d_map_index(source_col, cols1, boundary_code);
            const T sample = input_row < 0 || input_col < 0
                ? fillvalue
                : in1[input_row * cols1 + input_col];
            correlate_direct_accumulate<T>(
                sum, sample, in2[kernel_row * cols2 + kernel_col]);
        }
    }
    output[index] = correlate_direct_store<T>(sum);
}
```

- 双循环覆盖第二输入全部元素。
- source 公式与 CPU 一致，模板正序读取，坐标表达式承担相关方向。
- 每轴先映射；fill 任一轴越界时使用常数，否则 row-major 读取 in1。
- 每线程只写唯一输出，不需要原子操作或线程间同步。
- store 将累加器转换回公开 dtype。

#### 8.3 并行、内存与同步

- grid/block：统一 launcher 按 `expected.size()` 产生一维覆盖。
- 每线程：一个输出元素，线程内串行完成 $R_2C_2$ 次乘加。
- 布局：连续 row-major，索引 `row*cols+col`。
- 搬运：正式入口无 H2D/D2H，调用者准备 `DeviceArray`。
- scratch：无 device 临时缓冲区；workspace 只有 Host 标量。
- 同步：入口不同步，读取前由调用者同步。
- 平台库：不调用 FFTInterface、dlfft 或 RAND。

### 9. 数学—Python—CPU—GPU 四方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $R[u,v]=\sum_{a,b}\widetilde X[\cdot]H[a,b]$（实数） | cuSignal基准 `correlate.py:246-253` 与旧 CUDA 双循环 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:445-476`，`correlate2d_typed_cpu<T>` | 同 SHA，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_kernels.cuh:292-306`，`correlate2d_boundary_direct_kernel<T>` | C++正式范围仅实数；模板方向由 source 公式体现 |
| full | 两维和减一 | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:217-223`，`correlate2d_output_shape` | 同 SHA，`convolution_typed.cpp:258-260`，`Correlate2DWorkspace::reset` | 相同 |
| same | 原 in1 shape、偶数固定偏移 | 同 SHA，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:224-226,431-433`，shape/CPU函数 | 同 SHA，`convolution_typed.cpp:261-263`，workspace reset | C++ options 默认 same，Python函数默认 full |
| valid | Python可能交换再反转 | 同 SHA，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:228-235,434-436`，shape/CPU函数 | 同 SHA，`convolution_typed.cpp:264-266` 与 kernel source 公式 | C++避免实际交换，直接覆盖两种实数支配方向 |
| fill | Python显式 constant pad | `map=-1` 后选 fill | kernel 三元表达式 | C++按需映射，不物化 padded 数组 |
| wrap | `cp.pad(wrap)` | 模 size | 相同 device helper | 等价实现 |
| symm | `cp.pad(symmetric)` | 周期 $2N$ 折返 | 相同 device helper | edge-inclusive symmetric |
| INT32 | Python旧 kernel以 T 乘加 | 显式模 $2^{32}$ | 模 accumulator | C++消除有符号溢出未定义性 |
| FP16/窄整数 | Python direct不原生支持 | 项目批准扩展 | 项目批准扩展 | 扩展，不是上游原生能力 |
| FFT/NCC | 未采用 | 未采用 | 未采用 | 都是未归一化 direct 相关 |

### 10. 相同、等价替换和有意不同

相同：

- rank 固定二维；支持 `full/same/valid`、`fill/wrap/symm`、标量 fill。
- valid 要求某一输入逐维支配，same 保持原 in1 shape。
- 偶数模板按 floor-half 固定偏移。
- 都是未归一化 direct 相关，不走 FFT。

等价替换：

- Python显式 pad；C++在取样时映射边界。
- Python valid 可能交换/反转；C++用绝对差 shape 与 full 坐标起点处理。
- Python二维 grid、C++一维展平 grid 都是每线程一个输出。

有意不同：

- C++只支持 FP32、FP16、INT32、INT16、INT8 实数同型输入；Python还支持其他编译 dtype 和复数。
- C++ options 默认 same；Python API 默认 full。
- C++空维稳定返回 `{0,0}`，显式限制 32 位 kernel 索引并拒绝 alias。
- C++明确整数环绕和扩展 dtype 计算边界。
- C++不覆盖 Python复数 valid 交换风险，因为没有复数公开实例。

### 11. 精度、dtype 与 ZQ500 限制

| dtype | 计算 | 输出 | 说明 |
| --- | --- | --- | --- |
| FP32 | FP32 FMA | FP32 | 有限精度；FMA减少一次乘加舍入 |
| FP16 | load到FP32累加 | FP16 RN | 批准扩展，存在输出量化 |
| INT32 | 模 $2^{32}$ 乘加 | INT32位模式 | 原生路径，环绕不饱和 |
| INT16 | 同宽模乘加 | INT16 | 批准扩展，环绕不饱和 |
| INT8 | 同宽模乘加 | INT8 | 批准扩展，范围最窄 |

#### 11.1 代码与合同的 dtype 不一致

`operator_contracts.cmake:92-93`、公开头文件注释和 formal probe 元数据把 INT16/INT8 描述为“FP32 direct 累加后 round 并同宽环绕”。但 V1 实际源码中：

- CPU `correlate2d_typed_cpu:466-468` 对所有整数 T 调用 `modular_multiply_add<T>`；
- GPU `CorrelateDirectAccumulator` 对所有 `std::is_integral_v<T>` 选择 `modular_unsigned_t<T>`；
- GPU `correlate_direct_accumulate:58-60` 同样对所有整数直接模乘加。

因此 V1 的 INT16/INT8 实际计算是同宽整数模乘加，不是文档宣称的 FP32 accumulator。E3 和 formal 的 CPU/GPU exact-match 会让两边采用相同整数分支，能验证二者一致，却不能证明它们符合“FP32合法路径”合同。该差异需要在独立开发任务中决定是修代码、修合同还是增加真正独立的 Python/数学 oracle；学习阶段保持源码只读。

- 正式 kernel 不依赖 device FP64。
- 不依赖 FFTInterface/dlfft，backend 和 fft dtype 为 `not_applicable`。
- 无隐式 stream 同步，ZQ500 调用者读取前必须同步。
- 本阶段没有连接 ZQ500，只记录源码和已有证据。

### 12. 测试如何验证正确性

#### 12.1 E3 smoke

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX_dev/ZKX/cusignal_cpp/test/signal_processing/e3_convolution_type_smoke.cu:345-535`，符号 `run_type<T>`、`main`：

- 五种 dtype 均以 CPU reference 对照 GPU。
- 覆盖 full/fill、$2\times2$ 偶数 kernel same offset、非零 fill、wrap、symm。
- 覆盖第二输入更大的 valid。
- 验证交叉 valid shape、非法 boundary、输出 alias 被拒绝。
- 验证空维，以及整数 NaN fill 在 CPU/GPU 均被拒绝。
- `main` 对五类型各记录 correlate 与 correlate2d，总计期望 10/10。

#### 12.2 阶段08 formal matrix

`test_all/operators/cases/convolution/correlate2d/` 与 configs 定义：

- 五 dtype × 三规模，共 15 case；warmup 20、measured 100，自动落盘。
- $10^2$ 请求 same/fill；$10^3$ 请求 full/wrap，含 $5\times4$ 非方形模板；$10^4$ 请求 valid/symm，但 valid 忽略 boundary。
- FP32/FP16 使用阈值；三种整数要求 exact match 和零 mismatch。
- CPU independent reference 与 GPU 分别生成 digest、误差和时序指标。

仓库归档记录显示 formal matrix 在 `e458c69972d35a4506e1db9ca973933f634bc515` 获得 `OBJECT_PASS`。该提交到当前 SHA 的四个实现文件无差异；测试目录仅 CMake 和 deep verifier 有治理调整。因此旧结果证明同一实现源码曾通过，但本阶段没有用当前包装重新执行，不能声称本轮重新验证通过。

尚存风险：

- formal 把三种 mode 与三种 boundary 固定配对，不是全笛卡尔积。
- valid/symm 不激活 symm；活跃 symm 依赖 E3 小规模覆盖。
- 不覆盖 complex，也不验证 Python 复数语义。
- 最大正式模板仅 $7\times5$，不能代表超大模板。
- INT16/INT8 的 CPU 与 GPU 都走相同模整数规则，现有 exact-match 不能发现其与 FP32 合同的偏差。

### 13. 复杂度与性能

设输出为 $O_r\times O_c$、模板为 $R_2\times C_2$：

$$
T=\Theta(O_rO_cR_2C_2),\qquad S_{out}=\Theta(O_rO_c).
$$

- CPU：四重循环，单线程独立 reference。
- GPU：输出并行，每线程仍串行做 $R_2C_2$ 次乘加。
- workspace：$O(1)$ Host 标量，无规模相关 scratch。
- 边界按需映射，省掉显式 padded 输入空间。
- 相邻线程重复读取重叠窗口；无显式 shared-memory tile/constant template cache。
- 大模板可能比 FFT 慢，但正式合同固定 direct 以完整保持边界语义。

### 14. 自检问题

1. V1 的完整 SHA、分支与 dirty 状态是什么？
2. Options 与 workspace 的职责有何区别？
3. CPU 是否调用 GPU，GPU 是否读取 CPU 结果？
4. full/same/valid 的 shape 和 start 如何计算？
5. C++为何无需像 Python valid 那样实际交换输入？
6. fill/wrap/symm 如何映射索引？
7. 每个 GPU 线程负责什么？
8. INT32 与 FP16 的计算边界有何区别？
9. GPU launch 前有哪些门禁？
10. 是否使用 FFT、scratch、隐式传输或同步？
11. C++与Python的 dtype/default mode 有何差异？
12. 测试覆盖和未覆盖风险分别是什么？

### 15. 参考答案

1. `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`final-prep/benchmark-evidence-v1`，相关文件 clean；SHA把解释绑定不可变快照。
2. Options 保存原始参数；workspace 保存校验后 shape、裁剪起点、边界码和五种 fill 表示。
3. 不互调。SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41` 中，`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp:416-486` 的 `correlate2d_typed_cpu<T>` 只遍历 Host vector；`ZKX_dev/ZKX/cusignal_cpp/src/convolution/convolution_typed.cu:383-425` 的 `correlate2d_device<T>` 只 launch device kernel。
4. full：和减一、start 0；same：原 in1、start `rows2/2,cols2/2`；valid：绝对差加一、start 较小维减一。
5. 绝对差 shape 与统一 full 坐标 source 公式已覆盖两种支配方向，实数无需交换数组。
6. fill 域外返回 -1；wrap 为修正负余数后的模 N；symm 模 $2N$ 后将后半周期折回。
7. 一个展平输出；除法/取模恢复二维坐标，线程内遍历完整模板。
8. INT32同宽模乘加并恢复位模式；FP16转FP32 FMA，最后 round-to-nearest 回 half。另需指出 INT16/INT8 源码也走模整数分支，与合同声称的 FP32 扩展不一致。
9. workspace派生字段、三数组容量、整数 fill 有限性、alias；空输出不 launch。
10. 均不使用；只有 boundary-aware direct kernel，搬运和同步由调用者负责。
11. C++是五种实数且默认 same；Python还支持复数/其他编译类型且默认 full。
12. E3覆盖小规模边界、偶数偏移、逆向 valid、异常和空维；formal覆盖五 dtype 三规模。仍缺 mode×boundary 全组合、complex、超大模板和本阶段新运行。

## 版本差异与原理不变量

当前只有 V1。

| 项目 | V1 | 原理不变量 |
| --- | --- | --- |
| 数学 | boundary-aware direct | 每个位移对模板窗口逐项乘加 |
| CPU | 独立四重循环 | 与GPU共享 shape/边界/dtype 合同 |
| GPU | 每线程一输出 | 未归一化互相关公式不变 |
| 边界 | fill/wrap/symm 索引映射 | 只改变 $\widetilde X$ 域外定义 |
| dtype | 五种实数同型输出 | 工程边界不改变位移和乘加关系 |
