# cusignal_cpp_taylor 复现逻辑

## 版本索引

| 版本 | Git 快照 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前学习版本](#v1当前学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU FP64 reference；GPU FP32 系数与补偿累加；device 端拓宽为 FP64 存储 |

## V1：当前学习版本（fdd55ac）

### 1. 版本身份与取证边界

| 项目 | 值 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15T23:45:58+08:00` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| 当前算子相关文件 dirty 状态 | 全部 clean；`git status --short -- <相关文件>` 无输出 |
| 取证方式 | `git -C ZKX show fdd55ac8415d70379eb38a2c299047f90bcf0a41:<路径>` |
| 本阶段运行状态 | 只读学习；未构建、未运行测试、未连接 ZQ500 |

相关文件只限以下范围：

| 职责 | 共享根目录 `ZKX_dev/` 下的路径 | 当前符号与该提交行号 |
| --- | --- | --- |
| 公开 typed API | `ZKX/cusignal_cpp/src/windows/windows_typed.h` | `taylor_typed_cpu` 211–227；`taylor_device` 229–247 |
| CPU reference 与公开 GPU 包装 | `ZKX/cusignal_cpp/src/windows/windows_typed.cpp` | `host_load` 19–23；`host_taylor_value` 107–138；`taylor_typed_cpu` 491–515；`taylor_device` 517–534；实例化 566–600 |
| GPU Host wrapper | `ZKX/cusignal_cpp/src/windows/windows_typed.cu` | `taylor_fp32_compute_device` 79–113；实例化 116–127 |
| Taylor 系数与 device kernel | `ZKX/cusignal_cpp/src/windows/windows_kernels.cuh` | `window_store` 21–29；`build_taylor_coefficients` 392–456；`taylor_kernel` 458–488 |
| 输入类型策略 | `ZKX/cusignal_cpp/src/cuda_utils/simple_signal_typed.h` | `SimpleSignalTypePolicy` 17–47 |
| 补偿累加 | `ZKX/cusignal_cpp/src/cuda_utils/operator_compute_policy.h` | `CompensatedFloatAccumulator` 90–116 |
| 一维 kernel 启动 | `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h` | `div_up`、`make_1d_launch_config` 10–28；`launch_1d_kernel` 31–56 |
| FP64 存储收尾 | `ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h` | `finalize_fp64_device_storage` 40–47 |
| device 拓宽实现 | `ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu` | `widened_storage_bits` 11–43；`widen_real_storage_kernel` 45–50；`widen_fp32_storage_device` 63–73 |
| 直接测试 | `ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu` | Taylor 测试 673–733；证据记录 758–776；五类型入口 788–798 |

后续每个代码定位都绑定同一完整 SHA；行号指该提交中的行号，不依赖当前工作区变化。

另对该 SHA 下的 `cusignal_cpp/src/windows/windows.h`、`windows.cpp`、`windows_cuda.h`、`windows_cuda.cu` 做了精确 `taylor` 文本搜索，均无命中。因此 V1 不存在另一套 legacy Taylor 核心；正式 CPU/GPU 学习对象就是上述 typed 路径，不能把其他窗算子的 legacy 包装误认为 Taylor 实现。

### 2. 总体职责与调用链

```text
taylor_typed_cpu<T>
  → host_load(sidelobe)
  → FP64 计算 A、s²
  → 对每个输出 index 调 host_taylor_value<double>
      → 对每个 order 重新计算 F_order
      → 累加有限余弦级数
  → 可选除以连续中心 peak
  → std::vector<double>

taylor_device<T>
  → 校验输出 shape / 处理 M=0、1
  → 分配 DeviceArray<float> computed
  → taylor_fp32_compute_device<T>
      → SimpleSignalTypePolicy<T>::load(sll) 转 FP32
      → build_taylor_coefficients<float> 在 Host 生成 F_m 与 scale
      → DeviceArray<float>::from_host：系数 H2D
      → launch_1d_kernel(taylor_kernel<float,float>)
          → 每个线程负责一个 w[index]
          → CompensatedFloatAccumulator 累加
  → finalize_fp64_device_storage
      → device widening kernel 把每个 FP32 数值精确拓宽为 FP64 存储
  → DeviceArray<double>
```

这条链没有 FFT、卷积、插值或外部科学计算库。核心是闭式 Taylor 系数、有限余弦和、自定义 CUDA kernel，以及 dtype/storage 收尾。

### 3. 对外接口

#### 3.1 CPU reference 声明

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.h`；符号 `taylor_typed_cpu`；第 211–227 行：

```cpp
/**
 * @brief 在 CPU 上生成 Taylor 窗 reference。
 * @tparam T 五种sll输入类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param nbar 近似等幅旁瓣数；M大于1时必须至少为1。
 * @param sidelobe_level 旁瓣抑制度；0合法，负值按公式传播NaN。
 * @param normalize true 时使连续中心峰值为 1，默认 true。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`，或M大于1且`nbar<1`。
 * @details CPU reference将五类型（含FP32）参数转FP64系数并累加；M为0或1时其他参数不
 * 参与计算；不创建GPU stream或device workspace，保持cuSignal语义。
 */
template <class T>
std::vector<double> taylor_typed_cpu(
    int n, int nbar, T sidelobe_level, bool normalize = true,
    bool sym = true);
```

逐行解释这个完整声明块：

- `/**` 到 `*/` 是 Doxygen 文档；每个 `@param` 对应后面签名中的同名参数。
- `@tparam T` 明确只有 `sidelobe_level` 的业务输入类型可变，返回值不随 `T` 改变。
- `n=0/1` 的短路径位于 `nbar` 校验之前，所以这些长度不要求 `nbar>=1`，对齐 Python `_len_guards` 的边界顺序。
- `normalize` 和 `sym` 的 `= true` 是默认实参，只出现在声明处。
- `template <class T>` 声明函数模板；`std::vector<double>` 固定 Host FP64 输出；两行参数列表是同一个声明，末尾分号表示这里只有接口、没有函数体。

#### 3.2 GPU 声明

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.h`；符号 `taylor_device`；第 229–247 行：

```cpp
/**
 * @brief 在 GPU 上生成 Taylor 窗。
 * @tparam T 五种sll输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param nbar 近似等幅旁瓣数；M大于1时必须至少为1。
 * @param sidelobe_level 正旁瓣抑制度（dB）。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param normalize 是否归一化，默认 true。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument 长度/nbar非法或输出size不匹配；CUDA失败走项目异常机制。
 * @details 默认stream上M大于1时在Host生成FP32系数并H2D，再由kernel执行FP32累加；
 * 随后Host拓宽。
 * @note 对齐固定版cuSignal的FP64输出、nbar/sll/norm/sym与边界顺序。
 * @note host 系数与临时 device 系数共同构成函数内部 workspace，返回前释放。
 */
template <class T>
void taylor_device(
    int n, int nbar, T sidelobe_level, DeviceArray<double>& out,
    bool normalize = true, bool sym = true);
```

- 文档把业务输入 `T` 与正式 `DeviceArray<double>` 输出分开，说明“输入类型”不等于“计算/输出类型”。
- `out` 是调用者传入的 device 输出引用，函数返回类型为 `void`。
- “Host 生成 FP32 系数并 H2D”与实现一致；但“由 Host 拓宽”“随后 Host 拓宽”与本 SHA 的实际实现不一致。真正收尾走 `widen_real_storage_kernel`，发生在 device 上。本文以后以源码事实为准，并保留这处不一致。
- 最后两行把两个默认布尔参数与 CPU 声明保持一致。

### 4. CPU reference

#### 4.1 五类型输入加载为连续计算值

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`；符号 `host_load`；第 19–23 行：

```cpp
template <class T>
float host_load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}
```

- 模板接受 `float`、`__half`、`int32_t`、`int16_t` 或 `int8_t`。
- `SimpleSignalTypePolicy<T>::load` 把业务输入转成连续计算用 FP32；CPU 主函数随后再 `static_cast<double>`，所以整数与 FP16 先按策略转 FP32，再进入 FP64 数学。
- 花括号限定函数体，`return` 把策略加载结果交给调用者。

#### 4.2 单个 CPU 样本的完整 Taylor 求值

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`；符号 `host_taylor_value`；第 107–138 行。

签名和初值：

```cpp
template <typename Scalar>
Scalar host_taylor_value(
    Scalar position, int effective, int nbar, Scalar a, Scalar s2)
{
    const Scalar pi = static_cast<Scalar>(3.141592653589793238462643383279502884);
    Scalar value = static_cast<Scalar>(1);
```

- `Scalar` 统一位置、$A$、$\sigma^2$、系数和输出的计算类型；当前 CPU 调用实例为 `double`。
- `position` 是当前离散索引或连续中心位置，`effective` 是对称长度 $M$ 或周期内部长度 $M+1$。
- 高精度字面量经 `static_cast<Scalar>` 转成模板标量；`value=1` 是有限余弦级数的 DC 项。

逐阶构造系数的初始量：

```cpp
    for (int order = 1; order < nbar; ++order) {
        const Scalar order_squared = static_cast<Scalar>(order * order);
        Scalar numerator = static_cast<Scalar>(
            ((order - 1) & 1) ? -1 : 1);
        Scalar denominator = static_cast<Scalar>(2);
```

- `for` 覆盖 $m=1,\ldots,nbar-1$；`++order` 每轮递增。
- `order_squared` 对应 $m^2$。
- `((order-1)&1)` 检查数组式下标的奇偶：$m=1$ 得 `+1`，$m=2$ 得 `-1`，实现 $(-1)^{m+1}$。
- 分母初值 `2` 对应 $F_m$ 公式最前面的 $1/2$。

分子乘积：

```cpp
        for (int index = 1; index < nbar; ++index) {
            const Scalar offset = static_cast<Scalar>(index) -
                static_cast<Scalar>(0.5);
            numerator *= static_cast<Scalar>(1) - order_squared
                / s2 / (a * a + offset * offset);
        }
```

- 内层 `index` 对应公式中的 $j=1,\ldots,nbar-1$。
- 两行 `offset` 合成 $j-1/2$。
- 两行 `numerator *=` 组成一个完整乘法赋值，实现

  $$
  \prod_j\left[1-\frac{m^2}{\sigma^2(A^2+(j-1/2)^2)}\right].
  $$

分母排除 $j=m$：

```cpp
        for (int index = 1; index < nbar; ++index) {
            if (index != order) {
                denominator *= static_cast<Scalar>(1)
                    - order_squared / static_cast<Scalar>(index * index);
            }
        }
```

- 第二个内层循环再次覆盖所有 $j$。
- `if (index != order)` 跳过会产生零因子的 $j=m$。
- 两行乘法实现 $2\prod_{j\ne m}(1-m^2/j^2)$。

把当前 $F_m$ 乘余弦并累加：

```cpp
        value += static_cast<Scalar>(2) * (numerator / denominator)
            * std::cos(static_cast<Scalar>(2) * pi *
                static_cast<Scalar>(order) *
                (position - static_cast<Scalar>(effective) /
                    static_cast<Scalar>(2) + static_cast<Scalar>(0.5)) /
                static_cast<Scalar>(effective));
    }
    return value;
}
```

- `numerator/denominator` 就是 $F_m$；前面的 `2` 合并正负频率对。
- `std::cos` 的多行实参对应相位 $2\pi m(position-effective/2+1/2)/effective$。
- `value +=` 把该阶加入当前样本；结束全部阶次后 `return value`。
- 关键性能事实：这个 helper 每求一个位置都会重新计算全部 $F_m$，因此 CPU reference 没有复用跨样本相同的系数。

#### 4.3 CPU 公开函数：边界、参数和输出

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`；符号 `taylor_typed_cpu`；第 491–515 行。

签名与边界顺序：

```cpp
template <class T>
std::vector<double> taylor_typed_cpu(
    int length, int nbar, T sidelobe, bool normalize, bool symmetric)
{
    if (length < 0) throw std::invalid_argument("taylor: length must be nonnegative");
    if (length == 0) return {};
    if (length == 1) return {1.0};
    if (nbar < 1) throw std::invalid_argument("taylor: nbar must be at least one");
```

- 模板和返回类型与头文件声明对应；定义处不重复默认值。
- 负长度先抛 `invalid_argument`；`return {}` 是空 `vector<double>`；`return {1.0}` 是单元素 FP64。
- `nbar` 检查在两个长度短路径之后，所以 $M=0,1$ 时 `nbar=0` 合法且不参与计算。

内部长度、FP64 参数与归一化中心：

```cpp
    const int effective = symmetric ? length : length + 1;
    std::vector<double> out(length);
    constexpr double pi64 = 3.141592653589793238462643383279502884;
    const double sidelobe_value = static_cast<double>(host_load(sidelobe));
    const double a = std::acosh(std::pow(10.0, sidelobe_value / 20.0)) / pi64;
    const double nbar_offset = nbar - 0.5;
    const double s2 = static_cast<double>(nbar * nbar)
        / (a * a + nbar_offset * nbar_offset);
    const double peak = host_taylor_value(
        0.5 * (effective - 1), effective, nbar, a, s2);
```

- 三元运算符在对称模式取 $M$，周期模式取 $M+1$；输出仍只分配原 `length` 个元素。
- `host_load` 先得到 FP32 连续值，再拓为 `double`；之后 $B,A,\sigma^2$ 全为 FP64。
- `std::pow(10,sll/20)`、`std::acosh(...)/pi`、`s2` 与 Python 公式逐项一致。
- `peak` 用连续中心 `0.5*(effective-1)` 调同一求值 helper；即使 `normalize=false` 也会计算，属于 CPU reference 的可避免工作。

逐点生成并返回：

```cpp
    for (int index = 0; index < length; ++index) {
        const double value = host_taylor_value(
            static_cast<double>(index), effective, nbar, a, s2);
        out[index] = normalize ? value / peak : value;
    }
    return out;
}
```

- 循环只写原始 $M$ 个位置；周期模式自然相当于计算 $M+1$ 点对称窗的前 $M$ 点，不需要再切片。
- 每个 `value` 都重新执行完整 Taylor 系数与余弦计算。
- 三元表达式在 `normalize=true` 时除以连续中心 `peak`，否则保留 DC 归一化值。
- 最后一行返回 Host `vector<double>`。

### 5. GPU 路径

#### 5.1 公开 GPU wrapper

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`；符号 `taylor_device`；第 517–534 行。

接口定义与 shape 校验：

```cpp
template <class T>
void taylor_device(
    int length, int nbar, T sidelobe, DeviceArray<double>& out,
    bool normalize, bool symmetric)
{
    if (length < 0 || out.size() != static_cast<std::size_t>(length)) {
        throw std::invalid_argument("taylor_device: output size must equal requested length");
    }
```

- `out` 必须已具有 `length` 个 `double` 存储位置；函数定义处不重复默认值。
- `||` 短路：负长度为真后，不需要依赖把负数转换为巨大 `size_t` 的比较结果。
- 负长度或输出 size 不一致共用一个异常消息。

临时 FP32 输出与边界分支：

```cpp
    DeviceArray<float> computed(out.size());
    if (length == 1) {
        computed = DeviceArray<float>::from_host({1.0F});
    } else if (length > 1) {
    if (nbar < 1) throw std::invalid_argument("taylor_device: nbar must be at least one");
        taylor_fp32_compute_device(
            length, nbar, sidelobe, computed, normalize, symmetric);
    }
```

- 第一行在 device 分配与正式输出同长度的 FP32 临时数组；`length=0` 得到空数组。
- `length=1` 从 Host 单元素 `{1.0F}` 构造 device 数组，故 `nbar/sll/norm/sym` 都不参与。
- `length>1` 才检查 `nbar>=1` 并进入正式计算。
- 源码第 529 行的 `if (nbar < 1)` 缩进比周围少一级，这是原提交中的排版事实；语法上它仍位于 `else if` 花括号内，控制流不受影响。
- 两行调用把全部参数传给 FP32 compute wrapper。

正式 FP64 storage 收尾：

```cpp
    out = cuda_utils::finalize_fp64_device_storage(computed);
}
```

- 无论长度为 0、1 还是更大，都会经过统一收尾。
- 右侧先创建新的 `DeviceArray<double>`，再赋给引用 `out`；实际数值仍只有 FP32 精度。

#### 5.2 FP32 Host wrapper：一次生成系数并启动 kernel

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cu`；符号 `taylor_fp32_compute_device`；第 79–113 行。

签名和前置条件：

```cpp
template <class T>
void taylor_fp32_compute_device(
    int n,
    int sidelobe_count,
    T sidelobe_level,
    DeviceArray<float>& output,
    bool normalize,
    bool symmetric)
{
    if (n <= 1 || sidelobe_count < 1 ||
        output.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "taylor_device: n, nbar and output size are inconsistent");
    }
```

- 这是由 `.cpp` wrapper 调用的内部模板；它只接受 $n>1$。
- 跨行 `if` 同时校验长度、`nbar` 与输出大小；任一失败即抛异常。

内部长度与 Host FP32 系数：

```cpp
    const int effective = symmetric ? n : n + 1;
    float scale = 1.0F;
    const std::vector<float> coefficients =
        windows_detail::build_taylor_coefficients(
            effective,
            sidelobe_count,
            detail::SimpleSignalTypePolicy<T>::load(sidelobe_level),
            normalize,
            scale);
```

- `effective` 实现对称/周期规则。
- `scale` 是输出引用参数的接收变量，初值 1。
- `build_taylor_coefficients` 在 Host 执行；模板 `Scalar` 从返回类型和实参推导为 `float`。
- `SimpleSignalTypePolicy<T>::load` 把五种 `sll` 输入统一为 FP32。
- builder 同时返回 `vector<float> coefficients` 并通过 `scale` 引用返回归一化比例。

系数 H2D 与 kernel 启动：

```cpp
    DeviceArray<float> device_coefficients =
        DeviceArray<float>::from_host(coefficients);
    cuda_utils::launch_1d_kernel(
        windows_detail::taylor_kernel<float, float>,
        output.size(),
        n,
        device_coefficients.data(),
        static_cast<int>(coefficients.size()),
        scale,
        output.data(),
        effective);
}
```

- `from_host` 把 $K=nbar-1$ 个 FP32 系数复制到 device workspace。
- `launch_1d_kernel` 的第二个实参 `output.size()` 是线程覆盖元素数；后续实参按顺序传给 kernel。
- `taylor_kernel<float,float>` 表明输出元素 `T=float`、计算标量 `Scalar=float`。
- 指针 `.data()` 暴露 device 地址；系数个数转成 `int`；`scale` 按值传入。
- 闭合花括号结束 wrapper；局部 Host/device 系数 workspace 随函数退出释放。

#### 5.3 Host 端 Taylor 系数 builder

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_kernels.cuh`；符号 `build_taylor_coefficients`；第 392–456 行。

签名与基础参数：

```cpp
template <typename Scalar>
inline std::vector<Scalar> build_taylor_coefficients(
    int effective_count,
    int sidelobe_count,
    Scalar sidelobe_level,
    bool normalize,
    Scalar& scale)
{
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar amplitude = pow(
        static_cast<Scalar>(10),
        sidelobe_level / static_cast<Scalar>(20));
    const Scalar shape = acosh(amplitude) / pi;
```

- `inline` 允许头文件模板在多个翻译单元中定义。
- 返回值是 Host `vector<Scalar>`；`scale` 用非常量引用输出第二个结果。
- 当前实例 `Scalar=float`，而圆周率字面量本身也带 `F`，所以全部是 FP32。
- 三行 `amplitude` 实现 $B=10^{sll/20}$，`shape` 实现 $A=acosh(B)/\pi$。

$\sigma^2$ 与系数数量：

```cpp
    const Scalar offset = static_cast<Scalar>(sidelobe_count) -
        static_cast<Scalar>(0.5F);
    const Scalar denominator_scale =
        static_cast<Scalar>(sidelobe_count * sidelobe_count) /
        (shape * shape + offset * offset);
    const int coefficient_count = sidelobe_count > 1
        ? sidelobe_count - 1
        : 0;
    std::vector<Scalar> coefficients(coefficient_count);
```

- `offset=nbar-1/2`；`denominator_scale` 对应 Python `s2=σ²`，名称强调它位于分母中。
- 三元运算保证 `nbar=1` 时系数数为 0，不产生负长度。
- `vector` 在 Host 分配恰好 $K$ 个标量。

外层逐系数循环与符号：

```cpp
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
        const int order = coefficient_index + 1;
        const Scalar order_squared = static_cast<Scalar>(order * order);
        Scalar numerator = (coefficient_index & 1)
            ? static_cast<Scalar>(-1)
            : static_cast<Scalar>(1);
```

- 三行 `for` 是同一个循环头，覆盖数组下标 $0,\ldots,K-1$。
- `order=index+1` 映射到数学阶次 $m=1,\ldots,K$。
- 下标奇偶生成 $+,-,+,-$ 的 $(-1)^{m+1}$ 符号。

分子乘积：

```cpp
        for (int index = 1; index < sidelobe_count; ++index) {
            const Scalar index_offset =
                static_cast<Scalar>(index) - static_cast<Scalar>(0.5F);
            numerator *= static_cast<Scalar>(1) - order_squared /
                denominator_scale /
                (shape * shape + index_offset * index_offset);
        }
```

- 内层覆盖全部 $j$；`index_offset=j-1/2`。
- 三行乘法赋值逐项实现 $F_m$ 分子的零点乘积。

分母乘积与系数写入：

```cpp
        Scalar denominator = static_cast<Scalar>(2);
        for (int index = 1; index < sidelobe_count; ++index) {
            if (index != order) {
                denominator *= static_cast<Scalar>(1) - order_squared /
                    static_cast<Scalar>(index * index);
            }
        }
        coefficients[coefficient_index] = numerator / denominator;
    }
```

- `denominator=2` 与排除 $j=m$ 的循环共同形成完整分母。
- `coefficients[index]` 保存 $F_m$；外层循环闭合后所有系数仅计算一次，这比 Python kernel 内重复中心和、也比 C++ CPU 每样本重算更高效。

归一化 scale：

```cpp
    scale = static_cast<Scalar>(1);
    if (normalize) {
        const Scalar phase_scale = static_cast<Scalar>(2) * pi /
            static_cast<Scalar>(effective_count);
        const Scalar center_phase = phase_scale *
            ((static_cast<Scalar>(effective_count) - static_cast<Scalar>(1)) /
                static_cast<Scalar>(2) -
             static_cast<Scalar>(effective_count) / static_cast<Scalar>(2) +
             static_cast<Scalar>(0.5F));
        Scalar center = Scalar{0};
        for (int order = 1; order < sidelobe_count; ++order) {
            center += coefficients[order - 1] *
                cos(center_phase * static_cast<Scalar>(order));
        }
        scale = static_cast<Scalar>(1) /
            (static_cast<Scalar>(1) + static_cast<Scalar>(2) * center);
    }
    return coefficients;
}
```

- 默认 `scale=1`；只有 `normalize=true` 才进入中心求值。
- `phase_scale=2π/M_eff`；多行 `center_phase` 把连续中心 $(M_eff-1)/2$ 代入统一坐标，代数值为 0。
- `center` 累加 $\sum F_m\cos(m\phi_c)$；最终 `scale=1/(1+2center)`。
- `return coefficients` 返回系数，同时 `scale` 已经通过引用写回。

#### 5.4 device kernel：一个线程一个窗样本

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_kernels.cuh`；符号 `taylor_kernel`；第 458–488 行。

签名与线程索引：

```cpp
template <typename T, typename Scalar>
__global__ void taylor_kernel(
    int output_count,
    const Scalar* coefficients,
    int coefficient_count,
    Scalar scale,
    T* output,
    int effective_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= output_count) return;
```

- `__global__` 表示从 Host 启动、在 device 执行。
- `coefficients` 是只读 device 指针，`output` 是 device 写指针。
- 线性索引公式把 block 与 thread 坐标展开；越界线程立即返回。

单点短路径与相位：

```cpp
    if (effective_count == 1) {
        output[index] = window_store<T>(1.0F);
        return;
    }
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846F);
    const Scalar phase = static_cast<Scalar>(2) * pi /
        static_cast<Scalar>(effective_count) *
        (static_cast<Scalar>(index) -
         static_cast<Scalar>(effective_count) / static_cast<Scalar>(2) +
         static_cast<Scalar>(0.5F));
```

- 当前公开 wrapper 不会为 $M=1$ 启动此 kernel，因此短路径是防御性/可复用逻辑。
- 多行 `phase` 完整实现中心对齐坐标；周期模式只改变 `effective_count`，线程仍只覆盖原 $M$ 点。

补偿 FP32 余弦累加与写回：

```cpp
    static_assert(std::is_same_v<Scalar, float>);
    compute_policy::CompensatedFloatAccumulator sum;
    for (int order = 1; order <= coefficient_count; ++order) {
        sum.add(fmaf(
            coefficients[order - 1],
            cos(phase * static_cast<Scalar>(order)),
            0.0F));
    }
    output[index] = window_store<T>(
        (static_cast<Scalar>(1) + static_cast<Scalar>(2) * sum.value()) * scale);
}
```

- `static_assert` 在编译期禁止任何非 FP32 的 `Scalar` 实例，明确 ZQ500 device 计算契约。
- `CompensatedFloatAccumulator` 使用 Neumaier 补偿降低长 FP32 求和误差。
- `fmaf(a,b,0)` 用一次融合乘加完成 $F_m\cos(m\phi)$，再交给 `sum.add`。
- 循环条件 `<= coefficient_count` 对应阶次 $1,\ldots,K$。
- 最后恢复 $1+2\sum$、乘 `scale`，经 `window_store<T>` 写当前位置；本实例 `T=float`。

#### 5.5 `window_store` 的类型落地

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_kernels.cuh`；符号 `window_store`；第 21–29 行：

```cpp
template <typename T, typename Scalar>
__host__ __device__ inline T window_store(Scalar value)
{
    if constexpr (detail::is_simple_signal_input_v<T>) {
        return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
    } else {
        return static_cast<T>(value);
    }
}
```

- `__host__ __device__` 允许同一模板在两端编译；`if constexpr` 在编译期选择分支。
- Taylor GPU 实例的 `T=float` 属于 simple signal 类型，因而先转 FP32，再由 policy `store` 返回 float。
- `else` 是其他输出类型的通用路径，本 Taylor 实例不会执行。

#### 5.6 补偿累加器

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/cuda_utils/operator_compute_policy.h`；符号 `CompensatedFloatAccumulator`；第 90–116 行：

```cpp
// Neumaier compensation keeps long FP32 reductions stable without requiring
// unsupported device FP64.  Call value() only after all terms have been added.
struct CompensatedFloatAccumulator {
    float sum{0.0F};
    float correction{0.0F};

    __host__ __device__ void add(float value)
    {
        const float next = sum + value;
        if (fabsf(sum) >= fabsf(value)) {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }

    __host__ __device__ void add_product(float left, float right)
    {
        add(fmaf(left, right, 0.0F));
    }

    __host__ __device__ float value() const
    {
        return sum + correction;
    }
};
```

- 开头两行注释直接说明目的：不用 device FP64，仍改善长 FP32 reduction。
- `sum` 保存普通和，`correction` 保存被舍入丢失的低位补偿。
- `add` 先求 `next=sum+value`；根据两项绝对值大小选择 Neumaier 误差恢复式，再更新 `sum`。
- `add_product` 是通用的乘积入口；Taylor kernel 当前直接调用 `add(fmaf(...))`，效果是先融合计算乘积再补偿求和。
- `value() const` 不修改对象，返回主和加补偿；结构体末尾分号结束类型定义。

#### 5.7 grid、block 与启动错误检查

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h`；符号 `div_up`、`make_1d_launch_config`、`launch_1d_kernel`；第 10–28、31–56 行。

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
```

- `div_up` 是整数向上取整除法。
- `LaunchConfig1D` 按源码顺序存 `block`、`grid`。
- 默认 `block_size=256`；grid 大小为 $\lceil M/256\rceil$。
- 每个线程负责至多一个 `output[index]`，多出的线程由 kernel 越界判断退出。

```cpp
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

- 参数包 `Args&&...` 与 `std::forward` 保持实参类别并转交 kernel。
- 三尖括号启动使用 `grid, block, shared_memory=0, stream`。
- `CUDA_KERNEL_CHECK()` 检查启动错误；这里没有显式全设备同步。
- 简化入口固定默认 stream（`nullptr`）和 256 threads/block。

#### 5.8 FP32 数值拓宽为 FP64 存储

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h`；符号 `finalize_fp64_device_storage`；第 40–47 行：

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

- 分配同元素数的 `DeviceArray<double>`。
- `widen_fp32_storage_device` 接收 device 输入/输出地址；不是 D2H→Host 转换。
- 返回新的 double storage 数组。

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/cuda_utils/fp64_storage_finalize.cu`；符号 `widened_storage_bits`、`widen_real_storage_kernel`、`widen_fp32_storage_device`；第 11–50、63–73 行。

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
```

- device helper 用 union 读取 FP32 位模式，拆出符号、8 位指数和 23 位尾数。
- 后续不是重新计算 Taylor，只是把同一 FP32 数值编码成 IEEE FP64 位模式。

```cpp
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
```

- 指数为 0 时处理零和 FP32 次正规数；循环归一化尾数，再构造 FP64 指数与尾数。
- 位或 `|` 合并三个字段；零保留符号位。

```cpp
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

- `0xff` 分支保持无穷/NaN 类别并拓宽 payload。
- 普通数把 FP32 exponent bias 127 转成 FP64 bias 1023，差值是 896；尾数左移到 52 位字段。
- 因为每个 FP32 数都能被 FP64 精确表示，这一步不再产生数值舍入，但也不能恢复此前 FP32 计算丢失的精度。

```cpp
__global__ void widen_real_storage_kernel(
    const float* input, std::uint64_t* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = widened_storage_bits(input[index]);
}

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

- widening kernel 同样采用一线程一元素。
- 空数组直接返回；非空时在默认 stream 再启动一个 256-thread block 配置的 kernel。
- `void*` 被转成 `uint64_t*`，直接写 double storage 的位模式。

### 6. template、显式实例化、内存布局与同步

#### 6.1 五类型显式实例化

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cpp`；宏 `INSTANTIATE_WINDOWS_CPU` 中的 Taylor 项及调用；第 588–600 行：

```cpp
    template std::vector<double> taylor_typed_cpu(int, int, T, bool, bool); \
    template void taylor_device( \
        int, int, T, DeviceArray<double>&, bool, bool); \

INSTANTIATE_WINDOWS_CPU(float);
INSTANTIATE_WINDOWS_CPU(__half);
INSTANTIATE_WINDOWS_CPU(std::int32_t);
INSTANTIATE_WINDOWS_CPU(std::int16_t);
INSTANTIATE_WINDOWS_CPU(std::int8_t);

#undef INSTANTIATE_WINDOWS_CPU
```

- 反斜杠把宏体续到下一物理行；两条 `template` 强制生成 CPU 与公开 GPU wrapper 的具体符号。
- 五次宏调用覆盖 FP32、FP16、INT32、INT16、INT8。
- `#undef` 防止宏污染后续源码。

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/src/windows/windows_typed.cu`；宏 `INSTANTIATE` 中的 Taylor 项及调用；第 124–127 行：

```cpp
 template void taylor_fp32_compute_device(int,int,T,DeviceArray<float>&,bool,bool); \
INSTANTIATE(float); INSTANTIATE(__half); INSTANTIATE(std::int32_t); INSTANTIATE(std::int16_t); INSTANTIATE(std::int8_t);
#undef INSTANTIATE
```

- 同一物理行的五次宏调用为内部 FP32 compute wrapper 生成五种输入实例。
- 输出临时数组始终是 `DeviceArray<float>`，不随 `T` 改变。

#### 6.2 主要数据结构与搬运

| 数据 | 所在位置 | dtype | shape | 生命周期/搬运 |
| --- | --- | --- | --- | --- |
| `sidelobe` | Host 参数 | 五类型之一 | 标量 | policy load 到 FP32；CPU 再转 FP64 |
| CPU `out` | Host | `double` | `[M]` | 直接返回 |
| GPU `coefficients` | Host | `float` | `[nbar-1]` | builder 创建，随后 H2D |
| `device_coefficients` | Device | `float` | `[nbar-1]` | kernel 只读 workspace，wrapper 退出时释放 |
| `computed` | Device | `float` | `[M]` | Taylor kernel 写入；随后 device widening 读取 |
| 正式 `out` | Device | `double` storage | `[M]` | widening kernel 写入，返回调用者 |

默认 stream 上先启动 Taylor kernel，再启动 widening kernel；同一 stream 保证顺序。源码没有显式 `cudaDeviceSynchronize`，但每次通过 `CUDA_KERNEL_CHECK()` 检查 kernel 启动错误。本文不推断未展示的 `DeviceArray` 释放是否额外同步。

### 7. 数学原理、Python、CPU 与 GPU 映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $B=10^{sll/20}$ | `windows.py` 基准 1936 | `windows_typed.cpp:503` | `windows_kernels.cuh:401–404` | Python/CPU FP64；GPU builder FP32 |
| $A=acosh(B)/\pi$ | 基准 1937 | `windows_typed.cpp:503` | `windows_kernels.cuh:404` | 数学相同、精度不同 |
| $\sigma^2=nbar^2/(A^2+(nbar-1/2)^2)$ | 基准 1938 | `windows_typed.cpp:504–506` | `windows_kernels.cuh:405–409` | 变量分别为 `s2`、`denominator_scale` |
| $F_m$ 分子/分母乘积 | 基准 1941–1949 | `host_taylor_value:113–129` | `build_taylor_coefficients:414–436` | CPU 每样本重算；GPU Host builder 一次计算 |
| $w[i]=1+2\sum F_m\cos(m\phi_i)$ | kernel 基准 1834–1840 | `host_taylor_value:130–137` | `taylor_kernel:473–488` | GPU 使用 FMA+补偿 FP32 |
| 连续中心归一化 | kernel 基准 1842–1853 | `windows_typed.cpp:507–512` | builder `438–454`，kernel 487–488 | Python 每线程重复；C++ GPU 只算一次 scale |
| 周期窗 $M+1$ 后取前 $M$ | `_extend/_truncate` 基准 27–40 | `effective` 499；循环 509 | wrapper 93；线程数仍为 M | C++ 不分配第 M+1 个输出，直接用有效长度坐标 |
| FP64 输出 | Python kernel 声明 `float64` | 原生 FP64 算术/存储 | FP32 算术后 device 精确拓宽 storage | GPU 输出类型相同但有效精度不同 |

### 8. 与 cuSignal Python 相同、等价和有意不同之处

- **相同：** 参数顺序、$M=0/1$ 短路径、`nbar` 在短路径之后才参与、正 `sll` 公式、`norm`、`sym` 和最终一维 FP64 输出语义。
- **等价替换：** Python 周期窗“算 $M+1$ 再切片”；C++ 用 `effective=M+1` 但只启动 $M$ 个输出线程，得到相同前缀。
- **有意不同：** C++ typed API 显式支持五种 `sll` 输入业务类型；Python API没有这个模板分发。
- **有意不同：** C++ GPU 为 ZQ500 采用 FP32 系数、三角函数和补偿累加，再拓宽存储；Python CuPy kernel 使用 `float64` 系数与输出。
- **实现优化：** C++ GPU 在 Host 只计算一次系数和 `scale`；Python `norm=True` 时每个输出线程重复计算相同中心 scale。
- **reference 代价：** C++ CPU 为清晰 reference 每个样本重算系数，复杂度高于 Python 和 C++ GPU Host builder。
- **边界显式化：** C++ 对 $M>1,nbar<1$ 主动抛异常；Python 没有独立 `nbar` 检查。
- **保留传播：** `sll<0` 导致 $B<1$，`acosh(B)` 产生 NaN；C++ 测试要求 CPU/GPU 都传播 NaN，而不是暗自取绝对值。

### 9. 直接测试逐块解释

本节解释的是已提交测试源码，不声称本次会话执行过它。

#### 9.1 正常参数 CPU/GPU 对照

SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX/cusignal_cpp/test/signal_processing/e3_wave_window_type_smoke.cu`；`run_type<T>` 的 Taylor 测试；第 673–677 行：

```cpp
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::taylor);
    taylor_device(5, 3, convert<T>(30), fixed_fp64_window_out);
    const bool taylor_normal_matches = matches_fp64(
        fixed_fp64_window_out.to_host(),
        taylor_typed_cpu(5, 3, convert<T>(30)));
```

- 第一行切换/记录当前 accuracy 证据算子为 Taylor。
- 第二行以 $M=5,nbar=3,sll=30$ 调 GPU；`convert<T>` 把字面量转换成当前业务类型。
- 三行 `matches_fp64` 把 device 输出复制到 Host，与 CPU reference 比较，结果保存在布尔量。

#### 9.2 `nbar=1` 退化为全 1

同 SHA、同文件、同符号；第 678–682 行：

```cpp
    taylor_device(5, 1, convert<T>(-30), fixed_fp64_window_out);
    const bool taylor_nbar_one_is_one =
        fixed_fp64_window_out.to_host() == std::vector<double>(5, 1.0) &&
        taylor_typed_cpu(5, 1, convert<T>(-30)) ==
            std::vector<double>(5, 1.0);
```

- 即使 `sll=-30` 会让一般 `acosh` 非法，`nbar=1` 没有余弦系数，输出仍应为五个 1。
- `&&` 要求 GPU 和 CPU 两侧都严格等于构造出的全 1 FP64 vector。

#### 9.3 `sll=0` 与负 `sll` 传播

同 SHA、同文件、同符号；第 683–690 行：

```cpp
    taylor_device(5, 3, convert<T>(0), fixed_fp64_window_out);
    const bool taylor_zero_sll_matches = matches_fp64(
        fixed_fp64_window_out.to_host(),
        taylor_typed_cpu(5, 3, convert<T>(0)));
    taylor_device(5, 3, convert<T>(-30), fixed_fp64_window_out);
    const auto taylor_negative_sll = fixed_fp64_window_out.to_host();
    const auto taylor_negative_sll_reference =
        taylor_typed_cpu(5, 3, convert<T>(-30));
```

- 0 dB 被视为合法公式输入，并要求 GPU 与 CPU reference 匹配。
- 负值分别收集 GPU 和 CPU 输出；是否为 NaN 在最终总条件中检查。
- `auto` 让编译器从右侧推导 Host vector 类型。

#### 9.4 $M>1$ 时拒绝 `nbar=0`

同 SHA、同文件、同符号；第 691–700 行：

```cpp
    bool taylor_device_rejects_zero_nbar = false;
    bool taylor_cpu_rejects_zero_nbar = false;
    try { taylor_device(5, 0, convert<T>(30), fixed_fp64_window_out); }
    catch (const std::invalid_argument&) {
        taylor_device_rejects_zero_nbar = true;
    }
    try { (void)taylor_typed_cpu(5, 0, convert<T>(30)); }
    catch (const std::invalid_argument&) {
        taylor_cpu_rejects_zero_nbar = true;
    }
```

- 两个标志先为假；`try` 分别执行 GPU/CPU 非法调用。
- `catch (const std::invalid_argument&)` 只接受目标异常类型，并把相应标志设真。
- `(void)` 显式丢弃 CPU 返回值，避免未使用值告警。

#### 9.5 $M=0/1$ 短路径优先于其他参数

同 SHA、同文件、同符号；第 701–702 行：

```cpp
    taylor_device<T>(1, 0, convert<T>(-30), singleton_fp64_window_out);
    taylor_device<T>(0, 0, convert<T>(-30), empty_fp64_window_out);
```

- 显式模板实参 `<T>` 调用两种短长度。
- 两次都故意给 `nbar=0,sll=-30`，验证这些参数在 $M=0/1$ 时不参与计算。

#### 9.6 负长度异常

同 SHA、同文件、同符号；第 703–714 行：

```cpp
    bool taylor_device_rejects_negative_length = false;
    bool taylor_cpu_rejects_negative_length = false;
    try {
        taylor_device<T>(
            -1, 3, convert<T>(30), empty_fp64_window_out);
    } catch (const std::invalid_argument&) {
        taylor_device_rejects_negative_length = true;
    }
    try { (void)taylor_typed_cpu<T>(-1, 3, convert<T>(30)); }
    catch (const std::invalid_argument&) {
        taylor_cpu_rejects_negative_length = true;
    }
```

- 两个异常标志与前一组相同。
- GPU 调用跨行只是排版，传入长度 −1；CPU 调用同样传 −1。
- 两个 `catch` 必须执行，最终总条件才可能为真。

#### 9.7 周期且不归一化分支

同 SHA、同文件、同符号；第 715–717 行：

```cpp
    DeviceArray<double> taylor_periodic_out(4);
    taylor_device<T>(
        4, 3, convert<T>(30), taylor_periodic_out, false, false);
```

- 第一行分配四点正式 FP64 device 输出。
- 最后两个 `false` 分别关闭中心归一化、请求周期模式，覆盖两个非默认分支。

#### 9.8 汇总到 Taylor 的 `ok[12]`

同 SHA、同文件、同符号；第 718–733 行：

```cpp
    ok[12] = taylor_normal_matches && taylor_nbar_one_is_one &&
        taylor_zero_sll_matches && taylor_negative_sll.size() == 5 &&
        taylor_negative_sll_reference.size() == 5 &&
        std::isnan(taylor_negative_sll[0]) &&
        std::isnan(taylor_negative_sll_reference[0]) &&
        taylor_device_rejects_zero_nbar && taylor_cpu_rejects_zero_nbar &&
        singleton_fp64_window_out.to_host() == std::vector<double>{1.0} &&
        taylor_typed_cpu<T>(1, 0, convert<T>(-30)) ==
            std::vector<double>{1.0} &&
        empty_fp64_window_out.empty() &&
        taylor_typed_cpu<T>(0, 0, convert<T>(-30)).empty() &&
        taylor_device_rejects_negative_length &&
        taylor_cpu_rejects_negative_length &&
        matches_fp64(
            taylor_periodic_out.to_host(),
            taylor_typed_cpu(4, 3, convert<T>(30), false, false));
```

这是一条跨 16 行的完整逻辑与表达式，要求以下条件全部为真：正常匹配、`nbar=1` 全 1、0 dB 匹配、负 `sll` 两侧长度正确且首项 NaN、两侧拒绝非法 `nbar`、单点与空数组短路径、两侧拒绝负长度、周期非归一化 GPU/CPU 匹配。最后的分号结束对 `ok[12]` 的一次赋值。

#### 9.9 证据记录与五类型入口

同 SHA、同文件；符号 `run_type<T>`；第 758–776 行中与 Taylor 直接相关的行：

```cpp
    static const char* labels[14] = {"chirp", "gausspulse", "morlet", "morlet2",
        "ricker", "cwt", "chebwin", "general_cosine", "general_gaussian", "hamming",
        "kaiser", "parzen", "taylor", "triang"};
    ev::record<T,double>(td::OperatorId::taylor,fixed_fp64_window_out.size(),ok[12]),
```

- 标签数组中 `taylor` 与 `ok[12]` 位置一致。
- `record<T,double>` 记录输入业务类型 `T`、固定输出 `double`、输出长度和通过标志。

同 SHA、同文件；符号 `main`；第 788–798 行：

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

- `main` 依次实例化并运行五种业务类型的整组 wave/window 测试，Taylor 每种类型各占一项。
- `passed` 累加通过数，输出 70 项总计；全部通过返回 0，否则返回 1。
- 本次学习没有执行该二进制，因此只能确认“测试覆盖被编码”，不能声称当前环境重新通过。

### 10. 精度、复杂度、性能瓶颈与未覆盖风险

设 $M$ 为输出长度、$K=nbar-1$：

| 路径 | 时间复杂度 | 额外空间 | 主要瓶颈 |
| --- | --- | --- | --- |
| CPU reference | 每个位置重算 $K$ 个系数、每个系数含 $K$ 乘积，约 $O(MK^2)$；另有一次 peak $O(K^2)$ | 输出 $O(M)$ | 系数未跨样本复用；`normalize=false` 仍算 peak |
| GPU Host builder | $O(K^2)$ | Host 系数 $O(K)$ | 直接连乘在大 K/极端 sll 下的 FP32 稳定性 |
| Taylor device kernel | 总工作量 $O(MK)$ | device 系数 $O(K)$、computed $O(M)$ | 每项 `cos`；补偿累计增加少量常数成本 |
| storage widening | $O(M)$ | 正式 FP64 device 输出 $O(M)$ | 第二次 kernel 启动与完整数组写入 |

精度边界：

- CPU reference 的 Taylor 数学是 FP64，但五类型输入先经 policy 转 FP32，随后才转 double。
- GPU 的系数、`pow/acosh/cos/fmaf`、scale 和累加全部 FP32；补偿只改善求和，不提升系数或三角函数精度。
- FP64 storage widening 精确保存已得到的 FP32 数值，不等于 FP64 计算。
- `sll<0` 有意传播 NaN；`sll=0` 只要求 CPU/GPU 一致，测试没有断言具体理论窗形。

当前直接测试仍未覆盖：非常大的 `nbar`、极端正 `sll` 的上/下溢、所有输出元素的 NaN 传播形状、对 SciPy/cuSignal Python 的逐元素外部 oracle、主瓣宽度/旁瓣电平的频域指标、非默认组合的更多长度，以及实际 ZQ500 性能。本阶段没有补写或运行测试。

### 11. 建议阅读顺序

1. `windows_typed.h`：先固定 API、五输入类型与 FP64 输出契约。
2. `taylor_typed_cpu`：理解边界顺序、`effective` 和 FP64 reference。
3. `host_taylor_value`：把 $F_m$ 与余弦和逐项对上数学公式。
4. `taylor_device`：看 FP32 临时输出与短路径。
5. `taylor_fp32_compute_device`：看 Host 系数、H2D 和 kernel 实参。
6. `build_taylor_coefficients`：验证 GPU 系数只算一次、scale 只算一次。
7. `taylor_kernel`：看线程索引、相位、FMA 与补偿累加。
8. `finalize_fp64_device_storage`：确认“FP64 输出”是 storage widening，不是 FP64 device 算术。
9. E3 测试：按正常、退化、非法、周期分支核对契约。

### 12. 自检问题与参考答案

1. **问题：V1 的可追溯身份是什么？**  
   **答案：**完整 SHA 是 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，提交时间 `2026-08-15T23:45:58+08:00`，分支 `final-prep/benchmark-evidence-v1`；全部 Taylor 相关文件 clean。所有行号均通过 `git show` 读取该快照。

2. **问题：CPU 和 GPU 的核心职责如何区分？**  
   **答案：**CPU `taylor_typed_cpu`/`host_taylor_value` 是 Host FP64 reference；GPU 公共 wrapper 处理 device shape 与短路径，`.cu` wrapper 在 Host 生成 FP32 系数并 H2D，`taylor_kernel` 才负责 device 逐点余弦求值（`windows_typed.cpp:491–534`、`windows_typed.cu:79–113`、`windows_kernels.cuh:392–488`）。

3. **问题：CPU/GPU 是否实现同一数学公式？**  
   **答案：**是。两者都计算 $B,A,\sigma^2,F_m$ 和 $1+2\sum F_m\cos(m\phi)$；差异是系数复用、FP64/FP32 精度和 GPU 补偿累加，不是数学原理变化。

4. **问题：五种模板类型控制什么？**  
   **答案：**`T` 控制 `sll` 输入表示（FP32、FP16、INT32、INT16、INT8）；CPU 正式输出固定 `vector<double>`，GPU 正式输出固定 `DeviceArray<double>`，device 核心计算固定 float。显式实例化见 `windows_typed.cpp:588–598` 与 `windows_typed.cu:124–126`。

5. **问题：一个 GPU 线程负责什么？grid/block 如何确定？**  
   **答案：**线程索引 `blockIdx.x*blockDim.x+threadIdx.x` 对应一个 `w[index]`；block 固定 256 线程，grid 为 $\lceil M/256\rceil$，越界线程返回（`kernel_launch.h:10–55`、`windows_kernels.cuh:467–468`）。

6. **问题：有哪些同步与 workspace？**  
   **答案：**Host `vector<float>` 系数、device 系数、device `computed` 是内部 workspace；Taylor kernel 与 widening kernel 都在默认 stream，依赖同 stream 顺序。源码没有显式全设备同步，每次启动后用 `CUDA_KERNEL_CHECK()` 做启动检查。

7. **问题：为什么输出类型是 FP64，但不能说 GPU 做了 FP64 计算？**  
   **答案：**`static_assert(Scalar==float)` 固定核心计算为 FP32；`widened_storage_bits` 只把已得到的 FP32 位模式精确编码为 double 位模式，不能恢复精度（`windows_kernels.cuh:479`、`fp64_storage_finalize.cu:11–49`）。

8. **问题：边界顺序如何对齐 Python？**  
   **答案：**负长度先失败；$M=0$ 返回空，$M=1$ 返回 1；只有 $M>1$ 才检查 `nbar>=1`。因此短长度下 `nbar=0,sll=-30` 仍合法，E3 第 701–702、724–728 行直接验证。

9. **问题：周期窗为何不需要真的分配 $M+1$ 点再截断？**  
   **答案：**C++ 把相位分母设为 `effective=M+1`，但只启动/循环原 $M$ 个索引，数学上正好得到 $M+1$ 点对称窗的前 $M$ 点，与 Python 扩一再截等价。

10. **问题：测试覆盖了什么，不能声称什么？**  
    **答案：**覆盖五类型、正常 CPU/GPU 匹配、`nbar=1`、0/负 `sll`、非法 `nbar`、0/1/负长度以及周期非归一化；本会话没有运行测试，所以不能声称当前 ZQ500 再次 PASS，也不能用这些点推断极端参数和频域指标已覆盖。

## 版本差异与原理不变量

当前只有 V1，没有可比较的 V2/V3。V1 已固定以下不变量：Taylor 零点系数公式、有限余弦和、连续中心归一化、对称/周期坐标和短长度边界。未来若只优化系数复用、kernel 启动、内存搬运或补偿方式，应记录为“原理不变、实现优化”；若更换 Taylor 算法族或改变零点公式，则必须重新映射到 `数学物理原理.md` 的原理编号。
