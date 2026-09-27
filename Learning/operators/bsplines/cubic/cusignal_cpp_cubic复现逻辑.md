# cusignal_cpp_cubic 复现逻辑

## 版本索引

- 当前实现：[V1：正式五类型 CPU/GPU 实现（fdd55ac）](#v1正式五类型-cpugpu-实现fdd55ac)
- 当前只有 V1；后续优化必须追加 V2、V3，不覆盖本节。

## V1：正式五类型 CPU/GPU 实现（fdd55ac）

### 1. 版本身份与工作区状态

| 项目 | 值 |
| --- | --- |
| 完整提交 SHA | <code>fdd55ac8415d70379eb38a2c299047f90bcf0a41</code> |
| 提交时间 | <code>2026-08-15T23:45:58+08:00</code> |
| 分支 | <code>final-prep/benchmark-evidence-v1</code> |
| 相关文件 dirty 状态 | 无；检查时 <code>git status --short</code> 为空 |
| 阅读方式 | 以 <code>git show &lt;SHA&gt;:&lt;path&gt;</code> 交叉核对 |
| 本次运行状态 | 未构建、未运行测试、未连接 ZQ500；测试章节只描述源码覆盖设计 |

### 2. 文件、符号和职责

以下定位全部绑定上述完整 SHA。

| 层次 | 共享根目录相对路径 | 符号 | 提交内行号 | 职责 |
| --- | --- | --- | ---: | --- |
| 公共总头 | <code>ZKX/cusignal_cpp/src/signal_processing.h</code> | include | 18 | 暴露 typed API |
| 类型与接口 | <code>ZKX/cusignal_cpp/src/bsplines/bsplines_typed.h</code> | <code>BsplineTypePolicy</code> | 25-59 | 五类型加载、FP32 计算、同型写回 |
| 数值函数 | 同上 | <code>cubic_value</code> | 70-83 | CPU 分段公式 |
| CPU 声明 | 同上 | <code>cubic_cpu</code> | 123-124 | 正式 CPU reference |
| 兼容别名 | 同上 | <code>cubic</code> | 133-135 | 已弃用 CPU 别名 |
| GPU 声明 | 同上 | <code>cubic_device</code> | 149-150 | 正式 GPU 入口 |
| CPU helper | <code>ZKX/cusignal_cpp/src/bsplines/bsplines_typed.cpp</code> | <code>evaluate_cpu</code> | 9-18 | 遍历 host vector |
| CPU 实现 | 同上 | <code>cubic_cpu</code> | 22-26 | 调用数值函数 |
| GPU wrapper | <code>ZKX/cusignal_cpp/src/bsplines/bsplines_typed.cu</code> | <code>cubic_device</code> | 26-34 | 长度检查和启动 |
| GPU kernel | <code>ZKX/cusignal_cpp/src/bsplines/bsplines_kernels.cuh</code> | <code>cubic_kernel</code> | 10-27 | 每线程一个元素 |
| 启动 helper | <code>ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h</code> | <code>launch_1d_kernel</code> | 44-56 | 默认 stream、block=256 |
| 直接测试 | <code>ZKX/cusignal_cpp/test/signal_processing/e3_bsplines_type_smoke.cu</code> | <code>verify_case</code>、<code>run_type</code> | 69-206 | GPU 对 CPU、五类型和边界 |

### 3. 对外接口与类型契约

~~~cpp
template <typename T>
std::vector<T> cubic_cpu(const std::vector<T>& x);

template <typename T>
[[deprecated("CPU compatibility alias; use cubic_device for GPU execution or cubic_cpu for reference")]]
std::vector<T> cubic(const std::vector<T>& x);

template <typename T>
void cubic_device(const DeviceArray<T>& x, DeviceArray<T>& y);
~~~

证据：<code>bsplines_typed.h:123-150</code>。

- <code>cubic_cpu</code> 接收 host vector 并返回同长度同类型 host vector。
- <code>cubic</code> 只转发到 CPU reference，已 deprecated，不是正式 GPU 路径。
- <code>cubic_device</code> 接受设备驻留输入和预分配输出，不隐式搬运。
- 逻辑 rank 由调用者持有；实现只处理连续展平 buffer。
- 支持 <code>float</code>、<code>__half</code>、<code>int32_t</code>、<code>int16_t</code>、<code>int8_t</code>。
- <code>operator_type_contract.h:48</code> 将其标为 same-as-input、pointwise、FP32 compute。

### 4. 类型策略与共享数值函数

#### 4.1 通用类型策略

<code>bsplines_typed.h:25-43</code>：

~~~cpp
template <typename T>
struct BsplineTypePolicy {
    static constexpr bool supported =
        std::is_same_v<T, float> ||
        std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> ||
        std::is_same_v<T, std::int8_t>;
    using ComputeT = float;

    __host__ __device__ static ComputeT load(T value)
    {
        return static_cast<float>(value);
    }

    __host__ __device__ static T store(ComputeT value)
    {
        return static_cast<T>(value);
    }
};
~~~

输入由 <code>load</code> 转 FP32，输出由 <code>store</code> 转回 <code>T</code>。整数写回截断小数。

#### 4.2 FP16 特化

<code>bsplines_typed.h:45-59</code>：

~~~cpp
template <>
struct BsplineTypePolicy<__half> {
    static constexpr bool supported = true;
    using ComputeT = float;

    __host__ __device__ static ComputeT load(__half value)
    {
        return __half2float(value);
    }

    __host__ __device__ static __half store(ComputeT value)
    {
        return __float2half_rn(value);
    }
};
~~~

FP16 先转 FP32，写回采用 round-to-nearest。

#### 4.3 cubic_value

<code>bsplines_typed.h:70-83</code>：

~~~cpp
template <typename T>
inline T cubic_value(T value)
{
    const float x = BsplineTypePolicy<T>::load(value);
    const float ax = std::fabs(x);
    float result = 0.0F;
    if (ax < 1.0F) {
        result = 2.0F / 3.0F - 0.5F * ax * ax * (2.0F - ax);
    } else if (ax < 2.0F) {
        const float distance = 2.0F - ax;
        result = distance * distance * distance / 6.0F;
    }
    return BsplineTypePolicy<T>::store(result);
}
~~~

初值 0 覆盖 $|x|\ge2$，两分支落实阶段一公式。

### 5. CPU 调用链与实现

~~~text
cubic_cpu(vector<T>)
  → evaluate_cpu
  → 遍历 i
  → cubic_value(x[i])
  → load FP32 → 分段公式 → store T
  → y[i]
~~~

<code>bsplines_typed.cpp:9-18</code>：

~~~cpp
template <typename T, typename Evaluator>
std::vector<T> evaluate_cpu(const std::vector<T>& x, Evaluator evaluator)
{
    static_assert(detail::is_bspline_input_v<T>, "unsupported B-spline business input dtype");
    std::vector<T> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = evaluator(x[i]);
    }
    return y;
}
~~~

<code>bsplines_typed.cpp:22-32</code>：

~~~cpp
template <typename T>
std::vector<T> cubic_cpu(const std::vector<T>& x)
{
    return evaluate_cpu(x, [](T value) { return detail::cubic_value(value); });
}

template <typename T>
std::vector<T> cubic(const std::vector<T>& x)
{
    return cubic_cpu(x);
}
~~~

CPU 只分配 host 输出，不调用 CUDA、FFT 或 workspace；空输入返回空 vector。

### 6. GPU wrapper、启动配置与 kernel

#### 6.1 buffer 校验与 wrapper

<code>bsplines_typed.cu:13-34</code>：

~~~cpp
template <typename T>
void require_compatible_buffers(const DeviceArray<T>& x, const DeviceArray<T>& y, const char* name)
{
    static_assert(detail::is_bspline_input_v<T>, "unsupported B-spline business input dtype");
    if (x.size() != y.size()) {
        throw_error(ErrorCode::INVALID_SHAPE,
            std::string(name) + "输入输出shape不一致：input_size=" +
                std::to_string(x.size()) + "，output_size=" + std::to_string(y.size()));
    }
}

template <typename T>
void cubic_device(const DeviceArray<T>& x, DeviceArray<T>& y)
{
    require_compatible_buffers(x, y, "cubic_device");
    if (!x.empty()) {
        cuda_utils::launch_1d_kernel(
            bsplines_detail::cubic_kernel<T>, x.size(), x.data(), y.data(), x.size());
    }
}
~~~

长度不等抛 INVALID_SHAPE；空数组不启动。wrapper 不同步。

#### 6.2 grid 与 block

<code>kernel_launch.h:21-28,44-56</code>：

~~~cpp
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
inline void launch_1d_kernel(
    Kernel kernel,
    std::size_t elements,
    Args&&... args)
{
    launch_1d_kernel_with_config(
        kernel, elements, nullptr, 256, std::forward<Args>(args)...);
}
~~~

block=256，grid=$\lceil N/256\rceil$。底层在默认 stream 启动并检查 launch 错误，不同步。

#### 6.3 device kernel

<code>bsplines_kernels.cuh:10-27</code>：

~~~cpp
template <typename T>
__global__ void cubic_kernel(const T* x, T* y, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float value = detail::BsplineTypePolicy<T>::load(x[index]);
    const float absolute_value = fabsf(value);
    float result = 0.0F;
    if (absolute_value < 1.0F) {
        result = 2.0F / 3.0F -
            0.5F * absolute_value * absolute_value * (2.0F - absolute_value);
    } else if (absolute_value < 2.0F) {
        const float distance = 2.0F - absolute_value;
        result = distance * distance * distance / 6.0F;
    }
    y[index] = detail::BsplineTypePolicy<T>::store(result);
}
~~~

每线程一个元素；多余线程退出。连续访问，没有共享内存、原子、线程通信或临时 buffer。

### 7. 显式实例化

CPU 的 <code>bsplines_typed.cpp:62-76</code> 与 GPU 的 <code>bsplines_typed.cu:65-76</code> 对五种类型显式实例化：<code>float</code>、<code>__half</code>、<code>int32_t</code>、<code>int16_t</code>、<code>int8_t</code>。其他类型不在正式链接集合中，且 static_assert 会拒绝不支持类型。

### 8. Python、CPU、GPU 三方映射

| 原理 | Python | C++ CPU | C++ GPU | 差异 |
| --- | --- | --- | --- | --- |
| $a=|x|$ | 基准 <code>bsplines.py:54</code> | header 74 | kernel 17 | 数学相同 |
| $a&lt;1$ | Python 56-57 | header 76-77 | kernel 19-21 | 代数相同 |
| $1\le a&lt;2$ | Python 58-59 | header 78-80 | kernel 22-24 | C++ 使用 distance |
| $a\ge2$ | 显式 else | result 初值 | result 初值 | 等价实现 |
| dtype | 同型 T | FP32 后写回 T | FP32 后写回 T | C++ 限定五类型 |
| 调度 | CuPy kernel | host for | custom kernel | 原理不变 |

### 9. 精度、同步与平台限制

- 五类型统一 FP32 计算；FP16 只在边界转换。
- 整数分数写回时截断，属于同型量化扩展。
- $|x|=1$ 得 $1/6$；$|x|=2$ 得 0；NaN 比较均为假，也保留 0。
- 输入输出必须为等长连续 DeviceArray；逻辑 rank 不参与计算。
- 默认 stream、block 256；无 stream 参数、无 workspace。
- API 只检查 launch 错误，调用者读取输出前负责同步。
- 本次未在 ZQ500 运行，不能声称硬件实测通过。

### 10. 测试覆盖与风险

<code>e3_bsplines_type_smoke.cu</code> 中：

- <code>verify_case:72-92</code> 用 GPU 对 CPU reference；
- <code>run_type:94-206</code> 复用到五类型；
- <code>:97</code> 的典型输入覆盖中心、外侧、支撑外和正负；
- <code>:132-151</code> 覆盖最小、alternate、boundary、展平 rank-3；
- <code>:153-160</code> 覆盖输出长度错误；
- <code>:42-52</code> 定义 FP32、FP16、整数容差；
- <code>main:210-226</code> 要求 15 项证据全部通过。

未覆盖风险：

1. CPU/GPU 使用同一公式，可能共同复制错误，需要独立 Python/SciPy oracle。
2. 没有明确覆盖精确结点 $\pm1$。
3. NaN、无穷、整数截断边界没有专项测试。
4. 没有长数组性能和异步错误传播专项测试。
5. 本次静态阅读没有执行测试。

### 11. 阅读顺序

按 <code>bsplines_typed.h → bsplines_typed.cpp → bsplines_typed.cu → kernel_launch.h → bsplines_kernels.cuh → e3_bsplines_type_smoke.cu</code> 阅读，分别观察类型、CPU、wrapper、启动、线程计算和验证。

### 12. 自检问题与参考答案

1. **V1 绑定哪个版本，是否 dirty？**  
   答：完整 SHA 为 <code>fdd55ac8415d70379eb38a2c299047f90bcf0a41</code>，相关状态为空。

2. **三个入口分别是什么？**  
   答：<code>cubic_cpu</code> 是 CPU reference；<code>cubic</code> 是 deprecated CPU 别名；<code>cubic_device</code> 是正式 GPU 入口。

3. **五类型如何统一计算？**  
   答：TypePolicy 先 load 到 FP32，计算后 store 回 T；FP16 专门转换，整数截断。

4. **线程和 grid 如何分工？**  
   答：每线程处理索引 <code>blockIdx.x*blockDim.x+threadIdx.x</code>；block=256，grid=$\lceil N/256\rceil$。

5. **CPU/GPU 数学是否相同？**  
   答：相同，都按绝对值的三个区间求同一闭式公式。

6. **wrapper 是否搬运或同步？**  
   答：否，只检查长度并启动；显式 DeviceArray 边界负责搬运和同步。

7. **整数结果为何通常为零？**  
   答：实值范围在 $[0,2/3]$，写回整数时分数向零截断。

8. **测试怎样验证？**  
   答：同输入分别走 GPU 和 CPU reference，D2H 后按类型容差比较。

9. **最大共因风险是什么？**  
   答：CPU/GPU 复制同一公式，可能一起错，需要独立 oracle。

10. **能否声称 ZQ500 已通过？**  
    答：不能，本次未执行构建或硬件测试。

## 版本差异与原理不变量

当前只有 V1。数学不变量是中心化三次 B-spline 的三段公式、偶对称与紧支撑；后续类型、并行和内存实现可以优化。若数学算法改变，必须重新映射到《数学物理原理》的原理编号。
