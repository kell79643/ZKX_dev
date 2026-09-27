# cusignal_cpp_quadratic 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前学习版本](#v1当前学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU reference、deprecated CPU 别名、五类型 ZQ500 GPU 入口和 typed smoke |

当前只有 V1。后续若 `quadratic` 已提交优化，应保留本节和 V1 原文，按 V2、V3 追加。

## V1：当前学习版本（fdd55ac）

### 1. 版本身份与读取方式

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交主题：`test(archive): 增加任务结果范围治理审计`
- 检查时间：2026-08-16（Asia/Shanghai）
- `git status --short`：无输出，整个 `ZKX` 工作区 clean；因此 `quadratic` 相关文件也无未提交修改。
- 正式证据读取方式：`git -C ZKX show fdd55ac8415d70379eb38a2c299047f90bcf0a41:<仓库相对路径>`。

本阶段只读 `ZKX/cusignal_cpp`，没有修改 C++/CUDA 源码，没有构建、运行或连接 ZQ500。文中“测试意图”来自测试源码；“本次已验证”仅指静态源码核对，不声称本会话执行过测试。

### 2. 文件与职责

以下每项代码证据都绑定同一完整 SHA。

| 层次 | 仓库相对路径 | 符号 | 提交内行号 | 职责 |
| --- | --- | --- | ---: | --- |
| 类型与数学 helper | `cusignal_cpp/src/bsplines/bsplines_typed.h` | `BsplineTypePolicy<T>` | 25-59 | 五业务 dtype 的 FP32 load/compute/store 策略 |
| CPU 数学 helper | 同上 | `bspline_abs<T>` | 64-68 | 转 FP32 后求绝对值 |
| CPU 数学核心 | 同上 | `quadratic_value<T>` | 85-97 | 二次 B 样条分段闭式 |
| CPU 公共声明 | 同上 | `quadratic_cpu<T>` | 195-206 | 正式 CPU reference 接口 |
| CPU 兼容声明 | 同上 | `quadratic<T>` | 208-217 | deprecated host 别名 |
| GPU 公共声明 | 同上 | `quadratic_device<T>` | 219-231 | 正式 GPU 入口 |
| 显式实例化声明 | 同上 | `quadratic_cpu`、`quadratic_device` | 245-249、263-267 | 五业务类型的 `extern template` |
| CPU 迭代 helper | `cusignal_cpp/src/bsplines/bsplines_typed.cpp` | `evaluate_cpu<T, Evaluator>` | 9-18 | 分配等长输出并逐元素调用 evaluator |
| CPU 实现 | 同上 | `quadratic_cpu<T>` | 50-54 | 把 `quadratic_value` 交给通用循环 |
| CPU 兼容实现 | 同上 | `quadratic<T>` | 56-60 | 直接转发到 `quadratic_cpu` |
| CPU 显式实例化 | 同上 | `INSTANTIATE_BSPLINE_CPU` | 62-76 | 生成五类型定义 |
| GPU buffer 校验 | `cusignal_cpp/src/bsplines/bsplines_typed.cu` | `require_compatible_buffers<T>` | 13-22 | 编译期 dtype 门禁和运行期长度检查 |
| GPU host wrapper | 同上 | `quadratic_device<T>` | 55-63 | 校验、空数组短路、启动 kernel |
| GPU 显式实例化 | 同上 | `INSTANTIATE_BSPLINE_GPU` | 65-76 | 生成五类型 GPU 入口 |
| GPU device 核心 | `cusignal_cpp/src/bsplines/bsplines_kernels.cuh` | `quadratic_kernel<T>` | 45-60 | 每线程计算一个输出元素 |
| 1D 启动配置 | `cusignal_cpp/src/cuda_utils/kernel_launch.h` | `make_1d_launch_config`、`launch_1d_kernel` | 21-29、44-56 | 256 threads/block、向上取整 grid、默认 stream |
| 设备数组 | `cusignal_cpp/src/cuda_utils/device_array.h` | `DeviceArray<T>` | 19-121 | 设备内存所有权、显式 H2D/D2H、size/data 接口 |
| typed smoke | `cusignal_cpp/test/signal_processing/e3_bsplines_type_smoke.cu` | `verify_case<T>`、`run_type<T>`、`main` | 71-92、94-206、210-226 | CPU/GPU 对照、五类型、边界和非法输入意图 |

### 3. 对外接口与类型合同

完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`。

#### 3.1 CPU reference

路径/符号/行号：`cusignal_cpp/src/bsplines/bsplines_typed.h`，`quadratic_cpu<T>`，195-206。

```cpp
template <typename T>
std::vector<T> quadratic_cpu(const std::vector<T>& x);
```

- 输入是 host 上连续存储的 `std::vector<T>`；接口只保存展平长度，不保存逻辑 shape。
- 输出是同长度、同元素类型的 `std::vector<T>`。
- 它是 GPU 正确性对照，不分配设备内存、不启动 kernel。

#### 3.2 deprecated CPU 兼容别名

路径/符号/行号：同一头文件，`quadratic<T>`，208-217。

```cpp
template <typename T>
[[deprecated("CPU compatibility alias; use quadratic_device for GPU execution or quadratic_cpu for reference")]]
std::vector<T> quadratic(const std::vector<T>& x);
```

`[[deprecated(... )]]` 允许旧代码继续编译，但编译器可发出弃用警告。它不是 GPU 路径，只转发到 `quadratic_cpu`。

#### 3.3 正式 GPU 入口

路径/符号/行号：同一头文件，`quadratic_device<T>`，219-231。

```cpp
template <typename T>
void quadratic_device(const DeviceArray<T>& x, DeviceArray<T>& y);
```

- `x` 是设备驻留只读输入，`const &` 避免复制并禁止通过该引用修改对象。
- `y` 是调用者预分配的设备输出，非 const 引用允许 kernel 写入其设备内存。
- 返回 `void`；结果写入 `y`。
- 不隐式执行 H2D/D2H，不创建 workspace，不调用 FFT、随机数或科学计算库。

#### 3.4 支持的五种类型

`quadratic_cpu` 和 `quadratic_device` 均显式支持：

| 业务类型 | C++ 类型 | 内部计算 | 写回 |
| --- | --- | --- | --- |
| FP32 | `float` | FP32 | `static_cast<float>` |
| FP16 | `__half` | `__half2float` 后 FP32 | `__float2half_rn` |
| INT32 | `std::int32_t` | `static_cast<float>` 后 FP32 | 浮点转整数 |
| INT16 | `std::int16_t` | 同上 | 浮点转整数 |
| INT8 | `std::int8_t` | 同上 | 浮点转整数 |

整数写回遵循 C++ 浮点到整数转换，分数部分向零截断；若值超出目标整数可表示范围，不能把结果泛化成有保证的饱和转换。当前 B 样条值位于 $[0,0.75]$，所以对正常有限输入，整数同型输出实际只会量化为 0。

### 4. 类型策略与数学 helper

#### 4.1 通用类型策略

路径/符号/行号：`cusignal_cpp/src/bsplines/bsplines_typed.h`，`BsplineTypePolicy<T>`，25-43。

```cpp
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
```

主模板支持 FP32 与三种整数。`ComputeT=float` 固定内部计算精度。`__host__ __device__` 让 `load/store` 可从 CPU 和 device 代码调用。

#### 4.2 FP16 特化

路径/符号/行号：同一头文件，`BsplineTypePolicy<__half>`，45-59。

```cpp
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
```

`template <>` 是对 `__half` 的全特化。它避免用普通 `static_cast` 处理平台 half，显式采用 half/float 转换 intrinsic。

#### 4.3 二次 B 样条 CPU 数学 helper

路径/符号/行号：同一头文件，`bspline_abs<T>` 64-68，`quadratic_value<T>` 85-97。

```cpp
template <typename T>
inline float bspline_abs(T value)
{
    return std::fabs(BsplineTypePolicy<T>::load(value));
}

template <typename T>
inline T quadratic_value(T value)
{
    const float ax = bspline_abs(value);
    float result = 0.0F;
    if (ax < 0.5F) {
        result = 0.75F - ax * ax;
    } else if (ax < 1.5F) {
        const float distance = ax - 1.5F;
        result = 0.5F * distance * distance;
    }
    return BsplineTypePolicy<T>::store(result);
}
```

执行顺序：load 到 FP32 → 求 $a=|x|$ → 默认结果 0 → 两段多项式 → 写回 T。由于结果初值为零，不需要显式第三个 `else`。

### 5. CPU 调用链与算法

#### 5.1 通用 CPU 迭代器

路径/符号/行号：`cusignal_cpp/src/bsplines/bsplines_typed.cpp`，`evaluate_cpu<T, Evaluator>`，9-18。

```cpp
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
```

- `static_assert` 在模板实例化时拒绝五类型之外的 T。
- 输出一次性分配为 `x.size()`。
- 普通 for 循环按索引处理，每个元素互不依赖。
- 空输入时循环零次，返回空 vector。

#### 5.2 `quadratic_cpu`

路径/符号/行号：同一文件，`quadratic_cpu<T>`，50-54。

```cpp
template <typename T>
std::vector<T> quadratic_cpu(const std::vector<T>& x)
{
    return evaluate_cpu(x, [](T value) { return detail::quadratic_value(value); });
}
```

无捕获 lambda `[](T value)` 把单个元素转发给数学 helper。CPU 完整调用链为：

```text
quadratic_cpu(x)
  → evaluate_cpu(x, lambda)
  → 对每个 i 调用 lambda(x[i])
  → detail::quadratic_value(x[i])
  → BsplineTypePolicy<T>::load
  → 分段 FP32 公式
  → BsplineTypePolicy<T>::store
  → y[i]
```

#### 5.3 兼容别名

路径/符号/行号：同一文件，`quadratic<T>`，56-60。

```cpp
template <typename T>
std::vector<T> quadratic(const std::vector<T>& x)
{
    return quadratic_cpu(x);
}
```

没有隐式设备执行。它只保留旧 host API 的源代码兼容性。

#### 5.4 CPU 显式实例化

路径/符号/行号：同一文件，`INSTANTIATE_BSPLINE_CPU`，62-76。宏中与本算子直接相关的两行为：

```cpp
template std::vector<T> quadratic_cpu(const std::vector<T>&); \
template std::vector<T> quadratic(const std::vector<T>&)
```

宏分别以 `float`、`__half`、`std::int32_t`、`std::int16_t`、`std::int8_t` 展开，促使本翻译单元生成五种具体函数定义。头文件的 `extern template` 则抑制其他翻译单元重复隐式实例化 `quadratic_cpu`。

### 6. GPU host wrapper

#### 6.1 buffer 兼容性检查

路径/符号/行号：`cusignal_cpp/src/bsplines/bsplines_typed.cu`，`require_compatible_buffers<T>`，13-22。

```cpp
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
```

它只比较展平元素数量，不能证明调用者保存的多维 shape 完全一致。失败时实际调用项目 `throw_error`。

重要差异：头文件注释写 `@throws std::invalid_argument`，但本提交 `cusignal_cpp/src/common/errors.h:83-100` 显示 `throw_error` 抛出 `ProjectError`，而 `ProjectError` 继承 `std::runtime_error`，不是 `std::invalid_argument`。正式语义应以代码事实为准：异常对象为 `ProjectError`，其项目错误码是 `INVALID_SHAPE`。

#### 6.2 `quadratic_device`

路径/符号/行号：`cusignal_cpp/src/bsplines/bsplines_typed.cu`，`quadratic_device<T>`，55-63。

```cpp
template <typename T>
void quadratic_device(const DeviceArray<T>& x, DeviceArray<T>& y)
{
    require_compatible_buffers(x, y, "quadratic_device");
    if (!x.empty()) {
        cuda_utils::launch_1d_kernel(
            bsplines_detail::quadratic_kernel<T>, x.size(), x.data(), y.data(), x.size());
    }
}
```

调用顺序：

1. 无论是否为空，先检查输入输出长度；
2. 两者同为空时跳过 kernel，正常返回；
3. 非空时把 kernel 符号、元素数、输入指针、输出指针、count 传给统一 launcher；
4. wrapper 本身不显式同步。

#### 6.3 GPU 显式实例化

路径/符号/行号：同一文件，`INSTANTIATE_BSPLINE_GPU`，65-76。与本算子相关：

```cpp
template void quadratic_device(const DeviceArray<T>&, DeviceArray<T>&)
```

宏对五种业务类型展开，与头文件 263-267 行的五条 `extern template` 对应。

### 7. GPU kernel、并行布局与数据范围

#### 7.1 device 核心

路径/符号/行号：`cusignal_cpp/src/bsplines/bsplines_kernels.cuh`，`quadratic_kernel<T>`，45-60。

```cpp
template <typename T>
__global__ void quadratic_kernel(const T* x, T* y, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float absolute_value = fabsf(detail::BsplineTypePolicy<T>::load(x[index]));
    float result = 0.0F;
    if (absolute_value < 0.5F) {
        result = 0.75F - absolute_value * absolute_value;
    } else if (absolute_value < 1.5F) {
        const float distance = absolute_value - 1.5F;
        result = 0.5F * distance * distance;
    }
    y[index] = detail::BsplineTypePolicy<T>::store(result);
}
```

`__global__` 表示 host 启动、device 执行。一个线程负责一个展平元素 `index`：读取 `x[index]`，FP32 计算，写 `y[index]`。没有相邻元素访问、共享内存、原子操作、循环或线程间通信。

#### 7.2 grid 和 block

路径/符号/行号：`cusignal_cpp/src/cuda_utils/kernel_launch.h`，`make_1d_launch_config` 21-29，`launch_1d_kernel` 44-56。

默认：

$$
B=256,\qquad G=\left\lceil\frac{N}{256}\right\rceil.
$$

- `blockDim.x = 256`；
- `gridDim.x = div_up(N,256)`；
- 总启动线程数为 $256G$；
- `index >= count` 的线程立即返回，保护最后一个不满 block；
- 第三个 launch 参数为 0，动态 shared memory 为 0；
- stream 参数为 `nullptr`，即项目注释所称默认 stream。

#### 7.3 启动错误与同步

launcher 在 kernel launch 后调用 `CUDA_KERNEL_CHECK()`，其固定提交实现最终执行 `cudaGetLastError()`。这只能检查启动/已报告错误，不等价于等待 kernel 完成。

`quadratic_device` 不同步。若调用者随后使用 `DeviceArray<T>::to_host()`，`device_array.h:66-74` 会执行 D2H 并调用 `synchronize_stream()`；若继续在 device 上消费 `y`，则依赖同 stream 顺序或由上层安排同步。

### 8. 内存布局与数据搬运

`DeviceArray<T>` 是 move-only 的一维连续设备所有者：

- 构造 `DeviceArray(count)` 分配 `count*sizeof(T)`；
- `data()` 返回原始设备指针；
- `size()` 返回展平元素数量；
- `from_host(vector)` 显式 H2D 并同步；
- `to_host()` 显式 D2H 并同步；
- `quadratic_device` 自己不做任何传输。

所谓“支持任意逻辑 rank”不是 kernel 认识多维坐标，而是上层把任意 shape 展平成连续 row-major 数组，kernel 只保持元素数量和线性顺序。C++ 接口没有广播。

输入 `x` 和输出 `y` 的 aliasing 未被显式禁止或检查。由于每线程先读后写相同 index，完全同址在算法上可能可行，但 `DeviceArray` 所有权模型并未提供正式共享所有权/别名 API，不能把原地执行声明为已批准合同。

### 9. 数学—Python—CPU—GPU 三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $a=|x|$ | `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py` 学习副本 111；基准 83 | `bsplines_typed.h:88`，`bspline_abs:64-68` | `bsplines_kernels.cuh:51` | C++ 先 load 到 FP32；Python 的 `T` 由 CuPy 决定 |
| $a<0.5$ | 学习副本 114；基准 85 | `bsplines_typed.h:90` | `bsplines_kernels.cuh:53` | 条件相同 |
| $3/4-a^2$ | 学习副本 116；基准 86 | `bsplines_typed.h:91` | `bsplines_kernels.cuh:54` | C++ 使用带 `F` 的 FP32 字面量 |
| $0.5\le a<1.5$ | 学习副本 118；基准 87 | `bsplines_typed.h:92` | `bsplines_kernels.cuh:55` | C++ 借前一 `else` 隐含下界，不重复写否定条件 |
| $\frac12(a-1.5)^2$ | 学习副本 120；基准 88 | `bsplines_typed.h:93-94` | `bsplines_kernels.cuh:56-57` | C++ 先保存 `distance`，数学等价 |
| $a\ge1.5\Rightarrow0$ | 学习副本 122-124；基准 89-90 | `result=0`：89，未命中分支即保留 | `result=0`：52，未命中分支即保留 | 控制流写法不同，结果等价 |
| 逐元素保形 | `cp.ElementwiseKernel` | `evaluate_cpu` for 循环 | 一线程一元素 | C++ shape 在接口外部持有，仅按 size 保形 |

数学不变量是阶段一的同一个中心化二次基数 B 样条。C++ 没有使用递归、数值卷积、FFT 或近似替换；“FP32 内部计算后写回五种 T”是工程扩展。

### 10. CPU 与 GPU 的相同、等价和有意差异

| 维度 | CPU | GPU | 结论 |
| --- | --- | --- | --- |
| 数学分段 | `quadratic_value` | `quadratic_kernel` | 公式相同 |
| 并行方式 | host for 循环 | 一线程一元素 | 实现不同、逐元素等价 |
| 内部精度 | FP32 | FP32 | 相同 |
| 输出类型 | store 回 T | store 回 T | 相同 |
| 内存 | `std::vector` | 调用者预分配 `DeviceArray` | 所有权不同 |
| 空输入 | 返回空 vector | x/y 同空时跳过 launch | 语义等价 |
| shape 错误 | CPU 输出自行按 x 分配，无不匹配可能 | x/y size 不同抛项目错误 | GPU 特有校验 |
| 同步 | 普通同步 CPU 返回 | wrapper 异步，只查 launch error | GPU 调用者负责完成同步 |

### 11. 精度、dtype、边界和 ZQ500 限制

#### 11.1 数值边界

- $|x|=0.5$ 进入第二分支，结果 $0.5$；
- $|x|=1.5$ 未进入第二分支，保留零；
- 分段值在两个边界连续；
- NaN：`fabsf(NaN)` 为 NaN，两个 `<` 均为假，C++ 返回/写回 0；与 Python 分支效果一致；
- 无穷大：两个分支均不命中，输出 0。

#### 11.2 dtype 限制

- 正式业务集合为 FP32、FP16、INT32、INT16、INT8；
- ZQ500 当前不支持 FP64，正式 GPU 路径没有 double 实例化；
- FP16 与整数都提升到 FP32 计算；
- INT 输出量化会丢失所有 $(0,1)$ 分数，因而对本算子正常结果几乎总为 0，这是同型业务合同的代价，不是数学基函数本身为零。

#### 11.3 平台与构建边界

- ZQ500 兼容 CUDA 11.7 但不等同于 NVIDIA CUDA；实际 API 以平台 SDK 为准；
- 本 kernel 只使用基础索引、`fabsf`、模板 load/store，无 FFT 或供应商科学库依赖；
- CMake 必须在 `gpu_02` 初始化 SDK 后执行，当前实测 3.22.1，项目最低兼容仍为 3.16；
- 本学习阶段不构建、不运行，不能把静态审阅写成 ZQ500 实测结果。

### 12. 复杂度和资源

设元素数为 $N$：

- CPU 时间：$O(N)$；输出空间 $O(N)$；每元素临时空间 $O(1)$。
- GPU 工作量：$O(N)$；理论并行深度为常数级分段计算；输出设备空间由调用者预分配为 $O(N)$。
- 每线程：一个 `size_t index`、若干 FP32 局部量，无动态 shared memory、无原子、无循环。
- 访问模式：相邻线程读写相邻 `x[index]`/`y[index]`，属于连续线性访问。
- 分支：同一 warp 内元素可能落入不同区间，存在分支分歧，但各分支很短。
- 启动固定为 256 threads/block；源码没有为 `quadratic` 单独调参。

### 13. 测试如何验证以及未覆盖风险

#### 13.1 测试意图

路径/符号/行号：`cusignal_cpp/test/signal_processing/e3_bsplines_type_smoke.cu`。

`verify_case<T>`（71-92）执行：host input → `DeviceArray::from_host` → `quadratic_device` → `quadratic_cpu` reference → `to_host` → 容差比较。

`run_type<T>` 对五类场景收集证据：

1. 典型 7 元素输入：97-102、128-130；
2. 最小长度 1：132、142；
3. alternate 3 元素：133-134、143；
4. boundary 2 元素：135、144；
5. 逻辑 rank3 的展平 24 元素：145-151；
6. output size 不匹配：154、161-166；
7. 五个 dtype：`main` 214-218；
8. 总门禁期望 15 项：219-221。

容差为 FP32 `2e-6`、FP16 `2e-3`、整数 0（43-52）。

#### 13.2 已发现的异常类型不一致

测试 161-166 行只捕获：

```cpp
catch (const std::invalid_argument&)
```

但 `require_compatible_buffers` 调用的 `throw_error` 在本提交抛 `ProjectError : std::runtime_error`。因此按固定 SHA 的类型层次，非法 size 异常不会命中这个局部 catch，而会外溢至 `main` 的 `catch (const std::exception&)` 并令程序返回 1。头文件的 `@throws std::invalid_argument` 同样与实现不一致。

这是静态源码事实和风险，不在 Learning 阶段修改。任何既有“PASS”文档都不能替代对当前 SHA 重新运行或修复异常合同后的证据。

#### 13.3 其他覆盖不足

- CPU 与 GPU 使用同一类型策略和同一公式，比较能发现实现分歧，但不是独立的 SciPy/cuSignal oracle；
- 名为 `boundary_input` 的 `{-4,4}` 只是支撑外输入，不是实际分段边界 $\pm0.5,\pm1.5$；
- `minimum_input{0.5}` 覆盖 $+0.5$，没有精确覆盖 $-0.5$、$\pm1.5$；
- 整数测试在 `from_float<T>` 时已把小数种子截断，不能验证整数输入对半整数坐标的概念行为；
- rank3 只构造 24 个线性元素，没有把 shape 传入 API，验证的是展平长度处理而非多维索引；
- 没有本算子专属性能 benchmark、NaN/Inf、超大 N、原地 alias 或异步消费测试；
- 本会话没有实际执行 ZQ500 smoke。

### 14. 推荐阅读顺序

1. `bsplines_typed.h:195-231`：先辨认 CPU、deprecated alias、GPU 三个接口；
2. `bsplines_typed.h:25-68,85-97`：理解五类型 load/FP32/store 与数学 helper；
3. `bsplines_typed.cpp:9-18,50-60`：看 CPU 调用链；
4. `bsplines_typed.cu:13-22,55-63`：看 GPU wrapper、错误检查和 launch；
5. `bsplines_kernels.cuh:45-60`：逐句对应数学公式；
6. `kernel_launch.h:21-29,44-56`：算 grid/block；
7. `device_array.h:19-121`：区分内存所有权和显式传输；
8. `e3_bsplines_type_smoke.cu:71-92,94-206,210-226`：检查测试意图及异常类型缺口。

### 15. 自检问题与参考答案

#### 问题 1：V1 为什么可形成正式学习版本？

**参考答案 1：** 它绑定完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`、提交时间与分支；检查时 `git status --short` 无输出，相关源码不是未提交快照。代码证据均可通过 `git show <SHA>:<path>` 重现。

#### 问题 2：正式 CPU 与 GPU 入口分别是什么？deprecated `quadratic` 属于哪一层？

**参考答案 2：** CPU reference 是 `quadratic_cpu<T>`（`bsplines_typed.h:195-206`）；正式 GPU 入口是 `quadratic_device<T>`（219-231）；`quadratic<T>`（208-217）是 deprecated CPU 兼容别名，只转发 `quadratic_cpu`，不属于 GPU 主路径。

#### 问题 3：五种 dtype 如何共享同一数学公式？

**参考答案 3：** `BsplineTypePolicy<T>` 先把 T load 为 `float`，CPU/GPU 都用 FP32 分段式计算，再 store 回 T。`__half` 使用专门的 `__half2float`/`__float2half_rn`，整数使用 `static_cast`。显式实例化只生成 FP32、FP16、INT32、INT16、INT8。

#### 问题 4：一个 GPU 线程负责什么？grid 如何计算？

**参考答案 4：** `quadratic_kernel` 中线程索引为 `blockIdx.x*blockDim.x+threadIdx.x`，每线程最多读一个 `x[index]` 并写一个 `y[index]`。默认 block 为 256，grid 为 $\lceil N/256\rceil$；多余线程由 `index>=count` 返回。

#### 问题 5：GPU wrapper 是否同步或搬运数据？

**参考答案 5：** 否。`quadratic_device` 只校验 size、空输入短路并 launch。H2D/D2H 由 `DeviceArray::from_host/to_host` 显式完成；launcher 只用 `cudaGetLastError` 检查 launch，不等待 kernel 完成。

#### 问题 6：CPU、GPU 和 Python 的数学公式是否一致？

**参考答案 6：** 一致。三者都先求绝对值，按 $a<0.5$、$0.5\le a<1.5$、$a\ge1.5$ 计算 $0.75-a^2$、$0.5(a-1.5)^2$、0。C++ 把第二段距离保存为局部变量并用默认零省略最终 else，属于等价实现。

#### 问题 7：整数输出为什么几乎全为 0？

**参考答案 7：** 数学结果范围为 $[0,0.75]$；C++ 内部虽以 FP32 算出分数，`store` 回整数时向零截断，所以正常有限输入的结果变成 0。这是五类型同型量化合同的工程代价，不是原始 B 样条恒为零。

#### 问题 8：shape 不匹配实际抛什么？测试为何存在风险？

**参考答案 8：** `require_compatible_buffers` 调用 `throw_error(INVALID_SHAPE,...)`，而 `errors.h:83-100` 的 `throw_error` 抛 `ProjectError : std::runtime_error`。smoke 只局部捕获 `std::invalid_argument`，两者无继承关系，因此该异常会外溢，当前测试的 invalid case 合同与实现不一致。

#### 问题 9：测试覆盖了哪些边界，又漏了什么？

**参考答案 9：** 覆盖典型、长度 1、支撑外、24 元素展平、五 dtype 和 size mismatch 意图；`0.5` 对浮点/half 被覆盖。但未精确覆盖 $-0.5$、$\pm1.5$，`{-4,4}` 不是分段边界，整数种子提前截断，且没有独立 upstream oracle、NaN/Inf 或大规模测试。

#### 问题 10：ZQ500 平台限制如何影响本实现？

**参考答案 10：** 正式 device 路径不能依赖 FP64，因此只实例化五业务类型并统一 FP32 计算；基础 CUDA 语义仍须以 ZQ500 SDK 为准。此 kernel 不依赖 FFT/科学库，使用基础 1D launch 与 `fabsf`。任何构建验证必须在 `gpu_02` 初始化 `/zq500/sdk/env.sh` 后进行，本学习会话未执行。

## 版本差异与原理不变量

目前只有 V1，暂无可比较优化版本。后续追加版本时至少比较：数学分段、复杂度、load/store 精度、内存访问、block/grid、同步、错误类型、测试覆盖和 ZQ500 结果。

无论工程实现如何优化，只要仍复现同一 `quadratic`，原理不变量应保持为：偶对称、支撑 $[-1.5,1.5]$、中央段 $3/4-|x|^2$、外侧段 $\frac12(|x|-1.5)^2$、支撑外为零。若未来改变这些内容，就不是单纯“原理不变、实现优化”，必须重新映射阶段一原理。

## V1 完整相关源码逐行解释

本节把前文的结构性说明落实为逐行覆盖。解释边界严格限定为 `quadratic` 的公共声明、类型/数学 helper、CPU 实现、GPU wrapper、device kernel、直接启动/错误基础设施，以及 typed smoke 中实际验证 `quadratic` 的语句。同文件内只服务 `cubic`、`gauss_spline` 的算法行不属于当前算子，不在本节复制。

所有位置均属于完整 SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`。

### A. 公共接口与模板实例化声明

#### A.1 CPU reference 声明

`cusignal_cpp/src/bsplines/bsplines_typed.h` 第 195-206 行：

```cpp
/**
 * @brief 在 CPU 上计算二次 B-spline，作为正式 GPU 实现的数值 reference。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；返回元素类型与输入一致。
 * @param x 任意逻辑 rank 的连续 row-major host 输入；shape 由调用者持有，允许为空。
 * @return 与 `x` 同 shape、展平存储的 host vector；内部提升为 FP32，按 `|x| < 0.5`、
 * `0.5 <= |x| < 1.5` 两段多项式计算，`|x| >= 1.5` 输出零，再写回 T。
 * @details 不分配设备资源、不调用 CUDA API、dlfft 或 dlrand。
 * @throws std::bad_alloc host 输出分配失败。
 * @note 对齐 cuSignal quadratic 逐元素公式；无 stream 或 workspace，整数同型输出是项目扩展。
 */
template <typename T>
std::vector<T> quadratic_cpu(const std::vector<T>& x);
```

- `/**` 开始 Doxygen 文档块，`*/` 结束它；中间每行开头的 `*` 是文档排版前缀。
- `@brief` 给出接口职责：CPU 实现是 GPU 的数值 reference，不是正式设备主路径。
- `@tparam T` 列出五种允许实例化的业务类型，并说明输入输出同型。
- `@param x` 说明 host 输入连续、按 row-major 展平；本接口只接收 vector，所以逻辑 rank/shape 由外层保存，空 vector 合法。
- 两行 `@return` 属于同一说明：输出与输入等长，计算时先提升到 FP32，使用两个非零分段，支撑外保持零，最后转回 T。
- `@details` 明确 CPU reference 不接触设备资源和科学库。
- `@throws` 记录唯一直接可见的分配失败风险：构造输出 vector 可能抛 `std::bad_alloc`。
- `@note` 把公式来源、无 stream/workspace 与整数同型扩展记录为合同边界。
- `template <typename T>` 声明函数模板；下一行返回 `std::vector<T>`，参数 `const std::vector<T>&` 表示不复制且不修改输入，分号表明这里只有声明、没有函数体。

#### A.2 deprecated CPU 兼容别名声明

同一文件第 208-217 行：

```cpp
/**
 * @brief 已降级的二次 B-spline CPU 兼容别名；不属于正式 GPU 主路径。
 * @deprecated 新业务代码使用 `quadratic_device`；CPU 对照显式使用 `quadratic_cpu`。
 * @tparam T 与 `quadratic_cpu` 相同的五类型集合。
 * @param x 一维 host 输入。
 * @return 直接返回 `quadratic_cpu(x)`，不会隐式调用 GPU 或科学计算库。
 */
template <typename T>
[[deprecated("CPU compatibility alias; use quadratic_device for GPU execution or quadratic_cpu for reference")]]
std::vector<T> quadratic(const std::vector<T>& x);
```

- 文档块逐项说明这是 CPU 旧名、不是 GPU 入口；新业务应调用 `quadratic_device`，CPU 对照应调用 `quadratic_cpu`。
- `@tparam` 继承同一五类型集合，`@param` 表示 host vector，`@return` 明确只做 CPU 转发。
- `template <typename T>` 仍声明模板。
- `[[deprecated("...")]]` 是标准 C++ 属性：调用该符号时编译器可显示括号中的迁移提示，但符号仍可链接。
- 最后一行声明返回同型 vector 的 `quadratic`，分号结束声明。

#### A.3 正式 GPU 声明

同一文件第 219-231 行：

```cpp
/**
 * @brief 在 ZQ500 GPU 上计算二次 B-spline，是该算子的唯一正式五类型 GPU 入口。
 * @tparam T 仅支持 FP32、FP16、INT32、INT16、INT8；输入输出必须同型。
 * @param x 任意逻辑 rank、连续 row-major、设备驻留的只读输入。
 * @param y 调用者预分配的同 shape 展平设备输出，长度必须等于 `x.size()`；允许均为空。
 * @throws std::invalid_argument `x`、`y` 长度不一致时抛出。
 * @details 默认 stream 上启动 template custom kernel，使用 FP32 分段多项式后写回 T；
 * 无 workspace、dlfft、dlrand 或隐式 H2D/D2H，调用者读取 `y` 前负责同步。
 * @note 对齐 cuSignal quadratic 的逐元素实数公式；整数输入属于项目五类型同型扩展，
 * 不是 cuSignal 的隐式浮点输出规则。
 */
template <typename T>
void quadratic_device(const DeviceArray<T>& x, DeviceArray<T>& y);
```

- `@brief` 将它确认为唯一正式 GPU 入口；`@tparam` 再次约束五种同型输入输出。
- `@param x` 描述只读设备输入，`@param y` 描述由调用者预分配的设备输出；两者只把多维 shape 当作外部元数据，函数内部比较展平长度。
- `@throws` 声称长度错误抛 `std::invalid_argument`，但固定提交的实现实际抛 `ProjectError`；这条注释与代码的不一致已在第 13 节保留。
- 两行 `@details` 共同说明默认 stream、FP32 内部计算、无 workspace/科学库/隐式传输，并要求调用者负责完成同步。
- 两行 `@note` 说明数学公式对齐 Python，但五业务类型同型返回是项目合同。
- 模板声明返回 `void`；`x` 是 const 引用，`y` 是可写引用，分号结束声明。

#### A.4 五类型显式实例化声明

同一文件第 245-249、263-267 行：

```cpp
extern template std::vector<float> quadratic_cpu(const std::vector<float>&);
extern template std::vector<__half> quadratic_cpu(const std::vector<__half>&);
extern template std::vector<std::int32_t> quadratic_cpu(const std::vector<std::int32_t>&);
extern template std::vector<std::int16_t> quadratic_cpu(const std::vector<std::int16_t>&);
extern template std::vector<std::int8_t> quadratic_cpu(const std::vector<std::int8_t>&);

extern template void quadratic_device(const DeviceArray<float>&, DeviceArray<float>&);
extern template void quadratic_device(const DeviceArray<__half>&, DeviceArray<__half>&);
extern template void quadratic_device(const DeviceArray<std::int32_t>&, DeviceArray<std::int32_t>&);
extern template void quadratic_device(const DeviceArray<std::int16_t>&, DeviceArray<std::int16_t>&);
extern template void quadratic_device(const DeviceArray<std::int8_t>&, DeviceArray<std::int8_t>&);
```

- 前五行分别声明 FP32、FP16、INT32、INT16、INT8 的 `quadratic_cpu` 显式实例化定义位于其他翻译单元；`extern template` 抑制包含此头文件的其他翻译单元重复生成它们。
- 后五行对 `quadratic_device` 做同样处理；每行的输入输出 `DeviceArray` 模板实参完全相同，落实同型合同。
- 每行末尾分号结束显式实例化声明；这十行只影响模板代码生成和链接，不执行算子计算。

### B. 类型策略与单元素数学核心

#### B.1 主类型策略

`cusignal_cpp/src/bsplines/bsplines_typed.h` 第 25-43 行：

```cpp
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
```

- `template <typename T>` 和 `struct BsplineTypePolicy` 建立按元素类型选择行为的策略模板，开花括号进入结构体。
- `static constexpr bool supported =` 开始编译期支持标志；随后四个 `std::is_same_v` 分别比较 T 与 FP32、INT32、INT16、INT8，`||` 表示任一相同即支持，分号结束常量定义。
- `using ComputeT = float;` 把所有主模板类型的内部计算类型统一别名为 FP32。
- `__host__ __device__ static ComputeT load(T value)` 声明 CPU/device 均可调用的静态 load；函数体用 `static_cast<float>` 把 T 转为 FP32并返回。
- `__host__ __device__ static T store(ComputeT value)` 是反向写回；函数体用 `static_cast<T>` 把 FP32 转回业务类型。
- 每个 `}` 分别关闭函数体，最后的 `};` 关闭结构体定义并以分号结束类型声明。

#### B.2 FP16 全特化

同一文件第 45-59 行：

```cpp
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
```

- `template <>` 表示后面不是新的泛型模板，而是 `__half` 的完整特化。
- `supported=true` 把 FP16 加入支持集合，`ComputeT=float` 继续规定 FP32 内部计算。
- `load(__half value)` 使用 `__half2float`，而不是主模板的普通 cast；`store` 使用 `__float2half_rn` 按最近值舍入回 half。
- 花括号与分号分别结束两个函数和特化结构体。

#### B.3 支持变量模板与绝对值 helper

同一文件第 61-68 行：

```cpp
template <typename T>
inline constexpr bool is_bspline_input_v = BsplineTypePolicy<T>::supported;

template <typename T>
inline float bspline_abs(T value)
{
    return std::fabs(BsplineTypePolicy<T>::load(value));
}
```

- 第一组两行定义变量模板 `is_bspline_input_v<T>`，直接转发类型策略的 `supported`；`inline constexpr` 允许它安全地在头文件中定义并用于 `static_assert`。
- 第二个 `template` 开始 `bspline_abs` 函数模板；返回类型固定为 float。
- 返回表达式先调用 `load(value)`，再用 `std::fabs` 求绝对值；最后的 `}` 结束 helper。

#### B.4 二次 B 样条单元素 helper

同一文件第 85-97 行：

```cpp
template <typename T>
inline T quadratic_value(T value)
{
    const float ax = bspline_abs(value);
    float result = 0.0F;
    if (ax < 0.5F) {
        result = 0.75F - ax * ax;
    } else if (ax < 1.5F) {
        const float distance = ax - 1.5F;
        result = 0.5F * distance * distance;
    }
    return BsplineTypePolicy<T>::store(result);
}
```

- 模板签名表示输入和返回均为 T，`inline` 使头文件定义可被多个翻译单元包含。
- `ax` 接收 load 后绝对值并保持只读；`result=0.0F` 先表示支撑外结果。
- 第一分支检查 $a<0.5$，下一行实现 $0.75-a^2$，随后 `}` 关闭中央段。
- `else if (ax<1.5)` 只在第一条件失败时进入，因此隐含 $a\ge0.5$；`distance=a-1.5` 保存外侧距离，下一行实现 $0.5d^2$。
- 未命中两个分支时 `result` 保持零；最后调用 `store` 转回 T 并返回，末尾花括号结束函数。

### C. CPU reference、兼容别名与代码生成

#### C.1 通用逐元素 CPU 循环

`cusignal_cpp/src/bsplines/bsplines_typed.cpp` 第 9-18 行：

```cpp
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
```

- 模板参数 T 表示元素类型，Evaluator 表示调用者传入的单元素计算器类型。
- 函数接收只读 vector 引用与 evaluator，返回同型 vector；花括号进入函数体。
- `static_assert` 在编译期拒绝不受策略支持的 T，字符串是编译错误提示。
- `y(x.size())` 一次性创建与输入等长的输出；空输入得到空 vector。
- for 头部依次初始化 `i=0`、检查 `i<x.size()`、每轮 `++i`；循环体把 `evaluator(x[i])` 写入同索引 `y[i]`。
- 循环花括号结束后返回 y，最后花括号结束 helper。

#### C.2 `quadratic_cpu` 与 deprecated host 转发

同一文件第 50-60 行：

```cpp
template <typename T>
std::vector<T> quadratic_cpu(const std::vector<T>& x)
{
    return evaluate_cpu(x, [](T value) { return detail::quadratic_value(value); });
}

template <typename T>
std::vector<T> quadratic(const std::vector<T>& x)
{
    return quadratic_cpu(x);
}
```

- 第一组模板签名实现正式 CPU reference；函数体只有一个 return。
- `evaluate_cpu` 的第二个实参是 lambda：`[]` 表示不捕获外部变量，`(T value)` 接收一个元素，花括号内把它交给 `detail::quadratic_value` 并返回；外层调用再返回完整 vector。
- 第二组模板签名实现旧名 `quadratic`；函数体直接 `return quadratic_cpu(x)`，证明它没有 GPU、传输或额外数学路径。

#### C.3 CPU 显式实例化定义

同一文件第 62-76 行中，与 `quadratic` 的生成直接相关的完整宏框架为：

```cpp
#define INSTANTIATE_BSPLINE_CPU(T) \
    template std::vector<T> cubic_cpu(const std::vector<T>&); \
    template std::vector<T> cubic(const std::vector<T>&); \
    template std::vector<T> gauss_spline_cpu(const std::vector<T>&, int); \
    template std::vector<T> gauss_spline(const std::vector<T>&, int); \
    template std::vector<T> quadratic_cpu(const std::vector<T>&); \
    template std::vector<T> quadratic(const std::vector<T>&)

INSTANTIATE_BSPLINE_CPU(float);
INSTANTIATE_BSPLINE_CPU(__half);
INSTANTIATE_BSPLINE_CPU(std::int32_t);
INSTANTIATE_BSPLINE_CPU(std::int16_t);
INSTANTIATE_BSPLINE_CPU(std::int8_t);

#undef INSTANTIATE_BSPLINE_CPU
```

- `#define` 建立函数式预处理宏，参数名 T；每行末尾反斜杠把宏体续到下一物理行。
- 中间四条 `cubic/gauss_spline` 行只是共享宏的原始组成，不在这里解释其算法；不能删除它们，否则摘录就不再是完整宏定义。
- 最后两条 template 行要求生成 `quadratic_cpu<T>` 与兼容别名 `quadratic<T>` 的显式实例化定义。
- 五次宏调用分别把 T 替换成 FP32、FP16、INT32、INT16、INT8；每个调用末尾分号结束展开后的最后一条声明。
- `#undef` 删除预处理宏，防止它泄漏到文件后续内容。

### D. GPU host wrapper 与显式实例化

#### D.1 输入输出 buffer 校验

`cusignal_cpp/src/bsplines/bsplines_typed.cu` 第 13-22 行：

```cpp
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
```

- 模板签名接收同型的只读输入、只读检查用输出引用，以及用于错误文本的 C 字符串名称。
- `static_assert` 执行编译期 dtype 门禁。
- `if` 比较两个展平 size；不相等时进入花括号。
- `throw_error` 的首个实参选择项目错误码 `INVALID_SHAPE`；后续两行把函数名、输入 size 和输出 size 拼成完整中文消息，最内层 `));` 依次关闭 `to_string`、`throw_error` 并结束语句。
- 最后两个 `}` 依次关闭 if 与 helper。

#### D.2 `quadratic_device` wrapper

同一文件第 55-63 行：

```cpp
template <typename T>
void quadratic_device(const DeviceArray<T>& x, DeviceArray<T>& y)
{
    require_compatible_buffers(x, y, "quadratic_device");
    if (!x.empty()) {
        cuda_utils::launch_1d_kernel(
            bsplines_detail::quadratic_kernel<T>, x.size(), x.data(), y.data(), x.size());
    }
}
```

- 模板签名定义正式 GPU 入口：x 是只读设备数组，y 是可写设备数组，返回 void。
- 首条函数体语句校验类型/长度，并把当前入口名传给错误消息。
- `if (!x.empty())` 对非空输入启动 kernel；同为空时跳过花括号内内容并正常返回。
- `launch_1d_kernel(` 开始跨行调用；下一行依次传 kernel 模板实例、用于算 grid 的元素数、输入指针、输出指针和传给 kernel 的 count，最后 `);` 关闭调用。
- 末尾两个 `}` 关闭 if 和 wrapper；函数没有同步或返回数组。

#### D.3 GPU 显式实例化定义

同一文件第 65-76 行的完整共享宏：

```cpp
#define INSTANTIATE_BSPLINE_GPU(T) \
    template void cubic_device(const DeviceArray<T>&, DeviceArray<T>&); \
    template void gauss_spline_device(const DeviceArray<T>&, DeviceArray<T>&, int); \
    template void quadratic_device(const DeviceArray<T>&, DeviceArray<T>&)

INSTANTIATE_BSPLINE_GPU(float);
INSTANTIATE_BSPLINE_GPU(__half);
INSTANTIATE_BSPLINE_GPU(std::int32_t);
INSTANTIATE_BSPLINE_GPU(std::int16_t);
INSTANTIATE_BSPLINE_GPU(std::int8_t);

#undef INSTANTIATE_BSPLINE_GPU
```

- `#define` 与行尾反斜杠组成跨行宏；前两条其他算子行只是保持原宏完整。
- 第三条 template 行生成 `quadratic_device<T>` 的显式实例化定义。
- 五次宏调用覆盖五业务 dtype；`#undef` 在使用完毕后删除宏。

### E. GPU device 数学核心

`cusignal_cpp/src/bsplines/bsplines_kernels.cuh` 第 45-60 行：

```cpp
template <typename T>
__global__ void quadratic_kernel(const T* x, T* y, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float absolute_value = fabsf(detail::BsplineTypePolicy<T>::load(x[index]));
    float result = 0.0F;
    if (absolute_value < 0.5F) {
        result = 0.75F - absolute_value * absolute_value;
    } else if (absolute_value < 1.5F) {
        const float distance = absolute_value - 1.5F;
        result = 0.5F * distance * distance;
    }
    y[index] = detail::BsplineTypePolicy<T>::store(result);
}
```

- `template <typename T>` 建立 dtype 模板；`__global__ void` 表示 host 启动、device 执行且无返回值。`const T* x` 是只读输入指针，`T* y` 是输出指针，`count` 是展平元素数。
- 索引行按 `blockIdx.x*blockDim.x+threadIdx.x` 计算全局线性 index。
- 越界行把比较和 `return` 写在同一物理行；`index>=count` 的补齐线程立即结束。
- `absolute_value` 行先读取 `x[index]`、经策略 load 到 FP32，再用 `fabsf` 求绝对值；嵌套括号从内向外依次关闭数组索引、load 和 fabsf。
- `result=0.0F` 预置支撑外结果。
- 第一 if 及赋值行实现 $a<0.5$ 时 $0.75-a^2$；右花括号关闭中央段。
- `else if` 在第一条件失败后检查 $a<1.5$；`distance=a-1.5` 保存外侧距离，下一行实现 $0.5d^2$；右花括号关闭第二段。
- 最后一条赋值通过策略 store 把 FP32 转回 T，再写到与输入同索引的 `y[index]`；最终花括号结束 kernel。这是本算子 device 核心的最后一行。

### F. 直接 1D 启动基础设施

#### F.1 向上取整与 launch 配置

`cusignal_cpp/src/cuda_utils/kernel_launch.h` 第 10-29 行：

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

- `div_up` 是索引类型模板；返回表达式用整数公式 $(value+divisor-1)/divisor$ 计算正整数向上取整，花括号结束函数。
- `LaunchConfig1D` 结构体依次保存 `dim3 block` 和 `dim3 grid`，`};` 结束类型。
- `make_1d_launch_config` 的两行参数分别是元素数与默认值 256 的 block 大小；返回类型是前述结构体。
- return 的列表初始化先构造 `dim3(block_size)`，再把 `div_up(elements, size_t(block_size))` 的结果转换为 unsigned int 并构造 grid；`};` 结束 return 对象，最后花括号结束函数。

#### F.2 默认 launcher

同一文件第 31-56 行：

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

- 两个模板都接收 kernel 类型和可变参数包 `Args...`；`Args&&...` 配合 `std::forward` 保留实参值类别。
- `launch_1d_kernel_with_config` 的参数逐行给出 kernel、元素数、stream、block 大小和转发参数；函数体先创建 config。
- `kernel<<<grid,block,0,stream>>>` 是 CUDA launch 语法：第三项 0 表示无动态 shared memory；圆括号内把所有算子参数转发给 device kernel。
- `CUDA_KERNEL_CHECK()` 检查 launch 后的 runtime error，随后花括号结束带配置版本。
- 简化版 `launch_1d_kernel` 接收 kernel、元素数和参数包；函数体调用带配置版本，并逐行固定 `stream=nullptr`、`block_size=256`，最后转发算子参数。
- `);` 关闭调用，最终 `}` 结束默认 launcher。

### G. 直接错误与设备数组基础设施

#### G.1 `throw_error` 的实际异常类型

`cusignal_cpp/src/common/errors.h` 第 83-101 行：

```cpp
class ProjectError final : public std::runtime_error {
public:
    explicit ProjectError(Error error)
        : std::runtime_error(error.message), error_(std::move(error))
    {
    }

    const Error& detail() const noexcept { return error_; }

private:
    Error error_;
};

[[noreturn]] inline void throw_error(ErrorCode code, std::string message,
                                     BackendError backend = {}, bool safe_exit = true)
{
    if (message.empty()) message = "项目操作失败，但未提供错误描述";
    throw ProjectError({code, std::move(message), std::move(backend), safe_exit});
}
```

- `ProjectError final : public std::runtime_error` 定义不可再派生的项目异常，并明确其基类是 `runtime_error`；`public:` 开始公开成员。
- 构造函数以 `explicit` 防止隐式 Error→ProjectError 转换；初始化列表先用消息构造基类，再移动保存完整 Error，空函数体结束构造。
- `detail()` 返回保存错误的 const 引用，`noexcept` 保证该访问器不抛异常。
- `private:` 后的 `Error error_` 是私有状态，`};` 结束类。
- `[[noreturn]]` 声明 `throw_error` 不会正常返回；两行签名依次接收项目错误码、消息、默认空 backend 和默认 true 的 safe_exit。
- 若消息为空，同一行 if 为它补默认中文文本。
- 最后一条 throw 用列表初始化 Error，并移动消息/backend 构造 ProjectError；最终花括号结束函数。由此可证 shape mismatch 实际不是 `std::invalid_argument`。

#### G.2 测试和 wrapper 直接使用的 `DeviceArray` 操作

`cusignal_cpp/src/cuda_utils/device_array.h` 第 24-27、56-74、93-96 行：

```cpp
explicit DeviceArray(std::size_t count)
{
    allocate(count);
}

static DeviceArray from_host(const std::vector<T>& host)
{
    DeviceArray out(host.size());
    if (!host.empty()) {
        cuda_utils::copy_to_device(out.data_, host.data(), host.size());
        cuda_utils::synchronize_stream();
    }
    return out;
}

std::vector<T> to_host() const
{
    std::vector<T> host(size_);
    if (size_ != 0) {
        cuda_utils::copy_to_host(host.data(), data_, size_);
        cuda_utils::synchronize_stream();
    }
    return host;
}

[[nodiscard]] std::size_t size() const noexcept { return size_; }
[[nodiscard]] bool empty() const noexcept { return size_ == 0; }
[[nodiscard]] T* data() noexcept { return data_; }
[[nodiscard]] const T* data() const noexcept { return data_; }
```

- 显式 count 构造函数调用 `allocate(count)`；`explicit` 阻止整数自动隐式变成 DeviceArray。
- `from_host` 是静态工厂：先按 host 长度构造输出；非空时复制 H2D 并同步；最后返回 move-only 设备所有者。
- `to_host() const` 创建同长度 host vector；非零时复制 D2H 并同步；最后返回 host 数据。
- 四个单行访问器依次返回 size、是否为空、可写设备指针、只读设备指针；`[[nodiscard]]` 提醒调用者不要无意忽略返回值，`noexcept` 表示访问元数据/指针不抛异常。

### H. typed smoke 中的逐行验证路径

测试文件是三个 B-spline 算子的共享 smoke。本节只摘录 `quadratic` 直接命中的行和为它服务的共享比较/入口语句；不展开另外两个算子的数学分支。

#### H.1 类型转换、容差和向量比较

`cusignal_cpp/test/signal_processing/e3_bsplines_type_smoke.cu` 第 18-67 行：

```cpp
template <typename T>
T from_float(float value)
{
    return static_cast<T>(value);
}

template <>
__half from_float<__half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
float to_float(T value)
{
    return static_cast<float>(value);
}

template <>
float to_float<__half>(__half value)
{
    return __half2float(value);
}

template <typename T>
float tolerance()
{
    if constexpr (std::is_same_v<T, __half>) {
        return 2.0e-3F;
    }
    if constexpr (std::is_integral_v<T>) {
        return 0.0F;
    }
    return 2.0e-6F;
}

template <typename T>
bool matches(const std::vector<T>& actual, const std::vector<T>& expected)
{
    if (actual.size() != expected.size()) {
        return false;
    }
    const float limit = tolerance<T>();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(to_float(actual[i]) - to_float(expected[i])) > limit) {
            return false;
        }
    }
    return true;
}
```

- 通用 `from_float<T>` 用 `static_cast<T>` 生成测试输入；half 全特化改用 `__float2half_rn`。因此整数测试种子在进入算子前已经丢掉小数部分。
- 通用 `to_float<T>` 把比较值转成 float；half 全特化使用 `__half2float`。
- `tolerance<T>` 用两个 `if constexpr` 在编译期选择容差：half 为 `2e-3`，整数为精确 0，其他受支持类型即 FP32 为 `2e-6`；各个花括号结束条件块和函数。
- `matches` 先比较 vector 长度，不同立即 false；随后取得当前 T 的 limit。
- for 循环遍历 actual；内层 if 把 actual/expected 同索引转换为 float，求差的绝对值并与 limit 比较，超限立即 false。
- 全部元素通过后返回 true，最终花括号结束函数。

#### H.2 `verify_case` 的 quadratic 分支

同一测试文件第 69-92 行：

```cpp
enum class BsplineOperation { cubic, gauss, quadratic };

template <typename T>
bool verify_case(const std::vector<T>& input, BsplineOperation operation, int n = 5)
{
    auto device_input = cusignal::DeviceArray<T>::from_host(input);
    cusignal::DeviceArray<T> device_output(input.size());
    std::vector<T> reference;
    switch (operation) {
    case BsplineOperation::cubic:
        cusignal::cubic_device(device_input, device_output);
        reference = cusignal::cubic_cpu(input);
        break;
    case BsplineOperation::gauss:
        cusignal::gauss_spline_device(device_input, device_output, n);
        reference = cusignal::gauss_spline_cpu(input, n);
        break;
    case BsplineOperation::quadratic:
        cusignal::quadratic_device(device_input, device_output);
        reference = cusignal::quadratic_cpu(input);
        break;
    }
    return matches(device_output.to_host(), reference);
}
```

- `enum class` 定义强类型操作枚举；`quadratic` 是第三个枚举值。
- `verify_case<T>` 接收 host input、要测试的操作和仅供 Gaussian 使用的默认 n；返回 bool。
- 首条函数体语句显式 H2D 创建 device_input，下一行按相同 size 分配 device_output，再创建空 CPU reference。
- `switch(operation)` 开始操作分派；两组其他算子 case 保留共享函数原文，但不属于本算子算法。
- `case ... quadratic:` 命中本算子；下一行调用 GPU wrapper，再下一行调用 CPU reference，`break` 退出 switch。
- switch 右花括号结束分派；最后 return 先 `to_host()` 执行 D2H/同步，再把 GPU 结果和 CPU reference 交给 `matches`，最终花括号结束 `verify_case`。

#### H.3 每种 dtype 的典型输入和主对照

同一测试文件第 94-105、128-135 行中服务 `quadratic` 的输入准备与主对照：

```cpp
template <typename T>
int run_type(const char* dtype)
{
    const std::vector<float> seed = {-3.0F, -1.25F, -0.25F, 0.0F, 0.75F, 1.25F, 3.0F};
    std::vector<T> x;
    x.reserve(seed.size());
    for (const float value : seed) {
        x.push_back(from_float<T>(value));
    }

    auto dx = cusignal::DeviceArray<T>::from_host(x);
    cusignal::DeviceArray<T> dy(x.size());

    cusignal::quadratic_device(dx, dy);
    const auto quadratic_actual = dy.to_host();
    const auto quadratic_reference = cusignal::quadratic_cpu(x);

    const std::vector<T> minimum_input{from_float<T>(0.5F)};
    const std::vector<T> alternate_input{
        from_float<T>(-2.0F), from_float<T>(0.5F), from_float<T>(2.5F)};
    const std::vector<T> boundary_input{from_float<T>(-4.0F), from_float<T>(4.0F)};
```

- 模板函数 `run_type<T>` 接收用于输出证据的 dtype 名称并返回通过项数量；开花括号进入函数。
- `seed` 是七个 FP32 坐标；创建空 `vector<T> x` 后，`reserve` 预留七个元素容量但不改变 size。
- range-for 依次取得 seed 元素；`from_float<T>` 转换成当前业务类型并 `push_back`，右花括号结束循环。
- `dx` 通过 `from_host` 显式 H2D；`dy(x.size())` 分配同长设备输出。
- `quadratic_device(dx,dy)` 启动 GPU 路径；`dy.to_host()` 同步取回 actual；`quadratic_cpu(x)` 产生 reference。
- `minimum_input` 是长度 1 的 $0.5$；`alternate_input` 的跨行初始化包含支撑外、分段边界与另一支撑外值；`boundary_input` 实际是 $\pm4$ 的支撑外输入。每条声明末尾分号结束 vector 构造。

#### H.4 边界、展平 rank3 与非法 size

同一文件第 142-165 行中 `quadratic` 相关语句：

```cpp
const bool quadratic_minimum = verify_case(minimum_input, BsplineOperation::quadratic);
const bool quadratic_alternate = verify_case(alternate_input, BsplineOperation::quadratic);
const bool quadratic_boundary = verify_case(boundary_input, BsplineOperation::quadratic);
std::vector<T> rank3_input;
rank3_input.reserve(24);
for (int index = 0; index < 24; ++index)
    rank3_input.push_back(from_float<T>(seed[static_cast<std::size_t>(index) % seed.size()]));
const bool quadratic_rank3 = verify_case(rank3_input, BsplineOperation::quadratic);

bool quadratic_invalid_ok = false;
try {
    cusignal::DeviceArray<T> wrong_size(1);
    cusignal::quadratic_device(dx, wrong_size);
} catch (const std::invalid_argument&) {
    quadratic_invalid_ok = true;
}
```

- 前三行分别把长度 1、alternate 和支撑外输入交给 `verify_case`，结果保存为 const bool。
- 创建 `rank3_input` 后预留 24 个元素；无花括号 for 的下一条物理行就是唯一循环体，通过 `index % seed.size()` 周期复用种子并转换到 T。
- `quadratic_rank3` 对这 24 个展平元素执行 CPU/GPU 比较；名称 rank3 只表示上层把它视为 `[2,3,4]`，API 本身没有 shape。
- `quadratic_invalid_ok` 初始 false；`try` 开始非法输出长度测试。
- `wrong_size(1)` 分配长度 1 输出，而典型 dx 长度为 7；调用 wrapper 应触发 shape 错误。
- `catch (const std::invalid_argument&)` 只匹配 invalid_argument 及其派生类，函数体把标志设 true；最后花括号结束 catch。
- 固定提交实际抛 `ProjectError : runtime_error`，所以该 catch 不会执行，异常会继续外溢。这不是测试期望，而是当前源码缺口。

#### H.5 指标字段和证据落盘调用

同一文件第 168-200 行中服务 `quadratic` 的语句：

```cpp
cusignal::cuda_utils::synchronize_stream();
namespace ev = cusignal::test_evidence;
namespace td = cusignal::type_dispatch;
auto quadratic_metrics = ev::compare_real_vectors(
    quadratic_actual, quadratic_reference, tolerance<T>());
quadratic_metrics.alternate_case_review = quadratic_alternate && quadratic_rank3;
quadratic_metrics.minimum_legal_case = quadratic_minimum;
quadratic_metrics.typical_case = true;
quadratic_metrics.boundary_case = quadratic_boundary;
quadratic_metrics.invalid_case = quadratic_invalid_ok;
const bool quadratic_evidence = ev::record_detailed<T, T>(
    td::OperatorId::quadratic, "typed-nondegenerate-suite", "[7]", "[7]",
    "typical=7;rank3=[2,3,4];min=1;alternate=3;boundary=2;invalid=output-size", quadratic_metrics);
```

- `synchronize_stream()` 在汇总证据前显式等待默认 stream；不过前面的 `to_host()` 已对主 actual 做过同步，这里也覆盖共享测试的其他异步工作。
- 两条 namespace alias 分别把长命名空间缩写为 `ev` 和 `td`。
- `compare_real_vectors(` 跨两行接收 actual、reference 与当前 T 容差，返回的指标对象由 `auto` 推导。
- 五条字段赋值依次记录 alternate 与 rank3 的合取、最小合法、典型、支撑外“boundary”以及非法 size 标志。
- `record_detailed<T,T>(` 用输入/输出同类型模板参数记录证据；下一行指定 `quadratic` 算子 ID、套件名和输入输出 shape 文本；最后一行写入场景摘要与 metrics，并以 `);` 结束调用。返回 bool 保存为 `quadratic_evidence`。

#### H.6 `run_type` 收尾与测试文件最后入口

同一文件第 201-226 行：

```cpp
std::cout << "[E3][bsplines][" << dtype << "] cubic=" << cubic_evidence
          << " gauss_spline=" << gauss_evidence
          << " quadratic=" << quadratic_evidence << '\n';
return static_cast<int>(cubic_evidence) + static_cast<int>(gauss_evidence) +
    static_cast<int>(quadratic_evidence);
}

}  // namespace

int main()
{
    try {
        int passed = 0;
        passed += run_type<float>("fp32");
        passed += run_type<__half>("fp16");
        passed += run_type<std::int32_t>("int32");
        passed += run_type<std::int16_t>("int16");
        passed += run_type<std::int8_t>("int8");
        std::cout << "[E3][bsplines] total=15 pass=" << passed
                  << " fail=" << (15 - passed) << '\n';
        return passed == 15 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "[E3][bsplines][FAIL] " << ex.what() << '\n';
        return 1;
    }
}
```

- 三行 `std::cout` 是一个连续流表达式：依次输出 dtype 下三个共享 B-spline 证据布尔值，最后输出换行字符；与当前算子直接相关的是 `quadratic_evidence`。
- 两行 return 把三个 bool 显式转换为 int 后相加；`quadratic_evidence` 对每个 dtype 最多贡献 1。右花括号结束 `run_type`。
- `} // namespace` 关闭匿名命名空间；`main` 因而位于全局作用域。
- `int main()` 声明进程入口，下一行开花括号进入函数；`try` 开始总测试保护区。
- `passed=0` 初始化累计通过数；接下来五行分别以 FP32、FP16、INT32、INT16、INT8 实例化并调用 `run_type`，返回值累加到 passed。
- 两行总计输出是连续流表达式：套件共有 3 算子×5 dtype=15 项，失败数按 `15-passed` 计算。
- 条件运算符 `passed==15 ? 0 : 1` 在全 15 项通过时返回进程码 0，否则返回 1。
- `catch (const std::exception& ex)` 捕获从 try 外溢的标准异常派生对象；下一行把 `ex.what()` 写到标准错误并换行，再返回 1。
- 倒数第二个 `}` 结束 catch，最后一个 `}` 结束 `main`。这是当前 typed smoke 源文件的最后一行，也完成阶段三相关代码的逐行解释。

### I. 逐行覆盖结论

- 阶段二 `python源码算法.md` 第 3、4 节已经从完整 docstring、导入、两级导出、完整 `ElementwiseKernel` 到 `return _quadratic_kernel(x)` 覆盖最后一行。
- 本节按“接口 → 类型/数学 helper → CPU → GPU wrapper → kernel → launcher/error/DeviceArray → 测试 main 最后一行”完成阶段三覆盖。
- 共享宏、共享 switch 和共享测试收尾中不得删除的其他算子名称仅用于保持原始语义块完整；没有展开其他算子的数学实现。
