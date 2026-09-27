# cusignal_cpp_qmf 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前学习版本](#v1当前学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU/GPU 对齐 cuSignal 23.08.00 的实际 INT64 索引生成行为 |

## V1：当前学习版本（fdd55ac）

### 1. 版本身份与 dirty 状态

| 项目 | 值 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 提交时间 | `2026-08-15 23:45:58 +0800` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交标题 | `test(archive): 增加任务结果范围治理审计` |
| 仓库 | `ZKX_dev/ZKX` |
| 工作区整体状态 | `git status --short` 无输出，干净 |
| qmf 相关文件状态 | 下列源码、helper 与直接测试均无未提交修改 |

本章所有代码均通过 `git -C ZKX show fdd55ac8415d70379eb38a2c299047f90bcf0a41:<路径>` 读取，不依赖未来可能漂移的工作区内容。

### 2. 代码证据总表

| 职责 | 完整 SHA | 共享根目录 `ZKX_dev/` 下的路径 | 符号 | 提交内行号 |
| --- | --- | --- | --- | --- |
| 公共 CPU/GPU 声明 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h` | `qmf_typed_cpu`、`qmf_device` | 181--218 |
| CPU 一维核心 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp` | `qmf_typed_cpu(const std::vector<T>&)` | 285--295 |
| CPU 高维兼容核心 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp` | `qmf_typed_cpu(..., shape)` | 297--317 |
| CPU 显式实例化 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp` | `INSTANTIATE_CPU` 中的 qmf 项 | 319、338--346 |
| GPU 一维 wrapper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu` | `qmf_device(const DeviceArray<T>&, ...)` | 175--183 |
| GPU 高维兼容 wrapper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu` | `qmf_device(..., shape)` | 185--205 |
| GPU 显式实例化 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu` | `INSTANTIATE_WAVELET_GPU` 中的 qmf 项 | 207、213--223 |
| GPU device 核心 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh` | `wavelets_detail::qmf_kernel` | 33--41 |
| 1D 启动 helper | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h` | `div_up`、`make_1d_launch_config`、`launch_1d_kernel*` | 10--56 |
| 设备数组 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/cuda_utils/device_array.h` | `DeviceArray<T>` | 12--20、93--96 |
| 输入类型谓词 | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/src/cuda_utils/simple_signal_typed.h` | `is_simple_signal_input_v` | 49--50 |
| 五类型直接 smoke | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu` | `run_type<T>` 的 qmf 段 | 340--371、380--381、417--421 |
| 正式多尺度 probe | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | `ZKX/test_all/operators/cases/wavelets/qmf/operator_probe.cu` | `run<T>`、`main` | 22--25 |

### 3. 对外接口与完整 qmf 声明

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.h`，`qmf_typed_cpu` / `qmf_device`，181--218 行。

```cpp
/**
 * @brief 在 CPU 上由低通系数生成 QMF 高通系数 reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8。
 * @param hk host 一维低通系数 `[N]`。
 * @return host上`[N]`、dtype INT64的索引序列；空输入返回空数组。
 * @details 固定版cuSignal 23.08.00不读取hk数值，只使用长度：
 * `out[i]=(N-(i+1))*(i为奇数?-1:1)`；CPU reference不创建GPU stream或device
 * workspace，五类型含FP32输入均只贡献shape。
 * @throws std::bad_alloc host 输出分配失败。
 */
template<class T>
std::vector<std::int64_t> qmf_typed_cpu(const std::vector<T>& hk);

/**
 * @brief 固定版23.08高维兼容入口。
 * @details `len(hk)`只取首维，因此输入可为任意rank连续数组，输出shape为`[shape[0]]`。
 */
template<class T>
std::vector<std::int64_t> qmf_typed_cpu(
    const std::vector<T>& hk, const std::vector<int>& shape);

/**
 * @brief 在 GPU 上由低通系数生成 QMF 高通系数。
 * @tparam T FP32、FP16、INT32、INT16或INT8输入；输出固定为INT64。
 * @param hk 调用者持有的 device 一维低通系数 `[N]`。
 * @param out 调用者预分配的 device 输出 `[N]`。
 * @throws std::invalid_argument 输入输出 size 不同。
 * @details 空输入直接返回；默认 stream 异步 custom template kernel，无 workspace、
 * dlfft/dlrand 或隐式传输，调用者读取输出前负责同步。
 * @note 对齐固定版cuSignal的实际索引生成行为；输入值本身不参与计算。
 */
template<class T>
void qmf_device(const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out);

template<class T>
void qmf_device(
    const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out,
    const std::vector<int>& shape);
```

四个接口分成两个维度语义：

- 一维入口以 `hk.size()` 为 $N$，输出长度同为 $N$；
- shape-aware 入口验证扁平数据长度等于 $\prod_d shape[d]$，但模拟 Python `len(hk)`，只输出 `shape[0]` 个元素。

`hk` 的名称和 Doxygen 摘要保留“低通系数”说法，但详情准确说明它只贡献长度/shape。这里的“生成高通系数”是兼容 API 名称，不表示当前 C++ 实现执行了数学 QMF 的系数反转。

### 4. CPU 实现

#### 4.1 一维 CPU 核心

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`qmf_typed_cpu(const std::vector<T>&)`，285--295 行。

```cpp
template <typename T>
std::vector<std::int64_t> qmf_typed_cpu(const std::vector<T>& hk)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported qmf input dtype");
    std::vector<std::int64_t> out(hk.size());
    for (std::size_t i = 0; i < hk.size(); ++i) {
        const std::int64_t sign = (i & 1U) ? -1 : 1;
        out[i] = static_cast<std::int64_t>(hk.size() - (i + 1)) * sign;
    }
    return out;
}
```

执行顺序：先以 `static_assert` 在编译期限制 `T`；再按 `hk.size()` 分配 INT64 输出；循环计算交替符号和倒序索引值；最后返回独立 host vector。CPU 不读取 `hk[i]`，没有卷积、FFT、workspace、stream 或数据传输。

#### 4.2 shape-aware CPU 核心

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`qmf_typed_cpu(const std::vector<T>&, const std::vector<int>&)`，297--317 行。

```cpp
template <typename T>
std::vector<std::int64_t> qmf_typed_cpu(
    const std::vector<T>& hk, const std::vector<int>& shape)
{
    if (shape.empty()) throw std::invalid_argument("qmf shape must not be empty");
    std::size_t input_size = 1;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument("qmf shape");
        if (extent != 0 && input_size > hk.size() / static_cast<std::size_t>(extent))
            throw std::invalid_argument("qmf data/shape mismatch");
        input_size *= static_cast<std::size_t>(extent);
    }
    if (input_size != hk.size()) throw std::invalid_argument("qmf data/shape mismatch");
    const std::size_t count = static_cast<std::size_t>(shape.front());
    std::vector<std::int64_t> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::int64_t sign = (i & 1U) ? -1 : 1;
        out[i] = static_cast<std::int64_t>(count - (i + 1)) * sign;
    }
    return out;
}
```

前半段验证 shape 非空、维度非负、逐步乘积不超过 `hk.size()`，最终乘积必须等于扁平数据长度。后半段令 $N=shape.front()$，沿首维生成索引序列；其余维度只参与验证。

零元素 shape 有顺序边界：`shape={0,3}` 与空数据可通过；`shape={2,0}` 与空数据会在首个 `2` 处先判定超过 `hk.size()`。零维出现位置会影响接受性，是待补测风险。

#### 4.3 CPU 类型分发与显式实例化

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`INSTANTIATE_CPU` 的 qmf 项，319、338--346 行。

```cpp
#define INSTANTIATE_CPU(T) \
    template std::vector<std::int64_t> qmf_typed_cpu(const std::vector<T>&); \
    template std::vector<std::int64_t> qmf_typed_cpu( \
        const std::vector<T>&, const std::vector<int>&)
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(__half);
INSTANTIATE_CPU(std::int32_t);
INSTANTIATE_CPU(std::int16_t);
INSTANTIATE_CPU(std::int8_t);
#undef INSTANTIATE_CPU
```

原文件同一宏还包含其他 wavelet 算子；这里只摘录 qmf 相关宏项，没有省略 qmf 代码。五次展开生成 FP32、FP16、INT32、INT16、INT8 的两个 CPU 函数实体；输出始终为 `std::vector<std::int64_t>`。

### 5. GPU 实现

#### 5.1 一维 GPU host wrapper

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，一维 `qmf_device`，175--183 行。

```cpp
template <typename T>
void qmf_device(const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported qmf input dtype");
    if (hk.size() != out.size()) throw std::invalid_argument("qmf_device size mismatch");
    if (hk.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::qmf_kernel, hk.size(), out.data(), hk.size());
}
```

wrapper 只做类型门禁、长度门禁、空输入短路和 kernel 启动。传给 kernel 的只有 `out.data()` 与 `hk.size()`；没有 `hk.data()`，所以 device 端不读取输入系数。

#### 5.2 shape-aware GPU host wrapper

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，shape-aware `qmf_device`，185--205 行。

```cpp
template <typename T>
void qmf_device(
    const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out,
    const std::vector<int>& shape)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported qmf input dtype");
    if (shape.empty()) throw std::invalid_argument("qmf shape must not be empty");
    std::size_t input_size = 1;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument("qmf shape");
        if (extent != 0 && input_size > hk.size() / static_cast<std::size_t>(extent))
            throw std::invalid_argument("qmf data/shape mismatch");
        input_size *= static_cast<std::size_t>(extent);
    }
    if (input_size != hk.size()
        || out.size() != static_cast<std::size_t>(shape.front()))
        throw std::invalid_argument("qmf data/output shape mismatch");
    if (out.empty()) return;
    cuda_utils::launch_1d_kernel(
        wavelets_detail::qmf_kernel, out.size(), out.data(), out.size());
}
```

它与 CPU shape 验证同构，并额外要求 `out.size()==shape[0]`。合法空输出短路；非空时以输出长度作为 kernel 元素数和 `count`。

#### 5.3 GPU kernel

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh`，`wavelets_detail::qmf_kernel`，33--41 行。

```cpp
__global__ void qmf_kernel(std::int64_t* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::int64_t sign = (index & 1U) ? -1 : 1;
    output[index] = static_cast<std::int64_t>(count - (index + 1)) * sign;
}
```

每个线程计算一个全局一维索引。最后一个 block 的多余线程由边界分支退出；有效线程各写一个连续 INT64 元素，没有共享内存、屏障、原子操作或线程间依赖。

#### 5.4 grid、block 与启动 helper

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h`，`div_up`、`make_1d_launch_config`、`launch_1d_kernel_with_config`、`launch_1d_kernel`，10--56 行。

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

对 $N>0$，$block.x=256$，$grid.x=\lceil N/256\rceil=(N+255)/256$（整数除法）。启动使用 0 字节动态共享内存和默认 stream。`CUDA_KERNEL_CHECK()` 检查启动错误，但不等待 kernel 完成。

#### 5.5 GPU 类型实例化

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cu`，`INSTANTIATE_WAVELET_GPU` 的 qmf 项，207、213--223 行。

```cpp
#define INSTANTIATE_WAVELET_GPU(T) \
    template void qmf_device(const DeviceArray<T>&, DeviceArray<std::int64_t>&); \
    template void qmf_device( \
        const DeviceArray<T>&, DeviceArray<std::int64_t>&, const std::vector<int>&)

INSTANTIATE_WAVELET_GPU(float);
INSTANTIATE_WAVELET_GPU(__half);
INSTANTIATE_WAVELET_GPU(std::int32_t);
INSTANTIATE_WAVELET_GPU(std::int16_t);
INSTANTIATE_WAVELET_GPU(std::int8_t);

#undef INSTANTIATE_WAVELET_GPU
```

五种输入类型均生成两个 host wrapper。kernel 没有模板参数，因为它只接收固定 `int64_t*` 和长度；模板只存在于 `DeviceArray<T>` API 层。

### 6. 主要数据结构与内存布局

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/device_array.h`，`DeviceArray<T>`，12--20、93--96 行。

```cpp
/**
 * @brief Minimal 1D contiguous device-resident array.
 *
 * DeviceArray is intentionally move-only. It owns CUDA device memory and never
 * performs implicit host/device transfers. Use from_host() and to_host()
 * explicitly at API boundaries.
 */
template <typename T>
class DeviceArray {
public:
```

证据：同一 SHA 与文件，`DeviceArray<T>` 的 qmf 直接调用成员，93--96 行。

```cpp
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
```

两个原文片段只摘录 qmf 直接依赖的接口；中间省去的通用所有权实现不属于 qmf 核心。输入与输出都是一维连续 device allocation；输入元素大小取决于 `T`，输出固定每元素 8 字节。wrapper 不隐式 H2D/D2H，调用者通过 `from_host()` / `to_host()` 明确搬运。

### 7. 数学、Python、CPU 与 GPU 四方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| 数学 QMF：$g[i]=(-1)^i hk[N-1-i]$ | 接口/docstring 意图如此，但 kernel 未落实 | 未落实 | 未落实 | 三个可执行实现均不读取 `hk` 值 |
| 固定版实际公式：$y[i]=(-1)^i(N-1-i)$ | `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:23--24` | `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp:291--292,313--314` | `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/wavelets/wavelets_kernels.cuh:39--40` | C++ 有意复现 Python 23.08 实际行为 |
| 长度 $N$ | `len(hk)` | 一维 `hk.size()`；高维 `shape.front()` | wrapper 传 `hk.size()` 或 `out.size()` | 高维 C++ 额外验证 shape/product |
| 奇偶符号 | `(i & 1) ? -1 : 1` | `(i & 1U) ? -1 : 1` | `(index & 1U) ? -1 : 1` | 数学等价 |
| 输出 dtype | `int64 output` | `std::vector<std::int64_t>` | `DeviceArray<std::int64_t>` | 固定 INT64 |
| 空输入 | Python 未显式检查 | 空 vector | wrapper 返回 | GPU 避免零 grid |

结论：阶段一数学 QMF 与当前可执行代码仍是 `未对应`；C++ 与 Python 实际算法是 `等价实现`。C++ 的类型门禁、shape 契约、预分配输出、显式所有权和启动检查属于工程扩展。

### 8. CPU/GPU 职责划分

| 层次 | 职责 | 不负责的内容 |
| --- | --- | --- |
| API 声明 | 公开一维/高维 CPU、GPU 模板接口 | 不计算 |
| CPU/reference | host 验证 shape、分配、串行生成 | 不创建 stream、不搬运 device 数据 |
| GPU host wrapper | 类型/shape/size 门禁、短路、启动 | 不读取 hk、不做核心逐元素计算 |
| GPU kernel | 每线程生成一个输出 | 不加载输入、分配或通信 |
| 启动 helper | 计算 grid/block、启动、检查 launch | 不验证 qmf 契约 |
| 测试/benchmark | 构造、搬运、同步、比较、留证 | 不是核心实现 |

### 9. dtype、精度、边界与错误处理

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/src/cuda_utils/simple_signal_typed.h`，`is_simple_signal_input_v`，49--50 行。

```cpp
template <typename T>
inline constexpr bool is_simple_signal_input_v = SimpleSignalTypePolicy<T>::supported;
```

正式输入实体只有 FP32、FP16、INT32、INT16、INT8。同 shape 的五类输入输出逐位相同；输出固定 INT64 exact。核心没有浮点舍入。理论上 `count-1>INT64_MAX` 时转换超范围，但现实中会先受内存限制，当前无单独门禁。

| 条件 | CPU 一维 | CPU shape-aware | GPU 一维 | GPU shape-aware |
| --- | --- | --- | --- | --- |
| 空输入 | 空 vector | 取决于 shape product | 直接返回 | 验证后返回 |
| shape 空/负 extent | 不适用 | `invalid_argument` | 不适用 | `invalid_argument` |
| product 错误 | 不适用 | `invalid_argument` | 不适用 | `invalid_argument` |
| 输出长度错误 | 自行分配 | 自行分配 | `invalid_argument` | `invalid_argument` |
| host 分配失败 | `bad_alloc` | `bad_alloc` | 调用者分配阶段 | 调用者分配阶段 |
| kernel launch 错误 | 不适用 | 不适用 | `CUDA_KERNEL_CHECK` | `CUDA_KERNEL_CHECK` |

### 10. 复杂度、同步和资源

- CPU 时间 $O(N)$，输出空间 $8N=O(N)$；shape 验证另需 $O(rank)$。
- GPU 总工作 $O(N)$、每线程 $O(1)$，grid $\lceil N/256\rceil$、block 256。
- 无 workspace、FFT、科学库、共享内存和输入 device 读取。
- 默认 stream 异步提交；调用者在 `to_host()` 或 `synchronize_stream()` 同步。

### 11. ZQ500 平台限制与验证边界

实现使用 `__global__`、`dim3`、CUDA launch、`DeviceArray` 和项目 CUDA 错误封装。在 ZQ500 上必须由 `gpu_02` 内 SDK/$DLICC_PATH、`curt` 和平台兼容头解释，不能以本地或通用 NVIDIA 环境替代。平台依赖集中在 kernel 语法、INT64 device 运算、分配/复制封装、默认 stream 和错误检查。

本次为 Learning 只读阶段，没有构建、运行或连接 ZQ500。因此 V1 是“提交快照源码已核验”，不是“本次会话已在 ZQ500 复验”。

### 12. 测试与风险

证据：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu`，`run_type<T>` qmf 段 340--371、380--381 行，`main` 417--421 行。

E3 smoke 覆盖六抽头一般输入、三抽头奇数长度、shape `{2,3,4}` 的高维入口、错误输出长度，以及五种输入 dtype。它同步后记录 INT64 evidence，能证明 CPU/GPU 路径一致、模板可链接和基本门禁生效。

证据：同一 SHA，`ZKX/test_all/operators/cases/wavelets/qmf/operator_probe.cu`，`run<T>` / `main`，22--25 行。正式 probe 覆盖五 dtype 和 100、1024、10000 三规模，记录 digest、误差、计时和资源轨迹。

仍有风险：两套测试都以 `qmf_typed_cpu` 为 oracle，不能排除 CPU/GPU 共享错误；直接 smoke 未独立重算公式；零元素 shape 顺序边界、INT64 理论极限及本 SHA 的当前 ZQ500 运行状态未在本会话验证。

### 13. 阅读顺序

1. `wavelets_typed.h:181--218`：四个接口。
2. `wavelets_typed.cpp:285--317`：CPU 两个循环。
3. `wavelets_typed.cu:175--205`：wrapper，重点看没有 `hk.data()`。
4. `wavelets_kernels.cuh:33--41`：每线程一个输出。
5. `kernel_launch.h:10--56`：grid/block/default stream。
6. CPU/GPU 显式实例化：五输入类型、固定 INT64。
7. E3 smoke 与正式 probe：判断证据边界。

### 14. V1 自检问题与参考答案

#### 问题 1：版本身份是什么？

**参考答案 1：**SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`，整体及相关文件均 clean，可形成正式 V1。

#### 问题 2：CPU 核心公式及证据是什么？

**参考答案 2：**$out[i]=(-1)^i(N-1-i)$。证据是该 SHA 的 `ZKX/cusignal_cpp/src/wavelets/wavelets_typed.cpp`，`qmf_typed_cpu`，291--292 与 313--314 行。

#### 问题 3：为什么不是数学 QMF，却与 Python 等价？

**参考答案 3：**数学 QMF 要读取 $hk[N-1-i]$；Python 固定版和 C++ 都只使用长度。CPU 不引用 `hk[i]`，GPU 不传 `hk.data()`，所以 C++ 与 Python 实际行为等价，与通用原理 `未对应`。

#### 问题 4：CPU/GPU 职责如何分开？

**参考答案 4：**CPU 在 host 分配并串行生成；GPU wrapper 校验和启动；kernel 逐线程计算。证据是同一 SHA 的 `.cpp:285--317`、`.cu:175--205`、`.cuh:33--41`。

#### 问题 5：grid、block 和线程范围是什么？

**参考答案 5：**`block.x=256`，`grid.x=ceil(N/256)`。线程由 `blockIdx.x*blockDim.x+threadIdx.x` 得 index，越界返回，否则只写 `output[index]`。

#### 问题 6：支持哪些模板类型？

**参考答案 6：**FP32、FP16、INT32、INT16、INT8；输出统一 INT64。CPU/GPU 的显式实例化分别见 `.cpp:338--345` 与 `.cu:213--221`。

#### 问题 7：同步在哪里？

**参考答案 7：**wrapper 不同步；默认 stream 异步启动。调用者在 `to_host()` 或测试的 `synchronize_stream()` 同步，`CUDA_KERNEL_CHECK` 只检查 launch。

#### 问题 8：高维如何模拟 Python `len`？

**参考答案 8：**先验证 `product(shape)==hk.size()`，再只取 `shape.front()` 作为输出长度。CPU 证据 `.cpp:301--315`，GPU 证据 `.cu:191--204`。

#### 问题 9：测试能和不能证明什么？

**参考答案 9：**能证明五类型、CPU/GPU 一致、基本 shape/size 契约；不能证明数学 QMF，因为 CPU/GPU 使用同一个非独立公式。

#### 问题 10：ZQ500 验证边界是什么？

**参考答案 10：**应在 `gpu_02` 初始化 SDK 后验证平台编译器、运行时、INT64 kernel、默认 stream 和错误封装。本次未运行，不能声称当前 SHA 已由本会话实测。

## 版本差异与原理不变量

当前只有 V1。

| 比较项 | V1（fdd55ac） | 后续版本不变量或判断条件 |
| --- | --- | --- |
| 可执行公式 | $(-1)^i(N-1-i)$ | 若仍兼容固定版，公式与 INT64 不得暗改 |
| CPU | 串行 $O(N)$ | 优化不能改变逐元素精确结果 |
| GPU | 256 threads/block | 并行可变，索引/边界必须保持 |
| 输入值 | 不读取 | 改为真正 QMF 属于算法变化，须重映射阶段一 |
| shape | 高维取首维并验证 product | 首维退化是兼容契约 |
| 精度 | INT64 exact | 不使用浮点容差 |
