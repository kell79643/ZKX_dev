# `cusignal_cpp_unit_impulse` 复现逻辑

## 版本索引

| 版本 | 完整 Git SHA | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前正式实现](#v1当前正式实现55af4512) | `55af4512d44424a5309e3e9fe1faab484005a5d3` | 当前版本 | 固定 FP64 输出；CPU/GPU 均按首维生成一维冲激；GPU 直接写 IEEE-754 位模式 |

当前实现是 V1。后续若 `unit_impulse` 已提交优化，应在本文件追加 V2、V3，不覆盖本节。

## V1：当前正式实现（55af4512）

### 1. 版本与工作区状态

- 完整 SHA：`55af4512d44424a5309e3e9fe1faab484005a5d3`；
- 提交时间：`2026-08-14T20:46:51+08:00`；
- 分支：`final-prep/benchmark-evidence-v1`；
- 提交说明：`docs(errors): 记录Task2代表异常与开发者交接`；
- 检查时 `git status --short` 无输出；
- 下列 `unit_impulse` 相关源码和测试相对该 SHA 均为 clean，`RELATED_DIRTY=false`；
- 本阶段只读 `ZKX/cusignal_cpp`，没有修改 C++/CUDA 源码，也没有重新构建或运行 ZQ500 测试。

### 2. 文件角色与可追溯定位

| 完整 Git SHA | 共享根目录 `ZKX_dev/` 下的路径与行号 | 符号 | 角色 |
| --- | --- | --- | --- |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.h:315-361` | `unit_impulse_typed_cpu`、`unit_impulse_device` | 公共声明与行为契约 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:445-520` | CPU 实现、首维 helper、GPU host wrapper | CPU 核心与 GPU 调度入口 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:547-561` | `INSTANTIATE_CPU` | 五种模板契约显式实例化 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cu:207-214` | `unit_impulse_fp64_storage_device` | 非模板 CUDA host launch helper |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:352-362` | `unit_impulse_fp64_storage_kernel` | 当前正式 GPU device 核心计算 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:342-350` | `unit_impulse_kernel<T>` | 已定义的通用一维 kernel；当前 wrapper 未引用 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:364-382` | `unit_impulse_2d_kernel<T>` | 已定义的二维 kernel；当前 wrapper 未引用 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h:10-56` | `launch_1d_kernel` | block/grid 计算、默认 stream 和启动错误检查 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/src/cuda_utils/device_array.h:12-121` | `DeviceArray<T>` | 一维连续 device 内存所有权与显式 H2D/D2H |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:39-60` | independent bit golden | 独立 FP64 位模式判定 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:269-339` | `unit_impulse` E3 smoke | CPU/GPU、边界、首维退化和故障注入检查 |
| `55af4512d44424a5309e3e9fe1faab484005a5d3` | `ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:390-424` | `main` | 五种类型运行与总汇总 |

### 3. 对外接口与语义

#### 3.1 CPU 三个 overload

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.h:324-338`：

```cpp
template<class T>
std::vector<double> unit_impulse_typed_cpu(int shape, int idx = 0);

template<class T>
std::vector<double> unit_impulse_typed_cpu(
    const std::vector<int>& shape, const std::vector<int>& idx = {});

template<class T>
std::vector<double> unit_impulse_typed_cpu(
    const std::vector<int>& shape, const std::string& idx);
```

三个入口分别接受：

1. 一维长度与整数索引；
2. Python tuple/list 对应的 `std::vector<int>` shape/idx；
3. shape 与字符串索引，目前只允许 `"mid"`。

所有 CPU overload 都返回 `std::vector<double>`。模板参数 `T` 只是 TASK_F 五类型契约标签，不是输出 dtype，也没有业务输入数组。

#### 3.2 GPU 三个 overload

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.h:350-361`：

```cpp
template<class T>
void unit_impulse_device(DeviceArray<double>& out, int idx = 0);

template<class T>
void unit_impulse_device(
    DeviceArray<double>& out, const std::vector<int>& shape,
    const std::vector<int>& idx = {});

template<class T>
void unit_impulse_device(
    DeviceArray<double>& out, const std::vector<int>& shape,
    const std::string& idx);
```

GPU API 不返回新数组，而是原位写入调用者预分配的 `DeviceArray<double> out`。shape overload 只验证 `out.size()` 等于 `shape[0]`，不会分配 $\prod_r N_r$ 个元素。

### 4. CPU 调用链与逐步算法

CPU 调用链为：

```text
unit_impulse_typed_cpu<T>(shape, idx)
  ├─ 标量入口：直接创建 vector<double>
  ├─ vector shape/idx：取 shape.front() 与 idx.front()
  └─ vector shape/"mid"：取 shape.front()/2
      ↓
unit_impulse_typed_cpu<T>(int shape, int idx)
      ↓
创建 max(shape,0) 个 double 0.0
      ↓
若 0 <= idx < shape，则 y[idx] = 1.0
      ↓
返回一维 std::vector<double>
```

#### 4.1 一维 CPU 核心

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:444-453`：

```cpp
template <typename T>
std::vector<double> unit_impulse_typed_cpu(int shape, int idx)
{
    static_assert(detail::is_simple_signal_input_v<T>,
                  "unsupported unit_impulse output dtype");
    std::vector<double> y(
        shape > 0 ? static_cast<std::size_t>(shape) : 0, 0.0);
    if (idx >= 0 && idx < shape) {
        y[static_cast<std::size_t>(idx)] = 1.0;
    }
    return y;
}
```

数学上逐元素等价于：

$$
y[n]=
\begin{cases}
1,&0\le n<N\ \text{且}\ n=k,\\
0,&0\le n<N\ \text{且}\ n\ne k.
\end{cases}
$$

若 `shape<=0`，标量入口返回空 vector。若 `idx<0` 或 `idx>=shape`，条件不成立，返回同长度全零 vector。

#### 4.2 shape/idx helper

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:455-473`：

- `unit_impulse_first_extent(shape)`：空 shape 抛 `std::invalid_argument`；首维为负也抛异常；否则返回 `shape.front()`；
- `unit_impulse_first_index(idx)`：空 idx 返回默认 `0`，否则返回 `idx.front()`；
- `unit_impulse_named_index(shape, idx)`：字符串不是 `"mid"` 就抛异常；否则返回 `shape.front()/2`。

因此 `shape={3,4,5}, idx="mid"` 被归约成一维调用 `shape=3, idx=1`。其余 shape 维度不会参与输出长度或索引计算。

#### 4.3 vector overload 转发

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:475-489`：两个 overload 只调用 helper 提取首维/首索引，再转发到一维 CPU 核心，没有 reshape 或多维坐标展开。

### 5. GPU 调用链与 host wrapper

正式 GPU 调用链为：

```text
调用者预分配 DeviceArray<double> out(N)
  ↓
unit_impulse_device<T>(out, idx)
  ↓ static_assert 类型与 FP64 存储前提
unit_impulse_fp64_storage_device(out.data(), out.size(), idx)
  ↓ count==0 时直接返回
launch_1d_kernel(..., count, uint64_output, count, idx)
  ↓ block=256, grid=ceil(count/256), default stream
unit_impulse_fp64_storage_kernel
  ↓ 每线程处理一个线性 index
直接写 FP64 0.0/1.0 的 uint64 位模式
```

#### 5.1 GPU 公共 wrapper

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:491-498`：

```cpp
template <typename T>
void unit_impulse_device(DeviceArray<double>& out, int idx)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    unit_impulse_fp64_storage_device(out.data(), out.size(), idx);
}
```

三个 `static_assert` 都发生在编译期：

- `T` 必须属于支持的五类标签；
- `double` 与 `uint64_t` 都必须是 64 bit；
- `double` 必须符合 IEC 559/IEEE-754，才能安全使用固定位模式。

#### 5.2 shape overload 的验证与转发

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:500-520`：

1. 取 `extent=shape.front()`；
2. 要求 `out.size()==extent`，否则抛 `std::invalid_argument("unit_impulse output/shape mismatch")`；
3. vector idx 只取 `idx.front()`；
4. `"mid"` 只计算 `shape.front()/2`；
5. 转发到一维 `unit_impulse_device<T>(out, int idx)`。

这两个 overload 接受多维形状语法，但有意复现固定版 Python 的首维一维退化，而不实现 SciPy docstring 的 N-D 输出。

### 6. CUDA launch helper

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cu:207-214`：

```cpp
void unit_impulse_fp64_storage_device(
    void* output_storage, std::size_t count, int idx)
{
    if (count == 0) return;
    cuda_utils::launch_1d_kernel(
        waveforms_detail::unit_impulse_fp64_storage_kernel, count,
        static_cast<std::uint64_t*>(output_storage), count, idx);
}
```

- 空输出不启动 kernel；
- `void*` 被解释成 `uint64_t*`，用于写 `double` 的存储位，而不是把数值转换成整数；
- `launch_1d_kernel` 的第二个 `count` 是调度元素数；参数列表里的另一个 `count` 传给 device kernel 做边界判断。

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h:21-55` 规定：

$$
\text{block.x}=256,
\qquad
\text{grid.x}=\left\lceil\frac{N}{256}\right\rceil.
$$

启动使用 `stream=nullptr`，即默认 stream；动态共享内存为 `0`。`CUDA_KERNEL_CHECK()` 检查启动错误，但 helper 本身不做 stream synchronize。

### 7. 当前正式 GPU kernel

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:352-362`：

```cpp
__global__ void unit_impulse_fp64_storage_kernel(
    std::uint64_t* output, std::size_t count, int impulse_index)
{
    constexpr std::uint64_t fp64_one_bits =
        UINT64_C(0x3ff0000000000000);
    const std::size_t index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = static_cast<int>(index) == impulse_index
            ? fp64_one_bits
            : UINT64_C(0);
    }
}
```

线程职责：全局线性索引 `index` 对应一个输出元素。多余线程由 `index<count` 保护。合法冲激位置写 `0x3ff0000000000000`，它是 IEEE-754 `double 1.0` 的确定位模式；其他位置写全零位，即 `double 0.0`。

该 kernel 不读取 `T`、不进行 FP32/FP64 算术、不使用临时 workspace，也没有共享内存或原子操作。所有线程互不依赖，因此没有 kernel 内同步。

### 8. 已定义但不在当前调用链的 kernel

#### 8.1 通用一维 kernel

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:342-350` 的 `unit_impulse_kernel<T>` 会按 `T` 写数值 `0/1`。在当前 `src/waveforms` 中没有 wrapper 引用它，因此不能把它称为当前正式核心实现。

#### 8.2 二维 kernel

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:364-382` 的 `unit_impulse_2d_kernel<T>` 计算：

$$
N=\text{rows}\times\text{columns},\quad
r=\left\lfloor\frac{i}{\text{columns}}\right\rfloor,\quad
c=i\bmod\text{columns}.
$$

它在 `(r,c)==(impulse_row,impulse_column)` 时写 `1`。但当前 `src/waveforms` 只有定义，没有 host wrapper、公开声明或测试调用引用它，所以它是未接入代码，不能用来证明当前 API 支持真正二维输出。

### 9. template、类型分发与显式实例化

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/cuda_utils/simple_signal_typed.h:17-50` 通过 `is_simple_signal_input_v<T>` 限定支持类型。

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:547-561` 为以下五种 `T` 显式实例化 CPU 与 GPU wrapper：

- `float`（FP32）；
- `__half`（FP16）；
- `std::int32_t`；
- `std::int16_t`；
- `std::int8_t`。

对 `unit_impulse` 而言，这五种 `T` 不改变计算和输出：CPU 始终返回 `vector<double>`，GPU 始终写 `DeviceArray<double>`。`double` 是结果存储类型，不是第六种业务输入标签。

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cu:216-230` 的 `I(T)` 宏没有实例化 `unit_impulse`，因为这里的 CUDA launch helper 本身是非模板函数；模板公共 wrapper 已在 `.cpp` 中实例化。

### 10. 内存布局、数据搬运、同步与错误处理

#### 10.1 内存布局

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/cuda_utils/device_array.h:12-27` 定义 `DeviceArray<T>` 为一维连续 device 数组。`unit_impulse` 的正式输出是连续 `N` 个 `double`，没有 stride、shape 元数据或多维布局。

#### 10.2 分配和数据搬运

- `unit_impulse_device` 不分配输出；调用者必须先构造 `DeviceArray<double> out(N)`；
- wrapper 内没有 H2D，因为该算子没有业务输入数组；
- wrapper 内没有 D2H；V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3` 的测试调用 `out.to_host()` 时，才由 `ZKX/cusignal_cpp/src/cuda_utils/device_array.h:66-73` 显式 D2H 并同步；
- 没有临时 buffer、FFT 或平台数学库调用。

#### 10.3 同步

kernel launch 是默认 stream 上的异步提交。`launch_1d_kernel` 只做启动错误检查，不等待计算完成。需要 host 读取时，`DeviceArray::to_host()` 的复制与 `synchronize_stream()` 建立完成边界。

#### 10.4 边界与异常

| 情况 | CPU | GPU wrapper/kernel |
| --- | --- | --- |
| 标量 `shape<=0` | 返回空 vector | 输出大小由调用者预分配；`count==0` 不启动 kernel |
| vector shape 为空 | 抛 `invalid_argument` | 抛 `invalid_argument` |
| `shape.front()<0` | 抛 `invalid_argument` | 抛 `invalid_argument` |
| `idx` vector 为空 | 默认索引 0 | 默认索引 0 |
| `idx<0` 或 `idx>=N` | 同长度全零 | 所有线程写零 |
| 字符串不是 `"mid"` | 抛 `invalid_argument` | 抛 `invalid_argument` |
| `out.size()!=shape[0]` | 不适用 | 抛 output/shape mismatch |
| `shape[1:]` 或 `idx[1:]` | 忽略 | 忽略 |

### 11. 数学、Python、CPU 与 GPU 映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $y[n]=1$ 当且仅当 $n=k$ | `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:577-582`；基准 `ZKX/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:526-530` | V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:448-451` | V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:355-360` | 数学完全一致；GPU 写位模式而非做 double 算术 |
| 长度 $N$ | Python 只传 `shape[0]` 与 `size=shape[0]` | V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:456-460` | V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:505-519` | C++ 有意保存首维一维退化 |
| 默认 $k=0$ | `idx=None` 生成首项 0 | V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:463-466` | 使用同一 SHA、同一路径的 `unit_impulse_first_index` helper | 语义一致 |
| `"mid"` | 首项为 `shape[0]//2` | V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:468-471` | 使用同一 SHA、同一路径的 `unit_impulse_named_index` helper | 只对首维居中，不是真 N-D 中心 |
| dtype | Python 形参未转发，kernel 固定 float64 | 输出 `vector<double>` | 输出 `DeviceArray<double>` | C++ 复现固定版实际行为，不复现 SciPy/docstring dtype 契约 |
| 卷积单位元、DTFT 平坦 | 不在生成函数内部执行 | 不执行 | 不执行 | 是结果性质，不是构造步骤 |

### 12. 与 cuSignal Python 的相同、替换和有意差异

#### 12.1 相同

- 只使用首维长度和首个索引；
- `"mid"` 按首维整除 2；
- 越界 idx 产生全零；
- 输出固定 FP64，模板/形参 dtype 不改变结果；
- 核心都是逐位置判断是否等于目标索引。

#### 12.2 等价替换

- Python/CuPy `ElementwiseKernel` 被 C++ host wrapper + 显式 CUDA kernel 替代；
- Python 动态创建返回数组被“调用者预分配 `DeviceArray<double>`”替代；
- GPU 用 FP64 0/1 位模式写入，数学上等价于赋值 `0.0/1.0`，但避免 device double 算术与转换。

#### 12.3 有意差异

- C++ 对空 vector shape、负首维、非法字符串和 output/shape mismatch 给出明确异常；Python 固定版没有相同的显式检查；
- C++ 公开增加 vector shape/idx 和字符串 overload，但这些入口只负责复现首维退化，不承诺 SciPy N-D 语义；
- GPU API 原位写预分配输出，不返回新对象。

### 13. 测试如何验证正确性

#### 13.1 独立 golden

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:39-60` 不调用被测实现，而是独立构造 `uint64_t` golden：合法索引写 `0x3ff0000000000000`，其余写零；再用 `memcpy` 观察 CPU/GPU `double` 的实际存储位。

#### 13.2 典型、边界和越界

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:269-302` 对长度 9 检查：

- `idx=4`：典型中点；
- `idx=0`：左边界；
- `idx=99`：越界全零。

每项比较 GPU、CPU 和独立位模式 golden，要求零容差、全元素检查。

#### 13.3 首维退化入口

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:303-311` 显式调用 `shape={3,4,5}, idx="mid"`，预期输出不是 60 元素三维数组，而是长度 3、索引 1 为 `1` 的一维数组；CPU/GPU 都与独立 golden 比较。

#### 13.4 故障注入

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:313-327` 把正确中点冲激的索引 4 清零，再把索引 3 置一；检测器必须报告恰好两个 mismatch，证明测试不是只检查“程序能运行”。

#### 13.5 五种类型与汇总

V1 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`，`ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:390-424` 对 FP32、FP16、INT32、INT16、INT8 分别调用 `run_type<T>`。`double` 被明确排除为业务输入标签。该测试文件同时覆盖四个算子，所以总汇总 `total=20` 不是 `unit_impulse` 单独的 20 个案例。

本阶段没有重新执行该测试；以上是对提交中测试源码的解释，不把历史文档状态冒充本次运行结果。

### 14. 复杂度与性能结构

令一维输出长度为 $N$：

- CPU 时间复杂度：初始化 $N$ 个零并至多写一个 `1`，为 $O(N)$；
- CPU 输出空间：$N$ 个 `double`，为 $8N$ bytes，即 $O(N)$；
- GPU 总工作量：每元素一次比较和一次 64-bit 写入，为 $O(N)$；
- GPU block 数：$\lceil N/256\rceil$；
- GPU 输出空间：预分配 $8N$ bytes；wrapper 额外空间为 $O(1)$；
- 性能主要受 kernel launch、输出分配和连续全数组写带宽影响，而不是算术吞吐；
- 直接位模式路径消除了历史上可能出现的 FP32 临时结果、D2H 拓宽和 H2D 回写。

### 15. 仍未覆盖的风险

1. 当前正式 API 不实现 SciPy/docstring 的真正 N-D 输出，vector shape 仅保留首维；
2. `unit_impulse_2d_kernel<T>` 没有 wrapper 和直接测试，可能成为误导性死代码；
3. kernel 将 `std::size_t index` 转成 `int` 再比较，超出 `int` 范围的超大数组没有在所读测试中覆盖；
4. vector shape 只验证首维，后续维度即使非法也被忽略；
5. 默认 stream 异步语义要求调用者在 host 读取前建立同步边界；
6. 本阶段没有重新运行 ZQ500 smoke，当前结论绑定源码提交和已有测试结构，不新增运行证据；
7. 若未来目标改为遵守 SciPy N-D/dtype 契约，当前 CPU/GPU API、内存模型、kernel 和测试都需要成组升级，不能只接入现有未使用的 2D kernel。

### 16. 建议阅读顺序

以下各项都绑定 V1 完整 SHA `55af4512d44424a5309e3e9fe1faab484005a5d3`：

1. `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.h:315-361`：先识别 CPU/GPU 的六个 overload 和固定 FP64 契约；
2. `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:445-489`：阅读 CPU 一维核心与首维 helper；
3. `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cpp:491-520`：阅读 GPU wrapper 的静态断言、shape 检查和转发；
4. `ZKX/cusignal_cpp/src/waveforms/waveforms_typed.cu:207-214`：观察空数组门禁和 launch 参数；
5. `ZKX/cusignal_cpp/src/cuda_utils/kernel_launch.h:21-55`：计算 block/grid，并确认默认 stream、无同步；
6. `ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:352-362`：阅读当前 FP64 位模式 kernel；
7. `ZKX/cusignal_cpp/src/waveforms/waveforms_kernels.cuh:342-350、364-382`：最后辨认两个未接入 kernel，避免误当正式路径；
8. `ZKX/cusignal_cpp/test/signal_processing/e3_simple_batch_type_smoke.cu:39-60、269-339、390-424`：按 golden → 三类索引 → 首维退化 → 故障注入 → 五类型汇总检查测试。

### 17. 自检问题与答案

1. **为什么模板参数 `T` 不决定输出 dtype？**
   
   因为 CPU 返回类型写死为 `std::vector<double>`，GPU 参数写死为 `DeviceArray<double>`；`T` 只经过 `static_assert` 和五类型显式实例化。

2. **真正的 GPU 核心是哪一个 kernel？**
   
   是 `unit_impulse_fp64_storage_kernel`。通用 `unit_impulse_kernel<T>` 和二维 `unit_impulse_2d_kernel<T>` 当前都没有 wrapper 引用。

3. **`shape={3,4,5}, idx="mid"` 得到什么？**
   
   CPU/GPU 都归约为 `shape=3, idx=1`，得到长度 3 的一维 FP64 冲激 `[0,1,0]`。

4. **为什么 GPU 写 `uint64_t` 而输出仍是 double？**
   
   `DeviceArray<double>` 的存储按 `uint64_t*` 观察并写入 IEEE-754 位模式：全零是 `0.0`，`0x3ff0000000000000` 是 `1.0`。

5. **一个线程负责什么？**
   
   一个线程计算一个线性 `index`，若未越界就向 `output[index]` 写一个 64-bit 结果。

6. **GPU wrapper 是否同步？**
   
   不同步。它只在默认 stream 启动 kernel并做启动错误检查；测试通过 `to_host()` 的 D2H 与 `synchronize_stream()` 等待完成。

7. **测试如何证明 exact？**
   
   它独立构造 FP64 位模式 golden，逐元素比较 CPU/GPU 的实际 double 存储位，容差为零，并用故障注入确认检测器能发现移位错误。

8. **C++ 复现的是 docstring 还是固定 Python 实现？**
   
   复现固定 Python 23.08 的实际行为：首维一维退化、dtype 忽略、FP64 输出；不复现 SciPy/docstring 的真正 N-D 与 dtype selector。

## 版本差异与原理不变量

当前只有 V1，尚无可比较的后续已提交版本。

| 比较项 | V1：`55af4512d44424a5309e3e9fe1faab484005a5d3` | 原理不变量 |
| --- | --- | --- |
| 数学结构 | 一维 one-hot/Kronecker delta | 合法目标位置为 1，其余为 0 |
| CPU | `vector<double>` 初始化后单点置一 | $y[n]=\delta[n-k]$ |
| GPU | 256 threads/block，直接写 FP64 位模式 | 每个输出位置独立判断 $n=k$ |
| shape | 只保留首维 | 有限输出域为 $0\le n<N$ |
| dtype | 五种 `T` 标签共享 FP64 输出 | 0 和 1 可被精确表示 |
| 边界 | 越界 idx 全零 | 目标不在有限域内时无非零样本 |
