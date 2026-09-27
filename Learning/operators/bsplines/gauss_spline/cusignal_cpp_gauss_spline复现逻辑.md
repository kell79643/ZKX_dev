# gauss_spline 算子学习 — cusignal_cpp 复现逻辑

> 算子名：`gauss_spline`
> 所属模块：`bsplines`
> 阶段：三（cusignal_cpp CPU/GPU 复现逻辑）

---

## 版本索引

| 版本 | 短 SHA | 完整 SHA | 状态 |
| --- | --- | --- | --- |
| V1 | `1ccd32e` | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` | 当前实现 |

---

## V1：原始学习版本（1ccd32e）

### 1. 版本信息

| 项目 | 值 |
| --- | --- |
| 完整提交 SHA | `1ccd32eaefe7c8cfb7b0195fe1b892878d3ed15c` |
| 提交时间 | 2026-07-27 22:12:16 +0800 |
| 分支 | `snapshot/all-current-20260716` |
| 相关文件 dirty 状态 | 全部干净（无未提交修改） |

### 2. 对外接口、声明文件、实现文件与行号

| 层级 | 相对路径（`cusignal_cpp/` 下） | 符号名 | 行号 | 说明 |
| --- | --- | --- | --- | --- |
| 公开接口声明 | `src/bsplines/bsplines_typed.h` | `gauss_spline_cpu<T>` | 163 | CPU reference，返回 `std::vector<T>` |
| 公开接口声明 | `src/bsplines/bsplines_typed.h` | `gauss_spline<T>` | 177 | 已废弃 CPU 兼容别名 |
| 公开接口声明 | `src/bsplines/bsplines_typed.h` | `gauss_spline_device<T>` | 193 | GPU 正式入口，写入 `DeviceArray<T>&` |
| 内联计算 | `src/bsplines/bsplines_typed.h` | `detail::gauss_spline_value<T>` | 100–108 | 单元素计算，CPU/GPU 共用逻辑 |
| 类型策略 | `src/bsplines/bsplines_typed.h` | `detail::BsplineTypePolicy<T>` | 25–43 | FP32/INT32/INT16/INT8 加载→float、存回→T |
| 类型策略特化 | `src/bsplines/bsplines_typed.h` | `detail::BsplineTypePolicy<__half>` | 45–59 | FP16 特化：`__half2float`/`__float2half_rn` |
| CPU 实现 | `src/bsplines/bsplines_typed.cpp` | `gauss_spline_cpu<T>` | 35–41 | n < 0 校验 + 委托 `evaluate_cpu` |
| CPU 实现 | `src/bsplines/bsplines_typed.cpp` | `evaluate_cpu<T, Evaluator>` | 9–18 | 通用 CPU 逐元素求值循环 |
| CPU 别名 | `src/bsplines/bsplines_typed.cpp` | `gauss_spline<T>` | 44–47 | 废弃，直接调 `gauss_spline_cpu` |
| GPU host wrapper | `src/bsplines/bsplines_typed.cu` | `gauss_spline_device<T>` | 35–50 | 校验 + host 预计算 + 启动 kernel |
| GPU device kernel | `src/bsplines/bsplines_kernels.cuh` | `bsplines_detail::gauss_spline_kernel<T>` | 30–43 | CUDA kernel，逐元素 `norm * expf(...)` |
| 显式实例化 (CPU) | `src/bsplines/bsplines_typed.cpp` | — | 69–73 | float, `__half`, int32, int16, int8 |
| 显式实例化 (GPU) | `src/bsplines/bsplines_typed.cu` | — | 67–71 | 同上五种类型 |
| 显式实例化 (声明) | `src/bsplines/bsplines_typed.h` | — | 239–243 (CPU), 257–261 (GPU) | extern template 声明 |
| 测试 | `test/signal_processing/e3_bsplines_type_smoke.cu` | `run_type<T>` | 95–206 | 五类型 smoke + n 边界 + 证据记录 |

### 3. CPU 调用链、主要数据结构与逐步算法

#### 3.1 CPU 调用链

```
cusignal::gauss_spline_cpu<T>(x, n)                // bsplines_typed.cpp:35
  ├─ if (n < 0) throw std::invalid_argument(...)   // bsplines_typed.cpp:37-38
  └─ evaluate_cpu(x, lambda)                        // bsplines_typed.cpp:40
       ├─ std::vector<T> y(x.size())               // bsplines_typed.cpp:13
       ├─ for i = 0 .. x.size()-1:                 // bsplines_typed.cpp:14
       │    y[i] = detail::gauss_spline_value(x[i], n)  // 通过 lambda
       └─ return y                                  // bsplines_typed.cpp:17
```

展开 `detail::gauss_spline_value<T>(value, n)`（bsplines_typed.h:100–108）：

```
1. signsq  = static_cast<float>(n + 1) / 12.0F           // σ² = (n+1)/12
2. r_signsq = 0.5F / signsq                               // 1/(2σ²)
3. pi      = 3.14159265358979323846F                      // π 常量
4. norm    = 1.0F / std::sqrt(2.0F * pi * signsq)        // 1/√(2πσ²)
5. x       = BsplineTypePolicy<T>::load(value)            // T → float
6. result  = norm * std::exp(-(x * x) * r_signsq)        // 归一化·exp(...)
7. return  = BsplineTypePolicy<T>::store(result)          // float → T
```

#### 3.2 CPU 主要数据结构

| 数据结构 | 说明 |
| --- | --- |
| `std::vector<T>` | 输入 `x` 和输出 `y` 的存储，连续 host 内存 |
| `BsplineTypePolicy<T>` | 编译期类型策略：`load` 将 T 转为 `float`，`store` 将 `float` 转回 T |
| lambda `[n](T value) { return detail::gauss_spline_value(value, n); }` | 逐元素求值仿函数，捕获 `n` |

#### 3.3 CPU 逐步算法

1. **参数校验**：若 `n < 0`，抛出 `std::invalid_argument`（与 cuSignal Python 不同，Python 不检查，负 n 导致 NaN）。
2. **分配输出**：`std::vector<T> y(x.size())`，零初始化。
3. **逐元素循环**：对每个 `x[i]`：
   a. 通过 `BsplineTypePolicy<T>::load` 将 `x[i]` 转为 `float`；
   b. 计算 `signsq = (n+1)/12.0F`、`r_signsq = 0.5F/signsq`、`norm = 1.0F/sqrt(2π·signsq)`；
   c. 计算 `result = norm * exp(-(x_float * x_float) * r_signsq)`；
   d. 通过 `BsplineTypePolicy<T>::store` 将 `result`（float）转回 `T`；
   e. 写入 `y[i]`。
4. **返回** `y`。

**注意**：步骤 b 中 `signsq`、`r_signsq`、`norm` 在每个元素的计算中重复求值，但编译器在循环不变量外提优化下可能将其提到循环外。当前代码未显式外提。

### 4. GPU 调用链、host wrapper、kernel 启动与 device 计算逻辑

#### 4.1 GPU 完整调用链

```
cusignal::gauss_spline_device<T>(x, y, n)                // bsplines_typed.cu:35
  ├─ require_compatible_buffers(x, y, "gauss_spline_device")  // bsplines_typed.cu:37
  │    └─ if (x.size() != y.size()) throw ...            // bsplines_typed.cu:17-18
  ├─ if (n < 0) throw std::invalid_argument(...)         // bsplines_typed.cu:38-39
  ├─ if (!x.empty()) {                                   // bsplines_typed.cu:41
  │    ├─ signsq  = static_cast<float>(n + 1) / 12.0F   // bsplines_typed.cu:42
  │    ├─ pi      = 3.14159265358979323846F              // bsplines_typed.cu:43
  │    ├─ r_signsq = 0.5F / signsq                      // bsplines_typed.cu:44
  │    ├─ norm    = 1.0F / sqrtf(2.0F * pi * signsq)   // bsplines_typed.cu:45
  │    └─ cuda_utils::launch_1d_kernel(                  // bsplines_typed.cu:46-48
  │         bsplines_detail::gauss_spline_kernel<T>,
  │         x.size(), x.data(), y.data(), x.size(), r_signsq, norm)
  │         └─ launch_1d_kernel_with_config(...)         // kernel_launch.h:50-55
  │              ├─ config = make_1d_launch_config(N, 256)  // kernel_launch.h:39
  │              │    ├─ block = dim3(256)               // kernel_launch.h:26
  │              │    └─ grid  = dim3(ceil(N/256))       // kernel_launch.h:27
  │              ├─ kernel<<<grid, block, 0, nullptr>>>(...)  // kernel_launch.h:40
  │              └─ CUDA_KERNEL_CHECK()                  // kernel_launch.h:41
  │    }
  └─ (隐式返回 void)

// Device kernel 展开如下：
bsplines_detail::gauss_spline_kernel<T>(x, y, count, r_signsq, norm)
  // bsplines_kernels.cuh:30-43
  ├─ index = blockIdx.x * blockDim.x + threadIdx.x      // :37
  ├─ if (index >= count) return                          // :38
  ├─ value = BsplineTypePolicy<T>::load(x[index])       // :40
  ├─ result = norm * expf(-(value * value) * r_signsq)  // :41-42
  └─ y[index] = BsplineTypePolicy<T>::store(result)     // :41-42
```

#### 4.2 Host wrapper 详解（bsplines_typed.cu:35–50）

| 步骤 | 代码 | 行号 | 说明 |
| --- | --- | --- | --- |
| 1 | `require_compatible_buffers(x, y, ...)` | 37 | 检查输入输出 `DeviceArray` 长度一致 |
| 2 | `if (n < 0) throw ...` | 38–39 | 拒绝负 n |
| 3 | `signsq = static_cast<float>(n + 1) / 12.0F` | 42 | host 端计算 $\sigma^2$，**float 精度** |
| 4 | `r_signsq = 0.5F / signsq` | 44 | host 端计算 $1/(2\sigma^2)$，**float 精度** |
| 5 | `norm = 1.0F / sqrtf(2.0F * pi * signsq)` | 45 | host 端预计算归一化常数 $1/\sqrt{2\pi\sigma^2}$，**float 精度** |
| 6 | `launch_1d_kernel(...)` | 46–48 | 将 `r_signsq` 和 `norm` 作为 kernel 参数传入 |

**关键设计：归一化常数 `norm` 在 host 端预计算**，每个线程直接乘 `norm`，不再重复计算 `sqrtf` 和除法。这是相比 cuSignal Python 的一个优化点。

#### 4.3 Kernel 启动配置

| 参数 | 值 | 来源 |
| --- | --- | --- |
| `blockDim.x` | 256 | `kernel_launch.h:54`（默认 `block_size = 256`） |
| `gridDim.x` | $\lceil N / 256 \rceil$ | `kernel_launch.h:27`（`div_up(elements, block_size)`） |
| `stream` | `nullptr`（默认流） | `kernel_launch.h:53` |
| shared memory | 0 字节 | `kernel_launch.h:40`（第三个参数 `0`） |

#### 4.4 Device kernel 详解（bsplines_kernels.cuh:30–43）

```cpp
template <typename T>
__global__ void gauss_spline_kernel(
    const T* x,                          // 输入数组，设备内存
    T* y,                                 // 输出数组，设备内存
    std::size_t count,                    // 元素总数
    float reciprocal_twice_variance,      // r_signsq = 1/(2σ²)
    float normalization)                  // norm = 1/√(2πσ²)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float value = detail::BsplineTypePolicy<T>::load(x[index]);
    y[index] = detail::BsplineTypePolicy<T>::store(
        normalization * expf(-(value * value) * reciprocal_twice_variance));
}
```

**逐线程逻辑：**

1. 计算全局索引 `index = blockIdx.x * blockDim.x + threadIdx.x`；
2. 越界检查：`index >= count` 则返回（处理 grid 超出元素数的尾部线程）；
3. 从设备内存读取 `x[index]`，通过 `BsplineTypePolicy<T>::load` 转为 `float`；
4. 计算 `normalization * expf(-(value * value) * reciprocal_twice_variance)`：
   - `value * value`：$x^2$
   - `* reciprocal_twice_variance`：$x^2 \cdot \frac{1}{2\sigma^2}$
   - `expf(...)`：$\exp\!\left(-\frac{x^2}{2\sigma^2}\right)$
   - `* normalization`：$\frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\!\left(-\frac{x^2}{2\sigma^2}\right)$
5. 通过 `BsplineTypePolicy<T>::store` 将 `float` 结果转回 `T`，写入 `y[index]`。

**每个线程负责的数据范围**：恰好 1 个元素（`x[index]` → `y[index]`），无线程间数据依赖。

### 5. Template、类型分发、显式实例化、内存布局及数据搬运

#### 5.1 Template 参数体系

| Template | 定义位置 | 约束 | 说明 |
| --- | --- | --- | --- |
| `T` | 全部函数 | `is_bspline_input_v<T>` = true | 输入/输出元素类型 |
| `ComputeT` | `BsplineTypePolicy<T>::ComputeT` | 固定为 `float` | 中间计算类型 |

**支持的五种类型**（`BsplineTypePolicy` 限定）：

| `T` | `load` 行为 | `store` 行为 | `ComputeT` |
| --- | --- | --- | --- |
| `float` | 恒等 | 恒等 | `float` |
| `__half` | `__half2float` | `__float2half_rn` | `float` |
| `std::int32_t` | `static_cast<float>` | `static_cast<int32_t>`（截断） | `float` |
| `std::int16_t` | `static_cast<float>` | `static_cast<int16_t>`（截断） | `float` |
| `std::int8_t` | `static_cast<float>` | `static_cast<int8_t>`（截断） | `float` |

**核心设计**：所有五种类型统一提升到 `float`（FP32）计算，结果再写回 `T`。这确保了：
- FP16 不直接调用超越函数（`expf`/`sqrtf` 不接受 `__half`）；
- 整数类型虽然数学结果是小数，但遵循项目批准的"同型输出"契约（整数截断后通常为 0）。

#### 5.2 显式实例化

**CPU 端**（bsplines_typed.cpp:61–75）：

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
```

**GPU 端**（bsplines_typed.cu:62–73）：

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
```

**头文件 extern template 声明**（bsplines_typed.h:239–261）：防止其他翻译单元隐式实例化。

#### 5.3 内存布局与数据搬运

| 阶段 | 数据位置 | 布局 | 搬运方式 |
| --- | --- | --- | --- |
| CPU 输入 `x` | host（`std::vector<T>`） | 连续 row-major，1D 展平 | 调用者提供 |
| CPU 输出 `y` | host（`std::vector<T>`） | 同上 | 函数内分配 |
| GPU 输入 `x` | device（`DeviceArray<T>`） | 连续 1D 设备内存 | 调用者通过 `DeviceArray<T>::from_host` 做 H2D |
| GPU 输出 `y` | device（`DeviceArray<T>`） | 同上 | 调用者预分配，kernel 写入 |
| Host→Device | `x.data()` → kernel 参数 | — | `DeviceArray::from_host` 内 `cudaMemcpy` + `cudaStreamSynchronize` |
| Device→Host | `y.data()` → host | — | `DeviceArray::to_host` 内 `cudaMemcpy` + `cudaStreamSynchronize` |
| Kernel 常量 | host stack → kernel 参数 | — | CUDA kernel launch 参数传递（自动 H2D） |

**关键**：`gauss_spline_device` 不执行隐式 H2D/D2H。调用者负责通过 `DeviceArray` 的 `from_host`/`to_host` 显式搬运。

### 6. Grid、Block、线程索引与每个线程负责的数据范围

| 项目 | 值 | 来源 |
| --- | --- | --- |
| `blockDim.x` | 256 | `kernel_launch.h:54` |
| `gridDim.x` | $\lceil N / 256 \rceil$ | `kernel_launch.h:27` |
| `blockDim.y`, `blockDim.z` | 1 | `dim3` 默认值 |
| `gridDim.y`, `gridDim.z` | 1 | `dim3` 默认值 |
| 全局线程索引 | `blockIdx.x * blockDim.x + threadIdx.x` | `bsplines_kernels.cuh:37` |
| 每线程负责元素数 | 1 | 逐元素 kernel |
| 越界守护 | `if (index >= count) return;` | `bsplines_kernels.cuh:38` |
| 线程总数 | $\lceil N / 256 \rceil \times 256 \ge N$ | 可能略大于 N，尾部线程由越界守护跳过 |

**示例**：若 $N = 1000$，则 `gridDim.x = 4`，总线程数 $= 4 \times 256 = 1024$，最后 24 个线程越界返回。

### 7. 同步、临时缓冲区、FFT/平台库调用与错误处理

#### 7.1 同步

| 位置 | 同步方式 | 说明 |
| --- | --- | --- |
| `DeviceArray::from_host` | `cudaStreamSynchronize` | H2D 后同步，确保拷贝完成 |
| `DeviceArray::to_host` | `cudaStreamSynchronize` | D2H 前同步，确保 kernel 完成 |
| `gauss_spline_device` 内 | 无同步 | kernel 异步启动，调用者负责后续同步 |
| Kernel 内 | 无需同步 | 逐元素无数据依赖 |

#### 7.2 临时缓冲区

- **无临时缓冲区**：`gauss_spline_device` 不分配任何中间设备数组。
- 常量 `r_signsq` 和 `norm` 通过 kernel launch 参数传递，寄存器/常量缓存使用。

#### 7.3 FFT / 平台库调用

- **不调用 FFT**：高斯近似是直接公式计算，不涉及频域。
- **不调用 dlfft/dlrand**：无需科学计算库或随机数。
- **仅使用 CUDA math**：`sqrtf`、`expf`（device 端），`std::sqrt`、`std::exp`（host 端）。

#### 7.4 错误处理

| 检查 | 抛出条件 | 异常类型 | 位置 |
| --- | --- | --- | --- |
| 输出大小不匹配 | `x.size() != y.size()` | `std::invalid_argument` | `bsplines_typed.cu:17-18` |
| 负 n | `n < 0` | `std::invalid_argument` | `bsplines_typed.cu:38-39` |
| kernel 启动错误 | launch 后检查 | `CUDA_KERNEL_CHECK()` 宏 | `kernel_launch.h:41` |
| 类型不支持 | 编译期 | `static_assert` | `bsplines_typed.cpp:12`, `bsplines_typed.cu:16` |

### 8. 数学/物理原理到 CPU 代码、GPU 代码的三方映射

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $\sigma^2 = \dfrac{n+1}{12}$ | `bsplines.py:24` (基准) `loop_prep` 中 `signsq = (n+1)/12.0`，**double** 精度 | `bsplines_typed.h:102` `signsq = static_cast<float>(n+1)/12.0F`，**float** 精度 | `bsplines_typed.cu:42` `signsq = static_cast<float>(n+1)/12.0F`，**float** 精度 | Python 用 `double`；C++ CPU/GPU 均用 `float`。精度降低但节省寄存器 |
| $\dfrac{1}{2\sigma^2}$ | `bsplines.py:25` (基准) `r_signsq = 0.5/signsq`，**double** | `bsplines_typed.h:103` `r_signsq = 0.5F/signsq`，**float** | `bsplines_typed.cu:44` `r_signsq = 0.5F/signsq`，**float** | 同上，精度由 double→float |
| $\dfrac{1}{\sqrt{2\pi\sigma^2}}$ | `bsplines.py:20` (基准) 每线程重复计算 `1/sqrt(2.0*M_PI*signsq)` | `bsplines_typed.h:105` `norm = 1.0F/sqrt(2.0F*pi*signsq)`，每个元素内重复计算 | `bsplines_typed.cu:45` host 预计算 `norm`，作为 kernel 参数传入 | GPU 版本将归一化常数提到 host 端预计算，避免每个线程重复 sqrt+除法；Python 和 CPU 均未外提 |
| $\exp\!\left(-\dfrac{x^2}{2\sigma^2}\right)$ | `bsplines.py:20` (基准) `exp(-(x*x)*r_signsq)`，类型为 `T` | `bsplines_typed.h:107` `std::exp(-(x*x)*r_signsq)`，`float` | `bsplines_kernels.cuh:42` `expf(-(value*value)*reciprocal_twice_variance)`，`float` | Python 保留原始 `T`（float32 或 float64）；C++ 统一提升到 float 计算 |
| $f(x) = \dfrac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\dfrac{x^2}{2\sigma^2}\right)$ | `bsplines.py:20` (基准) `output = 1/sqrt(...)*exp(...)` | `bsplines_typed.h:107` `norm * std::exp(...)` | `bsplines_kernels.cuh:41-42` `normalization * expf(...)` | 表达式相同，组织方式不同：Python 逐元素内组合；C++ GPU 用预计算 `norm` 乘 `expf` |
| 类型提升 T→ComputeT | Python ElementwiseKernel 模板 `T`，保留原 dtype | `BsplineTypePolicy<T>::load` → `float`，`::store` → `T` | 同 CPU | Python 保留原始 dtype（含 float64）；C++ 固定提升到 float32 |
| $n < 0$ 校验 | 无校验，负 n 导致 NaN | `if (n < 0) throw std::invalid_argument(...)` | 同 CPU | C++ 增加了显式参数校验 |

### 9. 与 cuSignal Python 实现相同、等价替换和有意不同的部分

#### 9.1 相同部分

| 方面 | 说明 |
| --- | --- |
| 核心公式 | $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{x^2}{2\sigma^2}\right)$，$\sigma^2 = (n+1)/12$ |
| 方差公式 | `signsq = (n + 1) / 12.0`——Python 用 `double`，C++ 用 `float`，但公式结构相同 |
| 除法→乘法优化 | 均预计算 `r_signsq = 0.5 / signsq`，用乘法代替除法 |
| 逐元素独立性 | 每个元素独立计算，无数据依赖、无归约、无共享内存 |
| 同型输出 | 输出 dtype 与输入 dtype 一致（Python 通过 `T` 模板，C++ 通过 `BsplineTypePolicy<T>::store`） |

#### 9.2 等价替换

| Python 实现 | C++ 实现 | 等价性说明 |
| --- | --- | --- |
| `cp.ElementwiseKernel("T x, int32 n", "T output", ...)` | `__global__ void gauss_spline_kernel(const T* x, T* y, size_t count, float r_signsq, float norm)` | 功能等价：逐元素并行。CuPy 自动管理 grid/block；C++ 手动配置 `block=256`、`grid=ceil(N/256)` |
| `loop_prep` 字符串中 C++ 代码 | host wrapper 中的 `float` 变量计算 | 功能等价：都是在循环/kernel 外预计算常量。C++ 将常量作为 kernel 参数传入；Python 通过 CuPy JIT 将常量编译到 kernel 中 |
| `cp.asarray(x)` | `DeviceArray<T>::from_host(x)` | 功能等价：将 host 数据拷贝到设备。CuPy 对已有 CuPy 数组零拷贝；`DeviceArray::from_host` 总是拷贝 |
| `M_PI`（CUDA math.h） | `constexpr float pi = 3.14159265358979323846F` | 数值等价：同一 $\pi$ 值的不同表示 |

#### 9.3 有意不同的部分

| 方面 | cuSignal Python | cusignal_cpp C++ | 不同原因 |
| --- | --- | --- | --- |
| **常数精度** | `double`（`loop_prep` 中 `const double signsq`） | `float`（`static_cast<float>(n+1)/12.0F`） | C++ 统一使用 float 以匹配 ZQ500 GPU 的 FP32 计算能力，减少 double 运算开销 |
| **归一化常数预计算** | 每个线程重复计算 `1/sqrt(2.0*M_PI*signsq)` | host 端预计算 `norm`，kernel 参数传入 | C++ GPU 版本显式优化：避免每个线程的 sqrt+除法 |
| **$n < 0$ 校验** | 无校验，产生 NaN | `throw std::invalid_argument` | C++ 选择 fail-fast，避免 NaN 传播到后续计算 |
| **支持类型** | 任意 CuPy dtype（float32, float64, complex 等） | FP32, FP16, INT32, INT16, INT8 五种 | C++ 显式实例化有限类型集；FP16 和整数是项目扩展 |
| **类型提升策略** | 保留原始 `T`（float64 输入→float64 输出） | 统一提升到 float32 计算，再写回 T | C++ 强制 FP32 中间精度；不支持 float64 |
| **输出数组分配** | CuPy 自动分配输出 | 调用者预分配 `DeviceArray<T> y(x.size())` | C++ 显式内存管理，避免隐式分配 |
| **空输入处理** | kernel 不启动，返回空数组 | `if (!x.empty())` 跳过 kernel 启动 | 等价行为，C++ 显式判断 |
| **FP16 支持** | 不支持（CuPy ElementwiseKernel 的 `T` 不支持 `__half`） | 支持，通过 `__half2float`/`__float2half_rn` | C++ 项目扩展，ZQ500 平台需要 FP16 |

### 10. 精度、dtype、边界条件及 ZQ500 平台限制

#### 10.1 精度分析

| 计算步骤 | Python 精度 | C++ CPU 精度 | C++ GPU 精度 |
| --- | --- | --- | --- |
| `signsq = (n+1)/12.0` | double | float | float |
| `r_signsq = 0.5/signsq` | double | float | float |
| `norm = 1/sqrt(2π·signsq)` | 每线程 double | 每元素 float | host 端 float（一次性） |
| `exp(-(x²)·r_signsq)` | T（float32 或 float64） | float | float（`expf`） |
| 最终输出 | T | T（float→T 截断） | T（float→T 截断） |

**精度影响**：
- Python 的 `double` 常数提供约 15 位有效数字；C++ 的 `float` 常数提供约 7 位有效数字；
- 对 `signsq` 而言，当 `n` 较大（如 `n=1000`）时，`signsq ≈ 83.3`，float 表示误差约 $83.3 \times 2^{-23} \approx 10^{-5}$，相对误差 $\sim 10^{-7}$，通常可接受；
- 对 `norm` 而言，float 的 `sqrtf` 精度约为 $\pm 1$ ULP，归一化常数的相对误差约 $10^{-7}$；
- 最终结果 `norm * expf(...)` 的乘法误差约为 $\pm 1$ ULP，总相对误差约 $10^{-6}$~$10^{-7}$，在 FP32 应用中通常足够。

#### 10.2 dtype 行为

| 输入 dtype | Python 行为 | C++ CPU 行为 | C++ GPU 行为 |
| --- | --- | --- | --- |
| `float` / `float32` | float32 输出 | float 输出 | float 输出 |
| `float64` | float64 输出 | **不支持**（编译错误） | **不支持** |
| `__half` / `float16` | **不支持** | half 输出（float 中间） | half 输出（float 中间） |
| `int32` | int32 输出（截断为 0） | int32 输出（截断为 0） | int32 输出（截断为 0） |
| `int16` | **不支持** | int16 输出（截断为 0） | int16 输出（截断为 0） |
| `int8` | **不支持** | int8 输出（截断为 0） | int8 输出（截断为 0） |
| `complex` | 报错 | **不支持** | **不支持** |

#### 10.3 边界条件

| 条件 | Python | C++ CPU | C++ GPU |
| --- | --- | --- | --- |
| `n < 0` | NaN（`sqrt` 负数） | `std::invalid_argument` | `std::invalid_argument` |
| `n = 0` | $\sigma^2 = 1/12$，正常 | 同 | 同 |
| 空数组 | 返回空数组 | 返回空 `vector` | 跳过 kernel，`y` 不修改 |
| `|x|` 极大 | `exp` 下溢为 0 | 同 | 同 |
| `x` 整数型 | 输出全 0 | 输出全 0 | 输出全 0 |

#### 10.4 ZQ500 平台限制

| 限制 | 说明 |
| --- | --- |
| 无 float64 硬件 | ZQ500 GPU 的 double 性能极低或不支持；cusignal_cpp 统一用 float 计算 |
| FP16 需类型转换 | ZQ500 支持 FP16 存储，但超越函数需先转 FP32；`__half2float`/`__float2half_rn` 处理 |
| CUDA 流支持 | 当前使用默认流（`nullptr`）；ZQ500 支持多流但当前未利用 |
| Kernel 启动开销 | 小数组场景下 kernel 启动开销可能占比高；block=256 是保守选择 |

### 11. 相关测试如何验证正确性，以及仍未覆盖的风险

#### 11.1 测试文件

`test/signal_processing/e3_bsplines_type_smoke.cu`

#### 11.2 测试方法

测试采用 **CPU reference vs GPU output** 对照模式：

1. 对相同输入 `x`，分别调用 `gauss_spline_cpu(x, n)` 和 `gauss_spline_device(dx, dy, n)`；
2. 将 GPU 输出 `dy.to_host()` 与 CPU reference 逐元素比较；
3. 差异在类型相关容差内即通过。

#### 11.3 测试用例覆盖

| 用例名 | 输入 | n 值 | 测试目的 | 代码行号 |
| --- | --- | --- | --- | --- |
| typical | `{-3.0, -1.25, -0.25, 0.0, 0.75, 1.25, 3.0}` | 5 | 典型使用场景 | 111–113 |
| minimum | `{0.5}` | 0 | 最小合法 n | 139 |
| alternate | `{-2.0, 0.5, 2.5}` | 3 | 不同 n 值 | 140 |
| boundary | `{-4.0, 4.0}` | 8 | 大 |x| 和大 n | 141 |
| rank3 | 24 个元素（seed 循环） | 5 | 多维逻辑 shape | 145–150 |
| invalid (CPU) | 空 vector | -1 | 负 n 抛异常 | 115–119 |
| invalid (GPU) | 空 DeviceArray | -1 | 负 n 抛异常 | 120–126 |

#### 11.4 五类型覆盖

| 类型 | 容差 | 说明 |
| --- | --- | --- |
| `float` | `2.0e-6` | FP32 精度 |
| `__half` | `2.0e-3` | FP16 精度损失较大 |
| `int32_t` | `0.0` | 整数截断后应精确匹配（全 0） |
| `int16_t` | `0.0` | 同上 |
| `int8_t` | `0.0` | 同上 |

#### 11.5 仍未覆盖的风险

| 风险 | 说明 |
| --- | --- |
| **float64 输入** | 不支持且无测试；若用户传入 double 数据，编译期报错 |
| **大 N 性能** | 测试仅用 7/3/24 个元素，未验证大规模数据的正确性和性能 |
| **n 极大** | 最大测试 n=8；未验证 n=1000+ 时 float 精度是否足够 |
| **NaN/Inf 输入** | 未测试 `x` 包含 NaN 或 Inf 时 `expf` 的行为 |
| **并发流** | 所有测试使用默认流；未验证多流并发正确性 |
| **与 SciPy 数值对比** | 测试只比较 CPU vs GPU，未与 SciPy/NumPy 金标准对比绝对精度 |
| **非 1 的 blockDim** | block=256 硬编码，未测试其他 block size |
| **空输入 + 负 n** | 空 vector 与 n=-1 的组合测试存在（line 115-126），但空 DeviceArray 的 n=-1 也测试了 |

---

## 版本差异与原理不变量

（当前仅有 V1，此节待后续优化版本追加后填写。）

**原理不变量**：无论工程实现如何变化，以下数学物理原理始终不变：

1. $\sigma^2 = (n+1)/12$（原理 2：均匀分布方差 + 独立和方差可加性）；
2. $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{x^2}{2\sigma^2}\right)$（原理 4：零均值高斯 PDF）；
3. 近似成立的理论基础为中心极限定理（原理 3）；
4. 被近似对象为 cardinal B-spline $\beta^n(x)$（原理 1）。
