# hamming 算子 — Python 源码算法

> 本文件对应阶段二，深入 cuSignal 23.08.00 中 `hamming` 算子的 Python 源码语法与执行逻辑。
> 代码引用同时给出学习副本（`Learning/cusignal-23.08.00`，含学习注释）和只读基准（`ZKX/cusignal-23.08.00`，原始行号）的行号，以避免注释导致行号偏移的混淆。

---

## 一、公开导入路径、定义文件、符号名与行号

### 导入链

| 层级 | 文件（相对于 `operator_comprehension/`） | 符号名 | Learning 副本行号 | ZKX 基准行号 |
| --- | --- | --- | --- | --- |
| 顶层包导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py` | `hamming` | 102 | 102 |
| 模块导出 | `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py` | `hamming` | 30 | 30 |
| 函数定义 | `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py` | `def hamming(M, sym=True)` | 1155 | 1124 |
| GPU kernel | `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py` | `_hamming_kernel` | 1139 | 1112 |

用户可通过以下任一方式访问：

```python
import cusignal
w = cusignal.hamming(M)           # 顶层包

from cusignal.windows import hamming
w = hamming(M)                    # 模块直接导入
```

### 依赖导入

`windows.py` 文件开头（第 14-17 行）导入了运行所需的库：

```python
import warnings
import cupy as cp    # 第 16 行：CuPy 是 GPU 数组计算库，提供 ElementwiseKernel 等 API
import numpy as np   # 第 17 行：NumPy 用于部分 host 端辅助操作
```

`hamming` 函数仅依赖 `cp`（CuPy），不直接依赖 `np` 或 `warnings`。

---

## 二、函数签名、参数、返回值、dtype 与 shape 语义

### 函数签名

```python
def hamming(M, sym=True):
```

### 参数说明

| 参数 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `M` | `int` | （必填） | 窗长度（采样点数）。必须为非负整数；`M < 1` 返回空数组，`M == 1` 返回 `[1.0]`。 |
| `sym` | `bool` | `True` | 是否生成对称窗。`True`（默认）用于 FIR 滤波器设计；`False` 生成周期窗（DFT-even），用于频谱分析。 |

### 返回值

| 属性 | 说明 |
| --- | --- |
| 类型 | `cupy.ndarray`（GPU 数组） |
| dtype | `float64` |
| shape | `(M,)`，一维数组 |
| 归一化 | 最大值归一化为 1（当 `M` 为偶数且 `sym=True` 时，峰值 1 不出现，实际峰值略小于 1） |
| 端点值 | `w[0] = w[M-1] = 0.08`（Hamming 窗端点不为零） |

### 退化情况返回值

| 条件 | 返回值 | dtype | shape |
| --- | --- | --- | --- |
| `M < 1` | `cp.array([])` | float64（默认） | `(0,)` |
| `M == 1` | `cp.ones(1, "d")` | float64 | `(1,)`，值为 `[1.0]` |

---

## 三、从公开 API 到最终计算的调用链

```
cusignal.hamming(M, sym=True)                    # 用户调用
  └─ cusignal/windows/windows.py: hamming()       # 函数体（Learning:1232-1250 / ZKX:1200-1212）
       ├─ 长度守卫: if M < 1 → return cp.array([])
       ├─ 单点守卫: if M == 1 → return cp.ones(1, "d")
       ├─ 周期扩展: if not sym and not odd → M = M + 1
       ├─ GPU 计算: w = _hamming_kernel(size=M)   # 调用 CuPy ElementwiseKernel
       │    └─ loop_prep: const double N = 1.0 / (_ind.size() - 1)
       │    └─ kernel body: w = 0.54 - 0.46 * cos(2.0 * M_PI * i * N)
       ├─ 截断: if not sym and not odd → w = w[:-1]
       └─ return w
```

**关键特征**：`hamming` 没有调用 `general_hamming` 或 `general_cosine` 的通用路径，而是直接使用专用的 `_hamming_kernel`（CuPy ElementwiseKernel）。这是与 `hann`（调用 `general_hamming(M, 0.5, sym)`）和 `general_hamming`（调用 `general_cosine`）的实现路线不同。

---

## 四、按源码执行顺序拆解控制流、分支与表达式

以下按源码执行顺序逐段拆解。行号以 Learning 副本为准，括号内为 ZKX 基准行号。

### 4.1 `_hamming_kernel` 定义（Learning:1139-1152 / ZKX:1112-1121）

```python
_hamming_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    w = 0.54 - 0.46 * cos(2.0 * M_PI * i * N);
    """,
    "_hamming_kernel",
    options=("-std=c++11",),
    loop_prep="const double N { 1.0 / ( _ind.size() - 1 ) };",
)
```

这是模块级的一次性定义，在 `windows.py` 被导入时执行。`cp.ElementwiseKernel` 在此时编译 CUDA kernel 并缓存，后续调用 `_hamming_kernel(size=M)` 时直接启动已编译的 kernel。

### 4.2 长度守卫（Learning:1232-1233 / ZKX:1200-1201）

```python
    if M < 1:
        return cp.array([])
```

- 当 `M` 为 0 或负数时，返回空 CuPy 数组。
- `cp.array([])` 创建一个空的 GPU 数组，默认 dtype 为 float64。

### 4.3 单点守卫（Learning:1235-1236 / ZKX:1202-1203）

```python
    if M == 1:
        return cp.ones(1, "d")
```

- 当 `M == 1` 时，返回单元素数组 `[1.0]`。
- `"d"` 是 NumPy/CuPy 的 dtype 字符码，等价于 `np.float64` / `cp.float64`。
- 这样保证单点情况的返回类型与 kernel 输出的 `float64` 一致。

### 4.4 奇偶判断（Learning:1238 / ZKX:1204）

```python
    odd = M % 2
```

- `M % 2`：`M` 为奇数时 `odd = 1`（Python 中 `1` 为 truthy），`M` 为偶数时 `odd = 0`（falsy）。
- `odd` 的 truthy/falsy 性质用于后续条件判断。

### 4.5 周期窗扩展（Learning:1240-1241 / ZKX:1205-1206）

```python
    if not sym and not odd:
        M = M + 1
```

- 条件 `not sym and not odd`：`sym=False`（周期窗）**且** `M` 为偶数（`odd=0`）时为真。
- 满足条件时将 `M` 加 1，使其变为奇数后再计算窗。
- **原因**：DFT-even 约定要求周期窗的长度为奇数，这样窗的周期延拓在 DFT 周期内端点连续（避免周期延拓时出现重复点）。计算后再截断末点恢复原始长度。

### 4.6 GPU kernel 调用（Learning:1244 / ZKX:1208）

```python
    w = _hamming_kernel(size=M)
```

- 调用 `_hamming_kernel` 并传入 `size=M`，启动 GPU kernel。
- `size=M` 指定输出数组长度和线程数；CuPy 自动配置 grid/block 维度。
- 返回值 `w` 是长度为 `M` 的 `cupy.ndarray`，dtype 为 `float64`。
- 每个 GPU 线程 `i`（$0 \leq i \leq M-1$）独立计算 `w[i] = 0.54 - 0.46 * cos(2π * i / (M-1))`。

### 4.7 截断（Learning:1247-1248 / ZKX:1210-1211）

```python
    if not sym and not odd:
        w = w[:-1]
```

- 条件与 4.5 相同：`sym=False` 且原 `M` 为偶数时执行。
- `w[:-1]`：切片去掉最后一个元素，使输出长度从 `M+1` 恢复为原始 `M`。
- `w[:-1]` 返回的是视图（view）还是副本取决于 CuPy 内部实现，但不影响结果正确性。

### 4.8 返回（Learning:1250 / ZKX:1212）

```python
    return w
```

- 返回 CuPy GPU 数组 `w`，shape 为 `(M,)`，dtype 为 `float64`。

---

## 五、关键语句的 Python/CuPy 语法说明

### 5.1 `cp.ElementwiseKernel` 语法

`cp.ElementwiseKernel(input_params, output_params, body, name, options=..., loop_prep=...)` 是 CuPy 提供的 GPU kernel 定义 API：

| 参数 | `hamming` 中的值 | 语法说明 |
| --- | --- | --- |
| `input_params` | `""` | 输入参数声明字符串，格式为 `"type1 name1, type2 name2"`。空字符串表示无输入参数。 |
| `output_params` | `"float64 w"` | 输出参数声明，声明一个 `float64` 类型的输出变量 `w`。调用 kernel 时 CuPy 据此分配输出数组。 |
| `body` | `"w = 0.54 - ..."` | kernel 主体，C++ 代码字符串。每个线程执行一次。 |
| `name` | `"_hamming_kernel"` | kernel 名称，用于 CUDA 模块管理和调试。 |
| `options` | `("-std=c++11",)` | NVCC 编译选项元组。指定 C++11 标准，因 `loop_prep` 使用了 C++11 的花括号初始化 `{ }`。 |
| `loop_prep` | `"const double N { ... }"` | 在 kernel 启动前执行一次的预处理代码，用于预计算常量。 |

### 5.2 kernel body 中的内置变量

| 变量 | 来源 | 含义 |
| --- | --- | --- |
| `i` | CuPy 内置 | 当前线程的全局索引，范围 $0 \leq i \leq \text{size}-1$，对应采样点 $n$。 |
| `w` | output_params 声明 | 输出变量，每个线程写入一个元素 `w[i]`。 |
| `N` | loop_prep 定义 | 预计算常量 $1/(M-1)$，其中 `_ind.size()` 返回输出数组长度 $M$。 |
| `M_PI` | C++ `<cmath>` | 圆周率 $\pi$ 常量，NVCC 编译时自动可用。 |
| `_ind` | CuPy 内置 | 索引数组对象，`_ind.size()` 返回输出数组长度。 |

### 5.3 `cp.ones(1, "d")` 语法

- `cp.ones(shape, dtype)`：创建全 1 的 CuPy 数组。
- `1` 作为 shape 表示单元素数组，shape 为 `(1,)`。
- `"d"` 是 dtype 字符码，等价于 `cp.float64`，保证与 kernel 输出的 float64 一致。

### 5.4 `w[:-1]` 切片语法

- Python/CuPy 的负索引切片：`w[:-1]` 表示从开头到倒数第二个元素（不含最后一个）。
- 对 CuPy 数组，切片操作返回一个新的数组对象（可能是视图或副本）。

---

## 六、数学公式与具体代码语句的逐项映射

| 数学公式 | 代码语句 | Learning 行号 | ZKX 行号 | 映射说明 |
| --- | --- | --- | --- | --- |
| $w[n] = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{M-1}\right)$ | `w = 0.54 - 0.46 * cos(2.0 * M_PI * i * N);` | 1145 | 1116 | `0.54`→$\alpha$，`0.46`→$1-\alpha$，`M_PI`→$\pi$，`i`→$n$，`N`→$1/(M-1)$，`cos`→$\cos$ |
| $N = \frac{1}{M-1}$ | `const double N { 1.0 / ( _ind.size() - 1 ) };` | 1151 | 1120 | `_ind.size()`→$M$，`1.0 / (M - 1)`→$1/(M-1)$ |
| $n \in \{0, 1, \ldots, M-1\}$ | `i`（CuPy 内置线程索引） | 1145 | 1116 | `i` 从 0 到 `size-1`，即 $0$ 到 $M-1$ |
| 退化：$M < 1 \Rightarrow w = []$ | `if M < 1: return cp.array([])` | 1232-1233 | 1200-1201 | 空数组对应空窗 |
| 退化：$M = 1 \Rightarrow w = [1]$ | `if M == 1: return cp.ones(1, "d")` | 1235-1236 | 1202-1203 | 单点窗值为 1 |
| 周期窗：$M_{\text{even}} \to M+1$ | `if not sym and not odd: M = M + 1` | 1240-1241 | 1205-1206 | DFT-even 约定 |
| 截断：去掉 $w[M]$ | `w = w[:-1]` | 1248 | 1211 | 恢复原始长度 |

---

## 七、底层机制 — CuPy ElementwiseKernel

`hamming` 的核心计算依赖 CuPy 的 `ElementwiseKernel` 机制，而非 FFT、插值或线性代数。

### 7.1 ElementwiseKernel 的工作原理

1. **定义时编译**：`cp.ElementwiseKernel(...)` 在模块导入时将 body 字符串编译为 CUDA PTX 代码，缓存为 CUDA 模块。
2. **调用时启动**：`_hamming_kernel(size=M)` 根据输出参数 `"float64 w"` 分配长度为 $M$ 的 GPU 数组，自动计算 grid/block 维度，启动 kernel。
3. **并行执行**：每个 GPU 线程独立执行 body 一次，线程索引 `i` 从 0 到 $M-1$。
4. **loop_prep 预处理**：`loop_prep` 代码在 kernel 启动前于 host 端执行一次，计算常量 `N`，所有线程共享。

### 7.2 与手动 CUDA kernel 的区别

| 特性 | CuPy ElementwiseKernel | 手动 CUDA kernel |
| --- | --- | --- |
| grid/block 配置 | CuPy 自动 | 需手动指定 |
| 线程索引 | 内置 `i` | 需手动计算 `blockIdx.x * blockDim.x + threadIdx.x` |
| 内存分配 | 根据输出参数自动 | 需手动 `cudaMalloc` |
| 编译 | 模块导入时 JIT 编译 | 需 `nvcc` 预编译或 `cupy.RawKernel` |

### 7.3 未使用的底层机制

- **不使用 FFT**：`hamming` 仅生成时域窗序列，不涉及频域计算。
- **不使用 NumPy**：核心计算完全在 GPU 上完成，不回退到 CPU。
- **不使用通用窗框架**：未调用 `general_cosine` 或 `general_hamming`，使用专用 kernel 避免函数调用开销。

---

## 八、边界处理、异常检查、数值稳定性与 dtype 转换

### 8.1 边界处理

| 边界情况 | 处理方式 | 源码位置 |
| --- | --- | --- |
| `M < 1` | 返回空数组 `cp.array([])` | Learning:1232 / ZKX:1200 |
| `M == 1` | 返回 `cp.ones(1, "d")` 即 `[1.0]` | Learning:1235 / ZKX:1202 |
| `M == 2` 且 `sym=True` | kernel 计算 2 点：`[0.08, 0.08]`，峰值不出现 1 | kernel body |
| `sym=False` 且 `M` 偶数 | 扩展为 `M+1` 再截断 | Learning:1240-1241,1247-1248 / ZKX:1205-1206,1210-1211 |

### 8.2 异常检查

- `hamming` 函数本身**不做**类型检查或整数验证（不调用 `_len_guards`）。
- 若 `M` 为浮点数（如 `5.5`），`M < 1` 和 `M == 1` 的比较仍可执行，但 `M % 2` 和 kernel 的 `size` 参数可能产生意外行为。
- 若 `M` 为负数，`M < 1` 分支捕获并返回空数组。
- 与 `triang` 不同，`hamming` **没有**调用 `_len_guards` 做 `int(M) != M` 的整数验证。

### 8.3 数值稳定性

- kernel body 中 `2.0 * M_PI * i * N` 使用 `double` 精度计算，`M_PI` 和 `N` 均为 `double`，`i` 被 CuPy 提升为 `int64` 后隐式转换为 `double` 参与运算。
- 当 `M` 很大时，`i * N = i / (M-1)` 的最大值为 $M/(M-1) \approx 1$，不会产生大数精度问题。
- `loop_prep` 中 `1.0 / (_ind.size() - 1)` 在 `M = 1` 时会产生除零，但 `M == 1` 已被前置守卫拦截，不会到达 kernel 调用。

### 8.4 dtype 转换

- 输出 dtype 固定为 `float64`，由 kernel 的 `"float64 w"` 声明决定。
- `M == 1` 时 `cp.ones(1, "d")` 显式指定 `float64`，保持一致。
- `M < 1` 时 `cp.array([])` 默认 dtype 为 `float64`。
- 函数不接受 dtype 参数，用户无法更改输出精度。

---

## 九、时间复杂度、空间复杂度与性能瓶颈

### 9.1 时间复杂度

- **计算复杂度**：$O(M)$。每个采样点执行一次余弦运算和两次乘法/加减法。
- **GPU 并行**：$M$ 个线程并行，理想情况下 wall-clock 时间为 $O(1)$（与 $M$ 无关），实际受 GPU 核心数和带宽限制。
- **kernel 启动开销**：每次调用有固定的 kernel 启动延迟（约 5-20 μs），对小的 $M$（如 $M < 100$）可能成为瓶颈。

### 9.2 空间复杂度

- **输出空间**：$O(M)$，一个长度为 $M$ 的 `float64` 数组（$8M$ 字节）。
- **临时空间**：CuPy ElementwiseKernel 不额外分配临时缓冲区；`loop_prep` 的 `N` 是标量常量，不占数组空间。
- **截断 `w[:-1]`**：可能产生一个长度为 $M-1$ 的新数组（视图或副本），峰值空间约 $O(M)$。

### 9.3 性能瓶颈

1. **kernel 启动开销**：对小窗（$M < 256$），kernel 启动延迟可能超过计算时间。
2. **GPU 带宽**：对大窗（$M > 10^6$），写入输出数组的显存带宽可能成为瓶颈。
3. **余弦函数计算**：`cos` 是超越函数，GPU 上的 `cos` 实现需要若干时钟周期，但 `ElementwiseKernel` 的并行性通常能掩盖这一开销。
4. **与 `general_cosine` 路径对比**：`hamming` 使用专用 kernel，避免了 `general_cosine` 中的数组拼接和多次 `cp.arange` 调用，性能更优。

---

## 十、建议阅读顺序与观察要点

### 建议阅读顺序

1. **先看导出链**：从 `cusignal/__init__.py:102` → `cusignal/windows/__init__.py:30` → `windows.py` 的 `def hamming`，理解调用路径。

2. **看 `_hamming_kernel` 定义**（Learning:1139-1152 / ZKX:1112-1121）：
   - 观察问题：`cp.ElementwiseKernel` 的六个参数分别是什么？`""` 和 `"float64 w"` 各代表什么？`loop_prep` 中的 `_ind.size()` 返回什么？为什么用 C++11？

3. **看 `hamming` 函数 docstring**（Learning:1155-1230 / ZKX:1124-1199）：
   - 观察问题：数学公式 `w(n) = 0.54 - 0.46\cos(2\pi n/(M-1))` 与 kernel body 中的代码如何对应？`sym=True` 和 `sym=False` 分别用于什么场景？

4. **看函数体控制流**（Learning:1232-1250 / ZKX:1200-1212）：
   - 观察问题：两个守卫（`M < 1` 和 `M == 1`）为什么需要分别处理？`odd = M % 2` 的结果在 Python 中的 truthy/falsy 行为是什么？`not sym and not odd` 条件在什么情况下为真？为什么周期窗需要扩展再截断？

5. **对比 `hann` 和 `general_hamming` 的实现**（可选）：
   - 观察问题：`hann`（ZKX:855 附近）调用 `general_hamming(M, 0.5, sym)`，而 `hamming` 使用专用 kernel。为什么 `hamming` 不复用 `general_hamming(M, 0.54, sym)`？这种设计选择的性能含义是什么？

### 每段代码应观察的关键点

| 代码段 | 关键观察点 |
| --- | --- |
| `_hamming_kernel` 定义 | CuPy ElementwiseKernel 的参数格式；kernel body 是 C++ 不是 Python；`i` 和 `_ind` 是 CuPy 内置变量 |
| `loop_prep` | 预计算 `N = 1/(M-1)` 避免每个线程重复除法；花括号初始化需要 C++11 |
| `if M < 1` / `if M == 1` | 退化情况处理；`"d"` dtype 字符码的含义 |
| `odd = M % 2` | Python 中 `0` 为 falsy、非零为 truthy，影响后续 `not odd` 判断 |
| `if not sym and not odd` | 周期窗（`sym=False`）且偶数长度时才扩展；DFT-even 约定 |
| `w = _hamming_kernel(size=M)` | `size=M` 决定线程数和输出长度；返回 GPU 数组 |
| `w = w[:-1]` | 负索引切片截断末点；恢复用户指定的原始长度 |

---

## 附：学习副本与只读基准行号对照

由于 Learning 副本中 hamming 之前的其他算子已有学习注释，导致 hamming 部分行号偏移 27 行（kernel 定义处）。以下是关键行号对照：

| 代码位置 | Learning 副本行号 | ZKX 基准行号 | 偏移量 |
| --- | --- | --- | --- |
| `_hamming_kernel = cp.ElementwiseKernel(` | 1139 | 1112 | +27 |
| `w = 0.54 - 0.46 * cos(...)` | 1145 | 1116 | +29 |
| `loop_prep="const double N ..."` | 1151 | 1120 | +31 |
| `def hamming(M, sym=True):` | 1155 | 1124 | +31 |
| `if M < 1:` | 1232 | 1200 | +32 |
| `w = _hamming_kernel(size=M)` | 1244 | 1208 | +36 |
| `return w` | 1250 | 1212 | +38 |

> 注：偏移量逐渐增大是因为 hamming 的 kernel 部分插入了 8 行学习注释，函数体部分插入了 7 行学习注释。
