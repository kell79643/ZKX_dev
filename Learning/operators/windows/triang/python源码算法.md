# triang 算子 — Python 源码算法

> 本文档对应 `Learning/cusignal-23.08.00` 学习副本（已含 `<学习注释：...>`）。行号以学习副本为准，同时标注 `ZKX/cusignal-23.08.00` 只读基准的原始行号以便对照。

---

## 一、公开导入路径、定义文件、符号名和行号

### 导入链

| 层级 | 文件（学习副本相对路径） | 行号 | 语句 | 基准行号 |
| --- | --- | --- | --- | --- |
| 顶层包 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py` | 108-109 | `triang,`（从 `cusignal.windows` 导入） | 108 |
| 模块包 | `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py` | 36-37 | `triang,`（从 `cusignal.windows.windows` 导入） | 36 |
| 定义文件 | `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py` | 239-299 | `def triang(M, sym=True):` | 220-276 |

用户调用方式：`cusignal.triang(M)` 或 `cusignal.windows.triang(M)`。

### 核心组件

| 组件 | 学习副本行号 | 基准行号 | 作用 |
| --- | --- | --- | --- |
| `_triang_kernel` | 204-236 | 195-216 | CuPy ElementwiseKernel，GPU 并行计算三角窗 |
| `triang` 函数 | 239-299 | 220-276 | 公开 API，协调边界检查、对称扩展、kernel 调用和截断 |
| `_len_guards` | 20-27 | 20-24 | 通用辅助：检查窗长度合法性 |
| `_extend` | 30-38 | 27-32 | 通用辅助：DFT-even 对称性扩展 |
| `_truncate` | 41-48 | 35-40 | 通用辅助：截断最后一个采样点 |

---

## 二、函数签名与参数语义

```python
def triang(M, sym=True):
```

| 参数 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `M` | `int` | — | 窗长度（采样点数）。必须为非负整数；`M <= 1` 时返回全 1 数组。 |
| `sym` | `bool` | `True` | `True`：对称窗（用于 FIR 滤波器设计）；`False`：周期窗（DFT-even，用于频谱分析）。 |

| 返回值 | 类型 | dtype | shape | 语义 |
| --- | --- | --- | --- | --- |
| `w` | `cupy.ndarray` | `float64` | `(M,)` | 三角窗序列，最大值归一化为 1（偶数 `M` 且 `sym=True` 时峰值 1 不出现）。 |

---

## 三、调用链

```
cusignal.triang(M, sym)                           # __init__.py:108
  └─ cusignal.windows.triang(M, sym)              # windows/__init__.py:36
       └─ cusignal.windows.windows.triang(M, sym) # windows.py:239
            ├─ _len_guards(M)                     # windows.py:20
            │    └─ (M <= 1) → return cp.ones(M)  # 退化为全 1
            ├─ _extend(M, sym)                    # windows.py:30
            │    └─ sym=False → M:=M+1, needs_trunc=True
            ├─ _triang_kernel(size=M)             # windows.py:204 (GPU kernel)
            │    └─ 每个线程 i 计算 w[i]
            └─ _truncate(w, needs_trunc)          # windows.py:41
                 └─ needs_trunc=True → w:=w[:-1]
```

---

## 四、按源码执行顺序拆解

### 4.1 边界守卫 `_len_guards`（windows.py:20-27 / 基准 20-24）

```python
def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError("Window length M must be a non-negative integer")
    return M <= 1
```

**执行逻辑**：
1. `int(M) != M`：检查 `M` 是否为整数。若传入浮点数如 `3.5`，`int(3.5)=3 != 3.5`，触发异常。
2. `M < 0`：检查非负。负数触发异常。
3. `return M <= 1`：`M=0` 或 `M=1` 时返回 `True`，调用方将跳过 kernel 直接返回 `cp.ones(M)`。

**边界处理**：`M=0` 返回空数组 `cp.ones(0)`；`M=1` 返回 `cp.ones(1)` 即 `[1.0]`。

### 4.2 对称性扩展 `_extend`（windows.py:30-38 / 基准 27-32）

```python
def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False
```

**执行逻辑**：
- `sym=False`（周期窗）：返回 `(M+1, True)`。计算时多生成 1 个点，后续截断去掉最后一个，使窗满足 DFT-even 性质（`w[0] = w[M]` 的周期延拓）。
- `sym=True`（对称窗）：返回 `(M, False)`，原样计算。

**语法说明**：返回二元组 `(新长度, 是否需要截断)`，Python 元组解包到 `M, needs_trunc`。

### 4.3 GPU kernel `_triang_kernel`（windows.py:204-236 / 基准 195-216）

```python
_triang_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    int n {};
    if ( i < m ) {
        n = i + 1;
    } else {
        n = _ind.size() - i;
    }

    if ( odd ) {
        w = 2.0 * n / ( _ind.size() + 1.0 );
    } else {
        w = ( 2.0 * n - 1.0 ) / _ind.size();
    }
    """,
    "_triang_kernel",
    options=("-std=c++11",),
    loop_prep="const int m { static_cast<int>( 0.5 * _ind.size() ) }; \
               const bool odd { _ind.size() & 1 };",
)
```

#### CuPy `ElementwiseKernel` 语法说明

`cp.ElementwiseKernel(input_params, output_params, body, name, options, loop_prep)` 创建一个逐元素并行 GPU kernel：

| 参数 | 值 | 作用 |
| --- | --- | --- |
| `input_params` | `""` | 无输入参数。输出数组由调用时的 `size=M` 自动创建。 |
| `output_params` | `"float64 w"` | 输出为 `float64` 类型的标量 `w`，每个线程写入一个元素。 |
| `body` | C++ 代码字符串 | 每个线程执行的核函数体。 |
| `name` | `"_triang_kernel"` | kernel 名称，用于 CUDA 编译缓存。 |
| `options` | `("-std=c++11",)` | 编译选项，指定 C++11 标准（因代码使用花括号初始化 `int n {};`）。 |
| `loop_prep` | C++ 代码字符串 | 循环前执行一次的预处理代码，计算所有线程共享的常量。 |

#### CuPy 内置变量

| 变量 | 类型 | 语义 |
| --- | --- | --- |
| `i` | `int` | 线程索引（0-based），等于输出数组的元素下标。 |
| `_ind.size()` | `size_t` | 输出数组的大小，即窗长度 `M`。 |

#### `loop_prep` 预计算

```cpp
const int m { static_cast<int>( 0.5 * _ind.size() ) };  // m = floor(M/2)
const bool odd { _ind.size() & 1 };                      // odd = (M 为奇数)
```

- `m`：前半与后半的分界点。`static_cast<int>` 对正数等价于 `floor`。
- `odd`：`M & 1` 是按位与操作，`M` 为奇数时结果为 `1`（`true`），偶数时为 `0`（`false`）。

#### kernel body 逐行解析

```cpp
int n {};
```
声明中间变量 `n`，`{}` 是 C++11 值初始化，将 `n` 初始化为 `0`。`n` 表示"从最近端点到当前位置的 1-based 距离"。

```cpp
if ( i < m ) {
    n = i + 1;
} else {
    n = _ind.size() - i;
}
```
- 前半部分（`i < floor(M/2)`）：`n = i + 1`，从 `1` 递增。
- 后半部分（`i >= floor(M/2)`）：`n = M - i`，从右端递减。
- 这形成对称的三角形：`n` 在两端为 `1`，在中心最大。

```cpp
if ( odd ) {
    w = 2.0 * n / ( _ind.size() + 1.0 );
} else {
    w = ( 2.0 * n - 1.0 ) / _ind.size();
}
```
- **奇数 `M`**：$w = \frac{2n}{M+1}$。分母 $M+1$，端点 $w[0] = \frac{2}{M+1} \neq 0$。
- **偶数 `M`**：$w = \frac{2n - 1}{M}$。分母 $M$，端点 $w[0] = \frac{1}{M} \neq 0$。

### 4.4 截断 `_truncate`（windows.py:41-48 / 基准 35-40）

```python
def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w
```

**执行逻辑**：`needed=True` 时返回 `w[:-1]`（去掉最后一个元素的切片），否则原样返回。

**语法说明**：`w[:-1]` 是 CuPy 数组切片，等价于去掉最后一个元素。对 `cupy.ndarray` 产生视图（view），不复制数据。

### 4.5 `triang` 函数主体（windows.py:239-299 / 基准 220-276）

```python
def triang(M, sym=True):
    # ... docstring ...
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _triang_kernel(size=M)

    return _truncate(w, needs_trunc)
```

**执行顺序**：
1. `_len_guards(M)`：若 `M <= 1`，返回 `cp.ones(M)`（空数组或单元素 `[1.0]`）。
2. `_extend(M, sym)`：根据 `sym` 决定是否扩展 `M`。
3. `_triang_kernel(size=M)`：启动 GPU kernel，每个线程计算一个 `w[i]`，返回 `float64` CuPy 数组。
4. `_truncate(w, needs_trunc)`：若需要，去掉最后一个元素。

---

## 五、数学公式与代码语句逐项映射

### 5.1 奇数 `M` 分支

| 数学公式（原理 4 变体 B 奇数） | 代码语句 | 学习副本行号 |
| --- | --- | --- |
| $n = i + 1$（前半，$i < \lfloor M/2 \rfloor$） | `n = i + 1;` | 218 |
| $n = M - i$（后半，$i \geq \lfloor M/2 \rfloor$） | `n = _ind.size() - i;` | 220 |
| $w[i] = \frac{2n}{M+1}$ | `w = 2.0 * n / ( _ind.size() + 1.0 );` | 224 |

**验证（`M=5`，`m=2`，`odd=true`）**：

| $i$ | 分支 | $n$ | $w = 2n/6$ | 手算值 |
| --- | --- | --- | --- | --- |
| 0 | $i < 2$ | 1 | 2/6 | 0.333 |
| 1 | $i < 2$ | 2 | 4/6 | 0.667 |
| 2 | $i \geq 2$ | 5-2=3 | 6/6 | 1.0 |
| 3 | $i \geq 2$ | 5-3=2 | 4/6 | 0.667 |
| 4 | $i \geq 2$ | 5-4=1 | 2/6 | 0.333 |

### 5.2 偶数 `M` 分支

| 数学公式（原理 4 变体 B 偶数） | 代码语句 | 学习副本行号 |
| --- | --- | --- |
| $n = i + 1$（前半，$i < \lfloor M/2 \rfloor$） | `n = i + 1;` | 218 |
| $n = M - i$（后半，$i \geq \lfloor M/2 \rfloor$） | `n = _ind.size() - i;` | 220 |
| $w[i] = \frac{2n - 1}{M}$ | `w = ( 2.0 * n - 1.0 ) / _ind.size();` | 226 |

**验证（`M=4`，`m=2`，`odd=false`）**：

| $i$ | 分支 | $n$ | $w = (2n-1)/4$ | 手算值 |
| --- | --- | --- | --- | --- |
| 0 | $i < 2$ | 1 | 1/4 | 0.25 |
| 1 | $i < 2$ | 2 | 3/4 | 0.75 |
| 2 | $i \geq 2$ | 4-2=2 | 3/4 | 0.75 |
| 3 | $i \geq 2$ | 4-3=1 | 1/4 | 0.25 |

### 5.3 `loop_prep` 预计算映射

| 数学量 | 代码表达式 | 学习副本行号 |
| --- | --- | --- |
| $m = \lfloor M/2 \rfloor$ | `static_cast<int>( 0.5 * _ind.size() )` | 234 |
| $\text{odd} = (M \bmod 2 = 1)$ | `_ind.size() & 1` | 235 |

---

## 六、底层机制

### CuPy ElementwiseKernel

`triang` 的核心计算完全由 `cp.ElementwiseKernel` 完成，不使用 FFT、插值、卷积或线性代数库。具体机制：

1. **编译**：CuPy 在首次调用时将 C++ body 字符串编译为 CUDA PTX/SASS，缓存到 `~/.cupy/kernel_cache/`。`options=("-std=c++11",)` 指定编译器使用 C++11 标准。
2. **线程启动**：调用 `_triang_kernel(size=M)` 时，CuPy 自动启动足够多的 GPU 线程覆盖 `M` 个输出元素，每个线程执行一次 body。
3. **索引映射**：CuPy 内置变量 `i` 自动映射为线程全局索引，等于输出数组的元素下标。
4. **`loop_prep`**：在 kernel 入口处执行一次，所有线程共享 `m` 和 `odd` 的值（通过 constant/register 传播）。
5. **输出创建**：`size=M` 告知 CuPy 创建长度为 `M` 的 `float64` 输出数组，kernel 执行完毕后返回该数组。

---

## 七、边界处理、异常检查、数值稳定性与 dtype 转换

### 边界处理

| 情况 | 处理方式 | 代码位置 |
| --- | --- | --- |
| `M` 为负数 | `raise ValueError` | windows.py:23 |
| `M` 不是整数 | `raise ValueError` | windows.py:23 |
| `M = 0` | `_len_guards` 返回 `True`，返回 `cp.ones(0)`（空数组） | windows.py:289-290 |
| `M = 1` | `_len_guards` 返回 `True`，返回 `cp.ones(1)` 即 `[1.0]` | windows.py:289-290 |
| `sym=False` | `M` 扩展为 `M+1`，计算后截断最后 1 个点 | windows.py:293, 299 |

### 异常检查

- 仅 `_len_guards` 中的 `int(M) != M or M < 0` 检查。无其他显式异常。
- 若 `M` 为浮点数但等于整数（如 `M=5.0`），`int(5.0) == 5.0`，不触发异常，后续 `_extend` 和 kernel 使用 `size=5.0` 时 CuPy 会自动转为整数。

### 数值稳定性

- kernel 中所有计算为简单的加减乘除，无 `sin`、`cos`、`exp` 等超越函数，数值风险极低。
- 奇数分支分母 `M+1.0` 和偶数分支分母 `_ind.size()` 均为 `double` 浮点运算，`M >= 2` 时无除零风险。
- 输出始终为 `float64`，不涉及低精度 dtype 截断。

### dtype 转换

- 输出固定为 `float64`（由 `"float64 w"` 声明）。
- 即使输入 `M` 为 `int32`，输出仍为 `float64`。
- 不支持指定输出 dtype（与 SciPy `triang` 一致）。

---

## 八、时间复杂度、空间复杂度与性能瓶颈

### 时间复杂度

- $O(M)$：每个线程执行常数时间操作（比较、加减、乘除），共 $M$ 个线程。
- GPU 并行执行，实际墙钟时间约为 $O(M / \text{线程数}) + \text{kernel 启动开销}$。

### 空间复杂度

- $O(M)$：输出数组 `w` 占用 $8M$ 字节（`float64`）。
- `loop_prep` 中的 `m` 和 `odd` 为标量，$O(1)$。
- `_truncate` 的 `w[:-1]` 为视图，不额外分配内存。

### 性能瓶颈

1. **kernel 启动开销**：对于很小的 `M`（如 $M < 1000$），kernel 启动延迟（约 5-20 μs）可能占主导，GPU 并行优势不明显。
2. **首次编译**：首次调用 `_triang_kernel` 时触发 CUDA 编译，可能耗时数秒；后续调用使用缓存。
3. **内存分配**：`_triang_kernel(size=M)` 分配新的 `float64` 数组，频繁调用时分配开销可累积。
4. **`_truncate` 视图**：`w[:-1]` 返回视图而非连续数组，后续使用可能触发隐式拷贝。

---

## 九、建议阅读顺序与观察问题

### 推荐阅读顺序

1. **导出入口**：`cusignal/__init__.py:108` → `cusignal/windows/__init__.py:36`，确认 `triang` 的导入路径。
2. **辅助函数**：`windows.py:20-48`（基准 20-40），先理解 `_len_guards`、`_extend`、`_truncate` 三个通用 helper 的简单逻辑。
3. **`triang` 函数签名与 docstring**：`windows.py:239-288`（基准 220-275），了解参数、返回值和官方示例。
4. **`triang` 函数体**：`windows.py:289-299`（基准 276-276），跟踪 4 步调用流程。
5. **`_triang_kernel` 定义**：`windows.py:204-236`（基准 195-216），这是核心计算逻辑，需要结合 CuPy ElementwiseKernel 文档理解。
6. **`loop_prep`**：`windows.py:232-235`（基准 214-215），理解 `m` 和 `odd` 的预计算。
7. **kernel body**：`windows.py:215-228`（基准 198-211），逐行理解 `n` 的计算和奇偶分支。

### 每段代码应观察的问题

| 代码段 | 观察问题 |
| --- | --- |
| `_len_guards` | `int(M) != M` 如何区分 `5` 和 `5.0`？为什么 `M=1` 被视为退化？ |
| `_extend` | `sym=False` 时为什么要多算 1 个点？DFT-even 对称性的物理含义是什么？ |
| `_triang_kernel` 参数 | 为什么输入参数为空？`size=M` 如何决定输出数组大小？ |
| `loop_prep` | `static_cast<int>(0.5 * _ind.size())` 对奇数和偶数 `M` 分别得到什么？`& 1` 运算如何判断奇偶？ |
| kernel body 前半/后半 | `i < m` 和 `i >= m` 的分界点 `m` 对 `M=5` 和 `M=4` 分别是多少？`n` 的值如何形成对称三角形？ |
| 奇偶分支 | 奇数分母为 `M+1`，偶数分母为 `M`，为什么不同？这如何保证端点不为零？ |
| `_truncate` | `w[:-1]` 返回视图还是副本？对后续使用有何影响？ |

---

## 十、与阶段一数学原理的对应关系

| 阶段一原理 | 源码落实位置 | 对应关系 |
| --- | --- | --- |
| 原理 4 变体 B 奇数公式 $w = \frac{2n}{M+1}$ | `windows.py:224`（基准 207） | 逐项一致：$n = i+1$ 或 $M-i$，分母 $M+1$ |
| 原理 4 变体 B 偶数公式 $w = \frac{2n-1}{M}$ | `windows.py:226`（基准 209） | 逐项一致：$n = i+1$ 或 $M-i$，分母 $M$ |
| 原理 4 前后半分界 $\lfloor M/2 \rfloor$ | `windows.py:234`（基准 214） `loop_prep` 中 `m` | 一致 |
| 原理 5 加窗操作（时域乘法） | `triang` 返回 `w` 供后续 `x * w` 使用 | `triang` 只生成窗序列，加窗由调用方完成 |
| 原理 3 sinc² 频域特性 | docstring 示例（`windows.py:265-273`）展示频率响应 | 文档示例验证，非 kernel 内部计算 |
