# qmf Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开调用路径

qmf 在 cuSignal 23.08.00 中有两条公开访问路径，最终都指向同一个函数对象：

```text
cusignal.qmf(hk)
cusignal.wavelets.qmf(hk)
```

定位如下。学习副本行号包含后来插入的 `<学习注释：...>`，只读基准行号保持原始源码位置。

| 层次 | 共享根目录 `ZKX_dev/` 下的路径 | 符号 | 学习副本行号 | 只读基准行号 |
| --- | --- | --- | --- | --- |
| 顶层公开导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py` | 导入 `qmf` | 105 | 85 |
| wavelets 子包导出 | `Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py` | 导入 `qmf` | 17 | 14 |
| GPU kernel | `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py` | `_qmf_kernel` | 22--40 | 19--28 |
| 公开函数 | `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py` | `qmf` | 44--55 | 31--41 |

### 1.2 函数签名

```python
qmf(hk)
```

源码没有类型注解、默认值、关键字专用参数或可选参数。

### 1.3 参数、返回值、dtype 与 shape

| 项目 | docstring 意图 | 当前源码实际行为 |
| --- | --- | --- |
| `hk` | 标题称它为低通 FIR 系数；参数说明却误写为高通系数 | 只要求对象支持 `len(hk)`；系数值、dtype、shape 其余维度全部不读取 |
| 默认值 | 无 | 调用时必须传入一个位置参数或同名关键字参数 |
| 返回值 | 应为由 `hk` 生成的高通 QMF 系数 | 返回由 `_qmf_kernel` 新分配的 CuPy GPU 数组 |
| 输出 shape | 按意图应与一维 `hk` 等长 | 一维 `(len(hk),)`；若传多维数组，只使用第一维长度，而不是 `hk.size` |
| 输出 dtype | 按正确 QMF 语义应保存输入系数的信息与精度 | 固定为 `int64`，与 `hk.dtype` 无关 |
| 输出元素 | 应为 $(-1)^i hk[L-1-i]$ | 实际为 $(-1)^i(L-1-i)$ |

这里的“标题”是 docstring 第一句 `Return high-pass qmf filter from low-pass`。同一 docstring 的 `Coefficients of high-pass filter.` 与标题自相矛盾；结合函数名、SciPy 同名接口和 QMF 数学定义，标题所说的“输入低通、输出高通”才是接口意图。但解释当前程序时必须以函数体为准：它没有使用任何滤波器系数。

## 2. 当前算子的完整相关源码

下面严格摘录只读基准 `ZKX/cusignal-23.08.00` 的原文，不包含学习注释；没有省略 qmf 的签名、完整 docstring、函数体、直接 kernel 或必要公开导出。该 docstring 本身没有示例、`Returns`、异常或 Notes 段落，因此不存在被省略的 docstring 示例。

### 2.1 必要的 CuPy 导入

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:14`

```python
import cupy as cp
```

### 2.2 `_qmf_kernel` 完整定义

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:19`

```python
_qmf_kernel = cp.ElementwiseKernel(
    "",
    "int64 output",
    """
    const int sign { ( i & 1 ) ? -1 : 1 };
    output = ( _ind.size() - ( i + 1 ) ) * sign;
    """,
    "_qmf_kernel",
    options=("-std=c++11",),
)
```

### 2.3 `qmf` 完整定义与完整 docstring

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:31`

```python
def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    return _qmf_kernel(size=len(hk))
```

### 2.4 wavelets 子包公开导出

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14`

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

### 2.5 cusignal 顶层公开导出

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:85`

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

`cwt`、`morlet`、`morlet2` 和 `ricker` 只是与 qmf 共用同一导入语句的其他公开符号，不属于本次学习范围，本文不展开它们的定义。

## 3. docstring 逐行翻译与解释

空行只负责排版，按规则无需单独解释。下面覆盖 docstring 的每一行非空内容，包括起止三引号。

#### docstring 概要与接口意图

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:45` 至 `:46`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:32` 至 `:33`。

```python
    """
    Return high-pass qmf filter from low-pass
```

逐行对应说明：

- 第 1 行：开始函数 docstring。它是函数对象的 `__doc__`，不是普通注释。
- 第 2 行：“由低通滤波器返回高通 QMF 滤波器。”这句话声明输入应是低通系数、输出应是高通系数。

#### 参数章节与 `hk` 说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:48` 至 `:51`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:35` 至 `:38`。

```python
    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy/SciPy 风格 docstring 中的标题下划线，供文档生成器识别章节。
- 第 3 行：参数名为 `hk`，文档期望它是可转换或表现得像数组的对象；源码实际只调用 `len`。
- 第 4 行：“高通滤波器的系数。”它与第 33 行“from low-pass”矛盾，应视为继承自 SciPy 文档的历史笔误，而不能据此改变 QMF 定义。

#### 结束 docstring

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:53` 至 `:53`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:40` 至 `:40`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束 docstring。源码没有 `Returns`、shape、dtype、异常、Notes 或 Examples 章节。


## 4. 源代码逐行解释

### 4.1 CuPy 导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:16`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:14`。

```python
import cupy as cp
```

`import ... as ...` 加载 CuPy 并绑定别名 `cp`；本算子随后用 `cp.ElementwiseKernel` 构造 GPU 逐元素 kernel。相邻的 NumPy 与 `convolve` 导入服务同文件的其他小波函数，不属于 `qmf` 的直接依赖。

### 4.2 构造 `_qmf_kernel`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:22` 至 `:40`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:19` 至 `:28`。

```python
_qmf_kernel = cp.ElementwiseKernel(
    "",
    "int64 output",
    """
    const int sign { ( i & 1 ) ? -1 : 1 };
    output = ( _ind.size() - ( i + 1 ) ) * sign;
    """,
    "_qmf_kernel",
    options=("-std=c++11",),
)
```

整个代码块是一次 `cp.ElementwiseKernel(...)` 构造调用：

- 左侧赋值把构造出的可调用 kernel 保存为 `_qmf_kernel`。
- 空字符串 `""` 表示 kernel 没有输入数组参数；`"int64 output"` 声明输出名和固定的 64 位整数 dtype。因此传给公开函数的 `hk` 系数值不会进入设备计算。
- 三引号字符串是每个输出元素执行的 CUDA-C/C++ operation。`i & 1` 检查索引最低位，三元运算令偶数位置 `sign=1`、奇数位置 `sign=-1`；`_ind.size()-(i+1)` 等于 $L-1-i$，故当前实现输出 $(-1)^i(L-1-i)$。
- `"_qmf_kernel"` 是内部 kernel 名；`options=("-std=c++11",)` 是带尾逗号的单元素 tuple，要求用 C++11 编译，以支持花括号初始化。末行 `)` 关闭构造调用。

### 4.3 `qmf` 包装函数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:44` 至 `:55`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:31` 至 `:41`。

```python
def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    return _qmf_kernel(size=len(hk))
```

`def qmf(hk):` 定义只有一个必填参数的函数。docstring 的摘要说“由低通滤波器返回高通 QMF”，参数说明却写成“高通滤波器系数”，两处文字存在矛盾；它们都只是文档字符串，不进行类型检查。函数体先用 `len(hk)` 取得 $L$，再以 `size=L` 调用无输入参数的 kernel；`size` 同时决定输出 shape 和 `_ind.size()`。`return` 直接返回 kernel 生成的 CuPy `int64` 数组。由于 `hk` 本身没有传入 kernel，当前源码只使用长度而没有读取系数值，这是必须保留的实现事实。

### 4.4 两级公开导出

定位：子包学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:17`、只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14`；顶层学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:105`、只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:85`。

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

同一导入语句分别出现在子包与顶层包中，把实现模块里的 `qmf` 绑定为 `cusignal.wavelets.qmf` 和 `cusignal.qmf`。逗号分隔同一模块的其他公开函数；这些名称的存在不表示 `qmf` 会调用它们。

## 5. 调用链与执行顺序

```text
用户调用 cusignal.qmf(hk)
        │
        ├─ 顶层 __init__.py 只做符号导出
        ▼
wavelets.wavelets.qmf(hk)
        │
        ├─ len(hk) → L
        │
        └─ _qmf_kernel(size=L)
                  │
                  ├─ CuPy 分配 int64、shape=(L,) 的 GPU 输出
                  ├─ 对每个 i∈[0,L) 执行 operation
                  └─ output[i]=(-1)^i(L-1-i)
```

按源码执行顺序展开：

1. 首次导入模块时，Python 导入 CuPy并构造 `_qmf_kernel` 对象。
2. Python 创建 `qmf` 函数对象。
3. 两个 `__init__.py` 将同一函数对象导出到子包和顶层。
4. 用户调用时，Python 只读取 `hk` 的长度。
5. `size=L` 告诉无输入参数的 ElementwiseKernel 需要处理多少个元素。
6. CuPy 在当前 CUDA stream 上启动逐元素 kernel；每个逻辑元素索引 $i$ 独立计算符号和倒序索引值。
7. 函数立即返回一个 CuPy 数组。GPU 操作通常相对于主机异步；源码没有显式同步。

Python 层没有 `if`、循环、切片、广播输入或异常处理。唯一的选择逻辑位于 kernel 的 C++ 三元运算符中，用索引奇偶决定符号。

## 6. 意图算法与实际算法总结

### 6.1 正确 QMF 意图

令 $L=\operatorname{len}(hk)$，正确的有限长 QMF 系数变换应为

$$
g[i]=(-1)^i hk[L-1-i],\qquad 0\le i<L.
$$

它包含两个数组步骤：

1. 反转输入：`hk[::-1]`；
2. 偶数位置乘 $+1$、奇数位置乘 $-1$。

### 6.2 当前实现

当前源码执行

$$
y[i]=(-1)^i(L-1-i).
$$

把两式相减可得

$$
y[i]-g[i]=(-1)^i\big((L-1-i)-hk[L-1-i]\big).
$$

所以只有当输入逐项满足

$$
hk[m]=m
$$

时，两者才完全相同。这不是近似误差，而是输入数据通路缺失造成的语义差异。

### 6.3 手工跟踪

对 `hk=[1,2,3,4]`，$L=4$：

| $i$ | `sign` | 正确读取 $hk[L-1-i]$ | 正确 $g[i]$ | 当前 $L-1-i$ | 当前 $y[i]$ |
| --- | --- | --- | --- | --- | --- |
| 0 | $+1$ | 4 | 4 | 3 | 3 |
| 1 | $-1$ | 3 | -3 | 2 | -2 |
| 2 | $+1$ | 2 | 2 | 1 | 1 |
| 3 | $-1$ | 1 | -1 | 0 | 0 |

正确结果是 `[4,-3,2,-1]`，当前结果是 `[3,-2,1,0]`。

## 7. 数学公式与代码逐项映射

| 数学步骤 | 正确公式 | 当前代码位置 | 是否落实 |
| --- | --- | --- | --- |
| 获取长度 | $L=\operatorname{len}(hk)$ | `wavelets.py` 基准 41 / 学习 55 的 `len(hk)` | 是 |
| 输出索引 | $i=0,\ldots,L-1$ | `ElementwiseKernel` 特殊变量 `i` | 是 |
| 交替符号 | $s_i=(-1)^i$ | 基准 23 / 学习 30：`(i & 1) ? -1 : 1` | 是 |
| 反向索引 | $r_i=L-1-i$ | 基准 24 / 学习 32：`_ind.size() - (i + 1)` | 只计算了索引值 |
| 读取反转系数 | $hk[r_i]$ | kernel 没有输入参数 | **否** |
| 生成 QMF | $g[i]=s_i hk[r_i]$ | 当前写成 `output = r_i * sign` | **否** |
| 幅频镜像 | $|G(e^{j\omega})|=|H(e^{j(\pi-\omega)})|$ | 只有正确系数公式才能保证 | 一般不成立 |

kernel 的 `i` 和 `_ind.size()` 只是 GPU 工程机制；反转、交替符号及读取输入系数才构成 QMF 数学算法。当前实现完成了符号和倒序索引计算，却遗漏了真正的数组取值。

## 8. 底层机制与 GPU 执行语义

### 8.1 使用的机制

qmf 只使用 CuPy `ElementwiseKernel`：

- CuPy 根据 `in_params`、`out_params`、`operation` 和名称生成 CUDA kernel；
- `i` 是当前逻辑元素索引；
- `_ind.size()` 是逐元素操作的元素总数；
- 没有输入数组可供广播，因此调用时显式传 `size`；
- 各输出元素互不依赖，可并行计算；
- kernel 只写最终输出，不需要临时缓冲区。

这里没有使用 FFT、插值、卷积、线性代数、共享内存、原子操作或归约。文件顶部的 `convolve` 导入供其他 wavelet 算子使用，不是 qmf 调用链的一部分。

### 8.2 grid、block 与同步

Python 源码没有显式配置 grid 和 block。`ElementwiseKernel` 由 CuPy内部选择启动配置，并把长度 $L$ 的逻辑元素映射到 GPU 线程循环。本文只能确认“每个逻辑索引独立计算一个输出”，不能从这段源码声称固定 block 大小。

函数没有调用 `synchronize()`。现有测试在 `test_wavelets.py` 基准 33--36 中使用默认 stream 调用并显式同步，这是测试计时/取结果的包装，不是 qmf 核心实现。

## 9. 测试证据与缺陷为何被掩盖

直接测试位置：

| 内容 | 学习副本位置 | 只读基准位置 |
| --- | --- | --- |
| `TestWavelets.TestQmf` | `Learning/cusignal-23.08.00/python/cusignal/test/test_wavelets.py:25--49` | `ZKX/cusignal-23.08.00/python/cusignal/test/test_wavelets.py:25--49` |
| `range_data_gen` | `Learning/cusignal-23.08.00/python/cusignal/test/conftest.py:28--36` | `ZKX/cusignal-23.08.00/python/cusignal/test/conftest.py:28--36` |

测试把 `scipy.signal.qmf(sig)` 作为 CPU 参考，把 `cusignal.qmf(sig)` 作为 GPU 结果，并用 `array_equal` 比较。但 fixture 明确生成

```python
cpu_sig = np.arange(num_samps)
```

于是 $hk[m]=m$，正好是错误实现与正确公式相等的唯一特殊输入族。测试验证了该特殊输入，不能证明任意滤波器系数都正确。

更有效的最小测试应至少包含：

- `[1,2,3,4]`，排除输入恰等于索引；
- Haar 浮点系数 `[1/sqrt(2),1/sqrt(2)]`，检查数值与 dtype；
- 含负数、零和非整数的系数；
- 复数系数（若接口声明支持）；
- 长度 1、空数组和多维输入的契约检查。

本学习阶段没有运行这些测试，也没有修改只读基准或修复实现；这里只根据现有源码说明覆盖缺口。

## 10. 边界、异常、dtype 与数值稳定性

| 输入情况 | 当前源码行为或可确认结论 | 风险 |
| --- | --- | --- |
| Python 标量或 0 维数组 | `len(hk)` 无定义时在 kernel 启动前抛 `TypeError` | docstring 没有说明必须至少一维 |
| 一维数组/列表 | 使用元素个数作为 $L$，但不读元素 | 任意不同内容的同长度输入得到完全相同结果 |
| 多维数组 | Python/CuPy 的 `len` 取第一维 | 输出 shape 与输入总元素数不一致；没有维度检查 |
| 空输入 | 没有显式拒绝；按 `size=0` 语义预期得到空输出，但本文未运行确认 | 数学上空序列不是有效滤波器，契约不清晰 |
| 浮点输入 | 输出仍为 `int64` | 丢失全部输入系数与小数精度 |
| 复数输入 | 输出仍为实 `int64` | 丢失实部、虚部和共轭关系 |
| 极大 $L$ | 输出值范围约为 $[-(L-1),L-1]$ | 理论上可能触及 `int64` 范围；现实中会先受显存和数组长度限制 |

实际 kernel 只有整数加减乘，不会出现浮点舍入、除零或消减误差；但这是因为它根本没有执行应有的浮点 QMF 运算。不能把“整数计算稳定”当成算法数值正确。

正确实现若保留输入 dtype，交替乘 $\pm1$ 本身数值稳定，不放大系数绝对误差；主要风险来自低精度输入、错误 dtype 转换以及后续滤波器组累积误差。

## 11. 时间复杂度、空间复杂度与性能瓶颈

令 $L=\operatorname{len}(hk)$。

- **主机时间**：`len(hk)` 通常为 $O(1)$，然后发起一次 kernel 调用。
- **GPU 工作量**：每个输出做常数次整数操作，总工作量 $O(L)$。
- **并行深度**：理想情况下为 $O(1)$ 级逐元素工作加调度开销，实际受线程调度与硬件并发度约束。
- **额外空间**：输出数组占 $8L$ 字节，即 $O(L)$；没有临时数组。
- **读取流量**：当前实现不读取 `hk`，只顺序写一个 `int64` 输出。
- **主要瓶颈**：首次调用可能有 JIT 编译开销；小 $L$ 时 kernel 启动开销远大于两条整数运算；大 $L$ 时主要是输出写带宽。

正确 QMF 实现仍是 $O(L)$ 时间和 $O(L)$ 输出空间，但还需读取 $L$ 个输入系数。即使当前实现少了一次输入读取，也不能把这种“更快”视为有效优化，因为数学结果不正确。

## 12. 学习注释与完整性复核

本阶段只在以下学习副本插入独立合法注释：

- `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py`
- `Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py`
- `Learning/cusignal-23.08.00/python/cusignal/__init__.py`

Python 行使用 `# <学习注释：...>`；kernel 字符串内部的 C++/CUDA 行使用 `// <学习注释：...>`。剔除全部合法学习注释后，三个文件均与 `ZKX/cusignal-23.08.00` 对应只读基准逐行一致。没有修改任何原代码、原注释、空行、缩进或顺序。

## 13. 建议的源码阅读顺序与检查问题

### 第一步：公开入口

先看两个 `__init__.py` 的单行导入。

观察：

1. `cusignal.qmf` 是否经过额外包装？
2. `cusignal.qmf` 与 `cusignal.wavelets.qmf` 是否指向同一实现？

### 第二步：函数签名和 docstring

读 `wavelets.py` 基准 31--40。

观察：

1. 输入到底被描述为低通还是高通？
2. docstring 缺少哪些 dtype、shape、返回值和异常信息？

### 第三步：唯一可执行语句

读基准第 41 行。

观察：

1. `hk` 的值有没有传入 `_qmf_kernel`？
2. `len(hk)` 对一维数组、多维数组和标量分别意味着什么？

### 第四步：kernel 参数声明

读基准 19--22。

观察：

1. 空 `in_params` 能否读取 `hk`？

2. 为什么 `int64 output` 已经决定输出不会保留浮点小波系数？

### 第五步：kernel 两条语句

读基准 23--24，并手算 $i=0,1,2,3$。

观察：

1. `i & 1` 如何等价于索引奇偶判断？
2. `_ind.size()-(i+1)` 是“倒序索引”还是“倒序位置上的输入值”？
3. 正确公式中缺失的 `hk[...]` 应出现在哪里？

### 第六步：测试

最后读 `test_wavelets.py:25--49` 和 `conftest.py:28--36`。

观察：

1. 为什么 `np.arange` 会把索引误当成输入系数？
2. 哪个最小非 `arange` 输入能立刻暴露问题？

## 14. 阶段二自检

1. `_qmf_kernel` 的四个主要构造参数分别是什么？
2. 为什么无输入参数的 ElementwiseKernel 仍能生成长度 $L$ 的输出？
3. `i & 1` 在二进制层面如何检测奇偶？
4. `_ind.size()-(i+1)` 算出的量是什么？它为什么不能替代 `hk[L-1-i]`？
5. 当前函数对两个内容不同但长度相同的输入会返回什么关系的结果？
6. docstring 的哪两行互相矛盾？
7. 当前输出 dtype 为什么固定为 `int64`？
8. qmf 是否调用了卷积、FFT 或任何小波变换流程？
9. GPU 操作是否在函数返回前显式同步？
10. 为什么现有 `np.arange` 测试无法覆盖一般 QMF 输入？

### 阶段二自检参考答案

1. **参考答案 1：**四个主要参数依次是空输入声明 `""`、输出声明 `"int64 output"`、逐元素 operation 字符串、kernel 名 `"_qmf_kernel"`；另有 `options=("-std=c++11",)` 编译选项。证据是学习副本 `wavelets.py:22--40`、基准 19--28 行。
2. **参考答案 2：**调用 `_qmf_kernel(size=len(hk))` 时，`size` 显式给出逻辑元素总数；CuPy据此分配输出并令 `_ind.size()` 等于 $L$，所以即使没有输入数组也能生成结果。
3. **参考答案 3：**整数二进制最低位为 0 表示偶数、为 1 表示奇数；`i & 1` 只保留最低位，三元运算据此选 $+1$ 或 $-1$。
4. **参考答案 4：**它算出倒序位置编号 $L-1-i$，不是该位置的系数值。正确公式还必须执行数组读取 `hk[L-1-i]`；当前 kernel 没有输入参数，无法完成读取。
5. **参考答案 5：**只要 `len` 相同，结果就逐元素完全相同，因为输出只依赖 $L$ 和 $i$，不依赖任何输入内容。
6. **参考答案 6：**摘要 `Return high-pass qmf filter from low-pass` 说输入是低通；参数说明 `Coefficients of high-pass filter.` 却说 `hk` 是高通系数，两者矛盾。
7. **参考答案 7：**`ElementwiseKernel` 的输出参数被硬编码为 `"int64 output"`；函数没有其他 dtype 参数或转换路径，所以输出固定为 CuPy `int64`。
8. **参考答案 8：**没有。调用链只有 `len(hk)` 和自定义逐元素 kernel；同文件的 `convolve` 导入服务其他小波函数，不是 qmf 依赖。
9. **参考答案 9：**没有显式同步。函数返回 kernel 产生的 CuPy 数组；同步由后续取回 host 数据、stream 操作或测试中的 `synchronize()` 负责。
10. **参考答案 10：**测试 fixture 生成 $hk[m]=m$。此时正确式 $(-1)^ihk[L-1-i]$ 与错误式 $(-1)^i(L-1-i)$ 恰好相同，必须换用如 `[1,2,3,4]` 或 Haar 浮点系数才能暴露问题。

## 15. 资料

1. CuPy 开发团队，*User-Defined Kernels — CuPy 12.3.0 documentation*：[https://docs.cupy.dev/en/v12.3.0/user_guide/kernel.html](https://docs.cupy.dev/en/v12.3.0/user_guide/kernel.html)，访问日期：2026-08-09。该文档用于核对 `ElementwiseKernel`、`i`、`_ind.size()` 和显式 `size` 的语义。
2. SciPy 社区，*scipy.signal.qmf — SciPy v1.11.1 Manual*：[https://docs.scipy.org/doc/scipy-1.11.1/reference/generated/scipy.signal.qmf.html](https://docs.scipy.org/doc/scipy-1.11.1/reference/generated/scipy.signal.qmf.html)，访问日期：2026-08-09。该文档用于核对同名 API 的接口意图及其参数文字矛盾。

