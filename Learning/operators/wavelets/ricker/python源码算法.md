# ricker Python 源码算法

## 1. 代码定位与接口概览

### 1.1 阶段与源码边界

- 当前阶段：阶段二，学习 cuSignal 23.08.00 的 Python/CuPy 实现。
- 当前算子：`ricker`，所属模块为 `wavelets`，属于官方 53 项。
- 阅读范围：两个公开导出语句、`ricker` 定义、它直接调用的 `_ricker_kernel`，以及理解
  `cp.ElementwiseKernel` 所必需的 `cupy` 导入。
- 未读取和未摘录：同文件中的 `qmf`、`morlet`、`morlet2`、`cwt` 实现及测试树，因为它们
  不是理解 `ricker` 自身执行逻辑的必要依赖。
- 未运行 Python、CUDA 或测试；本阶段结论来自只读基准源码和 CuPy 官方接口语义。

### 1.2 学习副本与基准一致性

进入阶段二前重新比较了下列文件：

| 学习副本 | 只读基准 | 比较结论 |
| --- | --- | --- |
| `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py` | `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py` | 加注释前完全一致；加注释后过滤合法学习注释仍逐行一致 |
| `Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py` | `ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py` | 加注释前完全一致；加注释后过滤合法学习注释仍逐行一致 |
| `Learning/cusignal-23.08.00/python/cusignal/__init__.py` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py` | 原有差异全是其他算子的合法学习注释；过滤后逐行一致 |

本文写成时，`ricker` 相关定位如下：

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:16` 至 `:16`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:14` 至 `:14`。

```python
import cupy as cp
```

逐行对应说明：

- 第 1 行：`cupy` 导入

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:161` 至 `:161`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:136` 至 `:136`。

```python
)
```

逐行对应说明：

- 第 1 行：`_ricker_kernel`

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:204` 至 `:204`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:177` 至 `:177`。

```python
    return _ricker_kernel(a, size=points)
```

逐行对应说明：

- 第 1 行：`ricker`

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:14` 至 `:14`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:14` 至 `:14`。

```python
from cusignal.acoustics.cepstrum import (
```

逐行对应说明：

- 第 1 行：`cusignal.wavelets.ricker` 导出

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:105` 至 `:105`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:85` 至 `:85`。

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

逐行对应说明：

- 第 1 行：`cusignal.ricker` 顶层导出


学习副本行号会受其他合法学习注释影响；只读基准行号不受影响，是核对原始版本的稳定定位。

### 1.3 公开接口

```python
ricker(points, a)
```

| 项目 | 语义 |
| --- | --- |
| `points` | 输出点数，文档类型为 `int`；也作为 kernel 调用的 `size` |
| `a` | 宽度尺度，文档类型为 scalar；kernel 声明把它转换/接收为 `float64` |
| 返回值 | 长度为 `points` 的一维 CuPy GPU 数组 |
| 返回 dtype | `float64`，由 kernel 的输出声明固定 |
| 返回 shape | `(points,)` |
| 公开调用路径 | `cusignal.ricker`、`cusignal.wavelets.ricker` |

数学上有效的宽度要求 $a>0$。源码没有在 Python 层显式检查 `points` 或 `a`，因此无效参数
由 CuPy 调用规则或浮点计算结果体现，而不是由 `ricker` 主动给出领域化异常。

## 2. 当前算子的完整相关源码

以下摘录全部来自未加学习注释的只读基准 `ZKX/cusignal-23.08.00`，保留原始内容与空行，
没有使用省略号或伪代码。

### 2.1 必要的 CuPy 导入

基准：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:14`

```python
import cupy as cp
```

### 2.2 必要的公开导出语句

基准：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14`

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

基准：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:85`

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

### 2.3 直接 kernel：`_ricker_kernel`

基准：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:121`

```python
_ricker_kernel = cp.ElementwiseKernel(
    "float64 a",
    "float64 total",
    """
    const double vec { i - ( _ind.size() - 1.0 ) * 0.5 };
    const double xsq { vec * vec };
    const double mod { 1 - xsq / wsq };
    const double gauss { exp( -xsq / ( 2.0 * wsq ) ) };

    total = A * mod * gauss;
    """,
    "_ricker_kernel",
    options=("-std=c++11",),
    loop_prep="const double A { 2.0 / ( sqrt( 3 * a ) * pow( M_PI, 0.25 ) ) }; \
               const double wsq { a * a };",
)
```

### 2.4 公开函数：`ricker`

基准：`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:139`

```python
def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,

    where ``A = 2/sqrt(3a)pi^1/4``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = cusignal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(cp.asnumpy(vec2))
    >>> plt.show()

    """
    return _ricker_kernel(a, size=points)
```

## 3. docstring 逐行翻译与解释

以下语义块按只读基准原始顺序覆盖 docstring 的每一行非空内容，包括起止分隔符、标题、
参数、返回值和全部示例。定位中的学习副本行号已计入新增学习注释。

#### docstring 起始分隔符

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:166` 至 `:166`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:140` 至 `:140`。

```python
    """
```

逐行对应说明：

- 第 1 行：开始函数 docstring。它是函数对象的 `__doc__`，不是普通运行语句。

#### 概要与 Mexican-hat 别名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:167` 至 `:167`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:141` 至 `:141`。

```python
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".
```

逐行对应说明：

- 第 1 行：返回 Ricker 小波，它也叫 Mexican hat wavelet；说明两个名称指向同一形状。

#### 解析函数引介

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:169` 至 `:169`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:143` 至 `:143`。

```python
    It models the function:
```

逐行对应说明：

- 第 1 行：下面给出该 API 直接采样的解析函数。

#### Ricker 解析公式

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:171` 至 `:171`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:145` 至 `:145`。

```python
        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,
```

逐行对应说明：

- 第 1 行：形状由多项式调制项与高斯项相乘；按数学优先级应读为 $A(1-x^2/a^2)e^{-x^2/(2a^2)}$。

#### 单位能量归一化常数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:173` 至 `:173`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:147` 至 `:147`。

```python
    where ``A = 2/sqrt(3a)pi^1/4``.
```

逐行对应说明：

- 第 1 行：给出单位能量归一化常数；排版略含糊，kernel 明确实现为 $2/(\sqrt{3a}\pi^{1/4})$。

#### 参数说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:175` 至 `:181`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:149` 至 `:155`。

```python
    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.
```

逐行对应说明：

- 第 1 行：NumPy/SciPy 风格 docstring 的参数章节标题。
- 第 2 行：reStructuredText/NumPy 文档风格的标题下划线，不参与计算。
- 第 3 行：`points` 预期是整数，决定输出元素数量。
- 第 4 行：输出向量包含 `points` 个样本。
- 第 5 行：样本坐标围绕零对称；不是说数组索引从负数开始。
- 第 6 行：`a` 是标量宽度参数，而不是数组输入。
- 第 7 行：$a$ 越大波形越宽；它以采样坐标为单位，不自动表示 Hz。

#### 返回值说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:183` 至 `:186`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:157` 至 `:160`。

```python
    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：标题下划线。
- 第 3 行：返回一维数组；这里的 `N` 实际等于 `points`。在 cuSignal 中实际对象是 CuPy ndarray。
- 第 4 行：数组各元素组成 Ricker 曲线的离散采样。

#### 示例依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:188` 至 `:192`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:162` 至 `:166`。

```python
    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：标题下划线。
- 第 3 行：导入顶层包，以使用公开路径 `cusignal.ricker`。
- 第 4 行：导入 CuPy，用于把 GPU 数组转成绘图库可用的 CPU NumPy 数组。
- 第 5 行：导入 Matplotlib 绘图接口。

#### 示例生成、搬运与绘图

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:194` 至 `:200`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:168` 至 `:174`。

```python
    >>> points = 100
    >>> a = 4.0
    >>> vec2 = cusignal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(cp.asnumpy(vec2))
    >>> plt.show()
```

逐行对应说明：

- 第 1 行：设输出长度为 100；偶数长度意味着对称中心在两个样本之间。
- 第 2 行：设宽度为 4 个样本坐标单位。
- 第 3 行：经顶层导出调用函数，返回 GPU 上的 `float64` 一维数组。
- 第 4 行：查询数组第一维长度。
- 第 5 行：预期打印结果，验证长度与 `points` 相等。
- 第 6 行：`cp.asnumpy` 把设备数组复制到主机，随后 Matplotlib 绘制；这次数据搬运属于示例，不属于 `ricker` 算法。
- 第 7 行：显示图形窗口；不影响已生成的小波。

#### docstring 结束分隔符

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:202` 至 `:202`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:176` 至 `:176`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束 docstring；下一行才是函数体的实际执行语句。


## 4. 源代码逐行解释

### 4.1 导入与公开导出

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:16` 至 `:16`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:14` 至 `:14`。

```python
import cupy as cp
```

逐行对应说明：

- 第 1 行：Python `import ... as ...` 语法；把 `cupy` 模块绑定到别名 `cp`，随后调用 `cp.ElementwiseKernel`。

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:17` 至 `:17`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14` 至 `:14`。

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

逐行对应说明：

- 第 1 行：从实现模块导入多个公开符号；对当前算子而言，它建立 `cusignal.wavelets.ricker`。一行同时列出其他符号不代表需要读取它们的实现。

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:105` 至 `:105`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:85` 至 `:85`。

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

逐行对应说明：

- 第 1 行：在顶层包再次绑定 `ricker`，所以用户可调用 `cusignal.ricker`。这只是名称导出，不计算小波。


### 4.2 `_ricker_kernel` 定义逐行解释

#### kernel 声明、输入输出与逐元素公式

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:136` 至 `:149`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:121` 至 `:128`。

```python
_ricker_kernel = cp.ElementwiseKernel(
    "float64 a",
    "float64 total",
    """
    const double vec { i - ( _ind.size() - 1.0 ) * 0.5 };
    const double xsq { vec * vec };
    const double mod { 1 - xsq / wsq };
    const double gauss { exp( -xsq / ( 2.0 * wsq ) ) };
```

逐行对应说明：

- 第 1 行：调用 CuPy 构造器创建可调用 kernel 对象，并赋给模块私有变量；左括号开始多行实参列表。模块加载时创建对象，GPU 编译通常延迟到调用时。
- 第 2 行：第一个位置参数 `in_params`。字符串声明名为 `a` 的 `float64` 输入；逗号表示后面还有实参。这里没有数组输入来决定输出长度。
- 第 3 行：第二个位置参数 `out_params`。声明每个索引产生一个名为 `total` 的 `float64` 输出，因而返回 dtype 固定为双精度。
- 第 4 行：开始第三个位置参数 `operation` 的 Python 多行字符串；字符串内部是每个元素执行的 CUDA-C/C++ 代码。
- 第 5 行：C++11：定义不可修改的 `double vec`，花括号初始化。`i` 是当前元素索引，`_ind.size()` 是总元素数。公式为 $x_i=i-(N-1)/2$。分号结束语句。
- 第 6 行：计算 $x_i^2$ 并保存为只读局部变量；避免后两行重复乘法。
- 第 7 行：计算多项式调制项 $1-x_i^2/a^2$；`wsq` 在 `loop_prep` 中定义为 $a^2$。整数常量 `1` 会在表达式中提升为 `double`。
- 第 8 行：计算高斯项 $e^{-x_i^2/(2a^2)}$；`2.0` 明确是双精度，`exp` 是设备端指数函数。

#### 写回、编译选项与循环前预计算

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:151` 至 `:161`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:130` 至 `:136`。

```python
    total = A * mod * gauss;
    """,
    "_ricker_kernel",
    options=("-std=c++11",),
    loop_prep="const double A { 2.0 / ( sqrt( 3 * a ) * pow( M_PI, 0.25 ) ) }; \
               const double wsq { a * a };",
)
```

逐行对应说明：

- 第 1 行：将归一化常数、多项式项和高斯项相乘，写入当前索引对应的输出。
- 第 2 行：结束 `operation` 多行字符串；逗号继续参数列表。
- 第 3 行：第四个位置参数 `name`，设置生成 kernel 的内部名称，便于编译缓存、分析和错误定位。
- 第 4 行：关键字参数。`("-std=c++11",)` 是单元素 tuple，尾随逗号不可省略；选项传给运行时编译器，使花括号初始化按 C++11 解析。
- 第 5 行：开始关键字参数 `loop_prep` 字符串。Python 行末反斜杠续接下一物理行。它计算 $A=2/(\sqrt{3a}\pi^{1/4})$，并使用 `M_PI`、`sqrt`、`pow`。该片段位于逐元素循环之前。
- 第 6 行：续行仍属于同一个 Python 字符串，计算共享的 $wsq=a^2$；末尾引号结束字符串，逗号结束该实参。前导空格只影响生成代码排版。
- 第 7 行：结束 `cp.ElementwiseKernel(...)` 调用；返回的 kernel 对象完成赋值。


### 4.3 `ricker` 函数体逐行解释

docstring 已在上一节逐行覆盖，这里解释签名与真正执行的函数体。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:165` 至 `:165`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:139` 至 `:139`。

```python
def ricker(points, a):
```

逐行对应说明：

- 第 1 行：`def` 定义函数；两个参数都是必填位置或关键字参数，没有默认值、类型注解和仅关键字限制；冒号开始缩进函数体。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:204` 至 `:204`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:177` 至 `:177`。

```python
    return _ricker_kernel(a, size=points)
```

逐行对应说明：

- 第 1 行：调用 kernel：位置实参 `a` 对应 `float64 a`，关键字 `size=points` 明确索引范围。CuPy 根据输出声明分配长度为 `points` 的 `float64` 设备数组，启动 kernel 后由 `return` 把数组返回。


## 5. 调用链与算法总结

### 5.1 从公开 API 到设备计算

```text
用户调用 cusignal.ricker(points, a)
    ↓ cusignal/__init__.py 的导出绑定
wavelets.wavelets.ricker(points, a)
    ↓ Python 函数体只有一条 return
_ricker_kernel(a, size=points)
    ↓ CuPy ElementwiseKernel 调用、必要时编译并缓存
loop_prep：计算一次 A 与 wsq=a²
    ↓ 对 i=0,...,points-1 逐元素并行
vec → xsq → mod、gauss → total
    ↓
返回 shape=(points,), dtype=float64 的 CuPy 数组
```

`cp.ElementwiseKernel` 的四个核心组成是输入声明、输出声明、逐元素操作体和 kernel 名称。
CuPy 官方文档说明 `i` 表示循环中的当前索引，`_ind.size()` 表示参与操作的总元素数；
`loop_prep` 被放在 kernel 函数顶部、逐元素循环之前。这里没有可广播的数组输入，因此调用时
显式传入 `size=points` 决定索引范围。

### 5.2 按源码执行顺序拆解

模块首次导入时：

1. 导入 CuPy；
2. 构造 `_ricker_kernel` 描述对象；
3. 定义 Python 函数 `ricker`；
4. 两级 `__init__.py` 把同一个函数对象导出到公开命名空间。

每次调用 `ricker(points, a)` 时：

1. Python 不创建坐标数组，也不做循环或参数验证；
2. 调用 `_ricker_kernel(a, size=points)`；
3. kernel 的循环前片段计算
   $$
   A=\frac{2}{\sqrt{3a}\pi^{1/4}},\qquad wsq=a^2;
   $$
4. 每个索引 $i$ 独立计算
   $$
   x_i=i-\frac{points-1}{2},\quad x_i^2,
   $$
   $$
   y_i=A\left(1-\frac{x_i^2}{a^2}\right)
   \exp\left(-\frac{x_i^2}{2a^2}\right);
   $$
5. CuPy 返回设备数组。

没有 Python 分支、循环、FFT、卷积、插值、临时坐标数组或 CPU reference 路径。

## 6. 数学映射、底层机制、边界与复杂度

### 6.1 数学公式与代码逐项映射

| 数学量 | 代码位置（基准） | 说明 |
| --- | --- | --- |
| 输出长度 $N$ | `ricker:177` 的 `size=points` | 决定 kernel 索引范围和返回 shape |
| $x_i=i-(N-1)/2$ | `_ricker_kernel:125` | 奇数长度含零点，偶数长度使用半整数中心坐标 |
| $x_i^2$ | `_ricker_kernel:126` | 供多项式项和高斯项复用 |
| $a^2$ | `_ricker_kernel:135` | 循环前预计算为 `wsq` |
| $1-x_i^2/a^2$ | `_ricker_kernel:127` | 高斯负二阶导数的多项式因子 |
| $e^{-x_i^2/(2a^2)}$ | `_ricker_kernel:128` | 高斯包络 |
| $2/(\sqrt{3a}\pi^{1/4})$ | `_ricker_kernel:134` | 连续单位能量归一化常数 |
| 最终 $y_i$ | `_ricker_kernel:130` | 三个因子的乘积 |

### 6.2 GPU 与 CuPy 底层机制

- `ElementwiseKernel` 管理输出分配、索引范围和逐元素并行；源文件没有手写 grid、block 或
  `threadIdx`。
- CuPy 首次调用某一兼容签名时运行时编译 kernel，并缓存编译结果；后续兼容调用可复用缓存。
- 每个逻辑索引只写一个 `total`，不同索引之间没有数据依赖、同步、原子操作或共享内存。
- `loop_prep` 中的 $A$ 和 $a^2$ 对整个 kernel 调用相同，放在逐元素循环前可减少重复计算。
- 输出位于 GPU；只有 docstring 示例的 `cp.asnumpy(vec2)` 才发生设备到主机的数据复制。

CuPy 接口语义参考：[CuPy 11.6 `ElementwiseKernel` 官方文档](https://docs.cupy.dev/en/v11.6.0/reference/generated/cupy.ElementwiseKernel.html)
以及 [CuPy kernel 用户指南](https://docs.cupy.dev/en/v14.1.1/user_guide/kernel.html)。访问日期：2026-08-09。

### 6.3 dtype 与 shape

- `a` 的 kernel 声明是 `float64`，中间变量显式为 C++ `double`；
- 输出 `total` 固定为 `float64`，不会根据 Python `a` 的 dtype 产生 `float32` 输出；
- `size=points` 创建一维输出 `(points,)`；
- 函数没有 `axis`、批量维度或广播数组输入；每次只生成一条小波。

### 6.4 边界、异常和数值稳定性

| 情况 | 源码行为与风险 |
| --- | --- |
| $a>0$、`points` 足够大 | 对有效公式做正常有限采样 |
| $a=0$ | `sqrt(3*a)`、`xsq/wsq` 出现除零/无效浮点计算；Python 层未拦截 |
| $a<0$ | `sqrt(3*a)` 在实数 `double` 中无有效结果；不属于数学定义域 |
| `points` 为奇数 | 中央索引的 `vec=0`，精确采到理论中心峰 $A$ |
| `points` 为偶数 | 两个中央样本的 `vec=±0.5`，数组仍对称但不含 $x=0$ |
| `points` 相对 $a$ 太小 | 高斯尾部被截断，有限数组的离散和与能量偏离连续理论 |
| $|x|/a$ 很大 | 指数可能下溢到 0；远尾本来就趋近 0，通常是良性下溢 |
| $a$ 极小 | `xsq/wsq` 可能很大，除法和归一化项可能出现极端值；源码无稳定化分支 |
| 不合法的 `points` | 由 CuPy 对 `size` 的要求处理；`ricker` 没有自定义错误消息 |

源码没有显式 `if`、`raise`、有限值检查或离散归一化修正。不能把连续理论的“单位能量、
零均值”误解成任意截断数组都精确满足相应离散等式。

### 6.5 时间和空间复杂度

令 $N=points$：

- 算术工作量：$O(N)$；每个输出包含常数次加减乘除和一次 `exp`；
- 并行深度：理想情况下由 GPU 并行覆盖大量独立元素，但实际受 kernel 调度与硬件资源限制；
- 输出空间：$O(N)$；
- 算法额外空间：每个线程常数个局部标量，总体不创建额外长度为 $N$ 的坐标或临时数组；
- 首次调用还可能有运行时编译成本，不属于公式的 $O(N)$ 逐元素成本；
- 小 `points` 时 kernel 启动开销可能超过算术成本；大 `points` 时 `exp` 吞吐与输出写入是
  主要成本。`loop_prep` 已避免逐元素重复计算 `sqrt` 和 `pow`。

## 7. 与阶段一原理的对应及实现边界

| 阶段一原理 | Python/CuPy 落实位置 | 关系 |
| --- | --- | --- |
| 原理 1：高斯负二阶导数 | `mod`、`gauss`、`total`（基准 127–130） | 直接落实解析公式 |
| 原理 2：单位能量尺度族 | `A`（基准 134） | 使用 $a^{-1/2}$ 归一化 |
| 原理 2：零均值与消失矩 | 没有单独求和或约束语句 | 由连续解析形状隐含；有限采样只近似保留 |
| 原理 3：峰值频率与宽度关系 | 输入只接受 `a` | 不直接计算 $f_p$；用户可用 $f_p=1/(\sqrt2\pi a)$ 换算 |
| 连续到离散 | `vec` 与 `size=points` | 中心化、等间隔、有限截断采样 |

`ricker` 的 CPU/Python 层只是包装；核心数值计算发生在 CuPy 生成的 GPU kernel 中。
本阶段仍称它为“Python 源码算法”，是因为公开 API、kernel 描述和调用均由 Python/CuPy
表达；这不代表每个样本由 Python 解释器逐个计算。

## 8. 建议的源码阅读顺序

1. 先看基准 `wavelets.py:139` 的函数签名，确认两个输入都没有默认值。
2. 阅读完整 docstring，特别观察 `points`、`a`、返回 shape 和示例中的 `cp.asnumpy`。
3. 看基准 `wavelets.py:177`，确认 Python 函数体只有一次 kernel 调用。
4. 回到 `_ricker_kernel:121–136`，按 `in_params → out_params → operation → name →
   options → loop_prep` 的顺序识别 `ElementwiseKernel` 构造参数。
5. 先读 `loop_prep:134–135`，明确 $A$ 和 $a^2$ 在逐元素循环前计算。
6. 再读 `operation:125–130`，手工把 `vec/xsq/mod/gauss/total` 写回阶段一公式。
7. 最后看两个 `__init__.py` 的导出，理解为何一个实现函数能通过两个公开路径访问。

## 9. 阅读检查

1. 为什么 `_ricker_kernel` 没有数组输入时必须在调用处传 `size=points`？
2. `"float64 total"` 如何决定返回数组的 dtype？
3. `vec` 为什么使用 `(N-1)*0.5`，而不是 `N//2`？偶数长度时二者有什么差异？
4. `loop_prep` 与 `operation` 的执行频率有什么不同？
5. 为什么 `wsq` 和 `A` 可以对全部输出复用，而 `vec` 不可以？
6. 把基准 125–130 行逐项代入后，能否恢复完整 Ricker 公式？
7. docstring 中的 `A = 2/sqrt(3a)pi^1/4` 为什么要以 kernel 134 行消除排版歧义？
8. `cp.asnumpy` 是 `ricker` 核心算法的一部分吗？它发生了什么数据搬运？
9. 连续单位能量为什么不保证短离散数组的平方和严格为 1？
10. 这段实现为什么是 $O(N)$，它又为什么适合 GPU 逐元素并行？
11. 哪些语句属于数学算法，哪些属于导出、dtype、编译和 kernel 调度工程？
12. Python 层没有检查 $a>0$ 会带来哪些可预见的浮点结果或错误风险？

### 参考答案

1. `_ricker_kernel` 的输入只有标量 `a`，没有可供 CuPy 广播并推导长度的数组。基准
   `wavelets.py:177` 显式传入 `size=points`，CuPy 才能确定索引范围和输出长度。
2. 基准 `wavelets.py:123` 把输出参数声明为 `"float64 total"`；因此每个索引写一个
   `float64`，最终返回数组也固定为 `float64`，不随 `a` 的 Python dtype 改变。
3. 基准 `wavelets.py:125` 使用 $i-(N-1)/2$，能让任意 $N$ 的首尾坐标互为相反数。
   奇数 $N$ 含坐标 0；偶数 $N$ 的中央两点为 $-0.5$ 和 $0.5$。若用 `N//2`，偶数长度
   坐标会偏向一侧，破坏这一对称采样。
4. `loop_prep`（基准 134–135 行）位于 ElementwiseKernel 的逐元素循环之前，每次 kernel
   调用计算一次；`operation`（基准 125–130 行）对每个输出索引执行一次。
5. `A` 和 `wsq=a*a` 只依赖整次调用共享的宽度 `a`，可以复用；`vec` 依赖当前索引 `i`，
   每个输出都不同，必须逐元素计算。
6. 基准 125–128 行依次给出 $x_i$、$x_i^2$、$1-x_i^2/a^2$ 和
   $e^{-x_i^2/(2a^2)}$，130 行乘上 `A`，完整恢复
   $$y_i=A(1-x_i^2/a^2)e^{-x_i^2/(2a^2)}.$$
7. docstring 的纯文本缺少清晰括号；基准 134 行实际代码是
   `2.0 / (sqrt(3*a) * pow(M_PI, 0.25))`，所以无歧义地对应
   $2/(\sqrt{3a}\pi^{1/4})$。
8. 不是。`ricker` 在基准 177 行已经返回 GPU 数组；示例中的 `cp.asnumpy(vec2)` 只为
   Matplotlib 绘图而执行 D2H 复制，不参与小波生成。
9. 单位能量结论是全实轴连续积分；有限数组同时进行了等间隔采样和尾部截断，离散平方和
   一般只是该积分的近似，`points` 太小时误差更明显。
10. 每个 $i$ 只做常数次算术和一次 `exp`，总工作量为 $O(N)$；各索引没有数据依赖，
    因而适合 ElementwiseKernel 把不同元素交给不同 GPU 线程。
11. `vec/xsq/mod/gauss/total` 与 `A/wsq` 落实数学公式；两个 `__init__.py` 只做名称导出，
    `float64` 声明选择 dtype，`options`、kernel 名称和 `size` 属于编译/调度工程。
12. $a=0$ 会使 `sqrt(3*a)`、`xsq/wsq` 出现除零或无效值；$a<0$ 会使实数 `sqrt`
    无定义。Python 函数没有 `if`/`raise`，所以不能依赖它产生清晰的领域错误消息，结果由
    CuPy 参数处理与设备浮点规则决定。
