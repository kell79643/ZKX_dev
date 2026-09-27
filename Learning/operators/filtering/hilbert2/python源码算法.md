# hilbert2 Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开导入路径

```python
import cusignal
xa = cusignal.hilbert2(x, N=None)
```

导出链如下：

1. `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py` 定义 `_hilbert2_kernel` 和 `hilbert2`。
2. `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py:27`（只读基准第 23 行）导出为 `cusignal.filtering.hilbert2`。
3. `Learning/cusignal-23.08.00/python/cusignal/__init__.py:53`（只读基准第 48 行）再次导出为 `cusignal.hilbert2`。

学习副本加入注释后，`_hilbert2_kernel` 从第 989 行开始、`hilbert2` 从第 1052 行开始；只读基准中的原始起始行分别为 895、932。下文完整源码摘录始终使用只读基准原文，避免把学习注释误当作原始实现。

### 1.2 函数签名

```python
def hilbert2(x, N=None):
```

| 项目 | 语义 |
| --- | --- |
| `x` | array-like 输入。经 `cp.atleast_2d` 转为至少二维的 CuPy 数组；最终只允许二维实数组。标量变为 `(1,1)`，一维输入变为 `(1,L)`。|
| `N=None` | 二维 FFT 的 shape。省略时取 `x.shape`；Python `int` 会扩展为 `(N,N)`；其他对象必须可取 `len(N)` 且长度为 2、元素均大于 0。|
| 返回值 | 复数 CuPy `ndarray`，正常数学意图下 shape 为 `(N[0],N[1])`。|
| dtype | `fft2`/`ifft2` 产生复 dtype；具体精度由 CuPy FFT 对输入 dtype 的规则决定。掩膜 dtype 由 `cp.result_type(x[0].dtype, x[1].dtype)` 得到。|

源码没有显式验证 tuple 中每项都是整数；是否接受浮点等异常值最终还受 `cp.fft.fft2` 的参数检查约束。

## 2. 当前算子的完整相关源码

### 2.1 模块导出语句

只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:23`：

```python
    hilbert2,
```

只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:48`：

```python
    hilbert2,
```

这两行都位于各自的 `from ... import (` 列表内，第一行建立子包导出，第二行建立顶层导出。

### 2.2 直接 GPU helper：`_hilbert2_kernel`

只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:895`：

```python
_hilbert2_kernel = cp.ElementwiseKernel(
    "",
    "T h1, T h2",
    """
    if ( !odd ) {
        if ( ( i == 0 ) || ( i == bend ) ) {
            h1 = 1.0;
            h2 = 1.0;
        } else if ( i > 0 && i < bend ) {
            h1 = 2.0;
            h2 = 2.0;
        } else {
            h1 = 0.0;
            h2 = 0.0;
        }
    } else {
        if ( i == 0 ) {
            h1 = 1.0;
            h2 = 1.0;
        } else if ( i > 0 && i < bend) {
            h1 = 2.0;
            h2 = 2.0;
        } else {
            h1 = 0.0;
            h2 = 0.0;
        }
    }
    """,
    "_hilbert2_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)
```

### 2.3 `hilbert2` 完整定义

只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:932`：

```python
def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    x = cp.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or cp.any(cp.asarray(N) <= 0):
        raise ValueError(
            "When given as a tuple, N must hold exactly two positive integers"
        )

    Xf = cp.fft.fft2(x, N, axes=(0, 1))

    elements_dtype = cp.result_type(x[0].dtype, x[1].dtype)
    h1 = cp.empty((N[1],), dtype=elements_dtype)
    h2 = cp.empty((N[1],), dtype=elements_dtype)
    _hilbert2_kernel(h1, h2)

    h = h1[:, cp.newaxis] * h2[cp.newaxis, :]
    k = x.ndim
    while k > 2:
        h = h[:, cp.newaxis]
        k -= 1
    x = cp.fft.ifft2(Xf * h, axes=(0, 1))
    return x
```

## 3. docstring 逐行翻译与解释

空行仅用于排版，按规则不单列。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1083` 至 `:1084`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:933` 至 `:934`。

```python
    """
    Compute the '2-D' analytic signal of `x`
```

逐行对应说明：

- 第 1 行：开始 Python 三引号 docstring；它会成为 `hilbert2.__doc__`。
- 第 2 行：“计算 `x` 的‘二维’解析信号”。引号暗示二维解析信号存在多种定义。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1086` 至 `:1091`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:936` 至 `:941`。

```python
    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy/SciPy docstring 风格的标题下划线。
- 第 3 行：`x` 接受能转为数组的对象，不限定调用者必须预先传 CuPy 数组。
- 第 4 行：声称 `x` 是二维信号数据；函数实际会先把低维输入提升到二维。
- 第 5 行：`N` 可省略，可为一个整数或两个整数的 tuple。
- 第 6 行：`N` 决定两个轴的 Fourier 分量数，默认等于输入 shape。双反引号是 reStructuredText 行内代码。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1093` 至 `:1096`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:943` 至 `:946`。

```python
    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：返回值标题下划线。
- 第 3 行：返回对象名写作 `xa`，类型为数组；实现实际返回 CuPy 数组。
- 第 4 行：在第 0、1 轴上构造 `x` 的解析信号。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1098` 至 `:1101`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:948` 至 `:951`。

```python
    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal
```

逐行对应说明：

- 第 1 行：参考资料章节标题。
- 第 2 行：参考资料标题下划线。
- 第 3 行：reStructuredText 编号引用 `[1]` 的首行。
- 第 4 行：引用的网页地址；只说明一般解析信号，没有完整刻画该实现的二维定义。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1103` 至 `:1103`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:953` 至 `:953`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束 docstring。


原 docstring 没有示例，因此不存在被省略的 docstring 示例。

## 4. 源代码逐行解释

### 4.1 两级公开导出

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py:27` 至 `:27`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:23` 至 `:23`。

```python
    hilbert2,
```

逐行对应说明：

- 第 1 行：这是父级 `from .filtering import (` 中的一个名称；尾逗号允许多行导入列表。它把定义文件中的符号绑定到 `cusignal.filtering` 命名空间。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:59` 至 `:59`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:48` 至 `:48`。

```python
    hilbert2,
```

逐行对应说明：

- 第 1 行：这是顶层 `from cusignal.filtering import (` 列表的一项，使调用者可以直接写 `cusignal.hilbert2`。


### 4.2 `_hilbert2_kernel` 每一行

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1019` 至 `:1024`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:895` 至 `:898`。

```python
_hilbert2_kernel = cp.ElementwiseKernel(
    "",
    "T h1, T h2",
    """
```

逐行对应说明：

- 第 1 行：调用 CuPy 工厂创建 elementwise kernel，并把返回的可调用对象赋给模块变量。左括号开始多行实参列表。
- 第 2 行：第一个实参是输入参数声明；空串表示无输入数组。
- 第 3 行：第二个实参声明两个输出。`T` 是 CuPy 类型占位符，`h1`、`h2` 是 kernel 内可写的当前元素。
- 第 4 行：开始包含 C/C++ kernel operation 的多行 Python 字符串。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1026` 至 `:1026`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:899` 至 `:899`。

```python
    if ( !odd ) {
```

逐行对应说明：

- 第 1 行：C++ 条件语句；`!` 逻辑取反，所以进入偶数长度分支。`odd` 来自 `loop_prep`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1028` 至 `:1047`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:900` 至 `:910`。

```python
        if ( ( i == 0 ) || ( i == bend ) ) {
            h1 = 1.0;
            h2 = 1.0;
        } else if ( i > 0 && i < bend ) {
            h1 = 2.0;
            h2 = 2.0;
        } else {
            h1 = 0.0;
            h2 = 0.0;
        }
    } else {
```

逐行对应说明：

- 第 1 行：
- 第 2 行：当前索引的第一个输出写 1；分号结束 C++ 语句。
- 第 3 行：第二个输出同样写 1。
- 第 4 行：`&&` 是逻辑与；索引 `1..N/2-1` 是严格正频率区。
- 第 5 行：第一个掩膜的正频权重写 2。
- 第 6 行：第二个掩膜的正频权重写 2。
- 第 7 行：处理既非 DC/Nyquist、也非正频的剩余索引，即负频区。
- 第 8 行：第一个掩膜负频清零。
- 第 9 行：第二个掩膜负频清零。
- 第 10 行：结束偶数长度内部条件。
- 第 11 行：`odd` 为真时进入奇数长度分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1049` 至 `:1068`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:911` 至 `:922`。

```python
        if ( i == 0 ) {
            h1 = 1.0;
            h2 = 1.0;
        } else if ( i > 0 && i < bend) {
            h1 = 2.0;
            h2 = 2.0;
        } else {
            h1 = 0.0;
            h2 = 0.0;
        }
    }
    """,
```

逐行对应说明：

- 第 1 行：奇数长度没有独立 Nyquist 箱，只有 DC 单独处理。
- 第 2 行：第一个掩膜 DC 权重为 1。
- 第 3 行：第二个掩膜 DC 权重为 1。
- 第 4 行：当 `bend=(N+1)/2` 时，`1..(N-1)/2` 为正频箱。`bend)` 前少一个空格不影响 C++ 语法。
- 第 5 行：第一个掩膜正频乘 2。
- 第 6 行：第二个掩膜正频乘 2。
- 第 7 行：奇数长度剩余索引是负频箱。
- 第 8 行：第一个掩膜负频清零。
- 第 9 行：第二个掩膜负频清零。
- 第 10 行：结束奇数长度内部条件。
- 第 11 行：结束最外层奇偶条件。
- 第 12 行：结束 operation 多行字符串，并用逗号分隔下一个实参。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1070` 至 `:1078`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:923` 至 `:929`。

```python
    "_hilbert2_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)
```

逐行对应说明：

- 第 1 行：生成 kernel 使用的内部名称，便于编译缓存和诊断。
- 第 2 行：关键字实参；单元素 tuple 必须有尾逗号，要求后端用 C++11 编译。
- 第 3 行：开始预循环代码字符串。`_ind.size()` 是 elementwise 迭代总元素数，按位与 1 得到奇偶性；反斜杠续接下一物理行。
- 第 4 行：定义常量整数 `bend`；`?:` 是 C++ 条件运算符，反斜杠继续字符串。
- 第 5 行：奇数时计算 `(N+1)/2`，`static_cast<int>` 显式转整数；冒号引出偶数表达式。
- 第 6 行：偶数时计算 `N/2`；分号结束 C++ 声明，双引号结束 Python 字符串。
- 第 7 行：结束 `cp.ElementwiseKernel(...)` 调用。


关键执行事实：kernel 的迭代长度由输出数组广播结果决定。这里两个输出长度相同，因此一次调用可同时填充二者；但也正因为源码把二者都分配成 `N[1]`，kernel 没有机会为第 0 轴生成长度 `N[0]` 的正确掩膜。

### 4.3 `hilbert2` 函数体每一行

docstring 已在上一节逐行解释，本表从签名和可执行语句继续，仍保持原始顺序。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1082` 至 `:1082`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:932` 至 `:932`。

```python
def hilbert2(x, N=None):
```

逐行对应说明：

- 第 1 行：`def` 定义函数；`x` 无默认值，`N` 默认是 `None`；冒号开始缩进函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1105` 至 `:1105`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:954` 至 `:954`。

```python
    x = cp.atleast_2d(x)
```

逐行对应说明：

- 第 1 行：调用 CuPy 数组例程并重新绑定 `x`。0-D 变 `(1,1)`，1-D 变 `(1,L)`，二维保持；通常也完成 array-like 到 CuPy 数组的转换。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1107` 至 `:1109`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:955` 至 `:956`。

```python
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
```

逐行对应说明：

- 第 1 行：若提升后的维数大于 2，进入错误分支；没有检查 `<2`，因为前一行已经保证至少二维。
- 第 2 行：构造并抛出 `ValueError`，函数立即终止。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1111` 至 `:1113`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:957` 至 `:958`。

```python
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
```

逐行对应说明：

- 第 1 行：判断数组 dtype 是否为复数类型，而不是逐元素检查虚部是否恰为零。
- 第 2 行：复 dtype 输入被拒绝，即便所有虚部数值都是 0。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1115` 至 `:1117`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:959` 至 `:960`。

```python
    if N is None:
        N = x.shape
```

逐行对应说明：

- 第 1 行：用对象身份判断是否未提供 `N`。
- 第 2 行：默认把两维 shape tuple 赋给 `N`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1119` 至 `:1119`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:961` 至 `:961`。

```python
    elif isinstance(N, int):
```

逐行对应说明：

- 第 1 行：只有 Python `int` 进入此分支；某些 NumPy/CuPy 整数标量未必满足该判断。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1121` 至 `:1125`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:962` 至 `:964`。

```python
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
```

逐行对应说明：

- 第 1 行：标量 FFT 长度必须大于 0。
- 第 2 行：非正标量触发异常。
- 第 3 行：把标量复制成二元 tuple，要求方形 FFT 输出。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1127` 至 `:1132`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:965` 至 `:968`。

```python
    elif len(N) != 2 or cp.any(cp.asarray(N) <= 0):
        raise ValueError(
            "When given as a tuple, N must hold exactly two positive integers"
        )
```

逐行对应说明：

- 第 1 行：利用 `or` 短路：长度不是 2 时不再执行右侧；否则转成 CuPy 数组、逐元素比较 `<=0`，再由 `cp.any` 判断是否存在非正元素。
- 第 2 行：开始跨行构造 `ValueError`。
- 第 3 行：唯一位置实参，是完整错误消息。
- 第 4 行：结束异常构造并由第 966 行的 `raise` 抛出。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1135` 至 `:1135`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:970` 至 `:970`。

```python
    Xf = cp.fft.fft2(x, N, axes=(0, 1))
```

逐行对应说明：

- 第 1 行：计算二维 FFT。第二个位置参数 `N` 对应 FFT shape；`axes` 明确指定两个轴。结果绑定到 `Xf`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1138` 至 `:1144`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:972` 至 `:975`。

```python
    elements_dtype = cp.result_type(x[0].dtype, x[1].dtype)
    h1 = cp.empty((N[1],), dtype=elements_dtype)
    h2 = cp.empty((N[1],), dtype=elements_dtype)
    _hilbert2_kernel(h1, h2)
```

逐行对应说明：

- 第 1 行：取输入前两行的 dtype 做类型提升。两者本来来自同一数组，dtype 必然相同，故该写法冗余；第 0 轴长度为 1 时 `x[1]` 越界。
- 第 2 行：分配未初始化一维 GPU 数组。`(N[1],)` 中尾逗号创建一元 shape tuple。按算法它应对应第 0 轴，却错误使用了 `N[1]`。
- 第 3 行：分配对应第 1 轴的一维掩膜，长度 `N[1]` 合理。
- 第 4 行：调用刚创建的 CuPy kernel；两个实参作为输出数组被逐元素填满。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1147` 至 `:1149`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:977` 至 `:978`。

```python
    h = h1[:, cp.newaxis] * h2[cp.newaxis, :]
    k = x.ndim
```

逐行对应说明：

- 第 1 行：`:` 取全部元素；`cp.newaxis` 等价于 `None`。前者 shape 变 `(L,1)`，后者变 `(1,L)`，广播相乘得到外积 `(L,L)`。
- 第 2 行：保存输入维数。经过前面的约束，`k` 此处必为 2。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1151` 至 `:1157`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:979` 至 `:982`。

```python
    while k > 2:
        h = h[:, cp.newaxis]
        k -= 1
    x = cp.fft.ifft2(Xf * h, axes=(0, 1))
```

逐行对应说明：

- 第 1 行：循环条件永远为假，因此 980–981 行是不可达的遗留/通用化代码。
- 第 2 行：若可达，会在第二维位置插入新轴；但索引写法对一般高维掩膜的意图也不够清楚。
- 第 3 行：增强赋值，等价于 `k = k - 1`。
- 第 4 行：先逐元素/广播计算 `Xf*h`，再沿两轴做二维逆 FFT；结果覆盖局部变量 `x`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1159` 至 `:1159`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:983` 至 `:983`。

```python
    return x
```

逐行对应说明：

- 第 1 行：把复解析信号返回调用者并结束函数。


## 5. 调用链与算法总结

```text
cusignal.hilbert2
  → cusignal.filtering.hilbert2
    → cp.atleast_2d + 参数检查
    → cp.fft.fft2(x, N, axes=(0,1))
    → cp.empty × 2
    → _hilbert2_kernel(h1, h2)
    → h1[:,None] * h2[None,:]
    → cp.fft.ifft2(Xf * h, axes=(0,1))
    → 复数输出
```

### 5.1 按执行顺序概括

1. 规范输入维数并拒绝复数。
2. 把 `N` 规范成二轴 FFT 尺寸。
3. 二维 FFT 得到 $X[p,q]$。
4. GPU elementwise kernel 生成权重属于 `{0,1,2}` 的一维 Hilbert 掩膜。
5. 外积生成二维权重 `{0,1,2,4}`。
6. 频谱逐点加权并做二维 IFFT。

没有 Python 循环参与实际样本计算；并行工作主要由 CuPy FFT 和 elementwise kernel 完成。源码中的 `while` 因维数限制不执行。

## 6. 数学公式与代码映射

| 数学步骤 | 公式 | 代码位置（基准行） |
| --- | --- | --- |
| 二维 DFT | $X[p,q]=\sum_k\sum_l x[k,l]e^{-j2\pi kp/N_0}e^{-j2\pi lq/N_1}$ | `filtering.py:970` |
| 一维权重 | DC/Nyquist 为 1，正频为 2，负频为 0 | `_hilbert2_kernel`，`filtering.py:899-921` |
| 可分离二维掩膜 | $h[p,q]=h_{N_0}[p]h_{N_1}[q]$ | `filtering.py:977` |
| 单象限频谱 | $X_a[p,q]=X[p,q]h[p,q]$ | `filtering.py:982` 中的 `Xf * h` |
| 二维逆 DFT | $x_a=\operatorname{IDFT}_2(X_a)$ | `filtering.py:982` |

源码意图与阶段一 single-orthant 公式一致，但第 973 行的尺寸错误使 $h_{N_0}$ 实际被构造成 $h_{N_1}$。方阵时二者长度相同，错误被隐藏；非方阵时通常在 `Xf*h` 处出现 shape 不兼容，或无法代表理论掩膜。

## 7. 边界、异常、dtype 与数值稳定性

### 7.1 已实现检查

- 三维及以上：`ValueError("x must be 2-D.")`。
- 复 dtype：`ValueError("x must be real.")`。
- 标量 `N<=0`：`ValueError("N must be positive.")`。
- 序列 `N` 长度不是 2 或含非正元素：`ValueError`。

### 7.2 未完整覆盖的边界

- `cp.atleast_2d` 接受标量和一维输入，但随后 `x[1]` 对 shape `(1,L)` 越界，接口行为前后矛盾。
- 空数组也可能在 `x[0]` 或 `x[1]` 处失败。
- `N` 文档要求整数，但源码只检查正值，没有显式逐元素整数检查。
- 非方形 `N` 因 `h1` 长度错误存在确定的静态 shape 风险。
- FFT 假设周期延拓；不连续边界会产生泄漏与振铃，源码没有加窗或 padding 策略。

### 7.3 dtype

`elements_dtype` 实际等于输入数组 dtype，因为同一数组的各行共享 dtype。掩膜只写 `0.0/1.0/2.0`。FFT 输出为复数，最终结果也为复数。源码没有显式控制 FFT 精度，也没有额外归一化；逆 FFT 的归一化遵循 CuPy 默认约定。

### 7.4 数值稳定性

算法只做 FFT、有限权重乘法和 IFFT，没有递归反馈，通常不存在 IIR 那样的累积不稳定。但浮点舍入会让理论上应为零的分量留下小残差；边界泄漏是模型/周期延拓问题，不应误判为浮点不稳定。

## 8. 复杂度、内存与性能瓶颈

设 $M=N_0N_1$。

- 二维 FFT：时间复杂度约 $O(M(\log N_0+\log N_1))=O(M\log M)$。
- 掩膜生成：理论应为 $O(N_0+N_1)$；当前方阵实现一次 kernel 同时写两个长度 $N_1$ 的数组，工作量 $O(N_1)$。
- 外积与频谱乘法：各 $O(M)$。
- 二维 IFFT：$O(M\log M)$。
- 主要额外存储：复频谱 `Xf`、二维掩膜 `h`、复输出和两个一维掩膜，总体 $O(M)$。

主要瓶颈是二维 FFT/IFFT 及其全局内存流量。显式物化二维 `h` 还需要一个 $O(M)$ 数组；若实现允许按两个一维向量广播直接乘频谱，可减少掩膜存储。首次调用 `ElementwiseKernel` 还可能包含即时编译开销。

## 9. 学习副本注释与完整性复核

本阶段只修改了：

- `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py`
- `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py`
- `Learning/cusignal-23.08.00/python/cusignal/__init__.py`

新增内容均为独立的 `# <学习注释：...>` 或 kernel 字符串内的 `// <学习注释：...>`。复核时忽略所有合法学习注释行，三个学习副本均与 `ZKX/cusignal-23.08.00` 对应只读基准逐行一致，差异数均为 0。只读基准未修改。

本阶段依照学习规则没有运行 Python、CUDA 或测试；文中风险来自静态代码和 shape 推导，不冒充运行结果。

## 10. 建议阅读顺序与检查问题

1. 先读基准 970、977、982 行，抓住“FFT→外积掩膜→IFFT”主链。
2. 再读 899–921 行，分别手算长度 5 和长度 6 时 kernel 产生的掩膜。
3. 回到 954–968 行，列出标量、一维、二维、三维、复数输入分别走哪个分支。
4. 对照 972–975 行，说明为什么 `x[1]` 和两个 `N[1]` 是风险点。
5. 最后读导出链，确认 `cusignal.hilbert2` 如何绑定到定义函数。

阅读后应能回答：

- `odd`、`bend`、`i` 分别来自哪里？
- 偶数长度的 Nyquist 为什么乘 1？
- `h1[:, cp.newaxis]` 与 `h2[cp.newaxis, :]` 各是什么 shape？
- 哪些代码属于数学算法，哪些只是 CuPy/GPU 工程实现？
- 为什么 `while k > 2` 在当前函数中不可达？
- 方阵输入为何会掩盖第 973 行的问题？

### 参考答案

1. `odd` 与 `bend` 在 `_hilbert2_kernel` 的 `loop_prep` 中由 `_ind.size()` 计算；`i` 是 CuPy `ElementwiseKernel` 自动提供的当前线性索引。见基准 899–928 行。
2. 偶数长度 Nyquist 箱位于 $N/2$，在 DFT 环上与自己的负频位置重合，没有另一个可删除后再补偿的共轭箱，所以权重为 1。对应基准 900–902 行。
3. 若两者长度为 $L$，`h1[:, cp.newaxis]` shape 为 $(L,1)$，`h2[cp.newaxis,:]` 为 $(1,L)$，广播外积为 $(L,L)$。见基准 977 行。
4. FFT、`{0,1,2}` 掩膜、外积、频谱乘法和 IFFT 属于数学算法；输入检查、dtype 推导、CuPy kernel 声明、广播索引和错误消息属于数组/GPU 工程实现。
5. `cp.atleast_2d` 保证至少二维，随后 `x.ndim>2` 被拒绝，因此执行到 978 行时 `k` 恒为 2，979 行条件恒假。
6. 方阵时 $N_0=N_1$，错误的 $(N_1,N_1)$ mask 恰好与 $(N_0,N_1)$ 频谱同形，所以形状错误被隐藏；非方阵通常在基准 982 行的 `Xf*h` 暴露。

