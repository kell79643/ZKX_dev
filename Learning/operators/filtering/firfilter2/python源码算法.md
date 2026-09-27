# firfilter2 Python 源码算法

## 1. 代码定位与接口概览

公开调用路径为：

```python
import cusignal
y = cusignal.firfilter2(b, x, axis=-1, padtype="odd", padlen=None,
                        method="pad", irlen=None)
```

| 层级 | 学习副本定位 | 只读基准定位 | 符号 |
| --- | --- | --- | --- |
| 顶层公开导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:47` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:44` | `firfilter2` |
| filtering 子包导出 | `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py:20` | `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:19` | `firfilter2` |
| 算子定义 | `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:405-521` | `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:404-506` | `firfilter2` |
| 单次 FIR | 同文件 `109-201` | 同路径基准 `109-201` | `firfilter` |
| FIR 稳态初值包装 | 同文件 `320-360` | 同路径基准 `320-360` | `firfilter_zi` |
| 通用稳态初值 | 同文件 `247-318` | 同路径基准 `247-318` | `lfilter_zi` |
| padding 校验 | 同文件 `363-401` | 同路径基准 `363-401` | `_validate_pad` |
| 切片、反转、延拓 | `Learning/cusignal-23.08.00/python/cusignal/utils/arraytools.py:164-365` | `ZKX/cusignal-23.08.00/python/cusignal/utils/arraytools.py:164-365` | `_axis_slice`、`_axis_reverse`、`_odd_ext`、`_even_ext`、`_const_ext` |

新增学习注释使学习副本的主函数末行由基准 `506` 移到 `521`；本文的源码摘录始终采用只读基准原文。

### 参数、返回值、dtype 与 shape

| 名称 | 语义 |
| --- | --- |
| `b` | shape 为 `(M,)` 的 FIR 系数；入口用 `cp.atleast_1d` 处理，但真正的一维检查在 `firfilter` 中完成 |
| `x` | 至少一维的输入；可为 N 维，沿 `axis` 独立滤波 |
| `axis` | 滤波轴，默认 `-1`；输出 shape 与输入相同 |
| `padtype` | `"odd"`、`"even"`、`"constant"` 或 `None` |
| `padlen` | 每端延拓长度；`None` 实际取 `3 * len(b)`，必须满足 `x.shape[axis] > padlen` |
| `method` | 仅允许 `"pad"` 或 `"gust"`；本版本只实现 `"pad"` |
| `irlen` | 只为未实现的 `gust` 接口保留；padding 路径完全不读取它 |
| 返回 `y` | `cp.ndarray`，shape 与 `x` 相同；dtype 由 `cp.result_type(b, x, zi)` 决定并限制为浮点、复数或 object 类别 |

## 2. 当前算子的完整相关源码

### 2.1 公开导出语句

基准 `python/cusignal/filtering/__init__.py:9-29` 和顶层 `python/cusignal/__init__.py:38-61` 都从 `cusignal.filtering.filtering` 的括号导入列表中包含：

```python
    firfilter2,
```

这两行没有包装计算：第一行建立 `cusignal.filtering.firfilter2`，第二行建立 `cusignal.firfilter2`。

### 2.2 `firfilter2` 完整定义

以下为只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:404-506` 原文，签名、完整 docstring 和函数体均未省略：

```python
def firfilter2(b, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    """
    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.
    The function provides options for handling the edges of the signal.
    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.
    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.
    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.
    The option to use Gustaffson's method was added in scipy version 0.16.0.
    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.
    """
    b = cp.atleast_1d(b)
    x = cp.asarray(x)

    if method not in ["pad", "gust"]:
        raise ValueError("method must be 'pad' or 'gust'.")

    if method == "gust":
        raise NotImplementedError("gust method not supported yet")

    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=len(b))

    # Get the steady state of the filter's step response.
    zi = firfilter_zi(b)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = cp.reshape(zi, zi_shape)
    x0 = _axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = firfilter(b, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = _axis_slice(y, start=-1, axis=axis)
    (y, zf) = firfilter(b, _axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = _axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = _axis_slice(y, start=edge, stop=-edge, axis=axis)

    return cp.copy(y)
```

### 2.3 直接 helper 的完整计算语句

直接 helper 的长 docstring 主要重复接口说明；下列代码保留每个符号的完整可执行语句，定位到上一节表格所列的完整定义。阅读原文件时，docstring 也属于函数体的一部分。

```python
def firfilter(b, x, axis=-1, zi=None):
    b = cp.asarray(b)
    if b.ndim != 1:
        raise ValueError("object of too small depth for desired array")
    if x.ndim == 0:
        raise ValueError("x must be at least 1-D")
    inputs = [b, x]
    if zi is not None:
        zi = cp.asarray(zi)
        if zi.ndim != x.ndim:
            raise ValueError("object of too small depth for desired array")
        expected_shape = list(x.shape)
        expected_shape[axis] = b.shape[0] - 1
        expected_shape = tuple(expected_shape)
        if zi.shape != expected_shape:
            strides = zi.ndim * [None]
            if axis < 0:
                axis += zi.ndim
            for k in range(zi.ndim):
                if k == axis and zi.shape[k] == expected_shape[k]:
                    strides[k] = zi.strides[k]
                elif k != axis and zi.shape[k] == expected_shape[k]:
                    strides[k] = zi.strides[k]
                elif k != axis and zi.shape[k] == 1:
                    strides[k] = 0
                else:
                    raise ValueError(
                        "Unexpected shape for zi: expected "
                        "%s, found %s." % (expected_shape, zi.shape)
                    )
            zi = cp.lib.stride_tricks.as_strided(zi, expected_shape, strides)
        inputs.append(zi)
    dtype = cp.result_type(*inputs)
    if dtype.char not in "fdgFDGO":
        raise NotImplementedError("input type '%s' not supported" % dtype)
    b = cp.array(b, dtype=dtype)
    x = cp.array(x, dtype=dtype, copy=False)
    out_full = cp.apply_along_axis(lambda y: cp.convolve(b, y), axis, x)
    ind = out_full.ndim * [slice(None)]
    if zi is not None:
        ind[axis] = slice(zi.shape[axis])
        out_full[tuple(ind)] += zi
    ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
    out = out_full[tuple(ind)]
    if zi is None:
        return out
    else:
        ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
        zf = out_full[tuple(ind)]
        return out, zf

def firfilter_zi(b):
    return lfilter_zi(b, np.float32(1.0))

def lfilter_zi(b, a):
    b = cp.atleast_1d(b)
    a = cp.atleast_1d(a)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Numerator b and denominator a must be 1-D.")
    if a.size == 0:
        raise ValueError("No coefficients supplied to lfilter.")
    while a.size > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    n = max(a.size, b.size)
    if a.size < n:
        a = cp.r_[a, cp.zeros(n - a.size)]
    elif b.size < n:
        b = cp.r_[b, cp.zeros(n - b.size)]
    IminusA = cp.eye(n - 1) - companion(a).T
    B = b[1:] - a[1:] * b[0]
    zi = cp.linalg.solve(IminusA, B)
    return zi

def _validate_pad(padtype, padlen, x, axis, ntaps):
    if padtype not in ["even", "odd", "constant", None]:
        raise ValueError(
            ("Unknown value '%s' given to padtype.  padtype "
             "must be 'even', 'odd', 'constant', or None.") % padtype
        )
    if padtype is None:
        padlen = 0
    if padlen is None:
        edge = ntaps * 3
    else:
        edge = padlen
    if x.shape[axis] <= edge:
        raise ValueError(
            "The length of the input vector x must be greater "
            "than padlen, which is %d." % edge
        )
    if padtype is not None and edge > 0:
        if padtype == "even":
            ext = _even_ext(x, edge, axis=axis)
        elif padtype == "odd":
            ext = _odd_ext(x, edge, axis=axis)
        else:
            ext = _const_ext(x, edge, axis=axis)
    else:
        ext = x
    return edge, ext

def _axis_slice(a, start=None, stop=None, step=None, axis=-1):
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]
    return b

def _axis_reverse(a, axis=-1):
    return _axis_slice(a, step=-1, axis=axis)
```

三种延拓 helper 的核心构造语句为：

```python
# _odd_ext
left_end = _axis_slice(x, start=0, stop=1, axis=axis)
left_ext = _axis_slice(x, start=n, stop=0, step=-1, axis=axis)
right_end = _axis_slice(x, start=-1, axis=axis)
right_ext = _axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
ext = cp.concatenate((2 * left_end - left_ext, x,
                      2 * right_end - right_ext), axis=axis)

# _even_ext
left_ext = _axis_slice(x, start=n, stop=0, step=-1, axis=axis)
right_ext = _axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
ext = cp.concatenate((left_ext, x, right_ext), axis=axis)

# _const_ext
left_end = _axis_slice(x, start=0, stop=1, axis=axis)
ones_shape = [1] * x.ndim
ones_shape[axis] = n
ones = cp.ones(ones_shape, dtype=x.dtype)
left_ext = ones * left_end
right_end = _axis_slice(x, start=-1, axis=axis)
right_ext = ones * right_end
ext = cp.concatenate((left_ext, x, right_ext), axis=axis)
```

## 3. docstring 逐行翻译与解释

下表的基准行号与第 2.2 节逐行对应；相邻但语义不可分割的续行仍逐行列出。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:456` 至 `:464`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:405` 至 `:413`。

```python
    """
    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.
    The function provides options for handling the edges of the signal.
    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.
```

逐行对应说明：

- 第 1 行：开始函数 docstring；运行时成为 `firfilter2.__doc__`。
- 第 2 行：对信号正向、反向应用数字滤波器。
- 第 3 行：同一线性滤波器共执行两次。
- 第 4 行：第二次沿反方向执行，组合相位为零。
- 第 5 行：理想组合阶数是原滤波器的两倍。
- 第 6 行：API 提供有限数组端点处理选项。
- 第 7 行：这是从通用 SciPy `filtfilt` 继承的提示。
- 第 8 行：建议通用 IIR 使用二阶节；但 `firfilter2` 只有 FIR 系数 `b`。
- 第 9 行：二阶节可减小高阶 IIR 数值问题；对本 FIR 专用函数并非核心理由。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:465` 至 `:476`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:414` 至 `:425`。

```python
    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy docstring 标题下划线。
- 第 3 行：`b` 应是一维、长度 N 的类数组。
- 第 4 行：FIR 分子系数，也就是冲激响应抽头。
- 第 5 行：输入可由任意兼容类数组给出。
- 第 6 行：它是实际待滤波的数据。
- 第 7 行：指定整数轴，且为可选参数。
- 第 8 行：仅沿该轴形成一维序列。
- 第 9 行：默认最后一轴。
- 第 10 行：延拓方式可为字符串或 `None`。
- 第 11 行：四个允许值：odd/even/constant/None。
- 第 12 行：它决定传给 FIR 的扩展数据怎样构造。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:477` 至 `:488`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:426` 至 `:437`。

```python
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
```

逐行对应说明：

- 第 1 行：`None` 禁止延拓。
- 第 2 行：默认奇延拓。
- 第 3 行：每端延拓样本数。
- 第 4 行：左、右各扩展同样数量。
- 第 5 行：延拓发生在两次滤波前。
- 第 6 行：必须短于目标轴；0 表示不延拓。
- 第 7 行：文档沿用通用函数写法；本实现实际为 `3*len(b)`。
- 第 8 行：端点算法选择。
- 第 9 行：设计上只有两个方法名。
- 第 10 行：pad 分支会延拓。
- 第 11 行：延拓由两参数控制，`irlen` 无效。
- 第 12 行：文档说明 Gustafsson 方法。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:489` 至 `:494`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:438` 至 `:443`。

```python
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.
```

逐行对应说明：

- 第 1 行：若实现 gust，本应忽略 padding 参数。
- 第 2 行：冲激响应截断长度。
- 第 3 行：仅 gust 有意义。
- 第 4 行：`None` 表示不截断冲激响应。
- 第 5 行：对长信号可用截断减少工作量。
- 第 6 行：仍只是接口文档；本版本 gust 未实现。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:495` 至 `:498`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:444` 至 `:447`。

```python
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
```

逐行对应说明：

- 第 1 行：返回值章节。
- 第 2 行：NumPy docstring 格式。
- 第 3 行：返回数组。
- 第 4 行：返回 shape 与输入一致。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:499` 至 `:510`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:448` 至 `:459`。

```python
    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.
    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
```

逐行对应说明：

- 第 1 行：算法说明章节。
- 第 2 行：标题格式。
- 第 3 行：pad 沿选定轴工作。
- 第 4 行：三种非空延拓方式。
- 第 5 行：odd/even 分别采用奇、偶端点对称。
- 第 6 行：常量延拓复制端点值。
- 第 7 行：正向和反向都设置初始状态。
- 第 8 行：初值来自单位阶跃稳态状态。
- 第 9 行：再按每次输入的端点值缩放。
- 第 10 行：介绍另一种初值算法。
- 第 11 行：它优化两方向的初始状态。
- 第 12 行：目标是先正后反与先反后正相等。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:511` 至 `:512`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:460` 至 `:461`。

```python
    filter.
    The option to use Gustaffson's method was added in scipy version 0.16.0.
```

逐行对应说明：

- 第 1 行：完成上一行跨行句子。
- 第 2 行：说明 SciPy 历史，不代表 cuSignal 已实现。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:513` 至 `:518`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:462` 至 `:467`。

```python
    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.
    """
```

逐行对应说明：

- 第 1 行：参考文献章节。
- 第 2 行：标题格式。
- 第 3 行：文献编号与作者、标题开始。
- 第 4 行：期刊、卷和页码；源码写 Vol.46，但论文实际为 IEEE TSP 44(4)，应以论文为准。
- 第 5 行：发表年份 1996。
- 第 6 行：结束 docstring。


该 docstring **没有示例代码**，因此不存在被遗漏的 docstring 示例。

## 4. 源代码逐行解释

### 4.1 `firfilter2` 函数体

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:455` 至 `:455`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:404` 至 `:404`。

```python
def firfilter2(b, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
```

逐行对应说明：

- 第 1 行：`def` 创建函数；七个形参中后五个有默认值。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:520` 至 `:522`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:468` 至 `:469`。

```python
    b = cp.atleast_1d(b)
    x = cp.asarray(x)
```

逐行对应说明：

- 第 1 行：调用 CuPy API；标量会变成长度 1 数组。
- 第 2 行：转成 GPU 数组，已有兼容数组可复用存储。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:525` 至 `:526`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:471` 至 `:472`。

```python
    if method not in ["pad", "gust"]:
        raise ValueError("method must be 'pad' or 'gust'.")
```

逐行对应说明：

- 第 1 行：成员测试；不在二元素列表时进入分支。
- 第 2 行：参数值不合法时立即终止。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:529` 至 `:530`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:474` 至 `:475`。

```python
    if method == "gust":
        raise NotImplementedError("gust method not supported yet")
```

逐行对应说明：

- 第 1 行：单独识别保留但未实现的分支。
- 第 2 行：明确说明不是输入错误，而是功能缺失。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:532` 至 `:534`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:477` 至 `:478`。

```python
    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=len(b))
```

逐行对应说明：

- 第 1 行：原注释：能运行到此处时只剩 pad。
- 第 2 行：元组解包；`len(b)` 作为 `ntaps` 关键字实参。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:536` 至 `:538`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:480` 至 `:481`。

```python
    # Get the steady state of the filter's step response.
    zi = firfilter_zi(b)
```

逐行对应说明：

- 第 1 行：说明下一行求阶跃稳态。
- 第 2 行：返回长度 `len(b)-1` 的状态向量。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:540` 至 `:548`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:483` 至 `:489`。

```python
    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = cp.reshape(zi, zi_shape)
    x0 = _axis_slice(ext, stop=1, axis=axis)
```

逐行对应说明：

- 第 1 行：以下三行原注释开始。
- 第 2 行：目标是让 `zi*x0` 广播。
- 第 3 行：广播后的 shape 满足 `firfilter(..., zi=...)`。
- 第 4 行：列表重复，先为每个维度放置 1。
- 第 5 行：滤波轴改成状态长度；负索引同样有效。
- 第 6 行：改变视图形状，不改变状态元素次序。
- 第 7 行：`stop=1` 取得滤波轴首样本并保留该维度。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:550` 至 `:552`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:491` 至 `:492`。

```python
    # Forward filter.
    (y, zf) = firfilter(b, ext, axis=axis, zi=zi * x0)
```

逐行对应说明：

- 第 1 行：标记正向滤波。
- 第 2 行：`zi*x0` 按端点缩放；二元返回值解包。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:554` 至 `:559`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:494` 至 `:497`。

```python
    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = _axis_slice(y, start=-1, axis=axis)
    (y, zf) = firfilter(b, _axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)
```

逐行对应说明：

- 第 1 行：标记反向阶段。
- 第 2 行：解释广播目的。
- 第 3 行：取得正向输出最后一个样本。
- 第 4 行：内层先反转 `y`，再用同一 `b` 和 `zi*y0` 滤波。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:561` 至 `:563`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:499` 至 `:500`。

```python
    # Reverse y.
    y = _axis_reverse(y, axis=axis)
```

逐行对应说明：

- 第 1 行：标记恢复方向。
- 第 2 行：用步长 `-1` 的切片反转目标轴。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:566` 至 `:568`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:502` 至 `:504`。

```python
    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = _axis_slice(y, start=edge, stop=-edge, axis=axis)
```

逐行对应说明：

- 第 1 行：只有发生正长度延拓才裁边。
- 第 2 行：原注释说明取回实际信号区间。
- 第 3 行：Python 半开切片 `[edge:-edge]` 同时去掉两端。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:571` 至 `:571`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:506` 至 `:506`。

```python
    return cp.copy(y)
```

逐行对应说明：

- 第 1 行：返回独立数组；无论切片是否为视图都不共享结果视图。


### 4.2 helper 逐行执行逻辑

`firfilter`：

1. 基准 `142-147`：把 `b` 转数组并要求 `b` 一维、`x` 非标量。
2. `149-177`：若提供 `zi`，要求维数与 `x` 相同；目标 shape 等于 `x.shape`，但滤波轴长度改为 `len(b)-1`。singleton 非滤波轴通过 stride 0 广播。
3. `178-184`：用 `cp.result_type` 合并 dtype；只接受浮点、复数和 object 字符类别，再统一转换 `b`、`x`。
4. `186`：`cp.apply_along_axis` 对每条一维序列运行 `cp.convolve(b, y)`，得到 full convolution。
5. `188-194`：若有 `zi`，把它加到 full convolution 开头的状态区；随后裁成与输入等长的 `out`。
6. `196-201`：无初值只返回 `out`；有初值还把 full convolution 尾部作为 `zf` 返回。

`lfilter_zi` / `firfilter_zi`：

1. `firfilter_zi` 在基准 `360` 把分母固定为 `np.float32(1.0)`，即纯 FIR 的 $a=[1]$。
2. `lfilter_zi` 的 `b`、`a` 都被规范为一维，去除分母开头的零，并用 `a[0]` 归一化。
3. 两组系数补零到共同长度 $n$。
4. 构造 $I-A=I-\operatorname{companion}(a)^T$ 和 $B=b[1:]-a[1:]b[0]$。
5. `cp.linalg.solve(IminusA, B)` 解 $(I-A)z_i=B$，得到单位阶跃稳态状态。

`_validate_pad`：

1. `365-372` 验证 `padtype`；非法值抛 `ValueError`。
2. `374-381` 将 `None` 变成零延拓；未指定 `padlen` 时取 `3*ntaps`。
3. `383-388` 要求目标轴长度严格大于 `edge`。
4. `390-400` 分派到 even、odd、constant helper，或直接令 `ext=x`。
5. `401` 同时返回长度和扩展数组。

数组 helper：`_axis_slice` 创建每维均为 `slice(None)` 的列表，只替换目标轴切片；`_axis_reverse` 复用它并指定 `step=-1`。odd 延拓使用 $2x_{end}-x_{mirror}$；even 直接拼接镜像；constant 用全 1 数组广播端点值。三者最终都用 `cp.concatenate(..., axis=axis)` 拼成 `[左延拓, x, 右延拓]`。

## 5. 调用链与算法总结

```text
cusignal.firfilter2
  → _validate_pad
      → _odd_ext / _even_ext / _const_ext
          → _axis_slice
  → firfilter_zi
      → lfilter_zi
          → companion + cp.linalg.solve
  → _axis_slice(ext, 首样本)
  → firfilter（正向）
      → cp.convolve
  → _axis_reverse
  → firfilter（反向）
      → cp.convolve
  → _axis_reverse
  → _axis_slice（裁边）
  → cp.copy
```

按数学符号，忽略边界辅助项时：

$$
y = R\{b * R\{b*x\}\},\qquad
H_{fb}(e^{j\omega})=H(e^{j\omega})H(e^{-j\omega}).
$$

对实系数 $b$，后式为 $|H(e^{j\omega})|^2$。

## 6. 数学映射、边界、复杂度和阅读检查

### 6.1 数学—代码映射

| 数学步骤 | 代码位置 |
| --- | --- |
| FIR 卷积 $v[n]=\sum b[k]x[n-k]$ | `filtering.py` 基准 `186` |
| 稳态方程 $(I-A)z_i=B$ | 基准 `lfilter_zi:247-318`，核心为 `cp.linalg.solve` |
| 按端点缩放初态 | 基准 `489, 492, 496-497` |
| 时间反转 $R$ | 基准 `497, 500`；实现位于 `arraytools.py:_axis_reverse` |
| 正向与反向两次 FIR | 基准 `492, 497` |
| 去除辅助边界 | 基准 `502-504` |

### 6.2 边界、异常和 dtype

- `method` 非 pad/gust：`ValueError`；gust：`NotImplementedError`。
- `x.shape[axis] <= edge`：`ValueError`，所以默认条件是目标轴长度必须大于 `3*len(b)`。
- `b` 非一维、`x` 为标量或 `zi` shape 不兼容：在 `firfilter` 中报错。
- 整数输入不会被静默当整数卷积处理：合成 dtype 的字符不在允许集合时抛 `NotImplementedError`。
- `axis` 越界、负 `padlen` 等未被显式友好校验的情况可能由底层索引/切片行为决定，是接口风险点。
- 文档称 `padlen < x.shape[axis]-1`，实现检查是 `x.shape[axis] > edge`，等价于 `edge < x.shape[axis]`，比文字中的 `-1` 表述宽一个边界；应以代码行为为准。

### 6.3 复杂度与性能

设沿滤波轴长度为 $N$，每端 padding 为 $P$，FIR 长度为 $M$，其余维度共有 $B$ 条序列。代码实际调用 `cp.convolve`，复杂度取决于 CuPy 为当前规模选择的卷积实现：直接卷积上界可写为 $O(B(N+2P)M)$；若底层采用 FFT，典型为 $O(BL\log L)$，$L\approx N+2P+M$。两次滤波只改变常数因子。

空间包括扩展数组、两次 full convolution 和输出，量级为 $O(B(N+2P+M))$。`cp.apply_along_axis` 的逐轴包装、full convolution 后再裁剪、两次反转视图及最终复制都可能成为性能瓶颈。

### 6.4 建议阅读顺序与检查点

1. 先读 `firfilter2:468-506`：只追踪 `ext → y → reverse(y) → y`。
2. 再读 `_validate_pad` 和三种 extension：手画长度 $N=5,P=2$ 的数组。
3. 再读 `firfilter:142-201`：确认 full convolution 怎样裁成输入长度，`zi/zf` 分别落在哪一段。
4. 再读 `firfilter_zi → lfilter_zi`：把代码逐项对应到 $(I-A)z_i=B$。
5. 最后回到 `firfilter2`，检查为什么反向初值乘的是 `y0`，而不是原始 `x` 的末样本。

自检：

1. `method="gust"` 会经过哪些语句，为什么 `irlen` 永远不会被读取？
2. `zi_shape` 为什么除滤波轴外都是 1？
3. `cp.convolve` 的 full 输出怎样拆成等长输出和 `zf`？
4. odd extension 的 $2x_{end}-x_{mirror}$ 与阶段一公式如何对应？
5. 为什么最终必须先恢复方向再裁边？
6. 哪些语句是数学核心，哪些只是 CuPy shape/dtype/内存工程？

参考答案：

1. `method="gust"` 先通过 `method not in ["pad", "gust"]`，随后命中 `method == "gust"` 并在基准 `474-475` 抛 `NotImplementedError`；因此执行永远到不了 padding 路径，`irlen` 也没有任何读取语句。
2. `zi` 只沿滤波轴保存 `len(b)-1` 个状态；其他轴设为 1，使 `zi*x0` 能按 NumPy/CuPy 广播规则复用于每条 axis line，对应基准 `483-489`。
3. `cp.convolve` 先产生长度 `N+M-1` 的 full 输出；基准 `193-194` 取前 `N` 个元素作为 `out`，`199-200` 取剩余 `M-1` 个元素作为 `zf`。
4. `_odd_ext` 的左侧构造是 `2*left_end-left_ext`，正好对应阶段一的 $x_o[-k]=2x[0]-x[k]$；右侧以末端点作同样反射。
5. 第二次 FIR 的输入是反转序列，所以它的结果仍处于反向索引；必须先在基准 `500` 恢复方向，随后 `[edge:-edge]` 才对应原始信号的左右有效区。
6. `cp.convolve`、两次 `_axis_reverse`、稳态 `zi` 缩放和裁边构成数学算法；`cp.asarray`、dtype 合并、reshape、stride-0 广播、shape 检查和最终 `cp.copy` 是数组/GPU/内存工程。

## 7. 完整性说明

阶段二修改只发生在 `Learning/cusignal-23.08.00`：为主定义和两处公开导出插入了独立的 `<学习注释：...>` 行。`ZKX/cusignal-23.08.00` 只读基准未修改。交付前应以“删除所有合法学习注释行后的字节/逐行内容”再次核对学习副本和基准。
