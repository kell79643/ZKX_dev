# detrend 算子学习 — Python 源码算法

> 本文件对应 `Learning/operators/filtering/detrend/`，记录阶段二（cuSignal 23.08.00 Python 源码算法）。
> 阶段一原理见同目录 `数学物理原理.md`。本文所有源码摘录取自只读基准原文，行号同时给出「只读基准 / 学习副本」。
> 学习副本路径：`Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py`
> 只读基准路径：`ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py`
> 学习副本已为 detrend 相关代码插入 31 行 `<学习注释：...>`，原始内容与只读基准逐行一致（已核对）。

---

## 一、代码定位与接口概览

### 1. 公开导入路径

`detrend` 通过以下两级导出暴露给用户：

| 层级 | 文件（共享根 `ZKX_dev/` 下相对路径） | 行号（只读基准 / 学习副本） |
| --- | --- | --- |
| 模块包 | `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py` | 16 / 16 |
| 定义文件 | `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py` | 1003 / 1012 |

`__init__.py:16` 处 `    detrend,` 把 `filtering.py` 中的 `detrend` 函数重新导出，用户可通过 `cusignal.detrend(...)` 或 `cusignal.filtering.detrend(...)` 调用。

### 2. 函数签名

```python
def detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
```

| 参数 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `data` | `array_like` | 无（必填） | 输入数据，可为 CuPy/NumPy 数组或可转数组对象；实数或复数。 |
| `axis` | `int` | `-1` | 沿该轴去趋势。负值表示从未尾计数（`-1` 为最后一根轴）。 |
| `type` | `str` | `"linear"` | `'linear'`/`'l'`：减去线性最小二乘拟合；`'constant'`/`'c'`：只减均值。 |
| `bp` | `array_like of ints` | `0` | 断点索引序列，仅在 `type='linear'` 生效；把数据分段，每段独立线性拟合。 |
| `overwrite_data` | `bool` | `False` | `True` 时就地修改（仅 `type='linear'` 且浮点 dtype 时生效），省一次拷贝。 |

| 返回值 | 类型 | shape | 语义 |
| --- | --- | --- | --- |
| `ret` | `cupy.ndarray` | 与输入相同 | 去趋势后的数据。`constant` 分支必为新数组；`linear` 分支视 `overwrite_data` 可能为新数组或就地修改视图。 |

### 3. 依赖的模块级导入

定义文件 `filtering.py` 顶部导入了 `cupy as cp`（行 14）与 `numpy as np`（行 15）。`detrend` 中：
- `cp`：`cp.asarray`、`cp.mean`、`cp.expand_dims`、`cp.reshape`、`cp.transpose`、`cp.dot`、`cp.linalg.lstsq`、`cp.ElementwiseKernel`；
- `np`：`np.sort`、`np.unique`、`np.r_`、`np.any`、`np.take`、`np.reshape`（标量/形状运算，在 CPU 端构造索引与形状）。

### 4. 直接依赖的本项目 kernel 与 helper

| 名称 | 只读基准行号 | 学习副本行号 | 作用 |
| --- | --- | --- | --- |
| `_detrend_A_kernel` | 986-1000 | 988-1006 | CuPy `ElementwiseKernel`，在 GPU 上并行生成线性拟合的设计矩阵 `A`（`Npts×2`，常数基 + 归一化线性基）。 |
| `_prod(iterable)` | 1184-1192 | 1215-1223 | helper：返回可迭代对象元素乘积，用于把形状元组相乘得到总元素数。 |

`_detrend_A_kernel` 与 `_prod` 均定义在同一个 `filtering.py` 文件中。`_prod` 虽被文件内其他算子也可能使用，本文只摘录其完整定义（理解 detrend 的 reshape 所必需）。

---

## 二、当前算子的完整相关源码

> 以下按源码原有顺序完整摘录（取自只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py` 原文，无省略、无伪代码）。导出语句见 `__init__.py:16`。

### 2.1 导出语句（`filtering/__init__.py`）

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    firfilter,
    firfilter_zi,
    firfilter2,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

（`detrend` 在该导入块第 16 行；为保持上下文完整摘录整个导入块，逐行解释只覆盖 `detrend` 一行。）

### 2.2 `_detrend_A_kernel`（只读基准 986-1000）

```python
_detrend_A_kernel = cp.ElementwiseKernel(
    "",
    "float64 A",
    """
    if ( i & 1 ) {
        const int new_i { i >> 1 };
        A = new_i * den;
    } else {
        A = 1.0;
    }
    """,
    "_detrend_A_kernel",
    options=("-std=c++11",),
    loop_prep="const double den { 1.0 / _ind.size() };",
)
```

### 2.3 `detrend` 函数（只读基准 1003-1093）

```python
def detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> randgen = cp.random.RandomState(9)
    >>> npoints = 1000
    >>> noise = randgen.randn(npoints)
    >>> x = 3 + 2*cp.linspace(0, 1, npoints) + noise
    >>> (cusignal.detrend(x) - noise).max() < 0.01
    True
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = cp.asarray(data)
    dtype = data.dtype.char
    if dtype not in "dfDF":
        dtype = "d"
    if type in ["constant", "c"]:
        ret = data - cp.expand_dims(cp.mean(data, axis), axis)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError(
                "Breakpoints must be less than length of \
                data along given axis."
            )

        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk

        newdims = np.r_[axis, 0:axis, axis + 1 : rnk]
        newdata = cp.reshape(
            cp.transpose(data, tuple(newdims)), (N, _prod(dshape) // N)
        )

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in "dfDF":
            newdata = newdata.astype(dtype)

        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = int(bp[m + 1] - bp[m])
            A = _detrend_A_kernel(size=Npts * 2)
            A = cp.reshape(A, (Npts, 2))
            sl = slice(bp[m], bp[m + 1])
            coef, _, _, _ = cp.linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - cp.dot(A, coef)

        # Put data back in original shape.
        tdshape = np.take(dshape, newdims, 0)
        ret = cp.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = cp.transpose(ret, tuple(olddims))
        return ret
```

### 2.4 `_prod` helper（只读基准 1184-1192）

```python
def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than cp.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product
```

---

## 三、docstring 逐行翻译与解释

`detrend` 的 docstring 位于只读基准 1004-1041 / 学习副本 1013-1050。逐行说明：

| 只读基准 | 学习副本 | 原文 | 翻译与解释 |
| --- | --- | --- | --- |
| 1004 | 1013 | `    """` | docstring 起始三引号。 |
| 1005 | 1014 | `    Remove linear trend along axis from data.` | 一句话摘要：沿指定轴去除数据的线性趋势。 |
| 1006 | 1015 | （空行） | 摘要与参数段分隔。 |
| 1007-1008 | 1016-1017 | `    Parameters` / `    ----------` | NumPy 风格参数段标题与下划线。 |
| 1009-1010 | 1018-1019 | `    data : array_like` / `        The input data.` | 参数 `data`，类型 `array_like`，语义：输入数据。 |
| 1011-1013 | 1020-1022 | `    axis : int, optional` + 两行说明 | 参数 `axis`，可选整数，默认最后一根轴（`-1`）。 |
| 1014-1018 | 1023-1027 | `    type : {'linear', 'constant'}, optional` + 说明 | 参数 `type`：`'linear'`（默认）减去线性最小二乘拟合；`'constant'` 只减均值。RST `` ``code`` `` 与 `` `code` `` 是 Sphinx 行内标记。 |
| 1019-1022 | 1028-1031 | `    bp : array_like of ints, optional` + 说明 | 参数 `bp`：断点序列，仅在 linear 模式生效；每段独立线性拟合；断点是数据索引。 |
| 1023-1024 | 1032-1033 | `    overwrite_data : bool, optional` / `        If True, ...` | 参数 `overwrite_data`：`True` 就地去趋势、省拷贝，默认 `False`。 |
| 1025 | 1034 | （空行） | 参数段与返回段分隔。 |
| 1026-1027 | 1035-1036 | `    Returns` / `    -------` | 返回段标题与下划线。 |
| 1028-1029 | 1037-1038 | `    ret : ndarray` / `        The detrended input data.` | 返回 `ret`，类型 `ndarray`，语义：去趋势后的输入数据。 |
| 1030 | 1039 | （空行） | 返回段与示例段分隔。 |
| 1031-1032 | 1040-1041 | `    Examples` / `    --------` | 示例段标题与下划线。 |
| 1033 | 1042 | `    >>> import cusignal` | 导入 cusignal 包。 |
| 1034 | 1043 | `    >>> import cupy as cp` | 导入 CuPy 并别名 `cp`。 |
| 1035 | 1044 | `    >>> randgen = cp.random.RandomState(9)` | 用种子 9 创建 GPU 随机数生成器（保证可复现）。 |
| 1036 | 1045 | `    >>> npoints = 1000` | 信号长度 1000。 |
| 1037 | 1046 | `    >>> noise = randgen.randn(npoints)` | 生成 1000 点标准正态噪声。 |
| 1038 | 1047 | `    >>> x = 3 + 2*cp.linspace(0, 1, npoints) + noise` | 构造 $x[n]=3+2\cdot t+\text{noise}$，其中 $t\in[0,1]$ 等间距；含常数 3 与线性斜率 2 的趋势。 |
| 1039 | 1048 | `    >>> (cusignal.detrend(x) - noise).max() < 0.01` | 去趋势后应近似还原 `noise`；断言两者差的最大值 < 0.01。 |
| 1040 | 1049 | `    True` | 期望输出 `True`，验证 linear 去趋势正确去除常数+线性趋势。 |
| 1041 | 1050 | `    """` | docstring 结束三引号。 |

---

## 四、源代码逐行解释

### 4.1 `_detrend_A_kernel`（只读基准 986-1000 / 学习副本 988-1006）

| 只读基准 | 学习副本 | 源码 | 解释 |
| --- | --- | --- | --- |
| 986 | 988 | `_detrend_A_kernel = cp.ElementwiseKernel(` | 把一个 CuPy `ElementwiseKernel` 赋给模块级变量 `_detrend_A_kernel`。`ElementwiseKernel` 是 CuPy 的逐元素 GPU kernel 封装：调用 `_detrend_A_kernel(size=K)` 会在 GPU 上启动 K 个线程，每个线程执行一次给定的 C++/CUDA 主体。 |
| 987 | 989 | `    "",` | 第 1 参数（输入参数列表）为空字符串：kernel 无输入参数，只产出输出。 |
| 988 | 990 | `    "float64 A",` | 第 2 参数（输出参数列表）：声明一个 `float64` 输出 `A`；每个线程写出一个 `A` 值，最终得到长度为 `size` 的一维数组。 |
| 989 | 994 | `    """` | 第 3 参数（kernel 主体）开始的 C++/CUDA 三引号字符串。 |
| 990 | 995 | `    if ( i & 1 ) {` | C++ 语句：`i` 是 CuPy 注入的线程线性索引（0..size-1）；`i & 1` 判奇偶（按位与 1）。奇数 `i` 进 if 分支，偶数进 else。 |
| 991 | 996 | `        const int new_i { i >> 1 };` | `i >> 1` 即右移一位 = 整除 2。奇数 `i` 时 `new_i=(i-1)/2`，得到序列 0,1,2,…。用 C++11 列表初始化 `{ }` 声明 `const int`。 |
| 992 | 997 | `        A = new_i * den;` | 线性基：`A = new_i * den`，即归一化时间索引乘以常数 `den`。这些值 reshape 后成为设计矩阵第 1 列。 |
| 993 | 998 | `    } else {` | else 分支：偶数 `i`。 |
| 994 | 999 | `        A = 1.0;` | 常数基：偶数 `i` 写出 `1.0`，reshape 后成为设计矩阵第 0 列（全 1）。 |
| 995 | 1000 | `    }` | if-else 闭合。 |
| 996 | 1001 | `    """,` | kernel 主体字符串结束。 |
| 997 | 1002 | `    "_detrend_A_kernel",` | 第 4 参数：kernel 名称（用于编译/调试标识）。 |
| 998 | 1003 | `    options=("-std=c++11",),` | 关键字参数 `options`：传给 NVCC 的编译选项，启用 C++11（因为主体用了列表初始化 `{ }`）。 |
| 999 | 1005 | `    loop_prep="const double den { 1.0 / _ind.size() };",` | 关键字参数 `loop_prep`：在循环开始前执行一次的 C++ 语句。`_ind.size()` 是 CuPy 注入的总线程数（=调用时传入的 `size`=Npts*2），故 `den = 1.0/(Npts*2)`，是线性基的归一化系数。依原理 5，该缩放不改变去趋势残差。 |
| 1000 | 1006 | `)` | `ElementwiseKernel(...)` 调用闭合括号。 |

**kernel 语义总结**：调用 `_detrend_A_kernel(size=Npts*2)` 产出长度 `2*Npts` 的一维数组，元素交替为 `1.0, 0*den, 1.0, 1*den, 1.0, 2*den, …`；`reshape((Npts,2))` 后第 0 列全 1（常数基），第 1 列为 `0,den,2den,…,(Npts-1)*den`（归一化线性基）。即原理 2 的设计矩阵 $Z=[\mathbf{1},\alpha\mathbf{n}]$。

### 4.2 `detrend` 函数体（只读基准 1003-1093 / 学习副本 1012-1123）

| 只读基准 | 学习副本 | 源码 | 解释 |
| --- | --- | --- | --- |
| 1003 | 1012 | `def detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):` | 函数定义与签名。`type`/`bp` 默认实现单段线性去趋势。 |
| 1004 | 1013 | `    """` | docstring 开始（见第三节逐行翻译）。 |
| 1005-1040 | 1014-1049 | （docstring 内容与示例） | 见第三节，此处不重复。 |
| 1041 | 1050 | `    """` | docstring 结束。 |
| 1042 | 1052 | `    if type not in ["linear", "l", "constant", "c"]:` | 校验 `type`：只接受 `linear`/`l`/`constant`/`c` 四种合法值。 |
| 1043 | 1053 | `        raise ValueError("Trend type must be 'linear' or 'constant'.")` | 非法值抛 `ValueError`，终止执行。 |
| 1044 | 1055 | `    data = cp.asarray(data)` | 把输入转为 CuPy 数组（已在 GPU 上则原样返回；CPU 数组则搬运到 GPU），后续全部在 GPU 计算。 |
| 1045 | 1056 | `    dtype = data.dtype.char` | 取 dtype 的单字符码（如 `'f'`/`'d'`/`'F'`/`'D'`）。 |
| 1046 | 1058 | `    if dtype not in "dfDF":` | 判断是否为浮点：`d`=float64, `f`=float32, `D`=complex128, `F`=complex64。 |
| 1047 | 1059 | `        dtype = "d"` | 非浮点（如 int）回退到 `float64`，保证 `lstsq` 与相减的数值正确性（整数数组无法做最小二乘）。 |
| 1048 | 1061 | `    if type in ["constant", "c"]:` | constant 分支入口。 |
| 1049 | 1062 | `        ret = data - cp.expand_dims(cp.mean(data, axis), axis)` | `cp.mean(data, axis)` 沿 `axis` 求均值（shape 去掉该轴）；`cp.expand_dims(..., axis)` 在 `axis` 处插回长度 1 的维以便广播；相减得 $y=x-\bar x$（原理 3）。 |
| 1050 | 1063 | `        return ret` | 返回去均值结果，函数结束。 |
| 1051 | 1064 | `    else:` | linear 分支入口。 |
| 1052 | 1066 | `        dshape = data.shape` | 取输入形状元组。 |
| 1053 | 1067 | `        N = dshape[axis]` | 目标轴长度 $N$（去趋势方向的样本数）。 |
| 1054 | 1069 | `        bp = np.sort(np.unique(np.r_[0, bp, N]))` | `np.r_[0, bp, N]` 把端点 0、用户断点 `bp`、端点 N 拼成数组；`np.unique` 去重、`np.sort` 升序，得分段边界。默认 `bp=0` 时结果 `[0, N]`（单段）。用 NumPy 在 CPU 端处理少量整数。 |
| 1055 | 1071 | `        if np.any(bp > N):` | 检查是否有断点超过 $N$。 |
| 1056 | 1072 | `            raise ValueError(` | 抛异常（跨行）。 |
| 1057 | 1073 | `                "Breakpoints must be less than length of \` | 异常消息第 1 行；行尾 `\` 为 Python 续行符。 |
| 1058 | 1074 | `                data along given axis."` | 异常消息第 2 行。 |
| 1059 | 1075 | `            )` | `raise ValueError(...)` 闭合括号。 |
| 1060 | 1076 | （空行） | 逻辑分段。 |
| 1061 | 1078 | `        Nreg = len(bp) - 1` | 分段数 = 边界数 − 1。 |
| 1062 | 1079 | `        # Restructure data so that axis is along first dimension and` | 原注释：说明下面把目标轴提到第 0 维。 |
| 1063 | 1080 | `        #  all other dimensions are collapsed into second dimension` | 原注释续：其余维压成第 1 维。 |
| 1064 | 1082 | `        rnk = len(dshape)` | 维数（数组秩）。 |
| 1065 | 1083 | `        if axis < 0:` | 处理负轴。 |
| 1066 | 1084 | `            axis = axis + rnk` | 转正向索引（如 `axis=-1, rnk=2` → `axis=1`）。 |
| 1067 | 1085 | （空行） | 逻辑分段。 |
| 1068 | 1087 | `        newdims = np.r_[axis, 0:axis, axis + 1 : rnk]` | 构造轴重排顺序：`[axis, 0..axis-1, axis+1..rnk-1]`，即目标轴置首、其余轴按原序跟随。 |
| 1069 | 1089 | `        newdata = cp.reshape(` | 开始 reshape（跨行）。 |
| 1070 | 1090 | `            cp.transpose(data, tuple(newdims)), (N, _prod(dshape) // N)` | 先 `transpose` 按 `newdims` 重排轴，再 `reshape` 成 `(N, 总元素//N)`：每列是一条需去趋势的信号，共「其余维度乘积」条。`_prod(dshape)` 算总元素数。 |
| 1071 | 1091 | `        )` | `cp.reshape(...)` 闭合括号。 |
| 1072 | 1092 | （空行） | 逻辑分段。 |
| 1073 | 1094 | `        if not overwrite_data:` | 若不允许就地修改。 |
| 1074 | 1095 | `            newdata = newdata.copy()  # make sure we have a copy` | 拷贝一份，保证后续减法不污染原输入。原行内注释 `# make sure we have a copy`。 |
| 1075 | 1097 | `        if newdata.dtype.char not in "dfDF":` | 若转置/reshape 后 dtype 仍非浮点（如原为 int）。 |
| 1076 | 1098 | `            newdata = newdata.astype(dtype)` | 转成前面确定的浮点 `dtype`。 |
| 1077 | 1099 | （空行） | 逻辑分段。 |
| 1078 | 1100 | `        # Find leastsq fit and remove it for each piece` | 原注释：对每段做最小二乘拟合并减去。 |
| 1079 | 1102 | `        for m in range(Nreg):` | 遍历每个分段。 |
| 1080 | 1104 | `            Npts = int(bp[m + 1] - bp[m])` | 当前段样本数 = 边界差；`int()` 转为 Python int。 |
| 1081 | 1106 | `            A = _detrend_A_kernel(size=Npts * 2)` | 调用 kernel 生成 `2*Npts` 个设计矩阵元素。 |
| 1082 | 1107 | `            A = cp.reshape(A, (Npts, 2))` | reshape 成 `(Npts,2)`：第 0 列常数基 1，第 1 列归一化线性基。 |
| 1083 | 1108 | `            sl = slice(bp[m], bp[m + 1])` | 当前段在第 0 维上的切片 `[bp[m], bp[m+1])`。 |
| 1084 | 1111 | `            coef, _, _, _ = cp.linalg.lstsq(A, newdata[sl])` | 解最小二乘 $A\,\text{coef}\approx\text{newdata}[sl]$，即 $\hat{\boldsymbol\beta}=(A^{\mathsf T}A)^{-1}A^{\mathsf T}\mathbf{x}$；`coef` 形状 `(2, 列数)`。`lstsq` 返回 `(解, 残差, 秩, 奇异值)`，只取解，其余用 `_` 丢弃。 |
| 1085 | 1113 | `            newdata[sl] = newdata[sl] - cp.dot(A, coef)` | `cp.dot(A, coef)` 算拟合值 $A\hat{\boldsymbol\beta}$；相减得残差 $\mathbf{y}=(I-P)\mathbf{x}$，就地写回 `newdata[sl]`。 |
| 1086 | 1114 | （空行） | 逻辑分段。 |
| 1087 | 1115 | `        # Put data back in original shape.` | 原注释：还原原始形状。 |
| 1088 | 1117 | `        tdshape = np.take(dshape, newdims, 0)` | 按 `newdims` 重排形状元组，得到转置后的形状。 |
| 1089 | 1118 | `        ret = cp.reshape(newdata, tuple(tdshape))` | 把 `newdata` 还原成转置后的多维形状。 |
| 1090 | 1120 | `        vals = list(range(1, rnk))` | `[1,2,…,rnk-1]`，用于构造逆置换。 |
| 1091 | 1121 | `        olddims = vals[:axis] + [0] + vals[axis:]` | `newdims` 的逆置换：把第 0 维（目标轴）插回原 `axis` 位置。 |
| 1092 | 1122 | `        ret = cp.transpose(ret, tuple(olddims))` | 转置回原始轴顺序。 |
| 1093 | 1123 | `        return ret` | 返回去趋势结果。 |

### 4.3 `_prod` helper（只读基准 1184-1192 / 学习副本 1215-1223）

| 只读基准 | 学习副本 | 源码 | 解释 |
| --- | --- | --- | --- |
| 1184 | 1215 | `def _prod(iterable):` | 定义 helper，接收可迭代对象。 |
| 1185 | 1216 | `    """` | docstring 开始。 |
| 1186 | 1217 | `    Product of a list of numbers.` | 一句话：返回数值列表的乘积。 |
| 1187 | 1218 | `    Faster than cp.prod for short lists like array shapes.` | 说明：对形状元组这类短列表比 `cp.prod` 快（避免 GPU 内核开销）。 |
| 1188 | 1219 | `    """` | docstring 结束。 |
| 1189 | 1220 | `    product = 1` | 累乘器初始化为 1。 |
| 1190 | 1221 | `    for x in iterable:` | 遍历每个元素。 |
| 1191 | 1222 | `        product *= x` | 累乘。 |
| 1192 | 1223 | `    return product` | 返回乘积。 |

---

## 五、调用链与算法总结

### 调用链

```
cusignal.detrend(data, axis, type, bp, overwrite_data)      # __init__.py:16 导出
  └─ filtering.detrend(...)                                  # filtering.py:1003
       ├─ type 校验 / cp.asarray / dtype 归一化
       ├─ [constant] cp.mean + cp.expand_dims → 相减返回       # 原理 3
       └─ [linear]
            ├─ 构造断点 bp = sort(unique([0, bp, N]))          # 分段边界
            ├─ transpose + reshape → newdata (N, M)            # 目标轴置首、压成 2D
            ├─ for 每段:
            │     ├─ _detrend_A_kernel(size=Npts*2)            # GPU 生成设计矩阵 A
            │     ├─ reshape A → (Npts, 2)
            │     ├─ cp.linalg.lstsq(A, newdata[sl])           # 最小二乘解 β̂
            │     └─ newdata[sl] -= A @ coef                   # 残差 (I-P)x
            └─ reshape + transpose → 还原原始形状返回           # 原理 2+4+5
```

### 算法总结

- **constant 分支**：$y=x-\bar x_{\text{axis}}$，纯 GPU 数组运算，对应原理 3（0 阶多项式 OLS 残差）。
- **linear 分支**：把目标轴置首并压成 2D `(N, M)`，对每段构造设计矩阵 $A=[\mathbf{1},\alpha\mathbf{n}]$，用 `cp.linalg.lstsq` 求 $\hat{\boldsymbol\beta}$，再减去 $A\hat{\boldsymbol\beta}$。对应原理 2（1 阶 OLS）+ 原理 4（`bp` 分段）+ 原理 5（归一化不变）。
- 设计矩阵由 GPU `ElementwiseKernel` 并行生成，避免在 Python 端循环构造；`lstsq` 与 `dot` 也在 GPU 上执行，整体为 GPU 加速的 SciPy `signal.detrend` 等价实现。

---

## 六、数学映射、边界、复杂度与阅读检查

### 1. 数学公式与代码逐项映射

| 数学（原理） | 代码落实 | 只读基准行 / 学习副本行 |
| --- | --- | --- |
| $y[n]=x[n]-\bar x$（原理 3） | `data - cp.expand_dims(cp.mean(data, axis), axis)` | 1049 / 1062 |
| 设计矩阵 $Z=[\mathbf{1},\alpha\mathbf{n}]$（原理 2） | `_detrend_A_kernel` + `cp.reshape(A,(Npts,2))` | 986-1000, 1081-1082 / 988-1006, 1106-1107 |
| $\hat{\boldsymbol\beta}=(A^{\mathsf T}A)^{-1}A^{\mathsf T}\mathbf{x}$（原理 2） | `cp.linalg.lstsq(A, newdata[sl])` | 1084 / 1111 |
| $\mathbf{y}=(I-P)\mathbf{x}=\mathbf{x}-A\hat{\boldsymbol\beta}$（原理 2） | `newdata[sl] - cp.dot(A, coef)` | 1085 / 1113 |
| 分段独立拟合（原理 4） | `bp = sort(unique([0,bp,N]))` + `for m in range(Nreg)` | 1054, 1079 / 1069, 1102 |
| 归一化不变性（原理 5） | `den = 1.0/_ind.size()`，缩放线性基不改变残差 | 999 / 1005 |

### 2. 底层机制

- **`cp.ElementwiseKernel`**：CuPy 的 GPU 逐元素 kernel。`_detrend_A_kernel` 用 `loop_prep` 在循环前算一次 `den`，每个线程按 `i` 奇偶写出常数基或线性基。
- **`cp.linalg.lstsq`**：基于 GPU LAPACK（SVD 分解）解最小二乘，数值稳定，能处理秩亏情形。
- **`cp.dot`**：矩阵乘法，算拟合值 $A\hat{\boldsymbol\beta}$。
- 无 FFT、无插值、无卷积；仅线性代数 + 数组重排。

### 3. 边界处理、异常与数值稳定性

- **type 非法**：`1042` 抛 `ValueError`。
- **断点越界**：`bp > N` 时 `1055` 抛 `ValueError`。
- **dtype 非浮点**：回退 `float64`（`1046-1047`），转置后再次检查并 `astype`（`1075-1076`），保证 `lstsq` 可解。
- **负轴**：`1065-1066` 转正向索引。
- **断点去重排序**：`np.unique` 避免重复断点导致空段。
- **就地修改控制**：`overwrite_data=False` 时 `1073-1074` 拷贝，保护原输入。
- **数值稳定性**：`lstsq` 用 SVD，比直接解正规方程 $A^{\mathsf T}A$ 更稳；线性基归一化（`den`）也改善 $A$ 条件数（但依原理 5 不影响结果）。
- **段内样本数过少**：若某段 `Npts=1`，$A$ 为 `(1,2)` 列满秩不成立，`lstsq` 仍返回最小范数解（SVD），结果为该点减去拟合值（退化为 0）；`Npts=0`（重复断点已被 `unique` 去除）不会出现。

### 4. 时间/空间复杂度与性能瓶颈

- 设总元素数 $E=\prod\text{shape}$，分段数 $R$，平均段长 $L=N/R$。
- **时间**：constant 分支 $O(E)$（一次均值 + 一次相减）。linear 分支每段 `lstsq` 对 `(L,2)` 设计矩阵与 `(L,M)` 数据，复杂度约 $O(L^2\cdot 2 + L\cdot 2\cdot M)$，总计 $O(R L^2 + 2 N M)$；由于 $2\ll L$，主要是 SVD 开销。kernel 生成 $A$ 为 $O(N)$。
- **空间**：`newdata` 拷贝 $O(E)$；每段 `A` 为 $O(L)$、`coef` 为 $O(M)$。
- **瓶颈**：`cp.linalg.lstsq` 的 SVD 是主要开销；对小段重复调用 `lstsq` 有 kernel 启动开销。`transpose`/`reshape` 通常为视图操作，开销小。

### 5. 建议阅读顺序与观察要点

1. **先读导出**：`__init__.py:16`，确认 `detrend` 的公开入口。
2. **读签名与 docstring**（`1003-1041`）：理解四个参数与示例 $x[n]=3+2t+\text{noise}$ 的构造意图。
3. **读 constant 分支**（`1048-1050`）：最简情形，对应原理 3。
4. **读 linear 分支的数据重排**（`1052-1076`）：重点理解 `newdims`、`transpose`+`reshape` 如何把任意轴的去趋势归约为 2D 列独立处理。
5. **读 `_detrend_A_kernel`**（`986-1000`）：理解 GPU 如何并行生成设计矩阵，以及 `den` 归一化为何不影响结果（原理 5）。
6. **读最小二乘循环**（`1079-1085`）：核心算法，对应原理 2。
7. **读形状还原**（`1087-1093`）：理解 `olddims` 逆置换。
8. **观察问题**：为什么 `bp` 默认 `0` 却能表示「无断点」？`_prod(dshape)//N` 为何等于其余维度乘积？`lstsq` 返回的四元组中为何只取第一个？

---

## 七、阶段二自检

1. `_detrend_A_kernel` 的 `i & 1` 与 `i >> 1` 如何实现「偶数写 1、奇数写归一化索引」？reshape 成 `(Npts,2)` 后两列分别是什么？
2. `den = 1.0 / _ind.size()` 中 `_ind.size()` 等于多少？为什么这个归一化常数不影响去趋势输出（结合原理 5）？
3. `constant` 分支为何不需要 reshape/transpose？`cp.expand_dims` 在这里起什么作用？
4. `newdims = np.r_[axis, 0:axis, axis + 1 : rnk]` 对 `axis=1, rnk=3` 的结果是什么？为什么这样能把目标轴置首？
5. `olddims = vals[:axis] + [0] + vals[axis:]` 如何构成 `newdims` 的逆置换？以 `axis=1, rnk=3` 验证。
6. `cp.linalg.lstsq(A, newdata[sl])` 中 `A` 形状 `(Npts,2)`、`newdata[sl]` 形状 `(Npts, M)`，`coef` 的形状是什么？为什么？
7. `overwrite_data=True` 时省了哪一次拷贝？为什么 constant 分支不受 `overwrite_data` 影响？
8. 若输入为整数数组，`1046-1047` 与 `1075-1076` 两处 dtype 检查分别保证什么？
9. `_prod` 相比 `cp.prod` 在本场景为何更快？
10. 该实现与 SciPy `signal.detrend` 在数学上是否等价？设计矩阵的列顺序与归一化差异是否影响结果？
