# firfilter Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开导入路径

`firfilter` 的公开调用路径是：

```python
import cusignal
y = cusignal.firfilter(b, x)
```

导入链为：

```text
cusignal.__init__
  → cusignal.filtering.filtering.firfilter

cusignal.filtering.__init__
  → cusignal.filtering.filtering.firfilter
```

| 职责 | 学习注释副本定位 | 只读基准定位 |
| --- | --- | --- |
| 顶层 `cusignal.firfilter` 导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:41`，符号行为 46 | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:39`，符号行为 43 |
| `cusignal.filtering.firfilter` 导出 | `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14`，符号行为 19 | `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14`，符号行为 18 |
| 函数定义 `firfilter` | `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:110` | `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:109` |

### 1.2 签名和整体契约

```python
def firfilter(b, x, axis=-1, zi=None):
```

| 名称 | 语义 | dtype | shape |
| --- | --- | --- | --- |
| `b` | FIR 分子系数/抽头 | 参与 `cp.result_type` 提升，随后转为公共 dtype | 必须为一维 `(L,)` |
| `x` | N 维输入数组 | 参与公共 dtype 提升 | 至少一维；滤波轴长度记为 $N$ |
| `axis` | 独立执行一维 FIR 的轴 | Python `int` | 默认 `-1`，即最后一轴 |
| `zi` | 可选初始延迟状态 | 给定时也参与公共 dtype 提升 | 与 `x` 同维；滤波轴为 `L-1`，其他轴等于 `x` 或为 1 |
| `out` / docstring 中的 `y` | 因果 FIR 主输出 | 公共 dtype | 与 `x.shape` 相同 |
| `zf` | 可供下一块续接的最终状态 | 公共 dtype | 与广播后 `zi.shape` 相同 |

返回值的数量取决于 `zi`：`zi is None` 时只返回 `out`；否则返回 `(out, zf)`。

## 2. 当前算子的完整相关源码

以下摘录严格使用只读基准原文，不包含后加学习注释。`firfilter` 没有直接调用本项目自定义 helper 或 kernel；直接调用的 `cp.asarray`、`cp.result_type`、`cp.apply_along_axis`、`cp.convolve` 和 `cp.lib.stride_tricks.as_strided` 都是第三方 CuPy API，因此不复制 CuPy 内部实现。

### 2.1 必要的顶层导出语句

基准：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:39-54`。

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

### 2.2 必要的子包导出语句

基准：`ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14-28`。

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

### 2.3 `firfilter` 完整定义

基准：`ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:109-201`。

```python
def firfilter(b, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an FIR filter.

    Filter a data sequence, `x`, using a digital filter. This works for many
    fundamental data types (including Object type). Please note, cuSignal
    doesn't support IIR filters presently, and this implementation is optimized
    for large filtering operations (and inherently depends on fftconvolve)

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """
    b = cp.asarray(b)
    if b.ndim != 1:
        raise ValueError("object of too small depth for desired array")

    if x.ndim == 0:
        raise ValueError("x must be at least 1-D")

    inputs = [b, x]
    if zi is not None:
        # _linear_filter does not broadcast zi, but does do expansion of
        # singleton dims.
        zi = cp.asarray(zi)
        if zi.ndim != x.ndim:
            raise ValueError("object of too small depth for desired array")
        expected_shape = list(x.shape)
        expected_shape[axis] = b.shape[0] - 1
        expected_shape = tuple(expected_shape)
        # check the trivial case where zi is the right shape first
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
```

该 docstring 没有 `Examples` 章节，因而没有可额外摘录的 docstring 示例。

## 3. docstring 逐行翻译与解释

表中“基准行”对应上节原文，“学习行”是插入学习注释后的实时行号。空行按规则不单独解释。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:111` 至 `:112`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:110` 至 `:111`。

```python
    """
    Filter data along one-dimension with an FIR filter.
```

逐行对应说明：

- 第 1 行：开启函数 docstring；这个字符串成为 `firfilter.__doc__`。
- 第 2 行：“用 FIR 滤波器沿一个维度滤波数据。”这是摘要句。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:114` 至 `:116`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:113` 至 `:115`。

```python
    Filter a data sequence, `x`, using a digital filter. This works for many
    fundamental data types (including Object type). Please note, cuSignal
    doesn't support IIR filters presently, and this implementation is optimized
```

逐行对应说明：

- 第 1 行：说明对数据序列 `x` 施加数字滤波；本行句子延续到下行。
- 第 2 行：声称支持多种基本 dtype，并包括 Object；实际支持还受后面 `dtype.char` 检查和 CuPy 能力约束。
- 第 3 行：说明当时 cuSignal 不支持 IIR；`firfilter` 只有分子系数 `b`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:117` 至 `:117`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:116` 至 `:116`。

```python
    for large filtering operations (and inherently depends on fftconvolve)
```

逐行对应说明：

- 第 1 行：声称针对大规模滤波优化且依赖 `fftconvolve`。但函数体实际写的是 `cp.convolve`，两者不一致，不能以此行代替代码证据。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:119` 至 `:130`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:118` 至 `:129`。

```python
    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
```

逐行对应说明：

- 第 1 行：NumPy/SciPy 风格 docstring 的“参数”标题。
- 第 2 行：reStructuredText/NumPy doc 风格的标题下划线。
- 第 3 行：`b` 可接受能转成数组的对象。
- 第 4 行：`b` 是一维分子系数向量；对 FIR 而言就是冲激响应 taps。
- 第 5 行：`x` 是能按数组语义处理的输入。
- 第 6 行：`x` 可以是任意正维数的 N 维数组。
- 第 7 行：`axis` 是可选整数参数。
- 第 8 行：选定输入数组中执行线性滤波的轴，句子在下行继续。
- 第 9 行：固定其他维度后，对沿该轴的每个一维子数组分别滤波。
- 第 10 行：默认沿最后一轴滤波。
- 第 11 行：`zi` 是可选初始状态数组。
- 第 12 行：表示滤波器延迟元件在当前数据块开始时的状态。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:131` 至 `:133`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:130` 至 `:132`。

```python
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.
```

逐行对应说明：

- 第 1 行：N 维输入中，每条独立序列都可有自己的状态向量。
- 第 2 行：沿滤波轴的状态长度公式来自通用 `lfilter`。本 FIR 函数没有参数 `a`，实际代码使用 `len(b)-1`。
- 第 3 行：未给 `zi` 时假定初始静止（零状态）。`lfiltic` 是 SciPy 文档语境的参考，本函数体未调用它。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:135` 至 `:139`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:134` 至 `:138`。

```python
    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
```

逐行对应说明：

- 第 1 行：“返回值”标题。
- 第 2 行：返回值标题的下划线。
- 第 3 行：文档把主输出称为 `y`；函数体的局部变量名是 `out`。
- 第 4 行：`y` 是数字 FIR 滤波结果。
- 第 5 行：`zf` 是条件返回的最终状态数组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:140` 至 `:142`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:139` 至 `:141`。

```python
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """
```

逐行对应说明：

- 第 1 行：只有调用者传入 `zi` 时才返回 `zf`。
- 第 2 行：`zf` 保存当前块结束后对未来输出的延迟贡献。
- 第 3 行：结束 docstring。


## 4. 源代码逐行解释

### 4.1 导出语句逐行解释

顶层学习副本第 41-56 行，基准第 39-54 行：

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

`from ... import (` 开启括号包围的多行绝对导入，最后的 `)` 闭合它。中间每行是一个被绑定到 `cusignal` 顶层命名空间的公开符号，逗号允许继续列举。其中 `firfilter,` 是本算子的关键导出，因此用户可直接调用 `cusignal.firfilter`。其他名称只是同导入块的公开 API；`firfilter_zi` 也不是本函数直接调用的 helper。

子包学习副本第 14-29 行，基准第 14-28 行：

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

这一块语法与顶层导入相同，但绑定目标是 `cusignal.filtering` 子包，因而它提供 `cusignal.filtering.firfilter`。两个入口都指向 `filtering.py` 中的同一函数对象，没有额外包装或数学计算。

### 4.2 函数签名与输入检查

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:110` 至 `:110`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:109` 至 `:109`。

```python
def firfilter(b, x, axis=-1, zi=None):
```

逐行对应说明：

- 第 1 行：`def` 定义函数；`axis=-1` 和 `zi=None` 是默认参数；冒号开启缩进函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:144` 至 `:144`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:142` 至 `:142`。

```python
    b = cp.asarray(b)
```

逐行对应说明：

- 第 1 行：用 CuPy 数组语义规范化 `b`；左侧同名赋值覆盖局部变量原引用。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:146` 至 `:148`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:143` 至 `:144`。

```python
    if b.ndim != 1:
        raise ValueError("object of too small depth for desired array")
```

逐行对应说明：

- 第 1 行：`!=` 比较维数是否不等于 1；成立时执行缩进分支。
- 第 2 行：`raise` 抛出并终止当前调用；标量或多维 `b` 均被拒绝。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:151` 至 `:153`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:146` 至 `:147`。

```python
    if x.ndim == 0:
        raise ValueError("x must be at least 1-D")
```

逐行对应说明：

- 第 1 行：检查 `x` 是否为零维标量。注意此前没有 `cp.asarray(x)`，因而传入完全没有 `.ndim` 属性的普通 Python list 会先触发 `AttributeError`，这与 docstring 声称的广义 `array_like` 存在差异。
- 第 2 行：标量输入不被隐式扩为一维。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:156` 至 `:156`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:149` 至 `:149`。

```python
    inputs = [b, x]
```

逐行对应说明：

- 第 1 行：创建 list，供后面的可变参数展开 `*inputs` 使用。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:158` 至 `:162`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:150` 至 `:153`。

```python
    if zi is not None:
        # _linear_filter does not broadcast zi, but does do expansion of
        # singleton dims.
        zi = cp.asarray(zi)
```

逐行对应说明：

- 第 1 行：`is not None` 是身份检查，区分“未提供状态”与数值为零的有效状态。
- 第 2 行：原注释说明预期兼容行为：不做一般广播，但允许单例维扩展。注释句子下行续完。
- 第 3 行：“singleton dims”指长度为 1 的维度。
- 第 4 行：把初态转为 CuPy 数组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:164` 至 `:173`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:154` 至 `:159`。

```python
        if zi.ndim != x.ndim:
            raise ValueError("object of too small depth for desired array")
        expected_shape = list(x.shape)
        expected_shape[axis] = b.shape[0] - 1
        expected_shape = tuple(expected_shape)
        # check the trivial case where zi is the right shape first
```

逐行对应说明：

- 第 1 行：要求初态和输入轴数相等。
- 第 2 行：对维数不相等的 `zi` 抛错。
- 第 3 行：复制 `x.shape` 为可修改 list；原 tuple 不可就地替换元素。
- 第 4 行：将滤波轴长度设为 $L-1$。Python 列表索引本身支持合法负 `axis`。
- 第 5 行：把目标 shape 转回 tuple。
- 第 6 行：原注释：先检查 `zi` 已经完全符合 shape 的简单情形。


### 4.3 `zi` 单例维扩展

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:175` 至 `:177`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:160` 至 `:161`。

```python
        if zi.shape != expected_shape:
            strides = zi.ndim * [None]
```

逐行对应说明：

- 第 1 行：只对 shape 不完全相等的初态尝试扩展。
- 第 2 行：列表重复运算创建长度为 `zi.ndim` 的 stride 占位列表。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:179` 至 `:181`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:162` 至 `:163`。

```python
            if axis < 0:
                axis += zi.ndim
```

逐行对应说明：

- 第 1 行：判定是否使用 Python 负轴索引。
- 第 2 行：增强赋值，等价于 `axis = axis + zi.ndim`；将合法负轴归一化为非负轴。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:183` 至 `:183`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:164` 至 `:164`。

```python
            for k in range(zi.ndim):
```

逐行对应说明：

- 第 1 行：遍历轴索 $k=0,\ldots,\text{ndim}-1$。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:185` 至 `:187`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:165` 至 `:166`。

```python
                if k == axis and zi.shape[k] == expected_shape[k]:
                    strides[k] = zi.strides[k]
```

逐行对应说明：

- 第 1 行：当前是滤波轴且长度恰好符合 $L-1$ 时通过。
- 第 2 行：保留滤波轴的原始字节步长。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:189` 至 `:191`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:167` 至 `:168`。

```python
                elif k != axis and zi.shape[k] == expected_shape[k]:
                    strides[k] = zi.strides[k]
```

逐行对应说明：

- 第 1 行：`elif` 只在前一条件为假时判定；这里处理完全匹配的非滤波轴。
- 第 2 行：完全匹配轴仍保留原 stride。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:193` 至 `:195`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:169` 至 `:170`。

```python
                elif k != axis and zi.shape[k] == 1:
                    strides[k] = 0
```

逐行对应说明：

- 第 1 行：只允许非滤波轴的单例维扩展；滤波轴长度 1 不能用此分支代替 $L-1$。
- 第 2 行：stride 0 意味着沿该轴移动索引时内存地址不变，从而复用同一状态数据。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:197` 至 `:211`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:171` 至 `:178`。

```python
                else:
                    raise ValueError(
                        "Unexpected shape for zi: expected "
                        "%s, found %s." % (expected_shape, zi.shape)
                    )
            zi = cp.lib.stride_tricks.as_strided(zi, expected_shape, strides)
        inputs.append(zi)
    dtype = cp.result_type(*inputs)
```

逐行对应说明：

- 第 1 行：上述三种可接受条件都不满足时进入错误分支。
- 第 2 行：开始一个跨行的异常构造与抛出表达式。
- 第 3 行：括号内相邻字符串字面量会自动连接；本行是消息前缀。
- 第 4 行：旧式 `%` 字符串格式化，依次填入期望和实际 shape。
- 第 5 行：关闭第 172 行开始的 `ValueError(` 调用。
- 第 6 行：以新 shape 和 strides 构造视图。它不做一般数值广播复制，而是通过 stride 0 重复引用。
- 第 7 行：把处理后的 `zi` 追加到 dtype 推导列表。
- 第 8 行：`*` 把 list 展开为位置参数；`result_type` 按类型提升规则求能容纳所有输入的 dtype。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:214` 至 `:216`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:180` 至 `:181`。

```python
    if dtype.char not in "fdgFDGO":
        raise NotImplementedError("input type '%s' not supported" % dtype)
```

逐行对应说明：

- 第 1 行：成员测试；小写 `f/d/g` 表示浮点类，大写 `F/D/G` 表示复数类，`O` 表示 object。实际 CuPy 对某些类别的运算能力仍是额外约束。
- 第 2 行：对不支持的结果 dtype 报“未实现”，`% dtype` 把类型名填入消息。


### 4.4 卷积、初态叠加与输出分割

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:219` 至 `:221`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:183` 至 `:184`。

```python
    b = cp.array(b, dtype=dtype)
    x = cp.array(x, dtype=dtype, copy=False)
```

逐行对应说明：

- 第 1 行：将抽头转为共同 dtype；`cp.array` 得到 CuPy 数组。
- 第 2 行：将输入转为公共 dtype；`copy=False` 要求可能时不复制，但需要 dtype/设备转换时仍可分配。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:224` 至 `:224`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:186` 至 `:186`。

```python
    out_full = cp.apply_along_axis(lambda y: cp.convolve(b, y), axis, x)
```

逐行对应说明：

- 第 1 行：`lambda y: cp.convolve(b, y)` 定义一个匿名一参数函数；`apply_along_axis` 对 `x` 沿 `axis` 的每条一维切片调用它。`cp.convolve` 未给 `mode`，此调用按默认 `full` 卷积产生 $N+L-1$ 项。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:227` 至 `:227`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:188` 至 `:188`。

```python
    ind = out_full.ndim * [slice(None)]
```

逐行对应说明：

- 第 1 行：创建多维全切片 list；`slice(None)` 等价于 `:`。列表中的 slice 对象可安全复用，后面只替换某一个 list 元素。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:229` 至 `:233`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:189` 至 `:191`。

```python
    if zi is not None:
        ind[axis] = slice(zi.shape[axis])
        out_full[tuple(ind)] += zi
```

逐行对应说明：

- 第 1 行：有初态时执行前缀叠加。
- 第 2 行：`slice(stop)` 等价于 `0:stop`；选中滤波轴前 $L-1$ 项。
- 第 3 行：list 转 tuple 后作 N 维索引；`+=` 原地叠加广播后初态，对应 $c'[n]=c[n]+z_i[n]$。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:236` 至 `:238`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:193` 至 `:194`。

```python
    ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
    out = out_full[tuple(ind)]
```

逐行对应说明：

- 第 1 行：full 长度为 $N+L-1$，所以 stop 为 $(N+L-1)-L+1=N$；选中因果主输出。
- 第 2 行：取 full 卷积的前 $N$ 项；这是基于视图的基本切片语义。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:241` 至 `:241`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:196` 至 `:196`。

```python
    if zi is None:
```

逐行对应说明：

- 第 1 行：根据调用者是否传入初态决定返回结构。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:243` 至 `:243`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:197` 至 `:197`。

```python
        return out
```

逐行对应说明：

- 第 1 行：无 `zi` 时结束函数并返回单个数组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:245` 至 `:249`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:198` 至 `:200`。

```python
    else:
        ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
        zf = out_full[tuple(ind)]
```

逐行对应说明：

- 第 1 行：与第 196 行配对；此分支中 `zi` 必然不是 `None`。
- 第 2 行：构造从 $N$ 到轴末尾的切片；显式 `None` 表示没有 stop。
- 第 3 行：取出 full 结果剩余 $L-1$ 项为最终状态。因为初态已在第 191 行叠加，当 $N<L-1$ 时它也可能传播到 `zf`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:251` 至 `:251`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:201` 至 `:201`。

```python
        return out, zf
```

逐行对应说明：

- 第 1 行：Python 逗号返回会自动打包成二元 tuple。


## 5. 调用链与算法总结

### 5.1 调用链

```text
cusignal.firfilter(b, x, axis=-1, zi=None)
  → filtering.filtering.firfilter
    → cp.asarray(b)
    → 检查 b.ndim 和 x.ndim
    → [可选] cp.asarray(zi)
      → 检查/构造 expected_shape
      → [可选] cp.lib.stride_tricks.as_strided 扩展单例维
    → cp.result_type 确定公共 dtype
    → cp.apply_along_axis
      → 对每条一维切片调用 cp.convolve(b, y)
    → [可选] full 卷积前缀 += zi
    → 截取前 N 项为 out
    → [可选] 截取后 L-1 项为 zf
```

### 5.2 执行顺序的公式

固定其他轴后，对每条长度 $N$ 的一维序列 $x_s$：

$$
c_s[n]=\sum_{k=0}^{L-1}b[k]x_s[n-k],\qquad 0\le n<N+L-1,
$$

有 `zi` 时：

$$
c'_s[n]=
\begin{cases}
c_s[n]+z_{i,s}[n], & 0\le n<L-1,\\
c_s[n], & n\ge L-1.
\end{cases}
$$

最后：

$$
y_s=c'_s[0:N],\qquad z_{f,s}=c'_s[N:N+L-1].
$$

若没有 `zi`，则 $c'_s=c_s$，且 API 不返回 $z_f$。

### 5.3 底层机制界定

- **卷积**：本函数代码明示调用 `cp.convolve`。
- **FFT**：docstring 提到 `fftconvolve`，但函数体没有调用它。本阶段没有阅读第三方 CuPy 内部，因而不宣称 `cp.convolve` 在当前环境里使用 direct 还是 FFT 后端。
- **CUDA kernel**：`firfilter` 本身没有定义或直接启动项目自定义 kernel；GPU 执行由 CuPy API 承担。
- **内存视图**：`as_strided` 用 stride 0 扩展 `zi` 的非滤波单例轴，不需实体复制数据。
- **无 FFT/插值/线性代数 helper**：本算子没有直接使用这些项目机制。

## 6. 数学映射、边界、复杂度和阅读检查

### 6.1 数学公式与代码映射

| 数学/数组概念 | 代码位置（基准） | 实现关系 |
| --- | --- | --- |
| $y[n]=\sum b[k]x[n-k]$ | `filtering.py:186` | `cp.convolve(b, y)` 等价实现每条一维序列的线性卷积 |
| 沿指定轴独立滤波 | `filtering.py:186` | `cp.apply_along_axis(lambda y: cp.convolve(b, y), axis, x)` 固定其他维度并遍历一维切片 |
| FIR 状态长度 $L-1$ | `filtering.py:156-158` | 用 `b.shape[0]-1` 替换滤波轴 shape |
| $c'[0:L-1]=c[0:L-1]+z_i$ | `filtering.py:188-191` | 通过切片和原地加法叠加初态 |
| $y=c'[0:N]$ | `filtering.py:193-194` | 利用 full 长度反推 $N$ 并截取前缀 |
| $z_f=c'[N:N+L-1]$ | `filtering.py:199-200` | 截取 full 卷积尾部 |
| 状态在多条序列间广播 | `filtering.py:160-176` | 非滤波轴可以通过 stride 0 复用单例状态 |

### 6.2 边界处理和异常

1. `b.ndim != 1`：抛 `ValueError`。
2. `x.ndim == 0`：抛 `ValueError`；但普通 list 没有 `.ndim`，可能更早抛 `AttributeError`。
3. `zi.ndim != x.ndim`：抛 `ValueError`。
4. `zi` 滤波轴不是 $L-1$，或非滤波轴既不匹配也不是 1：抛包含期望/实际 shape 的 `ValueError`。
5. 结果 dtype 不在 `fdgFDGO` 字符集：抛 `NotImplementedError`。这意味着纯整数输入的 `result_type` 通常不被直接接受，需要浮点/复数输入使结果类型进入支持集。
6. 初始静止：`zi is None` 时卷积区间外按零贡献解释。
7. 输出边界：主输出不是 `same` 模式的中心截取，而是 `full` 结果的因果前 $N$ 项。
8. `axis` 的越界异常未在本函数中主动改写，由 list/CuPy 索引或 `apply_along_axis` 报出。

### 6.3 dtype 与数值稳定性

- `cp.result_type` 先联合 `b`、`x`、可选 `zi`，然后将 `b` 和 `x` 转为该 dtype，避免在卷积时出现两端 dtype 不一致。
- FIR 没有输出反馈，不会出现 IIR 极点导致的递归不稳定；但长卷积的累加仍有浮点舍入误差。
- 输入或系数幅度太大时仍可溢出为 `inf`；数学上 BIBO 稳定不等于有限精度永不溢出。
- 累加顺序由 CuPy 卷积实现决定，本函数没有 Kahan 求和或额外高精度累加。

### 6.4 时间、空间复杂度与瓶颈

设滤波轴长度为 $N$，taps 数为 $L$，其他轴组合出 $B$ 条独立序列。

- 按卷积定义直接计算时，算术复杂度为 $O(BNL)$。由于本阶段未读取 CuPy 内部，这是语义层的 direct 上界表述，不是对当前 CuPy 后端选择的实测结论。
- full 结果存储量为 $O(B(N+L-1))$；返回 `out`/`zf` 通过切片从它分割。
- `zi` 的 stride-0 扩展视图额外实体存储为 $O(1)$，但后续叠加仍要遍历对应输出元素。
- 可能瓶颈包括：对 $B$ 条切片的 `apply_along_axis` 调度、为每条创建 full 卷积尾部、以及当 $N,L$ 很大时的卷积本身。
- docstring 对“大规模优化”和 `fftconvolve` 的说法与当前函数体不一致，因而不能用该句宣称 $O(N\log N)$。

### 6.5 建议亲自阅读的顺序

1. **导出入口**：先看两个 `__init__.py`，确认 `cusignal.firfilter` 不是包装函数，而是直接导入同一函数对象。
2. **签名和 docstring**：标记 `b/x/axis/zi` 的 shape 契约，并找出 docstring 中 `a`、`lfiltic`、`fftconvolve` 与当前函数体的差异。
3. **初态 shape**：用一个 `x.shape=(2, 10)`、`axis=1`、`L=4` 的例子，手算 `expected_shape=(2, 3)`，再理解 `(1, 3)` 如何通过 stride 0 扩展。
4. **核心卷积**：将第 186 行写成数学和，确认 `apply_along_axis` 不是 N 维卷积。
5. **输出分割**：用 $N+L-1-L+1=N$ 验证第 193 行，再确认尾部长度是 $L-1$。
6. **分块续接**：把第一块的 `zf` 作为第二块的 `zi`，在纸上验证与整段一次卷积的因果前缀相同。

### 6.6 阅读自检

1. 为什么 `b` 会先 `asarray`，而 `x` 直到 dtype 检查后才 `cp.array`？这对 Python list 形式的 `x` 有什么影响？
2. `zi` 的滤波轴为什么必须是 $L-1$？为什么该轴不允许 singleton 广播？
3. stride 0 为什么能广播非滤波轴？
4. `cp.apply_along_axis` 中的 `y` 是最终输出吗？它与整个 `x` 是什么关系？
5. 为什么 `out_full.shape[axis] - len(b) + 1` 恰好等于输入长度 $N$？
6. `zi is None` 与传入全零 `zi` 的数值输出有何异同？返回结构有何不同？
7. 哪些代码证据支持“当前实现调用 `cp.convolve`”？为什么 docstring 不足以证明它调用 `fftconvolve`？
8. 这个函数哪些部分是 FIR 数学，哪些只是 dtype、shape、广播和 API 工程？

#### 参考答案 1

`b` 先在基准第 142 行执行 `cp.asarray(b)`，因为紧接着要检查 `b.ndim`。`x` 却在基准第 146 行直接访问 `x.ndim`，直到第 184 行才 `cp.array(x,...)`。因而具有 `.ndim` 的 NumPy/CuPy 数组可通过，普通 Python list 可能在第 146 行先抛 `AttributeError`，这与 docstring 的广义 `array_like` 说法存在差异。

#### 参考答案 2

长度 $L$ 的 FIR 在当前块结束后还有 $L-1$ 个 full-convolution 尾部位置，所以状态轴必须是 $L-1$；证据是基准 156-158 行的 `expected_shape[axis] = b.shape[0] - 1`。滤波轴中每个状态代表不同未来时刻的贡献，不能把一个值广播成所有 $L-1$ 个时刻；因而 165-170 行只允许非 axis 单例维广播。

#### 参考答案 3

stride 表示沿某轴增加一个索引时内存地址前进的字节数。基准第 170 行将单例非 axis 维的 stride 设为 0，使该维坐标变化时仍读取同一内存位置；第 176 行 `as_strided` 据此构造无实体复制的广播视图。

#### 参考答案 4

`lambda y: cp.convolve(b, y)` 中的 `y` 不是最终输出，而是 `cp.apply_along_axis` 从 N 维 `x` 中取出的一条一维切片。基准第 186 行对固定其他维度得到的每条切片独立卷积，再把结果组成 `out_full`。

#### 参考答案 5

对滤波轴长度 $N$ 和系数长度 $L$，full 卷积长度是 $N+L-1$。因而基准第 193 行的 stop 为

$$
(N+L-1)-L+1=N,
$$

恰好让第 194 行截取因果前 $N$ 项。

#### 参考答案 6

传入全零 `zi` 时，第 191 行加零，所以 `out` 数值与 `zi is None` 相同。但返回结构不同：`None` 在 196-197 行只返回 `out`；全零数组仍属于“有 `zi`”，在 199-201 行返回 `(out,zf)`。

#### 参考答案 7

基准第 186 行是直接代码证据：`cp.convolve(b, y)`。函数体中没有 `fftconvolve` 符号；只有 docstring 116 行提到它。docstring 是说明文本而不是执行路径，所以不足以证明当前实现调用 FFT。

#### 参考答案 8

FIR 数学核心是第 186 行的线性卷积、188-191 行的初态前缀叠加、193-201 行的 $y/zf$ 分割。`cp.asarray/cp.array`、`result_type`、shape 检查、负 axis 归一化、stride-0 广播和异常文本属于数组/API/GPU 工程；它们保障数学在 N 维 CuPy 数组上被正确调度，但不是卷积公式本身。

## 7. 源码完整性记录

- 本文完整摘录了基准 `firfilter` 的签名、全部 docstring、全部函数体和必要导出块，没有使用省略号或伪代码。
- docstring 中没有示例，因而不存在被遗漏的 docstring 示例行。
- 直接调用只涉及第三方 CuPy API，没有需额外摘录的本项目 helper/kernel。
- 学习副本中只插入了独立的 `# <学习注释：正文>` 行；完整性复核结果见本阶段交付记录。
