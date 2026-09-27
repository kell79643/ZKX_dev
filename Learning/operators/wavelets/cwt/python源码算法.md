# cwt Python 源码算法

## 1. 代码定位与接口概览

### 1.1 本阶段边界

本阶段只学习 cuSignal 23.08.00 的 Python/CuPy `cwt` 实现。直接依赖只展开到通用 `convolve`；它下一层的自动选择、FFT 和 CUDA 路径只读取解释 `cwt` 所需的符号，不扩展到 `convolve2d`、高阶卷积或其他小波算子。

开始前已逐文件执行完整性门禁。对以下学习副本删除所有合法独立学习注释行（Python 的 `# <学习注释：...>`，以及 kernel 字符串中的 `// <学习注释：...>`）后，均与 `ZKX/cusignal-23.08.00` 只读基准逐行一致：

- `python/cusignal/__init__.py`
- `python/cusignal/wavelets/__init__.py`
- `python/cusignal/wavelets/wavelets.py`
- `python/cusignal/convolution/convolve.py`
- `python/cusignal/convolution/_convolution_cuda.py`

### 1.2 公开路径与准确行号

| 作用 | 学习副本位置 | 只读基准位置 |
| --- | --- | --- |
| 顶层公开 API | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:105` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:100` |
| wavelets 包公开 API | `Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:17` | `ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14` |
| `convolve` 导入 | `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:19` | `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:17` |
| `cwt` 定义 | `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:287` | `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:259` |
| `convolve` 直接 helper | `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:33` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:32` |

### 1.3 函数签名与数据契约

```python
cwt(data, wavelet, widths)
```

| 项目 | 语义 |
| --- | --- |
| `data` | 文档要求 `(N,) ndarray`。实际代码依赖 `len(data)`，并把它传入 CuPy 卷积；预期是一维、非空、可转为 CuPy 数组的数据。 |
| `wavelet` | callable，调用约定为 `wavelet(length, width)`，返回长度为 `length` 的一维数组。代码会先额外调用 `wavelet(1, 1)` 探测 dtype。 |
| `widths` | 长度为 $M$ 的可迭代尺度/宽度序列。每个元素随后经 `int(width)` 截断；理论上应保证截断后为正。 |
| 返回 shape | `(len(widths), len(data)) = (M, N)`；每行对应一个 `width`，每列对应一个输入位置。 |
| 返回 dtype | 只由 `wavelet(1,1)` 决定：复数 dtype 字符 `F/D/G` → `complex128`，否则 → `float64`。它不根据 `data.dtype` 决定。 |

重要边界：若 `data` 是复数而 `wavelet` 返回实数，输出仍被分配为 `float64`，复卷积结果写入时可能丢失虚部；这是由源码可直接推出的风险，不是理论 CWT 的要求。

## 2. 当前算子的完整相关源码

以下摘录全部来自只读基准，保持原文、原顺序和原始行号；没有用省略号删减当前符号。学习注释只存在于学习副本，不混入原文摘录。

### 2.1 必要导入与公开导出

`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:17`

```python
from ..convolution.convolve import convolve
```

`ZKX/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14`

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:100`

```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

### 2.2 `cwt` 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:259`

```python
def cwt(data, wavelet, widths):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = cusignal.convolve(data, wavelet(length,
                                    width[ii]), mode='same')

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cp.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = cp.cos(2 * cp.pi * 7 * t) + cusignal.gausspulse(t - 0.4, fc=2)
    >>> widths = cp.arange(1, 31)
    >>> cwtmatr = cusignal.cwt(sig, cusignal.ricker, widths)
    >>> plt.imshow(abs(cp.asnumpy(cwtmatr)), extent=[-1, 1, 31, 1],
                   cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())
    >>> plt.show()

    """
    if cp.asarray(wavelet(1, 1)).dtype.char in "FDG":
        dtype = cp.complex128
    else:
        dtype = cp.float64

    output = cp.empty([len(widths), len(data)], dtype=dtype)

    for ind, width in enumerate(widths):
        N = np.min([10 * int(width), len(data)])
        wavelet_data = cp.conj(wavelet(N, int(width)))[::-1]
        output[ind, :] = convolve(data, wavelet_data, mode="same")
    return output
```

### 2.3 直接 helper `convolve` 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:32`

```python
def convolve(
    in1,
    in2,
    mode="full",
    method="auto",
):
    """
    Convolve two N-dimensional arrays.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).

    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve

    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force `method=direct` (more detail
    in `choose_conv_method` docstring).

    Examples
    --------
    Smooth a square pulse using a Hann window:

    >>> import cusignal
    >>> import cupy as cp
    >>> sig = cp.repeat(cp.asarray([0., 1., 0.]), 100)
    >>> win = cusignal.hann(50)
    >>> filtered = cusignal.convolve(sig, win, mode='same') / cp.sum(win)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(cp.asnumpy(sig))
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(cp.asnumpy(win))
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(cp.asnumpy(filtered))
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """

    volume = cp.asarray(in1)
    kernel = cp.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative
        # order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == "auto":
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == "fft":
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = cp.result_type(volume, kernel)
        if result_type.kind in {"u", "i"}:
            out = cp.around(out)
        return out.astype(result_type)
    elif method == "direct":
        if volume.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")

        swapped_inputs = (mode != "valid") and (kernel.size > volume.size)

        if swapped_inputs:
            volume, kernel = kernel, volume

        return _convolution_cuda._convolve(volume, kernel, True, swapped_inputs, mode)

    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")
```

## 3. docstring 逐行翻译与解释

### 3.1 `cwt` docstring（基准 260–309）

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:288` 至 `:289`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:260` 至 `:261`。

```python
    """
    Continuous wavelet transform.
```

逐行对应说明：

- 第 1 行：开始函数文档字符串。
- 第 2 行：标题：连续小波变换。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:291` 至 `:294`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:263` 至 `:266`。

```python
    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.
```

逐行对应说明：

- 第 1 行：说明在 `data` 上执行 CWT；句子在下一行续写。
- 第 2 行：指定使用用户给出的 `wavelet`，并把计算描述成卷积。
- 第 3 行：说明卷积核由小波函数产生；句子仍未结束。
- 第 4 行：小波 callable 由宽度和长度两个参数刻画。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:296` 至 `:307`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:268` 至 `:279`。

```python
    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy docstring 标题下划线。
- 第 3 行：`data` 是长度 $N$ 的一维数组。
- 第 4 行：它是被变换的输入数据。
- 第 5 行：`wavelet` 必须是函数/callable。
- 第 6 行：callable 应接受两个参数。
- 第 7 行：第一个参数指定返回向量的点数；下一行补全约束。
- 第 8 行：明确返回长度必须等于传入 `length`。注意这里文档括号顺序与实际调用一致。
- 第 9 行：第二个参数 `width` 控制小波宽度。
- 第 10 行：举例：可把宽度理解为高斯标准差，并指向 `ricker`。
- 第 11 行：`ricker` 满足上述 callable 契约。
- 第 12 行：`widths` 是长度 $M$ 的一维序列。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:308` 至 `:308`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:280` 至 `:280`。

```python
        Widths to use for transform.
```

逐行对应说明：

- 第 1 行：每个元素对应输出的一行。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:310` 至 `:313`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:282` 至 `:285`。

```python
    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：NumPy docstring 标题下划线。
- 第 3 行：返回 $M\times N$ 数组。
- 第 4 行：明确第一维是宽度数，第二维是数据长度。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:315` 至 `:317`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:287` 至 `:289`。

```python
    Notes
    -----
    ::
```

逐行对应说明：

- 第 1 行：备注章节标题。
- 第 2 行：标题下划线。
- 第 3 行：reStructuredText 的后续文字块标记。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:319` 至 `:321`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:291` 至 `:293`。

```python
        length = min(10 * width[ii], len(data))
        cwt[ii,:] = cusignal.convolve(data, wavelet(length,
                                    width[ii]), mode='same')
```

逐行对应说明：

- 第 1 行：伪代码说明有限小波长度取十倍宽度与数据长度的较小值。实际代码还执行 `int(width)`。
- 第 2 行：伪代码开始：当前宽度行由输入与生成小波的 `same` 卷积得到。
- 第 3 行：补全 wavelet 调用和 `same` 参数。此备注没有展示实际代码中的共轭反转。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:323` 至 `:334`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:295` 至 `:306`。

```python
    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cp.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = cp.cos(2 * cp.pi * 7 * t) + cusignal.gausspulse(t - 0.4, fc=2)
    >>> widths = cp.arange(1, 31)
    >>> cwtmatr = cusignal.cwt(sig, cusignal.ricker, widths)
    >>> plt.imshow(abs(cp.asnumpy(cwtmatr)), extent=[-1, 1, 31, 1],
                   cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：标题下划线。
- 第 3 行：导入 cuSignal。
- 第 4 行：导入 CuPy 并使用 `cp` 别名。
- 第 5 行：导入绘图库。
- 第 6 行：在 $[-1,1)$ 生成 200 个等距 GPU 样本。
- 第 7 行：构造 7 Hz 余弦与移位高斯脉冲之和，演示同时含周期与瞬态的信号。
- 第 8 行：创建整数宽度 1 到 30。
- 第 9 行：用 Ricker 小波计算 CWT 矩阵。
- 第 10 行：把 GPU 结果转回 NumPy，取绝对值并开始绘制尺度图；纵轴范围反向显示 31 到 1。
- 第 11 行：设置色图、自动纵横比和颜色上限。
- 第 12 行：设置对称颜色下限并结束 `imshow` 调用；但数据已取绝对值，负下限不会对应负数据。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:335` 至 `:335`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:307` 至 `:307`。

```python
    >>> plt.show()
```

逐行对应说明：

- 第 1 行：显示图像。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:337` 至 `:337`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:309` 至 `:309`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束文档字符串。


### 3.2 `convolve` docstring（基准 38–120）

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:39` 至 `:40`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:38` 至 `:39`。

```python
    """
    Convolve two N-dimensional arrays.
```

逐行对应说明：

- 第 1 行：开始 `convolve` 文档字符串。
- 第 2 行：标题：两个 N 维数组的卷积。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:42` 至 `:43`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:41` 至 `:42`。

```python
    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.
```

逐行对应说明：

- 第 1 行：输出大小由 `mode` 决定；第 42 行补全句子。
- 第 2 行：指向 `mode` 参数。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:45` 至 `:52`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:44` 至 `:51`。

```python
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：标题下划线。
- 第 3 行：第一个类数组输入。
- 第 4 行：第一输入说明。
- 第 5 行：第二个类数组输入。
- 第 6 行：第二输入必须与第一输入维数相同。
- 第 7 行：输出模式，默认值由签名给出为 `full`。
- 第 8 行：下列三项解释输出大小。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:54` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:53` 至 `:64`。

```python
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.
```

逐行对应说明：

- 第 1 行：完整线性卷积模式。
- 第 2 行：输出包含所有线性卷积位置。
- 第 3 行：补全说明，并指出它是默认模式。
- 第 4 行：只保留完全重叠位置。
- 第 5 行：开始解释不依赖填充的输出。
- 第 6 行：`valid` 不依赖零填充，且两输入之一必须逐维不小于另一输入。
- 第 7 行：补全逐维尺寸约束。
- 第 8 行：与第一输入同大小模式。
- 第 9 行：输出尺寸等于 `in1`，从完整卷积中央取得。
- 第 10 行：补全居中裁剪语义。
- 第 11 行：卷积算法选择参数。
- 第 12 行：下列三项解释计算方法。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:67` 至 `:75`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:66` 至 `:74`。

```python
        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).
```

逐行对应说明：

- 第 1 行：直接法。
- 第 2 行：按卷积定义直接累加。
- 第 3 行：补全直接法句子。
- 第 4 行：FFT 法。
- 第 5 行：使用傅里叶变换计算。
- 第 6 行：具体调用 `fftconvolve`。
- 第 7 行：自动选择。
- 第 8 行：根据估计选择 direct 或 FFT。
- 第 9 行：补全并指出默认采用自动选择。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:77` 至 `:81`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:76` 至 `:80`。

```python
    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：标题下划线。
- 第 3 行：返回数组。
- 第 4 行：返回完整线性卷积的某个子集。
- 第 5 行：补全返回值说明。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:83` 至 `:86`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:82` 至 `:85`。

```python
    See Also
    --------
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve
```

逐行对应说明：

- 第 1 行：相关 API 章节。
- 第 2 行：标题下划线。
- 第 3 行：指向自动选择 helper。
- 第 4 行：指向 FFT 卷积实现。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:88` 至 `:95`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:87` 至 `:94`。

```python
    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force `method=direct` (more detail
    in `choose_conv_method` docstring).
```

逐行对应说明：

- 第 1 行：备注章节标题。
- 第 2 行：标题下划线。
- 第 3 行：说明 `convolve`/`correlate` 默认自动选择。
- 第 4 行：使用预计算模型挑选较快方法。
- 第 5 行：helper 也能选择实际计时，但 `cwt` 没有开启测量。
- 第 6 行：FFT 使用浮点运算，存在适用限制。
- 第 7 行：某些 dtype/精度风险会强制 direct。
- 第 8 行：细节由选择 helper 说明。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:97` 至 `:99`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:96` 至 `:98`。

```python
    Examples
    --------
    Smooth a square pulse using a Hann window:
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：标题下划线。
- 第 3 行：示例目标：用 Hann 窗平滑方波。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:101` 至 `:105`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:100` 至 `:104`。

```python
    >>> import cusignal
    >>> import cupy as cp
    >>> sig = cp.repeat(cp.asarray([0., 1., 0.]), 100)
    >>> win = cusignal.hann(50)
    >>> filtered = cusignal.convolve(sig, win, mode='same') / cp.sum(win)
```

逐行对应说明：

- 第 1 行：导入 cuSignal。
- 第 2 行：导入 CuPy。
- 第 3 行：创建 0—1—0 的分段常量信号，每段重复 100 次。
- 第 4 行：生成长度 50 的 Hann 窗。
- 第 5 行：执行 `same` 卷积并除以窗和，实现归一化平滑。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:107` 至 `:118`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:106` 至 `:117`。

```python
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(cp.asnumpy(sig))
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(cp.asnumpy(win))
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(cp.asnumpy(filtered))
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
```

逐行对应说明：

- 第 1 行：导入绘图库。
- 第 2 行：创建三个共享横轴的子图。
- 第 3 行：绘制原信号。
- 第 4 行：设置原信号标题。
- 第 5 行：设置原信号图边距。
- 第 6 行：绘制 Hann 窗。
- 第 7 行：设置窗标题。
- 第 8 行：设置窗图边距。
- 第 9 行：绘制平滑结果。
- 第 10 行：设置结果标题。
- 第 11 行：设置结果图边距。
- 第 12 行：自动调整子图布局。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:119` 至 `:119`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:118` 至 `:118`。

```python
    >>> fig.show()
```

逐行对应说明：

- 第 1 行：显示图像。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:121` 至 `:121`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:120` 至 `:120`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束文档字符串。


## 4. 源代码逐行解释

### 4.1 导入、导出与 `cwt` 可执行代码

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:287` 至 `:287`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:259` 至 `:259`。

```python
def cwt(data, wavelet, widths):
```

逐行对应说明：

- 第 1 行：`def` 创建函数对象；三个形参均为必填，没有默认值、类型注解或关键字限制。冒号开始缩进函数体。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:339` 至 `:341`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:310` 至 `:311`。

```python
    if cp.asarray(wavelet(1, 1)).dtype.char in "FDG":
        dtype = cp.complex128
```

逐行对应说明：

- 第 1 行：先调用一次小波，再转为 CuPy 数组；`.dtype.char` 是 NumPy dtype 字符，`in` 检查它是否为复浮点类型字符。这个探测调用可能有额外 GPU 分配/编译开销。
- 第 2 行：复小波统一选择双精度复数输出。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:342` 至 `:344`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:312` 至 `:313`。

```python
    else:
        dtype = cp.float64
```

逐行对应说明：

- 第 1 行：dtype 字符不在集合时进入实数分支。
- 第 2 行：所有非复小波统一选择双精度实数输出。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:347` 至 `:347`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:315` 至 `:315`。

```python
    output = cp.empty([len(widths), len(data)], dtype=dtype)
```

逐行对应说明：

- 第 1 行：`len` 得到 $M,N$；列表作为 shape。`cp.empty` 在 GPU 上分配但不初始化，因为随后循环覆盖每一行。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:350` 至 `:356`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:317` 至 `:320`。

```python
    for ind, width in enumerate(widths):
        N = np.min([10 * int(width), len(data)])
        wavelet_data = cp.conj(wavelet(N, int(width)))[::-1]
        output[ind, :] = convolve(data, wavelet_data, mode="same")
```

逐行对应说明：

- 第 1 行：`enumerate` 逐项产生从 0 开始的行索引 `ind` 和当前宽度 `width`。循环执行 $M$ 次。
- 第 2 行：`int(width)` 向零截断；乘 10 得候选核长，与输入长度组成列表，`np.min` 取较小值。这里的局部变量 `N` 是核长，不是 docstring 中固定的数据长度符号。
- 第 3 行：再次调用 `wavelet`；`cp.conj` 逐元素复共轭；切片 `[::-1]` 以步长 -1 反转一维顺序。共轭反转让卷积实现相关。
- 第 4 行：`output[ind, :]` 选择整行；调用线性卷积，未传 `method`，所以默认 `auto`；`same` 返回与第一输入 `data` 同长，再写入该行并按输出 dtype 转换。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:358` 至 `:358`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py:321` 至 `:321`。

```python
    return output
```

逐行对应说明：

- 第 1 行：循环完成后返回全部宽度的二维系数矩阵。


### 4.2 `convolve` 可执行代码（完整逐行）

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:33` 至 `:38`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:32` 至 `:37`。

```python
def convolve(
    in1,
    in2,
    mode="full",
    method="auto",
):
```

逐行对应说明：

- 第 1 行：开始多行函数签名。
- 第 2 行：第一输入；在 `cwt` 中是 `data`。
- 第 3 行：第二输入；在 `cwt` 中是共轭反转的小波。
- 第 4 行：默认完整输出，但 `cwt` 显式覆盖为 `same`。
- 第 5 行：默认自动选择计算方法；`cwt` 没有覆盖。
- 第 6 行：结束签名并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:124` 至 `:125`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:122` 至 `:123`。

```python
    volume = cp.asarray(in1)
    kernel = cp.asarray(in2)
```

逐行对应说明：

- 第 1 行：将第一输入转为 CuPy 数组；已有 CuPy 数组通常避免复制。
- 第 2 行：将小波核转为 CuPy 数组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:127` 至 `:127`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:125` 至 `:125`。

```python
    if volume.ndim == kernel.ndim == 0:
```

逐行对应说明：

- 第 1 行：链式比较检查二者都是零维标量。正常 `cwt` 一维输入不满足。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:128` 至 `:128`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:126` 至 `:126`。

```python
        return volume * kernel
```

逐行对应说明：

- 第 1 行：标量卷积退化为乘法并提前返回。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:129` 至 `:130`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:127` 至 `:128`。

```python
    elif volume.ndim != kernel.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
```

逐行对应说明：

- 第 1 行：若维数不同则报错。
- 第 2 行：抛出维数不一致异常；`cwt` 要求两者均为一维。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:132` 至 `:135`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:130` 至 `:133`。

```python
    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative
        # order doesn't have any effect on output
        volume, kernel = kernel, volume
```

逐行对应说明：

- 第 1 行：只在 `valid` 模式检查逐维大小；`cwt` 使用 `same`，helper 立即返回 `False`。
- 第 2 行：原注释：卷积满足交换律。
- 第 3 行：原注释：交换输入不改变完整数学卷积。
- 第 4 行：Python 元组赋值同时交换两个引用；`cwt` 的 `same` 路径在这里不交换。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:138` 至 `:139`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:135` 至 `:136`。

```python
    if method == "auto":
        method = choose_conv_method(volume, kernel, mode=mode)
```

逐行对应说明：

- 第 1 行：`cwt` 默认命中该分支。
- 第 2 行：根据 dtype、精度风险和形状成本模型返回字符串 `direct` 或 `fft`。`measure` 保持默认 `False`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:142` 至 `:144`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:138` 至 `:140`。

```python
    if method == "fft":
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = cp.result_type(volume, kernel)
```

逐行对应说明：

- 第 1 行：自动结果为 FFT 时进入频域路径。
- 第 2 行：计算完整线性卷积并按 `same` 居中裁剪。实数组使用 RFFT，复数组使用复 FFT。
- 第 3 行：按 NumPy/CuPy 类型提升规则计算两输入的共同结果 dtype。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:145` 至 `:146`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:141` 至 `:142`。

```python
        if result_type.kind in {"u", "i"}:
            out = cp.around(out)
```

逐行对应说明：

- 第 1 行：若共同 dtype 是无符号或有符号整数，FFT 浮点误差需要舍入。`cwt` 小波通常为浮点，不走此分支。
- 第 2 行：把接近整数的 FFT 输出四舍五入。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:147` 至 `:147`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:143` 至 `:143`。

```python
        return out.astype(result_type)
```

逐行对应说明：

- 第 1 行：转回共同 dtype 并返回；之后写入 `cwt` 预分配 dtype。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:149` 至 `:149`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:144` 至 `:144`。

```python
    elif method == "direct":
```

逐行对应说明：

- 第 1 行：自动结果为直接法时进入 CUDA 直接卷积分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:150` 至 `:151`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:145` 至 `:146`。

```python
        if volume.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")
```

逐行对应说明：

- 第 1 行：直接后端只实现一维；正常 `cwt` 的一维数据通过。
- 第 2 行：多维输入选择 direct 时抛错。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:153` 至 `:153`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:148` 至 `:148`。

```python
        swapped_inputs = (mode != "valid") and (kernel.size > volume.size)
```

逐行对应说明：

- 第 1 行：对非 `valid` 模式，若核比数据长则记录需要交换。`cwt` 已把核长限制为不超过数据长，因此通常为 `False`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:155` 至 `:156`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:150` 至 `:151`。

```python
        if swapped_inputs:
            volume, kernel = kernel, volume
```

逐行对应说明：

- 第 1 行：判断是否真的需要交换。
- 第 2 行：交换以满足后端尺寸假设。正常 `cwt` 通常不执行。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:158` 至 `:158`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:153` 至 `:153`。

```python
        return _convolution_cuda._convolve(volume, kernel, True, swapped_inputs, mode)
```

逐行对应说明：

- 第 1 行：调用 CUDA 后端；`True` 表示卷积而非相关。注意 `cwt` 已经先自行共轭反转，所以这里仍应执行卷积。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolve.py:160` 至 `:161`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:155` 至 `:156`。

```python
    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")
```

逐行对应说明：

- 第 1 行：`method` 既非 `fft` 也非 `direct` 时进入错误分支。
- 第 2 行：抛出可接受方法标志列表；相邻两个字符串字面量在编译期自动拼接。


## 5. 调用链与算法总结

### 5.1 从公开 API 到最终计算

```text
cusignal.cwt
  → wavelets.wavelets.cwt
    → wavelet(1, 1) 探测实/复 dtype
    → 对每个 width：
       → L = min(10 * int(width), len(data))
       → wavelet(L, int(width))
       → cp.conj(...)[::-1]
       → convolve(data, wavelet_data, mode="same", method="auto")
          → choose_conv_method
             ├─ "direct" → _convolution_cuda._convolve → fatbin CUDA kernel
             └─ "fft"    → fftconvolve → FFT × FFT → IFFT → 中心裁剪
    → 返回 (M, N) output
```

### 5.2 下一层必要机制定位

| 符号 | 基准位置 | 与 `cwt` 的关系 |
| --- | --- | --- |
| `choose_conv_method` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:418–523` | `method="auto"` 的选择器；数值数组最终用 `_fftconv_faster` 成本模型判断。 |
| `_fftconv_faster` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:137–174` | 比较估计的 direct 与 FFT 成本；`same` 下输出 shape 取第一输入 shape。 |
| `fftconvolve` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolve.py:159–319` | 实数走 `rfftn → 频谱乘法 → irfftn`，复数走 `fftn → 乘法 → ifftn`；`same` 用 `_centered` 裁剪。 |
| `_convolution_cuda._convolve` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:408–463` | direct Python 后端：提升 dtype、计算输出 shape、分配 GPU 输出并调用 `_convolve_gpu`。 |
| `_convolve_gpu` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:230–276` | 加载/缓存与 dtype 对应的 fatbin `convolve` kernel，组织 grid/block 和参数后启动。 |

这里没有在 Python 字符串中定义 `cwt` 专用 CUDA kernel。direct 路径使用通用预编译卷积 fatbin；FFT 路径使用 CuPy FFT。不要把 dtype 探测、输出分配或 method 选择误称为小波数学核心。

### 5.3 按执行顺序总结

1. 以一次额外的最小 wavelet 调用判断输出是实数还是复数。
2. 一次性分配完整 $M\times N$ GPU 输出。
3. 宽度循环是 Python 串行循环；每轮创建有限长小波。
4. 共轭反转把离散相关变为线性卷积。
5. 通用卷积入口按当前 $N$、$L_j$ 和 dtype 每轮重新选择 direct/FFT。
6. `same` 保留每个尺度 $N$ 个位置，写入对应行。

## 6. 数学映射、边界、复杂度和阅读检查

### 6.1 数学公式到代码

连续 CWT：

$$
W_x(a,b)=\frac{1}{\sqrt a}\int x(t)\psi^*\left(\frac{t-b}{a}\right)dt.
$$

代码在第 $j$ 个宽度使用有限序列 $\psi_j[k]$，实现

$$
Y_j[m]=\sum_k x[k]\,\psi_j^*[k-m]
=\left(x*\operatorname{reverse}(\psi_j^*)\right)[m].
$$

| 数学对象 | 代码落实 |
| --- | --- |
| 尺度集合 $a_j$ | `for ind, width in enumerate(widths)` |
| 有限子小波 $\psi_j$ | `wavelet(N, int(width))` |
| 复共轭 $^*$ | `cp.conj(...)` |
| 时间反转 | `[::-1]` |
| 对所有平移求系数 | `convolve(..., mode="same")` 一次生成整行 |
| 系数矩阵 $W[j,m]$ | `output[ind, :]` |

`1/\sqrt a` 是否存在不由 `cwt` 添加，而由 `wavelet` callable 自己决定。例如使用兼容的 `morlet2` 或 `ricker` 时，其函数定义负责相应归一化。

### 6.2 dtype、shape、异常与数值边界

- `data` 或小波不是一维时，direct 路径会拒绝多维；文档契约本来就限定 `(N,)`。
- `widths` 为空时，`cp.empty([0, N])` 可形成空行矩阵，循环不执行。
- 宽度经 `int` 向零截断；不同浮点宽度可能变成相同整数，零/负值没有显式前置校验。
- 核长限制为 $L_j=\min(10\operatorname{int}(width_j),N)$，属于截断近似。
- `same` 是完整零扩展线性卷积的居中部分；边缘系数依赖缺失样本的零扩展。
- 输出 dtype 仅探测小波。复输入配实小波可能丢失虚部；低精度输入也统一写成 64 位输出。
- `wavelet(1,1)` 和每尺度正式 wavelet 调用必须返回 CuPy 可处理的数据；callable 的副作用会发生 $M+1$ 次。
- FFT 路径可能带浮点舍入；整数结果会显式 `around`，浮点/复数不做额外误差修正。
- 代码不检查可容许性、零均值、尺度归一化、Nyquist 混叠或影响锥。

### 6.3 时间与空间复杂度

令 $L_j=\min(10\lfloor width_j\rfloor,N)$。

- 小波生成与共轭反转：$O(\sum_j L_j)$ 时间；每轮至少需要 $O(L_j)$ 临时空间。
- direct 路径：按卷积定义约 $O(\sum_j NL_j)$ 时间。
- FFT 路径：若填充长度 $P_j\ge N+L_j-1$，约 $O(\sum_j P_j\log P_j)$ 时间和 $O(P_j)$ 临时频域空间。
- 持久输出：$O(MN)$，通常是主要显存占用。
- Python 外层有 $M$ 次循环、$M$ 次 method 选择和 $M$ 次 kernel/FFT 调度；尺度很多且核较短时，调度开销可能明显。
- 每轮重新生成 wavelet，没有跨调用或跨重复宽度缓存。

### 6.4 建议亲自阅读的顺序

1. 先读 `wavelets.py:310–315`：观察 dtype 为什么只取决于 wavelet。
2. 再读 `wavelets.py:317–321`：逐句把宽度循环映射到尺度、子小波、相关和输出行。
3. 读 `convolve.py:122–156`：重点观察 `cwt` 实际传入的 `same + auto` 会跳过哪些分支。
4. 读 `choose_conv_method:488–523`：区分精度保护与性能估算。
5. 分别读 `fftconvolve:282–313` 和 `_convolution_cuda._convolve:408–463`：对比频域与直接路径。
6. 回到阶段一的卷积等价公式，亲自证明为何必须同时有 `cp.conj` 和 `[::-1]`。

### 6.5 阅读自检

1. `wavelet` 为什么会被调用 $M+1$ 次而不是 $M$ 次？
2. `width=2.9` 最终传入 wavelet 的宽度是多少？核长候选是多少？
3. 哪两项操作共同把卷积变成小波相关？少掉其中一项对复小波意味着什么？
4. 为什么 `same` 控制输出长度却不能消除边界效应？
5. `cwt` 在什么条件下进入 FFT，什么条件下进入 CUDA direct？
6. `data` 为 `complex64`、wavelet 为实数时，当前输出 dtype 是什么？风险是什么？
7. 数学公式中的尺度归一化由 `cwt` 哪一行完成？如果找不到，应由谁负责？
8. 输出显存为何是 $O(MN)$，而非只与最长小波长度有关？

### 6.6 参考答案

1. 第一次 `wavelet(1, 1)` 用于 dtype 探测，随后宽度循环对 $M$ 个宽度各调用一次，所以总计 $M+1$ 次；依据基准 `wavelets.py:310`、`317–319`。
2. `int(2.9)` 为 2，候选核长为 $10\times2=20$，最终再与 `len(data)` 取较小值；依据基准 `wavelets.py:318–319`。
3. `cp.conj(...)` 完成复共轭，`[::-1]` 完成时间反转；二者合起来构造 $\operatorname{reverse}(\psi^*)$。少掉共轭会变成错误的非共轭复匹配，少掉反转则不再等价于相关形式。
4. `same` 只从完整线性卷积中居中裁出 $N$ 个结果；信号外仍按零处理，所以两端缺失样本造成的边界误差仍在。依据基准 `wavelets.py:320` 和 `convolve.py:60–62`。
5. `convolve.py:135–153` 先由 `choose_conv_method` 返回 `fft` 或 `direct`；FFT 路径调用 `fftconvolve`，direct 路径调用 `_convolution_cuda._convolve`。选择取决于 dtype 精度保护和 `_fftconv_faster` 的形状成本模型。
6. 输出由实 wavelet 探测为 `float64`，而卷积可能产生复数；写入实数组时虚部可能丢失。这源于基准 `wavelets.py:310–315` 只检查 wavelet dtype、不检查 `data.dtype`。
7. `cwt` 中找不到尺度归一化语句；它由 `wavelet(length, width)` callable 自己负责。依据基准 `wavelets.py:319`。
8. 输出为所有 $M$ 个尺度、每个尺度全部 $N$ 个位置的矩阵，元素总数固定为 $MN$；最长小波只决定单轮临时核和单个系数的累加成本。

## 7. 阶段二完整性复核

- 学习副本只增加独立的固定格式学习注释；没有改写、删除、移动或格式化原代码。
- 剔除 `# <学习注释：...>` 与 kernel 字符串中的 `// <学习注释：...>` 后，涉及文件与只读基准逐行一致。
- `ZKX/cusignal-23.08.00` 未被修改。
- 本阶段没有在本地或 ZQ500 上运行 Python、CUDA、构建或测试；结论来自源码静态核对。
- 尚未进入 `cusignal_cpp` 阶段。
