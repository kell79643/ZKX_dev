# `correlate` Python 源码算法

## 1. 代码定位与接口概览

### 1.1 范围与版本

- 学习对象：cuSignal 23.08.00 的一维/N 维公开 API `correlate`，不包括独立算子 `correlate2d` 和 helper `correlation_lags`。
- 顶层导出：`Learning/cusignal-23.08.00/python/cusignal/__init__.py:29`。
- 子包导出：`Learning/cusignal-23.08.00/python/cusignal/convolution/__init__.py:22`。
- 学习副本定义：`Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:23-169`。
- 只读基准定义：`ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:23-155`。
- 符号：`correlate(in1, in2, mode="full", method="auto")`。

阶段二开始前，已对下列相关文件执行完整性检查：从学习副本中忽略且仅忽略符合 `# <学习注释：...>` 或 `// <学习注释：...>` 的独立行后，与 `ZKX/cusignal-23.08.00` 基准逐行一致：

- `python/cusignal/__init__.py`
- `python/cusignal/convolution/__init__.py`
- `python/cusignal/convolution/correlate.py`
- `python/cusignal/convolution/convolution_utils.py`
- `python/cusignal/convolution/convolve.py`
- `python/cusignal/convolution/_convolution_cuda.py`
- `cpp/src/convolution/_convolution.cu`

### 1.2 签名、参数和返回值

```python
correlate(in1, in2, mode="full", method="auto")
```

| 项目 | 语义 |
| --- | --- |
| `in1` | 第一个 `array_like`，会被 `cp.asarray` 转为 CuPy 数组 |
| `in2` | 第二个 `array_like`，必须与 `in1` 具有相同 `ndim` |
| `mode="full"` | 返回完整线性互相关；每轴 shape 为 $N_i+M_i-1$ |
| `mode="same"` | 从 `full` 结果居中截取，公开语义是 shape 与 `in1` 相同 |
| `mode="valid"` | 只返回无需边界填零的部分，要求某个输入在每轴上不小于另一个 |
| `method="direct"` | 仅一维，调用预编译 CUDA 直接累加 kernel |
| `method="fft"` | 对 `in2` 逐维反转共轭，再调用 `fftconvolve` |
| `method="auto"` | 反转共轭后进入 `convolve` 的方法选择器 |
| 返回 | CuPy `ndarray`，dtype 由两个输入的类型提升/卷积路径决定，shape 由 `mode` 决定 |

零维标量是特例：直接返回 `in1 * in2.conj()`。非标量时，两输入的维数不同会抛出 `ValueError`。

## 2. 当前算子的完整相关源码

下面的主函数摘录保留只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:23-155` 的原文；未带入学习副本后加的注释，也未摘录同文件的 `correlate2d` 和 `correlation_lags`。

```python
def correlate(
    in1,
    in2,
    mode="full",
    method="auto",
):
    r"""
    Cross-correlate two N-dimensional arrays.

    Cross-correlate `in1` and `in2`, with the output size determined by the
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
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the correlation.

        ``direct``
           The correlation is determined directly from sums, the definition of
           correlation.
        ``fft``
           The Fast Fourier Transform is used to perform the correlation more
           quickly (only available for numerical arrays.)
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See `convolve` Notes for more detail.

    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : contains more documentation on `method`.

    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as::

        z[...,k,...] =
            sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])

    This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')``
    then

    .. math::

          z[k] = (x * y)(k - N + 1)
               = \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}

    for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`

    where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
    and :math:`y_m` is 0 when m is outside the range of y.

    ``method='fft'`` only works for numerical arrays as it relies on
    `fftconvolve`. In certain cases (i.e., arrays of objects or when
    rounding integers can lose precision), ``method='direct'`` is always used.

    Examples
    --------
    Implement a matched filter using cross-correlation, to recover a signal
    that has passed through a noisy channel.

    >>> import cusignal
    >>> import cupy as cp
    >>> sig = cp.repeat(cp.array([0., 1., 1., 0., 1., 0., 0., 1.]), 128)
    >>> sig_noise = sig + cp.random.randn(len(sig))
    >>> corr = cusignal.correlate(sig_noise, cp.ones(128), mode='same') / 128

    >>> import matplotlib.pyplot as plt
    >>> clock = cp.arange(64, len(sig), 128)
    >>> fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(cp.asnumpy(sig))
    >>> ax_orig.plot(cp.asnumpy(clock), cp.asnumpy(sig[clock]), 'ro')
    >>> ax_orig.set_title('Original signal')
    >>> ax_noise.plot(cp.asnumpy(sig_noise))
    >>> ax_noise.set_title('Signal with noise')
    >>> ax_corr.plot(cp.asnumpy(corr))
    >>> ax_corr.plot(cp.asnumpy(clock), cp.asnumpy(corr[clock]), 'ro')
    >>> ax_corr.axhline(0.5, ls=':')
    >>> ax_corr.set_title('Cross-correlated with rectangular pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """

    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # this either calls fftconvolve or this function with method=='direct'
    if method in ("fft", "auto"):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    elif method == "direct":

        if in1.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")

        swapped_inputs = in2.size > in1.size

        if swapped_inputs:
            in1, in2 = in2, in1

        return _convolution_cuda._convolve(in1, in2, False, swapped_inputs, mode)

    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")
```

### 2.1 必要的公开导出

```python
# ZKX/cusignal-23.08.00/python/cusignal/__init__.py:29
from cusignal.convolution.correlate import correlate, correlate2d

# ZKX/cusignal-23.08.00/python/cusignal/convolution/__init__.py:22
from cusignal.convolution.correlate import correlate, correlate2d, correlation_lags
```

第一行使用者可写 `cusignal.correlate`；第二行使用者可写 `cusignal.convolution.correlate`。同行的其他符号仅是导出事实，不纳入本算子逻辑。

### 2.2 直接 helper：`_reverse_and_conj`

只读基准：`ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:210-216`；添加注释后的学习副本：`Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:225-233`。

```python
def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    # return cp.flip(x, 0)
    return x[reverse].conj()
```

`slice(None, None, -1)` 表示沿一个轴从末尾走到开头；乘 `x.ndim` 构造每轴一个反向切片。`x[reverse]` 因此反转所有维，`.conj()` 再取逐元素复共轭。对实数，`.conj()` 数值不变。原注释中的 `cp.flip(x, 0)` 已被注释掉，而且它只会反转第 0 轴，不等价于当前 N 维实现。

### 2.3 FFT/自动路径的直接依赖

`convolve` 完整符号位于基准 `convolve.py:32-156`。对 `correlate` 真正执行的关键语句是：

```python
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
```

定位：`ZKX/.../convolve.py:135-153`，学习副本行号相同。这里 `kernel` 已经是 `correlate` 传入的 `_reverse_and_conj(in2)`，因而后续即使走卷积的 direct kernel，数学上仍是互相关。

`fftconvolve` 的完整符号位于基准 `convolve.py:159-319`，关键计算为：

```python
shape = np.maximum(s1, s2)
shape[axes] = s1[axes] + s2[axes] - 1
fshape = [next_fast_len(d) for d in shape[axes]]
fslice = tuple([slice(sz) for sz in shape])

if not complex_result:
    sp1 = cp.fft.rfftn(in1, fshape, axes=axes)
    sp2 = cp.fft.rfftn(in2, fshape, axes=axes)
    ret = cp.fft.irfftn(sp1 * sp2, fshape, axes=axes)[fslice].copy()
else:
    sp1 = cp.fft.fftn(in1, fshape, axes=axes)
    sp2 = cp.fft.fftn(in2, fshape, axes=axes)
    ret = cp.fft.ifftn(sp1 * sp2, axes=axes)[fslice].copy()
```

定位：`ZKX/.../convolve.py:282-304`。`shape=N+M-1` 防止循环回卷；`next_fast_len` 选择更适合 FFT 的填零长度；实数路径使用只保留非负频率的 `rfftn`，复数路径使用完整 `fftn`。此处频域乘积看似是 $X\widetilde Y$，而 `_reverse_and_conj` 保证 $\widetilde Y=Y^*$，所以整体是 $\mathcal F^{-1}\{XY^*\}$。

`choose_conv_method` 完整符号位于基准 `convolve.py:418-523`。它对不支持的超高精度复数、可能超过浮点精确整数范围的输入和布尔输入强制返回 `direct`；其他数值数组由 `_fftconv_faster` 预估。它是调度器，不参与公式计算。

### 2.4 显式 `direct` 路径的 Python 后端

`_convolve` 完整符号：基准 `_convolution_cuda.py:387-438`；学习副本 `_convolution_cuda.py:408-463`。它完成：

1. `_valfrommode(mode)` 把字符串转为 `VALID=0`、`SAME=1`、`FULL=2`。
2. `cp.promote_types` 得到公共 dtype，并对两输入 `astype`。
3. `valid` 输出长度为 $|N-M|+1$；`same` 依赖 `swapped_inputs` 恢复第一输入语义；`full` 为 $N+M-1$。
4. `cp.empty` 在 GPU 上分配输出。
5. `_convolve_gpu(..., use_convolve=False, ...)` 选择 `correlate` kernel。

`_convolve_gpu` 完整符号：基准 `_convolution_cuda.py:224-265`；学习副本 `_convolution_cuda.py:230-276`。它把 `inp` 和 `ker` 转为 CuPy 设备数组，通过 `_get_tpb_bpg()` 取 grid/block，在 `use_convolve=False` 时缓存并获取 `"correlate"` fatbin kernel，最后通过 wrapper 调用它。

### 2.5 直接 CUDA kernel 的完整核心符号

基准：`ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:142-193`；学习副本：`Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:143-204`。

```cpp
template<typename T>
__device__ void _cupy_correlate( const T *__restrict__ inp,
                                 const int inpW,
                                 const T *__restrict__ kernel,
                                 const int  kerW,
                                 const int  mode,
                                 const bool swapped_inputs,
                                 T *__restrict__ out,
                                 const int outW ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < outW; tid += stride ) {
        T temp {};

        if ( mode == 0 ) {  // Valid
            if ( tid >= 0 && tid < inpW ) {
                for ( int j = 0; j < kerW; j++ ) {
                    temp += inp[tid + j] * kernel[j];
                }
            }
        } else if ( mode == 1 ) {  // Same
            const int P1 { kerW / 2 };
            int       start {};
            if ( !swapped_inputs ) {
                start = 0 - P1 + tid;
            } else {
                start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid + 1;
            }
            for ( int j = 0; j < kerW; j++ ) {
                if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                    temp += inp[start + j] * kernel[j];
                }
            }
        } else {  // Full
            const int P1 { kerW - 1 };
            const int start { 0 - P1 + tid };
            for ( int j = 0; j < kerW; j++ ) {
                if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                    temp += inp[start + j] * kernel[j];
                }
            }
        }

        if ( swapped_inputs ) {
            out[outW - tid - 1] = temp;  // TODO: Move to shared memory
        } else {
            out[tid] = temp;
        }
    }
}
```

`tx` 是全局线程号，`stride` 是全 grid 的线程数。grid-stride loop 让一个线程处理 `tx, tx+stride, ...` 这些输出。每个 `tid` 使用寄存器局部变量 `temp` 从零累加。`valid`、`same`和 `full` 只在起始索引和边界检查上不同。`swapped_inputs=True` 时反向写回，是为了恢复交换输入前的滞后顺序。

该模板函数之后有 `int32`、`int64`、`float32`、`float64`、`complex64`和 `complex128` 的 `extern "C" __global__` 入口（基准 `_convolution.cu:195-260`）。每个入口只负责把具体指针类型转发给 `_cupy_correlate<T>`，不另行实现算法。

## 3. docstring 逐行翻译与解释

下表使用只读基准行号。空行不单独解释；连续的英文句子仍按原行号列出，不改变摘录顺序。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:29` 至 `:30`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:29` 至 `:30`。

```python
    r"""
    Cross-correlate two N-dimensional arrays.
```

逐行对应说明：

- 第 1 行：`r"""` 开始 raw docstring；`r` 使公式中的反斜杠不被 Python 转义。
- 第 2 行：“对两个 N 维数组做互相关。”

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:32` 至 `:33`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:32` 至 `:33`。

```python
    Cross-correlate `in1` and `in2`, with the output size determined by the
    `mode` argument.
```

逐行对应说明：

- 第 1 行：说明输入名为 `in1`、`in2`，本句在下一行续写。
- 第 2 行：输出大小由 `mode` 参数决定。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:35` 至 `:39`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:35` 至 `:39`。

```python
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
```

逐行对应说明：

- 第 1 行：`Parameters` 是 NumPy/SciPy docstring 参数节标题。
- 第 2 行：连续短横线是 reStructuredText 标题下划线。
- 第 3 行：`in1` 接受可转成数组的对象。
- 第 4 行：`in1` 是第一个输入。
- 第 5 行：`in2` 也是 `array_like`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:42` 至 `:42`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:35` 至 `:35`。

```python
    firwin2,
```

逐行对应说明：

- 第 1 行：`in2` 是第二个输入，应与 `in1` 维数相同。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:41` 至 `:42`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:41` 至 `:42`。

```python
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
```

逐行对应说明：

- 第 1 行：`mode` 是可选字符串，只接受 `full/valid/same`。
- 第 2 行：引出三种输出大小的说明。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:44` 至 `:55`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:44` 至 `:55`。

```python
        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the correlation.
```

逐行对应说明：

- 第 1 行：reStructuredText 双反引号把 `full` 渲染为行内代码。
- 第 2 行：`full` 返回完整离散线性互相关。
- 第 3 行：补充“两个输入”，并说明这是默认值。
- 第 4 行：开始 `valid` 模式。
- 第 5 行：输出只包含不依赖填零的元素，句子续到下行。
- 第 6 行：说明不依赖 zero-padding，并开始给出 shape 前提。
- 第 7 行：每个维度上必须有一个输入不小于另一个。
- 第 8 行：开始 `same` 模式。
- 第 9 行：输出大小与 `in1` 相同，句子续到下行。
- 第 10 行：它是相对 `full` 结果居中的子集。
- 第 11 行：`method` 是可选字符串，可取 `auto/direct/fft`。
- 第 12 行：它决定计算相关的实现方法。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:57` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:57` 至 `:65`。

```python
        ``direct``
           The correlation is determined directly from sums, the definition of
           correlation.
        ``fft``
           The Fast Fourier Transform is used to perform the correlation more
           quickly (only available for numerical arrays.)
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See `convolve` Notes for more detail.
```

逐行对应说明：

- 第 1 行：开始 `direct` 说明。
- 第 2 行：直接法按相关定义直接求和，句子续到下行。
- 第 3 行：完成“相关的定义”。复数 kernel 实际遗漏共轭，见后文差异节。
- 第 4 行：开始 `fft` 说明。
- 第 5 行：使用 FFT 计算相关，句子续到下行。
- 第 6 行：它通常更快，且只适用于数值数组。
- 第 7 行：开始 `auto` 说明。
- 第 8 行：根据速度估计自动选择 direct 或 FFT，句子续到下行。
- 第 9 行：`auto` 是默认值，详细选择规则见 `convolve` Notes。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:67` 至 `:71`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:67` 至 `:71`。

```python
    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.
```

逐行对应说明：

- 第 1 行：`Returns` 是返回值节标题。
- 第 2 行：标题下划线。
- 第 3 行：返回项名为 `correlate`，类型记为 array。
- 第 4 行：返回 N 维数组，包含线性互相关的一个子集，句子续到下行。
- 第 5 行：说明输入顺序是 `in1` 与 `in2`。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:73` 至 `:75`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:73` 至 `:75`。

```python
    See Also
    --------
    choose_conv_method : contains more documentation on `method`.
```

逐行对应说明：

- 第 1 行：`See Also` 开始相关 API 引用节。
- 第 2 行：标题下划线。
- 第 3 行：引用 `choose_conv_method`，它说明/实现 `method` 选择。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:77` 至 `:79`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:77` 至 `:79`。

```python
    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as::
```

逐行对应说明：

- 第 1 行：`Notes` 开始数学定义节。
- 第 2 行：标题下划线。
- 第 3 行：引出 $d$ 维数组 $x,y$ 的相关 $z$；末尾 `::` 引出字面代码块。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:81` 至 `:82`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:81` 至 `:82`。

```python
        z[...,k,...] =
            sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])
```

逐行对应说明：

- 第 1 行：左边 `z[...,k,...]` 表示某一轴的滞后 $k$。
- 第 2 行：对所有轴索引求和，并对第二输入的偏移元素取 `conj`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:84` 至 `:85`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:84` 至 `:85`。

```python
    This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')``
    then
```

逐行对应说明：

- 第 1 行：专门转到一维 `full` 情况。
- 第 2 行：“则”，引出下面的数学式。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:87` 至 `:87`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:87` 至 `:87`。

```python
    .. math::
```

逐行对应说明：

- 第 1 行：`.. math::` 是 Sphinx/reStructuredText 数学块指令。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:89` 至 `:90`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:89` 至 `:90`。

```python
          z[k] = (x * y)(k - N + 1)
               = \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}
```

逐行对应说明：

- 第 1 行：第一行把相关写成一个带索引偏移的乘积/卷积表达。
- 第 2 行：展开为有限求和，$y^*$ 表示复共轭。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:92` 至 `:92`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:92` 至 `:92`。

```python
    for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`
```

逐行对应说明：

- 第 1 行：文档使用输出数组索引 $0\ldots|x|+|y|-2$，而不是有符号的物理滞后。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:94` 至 `:95`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:94` 至 `:95`。

```python
    where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
    and :math:`y_m` is 0 when m is outside the range of y.
```

逐行对应说明：

- 第 1 行：$\|x\|$ 在此处表示长度而非范数，$N$ 取两输入长度较大者。
- 第 2 行：$y_m$ 越界时视为 0，这定义了线性而非循环相关。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:97` 至 `:99`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:97` 至 `:99`。

```python
    ``method='fft'`` only works for numerical arrays as it relies on
    `fftconvolve`. In certain cases (i.e., arrays of objects or when
    rounding integers can lose precision), ``method='direct'`` is always used.
```

逐行对应说明：

- 第 1 行：说明 FFT 路径只支持数值数组，因为它依赖下一行的函数。
- 第 2 行：点名 `fftconvolve`，并开始列出强制 direct 的情形。
- 第 3 行：对象数组或 FFT 会丢失整数精度时应选直接法。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:101` 至 `:104`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:101` 至 `:104`。

```python
    Examples
    --------
    Implement a matched filter using cross-correlation, to recover a signal
    that has passed through a noisy channel.
```

逐行对应说明：

- 第 1 行：`Examples` 开始示例节。
- 第 2 行：标题下划线。
- 第 3 行：示例要用互相关实现匹配滤波，句子续到下行。
- 第 4 行：目标是恢复经过噪声信道的信号。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:106` 至 `:110`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:106` 至 `:110`。

```python
    >>> import cusignal
    >>> import cupy as cp
    >>> sig = cp.repeat(cp.array([0., 1., 1., 0., 1., 0., 0., 1.]), 128)
    >>> sig_noise = sig + cp.random.randn(len(sig))
    >>> corr = cusignal.correlate(sig_noise, cp.ones(128), mode='same') / 128
```

逐行对应说明：

- 第 1 行：doctest 导入 `cusignal`。
- 第 2 行：导入 CuPy 并简写为 `cp`。
- 第 3 行：把 8 个 0/1 符号各重复 128 次，构造分段常值信号。
- 第 4 行：加上与信号等长的标准正态噪声。
- 第 5 行：与长度 128 的全 1 模板做 `same` 相关，再除以 128 归一化矩形脉冲累加值。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:112` 至 `:123`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:112` 至 `:123`。

```python
    >>> import matplotlib.pyplot as plt
    >>> clock = cp.arange(64, len(sig), 128)
    >>> fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(cp.asnumpy(sig))
    >>> ax_orig.plot(cp.asnumpy(clock), cp.asnumpy(sig[clock]), 'ro')
    >>> ax_orig.set_title('Original signal')
    >>> ax_noise.plot(cp.asnumpy(sig_noise))
    >>> ax_noise.set_title('Signal with noise')
    >>> ax_corr.plot(cp.asnumpy(corr))
    >>> ax_corr.plot(cp.asnumpy(clock), cp.asnumpy(corr[clock]), 'ro')
    >>> ax_corr.axhline(0.5, ls=':')
    >>> ax_corr.set_title('Cross-correlated with rectangular pulse')
```

逐行对应说明：

- 第 1 行：导入 Matplotlib 绘图 API。
- 第 2 行：每个 128 样本符号的中心在偏移 64 处，构造这些时钟索引。
- 第 3 行：创建三个共享 x 轴的子图，并分别解包为原信号、噪声和相关坐标轴。
- 第 4 行：`cp.asnumpy` 将 GPU 数组拷回 CPU，供 Matplotlib 绘制原信号。
- 第 5 行：在符号中心画红色圆点。
- 第 6 行：设置原始信号子图标题。
- 第 7 行：绘制加噪信号，同样先从 GPU 拷回 CPU。
- 第 8 行：设置加噪子图标题。
- 第 9 行：绘制相关结果。
- 第 10 行：在时钟中心绘制相关样本的红色圆点。
- 第 11 行：画高度 0.5 的水平虚线，用作直观阈值。
- 第 12 行：设置“与矩形脉冲互相关”标题。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:124` 至 `:126`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:124` 至 `:126`。

```python
    >>> ax_orig.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()
```

逐行对应说明：

- 第 1 行：给原图 y 轴增加 10% 边距，x 轴边距为 0。
- 第 2 行：自动调整子图布局。
- 第 3 行：显示图窗。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:128` 至 `:128`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:128` 至 `:128`。

```python
    """
```

逐行对应说明：

- 第 1 行：`"""` 结束 raw docstring。


## 4. 源代码逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:23` 至 `:28`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:23` 至 `:28`。

```python
def correlate(
    in1,
    in2,
    mode="full",
    method="auto",
):
```

逐行对应说明：

- 第 1 行：`def correlate(` 开始函数定义；左括号允许签名跨行。
- 第 2 行：`in1,` 定义第一个必填位置或关键字参数，逗号表示后面还有参数。
- 第 3 行：`in2,` 定义第二个必填参数。
- 第 4 行：`mode="full",` 定义默认值为 `"full"` 的参数。
- 第 5 行：`method="auto",` 定义默认自动选择方法。
- 第 6 行：`):` 关闭参数列表，冒号开始函数 suite。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:131` 至 `:133`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:130` 至 `:131`。

```python
    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)
```

逐行对应说明：

- 第 1 行：`cp.asarray(in1)` 转为 CuPy 数组；若已是兼容数组，可避免不必要拷贝。
- 第 2 行：对 `in2` 做同样转换。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:136` 至 `:136`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:133` 至 `:133`。

```python
    if in1.ndim == in2.ndim == 0:
```

逐行对应说明：

- 第 1 行：链式比较等价于 `in1.ndim == in2.ndim and in2.ndim == 0`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:137` 至 `:137`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:134` 至 `:134`。

```python
        return in1 * in2.conj()
```

逐行对应说明：

- 第 1 行：零维时返回第一标量乘第二标量共轭；`return` 立即结束函数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:139` 至 `:140`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:135` 至 `:136`。

```python
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
```

逐行对应说明：

- 第 1 行：`elif` 只在前一条件为假时判断维数不同。
- 第 2 行：用 `raise` 抛出 `ValueError`，阻止继续计算。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:142` 至 `:142`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:138` 至 `:138`。

```python
    # this either calls fftconvolve or this function with method=='direct'
```

逐行对应说明：

- 第 1 行：原有普通注释：下面会调用 `fftconvolve` 或回到 direct 路径。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:144` 至 `:144`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:139` 至 `:139`。

```python
    if method in ("fft", "auto"):
```

逐行对应说明：

- 第 1 行：`in` 检查 `method` 是否位于二元 tuple `("fft", "auto")`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:145` 至 `:145`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:140` 至 `:140`。

```python
        return convolve(in1, _reverse_and_conj(in2), mode, method)
```

逐行对应说明：

- 第 1 行：先计算 `_reverse_and_conj(in2)`，再按位置参数调用 `convolve`；结果直接返回。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:148` 至 `:148`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:142` 至 `:142`。

```python
    elif method == "direct":
```

逐行对应说明：

- 第 1 行：前分支未返回时，检查 `method == "direct"`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:151` 至 `:152`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:144` 至 `:145`。

```python
        if in1.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")
```

逐行对应说明：

- 第 1 行：direct 分支中检查维数是否大于 1。
- 第 2 行：多维 direct 未实现，抛出 `ValueError`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:155` 至 `:155`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:147` 至 `:147`。

```python
        swapped_inputs = in2.size > in1.size
```

逐行对应说明：

- 第 1 行：比较元素数，得到布尔值 `swapped_inputs`。在一维中 `.size` 就是长度。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:158` 至 `:159`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:149` 至 `:150`。

```python
        if swapped_inputs:
            in1, in2 = in2, in1
```

逐行对应说明：

- 第 1 行：若第二输入更长，进入交换分支。
- 第 2 行：Python tuple unpacking 同时交换引用，不需要显式临时变量。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:162` 至 `:162`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:152` 至 `:152`。

```python
        return _convolution_cuda._convolve(in1, in2, False, swapped_inputs, mode)
```

逐行对应说明：

- 第 1 行：调用 `_convolve`；第三参数 `False` 表示不用卷积 kernel，第四参数传入交换标志。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:165` 至 `:166`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:154` 至 `:155`。

```python
    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")
```

逐行对应说明：

- 第 1 行：`else` 覆盖所有未接受的 `method` 值。
- 第 2 行：两个相邻字符串字面量由 Python 在编译时自动拼接，然后构造并抛出 `ValueError`。


## 5. 调用链与算法总结

```text
cusignal.correlate
├── 标量：in1 * conj(in2)
├── method in {fft, auto}
│   └── _reverse_and_conj(in2)
│       └── convolve
│           ├── choose_conv_method        # 仅 auto
│           ├── fftconvolve              # FFT 路径
│           └── _convolution_cuda._convolve(use_convolve=True)
└── method == direct
    └── _convolution_cuda._convolve(use_convolve=False)
        └── _convolve_gpu
            └── fatbin _cupy_correlate_<dtype>
                └── device template _cupy_correlate<T>
```

FFT/自动路径在 Python 层先完成数学所需的反转共轭，因而后续可安全复用卷积。显式 direct 路径则依赖专用相关 kernel 直接管理索引和 mode 边界。

## 6. 数学映射、边界、数值和复杂度

### 6.1 公式到代码

| 数学步骤 | Python/CUDA 落实 |
| --- | --- |
| $\widetilde y[\mathbf n]=y^*[-\mathbf n]$ | `_reverse_and_conj`: `x[reverse].conj()` |
| $R_{xy}=x*\widetilde y$ | `correlate` 调用 `convolve(in1, _reverse_and_conj(in2), ...)` |
| $L_i=N_i+M_i-1$ | `fftconvolve`: `shape[axes] = s1[axes] + s2[axes] - 1` |
| $R=\mathcal F^{-1}\{XY^*\}$ | 反转共轭后的 `sp1 * sp2` 及 `irfftn/ifftn` |
| 实数 direct 滑动和 | `_cupy_correlate`: `temp += inp[...] * kernel[j]` |
| `same/valid` | `_centered` 截取（FFT）或 CUDA `start`/边界判断（direct） |

### 6.2 复数 direct 路径的不一致

这是本次逐行阅读中最重要的代码事实：

- docstring 在基准 `correlate.py:81-82` 明确写 `conj(y[...])`。
- FFT/自动路径在 `correlate.py:140` 调用 `_reverse_and_conj`，确实执行复共轭。
- 显式 direct 路径把原始 `in2` 传给 `_convolve`，未先共轭。
- `_cupy_correlate<T>` 在基准 `_convolution.cu:161,174,182` 都是 `inp * kernel`，没有 `conj(kernel)`。
- 文件又显式提供 `complex64/complex128` 入口，所以不能用“不支持复数”解释。

结论：实数 direct 路径与公式一致；复数 direct 路径计算的是未共轭的滑动乘积和，与文档及 FFT 路径不一致。本学习任务不修改只读基准，也没有在本地或 ZQ500 上运行测试；这是基于源码的静态结论。

### 6.3 dtype 与数值稳定性

- direct 路径使用 `cp.promote_types` 统一输入 dtype，累加器 `T temp {}` 与输出 dtype 相同；长序列可出现浮点累加误差或整数溢出。
- FFT 路径要经过浮点 Fourier 变换；整数输出在转回目标 dtype 前使用 `cp.around`。
- `auto` 用乘积上界估计整数 FFT 是否可能超过浮点尾数的精确表示范围；超出时选 direct。
- 输入为 `NaN/Inf` 时代码没有专门清洗；它们会参与乘法、累加或 FFT 并传播到输出。

### 6.4 复杂度与性能瓶颈

- 一维 direct：最坏 $O(NM)$ 乘加，输出空间 $O(N+M)$；每个输出线程顺序扫描模板，大模板时计算量和对 kernel 的重复读取是瓶颈。
- N 维 FFT：设填零后元素数为 $L$，时间约 $O(L\log L)$，频谱与临时数组空间 $O(L)$；小数组时 FFT 启动、计划和临时内存开销可能更大。
- `_reverse_and_conj` 产生反向视图后 `.conj()` 的具体内存行为由 CuPy 决定；后续 FFT 通常需要合适布局，可引入额外复制。
- direct 的 `swapped_inputs` 反向写回使全局写地址随 `tid` 反向连续，数据仍连续但与正向语义相反。

## 7. 建议的亲自阅读顺序与检查问题

1. 先读 `correlate.py:23-155`：确认三种 `method` 在哪一行分流，为什么 `fft` 和 `auto` 在同一分支。
2. 再读 `convolution_utils.py:210-216`：亲手对二维 shape 写出 `reverse` tuple，理解为何是“所有维度”。
3. 读 `convolve.py:122-153`：跟踪 `auto -> choose_conv_method -> fft/direct`，注意此时 `kernel` 已经反转共轭。
4. 读 `convolve.py:245-319`：找出线性填零 shape、FFT 友好 shape、频谱相乘和 mode 截取四个环节。
5. 读 `_convolution_cuda.py:224-265,387-438`：区分 shape/dtype 处理、kernel 选择和 kernel 启动。
6. 最后读 `_convolution.cu:142-260`：对每个 mode 手算 `start`，并专门搜索 `conj`，验证复数 direct 差异。

阅读后应能回答：

- `method="auto"` 为什么即使最终选 direct，也与显式 `method="direct"` 走不同的 Python 前处理？
- `same` 在交换输入后如何尽量恢复以原 `in1` 为中心的输出语义？
- 为什么 FFT 路径的 `sp1 * sp2` 没有显式 `.conj()` 仍可计算互相关？
- 哪三条 CUDA 累加语句证明复数 direct 路径没有共轭？
- 为什么 `correlate` 的结果不是 $[-1,1]$ 内的归一化相似度？

### 阅读检查参考答案

1. `auto/fft` 在 `correlate.py:139-140` 先执行 `_reverse_and_conj(in2)` 再进入 `convolve`；显式 `direct` 在 `correlate.py:142-152` 把原 `in2` 交给专用 correlate kernel。因此 `auto` 最终选 direct 时走的是“反转共轭+卷积 direct”，不是专用 correlate direct kernel。
2. Python 层记录 `swapped_inputs`，CUDA kernel 在交换后以 `out[outW-tid-1]` 反向写回；`same` 的起始索引也根据该标志分支，使输出尽量对齐原第一输入。
3. `sp2` 的时域输入已是 $y^*[-n]$，其 Fourier 变换等于 $Y^*$，所以 `sp1*sp2` 就是 $XY^*$。
4. 基准 `_convolution.cu:161,174,182` 的三条 `temp += inp[...] * kernel[j]` 都没有 `conj`，证明复数显式 direct 路径与文档公式不一致。
5. `correlate` 未执行去均值或除以能量范数，只返回滑动乘积和，所以数值无 $[-1,1]$ 界限。

## 8. 验证声明

- 本阶段只执行了源码静态阅读、行号核对和完整性比较。
- 未在本地运行 Python/CuPy/CUDA 测试，也未连接 ZQ500 服务器，符合学习任务默认不构建、不测试的限制。
- `ZKX/cusignal-23.08.00` 始终只读；所有新增源码注释均位于 `Learning/cusignal-23.08.00`，并使用固定 `<学习注释：...>` 格式。
