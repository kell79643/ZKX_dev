# resample_poly 算子学习 — Python 源码算法

> 本文件对应 `Learning/operators/filtering/resample_poly/` 目录，记录阶段二（cuSignal 23.08.00 Python 源码算法）。
> 基于阶段一（数学物理原理），本阶段深入到代码语法和执行逻辑。

---

## 一、代码定位与接口概览

### 1.1 公开导入路径

- **模块名**：`cusignal.filtering.resample`
- **公开导出**：
  - `python/cusignal/__init__.py:40`：`from .filtering import resample_poly`
  - `python/cusignal/filtering/__init__.py:15`：`from .resample import resample_poly`

### 1.2 函数定义位置

- **文件**：`python/cusignal/filtering/resample.py`
- **函数签名**（只读基准行号 `resample.py:304-445`）：
  ```python
  def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0), gpupath=True):
  ```

### 1.3 核心依赖

- **滤波器设计**：`_design_resample_poly`（`resample.py:24-79`）
- **多相滤波核心**：`upfirdn`（`resample.py:448-539`）
- **多相实现类**：`_UpFIRDn`（`_upfirdn_cuda.py:164-261`）

---

## 二、当前算子的完整相关源码

### 2.1 resample_poly 主函数（只读基准）

```python
# 文件：python/cusignal/filtering/resample.py
# 只读基准行号：304-445

def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0), gpupath=True):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. Values beyond the boundary of the signal are assumed to be zero
    during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    gpupath : bool, Optional
        Optional path for filter design. gpupath == False may be desirable if
        filter sizes are small.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `cusignal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `cusignal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `cusignal.get_window`
    and `cusignal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> import cusignal
    >>> import cupy as cp

    >>> x = cp.linspace(0, 10, 20, endpoint=False)
    >>> y = cp.cos(-x**2/6.0)
    >>> f_fft = cusignal.resample(y, 100)
    >>> f_poly = cusignal.resample_poly(y, 100, 20)
    >>> xnew = cp.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cp.asnumpy(xnew), cp.asnumpy(f_fft), 'b.-', \
                 cp.asnumpy(xnew), cp.asnumpy(f_poly), 'r.-')
    >>> plt.plot(cp.asnumpy(x), cp.asnumpy(y), 'ko-')
    >>> plt.plot(10, cp.asnumpy(y[0]), 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()
    """

    x = cp.asarray(x)
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")

    # Determine our up and down factors
    # Use a rational approimation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_out = x.shape[axis] * up
    n_out = n_out // down + bool(n_out % down)

    # If the window size is greater than 8192, use GPU
    if gpupath:
        pp = cp
    else:
        pp = np

    if isinstance(window, (list, pp.ndarray)):
        window = pp.asarray(window)
        if window.ndim > 1:
            raise ValueError("window must be 1-D")
        half_len = (window.size - 1) // 2
        h = up * window
    else:
        half_len = 10 * max(up, down)
        h = up * _design_resample_poly(up, down, window, gpupath)

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while (
        _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis], up, down)
        < n_out + n_pre_remove
    ):
        n_post_pad += 1

    h = pp.concatenate((pp.zeros(n_pre_pad, h.dtype), h, pp.zeros(n_post_pad, h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis)
    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)

    return y[tuple(keep)]
```

### 2.2 滤波器设计子程序（只读基准）

```python
# 文件：python/cusignal/filtering/resample.py
# 只读基准行号：24-79

def _design_resample_poly(up, down, window, gpupath=True):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.
    gpupath : bool, Optional
        Optional path for filter design. gpupath == False may be desirable if
        filter sizes are small.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `cusignal.get_window` and `cusignal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window, gpupath=gpupath)
    return h
```

### 2.3 upfirdn 多相滤波核心（只读基准）

```python
# 文件：python/cusignal/filtering/resample.py
# 只读基准行号：448-539

def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
):
    """
    Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    Examples
    --------
    Simple operations:
    >>> from cusignal import upfirdn
    >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
    array([ 1., 2., 3., 2., 1.])
    >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
    array([ 1., 0., 0., 2., 0., 0., 3., 0., 0.])
    >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
    array([ 1., 1., 1., 2., 2., 2., 3., 3., 3.])
    >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
    array([ 0.5, 1. , 1. , 1. , 1. , 1. , 0.5, 0. ])
    >>> upfirdn([1], cp.arange(10), 1, 3)  # decimation by 3
    array([ 0., 3., 6., 9.])
    >>> upfirdn([.5, 1, .5], cp.arange(10), 2, 3)  # linear interp, rate 2/3
    array([ 0. , 1. , 2.5, 4. , 5.5, 7. , 8.5, 0. ])
    Apply a single filter to multiple signals:
    >>> x = cp.reshape(cp.arange(8), (4, 2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])
    Apply along the last dimension of ``x``:
    >>> h = [1, 1]
    >>> upfirdn(h, x, 2)
    array([[ 0., 0., 1., 1.],
           [ 2., 2., 3., 3.],
           [ 4., 4., 5., 5.],
           [ 6., 6., 7., 7.]])
    Apply along the 0th dimension of ``x``:
    >>> upfirdn(h, x, 2, axis=0)
    array([[ 0., 1.],
           [ 0., 1.],
           [ 2., 3.],
           [ 2., 3.],
           [ 4., 5.],
           [ 4., 5.],
           [ 6., 7.],
           [ 6., 7.]])

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    """

    ufd = _UpFIRDn(h, x.dtype, up, down)
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis)
```

### 2.4 _UpFIRDn 核心实现类（只读基准）

```python
# 文件：python/cusignal/filtering/_upfirdn_cuda.py
# 只读基准行号：164-261

class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""

        if isinstance(h, cp.ndarray):
            pp = cp
        else:
            pp = np

        h = pp.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be 1D with non-zero length")

        self._output_type = pp.result_type(h.dtype, x_dtype, pp.float32)
        h = pp.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError("Both up and down must be >= 1")
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, pp, self._up)
        self._h_trans_flip = cp.asarray(self._h_trans_flip)
        self._h_trans_flip = cp.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(
        self,
        x,
        axis,
    ):
        """Apply the prepared filter to the specified axis of a nD signal x"""

        x = cp.asarray(x, self._output_type)

        output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        out = cp.empty(output_shape, dtype=self._output_type, order="C")
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim > 1:
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()

            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()

            blockspergrid = (blockspergrid_x, blockspergrid_y)

        else:
            threadsperblock, blockspergrid = _get_tpb_bpg()

        if out.ndim == 1:
            k_type = "upfirdn1D"

            _populate_kernel_cache(out.dtype, k_type)

            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                k_type,
            )
        elif out.ndim == 2:
            k_type = "upfirdn2D"

            _populate_kernel_cache(out.dtype, k_type)

            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                k_type,
            )
        else:
            raise NotImplementedError("upfirdn() requires ndim <= 2")

        kernel(
            x,
            self._h_trans_flip,
            self._up,
            self._down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
        )

        _print_atts(kernel)

        return out
```

### 2.5 辅助函数：_pad_h（多相系数重排）

```python
# 文件：python/cusignal/filtering/_upfirdn_cuda.py
# 只读基准行号：31-44

def _pad_h(h, pp, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = pp.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full
```

### 2.6 辅助函数：_output_len（输出长度计算）

```python
# 文件：python/cusignal/filtering/_upfirdn_cuda.py
# 只读基准行号：47-48

def _output_len(len_h, in_len, up, down):
    return (((in_len - 1) * up + len_h) - 1) // down + 1
```

---

## 三、函数签名与参数说明

### 3.1 resample_poly 参数

| 参数名 | 类型 | 默认值 | 语义 | 取值范围 | 数学对应 |
|--------|------|--------|------|----------|----------|
| `x` | array_like | - | 输入信号 | N 维数组 | $x[n]$（阶段一） |
| `up` | int | - | 插值因子 | $\geq 1$ | 原理 4 的 $up$ |
| `down` | int | - | 抽取因子 | $\geq 1$ | 原理 4 的 $down$ |
| `axis` | int | 0 | 重采样轴 | $-N \sim N-1$ | 沿哪个维度应用 |
| `window` | string/tuple/array_like | ("kaiser", 5.0) | FIR 滤波器设计 | 窗函数名或系数数组 | 原理 4 的滤波器 $h[n]$ |
| `gpupath` | bool | True | 滤波器设计路径 | True/False | GPU 或 CPU 设计 |

**返回值**：
- `resampled_x`：重采样后的数组，形状与 `x` 相同，除了 `axis` 维度变为 $N' = \lfloor N \cdot up'/down' \rfloor + \epsilon$。

### 3.2 upfirdn 参数

| 参数名 | 类型 | 默认值 | 语义 | 取值范围 | 数学对应 |
|--------|------|--------|------|----------|----------|
| `h` | array_like | - | FIR 滤波器系数 | 1-D 数组，非空 | 原理 2 的 $h[n]$ |
| `x` | array_like | - | 输入信号 | N 维数组 | $x[n]$ |
| `up` | int | 1 | 插值因子 | $\geq 1$ | 原理 4 的 $up$ |
| `down` | int | 1 | 抽取因子 | $\geq 1$ | 原理 4 的 $down$ |
| `axis` | int | -1 | 滤波轴 | $-N \sim N-1$ | 沿哪个维度应用 |

**返回值**：
- `y`：输出信号数组，形状与 `x` 相同，除了 `axis` 维度变为 `_output_len(len(h), x.shape[axis], up, down)`。

---

## 四、源代码逐行解释

### 4.1 resample_poly 主函数逐行解释

| 行号 | 源码行 | 逐行解释 | 数学映射 |
|------|--------|----------|----------|
| 393 | `x = cp.asarray(x)` | 将输入 `x` 转换为 CuPy 数组，确保在 GPU 上 | $x[n]$ 的数组表示 |
| 394-395 | `up = int(up); down = int(down)` | 将 `up` 和 `down` 转换为整数，确保类型安全 | 插值因子和抽取因子 |
| 396-397 | `if up < 1 or down < 1: raise ValueError(...)` | 边界检查，确保 `up` 和 `down` $\geq 1$ | 原理 4 的参数约束 |
| 402 | `g_ = gcd(up, down)` | 计算 `up` 和 `down` 的最大公约数 | 约化有理分数 $\frac{up}{down} \to \frac{up/g}{down/g}$ |
| 403-404 | `up //= g_; down //= g_` | 约化 `up` 和 `down`，减少计算量 | 原理 5 的 gcd 约化 |
| 405-406 | `if up == down == 1: return x.copy()` | 若约化后 `up' = down' = 1`，无需重采样，返回副本 | $f_{new} = f_s$（采样率不变） |
| 407-408 | `n_out = x.shape[axis] * up; n_out = n_out // down + bool(n_out % down)` | 计算输出样本数 $N' = \lfloor N \cdot up'/down' \rfloor + \epsilon$ | 原理 5 的输出长度公式 |
| 411-414 | `if gpupath: pp = cp; else: pp = np` | 选择 GPU 或 CPU 路径用于滤波器设计 | 工程实现细节 |
| 416-421 | `if isinstance(window, (list, pp.ndarray)): ...` | 若 `window` 为数组，直接使用作为滤波器系数 | 用户自定义滤波器 $h[n]$ |
| 423-424 | `else: half_len = 10 * max(up, down); h = up * _design_resample_poly(...)` | 若 `window` 为窗函数名，调用设计子程序生成滤波器，并乘以 `up` 补偿插值增益 | 原理 4 的滤波器设计，增益补偿 |
| 427 | `n_pre_pad = down - half_len % down` | 计算预补零长度，使输出样本居中 | 零相位对齐（原理 5 的采样时刻映射） |
| 428 | `n_post_pad = 0` | 初始化后补零长度 | 边界处理 |
| 429 | `n_pre_remove = (half_len + n_pre_pad) // down` | 计算需要移除的边界样本数 | 移除预补零引入的边界效应 |
| 431-435 | `while (_output_len(...) < n_out + n_pre_remove): n_post_pad += 1` | 循环增加后补零，直到输出长度足够 | 确保输出样本数正确 |
| 437 | `h = pp.concatenate((pp.zeros(n_pre_pad, h.dtype), h, pp.zeros(n_post_pad, h.dtype)))` | 在滤波器两端补零，实现零相位对齐 | 原理 5 的零相位处理 |
| 438 | `n_pre_remove_end = n_pre_remove + n_out` | 计算输出切片的结束索引 | 输出截取范围 |
| 441 | `y = upfirdn(h, x, up, down, axis)` | 调用 `upfirdn` 执行多相滤波 | 原理 2+3+4 的组合实现 |
| 442-443 | `keep = [slice(None)] * x.ndim; keep[axis] = slice(n_pre_remove, n_pre_remove_end)` | 构造切片对象，移除边界样本 | 输出截取 |
| 445 | `return y[tuple(keep)]` | 返回截取后的输出数组 | 最终重采样结果 $y[m]$ |

### 4.2 _design_resample_poly 子程序逐行解释

| 行号 | 源码行 | 逐行解释 | 数学映射 |
|------|--------|----------|----------|
| 67-69 | `g_ = gcd(up, down); up //= g_; down //= g_` | 约化 `up` 和 `down` | 原理 5 的 gcd 约化 |
| 72 | `max_rate = max(up, down)` | 计算 $\max(up', down')$ | 确定滤波器截止频率 |
| 73 | `f_c = 1.0 / max_rate` | 计算归一化截止频率 $f_c = 1/\max(up', down')$ | 原理 4 的截止频率 $\omega_c = \pi/\max(up', down')$ |
| 76 | `half_len = 10 * max_rate` | 计算半滤波器长度 $K = 10 \cdot \max(up', down')$ | 滤波器阶数（经验公式） |
| 78 | `h = firwin(2 * half_len + 1, f_c, window=window, gpupath=gpupath)` | 调用 `firwin` 设计线性相位 FIR 滤波器 | 原理 4 的滤波器设计 |
| 79 | `return h` | 返回滤波器系数数组 | $h[n]$ |

### 4.3 upfirdn 核心逐行解释

| 行号 | 源码行 | 逐行解释 | 数学映射 |
|------|--------|----------|----------|
| 537 | `ufd = _UpFIRDn(h, x.dtype, up, down)` | 创建 `_UpFIRDn` 对象，封装滤波器和参数 | 原理 2+3 的多相结构初始化 |
| 539 | `return ufd.apply_filter(x, axis)` | 调用 `apply_filter` 执行滤波并返回结果 | 原理 4 的组合滤波 |

### 4.4 _UpFIRDn.__init__ 逐行解释

| 行号 | 源码行 | 逐行解释 | 数学映射 |
|------|--------|----------|----------|
| 168-171 | `if isinstance(h, cp.ndarray): pp = cp; else: pp = np` | 判断滤波器数组类型，选择 CuPy 或 NumPy | 工程实现细节 |
| 173 | `h = pp.asarray(h)` | 将滤波器转换为数组 | $h[n]$ 的数组表示 |
| 174-175 | `if h.ndim != 1 or h.size == 0: raise ValueError(...)` | 边界检查，确保滤波器为 1 维非空 | 原理 2 的 FIR 约束 |
| 177-178 | `self._output_type = pp.result_type(h.dtype, x_dtype, pp.float32); h = pp.asarray(h, self._output_type)` | 计算输出 dtype，提升为至少 float32 | 数值精度保证 |
| 179-182 | `self._up = int(up); self._down = int(down); if self._up < 1 or self._down < 1: raise ValueError(...)` | 类型转换和边界检查 | 原理 4 的参数约束 |
| 184 | `self._h_trans_flip = _pad_h(h, pp, self._up)` | 调用 `_pad_h` 对滤波器系数进行多相重排和翻转 | 原理 2 的多相分解 + 翻转 |
| 185-186 | `self._h_trans_flip = cp.asarray(self._h_trans_flip); self._h_trans_flip = cp.ascontiguousarray(self._h_trans_flip)` | 转换为 CuPy 数组并确保连续内存布局 | GPU 内存优化 |
| 187 | `self._h_len_orig = len(h)` | 保存原始滤波器长度 | 用于输出长度计算 |

### 4.5 _pad_h 多相系数重排逐行解释

| 行号 | 源码行 | 逐行解释 | 数学映射 |
|------|--------|----------|----------|
| 40 | `h_padlen = len(h) + (-len(h) % up)` | 计算补零后的长度，使其为 `up` 的倍数 | 原理 2 的整数倍约束 |
| 41-42 | `h_full = pp.zeros(h_padlen, h.dtype); h_full[: len(h)] = h` | 创建全零数组并填充原始系数 | 补零对齐 |
| 43 | `h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()` | 重排为多相矩阵（每列为一个相位），转置后翻转，再展平 | 原理 2 的多相分解：$h[k] \to h_{phase}[n]$ |
| 44 | `return h_full` | 返回重排后的系数数组 | 多相系数矩阵 |

**示例**（注释中的例子）：
- 输入：`h = [h[0], h[1], ..., h[9]]`，`up = 3`
- 重排过程：
  1. 补零为长度 12（3 的倍数）：`h_full = [h[0], ..., h[9], 0, 0]`
  2. 重排为 $4 \times 3$ 矩阵：
     ```
     [[h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], h[8]],
      [h[9], 0,   0]]
     ```
  3. 转置（变为 $3 \times 4$）：
     ```
     [[h[0], h[3], h[6], h[9]],
      [h[1], h[4], h[7], 0],
      [h[2], h[5], h[8], 0]]
     ```
  4. 翻转列（[::-1]）：
     ```
     [[h[9], h[6], h[3], h[0]],  // 相位 0
      [0,    h[7], h[4], h[1]],  // 相位 1
      [0,    h[8], h[5], h[2]]]  // 相位 2
     ```
  5. 展平：`[h[9], h[6], h[3], h[0], 0, h[7], h[4], h[1], 0, h[8], h[5], h[2]]`

### 4.6 _output_len 输出长度计算逐行解释

| 行号 | 源码行 | 逐行解释 | 数学映射 |
|------|--------|----------|----------|
| 48 | `return (((in_len - 1) * up + len_h) - 1) // down + 1` | 计算输出长度：$N' = \lfloor ((N-1) \cdot up + L - 1) / down \rfloor + 1$ | 原理 5 的输出长度公式（考虑滤波器延迟） |

**公式推导**：
- 插值后长度：$(N-1) \cdot up + 1$
- 卷积后长度：$(N-1) \cdot up + L$
- 抽取后长度：$\lfloor ((N-1) \cdot up + L - 1) / down \rfloor + 1$

---

## 五、调用链与算法总结

### 5.1 完整调用链

```
resample_poly(x, up, down, axis, window, gpupath)
├── gcd(up, down)  # 约化
├── _design_resample_poly(up, down, window, gpupath)  # 滤波器设计（可选）
│   ├── gcd(up, down)  # 约化
│   └── firwin(...)  # FIR 窗函数设计
├── upfirdn(h, x, up, down, axis)  # 多相滤波
│   └── _UpFIRDn(h, x.dtype, up, down)  # 创建对象
│       └── _pad_h(h, pp, up)  # 多相系数重排
│           └── reshape(-1, up).T[:, ::-1].ravel()  # 多相分解 + 翻转
│       └── apply_filter(x, axis)  # 执行滤波
│           ├── _output_len(...)  # 计算输出长度
│           ├── _populate_kernel_cache(...)  # 加载 CUDA kernel
│           └── kernel(...)  # 执行 GPU 多相滤波
└── y[slice(n_pre_remove, n_pre_remove_end)]  # 输出截取
```

### 5.2 算法流程总结

1. **参数约化**：`gcd(up, down)`，减少计算量。
2. **滤波器设计**（可选）：`firwin` 设计线性相位 FIR 滤波器，截止频率 $f_c = 1/\max(up', down')$。
3. **多相系数重排**：`_pad_h` 将滤波器系数重排为 `up` 个相位，翻转并展平。
4. **零相位对齐**：预补零和后补零，使输出样本居中。
5. **多相滤波**：`upfirdn` 执行 GPU 多相卷积。
6. **输出截取**：移除边界样本，返回正确长度的输出。

---

## 六、数学映射、边界、复杂度和阅读检查

### 6.1 数学公式与代码语句对应表

| 数学公式 | 代码语句 | 文件:行号 |
|----------|----------|-----------|
| $up' = up/g$, $down' = down/g$ | `g_ = gcd(up, down); up //= g_; down //= g_` | `resample.py:402-404` |
| $f_c = 1/\max(up', down')$ | `max_rate = max(up, down); f_c = 1.0 / max_rate` | `resample.py:72-73` |
| $L = 2 \cdot 10 \cdot \max(up', down') + 1$ | `half_len = 10 * max_rate; h = firwin(2 * half_len + 1, f_c, ...)` | `resample.py:76-78` |
| $N' = \lfloor N \cdot up'/down' \rfloor + \epsilon$ | `n_out = x.shape[axis] * up; n_out = n_out // down + bool(n_out % down)` | `resample.py:407-408` |
| 多相分解：$h_{phase}[n] = h[n \cdot up + phase]$ | `h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()` | `_upfirdn_cuda.py:43` |
| 输出长度：$N' = \lfloor ((N-1) \cdot up + L - 1) / down \rfloor + 1$ | `return (((in_len - 1) * up + len_h) - 1) // down + 1` | `_upfirdn_cuda.py:48` |

### 6.2 边界处理与异常检查

| 边界条件 | 处理方式 | 代码位置 |
|----------|----------|----------|
| `up < 1` 或 `down < 1` | 抛出 `ValueError` | `resample.py:396-397` |
| `up == down == 1` | 直接返回输入副本 | `resample.py:405-406` |
| `window.ndim > 1` | 抛出 `ValueError` | `resample.py:418-419` |
| `h.ndim != 1` 或 `h.size == 0` | 抛出 `ValueError` | `_upfirdn_cuda.py:174-175` |
| 输出长度不足 | 循环增加后补零 | `resample.py:431-435` |
| 边界样本移除 | 切片 `slice(n_pre_remove, n_pre_remove_end)` | `resample.py:442-445` |

### 6.3 数值稳定性与 dtype 转换

| 处理 | 目的 | 代码位置 |
|------|------|----------|
| `self._output_type = pp.result_type(h.dtype, x_dtype, pp.float32)` | 确保输出至少为 float32，避免精度损失 | `_upfirdn_cuda.py:177` |
| `h = pp.asarray(h, self._output_type)` | 提升滤波器系数精度 | `_upfirdn_cuda.py:178` |
| `x = cp.asarray(x, self._output_type)` | 提升输入信号精度 | `_upfirdn_cuda.py:196` |

### 6.4 时间复杂度与空间复杂度

| 算法阶段 | 时间复杂度 | 空间复杂度 | 说明 |
|----------|------------|------------|------|
| 滤波器设计 | $O(L \log L)$ | $O(L)$ | `firwin` 使用 FFT，$L = 20 \cdot \max(up', down')$ |
| 多相系数重排 | $O(L)$ | $O(L)$ | 单次数组重排 |
| GPU 多相滤波 | $O(N \cdot L/up)$ | $O(N' + L)$ | 每个输出样本需 $L/up$ 次乘加 |
| 输出截取 | $O(N')$ | $O(N')$ | 切片操作 |

**对比传统方法**：
- 传统插零→滤波→抽取：$O(N \cdot L \cdot down)$（每次输出需 $L \cdot down$ 次乘加）
- 多相方法：$O(N \cdot L/up)$（节省因子 $up \cdot down$）

### 6.5 建议阅读顺序

1. **先读 resample_poly 主函数**（`resample.py:304-445`）：
   - 理解参数处理、约化、滤波器设计、零相位对齐、调用 `upfirdn`。
2. **再读 _design_resample_poly 子程序**（`resample.py:24-79`）：
   - 理解滤波器截止频率和长度设计。
3. **然后读 upfirdn 接口**（`resample.py:448-539`）：
   - 理解 `_UpFIRDn` 对象创建和 `apply_filter` 调用。
4. **最后读 _UpFIRDn 核心实现**（`_upfirdn_cuda.py:164-261`）：
   - 理解多相系数重排（`_pad_h`）、输出长度计算、GPU kernel 启动。

### 6.6 阅读检查问题

1. **gcd 约化的作用**：为什么 `resample_poly` 在设计滤波器前先约化 `up` 和 `down`？不约化会有什么后果？
2. **滤波器截止频率**：为什么 `f_c = 1.0 / max_rate`？若 `up' > down'`，哪个因素主导截止频率选择？
3. **滤波器增益补偿**：为什么 `h = up * window` 或 `h = up * _design_resample_poly(...)` 要乘以 `up`？
4. **多相系数重排**：`_pad_h` 中的 `reshape(-1, up).T[:, ::-1].ravel()` 分别完成了哪些操作？
5. **零相位对齐**：预补零 `n_pre_pad = down - half_len % down` 的作用是什么？为什么不直接从第一个输出样本开始？
6. **输出长度公式**：`_output_len` 的公式 `(((in_len - 1) * up + len_h) - 1) // down + 1` 如何推导？
7. **GPU kernel 选择**：`upfirdn1D` 和 `upfirdn2D` 如何选择？为什么限制了 `ndim <= 2`？
8. **与 `resample` 的对比**：为什么 docstring 说"polyphase method will likely be faster than the Fourier method when the number of samples is large and prime"？