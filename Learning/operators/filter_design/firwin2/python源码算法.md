# firwin2 Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开导入路径

`firwin2` 有两个公开入口：

- `cusignal.firwin2`：由 `Learning/cusignal-23.08.00/python/cusignal/__init__.py` 导出；
- `cusignal.filter_design.firwin2`：由 `Learning/cusignal-23.08.00/python/cusignal/filter_design/__init__.py` 导出。

函数定义位于：

```text
Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py
```

本阶段所需的直接项目内依赖只有：

- `_get_fs`：统一 `fs` 与旧参数 `nyq`；
- `get_window`：把窗口描述解析成长度为 `numtaps` 的窗数组。

`numpy.interp`、`numpy/cupy.fft.irfft`、`cupy.asarray` 等属于第三方库调用，只解释本次调用语义，不展开第三方内部源码。

### 1.2 函数签名

```python
firwin2(
    numtaps,
    freq,
    gain,
    nfreqs=None,
    window="hamming",
    nyq=None,
    antisymmetric=False,
    fs=None,
    gpupath=True,
)
```

### 1.3 参数、返回值、dtype 与 shape

| 名称 | 语义 | 默认值 | dtype / shape | 关键约束 |
| --- | --- | --- | --- | --- |
| `numtaps` | FIR 抽头数 $M$ | 无 | Python `int` 标量 | 必须小于 `nfreqs`；源码没有单独检查正数 |
| `freq` | 频率控制点 | 无 | 一维 array-like，长度 $P$ | 从 0 开始、以 $f_s/2$ 结束、非递减；内部值最多重复一次 |
| `gain` | 每个控制点的期望实值增益 | 无 | 一维 array-like，长度 $P$ | 长度必须等于 `freq`；II–IV 型有端点零增益约束 |
| `nfreqs` | 单边均匀插值网格点数 $K$ | `None` | `int` 标量 | 显式给出时必须满足 $K>M$ |
| `window` | 窗名、参数元组、Kaiser beta 或 `None` | `"hamming"` | `str`、`tuple`、数值或 `None` | `None` 表示不施加额外窗 |
| `nyq` | 已弃用的 Nyquist 频率 | `None` | 数值标量 | 不能与 `fs` 同时提供 |
| `antisymmetric` | 是否生成反对称 FIR | `False` | `bool` | 与 `numtaps` 奇偶共同决定 I–IV 型 |
| `fs` | 采样频率 | `None`，内部默认 2 | 数值标量 | Nyquist 频率为 `fs/2` |
| `gpupath` | 插值和 FFT 使用 GPU 还是 CPU 辅助路径 | `True` | `bool` | 无论哪条路径，公开返回值最终都是 CuPy 数组 |
| 返回 `out` | FIR 系数 | — | 一维 `cupy.ndarray`，shape 为 `(numtaps,)` | dtype 由插值、复相移、`irfft` 和窗运算共同决定，通常为浮点类型 |

### 1.4 四型分派

| `antisymmetric` | `numtaps` | 类型 | 系数关系 | 强制端点零值 |
| --- | --- | --- | --- | --- |
| `False` | 奇数 | I | 对称 | 无 |
| `False` | 偶数 | II | 对称 | Nyquist |
| `True` | 奇数 | III | 反对称 | 直流和 Nyquist |
| `True` | 偶数 | IV | 反对称 | 直流 |

## 2. 当前算子的完整相关源码

以下代码块逐字摘录自 `ZKX/cusignal-23.08.00` 只读基准。它们只包含两个必要公开导出、直接 helper `_get_fs`、完整 `firwin2` 定义，以及直接调用的 `get_window` 完整定义；没有把同文件中的其他算子一并复制。

### 2.1 顶层公开导出

基准定位：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:32`。

```python
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    firwin,
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
```


### 2.2 `filter_design` 子包公开导出

基准定位：`ZKX/cusignal-23.08.00/python/cusignal/filter_design/__init__.py:14`。

```python
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    firwin,
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
```


### 2.3 直接 helper `_get_fs`

基准定位：`ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:23`。

```python
def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    if nyq is None and fs is None:
        fs = 2
    elif nyq is not None:
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        fs = 2 * nyq
    return fs
```


### 2.4 `firwin2` 完整定义

基准定位：`ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:399`。

```python
def firwin2(
    numtaps,
    freq,
    gain,
    nfreqs=None,
    window="hamming",
    nyq=None,
    antisymmetric=False,
    fs=None,
    gpupath=True,
):
    """
    FIR filter design using the window method.
    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.
    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    nyq : float, optional
        *Deprecated. Use `fs` instead.* This is the Nyquist frequency.
        Each frequency in `freq` must be between 0 and `nyq`.  Default is 1.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``. Default is 2.
    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.
    See also
    --------
    firls
    firwin
    minimum_phase
    remez
    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.
    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:
       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced
    Magnitude response of all but type I filters are subjects to following
    constraints:
       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency
    .. versionadded:: 0.9.0
    References
    ----------
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)
    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm
    """

    if gpupath:
        pp = cp
    else:
        pp = np

    nyq = 0.5 * _get_fs(fs, nyq)

    if len(freq) != len(gain):
        raise ValueError("freq and gain must be of same length.")

    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(
            (
                "ntaps must be less than nfreqs, but firwin2 was "
                "called with ntaps=%d and nfreqs=%s"
            )
            % (numtaps, nfreqs)
        )

    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError("freq must start with 0 and end with fs/2.")
    d = pp.diff(freq)
    if (d < 0).any():
        raise ValueError("The values in freq must be nondecreasing.")
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError("A value in freq must not occur more than twice.")
    if freq[1] == 0:
        raise ValueError("Value 0 must not be repeated in freq")
    if freq[-2] == nyq:
        raise ValueError("Value fs/2 must not be repeated in freq")

    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    else:
        if numtaps % 2 == 0:
            ftype = 2
        else:
            ftype = 1

    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError(
            "A Type II filter must have zero gain at the " "Nyquist frequency."
        )
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError(
            "A Type III filter must have zero gain at zero " "and Nyquist frequencies."
        )
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero " "frequency.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))

    if (d == 0).any():
        # Tweak any repeated values in freq so that interp works.
        freq = pp.array(freq, copy=True)
        eps = pp.finfo(float).eps * nyq
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # Check if freq is strictly increasing after tweak
        d = pp.diff(freq)
        if (d <= 0).any():
            raise ValueError(
                "freq cannot contain numbers that are too close "
                "(within eps * (fs/2): "
                "{}) to a repeated value".format(eps)
            )

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = pp.linspace(0.0, nyq, nfreqs)
    if gpupath:
        fx = cp.asarray(np.interp(cp.asnumpy(x), freq, gain))
    else:
        fx = np.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = pp.exp(-(numtaps - 1) / 2.0 * 1.0j * pp.pi * x / nyq)
    if ftype > 2:
        shift *= 1j

    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = pp.fft.irfft(fx2)

    # Pass to device memory
    if not gpupath:
        out_full = cp.asarray(out_full)

    if window is not None:
        # Create the window to apply to the filter coefficients.
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind

    if ftype == 3:
        out[out.size // 2] = 0.0

    return out
```


### 2.5 直接 helper `get_window`

基准定位：`ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2010`。

```python
def get_window(window, Nx, fftbins=True):
    r"""
    Return a window of a given length and type.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create. See below for more details.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT (see also
        `fftpack.fftfreq`).
        If False, create a "symmetric" window, for use in filter design.

    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`

    Notes
    -----
    Window types:

    - `~cusignal.windows.windows.boxcar`
    - `~cusignal.windows.windows.triang`
    - `~cusignal.windows.windows.blackman`
    - `~cusignal.windows.windows.hamming`
    - `~cusignal.windows.windows.hann`
    - `~cusignal.windows.windows.bartlett`
    - `~cusignal.windows.windows.flattop`
    - `~cusignal.windows.windows.parzen`
    - `~cusignal.windows.windows.bohman`
    - `~cusignal.windows.windows.blackmanharris`
    - `~cusignal.windows.windows.nuttall`
    - `~cusignal.windows.windows.barthann`
    - `~cusignal.windows.windows.kaiser` (needs beta)
    - `~cusignal.windows.windows.gaussian` (needs standard deviation)
    - `~cusignal.windows.windows.general_gaussian` \
            (needs power, width)
    - `~cusignal.windows.windows.slepian` (needs width)
    - `~cusignal.windows.windows.dpss` \
            (needs normalized half-bandwidth)
    - `~cusignal.windows.windows.chebwin` (needs attenuation)
    - `~cusignal.windows.windows.exponential` (needs decay scale)
    - `~cusignal.windows.windows.tukey` (needs taper fraction)

    If the window requires no parameters, then `window` can be a string.

    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    If `window` is a floating point number, it is interpreted as the beta
    parameter of the `~cusignal.windows.windows.kaiser` window.

    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.

    Examples
    --------
    >>> import cusignal
    >>> cusignal.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> cusignal.get_window(('kaiser', 4.0), 9)
    array([0.08848053, 0.32578323, 0.63343178, 0.89640418, 1.,
           0.89640418, 0.63343178, 0.32578323, 0.08848053])
    >>> cusignal.get_window(4.0, 9)
    array([0.08848053, 0.32578323, 0.63343178, 0.89640418, 1.,
           0.89640418, 0.63343178, 0.32578323, 0.08848053])

    """
    sym = not fftbins
    try:
        beta = float(window)
    except (TypeError, ValueError):
        args = ()
        if isinstance(window, tuple):
            winstr = window[0]
            if len(window) > 1:
                args = window[1:]
        elif isinstance(window, str):
            if window in _needs_param:
                raise ValueError(
                    "The '" + window + "' window needs one or "
                    "more parameters -- pass a tuple."
                )
            else:
                winstr = window
        else:
            raise ValueError("%s as window type is not supported." % str(type(window)))

        try:
            winfunc = _win_equiv[winstr]
        except KeyError:
            raise ValueError("Unknown window type.")

        params = (Nx,) + args + (sym,)
    else:
        winfunc = kaiser
        params = (Nx, beta, sym)

    return winfunc(*params)
```


## 3. docstring 逐行翻译与解释

以下表格覆盖两个完整 docstring 的每一行非空内容。学习副本行号已在写文档时重新计算；基准行号不受学习注释影响。

### 3.1 `firwin2` docstring

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:517` 至 `:519`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:410` 至 `:412`。

```python
    """
    FIR filter design using the window method.
    From the given frequencies `freq` and corresponding gains `gain`,
```

逐行对应说明：

- 第 1 行：三引号开始 `firwin2` 的函数文档字符串。
- 第 2 行：中文：使用窗函数法设计 FIR 滤波器。
- 第 3 行：中文：输入频率数组 `freq` 及其对应的增益数组 `gain`，

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:520` 至 `:529`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:413` 至 `:422`。

```python
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.
    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
```

逐行对应说明：

- 第 1 行：中文：本函数构造具有线性相位的 FIR 滤波器，并且
- 第 2 行：中文：其频率响应近似用户给定的目标响应。
- 第 3 行：这是 NumPy 文档规范的“参数”标题。
- 第 4 行：短横线是 Parameters 标题的下划线标记。
- 第 5 行：声明参数 `numtaps`，类型为整数。
- 第 6 行：中文：`numtaps` 是 FIR 滤波器的抽头数量，并且必须小于
- 第 7 行：中文：插值网格长度 `nfreqs`；本行续接上一行。
- 第 8 行：声明参数 `freq`，接受一维 array-like。
- 第 9 行：中文：`freq` 是频率采样控制点，通常从 0 到 1，其中 1 表示
- 第 10 行：中文：Nyquist 频率；Nyquist 等于采样频率 `fs` 的一半。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:530` 至 `:535`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:423` 至 `:428`。

```python
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
```

逐行对应说明：

- 第 1 行：中文：`freq` 必须非递减；某个内部频率允许重复
- 第 2 行：中文：一次以表达频响跳变；第一个值必须
- 第 3 行：中文：为 0，最后一个值必须为 `fs/2`；本行同时开始说明端点
- 第 4 行：中文：0 与 `fs/2` 都不允许重复。
- 第 5 行：声明参数 `gain`，接受 array-like。
- 第 6 行：中文：它给出每个频率控制点的滤波器增益；某些

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:536` 至 `:547`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:429` 至 `:440`。

```python
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
```

逐行对应说明：

- 第 1 行：中文：增益约束取决于线性相位滤波器类型，
- 第 2 行：中文：详细规则见 Notes。
- 第 3 行：声明可选整数参数 `nfreqs`。
- 第 4 行：中文：它是构造滤波器时使用的插值网格大小。
- 第 5 行：中文：为了提高 FFT 效率，推荐取二次幂加一，
- 第 6 行：中文：例如 129、257；默认值是大于等于
- 第 7 行：中文：`numtaps` 的最小二次幂再加一，并且 `nfreqs` 必须大于
- 第 8 行：中文：`numtaps`；本行结束该参数说明。
- 第 9 行：声明可选参数 `window`，可用字符串、元组、数值或 `None`。
- 第 10 行：中文：指定所用窗函数，默认是 Hamming；可参考
- 第 11 行：中文：`scipy.signal.get_window` 支持的窗口类型。
- 第 12 行：中文：若为 `None`，则不施加额外窗函数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:548` 至 `:555`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:441` 至 `:448`。

```python
    nyq : float, optional
        *Deprecated. Use `fs` instead.* This is the Nyquist frequency.
        Each frequency in `freq` must be between 0 and `nyq`.  Default is 1.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
```

逐行对应说明：

- 第 1 行：声明已弃用的可选浮点参数 `nyq`。
- 第 2 行：中文：应改用 `fs`；`nyq` 表示 Nyquist 频率。
- 第 3 行：中文：`freq` 中每个频率必须位于 0 到 `nyq`，默认 `nyq=1`。
- 第 4 行：声明可选布尔参数 `antisymmetric`。
- 第 5 行：中文：它决定结果冲激响应采用对称还是反对称形式。
- 第 6 行：中文：更多细节见 Notes。
- 第 7 行：声明可选浮点参数 `fs`。
- 第 8 行：中文：它表示信号采样频率；源码文字写成 `cutoff`，

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:556` 至 `:565`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:449` 至 `:458`。

```python
        must be between 0 and ``fs/2``. Default is 2.
    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.
    See also
    --------
    firls
    firwin
```

逐行对应说明：

- 第 1 行：中文：但本函数实际检查的是 `freq`，范围到 `fs/2`；默认 `fs=2`。
- 第 2 行：这是返回值章节标题。
- 第 3 行：短横线是 Returns 标题的下划线。
- 第 4 行：声明返回对象名为 `taps`，类型为 ndarray。
- 第 5 行：中文：返回 FIR 系数的一维数组，其长度为
- 第 6 行：中文：`numtaps`。
- 第 7 行：这是“另请参阅”章节标题。
- 第 8 行：短横线是 See also 标题的下划线。
- 第 9 行：列出相关的最小二乘 FIR 设计器 `firls`。
- 第 10 行：列出相关的窗函数 FIR 设计器 `firwin`。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:566` 至 `:577`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:459` 至 `:470`。

```python
    minimum_phase
    remez
    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.
    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:
```

逐行对应说明：

- 第 1 行：列出线性相位转最小相位的 `minimum_phase`。
- 第 2 行：列出等纹波设计器 `remez`。
- 第 3 行：这是算法备注章节标题。
- 第 4 行：短横线是 Notes 标题的下划线。
- 第 5 行：中文：先根据给定频率与增益在频域构造目标响应。
- 第 6 行：中文：再对目标响应执行逆 FFT，得到相应的
- 第 7 行：中文：卷积核；随后选取该核的
- 第 8 行：中文：前 `numtaps` 个系数，并用 `window` 缩放，
- 第 9 行：中文：最后返回。
- 第 10 行：中文：所得 FIR 具有线性相位；类型由
- 第 11 行：中文：`numtaps` 与 `antisymmetric` 标志共同决定。
- 第 12 行：中文：共有以下四种组合。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:578` 至 `:589`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:471` 至 `:482`。

```python
       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced
    Magnitude response of all but type I filters are subjects to following
    constraints:
       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency
    .. versionadded:: 0.9.0
    References
    ----------
```

逐行对应说明：

- 第 1 行：中文：奇数抽头且不反对称时生成 I 型。
- 第 2 行：中文：偶数抽头且不反对称时生成 II 型。
- 第 3 行：中文：奇数抽头且反对称时生成 III 型。
- 第 4 行：中文：偶数抽头且反对称时生成 IV 型。
- 第 5 行：中文：除 I 型外，其余类型的幅度响应受以下
- 第 6 行：中文：端点约束。
- 第 7 行：中文：II 型在 Nyquist 频率必须为零。
- 第 8 行：中文：III 型在直流和 Nyquist 频率都必须为零。
- 第 9 行：中文：IV 型在直流频率必须为零。
- 第 10 行：Sphinx 指令：该 API 从版本 0.9.0 起加入。
- 第 11 行：这是参考资料章节标题。
- 第 12 行：短横线是 References 标题的下划线。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:590` 至 `:593`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:483` 至 `:486`。

```python
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)
    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
```

逐行对应说明：

- 第 1 行：参考文献 1：Oppenheim 与 Schafer 的《Discrete-Time Signal
- 第 2 行：Processing》，Prentice-Hall，1989；本行续接书目信息。
- 第 3 行：中文：例如可参见该书第 7.4 节。
- 第 4 行：参考文献 2：Steven W. Smith 的《The Scientist and Engineer's Guide to Digital

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:594` 至 `:595`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:487` 至 `:488`。

```python
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm
    """
```

逐行对应说明：

- 第 1 行：Signal Processing》第 17 章，并给出网页地址。
- 第 2 行：三引号结束 `firwin2` 文档字符串。


### 3.2 `get_window` docstring

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2124` 至 `:2124`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2011` 至 `:2011`。

```python
    r"""
```

逐行对应说明：

- 第 1 行：原始字符串前缀 `r` 与三引号共同开始 `get_window` 文档字符串。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2125` 至 `:2125`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2012` 至 `:2012`。

```python
    Return a window of a given length and type.
```

逐行对应说明：

- 第 1 行：中文：返回指定长度和指定类型的窗。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2127` 至 `:2133`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2014` 至 `:2020`。

```python
    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create. See below for more details.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
```

逐行对应说明：

- 第 1 行：这是参数章节标题。
- 第 2 行：短横线是 Parameters 标题的下划线。
- 第 3 行：声明 `window` 参数，可为字符串、浮点数或元组。
- 第 4 行：中文：它指定要创建的窗口类型，细节见下文。
- 第 5 行：声明整数参数 `Nx`。
- 第 6 行：中文：它是窗口的样本数量。
- 第 7 行：声明可选布尔参数 `fftbins`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2134` 至 `:2136`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2021` 至 `:2023`。

```python
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT (see also
        `fftpack.fftfreq`).
```

逐行对应说明：

- 第 1 行：中文：若为 True（默认），创建适合 FFT bin 的周期窗，可
- 第 2 行：中文：与 `ifftshift` 配合并乘到 FFT 结果上；同时提示参考
- 第 3 行：中文：`fftpack.fftfreq`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2137` 至 `:2137`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2024` 至 `:2024`。

```python
        If False, create a "symmetric" window, for use in filter design.
```

逐行对应说明：

- 第 1 行：中文：若为 False，则创建滤波器设计使用的对称窗。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2139` 至 `:2142`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2026` 至 `:2029`。

```python
    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`
```

逐行对应说明：

- 第 1 行：这是返回值章节标题。
- 第 2 行：短横线是 Returns 标题的下划线。
- 第 3 行：文档把返回项命名为 `get_window`，类型为 ndarray。
- 第 4 行：中文：返回长度 `Nx`、类型由 `window` 指定的窗数组。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2144` 至 `:2146`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2031` 至 `:2033`。

```python
    Notes
    -----
    Window types:
```

逐行对应说明：

- 第 1 行：这是 Notes 标题。
- 第 2 行：短横线是 Notes 标题的下划线。
- 第 3 行：中文：以下列出支持的窗口类型。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2148` 至 `:2159`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2035` 至 `:2046`。

```python
    - `~cusignal.windows.windows.boxcar`
    - `~cusignal.windows.windows.triang`
    - `~cusignal.windows.windows.blackman`
    - `~cusignal.windows.windows.hamming`
    - `~cusignal.windows.windows.hann`
    - `~cusignal.windows.windows.bartlett`
    - `~cusignal.windows.windows.flattop`
    - `~cusignal.windows.windows.parzen`
    - `~cusignal.windows.windows.bohman`
    - `~cusignal.windows.windows.blackmanharris`
    - `~cusignal.windows.windows.nuttall`
    - `~cusignal.windows.windows.barthann`
```

逐行对应说明：

- 第 1 行：窗口清单项：支持 `windows`。
- 第 2 行：窗口清单项：支持 `windows`。
- 第 3 行：窗口清单项：支持 `windows`。
- 第 4 行：窗口清单项：支持 `windows`。
- 第 5 行：窗口清单项：支持 `windows`。
- 第 6 行：窗口清单项：支持 `windows`。
- 第 7 行：窗口清单项：支持 `windows`。
- 第 8 行：窗口清单项：支持 `windows`。
- 第 9 行：窗口清单项：支持 `windows`。
- 第 10 行：窗口清单项：支持 `windows`。
- 第 11 行：窗口清单项：支持 `windows`。
- 第 12 行：窗口清单项：支持 `windows`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2160` 至 `:2169`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2047` 至 `:2056`。

```python
    - `~cusignal.windows.windows.kaiser` (needs beta)
    - `~cusignal.windows.windows.gaussian` (needs standard deviation)
    - `~cusignal.windows.windows.general_gaussian` \
            (needs power, width)
    - `~cusignal.windows.windows.slepian` (needs width)
    - `~cusignal.windows.windows.dpss` \
            (needs normalized half-bandwidth)
    - `~cusignal.windows.windows.chebwin` (needs attenuation)
    - `~cusignal.windows.windows.exponential` (needs decay scale)
    - `~cusignal.windows.windows.tukey` (needs taper fraction)
```

逐行对应说明：

- 第 1 行：窗口清单项：支持 `windows`，并提示它需要额外参数。
- 第 2 行：窗口清单项：支持 `windows`，并提示它需要额外参数。
- 第 3 行：列出 `general_gaussian`；末尾反斜杠表示文档源码续行。
- 第 4 行：中文：该窗需要 power 与 width 两个参数；这是上一项的续行。
- 第 5 行：窗口清单项：支持 `windows`，并提示它需要额外参数。
- 第 6 行：列出 `dpss`；末尾反斜杠表示文档源码续行。
- 第 7 行：中文：该窗需要归一化半带宽参数；这是上一项的续行。
- 第 8 行：窗口清单项：支持 `windows`，并提示它需要额外参数。
- 第 9 行：窗口清单项：支持 `windows`，并提示它需要额外参数。
- 第 10 行：窗口清单项：支持 `windows`，并提示它需要额外参数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2171` 至 `:2171`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2058` 至 `:2058`。

```python
    If the window requires no parameters, then `window` can be a string.
```

逐行对应说明：

- 第 1 行：中文：无需参数的窗口可以只用字符串指定。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2173` 至 `:2175`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2060` 至 `:2062`。

```python
    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.
```

逐行对应说明：

- 第 1 行：中文：需要参数的窗口必须用元组指定，
- 第 2 行：中文：元组首项是窗口名称，随后
- 第 3 行：中文：依次放置所需参数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2177` 至 `:2178`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2064` 至 `:2065`。

```python
    If `window` is a floating point number, it is interpreted as the beta
    parameter of the `~cusignal.windows.windows.kaiser` window.
```

逐行对应说明：

- 第 1 行：中文：若 `window` 是浮点数，则把它解释为
- 第 2 行：中文：Kaiser 窗的 beta 参数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2180` 至 `:2182`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2067` 至 `:2069`。

```python
    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.
```

逐行对应说明：

- 第 1 行：中文：上面列出的每种窗口类型同时也是
- 第 2 行：中文：一个可以直接调用来创建对应窗口的函数，
- 第 3 行：中文：本行结束该说明。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2184` 至 `:2194`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2071` 至 `:2081`。

```python
    Examples
    --------
    >>> import cusignal
    >>> cusignal.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> cusignal.get_window(('kaiser', 4.0), 9)
    array([0.08848053, 0.32578323, 0.63343178, 0.89640418, 1.,
           0.89640418, 0.63343178, 0.32578323, 0.08848053])
    >>> cusignal.get_window(4.0, 9)
    array([0.08848053, 0.32578323, 0.63343178, 0.89640418, 1.,
           0.89640418, 0.63343178, 0.32578323, 0.08848053])
```

逐行对应说明：

- 第 1 行：这是示例章节标题。
- 第 2 行：短横线是 Examples 标题的下划线。
- 第 3 行：示例导入 `cusignal`。
- 第 4 行：示例调用：生成长度 7 的周期三角窗。
- 第 5 行：展示上一调用返回的 7 个窗系数。
- 第 6 行：示例调用：用元组指定 Kaiser 窗及 `beta=4.0`，长度为 9。
- 第 7 行：展示 Kaiser 窗输出数组的第一行。
- 第 8 行：续写并闭合上一行的数组输出。
- 第 9 行：示例调用：直接用浮点数 4.0 表示 Kaiser beta。
- 第 10 行：展示数值简写形式所得数组的第一行。
- 第 11 行：续写并闭合数组；结果与元组形式一致。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2196` 至 `:2196`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2083` 至 `:2083`。

```python
    """
```

逐行对应说明：

- 第 1 行：三引号结束 `get_window` 文档字符串。



### 3.3 `_get_fs` docstring

`_get_fs` 的 24–26 行也构成完整 docstring，其逐行中文含义已纳入下一节 `_get_fs` 表的 24–26 行，未遗漏。

## 4. 源代码逐行解释

以下语义块覆盖第 2 节源码块中除已在第 3 节单列的 `firwin2` 与 `get_window` docstring 之外的每一行非空内容；连续签名、调用、分支和表达式保持为最小完整语义块，块内再按顺序说明每一行。

### 4.1 顶层公开导出

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:37` 至 `:45`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:32` 至 `:38`。

```python
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    firwin,
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
```

逐行对应说明：

- 第 1 行：`from ... import (` 开始一个带括号的多行导入，把实现模块中的选定符号绑定到 `cusignal` 顶层命名空间。
- 第 2 行：导入辅助 API `cmplx_sort`；末尾逗号表示列表继续。
- 第 3 行：导入 `firwin`，使用户能调用 `cusignal.firwin`。
- 第 4 行：导入本算子 `firwin2`，形成公开路径 `cusignal.firwin2`；这一行只是符号绑定，不执行滤波器设计。
- 第 5 行：导入 `kaiser_atten`。
- 第 6 行：导入 `kaiser_beta`。
- 第 7 行：右括号结束多行导入语句。


### 4.2 filter_design 子包公开导出

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/__init__.py:15` 至 `:23`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/__init__.py:14` 至 `:20`。

```python
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    firwin,
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
```

逐行对应说明：

- 第 1 行：多行 `from ... import (...)` 语句开始，把实现符号绑定到 `cusignal.filter_design` 子包。
- 第 2 行：导入 `cmplx_sort`。
- 第 3 行：导入 `firwin`。
- 第 4 行：导入 `firwin2`，形成第二条公开路径 `cusignal.filter_design.firwin2`。
- 第 5 行：导入 `kaiser_atten`。
- 第 6 行：导入 `kaiser_beta`。
- 第 7 行：右括号结束多行导入；该文件没有核心数值计算。


### 4.3 _get_fs

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:29` 至 `:32`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:23` 至 `:26`。

```python
def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
```

逐行对应说明：

- 第 1 行：`def` 定义内部函数 `_get_fs`；括号内是 `fs` 与 `nyq` 两个形参，冒号开始函数体。
- 第 2 行：三引号开始 `_get_fs` 的 docstring。
- 第 3 行：中文：该 helper 用新参数 `fs` 替代默认值为 1 的旧参数 `nyq`。
- 第 4 行：三引号结束 `_get_fs` 的 docstring。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:34` 至 `:35`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:27` 至 `:28`。

```python
    if nyq is None and fs is None:
        fs = 2
```

逐行对应说明：

- 第 1 行：`if` 用 `is None` 同时判断两个参数都未提供。
- 第 2 行：把默认采样频率赋为 2，因此默认 Nyquist 频率是 1。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:37` 至 `:37`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:29` 至 `:29`。

```python
    elif nyq is not None:
```

逐行对应说明：

- 第 1 行：`elif` 表示前一条件失败且旧参数 `nyq` 已提供时进入兼容分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:39` 至 `:42`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:30` 至 `:32`。

```python
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        fs = 2 * nyq
```

逐行对应说明：

- 第 1 行：嵌套 `if` 检查调用者是否同时给了新旧两个互斥参数。
- 第 2 行：`raise ValueError(...)` 立即抛出参数冲突异常。
- 第 3 行：依据 $f_s=2f_N$ 将旧 `nyq` 换算成统一的 `fs`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:44` 至 `:44`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:33` 至 `:33`。

```python
    return fs
```

逐行对应说明：

- 第 1 行：`return` 把统一后的采样频率返回给调用者。


### 4.4 firwin2

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:506` 至 `:516`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:399` 至 `:409`。

```python
def firwin2(
    numtaps,
    freq,
    gain,
    nfreqs=None,
    window="hamming",
    nyq=None,
    antisymmetric=False,
    fs=None,
    gpupath=True,
):
```

逐行对应说明：

- 第 1 行：`def` 开始多行函数定义，函数名为 `firwin2`。
- 第 2 行：第一个必需形参 `numtaps` 表示抽头数；逗号允许签名继续。
- 第 3 行：第二个必需形参 `freq` 表示频率控制点。
- 第 4 行：第三个必需形参 `gain` 表示对应目标增益。
- 第 5 行：可选 `nfreqs` 默认 `None`，稍后自动选择网格长度。
- 第 6 行：可选 `window` 默认字符串 `hamming`。
- 第 7 行：兼容旧参数 `nyq`，默认 `None`。
- 第 8 行：`antisymmetric=False` 默认选择对称 FIR。
- 第 9 行：`fs=None` 允许 helper 应用默认采样频率。
- 第 10 行：`gpupath=True` 默认选择 CuPy 运算路径。
- 第 11 行：右括号结束参数列表，冒号开始函数体。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:598` 至 `:600`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:490` 至 `:491`。

```python
    if gpupath:
        pp = cp
```

逐行对应说明：

- 第 1 行：按 `gpupath` 选择统一数组后端。
- 第 2 行：把局部别名 `pp` 绑定为 CuPy 模块 `cp`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:601` 至 `:603`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:492` 至 `:493`。

```python
    else:
        pp = np
```

逐行对应说明：

- 第 1 行：`else` 对应 CPU 辅助路径。
- 第 2 行：把 `pp` 绑定为 NumPy 模块 `np`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:606` 至 `:606`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:495` 至 `:495`。

```python
    nyq = 0.5 * _get_fs(fs, nyq)
```

逐行对应说明：

- 第 1 行：调用 `_get_fs` 统一参数，再乘 0.5 得到 Nyquist 频率。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:609` 至 `:610`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:497` 至 `:498`。

```python
    if len(freq) != len(gain):
        raise ValueError("freq and gain must be of same length.")
```

逐行对应说明：

- 第 1 行：比较两个控制点数组的 Python 长度。
- 第 2 行：长度不同时抛出 `ValueError`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:613` 至 `:620`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:500` 至 `:507`。

```python
    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(
            (
                "ntaps must be less than nfreqs, but firwin2 was "
                "called with ntaps=%d and nfreqs=%s"
            )
            % (numtaps, nfreqs)
        )
```

逐行对应说明：

- 第 1 行：短路条件同时检查 `nfreqs` 已显式给出且不大于 `numtaps`。
- 第 2 行：开始一个跨多行的 `ValueError` 构造。
- 第 3 行：额外括号把多段字符串组成一个格式模板。
- 第 4 行：相邻字符串字面量会在编译期自动拼接；这是错误信息前半句。
- 第 5 行：错误信息续行包含 `%d` 与 `%s` 占位符。
- 第 6 行：右括号结束字符串分组。
- 第 7 行：`%` 运算符用 `(numtaps, nfreqs)` 填充两个占位符。
- 第 8 行：右括号结束 `ValueError` 调用。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:623` 至 `:626`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:509` 至 `:511`。

```python
    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError("freq must start with 0 and end with fs/2.")
    d = pp.diff(freq)
```

逐行对应说明：

- 第 1 行：索引首尾控制点，要求分别严格等于 0 与 Nyquist。
- 第 2 行：端点不满足时抛出异常。
- 第 3 行：`pp.diff` 计算相邻频率差数组 `d`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:628` 至 `:631`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:512` 至 `:514`。

```python
    if (d < 0).any():
        raise ValueError("The values in freq must be nondecreasing.")
    d2 = d[:-1] + d[1:]
```

逐行对应说明：

- 第 1 行：`d < 0` 逐元素比较，`.any()` 判断是否存在降序。
- 第 2 行：发现降序时抛出异常。
- 第 3 行：相邻差分相加；若结果为零，说明同一频率至少连续出现三次。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:632` 至 `:633`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:515` 至 `:516`。

```python
    if (d2 == 0).any():
        raise ValueError("A value in freq must not occur more than twice.")
```

逐行对应说明：

- 第 1 行：逐元素等零后用 `.any()` 汇总。
- 第 2 行：内部频率出现超过两次时抛出异常。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:635` 至 `:636`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:517` 至 `:518`。

```python
    if freq[1] == 0:
        raise ValueError("Value 0 must not be repeated in freq")
```

逐行对应说明：

- 第 1 行：检查第二个控制点是否也等于 0，从而识别重复直流端点。
- 第 2 行：重复直流端点时抛出异常。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:638` 至 `:639`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:519` 至 `:520`。

```python
    if freq[-2] == nyq:
        raise ValueError("Value fs/2 must not be repeated in freq")
```

逐行对应说明：

- 第 1 行：检查倒数第二个控制点是否已等于 Nyquist。
- 第 2 行：重复 Nyquist 端点时抛出异常。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:642` 至 `:642`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:522` 至 `:522`。

```python
    if antisymmetric:
```

逐行对应说明：

- 第 1 行：根据 `antisymmetric` 进入反对称或对称类型分派。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:643` 至 `:645`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:523` 至 `:524`。

```python
        if numtaps % 2 == 0:
            ftype = 4
```

逐行对应说明：

- 第 1 行：`numtaps % 2 == 0` 用余数判断偶数长度。
- 第 2 行：偶长反对称赋类型号 4。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:646` 至 `:648`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:525` 至 `:526`。

```python
        else:
            ftype = 3
```

逐行对应说明：

- 第 1 行：`else` 对应奇数长度。
- 第 2 行：奇长反对称赋类型号 3。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:649` 至 `:649`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:527` 至 `:527`。

```python
    else:
```

逐行对应说明：

- 第 1 行：外层 `else` 对应对称响应。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:650` 至 `:652`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:528` 至 `:529`。

```python
        if numtaps % 2 == 0:
            ftype = 2
```

逐行对应说明：

- 第 1 行：再次检查抽头数奇偶。
- 第 2 行：偶长对称赋类型号 2。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:653` 至 `:655`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:530` 至 `:531`。

```python
        else:
            ftype = 1
```

逐行对应说明：

- 第 1 行：`else` 对应奇长对称。
- 第 2 行：奇长对称赋类型号 1。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:658` 至 `:661`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:533` 至 `:536`。

```python
    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError(
            "A Type II filter must have zero gain at the " "Nyquist frequency."
        )
```

逐行对应说明：

- 第 1 行：II 型若 Nyquist 增益不为浮点零则拒绝。
- 第 2 行：开始跨行构造 `ValueError`。
- 第 3 行：两个相邻字符串字面量自动拼成完整错误消息。
- 第 4 行：关闭异常构造。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:663` 至 `:666`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:537` 至 `:540`。

```python
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError(
            "A Type III filter must have zero gain at zero " "and Nyquist frequencies."
        )
```

逐行对应说明：

- 第 1 行：`elif` 检查 III 型的直流或 Nyquist 任一端非零。
- 第 2 行：开始 III 型错误异常。
- 第 3 行：相邻字符串拼出完整的两端点约束消息。
- 第 4 行：关闭异常构造。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:668` 至 `:669`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:541` 至 `:542`。

```python
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero " "frequency.")
```

逐行对应说明：

- 第 1 行：`elif` 检查 IV 型直流增益非零。
- 第 2 行：单行抛出 IV 型直流约束异常；相邻字符串仍自动拼接。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:672` 至 `:674`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:544` 至 `:545`。

```python
    if nfreqs is None:
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))
```

逐行对应说明：

- 第 1 行：`nfreqs` 未给出时才计算默认值。
- 第 2 行：用 `ceil(log2(numtaps))` 找最小二次幂指数，再计算二次幂加一。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:677` 至 `:682`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:547` 至 `:550`。

```python
    if (d == 0).any():
        # Tweak any repeated values in freq so that interp works.
        freq = pp.array(freq, copy=True)
        eps = pp.finfo(float).eps * nyq
```

逐行对应说明：

- 第 1 行：若差分中有零，则存在重复内部频率控制点。
- 第 2 行：原注释：调整重复频率，使插值函数可以工作。
- 第 3 行：`pp.array(..., copy=True)` 创建副本，避免原地修改调用者数据。
- 第 4 行：读取双精度机器 epsilon，并按 Nyquist 尺度缩放。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:684` 至 `:684`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:551` 至 `:551`。

```python
        for k in range(len(freq) - 1):
```

逐行对应说明：

- 第 1 行：`range(len(freq)-1)` 遍历所有相邻点对。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:686` 至 `:691`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:552` 至 `:556`。

```python
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # Check if freq is strictly increasing after tweak
        d = pp.diff(freq)
```

逐行对应说明：

- 第 1 行：比较当前点与下一个点是否相等。
- 第 2 行：把重复点的左侧值减去 `eps`。
- 第 3 行：把重复点的右侧值加上 `eps`。
- 第 4 行：原注释：扰动后检查 `freq` 是否严格递增。
- 第 5 行：重新计算扰动后的相邻差分。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:693` 至 `:698`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:557` 至 `:562`。

```python
        if (d <= 0).any():
            raise ValueError(
                "freq cannot contain numbers that are too close "
                "(within eps * (fs/2): "
                "{}) to a repeated value".format(eps)
            )
```

逐行对应说明：

- 第 1 行：若仍存在非正差分，则扰动失败。
- 第 2 行：开始构造跨行异常。
- 第 3 行：第一段错误字符串说明控制点太接近。
- 第 4 行：第二段字符串给出容差定义的开头。
- 第 5 行：`.format(eps)` 把实际 epsilon 填入 `{}`。
- 第 6 行：关闭异常构造。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:700` 至 `:702`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:564` 至 `:565`。

```python
    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = pp.linspace(0.0, nyq, nfreqs)
```

逐行对应说明：

- 第 1 行：原注释：在均匀网格 `x` 上线性插值目标响应。
- 第 2 行：`linspace` 在 0 与 Nyquist 间生成含端点的 `nfreqs` 个等距点。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:704` 至 `:706`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:566` 至 `:567`。

```python
    if gpupath:
        fx = cp.asarray(np.interp(cp.asnumpy(x), freq, gain))
```

逐行对应说明：

- 第 1 行：GPU 路径单独处理，因为使用的是 NumPy 插值。
- 第 2 行：`cp.asnumpy` 把网格送到主机，`np.interp` 插值，再由 `cp.asarray` 送回设备。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:707` 至 `:709`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:568` 至 `:569`。

```python
    else:
        fx = np.interp(x, freq, gain)
```

逐行对应说明：

- 第 1 行：`else` 对应 NumPy 路径。
- 第 2 行：直接在主机上执行一维分段线性插值。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:711` 至 `:714`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:571` 至 `:573`。

```python
    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = pp.exp(-(numtaps - 1) / 2.0 * 1.0j * pp.pi * x / nyq)
```

逐行对应说明：

- 第 1 行：原注释第一行：调整系数相位，使逆 FFT 的前若干点成为目标系数。
- 第 2 行：原注释第二行：说明上一行注释中的逆 FFT 语义。
- 第 3 行：构造 $e^{-j(M-1)\omega/2}$；其中 `pi*x/nyq` 是数字角频率。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:716` 至 `:717`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:574` 至 `:575`。

```python
    if ftype > 2:
        shift *= 1j
```

逐行对应说明：

- 第 1 行：类型号大于 2 表示 III 或 IV 型。
- 第 2 行：原地乘 `1j`，加入反对称响应所需的常相位因子。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:720` 至 `:720`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:577` 至 `:577`。

```python
    fx2 = fx * shift
```

逐行对应说明：

- 第 1 行：逐元素相乘，得到带线性相位的复单边频谱 `fx2`。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:722` 至 `:724`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:579` 至 `:580`。

```python
    # Use irfft to compute the inverse FFT.
    out_full = pp.fft.irfft(fx2)
```

逐行对应说明：

- 第 1 行：原注释：使用 `irfft` 计算逆 FFT。
- 第 2 行：实数逆 FFT 自动按 Hermitian 对称补齐负频率。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:726` 至 `:726`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:582` 至 `:582`。

```python
    # Pass to device memory
```

逐行对应说明：

- 第 1 行：原注释：需要时把结果传到设备内存。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:728` 至 `:730`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:583` 至 `:584`。

```python
    if not gpupath:
        out_full = cp.asarray(out_full)
```

逐行对应说明：

- 第 1 行：若前面选择了 NumPy 路径。
- 第 2 行：用 `cp.asarray` 把主机结果转成 CuPy 设备数组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:733` 至 `:736`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:586` 至 `:588`。

```python
    if window is not None:
        # Create the window to apply to the filter coefficients.
        wind = get_window(window, numtaps, fftbins=False)
```

逐行对应说明：

- 第 1 行：非空窗口描述才调用窗口工厂。
- 第 2 行：原注释：创建要乘到滤波器系数上的窗。
- 第 3 行：调用 `get_window`；`fftbins=False` 请求滤波器设计用对称窗。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:737` 至 `:739`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:589` 至 `:590`。

```python
    else:
        wind = 1
```

逐行对应说明：

- 第 1 行：`else` 对应 `window is None`。
- 第 2 行：标量 1 会广播到全部系数，相当于不加额外窗。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:741` 至 `:744`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:592` 至 `:594`。

```python
    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind
```

逐行对应说明：

- 第 1 行：原注释第一行：只保留前 `numtaps` 个系数并乘窗。
- 第 2 行：原注释第二行：完成上一行注释。
- 第 3 行：切片截取前 $M$ 点，再与窗数组逐元素相乘。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:747` 至 `:749`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:596` 至 `:597`。

```python
    if ftype == 3:
        out[out.size // 2] = 0.0
```

逐行对应说明：

- 第 1 行：III 型需要额外修正中心抽头。
- 第 2 行：`out.size // 2` 定位奇长数组中心并精确置零。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:752` 至 `:752`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/fir_filter_design.py:599` 至 `:599`。

```python
    return out
```

逐行对应说明：

- 第 1 行：返回最终 CuPy FIR 系数数组。


### 4.5 get_window

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2123` 至 `:2123`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2010` 至 `:2010`。

```python
def get_window(window, Nx, fftbins=True):
```

逐行对应说明：

- 第 1 行：`def` 定义窗口工厂 `get_window`；`fftbins` 默认 True。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2198` 至 `:2198`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2084` 至 `:2084`。

```python
    sym = not fftbins
```

逐行对应说明：

- 第 1 行：将公开参数 `fftbins` 逻辑取反为底层窗函数使用的 `sym`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2200` 至 `:2202`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2085` 至 `:2086`。

```python
    try:
        beta = float(window)
```

逐行对应说明：

- 第 1 行：`try` 开始尝试数值形式窗口参数。
- 第 2 行：把 `window` 转为浮点 `beta`；成功则代表 Kaiser 简写。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2204` 至 `:2206`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2087` 至 `:2088`。

```python
    except (TypeError, ValueError):
        args = ()
```

逐行对应说明：

- 第 1 行：只捕获类型错误或值错误，转入名称/元组解析。
- 第 2 行：先把额外参数初始化为空元组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2208` 至 `:2210`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2089` 至 `:2090`。

```python
        if isinstance(window, tuple):
            winstr = window[0]
```

逐行对应说明：

- 第 1 行：检查窗口描述是否为元组。
- 第 2 行：元组首元素作为窗口名称。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2212` 至 `:2214`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2091` 至 `:2092`。

```python
            if len(window) > 1:
                args = window[1:]
```

逐行对应说明：

- 第 1 行：若元组还包含其他元素。
- 第 2 行：切片取得全部额外参数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2216` 至 `:2216`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2093` 至 `:2093`。

```python
        elif isinstance(window, str):
```

逐行对应说明：

- 第 1 行：否则若描述本身是字符串。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2218` 至 `:2222`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2094` 至 `:2098`。

```python
            if window in _needs_param:
                raise ValueError(
                    "The '" + window + "' window needs one or "
                    "more parameters -- pass a tuple."
                )
```

逐行对应说明：

- 第 1 行：检查该名称是否属于必须提供额外参数的窗口。
- 第 2 行：开始跨行构造参数缺失异常。
- 第 3 行：拼接窗口名称与错误消息第一段。
- 第 4 行：错误消息续行提示改用元组。
- 第 5 行：关闭异常构造。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2223` 至 `:2225`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2099` 至 `:2100`。

```python
            else:
                winstr = window
```

逐行对应说明：

- 第 1 行：`else` 表示该字符串窗不需要额外参数。
- 第 2 行：直接把字符串保存为查表键。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2227` 至 `:2228`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2101` 至 `:2102`。

```python
        else:
            raise ValueError("%s as window type is not supported." % str(type(window)))
```

逐行对应说明：

- 第 1 行：外层 `else` 表示类型既非元组也非字符串。
- 第 2 行：把实际 Python 类型格式化进不支持类型异常。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2231` 至 `:2233`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2104` 至 `:2105`。

```python
        try:
            winfunc = _win_equiv[winstr]
```

逐行对应说明：

- 第 1 行：开始在窗口别名映射表中查找。
- 第 2 行：用 `winstr` 索引 `_win_equiv` 得到具体可调用函数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2235` 至 `:2236`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2106` 至 `:2107`。

```python
        except KeyError:
            raise ValueError("Unknown window type.")
```

逐行对应说明：

- 第 1 行：捕获未知名称产生的 `KeyError`。
- 第 2 行：对外转换成语义清楚的 `ValueError`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2239` 至 `:2239`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2109` 至 `:2109`。

```python
        params = (Nx,) + args + (sym,)
```

逐行对应说明：

- 第 1 行：拼接底层窗函数的位置参数：长度、额外参数、对称标志。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2241` 至 `:2245`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2110` 至 `:2112`。

```python
    else:
        winfunc = kaiser
        params = (Nx, beta, sym)
```

逐行对应说明：

- 第 1 行：`try` 的 `else` 仅在浮点转换没有抛异常时执行。
- 第 2 行：数值形式固定选择 `kaiser` 函数。
- 第 3 行：Kaiser 参数顺序为长度、beta、对称标志。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2248` 至 `:2248`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2114` 至 `:2114`。

```python
    return winfunc(*params)
```

逐行对应说明：

- 第 1 行：`*params` 展开位置参数并调用选中的窗函数，直接返回结果。



## 5. 调用链与算法总结

### 5.1 从公开 API 到最终系数

```text
cusignal.firwin2
  → fir_filter_design.firwin2
    → _get_fs(fs, nyq)
    → 参数与四型端点约束检查
    → numpy.interp：控制点到均匀单边频率网格
    → exp：加入线性相位；III/IV 型额外乘 1j
    → cupy.fft.irfft 或 numpy.fft.irfft
    → get_window(window, numtaps, fftbins=False)
      → _win_equiv 名称分派或 Kaiser 数值简写
      → 具体窗函数
    → 截取前 numtaps 点并乘窗
    → III 型中心抽头精确置零
    → 返回 cupy.ndarray
```

### 5.2 按实际执行顺序拆解

1. `gpupath` 选择 `pp=cp` 或 `pp=np`，让大部分数组代码复用同一写法。
2. `_get_fs` 把默认值、`fs` 和旧 `nyq` 统一成采样频率，再得到 `nyq=fs/2`。
3. 检查控制点数量、频率顺序、端点、重复次数及 `numtaps<nfreqs`。
4. 根据长度奇偶与 `antisymmetric` 选定 I、II、III 或 IV 型，并验证强制零点。
5. 自动选择或接受 `nfreqs`，把内部重复控制点左右扰动一个机器 epsilon 尺度。
6. 用 `linspace` 创建均匀网格，用 `interp` 生成期望实值增益 `fx`。
7. 用复指数加入群时延 $(M-1)/2$；反对称型再乘 $j$。
8. 用 `irfft` 得到实值周期冲激响应；CPU 辅助路径随后复制到 GPU。
9. `get_window` 解析窗口描述；`fftbins=False` 保证使用对称窗。
10. 截取前 $M$ 点、逐元素乘窗，III 型中心置零并返回。

### 5.3 GPU 与 CPU 辅助路径的实际差别

| 步骤 | `gpupath=True` | `gpupath=False` |
| --- | --- | --- |
| 差分、网格、相移 | CuPy | NumPy |
| 线性插值 | 网格临时转 NumPy，`np.interp` 后转回 CuPy | 直接 `np.interp` |
| 逆 FFT | `cupy.fft.irfft` | `numpy.fft.irfft` |
| 逆 FFT 后数据 | 已在设备 | 通过 `cp.asarray` 复制到设备 |
| 乘窗与返回 | CuPy | 转设备后仍以 CuPy 返回 |

`gpupath=False` 不是一个独立 CPU 公共实现，因为其最终结果仍被转换到设备内存；它更准确地说是“用 NumPy 辅助完成插值与逆 FFT 的路径”。

## 6. 数学公式与代码逐项映射

设 $M=\texttt{numtaps}$、$K=\texttt{nfreqs}$、$f_N=f_s/2$。

| 数学步骤 | 公式 | Python 落实位置（只读基准） |
| --- | --- | --- |
| 均匀单边网格 | $x_k=kf_N/(K-1)$ | `fir_filter_design.py:565` |
| 控制点线性插值 | $F_k=A_d(x_k)$ | `fir_filter_design.py:566-569` |
| 群时延 | $\alpha=(M-1)/2$ | `fir_filter_design.py:573` |
| 数字角频率 | $\omega_k=\pi x_k/f_N$ | `fir_filter_design.py:573` |
| I/II 型频谱 | $G_k=F_ke^{-j\alpha\omega_k}$ | `fir_filter_design.py:573,577` |
| III/IV 型频谱 | $G_k=jF_ke^{-j\alpha\omega_k}$ | `fir_filter_design.py:574-577` |
| 实数 IDFT | $g=\operatorname{irfft}(G)$ | `fir_filter_design.py:580` |
| 截取乘窗 | $h[n]=g[n]w[n],\ 0\le n<M$ | `fir_filter_design.py:586-594` |
| III 型中心约束 | $h[(M-1)/2]=0$ | `fir_filter_design.py:596-597` |

`irfft` 输入有 $K$ 个单边样本且没有显式 `n` 参数，因此默认完整变换长度为

$$
L=2(K-1).
$$

`shift` 中的表达式可直接整理为

$$
\exp\left[-\frac{M-1}{2}j\pi\frac{x_k}{f_N}\right]
=e^{-j\alpha\omega_k}.
$$

这正是阶段一文档中的广义线性相位因子。

## 7. 底层机制

### 7.1 `numpy.interp`

本次调用完成一维分段线性插值。输入 `freq` 是横坐标、`gain` 是纵坐标、`x` 是均匀查询网格。GPU 路径仍调用 NumPy 版本，因此存在一次设备到主机和一次主机到设备的数据搬运。

### 7.2 `irfft`

`irfft` 把 `fx2` 解释为实序列频谱的非负频率部分，按 Hermitian 共轭对称补齐负频率，再执行逆 FFT。该性质保证 `out_full` 为实数组，而不需要手工构造完整复频谱。

### 7.3 `get_window`

`get_window` 是窗口工厂而不是窗函数核心公式：

- 数值 `window` 直接解释为 Kaiser `beta`；
- 元组把第一项作为名称、后续项作为具体窗函数参数；
- 无参数字符串通过 `_win_equiv` 查到具体窗函数；
- `firwin2` 固定传 `fftbins=False`，所以 `sym=True`。

本阶段没有继续展开 Hamming、Kaiser 等具体窗函数源码，因为它们是独立算子；对 `firwin2` 而言，只需确认工厂返回长度 $M$ 的对称窗。

## 8. 边界处理、异常与 dtype

### 8.1 已实现的输入门禁

- `len(freq) == len(gain)`；
- 显式 `nfreqs > numtaps`；
- `freq[0] == 0` 且 `freq[-1] == fs/2`；
- `freq` 非递减；
- 内部频率最多出现两次；
- 0 与 Nyquist 不能重复；
- II、III、IV 型满足各自端点零增益约束；
- `window` 名称或参数形式能被 `get_window` 解析。

### 8.2 值得注意的未显式检查

源码没有在函数入口单独检查以下条件：

- `numtaps` 是否为正整数；
- `freq`、`gain` 是否至少含两个元素；
- `gain` 是否严格为实数；
- `fs` 是否为正；
- 自动计算 `nfreqs` 前 `numtaps` 是否适合 `log`。

这些输入可能在索引、对数、插值或底层数组函数中以其他异常形式失败，不能把缺少显式检查误写成“支持任意值”。

### 8.3 数值稳定性

- 重复内部频率用 `finfo(float).eps * nyq` 分离；若附近还有过近控制点，代码会在重新差分后拒绝输入。
- 线性相位复指数和 FFT 会产生舍入误差，因此 III 型中心抽头被显式置为精确零。
- 控制点跳变经过有限长度截断后仍会出现振铃；这是逼近误差，不是递归稳定性问题。
- FIR 系数有限时系统 BIBO 稳定，但幅度逼近精度仍取决于 $M$、控制点和窗口。

### 8.4 shape 与广播

- `x`、`fx`、`fx2` 的 shape 都是 `(nfreqs,)`；
- 默认 `irfft(fx2)` 的 `out_full` shape 是 `(2*(nfreqs-1),)`；
- `wind` 是 `(numtaps,)`，或在 `window=None` 时为标量 `1`；
- `out_full[:numtaps] * wind` 产生 `(numtaps,)`；标量 `1` 通过广播不改变数组。

## 9. 复杂度与性能瓶颈

令 $P=\operatorname{len}(\texttt{freq})$、$K=\texttt{nfreqs}$、$M=\texttt{numtaps}$，且默认 $M<K$。

| 步骤 | 时间复杂度 | 额外空间 |
| --- | ---: | ---: |
| 参数差分与重复点扫描 | $O(P)$ | $O(P)$ |
| `linspace` 与线性插值 | $O(K+P)$ | $O(K)$ |
| 相位向量与逐点乘法 | $O(K)$ | $O(K)$ |
| `irfft`，长度 $L=2(K-1)$ | $O(K\log K)$ | $O(K)$ |
| 窗生成、截取与乘法 | $O(M)$ | $O(M)$ |

主导项通常是逆 FFT 的 $O(K\log K)$。GPU 路径的明显工程瓶颈是 `cp.asnumpy(x) → np.interp → cp.asarray(...)` 的同步与数据搬运；当 $K$ 较小时，传输开销可能比插值计算本身更显著。窗工厂的名称分派是常数级，具体窗生成是 $O(M)$。

## 10. 阅读顺序与检查问题

建议按下列顺序亲自阅读：

1. 顶层与子包 `__init__.py`：确认公开 API 只是符号导出，不是算法核心。
2. `_get_fs`：观察兼容参数如何统一，以及为什么最后还要乘 0.5。
3. `firwin2` 497–542 行：逐项对应参数门禁和四型端点约束。
4. 544–569 行：区分控制点扰动、均匀网格和线性插值。
5. 571–580 行：把 `shift` 逐项映射到 $e^{-j\alpha\omega}$，再确认 `irfft` 的长度。
6. 582–599 行：观察设备搬运、对称窗、切片和 III 型中心置零。
7. `get_window` 2084–2114 行：只跟踪 `firwin2` 默认字符串 `"hamming"` 会走的分支，再比较元组与数值分支。

阅读后应能回答：

1. 为什么 `freq` 允许内部值重复一次，却禁止端点重复？
2. 为什么 `nfreqs` 默认值是二次幂加一，而 `irfft` 的完整长度仍是二次幂的两倍？
3. `shift` 中的 `pi*x/nyq` 为什么就是 $\omega$？
4. `ftype > 2` 为什么恰好覆盖 III、IV 型？
5. `gpupath=True` 为什么仍会调用 NumPy？这造成什么性能代价？
6. `fftbins=False` 对保持线性相位有什么作用？
7. 为什么 `window=None` 用标量 1 就能工作？
8. 为什么显式清零只针对 III 型中心抽头？
9. 哪些代码是数学算法，哪些只是输入检查、后端选择或数据搬运？
10. 为什么该实现不是最小二乘或等纹波优化？

## 11. 自检问题参考答案

### 答案 1：内部重复频率与端点重复

内部频率重复一次用来表达同一频率处不同的左、右极限，即频响跳变。源码在基准 `fir_filter_design.py:547-562` 把重复的两个点分别向左右扰动 `eps` 后交给 `interp`。直流和 Nyquist 是单边频谱边界，没有两侧可供这种表示，而且 `fir_filter_design.py:517-520` 明确拒绝端点重复。

### 答案 2：`nfreqs` 与 `irfft` 长度

默认

$$
K=1+2^{\lceil\log_2M\rceil}.
$$

`K` 是包含直流和 Nyquist 的单边频谱点数。实序列完整频谱长度满足 $K=L/2+1$，所以

$$
L=2(K-1)=2^{\lceil\log_2M\rceil+1},
$$

仍是二次幂。对应基准 `fir_filter_design.py:544-545,579-580`。

### 答案 3：`pi*x/nyq` 的含义

`x` 使用与 `nyq=f_s/2` 相同的频率单位，因此

$$
\pi\frac{x}{\texttt{nyq}}=pi\frac{f}{f_s/2}=\frac{2\pi f}{f_s}=\omega.
$$

所以基准 `fir_filter_design.py:573` 的指数就是 $e^{-j(M-1)\omega/2}$。

### 答案 4：`ftype > 2`

基准 `fir_filter_design.py:522-531` 只会产生类型号 1、2、3、4；其中 3、4 恰好是反对称类型。因此 `ftype > 2` 精确覆盖 III、IV 型，并在 `:574-575` 额外乘 $j$。

### 答案 5：GPU 路径中的 NumPy

基准 `fir_filter_design.py:566-569` 显示 GPU 路径先用 `cp.asnumpy(x)` 把网格复制到主机，调用 `np.interp`，再用 `cp.asarray` 复制回设备。这样复用了 NumPy 插值，但增加同步以及 D2H/H2D 小数组传输；`nfreqs` 较小时，传输固定开销可能超过插值本身。

### 答案 6：`fftbins=False`

`firwin2` 在基准 `fir_filter_design.py:588` 传入 `fftbins=False`。`get_window` 在 `windows.py:2084` 计算 `sym = not fftbins`，因此得到 `sym=True` 的对称窗。对称窗乘以对称或反对称冲激响应后保持原有对称类型，从而保持广义线性相位。

### 答案 7：`window=None` 与标量广播

基准 `fir_filter_design.py:589-594` 把 `wind` 设为标量 `1`，随后执行数组乘法 `out_full[:numtaps] * wind`。NumPy/CuPy 标量广播使每个元素都乘 1，所以无需创建全 1 数组。

### 答案 8：III 型中心清零

III 型是奇长反对称序列，中点满足

$$
h[(M-1)/2]=-h[(M-1)/2],
$$

故只能为 0。IV 型长度为偶数，没有单个中心抽头；II 型虽为偶长对称，也没有这一约束。对应基准 `fir_filter_design.py:596-597`。

### 答案 9：数学算法与工程代码

数学算法包括 `:564-580` 的均匀网格、线性插值、相位构造与 `irfft`，以及 `:586-597` 的乘窗和 III 型约束。`:490-520` 主要是后端选择与通用输入检查，`:547-562` 是跳变插值的数值工程处理，`:582-584` 是主机到设备的数据搬运。

### 答案 10：为什么不是最小二乘或等纹波

函数体没有建立误差目标、权重矩阵、正规方程，也没有 Remez 交换迭代。它直接执行“控制点插值 → 相位构造 → IDFT → 截取乘窗”，所以是直接频率采样/窗函数构造，而不是 `firls` 的最小二乘解或 `remez` 的 minimax 等纹波解。
