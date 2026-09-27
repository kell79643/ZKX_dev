# CSD (Cross Spectral Density) 算子 Python 源码算法

## 一、代码定位与接口概览

### 1.1 公开导入路径

```
cusignal.csd(x, y, ...)
```

### 1.2 定义文件与符号

| 项目 | 路径 | 符号名 | 只读基准行号 |
|-----|------|-------|------------|
| 导出入口 | `cusignal/spectral_analysis/__init__.py` | `csd` | 第16行 |
| 函数定义 | `cusignal/spectral_analysis/spectral.py` | `csd` | 第491-656行 |
| 核心辅助 | `cusignal/spectral_analysis/spectral.py` | `_spectral_helper` | 第1554-1869行 |
| FFT辅助 | `cusignal/spectral_analysis/spectral.py` | `_fft_helper` | 第1872-1918行 |
| 窗解析辅助 | `cusignal/spectral_analysis/spectral.py` | `_triage_segments` | 第1921-1981行 |
| 中值偏差 | `cusignal/spectral_analysis/spectral.py` | `_median_bias` | 第1984-2002行 |

所有路径相对于共享根目录 `ZKX/cusignal-23.08.00/python/`。

### 1.3 函数签名

```python
def csd(
    x, y, fs=1.0, window="hann", nperseg=None, noverlap=None,
    nfft=None, detrend="constant", return_onesided=True,
    scaling="density", axis=-1, average="mean",
) -> (freqs, Pxy)
```

### 1.4 参数、返回值语义

| 参数 | 类型 | 默认值 | 语义 | dtype/shape |
|-----|------|-------|------|------------|
| `x` | array_like | - | 第一个时域信号 | 任意数值型, 1D或多维 |
| `y` | array_like | - | 第二个时域信号 | 与x相同 |
| `fs` | float | 1.0 | 采样频率(Hz) | 标量 |
| `window` | str/tuple/array | "hann" | 窗函数名称或数组 | 若为数组则长度=nperseg |
| `nperseg` | int | None→256 | 每段数据点数 | 正整数 |
| `noverlap` | int | None→nperseg//2 | 段间重叠点数 | 0 ~ nperseg-1 |
| `nfft` | int | None→nperseg | FFT长度 | ≥ nperseg |
| `detrend` | str/function/False | "constant" | 去趋势方式 | - |
| `return_onesided` | bool | True | 是否返回单边谱 | - |
| `scaling` | str | "density" | 缩放模式 | "density"或"spectrum" |
| `axis` | int | -1 | 计算轴 | - |
| `average` | str | "mean" | 平均方式 | "mean"或"median" |

**返回值**:

| 返回值 | dtype | shape | 语义 |
|-------|-------|------|------|
| `freqs` | float64 | (nfft//2+1,) 或 (nfft,) | 频率数组(Hz) |
| `Pxy` | complex128 | 同freqs | 交叉谱密度(复数) |

## 二、当前算子的完整相关源码

> 以下源码摘自只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py`，保留原始行号。

### 2.1 csd 函数（第491-656行）

```python
def csd(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's
    method.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    axis : int, optional
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. Defaults to 'mean'.


    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density or cross power spectrum of x,y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method. [Equivalent to
           csd(x,x)]
    coherence: Magnitude squared coherence by Welch's method.

    Notes
    -----
    By convention, Pxy is computed with the conjugate FFT of X
    multiplied by the FFT of Y.

    If the input series differ in length, the shorter series will be
    zero-padded to match.

    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.


    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    Generate two test signals with some common features.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = cp.arange(N) / fs
    >>> b, a = cusignal.butter(2, 0.25, 'low')
    >>> x = cp.random.normal(scale=cp.sqrt(noise_power), size=time.shape)
    >>> # lfilter not currently implemented in cuSignal
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*cp.sin(2*cp.pi*freq*time)
    >>> y += cp.random.normal(scale=0.1*cp.sqrt(noise_power), size=time.shape)

    Compute and plot the magnitude of the cross spectral density.

    >>> f, Pxy = cusignal.csd(x, y, fs, nperseg=1024)
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(cp.abs(Pxy)))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('CSD [V**2/Hz]')
    >>> plt.show()
    """
    x = cp.asarray(x)
    y = cp.asarray(y)
    freqs, _, Pxy = _spectral_helper(
        x,
        y,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling,
        axis,
        mode="psd",
    )

    # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if average == "median":
                Pxy = cp.median(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])
            elif average == "mean":
                Pxy = Pxy.mean(axis=-1)
            else:
                raise ValueError(
                    'average must be "median" or "mean", got %s' % (average,)
                )
        else:
            Pxy = cp.reshape(Pxy, Pxy.shape[:-1])

    return freqs, Pxy
```

### 2.2 _spectral_helper 函数（第1554-1869行）

```python
def _spectral_helper(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
    boundary=None,
    padded=False,
):
    """
    Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        None.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to False. Padding occurs after boundary extension, if
        `boundary` is not None, and `padded` is True.
    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    """
    if mode not in ["psd", "stft"]:
        raise ValueError(
            "Unknown value for mode %s, must be one of: " "{'psd', 'stft'}" % mode
        )

    boundary_funcs = {
        "even": _even_ext,
        "odd": _odd_ext,
        "constant": _const_ext,
        "zeros": _zero_ext,
        None: None,
    }

    if boundary not in boundary_funcs:
        raise ValueError(
            "Unknown boundary option '{0}', must be one of: {1}".format(
                boundary, list(boundary_funcs.keys())
            )
        )

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != "psd":
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have cp.arrays, get outdtype
    x = cp.asarray(x)
    if not same_data:
        y = cp.asarray(y)
        outdtype = cp.result_type(x, y, cp.complex64)
    else:
        outdtype = cp.result_type(x, cp.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = cp.broadcast(cp.empty(xouter), cp.empty(youter)).shape
        except ValueError:
            raise ValueError("x and y cannot be broadcast together.")

    if same_data:
        if x.size == 0:
            return cp.empty(x.shape), cp.empty(x.shape), cp.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (cp.min([x.shape[axis], y.shape[axis]]),)
            emptyout = cp.rollaxis(cp.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = cp.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = cp.rollaxis(y, axis, len(y.shape))

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = cp.concatenate((x, cp.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = cp.concatenate((y, cp.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = cp.concatenate((x, cp.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = cp.concatenate((y, cp.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:

        def detrend_func(d):
            return d

    elif not hasattr(detrend, "__call__"):

        def detrend_func(d):
            return filtering.detrend(d, type=detrend, axis=-1)

    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = cp.rollaxis(d, -1, axis)
            d = detrend(d)
            return cp.rollaxis(d, axis, len(d.shape))

    else:
        detrend_func = detrend

    if cp.result_type(win, cp.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    if mode == "stft":
        scale = cp.sqrt(scale)

    if return_onesided:
        if cp.iscomplexobj(x):
            sides = "twosided"
            warnings.warn(
                "Input data is complex, switching to " "return_onesided=False"
            )
        else:
            sides = "onesided"
            if not same_data:
                if cp.iscomplexobj(y):
                    sides = "twosided"
                    warnings.warn(
                        "Input data is complex, switching to " "return_onesided=False"
                    )
    else:
        sides = "twosided"

    if sides == "twosided":
        freqs = cp.fft.fftfreq(nfft, 1 / fs)
    elif sides == "onesided":
        freqs = cp.fft.rfftfreq(nfft, 1 / fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft, sides)
        result = cp.conj(result) * result_y
    elif mode == "psd":
        result = cp.conj(result) * result

    result *= scale
    if sides == "onesided" and mode == "psd":
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = cp.arange(
        nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap
    ) / float(fs)
    if boundary is not None:
        time -= (nperseg / 2) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != "stft":
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = cp.rollaxis(result, -1, axis)

    return freqs, time, result
```

### 2.3 _fft_helper 函数（第1872-1918行）

```python
def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    cusignal.spectral_analysis.spectral._spectral_helper

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., cp.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        # Need to optimize this in cuSignal
        result = _as_strided(x, shape=shape, strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == "twosided":
        func = cp.fft.fft
    else:
        result = result.real
        func = cp.fft.rfft
    result = func(result, n=nfft)

    return result
```

### 2.4 _triage_segments 函数（第1921-1981行）

```python
def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """

    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn(
                "nperseg = {0:d} is greater than input length "
                " = {1:d}, using nperseg = {1:d}".format(nperseg, input_length)
            )
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = cp.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if input_length < win.shape[-1]:
            raise ValueError("window is longer than input signal")
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError(
                    "value specified for nperseg is different" " from length of window"
                )
    return win, nperseg
```

### 2.5 _median_bias 函数（第1984-2002行）

```python
def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See arXiv:gr-qc/0509116 Appendix B for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.
    """
    ii_2 = 2 * cp.arange(1.0, (n - 1) // 2 + 1)
    return 1 + cp.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)
```

## 三、docstring 逐行翻译与解释

### csd 函数 docstring（第505-624行）

| 原文 | 中文解释 |
|------|---------|
| "Estimate the cross power spectral density, Pxy, using Welch's method." | 使用 Welch 方法估计交叉功率谱密度 Pxy |
| "x : array_like — Time series of measurement values" | x: 测量值的时域序列 |
| "y : array_like — Time series of measurement values" | y: 测量值的时域序列 |
| "fs : float, optional — Sampling frequency... Defaults to 1.0." | fs: 采样频率, 默认1.0 Hz |
| "window : str or tuple or array_like... Defaults to a Hann window." | window: 窗函数, 可以是名称/参数元组/数组, 默认Hann窗 |
| "nperseg : int... is set to 256" | nperseg: 每段长度, 字符串窗时默认256 |
| "noverlap: int... ``noverlap = nperseg // 2``" | noverlap: 重叠点数, 默认段长度的一半(50%重叠) |
| "nfft : int... the FFT length is `nperseg`" | nfft: FFT长度, 默认等于段长度, 支持零填充 |
| "detrend : str or function or `False`... Defaults to 'constant'." | detrend: 去趋势方式, 默认去常数(去均值) |
| "return_onesided : bool... Defaults to `True`" | return_onesided: 对实信号返回单边谱, 默认True |
| "scaling : { 'density', 'spectrum' }... Defaults to 'density'" | scaling: 密度模式(V²/Hz)或谱模式(V²), 默认密度 |
| "axis : int... default is over the last axis" | axis: 计算轴, 默认最后一轴 |
| "average : { 'mean', 'median' }... Defaults to 'mean'." | average: 均值或中值平均, 默认均值 |
| "f : ndarray — Array of sample frequencies." | 返回f: 频率数组 |
| "Pxy : ndarray — Cross spectral density or cross power spectrum of x,y." | 返回Pxy: 交叉谱密度或交叉功率谱(复数) |
| "By convention, Pxy is computed with the conjugate FFT of X multiplied by the FFT of Y." | 约定: Pxy = X*(f)·Y(f), X取共轭 |
| "If the input series differ in length, the shorter series will be zero-padded to match." | 输入长度不同时, 短信号零填充到与长信号等长 |

**示例代码说明**:

docstring 中的示例(第598-623行):
1. 生成两个含公共 1234 Hz 分量和噪声的测试信号
2. 使用 `cusignal.csd(x, y, fs, nperseg=1024)` 计算互谱
3. 绘制互谱幅值 `|Pxy|` 的半对数图

## 四、源代码逐行解释

### 4.1 csd 函数体逐行解释（第625-656行）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 625 | `x = cp.asarray(x)` | 将输入 x 转换为 CuPy 数组, 确保在 GPU 上 |
| 626 | `y = cp.asarray(y)` | 将输入 y 转换为 CuPy 数组 |
| 627-640 | `freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode="psd")` | 调用核心辅助函数, mode="psd"表示计算功率谱/互谱而非STFT; 返回(freqs, time, result), time被丢弃(用`_`接收); 此时result的最后一维是段索引, 尚未平均 |
| 643 | `if len(Pxy.shape) >= 2 and Pxy.size > 0:` | 检查Pxy至少是2维(频率×段数)且非空, 才需要平均 |
| 644 | `if Pxy.shape[-1] > 1:` | 最后一个维度(段数)大于1, 有多段需要平均 |
| 645 | `if average == "median":` | 如果用户选择中值平均 |
| 646 | `Pxy = cp.median(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])` | 沿段维度取中值, 并除以中值偏差校正因子; 中值是均值的渐进无偏估计, 但有限段数时存在偏差, _median_bias给出校正系数 |
| 647 | `elif average == "mean":` | 如果用户选择均值平均 |
| 648 | `Pxy = Pxy.mean(axis=-1)` | 沿最后一个维度(段索引)求平均, 得到最终的交叉谱估计; 对应Welch方法的第5步: $\hat{S}_{xy}(f) = \frac{1}{K}\sum_{k} P_{xy,k}(f)$ |
| 649-652 | `else: raise ValueError(...)` | average参数非法时报错 |
| 653 | `else:` | 只有1段数据 |
| 654 | `Pxy = cp.reshape(Pxy, Pxy.shape[:-1])` | 去掉退化的最后一维, 使输出shape一致 |
| 656 | `return freqs, Pxy` | 返回频率数组和交叉谱密度(复数) |

### 4.2 _spectral_helper 函数体逐行解释（第1655-1869行）

#### 阶段 A: 模式与边界验证（1655-1673）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1655-1658 | `if mode not in ["psd", "stft"]: raise ValueError(...)` | 验证mode参数, CSD调用时mode="psd" |
| 1660-1666 | `boundary_funcs = {...}` | 定义边界扩展函数映射表, CSD调用时boundary=None |
| 1668-1673 | `if boundary not in boundary_funcs: raise ValueError(...)` | 验证boundary参数合法性 |

#### 阶段 B: 同数据检测与类型推导（1675-1709）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1676 | `same_data = y is x` | 用`is`判断y和x是否为同一对象(内存地址相同); 当welch调用csd(x,x)时same_data=True, 可跳过y的重复计算 |
| 1678-1679 | `if not same_data and mode != "psd": raise ValueError(...)` | STFT模式要求x=y; CSD模式(mode="psd")允许x≠y |
| 1681 | `axis = int(axis)` | 确保axis为整数 |
| 1684 | `x = cp.asarray(x)` | 转CuPy数组 |
| 1685-1689 | `if not same_data: y = cp.asarray(y); outdtype = ... else: outdtype = ...` | 当x≠y时, y也转换, 输出dtype取x,y和complex64的最小公共类型; 当x=y时, 只需x和complex64 |
| 1691-1700 | `if not same_data: ... broadcast check ...` | 检查x和y的非计算轴能否广播 |
| 1702-1709 | 空数组检查 | 若输入为空, 返回空结果 |

#### 阶段 C: 轴调整与长度匹配（1711-1727）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1711-1715 | `if x.ndim > 1 and axis != -1: x = cp.rollaxis(...)` | 多维数据且计算轴不是最后轴时, 将计算轴滚到最后 |
| 1718-1727 | 长度不匹配零填充 | 若x和y在计算轴上长度不同, 短者末尾零填充至等长; 这保证后续分段和FFT维度一致 |

#### 阶段 D: 窗函数与参数解析（1729-1750）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1729-1732 | `if nperseg is not None: nperseg = int(nperseg); if nperseg < 1: raise ...` | 用户指定nperseg时验证为正整数 |
| 1735 | `win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])` | 解析窗函数: 若window为字符串则生成窗数组并确定nperseg |
| 1737-1742 | nfft参数处理 | 默认nfft=nperseg; 若nfft < nperseg报错; 否则转为整数 |
| 1744-1750 | noverlap参数处理 | 默认50%重叠(nperseg//2); 若noverlap≥nperseg报错; 计算步长nstep=nperseg-noverlap |

#### 阶段 E: 边界扩展与零填充（1758-1772）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1758-1762 | `if boundary is not None: x = ext_func(x, ...)` | CSD默认boundary=None, 此分支不执行; STFT时会用到 |
| 1764-1772 | `if padded: ...` | CSD默认padded=False, 此分支不执行; STFT时会用到 |

#### 阶段 F: 去趋势与缩放因子（1774-1807）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1775-1794 | 去趋势函数构造 | 若detrend=False, 恒等函数; 若为字符串, 用filtering.detrend; 若为可调用函数, 直接使用 |
| 1796-1797 | `if cp.result_type(win, cp.complex64) != outdtype: win = win.astype(outdtype)` | 窗函数类型与输出类型不一致时, 转换窗函数dtype |
| 1799-1800 | `if scaling == "density": scale = 1.0 / (fs * (win * win).sum())` | **密度模式缩放**: $scale = \frac{1}{f_s \cdot \sum w^2[n]}$; 对应 $\frac{X^*(f)Y(f)}{f_s \cdot U}$ 中的分母 |
| 1801-1802 | `elif scaling == "spectrum": scale = 1.0 / win.sum() ** 2` | **谱模式缩放**: $scale = \frac{1}{(\sum w[n])^2}$ |
| 1806-1807 | `if mode == "stft": scale = cp.sqrt(scale)` | STFT模式取平方根; CSD模式(mode="psd")不执行 |

#### 阶段 G: 单边/双边谱判定与频率计算（1809-1829）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1809-1824 | 单边/双边判定 | 若return_onesided=True且输入为实数→单边; 若输入含复数→强制双边并警告; 若return_onesided=False→双边 |
| 1826-1829 | 频率数组计算 | 双边谱用`fftfreq`, 单边谱用`rfftfreq`; 两种函数都返回以Hz为单位的频率 |

#### 阶段 H: 核心FFT与交叉谱计算（1831-1847）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1832 | `result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)` | 对x执行分段加窗FFT; 返回result, shape=(..., nfreq, nseg), nfreq为频率点数, nseg为段数 |
| 1834-1837 | `if not same_data: result_y = _fft_helper(y, ...); result = cp.conj(result) * result_y` | **核心公式**: $P_{xy,k}(f) = X_k^*(f) \cdot Y_k(f)$; 对x的FFT取共轭后乘以y的FFT; 每个频率点和每段都独立计算 |
| 1838-1839 | `elif mode == "psd": result = cp.conj(result) * result` | 当same_data=True(welch调用), 简化为 $|X_k(f)|^2$, 节省一次FFT |
| 1841 | `result *= scale` | 应用缩放因子 |
| 1842-1847 | 单边谱修正 | 单边+PSD模式时, 除直流和奈奎斯特频率外所有频率乘以2; 这是因为单边谱只包含正频率, 需要将负频率的能量折算过来 |

#### 阶段 I: 时间数组与输出整理（1849-1869）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1849-1853 | `time = cp.arange(...)` | 计算每段中心时间; CSD不使用此返回值 |
| 1855 | `result = result.astype(outdtype)` | 确保输出为复数类型 |
| 1858-1859 | `if same_data and mode != "stft": result = result.real` | 自功率谱时虚部理论为零, 取实部避免浮点残差 |
| 1863-1864 | `if axis < 0: axis -= 1` | 负轴索引因新增段维度而偏移 |
| 1867 | `result = cp.rollaxis(result, -1, axis)` | 将段维度滚回原始轴位置 |
| 1869 | `return freqs, time, result` | 返回频率、时间、未平均的谱结果 |

### 4.3 _fft_helper 函数体逐行解释（第1893-1918行）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1894-1895 | `if nperseg == 1 and noverlap == 0: result = x[..., cp.newaxis]` | 退化情况: 每段1点无重叠, 直接增加一个维度 |
| 1896-1902 | 滑动窗口分段 | 使用`_as_strided`创建滑动窗口视图: shape=(..., nseg, nperseg), strides使得相邻段沿数据轴移动step个元素; **零拷贝**, 仅修改视图元数据 |
| 1905 | `result = detrend_func(result)` | 对每段独立去趋势; 默认去除均值(constant) |
| 1908 | `result = win * result` | 逐元素乘以窗函数, 实现加窗; 广播机制使win自动应用到每一段 |
| 1911-1916 | FFT选择与执行 | 双边谱用`cp.fft.fft`(完整FFT), 单边谱用`cp.fft.rfft`(实数FFT, 只返回正频率); `n=nfft`支持零填充; CuPy的FFT自动处理零填充 |
| 1918 | `return result` | 返回分段FFT结果, shape=(..., nfreq, nseg) |

### 4.4 _triage_segments 函数体逐行解释（第1956-1981行）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 1957 | `if isinstance(window, str) or isinstance(window, tuple):` | window为字符串(如"hann")或元组(如("kaiser", 4.0)) |
| 1959-1960 | `if nperseg is None: nperseg = 256` | 未指定nperseg时默认256点 |
| 1961-1966 | `if nperseg > input_length: ... nperseg = input_length` | 段长不能超过输入长度, 超过时警告并截断 |
| 1967 | `win = get_window(window, nperseg)` | 调用cuSignal窗函数生成器, 返回CuPy数组 |
| 1968-1980 | `else: ...` | window为数组: 验证1维, 长度不超过输入, nperseg与数组长度一致 |
| 1981 | `return win, nperseg` | 返回窗数组和确定的段长度 |

### 4.5 _median_bias 函数体逐行解释（第2001-2002行）

| 行号 | 代码 | 解释 |
|-----|------|------|
| 2001 | `ii_2 = 2 * cp.arange(1.0, (n - 1) // 2 + 1)` | 生成序列 [2, 4, 6, ..., 2*⌊(n-1)/2⌋] |
| 2002 | `return 1 + cp.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)` | 计算中值偏差校正因子; 参考 arXiv:gr-qc/0509116 Appendix B; 当n→∞时偏差趋近1(即中值→均值) |

## 五、调用链与算法总结

### 5.1 完整调用链

```
cusignal.csd(x, y, ...)
  ├── cp.asarray(x), cp.asarray(y)           # 输入转CuPy数组
  ├── _spectral_helper(x, y, mode="psd")     # 核心计算
  │   ├── same_data = y is x                 # 同数据检测
  │   ├── cp.asarray(x/y)                    # 类型转换
  │   ├── broadcast检查                       # 形状兼容性
  │   ├── 零填充(长度不匹配时)                 # 边界处理
  │   ├── _triage_segments(window, nperseg)  # 窗函数解析
  │   │   └── get_window(window, nperseg)    # 生成窗数组
  │   ├── 去趋势函数构造                       # detrend处理
  │   ├── 缩放因子计算(scale)                  # density/spectrum模式
  │   ├── 单边/双边判定(sides)                 # 频谱类型
  │   ├── cp.fft.rfftfreq / fftfreq          # 频率数组
  │   ├── _fft_helper(x, win, ...)           # x的FFT
  │   │   ├── _as_strided(x, ...)            # 滑动窗口分段(零拷贝)
  │   │   ├── detrend_func(result)           # 去趋势
  │   │   ├── win * result                   # 加窗
  │   │   └── cp.fft.rfft/fft(result, n=nfft) # FFT
  │   ├── _fft_helper(y, win, ...)           # y的FFT(若x≠y)
  │   ├── cp.conj(result) * result_y         # 交叉谱: X*(f)·Y(f)
  │   ├── result *= scale                    # 缩放
  │   └── 单边谱×2修正                        # 能量折算
  ├── Pxy.mean(axis=-1)                      # 段间平均
  │   或 cp.median(Pxy, axis=-1) / bias      # 中值平均+偏差校正
  └── return freqs, Pxy                      # 返回结果
```

### 5.2 算法流程总结

1. **输入准备**: 转CuPy数组 → 形状检查 → 零填充对齐
2. **参数解析**: 窗函数生成 → 段长度/重叠/FFT长度确定
3. **分段加窗FFT**: 滑动窗口视图 → 去趋势 → 加窗 → FFT
4. **交叉谱计算**: $X_k^*(f) \cdot Y_k(f)$ → 缩放 → 单边修正
5. **段间平均**: 均值或中值(含偏差校正)
6. **返回**: 频率数组 + 交叉谱密度(复数)

## 六、数学映射

### 6.1 核心公式与代码对应

| 数学公式 | 代码位置 | 代码 |
|---------|---------|------|
| $x_k[n] = x[n+kD] \cdot w[n]$ | `_fft_helper`: 1902, 1908 | `_as_strided(...)`; `win * result` |
| $X_k(f) = \text{FFT}(x_k[n], N_{FFT})$ | `_fft_helper`: 1916 | `func(result, n=nfft)` |
| $P_{xy,k}(f) = X_k^*(f) Y_k(f)$ | `_spectral_helper`: 1837 | `cp.conj(result) * result_y` |
| $scale = \frac{1}{f_s \sum w^2[n]}$ (density) | `_spectral_helper`: 1800 | `1.0 / (fs * (win * win).sum())` |
| $scale = \frac{1}{(\sum w[n])^2}$ (spectrum) | `_spectral_helper`: 1802 | `1.0 / win.sum() ** 2` |
| $\hat{S}_{xy}(f) = \frac{1}{K}\sum_k P_{xy,k}(f)$ | `csd`: 648 | `Pxy.mean(axis=-1)` |
| $\hat{S}_{xy}^{med}(f) = \frac{\text{median}_k P_{xy,k}(f)}{b(K)}$ | `csd`: 646 | `cp.median(Pxy, axis=-1) / _median_bias(...)` |
| 单边修正: $S[1:-1] \times 2$ | `_spectral_helper`: 1847 | `result[..., 1:-1] *= 2` |

### 6.2 去趋势映射

| detrend参数 | 数学操作 | 代码实现 |
|------------|---------|---------|
| "constant" | $x'[n] = x[n] - \bar{x}$ | `filtering.detrend(d, type="constant", axis=-1)` |
| "linear" | $x'[n] = x[n] - (a \cdot n + b)$ | `filtering.detrend(d, type="linear", axis=-1)` |
| False | $x'[n] = x[n]$ (不变) | 恒等函数 `def detrend_func(d): return d` |

## 七、边界处理与数值稳定性

### 7.1 边界处理

| 情况 | 处理方式 | 代码位置 |
|-----|---------|---------|
| x, y长度不同 | 短者末尾零填充 | 第1719-1727行 |
| nperseg > 输入长度 | 警告并截断为输入长度 | `_triage_segments` 第1961-1966行 |
| noverlap ≥ nperseg | 抛出ValueError | 第1748-1749行 |
| nfft < nperseg | 抛出ValueError | 第1739-1740行 |
| 输入为空数组 | 返回空结果 | 第1702-1709行 |
| 输入含复数 + return_onesided=True | 强制双边谱 + 警告 | 第1810-1822行 |

### 7.2 dtype 处理

- 输入为实数 → 输出为 `complex128` (因为CSD是复数)
- 输入为 `float32` → 中间计算提升到 `complex64`
- `outdtype = cp.result_type(x, y, cp.complex64)` 确保至少complex64

### 7.3 数值稳定性

- 自功率谱(welch)时取 `.real` 避免浮点残差虚部(第1858-1859行)
- 中值平均时除以偏差校正因子, 确保渐进无偏(第646行)
- 零填充对齐不同长度信号, 避免维度不匹配

## 八、时间复杂度与空间复杂度

### 8.1 时间复杂度

设信号长度为 $N$, 段长为 $L$, FFT长为 $N_F$, 重叠为 $O$, 段数 $K = \lfloor(N-O)/(L-O)\rfloor$。

| 步骤 | 复杂度 | 说明 |
|-----|-------|------|
| 滑动窗口视图 | $O(1)$ | `_as_strided` 零拷贝, 只修改元数据 |
| 加窗 | $O(KL)$ | 每段L个乘法 |
| FFT | $O(K \cdot N_F \log N_F)$ | K段FFT, 每段$N_F \log N_F$ |
| 交叉谱 | $O(K N_F)$ | 共轭乘法 |
| 平均 | $O(K N_F)$ | 求和/中值 |

**总复杂度**: $O(K \cdot N_F \log N_F)$, 主导项为FFT

### 8.2 空间复杂度

| 数据 | 大小 | 说明 |
|-----|------|------|
| 滑动窗口视图 | $O(1)$ 额外 | 零拷贝视图 |
| FFT结果 | $O(K \cdot N_F)$ | K段复数FFT |
| 交叉谱 | $O(K \cdot N_F)$ | K段复数结果 |
| 最终结果 | $O(N_F)$ | 平均后 |

**总空间**: $O(K \cdot N_F)$, 主要为FFT中间结果

### 8.3 性能瓶颈

1. **FFT计算**: 占总时间绝大部分; CuPy在GPU上并行执行
2. **`_as_strided`**: 源码注释"Need to optimize this in cuSignal"(第1901行), 表明滑动窗口视图可能不是GPU最优方式
3. **中值平均**: `cp.median` 需要排序, 比均值慢

## 九、建议阅读源码顺序

1. **入口**: 从 `csd` 函数(第491行)开始, 理解公开接口和参数
2. **核心调用**: 跟进 `_spectral_helper`(第1554行), 理解参数解析和流程编排
3. **FFT实现**: 阅读 `_fft_helper`(第1872行), 理解分段加窗FFT的细节
4. **窗解析**: 阅读 `_triage_segments`(第1921行), 理解窗函数参数处理
5. **平均逻辑**: 回到 `csd` 函数(第642行), 理解均值/中值平均
6. **辅助**: 阅读 `_median_bias`(第1984行), 理解偏差校正

### 每段代码应观察的问题

| 代码段 | 应关注的问题 |
|-------|------------|
| `csd` 第627行 | 为何mode="psd"而非"stft"? 答: CSD是功率谱估计, 不是短时傅里叶变换 |
| `csd` 第646行 | 中值偏差校正的数学依据是什么? 答: 有限样本时中值是均值的有偏估计 |
| `_spectral_helper` 第1676行 | `same_data = y is x` 为何用`is`而非`==`? 答: `is`比较对象身份, `==`比较值; welch调用csd(x,x)时x和y是同一对象 |
| `_spectral_helper` 第1837行 | 为何是`conj(result) * result_y`而非`result * conj(result_y)`? 答: 约定$S_{xy} = X^*Y$, 这样$S_{yx} = Y^*X = S_{xy}^*$, 满足对称性 |
| `_fft_helper` 第1902行 | `_as_strided`是否产生数据拷贝? 答: 否, 仅创建视图, 通过修改shape和strides实现滑动窗口 |
| `_fft_helper` 第1914行 | 单边谱时为何取`result.real`? 答: `rfft`要求实数输入, 取real确保类型正确 |

---

**文档版本**: V1.0  
**创建日期**: 2026-08-08  
**源码版本**: cuSignal 23.08.00  
**只读基准路径**: `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py`  
**学习副本路径**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py`