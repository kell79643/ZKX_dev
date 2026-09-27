# spectrogram Python 源码算法

## 1. 代码定位与接口概览

相关学习副本在忽略合法学习注释后，与当前 `ZKX/cusignal-23.08.00` 基准逐行一致。

```text
cusignal.spectrogram → spectrogram → _triage_segments
→ _spectral_helper → _fft_helper → CuPy fft/rfft
```

直接共享依赖为 `_as_strided`、`get_window`、`filtering.detrend`；本文只摘录这些必要符号，没有纳入同文件其他算子。

```python
spectrogram(x, fs=1.0, window=("tukey", 0.25), nperseg=None,
            noverlap=None, nfft=None, detrend="constant",
            return_onesided=True, scaling="density", axis=-1,
            mode="psd")
```

| 参数/返回 | 默认值 | 语义 |
| --- | --- | --- |
| `x` | 必填 | 沿 `axis` 分帧 |
| `fs` | 1.0 | 采样频率 |
| `window` | Tukey(0.25) | 窗规格或一维数组 |
| `nperseg` | None | 字符串窗默认 256；数组窗取长度 |
| `noverlap` | None | 最终为 `nperseg//8` |
| `nfft` | None | 最终等于帧长，可增大零填充 |
| `detrend` | constant | 默认逐帧减均值 |
| `return_onesided` | True | 实输入单边，复输入双边 |
| `scaling` | density | 输入平方/Hz 或输入平方 |
| `mode` | psd | 五种输出模式 |
| `f,t,Sxx` | — | 频率、帧中心时间、模式相关矩阵 |

中间结果至少为 `complex64`；内部帧视图为 `(...,nframes,nperseg)`，返回时分析轴替换为频率轴并追加时间轴。

## 2. 当前算子的完整相关源码

以下是当前只读基准原文，保留签名、完整 docstring、全部示例、函数体与必要直接 helper。

### 2.1 依赖导入

基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:14`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:14`。

```python
import warnings

import cupy as cp

from ..filtering import filtering
from ..utils.arraytools import _as_strided, _const_ext, _even_ext, _odd_ext, _zero_ext
from ..windows.windows import get_window
```

### 2.2 顶层公开导出

基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:60`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:73`。

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
```

### 2.3 模块公开导出

基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14`。

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
```

### 2.4 spectrogram

基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:659`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:659`。

```python
def spectrogram(
    x,
    fs=1.0,
    window=("tukey", 0.25),
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
):
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Spectrograms can be used as a way of visualizing the change of a
    nonstationary signal's frequency content over time.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
        Defaults to a Tukey window with shape parameter of 0.25.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 8``. Defaults to `None`.
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
        Selects between computing the power spectral density ('density')
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.
    axis : int, optional
        Axis along which the spectrogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds
        to the segment times.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. In contrast to welch's method, where the
    entire data stream is averaged over, one may wish to use a smaller
    overlap (or perhaps none at all) when computing a spectrogram, to
    maintain some statistical independence between individual segments.
    It is for this reason that the default window is a Tukey window with
    1/8th of a window's length overlap at each end.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * cp.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = cp.arange(N) / float(fs)
    >>> mod = 500*cp.cos(2*cp.pi*0.25*time)
    >>> carrier = amp * cp.sin(2*cp.pi*3e3*time + mod)
    >>> noise = cp.random.normal(scale=cp.sqrt(noise_power), size=time.shape)
    >>> noise *= cp.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the spectrogram.

    >>> f, t, Sxx = cusignal.spectrogram(x, fs)
    >>> plt.pcolormesh(cp.asnumpy(t), cp.asnumpy(f), cp.asnumpy(Sxx))
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()

    Note, if using output that is not one sided, then use the following:

    >>> f, t, Sxx = cusignal.spectrogram(x, fs, return_onesided=False)
    >>> plt.pcolormesh(cp.asnumpy(t), cp.fft.fftshift(f), \
        cp.fft.fftshift(Sxx, axes=0))
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    """
    modelist = ["psd", "complex", "magnitude", "angle", "phase"]
    if mode not in modelist:
        raise ValueError(
            "unknown value for mode {}, must be one of {}".format(mode, modelist)
        )

    # need to set default for nperseg before setting default for noverlap below
    window, nperseg = _triage_segments(window, nperseg, input_length=x.shape[axis])

    # Less overlap than welch, so samples are more statisically independent
    if noverlap is None:
        noverlap = nperseg // 8

    if mode == "psd":
        freqs, time, Sxx = _spectral_helper(
            x,
            x,
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

    else:
        freqs, time, Sxx = _spectral_helper(
            x,
            x,
            fs,
            window,
            nperseg,
            noverlap,
            nfft,
            detrend,
            return_onesided,
            scaling,
            axis,
            mode="stft",
        )

        if mode == "magnitude":
            Sxx = cp.abs(Sxx)
        elif mode in ["angle", "phase"]:
            Sxx = cp.angle(Sxx)
            if mode == "phase":
                # Sxx has one additional dimension for time strides
                if axis < 0:
                    axis -= 1
                Sxx = cp.unwrap(Sxx, axis=axis)

        # mode =='complex' is same as `stft`, doesn't need modification

    return freqs, time, Sxx
```

### 2.5 _spectral_helper

基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1554`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1584`。

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
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.
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

### 2.6 _fft_helper

基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1872`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1948`。

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

### 2.7 _triage_segments

基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1921`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:2015`。

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
        6
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

### 2.8 _as_strided

基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/arraytools.py:415`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/utils/arraytools.py:415`。

```python
def _as_strided(x, shape=None, strides=None):
    """
    Create a view into the array with the given shape and strides.
    .. warning:: This function has to be used with extreme care, see notes.
    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    Returns
    -------
    view : ndarray

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    """
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)

    return cp.ndarray(shape=shape, dtype=x.dtype, memptr=x.data, strides=strides)
```

### 2.9 get_window

基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2010`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2123`。

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

### 2.10 detrend

基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1003`；学习副本起始 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1179`。

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


_freq_shift_kernel = cp.ElementwiseKernel(
    "T x, float64 freq, float64 fs",
    "complex128 out",
    """
    thrust::complex<double> temp(0, neg2pi * freq / fs * i);
    out = x * exp(temp);
    """,
    "_freq_shift_kernel",
    options=("-std=c++11",),
    loop_prep="const double neg2pi { -1 * 2 * M_PI };",
)


def freq_shift(x, freq, fs):
    """
    Frequency shift signal by freq at fs sample rate

    Parameters
    ----------
    x : array_like, complex valued
        The data to be shifted.
    freq : float
        Shift by this many (Hz)
    fs : float
        Sampling rate of the signal
    domain : string
        freq or time
    """
    x = cp.asarray(x)
    return _freq_shift_kernel(x, freq, fs)


```

## 3. docstring 逐段翻译与解释

- `spectrogram`：用连续局部 Fourier 变换显示非平稳信号频率内容随时间的变化。
- `x` 是时间序列；`fs` 把样本索引换算为 Hz 和秒。
- 字符串/元组窗交给 `get_window` 生成 DFT-even 窗；数组窗直接使用且长度必须匹配 `nperseg`。
- `noverlap` 是帧间共享样本数；本 API 默认八分之一帧，以保留更多帧间统计独立性。
- `nfft` 大于帧长时零填充，只加密频率网格。
- `detrend` 可为字符串、函数或 `False`；默认 constant 逐帧减均值。
- `density` 单位是输入平方/Hz；`spectrum` 单位是输入平方。
- 五种 mode 分别为 PSD、复 STFT、幅度、包裹相位、解缠相位。
- 示例生成缓慢调频正弦和指数衰减噪声，并演示单边与双边绘图。
- `_spectral_helper` 明确每个窗不平均而逐帧返回，因此不是 Welch 平均。
- `_fft_helper` 负责滑窗视图、去趋势、窗乘和 FFT；`_triage_segments` 负责窗和帧长。
- `get_window` 是窗调度器；`detrend` 支持 constant/linear，当前默认只走 constant。

## 4. 源代码逐行解释

每个非空源码行在以下最小完整语义块中出现一次。定位同时给出学习副本和当前只读基准行号。
+
### 4.1 当前算子直接依赖

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:14` 至 `:20`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:14` 至 `:20`：

```python
import warnings

import cupy as cp

from ..filtering import filtering
from ..utils.arraytools import _as_strided, _const_ext, _even_ext, _odd_ext, _zero_ext
from ..windows.windows import get_window
```

逐行对应：

- 第 1 行：导入依赖或公开 API。
- 第 2 行：导入依赖或公开 API。
- 第 3 行：导入依赖或公开 API。
- 第 4 行：导入依赖或公开 API。
- 第 5 行：导入依赖或公开 API。

### 4.2 顶层公开入口

学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:73` 至 `:83`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:60` 至 `:69`：

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
```

逐行对应：

- 第 1 行：导入依赖或公开 API。
- 第 2 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.3 模块公开入口

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14` 至 `:24`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14` 至 `:23`：

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
```

逐行对应：

- 第 1 行：导入依赖或公开 API。
- 第 2 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.4 函数签名与默认参数

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:659` 至 `:671`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:659` 至 `:671`：

```python
def spectrogram(
    x,
    fs=1.0,
    window=("tukey", 0.25),
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
):
```

逐行对应：

- 第 1 行：`def` 声明函数及形参，冒号开启函数体。
- 第 2 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 3 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 4 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 5 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 6 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 7 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 8 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 9 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 10 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 11 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 12 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 13 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.5 docstring 概要

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:672` 至 `:677`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:672` 至 `:677`：

```python
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Spectrograms can be used as a way of visualizing the change of a
    nonstationary signal's frequency content over time.

```

逐行对应：

- 第 1 行：三引号开始或结束 docstring。
- 第 2 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.6 输入、窗与帧参数

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:678` 至 `:700`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:678` 至 `:700`：

```python
    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
        Defaults to a Tukey window with shape parameter of 0.25.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 8``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
```

逐行对应：

- 第 1 行：NumPy docstring 小节标题。
- 第 2 行：docstring 标题分隔线。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 11 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 12 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 13 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 14 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 15 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 16 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 17 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 18 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 19 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 20 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 21 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 22 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 23 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.7 去趋势、尺度、轴和 mode

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:701` 至 `:727`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:701` 至 `:727`：

```python
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
        Selects between computing the power spectral density ('density')
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.
    axis : int, optional
        Axis along which the spectrogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.

```

逐行对应：

- 第 1 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 2 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 11 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 12 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 13 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 14 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 15 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 16 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 17 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 18 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 19 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 20 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 21 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 22 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 23 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 24 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 25 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 26 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.8 返回、Notes 与参考资料

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:728` 至 `:760`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:728` 至 `:760`：

```python
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds
        to the segment times.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. In contrast to welch's method, where the
    entire data stream is averaged over, one may wish to use a smaller
    overlap (or perhaps none at all) when computing a spectrogram, to
    maintain some statistical independence between individual segments.
    It is for this reason that the default window is a Tukey window with
    1/8th of a window's length overlap at each end.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
```

逐行对应：

- 第 1 行：NumPy docstring 小节标题。
- 第 2 行：docstring 标题分隔线。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：NumPy docstring 小节标题。
- 第 11 行：docstring 标题分隔线。
- 第 12 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 13 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 14 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 15 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 16 行：NumPy docstring 小节标题。
- 第 17 行：docstring 标题分隔线。
- 第 18 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 19 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 20 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 21 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 22 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 23 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 24 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 25 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 26 行：NumPy docstring 小节标题。
- 第 27 行：docstring 标题分隔线。
- 第 28 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 29 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.9 示例：构造调频含噪信号

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:762` 至 `:781`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:762` 至 `:781`：

```python
    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * cp.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = cp.arange(N) / float(fs)
    >>> mod = 500*cp.cos(2*cp.pi*0.25*time)
    >>> carrier = amp * cp.sin(2*cp.pi*3e3*time + mod)
    >>> noise = cp.random.normal(scale=cp.sqrt(noise_power), size=time.shape)
    >>> noise *= cp.exp(-time/5)
    >>> x = carrier + noise
```

逐行对应：

- 第 1 行：NumPy docstring 小节标题。
- 第 2 行：docstring 标题分隔线。
- 第 3 行：docstring 示例调用。
- 第 4 行：docstring 示例调用。
- 第 5 行：docstring 示例调用。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：docstring 示例调用。
- 第 10 行：docstring 示例调用。
- 第 11 行：docstring 示例调用。
- 第 12 行：docstring 示例调用。
- 第 13 行：docstring 示例调用。
- 第 14 行：docstring 示例调用。
- 第 15 行：docstring 示例调用。
- 第 16 行：docstring 示例调用。
- 第 17 行：docstring 示例调用。
- 第 18 行：docstring 示例调用。

### 4.10 示例：绘制单双边谱图

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:783` 至 `:798`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:783` 至 `:798`：

```python
    Compute and plot the spectrogram.

    >>> f, t, Sxx = cusignal.spectrogram(x, fs)
    >>> plt.pcolormesh(cp.asnumpy(t), cp.asnumpy(f), cp.asnumpy(Sxx))
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()

    Note, if using output that is not one sided, then use the following:

    >>> f, t, Sxx = cusignal.spectrogram(x, fs, return_onesided=False)
    >>> plt.pcolormesh(cp.asnumpy(t), cp.fft.fftshift(f), \
        cp.fft.fftshift(Sxx, axes=0))
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
```

逐行对应：

- 第 1 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 2 行：docstring 示例调用。
- 第 3 行：docstring 示例调用。
- 第 4 行：docstring 示例调用。
- 第 5 行：docstring 示例调用。
- 第 6 行：docstring 示例调用。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：docstring 示例调用。
- 第 9 行：docstring 示例调用。
- 第 10 行：选择双边 FFT。
- 第 11 行：docstring 示例调用。
- 第 12 行：docstring 示例调用。
- 第 13 行：docstring 示例调用。

### 4.11 mode 白名单

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:799` 至 `:806`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:799` 至 `:804`：

```python
    """
    modelist = ["psd", "complex", "magnitude", "angle", "phase"]
    if mode not in modelist:
        raise ValueError(
            "unknown value for mode {}, must be one of {}".format(mode, modelist)
        )
```

逐行对应：

- 第 1 行：三引号开始或结束 docstring。
- 第 2 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 3 行：`if` 判断为真时进入缩进块。
- 第 4 行：主动抛出异常。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.12 帧参数与 PSD 路径

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:808` 至 `:832`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:806` 至 `:827`：

```python
    # need to set default for nperseg before setting default for noverlap below
    window, nperseg = _triage_segments(window, nperseg, input_length=x.shape[axis])

    # Less overlap than welch, so samples are more statisically independent
    if noverlap is None:
        noverlap = nperseg // 8

    if mode == "psd":
        freqs, time, Sxx = _spectral_helper(
            x,
            x,
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
```

逐行对应：

- 第 1 行：原源码注释，不参与运行。
- 第 2 行：解析窗和帧长。
- 第 3 行：原源码注释，不参与运行。
- 第 4 行：`if` 判断为真时进入缩进块。
- 第 5 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 6 行：`if` 判断为真时进入缩进块。
- 第 7 行：调用共同谱分析核心。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 11 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 12 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 13 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 14 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 15 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 16 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 17 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 18 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 19 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 20 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.13 复 STFT 路径

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:835` 至 `:849`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:829` 至 `:843`：

```python
    else:
        freqs, time, Sxx = _spectral_helper(
            x,
            x,
            fs,
            window,
            nperseg,
            noverlap,
            nfft,
            detrend,
            return_onesided,
            scaling,
            axis,
            mode="stft",
        )
```

逐行对应：

- 第 1 行：前述分支均未命中时执行。
- 第 2 行：调用共同谱分析核心。
- 第 3 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 4 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 5 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 6 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 7 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 8 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 9 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 10 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 11 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 12 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 13 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。
- 第 14 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 15 行：与相邻行共同构成声明、文档段落、表达式或闭合结构。

### 4.14 四种后处理

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:852` 至 `:864`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:845` 至 `:855`：

```python
        if mode == "magnitude":
            Sxx = cp.abs(Sxx)
        elif mode in ["angle", "phase"]:
            Sxx = cp.angle(Sxx)
            if mode == "phase":
                # Sxx has one additional dimension for time strides
                if axis < 0:
                    axis -= 1
                Sxx = cp.unwrap(Sxx, axis=axis)

        # mode =='complex' is same as `stft`, doesn't need modification
```

逐行对应：

- 第 1 行：`if` 判断为真时进入缩进块。
- 第 2 行：求 STFT 幅度。
- 第 3 行：前一分支未命中后继续判断。
- 第 4 行：求主值相角。
- 第 5 行：`if` 判断为真时进入缩进块。
- 第 6 行：原源码注释，不参与运行。
- 第 7 行：`if` 判断为真时进入缩进块。
- 第 8 行：计算右侧并绑定左侧，或传入关键字参数。
- 第 9 行：沿频率轴解缠。
- 第 10 行：原源码注释，不参与运行。

### 4.15 返回结果

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:866` 至 `:866`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:857` 至 `:857`：

```python
    return freqs, time, Sxx
```

逐行对应：

- 第 1 行：结束函数并返回。

+
### 4.16 helper 签名

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1584` 至 `:1599`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1554` 至 `:1569`：

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
```

逐行对应：

- 第 1 行：函数定义与形参声明。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：右侧求值后赋值，或关键字实参。
- 第 5 行：右侧求值后赋值，或关键字实参。
- 第 6 行：右侧求值后赋值，或关键字实参。
- 第 7 行：右侧求值后赋值，或关键字实参。
- 第 8 行：右侧求值后赋值，或关键字实参。
- 第 9 行：右侧求值后赋值，或关键字实参。
- 第 10 行：右侧求值后赋值，或关键字实参。
- 第 11 行：右侧求值后赋值，或关键字实参。
- 第 12 行：右侧求值后赋值，或关键字实参。
- 第 13 行：右侧求值后赋值，或关键字实参。
- 第 14 行：右侧求值后赋值，或关键字实参。
- 第 15 行：右侧求值后赋值，或关键字实参。
- 第 16 行：与相邻行共同组成当前语义块。

### 4.17 docstring：核心参数

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1600` 至 `:1656`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1570` 至 `:1626`：

```python
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
```

逐行对应：

- 第 1 行：三引号 docstring 边界。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：与相邻行共同组成当前语义块。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：docstring 小节标题。
- 第 8 行：docstring 标题分隔线。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：与相邻行共同组成当前语义块。
- 第 16 行：与相邻行共同组成当前语义块。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：与相邻行共同组成当前语义块。
- 第 20 行：与相邻行共同组成当前语义块。
- 第 21 行：与相邻行共同组成当前语义块。
- 第 22 行：与相邻行共同组成当前语义块。
- 第 23 行：与相邻行共同组成当前语义块。
- 第 24 行：与相邻行共同组成当前语义块。
- 第 25 行：与相邻行共同组成当前语义块。
- 第 26 行：与相邻行共同组成当前语义块。
- 第 27 行：与相邻行共同组成当前语义块。
- 第 28 行：与相邻行共同组成当前语义块。
- 第 29 行：与相邻行共同组成当前语义块。
- 第 30 行：右侧求值后赋值，或关键字实参。
- 第 31 行：与相邻行共同组成当前语义块。
- 第 32 行：与相邻行共同组成当前语义块。
- 第 33 行：与相邻行共同组成当前语义块。
- 第 34 行：与相邻行共同组成当前语义块。
- 第 35 行：与相邻行共同组成当前语义块。
- 第 36 行：与相邻行共同组成当前语义块。
- 第 37 行：与相邻行共同组成当前语义块。
- 第 38 行：与相邻行共同组成当前语义块。
- 第 39 行：与相邻行共同组成当前语义块。
- 第 40 行：与相邻行共同组成当前语义块。
- 第 41 行：与相邻行共同组成当前语义块。
- 第 42 行：与相邻行共同组成当前语义块。
- 第 43 行：与相邻行共同组成当前语义块。
- 第 44 行：与相邻行共同组成当前语义块。
- 第 45 行：与相邻行共同组成当前语义块。
- 第 46 行：与相邻行共同组成当前语义块。
- 第 47 行：与相邻行共同组成当前语义块。
- 第 48 行：与相邻行共同组成当前语义块。
- 第 49 行：与相邻行共同组成当前语义块。
- 第 50 行：与相邻行共同组成当前语义块。
- 第 51 行：与相邻行共同组成当前语义块。
- 第 52 行：右侧求值后赋值，或关键字实参。
- 第 53 行：与相邻行共同组成当前语义块。
- 第 54 行：与相邻行共同组成当前语义块。
- 第 55 行：与相邻行共同组成当前语义块。

### 4.18 docstring：边界与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1657` 至 `:1683`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1627` 至 `:1653`：

```python
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.
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

```

逐行对应：

- 第 1 行：与相邻行共同组成当前语义块。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：与相邻行共同组成当前语义块。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：docstring 小节标题。
- 第 16 行：docstring 标题分隔线。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：与相邻行共同组成当前语义块。
- 第 20 行：与相邻行共同组成当前语义块。
- 第 21 行：与相邻行共同组成当前语义块。
- 第 22 行：与相邻行共同组成当前语义块。
- 第 23 行：docstring 小节标题。
- 第 24 行：docstring 标题分隔线。
- 第 25 行：与相邻行共同组成当前语义块。

### 4.19 mode/boundary 校验

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1686` 至 `:1705`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1655` 至 `:1673`：

```python
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
```

逐行对应：

- 第 1 行：条件分支入口。
- 第 2 行：拒绝非法输入。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：右侧求值后赋值，或关键字实参。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：条件分支入口。
- 第 13 行：拒绝非法输入。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：与相邻行共同组成当前语义块。
- 第 16 行：与相邻行共同组成当前语义块。
- 第 17 行：与相邻行共同组成当前语义块。

### 4.20 同源、dtype、广播与空输入

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1707` 至 `:1746`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1675` 至 `:1709`：

```python
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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：右侧求值后赋值，或关键字实参。
- 第 3 行：条件分支入口。
- 第 4 行：拒绝非法输入。
- 第 5 行：右侧求值后赋值，或关键字实参。
- 第 6 行：原源码注释。
- 第 7 行：右侧求值后赋值，或关键字实参。
- 第 8 行：条件分支入口。
- 第 9 行：右侧求值后赋值，或关键字实参。
- 第 10 行：右侧求值后赋值，或关键字实参。
- 第 11 行：备用分支。
- 第 12 行：右侧求值后赋值，或关键字实参。
- 第 13 行：条件分支入口。
- 第 14 行：原源码注释。
- 第 15 行：右侧求值后赋值，或关键字实参。
- 第 16 行：右侧求值后赋值，或关键字实参。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：异常保护或捕获。
- 第 20 行：右侧求值后赋值，或关键字实参。
- 第 21 行：异常保护或捕获。
- 第 22 行：拒绝非法输入。
- 第 23 行：条件分支入口。
- 第 24 行：条件分支入口。
- 第 25 行：返回结果。
- 第 26 行：备用分支。
- 第 27 行：条件分支入口。
- 第 28 行：右侧求值后赋值，或关键字实参。
- 第 29 行：移动数组轴。
- 第 30 行：返回结果。

### 4.21 换轴与异长补零

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1749` 至 `:1766`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1711` 至 `:1727`：

```python
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
```

逐行对应：

- 第 1 行：条件分支入口。
- 第 2 行：条件分支入口。
- 第 3 行：移动数组轴。
- 第 4 行：条件分支入口。
- 第 5 行：移动数组轴。
- 第 6 行：原源码注释。
- 第 7 行：条件分支入口。
- 第 8 行：条件分支入口。
- 第 9 行：条件分支入口。
- 第 10 行：右侧求值后赋值，或关键字实参。
- 第 11 行：右侧求值后赋值，或关键字实参。
- 第 12 行：右侧求值后赋值，或关键字实参。
- 第 13 行：备用分支。
- 第 14 行：右侧求值后赋值，或关键字实参。
- 第 15 行：右侧求值后赋值，或关键字实参。
- 第 16 行：右侧求值后赋值，或关键字实参。

### 4.22 帧长、FFT、overlap 与 hop

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1768` 至 `:1797`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1729` 至 `:1750`：

```python
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
```

逐行对应：

- 第 1 行：条件分支入口。
- 第 2 行：右侧求值后赋值，或关键字实参。
- 第 3 行：条件分支入口。
- 第 4 行：拒绝非法输入。
- 第 5 行：原源码注释。
- 第 6 行：右侧求值后赋值，或关键字实参。
- 第 7 行：条件分支入口。
- 第 8 行：右侧求值后赋值，或关键字实参。
- 第 9 行：后续条件分支。
- 第 10 行：拒绝非法输入。
- 第 11 行：备用分支。
- 第 12 行：右侧求值后赋值，或关键字实参。
- 第 13 行：条件分支入口。
- 第 14 行：右侧求值后赋值，或关键字实参。
- 第 15 行：备用分支。
- 第 16 行：右侧求值后赋值，或关键字实参。
- 第 17 行：条件分支入口。
- 第 18 行：拒绝非法输入。
- 第 19 行：右侧求值后赋值，或关键字实参。

### 4.23 边界延拓和尾补齐

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1799` 至 `:1821`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1752` 至 `:1772`：

```python
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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：原源码注释。
- 第 3 行：原源码注释。
- 第 4 行：原源码注释。
- 第 5 行：原源码注释。
- 第 6 行：条件分支入口。
- 第 7 行：右侧求值后赋值，或关键字实参。
- 第 8 行：右侧求值后赋值，或关键字实参。
- 第 9 行：条件分支入口。
- 第 10 行：右侧求值后赋值，或关键字实参。
- 第 11 行：条件分支入口。
- 第 12 行：原源码注释。
- 第 13 行：原源码注释。
- 第 14 行：右侧求值后赋值，或关键字实参。
- 第 15 行：右侧求值后赋值，或关键字实参。
- 第 16 行：右侧求值后赋值，或关键字实参。
- 第 17 行：条件分支入口。
- 第 18 行：右侧求值后赋值，或关键字实参。
- 第 19 行：右侧求值后赋值，或关键字实参。

### 4.24 detrend 包装与窗 dtype

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1823` 至 `:1849`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1774` 至 `:1797`：

```python
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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：条件分支入口。
- 第 3 行：函数定义与形参声明。
- 第 4 行：返回结果。
- 第 5 行：后续条件分支。
- 第 6 行：函数定义与形参声明。
- 第 7 行：返回结果。
- 第 8 行：后续条件分支。
- 第 9 行：原源码注释。
- 第 10 行：原源码注释。
- 第 11 行：函数定义与形参声明。
- 第 12 行：移动数组轴。
- 第 13 行：右侧求值后赋值，或关键字实参。
- 第 14 行：返回结果。
- 第 15 行：备用分支。
- 第 16 行：右侧求值后赋值，或关键字实参。
- 第 17 行：条件分支入口。
- 第 18 行：右侧求值后赋值，或关键字实参。

### 4.25 缩放因子

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1852` 至 `:1864`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1799` 至 `:1807`：

```python
    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    if mode == "stft":
        scale = cp.sqrt(scale)
```

逐行对应：

- 第 1 行：条件分支入口。
- 第 2 行：右侧求值后赋值，或关键字实参。
- 第 3 行：后续条件分支。
- 第 4 行：右侧求值后赋值，或关键字实参。
- 第 5 行：备用分支。
- 第 6 行：拒绝非法输入。
- 第 7 行：条件分支入口。
- 第 8 行：使用幅度尺度。

### 4.26 单双边与频率坐标

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1867` 至 `:1891`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1809` 至 `:1829`：

```python
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
```

逐行对应：

- 第 1 行：条件分支入口。
- 第 2 行：条件分支入口。
- 第 3 行：右侧求值后赋值，或关键字实参。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：右侧求值后赋值，或关键字实参。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：备用分支。
- 第 8 行：右侧求值后赋值，或关键字实参。
- 第 9 行：条件分支入口。
- 第 10 行：条件分支入口。
- 第 11 行：右侧求值后赋值，或关键字实参。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：右侧求值后赋值，或关键字实参。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：备用分支。
- 第 16 行：右侧求值后赋值，或关键字实参。
- 第 17 行：条件分支入口。
- 第 18 行：双边频率坐标。
- 第 19 行：后续条件分支。
- 第 20 行：单边频率坐标。

### 4.27 FFT、自谱和单边功率

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1893` 至 `:1915`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1831` 至 `:1847`：

```python
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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：执行窗化 FFT。
- 第 3 行：条件分支入口。
- 第 4 行：原源码注释。
- 第 5 行：执行窗化 FFT。
- 第 6 行：计算模平方。
- 第 7 行：后续条件分支。
- 第 8 行：计算模平方。
- 第 9 行：右侧求值后赋值，或关键字实参。
- 第 10 行：条件分支入口。
- 第 11 行：条件分支入口。
- 第 12 行：单边功率翻倍。
- 第 13 行：备用分支。
- 第 14 行：原源码注释。
- 第 15 行：单边功率翻倍。

### 4.28 时间、dtype、轴与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1919` 至 `:1944`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1849` 至 `:1869`：

```python
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

逐行对应：

- 第 1 行：右侧求值后赋值，或关键字实参。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：条件分支入口。
- 第 5 行：右侧求值后赋值，或关键字实参。
- 第 6 行：右侧求值后赋值，或关键字实参。
- 第 7 行：原源码注释。
- 第 8 行：条件分支入口。
- 第 9 行：右侧求值后赋值，或关键字实参。
- 第 10 行：原源码注释。
- 第 11 行：原源码注释。
- 第 12 行：条件分支入口。
- 第 13 行：右侧求值后赋值，或关键字实参。
- 第 14 行：原源码注释。
- 第 15 行：移动数组轴。
- 第 16 行：返回结果。

### 4.29 FFT helper 文档

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1948` 至 `:1968`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1872` 至 `:1892`：

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
```

逐行对应：

- 第 1 行：函数定义与形参声明。
- 第 2 行：三引号 docstring 边界。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：与相邻行共同组成当前语义块。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：docstring 小节标题。
- 第 11 行：docstring 标题分隔线。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：docstring 小节标题。
- 第 15 行：docstring 标题分隔线。
- 第 16 行：与相邻行共同组成当前语义块。
- 第 17 行：三引号 docstring 边界。

### 4.30 stride 分帧

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1969` 至 `:1986`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1893` 至 `:1902`：

```python
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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：条件分支入口。
- 第 3 行：右侧求值后赋值，或关键字实参。
- 第 4 行：备用分支。
- 第 5 行：原源码注释。
- 第 6 行：右侧求值后赋值，或关键字实参。
- 第 7 行：帧视图形状。
- 第 8 行：帧视图步长。
- 第 9 行：原源码注释。
- 第 10 行：创建重叠视图。

### 4.31 去趋势与乘窗

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1988` 至 `:1996`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1904` 至 `:1908`：

```python
    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：逐帧去趋势。
- 第 3 行：原源码注释。
- 第 4 行：广播乘窗。

### 4.32 FFT/rFFT 与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1998` 至 `:2011`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1910` 至 `:1918`：

```python
    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == "twosided":
        func = cp.fft.fft
    else:
        result = result.real
        func = cp.fft.rfft
    result = func(result, n=nfft)

    return result
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：条件分支入口。
- 第 3 行：双边 FFT。
- 第 4 行：备用分支。
- 第 5 行：右侧求值后赋值，或关键字实参。
- 第 6 行：实输入单边 FFT。
- 第 7 行：右侧求值后赋值，或关键字实参。
- 第 8 行：返回结果。

+
### 4.33 窗解析 docstring

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:2015` 至 `:2048`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1921` 至 `:1954`：

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
        6
        window.
    """
```

逐行对应：

- 第 1 行：定义函数或辅助类。
- 第 2 行：三引号 docstring 边界。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：docstring 小节标题。
- 第 6 行：docstring 标题分隔线。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：与相邻行共同组成当前语义块。
- 第 16 行：与相邻行共同组成当前语义块。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：docstring 小节标题。
- 第 20 行：docstring 标题分隔线。
- 第 21 行：与相邻行共同组成当前语义块。
- 第 22 行：与相邻行共同组成当前语义块。
- 第 23 行：与相邻行共同组成当前语义块。
- 第 24 行：与相邻行共同组成当前语义块。
- 第 25 行：与相邻行共同组成当前语义块。
- 第 26 行：与相邻行共同组成当前语义块。
- 第 27 行：与相邻行共同组成当前语义块。
- 第 28 行：与相邻行共同组成当前语义块。
- 第 29 行：三引号 docstring 边界。

### 4.34 字符串/元组窗

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:2050` 至 `:2064`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1956` 至 `:1967`：

```python
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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：条件为真时执行缩进块。
- 第 3 行：原源码注释。
- 第 4 行：条件为真时执行缩进块。
- 第 5 行：求值并赋值，或设置关键字参数。
- 第 6 行：条件为真时执行缩进块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：求值并赋值，或设置关键字参数。
- 第 9 行：求值并赋值，或设置关键字参数。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：求值并赋值，或设置关键字参数。
- 第 12 行：调用窗调度器。

### 4.35 数组窗与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:2065` 至 `:2082`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1968` 至 `:1981`：

```python
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

逐行对应：

- 第 1 行：备用分支。
- 第 2 行：转换为 CuPy 数组。
- 第 3 行：条件为真时执行缩进块。
- 第 4 行：抛出参数异常。
- 第 5 行：条件为真时执行缩进块。
- 第 6 行：抛出参数异常。
- 第 7 行：条件为真时执行缩进块。
- 第 8 行：求值并赋值，或设置关键字参数。
- 第 9 行：继续判断另一条件。
- 第 10 行：条件为真时执行缩进块。
- 第 11 行：抛出参数异常。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：返回计算结果。

### 4.36 数组接口与 stride 视图

学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/arraytools.py:415` 至 `:432`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/arraytools.py:415` 至 `:432`：

```python
def _as_strided(x, shape=None, strides=None):
    """
    Create a view into the array with the given shape and strides.
    .. warning:: This function has to be used with extreme care, see notes.
    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    Returns
    -------
    view : ndarray

    Notes
    -----
```

逐行对应：

- 第 1 行：定义函数或辅助类。
- 第 2 行：三引号 docstring 边界。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：docstring 小节标题。
- 第 6 行：docstring 标题分隔线。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：docstring 小节标题。
- 第 14 行：docstring 标题分隔线。
- 第 15 行：与相邻行共同组成当前语义块。
- 第 16 行：docstring 小节标题。
- 第 17 行：docstring 标题分隔线。

### 4.37 所有权、dtype 与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/arraytools.py:433` 至 `:441`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/arraytools.py:433` 至 `:441`：

```python
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    """
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)

    return cp.ndarray(shape=shape, dtype=x.dtype, memptr=x.data, strides=strides)
```

逐行对应：

- 第 1 行：与相邻行共同组成当前语义块。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：三引号 docstring 边界。
- 第 6 行：求值并赋值，或设置关键字参数。
- 第 7 行：求值并赋值，或设置关键字参数。
- 第 8 行：返回计算结果。

### 4.38 get_window 参数与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2123` 至 `:2168`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2010` 至 `:2055`：

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
```

逐行对应：

- 第 1 行：定义函数或辅助类。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：docstring 小节标题。
- 第 5 行：docstring 标题分隔线。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：docstring 小节标题。
- 第 16 行：docstring 标题分隔线。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：docstring 小节标题。
- 第 20 行：docstring 标题分隔线。
- 第 21 行：与相邻行共同组成当前语义块。
- 第 22 行：与相邻行共同组成当前语义块。
- 第 23 行：与相邻行共同组成当前语义块。
- 第 24 行：与相邻行共同组成当前语义块。
- 第 25 行：与相邻行共同组成当前语义块。
- 第 26 行：与相邻行共同组成当前语义块。
- 第 27 行：与相邻行共同组成当前语义块。
- 第 28 行：与相邻行共同组成当前语义块。
- 第 29 行：与相邻行共同组成当前语义块。
- 第 30 行：与相邻行共同组成当前语义块。
- 第 31 行：与相邻行共同组成当前语义块。
- 第 32 行：与相邻行共同组成当前语义块。
- 第 33 行：与相邻行共同组成当前语义块。
- 第 34 行：与相邻行共同组成当前语义块。
- 第 35 行：与相邻行共同组成当前语义块。
- 第 36 行：与相邻行共同组成当前语义块。
- 第 37 行：与相邻行共同组成当前语义块。
- 第 38 行：与相邻行共同组成当前语义块。
- 第 39 行：与相邻行共同组成当前语义块。
- 第 40 行：与相邻行共同组成当前语义块。
- 第 41 行：与相邻行共同组成当前语义块。
- 第 42 行：与相邻行共同组成当前语义块。

### 4.39 窗列表与异常

学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2169` 至 `:2192`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2056` 至 `:2079`：

```python
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
```

逐行对应：

- 第 1 行：与相邻行共同组成当前语义块。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：与相邻行共同组成当前语义块。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：docstring 小节标题。
- 第 12 行：docstring 标题分隔线。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：调用窗调度器。
- 第 15 行：与相邻行共同组成当前语义块。
- 第 16 行：调用窗调度器。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：调用窗调度器。

### 4.40 窗规格正规化

学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2193` 至 `:2218`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2080` 至 `:2094`：

```python
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
```

逐行对应：

- 第 1 行：与相邻行共同组成当前语义块。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：三引号 docstring 边界。
- 第 4 行：求值并赋值，或设置关键字参数。
- 第 5 行：异常保护或捕获。
- 第 6 行：求值并赋值，或设置关键字参数。
- 第 7 行：异常保护或捕获。
- 第 8 行：求值并赋值，或设置关键字参数。
- 第 9 行：条件为真时执行缩进块。
- 第 10 行：求值并赋值，或设置关键字参数。
- 第 11 行：条件为真时执行缩进块。
- 第 12 行：求值并赋值，或设置关键字参数。
- 第 13 行：继续判断另一条件。
- 第 14 行：条件为真时执行缩进块。

### 4.41 窗分派与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2219` 至 `:2248`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:2095` 至 `:2114`：

```python
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

逐行对应：

- 第 1 行：抛出参数异常。
- 第 2 行：与相邻行共同组成当前语义块。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：备用分支。
- 第 6 行：求值并赋值，或设置关键字参数。
- 第 7 行：备用分支。
- 第 8 行：抛出参数异常。
- 第 9 行：异常保护或捕获。
- 第 10 行：求值并赋值，或设置关键字参数。
- 第 11 行：异常保护或捕获。
- 第 12 行：抛出参数异常。
- 第 13 行：求值并赋值，或设置关键字参数。
- 第 14 行：备用分支。
- 第 15 行：求值并赋值，或设置关键字参数。
- 第 16 行：求值并赋值，或设置关键字参数。
- 第 17 行：返回计算结果。

### 4.42 detrend 参数与返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1179` 至 `:1227`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1003` 至 `:1051`：

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
```

逐行对应：

- 第 1 行：定义函数或辅助类。
- 第 2 行：三引号 docstring 边界。
- 第 3 行：与相邻行共同组成当前语义块。
- 第 4 行：docstring 小节标题。
- 第 5 行：docstring 标题分隔线。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：求值并赋值，或设置关键字参数。
- 第 13 行：与相邻行共同组成当前语义块。
- 第 14 行：与相邻行共同组成当前语义块。
- 第 15 行：求值并赋值，或设置关键字参数。
- 第 16 行：与相邻行共同组成当前语义块。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：与相邻行共同组成当前语义块。
- 第 20 行：与相邻行共同组成当前语义块。
- 第 21 行：与相邻行共同组成当前语义块。
- 第 22 行：docstring 小节标题。
- 第 23 行：docstring 标题分隔线。
- 第 24 行：与相邻行共同组成当前语义块。
- 第 25 行：与相邻行共同组成当前语义块。
- 第 26 行：docstring 小节标题。
- 第 27 行：docstring 标题分隔线。
- 第 28 行：与相邻行共同组成当前语义块。
- 第 29 行：与相邻行共同组成当前语义块。
- 第 30 行：求值并赋值，或设置关键字参数。
- 第 31 行：求值并赋值，或设置关键字参数。
- 第 32 行：求值并赋值，或设置关键字参数。
- 第 33 行：求值并赋值，或设置关键字参数。
- 第 34 行：与相邻行共同组成当前语义块。
- 第 35 行：与相邻行共同组成当前语义块。
- 第 36 行：三引号 docstring 边界。
- 第 37 行：条件为真时执行缩进块。
- 第 38 行：抛出参数异常。
- 第 39 行：转换为 CuPy 数组。
- 第 40 行：求值并赋值，或设置关键字参数。
- 第 41 行：条件为真时执行缩进块。
- 第 42 行：求值并赋值，或设置关键字参数。
- 第 43 行：条件为真时执行缩进块。
- 第 44 行：求值并赋值，或设置关键字参数。
- 第 45 行：返回计算结果。
- 第 46 行：备用分支。

### 4.43 Notes、参考与示例

学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1228` 至 `:1252`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1052` 至 `:1076`：

```python
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
```

逐行对应：

- 第 1 行：求值并赋值，或设置关键字参数。
- 第 2 行：求值并赋值，或设置关键字参数。
- 第 3 行：求值并赋值，或设置关键字参数。
- 第 4 行：条件为真时执行缩进块。
- 第 5 行：抛出参数异常。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：求值并赋值，或设置关键字参数。
- 第 10 行：原源码注释。
- 第 11 行：原源码注释。
- 第 12 行：求值并赋值，或设置关键字参数。
- 第 13 行：条件为真时执行缩进块。
- 第 14 行：求值并赋值，或设置关键字参数。
- 第 15 行：求值并赋值，或设置关键字参数。
- 第 16 行：重塑为便于批量拟合的二维布局。
- 第 17 行：与相邻行共同组成当前语义块。
- 第 18 行：与相邻行共同组成当前语义块。
- 第 19 行：条件为真时执行缩进块。
- 第 20 行：求值并赋值，或设置关键字参数。
- 第 21 行：条件为真时执行缩进块。
- 第 22 行：求值并赋值，或设置关键字参数。

### 4.44 类型校验和 constant 分支

学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1253` 至 `:1266`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1077` 至 `:1090`：

```python

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
```

逐行对应：

- 第 1 行：原源码注释。
- 第 2 行：循环逐项处理。
- 第 3 行：求值并赋值，或设置关键字参数。
- 第 4 行：求值并赋值，或设置关键字参数。
- 第 5 行：重塑为便于批量拟合的二维布局。
- 第 6 行：求值并赋值，或设置关键字参数。
- 第 7 行：最小二乘求线性趋势。
- 第 8 行：求值并赋值，或设置关键字参数。
- 第 9 行：原源码注释。
- 第 10 行：求值并赋值，或设置关键字参数。
- 第 11 行：重塑为便于批量拟合的二维布局。
- 第 12 行：求值并赋值，或设置关键字参数。

### 4.45 linear：换轴、shape 与断点

学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1267` 至 `:1281`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1091` 至 `:1105`：

```python
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = cp.transpose(ret, tuple(olddims))
        return ret


_freq_shift_kernel = cp.ElementwiseKernel(
    "T x, float64 freq, float64 fs",
    "complex128 out",
    """
    thrust::complex<double> temp(0, neg2pi * freq / fs * i);
    out = x * exp(temp);
    """,
    "_freq_shift_kernel",
    options=("-std=c++11",),
    loop_prep="const double neg2pi { -1 * 2 * M_PI };",
```

逐行对应：

- 第 1 行：求值并赋值，或设置关键字参数。
- 第 2 行：求值并赋值，或设置关键字参数。
- 第 3 行：返回计算结果。
- 第 4 行：求值并赋值，或设置关键字参数。
- 第 5 行：与相邻行共同组成当前语义块。
- 第 6 行：与相邻行共同组成当前语义块。
- 第 7 行：三引号 docstring 边界。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：求值并赋值，或设置关键字参数。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：求值并赋值，或设置关键字参数。
- 第 13 行：求值并赋值，或设置关键字参数。

### 4.46 分段最小二乘

学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1282` 至 `:1297`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1106` 至 `:1121`：

```python
)


def freq_shift(x, freq, fs):
    """
    Frequency shift signal by freq at fs sample rate

    Parameters
    ----------
    x : array_like, complex valued
        The data to be shifted.
    freq : float
        Shift by this many (Hz)
    fs : float
        Sampling rate of the signal
    domain : string
```

逐行对应：

- 第 1 行：与相邻行共同组成当前语义块。
- 第 2 行：定义函数或辅助类。
- 第 3 行：三引号 docstring 边界。
- 第 4 行：与相邻行共同组成当前语义块。
- 第 5 行：docstring 小节标题。
- 第 6 行：docstring 标题分隔线。
- 第 7 行：与相邻行共同组成当前语义块。
- 第 8 行：与相邻行共同组成当前语义块。
- 第 9 行：与相邻行共同组成当前语义块。
- 第 10 行：与相邻行共同组成当前语义块。
- 第 11 行：与相邻行共同组成当前语义块。
- 第 12 行：与相邻行共同组成当前语义块。
- 第 13 行：与相邻行共同组成当前语义块。

### 4.47 恢复布局并返回

学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1298` 至 `:1303`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1122` 至 `:1127`：

```python
        freq or time
    """
    x = cp.asarray(x)
    return _freq_shift_kernel(x, freq, fs)


```

逐行对应：

- 第 1 行：与相邻行共同组成当前语义块。
- 第 2 行：三引号 docstring 边界。
- 第 3 行：转换为 CuPy 数组。
- 第 4 行：返回计算结果。

## 5. 调用链与算法总结

入口校验 mode 并设置八分之一默认重叠；PSD 直接请求功率路径，其余 mode 先求复 STFT。helper 统一 axis、dtype、nfft、detrend、单双边和 scaling；FFT helper 用 stride view 分帧、去趋势、乘窗并执行批量 FFT。随后形成自谱、缩放、合并单边功率、生成帧中心时间，并在入口完成幅度、相角或相位解缠。

## 6. 数学映射、边界、复杂度和阅读检查

| 步骤 | 基准位置 |
| --- | --- |
| hop $H=L-O$ | `spectral.py:1750,1898` |
| stride 分帧 | `spectral.py:1899-1902` |
| 去趋势、窗乘 | `spectral.py:1904-1908` |
| FFT/rFFT | `spectral.py:1910-1916` |
| $|X|^2$ | `spectral.py:1838-1839` |
| density/spectrum | `spectral.py:1799-1807` |
| 单边功率 | `spectral.py:1842-1847` |
| 幅度/相角/解缠 | `spectral.py:845-853` |

非法帧长、nfft、overlap 和窗 shape 会异常；不扩边、不补尾；复输入强制双边。复杂度主项为 $O(BFN\log N)$；输出空间单边约 $O(BFN/2)$、双边 $O(BFN)$。

### 6.1 自检问题与参考答案

1. **五种 mode 如何分路？** PSD 走功率路径；其余先求复 STFT，再原样、取模、取角或解缠，见 813-855。
2. **为何 helper 的 1/2 overlap 不生效？** 入口已把 None 改成 nperseg//8，见 809-811、1744-1747。
3. **stride shape？** 帧数为 `(len-overlap)//step`，末维 nperseg，见 1898-1902。
4. **STFT 为何使用 sqrt(scale)？** scale 是功率尺度，复幅度只乘平方根，见 1799-1807。
5. **单边端点？** DC 和偶数 nfft 的 Nyquist 不翻倍，见 1842-1847。
6. **dtype？** result_type 至少 complex64；PSD 取实部，其他后处理得到实值，见 1683-1689、1855-1859。
7. **为何不是 Welch？** 每窗原样返回且没有时间 mean，见 1573-1576、1831-1869。
8. **nfft 增大？** 增大 FFT 与输出成本，只加密 DTFT 采样，不提高窗决定的真实分辨力。
9. **constant detrend？** 每帧减均值，全常量帧变零；调用见 1780-1784，定义见 `filtering.py:1077-1090`。
10. **phase 沿何轴？** 原分析轴变成的频率轴；新增时间维后负 axis 先减 1，见 849-853。
11. **复输入为何双边？** 一般复信号不具共轭对称，删负频会丢独立信息，见 1809-1824。
12. **主要瓶颈？** 批量 FFT、随 overlap 增加的帧数、过大 nfft 与中间 GPU 数组。
