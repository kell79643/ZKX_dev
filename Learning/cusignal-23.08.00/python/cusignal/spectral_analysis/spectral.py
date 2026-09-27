# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import cupy as cp

from ..filtering import filtering
from ..utils.arraytools import _as_strided, _const_ext, _even_ext, _odd_ext, _zero_ext
from ..windows.windows import get_window
from ._spectral_cuda import _lombscargle


def lombscargle(
    x,
    y,
    freqs,
    precenter=False,
    normalize=False,
):
    """
    lombscargle(x, y, freqs)
    Computes the Lomb-Scargle periodogram.
    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
    extended by Scargle [2]_ to find, and test the significance of weak
    periodic signals with uneven temporal sampling.
    When *normalize* is False (default) the computed periodogram
    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
    signal with amplitude A for sufficiently large N.
    When *normalize* is True the computed periodogram is normalized by
    the residuals of the data around a constant reference model (at zero).
    Input arrays should be one-dimensional and will be cast to float64.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values.
    freqs : array_like
        Angular frequencies for output periodogram.
    precenter : bool, optional
        Pre-center amplitudes by subtracting the mean.
    normalize : bool, optional
        Compute normalized periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    Notes
    -----
    This subroutine calculates the periodogram using a slightly
    modified algorithm due to Townsend [3]_ which allows the
    periodogram to be calculated using only a single pass through
    the input arrays for each frequency.
    The algorithm running time scales roughly as O(x * freqs) or O(N^2)
    for a large number of samples and frequencies.

    References
    ----------
    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976
    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -
           Statistical aspects of spectral analysis of unevenly spaced data",
           The Astrophysical Journal, vol 263, pp. 835-853, 1982
    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle
           periodogram using graphics processing units.", The Astrophysical
           Journal Supplement Series, vol 191, pp. 247-253, 2010

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    welch: Power spectral density by Welch's method
    spectrogram: Spectrogram by Welch's method
    csd: Cross spectral density by Welch's method

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    First define some input parameters for the signal:
    >>> A = 2.
    >>> w = 1.
    >>> phi = 0.5 * cp.pi
    >>> nin = 1000
    >>> nout = 100000
    >>> frac_points = 0.9 # Fraction of points to select
    Randomly select a fraction of an array with timesteps:
    >>> r = cp.random.rand(nin)
    >>> x = cp.linspace(0.01, 10*cp.pi, nin)
    >>> x = x[r >= frac_points]
    Plot a sine wave for the selected times:
    >>> y = A * cp.sin(w*x+phi)
    Define the array of frequencies for which to compute the periodogram:
    >>> f = cp.linspace(0.01, 10, nout)
    Calculate Lomb-Scargle periodogram:
    >>> pgram = cusignal.lombscargle(x, y, f, normalize=True)
    Now make a plot of the input data:
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(cp.asnumpy(x), cp.asnumpy(y), 'b+')
    Then plot the normalized periodogram:
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(cp.asnumpy(f), cp.asnumpy(pgram))
    >>> plt.show()
    """

    x = cp.asarray(x, dtype=cp.float64)
    y = cp.asarray(y, dtype=cp.float64)
    freqs = cp.asarray(freqs, dtype=cp.float64)
    pgram = cp.empty(freqs.shape[0], dtype=cp.float64)

    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    y_dot = cp.zeros(1, dtype=cp.float64)
    if normalize:
        cp.dot(y, y, out=y_dot)

    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y

    _lombscargle(x, y_in, freqs, pgram, y_dot)

    return pgram


def periodogram(
    x,
    fs=1.0,
    window="boxcar",
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
):
    """
    Estimate power spectral density using a periodogram.

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
        directly as the window and its length must be nperseg. Defaults
        to 'boxcar'.
    nfft : int, optional
        Length of the FFT used. If `None` the length of `x` will be
        used.
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
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of `x`.

    See Also
    --------
    welch: Estimate power spectral density using Welch's method
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> cp.random.seed(1234)

    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    0.001 V**2/Hz of white noise sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2*cp.sqrt(2)
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = cp.arange(N) / fs
    >>> x = amp*cp.sin(2*cp.pi*freq*time)
    >>> x += cp.random.normal(scale=cp.sqrt(noise_power), size=time.shape)

    Compute and plot the power spectral density.

    >>> f, Pxx_den = cusignal.periodogram(x, fs)
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(Pxx_den))
    >>> plt.ylim([1e-7, 1e2])
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('PSD [V**2/Hz]')
    >>> plt.show()

    If we average the last half of the spectral density, to exclude the
    peak, we can recover the noise power on the signal.

    >>> cp.mean(Pxx_den[25000:])
    0.00099728892368242854

    Now compute and plot the power spectrum.

    >>> f, Pxx_spec = cusignal.periodogram(x, fs, 'flattop', \
            scaling='spectrum')
    >>> plt.figure()
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(cp.sqrt(Pxx_spec)))
    >>> plt.ylim([1e-4, 1e1])
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Linear spectrum [V RMS]')
    >>> plt.show()

    The peak height in the power spectrum is an estimate of the RMS
    amplitude.

    >>> cp.sqrt(Pxx_spec.max())
    2.0077340678640727

    """
    x = cp.asarray(x)

    if x.size == 0:
        return cp.empty(x.shape), cp.empty(x.shape)

    if window is None:
        window = "boxcar"

    if nfft is None:
        nperseg = x.shape[axis]
    elif nfft == x.shape[axis]:
        nperseg = nfft
    elif nfft > x.shape[axis]:
        nperseg = x.shape[axis]
    elif nfft < x.shape[axis]:
        # cp.s_ not implemented
        s = [cp.s_[:]] * len(x.shape)
        s[axis] = cp.s_[:nfft]
        x = cp.asarray(x[tuple(s)])
        nperseg = nfft
        nfft = None

    return welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
    )


def welch(
    x,
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
    Estimate power spectral density using Welch's method.

    Welch's method [1]_ computes an estimate of the power spectral
    density by dividing the data into overlapping segments, computing a
    modified periodogram for each segment and averaging the
    periodograms.

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
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. Defaults to 'mean'.


    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of x.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    If `noverlap` is 0, this method is equivalent to Bartlett's method
    [2]_.

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> cp.random.seed(1234)

    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    0.001 V**2/Hz of white noise sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2*cp.sqrt(2)
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = cp.arange(N) / fs
    >>> x = amp*cp.sin(2*cp.pi*freq*time)
    >>> x += cp.random.normal(scale=cp.sqrt(noise_power), size=time.shape)

    Compute and plot the power spectral density.

    >>> f, Pxx_den = cusignal.welch(x, fs, nperseg=1024)
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(Pxx_den))
    >>> plt.ylim([0.5e-3, 1])
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('PSD [V**2/Hz]')
    >>> plt.show()

    If we average the last half of the spectral density, to exclude the
    peak, we can recover the noise power on the signal.

    >>> cp.mean(Pxx_den[256:])
    0.0009924865443739191

    Now compute and plot the power spectrum.

    >>> f, Pxx_spec = cusignal.welch(x, fs, 'flattop', 1024, \
        scaling='spectrum')
    >>> plt.figure()
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(cp.sqrt(Pxx_spec)))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Linear spectrum [V RMS]')
    >>> plt.show()

    The peak height in the power spectrum is an estimate of the RMS
    amplitude.

    >>> cp.sqrt(Pxx_spec.max())
    2.0077340678640727

    If we now introduce a discontinuity in the signal, by increasing the
    amplitude of a small portion of the signal by 50, we can see the
    corruption of the mean average power spectral density, but using a
    median average better estimates the normal behaviour.

    >>> x[int(N//2):int(N//2)+10] *= 50.
    >>> f, Pxx_den = cusignal.welch(x, fs, nperseg=1024)
    >>> f_med, Pxx_den_med = cusignal.welch(x, fs, nperseg=1024,
                                          average='median')
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(Pxx_den), label='mean')
    >>> plt.semilogy(cp.asnumpy(f_med), cp.asnumpy(Pxx_den_med), \
        label='median')
    >>> plt.ylim([0.5e-3, 1])
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('PSD [V**2/Hz]')
    >>> plt.legend()
    >>> plt.show()

    """

    freqs, Pxx = csd(
        x,
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        average=average,
    )

    return freqs, Pxx.real


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
    # <学习注释：五种 mode 分成两条计算路径：psd 直接求功率，其余模式先求复 STFT。>
    modelist = ["psd", "complex", "magnitude", "angle", "phase"]
    # <学习注释：先做白名单检查，避免未知字符串静默落入错误的计算分支。>
    if mode not in modelist:
        raise ValueError(
            "unknown value for mode {}, must be one of {}".format(mode, modelist)
        )

    # need to set default for nperseg before setting default for noverlap below
    # <学习注释：_triage_segments 同时把窗描述解析成数组，并最终确定每帧长度 nperseg。>
    window, nperseg = _triage_segments(window, nperseg, input_length=x.shape[axis])

    # Less overlap than welch, so samples are more statisically independent
    # <学习注释：谱图默认只重叠 1/8 帧，使相邻时间切片比 Welch 默认设置更少相关。>
    if noverlap is None:
        noverlap = nperseg // 8

    # <学习注释：psd 分支要求 helper 返回每帧的功率谱，而不是复数 STFT。>
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

    # <学习注释：complex、magnitude、angle、phase 共用复 STFT 计算，再按需后处理。>
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

        # <学习注释：magnitude 对复 STFT 取模，结果非负但未平方。>
        if mode == "magnitude":
            Sxx = cp.abs(Sxx)
        # <学习注释：angle 与 phase 都先求主值相角；只有 phase 还会继续解缠。>
        elif mode in ["angle", "phase"]:
            Sxx = cp.angle(Sxx)
            if mode == "phase":
                # Sxx has one additional dimension for time strides
                if axis < 0:
                    axis -= 1
                # <学习注释：新增时间维后修正负 axis，再沿频率轴消除相邻点的 2π 跳变。>
                Sxx = cp.unwrap(Sxx, axis=axis)

        # mode =='complex' is same as `stft`, doesn't need modification

    return freqs, time, Sxx


# <学习注释：stft 是公开入口；它负责固定 STFT 模式参数，再把实际分帧、加窗和 FFT 工作交给 _spectral_helper。>
def stft(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend=False,
    return_onesided=True,
    boundary="zeros",
    padded=True,
    axis=-1,
):
    r"""
    Compute the Short Time Fourier Transform (STFT).

    STFTs can be used as a way of quantifying the change of a
    nonstationary signal's frequency and phase content over time.

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
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`. When
        specified, the COLA constraint must be met (see Notes below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        Axis along which the STFT is computed; the default is over the
        last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of `x`. By default, the last axis of `Zxx` corresponds
        to the segment times.

    See Also
    --------
    welch: Power spectral density by Welch's method.
    spectrogram: Spectrogram by Welch's method.
    csd: Cross spectral density by Welch's method.
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "Nonzero
    OverLap Add" (NOLA), and the input signal must have complete
    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %
    (nperseg-noverlap) == 0``). The `padded` argument may be used to
    accomplish this.

    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop
    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index
    :math:`t` is given by

    .. math:: x_{t}[n]=x[n]w[n-tH]

    The overlap-add (OLA) reconstruction equation is given by

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    The NOLA constraint ensures that every normalization term that appears
    in the denomimator of the OLA reconstruction equation is nonzero. Whether a
    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can
    be tested with `check_NOLA`.

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

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
    >>> noise = cp.random.normal(scale=cp.sqrt(noise_power),
    ...                          size=time.shape)
    >>> noise *= cp.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the STFT's magnitude.

    >>> f, t, Zxx = cusignal.stft(x, fs, nperseg=1000)
    >>> plt.pcolormesh(cp.asnumpy(t), cp.asnumpy(f), cp.asnumpy(cp.abs(Zxx)), \
        vmin=0, vmax=amp)
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    """

    # <学习注释：同一个 x 同时作为 x 和 y 传入，使 helper 识别 same_data=True；STFT 模式不会进行互谱相乘。>
    freqs, time, Zxx = _spectral_helper(
        x,
        x,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        # <学习注释：spectrum 标度在 STFT 模式下最终变成窗和幅度归一化 1/abs(sum(win))。>
        scaling="spectrum",
        axis=axis,
        # <学习注释：mode="stft" 使 helper 保留复 FFT，而不是计算共轭乘积得到功率谱。>
        mode="stft",
        boundary=boundary,
        padded=padded,
    )

    return freqs, time, Zxx


_istft_kernel = cp.ElementwiseKernel(
    "raw T xsubs, raw T win, int32 N, int32 half_N",
    "T x, T norm",
    """
    int p = static_cast<int>( i / N ) * 2;
    x = xsubs[(p * N) + (i % N)] * win[i % N];
    norm = win[i % N] * win[i % N];

    if ( ( i >= half_N ) && ( i < ( _ind.size() - half_N ) ) ) {
        p = static_cast<int>( ( i-half_N ) / N ) * 2 + 1;
        x += xsubs[(p * N) + ( (i - half_N) % N )] * win[(i - half_N) % N];
        norm += win[(i - half_N) % N] * win[(i - half_N) % N];
    }
    """,
    "_istft_kernel",
    options=("-std=c++11",),
)


def istft(
    Zxx,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    input_onesided=True,
    boundary=True,
    time_axis=-1,
    freq_axis=-2,
):
    r"""
    Perform the inverse Short Time Fourier transform (iSTFT).
    Parameters
    ----------
    Zxx : array_like
        STFT of the signal to be reconstructed. If a purely real array
        is passed, it will be cast to a complex data type.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window. Must match the window used to generate the
        STFT for faithful inversion.
    nperseg : int, optional
        Number of data points corresponding to each STFT segment. This
        parameter must be specified if the number of data points per
        segment is odd, or if the STFT was padded via ``nfft >
        nperseg``. If `None`, the value depends on the shape of
        `Zxx` and `input_onesided`. If `input_onesided` is `True`,
        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,
        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`, half
        of the segment length. Defaults to `None`. When specified, the
        COLA constraint must be met (see Notes below), and should match
        the parameter used to generate the STFT. Defaults to `None`.
    nfft : int, optional
        Number of FFT points corresponding to each STFT segment. This
        parameter must be specified if the STFT was padded via ``nfft >
        nperseg``. If `None`, the default values are the same as for
        `nperseg`, detailed above, with one exception: if
        `input_onesided` is True and
        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on
        that value. This case allows the proper inversion of an
        odd-length unpadded STFT using ``nfft=None``. Defaults to
        `None`.
    input_onesided : bool, optional
        If `True`, interpret the input array as one-sided FFTs, such
        as is returned by `stft` with ``return_onesided=True`` and
        `numpy.fft.rfft`. If `False`, interpret the input as a a
        two-sided FFT. Defaults to `True`.
    boundary : bool, optional
        Specifies whether the input signal was extended at its
        boundaries by supplying a non-`None` ``boundary`` argument to
        `stft`. Defaults to `True`.
    time_axis : int, optional
        Where the time segments of the STFT is located; the default is
        the last axis (i.e. ``axis=-1``).
    freq_axis : int, optional
        Where the frequency axis of the STFT is located; the default is
        the penultimate axis (i.e. ``axis=-2``).
    Returns
    -------
    t : ndarray
        Array of output data times.
    x : ndarray
        iSTFT of `Zxx`.
    See Also
    --------
    stft: Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT with
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):
    .. math:: \sum_{t}w^{2}[n-tH] \ne 0
    This ensures that the normalization factors that appear in the denominator
    of the overlap-add reconstruction equation
    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}
    are not zero. The NOLA constraint can be checked with the `check_NOLA`
    function.
    An STFT which has been modified (via masking or otherwise) is not
    guaranteed to correspond to a exactly realizible signal. This
    function implements the iSTFT via the least-squares estimation
    algorithm detailed in [2]_, which produces a signal that minimizes
    the mean squared error between the STFT of the returned signal and
    the modified STFT.
    .. versionadded:: 0.19.0
    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317
    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by
    0.001 V**2/Hz of white noise sampled at 1024 Hz.
    >>> fs = 1024
    >>> N = 10*fs
    >>> nperseg = 512
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.001 * fs / 2
    >>> time = cp.arange(N) / float(fs)
    >>> carrier = amp * cp.sin(2*cp.pi*50*time)
    >>> noise = cp.random.normal(scale=cp.sqrt(noise_power),
    ...                          size=time.shape)
    >>> x = carrier + noise
    Compute the STFT, and plot its magnitude
    >>> f, t, Zxx = cusignal.stft(x, fs=fs, nperseg=nperseg)
    >>> f = cp.asnumpy(f)
    >>> t = cp.asnumpy(t)
    >>> Zxx = cp.asnumpy(Zxx)
    >>> plt.figure()
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.ylim([f[1], f[-1]])
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.yscale('log')
    >>> plt.show()
    Zero the components that are 10% or less of the carrier magnitude,
    then convert back to a time series via inverse STFT
    >>> Zxx = cp.where(cp.abs(Zxx) >= amp/10, Zxx, 0)
    >>> _, xrec = cusignal.istft(Zxx, fs)
    >>> xrec = cp.asnumpy(xrec)
    >>> x = cp.asnumpy(x)
    >>> time = cp.asnumpy(time)
    >>> carrier = cp.asnumpy(carrier)
    Compare the cleaned signal with the original and true carrier signals.
    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([2, 2.1])*+
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()
    Note that the cleaned signal does not start as abruptly as the original,
    since some of the coefficients of the transient were also removed:
    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([0, 0.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()
    """

    # Make sure input is an ndarray of appropriate complex dtype
    Zxx = cp.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    if Zxx.ndim < 2:
        raise ValueError("Input stft must be at least 2d!")

    if freq_axis == time_axis:
        raise ValueError("Must specify differing time and frequency axes!")

    nseg = Zxx.shape[time_axis]

    if input_onesided:
        # Assume even segment length
        n_default = 2 * (Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    # Check windowing parameters
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            # Odd nperseg, no FFT padding
            nfft = nperseg
        else:
            nfft = n_default
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

    # Rearrange axes if necessary
    if time_axis != Zxx.ndim - 1 or freq_axis != Zxx.ndim - 2:
        # Turn negative indices to positive for the call to transpose
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        Zxx = cp.transpose(Zxx, zouter + [freq_axis, time_axis])

    # Get window as array
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cp.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if win.shape[0] != nperseg:
            raise ValueError("window must have length of {0}".format(nperseg))

    ifunc = cp.fft.irfft if input_onesided else cp.fft.ifft
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # Initialize output and normalization arrays
    outputlength = nperseg + (nseg - 1) * nstep

    if cp.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)

    xsubs *= win.sum()  # This takes care of the 'spectrum' scaling

    # Construct the output from the ifft segments
    # Transposing makes the indexing easier
    xsubs = cp.transpose(xsubs)
    x, norm = _istft_kernel(
        xsubs, win, win.shape[0], win.shape[0] // 2, size=outputlength
    )

    # Remove extension points
    if boundary:
        x = x[..., nperseg // 2 : -(nperseg // 2)]
        norm = norm[..., nperseg // 2 : -(nperseg // 2)]

    # Divide out normalization where non-tiny
    if cp.sum(norm > 1e-10) != len(norm):
        warnings.warn("NOLA condition failed, STFT may not be invertible")
    x /= cp.where(norm > 1e-10, norm, 1.0)

    if input_onesided:
        x = x.real

    # Put axes back
    if x.ndim > 1:
        if time_axis != Zxx.ndim - 1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = cp.rollaxis(x, -1, time_axis)

    time = cp.arange(x.shape[0]) / float(fs)
    return time, x


def coherence(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    axis=-1,
):
    r"""
    Estimate the magnitude squared coherence estimate, Cxy, of
    discrete-time signals X and Y using Welch's method.

    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power
    spectral density estimates of X and Y, and `Pxy` is the cross
    spectral density estimate of X and Y.

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
    axis : int, optional
        Axis along which the coherence is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Cxy : ndarray
        Magnitude squared coherence of x and y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
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
    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
           Signals" Prentice Hall, 2005

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
    >>> # lfilter not implemented in cuSignal
    >>> y = cusignal.lfilter(b, a, x)
    >>> x += amp*cp.sin(2*cp.pi*freq*time)
    >>> y += cp.random.normal(scale=0.1*cp.sqrt(noise_power), size=time.shape)

    Compute and plot the coherence.

    >>> f, Cxy = cusignal.coherence(x, y, fs, nperseg=1024)
    >>> plt.semilogy(cp.asnumpy(f), cp.asnumpy(Cxy))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Coherence')
    >>> plt.show()
    """

    freqs, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        axis=axis,
    )
    _, Pyy = welch(
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        axis=axis,
    )
    _, Pxy = csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        axis=axis,
    )

    Cxy = cp.abs(Pxy) ** 2 / Pxx / Pyy

    return freqs, Cxy


# <学习注释：vectorstrength 把事件时刻按候选周期映射到单位圆，并返回平均单位向量的长度与方向。>
def vectorstrength(events, period):
    """
    Determine the vector strength of the events corresponding to the given
    period.

    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.

    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".

    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.

    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.

    References
    ----------
    .. [1] van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating
           vector strength: Auditory system, electric fish, and noise.
           Chaos 21, 047508 (2011);
           :doi:`10.1063/1.3670512`.
    .. [2] van Hemmen, JL. Vector strength after Goldberg, Brown, and
           von Mises: biological and mathematical perspectives.  Biol Cybern.
           2013 Aug;107(4):385-96. :doi:`10.1007/s00422-013-0561-7`.
    .. [3] van Hemmen, JL and Vollmayr, AN.  Resonating vector strength:
           what happens when we vary the "probing" frequency while keeping
           the spike times fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
           :doi:`10.1007/s00422-013-0560-8`.
    """
    # <学习注释：cp.asarray 将事件时刻转为 CuPy 数组；若输入已经是 CuPy 数组，通常无需复制。>
    events = cp.asarray(events)
    # <学习注释：周期也转为 CuPy 数组，使标量周期和周期向量都能使用统一的 GPU 数组运算。>
    period = cp.asarray(period)
    # <学习注释：events 允许标量或一维序列，但拒绝二维及更高维输入。>
    if events.ndim > 1:
        raise ValueError("events cannot have dimensions more than 1")
    # <学习注释：period 同样只允许标量或一维序列；二维周期网格不属于接口契约。>
    if period.ndim > 1:
        raise ValueError("period cannot have dimensions more than 1")

    # we need to know later if period was originally a scalar
    # <学习注释：零维 CuPy 数组表示原输入为标量；not period.ndim 因而记录是否需要返回标量。>
    scalarperiod = not period.ndim

    # <学习注释：把 events 统一为形状 (1, N)，其中 N 是事件数。>
    events = cp.atleast_2d(events)
    # <学习注释：把 period 统一为形状 (1, P)，其中 P 是候选周期数；标量时 P=1。>
    period = cp.atleast_2d(period)
    # <学习注释：物理周期必须严格大于零；any 汇总所有候选周期的比较结果。>
    if (period <= 0).any():
        raise ValueError("periods must be positive")

    # this converts the times to vectors
    # <学习注释：period.T 形状为 (P,1)，与 events 的 (1,N) 点积/外积得到所有 2πt_i/T_p。>
    # <学习注释：乘以 1j 后取 exp，把每个事件映射为单位复向量 exp(j2πt_i/T_p)，形状为 (P,N)。>
    vectors = cp.exp(cp.dot(2j * cp.pi / period.T, events))

    # the vector strength is just the magnitude of the mean of the vectors
    # the vector phase is the angle of the mean of the vectors
    # <学习注释：axis=1 沿事件维求均值，为每个候选周期得到一个样本一阶圆矩。>
    vectormean = cp.mean(vectors, axis=1)
    # <学习注释：复均值的模是 vector strength，理论范围为 [0,1]。>
    strength = cp.abs(vectormean)
    # <学习注释：复均值的辐角是首选相位；当 strength 为零或极小时该方向没有稳定意义。>
    phase = cp.angle(vectormean)

    # if the original period was a scalar, return scalars
    # <学习注释：标量周期路径拆出第 0 项，使返回值保持为两个零维标量而不是长度 1 数组。>
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
    # <学习注释：返回同步强度和首选相位；周期数组输入时二者均为长度 P 的一维数组。>
    return strength, phase


# <学习注释：_spectral_helper 汇总谱分析公共流程；对 stft 而言，核心路径是参数校验、边界处理、分帧 FFT、幅度缩放和轴恢复。>
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
    # <学习注释：内部 helper 只认识 psd 和 stft；公开 API 的其他模式由 spectrogram 后处理。>
    if mode not in ["psd", "stft"]:
        raise ValueError(
            "Unknown value for mode %s, must be one of: " "{'psd', 'stft'}" % mode
        )

    # <学习注释：把用户字符串映射到四种边界延拓函数；None 表示不延拓。>
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
    # <学习注释：spectrogram 传入同一个 x 对象两次，因此 same_data 为 True，可跳过交叉谱计算。>
    # <学习注释：is 比较对象身份；stft 传入两次同一个 x，因此这里为 True。>
    same_data = y is x

    if not same_data and mode != "psd":
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have cp.arrays, get outdtype
    # <学习注释：把输入统一转换为 CuPy 数组，使后续分帧、窗乘和 FFT 都在 GPU 数组语义下执行。>
    # <学习注释：cp.asarray 把输入转换为 CuPy 数组，已有兼容设备数组通常不会额外复制。>
    x = cp.asarray(x)
    if not same_data:
        y = cp.asarray(y)
        outdtype = cp.result_type(x, y, cp.complex64)
    else:
        # <学习注释：输出至少为 complex64；更高精度或复数输入会按 result_type 提升。>
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

    # <学习注释：内部统一把待分析轴移到末尾，返回前再把频率轴移回用户指定的位置。>
    if x.ndim > 1:
        if axis != -1:
            # <学习注释：内部统一把待分析轴移动到最后，便于 stride 分帧和默认末轴 FFT。>
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
    # <学习注释：公开函数已解析过一次，但 helper 独立复核窗长与输入长度，供其他 API 安全复用。>
    # <学习注释：解析窗口说明或窗口数组，并使实际窗长与 nperseg 保持一致。>
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    # <学习注释：未指定 nfft 时令 FFT 长度等于帧长；更大的 nfft 表示尾部零填充。>
    if nfft is None:
        # <学习注释：未指定 FFT 长度时使用窗长，不进行额外零填充。>
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    # <学习注释：spectrogram 已传入显式 noverlap；这里的 1/2 默认主要服务直接调用 helper 的其他算子。>
    if noverlap is None:
        # <学习注释：默认重叠为半窗；整数除法对奇数窗长向下取整。>
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    # <学习注释：nstep 即 hop，决定相邻帧起点相差的样本数。>
    # <学习注释：帧移 H 等于窗长减重叠点数。>
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        # <学习注释：两端各延拓半窗，使第一个分析窗能够以原输入的第 0 个样本为中心。>
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        # <学习注释：计算最少尾部补零数，使扩展后的长度可被完整窗口和整数个帧移覆盖。>
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = cp.concatenate((x, cp.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = cp.concatenate((y, cp.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    # <学习注释：把三种 detrend 写法统一包装成接收“最后一轴为帧内样本”的函数。>
    if not detrend:

        def detrend_func(d):
            # <学习注释：detrend=False 时恒等返回每个分帧，不删除均值或趋势。>
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

    # <学习注释：窗转换到输出复 dtype，避免窗乘导致不期望的 dtype 提升或不匹配。>
    if cp.result_type(win, cp.complex64) != outdtype:
        win = win.astype(outdtype)

    # <学习注释：density 用窗能量与 fs 归一化，单位为输入平方每 Hz。>
    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    # <学习注释：spectrum 用窗系数和的平方归一化，单位为输入单位的平方。>
    elif scaling == "spectrum":
        # <学习注释：spectrum 标度先建立功率尺度 1/(sum(win))**2。>
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    # <学习注释：功率缩放因子作用于 |X|²；复 STFT 因此必须使用其平方根。>
    if mode == "stft":
        # <学习注释：STFT 是复幅度而非功率，因此对功率尺度开平方。>
        scale = cp.sqrt(scale)

    # <学习注释：实信号可用 rFFT 返回单边谱；复信号不满足一般的共轭对称性，强制双边。>
    if return_onesided:
        if cp.iscomplexobj(x):
            sides = "twosided"
            warnings.warn(
                "Input data is complex, switching to " "return_onesided=False"
            )
        else:
            # <学习注释：实输入可以利用共轭对称性，仅保留非负频率。>
            sides = "onesided"
            if not same_data:
                if cp.iscomplexobj(y):
                    sides = "twosided"
                    warnings.warn(
                        "Input data is complex, switching to " "return_onesided=False"
                    )
    else:
        sides = "twosided"

    # <学习注释：频率数组必须与后续 fft 或 rfft 的 bin 顺序严格对应。>
    if sides == "twosided":
        # <学习注释：fftfreq 返回双边 FFT 的频率坐标，单位由采样间隔 1/fs 决定为 Hz。>
        freqs = cp.fft.fftfreq(nfft, 1 / fs)
    elif sides == "onesided":
        # <学习注释：rfftfreq 只生成从 DC 到 Nyquist 附近的非负频率坐标。>
        freqs = cp.fft.rfftfreq(nfft, 1 / fs)

    # Perform the windowed FFTs
    # <学习注释：_fft_helper 完成零拷贝视图分帧、逐帧去趋势、乘窗与批量 FFT。>
    # <学习注释：进入核心数值步骤：重叠分帧、逐帧去趋势、乘窗和批量 FFT。>
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft, sides)
        result = cp.conj(result) * result_y
    # <学习注释：自谱功率等于 conj(X)·X，即 |X|²；数值上虚部应为零。>
    elif mode == "psd":
        result = cp.conj(result) * result

    # <学习注释：把前面依据 density/spectrum 和 psd/stft 选择的尺度应用到每个 bin。>
    # <学习注释：原地乘幅度尺度；STFT 分支不会进入随后仅针对 PSD 的单边翻倍。>
    result *= scale
    # <学习注释：单边 PSD 把被省略的负频率功率合并到正频率，但 DC 与 Nyquist 不重复。>
    if sides == "onesided" and mode == "psd":
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    # <学习注释：每个时间标签取帧中心位置，而不是帧起点。>
    # <学习注释：先用每个窗口的中心采样位置生成时间标签。>
    time = cp.arange(
        nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap
    ) / float(fs)
    if boundary is not None:
        # <学习注释：边界延拓引入了半窗偏移，减去它后第一帧标签回到原信号 t=0。>
        time -= (nperseg / 2) / fs

    # <学习注释：把 FFT 结果统一转换到前面推导出的复数输出 dtype。>
    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    # <学习注释：自功率谱理论上为实数，这里显式丢弃仅由复数运算产生的零虚部。>
    if same_data and mode != "stft":
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    # <学习注释：FFT 后频率轴位于末尾，把它移回原分析轴位置；时间轴保留为新的末轴。>
    # <学习注释：把频率轴移回原分析轴的位置；新增的帧时间轴保留为最后一轴。>
    result = cp.rollaxis(result, -1, axis)

    return freqs, time, result


# <学习注释：_fft_helper 是正向 STFT 的核心计算 helper，输入轴在进入时已经被统一到最后一轴。>
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
    # <学习注释：长度 1 的退化帧无需构造滑窗 stride，只新增帧内样本轴。>
    if nperseg == 1 and noverlap == 0:
        result = x[..., cp.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        # <学习注释：step 是相邻帧在原数组中移动的样本数。>
        # <学习注释：step 就是相邻分析帧起点之间的样本距离 H。>
        step = nperseg - noverlap
        # <学习注释：倒数第二维是帧数，最后一维是每帧的 nperseg 个样本。>
        # <学习注释：新视图在原外层 shape 后增加“帧数”和“每帧样本数”两个维度。>
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        # <学习注释：帧轴 stride 为 step 个样本，帧内轴 stride 为 1 个样本，因此重叠帧共享底层存储。>
        # <学习注释：帧维跨 step 个元素，帧内维跨 1 个元素，从而形成共享内存的重叠窗口。>
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        # Need to optimize this in cuSignal
        # <学习注释：_as_strided 不复制分帧数据；错误 shape/stride 会越界，因此其参数已在 helper 中严格计算。>
        result = _as_strided(x, shape=shape, strides=strides)

    # Detrend each data segment individually
    # <学习注释：去趋势逐帧作用，默认 spectrogram 会减去每帧均值。>
    # <学习注释：去趋势函数作用在最后一轴，所以每一帧独立处理。>
    result = detrend_func(result)

    # Apply window by multiplication
    # <学习注释：广播使一维窗乘到所有批次和所有时间帧。>
    # <学习注释：广播一维 win 到所有帧，落实 STFT 公式中的逐样本乘窗。>
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    # <学习注释：双边使用 fft；实信号单边先取实部再使用 rfft。>
    if sides == "twosided":
        # <学习注释：双边路径使用一般复数 FFT。>
        func = cp.fft.fft
    else:
        result = result.real
        # <学习注释：单边路径使用实数 FFT，只计算非负频率。>
        func = cp.fft.rfft
    # <学习注释：FFT 沿最后的帧内轴批量执行；nfft 大于 nperseg 时由 FFT 接口自动补零。>
    # <学习注释：在最后一轴批量执行 nfft 点 FFT；nfft 大于帧长时由 FFT 实现自动尾部补零。>
    result = func(result, n=nfft)

    return result


# <学习注释：_triage_segments 将窗口描述转换为实际一维窗数组，并协调窗长、输入长度与 nperseg。>
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
    # <学习注释：字符串或元组是窗规格，需要调用 get_window 生成具体系数。>
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
        # <学习注释：get_window 返回 DFT-even 窗数组；默认 Tukey 元组在这里被实例化。>
        # <学习注释：字符串或元组窗口交给 get_window 生成 DFT-even/periodic 窗；stft 默认由此生成 Hann 窗。>
        win = get_window(window, nperseg)
    else:
        # <学习注释：数组形式的窗由用户直接提供，必须是一维且不能长于输入。>
        # <学习注释：用户直接给数组时转换为 CuPy 数组，并在后续检查它必须是一维且长度匹配。>
        win = cp.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if input_length < win.shape[-1]:
            raise ValueError("window is longer than input signal")
        # <学习注释：未显式给 nperseg 时，以用户窗数组的长度作为帧长。>
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError(
                    "value specified for nperseg is different" " from length of window"
                )
    # <学习注释：调用者得到相互一致的窗系数数组和最终帧长。>
    return win, nperseg


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
