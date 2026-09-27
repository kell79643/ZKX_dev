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
import numpy as np


# <学习注释：定义所有窗函数共用的长度守卫，taylor 在正式计算前也会调用它。>
# <学习注释：parzen 同样先调用这个通用 helper 校验 M 并识别零点或单点窗口。>
def _len_guards(M):
    """Handle small or incorrect window lengths"""
    # <学习注释：要求 M 能无损转换为整数且不能为负，否则拒绝无效窗长。>
    if int(M) != M or M < 0:
        # <学习注释：抛出 ValueError，阻止负数或非整数长度进入后续数组分配。>
        raise ValueError("Window length M must be a non-negative integer")
    # <学习注释：M 为 0 或 1 时返回 True，使 taylor 直接返回同长度的全 1 数组。>
    return M <= 1


# <学习注释：定义对称窗与周期窗共用的长度扩展规则。>
# <学习注释：parzen 用这个 helper 把 sym=False 转换为内部 M+1 点对称窗构造。>
def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    # <学习注释：sym 为 False 表示请求周期窗，需要先生成 M+1 点对称窗。>
    if not sym:
        # <学习注释：返回扩展长度以及 needs_trunc=True，通知末尾删除一个重复端点。>
        return M + 1, True
    else:
        # <学习注释：对称窗保持原长度，并记录不需要截断。>
        return M, False


# <学习注释：定义周期窗生成流程的最后一步，按标志删除扩展样本。>
# <学习注释：parzen 用这个 helper 删除周期模式额外生成且与首点周期对应的末点。>
def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    # <学习注释：needed 为 True 说明此前为了周期采样把长度扩展了一点。>
    if needed:
        # <学习注释：切片 w[:-1] 保留前 M 点并删除 M+1 点对称窗的最后一个端点。>
        return w[:-1]
    else:
        # <学习注释：对称窗没有扩展，直接返回完整数组。>
        return w


_general_cosine_kernel = cp.ElementwiseKernel(
    "raw T a, int32 n",
    "T w",
    """
    const T fac { -M_PI + delta * i };
    T temp {};
    for ( int k = 0; k < n; k++ ) {
        temp += a[k] * cos( k * fac );
    }
    w = temp;
    """,
    "_general_cosine_kernel",
    options=("-std=c++11",),
    loop_prep="const double delta { ( M_PI - -M_PI ) / ( _ind.size() - 1 ) }",
)


def general_cosine(M, a, sym=True):
    r"""
    Generic weighted sum of cosine terms window

    Parameters
    ----------
    M : int
        Number of points in the output window
    a : array_like
        Sequence of weighting coefficients. This uses the convention of being
        centered on the origin, so these will typically all be positive
        numbers, not alternating sign.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_

    .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
               - 0.440811 \cos(3z) + 0.043097 \cos(4z)

    where

    .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1

    Since this uses the convention of starting at the origin, to reproduce the
    window, we need to convert every other coefficient to a positive number:

    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
    Figure 42 by plotting the window and its frequency response, and confirm
    the sidelobe level in red:

    >>> from cusignal.windows import general_cosine
    >>> from cupy.fft import fft, fftshift
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    >>> window = general_cosine(1000, HFT90D, sym=False)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("HFT90D window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 10000) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = cp.abs(fftshift(A / cp.abs(A).max()))
    >>> response = 20 * cp.log10(cp.maximum(response, 1e-10))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-50/1000, 50/1000, -140, 0])
    >>> plt.title("Frequency response of the HFT90D window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axhline(-90.2, color='red')
    >>> plt.show()
    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    a = cp.asarray(a, dtype=cp.float64)

    w = _general_cosine_kernel(a, len(a), size=M)

    return _truncate(w, needs_trunc)


def boxcar(M, sym=True):
    r"""Return a boxcar or rectangular window.

    Also known as a rectangular window or Dirichlet window, this is equivalent
    to no window at all.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        Whether the window is symmetric. (Has no effect for boxcar.)

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.boxcar(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Boxcar window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the boxcar window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = cp.ones(M, dtype=cp.float64)

    return _truncate(w, needs_trunc)


_triang_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    int n {};
    if ( i < m ) {
        n = i + 1;
    } else {
        n = _ind.size() - i;
    }

    if ( odd ) {
        w = 2.0 * n / ( _ind.size() + 1.0 );
    } else {
        w = ( 2.0 * n - 1.0 ) / _ind.size();
    }
    """,
    "_triang_kernel",
    options=("-std=c++11",),
    loop_prep="const int m { static_cast<int>( 0.5 * _ind.size() ) }; \
               const bool odd { _ind.size() & 1 };",
)


def triang(M, sym=True):
    r"""Return a triangular window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    See Also
    --------
    bartlett : A triangular window that touches zero

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.triang(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Triangular window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = cp.abs(fftshift(A / cp.abs(A).max()))
    >>> response = 20 * cp.log10(cp.maximum(response, 1e-10))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the triangular window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _triang_kernel(size=M)

    return _truncate(w, needs_trunc)


# <学习注释：该 CuPy 逐元素 kernel 直接计算 Parzen 分段三次闭式公式。>
_parzen_kernel = cp.ElementwiseKernel(
    # <学习注释：空输入参数字符串表示 kernel 不读取用户提供的逐元素输入数组。>
    "",
    # <学习注释：每个输出位置写入一个 float64 类型的窗口权重 w。>
    "float64 w",
    # <学习注释：三引号开始 CuPy 将编译的 C++/CUDA kernel 主体字符串。>
    """
    // <学习注释：n 保存当前元素相对于窗口中心的有符号离散坐标。>
    double n {};
    // <学习注释：temp 保存归一化距离或一减归一化距离，供三次多项式复用。>
    double temp {};
    // <学习注释：sizeS1 保存左侧外段包含的元素数量。>
    double sizeS1 {};

    // <学习注释：odd 是内部窗口长度的奇偶标志，奇数长度具有单个中心样本。>
    if ( odd ) {
        // <学习注释：奇数长度按整数分界点计算左侧外段元素数。>
        sizeS1 = s1 - start + 1.0;
    // <学习注释：else 处理中心位于两个样本之间的偶数长度窗口。>
    } else {
        // <学习注释：偶数长度把左分段边界平移半个样本。>
        s1 += 0.5;
        // <学习注释：右分段边界采用相同的半样本平移。>
        s2 += 0.5;
        // <学习注释：平移后用边界与 start 的差得到左侧外段元素数。>
        sizeS1 = s1 - start;
    // <学习注释：右花括号结束奇偶长度分支。>
    }

    // <学习注释：中间内段元素数由右边界覆盖范围减去左侧外段元素数得到。>
    double sizeS2 { s2 - start + 1.0 - sizeS1 };

    // <学习注释：前 sizeS1 个元素属于归一化距离较大的左侧外段。>
    if ( i < sizeS1 ) {
        // <学习注释：左侧外段的中心坐标为当前索引 i 加起始坐标 start。>
        n = i + start;
        // <学习注释：den 把绝对坐标归一化，temp 等于 1 减 |u|。>
        temp = 1.0 - abs( n ) * den;
        // <学习注释：外段公式为 w 等于 2 乘以 (1 减 |u|) 的三次方。>
        w = 2.0 * ( temp * temp * temp );
    // <学习注释：中间 sizeS2 个元素属于归一化距离不超过约二分之一的内段。>
    } else if ( i >= sizeS1 && i < ( sizeS1 + sizeS2 ) ) {
        // <学习注释：从全局索引扣除左段长度并结合 s2 恢复中心有符号坐标。>
        n = ( i - sizeS1 - s2 );
        // <学习注释：内段 temp 直接取归一化绝对距离 |u|。>
        temp = abs( n ) * den;
        // <学习注释：内段公式为 1 减 6|u|平方再加 6|u|立方。>
        w = 1.0 - 6.0 * temp * temp + 6.0 * temp * temp * temp;
    // <学习注释：剩余元素属于与左侧镜像对应的右侧外段。>
    } else {
        // <学习注释：该表达式把右侧索引转换为关于中心对称的有符号坐标。>
        n = -( i - sizeS2 + s1 + sizeS1 );
        // <学习注释：与左外段相同，temp 等于 1 减归一化绝对距离。>
        temp = 1.0 - abs( n ) * den;
        // <学习注释：右外段复用相同的三次公式以保证偶对称。>
        w = 2.0 * temp * temp * temp;
    // <学习注释：右花括号结束三段位置分支。>
    }
    // <学习注释：三引号下一行结束 kernel 主体字符串。>
    """,
    # <学习注释：该字符串是 CuPy 为编译缓存和诊断使用的 kernel 名称。>
    "_parzen_kernel",
    # <学习注释：编译选项要求 kernel C++ 代码使用 C++11 标准。>
    options=("-std=c++11",),
    # <学习注释：loop_prep 在逐元素循环前计算中心起点、归一化因子、奇偶标志和分段边界。>
    loop_prep="const double start { 0.5 * -( _ind.size () - 1 ) }; \
               const double den { 1.0 / ( 0.5 * _ind.size () ) }; \
               const bool odd { _ind.size() & 1 }; \
               double s1 { floor(-0.25 * ( _ind.size () - 1 ) ) }; \
               double s2 { floor(0.25 * ( _ind.size () - 1 ) ) };",
)


# <学习注释：公开函数 parzen 负责校验长度、选择对称或周期模式并启动 kernel。>
def parzen(M, sym=True):

    # <学习注释：先处理非法长度以及 M 为 0 或 1 的退化情况。>
    if _len_guards(M):
        # <学习注释：M 为 0 返回空数组，M 为 1 返回单个 1。>
        return cp.ones(M)
    # <学习注释：周期模式把内部长度扩为 M+1，对称模式保持 M，并记录最终是否截断。>
    M, needs_trunc = _extend(M, sym)

    # <学习注释：size=M 让 CuPy 为 M 个输出位置逐元素生成 float64 权重。>
    w = _parzen_kernel(size=M)

    # <学习注释：周期模式删除额外末点，对称模式原样返回，最终长度均为用户请求的 M。>
    return _truncate(w, needs_trunc)


_bohman_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    const double fac { abs( start + delta * ( i - 1 ) ) };
    if ( i != 0 && i != ( _ind.size() - 1 ) ) {
        w = ( 1.0 - fac ) * cos( M_PI * fac ) + 1.0 / M_PI * sin( M_PI * fac );
    } else {
        w = 0.0;
    }
    """,
    "_bohman_kernel",
    options=("-std=c++11",),
    loop_prep="const double delta { 2.0 / ( _ind.size() - 1 ) }; \
               const double start { -1.0 + delta };",
)


def bohman(M, sym=True):
    r"""Return a Bohman window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.bohman(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Bohman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bohman window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _bohman_kernel(size=M)

    return _truncate(w, needs_trunc)


def blackman(M, sym=True):
    r"""
    Return a Blackman window.

    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)

    The "exact Blackman" window was designed to null out the third and fourth
    sidelobes, but has discontinuities at the boundaries, resulting in a
    6 dB/oct fall-off.  This window is an approximation of the "exact" window,
    which does not null the sidelobes as well, but is smooth at the edges,
    improving the fall-off rate to 18 dB/oct. [3]_

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the Kaiser window.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
           Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
    .. [3] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.blackman(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Blackman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = cp.abs(fftshift(A / cp.abs(A).max()))
    >>> response = 20 * cp.log10(cp.maximum(response, 1e-10))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's blackman function
    return general_cosine(M, [0.42, 0.50, 0.08], sym)


def nuttall(M, sym=True):
    r"""Return a minimum 4-term Blackman-Harris window according to Nuttall.

    This variation is called "Nuttall4c" by Heinzel. [2]_

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.nuttall(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Nuttall window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Nuttall window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    return general_cosine(M, [0.3635819, 0.4891775, 0.1365995, 0.0106411], sym)


def blackmanharris(M, sym=True):
    r"""Return a minimum 4-term Blackman-Harris window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.blackmanharris(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Blackman-Harris window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman-Harris window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    return general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)


def flattop(M, sym=True):
    r"""Return a flat top window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    Flat top windows are used for taking accurate measurements of signal
    amplitude in the frequency domain, with minimal scalloping error from the
    center of a frequency bin to its edges, compared to others.  This is a
    5th-order cosine window, with the 5 terms optimized to make the main lobe
    maximally flat. [1]_

    References
    ----------
    .. [1] D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for
           Measurement Systems", Springer Media, 2006, p. 70
           :doi:`10.1007/0-387-28666-7`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.flattop(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Flat top window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the flat top window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    return general_cosine(M, a, sym)


_bartlett_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    if ( i <= temp ) {
        w = 2.0 * i * N;
    } else {
        w = 2.0 - 2.0 * i * N;
    }
    """,
    "_bartlett_kernel",
    options=("-std=c++11",),
    loop_prep="const double N { 1.0 / ( _ind.size() - 1 ) }; \
               const double temp { 0.5 * ( _ind.size() - 1 ) };",
)


def bartlett(M, sym=True):
    r"""
    Return a Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The triangular window, with the first and last samples equal to zero
        and the maximum value normalized to 1 (though the value 1 does not
        appear if `M` is even and `sym` is True).

    See Also
    --------
    triang : A triangular window that does not touch zero at the ends

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \frac{2}{M-1} \left(
              \frac{M-1}{2} - \left|n - \frac{M-1}{2}\right|
              \right)

    Most references to the Bartlett window come from the signal
    processing literature, where it is used as one of many windowing
    functions for smoothing values.  Note that convolution with this
    window produces linear interpolation.  It is also known as an
    apodization (which means"removing the foot", i.e. smoothing
    discontinuities at the beginning and end of the sampled signal) or
    tapering function. The Fourier transform of the Bartlett is the product
    of two sinc functions.
    Note the excellent discussion in Kanasewich. [2]_

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.bartlett(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Bartlett window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's bartlett function
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _bartlett_kernel(size=M)

    return _truncate(w, needs_trunc)


def hann(M, sym=True):
    r"""
    Return a Hann window.

    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hann window is defined as

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The window was named for Julius von Hann, an Austrian meteorologist. It is
    also known as the Cosine Bell. It is sometimes erroneously referred to as
    the "Hanning" window, from the use of "hann" as a verb in the original
    paper and confusion with the very similar Hamming window.

    Most references to the Hann window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.hann(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = cp.abs(fftshift(A / cp.abs(A).max()))
    >>> response = 20 * cp.log10(np.maximum(response, 1e-10))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hanning function
    return general_hamming(M, 0.5, sym)


_tukey_kernel = cp.ElementwiseKernel(
    "float64 alpha",
    "float64 w",
    """
    if ( i < ( width + 1 ) ) {
        w = 0.5 * ( 1 + cos( M_PI * ( -1.0 + 2.0 * i / alpha * N ) ) );
    } else if ( i > ( width + 1 ) && i < ( _ind.size() - width - 1) ) {
        w = 1.0;
    } else {
        w = 0.5 *
            ( 1.0 + cos( M_PI * ( -2.0 / alpha + 1 + 2.0 * i / alpha * N ) ) );
    }
    """,
    "_tukey_kernel",
    options=("-std=c++11",),
    loop_prep="const double N { 1.0 / ( _ind.size() - 1 ) }; \
               const int width { static_cast<int>( alpha * \
                   ( _ind.size() - 1 ) * 0.5 ) }",
)


def tukey(M, alpha=0.5, sym=True):
    r"""Return a Tukey window, also known as a tapered cosine window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Tukey_window

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.tukey(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Tukey window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.ylim([0, 1.1])

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Tukey window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)

    if alpha <= 0:
        return cp.ones(M, "d")
    elif alpha >= 1.0:
        return hann(M, sym=sym)

    M, needs_trunc = _extend(M, sym)

    w = _tukey_kernel(alpha, size=M)

    return _truncate(w, needs_trunc)


_barthann_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    const double fac { abs( i * N - 0.5 ) };
    w = 0.62 - 0.48 * fac + 0.38 * cos(2.0 * M_PI * fac);
    """,
    "_barthann_kernel",
    options=("-std=c++11",),
    loop_prep="const double N { 1.0 / ( _ind.size() - 1 ) };",
)


def barthann(M, sym=True):
    r"""Return a modified Bartlett-Hann window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.barthann(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Bartlett-Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett-Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _barthann_kernel(size=M)

    return _truncate(w, needs_trunc)


def general_hamming(M, alpha, sym=True):
    r"""Return a generalized Hamming window.

    The generalized Hamming window is constructed by multiplying a rectangular
    window by one period of a cosine function [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float
        The window coefficient, :math:`\alpha`
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Hamming window is defined as

    .. math:: w(n) = \alpha -
              \left(1 - \alpha\right) \cos\left(\frac{2\pi{n}}{M-1}\right)
              \qquad 0 \leq n \leq M-1

    Both the common Hamming window and Hann window are special cases of the
    generalized Hamming window with :math:`\alpha` = 0.54 and :math:`\alpha` =
    0.5, respectively [2]_.

    See Also
    --------
    hamming, hann

    Examples
    --------
    The Sentinel-1A/B Instrument Processing Facility uses generalized Hamming
    windows in the processing of spaceborne Synthetic Aperture Radar (SAR)
    data [3]_. The facility uses various values for the :math:`\alpha`
    parameter based on operating mode of the SAR instrument. Some common
    :math:`\alpha` values include 0.75, 0.7 and 0.52 [4]_. As an example, we
    plot these different windows.

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> fig1, spatial_plot = plt.subplots()
    >>> spatial_plot.set_title("Generalized Hamming Windows")
    >>> spatial_plot.set_ylabel("Amplitude")
    >>> spatial_plot.set_xlabel("Sample")

    >>> fig2, freq_plot = plt.subplots()
    >>> freq_plot.set_title("Frequency Responses")
    >>> freq_plot.set_ylabel("Normalized magnitude [dB]")
    >>> freq_plot.set_xlabel("Normalized frequency [cycles per sample]")

    >>> for alpha in [0.75, 0.7, 0.52]:
    ...     window = cusignal.general_hamming(41, alpha)
    ...     spatial_plot.plot(cp.asnumpy(window), label="{:.2f}".format(alpha))
    ...     A = fft(window, 2048) / (len(window)/2.0)
    ...     freq = cp.linspace(-0.5, 0.5, len(A))
    ...     response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    ...     freq_plot.plot(
    ...         cp.asnumpy(freq), cp.asnumpy(response),
    ...         label="{:.2f}".format(alpha)
    ...     )
    >>> freq_plot.legend(loc="upper right")
    >>> spatial_plot.legend(loc="upper right")

    References
    ----------
    .. [1] DSPRelated, "Generalized Hamming Window Family",
           https://www.dsprelated.com/freebooks/sasp/Generalized_Hamming_Window_Family.html
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [3] Riccardo Piantanida ESA, "Sentinel-1 Level 1 Detailed Algorithm
           Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition
    .. [4] Matthieu Bourbigot ESA, "Sentinel-1 Product Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Definition
    """
    return general_cosine(M, [alpha, 1.0 - alpha], sym)


_hamming_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    w = 0.54 - 0.46 * cos(2.0 * M_PI * i * N);
    """,
    "_hamming_kernel",
    options=("-std=c++11",),
    loop_prep="const double N { 1.0 / ( _ind.size() - 1 ) };",
)


def hamming(M, sym=True):
    r"""
    Return a Hamming window.

    The Hamming window is a taper formed by using a raised cosine with
    non-zero endpoints, optimized to minimize the nearest side lobe.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and
    is described in Blackman and Tukey. It was recommended for smoothing the
    truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.hamming(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Hamming window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hamming window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if M < 1:
        return cp.array([])
    if M == 1:
        return cp.ones(1, "d")
    odd = M % 2
    if not sym and not odd:
        M = M + 1

    w = _hamming_kernel(size=M)

    if not sym and not odd:
        w = w[:-1]
    return w


# <学习注释：定义逐元素 GPU kernel；每个输出索引独立计算一个 Kaiser 窗系数。>
_kaiser_kernel = cp.ElementwiseKernel(
    # <学习注释：kernel 只接收一个 float64 标量 beta，作为 Kaiser 窗的形状参数。>
    "float64 beta",
    # <学习注释：kernel 的每个线程写出一个 float64 窗系数 w。>
    "float64 w",
    """
    // <学习注释：把当前索引 i 平移到窗中心 alpha，再除以 alpha，得到范围为 [-1, 1] 的归一化坐标 temp。>
    const double temp { ( i - alpha ) / alpha };
    // <学习注释：按 I0(beta*sqrt(1-temp^2))/I0(beta) 计算归一化 Kaiser 窗，其中 cyl_bessel_i0 是零阶第一类修正 Bessel 函数。>
    w = cyl_bessel_i0( beta * sqrt( 1.0 - ( temp * temp ) ) ) /
    // <学习注释：这一续行给出分母 I0(beta)，使连续窗中心值归一化为 1。>
        cyl_bessel_i0( beta );
    """,
    # <学习注释：为 CuPy kernel 注册内部名称 _kaiser_kernel。>
    "_kaiser_kernel",
    # <学习注释：要求设备端 C++ 源码按 C++11 编译，以支持这里使用的语法和数学函数。>
    options=("-std=c++11",),
    # <学习注释：在元素循环前按实际输出长度预计算 alpha=(M-1)/2，供所有索引共享。>
    loop_prep="const double alpha { 0.5 * ( _ind.size() - 1 ) };",
)


# <学习注释：公开函数接收窗长 M、形状参数 beta 和对称性标志 sym，并返回 GPU 上的 float64 Kaiser 窗。>
def kaiser(M, beta, sym=True):
    r"""
    Return a Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.

    The Kaiser can approximate other windows by varying the beta parameter.
    (Some literature uses alpha = beta/pi.) [4]_

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hann
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    be returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transform," Proceedings of the IEEE, vol. 66,
           no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.kaiser(51, beta=14)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title(r"Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # <学习注释：长度小于 1 时提前返回空 CuPy 数组，避免进入需要除以 M-1 的通式。>
    if M < 1:
        # <学习注释：构造并立即返回空数组；此分支也会接纳负 M。>
        return cp.array([])
    # <学习注释：单点窗没有可采样的形状变化，按单位权重 [1] 处理。>
    if M == 1:
        # <学习注释：dtype 字符 d 表示双精度浮点数，即 float64。>
        return cp.ones(1, "d")
    # <学习注释：M%2 在偶数时为 0、奇数时为 1，后续条件用它决定是否扩展周期窗。>
    odd = M % 2
    # <学习注释：仅当请求周期窗且原始 M 为偶数时，把计算长度临时扩展为 M+1。>
    if not sym and not odd:
        # <学习注释：扩展后 kernel 的分母 M_extended-1 等于原始 M，符合周期采样所需的归一化坐标。>
        M = M + 1

    # <学习注释：以实际计算长度启动逐元素 kernel；CuPy 根据 size=M 分配并填充输出。>
    w = _kaiser_kernel(beta, size=M)

    # <学习注释：若前面为偶数周期窗增加了一个样本，这里删除最后的重复周期端点。>
    if not sym and not odd:
        # <学习注释：切片保留前 M_original 个样本，返回长度恢复为调用者请求的长度。>
        w = w[:-1]
    # <学习注释：把位于 GPU 设备内存中的 Kaiser 窗数组返回给调用者。>
    return w


_gaussian_kernel = cp.ElementwiseKernel(
    "float64 std",
    "float64 w",
    """
    const double n { i - (_ind.size() - 1.0) * 0.5 };
    w = exp( - ( n * n ) / sig2 );
    """,
    "_gaussian_kernel",
    options=("-std=c++11",),
    loop_prep="const double sig2 { 2.0 * std * std };",
)


def gaussian(M, std, sym=True):
    r"""Return a Gaussian window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    std : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Gaussian window is defined as

    .. math::  w(n) = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.gaussian(51, std=7)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title(r"Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _gaussian_kernel(std, size=M)

    return _truncate(w, needs_trunc)


_general_gaussian_kernel = cp.ElementwiseKernel(
    "float64 p, float64 sig",
    "float64 w",
    """
    const double n { i - ( _ind.size() - 1.0 ) * 0.5 };
    w = exp( -0.5 * pow( abs( n / sig ), 2.0 * p ) );
    """,
    "_general_gaussian_kernel",
    options=("-std=c++11",),
)


def general_gaussian(M, p, sig, sym=True):
    r"""Return a window with a generalized Gaussian shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    p : float
        Shape parameter.  p = 1 is identical to `gaussian`, p = 0.5 is
        the same shape as the Laplace distribution.
    sig : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Gaussian window is defined as

    .. math::  w(n) = e^{ -\frac{1}{2}\left|\frac{n}{\sigma}\right|^{2p} }

    the half-power point is at

    .. math::  (2 \log(2))^{1/(2 p)} \sigma

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.general_gaussian(51, p=1.5, sig=7)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title(r"Generalized Gaussian window (p=1.5, $\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Freq. resp. of the gen. Gaussian "
    ...           r"window (p=1.5, $\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _general_gaussian_kernel(p, sig, size=M)

    return _truncate(w, needs_trunc)


_chebwin_kernel = cp.ElementwiseKernel(
    "int64 order, float64 beta",
    "complex128 p",
    """
    double real {};
    const double x { beta * cos( i * N ) };

    if ( x > 1 ) {
        real = cosh( order * acosh( x ) );
    } else if ( x < -1 ) {
        real = ( 2.0 * ( _ind.size() & 1 ) - 1.0 ) *
            cosh( order * acosh( -x ) );
    } else {
        real = cos( order * acos( x ) );
    }

    if ( odd ) {
        p = real;
    } else {
        p = real * exp( thrust::complex<double>( 0.0, N * i ) );
    }
    """,
    "_chebwin_kernel",
    options=("-std=c++11",),
    loop_prep="const double N { M_PI * ( 1.0 / _ind.size() ) }; \
               const bool odd { _ind.size() & 1 };",
)


# `chebwin` contributed by Kumar Appaiah.
def chebwin(M, at, sym=True):
    r"""Return a Dolph-Chebyshev window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    at : float
        Attenuation (in dB).
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value always normalized to 1

    Notes
    -----
    This window optimizes for the narrowest main lobe width for a given order
    `M` and sidelobe equiripple attenuation `at`, using Chebyshev
    polynomials.  It was originally developed by Dolph to optimize the
    directionality of radio antenna arrays.

    Unlike most windows, the Dolph-Chebyshev is defined in terms of its
    frequency response:

    .. math:: W(k) = \frac
              {\cos\{M \cos^{-1}[\beta \cos(\frac{\pi k}{M})]\}}
              {\cosh[M \cosh^{-1}(\beta)]}

    where

    .. math:: \beta = \cosh \left [\frac{1}{M}
              \cosh^{-1}(10^\frac{A}{20}) \right ]

    and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).

    The time domain window is then generated using the IFFT, so
    power-of-two `M` are the fastest to generate, and prime number `M` are
    the slowest.

    The equiripple condition in the frequency domain creates impulses in the
    time domain, which appear at the ends of the window.

    References
    ----------
    .. [1] C. Dolph, "A current distribution for broadside arrays which
           optimizes the relationship between beam width and side-lobe level",
           Proceedings of the IEEE, Vol. 34, Issue 6
    .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",
           American Meteorological Society (April 1997)
           http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf
    .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,
           No. 1, January 1978

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.chebwin(51, at=100)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """

    if abs(at) < 45:
        warnings.warn(
            "This window is not suitable for spectral analysis "
            "for attenuation values lower than about 45dB because "
            "the equivalent noise bandwidth of a Chebyshev window "
            "does not grow monotonically with increasing sidelobe "
            "attenuation when the attenuation is smaller than "
            "about 45 dB."
        )
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    # compute the parameter beta
    order = M - 1.0
    beta = np.cosh(1.0 / order * np.arccosh(10 ** (abs(at) / 20.0)))

    # Appropriate IDFT and filling up
    # depending on even/odd M
    p = _chebwin_kernel(order, beta, size=M)
    if M % 2:
        w = cp.real(cp.fft.fft(p))
        n = (M + 1) // 2
        w = w[:n]
        w = cp.concatenate((w[n - 1 : 0 : -1], w))
    else:
        w = cp.real(cp.fft.fft(p))
        n = M // 2 + 1
        w = cp.concatenate((w[n - 1 : 0 : -1], w[1:n]))

    w = w / cp.max(w)

    return _truncate(w, needs_trunc)


_cosine_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    w = sin( M_PI / _ind.size() * ( i + 0.5 ) );
    """,
    "_cosine_kernel",
    options=("-std=c++11",),
)


def cosine(M, sym=True):
    r"""Return a window with a simple cosine shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----

    .. versionadded:: 0.13.0

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.cosine(51)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Cosine window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the cosine window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.show()

    """
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _cosine_kernel(size=M)

    return _truncate(w, needs_trunc)


_exponential_kernel = cp.ElementwiseKernel(
    "float64 center, float64 tau",
    "float64 w",
    """
    w = exp( -abs( i - center ) / tau );
    """,
    "_exponential_kernel",
    options=("-std=c++11",),
)


def exponential(M, center=None, tau=1.0, sym=True):
    r"""Return an exponential (or Poisson) window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    center : float, optional
        Parameter defining the center location of the window function.
        The default value if not given is ``center = (M-1) / 2``.  This
        parameter must take its default value for symmetric windows.
    tau : float, optional
        Parameter defining the decay.  For ``center = 0`` use
        ``tau = -(M-1) / ln(x)`` if ``x`` is the fraction of the window
        remaining at the end.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Exponential window is defined as

    .. math::  w(n) = e^{-|n-center| / \tau}

    References
    ----------
    S. Gade and H. Herlufsen, "Windows to FFT analysis (Part I)",
    Technical Review 3, Bruel & Kjaer, 1987.

    Examples
    --------
    Plot the symmetric window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> M = 51
    >>> tau = 3.0
    >>> window = cusignal.exponential(M, tau=tau)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title("Exponential Window (tau=3.0)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -35, 0])
    >>> plt.title("Frequency response of the Exponential window (tau=3.0)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    This function can also generate non-symmetric windows:

    >>> tau2 = -(M-1) / np.log(0.01)
    >>> window2 = cusignal.exponential(M, 0, tau2, False)
    >>> plt.figure()
    >>> plt.plot(cp.asnumpy(window2))
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    """
    if sym and center is not None:
        raise ValueError("If sym==True, center must be None.")
    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    if center is None:
        center = (M - 1) / 2

    w = _exponential_kernel(center, tau, size=M)

    return _truncate(w, needs_trunc)


# <学习注释：创建 CuPy 逐元素 Taylor kernel，每个 GPU 输出元素独立计算一个窗样本。>
_taylor_kernel = cp.ElementwiseKernel(
    # <学习注释：输入依次为近等旁瓣参数 nbar、只读 float64 系数数组 Fm 和归一化开关 norm。>
    "int64 nbar, raw float64 Fm, bool norm",
    # <学习注释：每个线程写出一个 float64 窗样本 out。>
    "float64 out",
    """
    // <学习注释：temp 是当前索引 i 的中心对齐离散相位基量，等于 2π(i-M/2+1/2)/M。>
    double temp { mod_pi * ( i - _ind.size() / 2.0 + 0.5 ) };
    // <学习注释：把余弦级数累加器 dot 值初始化为 0。>
    double dot {};

    // <学习注释：遍历 m=1 到 nbar-1，累加 Taylor 有限余弦级数。>
    for ( int k = 1; k < nbar; k++ ) {
        // <学习注释：Fm[k-1] 对应数学系数 F_k，cos(temp*k) 对应第 k 个余弦基。>
        dot += Fm[k-1] * cos( temp * k );
    }
    // <学习注释：加入常数项 1 和偶对称频率对带来的系数 2，得到未归一化窗值。>
    out = 1.0 + 2.0 * dot;

    // <学习注释：默认缩放因子为 1，norm=False 时保持 DC gain 归一化。>
    double scale { 1.0 };
    // <学习注释：只有 norm=True 时才计算连续窗中心值的倒数。>
    if (norm == 1) {
        // <学习注释：清空累加器，准备在统一中心位置重新计算余弦和。>
        dot = 0;
        // <学习注释：用 (M-1)/2 代入中心位置；偶数长度时该位置位于两个中心样本之间。>
        temp = mod_pi * ( ( ( _ind.size() - 1.0 ) / 2.0 )
            - _ind.size() / 2.0 + 0.5 );
        // <学习注释：在中心位置重新累加全部 Taylor 余弦项。>
        for ( int k = 1; k < nbar; k++ ) {
            dot += Fm[k-1] * cos( temp * k );
        }
        // <学习注释：取中心窗值 1+2*dot 的倒数，使连续中心峰值归一化为 1。>
        scale = 1.0 / ( 1.0 + 2.0 * dot );
    }

    // <学习注释：将统一缩放因子乘到当前线程的未归一化窗值。>
    out *= scale;
    """,
    # <学习注释：kernel 名称用于 CuPy 编译缓存和诊断标识。>
    "_taylor_kernel",
    # <学习注释：要求即时编译器按 C++11 语法编译 kernel 字符串。>
    options=("-std=c++11",),
    # <学习注释：在逐元素循环前计算一次 2π/M，供所有当前 kernel 元素复用。>
    loop_prep="const double mod_pi { 2.0 * M_PI / _ind.size() }",
)


# <学习注释：定义公开 Taylor 窗 API，默认生成长度 M、nbar=4、30 dB、峰值归一化的对称窗。>
def taylor(M, nbar=4, sll=30, norm=True, sym=True):
    """
    Return a Taylor window.
    The Taylor window taper function approximates the Dolph-Chebyshev window's
    constant sidelobe level for a parameterized number of near-in sidelobes,
    but then allows a taper beyond [2]_.
    The SAR (synthetic aperature radar) community commonly uses Taylor
    weighting for image formation processing because it provides strong,
    selectable sidelobe suppression with minimum broadening of the
    mainlobe [1]_.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    nbar : int, optional
        Number of nearly constant level sidelobes adjacent to the mainlobe.
    sll : float, optional
        Desired suppression of sidelobe level in decibels (dB) relative to the
        DC gain of the mainlobe. This should be a positive number.
    norm : bool, optional
        When True (default), divides the window by the largest (middle) value
        for odd-length windows or the value that would occur between the two
        repeated middle values for even-length windows such that all values
        are less than or equal to 1. When False the DC gain will remain at 1
        (0 dB) and the sidelobes will be `sll` dB down.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    out : array
        The window. When `norm` is True (default), the maximum value is
        normalized to 1 (though the value 1 does not appear if `M` is
        even and `sym` is True).
    See Also
    --------
    chebwin, kaiser, bartlett, blackman, hamming, hanning
    References
    ----------
    .. [1] W. Carrara, R. Goodman, and R. Majewski, "Spotlight Synthetic
           Aperture Radar: Signal Processing Algorithms" Pages 512-513,
           July 1995.
    .. [2] Armin Doerry, "Catalog of Window Taper Functions for
           Sidelobe Control", 2017.
           https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
    Examples
    --------
    Plot the window and its frequency response:
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    >>> window = signal.windows.taylor(51, nbar=20, sll=100, norm=False)
    >>> plt.plot(window)
    >>> plt.title("Taylor window (100 dB)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Taylor window (100 dB)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    """  # noqa: E501
    # <学习注释：先验证 M；对 M=0 或 M=1 走无需 Taylor 系数的短路径。>
    if _len_guards(M):
        # <学习注释：短窗返回 NumPy 全 1 数组；这是源码现状，与一般路径返回 CuPy 数组存在类型差异。>
        return np.ones(M)
    # <学习注释：周期窗先扩为 M+1 点，并保存最终是否需要删除末点的标志。>
    M, needs_trunc = _extend(M, sym)

    # Original text uses a negative sidelobe level parameter and then negates
    # it in the calculation of B. To keep consistent with other methods we
    # assume the sidelobe level parameter to be positive.
    # <学习注释：把正的 sll dB 抑制度转换为主瓣与旁瓣的线性幅度比 B。>
    B = 10 ** (sll / 20)
    # <学习注释：由 B 计算 Taylor 双曲余弦形状参数 A=arccosh(B)/π。>
    A = np.arccosh(B) / np.pi
    # <学习注释：计算近端零点尺度 σ²，变量 s2 对应数学公式中的 sigma squared。>
    s2 = nbar**2 / (A**2 + (nbar - 0.5) ** 2)
    # <学习注释：生成整数谐波索引 ma=[1,2,...,nbar-1]。>
    ma = np.arange(1, nbar)

    # <学习注释：分配 nbar-1 个 float64 Taylor 余弦系数 Fm。>
    Fm = np.empty(nbar - 1)
    # <学习注释：分配与 ma 同 dtype/shape 的交替符号数组。>
    signs = np.empty_like(ma)
    # <学习注释：偶数下标对应 m=1,3,...，设置 (-1)^(m+1)=+1。>
    signs[::2] = 1
    # <学习注释：奇数下标对应 m=2,4,...，设置 (-1)^(m+1)=-1。>
    signs[1::2] = -1
    # <学习注释：预先计算全部 m²，供系数乘积公式重复使用。>
    m2 = ma * ma
    # <学习注释：逐个计算 m=1 到 nbar-1 的闭式 Taylor 系数。>
    for mi, _ in enumerate(ma):
        # <学习注释：分子把修改后的 Taylor 零点位置编码为乘积，并乘交替符号。>
        numer = signs[mi] * np.prod(1 - m2[mi] / s2 / (A**2 + (ma - 0.5) ** 2))
        # <学习注释：分母计算 2 倍的整数谐波插值乘积，并用前后切片排除 j=m 的零因子。>
        denom = 2 * np.prod(1 - m2[mi] / m2[:mi]) * np.prod(1 - m2[mi] / m2[mi + 1 :])
        # <学习注释：分子除以分母得到当前 F_m，写入 Fm[mi]。>
        Fm[mi] = numer / denom

    # <学习注释：把 CPU NumPy 系数复制/转换为 CuPy 数组，并启动 M 个输出元素的 GPU kernel。>
    w = _taylor_kernel(nbar, cp.asarray(Fm), norm, size=M)

    # <学习注释：对周期窗删除扩展末点，对对称窗原样返回。>
    return _truncate(w, needs_trunc)


def _fftautocorr(x):
    """Compute the autocorrelation of a real array and crop the result."""
    N = x.shape[-1]
    use_N = cp.fft.next_fast_len(2 * N - 1)
    x_fft = cp.fft.rfft(x, use_N, axis=-1)
    cxy = cp.fft.irfft(x_fft * x_fft.conj(), n=use_N)[:, :N]
    # Or equivalently (but in most cases slower):
    # cxy = np.array([np.convolve(xx, yy[::-1], mode='full')
    #                 for xx, yy in zip(x, x)])[:, N-1:2*N-1]
    return cxy


_win_equiv_raw = {
    ("barthann", "brthan", "bth"): (barthann, False),
    ("bartlett", "bart", "brt"): (bartlett, False),
    ("blackman", "black", "blk"): (blackman, False),
    ("blackmanharris", "blackharr", "bkh"): (blackmanharris, False),
    ("bohman", "bman", "bmn"): (bohman, False),
    ("boxcar", "box", "ones", "rect", "rectangular"): (boxcar, False),
    ("chebwin", "cheb"): (chebwin, True),
    ("cosine", "halfcosine"): (cosine, False),
    ("exponential", "poisson"): (exponential, True),
    ("flattop", "flat", "flt"): (flattop, False),
    ("gaussian", "gauss", "gss"): (gaussian, True),
    (
        "general gaussian",
        "general_gaussian",
        "general gauss",
        "general_gauss",
        "ggs",
    ): (general_gaussian, True),
    ("hamming", "hamm", "ham"): (hamming, False),
    ("hanning", "hann", "han"): (hann, False),
    ("kaiser", "ksr"): (kaiser, True),
    ("nuttall", "nutl", "nut"): (nuttall, False),
    # <学习注释：窗口工厂把 parzen、parz 和 par 三个名称映射到同一函数，False 表示无需额外形状参数。>
    ("parzen", "parz", "par"): (parzen, False),
    # ('slepian', 'slep', 'optimal', 'dpss', 'dss'): (slepian, True),
    ("triangle", "triang", "tri"): (triang, False),
    ("tukey", "tuk"): (tukey, True),
}

# Fill dict with all valid window name strings
_win_equiv = {}
for k, v in _win_equiv_raw.items():
    for key in k:
        _win_equiv[key] = v[0]

# Keep track of which windows need additional parameters
_needs_param = set()
for k, v in _win_equiv_raw.items():
    if v[1]:
        _needs_param.update(k)


# <学习注释：get_window 把窗口名称、参数元组或 Kaiser beta 数值解析为实际窗函数数组。>
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
    # <学习注释：窗口函数使用 sym 标志，而公开接口使用相反语义的 fftbins，因此这里逻辑取反。>
    sym = not fftbins
    # <学习注释：首先尝试把 window 转成浮点数；成功时按 Kaiser 窗的 beta 参数解释。>
    try:
        # <学习注释：float 转换也接受可转换的数值字符串，但普通窗名会进入异常分支。>
        beta = float(window)
    # <学习注释：不能解释为数值时，再按窗口名称或参数元组进行分派。>
    except (TypeError, ValueError):
        # <学习注释：args 保存除窗口名称外要传给具体窗函数的额外参数。>
        args = ()
        # <学习注释：元组的第一个元素是窗口名，后续元素是 beta、标准差等参数。>
        if isinstance(window, tuple):
            # <学习注释：取出名称供后面的等价名称表查询。>
            winstr = window[0]
            # <学习注释：只有元组确实带额外元素时才覆盖空参数元组。>
            if len(window) > 1:
                # <学习注释：切片保留全部额外参数及其原始顺序。>
                args = window[1:]
        # <学习注释：字符串形式适合不需要额外参数的窗口。>
        elif isinstance(window, str):
            # <学习注释：若该窗必须带参数，单独字符串不足以完成调用。>
            if window in _needs_param:
                raise ValueError(
                    "The '" + window + "' window needs one or "
                    "more parameters -- pass a tuple."
                )
            else:
                # <学习注释：无需参数的字符串直接作为分派键。>
                winstr = window
        # <学习注释：既不是数值、元组也不是字符串的对象不属于支持的窗口描述格式。>
        else:
            raise ValueError("%s as window type is not supported." % str(type(window)))

        # <学习注释：使用名称到函数的映射表解析具体窗函数及其别名。>
        try:
            # <学习注释：查表结果 winfunc 是可调用的具体窗函数。>
            winfunc = _win_equiv[winstr]
        # <学习注释：名称不在映射表中时把内部 KeyError 转换为更清晰的 ValueError。>
        except KeyError:
            raise ValueError("Unknown window type.")

        # <学习注释：按具体窗函数签名拼出长度、额外参数和对称标志组成的位置参数元组。>
        params = (Nx,) + args + (sym,)
    # <学习注释：float 转换成功时固定选择 Kaiser 窗，无需再按名称查表。>
    else:
        # <学习注释：数值形式的 window 被约定为 Kaiser beta。>
        winfunc = kaiser
        # <学习注释：Kaiser 窗需要长度、beta 与对称标志三个参数。>
        params = (Nx, beta, sym)

    # <学习注释：星号把参数元组展开为位置实参并返回生成的长度 Nx 窗数组。>
    return winfunc(*params)
