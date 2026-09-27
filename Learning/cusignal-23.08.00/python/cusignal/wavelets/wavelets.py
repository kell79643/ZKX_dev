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

# <学习注释：导入 CuPy 并使用别名 cp；qmf 依靠 cp.ElementwiseKernel 在 GPU 上逐元素生成输出。>
# <学习注释：导入 CuPy 并命名为 cp；ricker 随后通过 cp.ElementwiseKernel 定义和启动 GPU 逐元素 kernel。>
import cupy as cp
import numpy as np

from ..convolution.convolve import convolve

# <学习注释：创建 qmf 的逐元素 GPU kernel；前导下划线表示它是模块内部实现。>
_qmf_kernel = cp.ElementwiseKernel(
    # <学习注释：输入参数声明为空字符串，说明 kernel 不会接收 hk 的任何系数值。>
    "",
    # <学习注释：每个输出元素被固定声明为 int64，因此输出 dtype 不随 hk 的 dtype 变化。>
    "int64 output",
    # <学习注释：三引号字符串中的内容是每个 GPU 元素执行的 C++/CUDA 操作体。>
    """
    // <学习注释：i 是 CuPy ElementwiseKernel 提供的当前元素索引；按最低位判断奇偶并生成 +1 或 -1。>
    const int sign { ( i & 1 ) ? -1 : 1 };
    // <学习注释：_ind.size() 是输出长度 L；该式计算 (-1)^i*(L-1-i)，使用的是索引而非 hk[L-1-i]。>
    output = ( _ind.size() - ( i + 1 ) ) * sign;
    // <学习注释：结束 kernel 操作体字符串；上面两条语句构成每个输出元素的全部计算。>
    """,
    # <学习注释：把该 kernel 命名为 _qmf_kernel，供 CuPy 编译缓存和诊断信息识别。>
    "_qmf_kernel",
    # <学习注释：要求以 C++11 语法编译操作体，花括号初始化依赖这一语言标准。>
    options=("-std=c++11",),
    # <学习注释：结束 cp.ElementwiseKernel 构造调用，并把返回的可调用 kernel 保存到模块变量。>
)


# <学习注释：定义公开 qmf(hk) 接口；按文档意图应由低通系数 hk 生成高通 QMF 系数。>
def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    # <学习注释：只把 hk 的长度作为输出规模启动 kernel；没有把 hk 本身传入，这是当前实现偏离 qmf 公式的根因。>
    return _qmf_kernel(size=len(hk))


_morlet_kernel = cp.ElementwiseKernel(
    "float64 w, float64 s, bool complete",
    "complex128 output",
    """
    const double x { start + delta * i };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    if ( complete ) {
        temp -= exp( -0.5 * ( w * w ) );
    }

    output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="const double end { s * 2.0 * M_PI }; \
               const double start { -s * 2.0 * M_PI }; \
               const double delta { ( end - start ) / ( _ind.size() - 1 ) };",
)


def morlet(M, w=5.0, s=1.0, complete=True):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    cusignal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    """
    return _morlet_kernel(w, s, complete, size=M)


# <学习注释：创建 ricker 专用的 CuPy 逐元素 kernel；每个 GPU 线程负责一个输出样本。>
_ricker_kernel = cp.ElementwiseKernel(
    # <学习注释：声明一个名为 a 的 float64 标量输入，它是以样本为单位的宽度参数。>
    "float64 a",
    # <学习注释：声明一个名为 total 的 float64 输出；CuPy 将为每个元素写入一个 double 值。>
    "float64 total",
    # <学习注释：下面的多行字符串是逐元素执行的 C++/CUDA 操作体。>
    # <学习注释：vec 用当前元素索引 i 减去 (输出长度-1)/2，从而让奇偶长度数组都关于零对称。>
    # <学习注释：xsq 保存 vec 的平方，mod 对应公式中的 1-x²/a²，gauss 对应 exp(-x²/(2a²))。>
    # <学习注释：最后把单位能量归一化系数 A、mod 与 gauss 相乘后写入当前输出 total。>
    """
    const double vec { i - ( _ind.size() - 1.0 ) * 0.5 };
    const double xsq { vec * vec };
    const double mod { 1 - xsq / wsq };
    const double gauss { exp( -xsq / ( 2.0 * wsq ) ) };

    total = A * mod * gauss;
    """,
    # <学习注释：为生成的 kernel 指定内部名称 _ricker_kernel，便于 CuPy 编译缓存和诊断。>
    "_ricker_kernel",
    # <学习注释：要求以 C++11 语法编译，因为 kernel 字符串使用了花括号初始化。>
    options=("-std=c++11",),
    # <学习注释：loop_prep 在逐元素循环前计算一次归一化系数 A，避免每个输出重复开方和幂运算。>
    # <学习注释：同一个 loop_prep 字符串还预计算 wsq=a*a，供所有元素计算 mod 和 gauss 时复用。>
    loop_prep="const double A { 2.0 / ( sqrt( 3 * a ) * pow( M_PI, 0.25 ) ) }; \
               const double wsq { a * a };",
)


# <学习注释：定义公开 Python API ricker；points 决定输出长度，a 决定小波宽度。>
def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,

    where ``A = 2/sqrt(3a)pi^1/4``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = cusignal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(cp.asnumpy(vec2))
    >>> plt.show()

    """
    # <学习注释：调用已定义的 GPU kernel，把 a 作为标量输入，并用 size=points 请求长度为 points 的一维输出。>
    return _ricker_kernel(a, size=points)


_morlet2_kernel = cp.ElementwiseKernel(
    "float64 w, float64 s",
    "complex128 output",
    """
    const double x { ( i - ( _ind.size() - 1.0 ) * 0.5 ) / s };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    output = sqrt( 1 / s ) * temp * exp( -0.5 * ( x * x ) ) *
        pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="",
)


def morlet2(M, s, w=5):
    """
    Complex Morlet wavelet, designed to work with `cwt`.
    Returns the complete version of morlet wavelet, normalised
    according to `s`::
        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)
    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter of the wavelet.
    w : float, optional
        Omega0. Default is 5
    Returns
    -------
    morlet : (M,) ndarray
    See Also
    --------
    morlet : Implementation of Morlet wavelet, incompatible with `cwt`
    Notes
    -----
    .. versionadded:: 1.4.0
    This function was designed to work with `cwt`. Because `morlet2`
    returns an array of complex numbers, the `dtype` argument of `cwt`
    should be set to `complex128` for best results.
    Note the difference in implementation with `morlet`.
    The fundamental frequency of this wavelet in Hz is given by::
        f = w*fs / (2*s*np.pi)
    where ``fs`` is the sampling rate and `s` is the wavelet width parameter.
    Similarly we can get the wavelet width parameter at ``f``::
        s = w*fs / (2*f*np.pi)
    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> M = 100
    >>> s = 4.0
    >>> w = 2.0
    >>> wavelet = signal.morlet2(M, s, w)
    >>> plt.plot(abs(wavelet))
    >>> plt.show()
    This example shows basic use of `morlet2` with `cwt` in time-frequency
    analysis:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t, dt = np.linspace(0, 1, 200, retstep=True)
    >>> fs = 1/dt
    >>> w = 6.
    >>> sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
    >>> freq = np.linspace(1, fs/2, 100)
    >>> widths = w*fs / (2*freq*np.pi)
    >>> cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
    >>> plt.pcolormesh(t, freq, np.abs(cwtm),
        cmap='viridis', shading='gouraud')
    >>> plt.show()
    """

    return _morlet2_kernel(w, s, size=M)


# <学习注释：cwt 接收一维数据、生成小波的 callable，以及需要计算的宽度序列。>
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
    # <学习注释：先用最小参数调用一次 wavelet，并转成 CuPy 数组以探测其返回 dtype。>
    if cp.asarray(wavelet(1, 1)).dtype.char in "FDG":
        # <学习注释：dtype 字符 F、D、G 分别代表不同精度的复数类型；复小波统一输出 complex128。>
        dtype = cp.complex128
    else:
        # <学习注释：非复数小波统一使用 float64 输出，不沿用输入 data 的 dtype。>
        dtype = cp.float64

    # <学习注释：输出每个 width 占一行、每个输入位置占一列，shape 为 (len(widths), len(data))。>
    output = cp.empty([len(widths), len(data)], dtype=dtype)

    # <学习注释：依次处理用户给出的每个 width；Python enumerate 同时产生行索引 ind 和当前 width。>
    for ind, width in enumerate(widths):
        # <学习注释：把 width 截断为整数，并把有限小波长度限制为 10 倍宽度且不超过输入长度。>
        N = np.min([10 * int(width), len(data)])
        # <学习注释：生成长度 N 的小波，取复共轭后反转，使后续卷积等价于信号与小波的相关/内积。>
        wavelet_data = cp.conj(wavelet(N, int(width)))[::-1]
        # <学习注释：same 线性卷积保留与 data 同长的居中结果，并写入当前 width 对应的输出行。>
        output[ind, :] = convolve(data, wavelet_data, mode="same")
    # <学习注释：返回二维尺度（width）—位置系数矩阵。>
    return output
