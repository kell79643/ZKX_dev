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

# <学习注释：从标准库 math 导入向上取整和对数；它们服务同文件其他 FIR 设计函数，firwin 本体不直接调用。>
from math import ceil, log

# <学习注释：CuPy 提供 GPU 数组、逐元素 kernel 和 GPU 路径下的数组运算。>
import cupy as cp
# <学习注释：NumPy 提供 CPU 路径的数组、sinc、余弦与求和运算。>
import numpy as np
# <学习注释：SciPy signal 在 gpupath=False 时提供 CPU 窗函数生成。>
from scipy import signal

# <学习注释：导入 cuSignal 自己的窗口分发器，GPU 路径通过它生成对称设计窗。>
from ..windows.windows import get_window


# <学习注释：这个内部 helper 统一新参数 fs 与已弃用参数 nyq，并返回采样频率。>
def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    # <学习注释：两个频率参数都未提供时采用默认采样频率 2，因此默认 Nyquist 频率为 1。>
    if nyq is None and fs is None:
        fs = 2
    # <学习注释：只要旧参数 nyq 被提供，就进入兼容分支。>
    elif nyq is not None:
        # <学习注释：新旧参数同时提供会产生歧义，因此立即拒绝。>
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        # <学习注释：Nyquist 频率等于采样频率的一半，所以反算 fs 为二倍 nyq。>
        fs = 2 * nyq
    # <学习注释：所有合法路径最终都返回统一语义的采样频率 fs。>
    return fs


# Some notes on function parameters:
#
# `cutoff` and `width` are given as numbers between 0 and 1.  These are
# relative frequencies, expressed as a fraction of the Nyquist frequency.
# For example, if the Nyquist frequency is 2 KHz, then width=0.15 is a width
# of 300 Hz.
#
# The `order` of a FIR filter is one less than the number of taps.
# This is a potential source of confusion, so in the following code,
# we will always use the number of taps as the parameterization of
# the 'size' of the filter. The "number of taps" means the number
# of coefficients, which is the same as the length of the impulse
# response of the filter.


# <学习注释：把正的衰减指标 a 映射成 Kaiser 窗形状参数 beta。>
def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.
    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.
    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.
    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
    """
    # <学习注释：高于 50 dB 时使用 Kaiser 经验公式的高衰减分支。>
    if a > 50:
        # <学习注释：线性经验式给出较大衰减所需的 beta。>
        beta = 0.1102 * (a - 8.7)
    # <学习注释：21 dB 到 50 dB 之间使用幂函数与线性项组合的经验分支。>
    elif a > 21:
        # <学习注释：0.4 次幂项和线性项共同计算中等衰减对应的 beta。>
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    # <学习注释：不超过 21 dB 时无需渐消，退化到 beta=0 的矩形窗。>
    else:
        # <学习注释：beta=0 是 Kaiser 窗的矩形窗特例。>
        beta = 0.0
    # <学习注释：返回供 get_window(('kaiser', beta), ...) 使用的形状参数。>
    return beta


# <学习注释：依据抽头数和归一化过渡宽度估算 Kaiser FIR 的衰减指标。>
def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.
    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:
        a = 2.285 * (N - 1) * pi * width + 7.95
    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter.
    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.
    """
    # <学习注释：经验式中 numtaps-1 是阶数，pi*width 把 Nyquist 归一化宽度换成 rad/sample。>
    a = 2.285 * (numtaps - 1) * np.pi * width + 7.95
    # <学习注释：返回正的衰减估计值，单位为 dB。>
    return a


# <学习注释：定义 GPU 逐元素 kernel；每个 GPU 元素对应一个 FIR 抽头。>
_firwin_kernel = cp.ElementwiseKernel(
    # <学习注释：输入依次为当前窗值、抽头数、通带边界原始数组、通带数和是否缩放。>
    "float64 win, int32 numtaps, raw float64 bands, int32 steps, bool scale",
    # <学习注释：输出 h 是未归一化抽头，hc 是缩放求和所需的 h*cos 项。>
    "float64 h, float64 hc",
    """
    // <学习注释：m 是当前抽头相对对称中心 alpha 的坐标；中心点用 1e-20 近似可去奇点。>
    const double m { static_cast<double>( i ) - alpha ?
        static_cast<double>( i ) - alpha : 1.0e-20 };

    // <学习注释：temp 累积多个通带的理想冲激响应贡献。>
    double temp {};
    // <学习注释：left 与 right 保存当前通带的归一化左右边界。>
    double left {};
    double right {};

    // <学习注释：遍历每一对通带边界并线性叠加矩形频响的逆 DTFT。>
    for ( int s = 0; s < steps; s++ ) {
        // <学习注释：从扁平 raw 数组取左右边界；零端点用极小正数近似。>
        left = bands[s * 2 + 0] ? bands[s * 2 + 0] : 1.0e-20;
        right = bands[s * 2 + 1] ? bands[s * 2 + 1] : 1.0e-20;

        // <学习注释：加入 right*sinc(right*m)，对应通带上边界积分项。>
        temp += right * ( sin( right * m * M_PI ) / ( right * m * M_PI ) );
        // <学习注释：减去 left*sinc(left*m)，完成区间 [left,right] 的 sinc 差。>
        temp -= left * ( sin( left * m * M_PI ) / ( left * m * M_PI ) );
    }

    // <学习注释：时域乘窗，把无限长理想响应截成指定长度并控制旁瓣。>
    temp *= win;
    // <学习注释：把当前线程算出的窗后系数写入 h。>
    h = temp;

    // <学习注释：预声明第一通带内用于单位增益归一化的参考频率。>
    double scale_frequency {};

    // <学习注释：仅当 scale=True 时计算当前抽头对归一化分母的贡献。>
    if ( scale ) {
        // <学习注释：归一化始终以第一通带为准，因此读取 bands 第一行。>
        left = bands[0];
        right = bands[1];

        // <学习注释：第一通带从 DC 开始时选择归一化频率 0。>
        if ( left == 0 ) {
            scale_frequency = 0.0;
        // <学习注释：第一通带到 Nyquist 结束时选择归一化频率 1。>
        } else if ( right == 1 ) {
            scale_frequency = 1.0;
        // <学习注释：普通带通选择第一通带中心作为代表频率。>
        } else {
            scale_frequency = 0.5 * ( left + right );
        }
        // <学习注释：对称 FIR 的零相位幅度可用 cos(pi*m*f) 加权求和。>
        double c { cos( M_PI * m * scale_frequency ) };
        // <学习注释：hc 保存当前抽头对尺度因子 s 的贡献，随后由 CuPy 全局求和。>
        hc = temp * c;
    }
    """,
    # <学习注释：为编译后的 CuPy kernel 指定可读名称。>
    "_firwin_kernel",
    # <学习注释：要求设备代码按 C++11 语法编译。>
    options=("-std=c++11",),
    # <学习注释：loop_prep 在逐元素循环前计算共享的对称中心 alpha。>
    loop_prep="const double alpha { 0.5 * ( numtaps - 1 ) };",
)


# <学习注释：公开的 firwin 接口按窗函数法生成 Type I 或 Type II 线性相位 FIR 系数。>
def firwin(
    # <学习注释：numtaps 是输出系数个数，滤波器阶数为 numtaps-1。>
    numtaps,
    # <学习注释：cutoff 是一个或多个通带/阻带交界频率。>
    cutoff,
    # <学习注释：width 非空时触发 Kaiser 经验参数路径。>
    width=None,
    # <学习注释：window 指定设计窗，默认使用 Hamming 窗。>
    window="hamming",
    # <学习注释：pass_zero 决定包含 DC 的第一个频段是通带还是阻带。>
    pass_zero=True,
    # <学习注释：scale 控制是否在第一通带代表频率处归一化到单位增益。>
    scale=True,
    # <学习注释：nyq 是已弃用的旧 Nyquist 频率参数。>
    nyq=None,
    # <学习注释：fs 是采样频率；默认归一化取 2。>
    fs=None,
    # <学习注释：gpupath=True 使用 CuPy kernel，False 使用 NumPy/SciPy CPU 参考路径。>
    gpupath=True,
):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`.  The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `cusignal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'},
        optional
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).

        .. versionadded:: 1.3.0
           Support for string arguments.
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise

    nyq : float, optional
        *Deprecated.  Use `fs` instead.*  This is the Nyquist frequency.
        Each frequency in `cutoff` must be between 0 and `nyq`. Default
        is 1.
    fs : float, optional
        The sampling frequency of the signal.  Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.
    gpupath : bool, Optional
        Optional path for filter design. gpupath == False may be desirable if
        filter sizes are small.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to ``fs/2``, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See Also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> import cusignal
    >>> numtaps = 3
    >>> f = 0.1
    >>> cusignal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> cusignal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> cusignal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> cusignal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> cusignal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """
    # <学习注释：根据 gpupath 选择后续参数检查和数组拼接使用的数组模块别名 pp。>
    if gpupath:
        # <学习注释：GPU 路径令 pp 指向 CuPy。>
        pp = cp
    else:
        # <学习注释：CPU 路径令 pp 指向 NumPy。>
        pp = np

    # <学习注释：_get_fs 返回采样频率，再乘 0.5 得到 Nyquist 频率。>
    nyq = 0.5 * _get_fs(fs, nyq)

    # <学习注释：把标量或序列至少转成一维数组，并除以 Nyquist 得到 (0,1) 归一化边界。>
    cutoff = pp.atleast_1d(cutoff) / float(nyq)

    # print("cutoff", cutoff.size)

    # Check for invalid input.
    # <学习注释：以下检查确保 cutoff 能唯一、合法地表示按频率递增的交替频带。>
    if cutoff.ndim > 1:
        raise ValueError("The cutoff argument must be at most " "one-dimensional.")
    # <学习注释：至少需要一个截止频率才能定义频带切换。>
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    # <学习注释：显式 cutoff 只能严格位于 DC 与 Nyquist 之间，端点由 pass_zero 逻辑补入。>
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError(
            "Invalid cutoff frequency: frequencies must be "
            "greater than 0 and less than nyq."
        )
    # <学习注释：相邻差分必须全部为正，保证 cutoff 严格递增且无重复。>
    if pp.any(pp.diff(cutoff) <= 0):
        raise ValueError(
            "Invalid cutoff frequencies: the frequencies "
            "must be strictly increasing."
        )

    # <学习注释：给定 width 时不再使用用户传入的 window，而是自动构造 Kaiser 窗参数。>
    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        # <学习注释：width/nyq 把物理过渡宽度归一化到 Nyquist=1。>
        atten = kaiser_atten(numtaps, float(width) / nyq)
        # <学习注释：由估计衰减 atten 计算 Kaiser 形状参数 beta。>
        beta = kaiser_beta(atten)
        # <学习注释：用 get_window 能识别的元组形式覆盖 window。>
        window = ("kaiser", beta)

    # <学习注释：字符串形式提供与 IIR btype 类似的可读滤波器类型。>
    if isinstance(pass_zero, str):
        # <学习注释：bandstop 和 lowpass 都意味着 DC 位于通带。>
        if pass_zero in ("bandstop", "lowpass"):
            # <学习注释：lowpass 只能有一个截止边界。>
            if pass_zero == "lowpass":
                if cutoff.size != 1:
                    raise ValueError(
                        "cutoff must have one element if "
                        'pass_zero=="lowpass", got %s' % (cutoff.shape,)
                    )
            # <学习注释：bandstop 至少需要两个边界才能围成一个阻带。>
            elif cutoff.size <= 1:
                raise ValueError(
                    "cutoff must have at least two elements if "
                    'pass_zero=="bandstop", got %s' % (cutoff.shape,)
                )
            # <学习注释：把可读字符串统一归约为后续布尔逻辑 True。>
            pass_zero = True
        # <学习注释：bandpass 和 highpass 都意味着 DC 位于阻带。>
        elif pass_zero in ("bandpass", "highpass"):
            # <学习注释：highpass 只能有一个截止边界。>
            if pass_zero == "highpass":
                if cutoff.size != 1:
                    raise ValueError(
                        "cutoff must have one element if "
                        'pass_zero=="highpass", got %s' % (cutoff.shape,)
                    )
            # <学习注释：bandpass 至少需要两个边界才能围成一个通带。>
            elif cutoff.size <= 1:
                raise ValueError(
                    "cutoff must have at least two elements if "
                    'pass_zero=="bandpass", got %s' % (cutoff.shape,)
                )
            # <学习注释：把可读字符串统一归约为后续布尔逻辑 False。>
            pass_zero = False
        # <学习注释：其余字符串不属于支持的六种表达。>
        else:
            raise ValueError(
                'pass_zero must be True, False, "bandpass", '
                '"lowpass", "highpass", or "bandstop", got '
                "{}".format(pass_zero)
            )

    # <学习注释：每遇到一个 cutoff，通带/阻带状态翻转；异或据此判断 Nyquist 是否在通带。>
    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero

    # <学习注释：Type II 即偶数 taps 在 Nyquist 必为零，不能让 Nyquist 落在通带。>
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError(
            "A filter with an even number of coefficients must "
            "have zero response at the Nyquist rate."
        )

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    # <学习注释：若 DC 或 Nyquist 属于通带，就补入端点 0 或 1，使数组可两两成对。>
    cutoff = pp.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    # <学习注释：把一维边界按每两个元素一组改形为 Q×2 的通带矩阵。>
    bands = cutoff.reshape(-1, 2)

    # <学习注释：GPU 路径由 cuSignal 生成 GPU 窗并启动逐元素 kernel。>
    if gpupath:
        # <学习注释：fftbins=False 请求关于中心对称的滤波器设计窗。>
        win = get_window(window, numtaps, fftbins=False)
        # <学习注释：kernel 同时返回窗后系数 h 和可选的缩放贡献 hc。>
        h, hc = _firwin_kernel(win, numtaps, bands, bands.shape[0], scale)
        # <学习注释：需要单位增益时，对所有抽头的 hc 求和得到尺度因子。>
        if scale:
            # <学习注释：GPU 归约求和产生参考频率的零相位响应 s。>
            s = cp.sum(hc)
            # <学习注释：所有系数同除以 s，使参考频率响应为 1。>
            h /= s
    # <学习注释：CPU 路径使用 SciPy 生成窗、NumPy 构造系数。>
    else:
        try:
            # <学习注释：同样用 fftbins=False 请求对称设计窗。>
            win = signal.get_window(window, numtaps, fftbins=False)
        except NameError:
            raise RuntimeError("CPU path requires SciPy Signal's get_windows.")

        # Build up the coefficients.
        # <学习注释：alpha 是长度 N 冲激响应的对称中心。>
        alpha = 0.5 * (numtaps - 1)
        # <学习注释：m 为每个抽头相对中心的位置，奇数 taps 为整数，偶数 taps 为半整数。>
        m = np.arange(0, numtaps) - alpha
        # <学习注释：h 从标量零开始，与数组相加时广播成长度 numtaps 的数组。>
        h = 0
        # <学习注释：逐个遍历 bands 的 [left,right] 通带边界对。>
        for left, right in bands:
            # <学习注释：加入通带右边界对应的 right*sinc(right*m)。>
            h += right * np.sinc(right * m)
            # <学习注释：减去左边界项，得到该矩形通带的逆 DTFT。>
            h -= left * np.sinc(left * m)

        # <学习注释：逐点乘对称窗，形成有限长线性相位 FIR 系数。>
        h *= win

        # Now handle scaling if desired.
        # <学习注释：只有 scale=True 时才执行参考频率单位增益归一化。>
        if scale:
            # Get the first passband.
            # <学习注释：取第一行通带边界用于选择参考频率。>
            left, right = bands[0]
            # <学习注释：第一通带从 DC 起始时选择 0。>
            if left == 0:
                scale_frequency = 0.0
            # <学习注释：第一通带到 Nyquist 结束时选择 1。>
            elif right == 1:
                scale_frequency = 1.0
            # <学习注释：其他情况选择第一通带中心。>
            else:
                scale_frequency = 0.5 * (left + right)
            # <学习注释：计算每个抽头在参考频率处的余弦基函数值。>
            c = np.cos(np.pi * m * scale_frequency)
            # <学习注释：利用对称性，h*c 的总和就是去掉线性相位后的实幅度。>
            s = np.sum(h * c)
            # <学习注释：统一缩放所有抽头，使该幅度精确为 1。>
            h /= s

    # <学习注释：返回 CPU NumPy 数组或 GPU CuPy 数组形式的一维 FIR 系数。>
    return h


# <学习注释：firwin2 从频率增益控制点设计具有对称或反对称线性相位的 FIR 系数。>
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

    # <学习注释：gpupath 决定数组运算使用 CuPy 设备后端还是 NumPy 主机后端。>
    if gpupath:
        # <学习注释：pp 是统一后端别名，GPU 路径把它绑定为 cupy。>
        pp = cp
    else:
        # <学习注释：CPU 辅助路径把同一别名绑定为 numpy，以复用后续算法代码。>
        pp = np

    # <学习注释：先统一 fs 与 nyq，再乘二分之一得到本函数使用的 Nyquist 频率上界。>
    nyq = 0.5 * _get_fs(fs, nyq)

    # <学习注释：每个频率控制点必须恰好对应一个目标增益。>
    if len(freq) != len(gain):
        raise ValueError("freq and gain must be of same length.")

    # <学习注释：显式网格长度必须严格大于最终抽头数，否则无法按本算法截取所需系数。>
    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(
            (
                "ntaps must be less than nfreqs, but firwin2 was "
                "called with ntaps=%d and nfreqs=%s"
            )
            % (numtaps, nfreqs)
        )

    # <学习注释：控制点必须完整覆盖从直流零频到 Nyquist 频率的单边频谱。>
    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError("freq must start with 0 and end with fs/2.")
    # <学习注释：相邻频率差 d 用于检查顺序、重复次数并识别跳变控制点。>
    d = pp.diff(freq)
    # <学习注释：任何负差值都说明 freq 不是非递减序列。>
    if (d < 0).any():
        raise ValueError("The values in freq must be nondecreasing.")
    # <学习注释：相邻两个差值之和为零意味着至少三个连续控制点取同一频率。>
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError("A value in freq must not occur more than twice.")
    # <学习注释：直流端点不能重复，因为端点没有可定义的左右两侧。>
    if freq[1] == 0:
        raise ValueError("Value 0 must not be repeated in freq")
    # <学习注释：Nyquist 端点同样禁止重复。>
    if freq[-2] == nyq:
        raise ValueError("Value fs/2 must not be repeated in freq")

    # <学习注释：长度奇偶与 antisymmetric 标志共同确定四类广义线性相位 FIR。>
    if antisymmetric:
        if numtaps % 2 == 0:
            # <学习注释：偶数抽头且反对称对应 IV 型。>
            ftype = 4
        else:
            # <学习注释：奇数抽头且反对称对应 III 型。>
            ftype = 3
    else:
        if numtaps % 2 == 0:
            # <学习注释：偶数抽头且对称对应 II 型。>
            ftype = 2
        else:
            # <学习注释：奇数抽头且对称对应 I 型。>
            ftype = 1

    # <学习注释：II 型的对称性强制 Nyquist 响应为零。>
    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError(
            "A Type II filter must have zero gain at the " "Nyquist frequency."
        )
    # <学习注释：III 型反对称奇长响应同时强制直流与 Nyquist 响应为零。>
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError(
            "A Type III filter must have zero gain at zero " "and Nyquist frequencies."
        )
    # <学习注释：IV 型反对称偶长响应强制直流响应为零。>
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero " "frequency.")

    # <学习注释：未指定网格长度时，选择不小于 numtaps 的最小二次幂再加一，以适配高效实数 FFT。>
    if nfreqs is None:
        # <学习注释：ceil 的结果是二次幂指数，前面的 1 对应单边实频谱包含的 Nyquist 端点。>
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))

    # <学习注释：相邻频率相等表示用户用两个不同增益描述了一个内部不连续跳变。>
    if (d == 0).any():
        # Tweak any repeated values in freq so that interp works.
        # <学习注释：先复制频率数组，避免为数值插值调整控制点时修改调用者的输入。>
        freq = pp.array(freq, copy=True)
        # <学习注释：扰动尺度取双精度机器 epsilon 与 Nyquist 频率的乘积。>
        eps = pp.finfo(float).eps * nyq
        # <学习注释：逐对检查相邻控制点，循环上界减一是为了安全访问 k 加一。>
        for k in range(len(freq) - 1):
            # <学习注释：发现重复横坐标时，将左点和右点分别向两侧移动一个 eps。>
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # Check if freq is strictly increasing after tweak
        # <学习注释：扰动后重新计算差分，验证插值横坐标已经严格递增。>
        d = pp.diff(freq)
        # <学习注释：若仍有非正差值，说明原控制点与重复点距离小到无法安全扰动。>
        if (d <= 0).any():
            raise ValueError(
                "freq cannot contain numbers that are too close "
                "(within eps * (fs/2): "
                "{}) to a repeated value".format(eps)
            )

    # Linearly interpolate the desired response on a uniform mesh `x`.
    # <学习注释：在零频到 Nyquist 之间创建包含两端点的 nfreqs 点均匀网格。>
    x = pp.linspace(0.0, nyq, nfreqs)
    # <学习注释：GPU 路径暂时把网格移到主机，因为这里调用的是 NumPy 的一维线性插值。>
    if gpupath:
        # <学习注释：先在 CPU 上对控制点做线性插值，再把期望增益网格复制回 GPU。>
        fx = cp.asarray(np.interp(cp.asnumpy(x), freq, gain))
    else:
        # <学习注释：CPU 路径直接用 numpy.interp 得到均匀网格上的期望增益。>
        fx = np.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    # <学习注释：该复指数加入群时延为二分之 numtaps 减一的线性相位，且 pi 乘 x 除 nyq 把物理频率归一化为数字角频率。>
    shift = pp.exp(-(numtaps - 1) / 2.0 * 1.0j * pp.pi * x / nyq)
    # <学习注释：III 与 IV 型还需乘以虚数单位，形成反对称 FIR 的恒定四分之一周期相位因子。>
    if ftype > 2:
        shift *= 1j

    # <学习注释：把实值期望增益与类型对应的相位因子逐点相乘，得到单边复频谱。>
    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    # <学习注释：irfft 按 Hermitian 共轭对称补齐负频率，并返回实值周期冲激响应。>
    out_full = pp.fft.irfft(fx2)

    # Pass to device memory
    # <学习注释：即使选择 NumPy 计算路径，函数的公开返回值仍统一转换为 CuPy 设备数组。>
    if not gpupath:
        # <学习注释：这一数据搬运把主机端逆 FFT 结果复制到设备内存。>
        out_full = cp.asarray(out_full)

    # <学习注释：非空 window 请求通过 get_window 创建长度 numtaps 的滤波器设计用对称窗。>
    if window is not None:
        # Create the window to apply to the filter coefficients.
        # <学习注释：fftbins 为 False 明确要求对称窗，从而保持最终 FIR 的对称或反对称结构。>
        wind = get_window(window, numtaps, fftbins=False)
    else:
        # <学习注释：window 为 None 时用标量一广播，相当于不施加额外窗。>
        wind = 1

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    # <学习注释：截取周期冲激响应的前 numtaps 项并逐点乘窗，得到最终 FIR 系数。>
    out = out_full[:numtaps] * wind

    # <学习注释：III 型是奇长反对称序列，中心抽头理论上必须严格等于零。>
    if ftype == 3:
        # <学习注释：显式清零中心抽头，消除逆 FFT 舍入可能留下的微小数值残差。>
        out[out.size // 2] = 0.0

    # <学习注释：返回长度为 numtaps 的 CuPy 一维滤波器系数数组。>
    return out


def cmplx_sort(p):
    """Sort roots based on magnitude.

    Parameters
    ----------
    p : array_like
        The roots to sort, as a 1-D array.

    Returns
    -------
    p_sorted : ndarray
        Sorted roots.
    indx : ndarray
        Array of indices needed to sort the input `p`.

    Examples
    --------
    >>> import cusignal
    >>> vals = [1, 4, 1+1.j, 3]
    >>> p_sorted, indx = cusignal.cmplx_sort(vals)
    >>> p_sorted
    array([1.+0.j, 1.+1.j, 3.+0.j, 4.+0.j])
    >>> indx
    array([0, 2, 3, 1])

    """
    p = cp.asarray(p)
    if cp.iscomplexobj(p):
        indx = cp.argsort(abs(p))
    else:
        indx = cp.argsort(p)
    return cp.take(p, indx, 0), indx
