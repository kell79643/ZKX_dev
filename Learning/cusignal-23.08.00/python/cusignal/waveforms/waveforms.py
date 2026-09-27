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

# <学习注释：导入 CuPy 并命名为 cp；unit_impulse 使用它的 ElementwiseKernel 在 GPU 上逐元素生成输出。>
import cupy as cp
# <学习注释：导入 NumPy 并命名为 np；unit_impulse 只用 np.atleast_1d 规范化 shape，不在 NumPy 中生成最终数组。>
import numpy as np

_sawtooth_kernel = cp.ElementwiseKernel(
    "T t, T w",
    "float64 y",
    """
    double out {};
    const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };
    if ( mask1 ) {
        out = nan("0xfff8000000000000ULL");
    }

    const T tmod { fmod( t, 2.0 * M_PI ) };
    const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };

    if ( mask2 ) {
        out = tmod / ( M_PI * w ) - 1;
    }

    const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };
    if ( mask3 ) {
        out = ( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) );
    }
    y = out;
    """,
    "_sawtooth_kernel",
    options=("-std=c++11",),
)


def sawtooth(t, width=1.0):
    """
    Return a periodic sawtooth or triangle waveform.
    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].
    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.
    Parameters
    ----------
    t : array_like
        Time.
    width : array_like, optional
        Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.
    Returns
    -------
    y : ndarray
        Output array containing the sawtooth waveform.
    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(0, 1, 500)
    >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))
    """
    t, w = cp.asarray(t), cp.asarray(width)

    y = _sawtooth_kernel(t, w)

    return y


# <学习注释：创建 CuPy 逐元素 GPU kernel；每个输出元素只依赖同位置的相位 t 和占空比 w。>
_square_kernel = cp.ElementwiseKernel(
    # <学习注释：输入参数使用同一个模板类型 T；t 是相位元素，w 是 duty 在 kernel 中的简称。>
    "T t, T w",
    # <学习注释：无论输入模板类型 T 是什么，kernel 都把输出 y 固定声明为 float64。>
    "float64 y",
    """
    // <学习注释：若当前元素的占空比 w 不在闭区间 [0,1]，mask1 为 true。>
    const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };
    // <学习注释：非法占空比不抛出异常，而是把当前输出元素写成 NaN。>
    if ( mask1 ) {
        y = nan("0xfff8000000000000ULL");
    }

    // <学习注释：用 fmod 计算 t 除以 2π 的余数；对非负 t，它给出当前 2π 周期内的相位。>
    // <学习注释：fmod 对负 t 会保留负号，并不等同于把相位正规化到 [0,2π) 的数学 modulo。>
    const T tmod { fmod( t, 2.0 * M_PI ) };
    // <学习注释：仅当占空比合法且周期内相位严格小于 2πw 时，mask2 才表示当前处于高电平区间。>
    const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };

    // <学习注释：高电平区间输出 +1；严格小于号意味着恰好位于下降沿时不会进入此分支。>
    if ( mask2 ) {
        y = 1;
    }

    // <学习注释：占空比合法但不在高电平区间时，mask3 表示当前应输出低电平。>
    const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };
    // <学习注释：低电平区间输出 -1。>
    if ( mask3 ) {
        y = -1;
    }

    """,
    "_square_kernel",
    options=("-std=c++11",),
)


def square(t, duty=0.5):
    # <学习注释：中文翻译——本函数返回一个周期性方波。>
    # <学习注释：中文翻译——方波的相位周期为 2π；从 0 到 2π×duty 取 +1，从 2π×duty 到 2π 取 -1。>
    # <学习注释：中文翻译——duty 必须位于闭区间 [0,1]。>
    # <学习注释：中文翻译——该理想方波不是带限信号，会产生无限多个谐波，采样时高次谐波会折叠并造成混叠。>
    # <学习注释：参数翻译——t 是类数组输入；虽然原文称其为时间数组，但函数实际按无量纲相位处理它。>
    # <学习注释：参数翻译——duty 是可选的类数组占空比，默认 0.5，即每周期有 50% 的相位区间输出高电平。>
    # <学习注释：参数翻译——duty 若为数组，就允许占空比逐元素变化；原文要求其长度与 t 相同。>
    # <学习注释：返回值翻译——y 是包含方波样本的 ndarray。>
    # <学习注释：示例一标题翻译——生成持续 1 秒、以 500 Hz 采样的 5 Hz 方波。>
    # <学习注释：示例一第 1 行——import cusignal 导入 cuSignal 公共包，以便调用 cusignal.square。>
    # <学习注释：示例一第 2 行——import cupy as cp 导入 CuPy 并使用短名称 cp；CuPy 数组驻留在 GPU 侧。>
    # <学习注释：示例一第 3 行——导入 matplotlib.pyplot 并命名为 plt，用于绘制结果。>
    # <学习注释：示例一第 4 行——cp.linspace(0,1,500,endpoint=False) 在 [0,1) 秒内均匀生成 500 个时间样本。>
    # <学习注释：示例一第 4 行——相邻样本间隔为 1/500 秒，因此采样率是 500 Hz；endpoint=False 避免把下一秒的周期起点重复取样。>
    # <学习注释：示例一第 5 行——2π×5×t 把秒时间转换成 5 Hz 信号的相位；一秒内相位增加 10π，正好经历 5 个周期。>
    # <学习注释：示例一第 5 行——cusignal.square 根据相位生成 ±1 方波；未显式传 duty，所以使用默认占空比 0.5。>
    # <学习注释：示例一第 5 行——cp.asnumpy 把绘图所需的 CuPy 时间数组和方波数组复制成 NumPy 数组，再交给 plt.plot。>
    # <学习注释：示例一第 6 行——plt.ylim(-2,2) 把纵轴范围固定为 [-2,2]，便于看清取值为 ±1 的方波。>
    # <学习注释：示例二标题翻译——生成一个以正弦信号控制脉宽的 PWM 波形。>
    # <学习注释：示例二第 1 行——plt.figure() 新建一张图，避免覆盖前一个 5 Hz 方波图。>
    # <学习注释：示例二第 2 行——sig=sin(2πt) 生成 1 Hz、取值范围为 [-1,1] 的正弦控制信号。>
    # <学习注释：示例二第 3 行——2π×30×t 产生 30 Hz 载波相位，载波周期为 1/30 秒。>
    # <学习注释：示例二第 3 行——(sig+1)/2 把正弦值从 [-1,1] 线性映射到合法占空比 [0,1]。>
    # <学习注释：示例二第 3 行——sig=-1、0、+1 时，duty 分别为 0、0.5、1，因而高电平脉冲分别最窄、半周期宽和最宽。>
    # <学习注释：示例二第 3 行——square 逐样本使用变化的 duty 生成 pwm；pwm 本身仍是 ±1 脉冲，并不是平滑正弦波。>
    # <学习注释：示例二第 4 行——plt.subplot(2,1,1) 把画布分成 2 行 1 列，并选中第 1 个子图。>
    # <学习注释：示例二第 5 行——把 t 和 sig 从 CuPy 转成 NumPy 后，在上方子图绘制正弦控制信号。>
    # <学习注释：示例二第 6 行——plt.subplot(2,1,2) 选中下方第 2 个子图。>
    # <学习注释：示例二第 7 行——把 t 和 pwm 转成 NumPy 后，在下方子图绘制脉宽随 sig 变化的 PWM 波形。>
    # <学习注释：示例二第 8 行——把当前下方子图的纵轴设置为 [-1.5,1.5]，以突出 PWM 的 ±1 电平。>
    """
    Return a periodic square-wave waveform.

    The square wave has a period ``2*pi``, has value +1 from 0 to
    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in
    the interval [0,1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        The input time array.
    duty : array_like, optional
        Duty cycle.  Default is 0.5 (50% duty cycle).
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    Returns
    -------
    y : ndarray
        Output array containing the square waveform.

    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:

    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cp.linspace(0, 1, 500, endpoint=False)
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(cusignal.square(2 * cp.pi * 5 * t)))
    >>> plt.ylim(-2, 2)

    A pulse-width modulated sine wave:

    >>> plt.figure()
    >>> sig = cp.sin(2 * cp.pi * t)
    >>> pwm = cusignal.square(2 * cp.pi * 30 * t, duty=(sig + 1)/2)
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(sig))
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(pwm))
    >>> plt.ylim(-1.5, 1.5)

    """
    # <学习注释：把输入 t 和 duty 分别转换成 CuPy 数组；w 只是 duty 的内部短变量名。>
    # <学习注释：标量 duty 会由 ElementwiseKernel 按 CuPy 的逐元素广播语义用于所有 t 元素。>
    t, w = cp.asarray(t), cp.asarray(duty)

    # <学习注释：调用预先定义的 _square_kernel，让 GPU 对广播后的每一对 t、w 元素执行同一套阈值判断。>
    y = _square_kernel(t, w)

    # <学习注释：返回 kernel 生成的 float64 CuPy 数组 y。>
    return y


_gausspulse_kernel_F_F = cp.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI",
    """
    T yenv = exp(-a * t * t);
    yI = yenv * cos( 2 * M_PI * fc * t);
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)

_gausspulse_kernel_F_T = cp.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI, T yenv",
    """
    yenv = exp(-a * t * t);
    yI = yenv * cos( 2 * M_PI * fc * t);
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)

_gausspulse_kernel_T_F = cp.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI, T yQ",
    """
    T yenv { exp(-a * t * t) };

    T l_yI {};
    T l_yQ {};
    sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);
    yI = yenv * l_yI;
    yQ = yenv * l_yQ;
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)

_gausspulse_kernel_T_T = cp.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI, T yQ, T yenv",
    """
    yenv = exp(-a * t * t);

    T l_yI {};
    T l_yQ {};
    sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);
    yI = yenv * l_yI;
    yQ = yenv * l_yQ;
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)


def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False, retenv=False):
    """
    Return a Gaussian modulated sinusoid:

        ``exp(-a t^2) exp(1j*2*pi*fc*t).``

    If `retquad` is True, then return the real and imaginary parts
    (in-phase and quadrature).
    If `retenv` is True, then return the envelope (unmodulated signal).
    Otherwise, return the real part of the modulated sinusoid.

    Parameters
    ----------
    t : ndarray or the string 'cutoff'
        Input array.
    fc : int, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.
    tpr : float, optional
        If `t` is 'cutoff', then the function returns the cutoff
        time for when the pulse amplitude falls below `tpr` (in dB).
        Default is -60.
    retquad : bool, optional
        If True, return the quadrature (imaginary) as well as the real part
        of the signal.  Default is False.
    retenv : bool, optional
        If True, return the envelope of the signal.  Default is False.

    Returns
    -------
    yI : ndarray
        Real part of signal.  Always returned.
    yQ : ndarray
        Imaginary part of signal.  Only returned if `retquad` is True.
    yenv : ndarray
        Envelope of signal.  Only returned if `retenv` is True.

    See Also
    --------
    cusignal.morlet

    Examples
    --------
    Plot real component, imaginary component, and envelope for a 5 Hz pulse,
    sampled at 100 Hz for 2 seconds:

    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cp.linspace(-1, 1, 2 * 100, endpoint=False)
    >>> i, q, e = cusignal.gausspulse(t, fc=5, retquad=True, retenv=True)
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(i), cp.asnumpy(t), cp.asnumpy(q),
                 cp.asnumpy(t), cp.asnumpy(e), '--')

    """
    if fc < 0:
        raise ValueError("Center frequency (fc=%.2f) must be >=0." % fc)
    if bw <= 0:
        raise ValueError("Fractional bandwidth (bw=%.2f) must be > 0." % bw)
    if bwr >= 0:
        raise ValueError(
            "Reference level for bandwidth (bwr=%.2f) must " "be < 0 dB" % bwr
        )

    # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)

    ref = pow(10.0, bwr / 20.0)
    # fdel = fc*bw/2:  g(fdel) = ref --- solve this for a
    #
    # pi^2/a * fc^2 * bw^2 /4=-log(ref)
    a = -((np.pi * fc * bw) ** 2) / (4.0 * np.log(ref))

    if isinstance(t, str):
        if t == "cutoff":  # compute cut_off point
            #  Solve exp(-a tc**2) = tref  for tc
            #   tc = sqrt(-log(tref) / a) where tref = 10^(tpr/20)
            if tpr >= 0:
                raise ValueError("Reference level for time cutoff must " "be < 0 dB")
            tref = pow(10.0, tpr / 20.0)
            return np.sqrt(-np.log(tref) / a)
        else:
            raise ValueError("If `t` is a string, it must be 'cutoff'")

    t = cp.asarray(t)

    if not retquad and not retenv:
        return _gausspulse_kernel_F_F(t, a, fc)
    if not retquad and retenv:
        return _gausspulse_kernel_F_T(t, a, fc)
    if retquad and not retenv:
        return _gausspulse_kernel_T_F(t, a, fc)
    if retquad and retenv:
        return _gausspulse_kernel_T_T(t, a, fc)


# <学习注释：创建实值线性 chirp 的 CuPy 逐元素 kernel；每个 GPU 输出元素独立处理一个 t。>
_chirp_phase_lin_kernel_real = cp.ElementwiseKernel(
    # <学习注释：五个输入共享模板类型 T；t 是当前采样时刻，其余是标量扫频参数和弧度初相位。>
    "T t, T f0, T t1, T f1, T phi",
    # <学习注释：输出 phase 也使用模板类型 T；变量名虽叫 phase，实际保存的是 cos 后的实值波形。>
    "T phase",
    """
    // <学习注释：线性扫频斜率 beta=(f1-f0)/t1；若时间用秒，单位为 Hz/s。>
    const T beta { (f1 - f0) / t1 };
    // <学习注释：对 f(t)=f0+beta*t 从 0 积分，得到周期数 f0*t+beta*t^2/2，再乘 2π 得弧度相位。>
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    // <学习注释：此原注释与实际位置有偏差；phi 已在 Python 函数中转成弧度，这里只把它加到累计相位上。>
    // <学习注释：取 cos 得到单位幅度实值 chirp；当前元素的结果写入输出变量 phase。>
    phase = cos(temp + phi);
    """,
    # <学习注释：指定生成 kernel 的内部名称；实值和复值线性 kernel 使用了同一个字符串名称。>
    "_chirp_phase_lin_kernel",
    # <学习注释：向 CuPy 后端编译器传入 C++11 标准选项，属于工程配置而非数学算法。>
    options=("-std=c++11",),
)

# <学习注释：创建复值线性 chirp 的逐元素 kernel；相位公式与实值分支相同，但输出是复指数。>
_chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
    # <学习注释：输入仍全部采用模板类型 T。>
    "T t, T f0, T t1, T f1, T phi",
    # <学习注释：输出采用独立模板类型 Y，使 Python 侧可传入 complex64 或 complex128 数组。>
    "Y phase",
    """
    // <学习注释：计算线性扫频斜率 beta。>
    const T beta { (f1 - f0) / t1 };
    // <学习注释：计算线性频率律积分形成的弧度相位 temp。>
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    // <学习注释：Y(real,imag) 构造复数；-cos(psi+π/2)=sin(psi)，所以输出等于 cos(psi)+j*sin(psi)=exp(j*psi)。>
    phase = Y(cos(temp + phi), cos(temp + phi + M_PI/2) * -1);
    """,
    # <学习注释：kernel 内部名称与实值版本相同，但 CuPy 会按不同签名和类型编译相应代码。>
    "_chirp_phase_lin_kernel",
    # <学习注释：使用 C++11 编译 kernel 字符串。>
    options=("-std=c++11",),
)

# <学习注释：创建二次瞬时频率 chirp 的实值逐元素 kernel；vertex_zero 决定频率抛物线的顶点位置。>
_chirp_phase_quad_kernel = cp.ElementwiseKernel(
    # <学习注释：前五项采用模板类型 T，vertex_zero 固定为 bool。>
    "T t, T f0, T t1, T f1, T phi, bool vertex_zero",
    # <学习注释：输出为模板类型 T 的实值余弦样本。>
    "T phase",
    """
    // <学习注释：值初始化临时相位 temp；花括号空初始化使算术类型初值为 0。>
    T temp {};
    // <学习注释：beta=(f1-f0)/t1^2，使二次频率曲线经过两个指定端点。>
    const T beta { (f1 - f0) / (t1 * t1) };
    // <学习注释：vertex_zero 为真时，频率抛物线顶点位于 t=0。>
    if ( vertex_zero ) {
        // <学习注释：积分 f0+beta*t^2 得 f0*t+beta*t^3/3，再乘 2π。>
        temp = 2 * M_PI * (f0 * t + beta * (t * t * t) / 3);
    } else {
        // <学习注释：vertex_zero 为假时进入终点顶点分支；表达式分两行只是 C++ 排版续行。>
        temp = 2 * M_PI *
            ( f1 * t + beta *
            // <学习注释：源码使用 ((t1-t)^3-t1^3)/3；其导数符号与文档声明的 f1+beta*(t1-t)^2 不一致。>
            ( ( (t1 - t) * (t1 - t) * (t1 - t) ) - (t1 * t1 * t1)) / 3);
    }
    // Convert  phi to radians.
    // <学习注释：phi 已由 Python 层转换为弧度；这里计算实值余弦输出。>
    phase = cos(temp + phi);
    """,
    # <学习注释：指定二次 chirp kernel 的内部名称。>
    "_chirp_phase_quad_kernel",
    # <学习注释：使用 C++11 编译。>
    options=("-std=c++11",),
)

# <学习注释：创建对数/几何 chirp 的实值逐元素 kernel。>
_chirp_phase_log_kernel = cp.ElementwiseKernel(
    # <学习注释：输入参数全部共享模板类型 T。>
    "T t, T f0, T t1, T f1, T phi",
    # <学习注释：输出为模板类型 T 的实值样本。>
    "T phase",
    """
    // <学习注释：值初始化临时相位 temp。>
    T temp {};
    // <学习注释：端点频率相等时单独处理，避免 log(f1/f0)=0 导致除零。>
    if ( f0 == f1 ) {
        // <学习注释：相等端点退化为固定频率 f0 的弧度相位 2πf0t。>
        temp = 2 * M_PI * f0 * t;
    } else {
        // <学习注释：beta=t1/log(f1/f0) 是指数频率律积分中的比例因子。>
        T beta { t1 / log(f1 / f0) };
        // <学习注释：pow(f1/f0,t/t1) 是瞬时频率相对 f0 的几何比例；减 1 保证 t=0 时累计相位为 0。>
        temp = 2 * M_PI * beta * f0 * ( pow(f1 / f0, t / t1) - 1.0 );
    }
    // Convert  phi to radians.
    // <学习注释：将累计相位与初相位相加后取余弦。>
    phase = cos(temp + phi);
    """,
    # <学习注释：指定对数 chirp kernel 的内部名称。>
    "_chirp_phase_log_kernel",
    # <学习注释：使用 C++11 编译。>
    options=("-std=c++11",),
)

# <学习注释：创建双曲 chirp 的实值逐元素 kernel。>
_chirp_phase_hyp_kernel = cp.ElementwiseKernel(
    # <学习注释：输入参数全部共享模板类型 T。>
    "T t, T f0, T t1, T f1, T phi",
    # <学习注释：输出为模板类型 T 的实值样本。>
    "T phase",
    """
    // <学习注释：值初始化临时相位 temp。>
    T temp {};
    // <学习注释：端点频率相等时避免后续 f0-f1 为零。>
    if ( f0 == f1 ) {
        // <学习注释：相等端点退化为固定频率信号。>
        temp = 2 * M_PI * f0 * t;
    } else {
        // <学习注释：sing=-f1*t1/(f0-f1) 是双曲频率分母为零的奇点位置。>
        T sing { -f1 * t1 / (f0 - f1) };
        // <学习注释：对双曲频率律积分得到对数相位；abs 允许对数自变量取绝对值，但奇点处仍发散。>
        temp = 2 * M_PI * ( -sing * f0 ) * log( abs( 1 - t / sing ) );
    }
    // Convert  phi to radians.
    // <学习注释：将初相位加到累计相位后取余弦。>
    phase = cos(temp + phi);
    """,
    # <学习注释：指定双曲 chirp kernel 的内部名称。>
    "_chirp_phase_hyp_kernel",
    # <学习注释：使用 C++11 编译。>
    options=("-std=c++11",),
)


def chirp(t, f0, t1, f1, method="linear", phi=0, vertex_zero=True, type="real"):
    # <学习注释：概要翻译——生成瞬时频率随自变量变化的余弦 chirp。>
    # <学习注释：单位说明翻译——这里的 Hz 应理解为每单位自变量的周期数；t 不必以秒计，也可以表示空间位置。>
    # <学习注释：参数 t——类数组采样位置，函数会先把它转换为 CuPy 数组，并逐元素求值。>
    # <学习注释：参数 f0——t=0 时的瞬时频率。>
    # <学习注释：参数 t1——指定 f1 所对应的自变量位置，而不是自动裁剪输出的终止时间。>
    # <学习注释：参数 f1——t=t1 时的瞬时频率。>
    # <学习注释：参数 method——选择 linear、quadratic、logarithmic 或 hyperbolic 频率律，默认 linear；每种还支持若干短别名。>
    # <学习注释：参数 phi——以 degree 给出的相位偏移，默认 0；函数体会乘 π/180 转成 rad。>
    # <学习注释：参数 vertex_zero——只影响 quadratic 分支；真表示频率抛物线顶点在 t=0，假表示顶点按文档意图在 t=t1。>
    # <学习注释：参数 type——仅 linear 分支检查 real 或 complex；其他 method 不读取该参数并总是返回实值。>
    # <学习注释：返回值翻译——输出数组与 t 具有相同 shape；文档称 numpy array，但实现实际返回 CuPy GPU 数组。>
    # <学习注释：返回公式翻译——实值结果是 cos(phase+phi)，其中 phase 是从 0 到 t 对 2πf(t) 的积分。>
    # <学习注释：示例准备第 1 行——从 cusignal 顶层导入 chirp 和 spectrogram。>
    # <学习注释：示例准备第 2 行——导入 matplotlib.pyplot 作为 plt，用于绘图。>
    # <学习注释：示例准备第 3 行——导入 CuPy 作为 cp，用于创建 GPU 数组。>
    # <学习注释：线性示例——cp.linspace(0,10,5001) 包含两个端点，产生 5001 个时刻，间隔为 0.002 秒。>
    # <学习注释：线性示例——chirp 从 6 Hz 下扫到 1 Hz；省略 phi、vertex_zero、type，因而使用零相位和实值输出。>
    # <学习注释：线性示例——cp.asnumpy 将 GPU 数组复制到 CPU NumPy 数组，供 Matplotlib 绘制。>
    # <学习注释：线性示例——title 和 xlabel 设置图标题与横轴标签，show 显示图形。>
    # <学习注释：二次示例准备——fs=8000 表示采样率 8000 Hz，T=10 表示 10 秒。>
    # <学习注释：二次示例准备——endpoint=False 生成 [0,T) 的 80000 个时刻，避免重复采到下一时间区间的端点。>
    # <学习注释：二次示例——默认 vertex_zero=True，因此频率抛物线顶点位于 t=0，从 1500 Hz 下扫到 250 Hz。>
    # <学习注释：二次示例——spectrogram 返回频率坐标 ff、时间坐标 tt 和时频能量 Sxx；续行省略号是 doctest 提示符。>
    # <学习注释：二次示例——pcolormesh 绘制时频图；切片 [:513] 选取前 513 个频率点并与 Sxx 对齐。>
    # <学习注释：二次示例——后续语句设置标题、横纵轴、网格并显示图形。>
    """Frequency-swept cosine generator.

    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
    phi : float, optional
        Phase offset, in degrees. Default is 0.
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.
    type : {'real', 'complex'}, optional
        Specify output chirp type, only applicable when `method` is 'linear'.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.

    Examples
    --------
    The following will be used in the examples:

    >>> from cusignal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt
    >>> import cupy as cp

    For the first example, we'll plot the waveform for a linear chirp
    from 6 Hz to 1 Hz over 10 seconds:

    >>> t = cp.linspace(0, 10, 5001)
    >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(w))
    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
    >>> plt.xlabel('t (sec)')
    >>> plt.show()

    For the remaining examples, we'll use higher frequency ranges,
    and demonstrate the result using `cusignal.spectrogram`.
    We'll use a 10 second interval sampled at 8000 Hz.

    >>> fs = 8000
    >>> T = 10
    >>> t = cp.linspace(0, T, T*fs, endpoint=False)

    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
    (vertex of the parabolic curve of the frequency is at t=0):

    >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')
    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
    ...                           nfft=2048)
    >>> plt.pcolormesh(cp.asnumpy(tt), cp.asnumpy(ff[:513]),
                       cp.asnumpy(Sxx[:513]), cmap='gray_r')
    >>> plt.title('Quadratic Chirp, f(0)=1500, f(10)=250')
    >>> plt.xlabel('t (sec)')
    >>> plt.ylabel('Frequency (Hz)')
    >>> plt.grid()
    >>> plt.show()
    """

    # <学习注释：cp.asarray 把 Python 序列、NumPy 数组或 CuPy 数组统一为 CuPy 数组；已有兼容 CuPy 数组通常无需复制。>
    t = cp.asarray(t)

    # <学习注释：复合赋值把 degree 相位乘 π/180 转成 rad；这里 np.pi 是主机侧 NumPy 浮点常量。>
    phi *= np.pi / 180

    # <学习注释：成员测试接受 linear 的全名和两个短别名。>
    if method in ["linear", "lin", "li"]:
        # <学习注释：real 分支直接调用自动分配输出的实值 ElementwiseKernel，并立即返回。>
        if type == "real":
            return _chirp_phase_lin_kernel_real(t, f0, t1, f1, phi)
        # <学习注释：elif 只在前一个 real 条件为假时检查 complex。>
        elif type == "complex":
            # <学习注释：先按 t.shape 分配 complex64 输出；这是非 float64 输入的默认复数精度。>
            phase = cp.empty(t.shape, dtype=cp.complex64)
            # <学习注释：np.issubclass_ 判断 t.dtype 是否为 float64 的子类型；第二参数虽写成括号，单元素且无逗号时本质仍是 np.float64。>
            if np.issubclass_(t.dtype, (np.float64)):
                # <学习注释：float64 时间输入改用 complex128 输出，保留双精度实部和虚部。>
                phase = cp.empty(t.shape, dtype=cp.complex128)
            # <学习注释：把预分配 phase 作为显式输出参数传给复值 kernel；kernel 对 t 的每个元素写入一个复指数样本。>
            _chirp_phase_lin_kernel_cplx(t, f0, t1, f1, phi, phase)
            # <学习注释：返回已由 GPU kernel 填充的复值 CuPy 数组。>
            return phase
        else:
            # <学习注释：linear method 下 type 既非 real 也非 complex 时，抛出 NotImplementedError 并在消息中插入实际 type。>
            raise NotImplementedError("No kernel for type {}".format(type))

    # <学习注释：quadratic 分支接受全名和两个短别名；type 参数在此分支被忽略。>
    elif method in ["quadratic", "quad", "q"]:
        # <学习注释：调用二次 chirp kernel，并把 vertex_zero 传给每个元素用于选择频率抛物线分支。>
        return _chirp_phase_quad_kernel(t, f0, t1, f1, phi, vertex_zero)

    # <学习注释：logarithmic 分支接受全名及两个短别名。>
    elif method in ["logarithmic", "log", "lo"]:
        # <学习注释：乘积小于等于零涵盖任一端点为零或两端异号，都会使实数 log(f1/f0) 公式无效。>
        if f0 * f1 <= 0.0:
            # <学习注释：raise ValueError 表示参数值违反对数 chirp 的数学定义域。>
            raise ValueError(
                "For a logarithmic chirp, f0 and f1 must be "
                # <学习注释：相邻字符串字面量会在 Python 编译期自动拼接成一条完整错误消息。>
                "nonzero and have the same sign."
            )
        # <学习注释：参数合法时调用对数 chirp kernel并返回实值数组。>
        return _chirp_phase_log_kernel(t, f0, t1, f1, phi)

    # <学习注释：hyperbolic 分支接受全名和 hyp 短名。>
    elif method in ["hyperbolic", "hyp"]:
        # <学习注释：任一端点频率为零都会使双曲频率律参数或相位公式退化，因此拒绝。>
        if f0 == 0 or f1 == 0:
            # <学习注释：同一行相邻的两个字符串字面量自动拼接；源码未检查采样区间是否跨越双曲奇点。>
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be " "nonzero.")
        # <学习注释：调用双曲 chirp kernel 并返回实值数组。>
        return _chirp_phase_hyp_kernel(t, f0, t1, f1, phi)

    else:
        # <学习注释：method 不匹配任何合法名称或别名时，构造并抛出 ValueError。>
        raise ValueError(
            "method must be 'linear', 'quadratic', 'logarithmic',"
            # <学习注释：相邻字符串先拼接，%r 再用 repr 风格插入非法 method，便于看清引号和类型表示。>
            " or 'hyperbolic', but a value of %r was given." % method
        )


# <学习注释：创建并缓存一个 CuPy 逐元素 kernel；调用时每个输出位置独立判断自己是否等于目标索引。>
_unit_impulse_kernel = cp.ElementwiseKernel(
    # <学习注释：kernel 只有一个显式输入 idx，并强制按 int32 解释；CuPy 还会自动提供当前线性元素索引 i。>
    "int32 idx",
    # <学习注释：kernel 输出变量名为 out，元素类型固定为 float64，因此函数形参 dtype 没有控制这里的输出类型。>
    "float64 out",
    """
    // <学习注释：i 是 ElementwiseKernel 自动提供的当前一维线性索引；不等于 idx 时写入 0。>
    if (i != idx) {
        out = 0;
    } else {
        // <学习注释：只有当前线性索引 i 等于目标 idx 时才写入 1，直接落实 Kronecker delta 的分段定义。>
        out = 1;
    }
    """,
    # <学习注释：为生成的 kernel 指定内部名称 _unit_impulse_kernel，便于 CuPy 编译、缓存与标识。>
    "_unit_impulse_kernel",
    # <学习注释：把 -std=c++11 作为编译选项传给后端编译器；它属于工程配置，不改变单位冲激公式。>
    options=("-std=c++11",),
)


def unit_impulse(shape, idx=None, dtype=float):
    # <学习注释：函数签名接收输出形状 shape、目标索引 idx 和期望 dtype；当前实现实际未使用 dtype。>
    # <学习注释：docstring 宣称支持 N-D shape 和 tuple idx，但函数末尾只读取 shape[0] 与 idx[0]，应以实际代码为准。>
    # <学习注释：概要翻译——本函数用于生成单位冲激信号，也就是离散 delta；一维时也可看成标准单位基向量。>
    # <学习注释：Parameters、Returns、Notes 和 Examples 是 NumPy 风格 docstring 的章节标题及下划线，不是可执行 Python 语句。>
    # <学习注释：shape 参数翻译——整数表示一维输出样本数，整数 tuple 按文档意图表示 N 维输出形状。>
    # <学习注释：idx 参数翻译——它指定唯一值为 1 的位置；None 表示默认第 0 个元素。>
    # <学习注释：idx='mid' 的文档语义是所有维度都取 shape // 2；// 表示逐维向下取整除法。>
    # <学习注释：标量 idx 的文档语义是广播到所有维度，例如二维 shape 配合 idx=2 意图表示位置 (2,2)。>
    # <学习注释：dtype 参数翻译——文档声称它控制数组元素类型，示例 numpy.int8 表示 8 位有符号整数，默认声称为 numpy.float64。>
    # <学习注释：Returns 翻译——y 应是包含单位冲激的 ndarray；这里的 ndarray 在 cuSignal 语境中预期是 CuPy 数组。>
    # <学习注释：Notes 翻译——一维序列只有目标索引为 1、其余为 0，因此也称 Kronecker delta。>
    # <学习注释：Sphinx 语法 :math:`...` 表示把反引号中的内容按数学公式渲染，双反斜杠在 Python 字符串中保留 LaTeX 的反斜杠。>
    # <学习注释：示例一——cusignal.unit_impulse(8) 省略 idx，文档预期在索引 0 生成长度 8 的冲激。>
    # <学习注释：示例中的 import cupy as cp 实际未被后续示例语句直接使用，可能用于强调返回对象属于 CuPy/GPU 数组语境。>
    # <学习注释：示例二——unit_impulse(7,2) 文档预期生成 delta[n-2]，即长度 7 且索引 2 为 1。>
    # <学习注释：示例三——shape=(3,3)、idx='mid' 按文档意图取中心坐标 (1,1)，输出 3×3 数组。>
    # <学习注释：示例四——shape=(4,4)、标量 idx=2 按文档意图广播成坐标 (2,2)，输出 4×4 数组。>
    # <学习注释：重要核对——前两个一维示例与当前函数体一致；后两个二维示例和 dtype 说明没有被当前返回语句落实。>
    """
    Unit impulse signal (discrete delta function) or unit basis vector.

    Parameters
    ----------
    shape : int or tuple of int
        Number of samples in the output (1-D), or a tuple that represents the
        shape of the output (N-D).
    idx : None or int or tuple of int or 'mid', optional
        Index at which the value is 1.  If None, defaults to the 0th element.
        If ``idx='mid'``, the impulse will be centered at ``shape // 2`` in
        all dimensions.  If an int, the impulse will be at `idx` in all
        dimensions.
    dtype : data-type, optional
        The desired data-type for the array, e.g., ``numpy.int8``.  Default is
        ``numpy.float64``.

    Returns
    -------
    y : ndarray
        Output array containing an impulse signal.

    Notes
    -----
    The 1D case is also known as the Kronecker delta.

    Examples
    --------
    An impulse at the 0th element (:math:`\\delta[n]`):

    >>> import cusignal
    >>> import cupy as cp
    >>> cusignal.unit_impulse(8)
    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    Impulse offset by 2 samples (:math:`\\delta[n-2]`):

    >>> cusignal.unit_impulse(7, 2)
    array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.])

    2-dimensional impulse, centered:

    >>> cusignal.unit_impulse((3, 3), 'mid')
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])

    Impulse at (2, 2), using broadcasting:

    >>> cusignal.unit_impulse((4, 4), 2)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """
    # <学习注释：np.atleast_1d 保证 shape 至少是一维；标量 N 变成长度 1 的数组 [N]，tuple 则变成包含各维长度的一维数组。>
    shape = np.atleast_1d(shape)

    # <学习注释：None 表示每一维都选第 0 个索引；(0,) 先构造单元素 tuple，再按维数 len(shape) 重复。>
    if idx is None:
        idx = (0,) * len(shape)
    # <学习注释：字符串 mid 表示逐维取 shape // 2；// 是向下取整除法，tuple 将 NumPy 数组结果转成索引元组。>
    elif idx == "mid":
        idx = tuple(shape // 2)
    # <学习注释：若 idx 没有 __iter__ 属性，就把它视为标量，并复制到 shape 的每一维；tuple 等 iterable 会原样保留。>
    elif not hasattr(idx, "__iter__"):
        idx = (idx,) * len(shape)

    # <学习注释：实际只把第一维目标 idx[0] 传给 kernel，并用 size=shape[0] 创建一维输出；其余维度与 dtype 均未参与。>
    # <学习注释：CuPy 13.6.0 的 _kernel.pyx:891-897 会把显式 size 设为一维 shape 并据此分配输出，所以这里实际返回长度 shape[0] 的一维数组。>
    return _unit_impulse_kernel(idx[0], size=shape[0])
