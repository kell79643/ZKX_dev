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

import cupy as cp

from . import _convolution_cuda
from .convolution_utils import _inputs_swap_needed, _reverse_and_conj
from .convolve import convolve

_modedict = {"valid": 0, "same": 1, "full": 2}


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

    # <学习注释：将第一个输入转为 CuPy GPU 数组，不显式指定 dtype。>
    in1 = cp.asarray(in1)
    # <学习注释：将第二个输入转为 CuPy GPU 数组。>
    in2 = cp.asarray(in2)

    # <学习注释：两个零维数组按复数内积规则直接相乘，第二个标量取共轭。>
    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    # <学习注释：非标量输入的维数必须一致，否则无法定义逐轴滞后。>
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # this either calls fftconvolve or this function with method=='direct'
    # <学习注释：FFT 和自动路径都复用 convolve，先对 in2 的每个维度反转并取共轭。>
    if method in ("fft", "auto"):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    # <学习注释：显式 direct 路径进入预编译 CUDA kernel，不经过 _reverse_and_conj。>
    elif method == "direct":

        # <学习注释：当前直接 kernel 只提供一维互相关。>
        if in1.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")

        # <学习注释：记录第二个输入是否更长，以便交换后恢复相关的输出方向。>
        swapped_inputs = in2.size > in1.size

        # <学习注释：让 kernel 的 inp 不短于 kernel，降低边界处理的复杂度。>
        if swapped_inputs:
            in1, in2 = in2, in1

        # <学习注释：use_convolve=False 使后端选择 correlate kernel；注意该 kernel 本身未对复数 in2 取共轭。>
        return _convolution_cuda._convolve(in1, in2, False, swapped_inputs, mode)

    # <学习注释：拒绝除 auto、direct 和 fft 之外的方法名。>
    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")


# <学习注释：定义二维互相关公开函数；下一组续行列出五个形参。>
def correlate2d(
# <学习注释：in1 是第一个二维输入，输出的 same 尺寸以它为基准。>
    in1,
# <学习注释：in2 是第二个二维输入，在复数情形下稍后会先取共轭。>
    in2,
# <学习注释：mode 默认 full，决定输出保留完整、同尺寸还是仅完全重叠区域。>
    mode="full",
# <学习注释：boundary 默认 fill，决定输入边界外采用常数、周期还是对称延拓。>
    boundary="fill",
# <学习注释：fillvalue 只在 fill 边界模式下作为常数填充值，默认 0。>
    fillvalue=0,
# <学习注释：右括号结束多行函数签名，函数没有 Python 返回类型注解。>
):
    """
    Cross-correlate two 2-dimensional arrays.
    Cross correlate `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

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
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:
        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.
    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:
    >>> import cusignal
    >>> import cupy as cp
    >>> from scipy import misc
    >>> face = cp.asarray(misc.face(gray=True) - misc.face(gray=True).mean())
    >>> template = cp.copy(face[300:365, 670:750])  # right eye
    >>> template -= template.mean()
    >>> face = face + cp.random.randn(*face.shape) * 50  # add noise
    >>> corr = cusignal.correlate2d(face, template, boundary='symm', \
        mode='same')
    >>> y, x = cp.unravel_index(cp.argmax(corr), corr.shape)  # find the match
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_template, ax_corr) =
    ...     plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(cp.asnumpy(face), cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(cp.asnumpy(template), cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(cp.asnumpy(corr), cmap='gray')
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    >>> ax_orig.plot(cp.asnumpy(x), cp.asnumpy(y), 'ro')
    >>> fig.show()

    """

# <学习注释：把第一个 array_like 转成 CuPy 数组；已有 GPU 数组通常无需主机复制。>
    in1 = cp.asarray(in1)
# <学习注释：把第二个 array_like 转成 CuPy 数组，使后续 ndim、shape、dtype 和 GPU 运算接口统一。>
    in2 = cp.asarray(in2)

# <学习注释：链式比较要求 in1.ndim、in2.ndim 和常量 2 同时相等。>
    if not in1.ndim == in2.ndim == 2:
# <学习注释：任一输入不是二维时立即抛出 ValueError，不启动后端 kernel。>
        raise ValueError("correlate2d inputs must both be 2D arrays")

# <学习注释：valid 模式要求一个输入在每一维都不小于另一个；helper 返回是否应交换两者。>
    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
# <学习注释：只有 helper 判定第二输入才是逐维较大者时进入交换分支。>
    if swapped_inputs:
# <学习注释：Python 元组赋值原子地交换两个数组引用，使较大数组成为后端的第一个输入。>
        in1, in2 = in2, in1

# <学习注释：调用统一的二维卷积/相关后端，并把返回的 GPU 数组保存为 out。>
    out = _convolution_cuda._convolve2d(
# <学习注释：传入可能已交换的较大第一输入。>
        in1,
# <学习注释：对第二输入逐元素取复共轭；实数不变，复数由此符合互相关定义。>
        in2.conj(),
# <学习注释：第三个位置参数 use_convolve=0，要求后端选择 correlate2D 而非 convolve2D kernel。>
        0,
# <学习注释：把 full、same 或 valid 模式字符串传给后端解析。>
        mode,
# <学习注释：把 fill、wrap 或 symm 边界字符串传给后端解析。>
        boundary,
# <学习注释：把常数填充值传给后端；非 fill 模式下不会作为边界数据使用。>
        fillvalue,
# <学习注释：右括号结束 _convolve2d 的多行实参列表。>
    )

# <学习注释：若 valid 模式为满足后端尺寸约束交换过输入，就要恢复原调用的位移方向。>
    if swapped_inputs:
# <学习注释：两个步长均为 -1 的切片把输出沿行、列两轴反转；此处没有再次共轭。>
        out = out[::-1, ::-1]

# <学习注释：返回位于 GPU 上的二维 CuPy 数组。>
    return out


def correlation_lags(in1_len, in2_len, mode="full"):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = cp.concatenate([rng.standard_normal(100), x])
    >>> correlation = cusignal.correlate(x, y, mode="full")
    >>> lags = cusignal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[cp.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = cp.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = cp.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid - lag_bound) : (mid + lag_bound)]
        else:
            lags = lags[(mid - lag_bound) : (mid + lag_bound) + 1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = cp.arange(lag_bound + 1)
        else:
            lags = cp.arange(lag_bound, 1)
    return lags
