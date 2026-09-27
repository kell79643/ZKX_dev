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
import numpy as np
from cupyx.scipy import linalg

from ..convolution.correlate import correlate
from ..filter_design.filter_design_utils import _validate_sos
from ..utils.arraytools import (
    _axis_reverse,
    _axis_slice,
    _const_ext,
    _even_ext,
    _odd_ext,
)
from ..utils.helper_tools import _get_max_smem, _get_max_tpb
from ._channelizer_cuda import _channelizer
from ._sosfilt_cuda import _sosfilt

_wiener_prep_kernel = cp.ElementwiseKernel(
    "T iMean, T iVar, T prod",
    "T lMean, T lVar",
    """
    // Estimate the local mean
    T temp { iMean / prod };
    lMean = temp;
    lVar = iVar / prod - (temp * temp);
    """,
    "_wiener_prep_kernel",
    options=("-std=c++11",),
)

_wiener_post_kernel = cp.ElementwiseKernel(
    "T im, T lMean, T lVar, T noise",
    "T out",
    """
    // Estimate the local mean
    T res { im - lMean };
    res *= 1.0 - noise / lVar;
    res += lMean;
    if ( lVar < noise ) {
        out = lMean;
    } else {
        out = res;
    }
    """,
    "_wiener_post_kernel",
    options=("-std=c++11",),
)


def wiener(im, mysize=None, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or array_like, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.

    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.

    """
    im = cp.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = cp.repeat(mysize.item(), im.ndim)
        mysize = np.asarray(mysize)

    lprod = cp.prod(mysize, axis=0)
    lMean = correlate(im, cp.ones(mysize), "same")
    lVar = correlate(im**2, cp.ones(mysize), "same")

    lMean, lVar = _wiener_prep_kernel(lMean, lVar, lprod)

    # Estimate the noise power if needed.
    if noise is None:
        noise = cp.mean(cp.ravel(lVar), axis=0)

    return _wiener_post_kernel(im, lMean, lVar, noise)


# <学习注释：定义 FIR 滤波公开函数；b 是一维抽头，x 是输入，axis 选择滤波轴，zi 可传入上一块留下的初始状态。>
def firfilter(b, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an FIR filter.

    Filter a data sequence, `x`, using a digital filter. This works for many
    fundamental data types (including Object type). Please note, cuSignal
    doesn't support IIR filters presently, and this implementation is optimized
    for large filtering operations (and inherently depends on fftconvolve)

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """
    # <学习注释：把系数 b 转换为 CuPy 数组；若已在 GPU 上且条件允许，asarray 可避免不必要的复制。>
    b = cp.asarray(b)
    # <学习注释：检查抽头必须恰好一维，b.ndim != 1 会进入异常分支。>
    if b.ndim != 1:
        # <学习注释：拒绝标量或多维系数，因为后续 cp.convolve 按一维 FIR 系数解释 b。>
        raise ValueError("object of too small depth for desired array")

    # <学习注释：x.ndim == 0 表示 x 是标量；FIR 需要至少一条样本轴。>
    if x.ndim == 0:
        # <学习注释：对零维输入报错，不自动把标量扩成长度 1 的序列。>
        raise ValueError("x must be at least 1-D")

    # <学习注释：创建参与共同 dtype 推导的输入列表，初始包含 b 和 x。>
    inputs = [b, x]
    # <学习注释：只有用户显式给出初始状态时，才执行状态 shape、广播和 dtype 处理。>
    if zi is not None:
        # _linear_filter does not broadcast zi, but does do expansion of
        # singleton dims.
        # <学习注释：把 zi 转换成 CuPy 数组，便于在 GPU 数组上做 shape、stride 和加法。>
        zi = cp.asarray(zi)
        # <学习注释：要求 zi 和 x 有相同的轴数；后续只允许单例维扩展，不自动增加维度。>
        if zi.ndim != x.ndim:
            # <学习注释：当 zi 维数不符合接口契约时立即终止。>
            raise ValueError("object of too small depth for desired array")
        # <学习注释：从 x.shape 复制出可修改列表，用来构造 zi 的目标 shape。>
        expected_shape = list(x.shape)
        # <学习注释：FIR 长度为 len(b) 时需要 len(b)-1 个延迟状态，因此替换滤波轴长度。>
        expected_shape[axis] = b.shape[0] - 1
        # <学习注释：将可变 list 固化为 shape API 常用的 tuple。>
        expected_shape = tuple(expected_shape)
        # check the trivial case where zi is the right shape first
        # <学习注释：当 zi 已经是完整目标 shape 时直接使用；只在不相等时尝试单例维广播。>
        if zi.shape != expected_shape:
            # <学习注释：预分配每个轴的 stride 描述，None 会在下方循环中被实际字节步长或 0 替换。>
            strides = zi.ndim * [None]
            # <学习注释：将负轴索引转换为等价的非负索引，便于在 range(zi.ndim) 中比较。>
            if axis < 0:
                # <学习注释：例如 ndim=3 时 axis=-1 转成 2。>
                axis += zi.ndim
            # <学习注释：遍历 zi 的每个轴，逐轴判定是保留原 stride、用 stride 0 广播，还是报错。>
            for k in range(zi.ndim):
                # <学习注释：滤波轴不允许广播，它必须恰好容纳 len(b)-1 个状态。>
                if k == axis and zi.shape[k] == expected_shape[k]:
                    # <学习注释：保留该轴原始字节步长，使每个状态元素按原布局访问。>
                    strides[k] = zi.strides[k]
                # <学习注释：非滤波轴若已与 x 完全匹配，也保留原 stride。>
                elif k != axis and zi.shape[k] == expected_shape[k]:
                    # <学习注释：记录已匹配非滤波轴的原始字节步长。>
                    strides[k] = zi.strides[k]
                # <学习注释：非滤波轴长度为 1 时，允许把该单个状态切片复用到所有位置。>
                elif k != axis and zi.shape[k] == 1:
                    # <学习注释：将 stride 设为 0，使该轴索引变化时仍访问同一内存位置，实现无复制广播。>
                    strides[k] = 0
                # <学习注释：除“完全匹配”和“非滤波轴为 1”以外的 shape 均不可广播。>
                else:
                    # <学习注释：格式化异常消息，同时显示期望 shape 和实际 shape。>
                    raise ValueError(
                        # <学习注释：这是跨行字符串的第一部分，Python 会在括号内自动与下一个字符串连接。>
                        "Unexpected shape for zi: expected "
                        # <学习注释：用 % 操作符将 expected_shape 和 zi.shape 代入两个 %s 占位符。>
                        "%s, found %s." % (expected_shape, zi.shape)
                    # <学习注释：关闭 ValueError(...) 调用的右括号。>
                    )
            # <学习注释：as_strided 使用目标 shape 和上述 strides 创建 zi 视图；stride 0 的轴会广播而不复制数据。>
            zi = cp.lib.stride_tricks.as_strided(zi, expected_shape, strides)
        # <学习注释：将已验证/广播的 zi 加入 dtype 推导列表，避免后续加法降精度。>
        inputs.append(zi)
    # <学习注释：按 CuPy/NumPy 类型提升规则求 b、x 以及可选 zi 的公共结果 dtype。>
    dtype = cp.result_type(*inputs)

    # <学习注释：dtype.char 是类型字符码；此处只接受单/双/扩展浮点、对应复数和 object 类别。>
    if dtype.char not in "fdgFDGO":
        # <学习注释：对不在允许集合中的结果 dtype 报 NotImplementedError，并在消息中显示该 dtype。>
        raise NotImplementedError("input type '%s' not supported" % dtype)

    # <学习注释：把 b 实体化为公共 dtype 的 CuPy 数组，使卷积两端类型一致。>
    b = cp.array(b, dtype=dtype)
    # <学习注释：把 x 转为公共 dtype；copy=False 表示在可能时复用现有存储，但类型转换仍可能分配。>
    x = cp.array(x, dtype=dtype, copy=False)

    # <学习注释：沿 axis 取出每条一维 y，用 lambda 计算 cp.convolve(b, y) 的默认 full 线性卷积，再组回 N 维结果。>
    out_full = cp.apply_along_axis(lambda y: cp.convolve(b, y), axis, x)

    # <学习注释：为每个轴创建 slice(None)，它等价于切片语法中的冒号 :。>
    ind = out_full.ndim * [slice(None)]
    # <学习注释：如果有初始状态，则把它叠加到 full 卷积的前 len(b)-1 个输出位置。>
    if zi is not None:
        # <学习注释：把滤波轴切片替换为从 0 到 zi.shape[axis] 的前缀，其他轴仍取全部。>
        ind[axis] = slice(zi.shape[axis])
        # <学习注释：对选中前缀做原地加法，实现 c'[n]=c[n]+zi[n]。>
        out_full[tuple(ind)] += zi

    # <学习注释：full 卷积长度为 N+L-1，减去 L 再加 1 得 N，因此此切片选中与输入同长的前 N 项。>
    ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
    # <学习注释：将切片列表转成 tuple 作多维索引，得到因果主输出 out。>
    out = out_full[tuple(ind)]

    # <学习注释：无 zi 时 API 只返回输出数组，不返回最终状态。>
    if zi is None:
        # <学习注释：返回沿 axis 与 x 同长、其他维度不变的滤波结果。>
        return out
    # <学习注释：有 zi 时进入 else，需要同时构造并返回 zf。>
    else:
        # <学习注释：把滤波轴切片改为从主输出结束位置 N 直到 full 卷积末尾。>
        ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
        # <学习注释：取出尾部 len(b)-1 个样本作为最终滤波延迟状态 zf。>
        zf = out_full[tuple(ind)]
        # <学习注释：返回二元组 (out, zf)，zf 可作为下一输入块的 zi。>
        return out, zf


def lfilter(b, a, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an FIR filter.

    Filter a data sequence, `x`, using a digital filter. This works for many
    fundamental data types (including Object type). Please note, cuSignal
    doesn't support IIR filters presently, and this implementation is optimized
    for large filtering operations (and inherently depends on fftconvolve)

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """
    a = cp.atleast_1d(a)
    if len(a) == 1:
        return firfilter(b, x, axis=axis, zi=zi)
    else:
        raise NotImplementedError("IIR support isn't supported yet")


def lfilter_zi(b, a):
    """
    Construct initial conditions for lfilter for step response steady-state.
    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.
    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.
    Parameters
    ----------
    b, a : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.
    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.
    See Also
    --------
    lfilter, lfiltic, filtfilt
    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::
        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)
    where z(n) is a vector of length m, A has shape (m, m), B has shape
    (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
    a scalar).  lfilter_zi solves::
        zi = A*zi + B
    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.
    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter,
    which is the implementation used by scipy.signal.lfilter, are::
        A = scipy.linalg.companion(a).T
        B = b[1:] - a[1:]*b[0]
    assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
    divided by a[0].
    """
    b = cp.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be 1-D.")
    a = cp.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be 1-D.")

    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    if a[0] != 1.0:
        # Normalize the coefficients so a[0] == 1.
        b = b / a[0]
        a = a / a[0]

    n = max(len(a), len(b))

    # Pad a or b with zeros so they are the same length.
    if len(a) < n:
        a = cp.r_[a, cp.zeros(n - len(a), dtype=a.dtype)]
    elif len(b) < n:
        b = cp.r_[b, cp.zeros(n - len(b), dtype=b.dtype)]

    IminusA = cp.eye(n - 1, dtype=cp.result_type(a, b)) - linalg.companion(a).T
    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    zi = cp.linalg.solve(IminusA, B)

    return zi


def firfilter_zi(b):
    """
    Construct initial conditions for lfilter for step response steady-state.
    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.
    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.
    Parameters
    ----------
    b : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.
    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.
    See Also
    --------
    lfilter, lfiltic, filtfilt
    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::
        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)
    where z(n) is a vector of length m, A has shape (m, m), B has shape
    (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
    a scalar).  lfilter_zi solves::
        zi = A*zi + B
    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.
    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter,
    which is the implementation used by scipy.signal.lfilter, are::
        A = scipy.linalg.companion(a).T
        B = b[1:] - a[1:]*b[0]
    assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
    divided by a[0].
    """
    return lfilter_zi(b, np.float32(1.0))


def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    if padtype not in ["even", "odd", "constant", None]:
        raise ValueError(
            (
                "Unknown value '%s' given to padtype.  padtype "
                "must be 'even', 'odd', 'constant', or None."
            )
            % padtype
        )

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError(
            "The length of the input vector x must be greater "
            "than padlen, which is %d." % edge
        )

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == "even":
            ext = _even_ext(x, edge, axis=axis)
        elif padtype == "odd":
            ext = _odd_ext(x, edge, axis=axis)
        else:
            ext = _const_ext(x, edge, axis=axis)
    else:
        ext = x
    return edge, ext


# <学习注释：firfilter2 只接收 FIR 分子系数 b；它沿 axis 对 x 做正向和反向各一次的零相位离线滤波。>
def firfilter2(b, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    """
    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.
    The function provides options for handling the edges of the signal.
    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.
    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.
    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.
    The option to use Gustaffson's method was added in scipy version 0.16.0.
    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.
    """
    # <学习注释：保证 b 至少是一维 CuPy 数组，后续 len(b) 才能表示 FIR 抽头数。>
    b = cp.atleast_1d(b)
    # <学习注释：把输入数据转换成 CuPy 数组；若已是兼容数组通常避免不必要复制。>
    x = cp.asarray(x)

    # <学习注释：接口只承认 pad 和 gust 两个字符串，其他值属于参数错误。>
    if method not in ["pad", "gust"]:
        raise ValueError("method must be 'pad' or 'gust'.")

    # <学习注释：23.08.00 仅保留 gust 的接口位置，实际没有实现 Gustafsson 初值算法。>
    if method == "gust":
        raise NotImplementedError("gust method not supported yet")

    # method == "pad"
    # <学习注释：校验延拓参数并得到每端延拓长度 edge 以及延拓后的数组 ext。>
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=len(b))

    # Get the steady state of the filter's step response.
    # <学习注释：求单位阶跃输入下 FIR 滤波器的稳态延迟状态，供两次滤波抑制启动暂态。>
    zi = firfilter_zi(b)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    # <学习注释：只有滤波轴维度保存 len(b)-1 个状态，其他轴维度保留 1 以便广播。>
    zi_shape[axis] = zi.size
    zi = cp.reshape(zi, zi_shape)
    # <学习注释：取得延拓信号沿滤波轴的第一个样本，用它缩放单位阶跃稳态 zi。>
    x0 = _axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    # <学习注释：第一次按原时间方向做 FIR 滤波；zf 是末状态，本函数后续不使用该值。>
    (y, zf) = firfilter(b, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    # <学习注释：取得正向输出末样本，作为反向滤波稳态初值的缩放量。>
    y0 = _axis_slice(y, start=-1, axis=axis)
    # <学习注释：先反转正向输出，再用相同 b 做第二次 FIR；恢复方向后组合响应为 H(z)H(z^{-1})。>
    (y, zf) = firfilter(b, _axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    # <学习注释：把第二次滤波结果恢复为与输入一致的样本方向。>
    y = _axis_reverse(y, axis=axis)

    # <学习注释：只有确实延拓过时才需要从两端裁掉 edge 个辅助样本。>
    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = _axis_slice(y, start=edge, stop=-edge, axis=axis)

    # <学习注释：返回独立的 CuPy 数组，避免结果只是原中间数组切片的视图。>
    return cp.copy(y)


def filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    """
    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.
    The function provides options for handling the edges of the signal.
    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.
    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.
    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.
    The option to use Gustaffson's method was added in scipy version 0.16.0.
    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.
    """
    a = cp.atleast_1d(a)
    if len(a) == 1:
        return firfilter2(
            b,
            x,
            axis=axis,
            padtype=padtype,
            padlen=padlen,
            method=method,
            irlen=irlen,
        )
    else:
        raise NotImplementedError("IIR support isn't supported yet")


def sosfilt(
    sos,
    x,
    axis=-1,
    zi=None,
):
    # <学习注释：以下 docstring 定义 SOS shape、滤波轴、初末状态和当前 GPU 实现限制。>
    """
    Filter data along one dimension using cascaded second-order sections.
    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz

    Notes
    -----
    WARNING: This is an experimental API and is prone to change in future
    versions of cuSignal.

    The filter function is implemented as a series of second-order filters
    with direct-form II transposed structure. It is designed to minimize
    numerical precision errors for high-order filters.

    Limitations
    -----------
    1. The number of n_sections must be less than 513.
    2. The number of samples must be greater than the number of sections

    Examples
    --------
    sosfilt is a stable alternative to `lfilter` as using 2nd order sections
    reduces numerical error. We are working on building out sos filter output,
    so please submit GitHub feature requests as needed. You can also generate
    a filter on CPU with scipy.signal and then move that to GPU for actual
    filtering operations with `cp.asarray`.

    Plot a 13th-order filter's impulse response using both `sosfilt`:
    >>> from scipy import signal
    >>> import cusignal
    >>> import cupy as cp
    >>> # Generate filter on CPU with Scipy.Signal
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> # Move data to GPU
    >>> sos = cp.asarray(sos)
    >>> x = cp.random.randn(100_000_000)
    >>> y = cusignal.sosfilt(sos, x)
    """

    # <学习注释：把输入转换为 CuPy 数组；已在 GPU 上时通常避免不必要复制。>
    x = cp.asarray(x)
    # <学习注释：标量没有可供滤波的时间轴，必须拒绝。>
    if x.ndim == 0:
        raise ValueError("x must be at least 1D")

    # <学习注释：校验 SOS 为二维六列且 a0 全为 1，并取得节数。>
    sos, n_sections = _validate_sos(sos)
    # <学习注释：将校验后的 SOS 转为 GPU 数组。>
    sos = cp.asarray(sos)

    # <学习注释：先复制输入 shape，再把滤波轴长度替换成每节所需的两个状态。>
    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    # <学习注释：在最前面增加 n_sections 维，形成 API 约定的 zi shape。>
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    # <学习注释：输出 dtype 由 SOS、输入和可选初态共同提升决定。>
    inputs = [sos, x]

    if zi is not None:
        # <学习注释：这里用 NumPy 视角把 zi 纳入 result_type 推断；随后才复制到 GPU。>
        inputs.append(np.asarray(zi))

    # <学习注释：cp.result_type 按 NumPy/CuPy 类型提升规则求共同计算 dtype。>
    dtype = cp.result_type(*inputs)

    # <学习注释：仅接受浮点、复数、long double/complex long double 或 object 类别字符；预编译后端实际只列出 float32/float64。>
    if dtype.char not in "fdgFDGO":
        raise NotImplementedError("input type '%s' not supported" % dtype)
    if zi is not None:
        # <学习注释：显式复制初态，允许 kernel 原地修改而不直接覆盖调用者对象。>
        zi = cp.array(zi, dtype)  # make a copy so that we can operate in place
        # <学习注释：每节、每条独立信号必须恰有两个 DF-II-T 状态。>
        if zi.shape != x_zi_shape:
            raise ValueError(
                "Invalid zi shape. With axis=%r, an input with "
                "shape %r, and an sos array with %d sections, zi "
                "must have shape %r, got %r."
                % (axis, x.shape, n_sections, x_zi_shape, zi.shape)
            )
        return_zi = True
    else:
        # <学习注释：未给初态即构造全零状态，表示 initial rest。>
        zi = cp.zeros(x_zi_shape, dtype=dtype)
        return_zi = False

    # <学习注释：把负轴索引规范化为 [0, x.ndim) 内的正索引。>
    axis = axis % x.ndim  # make positive
    # <学习注释：kernel 要求样本维在最后，因此把滤波轴移到末尾。>
    x = cp.moveaxis(x, axis, -1)
    # <学习注释：把 zi 的“节”维和“两状态”维移动到最后两个轴。>
    zi = cp.moveaxis(zi, [0, axis + 1], [-2, -1])
    # <学习注释：保存移动轴后的 shape，kernel 返回后用于恢复。>
    x_shape, zi_shape = x.shape, zi.shape
    # <学习注释：把除样本轴外的所有维展平成独立信号行。>
    x = cp.reshape(x, (-1, x.shape[-1]))
    # <学习注释：复制为指定 dtype 的 C 连续数组，因为 kernel 会把输出原地写回 x。>
    x = cp.array(x, dtype, order="C")  # make a copy, can modify in place
    # <学习注释：把状态整理为 [独立信号, 节, 2] 的连续布局。>
    zi = cp.ascontiguousarray(cp.reshape(zi, (-1, n_sections, 2)))
    # <学习注释：SOS 系数转换到共同 dtype；相同时不复制。>
    sos = sos.astype(dtype, copy=False)

    # <学习注释：查询当前 GPU 每 block 的共享内存和线程数上限。>
    max_smem = _get_max_smem()
    max_tpb = _get_max_tpb()

    # Determine how much shared memory is needed
    # <学习注释：共享内存保存每节一个流水输出和每节六个 SOS 系数。>
    out_size = sos.shape[0]
    sos_size = sos.shape[0] * sos.shape[1]
    shared_mem = (out_size + sos_size) * x.dtype.itemsize

    if shared_mem > max_smem:
        max_sections = max_smem // (1 + zi.shape[2] + sos.shape[1]) // x.dtype.itemsize
        raise ValueError(
            "The number of sections ({}), requires too much "
            "shared memory ({}B) > ({}B). \n"
            "\n**Max sections possible ({})**".format(
                sos.shape[0], shared_mem, max_smem, max_sections
            )
        )

    # <学习注释：每节映射一个线程，因此节数不能超过每 block 最大线程数。>
    if sos.shape[0] > max_tpb:
        raise ValueError(
            "The number of sections ({}), must be less "
            "than max threads per block ({})".format(sos.shape[0], max_tpb)
        )

    # <学习注释：流水线装载需要样本数至少多于节数；这是实现限制而非 SOS 数学限制。>
    if sos.shape[0] > x.shape[1]:
        raise ValueError(
            "The number of samples ({}), must be greater "
            "than the number of sections ({})".format(x.shape[1], sos.shape[0])
        )

    # <学习注释：启动 CUDA 后端；x 和预期的 zi 都按原地语义传入。>
    _sosfilt(sos, x, zi)

    # <学习注释：恢复样本数组的移动轴前 shape 与原滤波轴位置。>
    x.shape = x_shape
    x = cp.moveaxis(x, -1, axis)
    if return_zi:
        # <学习注释：若调用者提供 zi，则把状态恢复成 API 约定的维序并与 y 一起返回。>
        zi.shape = zi_shape
        zi = cp.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x

    # <学习注释：返回 y，或在给定 zi 时返回 (y, zf)。>
    return out


# <学习注释：创建逐元素 CuPy kernel，用 GPU 线程并行生成解析信号所需的一维频域掩码 h。>
_hilbert_kernel = cp.ElementwiseKernel(
    # <学习注释：空字符串表示 kernel 没有显式输入数组，元素数量由输出 h 决定。>
    "",
    # <学习注释：声明一个类型占位符为 T 的输出数组 h；CuPy 会根据实际 dtype 生成具体 kernel。>
    "T h",
    """
    // <学习注释：odd 为 false 时 N 为偶数，此时 DC 和 Nyquist 两个自共轭频点都保留为 1。>
    if ( !odd ) {
        // <学习注释：i==0 对应 DC，i==bend 对应偶数长度的 Nyquist 索引 N/2。>
        if ( ( i == 0 ) || ( i == bend ) ) {
            h = 1.0;
        // <学习注释：位于 DC 与 Nyquist 之间的严格正频率乘 2，以补偿被删除的负频率。>
        } else if ( i > 0 && i < bend ) {
            h = 2.0;
        // <学习注释：Nyquist 之后的 DFT 索引代表负频率，因此置零。>
        } else {
            h = 0.0;
        }
    // <学习注释：odd 为 true 时 N 为奇数，不存在单独的 Nyquist bin。>
    } else {
        // <学习注释：奇数长度仍保留 DC 分量。>
        if ( i == 0 ) {
            h = 1.0;
        // <学习注释：索引 1 到 (N-1)/2 是严格正频率，均乘 2。>
        } else if ( i > 0 && i < bend) {
            h = 2.0;
        // <学习注释：其余索引对应负频率，均置零。>
        } else {
            h = 0.0;
        }
    }
    """,
    # <学习注释：为生成的 CuPy kernel 指定可缓存和诊断的名称 _hilbert_kernel。>
    "_hilbert_kernel",
    # <学习注释：要求运行时编译器按 C++11 解析 kernel 代码。>
    options=("-std=c++11",),
    # <学习注释：loop_prep 在逐元素循环前计算一次 N 的奇偶性和正频率边界 bend。>
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)


# <学习注释：定义公开 API；x 是实值输入，N 是 FFT 长度，axis 指定逐条执行一维变换的轴。>
def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``cp.imag(hilbert(x))``, and the
    original signal from ``cp.real(hilbert(x))``.

    Examples
    ---------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.

    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> from cusignal import hilbert, chirp

    >>> duration = 1.0
    >>> fs = 400.0
    >>> samples = int(fs*duration)
    >>> t = cp.arange(samples) / fs

    We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation.

    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * cp.sin(2.0*cp.pi*3.0*t) )

    The amplitude envelope is given by magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.

    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = cp.abs(analytic_signal)
    >>> instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))
    >>> instantaneous_frequency = (cp.diff(instantaneous_phase) /
    ...                            (2.0*cp.pi) * fs)

    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(signal), label='signal')
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(amplitude_envelope), \
        label='envelope')
    >>> ax0.set_xlabel("time in seconds")
    >>> ax0.legend()
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t[1:], instantaneous_frequency)
    >>> ax1.set_xlabel("time in seconds")
    >>> ax1.set_ylim(0.0, 120.0)

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    """
    # <学习注释：把 array-like 输入转换为 CuPy 数组；已有 CuPy 数组通常不会发生不必要复制。>
    x = cp.asarray(x)
    # <学习注释：解析信号构造依赖实信号频谱的共轭对称，因此拒绝复数输入。>
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
    # <学习注释：未提供 N 时，FFT 长度默认等于目标轴的现有长度。>
    if N is None:
        N = x.shape[axis]
    # <学习注释：FFT 长度必须是正整数语义；这里显式拒绝零和负数。>
    if N <= 0:
        raise ValueError("N must be positive.")

    # <学习注释：沿 axis 做 N 点 FFT；N 较大时尾部补零，较小时截断。>
    Xf = cp.fft.fft(x, N, axis=axis)

    # <学习注释：分配长度 N 的实值频域掩码，dtype 跟随输入以供 elementwise kernel 写入。>
    h = cp.empty((N,), dtype=x.dtype)
    # <学习注释：并行填充 h，使其包含 DC/Nyquist 的 1、正频率的 2 和负频率的 0。>
    _hilbert_kernel(h)

    # <学习注释：多维输入需要把一维 h 改造成仅在 axis 维非单例的广播形状。>
    if x.ndim > 1:
        # <学习注释：先为每个输入维度放置一个 newaxis，形成全单例索引模板。>
        ind = [cp.newaxis] * x.ndim
        # <学习注释：目标 axis 使用完整切片，使该维长度为 N。>
        ind[axis] = slice(None)
        # <学习注释：应用索引后 h 的形状可与 Xf 沿非目标轴广播。>
        h = h[tuple(ind)]
    # <学习注释：频谱逐点乘掩码后做 IFFT，得到实部为输入、虚部为 Hilbert 变换的解析信号。>
    x = cp.fft.ifft(Xf * h, axis=axis)
    # <学习注释：返回复值 CuPy 数组，其目标轴长度为 N。>
    return x


# <学习注释：定义逐元素 GPU kernel，用同一套索引规则生成两个一维 Hilbert 频域掩膜。>
_hilbert2_kernel = cp.ElementwiseKernel(
    # <学习注释：空字符串表示 kernel 没有输入数组。>
    "",
    # <学习注释：声明两个泛型输出数组 h1 和 h2，T 由 CuPy 调用时推断。>
    "T h1, T h2",
    """
    // <学习注释：odd 由 loop_prep 预先计算；为 false 时表示掩膜长度为偶数。>
    if ( !odd ) {
        // <学习注释：偶数长度时索引 0 是 DC，索引 bend=N/2 是 Nyquist 箱。>
        if ( ( i == 0 ) || ( i == bend ) ) {
            // <学习注释：DC 与 Nyquist 都没有不同的共轭伙伴，因此 h1 权重保持 1。>
            h1 = 1.0;
            // <学习注释：h2 使用与 h1 相同的一维掩膜规则。>
            h2 = 1.0;
        // <学习注释：索引 1 到 N/2-1 对应 FFT 未移位排列中的严格正频率。>
        } else if ( i > 0 && i < bend ) {
            // <学习注释：正频率乘 2，以补偿被清零的负频率共轭伙伴。>
            h1 = 2.0;
            // <学习注释：第二个输出掩膜的正频率同样乘 2。>
            h2 = 2.0;
        // <学习注释：剩余索引属于 FFT 排列中的负频率。>
        } else {
            // <学习注释：清零 h1 的负频率箱。>
            h1 = 0.0;
            // <学习注释：清零 h2 的负频率箱。>
            h2 = 0.0;
        }
    // <学习注释：odd 为 true 时表示掩膜长度为奇数，没有独立 Nyquist 箱。>
    } else {
        // <学习注释：奇数长度只有索引 0 是直流箱。>
        if ( i == 0 ) {
            // <学习注释：h1 的直流权重保持 1。>
            h1 = 1.0;
            // <学习注释：h2 的直流权重保持 1。>
            h2 = 1.0;
        // <学习注释：索引 1 到 (N-1)/2 是奇数长度 DFT 的正频率箱。>
        } else if ( i > 0 && i < bend) {
            // <学习注释：h1 的正频率权重设为 2。>
            h1 = 2.0;
            // <学习注释：h2 的正频率权重设为 2。>
            h2 = 2.0;
        // <学习注释：其余索引是奇数长度 DFT 的负频率箱。>
        } else {
            // <学习注释：清零 h1 的负频率。>
            h1 = 0.0;
            // <学习注释：清零 h2 的负频率。>
            h2 = 0.0;
        }
    }
    """,
    # <学习注释：为生成的 CuPy kernel 指定内部名称。>
    "_hilbert2_kernel",
    # <学习注释：要求后端以 C++11 语法编译 kernel。>
    options=("-std=c++11",),
    # <学习注释：在逐元素循环前计算长度奇偶性，按位与 1 可判断长度是否为奇数。>
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)


# <学习注释：定义公开算子 hilbert2，x 是实二维输入，N 控制二维 FFT 输出尺寸。>
def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    # <学习注释：把标量或一维输入提升到至少二维；已有二维数组保持二维。>
    x = cp.atleast_2d(x)
    # <学习注释：提升后若维数仍超过 2，则拒绝三维及更高维输入。>
    if x.ndim > 2:
        # <学习注释：抛出 ValueError 明确限制输入只能是二维。>
        raise ValueError("x must be 2-D.")
    # <学习注释：解析信号构造依赖实输入频谱的 Hermitian 对称性，因此拒绝复输入。>
    if cp.iscomplexobj(x):
        # <学习注释：复数输入不满足此 API 的输入契约。>
        raise ValueError("x must be real.")
    # <学习注释：未指定 N 时使用提升后输入数组的两轴长度。>
    if N is None:
        # <学习注释：x.shape 是形如 (N0, N1) 的二元组。>
        N = x.shape
    # <学习注释：单个整数 N 表示两个变换轴都采用相同长度。>
    elif isinstance(N, int):
        # <学习注释：FFT 长度必须为正。>
        if N <= 0:
            # <学习注释：零或负长度不能用于 FFT。>
            raise ValueError("N must be positive.")
        # <学习注释：把标量 N 扩展成方形输出尺寸 (N, N)。>
        N = (N, N)
    # <学习注释：序列形式必须恰含两个元素，且每个元素都应大于零。>
    elif len(N) != 2 or cp.any(cp.asarray(N) <= 0):
        # <学习注释：错误信息说明 tuple 形式的长度和正值要求。>
        raise ValueError(
            # <学习注释：这是传给 ValueError 的完整文本参数。>
            "When given as a tuple, N must hold exactly two positive integers"
        )

    # <学习注释：沿第 0、1 轴计算尺寸为 N 的二维 FFT；较短尺寸截断，较长尺寸补零。>
    Xf = cp.fft.fft2(x, N, axes=(0, 1))

    # <学习注释：由 x 的前两行 dtype 推导掩膜元素 dtype；第 0 轴不足两行时 x[1] 存在越界风险。>
    elements_dtype = cp.result_type(x[0].dtype, x[1].dtype)
    # <学习注释：分配第一个一维掩膜；源码使用 N[1]，理论上第 0 轴掩膜应为 N[0]，非方阵存在形状风险。>
    h1 = cp.empty((N[1],), dtype=elements_dtype)
    # <学习注释：分配第二个长度为 N[1] 的一维掩膜，对应第 1 轴。>
    h2 = cp.empty((N[1],), dtype=elements_dtype)
    # <学习注释：启动逐元素 kernel，同时填充 h1 与 h2。>
    _hilbert2_kernel(h1, h2)

    # <学习注释：通过广播计算列向量 h1 与行向量 h2 的外积，形成二维单象限掩膜。>
    h = h1[:, cp.newaxis] * h2[cp.newaxis, :]
    # <学习注释：保存输入维数；由于前面强制 x 至多二维，这里 k 实际恒为 2。>
    k = x.ndim
    # <学习注释：该循环为更高维广播预留，但当前维数检查使条件永远为 false。>
    while k > 2:
        # <学习注释：若能进入循环，就在末尾增加一个长度为 1 的广播轴。>
        h = h[:, cp.newaxis]
        # <学习注释：每次循环把计数减 1，直至二维。>
        k -= 1
    # <学习注释：频谱乘单象限掩膜后沿两轴做逆 FFT，得到复解析信号。>
    x = cp.fft.ifft2(Xf * h, axes=(0, 1))
    # <学习注释：返回 shape 由 N 决定的复数 CuPy 数组。>
    return x


_detrend_A_kernel = cp.ElementwiseKernel(
    "",
    "float64 A",
    """
    if ( i & 1 ) {
        const int new_i { i >> 1 };
        A = new_i * den;
    } else {
        A = 1.0;
    }
    """,
    "_detrend_A_kernel",
    options=("-std=c++11",),
    loop_prep="const double den { 1.0 / _ind.size() };",
)


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


def channelize_poly(x, h, n_chans):
    """
    Polyphase channelize signal into n channels

    Parameters
    ----------
    x : array_like
        The input data to be channelized
    h : array_like
        The 1-D input filter; will be split into n
        channels of int number of taps
    n_chans : int
        Number of channels for channelizer

    Returns
    -------
    yy : channelized output matrix

    Notes
    -----
    Currently only supports simple channelizer where channel
    spacing is equivalent to the number of channels used (zero overlap).
    Number of filter taps (len of filter / n_chans) must be <=32.

    """
    dtype = cp.promote_types(x.dtype, h.dtype)

    x = cp.asarray(x, dtype=dtype)
    h = cp.asarray(h, dtype=dtype)

    # number of taps in each h_n filter
    n_taps = int(len(h) / n_chans)
    if n_taps > 32:
        raise NotImplementedError(
            "The number of calculated taps ({}) in  \
            each filter is currently capped at 32. Please reduce filter \
                length or number of channels".format(
                n_taps
            )
        )

    # number of outputs
    n_pts = int(len(x) / n_chans)

    if x.dtype == np.float32 or x.dtype == np.complex64:
        y = cp.empty((n_pts, n_chans), dtype=cp.complex64)
    elif x.dtype == np.float64 or x.dtype == np.complex128:
        y = cp.empty((n_pts, n_chans), dtype=cp.complex128)
    else:
        raise NotImplementedError("Data type ({}) not allowed.".format(x.dtype))

    _channelizer(x, h, y, n_chans, n_taps, n_pts)

    return cp.conj(cp.fft.fft(y)).T


def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than cp.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product
