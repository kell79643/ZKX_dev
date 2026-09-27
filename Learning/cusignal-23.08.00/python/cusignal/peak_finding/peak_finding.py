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

# cuSignal does not support cupy.take(mode='foo')

import cupy as cp

from ._peak_finding_cuda import _peak_finding


def _boolrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin
    """
    # <学习注释：先把 order 转成 int 再与原值比较，用来拒绝 1.5 之类的非整数；同时要求每侧至少比较一个样本。>
    if (int(order) != order) or (order < 1):
        # <学习注释：非法邻域阶数立即抛出 ValueError，后续索引循环不会执行。>
        raise ValueError("Order must be an int >= 1")

    # <学习注释：一维和二维数组走预编译 CUDA kernel；三维及以上数组走后面的 CuPy 向量化循环。>
    if data.ndim < 3:
        # <学习注释：结果掩码与输入 shape 相同，每个布尔元素表示对应位置是否通过全部邻域比较。>
        results = cp.empty(data.shape, dtype=bool)
        # <学习注释：把数据、比较器、轴、邻域阶数和边界模式交给 GPU 后端填充 results。>
        _peak_finding(data, comparator, axis, order, mode, results)
    else:
        # <学习注释：datalen 是待检测轴的长度，后续正负偏移都相对于这个轴生成。>
        datalen = data.shape[axis]
        # <学习注释：locs 依次包含待检测轴上的全部合法中心索引 0 到 datalen-1。>
        locs = cp.arange(0, datalen)
        # <学习注释：先假设所有位置都是候选极值，再用每次比较的布尔结果逐步淘汰。>
        results = cp.ones(data.shape, dtype=bool)
        # <学习注释：main 沿指定轴取全部中心值；其 shape 与 data 相同。>
        main = cp.take(data, locs, axis=axis)
        # <学习注释：shift 从 1 遍历到 order，对应数学公式中每侧的邻居距离 k。>
        for shift in cp.arange(1, order + 1):
            # <学习注释：clip 模式把越界正负索引截到首尾端点。>
            if mode == "clip":
                p_locs = cp.clip(locs + shift, a_min=None, a_max=(datalen - 1))
                m_locs = cp.clip(locs - shift, a_min=0, a_max=None)
            else:
                # <学习注释：非 clip 模式保留越界整数索引，依赖 CuPy 整数数组索引的回绕语义。>
                p_locs = locs + shift
                m_locs = locs - shift
            # <学习注释：plus 和 minus 分别保存沿 axis 距中心 +shift 与 -shift 的邻居数组。>
            plus = cp.take(data, p_locs, axis=axis)
            minus = cp.take(data, m_locs, axis=axis)
            # <学习注释：中心必须同时通过正向邻居比较；&= 把本轮结果累积到此前全部偏移的结果中。>
            results &= comparator(main, plus)
            # <学习注释：中心还必须通过负向邻居比较，由此落实“两侧所有邻居均满足 comparator”的全称条件。>
            results &= comparator(main, minus)

            # <学习注释：如果当前已没有任何候选，继续比较更远邻居也不可能把 False 恢复为 True，因此可以提前返回。>
            if ~results.any():
                return results

    # <学习注释：helper 返回布尔掩码；坐标元组的转换由公开函数 argrelextrema 完成。>
    return results


def argrelmin(data, axis=0, order=1, mode="clip"):
    """
    Calculate the relative minima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative minima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the minima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelextrema, argrelmax, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with cp.less as comparator. Therefore it
    requires a strict inequality on both sides of a value to consider it a
    minimum. This means flat minima (more than one sample wide) are not
    detected. In case of one-dimensional `data` `find_peaks` can be used to
    detect all local minima, including flat ones, by calling it with negated
    `data`.

    Examples
    --------
    >>> from cusignal import argrelmin
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmin(x)
    (array([1, 5]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cp.asarray(data)
    return argrelextrema(data, cp.less, axis, order, mode)


def argrelmax(data, axis=0, order=1, mode="clip"):
    """
    Calculate the relative maxima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative maxima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelextrema, argrelmin, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with cp.greater as comparator. Therefore
    it  requires a strict inequality on both sides of a value to consider it a
    maximum. This means flat maxima (more than one sample wide) are not
    detected. In case of one-dimensional `data` `find_peaks` can be used to
    detect all local maxima, including flat ones.

    Examples
    --------
    >>> from cusignal import argrelmax
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmax(x)
    (array([3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0]), array([1]))
    """
    data = cp.asarray(data)
    return argrelextrema(data, cp.greater, axis, order, mode)


def argrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelmin, argrelmax

    Examples
    --------
    >>> from cusignal import argrelextrema
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, cp.greater)
    (array([3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cp.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    # <学习注释：把列表、NumPy 数组或已有 CuPy 数组统一转换成 GPU 侧 CuPy ndarray。>
    data = cp.asarray(data)
    # <学习注释：调用直接 helper 生成与 data 同 shape 的极值候选布尔掩码。>
    results = _boolrelextrema(data, comparator, axis, order, mode)

    # <学习注释：cuSignal 这份实现不支持 raise 边界模式，因此显式拒绝该取值。>
    if mode == "raise":
        raise NotImplementedError("CuPy `take` doesn't support `mode='raise'`.")

    # <学习注释：cp.nonzero 把 True 元素转换为坐标数组元组；一维输入也返回单元素元组。>
    return cp.nonzero(results)
