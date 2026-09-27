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

import timeit

import cupy as cp
import numpy as np

# <学习注释：FULL 的内部整数编码为 2，与 _modedict 和后端分支保持一致。>
FULL = 2
# <学习注释：SAME 的内部整数编码为 1。>
SAME = 1
# <学习注释：VALID 的内部整数编码为 0。>
VALID = 0

# <学习注释：周期边界在左移两位后的编码为 8。>
CIRCULAR = 8
# <学习注释：对称边界在左移两位后的编码为 4。>
REFLECT = 4
# <学习注释：常数填充边界编码为 0。>
PAD = 0

# <学习注释：把公开 mode 字符串映射为后端使用的 0、1、2。>
_modedict = {"valid": 0, "same": 1, "full": 2}

# <学习注释：边界字符串先映射为基础值，_bvalfromboundary 再统一左移两位。>
_boundarydict = {
    "fill": 0,
    "pad": 0,
    "wrap": 2,
    "circular": 2,
    "symm": 1,
    "symmetric": 1,
    "reflect": 4,
}


# <学习注释：定义输入交换判定 helper；correlate2d 只在 valid 模式依赖它。>
def _inputs_swap_needed(mode, shape1, shape2):
    """
    If in 'valid' mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every dimension.

    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.

    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
# <学习注释：非 valid 模式不需要逐维包含关系，只有 valid 才进入尺寸检查。>
    if mode == "valid":
# <学习注释：ok1 表示 shape1 逐维不小于 shape2，ok2 表示相反关系，初值都假定成立。>
        ok1, ok2 = True, True

# <学习注释：zip 按维配对两个 shape；correlate2d 已保证二者维数同为 2。>
        for d1, d2 in zip(shape1, shape2):
# <学习注释：若当前维度 d1 小于 d2，shape1 不能完全包住 shape2。>
            if not d1 >= d2:
# <学习注释：记录第一种逐维包含关系已经失败。>
                ok1 = False
# <学习注释：若当前维度 d2 小于 d1，shape2 不能完全包住 shape1。>
            if not d2 >= d1:
# <学习注释：记录第二种逐维包含关系已经失败。>
                ok2 = False

# <学习注释：两种包含关系都失败，说明两个 shape 在不同维度各有长短，无法形成矩形 valid 区域。>
        if not (ok1 or ok2):
# <学习注释：构造并抛出 ValueError；相邻字符串字面量由 Python 自动拼接。>
            raise ValueError(
# <学习注释：错误消息第一段说明 valid 模式的约束。>
                "For 'valid' mode, one must be at least "
# <学习注释：错误消息第二段说明较大者必须在每个维度都不小于另一者。>
                "as large as the other in every dimension"
# <学习注释：右括号结束 ValueError 调用。>
            )

# <学习注释：若 shape1 不能包住 shape2，则 ok2 必为真，此时返回 True 要求交换输入。>
        return not ok1

# <学习注释：full、same 或其他字符串都返回 False；模式合法性由后端的 _valfrommode 检查。>
    return False


def _numeric_arrays(arrays, kinds="buifc"):
    """
    See if a list of arrays are all numeric.

    Parameters
    ----------
    ndarrays : array or list of arrays
        arrays to check if numeric.
    numeric_kinds : string-like
        The dtypes of the arrays to be checked. If the dtype.kind of
        the ndarrays are not in this string the function returns False and
        otherwise returns True.
    """
    if type(arrays) == cp.ndarray:
        return arrays.dtype.kind in kinds
    for array_ in arrays:
        if array_.dtype.kind not in kinds:
            return False
    return True


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = arr.shape
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than np.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product


def _fftconv_faster(x, h, mode):
    """
    See if using `fftconvolve` or `_correlateND` is faster. The boolean value
    returned depends on the sizes and shapes of the input values.

    The big O ratios were found to hold across different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    if mode == "full":
        out_shape = [n + k - 1 for n, k in zip(x.shape, h.shape)]
        big_O_constant = 10963.92823819 if x.ndim == 1 else 8899.1104874
    elif mode == "same":
        out_shape = x.shape
        if x.ndim == 1:
            if h.size <= x.size:
                big_O_constant = 7183.41306773
            else:
                big_O_constant = 856.78174111
        else:
            big_O_constant = 34519.21021589
    elif mode == "valid":
        out_shape = [n - k + 1 for n, k in zip(x.shape, h.shape)]
        big_O_constant = 41954.28006344 if x.ndim == 1 else 66453.24316434
    else:
        raise ValueError(
            "Acceptable mode flags are \
                         'valid',"
            " 'same', or 'full'."
        )

    # see whether the Fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    direct_time = x.size * h.size * _prod(out_shape)
    fft_time = sum(n * np.log(n) for n in (x.shape + h.shape + tuple(out_shape)))

    return big_O_constant * fft_time < direct_time


def _timeit_fast(stmt="pass", setup="pass", repeat=3):
    """
    Returns the time the statement/function took, in seconds.

    Faster, less precise version of IPython's timeit. `stmt` can be a statement
    written as a string or a callable.

    Will do only 1 loop (like IPython's timeit) with no repetitions
    (unlike IPython) for very slow functions.  For fast functions, only does
    enough loops to take 5 ms, which seems to produce similar results (on
    Windows at least), and avoids doing an extraneous cycle that isn't
    measured.

    """
    timer = timeit.Timer(stmt, setup)

    # determine number of calls per rep so total time for 1 rep >= 5 ms
    x = 0
    for p in range(0, 10):
        number = 10**p
        x = timer.timeit(number)  # seconds
        if x >= 5e-3 / 10:  # 5 ms for final test, 1/10th that for this one
            break
    if x > 1:  # second
        # If it's macroscopic, don't bother with repetitions
        best = x
    else:
        number *= 10
        r = timer.repeat(repeat, number)
        best = min(r)

    sec = best / number
    return sec


# <学习注释：把公开输出模式字符串转成后端整数，非法键转换为面向用户的 ValueError。>
def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")


# <学习注释：把公开边界字符串转成 PAD、REFLECT 或 CIRCULAR 的内部编码。>
def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError:
        raise ValueError(
            "Acceptable boundary flags are 'fill', 'circular' "
            "(or 'wrap'), and 'symmetric' (or 'symm')."
        )


# <学习注释：计算正整数 a/b 的向上取整，用于让 CUDA grid 覆盖所有输出元素。>
def _iDivUp(a, b):
    return (a // b + 1) if (a % b != 0) else (a // b)


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    # <学习注释：为 x 的每个维度构造步长为 -1 的切片，即逐维反转。>
    reverse = (slice(None, None, -1),) * x.ndim
    # return cp.flip(x, 0)
    # <学习注释：先用切片反转所有轴，再逐元素取复共轭，得到相关所需的 y^*[-n]。>
    return x[reverse].conj()
