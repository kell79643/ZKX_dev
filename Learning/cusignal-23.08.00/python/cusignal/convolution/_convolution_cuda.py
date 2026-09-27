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

from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _get_tpb_bpg, _print_atts
from .convolution_utils import (
    CIRCULAR,
    FULL,
    PAD,
    REFLECT,
    SAME,
    VALID,
    _bvalfromboundary,
    _iDivUp,
    _valfrommode,
)

_SUPPORTED_TYPES = [
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


class _cupy_convolve_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        d_inp,
        d_kernel,
        mode,
        swapped_inputs,
        out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            mode,
            swapped_inputs,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_1d2o_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        d_inp,
        d_kernel,
        mode,
        out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            mode,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_1d3o_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        d_inp,
        d_kernel,
        mode,
        out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            d_kernel.shape[2],
            mode,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


# <学习注释：该包装类保存 CUDA grid、block 和 RawKernel，并把 Python 参数整理成底层 kernel 实参。>
class _cupy_convolve_2d_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

# <学习注释：实现可调用对象协议，使 wrapper(...) 看起来像普通函数调用。>
    def __call__(
        self,
        d_inp,
        paddedW,
        paddedH,
        d_kernel,
        S0,
        S1,
        out,
        outW,
        outH,
        pick,
    ):

# <学习注释：按 CUDA 导出函数签名的严格顺序组装参数元组。>
        kernel_args = (
            d_inp,
            paddedW,
            paddedH,
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            S0,
            S1,
            out,
            outW,
            outH,
            pick,
        )

# <学习注释：用保存的网格、线程块和参数启动 CuPy RawKernel；调用是异步 GPU 调度。>
        self.kernel(self.grid, self.block, kernel_args)


# <学习注释：按 dtype 和 kernel 类型延迟加载 fatbin 中的编译后函数并缓存。>
def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/convolution/_convolution.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


# <学习注释：从缓存取出底层 kernel，再包装成参数组织方式匹配的 Python 可调用对象。>
def _get_backend_kernel(
    dtype,
    grid,
    block,
    k_type,
):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "convolve" or k_type == "correlate":
            return _cupy_convolve_wrapper(grid, block, kernel)
        elif k_type == "convolve2D" or k_type == "correlate2D":
            return _cupy_convolve_2d_wrapper(grid, block, kernel)
        elif k_type == "convolve1D2O":
            return _cupy_convolve_1d2o_wrapper(grid, block, kernel)
        elif k_type == "convolve1D3O":
            return _cupy_convolve_1d3o_wrapper(grid, block, kernel)
        else:
            raise NotImplementedError(
                "No CuPY kernel found for k_type {}, datatype {}".format(k_type, dtype)
            )
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))


def _convolve_gpu(
    inp,
    out,
    ker,
    mode,
    use_convolve,
    swapped_inputs,
):

    # <学习注释：确保主输入位于 CuPy 设备数组中。>
    d_inp = cp.asarray(inp)
    # <学习注释：确保滑动核/模板位于 CuPy 设备数组中。>
    d_kernel = cp.asarray(ker)

    # <学习注释：获取统一的 CUDA block 和 grid 配置。>
    threadsperblock, blockspergrid = _get_tpb_bpg()

    if use_convolve:
        k_type = "convolve"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    # <学习注释：use_convolve=False 时从 fatbin 缓存中选择与 dtype 匹配的 correlate kernel。>
    else:
        k_type = "correlate"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    # <学习注释：通过 CuPy wrapper 传入数据指针、模式和交换标志并启动 CUDA kernel。>
    kernel(d_inp, d_kernel, mode, swapped_inputs, out)

    _print_atts(kernel)

    return out


# <学习注释：准备二维卷积或相关的 padding、launch geometry 和后端 kernel；correlate2d 令 use_convolve 为 0。>
def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
):

# <学习注释：后端只接受常数、对称和周期三种已经编码为整数的边界标志。>
    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")

# <学习注释：S 保存模板半径或非方形模板尺寸，用于 padding 和 kernel 索引。>
    S = np.zeros(2, dtype=int)

    # If kernel is square and odd
# <学习注释：先区分方形模板与非方形模板，pick 会把对应索引公式传给 CUDA kernel。>
    if ker.shape[0] == ker.shape[1]:  # square
        if ker.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (ker.shape[0] - 1) // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
        else:  # even
            pick = 2
            S[0] = ker.shape[0] // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
            else:  # same/valid
                if use_convolve:
                    P1 = P2 = P3 = P4 = S[0]
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
    else:  # Non-square
        pick = 3
        S[0] = ker.shape[0]
        S[1] = ker.shape[1]
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
        else:  # same/valid
            if use_convolve:
                P1 = S[0] // 2
                P2 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P3 = S[1] // 2
                P4 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
            else:
                P1 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P2 = S[0] // 2
                P3 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
                P4 = S[1] // 2

# <学习注释：same 模式先扩展第一输入，使后端每个输出线程都能读取完整模板窗口。>
    if mode == 1:  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

# <学习注释：full 模式使用更宽的 padding，以产生所有部分重叠位移。>
    if mode == 2:  # FULL
        pad = ((P1, P2), (P3, P4))
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    paddedW = inp.shape[1]
    paddedH = inp.shape[0]

    outW = out.shape[1]
    outH = out.shape[0]

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

# <学习注释：二维线程块固定为 16×16，即每个 block 最多负责 256 个输出元素。>
    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(out.shape[1], threadsperblock[0]),
        _iDivUp(out.shape[0], threadsperblock[1]),
    )

# <学习注释：同一 host helper 通过 use_convolve 在卷积和互相关 kernel 之间分派。>
    if use_convolve:
        k_type = "convolve2D"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
# <学习注释：correlate2d 传入 0，因此实际进入此分支并选择 correlate2D。>
    else:
        k_type = "correlate2D"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

# <学习注释：启动包装后的 CUDA kernel；每个有效线程计算一个 out 行列位置。>
    kernel(d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick)

    _print_atts(kernel)

    return out


def _convolve(
    in1,
    in2,
    use_convolve,
    swapped_inputs,
    mode,
):

    # <学习注释：将 valid、same、full 字符串转换为 CUDA kernel 使用的整数枚举。>
    val = _valfrommode(mode)

    # Promote inputs
    # <学习注释：计算两个输入的公共 dtype，使乘法和累加类型一致。>
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = (
                max(in1.shape[i], in2.shape[i]) - min(in1.shape[i], in2.shape[i]) + 1
            )
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            if not swapped_inputs:
                out_dimens[i] = in1.shape[i]  # Per scipy docs
            else:
                out_dimens[i] = min(in1.shape[i], in2.shape[i])
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    # <学习注释：在 GPU 上按 mode 计算得到的 shape 分配未初始化输出。>
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    # <学习注释：把输入、输出和调度标志交给底层 GPU 启动函数。>
    out = _convolve_gpu(
        in1,
        out,
        in2,
        val,
        use_convolve,
        swapped_inputs,
    )

    return out


# <学习注释：二维卷积/相关的统一 Python 后端入口，负责参数解析、dtype 和输出分配。>
def _convolve2d(in1, in2, use_convolve, mode, boundary, fillvalue):

# <学习注释：把 mode 字符串映射为 VALID、SAME 或 FULL 整数常量，非法值会抛错。>
    val = _valfrommode(mode)
# <学习注释：把 boundary 字符串映射为 PAD、REFLECT 或 CIRCULAR 整数常量。>
    bval = _bvalfromboundary(boundary)

    # Promote inputs
# <学习注释：按 CuPy 类型提升规则选择能共同表示两个输入的结果 dtype。>
    promType = cp.promote_types(in1.dtype, in2.dtype)
# <学习注释：把第一输入转换到共同 dtype；必要时在 GPU 上分配转换结果。>
    in1 = in1.astype(promType)
# <学习注释：把已共轭的第二输入转换到同一 dtype。>
    in2 = in2.astype(promType)

    if (bval != PAD) and (bval != REFLECT) and (bval != CIRCULAR):
        raise Exception("Incorrect boundary value.")

    if (bval == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
        if fill is None:
            raise Exception("fill must no be None.")
        if fill.size != 1:
            if fill.size == 0:
                raise Exception("`fillvalue` cannot be an empty array.")
            raise Exception("`fillvalue` must be scalar or an array with one element")
    else:
        fill = np.zeros(1, in1.dtype)
        if fill is None:
            raise Exception("Unable to create fill array")

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
# <学习注释：valid 输出每维长度为较大输入减模板再加一。>
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
# <学习注释：same 输出逐维复制第一输入的 shape。>
    elif val == SAME:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
# <学习注释：full 输出每维长度为两个输入长度之和减一。>
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
# <学习注释：在 GPU 上分配未初始化输出；CUDA kernel 必须覆盖每个有效元素。>
    out = cp.empty(out_dimens.tolist(), in1.dtype)

# <学习注释：进入 GPU 准备与调度层，返回的 out 与已分配数组具有相同存储。>
    out = _convolve2d_gpu(
        in1,
        out,
        in2,
        val,
        bval,
        use_convolve,
        fill,
    )

# <学习注释：把 GPU 输出数组返回给公开 correlate2d。>
    return out


def _convolve1d2o_gpu(
    inp,
    out,
    ker,
    mode,
):

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock, blockspergrid = _get_tpb_bpg()

    k_type = "convolve1D2O"

    _populate_kernel_cache(out.dtype, k_type)

    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )

    kernel(d_inp, d_kernel, mode, out)

    _print_atts(kernel)

    return out


def _convolve1d2o(in1, in2, mode):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve1d2o_gpu(
        in1,
        out,
        in2,
        val,
    )

    return out


def _convolve1d3o_gpu(
    inp,
    out,
    ker,
    mode,
):

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock, blockspergrid = _get_tpb_bpg()

    k_type = "convolve1D3O"

    _populate_kernel_cache(out.dtype, k_type)

    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )

    kernel(d_inp, d_kernel, mode, out)

    _print_atts(kernel)

    return out


def _convolve1d3o(in1, in2, mode):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve1d3o_gpu(
        in1,
        out,
        in2,
        val,
    )

    return out
