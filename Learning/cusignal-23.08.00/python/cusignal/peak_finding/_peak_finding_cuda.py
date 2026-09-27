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

from ..convolution.convolution_utils import _iDivUp
from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _get_tpb_bpg, _print_atts

# <学习注释：把六个受支持的 CuPy 比较器对象映射为整数操作码，CUDA kernel 用该操作码选择 device 函数。>
_modedict = {
    cp.less: 0,
    cp.greater: 1,
    cp.less_equal: 2,
    cp.greater_equal: 3,
    cp.equal: 4,
    cp.not_equal: 5,
}

# <学习注释：预编译一、二维 kernel 只为以下四种输入 dtype 提供入口。>
_SUPPORTED_TYPES = [
    "int32",
    "int64",
    "float32",
    "float64",
]


class _cupy_boolrelextrema_1d_wrapper(object):
    # <学习注释：构造器保存本次 kernel 启动使用的 grid、block 和已加载 kernel 句柄。>
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
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):

        # <学习注释：一维 kernel 参数依次是轴长、邻域阶数、边界标志、比较器操作码、输入和输出。>
        kernel_args = (data.shape[axis], order, clip, comp, data, out)

        # <学习注释：以保存的 grid 和 block 启动 RawModule 中的 CUDA kernel。>
        self.kernel(self.grid, self.block, kernel_args)


class _cupy_boolrelextrema_2d_wrapper(object):
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
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):

        # <学习注释：二维 kernel 还需要列数、行数和 axis，以便把二维坐标换算为线性地址。>
        kernel_args = (
            data.shape[1],
            data.shape[0],
            order,
            clip,
            comp,
            axis,
            data,
            out,
        )

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    # <学习注释：若 dtype 没有预编译实现，立即拒绝，避免用错误的二进制入口解释数据。>
    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    # <学习注释：同一 dtype 与 kernel 类型已缓存时不重复加载 fatbin。>
    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    # <学习注释：从预编译 fatbin 按符号名取得函数，并以 dtype、kernel 类型二元组作为 cache key。>
    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/peak_finding/_peak_finding.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    # <学习注释：从全局 cache 取得与 dtype 和一/二维类型对应的底层 kernel 句柄。>
    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "boolrelextrema_1D":
            return _cupy_boolrelextrema_1d_wrapper(grid, block, kernel)
        else:
            return _cupy_boolrelextrema_2d_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))


def _peak_finding(data, comparator, axis, order, mode, results):

    # <学习注释：字典查找把 Python 比较器变成 device 操作码；未登记的比较器会在这里失败。>
    comp = _modedict[comparator]

    # <学习注释：只有字符串 clip 映射为端点截断；其他模式在 kernel 中都走回绕分支。>
    if mode == "clip":
        clip = True
    else:
        clip = False

    # <学习注释：一维数据选择一维 kernel 和通用的一维线程配置。>
    if data.ndim == 1:
        k_type = "boolrelextrema_1D"

        threadsperblock, blockspergrid = _get_tpb_bpg()

        # <学习注释：确保对应 dtype 的一维 fatbin 入口已经放入 cache。>
        _populate_kernel_cache(data.dtype, k_type)

        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    else:
        # <学习注释：调用方只会把二维数据送到这里，因此 else 选择二维 kernel。>
        k_type = "boolrelextrema_2D"

        # <学习注释：二维 block 使用 16×16 共 256 个线程。>
        threadsperblock = (16, 16)
        # <学习注释：两个 grid 维度分别对列数和行数向上取整，覆盖整个二维输入。>
        blockspergrid = (
            _iDivUp(data.shape[1], threadsperblock[0]),
            _iDivUp(data.shape[0], threadsperblock[1]),
        )

        _populate_kernel_cache(data.dtype, k_type)

        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    # <学习注释：调用包装器组装参数并启动一维或二维 CUDA kernel，把布尔结果写入 results。>
    kernel(data, comp, axis, order, clip, results)

    # <学习注释：调试辅助函数可输出 kernel 的 grid、block 等属性，不参与极值计算。>
    _print_atts(kernel)
