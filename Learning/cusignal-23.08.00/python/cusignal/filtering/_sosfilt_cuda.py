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

from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _print_atts

_SUPPORTED_TYPES = ["float32", "float64"]


class _cupy_sosfilt_wrapper(object):
    # <学习注释：该对象封装 kernel 的 grid、block、动态共享内存字节数和函数句柄。>
    def __init__(self, grid, block, smem, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.smem = smem
        self.kernel = kernel

    def __call__(self, sos, x, zi):

        # <学习注释：按 CUDA kernel 形参顺序组装信号数、样本数、节数、状态宽度和三个数组。>
        kernel_args = (
            x.shape[0],
            x.shape[1],
            sos.shape[0],
            zi.shape[2],
            sos,
            zi,
            x,
        )

        # <学习注释：RawKernel 调用使用已保存的 launch 配置和动态共享内存大小。>
        self.kernel(self.grid, self.block, kernel_args, shared_mem=self.smem)


def _populate_kernel_cache(np_type, k_type):

    # <学习注释：预编译 fatbin 只提供 float32 和 float64 两个入口。>
    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    # <学习注释：相同 dtype 与 kernel 类型已加载时直接复用缓存。>
    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    # <学习注释：从 fatbin 取出形如 _cupy_sosfilt_float32 的导出函数并缓存。>
    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_sosfilt.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, smem, k_type):
    # <学习注释：用 dtype 名和算子名定位已缓存的 CUDA 函数。>
    kernel = _cupy_kernel_cache[(dtype.name, k_type)]
    if kernel:
        return _cupy_sosfilt_wrapper(grid, block, smem, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))

    raise NotImplementedError("No kernel found for datatype {}".format(dtype.name))


def _sosfilt(sos, x, zi):

    # <学习注释：一个 block 内每个线程负责一个二阶节。>
    threadsperblock = sos.shape[0]  # Up-to (1024, 1) = 1024 max per block
    # <学习注释：每条展平后的独立信号分配一个 block。>
    blockspergrid = x.shape[0]

    k_type = "sosfilt"

    # <学习注释：确保与输入 dtype 对应的预编译 kernel 已进入缓存。>
    _populate_kernel_cache(x.dtype, k_type)

    out_size = threadsperblock
    sos_size = sos.shape[0] * sos.shape[1]

    # <学习注释：动态共享内存包括 n_sections 个节间输出和 n_sections×6 个系数。>
    shared_mem = (out_size + sos_size) * x.dtype.itemsize

    # <学习注释：按本次 shape 和共享内存需求构造可调用包装器。>
    kernel = _get_backend_kernel(
        x.dtype,
        blockspergrid,
        threadsperblock,
        shared_mem,
        k_type,
    )
    # <学习注释：该原始调试输出会在每次调用时打印 zi shape，不参与算法。>
    print(zi.shape)

    # <学习注释：启动 kernel；返回值通过 x（以及设计意图中的 zi）原地写回。>
    kernel(sos, x, zi)

    # <学习注释：打印 launch 属性属于诊断副作用，不属于滤波数学。>
    _print_atts(kernel)
