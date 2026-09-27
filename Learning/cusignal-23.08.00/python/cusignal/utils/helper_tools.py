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

import os
from pathlib import Path

import cupy as cp


def _get_numSM():

    device_id = cp.cuda.Device()

    return device_id.attributes["MultiProcessorCount"]


def _get_max_smem():

    # <学习注释：取得当前 CuPy CUDA device，用于查询硬件属性。>
    device_id = cp.cuda.Device()

    # <学习注释：返回单个 thread block 可用的最大共享内存字节数。>
    return device_id.attributes["MaxSharedMemoryPerBlock"]


def _get_max_tpb():

    # <学习注释：取得当前 device，后续读取每 block 最大线程数。>
    device_id = cp.cuda.Device()

    # <学习注释：sosfilt 采用每节一个线程，因此该上限约束 n_sections。>
    return device_id.attributes["MaxThreadsPerBlock"]


def _get_max_gdx():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxGridDimX"]


def _get_max_gdy():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxGridDimY"]


def _get_tpb_bpg():

    numSM = _get_numSM()
    threadsperblock = 512
    blockspergrid = numSM * 20

    return threadsperblock, blockspergrid


# <学习注释：从随包分发的 fatbin 文件加载指定 CUDA 导出符号。>
def _get_function(fatbin, func):

    dir = os.path.dirname(Path(__file__).parent)

# <学习注释：CuPy RawModule 直接从二进制模块路径创建可查询的 CUDA module。>
    module = cp.RawModule(
        path=dir + fatbin,
    )
# <学习注释：按字符串符号名取得 RawKernel 并返回给缓存层。>
    return module.get_function(func)


# <学习注释：仅在开发调试环境变量开启时打印 kernel 属性，不参与数值计算。>
def _print_atts(func):
    if os.environ.get("CUSIGNAL_DEV_DEBUG") == "True":
        print("name:", func.kernel.name)
        print("max_threads_per_block:", func.kernel.max_threads_per_block)
        print("num_regs:", func.kernel.num_regs)
        print(
            "max_dynamic_shared_size_bytes:",
            func.kernel.max_dynamic_shared_size_bytes,
        )
        print("shared_size_bytes:", func.kernel.shared_size_bytes)
        print(
            "preferred_shared_memory_carveout:",
            func.kernel.preferred_shared_memory_carveout,
        )
        print("const_size_bytes:", func.kernel.const_size_bytes)
        print("local_size_bytes:", func.kernel.local_size_bytes)
        print("ptx_version:", func.kernel.ptx_version)
        print("binary_version:", func.kernel.binary_version)
        print()
