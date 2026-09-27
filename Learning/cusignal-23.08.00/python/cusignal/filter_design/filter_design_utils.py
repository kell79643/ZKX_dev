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

import numpy as np


def _validate_sos(sos):
    # <学习注释：该 helper 只校验并规范化 SOS 系数矩阵，不执行滤波递推。>
    """Helper to validate a SOS input"""
    # <学习注释：np.atleast_2d 保证一维输入至少具有“节数×系数”两个轴。>
    sos = np.atleast_2d(sos)
    # <学习注释：高于二维的输入仍不符合每行一个二阶节的接口约定。>
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    # <学习注释：n_sections 是二阶节数量，m 是每节系数个数。>
    n_sections, m = sos.shape
    # <学习注释：每节必须依次提供 b0、b1、b2、a0、a1、a2 六个系数。>
    if m != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    # <学习注释：当前递推公式省略除以 a0，因此强制每节 a0 即第 4 列为 1。>
    if not (sos[:, 3] == 1).all():
        raise ValueError("sos[:, 3] should be all ones")
    # <学习注释：同时返回规范化后的二维数组和节数，供调用者构造状态 shape。>
    return sos, n_sections
