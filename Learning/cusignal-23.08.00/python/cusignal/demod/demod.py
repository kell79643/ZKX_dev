# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

# <学习注释：导入 CuPy 并命名为 cp；后续数组转换、复相角、相位展开和差分都在 GPU 数组上执行。>
import cupy as cp


# <学习注释：定义 FM 解调函数；x 是复数输入，axis 指定鉴频所沿的维度，-1 表示最后一轴。>
def fm_demod(x, axis=-1):
    """
    Demodulate Frequency Modulated Signal

    Parameters
    ----------
    x : ndarray
        Received complex valued signal or batch of signals

    Returns
    -------
    y : ndarray
        The demodulated output with the same shape as `x`.
    """

    # <学习注释：把输入转换为 CuPy ndarray；若输入已是兼容的 CuPy 数组，通常可避免不必要复制。>
    x = cp.asarray(x)

    # <学习注释：isrealobj 按数组 dtype 判断是否为实数对象；FM 相位鉴频要求输入保存复数 I/Q 或解析信号。>
    if cp.isrealobj(x):
        # <学习注释：实数输入不能直接提供唯一复相角，因此主动抛出 AssertionError 并终止函数。>
        raise AssertionError("Input signal must be complex-valued")
    # <学习注释：angle 计算主值相角，unwrap 沿 axis 通过补减 2π 消除被判定为相位包裹的跳变。>
    x_angle = cp.unwrap(cp.angle(x), axis=axis)
    # <学习注释：diff 沿 axis 计算相邻展开相位之差 y[n]=θ_u[n+1]-θ_u[n]，输出单位为 rad/sample，且该轴长度减少 1。>
    y = cp.diff(x_angle, axis=axis)
    # <学习注释：返回相位增量数组；函数未乘 fs/(2π)，也未减载频或除以频率灵敏度。>
    return y
