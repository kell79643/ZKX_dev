"""Task1 在 ZQ500 上使用 cuSignal 窗函数语义的最小 FP32 适配。"""

from __future__ import annotations

from task1_common import cp


_hamming_fp32_kernel = cp.ElementwiseKernel(
    "int32 length",
    "float32 window",
    """
    if (length <= 1) {
        window = 1.0f;
    } else {
        const float phase = 2.0f * 3.14159265358979323846f * (float)i / (float)(length - 1);
        window = 0.54f - 0.46f * cosf(phase);
    }
    """,
    "task1_hamming_fp32_adapter",
    options=("-std=c++11",),
)


def task1_hamming_window_fp32(length: int):
    """生成与 ``cusignal.hamming(length, sym=True)`` 同公式的 FP32 窗。

    cuSignal 23.08.00 的原版 ``hamming`` kernel 固定输出 ``float64``，会把
    Task1 的 ``complex64`` 慢时间输入提升到 ``complex128`` 并触发 ZQ500
    不支持的 Z2Z FFT。本函数只适配该 dtype 边界，不实现 pulse Doppler 主计算。
    """

    size = int(length)
    if size < 1:
        raise ValueError("Task1 Hamming length must be positive")
    return _hamming_fp32_kernel(cp.int32(size), size=size)
