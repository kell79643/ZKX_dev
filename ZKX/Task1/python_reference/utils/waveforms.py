"""cuSignal 23.08.00 与当前 NumPy 的最小调用边界适配。"""

from __future__ import annotations

import numpy as np

from task1_common import cp, cusignal


def ensure_numpy_issubclass_compat() -> None:
    """恢复 cuSignal 23.08.00 使用、NumPy 2.x 已移除的类型判断别名。

    该函数不生成波形，也不替代 ``cusignal.chirp``。它只在缺失时把上游所需的
    ``np.issubclass_`` 语义映射到 ``np.issubdtype``。
    """

    if not hasattr(np, "issubclass_"):
        np.issubclass_ = np.issubdtype  # type: ignore[attr-defined]


def ensure_zq500_chirp_kernel_compat() -> None:
    """为官方 ``cusignal.chirp`` 安装等公式的 FP32 显式窄化边界。

    上游 kernel 使用 ``M_PI`` 和双精度字面量初始化模板类型 ``T``。NVIDIA nvrtc
    接受该隐式窄化，ZQ500 clang 驱动会拒绝。这里仅把常量显式转换为 ``T``，随后仍由
    原版 ``cusignal.chirp`` 完成参数处理、输出分配和 kernel 调用。
    """

    from cusignal.waveforms import waveforms as upstream_waveforms

    if getattr(upstream_waveforms, "_task1_zq500_chirp_adapter", False):
        return
    upstream_waveforms._chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "Y phase",
        """
        const T beta { (f1 - f0) / t1 };
        const T two { static_cast<T>(2.0f) };
        const T half { static_cast<T>(0.5f) };
        const T pi { static_cast<T>(M_PI) };
        const T temp { two * pi * (f0 * t + half * beta * t * t) };
        phase = Y(
            cos(temp + phi),
            cos(temp + phi + pi / two) * static_cast<T>(-1.0f)
        );
        """,
        "task1_zq500_chirp_phase_lin_kernel",
        options=("-std=c++11",),
    )
    upstream_waveforms._task1_zq500_chirp_adapter = True


def ensure_cusignal_chirp_compat() -> None:
    """安装 cuSignal 23.08.00 在当前 NumPy/ZQ500 上所需的最小边界适配。"""

    ensure_numpy_issubclass_compat()
    ensure_zq500_chirp_kernel_compat()
