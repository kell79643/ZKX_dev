"""Task1 在 ZQ500 支持范围内使用原版 cuSignal radartools。"""

from __future__ import annotations

import math
from typing import Any

from task1_common import cp, cusignal


def _grid_index(value: float, spacing: float, center: int, size: int, name: str) -> int:
    if not math.isfinite(float(value)) or spacing <= 0.0:
        raise ValueError(f"invalid {name} cut value or spacing")
    grid_offset = float(value) / float(spacing)
    rounded = int(round(grid_offset))
    if not math.isclose(grid_offset, rounded, rel_tol=0.0, abs_tol=1.0e-5):
        raise ValueError(
            f"Task1 ZQ500 {name} cut must align to the 2D ambiguity grid"
        )
    index = center + rounded
    if index < 0 or index >= size:
        raise ValueError(f"Task1 ZQ500 {name} cut is outside the ambiguity grid")
    return index


def task1_ambgfun_2d(
    x: Any,
    fs: float,
    prf: float,
    y: Any | None = None,
) -> Any:
    """只调用未修改的 ``cusignal.ambgfun(..., cut="2d")`` 主计算。"""

    d_x = cp.asarray(x, dtype=cp.complex64)
    d_y = None if y is None else cp.asarray(y, dtype=cp.complex64)
    if d_x.ndim != 1 or d_x.size == 0:
        raise ValueError("Task1 ZQ500 ambgfun x must be a nonempty rank-1 array")
    if d_y is not None and (d_y.ndim != 1 or d_y.size != d_x.size):
        raise ValueError(
            "Task1 ZQ500 ambgfun supports autocorrelation or equal-length y"
        )
    return cusignal.ambgfun(
        d_x, fs, prf, y=d_y, cut="2d", cutValue=0
    ).astype(cp.float32, copy=False)


def task1_ambiguity_cut(
    ambiguity_2d: Any,
    x_length: int,
    fs: float,
    cut: str,
    cutValue: float = 0.0,
) -> Any:
    """从官方二维 cuSignal 结果取得网格对齐切片，不执行模糊函数主计算。"""

    normalized_cut = str(cut).lower()
    if normalized_cut not in {"delay", "doppler"}:
        raise ValueError("delay and doppler are the only adapter cut values")
    ambiguity = cp.asarray(ambiguity_2d, dtype=cp.float32)
    if ambiguity.ndim != 2:
        raise ValueError("Task1 ambiguity adapter requires a rank-2 array")
    rows, nfreq = map(int, ambiguity.shape)
    if normalized_cut == "delay":
        row = _grid_index(
            cutValue, 1.0 / float(fs), int(x_length) - 1, rows, "delay"
        )
        return cp.ascontiguousarray(ambiguity[row, :])
    column = _grid_index(
        cutValue, float(fs) / float(nfreq), nfreq // 2, nfreq, "doppler"
    )
    return cp.ascontiguousarray(ambiguity[:, column])


def task1_ambgfun(
    x: Any,
    fs: float,
    prf: float,
    y: Any | None = None,
    cut: str = "2d",
    cutValue: float = 0.0,
) -> Any:
    """组合原版二维 cuSignal 调用与 Task1 网格对齐切片。

    上游 ``cusignal-23.08.00`` 保持只读。ZQ500 不支持原版一维切片中隐式产生的
    ComplexFP64 FFT，也不接受 doppler 分支的非 2 次幂 FFT。这里始终调用原版
    ``cusignal.ambgfun(..., cut="2d")``；该路径对 Task1 ComplexFP32 输入使用
    2 次幂 ``nfreq``，随后在 GPU 数组上取得官方二维网格中对应的行或列。

    Task1 正式场景只请求零 delay/doppler cut。接口也接受与二维 ambiguity 网格
    精确对齐的其他 cutValue；不对齐值明确报错，不伪造插值语义。
    """

    normalized_cut = str(cut).lower()
    if normalized_cut not in {"2d", "delay", "doppler"}:
        raise ValueError("2d, delay, and doppler are the only cut values allowed")

    ambiguity_2d = task1_ambgfun_2d(x, fs, prf, y=y)
    if normalized_cut == "2d":
        return ambiguity_2d
    return task1_ambiguity_cut(
        ambiguity_2d, int(cp.asarray(x).size), fs, normalized_cut, cutValue
    )
