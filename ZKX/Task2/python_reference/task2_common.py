from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from demo.task_config import Task2Config


def _project_cusignal_python() -> Path:
    override = os.environ.get("TASK2_CUSIGNAL_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    source = Path(__file__).resolve()
    candidates = (
        source.parents[2] / "cusignal-23.08.00" / "python",
        source.parents[3] / "cusignal-23.08.00" / "python",
    )
    for candidate in candidates:
        if (candidate / "cusignal").is_dir():
            return candidate
    return candidates[-1]


_CUSIGNAL_SOURCE = _project_cusignal_python()
if _CUSIGNAL_SOURCE.is_dir() and str(_CUSIGNAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_CUSIGNAL_SOURCE))

try:
    import cupy as cp  # type: ignore
    import cusignal  # type: ignore
except Exception as error:
    cp = None
    cusignal = None
    _IMPORT_ERROR: Exception | None = error
else:
    _IMPORT_ERROR = None
    try:
        from utils import install as install_zq500_cusignal_adapter

        install_zq500_cusignal_adapter(cusignal, cp)
    except Exception as error:
        _IMPORT_ERROR = error


PI = np.pi
TASK_STEP_FORMAL_TIMING = os.environ.get("TASK_STEP_FORMAL_TIMING") == "1"


@dataclass
class StepEvidence:
    name: str
    prep_ms: float = 0.0
    h2d_ms: float = 0.0
    compute_ms: float = 0.0
    d2h_ms: float = 0.0
    post_ms: float = 0.0
    total_ms: float = 0.0
    formal_execution_ms: float = 0.0
    operator_ms: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    output_shape: list[int] = field(default_factory=list)
    output_dtype: str = ""
    status: str = "pass"


@dataclass
class PipelineState:
    config: Task2Config
    time: np.ndarray | None = None
    target_waveform: np.ndarray | None = None
    target_component: np.ndarray | None = None
    interference_component: np.ndarray | None = None
    noise_component: np.ndarray | None = None
    echo: np.ndarray | None = None
    filtered: np.ndarray | None = None
    filter_taps: np.ndarray | None = None
    smoothed: np.ndarray | None = None
    reference_for_correlation: np.ndarray | None = None
    fm_feature: np.ndarray | None = None
    correlation_feature: np.ndarray | None = None
    spectral_feature: np.ndarray | None = None
    wavelet_feature: np.ndarray | None = None
    feature_bundle: np.ndarray | None = None
    extrema: np.ndarray | None = None
    kalman_observations: np.ndarray | None = None
    kalman_covariance: np.ndarray | None = None
    kalman_anchor: int = 0
    estimate: float = 0.0
    truth: float = 0.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_cusignal_runtime() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "cuSignal Python runtime is unavailable; expected a CuPy-compatible GPU "
            f"runtime plus repository source at {_CUSIGNAL_SOURCE}: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR
    require(cp.cuda.runtime.getDeviceCount() > 0, "no CuPy-visible GPU device")
    module_file = Path(cusignal.__file__).resolve()
    expected_root = _CUSIGNAL_SOURCE.resolve()
    require(
        os.path.commonpath((str(module_file), str(expected_root))) == str(expected_root),
        f"imported cuSignal is not the repository source: {module_file}",
    )


T = TypeVar("T")


def time_gpu(function: Callable[[], T]) -> tuple[T, float]:
    # Task Step formal_execution 只在整个 Step 边界同步一次。这里若继续为每个
    # 子算子 stop.synchronize()，这些诊断同步会被重复计入 Python 分子，而 C++
    # 仅在 Step 末尾同步，导致端到端口径不对称。
    if TASK_STEP_FORMAL_TIMING:
        return function(), 0.0
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    result = function()
    stop.record()
    stop.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start, stop))


def synchronize() -> None:
    if TASK_STEP_FORMAL_TIMING:
        return
    cp.cuda.get_current_stream().synchronize()


def wall_ms(begin: float) -> float:
    return (time.perf_counter() - begin) * 1000.0


def as_host(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return cp.asnumpy(array)


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


def roughness(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0


def runtime_metadata() -> dict[str, Any]:
    require_cusignal_runtime()
    from utils import adapter_status

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw_name = properties.get("name", "unknown")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8", errors="replace")
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cusignal": getattr(cusignal, "__version__", "23.08.00-source"),
        "cusignal_source": str(_CUSIGNAL_SOURCE),
        "cusignal_module_file": str(Path(cusignal.__file__).resolve()),
        "device_name": str(raw_name),
        "device_id": int(cp.cuda.Device().id),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "zq500_cusignal_adapter": adapter_status(),
        "evidence_git_commit": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_COMMIT", "unbound"),
        "evidence_git_dirty": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_DIRTY", "unknown"),
        "evidence_test_time": os.environ.get("TASK2_PYTHON_EVIDENCE_TEST_TIME", "unbound"),
    }
