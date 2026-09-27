"""Shared contracts, constants and timing helpers for Task1 cuSignal Python."""

from __future__ import annotations

import os
import sys
import time
from importlib import metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from demo.task_config import Task1Config

from utils.source_guard import verify_immutable_cusignal_source


def _project_cusignal_python() -> Path:
    override = os.environ.get("TASK1_CUSIGNAL_PYTHON") or os.environ.get("CUSIGNAL_SOURCE")
    if override:
        return Path(override).expanduser().resolve()
    source = Path(__file__).resolve()
    candidates = (
        source.parents[2] / "cusignal-23.08.00" / "python",
        source.parents[3] / "cusignal-23.08.00" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


_CUSIGNAL_SOURCE = _project_cusignal_python()
_CUSIGNAL_INTEGRITY = verify_immutable_cusignal_source(_CUSIGNAL_SOURCE)
if _CUSIGNAL_SOURCE.is_dir() and str(_CUSIGNAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_CUSIGNAL_SOURCE))
try:
    import cupy as cp  # type: ignore
    import cusignal  # type: ignore
except Exception as error:  # Reported explicitly by the runtime probe/entry point.
    cp = None
    cusignal = None
    _IMPORT_ERROR: Exception | None = error
else:
    _IMPORT_ERROR = None


SPEED_OF_LIGHT = 299792458.0
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


@dataclass
class PipelineState:
    """Host-resident handoff state matching the C++/GPU pipeline transfers."""

    config: Task1Config
    waveform: np.ndarray | None = None
    noiseless_echo: np.ndarray | None = None
    noise: np.ndarray | None = None
    echo: np.ndarray | None = None
    compressed: np.ndarray | None = None
    range_doppler: np.ndarray | None = None
    power: np.ndarray | None = None
    cfar_alpha: float = 0.0
    cfar_threshold: np.ndarray | None = None
    detections: np.ndarray | None = None
    ambiguity_2d: np.ndarray | None = None
    ambiguity_delay: np.ndarray | None = None
    ambiguity_doppler: np.ndarray | None = None


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


T = TypeVar("T")


def time_gpu(function: Callable[[], T]) -> tuple[T, float]:
    """Time queued GPU work with CuPy events and synchronize the stop event."""
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    result = function()
    stop.record()
    stop.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start, stop))


def wall_ms(begin: float) -> float:
    return (time.perf_counter() - begin) * 1000.0


def synchronize() -> None:
    cp.cuda.get_current_stream().synchronize()


def as_host(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return cp.asnumpy(array)


def runtime_metadata() -> dict[str, Any]:
    require_cusignal_runtime()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw_name = properties.get("name", "unknown")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8", errors="replace")
    dependency_versions = {}
    for distribution in ("numpy", "scipy", "numba", "cupy", "cusignal"):
        try:
            dependency_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependency_versions[distribution] = "source-or-unavailable"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cusignal": getattr(cusignal, "__version__", "23.08.00-source"),
        "cusignal_source": str(_CUSIGNAL_SOURCE),
        "cusignal_immutable_source": _CUSIGNAL_INTEGRITY,
        "cupy_source": str(Path(cp.__file__).resolve()),
        "dependency_versions": dependency_versions,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "device_name": str(raw_name),
        "device_id": int(cp.cuda.Device().id),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "evidence_git_commit": os.environ.get(
            "TASK1_PYTHON_EVIDENCE_GIT_COMMIT", "unbound"
        ),
        "evidence_git_dirty": os.environ.get(
            "TASK1_PYTHON_EVIDENCE_GIT_DIRTY", "unknown"
        ),
        "evidence_test_time": os.environ.get(
            "TASK1_PYTHON_EVIDENCE_TEST_TIME", "unbound"
        ),
    }
