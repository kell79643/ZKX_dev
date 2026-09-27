# Task2 Python 实现逻辑

## 1. 职责与版本

Python入口、common、runner、全部step和任务私有helper/adapter。

- 当前ZKX SHA：`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交时间：`2026-08-18T20:22:35+08:00`
- 每段源码另绑定实际SHA。
- tracked源码状态：clean
- 本轮只静态阅读。

## 2. 源码覆盖与唯一归属

|源码|SHA|
|---|---|
|`ZKX/Task2/task2_python/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/task2_common.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/task2_runner.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step1/step1.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step2/step2.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step3/step3.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step4/step4.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step5/step5.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step6/step6.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step1/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step1/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step2/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step2/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step3/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step3/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step4/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step4/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step5/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step5/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step6/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/step6/main.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_python/utils/__init__.py`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_python/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import argparse

from task2_runner import run_task2_until
from demo.task_config import Task2Config


def main() -> int:
    parser = argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")
    parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--evidence-id", default="pipeline")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = Task2Config.load(args.config)
    run_task2_until(args.stop_after, args.evidence_id, config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 3.2 `ZKX/Task2/task2_python/task2_common.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
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
```

### 3.3 `ZKX/Task2/task2_python/task2_runner.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from step1.step1 import run_step1
from step2.step2 import run_step2
from step3.step3 import run_step3
from step4.step4 import run_step4
from step5.step5 import run_step5
from step6.step6 import run_step6
from task2_common import (
    PipelineState,
    StepEvidence,
    require,
    require_cusignal_runtime,
    runtime_metadata,
    wall_ms,
)
from demo.task_config import Task2Config


RUN_STEPS: list[Callable[[PipelineState], StepEvidence]] = [
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    run_step5,
    run_step6,
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # bool is a subclass of int; preserve JSON booleans before integer handling.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _step_json(evidence: StepEvidence) -> dict[str, Any]:
    raw = asdict(evidence)
    return {
        "name": evidence.name,
        "status": evidence.status,
        "timing_ms": {
            "prepare": raw["prep_ms"],
            "h2d": raw["h2d_ms"],
            "compute": raw["compute_ms"],
            "d2h": raw["d2h_ms"],
            "postprocess": raw["post_ms"],
            "formal_execution": raw["formal_execution_ms"],
            "total": raw["total_ms"],
        },
        "operators_ms": raw["operator_ms"],
        "metrics": raw["metrics"],
        "output": {"shape": raw["output_shape"], "dtype": raw["output_dtype"]},
    }


def run_task2_until(
    stop_after: int,
    evidence_id: str,
    config: Task2Config,
    output_directory: str | Path = ".",
    write_json: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    require_cusignal_runtime()
    require(1 <= stop_after <= 6, "Task2 stop_after must be in [1,6]")
    require(bool(evidence_id), "Task2 evidence_id must be nonempty")
    pipeline_begin = time.perf_counter()
    state = PipelineState(config=config)
    steps = [RUN_STEPS[index](state) for index in range(stop_after)]
    pipeline_total_ms = wall_ms(pipeline_begin)
    runtime = runtime_metadata()
    result = _json_safe(
        {
            "task": "Task2",
            "implementation": "cusignal_python_23.08.00",
            "performance_identity": "cuSignal Python GPU + required ZQ500 adapter",
            "evidence_id": evidence_id,
            "requested_steps": stop_after,
            "completed_steps": len(steps),
            "status": "pass",
            "git_commit": runtime["evidence_git_commit"],
            "git_dirty": runtime["evidence_git_dirty"],
            "test_time": runtime["evidence_test_time"],
            "runtime": runtime,
            "config": {
                "path": str(config.path),
                "sha256": config.sha256,
                "samples": config.samples,
                "sample_rate_hz": config.sample_rate_hz,
                "target_delay_samples": config.target_delay_samples,
                "noise_seed": config.noise_seed,
                "target_weight": config.target_weight,
                "gaussian_weight": config.gaussian_weight,
                "sawtooth_weight": config.sawtooth_weight,
                "square_weight": config.square_weight,
                "noise_amplitude": config.noise_amplitude,
                "filter_taps": config.filter_taps,
                "filter_cutoff_hz": config.filter_cutoff_hz,
                "step3_crop_each": config.step3_crop_each,
                "fusion_weights": {
                    "fm": config.fusion_weight_fm,
                    "correlation": config.fusion_weight_correlation,
                    "spectral": config.fusion_weight_spectral,
                    "wavelet": config.fusion_weight_wavelet,
                },
                "argrelextrema_order": config.argrelextrema_order,
                "kalman_observation_count": config.kalman_observation_count,
                "kalman": {"F": config.kalman_F, "Q": config.kalman_Q,
                           "H": config.kalman_H, "R": config.kalman_R},
            },
            "input_contract": {
                "dtype": "FP32",
                "shape": [config.samples],
                "sample_rate_hz": config.sample_rate_hz,
                "target_delay_samples": config.target_delay_samples,
                "noise_seed": config.noise_seed,
                "step3_output_samples": config.samples - 2 * config.step3_crop_each,
                "feature_fusion_weights": {
                    "fm": config.fusion_weight_fm,
                    "correlation": config.fusion_weight_correlation,
                    "spectral": config.fusion_weight_spectral,
                    "wavelet": config.fusion_weight_wavelet,
                },
                "argrelextrema": {"axis": 0, "order": config.argrelextrema_order, "mode": "clip"},
            },
            "timing_policy": {
                "gpu_clock": "cupy.cuda.Event",
                "step_clock": "time.perf_counter",
                "h2d_scope": "cp.asarray followed by current-stream synchronization",
                "pipeline_handoff": "Host PipelineState with explicit H2D/D2H, matching task2_gpu_cpu",
                "d2h_scope": "formal step output copied for downstream step and evidence",
                "postprocess_scope": "mandatory Host handoff where applicable plus semantic/task-effect checks; no CPU precision reference",
                "formal_execution_scope": "prepare + H2D + GPU operators + required D2H and mandatory handoff; excludes semantic checks",
                "adapter_scope": "all required Task2/utils ZQ500 adapter execution is inside the cuSignal Python GPU timing numerator",
            },
            "precision_policy": {
                "cpu_validation": False,
                "cpu_comparison": False,
                "reason": "Task2 Python is the cuSignal performance baseline; CPU precision evidence belongs to task2_gpu_cpu only",
            },
            "formal_apis": [
                "cusignal.chirp/gausspulse/sawtooth/square",
                "cusignal.hamming/firwin/firfilter",
                "cusignal.cubic/firfilter",
                "cusignal.fm_demod/correlate/spectrogram/cwt/ricker",
                "cusignal.argrelextrema",
                "cusignal.KalmanFilter.predict/update",
            ],
            "pipeline_total_ms": pipeline_total_ms,
            "sum_step_total_ms": sum(step.total_ms for step in steps),
            "steps": [_step_json(step) for step in steps],
        }
    )
    if write_json:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        destination = output_path / f"task2_{evidence_id}_evidence.json"
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not quiet:
        print(f"[TASK2_PYTHON][CONFIG] path={config.path} sha256={config.sha256}")
        for step in steps:
            print(
                f"[TASK2_PYTHON][{step.name}][TIMING] prep_ms={step.prep_ms:.6f} "
                f"h2d_ms={step.h2d_ms:.6f} compute_ms={step.compute_ms:.6f} "
                f"d2h_ms={step.d2h_ms:.6f} post_ms={step.post_ms:.6f} "
                f"formal_execution_ms={step.formal_execution_ms:.6f} total_ms={step.total_ms:.6f}"
            )
            print(
                f"[TASK2_PYTHON][{step.name}][EFFECT] metrics={len(step.metrics)} "
                f"operators={len(step.operator_ms)} status={step.status}"
            )
        print(
            f"[TASK2_PYTHON][PIPELINE] completed_steps={len(steps)} "
            f"pipeline_total_ms={pipeline_total_ms:.6f} status=pass"
        )
    return result
```

### 3.4 `ZKX/Task2/task2_python/step1/step1.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)


def _noise_bits(indices, seed: int):
    values = cp.uint32(seed) ^ (
        indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)
    )
    values ^= values >> cp.uint32(16)
    values *= cp.uint32(2246822519)
    values ^= values >> cp.uint32(13)
    values *= cp.uint32(3266489917)
    return values ^ (values >> cp.uint32(16))


def _mix_echo(
    chirp, gaussian, sawtooth, square, config,
    noise_seed: int | None = None, target_weight: float | None = None,
):
    seed = config.noise_seed if noise_seed is None else noise_seed
    weight = config.target_weight if target_weight is None else target_weight
    indices = cp.arange(config.samples, dtype=cp.int32)
    source = indices - config.target_delay_samples
    target = cp.where(
        source >= 0,
        cp.float32(weight) * chirp[cp.maximum(source, 0)],
        0.0,
    )
    interference = (
        cp.float32(config.gaussian_weight) * gaussian
        + cp.float32(config.sawtooth_weight) * sawtooth.astype(cp.float32)
        + cp.float32(config.square_weight) * square.astype(cp.float32)
    )
    bits = _noise_bits(indices.astype(cp.uint32), seed)
    noise = (
        ((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)
        * cp.float32(config.noise_amplitude)
    )
    return (
        target.astype(cp.float32),
        interference.astype(cp.float32),
        noise.astype(cp.float32),
        (target + interference + noise).astype(cp.float32),
    )


def run_step1(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step1")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    state.time = np.arange(config.samples, dtype=np.float32) / np.float32(config.sample_rate_hz)
    pulse_time = state.time - np.float32(0.45)
    sample_indices = np.arange(config.samples, dtype=np.float64)
    phase = (
        2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)
    ).astype(np.float32)
    square_phase = (
        2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)
    ).astype(np.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_time = cp.asarray(state.time)
    d_pulse_time = cp.asarray(pulse_time)
    d_phase = cp.asarray(phase)
    d_square_phase = cp.asarray(square_phase)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_chirp, chirp_ms = time_gpu(
        lambda: cusignal.chirp(
            d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0
        )
    )
    d_gaussian, gaussian_ms = time_gpu(
        lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)
    )
    d_saw, saw_ms = time_gpu(
        lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65))
    )
    d_square, square_ms = time_gpu(
        lambda: cusignal.square(d_square_phase, duty=np.float32(0.35))
    )
    outputs, mix_ms = time_gpu(lambda: _mix_echo(d_chirp, d_gaussian, d_saw, d_square, config))
    d_target, d_interference, d_noise, d_echo = outputs
    evidence.operator_ms = {
        "cusignal.chirp": chirp_ms,
        "cusignal.gausspulse": gaussian_ms,
        "cusignal.sawtooth": saw_ms,
        "cusignal.square": square_ms,
        "cupy.echo_superposition": mix_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.target_waveform = as_host(d_chirp).astype(np.float32, copy=False)
    state.target_component = as_host(d_target)
    state.interference_component = as_host(d_interference)
    state.noise_component = as_host(d_noise)
    state.echo = as_host(d_echo)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(np.all(np.isfinite(state.echo)), "step1 produced non-finite echo")
    require(np.max(np.abs(state.target_component)) > 0.1, "step1 target component degenerate")
    require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")
    require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")
    perturbed_weight = _mix_echo(
        d_chirp, d_gaussian, d_saw, d_square, config,
        target_weight=max(0.0, config.target_weight - 0.07),
    )[3]
    perturbed_seed = _mix_echo(
        d_chirp, d_gaussian, d_saw, d_square, config, noise_seed=config.noise_seed + 1
    )[3]
    weight_delta = float(np.max(np.abs(as_host(perturbed_weight) - state.echo)))
    seed_delta = float(np.max(np.abs(as_host(perturbed_seed) - state.echo)))
    require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")
    require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")
    evidence.metrics = {
        "waveform_functions_selected": 4.0,
        "target_component": 1.0,
        "interference_component": 1.0,
        "noise_component": 1.0,
        "custom_waveform_superposition": 1.0,
        "sample_rate_hz": config.sample_rate_hz,
        "samples": float(config.samples),
        "target_delay_samples": float(config.target_delay_samples),
        "noise_seed": float(config.noise_seed),
        "echo_rms": float(np.sqrt(np.mean(state.echo.astype(np.float64) ** 2))),
        "weight_perturbation_delta": weight_delta,
        "seed_perturbation_delta": seed_delta,
        "perturbation_checks_pass": 1.0,
    }
    evidence.output_shape = list(state.echo.shape)
    evidence.output_dtype = str(state.echo.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

### 3.5 `ZKX/Task2/task2_python/step2/step2.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)


def run_step2(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step2")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.echo is not None and state.echo.size == config.samples, "step2 did not receive step1 echo")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_echo = cp.asarray(state.echo, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_window, window_ms = time_gpu(lambda: cusignal.hamming(config.filter_taps, sym=True))
    d_taps, firwin_ms = time_gpu(
        lambda: cusignal.firwin(
            config.filter_taps, config.filter_cutoff_hz, window="hamming", pass_zero=True,
            scale=True, fs=config.sample_rate_hz, gpupath=True,
        ).astype(cp.float32)
    )
    d_filtered, filter_ms = time_gpu(lambda: cusignal.firfilter(d_taps, d_echo, axis=-1))
    evidence.operator_ms = {
        "cusignal.hamming": window_ms,
        "cusignal.firwin": firwin_ms,
        "cusignal.firfilter": filter_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.filter_taps = as_host(d_taps).astype(np.float32)
    state.filtered = as_host(d_filtered).astype(np.float32)
    window_host = as_host(d_window)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(state.filtered.size == config.samples, "step2 output length mismatch")
    require(np.all(np.isfinite(state.filtered)), "step2 produced non-finite output")
    target_filtered = np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]
    residual = state.echo - state.target_component
    residual_filtered = np.convolve(residual, state.filter_taps, mode="full")[:config.samples]
    before_snr = 20.0 * np.log10(
        max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)
        / max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)
    )
    after_snr = 20.0 * np.log10(
        max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12)
        / max(np.sqrt(np.mean(residual_filtered.astype(np.float64) ** 2)), 1.0e-12)
    )
    require(after_snr - before_snr >= 1.0, "step2 SNR improvement is below 1 dB")
    evidence.metrics = {
        "filter_design_functions": 1.0,
        "filtering_functions": 1.0,
        "window_functions": 1.0,
        "step1_to_step2_data_match": 1.0,
        "taps": float(config.filter_taps),
        "cutoff_hz": config.filter_cutoff_hz,
        "window_nonzero": float(np.max(np.abs(window_host)) > 0.0),
        "snr_before_db": float(before_snr),
        "snr_after_db": float(after_snr),
        "snr_improvement_db": float(after_snr - before_snr),
    }
    evidence.output_shape = list(state.filtered.shape)
    evidence.output_dtype = str(state.filtered.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

### 3.6 `ZKX/Task2/task2_python/step3/step3.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    roughness,
    synchronize,
    time_gpu,
    wall_ms,
)


def run_step3(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step3")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.filtered is not None and state.filtered.size == config.samples, "step3 did not receive step2 output")
    crop = config.step3_crop_each
    cropped = state.filtered[crop:-crop].astype(np.float32, copy=True)
    offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_cropped = cp.asarray(cropped)
    d_offsets = cp.asarray(offsets)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_cubic, cubic_ms = time_gpu(lambda: cusignal.cubic(d_offsets))
    d_weights, normalization_ms = time_gpu(lambda: d_cubic / cp.sum(d_cubic))
    d_smoothed, filter_ms = time_gpu(lambda: cusignal.firfilter(d_weights, d_cropped, axis=-1))
    evidence.operator_ms = {
        "cusignal.cubic": cubic_ms,
        "cupy.cubic_normalization": normalization_ms,
        "cusignal.firfilter_b_spline": filter_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.smoothed = as_host(d_smoothed).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    handoff_begin = time.perf_counter()
    state.reference_for_correlation = state.target_waveform[
        crop:-crop
    ].astype(np.float32, copy=True)
    evidence.post_ms = wall_ms(handoff_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    rough_before = roughness(cropped)
    rough_after = roughness(state.smoothed)
    peak_ratio = float(
        np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12)
    )
    require(rough_after < rough_before, "step3 did not reduce roughness")
    require(peak_ratio > 0.20, "step3 over-smoothed the target structure")
    evidence.metrics = {
        "bsplines_functions": 1.0,
        "step2_to_step3_data_match": 1.0,
        "crop_begin": float(crop),
        "crop_end": float(config.samples - crop),
        "roughness_before": rough_before,
        "roughness_after": rough_after,
        "peak_preservation_ratio": peak_ratio,
    }
    evidence.output_shape = list(state.smoothed.shape)
    evidence.output_dtype = str(state.smoothed.dtype)
    evidence.post_ms += wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

### 3.7 `ZKX/Task2/task2_python/step4/step4.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)


def _analytic_signal(signal):
    previous = cp.concatenate((signal[:1], signal[:-1]))
    following = cp.concatenate((signal[1:], signal[-1:]))
    return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)


def _resample_linear(values, count: int):
    if values.size == count:
        return values.astype(cp.float32)
    old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)
    new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)
    return cp.interp(new, old, values.astype(cp.float32))


def _normalize_abs(values):
    magnitudes = cp.abs(values).astype(cp.float32)
    maximum = cp.max(magnitudes)
    return cp.where(maximum > 0.0, magnitudes / maximum, cp.zeros_like(magnitudes))


def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):
    fm_n = _normalize_abs(_resample_linear(fm, count))
    corr_n = _normalize_abs(_resample_linear(correlation, count))
    spec_n = _normalize_abs(_resample_linear(spectral, count))
    wave_n = _normalize_abs(_resample_linear(wavelet, count))
    return (
        cp.float32(config.fusion_weight_fm) * fm_n
        + cp.float32(config.fusion_weight_correlation) * corr_n
        + cp.float32(config.fusion_weight_spectral) * spec_n
        + cp.float32(config.fusion_weight_wavelet) * wave_n
    ).astype(cp.float32)


def run_step4(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step4")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.smoothed is not None and state.smoothed.size > 64, "step4 did not receive step3 output")
    require(
        state.reference_for_correlation is not None
        and state.reference_for_correlation.size == state.smoothed.size,
        "step4 correlation reference mismatch",
    )
    count = int(state.smoothed.size)
    widths = [2, 4, 8, 12]
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_signal = cp.asarray(state.smoothed, dtype=cp.float32)
    d_reference = cp.asarray(state.reference_for_correlation, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_analytic, analytic_ms = time_gpu(lambda: _analytic_signal(d_signal))
    d_fm, fm_ms = time_gpu(lambda: cusignal.fm_demod(d_analytic, axis=-1))
    d_correlation, correlation_ms = time_gpu(
        lambda: cusignal.correlate(d_signal, d_reference, mode="same", method="auto")
    )
    def run_spectrogram():
        return cusignal.spectrogram(
            d_signal,
            fs=config.sample_rate_hz,
            window="boxcar",
            nperseg=64,
            noverlap=32,
            nfft=64,
            detrend=False,
            return_onesided=False,
            scaling="spectrum",
            mode="magnitude",
        )
    spectrogram_outputs, spectral_ms = time_gpu(run_spectrogram)
    _, _, d_spectrogram = spectrogram_outputs
    d_spectral, spectral_envelope_ms = time_gpu(
        lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32)
    )
    d_cwt, cwt_ms = time_gpu(lambda: cusignal.cwt(d_signal, cusignal.ricker, widths))
    d_wavelet, cwt_envelope_ms = time_gpu(
        lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32)
    )
    d_bundle, bundle_ms = time_gpu(
        lambda: _make_bundle(d_fm, d_correlation, d_spectral, d_wavelet, count, config)
    )
    evidence.operator_ms = {
        "cupy.analytic_glue": analytic_ms,
        "cusignal.fm_demod": fm_ms,
        "cusignal.correlate": correlation_ms,
        "cusignal.spectrogram": spectral_ms,
        "cupy.spectral_envelope": spectral_envelope_ms,
        "cusignal.cwt_ricker": cwt_ms,
        "cupy.cwt_envelope": cwt_envelope_ms,
        "cupy.feature_fusion": bundle_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.fm_feature = as_host(d_fm).astype(np.float32)
    state.correlation_feature = as_host(d_correlation).astype(np.float32)
    state.spectral_feature = as_host(d_spectral).astype(np.float32)
    state.wavelet_feature = as_host(d_wavelet).astype(np.float32)
    state.feature_bundle = as_host(d_bundle).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    features = {
        "fm": state.fm_feature,
        "correlation": state.correlation_feature,
        "spectral": state.spectral_feature,
        "wavelet": state.wavelet_feature,
    }
    for name, item in features.items():
        require(np.all(np.isfinite(item)), f"step4 {name} feature is non-finite")
        require(np.max(np.abs(item)) > 0.0, f"step4 {name} feature is degenerate")
    require(np.max(state.feature_bundle) > 0.1, "step4 fused bundle degenerate")
    evidence.metrics = {
        "feature_families": 4.0,
        "demod_functions": 1.0,
        "correlate_functions": 1.0,
        "spectral_functions": 1.0,
        "wavelets_functions": 2.0,
        "step3_to_step4_data_match": 1.0,
        "feature_bundle_complete": 1.0,
        "fusion_weight_fm": config.fusion_weight_fm,
        "fusion_weight_correlation": config.fusion_weight_correlation,
        "fusion_weight_spectral": config.fusion_weight_spectral,
        "fusion_weight_wavelet": config.fusion_weight_wavelet,
        "bundle_samples": float(count),
        "bundle_peak": float(np.max(state.feature_bundle)),
    }
    evidence.output_shape = list(state.feature_bundle.shape)
    evidence.output_dtype = str(state.feature_bundle.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

### 3.8 `ZKX/Task2/task2_python/step5/step5.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)


def run_step5(state: PipelineState) -> StepEvidence:
    order = state.config.argrelextrema_order
    evidence = StepEvidence("step5")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.feature_bundle is not None and state.feature_bundle.size > 0, "step5 did not receive step4 bundle")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_bundle = cp.asarray(state.feature_bundle, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    result, extrema_ms = time_gpu(
        lambda: cusignal.argrelextrema(
            d_bundle, cp.greater, axis=0, order=order, mode="clip"
        )
    )
    require(len(result) == 1, "step5 coordinate rank mismatch")
    evidence.operator_ms = {"cusignal.argrelextrema": extrema_ms}
    evidence.compute_ms = extrema_ms
    d2h_begin = time.perf_counter()
    state.extrema = as_host(result[0]).astype(np.int64)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(state.extrema.size > 0, "step5 found no feature points")
    require(np.all(np.diff(state.extrema) > 0), "step5 extrema are not ordered unique indices")
    require(
        int(state.extrema[0]) >= 0 and int(state.extrema[-1]) < state.feature_bundle.size,
        "step5 extrema index out of bounds",
    )
    injection_index = 50
    moved_index = 70
    injected = state.feature_bundle.copy()
    injected[injection_index - order : injection_index + order + 1] = 0.0
    injected[injection_index] = float(np.max(state.feature_bundle) + 1.0)
    d_injected = cp.asarray(injected, dtype=cp.float32)
    injected_result = cusignal.argrelextrema(
        d_injected, cp.greater, axis=0, order=order, mode="clip"
    )
    injected_indices = as_host(injected_result[0]).astype(np.int64)
    moved = injected.copy()
    moved[injection_index] = 0.0
    moved[moved_index - order : moved_index + order + 1] = 0.0
    moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)
    d_moved = cp.asarray(moved, dtype=cp.float32)
    moved_result = cusignal.argrelextrema(
        d_moved, cp.greater, axis=0, order=order, mode="clip"
    )
    moved_indices = as_host(moved_result[0]).astype(np.int64)
    require(injection_index in injected_indices, "step5 injected peak was not detected")
    require(moved_index in moved_indices, "step5 moved peak was not detected")
    require(injection_index not in moved_indices, "step5 old injected peak remained after movement")
    evidence.metrics = {
        "argrelextrema_called": 1.0,
        "step4_to_step5_data_match": 1.0,
        "comparator_greater": 1.0,
        "axis": 0.0,
        "order": float(order),
        "mode_clip": 1.0,
        "extrema_count": float(state.extrema.size),
        "injected_peak_index": float(injection_index),
        "moved_peak_index": float(moved_index),
        "injected_peak_followed": 1.0,
        "moved_peak_followed": 1.0,
        "perturbation_checks_pass": 1.0,
    }
    evidence.output_shape = list(state.extrema.shape)
    evidence.output_dtype = str(state.extrema.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

### 3.9 `ZKX/Task2/task2_python/step6/step6.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)


def _observations_from_feature_point(bundle: np.ndarray, anchor: int, count: int) -> np.ndarray:
    observations: list[float] = []
    for radius in range(1, count + 1):
        left = max(0, anchor - radius)
        right = min(bundle.size - 1, anchor + radius)
        indices = np.arange(left, right + 1, dtype=np.float64)
        weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)
        coordinate = float(np.sum(indices * weights) / np.sum(weights)) if np.sum(weights) > 0 else float(anchor)
        observations.append(coordinate / (bundle.size - 1))
    return np.asarray(observations, dtype=np.float32)


def _estimate_with_noise(observations: np.ndarray, measurement_noise: float, config) -> float:
    kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)
    kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)
    kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)
    kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)
    kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)
    kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)
    kalman.R[...] = cp.asarray([[[measurement_noise]]], dtype=cp.float32)
    for observation in observations:
        kalman.predict()
        kalman.update(cp.asarray([[[observation]]], dtype=cp.float32))
    return float(as_host(kalman.x).reshape(-1)[0])


def run_step6(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step6")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.extrema is not None and state.extrema.size > 0, "step6 did not receive step5 points")
    strongest_position = int(np.argmax(state.feature_bundle[state.extrema]))
    state.kalman_anchor = int(state.extrema[strongest_position])
    state.kalman_observations = _observations_from_feature_point(
        state.feature_bundle, state.kalman_anchor, config.kalman_observation_count
    )
    state.truth = float(
        (0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples)
        / (state.feature_bundle.size - 1)
    )
    kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)
    kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)
    kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)
    kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)
    kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)
    kalman.R[...] = cp.asarray([[[config.kalman_R]]], dtype=cp.float32)
    d_observations = cp.asarray(state.kalman_observations, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    if hasattr(kalman, "predict_update_sequence"):
        _, kalman_ms = time_gpu(
            lambda: kalman.predict_update_sequence(d_observations))
    else:
        # 非 ZQ500 scalar adapter 的兼容路径仍保持公开 predict/update 语义。
        kalman_ms = 0.0
        for observation in state.kalman_observations:
            d_observation = cp.asarray([[[observation]]], dtype=cp.float32)

            def predict_update():
                kalman.predict()
                kalman.update(d_observation)

            _, iteration_ms = time_gpu(predict_update)
            kalman_ms += iteration_ms
    evidence.operator_ms = {"cusignal.KalmanFilter.predict_update": kalman_ms}
    evidence.compute_ms = kalman_ms
    d2h_begin = time.perf_counter()
    estimated_state = as_host(kalman.x)
    state.kalman_covariance = as_host(kalman.P).reshape(-1).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    state.estimate = float(estimated_state.reshape(-1)[0])
    estimate_error = abs(state.estimate - state.truth)
    require(np.isfinite(state.estimate), "step6 estimate is non-finite")
    require(np.all(np.isfinite(state.kalman_covariance)), "step6 covariance is non-finite")
    require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")
    high_noise_estimate = _estimate_with_noise(
        state.kalman_observations, config.kalman_measurement_noise_variant, config
    )
    measurement_noise_response = abs(high_noise_estimate - state.estimate)
    require(np.isfinite(high_noise_estimate), "step6 noise perturbation is non-finite")
    require(
        measurement_noise_response > 1.0e-6,
        "step6 measurement-noise perturbation did not change estimate",
    )
    evidence.metrics = {
        "kalmanfilter_called": 1.0,
        "step5_to_step6_data_match": 1.0,
        "kalman_anchor": float(state.kalman_anchor),
        "observation_count": float(state.kalman_observations.size),
        "estimate": state.estimate,
        "truth": state.truth,
        "estimate_abs_error": estimate_error,
        "estimate_error_limit": 0.18,
        "final_covariance": float(state.kalman_covariance[0]),
        "high_measurement_noise_estimate": high_noise_estimate,
        "measurement_noise_response": measurement_noise_response,
        "perturbation_checks_pass": 1.0,
    }
    evidence.output_shape = [1]
    evidence.output_dtype = "float32"
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

### 3.10 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""

from __future__ import annotations

import copy
import importlib
from typing import Any

import numpy as np


_INSTALLED = False
_DETAILS: dict[str, Any] = {}


ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {
    "waveform_source_kernels": {
        "public_apis": ["cusignal.chirp", "cusignal.sawtooth", "cusignal.square"],
        "upstream_backend": "cusignal.waveforms.waveforms ElementwiseKernel symbols",
        "zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",
        "adapter_action": "install source-JIT kernels with the same public parameters, equations and upstream output dtypes",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "cupy_direct_convolution_backend": {
        "public_apis": ["cusignal.firfilter", "cusignal.correlate", "cusignal.cwt"],
        "upstream_backend": "CuPy/cupyx one-dimensional direct convolve used by cuSignal",
        "zq500_reason": "the private ZQ500 CuPy convolution backend lacks the required working one-dimensional source path",
        "adapter_action": "provide the same full/same/valid direct-convolution primitive on ZQ500 device arrays",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "argrelextrema_backend": {
        "public_apis": ["cusignal.argrelextrema"],
        "upstream_backend": "cusignal.peak_finding._boolrelextrema compiled-kernel path",
        "zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable on ZQ500",
        "adapter_action": "use cuSignal's own generic take/compare algorithm for the Task2 one-dimensional input",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "scalar_kalman_backend": {
        "public_apis": ["cusignal.KalmanFilter", "KalmanFilter.predict", "KalmanFilter.update", "KalmanFilter.predict_update_sequence"],
        "upstream_backend": "cusignal.estimation._filters_cuda RawModule specialization",
        "zq500_reason": "ZQ500 CuPy does not support the upstream NVRTC lowered-name/name_expressions path",
        "adapter_action": "evaluate the same scalar predict/update equations with CuPy device operations for dim_x=dim_z=1",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
        "approved_task_shape": {"dim_x": 1, "dim_z": 1, "dim_u": 0},
    },
}

INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {
    "waveform.sawtooth_square_outputs": {
        "business_input_dtype": "FP32",
        "intermediate_output_dtype": "FP64",
        "source": "cuSignal 23.08 waveform public output contract",
        "reason_retained": "changing the public cuSignal output dtype would make the comparison semantically unequal",
        "device_only": True,
        "runtime_evidence": "stock/adapted probe plus Task2 Step1 and pipeline acceptance on ZQ500",
    },
    "step4.cwt_output": {
        "business_input_dtype": "FP32",
        "intermediate_output_dtype": "FP64",
        "source": "cuSignal 23.08 cwt/ricker implementation-selected output",
        "reason_retained": "the Task2 adapter preserves the upstream public API instead of silently narrowing it",
        "device_only": True,
        "runtime_evidence": "Task2 Step4 and pipeline acceptance on ZQ500",
    },
}


class _CupyProxy:
    def __init__(self, cupy_module: Any, convolve_function: Any) -> None:
        self._cupy = cupy_module
        self.convolve = convolve_function

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cupy, name)


def _install_source_convolution(cp: Any) -> Any:
    kernel = cp.ElementwiseKernel(
        "raw T first, raw T second, int32 first_size, int32 second_size, int32 offset",
        "T output",
        """
        const int full_index = i + offset;
        T accumulator {};
        for (int second_index = 0; second_index < second_size; ++second_index) {
            const int first_index = full_index - second_index;
            if (first_index >= 0 && first_index < first_size) {
                accumulator += first[first_index] * second[second_index];
            }
        }
        output = accumulator;
        """,
        "_task2_zq500_convolve_1d",
        options=("-std=c++11",),
    )

    def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:
        del method
        first = cp.asarray(first)
        second = cp.asarray(second)
        if first.ndim == second.ndim == 0:
            return first * second
        if first.ndim != 1 or second.ndim != 1:
            raise ValueError("ZQ500 source convolution adapter supports one-dimensional inputs")
        if mode not in ("full", "same", "valid"):
            raise ValueError("mode must be 'full', 'same', or 'valid'")
        dtype = cp.result_type(first, second)
        first = first.astype(dtype, copy=False)
        second = second.astype(dtype, copy=False)
        full_size = int(first.size + second.size - 1)
        if mode == "full":
            output_size = full_size
            offset = 0
        elif mode == "same":
            output_size = int(first.size)
            offset = (full_size - output_size) // 2
        else:
            output_size = abs(int(first.size) - int(second.size)) + 1
            offset = min(int(first.size), int(second.size)) - 1
        return kernel(
            first,
            second,
            np.int32(first.size),
            np.int32(second.size),
            np.int32(offset),
            size=output_size,
        )

    return direct_convolve


def _install_waveform_kernels(cp: Any) -> None:
    module = importlib.import_module("cusignal.waveforms.waveforms")
    module._sawtooth_kernel = cp.ElementwiseKernel(
        "T t, T w",
        "float64 y",
        """
        double out {};
        const bool invalid { ((w > 1) || (w < 0)) };
        if (invalid) out = nan("0xfff8000000000000ULL");
        const T tmod = fmod(t, 2.0 * M_PI);
        const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };
        if (rising) out = tmod / (M_PI * w) - 1;
        if ((1 - invalid) && (1 - rising))
            out = (M_PI * (w + 1) - tmod) / (M_PI * (1 - w));
        y = out;
        """,
        "_task2_zq500_sawtooth",
        options=("-std=c++11",),
    )
    module._square_kernel = cp.ElementwiseKernel(
        "T t, T w",
        "float64 y",
        """
        const bool invalid { ((w > 1) || (w < 0)) };
        if (invalid) y = nan("0xfff8000000000000ULL");
        const T tmod = fmod(t, 2.0 * M_PI);
        const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };
        if (high) y = 1;
        if ((1 - invalid) && (1 - high)) y = -1;
        """,
        "_task2_zq500_square",
        options=("-std=c++11",),
    )
    chirp_source = """
        const T beta { (f1 - f0) / t1 };
        const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);
        phase = cos(temp + phi);
    """
    module._chirp_phase_lin_kernel_real = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "T phase",
        chirp_source,
        "_task2_zq500_chirp_linear_real",
        options=("-std=c++11",),
    )
    module._chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "Y phase",
        """
        const T beta { (f1 - f0) / t1 };
        const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);
        phase = Y(cos(temp + phi), cos(temp + phi + M_PI / 2) * -1);
        """,
        "_task2_zq500_chirp_linear_complex",
        options=("-std=c++11",),
    )


def _install_peak_helper(cp: Any) -> None:
    module = importlib.import_module("cusignal.peak_finding.peak_finding")

    def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:
        if int(order) != order or order < 1:
            raise ValueError("Order must be an int >= 1")
        length = data.shape[axis]
        locations = cp.arange(length)
        result = cp.ones(data.shape, dtype=bool)
        main = cp.take(data, locations, axis=axis)
        for shift in range(1, order + 1):
            if mode == "clip":
                plus_locations = cp.clip(locations + shift, None, length - 1)
                minus_locations = cp.clip(locations - shift, 0, None)
            elif mode == "wrap":
                plus_locations = (locations + shift) % length
                minus_locations = (locations - shift) % length
            else:
                raise NotImplementedError("CuPy `take` doesn't support mode='raise'.")
            result &= comparator(main, cp.take(data, plus_locations, axis=axis))
            result &= comparator(main, cp.take(data, minus_locations, axis=axis))
        return result

    module._boolrelextrema = boolrelextrema


def _install_scalar_kalman(cp: Any, cusignal: Any) -> None:
    klass = cusignal.KalmanFilter
    original_init = klass.__init__
    original_predict = klass.predict
    original_update = klass.update
    sequence_kernel = cp.ElementwiseKernel(
        "raw float32 observations, int32 observation_count, raw float32 initial_x, "
        "raw float32 initial_p, raw float32 f, raw float32 q, raw float32 alpha_sq, "
        "raw float32 h, raw float32 r",
        "float32 next_x, float32 next_p",
        """
        float state = initial_x[0];
        float covariance = initial_p[0];
        const float transition = f[0];
        const float process_noise = q[0];
        const float alpha = alpha_sq[0];
        const float measurement = h[0];
        const float measurement_noise = r[0];
        for (int index = 0; index < observation_count; ++index) {
            state = transition * state;
            covariance = alpha * transition * covariance * transition + process_noise;
            const float innovation = observations[index] - measurement * state;
            const float pht = covariance * measurement;
            const float gain = pht / (measurement * pht + measurement_noise);
            state += gain * innovation;
            const float i_kh = 1.0F - gain * measurement;
            covariance = i_kh * covariance * i_kh + gain * measurement_noise * gain;
        }
        next_x = state;
        next_p = covariance;
        """,
        "_task2_zq500_kalman_scalar_sequence",
        options=("-std=c++11",),
    )

    def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:
        dtype = cp.float32 if dtype is None else dtype
        if dim_x != 1 or dim_z != 1 or dim_u != 0:
            original_init(self, dim_x, dim_z, dim_u, points, dtype)
            return
        self.points = points
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        shape = (points, 1, 1)
        self.x = cp.zeros(shape, dtype=dtype)
        self.P = cp.ones(shape, dtype=dtype)
        self.Q = cp.ones(shape, dtype=dtype)
        self.B = None
        self.F = cp.ones(shape, dtype=dtype)
        self.H = cp.zeros(shape, dtype=dtype)
        self.R = cp.ones(shape, dtype=dtype)
        self._alpha_sq = cp.ones(shape, dtype=dtype)
        self.z = cp.empty(shape, dtype=dtype)
        self._task2_zq500_scalar_backend = True

    def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            original_predict(self, u, B, F, Q)
            return
        if u is not None:
            raise NotImplementedError("Control Matrix implementation in process")
        del B
        F = self.F if F is None else cp.asarray(F)
        if Q is None:
            Q = self.Q
        elif cp.isscalar(Q):
            Q = cp.ones_like(self.Q) * Q
        else:
            Q = cp.asarray(Q)
        self.x[...] = F * self.x
        self.P[...] = self._alpha_sq * F * self.P * F + Q

    def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            original_update(self, z, R, H)
            return
        if z is None:
            return
        H = self.H if H is None else cp.asarray(H)
        if R is None:
            R = self.R
        elif cp.isscalar(R):
            R = cp.ones_like(self.R) * R
        else:
            R = cp.asarray(R)
        self.z[...] = cp.asarray(z)
        residual = self.z - H * self.x
        innovation = H * self.P * H + R
        gain = self.P * H / innovation
        identity_minus_gain_h = cp.ones_like(self.P) - gain * H
        self.x[...] = self.x + gain * residual
        self.P[...] = (
            identity_minus_gain_h * self.P * identity_minus_gain_h
            + gain * R * gain
        )

    def predict_update_sequence(self: Any, observations: Any) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            raise NotImplementedError(
                "predict_update_sequence is only defined for the approved scalar Kalman layout")
        values = cp.asarray(observations, dtype=cp.float32).reshape(-1)
        if values.size == 0:
            return
        next_x, next_p = sequence_kernel(
            values,
            np.int32(values.size),
            self.x,
            self.P,
            self.F,
            self.Q,
            self._alpha_sq,
            self.H,
            self.R,
            size=1,
        )
        self.x = next_x.reshape((1, 1, 1))
        self.P = next_p.reshape((1, 1, 1))
        self.z = values[-1:].reshape((1, 1, 1))

    klass.__init__ = init
    klass.predict = predict
    klass.update = update
    klass.predict_update_sequence = predict_update_sequence


def install(cusignal: Any, cp: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    direct_convolve = _install_source_convolution(cp)
    _install_waveform_kernels(cp)
    filtering = importlib.import_module("cusignal.filtering.filtering")
    filtering.cp = _CupyProxy(cp, direct_convolve)
    wavelets = importlib.import_module("cusignal.wavelets.wavelets")
    wavelets.convolve = direct_convolve
    correlate = importlib.import_module("cusignal.convolution.correlate")
    correlate.convolve = direct_convolve
    _install_peak_helper(cp)
    _install_scalar_kalman(cp, cusignal)
    _DETAILS.update(
        implementation="cusignal_python_gpu_plus_required_zq500_adapter",
        public_cusignal_api_preserved=True,
        adapter_overhead_included_in_timing_numerator=True,
        cpu_fallback=False,
        waveform_kernels=True,
        direct_convolution=True,
        peak_helper=True,
        scalar_kalman=True,
        active_adapters=copy.deepcopy(ADAPTER_CONTRACTS),
        intermediate_double_audit=copy.deepcopy(INTERMEDIATE_DOUBLE_AUDIT),
        vendored_source_modified=False,
    )
    _INSTALLED = True


def adapter_contract() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(ADAPTER_CONTRACTS)


def adapter_status() -> dict[str, Any]:
    return {"installed": _INSTALLED, **_DETAILS}
```

### 3.11 `ZKX/Task2/task2_python/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 cuSignal Python performance/result pipeline."""
```

### 3.12 `ZKX/Task2/task2_python/step1/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 step1."""
```

### 3.13 `ZKX/Task2/task2_python/step1/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(1, "step1", Task2Config.load(args.config))
```

### 3.14 `ZKX/Task2/task2_python/step2/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 step2."""
```

### 3.15 `ZKX/Task2/task2_python/step2/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(2, "step2", Task2Config.load(args.config))
```

### 3.16 `ZKX/Task2/task2_python/step3/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 step3."""
```

### 3.17 `ZKX/Task2/task2_python/step3/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(3, "step3", Task2Config.load(args.config))
```

### 3.18 `ZKX/Task2/task2_python/step4/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 step4."""
```

### 3.19 `ZKX/Task2/task2_python/step4/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(4, "step4", Task2Config.load(args.config))
```

### 3.20 `ZKX/Task2/task2_python/step5/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 step5."""
```

### 3.21 `ZKX/Task2/task2_python/step5/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(5, "step5", Task2Config.load(args.config))
```

### 3.22 `ZKX/Task2/task2_python/step6/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2 step6."""
```

### 3.23 `ZKX/Task2/task2_python/step6/main.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(6, "step6", Task2Config.load(args.config))
```

### 3.24 `ZKX/Task2/task2_python/utils/__init__.py`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```python
"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""

from .cusignal_source_adapter import adapter_contract, adapter_status, install

__all__ = ["adapter_contract", "adapter_status", "install"]
```

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/main.py`；符号 `文件级代码`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import argparse
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`argparse`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_python/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.2 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/main.py`；符号 `文件级代码`；源码锚点 `from task2_runner import run_task2_until`。

```python
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `task2_runner`、`run_task2_until`、`demo`、`task_config`、`Task2Config`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|输入准备/数据契约|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from task2_runner import run_task2_until` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.3 main：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/main.py`；符号 `main`；源码锚点 `def main() -> int:`。

```python

def main() -> int:
    parser = argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")
    parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--evidence-id", default="pipeline")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = Task2Config.load(args.config)
    run_task2_until(args.stop_after, args.evidence_id, config, args.output_dir)
    return 0
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `main` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `6` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `7` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`cuSignal`|当前函数或调用的参数名称，接收调用者绑定的输入 `cuSignal`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `parser = argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")`；值在 `ZKX/Task2/task2_python/main.py` 当前作用域中产生或消费|
|`pipeline`|当前函数或调用的参数名称，接收调用者绑定的输入 `pipeline`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `parser = argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")`；值在 `ZKX/Task2/task2_python/main.py` 当前作用域中产生或消费|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`main`|自定义函数/可调用入口 `main`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/main.py` 中承担 `main` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`6`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))` 中，它是命令行或配置字段的默认值；源码定义了默认策略，但没有自动证明该值在所有输入上最优。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))` 中，它是命令行或配置字段的默认值；源码定义了默认策略，但没有自动证明该值在所有输入上最优。|
|`7`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))` 中，它是命令行或配置字段的默认值；源码定义了默认策略，但没有自动证明该值在所有输入上最优。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `return 0`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `def main() -> int:`：`def` 创建名为 `main` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> int` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `parser = argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--stop-after"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`type=int` 是关键字实参：先计算 `int`，再按参数名 `type` 绑定，不依赖它在形参表中的位置；`default=6` 是关键字实参：先计算 `6`，再按参数名 `default` 绑定，不依赖它在形参表中的位置；`choices=range(1, 7)` 是关键字实参：先计算 `range(1, 7)`，再按参数名 `choices` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `parser.add_argument("--evidence-id", default="pipeline")`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--evidence-id"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`default="pipeline"` 是关键字实参：先计算 `"pipeline"`，再按参数名 `default` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `parser.add_argument("--output-dir", default=".")`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--output-dir"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`default="."` 是关键字实参：先计算 `"."`，再按参数名 `default` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `config = Task2Config.load(args.config)`：这是 Python 赋值语句。解释器先完整计算右侧 `Task2Config.load(args.config)` 得到一个对象，再把名称/属性/下标 `config` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `config` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(args.stop_after, args.evidence_id, config, args.output_dir)`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`args.stop_after` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`args.evidence_id` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`config` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`args.output_dir` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return 0`：`return` 先计算 `0`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|输入准备/数据契约|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.4 main：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/main.py`；符号 `main`；源码锚点 `if __name__ == "__main__":`。

```python

if __name__ == "__main__":
    raise SystemExit(main())
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `__name__` 由 `if` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`__name__`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `__name__`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if __name__ == "__main__":`；值在 `ZKX/Task2/task2_python/main.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise SystemExit(main())`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`SystemExit(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`main()` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `main` 所在的 `ZKX/Task2/task2_python/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `main`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `dataclass` 由 `import` 声明：类型控制可表示值、可用操作和传参方式；
- `Any` 由 `import` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`from`|当前表达式读取或传递的工程名称 `from`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `from __future__ import annotations`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`dataclass`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|
|`Any`|当前表达式读取或传递的工程名称 `Any`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `from typing import Any, Callable, TypeVar`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import os`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from dataclasses import dataclass, field`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from typing import Any, Callable, TypeVar`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.6 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `main`；源码锚点 `import numpy as np`。

```python
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_PROJECT_ROOT` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`_PROJECT_ROOT`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `_PROJECT_ROOT`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `_PROJECT_ROOT = Path(__file__).resolve().parents[2]`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `_PROJECT_ROOT = Path(__file__).resolve().parents[2]` 中，它为 `_PROJECT_ROOT` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；它直接参与表达式 `from demo.task_config import Task2Config`，作用由同一表达式中的运算符决定。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(_PROJECT_ROOT))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `_PROJECT_ROOT = Path(__file__).resolve().parents[2]`：这是 Python 赋值语句。解释器先完整计算右侧 `Path(__file__).resolve().parents[2]` 得到一个对象，再把名称/属性/下标 `_PROJECT_ROOT` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_PROJECT_ROOT` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if str(_PROJECT_ROOT) not in sys.path:`：`if` 先计算条件 `str(_PROJECT_ROOT) not in sys.path` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `sys.path.insert(0, str(_PROJECT_ROOT))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(_PROJECT_ROOT)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|输入准备/数据契约|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.7 _project_cusignal_python：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `def _project_cusignal_python() -> Path:`。

```python

def _project_cusignal_python() -> Path:
    override = os.environ.get("TASK2_CUSIGNAL_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `override` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_project_cusignal_python` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`override`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `override`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `override = os.environ.get("TASK2_CUSIGNAL_PYTHON")`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`_project_cusignal_python`|自定义函数/可调用入口 `_project_cusignal_python`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_common.py` 中承担 `_project_cusignal_python` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def _project_cusignal_python() -> Path:`：`def` 创建名为 `_project_cusignal_python` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> Path` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `override = os.environ.get("TASK2_CUSIGNAL_PYTHON")`：这是 Python 赋值语句。解释器先完整计算右侧 `os.environ.get("TASK2_CUSIGNAL_PYTHON")` 得到一个对象，再把名称/属性/下标 `override` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `override` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if override:`：`if` 先计算条件 `override` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return Path(override).expanduser().resolve()`：`return` 先计算 `Path(override).expanduser().resolve()`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.8 _project_cusignal_python：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `source = Path(__file__).resolve()`。

```python
    source = Path(__file__).resolve()
    candidates = (
        source.parents[2] / "cusignal-23.08.00" / "python",
        source.parents[3] / "cusignal-23.08.00" / "python",
    )
    for candidate in candidates:
        if (candidate / "cusignal").is_dir():
            return candidate
```

**语法结构**

这个 Python 代码块由条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `source` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `candidates` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `00` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `00` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`source`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|
|`candidates`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `candidates`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `candidates = (`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `source.parents[2] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定；它直接参与表达式 `source.parents[3] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定。|
|`23.08`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `source.parents[2] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定；它直接参与表达式 `source.parents[3] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定。|
|`.00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `source.parents[2] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定；它直接参与表达式 `source.parents[3] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定。|
|`3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `source.parents[2] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定；它直接参与表达式 `source.parents[3] / "cusignal-23.08.00" / "python",`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `source = Path(__file__).resolve()`：这是 Python 赋值语句。解释器先完整计算右侧 `Path(__file__).resolve()` 得到一个对象，再把名称/属性/下标 `source` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `source` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `candidates = ( source.parents[2] / "cusignal-23.08.00" / "python", source.parents[3] / "cusignal-23.08.00" / "python", )`：这是 Python 赋值语句。解释器先完整计算右侧 `( source.parents[2] / "cusignal-23.08.00" / "python", source.parents[3] / "cusignal-23.08.00" / "python", )` 得到一个对象，再把名称/属性/下标 `candidates` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `candidates` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `for candidate in candidates:`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `if (candidate / "cusignal").is_dir():`：`if` 先计算条件 `(candidate / "cusignal").is_dir()` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return candidate`：`return` 先计算 `candidate`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.9 _project_cusignal_python：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `return candidates[-1]`。

```python
    return candidates[-1]
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `return candidates[-1]`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `return candidates[-1]`：`return` 先计算 `candidates[-1]`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `return candidates[-1]` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.10 _project_cusignal_python：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `_CUSIGNAL_SOURCE = _project_cusignal_python()`。

```python

_CUSIGNAL_SOURCE = _project_cusignal_python()
if _CUSIGNAL_SOURCE.is_dir() and str(_CUSIGNAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_CUSIGNAL_SOURCE))
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_CUSIGNAL_SOURCE` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`_CUSIGNAL_SOURCE`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(_CUSIGNAL_SOURCE))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `_CUSIGNAL_SOURCE = _project_cusignal_python()`：这是 Python 赋值语句。解释器先完整计算右侧 `_project_cusignal_python()` 得到一个对象，再把名称/属性/下标 `_CUSIGNAL_SOURCE` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_CUSIGNAL_SOURCE` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if _CUSIGNAL_SOURCE.is_dir() and str(_CUSIGNAL_SOURCE) not in sys.path:`：`if` 先计算条件 `_CUSIGNAL_SOURCE.is_dir() and str(_CUSIGNAL_SOURCE) not in sys.path` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `sys.path.insert(0, str(_CUSIGNAL_SOURCE))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(_CUSIGNAL_SOURCE)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.11 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `try:`。

```python
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
```

**语法结构**

这个 Python 代码块由条件分支、异常处理构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `try` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `cusignal` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_IMPORT_ERROR` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `else` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`as`|当前表达式读取或传递的工程名称 `as`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `import cupy as cp  # type: ignore`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`_IMPORT_ERROR`|当前名称指定的误差指标或异常状态|对应 $g-r$、其范数或语义判定|由 GPU 与 reference 差异产生，进入门禁|
|`try`|当前表达式读取或传递的工程名称 `try`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `try:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`else`|当前表达式读取或传递的工程名称 `else`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `else:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `from utils import install as install_zq500_cusignal_adapter`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `try:`：关键字语法：`try`：try 标出可能抛异常的受保护代码。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `import cupy as cp  # type: ignore`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import cusignal  # type: ignore`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `except Exception as error:`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `cp = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `cp` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `cp` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `cusignal = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `cusignal` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `cusignal` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `_IMPORT_ERROR: Exception | None = error`：这是 Python 赋值语句。解释器先完整计算右侧 `error` 得到一个对象，再把名称/属性/下标 `_IMPORT_ERROR: Exception | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_IMPORT_ERROR: Exception | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `else:`：关键字语法：`else`：else 只在前一 if/else if 未进入时执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `_IMPORT_ERROR = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `_IMPORT_ERROR` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_IMPORT_ERROR` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `try:`：关键字语法：`try`：try 标出可能抛异常的受保护代码。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `from utils import install as install_zq500_cusignal_adapter`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.12 _project_cusignal_python：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `install_zq500_cusignal_adapter(cusignal, cp)`。

```python
        install_zq500_cusignal_adapter(cusignal, cp)
    except Exception as error:
        _IMPORT_ERROR = error
```

**语法结构**

这个 Python 代码块由异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_IMPORT_ERROR` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`as`|当前表达式读取或传递的工程名称 `as`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `except Exception as error:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`_IMPORT_ERROR`|当前名称指定的误差指标或异常状态|对应 $g-r$、其范数或语义判定|由 GPU 与 reference 差异产生，进入门禁|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `install_zq500_cusignal_adapter(cusignal, cp)` 中，它作为 `install_zq500_cusignal_adapter` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `install_zq500_cusignal_adapter(cusignal, cp)`：`install_zq500_cusignal_adapter(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`cusignal` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`cp` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `except Exception as error:`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `_IMPORT_ERROR = error`：这是 Python 赋值语句。解释器先完整计算右侧 `error` 得到一个对象，再把名称/属性/下标 `_IMPORT_ERROR` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_IMPORT_ERROR` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.13 _project_cusignal_python：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `_project_cusignal_python`；源码锚点 `PI = np.pi`。

```python

PI = np.pi
TASK_STEP_FORMAL_TIMING = os.environ.get("TASK_STEP_FORMAL_TIMING") == "1"
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `PI` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `TASK_STEP_FORMAL_TIMING` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`PI`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `PI`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `PI = np.pi`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`TASK_STEP_FORMAL_TIMING`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `TASK_STEP_FORMAL_TIMING`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `TASK_STEP_FORMAL_TIMING = os.environ.get("TASK_STEP_FORMAL_TIMING") == "1"`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `TASK_STEP_FORMAL_TIMING = os.environ.get("TASK_STEP_FORMAL_TIMING") == "1"` 中，它为 `TASK_STEP_FORMAL_TIMING` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `PI = np.pi`：这是 Python 赋值语句。解释器先完整计算右侧 `np.pi` 得到一个对象，再把名称/属性/下标 `PI` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `PI` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `TASK_STEP_FORMAL_TIMING = os.environ.get("TASK_STEP_FORMAL_TIMING") == "1"`：这是 Python 赋值语句。解释器先完整计算右侧 `os.environ.get("TASK_STEP_FORMAL_TIMING") == "1"` 得到一个对象，再把名称/属性/下标 `TASK_STEP_FORMAL_TIMING` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `TASK_STEP_FORMAL_TIMING` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.14 StepEvidence：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `StepEvidence`；源码锚点 `@dataclass`。

```python

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
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `name` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `h2d_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `compute_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d2h_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_execution_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `operator_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`name`|当前表达式读取或传递的工程名称 `name`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `name: str`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`h2d_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`compute_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`d2h_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`post_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`total_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`formal_execution_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`operator_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `prep_ms: float = 0.0` 中，它为 `float` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `h2d_ms: float = 0.0` 中，它为 `float` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `compute_ms: float = 0.0` 中，它为 `float` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `@dataclass`：`@` 是 Python 装饰器语法：定义完成后把函数或类交给 `dataclass` 处理，再把处理结果绑定回原名称；例如 `@dataclass` 会依据类型注解生成初始化与比较等方法。
- 对源码锚点 `class StepEvidence:`：关键字语法：`class`：class 创建类型对象并建立其属性与方法命名空间。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `name: str`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `prep_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `prep_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `h2d_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `h2d_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `compute_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `compute_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `compute_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `d2h_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `post_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `total_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_execution_ms: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `formal_execution_ms: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_execution_ms: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `operator_ms: dict[str, float] = field(default_factory=dict)`：这是 Python 赋值语句。解释器先完整计算右侧 `field(default_factory=dict)` 得到一个对象，再把名称/属性/下标 `operator_ms: dict[str, float]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `operator_ms: dict[str, float]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.15 StepEvidence：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `StepEvidence`；源码锚点 `metrics: dict[str, float] = field(default_factory=dict)`。

```python
    metrics: dict[str, float] = field(default_factory=dict)
    output_shape: list[int] = field(default_factory=list)
    output_dtype: str = ""
    status: str = "pass"
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `metrics` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `output_shape` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `output_dtype` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `status` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`metrics`|当前表达式读取或传递的工程名称 `metrics`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `metrics: dict[str, float] = field(default_factory=dict)`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`output_shape`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`output_dtype`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`status`|当前表达式读取或传递的工程名称 `status`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `status: str = "pass"`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `metrics: dict[str, float] = field(default_factory=dict)`：这是 Python 赋值语句。解释器先完整计算右侧 `field(default_factory=dict)` 得到一个对象，再把名称/属性/下标 `metrics: dict[str, float]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `metrics: dict[str, float]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `output_shape: list[int] = field(default_factory=list)`：这是 Python 赋值语句。解释器先完整计算右侧 `field(default_factory=list)` 得到一个对象，再把名称/属性/下标 `output_shape: list[int]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `output_shape: list[int]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `output_dtype: str = ""`：这是 Python 赋值语句。解释器先完整计算右侧 `""` 得到一个对象，再把名称/属性/下标 `output_dtype: str` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `output_dtype: str` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `status: str = "pass"`：这是 Python 赋值语句。解释器先完整计算右侧 `"pass"` 得到一个对象，再把名称/属性/下标 `status: str` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `status: str` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.16 PipelineState：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `PipelineState`；源码锚点 `@dataclass`。

```python

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
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `target_waveform` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `target_component` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `interference_component` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `noise_component` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `echo` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `filtered` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `filter_taps` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`target_waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`target_component`|当前表达式读取或传递的工程名称 `target_component`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `target_component: np.ndarray | None = None`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`interference_component`|当前表达式读取或传递的工程名称 `interference_component`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `interference_component: np.ndarray | None = None`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`noise_component`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`filtered`|当前表达式读取或传递的工程名称 `filtered`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `filtered: np.ndarray | None = None`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`filter_taps`|FIR 滤波器抽头数|记作 $L$|决定卷积长度、过渡带能力与延迟|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `@dataclass`：`@` 是 Python 装饰器语法：定义完成后把函数或类交给 `dataclass` 处理，再把处理结果绑定回原名称；例如 `@dataclass` 会依据类型注解生成初始化与比较等方法。
- 对源码锚点 `class PipelineState:`：关键字语法：`class`：class 创建类型对象并建立其属性与方法命名空间。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `config: Task2Config`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `time: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `time: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `time: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `target_waveform: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `target_waveform: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `target_waveform: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `target_component: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `target_component: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `target_component: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `interference_component: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `interference_component: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `interference_component: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `noise_component: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `noise_component: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `noise_component: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `echo: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `echo: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `echo: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `filtered: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `filtered: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `filtered: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `filter_taps: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `filter_taps: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `filter_taps: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 阅读 `@dataclass` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.17 PipelineState：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `PipelineState`；源码锚点 `smoothed: np.ndarray | None = None`。

```python
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
```

**语法结构**

这个 Python 代码块由变量声明与初始化构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `smoothed` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `reference_for_correlation` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `fm_feature` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `correlation_feature` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `spectral_feature` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `wavelet_feature` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `feature_bundle` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `extrema` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `kalman_observations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `kalman_covariance` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `kalman_anchor` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `estimate` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`reference_for_correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`fm_feature`|当前名称指定的特征数组或字段|记作 $f_m[n]$；$m$ 表示特征域|由 Step4 提取/融合，Step5/6 消费|
|`correlation_feature`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`spectral_feature`|当前名称指定的特征数组或字段|记作 $f_m[n]$；$m$ 表示特征域|由 Step4 提取/融合，Step5/6 消费|
|`wavelet_feature`|当前名称指定的特征数组或字段|记作 $f_m[n]$；$m$ 表示特征域|由 Step4 提取/融合，Step5/6 消费|
|`feature_bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|
|`extrema`|当前表达式读取或传递的工程名称 `extrema`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `extrema: np.ndarray | None = None`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`kalman_observations`|按特征点截取的观测序列|记作 $z_1,\ldots,z_K$|由 Step5 特征点附近数据构造，逐项送入 update|
|`kalman_covariance`|最终估计协方差|记作 $P_K$|Kalman 输出，作为不确定度证据|
|`kalman_anchor`|局部窗口或观测序列的中心索引|记作 $a$|通常来自最强峰位置，决定左右截取范围|
|`smoothed`|当前表达式读取或传递的工程名称 `smoothed`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `smoothed: np.ndarray | None = None`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`estimate`|最终状态估计|记作 $\hat x_K$|Kalman 输出，写入任务结果和证据|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `kalman_anchor: int = 0` 中，它为 `int` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `estimate: float = 0.0` 中，它为 `float` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `estimate: float = 0.0` 中，它为 `float` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `smoothed: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `smoothed: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `smoothed: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `reference_for_correlation: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `reference_for_correlation: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `reference_for_correlation: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `fm_feature: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `fm_feature: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `fm_feature: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `correlation_feature: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `correlation_feature: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `correlation_feature: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `spectral_feature: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `spectral_feature: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `spectral_feature: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `wavelet_feature: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `wavelet_feature: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `wavelet_feature: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `feature_bundle: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `feature_bundle: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `feature_bundle: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `extrema: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `extrema: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `extrema: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman_observations: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `kalman_observations: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman_observations: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman_covariance: np.ndarray | None = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `kalman_covariance: np.ndarray | None` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman_covariance: np.ndarray | None` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman_anchor: int = 0`：这是 Python 赋值语句。解释器先完整计算右侧 `0` 得到一个对象，再把名称/属性/下标 `kalman_anchor: int` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman_anchor: int` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `estimate: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `estimate: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `estimate: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 `feature`、`extrema`、`kalman`、`estimate`。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 阅读 `smoothed: np.ndarray | None = None` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.18 PipelineState：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `PipelineState`；源码锚点 `truth: float = 0.0`。

```python
    truth: float = 0.0
```

**语法结构**

这个 Python 代码块由变量声明与初始化构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `truth` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`truth`|当前表达式读取或传递的工程名称 `truth`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `truth: float = 0.0`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `truth: float = 0.0` 中，它为 `float` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `truth: float = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `truth: float` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `truth: float` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `truth: float = 0.0` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.19 require：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `require`；源码锚点 `def require(condition: bool, message: str) -> None:`。

```python

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `require` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if not condition:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`condition`|当前函数或调用的参数名称，接收调用者绑定的输入 `condition`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def require(condition: bool, message: str) -> None:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`message`|当前函数或调用的参数名称，接收调用者绑定的输入 `message`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def require(condition: bool, message: str) -> None:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`require`|自定义函数/可调用入口 `require`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_common.py` 中承担 `require` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def require(condition: bool, message: str) -> None:`：`def` 创建名为 `require` 的函数对象；括号中的 `condition: bool, message: str` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if not condition:`：`if` 先计算条件 `not condition` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise RuntimeError(message)`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`RuntimeError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`message` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.20 require_cusignal_runtime：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `require_cusignal_runtime`；源码锚点 `def require_cusignal_runtime() -> None:`。

```python

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
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `module_file` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `expected_root` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `require_cusignal_runtime` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if _IMPORT_ERROR is not None:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`raise`|当前表达式读取或传递的工程名称 `raise`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raise RuntimeError(`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`runtime`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`device`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`module_file`|文件系统路径或文件对象|无独立数学符号|定位配置、证据、输出或源码；不参与数值算法|
|`expected_root`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `expected_root`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `expected_root = _CUSIGNAL_SOURCE.resolve()`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`require_cusignal_runtime`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `require(cp.cuda.runtime.getDeviceCount() > 0, "no CuPy-visible GPU device")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def require_cusignal_runtime() -> None:`：`def` 创建名为 `require_cusignal_runtime` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if _IMPORT_ERROR is not None:`：`if` 先计算条件 `_IMPORT_ERROR is not None` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise RuntimeError( "cuSignal Python runtime is unavailable; expected a CuPy-compatible GPU " f"runtime plus repository source at {_CUSIGNAL_SOURCE}: {_IMPORT_ERROR}" ) from _IM...`：关键字语法：`from`：from ... import ... 从模块中直接绑定指定名称。 函数调用语法：`RuntimeError(...)` 先准备括号内实参，再把控制权交给 `RuntimeError`，返回值回到调用点。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `require(cp.cuda.runtime.getDeviceCount() > 0, "no CuPy-visible GPU device")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`cp.cuda.runtime.getDeviceCount() > 0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"no CuPy-visible GPU device"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `module_file = Path(cusignal.__file__).resolve()`：这是 Python 赋值语句。解释器先完整计算右侧 `Path(cusignal.__file__).resolve()` 得到一个对象，再把名称/属性/下标 `module_file` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module_file` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `expected_root = _CUSIGNAL_SOURCE.resolve()`：这是 Python 赋值语句。解释器先完整计算右侧 `_CUSIGNAL_SOURCE.resolve()` 得到一个对象，再把名称/属性/下标 `expected_root` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `expected_root` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require( os.path.commonpath((str(module_file), str(expected_root))) == str(expected_root), f"imported cuSignal is not the repository source: {module_file}", )`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`os.path.commonpath((str(module_file), str(expected_root))) == str(expected_root)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`f"imported cuSignal is not the repository source: {module_file}"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.21 require_cusignal_runtime：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `require_cusignal_runtime`；源码锚点 `T = TypeVar("T")`。

```python

T = TypeVar("T")
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `T` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`T`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `T`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `T = TypeVar("T")`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `T = TypeVar("T")`：这是 Python 赋值语句。解释器先完整计算右侧 `TypeVar("T")` 得到一个对象，再把名称/属性/下标 `T` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `T` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.22 time_gpu：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `time_gpu`；源码锚点 `def time_gpu(function: Callable[[], T]) -> tuple[T, float]:`。

```python

def time_gpu(function: Callable[[], T]) -> tuple[T, float]:
    # Task Step formal_execution 只在整个 Step 边界同步一次。这里若继续为每个
    # 子算子 stop.synchronize()，这些诊断同步会被重复计入 Python 分子，而 C++
    # 仅在 Step 末尾同步，导致端到端口径不对称。
    if TASK_STEP_FORMAL_TIMING:
        return function(), 0.0
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `time_gpu` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`Step`|当前表达式读取或传递的工程名称 `Step`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `# Task Step formal_execution 只在整个 Step 边界同步一次。这里若继续为每个`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`return`|当前表达式读取或传递的工程名称 `return`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `return function(), 0.0`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`function`|当前函数或调用的参数名称，接收调用者绑定的输入 `function`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def time_gpu(function: Callable[[], T]) -> tuple[T, float]:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`T`|当前函数或调用的参数名称，接收调用者绑定的输入 `T`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def time_gpu(function: Callable[[], T]) -> tuple[T, float]:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`time_gpu`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `return function(), 0.0` 中，它作为 `function` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def time_gpu(function: Callable[[], T]) -> tuple[T, float]:`：`def` 创建名为 `time_gpu` 的函数对象；括号中的 `function: Callable[[], T]` 是形参表，调用时实参按位置或关键字绑定。`-> tuple[T, float]` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `# Task Step formal_execution 只在整个 Step 边界同步一次。这里若继续为每个`：这是源码注释；注释不参与执行。文字说明作者意图，后续解释仍以实际语句为准。
- 对源码锚点 `# 子算子 stop.synchronize()，这些诊断同步会被重复计入 Python 分子，而 C++`：这是源码注释；注释不参与执行。文字说明作者意图，后续解释仍以实际语句为准。
- 对源码锚点 `# 仅在 Step 末尾同步，导致端到端口径不对称。`：这是源码注释；注释不参与执行。文字说明作者意图，后续解释仍以实际语句为准。
- 对源码锚点 `if TASK_STEP_FORMAL_TIMING:`：`if` 先计算条件 `TASK_STEP_FORMAL_TIMING` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return function(), 0.0`：`return` 先计算 `function(), 0.0`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.23 time_gpu：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `time_gpu`；源码锚点 `start = cp.cuda.Event()`。

```python
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    result = function()
    stop.record()
    stop.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start, stop))
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `start` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `stop` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `result` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`start`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `start`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `start = cp.cuda.Event()`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`stop`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `stop`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `stop = cp.cuda.Event()`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `start = cp.cuda.Event()`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.cuda.Event()` 得到一个对象，再把名称/属性/下标 `start` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `start` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `stop = cp.cuda.Event()`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.cuda.Event()` 得到一个对象，再把名称/属性/下标 `stop` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `stop` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `start.record()`：`start.record(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `result = function()`：这是 Python 赋值语句。解释器先完整计算右侧 `function()` 得到一个对象，再把名称/属性/下标 `result` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `result` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `stop.record()`：`stop.record(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `stop.synchronize()`：`stop.synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return result, float(cp.cuda.get_elapsed_time(start, stop))`：`return` 先计算 `result, float(cp.cuda.get_elapsed_time(start, stop))`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于耗时或稳定性证据：它界定测量区间、迭代次数、构建 target 或资源采样，输出统计量而不是任务算法的新数学结果。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.24 synchronize：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `synchronize`；源码锚点 `def synchronize() -> None:`。

```python

def synchronize() -> None:
    if TASK_STEP_FORMAL_TIMING:
        return
    cp.cuda.get_current_stream().synchronize()
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `synchronize` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if TASK_STEP_FORMAL_TIMING:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`return`|当前表达式读取或传递的工程名称 `return`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `return`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`synchronize`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def synchronize() -> None:`：`def` 创建名为 `synchronize` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if TASK_STEP_FORMAL_TIMING:`：`if` 先计算条件 `TASK_STEP_FORMAL_TIMING` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `cp.cuda.get_current_stream().synchronize()`：`cp.cuda.get_current_stream(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`).synchronize(` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.25 wall_ms：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `wall_ms`；源码锚点 `def wall_ms(begin: float) -> float:`。

```python

def wall_ms(begin: float) -> float:
    return (time.perf_counter() - begin) * 1000.0
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `wall_ms` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1000.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`wall_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1000.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|单位换算常数：$1\,\mathrm{s}=1000\,\mathrm{ms}$。它不改变算法，只改变计时输出单位。 当前上下文：在 `return (time.perf_counter() - begin) * 1000.0` 中，它作为 `return` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def wall_ms(begin: float) -> float:`：`def` 创建名为 `wall_ms` 的函数对象；括号中的 `begin: float` 是形参表，调用时实参按位置或关键字绑定。`-> float` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `return (time.perf_counter() - begin) * 1000.0`：`return` 先计算 `(time.perf_counter() - begin) * 1000.0`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.26 as_host：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `as_host`；源码锚点 `def as_host(array: Any) -> np.ndarray:`。

```python

def as_host(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `as_host` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`array`|当前函数或调用的参数名称，接收调用者绑定的输入 `array`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def as_host(array: Any) -> np.ndarray:`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`as_host`|自定义函数/可调用入口 `as_host`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_common.py` 中承担 `as_host` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def as_host(array: Any) -> np.ndarray:`：`def` 创建名为 `as_host` 的函数对象；括号中的 `array: Any` 是形参表，调用时实参按位置或关键字绑定。`-> np.ndarray` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if isinstance(array, np.ndarray):`：`if` 先计算条件 `isinstance(array, np.ndarray)` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return array`：`return` 先计算 `array`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.27 as_host：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `as_host`；源码锚点 `return cp.asnumpy(array)`。

```python
    return cp.asnumpy(array)
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `cp`、`asnumpy`、`array`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `return cp.asnumpy(array)`：`return` 先计算 `cp.asnumpy(array)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.28 rms：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `rms`；源码锚点 `def rms(values: np.ndarray) -> float:`。

```python

def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `array` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `rms` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`array`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `array`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `array = np.asarray(values, dtype=np.float64)`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`rms`|自定义函数/可调用入口 `rms`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_common.py` 中承担 `rms` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `array = np.asarray(values, dtype=np.float64)` 中，它为 `array` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `return float(np.sqrt(np.mean(array * array))) if array.size else 0.0` 中，它作为 `float` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def rms(values: np.ndarray) -> float:`：`def` 创建名为 `rms` 的函数对象；括号中的 `values: np.ndarray` 是形参表，调用时实参按位置或关键字绑定。`-> float` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `array = np.asarray(values, dtype=np.float64)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.asarray(values, dtype=np.float64)` 得到一个对象，再把名称/属性/下标 `array` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `array` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return float(np.sqrt(np.mean(array * array))) if array.size else 0.0`：`return` 先计算 `float(np.sqrt(np.mean(array * array))) if array.size else 0.0`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.29 roughness：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `roughness`；源码锚点 `def roughness(values: np.ndarray) -> float:`。

```python

def roughness(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `array` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `roughness` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`array`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `array`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `array = np.asarray(values, dtype=np.float64)`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`roughness`|自定义函数/可调用入口 `roughness`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_common.py` 中承担 `roughness` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `array = np.asarray(values, dtype=np.float64)` 中，它为 `array` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `return float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0` 中，它作为 `float` 的实参参与当前调用。|
|`3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `return float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `return float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def roughness(values: np.ndarray) -> float:`：`def` 创建名为 `roughness` 的函数对象；括号中的 `values: np.ndarray` 是形参表，调用时实参按位置或关键字绑定。`-> float` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `array = np.asarray(values, dtype=np.float64)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.asarray(values, dtype=np.float64)` 得到一个对象，再把名称/属性/下标 `array` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `array` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0`：`return` 先计算 `float(np.sqrt(np.mean(np.diff(array, n=2) ** 2))) if array.size >= 3 else 0.0`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.30 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `runtime_metadata`；源码锚点 `def runtime_metadata() -> dict[str, Any]:`。

```python

def runtime_metadata() -> dict[str, Any]:
    require_cusignal_runtime()
    from utils import adapter_status
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `runtime_metadata` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`runtime_metadata`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def runtime_metadata() -> dict[str, Any]:`：`def` 创建名为 `runtime_metadata` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> dict[str, Any]` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `require_cusignal_runtime()`：`require_cusignal_runtime(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from utils import adapter_status`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.31 runtime_metadata：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `runtime_metadata`；源码锚点 `properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)`。

```python
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw_name = properties.get("name", "unknown")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8", errors="replace")
    return {
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `properties` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `raw_name` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `raw_name` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `8` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`properties`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `properties`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|
|`raw_name`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `raw_name`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raw_name = properties.get("name", "unknown")`；值在 `ZKX/Task2/task2_python/task2_common.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`8`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{3}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `raw_name = raw_name.decode("utf-8", errors="replace")` 中，它为 `raw_name` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)` 得到一个对象，再把名称/属性/下标 `properties` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `properties` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `raw_name = properties.get("name", "unknown")`：这是 Python 赋值语句。解释器先完整计算右侧 `properties.get("name", "unknown")` 得到一个对象，再把名称/属性/下标 `raw_name` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `raw_name` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if isinstance(raw_name, bytes):`：`if` 先计算条件 `isinstance(raw_name, bytes)` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raw_name = raw_name.decode("utf-8", errors="replace")`：这是 Python 赋值语句。解释器先完整计算右侧 `raw_name.decode("utf-8", errors="replace")` 得到一个对象，再把名称/属性/下标 `raw_name` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `raw_name` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return {`：`return` 先计算 `{`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.32 runtime_metadata：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `runtime_metadata`；源码锚点 `"python": sys.version.split()[0],`。

```python
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
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `00` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `"python": sys.version.split()[0],` 中，它作为 `sys.version.split` 的实参参与当前调用；在 `"cusignal": getattr(cusignal, "__version__", "23.08.00-source"),` 中，它作为 `getattr` 的实参参与当前调用；在 `"zq500_cusignal_adapter": adapter_status(),` 中，它作为 `adapter_status` 的实参参与当前调用。|
|`23.08`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `"cusignal": getattr(cusignal, "__version__", "23.08.00-source"),` 中，它作为 `getattr` 的实参参与当前调用。|
|`.00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `"cusignal": getattr(cusignal, "__version__", "23.08.00-source"),` 中，它作为 `getattr` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `"python": sys.version.split()[0], "numpy": np.__version__, "cupy": cp.__version__, "cusignal": getattr(cusignal, "__version__", "23.08.00-source"), "cusignal_source": str(_CUSIG...`：函数调用语法：`sys.version.split(...)` 先准备括号内实参，再把控制权交给 `sys.version.split`，返回值回到调用点；`getattr(...)` 先准备括号内实参，再把控制权交给 `getattr`，返回值回到调用点；`str(...)` 先准备括号内实参，再把控制权交给 `str`，返回值回到调用点；`Path(...)` 先准备括号内实参，再把控制权交给 `Path`，返回值回到调用点；`resolve(...)` 先准备括号内实参，再把控制权交给 `resolve`，返回值回到调用点；`int(...)` 先准备括号内实参，再把控制权交给 `int`，返回值回到调用点；`cp.cuda.Device(...)` 先准备括号内实参，再把控制权交给 `cp.cuda.Device`，返回值回到调用点；`cp.cuda.runtime.getDeviceCount(...)` 先准备括号内实参，再把控制权交给 `cp.cuda.runtime.getDeviceCount`，返回值回到调用点；`cp.cuda.runtime.runtimeGetVersion(...)` 先准备括号内实参，再把控制权交给 `cp.cuda.runtime.runtimeGetVersion`，返回值回到调用点；`cp.cuda.runtime.driverGetVersion(...)` 先准备括号内实参，再把控制权交给 `cp.cuda.runtime.driverGetVersion`，返回值回到调用点；`adapter_status(...)` 先准备括号内实参，再把控制权交给 `adapter_status`，返回值回到调用点。 字面量与类型：`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`00` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.33 runtime_metadata：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_common.py`；符号 `runtime_metadata`；源码锚点 `"evidence_git_commit": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_COMMIT", "unbound"),`。

```python
        "evidence_git_commit": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_COMMIT", "unbound"),
        "evidence_git_dirty": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_DIRTY", "unknown"),
        "evidence_test_time": os.environ.get("TASK2_PYTHON_EVIDENCE_TEST_TIME", "unbound"),
    }
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence_git_commit`、`os`、`environ`、`get`、`TASK2_PYTHON_EVIDENCE_GIT_COMMIT`、`unbound`、`evidence_git_dirty`、`TASK2_PYTHON_EVIDENCE_GIT_DIRTY`、`unknown`、`evidence_test_time`、`TASK2_PYTHON_EVIDENCE_TEST_TIME`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"evidence_git_commit": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_COMMIT", "unbound"), "evidence_git_dirty": os.environ.get("TASK2_PYTHON_EVIDENCE_GIT_DIRTY", "unknown"), "eviden...`：函数调用语法：`os.environ.get(...)` 先准备括号内实参，再把控制权交给 `os.environ.get`，返回值回到调用点。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.34 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `runtime_metadata`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `Any` 由 `import` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`from`|当前表达式读取或传递的工程名称 `from`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `from __future__ import annotations`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`Any`|当前表达式读取或传递的工程名称 `Any`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `from typing import Any, Callable`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import json`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from dataclasses import asdict`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from typing import Any, Callable`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `runtime_metadata` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.35 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `runtime_metadata`；源码锚点 `import numpy as np`。

```python
import numpy as np

from step1.step1 import run_step1
from step2.step2 import run_step2
from step3.step3 import run_step3
from step4.step4 import run_step4
from step5.step5 import run_step5
from step6.step6 import run_step6
from task2_common import (
    PipelineState,
    StepEvidence,
    require,
    require_cusignal_runtime,
    runtime_metadata,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`step1`、`run_step1`、`step2`、`run_step2`、`step3`、`run_step3`、`step4`、`run_step4`、`step5`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from step1.step1 import run_step1`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from step2.step2 import run_step2`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from step3.step3 import run_step3`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from step4.step4 import run_step4`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from step5.step5 import run_step5`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from step6.step6 import run_step6`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, require, require_cusignal_runtime, runtime_metadata, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|公共基础设施|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.36 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `runtime_metadata`；源码锚点 `from demo.task_config import Task2Config`。

```python
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `demo`、`task_config`、`Task2Config`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|输入准备/数据契约|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from demo.task_config import Task2Config` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.37 runtime_metadata：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `runtime_metadata`；源码锚点 `RUN_STEPS: list[Callable[[PipelineState], StepEvidence]] = [`。

```python

RUN_STEPS: list[Callable[[PipelineState], StepEvidence]] = [
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    run_step5,
    run_step6,
]
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `RUN_STEPS` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`list`|当前表达式读取或传递的工程名称 `list`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `RUN_STEPS: list[Callable[[PipelineState], StepEvidence]] = [`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`RUN_STEPS`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `RUN_STEPS: list[Callable[[PipelineState], StepEvidence]] = [ run_step1, run_step2, run_step3, run_step4, run_step5, run_step6, ]`：这是 Python 赋值语句。解释器先完整计算右侧 `[ run_step1, run_step2, run_step3, run_step4, run_step5, run_step6, ]` 得到一个对象，再把名称/属性/下标 `RUN_STEPS: list[Callable[[PipelineState], StepEvidence]]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `RUN_STEPS: list[Callable[[PipelineState], StepEvidence]]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `runtime_metadata` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `RUN_STEPS: list[Callable[[PipelineState], StepEvidence]] = [` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.38 _json_safe：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `_json_safe`；源码锚点 `def _json_safe(value: Any) -> Any:`。

```python

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_json_safe` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if isinstance(value, dict):`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`_json_safe`|自定义函数/可调用入口 `_json_safe`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_runner.py` 中承担 `_json_safe` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def _json_safe(value: Any) -> Any:`：`def` 创建名为 `_json_safe` 的函数对象；括号中的 `value: Any` 是形参表，调用时实参按位置或关键字绑定。`-> Any` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if isinstance(value, dict):`：`if` 先计算条件 `isinstance(value, dict)` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return {str(key): _json_safe(item) for key, item in value.items()}`：`return` 先计算 `{str(key): _json_safe(item) for key, item in value.items()}`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `_json_safe` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.39 isinstance：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `isinstance`；源码锚点 `if isinstance(value, (list, tuple)):`。

```python
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # bool is a subclass of int; preserve JSON booleans before integer handling.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `int` 由 `of` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`is`|当前表达式读取或传递的工程名称 `is`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `# bool is a subclass of int; preserve JSON booleans before integer handling.`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`booleans`|当前表达式读取或传递的工程名称 `booleans`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `# bool is a subclass of int; preserve JSON booleans before integer handling.`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`int`|当前表达式读取或传递的工程名称 `int`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `# bool is a subclass of int; preserve JSON booleans before integer handling.`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `if isinstance(value, (list, tuple)):`：`if` 先计算条件 `isinstance(value, (list, tuple))` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return [_json_safe(item) for item in value]`：`return` 先计算 `[_json_safe(item) for item in value]`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。
- 对源码锚点 `# bool is a subclass of int; preserve JSON booleans before integer handling.`：这是源码注释；注释不参与执行。文字说明作者意图，后续解释仍以实际语句为准。
- 对源码锚点 `if isinstance(value, (bool, np.bool_)):`：`if` 先计算条件 `isinstance(value, (bool, np.bool_))` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return bool(value)`：`return` 先计算 `bool(value)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `isinstance` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.40 isinstance：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `isinstance`；源码锚点 `if isinstance(value, (np.floating, float)):`。

```python
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `isinstance`、`value`、`np`、`floating`、`float`、`integer`、`int`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `if isinstance(value, (np.floating, float)):`：`if` 先计算条件 `isinstance(value, (np.floating, float))` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return float(value)`：`return` 先计算 `float(value)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。
- 对源码锚点 `if isinstance(value, (np.integer, int)):`：`if` 先计算条件 `isinstance(value, (np.integer, int))` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return int(value)`：`return` 先计算 `int(value)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `isinstance` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.41 isinstance：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `isinstance`；源码锚点 `return value`。

```python
    return value
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `value`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `return value`：`return` 先计算 `value`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `isinstance` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `return value` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.42 _step_json：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `_step_json`；源码锚点 `def _step_json(evidence: StepEvidence) -> dict[str, Any]:`。

```python

def _step_json(evidence: StepEvidence) -> dict[str, Any]:
    raw = asdict(evidence)
    return {
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `raw` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_step_json` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`raw`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `raw`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raw = asdict(evidence)`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`_step_json`|自定义函数/可调用入口 `_step_json`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_runner.py` 中承担 `_step_json` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def _step_json(evidence: StepEvidence) -> dict[str, Any]:`：`def` 创建名为 `_step_json` 的函数对象；括号中的 `evidence: StepEvidence` 是形参表，调用时实参按位置或关键字绑定。`-> dict[str, Any]` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `raw = asdict(evidence)`：这是 Python 赋值语句。解释器先完整计算右侧 `asdict(evidence)` 得到一个对象，再把名称/属性/下标 `raw` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `raw` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return {`：`return` 先计算 `{`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `_step_json` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.43 _step_json：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `_step_json`；源码锚点 `"name": evidence.name,`。

```python
        "name": evidence.name,
        "status": evidence.status,
        "timing_ms": {
            "prepare": raw["prep_ms"],
            "h2d": raw["h2d_ms"],
            "compute": raw["compute_ms"],
            "d2h": raw["d2h_ms"],
            "postprocess": raw["post_ms"],
            "formal_execution": raw["formal_execution_ms"],
            "total": raw["total_ms"],
        },
        "operators_ms": raw["operator_ms"],
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `name`、`evidence`、`status`、`timing_ms`、`prepare`、`raw`、`prep_ms`、`h2d`、`h2d_ms`、`compute`、`compute_ms`、`d2h`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"name": evidence.name, "status": evidence.status, "timing_ms": {`：运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `"prepare": raw["prep_ms"], "h2d": raw["h2d_ms"], "compute": raw["compute_ms"], "d2h": raw["d2h_ms"], "postprocess": raw["post_ms"], "formal_execution": raw["formal_execution_ms"...`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `_step_json` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"name": evidence.name,` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.44 _step_json：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `_step_json`；源码锚点 `"metrics": raw["metrics"],`。

```python
        "metrics": raw["metrics"],
        "output": {"shape": raw["output_shape"], "dtype": raw["output_dtype"]},
    }
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `metrics`、`raw`、`output`、`shape`、`output_shape`、`dtype`、`output_dtype`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"metrics": raw["metrics"], "output": {"shape": raw["output_shape"], "dtype": raw["output_dtype"]}, }`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `_step_json` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"metrics": raw["metrics"],` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.45 run_task2_until：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `run_task2_until`；源码锚点 `def run_task2_until(`。

```python

def run_task2_until(
    stop_after: int,
    evidence_id: str,
    config: Task2Config,
    output_directory: str | Path = ".",
    write_json: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    require_cusignal_runtime()
    require(1 <= stop_after <= 6, "Task2 stop_after must be in [1,6]")
    require(bool(evidence_id), "Task2 evidence_id must be nonempty")
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `stop_after` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `write_json` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `quiet` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_task2_until` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `6` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `6` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`stop_after`|当前函数或调用的参数名称，接收调用者绑定的输入 `stop_after`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `stop_after: int,`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`evidence_id`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`write_json`|当前函数或调用的参数名称，接收调用者绑定的输入 `write_json`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `write_json: bool = True,`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`quiet`|当前函数或调用的参数名称，接收调用者绑定的输入 `quiet`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `quiet: bool = False,`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`output_directory`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`run_task2_until`|自定义函数/可调用入口 `run_task2_until`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/task2_runner.py` 中承担 `run_task2_until` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `require(1 <= stop_after <= 6, "Task2 stop_after must be in [1,6]")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`6`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(1 <= stop_after <= 6, "Task2 stop_after must be in [1,6]")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def run_task2_until( stop_after: int, evidence_id: str, config: Task2Config, output_directory: str | Path = ".", write_json: bool = True, quiet: bool = False, ) -> dict[str, Any]:`：`def` 创建名为 `run_task2_until` 的函数对象；括号中的 `stop_after: int, evidence_id: str, config: Task2Config, output_directory: str | Path = ".", write_json: bool = True, quiet: bool = False,` 是形参表，调用时实参按位置或关键字绑定。`-> dict[str, Any]` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `require_cusignal_runtime()`：`require_cusignal_runtime(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(1 <= stop_after <= 6, "Task2 stop_after must be in [1,6]")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`1 <= stop_after <= 6` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"Task2 stop_after must be in [1,6]"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(bool(evidence_id), "Task2 evidence_id must be nonempty")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`bool(evidence_id)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"Task2 evidence_id must be nonempty"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|输入准备/数据契约|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.46 run_task2_until：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `run_task2_until`；源码锚点 `pipeline_begin = time.perf_counter()`。

```python
    pipeline_begin = time.perf_counter()
    state = PipelineState(config=config)
    steps = [RUN_STEPS[index](state) for index in range(stop_after)]
    pipeline_total_ms = wall_ms(pipeline_begin)
    runtime = runtime_metadata()
    result = _json_safe(
        {
            "task": "Task2",
            "implementation": "cusignal_python_23.08.00",
            "performance_identity": "cuSignal Python GPU + required ZQ500 adapter",
            "evidence_id": evidence_id,
            "requested_steps": stop_after,
            "completed_steps": len(steps),
            "status": "pass",
            "git_commit": runtime["evidence_git_commit"],
            "git_dirty": runtime["evidence_git_dirty"],
            "test_time": runtime["evidence_test_time"],
            "runtime": runtime,
            "config": {
                "path": str(config.path),
                "sha256": config.sha256,
                "samples": config.samples,
                "sample_rate_hz": config.sample_rate_hz,
                "target_delay_samples": config.target_delay_samples,
                "noise_seed": config.noise_seed,
                "target_weight": config.target_weight,
                "gaussian_weight": config.gaussian_weight,
                "sawtooth_weight": config.sawtooth_weight,
                "square_weight": config.square_weight,
                "noise_amplitude": config.noise_amplitude,
                "filter_taps": config.filter_taps,
                "filter_cutoff_hz": config.filter_cutoff_hz,
                "step3_crop_each": config.step3_crop_each,
                "fusion_weights": {
                    "fm": config.fusion_weight_fm,
                    "correlation": config.fusion_weight_correlation,
                    "spectral": config.fusion_weight_spectral,
                    "wavelet": config.fusion_weight_wavelet,
                },
                "argrelextrema_order": config.argrelextrema_order,
                "kalman_observation_count": config.kalman_observation_count,
                "kalman": {"F": config.kalman_F, "Q": config.kalman_Q,
                           "H": config.kalman_H, "R": config.kalman_R},
            },
            "input_contract": {
                "dtype": "FP32",
                "shape": [config.samples],
                "sample_rate_hz": config.sample_rate_hz,
                "target_delay_samples": config.target_delay_samples,
                "noise_seed": config.noise_seed,
                "step3_output_samples": config.samples - 2 * config.step3_crop_each,
                "feature_fusion_weights": {
                    "fm": config.fusion_weight_fm,
                    "correlation": config.fusion_weight_correlation,
                    "spectral": config.fusion_weight_spectral,
                    "wavelet": config.fusion_weight_wavelet,
                },
                "argrelextrema": {"axis": 0, "order": config.argrelextrema_order, "mode": "clip"},
            },
            "timing_policy": {
                "gpu_clock": "cupy.cuda.Event",
                "step_clock": "time.perf_counter",
                "h2d_scope": "cp.asarray followed by current-stream synchronization",
                "pipeline_handoff": "Host PipelineState with explicit H2D/D2H, matching task2_gpu_cpu",
                "d2h_scope": "formal step output copied for downstream step and evidence",
                "postprocess_scope": "mandatory Host handoff where applicable plus semantic/task-effect checks; no CPU precision reference",
                "formal_execution_scope": "prepare + H2D + GPU operators + required D2H and mandatory handoff; excludes semantic checks",
                "adapter_scope": "all required Task2/utils ZQ500 adapter execution is inside the cuSignal Python GPU timing numerator",
            },
            "precision_policy": {
                "cpu_validation": False,
                "cpu_comparison": False,
                "reason": "Task2 Python is the cuSignal performance baseline; CPU precision evidence belongs to task2_gpu_cpu only",
            },
            "formal_apis": [
                "cusignal.chirp/gausspulse/sawtooth/square",
                "cusignal.hamming/firwin/firfilter",
                "cusignal.cubic/firfilter",
                "cusignal.fm_demod/correlate/spectrogram/cwt/ricker",
                "cusignal.argrelextrema",
                "cusignal.KalmanFilter.predict/update",
            ],
            "pipeline_total_ms": pipeline_total_ms,
            "sum_step_total_ms": sum(step.total_ms for step in steps),
            "steps": [_step_json(step) for step in steps],
        }
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、循环、lambda、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `pipeline_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `state` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `steps` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `pipeline_total_ms` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `runtime` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `result` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `08.00` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`GPU`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`adapter`|当前函数或调用的参数名称，接收调用者绑定的输入 `adapter`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"performance_identity": "cuSignal Python GPU + required ZQ500 adapter",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`PipelineState`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`handoff`|当前表达式读取或传递的工程名称 `handoff`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"postprocess_scope": "mandatory Host handoff where applicable plus semantic/task-effect checks; no CPU precision reference",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`precision`|当前表达式读取或传递的工程名称 `precision`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"postprocess_scope": "mandatory Host handoff where applicable plus semantic/task-effect checks; no CPU precision reference",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`operators`|当前表达式读取或传递的工程名称 `operators`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"formal_execution_scope": "prepare + H2D + GPU operators + required D2H and mandatory handoff; excludes semantic checks",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`and`|当前表达式读取或传递的工程名称 `and`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"d2h_scope": "formal step output copied for downstream step and evidence",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`Python`|当前函数或调用的参数名称，接收调用者绑定的输入 `Python`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"performance_identity": "cuSignal Python GPU + required ZQ500 adapter",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`pipeline_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`steps`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `steps`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `steps = [RUN_STEPS[index](state) for index in range(stop_after)]`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`pipeline_total_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|
|`runtime`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`3.08`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `"implementation": "cusignal_python_23.08.00",`，作用由同一表达式中的运算符决定。|
|`.00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"implementation": "cusignal_python_23.08.00",`，作用由同一表达式中的运算符决定。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"implementation": "cusignal_python_23.08.00",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"performance_identity": "cuSignal Python GPU + required ZQ500 adapter",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"adapter_scope": "all required Task2/utils ZQ500 adapter execution is inside the cuSignal Python GPU timing numerator",`，作用由同一表达式中的运算符决定。|
|`56`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `"sha256": config.sha256,`，作用由同一表达式中的运算符决定。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：它直接参与表达式 `"task": "Task2",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"implementation": "cusignal_python_23.08.00",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"sha256": config.sha256,`，作用由同一表达式中的运算符决定。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"implementation": "cusignal_python_23.08.00",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"performance_identity": "cuSignal Python GPU + required ZQ500 adapter",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"argrelextrema": {"axis": 0, "order": config.argrelextrema_order, "mode": "clip"},`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `pipeline_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `pipeline_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `pipeline_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state = PipelineState(config=config)`：这是 Python 赋值语句。解释器先完整计算右侧 `PipelineState(config=config)` 得到一个对象，再把名称/属性/下标 `state` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `steps = [RUN_STEPS[index](state) for index in range(stop_after)]`：这是 Python 赋值语句。解释器先完整计算右侧 `[RUN_STEPS[index](state) for index in range(stop_after)]` 得到一个对象，再把名称/属性/下标 `steps` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `steps` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `pipeline_total_ms = wall_ms(pipeline_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(pipeline_begin)` 得到一个对象，再把名称/属性/下标 `pipeline_total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `pipeline_total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `runtime = runtime_metadata()`：这是 Python 赋值语句。解释器先完整计算右侧 `runtime_metadata()` 得到一个对象，再把名称/属性/下标 `runtime` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `runtime` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `result = _json_safe( { "task": "Task2", "implementation": "cusignal_python_23.08.00", "performance_identity": "cuSignal Python GPU + required ZQ500 adapter", "evidence_id": evid...`：这是 Python 赋值语句。解释器先完整计算右侧 `_json_safe( { "task": "Task2", "implementation": "cusignal_python_23.08.00", "performance_identity": "cuSignal Python GPU + required ZQ500 adapter", "evidence_id": evidence_id, "requested_steps": stop_after, "completed_steps": len(steps), "status": "pass", "git_commit": runtime["evidence_git_commit"], "git_dirty": runtime["evidence_git_dirty"], "test_time": runtime["evidence_test_time"], "runtime": runtime, "config": { "path": str(config.path), "sha256": config.sha256, "samples": config.samples, "sample_rate_hz": config.sample_rate_hz, "target_delay_samples": config.target_delay_samples, "noise_seed": config.noise_seed, "target_weight": config.target_weight, "gaussian_weight": config.gaussian_weight, "sawtooth_weight": config.sawtooth_weight, "square_weight": config.square_weight, "noise_amplitude": config.noise_amplitude, "filter_taps": config.filter_taps, "filter_cutoff_hz": config.filter_cutoff_hz, "step3_crop_each": config.step3_crop_each, "fusion_weights": { "fm": config.fusion_weight_fm, "correlation": config.fusion_weight_correlation, "spectral": config.fusion_weight_spectral, "wavelet": config.fusion_weight_wavelet, }, "argrelextrema_order": config.argrelextrema_order, "kalman_observation_count": config.kalman_observation_count, "kalman": {"F": config.kalman_F, "Q": config.kalman_Q, "H": config.kalman_H, "R": config.kalman_R}, }, "input_contract": { "dtype": "FP32", "shape": [config.samples], "sample_rate_hz": config.sample_rate_hz, "target_delay_samples": config.target_delay_samples, "noise_seed": config.noise_seed, "step3_output_samples": config.samples - 2 * config.step3_crop_each, "feature_fusion_weights": { "fm": config.fusion_weight_fm, "correlation": config.fusion_weight_correlation, "spectral": config.fusion_weight_spectral, "wavelet": config.fusion_weight_wavelet, }, "argrelextrema": {"axis": 0, "order": config.argrelextrema_order, "mode": "clip"}, }, "timing_policy": { "gpu_clock": "cupy.cuda.Event", "step_clock": "time.perf_counter", "h2d_scope": "cp.asarray followed by current-stream synchronization", "pipeline_handoff": "Host PipelineState with explicit H2D/D2H, matching task2_gpu_cpu", "d2h_scope": "formal step output copied for downstream step and evidence", "postprocess_scope": "mandatory Host handoff where applicable plus semantic/task-effect checks; no CPU precision reference", "formal_execution_scope": "prepare + H2D + GPU operators + required D2H and mandatory handoff; excludes semantic checks", "adapter_scope": "all required Task2/utils ZQ500 adapter execution is inside the cuSignal Python GPU timing numerator", }, "precision_policy": { "cpu_validation": False, "cpu_comparison": False, "reason": "Task2 Python is the cuSignal performance baseline; CPU precision evidence belongs to task2_gpu_cpu only", }, "formal_apis": [ "cusignal.chirp/gausspulse/sawtooth/square", "cusignal.hamming/firwin/firfilter", "cusignal.cubic/firfilter", "cusignal.fm_demod/correlate/spectrogram/cwt/ricker", "cusignal.argrelextrema", "cusignal.KalmanFilter.predict/update", ], "pipeline_total_ms": pipeline_total_ms, "sum_step_total_ms": sum(step.total_ms for step in steps), "steps": [_step_json(step) for step in steps], } )` 得到一个对象，再把名称/属性/下标 `result` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `result` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`gausspulse`、`sawtooth`、`square`、`firwin`、`firfilter`、`hamming`、`cubic`、`fm_demod`、`correlate`、`spectrogram`、`cwt`、`ricker`、`feature`、`argrelextrema`、`kalman`。 本块直接出现 4 种波形符号：`chirp`、`gausspulse`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.47 run_task2_until：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `run_task2_until`；源码锚点 `if write_json:`。

```python
    if write_json:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        destination = output_path / f"task2_{evidence_id}_evidence.json"
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not quiet:
        print(f"[TASK2_PYTHON][CONFIG] path={config.path} sha256={config.sha256}")
        for step in steps:
            print(
                f"[TASK2_PYTHON][{step.name}][TIMING] prep_ms={step.prep_ms:.6f} "
                f"h2d_ms={step.h2d_ms:.6f} compute_ms={step.compute_ms:.6f} "
                f"d2h_ms={step.d2h_ms:.6f} post_ms={step.post_ms:.6f} "
                f"formal_execution_ms={step.formal_execution_ms:.6f} total_ms={step.total_ms:.6f}"
            )
```

**语法结构**

这个 Python 代码块由条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `output_path` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `destination` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `encoding` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `8` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `6f` 是数值字面量；F 指定 float，而非默认 double；
- `6f` 是数值字面量；F 指定 float，而非默认 double；
- `6f` 是数值字面量；F 指定 float，而非默认 double；
- `6f` 是数值字面量；F 指定 float，而非默认 double；
- `6f` 是数值字面量；F 指定 float，而非默认 double；
- `6f` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output_path`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`destination`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `destination`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `destination = output_path / f"task2_{evidence_id}_evidence.json"`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`encoding`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `encoding`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `encoding="utf-8",`；值在 `ZKX/Task2/task2_python/task2_runner.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `destination = output_path / f"task2_{evidence_id}_evidence.json"` 中，它为 `destination` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",` 中，它作为 `json.dumps` 的实参参与当前调用；在 `print(f"[TASK2_PYTHON][CONFIG] path={config.path} sha256={config.sha256}")` 中，它作为 `print` 的实参参与当前调用。|
|`8`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{3}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `encoding="utf-8",` 中，它为 `encoding` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`56`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `print(f"[TASK2_PYTHON][CONFIG] path={config.path} sha256={config.sha256}")` 中，它作为 `print` 的实参参与当前调用。|
|`.6f`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `f"[TASK2_PYTHON][{step.name}][TIMING] prep_ms={step.prep_ms:.6f} "`，作用由同一表达式中的运算符决定；它直接参与表达式 `f"h2d_ms={step.h2d_ms:.6f} compute_ms={step.compute_ms:.6f} "`，作用由同一表达式中的运算符决定；它直接参与表达式 `f"d2h_ms={step.d2h_ms:.6f} post_ms={step.post_ms:.6f} "`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `if write_json:`：`if` 先计算条件 `write_json` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `output_path = Path(output_directory)`：这是 Python 赋值语句。解释器先完整计算右侧 `Path(output_directory)` 得到一个对象，再把名称/属性/下标 `output_path` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `output_path` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `output_path.mkdir(parents=True, exist_ok=True)`：`output_path.mkdir(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`parents=True` 是关键字实参：先计算 `True`，再按参数名 `parents` 绑定，不依赖它在形参表中的位置；`exist_ok=True` 是关键字实参：先计算 `True`，再按参数名 `exist_ok` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `destination = output_path / f"task2_{evidence_id}_evidence.json"`：这是 Python 赋值语句。解释器先完整计算右侧 `output_path / f"task2_{evidence_id}_evidence.json"` 得到一个对象，再把名称/属性/下标 `destination` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `destination` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `destination.write_text( json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", )`：`destination.write_text(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`encoding="utf-8"` 是关键字实参：先计算 `"utf-8"`，再按参数名 `encoding` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `if not quiet:`：`if` 先计算条件 `not quiet` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `print(f"[TASK2_PYTHON][CONFIG] path={config.path} sha256={config.sha256}")`：`print(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`f"[TASK2_PYTHON][CONFIG] path={config.path} sha256={config.sha256}"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `for step in steps:`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `print( f"[TASK2_PYTHON][{step.name}][TIMING] prep_ms={step.prep_ms:.6f} " f"h2d_ms={step.h2d_ms:.6f} compute_ms={step.compute_ms:.6f} " f"d2h_ms={step.d2h_ms:.6f} post_ms={step....`：`print(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`f"[TASK2_PYTHON][{step.name}][TIMING] prep_ms={step.prep_ms:.6f} " f"h2d_ms={step.h2d_ms:.6f} compute_ms={step.compute_ms:.6f} " f"d2h_ms={step.d2h_ms:.6f} post_ms={step.post_ms:.6f} " f"formal_execution_ms={step.formal_execution_ms:.6f} total_ms={step.total_ms:.6f}"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|输入准备/数据契约|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 `path`、`sha256`。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.48 run_task2_until：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/task2_runner.py`；符号 `run_task2_until`；源码锚点 `print(`。

```python
            print(
                f"[TASK2_PYTHON][{step.name}][EFFECT] metrics={len(step.metrics)} "
                f"operators={len(step.operator_ms)} status={step.status}"
            )
        print(
            f"[TASK2_PYTHON][PIPELINE] completed_steps={len(steps)} "
            f"pipeline_total_ms={pipeline_total_ms:.6f} status=pass"
        )
    return result
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `6f` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`.6f`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `f"pipeline_total_ms={pipeline_total_ms:.6f} status=pass"`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `print( f"[TASK2_PYTHON][{step.name}][EFFECT] metrics={len(step.metrics)} " f"operators={len(step.operator_ms)} status={step.status}" )`：`print(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`f"[TASK2_PYTHON][{step.name}][EFFECT] metrics={len(step.metrics)} " f"operators={len(step.operator_ms)} status={step.status}"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `print( f"[TASK2_PYTHON][PIPELINE] completed_steps={len(steps)} " f"pipeline_total_ms={pipeline_total_ms:.6f} status=pass" )`：`print(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`f"[TASK2_PYTHON][PIPELINE] completed_steps={len(steps)} " f"pipeline_total_ms={pipeline_total_ms:.6f} status=pass"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return result`：`return` 先计算 `result`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|跨步编排|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `run_task2_until` 所在的 `ZKX/Task2/task2_python/task2_runner.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.49 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_task2_until`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import time
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`time`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_task2_until` 所在的 `ZKX/Task2/task2_python/step1/step1.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.50 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `import`；源码锚点 `import numpy as np`。

```python
import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`task2_common`、`PipelineState`、`StepEvidence`、`as_host`、`cp`、`cusignal`、`require`、`synchronize`、`time_gpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, as_host, cp, cusignal, require, synchronize, time_gpu, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.51 _noise_bits：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `_noise_bits`；源码锚点 `def _noise_bits(indices, seed: int):`。

```python

def _noise_bits(indices, seed: int):
    values = cp.uint32(seed) ^ (
        indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)
    )
    values ^= values >> cp.uint32(16)
    values *= cp.uint32(2246822519)
    values ^= values >> cp.uint32(13)
    values *= cp.uint32(3266489917)
    return values ^ (values >> cp.uint32(16))
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `values` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_noise_bits` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `747796405` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2891336453` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2246822519` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `13` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3266489917` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`indices`|当前函数或调用的参数名称，接收调用者绑定的输入 `indices`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def _noise_bits(indices, seed: int):`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`_noise_bits`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `values = cp.uint32(seed) ^ (` 中，它为 `values` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)` 中，它作为 `indices.astype` 的实参参与当前调用；在 `values ^= values >> cp.uint32(16)` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`747796405`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)` 中，它作为 `indices.astype` 的实参参与当前调用。|
|`2891336453`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)` 中，它作为 `indices.astype` 的实参参与当前调用。|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `values ^= values >> cp.uint32(16)` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `return values ^ (values >> cp.uint32(16))` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2246822519`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `values *= cp.uint32(2246822519)` 中，它作为 `cp.uint32` 的实参参与当前调用。|
|`13`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`13` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)` 中，它作为 `indices.astype` 的实参参与当前调用；在 `values ^= values >> cp.uint32(13)` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`3266489917`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `values *= cp.uint32(3266489917)` 中，它作为 `cp.uint32` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def _noise_bits(indices, seed: int):`：`def` 创建名为 `_noise_bits` 的函数对象；括号中的 `indices, seed: int` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `values = cp.uint32(seed) ^ ( indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453) )`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.uint32(seed) ^ ( indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453) )` 得到一个对象，再把名称/属性/下标 `values` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `values` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `values ^= values >> cp.uint32(16)`：这是 Python 赋值语句。解释器先完整计算右侧 `values >> cp.uint32(16)` 得到一个对象，再把名称/属性/下标 `values ^` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `values ^` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `values *= cp.uint32(2246822519)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.uint32(2246822519)` 得到一个对象，再把名称/属性/下标 `values *` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `values *` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `values ^= values >> cp.uint32(13)`：这是 Python 赋值语句。解释器先完整计算右侧 `values >> cp.uint32(13)` 得到一个对象，再把名称/属性/下标 `values ^` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `values ^` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `values *= cp.uint32(3266489917)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.uint32(3266489917)` 得到一个对象，再把名称/属性/下标 `values *` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `values *` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return values ^ (values >> cp.uint32(16))`：`return` 先计算 `values ^ (values >> cp.uint32(16))`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- Python 的 `^` 也是整数按位异或，不是乘方；乘方运算符是 `**`；
- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.52 _mix_echo：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `_mix_echo`；源码锚点 `def _mix_echo(`。

```python

def _mix_echo(
    chirp, gaussian, sawtooth, square, config,
    noise_seed: int | None = None, target_weight: float | None = None,
):
    seed = config.noise_seed if noise_seed is None else noise_seed
    weight = config.target_weight if target_weight is None else target_weight
    indices = cp.arange(config.samples, dtype=cp.int32)
    source = indices - config.target_delay_samples
    target = cp.where(
        source >= 0,
        cp.float32(weight) * chirp[cp.maximum(source, 0)],
        0.0,
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `noise_seed` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `seed` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `weight` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `indices` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `source` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `target` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_mix_echo` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`else`|当前表达式读取或传递的工程名称 `else`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `seed = config.noise_seed if noise_seed is None else noise_seed`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`noise_seed`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`indices`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `indices`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `indices = cp.arange(config.samples, dtype=cp.int32)`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`source`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|
|`target`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `target`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `target = cp.where(`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`chirp`|当前函数或调用的参数名称，接收调用者绑定的输入 `chirp`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `chirp, gaussian, sawtooth, square, config,`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`gaussian`|当前函数或调用的参数名称，接收调用者绑定的输入 `gaussian`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `chirp, gaussian, sawtooth, square, config,`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`sawtooth`|当前函数或调用的参数名称，接收调用者绑定的输入 `sawtooth`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `chirp, gaussian, sawtooth, square, config,`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`square`|当前函数或调用的参数名称，接收调用者绑定的输入 `square`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `chirp, gaussian, sawtooth, square, config,`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`target_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`_mix_echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `indices = cp.arange(config.samples, dtype=cp.int32)` 中，它为 `indices` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `cp.float32(weight) * chirp[cp.maximum(source, 0)],` 中，它作为 `cp.float32` 的实参参与当前调用。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `source >= 0,` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `cp.float32(weight) * chirp[cp.maximum(source, 0)],` 中，它作为 `cp.float32` 的实参参与当前调用；它直接参与表达式 `0.0,`，作用由同一表达式中的运算符决定。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `0.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `def _mix_echo( chirp, gaussian, sawtooth, square, config, noise_seed: int | None = None, target_weight: float | None = None, ):`：`def` 创建名为 `_mix_echo` 的函数对象；括号中的 `chirp, gaussian, sawtooth, square, config, noise_seed: int | None = None, target_weight: float | None = None,` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `seed = config.noise_seed if noise_seed is None else noise_seed`：这是 Python 赋值语句。解释器先完整计算右侧 `config.noise_seed if noise_seed is None else noise_seed` 得到一个对象，再把名称/属性/下标 `seed` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `seed` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `weight = config.target_weight if target_weight is None else target_weight`：这是 Python 赋值语句。解释器先完整计算右侧 `config.target_weight if target_weight is None else target_weight` 得到一个对象，再把名称/属性/下标 `weight` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `weight` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `indices = cp.arange(config.samples, dtype=cp.int32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.arange(config.samples, dtype=cp.int32)` 得到一个对象，再把名称/属性/下标 `indices` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `indices` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `source = indices - config.target_delay_samples`：这是 Python 赋值语句。解释器先完整计算右侧 `indices - config.target_delay_samples` 得到一个对象，再把名称/属性/下标 `source` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `source` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `target = cp.where( source >= 0, cp.float32(weight) * chirp[cp.maximum(source, 0)], 0.0, )`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.where( source >= 0, cp.float32(weight) * chirp[cp.maximum(source, 0)], 0.0, )` 得到一个对象，再把名称/属性/下标 `target` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `target` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`、`noise_seed`、`target_weight`、`samples`、`target_delay_samples`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.53 _mix_echo：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `_mix_echo`；源码锚点 `interference = (`。

```python
    interference = (
        cp.float32(config.gaussian_weight) * gaussian
        + cp.float32(config.sawtooth_weight) * sawtooth.astype(cp.float32)
        + cp.float32(config.square_weight) * square.astype(cp.float32)
    )
    bits = _noise_bits(indices.astype(cp.uint32), seed)
    noise = (
        ((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)
        * cp.float32(config.noise_amplitude)
    )
    return (
        target.astype(cp.float32),
        interference.astype(cp.float32),
        noise.astype(cp.float32),
        (target + interference + noise).astype(cp.float32),
    )
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `interference` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `bits` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `noise` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0xFFFF` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `32767.5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`interference`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `interference`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `interference = (`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`bits`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `bits`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `bits = _noise_bits(indices.astype(cp.uint32), seed)`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `cp.float32(config.gaussian_weight) * gaussian` 中，它作为 `cp.float32` 的实参参与当前调用；在 `+ cp.float32(config.sawtooth_weight) * sawtooth.astype(cp.float32)` 中，它作为 `cp.float32` 的实参参与当前调用；在 `+ cp.float32(config.square_weight) * square.astype(cp.float32)` 中，它作为 `cp.float32` 的实参参与当前调用。|
|`0xFFFF`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。 当前上下文：在 `((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)` 中，它作为 `cp.uint32` 的实参参与当前调用。|
|`32767.5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。 当前上下文：在 `((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)` 中，它作为 `cp.uint32` 的实参参与当前调用。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。 当前上下文：在 `((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)` 中，它作为 `cp.uint32` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `interference = ( cp.float32(config.gaussian_weight) * gaussian + cp.float32(config.sawtooth_weight) * sawtooth.astype(cp.float32) + cp.float32(config.square_weight) * square.ast...`：这是 Python 赋值语句。解释器先完整计算右侧 `( cp.float32(config.gaussian_weight) * gaussian + cp.float32(config.sawtooth_weight) * sawtooth.astype(cp.float32) + cp.float32(config.square_weight) * square.astype(cp.float32) )` 得到一个对象，再把名称/属性/下标 `interference` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `interference` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `bits = _noise_bits(indices.astype(cp.uint32), seed)`：这是 Python 赋值语句。解释器先完整计算右侧 `_noise_bits(indices.astype(cp.uint32), seed)` 得到一个对象，再把名称/属性/下标 `bits` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `bits` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `noise = ( ((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0) * cp.float32(config.noise_amplitude) )`：这是 Python 赋值语句。解释器先完整计算右侧 `( ((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0) * cp.float32(config.noise_amplitude) )` 得到一个对象，再把名称/属性/下标 `noise` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `noise` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return ( target.astype(cp.float32), interference.astype(cp.float32), noise.astype(cp.float32), (target + interference + noise).astype(cp.float32), )`：`return` 先计算 `( target.astype(cp.float32), interference.astype(cp.float32), noise.astype(cp.float32), (target + interference + noise).astype(cp.float32), )`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`square`、`gaussian_weight`、`sawtooth_weight`、`square_weight`、`noise_amplitude`。 本块直接出现 2 种波形符号：`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.54 run_step1：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `def run_step1(state: PipelineState) -> StepEvidence:`。

```python

def run_step1(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step1")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    state.time = np.arange(config.samples, dtype=np.float32) / np.float32(config.sample_rate_hz)
    pulse_time = state.time - np.float32(0.45)
    sample_indices = np.arange(config.samples, dtype=np.float64)
    phase = (
        2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)
    ).astype(np.float32)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `evidence` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `prep_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `pulse_time` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `sample_indices` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `phase` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_step1` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0.45` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `190.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|
|`prep_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`pulse_time`|当前慢时间脉冲编号|记作 $p=\lfloor i/N_s\rfloor$|由展平索引整除每脉冲采样数得到|
|`sample_indices`|当前脉冲内快时间采样编号|记作 $n=i\bmod N_s$|由展平索引取余得到|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`run_step1`|自定义函数/可调用入口 `run_step1`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step1/step1.py` 中承担 `run_step1` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `state.time = np.arange(config.samples, dtype=np.float32) / np.float32(config.sample_rate_hz)` 中，它为 `state.time` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `pulse_time = state.time - np.float32(0.45)` 中，它为 `pulse_time` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用。|
|`0.45`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `pulse_time = state.time - np.float32(0.45)` 中，它为 `pulse_time` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `pulse_time = state.time - np.float32(0.45)` 中，它为 `pulse_time` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `sample_indices = np.arange(config.samples, dtype=np.float64)` 中，它为 `sample_indices` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用。|
|`190.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def run_step1(state: PipelineState) -> StepEvidence:`：`def` 创建名为 `run_step1` 的函数对象；括号中的 `state: PipelineState` 是形参表，调用时实参按位置或关键字绑定。`-> StepEvidence` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `config = state.config`：这是 Python 赋值语句。解释器先完整计算右侧 `state.config` 得到一个对象，再把名称/属性/下标 `config` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `config` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence = StepEvidence("step1")`：这是 Python 赋值语句。解释器先完整计算右侧 `StepEvidence("step1")` 得到一个对象，再把名称/属性/下标 `evidence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `total_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `formal_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `prep_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `prep_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.time = np.arange(config.samples, dtype=np.float32) / np.float32(config.sample_rate_hz)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.arange(config.samples, dtype=np.float32) / np.float32(config.sample_rate_hz)` 得到一个对象，再把名称/属性/下标 `state.time` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.time` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `pulse_time = state.time - np.float32(0.45)`：这是 Python 赋值语句。解释器先完整计算右侧 `state.time - np.float32(0.45)` 得到一个对象，再把名称/属性/下标 `pulse_time` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `pulse_time` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `sample_indices = np.arange(config.samples, dtype=np.float64)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.arange(config.samples, dtype=np.float64)` 得到一个对象，再把名称/属性/下标 `sample_indices` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `sample_indices` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `phase = ( 2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0) ).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `( 2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0) ).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `phase` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `phase` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `config`、`time`、`samples`、`sample_rate_hz`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.55 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `square_phase = (`。

```python
    square_phase = (
        2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)
    ).astype(np.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_time = cp.asarray(state.time)
    d_pulse_time = cp.asarray(pulse_time)
    d_phase = cp.asarray(phase)
    d_square_phase = cp.asarray(square_phase)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_chirp, chirp_ms = time_gpu(
        lambda: cusignal.chirp(
            d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0
        )
    )
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `square_phase` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `h2d_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_time` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_pulse_time` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_phase` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_square_phase` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `125.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `20.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `70.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square_phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`h2d_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`d_time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`d_pulse_time`|当前慢时间脉冲编号|记作 $p=\lfloor i/N_s\rfloor$|由展平索引整除每脉冲采样数得到|
|`d_phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`d_square_phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `lambda: cusignal.chirp(`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用。|
|`125.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用；在 `d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0` 中，它作为 `max` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)` 中，它作为 `np.remainder` 的实参参与当前调用；在 `).astype(np.float32)` 中，它作为 `astype` 的实参参与当前调用；在 `h2d_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用。|
|`20.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0` 中，它作为 `max` 的实参参与当前调用。|
|`70.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0` 中，它作为 `max` 的实参参与当前调用。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0` 中，它作为 `max` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `square_phase = ( 2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0) ).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `( 2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0) ).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `square_phase` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `square_phase` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.prep_ms = wall_ms(prep_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(prep_begin)` 得到一个对象，再把名称/属性/下标 `evidence.prep_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.prep_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `h2d_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `h2d_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_time = cp.asarray(state.time)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(state.time)` 得到一个对象，再把名称/属性/下标 `d_time` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_time` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_pulse_time = cp.asarray(pulse_time)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(pulse_time)` 得到一个对象，再把名称/属性/下标 `d_pulse_time` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_pulse_time` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_phase = cp.asarray(phase)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(phase)` 得到一个对象，再把名称/属性/下标 `d_phase` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_phase` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_square_phase = cp.asarray(square_phase)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(square_phase)` 得到一个对象，再把名称/属性/下标 `d_square_phase` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_square_phase` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `synchronize()`：`synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(h2d_begin)` 得到一个对象，再把名称/属性/下标 `evidence.h2d_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.h2d_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_chirp, chirp_ms = time_gpu( lambda: cusignal.chirp( d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0 ) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.chirp( d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0 ) )` 得到一个对象，再把名称/属性/下标 `d_chirp, chirp_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_chirp, chirp_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`square`、`sample_rate_hz`、`time`、`samples`。 本块直接出现 2 种波形符号：`chirp`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.56 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `d_gaussian, gaussian_ms = time_gpu(`。

```python
    d_gaussian, gaussian_ms = time_gpu(
        lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)
    )
    d_saw, saw_ms = time_gpu(
        lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65))
    )
    d_square, square_ms = time_gpu(
        lambda: cusignal.square(d_square_phase, duty=np.float32(0.35))
    )
    outputs, mix_ms = time_gpu(lambda: _mix_echo(d_chirp, d_gaussian, d_saw, d_square, config))
    d_target, d_interference, d_noise, d_echo = outputs
    evidence.operator_ms = {
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `170.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.20` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.65` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`170.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)` 中，它作为 `cusignal.gausspulse` 的实参参与当前调用。|
|`0.20`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)` 中，它作为 `cusignal.gausspulse` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)` 中，它作为 `cusignal.gausspulse` 的实参参与当前调用；在 `lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65))` 中，它作为 `cusignal.sawtooth` 的实参参与当前调用；在 `lambda: cusignal.square(d_square_phase, duty=np.float32(0.35))` 中，它作为 `cusignal.square` 的实参参与当前调用。|
|`0.65`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65))` 中，它作为 `cusignal.sawtooth` 的实参参与当前调用。|
|`0.35`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `lambda: cusignal.square(d_square_phase, duty=np.float32(0.35))` 中，它作为 `cusignal.square` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `d_gaussian, gaussian_ms = time_gpu( lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20) )` 得到一个对象，再把名称/属性/下标 `d_gaussian, gaussian_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_gaussian, gaussian_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_saw, saw_ms = time_gpu( lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65)) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65)) )` 得到一个对象，再把名称/属性/下标 `d_saw, saw_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_saw, saw_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_square, square_ms = time_gpu( lambda: cusignal.square(d_square_phase, duty=np.float32(0.35)) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.square(d_square_phase, duty=np.float32(0.35)) )` 得到一个对象，再把名称/属性/下标 `d_square, square_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_square, square_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `outputs, mix_ms = time_gpu(lambda: _mix_echo(d_chirp, d_gaussian, d_saw, d_square, config))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: _mix_echo(d_chirp, d_gaussian, d_saw, d_square, config))` 得到一个对象，再把名称/属性/下标 `outputs, mix_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `outputs, mix_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_target, d_interference, d_noise, d_echo = outputs`：这是 Python 赋值语句。解释器先完整计算右侧 `outputs` 得到一个对象，再把名称/属性/下标 `d_target, d_interference, d_noise, d_echo` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_target, d_interference, d_noise, d_echo` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.operator_ms = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.operator_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.operator_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `gausspulse`、`sawtooth`、`square`。 本块直接出现 4 种波形符号：`chirp`、`gausspulse`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.57 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `"cusignal.chirp": chirp_ms,`。

```python
        "cusignal.chirp": chirp_ms,
        "cusignal.gausspulse": gaussian_ms,
        "cusignal.sawtooth": saw_ms,
        "cusignal.square": square_ms,
        "cupy.echo_superposition": mix_ms,
    }
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`chirp`、`chirp_ms`、`gausspulse`、`gaussian_ms`、`sawtooth`、`saw_ms`、`square`、`square_ms`、`cupy`、`echo_superposition`、`mix_ms`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"cusignal.chirp": chirp_ms, "cusignal.gausspulse": gaussian_ms, "cusignal.sawtooth": saw_ms, "cusignal.square": square_ms, "cupy.echo_superposition": mix_ms, }`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`gausspulse`、`sawtooth`、`square`、`echo`。 本块直接出现 4 种波形符号：`chirp`、`gausspulse`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.58 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `evidence.compute_ms = sum(evidence.operator_ms.values())`。

```python
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.target_waveform = as_host(d_chirp).astype(np.float32, copy=False)
    state.target_component = as_host(d_target)
    state.interference_component = as_host(d_interference)
    state.noise_component = as_host(d_noise)
    state.echo = as_host(d_echo)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(np.all(np.isfinite(state.echo)), "step1 produced non-finite echo")
    require(np.max(np.abs(state.target_component)) > 0.1, "step1 target component degenerate")
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d2h_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d2h_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`post_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d2h_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `state.target_waveform = as_host(d_chirp).astype(np.float32, copy=False)` 中，它为 `state.target_waveform` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `evidence.d2h_ms = wall_ms(d2h_begin)` 中，它为 `evidence.d2h_ms` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(np.max(np.abs(state.target_component)) > 0.1, "step1 target component degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `evidence.compute_ms = sum(evidence.operator_ms.values())`：这是 Python 赋值语句。解释器先完整计算右侧 `sum(evidence.operator_ms.values())` 得到一个对象，再把名称/属性/下标 `evidence.compute_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.compute_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `d2h_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.target_waveform = as_host(d_chirp).astype(np.float32, copy=False)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_chirp).astype(np.float32, copy=False)` 得到一个对象，再把名称/属性/下标 `state.target_waveform` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.target_waveform` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.target_component = as_host(d_target)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_target)` 得到一个对象，再把名称/属性/下标 `state.target_component` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.target_component` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.interference_component = as_host(d_interference)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_interference)` 得到一个对象，再把名称/属性/下标 `state.interference_component` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.interference_component` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.noise_component = as_host(d_noise)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_noise)` 得到一个对象，再把名称/属性/下标 `state.noise_component` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.noise_component` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.echo = as_host(d_echo)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_echo)` 得到一个对象，再把名称/属性/下标 `state.echo` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.echo` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.d2h_ms = wall_ms(d2h_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(d2h_begin)` 得到一个对象，再把名称/属性/下标 `evidence.d2h_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.d2h_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.formal_execution_ms = wall_ms(formal_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(formal_begin)` 得到一个对象，再把名称/属性/下标 `evidence.formal_execution_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.formal_execution_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `post_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(np.all(np.isfinite(state.echo)), "step1 produced non-finite echo")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.all(np.isfinite(state.echo))` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1 produced non-finite echo"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.max(np.abs(state.target_component)) > 0.1, "step1 target component degenerate")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.max(np.abs(state.target_component)) > 0.1` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1 target component degenerate"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`、`target_waveform`、`target_component`、`interference_component`、`noise_component`。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.59 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")`。

```python
    require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")
    require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")
    perturbed_weight = _mix_echo(
        d_chirp, d_gaussian, d_saw, d_square, config,
        target_weight=max(0.0, config.target_weight - 0.07),
    )[3]
    perturbed_seed = _mix_echo(
        d_chirp, d_gaussian, d_saw, d_square, config, noise_seed=config.noise_seed + 1
    )[3]
    weight_delta = float(np.max(np.abs(as_host(perturbed_weight) - state.echo)))
    seed_delta = float(np.max(np.abs(as_host(perturbed_seed) - state.echo)))
    require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `perturbed_weight` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `target_weight` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `perturbed_seed` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `weight_delta` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `seed_delta` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.01` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.001` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.07` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0e-3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`perturbed_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`target_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`perturbed_seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`weight_delta`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`seed_delta`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`1.0e-3`|当前函数或调用的参数名称，接收调用者绑定的输入 `1.0e-3`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.01`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`0.001`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `target_weight=max(0.0, config.target_weight - 0.07),` 中，它为 `target_weight` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.07`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `target_weight=max(0.0, config.target_weight - 0.07),` 中，它为 `target_weight` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `)[3]`，作用由同一表达式中的运算符决定；在 `require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行；它直接参与表达式 `d_chirp, d_gaussian, d_saw, d_square, config, noise_seed=config.noise_seed + 1`，作用由同一表达式中的运算符决定。|
|`1.0e-3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.max(np.abs(state.interference_component)) > 0.01` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1 interference degenerate"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.max(np.abs(state.noise_component)) > 0.001` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1 noise degenerate"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `perturbed_weight = _mix_echo( d_chirp, d_gaussian, d_saw, d_square, config, target_weight=max(0.0, config.target_weight - 0.07), )[3]`：这是 Python 赋值语句。解释器先完整计算右侧 `_mix_echo( d_chirp, d_gaussian, d_saw, d_square, config, target_weight=max(0.0, config.target_weight - 0.07), )[3]` 得到一个对象，再把名称/属性/下标 `perturbed_weight` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `perturbed_weight` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `perturbed_seed = _mix_echo( d_chirp, d_gaussian, d_saw, d_square, config, noise_seed=config.noise_seed + 1 )[3]`：这是 Python 赋值语句。解释器先完整计算右侧 `_mix_echo( d_chirp, d_gaussian, d_saw, d_square, config, noise_seed=config.noise_seed + 1 )[3]` 得到一个对象，再把名称/属性/下标 `perturbed_seed` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `perturbed_seed` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `weight_delta = float(np.max(np.abs(as_host(perturbed_weight) - state.echo)))`：这是 Python 赋值语句。解释器先完整计算右侧 `float(np.max(np.abs(as_host(perturbed_weight) - state.echo)))` 得到一个对象，再把名称/属性/下标 `weight_delta` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `weight_delta` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `seed_delta = float(np.max(np.abs(as_host(perturbed_seed) - state.echo)))`：这是 Python 赋值语句。解释器先完整计算右侧 `float(np.max(np.abs(as_host(perturbed_seed) - state.echo)))` 得到一个对象，再把名称/属性/下标 `seed_delta` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `seed_delta` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`weight_delta > 1.0e-3` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1 target-weight perturbation did not change output"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`、`interference_component`、`noise_component`、`target_weight`、`noise_seed`。 本块直接出现 2 种波形符号：`chirp`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.60 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")`。

```python
    require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")
    evidence.metrics = {
        "waveform_functions_selected": 4.0,
        "target_component": 1.0,
        "interference_component": 1.0,
        "noise_component": 1.0,
        "custom_waveform_superposition": 1.0,
        "sample_rate_hz": config.sample_rate_hz,
        "samples": float(config.samples),
        "target_delay_samples": float(config.target_delay_samples),
        "noise_seed": float(config.noise_seed),
        "echo_rms": float(np.sqrt(np.mean(state.echo.astype(np.float64) ** 2))),
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1.0e-3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `4.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`1.0e-3`|当前函数或调用的参数名称，接收调用者绑定的输入 `1.0e-3`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")`；值在 `ZKX/Task2/task2_python/step1/step1.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0e-3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`4.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"waveform_functions_selected": 4.0,`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")` 中，它是分支或边界比较值，决定哪条控制路径被执行；它直接参与表达式 `"target_component": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"interference_component": 1.0,`，作用由同一表达式中的运算符决定。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"waveform_functions_selected": 4.0,`，作用由同一表达式中的运算符决定；在 `"echo_rms": float(np.sqrt(np.mean(state.echo.astype(np.float64) ** 2))),` 中，它作为 `float` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `"echo_rms": float(np.sqrt(np.mean(state.echo.astype(np.float64) ** 2))),` 中，它作为 `float` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`seed_delta > 1.0e-3` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1 seed perturbation did not change output"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.metrics = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.metrics` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.metrics` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"waveform_functions_selected": 4.0, "target_component": 1.0, "interference_component": 1.0, "noise_component": 1.0, "custom_waveform_superposition": 1.0, "sample_rate_hz": confi...`：函数调用语法：`float(...)` 先准备括号内实参，再把控制权交给 `float`，返回值回到调用点；`np.sqrt(...)` 先准备括号内实参，再把控制权交给 `np.sqrt`，返回值回到调用点；`np.mean(...)` 先准备括号内实参，再把控制权交给 `np.mean`，返回值回到调用点；`state.echo.astype(...)` 先准备括号内实参，再把控制权交给 `state.echo.astype`，返回值回到调用点。 字面量与类型：`4.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`、`sample_rate_hz`、`samples`、`target_delay_samples`、`noise_seed`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.61 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `"weight_perturbation_delta": weight_delta,`。

```python
        "weight_perturbation_delta": weight_delta,
        "seed_perturbation_delta": seed_delta,
        "perturbation_checks_pass": 1.0,
    }
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `"perturbation_checks_pass": 1.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"weight_perturbation_delta": weight_delta, "seed_perturbation_delta": seed_delta, "perturbation_checks_pass": 1.0, }`：字面量与类型：`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step1` 所在的 `ZKX/Task2/task2_python/step1/step1.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"weight_perturbation_delta": weight_delta,` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.62 run_step1：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/step1.py`；符号 `run_step1`；源码锚点 `evidence.output_shape = list(state.echo.shape)`。

```python
    evidence.output_shape = list(state.echo.shape)
    evidence.output_dtype = str(state.echo.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`output_shape`、`list`、`state`、`echo`、`shape`、`output_dtype`、`str`、`dtype`、`post_ms`、`wall_ms`、`post_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.output_shape = list(state.echo.shape)`：这是 Python 赋值语句。解释器先完整计算右侧 `list(state.echo.shape)` 得到一个对象，再把名称/属性/下标 `evidence.output_shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.output_dtype = str(state.echo.dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `str(state.echo.dtype)` 得到一个对象，再把名称/属性/下标 `evidence.output_dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms = wall_ms(post_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(post_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.total_ms = wall_ms(total_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(total_begin)` 得到一个对象，再把名称/属性/下标 `evidence.total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return evidence`：`return` 先计算 `evidence`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.63 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step1`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import time
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`time`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step1` 所在的 `ZKX/Task2/task2_python/step2/step2.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.64 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `import`；源码锚点 `import numpy as np`。

```python
import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`task2_common`、`PipelineState`、`StepEvidence`、`as_host`、`cp`、`cusignal`、`require`、`synchronize`、`time_gpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, as_host, cp, cusignal, require, synchronize, time_gpu, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.65 run_step2：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step2`；源码锚点 `def run_step2(state: PipelineState) -> StepEvidence:`。

```python

def run_step2(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step2")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.echo is not None and state.echo.size == config.samples, "step2 did not receive step1 echo")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_echo = cp.asarray(state.echo, dtype=cp.float32)
    synchronize()
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `evidence` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `prep_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `h2d_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_echo` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_step2` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`and`|当前函数或调用的参数名称，接收调用者绑定的输入 `and`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(state.echo is not None and state.echo.size == config.samples, "step2 did not receive step1 echo")`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|
|`prep_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`h2d_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`d_echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`run_step2`|自定义函数/可调用入口 `run_step2`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step2/step2.py` 中承担 `run_step2` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `def run_step2(state: PipelineState) -> StepEvidence:` 中，它作为 `run_step2` 的实参参与当前调用；在 `evidence = StepEvidence("step2")` 中，它为 `evidence` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `require(state.echo is not None and state.echo.size == config.samples, "step2 did not receive step1 echo")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def run_step2(state: PipelineState) -> StepEvidence:`：`def` 创建名为 `run_step2` 的函数对象；括号中的 `state: PipelineState` 是形参表，调用时实参按位置或关键字绑定。`-> StepEvidence` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `config = state.config`：这是 Python 赋值语句。解释器先完整计算右侧 `state.config` 得到一个对象，再把名称/属性/下标 `config` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `config` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence = StepEvidence("step2")`：这是 Python 赋值语句。解释器先完整计算右侧 `StepEvidence("step2")` 得到一个对象，再把名称/属性/下标 `evidence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `total_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `formal_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `prep_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `prep_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.echo is not None and state.echo.size == config.samples, "step2 did not receive step1 echo")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.echo is not None and state.echo.size == config.samples` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step2 did not receive step1 echo"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.prep_ms = wall_ms(prep_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(prep_begin)` 得到一个对象，再把名称/属性/下标 `evidence.prep_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.prep_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `h2d_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `h2d_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_echo = cp.asarray(state.echo, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(state.echo, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_echo` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_echo` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `synchronize()`：`synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `echo`、`config`、`samples`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.66 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step2`；源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`。

```python
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_window, window_ms = time_gpu(lambda: cusignal.hamming(config.filter_taps, sym=True))
    d_taps, firwin_ms = time_gpu(
        lambda: cusignal.firwin(
            config.filter_taps, config.filter_cutoff_hz, window="hamming", pass_zero=True,
            scale=True, fs=config.sample_rate_hz, gpupath=True,
        ).astype(cp.float32)
    )
    d_filtered, filter_ms = time_gpu(lambda: cusignal.firfilter(d_taps, d_echo, axis=-1))
    evidence.operator_ms = {
        "cusignal.hamming": window_ms,
        "cusignal.firwin": firwin_ms,
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `scale` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `d_window, window_ms = time_gpu(lambda: cusignal.hamming(config.filter_taps, sym=True))`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`scale`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `scale`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `scale=True, fs=config.sample_rate_hz, gpupath=True,`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `evidence.h2d_ms = wall_ms(h2d_begin)` 中，它为 `evidence.h2d_ms` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `).astype(cp.float32)` 中，它作为 `astype` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `d_filtered, filter_ms = time_gpu(lambda: cusignal.firfilter(d_taps, d_echo, axis=-1))` 中，它作为 `time_gpu` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(h2d_begin)` 得到一个对象，再把名称/属性/下标 `evidence.h2d_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.h2d_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_window, window_ms = time_gpu(lambda: cusignal.hamming(config.filter_taps, sym=True))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: cusignal.hamming(config.filter_taps, sym=True))` 得到一个对象，再把名称/属性/下标 `d_window, window_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_window, window_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_taps, firwin_ms = time_gpu( lambda: cusignal.firwin( config.filter_taps, config.filter_cutoff_hz, window="hamming", pass_zero=True, scale=True, fs=config.sample_rate_hz, gpupa...`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.firwin( config.filter_taps, config.filter_cutoff_hz, window="hamming", pass_zero=True, scale=True, fs=config.sample_rate_hz, gpupath=True, ).astype(cp.float32) )` 得到一个对象，再把名称/属性/下标 `d_taps, firwin_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_taps, firwin_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_filtered, filter_ms = time_gpu(lambda: cusignal.firfilter(d_taps, d_echo, axis=-1))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: cusignal.firfilter(d_taps, d_echo, axis=-1))` 得到一个对象，再把名称/属性/下标 `d_filtered, filter_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_filtered, filter_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.operator_ms = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.operator_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.operator_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"cusignal.hamming": window_ms, "cusignal.firwin": firwin_ms,`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `firwin`、`firfilter`、`hamming`、`filter_taps`、`filter_cutoff_hz`、`sample_rate_hz`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.67 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step2`；源码锚点 `"cusignal.firfilter": filter_ms,`。

```python
        "cusignal.firfilter": filter_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.filter_taps = as_host(d_taps).astype(np.float32)
    state.filtered = as_host(d_filtered).astype(np.float32)
    window_host = as_host(d_window)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(state.filtered.size == config.samples, "step2 output length mismatch")
    require(np.all(np.isfinite(state.filtered)), "step2 produced non-finite output")
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d2h_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `window_host` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d2h_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`window_host`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `window_host`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `window_host = as_host(d_window)`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`post_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d2h_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `state.filter_taps = as_host(d_taps).astype(np.float32)` 中，它为 `state.filter_taps` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `state.filtered = as_host(d_filtered).astype(np.float32)` 中，它为 `state.filtered` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `"cusignal.firfilter": filter_ms, }`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `evidence.compute_ms = sum(evidence.operator_ms.values())`：这是 Python 赋值语句。解释器先完整计算右侧 `sum(evidence.operator_ms.values())` 得到一个对象，再把名称/属性/下标 `evidence.compute_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.compute_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `d2h_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.filter_taps = as_host(d_taps).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_taps).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.filter_taps` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.filter_taps` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.filtered = as_host(d_filtered).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_filtered).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.filtered` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.filtered` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `window_host = as_host(d_window)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_window)` 得到一个对象，再把名称/属性/下标 `window_host` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `window_host` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.d2h_ms = wall_ms(d2h_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(d2h_begin)` 得到一个对象，再把名称/属性/下标 `evidence.d2h_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.d2h_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.formal_execution_ms = wall_ms(formal_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(formal_begin)` 得到一个对象，再把名称/属性/下标 `evidence.formal_execution_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.formal_execution_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `post_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.filtered.size == config.samples, "step2 output length mismatch")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.filtered.size == config.samples` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step2 output length mismatch"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.all(np.isfinite(state.filtered)), "step2 produced non-finite output")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.all(np.isfinite(state.filtered))` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step2 produced non-finite output"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `firfilter`、`filter_taps`、`filtered`、`samples`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.68 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step2`；源码锚点 `target_filtered = np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]`。

```python
    target_filtered = np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]
    residual = state.echo - state.target_component
    residual_filtered = np.convolve(residual, state.filter_taps, mode="full")[:config.samples]
    before_snr = 20.0 * np.log10(
        max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)
        / max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)
    )
    after_snr = 20.0 * np.log10(
        max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12)
        / max(np.sqrt(np.mean(residual_filtered.astype(np.float64) ** 2)), 1.0e-12)
    )
    require(after_snr - before_snr >= 1.0, "step2 SNR improvement is below 1 dB")
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `target_filtered` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `residual` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `residual_filtered` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `before_snr` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `after_snr` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `20.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0e-12` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0e-12` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `20.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0e-12` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`improvement`|当前函数或调用的参数名称，接收调用者绑定的输入 `improvement`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(after_snr - before_snr >= 1.0, "step2 SNR improvement is below 1 dB")`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`target_filtered`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `target_filtered`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `target_filtered = np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`residual`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `residual`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `residual = state.echo - state.target_component`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`residual_filtered`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `residual_filtered`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `residual_filtered = np.convolve(residual, state.filter_taps, mode="full")[:config.samples]`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`before_snr`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `before_snr`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `before_snr = 20.0 * np.log10(`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`after_snr`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `after_snr`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `after_snr = 20.0 * np.log10(`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`1.0e-12`|当前表达式读取或传递的工程名称 `1.0e-12`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)`；值在 `ZKX/Task2/task2_python/step2/step2.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`20.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `before_snr = 20.0 * np.log10(` 中，它为 `before_snr` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `after_snr = 20.0 * np.log10(` 中，它为 `after_snr` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `before_snr = 20.0 * np.log10(` 中，它为 `before_snr` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `/ max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `/ max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `before_snr = 20.0 * np.log10(` 中，它为 `before_snr` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `/ max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用。|
|`1.0e-12`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `/ max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `/ max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `before_snr = 20.0 * np.log10(` 中，它为 `before_snr` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用；在 `/ max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)` 中，它作为 `max` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `target_filtered = np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]`：这是 Python 赋值语句。解释器先完整计算右侧 `np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]` 得到一个对象，再把名称/属性/下标 `target_filtered` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `target_filtered` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `residual = state.echo - state.target_component`：这是 Python 赋值语句。解释器先完整计算右侧 `state.echo - state.target_component` 得到一个对象，再把名称/属性/下标 `residual` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `residual` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `residual_filtered = np.convolve(residual, state.filter_taps, mode="full")[:config.samples]`：这是 Python 赋值语句。解释器先完整计算右侧 `np.convolve(residual, state.filter_taps, mode="full")[:config.samples]` 得到一个对象，再把名称/属性/下标 `residual_filtered` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `residual_filtered` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `before_snr = 20.0 * np.log10( max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12) / max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12) )`：这是 Python 赋值语句。解释器先完整计算右侧 `20.0 * np.log10( max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12) / max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12) )` 得到一个对象，再把名称/属性/下标 `before_snr` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `before_snr` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `after_snr = 20.0 * np.log10( max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12) / max(np.sqrt(np.mean(residual_filtered.astype(np.float64) ** 2)), 1.0e-12) )`：这是 Python 赋值语句。解释器先完整计算右侧 `20.0 * np.log10( max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12) / max(np.sqrt(np.mean(residual_filtered.astype(np.float64) ** 2)), 1.0e-12) )` 得到一个对象，再把名称/属性/下标 `after_snr` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `after_snr` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(after_snr - before_snr >= 1.0, "step2 SNR improvement is below 1 dB")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`after_snr - before_snr >= 1.0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step2 SNR improvement is below 1 dB"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `echo`、`target_component`、`filter_taps`、`samples`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.69 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step2`；源码锚点 `evidence.metrics = {`。

```python
    evidence.metrics = {
        "filter_design_functions": 1.0,
        "filtering_functions": 1.0,
        "window_functions": 1.0,
        "step1_to_step2_data_match": 1.0,
        "taps": float(config.filter_taps),
        "cutoff_hz": config.filter_cutoff_hz,
        "window_nonzero": float(np.max(np.abs(window_host)) > 0.0),
        "snr_before_db": float(before_snr),
        "snr_after_db": float(after_snr),
        "snr_improvement_db": float(after_snr - before_snr),
    }
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `"filter_design_functions": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"filtering_functions": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"window_functions": 1.0,`，作用由同一表达式中的运算符决定。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `"window_nonzero": float(np.max(np.abs(window_host)) > 0.0),` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `evidence.metrics = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.metrics` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.metrics` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"filter_design_functions": 1.0, "filtering_functions": 1.0, "window_functions": 1.0, "step1_to_step2_data_match": 1.0, "taps": float(config.filter_taps), "cutoff_hz": config.fil...`：函数调用语法：`float(...)` 先准备括号内实参，再把控制权交给 `float`，返回值回到调用点；`np.max(...)` 先准备括号内实参，再把控制权交给 `np.max`，返回值回到调用点；`np.abs(...)` 先准备括号内实参，再把控制权交给 `np.abs`，返回值回到调用点。 字面量与类型：`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`-`：减法；位于单个操作数前时是一元负号；`>`：大于比较；模板实参列表中也可作为右尖括号；`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `filter_taps`、`filter_cutoff_hz`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.70 run_step2：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/step2.py`；符号 `run_step2`；源码锚点 `evidence.output_shape = list(state.filtered.shape)`。

```python
    evidence.output_shape = list(state.filtered.shape)
    evidence.output_dtype = str(state.filtered.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`output_shape`、`list`、`state`、`filtered`、`shape`、`output_dtype`、`str`、`dtype`、`post_ms`、`wall_ms`、`post_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.output_shape = list(state.filtered.shape)`：这是 Python 赋值语句。解释器先完整计算右侧 `list(state.filtered.shape)` 得到一个对象，再把名称/属性/下标 `evidence.output_shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.output_dtype = str(state.filtered.dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `str(state.filtered.dtype)` 得到一个对象，再把名称/属性/下标 `evidence.output_dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms = wall_ms(post_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(post_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.total_ms = wall_ms(total_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(total_begin)` 得到一个对象，再把名称/属性/下标 `evidence.total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return evidence`：`return` 先计算 `evidence`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `filtered`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.71 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step2`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import time
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`time`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step2` 所在的 `ZKX/Task2/task2_python/step3/step3.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.72 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `import`；源码锚点 `import numpy as np`。

```python
import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    roughness,
    synchronize,
    time_gpu,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`task2_common`、`PipelineState`、`StepEvidence`、`as_host`、`cp`、`cusignal`、`require`、`roughness`、`synchronize`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, as_host, cp, cusignal, require, roughness, synchronize, time_gpu, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.73 run_step3：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step3`；源码锚点 `def run_step3(state: PipelineState) -> StepEvidence:`。

```python

def run_step3(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step3")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.filtered is not None and state.filtered.size == config.samples, "step3 did not receive step2 output")
    crop = config.step3_crop_each
    cropped = state.filtered[crop:-crop].astype(np.float32, copy=True)
    offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)
    evidence.prep_ms = wall_ms(prep_begin)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `evidence` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `prep_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `crop` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `cropped` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `offsets` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_step3` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.01` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`and`|当前函数或调用的参数名称，接收调用者绑定的输入 `and`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(state.filtered is not None and state.filtered.size == config.samples, "step3 did not receive step2 output")`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|
|`prep_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`crop`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `crop`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `crop = config.step3_crop_each`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`cropped`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `cropped`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cropped = state.filtered[crop:-crop].astype(np.float32, copy=True)`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`offsets`|相对起点的离散偏移|记作 $\Delta n$|参与索引换算或数据切片|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`run_step3`|自定义函数/可调用入口 `run_step3`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step3/step3.py` 中承担 `run_step3` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `require(state.filtered is not None and state.filtered.size == config.samples, "step3 did not receive step2 output")` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `cropped = state.filtered[crop:-crop].astype(np.float32, copy=True)` 中，它为 `cropped` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)` 中，它为 `offsets` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)` 中，它为 `offsets` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2.01`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)` 中，它为 `offsets` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。 当前上下文：在 `offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)` 中，它为 `offsets` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def run_step3(state: PipelineState) -> StepEvidence:`：`def` 创建名为 `run_step3` 的函数对象；括号中的 `state: PipelineState` 是形参表，调用时实参按位置或关键字绑定。`-> StepEvidence` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `config = state.config`：这是 Python 赋值语句。解释器先完整计算右侧 `state.config` 得到一个对象，再把名称/属性/下标 `config` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `config` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence = StepEvidence("step3")`：这是 Python 赋值语句。解释器先完整计算右侧 `StepEvidence("step3")` 得到一个对象，再把名称/属性/下标 `evidence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `total_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `formal_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `prep_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `prep_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.filtered is not None and state.filtered.size == config.samples, "step3 did not receive step2 output")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.filtered is not None and state.filtered.size == config.samples` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step3 did not receive step2 output"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `crop = config.step3_crop_each`：这是 Python 赋值语句。解释器先完整计算右侧 `config.step3_crop_each` 得到一个对象，再把名称/属性/下标 `crop` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `crop` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `cropped = state.filtered[crop:-crop].astype(np.float32, copy=True)`：这是 Python 赋值语句。解释器先完整计算右侧 `state.filtered[crop:-crop].astype(np.float32, copy=True)` 得到一个对象，再把名称/属性/下标 `cropped` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `cropped` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.arange(-2.0, 2.01, 0.5, dtype=np.float32)` 得到一个对象，再把名称/属性/下标 `offsets` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `offsets` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.prep_ms = wall_ms(prep_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(prep_begin)` 得到一个对象，再把名称/属性/下标 `evidence.prep_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.prep_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 `config`、`filtered`、`samples`、`step3_crop_each`。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.74 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step3`；源码锚点 `h2d_begin = time.perf_counter()`。

```python
    h2d_begin = time.perf_counter()
    d_cropped = cp.asarray(cropped)
    d_offsets = cp.asarray(offsets)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_cubic, cubic_ms = time_gpu(lambda: cusignal.cubic(d_offsets))
    d_weights, normalization_ms = time_gpu(lambda: d_cubic / cp.sum(d_cubic))
    d_smoothed, filter_ms = time_gpu(lambda: cusignal.firfilter(d_weights, d_cropped, axis=-1))
    evidence.operator_ms = {
        "cusignal.cubic": cubic_ms,
        "cupy.cubic_normalization": normalization_ms,
        "cusignal.firfilter_b_spline": filter_ms,
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `h2d_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_cropped` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_offsets` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`h2d_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`d_cropped`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_python/step3/step3.py` 的 GPU 调用链|
|`d_offsets`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_python/step3/step3.py` 的 GPU 调用链|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `d_smoothed, filter_ms = time_gpu(lambda: cusignal.firfilter(d_weights, d_cropped, axis=-1))` 中，它作为 `time_gpu` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `h2d_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `h2d_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_cropped = cp.asarray(cropped)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(cropped)` 得到一个对象，再把名称/属性/下标 `d_cropped` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_cropped` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_offsets = cp.asarray(offsets)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(offsets)` 得到一个对象，再把名称/属性/下标 `d_offsets` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_offsets` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `synchronize()`：`synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(h2d_begin)` 得到一个对象，再把名称/属性/下标 `evidence.h2d_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.h2d_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_cubic, cubic_ms = time_gpu(lambda: cusignal.cubic(d_offsets))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: cusignal.cubic(d_offsets))` 得到一个对象，再把名称/属性/下标 `d_cubic, cubic_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_cubic, cubic_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_weights, normalization_ms = time_gpu(lambda: d_cubic / cp.sum(d_cubic))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: d_cubic / cp.sum(d_cubic))` 得到一个对象，再把名称/属性/下标 `d_weights, normalization_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_weights, normalization_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_smoothed, filter_ms = time_gpu(lambda: cusignal.firfilter(d_weights, d_cropped, axis=-1))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: cusignal.firfilter(d_weights, d_cropped, axis=-1))` 得到一个对象，再把名称/属性/下标 `d_smoothed, filter_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_smoothed, filter_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.operator_ms = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.operator_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.operator_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"cusignal.cubic": cubic_ms, "cupy.cubic_normalization": normalization_ms, "cusignal.firfilter_b_spline": filter_ms,`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 `firfilter`、`cubic`。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.75 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step3`；源码锚点 `}`。

```python
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.smoothed = as_host(d_smoothed).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    handoff_begin = time.perf_counter()
    state.reference_for_correlation = state.target_waveform[
        crop:-crop
    ].astype(np.float32, copy=True)
    evidence.post_ms = wall_ms(handoff_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d2h_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `handoff_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `crop` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d2h_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`handoff_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`crop`|当前表达式读取或传递的工程名称 `crop`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `crop:-crop`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`post_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d2h_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `state.smoothed = as_host(d_smoothed).astype(np.float32)` 中，它为 `state.smoothed` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `evidence.d2h_ms = wall_ms(d2h_begin)` 中，它为 `evidence.d2h_ms` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `}`：该边界结束当前结构。
- 对源码锚点 `evidence.compute_ms = sum(evidence.operator_ms.values())`：这是 Python 赋值语句。解释器先完整计算右侧 `sum(evidence.operator_ms.values())` 得到一个对象，再把名称/属性/下标 `evidence.compute_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.compute_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `d2h_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.smoothed = as_host(d_smoothed).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_smoothed).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.smoothed` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.smoothed` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.d2h_ms = wall_ms(d2h_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(d2h_begin)` 得到一个对象，再把名称/属性/下标 `evidence.d2h_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.d2h_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `handoff_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `handoff_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `handoff_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.reference_for_correlation = state.target_waveform[ crop:-crop ].astype(np.float32, copy=True)`：这是 Python 赋值语句。解释器先完整计算右侧 `state.target_waveform[ crop:-crop ].astype(np.float32, copy=True)` 得到一个对象，再把名称/属性/下标 `state.reference_for_correlation` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.reference_for_correlation` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms = wall_ms(handoff_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(handoff_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.formal_execution_ms = wall_ms(formal_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(formal_begin)` 得到一个对象，再把名称/属性/下标 `evidence.formal_execution_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.formal_execution_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `post_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 `smoothed`、`reference_for_correlation`、`target_waveform`。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.76 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step3`；源码锚点 `rough_before = roughness(cropped)`。

```python
    rough_before = roughness(cropped)
    rough_after = roughness(state.smoothed)
    peak_ratio = float(
        np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12)
    )
    require(rough_after < rough_before, "step3 did not reduce roughness")
    require(peak_ratio > 0.20, "step3 over-smoothed the target structure")
    evidence.metrics = {
        "bsplines_functions": 1.0,
        "step2_to_step3_data_match": 1.0,
        "crop_begin": float(crop),
        "crop_end": float(config.samples - crop),
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `rough_before` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `rough_after` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `peak_ratio` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1.0e-12` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.20` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`rough_before`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `rough_before`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `rough_before = roughness(cropped)`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`rough_after`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `rough_after`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `rough_after = roughness(state.smoothed)`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`peak_ratio`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `peak_ratio`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `peak_ratio = float(`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`1.0e-12`|当前表达式读取或传递的工程名称 `1.0e-12`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12)`；值在 `ZKX/Task2/task2_python/step3/step3.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0e-12`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12)` 中，它作为 `np.max` 的实参参与当前调用。|
|`0.20`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(peak_ratio > 0.20, "step3 over-smoothed the target structure")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12)` 中，它作为 `np.max` 的实参参与当前调用；它直接参与表达式 `"bsplines_functions": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"step2_to_step3_data_match": 1.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `rough_before = roughness(cropped)`：这是 Python 赋值语句。解释器先完整计算右侧 `roughness(cropped)` 得到一个对象，再把名称/属性/下标 `rough_before` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `rough_before` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `rough_after = roughness(state.smoothed)`：这是 Python 赋值语句。解释器先完整计算右侧 `roughness(state.smoothed)` 得到一个对象，再把名称/属性/下标 `rough_after` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `rough_after` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `peak_ratio = float( np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12) )`：这是 Python 赋值语句。解释器先完整计算右侧 `float( np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12) )` 得到一个对象，再把名称/属性/下标 `peak_ratio` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `peak_ratio` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(rough_after < rough_before, "step3 did not reduce roughness")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`rough_after < rough_before` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step3 did not reduce roughness"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(peak_ratio > 0.20, "step3 over-smoothed the target structure")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`peak_ratio > 0.20` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step3 over-smoothed the target structure"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.metrics = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.metrics` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.metrics` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"bsplines_functions": 1.0, "step2_to_step3_data_match": 1.0, "crop_begin": float(crop), "crop_end": float(config.samples - crop),`：函数调用语法：`float(...)` 先准备括号内实参，再把控制权交给 `float`，返回值回到调用点。 字面量与类型：`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`-`：减法；位于单个操作数前时是一元负号；`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 `smoothed`、`samples`。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.77 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step3`；源码锚点 `"roughness_before": rough_before,`。

```python
        "roughness_before": rough_before,
        "roughness_after": rough_after,
        "peak_preservation_ratio": peak_ratio,
    }
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `roughness_before`、`rough_before`、`roughness_after`、`rough_after`、`peak_preservation_ratio`、`peak_ratio`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"roughness_before": rough_before, "roughness_after": rough_after, "peak_preservation_ratio": peak_ratio, }`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"roughness_before": rough_before,` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.78 run_step3：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/step3.py`；符号 `run_step3`；源码锚点 `evidence.output_shape = list(state.smoothed.shape)`。

```python
    evidence.output_shape = list(state.smoothed.shape)
    evidence.output_dtype = str(state.smoothed.dtype)
    evidence.post_ms += wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`output_shape`、`list`、`state`、`smoothed`、`shape`、`output_dtype`、`str`、`dtype`、`post_ms`、`wall_ms`、`post_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.output_shape = list(state.smoothed.shape)`：这是 Python 赋值语句。解释器先完整计算右侧 `list(state.smoothed.shape)` 得到一个对象，再把名称/属性/下标 `evidence.output_shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.output_dtype = str(state.smoothed.dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `str(state.smoothed.dtype)` 得到一个对象，再把名称/属性/下标 `evidence.output_dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms += wall_ms(post_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(post_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms +` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms +` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.total_ms = wall_ms(total_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(total_begin)` 得到一个对象，再把名称/属性/下标 `evidence.total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return evidence`：`return` 先计算 `evidence`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 `smoothed`。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_python/step3/step3.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.79 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `run_step3`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import time
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`time`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_python/step4/step4.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.80 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `import`；源码锚点 `import numpy as np`。

```python
import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`task2_common`、`PipelineState`、`StepEvidence`、`as_host`、`cp`、`cusignal`、`require`、`synchronize`、`time_gpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, as_host, cp, cusignal, require, synchronize, time_gpu, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.81 _analytic_signal：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_analytic_signal`；源码锚点 `def _analytic_signal(signal):`。

```python

def _analytic_signal(signal):
    previous = cp.concatenate((signal[:1], signal[:-1]))
    following = cp.concatenate((signal[1:], signal[-1:]))
    return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `previous` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `following` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_analytic_signal` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`previous`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `previous`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `previous = cp.concatenate((signal[:1], signal[:-1]))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`following`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `following`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `following = cp.concatenate((signal[1:], signal[-1:]))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`signal`|当前函数或调用的参数名称，接收调用者绑定的输入 `signal`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def _analytic_signal(signal):`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`_analytic_signal`|自定义函数/可调用入口 `_analytic_signal`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step4/step4.py` 中承担 `_analytic_signal` 所表达的步骤职责|
|`0.`|当前函数或调用的参数名称，接收调用者绑定的输入 `0.`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `previous = cp.concatenate((signal[:1], signal[:-1]))` 中，它为 `previous` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `following = cp.concatenate((signal[1:], signal[-1:]))` 中，它为 `following` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)` 中，它作为 `return` 的实参参与当前调用。|
|`0.`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)` 中，它作为 `return` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def _analytic_signal(signal):`：`def` 创建名为 `_analytic_signal` 的函数对象；括号中的 `signal` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `previous = cp.concatenate((signal[:1], signal[:-1]))`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.concatenate((signal[:1], signal[:-1]))` 得到一个对象，再把名称/属性/下标 `previous` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `previous` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `following = cp.concatenate((signal[1:], signal[-1:]))`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.concatenate((signal[1:], signal[-1:]))` 得到一个对象，再把名称/属性/下标 `following` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `following` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)`：`return` 先计算 `(signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_analytic_signal` 所在的 `ZKX/Task2/task2_python/step4/step4.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.82 _resample_linear：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_resample_linear`；源码锚点 `def _resample_linear(values, count: int):`。

```python

def _resample_linear(values, count: int):
    if values.size == count:
        return values.astype(cp.float32)
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_resample_linear` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`_resample_linear`|当前脉冲内快时间采样编号|记作 $n=i\bmod N_s$|由展平索引取余得到|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `return values.astype(cp.float32)` 中，它作为 `values.astype` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def _resample_linear(values, count: int):`：`def` 创建名为 `_resample_linear` 的函数对象；括号中的 `values, count: int` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if values.size == count:`：`if` 先计算条件 `values.size == count` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return values.astype(cp.float32)`：`return` 先计算 `values.astype(cp.float32)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_resample_linear` 所在的 `ZKX/Task2/task2_python/step4/step4.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.83 _resample_linear：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_resample_linear`；源码锚点 `old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)`。

```python
    old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)
    new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)
    return cp.interp(new, old, values.astype(cp.float32))
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `old` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `new` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`old`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `old`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`new`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `new`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)` 中，它为 `old` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)` 中，它为 `new` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)` 中，它为 `old` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)` 中，它为 `new` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)` 中，它为 `old` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)` 中，它为 `new` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `return cp.interp(new, old, values.astype(cp.float32))` 中，它作为 `cp.interp` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `old` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `old` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.linspace(0.0, 1.0, count, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `new` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `new` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return cp.interp(new, old, values.astype(cp.float32))`：`return` 先计算 `cp.interp(new, old, values.astype(cp.float32))`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_resample_linear` 所在的 `ZKX/Task2/task2_python/step4/step4.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.84 _normalize_abs：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_normalize_abs`；源码锚点 `def _normalize_abs(values):`。

```python

def _normalize_abs(values):
    magnitudes = cp.abs(values).astype(cp.float32)
    maximum = cp.max(magnitudes)
    return cp.where(maximum > 0.0, magnitudes / maximum, cp.zeros_like(magnitudes))
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `magnitudes` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `maximum` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_normalize_abs` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`magnitudes`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `magnitudes`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `magnitudes = cp.abs(values).astype(cp.float32)`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`maximum`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `maximum`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `maximum = cp.max(magnitudes)`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`_normalize_abs`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `magnitudes = cp.abs(values).astype(cp.float32)` 中，它为 `magnitudes` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `return cp.where(maximum > 0.0, magnitudes / maximum, cp.zeros_like(magnitudes))` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def _normalize_abs(values):`：`def` 创建名为 `_normalize_abs` 的函数对象；括号中的 `values` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `magnitudes = cp.abs(values).astype(cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.abs(values).astype(cp.float32)` 得到一个对象，再把名称/属性/下标 `magnitudes` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `magnitudes` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `maximum = cp.max(magnitudes)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.max(magnitudes)` 得到一个对象，再把名称/属性/下标 `maximum` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `maximum` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return cp.where(maximum > 0.0, magnitudes / maximum, cp.zeros_like(magnitudes))`：`return` 先计算 `cp.where(maximum > 0.0, magnitudes / maximum, cp.zeros_like(magnitudes))`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_normalize_abs` 所在的 `ZKX/Task2/task2_python/step4/step4.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.85 _make_bundle：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):`。

```python

def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):
    fm_n = _normalize_abs(_resample_linear(fm, count))
    corr_n = _normalize_abs(_resample_linear(correlation, count))
    spec_n = _normalize_abs(_resample_linear(spectral, count))
    wave_n = _normalize_abs(_resample_linear(wavelet, count))
    return (
        cp.float32(config.fusion_weight_fm) * fm_n
        + cp.float32(config.fusion_weight_correlation) * corr_n
        + cp.float32(config.fusion_weight_spectral) * spec_n
        + cp.float32(config.fusion_weight_wavelet) * wave_n
    ).astype(cp.float32)
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `fm_n` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `corr_n` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `spec_n` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `wave_n` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_make_bundle` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`fm_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `fm_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `fm_n = _normalize_abs(_resample_linear(fm, count))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`corr_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `corr_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `corr_n = _normalize_abs(_resample_linear(correlation, count))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`spec_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `spec_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `spec_n = _normalize_abs(_resample_linear(spectral, count))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`wave_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `wave_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `wave_n = _normalize_abs(_resample_linear(wavelet, count))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`fm`|当前函数或调用的参数名称，接收调用者绑定的输入 `fm`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`spectral`|当前函数或调用的参数名称，接收调用者绑定的输入 `spectral`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`wavelet`|当前函数或调用的参数名称，接收调用者绑定的输入 `wavelet`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`_make_bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `cp.float32(config.fusion_weight_fm) * fm_n` 中，它作为 `cp.float32` 的实参参与当前调用；在 `+ cp.float32(config.fusion_weight_correlation) * corr_n` 中，它作为 `cp.float32` 的实参参与当前调用；在 `+ cp.float32(config.fusion_weight_spectral) * spec_n` 中，它作为 `cp.float32` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):`：`def` 创建名为 `_make_bundle` 的函数对象；括号中的 `fm, correlation, spectral, wavelet, count: int, config` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `fm_n = _normalize_abs(_resample_linear(fm, count))`：这是 Python 赋值语句。解释器先完整计算右侧 `_normalize_abs(_resample_linear(fm, count))` 得到一个对象，再把名称/属性/下标 `fm_n` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `fm_n` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `corr_n = _normalize_abs(_resample_linear(correlation, count))`：这是 Python 赋值语句。解释器先完整计算右侧 `_normalize_abs(_resample_linear(correlation, count))` 得到一个对象，再把名称/属性/下标 `corr_n` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `corr_n` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `spec_n = _normalize_abs(_resample_linear(spectral, count))`：这是 Python 赋值语句。解释器先完整计算右侧 `_normalize_abs(_resample_linear(spectral, count))` 得到一个对象，再把名称/属性/下标 `spec_n` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `spec_n` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `wave_n = _normalize_abs(_resample_linear(wavelet, count))`：这是 Python 赋值语句。解释器先完整计算右侧 `_normalize_abs(_resample_linear(wavelet, count))` 得到一个对象，再把名称/属性/下标 `wave_n` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `wave_n` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return ( cp.float32(config.fusion_weight_fm) * fm_n + cp.float32(config.fusion_weight_correlation) * corr_n + cp.float32(config.fusion_weight_spectral) * spec_n + cp.float32(con...`：`return` 先计算 `( cp.float32(config.fusion_weight_fm) * fm_n + cp.float32(config.fusion_weight_correlation) * corr_n + cp.float32(config.fusion_weight_spectral) * spec_n + cp.float32(config.fusion_weight_wavelet) * wave_n ).astype(cp.float32)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `fusion_weight_fm`、`fusion_weight_correlation`、`fusion_weight_spectral`、`fusion_weight_wavelet`。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.86 run_step4：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `run_step4`；源码锚点 `def run_step4(state: PipelineState) -> StepEvidence:`。

```python

def run_step4(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step4")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.smoothed is not None and state.smoothed.size > 64, "step4 did not receive step3 output")
    require(
        state.reference_for_correlation is not None
        and state.reference_for_correlation.size == state.smoothed.size,
        "step4 correlation reference mismatch",
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `evidence` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `prep_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_step4` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `64` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`and`|当前函数或调用的参数名称，接收调用者绑定的输入 `and`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(state.smoothed is not None and state.smoothed.size > 64, "step4 did not receive step3 output")`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|
|`prep_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`run_step4`|自定义函数/可调用入口 `run_step4`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step4/step4.py` 中承担 `run_step4` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`64`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{6}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `require(state.smoothed is not None and state.smoothed.size > 64, "step4 did not receive step3 output")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def run_step4(state: PipelineState) -> StepEvidence:`：`def` 创建名为 `run_step4` 的函数对象；括号中的 `state: PipelineState` 是形参表，调用时实参按位置或关键字绑定。`-> StepEvidence` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `config = state.config`：这是 Python 赋值语句。解释器先完整计算右侧 `state.config` 得到一个对象，再把名称/属性/下标 `config` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `config` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence = StepEvidence("step4")`：这是 Python 赋值语句。解释器先完整计算右侧 `StepEvidence("step4")` 得到一个对象，再把名称/属性/下标 `evidence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `total_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `formal_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `prep_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `prep_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.smoothed is not None and state.smoothed.size > 64, "step4 did not receive step3 output")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.smoothed is not None and state.smoothed.size > 64` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step4 did not receive step3 output"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require( state.reference_for_correlation is not None and state.reference_for_correlation.size == state.smoothed.size, "step4 correlation reference mismatch", )`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.reference_for_correlation is not None and state.reference_for_correlation.size == state.smoothed.size` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step4 correlation reference mismatch"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `config`、`smoothed`、`reference_for_correlation`。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.87 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `run_step4`；源码锚点 `count = int(state.smoothed.size)`。

```python
    count = int(state.smoothed.size)
    widths = [2, 4, 8, 12]
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_signal = cp.asarray(state.smoothed, dtype=cp.float32)
    d_reference = cp.asarray(state.reference_for_correlation, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_analytic, analytic_ms = time_gpu(lambda: _analytic_signal(d_signal))
    d_fm, fm_ms = time_gpu(lambda: cusignal.fm_demod(d_analytic, axis=-1))
    d_correlation, correlation_ms = time_gpu(
        lambda: cusignal.correlate(d_signal, d_reference, mode="same", method="auto")
    )
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `count` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `widths` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `h2d_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_signal` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_reference` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `4` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `8` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `12` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`widths`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `widths`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `widths = [2, 4, 8, 12]`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`h2d_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`d_signal`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_python/step4/step4.py` 的 GPU 调用链|
|`d_reference`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_python/step4/step4.py` 的 GPU 调用链|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `d_analytic, analytic_ms = time_gpu(lambda: _analytic_signal(d_signal))`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `widths = [2, 4, 8, 12]` 中，它为 `widths` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `h2d_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `d_signal = cp.asarray(state.smoothed, dtype=cp.float32)` 中，它为 `d_signal` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `widths = [2, 4, 8, 12]` 中，它为 `widths` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`8`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{3}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `widths = [2, 4, 8, 12]` 中，它为 `widths` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`12`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `widths = [2, 4, 8, 12]` 中，它为 `widths` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `widths = [2, 4, 8, 12]` 中，它为 `widths` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `d_fm, fm_ms = time_gpu(lambda: cusignal.fm_demod(d_analytic, axis=-1))` 中，它作为 `time_gpu` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `count = int(state.smoothed.size)`：这是 Python 赋值语句。解释器先完整计算右侧 `int(state.smoothed.size)` 得到一个对象，再把名称/属性/下标 `count` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `count` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `widths = [2, 4, 8, 12]`：这是 Python 赋值语句。解释器先完整计算右侧 `[2, 4, 8, 12]` 得到一个对象，再把名称/属性/下标 `widths` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `widths` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.prep_ms = wall_ms(prep_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(prep_begin)` 得到一个对象，再把名称/属性/下标 `evidence.prep_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.prep_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `h2d_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `h2d_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_signal = cp.asarray(state.smoothed, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(state.smoothed, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_signal` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_signal` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_reference = cp.asarray(state.reference_for_correlation, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(state.reference_for_correlation, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_reference` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_reference` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `synchronize()`：`synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(h2d_begin)` 得到一个对象，再把名称/属性/下标 `evidence.h2d_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.h2d_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_analytic, analytic_ms = time_gpu(lambda: _analytic_signal(d_signal))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: _analytic_signal(d_signal))` 得到一个对象，再把名称/属性/下标 `d_analytic, analytic_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_analytic, analytic_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_fm, fm_ms = time_gpu(lambda: cusignal.fm_demod(d_analytic, axis=-1))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: cusignal.fm_demod(d_analytic, axis=-1))` 得到一个对象，再把名称/属性/下标 `d_fm, fm_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_fm, fm_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_correlation, correlation_ms = time_gpu( lambda: cusignal.correlate(d_signal, d_reference, mode="same", method="auto") )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.correlate(d_signal, d_reference, mode="same", method="auto") )` 得到一个对象，再把名称/属性/下标 `d_correlation, correlation_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_correlation, correlation_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `fm_demod`、`correlate`、`smoothed`、`reference_for_correlation`。 `fm_demod` 落实“解调”大项；`correlate` 落实“相关性”大项；四类大项是否全部完成必须结合 Step4 全调用链判断。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.88 run_spectrogram：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `run_spectrogram`；源码锚点 `def run_spectrogram():`。

```python
    def run_spectrogram():
        return cusignal.spectrogram(
            d_signal,
            fs=config.sample_rate_hz,
            window="boxcar",
            nperseg=64,
            noverlap=32,
            nfft=64,
            detrend=False,
            return_onesided=False,
            scaling="spectrum",
            mode="magnitude",
        )
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `fs` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `window` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `nperseg` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `noverlap` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `nfft` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `detrend` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `return_onesided` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `scaling` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `mode` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_spectrogram` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `64` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `32` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `64` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`fs`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `fs`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `fs=config.sample_rate_hz,`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`window`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `window`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `window="boxcar",`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`nperseg`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `nperseg`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `nperseg=64,`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`noverlap`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `noverlap`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `noverlap=32,`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`nfft`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `nfft`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `nfft=64,`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`detrend`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `detrend`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `detrend=False,`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`return_onesided`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `return_onesided`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `return_onesided=False,`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`scaling`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `scaling`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `scaling="spectrum",`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`mode`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `mode`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `mode="magnitude",`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`run_spectrogram`|自定义函数/可调用入口 `run_spectrogram`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step4/step4.py` 中承担 `run_spectrogram` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`64`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{6}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `nperseg=64,` 中，它为 `nperseg` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `nfft=64,` 中，它为 `nfft` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`32`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{5}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `noverlap=32,` 中，它为 `noverlap` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def run_spectrogram():`：`def` 创建名为 `run_spectrogram` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `return cusignal.spectrogram( d_signal, fs=config.sample_rate_hz, window="boxcar", nperseg=64, noverlap=32, nfft=64, detrend=False, return_onesided=False, scaling="spectrum", mod...`：`return` 先计算 `cusignal.spectrogram( d_signal, fs=config.sample_rate_hz, window="boxcar", nperseg=64, noverlap=32, nfft=64, detrend=False, return_onesided=False, scaling="spectrum", mode="magnitude", )`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `spectrogram`、`sample_rate_hz`。 `spectrogram` 落实“谱分析”大项；四类大项是否全部完成必须结合 Step4 全调用链判断。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.89 _make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `spectrogram_outputs, spectral_ms = time_gpu(run_spectrogram)`。

```python
    spectrogram_outputs, spectral_ms = time_gpu(run_spectrogram)
    _, _, d_spectrogram = spectrogram_outputs
    d_spectral, spectral_envelope_ms = time_gpu(
        lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32)
    )
    d_cwt, cwt_ms = time_gpu(lambda: cusignal.cwt(d_signal, cusignal.ricker, widths))
    d_wavelet, cwt_envelope_ms = time_gpu(
        lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32)
    )
    d_bundle, bundle_ms = time_gpu(
        lambda: _make_bundle(d_fm, d_correlation, d_spectral, d_wavelet, count, config)
    )
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32)`；值在 `ZKX/Task2/task2_python/step4/step4.py` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32)` 中，它作为 `cp.max` 的实参参与当前调用；在 `lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32)` 中，它作为 `cp.max` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32)` 中，它作为 `cp.max` 的实参参与当前调用；在 `lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32)` 中，它作为 `cp.max` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `spectrogram_outputs, spectral_ms = time_gpu(run_spectrogram)`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(run_spectrogram)` 得到一个对象，再把名称/属性/下标 `spectrogram_outputs, spectral_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `spectrogram_outputs, spectral_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `_, _, d_spectrogram = spectrogram_outputs`：这是 Python 赋值语句。解释器先完整计算右侧 `spectrogram_outputs` 得到一个对象，再把名称/属性/下标 `_, _, d_spectrogram` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_, _, d_spectrogram` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_spectral, spectral_envelope_ms = time_gpu( lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32) )` 得到一个对象，再把名称/属性/下标 `d_spectral, spectral_envelope_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_spectral, spectral_envelope_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_cwt, cwt_ms = time_gpu(lambda: cusignal.cwt(d_signal, cusignal.ricker, widths))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(lambda: cusignal.cwt(d_signal, cusignal.ricker, widths))` 得到一个对象，再把名称/属性/下标 `d_cwt, cwt_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_cwt, cwt_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_wavelet, cwt_envelope_ms = time_gpu( lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32) )` 得到一个对象，再把名称/属性/下标 `d_wavelet, cwt_envelope_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_wavelet, cwt_envelope_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_bundle, bundle_ms = time_gpu( lambda: _make_bundle(d_fm, d_correlation, d_spectral, d_wavelet, count, config) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: _make_bundle(d_fm, d_correlation, d_spectral, d_wavelet, count, config) )` 得到一个对象，再把名称/属性/下标 `d_bundle, bundle_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_bundle, bundle_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `spectrogram`、`cwt`、`ricker`。 `spectrogram` 落实“谱分析”大项；`cwt`/`ricker` 落实“小波变换”大项；四类大项是否全部完成必须结合 Step4 全调用链判断。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.90 _make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `evidence.operator_ms = {`。

```python
    evidence.operator_ms = {
        "cupy.analytic_glue": analytic_ms,
        "cusignal.fm_demod": fm_ms,
        "cusignal.correlate": correlation_ms,
        "cusignal.spectrogram": spectral_ms,
        "cupy.spectral_envelope": spectral_envelope_ms,
        "cusignal.cwt_ricker": cwt_ms,
        "cupy.cwt_envelope": cwt_envelope_ms,
        "cupy.feature_fusion": bundle_ms,
    }
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`cupy`、`analytic_glue`、`analytic_ms`、`cusignal`、`fm_demod`、`fm_ms`、`correlate`、`correlation_ms`、`spectrogram`、`spectral_ms`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.operator_ms = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.operator_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.operator_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"cupy.analytic_glue": analytic_ms, "cusignal.fm_demod": fm_ms, "cusignal.correlate": correlation_ms, "cusignal.spectrogram": spectral_ms, "cupy.spectral_envelope": spectral_enve...`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `fm_demod`、`correlate`、`spectrogram`、`cwt`、`feature`。 `fm_demod` 落实“解调”大项；`correlate` 落实“相关性”大项；`spectrogram` 落实“谱分析”大项；`cwt`/`ricker` 落实“小波变换”大项；四类大项是否全部完成必须结合 Step4 全调用链判断。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.91 _make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `evidence.compute_ms = sum(evidence.operator_ms.values())`。

```python
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.fm_feature = as_host(d_fm).astype(np.float32)
    state.correlation_feature = as_host(d_correlation).astype(np.float32)
    state.spectral_feature = as_host(d_spectral).astype(np.float32)
    state.wavelet_feature = as_host(d_wavelet).astype(np.float32)
    state.feature_bundle = as_host(d_bundle).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    features = {
        "fm": state.fm_feature,
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d2h_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `features` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d2h_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`post_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`features`|当前名称指定的特征数组或字段|记作 $f_m[n]$；$m$ 表示特征域|由 Step4 提取/融合，Step5/6 消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d2h_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `state.fm_feature = as_host(d_fm).astype(np.float32)` 中，它为 `state.fm_feature` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `state.correlation_feature = as_host(d_correlation).astype(np.float32)` 中，它为 `state.correlation_feature` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `evidence.compute_ms = sum(evidence.operator_ms.values())`：这是 Python 赋值语句。解释器先完整计算右侧 `sum(evidence.operator_ms.values())` 得到一个对象，再把名称/属性/下标 `evidence.compute_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.compute_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `d2h_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.fm_feature = as_host(d_fm).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_fm).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.fm_feature` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.fm_feature` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.correlation_feature = as_host(d_correlation).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_correlation).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.correlation_feature` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.correlation_feature` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.spectral_feature = as_host(d_spectral).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_spectral).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.spectral_feature` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.spectral_feature` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.wavelet_feature = as_host(d_wavelet).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_wavelet).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.wavelet_feature` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.wavelet_feature` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.feature_bundle = as_host(d_bundle).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(d_bundle).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.feature_bundle` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.feature_bundle` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.d2h_ms = wall_ms(d2h_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(d2h_begin)` 得到一个对象，再把名称/属性/下标 `evidence.d2h_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.d2h_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.formal_execution_ms = wall_ms(formal_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(formal_begin)` 得到一个对象，再把名称/属性/下标 `evidence.formal_execution_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.formal_execution_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `post_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `features = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `features` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `features` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"fm": state.fm_feature,`：运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `feature`、`fm_feature`、`correlation_feature`、`spectral_feature`、`wavelet_feature`、`feature_bundle`。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.92 _make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `"correlation": state.correlation_feature,`。

```python
        "correlation": state.correlation_feature,
        "spectral": state.spectral_feature,
        "wavelet": state.wavelet_feature,
    }
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `correlation`、`state`、`correlation_feature`、`spectral`、`spectral_feature`、`wavelet`、`wavelet_feature`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"correlation": state.correlation_feature, "spectral": state.spectral_feature, "wavelet": state.wavelet_feature, }`：运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `correlation_feature`、`spectral_feature`、`wavelet_feature`。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"correlation": state.correlation_feature,` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.93 _make_bundle：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `for name, item in features.items():`。

```python
    for name, item in features.items():
        require(np.all(np.isfinite(item)), f"step4 {name} feature is non-finite")
        require(np.max(np.abs(item)) > 0.0, f"step4 {name} feature is degenerate")
    require(np.max(state.feature_bundle) > 0.1, "step4 fused bundle degenerate")
    evidence.metrics = {
        "feature_families": 4.0,
        "demod_functions": 1.0,
        "correlate_functions": 1.0,
        "spectral_functions": 1.0,
        "wavelets_functions": 2.0,
        "step3_to_step4_data_match": 1.0,
        "feature_bundle_complete": 1.0,
```

**语法结构**

这个 Python 代码块由循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `4.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`fused`|多个特征加权融合后的序列|记作 $F[n]=\sum_m w_m f_m[n]$|峰值定位的直接输入|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `require(np.max(np.abs(item)) > 0.0, f"step4 {name} feature is degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`0.1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(np.max(state.feature_bundle) > 0.1, "step4 fused bundle degenerate")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`4.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"feature_families": 4.0,`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `"demod_functions": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"correlate_functions": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"spectral_functions": 1.0,`，作用由同一表达式中的运算符决定。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"wavelets_functions": 2.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `for name, item in features.items():`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 函数调用语法：`features.items(...)` 先准备括号内实参，再把控制权交给 `features.items`，返回值回到调用点。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `require(np.all(np.isfinite(item)), f"step4 {name} feature is non-finite")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.all(np.isfinite(item))` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`f"step4 {name} feature is non-finite"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.max(np.abs(item)) > 0.0, f"step4 {name} feature is degenerate")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.max(np.abs(item)) > 0.0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`f"step4 {name} feature is degenerate"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.max(state.feature_bundle) > 0.1, "step4 fused bundle degenerate")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.max(state.feature_bundle) > 0.1` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step4 fused bundle degenerate"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.metrics = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.metrics` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.metrics` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"feature_families": 4.0, "demod_functions": 1.0, "correlate_functions": 1.0, "spectral_functions": 1.0, "wavelets_functions": 2.0, "step3_to_step4_data_match": 1.0, "feature_bun...`：字面量与类型：`4.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `correlate`、`feature`、`feature_bundle`。 `correlate` 落实“相关性”大项；四类大项是否全部完成必须结合 Step4 全调用链判断。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.94 _make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `"fusion_weight_fm": config.fusion_weight_fm,`。

```python
        "fusion_weight_fm": config.fusion_weight_fm,
        "fusion_weight_correlation": config.fusion_weight_correlation,
        "fusion_weight_spectral": config.fusion_weight_spectral,
        "fusion_weight_wavelet": config.fusion_weight_wavelet,
        "bundle_samples": float(count),
        "bundle_peak": float(np.max(state.feature_bundle)),
    }
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `fusion_weight_fm`、`config`、`fusion_weight_correlation`、`fusion_weight_spectral`、`fusion_weight_wavelet`、`bundle_samples`、`float`、`count`、`bundle_peak`、`np`、`max`、`state`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"fusion_weight_fm": config.fusion_weight_fm, "fusion_weight_correlation": config.fusion_weight_correlation, "fusion_weight_spectral": config.fusion_weight_spectral, "fusion_weig...`：函数调用语法：`float(...)` 先准备括号内实参，再把控制权交给 `float`，返回值回到调用点；`np.max(...)` 先准备括号内实参，再把控制权交给 `np.max`，返回值回到调用点。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `feature`、`fusion_weight_fm`、`fusion_weight_correlation`、`fusion_weight_spectral`、`fusion_weight_wavelet`、`feature_bundle`。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.95 _make_bundle：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/step4.py`；符号 `_make_bundle`；源码锚点 `evidence.output_shape = list(state.feature_bundle.shape)`。

```python
    evidence.output_shape = list(state.feature_bundle.shape)
    evidence.output_dtype = str(state.feature_bundle.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`output_shape`、`list`、`state`、`feature_bundle`、`shape`、`output_dtype`、`str`、`dtype`、`post_ms`、`wall_ms`、`post_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.output_shape = list(state.feature_bundle.shape)`：这是 Python 赋值语句。解释器先完整计算右侧 `list(state.feature_bundle.shape)` 得到一个对象，再把名称/属性/下标 `evidence.output_shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.output_dtype = str(state.feature_bundle.dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `str(state.feature_bundle.dtype)` 得到一个对象，再把名称/属性/下标 `evidence.output_dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms = wall_ms(post_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(post_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.total_ms = wall_ms(total_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(total_begin)` 得到一个对象，再把名称/属性/下标 `evidence.total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return evidence`：`return` 先计算 `evidence`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `feature`、`feature_bundle`。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_make_bundle` 所在的 `ZKX/Task2/task2_python/step4/step4.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.96 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `_make_bundle`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import time
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`time`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_make_bundle` 所在的 `ZKX/Task2/task2_python/step5/step5.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.97 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `import`；源码锚点 `import numpy as np`。

```python
import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`task2_common`、`PipelineState`、`StepEvidence`、`as_host`、`cp`、`cusignal`、`require`、`synchronize`、`time_gpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, as_host, cp, cusignal, require, synchronize, time_gpu, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|公共基础设施|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.98 run_step5：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `def run_step5(state: PipelineState) -> StepEvidence:`。

```python

def run_step5(state: PipelineState) -> StepEvidence:
    order = state.config.argrelextrema_order
    evidence = StepEvidence("step5")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.feature_bundle is not None and state.feature_bundle.size > 0, "step5 did not receive step4 bundle")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_bundle = cp.asarray(state.feature_bundle, dtype=cp.float32)
    synchronize()
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `order` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `evidence` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `prep_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `h2d_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_bundle` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_step5` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`order`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `order`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `order = state.config.argrelextrema_order`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`and`|当前函数或调用的参数名称，接收调用者绑定的输入 `and`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(state.feature_bundle is not None and state.feature_bundle.size > 0, "step5 did not receive step4 bundle")`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|
|`prep_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`h2d_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`d_bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`run_step5`|自定义函数/可调用入口 `run_step5`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step5/step5.py` 中承担 `run_step5` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `require(state.feature_bundle is not None and state.feature_bundle.size > 0, "step5 did not receive step4 bundle")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `h2d_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `d_bundle = cp.asarray(state.feature_bundle, dtype=cp.float32)` 中，它为 `d_bundle` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def run_step5(state: PipelineState) -> StepEvidence:`：`def` 创建名为 `run_step5` 的函数对象；括号中的 `state: PipelineState` 是形参表，调用时实参按位置或关键字绑定。`-> StepEvidence` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `order = state.config.argrelextrema_order`：这是 Python 赋值语句。解释器先完整计算右侧 `state.config.argrelextrema_order` 得到一个对象，再把名称/属性/下标 `order` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `order` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence = StepEvidence("step5")`：这是 Python 赋值语句。解释器先完整计算右侧 `StepEvidence("step5")` 得到一个对象，再把名称/属性/下标 `evidence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `total_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `formal_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `prep_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `prep_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.feature_bundle is not None and state.feature_bundle.size > 0, "step5 did not receive step4 bundle")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.feature_bundle is not None and state.feature_bundle.size > 0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 did not receive step4 bundle"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.prep_ms = wall_ms(prep_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(prep_begin)` 得到一个对象，再把名称/属性/下标 `evidence.prep_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.prep_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `h2d_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `h2d_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_bundle = cp.asarray(state.feature_bundle, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(state.feature_bundle, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_bundle` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_bundle` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `synchronize()`：`synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|输入准备/数据契约|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `feature`、`argrelextrema`、`config`、`feature_bundle`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.99 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`。

```python
    evidence.h2d_ms = wall_ms(h2d_begin)
    result, extrema_ms = time_gpu(
        lambda: cusignal.argrelextrema(
            d_bundle, cp.greater, axis=0, order=order, mode="clip"
        )
    )
    require(len(result) == 1, "step5 coordinate rank mismatch")
    evidence.operator_ms = {"cusignal.argrelextrema": extrema_ms}
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `lambda: cusignal.argrelextrema(`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `d_bundle, cp.greater, axis=0, order=order, mode="clip"`，作用由同一表达式中的运算符决定。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `require(len(result) == 1, "step5 coordinate rank mismatch")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(h2d_begin)` 得到一个对象，再把名称/属性/下标 `evidence.h2d_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.h2d_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `result, extrema_ms = time_gpu( lambda: cusignal.argrelextrema( d_bundle, cp.greater, axis=0, order=order, mode="clip" ) )`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: cusignal.argrelextrema( d_bundle, cp.greater, axis=0, order=order, mode="clip" ) )` 得到一个对象，再把名称/属性/下标 `result, extrema_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `result, extrema_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(len(result) == 1, "step5 coordinate rank mismatch")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`len(result) == 1` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 coordinate rank mismatch"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.operator_ms = {"cusignal.argrelextrema": extrema_ms}`：这是 Python 赋值语句。解释器先完整计算右侧 `{"cusignal.argrelextrema": extrema_ms}` 得到一个对象，再把名称/属性/下标 `evidence.operator_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.operator_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `argrelextrema`、`extrema`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.100 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `evidence.compute_ms = extrema_ms`。

```python
    evidence.compute_ms = extrema_ms
    d2h_begin = time.perf_counter()
    state.extrema = as_host(result[0]).astype(np.int64)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(state.extrema.size > 0, "step5 found no feature points")
    require(np.all(np.diff(state.extrema) > 0), "step5 extrema are not ordered unique indices")
    require(
        int(state.extrema[0]) >= 0 and int(state.extrema[-1]) < state.feature_bundle.size,
        "step5 extrema index out of bounds",
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d2h_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d2h_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`post_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `state.extrema = as_host(result[0]).astype(np.int64)` 中，它为 `state.extrema` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `require(state.extrema.size > 0, "step5 found no feature points")` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `require(np.all(np.diff(state.extrema) > 0), "step5 extrema are not ordered unique indices")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `state.extrema = as_host(result[0]).astype(np.int64)` 中，它为 `state.extrema` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `int(state.extrema[0]) >= 0 and int(state.extrema[-1]) < state.feature_bundle.size,` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `evidence.compute_ms = extrema_ms`：这是 Python 赋值语句。解释器先完整计算右侧 `extrema_ms` 得到一个对象，再把名称/属性/下标 `evidence.compute_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.compute_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `d2h_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.extrema = as_host(result[0]).astype(np.int64)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(result[0]).astype(np.int64)` 得到一个对象，再把名称/属性/下标 `state.extrema` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.extrema` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.d2h_ms = wall_ms(d2h_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(d2h_begin)` 得到一个对象，再把名称/属性/下标 `evidence.d2h_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.d2h_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.formal_execution_ms = wall_ms(formal_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(formal_begin)` 得到一个对象，再把名称/属性/下标 `evidence.formal_execution_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.formal_execution_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `post_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.extrema.size > 0, "step5 found no feature points")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.extrema.size > 0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 found no feature points"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.all(np.diff(state.extrema) > 0), "step5 extrema are not ordered unique indices")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.all(np.diff(state.extrema) > 0)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 extrema are not ordered unique indices"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require( int(state.extrema[0]) >= 0 and int(state.extrema[-1]) < state.feature_bundle.size, "step5 extrema index out of bounds", )`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`int(state.extrema[0]) >= 0 and int(state.extrema[-1]) < state.feature_bundle.size` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 extrema index out of bounds"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `feature`、`extrema`、`feature_bundle`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task2/task2_python/step5/step5.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.101 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `injection_index = 50`。

```python
    injection_index = 50
    moved_index = 70
    injected = state.feature_bundle.copy()
    injected[injection_index - order : injection_index + order + 1] = 0.0
    injected[injection_index] = float(np.max(state.feature_bundle) + 1.0)
    d_injected = cp.asarray(injected, dtype=cp.float32)
    injected_result = cusignal.argrelextrema(
        d_injected, cp.greater, axis=0, order=order, mode="clip"
    )
    injected_indices = as_host(injected_result[0]).astype(np.int64)
    moved = injected.copy()
    moved[injection_index] = 0.0
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `injection_index` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `moved_index` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `injected` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_injected` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `injected_result` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `injected_indices` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `moved` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `50` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `70` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`injection_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`moved_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`injected`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `injected`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `injected = state.feature_bundle.copy()`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`d_injected`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_python/step5/step5.py` 的 GPU 调用链|
|`injected_result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`injected_indices`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `injected_indices`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `injected_indices = as_host(injected_result[0]).astype(np.int64)`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`moved`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `moved`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `moved = injected.copy()`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`50`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `injection_index = 50` 中，它为 `injection_index` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`70`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `moved_index = 70` 中，它为 `moved_index` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `injected[injection_index - order : injection_index + order + 1] = 0.0`，作用由同一表达式中的运算符决定；在 `injected[injection_index] = float(np.max(state.feature_bundle) + 1.0)` 中，它作为 `float` 的实参参与当前调用。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `injected[injection_index - order : injection_index + order + 1] = 0.0`，作用由同一表达式中的运算符决定；它直接参与表达式 `moved[injection_index] = 0.0`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `injected[injection_index] = float(np.max(state.feature_bundle) + 1.0)` 中，它作为 `float` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d_injected = cp.asarray(injected, dtype=cp.float32)` 中，它为 `d_injected` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `injection_index = 50` 中，它为 `injection_index` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `moved_index = 70` 中，它为 `moved_index` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；它直接参与表达式 `injected[injection_index - order : injection_index + order + 1] = 0.0`，作用由同一表达式中的运算符决定。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `injected_indices = as_host(injected_result[0]).astype(np.int64)` 中，它为 `injected_indices` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `injection_index = 50`：这是 Python 赋值语句。解释器先完整计算右侧 `50` 得到一个对象，再把名称/属性/下标 `injection_index` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `injection_index` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `moved_index = 70`：这是 Python 赋值语句。解释器先完整计算右侧 `70` 得到一个对象，再把名称/属性/下标 `moved_index` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved_index` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `injected = state.feature_bundle.copy()`：这是 Python 赋值语句。解释器先完整计算右侧 `state.feature_bundle.copy()` 得到一个对象，再把名称/属性/下标 `injected` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `injected` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `injected[injection_index - order : injection_index + order + 1] = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `injected[injection_index - order : injection_index + order + 1]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `injected[injection_index - order : injection_index + order + 1]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `injected[injection_index] = float(np.max(state.feature_bundle) + 1.0)`：这是 Python 赋值语句。解释器先完整计算右侧 `float(np.max(state.feature_bundle) + 1.0)` 得到一个对象，再把名称/属性/下标 `injected[injection_index]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `injected[injection_index]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_injected = cp.asarray(injected, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(injected, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_injected` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_injected` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `injected_result = cusignal.argrelextrema( d_injected, cp.greater, axis=0, order=order, mode="clip" )`：这是 Python 赋值语句。解释器先完整计算右侧 `cusignal.argrelextrema( d_injected, cp.greater, axis=0, order=order, mode="clip" )` 得到一个对象，再把名称/属性/下标 `injected_result` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `injected_result` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `injected_indices = as_host(injected_result[0]).astype(np.int64)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(injected_result[0]).astype(np.int64)` 得到一个对象，再把名称/属性/下标 `injected_indices` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `injected_indices` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `moved = injected.copy()`：这是 Python 赋值语句。解释器先完整计算右侧 `injected.copy()` 得到一个对象，再把名称/属性/下标 `moved` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `moved[injection_index] = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `moved[injection_index]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved[injection_index]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `feature`、`argrelextrema`、`feature_bundle`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.102 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `moved[moved_index - order : moved_index + order + 1] = 0.0`。

```python
    moved[moved_index - order : moved_index + order + 1] = 0.0
    moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)
    d_moved = cp.asarray(moved, dtype=cp.float32)
    moved_result = cusignal.argrelextrema(
        d_moved, cp.greater, axis=0, order=order, mode="clip"
    )
    moved_indices = as_host(moved_result[0]).astype(np.int64)
    require(injection_index in injected_indices, "step5 injected peak was not detected")
    require(moved_index in moved_indices, "step5 moved peak was not detected")
    require(injection_index not in moved_indices, "step5 old injected peak remained after movement")
    evidence.metrics = {
        "argrelextrema_called": 1.0,
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d_moved` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `moved_result` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `moved_indices` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_moved`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_python/step5/step5.py` 的 GPU 调用链|
|`moved_result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`moved_indices`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `moved_indices`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `moved_indices = as_host(moved_result[0]).astype(np.int64)`；值在 `ZKX/Task2/task2_python/step5/step5.py` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `moved[moved_index - order : moved_index + order + 1] = 0.0`，作用由同一表达式中的运算符决定；在 `moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)` 中，它作为 `float` 的实参参与当前调用；它直接参与表达式 `"argrelextrema_called": 1.0,`，作用由同一表达式中的运算符决定。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `moved[moved_index - order : moved_index + order + 1] = 0.0`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)` 中，它作为 `float` 的实参参与当前调用；它直接参与表达式 `"argrelextrema_called": 1.0,`，作用由同一表达式中的运算符决定。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d_moved = cp.asarray(moved, dtype=cp.float32)` 中，它为 `d_moved` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `moved[moved_index - order : moved_index + order + 1] = 0.0`，作用由同一表达式中的运算符决定；在 `moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)` 中，它作为 `float` 的实参参与当前调用；它直接参与表达式 `d_moved, cp.greater, axis=0, order=order, mode="clip"`，作用由同一表达式中的运算符决定。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `moved_indices = as_host(moved_result[0]).astype(np.int64)` 中，它为 `moved_indices` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `moved[moved_index - order : moved_index + order + 1] = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `moved[moved_index - order : moved_index + order + 1]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved[moved_index - order : moved_index + order + 1]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)`：这是 Python 赋值语句。解释器先完整计算右侧 `float(np.max(state.feature_bundle) + 1.0)` 得到一个对象，再把名称/属性/下标 `moved[moved_index]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved[moved_index]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_moved = cp.asarray(moved, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(moved, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_moved` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_moved` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `moved_result = cusignal.argrelextrema( d_moved, cp.greater, axis=0, order=order, mode="clip" )`：这是 Python 赋值语句。解释器先完整计算右侧 `cusignal.argrelextrema( d_moved, cp.greater, axis=0, order=order, mode="clip" )` 得到一个对象，再把名称/属性/下标 `moved_result` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved_result` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `moved_indices = as_host(moved_result[0]).astype(np.int64)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(moved_result[0]).astype(np.int64)` 得到一个对象，再把名称/属性/下标 `moved_indices` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `moved_indices` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(injection_index in injected_indices, "step5 injected peak was not detected")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`injection_index in injected_indices` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 injected peak was not detected"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(moved_index in moved_indices, "step5 moved peak was not detected")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`moved_index in moved_indices` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 moved peak was not detected"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(injection_index not in moved_indices, "step5 old injected peak remained after movement")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`injection_index not in moved_indices` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5 old injected peak remained after movement"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.metrics = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.metrics` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.metrics` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"argrelextrema_called": 1.0,`：字面量与类型：`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `feature`、`argrelextrema`、`feature_bundle`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.103 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `"step4_to_step5_data_match": 1.0,`。

```python
        "step4_to_step5_data_match": 1.0,
        "comparator_greater": 1.0,
        "axis": 0.0,
        "order": float(order),
        "mode_clip": 1.0,
        "extrema_count": float(state.extrema.size),
        "injected_peak_index": float(injection_index),
        "moved_peak_index": float(moved_index),
        "injected_peak_followed": 1.0,
        "moved_peak_followed": 1.0,
        "perturbation_checks_pass": 1.0,
    }
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `"step4_to_step5_data_match": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"comparator_greater": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"mode_clip": 1.0,`，作用由同一表达式中的运算符决定。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"axis": 0.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"step4_to_step5_data_match": 1.0, "comparator_greater": 1.0, "axis": 0.0, "order": float(order), "mode_clip": 1.0, "extrema_count": float(state.extrema.size), "injected_peak_ind...`：函数调用语法：`float(...)` 先准备括号内实参，再把控制权交给 `float`，返回值回到调用点。 字面量与类型：`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `extrema`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.104 run_step5：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/step5.py`；符号 `run_step5`；源码锚点 `evidence.output_shape = list(state.extrema.shape)`。

```python
    evidence.output_shape = list(state.extrema.shape)
    evidence.output_dtype = str(state.extrema.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`output_shape`、`list`、`state`、`extrema`、`shape`、`output_dtype`、`str`、`dtype`、`post_ms`、`wall_ms`、`post_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.output_shape = list(state.extrema.shape)`：这是 Python 赋值语句。解释器先完整计算右侧 `list(state.extrema.shape)` 得到一个对象，再把名称/属性/下标 `evidence.output_shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.output_dtype = str(state.extrema.dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `str(state.extrema.dtype)` 得到一个对象，再把名称/属性/下标 `evidence.output_dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms = wall_ms(post_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(post_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.total_ms = wall_ms(total_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(total_begin)` 得到一个对象，再把名称/属性/下标 `evidence.total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return evidence`：`return` 先计算 `evidence`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `extrema`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task2/task2_python/step5/step5.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.105 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `run_step5`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import time
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`time`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import time`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task2/task2_python/step6/step6.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.106 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `import`；源码锚点 `import numpy as np`。

```python
import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`、`task2_common`、`PipelineState`、`StepEvidence`、`as_host`、`cp`、`cusignal`、`require`、`synchronize`、`time_gpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from task2_common import ( PipelineState, StepEvidence, as_host, cp, cusignal, require, synchronize, time_gpu, wall_ms, )`：函数定义头：`import` 是函数名；名称前是返回类型和 CUDA/存储限定符，括号内逐项写“参数类型 + 参数名”，这里只建立接口，花括号内语句要等函数被调用才执行。 关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.107 _observations_from_feature_point：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `_observations_from_feature_point`；源码锚点 `def _observations_from_feature_point(bundle: np.ndarray, anchor: int, count: int) -> np.ndarray:`。

```python

def _observations_from_feature_point(bundle: np.ndarray, anchor: int, count: int) -> np.ndarray:
    observations: list[float] = []
    for radius in range(1, count + 1):
        left = max(0, anchor - radius)
        right = min(bundle.size - 1, anchor + radius)
        indices = np.arange(left, right + 1, dtype=np.float64)
        weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)
        coordinate = float(np.sum(indices * weights) / np.sum(weights)) if np.sum(weights) > 0 else float(anchor)
        observations.append(coordinate / (bundle.size - 1))
    return np.asarray(observations, dtype=np.float32)
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `observations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `left` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `right` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `indices` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `weights` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `coordinate` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_observations_from_feature_point` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`observations`|Kalman/定位观测序列|记作 $z_1,\ldots,z_K$|由特征点附近数据形成，供参数估计|
|`left`|局部区间左端索引|记作 $\ell=\max(0,a-r)$|用于安全切片或循环起点|
|`right`|局部区间右端索引|记作 $u=\min(N,a+r+1)$|用于安全切片或循环终点|
|`indices`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `indices`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `indices = np.arange(left, right + 1, dtype=np.float64)`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|
|`weights`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`coordinate`|当前特征点的离散坐标/索引|记作 $n_i$|与权重组合形成局部观测|
|`bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|
|`anchor`|局部窗口或观测序列的中心索引|记作 $a$|通常来自最强峰位置，决定左右截取范围|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`_observations_from_feature_point`|Kalman/定位观测序列|记作 $z_1,\ldots,z_K$|由特征点附近数据形成，供参数估计|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `for radius in range(1, count + 1):` 中，它参与迭代起点、终点或步长，直接决定循环执行次数；在 `right = min(bundle.size - 1, anchor + radius)` 中，它为 `right` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `indices = np.arange(left, right + 1, dtype=np.float64)` 中，它为 `indices` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `left = max(0, anchor - radius)` 中，它为 `left` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)` 中，它为 `weights` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `coordinate = float(np.sum(indices * weights) / np.sum(weights)) if np.sum(weights) > 0 else float(anchor)` 中，它为 `coordinate` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `indices = np.arange(left, right + 1, dtype=np.float64)` 中，它为 `indices` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)` 中，它为 `weights` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)` 中，它为 `weights` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `return np.asarray(observations, dtype=np.float32)` 中，它作为 `np.asarray` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def _observations_from_feature_point(bundle: np.ndarray, anchor: int, count: int) -> np.ndarray:`：`def` 创建名为 `_observations_from_feature_point` 的函数对象；括号中的 `bundle: np.ndarray, anchor: int, count: int` 是形参表，调用时实参按位置或关键字绑定。`-> np.ndarray` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `observations: list[float] = []`：这是 Python 赋值语句。解释器先完整计算右侧 `[]` 得到一个对象，再把名称/属性/下标 `observations: list[float]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `observations: list[float]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `for radius in range(1, count + 1):`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 函数调用语法：`range(...)` 先准备括号内实参，再把控制权交给 `range`，返回值回到调用点。 字面量与类型：`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`+`：加法；也可能是一元正号；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `left = max(0, anchor - radius)`：这是 Python 赋值语句。解释器先完整计算右侧 `max(0, anchor - radius)` 得到一个对象，再把名称/属性/下标 `left` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `left` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `right = min(bundle.size - 1, anchor + radius)`：这是 Python 赋值语句。解释器先完整计算右侧 `min(bundle.size - 1, anchor + radius)` 得到一个对象，再把名称/属性/下标 `right` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `right` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `indices = np.arange(left, right + 1, dtype=np.float64)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.arange(left, right + 1, dtype=np.float64)` 得到一个对象，再把名称/属性/下标 `indices` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `indices` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)`：这是 Python 赋值语句。解释器先完整计算右侧 `np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)` 得到一个对象，再把名称/属性/下标 `weights` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `weights` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `coordinate = float(np.sum(indices * weights) / np.sum(weights)) if np.sum(weights) > 0 else float(anchor)`：这是 Python 赋值语句。解释器先完整计算右侧 `float(np.sum(indices * weights) / np.sum(weights)) if np.sum(weights) > 0 else float(anchor)` 得到一个对象，再把名称/属性/下标 `coordinate` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `coordinate` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `observations.append(coordinate / (bundle.size - 1))`：`observations.append(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`coordinate / (bundle.size - 1)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return np.asarray(observations, dtype=np.float32)`：`return` 先计算 `np.asarray(observations, dtype=np.float32)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `_observations_from_feature_point` 所在的 `ZKX/Task2/task2_python/step6/step6.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.108 _estimate_with_noise：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `_estimate_with_noise`；源码锚点 `def _estimate_with_noise(observations: np.ndarray, measurement_noise: float, config) -> float:`。

```python

def _estimate_with_noise(observations: np.ndarray, measurement_noise: float, config) -> float:
    kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)
    kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)
    kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)
    kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)
    kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)
    kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)
    kalman.R[...] = cp.asarray([[[measurement_noise]]], dtype=cp.float32)
    for observation in observations:
        kalman.predict()
        kalman.update(cp.asarray([[[observation]]], dtype=cp.float32))
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `kalman` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_estimate_with_noise` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`kalman`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `kalman`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|
|`observations`|按特征点截取的观测序列|记作 $z_1,\ldots,z_K$|由 Step5 特征点附近数据构造，逐项送入 update|
|`measurement_noise`|测量噪声协方差参数|记作 $R$|用于扰动测试或 update 权重|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`_estimate_with_noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)` 中，它为 `kalman` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)` 中，它为 `kalman` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用；在 `kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def _estimate_with_noise(observations: np.ndarray, measurement_noise: float, config) -> float:`：`def` 创建名为 `_estimate_with_noise` 的函数对象；括号中的 `observations: np.ndarray, measurement_noise: float, config` 是形参表，调用时实参按位置或关键字绑定。`-> float` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[0.0]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.x[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.x[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[1.0]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.P[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.P[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_F]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.F[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.F[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.Q[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.Q[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_H]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.H[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.H[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.R[...] = cp.asarray([[[measurement_noise]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[measurement_noise]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.R[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.R[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `for observation in observations:`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `kalman.predict()`：`kalman.predict(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `kalman.update(cp.asarray([[[observation]]], dtype=cp.float32))`：`kalman.update(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`cp.asarray([[[observation]]], dtype=cp.float32)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`、`kalman_F`、`kalman_Q`、`kalman_H`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.109 float：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `float`；源码锚点 `return float(as_host(kalman.x).reshape(-1)[0])`。

```python
    return float(as_host(kalman.x).reshape(-1)[0])
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `return float(as_host(kalman.x).reshape(-1)[0])` 中，它是数组 shape/重排维度的一部分；改变后必须保持总元素数和下游数据契约一致。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `return float(as_host(kalman.x).reshape(-1)[0])` 中，它是数组 shape/重排维度的一部分；改变后必须保持总元素数和下游数据契约一致。|

**执行过程**

- 对源码锚点 `return float(as_host(kalman.x).reshape(-1)[0])`：`return` 先计算 `float(as_host(kalman.x).reshape(-1)[0])`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.110 run_step6：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `run_step6`；源码锚点 `def run_step6(state: PipelineState) -> StepEvidence:`。

```python

def run_step6(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step6")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.extrema is not None and state.extrema.size > 0, "step6 did not receive step5 points")
    strongest_position = int(np.argmax(state.feature_bundle[state.extrema]))
    state.kalman_anchor = int(state.extrema[strongest_position])
    state.kalman_observations = _observations_from_feature_point(
        state.feature_bundle, state.kalman_anchor, config.kalman_observation_count
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `config` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `evidence` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `total_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `formal_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `prep_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `strongest_position` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `run_step6` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`and`|当前函数或调用的参数名称，接收调用者绑定的输入 `and`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `require(state.extrema is not None and state.extrema.size > 0, "step6 did not receive step5 points")`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|
|`prep_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`strongest_position`|幅值或得分最大的特征点索引|记作 $n^*=\arg\max_n F[n]$|作为 Kalman anchor 或最终定位候选|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`run_step6`|自定义函数/可调用入口 `run_step6`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step6/step6.py` 中承担 `run_step6` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `require(state.extrema is not None and state.extrema.size > 0, "step6 did not receive step5 points")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def run_step6(state: PipelineState) -> StepEvidence:`：`def` 创建名为 `run_step6` 的函数对象；括号中的 `state: PipelineState` 是形参表，调用时实参按位置或关键字绑定。`-> StepEvidence` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `config = state.config`：这是 Python 赋值语句。解释器先完整计算右侧 `state.config` 得到一个对象，再把名称/属性/下标 `config` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `config` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence = StepEvidence("step6")`：这是 Python 赋值语句。解释器先完整计算右侧 `StepEvidence("step6")` 得到一个对象，再把名称/属性/下标 `evidence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `total_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `total_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `total_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `formal_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `formal_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `formal_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `prep_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `prep_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `prep_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(state.extrema is not None and state.extrema.size > 0, "step6 did not receive step5 points")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`state.extrema is not None and state.extrema.size > 0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6 did not receive step5 points"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `strongest_position = int(np.argmax(state.feature_bundle[state.extrema]))`：这是 Python 赋值语句。解释器先完整计算右侧 `int(np.argmax(state.feature_bundle[state.extrema]))` 得到一个对象，再把名称/属性/下标 `strongest_position` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `strongest_position` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.kalman_anchor = int(state.extrema[strongest_position])`：这是 Python 赋值语句。解释器先完整计算右侧 `int(state.extrema[strongest_position])` 得到一个对象，再把名称/属性/下标 `state.kalman_anchor` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.kalman_anchor` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.kalman_observations = _observations_from_feature_point( state.feature_bundle, state.kalman_anchor, config.kalman_observation_count )`：这是 Python 赋值语句。解释器先完整计算右侧 `_observations_from_feature_point( state.feature_bundle, state.kalman_anchor, config.kalman_observation_count )` 得到一个对象，再把名称/属性/下标 `state.kalman_observations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.kalman_observations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `feature`、`extrema`、`kalman`、`config`、`feature_bundle`、`kalman_anchor`、`kalman_observations`、`kalman_observation_count`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.111 run_step6：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `run_step6`；源码锚点 `state.truth = float(`。

```python
    state.truth = float(
        (0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples)
        / (state.feature_bundle.size - 1)
    )
    kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)
    kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)
    kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)
    kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)
    kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `kalman` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `h2d_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`kalman`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `kalman`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|
|`h2d_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。 当前上下文：它直接参与表达式 `(0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples)`，作用由同一表达式中的运算符决定。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `(0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples)`，作用由同一表达式中的运算符决定；它直接参与表达式 `/ (state.feature_bundle.size - 1)`，作用由同一表达式中的运算符决定；在 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)` 中，它为 `kalman` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)` 中，它为 `kalman` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `h2d_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `state.truth = float( (0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples) / (state.feature_bundle.size - 1) )`：这是 Python 赋值语句。解释器先完整计算右侧 `float( (0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples) / (state.feature_bundle.size - 1) )` 得到一个对象，再把名称/属性/下标 `state.truth` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.truth` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.prep_ms = wall_ms(prep_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(prep_begin)` 得到一个对象，再把名称/属性/下标 `evidence.prep_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.prep_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `h2d_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `h2d_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `h2d_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[0.0]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.x[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.x[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[1.0]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.P[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.P[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_F]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.F[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.F[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.Q[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.Q[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_H]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.H[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.H[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `feature`、`kalman`、`truth`、`feature_bundle`、`target_delay_samples`、`kalman_F`、`kalman_Q`、`kalman_H`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.112 run_step6：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `run_step6`；源码锚点 `kalman.R[...] = cp.asarray([[[config.kalman_R]]], dtype=cp.float32)`。

```python
    kalman.R[...] = cp.asarray([[[config.kalman_R]]], dtype=cp.float32)
    d_observations = cp.asarray(state.kalman_observations, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    if hasattr(kalman, "predict_update_sequence"):
        _, kalman_ms = time_gpu(
            lambda: kalman.predict_update_sequence(d_observations))
    else:
        # 非 ZQ500 scalar adapter 的兼容路径仍保持公开 predict/update 语义。
        kalman_ms = 0.0
        for observation in state.kalman_observations:
            d_observation = cp.asarray([[[observation]]], dtype=cp.float32)
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d_observations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `lambda` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `d_observation` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`scalar`|当前表达式读取或传递的工程名称 `scalar`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `# 非 ZQ500 scalar adapter 的兼容路径仍保持公开 predict/update 语义。`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|
|`d_observations`|Kalman/定位观测序列|记作 $z_1,\ldots,z_K$|由特征点附近数据形成，供参数估计|
|`lambda`|当前函数或调用的参数名称，接收调用者绑定的输入 `lambda`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `lambda: kalman.predict_update_sequence(d_observations))`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|
|`d_observation`|当前一次 Kalman 测量值|记作 $z_k$|逐项送入 update|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `kalman.R[...] = cp.asarray([[[config.kalman_R]]], dtype=cp.float32)` 中，它作为 `cp.asarray` 的实参参与当前调用；在 `d_observations = cp.asarray(state.kalman_observations, dtype=cp.float32)` 中，它为 `d_observations` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `evidence.h2d_ms = wall_ms(h2d_begin)` 中，它为 `evidence.h2d_ms` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `# 非 ZQ500 scalar adapter 的兼容路径仍保持公开 predict/update 语义。`，作用由同一表达式中的运算符决定。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `kalman_ms = 0.0` 中，它为 `kalman_ms` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `kalman.R[...] = cp.asarray([[[config.kalman_R]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[config.kalman_R]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `kalman.R[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman.R[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d_observations = cp.asarray(state.kalman_observations, dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(state.kalman_observations, dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_observations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_observations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `synchronize()`：`synchronize(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.h2d_ms = wall_ms(h2d_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(h2d_begin)` 得到一个对象，再把名称/属性/下标 `evidence.h2d_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.h2d_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if hasattr(kalman, "predict_update_sequence"):`：`if` 先计算条件 `hasattr(kalman, "predict_update_sequence")` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `_, kalman_ms = time_gpu( lambda: kalman.predict_update_sequence(d_observations))`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu( lambda: kalman.predict_update_sequence(d_observations))` 得到一个对象，再把名称/属性/下标 `_, kalman_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_, kalman_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `else:`：关键字语法：`else`：else 只在前一 if/else if 未进入时执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `# 非 ZQ500 scalar adapter 的兼容路径仍保持公开 predict/update 语义。`：这是源码注释；注释不参与执行。文字说明作者意图，后续解释仍以实际语句为准。
- 对源码锚点 `kalman_ms = 0.0`：这是 Python 赋值语句。解释器先完整计算右侧 `0.0` 得到一个对象，再把名称/属性/下标 `kalman_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `for observation in state.kalman_observations:`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `d_observation = cp.asarray([[[observation]]], dtype=cp.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray([[[observation]]], dtype=cp.float32)` 得到一个对象，再把名称/属性/下标 `d_observation` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d_observation` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`、`kalman_R`、`kalman_observations`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.113 predict_update：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `predict_update`；源码锚点 `def predict_update():`。

```python

            def predict_update():
                kalman.predict()
                kalman.update(d_observation)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `predict_update` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`predict_update`|自定义函数/可调用入口 `predict_update`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/step6/step6.py` 中承担 `predict_update` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def predict_update():`：`def` 创建名为 `predict_update` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> 未写返回注解` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `kalman.predict()`：`kalman.predict(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。该调用没有显式实参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `kalman.update(d_observation)`：`kalman.update(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`d_observation` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.114 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `predict_update`；源码锚点 `_, iteration_ms = time_gpu(predict_update)`。

```python
            _, iteration_ms = time_gpu(predict_update)
            kalman_ms += iteration_ms
    evidence.operator_ms = {"cusignal.KalmanFilter.predict_update": kalman_ms}
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `_`、`iteration_ms`、`time_gpu`、`predict_update`、`kalman_ms`、`evidence`、`operator_ms`、`cusignal`、`KalmanFilter`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `_, iteration_ms = time_gpu(predict_update)`：这是 Python 赋值语句。解释器先完整计算右侧 `time_gpu(predict_update)` 得到一个对象，再把名称/属性/下标 `_, iteration_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_, iteration_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `kalman_ms += iteration_ms`：这是 Python 赋值语句。解释器先完整计算右侧 `iteration_ms` 得到一个对象，再把名称/属性/下标 `kalman_ms +` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kalman_ms +` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.operator_ms = {"cusignal.KalmanFilter.predict_update": kalman_ms}`：这是 Python 赋值语句。解释器先完整计算右侧 `{"cusignal.KalmanFilter.predict_update": kalman_ms}` 得到一个对象，再把名称/属性/下标 `evidence.operator_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.operator_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.115 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `predict_update`；源码锚点 `evidence.compute_ms = kalman_ms`。

```python
    evidence.compute_ms = kalman_ms
    d2h_begin = time.perf_counter()
    estimated_state = as_host(kalman.x)
    state.kalman_covariance = as_host(kalman.P).reshape(-1).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    state.estimate = float(estimated_state.reshape(-1)[0])
    estimate_error = abs(state.estimate - state.truth)
    require(np.isfinite(state.estimate), "step6 estimate is non-finite")
    require(np.all(np.isfinite(state.kalman_covariance)), "step6 covariance is non-finite")
    require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `d2h_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `estimated_state` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `post_begin` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `estimate_error` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.18` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d2h_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`estimated_state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`post_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`estimate_error`|当前名称指定的误差指标或异常状态|对应 $g-r$、其范数或语义判定|由 GPU 与 reference 差异产生，进入门禁|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`threshold`|逐单元检测门限|记作 $T[i]$|与 CUT 功率比较产生 detection|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `state.kalman_covariance = as_host(kalman.P).reshape(-1).astype(np.float32)` 中，它为 `state.kalman_covariance` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `state.estimate = float(estimated_state.reshape(-1)[0])` 中，它为 `state.estimate` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `d2h_begin = time.perf_counter()` 中，它作为 `time.perf_counter` 的实参参与当前调用；在 `state.kalman_covariance = as_host(kalman.P).reshape(-1).astype(np.float32)` 中，它为 `state.kalman_covariance` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `evidence.d2h_ms = wall_ms(d2h_begin)` 中，它为 `evidence.d2h_ms` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `state.estimate = float(estimated_state.reshape(-1)[0])` 中，它为 `state.estimate` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`0.18`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `evidence.compute_ms = kalman_ms`：这是 Python 赋值语句。解释器先完整计算右侧 `kalman_ms` 得到一个对象，再把名称/属性/下标 `evidence.compute_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.compute_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `d2h_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `d2h_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `d2h_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `estimated_state = as_host(kalman.x)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(kalman.x)` 得到一个对象，再把名称/属性/下标 `estimated_state` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `estimated_state` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.kalman_covariance = as_host(kalman.P).reshape(-1).astype(np.float32)`：这是 Python 赋值语句。解释器先完整计算右侧 `as_host(kalman.P).reshape(-1).astype(np.float32)` 得到一个对象，再把名称/属性/下标 `state.kalman_covariance` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.kalman_covariance` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.d2h_ms = wall_ms(d2h_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(d2h_begin)` 得到一个对象，再把名称/属性/下标 `evidence.d2h_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.d2h_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.formal_execution_ms = wall_ms(formal_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(formal_begin)` 得到一个对象，再把名称/属性/下标 `evidence.formal_execution_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.formal_execution_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `post_begin = time.perf_counter()`：这是 Python 赋值语句。解释器先完整计算右侧 `time.perf_counter()` 得到一个对象，再把名称/属性/下标 `post_begin` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `post_begin` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `state.estimate = float(estimated_state.reshape(-1)[0])`：这是 Python 赋值语句。解释器先完整计算右侧 `float(estimated_state.reshape(-1)[0])` 得到一个对象，再把名称/属性/下标 `state.estimate` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `state.estimate` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `estimate_error = abs(state.estimate - state.truth)`：这是 Python 赋值语句。解释器先完整计算右侧 `abs(state.estimate - state.truth)` 得到一个对象，再把名称/属性/下标 `estimate_error` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `estimate_error` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(np.isfinite(state.estimate), "step6 estimate is non-finite")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.isfinite(state.estimate)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6 estimate is non-finite"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(np.all(np.isfinite(state.kalman_covariance)), "step6 covariance is non-finite")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.all(np.isfinite(state.kalman_covariance))` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6 covariance is non-finite"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`estimate_error <= 0.18` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6 estimate error exceeds approved threshold"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`、`estimate`、`covariance`、`kalman_covariance`、`truth`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.116 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `predict_update`；源码锚点 `high_noise_estimate = _estimate_with_noise(`。

```python
    high_noise_estimate = _estimate_with_noise(
        state.kalman_observations, config.kalman_measurement_noise_variant, config
    )
    measurement_noise_response = abs(high_noise_estimate - state.estimate)
    require(np.isfinite(high_noise_estimate), "step6 noise perturbation is non-finite")
    require(
        measurement_noise_response > 1.0e-6,
        "step6 measurement-noise perturbation did not change estimate",
    )
    evidence.metrics = {
        "kalmanfilter_called": 1.0,
        "step5_to_step6_data_match": 1.0,
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `high_noise_estimate` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `measurement_noise_response` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1.0e-6` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`high_noise_estimate`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`measurement_noise_response`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`1.0e-6`|当前函数或调用的参数名称，接收调用者绑定的输入 `1.0e-6`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `measurement_noise_response > 1.0e-6,`；值在 `ZKX/Task2/task2_python/step6/step6.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0e-6`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `measurement_noise_response > 1.0e-6,` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `measurement_noise_response > 1.0e-6,` 中，它是分支或边界比较值，决定哪条控制路径被执行；它直接参与表达式 `"kalmanfilter_called": 1.0,`，作用由同一表达式中的运算符决定；它直接参与表达式 `"step5_to_step6_data_match": 1.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `high_noise_estimate = _estimate_with_noise( state.kalman_observations, config.kalman_measurement_noise_variant, config )`：这是 Python 赋值语句。解释器先完整计算右侧 `_estimate_with_noise( state.kalman_observations, config.kalman_measurement_noise_variant, config )` 得到一个对象，再把名称/属性/下标 `high_noise_estimate` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `high_noise_estimate` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `measurement_noise_response = abs(high_noise_estimate - state.estimate)`：这是 Python 赋值语句。解释器先完整计算右侧 `abs(high_noise_estimate - state.estimate)` 得到一个对象，再把名称/属性/下标 `measurement_noise_response` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `measurement_noise_response` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `require(np.isfinite(high_noise_estimate), "step6 noise perturbation is non-finite")`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`np.isfinite(high_noise_estimate)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6 noise perturbation is non-finite"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `require( measurement_noise_response > 1.0e-6, "step6 measurement-noise perturbation did not change estimate", )`：`require(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`measurement_noise_response > 1.0e-6` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6 measurement-noise perturbation did not change estimate"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `evidence.metrics = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `evidence.metrics` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.metrics` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"kalmanfilter_called": 1.0, "step5_to_step6_data_match": 1.0,`：字面量与类型：`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`、`estimate`、`kalman_observations`、`kalman_measurement_noise_variant`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.117 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `predict_update`；源码锚点 `"kalman_anchor": float(state.kalman_anchor),`。

```python
        "kalman_anchor": float(state.kalman_anchor),
        "observation_count": float(state.kalman_observations.size),
        "estimate": state.estimate,
        "truth": state.truth,
        "estimate_abs_error": estimate_error,
        "estimate_error_limit": 0.18,
        "final_covariance": float(state.kalman_covariance[0]),
        "high_measurement_noise_estimate": high_noise_estimate,
        "measurement_noise_response": measurement_noise_response,
        "perturbation_checks_pass": 1.0,
    }
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0.18` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.18`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `"estimate_error_limit": 0.18,`，作用由同一表达式中的运算符决定。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"estimate_error_limit": 0.18,`，作用由同一表达式中的运算符决定；在 `"final_covariance": float(state.kalman_covariance[0]),` 中，它作为 `float` 的实参参与当前调用；它直接参与表达式 `"perturbation_checks_pass": 1.0,`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `"perturbation_checks_pass": 1.0,`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"kalman_anchor": float(state.kalman_anchor), "observation_count": float(state.kalman_observations.size), "estimate": state.estimate, "truth": state.truth, "estimate_abs_error": ...`：函数调用语法：`float(...)` 先准备括号内实参，再把控制权交给 `float`，返回值回到调用点。 字面量与类型：`0.18` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`、`estimate`、`kalman_anchor`、`kalman_observations`、`truth`、`kalman_covariance`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.118 predict_update：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/step6.py`；符号 `predict_update`；源码锚点 `evidence.output_shape = [1]`。

```python
    evidence.output_shape = [1]
    evidence.output_dtype = "float32"
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `evidence.output_shape = [1]` 中，它为 `evidence.output_shape` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `evidence.output_dtype = "float32"` 中，它为 `evidence.output_dtype` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `evidence.output_shape = [1]`：这是 Python 赋值语句。解释器先完整计算右侧 `[1]` 得到一个对象，再把名称/属性/下标 `evidence.output_shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.output_dtype = "float32"`：这是 Python 赋值语句。解释器先完整计算右侧 `"float32"` 得到一个对象，再把名称/属性/下标 `evidence.output_dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.output_dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.post_ms = wall_ms(post_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(post_begin)` 得到一个对象，再把名称/属性/下标 `evidence.post_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.post_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `evidence.total_ms = wall_ms(total_begin)`：这是 Python 赋值语句。解释器先完整计算右侧 `wall_ms(total_begin)` 得到一个对象，再把名称/属性/下标 `evidence.total_ms` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `evidence.total_ms` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return evidence`：`return` 先计算 `evidence`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update` 所在的 `ZKX/Task2/task2_python/step6/step6.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.119 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""`。

```python
"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""

from __future__ import annotations
```

**语法结构**

这个 Python 代码块由循环构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Runtime`、`only`、`ZQ500`、`compatibility`、`helpers`、`the`、`immutable`、`cuSignal`、`source`、`__future__`、`annotations`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`compatibility`|当前表达式读取或传递的工程名称 `compatibility`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""` 中，它参与迭代起点、终点或步长，直接决定循环执行次数。|

**执行过程**

- 对源码锚点 `"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。
- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.120 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `import copy`。

```python
import copy
import importlib
from typing import Any
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `copy`、`importlib`、`typing`、`Any`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import copy`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import importlib`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from typing import Any`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `import copy` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.121 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `import numpy as np`。

```python
import numpy as np
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `numpy`、`as`、`np`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `import numpy as np`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `import numpy as np` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.122 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `_INSTALLED = False`。

```python

_INSTALLED = False
_DETAILS: dict[str, Any] = {}
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_INSTALLED` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_DETAILS` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`_DETAILS`|当前表达式读取或传递的工程名称 `_DETAILS`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `_DETAILS: dict[str, Any] = {}`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`_INSTALLED`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `_INSTALLED`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `_INSTALLED = False`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `_INSTALLED = False`：这是 Python 赋值语句。解释器先完整计算右侧 `False` 得到一个对象，再把名称/属性/下标 `_INSTALLED` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_INSTALLED` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `_DETAILS: dict[str, Any] = {}`：这是 Python 赋值语句。解释器先完整计算右侧 `{}` 得到一个对象，再把名称/属性/下标 `_DETAILS: dict[str, Any]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_DETAILS: dict[str, Any]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `_INSTALLED = False` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.123 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {`。

```python

ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {
    "waveform_source_kernels": {
        "public_apis": ["cusignal.chirp", "cusignal.sawtooth", "cusignal.square"],
        "upstream_backend": "cusignal.waveforms.waveforms ElementwiseKernel symbols",
        "zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",
        "adapter_action": "install source-JIT kernels with the same public parameters, equations and upstream output dtypes",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `ADAPTER_CONTRACTS` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`dict`|当前表达式读取或传递的工程名称 `dict`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`symbols`|当前表达式读取或传递的工程名称 `symbols`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"upstream_backend": "cusignal.waveforms.waveforms ElementwiseKernel symbols",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`kernel`|当前表达式读取或传递的工程名称 `kernel`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`CuPy`|当前表达式读取或传递的工程名称 `CuPy`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`kernels`|当前表达式读取或传递的工程名称 `kernels`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"adapter_action": "install source-JIT kernels with the same public parameters, equations and upstream output dtypes",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`ADAPTER_CONTRACTS`|当前表达式读取或传递的工程名称 `ADAPTER_CONTRACTS`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",`，作用由同一表达式中的运算符决定。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `ADAPTER_CONTRACTS: dict[str, dict[str, Any]]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `ADAPTER_CONTRACTS: dict[str, dict[str, Any]]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"waveform_source_kernels": {`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `"public_apis": ["cusignal.chirp", "cusignal.sawtooth", "cusignal.square"], "upstream_backend": "cusignal.waveforms.waveforms ElementwiseKernel symbols", "zq500_reason": "upstrea...`：关键字语法：`with`：with 调用上下文管理器，保证退出时执行清理。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.124 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `"cupy_direct_convolution_backend": {`。

```python
    "cupy_direct_convolution_backend": {
        "public_apis": ["cusignal.firfilter", "cusignal.correlate", "cusignal.cwt"],
        "upstream_backend": "CuPy/cupyx one-dimensional direct convolve used by cuSignal",
        "zq500_reason": "the private ZQ500 CuPy convolution backend lacks the required working one-dimensional source path",
        "adapter_action": "provide the same full/same/valid direct-convolution primitive on ZQ500 device arrays",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "argrelextrema_backend": {
        "public_apis": ["cusignal.argrelextrema"],
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `cupy_direct_convolution_backend`、`public_apis`、`cusignal`、`firfilter`、`correlate`、`cwt`、`upstream_backend`、`CuPy`、`cupyx`、`one`、`dimensional`、`direct`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`CuPy`|当前表达式读取或传递的工程名称 `CuPy`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"upstream_backend": "CuPy/cupyx one-dimensional direct convolve used by cuSignal",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`device`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"zq500_reason": "the private ZQ500 CuPy convolution backend lacks the required working one-dimensional source path",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"adapter_action": "provide the same full/same/valid direct-convolution primitive on ZQ500 device arrays",`，作用由同一表达式中的运算符决定。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"zq500_reason": "the private ZQ500 CuPy convolution backend lacks the required working one-dimensional source path",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"adapter_action": "provide the same full/same/valid direct-convolution primitive on ZQ500 device arrays",`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"cupy_direct_convolution_backend": {`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `"public_apis": ["cusignal.firfilter", "cusignal.correlate", "cusignal.cwt"], "upstream_backend": "CuPy/cupyx one-dimensional direct convolve used by cuSignal", "zq500_reason": "...`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。
- 对源码锚点 `"public_apis": ["cusignal.argrelextrema"],`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 方括号用于序列/映射下标、切片或列表字面量；具体含义由括号前后的对象决定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 `firfilter`、`correlate`、`cwt`、`argrelextrema`。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.125 predict_update：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `"upstream_backend": "cusignal.peak_finding._boolrelextrema compiled-kernel path",`。

```python
        "upstream_backend": "cusignal.peak_finding._boolrelextrema compiled-kernel path",
        "zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable on ZQ500",
        "adapter_action": "use cuSignal's own generic take/compare algorithm for the Task2 one-dimensional input",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "scalar_kalman_backend": {
        "public_apis": ["cusignal.KalmanFilter", "KalmanFilter.predict", "KalmanFilter.update", "KalmanFilter.predict_update_sequence"],
        "upstream_backend": "cusignal.estimation._filters_cuda RawModule specialization",
        "zq500_reason": "ZQ500 CuPy does not support the upstream NVRTC lowered-name/name_expressions path",
```

**语法结构**

这个 Python 代码块由循环构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `upstream_backend`、`cusignal`、`peak_finding`、`_boolrelextrema`、`compiled`、`kernel`、`path`、`zq500_reason`、`the`、`upstream`、`helper`、`depends`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`kernel`|当前表达式读取或传递的工程名称 `kernel`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"upstream_backend": "cusignal.peak_finding._boolrelextrema compiled-kernel path",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`one`|当前表达式读取或传递的工程名称 `one`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"adapter_action": "use cuSignal's own generic take/compare algorithm for the Task2 one-dimensional input",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`specialization`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`CuPy`|当前表达式读取或传递的工程名称 `CuPy`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable on ZQ500",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`lowered`|当前表达式读取或传递的工程名称 `lowered`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"zq500_reason": "ZQ500 CuPy does not support the upstream NVRTC lowered-name/name_expressions path",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable on ZQ500",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"zq500_reason": "ZQ500 CuPy does not support the upstream NVRTC lowered-name/name_expressions path",`，作用由同一表达式中的运算符决定。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable on ZQ500",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"zq500_reason": "ZQ500 CuPy does not support the upstream NVRTC lowered-name/name_expressions path",`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"upstream_backend": "cusignal.peak_finding._boolrelextrema compiled-kernel path", "zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable...`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `"public_apis": ["cusignal.KalmanFilter", "KalmanFilter.predict", "KalmanFilter.update", "KalmanFilter.predict_update_sequence"], "upstream_backend": "cusignal.estimation._filter...`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 `kalman`。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.126 predict_update：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `"adapter_action": "evaluate the same scalar predict/update equations with CuPy device operations for dim_x=dim_z=1",`。

```python
        "adapter_action": "evaluate the same scalar predict/update equations with CuPy device operations for dim_x=dim_z=1",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
        "approved_task_shape": {"dim_x": 1, "dim_z": 1, "dim_u": 0},
    },
}
```

**语法结构**

这个 Python 代码块由循环构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`device`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `"adapter_action": "evaluate the same scalar predict/update equations with CuPy device operations for dim_x=dim_z=1",` 中，它参与迭代起点、终点或步长，直接决定循环执行次数；在 `"approved_task_shape": {"dim_x": 1, "dim_z": 1, "dim_u": 0},` 中，它是数组 shape/重排维度的一部分；改变后必须保持总元素数和下游数据契约一致。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `"approved_task_shape": {"dim_x": 1, "dim_z": 1, "dim_u": 0},` 中，它是数组 shape/重排维度的一部分；改变后必须保持总元素数和下游数据契约一致。|

**执行过程**

- 对源码锚点 `"adapter_action": "evaluate the same scalar predict/update equations with CuPy device operations for dim_x=dim_z=1", "semantic_change": False, "cpu_fallback": False, "device_exe...`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行；`with`：with 调用上下文管理器，保证退出时执行清理。 字面量与类型：`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作。

### 4.127 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {`。

```python

INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {
    "waveform.sawtooth_square_outputs": {
        "business_input_dtype": "FP32",
        "intermediate_output_dtype": "FP64",
        "source": "cuSignal 23.08 waveform public output contract",
        "reason_retained": "changing the public cuSignal output dtype would make the comparison semantically unequal",
        "device_only": True,
        "runtime_evidence": "stock/adapted probe plus Task2 Step1 and pipeline acceptance on ZQ500",
    },
    "step4.cwt_output": {
        "business_input_dtype": "FP32",
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `INTERMEDIATE_DOUBLE_AUDIT` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`dict`|当前表达式读取或传递的工程名称 `dict`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`Step1`|当前表达式读取或传递的工程名称 `Step1`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"runtime_evidence": "stock/adapted probe plus Task2 Step1 and pipeline acceptance on ZQ500",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`INTERMEDIATE_DOUBLE_AUDIT`|当前表达式读取或传递的工程名称 `INTERMEDIATE_DOUBLE_AUDIT`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：它直接参与表达式 `"business_input_dtype": "FP32",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"source": "cuSignal 23.08 waveform public output contract",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"runtime_evidence": "stock/adapted probe plus Task2 Step1 and pipeline acceptance on ZQ500",`，作用由同一表达式中的运算符决定。|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"intermediate_output_dtype": "FP64",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"step4.cwt_output": {`，作用由同一表达式中的运算符决定。|
|`23.08`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `"source": "cuSignal 23.08 waveform public output contract",`，作用由同一表达式中的运算符决定。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"runtime_evidence": "stock/adapted probe plus Task2 Step1 and pipeline acceptance on ZQ500",`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {`：这是 Python 赋值语句。解释器先完整计算右侧 `{` 得到一个对象，再把名称/属性/下标 `INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"waveform.sawtooth_square_outputs": {`：运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `"business_input_dtype": "FP32", "intermediate_output_dtype": "FP64", "source": "cuSignal 23.08 waveform public output contract", "reason_retained": "changing the public cuSignal...`：字面量与类型：`23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `"business_input_dtype": "FP32",`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`cwt`。 本块直接出现 2 种波形符号：`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同。

### 4.128 predict_update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update`；源码锚点 `"intermediate_output_dtype": "FP64",`。

```python
        "intermediate_output_dtype": "FP64",
        "source": "cuSignal 23.08 cwt/ricker implementation-selected output",
        "reason_retained": "the Task2 adapter preserves the upstream public API instead of silently narrowing it",
        "device_only": True,
        "runtime_evidence": "Task2 Step4 and pipeline acceptance on ZQ500",
    },
}
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`adapter`|当前表达式读取或传递的工程名称 `adapter`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"reason_retained": "the Task2 adapter preserves the upstream public API instead of silently narrowing it",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`instead`|当前表达式读取或传递的工程名称 `instead`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"reason_retained": "the Task2 adapter preserves the upstream public API instead of silently narrowing it",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`Step4`|当前表达式读取或传递的工程名称 `Step4`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"runtime_evidence": "Task2 Step4 and pipeline acceptance on ZQ500",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"intermediate_output_dtype": "FP64",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"runtime_evidence": "Task2 Step4 and pipeline acceptance on ZQ500",`，作用由同一表达式中的运算符决定。|
|`23.08`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `"source": "cuSignal 23.08 cwt/ricker implementation-selected output",`，作用由同一表达式中的运算符决定。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"runtime_evidence": "Task2 Step4 and pipeline acceptance on ZQ500",`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"intermediate_output_dtype": "FP64", "source": "cuSignal 23.08 cwt/ricker implementation-selected output", "reason_retained": "the Task2 adapter preserves the upstream public AP...`：字面量与类型：`23.08` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 `cwt`、`ricker`。 `cwt`/`ricker` 落实“小波变换”大项；四类大项是否全部完成必须结合 Step4 全调用链判断。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同。

### 4.129 __init__：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `__init__`；源码锚点 `class _CupyProxy:`。

```python

class _CupyProxy:
    def __init__(self, cupy_module: Any, convolve_function: Any) -> None:
        self._cupy = cupy_module
        self.convolve = convolve_function
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `__init__` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def __init__(self, cupy_module: Any, convolve_function: Any) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`cupy_module`|当前函数或调用的参数名称，接收调用者绑定的输入 `cupy_module`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def __init__(self, cupy_module: Any, convolve_function: Any) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`convolve_function`|当前函数或调用的参数名称，接收调用者绑定的输入 `convolve_function`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def __init__(self, cupy_module: Any, convolve_function: Any) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`__init__`|自定义函数/可调用入口 `__init__`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `__init__` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `class _CupyProxy:`：关键字语法：`class`：class 创建类型对象并建立其属性与方法命名空间。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `def __init__(self, cupy_module: Any, convolve_function: Any) -> None:`：`def` 创建名为 `__init__` 的函数对象；括号中的 `self, cupy_module: Any, convolve_function: Any` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `self._cupy = cupy_module`：这是 Python 赋值语句。解释器先完整计算右侧 `cupy_module` 得到一个对象，再把名称/属性/下标 `self._cupy` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self._cupy` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.convolve = convolve_function`：这是 Python 赋值语句。解释器先完整计算右侧 `convolve_function` 得到一个对象，再把名称/属性/下标 `self.convolve` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.convolve` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `__init__` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.130 __getattr__：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `__getattr__`；源码锚点 `def __getattr__(self, name: str) -> Any:`。

```python
    def __getattr__(self, name: str) -> Any:
        return getattr(self._cupy, name)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `__getattr__` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`return`|当前表达式读取或传递的工程名称 `return`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `return getattr(self._cupy, name)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def __getattr__(self, name: str) -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`name`|当前函数或调用的参数名称，接收调用者绑定的输入 `name`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def __getattr__(self, name: str) -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`__getattr__`|自定义函数/可调用入口 `__getattr__`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `__getattr__` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def __getattr__(self, name: str) -> Any:`：`def` 创建名为 `__getattr__` 的函数对象；括号中的 `self, name: str` 是形参表，调用时实参按位置或关键字绑定。`-> Any` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `return getattr(self._cupy, name)`：`return` 先计算 `getattr(self._cupy, name)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `__getattr__` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.131 _install_source_convolution：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_source_convolution`；源码锚点 `def _install_source_convolution(cp: Any) -> Any:`。

```python

def _install_source_convolution(cp: Any) -> Any:
    kernel = cp.ElementwiseKernel(
        "raw T first, raw T second, int32 first_size, int32 second_size, int32 offset",
        "T output",
        """
        const int full_index = i + offset;
        T accumulator {};
        for (int second_index = 0; second_index < second_size; ++second_index) {
            const int first_index = full_index - second_index;
            if (first_index >= 0 && first_index < first_size) {
                accumulator += first[first_index] * second[second_index];
            }
        }
        output = accumulator;
        """,
        "_task2_zq500_convolve_1d",
        options=("-std=c++11",),
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `kernel` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `output` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `options` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_install_source_convolution` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `11` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`kernel`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `kernel`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `kernel = cp.ElementwiseKernel(`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`first`|当前函数或调用的参数名称，接收调用者绑定的输入 `first`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"raw T first, raw T second, int32 first_size, int32 second_size, int32 offset",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`second`|当前函数或调用的参数名称，接收调用者绑定的输入 `second`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"raw T first, raw T second, int32 first_size, int32 second_size, int32 offset",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`full_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`accumulator`|当前函数或调用的参数名称，接收调用者绑定的输入 `accumulator`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `T accumulator {};`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`second_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`first_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`options`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `options=("-std=c++11",),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`_install_source_convolution`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"raw T first, raw T second, int32 first_size, int32 second_size, int32 offset",`，作用由同一表达式中的运算符决定；它直接参与表达式 `"_task2_zq500_convolve_1d",`，作用由同一表达式中的运算符决定。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `for (int second_index = 0; second_index < second_size; ++second_index) {` 中，它参与迭代起点、终点或步长，直接决定循环执行次数；在 `if (first_index >= 0 && first_index < first_size) {` 中，它是分支或边界比较值，决定哪条控制路径被执行；它直接参与表达式 `"_task2_zq500_convolve_1d",`，作用由同一表达式中的运算符决定。|
|`11`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `options=("-std=c++11",),` 中，它为 `options` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def _install_source_convolution(cp: Any) -> Any:`：`def` 创建名为 `_install_source_convolution` 的函数对象；括号中的 `cp: Any` 是形参表，调用时实参按位置或关键字绑定。`-> Any` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `kernel = cp.ElementwiseKernel( "raw T first, raw T second, int32 first_size, int32 second_size, int32 offset", "T output", """ const int full_index = i + offset; T accumulator {...`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ElementwiseKernel( "raw T first, raw T second, int32 first_size, int32 second_size, int32 offset", "T output", """ const int full_index = i + offset; T accumulator {}; for (int second_index = 0; second_index < second_size; ++second_index) { const int first_index = full_index - second_index; if (first_index >= 0 && first_index < first_size) { accumulator += first[first_index] * second[second_index]; } } output = accumulator; """, "_task2_zq500_convolve_1d", options=("-std=c++11",), )` 得到一个对象，再把名称/属性/下标 `kernel` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `kernel` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.132 direct_convolve：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `direct_convolve`；源码锚点 `def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:`。

```python

    def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:
        del method
        first = cp.asarray(first)
        second = cp.asarray(second)
        if first.ndim == second.ndim == 0:
            return first * second
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `first` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `second` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `direct_convolve` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`del`|当前表达式读取或传递的工程名称 `del`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `del method`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`first`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `first`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`second`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `second`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`mode`|当前函数或调用的参数名称，接收调用者绑定的输入 `mode`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`method`|当前函数或调用的参数名称，接收调用者绑定的输入 `method`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`direct_convolve`|自定义函数/可调用入口 `direct_convolve`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `direct_convolve` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `if first.ndim == second.ndim == 0:` 中，它为 `first.ndim` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:`：`def` 创建名为 `direct_convolve` 的函数对象；括号中的 `first: Any, second: Any, mode: str = "full", method: str = "auto"` 是形参表，调用时实参按位置或关键字绑定。`-> Any` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `del method`：名称解析：`del`、`method`按词法作用域查找对应变量、类型、函数或常量；逗号把当前项目与同一外层调用/容器中的下一项目分开，分号仅在 C/C++ 中结束完整语句。
- 对源码锚点 `first = cp.asarray(first)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(first)` 得到一个对象，再把名称/属性/下标 `first` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `first` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `second = cp.asarray(second)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(second)` 得到一个对象，再把名称/属性/下标 `second` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `second` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if first.ndim == second.ndim == 0:`：`if` 先计算条件 `first.ndim == second.ndim == 0` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return first * second`：`return` 先计算 `first * second`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.133 direct_convolve：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `direct_convolve`；源码锚点 `if first.ndim != 1 or second.ndim != 1:`。

```python
        if first.ndim != 1 or second.ndim != 1:
            raise ValueError("ZQ500 source convolution adapter supports one-dimensional inputs")
        if mode not in ("full", "same", "valid"):
            raise ValueError("mode must be 'full', 'same', or 'valid'")
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`source`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `if first.ndim != 1 or second.ndim != 1:` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `raise ValueError("ZQ500 source convolution adapter supports one-dimensional inputs")` 中，它作为 `ValueError` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if first.ndim != 1 or second.ndim != 1:`：`if` 先计算条件 `first.ndim != 1 or second.ndim != 1` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise ValueError("ZQ500 source convolution adapter supports one-dimensional inputs")`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`ValueError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"ZQ500 source convolution adapter supports one-dimensional inputs"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `if mode not in ("full", "same", "valid"):`：`if` 先计算条件 `mode not in ("full", "same", "valid")` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise ValueError("mode must be 'full', 'same', or 'valid'")`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`ValueError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"mode must be 'full', 'same', or 'valid'"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.134 direct_convolve：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `direct_convolve`；源码锚点 `dtype = cp.result_type(first, second)`。

```python
        dtype = cp.result_type(first, second)
        first = first.astype(dtype, copy=False)
        second = second.astype(dtype, copy=False)
        full_size = int(first.size + second.size - 1)
        if mode == "full":
            output_size = full_size
            offset = 0
        elif mode == "same":
            output_size = int(first.size)
            offset = (full_size - output_size) // 2
        else:
            output_size = abs(int(first.size) - int(second.size)) + 1
```

**语法结构**

这个 Python 代码块由变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `dtype` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `first` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `second` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `full_size` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `output_size` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `offset` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `output_size` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `offset` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `else` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`dtype`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `dtype`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `dtype = cp.result_type(first, second)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`first`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `first`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `dtype = cp.result_type(first, second)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`second`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `second`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `dtype = cp.result_type(first, second)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`full_size`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`output_size`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`offset`|相对起点的离散偏移|记作 $\Delta n$|参与索引换算或数据切片|
|`else`|当前表达式读取或传递的工程名称 `else`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `else:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `full_size = int(first.size + second.size - 1)` 中，它为 `full_size` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `output_size = abs(int(first.size) - int(second.size)) + 1` 中，它为 `output_size` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `offset = 0` 中，它为 `offset` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `offset = (full_size - output_size) // 2` 中，它为 `offset` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `dtype = cp.result_type(first, second)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.result_type(first, second)` 得到一个对象，再把名称/属性/下标 `dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `first = first.astype(dtype, copy=False)`：这是 Python 赋值语句。解释器先完整计算右侧 `first.astype(dtype, copy=False)` 得到一个对象，再把名称/属性/下标 `first` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `first` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `second = second.astype(dtype, copy=False)`：这是 Python 赋值语句。解释器先完整计算右侧 `second.astype(dtype, copy=False)` 得到一个对象，再把名称/属性/下标 `second` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `second` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `full_size = int(first.size + second.size - 1)`：这是 Python 赋值语句。解释器先完整计算右侧 `int(first.size + second.size - 1)` 得到一个对象，再把名称/属性/下标 `full_size` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `full_size` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if mode == "full":`：`if` 先计算条件 `mode == "full"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `output_size = full_size`：这是 Python 赋值语句。解释器先完整计算右侧 `full_size` 得到一个对象，再把名称/属性/下标 `output_size` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `output_size` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `offset = 0`：这是 Python 赋值语句。解释器先完整计算右侧 `0` 得到一个对象，再把名称/属性/下标 `offset` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `offset` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `elif mode == "same":`：运算符：`==`：相等比较，结果类型为 bool；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `output_size = int(first.size)`：这是 Python 赋值语句。解释器先完整计算右侧 `int(first.size)` 得到一个对象，再把名称/属性/下标 `output_size` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `output_size` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `offset = (full_size - output_size) // 2`：这是 Python 赋值语句。解释器先完整计算右侧 `(full_size - output_size) // 2` 得到一个对象，再把名称/属性/下标 `offset` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `offset` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `else:`：关键字语法：`else`：else 只在前一 if/else if 未进入时执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `output_size = abs(int(first.size) - int(second.size)) + 1`：这是 Python 赋值语句。解释器先完整计算右侧 `abs(int(first.size) - int(second.size)) + 1` 得到一个对象，再把名称/属性/下标 `output_size` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `output_size` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.135 direct_convolve：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `direct_convolve`；源码锚点 `offset = min(int(first.size), int(second.size)) - 1`。

```python
            offset = min(int(first.size), int(second.size)) - 1
        return kernel(
            first,
            second,
            np.int32(first.size),
            np.int32(second.size),
            np.int32(offset),
            size=output_size,
        )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `offset` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `size` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`offset`|相对起点的离散偏移|记作 $\Delta n$|参与索引换算或数据切片|
|`size`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `offset = min(int(first.size), int(second.size)) - 1` 中，它为 `offset` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `np.int32(first.size),` 中，它作为 `np.int32` 的实参参与当前调用；在 `np.int32(second.size),` 中，它作为 `np.int32` 的实参参与当前调用；在 `np.int32(offset),` 中，它作为 `np.int32` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `offset = min(int(first.size), int(second.size)) - 1`：这是 Python 赋值语句。解释器先完整计算右侧 `min(int(first.size), int(second.size)) - 1` 得到一个对象，再把名称/属性/下标 `offset` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `offset` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return kernel( first, second, np.int32(first.size), np.int32(second.size), np.int32(offset), size=output_size, )`：`return` 先计算 `kernel( first, second, np.int32(first.size), np.int32(second.size), np.int32(offset), size=output_size, )`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.136 direct_convolve：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `direct_convolve`；源码锚点 `return direct_convolve`。

```python
    return direct_convolve
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `direct_convolve`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `return direct_convolve`：`return` 先计算 `direct_convolve`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `direct_convolve` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `return direct_convolve` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.137 _install_waveform_kernels：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_waveform_kernels`；源码锚点 `def _install_waveform_kernels(cp: Any) -> None:`。

```python

def _install_waveform_kernels(cp: Any) -> None:
    module = importlib.import_module("cusignal.waveforms.waveforms")
    module._sawtooth_kernel = cp.ElementwiseKernel(
        "T t, T w",
        "float64 y",
        """
        double out {};
        const bool invalid { ((w > 1) || (w < 0)) };
        if (invalid) out = nan("0xfff8000000000000ULL");
        const T tmod = fmod(t, 2.0 * M_PI);
        const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };
        if (rising) out = tmod / (M_PI * w) - 1;
        if ((1 - invalid) && (1 - rising))
            out = (M_PI * (w + 1) - tmod) / (M_PI * (1 - w));
        y = out;
        """,
        "_task2_zq500_sawtooth",
        options=("-std=c++11",),
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `module` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `out` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `y` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `options` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_install_waveform_kernels` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0xfff8000000000000ULL` 是数值字面量；U 指定 unsigned int 起始类型，LL 指定 long long；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`module`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `module`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `module = importlib.import_module("cusignal.waveforms.waveforms")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`t`|当前函数或调用的参数名称，接收调用者绑定的输入 `t`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T w",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`w`|当前函数或调用的参数名称，接收调用者绑定的输入 `w`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T w",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`out`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `out`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `double out {};`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`invalid`|当前函数或调用的参数名称，接收调用者绑定的输入 `invalid`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const bool invalid { ((w > 1) || (w < 0)) };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`tmod`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `tmod`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T tmod = fmod(t, 2.0 * M_PI);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`rising`|当前函数或调用的参数名称，接收调用者绑定的输入 `rising`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`y`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `y`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"float64 y",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`options`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `options=("-std=c++11",),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`_install_waveform_kernels`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"float64 y",`，作用由同一表达式中的运算符决定。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `const bool invalid { ((w > 1) \|\| (w < 0)) };` 中，它是分支或边界比较值，决定哪条控制路径被执行；它直接参与表达式 `const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };`，作用由同一表达式中的运算符决定；在 `if (rising) out = tmod / (M_PI * w) - 1;` 中，它作为 `if` 的实参参与当前调用。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `const bool invalid { ((w > 1) \|\| (w < 0)) };` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `if (invalid) out = nan("0xfff8000000000000ULL");` 中，它作为 `if` 的实参参与当前调用；在 `const T tmod = fmod(t, 2.0 * M_PI);` 中，它为 `tmod` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0xfff8000000000000ULL`|`U` 使整数字面量从 unsigned int 候选类型开始|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `if (invalid) out = nan("0xfff8000000000000ULL");` 中，它作为 `if` 的实参参与当前调用。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `const T tmod = fmod(t, 2.0 * M_PI);` 中，它为 `tmod` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`11`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `options=("-std=c++11",),` 中，它为 `options` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def _install_waveform_kernels(cp: Any) -> None:`：`def` 创建名为 `_install_waveform_kernels` 的函数对象；括号中的 `cp: Any` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `module = importlib.import_module("cusignal.waveforms.waveforms")`：这是 Python 赋值语句。解释器先完整计算右侧 `importlib.import_module("cusignal.waveforms.waveforms")` 得到一个对象，再把名称/属性/下标 `module` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `module._sawtooth_kernel = cp.ElementwiseKernel( "T t, T w", "float64 y", """ double out {}; const bool invalid { ((w > 1) || (w < 0)) }; if (invalid) out = nan("0xfff80000000000...`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ElementwiseKernel( "T t, T w", "float64 y", """ double out {}; const bool invalid { ((w > 1) || (w < 0)) }; if (invalid) out = nan("0xfff8000000000000ULL"); const T tmod = fmod(t, 2.0 * M_PI); const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) }; if (rising) out = tmod / (M_PI * w) - 1; if ((1 - invalid) && (1 - rising)) out = (M_PI * (w + 1) - tmod) / (M_PI * (1 - w)); y = out; """, "_task2_zq500_sawtooth", options=("-std=c++11",), )` 得到一个对象，再把名称/属性/下标 `module._sawtooth_kernel` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module._sawtooth_kernel` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。 本块直接出现 1 种波形符号：`sawtooth`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.138 _install_waveform_kernels：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_waveform_kernels`；源码锚点 `module._square_kernel = cp.ElementwiseKernel(`。

```python
    module._square_kernel = cp.ElementwiseKernel(
        "T t, T w",
        "float64 y",
        """
        const bool invalid { ((w > 1) || (w < 0)) };
        if (invalid) y = nan("0xfff8000000000000ULL");
        const T tmod = fmod(t, 2.0 * M_PI);
        const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };
        if (high) y = 1;
        if ((1 - invalid) && (1 - high)) y = -1;
        """,
        "_task2_zq500_square",
        options=("-std=c++11",),
    )
```

**语法结构**

这个 Python 代码块由变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `options` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0xfff8000000000000ULL` 是数值字面量；U 指定 unsigned int 起始类型，LL 指定 long long；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`t`|当前函数或调用的参数名称，接收调用者绑定的输入 `t`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T w",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`w`|当前函数或调用的参数名称，接收调用者绑定的输入 `w`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T w",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`invalid`|当前函数或调用的参数名称，接收调用者绑定的输入 `invalid`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const bool invalid { ((w > 1) || (w < 0)) };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`tmod`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `tmod`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T tmod = fmod(t, 2.0 * M_PI);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`high`|当前函数或调用的参数名称，接收调用者绑定的输入 `high`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`options`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `options=("-std=c++11",),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"float64 y",`，作用由同一表达式中的运算符决定。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `const bool invalid { ((w > 1) \|\| (w < 0)) };` 中，它是分支或边界比较值，决定哪条控制路径被执行；它直接参与表达式 `const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };`，作用由同一表达式中的运算符决定；在 `if (high) y = 1;` 中，它作为 `if` 的实参参与当前调用。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `const bool invalid { ((w > 1) \|\| (w < 0)) };` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `if (invalid) y = nan("0xfff8000000000000ULL");` 中，它作为 `if` 的实参参与当前调用；在 `const T tmod = fmod(t, 2.0 * M_PI);` 中，它为 `tmod` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0xfff8000000000000ULL`|`U` 使整数字面量从 unsigned int 候选类型开始|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `if (invalid) y = nan("0xfff8000000000000ULL");` 中，它作为 `if` 的实参参与当前调用。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `const T tmod = fmod(t, 2.0 * M_PI);` 中，它为 `tmod` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`11`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `options=("-std=c++11",),` 中，它为 `options` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `module._square_kernel = cp.ElementwiseKernel( "T t, T w", "float64 y", """ const bool invalid { ((w > 1) || (w < 0)) }; if (invalid) y = nan("0xfff8000000000000ULL"); const T tm...`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ElementwiseKernel( "T t, T w", "float64 y", """ const bool invalid { ((w > 1) || (w < 0)) }; if (invalid) y = nan("0xfff8000000000000ULL"); const T tmod = fmod(t, 2.0 * M_PI); const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) }; if (high) y = 1; if ((1 - invalid) && (1 - high)) y = -1; """, "_task2_zq500_square", options=("-std=c++11",), )` 得到一个对象，再把名称/属性/下标 `module._square_kernel` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module._square_kernel` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.139 _install_waveform_kernels：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_waveform_kernels`；源码锚点 `chirp_source = """`。

```python
    chirp_source = """
        const T beta { (f1 - f0) / t1 };
        const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);
        phase = cos(temp + phi);
    """
    module._chirp_phase_lin_kernel_real = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "T phase",
        chirp_source,
        "_task2_zq500_chirp_linear_real",
        options=("-std=c++11",),
    )
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `chirp_source` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `phase` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `options` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `11` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`beta`|当前函数或调用的参数名称，接收调用者绑定的输入 `beta`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T beta { (f1 - f0) / t1 };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`temp`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `temp`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`t`|当前函数或调用的参数名称，接收调用者绑定的输入 `t`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`f0`|当前函数或调用的参数名称，接收调用者绑定的输入 `f0`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T beta { (f1 - f0) / t1 };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`t1`|当前函数或调用的参数名称，接收调用者绑定的输入 `t1`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T beta { (f1 - f0) / t1 };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`f1`|当前函数或调用的参数名称，接收调用者绑定的输入 `f1`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T beta { (f1 - f0) / t1 };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`phi`|当前函数或调用的参数名称，接收调用者绑定的输入 `phi`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `phase = cos(temp + phi);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`chirp_source`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|
|`options`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `options=("-std=c++11",),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 中，它为 `temp` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；它直接参与表达式 `"_task2_zq500_chirp_linear_real",`，作用由同一表达式中的运算符决定。|
|`0.5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。 当前上下文：在 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 中，它为 `temp` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `const T beta { (f1 - f0) / t1 };`，作用由同一表达式中的运算符决定；在 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 中，它为 `temp` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；它直接参与表达式 `"T t, T f0, T t1, T f1, T phi",`，作用由同一表达式中的运算符决定。|
|`11`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `options=("-std=c++11",),` 中，它为 `options` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `chirp_source = """`：这是 Python 赋值语句。解释器先完整计算右侧 `"""` 得到一个对象，再把名称/属性/下标 `chirp_source` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `chirp_source` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `const T beta { (f1 - f0) / t1 };`：关键字语法：`const`：const 禁止通过当前名字修改对象。 运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`-`：减法；位于单个操作数前时是一元负号。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);`：这是 Python 赋值语句。解释器先完整计算右侧 `2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 得到一个对象，再把名称/属性/下标 `const T temp` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `const T temp` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `phase = cos(temp + phi);`：这是 Python 赋值语句。解释器先完整计算右侧 `cos(temp + phi);` 得到一个对象，再把名称/属性/下标 `phase` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `phase` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `"""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。
- 对源码锚点 `module._chirp_phase_lin_kernel_real = cp.ElementwiseKernel( "T t, T f0, T t1, T f1, T phi", "T phase", chirp_source, "_task2_zq500_chirp_linear_real", options=("-std=c++11",), )`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ElementwiseKernel( "T t, T f0, T t1, T f1, T phi", "T phase", chirp_source, "_task2_zq500_chirp_linear_real", options=("-std=c++11",), )` 得到一个对象，再把名称/属性/下标 `module._chirp_phase_lin_kernel_real` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module._chirp_phase_lin_kernel_real` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.140 _install_waveform_kernels：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_waveform_kernels`；源码锚点 `module._chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(`。

```python
    module._chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "Y phase",
        """
        const T beta { (f1 - f0) / t1 };
        const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);
        phase = Y(cos(temp + phi), cos(temp + phi + M_PI / 2) * -1);
        """,
        "_task2_zq500_chirp_linear_complex",
        options=("-std=c++11",),
    )
```

**语法结构**

这个 Python 代码块由变量声明与初始化、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `phase` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `options` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `11` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`t`|当前函数或调用的参数名称，接收调用者绑定的输入 `t`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T f0, T t1, T f1, T phi",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`f0`|当前函数或调用的参数名称，接收调用者绑定的输入 `f0`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T f0, T t1, T f1, T phi",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`t1`|当前函数或调用的参数名称，接收调用者绑定的输入 `t1`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T f0, T t1, T f1, T phi",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`f1`|当前函数或调用的参数名称，接收调用者绑定的输入 `f1`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T f0, T t1, T f1, T phi",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`phi`|当前函数或调用的参数名称，接收调用者绑定的输入 `phi`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"T t, T f0, T t1, T f1, T phi",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`beta`|当前函数或调用的参数名称，接收调用者绑定的输入 `beta`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T beta { (f1 - f0) / t1 };`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`temp`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `temp`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`options`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `options=("-std=c++11",),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 中，它为 `temp` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `phase = Y(cos(temp + phi), cos(temp + phi + M_PI / 2) * -1);` 中，它为 `phase` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；它直接参与表达式 `"_task2_zq500_chirp_linear_complex",`，作用由同一表达式中的运算符决定。|
|`0.5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。 当前上下文：在 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 中，它为 `temp` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `"T t, T f0, T t1, T f1, T phi",`，作用由同一表达式中的运算符决定；它直接参与表达式 `const T beta { (f1 - f0) / t1 };`，作用由同一表达式中的运算符决定；在 `phase = Y(cos(temp + phi), cos(temp + phi + M_PI / 2) * -1);` 中，它为 `phase` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"T t, T f0, T t1, T f1, T phi",`，作用由同一表达式中的运算符决定；它直接参与表达式 `const T beta { (f1 - f0) / t1 };`，作用由同一表达式中的运算符决定；在 `const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);` 中，它为 `temp` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`11`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `options=("-std=c++11",),` 中，它为 `options` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `module._chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel( "T t, T f0, T t1, T f1, T phi", "Y phase", """ const T beta { (f1 - f0) / t1 }; const T temp = 2 * M_PI * (f0 * t + 0...`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ElementwiseKernel( "T t, T f0, T t1, T f1, T phi", "Y phase", """ const T beta { (f1 - f0) / t1 }; const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t); phase = Y(cos(temp + phi), cos(temp + phi + M_PI / 2) * -1); """, "_task2_zq500_chirp_linear_complex", options=("-std=c++11",), )` 得到一个对象，再把名称/属性/下标 `module._chirp_phase_lin_kernel_cplx` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module._chirp_phase_lin_kernel_cplx` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.141 _install_peak_helper：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_peak_helper`；源码锚点 `def _install_peak_helper(cp: Any) -> None:`。

```python

def _install_peak_helper(cp: Any) -> None:
    module = importlib.import_module("cusignal.peak_finding.peak_finding")
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `module` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_install_peak_helper` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`module`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `module`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `module = importlib.import_module("cusignal.peak_finding.peak_finding")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`_install_peak_helper`|自定义函数/可调用入口 `_install_peak_helper`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `_install_peak_helper` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def _install_peak_helper(cp: Any) -> None:`：`def` 创建名为 `_install_peak_helper` 的函数对象；括号中的 `cp: Any` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `module = importlib.import_module("cusignal.peak_finding.peak_finding")`：这是 Python 赋值语句。解释器先完整计算右侧 `importlib.import_module("cusignal.peak_finding.peak_finding")` 得到一个对象，再把名称/属性/下标 `module` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.142 boolrelextrema：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `boolrelextrema`；源码锚点 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:`。

```python
    def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:
        if int(order) != order or order < 1:
            raise ValueError("Order must be an int >= 1")
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `boolrelextrema` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if int(order) != order or order < 1:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`must`|当前函数或调用的参数名称，接收调用者绑定的输入 `must`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raise ValueError("Order must be an int >= 1")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|
|`comparator`|当前函数或调用的参数名称，接收调用者绑定的输入 `comparator`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`axis`|当前函数或调用的参数名称，接收调用者绑定的输入 `axis`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`order`|当前函数或调用的参数名称，接收调用者绑定的输入 `order`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`mode`|当前函数或调用的参数名称，接收调用者绑定的输入 `mode`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`boolrelextrema`|自定义函数/可调用入口 `boolrelextrema`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `boolrelextrema` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:` 中，它作为 `boolrelextrema` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:` 中，它作为 `boolrelextrema` 的实参参与当前调用；在 `if int(order) != order or order < 1:` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `raise ValueError("Order must be an int >= 1")` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:`：`def` 创建名为 `boolrelextrema` 的函数对象；括号中的 `data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip"` 是形参表，调用时实参按位置或关键字绑定。`-> Any` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if int(order) != order or order < 1:`：`if` 先计算条件 `int(order) != order or order < 1` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise ValueError("Order must be an int >= 1")`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`ValueError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"Order must be an int >= 1"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `boolrelextrema` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.143 boolrelextrema：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `boolrelextrema`；源码锚点 `length = data.shape[axis]`。

```python
        length = data.shape[axis]
        locations = cp.arange(length)
        result = cp.ones(data.shape, dtype=bool)
        main = cp.take(data, locations, axis=axis)
        for shift in range(1, order + 1):
            if mode == "clip":
                plus_locations = cp.clip(locations + shift, None, length - 1)
                minus_locations = cp.clip(locations - shift, 0, None)
            elif mode == "wrap":
                plus_locations = (locations + shift) % length
                minus_locations = (locations - shift) % length
            else:
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `length` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `locations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `result` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `main` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `plus_locations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `minus_locations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `plus_locations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `minus_locations` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`length`|数组、窗口或缓冲区长度|记作 $N$ 或 $L$|用于 shape 校验、分配和边界判断|
|`locations`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `locations`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `locations = cp.arange(length)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`main`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `main`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `main = cp.take(data, locations, axis=axis)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`plus_locations`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `plus_locations`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `plus_locations = cp.clip(locations + shift, None, length - 1)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`minus_locations`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `minus_locations`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `minus_locations = cp.clip(locations - shift, 0, None)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `for shift in range(1, order + 1):` 中，它参与迭代起点、终点或步长，直接决定循环执行次数；在 `plus_locations = cp.clip(locations + shift, None, length - 1)` 中，它为 `plus_locations` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `minus_locations = cp.clip(locations - shift, 0, None)` 中，它为 `minus_locations` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `length = data.shape[axis]`：这是 Python 赋值语句。解释器先完整计算右侧 `data.shape[axis]` 得到一个对象，再把名称/属性/下标 `length` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `length` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `locations = cp.arange(length)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.arange(length)` 得到一个对象，再把名称/属性/下标 `locations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `locations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `result = cp.ones(data.shape, dtype=bool)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones(data.shape, dtype=bool)` 得到一个对象，再把名称/属性/下标 `result` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `result` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `main = cp.take(data, locations, axis=axis)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.take(data, locations, axis=axis)` 得到一个对象，再把名称/属性/下标 `main` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `main` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `for shift in range(1, order + 1):`：关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 函数调用语法：`range(...)` 先准备括号内实参，再把控制权交给 `range`，返回值回到调用点。 字面量与类型：`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`+`：加法；也可能是一元正号；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `if mode == "clip":`：`if` 先计算条件 `mode == "clip"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `plus_locations = cp.clip(locations + shift, None, length - 1)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.clip(locations + shift, None, length - 1)` 得到一个对象，再把名称/属性/下标 `plus_locations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `plus_locations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `minus_locations = cp.clip(locations - shift, 0, None)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.clip(locations - shift, 0, None)` 得到一个对象，再把名称/属性/下标 `minus_locations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `minus_locations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `elif mode == "wrap":`：运算符：`==`：相等比较，结果类型为 bool；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `plus_locations = (locations + shift) % length`：这是 Python 赋值语句。解释器先完整计算右侧 `(locations + shift) % length` 得到一个对象，再把名称/属性/下标 `plus_locations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `plus_locations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `minus_locations = (locations - shift) % length`：这是 Python 赋值语句。解释器先完整计算右侧 `(locations - shift) % length` 得到一个对象，再把名称/属性/下标 `minus_locations` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `minus_locations` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `else:`：关键字语法：`else`：else 只在前一 if/else if 未进入时执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `boolrelextrema` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.144 NotImplementedError：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `NotImplementedError`；源码锚点 `raise NotImplementedError("CuPy \`take\` doesn't support mode='raise'.")`。

```python
                raise NotImplementedError("CuPy `take` doesn't support mode='raise'.")
            result &= comparator(main, cp.take(data, plus_locations, axis=axis))
            result &= comparator(main, cp.take(data, minus_locations, axis=axis))
        return result
```

**语法结构**

这个 Python 代码块由函数或方法定义、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `mode` 由 `support` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`mode`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `mode`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raise NotImplementedError("CuPy `take` doesn't support mode='raise'.")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `raise NotImplementedError("CuPy \`take\` doesn't support mode='raise'.")`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`NotImplementedError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"CuPy `take` doesn't support mode='raise'."` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `result &= comparator(main, cp.take(data, plus_locations, axis=axis))`：这是 Python 赋值语句。解释器先完整计算右侧 `comparator(main, cp.take(data, plus_locations, axis=axis))` 得到一个对象，再把名称/属性/下标 `result &` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `result &` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `result &= comparator(main, cp.take(data, minus_locations, axis=axis))`：这是 Python 赋值语句。解释器先完整计算右侧 `comparator(main, cp.take(data, minus_locations, axis=axis))` 得到一个对象，再把名称/属性/下标 `result &` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `result &` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `return result`：`return` 先计算 `result`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `NotImplementedError` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.145 NotImplementedError：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `NotImplementedError`；源码锚点 `module._boolrelextrema = boolrelextrema`。

```python

    module._boolrelextrema = boolrelextrema
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `module`、`_boolrelextrema`、`boolrelextrema`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `module._boolrelextrema = boolrelextrema`：这是 Python 赋值语句。解释器先完整计算右侧 `boolrelextrema` 得到一个对象，再把名称/属性/下标 `module._boolrelextrema` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `module._boolrelextrema` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `NotImplementedError` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `module._boolrelextrema = boolrelextrema` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.146 _install_scalar_kalman：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `_install_scalar_kalman`；源码锚点 `def _install_scalar_kalman(cp: Any, cusignal: Any) -> None:`。

```python

def _install_scalar_kalman(cp: Any, cusignal: Any) -> None:
    klass = cusignal.KalmanFilter
    original_init = klass.__init__
    original_predict = klass.predict
    original_update = klass.update
    sequence_kernel = cp.ElementwiseKernel(
        "raw float32 observations, int32 observation_count, raw float32 initial_x, "
        "raw float32 initial_p, raw float32 f, raw float32 q, raw float32 alpha_sq, "
        "raw float32 h, raw float32 r",
        "float32 next_x, float32 next_p",
        """
        float state = initial_x[0];
        float covariance = initial_p[0];
        const float transition = f[0];
        const float process_noise = q[0];
        const float alpha = alpha_sq[0];
        const float measurement = h[0];
        const float measurement_noise = r[0];
        for (int index = 0; index < observation_count; ++index) {
            state = transition * state;
            covariance = alpha * transition * covariance * transition + process_noise;
            const float innovation = observations[index] - measurement * state;
            const float pht = covariance * measurement;
            const float gain = pht / (measurement * pht + measurement_noise);
            state += gain * innovation;
            const float i_kh = 1.0F - gain * measurement;
            covariance = i_kh * covariance * i_kh + gain * measurement_noise * gain;
        }
        next_x = state;
        next_p = covariance;
        """,
        "_task2_zq500_kalman_scalar_sequence",
        options=("-std=c++11",),
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、循环、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `klass` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `original_init` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `original_predict` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `original_update` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `sequence_kernel` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `state` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `covariance` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `covariance` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `next_x` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `next_p` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `options` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `_install_scalar_kalman` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`klass`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `klass`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `klass = cusignal.KalmanFilter`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`original_init`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `original_init`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `original_init = klass.__init__`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`covariance`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `covariance`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `float covariance = initial_p[0];`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`transition`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `transition`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float transition = f[0];`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`process_noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`alpha`|Kalman API 的标量辅助系数|对应本仓库调用的 `alpha` 参数；当前任务代码未在本层重新推导其公式|由 wrapper 传入正式算子；具体定义应与 KalmanFilter 算子文档核对|
|`measurement`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `measurement`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float measurement = h[0];`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`measurement_noise`|测量噪声协方差参数|记作 $R$|用于扰动测试或 update 权重|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`innovation`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `innovation`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float innovation = observations[index] - measurement * state;`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`pht`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `pht`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float pht = covariance * measurement;`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`gain`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `gain`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float gain = pht / (measurement * pht + measurement_noise);`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`i_kh`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `i_kh`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float i_kh = 1.0F - gain * measurement;`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`original_predict`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `original_predict`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `original_predict = klass.predict`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`original_update`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `original_update`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `original_update = klass.update`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`sequence_kernel`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `sequence_kernel`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `sequence_kernel = cp.ElementwiseKernel(`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`next_x`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `next_x`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"float32 next_x, float32 next_p",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`next_p`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `next_p`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"float32 next_x, float32 next_p",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`options`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `options=("-std=c++11",),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`_install_scalar_kalman`|自定义函数/可调用入口 `_install_scalar_kalman`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `_install_scalar_kalman` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：它直接参与表达式 `"raw float32 observations, int32 observation_count, raw float32 initial_x, "`，作用由同一表达式中的运算符决定；它直接参与表达式 `"raw float32 initial_p, raw float32 f, raw float32 q, raw float32 alpha_sq, "`，作用由同一表达式中的运算符决定；它直接参与表达式 `"raw float32 h, raw float32 r",`，作用由同一表达式中的运算符决定。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `float state = initial_x[0];` 中，它为 `state` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `float covariance = initial_p[0];` 中，它为 `covariance` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `const float transition = f[0];` 中，它为 `transition` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `const float i_kh = 1.0F - gain * measurement;` 中，它为 `i_kh` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`11`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `options=("-std=c++11",),` 中，它为 `options` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def _install_scalar_kalman(cp: Any, cusignal: Any) -> None:`：`def` 创建名为 `_install_scalar_kalman` 的函数对象；括号中的 `cp: Any, cusignal: Any` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `klass = cusignal.KalmanFilter`：这是 Python 赋值语句。解释器先完整计算右侧 `cusignal.KalmanFilter` 得到一个对象，再把名称/属性/下标 `klass` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `klass` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `original_init = klass.__init__`：这是 Python 赋值语句。解释器先完整计算右侧 `klass.__init__` 得到一个对象，再把名称/属性/下标 `original_init` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `original_init` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `original_predict = klass.predict`：这是 Python 赋值语句。解释器先完整计算右侧 `klass.predict` 得到一个对象，再把名称/属性/下标 `original_predict` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `original_predict` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `original_update = klass.update`：这是 Python 赋值语句。解释器先完整计算右侧 `klass.update` 得到一个对象，再把名称/属性/下标 `original_update` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `original_update` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `sequence_kernel = cp.ElementwiseKernel( "raw float32 observations, int32 observation_count, raw float32 initial_x, " "raw float32 initial_p, raw float32 f, raw float32 q, raw fl...`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ElementwiseKernel( "raw float32 observations, int32 observation_count, raw float32 initial_x, " "raw float32 initial_p, raw float32 f, raw float32 q, raw float32 alpha_sq, " "raw float32 h, raw float32 r", "float32 next_x, float32 next_p", """ float state = initial_x[0]; float covariance = initial_p[0]; const float transition = f[0]; const float process_noise = q[0]; const float alpha = alpha_sq[0]; const float measurement = h[0]; const float measurement_noise = r[0]; for (int index = 0; index < observation_count; ++index) { state = transition * state; covariance = alpha * transition * covariance * transition + process_noise; const float innovation = observations[index] - measurement * state; const float pht = covariance * measurement; const float gain = pht / (measurement * pht + measurement_noise); state += gain * innovation; const float i_kh = 1.0F - gain * measurement; covariance = i_kh * covariance * i_kh + gain * measurement_noise * gain; } next_x = state; next_p = covariance; """, "_task2_zq500_kalman_scalar_sequence", options=("-std=c++11",), )` 得到一个对象，再把名称/属性/下标 `sequence_kernel` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `sequence_kernel` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`、`covariance`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.147 init：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `init`；源码锚点 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`。

```python

    def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:
        dtype = cp.float32 if dtype is None else dtype
        if dim_x != 1 or dim_z != 1 or dim_u != 0:
            original_init(self, dim_x, dim_z, dim_u, points, dtype)
            return
        self.points = points
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        shape = (points, 1, 1)
        self.x = cp.zeros(shape, dtype=dtype)
```

**语法结构**

这个 Python 代码块由函数或方法定义、变量声明与初始化、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `dtype` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `shape` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `init` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`dtype`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `dtype`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`else`|当前表达式读取或传递的工程名称 `else`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `dtype = cp.float32 if dtype is None else dtype`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`shape`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `shape`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `shape = (points, 1, 1)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`dim_x`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `dim_x`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`dim_z`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`dim_u`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `dim_u`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`points`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `points`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`init`|自定义函数/可调用入口 `init`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `init` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:` 中，它作为 `init` 的实参参与当前调用；在 `if dim_x != 1 or dim_z != 1 or dim_u != 0:` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:` 中，它作为 `init` 的实参参与当前调用；在 `if dim_x != 1 or dim_z != 1 or dim_u != 0:` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `shape = (points, 1, 1)` 中，它为 `shape` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `dtype = cp.float32 if dtype is None else dtype` 中，它为 `dtype` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:`：`def` 创建名为 `init` 的函数对象；括号中的 `self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `dtype = cp.float32 if dtype is None else dtype`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.float32 if dtype is None else dtype` 得到一个对象，再把名称/属性/下标 `dtype` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `dtype` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if dim_x != 1 or dim_z != 1 or dim_u != 0:`：`if` 先计算条件 `dim_x != 1 or dim_z != 1 or dim_u != 0` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `original_init(self, dim_x, dim_z, dim_u, points, dtype)`：`original_init(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`self` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`dim_x` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`dim_z` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`dim_u` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`points` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`dtype` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `self.points = points`：这是 Python 赋值语句。解释器先完整计算右侧 `points` 得到一个对象，再把名称/属性/下标 `self.points` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.points` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.dim_x = dim_x`：这是 Python 赋值语句。解释器先完整计算右侧 `dim_x` 得到一个对象，再把名称/属性/下标 `self.dim_x` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.dim_x` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.dim_z = dim_z`：这是 Python 赋值语句。解释器先完整计算右侧 `dim_z` 得到一个对象，再把名称/属性/下标 `self.dim_z` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.dim_z` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.dim_u = dim_u`：这是 Python 赋值语句。解释器先完整计算右侧 `dim_u` 得到一个对象，再把名称/属性/下标 `self.dim_u` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.dim_u` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `shape = (points, 1, 1)`：这是 Python 赋值语句。解释器先完整计算右侧 `(points, 1, 1)` 得到一个对象，再把名称/属性/下标 `shape` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `shape` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.x = cp.zeros(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.zeros(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.x` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.x` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `init` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.148 init：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `init`；源码锚点 `self.P = cp.ones(shape, dtype=dtype)`。

```python
        self.P = cp.ones(shape, dtype=dtype)
        self.Q = cp.ones(shape, dtype=dtype)
        self.B = None
        self.F = cp.ones(shape, dtype=dtype)
        self.H = cp.zeros(shape, dtype=dtype)
        self.R = cp.ones(shape, dtype=dtype)
        self._alpha_sq = cp.ones(shape, dtype=dtype)
        self.z = cp.empty(shape, dtype=dtype)
        self._task2_zq500_scalar_backend = True
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `self`、`P`、`cp`、`ones`、`shape`、`dtype`、`Q`、`B`、`None`、`F`、`H`、`zeros`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`self`|当前表达式读取或传递的工程名称 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `self.P = cp.ones(shape, dtype=dtype)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `self._task2_zq500_scalar_backend = True`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `self.P = cp.ones(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.P` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.P` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.Q = cp.ones(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.Q` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.Q` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.B = None`：这是 Python 赋值语句。解释器先完整计算右侧 `None` 得到一个对象，再把名称/属性/下标 `self.B` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.B` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.F = cp.ones(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.F` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.F` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.H = cp.zeros(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.zeros(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.H` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.H` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.R = cp.ones(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.R` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.R` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self._alpha_sq = cp.ones(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self._alpha_sq` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self._alpha_sq` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.z = cp.empty(shape, dtype=dtype)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.empty(shape, dtype=dtype)` 得到一个对象，再把名称/属性/下标 `self.z` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.z` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self._task2_zq500_scalar_backend = True`：这是 Python 赋值语句。解释器先完整计算右侧 `True` 得到一个对象，再把名称/属性/下标 `self._task2_zq500_scalar_backend` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self._task2_zq500_scalar_backend` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `init` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.149 predict：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict`；源码锚点 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`。

```python
    def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            original_predict(self, u, B, F, Q)
            return
        if u is not None:
            raise NotImplementedError("Control Matrix implementation in process")
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `predict` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`Any`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `Any`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if not getattr(self, "_task2_zq500_scalar_backend", False):`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`raise`|当前表达式读取或传递的工程名称 `raise`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raise NotImplementedError("Control Matrix implementation in process")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`Matrix`|当前函数或调用的参数名称，接收调用者绑定的输入 `Matrix`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `raise NotImplementedError("Control Matrix implementation in process")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`u`|当前函数或调用的参数名称，接收调用者绑定的输入 `u`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`B`|当前函数或调用的参数名称，接收调用者绑定的输入 `B`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`F`|当前函数或调用的参数名称，接收调用者绑定的输入 `F`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`Q`|当前函数或调用的参数名称，接收调用者绑定的输入 `Q`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`predict`|自定义函数/可调用入口 `predict`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `predict` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `if not getattr(self, "_task2_zq500_scalar_backend", False):` 中，它作为 `getattr` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:`：`def` 创建名为 `predict` 的函数对象；括号中的 `self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if not getattr(self, "_task2_zq500_scalar_backend", False):`：`if` 先计算条件 `not getattr(self, "_task2_zq500_scalar_backend", False)` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `original_predict(self, u, B, F, Q)`：`original_predict(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`self` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`u` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`B` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`F` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Q` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `if u is not None:`：`if` 先计算条件 `u is not None` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise NotImplementedError("Control Matrix implementation in process")`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`NotImplementedError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"Control Matrix implementation in process"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.150 predict：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict`；源码锚点 `del B`。

```python
        del B
        F = self.F if F is None else cp.asarray(F)
        if Q is None:
            Q = self.Q
        elif cp.isscalar(Q):
            Q = cp.ones_like(self.Q) * Q
        else:
            Q = cp.asarray(Q)
        self.x[...] = F * self.x
        self.P[...] = self._alpha_sq * F * self.P * F + Q
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `F` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `Q` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `Q` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `else` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`F`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `F`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `F = self.F if F is None else cp.asarray(F)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `F = self.F if F is None else cp.asarray(F)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`is`|当前表达式读取或传递的工程名称 `is`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `F = self.F if F is None else cp.asarray(F)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`else`|当前表达式读取或传递的工程名称 `else`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `F = self.F if F is None else cp.asarray(F)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`Q`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `Q`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if Q is None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`elif`|当前表达式读取或传递的工程名称 `elif`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `elif cp.isscalar(Q):`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `F = self.F if F is None else cp.asarray(F)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `del B`：名称解析：`del`、`B`按词法作用域查找对应变量、类型、函数或常量；逗号把当前项目与同一外层调用/容器中的下一项目分开，分号仅在 C/C++ 中结束完整语句。
- 对源码锚点 `F = self.F if F is None else cp.asarray(F)`：这是 Python 赋值语句。解释器先完整计算右侧 `self.F if F is None else cp.asarray(F)` 得到一个对象，再把名称/属性/下标 `F` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `F` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if Q is None:`：`if` 先计算条件 `Q is None` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `Q = self.Q`：这是 Python 赋值语句。解释器先完整计算右侧 `self.Q` 得到一个对象，再把名称/属性/下标 `Q` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `Q` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `elif cp.isscalar(Q):`：函数调用语法：`cp.isscalar(...)` 先准备括号内实参，再把控制权交给 `cp.isscalar`，返回值回到调用点。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `Q = cp.ones_like(self.Q) * Q`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones_like(self.Q) * Q` 得到一个对象，再把名称/属性/下标 `Q` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `Q` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `else:`：关键字语法：`else`：else 只在前一 if/else if 未进入时执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。
- 对源码锚点 `Q = cp.asarray(Q)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(Q)` 得到一个对象，再把名称/属性/下标 `Q` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `Q` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.x[...] = F * self.x`：这是 Python 赋值语句。解释器先完整计算右侧 `F * self.x` 得到一个对象，再把名称/属性/下标 `self.x[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.x[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.P[...] = self._alpha_sq * F * self.P * F + Q`：这是 Python 赋值语句。解释器先完整计算右侧 `self._alpha_sq * F * self.P * F + Q` 得到一个对象，再把名称/属性/下标 `self.P[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.P[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.151 update：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `update`；源码锚点 `def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:`。

```python
    def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            original_update(self, z, R, H)
            return
        if z is None:
            return
        H = self.H if H is None else cp.asarray(H)
        if R is None:
            R = self.R
        elif cp.isscalar(R):
            R = cp.ones_like(self.R) * R
        else:
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `H` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `R` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `R` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `update` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`Any`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `Any`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if not getattr(self, "_task2_zq500_scalar_backend", False):`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`return`|当前表达式读取或传递的工程名称 `return`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `return`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`is`|当前表达式读取或传递的工程名称 `is`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if z is None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`else`|当前表达式读取或传递的工程名称 `else`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `H = self.H if H is None else cp.asarray(H)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`R`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `R`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`elif`|当前表达式读取或传递的工程名称 `elif`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `elif cp.isscalar(R):`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`H`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `H`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`z`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`update`|自定义函数/可调用入口 `update`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `update` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `if not getattr(self, "_task2_zq500_scalar_backend", False):` 中，它作为 `getattr` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:`：`def` 创建名为 `update` 的函数对象；括号中的 `self: Any, z: Any, R: Any = None, H: Any = None` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if not getattr(self, "_task2_zq500_scalar_backend", False):`：`if` 先计算条件 `not getattr(self, "_task2_zq500_scalar_backend", False)` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `original_update(self, z, R, H)`：`original_update(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`self` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`z` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`R` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`H` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `if z is None:`：`if` 先计算条件 `z is None` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `H = self.H if H is None else cp.asarray(H)`：这是 Python 赋值语句。解释器先完整计算右侧 `self.H if H is None else cp.asarray(H)` 得到一个对象，再把名称/属性/下标 `H` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `H` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if R is None:`：`if` 先计算条件 `R is None` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `R = self.R`：这是 Python 赋值语句。解释器先完整计算右侧 `self.R` 得到一个对象，再把名称/属性/下标 `R` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `R` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `elif cp.isscalar(R):`：函数调用语法：`cp.isscalar(...)` 先准备括号内实参，再把控制权交给 `cp.isscalar`，返回值回到调用点。 运算符：`.`：通过对象本身访问成员；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `R = cp.ones_like(self.R) * R`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones_like(self.R) * R` 得到一个对象，再把名称/属性/下标 `R` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `R` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `else:`：关键字语法：`else`：else 只在前一 if/else if 未进入时执行。 运算符：`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.152 update：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `update`；源码锚点 `R = cp.asarray(R)`。

```python
            R = cp.asarray(R)
        self.z[...] = cp.asarray(z)
        residual = self.z - H * self.x
        innovation = H * self.P * H + R
        gain = self.P * H / innovation
        identity_minus_gain_h = cp.ones_like(self.P) - gain * H
        self.x[...] = self.x + gain * residual
        self.P[...] = (
            identity_minus_gain_h * self.P * identity_minus_gain_h
            + gain * R * gain
        )
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `R` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `residual` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `innovation` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `gain` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `identity_minus_gain_h` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `self.z[...] = cp.asarray(z)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`H`|当前表达式读取或传递的工程名称 `H`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `residual = self.z - H * self.x`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`gain`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `gain`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `gain = self.P * H / innovation`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`identity_minus_gain_h`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `identity_minus_gain_h`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `identity_minus_gain_h = cp.ones_like(self.P) - gain * H`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`R`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `R`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `R = cp.asarray(R)`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`residual`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `residual`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `residual = self.z - H * self.x`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`innovation`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `innovation`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `innovation = H * self.P * H + R`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `R = cp.asarray(R)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(R)` 得到一个对象，再把名称/属性/下标 `R` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `R` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.z[...] = cp.asarray(z)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(z)` 得到一个对象，再把名称/属性/下标 `self.z[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.z[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `residual = self.z - H * self.x`：这是 Python 赋值语句。解释器先完整计算右侧 `self.z - H * self.x` 得到一个对象，再把名称/属性/下标 `residual` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `residual` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `innovation = H * self.P * H + R`：这是 Python 赋值语句。解释器先完整计算右侧 `H * self.P * H + R` 得到一个对象，再把名称/属性/下标 `innovation` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `innovation` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `gain = self.P * H / innovation`：这是 Python 赋值语句。解释器先完整计算右侧 `self.P * H / innovation` 得到一个对象，再把名称/属性/下标 `gain` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `gain` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `identity_minus_gain_h = cp.ones_like(self.P) - gain * H`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.ones_like(self.P) - gain * H` 得到一个对象，再把名称/属性/下标 `identity_minus_gain_h` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `identity_minus_gain_h` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.x[...] = self.x + gain * residual`：这是 Python 赋值语句。解释器先完整计算右侧 `self.x + gain * residual` 得到一个对象，再把名称/属性/下标 `self.x[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.x[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.P[...] = ( identity_minus_gain_h * self.P * identity_minus_gain_h + gain * R * gain )`：这是 Python 赋值语句。解释器先完整计算右侧 `( identity_minus_gain_h * self.P * identity_minus_gain_h + gain * R * gain )` 得到一个对象，再把名称/属性/下标 `self.P[...]` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.P[...]` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `update` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.153 predict_update_sequence：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update_sequence`；源码锚点 `def predict_update_sequence(self: Any, observations: Any) -> None:`。

```python
    def predict_update_sequence(self: Any, observations: Any) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            raise NotImplementedError(
                "predict_update_sequence is only defined for the approved scalar Kalman layout")
        values = cp.asarray(observations, dtype=cp.float32).reshape(-1)
        if values.size == 0:
            return
        next_x, next_p = sequence_kernel(
            values,
            np.int32(values.size),
            self.x,
            self.P,
            self.F,
            self.Q,
            self._alpha_sq,
            self.H,
            self.R,
            size=1,
        )
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、循环、异常处理、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `values` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `size` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `predict_update_sequence` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`if`|当前表达式读取或传递的工程名称 `if`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `if not getattr(self, "_task2_zq500_scalar_backend", False):`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`layout`|数组维度、步长或算子状态布局描述|无独立数学符号|保证 Host/GPU/算子对 shape 的解释一致|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`size`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`self`|当前函数或调用的参数名称，接收调用者绑定的输入 `self`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `def predict_update_sequence(self: Any, observations: Any) -> None:`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`observations`|按特征点截取的观测序列|记作 $z_1,\ldots,z_K$|由 Step5 特征点附近数据构造，逐项送入 update|
|`predict_update_sequence`|自定义函数/可调用入口 `predict_update_sequence`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `predict_update_sequence` 所表达的步骤职责|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `if not getattr(self, "_task2_zq500_scalar_backend", False):` 中，它作为 `getattr` 的实参参与当前调用；在 `if values.size == 0:` 中，它为 `values.size` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `if not getattr(self, "_task2_zq500_scalar_backend", False):` 中，它作为 `getattr` 的实参参与当前调用；在 `values = cp.asarray(observations, dtype=cp.float32).reshape(-1)` 中，它为 `values` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `np.int32(values.size),` 中，它作为 `np.int32` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `values = cp.asarray(observations, dtype=cp.float32).reshape(-1)` 中，它为 `values` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `size=1,` 中，它为 `size` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `def predict_update_sequence(self: Any, observations: Any) -> None:`：`def` 创建名为 `predict_update_sequence` 的函数对象；括号中的 `self: Any, observations: Any` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `if not getattr(self, "_task2_zq500_scalar_backend", False):`：`if` 先计算条件 `not getattr(self, "_task2_zq500_scalar_backend", False)` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `raise NotImplementedError( "predict_update_sequence is only defined for the approved scalar Kalman layout")`：`raise` 会把创建出的异常对象抛出并中断当前正常流程；`NotImplementedError(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"predict_update_sequence is only defined for the approved scalar Kalman layout"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `values = cp.asarray(observations, dtype=cp.float32).reshape(-1)`：这是 Python 赋值语句。解释器先完整计算右侧 `cp.asarray(observations, dtype=cp.float32).reshape(-1)` 得到一个对象，再把名称/属性/下标 `values` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `values` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `if values.size == 0:`：`if` 先计算条件 `values.size == 0` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `next_x, next_p = sequence_kernel( values, np.int32(values.size), self.x, self.P, self.F, self.Q, self._alpha_sq, self.H, self.R, size=1, )`：这是 Python 赋值语句。解释器先完整计算右侧 `sequence_kernel( values, np.int32(values.size), self.x, self.P, self.F, self.Q, self._alpha_sq, self.H, self.R, size=1, )` 得到一个对象，再把名称/属性/下标 `next_x, next_p` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `next_x, next_p` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 `kalman`。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.154 predict_update_sequence：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update_sequence`；源码锚点 `self.x = next_x.reshape((1, 1, 1))`。

```python
        self.x = next_x.reshape((1, 1, 1))
        self.P = next_p.reshape((1, 1, 1))
        self.z = values[-1:].reshape((1, 1, 1))
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `self.x = next_x.reshape((1, 1, 1))` 中，它为 `self.x` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `self.P = next_p.reshape((1, 1, 1))` 中，它为 `self.P` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射；在 `self.z = values[-1:].reshape((1, 1, 1))` 中，它为 `self.z` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `self.x = next_x.reshape((1, 1, 1))`：这是 Python 赋值语句。解释器先完整计算右侧 `next_x.reshape((1, 1, 1))` 得到一个对象，再把名称/属性/下标 `self.x` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.x` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.P = next_p.reshape((1, 1, 1))`：这是 Python 赋值语句。解释器先完整计算右侧 `next_p.reshape((1, 1, 1))` 得到一个对象，再把名称/属性/下标 `self.P` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.P` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `self.z = values[-1:].reshape((1, 1, 1))`：这是 Python 赋值语句。解释器先完整计算右侧 `values[-1:].reshape((1, 1, 1))` 得到一个对象，再把名称/属性/下标 `self.z` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `self.z` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update_sequence` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.155 predict_update_sequence：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `predict_update_sequence`；源码锚点 `klass.__init__ = init`。

```python
    klass.__init__ = init
    klass.predict = predict
    klass.update = update
    klass.predict_update_sequence = predict_update_sequence
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `klass`、`__init__`、`init`、`predict`、`update`、`predict_update_sequence`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `klass.__init__ = init`：这是 Python 赋值语句。解释器先完整计算右侧 `init` 得到一个对象，再把名称/属性/下标 `klass.__init__` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `klass.__init__` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `klass.predict = predict`：这是 Python 赋值语句。解释器先完整计算右侧 `predict` 得到一个对象，再把名称/属性/下标 `klass.predict` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `klass.predict` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `klass.update = update`：这是 Python 赋值语句。解释器先完整计算右侧 `update` 得到一个对象，再把名称/属性/下标 `klass.update` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `klass.update` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `klass.predict_update_sequence = predict_update_sequence`：这是 Python 赋值语句。解释器先完整计算右侧 `predict_update_sequence` 得到一个对象，再把名称/属性/下标 `klass.predict_update_sequence` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `klass.predict_update_sequence` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `predict_update_sequence` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `klass.__init__ = init` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.156 install：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `install`；源码锚点 `def install(cusignal: Any, cp: Any) -> None:`。

```python

def install(cusignal: Any, cp: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    direct_convolve = _install_source_convolution(cp)
    _install_waveform_kernels(cp)
    filtering = importlib.import_module("cusignal.filtering.filtering")
    filtering.cp = _CupyProxy(cp, direct_convolve)
    wavelets = importlib.import_module("cusignal.wavelets.wavelets")
    wavelets.convolve = direct_convolve
    correlate = importlib.import_module("cusignal.convolution.correlate")
```

**语法结构**

这个 Python 代码块由函数或方法定义、条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `direct_convolve` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `filtering` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `wavelets` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `correlate` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `install` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`global`|当前表达式读取或传递的工程名称 `global`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `global _INSTALLED`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`direct_convolve`|文件系统路径或文件对象|无独立数学符号|定位配置、证据、输出或源码；不参与数值算法|
|`filtering`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `filtering`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `filtering = importlib.import_module("cusignal.filtering.filtering")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`wavelets`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `wavelets`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `wavelets = importlib.import_module("cusignal.wavelets.wavelets")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`correlate`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `correlate`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `correlate = importlib.import_module("cusignal.convolution.correlate")`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`install`|自定义函数/可调用入口 `install`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `install` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def install(cusignal: Any, cp: Any) -> None:`：`def` 创建名为 `install` 的函数对象；括号中的 `cusignal: Any, cp: Any` 是形参表，调用时实参按位置或关键字绑定。`-> None` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `global _INSTALLED`：名称解析：`global`、`_INSTALLED`按词法作用域查找对应变量、类型、函数或常量；逗号把当前项目与同一外层调用/容器中的下一项目分开，分号仅在 C/C++ 中结束完整语句。
- 对源码锚点 `if _INSTALLED:`：`if` 先计算条件 `_INSTALLED` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `return`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。
- 对源码锚点 `direct_convolve = _install_source_convolution(cp)`：这是 Python 赋值语句。解释器先完整计算右侧 `_install_source_convolution(cp)` 得到一个对象，再把名称/属性/下标 `direct_convolve` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `direct_convolve` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `_install_waveform_kernels(cp)`：`_install_waveform_kernels(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`cp` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `filtering = importlib.import_module("cusignal.filtering.filtering")`：这是 Python 赋值语句。解释器先完整计算右侧 `importlib.import_module("cusignal.filtering.filtering")` 得到一个对象，再把名称/属性/下标 `filtering` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `filtering` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `filtering.cp = _CupyProxy(cp, direct_convolve)`：这是 Python 赋值语句。解释器先完整计算右侧 `_CupyProxy(cp, direct_convolve)` 得到一个对象，再把名称/属性/下标 `filtering.cp` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `filtering.cp` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `wavelets = importlib.import_module("cusignal.wavelets.wavelets")`：这是 Python 赋值语句。解释器先完整计算右侧 `importlib.import_module("cusignal.wavelets.wavelets")` 得到一个对象，再把名称/属性/下标 `wavelets` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `wavelets` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `wavelets.convolve = direct_convolve`：这是 Python 赋值语句。解释器先完整计算右侧 `direct_convolve` 得到一个对象，再把名称/属性/下标 `wavelets.convolve` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `wavelets.convolve` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `correlate = importlib.import_module("cusignal.convolution.correlate")`：这是 Python 赋值语句。解释器先完整计算右侧 `importlib.import_module("cusignal.convolution.correlate")` 得到一个对象，再把名称/属性/下标 `correlate` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `correlate` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `correlate`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- CuPy 数组通常位于 GPU；访问结果或转成 NumPy 可能触发同步和 D2H，不能把它当作零成本 Python 容器操作；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.157 install：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `install`；源码锚点 `correlate.convolve = direct_convolve`。

```python
    correlate.convolve = direct_convolve
    _install_peak_helper(cp)
    _install_scalar_kalman(cp, cusignal)
    _DETAILS.update(
        implementation="cusignal_python_gpu_plus_required_zq500_adapter",
        public_cusignal_api_preserved=True,
        adapter_overhead_included_in_timing_numerator=True,
        cpu_fallback=False,
        waveform_kernels=True,
        direct_convolution=True,
        peak_helper=True,
        scalar_kalman=True,
        active_adapters=copy.deepcopy(ADAPTER_CONTRACTS),
        intermediate_double_audit=copy.deepcopy(INTERMEDIATE_DOUBLE_AUDIT),
        vendored_source_modified=False,
    )
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `implementation` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `public_cusignal_api_preserved` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `adapter_overhead_included_in_timing_numerator` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `cpu_fallback` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `waveform_kernels` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `direct_convolution` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `peak_helper` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `scalar_kalman` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `active_adapters` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `intermediate_double_audit` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `vendored_source_modified` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`implementation`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `implementation`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `implementation="cusignal_python_gpu_plus_required_zq500_adapter",`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`public_cusignal_api_preserved`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `public_cusignal_api_preserved`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `public_cusignal_api_preserved=True,`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`adapter_overhead_included_in_timing_numerator`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `adapter_overhead_included_in_timing_numerator`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `adapter_overhead_included_in_timing_numerator=True,`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`cpu_fallback`|CPU/reference 路径的对象或结果|与任务正式公式相同，用作独立参考|由 Host 计算产生，供 GPU/CPU comparison|
|`waveform_kernels`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`direct_convolution`|文件系统路径或文件对象|无独立数学符号|定位配置、证据、输出或源码；不参与数值算法|
|`peak_helper`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `peak_helper`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `peak_helper=True,`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`scalar_kalman`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `scalar_kalman`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `scalar_kalman=True,`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`active_adapters`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `active_adapters`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `active_adapters=copy.deepcopy(ADAPTER_CONTRACTS),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`intermediate_double_audit`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `intermediate_double_audit`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `intermediate_double_audit=copy.deepcopy(INTERMEDIATE_DOUBLE_AUDIT),`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|
|`vendored_source_modified`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `implementation="cusignal_python_gpu_plus_required_zq500_adapter",` 中，它为 `implementation` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `correlate.convolve = direct_convolve`：这是 Python 赋值语句。解释器先完整计算右侧 `direct_convolve` 得到一个对象，再把名称/属性/下标 `correlate.convolve` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `correlate.convolve` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `_install_peak_helper(cp)`：`_install_peak_helper(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`cp` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `_install_scalar_kalman(cp, cusignal)`：`_install_scalar_kalman(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`cp` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`cusignal` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `_DETAILS.update( implementation="cusignal_python_gpu_plus_required_zq500_adapter", public_cusignal_api_preserved=True, adapter_overhead_included_in_timing_numerator=True, cpu_fa...`：`_DETAILS.update(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`implementation="cusignal_python_gpu_plus_required_zq500_adapter"` 是关键字实参：先计算 `"cusignal_python_gpu_plus_required_zq500_adapter"`，再按参数名 `implementation` 绑定，不依赖它在形参表中的位置；`public_cusignal_api_preserved=True` 是关键字实参：先计算 `True`，再按参数名 `public_cusignal_api_preserved` 绑定，不依赖它在形参表中的位置；`adapter_overhead_included_in_timing_numerator=True` 是关键字实参：先计算 `True`，再按参数名 `adapter_overhead_included_in_timing_numerator` 绑定，不依赖它在形参表中的位置；`cpu_fallback=False` 是关键字实参：先计算 `False`，再按参数名 `cpu_fallback` 绑定，不依赖它在形参表中的位置；`waveform_kernels=True` 是关键字实参：先计算 `True`，再按参数名 `waveform_kernels` 绑定，不依赖它在形参表中的位置；`direct_convolution=True` 是关键字实参：先计算 `True`，再按参数名 `direct_convolution` 绑定，不依赖它在形参表中的位置；`peak_helper=True` 是关键字实参：先计算 `True`，再按参数名 `peak_helper` 绑定，不依赖它在形参表中的位置；`scalar_kalman=True` 是关键字实参：先计算 `True`，再按参数名 `scalar_kalman` 绑定，不依赖它在形参表中的位置；`active_adapters=copy.deepcopy(ADAPTER_CONTRACTS)` 是关键字实参：先计算 `copy.deepcopy(ADAPTER_CONTRACTS)`，再按参数名 `active_adapters` 绑定，不依赖它在形参表中的位置；`intermediate_double_audit=copy.deepcopy(INTERMEDIATE_DOUBLE_AUDIT)` 是关键字实参：先计算 `copy.deepcopy(INTERMEDIATE_DOUBLE_AUDIT)`，再按参数名 `intermediate_double_audit` 绑定，不依赖它在形参表中的位置；`vendored_source_modified=False` 是关键字实参：先计算 `False`，再按参数名 `vendored_source_modified` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `correlate`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.158 install：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `install`；源码锚点 `_INSTALLED = True`。

```python
    _INSTALLED = True
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `_INSTALLED` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`_INSTALLED`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `_INSTALLED`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `_INSTALLED = True`；值在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `_INSTALLED = True`：这是 Python 赋值语句。解释器先完整计算右侧 `True` 得到一个对象，再把名称/属性/下标 `_INSTALLED` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `_INSTALLED` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `install` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `_INSTALLED = True` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.159 adapter_contract：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `adapter_contract`；源码锚点 `def adapter_contract() -> dict[str, dict[str, Any]]:`。

```python

def adapter_contract() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(ADAPTER_CONTRACTS)
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `adapter_contract` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`adapter_contract`|自定义函数/可调用入口 `adapter_contract`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `adapter_contract` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def adapter_contract() -> dict[str, dict[str, Any]]:`：`def` 创建名为 `adapter_contract` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> dict[str, dict[str, Any]]` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `return copy.deepcopy(ADAPTER_CONTRACTS)`：`return` 先计算 `copy.deepcopy(ADAPTER_CONTRACTS)`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_contract` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.160 adapter_status：定义接口、类型或模板

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py`；符号 `adapter_status`；源码锚点 `def adapter_status() -> dict[str, Any]:`。

```python

def adapter_status() -> dict[str, Any]:
    return {"installed": _INSTALLED, **_DETAILS}
```

**语法结构**

这个 Python 代码块由函数或方法定义、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `adapter_status` 是 Python 函数名；`def` 创建函数对象，调用前不执行缩进函数体。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`adapter_status`|自定义函数/可调用入口 `adapter_status`|无独立数学变量；函数体实现的公式由参数、中间量和返回值分别映射|调用者传入实参；在 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 中承担 `adapter_status` 所表达的步骤职责|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `def adapter_status() -> dict[str, Any]:`：`def` 创建名为 `adapter_status` 的函数对象；括号中的 `无显式形参` 是形参表，调用时实参按位置或关键字绑定。`-> dict[str, Any]` 是返回类型注解，供读者、IDE 和静态检查器使用，Python 默认不会据此强制转换返回值；这里的 `->` 不是 C/C++ 指针成员访问。末尾冒号开始缩进函数体，函数体要等调用时才执行。
- 对源码锚点 `return {"installed": _INSTALLED, **_DETAILS}`：`return` 先计算 `{"installed": _INSTALLED, **_DETAILS}`，随后立即结束当前函数，把所得对象交给调用者；同一函数体中位于该路径后面的语句不再执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/utils/cusignal_source_adapter.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 表达式中的 `*` 可表示乘法，也可在调用/容器中解包；Python 没有 C++ 的 `Type*` 指针声明语法；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.161 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 cuSignal Python performance/result pipeline."""`。

```python
"""Task2 cuSignal Python performance/result pipeline."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`cuSignal`、`Python`、`performance`、`result`、`pipeline`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`cuSignal`|当前表达式读取或传递的工程名称 `cuSignal`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 cuSignal Python performance/result pipeline."""`；值在 `ZKX/Task2/task2_python/__init__.py` 当前作用域中产生或消费|
|`performance`|当前表达式读取或传递的工程名称 `performance`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 cuSignal Python performance/result pipeline."""`；值在 `ZKX/Task2/task2_python/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 cuSignal Python performance/result pipeline."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- Python 的 `/` 总是产生真除法结果，整数地板除法写作 `//`；这与 C++ 两整数相除会截断不同。

### 4.162 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 step1."""`。

```python
"""Task2 step1."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`step1`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`step1`|当前表达式读取或传递的工程名称 `step1`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 step1."""`；值在 `ZKX/Task2/task2_python/step1/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 step1."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step1/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2 step1."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.163 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/main.py`；符号 `adapter_status`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`sys`、`argparse`、`pathlib`、`Path`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|跨步编排|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step1/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.164 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/main.py`；符号 `adapter_status`；源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`。

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(Path(__file__).resolve().parents[1])` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.165 adapter_status：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step1/main.py`；符号 `adapter_status`；源码锚点 `if __name__ == "__main__":`。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(1, "step1", Task2Config.load(args.config))
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `run_task2_until(1, "step1", Task2Config.load(args.config))` 中，它作为 `run_task2_until` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `parser = argparse.ArgumentParser()`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser()` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(1, "step1", Task2Config.load(args.config))`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`1` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step1"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Task2Config.load(args.config)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.166 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 step2."""`。

```python
"""Task2 step2."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`step2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`step2`|当前表达式读取或传递的工程名称 `step2`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 step2."""`；值在 `ZKX/Task2/task2_python/step2/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 step2."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step2/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2 step2."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.167 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/main.py`；符号 `adapter_status`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`sys`、`argparse`、`pathlib`、`Path`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|跨步编排|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step2/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.168 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/main.py`；符号 `adapter_status`；源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`。

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(Path(__file__).resolve().parents[1])` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.169 adapter_status：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step2/main.py`；符号 `adapter_status`；源码锚点 `if __name__ == "__main__":`。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(2, "step2", Task2Config.load(args.config))
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{1}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `run_task2_until(2, "step2", Task2Config.load(args.config))` 中，它作为 `run_task2_until` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `parser = argparse.ArgumentParser()`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser()` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(2, "step2", Task2Config.load(args.config))`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`2` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step2"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Task2Config.load(args.config)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.170 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 step3."""`。

```python
"""Task2 step3."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`step3`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`step3`|当前表达式读取或传递的工程名称 `step3`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 step3."""`；值在 `ZKX/Task2/task2_python/step3/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 step3."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step3/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2 step3."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.171 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/main.py`；符号 `adapter_status`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`sys`、`argparse`、`pathlib`、`Path`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|跨步编排|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step3/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.172 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/main.py`；符号 `adapter_status`；源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`。

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(Path(__file__).resolve().parents[1])` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.173 adapter_status：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step3/main.py`；符号 `adapter_status`；源码锚点 `if __name__ == "__main__":`。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(3, "step3", Task2Config.load(args.config))
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `run_task2_until(3, "step3", Task2Config.load(args.config))` 中，它作为 `run_task2_until` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `parser = argparse.ArgumentParser()`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser()` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(3, "step3", Task2Config.load(args.config))`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`3` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step3"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Task2Config.load(args.config)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.174 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 step4."""`。

```python
"""Task2 step4."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`step4`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`step4`|当前表达式读取或传递的工程名称 `step4`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 step4."""`；值在 `ZKX/Task2/task2_python/step4/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 step4."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step4/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2 step4."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.175 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/main.py`；符号 `adapter_status`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`sys`、`argparse`、`pathlib`、`Path`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|跨步编排|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step4/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.176 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/main.py`；符号 `adapter_status`；源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`。

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(Path(__file__).resolve().parents[1])` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.177 adapter_status：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step4/main.py`；符号 `adapter_status`；源码锚点 `if __name__ == "__main__":`。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(4, "step4", Task2Config.load(args.config))
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `4` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是 $2^{2}$。二的幂常用于 FFT/规模/线程配置以便整齐分块；但若它来自 demo 或测试配置，源码只证明“采用该值”，没有证明它是唯一最优值。 当前上下文：在 `run_task2_until(4, "step4", Task2Config.load(args.config))` 中，它作为 `run_task2_until` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `parser = argparse.ArgumentParser()`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser()` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(4, "step4", Task2Config.load(args.config))`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`4` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step4"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Task2Config.load(args.config)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.178 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 step5."""`。

```python
"""Task2 step5."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`step5`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`step5`|当前表达式读取或传递的工程名称 `step5`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 step5."""`；值在 `ZKX/Task2/task2_python/step5/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 step5."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step5/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2 step5."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.179 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/main.py`；符号 `adapter_status`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`sys`、`argparse`、`pathlib`、`Path`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|跨步编排|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step5/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.180 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/main.py`；符号 `adapter_status`；源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`。

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(Path(__file__).resolve().parents[1])` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|输入准备/数据契约|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.181 adapter_status：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step5/main.py`；符号 `adapter_status`；源码锚点 `if __name__ == "__main__":`。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(5, "step5", Task2Config.load(args.config))
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `run_task2_until(5, "step5", Task2Config.load(args.config))` 中，它作为 `run_task2_until` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `parser = argparse.ArgumentParser()`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser()` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(5, "step5", Task2Config.load(args.config))`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`5` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step5"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Task2Config.load(args.config)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|输入准备/数据契约|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.182 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2 step6."""`。

```python
"""Task2 step6."""
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `Task2`、`step6`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`step6`|当前表达式读取或传递的工程名称 `step6`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2 step6."""`；值在 `ZKX/Task2/task2_python/step6/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `"""Task2 step6."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step6/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2 step6."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.183 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/main.py`；符号 `adapter_status`；源码锚点 `from __future__ import annotations`。

```python
from __future__ import annotations

import sys
import argparse
from pathlib import Path
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `__future__`、`annotations`、`sys`、`argparse`、`pathlib`、`Path`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `from __future__ import annotations`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import sys`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `import argparse`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from pathlib import Path`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|跨步编排|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明各业务步骤按既定顺序连接以及状态如何传递；每一步的数学正确性仍由对应 step 实现和测试文档证明。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/step6/main.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `from __future__ import annotations` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.184 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/main.py`；符号 `adapter_status`；源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`。

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_runner import run_task2_until
from demo.task_config import Task2Config
```

**语法结构**

这个 Python 代码块由函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 中，它作为 `sys.path.insert` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`：`sys.path.insert(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`0` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`str(Path(__file__).resolve().parents[1])` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `from task2_runner import run_task2_until`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 导入只建立名称依赖，不等于执行任务流水线入口。
- 对源码锚点 `from demo.task_config import Task2Config`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.185 adapter_status：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/step6/main.py`；符号 `adapter_status`；源码锚点 `if __name__ == "__main__":`。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task2_until(6, "step6", Task2Config.load(args.config))
```

**语法结构**

这个 Python 代码块由条件分支、函数调用构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `parser` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `args` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定；
- `6` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`parser`|命令行解析器对象|无独立数学符号|定义 CLI 字段并生成 args|
|`args`|解析后的命令行参数对象|无独立数学符号|向配置加载和 runner 提供路径/运行选择|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`6`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `run_task2_until(6, "step6", Task2Config.load(args.config))` 中，它作为 `run_task2_until` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `if __name__ == "__main__":`：`if` 先计算条件 `__name__ == "__main__"` 并调用其真假规则；只有结果为真才执行后续缩进块。`==`（若出现）是相等比较而不是赋值，末尾冒号开始受该条件控制的代码块。
- 对源码锚点 `parser = argparse.ArgumentParser()`：这是 Python 赋值语句。解释器先完整计算右侧 `argparse.ArgumentParser()` 得到一个对象，再把名称/属性/下标 `parser` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `parser` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `parser.add_argument("--config", required=True)`：`parser.add_argument(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`"--config"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`required=True` 是关键字实参：先计算 `True`，再按参数名 `required` 绑定，不依赖它在形参表中的位置。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。
- 对源码锚点 `args = parser.parse_args()`：这是 Python 赋值语句。解释器先完整计算右侧 `parser.parse_args()` 得到一个对象，再把名称/属性/下标 `args` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `args` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。
- 对源码锚点 `run_task2_until(6, "step6", Task2Config.load(args.config))`：`run_task2_until(...)` 是函数/方法调用；点号逐级取得对象属性或方法，圆括号触发调用。；`6` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`"step6"` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参；`Task2Config.load(args.config)` 是位置实参，按出现顺序绑定到下一个尚未赋值的形参。调用返回后，若外层没有赋值或 return，普通返回值会被忽略，但函数仍可能修改对象、写文件或产生其他副作用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `==` 比较值是否相等；`is` 比较是否为同一个对象，除 `None` 等单例判断外通常不能互换；
- 调用中的 `name=value` 是关键字实参绑定，不是给当前作用域变量赋值；真正赋值只发生在调用括号外的顶层 `=`。

### 4.186 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/__init__.py`；符号 `adapter_status`；源码锚点 `"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""`。

```python
"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""

from .cusignal_source_adapter import adapter_contract, adapter_status, install
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `adapter_contract` 由 `import` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`utilities`|当前表达式读取或传递的工程名称 `utilities`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""`；值在 `ZKX/Task2/task2_python/utils/__init__.py` 当前作用域中产生或消费|
|`adapter_contract`|当前表达式读取或传递的工程名称 `adapter_contract`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `from .cusignal_source_adapter import adapter_contract, adapter_status, install`；值在 `ZKX/Task2/task2_python/utils/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`00`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""`：引号创建字符串对象；字符串内容是消息、键名、路径或下一层函数实参，不会被当作代码执行。末尾逗号表示外层实参/容器仍未结束，`)` 则结束外层调用。
- 对源码锚点 `from .cusignal_source_adapter import adapter_contract, adapter_status, install`：关键字语法：`import`：import 加载模块并在当前命名空间绑定名称；`from`：from ... import ... 从模块中直接绑定指定名称。 运算符：`.`：通过对象本身访问成员。 导入只建立名称依赖，不等于执行任务流水线入口。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/utils/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `"""Task2-local ZQ500 utilities kept outside the immutable cuSignal tree."""` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

### 4.187 adapter_status：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_python/utils/__init__.py`；符号 `adapter_status`；源码锚点 `__all__ = ["adapter_contract", "adapter_status", "install"]`。

```python
__all__ = ["adapter_contract", "adapter_status", "install"]
```

**语法结构**

这个 Python 代码块由表达式或容器续接构成。Python 用冒号引出复合语句、用缩进确定函数/分支/循环的归属，不使用 C++ 花括号划分代码块。圆括号既可组成调用实参表，也可分组表达式或让一条语句跨物理换行；方括号用于列表、下标和切片；花括号若出现则创建 dict/set 或格式化字段。只有括号尚未闭合、反斜杠续行或三引号字符串未结束时，下一物理行才自动续接当前逻辑语句。

**名称与类型**

- `__all__` 是 Python 名称；赋值会把右侧对象引用绑定到该名称，类型在运行时由对象决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`__all__`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `__all__`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `__all__ = ["adapter_contract", "adapter_status", "install"]`；值在 `ZKX/Task2/task2_python/utils/__init__.py` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `__all__ = ["adapter_contract", "adapter_status", "install"]`：这是 Python 赋值语句。解释器先完整计算右侧 `["adapter_contract", "adapter_status", "install"]` 得到一个对象，再把名称/属性/下标 `__all__` 绑定或写成对该对象的引用；赋值本身不复制任意对象内容。若右侧含函数调用，先计算调用实参、执行函数并取得返回对象；后续代码通过 `__all__` 读取这个结果。`=` 是赋值，不是相等比较；比较应写 `==`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|跨条款支持：T2-R1、T2-R2、T2-R3、T2-R4、T2-R5、T2-R6|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置、共享状态、已产生的步骤结果或证据文件，具体由本块实参决定。|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|工程状态、运行控制、统计记录或图形派生物；不替代任何 step 的数学输出。|
|下一消费者|相应 step、测试门禁或可视化消费者。|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `adapter_status` 所在的 `ZKX/Task2/task2_python/utils/__init__.py` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 阅读 `__all__ = ["adapter_contract", "adapter_status", "install"]` 时要区分名称绑定与对象复制：Python 赋值通常只让名称引用对象，不会自动深拷贝可变数组、列表或配置对象。

## 5. 调用链与自检

main→配置→runner→唯一PipelineState→各step→Host/JSON。逐步辨认prep、H2D、GPU、D2H、postprocess；NumPy是Host契约，CuPy是device数据。完整源码与逐行解释不能互相替代。
