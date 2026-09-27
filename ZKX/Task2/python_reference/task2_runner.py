from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from steps.step1.step1 import run_step1
from steps.step2.step2 import run_step2
from steps.step3.step3 import run_step3
from steps.step4.step4 import run_step4
from steps.step5.step5 import run_step5
from steps.step6.step6 import run_step6
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
                "pipeline_handoff": "Host PipelineState with explicit H2D/D2H, matching the C++/GPU pipeline",
                "d2h_scope": "formal step output copied for downstream step and evidence",
                "postprocess_scope": "mandatory Host handoff where applicable plus semantic/task-effect checks; no CPU precision reference",
                "formal_execution_scope": "prepare + H2D + GPU operators + required D2H and mandatory handoff; excludes semantic checks",
                "adapter_scope": "all required Task2/utils ZQ500 adapter execution is inside the cuSignal Python GPU timing numerator",
            },
            "precision_policy": {
                "cpu_validation": False,
                "cpu_comparison": False,
                "reason": "Task2 Python is the cuSignal performance baseline; CPU precision evidence belongs to the C++/GPU pipeline only",
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
