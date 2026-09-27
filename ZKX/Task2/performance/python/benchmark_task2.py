from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PYTHON_REFERENCE = Path(__file__).resolve().parents[2] / "python_reference"
sys.path.insert(0, str(PYTHON_REFERENCE))

from task2_runner import run_task2_until
from demo.task_config import Task2Config


OPERATOR_CONTRACTS: dict[str, dict[str, dict[str, Any]]] = {
    "step1": {
        "cusignal.chirp": {"dtype": "FP32", "input_shape": [512], "parameters": {"f0": 20.0, "t1": 1.0, "f1": 70.0, "phi": 0.0}},
        "cusignal.gausspulse": {"dtype": "FP32", "input_shape": [512], "parameters": {"fc": 170.0, "bw": 0.20}},
        "cusignal.sawtooth": {"dtype": "FP32 input", "input_shape": [512], "parameters": {"width": 0.65}},
        "cusignal.square": {"dtype": "FP32 input", "input_shape": [512], "parameters": {"duty": 0.35}},
        "cupy.echo_superposition": {"dtype": "FP32", "input_shape": [4, 512], "parameters": {"target_delay": 96, "target_weight": 0.80, "noise_seed": 20260714}},
    },
    "step2": {
        "cusignal.hamming": {"dtype": "implementation-selected floating", "input_shape": [33], "parameters": {"M": 33, "sym": True}},
        "cusignal.firwin": {"dtype": "implementation-selected floating", "input_shape": [33], "parameters": {"numtaps": 33, "cutoff_hz": 90.0, "window": "hamming", "fs_hz": 512.0, "gpupath": True}},
        "cusignal.firfilter": {"dtype": "FP32 signal", "input_shape": [512], "parameters": {"axis": -1, "taps": 33}},
    },
    "step3": {
        "cusignal.cubic": {"dtype": "FP32 input", "input_shape": [9], "parameters": {"offset_begin": -2.0, "offset_end": 2.0, "offset_step": 0.5}},
        "cupy.cubic_normalization": {"dtype": "implementation-selected floating", "input_shape": [9], "parameters": {"normalization": "sum=1"}},
        "cusignal.firfilter_b_spline": {"dtype": "FP32 signal", "input_shape": [480], "parameters": {"axis": -1, "taps": 9}},
    },
    "step4": {
        "cupy.analytic_glue": {"dtype": "FP32 to ComplexFP32", "input_shape": [480], "parameters": {"imaginary": "0.5*(next-previous)"}},
        "cusignal.fm_demod": {"dtype": "ComplexFP32", "input_shape": [480], "parameters": {"axis": -1}},
        "cusignal.correlate": {"dtype": "FP32", "input_shape": [[480], [480]], "parameters": {"mode": "same", "method": "auto"}},
        "cusignal.spectrogram": {"dtype": "FP32", "input_shape": [480], "parameters": {"fs_hz": 512.0, "window": "boxcar", "nperseg": 64, "noverlap": 32, "nfft": 64, "return_onesided": False, "scaling": "spectrum", "mode": "magnitude"}},
        "cupy.spectral_envelope": {"dtype": "implementation spectrogram magnitude to FP32", "input_shape": [64, 14], "parameters": {"reduction": "max(abs), axis=frequency"}},
        "cusignal.cwt_ricker": {"dtype": "FP32 input / implementation FP64 output", "input_shape": [480], "parameters": {"widths": [2, 4, 8, 12], "wavelet": "ricker"}},
        "cupy.cwt_envelope": {"dtype": "implementation FP64 to FP32", "input_shape": [4, 480], "parameters": {"reduction": "max(abs), axis=width"}},
        "cupy.feature_fusion": {"dtype": "FP32", "input_shape": "four variable-length feature vectors -> [480]", "parameters": {"weights": [0.10, 0.70, 0.10, 0.10], "normalization": "per-feature max-abs", "resampling": "linear"}},
    },
    "step5": {
        "cusignal.argrelextrema": {"dtype": "FP32 input / INT64 indices", "input_shape": [480], "parameters": {"comparator": "cupy.greater", "axis": 0, "order": 3, "mode": "clip"}},
    },
    "step6": {
        "cusignal.KalmanFilter.predict_update": {"dtype": "FP32", "input_shape": {"observations": [8], "state": [1], "covariance": [1]}, "parameters": {"dim_x": 1, "dim_z": 1, "points": 1, "F": 1.0, "Q": 0.0001, "H": 1.0, "R": 0.002, "iterations": 8}},
    },
}


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "total": sum(values),
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "sample_stddev": deviation,
        "median": statistics.median(values),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "cv": deviation / mean if mean else 0.0,
        "raw_samples_ms": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Task2 cuSignal Python")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--output-dir", default="task2_python_evidence")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.batch_id < 1:
        parser.error("warmup must be nonnegative; repeats and batch-id must be positive")
    config = Task2Config.load(args.config)
    operator_contracts = copy.deepcopy(OPERATOR_CONTRACTS)
    step3_samples = config.samples - 2 * config.step3_crop_each
    for contract in operator_contracts["step1"].values():
        contract["input_shape"] = [config.samples] if contract["input_shape"] != [4, 512] else [4, config.samples]
    operator_contracts["step1"]["cupy.echo_superposition"]["parameters"].update(
        target_delay=config.target_delay_samples, target_weight=config.target_weight,
        noise_seed=config.noise_seed
    )
    operator_contracts["step2"]["cusignal.hamming"]["input_shape"] = [config.filter_taps]
    operator_contracts["step2"]["cusignal.firwin"]["input_shape"] = [config.filter_taps]
    operator_contracts["step2"]["cusignal.firwin"]["parameters"].update(
        numtaps=config.filter_taps, cutoff_hz=config.filter_cutoff_hz,
        fs_hz=config.sample_rate_hz
    )
    operator_contracts["step2"]["cusignal.firfilter"]["input_shape"] = [config.samples]
    operator_contracts["step2"]["cusignal.firfilter"]["parameters"]["taps"] = config.filter_taps
    for contract in operator_contracts["step3"].values():
        if contract["input_shape"] == [480]:
            contract["input_shape"] = [step3_samples]
    for contract in operator_contracts["step4"].values():
        shape = contract["input_shape"]
        if shape == [480]:
            contract["input_shape"] = [step3_samples]
        elif shape == [[480], [480]]:
            contract["input_shape"] = [[step3_samples], [step3_samples]]
    operator_contracts["step4"]["cusignal.spectrogram"]["parameters"]["fs_hz"] = config.sample_rate_hz
    operator_contracts["step4"]["cupy.feature_fusion"]["parameters"]["weights"] = [
        config.fusion_weight_fm, config.fusion_weight_correlation,
        config.fusion_weight_spectral, config.fusion_weight_wavelet,
    ]
    operator_contracts["step5"]["cusignal.argrelextrema"]["input_shape"] = [step3_samples]
    operator_contracts["step5"]["cusignal.argrelextrema"]["parameters"]["order"] = config.argrelextrema_order
    kalman_contract = operator_contracts["step6"]["cusignal.KalmanFilter.predict_update"]
    kalman_contract["input_shape"]["observations"] = [config.kalman_observation_count]
    kalman_contract["parameters"].update(
        F=config.kalman_F, Q=config.kalman_Q, H=config.kalman_H, R=config.kalman_R,
        iterations=config.kalman_observation_count
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmup):
        run_task2_until(6, f"warmup_{index + 1}", config, output / "warmup", write_json=False, quiet=True)
    runs: list[dict[str, Any]] = []
    for index in range(args.repeats):
        runs.append(
            run_task2_until(6, f"benchmark_{index + 1}", config, output / f"run_{index + 1}", quiet=True)
        )
    summary: dict[str, Any] = {
        "task": "Task2",
        "implementation": "cusignal_python_23.08.00",
        "performance_identity": "cuSignal Python GPU + required ZQ500 adapter",
        "status": "pass",
        "batch_id": args.batch_id,
        "git_commit": runs[-1]["git_commit"],
        "git_dirty": runs[-1]["git_dirty"],
        "test_time": runs[-1]["test_time"],
        "warmup_runs": args.warmup,
        "measured_runs": args.repeats,
        "timing_scope": {
            "pipeline_total_ms": "six formal steps plus Host semantic checks",
            "formal_execution_ms": "prepare, H2D, GPU operators and required D2H; excludes semantic checks",
            "compute_ms": "sum of per-operator CuPy CUDA-event times",
            "operator_sync": "each CUDA-event stop is synchronized before elapsed time is read",
            "operator_h2d_included": False,
            "operator_d2h_included": False,
            "operator_allocation": "high-level cuSignal/CuPy output and temporary allocations are included inside each timed call",
            "operator_plan_workspace": "implementation-internal setup is included when performed inside the high-level call; Kalman object/state setup is excluded",
        },
        "pipeline_total_ms": _stats([run["pipeline_total_ms"] for run in runs]),
        "sum_step_total_ms": _stats([run["sum_step_total_ms"] for run in runs]),
        "sum_step_formal_execution_ms": _stats([
            sum(float(step["timing_ms"]["formal_execution"]) for step in run["steps"])
            for run in runs
        ]),
        "steps": {},
        "runtime": runs[-1]["runtime"],
        "input_contract": runs[-1]["input_contract"],
        "config": runs[-1]["config"],
    }
    timing_fields = ("prepare", "h2d", "compute", "d2h", "postprocess", "formal_execution", "total")
    for step_index in range(6):
        name = f"step{step_index + 1}"
        summary["steps"][name] = {
            field: _stats([run["steps"][step_index]["timing_ms"][field] for run in runs])
            for field in timing_fields
        }
        operator_names = runs[-1]["steps"][step_index]["operators_ms"].keys()
        summary["steps"][name]["operators_ms"] = {
            operator: {
                **operator_contracts[name][operator],
                "warmup_runs": args.warmup,
                "measured_runs": args.repeats,
                "synchronization": "cupy.cuda.Event stop.synchronize",
                "includes_h2d": False,
                "includes_d2h": False,
                "includes_allocation": operator != "cusignal.KalmanFilter.predict_update",
                "plan_workspace_scope": (
                    "pre-created Kalman object/state excluded; predict/update kernel internals included"
                    if operator == "cusignal.KalmanFilter.predict_update"
                    else "implementation-internal setup included when the high-level call performs it"
                ),
                "timing_ms": _stats(
                    [float(run["steps"][step_index]["operators_ms"][operator]) for run in runs]
                ),
            }
            for operator in operator_names
        }
    destination = output / "task2_python_benchmark_summary.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[TASK2_PYTHON][BENCHMARK] warmup={args.warmup} repeats={args.repeats} "
        f"pipeline_mean_ms={summary['pipeline_total_ms']['mean']:.6f} "
        f"summary={destination} status=pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
