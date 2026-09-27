from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from task1_runner import run_task1_until
from task1_common import Task1Config


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
    parser = argparse.ArgumentParser(description="Benchmark Task1 cuSignal Python")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default="task1_python_evidence")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >=0 and repeats must be >=1")
    output = Path(args.output_dir)
    config = Task1Config.load(args.config)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmup):
        run_task1_until(
            5, f"warmup_{index + 1}", config, output / "warmup", write_json=False, quiet=True
        )
    runs: list[dict[str, Any]] = []
    for index in range(args.repeats):
        runs.append(
            run_task1_until(
                5, f"benchmark_{index + 1}", config, output / f"run_{index + 1}", quiet=True
            )
        )
    summary: dict[str, Any] = {
        "task": "Task1",
        "implementation": "cusignal_python_23.08.00",
        "status": "pass",
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "warmup_runs": args.warmup,
        "measured_runs": args.repeats,
        "timing_scope": {
            "pipeline_total_ms": "formal GPU steps, D2H result copies and result summarization; no CPU reference/comparison",
            "formal_execution_ms": "formal step plus required D2H result copy; excludes result summarization",
            "compute_ms": "sum of per-operator CuPy CUDA-event times; cfar_alpha is Host wall time",
            "speedup_numerator": (
                "cuSignal Python GPU formal_execution_ms including required ZQ500 adapters"
            ),
        },
        "profiling": {
            "self_timed_benchmark": True,
            "dlpti_used": False,
            "dlpti_scope": "optional auxiliary kernel/transfer/synchronization analysis only",
        },
        "pipeline_total_ms": _stats([run["pipeline_total_ms"] for run in runs]),
        "sum_step_total_ms": _stats([run["sum_step_total_ms"] for run in runs]),
        "steps": {},
        "runtime": runs[-1]["runtime"],
        "input_contract": runs[-1]["input_contract"],
        "implementation_contract": runs[-1]["implementation_contract"],
        "speedup_definition": {
            "id": "task1_cusignal_python_gpu_adapter_over_cpp_cuda_gpu",
            "formula": (
                "T_(cuSignal Python GPU + required ZQ500 adapter) / "
                "T_(C++/CUDA GPU)"
            ),
            "numerator": (
                "cuSignal Python GPU formal_execution_ms, including all required "
                "Task1 utils ZQ500 adapter time"
            ),
            "denominator": "C++/CUDA GPU formal_execution_ms for the matching Task1 step",
        },
    }
    for step_index in range(5):
        name = f"step{step_index + 1}"
        summary["steps"][name] = {
            field: _stats(
                [run["steps"][step_index]["timing_ms"][field] for run in runs]
            )
            for field in ("prepare", "h2d", "compute", "d2h", "postprocess", "formal_execution", "total")
        }
        operator_names = sorted(runs[-1]["steps"][step_index]["operators_ms"])
        summary["steps"][name]["operators_ms"] = {
            operator: _stats([
                run["steps"][step_index]["operators_ms"][operator] for run in runs
            ])
            for operator in operator_names
        }
        summary["steps"][name]["timing_categories_ms"] = {
            category: _stats([
                run["steps"][step_index]["timing_categories_ms"][category]
                for run in runs
            ])
            for category in (
                "cusignal_api", "required_zq500_adapter", "task_scaffolding"
            )
        }
        summary["steps"][name]["result_metrics"] = runs[-1]["steps"][step_index][
            "result_metrics"
        ]
    destination = output / "task1_python_benchmark_summary.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[TASK1_PYTHON][BENCHMARK] warmup={args.warmup} repeats={args.repeats} "
        f"pipeline_mean_ms={summary['pipeline_total_ms']['mean']:.6f} "
        f"summary={destination} status=pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
