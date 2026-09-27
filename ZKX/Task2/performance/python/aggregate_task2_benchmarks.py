from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate(stats: list[dict[str, Any]]) -> dict[str, Any]:
    batches = [[float(value) for value in item["raw_samples_ms"]] for item in stats]
    if len(batches) != 5 or any(len(batch) != 20 for batch in batches):
        raise RuntimeError("formal aggregate requires five batches of twenty samples")
    batch_means = [statistics.fmean(batch) for batch in batches]
    batch_cvs = [statistics.stdev(batch) / mean if mean else 0.0 for batch, mean in zip(batches, batch_means)]
    pooled = [value for batch in batches for value in batch]
    pooled_mean = statistics.fmean(pooled)
    deviation = statistics.stdev(pooled)
    return {
        "total": sum(pooled),
        "mean": statistics.median(batch_means),
        "formal_center": "median_of_five_batch_means",
        "min": min(pooled),
        "max": max(pooled),
        "sample_stddev": deviation,
        "median": statistics.median(pooled),
        "p95": _percentile(pooled, 0.95),
        "p99": _percentile(pooled, 0.99),
        "cv": deviation / pooled_mean if pooled_mean else 0.0,
        "batch_means": batch_means,
        "batch_mean_min": min(batch_means),
        "batch_mean_max": max(batch_means),
        "max_batch_cv": max(batch_cvs),
        "raw_sample_count": len(pooled),
        "raw_samples_ms": pooled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate five formal Task2 Python benchmark batches")
    parser.add_argument("--batch-summary", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = [_load(path) for path in args.batch_summary]
    ids = {int(item.get("batch_id", 0)) for item in batches}
    if ids != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"batch ids must be 1..5, found {sorted(ids)}")
    for item in batches:
        if item.get("task") != "Task2" or item.get("status") != "pass":
            raise RuntimeError("invalid Task2 batch identity/status")
        if item.get("warmup_runs") != 5 or item.get("measured_runs") != 20:
            raise RuntimeError("each formal batch requires warmup=5 and measured=20")
    for field in ("git_commit", "git_dirty", "input_contract", "performance_identity"):
        if any(item.get(field) != batches[0].get(field) for item in batches[1:]):
            raise RuntimeError(f"batch contract mismatch: {field}")
    result = {key: value for key, value in batches[0].items() if key not in {
        "batch_id", "warmup_runs", "measured_runs", "pipeline_total_ms",
        "sum_step_total_ms", "sum_step_formal_execution_ms", "steps",
    }}
    result.update({
        "benchmark_policy": "five_independent_batches_median_of_batch_means",
        "batches": 5,
        "warmup_runs_per_batch": 5,
        "measured_runs_per_batch": 20,
        "total_measured_runs": 100,
        "batch_ids": sorted(ids),
        "pipeline_total_ms": _aggregate([item["pipeline_total_ms"] for item in batches]),
        "sum_step_total_ms": _aggregate([item["sum_step_total_ms"] for item in batches]),
        "sum_step_formal_execution_ms": _aggregate([item["sum_step_formal_execution_ms"] for item in batches]),
        "steps": {},
    })
    for step_number in range(1, 7):
        name = f"step{step_number}"
        result["steps"][name] = {}
        for field in ("prepare", "h2d", "compute", "d2h", "postprocess", "formal_execution", "total"):
            result["steps"][name][field] = _aggregate([item["steps"][name][field] for item in batches])
        result["steps"][name]["operators_ms"] = {}
        for operator, first in batches[0]["steps"][name]["operators_ms"].items():
            metadata = {key: value for key, value in first.items() if key not in {"warmup_runs", "measured_runs", "timing_ms"}}
            metadata.update({
                "batches": 5,
                "warmup_runs_per_batch": 5,
                "measured_runs_per_batch": 20,
                "total_measured_runs": 100,
                "timing_ms": _aggregate([item["steps"][name]["operators_ms"][operator]["timing_ms"] for item in batches]),
            })
            result["steps"][name]["operators_ms"][operator] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[TASK2_PYTHON][BENCHMARK_AGGREGATE] batches=5 measured=100 output={args.output} status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
