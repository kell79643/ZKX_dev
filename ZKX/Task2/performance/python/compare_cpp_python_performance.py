from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


FORMULA = "T_cusignal_python_gpu_plus_required_zq500_adapter / T_cpp_cuda_gpu"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _formal_stats(batches: list[list[float]]) -> dict[str, Any]:
    if len(batches) != 5 or any(len(batch) != 20 for batch in batches):
        raise RuntimeError("C++ formal benchmark requires five batches of twenty measured runs")
    means = [statistics.fmean(batch) for batch in batches]
    pooled = [value for batch in batches for value in batch]
    pooled_mean = statistics.fmean(pooled)
    deviation = statistics.stdev(pooled)
    cvs = [statistics.stdev(batch) / mean if mean else 0.0 for batch, mean in zip(batches, means)]
    return {
        "total": sum(pooled), "mean": statistics.median(means),
        "formal_center": "median_of_five_batch_means", "min": min(pooled), "max": max(pooled),
        "sample_stddev": deviation, "median": statistics.median(pooled),
        "p95": _percentile(pooled, 0.95), "p99": _percentile(pooled, 0.99),
        "cv": deviation / pooled_mean if pooled_mean else 0.0,
        "batch_means": means, "batch_mean_min": min(means), "batch_mean_max": max(means),
        "max_batch_cv": max(cvs), "raw_sample_count": 100,
    }


def _collect_cpp(root: Path) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    for batch_id in range(1, 6):
        measured = root / f"batch_{batch_id}" / "measured"
        runs = [_load(path) for path in sorted(measured.glob("run_*/task2_step6_evidence.json"), key=lambda p: int(p.parent.name.split("_")[-1]))]
        if len(runs) != 20:
            raise RuntimeError(f"C++ batch {batch_id} requires twenty measured runs, found {len(runs)}")
        batches.append(runs)
    return batches


def _speedup(python_stats: dict[str, Any], cpp_stats: dict[str, Any], formal: bool = True) -> dict[str, Any]:
    numerator = float(python_stats["mean"])
    denominator = float(cpp_stats["mean"])
    if numerator <= 0.0 or denominator <= 0.0:
        raise RuntimeError("performance comparison requires positive timing values")
    value = numerator / denominator
    return {
        "formal_speedup": formal,
        "speedup_name": "cuSignal Python GPU + required ZQ500 adapter / C++/CUDA GPU",
        "speedup_formula": FORMULA,
        "numerator_identity": "cuSignal Python GPU + required ZQ500 adapter",
        "denominator_identity": "C++/CUDA GPU",
        "numerator_ms": numerator,
        "denominator_ms": denominator,
        "speedup": value,
        "interpretation": "C++/CUDA GPU is faster by this factor" if value >= 1.0 else "cuSignal Python GPU + required ZQ500 adapter is faster by the reciprocal factor",
    }


def _cpp_stats(batches: list[list[dict[str, Any]]], extractor: Any) -> dict[str, Any]:
    return _formal_stats([[float(extractor(run)) for run in batch] for batch in batches])


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal Task2 cuSignal Python GPU and C++/CUDA GPU comparison")
    parser.add_argument("--python-summary", required=True, type=Path)
    parser.add_argument("--cpp-evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    python = _load(args.python_summary)
    if python.get("task") != "Task2" or python.get("status") != "pass" or python.get("total_measured_runs") != 100:
        raise RuntimeError("invalid formal Task2 Python aggregate")
    cpp = _collect_cpp(args.cpp_evidence_root)
    flat_cpp = [run for batch in cpp for run in batch]
    for run in flat_cpp:
        if run.get("task") != "Task2" or run.get("status") != "pass" or len(run.get("steps", [])) != 6:
            raise RuntimeError("invalid C++ Task2 evidence")
        if run.get("input_contract") != python.get("input_contract"):
            raise RuntimeError("C++/Python input_contract mismatch")
        if run.get("git_commit") != python.get("git_commit"):
            raise RuntimeError("C++/Python evidence commit mismatch")
    result: dict[str, Any] = {
        "task": "Task2", "status": "pass",
        "comparison": "cusignal_python_gpu_plus_required_zq500_adapter_vs_cpp_cuda_gpu",
        "git_commit": python["git_commit"], "git_dirty": python["git_dirty"], "test_time": python["test_time"],
        "speedup_definition": {
            "formula": FORMULA,
            "numerator": "cuSignal 23.08 public Python/GPU calls plus only the required Task2 utils ZQ500 adapters; all adapter execution is inside the measured numerator; warmup/JIT is excluded by the common warmup policy",
            "denominator": "C++/CUDA GPU six-step formal_execution scope with matched input/parameters; CPU validation, CPU comparison and semantic post-checks are excluded",
        },
        "fairness_contract": {
            "same_input": True, "same_six_steps": True, "same_dtype": "FP32 business input",
            "batches_per_implementation": 5, "warmup_runs_per_batch": 5,
            "measured_runs_per_batch": 20, "measured_runs_per_implementation": 100,
            "formal_center": "median_of_five_batch_means", "pooled_tail": "p95",
            "python_public_cusignal_api": True, "vendored_cusignal_source_modified": False,
            "required_adapter_execution_in_python_numerator": True, "python_cpu_fallback": False,
        },
        "task_formal": {}, "pipeline_diagnostic": {}, "steps": {},
        "python_runtime": python.get("runtime", {}), "input_contract": python["input_contract"],
    }
    py_task = python["sum_step_formal_execution_ms"]
    cpp_task = _cpp_stats(cpp, lambda run: sum(float(step["timing_ms"]["formal_execution"]) for step in run["steps"]))
    result["task_formal"] = {"python_ms": py_task, "cpp_ms": cpp_task, **_speedup(py_task, cpp_task)}
    py_pipeline = python["pipeline_total_ms"]
    cpp_pipeline = _cpp_stats(cpp, lambda run: run["pipeline_total_ms"])
    result["pipeline_diagnostic"] = {
        "python_ms": py_pipeline, "cpp_ms": cpp_pipeline,
        **_speedup(py_pipeline, cpp_pipeline, formal=False),
        "reason": "pipeline totals include different validation/semantic-check responsibilities; this ratio is diagnostic only",
    }
    for index in range(6):
        name = f"step{index + 1}"
        py_step = python["steps"][name]["formal_execution"]
        cpp_step = _cpp_stats(cpp, lambda run, i=index: run["steps"][i]["timing_ms"]["formal_execution"])
        result["steps"][name] = {"python_ms": py_step, "cpp_ms": cpp_step, **_speedup(py_step, cpp_step)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[TASK2_CPP_PYTHON][PERFORMANCE] formula={FORMULA} measured_each=100 output={args.output} status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
