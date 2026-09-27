#!/usr/bin/env python3
"""深检一个已完成的 Task2 Step 或 pipeline 110-case 正式矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED_CASES = 110


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_at_most(row: dict[str, str], actual: str, limit: str) -> bool:
    value = float(row[actual])
    maximum = float(row[limit])
    return math.isfinite(value) and math.isfinite(maximum) and value <= maximum


def verify_case(
    case: Path,
    scale_id: str,
    backend: str,
    step_name: str,
) -> None:
    suffix = f"{backend}_formal.csv"
    main = read_csv(case / f"task2_benchmark_{suffix}")
    timing = read_csv(case / f"task2_iteration_timing_{suffix}")
    memory = read_csv(case / f"task2_memory_trace_{suffix}")
    accuracy = read_csv(case / f"task2_accuracy_details_{suffix}")

    require(len(main) == 2, f"main row count mismatch: {case}")
    require(len(timing) == 200, f"timing row count mismatch: {case}")
    expected_memory_rows = 4200 if step_name == "pipeline" else 600
    require(len(memory) == expected_memory_rows, f"memory row count mismatch: {case}")
    if step_name == "pipeline":
        require(
            {row["resource_scope"] for row in memory}
            == {"step1", "step2", "step3", "step4", "step5", "step6", "pipeline_total"},
            f"pipeline memory scope mismatch: {case}",
        )
        expected_trace_keys = {
            (str(sample), device, scope, phase)
            for sample in range(100)
            for device in ("cpu", "gpu")
            for scope in (
                "step1", "step2", "step3", "step4", "step5", "step6",
                "pipeline_total",
            )
            for phase in ("before", "peak", "after")
        }
        require(
            {
                (
                    row["sample_index"], row["device"],
                    row["resource_scope"], row["trace_phase"],
                )
                for row in memory
            } == expected_trace_keys,
            f"pipeline memory coverage mismatch: {case}",
        )
    expected_accuracy_rows = {
        "step1": 5,
        "step2": 1,
        "step3": 2,
        "step4": 5,
        "step5": 1,
        "step6": 2,
        "pipeline": 16,
    }[step_name]
    require(
        len(accuracy) == expected_accuracy_rows,
        f"accuracy row count mismatch: {case}",
    )

    for row in main:
        require(row["backend"] == backend, f"backend mismatch: {case}")
        require(row["scale_id"] == scale_id, f"scale ID mismatch: {case}")
        require(row["run_type"] == "formal", f"run type mismatch: {case}")
        require(row["warmup_runs"] == "20", f"warmup count mismatch: {case}")
        require(row["measured_runs"] == "100", f"measured count mismatch: {case}")
        require(row["status"] == "PASS", f"main status mismatch: {case}")
        require(row["accuracy_status"] == "PASS", f"main accuracy mismatch: {case}")

    by_device = {
        device: [row for row in timing if row["device"] == device]
        for device in ("cpu", "gpu")
    }
    for device, rows in by_device.items():
        require(len(rows) == 100, f"{device} timing count mismatch: {case}")
        require(
            {int(row["sample_index"]) for row in rows} == set(range(100)),
            f"{device} sample coverage mismatch: {case}",
        )
        require(
            all(row["backend"] == backend and row["scale_id"] == scale_id for row in rows),
            f"{device} timing identity mismatch: {case}",
        )
    cpu_digests = {row["sample_index"]: row["input_digest"] for row in by_device["cpu"]}
    gpu_digests = {row["sample_index"]: row["input_digest"] for row in by_device["gpu"]}
    require(cpu_digests == gpu_digests, f"CPU/GPU input digest mismatch: {case}")

    for row in accuracy:
        require(row["accuracy_status"] == "PASS", f"accuracy failure: {case}")
        require(row["semantic_check"].endswith(":PASS"), f"semantic failure: {case}")
        if row["output_kind"] == "discrete":
            require(row["exact_match"] == "true", f"exact match failure: {case}")
            require(row["mismatch_count"] == "0", f"mismatch count failure: {case}")
            require(row["max_mismatch_count"] == "0", f"mismatch limit failure: {case}")
            require(
                all(row[name] == "NA" for name in (
                    "mse", "mse_max", "rmse", "rmse_max",
                    "relative_l2", "relative_l2_max",
                    "relative_linf", "relative_linf_max", "relative_floor",
                )),
                f"discrete numeric fields must be NA: {case}",
            )
        else:
            for actual, limit in (
                ("mse", "mse_max"),
                ("rmse", "rmse_max"),
                ("relative_l2", "relative_l2_max"),
                ("relative_linf", "relative_linf_max"),
            ):
                require(
                    finite_at_most(row, actual, limit),
                    f"threshold failure: {case}/{row['output_name']}/{actual}",
                )

    summary_json = case / f"task2_{step_name}_summary.json"
    summary_text = case / f"task2_{step_name}_summary.txt"
    full_log = case / f"task2_{step_name}_full.log"
    for path in (summary_json, summary_text, full_log):
        require(path.is_file() and path.stat().st_size > 0, f"missing result file: {path}")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    require(summary["status"] == "PASS", f"summary status mismatch: {case}")
    require(summary["backend"] == backend, f"summary backend mismatch: {case}")
    require(summary["scale_id"] == scale_id, f"summary scale mismatch: {case}")
    require(summary["run_type"] == "formal", f"summary run type mismatch: {case}")
    require(summary["warmup_runs"] == 20, f"summary warmup mismatch: {case}")
    require(summary["measured_runs"] == 100, f"summary measured mismatch: {case}")
    if step_name == "pipeline":
        pipeline = read_csv(case / f"task2_pipeline_summary_{suffix}")
        require(len(pipeline) == 14, f"pipeline summary row count mismatch: {case}")
        require(summary["semantic_gates_passed"] == 16, f"pipeline gate mismatch: {case}")
        require(summary["pipeline_total_direct"] is True, f"pipeline timing mismatch: {case}")
        require(
            summary["cpu_correlation_method"] == "direct_parallel_exact",
            f"pipeline CPU correlation method mismatch: {case}",
        )
        require(
            isinstance(summary["cpu_correlation_workers"], int)
            and summary["cpu_correlation_workers"] >= 1,
            f"pipeline CPU correlation worker count mismatch: {case}",
        )
        require(
            summary["resource_sampling_mode"] == "same_pipeline_execution",
            f"pipeline resource sampling mode mismatch: {case}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("fft_thrust", "dlfft"))
    parser.add_argument(
        "--step", choices=("step1", "step2", "step3", "step4", "step5", "step6", "pipeline"),
        default="step1"
    )
    args = parser.parse_args()

    expected_target = f"task2_{args.step}_benchmark"
    root = args.result_root
    require(root.is_dir(), f"missing result root: {root}")
    manifest = read_csv(root / "matrix_status.csv")
    require(len(manifest) == EXPECTED_CASES, f"manifest rows={len(manifest)}, expected=110")
    require(
        all(row["exit_code"] == "0" and row["status"] == "PASS" for row in manifest),
        "manifest contains failed rows",
    )
    require(
        all(row["target"] == expected_target for row in manifest),
        "manifest target mismatch",
    )

    summary_json = root / "deep_verify_summary.json"
    summary_text = root / "deep_verify_summary.txt"
    require(not summary_json.exists(), f"refusing to overwrite: {summary_json}")
    require(not summary_text.exists(), f"refusing to overwrite: {summary_text}")

    for index, row in enumerate(manifest, start=1):
        expected_case = root / "cases" / row["scale_id"]
        recorded_case = Path(row["result_path"])
        require(recorded_case == expected_case, f"manifest result path mismatch: {row['scale_id']}")
        verify_case(
            expected_case,
            row["scale_id"],
            args.backend,
            args.step,
        )
        if index % 10 == 0 or index == EXPECTED_CASES:
            print(f"[DEEP VERIFY] checked={index}/110 scale={row['scale_id']}")

    payload = {
        "status": "PASS",
        "target": expected_target,
        "backend": args.backend,
        "cases": EXPECTED_CASES,
        "warmup_runs": 20,
        "measured_runs": 100,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_text.write_text(
        "\n".join(f"{key}={value}" for key, value in payload.items()) + "\n",
        encoding="utf-8",
    )
    print(
        f"TASK2 {args.step.upper()} {args.backend.upper()} DEEP VERIFY PASS "
        f"cases={EXPECTED_CASES}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"TASK2 FORMAL MATRIX DEEP VERIFY FAIL: {error}")
        raise SystemExit(1)
