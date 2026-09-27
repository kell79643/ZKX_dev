#!/usr/bin/env python3
"""严格深检一个已完成的 Task1 pipeline 110-case 正式矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED_ACCURACY_TARGETS = {
    "Task1.step1.waveform",
    "Task1.step1.noiseless_echo",
    "Task1.step1.noise",
    "Task1.step1.echo",
    "Task1.step2.compressed",
    "Task1.step3.range_doppler",
    "Task1.step4.power",
    "Task1.step4.cfar_threshold",
    "Task1.step4.cfar_alpha",
    "Task1.step4.detections",
    "Task1.step5.ambiguity_2d",
    "Task1.step5.ambiguity_delay",
    "Task1.step5.ambiguity_doppler",
}
EXPECTED_SCOPES = {"step1", "step2", "step3", "step4", "step5", "pipeline_total"}
EXPECTED_COMPONENTS = EXPECTED_SCOPES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_finite(value: str, label: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite {label}: {value}")
    return parsed


def verify_case(
    case: Path,
    scale_id: str,
    backend: str,
) -> None:
    suffix = f"{backend}_formal.csv"
    main = read_csv(case / f"task1_benchmark_{suffix}")
    timing = read_csv(case / f"task1_iteration_timing_{suffix}")
    memory = read_csv(case / f"task1_memory_trace_{suffix}")
    accuracy = read_csv(case / f"task1_accuracy_details_{suffix}")
    components = read_csv(case / f"task1_pipeline_summary_{suffix}")
    summary_path = case / "task1_pipeline_summary.json"
    summary_text_path = case / "task1_pipeline_summary.txt"
    full_log_path = case / "task1_pipeline_full.log"
    for path in (summary_path, summary_text_path, full_log_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty evidence: {path}")

    require(len(main) == 2, f"main row count mismatch: {case}")
    require(len(timing) == 200, f"timing row count mismatch: {case}")
    require(len(memory) == 3600, f"memory row count mismatch: {case}")
    require(len(accuracy) == 13, f"accuracy row count mismatch: {case}")
    require(len(components) == 12, f"component row count mismatch: {case}")

    common_contract = {
        "backend": backend,
        "run_type": "formal",
        "scale_id": scale_id,
    }
    for row in main:
        for key, expected in common_contract.items():
            require(row[key] == expected, f"main {key} mismatch: {case}")
        require(row["warmup_runs"] == "20" and row["measured_runs"] == "100",
                f"main repetition mismatch: {case}")
        require(row["accuracy_status"] == "PASS" and row["return_code"] == "0",
                f"main status mismatch: {case}")
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "std_ms", "cv"):
            require_finite(row[metric], f"{case}/{row['device']}/{metric}")
        for metric in ("mse", "rmse", "relative_l2", "relative_linf"):
            require_finite(row[metric], f"{case}/{metric}")
        require(row["exact_match"] == "true" and row["mismatch_count"] == "0",
                f"same-input detection mismatch: {case}")

    by_device = {device: [row for row in main if row["device"] == device]
                 for device in ("cpu", "gpu")}
    require(all(len(rows) == 1 for rows in by_device.values()), f"main device coverage mismatch: {case}")
    require(by_device["cpu"][0]["cpu_gpu_speedup"] == "NA", f"CPU speedup placement mismatch: {case}")
    require(require_finite(by_device["gpu"][0]["cpu_gpu_speedup"], f"{case}/speedup") >= 0.0,
            f"negative speedup: {case}")

    timing_by_device = {
        device: [row for row in timing if row["device"] == device]
        for device in ("cpu", "gpu")
    }
    require(all(len(rows) == 100 for rows in timing_by_device.values()),
            f"timing device coverage mismatch: {case}")
    for device, rows in timing_by_device.items():
        require({int(row["sample_index"]) for row in rows} == set(range(100)),
                f"timing sample coverage mismatch: {case}/{device}")
        require(all(row["synchronized"] == "true" for row in rows),
                f"unsynchronized timing row: {case}/{device}")
        for row in rows:
            require_finite(row["latency_ms"], f"{case}/{device}/latency_ms")
    cpu_digests = {row["sample_index"]: row["input_digest"] for row in timing_by_device["cpu"]}
    gpu_digests = {row["sample_index"]: row["input_digest"] for row in timing_by_device["gpu"]}
    require(cpu_digests == gpu_digests, f"CPU/GPU input digest mismatch: {case}")

    for label, rows in (("timing", timing), ("memory", memory),
                        ("component", components)):
        for row in rows:
            for key, expected in common_contract.items():
                require(row[key] == expected,
                        f"{label} {key} mismatch: {case}")

    expected_memory_keys = {
        (str(sample), device, scope, phase)
        for sample in range(100)
        for device in ("cpu", "gpu")
        for scope in EXPECTED_SCOPES
        for phase in ("before", "peak", "after")
    }
    actual_memory_keys = {
        (row["sample_index"], row["device"], row["resource_scope"], row["trace_phase"])
        for row in memory
    }
    require(actual_memory_keys == expected_memory_keys, f"memory coverage mismatch: {case}")
    require({row["resource_scope"] for row in memory} == EXPECTED_SCOPES,
            f"memory scope mismatch: {case}")

    require({row["target"] for row in accuracy} == EXPECTED_ACCURACY_TARGETS,
            f"accuracy target set mismatch: {case}")
    for row in accuracy:
        require(row["accuracy_status"] == "PASS" and row["semantic_check"].endswith(":PASS"),
                f"accuracy status mismatch: {case}/{row['target']}")
        if row["output_kind"] == "discrete":
            require(row["exact_match"] == "true" and row["mismatch_count"] == "0",
                    f"discrete mismatch: {case}/{row['target']}")
        else:
            for actual, maximum in (
                ("mse", "mse_max"),
                ("rmse", "rmse_max"),
                ("relative_l2", "relative_l2_max"),
                ("relative_linf", "relative_linf_max"),
            ):
                value = require_finite(row[actual], f"{case}/{row['target']}/{actual}")
                limit = require_finite(row[maximum], f"{case}/{row['target']}/{maximum}")
                require(value <= limit, f"threshold failure: {case}/{row['target']}/{actual}")

    require({row["component"] for row in components} == EXPECTED_COMPONENTS,
            f"component set mismatch: {case}")
    require({row["device"] for row in components} == {"cpu", "gpu"},
            f"component device mismatch: {case}")
    for row in components:
        require(row["measured_runs"] == "100", f"component repetition mismatch: {case}")
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "std_ms", "cv"):
            require_finite(row[metric], f"{case}/{row['device']}/{row['component']}/{metric}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["status"] == "PASS" and summary["backend"] == backend,
            f"summary status/backend mismatch: {case}")
    require(summary["scale_id"] == scale_id and summary["warmup_runs"] == 20 and
            summary["measured_runs"] == 100, f"summary identity/repetition mismatch: {case}")
    require(summary["semantic_gates_passed"] == 5,
            f"summary semantic gate mismatch: {case}")
    require(summary["resource_sampling_mode"] == "same_pipeline_execution" and
            summary["resource_sampling_overhead_excluded"] is True,
            f"summary resource sampling mismatch: {case}")
    require(summary["same_input_detection_exact_match"] is True and
            summary["same_input_detection_mismatch_count"] == 0,
            f"summary same-input mismatch: {case}")

    layered = set()
    for row in main:
        total = int(row["independent_pipeline_mismatch_count"])
        valid = int(row["independent_pipeline_valid_mismatch_count"])
        boundary = int(row["independent_pipeline_boundary_mismatch_count"])
        exact = row["independent_pipeline_exact_match"]
        require(min(total, valid, boundary) >= 0 and total == valid + boundary,
                f"independent mismatch partition invalid: {case}")
        require(exact == ("true" if total == 0 else "false"),
                f"independent exact flag mismatch: {case}")
        layered.add((total, valid, boundary, exact))
    require(len(layered) == 1, f"CPU/GPU layered diagnostics mismatch: {case}")
    total, valid, boundary, _ = next(iter(layered))
    require(summary["independent_pipeline_detection_mismatch_count"] == total and
            summary["independent_pipeline_valid_mismatch_count"] == valid and
            summary["independent_pipeline_boundary_mismatch_count"] == boundary,
            f"summary layered diagnostics mismatch: {case}")

    summary_text = summary_text_path.read_text(encoding="utf-8")
    full_log = full_log_path.read_text(encoding="utf-8")
    for marker in ("status=PASS", "semantic_gates=5/5", "resource_sampling_mode=same_pipeline_execution"):
        require(marker in summary_text, f"summary text marker missing: {case}/{marker}")
    for marker in (
        "timing=pipeline_total measured directly;not=sum(step timings)",
        "resource_sampling_mode=same_pipeline_execution",
        "resource_sampling_overhead_excluded=true",
        "status=PASS return_code=0",
    ):
        require(marker in full_log, f"full log marker missing: {case}/{marker}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--backend", choices=("fft_thrust", "dlfft"), required=True)
    args = parser.parse_args()

    root = Path(args.result_root).resolve(strict=True)
    for output in (root / "deep_verify_summary.json", root / "deep_verify_summary.txt"):
        require(not output.exists(), f"deep verify output already exists: {output}")
    summary_text = (root / "matrix_summary.txt").read_text(encoding="utf-8")
    for marker in (
        "status=PASS",
        "target=task1_pipeline_benchmark",
        f"backend={args.backend}",
        "cases=110",
        "warmup_runs=20",
        "measured_runs=100",
    ):
        require(marker in summary_text, f"matrix summary marker missing: {marker}")

    manifest = read_csv(root / "matrix_status.csv")
    require(len(manifest) == 110, f"manifest case count mismatch: {len(manifest)}")
    require(len({row["scale_id"] for row in manifest}) == 110,
            "manifest scale IDs are not unique")
    require(all(row["target"] == "task1_pipeline_benchmark" and
                row["backend"] == args.backend and row["exit_code"] == "0" and
                row["status"] == "PASS" for row in manifest),
            "manifest contains identity or status mismatch")
    case_directories = {path.name for path in (root / "cases").iterdir() if path.is_dir()}
    driver_logs = {path.stem for path in (root / "driver_logs").glob("*.log") if path.is_file()}
    expected_scales = {row["scale_id"] for row in manifest}
    require(case_directories == expected_scales, "case directory coverage mismatch")
    require(driver_logs == expected_scales, "driver log coverage mismatch")

    for index, row in enumerate(manifest, start=1):
        case = root / "cases" / row["scale_id"]
        require(Path(row["result_path"]).resolve() == case.resolve(),
                f"manifest result path mismatch: {row['scale_id']}")
        verify_case(
            case,
            row["scale_id"],
            args.backend,
        )
        if index % 10 == 0 or index == 110:
            print(f"[DEEP VERIFY] checked={index}/110 scale={row['scale_id']}")

    payload = {
        "status": "PASS",
        "target": "task1_pipeline_benchmark",
        "backend": args.backend,
        "cases": 110,
        "warmup_runs": 20,
        "measured_runs": 100,
        "resource_sampling_mode": "same_pipeline_execution",
        "resource_sampling_overhead_excluded": True,
    }
    (root / "deep_verify_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (root / "deep_verify_summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in payload.items()) + "\n",
        encoding="utf-8",
    )
    print("TASK1 PIPELINE DEEP VERIFY PASS cases=110")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
