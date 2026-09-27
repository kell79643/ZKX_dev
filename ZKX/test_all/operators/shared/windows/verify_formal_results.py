#!/usr/bin/env python3
"""Deep-verify one Remaining operator case windows operator formal result matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


DTYPES = {"FP32", "FP16", "INT32", "INT16", "INT8"}
ORDERS = {"10^2", "10^3", "10^4"}
DEVICES = {"CPU", "GPU"}
PHASES = [
    "before",
    "allocate",
    "h2d",
    "execute_sync",
    "d2h",
    "release_sync",
    "after",
]
DTYPE_EVIDENCE = {
    "chebwin": ("actual_typed_scalar_input", "1"),
    "general_cosine": ("actual_typed_array_input", "3"),
    "general_gaussian": ("actual_typed_scalar_inputs", "2"),
    "kaiser": ("actual_typed_scalar_input", "1"),
    "parzen": ("request_variant_no_typed_data_input", "0"),
    "taylor": ("actual_typed_scalar_input", "1"),
    "triang": ("request_variant_no_typed_data_input", "0"),
}
INPUT_NAMES = {
    "chebwin": ["window_positions", "attenuation_db"],
    "general_cosine": ["window_positions", "coefficient_values"],
    "general_gaussian": ["window_positions", "power", "width"],
    "kaiser": ["window_positions", "beta"],
    "parzen": ["window_positions"],
    "taylor": ["window_positions", "nbar", "sidelobe_level_db", "normalize"],
    "triang": ["window_positions"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    require(len(matches) == 1, f"{directory}: expected one {pattern}, got {len(matches)}")
    require(matches[0].stat().st_size > 0, f"{matches[0]}: empty file")
    return matches[0]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        result = list(csv.DictReader(source))
    require(result, f"{path}: no data rows")
    return result


def finite(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{label}: not numeric: {value}") from error
    require(math.isfinite(number), f"{label}: not finite: {value}")
    return number


def common(
    row: dict[str, str], *, operator: str, backend: str, commit: str, dirty: str
) -> None:
    require(row["operator_name"] == operator, "operator mismatch")
    require(row["backend"] == backend, "backend mismatch")
    require(row["git_commit"] == commit, "git_commit mismatch")
    require(row["git_dirty"] == dirty, "git_dirty mismatch")
    require(row["run_type"] == "formal", "run_type must be formal")
    require(row["status"] == "PASS", "status must be PASS")
    require(row["error_code"] == "OK", "error_code must be OK")


def verify_case(
    directory: Path,
    *,
    operator: str,
    backend: str,
    batch_id: str,
    commit: str,
    dirty: str,
    warmup: int,
    measured: int,
) -> tuple[str, str, str]:
    main_path = one(directory, f"operator_main_results_{operator}_{backend}_*.csv")
    timing_path = one(directory, f"operator_timing_samples_{operator}_{backend}_*.csv")
    memory_path = one(directory, f"operator_windows_{operator}_memory_trace_{backend}.csv")
    module_path = one(
        directory,
        f"windows_batch{batch_id[1:]}_module_summary_{backend}_*.csv",
    )
    report_path = one(directory, f"operator_{operator}_report.json")
    summary_path = one(directory, f"operator_{operator}_summary.txt")
    log_path = one(directory, f"operator_{operator}_full.log")
    dtype_path = one(directory, f"operator_dTYPE_EVIDENCE_{operator}_{backend}_*.csv")
    input_path = one(directory, f"operator_input_evidence_{operator}_{backend}_*.csv")

    main = rows(main_path)
    require(len(main) == 2, f"{main_path}: expected 2 rows")
    require({row["device"] for row in main} == DEVICES, f"{main_path}: device set")
    for row in main:
        common(row, operator=operator, backend=backend, commit=commit, dirty=dirty)
        require(int(row["warmup_runs"]) == warmup, "main warmup mismatch")
        require(int(row["measured_runs"]) == measured, "main measured mismatch")
        for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "std_ms", "cv"):
            finite(row[field], f"{main_path}:{row['device']}:{field}")
        for field in ("mse", "rmse", "relative_l2", "relative_linf"):
            finite(row[field], f"{main_path}:{row['device']}:{field}")
        require(row["accuracy_status"] == "PASS", "accuracy_status mismatch")
        if row["device"] == "CPU":
            require(row["cpu_gpu_speedup"] == "NA", "CPU speedup must be NA")
        else:
            finite(row["cpu_gpu_speedup"], f"{main_path}:GPU:speedup")

    dtype = main[0]["dtype"]
    order = main[0]["order_of_magnitude"]
    require(dtype in DTYPES, f"unexpected dtype {dtype}")
    require(order in ORDERS, f"unexpected order {order}")

    dtype_rows = rows(dtype_path)
    require(len(dtype_rows) == 1, f"{dtype_path}: expected one row")
    dtype_row = dtype_rows[0]
    require(operator in DTYPE_EVIDENCE, f"no dtype evidence contract for {operator}")
    expected_semantics, expected_count = DTYPE_EVIDENCE[operator]
    require(dtype_row["requested_dtype"] == dtype, "dtype evidence request mismatch")
    require(dtype_row["dtype_semantics"] == expected_semantics,
            "dtype evidence semantics mismatch")
    require(dtype_row["typed_input_count"] == expected_count,
            "dtype evidence typed input count mismatch")
    require(dtype_row["typed_input_manifest"], "dtype evidence manifest empty")
    require(dtype_row["actual_input_digest"], "dtype evidence digest empty")
    require(
        hashlib.sha256(dtype_row["typed_input_manifest"].encode("utf-8")).hexdigest()
        == dtype_row["actual_input_digest"],
        "dtype evidence digest does not hash manifest",
    )
    require(dtype_row["status"] == "PASS" and dtype_row["error_code"] == "OK",
            "dtype evidence status mismatch")
    input_content = json.loads(dtype_row["input_content_json"])
    case_parameters = json.loads(dtype_row["case_parameters_json"])
    require(operator in INPUT_NAMES, f"no input content contract for {operator}")
    require([item["input_name"] for item in input_content] == INPUT_NAMES[operator],
            "input content JSON names mismatch")
    require(case_parameters["requested_dtype"] == dtype, "case parameter dtype mismatch")
    require(str(case_parameters["sym"]).lower() in {"true", "false"},
            "case parameter sym missing")
    require(int(case_parameters["length"]) > 0, "case parameter length invalid")

    input_rows = rows(input_path)
    require([row["input_name"] for row in input_rows] == INPUT_NAMES[operator],
            f"{input_path}: input names mismatch")
    comparable_fields = ["input_name", "shape", "dtype", "element_count",
                         "generator", "preview_heacuda_api_tail2"]
    require([{key: row[key] for key in comparable_fields} for row in input_rows]
            == [{key: row[key] for key in comparable_fields} for row in input_content],
            f"{input_path}: CSV/JSON input content mismatch")
    for row in input_rows:
        require(row["operator_name"] == operator and row["case_id"] == main[0]["case_id"],
                f"{input_path}: identity mismatch")
        require(row["status"] == "PASS" and row["error_code"] == "OK",
                f"{input_path}: status mismatch")
        count = int(row["element_count"])
        require(count > 0 and row["shape"] and row["dtype"] and row["generator"],
                f"{input_path}: incomplete input description")
        preview = row["preview_heacuda_api_tail2"]
        if row["shape"] == "scalar":
            require(preview.startswith("value="), f"{input_path}: scalar preview missing")
        else:
            values = preview.split(";")
            require(len(values) == min(count, 6), f"{input_path}: preview count mismatch")
            indices = [int(re.fullmatch(r"i=([0-9]+):.+", value).group(1)) for value in values]
            expected_indices = list(range(count)) if count <= 6 else [0, 1, 2, 3, count - 2, count - 1]
            require(indices == expected_indices, f"{input_path}: preview indices mismatch")

    timing = rows(timing_path)
    require(len(timing) == 2 * measured, f"{timing_path}: row count")
    input_digests: set[str] = set()
    for device in DEVICES:
        selected = [row for row in timing if row["device"] == device]
        require(len(selected) == measured, f"{timing_path}:{device}: count")
        require({int(row["sample_index"]) for row in selected} == set(range(measured)),
                f"{timing_path}:{device}: sample indexes")
        require(len({row["output_digest"] for row in selected}) == 1,
                f"{timing_path}:{device}: output digest unstable")
        for row in selected:
            common(row, operator=operator, backend=backend, commit=commit, dirty=dirty)
            require(int(row["warmup_runs"]) == warmup, "timing warmup mismatch")
            require(int(row["measured_runs"]) == measured, "timing measured mismatch")
            require(row["synchronized"] == "true", "timing must be synchronized")
            finite(row["latency_ms"], f"{timing_path}:{device}:latency")
            require(row["input_digest"], "empty input digest")
            require(row["output_digest"], "empty output digest")
            input_digests.add(row["input_digest"])
    require(len(input_digests) == 1, f"{timing_path}: CPU/GPU input digest mismatch")
    require(input_digests == {dtype_row["actual_input_digest"]},
            f"{timing_path}: digest does not match dtype evidence")

    memory = rows(memory_path)
    require(len(memory) == 14, f"{memory_path}: expected 14 rows")
    for device in DEVICES:
        selected = [row for row in memory if row["device"] == device]
        selected.sort(key=lambda row: int(row["phase_index"]))
        require([row["trace_phase"] for row in selected] == PHASES,
                f"{memory_path}:{device}: phases")
        for row in selected:
            common(row, operator=operator, backend=backend, commit=commit, dirty=dirty)
            require(row["cpu_live_heap_bytes"] != "NA", "heap unavailable")
            require(row["rss_bytes"] != "NA", "rss unavailable")
            if device == "GPU":
                require(row["gpu_used_bytes"] != "NA", "GPU memory unavailable")

    module = rows(module_path)
    require(len(module) == 1, f"{module_path}: expected one row")
    module_row = module[0]
    require(module_row["operator_name"] == operator, "module operator mismatch")
    require(module_row["backend"] == backend, "module backend mismatch")
    require(module_row["status"] == "PASS" and module_row["error_code"] == "OK",
            "module status mismatch")
    require(int(module_row["warmup_runs"]) == warmup, "module warmup mismatch")
    require(int(module_row["measured_runs"]) == measured, "module measured mismatch")
    for field in ("cpu_mean_ms", "gpu_mean_ms", "cpu_gpu_speedup", "mse", "rmse", "relative_l2", "relative_linf"):
        finite(module_row[field], f"{module_path}:{field}")
    symmetric = module_row["sym"]
    require(symmetric in {"true", "false"}, "module sym mismatch")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["operator_name"] == operator, "report operator mismatch")
    require(report["backend"] == backend, "report backend mismatch")
    require(report["status"] == "PASS" and report["error_code"] == "OK",
            "report status mismatch")
    require(report["warmup_runs"] == warmup and report["measured_runs"] == measured,
            "report run counts mismatch")
    require(report["dtype_semantics"] == expected_semantics,
            "report dtype semantics mismatch")
    require(str(report["typed_input_count"]) == expected_count,
            "report typed input count mismatch")
    require(str(report["sym"]).lower() == symmetric, "report sym mismatch")
    require(report["input_content"] == input_content, "report input content mismatch")
    require(report["case_parameters"] == case_parameters, "report case parameters mismatch")
    for field in ("mse", "rmse", "relative_l2", "relative_linf"):
        finite(str(report[field]), f"{report_path}:{field}")

    summary = summary_path.read_text(encoding="utf-8")
    full_log = log_path.read_text(encoding="utf-8")
    for token in ("status=PASS", f"operator={operator}", f"backend={backend}", "error_code=OK"):
        require(token in summary, f"{summary_path}: missing {token}")
        require(token in full_log, f"{log_path}: missing {token}")
    for token in ("input_content_json=", "case_parameters_json="):
        require(token in summary, f"{summary_path}: missing {token}")
    for token in ("[remaining_operator_case] input content", "[remaining_operator_case] case parameters"):
        require(token in full_log, f"{log_path}: missing {token}")

    return dtype, order, symmetric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--batch-id", default="b01")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", required=True, choices=("true", "false"))
    parser.add_argument("--expected-cases", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    status = "FAIL"
    detail = "not started"
    try:
        require(len(args.batch_id) == 3 and args.batch_id[0] == "b"
                and args.batch_id[1:].isdigit(), "batch-id must match bNN")
        main_pattern = f"operator_main_results_{args.operator}_{args.backend}_*.csv"
        directories = sorted(
            path for path in args.formal_root.iterdir()
            if path.is_dir() and any(path.glob(main_pattern))
        )
        require(len(directories) == args.expected_cases,
                f"formal directories: expected {args.expected_cases}, got {len(directories)}")
        coverage = {
            verify_case(
                directory,
                operator=args.operator,
                backend=args.backend,
                batch_id=args.batch_id,
                commit=args.git_commit,
                dirty=args.git_dirty,
                warmup=args.warmup,
                measured=args.measured,
            )
            for directory in directories
        }
        expected = {(dtype, order, symmetric) for dtype in DTYPES for order in ORDERS
                    for symmetric in ("true", "false")}
        require(coverage == expected,
                f"coverage mismatch missing={sorted(expected - coverage)} extra={sorted(coverage - expected)}")
        status = "PASS"
        detail = (
            f"batch={args.batch_id} operator={args.operator} backend={args.backend} cases={len(directories)} "
            f"warmup={args.warmup} measured={args.measured} request_variant_coverage=5x3x2 "
            f"dtype_semantics={DTYPE_EVIDENCE[args.operator][0]} input_preview=heacuda_api_tail2"
        )
    except Exception as error:  # Preserve a machine-readable failure artifact.
        detail = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY {status} {detail}\n",
            encoding="utf-8",
        )
        print(args.output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
