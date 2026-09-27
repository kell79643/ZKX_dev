#!/usr/bin/env python3
"""Audit Stage07 Batch03 wavelets formal evidence without rerunning operators."""

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
import sys


OPERATORS = {
    "cwt": ("not_applicable", "Task2", "Step4"),
    "ricker": ("not_applicable", "Task2", "Step4"),
}
PHASES = ["before", "allocate", "h2d", "execute_sync", "d2h", "release_sync", "after"]
FLOAT_METRICS = ["mse", "rmse", "relative_l2", "relative_linf"]
TABLE_MARKERS = [
    "[stage07] identity and status",
    "[stage07] timing detail",
    "[stage07] accuracy detail",
    "[stage07] resource detail",
    "[stage07] input content",
    "[stage07] case parameters",
    "[stage07] input and output identity",
    "[stage07] dtype evidence",
    "[stage07] evidence files",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--driver-log-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", choices=("true", "false"), required=True)
    parser.add_argument("--verifier-commit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with pathlib.Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_file(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern}, found {len(matches)}")
    return matches[0]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, field):
    number = float(value)
    require(math.isfinite(number), f"{field} is not finite")
    return number


def close(actual, expected, field):
    tolerance = max(1.0e-12, abs(expected) * 1.0e-10)
    require(abs(actual - expected) <= tolerance, f"{field} mismatch: {actual} != {expected}")


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timing_summary(values):
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "mean_ms": mean,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": std,
        "cv": std / mean,
    }


def expected_cases(config):
    cases = {}
    for operator in config["operators"]:
        name = operator["operator_name"]
        require(name in OPERATORS, f"unexpected configured operator {name}")
        for scale in operator["scales"]:
            key = (name, scale["scale_id"])
            require(key not in cases, f"duplicate configured case {key}")
            cases[key] = scale
    require(len(cases) == 30, f"expected 30 configured cases, found {len(cases)}")
    return cases


def thresholds_by_key(document):
    result = {}
    for entry in document["entries"]:
        key = (entry["target"], entry["dtype"])
        require(key not in result, f"duplicate threshold {key}")
        result[key] = entry
    return result


def selected_result_directories(driver_log_root, formal_root):
    log_root = pathlib.Path(driver_log_root)
    logs = sorted(log_root.glob("*/*.log"))
    require(len(logs) == 30, f"expected 30 driver logs, found {len(logs)}")
    selected = []
    records = []
    pattern = re.compile(r"^OPERATOR_CASE_WAVELET_CASE PASS .* results=(\S+)$", re.MULTILINE)
    formal_resolved = pathlib.Path(formal_root).resolve()
    for log in logs:
        matches = pattern.findall(log.read_text(encoding="utf-8"))
        require(len(matches) == 1, f"driver log {log} has {len(matches)} PASS result paths")
        directory = pathlib.Path(matches[0]).resolve()
        require(directory.parent == formal_resolved, f"driver result outside formal root: {directory}")
        require(directory.is_dir(), f"driver result directory missing: {directory}")
        selected.append(directory)
        records.append({"driver_log": str(log.resolve()), "result_directory": str(directory)})
    require(len(set(selected)) == 30, "driver logs do not select 30 unique result directories")
    return selected, records


def identity_check(row, args, operator, scale, dtype, backend, device, task, step):
    expected = {
        "run_type": "formal",
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
        "target_kind": "operator",
        "target": operator,
        "task_name": task,
        "step_name": step,
        "operator_name": operator,
        "scale_id": scale["scale_id"],
        "dtype": dtype,
        "device": device,
        "backend": backend,
        "timing_scope": "compatibility/end-to-end",
        "status": "PASS",
        "error_code": "OK",
    }
    for field, value in expected.items():
        require(row.get(field) == value, f"{field} mismatch for {device}: {row.get(field)} != {value}")


def audit_case(directory, args, scale, threshold):
    report_path = unique_file(directory, "operator_*_report.json")
    report = load_json(report_path)
    operator = report["operator_name"]
    backend, task, step = OPERATORS[operator]
    dtype = scale["inputs"][0]["dtype"].replace("Complex", "")
    require(report["scale_id"] == scale["scale_id"], "report scale mismatch")
    require(report["requested_dtype"] == dtype, "report dtype mismatch")
    require(report["backend"] == backend, "report backend mismatch")
    require(report["warmup_runs"] == 20 and report["measured_runs"] == 100, "report run count mismatch")
    require(report["status"] == "PASS" and report["error_code"] == "OK", "report status mismatch")

    main_path = unique_file(directory, "operator_main_results_*.csv")
    timing_path = unique_file(directory, "operator_timing_samples_*.csv")
    memory_path = unique_file(directory, "operator_wavelets_*_memory_trace_*.csv")
    dtype_path = unique_file(directory, "operator_dtype_evidence_*.csv")
    log_path = unique_file(directory, "operator_*_full.log")
    unique_file(directory, "operator_*_summary.txt")
    unique_file(directory, "*_probe_raw.json")

    main_rows = read_csv(main_path)
    require(len(main_rows) == 2, f"main CSV rows={len(main_rows)}")
    by_device = {row["device"]: row for row in main_rows}
    require(set(by_device) == {"CPU", "GPU"}, "main CSV must contain CPU and GPU")
    for device, row in by_device.items():
        identity_check(row, args, operator, scale, dtype, backend, device, task, step)
        require(row["warmup_runs"] == "20" and row["measured_runs"] == "100", "main run count mismatch")

    timing_rows = read_csv(timing_path)
    require(len(timing_rows) == 200, f"timing rows={len(timing_rows)}")
    input_digests = set()
    output_digests = {}
    summaries = {}
    for device in ("CPU", "GPU"):
        rows = [row for row in timing_rows if row["device"] == device]
        require(len(rows) == 100, f"{device} timing rows={len(rows)}")
        require({int(row["sample_index"]) for row in rows} == set(range(100)), f"{device} sample indices")
        for row in rows:
            identity_check(row, args, operator, scale, dtype, backend, device, task, step)
            require(row["warmup_runs"] == "20" and row["measured_runs"] == "100", "timing run count")
            require(row["synchronized"] == "true", "timing sample not synchronized")
            require(len(row["input_digest"]) == 64 and len(row["output_digest"]) == 64, "digest length")
            input_digests.add(row["input_digest"])
            output_digests.setdefault(device, set()).add(row["output_digest"])
        require(len(output_digests[device]) == 1, f"{device} output digest changed across repetitions")
        values = [finite_number(row["latency_ms"], "latency_ms") for row in rows]
        require(all(value > 0 for value in values), f"{device} latency must be positive")
        summaries[device] = timing_summary(values)
        for field, expected in summaries[device].items():
            close(finite_number(by_device[device][field], field), expected, f"{device}.{field}")
    require(len(input_digests) == 1, "CPU/GPU input_digest mismatch")
    speedup = summaries["CPU"]["mean_ms"] / summaries["GPU"]["mean_ms"]
    require(by_device["CPU"]["cpu_gpu_speedup"] == "NA", "CPU speedup must be NA")
    close(float(by_device["GPU"]["cpu_gpu_speedup"]), speedup, "GPU speedup")

    memory_rows = read_csv(memory_path)
    require(len(memory_rows) == 14, f"memory rows={len(memory_rows)}")
    for device in ("CPU", "GPU"):
        rows = sorted((row for row in memory_rows if row["device"] == device), key=lambda row: int(row["phase_index"]))
        require(len(rows) == 7, f"{device} memory phases={len(rows)}")
        require([row["trace_phase"] for row in rows] == PHASES, f"{device} phase names")
        require([int(row["phase_index"]) for row in rows] == list(range(7)), f"{device} phase indices")
        for row in rows:
            identity_check(row, args, operator, scale, dtype, backend, device, task, step)
        for csv_field, summary_prefix in (("cpu_live_heap_bytes", "cpu_heap"), ("rss_bytes", "rss"), ("gpu_used_bytes", "gpu")):
            values = [row[csv_field] for row in rows]
            if device == "CPU" and csv_field == "gpu_used_bytes":
                require(set(values) == {"NA"}, "CPU GPU-memory samples must be NA")
                continue
            require(all(value != "NA" for value in values), f"{device}.{csv_field} contains NA")
            numbers = [int(value) for value in values]
            expected = {
                "before": numbers[0], "after": numbers[-1],
                "peak": max(numbers), "delta": numbers[-1] - numbers[0],
            }
            for name, value in expected.items():
                field = f"{summary_prefix}_{name}_bytes"
                require(int(by_device[device][field]) == value, f"{device}.{field} mismatch")

    for field in FLOAT_METRICS:
        value = finite_number(by_device["GPU"][field], field)
        require(value <= float(threshold[f"{field}_max"]), f"{field} threshold failed")
        close(value, float(report[field]), f"report.{field}")
    require(by_device["GPU"]["accuracy_status"] == "PASS", "accuracy_status")
    require(by_device["GPU"]["exact_match"] == "NA" and by_device["GPU"]["mismatch_count"] == "NA", "numeric waveform fields")

    dtype_rows = read_csv(dtype_path)
    require(len(dtype_rows) == 1, f"dtype rows={len(dtype_rows)}")
    dtype_row = dtype_rows[0]
    require(dtype_row["operator_name"] == operator and dtype_row["requested_dtype"] == dtype, "dtype identity")
    require(dtype_row["config_input_dtype"] == scale["inputs"][0]["dtype"], "config input dtype")
    require(dtype_row["actual_input_digest"] in input_digests, "dtype/timing digest mismatch")
    require(dtype_row["status"] == "PASS" and dtype_row["error_code"] == "OK", "dtype status")
    content = json.loads(dtype_row["input_content_json"])
    parameters = json.loads(dtype_row["case_parameters_json"])
    require(parameters == scale["parameters"], "case parameter evidence mismatch")
    require(content == report["input_content"] and parameters == report["case_parameters"], "report input evidence mismatch")
    expected_names = [item["input_name"] for item in scale["inputs"]]
    require([item["input_name"] for item in content] == expected_names, "input names")
    for evidence, configured in zip(content, scale["inputs"]):
        require(evidence["element_count"] == str(configured["element_count"]), "input element_count")
        require(evidence["dtype"] == configured["dtype"], "input dtype")
        require(evidence["generator"] and evidence["preview_heacuda_api_tail2"], "input preview missing")

    log_text = log_path.read_text(encoding="utf-8")
    for marker in TABLE_MARKERS:
        require(marker in log_text, f"full log missing {marker}")
    require("OPERATOR_CASE_WAVELET_PROBE PASS" in log_text, "probe PASS missing from full log")
    require(report["actual_input_digest"] in input_digests, "report/timing digest mismatch")
    return {
        "operator": operator,
        "scale_id": scale["scale_id"],
        "dtype": dtype,
        "backend": backend,
        "run_id": by_device["CPU"]["run_id"],
        "case_id": by_device["CPU"]["case_id"],
        "input_digest": next(iter(input_digests)),
        "status": "PASS",
    }


def print_summary(report):
    print("[stage07-audit] operator summary")
    print("+-------------------+-------+--------+")
    print("| operator          | cases | status |")
    print("+-------------------+-------+--------+")
    for operator in OPERATORS:
        rows = [case for case in report["cases"] if case.get("operator") == operator]
        passed = sum(case.get("status") == "PASS" for case in rows)
        status = "PASS" if len(rows) == 15 and passed == 15 else "FAIL"
        print(f"| {operator:<17} | {len(rows):>5} | {status:<6} |")
    print("+-------------------+-------+--------+")
    print(f"OPERATOR_CASE_WAVELET_OPERATOR_AUDIT {report['status']} cases={report['case_count']} failures={len(report['failures'])} output={report['output']}")


def main():
    args = arguments()
    output = pathlib.Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    config = load_json(args.config)
    expected = expected_cases(config)
    thresholds = thresholds_by_key(load_json(args.thresholds))
    formal_root = pathlib.Path(args.formal_root)
    selected_directories, selection_records = selected_result_directories(
        args.driver_log_root, formal_root)
    report_paths = [unique_file(directory, "operator_*_report.json") for directory in selected_directories]
    all_report_paths = sorted(formal_root.glob("*/operator_*_report.json"))
    selected_resolved = {path.parent.resolve() for path in report_paths}
    unselected = [str(path.parent.resolve()) for path in all_report_paths
                  if path.parent.resolve() not in selected_resolved]
    cases = []
    failures = []
    observed = set()
    for report_path in report_paths:
        directory = report_path.parent
        try:
            minimal = load_json(report_path)
            key = (minimal["operator_name"], minimal["scale_id"])
            require(key in expected, f"unexpected case {key}")
            require(key not in observed, f"duplicate result case {key}")
            observed.add(key)
            dtype = expected[key]["inputs"][0]["dtype"].replace("Complex", "")
            require((key[0], dtype) in thresholds, f"missing threshold {(key[0], dtype)}")
            cases.append(audit_case(directory, args, expected[key], thresholds[(key[0], dtype)]))
        except Exception as error:
            failures.append({"directory": str(directory), "error": str(error)})
    missing = sorted(set(expected) - observed)
    for key in missing:
        failures.append({"directory": "NA", "error": f"missing result case {key}"})
    status = "PASS" if len(report_paths) == 30 and len(cases) == 30 and not failures else "FAIL"
    result = {
        "schema_version": 1,
        "audit_kind": "operator_case_batch03_formal_evidence",
        "formal_root": str(formal_root),
        "driver_log_root": str(pathlib.Path(args.driver_log_root)),
        "result_git_commit": args.git_commit,
        "result_git_dirty": args.git_dirty,
        "verifier_git_commit": args.verifier_commit,
        "expected_case_count": 30,
        "formal_report_file_count": len(all_report_paths),
        "selected_report_file_count": len(report_paths),
        "case_count": len(cases),
        "selection_records": selection_records,
        "unselected_historical_result_count": len(unselected),
        "unselected_historical_results": unselected,
        "cases": cases,
        "failures": failures,
        "status": status,
        "output": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(result)
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"OPERATOR_CASE_WAVELET_OPERATOR_AUDIT FAIL error={error}", file=sys.stderr)
        sys.exit(2)
