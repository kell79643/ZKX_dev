#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import pathlib
import re
import sys

SUPPORT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SUPPORT))
from batch05_config import expand_config

B04_PATH = SUPPORT.parents[1] / "shared" / "filtering" / "verify_formal_results.py"
spec = importlib.util.spec_from_file_location("operator_case_b04_audit_helpers", B04_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OPERATORS = {
    "fm_demod": ("not_applicable", "demod"),
    "correlate": ("not_applicable", "convolution"),
    "spectrogram": ("fft_thrust", "spectral_analysis"),
}


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


def identity(row, args, scale, device):
    expected = {
        "run_type": "formal", "git_commit": args.git_commit,
        "git_dirty": args.git_dirty, "target_kind": "operator",
        "target": scale["operator_name"], "task_name": "Task2",
        "step_name": "Step4", "operator_name": scale["operator_name"],
        "scale_id": scale["scale_id"],
        "dtype": scale["inputs"][0]["dtype"].replace("Complex", ""),
        "device": device, "backend": scale["backend"],
        "timing_scope": "compatibility/end-to-end",
        "status": "PASS", "error_code": "OK",
    }
    for key, value in expected.items():
        base.require(row.get(key) == value,
                     f"{scale['scale_id']} {device} {key}: {row.get(key)} != {value}")


def selected_directories(driver_root, formal_root):
    logs = sorted(pathlib.Path(driver_root).glob("*/*.log"))
    base.require(len(logs) == 45, f"expected 45 driver logs, found {len(logs)}")
    pattern = re.compile(r"^OPERATOR_CASE_OPERATOR_BENCHMARKEATURE_CASE PASS .* results=(\S+)$", re.MULTILINE)
    formal = pathlib.Path(formal_root).resolve()
    selected = []
    records = []
    for log in logs:
        matches = pattern.findall(log.read_text(encoding="utf-8"))
        base.require(len(matches) == 1, f"{log} PASS paths={len(matches)}")
        directory = pathlib.Path(matches[0]).resolve()
        base.require(directory.parent == formal, f"result outside formal root: {directory}")
        base.require(directory.is_dir(), f"missing result directory: {directory}")
        selected.append(directory)
        records.append({"driver_log": str(log.resolve()), "result_directory": str(directory)})
    base.require(len(set(selected)) == 45, "driver logs do not select 45 unique directories")
    return selected, records


def read_rows(path):
    with pathlib.Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_case(directory, args, scale, threshold):
    report_path = base.unique_file(directory, "operator_*_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    operator = scale["operator_name"]
    dtype = scale["inputs"][0]["dtype"].replace("Complex", "")
    base.require(report["operator_name"] == operator, "report operator")
    base.require(report["scale_id"] == scale["scale_id"], "report scale")
    base.require(report["requested_dtype"] == dtype, "report dtype")
    base.require(report["backend"] == scale["backend"], "report backend")
    base.require(report["output_count"] == scale["parameters"]["output_count"], "output count")
    base.require(report["warmup_runs"] == 20 and report["measured_runs"] == 100, "run counts")
    base.require(report["status"] == "PASS" and report["error_code"] == "OK", "report status")

    main_path = base.unique_file(directory, "operator_main_results_*.csv")
    timing_path = base.unique_file(directory, "operator_timing_samples_*.csv")
    memory_path = base.unique_file(directory, "operator_*_memory_trace_*.csv")
    dtype_path = base.unique_file(directory, "operator_dtype_evidence_*.csv")
    log_path = base.unique_file(directory, "operator_*_full.log")
    base.unique_file(directory, "operator_*_summary.txt")
    base.unique_file(directory, "*_probe_raw.json")

    main = read_rows(main_path)
    base.require(len(main) == 2, "main row count")
    by_device = {row["device"]: row for row in main}
    base.require(set(by_device) == {"CPU", "GPU"}, "main devices")
    for device, row in by_device.items():
        identity(row, args, scale, device)
        base.require(row["warmup_runs"] == "20" and row["measured_runs"] == "100", "main counts")

    timing = read_rows(timing_path)
    base.require(len(timing) == 200, "timing row count")
    input_digests = set()
    summaries = {}
    for device in ("CPU", "GPU"):
        rows = [row for row in timing if row["device"] == device]
        base.require(len(rows) == 100, f"{device} timing count")
        base.require({int(row["sample_index"]) for row in rows} == set(range(100)), "sample indices")
        values = []
        output_digests = set()
        for row in rows:
            identity(row, args, scale, device)
            base.require(row["synchronized"] == "true", "timing sync")
            input_digests.add(row["input_digest"])
            output_digests.add(row["output_digest"])
            values.append(base.finite_number(row["latency_ms"], "latency_ms"))
        base.require(len(output_digests) == 1, f"{device} output digest changed")
        base.require(all(value > 0 for value in values), "nonpositive latency")
        summaries[device] = base.timing_summary(values)
        for field, expected in summaries[device].items():
            base.close(float(by_device[device][field]), expected, f"{device}.{field}")
    base.require(len(input_digests) == 1, "CPU/GPU input digest mismatch")
    base.close(float(by_device["GPU"]["cpu_gpu_speedup"]),
               summaries["CPU"]["mean_ms"] / summaries["GPU"]["mean_ms"], "speedup")

    for metric in base.FLOAT_METRICS:
        value = base.finite_number(by_device["GPU"][metric], metric)
        base.require(value <= float(threshold[metric + "_max"]), f"{metric} threshold")
        base.close(value, float(report[metric]), f"report.{metric}")
    base.require(by_device["GPU"]["accuracy_status"] == "PASS", "accuracy status")

    memory = read_rows(memory_path)
    base.require(len(memory) == 14, "memory row count")
    for device in ("CPU", "GPU"):
        rows = sorted((row for row in memory if row["device"] == device),
                      key=lambda row: int(row["phase_index"]))
        base.require([row["trace_phase"] for row in rows] == base.PHASES, f"{device} phases")
        for row in rows:
            identity(row, args, scale, device)
        for csv_field, prefix in (("cpu_live_heap_bytes", "cpu_heap"),
                                  ("rss_bytes", "rss"), ("gpu_used_bytes", "gpu")):
            values = [row[csv_field] for row in rows]
            if device == "CPU" and csv_field == "gpu_used_bytes":
                base.require(set(values) == {"NA"}, "CPU GPU memory must be NA")
                continue
            base.require(all(value != "NA" for value in values), f"{device}.{csv_field} NA")
            numbers = [int(value) for value in values]
            expected = {"before": numbers[0], "after": numbers[-1],
                        "peak": max(numbers), "delta": numbers[-1] - numbers[0]}
            for field, value in expected.items():
                base.require(int(by_device[device][f"{prefix}_{field}_bytes"]) == value,
                             f"{device}.{prefix}_{field}")

    dtype_rows = read_rows(dtype_path)
    base.require(len(dtype_rows) == 1, "dtype row count")
    dtype_row = dtype_rows[0]
    base.require(dtype_row["actual_input_digest"] in input_digests, "dtype digest")
    base.require(dtype_row["status"] == "PASS" and dtype_row["error_code"] == "OK", "dtype status")
    content = json.loads(dtype_row["input_content_json"])
    parameters = json.loads(dtype_row["case_parameters_json"])
    base.require(parameters == scale["parameters"], "parameters mismatch")
    base.require(content == report["input_content"], "report content mismatch")
    base.require([item["input_name"] for item in content] ==
                 [item["input_name"] for item in scale["inputs"]], "input names")
    for evidence, configured in zip(content, scale["inputs"]):
        base.require(evidence["element_count"] == str(configured["element_count"]), "input count")
        base.require(evidence["dtype"] == configured["dtype"], "input dtype")
        base.require(evidence["generator"] and evidence["preview_heacuda_api_tail2"], "input preview")

    log_text = log_path.read_text(encoding="utf-8")
    for marker in base.TABLE_MARKERS:
        base.require(marker in log_text, f"missing terminal table {marker}")
    base.require("OPERATOR_CASE_OPERATOR_BENCHMARKEATURE_PROBE PASS" in log_text, "probe PASS marker")
    return {"operator": operator, "scale_id": scale["scale_id"], "dtype": dtype,
            "backend": scale["backend"], "input_digest": next(iter(input_digests)),
            "status": "PASS"}


def main():
    args = arguments()
    output = pathlib.Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    scales = expand_config(config)
    expected = {(item["operator_name"], item["scale_id"]): item for item in scales}
    base.require(len(expected) == 45, f"expected cases={len(expected)}")
    thresholds_doc = json.loads(pathlib.Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds = {(item["target"], item["dtype"]): item for item in thresholds_doc["entries"]}
    directories, records = selected_directories(args.driver_log_root, args.formal_root)
    cases = []
    failures = []
    observed = set()
    for directory in directories:
        try:
            report = json.loads(base.unique_file(directory, "operator_*_report.json").read_text())
            key = (report["operator_name"], report["scale_id"])
            base.require(key in expected, f"unexpected case {key}")
            base.require(key not in observed, f"duplicate case {key}")
            observed.add(key)
            dtype = expected[key]["inputs"][0]["dtype"].replace("Complex", "")
            cases.append(audit_case(directory, args, expected[key], thresholds[(key[0], dtype)]))
        except Exception as error:
            failures.append({"directory": str(directory), "error": str(error)})
    for key in sorted(set(expected) - observed):
        failures.append({"directory": "NA", "error": f"missing result case {key}"})
    status = "PASS" if len(cases) == 45 and not failures else "FAIL"
    result = {"schema_version": 1, "audit_kind": "operator_case_batch05_formal_evidence",
              "formal_root": args.formal_root, "driver_log_root": args.driver_log_root,
              "result_git_commit": args.git_commit, "result_git_dirty": args.git_dirty,
              "verifier_git_commit": args.verifier_commit, "expected_case_count": 45,
              "case_count": len(cases), "selection_records": records, "cases": cases,
              "failures": failures, "status": status, "output": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[stage07-audit] operator summary")
    for operator in OPERATORS:
        rows = [item for item in cases if item["operator"] == operator]
        print(f"{operator}: cases={len(rows)} status={'PASS' if len(rows) == 15 else 'FAIL'}")
    print(f"OPERATOR_CASE_TASK2_FEATURE_OPERATOR_AUDIT {status} cases={len(cases)} failures={len(failures)} output={output}")
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"OPERATOR_CASE_TASK2_FEATURE_OPERATOR_AUDIT FAIL error={error}", file=sys.stderr)
        sys.exit(2)
