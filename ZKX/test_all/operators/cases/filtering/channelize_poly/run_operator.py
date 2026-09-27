#!/usr/bin/env python3
"""Remaining operator case channelize_poly single-case evidence driver."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys

from remaining_operator_case_terminal_summary import register_main_csv

IDENTITY = [
    "schema_version", "run_id", "run_type", "test_time_utc", "git_commit",
    "git_dirty", "config_id", "config_sha256", "target_kind", "target",
    "task_name", "step_name", "operator_name", "case_id", "scale_id",
    "order_of_magnitude", "actual_elements", "input_shape", "dtype", "device",
    "backend", "timing_scope", "status", "error_code",
]
SUMMARY = [
    "warmup_runs", "measured_runs", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
    "min_ms", "max_ms", "std_ms", "cv", "cpu_gpu_speedup",
    "accuracy_reference", "mse", "rmse", "relative_l2", "relative_linf",
    "exact_match", "mismatch_count", "semantic_check", "accuracy_status",
    "cpu_heap_before_bytes", "cpu_heap_after_bytes", "cpu_heap_peak_bytes",
    "cpu_heap_delta_bytes", "rss_before_bytes", "rss_after_bytes",
    "rss_peak_bytes", "rss_delta_bytes", "gpu_before_bytes", "gpu_after_bytes",
    "gpu_peak_bytes", "gpu_delta_bytes", "stability_case_index",
    "stability_total_cases", "perturbation_file", "perturbation_sha256",
    "perturbation_summary", "input_mode", "input_source_file",
    "input_source_sha256", "selection_seed", "profile_entry_id",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--backend", choices=("fft_thrust", "dlfft"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--run-type", choices=("smoke", "formal"), required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", choices=("true", "false"), required=True)
    parser.add_argument("--test-time-utc", required=True)
    parser.add_argument("--random", required=True)
    return parser.parse_args()


def load_json(path):
    raw = pathlib.Path(path).read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(values):
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "mean_ms": mean, "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95), "p99_ms": percentile(values, 0.99),
        "min_ms": min(values), "max_ms": max(values), "std_ms": std,
        "cv": std / mean if mean else 0.0,
    }


def resource_summary(trace, key):
    samples = [item[key] for item in trace]
    if not all(item["available"] for item in samples):
        return None
    values = [int(item["bytes"]) for item in samples]
    return {"before": values[0], "after": values[-1], "peak": max(values),
            "delta": values[-1] - values[0]}


def write_csv(path, columns, rows):
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    register_main_csv(path)


def identity(args, config, config_sha, scale, run_id, case_id, device, status, code):
    shapes = ";".join(
        f"{item['input_name']}=[{'x'.join(str(v) for v in item['shape'])}]"
        for item in scale["inputs"]
    )
    return {
        "schema_version": "1", "run_id": run_id, "run_type": args.run_type,
        "test_time_utc": args.test_time_utc, "git_commit": args.git_commit,
        "git_dirty": args.git_dirty, "config_id": config["scale_set_id"],
        "config_sha256": config_sha, "target_kind": "operator",
        "target": "channelize_poly", "task_name": "NA", "step_name": "NA",
        "operator_name": "channelize_poly", "case_id": case_id,
        "scale_id": scale["scale_id"],
        "order_of_magnitude": scale["order_of_magnitude"],
        "actual_elements": str(scale["actual_elements"]), "input_shape": shapes,
        "dtype": scale["inputs"][0]["dtype"], "device": device,
        "backend": args.backend, "timing_scope": "compatibility/end-to-end",
        "status": status, "error_code": code,
    }


def validate_contract(config, scale):
    if config.get("schema_version") != 1 or config.get("coverage") != "minimal_example_not_full_53":
        raise ValueError("invalid Remaining operator case scale config header")
    inputs = scale.get("inputs", [])
    if [item.get("input_name") for item in inputs] != ["x", "h"]:
        raise ValueError("channelize_poly requires typed x and h inputs")
    dtype = inputs[0].get("dtype")
    if dtype not in ("FP32", "FP16", "INT32", "INT16", "INT8") or inputs[1].get("dtype") != dtype:
        raise ValueError("x/h dtype contract mismatch")
    for item in inputs:
        count = math.prod(item["shape"])
        if count != item["element_count"]:
            raise ValueError("shape/element_count mismatch")
    params = scale.get("parameters", {})
    required = {
        "dtype_semantics": "actual_typed_array_inputs_fp32_compute_extension",
        "typed_input_names": "x|h", "compute_dtype": "FP32",
        "fft_dtype": "ComplexFP32", "output_dtype": "ComplexFP32",
        "backend": "fft_thrust",
    }
    for key, value in required.items():
        if params.get(key) != value:
            raise ValueError(f"invalid {key} contract")
    x_count, h_count = (item["element_count"] for item in inputs)
    n_chans = params.get("n_chans")
    if not isinstance(n_chans, int) or n_chans <= 0 or x_count % n_chans or h_count % n_chans:
        raise ValueError("invalid channel divisibility")
    if h_count // n_chans > 32 or scale["actual_elements"] != x_count + h_count:
        raise ValueError("invalid taps-per-phase or actual_elements")


def main():
    args = arguments()
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        raise ValueError("git commit must be 40 lowercase hex characters")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", args.test_time_utc):
        raise ValueError("test time must be YYYYMMDDThhmmssZ")
    if not re.fullmatch(r"[0-9a-f]{8}", args.random):
        raise ValueError("random must be eight lowercase hex characters")

    config, config_sha = load_json(args.config)
    thresholds_doc, _ = load_json(args.thresholds)
    scales = [item for op in config.get("operators", [])
              if op.get("operator_name") == "channelize_poly"
              for item in op.get("scales", []) if item.get("scale_id") == args.scale]
    if len(scales) != 1:
        raise ValueError("scale must identify exactly one channelize_poly case")
    scale = scales[0]
    validate_contract(config, scale)
    dtype = scale["inputs"][0]["dtype"]
    threshold = [item for item in thresholds_doc.get("entries", [])
                 if item.get("target") == "channelize_poly" and item.get("dtype") == dtype]
    if len(threshold) != 1:
        raise ValueError("missing unique dtype accuracy threshold")
    threshold = threshold[0]

    warmup, measured = ((20, 100) if args.run_type == "formal" else (1, 1))
    profile = f"remaining_operator_case_b03_{args.run_type}"
    run_id = f"{args.test_time_utc}_{args.git_commit[:12]}_{profile}_{args.random}"
    output_dir = pathlib.Path(args.results_root) / args.run_type / run_id
    if output_dir.exists():
        raise RuntimeError(f"run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "channelize_poly_probe_raw.json"
    log_path = output_dir / "operator_channelize_poly_full.log"
    x_count, h_count = (item["element_count"] for item in scale["inputs"])
    command = [args.binary, "--dtype", dtype, "--x-count", str(x_count),
               "--h-count", str(h_count), "--n-chans", str(scale["parameters"]["n_chans"]),
               "--seed", str(scale["parameters"]["generator_seed"]),
               "--warmup", str(warmup), "--measured", str(measured),
               "--output", str(raw_path)]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    log_path.write_text("command=" + " ".join(command) + "\n" + completed.stdout,
                        encoding="utf-8")
    if completed.returncode != 0 or not raw_path.is_file():
        raise RuntimeError(f"probe failed rc={completed.returncode}; log={log_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (raw.get("backend"), raw.get("requested_dtype"), raw.get("typed_input_count")) != (args.backend, dtype, 2):
        raise RuntimeError("probe dtype/backend evidence mismatch")
    if len(raw["cpu_samples_ms"]) != measured or len(raw["gpu_samples_ms"]) != measured:
        raise RuntimeError("probe timing sample count mismatch")
    expected_shape = [scale["parameters"]["n_chans"], x_count // scale["parameters"]["n_chans"]]
    if raw.get("output_shape") != expected_shape:
        raise RuntimeError("probe output shape mismatch")

    accuracy_pass = all(raw[name] <= threshold[f"{name}_max"] for name in
                        ("mse", "rmse", "relative_l2", "relative_linf"))
    cpu_heap = resource_summary(raw["cpu_trace"], "heap")
    cpu_rss = resource_summary(raw["cpu_trace"], "rss")
    gpu_heap = resource_summary(raw["gpu_trace"], "heap")
    gpu_rss = resource_summary(raw["gpu_trace"], "rss")
    gpu_memory = resource_summary(raw["gpu_trace"], "gpu")
    resource_pass = all(value is not None for value in
                        (cpu_heap, cpu_rss, gpu_heap, gpu_rss, gpu_memory))
    passed = accuracy_pass and resource_pass
    status = "PASS" if passed else "FAIL"
    code = "OK" if passed else ("ACCURACY_FAILED" if not accuracy_pass else "RESOURCE_FAILED")
    case_id = f"channelize_poly__{scale['scale_id']}__{raw['actual_input_digest'][:16]}"
    cpu_stats, gpu_stats = stats(raw["cpu_samples_ms"]), stats(raw["gpu_samples_ms"])
    speedup = cpu_stats["mean_ms"] / gpu_stats["mean_ms"]

    def main_row(device, timing, heap, rss, gpu):
        row = identity(args, config, config_sha, scale, run_id, case_id, device, status, code)
        row.update({key: f"{value:.17g}" for key, value in timing.items()})
        row.update({"warmup_runs": str(warmup), "measured_runs": str(measured),
                    "cpu_gpu_speedup": f"{speedup:.17g}" if device == "GPU" else "NA",
                    "accuracy_reference": "cpu_typed_double_direct_dft",
                    "mse": f"{raw['mse']:.17g}" if device == "GPU" else "0",
                    "rmse": f"{raw['rmse']:.17g}" if device == "GPU" else "0",
                    "relative_l2": f"{raw['relative_l2']:.17g}" if device == "GPU" else "0",
                    "relative_linf": f"{raw['relative_linf']:.17g}" if device == "GPU" else "0",
                    "exact_match": "NA", "mismatch_count": "NA",
                    "semantic_check": threshold["semantic_rule"] + f":shape={expected_shape};finite=true",
                    "accuracy_status": "PASS" if accuracy_pass else "FAIL"})
        for prefix, summary in (("cpu_heap", heap), ("rss", rss), ("gpu", gpu)):
            for field in ("before", "after", "peak", "delta"):
                row[f"{prefix}_{field}_bytes"] = str(summary[field]) if summary else "NA"
        for key in SUMMARY:
            row.setdefault(key, "NA")
        return row

    main_rows = [main_row("CPU", cpu_stats, cpu_heap, cpu_rss, None),
                 main_row("GPU", gpu_stats, gpu_heap, gpu_rss, gpu_memory)]
    write_csv(output_dir / f"operator_main_results_channelize_poly_{args.backend}_{run_id}.csv",
              IDENTITY + SUMMARY, main_rows)

    timing_columns = IDENTITY + ["warmup_runs", "measured_runs", "sample_index",
                                 "latency_ms", "synchronized", "input_digest", "output_digest"]
    timing_rows = []
    for device, samples, digest in (("CPU", raw["cpu_samples_ms"], raw["cpu_output_digest"]),
                                    ("GPU", raw["gpu_samples_ms"], raw["gpu_output_digest"])):
        for index, latency in enumerate(samples):
            row = identity(args, config, config_sha, scale, run_id, case_id, device, status, code)
            row.update({"warmup_runs": str(warmup), "measured_runs": str(measured),
                        "sample_index": str(index), "latency_ms": f"{latency:.17g}",
                        "synchronized": "true", "input_digest": raw["actual_input_digest"],
                        "output_digest": digest})
            timing_rows.append(row)
    write_csv(output_dir / f"operator_timing_samples_channelize_poly_{args.backend}_{run_id}.csv",
              timing_columns, timing_rows)

    memory_columns = IDENTITY + ["sample_index", "trace_phase", "phase_index",
                                 "cpu_live_heap_bytes", "rss_bytes", "gpu_used_bytes"]
    memory_rows = []
    for device, trace in (("CPU", raw["cpu_trace"]), ("GPU", raw["gpu_trace"])):
        for item in trace:
            row = identity(args, config, config_sha, scale, run_id, case_id, device, status, code)
            row.update({"sample_index": "0", "trace_phase": item["phase"],
                        "phase_index": str(item["phase_index"]),
                        "cpu_live_heap_bytes": str(item["heap"]["bytes"]) if item["heap"]["available"] else "NA",
                        "rss_bytes": str(item["rss"]["bytes"]) if item["rss"]["available"] else "NA",
                        "gpu_used_bytes": str(item["gpu"]["bytes"]) if item["gpu"]["available"] else "NA"})
            memory_rows.append(row)
    write_csv(output_dir / f"operator_filtering_channelize_poly_memory_trace_{args.backend}.csv",
              memory_columns, memory_rows)

    dtype_columns = ["run_id", "case_id", "operator_name", "requested_dtype",
                     "dtype_semantics", "typed_input_count", "typed_input_manifest",
                     "actual_input_digest", "compute_dtype", "fft_dtype", "output_dtype",
                     "backend", "status", "error_code"]
    write_csv(output_dir / f"operator_dtype_evidence_channelize_poly_{args.backend}_{run_id}.csv",
              dtype_columns, [{"run_id": run_id, "case_id": case_id,
              "operator_name": "channelize_poly", "requested_dtype": dtype,
              "dtype_semantics": "actual_typed_array_inputs_fp32_compute_extension",
              "typed_input_count": "2", "typed_input_manifest": raw["typed_input_manifest"],
              "actual_input_digest": raw["actual_input_digest"], "compute_dtype": "FP32",
              "fft_dtype": "ComplexFP32", "output_dtype": "ComplexFP32", "backend": args.backend,
              "status": status, "error_code": code}])

    module_columns = ["run_id", "case_id", "operator_name", "scale_id", "dtype",
                      "backend", "warmup_runs", "measured_runs", "cpu_mean_ms",
                      "gpu_mean_ms", "cpu_gpu_speedup", "mse", "rmse", "relative_l2",
                      "relative_linf", "status", "error_code"]
    write_csv(output_dir / f"filtering_batch03_module_summary_{args.backend}_{run_id}.csv",
              module_columns, [{"run_id": run_id, "case_id": case_id,
              "operator_name": "channelize_poly", "scale_id": scale["scale_id"],
              "dtype": dtype, "backend": args.backend, "warmup_runs": str(warmup),
              "measured_runs": str(measured), "cpu_mean_ms": f"{cpu_stats['mean_ms']:.17g}",
              "gpu_mean_ms": f"{gpu_stats['mean_ms']:.17g}", "cpu_gpu_speedup": f"{speedup:.17g}",
              "mse": f"{raw['mse']:.17g}", "rmse": f"{raw['rmse']:.17g}",
              "relative_l2": f"{raw['relative_l2']:.17g}",
              "relative_linf": f"{raw['relative_linf']:.17g}", "status": status,
              "error_code": code}])

    report = {"schema_version": 1, "run_id": run_id, "case_id": case_id,
              "operator_name": "channelize_poly", "scale_id": scale["scale_id"],
              "requested_dtype": dtype, "dtype_semantics":
              "actual_typed_array_inputs_fp32_compute_extension", "typed_input_count": 2,
              "compute_dtype": "FP32", "fft_dtype": "ComplexFP32",
              "output_dtype": "ComplexFP32", "backend": args.backend,
              "warmup_runs": warmup, "measured_runs": measured,
              "actual_input_digest": raw["actual_input_digest"], "metrics": {
              name: raw[name] for name in ("mse", "rmse", "relative_l2", "relative_linf")},
              "status": status, "error_code": code}
    (output_dir / "operator_channelize_poly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "operator_channelize_poly_summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in {
            "status": status, "error_code": code, "run_id": run_id,
            "case_id": case_id, "scale_id": scale["scale_id"], "dtype": dtype,
            "dtype_semantics": "actual_typed_array_inputs_fp32_compute_extension",
            "typed_input_count": 2, "compute_dtype": "FP32", "fft_dtype": "ComplexFP32",
            "output_dtype": "ComplexFP32", "backend": args.backend,
            "warmup_runs": warmup, "measured_runs": measured,
            "actual_input_digest": raw["actual_input_digest"]}.items()) + "\n",
        encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"status={status}\nerror_code={code}\ncase_id={case_id}\n")
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"REMAINING_OPERATOR_CASE_CHANNELIZE_CASE {status} scale={scale['scale_id']} dtype={dtype} "
          f"backend={args.backend} measured={measured} results={output_dir}")
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"REMAINING_OPERATOR_CASE_CHANNELIZE_DRIVER FAIL error={error}", file=sys.stderr)
        sys.exit(2)
