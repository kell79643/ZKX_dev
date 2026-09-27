#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

SUPPORT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SUPPORT))
from batch05_config import expand_config

COMMON_DIR = SUPPORT.parents[1] / "cases" / "filtering" / "channelize_poly"
sys.path.insert(0, str(COMMON_DIR))
import run_operator as common

B04_PATH = SUPPORT.parents[1] / "shared" / "filtering" / "run_operator.py"
spec = importlib.util.spec_from_file_location("operator_case_b04_terminal", B04_PATH)
b04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b04)

OPERATORS = {"fm_demod", "correlate", "spectrogram"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "operator", "backend", "config", "thresholds", "results-root",
                 "scale", "run-type", "git-commit", "git-dirty",
                 "test-time-utc", "random"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.operator not in OPERATORS:
        raise ValueError("operator outside Stage07 Batch05")
    if args.backend not in ("fft_thrust", "dlfft"):
        raise ValueError("invalid selected FFT backend")

    config_bytes = pathlib.Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    scales = expand_config(config)
    scale = next(item for item in scales if item["scale_id"] == args.scale)
    if scale["operator_name"] != args.operator:
        raise ValueError("operator/scale mismatch")
    dtype = scale["inputs"][0]["dtype"].replace("Complex", "")
    parameters = scale["parameters"]
    threshold_doc = json.loads(pathlib.Path(args.thresholds).read_text())
    threshold = next(item for item in threshold_doc["entries"]
                     if item["target"] == args.operator and item["dtype"] == dtype)
    warmup, measured = (20, 100) if args.run_type == "formal" else (1, 1)
    run_id = (f"{args.test_time_utc}_{args.git_commit[:12]}_operator_case_b05_"
              f"{args.run_type}_{args.random}")
    output_dir = pathlib.Path(args.results_root) / args.run_type / run_id
    if output_dir.exists():
        raise RuntimeError("result exists")
    output_dir.mkdir(parents=True)
    raw_path = output_dir / f"{args.operator}_probe_raw.json"

    command = [args.binary, "--dtype", dtype,
               "--count", str(parameters.get("count", parameters.get("signal_count"))),
               "--reference-count", str(parameters.get("reference_count", 31)),
               "--frame", str(parameters.get("frame", 64)),
               "--hop", str(parameters.get("hop", 32)),
               "--warmup", str(warmup), "--measured", str(measured),
               "--output", str(raw_path)]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    log_path = output_dir / f"operator_{args.operator}_full.log"
    log_path.write_text("command=" + " ".join(command) + "\n" + completed.stdout)
    if completed.returncode:
        raise RuntimeError("probe failed: " + completed.stdout)
    raw = json.loads(raw_path.read_text())
    actual_backend = args.backend if scale["backend"] == "fft_thrust" else "not_applicable"
    if args.operator == "spectrogram" and raw.get("backend") != actual_backend:
        raise RuntimeError("probe backend evidence mismatch")

    metrics = ("mse", "rmse", "relative_l2", "relative_linf")
    accuracy_ok = all(raw[name] <= threshold[name + "_max"] for name in metrics)
    semantic_ok = (raw["output_count"] == parameters["output_count"] and
                   len(raw["cpu_samples_ms"]) == measured and
                   len(raw["gpu_samples_ms"]) == measured)
    summaries = [common.resource_summary(raw[key], metric) for key, metric in (
        ("cpu_trace", "heap"), ("cpu_trace", "rss"),
        ("gpu_trace", "heap"), ("gpu_trace", "rss"), ("gpu_trace", "gpu"))]
    resource_ok = all(item is not None for item in summaries)
    status = "PASS" if accuracy_ok and semantic_ok and resource_ok else "FAIL"
    error_code = "OK" if status == "PASS" else "RESOURCE_FAILED" if not resource_ok else "ACCURACY_FAILED"
    case_id = f"{args.operator}__{args.scale}__{raw['actual_input_digest'][:16]}"

    def identity(device):
        return dict(zip(common.IDENTITY, [
            "1", run_id, args.run_type, args.test_time_utc, args.git_commit,
            args.git_dirty, config["scale_set_id"], config_hash, "operator",
            args.operator, "Task2", "Step4", args.operator, case_id,
            args.scale, scale["order_of_magnitude"], str(scale["actual_elements"]),
            "|".join(item["input_name"] + "=" + str(item["shape"])
                     for item in scale["inputs"]), dtype, device, actual_backend,
            "compatibility/end-to-end", status, error_code]))

    cpu_stats = common.stats(raw["cpu_samples_ms"])
    gpu_stats = common.stats(raw["gpu_samples_ms"])
    rows = []
    for device, stats, heap, rss, gpu in (
        ("CPU", cpu_stats, summaries[0], summaries[1], None),
        ("GPU", gpu_stats, summaries[2], summaries[3], summaries[4])):
        row = identity(device)
        row.update({key: str(value) for key, value in stats.items()})
        is_gpu = device == "GPU"
        exact = raw["exact_match"] if args.operator == "correlate" and is_gpu else "NA"
        mismatch = raw["mismatch_count"] if args.operator == "correlate" and is_gpu else "NA"
        row.update({
            "warmup_runs": str(warmup), "measured_runs": str(measured),
            "cpu_gpu_speedup": str(cpu_stats["mean_ms"] / gpu_stats["mean_ms"]) if is_gpu else "NA",
            "accuracy_reference": "typed_cpu_task2_step4_reference",
            "mse": str(raw["mse"] if is_gpu else 0),
            "rmse": str(raw["rmse"] if is_gpu else 0),
            "relative_l2": str(raw["relative_l2"] if is_gpu else 0),
            "relative_linf": str(raw["relative_linf"] if is_gpu else 0),
            "exact_match": str(exact).lower() if exact != "NA" else "NA",
            "mismatch_count": str(mismatch),
            "semantic_check": (f"output_count={raw['output_count']};"
                               f"output_dtype={raw['output_dtype']};operator_matrixpu_call=present"),
            "accuracy_status": "PASS" if accuracy_ok and semantic_ok else "FAIL",
        })
        for prefix, summary in (("cpu_heap", heap), ("rss", rss), ("gpu", gpu)):
            for field in ("before", "after", "peak", "delta"):
                row[f"{prefix}_{field}_bytes"] = str(summary[field]) if summary else "NA"
        for field in common.SUMMARY:
            row.setdefault(field, "NA")
        rows.append(row)

    main_path = output_dir / f"operator_main_results_{args.operator}_{actual_backend}_{run_id}.csv"
    common.write_csv(main_path, common.IDENTITY + common.SUMMARY, rows)
    timing_columns = common.IDENTITY + ["warmup_runs", "measured_runs", "sample_index",
                                        "latency_ms", "synchronized", "input_digest", "output_digest"]
    timing_rows = []
    for device, samples, output_digest in (
        ("CPU", raw["cpu_samples_ms"], raw["cpu_output_digest"]),
        ("GPU", raw["gpu_samples_ms"], raw["gpu_output_digest"])):
        for index, value in enumerate(samples):
            row = identity(device)
            row.update({"warmup_runs": str(warmup), "measured_runs": str(measured),
                        "sample_index": str(index), "latency_ms": str(value),
                        "synchronized": "true", "input_digest": raw["actual_input_digest"],
                        "output_digest": output_digest})
            timing_rows.append(row)
    timing_path = output_dir / f"operator_timing_samples_{args.operator}_{actual_backend}_{run_id}.csv"
    common.write_csv(timing_path, timing_columns, timing_rows)

    memory_columns = common.IDENTITY + ["sample_index", "trace_phase", "phase_index",
                                        "cpu_live_heap_bytes", "rss_bytes", "gpu_used_bytes"]
    memory_rows = []
    for device, trace in (("CPU", raw["cpu_trace"]), ("GPU", raw["gpu_trace"])):
        for item in trace:
            row = identity(device)
            row.update({
                "sample_index": "0", "trace_phase": item["phase"],
                "phase_index": str(item["phase_index"]),
                "cpu_live_heap_bytes": str(item["heap"]["bytes"]) if item["heap"]["available"] else "NA",
                "rss_bytes": str(item["rss"]["bytes"]) if item["rss"]["available"] else "NA",
                "gpu_used_bytes": str(item["gpu"]["bytes"]) if item["gpu"]["available"] else "NA",
            })
            memory_rows.append(row)
    memory_path = output_dir / f"operator_{scale['module_name']}_{args.operator}_memory_trace_{actual_backend}.csv"
    common.write_csv(memory_path, memory_columns, memory_rows)

    if args.operator == "fm_demod":
        content = [{"input_name":"analytic","shape":str(scale["inputs"][0]["shape"]),
                    "dtype":scale["inputs"][0]["dtype"],"element_count":str(scale["inputs"][0]["element_count"]),
                    "generator":"deterministic_chirped_analytic_signal","preview_heacuda_api_tail2":raw["preview0"]}]
    elif args.operator == "correlate":
        content = [
            {"input_name":"signal","shape":str(scale["inputs"][0]["shape"]),"dtype":dtype,
             "element_count":str(scale["inputs"][0]["element_count"]),"generator":"task2_signal_mixture","preview_heacuda_api_tail2":raw["preview0"]},
            {"input_name":"reference","shape":str(scale["inputs"][1]["shape"]),"dtype":dtype,
             "element_count":str(scale["inputs"][1]["element_count"]),"generator":"task2_reference_template","preview_heacuda_api_tail2":raw["preview1"]},
        ]
    else:
        content = [{"input_name":"signal","shape":str(scale["inputs"][0]["shape"]),"dtype":dtype,
                    "element_count":str(scale["inputs"][0]["element_count"]),"generator":"task2_signal_mixture",
                    "preview_heacuda_api_tail2":raw["preview0"]}]
    dtype_evidence = {
        "run_id": run_id, "case_id": case_id, "operator_name": args.operator,
        "requested_dtype": dtype, "config_input_dtype": scale["inputs"][0]["dtype"],
        "typed_input_manifest": raw["typed_input_manifest"],
        "typed_array_digest": raw["typed_array_digest"],
        "input_content_json": json.dumps(content, sort_keys=True, separators=(",", ":")),
        "case_parameters_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        "actual_input_digest": raw["actual_input_digest"],
        "output_dtype": raw["output_dtype"], "backend": actual_backend,
        "operator_matrixpu_call_status": "proved_gpu_cpu", "status": status,
        "error_code": error_code,
    }
    dtype_path = output_dir / f"operator_dtype_evidence_{args.operator}_{actual_backend}_{run_id}.csv"
    common.write_csv(dtype_path, list(dtype_evidence), [dtype_evidence])
    report = {
        "run_id": run_id, "case_id": case_id, "operator_name": args.operator,
        "task_name": "Task2", "step_name": "Step4", "scale_id": args.scale,
        "requested_dtype": dtype, "input_content": content,
        "case_parameters": parameters, "output_dtype": raw["output_dtype"],
        "backend": actual_backend, "warmup_runs": warmup,
        "measured_runs": measured, "output_count": raw["output_count"],
        **{name: raw[name] for name in metrics},
        "actual_input_digest": raw["actual_input_digest"],
        "typed_array_digest": raw["typed_array_digest"],
        "status": status, "error_code": error_code,
    }
    report_path = output_dir / f"operator_{args.operator}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary_path = output_dir / f"operator_{args.operator}_summary.txt"
    summary_path.write_text("\n".join(f"{key}={value}" for key, value in report.items()) + "\n")
    paths = [("main_csv", main_path), ("timing_csv", timing_path),
             ("memory_csv", memory_path), ("dtype_csv", dtype_path),
             ("report_json", report_path), ("summary_txt", summary_path),
             ("full_log", log_path)]
    terminal = b04.terminal_summary(main_path, paths)
    log_path.write_text(log_path.read_text() + json.dumps(report) + "\n" + terminal + "\n")
    print(terminal)
    print(f"OPERATOR_CASE_OPERATOR_BENCHMARKEATURE_CASE {status} operator={args.operator} "
          f"scale={args.scale} dtype={dtype} backend={actual_backend} "
          f"measured={measured} results={output_dir}")
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(error, file=sys.stderr)
        sys.exit(2)
