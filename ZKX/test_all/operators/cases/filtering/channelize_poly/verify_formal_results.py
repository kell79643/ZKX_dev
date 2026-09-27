#!/usr/bin/env python3
"""Strict Remaining operator case B03 channelize_poly formal evidence verifier."""

import argparse
import csv
import json
import pathlib
import re
import sys

from remaining_operator_case_terminal_summary import verify_input_content_evidence


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", choices=("true", "false"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def one(directory, pattern):
    values = list(directory.glob(pattern))
    if len(values) != 1:
        raise ValueError(f"expected one {pattern} under {directory}, found {len(values)}")
    return values[0]


def rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = arguments()
    root = pathlib.Path(args.formal_root)
    reports = sorted(root.glob("*/operator_channelize_poly_report.json"))
    if len(reports) != 15:
        raise ValueError(f"expected 15 report files, found {len(reports)}")
    coverage = set()
    case_ids = set()
    for report_path in reports:
        directory = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("backend") != "fft_thrust":
            raise ValueError(f"non-PASS or wrong backend: {report_path}")
        dtype = report.get("requested_dtype")
        scale_id = report.get("scale_id", "")
        match = re.fullmatch(r"channelize_poly_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)", scale_id)
        if not match or match.group(1).upper() != dtype:
            raise ValueError(f"dtype/scale mismatch: {report_path}")
        coverage.add((dtype, match.group(2)))
        case_id = report.get("case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id {case_id}")
        case_ids.add(case_id)

        main_rows = rows(one(directory, "operator_main_results_channelize_poly_fft_thrust_*.csv"))
        timing = rows(one(directory, "operator_timing_samples_channelize_poly_fft_thrust_*.csv"))
        memory = rows(one(directory, "operator_filtering_channelize_poly_memory_trace_fft_thrust.csv"))
        dtype_rows = rows(one(directory, "operator_dtype_evidence_channelize_poly_fft_thrust_*.csv"))
        if len(main_rows) != 2 or {item["device"] for item in main_rows} != {"CPU", "GPU"}:
            raise ValueError(f"main CSV pairing failure: {directory}")
        if len(timing) != 200 or {item["device"] for item in timing} != {"CPU", "GPU"}:
            raise ValueError(f"timing CSV count failure: {directory}")
        for device in ("CPU", "GPU"):
            selected = [item for item in timing if item["device"] == device]
            if len(selected) != 100 or {int(item["sample_index"]) for item in selected} != set(range(100)):
                raise ValueError(f"timing sample identity failure: {directory}/{device}")
        if len(memory) != 14 or {item["trace_phase"] for item in memory} != {
                "before", "allocate", "h2d", "execute_sync", "d2h", "release_sync", "after"}:
            raise ValueError(f"memory trace failure: {directory}")
        if len(dtype_rows) != 1:
            raise ValueError(f"dtype evidence count failure: {directory}")
        evidence = dtype_rows[0]
        verify_input_content_evidence(evidence, ("x", "h"))
        if (evidence.get("requested_dtype"), evidence.get("dtype_semantics"),
                evidence.get("typed_input_count"), evidence.get("compute_dtype"),
                evidence.get("fft_dtype"), evidence.get("output_dtype")) != (
                dtype, "actual_typed_array_inputs_fp32_compute_extension", "2",
                "FP32", "ComplexFP32", "ComplexFP32"):
            raise ValueError(f"dtype evidence contract failure: {directory}")
        manifest = evidence.get("typed_input_manifest", "")
        if not re.fullmatch(r"x:" + dtype + r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|"
                            r"h:" + dtype + r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}", manifest):
            raise ValueError(f"typed input byte evidence failure: {directory}")
        if not re.fullmatch(r"[0-9a-f]{64}", evidence.get("actual_input_digest", "")):
            raise ValueError(f"actual input digest failure: {directory}")
        for row in main_rows + timing + memory:
            if (row.get("git_commit"), row.get("git_dirty"), row.get("backend"),
                    row.get("status"), row.get("case_id")) != (
                    args.git_commit, args.git_dirty, "fft_thrust", "PASS", case_id):
                raise ValueError(f"identity/status failure: {directory}")

    expected = {(dtype, scale) for dtype in ("FP32", "FP16", "INT32", "INT16", "INT8")
                for scale in ("10e2", "10e3", "10e4")}
    if coverage != expected:
        raise ValueError(f"coverage mismatch missing={sorted(expected - coverage)} extra={sorted(coverage - expected)}")
    result = ("REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b03 operator=channelize_poly "
              "backend=fft_thrust cases=15 warmup=20 measured=100 "
              "dtype_semantics=actual_typed_array_inputs_fp32_compute_extension "
              "typed_inputs=x|h input_preview=heacuda_api_tail2 request_variant_coverage=5x3\n")
    output = pathlib.Path(args.output)
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.write_text(result, encoding="utf-8")
    print(result, end="")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={error}", file=sys.stderr)
        sys.exit(2)
