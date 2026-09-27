#!/usr/bin/env python3
import argparse
import csv
import pathlib
import sys


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=pathlib.Path)
    parser.add_argument("--results", required=True, type=pathlib.Path)
    parser.add_argument("--expected-cases", required=True, type=int, choices=(1, 6))
    args = parser.parse_args()
    catalog = {row["case_id"]: row for row in read_rows(args.catalog)}
    results = read_rows(args.results)
    errors = []
    if len(catalog) != 6:
        errors.append(f"catalog must contain 6 unique cases: {len(catalog)}")
    if len(results) != args.expected_cases:
        errors.append(f"expected {args.expected_cases} result rows, got {len(results)}")
    seen = set()
    for row in results:
        case_id = row.get("case_id", "")
        if not case_id or case_id in seen:
            errors.append(f"empty/duplicate case_id={case_id}")
            continue
        seen.add(case_id)
        expected = catalog.get(case_id)
        if expected is None:
            errors.append(f"unknown case_id={case_id}")
            continue
        if row.get("task_name") != "Task1" or row.get("step_name") != expected["step_name"]:
            errors.append(f"{case_id} task/step mismatch")
        if row.get("abnormal_type") != expected["abnormal_type"]:
            errors.append(f"{case_id} abnormal_type mismatch")
        codes = tuple(row.get(key) for key in
                      ("expected_code", "captured_code", "returned_code", "actual_code"))
        if any(code != expected["expected_code"] for code in codes):
            errors.append(f"{case_id} code mismatch={codes}")
        if not row.get("input_summary") or not row.get("error_message"):
            errors.append(f"{case_id} missing input/error message")
        if row.get("safe_exit") != "true" or row.get("output_consumed") != "false":
            errors.append(f"{case_id} unsafe exit/output consumption")
        if row.get("status") != "PASS":
            errors.append(f"{case_id} status={row.get('status')}")
        backend = tuple(row.get(key) for key in
                        ("backend_error_code", "backend_error_name", "backend_error_message"))
        if backend != ("NA", "NA", "NA"):
            errors.append(f"{case_id} pre-backend validation must use NA triplet={backend}")
    if args.expected_cases == 6 and seen != set(catalog):
        errors.append(f"full result case set mismatch missing={sorted(set(catalog)-seen)}")
    if errors:
        for error in errors:
            print("TASK1_RESULT_FAIL " + error)
        return 1
    print(f"TASK1_RESULT_PASS cases={len(results)} pass={len(results)} fail=0 "
          f"safe_exit={len(results)} output_consumed=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
