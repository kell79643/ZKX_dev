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
    parser.add_argument("--normal-links", required=True, type=pathlib.Path)
    parser.add_argument("--results", required=True, nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    catalog = read_rows(args.catalog)
    links = read_rows(args.normal_links)
    results = []
    for path in args.results:
        results.extend(read_rows(path))
    errors = []
    catalog_by_case = {row["abnormal_case_id"]: row for row in catalog}
    links_by_operator = {(row["module_name"], row["operator_name"]):
                         (row["normal_run_id"], row["normal_case_id"]) for row in links}
    if len(catalog) != 159 or len(catalog_by_case) != 159:
        errors.append(f"catalog rows/unique must be 159: {len(catalog)}/{len(catalog_by_case)}")
    if len(links_by_operator) != 53:
        errors.append(f"normal links must contain 53 operators: {len(links_by_operator)}")
    if len(results) != 159:
        errors.append(f"result rows must be 159: {len(results)}")
    seen = set()
    for row in results:
        case_id = row.get("case_id", "")
        if case_id in seen:
            errors.append(f"duplicate result case_id={case_id}")
            continue
        seen.add(case_id)
        contract = catalog_by_case.get(case_id)
        if contract is None:
            errors.append(f"unknown result case_id={case_id}")
            continue
        identity = (row.get("module_name", ""), row.get("operator_name", ""))
        expected_link = links_by_operator.get(identity)
        actual_link = (row.get("normal_run_id", ""), row.get("normal_case_id", ""))
        if expected_link is None or actual_link != expected_link:
            errors.append(f"{case_id} normal link mismatch")
        if identity != (contract["module_name"], contract["operator_name"]):
            errors.append(f"{case_id} module/operator mismatch")
        if row.get("abnormal_type") != contract["abnormal_type"]:
            errors.append(f"{case_id} abnormal_type mismatch")
        codes = (row.get("expected_code"), row.get("captured_code"),
                 row.get("returned_code"), row.get("actual_code"))
        if any(code != contract["expected_code"] for code in codes):
            errors.append(f"{case_id} code mismatch={codes}")
        if not row.get("input_summary") or not row.get("error_message"):
            errors.append(f"{case_id} missing input/error message")
        if row.get("safe_exit") != "true" or row.get("output_consumed") != "false":
            errors.append(f"{case_id} unsafe exit/output consumption")
        if row.get("status") != "PASS":
            errors.append(f"{case_id} status={row.get('status')}")
        backend_fields = (row.get("backend_error_code"), row.get("backend_error_name"),
                          row.get("backend_error_message"))
        if contract["abnormal_type"] == "backend_call_failure":
            if any(not value or value == "NA" for value in backend_fields):
                errors.append(f"{case_id} missing backend triplet")
        elif backend_fields != ("NA", "NA", "NA"):
            errors.append(f"{case_id} input validation must use NA backend triplet")
    missing = sorted(set(catalog_by_case) - seen)
    if missing:
        errors.append(f"missing result cases={len(missing)} first={missing[:3]}")
    if errors:
        for error in errors:
            print("RESULT_FAIL " + error)
        return 1
    operators = {(row["module_name"], row["operator_name"]) for row in results}
    print(f"RESULT_PASS operators={len(operators)} cases={len(results)} pass=159 fail=0 "
          "safe_exit=159 output_consumed=0 normal_links=53")
    return 0


if __name__ == "__main__":
    sys.exit(main())
