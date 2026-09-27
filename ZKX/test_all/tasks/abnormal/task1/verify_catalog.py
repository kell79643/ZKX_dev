#!/usr/bin/env python3
import csv
import pathlib
import sys


EXPECTED = {
    "config_error": "CONFIG_ERROR",
    "upstream_step_failed": "UPSTREAM_STEP_FAILED",
    "invalid_parameter": "INVALID_ARGUMENT",
    "invalid_dtype": "INVALID_DTYPE",
    "controlled_oom": "OUT_OF_MEMORY",
    "invalid_backend": "UNSUPPORTED_OPERATION",
}


def main():
    path = pathlib.Path(__file__).with_name("task1_abnormal_case_catalog.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if len(rows) != 6:
        errors.append(f"expected 6 cases, got {len(rows)}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case_id")
    semantics = {row["abnormal_type"] for row in rows}
    if semantics != set(EXPECTED):
        errors.append(f"semantic mismatch={sorted(semantics)}")
    for row in rows:
        expected = EXPECTED.get(row["abnormal_type"])
        if row["expected_code"] != expected:
            errors.append(f"{row['case_id']} expected_code must be {expected}")
        if row["implementation_status"] != "implemented":
            errors.append(f"{row['case_id']} is not implemented")
        if not row["input_summary"] or not row["step_name"]:
            errors.append(f"{row['case_id']} missing input/step")
    owners = [row["verification_owner"] for row in rows]
    if owners.count("AI") != 1 or owners.count("developer") != 5:
        errors.append(f"verification owners must be AI=1/developer=5, got {owners}")
    if errors:
        for error in errors:
            print("TASK1_CATALOG_FAIL " + error)
        return 1
    print("TASK1_CATALOG_PASS task=Task1 cases=6 semantics=6 implemented=6 ai_owned=1 developer_owned=5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
