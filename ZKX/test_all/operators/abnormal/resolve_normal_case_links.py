#!/usr/bin/env python3
"""从本轮正常FP32结果生成异常测试外键；不读取历史inventory。"""
import argparse
import csv
import pathlib
import re
import sys


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_operator(value):
    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    if compact == "kalmanfilter":
        return "kalman_filter"
    return value.strip().lower()


def fp32(row):
    for key in ("requested_dtype", "dtype", "input_dtype", "config_input_dtype"):
        value = row.get(key, "").upper()
        if value:
            return value == "FP32"
    return False


def passed(row):
    return row.get("status", "").strip().upper() in {"PASS", "PASSED", "OBJECT_PASS"}


def scale_score(row):
    value = row.get("actual_elements", "")
    if value.isdigit():
        return int(value)
    match = re.search(r"10e([0-9]+)", row.get("case_id", ""))
    return 10 ** int(match.group(1)) if match else 10**18


def catalog_identities(path):
    rows = read_csv(path)
    by_identity = {}
    case_ids = set()
    for row in rows:
        identity = (row.get("module_name", ""), normalized_operator(row.get("operator_name", "")))
        case_id = row.get("abnormal_case_id", "")
        if not all(identity) or not case_id or case_id in case_ids:
            raise SystemExit("catalog contains empty/duplicate identity or case_id")
        case_ids.add(case_id)
        by_identity.setdefault(identity, []).append(row)
    if len(rows) != 159 or len(case_ids) != 159 or len(by_identity) != 53:
        raise SystemExit(f"catalog must contain 159 unique cases and 53 operators: rows={len(rows)} cases={len(case_ids)} operators={len(by_identity)}")
    invalid = [identity for identity, cases in by_identity.items() if len(cases) != 3]
    if invalid:
        raise SystemExit(f"catalog operators must each contain three cases: {invalid[:3]}")
    operators = [identity[1] for identity in by_identity]
    if len(set(operators)) != 53:
        raise SystemExit("catalog operator_name values must be globally unique")
    return by_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=pathlib.Path)
    parser.add_argument("--catalog", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--module")
    parser.add_argument("--operator")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refuse overwrite: {args.output}")
    identities = catalog_identities(args.catalog)
    if bool(args.module) != bool(args.operator):
        raise SystemExit("--module and --operator must be provided together")
    if args.module:
        selected = (args.module, normalized_operator(args.operator))
        if selected not in identities:
            raise SystemExit(f"selected operator is absent from catalog: {selected}")
        identities = {selected: identities[selected]}
    by_operator = {identity[1]: identity for identity in identities}
    candidates = {identity: [] for identity in identities}
    for path in args.results_root.rglob("operator_dtype_evidence_*.csv"):
        try:
            rows = read_csv(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        for row in rows:
            operator = normalized_operator(row.get("operator_name", ""))
            identity = by_operator.get(operator)
            if identity is None or not fp32(row) or not passed(row):
                continue
            run_id = row.get("run_id", "").strip()
            case_id = row.get("case_id", "").strip()
            if run_id and case_id:
                candidates[identity].append((scale_score(row), str(path), run_id, case_id))
    missing = [".".join(identity) for identity, rows in candidates.items() if not rows]
    if missing:
        print("NORMAL_LINK_FAIL missing=" + ",".join(sorted(missing)))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module_name", "operator_name", "normal_run_id", "normal_case_id"])
        for identity in sorted(identities):
            _, _, run_id, case_id = min(candidates[identity])
            writer.writerow([identity[0], identity[1], run_id, case_id])
    print(f"NORMAL_LINK_PASS operators={len(identities)} output={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
