#!/usr/bin/env python3
import argparse
import csv
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs="+", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refuse existing output directory: {args.output_dir}")
    header = None
    groups = {}
    seen = set()
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if header is None:
                header = reader.fieldnames
            elif reader.fieldnames != header:
                raise SystemExit(f"header mismatch: {path}")
            for row in reader:
                case_id = row.get("case_id", "")
                if not case_id or case_id in seen:
                    raise SystemExit(f"empty/duplicate case_id: {case_id}")
                seen.add(case_id)
                groups.setdefault(row["backend"], []).append(row)
    if len(seen) != 159:
        raise SystemExit(f"expected 159 unique rows, got {len(seen)}")
    args.output_dir.mkdir(parents=True)
    outputs = []
    for backend, rows in sorted(groups.items()):
        safe_backend = "".join(character if character.isalnum() or character in "_-" else "_"
                               for character in backend)
        output = args.output_dir / f"operator_abnormal_cases_{safe_backend}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["module_name"],
                                                            row["operator_name"], row["case_id"])))
        outputs.append(f"{backend}:{len(rows)}:{output}")
    print("MERGE_PASS cases=159 files=" + "|".join(outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
