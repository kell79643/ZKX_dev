#!/usr/bin/env python3
"""代表性负向门禁：缺外键触发真实runner退出码2时/demo必须FAIL并清理。"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.demo_capabilities import build_demo_capabilities
from execution import abnormal_adapter
from execution.common import CleanTemporaryDirectory


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",required=True)
    args=parser.parse_args()
    caps=build_demo_capabilities()
    allowed=caps["menu"]["backends"]
    if args.backend not in allowed: raise SystemExit(f"unsupported backend: {args.backend}")
    case=next(row for row in caps["abnormal_cases"]["operator"] if row["module"]=="windows" and row["operator"]=="hamming")
    temporary_path=None
    with CleanTemporaryDirectory() as root:
        temporary_path=root
        links=root/"normal_case_links.csv"
        with links.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["module_name","operator_name","normal_run_id","normal_case_id"])
            writer.writerow(["bsplines","cubic","negative_test_run","negative_test_case"])
        result=abnormal_adapter.execute("operator",args.backend,case,links,root/"abnormal")
        if result.get("status")!="FAIL": raise SystemExit(f"missing-link gate accepted invalid run: {result}")
        if result.get("config",{}).get("exit_code")!=2: raise SystemExit(f"expected real runner exit_code=2: {result}")
        print("MISSING_LINK_GATE_PASS status=FAIL exit_code=2")
    if temporary_path is None or temporary_path.exists(): raise SystemExit("temporary directory cleanup failed")
    print("ABNORMAL_TEMP_CLEANUP_PASS")
    return 0


if __name__=="__main__": raise SystemExit(main())
