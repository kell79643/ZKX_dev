#!/usr/bin/env python3
"""只读核验演示发布树和demo_build_manifest.json。"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; BUILD=ROOT/"build/demo"; PLAN=ROOT/"demo/config/build/demo_build_plan.json"
def sha(path:Path)->str:
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def main()->int:
 plan=json.loads(PLAN.read_text(encoding="utf-8"))
 p=argparse.ArgumentParser(description="只读核验指定后端的演示构建发布树；该参数不选择现场运行后端")
 p.add_argument("--backend",choices=tuple(plan["supported_backends"]),required=True,help="要核验的build/demo/<backend>发布树，不会传入交互菜单或统一runner")
 a=p.parse_args()
 tree=BUILD/a.backend; path=tree/"demo_build_manifest.json"; status_path=tree/"demo_build_status.json"
 if not status_path.is_file(): print(f"[DEMO][BUILD_VERIFY] backend={a.backend} status=FAIL reason=build_status_missing"); return 1
 try: build_status=json.loads(status_path.read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError) as error:
  print(f"[DEMO][BUILD_VERIFY] backend={a.backend} status=FAIL reason=build_status_invalid detail={error}"); return 1
 if build_status.get("backend")!=a.backend or build_status.get("state")!="PASS":
  print(f"[DEMO][BUILD_VERIFY] backend={a.backend} status=FAIL reason=latest_build_not_pass state={build_status.get('state','UNKNOWN')} detail={build_status.get('message','')}"); return 1
 if not path.is_file(): print(f"[DEMO][BUILD_VERIFY] backend={a.backend} status=FAIL reason=manifest_missing"); return 1
 data=json.loads(path.read_text(encoding="utf-8")); errors=[]; warnings=[]; forbidden=(r"task_f",r"cusignal_task_f",r"stage[0-9]+",r"[0-9a-f]{7,40}")
 for item in data.get("artifacts",[])+data.get("libraries",[]):
  relative=item["published_path"]; target=BUILD/relative
  if any(re.search(pattern,relative) for pattern in forbidden): errors.append(f"禁止的公开名称：{relative}")
  elif not target.is_file(): errors.append(f"文件不存在：{relative}")
  elif target.stat().st_size!=item["file_size"] or sha(target)!=item["sha256"]: warnings.append(f"SHA-256或大小与追溯清单不一致：{relative}")
 print(f"[DEMO][BUILD_VERIFY] backend={a.backend} artifacts={len(data.get('artifacts',[]))} warnings={len(warnings)} status={'FAIL' if errors else 'PASS'}")
 for error in errors: print(f"  {error}")
 for warning in warnings: print(f"  WARNING: {warning}")
 return 1 if errors else 0
if __name__=="__main__": raise SystemExit(main())
