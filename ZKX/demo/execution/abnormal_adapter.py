"""调用现有异常runner并严格验证单条结果；不复制异常业务逻辑。"""
from __future__ import annotations
import csv
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from .common import ROOT, CleanTemporaryDirectory, run_process, verify_published_binary

OPERATOR_ABNORMAL_ROOT=ROOT/"test_all/operators/abnormal"


def _rows(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def _contract(target: str, catalog: Path, case_id: str) -> dict[str,str]|None:
    key="abnormal_case_id" if target=="operator" else "case_id"
    matches=[row for row in _rows(catalog) if row.get(key)==case_id]
    return matches[0] if len(matches)==1 else None


def _validate_row(target: str, row: dict[str,str], contract: dict[str,str], case: dict[str,str]) -> list[str]:
    errors=[]
    if row.get("case_id")!=case["case_id"]: errors.append("结果case_id与请求不一致")
    if row.get("abnormal_type")!=contract.get("abnormal_type"): errors.append("abnormal_type不一致")
    if target=="operator" and (row.get("module_name"),row.get("operator_name"))!=(case["module"],case["operator"]): errors.append("module/operator不一致")
    if target in ("task1","task2") and row.get("task_name")!=target.capitalize(): errors.append("task_name不一致")
    expected=contract.get("expected_code","")
    codes=tuple(row.get(name,"") for name in ("expected_code","captured_code","returned_code","actual_code"))
    if not expected or any(code!=expected for code in codes): errors.append(f"错误码不一致：expected={expected} actual={codes}")
    if row.get("status")!="PASS": errors.append(f"status={row.get('status','')}")
    if row.get("safe_exit")!="true": errors.append(f"safe_exit={row.get('safe_exit','')}")
    if row.get("output_consumed")!="false": errors.append(f"output_consumed={row.get('output_consumed','')}")
    if not row.get("error_message") or not row.get("input_summary"): errors.append("缺少输入或错误说明")
    return errors


def resolve_normal_links(results_root: Path, output: Path, module: str|None=None, operator: str|None=None) -> tuple[bool,str]:
    command=[sys.executable,str(OPERATOR_ABNORMAL_ROOT/"resolve_normal_case_links.py"),"--results-root",str(results_root),"--catalog",str(OPERATOR_ABNORMAL_ROOT/"operator_abnormal_case_catalog.csv"),"--output",str(output)]
    if module and operator: command += ["--module",module,"--operator",operator]
    process=run_process(command)
    expected=1 if module else 53
    if process["timed_out"] or process["returncode"]!=0 or not output.is_file():
        return False,(process["stdout"]+process["stderr"])[-1600:]
    try: rows=_rows(output)
    except (OSError,UnicodeDecodeError,csv.Error) as caught: return False,f"临时外键无法解析：{caught}"
    identities={(row.get("module_name"),row.get("operator_name")) for row in rows}
    if len(rows)!=expected or len(identities)!=expected or any(not all(identity) for identity in identities):
        return False,f"临时外键数量或唯一性错误：rows={len(rows)} unique={len(identities)} expected={expected}"
    marker=f"NORMAL_LINK_PASS operators={expected}"
    if marker not in process["stdout"]: return False,f"resolver缺少成功标志：{process['stdout'][-800:]}"
    return True,process["stdout"].strip()


def merge_and_verify(result_csvs: list[Path], output_dir: Path, normal_links: Path) -> tuple[bool,str]:
    if len(result_csvs)!=159 or len(set(result_csvs))!=159:
        return False,f"异常结果文件必须为159个唯一文件：actual={len(result_csvs)} unique={len(set(result_csvs))}"
    merge=[sys.executable,str(OPERATOR_ABNORMAL_ROOT/"merge_results.py"),"--inputs",*[str(path) for path in result_csvs],"--output-dir",str(output_dir)]
    merged=run_process(merge)
    if merged["timed_out"] or merged["returncode"]!=0 or "MERGE_PASS cases=159" not in merged["stdout"]:
        return False,(merged["stdout"]+merged["stderr"])[-2000:]
    outputs=sorted(output_dir.glob("operator_abnormal_cases_*.csv"))
    if len(outputs)!=3: return False,f"合并结果必须为三个后端文件：actual={len(outputs)}"
    verify=[sys.executable,str(OPERATOR_ABNORMAL_ROOT/"verify_results.py"),"--catalog",str(OPERATOR_ABNORMAL_ROOT/"operator_abnormal_case_catalog.csv"),"--normal-links",str(normal_links),"--results",*[str(path) for path in outputs]]
    verified=run_process(verify)
    expected="RESULT_PASS operators=53 cases=159 pass=159 fail=0 safe_exit=159 output_consumed=0 normal_links=53"
    if verified["timed_out"] or verified["returncode"]!=0 or expected not in verified["stdout"]:
        return False,(verified["stdout"]+verified["stderr"])[-2400:]
    return True,merged["stdout"].strip()+"\n"+verified["stdout"].strip()


def execute(target: str, backend: str, case: dict[str,str], normal_links: Path|None=None, output_root: Path|None=None) -> dict[str,Any]:
    name={"task1":"task1_abnormal_input","task2":"task2_abnormal_input","operator":"operator_abnormal_input"}[target]
    binary,error=verify_published_binary(backend,backend,f"bin/{name}")
    if error or binary is None: return {"status":"FAIL","target":target,"message":error or "异常入口构建预检失败"}
    if target=="operator" and (normal_links is None or not normal_links.is_file()):
        return {"status":"FAIL","target":target,"message":"缺少本轮正常结果外键"}
    catalog=ROOT/("test_all/operators/abnormal/operator_abnormal_case_catalog.csv" if target=="operator" else f"test_all/tasks/abnormal/{target}/{target}_abnormal_case_catalog.csv")
    contract=_contract(target,catalog,case["case_id"])
    if contract is None: return {"status":"FAIL","target":target,"message":"异常catalog未唯一匹配请求case"}
    if output_root is not None:
        output_root.mkdir(parents=True,exist_ok=False)
    output_context=CleanTemporaryDirectory() if output_root is None else nullcontext(output_root)
    retained_csv=None
    with output_context as output:
        result_csv=output/"result.csv"
        runner_backend="all" if target=="operator" else backend
        command=[str(binary),"--timestamp",time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()),"--run-id",f"demo_abnormal_{time.time_ns()}","--backend",runner_backend,"--case-id",case["case_id"]]
        if target=="operator":
            command += ["--catalog",str(catalog),"--normal-links",str(normal_links),"--output",str(result_csv),"--module",case["module"],"--operator",case["operator"]]
        else:
            command += ["--catalog",str(catalog),"--fixture-root",str(ROOT/f"test_all/tasks/abnormal/{target}/fixtures"),"--output-dir",str(output)]
        process=run_process(command)
        combined=process["stdout"]+process["stderr"]
        csvs=sorted(output.rglob("*.csv"))
        errors=[]
        if process["timed_out"]: errors.append("异常进程安全超时")
        if process["returncode"]!=0: errors.append(f"异常runner退出码={process['returncode']}")
        if len(csvs)!=1: errors.append(f"结果CSV数量必须为1，实际={len(csvs)}")
        rows=[]
        if len(csvs)==1:
            try: rows=_rows(csvs[0])
            except (OSError,UnicodeDecodeError,csv.Error) as caught: errors.append(f"结果CSV无法解析：{caught}")
        if len(rows)!=1: errors.append(f"结果CSV必须恰好包含请求case，实际行数={len(rows)}")
        if len(rows)==1: errors.extend(_validate_row(target,rows[0],contract,case))
        status="PASS" if not errors else "FAIL"
        result={"status":status,"target":target,"message":"程序正确识别异常并通过严格合同核验" if not errors else "; ".join(errors),"config":{"backend":backend,"case_id":case["case_id"],"abnormal_type":case["abnormal_type"],"exit_code":process["returncode"]},"details":combined[-1200:]}
        if output_root is not None and len(csvs)==1:
            retained_csv=str(csvs[0]); result["_result_csv"]=retained_csv
    result["cleanup_verified"]=output_root is None
    return result
