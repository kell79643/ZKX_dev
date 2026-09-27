"""调用真实算子 runner；单算子与53算子入口共同复用。"""
from __future__ import annotations
import sys, time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from .common import ROOT, BUILD_ROOT, CleanTemporaryDirectory, first_number, load_json, rows_from_csvs, run_process, verify_published_binary

def execute(command_entry: dict[str,Any], profile: dict[str,Any], selected_backend: str, runs: int, results_root: Path|None=None) -> dict[str,Any]:
    module,operator=command_entry["module"],command_entry["operator"]
    tree=command_entry["binary_tree"]; binary,error=verify_published_binary(selected_backend,tree,command_entry["binary"])
    if error or binary is None: return {"status":"FAIL","target":f"{module}.{operator}","message":error or "算子构建预检失败"}
    runner=ROOT/command_entry["runner_path"] if command_entry.get("runner_path") else None
    if runner is not None and not runner.is_file(): return {"status":"FAIL","target":f"{module}.{operator}","message":f"runner不存在：{runner}"}
    manifest=BUILD_ROOT/selected_backend/"demo_build_manifest.json"
    if not manifest.is_file(): return {"status":"FAIL","target":f"{module}.{operator}","message":"demo_build_manifest.json不存在"}
    commit=load_json(manifest).get("git_commit","")
    if results_root is not None:
        results_root.mkdir(parents=True,exist_ok=False)
    output_context=CleanTemporaryDirectory() if results_root is None else nullcontext(results_root)
    with output_context as output:
        if command_entry["runner_kind"]=="shell_wrapper":
            command=["bash",str(runner),str(binary),profile["best_scale_id"],"smoke",commit,"false",str(output)]
            if command_entry.get("runner_backend_argument"): command.append(selected_backend)
            config_path=Path(command_entry["config"]); config_path=config_path if config_path.is_absolute() else ROOT/config_path
            command.append(str(config_path))
        elif command_entry["runner_kind"]=="windows_binary":
            symmetric="false" if profile["case_id"].endswith("__periodic") else "true"
            config_path=Path(command_entry["config"]); config_path=config_path if config_path.is_absolute() else ROOT/config_path
            command=[str(binary),"--config",str(config_path),"--thresholds",str(ROOT/command_entry["thresholds"]),"--results-root",str(output),"--scale",profile["best_scale_id"],"--sym",symmetric,"--run-type","smoke","--git-commit",commit,"--git-dirty","false","--test-time-utc",time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()),"--random","d3e0cafe"]
        else:
            config_path=Path(command_entry["config"]); config_path=config_path if config_path.is_absolute() else ROOT/config_path
            command=[sys.executable,str(runner),"--binary",str(binary)]
            if command_entry.get("runner_backend_argument"): command.extend(["--backend",selected_backend])
            command.extend(["--config",str(config_path),"--thresholds",str(ROOT/command_entry["thresholds"]),"--results-root",str(output),"--scale",profile["best_scale_id"],"--run-type","smoke","--git-commit",commit,"--git-dirty","false","--test-time-utc",time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()),"--random","d3e0cafe"])
        process=run_process(command)
        rows=rows_from_csvs(output)
        main=[r for r in rows if "main_results" in r.get("_source","")]
        expected_runtime_backend=command_entry["runtime_backend"]
        if process["timed_out"]: result={"status":"FAIL","target":f"{module}.{operator}","message":"安全超时"}
        elif process["returncode"]!=0:
            diagnostics="\n".join(x for x in (process["stderr"].strip(),process["stdout"].strip()) if x)
            result={"status":"FAIL","target":f"{module}.{operator}","message":f"算子退出码={process['returncode']}","details":diagnostics[-4000:]}
        elif not main: result={"status":"FAIL","target":f"{module}.{operator}","message":"业务输出为空"}
        elif any(row.get("backend")!=expected_runtime_backend for row in main): result={"status":"FAIL","target":f"{module}.{operator}","message":f"运行结果backend身份错误：expected={expected_runtime_backend}"}
        else:
            cpu_row=next((r for r in main if r.get("device","").lower()=="cpu"),None); gpu_row=next((r for r in main if r.get("device","").lower()=="gpu"),None)
            row=gpu_row or main[0]; cpu=first_number(cpu_row or {},["mean_ms","cpu_mean_ms"]); gpu=first_number(gpu_row or {},["mean_ms","gpu_mean_ms"])
            if cpu is None or gpu is None: result={"status":"FAIL","target":f"{module}.{operator}","message":"CPU或GPU没有产生结果"}
            else:
                mismatch=row.get("mismatch_count","0"); exact=row.get("exact_match","true").lower()
                warnings=[]
                if exact=="false" or mismatch not in ("","0","0.0","NA"): warnings.append({"level":"WARNING-HIGH","message":f"离散输出不完全匹配：exact_match={exact} mismatch_count={mismatch}"})
                elif row.get("accuracy_status")!="PASS": warnings.append({"level":"WARNING","message":"精度阈值异常；业务输出仍已产生"})
                result={"status":"PASS","target":f"{module}.{operator}","message":"算子执行完成","config":{"backend":selected_backend,"default_input_profile_backend":profile["backend"],"dtype":profile["dtype"],"case_id":profile["case_id"],"scale_id":profile["best_scale_id"],"wall_ms":f"{process['wall_ms']:.3f}"},"timing":[{"scope":"operator","cpu_ms":cpu,"gpu_ms":gpu,"speedup":cpu/gpu if gpu else None}],"accuracy":[{"scope":"operator","mse":row.get("mse","-"),"rmse":row.get("rmse","-"),"relative_l2":row.get("relative_l2","-"),"relative_linf":row.get("relative_linf","-"),"status":row.get("accuracy_status","-")}],"resources":[{"metric":"gpu_peak_bytes","value":row.get("gpu_peak_bytes","-"),"status":"INFO"},{"metric":"rss_peak_bytes","value":row.get("rss_peak_bytes","-"),"status":"INFO"}],"warnings":warnings}
    result["cleanup_verified"]=results_root is None
    return result
