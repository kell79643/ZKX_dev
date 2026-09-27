"""调用现有 Task benchmark；不复制 pipeline 业务。"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
from .common import ROOT, CleanTemporaryDirectory, first_number, rows_from_csvs, run_process, verify_published_binary

def _summarize(rows: list[dict[str,str]], target: str, scale_id: str, backend: str, default_input_profile_backend: str, wall_ms: float, scales_path: Path) -> dict[str,Any]:
    summary=[r for r in rows if "pipeline_summary" in r.get("_source","")]
    accuracy_rows=[r for r in rows if "accuracy_details" in r.get("_source","")]
    timing=[]
    for scope in sorted({r.get("component") or r.get("step_name") or "total" for r in summary}):
        scoped=[r for r in summary if (r.get("component") or r.get("step_name") or "total")==scope]
        cpu=next((first_number(r,["mean_ms","latency_ms"]) for r in scoped if r.get("device")=="cpu"),None)
        gpu=next((first_number(r,["mean_ms","latency_ms"]) for r in scoped if r.get("device")=="gpu"),None)
        timing.append({"scope":scope,"cpu_ms":cpu,"gpu_ms":gpu,"speedup":cpu/gpu if cpu is not None and gpu not in (None,0) else None})
    accuracy=[]
    step_names=sorted({".".join(r.get("target","").split(".")[:2]) for r in accuracy_rows if r.get("target")})
    for scope in step_names+[f"{target.capitalize()}.pipeline_total"]:
        scoped=accuracy_rows if scope.endswith("pipeline_total") else [r for r in accuracy_rows if r.get("target","").startswith(scope+".")]
        def maximum(name: str):
            values=[first_number(r,[name]) for r in scoped]; values=[x for x in values if x is not None]
            return max(values) if values else None
        discrete=[r for r in scoped if r.get("output_kind")=="discrete"]
        exact="-"
        if discrete:
            mismatch=sum(int(float(r.get("mismatch_count","0") or 0)) for r in discrete)
            exact=f"exact={str(all(r.get('exact_match','').lower()=='true' for r in discrete)).lower()}, mismatch={mismatch}"
        accuracy.append({"scope":scope,"mse":maximum("mse"),"rmse":maximum("rmse"),"relative_l2":maximum("relative_l2"),"relative_linf":maximum("relative_linf"),"discrete":exact,"status":"PASS" if all(r.get("accuracy_status")=="PASS" for r in scoped) else "WARNING"})
    main=[r for r in rows if f"{target}_benchmark" in r.get("_source","")]
    cpu=next((r for r in main if r.get("device","").lower()=="cpu"),None); gpu=next((r for r in main if r.get("device","").lower()=="gpu"),None)
    if cpu is None or gpu is None: return {"status":"FAIL","target":target,"message":"CPU或GPU没有产生结果"}
    if not cpu.get("input_shape") or cpu.get("status")!="PASS" or gpu.get("status")!="PASS": return {"status":"FAIL","target":target,"message":"输出shape为空或正常case执行失败"}
    resources=[]
    for row in (cpu,gpu):
        device=row.get("device","")
        for metric in ("cpu_heap_peak_bytes","rss_peak_bytes","gpu_peak_bytes","cpu_heap_delta_bytes","rss_delta_bytes","gpu_delta_bytes"):
            if row.get(metric) not in (None,"","NA"): resources.append({"metric":f"{device}.{metric}","value":row[metric],"status":"INFO"})
    payload=json.loads(scales_path.read_text(encoding="utf-8")); task_name=target.capitalize(); scale=next(x for x in payload["tasks"][task_name]["scales"] if x["scale_id"]==scale_id)
    config={"backend":backend,"default_input_profile_backend":default_input_profile_backend,"scale_id":scale_id,"input_shape":cpu.get("input_shape","-"),"dtype":cpu.get("dtype","-"),"warmup_runs":cpu.get("warmup_runs", "-"),"measured_runs":cpu.get("measured_runs", "-"),"wall_ms":f"{wall_ms:.3f}"}
    for name,value in scale.get("parameters",{}).items(): config[f"parameter.{name}"]=value
    warnings=[]
    if any(x["status"]!="PASS" for x in accuracy): warnings.append({"level":"WARNING","message":"精度阈值异常；业务输出仍已产生"})
    return {"status":"PASS","target":target,"message":"Task pipeline执行完成", "config":config,"timing":timing,"accuracy":accuracy,"resources":resources,"warnings":warnings}

def execute(target: str, backend: str, scale_id: str, scales_path: Path, runs: int, warmup_runs: int, default_input_profile_backend: str) -> dict[str,Any]:
    binary,error=verify_published_binary(backend,backend,f"bin/{target}_pipeline_benchmark")
    if error or binary is None: return {"status":"FAIL","target":target,"message":error or "Task构建预检失败"}
    with CleanTemporaryDirectory() as temporary_root:
        output=temporary_root/"result"
        command=[str(binary),"--task-scales",str(scales_path),"--accuracy-thresholds",str(ROOT/"test_all/config/task_benchmarks/accuracy_thresholds.json"),"--scale-id",scale_id,"--backend",backend,"--run-id",f"demo_{target}_{int(time.time())}","--run-type","smoke","--warmup-runs",str(warmup_runs),"--measured-runs",str(runs),"--output-dir",str(output)]
        process=run_process(command)
        rows=rows_from_csvs(output)
        if process["timed_out"]: result={"status":"FAIL","target":target,"message":"安全超时"}
        elif process["returncode"]!=0: result={"status":"FAIL","target":target,"message":f"pipeline退出码={process['returncode']}","details":(process["stdout"]+process["stderr"])[-2000:]}
        elif not rows: result={"status":"FAIL","target":target,"message":"业务输出为空"}
        else: result=_summarize(rows,target,scale_id,backend,default_input_profile_backend,process["wall_ms"],scales_path)
    result["cleanup_verified"]=True
    return result
