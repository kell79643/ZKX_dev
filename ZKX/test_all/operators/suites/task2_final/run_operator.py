#!/usr/bin/env python3
import argparse,csv,hashlib,importlib.util,json,pathlib,subprocess,sys
S=pathlib.Path(__file__).resolve().parent; sys.path.insert(0,str(S))
from batch06_config import expand_config
C=S.parents[1]/"cases"/"filtering"/"channelize_poly"; sys.path.insert(0,str(C)); import run_operator as common
B=S.parents[1]/"shared"/"filtering"/"run_operator.py"
spec=importlib.util.spec_from_file_location("b04",B); b04=importlib.util.module_from_spec(spec); spec.loader.exec_module(b04)
OPS={"cubic":"Step3","argrelextrema":"Step5","kalman_filter":"Step6"}; METRICS=("mse","rmse","relative_l2","relative_linf")

def write(path,cols,rows): common.write_csv(path,cols,rows)
def main():
 p=argparse.ArgumentParser()
 for n in ("binary","operator","config","thresholds","results-root","scale","run-type","git-commit","git-dirty","test-time-utc","random"): p.add_argument("--"+n,required=True)
 a=p.parse_args(); assert a.operator in OPS and a.run_type in ("smoke","formal")
 cb=pathlib.Path(a.config).read_bytes(); cfg=json.loads(cb); ch=hashlib.sha256(cb).hexdigest()
 scale=next(x for x in expand_config(cfg) if x["scale_id"]==a.scale); assert scale["operator_name"]==a.operator
 dtype=scale["inputs"][0]["dtype"]; prm=scale["parameters"]; discrete=a.operator=="argrelextrema"
 thresholds=json.loads(pathlib.Path(a.thresholds).read_text()); threshold=None if discrete else next(x for x in thresholds["entries"] if x["target"]==a.operator and x["dtype"]==dtype)
 warm,measured=(20,100) if a.run_type=="formal" else (1,1)
 rid=f"{a.test_time_utc}_{a.git_commit[:12]}_operator_case_b06_{a.run_type}_{a.random}"; out=pathlib.Path(getattr(a,"results_root"))/a.run_type/rid
 if out.exists(): raise RuntimeError("result exists")
 out.mkdir(parents=True); rawp=out/f"{a.operator}_probe_raw.json"
 cmd=[a.binary,"--dtype",dtype,"--count",str(prm.get("count",prm.get("observation_count"))),"--order",str(prm.get("order",3)),"--warmup",str(warm),"--measured",str(measured),"--output",str(rawp)]
 done=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); log=out/f"operator_{a.operator}_full.log"; log.write_text("command="+" ".join(cmd)+"\n"+done.stdout)
 if done.returncode: raise RuntimeError("probe failed: "+done.stdout)
 raw=json.loads(rawp.read_text()); accuracy_ok=(raw["exact_match"] and raw["mismatch_count"]==0) if discrete else all(raw[m]<=threshold[m+"_max"] for m in METRICS)
 semantic_ok=len(raw["cpu_samples_ms"])==measured and len(raw["gpu_samples_ms"])==measured and (raw["output_count"]>=0 if discrete else raw["output_count"]==prm["output_count"])
 sums=[common.resource_summary(raw[k],m) for k,m in (("cpu_trace","heap"),("cpu_trace","rss"),("gpu_trace","heap"),("gpu_trace","rss"),("gpu_trace","gpu"))]
 resource_ok=all(x is not None for x in sums); status="PASS" if accuracy_ok and semantic_ok and resource_ok else "FAIL"; code="OK" if status=="PASS" else "RESOURCE_FAILED" if not resource_ok else "ACCURACY_FAILED"
 case=f"{a.operator}__{a.scale}__{raw['actual_input_digest'][:16]}"
 def ident(dev):
  vals=["1",rid,a.run_type,a.test_time_utc,a.git_commit,a.git_dirty,cfg["scale_set_id"],ch,"operator",a.operator,"Task2",OPS[a.operator],a.operator,case,a.scale,scale["order_of_magnitude"],str(scale["actual_elements"]),"|".join(i["input_name"]+"="+str(i["shape"]) for i in scale["inputs"]),dtype,dev,"not_applicable","compatibility/end-to-end",status,code]
  return dict(zip(common.IDENTITY,vals))
 cs,gs=common.stats(raw["cpu_samples_ms"]),common.stats(raw["gpu_samples_ms"]); rows=[]
 for dev,st,heap,rss,gpu in (("CPU",cs,sums[0],sums[1],None),("GPU",gs,sums[2],sums[3],sums[4])):
  r=ident(dev); r.update({k:str(v) for k,v in st.items()}); gpurow=dev=="GPU"
  r.update({"warmup_runs":str(warm),"measured_runs":str(measured),"cpu_gpu_speedup":str(cs["mean_ms"]/gs["mean_ms"]) if gpurow else "NA","accuracy_reference":"typed_cpu_task2_reference","mse":str(raw["mse"]) if gpurow and not discrete else "NA" if discrete else "0","rmse":str(raw["rmse"]) if gpurow and not discrete else "NA" if discrete else "0","relative_l2":str(raw["relative_l2"]) if gpurow and not discrete else "NA" if discrete else "0","relative_linf":str(raw["relative_linf"]) if gpurow and not discrete else "NA" if discrete else "0","exact_match":str(raw["exact_match"]).lower() if gpurow and discrete else "NA","mismatch_count":str(raw["mismatch_count"]) if gpurow and discrete else "NA","semantic_check":f"output_count={raw['output_count']};output_dtype={raw['output_dtype']};operator_matrixpu_call=present","accuracy_status":"PASS" if accuracy_ok and semantic_ok else "FAIL"})
  for pre,sm in (("cpu_heap",heap),("rss",rss),("gpu",gpu)):
   for f in ("before","after","peak","delta"): r[f"{pre}_{f}_bytes"]=str(sm[f]) if sm else "NA"
  for f in common.SUMMARY:r.setdefault(f,"NA")
  rows.append(r)
 mainp=out/f"operator_main_results_{a.operator}_not_applicable_{rid}.csv"; write(mainp,common.IDENTITY+common.SUMMARY,rows)
 timing=[]
 for dev,samples,od in (("CPU",raw["cpu_samples_ms"],raw["cpu_output_digest"]),("GPU",raw["gpu_samples_ms"],raw["gpu_output_digest"])):
  for i,v in enumerate(samples): r=ident(dev); r.update({"warmup_runs":str(warm),"measured_runs":str(measured),"sample_index":str(i),"latency_ms":str(v),"synchronized":"true","input_digest":raw["actual_input_digest"],"output_digest":od}); timing.append(r)
 tp=out/f"operator_timing_samples_{a.operator}_not_applicable_{rid}.csv"; write(tp,common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"],timing)
 memory=[]
 for dev,tr in (("CPU",raw["cpu_trace"]),("GPU",raw["gpu_trace"])):
  for x in tr:
   r=ident(dev); r.update({"sample_index":"0","trace_phase":x["phase"],"phase_index":str(x["phase_index"]),"cpu_live_heap_bytes":str(x["heap"]["bytes"]) if x["heap"]["available"] else "NA","rss_bytes":str(x["rss"]["bytes"]) if x["rss"]["available"] else "NA","gpu_used_bytes":str(x["gpu"]["bytes"]) if x["gpu"]["available"] else "NA"}); memory.append(r)
 mp=out/f"operator_{scale['module_name']}_{a.operator}_memory_trace_not_applicable.csv"; write(mp,common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"],memory)
 content=[{"input_name":scale["inputs"][0]["input_name"],"shape":str(scale["inputs"][0]["shape"]),"dtype":dtype,"element_count":str(scale["inputs"][0]["element_count"]),"generator":{"cubic":"deterministic_cubic_domain","argrelextrema":"deterministic_task2_feature_bundle","kalman_filter":"deterministic_task2_observations"}[a.operator],"preview_heacuda_api_tail2":raw["preview0"]}]
 de={"run_id":rid,"case_id":case,"operator_name":a.operator,"requested_dtype":dtype,"config_input_dtype":dtype,"execution_dtype":prm.get("execution_dtype",dtype),"typed_input_manifest":raw["typed_input_manifest"],"typed_array_digest":raw["typed_array_digest"],"input_content_json":json.dumps(content,sort_keys=True,separators=(",",":")),"case_parameters_json":json.dumps(prm,sort_keys=True,separators=(",",":")),"actual_input_digest":raw["actual_input_digest"],"output_dtype":raw["output_dtype"],"backend":"not_applicable","operator_matrixpu_call_status":"proved_gpu_cpu","status":status,"error_code":code}
 dp=out/f"operator_dtype_evidence_{a.operator}_not_applicable_{rid}.csv"; write(dp,list(de),[de])
 report={"run_id":rid,"case_id":case,"operator_name":a.operator,"task_name":"Task2","step_name":OPS[a.operator],"scale_id":a.scale,"requested_dtype":dtype,"input_content":content,"case_parameters":prm,"output_dtype":raw["output_dtype"],"backend":"not_applicable","warmup_runs":warm,"measured_runs":measured,"output_count":raw["output_count"],"actual_input_digest":raw["actual_input_digest"],"typed_array_digest":raw["typed_array_digest"],"status":status,"error_code":code}
 if discrete: report.update({"exact_match":raw["exact_match"],"mismatch_count":raw["mismatch_count"],"semantic_check":semantic_ok})
 else: report.update({m:raw[m] for m in METRICS})
 rp=out/f"operator_{a.operator}_report.json"; rp.write_text(json.dumps(report,indent=2)+"\n"); sp=out/f"operator_{a.operator}_summary.txt"; sp.write_text("\n".join(f"{k}={v}" for k,v in report.items())+"\n")
 paths=[("main_csv",mainp),("timing_csv",tp),("memory_csv",mp),("dtype_csv",dp),("report_json",rp),("summary_txt",sp),("full_log",log)]
 terminal=b04.terminal_summary(mainp,paths); log.write_text(log.read_text()+json.dumps(report)+"\n"+terminal+"\n"); print(terminal); print(f"OPERATOR_CASE_OPERATOR_BENCHMARKINAL_CASE {status} operator={a.operator} scale={a.scale} dtype={dtype} backend=not_applicable measured={measured} results={out}")
 return 0 if status=="PASS" else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
