#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,pathlib,re,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"filtering"/"channelize_poly"))
import run_operator as common

def arguments():
 p=argparse.ArgumentParser();p.add_argument("--binary",required=True);p.add_argument("--config",required=True);p.add_argument("--thresholds",required=True);p.add_argument("--results-root",required=True);p.add_argument("--scale",required=True);p.add_argument("--run-type",choices=("smoke","formal"),required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",choices=("true","false"),required=True);p.add_argument("--test-time-utc",required=True);p.add_argument("--random",required=True);return p.parse_args()
def load(path):
 raw=pathlib.Path(path).read_bytes();return json.loads(raw.decode()),hashlib.sha256(raw).hexdigest()
def ident(a,c,sha,s,run,case,device,status,code):
 shape=";".join(x["input_name"]+"=["+"x".join(map(str,x["shape"]))+"]" for x in s["inputs"])
 return dict(zip(common.IDENTITY,["1",run,a.run_type,a.test_time_utc,a.git_commit,a.git_dirty,c["scale_set_id"],sha,"operator","correlate2d","NA","NA","correlate2d",case,s["scale_id"],s["order_of_magnitude"],str(s["actual_elements"]),shape,s["inputs"][0]["dtype"],device,"not_applicable","compatibility/end-to-end",status,code]))
def summary(trace,key):return common.resource_summary(trace,key)
def main():
 a=arguments()
 if not re.fullmatch(r"[0-9a-f]{40}",a.git_commit) or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z",a.test_time_utc) or not re.fullmatch(r"[0-9a-f]{8}",a.random):raise ValueError("invalid identity")
 c,sha=load(a.config);thresholds,_=load(a.thresholds);found=[s for o in c["operators"] if o["operator_name"]=="correlate2d" for s in o["scales"] if s["scale_id"]==a.scale]
 if len(found)!=1:raise ValueError("scale identity")
 s=found[0];inputs=s["inputs"];p=s["parameters"]
 if len(inputs)!=2 or [x["input_name"] for x in inputs] != ["in1","in2"] or len({x["dtype"] for x in inputs})!=1 or any(len(x["shape"])!=2 or math.prod(x["shape"])!=x["element_count"] for x in inputs) or s["actual_elements"]!=sum(x["element_count"] for x in inputs):raise ValueError("actual typed input contract")
 dtype=inputs[0]["dtype"];native=dtype in ("FP32","INT32");semantics="actual_same_dtype_arrays_native_modular_direct" if dtype=="INT32" else ("actual_same_dtype_arrays_native_direct" if native else "actual_same_dtype_arrays_fp32_legal_extension");compute="INT32_modular" if dtype=="INT32" else "FP32"
 expected={"dtype_semantics":semantics,"typed_input_names":"in1|in2","compute_dtype":compute,"fft_dtype":"not_applicable","output_dtype":dtype,"backend":"not_applicable"}
 if any(p.get(k)!=v for k,v in expected.items()) or p.get("mode") not in ("full","same","valid") or p.get("boundary") not in ("fill","wrap","symm") or not isinstance(p.get("fillvalue"),(int,float)):raise ValueError("compute contract")
 matches=[x for x in thresholds["entries"] if x["target"]=="correlate2d" and x["dtype"]==dtype]
 if len(matches)!=1:raise ValueError("threshold identity")
 th=matches[0];warm,measured=(20,100) if a.run_type=="formal" else (1,1);run=f"{a.test_time_utc}_{a.git_commit[:12]}_remaining_operator_case_b04_{a.run_type}_{a.random}";d=pathlib.Path(a.results_root)/a.run_type/run
 if d.exists():raise RuntimeError("run exists")
 d.mkdir(parents=True);rawfile=d/"correlate2d_probe_raw.json";log=d/"operator_correlate2d_full.log"
 i1,i2=inputs;cmd=[a.binary,"--dtype",dtype,"--rows1",str(i1["shape"][0]),"--cols1",str(i1["shape"][1]),"--rows2",str(i2["shape"][0]),"--cols2",str(i2["shape"][1]),"--mode",p["mode"],"--boundary",p["boundary"],"--fillvalue",str(p["fillvalue"]),"--seed",str(p["generator_seed"]),"--warmup",str(warm),"--measured",str(measured),"--output",str(rawfile)]
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write_text("command="+" ".join(cmd)+"\n"+r.stdout)
 if r.returncode or not rawfile.exists():raise RuntimeError(f"probe rc={r.returncode}")
 raw=json.loads(rawfile.read_text());shape={"same":i1["shape"],"full":[i1["shape"][0]+i2["shape"][0]-1,i1["shape"][1]+i2["shape"][1]-1],"valid":[i1["shape"][0]-i2["shape"][0]+1,i1["shape"][1]-i2["shape"][1]+1]}[p["mode"]]
 if (raw["requested_dtype"],raw["compute_dtype"],raw["output_dtype"],raw["backend"],raw["typed_input_count"],raw["mode"],raw["boundary"],raw["output_shape"])!=(dtype,compute,dtype,"not_applicable",2,p["mode"],p["boundary"],shape):raise RuntimeError("raw contract")
 metric_ok=all(raw[k]<=th[k+"_max"] for k in ("mse","rmse","relative_l2","relative_linf"));exact_ok=(not th["require_exact_match"]) or (raw["exact_match"] and raw["mismatch_count"]==0);accuracy_ok=metric_ok and exact_ok
 ch=summary(raw["cpu_trace"],"heap");cr=summary(raw["cpu_trace"],"rss");gh=summary(raw["gpu_trace"],"heap");gr=summary(raw["gpu_trace"],"rss");gm=summary(raw["gpu_trace"],"gpu");resource_ok=all(x is not None for x in (ch,cr,gh,gr,gm));passed=accuracy_ok and resource_ok;status="PASS" if passed else "FAIL";code="OK" if passed else ("ACCURACY_FAILED" if not accuracy_ok else "RESOURCE_FAILED")
 case=f"correlate2d__{s['scale_id']}__{raw['actual_input_digest'][:16]}";cs=common.stats(raw["cpu_samples_ms"]);gs=common.stats(raw["gpu_samples_ms"]);speed=cs["mean_ms"]/gs["mean_ms"]
 def mainrow(dev,st,h,rss,gpu):
  row=ident(a,c,sha,s,run,case,dev,status,code);row.update({k:f"{v:.17g}" for k,v in st.items()});row.update({"warmup_runs":str(warm),"measured_runs":str(measured),"cpu_gpu_speedup":f"{speed:.17g}" if dev=="GPU" else "NA","accuracy_reference":"cpu_independent_boundary_direct_reference","mse":f"{raw['mse']:.17g}" if dev=="GPU" else "0","rmse":f"{raw['rmse']:.17g}" if dev=="GPU" else "0","relative_l2":f"{raw['relative_l2']:.17g}" if dev=="GPU" else "0","relative_linf":f"{raw['relative_linf']:.17g}" if dev=="GPU" else "0","exact_match":str(raw["exact_match"]).lower() if dev=="GPU" else "true","mismatch_count":str(raw["mismatch_count"] if dev=="GPU" else 0),"semantic_check":th["semantic_rule"]+f":mode={p['mode']};boundary={p['boundary']};valid_boundary_effect={'ignored' if p['mode']=='valid' else 'active'}","accuracy_status":"PASS" if accuracy_ok else "FAIL"})
  for pre,val in (("cpu_heap",h),("rss",rss),("gpu",gpu)):
   for field in ("before","after","peak","delta"):row[f"{pre}_{field}_bytes"]=str(val[field]) if val else "NA"
  for k in common.SUMMARY:row.setdefault(k,"NA")
  return row
 common.write_csv(d/f"operator_main_results_correlate2d_not_applicable_{run}.csv",common.IDENTITY+common.SUMMARY,[mainrow("CPU",cs,ch,cr,None),mainrow("GPU",gs,gh,gr,gm)])
 timing_columns=common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"];timing=[]
 for dev,vals,od in (("CPU",raw["cpu_samples_ms"],raw["cpu_output_digest"]),("GPU",raw["gpu_samples_ms"],raw["gpu_output_digest"])):
  for i,v in enumerate(vals):q=ident(a,c,sha,s,run,case,dev,status,code);q.update({"warmup_runs":str(warm),"measured_runs":str(measured),"sample_index":str(i),"latency_ms":f"{v:.17g}","synchronized":"true","input_digest":raw["actual_input_digest"],"output_digest":od});timing.append(q)
 common.write_csv(d/f"operator_timing_samples_correlate2d_not_applicable_{run}.csv",timing_columns,timing)
 memory_columns=common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"];memory=[]
 for dev,trace in (("CPU",raw["cpu_trace"]),("GPU",raw["gpu_trace"])):
  for x in trace:q=ident(a,c,sha,s,run,case,dev,status,code);q.update({"sample_index":"0","trace_phase":x["phase"],"phase_index":str(x["phase_index"]),"cpu_live_heap_bytes":str(x["heap"]["bytes"]) if x["heap"]["available"] else "NA","rss_bytes":str(x["rss"]["bytes"]) if x["rss"]["available"] else "NA","gpu_used_bytes":str(x["gpu"]["bytes"]) if x["gpu"]["available"] else "NA"});memory.append(q)
 common.write_csv(d/"operator_convolution_correlate2d_memory_trace_not_applicable.csv",memory_columns,memory)
 common.write_csv(d/f"operator_dtype_evidence_correlate2d_not_applicable_{run}.csv",["run_id","case_id","operator_name","requested_dtype","dtype_semantics","typed_input_count","typed_input_manifest","actual_input_digest","compute_dtype","fft_dtype","output_dtype","native_capability","status","error_code"],[{"run_id":run,"case_id":case,"operator_name":"correlate2d","requested_dtype":dtype,"dtype_semantics":semantics,"typed_input_count":"2","typed_input_manifest":raw["typed_input_manifest"],"actual_input_digest":raw["actual_input_digest"],"compute_dtype":compute,"fft_dtype":"not_applicable","output_dtype":dtype,"native_capability":"native" if native else "approved_legal_extension","status":status,"error_code":code}])
 report={"schema_version":1,"run_id":run,"case_id":case,"operator_name":"correlate2d","scale_id":s["scale_id"],"requested_dtype":dtype,**expected,"native_capability":"native" if native else "approved_legal_extension","mode":p["mode"],"boundary":p["boundary"],"boundary_effect":"ignored_by_valid" if p["mode"]=="valid" else "active","typed_input_count":2,"warmup_runs":warm,"measured_runs":measured,"actual_input_digest":raw["actual_input_digest"],"metrics":{k:raw[k] for k in ("mse","rmse","relative_l2","relative_linf","exact_match","mismatch_count")},"status":status,"error_code":code};(d/"operator_correlate2d_report.json").write_text(json.dumps(report,indent=2)+"\n");(d/"operator_correlate2d_summary.txt").write_text("\n".join(f"{k}={v}" for k,v in report.items() if not isinstance(v,dict))+"\n");log.write_text(log.read_text()+json.dumps(report,sort_keys=True)+"\n")
 print(f"REMAINING_OPERATOR_CASE_CORRELATE2D_CASE {status} scale={s['scale_id']} dtype={dtype} backend=not_applicable measured={measured} results={d}");return 0 if passed else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_CORRELATE2D_DRIVER FAIL error={e}",file=sys.stderr);sys.exit(2)
