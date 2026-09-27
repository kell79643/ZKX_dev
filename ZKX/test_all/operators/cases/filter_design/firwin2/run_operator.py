#!/usr/bin/env python3
import argparse,csv,hashlib,json,pathlib,re,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"filtering"/"channelize_poly"))
import run_operator as common
def arguments():
 p=argparse.ArgumentParser();p.add_argument("--binary",required=True);p.add_argument("--backend",choices=("fft_thrust","dlfft"),required=True);p.add_argument("--config",required=True);p.add_argument("--thresholds",required=True);p.add_argument("--results-root",required=True);p.add_argument("--scale",required=True);p.add_argument("--run-type",choices=("smoke","formal"),required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",choices=("true","false"),required=True);p.add_argument("--test-time-utc",required=True);p.add_argument("--random",required=True);return p.parse_args()
def load(path):raw=pathlib.Path(path).read_bytes();return json.loads(raw.decode()),hashlib.sha256(raw).hexdigest()
def ident(a,c,sha,s,run,case,device,status,code):
 shape=";".join(x["input_name"]+"=["+"x".join(map(str,x["shape"]))+"]" for x in s["inputs"])
 return dict(zip(common.IDENTITY,["1",run,a.run_type,a.test_time_utc,a.git_commit,a.git_dirty,c["scale_set_id"],sha,"operator","firwin2","NA","NA","firwin2",case,s["scale_id"],s["order_of_magnitude"],str(s["actual_elements"]),shape,s["inputs"][0]["dtype"],device,a.backend,"compatibility/end-to-end",status,code]))
def main():
 a=arguments()
 if not re.fullmatch(r"[0-9a-f]{40}",a.git_commit) or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z",a.test_time_utc) or not re.fullmatch(r"[0-9a-f]{8}",a.random):raise ValueError("invalid identity")
 c,sha=load(a.config);thresholds,_=load(a.thresholds);found=[s for o in c["operators"] if o["operator_name"]=="firwin2" for s in o["scales"] if s["scale_id"]==a.scale]
 if len(found)!=1:raise ValueError("scale identity")
 s=found[0];inputs=s["inputs"];p=s["parameters"]
 if len(inputs)!=2 or [x["input_name"] for x in inputs] != ["freq","gain"] or len({x["dtype"] for x in inputs})!=1 or any(x["shape"]!=[3] or x["element_count"]!=3 for x in inputs):raise ValueError("actual typed inputs")
 dtype=inputs[0]["dtype"];expected={"dtype_semantics":"actual_same_dtype_freq_gain_arrays","typed_input_names":"freq|gain","compute_dtype":"FP32","fft_dtype":"ComplexFP32","output_dtype":"FP64","backend":"fft_thrust"}
 if any(p.get(k)!=v for k,v in expected.items()) or p.get("work_formula")!="numtaps*nfreqs" or s["actual_elements"]!=p["numtaps"]*p["nfreqs"] or p["nfreqs"]<=p["numtaps"] or p["window"] not in ("hamming","none","explicit") or p["filter_type"] not in ("I","II","III"):raise ValueError("scientific compute contract")
 matches=[x for x in thresholds["entries"] if x["target"]=="firwin2" and x["dtype"]==dtype]
 if len(matches)!=1:raise ValueError("threshold identity")
 th=matches[0];warm,measured=(20,100) if a.run_type=="formal" else (1,1);run=f"{a.test_time_utc}_{a.git_commit[:12]}_remaining_operator_case_b04_{a.run_type}_{a.random}";d=pathlib.Path(a.results_root)/a.run_type/run
 if d.exists():raise RuntimeError("run exists")
 d.mkdir(parents=True);rawfile=d/"firwin2_probe_raw.json";log=d/"operator_firwin2_full.log";cmd=[a.binary,"--dtype",dtype,"--numtaps",str(p["numtaps"]),"--nfreqs",str(p["nfreqs"]),"--window",p["window"],"--antisymmetric",str(p["antisymmetric"]).lower(),"--fs",str(p["fs"]),"--warmup",str(warm),"--measured",str(measured),"--output",str(rawfile)]
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write_text("command="+" ".join(cmd)+"\n"+r.stdout)
 if r.returncode or not rawfile.exists():raise RuntimeError(f"probe rc={r.returncode}")
 raw=json.loads(rawfile.read_text())
 if (raw["requested_dtype"],raw["compute_dtype"],raw["fft_dtype"],raw["output_dtype"],raw["backend"],raw["typed_input_count"],raw["numtaps"],raw["nfreqs"],raw["window"],raw["antisymmetric"],raw["output_shape"])!=(dtype,"FP32","ComplexFP32","FP64",a.backend,2,p["numtaps"],p["nfreqs"],p["window"],p["antisymmetric"],[p["numtaps"]]):raise RuntimeError("raw contract")
 accuracy_ok=all(raw[k]<=th[k+"_max"] for k in ("mse","rmse","relative_l2","relative_linf"));summaries=[common.resource_summary(raw[t],k) for t,k in (("cpu_trace","heap"),("cpu_trace","rss"),("gpu_trace","heap"),("gpu_trace","rss"),("gpu_trace","gpu"))];resource_ok=all(x is not None for x in summaries);ch,cr,gh,gr,gm=summaries;passed=accuracy_ok and resource_ok;status="PASS" if passed else "FAIL";code="OK" if passed else ("ACCURACY_FAILED" if not accuracy_ok else "RESOURCE_FAILED")
 case=f"firwin2__{s['scale_id']}__{raw['actual_input_digest'][:16]}";cs=common.stats(raw["cpu_samples_ms"]);gs=common.stats(raw["gpu_samples_ms"]);speed=cs["mean_ms"]/gs["mean_ms"]
 def mainrow(dev,st,h,rss,gpu):
  row=ident(a,c,sha,s,run,case,dev,status,code);row.update({k:f"{v:.17g}" for k,v in st.items()});row.update({"warmup_runs":str(warm),"measured_runs":str(measured),"cpu_gpu_speedup":f"{speed:.17g}" if dev=="GPU" else "NA","accuracy_reference":"cpu_independent_firwin2_reference","mse":f"{raw['mse']:.17g}" if dev=="GPU" else "0","rmse":f"{raw['rmse']:.17g}" if dev=="GPU" else "0","relative_l2":f"{raw['relative_l2']:.17g}" if dev=="GPU" else "0","relative_linf":f"{raw['relative_linf']:.17g}" if dev=="GPU" else "0","exact_match":"NA","mismatch_count":"NA","semantic_check":th["semantic_rule"]+f":type={p['filter_type']};window={p['window']};numtaps={p['numtaps']};nfreqs={p['nfreqs']}","accuracy_status":"PASS" if accuracy_ok else "FAIL"})
  for pre,val in (("cpu_heap",h),("rss",rss),("gpu",gpu)):
   for field in ("before","after","peak","delta"):row[f"{pre}_{field}_bytes"]=str(val[field]) if val else "NA"
  for k in common.SUMMARY:row.setdefault(k,"NA")
  return row
 common.write_csv(d/f"operator_main_results_firwin2_{a.backend}_{run}.csv",common.IDENTITY+common.SUMMARY,[mainrow("CPU",cs,ch,cr,None),mainrow("GPU",gs,gh,gr,gm)])
 timing_columns=common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"];timing=[]
 for dev,vals,od in (("CPU",raw["cpu_samples_ms"],raw["cpu_output_digest"]),("GPU",raw["gpu_samples_ms"],raw["gpu_output_digest"])):
  for i,v in enumerate(vals):q=ident(a,c,sha,s,run,case,dev,status,code);q.update({"warmup_runs":str(warm),"measured_runs":str(measured),"sample_index":str(i),"latency_ms":f"{v:.17g}","synchronized":"true","input_digest":raw["actual_input_digest"],"output_digest":od});timing.append(q)
 common.write_csv(d/f"operator_timing_samples_firwin2_{a.backend}_{run}.csv",timing_columns,timing)
 memory_columns=common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"];memory=[]
 for dev,trace in (("CPU",raw["cpu_trace"]),("GPU",raw["gpu_trace"])):
  for x in trace:q=ident(a,c,sha,s,run,case,dev,status,code);q.update({"sample_index":"0","trace_phase":x["phase"],"phase_index":str(x["phase_index"]),"cpu_live_heap_bytes":str(x["heap"]["bytes"]) if x["heap"]["available"] else "NA","rss_bytes":str(x["rss"]["bytes"]) if x["rss"]["available"] else "NA","gpu_used_bytes":str(x["gpu"]["bytes"]) if x["gpu"]["available"] else "NA"});memory.append(q)
 common.write_csv(d/f"operator_filter_design_firwin2_memory_trace_{a.backend}.csv",memory_columns,memory)
 common.write_csv(d/f"operator_dtype_evidence_firwin2_{a.backend}_{run}.csv",["run_id","case_id","operator_name","requested_dtype","dtype_semantics","typed_input_count","typed_input_manifest","actual_input_digest","compute_dtype","fft_dtype","output_dtype","backend","status","error_code"],[{"run_id":run,"case_id":case,"operator_name":"firwin2","requested_dtype":dtype,"dtype_semantics":expected["dtype_semantics"],"typed_input_count":"2","typed_input_manifest":raw["typed_input_manifest"],"actual_input_digest":raw["actual_input_digest"],"compute_dtype":"FP32","fft_dtype":"ComplexFP32","output_dtype":"FP64","backend":a.backend,"status":status,"error_code":code}])
 report={"schema_version":1,"run_id":run,"case_id":case,"operator_name":"firwin2","scale_id":s["scale_id"],"requested_dtype":dtype,**expected,"backend":a.backend,"filter_type":p["filter_type"],"window":p["window"],"antisymmetric":p["antisymmetric"],"numtaps":p["numtaps"],"nfreqs":p["nfreqs"],"work_items":s["actual_elements"],"typed_input_count":2,"warmup_runs":warm,"measured_runs":measured,"actual_input_digest":raw["actual_input_digest"],"metrics":{k:raw[k] for k in ("mse","rmse","relative_l2","relative_linf")},"status":status,"error_code":code};(d/"operator_firwin2_report.json").write_text(json.dumps(report,indent=2)+"\n");(d/"operator_firwin2_summary.txt").write_text("\n".join(f"{k}={v}" for k,v in report.items() if not isinstance(v,dict))+"\n");log.write_text(log.read_text()+json.dumps(report,sort_keys=True)+"\n")
 print(f"REMAINING_OPERATOR_CASE_FIRWIN2_CASE {status} scale={s['scale_id']} dtype={dtype} backend={a.backend} measured={measured} results={d}");return 0 if passed else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FIRWIN2_DRIVER FAIL error={e}",file=sys.stderr);sys.exit(2)
