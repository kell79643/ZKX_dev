#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,pathlib,re,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"))
import run_operator as common

def args():
 p=argparse.ArgumentParser();p.add_argument("--binary",required=True);p.add_argument("--backend",choices=("fft_thrust","dlfft"),required=True);p.add_argument("--config",required=True);p.add_argument("--thresholds",required=True);p.add_argument("--results-root",required=True);p.add_argument("--scale",required=True);p.add_argument("--run-type",choices=("smoke","formal"),required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",choices=("true","false"),required=True);p.add_argument("--test-time-utc",required=True);p.add_argument("--random",required=True);return p.parse_args()
def load(path):
 raw=pathlib.Path(path).read_bytes();return json.loads(raw.decode()),hashlib.sha256(raw).hexdigest()
def ident(a,c,sha,s,run,case,device,status,code):
 shape="x=["+"x".join(map(str,s["inputs"][0]["shape"]))+"]"
 return dict(zip(common.IDENTITY,["1",run,a.run_type,a.test_time_utc,a.git_commit,a.git_dirty,c["scale_set_id"],sha,"operator","hilbert","NA","NA","hilbert",case,s["scale_id"],s["order_of_magnitude"],str(s["actual_elements"]),shape,s["inputs"][0]["dtype"],device,a.backend,"compatibility/end-to-end",status,code]))
def summary(trace,key): return common.resource_summary(trace,key)
def main():
 a=args()
 if not re.fullmatch(r"[0-9a-f]{40}",a.git_commit) or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z",a.test_time_utc) or not re.fullmatch(r"[0-9a-f]{8}",a.random):raise ValueError("invalid identity")
 c,sha=load(a.config);t,_=load(a.thresholds);ss=[s for o in c["operators"] if o["operator_name"]=="hilbert" for s in o["scales"] if s["scale_id"]==a.scale]
 if len(ss)!=1:raise ValueError("scale identity")
 s=ss[0];inp=s["inputs"]
 if len(inp)!=1 or inp[0]["input_name"]!="x" or math.prod(inp[0]["shape"])!=inp[0]["element_count"] or s["actual_elements"]!=inp[0]["element_count"]:raise ValueError("typed input contract")
 dtype=inp[0]["dtype"];p=s["parameters"];expected_out="ComplexFP64" if dtype.startswith("INT") else "ComplexFP32"
 expected={"dtype_semantics":"actual_typed_array_input_fp32_fft_extension","typed_input_names":"x","compute_dtype":"FP32","fft_dtype":"ComplexFP32","output_dtype":expected_out,"backend":"fft_thrust"}
 if any(p.get(k)!=v for k,v in expected.items()) or p.get("shape")!=inp[0]["shape"] or p.get("axis") not in (0,1):raise ValueError("hilbert compute contract")
 th=[x for x in t["entries"] if x["target"]=="hilbert" and x["dtype"]==dtype]
 if len(th)!=1:raise ValueError("threshold")
 th=th[0];warm,measured=(20,100) if a.run_type=="formal" else (1,1)
 run=f"{a.test_time_utc}_{a.git_commit[:12]}_remaining_operator_case_b03_{a.run_type}_{a.random}";d=pathlib.Path(a.results_root)/a.run_type/run
 if d.exists():raise RuntimeError("run exists")
 d.mkdir(parents=True);rawfile=d/"hilbert_probe_raw.json";log=d/"operator_hilbert_full.log"
 cmd=[a.binary,"--dtype",dtype,"--rows",str(inp[0]["shape"][0]),"--cols",str(inp[0]["shape"][1]),"--axis",str(p["axis"]),"--seed",str(p["generator_seed"]),"--warmup",str(warm),"--measured",str(measured),"--output",str(rawfile)]
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write_text("command="+" ".join(cmd)+"\n"+r.stdout)
 if r.returncode or not rawfile.exists():raise RuntimeError(f"probe rc={r.returncode}")
 raw=json.loads(rawfile.read_text())
 if (raw["requested_dtype"],raw["output_dtype"],raw["typed_input_count"],raw["backend"])!=(dtype,expected_out,1,a.backend) or raw["output_shape"]!=inp[0]["shape"]:raise RuntimeError("raw contract")
 ap=all(raw[k]<=th[k+"_max"] for k in ("mse","rmse","relative_l2","relative_linf"));ch=summary(raw["cpu_trace"],"heap");cr=summary(raw["cpu_trace"],"rss");gh=summary(raw["gpu_trace"],"heap");gr=summary(raw["gpu_trace"],"rss");gm=summary(raw["gpu_trace"],"gpu");rp=all(x is not None for x in (ch,cr,gh,gr,gm));passed=ap and rp;status="PASS" if passed else "FAIL";code="OK" if passed else ("ACCURACY_FAILED" if not ap else "RESOURCE_FAILED")
 case=f"hilbert__{s['scale_id']}__{raw['actual_input_digest'][:16]}";cs=common.stats(raw["cpu_samples_ms"]);gs=common.stats(raw["gpu_samples_ms"]);speed=cs["mean_ms"]/gs["mean_ms"]
 def mainrow(dev,st,h,rss,gpu):
  row=ident(a,c,sha,s,run,case,dev,status,code);row.update({k:f"{v:.17g}" for k,v in st.items()});row.update({"warmup_runs":str(warm),"measured_runs":str(measured),"cpu_gpu_speedup":f"{speed:.17g}" if dev=="GPU" else "NA","accuracy_reference":"cpu_typed_double_direct_dft","mse":f"{raw['mse']:.17g}" if dev=="GPU" else "0","rmse":f"{raw['rmse']:.17g}" if dev=="GPU" else "0","relative_l2":f"{raw['relative_l2']:.17g}" if dev=="GPU" else "0","relative_linf":f"{raw['relative_linf']:.17g}" if dev=="GPU" else "0","exact_match":"NA","mismatch_count":"NA","semantic_check":th["semantic_rule"]+f":shape={inp[0]['shape']};finite=true","accuracy_status":"PASS" if ap else "FAIL"})
  for pre,val in (("cpu_heap",h),("rss",rss),("gpu",gpu)):
   for field in ("before","after","peak","delta"):row[f"{pre}_{field}_bytes"]=str(val[field]) if val else "NA"
  for k in common.SUMMARY:row.setdefault(k,"NA")
  return row
 common.write_csv(d/f"operator_main_results_hilbert_{a.backend}_{run}.csv",common.IDENTITY+common.SUMMARY,[mainrow("CPU",cs,ch,cr,None),mainrow("GPU",gs,gh,gr,gm)])
 tc=common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"];trs=[]
 for dev,vals,od in (("CPU",raw["cpu_samples_ms"],raw["cpu_output_digest"]),("GPU",raw["gpu_samples_ms"],raw["gpu_output_digest"])):
  for i,v in enumerate(vals):q=ident(a,c,sha,s,run,case,dev,status,code);q.update({"warmup_runs":str(warm),"measured_runs":str(measured),"sample_index":str(i),"latency_ms":f"{v:.17g}","synchronized":"true","input_digest":raw["actual_input_digest"],"output_digest":od});trs.append(q)
 common.write_csv(d/f"operator_timing_samples_hilbert_{a.backend}_{run}.csv",tc,trs)
 mc=common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"];mrs=[]
 for dev,tr in (("CPU",raw["cpu_trace"]),("GPU",raw["gpu_trace"])):
  for x in tr:q=ident(a,c,sha,s,run,case,dev,status,code);q.update({"sample_index":"0","trace_phase":x["phase"],"phase_index":str(x["phase_index"]),"cpu_live_heap_bytes":str(x["heap"]["bytes"]) if x["heap"]["available"] else "NA","rss_bytes":str(x["rss"]["bytes"]) if x["rss"]["available"] else "NA","gpu_used_bytes":str(x["gpu"]["bytes"]) if x["gpu"]["available"] else "NA"});mrs.append(q)
 common.write_csv(d/f"operator_filtering_hilbert_memory_trace_{a.backend}.csv",mc,mrs)
 common.write_csv(d/f"operator_dtype_evidence_hilbert_fft_thrust_{run}.csv",["run_id","case_id","operator_name","requested_dtype","dtype_semantics","typed_input_count","typed_input_manifest","actual_input_digest","compute_dtype","fft_dtype","output_dtype","status","error_code"],[{"run_id":run,"case_id":case,"operator_name":"hilbert","requested_dtype":dtype,"dtype_semantics":expected["dtype_semantics"],"typed_input_count":"1","typed_input_manifest":raw["typed_input_manifest"],"actual_input_digest":raw["actual_input_digest"],"compute_dtype":"FP32","fft_dtype":"ComplexFP32","output_dtype":expected_out,"status":status,"error_code":code}])
 report={"schema_version":1,"run_id":run,"case_id":case,"operator_name":"hilbert","scale_id":s["scale_id"],"requested_dtype":dtype,**expected,"typed_input_count":1,"warmup_runs":warm,"measured_runs":measured,"actual_input_digest":raw["actual_input_digest"],"metrics":{k:raw[k] for k in ("mse","rmse","relative_l2","relative_linf")},"status":status,"error_code":code};(d/"operator_hilbert_report.json").write_text(json.dumps(report,indent=2)+"\n");(d/"operator_hilbert_summary.txt").write_text("\n".join(f"{k}={v}" for k,v in report.items() if not isinstance(v,dict))+"\n");log.write_text(log.read_text()+json.dumps(report,sort_keys=True)+"\n")
 print(f"REMAINING_OPERATOR_CASE_HILBERT_CASE {status} scale={s['scale_id']} dtype={dtype} backend={a.backend} measured={measured} results={d}");return 0 if passed else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_HILBERT_DRIVER FAIL error={e}",file=sys.stderr);sys.exit(2)
