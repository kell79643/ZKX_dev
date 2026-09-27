#!/usr/bin/env python3
import argparse,csv,hashlib,json,pathlib,re,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));import run_operator as common
def arguments():
 p=argparse.ArgumentParser();[p.add_argument("--"+x,required=True) for x in ("binary","config","thresholds","results-root","scale","run-type","git-commit","git-dirty","test-time-utc","random")];return p.parse_args()
def load(path):
 b=pathlib.Path(path).read_bytes();return json.loads(b.decode()),hashlib.sha256(b).hexdigest()
def identity(a,c,h,s,run,case,device,status,error):
 shape="x=["+"x".join(map(str,s["inputs"][0]["shape"]))+"]";return dict(zip(common.IDENTITY,["1",run,a.run_type,a.test_time_utc,a.git_commit,a.git_dirty,c["scale_set_id"],h,"operator","freq_shift","NA","NA","freq_shift",case,s["scale_id"],s["order_of_magnitude"],str(s["actual_elements"]),shape,s["inputs"][0]["dtype"],device,"not_applicable","compatibility/end-to-end",status,error]))
def main():
 a=arguments();c,h=load(a.config);thresholds,_=load(a.thresholds);s=[x for op in c["operators"] for x in op["scales"] if x["scale_id"]==a.scale][0];inp=s["inputs"][0];p=s["parameters"];dtype=inp["dtype"];th=[x for x in thresholds["entries"] if x["dtype"]==dtype][0];warmup,measured=(20,100) if a.run_type=="formal" else (1,1);run=f"{a.test_time_utc}_{a.git_commit[:12]}_remaining_operator_case_b05_{a.run_type}_{a.random}";directory=pathlib.Path(a.results_root)/a.run_type/run
 if directory.exists():raise RuntimeError("exists")
 directory.mkdir(parents=True);raw_path=directory/"freq_shift_probe_raw.json";log=directory/"operator_freq_shift_full.log";cmd=[a.binary,"--dtype",dtype,"--count",str(s["actual_elements"]),"--freq",str(p["freq"]),"--fs",str(p["fs"]),"--seed",str(p["seed"]),"--warmup",str(warmup),"--measured",str(measured),"--output",str(raw_path)];proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write_text("command="+" ".join(cmd)+"\n"+proc.stdout)
 if proc.returncode:raise RuntimeError("probe")
 raw=json.loads(raw_path.read_text());metrics=("mse","rmse","relative_l2","relative_linf");accuracy_ok=all(raw[x]<=th[x+"_max"] for x in metrics);summaries=[common.resource_summary(raw[x],y) for x,y in (("cpu_trace","heap"),("cpu_trace","rss"),("gpu_trace","heap"),("gpu_trace","rss"),("gpu_trace","gpu"))];passed=accuracy_ok and all(x is not None for x in summaries);status="PASS" if passed else "FAIL";error="OK" if passed else "FAILED";case=f"freq_shift__{s['scale_id']}__{raw['actual_input_digest'][:16]}";cpu_stats,gpu_stats=common.stats(raw["cpu_samples_ms"]),common.stats(raw["gpu_samples_ms"]);speedup=cpu_stats["mean_ms"]/gpu_stats["mean_ms"]
 def main_row(device,stats,heap,rss,gpu):
  row=identity(a,c,h,s,run,case,device,status,error);row.update({k:f"{v:.17g}" for k,v in stats.items()});row.update({"warmup_runs":str(warmup),"measured_runs":str(measured),"cpu_gpu_speedup":f"{speedup:.17g}" if device=="GPU" else "NA","accuracy_reference":"cpu_freq_shift_complex128_reference","mse":str(raw["mse"] if device=="GPU" else 0),"rmse":str(raw["rmse"] if device=="GPU" else 0),"relative_l2":str(raw["relative_l2"] if device=="GPU" else 0),"relative_linf":str(raw["relative_linf"] if device=="GPU" else 0),"exact_match":"NA","mismatch_count":"NA","semantic_check":th["semantic_rule"],"accuracy_status":"PASS" if accuracy_ok else "FAIL"})
  for prefix,value in (("cpu_heap",heap),("rss",rss),("gpu",gpu)):
   for field in ("before","after","peak","delta"):row[f"{prefix}_{field}_bytes"]=str(value[field]) if value else "NA"
  for key in common.SUMMARY:row.setdefault(key,"NA")
  return row
 common.write_csv(directory/f"operator_main_results_freq_shift_not_applicable_{run}.csv",common.IDENTITY+common.SUMMARY,[main_row("CPU",cpu_stats,summaries[0],summaries[1],None),main_row("GPU",gpu_stats,summaries[2],summaries[3],summaries[4])])
 timing_columns=common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"];timing=[]
 for device,samples,digest in (("CPU",raw["cpu_samples_ms"],raw["cpu_output_digest"]),("GPU",raw["gpu_samples_ms"],raw["gpu_output_digest"])):
  for index,value in enumerate(samples):row=identity(a,c,h,s,run,case,device,status,error);row.update({"warmup_runs":str(warmup),"measured_runs":str(measured),"sample_index":str(index),"latency_ms":str(value),"synchronized":"true","input_digest":raw["actual_input_digest"],"output_digest":digest});timing.append(row)
 common.write_csv(directory/f"operator_timing_samples_freq_shift_not_applicable_{run}.csv",timing_columns,timing)
 memory_columns=common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"];memory=[]
 for device,trace in (("CPU",raw["cpu_trace"]),("GPU",raw["gpu_trace"])):
  for sample in trace:row=identity(a,c,h,s,run,case,device,status,error);row.update({"sample_index":"0","trace_phase":sample["phase"],"phase_index":str(sample["phase_index"]),"cpu_live_heap_bytes":str(sample["heap"]["bytes"]) if sample["heap"]["available"] else "NA","rss_bytes":str(sample["rss"]["bytes"]) if sample["rss"]["available"] else "NA","gpu_used_bytes":str(sample["gpu"]["bytes"]) if sample["gpu"]["available"] else "NA"});memory.append(row)
 common.write_csv(directory/"operator_filtering_freq_shift_memory_trace_not_applicable.csv",memory_columns,memory)
 evidence={"run_id":run,"case_id":case,"operator_name":"freq_shift","requested_dtype":dtype,"dtype_semantics":"actual_typed_array_input_complex_fp32_compute_host_complex_fp64_extension","typed_input_count":"1","typed_input_manifest":raw["typed_input_manifest"],"actual_input_digest":raw["actual_input_digest"],"compute_dtype":"ComplexFP32","fft_dtype":"not_applicable","output_dtype":"ComplexFP64","status":status,"error_code":error};common.write_csv(directory/f"operator_dtype_evidence_freq_shift_not_applicable_{run}.csv",list(evidence),[evidence])
 report={"run_id":run,"case_id":case,"operator_name":"freq_shift","scale_id":s["scale_id"],"requested_dtype":dtype,"output_dtype":"ComplexFP64","backend":"not_applicable","freq":p["freq"],"fs":p["fs"],"rank":len(inp["shape"]),"shape":inp["shape"],"warmup_runs":warmup,"measured_runs":measured,**{x:raw[x] for x in metrics},"status":status,"error_code":error};(directory/"operator_freq_shift_report.json").write_text(json.dumps(report,indent=2)+"\n");(directory/"operator_freq_shift_summary.txt").write_text("\n".join(f"{k}={v}" for k,v in report.items())+"\n");log.write_text(log.read_text()+json.dumps(report)+"\n");print(f"REMAINING_OPERATOR_CASE_FREQ_SHIFT_CASE {status} scale={s['scale_id']} dtype={dtype} measured={measured} results={directory}");return 0 if passed else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as exc:print(exc,file=sys.stderr);sys.exit(2)
