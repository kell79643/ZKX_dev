#!/usr/bin/env python3
import argparse,csv,hashlib,json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"cases"/"filtering"/"channelize_poly"));import run_operator as common
OPS={"pulse_compression":"fft_thrust","pulse_doppler":"fft_thrust","ambgfun":"fft_thrust","ca_cfar":"not_applicable","cfar_alpha":"not_applicable"}
def preview_indices(count):return list(range(count))if count<=6 else[0,1,2,3,count-2,count-1]
def complex_preview(count,seed,integral):
 def value(i):
  a=(i*13+seed)%11-5;b=(i*7+seed)%9-4
  return f"({a},{b})"if integral else f"({a*.25:g},{b*.25:g})"
 return";".join(f"i={i}:{value(i)}"for i in preview_indices(count))
def real_preview(count,seed,integral):
 def value(i):
  z=(i*17+seed)%19-9
  return f"{z}"if integral else f"{z*.125+(i%7)*.03125:g}"
 return";".join(f"i={i}:{value(i)}"for i in preview_indices(count))
def input_content(operator,scale,dtype):
 integral=dtype.startswith("INT");seeds={"pulse_compression":{"echo":3,"waveform":7},"pulse_doppler":{"compressed":11},"ambgfun":{"waveform":19},"ca_cfar":{"power":23}};rows=[]
 for item in scale["inputs"]:
  name=item["input_name"];count=item["element_count"];shape="["+"x".join(str(x)for x in item["shape"])+"]"
  if operator=="cfar_alpha":generator="exact_scalar_from_case_config";preview=f"value={scale['parameters']['pfa']:g}"
  elif operator=="ca_cfar":seed=seeds[operator][name];generator=f"real_modular(seed={seed},integral={str(integral).lower()})";preview=real_preview(count,seed,integral)
  else:seed=seeds[operator][name];generator=f"complex_modular(seed={seed},integral={str(integral).lower()})";preview=complex_preview(count,seed,integral)
  rows.append({"input_name":name,"shape":shape,"dtype":item["dtype"],"element_count":str(count),"generator":generator,"preview_heacuda_api_tail2":preview})
 return rows
def table(title,columns,rows):
 values=[[str(row.get(key,"NA"))for key,_ in columns]for row in rows];widths=[max([len(label)]+[len(row[i])for row in values])for i,(_,label)in enumerate(columns)];line="+-"+"-+-".join("-"*width for width in widths)+"-+";body=[title,line,"| "+" | ".join(label.ljust(widths[i])for i,(_,label)in enumerate(columns))+" |",line];body.extend("| "+" | ".join(row[i].ljust(widths[i])for i in range(len(columns)))+" |"for row in values);body.append(line);return"\n".join(body)
def terminal_summary(main_path,paths):
 with main_path.open(encoding="utf-8",newline="")as handle:main_rows=list(csv.DictReader(handle))
 path_map=dict(paths);timing_path=path_map["timing_csv"];dtype_path=path_map["dtype_csv"]
 with timing_path.open(encoding="utf-8",newline="")as handle:timing_rows=list(csv.DictReader(handle))
 with dtype_path.open(encoding="utf-8",newline="")as handle:dtype_rows=list(csv.DictReader(handle))
 blocks=[table("[stage07] identity and status",[("target","target"),("scale_id","scale_id"),("device","device"),("backend","backend"),("dtype","dtype"),("warmup_runs","warmup"),("measured_runs","measured"),("timing_scope","timing_scope"),("status","status"),("error_code","error_code")],main_rows)]
 blocks.append(table("[stage07] timing detail",[("device","device"),("mean_ms","mean_ms"),("p50_ms","p50_ms"),("p95_ms","p95_ms"),("p99_ms","p99_ms"),("min_ms","min_ms"),("max_ms","max_ms"),("std_ms","std_ms"),("cv","cv"),("cpu_gpu_speedup","speedup")],main_rows))
 gpu=next(row for row in main_rows if row["device"]=="GPU");blocks.append(table("[stage07] accuracy detail",[("accuracy_reference","reference"),("mse","mse"),("rmse","rmse"),("relative_l2","relative_l2"),("relative_linf","relative_linf"),("exact_match","exact_match"),("mismatch_count","mismatch_count"),("semantic_check","semantic_check"),("accuracy_status","status")],[gpu]))
 resources=[]
 for row in main_rows:
  for prefix,label in (("cpu_heap","cpu_heap"),("rss","rss"),("gpu","gpu_memory")):
   if row[f"{prefix}_before_bytes"]!="NA":resources.append({"device":row["device"],"metric":label,"before":row[f"{prefix}_before_bytes"],"after":row[f"{prefix}_after_bytes"],"peak":row[f"{prefix}_peak_bytes"],"delta":row[f"{prefix}_delta_bytes"]})
 blocks.append(table("[stage07] resource detail",[("device","device"),("metric","metric"),("before","before_bytes"),("after","after_bytes"),("peak","peak_bytes"),("delta","delta_bytes")],resources))
 io=[]
 for device in ("CPU","GPU"):
  row=next(item for item in timing_rows if item["device"]==device);io.append({"device":device,"input_digest":row["input_digest"],"output_digest":row["output_digest"]})
 content=json.loads(dtype_rows[0]["input_content_json"]);parameters=[{"parameters":dtype_rows[0]["case_parameters_json"]}];blocks.append(table("[stage07] input content",[("input_name","input_name"),("shape","shape"),("dtype","dtype"),("element_count","elements"),("generator","generator"),("preview_heacuda_api_tail2","values(head4/tail2)")],content));blocks.append(table("[stage07] case parameters",[("parameters","parameters")],parameters));blocks.append(table("[stage07] input and output identity",[("device","device"),("input_digest","input_digest"),("output_digest","output_digest")],io));blocks.append(table("[stage07] dtype evidence",[("requested_dtype","requested_dtype"),("config_input_dtype","config_input_dtype"),("output_dtype","output_dtype"),("actual_input_digest","actual_input_digest")],dtype_rows));blocks.append(table("[stage07] evidence files",[("kind","kind"),("path","path")],[{"kind":kind,"path":str(path)}for kind,path in paths]+[{"kind":"result_dir","path":str(main_path.parent)}]));return"\n\n".join(blocks)
def main():
 p=argparse.ArgumentParser();[p.add_argument("--"+x,required=True)for x in("binary","operator","backend","config","thresholds","results-root","scale","run-type","git-commit","git-dirty","test-time-utc","random")];a=p.parse_args();cb=pathlib.Path(a.config).read_bytes();cfg=json.loads(cb);ch=hashlib.sha256(cb).hexdigest();th=json.loads(pathlib.Path(a.thresholds).read_text());op=[o for o in cfg["operators"]if o["operator_name"]==a.operator][0];s=[x for x in op["scales"]if x["scale_id"]==a.scale][0];q=s["parameters"];raw_dtype=s["inputs"][0]["dtype"];dtype=raw_dtype.replace("Complex","");backend=a.backend if OPS[a.operator]=="fft_thrust" else "not_applicable";threshold=[x for x in th["entries"]if x["target"]==a.operator and x["dtype"]==dtype][0];warm,measured=(20,100)if a.run_type=="formal"else(1,1);run=f"{a.test_time_utc}_{a.git_commit[:12]}_operator_case_b01_{a.run_type}_{a.random}";out=pathlib.Path(a.results_root)/a.run_type/run
 if out.exists():raise RuntimeError("result exists")
 out.mkdir(parents=True);raw=out/f"{a.operator}_probe_raw.json";cmd=[a.binary,"--dtype",dtype,"--warmup",str(warm),"--measured",str(measured),"--output",str(raw)]
 if a.operator in ("pulse_compression","pulse_doppler","ca_cfar"):cmd += ["--n0",str(s["inputs"][0]["shape"][0]),"--n1",str(s["inputs"][0]["shape"][1])]
 if a.operator=="pulse_compression":cmd += ["--template",str(s["inputs"][1]["element_count"]),"--nfft",str(q["nfft"])]
 if a.operator=="pulse_doppler":cmd += ["--nfft",str(q["nfft"])]
 if a.operator=="ambgfun":cmd += ["--n0",str(s["inputs"][0]["element_count"]),"--fs",str(q["fs"]),"--prf",str(q["prf"])]
 if a.operator=="ca_cfar":cmd += ["--pfa",str(q["pfa"])]
 if a.operator=="cfar_alpha":cmd += ["--pfa",str(q["pfa"]),"--reference",str(q["reference_count"])]
 z=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log=out/f"operator_{a.operator}_full.log";log.write_text("command="+" ".join(cmd)+"\n"+z.stdout)
 if z.returncode:raise RuntimeError("probe failed: "+z.stdout)
 r=json.loads(raw.read_text());metrics=("mse","rmse","relative_l2","relative_linf");floating_ok=all(r[x]<=threshold[x+"_max"] for x in metrics);discrete_ok=a.operator!="ca_cfar" or (r["exact_match"] and r["mismatch_count"]==0 and r["semantic_check"]);resources=[common.resource_summary(r[x],y)for x,y in (("cpu_trace","heap"),("cpu_trace","rss"),("gpu_trace","heap"),("gpu_trace","rss"),("gpu_trace","gpu"))];status="PASS"if floating_ok and discrete_ok and all(x is not None for x in resources)else"FAIL";error="OK"if status=="PASS"else"RESOURCE_FAILED"if not all(x is not None for x in resources)else"ACCURACY_FAILED";case=f"{a.operator}__{s['scale_id']}__{r['actual_input_digest'][:16]}"
 def identity(device):return dict(zip(common.IDENTITY,["1",run,a.run_type,a.test_time_utc,a.git_commit,a.git_dirty,cfg["scale_set_id"],ch,"operator",a.operator,"Task1",{"pulse_compression":"Step2","pulse_doppler":"Step3","ca_cfar":"Step4","cfar_alpha":"Step4","ambgfun":"Step5"}[a.operator],a.operator,case,s["scale_id"],s["order_of_magnitude"],str(s["actual_elements"]),"|".join(i["input_name"]+"="+str(i["shape"]) for i in s["inputs"]),dtype,device,backend,"compatibility/end-to-end",status,error]))
 cs,gs=common.stats(r["cpu_samples_ms"]),common.stats(r["gpu_samples_ms"]);main=[]
 for device,stats,heap,rss,gpu in (("CPU",cs,resources[0],resources[1],None),("GPU",gs,resources[2],resources[3],resources[4])):
  row=identity(device);row.update({k:str(v)for k,v in stats.items()});row.update({"warmup_runs":str(warm),"measured_runs":str(measured),"cpu_gpu_speedup":str(cs["mean_ms"]/gs["mean_ms"])if device=="GPU"else"NA","accuracy_reference":"typed_cpu_reference","mse":str(r["mse"]if device=="GPU"else 0),"rmse":str(r["rmse"]if device=="GPU"else 0),"relative_l2":str(r["relative_l2"]if device=="GPU"else 0),"relative_linf":str(r["relative_linf"]if device=="GPU"else 0),"exact_match":str(r["exact_match"]).lower()if a.operator=="ca_cfar"else"NA","mismatch_count":str(r["mismatch_count"])if a.operator=="ca_cfar"else"NA","semantic_check":("detection_binary_and_shape="+str(r["semantic_check"]).lower())if a.operator=="ca_cfar"else"typed_output_shape_dtype_finite","accuracy_status":"PASS"if floating_ok and discrete_ok else"FAIL"})
  for prefix,summary in (("cpu_heap",heap),("rss",rss),("gpu",gpu)):
   for field in ("before","after","peak","delta"):row[f"{prefix}_{field}_bytes"]=str(summary[field])if summary else"NA"
  [row.setdefault(k,"NA")for k in common.SUMMARY];main.append(row)
 main_path=out/f"operator_main_results_{a.operator}_{backend}_{run}.csv";common.write_csv(main_path,common.IDENTITY+common.SUMMARY,main);timing_columns=common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"];timing=[]
 for device,samples,digest in (("CPU",r["cpu_samples_ms"],r["cpu_output_digest"]),("GPU",r["gpu_samples_ms"],r["gpu_output_digest"])):
  for i,value in enumerate(samples):row=identity(device);row.update({"warmup_runs":str(warm),"measured_runs":str(measured),"sample_index":str(i),"latency_ms":str(value),"synchronized":"true","input_digest":r["actual_input_digest"],"output_digest":digest});timing.append(row)
 timing_path=out/f"operator_timing_samples_{a.operator}_{backend}_{run}.csv";common.write_csv(timing_path,timing_columns,timing);memory_columns=common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"];memory=[]
 for device,trace in (("CPU",r["cpu_trace"]),("GPU",r["gpu_trace"])):
  for point in trace:row=identity(device);row.update({"sample_index":"0","trace_phase":point["phase"],"phase_index":str(point["phase_index"]),"cpu_live_heap_bytes":str(point["heap"]["bytes"])if point["heap"]["available"]else"NA","rss_bytes":str(point["rss"]["bytes"])if point["rss"]["available"]else"NA","gpu_used_bytes":str(point["gpu"]["bytes"])if point["gpu"]["available"]else"NA"});memory.append(row)
 memory_path=out/f"operator_radartools_{a.operator}_memory_trace_{backend}.csv";common.write_csv(memory_path,memory_columns,memory);content=input_content(a.operator,s,dtype);parameters=json.dumps(q,sort_keys=True,separators=(",",":"));dtype_row={"run_id":run,"case_id":case,"operator_name":a.operator,"requested_dtype":dtype,"config_input_dtype":raw_dtype,"typed_input_manifest":r["typed_input_manifest"],"input_content_json":json.dumps(content,sort_keys=True,separators=(",",":")),"case_parameters_json":parameters,"actual_input_digest":r["actual_input_digest"],"output_dtype":r["output_dtype"],"backend":backend,"status":status,"error_code":error};dtype_path=out/f"operator_dtype_evidence_{a.operator}_{backend}_{run}.csv";common.write_csv(dtype_path,list(dtype_row),[dtype_row]);report={"run_id":run,"case_id":case,"operator_name":a.operator,"scale_id":s["scale_id"],"requested_dtype":dtype,"config_input_dtype":raw_dtype,"input_content":content,"case_parameters":q,"output_dtype":r["output_dtype"],"backend":backend,"warmup_runs":warm,"measured_runs":measured,**{x:r[x]for x in metrics},"exact_match":r["exact_match"],"mismatch_count":r["mismatch_count"],"semantic_check":r["semantic_check"],"actual_input_digest":r["actual_input_digest"],"status":status,"error_code":error};report_path=out/f"operator_{a.operator}_report.json";report_path.write_text(json.dumps(report,indent=2)+"\n");summary_path=out/f"operator_{a.operator}_summary.txt";summary_path.write_text("\n".join(f"{k}={v}"for k,v in report.items())+"\n");terminal=terminal_summary(main_path,[("main_csv",main_path),("timing_csv",timing_path),("memory_csv",memory_path),("dtype_csv",dtype_path),("report_json",report_path),("summary_txt",summary_path),("full_log",log)]);log.write_text(log.read_text()+json.dumps(report)+"\n"+terminal+"\n");print(terminal);print(f"OPERATOR_CASE_RADAR_CASE {status} operator={a.operator} scale={s['scale_id']} dtype={dtype} backend={backend} measured={measured} results={out}");return 0 if status=="PASS"else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
