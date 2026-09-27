#!/usr/bin/env python3
import argparse,csv,hashlib,json,pathlib,subprocess,sys

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"cases"/"filtering"/"channelize_poly"))
import run_operator as common

OPS={"chirp","gausspulse","sawtooth","square"}

def table(title,columns,rows):
 values=[[str(row.get(key,"NA"))for key,_ in columns]for row in rows]
 widths=[max([len(label)]+[len(row[i])for row in values])for i,(_,label)in enumerate(columns)]
 line="+-"+"-+-".join("-"*width for width in widths)+"-+"
 body=[title,line,"| "+" | ".join(label.ljust(widths[i])for i,(_,label)in enumerate(columns))+" |",line]
 body.extend("| "+" | ".join(row[i].ljust(widths[i])for i in range(len(columns)))+" |"for row in values)
 body.append(line)
 return"\n".join(body)

def terminal_summary(main_path,paths):
 with main_path.open(encoding="utf-8",newline="")as handle:main_rows=list(csv.DictReader(handle))
 path_map=dict(paths)
 with path_map["timing_csv"].open(encoding="utf-8",newline="")as handle:timing_rows=list(csv.DictReader(handle))
 with path_map["dtype_csv"].open(encoding="utf-8",newline="")as handle:dtype_rows=list(csv.DictReader(handle))
 blocks=[table("[stage07] identity and status",[("target","target"),("scale_id","scale_id"),("device","device"),("backend","backend"),("dtype","dtype"),("warmup_runs","warmup"),("measured_runs","measured"),("timing_scope","timing_scope"),("status","status"),("error_code","error_code")],main_rows)]
 blocks.append(table("[stage07] timing detail",[("device","device"),("mean_ms","mean_ms"),("p50_ms","p50_ms"),("p95_ms","p95_ms"),("p99_ms","p99_ms"),("min_ms","min_ms"),("max_ms","max_ms"),("std_ms","std_ms"),("cv","cv"),("cpu_gpu_speedup","speedup")],main_rows))
 gpu=next(row for row in main_rows if row["device"]=="GPU")
 blocks.append(table("[stage07] accuracy detail",[("accuracy_reference","reference"),("mse","mse"),("rmse","rmse"),("relative_l2","relative_l2"),("relative_linf","relative_linf"),("exact_match","exact_match"),("mismatch_count","mismatch_count"),("semantic_check","semantic_check"),("accuracy_status","status")],[gpu]))
 resources=[]
 for row in main_rows:
  for prefix,label in (("cpu_heap","cpu_heap"),("rss","rss"),("gpu","gpu_memory")):
   if row[f"{prefix}_before_bytes"]!="NA":resources.append({"device":row["device"],"metric":label,"before":row[f"{prefix}_before_bytes"],"after":row[f"{prefix}_after_bytes"],"peak":row[f"{prefix}_peak_bytes"],"delta":row[f"{prefix}_delta_bytes"]})
 blocks.append(table("[stage07] resource detail",[("device","device"),("metric","metric"),("before","before_bytes"),("after","after_bytes"),("peak","peak_bytes"),("delta","delta_bytes")],resources))
 content=json.loads(dtype_rows[0]["input_content_json"])
 parameters=[{"parameters":dtype_rows[0]["case_parameters_json"]}]
 blocks.append(table("[stage07] input content",[("input_name","input_name"),("shape","shape"),("dtype","dtype"),("element_count","elements"),("generator","generator"),("preview_heacuda_api_tail2","values(head4/tail2)")],content))
 blocks.append(table("[stage07] case parameters",[("parameters","parameters")],parameters))
 io=[]
 for device in ("CPU","GPU"):
  row=next(item for item in timing_rows if item["device"]==device)
  io.append({"device":device,"input_digest":row["input_digest"],"output_digest":row["output_digest"]})
 blocks.append(table("[stage07] input and output identity",[("device","device"),("input_digest","input_digest"),("output_digest","output_digest")],io))
 blocks.append(table("[stage07] dtype evidence",[("requested_dtype","requested_dtype"),("config_input_dtype","config_input_dtype"),("output_dtype","output_dtype"),("typed_array_digest","typed_array_digest"),("actual_input_digest","actual_input_digest")],dtype_rows))
 blocks.append(table("[stage07] evidence files",[("kind","kind"),("path","path")],[{"kind":kind,"path":str(path)}for kind,path in paths]+[{"kind":"result_dir","path":str(main_path.parent)}]))
 return"\n\n".join(blocks)

def main():
 parser=argparse.ArgumentParser()
 for name in("binary","operator","config","thresholds","results-root","scale","run-type","git-commit","git-dirty","test-time-utc","random"):parser.add_argument("--"+name,required=True)
 args=parser.parse_args()
 if args.operator not in OPS:raise ValueError("operator is outside Stage07 Batch02")
 config_bytes=pathlib.Path(args.config).read_bytes();config=json.loads(config_bytes);config_sha=hashlib.sha256(config_bytes).hexdigest()
 operator=next(item for item in config["operators"]if item["operator_name"]==args.operator)
 scale=next(item for item in operator["scales"]if item["scale_id"]==args.scale)
 dtype=scale["inputs"][0]["dtype"];params=scale["parameters"]
 threshold=next(item for item in json.loads(pathlib.Path(args.thresholds).read_text())["entries"]if item["target"]==args.operator and item["dtype"]==dtype)
 warmup,measured=(20,100)if args.run_type=="formal"else(1,1)
 run_id=f"{args.test_time_utc}_{args.git_commit[:12]}_operator_case_b02_{args.run_type}_{args.random}"
 output_dir=pathlib.Path(args.results_root)/args.run_type/run_id
 if output_dir.exists():raise RuntimeError("result exists")
 output_dir.mkdir(parents=True)
 raw_path=output_dir/f"{args.operator}_probe_raw.json"
 if args.operator=="chirp":values=(params["f0"],params["t1"],params["f1"],params["phase_degrees"])
 elif args.operator=="gausspulse":values=(params["fc"],params["bw"],0,0)
 else:values=(params["width"]if args.operator=="sawtooth"else params["duty"],0,0,0)
 command=[args.binary,"--dtype",dtype,"--count",str(scale["actual_elements"]),"--p0",str(values[0]),"--p1",str(values[1]),"--p2",str(values[2]),"--p3",str(values[3]),"--warmup",str(warmup),"--measured",str(measured),"--output",str(raw_path)]
 completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 log_path=output_dir/f"operator_{args.operator}_full.log"
 log_path.write_text("command="+" ".join(command)+"\n"+completed.stdout)
 if completed.returncode:raise RuntimeError("probe failed: "+completed.stdout)
 raw=json.loads(raw_path.read_text())
 metric_names=("mse","rmse","relative_l2","relative_linf")
 accuracy_ok=all(raw[name]<=threshold[name+"_max"]for name in metric_names)
 summaries=[common.resource_summary(raw[key],metric)for key,metric in (("cpu_trace","heap"),("cpu_trace","rss"),("gpu_trace","heap"),("gpu_trace","rss"),("gpu_trace","gpu"))]
 resource_ok=all(item is not None for item in summaries)
 status="PASS"if accuracy_ok and resource_ok else"FAIL"
 error="OK"if status=="PASS"else"RESOURCE_FAILED"if not resource_ok else"ACCURACY_FAILED"
 case_id=f"{args.operator}__{scale['scale_id']}__{raw['actual_input_digest'][:16]}"
 def identity(device):
  return dict(zip(common.IDENTITY,["1",run_id,args.run_type,args.test_time_utc,args.git_commit,args.git_dirty,config["scale_set_id"],config_sha,"operator",args.operator,"Task2","Step1",args.operator,case_id,scale["scale_id"],scale["order_of_magnitude"],str(scale["actual_elements"]),f"{scale['inputs'][0]['input_name']}={scale['inputs'][0]['shape']}",dtype,device,"not_applicable","compatibility/end-to-end",status,error]))
 cpu_stats,gpu_stats=common.stats(raw["cpu_samples_ms"]),common.stats(raw["gpu_samples_ms"]);main_rows=[]
 for device,stats,heap,rss,gpu in (("CPU",cpu_stats,summaries[0],summaries[1],None),("GPU",gpu_stats,summaries[2],summaries[3],summaries[4])):
  row=identity(device);row.update({key:str(value)for key,value in stats.items()})
  row.update({"warmup_runs":str(warmup),"measured_runs":str(measured),"cpu_gpu_speedup":str(cpu_stats["mean_ms"]/gpu_stats["mean_ms"])if device=="GPU"else"NA","accuracy_reference":"typed_cpu_reference","mse":str(raw["mse"]if device=="GPU"else 0),"rmse":str(raw["rmse"]if device=="GPU"else 0),"relative_l2":str(raw["relative_l2"]if device=="GPU"else 0),"relative_linf":str(raw["relative_linf"]if device=="GPU"else 0),"exact_match":"NA","mismatch_count":"NA","semantic_check":f"output_shape=[{scale['actual_elements']}];output_dtype={raw['output_dtype']};finite_numeric_output=true","accuracy_status":"PASS"if accuracy_ok else"FAIL"})
  for prefix,summary in (("cpu_heap",heap),("rss",rss),("gpu",gpu)):
   for field in("before","after","peak","delta"):row[f"{prefix}_{field}_bytes"]=str(summary[field])if summary else"NA"
  [row.setdefault(key,"NA")for key in common.SUMMARY];main_rows.append(row)
 main_path=output_dir/f"operator_main_results_{args.operator}_not_applicable_{run_id}.csv"
 common.write_csv(main_path,common.IDENTITY+common.SUMMARY,main_rows)
 timing_columns=common.IDENTITY+["warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"]
 timing_rows=[]
 for device,samples,digest in (("CPU",raw["cpu_samples_ms"],raw["cpu_output_digest"]),("GPU",raw["gpu_samples_ms"],raw["gpu_output_digest"])):
  for index,value in enumerate(samples):
   row=identity(device);row.update({"warmup_runs":str(warmup),"measured_runs":str(measured),"sample_index":str(index),"latency_ms":str(value),"synchronized":"true","input_digest":raw["actual_input_digest"],"output_digest":digest});timing_rows.append(row)
 timing_path=output_dir/f"operator_timing_samples_{args.operator}_not_applicable_{run_id}.csv"
 common.write_csv(timing_path,timing_columns,timing_rows)
 memory_columns=common.IDENTITY+["sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"]
 memory_rows=[]
 for device,trace in (("CPU",raw["cpu_trace"]),("GPU",raw["gpu_trace"])):
  for point in trace:
   row=identity(device);row.update({"sample_index":"0","trace_phase":point["phase"],"phase_index":str(point["phase_index"]),"cpu_live_heap_bytes":str(point["heap"]["bytes"])if point["heap"]["available"]else"NA","rss_bytes":str(point["rss"]["bytes"])if point["rss"]["available"]else"NA","gpu_used_bytes":str(point["gpu"]["bytes"])if point["gpu"]["available"]else"NA"});memory_rows.append(row)
 memory_path=output_dir/f"operator_waveforms_{args.operator}_memory_trace_not_applicable.csv"
 common.write_csv(memory_path,memory_columns,memory_rows)
 input_item=scale["inputs"][0]
 input_content=[{"input_name":input_item["input_name"],"shape":"["+"x".join(str(x)for x in input_item["shape"])+"]","dtype":dtype,"element_count":str(input_item["element_count"]),"generator":raw["input_generator"],"preview_heacuda_api_tail2":raw["input_preview_heacuda_api_tail2"]}]
 dtype_row={"run_id":run_id,"case_id":case_id,"operator_name":args.operator,"requested_dtype":dtype,"config_input_dtype":dtype,"typed_input_manifest":raw["typed_input_manifest"],"typed_array_digest":raw["typed_array_digest"],"input_content_json":json.dumps(input_content,sort_keys=True,separators=(",",":")),"case_parameters_json":json.dumps(params,sort_keys=True,separators=(",",":")),"actual_input_digest":raw["actual_input_digest"],"output_dtype":raw["output_dtype"],"backend":"not_applicable","status":status,"error_code":error}
 dtype_path=output_dir/f"operator_dtype_evidence_{args.operator}_not_applicable_{run_id}.csv"
 common.write_csv(dtype_path,list(dtype_row),[dtype_row])
 report={"run_id":run_id,"case_id":case_id,"operator_name":args.operator,"task_name":"Task2","step_name":"Step1","scale_id":scale["scale_id"],"requested_dtype":dtype,"config_input_dtype":dtype,"input_content":input_content,"case_parameters":params,"output_dtype":raw["output_dtype"],"backend":"not_applicable","warmup_runs":warmup,"measured_runs":measured,**{name:raw[name]for name in metric_names},"actual_input_digest":raw["actual_input_digest"],"typed_array_digest":raw["typed_array_digest"],"status":status,"error_code":error}
 report_path=output_dir/f"operator_{args.operator}_report.json";report_path.write_text(json.dumps(report,indent=2)+"\n")
 summary_path=output_dir/f"operator_{args.operator}_summary.txt";summary_path.write_text("\n".join(f"{key}={value}"for key,value in report.items())+"\n")
 paths=[("main_csv",main_path),("timing_csv",timing_path),("memory_csv",memory_path),("dtype_csv",dtype_path),("report_json",report_path),("summary_txt",summary_path),("full_log",log_path)]
 terminal=terminal_summary(main_path,paths)
 log_path.write_text(log_path.read_text()+json.dumps(report)+"\n"+terminal+"\n")
 print(terminal)
 print(f"OPERATOR_CASE_WAVEFORM_CASE {status} operator={args.operator} scale={scale['scale_id']} dtype={dtype} backend=not_applicable measured={measured} results={output_dir}")
 return 0 if status=="PASS"else 3

if __name__=="__main__":
 try:sys.exit(main())
 except Exception as error:print(error,file=sys.stderr);sys.exit(2)
