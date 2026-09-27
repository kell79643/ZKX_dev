#!/usr/bin/env python3
import argparse,csv,importlib.util,json,pathlib,re,sys
S=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(S));from batch06_config import expand_config
B=S.parents[1]/"shared"/"filtering"/"verify_formal_results.py";sp=importlib.util.spec_from_file_location("base",B);base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
OPS={"cubic":"Step3","argrelextrema":"Step5","kalman_filter":"Step6"}
def args():
 p=argparse.ArgumentParser()
 for n in ("formal-root","driver-log-root","config","thresholds","git-commit","git-dirty","verifier-commit","output"):p.add_argument("--"+n,required=True)
 return p.parse_args()
def rows(p):
 with pathlib.Path(p).open(encoding="utf-8",newline="") as h:return list(csv.DictReader(h))
def identity(r,a,s,d):
 expected={"run_type":"formal","git_commit":a.git_commit,"git_dirty":a.git_dirty,"target_kind":"operator","target":s["operator_name"],"task_name":"Task2","step_name":OPS[s["operator_name"]],"operator_name":s["operator_name"],"scale_id":s["scale_id"],"dtype":s["inputs"][0]["dtype"],"device":d,"backend":"not_applicable","timing_scope":"compatibility/end-to-end","status":"PASS","error_code":"OK"}
 for k,v in expected.items():base.require(r.get(k)==v,f"{s['scale_id']} {d} {k}: {r.get(k)} != {v}")
def selected(a):
 logs=sorted(pathlib.Path(a.driver_log_root).glob("*/*.log"));base.require(len(logs)==45,f"expected 45 logs, found {len(logs)}")
 pat=re.compile(r"^OPERATOR_CASE_OPERATOR_BENCHMARKINAL_CASE PASS .* results=(\S+)$",re.M);root=pathlib.Path(a.formal_root).resolve();out=[];records=[]
 for log in logs:
  m=pat.findall(log.read_text());base.require(len(m)==1,f"{log} PASS paths={len(m)}");d=pathlib.Path(m[0]).resolve();base.require(d.parent==root and d.is_dir(),f"bad result {d}");out.append(d);records.append({"driver_log":str(log.resolve()),"result_directory":str(d)})
 base.require(len(set(out))==45,"selected directories not unique");return out,records
def audit(d,a,s,t):
 report=json.loads(base.unique_file(d,"operator_*_report.json").read_text());op=s["operator_name"];dtype=s["inputs"][0]["dtype"];disc=op=="argrelextrema"
 for k,v in (("operator_name",op),("scale_id",s["scale_id"]),("requested_dtype",dtype),("backend","not_applicable"),("step_name",OPS[op]),("status","PASS"),("error_code","OK")):base.require(report[k]==v,f"report {k}")
 base.require(report["warmup_runs"]==20 and report["measured_runs"]==100,"run counts")
 if disc:base.require(report["exact_match"] is True and report["mismatch_count"]==0 and report["semantic_check"] is True,"discrete accuracy")
 else:base.require(report["output_count"]==s["parameters"]["output_count"],"output count")
 main=rows(base.unique_file(d,"operator_main_results_*.csv"));base.require(len(main)==2,"main rows");by={r["device"]:r for r in main};base.require(set(by)=={"CPU","GPU"},"devices")
 for dev,r in by.items():identity(r,a,s,dev);base.require(r["warmup_runs"]=="20" and r["measured_runs"]=="100","main counts")
 timing=rows(base.unique_file(d,"operator_timing_samples_*.csv"));base.require(len(timing)==200,"timing rows");dig=set();summ={}
 for dev in ("CPU","GPU"):
  rr=[x for x in timing if x["device"]==dev];base.require(len(rr)==100,"timing count");base.require({int(x["sample_index"]) for x in rr}==set(range(100)),"indices");vals=[];ods=set()
  for x in rr:identity(x,a,s,dev);base.require(x["synchronized"]=="true","sync");dig.add(x["input_digest"]);ods.add(x["output_digest"]);vals.append(base.finite_number(x["latency_ms"],"latency"))
  base.require(len(ods)==1 and all(v>0 for v in vals),"timing evidence");summ[dev]=base.timing_summary(vals)
  for f,v in summ[dev].items():base.close(float(by[dev][f]),v,f"{dev}.{f}")
 base.require(len(dig)==1,"input digest");base.close(float(by["GPU"]["cpu_gpu_speedup"]),summ["CPU"]["mean_ms"]/summ["GPU"]["mean_ms"],"speedup")
 if disc:
  for m in base.FLOAT_METRICS:base.require(by["GPU"][m]=="NA",f"discrete {m} must be NA")
  base.require(by["GPU"]["exact_match"]=="true" and by["GPU"]["mismatch_count"]=="0","discrete fields")
 else:
  for m in base.FLOAT_METRICS:
   v=base.finite_number(by["GPU"][m],m);base.require(v<=float(t[m+"_max"]),f"{m} threshold");base.close(v,float(report[m]),f"report {m}")
 base.require(by["GPU"]["accuracy_status"]=="PASS","accuracy status")
 mem=rows(base.unique_file(d,"operator_*_memory_trace_*.csv"));base.require(len(mem)==14,"memory rows")
 for dev in ("CPU","GPU"):
  rr=sorted((x for x in mem if x["device"]==dev),key=lambda x:int(x["phase_index"]));base.require([x["trace_phase"] for x in rr]==base.PHASES,f"{dev} phases")
  for x in rr:identity(x,a,s,dev)
  for cf,pre in (("cpu_live_heap_bytes","cpu_heap"),("rss_bytes","rss"),("gpu_used_bytes","gpu")):
   vv=[x[cf] for x in rr]
   if dev=="CPU" and cf=="gpu_used_bytes":base.require(set(vv)=={"NA"},"CPU gpu memory");continue
   base.require(all(x!="NA" for x in vv),f"{dev}.{cf} NA");nn=[int(x) for x in vv];ex={"before":nn[0],"after":nn[-1],"peak":max(nn),"delta":nn[-1]-nn[0]}
   for f,v in ex.items():base.require(int(by[dev][f"{pre}_{f}_bytes"])==v,f"{dev}.{pre}.{f}")
 dtype_row=rows(base.unique_file(d,"operator_dtype_evidence_*.csv"));base.require(len(dtype_row)==1,"dtype rows");dr=dtype_row[0];base.require(dr["actual_input_digest"] in dig and dr["status"]=="PASS","dtype digest/status")
 content=json.loads(dr["input_content_json"]);params=json.loads(dr["case_parameters_json"]);base.require(params==s["parameters"] and content==report["input_content"],"content/params")
 base.require(len(content)==1 and content[0]["input_name"]==s["inputs"][0]["input_name"] and content[0]["element_count"]==str(s["inputs"][0]["element_count"]),"input evidence");base.require(content[0]["generator"] and content[0]["preview_heacuda_api_tail2"],"input preview")
 log=base.unique_file(d,"operator_*_full.log").read_text();base.unique_file(d,"operator_*_summary.txt");base.unique_file(d,"*_probe_raw.json")
 for marker in base.TABLE_MARKERS:base.require(marker in log,f"missing table {marker}")
 base.require("OPERATOR_CASE_OPERATOR_BENCHMARKINAL_PROBE PASS" in log,"probe marker")
 return {"operator":op,"scale_id":s["scale_id"],"dtype":dtype,"backend":"not_applicable","input_digest":next(iter(dig)),"status":"PASS"}
def main():
 a=args();o=pathlib.Path(a.output)
 if o.exists():raise RuntimeError(f"refusing to overwrite {o}")
 cfg=json.loads(pathlib.Path(a.config).read_text());sc=expand_config(cfg);expected={(x["operator_name"],x["scale_id"]):x for x in sc};base.require(len(expected)==45,"expected 45")
 td=json.loads(pathlib.Path(a.thresholds).read_text());th={(x["target"],x["dtype"]):x for x in td["entries"]};dirs,records=selected(a);cases=[];fails=[];seen=set()
 for d in dirs:
  try:
   r=json.loads(base.unique_file(d,"operator_*_report.json").read_text());k=(r["operator_name"],r["scale_id"]);base.require(k in expected,f"unexpected {k}");base.require(k not in seen,f"duplicate {k}");seen.add(k);cases.append(audit(d,a,expected[k],None if k[0]=="argrelextrema" else th[(k[0],expected[k]["inputs"][0]["dtype"])]))
  except Exception as e:fails.append({"directory":str(d),"error":str(e)})
 for k in sorted(set(expected)-seen):fails.append({"directory":"NA","error":f"missing {k}"})
 status="PASS" if len(cases)==45 and not fails else "FAIL";result={"schema_version":1,"audit_kind":"operator_case_batch06_formal_evidence","formal_root":a.formal_root,"driver_log_root":a.driver_log_root,"result_git_commit":a.git_commit,"result_git_dirty":a.git_dirty,"verifier_git_commit":a.verifier_commit,"expected_case_count":45,"case_count":len(cases),"selection_records":records,"cases":cases,"failures":fails,"status":status,"output":str(o)};o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(result,indent=2)+"\n")
 for op in OPS:print(f"{op}: cases={sum(x['operator']==op for x in cases)} status={'PASS' if sum(x['operator']==op for x in cases)==15 else 'FAIL'}")
 print(f"OPERATOR_CASE_TASK2_PIPELINE_OPERATOR_AUDIT {status} cases={len(cases)} failures={len(fails)} output={o}");return 0 if status=="PASS" else 3
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"OPERATOR_CASE_TASK2_PIPELINE_OPERATOR_AUDIT FAIL error={e}",file=sys.stderr);sys.exit(2)
