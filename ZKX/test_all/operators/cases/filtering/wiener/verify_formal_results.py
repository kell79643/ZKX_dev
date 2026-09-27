#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(d,p):
 v=list(d.glob(p))
 if len(v)!=1:raise ValueError(p)
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="")as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();rr=sorted(pathlib.Path(a.formal_root).glob("*/operator_wiener_report.json"));cov=set();windows=set();noise=set();ranks=set();shapes=set()
 if len(rr)!=15:raise ValueError("reports")
 for q in rr:
  d=q.parent;r=json.loads(q.read_text());m=re.fullmatch(r"wiener_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dt=r["requested_dtype"]
  if not m or m.group(1).upper()!=dt or r["status"]!="PASS"or r["output_dtype"]!="FP64":raise ValueError("report")
  cov.add((dt,m.group(2)));windows.add((r["window_mode"],tuple(r["window_shape"])));noise.add(r["noise_mode"]);ranks.add(r["rank"]);shapes.add(tuple(r["shape"]));ma=rows(one(d,"operator_main_results_wiener_not_applicable_*.csv"));ti=rows(one(d,"operator_timing_samples_wiener_not_applicable_*.csv"));me=rows(one(d,"operator_filtering_wiener_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_wiener_not_applicable_*.csv"))
  if len(ma)!=2 or len(ti)!=200 or len(me)!=14 or len(ev)!=1 or ev[0]["dtype_semantics"]!="actual_typed_x_fp32_nd_stats_host_fp64_extension":raise ValueError("evidence")
  verify_input_content_evidence(ev[0],("x",))
  if not re.fullmatch(r"x:"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}",ev[0]["typed_input_manifest"]):raise ValueError("manifest")
  for dev in("CPU","GPU"):
   z=[x for x in ti if x["device"]==dev]
   if len(z)!=100 or{int(x["sample_index"])for x in z}!=set(range(100)):raise ValueError("samples")
  for x in ma+ti+me:
   if(x["git_commit"],x["git_dirty"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"]):raise ValueError("identity")
 ex={(d,s)for d in("FP32","FP16","INT32","INT16","INT8")for s in("10e2","10e3","10e4")}
 if cov!=ex or windows!={("default",(3,)),("scalar",(5,5)),("per_dimension",(3,5))}or noise!={"automatic","explicit"}or ranks!={1,2}or shapes!={(100,),(32,32),(100,100)}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b05 operator=wiener backend=not_applicable cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_x_fp32_nd_stats_host_fp64_extension typed_inputs=x input_preview=heacuda_api_tail2 window_coverage=default3|scalar5x5|per_dimension3x5 noise_coverage=automatic|explicit rank_coverage=1|2 output_dtype=FP64 request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
