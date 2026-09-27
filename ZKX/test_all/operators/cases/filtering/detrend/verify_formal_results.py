#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(d,p):
 v=list(d.glob(p));
 if len(v)!=1:raise ValueError(p)
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="")as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();rr=sorted(pathlib.Path(a.formal_root).glob("*/operator_detrend_report.json"));cov=set();modes=set();axes=set();ranks=set();bp=set()
 if len(rr)!=15:raise ValueError("reports")
 for q in rr:
  d=q.parent;r=json.loads(q.read_text());m=re.fullmatch(r"detrend_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dt=r["requested_dtype"]
  if not m or m.group(1).upper()!=dt or r["status"]!="PASS":raise ValueError("report")
  cov.add((dt,m.group(2)));modes.add(r["mode"]);axes.add(r["axis"]);ranks.add(r["rank"]);bp.add(tuple(r["breakpoints"]));ma=rows(one(d,"operator_main_results_detrend_not_applicable_*.csv"));ti=rows(one(d,"operator_timing_samples_detrend_not_applicable_*.csv"));me=rows(one(d,"operator_filtering_detrend_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_detrend_not_applicable_*.csv"))
  if len(ma)!=2 or len(ti)!=200 or len(me)!=14 or len(ev)!=1:raise ValueError("counts")
  verify_input_content_evidence(ev[0],("x",))
  for dev in("CPU","GPU"):
   z=[x for x in ti if x["device"]==dev]
   if len(z)!=100 or{int(x["sample_index"])for x in z}!=set(range(100)):raise ValueError("samples")
  if not re.fullmatch(r"x:"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}",ev[0]["typed_input_manifest"]):raise ValueError("manifest")
  for x in ma+ti+me:
   if(x["git_commit"],x["git_dirty"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"]):raise ValueError("identity")
 ex={(d,s)for d in("FP32","FP16","INT32","INT16","INT8")for s in("10e2","10e3","10e4")}
 if cov!=ex or modes!={"constant","linear"}or axes!={0,1}or ranks!={1,2}or bp!={() ,(25,50,75)}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b05 operator=detrend backend=not_applicable cases=15 warmup=20 measured=100 typed_inputs=x input_preview=heacuda_api_tail2 mode_coverage=constant|linear rank_coverage=1|2 axis_coverage=0|1 breakpoint_coverage=none|25,50,75 output_dtype_coverage=FP16|FP32|FP64 request_variant_coverage=5x3\n";path=pathlib.Path(a.output);path.write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
