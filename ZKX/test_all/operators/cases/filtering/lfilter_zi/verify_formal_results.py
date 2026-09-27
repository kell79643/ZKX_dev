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
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();rr=sorted(pathlib.Path(a.formal_root).glob("*/operator_lfilter_zi_report.json"));cov=set();orders=set();work=set();leading=set()
 if len(rr)!=15:raise ValueError("reports")
 for q in rr:
  d=q.parent;r=json.loads(q.read_text());m=re.fullmatch(r"lfilter_zi_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dt=r["requested_dtype"]
  if not m or m.group(1).upper()!=dt or r["status"]!="PASS"or r["output_dtype"]!="FP64"or r["output_count"]!=r["state_order"]:raise ValueError("report")
  cov.add((dt,m.group(2)));orders.add(r["state_order"]);work.add(r["scientific_work_units"]);leading.add(r["leading_denominator_zeros"]);ma=rows(one(d,"operator_main_results_lfilter_zi_not_applicable_*.csv"));ti=rows(one(d,"operator_timing_samples_lfilter_zi_not_applicable_*.csv"));me=rows(one(d,"operator_filtering_lfilter_zi_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_lfilter_zi_not_applicable_*.csv"))
  if len(ma)!=2 or len(ti)!=200 or len(me)!=14 or len(ev)!=1:raise ValueError("counts")
  verify_input_content_evidence(ev[0],("b","a"))
  for dev in("CPU","GPU"):
   z=[x for x in ti if x["device"]==dev]
   if len(z)!=100 or{int(x["sample_index"])for x in z}!=set(range(100)):raise ValueError("samples")
  pat=r"b:"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|a:"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}"
  if not re.fullmatch(pat,ev[0]["typed_input_manifest"])or ev[0]["dtype_semantics"]!="actual_typed_b_a_arrays_fp32_solve_host_fp64_extension":raise ValueError("evidence")
  for x in ma+ti+me:
   if(x["git_commit"],x["git_dirty"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"]):raise ValueError("identity")
 ex={(d,s)for d in("FP32","FP16","INT32","INT16","INT8")for s in("10e2","10e3","10e4")}
 if cov!=ex or orders!={5,10,22}or work!={125,1000,10648}or leading!={0,1}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b05 operator=lfilter_zi backend=not_applicable cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_b_a_arrays_fp32_solve_host_fp64_extension typed_inputs=b|a input_preview=heacuda_api_tail2 state_order_coverage=5|10|22 scientific_work_coverage=125|1000|10648 leading_zero_coverage=0|1 output_dtype=FP64 request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
