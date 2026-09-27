#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"filtering"/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(d,p):
 v=list(d.glob(p))
 if len(v)!=1:raise ValueError(p)
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="")as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();rr=sorted(pathlib.Path(a.formal_root).glob("*/operator_vectorstrength_report.json"));cov=set();works=set();modes=set();counts=set();outputs=set()
 if len(rr)!=15:raise ValueError("reports")
 for q in rr:
  d=q.parent;r=json.loads(q.read_text());m=re.fullmatch(r"vectorstrength_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dt=r["requested_dtype"]
  if not m or m.group(1).upper()!=dt or r["status"]!="PASS"or r["backend"]!="not_applicable"or r["output_dtype"]!="FP64_pair"or r["scientific_work"]!=r["event_count"]*r["period_count"]or r["output_count"]!=r["period_count"]or r["scalar_period"]!=(r["period_mode"]=="scalar"):raise ValueError("report")
  cov.add((dt,m.group(2)));works.add(r["scientific_work"]);modes.add(r["period_mode"]);counts.add(r["period_count"]);outputs.add(r["output_dtype"]);ma=rows(one(d,"operator_main_results_vectorstrength_not_applicable_*.csv"));ti=rows(one(d,"operator_timing_samples_vectorstrength_not_applicable_*.csv"));me=rows(one(d,"operator_spectral_analysis_vectorstrength_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_vectorstrength_not_applicable_*.csv"))
  if len(ma)!=2 or len(ti)!=200 or len(me)!=14 or len(ev)!=1 or ev[0]["dtype_semantics"]!="actual_typed_events_and_scalar_or_array_period_fp32_compute_fp64_pair_extension":raise ValueError("evidence")
  verify_input_content_evidence(ev[0],("events","period") if r["period_mode"]=="scalar" else ("events","periods"))
  base=r"events:"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|"
  tail=(r"period:"+dt+r":scalar:bytes=[0-9]+:sha256=[0-9a-f]{64}"if r["period_mode"]=="scalar"else r"periods:"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}")
  if not re.fullmatch(base+tail,ev[0]["typed_input_manifest"]):raise ValueError("manifest")
  for dev in("CPU","GPU"):
   z=[x for x in ti if x["device"]==dev]
   if len(z)!=100 or{int(x["sample_index"])for x in z}!=set(range(100)):raise ValueError("samples")
  for x in ma+ti+me:
   if(x["git_commit"],x["git_dirty"],x["status"],x["case_id"],x["backend"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"],"not_applicable"):raise ValueError("identity")
 ex={(d,s)for d in("FP32","FP16","INT32","INT16","INT8")for s in("10e2","10e3","10e4")}
 if cov!=ex or works!={100,1024,10280}or modes!={"scalar","array"}or counts!={1,16,40}or outputs!={"FP64_pair"}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b06 operator=vectorstrength backend=not_applicable cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_events_and_scalar_or_array_period_fp32_compute_fp64_pair_extension typed_inputs=events|period_or_periods input_preview=heacuda_api_tail2 scientific_work_coverage=100|1024|10280 period_mode_coverage=scalar|array period_count_coverage=1|16|40 output_dtype=FP64_strength|FP64_phase request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
