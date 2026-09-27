#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(d,p):
 v=list(d.glob(p));
 if len(v)!=1:raise ValueError(p)
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();reports=sorted(pathlib.Path(a.formal_root).glob("*/operator_upfirdn_report.json"))
 if len(reports)!=15:raise ValueError("reports")
 cov=set();ranks=set();axes=set();ratios=set();taps=set()
 for rp in reports:
  d=rp.parent;r=json.loads(rp.read_text());m=re.fullmatch(r"upfirdn_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dtype=r["requested_dtype"]
  if not m or m.group(1).upper()!=dtype or r["status"]!="PASS" or r["backend"]!="not_applicable" or r["output_dtype"]!=("FP64" if dtype=="INT32" else "FP32"):raise ValueError("report")
  expected={"10e2":(1,0,2,1,5),"10e3":(2,1,3,2,7),"10e4":(2,0,2,3,9)}[m.group(2)]
  if (r["rank"],r["axis"],r["up"],r["down"],r["tap_count"])!=expected:raise ValueError("matrix")
  cov.add((dtype,m.group(2)));ranks.add(r["rank"]);axes.add(r["axis"]);ratios.add((r["up"],r["down"]));taps.add(r["tap_count"]);mainrows=rows(one(d,"operator_main_results_upfirdn_not_applicable_*.csv"));timing=rows(one(d,"operator_timing_samples_upfirdn_not_applicable_*.csv"));memory=rows(one(d,"operator_filtering_upfirdn_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_upfirdn_not_applicable_*.csv"))
  if len(mainrows)!=2 or len(timing)!=200 or len(memory)!=14 or len(ev)!=1:raise ValueError("counts")
  for dev in ("CPU","GPU"):
   z=[x for x in timing if x["device"]==dev]
   if len(z)!=100 or {int(x["sample_index"]) for x in z}!=set(range(100)):raise ValueError("samples")
  e=ev[0];verify_input_content_evidence(e,("h","x"))
  if (e["requested_dtype"],e["dtype_semantics"],e["typed_input_count"],e["compute_dtype"],e["output_dtype"])!=(dtype,"actual_same_dtype_h_x_arrays","2","FP32","FP64" if dtype=="INT32" else "FP32"):raise ValueError("dtype")
  if not re.fullmatch(r"h:"+dtype+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|x:"+dtype+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}",e["typed_input_manifest"]):raise ValueError("manifest")
  for x in mainrows+timing+memory:
   if (x["git_commit"],x["git_dirty"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"]):raise ValueError("identity")
 expected={(d,s) for d in ("FP32","FP16","INT32","INT16","INT8") for s in ("10e2","10e3","10e4")}
 if cov!=expected or ranks!={1,2} or axes!={0,1} or ratios!={(2,1),(3,2),(2,3)} or taps!={5,7,9}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b04 operator=upfirdn backend=not_applicable cases=15 warmup=20 measured=100 typed_inputs=h|x input_preview=heacuda_api_tail2 dtype_semantics=actual_same_dtype_h_x_arrays rank_coverage=1|2 axis_coverage=0|1 ratio_coverage=2/1|3/2|2/3 tap_coverage=5|7|9 output_dtype_coverage=FP32|FP64 request_variant_coverage=5x3\n";path=pathlib.Path(a.output)
 if path.exists():raise ValueError("exists")
 path.write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={e}",file=sys.stderr);sys.exit(2)
