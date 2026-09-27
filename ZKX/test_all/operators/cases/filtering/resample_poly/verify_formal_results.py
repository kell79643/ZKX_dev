#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(d,p):
 v=list(d.glob(p));
 if len(v)!=1:raise ValueError(f"expected one {p}, found {len(v)}")
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();reports=sorted(pathlib.Path(a.formal_root).glob("*/operator_resample_poly_report.json"))
 if len(reports)!=15:raise ValueError("reports")
 cov=set();modes=set();axes=set()
 for rp in reports:
  d=rp.parent;r=json.loads(rp.read_text());m=re.fullmatch(r"resample_poly_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dtype=r["requested_dtype"]
  if not m or m.group(1).upper()!=dtype or r["status"]!="PASS" or r["backend"]!="not_applicable":raise ValueError("report")
  mode={"10e2":"default_kaiser","10e3":"identity","10e4":"custom"}[m.group(2)];out=dtype if mode=="identity" else ("FP64" if mode=="default_kaiser" or dtype=="INT32" else "FP32")
  if (r["filter_mode"],r["output_dtype"],r["typed_input_count"])!=(mode,out,2 if mode=="custom" else 1):raise ValueError("semantic")
  cov.add((dtype,m.group(2)));modes.add(mode);axes.add(r["axis"]);mainrows=rows(one(d,"operator_main_results_resample_poly_not_applicable_*.csv"));timing=rows(one(d,"operator_timing_samples_resample_poly_not_applicable_*.csv"));memory=rows(one(d,"operator_filtering_resample_poly_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_resample_poly_not_applicable_*.csv"))
  if len(mainrows)!=2 or len(timing)!=200 or len(memory)!=14 or len(ev)!=1:raise ValueError("counts")
  for dev in ("CPU","GPU"):
   x=[v for v in timing if v["device"]==dev]
   if len(x)!=100 or {int(v["sample_index"]) for v in x}!=set(range(100)):raise ValueError("samples")
  e=ev[0];verify_input_content_evidence(e,("x","h") if mode=="custom" else ("x",));count="2" if mode=="custom" else "1";sem="actual_typed_array_inputs" if mode=="custom" else "actual_typed_array_input";compute="identity_copy" if mode=="identity" else "FP32"
  if (e["requested_dtype"],e["dtype_semantics"],e["typed_input_count"],e["compute_dtype"],e["output_dtype"])!=(dtype,sem,count,compute,out):raise ValueError("dtype evidence")
  pat=r"x:"+dtype+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}"+(r"\|h:"+dtype+r":\[5\]:bytes=[0-9]+:sha256=[0-9a-f]{64}" if mode=="custom" else "")
  if not re.fullmatch(pat,e["typed_input_manifest"]):raise ValueError("manifest")
  case=r["case_id"]
  for x in mainrows+timing+memory:
   if (x["git_commit"],x["git_dirty"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"PASS",case):raise ValueError("identity")
 expected={(d,s) for d in ("FP32","FP16","INT32","INT16","INT8") for s in ("10e2","10e3","10e4")}
 if cov!=expected or modes!={"default_kaiser","identity","custom"} or axes!={0,1}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b04 operator=resample_poly backend=not_applicable cases=15 warmup=20 measured=100 typed_inputs=x|x+h input_preview=heacuda_api_tail2 filter_coverage=default_kaiser|identity|custom axis_coverage=0|1 output_dtype_coverage=request_dtype|FP32|FP64 request_variant_coverage=5x3\n";path=pathlib.Path(a.output)
 if path.exists():raise ValueError("output exists")
 path.write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={e}",file=sys.stderr);sys.exit(2)
