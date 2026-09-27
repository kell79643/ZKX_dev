#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"filtering"/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def arguments():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",choices=("true","false"),required=True);p.add_argument("--output",required=True);return p.parse_args()
def one(d,p):
 v=list(d.glob(p))
 if len(v)!=1:raise ValueError(f"expected one {p} under {d}, found {len(v)}")
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 a=arguments();root=pathlib.Path(a.formal_root);reports=sorted(root.glob("*/operator_correlate2d_report.json"))
 if len(reports)!=15:raise ValueError(f"expected 15 reports, found {len(reports)}")
 coverage=set();cases=set();modes=set();boundaries=set()
 for rp in reports:
  d=rp.parent;r=json.loads(rp.read_text());dtype=r["requested_dtype"];m=re.fullmatch(r"correlate2d_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"])
  if not m or m.group(1).upper()!=dtype or r["status"]!="PASS" or r["backend"]!="not_applicable" or r["fft_dtype"]!="not_applicable" or r["output_dtype"]!=dtype or r["typed_input_count"]!=2:raise ValueError(f"report contract {d}")
  expected_mode={"10e2":"same","10e3":"full","10e4":"valid"}[m.group(2)];expected_boundary={"10e2":"fill","10e3":"wrap","10e4":"symm"}[m.group(2)]
  if (r["mode"],r["boundary"],r["boundary_effect"])!=(expected_mode,expected_boundary,"ignored_by_valid" if expected_mode=="valid" else "active"):raise ValueError(f"mode/boundary contract {d}")
  native=dtype in ("FP32","INT32");semantics="actual_same_dtype_arrays_native_modular_direct" if dtype=="INT32" else ("actual_same_dtype_arrays_native_direct" if native else "actual_same_dtype_arrays_fp32_legal_extension");compute="INT32_modular" if dtype=="INT32" else "FP32"
  if (r["dtype_semantics"],r["compute_dtype"],r["native_capability"])!=(semantics,compute,"native" if native else "approved_legal_extension"):raise ValueError(f"dtype semantics {d}")
  coverage.add((dtype,m.group(2)));modes.add(r["mode"]);boundaries.add(r["boundary"]);case=r["case_id"]
  if case in cases:raise ValueError(f"duplicate case {case}")
  cases.add(case);mainrows=rows(one(d,"operator_main_results_correlate2d_not_applicable_*.csv"));timing=rows(one(d,"operator_timing_samples_correlate2d_not_applicable_*.csv"));memory=rows(one(d,"operator_convolution_correlate2d_memory_trace_not_applicable.csv"));evidence=rows(one(d,"operator_dtype_evidence_correlate2d_not_applicable_*.csv"))
  if len(mainrows)!=2 or {x["device"] for x in mainrows}!={"CPU","GPU"} or len(timing)!=200 or len(memory)!=14 or len(evidence)!=1:raise ValueError(f"CSV counts {d}")
  for dev in ("CPU","GPU"):
   samples=[x for x in timing if x["device"]==dev]
   if len(samples)!=100 or {int(x["sample_index"]) for x in samples}!=set(range(100)):raise ValueError(f"samples {d}/{dev}")
  e=evidence[0];verify_input_content_evidence(e,("in1","in2"))
  if (e["requested_dtype"],e["dtype_semantics"],e["typed_input_count"],e["compute_dtype"],e["fft_dtype"],e["output_dtype"],e["native_capability"])!=(dtype,semantics,"2",compute,"not_applicable",dtype,"native" if native else "approved_legal_extension"):raise ValueError(f"dtype evidence {d}")
  pattern=r"in1:"+dtype+r":\[[0-9]+x[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|in2:"+dtype+r":\[[0-9]+x[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}"
  if not re.fullmatch(pattern,e["typed_input_manifest"]):raise ValueError(f"byte evidence {d}")
  for x in mainrows+timing+memory:
   if (x["git_commit"],x["git_dirty"],x["backend"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"not_applicable","PASS",case):raise ValueError(f"identity {d}")
 expected={(dtype,scale) for dtype in ("FP32","FP16","INT32","INT16","INT8") for scale in ("10e2","10e3","10e4")}
 if coverage!=expected or modes!={"full","same","valid"} or boundaries!={"fill","wrap","symm"}:raise ValueError("coverage")
 result="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b04 operator=correlate2d backend=not_applicable cases=15 warmup=20 measured=100 typed_inputs=in1|in2 input_preview=heacuda_api_tail2 native_dtypes=FP32|INT32 legal_extension_dtypes=FP16|INT16|INT8 mode_coverage=full|same|valid boundary_request_coverage=fill|wrap|symm valid_boundary_effect=ignored request_variant_coverage=5x3\n";out=pathlib.Path(a.output)
 if out.exists():raise ValueError("output exists")
 out.write_text(result);print(result,end="");return 0
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={e}",file=sys.stderr);sys.exit(2)
