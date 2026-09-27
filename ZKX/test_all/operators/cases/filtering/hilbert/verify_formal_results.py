#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def args():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",choices=("true","false"),required=True);p.add_argument("--output",required=True);return p.parse_args()
def one(d,p):
 v=list(d.glob(p))
 if len(v)!=1:raise ValueError(f"expected one {p} under {d}, found {len(v)}")
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 a=args();root=pathlib.Path(a.formal_root);reports=sorted(root.glob("*/operator_hilbert_report.json"))
 if len(reports)!=15:raise ValueError(f"expected 15 reports, found {len(reports)}")
 coverage=set();cases=set()
 for rp in reports:
  d=rp.parent;r=json.loads(rp.read_text());dtype=r["requested_dtype"];m=re.fullmatch(r"hilbert_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"])
  if not m or m.group(1).upper()!=dtype or r["status"]!="PASS" or r["backend"]!="fft_thrust":raise ValueError(f"report contract {d}")
  expected_out="ComplexFP64" if dtype.startswith("INT") else "ComplexFP32"
  if r["output_dtype"]!=expected_out:raise ValueError(f"output dtype {d}")
  coverage.add((dtype,m.group(2)));case=r["case_id"]
  if case in cases:raise ValueError(f"duplicate case {case}")
  cases.add(case);mainrows=rows(one(d,"operator_main_results_hilbert_fft_thrust_*.csv"));timing=rows(one(d,"operator_timing_samples_hilbert_fft_thrust_*.csv"));memory=rows(one(d,"operator_filtering_hilbert_memory_trace_fft_thrust.csv"));ev=rows(one(d,"operator_dtype_evidence_hilbert_fft_thrust_*.csv"))
  if len(mainrows)!=2 or {x["device"] for x in mainrows}!={"CPU","GPU"} or len(timing)!=200 or len(memory)!=14 or len(ev)!=1:raise ValueError(f"CSV counts {d}")
  for dev in ("CPU","GPU"):
   x=[v for v in timing if v["device"]==dev]
   if len(x)!=100 or {int(v["sample_index"]) for v in x}!=set(range(100)):raise ValueError(f"samples {d}/{dev}")
  e=ev[0];verify_input_content_evidence(e,("x",))
  if (e["requested_dtype"],e["dtype_semantics"],e["typed_input_count"],e["compute_dtype"],e["fft_dtype"],e["output_dtype"])!=(dtype,"actual_typed_array_input_fp32_fft_extension","1","FP32","ComplexFP32",expected_out):raise ValueError(f"dtype evidence {d}")
  if not re.fullmatch(r"x:"+dtype+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}",e["typed_input_manifest"]):raise ValueError(f"byte evidence {d}")
  for x in mainrows+timing+memory:
   if (x["git_commit"],x["git_dirty"],x["backend"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"fft_thrust","PASS",case):raise ValueError(f"identity {d}")
 expected={(d,s) for d in ("FP32","FP16","INT32","INT16","INT8") for s in ("10e2","10e3","10e4")}
 if coverage!=expected:raise ValueError("coverage")
 result="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b03 operator=hilbert backend=fft_thrust cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_array_input_fp32_fft_extension typed_inputs=x input_preview=heacuda_api_tail2 request_variant_coverage=5x3\n";out=pathlib.Path(a.output)
 if out.exists():raise ValueError("output exists")
 out.write_text(result);print(result,end="");return 0
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={e}",file=sys.stderr);sys.exit(2)
