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
 a=arguments();root=pathlib.Path(a.formal_root);reports=sorted(root.glob("*/operator_firwin2_report.json"))
 if len(reports)!=15:raise ValueError(f"expected 15 reports, found {len(reports)}")
 coverage=set();cases=set();types=set();windows=set()
 for rp in reports:
  d=rp.parent;r=json.loads(rp.read_text());dtype=r["requested_dtype"];m=re.fullmatch(r"firwin2_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"])
  if not m or m.group(1).upper()!=dtype or r["status"]!="PASS" or (r["backend"],r["fft_dtype"],r["output_dtype"],r["typed_input_count"])!=("fft_thrust","ComplexFP32","FP64",2):raise ValueError(f"report contract {d}")
  expected={"10e2":("I","hamming",17,33,561,False),"10e3":("II","none",32,65,2080,False),"10e4":("III","explicit",129,257,33153,True)}[m.group(2)]
  if (r["filter_type"],r["window"],r["numtaps"],r["nfreqs"],r["work_items"],r["antisymmetric"])!=expected:raise ValueError(f"scientific matrix {d}")
  coverage.add((dtype,m.group(2)));types.add(r["filter_type"]);windows.add(r["window"]);case=r["case_id"]
  if case in cases:raise ValueError(f"duplicate case {case}")
  cases.add(case);mainrows=rows(one(d,"operator_main_results_firwin2_fft_thrust_*.csv"));timing=rows(one(d,"operator_timing_samples_firwin2_fft_thrust_*.csv"));memory=rows(one(d,"operator_filter_design_firwin2_memory_trace_fft_thrust.csv"));evidence=rows(one(d,"operator_dtype_evidence_firwin2_fft_thrust_*.csv"))
  if len(mainrows)!=2 or {x["device"] for x in mainrows}!={"CPU","GPU"} or len(timing)!=200 or len(memory)!=14 or len(evidence)!=1:raise ValueError(f"CSV counts {d}")
  for dev in ("CPU","GPU"):
   samples=[x for x in timing if x["device"]==dev]
   if len(samples)!=100 or {int(x["sample_index"]) for x in samples}!=set(range(100)):raise ValueError(f"samples {d}/{dev}")
  e=evidence[0];verify_input_content_evidence(e,("freq","gain"))
  if (e["requested_dtype"],e["dtype_semantics"],e["typed_input_count"],e["compute_dtype"],e["fft_dtype"],e["output_dtype"],e["backend"])!=(dtype,"actual_same_dtype_freq_gain_arrays","2","FP32","ComplexFP32","FP64","fft_thrust"):raise ValueError(f"dtype evidence {d}")
  pattern=r"freq:"+dtype+r":\[3\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|gain:"+dtype+r":\[3\]:bytes=[0-9]+:sha256=[0-9a-f]{64}"
  if not re.fullmatch(pattern,e["typed_input_manifest"]):raise ValueError(f"byte evidence {d}")
  for x in mainrows+timing+memory:
   if (x["git_commit"],x["git_dirty"],x["backend"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"fft_thrust","PASS",case):raise ValueError(f"identity {d}")
 expected_coverage={(dtype,scale) for dtype in ("FP32","FP16","INT32","INT16","INT8") for scale in ("10e2","10e3","10e4")}
 if coverage!=expected_coverage or types!={"I","II","III"} or windows!={"hamming","none","explicit"}:raise ValueError("coverage")
 result="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b04 operator=firwin2 backend=fft_thrust cases=15 warmup=20 measured=100 typed_inputs=freq|gain input_preview=heacuda_api_tail2 dtype_semantics=actual_same_dtype_freq_gain_arrays compute_dtype=FP32 fft_dtype=ComplexFP32 output_dtype=FP64 filter_type_coverage=I|II|III window_coverage=hamming|none|explicit scale_formula=numtaps*nfreqs request_variant_coverage=5x3\n";out=pathlib.Path(a.output)
 if out.exists():raise ValueError("output exists")
 out.write_text(result);print(result,end="");return 0
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={e}",file=sys.stderr);sys.exit(2)
