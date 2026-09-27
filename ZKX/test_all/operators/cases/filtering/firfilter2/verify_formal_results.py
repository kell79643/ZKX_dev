#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def rows(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def one(d,p):
 v=list(d.glob(p))
 if len(v)!=1:raise ValueError(f"expected one {p}, found {len(v)}")
 return v[0]
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();reports=sorted(pathlib.Path(a.formal_root).glob("*/operator_firfilter2_report.json"))
 if len(reports)!=15:raise ValueError("reports")
 cov=set();pads=set();axes=set()
 for rp in reports:
  d=rp.parent;r=json.loads(rp.read_text());m=re.fullmatch(r"firfilter2_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dtype=r["requested_dtype"]
  if not m or m.group(1).upper()!=dtype or r["status"]!="PASS" or r["backend"]!="not_applicable" or r["output_dtype"]!=("FP64" if dtype=="INT32" else "FP32"):raise ValueError(f"report {d}")
  cov.add((dtype,m.group(2)));pads.add(r["padtype"]);axes.add(r["axis"]);mainrows=rows(one(d,"operator_main_results_firfilter2_not_applicable_*.csv"));timing=rows(one(d,"operator_timing_samples_firfilter2_not_applicable_*.csv"));memory=rows(one(d,"operator_filtering_firfilter2_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_firfilter2_not_applicable_*.csv"))
  if len(mainrows)!=2 or len(timing)!=200 or len(memory)!=14 or len(ev)!=1:raise ValueError(f"counts {d}")
  verify_input_content_evidence(ev[0],("x","b"))
  if not re.fullmatch(r"x:"+dtype+r":\[[0-9]+x[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}\|b:"+dtype+r":\[3\]:bytes=[0-9]+:sha256=[0-9a-f]{64}",ev[0]["typed_input_manifest"]):raise ValueError(f"typed {d}")
  for x in mainrows+timing+memory:
   if (x["git_commit"],x["git_dirty"],x["backend"],x["status"])!=(a.git_commit,a.git_dirty,"not_applicable","PASS"):raise ValueError(f"identity {d}")
 expected={(d,s) for d in ("FP32","FP16","INT32","INT16","INT8") for s in ("10e2","10e3","10e4")}
 if cov!=expected or pads!={"odd","even","constant"} or axes!={0,1}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b04 operator=firfilter2 backend=not_applicable cases=15 warmup=20 measured=100 typed_inputs=x|b input_preview=heacuda_api_tail2 compute_dtype=FP32 output_dtype=FP32|FP64 padtype_coverage=odd|even|constant axis_coverage=0|1 request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(f"REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY FAIL error={e}",file=sys.stderr);sys.exit(2)
