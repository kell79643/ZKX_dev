#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(directory,pattern):
 values=list(directory.glob(pattern))
 if len(values)!=1:raise ValueError(pattern)
 return values[0]
def rows(path):
 with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();reports=sorted(pathlib.Path(a.formal_root).glob("*/operator_freq_shift_report.json"));coverage=set();freqs=set();ranks=set();shapes=set()
 if len(reports)!=15:raise ValueError("reports")
 for report_path in reports:
  directory=report_path.parent;report=json.loads(report_path.read_text());match=re.fullmatch(r"freq_shift_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",report["scale_id"]);dtype=report["requested_dtype"]
  if not match or match.group(1).upper()!=dtype or report["status"]!="PASS" or report["output_dtype"]!="ComplexFP64" or report["backend"]!="not_applicable":raise ValueError("report")
  coverage.add((dtype,match.group(2)));freqs.add((report["freq"],report["fs"]));ranks.add(report["rank"]);shapes.add(tuple(report["shape"]));main_rows=rows(one(directory,"operator_main_results_freq_shift_not_applicable_*.csv"));timing=rows(one(directory,"operator_timing_samples_freq_shift_not_applicable_*.csv"));memory=rows(one(directory,"operator_filtering_freq_shift_memory_trace_not_applicable.csv"));evidence=rows(one(directory,"operator_dtype_evidence_freq_shift_not_applicable_*.csv"))
  if len(main_rows)!=2 or len(timing)!=200 or len(memory)!=14 or len(evidence)!=1:raise ValueError("counts")
  verify_input_content_evidence(evidence[0],("x",))
  for device in ("CPU","GPU"):
   selected=[x for x in timing if x["device"]==device]
   if len(selected)!=100 or {int(x["sample_index"]) for x in selected}!=set(range(100)):raise ValueError("samples")
  expected_manifest=r"x:"+dtype+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}"
  if not re.fullmatch(expected_manifest,evidence[0]["typed_input_manifest"]) or evidence[0]["dtype_semantics"]!="actual_typed_array_input_complex_fp32_compute_host_complex_fp64_extension" or evidence[0]["output_dtype"]!="ComplexFP64":raise ValueError("dtype evidence")
  for row in main_rows+timing+memory:
   if (row["git_commit"],row["git_dirty"],row["status"],row["case_id"])!=(a.git_commit,a.git_dirty,"PASS",report["case_id"]):raise ValueError("identity")
 expected={(dtype,scale) for dtype in ("FP32","FP16","INT32","INT16","INT8") for scale in ("10e2","10e3","10e4")}
 if coverage!=expected or freqs!={(1.25,100.0),(-7.5,256.0),(37.25,2000.0)} or ranks!={1,2} or shapes!={(100,),(32,32),(100,100)}:raise ValueError("coverage")
 output="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b05 operator=freq_shift backend=not_applicable cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_array_input_complex_fp32_compute_host_complex_fp64_extension typed_inputs=x input_preview=heacuda_api_tail2 frequency_coverage=1.25/100|-7.5/256|37.25/2000 rank_coverage=1|2 output_dtype=ComplexFP64 request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(output);print(output,end="")
if __name__=="__main__":
 try:main()
 except Exception as exc:print(exc,file=sys.stderr);sys.exit(2)
