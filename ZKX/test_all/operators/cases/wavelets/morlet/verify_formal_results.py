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
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();rr=sorted(pathlib.Path(a.formal_root).glob("*/operator_morlet_report.json"));cov=set();ns=set();complete=set();outputs=set()
 if len(rr)!=15:raise ValueError("reports")
 for q in rr:
  d=q.parent;r=json.loads(q.read_text());m=re.fullmatch(r"morlet_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dt=r["requested_dtype"]
  if not m or m.group(1).upper()!=dt or r["status"]!="PASS"or r["backend"]!="not_applicable"or r["output_dtype"]!="ComplexFP64"or r["output_count"]!=r["n"]:raise ValueError("report")
  cov.add((dt,m.group(2)));ns.add(r["n"]);complete.add(r["complete"]);outputs.add(r["output_dtype"]);ma=rows(one(d,"operator_main_results_morlet_not_applicable_*.csv"));ti=rows(one(d,"operator_timing_samples_morlet_not_applicable_*.csv"));me=rows(one(d,"operator_wavelets_morlet_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_morlet_not_applicable_*.csv"))
  if len(ma)!=2 or len(ti)!=200 or len(me)!=14 or len(ev)!=1 or ev[0]["dtype_semantics"]!="actual_typed_frequency_scale_scalars_fp32_compute_complex_fp64_storage_extension"or ev[0]["typed_input_count"]!="2":raise ValueError("evidence")
  verify_input_content_evidence(ev[0],("frequency","scale"))
  pat=r"frequency:"+dt+r":scalar:bytes=[0-9]+:sha256=[0-9a-f]{64}\|scale:"+dt+r":scalar:bytes=[0-9]+:sha256=[0-9a-f]{64}"
  if not re.fullmatch(pat,ev[0]["typed_input_manifest"]):raise ValueError("manifest")
  for dev in("CPU","GPU"):
   z=[x for x in ti if x["device"]==dev]
   if len(z)!=100 or{int(x["sample_index"])for x in z}!=set(range(100)):raise ValueError("samples")
  for x in ma+ti+me:
   if(x["git_commit"],x["git_dirty"],x["status"],x["case_id"],x["backend"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"],"not_applicable"):raise ValueError("identity")
 ex={(d,s)for d in("FP32","FP16","INT32","INT16","INT8")for s in("10e2","10e3","10e4")}
 if cov!=ex or ns!={100,1024,10000}or complete!={True,False}or outputs!={"ComplexFP64"}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b07 operator=morlet backend=not_applicable cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_frequency_scale_scalars_fp32_compute_complex_fp64_storage_extension typed_inputs=frequency|scale input_preview=heacuda_api_tail2 n_coverage=100|1024|10000 complete_coverage=true|false output_dtype=ComplexFP64 request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
