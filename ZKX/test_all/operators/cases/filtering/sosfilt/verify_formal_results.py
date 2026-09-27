#!/usr/bin/env python3
import argparse,csv,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"channelize_poly"));from remaining_operator_case_terminal_summary import verify_input_content_evidence
def one(d,p):
 v=list(d.glob(p));
 if len(v)!=1:raise ValueError(p)
 return v[0]
def rows(p):
 with p.open(encoding="utf-8",newline="")as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",required=True);p.add_argument("--git-commit",required=True);p.add_argument("--git-dirty",required=True);p.add_argument("--output",required=True);a=p.parse_args();rr=sorted(pathlib.Path(a.formal_root).glob("*/operator_sosfilt_report.json"));cov=set();sec=set();axes=set();ranks=set();zi=set()
 if len(rr)!=15:raise ValueError("reports")
 for q in rr:
  d=q.parent;r=json.loads(q.read_text());m=re.fullmatch(r"sosfilt_(fp32|fp16|int32|int16|int8)_(10e2|10e3|10e4)",r["scale_id"]);dt=r["requested_dtype"]
  if not m or m.group(1).upper()!=dt or r["status"]!="PASS"or r["output_dtype"]!="FP32"or r["has_zf"]!=r["has_zi"]:raise ValueError("report")
  cov.add((dt,m.group(2)));sec.add(r["sections"]);axes.add(r["axis"]);ranks.add(r["rank"]);zi.add(r["has_zi"]);ma=rows(one(d,"operator_main_results_sosfilt_not_applicable_*.csv"));ti=rows(one(d,"operator_timing_samples_sosfilt_not_applicable_*.csv"));me=rows(one(d,"operator_filtering_sosfilt_memory_trace_not_applicable.csv"));ev=rows(one(d,"operator_dtype_evidence_sosfilt_not_applicable_*.csv"))
  if len(ma)!=2 or len(ti)!=200 or len(me)!=14 or len(ev)!=1 or ev[0]["dtype_semantics"]!="actual_typed_sos_x_optional_zi_fp32_extension":raise ValueError("evidence")
  verify_input_content_evidence(ev[0],("sos","x","zi") if r["has_zi"] else ("sos","x"))
  for dev in("CPU","GPU"):
   samples=[x for x in ti if x["device"]==dev]
   if len(samples)!=100 or {int(x["sample_index"])for x in samples}!=set(range(100)):raise ValueError("samples")
  manifest=ev[0]["typed_input_manifest"];expected_count="3"if r["has_zi"]else"2"
  for name in(("sos","x","zi")if r["has_zi"]else("sos","x")):
   if not re.search(r"(?:^|\|)"+name+":"+dt+r":\[[0-9]+\]:bytes=[0-9]+:sha256=[0-9a-f]{64}(?:\||$)",manifest):raise ValueError("manifest")
  if ev[0]["typed_input_count"]!=expected_count or (("|zi:"in manifest)!=r["has_zi"]):raise ValueError("typed input count")
  for x in ma+ti+me:
   if(x["git_commit"],x["git_dirty"],x["status"],x["case_id"])!=(a.git_commit,a.git_dirty,"PASS",r["case_id"]):raise ValueError("identity")
 ex={(d,s)for d in("FP32","FP16","INT32","INT16","INT8")for s in("10e2","10e3","10e4")}
 if cov!=ex or sec!={1,2,4}or axes!={0,1}or ranks!={1,2}or zi!={False,True}:raise ValueError("coverage")
 out="REMAINING_OPERATOR_CASE_FORMAL_DEEP_VERIFY PASS batch=b05 operator=sosfilt backend=not_applicable cases=15 warmup=20 measured=100 dtype_semantics=actual_typed_sos_x_optional_zi_fp32_extension typed_inputs=sos|x|optional_zi input_preview=heacuda_api_tail2 sections_coverage=1|2|4 axis_coverage=0|1 rank_coverage=1|2 zi_coverage=false|true output_dtype=FP32 request_variant_coverage=5x3\n";pathlib.Path(a.output).write_text(out);print(out,end="")
if __name__=="__main__":
 try:main()
 except Exception as e:print(e,file=sys.stderr);sys.exit(2)
