#!/usr/bin/env python3
"""CLI与交互菜单共用的唯一演示runner。"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_capabilities import ROOT, build_demo_capabilities
from execution import abnormal_adapter, fft_library_swap_adapter, operator_adapter, task_adapter
from execution.common import CleanTemporaryDirectory, verify_published_binary
from terminal_tables import render_major_end, render_major_start, render_minor_end, render_minor_start, render_result

EXIT_OK=0; EXIT_INVALID=2; EXIT_UNSUPPORTED=3; EXIT_FAIL=4
_BUILD_PLAN=json.loads((ROOT/"demo/config/build/demo_build_plan.json").read_text(encoding="utf-8"))
SUPPORTED_BACKENDS=tuple(_BUILD_PLAN["supported_backends"])
DEFAULT_BACKEND=SUPPORTED_BACKENDS[0]
@dataclass(frozen=True)
class DemoRequest:
 target:str; backend:str=DEFAULT_BACKEND; input_mode:str="default"; case_id:str|None=None
 operator:str|None=None; module:str|None=None; abnormal_type:str|None=None; runs:int=1; seed:int=0; dtype:str="FP32"; input_json:str|None=None
def _result(status:str,request:DemoRequest,message:str)->tuple[int,dict[str,Any]]:
 code={"INVALID_ARGUMENT":EXIT_INVALID,"UNSUPPORTED":EXIT_UNSUPPORTED,"FAIL":EXIT_FAIL}.get(status,EXIT_OK)
 return code,{"status":status,"target":request.target,"message":message,"request":asdict(request)}
def _task_profile(request:DemoRequest,caps:dict[str,Any])->dict[str,Any]|None:
 name=request.target.capitalize()
 matches=[x for x in caps["best_profiles"] if x.get("target_kind")=="task" and x.get("target")==name and x.get("backend")==caps["default_input_profile_backend"]]
 return matches[0] if len(matches)==1 else None
def _operator_profile(request:DemoRequest,caps:dict[str,Any],module:str,operator:str)->dict[str,Any]|None:
 matches=[x for x in caps["best_profiles"] if x.get("target_kind")=="operator" and x.get("module")==module and x.get("target")==operator and x.get("dtype")==request.dtype and x.get("backend") in (caps["default_input_profile_backend"],caps["shared_backend"])]
 return matches[0] if len(matches)==1 else None
def _command(caps:dict[str,Any],module:str,operator:str)->dict[str,Any]|None:
 matches=[x for x in caps["operator_commands"] if x["module"]==module and x["operator"]==operator]
 return matches[0] if len(matches)==1 else None
def _operator_selection(request:DemoRequest,caps:dict[str,Any],module:str,operator:str)->tuple[dict[str,Any]|None,dict[str,Any]|None]:
 profile=_operator_profile(request,caps,module,operator); command=_command(caps,module,operator)
 if not profile or not command: return None,None
 selected=dict(profile); routed=dict(command)
 trees=routed.get("binary_tree_by_backend",{})
 if request.backend not in routed.get("selectable_backends",[]) or request.backend not in trees: return None,None
 routed["binary_tree"]=trees[request.backend]
 routed["runtime_backend"]=request.backend if profile.get("backend")!=caps["shared_backend"] else caps["shared_backend"]
 if request.input_mode=="default": return selected,routed
 if request.input_mode=="json":
  if not request.input_json: return None,None
  path=Path(request.input_json).expanduser().resolve()
  try:
   payload=json.loads(path.read_text(encoding="utf-8")); operators=payload["operators"]
   entry=next(x for x in operators if x["operator_name"]==operator); scales=entry["scales"]
  except (OSError,ValueError,KeyError,TypeError,StopIteration): return None,None
  ids=[x.get("scale_id") for x in scales if isinstance(x,dict)]
  if len(ids)!=1 or not ids[0]: return None,None
  scale=ids[0]
  selected["best_scale_id"]=scale; selected["case_id"]=f"external_json__{operator}__{scale}"; routed["config"]=str(path)
  return selected,routed
 return None,None
def _task_input(request:DemoRequest,caps:dict[str,Any])->tuple[str,Path]|tuple[None,None]:
 standard=ROOT/"test_all/config/task_benchmarks/task_input_scales.json"
 if request.input_mode=="default":
  profile=_task_profile(request,caps); return (profile["best_scale_id"],standard) if profile else (None,None)
 if request.input_mode=="json":
  if not request.input_json: return None,None
  path=Path(request.input_json).expanduser().resolve()
  try: payload=json.loads(path.read_text(encoding="utf-8")); entries=payload["tasks"][request.target.capitalize()]["scales"]
  except (OSError,ValueError,KeyError,TypeError): return None,None
  ids=[x.get("scale_id") for x in entries if isinstance(x,dict)]
  return (ids[0],path) if len(ids)==1 and ids[0] else (None,None)
 return None,None
def _abnormal_case(request:DemoRequest,caps:dict[str,Any])->dict[str,str]|None:
 target=request.module if request.target=="abnormal" and request.module in ("task1","task2") else "operator"
 rows=caps["abnormal_cases"][target]
 if target=="operator": rows=[x for x in rows if (not request.module or x["module"]==request.module) and (not request.operator or x["operator"]==request.operator)]
 if request.case_id: rows=[x for x in rows if x["case_id"]==request.case_id]
 if request.abnormal_type: rows=[x for x in rows if x["abnormal_type"]==request.abnormal_type]
 return rows[0] if rows else None
def _operator_all_selections(request:DemoRequest,caps:dict[str,Any])->tuple[list[tuple[dict[str,Any],dict[str,Any]]],str|None]:
 selections=[]
 for item in caps["operators"]:
  profile,command=_operator_selection(request,caps,item["module"],item["operator"])
  if not profile or not command: return [],f"53/53预检失败：{item['module']}.{item['operator']}"
  _,binary_error=verify_published_binary(request.backend,command["binary_tree"],command["binary"])
  if binary_error: return [],f"53/53构建预检失败：{item['module']}.{item['operator']}：{binary_error}"
  selections.append((command,profile))
 return selections,None

def _execute_operator_all(request:DemoRequest,caps:dict[str,Any],results_root:Path|None=None)->dict[str,Any]:
 selections,error=_operator_all_selections(request,caps)
 if error: return {"status":"UNSUPPORTED","target":"operator-all","message":error}
 started=time.perf_counter(); completed=0; failures=[]
 for index,(command,profile) in enumerate(selections,1):
  print(f"\n[DEMO][OPERATOR] {index}/53 {command['module']}.{command['operator']}")
  retained=results_root/command["module"]/command["operator"] if results_root is not None else None
  item=operator_adapter.execute(command,profile,request.backend,1,retained); render_result(item)
  if item["status"]=="PASS": completed+=1
  else: failures.append({"target":item.get("target",f"operator-{index}"),"status":item.get("status","FAIL"),"message":item.get("message","")})
 elapsed=time.perf_counter()-started; status="PASS" if not failures else "FAIL"
 return {"status":status,"target":"operator-all","message":f"完成={completed}/53，失败={len(failures)}，wall_seconds={elapsed:.3f}","config":{"backend":request.backend,"dtype":request.dtype,"completed":f"{completed}/53","failed":len(failures)},"batch_failures":failures}

def _execute_fft_library_swap(major_index:int=1,major_total:int=1,render_major:bool=True)->dict[str,Any]:
 title="fft-library-swap（libzkx_fft_thrust.so平替libdlfft.so）"
 if render_major: render_major_start(major_index,major_total,title)
 started=time.perf_counter(); variants=[]
 for index,(name,label) in enumerate((("dlfft","libdlfft.so"),("zkx_fft_thrust","libzkx_fft_thrust.so")),1):
  render_minor_start(major_index,major_total,index,2,label); item_started=time.perf_counter()
  result=fft_library_swap_adapter.execute_variant(name); render_result(result); item_elapsed=time.perf_counter()-item_started
  render_minor_end(major_index,major_total,index,2,label,result.get("status","FAIL"),item_elapsed)
  variants.append(result)
 comparison=fft_library_swap_adapter.compare(variants[0],variants[1]); elapsed=time.perf_counter()-started
 if render_major: render_major_end(major_index,major_total,title,comparison["status"],elapsed)
 comparison["wall_seconds"]=elapsed
 return comparison

def _batch_preflight(request:DemoRequest,caps:dict[str,Any])->str|None:
 batch=caps["batch"]
 if batch["status"]!="READY": return batch["reason"]
 for target in ("task1","task2"):
  if not _task_profile(DemoRequest(target=target,backend=request.backend),caps): return f"{target}默认配置未准入"
  _,error=verify_published_binary(request.backend,request.backend,f"bin/{target}_pipeline_benchmark")
  if error: return error
 for dtype in batch["operator_dtypes"]:
  _,error=_operator_all_selections(DemoRequest(target="operator-all",backend=request.backend,dtype=dtype),caps)
  if error: return f"operator-all {dtype}：{error}"
 for name in ("task1_abnormal_input","task2_abnormal_input","operator_abnormal_input"):
  _,error=verify_published_binary(request.backend,request.backend,f"bin/{name}")
  if error: return error
 error=fft_library_swap_adapter.preflight()
 if error: return f"fft-library-swap：{error}"
 return None

def _run_batch_in_root(request:DemoRequest,caps:dict[str,Any],batch_root:Path)->dict[str,Any]:
 error=_batch_preflight(request,caps)
 if error: return {"status":"UNSUPPORTED","target":"batch-all","message":f"一键完整演示预检失败：{error}"}
 batch_started=time.perf_counter(); summary=[]; failures=[]; major_total=5
 def remember(section:str,code:int,result:dict[str,Any],elapsed:float,completed:str="1/1",failed:int|None=None)->None:
  count=failed if failed is not None else (0 if code==EXIT_OK else 1)
  summary.append({"section":section,"status":"PASS" if code==EXIT_OK else "FAIL","completed":completed,"failed":count,"elapsed_seconds":f"{elapsed:.3f}"})
  if code!=EXIT_OK:
   nested=result.get("batch_failures") or [{"target":result.get("target","-"),"status":result.get("status","FAIL"),"message":result.get("message","")}]
   for item in nested: failures.append({"section":section,**item})
 for major_index,target in ((1,"task1"),(2,"task2")):
  title=target.capitalize(); render_major_start(major_index,major_total,title); started=time.perf_counter()
  code,result=run_request(DemoRequest(target=target,backend=request.backend),caps); render_result(result); elapsed=time.perf_counter()-started
  status="PASS" if code==EXIT_OK else "FAIL"; render_major_end(major_index,major_total,title,status,elapsed); remember(title,code,result,elapsed)
 render_major_start(3,major_total,"operator-all"); operator_started=time.perf_counter(); operator_failed=0; fp32_failed=53
 dtypes=caps["batch"]["operator_dtypes"]
 for index,dtype in enumerate(dtypes,1):
  title=f"operator-all · {dtype}"; render_minor_start(3,major_total,index,len(dtypes),title); started=time.perf_counter()
  operator_request=DemoRequest(target="operator-all",backend=request.backend,dtype=dtype)
  if dtype=="FP32":
   result=_execute_operator_all(operator_request,caps,batch_root/"normal_fp32"); code=EXIT_OK if result.get("status")=="PASS" else EXIT_FAIL
  else: code,result=run_request(operator_request,caps)
  render_result(result); elapsed=time.perf_counter()-started; status="PASS" if code==EXIT_OK else "FAIL"; config=result.get("config",{})
  failed=int(config.get("failed",0)) if code!=EXIT_OK else 0
  if code!=EXIT_OK and failed==0: failed=1
  if dtype=="FP32": fp32_failed=failed
  render_minor_end(3,major_total,index,len(dtypes),title,status,elapsed,f"完成={config.get('completed','0/53')}，失败={failed}")
  operator_failed+=failed; remember(f"operator-all {dtype}",code,result,elapsed,config.get("completed","0/53"),failed)
 operator_elapsed=time.perf_counter()-operator_started; render_major_end(3,major_total,"operator-all","PASS" if operator_failed==0 else "FAIL",operator_elapsed)
 normal_links=batch_root/"normal_case_links.csv"; links_ready=False
 if fp32_failed==0:
  links_ready,link_message=abnormal_adapter.resolve_normal_links(batch_root/"normal_fp32",normal_links); print(link_message)
  if not links_ready: failures.append({"section":"normal-links","target":"53 FP32 operators","status":"FAIL","message":link_message})
 else:
  link_message=f"正常FP32算子未达到53/53（失败={fp32_failed}），禁止生成异常外键"; print("NORMAL_LINK_FAIL "+link_message)
 render_major_start(4,major_total,"abnormal"); abnormal_started=time.perf_counter(); abnormal_completed=0; abnormal_failed=0
 cases=[("task1",x) for x in caps["abnormal_cases"]["task1"]]+[("task2",x) for x in caps["abnormal_cases"]["task2"]]+[("operator",x) for x in caps["abnormal_cases"]["operator"]]
 operator_result_csvs=[]
 for index,(kind,case) in enumerate(cases,1):
  object_name=kind.capitalize() if kind!="operator" else f"{case['module']}.{case['operator']}"
  title=f"{object_name} · {case['abnormal_type']}"; render_minor_start(4,major_total,index,len(cases),title); started=time.perf_counter()
  kwargs={"target":"abnormal","backend":request.backend,"case_id":case["case_id"],"module":kind if kind!="operator" else case["module"]}
  if kind=="operator": kwargs["operator"]=case["operator"]
  if kind=="operator" and links_ready:
   result=abnormal_adapter.execute("operator",request.backend,case,normal_links,batch_root/"operator_abnormal"/f"case_{index:03d}")
   code=EXIT_OK if result.get("status")=="PASS" else EXIT_FAIL
   if code==EXIT_OK and result.get("_result_csv"): operator_result_csvs.append(Path(result["_result_csv"]))
  elif kind=="operator":
   result={"status":"FAIL","target":f"{case['module']}.{case['operator']}","message":"本轮53行正常外键未生成，拒绝运行异常case"}; code=EXIT_FAIL
  else: code,result=run_request(DemoRequest(**kwargs),caps)
  render_result(result); elapsed=time.perf_counter()-started; status=result.get("status","FAIL")
  render_minor_end(4,major_total,index,len(cases),title,status,elapsed,f"完成={'1/1' if code==EXIT_OK else '0/1'}，失败={0 if code==EXIT_OK else 1}")
  if code==EXIT_OK: abnormal_completed+=1
  else:
   abnormal_failed+=1; failures.append({"section":f"abnormal {index}/{len(cases)}","target":case["case_id"],"status":status,"message":result.get("message","")})
 if links_ready and len(operator_result_csvs)==159:
  deep_ok,deep_message=abnormal_adapter.merge_and_verify(operator_result_csvs,batch_root/"operator_abnormal_merged",normal_links); print(deep_message)
  if not deep_ok:
   abnormal_failed+=1; failures.append({"section":"abnormal deep verify","target":"159 operator cases","status":"FAIL","message":deep_message})
 elif links_ready:
  deep_message=f"MERGE_FAIL expected=159 actual={len(operator_result_csvs)}"; print(deep_message)
  abnormal_failed+=1; failures.append({"section":"abnormal merge","target":"159 operator cases","status":"FAIL","message":deep_message})
 abnormal_elapsed=time.perf_counter()-abnormal_started; abnormal_status="PASS" if abnormal_failed==0 else "FAIL"
 render_major_end(4,major_total,"abnormal",abnormal_status,abnormal_elapsed)
 summary.append({"section":"abnormal","status":abnormal_status,"completed":f"{abnormal_completed}/{len(cases)}","failed":abnormal_failed,"elapsed_seconds":f"{abnormal_elapsed:.3f}"})
 render_major_start(5,major_total,"fft-library-swap"); swap_started=time.perf_counter()
 swap_result=_execute_fft_library_swap(5,major_total,render_major=False); render_result(swap_result); swap_elapsed=time.perf_counter()-swap_started
 swap_code=EXIT_OK if swap_result.get("status")=="PASS" else EXIT_FAIL
 render_major_end(5,major_total,"fft-library-swap",swap_result.get("status","FAIL"),swap_elapsed); remember("fft-library-swap",swap_code,swap_result,swap_elapsed)
 elapsed=time.perf_counter()-batch_started
 return {"status":"PASS" if not failures else "FAIL","target":"batch-all","message":f"一键完整演示完成，失败={len(failures)}，wall_seconds={elapsed:.3f}","config":{"backend":request.backend,"execution":"CPU+GPU","input_mode":"default","abnormal_cases":len(cases),"wall_seconds":f"{elapsed:.3f}"},"batch_summary":summary,"batch_failures":failures}
def _run_batch(request:DemoRequest,caps:dict[str,Any])->dict[str,Any]:
 with CleanTemporaryDirectory() as batch_root:
  result=_run_batch_in_root(request,caps,batch_root)
 result["cleanup_verified"]=True
 return result

def _execute_single_operator_abnormal(request:DemoRequest,caps:dict[str,Any],case:dict[str,str])->dict[str,Any]:
 normal_request=DemoRequest(target="operator-one",backend=request.backend,dtype="FP32",module=case["module"],operator=case["operator"])
 profile,command=_operator_selection(normal_request,caps,case["module"],case["operator"])
 if not profile or not command: return {"status":"FAIL","target":"operator","message":"所选算子缺少本轮正常FP32前置路由"}
 with CleanTemporaryDirectory() as root:
  normal=operator_adapter.execute(command,profile,request.backend,1,root/"normal")
  if normal.get("status")!="PASS": return {"status":"FAIL","target":"operator","message":f"正常FP32前置失败：{normal.get('message','')}"}
  links=root/"normal_case_links.csv"
  ready,message=abnormal_adapter.resolve_normal_links(root/"normal",links,case["module"],case["operator"])
  if not ready: return {"status":"FAIL","target":"operator","message":message}
  result=abnormal_adapter.execute("operator",request.backend,case,links,root/"abnormal")
 result["cleanup_verified"]=True
 return result
def run_request(request:DemoRequest,caps:dict[str,Any]|None=None)->tuple[int,dict[str,Any]]:
 caps=caps or build_demo_capabilities(); runtime=json.loads((ROOT/"demo/config/runtime/demo_runtime.json").read_text(encoding="utf-8"))
 if request.runs!=1: return _result("INVALID_ARGUMENT",request,"现场演示固定正式执行一次，--runs必须为1")
 if request.target=="fft-library-swap":
  error=fft_library_swap_adapter.preflight()
  if error: return _result("UNSUPPORTED",request,error)
  result=_execute_fft_library_swap()
  return (EXIT_OK if result["status"]=="PASS" else EXIT_FAIL),result
 if request.backend not in caps["menu"]["backends"]: return _result("INVALID_ARGUMENT",request,"backend无效")
 backend_capability=next(x for x in caps["menu"]["backend_options"] if x["value"]==request.backend)
 if backend_capability["status"]!="READY": return _result("UNSUPPORTED",request,f"{request.backend}：{backend_capability['reason']}")
 if request.target in ("task1","task2"):
  scale,path=_task_input(request,caps)
  if not scale or not path: return _result("UNSUPPORTED" if request.input_mode=="default" else "INVALID_ARGUMENT",request,"输入未匹配已准入默认配置，或JSON没有为所选对象提供唯一scale")
  result=task_adapter.execute(request.target,request.backend,scale,path,1,runtime["warmup_runs"],caps["default_input_profile_backend"])
 elif request.target=="operator-one":
  if not request.module or not request.operator: return _result("INVALID_ARGUMENT",request,"单算子必须提供module和operator")
  profile,command=_operator_selection(request,caps,request.module,request.operator)
  if not profile or not command: return _result("UNSUPPORTED",request,"该operator/backend/dtype没有已准入best或真实命令")
  result=operator_adapter.execute(command,profile,request.backend,1)
 elif request.target=="operator-all":
  result=_execute_operator_all(request,caps)
 elif request.target=="abnormal":
  case=_abnormal_case(request,caps)
  if not case: return _result("INVALID_ARGUMENT",request,"异常选择未匹配真实catalog")
  kind=request.module if request.module in ("task1","task2") else "operator"
  result=_execute_single_operator_abnormal(request,caps,case) if kind=="operator" else abnormal_adapter.execute(kind,request.backend,case)
 elif request.target=="batch-all": result=_run_batch(request,caps)
 else: return _result("INVALID_ARGUMENT",request,"target无效")
 code={"PASS":EXIT_OK,"UNSUPPORTED":EXIT_UNSUPPORTED,"INVALID_ARGUMENT":EXIT_INVALID}.get(result["status"],EXIT_FAIL)
 return code,result
def parser()->argparse.ArgumentParser:
 cli_targets=("task1","task2","operator-one","operator-all","abnormal")
 p=argparse.ArgumentParser(description="ZKX全国总决赛统一演示系统"); p.add_argument("command",nargs="?",choices=cli_targets); p.add_argument("--target",choices=cli_targets)
 p.add_argument("--backend",choices=SUPPORTED_BACKENDS,default=DEFAULT_BACKEND); p.add_argument("--input-mode",choices=("default","json"),default="default")
 p.add_argument("--operator"); p.add_argument("--module"); p.add_argument("--abnormal-type"); p.add_argument("--runs",type=int,default=1); p.add_argument("--seed",type=int,default=0)
 p.add_argument("--dtype",choices=("FP32","FP16","INT32","INT16","INT8"),default="FP32"); p.add_argument("--input-json"); return p
def main(argv:list[str]|None=None)->int:
 raw_argv=list(sys.argv[1:] if argv is None else argv)
 args=parser().parse_args(raw_argv)
 if not args.command and not args.target:
  if raw_argv:
   parser().error("交互菜单只能通过不带参数的 ./demo/bin/demo.sh 进入；使用--backend等CLI参数时必须同时提供command或--target")
  from interactive_menu import main as menu_main
  return menu_main()
 target=args.target or args.command
 if args.target and args.command and args.target!=args.command: parser().error("command与--target冲突")
 request=DemoRequest(target=target,backend=args.backend,input_mode=args.input_mode,operator=args.operator,module=args.module,abnormal_type=args.abnormal_type,runs=args.runs,seed=args.seed,dtype=args.dtype,input_json=args.input_json)
 code,result=run_request(request); render_result(result); return code
if __name__=="__main__": raise SystemExit(main())
