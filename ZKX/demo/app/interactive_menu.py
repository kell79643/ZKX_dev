"""只收集菜单输入；所有执行均交给统一run_request。"""
from __future__ import annotations
from demo_capabilities import ROOT, build_demo_capabilities
from terminal_tables import render_result
def choose(title:str,values:list[str])->str:
 print(f"\n{title}")
 for i,value in enumerate(values,1): print(f"  {i}. {value}")
 while True:
  raw=input("请选择：").strip()
  if raw.isdigit() and 1<=int(raw)<=len(values): return values[int(raw)-1]
def main()->int:
 from unified_demo_runner import DemoRequest,run_request
 caps=build_demo_capabilities(); target_options=caps["menu"]["target_options"]
 target_labels=[x["label"] for x in target_options]+["exit"]
 selected_target=choose("ZKX统一演示",target_labels)
 if selected_target=="exit": return 0
 target=target_options[target_labels.index(selected_target)]["value"]
 if target=="fft-library-swap":
  code,result=run_request(DemoRequest(target=target),caps); render_result(result); return code
 backend_labels=[x["value"] for x in caps["menu"]["backend_options"]]
 selected_backend=choose("后端",backend_labels); backend_capability=caps["menu"]["backend_options"][backend_labels.index(selected_backend)]; backend=backend_capability["value"]
 if backend_capability["status"]!="READY":
  code,result=run_request(DemoRequest(target=target,backend=backend),caps); render_result(result); return code
 kwargs={"target":target,"backend":backend}
 if target in ("task1","task2"):
  mode=choose("输入",caps["menu"]["task_input_modes"])
  kwargs["input_mode"]=mode
  if mode=="json":
   example=ROOT/caps["menu"]["task_json_example"]
   print(f"示例JSON：{example.relative_to(ROOT)}")
   value=input("JSON路径（直接回车使用示例）：").strip()
   kwargs["input_json"]=value or str(example)
 elif target in ("operator-one","operator-all"):
  kwargs["dtype"]=choose("数据类型",caps["menu"]["dtypes"])
  if target=="operator-one":
   module=choose("模块",caps["modules"]); choices=[x["operator"] for x in caps["operators"] if x["module"]==module]
   kwargs.update(module=module,operator=choose("算子",choices))
   mode=choose("输入",caps["menu"]["input_modes"]); kwargs["input_mode"]=mode
   if mode=="json":
    example=ROOT/caps["menu"]["operator_json_example"]
    print(f"示例JSON：{example.relative_to(ROOT)}")
    value=input("JSON路径（直接回车使用示例；所选算子必须恰好包含一个scale）：").strip()
    kwargs["input_json"]=value or str(example)
 elif target=="abnormal":
  kind=choose("异常对象",["task1","task2","operator"]); kwargs["module"]=kind
  if kind=="operator":
   module=choose("算子模块",caps["modules"]); kwargs["module"]=module; kwargs["operator"]=choose("算子",[x["operator"] for x in caps["operators"] if x["module"]==module])
  catalog_target=kind if kind in ("task1","task2") else "operator"
  cases=caps["abnormal_cases"][catalog_target]
  if kind=="operator": cases=[x for x in cases if x["module"]==kwargs["module"] and x["operator"]==kwargs["operator"]]
  labels=[f"{x['abnormal_type']} — {x['case_id']}" for x in cases]
  selected=choose("异常类型（第1项为默认代表性异常）",labels)
  kwargs["case_id"]=cases[labels.index(selected)]["case_id"]
 code,result=run_request(DemoRequest(**kwargs),caps); render_result(result); return code
