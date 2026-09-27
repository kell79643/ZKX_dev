"""统一演示终端表格。只消费内存结果，不实现业务判断。"""
from __future__ import annotations
from typing import Any, Iterable
from unicodedata import east_asian_width
MAJOR_LINE="="*72
MAJOR_END_LINE="-"*72
MINOR_LINE="."*72
FFT_CASE_LABELS={
    "c2c_1d_forward_batch":"一维复数正变换（批量）",
    "c2c_1d_inverse":"一维复数逆变换",
    "c2c_1d_inplace":"一维复数原位变换",
    "r2c_1d_batch":"一维实数转复数（批量）",
    "c2r_1d_batch":"一维复数转实数（批量）",
    "c2c_2d_forward":"二维复数正变换",
    "c2c_2d_inverse":"二维复数逆变换",
    "TOTAL (7 cases)":"七项业务合计",
}
def _scope_label(value: Any) -> str:
    scope=str(value)
    if scope.startswith("cross_library."):
        case=scope.split(".",1)[1]
        return "双库输出对比·"+FFT_CASE_LABELS.get(case,case)
    return FFT_CASE_LABELS.get(scope,scope)
def _resource_label(value: Any) -> str:
    return {
        "dlfft_version":"官方libdlfft.so版本",
        "fft_thrust_version":"自研libzkx_fft_thrust.so版本",
    }.get(str(value),str(value))
def _display_width(value: str) -> int:
    return sum(2 if east_asian_width(character) in ("W","F") else 1 for character in value)
def _pad(value: str, width: int) -> str:
    return value+" "*(width-_display_width(value))
def _table(title: str, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    values=[[str(x) for x in row] for row in rows]; widths=[_display_width(x) for x in headers]
    for row in values: widths=[max(widths[i],_display_width(row[i])) for i in range(len(headers))]
    line="+"+"+".join("-"*(x+2) for x in widths)+"+"; print(f"\n{title}\n{line}")
    print("| "+" | ".join(_pad(headers[i],widths[i]) for i in range(len(headers)))+" |\n"+line)
    for row in values: print("| "+" | ".join(_pad(row[i],widths[i]) for i in range(len(headers)))+" |")
    print(line)
def render_result(result: dict[str, Any]) -> None:
    print(f"\n[DEMO] status={result.get('status','UNKNOWN')} target={result.get('target','-')} message={result.get('message','')}")
    target=str(result.get("target","")); operator_result="." in target or target=="operator-all"
    library_swap_result=target.startswith("fft-library-swap")
    if result.get("config"):
        if operator_result: print("[DEMO][CONFIG] "+" ".join(f"{key}={value}" for key,value in sorted(result["config"].items())))
        else: _table("配置",["参数","值"],sorted(result["config"].items()))
    if result.get("timing"):
        if library_swap_result:
            _table("性能",["测试项目","CPU参考耗时(ms)","动态库耗时(ms)","状态"],
                ([_scope_label(x.get("scope","total")),x.get("cpu_ms","-"),x.get("gpu_ms","-"),"INFO"] for x in result["timing"]))
        elif operator_result:
            performance=[["timing."+x.get("scope","total"),f"CPU={x.get('cpu_ms','-')}ms GPU={x.get('gpu_ms','-')}ms speedup={x.get('speedup','-')}","INFO"] for x in result["timing"]]
            performance += [[x.get("metric","-"),x.get("value","-"),x.get("status","-")] for x in result.get("resources",[])]
            _table("性能与资源",["指标","值","状态"],performance)
        else: _table("耗时",["范围","CPU ms","GPU ms","加速比"],([x.get("scope","total"),x.get("cpu_ms","-"),x.get("gpu_ms","-"),x.get("speedup","-")] for x in result["timing"]))
    if result.get("accuracy"):
        accuracy_header="测试项目" if library_swap_result else "范围"
        _table("精度",[accuracy_header,"MSE","RMSE","相对L2","相对L∞","离散匹配","状态"],([_scope_label(x.get("scope","total")),"NA" if x.get("mse") is None else x.get("mse"),"NA" if x.get("rmse") is None else x.get("rmse"),"NA" if x.get("relative_l2") is None else x.get("relative_l2"),"NA" if x.get("relative_linf") is None else x.get("relative_linf"),x.get("discrete","-"),x.get("status","-")] for x in result["accuracy"]))
    if result.get("resources") and not operator_result: _table("资源",["指标","值","状态"],([_resource_label(x.get("metric","-")),x.get("value","-"),x.get("status","-")] for x in result["resources"]))
    if result.get("library_swap_cases"): _table("动态库平替对照",["测试项目","官方库耗时(ms)","自研库耗时(ms)","自研库加速比","MSE","RMSE","相对L2","相对L∞","状态"],([_scope_label(x.get("case","-")),x.get("dlfft_ms","-"),x.get("fft_thrust_ms","-"),x.get("speedup","-"),x.get("mse","-"),x.get("rmse","-"),x.get("relative_l2","-"),x.get("relative_linf","-"),x.get("status","-")] for x in result["library_swap_cases"]))
    if result.get("batch_summary"): _table("一键完整演示汇总",["区段","状态","完成","失败","耗时 s"],([x.get("section","-"),x.get("status","-"),x.get("completed","-"),x.get("failed","-"),x.get("elapsed_seconds","-")] for x in result["batch_summary"]))
    if result.get("batch_failures"): _table("失败项",["区段","对象","状态","说明"],([x.get("section","-"),x.get("target","-"),x.get("status","-"),x.get("message","-")] for x in result["batch_failures"]))
    for warning in result.get("warnings",[]): print(f"[{warning.get('level','WARNING')}] {warning.get('message','')}")
    if result.get("details"): print("\n运行详情\n"+str(result["details"]))
    if result.get("cleanup_verified"): print("[DEMO] temporary_results=DELETED")

def render_major_start(index:int,total:int,title:str)->None:
    print(f"\n\n\n{MAJOR_LINE}\n[ {index}/{total} ] {title} — 开始\n{MAJOR_LINE}")

def render_major_end(index:int,total:int,title:str,status:str,elapsed_seconds:float)->None:
    print(f"\n{MAJOR_END_LINE}\n[ {index}/{total} ] {title} — 完成，状态={status}，耗时={elapsed_seconds:.3f}s\n{MAJOR_END_LINE}\n\n")

def render_minor_start(major_index:int,major_total:int,index:int,total:int,title:str)->None:
    print(f"\n\n{MINOR_LINE}\n[ {major_index}/{major_total} · {index}/{total} ] {title} — 开始\n{MINOR_LINE}")

def render_minor_end(major_index:int,major_total:int,index:int,total:int,title:str,status:str,elapsed_seconds:float,details:str="")->None:
    detail_text=f"，{details}" if details else ""
    print(f"\n{MINOR_LINE}\n[ {major_index}/{major_total} · {index}/{total} ] {title} — 完成，状态={status}{detail_text}，耗时={elapsed_seconds:.3f}s\n{MINOR_LINE}")
