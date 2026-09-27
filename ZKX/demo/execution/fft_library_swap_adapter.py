"""同一测试源码分别链接libdlfft.so与libzkx_fft_thrust.so的适配器。"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from .common import BUILD_ROOT, ROOT, load_json, run_process, sha256, verify_published_binary

VARIANTS={
    "dlfft":{"binary":"bin/fft_library_swap_dlfft","identity":"DLFFT","library":"libdlfft.so"},
    "zkx_fft_thrust":{"binary":"bin/fft_library_swap_zkx_fft_thrust","identity":"FFT_THRUST","library":"libzkx_fft_thrust.so"},
}
ABSOLUTE_TOLERANCE=2.0e-4
RELATIVE_TOLERANCE=2.0e-4
EXPECTED_CASES=(
    "c2c_1d_forward_batch",
    "c2c_1d_inverse",
    "c2c_1d_inplace",
    "r2c_1d_batch",
    "c2r_1d_batch",
    "c2c_2d_forward",
    "c2c_2d_inverse",
)

def _manifest_entries()->tuple[dict[str,dict[str,Any]],str|None]:
    path=BUILD_ROOT/"fft_thrust/demo_build_manifest.json"
    if not path.is_file(): return {},f"构建清单不存在：{path}"
    try: manifest=load_json(path)
    except (OSError,ValueError): return {},"fft_thrust demo_build_manifest.json无法解析"
    rows={x.get("link_configuration","").split("=",1)[-1]:x for x in manifest.get("artifacts",[]) if x.get("kind")=="fft_library_swap"}
    if set(rows)!=set(VARIANTS): return {},"构建清单没有唯一登记两种动态库平替二进制"
    source=ROOT/rows["dlfft"].get("source","")
    source_hashes={x.get("source_sha256") for x in rows.values()}
    if len(source_hashes)!=1 or not source.is_file() or sha256(source) not in source_hashes:
        return {},"两种二进制未由当前同一测试源码生成"
    for name,row in rows.items():
        if row.get("link_configuration")!=f"FFT_LIBRARY={name}": return {},f"链接配置不符合仅切换库名契约：{name}"
    return rows,None

def preflight()->str|None:
    _,error=_manifest_entries()
    if error: return error
    for name,entry in VARIANTS.items():
        _,error=verify_published_binary("fft_thrust","fft_library_swap",entry["binary"])
        if error: return f"{name}：{error}"
    return None

def _tokens(line:str)->dict[str,str]:
    result={}
    for token in line.split("|")[1:]:
        if "=" in token:
            key,value=token.split("=",1); result[key]=value
    return result

def _number(value:str)->float:
    number=float(value)
    if not math.isfinite(number): raise ValueError("non-finite number")
    return number

def _output(value:str)->list[complex]:
    if not value: return []
    result=[]
    for item in value.split(";"):
        real,imaginary=item.split(",",1); result.append(complex(_number(real),_number(imaginary)))
    return result

def execute_variant(name:str)->dict[str,Any]:
    if name not in VARIANTS: return {"status":"INVALID_ARGUMENT","target":"fft-library-swap","message":"未知动态库变体"}
    manifest,error=_manifest_entries()
    if error: return {"status":"UNSUPPORTED","target":f"fft-library-swap.{name}","message":error}
    entry=VARIANTS[name]; binary,error=verify_published_binary("fft_thrust","fft_library_swap",entry["binary"])
    if error or binary is None: return {"status":"UNSUPPORTED","target":f"fft-library-swap.{name}","message":error or "二进制不可用"}
    process=run_process([str(binary)],timeout=120)
    identity=None; cases=[]; summary=None
    try:
        for line in process["stdout"].splitlines():
            if line.startswith("[FFT_SWAP][IDENTITY]"): identity=_tokens(line)
            elif line.startswith("[FFT_SWAP][CASE]"):
                row=_tokens(line); row["gpu_ms"]=_number(row["gpu_ms"]); row["cpu_reference_ms"]=_number(row["cpu_reference_ms"])
                for metric in ("mse","rmse","relative_l2","relative_linf"): row[metric]=_number(row[metric])
                row["decoded_output"]=_output(row.pop("output")); cases.append(row)
            elif line.startswith("[FFT_SWAP][SUMMARY]"): summary=_tokens(line)
    except (KeyError,ValueError) as error:
        return {"status":"FAIL","target":f"fft-library-swap.{name}","message":f"测试输出无法完整解析：{error}",
            "_cases":cases,"_returncode":process["returncode"],"_timed_out":process["timed_out"],"_wall_ms":process["wall_ms"],
            "details":_process_details(process)}
    loaded_name=Path(identity.get("library_path","")).name if identity else ""
    expected_path=entry["library"]
    valid_identity=identity is not None and identity.get("backend_identity")==entry["identity"] and (loaded_name==expected_path or loaded_name.startswith(expected_path+".")) and identity.get("version_status")=="CUFFT_SUCCESS" and identity.get("source_contract")=="identical_source_only_library_name_changes"
    try: valid_version=identity is not None and int(identity.get("version","0"))>0
    except ValueError: valid_version=False
    cases_ok=len(cases)==len(EXPECTED_CASES) and {x["name"] for x in cases}==set(EXPECTED_CASES) and all(x.get("status")=="PASS" and x["decoded_output"] and x["gpu_ms"]>0 for x in cases)
    passed=process["returncode"]==0 and not process["timed_out"] and valid_identity and valid_version and cases_ok and summary is not None and summary.get("status")=="PASS"
    return {"status":"PASS" if passed else "FAIL","target":f"fft-library-swap.{name}",
        "message":"同源测试执行成功" if passed else "动态库身份、版本、输出或业务状态核验失败",
        "config":{"link_configuration":manifest[name]["link_configuration"],"backend_identity":identity.get("backend_identity","-") if identity else "-","loaded_library":identity.get("library_path","-") if identity else "-","cufftGetVersion_status":identity.get("version_status","-") if identity else "-","version":identity.get("version","-") if identity else "-","warmup_runs":1,"measured_runs":1,"input":"deterministic identical input; C2C/R2C/C2R; 1D/2D/batch/in-place"},
        "timing":[{"scope":x["name"],"cpu_ms":f"{x['cpu_reference_ms']:.6f}","gpu_ms":f"{x['gpu_ms']:.6f}","speedup":"-"} for x in cases],
        "accuracy":[{"scope":x["name"],"mse":f"{x['mse']:.9e}","rmse":f"{x['rmse']:.9e}","relative_l2":f"{x['relative_l2']:.9e}","relative_linf":f"{x['relative_linf']:.9e}","discrete":"-","status":x["status"]} for x in cases],
        "_cases":cases,"_returncode":process["returncode"],"_timed_out":process["timed_out"],"_wall_ms":process["wall_ms"],
        "details":_process_details(process) if not passed else ""}

def _process_details(process:dict[str,Any])->str:
    stdout=process.get("stdout","").strip(); stderr=process.get("stderr","").strip()
    parts=[f"exit_code={process.get('returncode','-')} timed_out={str(process.get('timed_out',False)).lower()} wall_ms={process.get('wall_ms','-')}"]
    if stderr: parts.append("stderr:\n"+stderr)
    if stdout: parts.append("stdout:\n"+stdout)
    return "\n".join(parts)

def _metrics(actual:list[complex],reference:list[complex])->dict[str,float]:
    if not actual or len(actual)!=len(reference): return {x:math.inf for x in ("mse","rmse","relative_l2","relative_linf")}
    errors=[abs(a-b) for a,b in zip(actual,reference)]; squared=sum(x*x for x in errors)
    reference_squared=sum(abs(x)**2 for x in reference)
    mse=squared/len(errors)
    return {"mse":mse,"rmse":math.sqrt(mse),"relative_l2":math.sqrt(squared)/max(math.sqrt(reference_squared),1.0e-30),"relative_linf":max(errors)/max(max(abs(x) for x in reference),1.0e-30)}

def compare(dlfft:dict[str,Any],fft_thrust:dict[str,Any])->dict[str,Any]:
    left={x["name"]:x for x in dlfft.get("_cases",[])}; right={x["name"]:x for x in fft_thrust.get("_cases",[])}
    rows=[]; accuracy=[]; dlfft_total=0.0; thrust_total=0.0
    passed=dlfft.get("status")=="PASS" and fft_thrust.get("status")=="PASS" and set(left)==set(right)==set(EXPECTED_CASES)
    for name in EXPECTED_CASES:
        if name not in left or name not in right:
            passed=False
            rows.append({"case":name,"dlfft_ms":f"{left[name]['gpu_ms']:.6f}" if name in left else "N/A",
                "fft_thrust_ms":f"{right[name]['gpu_ms']:.6f}" if name in right else "N/A","speedup":"N/A",
                "mse":"N/A","rmse":"N/A","relative_l2":"N/A","relative_linf":"N/A","status":"FAIL"})
            continue
        metrics=_metrics(right[name]["decoded_output"],left[name]["decoded_output"])
        dlfft_ms=left[name]["gpu_ms"]; thrust_ms=right[name]["gpu_ms"]
        timing_ok=math.isfinite(dlfft_ms) and math.isfinite(thrust_ms) and dlfft_ms>0 and thrust_ms>0
        row_pass=timing_ok and left[name].get("shape")==right[name].get("shape") and all(math.isfinite(x) for x in metrics.values()) and metrics["rmse"]<=ABSOLUTE_TOLERANCE and metrics["relative_l2"]<=RELATIVE_TOLERANCE and metrics["relative_linf"]<=RELATIVE_TOLERANCE
        passed=passed and row_pass
        speedup=dlfft_ms/thrust_ms if timing_ok else None
        if timing_ok: dlfft_total+=dlfft_ms; thrust_total+=thrust_ms
        rows.append({"case":name,"dlfft_ms":f"{dlfft_ms:.6f}" if math.isfinite(dlfft_ms) else "N/A",
            "fft_thrust_ms":f"{thrust_ms:.6f}" if math.isfinite(thrust_ms) else "N/A",
            "speedup":f"{speedup:.3f}" if speedup is not None else "N/A","status":"PASS" if row_pass else "FAIL",**{key:f"{value:.9e}" for key,value in metrics.items()}})
        accuracy.append({"scope":f"cross_library.{name}","mse":f"{metrics['mse']:.9e}","rmse":f"{metrics['rmse']:.9e}","relative_l2":f"{metrics['relative_l2']:.9e}","relative_linf":f"{metrics['relative_linf']:.9e}","discrete":"-","status":"PASS" if row_pass else "FAIL"})
    total_speedup=dlfft_total/thrust_total if dlfft_total>0 and thrust_total>0 else None
    rows.append({"case":"TOTAL (7 cases)","dlfft_ms":f"{dlfft_total:.6f}" if dlfft_total>0 else "N/A",
        "fft_thrust_ms":f"{thrust_total:.6f}" if thrust_total>0 else "N/A",
        "speedup":f"{total_speedup:.3f}" if total_speedup is not None else "N/A",
        "mse":"-","rmse":"-","relative_l2":"-","relative_linf":"-","status":"PASS" if passed else "FAIL"})
    diagnostics=(
        f"dlfft: status={dlfft.get('status','-')} exit_code={dlfft.get('_returncode','-')} "
        f"timed_out={str(dlfft.get('_timed_out',False)).lower()} message={dlfft.get('message','-')}\n"
        f"fft_thrust: status={fft_thrust.get('status','-')} exit_code={fft_thrust.get('_returncode','-')} "
        f"timed_out={str(fft_thrust.get('_timed_out',False)).lower()} message={fft_thrust.get('message','-')}"
    )
    return {"status":"PASS" if passed else "FAIL","target":"fft-library-swap","message":"相同源码仅替换链接库名，两端业务结果一致" if passed else "动态库平替对照失败",
        "config":{"source_status":"unavailable_in_formal_whitelist","link_dlfft":"FFT_LIBRARY=dlfft","link_fft_thrust":"FFT_LIBRARY=zkx_fft_thrust","same_input":"YES","speedup_definition":"libdlfft_ms / libzkx_fft_thrust_ms (>1表示自研库更快)","tolerance_absolute":ABSOLUTE_TOLERANCE,"tolerance_relative":RELATIVE_TOLERANCE},
        "accuracy":accuracy,"library_swap_cases":rows,
        "resources":[{"metric":"dlfft_version","value":dlfft.get("config",{}).get("version","-"),"status":dlfft.get("config",{}).get("cufftGetVersion_status","-")},{"metric":"fft_thrust_version","value":fft_thrust.get("config",{}).get("version","-"),"status":fft_thrust.get("config",{}).get("cufftGetVersion_status","-")}],
        "details":"两端版本号只要求为合法正整数，不要求数值相同。\n"+diagnostics}
