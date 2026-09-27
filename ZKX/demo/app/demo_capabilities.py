"""从仓库事实源生成统一演示能力；本模块不包含业务执行逻辑。"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "demo/config/runtime/demo_profile.json"
BUILD_PLAN = ROOT / "demo/config/build/demo_build_plan.json"
RUNTIME = ROOT / "demo/config/runtime/demo_runtime.json"
COMMANDS = ROOT / "demo/config/runtime/operator_demo_commands.json"
TASK_JSON_EXAMPLE = ROOT / "demo/config/examples/task_demo_input_example.json"
OPERATOR_JSON_EXAMPLE = ROOT / "demo/config/examples/operator_demo_input_example.json"
TASK_SCALES = ROOT / "test_all/config/task_benchmarks/task_input_scales.json"
REGISTRY = ROOT / "cusignal_cpp/include/cusignal/runtime/operator_abnormal_validation.h"
ABNORMAL = {
    "operator": ROOT / "test_all/operators/abnormal/operator_abnormal_case_catalog.csv",
    "task1": ROOT / "test_all/tasks/abnormal/task1/task1_abnormal_case_catalog.csv",
    "task2": ROOT / "test_all/tasks/abnormal/task2/task2_abnormal_case_catalog.csv",
}

def _backend_options(entries: list[dict[str, Any]], operators: list[dict[str,str]], plan: dict[str,Any], runtime: dict[str,Any]) -> list[dict[str,str]]:
    result=[]
    supported=plan["supported_backends"]; profile_backend=runtime["default_input_profile_backend"]; shared_backend=plan["shared_backend"]
    for backend in supported:
        task_targets={x.get("target") for x in entries if x.get("target_kind")=="task" and x.get("backend")==profile_backend}
        operator_keys={(x.get("module"),x.get("target"),x.get("dtype")) for x in entries if x.get("target_kind")=="operator" and x.get("backend") in (profile_backend,shared_backend)}
        expected={(x["module"],x["operator"],dtype) for x in operators for dtype in ("FP32","FP16","INT32","INT16","INT8")}
        manifest=ROOT/f"build/demo/{backend}/demo_build_manifest.json"
        expected_artifacts=sum(len(group.get("targets",[])) for group in plan["groups"] if group.get("logical_backend") in (backend,"not_applicable"))
        build_ready=False
        if manifest.is_file():
            try:
                payload=_json(manifest); build_ready=payload.get("scope")=="all" and len(payload.get("artifacts",[]))==expected_artifacts
            except RuntimeError: build_ready=False
        missing=[]
        if task_targets!={"Task1","Task2"}: missing.append("Task shared default input profiles")
        if operator_keys!=expected: missing.append("53x5 operator shared default input profiles")
        if not build_ready: missing.append(f"{expected_artifacts}-file build manifest")
        result.append({"value":backend,"status":"READY" if not missing else "UNSUPPORTED","reason":"shared default input profiles and backend build manifest are complete" if not missing else "missing: "+", ".join(missing)})
    return result

def _json(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise RuntimeError(f"能力事实源不存在：{path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise RuntimeError(f"能力事实源不是JSON对象：{path.relative_to(ROOT)}")
    return value

def _operators() -> list[dict[str, str]]:
    text = REGISTRY.read_text(encoding="utf-8")
    start = text.find("kOperatorAbnormalRegistry"); end = text.find("}};", start)
    pairs = re.findall(r'\{"([a-z0-9_]+)","([a-z0-9_]+)"\}', text[start:end])
    if len(pairs) != 53 or len(set(pairs)) != 53: raise RuntimeError(f"统一算子注册表异常：expected=53 actual={len(pairs)}")
    return [{"module": m, "operator": o} for m, o in pairs]

def _abnormal_cases() -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for target, path in ABNORMAL.items():
        with path.open("r", encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
        if target == "operator":
            result[target] = [{"module": r["module_name"], "operator": r["operator_name"], "case_id": r["abnormal_case_id"], "abnormal_type": r["abnormal_type"]} for r in rows]
        else: result[target] = [{"case_id": r["case_id"], "abnormal_type": r["abnormal_type"]} for r in rows]
    return result

def build_demo_capabilities() -> dict[str, Any]:
    profile = _json(PROFILE)
    plan = _json(BUILD_PLAN); runtime = _json(RUNTIME)
    supported=plan.get("supported_backends",[]); profile_backend=runtime.get("default_input_profile_backend")
    if not supported or profile_backend not in supported: raise RuntimeError("演示后端配置无效：default_input_profile_backend必须属于supported_backends")
    if profile.get("schema_version") != 2 or not isinstance(profile.get("entries"), list): raise RuntimeError("demo_profile.json必须是schema_version=2")
    operators = _operators(); commands = _json(COMMANDS); command_entries = commands.get("entries", [])
    if commands.get("schema_version") != 2: raise RuntimeError("operator_demo_commands.json必须是schema_version=2")
    if {(x["module"], x["operator"]) for x in command_entries} != {(x["module"], x["operator"]) for x in operators}: raise RuntimeError("operator_demo_commands.json与真实53算子注册表不一致")
    _json(TASK_SCALES)
    example_tasks = _json(TASK_JSON_EXAMPLE).get("tasks", {})
    if any(len(example_tasks.get(task, {}).get("scales", [])) != 1 for task in ("Task1", "Task2")):
        raise RuntimeError("task_demo_input_example.json必须为Task1和Task2各提供一个scale")
    operator_example = _json(OPERATOR_JSON_EXAMPLE).get("operators", [])
    if len(operator_example) != 1 or len(operator_example[0].get("scales", [])) != 1:
        raise RuntimeError("operator_demo_input_example.json必须恰好提供一个算子的一个scale")
    example_identity=(operator_example[0].get("module_name"),operator_example[0].get("operator_name"))
    if example_identity not in {(x["module"],x["operator"]) for x in operators}:
        raise RuntimeError("operator_demo_input_example.json的算子不在真实53算子注册表")
    abnormal = _abnormal_cases()
    abnormal_count=sum(len(rows) for rows in abnormal.values())
    backend_options=_backend_options(profile["entries"],operators,plan,runtime)
    swap_groups=[group for group in plan["groups"] if group.get("kind")=="fft_library_swap"]
    swap_ready=len(swap_groups)==1 and len(swap_groups[0].get("targets",[]))==2
    swap_reason="构建计划已登记同源双库平替目标" if swap_ready else "构建计划缺少唯一的同源双库平替目标"
    return {"schema_version": 2, "status": "READY", "operators": operators,
        "modules": sorted({x["module"] for x in operators}), "operator_commands": command_entries,
        "best_profiles": profile["entries"], "abnormal_cases": abnormal,
        "build_plan": plan, "default_input_profile_backend":profile_backend, "shared_backend":plan["shared_backend"],
        "fft_library_swap":{"status":"READY" if swap_ready else "UNSUPPORTED","reason":swap_reason},
        "batch": {"status": "READY" if abnormal_count==171 and swap_ready else "UNSUPPORTED",
        "reason": "all 171 abnormal cases and fft-library-swap artifacts are available" if abnormal_count==171 and swap_ready else f"expected 171 abnormal cases (actual {abnormal_count}) and ready fft-library-swap ({swap_ready})",
        "operator_dtypes": ["INT8", "INT16", "INT32", "FP16", "FP32"], "abnormal_case_count": abnormal_count},
        "menu": {"targets": ["task1", "task2", "operator-one", "operator-all", "abnormal", "fft-library-swap", "batch-all"],
        "target_options": [{"value":"task1","label":"task1"},{"value":"task2","label":"task2"},{"value":"operator-one","label":"operator-one"},{"value":"operator-all","label":"operator-all"},{"value":"abnormal","label":"abnormal"},{"value":"fft-library-swap","label":"fft-library-swap（libzkx_fft_thrust.so平替libdlfft.so）"},{"value":"batch-all","label":"batch-all（一键完整演示）"}],
        "backends": [x["value"] for x in backend_options], "backend_options": backend_options,
        "input_modes": ["default", "json"], "task_input_modes": ["default", "json"],
        "task_json_example": str(TASK_JSON_EXAMPLE.relative_to(ROOT)).replace("\\", "/"),
        "operator_json_example": str(OPERATOR_JSON_EXAMPLE.relative_to(ROOT)).replace("\\", "/"),
        "dtypes": ["FP32", "FP16", "INT32", "INT16", "INT8"],
        "abnormal_types": sorted({x["abnormal_type"] for rows in abnormal.values() for x in rows})}}
