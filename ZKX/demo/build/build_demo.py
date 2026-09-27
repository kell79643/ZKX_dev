#!/usr/bin/env python3
"""按审计计划构建并发布演示二进制；运行菜单绝不调用本文件。"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]; PLAN=ROOT/"demo/config/build/demo_build_plan.json"; BUILD=ROOT/"build/demo"
def write_build_status(backend: str, state: str, message: str) -> None:
    tree=BUILD/backend; tree.mkdir(parents=True,exist_ok=True)
    path=tree/"demo_build_status.json"; temporary=tree/"demo_build_status.json.tmp"
    payload={"schema_version":1,"backend":backend,"state":state,"message":message,
             "updated_at":datetime.now(timezone.utc).isoformat()}
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(temporary,path)
def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def run(command: list[str]) -> None:
    print("[DEMO][BUILD] "+" ".join(command),flush=True); subprocess.run(command,cwd=ROOT,check=True)
def load(path: Path) -> dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def main() -> int:
    plan=load(PLAN); supported_backends=tuple(plan["supported_backends"]); shared_backend=plan["shared_backend"]
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=supported_backends,required=True)
    p.add_argument("--git-commit"); p.add_argument("--git-dirty",choices=("true","false"))
    p.add_argument("--jobs",type=int,default=1); p.add_argument("--scope",choices=("all","tasks","operators","abnormal"),default="all")
    p.add_argument("--refresh-existing-build",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.git_commit and not re.fullmatch(r"[0-9a-f]{40}",a.git_commit): raise SystemExit("--git-commit若提供，必须是40位小写SHA")
    identity_status="provided" if a.git_commit else "unavailable"
    a.git_commit=a.git_commit or ("0"*40); a.git_dirty=a.git_dirty or "true"
    tree=BUILD/a.backend; manifest_path=tree/"demo_build_manifest.json"
    requested={"backend":a.backend,"git_commit":a.git_commit,"git_dirty":a.git_dirty,"plan_sha256":digest(PLAN)}
    if manifest_path.exists():
        old=load(manifest_path); actual={k:old.get(k) for k in requested}
        if actual!=requested: print(f"[DEMO][BUILD] existing_manifest=STALE action=refresh_after_success identity={actual}",flush=True)
    kinds={"tasks":"task","operators":"operator","abnormal":"abnormal"}
    groups=[]
    for group in plan["groups"]:
        if a.scope!="all" and group["kind"]!=kinds[a.scope]: continue
        logical=group["logical_backend"]
        if logical==a.backend or logical==shared_backend: groups.append(group)
    records=[]; libraries={}; library_contracts={}; configured_groups=[]
    build_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for group in groups:
        source=ROOT/group["source"]; work=BUILD/group["work"]
        options=list(group["cmake_options"])
        if group["kind"]=="task":
            for task in ("TASK1","TASK2"):
                options.extend([f"-D{task}_EVIDENCE_GIT_COMMIT={a.git_commit}",f"-D{task}_EVIDENCE_GIT_DIRTY={a.git_dirty}",f"-D{task}_EVIDENCE_TEST_TIME={build_time}"])
        run(["cmake","-S",str(source),"-B",str(work),*options])
        configured_groups.append({"build_group_id":group["build_group_id"],"logical_backend":group["logical_backend"],"source":group["source"],"work":group["work"],"cmake_options":options})
        for target in group["targets"]: run(["cmake","--build",str(work),"--target",target["cmake_target"],f"-j{a.jobs}"])
        for target in group["targets"]:
            internal=work/target["internal_binary"]; published=BUILD/target["published_binary"]
            if not internal.is_file(): raise RuntimeError(f"构建目标未产生预期文件：{internal}")
            published.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(internal,published); published.chmod(published.stat().st_mode|0o111)
            record={"build_group_id":group["build_group_id"],"kind":group["kind"],"logical_backend":group["logical_backend"],"cmake_source":group["source"],"cmake_target":target["cmake_target"],"published_path":str(published.relative_to(BUILD)).replace(os.sep,"/"),"sha256":digest(published),"file_size":published.stat().st_size}
            if "link_configuration" in target:
                record["link_configuration"]=target["link_configuration"]
            if "source" in target:
                source_path=ROOT/target["source"]
                if not source_path.is_file(): raise RuntimeError(f"动态库平替同源文件不存在：{source_path}")
                record.update(source=target["source"],source_sha256=digest(source_path))
            records.append(record)
        libdir=work/"lib"
        if libdir.is_dir():
            library_tree=BUILD/group["logical_backend"]
            for library in libdir.glob("*.so*"):
                destination=library_tree/"lib"/library.name; destination.parent.mkdir(parents=True,exist_ok=True)
                value=digest(library); key=f"{group['logical_backend']}/{library.name}"
                contract={"logical_backend":group["logical_backend"],"configured_backend":group["configured_backend"],"cmake_options":group["cmake_options"]}
                if key in libraries:
                    if library_contracts[key]!=contract:
                        raise RuntimeError(f"同名运行库构建契约冲突：{key} canonical={library_contracts[key]} candidate={contract}")
                    if libraries[key]["sha256"]!=value:
                        print(f"[DEMO][BUILD] library={key} action=reuse_canonical canonical_sha256={libraries[key]['sha256']} candidate_sha256={value}",flush=True)
                    continue
                if destination.exists() and digest(destination)!=value:
                    print(f"[DEMO][BUILD] library={key} action=replace_stale_published_copy",flush=True)
                shutil.copy2(library,destination)
                libraries[key]={"published_path":str(destination.relative_to(BUILD)).replace(os.sep,"/"),"sha256":value,"file_size":library.stat().st_size,"logical_backend":group["logical_backend"],"canonical_build_group_id":group["build_group_id"],"configured_backend":group["configured_backend"],"cmake_options":group["cmake_options"]}
                library_contracts[key]=contract
    manifest={"schema_version":1,**requested,"git_identity_status":identity_status,"cmake_version":subprocess.check_output(["cmake","--version"],text=True).splitlines()[0],"generated_at":datetime.now(timezone.utc).isoformat(),"scope":a.scope,"configured_groups":configured_groups,"artifacts":records,"libraries":list(libraries.values())}
    tree.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"[DEMO][BUILD] backend={a.backend} artifacts={len(records)} manifest={manifest_path} status=PASS")
    return 0
if __name__=="__main__":
    status_parser=argparse.ArgumentParser(add_help=False)
    status_parser.add_argument("--backend",required=True)
    status_args,_=status_parser.parse_known_args()
    write_build_status(status_args.backend,"BUILDING","构建与发布尚未完成")
    try:
        exit_code=main()
    except BaseException as error:
        write_build_status(
            status_args.backend,
            "FAIL",
            f"{type(error).__name__}: {error}")
        raise
    if exit_code==0:
        write_build_status(status_args.backend,"PASS","构建与发布完整完成")
    else:
        write_build_status(status_args.backend,"FAIL",f"构建返回退出码{exit_code}")
    raise SystemExit(exit_code)
