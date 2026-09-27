"""演示适配器共用的进程、临时目录与结果读取工具。"""
from __future__ import annotations
import csv, hashlib, json, math, shutil, subprocess, tempfile, time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[2]; BUILD_ROOT = ROOT / "build/demo"
def load_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def binary_path(tree: str, relative: str) -> Path: return BUILD_ROOT / tree / relative
def run_process(command: list[str], timeout: int = 300) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "wall_ms": (time.perf_counter()-started)*1000, "command": command, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"returncode": 124, "stdout": error.stdout or "", "stderr": error.stderr or "", "wall_ms": (time.perf_counter()-started)*1000, "command": command, "timed_out": True}
def rows_from_csvs(directory: Path) -> list[dict[str, str]]:
    rows=[]
    for path in sorted(directory.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream): row["_source"]=path.name; rows.append(row)
        except (UnicodeDecodeError, csv.Error): pass
    return rows
def first_number(row: dict[str, str], names: list[str]) -> float | None:
    for name in names:
        value=row.get(name)
        if value not in (None, ""):
            try:
                number=float(value)
                if math.isfinite(number): return number
            except ValueError: pass
    return None
class CleanTemporaryDirectory:
    def __enter__(self) -> Path: self.path=Path(tempfile.mkdtemp(prefix="zkx_demo_")); return self.path
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        shutil.rmtree(self.path, ignore_errors=False)
        if self.path.exists(): raise RuntimeError(f"临时结果自动删除失败：{self.path}")
def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def verify_published_binary(selected_backend: str, tree: str, relative: str) -> tuple[Path | None, str | None]:
    """按所选后端manifest核验发布文件；支持manifest内共享not_applicable树。"""
    manifest_path=BUILD_ROOT/selected_backend/"demo_build_manifest.json"
    if not manifest_path.is_file(): return None,f"构建清单不存在：{manifest_path}"
    try: manifest=load_json(manifest_path)
    except (OSError,ValueError): return None,"demo_build_manifest.json无法解析"
    published=f"{tree}/{relative}"; matches=[x for x in manifest.get("artifacts",[]) if x.get("published_path")==published]
    if len(matches)!=1: return None,f"构建清单未唯一登记：{published}"
    path=BUILD_ROOT/published
    if not path.is_file(): return None,f"可执行文件不存在：{path}"
    entry=matches[0]
    # SHA/大小只用于开发追溯，不是现场业务门禁。真实后端身份与业务输出由各适配器核验。
    return path,None
