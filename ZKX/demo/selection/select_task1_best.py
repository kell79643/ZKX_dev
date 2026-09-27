#!/usr/bin/env python3
"""从完整Task1正式候选集追加唯一best entry并输出完整排名证据。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FORMAL_COMMIT = "d186f2993f4a5211c7662571e948fb6b6d4e8aab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_scales(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scales = [item["scale_id"] for item in payload["tasks"]["Task1"]["scales"]]
    require(len(scales) == 110 and len(set(scales)) == 110, "Task1预定规模必须恰为110个")
    return scales


def verify_memory(path: Path) -> None:
    identity = dict(
        line.split("=", 1)
        for line in (path / "evidence_identity.txt").read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    require(identity.get("git_commit") == FORMAL_COMMIT, "资源证据commit不匹配")
    require(identity.get("git_dirty") == "false", "资源证据git_dirty不是false")
    require(identity.get("backend") == "fft_thrust", "资源证据backend不匹配")
    require(identity.get("warmup_runs") == "1024", "资源warmup次数不匹配")
    require(identity.get("measured_runs") == "1024", "资源measured次数不匹配")
    require((path / "preflight_processes.txt").read_bytes() == b"", "clean preflight存在竞争任务进程")
    require("No running processes found" in (path / "preflight_dlsmi.txt").read_text(encoding="utf-8"), "clean preflight GPU非空闲")
    require((path / "exit_status.txt").read_text(encoding="utf-8").strip() == "exit_code=0", "资源证据退出码非0")
    log = (path / "task1_memory_stability.log").read_text(encoding="utf-8")
    summaries = [line for line in log.splitlines() if "[TASK][MEMORY]" in line]
    require(len(summaries) == 1, "资源日志最终汇总数量异常")
    for marker in ("iterations=1024", "allocator_status=pass", "cpu_status=pass", "gpu_status=pass", "status=pass"):
        require(marker in summaries[0], f"资源日志缺少 {marker}")
    for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, recorded = line.split(maxsplit=1)
        candidate = path / Path(recorded.strip()).name
        require(candidate.is_file() and sha256(candidate) == expected, f"资源证据SHA不匹配：{candidate.name}")


def load_candidate(formal_root: Path, source_prefix: Path, scale_id: str) -> dict[str, Any]:
    relative = Path("cases") / scale_id / "task1_benchmark_fft_thrust_formal.csv"
    path = formal_root / relative
    require(path.is_file(), f"缺少正式CSV：{scale_id}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2, f"CPU/GPU行数异常：{scale_id}")
    by_device = {row["device"]: row for row in rows}
    require(set(by_device) == {"cpu", "gpu"}, f"CPU/GPU配对不完整：{scale_id}")
    cpu, gpu = by_device["cpu"], by_device["gpu"]
    for row in rows:
        require(row["scale_id"] == scale_id, f"scale_id不一致：{scale_id}")
        require(row["backend"] == "fft_thrust", f"backend不一致：{scale_id}")
        require(row["dtype"] == "ComplexFP32", f"dtype不一致：{scale_id}")
        require(row["run_type"] == "formal", f"非正式候选：{scale_id}")
        require(row["git_commit"] == FORMAL_COMMIT and row["git_dirty"] == "false", f"身份不一致：{scale_id}")
        require(row["status"] == "PASS" and row["accuracy_status"] == "PASS", f"状态未通过：{scale_id}")
        require(row["return_code"] == "0", f"返回码非0：{scale_id}")
    require(cpu["run_id"] == gpu["run_id"] and cpu["case_id"] == gpu["case_id"], f"CPU/GPU身份不一致：{scale_id}")
    speedup = float(gpu["cpu_gpu_speedup"])
    require(abs(speedup - float(cpu["mean_ms"]) / float(gpu["mean_ms"])) <= 1e-9, f"speedup公式不一致：{scale_id}")
    return {
        "scale_id": scale_id,
        "run_id": gpu["run_id"],
        "case_id": gpu["case_id"],
        "source_csv": (source_prefix / relative).as_posix(),
        "source_csv_sha256": sha256(path),
        "cpu_mean_ms": float(cpu["mean_ms"]),
        "gpu_mean_ms": float(gpu["mean_ms"]),
        "cpu_gpu_speedup": speedup,
        "gpu_p95_ms": float(gpu["p95_ms"]),
        "gpu_cv": float(gpu["cv"]),
        "gpu_peak_bytes": int(gpu["gpu_peak_bytes"]),
        "status": "PASS",
        "accuracy_status": "PASS",
        "return_code": 0,
        "backend": "fft_thrust",
        "dtype": "ComplexFP32",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--source-prefix", required=True, type=Path)
    parser.add_argument("--task-scales", required=True, type=Path)
    parser.add_argument("--memory-evidence", required=True, type=Path)
    parser.add_argument("--existing-profile", required=True, type=Path)
    parser.add_argument("--output-profile", required=True, type=Path)
    parser.add_argument("--output-ranking", required=True, type=Path)
    args = parser.parse_args()

    verify_memory(args.memory_evidence)
    scales = load_scales(args.task_scales)
    candidates = [load_candidate(args.formal_root, args.source_prefix, scale) for scale in scales]
    candidates.sort(key=lambda item: (-item["cpu_gpu_speedup"], item["gpu_p95_ms"], item["gpu_cv"], item["gpu_peak_bytes"], item["scale_id"]))
    require(len(candidates) == 110, "完整候选数量异常")
    best = candidates[0]

    profile = json.loads(args.existing_profile.read_text(encoding="utf-8"))
    entries = profile.get("entries", [])
    require(any(item.get("profile_entry_id") == "task2_fft_thrust_fp32" for item in entries), "现有Task2 entry缺失")
    require(not any(item.get("target") == "Task1" for item in entries), "Task1 entry已存在，拒绝覆盖")
    entries.append({
        "profile_entry_id": "task1_fft_thrust_complexfp32",
        "target_kind": "task",
        "target": "Task1",
        "backend": "fft_thrust",
        "dtype": "ComplexFP32",
        "best_scale_id": best["scale_id"],
        "source_csv": best["source_csv"],
        "source_csv_sha256": best["source_csv_sha256"],
        "run_id": best["run_id"],
        "case_id": best["case_id"],
        "selection_metric": "cpu_gpu_speedup",
        "selection_value": best["cpu_gpu_speedup"],
        "candidate_scale_ids": [item["scale_id"] for item in candidates],
        "selection_reason": "完整110个预定scale均满足正式CPU/GPU、accuracy、正常退出、fft_thrust backend与同commit clean 1024+1024资源门禁；按speedup降序、GPU p95/CV/峰值显存升序及scale_id稳定兜底排序，candidate_scale_ids即完整排名。",
    })
    profile["profile_id"] = "demo_task1_task2_best_v1"
    profile["coverage"] = "minimal_example"
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_ranking.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", *candidates[0].keys()]
    with args.output_ranking.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(candidates, 1):
            writer.writerow({"rank": rank, **item})
    print(json.dumps({"status": "PASS", "candidate_count": len(candidates), "best_scale_id": best["scale_id"], "profile": str(args.output_profile), "ranking": str(args.output_ranking)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
