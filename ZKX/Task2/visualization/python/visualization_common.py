#!/usr/bin/env python3
"""Task1/Task2 真实 pipeline 可视化的公共运行与报告组件。"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


EPSILON = np.finfo(float).eps


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str
    formula: str
    standard: str
    grade: str
    passed: bool


@dataclass(frozen=True)
class StepEvaluation:
    name: str
    metrics: Sequence[Metric]
    annotation: str
    narrative: str

    @property
    def grade(self) -> str:
        return worst_grade(self.metrics)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    data: Path
    figures: Path
    evaluation: Path
    config: Path
    log: Path


def grade_higher(value: float, excellent: float, good: float) -> Tuple[str, bool]:
    if value >= excellent:
        return "优秀", True
    if value >= good:
        return "良好", True
    return "一般", False


def grade_lower(value: float, excellent: float, good: float) -> Tuple[str, bool]:
    if value <= excellent:
        return "优秀", True
    if value <= good:
        return "良好", True
    return "一般", False


def worst_grade(metrics: Iterable[Metric]) -> str:
    ranks = {"一般": 1, "良好": 2, "优秀": 3}
    values = [ranks[item.grade] for item in metrics]
    return {1: "一般", 2: "良好", 3: "优秀"}[min(values)]


def metric(
    name: str,
    value: float,
    unit: str,
    formula: str,
    standard: str,
    grade_and_pass: Tuple[str, bool],
) -> Metric:
    grade, passed = grade_and_pass
    return Metric(name, float(value), unit, formula, standard, grade, passed)


def read_manifest(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.reader(input_file))
    if not rows or rows[0] != ["key", "value"]:
        raise RuntimeError("manifest header is invalid: {}".format(path))
    result: Dict[str, str] = {}
    for row in rows[1:]:
        if len(row) != 2 or not row[0] or row[0] in result:
            raise RuntimeError("manifest row is invalid: {}".format(row))
        result[row[0]] = row[1]
    return result


def require_manifest(
    manifest: Mapping[str, str], expected: Mapping[str, str], numeric: Sequence[str]
) -> Dict[str, float]:
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                "manifest {} mismatch: expected {!r}, got {!r}".format(
                    key, value, manifest.get(key)
                )
            )
    parsed: Dict[str, float] = {}
    for key in numeric:
        if key not in manifest:
            raise RuntimeError("manifest is missing {}".format(key))
        parsed[key] = float(manifest[key])
    return parsed


def read_numeric_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        header = next(csv.reader(input_file))
    values = np.loadtxt(str(path), delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] != len(header):
        raise RuntimeError("CSV column count mismatch: {}".format(path))
    return {name: values[:, index] for index, name in enumerate(header)}


def read_feature_csv(path: Path) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["family", "index", "normalized_position", "value"]:
            raise RuntimeError("Task2 feature CSV header is invalid")
        for row in reader:
            grouped.setdefault(row["family"], []).append(float(row["value"]))
    return {key: np.asarray(value, dtype=float) for key, value in grouped.items()}


def prepare_run_paths(evidence_root: Path, task: str, mode: str) -> RunPaths:
    root = evidence_root.resolve()
    if root.name != "visualization" or root.parent.name != "evidence":
        raise RuntimeError("unsafe visualization output root: {}".format(root))
    if mode not in {"full", "export", "plot"}:
        raise RuntimeError("unknown visualization mode: {}".format(mode))
    if mode in {"full", "export"} and root.exists():
        for child in root.iterdir():
            if child.name == ".gitignore":
                continue
            if child.is_dir():
                shutil.rmtree(str(child))
            else:
                child.unlink()
    data = root / "data"
    figures = root / "figures"
    evaluation = root / "evaluation"
    if mode == "plot":
        if not data.is_dir():
            raise RuntimeError(
                "visualization data is missing: {}; run --export-only on ZQ500 "
                "and pull the data once before local plotting".format(data)
            )
    for directory in (data, figures, evaluation):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        data=data,
        figures=figures,
        evaluation=evaluation,
        config=root / "{}_selected.conf".format(task.lower()),
        log=root / "pipeline.log",
    )


def materialize_best_config(
    repository_root: Path, task: str, output_path: Path
) -> Tuple[str, Dict[str, object]]:
    profile_path = repository_root / "demo" / "config" / "runtime" / "demo_profile.json"
    scales_path = (
        repository_root / "test_all" / "configs" / "benchmark" / "task_input_scales.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    entries = [entry for entry in profile.get("entries", []) if entry.get("target") == task]
    if len(entries) != 1:
        raise RuntimeError("demo profile must contain exactly one {} entry".format(task))
    scale_id = str(entries[0]["best_scale_id"])
    scales = json.loads(scales_path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in scales["tasks"][task]["scales"]
        if item.get("scale_id") == scale_id
    ]
    if len(candidates) != 1:
        raise RuntimeError("scale {} is missing or duplicated".format(scale_id))
    parameters = candidates[0]["parameters"]
    lines = ["# Generated from demo/config/runtime/demo_profile.json: {}".format(scale_id)]
    lines.extend("{}={}".format(key, value) for key, value in parameters.items())
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return scale_id, candidates[0]


def resolve_pipeline(
    requested: str, repository_root: Path, task: str
) -> Path:
    if requested:
        candidate = Path(requested).expanduser().resolve()
        if not candidate.is_file():
            raise RuntimeError("pipeline executable does not exist: {}".format(candidate))
        return candidate
    lower = task.lower()
    candidates = [
        repository_root
        / "build"
        / "zq500"
        / backend
        / "bin"
        / "{}_pipeline".format(lower)
        for backend in ("fft-thrust", "dlfft")
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "cannot find {} pipeline; pass --pipeline with the built executable".format(task)
    )


def run_pipeline(
    pipeline: Path, config: Path, paths: RunPaths, repository_root: Path
) -> float:
    environment = os.environ.copy()
    environment["ZKX_VISUALIZATION_DIR"] = str(paths.data)
    command = [str(pipeline), "--config", str(config)]
    begin = time.monotonic()
    with paths.log.open("w", encoding="utf-8", newline="\n") as log_file:
        log_file.write("command={}\n".format(" ".join(command)))
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=str(repository_root),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write("return_code={}\n".format(completed.returncode))
    elapsed = time.monotonic() - begin
    if completed.returncode != 0:
        raise RuntimeError(
            "pipeline failed with return code {}; see {}".format(
                completed.returncode, paths.log
            )
        )
    return elapsed


def configure_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path(tempfile.gettempdir()) / "zkx-matplotlib-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    windows_font = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "simsun.ttc"
    if windows_font.is_file():
        font_manager.fontManager.addfont(str(windows_font))
        font_family = "SimSun"
    else:
        font_family = "Noto Serif CJK SC"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": [font_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 15,
            "savefig.bbox": "tight",
        }
    )
    return plt


def downsample_xy(x: np.ndarray, y: np.ndarray, limit: int = 4000):
    if x.size <= limit:
        return x, y
    indices = np.linspace(0, x.size - 1, limit, dtype=int)
    return x[indices], y[indices]


def save_figure(figure, directory: Path, filename: str) -> None:
    figure.savefig(str(directory / filename), dpi=160)
    import matplotlib.pyplot as plt

    plt.close(figure)


def write_evaluation(
    task_slug: str,
    title: str,
    steps: Sequence[StepEvaluation],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(steps, start=1):
        path = output_dir / "evaluation_{}_step{}.csv".format(task_slug, index)
        with path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "指标名称",
                    "数值",
                    "单位",
                    "计算公式",
                    "项目建议标准",
                    "评价结果",
                    "是否满足项目建议",
                ]
            )
            for item in step.metrics:
                writer.writerow(
                    [
                        item.name,
                        "{:.17g}".format(item.value),
                        item.unit,
                        item.formula,
                        item.standard,
                        item.grade,
                        "是" if item.passed else "否",
                    ]
                )
    report = output_dir / "summary_report.txt"
    with report.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(title + "\n" + "=" * len(title) + "\n\n")
        output_file.write(
            "说明：以下阈值为项目建议范围，不是赛事官方验收标准。\n\n"
        )
        for step in steps:
            output_file.write("Step名称：{}\n".format(step.name))
            for item in step.metrics:
                output_file.write(
                    "指标：{}\n结果：{:.10g} {}\n公式：{}\n项目建议范围：{}\n"
                    "是否满足项目建议：{}\n评价：{}\n\n".format(
                        item.name,
                        item.value,
                        item.unit,
                        item.formula,
                        item.standard,
                        "是" if item.passed else "否",
                        item.grade,
                    )
                )
            output_file.write(
                "自动评价：{}\n结论：{}\n\n{}\n\n".format(
                    step.narrative, step.grade, "-" * 72
                )
            )


def write_run_summary(
    paths: RunPaths,
    task: str,
    scale_id: str,
    pipeline_seconds: float,
    total_seconds: float,
) -> None:
    payload = {
        "task": task,
        "scale_id": scale_id,
        "pipeline_seconds": pipeline_seconds,
        "total_seconds": total_seconds,
        "status": "PASS",
        "data_directory": str(paths.data),
        "figures_directory": str(paths.figures),
        "evaluation_directory": str(paths.evaluation),
    }
    (paths.root / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def finite(value: float) -> bool:
    return math.isfinite(float(value))
