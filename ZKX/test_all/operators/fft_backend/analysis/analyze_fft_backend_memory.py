#!/usr/bin/env python3
"""统一校验、合并、统计并绘制 fft_thrust 与 dlfft 内存数据。"""

from __future__ import annotations

import argparse
import cmath
import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


BACKENDS = ("fft_thrust", "dlfft")
RUN_IDS = (1, 2, 3)
ITERATIONS = 2000
FFT_LENGTH = 1024
BATCH = 1
DTYPE = "complex64"
SAMPLE_ITERATIONS = tuple(range(1, 11)) + tuple(range(991, 1011)) + tuple(range(1991, 2001))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV 无数据：{path}")
    return rows


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def verify_outputs(raw_dir: Path, output_path: Path) -> dict[str, object]:
    rows = {backend: read_csv(raw_dir / f"verify_{backend}.csv") for backend in BACKENDS}
    if any(len(value) != FFT_LENGTH * BATCH for value in rows.values()):
        raise ValueError("正确性预检输出长度不正确")
    for index in range(FFT_LENGTH * BATCH):
        if rows["fft_thrust"][index]["index"] != str(index) or rows["dlfft"][index]["index"] != str(index):
            raise ValueError("正确性预检 index 不连续")
        for field in ("input_real", "input_imag"):
            if rows["fft_thrust"][index][field] != rows["dlfft"][index][field]:
                raise ValueError("两种后端的预检输入不一致")

    input_values = [
        complex(float(row["input_real"]), float(row["input_imag"]))
        for row in rows["fft_thrust"][:FFT_LENGTH]
    ]
    reference: list[complex] = []
    for frequency in range(FFT_LENGTH):
        total = 0j
        for index, value in enumerate(input_values):
            total += value * cmath.exp(-2j * math.pi * frequency * index / FFT_LENGTH)
        reference.append(total)

    outputs = {
        backend: [
            complex(float(row["output_real"]), float(row["output_imag"]))
            for row in backend_rows[:FFT_LENGTH]
        ]
        for backend, backend_rows in rows.items()
    }
    backend_difference = max(
        abs(left - right) for left, right in zip(outputs["fft_thrust"], outputs["dlfft"])
    )
    reference_errors = {
        backend: max(abs(actual - expected) for actual, expected in zip(values, reference))
        for backend, values in outputs.items()
    }
    reference_scale = max(1.0, max(abs(value) for value in reference))
    tolerance = 1.0e-5 * reference_scale + 1.0e-4
    passed = backend_difference <= tolerance and all(
        error <= tolerance for error in reference_errors.values()
    )
    result = {
        "fft_length": FFT_LENGTH,
        "batch": BATCH,
        "dtype": DTYPE,
        "tolerance": tolerance,
        "fft_thrust_max_abs_error_vs_reference": reference_errors["fft_thrust"],
        "dlfft_max_abs_error_vs_reference": reference_errors["dlfft"],
        "backend_max_abs_difference": backend_difference,
        "status": "passed" if passed else "failed",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if not passed:
        raise ValueError(f"FFT 输出正确性预检失败：{result}")
    return result


def validate_lifecycle(rows: list[dict[str, str]], backend: str, run_id: int, path: Path) -> None:
    if len(rows) != ITERATIONS:
        raise ValueError(f"{path} 必须有 {ITERATIONS} 行，实际 {len(rows)}")
    required_zero = (
        "create_status",
        "input_alloc_status",
        "output_alloc_status",
        "buffer_alloc_status",
        "input_copy_status",
        "initialization_sync_status",
        "execute_status",
        "execution_sync_status",
        "destroy_status",
        "input_free_status",
        "output_free_status",
        "buffer_free_status",
        "post_free_sync_status",
        "memory_read_status",
    )
    for iteration, row in enumerate(rows, start=1):
        if as_int(row, "run_id") != run_id or as_int(row, "iteration") != iteration:
            raise ValueError(f"{path}:{iteration} run_id/iteration 不正确")
        if row["backend"] != backend:
            raise ValueError(f"{path}:{iteration} backend 不正确")
        if as_int(row, "fft_length") != FFT_LENGTH or as_int(row, "batch") != BATCH or row["dtype"] != DTYPE:
            raise ValueError(f"{path}:{iteration} FFT 参数不一致")
        for field in required_zero:
            if as_int(row, field) != 0:
                raise ValueError(f"{path}:{iteration} {field}={row[field]}")
        if row["error_message"]:
            raise ValueError(f"{path}:{iteration} error_message={row['error_message']}")


def validate_resident(
    rows: list[dict[str, str]], backend: str, run_id: int, path: Path, cleanup_path: Path
) -> None:
    if len(rows) != ITERATIONS:
        raise ValueError(f"{path} 必须有 {ITERATIONS} 行")
    for iteration, row in enumerate(rows, start=1):
        if as_int(row, "run_id") != run_id or as_int(row, "iteration") != iteration:
            raise ValueError(f"{path}:{iteration} run_id/iteration 不正确")
        if row["backend"] != backend:
            raise ValueError(f"{path}:{iteration} backend 不正确")
        if as_int(row, "fft_length") != FFT_LENGTH or as_int(row, "batch") != BATCH or row["dtype"] != DTYPE:
            raise ValueError(f"{path}:{iteration} FFT 参数不一致")
        for field in ("execute_status", "synchronize_status", "memory_read_status"):
            if as_int(row, field) != 0:
                raise ValueError(f"{path}:{iteration} {field}={row[field]}")
        if row["error_message"]:
            raise ValueError(f"{path}:{iteration} error_message={row['error_message']}")
    cleanup = read_csv(cleanup_path)
    if len(cleanup) != 1 or cleanup[0]["backend"] != backend:
        raise ValueError(f"{cleanup_path} cleanup 记录无效")
    for field in ("destroy_status", "output_free_status", "input_free_status", "synchronize_status"):
        if as_int(cleanup[0], field) != 0:
            raise ValueError(f"{cleanup_path} {field}={cleanup[0][field]}")
    if cleanup[0]["error_message"]:
        raise ValueError(f"{cleanup_path} error_message={cleanup[0]['error_message']}")


def regression(values: list[int]) -> tuple[float, float]:
    count = len(values)
    x_mean = (count + 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(1, count + 1))
    slope = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values, start=1)
    ) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum(
        (value - (intercept + slope * index)) ** 2
        for index, value in enumerate(values, start=1)
    )
    total = sum((value - y_mean) ** 2 for value in values)
    return slope, 1.0 if total == 0.0 else 1.0 - residual / total


def step_counts(steps: Iterable[int]) -> tuple[int, int, int]:
    positive = negative = zero = 0
    for step in steps:
        if step > 0:
            positive += 1
        elif step < 0:
            negative += 1
        else:
            zero += 1
    return positive, negative, zero


def metric_row(
    scenario: str,
    run_id: int,
    series: str,
    resource: str,
    current: list[int],
    growth: list[int],
    steps: list[int],
) -> dict[str, object]:
    slope, r_squared = regression(current)
    positive, negative, zero = step_counts(steps)
    return {
        "scenario": scenario,
        "run_id": run_id,
        "series": series,
        "resource": resource,
        "net_growth_bytes": growth[-1],
        "first_100_growth_bytes": growth[99],
        "last_100_growth_bytes": current[-1] - current[-101],
        "first_1000_growth_bytes": growth[999],
        "last_1000_growth_bytes": current[-1] - current[999],
        "positive_step_count": positive,
        "negative_step_count": negative,
        "zero_step_count": zero,
        "mean_step_growth_bytes": f"{statistics.fmean(steps):.9f}",
        "median_step_growth_bytes": f"{statistics.median(steps):.9f}",
        "minimum_step_growth_bytes": min(steps),
        "maximum_step_growth_bytes": max(steps),
        "linear_regression_slope_bytes_per_iteration": f"{slope:.9f}",
        "r_squared": f"{r_squared:.9f}",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_pair(
    scenario: str,
    run_id: int,
    thrust: list[dict[str, str]],
    dlfft: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, object]]:
    fields = [field for field in thrust[0] if field not in ("run_id", "iteration")]
    merged: list[dict[str, object]] = []
    if scenario == "lifecycle":
        cpu_growth_field = "cpu_after_free_growth_from_baseline_bytes"
        gpu_growth_field = "gpu_after_free_growth_from_baseline_bytes"
    else:
        cpu_growth_field = "cpu_growth_from_baseline_bytes"
        gpu_growth_field = "gpu_growth_from_baseline_bytes"
    for thrust_row, dlfft_row in zip(thrust, dlfft):
        row: dict[str, object] = {"run_id": run_id, "iteration": thrust_row["iteration"]}
        for field in fields:
            row[f"fft_thrust_{field}"] = thrust_row[field]
            row[f"dlfft_{field}"] = dlfft_row[field]
        row["dlfft_minus_fft_thrust_cpu_growth_bytes"] = (
            as_int(dlfft_row, cpu_growth_field) - as_int(thrust_row, cpu_growth_field)
        )
        row["dlfft_minus_fft_thrust_gpu_growth_bytes"] = (
            as_int(dlfft_row, gpu_growth_field) - as_int(thrust_row, gpu_growth_field)
        )
        merged.append(row)
    write_rows(output_path, merged)
    return merged


def statistics_for_pair(
    scenario: str,
    run_id: int,
    backend_rows: dict[str, list[dict[str, str]]],
    merged: list[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for backend in BACKENDS:
        rows = backend_rows[backend]
        for resource in ("CPU", "GPU"):
            prefix = resource.lower()
            if scenario == "lifecycle":
                current_field = f"{prefix}_after_free_bytes"
                growth_field = f"{prefix}_after_free_growth_from_baseline_bytes"
                step_field = f"{prefix}_after_free_step_growth_bytes"
            else:
                current_field = f"{prefix}_current_bytes"
                growth_field = f"{prefix}_growth_from_baseline_bytes"
                step_field = f"{prefix}_step_growth_bytes"
            result.append(
                metric_row(
                    scenario,
                    run_id,
                    backend,
                    resource,
                    [as_int(row, current_field) for row in rows],
                    [as_int(row, growth_field) for row in rows],
                    [as_int(row, step_field) for row in rows],
                )
            )
    for resource in ("CPU", "GPU"):
        field = f"dlfft_minus_fft_thrust_{resource.lower()}_growth_bytes"
        values = [int(row[field]) for row in merged]
        steps = [values[0]] + [values[index] - values[index - 1] for index in range(1, len(values))]
        result.append(metric_row(scenario, run_id, "dlfft-fft_thrust", resource, values, values, steps))
    return result


def sustained_growth(row: dict[str, object]) -> bool:
    return (
        int(row["net_growth_bytes"]) > 65536
        and int(row["last_1000_growth_bytes"]) > 32768
        and float(row["linear_regression_slope_bytes_per_iteration"]) > 1.0
        and float(row["r_squared"]) > 0.9
    )


def classify(summary: list[dict[str, object]], scenario: str) -> tuple[str, str]:
    rows = [
        row for row in summary
        if row["scenario"] == scenario and row["resource"] == "CPU" and row["series"] in BACKENDS
    ]
    by_backend = {
        backend: [row for row in rows if row["series"] == backend]
        for backend in BACKENDS
    }
    thrust_sustained = [sustained_growth(row) for row in by_backend["fft_thrust"]]
    dlfft_sustained = [sustained_growth(row) for row in by_backend["dlfft"]]
    if all(dlfft_sustained) and not any(thrust_sustained):
        conclusion = (
            "当前 SDK 的 dlfft 完整调用生命周期存在可重复的同进程 CPU 内存泄漏；"
            "fft_thrust 在该测试条件下未观察到同类增长。"
            if scenario == "lifecycle"
            else "当前 SDK 的 dlfft 在常驻执行场景存在可重复的同进程 CPU 内存泄漏；"
            "fft_thrust 在该测试条件下未观察到同类增长。"
        )
        return (
            "dlfft_reproducible_cpu_leak",
            conclusion,
        )
    if all(dlfft_sustained) and all(thrust_sustained):
        return (
            "both_backends_grow",
            "两个后端均持续增长，应优先检查公共测试代码、allocator、输入初始化和公共释放逻辑。",
        )
    if not any(dlfft_sustained) and not any(thrust_sustained):
        return (
            "no_sustained_growth_or_plateau",
            "两个后端均未满足持续增长判据；若存在早期增长，应按初始化或缓存趋稳解释。",
        )
    return ("inconclusive", "三次重复的增长形态不一致，当前数据不足以形成稳定归因。")


def svg_chart(
    path: Path,
    title: str,
    ylabel: str,
    series: list[tuple[str, list[int], str, str]],
) -> None:
    width, height = 960, 480
    left, right, top, bottom = 86, 28, 45, 62
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for _, data, _, _ in series for value in data]
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        minimum -= 1
        maximum += 1
    padding = (maximum - minimum) * 0.08
    minimum -= padding
    maximum += padding

    def point(index: int, value: int, count: int) -> tuple[float, float]:
        x = left + index * plot_width / max(1, count - 1)
        y = top + (maximum - value) * plot_height / (maximum - minimum)
        return x, y

    ticks = 5
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{html.escape(title)}</title>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(title)}</text>',
    ]
    for tick in range(ticks + 1):
        y = top + tick * plot_height / ticks
        value = maximum - tick * (maximum - minimum) / ticks
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d7dce2"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:,.0f}</text>')
    for iteration in (1, 500, 1000, 1500, 2000):
        x = left + (iteration - 1) * plot_width / (ITERATIONS - 1)
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{iteration}</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">iteration</text>')
    parts.append(f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(ylabel)}</text>')
    for label, data, color, dash in series:
        selected = list(range(0, len(data), 10))
        if selected[-1] != len(data) - 1:
            selected.append(len(data) - 1)
        coordinates = [point(index, data[index], len(data)) for index in selected]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"{dash_attribute}/>')
    legend_x = left + 8
    legend_y = top + 16
    for index, (label, _, color, dash) in enumerate(series):
        x = legend_x + (index % 3) * 265
        y = legend_y + (index // 3) * 20
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+24}" y2="{y}" stroke="{color}" stroke-width="2"{dash_attribute}/>')
        parts.append(f'<text x="{x+30}" y="{y+4}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def generate_charts(
    output_dir: Path,
    lifecycle: dict[int, dict[str, list[dict[str, str]]]],
    resident: dict[int, dict[str, list[dict[str, str]]]],
    merged_lifecycle: dict[int, list[dict[str, object]]],
) -> None:
    colors = {"fft_thrust": "#2563a6", "dlfft": "#d97706"}
    dashes = {1: "", 2: "6 4", 3: "2 3"}
    figures = output_dir / "figures"
    for scenario, datasets in (("lifecycle", lifecycle), ("resident", resident)):
        for resource in ("cpu", "gpu"):
            series = []
            for backend in BACKENDS:
                for run_id in RUN_IDS:
                    rows = datasets[run_id][backend]
                    field = (
                        f"{resource}_after_free_growth_from_baseline_bytes"
                        if scenario == "lifecycle"
                        else f"{resource}_growth_from_baseline_bytes"
                    )
                    series.append((
                        f"{backend} run{run_id}",
                        [as_int(row, field) for row in rows],
                        colors[backend],
                        dashes[run_id],
                    ))
            svg_chart(
                figures / f"{scenario}_{resource}_growth.svg",
                f"{scenario} {resource.upper()} cumulative growth",
                "growth from baseline (bytes)",
                series,
            )
    delta_series = []
    for run_id in RUN_IDS:
        rows = merged_lifecycle[run_id]
        delta_series.append((
            f"CPU run{run_id}",
            [int(row["dlfft_minus_fft_thrust_cpu_growth_bytes"]) for row in rows],
            "#7c3aed",
            dashes[run_id],
        ))
        delta_series.append((
            f"GPU run{run_id}",
            [int(row["dlfft_minus_fft_thrust_gpu_growth_bytes"]) for row in rows],
            "#0f766e",
            dashes[run_id],
        ))
    svg_chart(
        figures / "lifecycle_backend_difference.svg",
        "DLFFT minus fft_thrust cumulative growth",
        "difference (bytes)",
        delta_series,
    )


def sample_rows(
    scenario: str,
    datasets: dict[int, dict[str, list[dict[str, str]]]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for run_id in RUN_IDS:
        for iteration in SAMPLE_ITERATIONS:
            row: dict[str, object] = {"scenario": scenario, "run_id": run_id, "iteration": iteration}
            for backend in BACKENDS:
                source = datasets[run_id][backend][iteration - 1]
                if scenario == "lifecycle":
                    row[f"{backend}_cpu_growth_bytes"] = source["cpu_after_free_growth_from_baseline_bytes"]
                    row[f"{backend}_gpu_growth_bytes"] = source["gpu_after_free_growth_from_baseline_bytes"]
                else:
                    row[f"{backend}_cpu_growth_bytes"] = source["cpu_growth_from_baseline_bytes"]
                    row[f"{backend}_gpu_growth_bytes"] = source["gpu_growth_from_baseline_bytes"]
            row["dlfft_minus_fft_thrust_cpu_growth_bytes"] = (
                int(row["dlfft_cpu_growth_bytes"]) - int(row["fft_thrust_cpu_growth_bytes"])
            )
            row["dlfft_minus_fft_thrust_gpu_growth_bytes"] = (
                int(row["dlfft_gpu_growth_bytes"]) - int(row["fft_thrust_gpu_growth_bytes"])
            )
            result.append(row)
    return result


def report_table(stream, scenario: str, datasets) -> None:
    stream.write(f"### {scenario}：run 1 指定轮次原始增长值\n\n")
    stream.write("| iteration | fft_thrust CPU/B | dlfft CPU/B | CPU差分/B | fft_thrust GPU/B | dlfft GPU/B |\n")
    stream.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for iteration in SAMPLE_ITERATIONS:
        thrust = datasets[1]["fft_thrust"][iteration - 1]
        dlfft = datasets[1]["dlfft"][iteration - 1]
        if scenario == "完整生命周期":
            cpu = "cpu_after_free_growth_from_baseline_bytes"
            gpu = "gpu_after_free_growth_from_baseline_bytes"
        else:
            cpu = "cpu_growth_from_baseline_bytes"
            gpu = "gpu_growth_from_baseline_bytes"
        thrust_cpu, dlfft_cpu = as_int(thrust, cpu), as_int(dlfft, cpu)
        stream.write(
            f"| {iteration} | {thrust_cpu} | {dlfft_cpu} | {dlfft_cpu-thrust_cpu} | "
            f"{as_int(thrust, gpu)} | {as_int(dlfft, gpu)} |\n"
        )


def summary_table(stream, summary: list[dict[str, object]], scenario: str) -> None:
    stream.write("| run | 序列 | 资源 | 净增长/B | 前100/B | 后100/B | 前1000/B | 后1000/B | 正/负/零 | 单步均值/B | 中位数/B | 最小/B | 最大/B | 斜率/B·轮⁻¹ | R² |\n")
    stream.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in summary:
        if row["scenario"] != scenario:
            continue
        stream.write(
            f"| {row['run_id']} | {row['series']} | {row['resource']} | "
            f"{row['net_growth_bytes']} | {row['first_100_growth_bytes']} | "
            f"{row['last_100_growth_bytes']} | {row['first_1000_growth_bytes']} | "
            f"{row['last_1000_growth_bytes']} | {row['positive_step_count']}/"
            f"{row['negative_step_count']}/{row['zero_step_count']} | "
            f"{row['mean_step_growth_bytes']} | {row['median_step_growth_bytes']} | "
            f"{row['minimum_step_growth_bytes']} | {row['maximum_step_growth_bytes']} | "
            f"{row['linear_regression_slope_bytes_per_iteration']} | {row['r_squared']} |\n"
        )


def write_report(
    path: Path,
    verification: dict[str, object],
    summary: list[dict[str, object]],
    lifecycle,
    resident,
) -> None:
    lifecycle_class, lifecycle_text = classify(summary, "lifecycle")
    resident_class, resident_text = classify(summary, "resident")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# ZQ500 FFT 后端统一内存稳定性测试报告\n\n")
        stream.write("> 本报告由 `analyze_fft_backend_memory.py` 从本轮原始 CSV 自动生成；数值、表格、曲线和分类不得人工回填。\n\n")
        stream.write("## 1. 结论\n\n")
        stream.write(f"完整生命周期自动分类：`{lifecycle_class}`。{lifecycle_text}\n\n")
        stream.write(f"常驻执行自动分类：`{resident_class}`。{resident_text}\n\n")
        stream.write("结论只覆盖 ZQ500 `gpu_02`、长度 1024、batch 1、complex64、100 轮预热、2,000 轮正式数据及本文同步/采样流程，不外推到其他长度、类型、batch 或并发场景。\n\n")
        stream.write("## 2. 统一测试流程\n\n")
        stream.write("```mermaid\nflowchart LR\n  A[before_create 采样] --> B[创建正式后端 plan]\n  B --> C[申请相同输入/输出 buffer]\n  C --> D[写入相同 complex64 输入]\n  D --> E[设备同步]\n  E --> F[执行一次正式 fft_execute]\n  F --> G[设备同步并采样释放前内存]\n  G --> H[销毁 plan/内部 workspace]\n  H --> I[释放输入输出 buffer]\n  I --> J[设备同步 + malloc_trim]\n  J --> K[after_destroy_and_free 采样]\n```\n\n")
        stream.write("测试框架、输入、四阶段采样、错误检查和 CSV 写入完全共用。两个构建目录只替换 `fft_execute` 的正式实现源：`fft_thrust.cu` 或 `fft_dlfft.cu`；后者内部调用 `cufftExecC2C`。\n\n")
        stream.write("接口结构上，两端都通过项目正式 `FFTPlanHandle` API 创建和销毁执行对象。DLFFT plan 内部持有 SDK `cufftHandle`，销毁时调用 `cufftDestroy`；fft_thrust plan 内部持有 radix-2 twiddle/workspace device buffer，销毁时逐项 `tracked_cuda_free` 并检查返回值。测试框架还统一申请和释放同样的外部输入/输出 buffer。\n\n")
        stream.write("## 3. 构建与运行命令\n\n")
        stream.write("必须先按 `docs/development/ZQ500_RUNTIME.md` 完成本地核对、宿主机同步、精确 `docker cp`、交互进入 `gpu_02` 和 SDK 初始化。容器内命令为：\n\n")
        stream.write("```bash\nsource /zq500/sdk/env.sh\ncd /tmp/ZKX_dev\ncmake -S . -B build_fft_memory_thrust -DUSE_DLFFT=OFF -DUSE_THRUST=ON\ncmake --build build_fft_memory_thrust --target fft_backend_memory_unified -j1\ncmake -S . -B build_fft_memory_dlfft -DUSE_DLFFT=ON -DUSE_THRUST=OFF\ncmake --build build_fft_memory_dlfft --target fft_backend_memory_unified -j1\nbash test_all/operators/fft_backend/scripts/run_fft_backend_memory_campaign.sh \\\n  ./build_fft_memory_thrust/bin/fft_backend_memory_unified \\\n  ./build_fft_memory_dlfft/bin/fft_backend_memory_unified \\\n  fft_backend_memory_unified_20260720\n```\n\n")
        stream.write("## 4. 输出正确性预检\n\n")
        stream.write(
            f"两种后端使用同一输入，并与 CPU DFT reference 比较。容差 {verification['tolerance']:.9g}；"
            f"fft_thrust 最大绝对误差 {verification['fft_thrust_max_abs_error_vs_reference']:.9g}，"
            f"dlfft 最大绝对误差 {verification['dlfft_max_abs_error_vs_reference']:.9g}，"
            f"后端间最大差异 {verification['backend_max_abs_difference']:.9g}，状态 `{verification['status']}`。\n\n"
        )
        stream.write("## 5. 完整生命周期结果\n\n")
        summary_table(stream, summary, "lifecycle")
        stream.write("\n")
        stream.write("![完整生命周期 CPU 累计增长](figures/lifecycle_cpu_growth.svg)\n\n")
        stream.write("![完整生命周期 GPU 累计增长](figures/lifecycle_gpu_growth.svg)\n\n")
        stream.write("![DLFFT 减 fft_thrust 差分增长](figures/lifecycle_backend_difference.svg)\n\n")
        stream.write("## 6. 常驻执行结果\n\n")
        summary_table(stream, summary, "resident")
        stream.write("\n")
        stream.write("![常驻执行 CPU 累计增长](figures/resident_cpu_growth.svg)\n\n")
        stream.write("![常驻执行 GPU 累计增长](figures/resident_gpu_growth.svg)\n\n")
        stream.write("## 7. 指定轮次真实数据\n\n")
        report_table(stream, "完整生命周期", lifecycle)
        stream.write("\n")
        report_table(stream, "常驻执行", resident)
        stream.write("\n完整三次重复的这些轮次见 `report_samples.csv`；完整 2,000 轮见 `raw/` 和 `comparison/`。\n\n")
        stream.write("## 8. 指标边界\n\n")
        stream.write("`mallinfo2().uordblks` 是 glibc allocator 当前仍分配空间，不等于 RSS；VmRSS 只作辅助，RSS 不下降不能直接判泄漏。GPU 使用同序的 `cudaMemGetInfo(total-free)`，是设备级视图。主要判据始终是资源释放后的累计增长、后 1,000 轮是否仍增长、三次独立进程能否复现，以及 DLFFT−fft_thrust 差分是否扩大。自动持续增长判据要求净增长大于 64 KiB、后 1,000 轮增长大于 32 KiB、全窗斜率大于 1 B/轮且 R² 大于 0.9；不要求每轮上涨，也不使用正负步数比例作硬门槛。\n")
        stream.write("\n## 9. 交付清单\n\n")
        stream.write("- 统一测试源码：`test_all/operators/fft_backend/src/fft_backend_memory_unified.cu`；\n- 后端适配：正式 `fft_thrust.cu` 与 `fft_dlfft.cu`；\n- CMake：目标 `fft_backend_memory_unified`；\n- 自动运行：`fft_backend/scripts/run_fft_backend_memory_campaign.sh`；\n- 统计与绘图：`fft_backend/analysis/analyze_fft_backend_memory.py`；\n- 原始数据：`raw/` 中正确性输出、生命周期 6 个 CSV、常驻执行 6 个 CSV 及 cleanup CSV；\n- 对照数据：`analysis/comparison/`、`lifecycle_all_runs.csv`、`resident_all_runs.csv`；\n- 统计与抽样：`backend_memory_summary.csv`、`report_samples.csv`、`manifest.json`；\n- 曲线：`analysis/figures/` 中 5 张由 CSV 自动生成的 SVG。\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_outputs(arguments.raw_dir, arguments.output_dir / "correctness_verification.json")
    if arguments.verify_only:
        print("[FFT-BACKEND-MEMORY][VERIFY] status=passed")
        return 0

    datasets: dict[str, dict[int, dict[str, list[dict[str, str]]]]] = {
        "lifecycle": {}, "resident": {}
    }
    merged_by_scenario: dict[str, dict[int, list[dict[str, object]]]] = {
        "lifecycle": {}, "resident": {}
    }
    summary: list[dict[str, object]] = []
    combined: dict[str, list[dict[str, object]]] = {"lifecycle": [], "resident": []}
    comparison_dir = arguments.output_dir / "comparison"
    for scenario in ("lifecycle", "resident"):
        for run_id in RUN_IDS:
            datasets[scenario][run_id] = {}
            for backend in BACKENDS:
                path = arguments.raw_dir / f"{scenario}_run_{run_id}_{backend}.csv"
                rows = read_csv(path)
                if scenario == "lifecycle":
                    validate_lifecycle(rows, backend, run_id, path)
                else:
                    validate_resident(rows, backend, run_id, path, Path(str(path) + ".cleanup.csv"))
                datasets[scenario][run_id][backend] = rows
                combined[scenario].extend(rows)
            merged = merge_pair(
                scenario,
                run_id,
                datasets[scenario][run_id]["fft_thrust"],
                datasets[scenario][run_id]["dlfft"],
                comparison_dir / f"{scenario}_run_{run_id}_comparison.csv",
            )
            merged_by_scenario[scenario][run_id] = merged
            summary.extend(statistics_for_pair(scenario, run_id, datasets[scenario][run_id], merged))
        write_rows(arguments.output_dir / f"{scenario}_all_runs.csv", combined[scenario])

    write_rows(arguments.output_dir / "backend_memory_summary.csv", summary)
    samples = sample_rows("lifecycle", datasets["lifecycle"]) + sample_rows("resident", datasets["resident"])
    write_rows(arguments.output_dir / "report_samples.csv", samples)
    generate_charts(
        arguments.output_dir,
        datasets["lifecycle"],
        datasets["resident"],
        merged_by_scenario["lifecycle"],
    )
    lifecycle_class, lifecycle_text = classify(summary, "lifecycle")
    resident_class, resident_text = classify(summary, "resident")
    manifest = {
        "fft_length": FFT_LENGTH,
        "batch": BATCH,
        "dtype": DTYPE,
        "warmup_rounds": 100,
        "formal_iterations": ITERATIONS,
        "independent_runs": len(RUN_IDS),
        "lifecycle_classification": lifecycle_class,
        "lifecycle_conclusion": lifecycle_text,
        "resident_classification": resident_class,
        "resident_conclusion": resident_text,
        "correctness": verification,
    }
    with (arguments.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_report(
        arguments.output_dir / "FFT后端统一内存稳定性测试报告.md",
        verification,
        summary,
        datasets["lifecycle"],
        datasets["resident"],
    )
    print(
        "[FFT-BACKEND-MEMORY][ANALYZE] "
        f"lifecycle={lifecycle_class} resident={resident_class} status=complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
