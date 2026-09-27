#!/usr/bin/env python3
"""导出 Task1 正式数据，或从已有 CSV 生成五步效果图与评价报告。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "Task1" / "evidence" / "visualization"

from visualization_common import (
    EPSILON,
    Metric,
    StepEvaluation,
    configure_matplotlib,
    downsample_xy,
    grade_higher,
    grade_lower,
    materialize_best_config,
    metric,
    prepare_run_paths,
    read_manifest,
    read_numeric_csv,
    require_manifest,
    resolve_pipeline,
    run_pipeline,
    save_figure,
    worst_grade,
    write_evaluation,
    write_run_summary,
)


SPEED_OF_LIGHT = 299792458.0


def mainlobe_bounds(values: np.ndarray, peak: int) -> Tuple[int, int]:
    left = peak
    while left > 0 and values[left - 1] <= values[left]:
        left -= 1
    right = peak
    while right + 1 < values.size and values[right + 1] <= values[right]:
        right += 1
    if left == right:
        left = max(0, peak - 2)
        right = min(values.size - 1, peak + 2)
    return left, right


def cut_metrics(axis: np.ndarray, values: np.ndarray) -> Tuple[int, float, float]:
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    threshold = peak / np.sqrt(2.0)
    left = peak_index
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    right = peak_index
    while right + 1 < values.size and values[right + 1] >= threshold:
        right += 1
    width_bins = right - left + 1
    spacing = float(np.median(np.abs(np.diff(axis))))
    mask = np.abs(np.arange(values.size) - peak_index) > max(2, width_bins)
    side_peak = float(np.max(values[mask])) if np.any(mask) else 0.0
    pslr = 20.0 * np.log10(max(side_peak, EPSILON) / max(peak, EPSILON))
    return width_bins, width_bins * spacing, pslr


def evaluate(data: Dict[str, Dict[str, np.ndarray]], meta: Dict[str, float]):
    steps: List[StepEvaluation] = []

    echo = data["echo"]
    signal_power = np.mean(echo["noiseless_real"] ** 2 + echo["noiseless_imag"] ** 2)
    noise_power = np.mean(echo["noise_real"] ** 2 + echo["noise_imag"] ** 2)
    snr = 10.0 * np.log10(max(signal_power, EPSILON) / max(noise_power, EPSILON))
    metrics = [
        metric(
            "信噪比 SNR",
            snr,
            "dB",
            "10log10(P_signal/P_noise)",
            ">=20优秀；>=10良好",
            grade_higher(snr, 20, 10),
        )
    ]
    steps.append(
        StepEvaluation(
            "Step1 接收信号模拟",
            metrics,
            "信噪比（SNR）{:.2f} dB".format(snr),
            "目标回波 SNR 为 {:.2f} dB，综合评价为{}。".format(
                snr, worst_grade(metrics)
            ),
        )
    )

    compressed = data["compressed"]
    grid = compressed["magnitude"].reshape(
        int(meta["num_pulses"]), int(meta["samples_per_pulse"])
    )
    profile = np.max(grid, axis=0)
    peak_index = int(np.argmax(profile))
    main_peak = float(profile[peak_index])
    left, right = mainlobe_bounds(profile, peak_index)
    main_width = right - left + 1
    sidelobes = profile.copy()
    sidelobes[left : right + 1] = 0.0
    pslr = 20.0 * np.log10(
        max(float(np.max(sidelobes)), EPSILON) / max(main_peak, EPSILON)
    )
    resolution_m = SPEED_OF_LIGHT / (2.0 * meta["bandwidth_hz"])
    range_bin_m = SPEED_OF_LIGHT / (2.0 * meta["sample_rate_hz"])
    resolution_bins = resolution_m / range_bin_m
    before_peak = np.max(echo["echo_real"] ** 2 + echo["echo_imag"] ** 2)
    gain = 10.0 * np.log10(max(main_peak**2, EPSILON) / max(before_peak, EPSILON))
    pslr_grade = grade_lower(pslr, -30, -20)
    resolution_grade = grade_lower(resolution_bins, 2, 4)
    gain_grade = grade_higher(gain, 20, 10)
    metrics = [
        metric("峰值旁瓣比 PSLR", pslr, "dB", "20log10(A_side/A_main)", "<=-30优秀；<=-20良好", pslr_grade),
        metric("理论距离分辨率", resolution_m, "m", "c/(2B)", "<=2 bin优秀；<=4 bin良好", resolution_grade),
        metric("距离分辨率采样栅格数", resolution_bins, "bin", "DeltaR/range_bin", "<=2优秀；<=4良好", resolution_grade),
        metric("压缩后主瓣宽度", main_width, "sample", "N_mainlobe", "<=3优秀；<=8良好", grade_lower(main_width, 3, 8)),
        metric("压缩增益", gain, "dB", "10log10(P_after/P_before)", ">=20优秀；>=10良好", gain_grade),
    ]
    steps.append(
        StepEvaluation(
            "Step2 脉冲压缩",
            metrics,
            "主瓣宽度 {} 点｜峰值旁瓣比（PSLR）{:.2f} dB｜压缩增益 {:.2f} dB".format(
                main_width, pslr, gain
            ),
            "主瓣由 {} sample 缩窄为 {} sample，PSLR {:.2f} dB，压缩增益 {:.2f} dB。".format(
                int(meta["pulse_samples"]), main_width, pslr, gain
            ),
        )
    )

    rd = data["range_doppler"]
    index = int(np.argmax(rd["magnitude"]))
    peak_amplitude = float(rd["magnitude"][index])
    range_est = float(rd["range_meters"][index])
    doppler_est = float(rd["doppler_hz"][index])
    range_true = meta["target_delay_samples"] * range_bin_m
    doppler_true = meta["target_doppler_hz"]
    range_error = abs(range_est - range_true)
    velocity_error = abs(
        (doppler_est - doppler_true) * SPEED_OF_LIGHT / (2.0 * meta["carrier_frequency_hz"])
    )
    velocity_bin = (
        meta["prf_hz"]
        / meta["num_pulses"]
        * SPEED_OF_LIGHT
        / (2.0 * meta["carrier_frequency_hz"])
    )
    background = ~(
        (np.abs(rd["doppler_index"] - meta["target_doppler_bin"]) <= 1)
        & (np.abs(rd["range_bin"] - meta["target_delay_samples"]) <= 2)
    )
    background_power = np.median(rd["magnitude"][background] ** 2)
    peak_background = 10.0 * np.log10(
        max(peak_amplitude**2, EPSILON) / max(background_power, EPSILON)
    )
    range_grade = grade_lower(range_error, range_bin_m, 2 * range_bin_m)
    velocity_grade = grade_lower(velocity_error, velocity_bin, 2 * velocity_bin)
    metrics = [
        metric("目标峰值距离", range_est / 1000.0, "km", "argmax |RD|", "结合距离误差判断", range_grade),
        metric("目标峰值多普勒频率", doppler_est, "Hz", "argmax |RD|", "结合速度误差判断", velocity_grade),
        metric("距离估计绝对误差", range_error, "m", "|R_est-R_true|", "<=1 bin优秀；<=2 bin良好", range_grade),
        metric("速度估计绝对误差", velocity_error, "m/s", "|v_est-v_true|", "<=1 bin优秀；<=2 bin良好", velocity_grade),
        metric("距离-多普勒峰值背景比", peak_background, "dB", "10log10(P_peak/median(P_bg))", ">=20优秀；>=10良好", grade_higher(peak_background, 20, 10)),
    ]
    steps.append(
        StepEvaluation(
            "Step3 距离-多普勒处理",
            metrics,
            "目标峰值 {:.3f} km / {:.1f} Hz｜峰值背景比 {:.2f} dB".format(
                range_est / 1000.0, doppler_est, peak_background
            ),
            "最大响应位于 {:.3f} km、{:.1f} Hz；距离误差 {:.3f} m，峰值背景比 {:.2f} dB。".format(
                range_est / 1000.0, doppler_est, range_error, peak_background
            ),
        )
    )

    cfar = data["cfar"]
    valid = (
        (cfar["doppler_index"] >= 3)
        & (cfar["doppler_index"] < meta["num_pulses"] - 3)
        & (cfar["range_bin"] >= 8)
        & (cfar["range_bin"] < meta["samples_per_pulse"] - 8)
    )
    target = (
        (np.abs(cfar["doppler_index"] - meta["target_doppler_bin"]) <= 1)
        & (np.abs(cfar["range_bin"] - meta["target_delay_samples"]) <= 2)
    )
    detected = cfar["detection"] > 0
    pd = float(np.any(detected & target & valid))
    false_count = int(np.count_nonzero(detected & valid & ~target))
    non_target = int(np.count_nonzero(valid & ~target))
    detected_count = int(np.count_nonzero(detected & valid))
    pfa = false_count / max(non_target, 1)
    false_fraction = false_count / max(detected_count, 1)
    candidate = detected & valid & (
        np.abs(cfar["doppler_index"] - meta["target_doppler_bin"]) <= 1
    )
    distance_error = (
        float(np.min(np.abs(cfar["range_meters"][candidate] - range_true)))
        if np.any(candidate)
        else float("inf")
    )
    metrics = [
        metric("单场景检测概率 Pd", pd, "1", "N_correct/N_target", ">=0.99优秀；>=0.90良好", grade_higher(pd, 0.99, 0.90)),
        metric("经验虚警概率 Pfa", pfa, "1", "N_false/N_non-target", "<=1e-4优秀；<=1e-3良好", grade_lower(pfa, 1e-4, 1e-3)),
        metric("虚警占检测结果比例", false_fraction, "1", "N_false/N_detections", "<=0.1优秀；<=0.3良好", grade_lower(false_fraction, 0.1, 0.3)),
        metric("检测距离绝对误差", distance_error, "m", "min|R_detect-R_true|", "<=1 bin优秀；<=2 bin良好", grade_lower(distance_error, range_bin_m, 2 * range_bin_m)),
    ]
    steps.append(
        StepEvaluation(
            "Step4 CFAR 检测",
            metrics,
            "检测概率（Pd）{:.3f}｜虚警概率（Pfa）{:.4g}｜距离误差 {:.2f} m".format(
                pd, pfa, distance_error
            ),
            "CFAR 单场景 Pd={:.3f}、经验 Pfa={:.4g}，检测距离误差 {:.2f} m。".format(
                pd, pfa, distance_error
            ),
        )
    )

    delay = data["delay_cut"]
    doppler = data["doppler_cut"]
    delay_bins, delay_width, delay_pslr = cut_metrics(delay["delay_seconds"], delay["value"])
    doppler_bins, doppler_width, doppler_pslr = cut_metrics(doppler["doppler_hz"], doppler["value"])
    metrics = [
        metric("时延方向半功率主瓣宽度", delay_width * 1e6, "us", "N_3dB*delay_bin", "<=2 bin优秀；<=3 bin良好", grade_lower(delay_bins, 2, 3)),
        metric("多普勒方向半功率主瓣宽度", doppler_width, "Hz", "N_3dB*doppler_bin", "<=3 bin优秀；<=5 bin良好", grade_lower(doppler_bins, 3, 5)),
        metric("时延方向最大旁瓣比", delay_pslr, "dB", "20log10(A_side/A_main)", "<=-20优秀；<=-13良好", grade_lower(delay_pslr, -20, -13)),
        metric("多普勒方向最大旁瓣比", doppler_pslr, "dB", "20log10(A_side/A_main)", "<=-20优秀；<=-13良好", grade_lower(doppler_pslr, -20, -13)),
    ]
    steps.append(
        StepEvaluation(
            "Step5 模糊函数分析",
            metrics,
            "时延/多普勒主瓣 {}/{} 个栅格｜PSLR {:.2f}/{:.2f} dB".format(
                delay_bins, doppler_bins, delay_pslr, doppler_pslr
            ),
            "模糊函数主瓣宽度为时延 {} bin、多普勒 {} bin，PSLR 为 {:.2f}/{:.2f} dB。".format(
                delay_bins, doppler_bins, delay_pslr, doppler_pslr
            ),
        )
    )
    return steps


def plot(data: Dict[str, Dict[str, np.ndarray]], meta: Dict[str, float], steps, output: Path):
    plt = configure_matplotlib()
    pulses = int(meta["num_pulses"])
    samples = int(meta["samples_per_pulse"])

    wave = data["waveform"]
    echo = data["echo"]
    echo_mag = echo["echo_magnitude"].reshape(pulses, samples)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(wave["time_seconds"] * 1e6, wave["real"], label="实部")
    axes[0, 0].plot(wave["time_seconds"] * 1e6, wave["imag"], label="虚部")
    axes[0, 0].plot(wave["time_seconds"] * 1e6, wave["magnitude"], label="包络幅值")
    axes[0, 0].set(title="发射线性调频（LFM）波形", xlabel="时间（μs）", ylabel="幅值")
    axes[0, 0].legend()
    fast_time = np.arange(samples) / meta["sample_rate_hz"] * 1e6
    axes[0, 1].plot(fast_time, echo_mag[0], label="接收幅值")
    axes[0, 1].axvline(meta["target_delay_samples"] / meta["sample_rate_hz"] * 1e6, color="r", linestyle="--", label="目标时延")
    axes[0, 1].set(title="首个脉冲的接收回波", xlabel="快时间（μs）", ylabel="幅值")
    axes[0, 1].legend()
    image = axes[1, 0].imshow(echo_mag, aspect="auto", origin="lower", extent=[fast_time[0], fast_time[-1], 0, pulses - 1])
    axes[1, 0].set(title="多脉冲接收回波幅值", xlabel="快时间（μs）", ylabel="脉冲序号")
    figure.colorbar(image, ax=axes[1, 0], label="幅值")
    roi = slice(max(0, int(meta["target_delay_samples"]) - 16), min(samples, int(meta["target_delay_samples"]) + 17))
    axes[1, 1].plot(fast_time[roi], np.mean(echo_mag[:, roi], axis=0), marker="o")
    axes[1, 1].set(title="目标时延邻域的脉冲平均响应", xlabel="快时间（μs）", ylabel="平均幅值")
    figure.suptitle("任务一 步骤1：发射波形与接收回波\n" + steps[0].annotation)
    save_figure(figure, output, "task1_step1_echo.png")

    compressed = data["compressed"]
    compressed_mag = compressed["magnitude"].reshape(pulses, samples)
    range_km = compressed["range_meters"][:samples] / 1000.0
    profile = np.max(compressed_mag, axis=0)
    peak = int(np.argmax(profile))
    roi = slice(max(0, peak - 18), min(samples, peak + 19))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(range_km, echo_mag[0], label="脉冲压缩前")
    axes[0, 0].plot(range_km, compressed_mag[0], label="脉冲压缩后")
    axes[0, 0].set(title="首个脉冲压缩前后对比", xlabel="距离（km）", ylabel="幅值")
    axes[0, 0].legend()
    image = axes[0, 1].imshow(20 * np.log10(np.maximum(compressed_mag, 1e-10)), aspect="auto", origin="lower", extent=[range_km[0], range_km[-1], 0, pulses - 1])
    axes[0, 1].set(title="脉冲压缩距离像", xlabel="距离（km）", ylabel="脉冲序号")
    figure.colorbar(image, ax=axes[0, 1], label="幅值（dB）")
    axes[1, 0].plot(range_km, profile)
    axes[1, 0].axvspan(range_km[roi.start], range_km[roi.stop - 1], color="r", alpha=0.15)
    axes[1, 0].set(title="最大距离向响应", xlabel="距离（km）", ylabel="幅值")
    axes[1, 1].plot(range_km[roi], profile[roi], marker="o")
    axes[1, 1].set(title="压缩主瓣局部细节", xlabel="距离（km）", ylabel="幅值")
    figure.suptitle("任务一 步骤2：脉冲压缩效果\n" + steps[1].annotation)
    save_figure(figure, output, "task1_step2_pulse_compression.png")

    rd = data["range_doppler"]
    rd_mag = rd["magnitude"].reshape(pulses, samples)
    doppler_axis = rd["doppler_hz"][::samples]
    order = np.argsort(doppler_axis)
    doppler_axis = doppler_axis[order]
    rd_mag = rd_mag[order]
    peak_row, peak_col = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
    rows = slice(max(0, peak_row - 8), min(pulses, peak_row + 9))
    cols = slice(max(0, peak_col - 18), min(samples, peak_col + 19))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    image = axes[0].imshow(20 * np.log10(np.maximum(rd_mag, 1e-10)), aspect="auto", origin="lower", extent=[range_km[0], range_km[-1], doppler_axis[0], doppler_axis[-1]])
    axes[0].scatter(range_km[peak_col], doppler_axis[peak_row], color="red", marker="*")
    axes[0].set(title="距离–多普勒响应", xlabel="距离（km）", ylabel="多普勒频率（Hz）")
    figure.colorbar(image, ax=axes[0], label="幅值（dB）")
    image = axes[1].imshow(20 * np.log10(np.maximum(rd_mag[rows, cols], 1e-10)), aspect="auto", origin="lower", extent=[range_km[cols.start], range_km[cols.stop - 1], doppler_axis[rows.start], doppler_axis[rows.stop - 1]])
    axes[1].scatter(range_km[peak_col], doppler_axis[peak_row], color="red", marker="*")
    axes[1].set(title="目标峰值邻域", xlabel="距离（km）", ylabel="多普勒频率（Hz）")
    figure.colorbar(image, ax=axes[1], label="幅值（dB）")
    figure.suptitle("任务一 步骤3：距离–多普勒定位\n" + steps[2].annotation)
    save_figure(figure, output, "task1_step3_range_doppler.png")

    cfar = data["cfar"]
    power = cfar["power"].reshape(pulses, samples)[order]
    threshold = cfar["threshold"].reshape(pulses, samples)[order]
    detection = cfar["detection"].reshape(pulses, samples)[order]
    target_row = int(np.argmin(np.abs(doppler_axis - meta["target_doppler_hz"])))
    cols = slice(max(0, int(meta["target_delay_samples"]) - 8), min(samples, int(meta["target_delay_samples"]) + 9))
    rows = slice(max(0, target_row - 4), min(pulses, target_row + 5))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    image = axes[0, 0].imshow(detection, aspect="auto", origin="lower", extent=[range_km[0], range_km[-1], doppler_axis[0], doppler_axis[-1]])
    axes[0, 0].set(title="单元平均恒虚警率（CA-CFAR）检测结果", xlabel="距离（km）", ylabel="多普勒频率（Hz）")
    figure.colorbar(image, ax=axes[0, 0], label="检测标记")
    axes[0, 1].semilogy(range_km, np.maximum(power[target_row], 1e-12), label="单元功率")
    axes[0, 1].semilogy(range_km, np.maximum(threshold[target_row], 1e-12), label="检测门限")
    marks = detection[target_row] > 0
    axes[0, 1].scatter(range_km[marks], power[target_row, marks], label="检出单元", s=20)
    axes[0, 1].set(title="目标多普勒切片的功率与门限", xlabel="距离（km）", ylabel="功率")
    axes[0, 1].yaxis.set_major_formatter(lambda value, position: "{:.0e}".format(value))
    axes[0, 1].legend()
    image = axes[1, 0].imshow(detection[rows, cols], aspect="auto", origin="lower", extent=[range_km[cols.start], range_km[cols.stop - 1], doppler_axis[rows.start], doppler_axis[rows.stop - 1]])
    axes[1, 0].set(title="目标检测邻域", xlabel="距离（km）", ylabel="多普勒频率（Hz）")
    figure.colorbar(image, ax=axes[1, 0], label="检测标记")
    axes[1, 1].semilogy(range_km[cols], np.maximum(power[target_row, cols], 1e-12), label="单元功率")
    axes[1, 1].semilogy(range_km[cols], np.maximum(threshold[target_row, cols], 1e-12), label="检测门限")
    axes[1, 1].set(title="目标邻域功率–门限细节", xlabel="距离（km）", ylabel="功率")
    axes[1, 1].yaxis.set_major_formatter(lambda value, position: "{:.0e}".format(value))
    axes[1, 1].legend()
    figure.suptitle("任务一 步骤4：恒虚警率检测（CFAR）效果\n" + steps[3].annotation)
    save_figure(figure, output, "task1_step4_cfar.png")

    ambiguity = data["ambiguity"]
    delay = data["delay_cut"]
    doppler = data["doppler_cut"]
    delay_axis = np.unique(ambiguity["delay_seconds"]) * 1e6
    doppler_axis = np.unique(ambiguity["doppler_hz"]) / 1000.0
    ambiguity_grid = ambiguity["value"].reshape(delay_axis.size, doppler_axis.size)
    peak_row, peak_col = np.unravel_index(np.argmax(ambiguity_grid), ambiguity_grid.shape)
    rows = slice(max(0, peak_row - 14), min(delay_axis.size, peak_row + 15))
    cols = slice(max(0, peak_col - 14), min(doppler_axis.size, peak_col + 15))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    image = axes[0, 0].imshow(ambiguity_grid, aspect="auto", origin="lower", extent=[doppler_axis[0], doppler_axis[-1], delay_axis[0], delay_axis[-1]])
    axes[0, 0].set(title="二维模糊函数", xlabel="多普勒频率（kHz）", ylabel="时延（μs）")
    figure.colorbar(image, ax=axes[0, 0], label="归一化幅值")
    image = axes[0, 1].imshow(ambiguity_grid[rows, cols], aspect="auto", origin="lower", extent=[doppler_axis[cols.start], doppler_axis[cols.stop - 1], delay_axis[rows.start], delay_axis[rows.stop - 1]])
    axes[0, 1].set(title="模糊函数主峰局部细节", xlabel="多普勒频率（kHz）", ylabel="时延（μs）")
    figure.colorbar(image, ax=axes[0, 1], label="归一化幅值")
    axes[1, 0].plot(doppler["doppler_hz"] / 1000.0, doppler["value"])
    axes[1, 0].set(title="零时延多普勒切片", xlabel="多普勒频率（kHz）", ylabel="归一化幅值")
    axes[1, 1].plot(delay["delay_seconds"] * 1e6, delay["value"])
    axes[1, 1].set(title="零多普勒时延切片", xlabel="时延（μs）", ylabel="归一化幅值")
    figure.suptitle("任务一 步骤5：模糊函数分析\n" + steps[4].annotation)
    save_figure(figure, output, "task1_step5_ambiguity.png")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="", help="已构建的 task1_pipeline 路径")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--export-only",
        action="store_true",
        help="仅在 ZQ500 上运行正式 pipeline 并导出、校验 CSV",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="仅从已有 CSV 重新评价和绘图，不运行 pipeline",
    )
    return parser.parse_args()


def run() -> int:
    arguments = parse_arguments()
    mode = "export" if arguments.export_only else "plot" if arguments.plot_only else "full"
    total_begin = time.monotonic()
    paths = prepare_run_paths(EVIDENCE_ROOT, "Task1", mode)
    scale_id = "existing_data"
    pipeline_seconds = 0.0
    if mode != "plot":
        scale_id, _ = materialize_best_config(REPOSITORY_ROOT, "Task1", paths.config)
        pipeline = resolve_pipeline(arguments.pipeline, REPOSITORY_ROOT, "Task1")
        pipeline_seconds = run_pipeline(pipeline, paths.config, paths, REPOSITORY_ROOT)
    manifest = read_manifest(paths.data / "task1_manifest.csv")
    meta = require_manifest(
        manifest,
        {
            "task": "Task1",
            "source": "cusignal_cpp_zq500_actual_pipeline",
            "completed_steps": "5",
            "step2_input": "step1.echo+step1.waveform",
            "step3_input": "step2.compressed",
            "step4_input": "step3.range_doppler",
            "step5_input": "step1.waveform",
        },
        [
            "num_pulses",
            "samples_per_pulse",
            "pulse_samples",
            "sample_rate_hz",
            "prf_hz",
            "target_delay_samples",
            "target_doppler_bin",
            "target_doppler_hz",
            "bandwidth_hz",
            "carrier_frequency_hz",
        ],
    )
    data = {
        "waveform": read_numeric_csv(paths.data / "task1_step1_waveform.csv"),
        "echo": read_numeric_csv(paths.data / "task1_step1_echo.csv"),
        "compressed": read_numeric_csv(paths.data / "task1_step2_compressed.csv"),
        "range_doppler": read_numeric_csv(paths.data / "task1_step3_range_doppler.csv"),
        "cfar": read_numeric_csv(paths.data / "task1_step4_cfar.csv"),
        "ambiguity": read_numeric_csv(paths.data / "task1_step5_ambiguity.csv"),
        "delay_cut": read_numeric_csv(paths.data / "task1_step5_delay_cut.csv"),
        "doppler_cut": read_numeric_csv(paths.data / "task1_step5_doppler_cut.csv"),
    }
    expected = int(meta["num_pulses"] * meta["samples_per_pulse"])
    for name in ("echo", "compressed", "range_doppler", "cfar"):
        if next(iter(data[name].values())).size != expected:
            raise RuntimeError("{} row count does not match manifest".format(name))
    if mode == "export":
        total_seconds = time.monotonic() - total_begin
        write_run_summary(paths, "Task1", scale_id, pipeline_seconds, total_seconds)
        print(
            "[TASK1][VISUALIZATION][EXPORT] scale={} pipeline_seconds={:.3f} "
            "total_seconds={:.3f} status=PASS".format(
                scale_id, pipeline_seconds, total_seconds
            )
        )
        return 0
    steps = evaluate(data, meta)
    write_evaluation("task1", "Task1 雷达目标检测与多普勒分析", steps, paths.evaluation)
    plot(data, meta, steps, paths.figures)
    total_seconds = time.monotonic() - total_begin
    if mode == "full":
        write_run_summary(paths, "Task1", scale_id, pipeline_seconds, total_seconds)
        print(
            "[TASK1][VISUALIZATION] scale={} pipeline_seconds={:.3f} "
            "total_seconds={:.3f} status=PASS".format(
                scale_id, pipeline_seconds, total_seconds
            )
        )
    else:
        print(
            "[TASK1][VISUALIZATION][PLOT] data_source=existing_csv "
            "total_seconds={:.3f} status=PASS".format(total_seconds)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print("[TASK1][VISUALIZATION][FAIL] {}".format(error), file=sys.stderr)
        raise SystemExit(1)
