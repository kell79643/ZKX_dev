#!/usr/bin/env python3
"""导出 Task2 正式数据，或从已有 CSV 生成六步效果图与评价报告。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "Task2" / "evidence" / "visualization"

from visualization_common import (
    EPSILON,
    StepEvaluation,
    configure_matplotlib,
    downsample_xy,
    finite,
    grade_higher,
    grade_lower,
    materialize_best_config,
    metric,
    prepare_run_paths,
    read_feature_csv,
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


def causal_filter(taps: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.convolve(values, taps, mode="full")[: values.size]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def normalize_abs(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(np.abs(values)))
    return np.abs(values) / maximum if maximum > 0 else np.zeros_like(values)


def resample(values: np.ndarray, count: int) -> np.ndarray:
    if values.size == count:
        return values.astype(float, copy=False)
    if values.size == 1:
        return np.full(count, float(values[0]))
    return np.interp(
        np.linspace(0.0, 1.0, count),
        np.linspace(0.0, 1.0, values.size),
        values,
    )


def evaluate(data, features: Dict[str, np.ndarray], meta: Dict[str, float]):
    wave = data["waveforms"]
    echo = data["echo"]
    filtered = data["filter"]
    taps = data["taps"]["value"]
    smoothing = data["smoothing"]
    weights = data["weights"]["value"]
    target_filtered = causal_filter(taps, echo["target"])
    sources = smoothing["source_sample"].astype(int)
    target_crop = target_filtered[sources]
    target_smoothed = causal_filter(weights, target_crop)
    steps: List[StepEvaluation] = []

    waveform_names = ["chirp", "gausspulse", "sawtooth", "square"]
    ranges = [float(np.ptp(wave[name])) for name in waveform_names]
    target_power = float(np.mean(echo["target"] ** 2))
    interference_power = float(np.mean(echo["interference"] ** 2))
    noise_power = float(np.mean(echo["noise"] ** 2))
    total_power = target_power + interference_power + noise_power
    echo_snr = 10.0 * np.log10(
        max(target_power, EPSILON)
        / max(interference_power + noise_power, EPSILON)
    )
    metrics = [
        metric(
            "{} 波形幅值跨度".format(name),
            value,
            "Amplitude",
            "max(x)-min(x)",
            ">=1.5优秀；>=0.5良好",
            grade_higher(value, 1.5, 0.5),
        )
        for name, value in zip(waveform_names, ranges)
    ]
    metrics.extend(
        [
            metric("回波 SNR", echo_snr, "dB", "10log10(P_target/(P_interference+P_noise))", ">=20优秀；>=10良好", grade_higher(echo_snr, 20, 10)),
            metric("目标功率比例", target_power / total_power, "1", "P_target/P_components", ">=0.8优秀；>=0.5良好", grade_higher(target_power / total_power, 0.8, 0.5)),
            metric("干扰功率比例", interference_power / total_power, "1", "P_interference/P_components", "<=0.1优秀；<=0.3良好", grade_lower(interference_power / total_power, 0.1, 0.3)),
            metric("噪声功率比例", noise_power / total_power, "1", "P_noise/P_components", "<=0.05优秀；<=0.15良好", grade_lower(noise_power / total_power, 0.05, 0.15)),
        ]
    )
    steps.append(
        StepEvaluation(
            "Step1 波形生成与回波模拟",
            metrics,
            "回波信噪比（SNR）{:.2f} dB｜目标功率占比 {:.1f}%".format(
                echo_snr, 100 * target_power / total_power
            ),
            "四种波形幅值跨度均不小于 {:.3f}；回波 SNR {:.2f} dB，目标功率占比 {:.2f}%。".format(
                min(ranges), echo_snr, 100 * target_power / total_power
            ),
        )
    )

    input_residual = filtered["input_echo"] - echo["target"]
    output_residual = filtered["filtered"] - target_filtered
    snr_before = 20.0 * np.log10(rms(echo["target"]) / max(rms(input_residual), EPSILON))
    snr_after = 20.0 * np.log10(rms(target_filtered) / max(rms(output_residual), EPSILON))
    snr_gain = snr_after - snr_before
    mse = float(np.mean((filtered["input_echo"] - filtered["filtered"]) ** 2))
    nmse = mse / max(float(np.mean(filtered["input_echo"] ** 2)), EPSILON)
    nmse_grade = grade_lower(nmse, 0.01, 0.05)
    metrics = [
        metric("滤波前 SNR", snr_before, "dB", "20log10(RMS(target)/RMS(residual))", ">=20优秀；>=10良好", grade_higher(snr_before, 20, 10)),
        metric("滤波后 SNR", snr_after, "dB", "20log10(RMS(target_filtered)/RMS(residual))", ">=20优秀；>=10良好", grade_higher(snr_after, 20, 10)),
        metric("SNR 增益", snr_gain, "dB", "SNR_after-SNR_before", ">=6优秀；>=1良好", grade_higher(snr_gain, 6, 1)),
        metric("滤波前后均方差 MSE", mse, "Amplitude^2", "mean((input-output)^2)", "结合归一化MSE判断", nmse_grade),
        metric("归一化滤波 MSE", nmse, "1", "MSE/mean(input^2)", "<=0.01优秀；<=0.05良好", nmse_grade),
    ]
    steps.append(
        StepEvaluation(
            "Step2 FIR 滤波",
            metrics,
            "滤波前后 SNR：{:.2f}→{:.2f} dB（{:+.2f} dB）｜归一化均方误差（NMSE）{:.4g}".format(
                snr_before, snr_after, snr_gain, nmse
            ),
            "FIR 滤波使 SNR 从 {:.2f} dB 变为 {:.2f} dB，增益 {:.2f} dB，NMSE {:.4g}。".format(
                snr_before, snr_after, snr_gain, nmse
            ),
        )
    )

    smooth_mse = float(np.mean((smoothing["cropped_input"] - smoothing["smoothed"]) ** 2))
    smooth_nmse = smooth_mse / max(float(np.mean(smoothing["cropped_input"] ** 2)), EPSILON)
    noise_before = smoothing["cropped_input"] - target_crop
    noise_after = smoothing["smoothed"] - target_smoothed
    variance_before = float(np.var(noise_before))
    variance_after = float(np.var(noise_after))
    reduction = 100.0 * (variance_before - variance_after) / max(variance_before, EPSILON)
    peak_ratio = float(np.max(np.abs(smoothing["smoothed"]))) / max(
        float(np.max(np.abs(smoothing["cropped_input"]))), EPSILON
    )
    smooth_grade = grade_lower(smooth_nmse, 0.01, 0.05)
    metrics = [
        metric("B样条平滑 MSE", smooth_mse, "Amplitude^2", "mean((input-smoothed)^2)", "结合归一化MSE判断", smooth_grade),
        metric("B样条平滑归一化 MSE", smooth_nmse, "1", "MSE/mean(input^2)", "<=0.01优秀；<=0.05良好", smooth_grade),
        metric("噪声方差降低比例", reduction, "%", "100*(var_before-var_after)/var_before", ">=30优秀；>=10良好", grade_higher(reduction, 30, 10)),
        metric("峰值保持比例", peak_ratio, "1", "max|smoothed|/max|input|", ">=0.8优秀；>=0.5良好", grade_higher(peak_ratio, 0.8, 0.5)),
    ]
    steps.append(
        StepEvaluation(
            "Step3 B样条平滑",
            metrics,
            "噪声方差变化 {:+.1f}%｜峰值保持 {:.1f}%｜NMSE {:.4g}".format(
                reduction, 100 * peak_ratio, smooth_nmse
            ),
            "B 样条平滑后噪声方差变化 {:.2f}%，峰值保留 {:.2f}%，NMSE {:.4g}。".format(
                reduction, 100 * peak_ratio, smooth_nmse
            ),
        )
    )

    families = ["fm_demod", "correlation", "spectral_envelope", "wavelet_envelope"]
    bundle = features["fused_bundle"]
    correlations = []
    metrics = []
    for family in families:
        values = normalize_abs(resample(features[family], bundle.size))
        correlation = float(np.corrcoef(values, bundle)[0, 1])
        correlations.append(correlation)
        metrics.append(
            metric(
                "{} 与融合特征相关系数".format(family),
                correlation,
                "1",
                "corr(feature,fused_bundle)",
                ">=0.8优秀；>=0.5良好",
                grade_higher(correlation, 0.8, 0.5),
            )
        )
    spectrogram = data["spectrogram"]
    peak = int(np.argmax(spectrogram["value"]))
    peak_frequency = abs(float(spectrogram["frequency_hz"][peak]))
    peak_time = float(spectrogram["time_seconds"][peak])
    source_time = max(0.0, peak_time - meta["target_delay_samples"] / meta["sample_rate_hz"])
    chirp_duration = max(meta["samples"] / meta["sample_rate_hz"], 1.0)
    expected_frequency = 20.0 + 50.0 * min(source_time / chirp_duration, 1.0)
    frequency_error = abs(peak_frequency - expected_frequency)
    frequency_bin = meta["sample_rate_hz"] / 64.0
    wavelet = normalize_abs(resample(features["wavelet_envelope"], smoothing["smoothed"].size))
    signal_envelope = normalize_abs(smoothing["smoothed"])
    wavelet_mse = float(np.mean((wavelet - signal_envelope) ** 2))
    metrics.extend(
        [
            metric("频谱峰值频率误差", frequency_error, "Hz", "|f_peak-f_theory|", "<=1 bin优秀；<=2 bin良好", grade_lower(frequency_error, frequency_bin, 2 * frequency_bin)),
            metric("小波包络重构代理 MSE", wavelet_mse, "1", "mean((norm(CWT_env)-norm(|signal|))^2)", "<=0.01优秀；<=0.05良好", grade_lower(wavelet_mse, 0.01, 0.05)),
        ]
    )
    steps.append(
        StepEvaluation(
            "Step4 多域特征融合",
            metrics,
            "特征相关系数 {:.2f}～{:.2f}｜频谱误差 {:.2f} Hz｜小波包络 MSE {:.4g}".format(
                min(correlations), max(correlations), frequency_error, wavelet_mse
            ),
            "四域特征相关系数范围 {:.3f}~{:.3f}；频谱误差 {:.2f} Hz，小波代理 MSE {:.4g}。".format(
                min(correlations), max(correlations), frequency_error, wavelet_mse
            ),
        )
    )

    extrema = data["extrema"]
    strongest = int(np.argmax(extrema["value"]))
    detected_index = float(extrema["index"][strongest])
    filter_delay = (taps.size - 1) / 2.0
    spline_delay = (weights.size - 1) / 2.0
    truth = (
        0.5 * (meta["step3_output_samples"] - 1)
        + meta["target_delay_samples"]
        + filter_delay
        + spline_delay
    )
    location_error = abs(detected_index - truth)
    detection_accuracy = float(location_error <= 3)
    density = extrema["index"].size / meta["step3_output_samples"]
    location_grade = grade_lower(location_error, 1, 3)
    metrics = [
        metric("最强峰定位误差", location_error, "sample", "|peak_detect-peak_true|", "<=1优秀；<=3良好", location_grade),
        metric("单目标峰值检测准确率", detection_accuracy, "1", "N_correct/N_target", ">=0.99优秀；>=0.90良好", grade_higher(detection_accuracy, 0.99, 0.90)),
        metric("局部极大值密度", density, "1", "N_extrema/N_samples", "描述性指标", location_grade),
    ]
    steps.append(
        StepEvaluation(
            "Step5 峰值定位",
            metrics,
            "最强峰定位误差 {:.2f} 点｜命中率 {:.0f}%｜局部极大值 {} 个".format(
                location_error, 100 * detection_accuracy, extrema["index"].size
            ),
            "在 {} 个局部极大值中，最强峰定位误差 {:.2f} sample，关键点密度 {:.2f}%。".format(
                extrema["index"].size, location_error, 100 * density
            ),
        )
    )

    kalman = data["kalman"]
    position_scale = meta["step3_output_samples"] - 1
    estimate = float(kalman["estimate"][0] * position_scale)
    truth = float(kalman["truth"][0] * position_scale)
    observations = kalman["observation"] * position_scale
    covariance = float(kalman["covariance"][0] * position_scale**2)
    final_error = abs(estimate - truth)
    state_rmse = final_error
    observation_rmse = float(np.sqrt(np.mean((observations - truth) ** 2)))
    improvement = observation_rmse - state_rmse
    metrics = [
        metric("最终状态 RMSE", state_rmse, "sample", "sqrt(mean((estimate-truth)^2))", "<=1优秀；<=3良好", grade_lower(state_rmse, 1, 3)),
        metric("最终收敛绝对误差", final_error, "sample", "|estimate-truth|", "<=1优秀；<=3良好", grade_lower(final_error, 1, 3)),
        metric("观测值 RMSE", observation_rmse, "sample", "sqrt(mean((observation-truth)^2))", "最终RMSE应更小", grade_lower(observation_rmse, 1, 3)),
        metric("Kalman RMSE 改善量", improvement, "sample", "RMSE_observation-RMSE_final", ">0表示改善", grade_higher(improvement, 3, 0)),
        metric("最终协方差", covariance, "sample^2", "P_final*(N-1)^2", "<=1优秀；<=9良好", (grade_lower(covariance, 1, 9)[0], finite(covariance) and 0 < covariance <= 9)),
    ]
    steps.append(
        StepEvaluation(
            "Step6 Kalman参数估计",
            metrics,
            "最终估计 {:.2f} 点｜理论真值 {:.2f} 点｜均方根误差（RMSE）{:.2f} 点".format(
                estimate, truth, state_rmse
            ),
            "Kalman 最终估计 {:.2f} sample，真值 {:.2f} sample，状态 RMSE {:.2f} sample。".format(
                estimate, truth, state_rmse
            ),
        )
    )
    return steps


def plot(data, features, meta, steps, output: Path):
    plt = configure_matplotlib()
    time_axis = data["waveforms"]["time_seconds"]
    seconds = time_axis
    time_zoom = (seconds >= 5.5) & (seconds <= 7.5)
    source_zoom = (seconds >= 0.40) & (seconds <= 0.50)
    source_series = (
        ("chirp", "chirp", "tab:blue"),
        ("sawtooth", "sawtooth", "tab:orange"),
        ("gausspulse", "gausspulse", "tab:green"),
        ("square", "square", "tab:red"),
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name, label, color in source_series:
        axes[0, 0].plot(seconds[source_zoom], data["waveforms"][name][source_zoom], label=label, color=color)
    axes[0, 0].set(title="四类源信号局部波形", xlabel="时间（s）", ylabel="幅值", xlim=(0.40, 0.50))
    axes[0, 0].legend(ncol=2, loc="upper left")
    for name, label, color in (
        ("target", "目标回波分量", "tab:blue"),
        ("interference", "干扰分量", "tab:orange"),
        ("noise", "噪声分量", "tab:green"),
    ):
        axes[0, 1].plot(seconds[time_zoom], data["echo"][name][time_zoom], label=label, color=color)
    axes[0, 1].set(title="回波组成分量", xlabel="时间（s）", ylabel="幅值", xlim=(5.5, 7.5))
    axes[0, 1].legend()
    axes[1, 0].plot(seconds[time_zoom], data["echo"]["echo"][time_zoom], label="混合接收回波", color="tab:blue")
    axes[1, 0].set(title="混合接收回波局部波形", xlabel="时间（s）", ylabel="幅值", xlim=(5.5, 7.5))
    axes[1, 0].legend()
    axes[1, 1].plot(seconds[time_zoom], data["echo"]["target"][time_zoom], label="目标回波分量", color="tab:blue")
    axes[1, 1].plot(seconds[time_zoom], data["echo"]["echo"][time_zoom], label="混合接收回波", color="tab:orange", alpha=0.85)
    axes[1, 1].set(title="目标回波到达区间细节", xlabel="时间（s）", ylabel="幅值", xlim=(5.5, 7.5))
    axes[1, 1].legend()
    figure.suptitle("任务二 步骤1：波形生成与回波模拟\n" + steps[0].annotation)
    save_figure(figure, output, "task2_step1_waveform_and_echo.png")

    filtered = data["filter"]
    taps = data["taps"]
    difference = filtered["filtered"] - filtered["input_echo"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for key, label, color in (("input_echo", "滤波前", "tab:blue"), ("filtered", "滤波后", "tab:orange")):
        x, y = downsample_xy(seconds, filtered[key])
        axes[0, 0].plot(x, y, label=label, color=color)
    axes[0, 0].set(title="有限冲激响应（FIR）滤波前后全局对比", xlabel="时间（s）", ylabel="幅值")
    axes[0, 0].legend()
    axes[0, 1].plot(seconds[time_zoom], filtered["input_echo"][time_zoom], label="滤波前", color="tab:blue")
    axes[0, 1].plot(seconds[time_zoom], filtered["filtered"][time_zoom], label="滤波后", color="tab:orange")
    axes[0, 1].set(title="目标到达区间的滤波细节", xlabel="时间（s）", ylabel="幅值", xlim=(5.5, 7.5))
    axes[0, 1].legend()
    axes[1, 0].plot(taps["relative_sample"], taps["value"])
    axes[1, 0].set(title="实际 FIR 滤波器抽头系数", xlabel="相对采样点", ylabel="系数")
    axes[1, 1].plot(seconds[time_zoom], difference[time_zoom], color="tab:blue")
    axes[1, 1].set(title="滤波引起的信号改变量", xlabel="时间（s）", ylabel="滤波后 - 滤波前", xlim=(5.5, 7.5))
    figure.suptitle("任务二 步骤2：有限冲激响应（FIR）滤波效果\n" + steps[1].annotation)
    save_figure(figure, output, "task2_step2_filter.png")

    smooth = data["smoothing"]
    sample_axis = smooth["source_sample"]
    detail = (sample_axis >= 3300) & (sample_axis <= 4300)
    center = int(np.argmax(np.abs(smooth["smoothed"])))
    roi = slice(max(0, center - 256), min(sample_axis.size, center + 257))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for key, label, color in (("cropped_input", "平滑前", "tab:blue"), ("smoothed", "B 样条平滑后", "tab:orange")):
        axes[0, 0].plot(sample_axis[detail], smooth[key][detail], label=label, color=color)
    axes[0, 0].set(title="B 样条平滑前后局部对比", xlabel="源信号采样点", ylabel="幅值", xlim=(3300, 4300))
    axes[0, 0].legend()
    axes[0, 1].plot(sample_axis[roi], smooth["cropped_input"][roi], label="平滑前", color="tab:blue")
    axes[0, 1].plot(sample_axis[roi], smooth["smoothed"][roi], label="B 样条平滑后", color="tab:orange")
    axes[0, 1].set(title="最强响应邻域细节", xlabel="源信号采样点", ylabel="幅值")
    axes[0, 1].legend()
    axes[1, 0].stem(data["weights"]["offset"], data["weights"]["value"])
    axes[1, 0].set(title="实际 B 样条平滑权重", xlabel="相对偏移", ylabel="权重")
    residual = smooth["smoothed"] - smooth["cropped_input"]
    axes[1, 1].plot(sample_axis[detail], residual[detail], color="tab:blue")
    axes[1, 1].set(title="B 样条平滑改变量", xlabel="源信号采样点", ylabel="平滑后 - 平滑前", xlim=(3300, 4300))
    figure.suptitle("任务二 步骤3：B 样条平滑效果\n" + steps[2].annotation)
    save_figure(figure, output, "task2_step3_bspline_smoothing.png")

    bundle = features["fused_bundle"]
    peak = int(np.argmax(bundle))
    roi = slice(max(0, peak - 128), min(bundle.size, peak + 129))
    feature_series = (
        ("wavelet_envelope", "小波包络特征", "tab:red"),
        ("fm_demod", "调频解调特征", "tab:blue"),
        ("fused_bundle", "加权融合特征", "tab:purple"),
        ("spectral_envelope", "频谱包络特征", "tab:green"),
        ("correlation", "时域相关特征", "tab:orange"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    for name, label, color in feature_series:
        values = resample(features[name], bundle.size)
        x, y = downsample_xy(np.arange(bundle.size), normalize_abs(values))
        axes[0, 0].plot(x, y, label=label, color=color)
    axes[0, 0].set(title="五类归一化多域特征总览", xlabel="采样点序号", ylabel="归一化幅值")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].plot(np.arange(bundle.size)[roi], bundle[roi], marker="o", markersize=2, color="tab:purple")
    axes[0, 1].scatter(peak, bundle[peak], color="red", marker="*")
    axes[0, 1].set(title="融合特征主峰局部细节", xlabel="采样点序号", ylabel="融合特征值")
    feature_zoom = np.arange(11200, min(11601, bundle.size))
    for name, label, color in (
        ("correlation", "时域相关特征", "tab:orange"),
        ("fm_demod", "调频解调特征", "tab:blue"),
    ):
        values = resample(features[name], bundle.size)
        axes[1, 0].plot(feature_zoom, values[feature_zoom], label=label, color=color)
    axes[1, 0].set(title="调制域与时域相关特征局部对比", xlabel="采样点序号", ylabel="特征值", xlim=(11200, 11600))
    axes[1, 0].legend()
    for name, label, color in (
        ("spectral_envelope", "频谱包络特征", "tab:blue"),
        ("wavelet_envelope", "小波包络特征", "tab:orange"),
    ):
        values = resample(features[name], bundle.size)
        x, y = downsample_xy(np.arange(bundle.size), values)
        axes[1, 1].plot(x, y, label=label, color=color)
    axes[1, 1].set(title="频谱包络与小波包络特征", xlabel="采样点序号", ylabel="特征值")
    axes[1, 1].legend()
    spectrogram = data["spectrogram"]
    frequency_count = np.unique(spectrogram["frequency_index"]).size
    frame_count = np.unique(spectrogram["frame_index"]).size
    spectral_grid = spectrogram["value"].reshape(frequency_count, frame_count)
    freq_axis = spectrogram["frequency_hz"].reshape(frequency_count, frame_count)[:, 0]
    frequency_order = np.argsort(freq_axis)
    freq_axis = freq_axis[frequency_order]
    spectral_grid = spectral_grid[frequency_order]
    spec_time = spectrogram["time_seconds"].reshape(frequency_count, frame_count)[0]
    image = axes[2, 0].imshow(10 * np.log10(np.maximum(spectral_grid, 1e-12)), aspect="auto", origin="lower", extent=[spec_time[0], spec_time[-1], freq_axis[0], freq_axis[-1]])
    axes[2, 0].set(title="短时频谱图", xlabel="时间（s）", ylabel="频率（Hz）")
    figure.colorbar(image, ax=axes[2, 0], label="功率（dB）")
    cwt = data["cwt"]
    width_count = np.unique(cwt["width_index"]).size
    sample_count = np.unique(cwt["sample"]).size
    cwt_grid = cwt["value"].reshape(width_count, sample_count)
    cwt_time = cwt["time_seconds"].reshape(width_count, sample_count)[0]
    widths = cwt["width"].reshape(width_count, sample_count)[:, 0]
    cwt_magnitude = np.abs(cwt_grid)
    cwt_limit = float(np.quantile(cwt_magnitude, 0.98))
    cwt_ticks = np.linspace(0.0, cwt_limit, 11)
    image = axes[2, 1].imshow(
        cwt_magnitude,
        aspect="auto",
        origin="lower",
        extent=[cwt_time[0], cwt_time[-1], widths[0], widths[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=cwt_limit,
    )
    axes[2, 1].set(title="连续小波变换（CWT）时频响应", xlabel="时间（s）", ylabel="Ricker 小波尺度")
    figure.colorbar(
        image,
        ax=axes[2, 1],
        label="响应幅值",
        ticks=cwt_ticks,
        extend="max",
    )
    figure.suptitle("任务二 步骤4：多域特征提取与融合（原版布局）\n" + steps[3].annotation)
    save_figure(figure, output, "task2_step4_multidomain_features.png")

    # 四域结构化对比版：保留原版布局，供人工比较后再决定最终保留版本。
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    for name, label, color in feature_series:
        values = resample(features[name], bundle.size)
        x, y = downsample_xy(np.arange(bundle.size), normalize_abs(values))
        axes[0, 0].plot(x, y, label=label, color=color)
    axes[0, 0].set(title="五类归一化特征总览", xlabel="采样点序号", ylabel="归一化幅值")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].plot(np.arange(bundle.size)[roi], bundle[roi], marker="o", markersize=2, color="tab:purple")
    axes[0, 1].scatter(peak, bundle[peak], color="red", marker="*", label="融合特征主峰")
    axes[0, 1].set(title="融合特征主峰局部图", xlabel="采样点序号", ylabel="融合特征值")
    axes[0, 1].legend()
    fm_values = resample(features["fm_demod"], bundle.size)
    x, y = downsample_xy(np.arange(bundle.size), fm_values)
    axes[1, 0].plot(x, y, color="tab:blue", label="调频解调特征")
    axes[1, 0].set(title="调制域特征：瞬时频率解调", xlabel="采样点序号", ylabel="瞬时相位差")
    axes[1, 0].legend()
    correlation_values = resample(features["correlation"], bundle.size)
    axes[1, 1].plot(
        feature_zoom,
        correlation_values[feature_zoom],
        color="tab:orange",
        label="时域相关特征",
    )
    axes[1, 1].set(
        title="时域特征：相关主峰局部响应",
        xlabel="采样点序号",
        ylabel="相关特征值",
        xlim=(11200, 11600),
    )
    axes[1, 1].legend()
    mean_spectrum = np.mean(spectral_grid, axis=1)
    max_spectrum = np.max(spectral_grid, axis=1)
    axes[2, 0].plot(freq_axis, 10 * np.log10(np.maximum(mean_spectrum, 1e-12)), color="tab:blue", label="时间平均功率谱")
    axes[2, 0].plot(freq_axis, 10 * np.log10(np.maximum(max_spectrum, 1e-12)), color="tab:orange", label="时间最大功率谱")
    axes[2, 0].set(title="频域特征：平均/最大功率谱", xlabel="频率（Hz）", ylabel="功率（dB）")
    axes[2, 0].legend()
    image = axes[2, 1].imshow(
        cwt_magnitude,
        aspect="auto",
        origin="lower",
        extent=[cwt_time[0], cwt_time[-1], widths[0], widths[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=cwt_limit,
    )
    axes[2, 1].set(title="时频联合域特征：连续小波变换（CWT）", xlabel="时间（s）", ylabel="Ricker 小波尺度")
    figure.colorbar(
        image,
        ax=axes[2, 1],
        label="响应幅值",
        ticks=cwt_ticks,
        extend="max",
    )
    figure.suptitle("任务二 步骤4：调制域、时域、频域及时频联合域特征\n" + steps[3].annotation)
    save_figure(figure, output, "task2_step4_four_domain_features.png")

    extrema = data["extrema"]
    strongest = int(np.argmax(extrema["value"]))
    strongest_index = int(extrema["index"][strongest])
    roi = slice(max(0, strongest_index - 180), min(bundle.size, strongest_index + 181))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x, y = downsample_xy(np.arange(bundle.size), bundle)
    axes[0].plot(x, y, label="融合特征", color="tab:blue")
    axes[0].scatter(extrema["index"], extrema["value"], s=6, alpha=0.35, label="局部极大值")
    axes[0].scatter(strongest_index, extrema["value"][strongest], color="red", marker="*", s=120, label="最强特征点")
    axes[0].set(title="融合特征关键点检测", xlabel="采样点序号", ylabel="特征值")
    axes[0].legend()
    axes[1].plot(np.arange(bundle.size)[roi], bundle[roi])
    inside = (extrema["index"] >= roi.start) & (extrema["index"] < roi.stop)
    axes[1].scatter(extrema["index"][inside], extrema["value"][inside], s=18)
    axes[1].scatter(strongest_index, extrema["value"][strongest], color="red", marker="*", s=120)
    axes[1].set(title="最强特征点邻域", xlabel="采样点序号", ylabel="特征值")
    figure.suptitle("任务二 步骤5：相对极值定位\n" + steps[4].annotation)
    save_figure(figure, output, "task2_step5_feature_points.png")

    kalman = data["kalman"]
    scale = meta["step3_output_samples"] - 1
    observations = kalman["observation"] * scale
    estimate = kalman["estimate"][0] * scale
    truth = kalman["truth"][0] * scale
    indices = kalman["observation_index"]
    roi = slice(max(0, indices.size - 224), indices.size)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].plot(indices, observations, marker="o", markersize=2, label="观测位置")
    axes[0].axhline(estimate, color="tab:orange", label="最终状态估计")
    axes[0].axhline(truth, color="tab:green", linestyle="--", label="理论真值")
    axes[0].set(title="Kalman 滤波观测序列与最终状态", xlabel="观测序号", ylabel="位置（采样点）")
    axes[0].legend()
    axes[1].plot(indices[roi], observations[roi], marker="o", markersize=3, label="观测位置")
    axes[1].axhline(estimate, color="tab:orange", label="最终状态估计")
    axes[1].axhline(truth, color="tab:green", linestyle="--", label="理论真值")
    axes[1].set(title="末段观测及收敛状态", xlabel="观测序号", ylabel="位置（采样点）")
    axes[1].legend()
    figure.suptitle("任务二 步骤6：Kalman 状态估计\n" + steps[5].annotation)
    save_figure(figure, output, "task2_step6_kalman_estimation.png")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="", help="已构建的 task2_pipeline 路径")
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
    paths = prepare_run_paths(EVIDENCE_ROOT, "Task2", mode)
    scale_id = "existing_data"
    pipeline_seconds = 0.0
    if mode != "plot":
        scale_id, _ = materialize_best_config(REPOSITORY_ROOT, "Task2", paths.config)
        pipeline = resolve_pipeline(arguments.pipeline, REPOSITORY_ROOT, "Task2")
        pipeline_seconds = run_pipeline(pipeline, paths.config, paths, REPOSITORY_ROOT)
    manifest = read_manifest(paths.data / "task2_manifest.csv")
    meta = require_manifest(
        manifest,
        {
            "task": "Task2",
            "source": "cusignal_cpp_zq500_actual_pipeline",
            "completed_steps": "6",
            "step2_input": "step1.echo",
            "step3_input": "step2.filtered",
            "step4_input": "step3.smoothed+step3.reference",
            "step5_input": "step4.feature_bundle",
            "step6_input": "step5.extrema+step4.feature_bundle",
        },
        [
            "samples",
            "filter_taps",
            "step3_crop_each",
            "step3_output_samples",
            "kalman_observations",
            "sample_rate_hz",
            "target_delay_samples",
        ],
    )
    data = {
        "waveforms": read_numeric_csv(paths.data / "task2_step1_waveforms.csv"),
        "echo": read_numeric_csv(paths.data / "task2_step1_echo.csv"),
        "filter": read_numeric_csv(paths.data / "task2_step2_filter.csv"),
        "taps": read_numeric_csv(paths.data / "task2_step2_taps.csv"),
        "smoothing": read_numeric_csv(paths.data / "task2_step3_smoothing.csv"),
        "weights": read_numeric_csv(paths.data / "task2_step3_spline_weights.csv"),
        "spectrogram": read_numeric_csv(paths.data / "task2_step4_spectrogram.csv"),
        "cwt": read_numeric_csv(paths.data / "task2_step4_cwt.csv"),
        "extrema": read_numeric_csv(paths.data / "task2_step5_extrema.csv"),
        "kalman": read_numeric_csv(paths.data / "task2_step6_kalman.csv"),
    }
    features = read_feature_csv(paths.data / "task2_step4_features.csv")
    if data["waveforms"]["sample"].size != int(meta["samples"]):
        raise RuntimeError("Task2 waveform row count does not match manifest")
    if data["smoothing"]["sample"].size != int(meta["step3_output_samples"]):
        raise RuntimeError("Task2 smoothing row count does not match manifest")
    if data["kalman"]["observation_index"].size != int(meta["kalman_observations"]):
        raise RuntimeError("Task2 Kalman row count does not match manifest")
    if mode == "export":
        total_seconds = time.monotonic() - total_begin
        write_run_summary(paths, "Task2", scale_id, pipeline_seconds, total_seconds)
        print(
            "[TASK2][VISUALIZATION][EXPORT] scale={} pipeline_seconds={:.3f} "
            "total_seconds={:.3f} status=PASS".format(
                scale_id, pipeline_seconds, total_seconds
            )
        )
        return 0
    steps = evaluate(data, features, meta)
    write_evaluation("task2", "Task2 多域特征提取定位算法", steps, paths.evaluation)
    plot(data, features, meta, steps, paths.figures)
    total_seconds = time.monotonic() - total_begin
    if mode == "full":
        write_run_summary(paths, "Task2", scale_id, pipeline_seconds, total_seconds)
        print(
            "[TASK2][VISUALIZATION] scale={} pipeline_seconds={:.3f} "
            "total_seconds={:.3f} status=PASS".format(
                scale_id, pipeline_seconds, total_seconds
            )
        )
    else:
        print(
            "[TASK2][VISUALIZATION][PLOT] data_source=existing_csv "
            "total_seconds={:.3f} status=PASS".format(total_seconds)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print("[TASK2][VISUALIZATION][FAIL] {}".format(error), file=sys.stderr)
        raise SystemExit(1)
