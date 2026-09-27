# -*- coding: utf-8 -*-
"""
fig_accuracy_step_linechart.py
任务一/任务二各step的精度折线图（MSE/RMSE/相对误差L2/相对误差Linf）
采用双纵坐标，对数刻度，严格按CSV原始数据取中位数
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体设置（防止乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 路径配置 =====================
BASE_DIR = r"c:\Users\asus\Desktop\0817结果（含耗时）\03_任务结果"
TASK1_ROOT = os.path.join(BASE_DIR, "task1")
TASK2_ROOT = os.path.join(BASE_DIR, "task2")
OUT_DIR = r"c:\Users\asus\Desktop\0817结果（含耗时）\03_任务结果"
if not os.path.isdir(OUT_DIR):
    OUT_DIR = r"c:\Users\asus\Desktop\0817结果（含耗时）"

METRIC_NAMES = ['mse', 'rmse', 'relative_l2', 'relative_linf']


# ===================== 数据收集 =====================
def collect_task_metrics(cases_root, task_prefix, n_step, metric_names):
    """
    读取单个任务下所有case的*_accuracy_details_fft_thrust_formal.csv
    对每个step的每个指标，收集所有有效样本（排除NA/discrete）
    返回：stepMedians(n_step,4), stepCounts(n_step,)
    """
    step_buckets = [[[] for _ in range(len(metric_names))] for _ in range(n_step)]

    # 列出所有 case 目录：task1_s* 或 task2_s*
    pattern = os.path.join(cases_root, f"{task_prefix}_s*")
    case_dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    print(f"  Found {len(case_dirs)} case directories")

    n_used = 0
    for case_dir in case_dirs:
        csv_file = os.path.join(case_dir, f"{task_prefix}_accuracy_details_fft_thrust_formal.csv")
        if not os.path.isfile(csv_file):
            continue
        n_used += 1

        try:
            T = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  读取失败 {csv_file}: {e}")
            continue

        needed = ['target', 'output_kind'] + metric_names
        if not all(c in T.columns for c in needed):
            print(f"  列缺失，跳过：{csv_file}")
            continue

        # 解析 step
        step_ids = []
        for t in T['target'].astype(str):
            m = re.search(r'\.step(\d+)\.', t)
            step_ids.append(int(m.group(1)) if m else np.nan)
        step_ids = np.array(step_ids)

        for r in range(len(T)):
            sid = step_ids[r]
            if np.isnan(sid) or sid < 1 or sid > n_step:
                continue
            kind = str(T['output_kind'].iloc[r]).strip().lower()
            if kind == 'discrete':
                continue
            for m_idx, mn in enumerate(metric_names):
                val = T[mn].iloc[r]
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    continue
                if np.isfinite(v) and v >= 0:
                    step_buckets[int(sid) - 1][m_idx].append(v)

    # 计算中位数
    step_medians = np.full((n_step, len(metric_names)), np.nan, dtype=float)
    step_counts = np.zeros(n_step, dtype=int)
    for s in range(n_step):
        step_counts[s] = len(step_buckets[s][0])
        for m in range(len(metric_names)):
            b = step_buckets[s][m]
            if len(b) > 0:
                step_medians[s, m] = np.median(b)
    return step_medians, step_counts


# ===================== 打印表格 =====================
def fmt_med(v):
    if np.isnan(v):
        return "NA            "
    elif v == 0:
        return "0.0000e+00    "
    else:
        return f"{v:.4e}    "


def print_table(name, medians, counts, n_step):
    print(f"\n===== {name} 各 step 中位数 =====")
    print(f"{'step':<5}{'count':<6}{'MSE':<15}{'RMSE':<15}{'rel_l2':<15}{'rel_linf':<15}")
    for s in range(n_step):
        print(f"S{s+1:<4}{counts[s]:<6}"
              f"{fmt_med(medians[s,0])}"
              f"{fmt_med(medians[s,1])}"
              f"{fmt_med(medians[s,2])}"
              f"{fmt_med(medians[s,3])}")


# ===================== 绘图辅助 =====================
def prepare_for_log(medians):
    nan_mask = np.isnan(medians)
    zero_mask = (medians == 0) & ~nan_mask
    refs = medians[~nan_mask & ~zero_mask]
    if refs.size == 0:
        floor_val = 1e-18
    else:
        floor_val = np.min(refs) * 0.01
    plot_data = medians.copy()
    plot_data[zero_mask] = floor_val
    return plot_data, zero_mask, nan_mask


def sci_str(v):
    if np.isnan(v):
        return "NA"
    if v == 0:
        return "0"
    # 紧凑科学计数法（2位小数）
    return f"{v:.2e}"


def fmt_med2(v):
    return sci_str(v)


# ===================== 主绘图 =====================
def main():
    # ---- 读数据 ----
    print("开始读取 Task1 精度数据...")
    t1_median, t1_count = collect_task_metrics(TASK1_ROOT, "task1", 5, METRIC_NAMES)
    print("Task1 读取完成。")
    print("开始读取 Task2 精度数据...")
    t2_median, t2_count = collect_task_metrics(TASK2_ROOT, "task2", 6, METRIC_NAMES)
    print("Task2 读取完成。")

    # ---- 打印校验 ----
    print_table("Task1", t1_median, t1_count, 5)
    print_table("Task2", t2_median, t2_count, 6)

    # ---- 颜色：低饱和度、4色区分度好 ----
    #   蓝 / 橙 / 绿 / 紫
    colors = [
        (0.22, 0.42, 0.70, 1.0),   # 蓝：MSE
        (0.88, 0.54, 0.16, 1.0),   # 橙：RMSE
        (0.27, 0.65, 0.36, 1.0),   # 绿：rel_l2
        (0.58, 0.32, 0.73, 1.0),   # 紫：rel_linf
    ]
    marker_styles = ['o', 's', '^', 'd']
    line_width = 1.8
    marker_size = 7

    # ---- 画布：上下两个子图 ----
    fig = plt.figure(figsize=(10, 11), dpi=120)
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 1, hspace=0.35)

    # ================= 上方子图 Task1 =================
    ax1L = fig.add_subplot(gs[0])
    ax1R = ax1L.twinx()
    x1 = np.arange(1, 5 + 1)
    t1_plot, t1_zero_mask, t1_nan_mask = prepare_for_log(t1_median)

    # 左轴：MSE + RMSE
    h_t1_mse,  = ax1L.plot(x1, t1_plot[:, 0], color=colors[0], linestyle='-',
                           linewidth=line_width, marker=marker_styles[0],
                           markersize=marker_size, markerfacecolor=colors[0],
                           markeredgecolor='white', label='MSE')
    h_t1_rmse, = ax1L.plot(x1, t1_plot[:, 1], color=colors[1], linestyle='-',
                           linewidth=line_width, marker=marker_styles[1],
                           markersize=marker_size, markerfacecolor=colors[1],
                           markeredgecolor='white', label='RMSE')
    # 右轴：rel_l2 + rel_linf
    h_t1_rl2,  = ax1R.plot(x1, t1_plot[:, 2], color=colors[2], linestyle='-',
                           linewidth=line_width, marker=marker_styles[2],
                           markersize=marker_size, markerfacecolor=colors[2],
                           markeredgecolor='white', label='relative_l2')
    h_t1_rli,  = ax1R.plot(x1, t1_plot[:, 3], color=colors[3], linestyle='-',
                           linewidth=line_width, marker=marker_styles[3],
                           markersize=marker_size, markerfacecolor=colors[3],
                           markeredgecolor='white', label='relative_linf')
    format_subplot(ax1L, ax1R, x1, 5, "任务一（Task1）",
                   t1_median, t1_plot, t1_zero_mask, t1_nan_mask, colors)

    # ================= 下方子图 Task2 =================
    ax2L = fig.add_subplot(gs[1])
    ax2R = ax2L.twinx()
    x2 = np.arange(1, 6 + 1)
    t2_plot, t2_zero_mask, t2_nan_mask = prepare_for_log(t2_median)

    h_t2_mse,  = ax2L.plot(x2, t2_plot[:, 0], color=colors[0], linestyle='-',
                           linewidth=line_width, marker=marker_styles[0],
                           markersize=marker_size, markerfacecolor=colors[0],
                           markeredgecolor='white', label='MSE')
    h_t2_rmse, = ax2L.plot(x2, t2_plot[:, 1], color=colors[1], linestyle='-',
                           linewidth=line_width, marker=marker_styles[1],
                           markersize=marker_size, markerfacecolor=colors[1],
                           markeredgecolor='white', label='RMSE')
    h_t2_rl2,  = ax2R.plot(x2, t2_plot[:, 2], color=colors[2], linestyle='-',
                           linewidth=line_width, marker=marker_styles[2],
                           markersize=marker_size, markerfacecolor=colors[2],
                           markeredgecolor='white', label='relative_l2')
    h_t2_rli,  = ax2R.plot(x2, t2_plot[:, 3], color=colors[3], linestyle='-',
                           linewidth=line_width, marker=marker_styles[3],
                           markersize=marker_size, markerfacecolor=colors[3],
                           markeredgecolor='white', label='relative_linf')
    format_subplot(ax2L, ax2R, x2, 6, "任务二（Task2）",
                   t2_median, t2_plot, t2_zero_mask, t2_nan_mask, colors)

    # ================= 顶部全局图例 =================
    handles = [h_t1_mse, h_t1_rmse, h_t1_rl2, h_t1_rli]
    labels = ['MSE', 'RMSE', '相对误差 L2', '相对误差 Linf']
    leg = fig.legend(handles, labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=4,
                     frameon=False,
                     fontsize=10)

    # ---- 保存 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = "accuracy_step_linechart"
    png_path = os.path.join(OUT_DIR, base_name + ".png")
    pdf_path = os.path.join(OUT_DIR, base_name + ".pdf")
    svg_path = os.path.join(OUT_DIR, base_name + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n已保存 PNG (300dpi): {png_path}")
    try:
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"已保存 PDF: {pdf_path}")
    except Exception as e:
        print(f"PDF 保存失败：{e}")
    try:
        fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        print(f"已保存 SVG: {svg_path}")
    except Exception as e:
        print(f"SVG 保存失败：{e}")

    plt.show()


def format_subplot(axL, axR, xVals, nStep, subplotTitle,
                   medians, plotData, zeroMask, nanMask, colors):
    axL.set_xlim(0.5, nStep + 0.5)
    axL.set_xticks(xVals)
    axL.set_xticklabels([f'Step {s}' for s in xVals], fontsize=10)

    axL.grid(False, axis='x')
    axL.grid(True, axis='y', which='major', linestyle='--', linewidth=0.6, alpha=0.6)
    axL.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.4, alpha=0.4)
    axL.tick_params(axis='both', labelsize=10)

    # 双对数坐标
    axL.set_yscale('log')
    axR.set_yscale('log')

    axL.set_ylabel("MSE / RMSE（左轴，对数刻度）", fontsize=11, fontweight='bold')
    axR.set_ylabel("相对误差 L2 / Linf（右轴，对数刻度）", fontsize=11, fontweight='bold')
    axR.tick_params(axis='both', labelsize=10)

    axL.set_title(subplotTitle, fontsize=12, fontweight='bold', pad=12)

    # 坐标轴范围
    allLeft = plotData[:, [0, 1]]
    allLeft = allLeft[np.isfinite(allLeft) & (allLeft > 0)]
    if allLeft.size > 0:
        yMinL = np.min(allLeft) * 0.5
        yMaxL = np.max(allLeft) * 3.0
        axL.set_ylim(yMinL, yMaxL)
    else:
        axL.set_ylim(1e-18, 1e-10)

    allRight = plotData[:, [2, 3]]
    allRight = allRight[np.isfinite(allRight) & (allRight > 0)]
    if allRight.size > 0:
        yMinR = np.min(allRight) * 0.5
        yMaxR = np.max(allRight) * 3.0
        axR.set_ylim(yMinR, yMaxR)
    else:
        axR.set_ylim(1e-18, 1e-10)

    # 数据点数值标注
    # 对数坐标的相对偏移：yPos * 10^(delta) 形式
    log_rangeL = np.log10(axL.get_ylim()[1]) - np.log10(axL.get_ylim()[0])
    log_rangeR = np.log10(axR.get_ylim()[1]) - np.log10(axR.get_ylim()[0])
    offLup  = 10 ** (log_rangeL * 0.07)
    offLdn  = 10 ** (log_rangeL * 0.05)
    offRup  = 10 ** (log_rangeR * 0.07)
    offRdn  = 10 ** (log_rangeR * 0.05)

    for s_idx, s in enumerate(xVals):
        # ---- 左轴 1：MSE ----
        if nanMask[s_idx, 0]:
            yMidL = np.sqrt(axL.get_ylim()[0] * axL.get_ylim()[1])
            axL.text(s, yMidL, 'NA', color=colors[0], fontsize=8,
                     fontweight='bold', ha='center', va='center')
        else:
            yPos = plotData[s_idx, 0]
            lbl = '0' if zeroMask[s_idx, 0] else sci_str(medians[s_idx, 0])
            axL.text(s, yPos * offLup, lbl, color=colors[0], fontsize=8,
                     ha='center', va='bottom')
        # ---- 左轴 2：RMSE ----
        if not nanMask[s_idx, 1]:
            yPos = plotData[s_idx, 1]
            lbl = '0' if zeroMask[s_idx, 1] else sci_str(medians[s_idx, 1])
            axL.text(s, yPos / offLdn, lbl, color=colors[1], fontsize=8,
                     ha='center', va='top')
        # ---- 右轴 1：relative_l2 ----
        if nanMask[s_idx, 2]:
            yMidR = np.sqrt(axR.get_ylim()[0] * axR.get_ylim()[1])
            axR.text(s, yMidR, 'NA', color=colors[2], fontsize=8,
                     fontweight='bold', ha='center', va='center')
        else:
            yPos = plotData[s_idx, 2]
            lbl = '0' if zeroMask[s_idx, 2] else sci_str(medians[s_idx, 2])
            axR.text(s, yPos * offRup, lbl, color=colors[2], fontsize=8,
                     ha='center', va='bottom')
        # ---- 右轴 2：relative_linf ----
        if not nanMask[s_idx, 3]:
            yPos = plotData[s_idx, 3]
            lbl = '0' if zeroMask[s_idx, 3] else sci_str(medians[s_idx, 3])
            axR.text(s, yPos / offRdn, lbl, color=colors[3], fontsize=8,
                     ha='center', va='top')


if __name__ == "__main__":
    main()
