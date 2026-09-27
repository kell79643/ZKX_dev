# Task2 Step4 cusignal_cpp 实现逻辑

## 1. 职责与版本

当前step接口、CPU reference、GPU wrapper、私有kernel、线程职责和跨步衔接。

- 当前ZKX SHA：`26865d99d04035b51aa8bebcfad7392b8832af72`
- 分支：`snapshot/all-current-20260716`
- 提交时间：`2026-08-19T16:06:49+08:00`
- 每段源码另绑定实际SHA。
- tracked源码状态：clean
- 本轮只静态阅读。

## 2. 源码覆盖与唯一归属

|源码|SHA|
|---|---|
|`Task2/task2_gpu_cpu/step4/step4.h`|`26865d99d04035b51aa8bebcfad7392b8832af72`|
|`Task2/task2_gpu_cpu/step4/step4.cu`|`26865d99d04035b51aa8bebcfad7392b8832af72`|
|`Task2/task2_gpu_cpu/accuracy/validation/step4_validation.h`|`26865d99d04035b51aa8bebcfad7392b8832af72`|
|`Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`|`26865d99d04035b51aa8bebcfad7392b8832af72`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `Task2/task2_gpu_cpu/step4/step4.h`

完整SHA `26865d99d04035b51aa8bebcfad7392b8832af72`。

```cpp
#pragma once

#include "../task2_common.h"

namespace task2 {

// <学习注释：Step4 入口函数 —— 多域特征提取与融合流水线。
//
//  【函数总体处理方法】
//  run_step4 负责从 Step3 平滑信号中提取四类特征并加权融合。处理流程分五个阶段：
//    阶段1（CPU Prep）：校验上游输入、计算 frame/hop/frames、准备 CWT workspace
//    阶段2（H2D）：将 smoothed 和 reference_for_correlation 拷贝到 GPU 显存
//    阶段3（GPU compute）：依次执行 analytic_signal_kernel（私有 glue）、fm_demod_device、
//                        correlate_device、spectrogram_device、cwt_device
//    阶段4（D2H）：将四类特征回传 Host
//    阶段5（CPU Post）：spectral_envelope、wavelet_envelope 提取包络，make_bundle 融合
//
//  【核心数学公式 —— 多域特征融合模型】
//    对采样点 n（n = 0, 1, ..., N-1），最终融合特征为：
//      F[n] = γ_fm * F̃_fm[n] + γ_corr * F̃_corr[n] + γ_spec * F̃_spec[n] + γ_wave * F̃_wave[n]
//
//    其中各分量定义为：
//      F̃_fm[n]    = normalize_abs(resample_linear(fm_feature, count))
//      F̃_corr[n]  = normalize_abs(resample_linear(correlation_feature, count))
//      F̃_spec[n]  = normalize_abs(resample_linear(spectral_feature, count))
//      F̃_wave[n]  = normalize_abs(resample_linear(wavelet_feature, count))
//
//    符号说明：
//      γ_fm    = fusion_weight_fm          ：FM 特征融合权重
//      γ_corr  = fusion_weight_correlation ：相关特征融合权重
//      γ_spec  = fusion_weight_spectral    ：谱图特征融合权重
//      γ_wave  = fusion_weight_wavelet     ：小波特征融合权重
//      约束 Σγ = 1 且 γ ≥ 0（凸组合）
//
//  【物理意义】
//  从调制域（FM 解调）、时域（互相关）、频域（STFT 谱图）和时频域（CWT）四个维度
//  提取信号特征，再通过重采样统一长度、幅值归一化后加权融合。这是 T2-R4
//  （多域特征提取）条款的核心实现，输出 feature_bundle 将作为 Step5 峰值定位的输入。>
StepEvidence run_step4(PipelineState& state);

}  // namespace task2

```

### 3.2 `Task2/task2_gpu_cpu/step4/step4.cu`

完整SHA `26865d99d04035b51aa8bebcfad7392b8832af72`。

```cpp
#include "step4.h"

#include "convolution/convolution_typed.h"
#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
#include "demod/demod_typed.h"
#include "spectral_analysis/spectral_analysis_typed.h"
#include "wavelets/wavelets_typed.h"

#include <numeric>

namespace task2 {
namespace {

// <学习注释：GPU kernel —— 解析信号构造（analytic_signal_kernel）。
//
//  【函数总体处理方法】
//  每个 CUDA 线程独立处理一个采样点 n，用中心差分近似构造复解析信号：
//    步骤1：计算全局线程索引 index = blockIdx.x * blockDim.x + threadIdx.x
//    步骤2：越界检查（index >= count 则返回）
//    步骤3：取前一点 previous（index == 0 时用 input[0] 复制边界）
//    步骤4：取后一点 next（index == count-1 时用 input[count-1] 复制边界）
//    步骤5：输出复数值 {input[index], 0.5F * (next - previous)}
//
//  【核心数学公式】
//    z[n] = x[n] + j * (x[n+1] - x[n-1]) / 2
//
//  这不是 Hilbert 变换，而是用局部差分近似导数放入虚部。端点用复制策略。
//  这是任务私有 glue kernel，不属于官方 fm_demod 算子。>
__global__ void analytic_signal_kernel(
    const float* input, cusignal::DemodComplex<float>* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float previous = index == 0 ? input[0] : input[index - 1];
    const float next = index + 1 == count ? input[count - 1] : input[index + 1];
    output[index] = {input[index], 0.5F * (next - previous)};
}

// <学习注释：CPU reference —— 谱图跨频率最大包络（spectral_envelope）。
//
//  【函数总体处理方法】
//  对展平的二维谱图（frequencies × frames）沿频率轴取每个时间帧的最大值，
//  得到一维时间包络：双重循环遍历所有频率和帧，对每帧取 max。
//
//  【核心数学公式】
//    F_spec[m] = max_k S[k, m]   其中 k 为频率索引，m 为帧索引
//
//  【物理意义】
//  将二维时频表示压缩为一维时间包络，保留最强的频率分量，便于后续融合。>
std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
    return output;
}

// <学习注释：CPU reference —— CWT 跨尺度最大包络（wavelet_envelope）。
//
//  【函数总体处理方法】
//  对展平的二维 CWT 结果（widths × samples）沿尺度轴取每个采样点的最大值，
//  得到一维包络，同时将 FP64 窄化为 FP32。
//
//  【核心数学公式】
//    F_wave[n] = max_a |W_x(a, n)|   其中 a 为尺度索引，n 为采样点索引
//
//  【物理意义】
//  将二维时频表示压缩为一维包络，保留各位置最强尺度响应。FP64→FP32 窄化通过
//  narrow_fp64_value_to_fp32_on_host 显式完成。>
std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample],
                cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(
                    flattened[static_cast<std::size_t>(width) * samples + sample])));
    return output;
}

// <学习注释：CPU reference —— 多域特征加权融合（make_bundle）。
//
//  【函数总体处理方法】
//  将四类一维特征统一长度（resample_linear）、归一化幅值（normalize_abs）后加权求和。
//
//  【核心数学公式】
//    F[n] = γ_fm * F̃_fm[n] + γ_corr * F̃_corr[n] + γ_spec * F̃_spec[n] + γ_wave * F̃_wave[n]
//    其中 F̃_k[n] = normalize_abs(resample_linear(feature_k, count))
//
//  【物理意义】
//  凸组合（Σγ=1, γ≥0）统一了四类特征的幅值尺度，但也消除了各特征绝对能量差异。>
std::vector<float> make_bundle(
    const std::vector<float>& fm,
    const std::vector<float>& correlation,
    const std::vector<float>& spectral,
    const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
    return bundle;
}

}  // namespace

// <学习注释：Step4 入口函数 —— 多域特征提取与融合流水线（run_step4）。
//
//  【函数总体处理方法】
//  run_step4 是 Step4 的入口函数，编排整个多域特征提取流水线：
//    阶段1（CPU Prep）：校验上游输入、计算 frame/hop/frames、准备 CWT workspace
//    阶段2（H2D）：smoothed 和 reference_for_correlation 拷贝到 GPU
//    阶段3（GPU compute）：依次执行 analytic_signal_kernel、fm_demod_device、
//                        correlate_device、spectrogram_device、cwt_device
//    阶段4（D2H）：四类特征回传 Host
//    阶段5（CPU Post）：spectral_envelope、wavelet_envelope 提取包络，make_bundle 融合
//
//  【核心数学公式】
//    F[n] = γ_fm * F̃_fm[n] + γ_corr * F̃_corr[n] + γ_spec * F̃_spec[n] + γ_wave * F̃_wave[n]
//
//  【物理意义】
//  从调制域、时域、频域和时频域四个维度提取特征后加权融合。这是 T2-R4
//  （多域特征提取）条款的核心实现。>
StepEvidence run_step4(PipelineState& state)
{
    // <学习注释：分配证据对象并命名当前 step，用于后续 JSON 输出和日志追踪。
    //  StepEvidence 是任务共享结构体，包含 name、prep_ms、h2d_ms、compute_ms、d2h_ms、post_ms、
    //  total_ms、operator_ms（map）、metrics（map）等字段。>
    StepEvidence evidence;
    evidence.name = "step4";
    // <学习注释：记录总耗时起点 total_begin，用于最终 evidence.total_ms。
    //  Clock 是项目定义的稳定时钟别名（通常 std::chrono::steady_clock）。
    //  begin 是阶段计时变量，每个阶段开始时重新赋值，结束时用 milliseconds(begin, Clock::now()) 求差。>
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    // <学习注释：阶段1（CPU Prep）—— 校验上游输入并准备 STFT/CWT 参数。
    //  count = state.smoothed.size()：Step3 平滑后的信号长度 N。
    //  static_cast<int> 显式转换 size_t 为 int，避免后续算术运算中的有符号/无符号混合。>
    const int count = static_cast<int>(state.smoothed.size());
    // <学习注释：require 是项目断言宏：count > 64 保证 STFT 帧（frame=64）至少能形成一帧。
    //  若上游 Step3 输出过短，抛出错误终止流水线，避免后续 frames 计算为负数或零。>
    require(count > 64, "step4 did not receive a usable step3 output");
    // <学习注释：STFT 帧长 frame = 64：每个时间窗包含 64 个采样点。
    //  constexpr 修饰：编译期常量，不占用运行时内存，编译器可做常量折叠优化。
    //  64 是 2 的幂，便于后续 FFT（基-2 算法）。>
    constexpr int frame = 64;
    // <学习注释：STFT 步长 hop = 32：相邻帧起点间隔 32 个样本。
    //  hop < frame 导致帧重叠 50%（64-32=32），在帧间提供平滑过渡，减少频谱泄漏。
    //  典型值：hop = frame/2 是 STFT 中常用的 50% 重叠。>
    constexpr int hop = 32;
    // <学习注释：frames = 1 + (count - frame) / hop：可用时间帧数。
    //  公式推导：(count - frame) 是首帧之后剩余样本数，整除 hop 得到可平移次数，+1 计入首帧。
    //  整数除法向下取整，末尾不足一帧的样本被丢弃。
    //  数学公式：M = 1 + ⌊(N - L) / H⌋，其中 N=count, L=frame, H=hop。>
    const int frames = 1 + (count - frame) / hop;
    // <学习注释：CWT 尺度列表 widths = {2.0, 4.0, 8.0, 12.0}：四个 Ricker 小波尺度。
    //  尺度越大对应中心频率越低，形成多分辨率分析（大尺度看低频轮廓，小尺度看高频细节）。
    //  std::vector<float> 列表初始化（C++11 brace-init），F 后缀指定 float 类型。>
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    // <学习注释：ricker 是 std::function 类型的小波生成回调（CwtRealWaveletCallable）。
    //  lambda 捕获 []（无捕获）：纯函数，调用 cusignal::ricker_typed_cpu<float>(points, width)。
    //  ricker_typed_cpu 是 Host 端模板函数，生成实数 Ricker（墨西哥帽）小波波形，返回 std::vector<double>（FP64），内部已包含 1/√a 归一化。
    //  Ricker 小波 = 高斯函数二阶导数的负数：ψ(t) = (1-π²f_c²t²)·exp(-π²f_c²t²)。>
    const cusignal::CwtRealWaveletCallable ricker =
        [](int points, int width) {
            return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
        };
    // <学习注释：prepare_cwt_workspace 预计算 CWT 所需的小波滤波器组。
    //  传入 count（信号长度 N）、widths（尺度列表）、ricker（小波生成回调）。
    //  workspace 内部存储每个尺度对应的预卷积小波，避免 GPU kernel 内重复生成。>
    auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);
    // <学习注释：记录 CPU Prep 阶段耗时。>
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段2（H2D）—— 将 Step3 平滑信号和参考信号从 Host 拷贝到 GPU 显存。
    //  DeviceArray<float>::from_host 在 GPU 上分配 N*sizeof(float) 字节显存并完成 H2D 传输。
    //  d_signal = state.smoothed：待分析信号 x[n]，来自 Step3 平滑输出。
    //  d_reference = state.reference_for_correlation：互相关参考信号 y[n]。>
    begin = Clock::now();
    auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);
    auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    // <学习注释：分配 GPU 输出缓冲区 d_analytic，长度 count，元素类型 DemodComplex<float>。
    //  DemodComplex<float> 是 cusignal 定义的复数结构体 {re, im}（成员名 re 与 im），作为 fm_demod_device 的输入。>
    cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);
    // <学习注释：阶段3.1（GPU compute）—— analytic_signal_kernel（私有 glue kernel）。
    //  time_gpu 包装 CUDA 事件计时，记录到 evidence.operator_ms["analytic_glue"]。
    //  [&] lambda 闭包按引用捕获 d_signal/d_analytic/count，在 GPU 上执行。
    //  launch_1d_kernel 自动计算 grid/block 配置，覆盖 count 个线程。
    //  数学公式：z[n] = x[n] + j·(x[n+1] - x[n-1])/2，中心差分近似导数放入虚部。>
    evidence.operator_ms["analytic_glue"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            analytic_signal_kernel, static_cast<std::size_t>(count),
            d_signal.data(), d_analytic.data(), count);
    });
    // <学习注释：构造 FM 解调 workspace，配置输入 shape 与 axis。
    //  FmDemodOptions{{count}, -1}：花括号初始化，{count} 设置输入 shape（一维 N=count，输出轴长度由 workspace 推导为 max(count-1, 0)），-1 是 axis 参数（最后一维，1D 时即 axis 0）。
    //  workspace 仅持有 Host-only 布局，不分配 device scratch。>
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    // <学习注释：分配 FM 解调输出缓冲区 d_fm，长度由 workspace.output_size() 决定，等于 max(count-1, 0)（输出轴长度 = 输入轴长度 - 1）。>
    cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());
    // <学习注释：阶段3.2（GPU compute）—— fm_demod_device（正式算子）。
    //  从 d_analytic 提取相邻样本的相位差，写入 d_fm。
    //  数学公式：output[n] = unwrap_local_delta(angle(z[n-1]), angle(z[n]))；即 angle → unwrap delta → diff 链路，输出为相邻样本相位差（单位弧度，未除以 2π，未使用 z·z* 共轭乘积以避免大分量溢出）。
    //  这是 T2-R4 四类大项中"解调"的 GPU 实现。>
    evidence.operator_ms["fm_demod"] = time_gpu([&] {
        cusignal::fm_demod_device(d_analytic, d_fm, demod_workspace);
    });
    // <学习注释：分配互相关输出缓冲区 d_correlation，长度 count。
    //  "same" 模式：cusignal_cpp 中输出长度 = len(in1) = len(x) = count（与 cuSignal 23.08 rank-one real 子集对齐，不取 max(L1, L2)；本任务中 in1=d_signal 长度 count，故输出 = count）。>
    cusignal::DeviceArray<float> d_correlation(count);
    // <学习注释：阶段3.3（GPU compute）—— correlate_device（正式算子）。
    //  d_signal = x[n]（平滑信号），d_reference = y[n]（参考信号）。
    //  数学公式：r_xy[n] = Σ_m x[m] · y[m - n + offset]。
    //  这是 T2-R4 四类大项中"相关"的 GPU 实现。>
    evidence.operator_ms["correlate"] = time_gpu([&] {
        cusignal::correlate_device(d_signal, d_reference, d_correlation, "same");
    });
    // <学习注释：分配谱图输出缓冲区 d_spectral，长度 frame * frames（展平二维谱图）。
    //  static_cast<std::size_t> 防止有符号乘法溢出，确保分配足够显存。>
    cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);
    // <学习注释：阶段3.4（GPU compute）—— spectrogram_device（正式算子）。
    //  STFT：帧长 frame=64，步长 hop=32，输出 PSD 谱图（periodic Hann 窗、双边频谱、spectrum scaling）。
    //  数学公式：S[k, m] = |DFT_k(periodic_Hann_L · x_frame_m)|² / (Σ Hann_L)²，其中 k 为双边频率 bin（nf = L = frame），m 为时间帧索引。
    //  这是 T2-R4 四类大项中"谱分析"的 GPU 实现。>
    evidence.operator_ms["spectrogram"] = time_gpu([&] {
        cusignal::spectrogram_device(d_signal, frame, hop, d_spectral);
    });
    // <学习注释：分配 CWT 输出缓冲区 d_cwt，长度由 workspace 决定，元素类型 double（FP64）。
    //  CWT 算子输出 FP64 是算子契约决定的，后续 wavelet_envelope 会窄化为 FP32。>
    cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());
    // <学习注释：阶段3.5（GPU compute）—— cwt_device（正式算子）。
    //  使用预计算的 workspace（含小波滤波器组）执行 CWT。
    //  数学公式：W_x(a, n) = Σ_m x[m] · ψ_a[L_a-1-(m'-n)]（same 模式卷积，输出长度 = N；ψ_a 由 ricker_typed_cpu 生成，长度 L_a = min(10a, N)，已内置 1/√a 归一化）。
    //  这是 T2-R4 四类大项中"小波变换"的 GPU 实现。>
    evidence.operator_ms["cwt_ricker"] = time_gpu([&] {
        cusignal::cwt_device(d_signal, cwt_workspace, d_cwt);
    });
    // <学习注释：汇总所有 GPU 算子耗时（analytic_glue + fm_demod + correlate + spectrogram + cwt_ricker）。
    //  std::accumulate 从 0.0 开始累加 operator_ms map 中的所有 second（耗时值）。
    //  lambda 接收 (double sum, const auto& item)，返回 sum + item.second。
    //  item.second 是 double 类型的耗时毫秒数。>
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });

    // <学习注释：阶段4（D2H）—— 将四类特征从 GPU 传回 Host。
    //  to_host() 在 Host 端分配等长 vector 并执行 D2H 传输。
    //  fm_feature / correlation_feature：一维特征，长度 count，写入 state 供后续融合使用。
    //  spectral_flat：展平二维谱图，长度 frame * frames，供 spectral_envelope 后处理。
    //  cwt_flat：展平二维 CWT 结果，FP64，供 wavelet_envelope 后处理。>
    begin = Clock::now();
    state.fm_feature = d_fm.to_host();
    state.correlation_feature = d_correlation.to_host();
    const auto spectral_flat = d_spectral.to_host();
    const auto cwt_flat = d_cwt.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    // <学习注释：阶段5（CPU Post）—— 提取包络并融合特征。
    //  spectral_envelope：沿频率轴取 max，将二维谱图压缩为一维时间包络。
    //  wavelet_envelope：沿尺度轴取 |·| 的 max，同时 FP64→FP32 窄化。
    //  make_bundle：重采样统一长度 + 归一化 + 加权融合四类特征。>
    begin = Clock::now();
    state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);
    state.wavelet_feature = wavelet_envelope(
        cwt_flat, static_cast<int>(widths.size()), count);
    // <学习注释：make_bundle 融合四类特征。
    //  传入 fm_feature / correlation_feature / spectral_feature / wavelet_feature、count、config。
    //  config 含四个融合权重（fusion_weight_fm/correlation/spectral/wavelet）。
    //  数学公式：F[n] = γ_fm·F̃_fm[n] + γ_corr·F̃_corr[n] + γ_spec·F̃_spec[n] + γ_wave·F̃_wave[n]。
    //  输出 state.feature_bundle 将作为 Step5 峰值定位的输入。>
    state.feature_bundle = make_bundle(
        state.fm_feature, state.correlation_feature,
        state.spectral_feature, state.wavelet_feature, count, state.config);

    // <学习注释：记录 CPU Post 阶段和总耗时。>
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    // <学习注释：可视化捕获：如果启用，保存二维谱图和 CWT grid 供后续绘图分析。
    //  spectrogram_grid / cwt_grid：完整二维数据（非包络）。
    //  spectrogram_frequency_bins = frame = 64：频率 bin 数。
    //  spectrogram_frames = frames：时间帧数。
    //  cwt_width_count = widths.size() = 4：尺度数。>
    if (visualization_capture_enabled()) {
        state.spectrogram_grid = spectral_flat;
        state.cwt_grid = cwt_flat;
        state.spectrogram_frequency_bins = frame;
        state.spectrogram_frames = frames;
        state.cwt_width_count = static_cast<int>(widths.size());
    }
    return evidence;
}

}  // namespace task2

```

### 3.3 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.h`

完整SHA `26865d99d04035b51aa8bebcfad7392b8832af72`。

```cpp
#pragma once

#include "accuracy/accuracy_types.h"

// <学习注释：CPU reference 验证数据收集入口 —— Step4 精度验证。
//
//  【函数总体处理方法】
//  collect_step4_validation 用纯 CPU 路径重新计算四类特征和融合结果，
//  作为 GPU 输出的精度基准。它调用 fm_demod_typed_cpu、correlate_typed_cpu、
//  spectrogram_typed_cpu、cwt_typed_cpu 四个正式算子的 CPU 版本，以及
//  与 step4.cu 相同的 spectral_envelope、wavelet_envelope、make_bundle。
//
//  【物理意义】
//  为 Step4 GPU 输出提供逐元素精度比较基准，验证 GPU 实现的正确性。>
namespace task2::accuracy {
Step4ValidationData collect_step4_validation(const PipelineState& state);
}

```

### 3.4 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`

完整SHA `26865d99d04035b51aa8bebcfad7392b8832af72`。

```cpp
#include "accuracy/validation/step4_validation.h"

#include "convolution/convolution_typed.h"
#include "demod/demod_typed.h"
#include "spectral_analysis/spectral_analysis_typed.h"
#include "wavelets/wavelets_typed.h"

namespace task2::accuracy {
namespace {

// <学习注释：CPU reference —— 解析信号构造（analytic_cpu）。
//  与 GPU 的 analytic_signal_kernel 逻辑完全一致，但以串行循环在 Host 端执行。
//  z[n] = x[n] + j * (x[n+1] - x[n-1]) / 2，端点用 front()/back() 复制边界。>
std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)
{
    std::vector<cusignal::DemodComplex<float>> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float previous = index == 0 ? input.front() : input[index - 1];
        const float next = index + 1 == input.size() ? input.back() : input[index + 1];
        output[index] = {input[index], 0.5F * (next - previous)};
    }
    return output;
}

// <学习注释：CPU reference —— 谱图跨频率最大包络（spectral_envelope）。
//  与 step4.cu 中同名函数逻辑完全一致，用于验证 GPU 谱图后处理。>
std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
    return output;
}

// <学习注释：CPU reference —— CWT 跨尺度最大包络（wavelet_envelope）。
//  与 step4.cu 中同名函数逻辑完全一致，但使用 static_cast<float> 窄化 FP64。>
std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample], static_cast<float>(std::abs(
                flattened[static_cast<std::size_t>(width) * samples + sample])));
    return output;
}

// <学习注释：CPU reference —— 多域特征加权融合（make_bundle）。
//  与 step4.cu 中同名函数逻辑完全一致，用于验证 GPU 融合结果。>
std::vector<float> make_bundle(
    const std::vector<float>& fm, const std::vector<float>& correlation,
    const std::vector<float>& spectral, const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
    return bundle;
}

}  // namespace

// <学习注释：CPU reference 验证数据收集入口 —— 完整 CPU 精度基准。
//
//  【函数总体处理方法】
//  用纯 CPU 路径依次计算：
//    1. analytic_cpu 构造解析信号
//    2. fm_demod_typed_cpu 做 FM 解调
//    3. correlate_typed_cpu 做互相关（direct 方法）
//    4. spectrogram_typed_cpu 做 STFT 谱图，再 spectral_envelope 提取包络
//    5. cwt_typed_cpu 做 CWT，再 wavelet_envelope 提取包络
//    6. make_bundle 融合四类特征
//  所有结果存入 Step4ValidationData 供 comparison 比较。>
Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    // <学习注释：分配 CPU 验证数据对象，包含 fm_cpu/correlation_cpu/spectral_cpu/wavelet_cpu/bundle_cpu 五个字段。>
    Step4ValidationData data;
    // <学习注释：CPU Prep —— 与 run_step4 完全一致的参数计算逻辑。
    //  count = state.smoothed.size()：Step3 平滑后的信号长度 N。
    //  static_cast<int> 显式转换 size_t 为 int，避免有符号/无符号混合运算。>
    const int count = static_cast<int>(state.smoothed.size());
    // <学习注释：STFT 帧长 frame = 64，与 run_step4 一致，保证 CPU/GPU 谱图维度相同。>
    constexpr int frame = 64;
    // <学习注释：STFT 步长 hop = 32，与 run_step4 一致。>
    constexpr int hop = 32;
    // <学习注释：frames = 1 + (count - frame) / hop：可用时间帧数，与 run_step4 一致。>
    const int frames = 1 + (count - frame) / hop;
    // <学习注释：CWT 尺度列表 widths = {2.0, 4.0, 8.0, 12.0}，与 run_step4 一致。>
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    // <学习注释：Ricker 小波生成回调，与 run_step4 一致。
    //  lambda 无捕获，调用 ricker_typed_cpu<float> 生成实数 Ricker 小波。>
    const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {
        return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
    };
    // <学习注释：构造 FM 解调 workspace，配置与 run_step4 一致（输入 shape={count}，axis=-1，输出长度 max(count-1, 0)）。>
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    // <学习注释：阶段1（CPU compute）—— FM 解调特征。
    //  analytic_cpu(state.smoothed)：构造解析信号 z[n] = x[n] + j·(x[n+1]-x[n-1])/2。
    //  fm_demod_typed_cpu：CPU 版本 FM 解调，执行 angle → unwrap delta → diff 链路。
    //  数学公式：output[n] = unwrap_local_delta(angle(z[n-1]), angle(z[n]))；输出为相邻样本相位差（单位弧度，未除以 2π，未使用 z·z* 共轭乘积以避免大分量溢出）。
    //  与 GPU 版本 fm_demod_device 的数学公式完全一致。>
    data.fm_cpu = cusignal::fm_demod_typed_cpu(
        analytic_cpu(state.smoothed), demod_workspace);
    // <学习注释：阶段2（CPU compute）—— 互相关特征。
    //  correlate_typed_cpu：CPU 版本互相关，使用 direct 方法（直接卷积，非 FFT）。
    //  "same" 模式：输出长度 = len(in1) = len(x) = count（cusignal_cpp same 模式返回第一输入长度 L1，与 cuSignal 23.08 rank-one real 子集对齐）。
    //  数学公式：r_xy[n] = Σ_m x[m] · y[m - n + offset]（实数信号下与"卷积翻转 y"等价）。
    //  与 GPU 版本 correlate_device 的数学公式一致，但实现方法不同（direct vs 可能的 FFT 方法）。>
    data.correlation_cpu = cusignal::correlate_typed_cpu(
        state.smoothed, state.reference_for_correlation, "same",
        cusignal::CorrelateMethod::direct);
    // <学习注释：阶段3（CPU compute）—— 谱图特征。
    //  spectrogram_typed_cpu：CPU 版本 PSD 谱图（periodic Hann 窗、双边频谱、spectrum scaling），输出 |DFT|² / (ΣHann)²。
    //  spectral_envelope：沿频率轴取 max，压缩为一维时间包络。
    //  与 GPU 版本 spectrogram_device + spectral_envelope 逻辑一致。>
    data.spectral_cpu = spectral_envelope(
        cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames);
    // <学习注释：阶段4（CPU compute）—— CWT 特征。
    //  cwt_typed_cpu：CPU 版本 CWT，使用 widths 和 ricker 回调，输出 same 模式卷积结果（FP64，shape [W, N]，W=widths.size()）。
    //  wavelet_envelope：沿尺度轴取 |·| 的 max，同时 FP64→FP32 窄化。
    //  与 GPU 版本 cwt_device + wavelet_envelope 逻辑一致。>
    data.wavelet_cpu = wavelet_envelope(
        cusignal::cwt_typed_cpu(state.smoothed, widths, ricker),
        static_cast<int>(widths.size()), count);
    // <学习注释：阶段5（CPU compute）—— 特征融合。
    //  make_bundle：重采样统一长度 + 归一化 + 加权融合四类 CPU 特征。
    //  数学公式：F_cpu[n] = γ_fm·F̃_fm[n] + γ_corr·F̃_corr[n] + γ_spec·F̃_spec[n] + γ_wave·F̃_wave[n]。
    //  与 GPU 版本 make_bundle 使用相同函数和相同 config 权重，保证融合逻辑可复现。>
    data.bundle_cpu = make_bundle(
        data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu,
        count, state.config);
    return data;
}

}  // namespace task2::accuracy

```

### 3.5 函数总体处理方法

本节从方案设计的角度，概述 Step4 中所有函数的总体处理方法和它们之间的协作关系。

#### 3.5.1 总体流水线架构

Step4 的完整多域特征提取流水线由以下函数协作完成：

```
                    ┌─────────────────────────────────────────────────┐
                    │         collect_step4_validation               │
                    │  (CPU 参考实现，用于精度验证)                    │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  analytic_cpu → fm_demod_typed_cpu       │   │
                    │  │  correlate_typed_cpu                     │   │
                    │  │  spectrogram_typed_cpu → spectral_env    │   │
                    │  │  cwt_typed_cpu → wavelet_envelope        │   │
                    │  │         ↓ (四类特征)                      │   │
                    │  │      make_bundle                          │   │
                    │  │  (FM+corr+spec+wave → bundle)             │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                         ↑ 精度对比验证
                    ┌─────────────────────────────────────────────────┐
                    │              run_step4 (GPU 实现)               │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  analytic_signal_kernel (glue)           │   │
                    │  │  → fm_demod_device                       │   │
                    │  │  correlate_device                        │   │
                    │  │  spectrogram_device → spectral_envelope  │   │
                    │  │  cwt_device → wavelet_envelope           │   │
                    │  │         ↓ (四类特征)                      │   │
                    │  │      make_bundle                          │   │
                    │  │  (FM+corr+spec+wave → bundle)             │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### 3.5.2 各函数处理方法一览

| 函数 | 类型 | 职责 | 核心数学公式 | 并行策略 |
|------|------|------|-------------|----------|
| `run_step4` | GPU 入口 | 编排整个流水线：CPU Prep → H2D → GPU compute → D2H → CPU Post → 证据收集 | $F[n] = \sum_k \gamma_k \tilde{F}_k[n]$ | 分阶段串行，各阶段内 GPU 并行 |
| `analytic_signal_kernel` | GPU kernel | 每个线程构造一个复解析信号样本，用中心差分近似导数放入虚部 | $z[n] = x[n] + j \cdot (x[n+1] - x[n-1]) / 2$ | 一维 grid/block，每线程一个采样点 |
| `spectral_envelope` | CPU 后处理 | 沿频率轴取最大值，将二维谱图压缩为一维时间包络 | $F_{spec}[m] = \max_k S[k, m]$ | 串行双重循环 |
| `wavelet_envelope` | CPU 后处理 | 沿尺度轴取绝对值最大值，同时 FP64→FP32 窄化 | $F_{wave}[n] = \max_a \|W_x(a, n)\|$ | 串行双重循环 |
| `make_bundle` | CPU 后处理 | 重采样统一长度+归一化+加权融合四类特征 | $F[n] = \sum_k \gamma_k \tilde{F}_k[n]$ | 串行 for 循环 |
| `analytic_cpu` | CPU 参考 | 与 GPU kernel 等价的串行实现，用于精度验证 | 与 `analytic_signal_kernel` 一致 | 串行 for 循环 |
| `collect_step4_validation` | CPU 参考入口 | 编排 CPU 流水线，生成基准数据 | 与 `run_step4` 等价 | 分阶段串行 |

#### 3.5.3 四类多域特征的数学与物理模型

##### (a) FM 解调特征（调制域）

- **数学公式**：$\text{output}[n] = \text{unwrap\_local\_delta}(\text{angle}(z[n-1]), \text{angle}(z[n]))$，即 `angle → unwrap delta → diff` 链路；输出为相邻样本的相位差，单位弧度，未除以 $2\pi$，未使用 $z \cdot z^*$ 共轭乘积以避免大分量溢出
- **本次参数**：`FmDemodOptions{{count}, -1}`（输入 shape `{count}`、axis = -1），$N$ = `state.smoothed.size()`，输出长度 = $\max(N-1, 0)$
- **物理意义**：从解析信号 $z[n]$ 中提取相邻样本的相位差，反映信号相位的局部变化率，用于刻画调制规律。这是 T2-R4 四类大项中的"解调"实现
- **调用方式**：`fm_demod_device(d_analytic, d_fm, demod_workspace)`

##### (b) 互相关特征（时域）

- **数学公式**：$r_{xy}[n] = \sum_m x[m] \cdot y[m - n + \text{offset}]$（实数信号下与"卷积并翻转 $y$"等价）
- **本次参数**：`"same"` 模式（cusignal_cpp 中输出长度 = $\text{len}(x)$，与 cuSignal 23.08 rank-one real 子集对齐），参考信号为 `state.reference_for_correlation`
- **物理意义**：测量平滑信号 $x$ 与参考信号 $y$ 的相似度随平移量 $n$ 的变化，峰值对应二者对齐位置。这是 T2-R4 四类大项中的"相关"实现
- **调用方式**：`correlate_device(d_signal, d_reference, d_correlation, "same")`

##### (c) 谱图特征（频域）

- **数学公式**：$S[k, m] = \frac{|\text{DFT}_k(\text{periodic\_Hann}_L \cdot x_{\text{frame},m})|^2}{(\sum \text{Hann}_L)^2}$，periodic Hann 窗、双边频谱（$k \in [0, L)$）、spectrum scaling，帧长 $L = 64$，hop = 32
- **本次参数**：帧长 `frame = 64`，步长 `hop = 32`，帧数 `frames = 1 + (count - 64) / 32`，频率 bin 数 `nf = frame = 64`
- **物理意义**：将一维信号分解为时频二维表示，反映不同时刻各频率分量的能量分布；通过 `spectral_envelope` 沿频率轴取最大值后压缩为一维时间包络。这是 T2-R4 四类大项中的"谱分析"实现
- **调用方式**：`spectrogram_device(d_signal, frame, hop, d_spectral)` → `spectral_envelope` 提取包络

##### (d) CWT 特征（时频域）

- **数学公式**：$W_x(a, n) = \sum_m x[m] \cdot \psi_a[L_a-1-(m'-n)]$（same 模式卷积，输出长度 = $N$；$\psi_a$ 由 `ricker_typed_cpu` 生成，长度 $L_a = \min(10a, N)$，已内置 $1/\sqrt{a}$ 归一化）
- **本次参数**：尺度 $a \in \{2, 4, 8, 12\}$，小波函数 `ricker_typed_cpu<float>`，输出 shape $[W, N]$（$W$ = `widths.size()` = 4）
- **物理意义**：用不同尺度小波与信号做卷积，大尺度捕捉低频成分，小尺度捕捉高频细节，提供多分辨率分析；通过 `wavelet_envelope` 沿尺度轴取绝对值最大值后压缩为一维包络。这是 T2-R4 四类大项中的"小波变换"实现
- **调用方式**：`cwt_device(d_signal, cwt_workspace, d_cwt)` → `wavelet_envelope` 提取包络

#### 3.5.4 多域特征融合模型的信号处理含义

Step4 的核心数学模型是四类特征的凸组合：

$$F[n] = \gamma_{fm} \tilde{F}_{fm}[n] + \gamma_{corr} \tilde{F}_{corr}[n] + \gamma_{spec} \tilde{F}_{spec}[n] + \gamma_{wave} \tilde{F}_{wave}[n]$$

其中 $\tilde{F}_k[n] = \text{normalize\_abs}(\text{resample\_linear}(F_k, N))$，约束 $\sum_k \gamma_k = 1$，$\gamma_k \geq 0$。

| 特征 | 来源 | 维度 | 物理含义 | 输出用途 |
|------|------|------|----------|---------|
| FM 解调 | `fm_demod_device` | $N$ | 瞬时频率刻画调制规律 | 识别调制类信号 |
| 互相关 | `correlate_device` | $N$ | 与参考信号的相似度 | 目标模板匹配 |
| 谱包络 | `spectrogram_device` → `spectral_envelope` | $N$ | 时频能量分布最强分量 | 检测瞬态/周期信号 |
| CWT 包络 | `cwt_device` → `wavelet_envelope` | $N$ | 多尺度小波系数最强响应 | 多分辨率瞬态检测 |
| 融合特征 | `make_bundle` | $N$ | 四类特征凸组合 | Step5 峰值定位 |

**凸组合的物理意义**：

- 权重 $\gamma_k \geq 0$ 且 $\sum_k \gamma_k = 1$ 保证融合特征幅值有界，各分量贡献按相对重要性叠加
- 归一化 `normalize_abs` 消除各特征量纲差异，使权重表达相对重要性而非绝对能量
- 重采样 `resample_linear` 统一长度为 $N$，保证逐元素加权求和合法
- 缺点：归一化也会消除各特征绝对能量差异，无法保留原始 SNR 信息

#### 3.5.5 CPU/GPU 精度验证策略

| 验证维度 | CPU 实现 | GPU 实现 | 验证方法 |
|---------|---------|---------|---------|
| 解析信号构造 | `analytic_cpu` | `analytic_signal_kernel` | 相同边界复制策略，逐元素比较 |
| FM 解调 | `fm_demod_typed_cpu` | `fm_demod_device` | 相同 `FmDemodWorkspace`，逐元素比较 |
| 互相关 | `correlate_typed_cpu(..., direct)` | `correlate_device(..., "same")` | 比较输出数组 |
| 谱图包络 | `spectrogram_typed_cpu` → `spectral_envelope` | `spectrogram_device` → `spectral_envelope` | 同一 CPU 后处理，验证 GPU 谱图 |
| CWT 包络 | `cwt_typed_cpu` → `wavelet_envelope` | `cwt_device` → `wavelet_envelope` | 同一 CPU 后处理，验证 GPU CWT |
| 特征融合 | `make_bundle` | `make_bundle` | 同一 CPU 函数，逐元素比较 |
| 端到端 | `collect_step4_validation` | `run_step4` | 比较 `bundle_cpu` 与 `state.feature_bundle` |

---

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按"语法结构—名称与类型—执行过程—任务语义—初学者易错点"讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once

#include "../task2_common.h"
```

**语法结构**

这个代码块由预处理指令构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 0 个闭合分隔符。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `pragma`、`once`、`include`、`task2_common`、`h`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `#pragma once`：`#pragma once` 要求编译器在同一翻译单元中只展开一次本头文件，避免重复声明或定义。
- 对源码锚点 `#include "../task2_common.h"`：`#include` 是预处理指令，在编译前把双引号中的项目头展开到当前翻译单元；`../task2_common.h` 引入 PipelineState、TaskConfig、StepEvidence 等任务共享类型。它本身不执行运行时算法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.2 文件级代码：run_step4 函数声明

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step4(PipelineState& state);
```

**语法结构**

这个代码块由函数或方法定义、函数调用构成。其中可见 2 个开放分隔符与 1 个闭合分隔符。续行不是独立语句。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step4`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `namespace task2 {`：`namespace` 建立命名空间 `task2`，把后续符号放进独立名称范围；末尾 `{` 同时打开该范围。
- 对源码锚点 `StepEvidence run_step4(PipelineState& state);`：函数声明语法：`run_step4` 是函数名，`StepEvidence` 是返回类型，`PipelineState& state` 是形参。声明符中的 `&` 表示引用。末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `Task2/task2_gpu_cpu/step4/step4.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断。
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号。

### 4.3 文件级代码：关闭命名空间

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

```cpp

}  // namespace task2
```

**语法结构**

这个代码块由声明/表达式续接构成。其中可见 0 个开放分隔符与 1 个闭合分隔符。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域解析。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 对源码锚点 `}  // namespace task2`：右花括号关闭此前 `namespace task2 {` 打开的命名空间作用域。`//` 后面的文字仅标记所关闭的命名空间，不参与编译。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `Task2/task2_gpu_cpu/step4/step4.h` 负责。关闭 `task2` 命名空间，结束头文件声明。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。

**初学者易错点**

- 当前锚点 `}  // namespace task2` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色。

### 4.4 建立当前符号所需的依赖名称（step4.cu）

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `#include "step4.h"`。

```cpp
#include "step4.h"

#include "convolution/convolution_typed.h"
#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
#include "demod/demod_typed.h"
#include "spectral_analysis/spectral_analysis_typed.h"
#include "wavelets/wavelets_typed.h"
```

**语法结构**

这个代码块由预处理指令构成。引入当前 step 头文件和四个正式算子头文件。续行不是独立语句。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step4`、`h`、`convolution`、`convolution_typed`、`cuda_utils`、`device_array`、`kernel_launch`、`demod`、`demod_typed`、`spectral_analysis`、`spectral_analysis_typed`、`wavelets`、`wavelets_typed`。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 对源码锚点 `#include "step4.h"`：引入当前 step 的头文件声明。
- 对源码锚点 `#include "convolution/convolution_typed.h"`：引入 `correlate_device` 等卷积相关算子声明。
- 对源码锚点 `#include "cuda_utils/device_array.h"`：引入 `DeviceArray` 模板，管理 GPU 显存缓冲区。
- 对源码锚点 `#include "cuda_utils/kernel_launch.h"`：引入 `launch_1d_kernel` 等 1D kernel 启动工具。
- 对源码锚点 `#include "demod/demod_typed.h"`：引入 `fm_demod_device`、`FmDemodWorkspace`、`DemodComplex` 等 FM 解调相关声明。
- 对源码锚点 `#include "spectral_analysis/spectral_analysis_typed.h"`：引入 `spectrogram_device` 等谱分析算子声明。
- 对源码锚点 `#include "wavelets/wavelets_typed.h"`：引入 `cwt_device`、`prepare_cwt_workspace`、`ricker_typed_cpu` 等小波变换相关声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块引入的四类算子头文件（convolution/demod/spectral_analysis/wavelets）直接对应 T2-R4 要求的四类大项。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：引入四类算子头文件，为 GPU compute 阶段提供接口声明。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。

**初学者易错点**

- 引号内文本是字符串数据而不是变量名或可执行表达式。

### 4.5 文件级代码：打开命名空间

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

```cpp
#include <numeric>

namespace task2 {
namespace {
```

**语法结构**

这个代码块由预处理指令和命名空间声明构成。其中可见 2 个开放分隔符与 0 个闭合分隔符。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`numeric`、`namespace`、`task2`。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 对源码锚点 `#include <numeric>`：引入标准库 `<numeric>`，提供 `std::accumulate`（用于累加 GPU 算子耗时）。
- 对源码锚点 `namespace task2 {`：打开命名空间 `task2`。
- 对源码锚点 `namespace {`：打开匿名命名空间，其中定义的所有符号（`analytic_signal_kernel`、`spectral_envelope`、`wavelet_envelope`、`make_bundle`）只在当前翻译单元可见，相当于内部链接。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

匿名命名空间确保私有 kernel 和 CPU reference 函数不被其他翻译单元意外链接。匿名命名空间是现代 C++ 推荐的文件作用域替代方案，优于 `static` 关键字。

**设计理由与替代方案**

- 匿名命名空间对类型和模板也有效，比 `static` 更通用。这保证了 `analytic_signal_kernel` 等私有函数不会与项目中其他同名符号冲突。

**初学者易错点**

- 匿名命名空间 `namespace {` 不是空命名空间；它创建了内部链接，与 `static` 函数类似但更通用。

### 4.6 analytic_signal_kernel：GPU kernel 定义

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `analytic_signal_kernel`；源码锚点 `__global__ void analytic_signal_kernel(`。

```cpp
__global__ void analytic_signal_kernel(
    const float* input, cusignal::DemodComplex<float>* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float previous = index == 0 ? input[0] : input[index - 1];
    const float next = index + 1 == count ? input[count - 1] : input[index + 1];
    output[index] = {input[index], 0.5F * (next - previous)};
}
```

**语法结构**

这个代码块由 CUDA kernel 定义构成。`__global__` 是 CUDA 扩展关键字，表明该函数由 Host 端调用、在 Device 端并行执行。函数体包含声明、条件分支和赋值。

**名称与类型**

- `input` 由 `const float*` 声明：指向 GPU 显存中只读 FP32 数组的指针。
- `output` 由 `cusignal::DemodComplex<float>*` 声明：指向 GPU 显存中 `DemodComplex<float>` 数组的指针，kernel 写入结果。
- `count` 是 `int` 类型，表示信号采样点数 $N$。
- `index` 是 `const int`，由 CUDA 内置变量计算：`blockIdx.x * blockDim.x + threadIdx.x` 将块号和块内线程号映射为一维全局线程索引。
- `previous` 是 `const float`，保存前一个采样点的值。
- `next` 是 `const float`，保存后一个采样点的值。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`input`|实信号输入数组 $x[n]$|$x[n]$|由 `d_signal.data()` 传入，Step3 输出|
|`output`|复解析信号输出数组 $z[n]$|$z[n] = x[n] + j \cdot (x[n+1] - x[n-1]) / 2$|写入后由 `fm_demod_device` 消费|
|`index`|采样点索引 $n$|$n = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$|由 CUDA 线程号产生|
|`previous`|前一采样点 $x[n-1]$|边界用 $x[0]$ 复制|用于计算虚部差分|
|`next`|后一采样点 $x[n+1]$|边界用 $x[N-1]$ 复制|用于计算虚部差分|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|`int`|零：加法单位元、空索引。在 `index == 0` 中用作边界检测。|
|`1`|`int`|一：单步增量。在 `index - 1` 和 `index + 1` 中取相邻采样点。|
|`0.5F`|`float`（`F` 后缀）|二分之一：中心差分的系数 $1/2$。来自导数近似公式 $(f(x+h)-f(x-h))/(2h)$ 中 $h=1$ 的特例。改变它会缩放虚部幅值。|

**执行过程**

- 对源码锚点 `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);`：声明全局线程索引。`*` 先于 `+` 求值。`static_cast<int>` 将 `size_t` 类型显式转换为 `int`。
- 对源码锚点 `if (index >= count) return;`：越界检查。grid 通常向上取整到 block 大小的整数倍，多余线程直接返回。
- 对源码锚点 `const float previous = index == 0 ? input[0] : input[index - 1];`：条件运算符选择前一个采样点。$n=0$ 时用 `input[0]` 复制边界。
- 对源码锚点 `const float next = index + 1 == count ? input[count - 1] : input[index + 1];`：条件运算符选择后一个采样点。$n=N-1$ 时用 `input[N-1]` 复制边界。
- 对源码锚点 `output[index] = {input[index], 0.5F * (next - previous)};`：聚合初始化 `DemodComplex<float>` 结构体。实部为 `input[index]`（原信号值），虚部为 `0.5F * (next - previous)`（中心差分近似导数）。`*` 优先级高于 `-`，`0.5F * (next - previous)` 先计算 `next - previous` 再乘以 0.5。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过中心差分近似导数构造复解析信号，作为 FM 解调的前置 glue 步骤。这是任务私有 kernel，不属于官方 `fm_demod` 算子。|
|业务输出|复解析信号 `d_analytic`，由 `fm_demod_device` 消费|
|下一消费者|`d_analytic` 交给 `fm_demod_device` 做 FM 解调|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块是解调（FM）特征提取的前置步骤，将实信号转换为复信号供 FM 解调算子使用。每个 CUDA 线程独立处理一个采样点。

**设计理由与替代方案**

- 使用中心差分而非 Hilbert 变换：中心差分计算量小（每个点只需两次减法一次乘法），适合 GPU 并行。Hilbert 变换需要 FFT/IFFT，计算量更大。但中心差分不是严格的解析信号，不保证抑制负频率。
- 端点复制策略：避免越界访问，同时保持输出长度不变。替代方案是只处理内部点并缩短输出，但会改变后续 FM 解调的输入长度。

**初学者易错点**

- `__global__` 是 CUDA 扩展关键字，不是标准 C++；它表明该函数由 Host 调用、Device 执行。
- `blockIdx.x * blockDim.x + threadIdx.x` 中的 `*` 先于 `+` 求值。
- `output[index] = {input[index], 0.5F * (next - previous)};` 中的花括号是聚合初始化，不是新的控制流作用域。
- `0.5F` 的 `F` 后缀指定 `float` 类型，不带后缀的 `0.5` 默认为 `double`。

### 4.7 spectral_envelope：CPU 谱图包络

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `spectral_envelope`；源码锚点 `std::vector<float> spectral_envelope(`。

```cpp
std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
    return output;
}
```

**语法结构**

这个代码块由函数定义构成。函数体包含声明、双重 for 循环和 return 语句。

**名称与类型**

- `flattened` 由 `const std::vector<float>&` 声明：展平的二维谱图，按行主序存储（frequency × frames），引用传递避免复制。
- `frequencies` 是 `int`，表示频率 bin 数（等于 frame 大小 $L=64$）。
- `frames` 是 `int`，表示时间帧数。
- `output` 是 `std::vector<float>`，长度为 `frames`，初始化为全 0.0F。
- `frequency` 是循环变量 `int`，遍历频率轴。
- `frame` 是循环变量 `int`，遍历时间轴。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`flattened`|展平谱图 $S[k, m]$|$S[k, m]$，按行主序展平为 `flattened[k * frames + m]`|由 `d_spectral.to_host()` 产生|
|`output`|谱图时间包络 $F_{spec}[m]$|$F_{spec}[m] = \max_k S[k, m]$|写入 `state.spectral_feature`，由 `make_bundle` 消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`float`（`F` 后缀）|零：初始化 output 为全零，使第一轮 `std::max` 比较时任何非负值都能胜出。|
|`0`|`int`|零：循环起始值。|

**执行过程**

- 对源码锚点 `std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);`：构造长度为 `frames`、初始值为 `0.0F` 的向量。`static_cast<std::size_t>` 将 `int` 显式转换为 `size_t`。
- 对源码锚点 `for (int frequency = 0; frequency < frequencies; ++frequency)`：外循环遍历频率轴。
- 对源码锚点 `for (int frame = 0; frame < frames; ++frame)`：内循环遍历时间轴。
- 对源码锚点 `output[frame] = std::max(...)`：对每个时间帧取所有频率中的最大值。`frequency * frames + frame` 将二维索引展平为一维。
- 对源码锚点 `return output;`：返回一维时间包络。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过 `std::max` 沿频率轴取最大值，将二维谱图压缩为一维时间包络。|
|业务输出|`state.spectral_feature`（一维谱图包络）|
|下一消费者|由 `make_bundle` 融合|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 CPU 后处理：在 GPU 完成 `spectrogram_device` 并 D2H 后，Host 端提取时间包络。它不涉及 GPU 计算。

**设计理由与替代方案**

- 只取最大值而非平均值：保留最强频率分量，适合检测瞬态信号。平均值会模糊峰值。
- 在 Host 端做包络提取：避免额外 GPU kernel 和 D2H 传输完整二维谱图（仅在可视化时保存完整 grid）。

**初学者易错点**

- `static_cast<std::size_t>(frequency) * frames + frame` 中 `static_cast` 只作用于 `frequency`，`* frames` 使用转换后的 `size_t` 类型。
- `std::max` 是函数模板，不是宏；它返回两个参数的较大值。

### 4.8 wavelet_envelope：CPU CWT 包络

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `wavelet_envelope`；源码锚点 `std::vector<float> wavelet_envelope(`。

```cpp
std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample],
                cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(
                    flattened[static_cast<std::size_t>(width) * samples + sample])));
    return output;
}
```

**语法结构**

这个代码块由函数定义构成。与 `spectral_envelope` 结构类似，但输入是 `double` 类型，输出需经过 `narrow_fp64_value_to_fp32_on_host` 窄化。

**名称与类型**

- `flattened` 由 `const std::vector<double>&` 声明：展平的二维 CWT 结果（FP64），按行主序存储（widths × samples）。
- `widths` 是 `int`，表示尺度数（4 个尺度：2, 4, 8, 12）。
- `samples` 是 `int`，表示采样点数 $N$。
- `output` 是 `std::vector<float>`，长度为 `samples`，初始化为全 0.0F。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`flattened`|展平 CWT 结果 $W_x(a, n)$|$W_x(a, n)$，按行主序展平|由 `d_cwt.to_host()` 产生|
|`output`|CWT 包络 $F_{wave}[n]$|$F_{wave}[n] = \max_a \|W_x(a, n)\|$|写入 `state.wavelet_feature`，由 `make_bundle` 消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`float`（`F` 后缀）|零：初始化 output 为全零。|
|`0`|`int`|零：循环起始值。|

**执行过程**

- 对源码锚点 `std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);`：构造长度为 `samples` 的 FP32 输出向量。
- 对源码锚点 `for (int width = 0; width < widths; ++width)`：外循环遍历尺度轴。
- 对源码锚点 `for (int sample = 0; sample < samples; ++sample)`：内循环遍历时间轴。
- 对源码锚点 `output[sample] = std::max(...)`：对每个采样点取所有尺度中的最大值。
- `cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(...))`：先取绝对值，再将 FP64 窄化为 FP32。`narrow_*` 函数是项目内定义的显式类型窄化工具。
- 对源码锚点 `return output;`：返回一维包络。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过 `std::max` 沿尺度轴取绝对值最大值，将二维 CWT 压缩为一维包络，同时将 FP64 显式窄化为 FP32。|
|业务输出|`state.wavelet_feature`（一维 CWT 包络）|
|下一消费者|由 `make_bundle` 融合|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 CPU 后处理：在 GPU 完成 `cwt_device` 并 D2H 后，Host 端提取包络并窄化类型。CWT 输出为 FP64 是算子契约决定的，任务通过显式窄化适配后续 FP32 融合路径。

**设计理由与替代方案**

- 显式 `narrow_fp64_value_to_fp32_on_host`：FP64→FP32 窄化会丢失精度，显式函数名让类型转换可审计。替代的隐式转换或 C 风格强制转换可能隐藏精度损失。
- 在 Host 端做包络提取：CWT 输出数据量大（widths × samples），在 GPU 端做包络可减少 D2H 传输量，但当前实现为简化架构选择在 Host 端做。

**初学者易错点**

- `std::abs` 对 `double` 返回 `double`，不是 `float`；因此需要 `narrow_fp64_value_to_fp32_on_host` 窄化。
- `static_cast<std::size_t>(width) * samples + sample` 中 `static_cast` 只作用于 `width`，防止有符号/无符号混合运算。

### 4.9 make_bundle：CPU 特征融合

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `make_bundle`；源码锚点 `std::vector<float> make_bundle(`。

```cpp
std::vector<float> make_bundle(
    const std::vector<float>& fm,
    const std::vector<float>& correlation,
    const std::vector<float>& spectral,
    const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
    return bundle;
}
```

**语法结构**

这个代码块由函数定义构成。函数体包含四次 `auto` 声明（`resample_linear` + `normalize_abs` 组合）、一个向量构造、一个 for 循环和 return 语句。

**名称与类型**

- `fm`、`correlation`、`spectral`、`wavelet` 由 `const std::vector<float>&` 声明：四类一维特征，引用传递。
- `count` 是 `std::size_t`，统一的目标长度。
- `config` 由 `const TaskConfig&` 声明：外部配置，包含四个融合权重字段。
- `fm_n`、`corr_n`、`spec_n`、`wave_n` 由 `const auto` 推导为 `std::vector<float>`：重采样并归一化后的特征。
- `bundle` 是 `std::vector<float>`，长度为 `count`，存储融合结果。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`fm`|FM 解调特征|原始长度由 `fm_demod_device` 决定|`state.fm_feature`|
|`correlation`|互相关特征|原始长度与 smoothed 相同|`state.correlation_feature`|
|`spectral`|谱图包络特征|原始长度为 frames|`state.spectral_feature`|
|`wavelet`|CWT 包络特征|原始长度为 count|`state.wavelet_feature`|
|`fm_n` 等|归一化特征 $\tilde{F}_k[n]$|$\tilde{F}_k[n] = \text{normalize\_abs}(\text{resample\_linear}(F_k, N))$|由 `resample_linear` 和 `normalize_abs` 产生|
|`bundle`|融合特征 $F[n]$|$F[n] = \sum_k \gamma_k \tilde{F}_k[n]$|写入 `state.feature_bundle`，由 Step5 消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|`std::size_t`|零：循环起始值。|

**执行过程**

- 对源码锚点 `const auto fm_n = normalize_abs(resample_linear(fm, count));`：先调用 `resample_linear` 将 FM 特征线性重采样到 `count` 长度，再调用 `normalize_abs` 取绝对值并按最大值归一化。`auto` 推导返回类型为 `std::vector<float>`。其余三个特征同理。
- 对源码锚点 `std::vector<float> bundle(count);`：构造长度为 `count` 的向量，默认初始化为 0.0F。
- 对源码锚点 `for (std::size_t index = 0; index < count; ++index)`：逐元素遍历。
- 对源码锚点 `bundle[index] = config.fusion_weight_fm * fm_n[index] + ...`：加权求和。`+` 和 `*` 按标准优先级计算：先做四个乘法，再做三次加法。
- 对源码锚点 `return bundle;`：返回融合特征。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过 `resample_linear` 统一长度、`normalize_abs` 统一幅值尺度、加权求和融合四类特征。四类特征来自 FM 解调、互相关、谱图和 CWT，涵盖 T2-R4 要求的四类大项。|
|业务输出|`state.feature_bundle`（融合特征）|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 CPU 后处理的最后一步：将四类一维特征统一长度和幅值尺度后加权融合。融合权重来自 `TaskConfig`，强制 $\sum\gamma=1$ 且非负（凸组合）。

**设计理由与替代方案**

- 先重采样再归一化最后融合：重采样使不同长度的特征可以加权求和；归一化使各特征对融合结果的贡献由权重 $\gamma$ 决定，不受原始幅值影响。
- 凸组合约束：$\sum\gamma=1$ 保证融合特征有界，$\gamma \ge 0$ 保证各项同向贡献。

**初学者易错点**

- `const auto` 推导的是 `std::vector<float>`，不是引用；`normalize_abs` 和 `resample_linear` 都返回新向量。
- `config.fusion_weight_fm * fm_n[index]` 中 `*` 先于 `+`，不需要额外括号。

### 4.10 关闭匿名命名空间

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `}  // namespace`。

```cpp
}  // namespace
```

**语法结构**

这个代码块由声明/表达式续接构成。其中可见 0 个开放分隔符与 1 个闭合分隔符。

**执行过程**

- 对源码锚点 `}  // namespace`：右花括号关闭此前 `namespace {` 打开的匿名命名空间。匿名命名空间中的私有函数在此处结束作用域。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

关闭匿名命名空间后，`run_step4` 定义在 `task2` 命名空间中（非匿名），可被外部链接。

**初学者易错点**

- 当前锚点 `}  // namespace` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色。

### 4.11 run_step4：入口函数开始（初始化 + Prep 阶段）

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `StepEvidence run_step4(PipelineState& state)`。

```cpp
StepEvidence run_step4(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step4";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    const int count = static_cast<int>(state.smoothed.size());
    require(count > 64, "step4 did not receive a usable step3 output");
    constexpr int frame = 64;
    constexpr int hop = 32;
    const int frames = 1 + (count - frame) / hop;
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker =
        [](int points, int width) {
            return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
        };
    auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

这个代码块由函数定义开始部分和 Prep 阶段构成。包含函数签名、对象声明、成员赋值、`Clock::now()` 调用、上游校验、编译期常量声明、lambda 定义和 `prepare_cwt_workspace` 调用。

**名称与类型**

- `state` 由 `PipelineState&` 声明：引用传递，持有 `smoothed`、`reference_for_correlation` 等上游输入和各特征输出字段。
- `evidence` 是 `StepEvidence` 类型，默认构造。
- `total_begin` 由 `const auto` 推导为 `Clock::time_point`：记录函数入口时间戳。
- `begin` 由 `auto` 推导为 `Clock::time_point`：记录各阶段起始时间戳（可被后续赋值覆盖）。
- `count` 是 `const int`，信号采样点数 $N$。
- `frame` 是 `constexpr int`，编译期常量 64：谱图帧长。
- `hop` 是 `constexpr int`，编译期常量 32：谱图帧移。
- `frames` 是 `const int`，计算得出：$1 + (N - 64) / 32$。
- `widths` 是 `const std::vector<float>`，CWT 尺度列表 $\{2, 4, 8, 12\}$。
- `ricker` 是 `const cusignal::CwtRealWaveletCallable`，lambda 封装的 Ricker 小波工厂函数。
- `cwt_workspace` 由 `auto` 推导：CWT 工作空间。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号|Step3 写入，Step4 读取并写入新字段|
|`evidence`|步骤证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标|
|`count`|信号采样点数 $N$|$N = \text{smoothed.size()}$|用于 kernel 越界保护、数组分配|
|`frame`|谱图帧长 $L$|$L = 64$|传入 `spectrogram_device`|
|`hop`|谱图帧移 $H$|$H = 32$|传入 `spectrogram_device`|
|`frames`|谱图帧数|$M = 1 + \lfloor(N-64)/32\rfloor$|用于分配 `d_spectral`|
|`widths`|CWT 尺度列表|$\{2, 4, 8, 12\}$|传入 `prepare_cwt_workspace`|
|`ricker`|Ricker 小波工厂|$\psi(t) = \frac{2}{\sqrt{3a}\pi^{1/4}}(1 - (t/a)^2)e^{-(t/a)^2/2}$|传入 `prepare_cwt_workspace`|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`64`|`int`（`constexpr`）|谱图帧长。2 的幂适合 FFT 计算。源码未记录为何选 64 而非 128/256。|
|`32`|`int`（`constexpr`）|谱图帧移。帧长的一半（50% 重叠）。源码未记录选值依据。|
|`1`|`int`|帧数公式中的偏移量：至少产生 1 帧。|
|`2.0F`|`float`|CWT 最小尺度。|
|`4.0F`|`float`|CWT 第二尺度。|
|`8.0F`|`float`|CWT 第三尺度。|
|`12.0F`|`float`|CWT 最大尺度。四个尺度呈倍数增长，源码未记录选值依据。|

**执行过程**

- 对源码锚点 `StepEvidence run_step4(PipelineState& state)`：函数定义开始。`{` 打开函数体。
- 对源码锚点 `StepEvidence evidence; evidence.name = "step4";`：默认构造证据记录对象并设置名称。
- 对源码锚点 `const auto total_begin = Clock::now(); auto begin = Clock::now();`：记录总耗时起点和 prep 阶段起点。`begin` 不是 `const`，后续会被重新赋值。
- 对源码锚点 `const int count = static_cast<int>(state.smoothed.size());`：从 `size_t` 显式转换为 `int`。
- 对源码锚点 `require(count > 64, ...)`：上游契约检查。`count` 必须大于 64（至少需要一帧谱图）。
- 对源码锚点 `constexpr int frame = 64; constexpr int hop = 32;`：编译期常量声明。
- 对源码锚点 `const int frames = 1 + (count - frame) / hop;`：整数除法计算帧数。`/` 是整数除法，向零截断（非负时等同于向下取整）。
- 对源码锚点 `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};`：列表初始化向量。
- 对源码锚点 `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) { return ...; };`：lambda 定义 callable 对象。`[]` 不捕获外部变量，定义 lambda 本身不会立即运行函数体。
- 对源码锚点 `auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);`：预计算 CWT 工作空间。
- 对源码锚点 `evidence.prep_ms = milliseconds(begin, Clock::now());`：记录 prep 阶段耗时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块校验上游数据有效性（`count > 64`），计算谱图参数 frame/hop/frames，准备 CWT 尺度和 Ricker 小波工厂。|
|业务输出|`cwt_workspace`（预计算的小波参数）|
|下一消费者|`cwt_workspace` 由 `cwt_device` 消费|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于 CPU 准备阶段：校验上游输入完整性，计算谱图和 CWT 所需的配置参数。`prepare_cwt_workspace` 在 Host 端预计算每个尺度的 Ricker 小波采样值。

**设计理由与替代方案**

- `constexpr` 用于 frame/hop：编译期常量不占用运行时存储，且编译器可对其做常量折叠优化。
- 整数除法计算帧数：`(count - frame) / hop` 向下取整，加上 1 得到能覆盖全部采样的帧数。这是标准 STFT 帧数公式。

**初学者易错点**

- 整数除法 `(count - frame) / hop` 会丢弃小数部分，不是浮点除法。
- `constexpr` 变量必须在编译期可求值；`frames` 依赖运行时 `count`，不能用 `constexpr`。
- lambda 定义不执行函数体；只有 `ricker(points, width)` 调用时才执行。

### 4.12 run_step4：H2D + GPU compute + D2H + Post + 可视化

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `begin = Clock::now();`（H2D 前）。

```cpp
    begin = Clock::now();
    auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);
    auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);
    evidence.operator_ms["analytic_glue"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            analytic_signal_kernel, static_cast<std::size_t>(count),
            d_signal.data(), d_analytic.data(), count);
    });
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());
    evidence.operator_ms["fm_demod"] = time_gpu([&] {
        cusignal::fm_demod_device(d_analytic, d_fm, demod_workspace);
    });
    cusignal::DeviceArray<float> d_correlation(count);
    evidence.operator_ms["correlate"] = time_gpu([&] {
        cusignal::correlate_device(d_signal, d_reference, d_correlation, "same");
    });
    cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);
    evidence.operator_ms["spectrogram"] = time_gpu([&] {
        cusignal::spectrogram_device(d_signal, frame, hop, d_spectral);
    });
    cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());
    evidence.operator_ms["cwt_ricker"] = time_gpu([&] {
        cusignal::cwt_device(d_signal, cwt_workspace, d_cwt);
    });
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });

    begin = Clock::now();
    state.fm_feature = d_fm.to_host();
    state.correlation_feature = d_correlation.to_host();
    const auto spectral_flat = d_spectral.to_host();
    const auto cwt_flat = d_cwt.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);
    state.wavelet_feature = wavelet_envelope(
        cwt_flat, static_cast<int>(widths.size()), count);
    state.feature_bundle = make_bundle(
        state.fm_feature, state.correlation_feature,
        state.spectral_feature, state.wavelet_feature, count, state.config);

    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled()) {
        state.spectrogram_grid = spectral_flat;
        state.cwt_grid = cwt_flat;
        state.spectrogram_frequency_bins = frame;
        state.spectrogram_frames = frames;
        state.cwt_width_count = static_cast<int>(widths.size());
    }
    return evidence;
```

**语法结构**

这个代码块是 `run_step4` 的核心执行部分，包含 H2D、五个 GPU 算子调用、D2H、三个 CPU 后处理函数调用、计时记录和可视化捕获。`cusignal::DeviceArray<cusignal::DemodComplex<float>>` 中的 `>>` 是嵌套模板闭合，不是右移运算符。

**名称与类型**

- `d_signal`、`d_reference` 由 `auto` 推导为 `DeviceArray<float>`：GPU 端平滑信号和参考信号。
- `d_analytic` 是 `DeviceArray<DemodComplex<float>>`：GPU 端复解析信号。
- `demod_workspace` 是 `FmDemodWorkspace`：FM 解调工作空间。
- `d_fm` 是 `DeviceArray<float>`：GPU 端 FM 解调输出。
- `d_correlation` 是 `DeviceArray<float>`：GPU 端互相关输出。
- `d_spectral` 是 `DeviceArray<float>`：GPU 端展平谱图（`frame × frames`）。
- `d_cwt` 是 `DeviceArray<double>`：GPU 端展平 CWT 结果（FP64）。
- `spectral_flat` 由 `const auto` 推导为 `std::vector<float>`：展平谱图 Host 端副本。
- `cwt_flat` 由 `const auto` 推导为 `std::vector<double>`：展平 CWT Host 端副本（FP64）。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_signal`|GPU 端平滑信号 $x[n]$|$x[n]$|由 `state.smoothed` H2D；所有 GPU 算子消费|
|`d_reference`|GPU 端参考信号 $q[n]$|$q[n]$|由 `state.reference_for_correlation` H2D；`correlate_device` 消费|
|`d_analytic`|复解析信号 $z[n]$|$z[n] = x[n] + j \cdot (x[n+1] - x[n-1]) / 2$|`analytic_signal_kernel` 写入；`fm_demod_device` 消费|
|`d_fm`|FM 解调特征|$f_{inst}[n] \propto \arg(z[n]z^*[n-1])$|`fm_demod_device` 写入；D2H 后存入 `state.fm_feature`|
|`d_correlation`|互相关序列 $r[k]$|$r[k] = \sum_n x[n] q^*[n-k]$（same 模式）|`correlate_device` 写入；D2H 后存入 `state.correlation_feature`|
|`d_spectral`|展平谱图 $S[k, m]$|$S[k, m]$ 展平|`spectrogram_device` 写入；D2H 后由 `spectral_envelope` 消费|
|`d_cwt`|展平 CWT $W_x(a, n)$|$W_x(a, n)$ 展平|`cwt_device` 写入；D2H 后由 `wavelet_envelope` 消费|
|`state.feature_bundle`|融合特征 $F[n]$|$F[n] = \sum_k \gamma_k \tilde{F}_k[n]$|`make_bundle` 写入；Step5 消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`-1`|`int`|`FmDemodOptions{{count}, -1}` 中的第二个参数。源码未记录 `-1` 的具体含义。|
|`0.0`|`double`|`std::accumulate` 的初始累加值。`double` 类型确保浮点累加精度。|

**执行过程**

- **H2D**：`begin = Clock::now();` 重新赋值记录起点。`DeviceArray<float>::from_host()` 将 Host 向量拷贝到 GPU 显存。`evidence.h2d_ms` 记录传输耗时。
- **GPU compute（第 1-2 步：解析信号 + FM 解调）**：构造 `d_analytic` 和 `demod_workspace`。`time_gpu` 分别计量 `analytic_signal_kernel`（私有 glue）和 `fm_demod_device`（正式算子）的耗时。
- **GPU compute（第 3-5 步：相关 + 谱图 + CWT）**：`time_gpu` 分别计量 `correlate_device`（`"same"` 模式）、`spectrogram_device`（frame=64, hop=32）和 `cwt_device`（使用预计算的 workspace）的耗时。`d_cwt` 是 `DeviceArray<double>`（FP64），与其他 DeviceArray 的 `float` 不同。
- **compute_ms 累加**：`std::accumulate` 遍历 `evidence.operator_ms` 映射，lambda `[](double sum, const auto& item) { return sum + item.second; }` 累加所有算子耗时。
- **D2H**：`begin = Clock::now();` 重新赋值。`to_host()` 将 GPU 数据拷贝回 Host。`fm_feature` 和 `correlation_feature` 直接存入 `state`；`spectral_flat` 和 `cwt_flat` 是局部变量。`cwt_flat` 是 `std::vector<double>`（FP64）。
- **Post**：`begin = Clock::now();` 重新赋值。`spectral_envelope` 将展平谱图压缩为一维包络；`wavelet_envelope` 将展平 CWT 压缩为一维包络（含 FP64→FP32 窄化）；`make_bundle` 融合四类特征。
- **计时收尾**：`evidence.post_ms` 记录后处理耗时；`evidence.total_ms` 记录从 `total_begin` 到当前的总耗时。
- **可视化捕获**：`if (visualization_capture_enabled())` 条件分支，仅在使能时保存完整二维谱图和 CWT 数据及元数据（frame/frames/width_count）到 state。
- **返回**：`return evidence;` 返回证据记录对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块依次执行 `analytic_signal_kernel`（私有 glue）、`fm_demod_device`（解调）、`correlate_device`（相关）、`spectrogram_device`（谱分析）、`cwt_device`（小波变换），完整覆盖 T2-R4 要求的四类大项。本块直接出现 4 种算子符号：`fm_demod`、`correlate`、`spectrogram`、`cwt`。|
|业务输出|`state.fm_feature`、`state.correlation_feature`、`state.spectral_feature`、`state.wavelet_feature`、`state.feature_bundle`|
|下一消费者|`feature_bundle` 交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块是 Step4 的核心执行区：依次完成 H2D → 五个 GPU 操作 → D2H → 三个 CPU 后处理 → 可视化捕获。`analytic_glue` 与 `fm_demod` 是两个独立计时项，必须分开描述。四个正式算子（fm_demod、correlate、spectrogram、cwt）涵盖 T2-R4 要求的四类特征提取大项。

**设计理由与替代方案**

- `time_gpu` 封装确保 GPU 操作完成后再计时（内部应包含 `cudaDeviceSynchronize`）。
- `d_cwt` 使用 FP64：CWT 数值精度要求更高，算子契约输出 FP64。后续在 `wavelet_envelope` 中显式窄化为 FP32。
- 条件保存可视化数据：完整二维数据量大，仅在需要可视化时保存，避免不必要的内存占用。

**初学者易错点**

- `cusignal::DeviceArray<cusignal::DemodComplex<float>>` 中的 `>>` 在 C++11 及之后可以正确解析为嵌套模板闭合，不是右移运算符。
- `time_gpu` 中的 lambda `[&]` 按引用捕获，lambda 内部对 `d_analytic`、`d_fm` 等的访问都是对当前作用域对象的直接引用。
- `d_cwt` 是 `DeviceArray<double>`，不是 `DeviceArray<float>`。D2H 后 `cwt_flat` 是 `std::vector<double>`。
- `total_ms` 使用 `total_begin`（函数入口时刻），而 `post_ms` 使用 `begin`（Post 阶段起点）；两者是不同的计时区间。

### 4.13 关闭 task2 命名空间

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

```cpp

}  // namespace task2
```

**语法结构**

这个代码块由声明/表达式续接构成。其中可见 0 个开放分隔符与 1 个闭合分隔符。

**执行过程**

- 对源码锚点 `}  // namespace task2`：右花括号关闭此前 `namespace task2 {` 打开的命名空间。`run_step4` 在此结束作用域。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

关闭 `task2` 命名空间，结束 Step4 的完整实现。

**初学者易错点**

- 当前锚点 `}  // namespace task2` 没有引入额外的数值运算或所有权转移。

### 4.14 建立当前符号所需的依赖名称（validation header）

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once

#include "accuracy/accuracy_types.h"
```

**语法结构**

这个代码块由预处理指令构成。

**执行过程**

- 对源码锚点 `#pragma once`：要求编译器在同一翻译单元中只展开一次本头文件。
- 对源码锚点 `#include "accuracy/accuracy_types.h"`：引入 `Step4ValidationData` 等精度验证类型定义。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|`Step4ValidationData`（CPU reference 数据容器）|
|下一消费者|由 comparison 模块消费，比较 CPU 和 GPU 输出|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：声明 CPU reference 验证数据收集接口。

**初学者易错点**

- 引号内文本是字符串数据而不是变量名或可执行表达式。

### 4.15 collect_step4_validation 函数声明

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.h`；符号 `collect_step4_validation`；源码锚点 `namespace task2::accuracy {`。

```cpp
namespace task2::accuracy {
Step4ValidationData collect_step4_validation(const PipelineState& state);
}
```

**语法结构**

这个代码块由命名空间声明和函数声明构成。`task2::accuracy` 是嵌套命名空间的简洁写法（C++17）。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：只读引用，不修改上游状态。与 `run_step4` 的 `PipelineState& state`（可修改）不同。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step4_validation`|CPU reference 验证入口|无独立数学符号|组织 CPU 算子调用并收集验证数据|
|`state`|任务流水线共享状态|无单一数学符号|只读访问 `smoothed`、`reference_for_correlation`、`config`|

**执行过程**

- 对源码锚点 `namespace task2::accuracy {`：打开嵌套命名空间。
- 对源码锚点 `Step4ValidationData collect_step4_validation(const PipelineState& state);`：函数声明。`const PipelineState&` 表示只读引用。
- 对源码锚点 `}`：关闭命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|`Step4ValidationData`（CPU reference 数据）|
|下一消费者|由 comparison 模块消费|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：声明 CPU reference 验证数据收集接口。

**初学者易错点**

- `const PipelineState& state` 中的 `const` 表示不修改 `state`，与 `run_step4` 的 `PipelineState& state`（可修改）不同。

### 4.16 建立当前符号所需的依赖名称（validation implementation）

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `文件级代码`；源码锚点 `#include "accuracy/validation/step4_validation.h"`。

```cpp
#include "accuracy/validation/step4_validation.h"

#include "convolution/convolution_typed.h"
#include "demod/demod_typed.h"
#include "spectral_analysis/spectral_analysis_typed.h"
#include "wavelets/wavelets_typed.h"
```

**语法结构**

这个代码块由预处理指令构成。引入验证头文件和四个正式算子头文件。

**执行过程**

- 引入 `step4_validation.h`、四个正式算子头文件。这些头文件提供了 CPU 版本的算子函数（`fm_demod_typed_cpu`、`correlate_typed_cpu`、`spectrogram_typed_cpu`、`cwt_typed_cpu`）。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|代码如何落实|本块引入的四类算子头文件直接对应 T2-R4 要求的四类大项。|
|业务输出|`Step4ValidationData`（CPU reference 数据）|
|下一消费者|由 comparison 模块消费|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：引入 CPU reference 计算所需的算子头文件。

### 4.17 analytic_cpu：CPU reference 解析信号

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `analytic_cpu`；源码锚点 `std::vector<cusignal::DemodComplex<float>> analytic_cpu(`。

```cpp
namespace task2::accuracy {
namespace {

std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)
{
    std::vector<cusignal::DemodComplex<float>> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float previous = index == 0 ? input.front() : input[index - 1];
        const float next = index + 1 == input.size() ? input.back() : input[index + 1];
        output[index] = {input[index], 0.5F * (next - previous)};
    }
    return output;
}
```

**语法结构**

这个代码块由函数定义构成。与 GPU 的 `analytic_signal_kernel` 逻辑完全一致，但以串行 `for` 循环在 Host 端执行。使用 `input.front()` 和 `input.back()` 做边界处理。

**名称与类型**

- `input` 由 `const std::vector<float>&` 声明：实信号输入。
- `output` 是 `std::vector<cusignal::DemodComplex<float>>`，长度为 `input.size()`。
- `index` 是 `std::size_t` 循环变量。
- `previous`、`next` 是 `const float`。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`input`|实信号 $x[n]$|$x[n]$|由 `state.smoothed` 传入|
|`output`|复解析信号 $z[n]$|$z[n] = x[n] + j \cdot (x[n+1] - x[n-1]) / 2$|由 `fm_demod_typed_cpu` 消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|`std::size_t`|零：循环起始值和边界检测。|
|`1`|`std::size_t`|一：单步增量。|
|`0.5F`|`float`（`F` 后缀）|二分之一：中心差分系数。|

**执行过程**

- 对源码锚点 `std::vector<cusignal::DemodComplex<float>> output(input.size());`：构造与输入等长的输出向量。
- 对源码锚点 `for (std::size_t index = 0; index < input.size(); ++index)`：串行遍历每个采样点。
- 对源码锚点 `const float previous = index == 0 ? input.front() : input[index - 1];`：条件运算符选择前一个采样点。`input.front()` 等价于 `input[0]`。
- 对源码锚点 `const float next = index + 1 == input.size() ? input.back() : input[index + 1];`：条件运算符选择后一个采样点。`input.back()` 等价于 `input[input.size()-1]`。
- 对源码锚点 `output[index] = {input[index], 0.5F * (next - previous)};`：聚合初始化 `DemodComplex<float>`。
- 对源码锚点 `return output;`：返回复解析信号。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过串行循环实现与 GPU kernel 相同的中心差分逻辑，作为 FM 解调前 glue 步骤的 CPU reference。|
|业务输出|复解析信号，由 `fm_demod_typed_cpu` 消费|
|下一消费者|`fm_demod_typed_cpu` 做 FM 解调|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：提供 CPU reference 实现，与 GPU `analytic_signal_kernel` 逐元素对应，用于验证 GPU 输出的正确性。

**设计理由与替代方案**

- `input.front()` 和 `input.back()` 代替 `input[0]` 和 `input[input.size()-1]`：语义更清晰，且对空 vector 有未定义行为（但代码已保证输入非空）。

**初学者易错点**

- `>>` 在 `std::vector<cusignal::DemodComplex<float>>` 中是嵌套模板闭合，在 C++11 及之后正确解析。

### 4.18 collect_step4_validation：完整 CPU 验证入口

完整 SHA `26865d99d04035b51aa8bebcfad7392b8832af72`；路径 `Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation`；源码锚点 `Step4ValidationData collect_step4_validation(`。

```cpp
Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    Step4ValidationData data;
    const int count = static_cast<int>(state.smoothed.size());
    constexpr int frame = 64;
    constexpr int hop = 32;
    const int frames = 1 + (count - frame) / hop;
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {
        return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
    };
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    data.fm_cpu = cusignal::fm_demod_typed_cpu(
        analytic_cpu(state.smoothed), demod_workspace);
    data.correlation_cpu = cusignal::correlate_typed_cpu(
        state.smoothed, state.reference_for_correlation, "same",
        cusignal::CorrelateMethod::direct);
    data.spectral_cpu = spectral_envelope(
        cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames);
    data.wavelet_cpu = wavelet_envelope(
        cusignal::cwt_typed_cpu(state.smoothed, widths, ricker),
        static_cast<int>(widths.size()), count);
    data.bundle_cpu = make_bundle(
        data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu,
        count, state.config);
    return data;
}
```

**语法结构**

这个代码块由函数定义构成。结构与 `run_step4` 的 Prep 和 Post 阶段类似，但所有算子使用 `_typed_cpu` 后缀的 CPU 版本。

**名称与类型**

- `data` 是 `Step4ValidationData` 类型，包含 `fm_cpu`、`correlation_cpu`、`spectral_cpu`、`wavelet_cpu`、`bundle_cpu` 五个 `std::vector<float>` 字段。
- 其余变量与 `run_step4` 中的同名变量含义相同。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data.fm_cpu`|CPU 版 FM 解调特征|$f_{inst}[n]$|`fm_demod_typed_cpu` 写入；由 comparison 与 GPU 输出比较|
|`data.correlation_cpu`|CPU 版互相关特征|$r[k]$（direct 方法）|`correlate_typed_cpu` 写入；由 comparison 比较|
|`data.spectral_cpu`|CPU 版谱图包络|$F_{spec}[m] = \max_k S[k, m]$|`spectral_envelope` 写入；由 comparison 比较|
|`data.wavelet_cpu`|CPU 版 CWT 包络|$F_{wave}[n] = \max_a \|W_x(a, n)\|$|`wavelet_envelope` 写入；由 comparison 比较|
|`data.bundle_cpu`|CPU 版融合特征|$F[n] = \sum_k \gamma_k \tilde{F}_k[n]$|`make_bundle` 写入；由 comparison 比较|

**执行过程**

- 对源码锚点 `Step4ValidationData data;`：默认构造验证数据结构。
- Prep 阶段：与 `run_step4` 相同的参数计算（count、frame、hop、frames、widths、ricker、demod_workspace）。
- 对源码锚点 `data.fm_cpu = cusignal::fm_demod_typed_cpu(analytic_cpu(state.smoothed), demod_workspace);`：先调用 `analytic_cpu` 构造解析信号，再调用 `fm_demod_typed_cpu` 做 FM 解调。两个操作组合在一条语句中。
- 对源码锚点 `data.correlation_cpu = cusignal::correlate_typed_cpu(..., "same", cusignal::CorrelateMethod::direct);`：使用 `direct` 方法（非 FFT）做互相关，确保与 GPU 版本精度可比。
- 对源码锚点 `data.spectral_cpu = spectral_envelope(cusignal::spectrogram_typed_cpu(...), frame, frames);`：先调用 `spectrogram_typed_cpu` 做 STFT 谱图，再调用 `spectral_envelope` 提取包络。
- 对源码锚点 `data.wavelet_cpu = wavelet_envelope(cusignal::cwt_typed_cpu(...), ...);`：先调用 `cwt_typed_cpu` 做 CWT，再调用 `wavelet_envelope` 提取包络。
- 对源码锚点 `data.bundle_cpu = make_bundle(...);`：融合四类 CPU 特征。
- 对源码锚点 `return data;`：返回验证数据。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块用纯 CPU 路径完整复现 Step4 的五个计算步骤（analytic + FM、correlate、spectrogram + envelope、CWT + envelope、bundle），为 GPU 输出提供逐元素精度基准。本块直接出现 4 种算子符号：`fm_demod`、`correlate`、`spectrogram`、`cwt`。|
|业务输出|`Step4ValidationData`（五个 CPU reference 数组）|
|下一消费者|由 comparison 模块消费，计算 CPU vs GPU 误差|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：完整 CPU reference 实现，覆盖 Step4 的全部五个计算步骤。`CorrelateMethod::direct` 确保使用与 GPU 可比的直接卷积方法（非 FFT）。

**设计理由与替代方案**

- `correlate_typed_cpu` 使用 `CorrelateMethod::direct`：直接方法计算量与 GPU 版本一致，精度可比。FFT 方法可能因浮点运算顺序不同产生微小差异。
- 组合调用（`spectral_envelope(spectrogram_typed_cpu(...))`）：减少中间变量，直接形成最终 CPU reference 值。

**初学者易错点**

- `analytic_cpu(state.smoothed)` 作为临时对象传给 `fm_demod_typed_cpu`，函数返回后临时对象析构。这是安全的，因为 `fm_demod_typed_cpu` 按值接收或已完成拷贝。
- `correlate_typed_cpu` 的 `CorrelateMethod::direct` 枚举值需显式传入，不能省略。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。

Step4 的 CPU/GPU 分工：

|层次|函数/符号|执行位置|职责|
|---|---|---|---|
|CPU reference|`analytic_cpu`|Host|串行中心差分，构造复解析信号|
|CPU reference|`spectral_envelope`|Host|沿频率轴取最大值，压缩谱图|
|CPU reference|`wavelet_envelope`|Host|沿尺度轴取最大值，FP64→FP32 窄化|
|CPU reference|`make_bundle`|Host|重采样 + 归一化 + 加权融合|
|CPU reference|`collect_step4_validation`|Host|编排全部 CPU 算子调用，收集验证数据|
|GPU wrapper|`run_step4`|Host|编排 H2D → GPU compute → D2H → CPU post |
|GPU kernel|`analytic_signal_kernel`|Device|并行中心差分，构造复解析信号|
|GPU 正式算子|`fm_demod_device`|Device|并行 FM 解调|
|GPU 正式算子|`correlate_device`|Device|并行互相关（same 模式）|
|GPU 正式算子|`spectrogram_device`|Device|并行 STFT 谱图|
|GPU 正式算子|`cwt_device`|Device|并行 CWT/Ricker|

关键差异：
- `analytic_signal_kernel`（GPU）与 `analytic_cpu`（CPU reference）逻辑相同，但前者并行、后者串行。
- CWT 路径：GPU 输出 FP64（`DeviceArray<double>`），CPU reference 在 `wavelet_envelope` 中通过 `static_cast<float>` 窄化，GPU 路径在 `wavelet_envelope` 中通过 `narrow_fp64_value_to_fp32_on_host` 窄化。两者窄化方式不同，但数学语义等价。
- 验证路径的 `correlate_typed_cpu` 使用 `CorrelateMethod::direct`，与 GPU 的 `correlate_device` 计算方式一致，确保精度可比。