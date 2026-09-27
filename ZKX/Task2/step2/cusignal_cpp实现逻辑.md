# Task2 Step2 cusignal_cpp 实现逻辑

## 1. 职责与版本

当前step接口、CPU reference、GPU wrapper、私有kernel、线程职责和跨步衔接。

- 当前ZKX SHA：`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交时间：`2026-08-18T20:22:35+08:00`
- 每段源码另绑定实际SHA。
- tracked源码状态：clean
- 本轮只静态阅读。

## 2. 源码覆盖与唯一归属

|源码|SHA|
|---|---|
|`ZKX/Task2/task2_gpu_cpu/step2/step2.h`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step2/step2.cu`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#pragma once

#include "../task2_common.h"

// <学习注释：Step2 入口函数 —— FIR 低通滤波流水线。
//
//  【函数总体处理方法】
//  run_step2 负责对 Step1 输出的 echo 信号执行 FIR 低通滤波，去除高频噪声。
//  处理流程分四个阶段：
//    阶段1（CPU）：验证上游数据契约，读取滤波配置参数
//    阶段2（H2D）：将 echo 和截止频率拷贝到 GPU 显存
//    阶段3（GPU滤波）：调用三个算子 kernel 分别生成 Hamming 窗、FIR 低通滤波器系数、
//                     执行 FIR 卷积滤波
//    阶段4（D2H）：将滤波结果回传 Host，记录各阶段耗时证据
//
//  【核心数学公式 —— FIR 低通滤波】
//    对输出采样点 n（n = 0, 1, ..., N-1），滤波结果为：
//      filtered[n] = Σ_{m=0}^{L-1} h[m] × echo[n − m]
//
//    其中 FIR 系数 h[m] 由窗函数法设计：
//      h[m] = h_d[m] × w[m]  （理想低通冲激响应 × 窗函数）
//      h_d[m] = 2·f_c/f_s·sinc(2·f_c/f_s·(m − (L−1)/2))  （理想低通 sinc 函数；sinc 采用归一化定义 sinc(x)=sin(πx)/(πx)，与 cusignal_cpp 源码一致）
//      w[m] = 0.54 − 0.46·cos(2π·m/(L−1))  （Hamming 窗）
//      然后按 scale 选项归一化使直流增益为 1
//
//    符号说明：
//      L   = taps_count     ：FIR 滤波器阶数（抽头数）
//      f_c = filter_cutoff_hz ：低通截止频率（Hz）
//      f_s = sample_rate_hz   ：采样率（Hz）
//      echo[n]                ：Step1 输出的回波信号（含目标+干扰+噪声）
//      filtered[n]            ：滤波后输出，将作为 Step3 B 样条平滑的输入
//
//  【物理意义】
//  Step1 的回波信号中混入了高频噪声和干扰分量。FIR 低通滤波通过截止频率 f_c
//  保留低频目标回波成分，衰减高频噪声，为后续 B 样条平滑和特征提取提供
//  更干净的信号。这是 T2-R2（滤波）条款的核心实现。>
namespace task2 {

StepEvidence run_step2(PipelineState& state);

}  // namespace task2

```

### 3.2 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#include "step2.h"

#include "cuda_utils/device_array.h"
#include "filter_design/filter_design_typed.h"
#include "filtering/filtering_typed.h"
#include "windows/windows_typed.h"

#include <numeric>

namespace task2 {

// <学习注释：GPU 入口函数 —— FIR 低通滤波流水线编排。
//
//  【函数总体处理方法】
//  run_step2 按以下阶段编排 FIR 低通滤波流水线：
//    阶段1（CPU 准备）：验证 echo 大小契约，读取 cutoff 和 taps_count 配置
//    阶段2（H2D）：将 echo 信号和截止频率 cutoff 拷贝到 GPU 显存
//    阶段3（GPU 算子链）：
//      (a) hamming_resident_device：在 GPU 上生成 Hamming 窗系数 w[m]
//      (b) firwin_lowpass_explicit_window_resident_device：用窗函数法设计 FIR 系数 h[m]
//      (c) firfilter_fp32_resident_device：执行 FIR 卷积滤波 y[n] = Σ h[m]·x[n−m]
//    阶段4（D2H）：将滤波结果 filtered 传回 Host，记录证据
//
//  【核心数学公式 —— 窗函数法 FIR 低通滤波器设计】
//    步骤1 —— Hamming 窗：
//      w[m] = 0.54 − 0.46·cos(2π·m/(L−1))，m = 0, 1, ..., L−1
//
//    步骤2 —— 窗化理想低通冲激响应：
//      h_d[m] = 2·f_c/f_s·sinc(2·f_c/f_s·(m − (L−1)/2))
//      h[m] = h_d[m] × w[m]  （时域加窗，抑制旁瓣）
//
//    步骤3 —— FIR 卷积滤波（时域直接型）：
//      filtered[n] = Σ_{m=0}^{L-1} h[m] × echo[n − m]，n = 0, 1, ..., N−1
//
//  【并行策略】
//  三个算子均为 GPU resident 版本，在 device 端直接分配和计算，不经过
//  H2D/D2H 中间传输。每个算子的并行策略由其各自实现决定（详见算子学习文档）。
//  本函数负责编排调用顺序、计时和证据收集。>
StepEvidence run_step2(PipelineState& state)
{
    // <学习注释：获取只读配置引用，避免拷贝。
    //  config 包含滤波参数：filter_cutoff_hz（截止频率 f_c）、filter_taps（FIR 阶数 L）、
    //  sample_rate_hz（采样率 f_s）。>
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step2";
    // <学习注释：记录总耗时起点。>
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    // <学习注释：验证上游数据契约 —— echo 信号长度必须等于 config.samples。
    //  require 在条件不满足时报告错误，防止后续越界访问。>
    require(state.echo.size() == static_cast<std::size_t>(config.samples),
        "step2 did not receive step1 echo");
    // <学习注释：准备滤波参数 —— 截止频率列表和 FIR 抽头数。
    //  cutoff 是单元素向量 {f_c}，支持 firwin 的低通设计接口。
    //  taps_count = L：FIR 滤波器阶数，决定了滤波器的频率选择性。>
    const std::vector<float> cutoff{config.filter_cutoff_hz};
    const int taps_count = config.filter_taps;
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段2（H2D）—— 将 echo 信号和 cutoff 截止频率从 Host 拷贝到 GPU 显存。
    //  DeviceArray::from_host 在 GPU 上分配显存并完成 H2D 传输。>
    begin = Clock::now();
    auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);
    auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);
    evidence.h2d_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段3（GPU 算子链）—— 在 GPU 上依次执行 Hamming 窗、FIR 系数设计、FIR 滤波。
    //  三个算子均为 resident 版本，在 device 端直接分配 buffers 和计算，避免中间 H2D/D2H。
    //
    //  步骤(a) —— Hamming 窗生成：
    //    分配 d_window（L 个 float 元素），在 GPU 上生成 Hamming 窗系数。
    //    数学公式：w[m] = 0.54 − 0.46·cos(2π·m/(L−1))。
    //    参数 true 表示对称窗（symmetric），两端对称，常用于 FIR 滤波器设计。>
    cusignal::DeviceArray<float> d_window(taps_count);
    evidence.operator_ms["hamming"] = time_gpu([&] {
        cusignal::hamming_resident_device<float>(taps_count, d_window, true);
    });
    // <学习注释：步骤(b) —— FIR 低通滤波器系数设计（窗函数法）：
    //    分配 d_taps（L 个 float 元素），在 GPU 上计算 FIR 系数。
    //    参数：taps_count=L（滤波器阶数）、d_cutoff={f_c}（截止频率）、
    //          d_window（Hamming 窗系数）、sample_rate_hz=f_s（采样率）、
    //          true（scale 归一化，使直流增益为 1）。
    //    数学公式：
    //      h_d[m] = 2·f_c/f_s·sinc(2·f_c/f_s·(m − (L−1)/2))
//      h[m] = h_d[m] × w[m]
//      h[m] = h[m] / Σ h[m]（scale 归一化，使 Σ h[m] = 1）>
    cusignal::DeviceArray<float> d_taps(taps_count);
    evidence.operator_ms["firwin"] = time_gpu([&] {
        cusignal::firwin_lowpass_explicit_window_resident_device(
            taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true);
    });
    // <学习注释：步骤(c) —— FIR 卷积滤波：
    //    分配 filtered（N 个 float 元素），在 GPU 上执行 FIR 卷积。
    //    参数：d_taps（FIR 系数 h）、d_echo（输入信号 x）、filtered（输出信号 y）。
    //    数学公式：filtered[n] = Σ_{m=0}^{L-1} h[m] × echo[n − m]。
    //    物理意义：用窗函数法设计的 FIR 低通滤波器对回波信号进行卷积滤波，
    //    保留低频目标回波成分，衰减高频噪声和干扰。>
    cusignal::DeviceArray<float> filtered(config.samples);
    evidence.operator_ms["firfilter"] = time_gpu([&] {
        cusignal::firfilter_fp32_resident_device(d_taps, d_echo, filtered);
    });
    // <学习注释：汇总所有算子耗时（hamming + firwin + firfilter）。>
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });

    // <学习注释：阶段4（D2H）—— 将滤波结果从 GPU 传回 Host。
    //  state.filtered 将作为 Step3 B 样条平滑的输入。>
    begin = Clock::now();
    state.filtered = filtered.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    // <学习注释：记录 resident 算子内部统计 —— 均为 0，因为 resident 版本
    //  在 device 端直接分配和计算，不经过显式 H2D/D2H。>
    evidence.metrics["resident_internal_h2d_count"] = 0.0;
    evidence.metrics["resident_internal_d2h_count"] = 0.0;
    evidence.metrics["resident_internal_allocation_count"] = 0.0;
    // <学习注释：记录 resident kernel 启动次数 —— 3 个算子各启动 1 个 kernel。>
    evidence.metrics["resident_kernel_count"] = 3.0;
    // <学习注释：估算 GPU 显存持久占用（device 端分配的总字节数）：
    //   - echo（N × 4 字节）
    //   - cutoff（1 × 4 字节）
    //   - window（L × 4 字节）
    //   - taps（L × 4 字节）
    //   - filtered（N × 4 字节）
    //   合计：N×4 + 4 + L×4×3 字节。>
    evidence.metrics["resident_persistent_device_bytes"] =
        static_cast<double>(config.samples * sizeof(float)
            + sizeof(float)
            + taps_count * sizeof(float) * 3);

    // <学习注释：记录总耗时（从 total_begin 到此刻）。>
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    // <学习注释：可视化捕获：如果启用，保存 FIR 系数供后续绘图分析。>
    if (visualization_capture_enabled())
        state.filter_taps = d_taps.to_host();
    return evidence;
}

}  // namespace task2
```

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#include "accuracy/validation/step2_validation.h"

#include "filter_design/filter_design_typed.h"
#include "filtering/filtering_typed.h"
#include "windows/windows_typed.h"

namespace task2::accuracy {

// <学习注释：CPU 参考实现 —— 收集 Step2 验证数据（collect_step2_validation）。
//  这是 run_step2 的 CPU 等价实现，用于生成 GPU 精度验证的基准数据。
//
//  【函数总体处理方法】
//  步骤1：调用 hamming_typed_cpu 生成 Hamming 窗系数（CPU 版本）
//  步骤2：配置 FirwinOptions（显式窗模式、低通、scale 归一化）
//  步骤3：调用 firwin_typed_cpu 生成 FIR 滤波器系数（fp64→fp32 窄化）
//  步骤4：配置 FirfilterOptions（shape、axis）
//  步骤5：调用 firfilter_typed_cpu 对 echo 和目标分量分别滤波
//          - filtered_cpu：echo 的滤波结果（与 GPU 对比）
//          - target_filtered_cpu：目标分量的滤波结果（分析目标信号保留情况）
//
//  【与 GPU 版本的差异】
//  - 使用 CPU 版本的算子函数（_typed_cpu 后缀），而非 GPU resident 版本
//  - firwin 输出 fp64，需 narrow_fp64_to_fp32_on_host 窄化为 fp32
//  - 串行执行，不使用 CUDA kernel
//  - 直接输出 std::vector<float>，不经过 DeviceArray
//  - 额外对 target_component 滤波，验证目标信号在滤波后的保留情况>
Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data;
    // <学习注释：从配置中读取滤波参数：taps_count = L（FIR 阶数）、
    //  cutoff = {f_c}（截止频率列表）。>
    const int taps_count = state.config.filter_taps;
    const std::vector<float> cutoff{state.config.filter_cutoff_hz};
    // <学习注释：调用 CPU 版本的 Hamming 窗生成。
    //  参数：taps_count=L（窗长度）、true（对称窗）。
    //  数学公式：w[m] = 0.54 − 0.46·cos(2π·m/(L−1))。>
    const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);
    // <学习注释：配置 FIR 滤波器设计选项 —— 显式窗模式（使用预计算的 Hamming 窗），
    //  而非让 firwin 内部生成窗函数。>
    cusignal::FirwinOptions design_options;
    design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;
    design_options.window = window_cpu;
    // <学习注释：pass_zero = boolean_true 表示低通（DC 附近频带通过）。
    //  scale = true 表示归一化使直流增益为 1（Σ h[m] = 1）。
    //  fs = sample_rate_hz 设置采样率。>
    design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;
    design_options.scale = true;
    design_options.fs = state.config.sample_rate_hz;
    // <学习注释：调用 CPU 版本的 firwin 滤波器设计。
    //  firwin_typed_cpu 输出 fp64 精度，通过 narrow_fp64_to_fp32_on_host 窄化为 fp32。
    //  数学公式：h[m] = h_d[m] × w[m]，其中 h_d[m] = 2·f_c/f_s·sinc(2·f_c/f_s·(m−(L−1)/2))。>
    const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));
    // <学习注释：配置 FIR 滤波选项 —— shape（输出形状）、axis（滤波轴）。>
    cusignal::FirfilterOptions filter_options;
    filter_options.shape = {state.config.samples};
    filter_options.axis = -1;
    // <学习注释：对 echo 信号执行 FIR 滤波（CPU 版本）。
    //  输出 data.filtered_cpu，用于与 GPU 版本 filtered 做精度对比。
    //  数学公式：filtered[n] = Σ_{m=0}^{L-1} h[m] × echo[n − m]。>
    data.filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.echo, filter_options).y;
    // <学习注释：对 target_component 分量执行 FIR 滤波（CPU 版本）。
    //  输出 data.target_filtered_cpu，用于分析目标信号在滤波后的保留情况。
    //  验证滤波是否在去除噪声的同时保留了目标回波成分。>
    data.target_filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.target_component, filter_options).y;
    return data;
}

}  // namespace task2::accuracy
```

### 3.4 函数总体处理方法

本节从方案设计的角度，概述 Step2 中所有函数的总体处理方法和它们之间的协作关系。

#### 3.4.1 总体流水线架构

Step2 的 FIR 低通滤波流水线由以下函数协作完成：

```
                    ┌─────────────────────────────────────────────────┐
                    │     collect_step2_validation (CPU 参考实现)      │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  hamming_typed_cpu → firwin_typed_cpu    │   │
                    │  │         ↓ (FIR 系数 h)                   │   │
                    │  │  firfilter_typed_cpu (echo → filtered)   │   │
                    │  │  firfilter_typed_cpu (target → target_f) │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                         ↑ 精度对比验证
                    ┌─────────────────────────────────────────────────┐
                    │              run_step2 (GPU 实现)                │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  hamming_resident_device                 │   │
                    │  │  firwin_lowpass_explicit_window_resident  │   │
                    │  │         ↓ (FIR 系数 h)                   │   │
                    │  │  firfilter_fp32_resident_device          │   │
                    │  │  (echo → filtered)                       │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### 3.4.2 各函数处理方法一览

| 函数 | 类型 | 职责 | 核心数学公式 | 并行策略 |
|------|------|------|-------------|----------|
| `run_step2` | GPU 入口 | 编排滤波流水线：契约验证 → H2D → 窗/系数/滤波 → D2H → 证据收集 | $filtered[n] = \sum h[m]·echo[n-m]$ | 分阶段串行，各阶段内 GPU 并行 |
| `collect_step2_validation` | CPU 参考入口 | 编排 CPU 流水线，生成基准滤波结果和 target 分量滤波结果 | 与 `run_step2` 等价 | 串行 |
| `hamming_resident_device` | GPU 算子 | 在 GPU 上生成 Hamming 窗系数 | $w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$ | 每抽头系数一个线程 |
| `firwin_lowpass_explicit_window_resident_device` | GPU 算子 | 在 GPU 上用窗函数法设计 FIR 低通滤波器系数 | $h[m] = h_d[m]·w[m]$，$h_d[m] = \frac{2 f_c}{f_s}\text{sinc}\left(\frac{2 f_c}{f_s}(m-\frac{L-1}{2})\right)$（归一化 sinc） | 每抽头系数一个线程 |
| `firfilter_fp32_resident_device` | GPU 算子 | 在 GPU 上执行 FIR 卷积滤波 | $y[n] = \sum_{m=0}^{L-1} h[m]·x[n-m]$ | 每输出采样点一个线程，内积归约 |

#### 3.4.3 算子数学原理在当前任务中的实例化

##### (a) Hamming 窗（`hamming`）

- 学习入口：[数学物理原理](../../operators/windows/hamming/数学物理原理.md)、[Python 源码算法](../../operators/windows/hamming/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/windows/hamming/cusignal_cpp_hamming复现逻辑.md)
- 当前调用采用的数学关系：$w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$，$m = 0, 1, ..., L-1$
- 任务变量到数学符号的映射：`taps_count` → $L$（窗长度），`d_window` → $w[m]$（窗系数数组）
- 参数实例化：`symmetric=true`（对称窗，两端对称），窗长度 = `config.filter_taps`
- 物理意义：Hamming 窗旁瓣衰减约 −43 dB，主瓣宽度适中，是 FIR 低通滤波器设计的常用窗函数

##### (b) FIR 滤波器系数设计（`firwin`）

- 学习入口：[数学物理原理](../../operators/filter_design/firwin/数学物理原理.md)、[Python 源码算法](../../operators/filter_design/firwin/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/filter_design/firwin/cusignal_cpp_firwin复现逻辑.md)
- 当前调用采用的数学关系：$h[m] = h_d[m] \times w[m]$，其中 $h_d[m] = 2\frac{f_c}{f_s}\text{sinc}\left(\frac{2 f_c}{f_s}(m - \frac{L-1}{2})\right)$（sinc 为归一化定义 $\text{sinc}(x)=\sin(\pi x)/(\pi x)$，与 `cusignal_cpp/src/filter_design/filter_design_kernels.cuh` 中 `sinc` 实现一致）
- 任务变量到数学符号的映射：`taps_count` → $L$（滤波器阶数），`filter_cutoff_hz` → $f_c$（截止频率），`sample_rate_hz` → $f_s$（采样率），`d_taps` → $h[m]$（FIR 系数）
- 参数实例化：`window_mode=explicit_values`（使用预计算的 Hamming 窗）、`pass_zero=boolean_true`（低通）、`scale=true`（归一化 Σh = 1）
- 物理意义：窗函数法通过时域加窗截断理想无限长冲激响应，实现可实现的 FIR 滤波器；Hamming 窗提供约 −43 dB 的旁瓣衰减（与 3.4.3(a) 一致；−53 dB 为 Blackman 窗的值，不适用于 Hamming 窗）

##### (c) FIR 卷积滤波（`firfilter`）

- 学习入口：[数学物理原理](../../operators/filtering/firfilter/数学物理原理.md)、[Python 源码算法](../../operators/filtering/firfilter/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/filtering/firfilter/cusignal_cpp_firfilter复现逻辑.md)
- 当前调用采用的数学关系：$y[n] = \sum_{m=0}^{L-1} h[m] \times x[n-m]$，$n = 0, 1, ..., N-1$
- 任务变量到数学符号的映射：`echo` → $x[n]$（输入信号），`d_taps` → $h[m]$（FIR 系数），`filtered` → $y[n]$（滤波输出）
- 参数实例化：`shape={N}`（一维输出）、`axis=-1`（沿最后一维滤波）
- 物理意义：时域卷积等价于频域乘积（$Y(f) = H(f) \times X(f)$），FIR 低通滤波器在频域上保留 $|f| < f_c$ 的低频成分，衰减高频噪声

#### 3.4.4 回波信号滤波的物理含义

Step2 的核心数学模型是 FIR 低通滤波对回波信号的频域选择：

$$filtered[n] = \sum_{m=0}^{L-1} h[m] \times echo[n-m]$$

| 分量 | 滤波前状态 | 滤波后预期 | 物理意义 |
|------|-----------|-----------|---------|
| 目标回波（chirp，20–70 Hz） | 占用 20–70 Hz 频带 | 低通滤波器保留 | 目标信号在通带内，应被保留 |
| 高斯脉冲干扰（170 Hz 中心） | 中心频率 170 Hz | 若 f_c < 170 Hz 则被衰减 | 窄带干扰在阻带内，应被滤除 |
| 锯齿波干扰（190 Hz 基频，含谐波） | 基频 190 Hz + 谐波 | 基频和谐波大部分在阻带 | 宽带干扰中高频分量被衰减 |
| 方波干扰（125 Hz 基频，含谐波） | 基频 125 Hz + 谐波 | 取决于 f_c 与 125 Hz 的关系 | 仅低频谐波可能残留 |
| 噪声分量 | 全频带白噪声 | 高频部分被衰减 | 信噪比提升 |

**截止频率 $f_c$ 的选择**：$f_c$ 应设置在目标 chirp 最高频率（70 Hz）和最低干扰频率（125 Hz）之间，以最大化信号保留和干扰抑制。

#### 3.4.5 CPU/GPU 精度验证策略

| 验证维度 | CPU 实现 | GPU 实现 | 验证方法 |
|---------|---------|---------|---------|
| 窗函数生成 | `hamming_typed_cpu` | `hamming_resident_device` | 逐元素比较 Hamming 窗系数 |
| FIR 系数设计 | `firwin_typed_cpu` + `narrow_fp64_to_fp32_on_host` | `firwin_lowpass_explicit_window_resident_device` | 逐元素比较 FIR 系数 |
| FIR 滤波 | `firfilter_typed_cpu` | `firfilter_fp32_resident_device` | 逐元素比较 filtered 输出 |
| 目标信号保留 | 对 `target_component` 单独滤波 | 无独立 GPU 对比 | 分析目标分量在滤波后的幅度衰减 |

---

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按"语法结构—名称与类型—执行过程—任务语义—初学者易错点"讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once

#include "../task2_common.h"
```

**语法结构**

这个代码块由预处理指令构成。`#pragma once` 和 `#include` 均在编译预处理阶段执行，不产生运行时代码。

**名称与类型**

- `pragma once`：编译器指令，确保同一翻译单元只展开本头文件一次，防止重复声明
- `include "../task2_common.h"`：预处理指令，将 `task2_common.h` 的内容展开到当前翻译单元，引入 `PipelineState`、`StepEvidence`、`Clock` 等共享定义

**变量—公式—任务状态映射**

当前块只包含预处理指令，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `#pragma once`：要求编译器在同一翻译单元中只展开一次本头文件，避免重复声明或定义
- `#include "../task2_common.h"`：在预处理阶段将 `task2_common.h` 的声明展开到当前文件，使本文件能使用 `PipelineState`、`StepEvidence` 等类型

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只建立头文件依赖和声明归属，不执行当前 step 的数值算法|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法|

**任务语义**

本块属于公共基础设施：建立头文件依赖，使后续声明能使用 `PipelineState`、`StepEvidence` 等共享类型。

**设计理由与替代方案**

`#pragma once` 是现代编译器广泛支持的 include guard 替代方案，比传统的 `#ifndef`/`#define`/`#endif` 更简洁且不会出现宏名冲突。当前块承担接口声明职责，没有独立数学变换。

**初学者易错点**

- `#include` 是预处理指令，不是运行时函数调用；它在编译前展开，不产生可执行代码

### 4.2 Step2 入口函数声明

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`；符号 `run_step2`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step2(PipelineState& state);
```

**语法结构**

命名空间声明后跟函数声明。

**名称与类型**

- `namespace task2`：命名空间，将内部声明放入 `task2` 名称范围，避免与其他模块的符号冲突
- `StepEvidence`：返回类型，包含 step 名称、各阶段耗时（prep/h2d/compute/d2h/total）和 metrics 指标
- `run_step2`：函数名，Step2 的入口函数
- `PipelineState& state`：参数类型为 `PipelineState` 的引用（`&`），非 const 引用表示函数可以修改 state 的内容

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step2`|Step2 流水线入口函数|无独立数学符号|接收 Step1 的 echo，输出 filtered 给 Step3|
|`state`|任务流水线共享状态|无单一数学符号|保存各 step 的数组与证据；Step1 写入 echo，Step2 读取并写入 filtered|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `namespace task2 {`：打开命名空间 `task2`，后续声明被放入该名称范围
- `StepEvidence run_step2(PipelineState& state);`：函数声明，末尾分号结束声明。真正执行发生在 `step2.cu` 中的函数定义处

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法入口|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块声明 `run_step2` 函数签名，定义其接口契约|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明接口存在，但声明本身不执行算法|

**任务语义**

本块声明 Step2 的公开接口：接收流水线状态引用，返回步骤证据对象。`PipelineState&` 为非 const 引用，表明函数会修改 state 中的 filtered 字段。

**设计理由与替代方案**

引用传递（`&`）而非值传递，避免了拷贝整个 `PipelineState` 对象的开销。非 const 引用允许函数直接修改 state 的 filtered 字段，实现数据在步骤间传递。

**初学者易错点**

- 声明符中的 `&` 表示引用（现有对象的别名），不是取地址或按位与
- 函数声明末尾的分号是必需的，忘记会导致编译错误

### 4.3 命名空间闭合

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

```cpp
}  // namespace task2
```

**语法结构**

右花括号关闭命名空间作用域。

**名称与类型**

- `}  // namespace task2`：关闭 `namespace task2` 的作用域，`//` 后的文字是注释标记所关闭的命名空间

**变量—公式—任务状态映射**

当前块只关闭词法作用域，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 右花括号关闭 `namespace task2` 的作用域，在该命名空间内声明的符号不再可见

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只关闭词法作用域，不产生新的业务数组或数值结果|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只管理作用域，不能冒充核心算法|

**任务语义**

本块只关闭命名空间作用域，没有独立的任务语义。

**设计理由与替代方案**

`// namespace task2` 注释是代码风格约定，帮助快速识别闭合的命名空间，提高可读性。

**初学者易错点**

- 花括号本身不产生运行时操作；它只是编译期作用域边界

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `文件级代码`；源码锚点 `#include "step2.h"`。

```cpp
#include "step2.h"

#include "cuda_utils/device_array.h"
#include "filter_design/filter_design_typed.h"
#include "filtering/filtering_typed.h"
#include "windows/windows_typed.h"

#include <numeric>
```

**语法结构**

这个代码块由 6 条 `#include` 预处理指令构成。

**名称与类型**

- `"step2.h"`：包含 Step2 的头文件，引入 `run_step2` 声明和 `PipelineState`、`StepEvidence` 等类型
- `"cuda_utils/device_array.h"`：包含 `DeviceArray` 模板类声明，提供 GPU 显存管理和 H2D/D2H 传输
- `"filter_design/filter_design_typed.h"`：包含 `firwin_lowpass_explicit_window_resident_device` 等 FIR 滤波器设计函数的声明
- `"filtering/filtering_typed.h"`：包含 `firfilter_fp32_resident_device` 等 FIR 滤波函数的声明
- `"windows/windows_typed.h"`：包含 `hamming_resident_device` 等窗函数生成函数的声明
- `<numeric>`：C++ 标准库，提供 `std::accumulate` 用于汇总算子耗时

**变量—公式—任务状态映射**

当前块只包含预处理指令，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 6 条 `#include` 指令在预处理阶段按顺序展开，将各自头文件的内容插入当前翻译单元
- `"..."` 表示项目内头文件（相对于源文件目录搜索），`<...>` 表示系统/标准库头文件

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只建立头文件依赖和声明归属，不执行当前 step 的数值算法|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法|

**任务语义**

本块属于公共基础设施：建立三个算子模块（windows、filter_design、filtering）和 `DeviceArray`、标准库的依赖关系。

**设计理由与替代方案**

每个 `#include` 引入一个明确的依赖模块，使得编译依赖可审计。使用 `<numeric>` 而非手写循环来汇总耗时，表达能力更强且不易出错。

**初学者易错点**

- `#include` 是预处理指令，不是运行时函数调用；`<numeric>` 的尖括号表示系统头文件搜索路径

### 4.5 命名空间和函数定义入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step2(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step2";
    const auto total_begin = Clock::now();
```

**语法结构**

命名空间、函数定义（含函数体开头）、多个局部变量声明与初始化。

**名称与类型**

- `namespace task2`：打开命名空间
- `run_step2(PipelineState& state)`：函数定义，`PipelineState&` 是非 const 引用参数
- `const auto& config`：`const auto&` 推导为 `const TaskConfig&`，只读引用避免拷贝
- `StepEvidence evidence`：默认构造的证据记录对象
- `evidence.name = "step2"`：字符串字面量赋值给 `std::string` 成员
- `const auto total_begin`：`auto` 推导为 `Clock::time_point`，记录总耗时起点

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置只读引用|无独立数学符号；包含 `filter_cutoff_hz`、`filter_taps`、`sample_rate_hz`、`samples`|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存各阶段耗时和 metrics，最终序列化|
|`total_begin`|总耗时计时起点|时间戳记作 $t_0$|与后续 `Clock::now()` 相减得 total_ms|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `namespace task2 {`：打开命名空间
- `PipelineState& state`：参数为引用，不复制对象
- `const auto& config = state.config`：声明只读引用 `config` 绑定到 `state.config`，不复制
- `StepEvidence evidence`：默认构造 `evidence` 对象
- `evidence.name = "step2"`：将字符串字面量 `"step2"` 赋值给 `evidence.name`
- `const auto total_begin = Clock::now()`：调用 `Clock::now()` 获取当前时间点，初始化 `total_begin`

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法入口|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块定义 `run_step2` 函数体，初始化 config 引用和 evidence 对象|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块建立函数入口和计时起点，是后续算法执行的前置条件|

**任务语义**

本块是 Step2 函数体的入口：建立只读配置引用、创建证据记录对象、启动总计时器。

**设计理由与替代方案**

`const auto&` 避免了拷贝 `TaskConfig` 对象（可能包含多个 vector 成员）。`Clock::now()` 在函数体开头调用，使 `total_ms` 能覆盖完整的执行时间。

**初学者易错点**

- `const auto&` 中的 `&` 是引用声明，不是取地址；`const` 保证不能通过该引用修改原对象
- `Clock::now()` 是静态成员函数调用，`::` 是作用域解析运算符

### 4.6 契约验证和参数准备

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    require(state.echo.size() == static_cast<std::size_t>(config.samples),
        "step2 did not receive step1 echo");
    const std::vector<float> cutoff{config.filter_cutoff_hz};
    const int taps_count = config.filter_taps;
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

变量声明与初始化、函数调用、赋值表达式。

**名称与类型**

- `auto begin`：`auto` 推导为 `Clock::time_point`，记录 prep 阶段计时起点
- `require(...)`：契约检查函数，第一个参数为 bool 条件，第二个参数为错误消息
- `state.echo.size()`：`std::vector<float>::size()` 返回 `std::size_t`（无符号整数）
- `static_cast<std::size_t>(config.samples)`：将 `int` 显式转换为 `std::size_t`，避免有符号/无符号比较警告
- `const std::vector<float> cutoff{...}`：使用列表初始化构造单元素向量
- `const int taps_count`：`const int`，FIR 滤波器阶数 L
- `evidence.prep_ms = milliseconds(begin, Clock::now())`：计算 prep 阶段耗时并赋值

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|prep 阶段计时起点|时间戳 $t_p$|与 `Clock::now()` 相减得 prep_ms|
|`echo`|Step1 输出的回波信号|记作 $echo[n]$|Step1 写入，Step2 读取并滤波|
|`cutoff`|截止频率列表|$f_c$（Hz）|从 config 读取，传给 firwin|
|`taps_count`|FIR 滤波器阶数|$L$|从 config 读取，决定窗长度和 FIR 系数数量|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `auto begin = Clock::now()`：记录 prep 阶段开始时间
- `require(...)`：运行时检查 `echo.size() == config.samples`，不满足时抛出错误
- `const std::vector<float> cutoff{config.filter_cutoff_hz}`：用列表初始化构造 `cutoff` 向量，包含一个元素 `filter_cutoff_hz`
- `const int taps_count = config.filter_taps`：从配置读取 FIR 阶数
- `evidence.prep_ms = milliseconds(begin, Clock::now())`：计算 prep 阶段耗时（毫秒）

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块验证 echo 大小契约，读取 cutoff 和 taps_count 配置参数|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入|

**任务语义**

本块完成 Step2 的 CPU 准备阶段：验证上游数据契约（echo 长度正确），从配置中提取滤波参数（截止频率、FIR 阶数），记录 prep 耗时。

**设计理由与替代方案**

`require` 在条件不满足时立即终止并报告错误，比静默返回或继续执行更安全。`static_cast<std::size_t>` 显式转换避免有符号/无符号比较的编译器警告。

**初学者易错点**

- `static_cast<std::size_t>(config.samples)`：`static_cast` 是编译期类型转换，不是函数调用；`<std::size_t>` 是模板参数
- `{config.filter_cutoff_hz}` 是列表初始化，花括号不是控制流作用域

### 4.7 H2D 数据传输

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);
    auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
```

**语法结构**

赋值、变量声明与初始化、函数调用。

**名称与类型**

- `auto d_echo`：`auto` 推导为 `cusignal::DeviceArray<float>`，封装 GPU 显存中的 echo 数组
- `cusignal::DeviceArray<float>::from_host(state.echo)`：静态工厂方法，在 GPU 上分配显存并将 Host 数据拷贝到 device
- `auto d_cutoff`：同理，封装 GPU 显存中的 cutoff 数组
- `evidence.h2d_ms = ...`：覆盖赋值（非 `+=`），记录 H2D 阶段耗时

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_echo`|GPU 端 echo 信号|与 Host 端 `echo` 数学含义相同|由 H2D 产生，传给 firfilter|
|`d_cutoff`|GPU 端截止频率|与 Host 端 `cutoff` 数学含义相同|由 H2D 产生，传给 firwin|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `begin = Clock::now()`：记录 H2D 阶段开始时间
- `DeviceArray<float>::from_host(state.echo)`：在 GPU 上分配 N 个 float 的显存，将 `state.echo` 的内容拷贝到 device
- `DeviceArray<float>::from_host(cutoff)`：在 GPU 上分配 1 个 float 的显存，将 cutoff 拷贝到 device
- `evidence.h2d_ms = milliseconds(begin, Clock::now())`：计算 H2D 耗时（毫秒）

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块将 echo 和 cutoff 从 Host 拷贝到 GPU 显存|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出|

**任务语义**

本块完成 H2D 数据传输：将 echo 信号（N 个 float）和 cutoff（1 个 float）从 Host 内存拷贝到 GPU 显存，为后续 GPU 算子链准备输入数据。

**设计理由与替代方案**

`DeviceArray::from_host` 是 RAII 封装：构造时分配显存并完成 H2D，析构时自动释放显存。使用 `evidence.h2d_ms =` 而非 `+=`，因为这是 Step2 唯一的 H2D 阶段。

**初学者易错点**

- `DeviceArray<float>::from_host` 是静态成员函数模板的显式实例化，`::` 是作用域解析
- `auto` 推导为 `DeviceArray<float>`，不是 `float*` 或 `std::vector<float>`

### 4.8 Hamming 窗生成

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `cusignal::DeviceArray<float> d_window(taps_count);`。

```cpp
    cusignal::DeviceArray<float> d_window(taps_count);
    evidence.operator_ms["hamming"] = time_gpu([&] {
        cusignal::hamming_resident_device<float>(taps_count, d_window, true);
    });
```

**语法结构**

变量声明与初始化、lambda 表达式、函数调用。

**名称与类型**

- `cusignal::DeviceArray<float> d_window(taps_count)`：在 GPU 上分配 L 个 float 的显存
- `time_gpu([&] { ... })`：执行 lambda 并计量 GPU 耗时，返回毫秒值
- `[&]`：lambda 按引用捕获当前作用域的所有变量
- `cusignal::hamming_resident_device<float>(taps_count, d_window, true)`：GPU resident 版 Hamming 窗生成

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_window`|GPU 端 Hamming 窗系数|$w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$|由 hamming 算子填充，传给 firwin|
|`taps_count`|窗长度 = FIR 阶数|$L$|决定窗系数数量|

**数字常量逐项说明**

- `true`：布尔字面量，表示 symmetric=true（对称窗），使窗两端对称，常用于 FIR 滤波器设计

**执行过程**

- `DeviceArray<float> d_window(taps_count)`：在 GPU 上分配 L 个 float 的显存
- `time_gpu([&] { ... })`：创建 lambda，执行内部的 `hamming_resident_device` 调用，并计量 GPU 耗时
- `hamming_resident_device<float>(taps_count, d_window, true)`：在 GPU 上生成 Hamming 窗系数，结果写入 `d_window`
- `evidence.operator_ms["hamming"] = ...`：将耗时记录到 evidence 的 operator_ms 映射中

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法（窗函数选择）|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块在 GPU 上生成 Hamming 窗系数，作为 FIR 滤波器设计的窗函数|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明窗函数已生成，但单独的窗系数不等于完整滤波|

**任务语义**

本块在 GPU 上生成 Hamming 窗系数 $w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$。Hamming 窗的旁瓣衰减约 −43 dB，是 FIR 滤波器设计的常用窗函数。`symmetric=true` 使窗两端对称，常用于滤波器设计。

**设计理由与替代方案**

使用 resident 版本（而非 H2D/D2H 版本）避免了中间数据传输开销。`time_gpu` 包装 lambda 确保在 GPU 操作完成后才停止计时。替代方案包括 Hann 窗（旁瓣更低但主瓣更宽）或 Blackman 窗（旁瓣更低但计算更复杂）。

**初学者易错点**

- `[&]` 是 lambda 捕获列表，`&` 表示按引用捕获，不是取地址
- `{ ... }` 是 lambda 函数体，花括号不是控制流作用域

### 4.9 FIR 滤波器系数设计

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `cusignal::DeviceArray<float> d_taps(taps_count);`。

```cpp
    cusignal::DeviceArray<float> d_taps(taps_count);
    evidence.operator_ms["firwin"] = time_gpu([&] {
        cusignal::firwin_lowpass_explicit_window_resident_device(
            taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true);
    });
```

**语法结构**

变量声明与初始化、lambda 表达式、跨行函数调用。

**名称与类型**

- `cusignal::DeviceArray<float> d_taps(taps_count)`：在 GPU 上分配 L 个 float 的显存
- `firwin_lowpass_explicit_window_resident_device`：GPU resident 版 FIR 低通滤波器系数设计（显式窗模式）
- `config.sample_rate_hz`：采样率 $f_s$（Hz），用于计算归一化截止频率 $f_c/f_s$
- 最后一个 `true`：scale 参数，表示归一化使直流增益为 1（$\sum h[m] = 1$）

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_taps`|GPU 端 FIR 滤波器系数|$h[m] = h_d[m] \times w[m]$|由 firwin 算子填充，传给 firfilter|
|`d_cutoff`|GPU 端截止频率|$f_c$（Hz）|传给 firwin 作为低通截止频率|
|`d_window`|GPU 端 Hamming 窗系数|$w[m]$|传给 firwin 作为窗函数|
|`sample_rate_hz`|采样率|$f_s$（Hz）|用于计算归一化截止频率|

**数字常量逐项说明**

- `true`：布尔字面量，scale=true 表示归一化使 $\sum h[m] = 1$（直流增益为 1）

**执行过程**

- `DeviceArray<float> d_taps(taps_count)`：在 GPU 上分配 L 个 float 的显存
- `time_gpu([&] { ... })`：计量 GPU 耗时
- `firwin_lowpass_explicit_window_resident_device(...)`：在 GPU 上执行窗函数法 FIR 滤波器设计
  - 计算理想低通冲激响应 $h_d[m] = 2\frac{f_c}{f_s}\text{sinc}(2\pi\frac{f_c}{f_s}(m - \frac{L-1}{2}))$
  - 应用 Hamming 窗 $h[m] = h_d[m] \times w[m]$
  - scale 归一化 $h[m] = h[m] / \sum h[m]$
- `evidence.operator_ms["firwin"] = ...`：记录耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法（滤波器设计）|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块在 GPU 上用窗函数法设计 FIR 低通滤波器系数|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明 FIR 系数已生成，但单独的系数不等于完整滤波|

**任务语义**

本块在 GPU 上用窗函数法设计 FIR 低通滤波器系数。核心步骤：(1) 计算理想低通 sinc 冲激响应；(2) 乘以 Hamming 窗抑制旁瓣；(3) scale 归一化使直流增益为 1。`explicit_window` 模式使用预计算的 Hamming 窗，而非在 firwin 内部重新生成。

**设计理由与替代方案**

使用 `explicit_window` 模式（而非 `window_name` 模式）使窗函数生成和滤波器系数设计解耦，便于分别计时和分析。`scale=true` 确保滤波后信号幅度不变（直流增益为 1）。

**初学者易错点**

- `firwin_lowpass_explicit_window_resident_device` 是函数名，`resident` 表示 GPU 端直接分配和计算
- 跨行调用中，续行参数必须与函数名一起阅读，不要单独求值

### 4.10 FIR 卷积滤波

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `cusignal::DeviceArray<float> filtered(config.samples);`。

```cpp
    cusignal::DeviceArray<float> filtered(config.samples);
    evidence.operator_ms["firfilter"] = time_gpu([&] {
        cusignal::firfilter_fp32_resident_device(d_taps, d_echo, filtered);
    });
```

**语法结构**

变量声明与初始化、lambda 表达式、函数调用。

**名称与类型**

- `cusignal::DeviceArray<float> filtered(config.samples)`：在 GPU 上分配 N 个 float 的显存用于输出
- `firfilter_fp32_resident_device`：GPU resident 版 FIR 卷积滤波（fp32 精度）
- `d_taps`：FIR 滤波器系数 $h[m]$（输入）
- `d_echo`：回波信号 $echo[n]$（输入）
- `filtered`：滤波输出 $y[n]$（输出）

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`filtered`|GPU 端滤波输出|$y[n] = \sum_{m=0}^{L-1} h[m] \times echo[n-m]$|由 firfilter 算子填充，D2H 后传给 Step3|
|`d_taps`|FIR 滤波器系数|$h[m]$|firwin 输出，firfilter 输入|
|`d_echo`|回波信号|$echo[n]$|Step1 输出，firfilter 输入|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `DeviceArray<float> filtered(config.samples)`：在 GPU 上分配 N 个 float 的显存
- `time_gpu([&] { ... })`：计量 GPU 耗时
- `firfilter_fp32_resident_device(d_taps, d_echo, filtered)`：在 GPU 上执行 FIR 卷积滤波
  - 对每个输出采样点 n，计算内积 $\sum_{m=0}^{L-1} h[m] \times echo[n-m]$
  - 结果写入 `filtered`
- `evidence.operator_ms["firfilter"] = ...`：记录耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法（滤波执行）|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块在 GPU 上执行 FIR 卷积滤波，输出 filtered|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明 FIR 滤波算法已执行，输出 filtered 是 Step2 的核心业务结果|

**任务语义**

本块是 Step2 的核心计算步骤：在 GPU 上执行 FIR 卷积滤波。时域卷积 $y[n] = \sum h[m] \times echo[n-m]$ 等价于频域乘积 $Y(f) = H(f) \times X(f)$，低通滤波器保留 $|f| < f_c$ 的低频目标回波成分，衰减高频噪声和干扰。

**设计理由与替代方案**

`fp32` 精度（而非 fp64）在 GPU 上计算效率更高，且对滤波应用来说精度足够。resident 版本避免了中间 H2D/D2H 传输。

**初学者易错点**

- `filtered` 是输出缓冲区，在调用前已分配但内容未初始化；firfilter 会覆盖所有元素
- `firfilter_fp32_resident_device` 的前两个参数是输入（只读），第三个参数是输出（写入）

### 4.11 耗时汇总

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `evidence.compute_ms = std::accumulate(...)`。

```cpp
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
```

**语法结构**

函数调用（`std::accumulate`）与 lambda 表达式。

**名称与类型**

- `std::accumulate`：标准库归约算法，遍历范围并累加
- `evidence.operator_ms.begin()` / `end()`：`std::map<std::string, double>` 的迭代器
- `0.0`：`double` 类型初始值（加法单位元）
- `[](double sum, const auto& item) { return sum + item.second; }`：lambda，累加每个 operator 的耗时（`item.second`）

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`compute_ms`|总计算耗时|$T_{compute} = \sum T_i$（各算子耗时之和）|汇总三个算子的耗时|
|`sum`|累加器|$S = \sum_i a_i$|每轮累加当前 item 的耗时|
|`item`|operator_ms 中的键值对|$a_i$ = 第 i 个算子的耗时|由迭代器产生|

**数字常量逐项说明**

- `0.0`：`double` 类型加法单位元，作为 `std::accumulate` 的初始累加值（$S_0 = 0$）

**执行过程**

- `std::accumulate` 从 `evidence.operator_ms.begin()` 遍历到 `end()`
- 初始累加值 `sum = 0.0`
- 对每个键值对 `item`，lambda 执行 `sum = sum + item.second`（`item.second` 是算子耗时，单位毫秒）
- 最终 `compute_ms = T_hamming + T_firwin + T_firfilter`

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法（证据收集）|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块汇总三个 GPU 算子的耗时，写入 evidence.compute_ms|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明各算子耗时已被计量和汇总|

**任务语义**

本块汇总三个 GPU 算子（hamming、firwin、firfilter）的耗时，得到总计算时间。`std::accumulate` 是表达力强的归约方式，比手写 for 循环更简洁。

**设计理由与替代方案**

使用 `std::accumulate` 而非手写循环，代码意图更清晰。lambda 的 `auto&` 参数避免拷贝键值对。

**初学者易错点**

- `item.second` 是 `std::pair<std::string, double>` 的第二个成员（耗时值），不是"第二项"
- `0.0` 是 `double` 类型，决定 `std::accumulate` 的累加类型为 `double`
- lambda 的 `const auto& item` 中 `&` 是引用，避免每次迭代拷贝键值对

### 4.12 D2H 数据传输

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    state.filtered = filtered.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
```

**语法结构**

赋值、函数调用、复合赋值。

**名称与类型**

- `begin = Clock::now()`：更新计时起点
- `filtered.to_host()`：`DeviceArray::to_host()` 将 GPU 显存内容拷贝回 Host，返回 `std::vector<float>`
- `state.filtered = ...`：将滤波结果写入流水线状态，供 Step3 使用
- `evidence.d2h_ms += ...`：复合赋值（`+=`），累加 D2H 耗时

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state.filtered`|滤波后信号|$y[n]$（滤波输出）|Step2 写入，Step3 读取|
|`filtered`|GPU 端滤波输出|$y[n]$|由 firfilter 产生，D2H 后释放|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `begin = Clock::now()`：记录 D2H 阶段开始时间
- `filtered.to_host()`：将 GPU 显存中的滤波结果拷贝到 Host 内存，返回 `std::vector<float>`
- `state.filtered = ...`：将滤波结果赋值给流水线状态
- `evidence.d2h_ms += milliseconds(begin, Clock::now())`：累加 D2H 耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法（输出交付）|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块将滤波结果从 GPU 传回 Host，写入 state.filtered|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务输出已交付给流水线状态|

**任务语义**

本块完成 D2H 数据传输：将 GPU 端滤波结果 filtered 拷贝回 Host 内存，写入 `state.filtered`，供 Step3 B 样条平滑消费。

**设计理由与替代方案**

使用 `+=` 累加 D2H 耗时（而非 `=`），因为 Step2 还有可视化捕获时的额外 D2H（见 4.14）。`to_host()` 返回 `std::vector<float>` 而非指针，由 RAII 管理内存生命周期。

**初学者易错点**

- `evidence.d2h_ms +=` 是复合赋值（先读取旧值，加上新值，再写回），不是简单赋值 `=`
- `filtered.to_host()` 是成员函数调用，`.to_host()` 触发 D2H 传输

### 4.13 证据指标记录

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `evidence.metrics["resident_internal_h2d_count"] = 0.0;`。

```cpp
    evidence.metrics["resident_internal_h2d_count"] = 0.0;
    evidence.metrics["resident_internal_d2h_count"] = 0.0;
    evidence.metrics["resident_internal_allocation_count"] = 0.0;
    evidence.metrics["resident_kernel_count"] = 3.0;
    evidence.metrics["resident_persistent_device_bytes"] =
        static_cast<double>(config.samples * sizeof(float)
            + sizeof(float)
            + taps_count * sizeof(float) * 3);
```

**语法结构**

多条赋值语句，其中最后一条为跨行表达式。

**名称与类型**

- `evidence.metrics["..."]`：`std::map<std::string, double>` 的下标访问，键不存在时插入默认值
- `resident_internal_h2d_count` / `resident_internal_d2h_count` / `resident_internal_allocation_count`：均为 0.0，因为 resident 算子不经过显式 H2D/D2H/分配
- `resident_kernel_count`：3.0，对应三个 GPU kernel（hamming、firwin、firfilter）
- `resident_persistent_device_bytes`：GPU 显存持久占用估算
- `sizeof(float)`：`float` 类型的字节数（4）
- `static_cast<double>(...)`：将整数表达式结果转换为 double

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`resident_internal_h2d_count`|resident 算子内部 H2D 次数|0（resident 不经过 H2D）|统计指标|
|`resident_internal_d2h_count`|resident 算子内部 D2H 次数|0（resident 不经过 D2H）|统计指标|
|`resident_internal_allocation_count`|resident 算子内部分配次数|0|统计指标|
|`resident_kernel_count`|GPU kernel 启动次数|3（hamming + firwin + firfilter）|统计指标|
|`resident_persistent_device_bytes`|GPU 显存持久占用|$N \times 4 + 4 + L \times 4 \times 3$ 字节|资源使用统计|

**数字常量逐项说明**

- `0.0`：`double` 类型零值，表示 resident 算子不经过显式 H2D/D2H/分配
- `3.0`：`double` 类型，表示三个 GPU kernel 启动；源码未记录推导依据，由当前仓库源码直接写入
- `3`：`int` 类型，在 `sizeof(float) * 3` 中表示三个 device 缓冲区（window + taps + filtered）
- `sizeof(float)`：编译期常量，`float` 类型字节数（4）

**执行过程**

- 前三条赋值：将 resident 内部统计设为 0.0（resident 算子不经过显式传输/分配）
- `evidence.metrics["resident_kernel_count"] = 3.0`：记录 kernel 启动次数
- 跨行表达式计算显存持久占用：
  - `config.samples * sizeof(float)`：echo 数组占用（N × 4 字节）
  - `sizeof(float)`：cutoff 占用（1 × 4 字节）
  - `taps_count * sizeof(float) * 3`：window + taps + filtered 占用（L × 4 × 3 字节）
  - `static_cast<double>(...)`：将结果转为 double

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块记录 resident 算子统计和显存占用指标|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明资源使用指标已被记录，但指标本身不等于算法正确性|

**任务语义**

本块记录 evidence 的 metrics 指标：resident 算子内部统计（均为 0）、kernel 启动次数（3）、GPU 显存持久占用估算。这些指标用于性能分析和资源使用评估。

**设计理由与替代方案**

`resident_internal_*` 均为 0 是因为 resident 版本在 device 端直接分配和计算，不经过显式 H2D/D2H。显式记录这些统计便于统一证据格式，即使当前值为 0。`static_cast<double>` 确保 metrics 值统一为 double 类型。

**初学者易错点**

- `sizeof(float)` 是编译期运算符，不是运行时函数调用；它返回类型的字节数（4）
- `*` 是乘法运算符，`sizeof` 优先级高于 `*`
- `3` 在 `* 3` 中是乘法因子，`3.0` 在 `= 3.0` 中是 double 字面量，两者类型不同但语义相关

### 4.14 总耗时和可视化捕获

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`。

```cpp
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.filter_taps = d_taps.to_host();
    return evidence;
```

**语法结构**

赋值、if 条件语句、return 语句。

**名称与类型**

- `evidence.total_ms = ...`：记录总耗时（从 total_begin 到此刻）
- `visualization_capture_enabled()`：全局函数，返回 bool，判断是否启用可视化数据捕获
- `state.filter_taps = d_taps.to_host()`：条件执行，将 FIR 系数从 GPU 传回 Host 保存
- `return evidence`：返回 evidence 对象（NRVO 优化，不拷贝）

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`total_ms`|总耗时|$T_{total} = t_{now} - t_0$|从函数入口到此刻的 wall clock 时间|
|`filter_taps`|FIR 滤波器系数|$h[m]$|仅在可视化模式下保存，供绘图分析|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `evidence.total_ms = milliseconds(total_begin, Clock::now())`：计算从函数入口到此刻的总耗时
- `if (visualization_capture_enabled())`：条件判断，仅在可视化模式启用时执行
- `state.filter_taps = d_taps.to_host()`：将 FIR 系数从 GPU 传回 Host，保存到流水线状态
- `return evidence`：返回 evidence 对象，调用者接收 Step2 的完整证据

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法（证据交付）|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块记录总耗时，条件保存 FIR 系数，返回完整证据|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明 Step2 流水线已完整执行，证据已交付|

**任务语义**

本块完成 Step2 的收尾工作：记录总耗时、条件保存可视化数据、返回 evidence。`visualization_capture_enabled()` 控制是否将 FIR 系数从 GPU 传回 Host 保存到 `state.filter_taps`，用于后续绘图分析。

**设计理由与替代方案**

可视化数据捕获是条件执行的，避免在非可视化模式下产生不必要的 D2H 传输开销。`return evidence` 利用 NRVO（Named Return Value Optimization）避免拷贝。

**初学者易错点**

- `if` 是条件选择语句，条件为真时只执行其直接子语句 `state.filter_taps = d_taps.to_host();`
- `return` 是跳转语句，结束当前函数并将 evidence 返回给调用者

### 4.15 命名空间闭合

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

```cpp
}  // namespace task2
```

**语法结构**

右花括号关闭命名空间作用域。

**名称与类型**

- `}  // namespace task2`：关闭 `namespace task2` 的作用域

**变量—公式—任务状态映射**

当前块只关闭词法作用域，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 右花括号关闭 `namespace task2` 的作用域

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只关闭词法作用域，不产生新的业务数组或数值结果|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只管理作用域，不能冒充核心算法|

**任务语义**

本块只关闭命名空间作用域，没有独立的任务语义。

**设计理由与替代方案**

`// namespace task2` 注释是代码风格约定，帮助快速识别闭合的命名空间。

**初学者易错点**

- 花括号本身不产生运行时操作；它只是编译期作用域边界

### 4.16 CPU 参考实现：验证数据收集入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `Step2ValidationData collect_step2_validation(const PipelineState& state)`。

```cpp
Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data;
    const int taps_count = state.config.filter_taps;
    const std::vector<float> cutoff{state.config.filter_cutoff_hz};
```

**语法结构**

函数定义、局部变量声明与初始化。

**名称与类型**

- `collect_step2_validation`：CPU 验证数据收集函数
- `const PipelineState& state`：const 引用参数（只读），不修改流水线状态
- `Step2ValidationData data`：返回数据结构，包含 `filtered_cpu`、`target_filtered_cpu` 等字段
- `const int taps_count`：FIR 滤波器阶数 L
- `const std::vector<float> cutoff{...}`：截止频率列表

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step2_validation`|CPU 验证数据收集函数|无独立数学符号|生成 CPU 参考基准，供 GPU/CPU 精度对比|
|`state`|任务流水线共享状态|无单一数学符号|只读消费 echo、target_component 和 config|
|`data`|验证数据结构|无单一数学符号|字段分别映射到 filtered_cpu、target_filtered_cpu|
|`taps_count`|FIR 滤波器阶数|$L$|从 config 读取|
|`cutoff`|截止频率列表|$f_c$（Hz）|从 config 读取|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 函数定义：`const PipelineState& state` 为只读引用参数
- `Step2ValidationData data`：默认构造验证数据结构
- `const int taps_count = state.config.filter_taps`：从配置读取 FIR 阶数
- `const std::vector<float> cutoff{...}`：用列表初始化构造截止频率列表

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块定义 CPU 验证数据收集函数，读取配置参数|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径存在；不能替代核心实现，也不能单独证明历史运行结果真实发生|

**任务语义**

本块是 CPU 参考实现的入口：接收只读流水线状态，创建验证数据结构，从配置中读取滤波参数。`const PipelineState&` 确保不会意外修改流水线状态。

**设计理由与替代方案**

`const` 引用参数确保函数不会修改 `state`，这是验证函数的正确语义——它只读取数据生成参考基准，不应修改流水线状态。

**初学者易错点**

- `const PipelineState& state`：`const` 修饰的是 `PipelineState`（不能通过该引用修改对象），`&` 是引用声明
- `Step2ValidationData data`：类类型默认构造，字段初始值由构造函数决定

### 4.17 CPU 参考实现：Hamming 窗和 FIR 系数设计

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);`。

```cpp
    const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);
    cusignal::FirwinOptions design_options;
    design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;
    design_options.window = window_cpu;
    design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;
    design_options.scale = true;
    design_options.fs = state.config.sample_rate_hz;
    const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));
```

**语法结构**

变量声明与初始化、对象声明与字段赋值、跨行函数调用。

**名称与类型**

- `const auto window_cpu`：`auto` 推导为 `std::vector<float>`，CPU 版 Hamming 窗系数
- `cusignal::hamming_typed_cpu<float>(taps_count, true)`：CPU 版 Hamming 窗生成（`float` 模板参数）
- `cusignal::FirwinOptions design_options`：FIR 滤波器设计选项结构体
- `cusignal::FirwinWindowMode::explicit_values`：枚举值，表示使用显式预计算窗函数
- `cusignal::FirwinPassZero::boolean_true`：枚举值，表示低通（DC 附近频带通过）
- `design_options.scale = true`：布尔值，表示归一化使直流增益为 1
- `design_options.fs = state.config.sample_rate_hz`：采样率 $f_s$
- `const auto taps_cpu`：`auto` 推导为 `std::vector<float>`，CPU 版 FIR 系数
- `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(...)`：将 fp64 数据窄化为 fp32

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`window_cpu`|CPU 版 Hamming 窗系数|$w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$|生成后传给 firwin 作为窗函数|
|`design_options`|FIR 滤波器设计选项|无独立数学符号|配置窗模式、低通/高通、scale、采样率|
|`taps_cpu`|CPU 版 FIR 滤波器系数|$h[m] = h_d[m] \times w[m]$|生成后传给 firfilter|

**数字常量逐项说明**

- `true`（第一次）：布尔字面量，symmetric=true 表示对称 Hamming 窗
- `true`（第二次）：布尔字面量，scale=true 表示归一化使 $\sum h[m] = 1$

**执行过程**

- `hamming_typed_cpu<float>(taps_count, true)`：在 CPU 上生成 Hamming 窗系数
- `design_options.window_mode = explicit_values`：设置窗模式为显式预计算窗
- `design_options.window = window_cpu`：传入预计算的 Hamming 窗
- `design_options.pass_zero = boolean_true`：设置低通模式
- `design_options.scale = true`：启用 scale 归一化
- `design_options.fs = sample_rate_hz`：设置采样率
- `firwin_typed_cpu(taps_count, cutoff, design_options)`：在 CPU 上生成 FIR 系数（fp64 输出）
- `narrow_fp64_to_fp32_on_host(...)`：将 fp64 结果窄化为 fp32，与 GPU 版本精度一致

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块在 CPU 上生成 Hamming 窗和 FIR 系数，作为 GPU 精度验证的基准|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径存在；不能替代核心实现|

**任务语义**

本块在 CPU 上生成 Hamming 窗和 FIR 滤波器系数，作为 GPU 精度验证的基准。`firwin_typed_cpu` 输出 fp64 精度，通过 `narrow_fp64_to_fp32_on_host` 窄化为 fp32 以匹配 GPU 版本精度。这与 GPU 版本的 `firwin_lowpass_explicit_window_resident_device` 使用相同的数学公式和参数。

**设计理由与替代方案**

CPU 版本的 firwin 输出 fp64（双精度），需要窄化为 fp32 才能与 GPU 版本（fp32）做精度对比。`narrow_fp64_to_fp32_on_host` 在 Host 端执行转换，避免 GPU kernel 内的类型转换开销。

**初学者易错点**

- `cusignal::FirwinWindowMode::explicit_values` 是枚举值，`::` 是作用域解析运算符
- `cusignal::FirwinPassZero::boolean_true` 表示低通（DC 通过），`boolean_true` 是枚举值名称，不是 bool 类型
- 跨行调用中，`cusignal::firwin_typed_cpu(...)` 的返回值直接作为 `narrow_fp64_to_fp32_on_host` 的参数

### 4.18 CPU 参考实现：FIR 滤波和结果返回

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `cusignal::FirfilterOptions filter_options;`。

```cpp
    cusignal::FirfilterOptions filter_options;
    filter_options.shape = {state.config.samples};
    filter_options.axis = -1;
    data.filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.echo, filter_options).y;
    data.target_filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.target_component, filter_options).y;
    return data;
```

**语法结构**

对象声明与字段赋值、跨行函数调用、return 语句。

**名称与类型**

- `cusignal::FirfilterOptions filter_options`：FIR 滤波选项结构体
- `filter_options.shape = {state.config.samples}`：列表初始化形状为 `{N}`
- `filter_options.axis = -1`：沿最后一维滤波（一维数组情况下即沿第 0 维）
- `data.filtered_cpu`：CPU 版 echo 滤波结果
- `data.target_filtered_cpu`：CPU 版 target 分量滤波结果
- `cusignal::firfilter_typed_cpu(...).y`：调用 CPU 版 FIR 滤波，`.y` 访问输出信号的成员

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`filter_options`|FIR 滤波选项|无独立数学符号|配置输出形状和滤波轴|
|`filtered_cpu`|CPU 版 echo 滤波结果|$y[n] = \sum h[m] \times echo[n-m]$|与 GPU 版 filtered 做精度对比|
|`target_filtered_cpu`|CPU 版 target 分量滤波结果|$y_{target}[n] = \sum h[m] \times target[n-m]$|验证目标信号在滤波后的保留情况|

**数字常量逐项说明**

- `-1`：`int` 类型，表示最后一维（Python/NumPy 风格），一维数组中等价于 axis=0

**执行过程**

- `FirfilterOptions filter_options`：默认构造滤波选项
- `filter_options.shape = {state.config.samples}`：设置输出形状为 `{N}`（一维，N 个元素）
- `filter_options.axis = -1`：设置滤波轴为最后一维
- `firfilter_typed_cpu(taps_cpu, state.echo, filter_options).y`：对 echo 执行 FIR 滤波，`.y` 获取输出信号
- `firfilter_typed_cpu(taps_cpu, state.target_component, filter_options).y`：对 target 分量执行 FIR 滤波
- `return data`：返回完整的验证数据集

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块在 CPU 上执行 FIR 滤波，生成 filtered_cpu 和 target_filtered_cpu 基准|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径存在；不能替代核心实现，也不能单独证明历史运行结果真实发生|

**任务语义**

本块在 CPU 上执行两次 FIR 滤波：(1) 对 echo 信号滤波，生成 `filtered_cpu` 作为 GPU 精度验证基准；(2) 对 target_component 分量滤波，生成 `target_filtered_cpu` 用于分析目标信号在滤波后的保留情况。两次滤波使用相同的 FIR 系数和滤波选项。

**设计理由与替代方案**

对 `target_component` 单独滤波是额外的验证步骤：分析目标信号（纯 chirp 延迟版）在滤波后的幅度衰减，验证滤波器是否在去除噪声的同时保留了目标回波成分。如果 `target_filtered_cpu` 与 `target_component` 差异过大，说明截止频率 $f_c$ 设置过低，目标信号也被衰减了。

**初学者易错点**

- `.y` 是成员访问，`firfilter_typed_cpu` 返回一个包含 `.y`（输出信号）的结构体
- `axis = -1` 是 Python/NumPy 风格的负索引约定，一维数组中等价于 `axis = 0`
- `return data` 利用 NRVO 优化，不拷贝 `Step2ValidationData` 对象

---

## 5. 自检问题

### 5.1 版本身份

1. Step2 当前学习版本绑定哪个 Git SHA？该提交时间是什么？
2. Step2 涉及哪些源码文件？每个文件的作用是什么？
3. Step2 的 CPU 参考实现和 GPU 实现分别在哪些文件中？

### 5.2 CPU/GPU 职责

4. `run_step2` 的几个处理阶段分别是什么？每个阶段做什么？
5. `collect_step2_validation` 的职责是什么？它生成哪些验证数据？
6. CPU 版本和 GPU 版本的 FIR 系数设计有什么差异？为什么需要 `narrow_fp64_to_fp32_on_host`？

### 5.3 数学映射

7. Step2 的核心数学公式是什么？各符号的含义是什么？
8. Hamming 窗的数学公式是什么？任务中使用的参数是什么？
9. 窗函数法 FIR 滤波器设计的数学步骤是什么？
10. `scale=true` 的数学含义是什么？为什么需要它？

### 5.4 模板/类型

11. `DeviceArray<float>::from_host` 的模板参数是什么？为什么使用 `float` 而非 `double`？
12. `FirwinWindowMode::explicit_values` 和 `FirwinPassZero::boolean_true` 分别表示什么？

### 5.5 并行/同步

13. 三个 GPU 算子（hamming、firwin、firfilter）的并行策略是什么？
14. `time_gpu` 如何确保 GPU 操作完成后才停止计时？
15. resident 版本和普通 H2D/D2H 版本的区别是什么？为什么 resident 版本的内部统计均为 0？

### 5.6 精度/测试边界

16. CPU 和 GPU 版本之间如何做精度验证？
17. 为什么 CPU 版本要对 `target_component` 单独滤波？
18. `resident_persistent_device_bytes` 的计算公式是什么？各部分的含义是什么？

---

## 6. 参考答案

### 6.1 版本身份

1. **SHA**：`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`，提交时间 `2026-08-18T20:22:35+08:00`。
2. **三个文件**：
   - `step2.h`：头文件，声明 `run_step2` 接口
   - `step2.cu`：实现文件，包含 `run_step2` 的 GPU 流水线编排
   - `step2_validation.cpp`：CPU 参考实现，包含 `collect_step2_validation` 用于精度验证
3. GPU 实现在 `step2.cu` 中，CPU 参考实现在 `step2_validation.cpp` 中。

### 6.2 CPU/GPU 职责

4. **四个阶段**：
   - 阶段1（CPU 准备）：验证 echo 契约，读取 cutoff 和 taps_count
   - 阶段2（H2D）：将 echo 和 cutoff 拷贝到 GPU
   - 阶段3（GPU 算子链）：Hamming 窗 → FIR 系数设计 → FIR 卷积滤波
   - 阶段4（D2H）：将 filtered 传回 Host，记录证据
5. `collect_step2_validation` 在 CPU 上生成 FIR 滤波的参考基准，输出 `filtered_cpu`（echo 滤波结果）和 `target_filtered_cpu`（target 分量滤波结果），用于与 GPU 版本做精度对比。
6. CPU 版本的 `firwin_typed_cpu` 输出 fp64 精度，GPU 版本输出 fp32。需要 `narrow_fp64_to_fp32_on_host` 将 CPU 结果窄化为 fp32 才能与 GPU 版本做精度对比。

### 6.3 数学映射

7. **核心公式**：$filtered[n] = \sum_{m=0}^{L-1} h[m] \times echo[n-m]$，其中 $h[m] = h_d[m] \times w[m]$，$h_d[m] = 2\frac{f_c}{f_s}\text{sinc}(2\pi\frac{f_c}{f_s}(m - \frac{L-1}{2}))$，$w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$。
   - $L$ = `taps_count`（FIR 阶数），$f_c$ = `filter_cutoff_hz`（截止频率），$f_s$ = `sample_rate_hz`（采样率）
8. **Hamming 窗公式**：$w[m] = 0.54 - 0.46\cos(2\pi m/(L-1))$，$m = 0, 1, ..., L-1$。任务参数：`taps_count` = $L$，`symmetric=true`。
9. **窗函数法步骤**：(1) 计算理想低通冲激响应 $h_d[m]$（sinc 函数）；(2) 乘以 Hamming 窗 $w[m]$ 得到 $h[m]$；(3) scale 归一化使 $\sum h[m] = 1$。
10. `scale=true` 的数学含义是归一化：$h[m] = h[m] / \sum h[m]$，使得直流增益为 1（$H(0) = 1$），滤波后信号幅度不变。

### 6.4 模板/类型

11. 模板参数是 `float`，因为 Step1 的 echo 信号是 `float` 类型，且 fp32 在 GPU 上计算效率更高。
12. `explicit_values` 表示使用预计算的显式窗函数值（而非让 firwin 内部生成窗）；`boolean_true` 表示低通滤波器（DC 附近频带通过）。

### 6.5 并行/同步

13. 每个算子的并行策略由其各自实现决定（详见算子学习文档）。hamming 和 firwin 通常每抽头系数一个线程，firfilter 每输出采样点一个线程（内积归约）。
14. `time_gpu` 在执行 lambda 后调用 `cudaDeviceSynchronize()` 确保 GPU 操作完成，然后才计算耗时。
15. resident 版本在 device 端直接分配 buffers 和计算，不经过显式 H2D/D2H。因此内部统计（h2d_count、d2h_count、allocation_count）均为 0。

### 6.6 精度/测试边界

16. 通过 `collect_step2_validation` 生成 CPU 参考基准 `filtered_cpu`，与 GPU 输出的 `state.filtered` 逐元素比较（通过 validation 框架的 comparison 函数）。
17. 对 `target_component` 单独滤波是为了验证目标信号在滤波后的保留情况：如果 `target_filtered_cpu` 与 `target_component` 差异过大，说明截止频率 $f_c$ 设置过低，目标信号也被衰减了。
18. 公式：$N \times 4 + 4 + L \times 4 \times 3$ 字节。各部分：
    - $N \times 4$：echo 数组（N 个 float）
    - $4$：cutoff（1 个 float）
    - $L \times 4 \times 3$：window + taps + filtered（各 L 个 float）