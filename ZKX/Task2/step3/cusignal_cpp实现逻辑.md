# Task2 Step3 cusignal_cpp 实现逻辑

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
|`ZKX/Task2/task2_gpu_cpu/step3/step3.h`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step3/step3.cu`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#pragma once

#include "../task2_common.h"

// <学习注释：Step3 入口函数 —— B 样条平滑流水线。
//
//  【函数总体处理方法】
//  run_step3 负责对 Step2 滤波后的信号进行 B 样条平滑，进一步弱化残余噪声。
//  处理流程分六个阶段：
//    阶段1（CPU）：验证上游数据契约，裁剪信号两端以消除边界效应，
//                定义三次 B 样条偏移量数组
//    阶段2（H2D）：将裁剪信号和偏移量拷贝到 GPU 显存
//    阶段3（GPU 权重计算）：调用 cubic_device 在 GPU 上计算三次 B 样条核权重
//    阶段4（D2H + CPU 归一化）：将权重回传 Host，归一化使权重和为 1，
//                              再上传回 GPU
//    阶段5（GPU 平滑滤波）：用归一化权重作为 FIR 系数，对裁剪信号执行卷积平滑
//    阶段6（D2H + CPU 参考）：将平滑结果回传 Host，准备与目标波形的互相关参考
//
//  【核心数学公式 —— 三次 B 样条平滑】
//    对裁剪后的信号 x_cropped[n]（n = 0, ..., N_crop−1），平滑结果为：
//      smoothed[n] = Σ_{k=0}^{8} w[k] × x_cropped[n − k]
//
//    其中权重 w[k] 由三次 B 样条基函数在偏移量 τ ∈ {−2.0, −1.5, ..., 2.0} 上求值：
//      w[k] = cubic(τ_k) / Σ_{j=0}^{8} cubic(τ_j)  （归一化）
//
//    三次 B 样条基函数（分段三次多项式）：
//      cubic(τ) = { (2−|τ|)³/6 − 2(1−|τ|)³/3 }  0 ≤ |τ| < 1
//               = { (2−|τ|)³/6 }                 1 ≤ |τ| < 2
//               = 0                              |τ| ≥ 2
//
//    符号说明：
//      N_crop = samples − 2×crop_each ：裁剪后信号长度
//      τ_k    = −2.0, −1.5, ..., 2.0 ：9 个偏移量（步长 0.5）
//      w[k]   = cubic(τ_k) / Σ_{j=0}^{8} cubic(τ_j)  ：归一化三次 B 样条权重
//
//  【物理意义】
//  B 样条平滑是一种低通滤波：用三次 B 样条核与信号卷积，在保留信号主要趋势的
//  同时进一步平滑残余高频噪声。信号两端各裁剪 crop_each 个样本以消除 FIR 滤波
//  的边界效应。平滑后的信号将作为 Step4 多域特征提取的输入。这是 T2-R3（B 样条
//  平滑）条款的核心实现。>
namespace task2 {

StepEvidence run_step3(PipelineState& state);

}  // namespace task2

```

### 3.2 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#include "step3.h"

#include "bsplines/bsplines_typed.h"
#include "cuda_utils/device_array.h"
#include "filtering/filtering_typed.h"

#include <numeric>

namespace task2 {

// <学习注释：GPU 入口函数 —— B 样条平滑流水线编排。
//
//  【函数总体处理方法】
//  run_step3 按以下阶段编排 B 样条平滑流水线：
//    阶段1（CPU 准备）：验证 filtered 契约，裁剪信号两端，定义偏移量
//    阶段2（H2D）：将裁剪信号和偏移量拷贝到 GPU
//    阶段3（GPU 权重计算）：cubic_device 在 GPU 上计算三次 B 样条核权重
//    阶段4（D2H + CPU 归一化）：权重回传 → 验证 sum > 0 → 归一化 → 上传回 GPU
//    阶段5（GPU 平滑滤波）：firfilter_device 用归一化权重对裁剪信号卷积
//    阶段6（D2H + CPU 参考）：平滑结果回传，准备与目标波形的互相关参考
//
//  【核心数学公式 —— 三次 B 样条核与卷积平滑】
//    步骤1 —— 三次 B 样条基函数求值：
//      cubic(τ) = { (2−|τ|)³/6 − 2(1−|τ|)³/3 }  |τ| < 1
//               = { (2−|τ|)³/6 }                 1 ≤ |τ| < 2
//               = 0                               |τ| ≥ 2
//
//    步骤2 —— 权重归一化：
//      w[k] = cubic(τ_k) / Σ_{j=0}^{8} cubic(τ_j)
//
//    步骤3 —— FIR 卷积平滑：
//      smoothed[n] = Σ_{k=0}^{8} w[k] × cropped[n − k]
//
//  【并行策略】
//  cubic_device 和 firfilter_device 的并行策略由其各自实现决定（详见算子学习
//  文档）。本函数负责编排调用顺序、H2D/D2H 传输、权重归一化和证据收集。>
StepEvidence run_step3(PipelineState& state)
{
    // <学习注释：获取只读配置引用，避免拷贝。
    //  config 包含：step3_crop_each（每端裁剪样本数）、samples（总采样数）。>
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step3";
    // <学习注释：记录总耗时起点。>
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    // <学习注释：验证上游数据契约 —— filtered 信号长度必须等于 config.samples。>
    require(state.filtered.size() == static_cast<std::size_t>(config.samples),
        "step3 did not receive step2 output");
    // <学习注释：计算裁剪范围 —— 去除信号两端各 crop_each 个样本，消除 FIR 滤波边界效应。
    //  crop_begin = step3_crop_each：裁剪起始索引。
    //  crop_end = samples − step3_crop_each：裁剪结束索引（半开区间 [begin, end)）。>
    const int crop_begin = config.step3_crop_each;
    const int crop_end = config.samples - config.step3_crop_each;
    // <学习注释：裁剪 filtered 信号，从 filtered[begin] 到 filtered[end−1]。
    //  std::vector 的迭代器区间构造：new_vector(begin_iter, end_iter)。>
    std::vector<float> cropped(state.filtered.begin() + crop_begin,
                               state.filtered.begin() + crop_end);
    // <学习注释：定义三次 B 样条偏移量数组 —— 9 个均匀偏移量，步长 0.5。
    //  τ ∈ {−2.0, −1.5, −1.0, −0.5, 0.0, 0.5, 1.0, 1.5, 2.0}
    //  覆盖三次 B 样条基函数的非零支撑区间 [−2, 2]，步长 0.5 在精度和计算量间平衡。>
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段2（H2D）—— 将裁剪信号和偏移量从 Host 拷贝到 GPU 显存。>
    begin = Clock::now();
    auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);
    auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);
    evidence.h2d_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段3（GPU 权重计算）—— 在 GPU 上计算三次 B 样条核权重。
    //  分配 d_spline（9 个 float 元素），调用 cubic_device 对每个偏移量 τ_k
    //  计算 cubic(τ_k)，结果写入 d_spline。
    //  数学公式（对每个 τ_k）：
    //    cubic(τ_k) = { (2−|τ_k|)³/6 − 2(1−|τ_k|)³/3 }  |τ_k| < 1
    //                = { (2−|τ_k|)³/6 }                 1 ≤ |τ_k| < 2
    //                = 0                                |τ_k| ≥ 2>
    cusignal::DeviceArray<float> d_spline(offsets.size());
    evidence.operator_ms["cubic"] = time_gpu([&] {
        cusignal::cubic_device(d_offsets, d_spline);
    });

    // <学习注释：阶段4（D2H + CPU 归一化）—— 将权重回传 Host 做归一化。
    //  步骤(a)：D2H 将 GPU 权重拷贝回 Host。>
    begin = Clock::now();
    auto weights = d_spline.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    // <学习注释：步骤(b)：验证权重和 > 0，防止退化（所有权重为 0 导致除零）。
    //  std::accumulate 从 0.0F 开始累加所有权重。>
    const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
    require(weight_sum > 0.0F, "step3 degenerate cubic window");
    // <学习注释：步骤(c)：归一化权重 —— 每个权重除以总和，使 Σ w[k] = 1。
    //  范围 for 循环：for (float& value : weights) 遍历每个元素并原地修改。
    //  value /= weight_sum 将每个权重除以总和，使归一化后权重和为 1。>
    for (float& value : weights) value /= weight_sum;
    // <学习注释：步骤(d)：H2D 将归一化权重上传回 GPU。>
    begin = Clock::now();
    auto d_weights = cusignal::DeviceArray<float>::from_host(weights);
    evidence.h2d_ms += milliseconds(begin, Clock::now());

    // <学习注释：阶段5（GPU 平滑滤波）—— 用归一化三次 B 样条权重作为 FIR 系数，
    //  对裁剪信号执行卷积平滑。
    //  配置 FIR 滤波选项：shape = {N_crop}（一维输出）、axis = −1（沿最后一维）。
    //  FirfilterDeviceResult 是 GPU 版本 firfilter 的返回结构体，包含 .y 成员。>
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(cropped.size())};
    options.axis = -1;
    cusignal::FirfilterDeviceResult smoothed;
    evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] {
        cusignal::firfilter_device(d_weights, d_cropped, smoothed, options);
    });
    // <学习注释：汇总 compute_ms = cubic 耗时 + firfilter 耗时。>
    evidence.compute_ms = evidence.operator_ms["cubic"] +
        evidence.operator_ms["firfilter_b_spline"];

    // <学习注释：阶段6（D2H + CPU 参考）—— 将平滑结果回传 Host。
    //  smoothed.y 是 firfilter_device 的输出信号成员。>
    begin = Clock::now();
    state.smoothed = smoothed.y.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    // <学习注释：准备互相关参考 —— 裁剪 target_waveform（原始 chirp），
    //  与 smoothed 做相同范围的裁剪，供 Step4 互相关峰值定位使用。
    //  reference_for_correlation 是原始 chirp 的裁剪版本，用于与 smoothed 做互相关。>
    begin = Clock::now();
    state.reference_for_correlation.assign(
        state.target_waveform.begin() + crop_begin,
        state.target_waveform.begin() + crop_end);
    evidence.post_ms = milliseconds(begin, Clock::now());

    // <学习注释：记录总耗时（从 total_begin 到此刻）。>
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    // <学习注释：可视化捕获：如果启用，保存归一化后的样条权重供后续绘图分析。>
    if (visualization_capture_enabled())
        state.spline_weights = weights;
    return evidence;
}

}  // namespace task2
```

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#include "accuracy/validation/step3_validation.h"

#include "bsplines/bsplines_typed.h"
#include "filtering/filtering_typed.h"

#include <numeric>

namespace task2::accuracy {

// <学习注释：CPU 参考实现 —— 收集 Step3 验证数据（collect_step3_validation）。
//  这是 run_step3 的 CPU 等价实现，用于生成 GPU 精度验证的基准数据。
//
//  【函数总体处理方法】
//  步骤1：裁剪 filtered 信号（与 GPU 版本相同的裁剪范围）
//  步骤2：调用 cubic_cpu 在 CPU 上计算三次 B 样条核权重
//  步骤3：归一化权重（除以总和）
//  步骤4：配置 FirfilterOptions，调用 firfilter_typed_cpu 执行 FIR 平滑滤波
//
//  【与 GPU 版本的差异】
//  - 使用 CPU 版本的算子函数（cubic_cpu、firfilter_typed_cpu），而非 GPU 版本
//  - 无 H2D/D2H 传输，直接操作 Host 端 std::vector<float>
//  - 串行执行，不使用 CUDA kernel
//  - 输出 smoothed_cpu 用于与 GPU 版本的 smoothed 做精度对比>
Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data;
    // <学习注释：裁剪 filtered 信号 —— 与 GPU 版本相同的裁剪范围。
    //  assign(begin_iter, end_iter)：将迭代器区间内的元素复制到 data.cropped_input。>
    data.cropped_input.assign(
        state.filtered.begin() + state.config.step3_crop_each,
        state.filtered.end() - state.config.step3_crop_each);
    // <学习注释：定义偏移量数组 —— 与 GPU 版本完全相同的 9 个浮点偏移量。
    //  τ ∈ {−2.0, −1.5, ..., 2.0}，步长 0.5。>
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    // <学习注释：调用 CPU 版本的 cubic 算子，计算三次 B 样条核权重。
    //  数学公式：对每个 τ_k，cubic(τ_k) = 分段三次多项式（见 run_step3 注释）。>
    auto weights_cpu = cusignal::cubic_cpu(offsets);
    // <学习注释：归一化权重 —— 除以总和使 Σ w[k] = 1。
    //  std::accumulate 从 0.0F 开始累加。范围 for 循环原地归一化。>
    const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);
    for (float& value : weights_cpu) value /= sum;
    // <学习注释：配置 FIR 滤波选项 —— 与 GPU 版本相同的参数。>
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(data.cropped_input.size())};
    options.axis = -1;
    // <学习注释：调用 CPU 版本的 firfilter 执行 B 样条平滑卷积。
    //  数学公式：smoothed[n] = Σ_{k=0}^{8} w[k] × cropped[n − k]。
    //  .y 访问输出信号成员。>
    data.smoothed_cpu = cusignal::firfilter_typed_cpu(
        weights_cpu, data.cropped_input, options).y;
    return data;
}

}  // namespace task2::accuracy
```

### 3.4 函数总体处理方法

本节从方案设计的角度，概述 Step3 中所有函数的总体处理方法和它们之间的协作关系。

#### 3.4.1 总体流水线架构

Step3 的 B 样条平滑流水线由以下函数协作完成：

```
                    ┌─────────────────────────────────────────────────┐
                    │     collect_step3_validation (CPU 参考实现)      │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  cropped_input = filtered[crop:]          │   │
                    │  │  cubic_cpu(offsets) → weights_cpu         │   │
                    │  │  normalize: w /= Σw                       │   │
                    │  │  firfilter_typed_cpu(w, cropped) → smooth │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                         ↑ 精度对比验证
                    ┌─────────────────────────────────────────────────┐
                    │              run_step3 (GPU 实现)                │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  cropped = filtered[crop:]  (CPU crop)   │   │
                    │  │  cubic_device(offsets) → weights (GPU)   │   │
                    │  │  D2H → normalize → H2D (CPU norm)        │   │
                    │  │  firfilter_device(w, cropped) → smoothed │   │
                    │  │  reference = target_waveform[crop:]      │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### 3.4.2 各函数处理方法一览

| 函数 | 类型 | 职责 | 核心数学公式 | 并行策略 |
|------|------|------|-------------|----------|
| `run_step3` | GPU 入口 | 编排平滑流水线：裁剪 → 权重 → 归一化 → 卷积 → 参考 | $smoothed[n] = \sum w[k] \cdot cropped[n-k]$ | 分阶段串行，GPU 阶段内并行 |
| `collect_step3_validation` | CPU 参考入口 | 编排 CPU 流水线，生成基准平滑结果 | 与 `run_step3` 等价 | 串行 |
| `cubic_device` / `cubic_cpu` | GPU/CPU 算子 | 计算三次 B 样条基函数在偏移量上的值 | $cubic(\tau)$ = 分段三次多项式 | 每偏移量一个线程 |
| `firfilter_device` / `firfilter_typed_cpu` | GPU/CPU 算子 | 用归一化权重对裁剪信号做 FIR 卷积平滑 | $y[n] = \sum_{k=0}^{8} w[k] \cdot x[n-k]$ | 每输出采样点一个线程 |

#### 3.4.3 算子数学原理在当前任务中的实例化

##### (a) 三次 B 样条基函数（`cubic`）

- 学习入口：[数学物理原理](../../operators/bsplines/cubic/数学物理原理.md)、[Python 源码算法](../../operators/bsplines/cubic/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/bsplines/cubic/cusignal_cpp_cubic复现逻辑.md)
- 当前调用采用的数学关系（三次 B 样条基函数，分段多项式）：

  $$cubic(\tau) = \begin{cases} \frac{(2-|\tau|)^3}{6} - \frac{2(1-|\tau|)^3}{3}, & 0 \le |\tau| < 1 \\[6pt] \frac{(2-|\tau|)^3}{6}, & 1 \le |\tau| < 2 \\[6pt] 0, & |\tau| \ge 2 \end{cases}$$

- 任务变量到数学符号的映射：`offsets` → $\tau_k$（9 个偏移量），`d_spline` / `weights_cpu` → $cubic(\tau_k)$（基函数值）
- 参数实例化：$\tau \in \{-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0\}$，步长 0.5
- 物理意义：三次 B 样条是 $C^2$ 连续的分段多项式，具有紧支撑 $[-2, 2]$ 和单位积分性质，是平滑滤波的理想核函数

##### (b) FIR 卷积平滑（`firfilter`）

- 学习入口：[数学物理原理](../../operators/filtering/firfilter/数学物理原理.md)、[Python 源码算法](../../operators/filtering/firfilter/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/filtering/firfilter/cusignal_cpp_firfilter复现逻辑.md)
- 当前调用采用的数学关系：$y[n] = \sum_{k=0}^{8} w[k] \times x[n-k]$
- 任务变量到数学符号的映射：`cropped` → $x[n]$（裁剪信号），`weights` → $w[k]$（归一化权重），`smoothed` → $y[n]$（平滑输出）
- 参数实例化：`shape={N_crop}`（一维输出）、`axis=-1`（沿最后一维滤波）
- 物理意义：归一化后的三次 B 样条权重构成一个低通 FIR 滤波器，卷积后平滑信号中的高频残余噪声

#### 3.4.4 B 样条平滑的物理含义

Step3 的核心数学模型是三次 B 样条核与信号的卷积平滑：

$$smoothed[n] = \sum_{k=0}^{8} w[k] \times cropped[n-k]$$

其中 $w[k] = cubic(\tau_k) / \sum_{j=0}^{8} cubic(\tau_j)$，$\tau_k \in \{-2.0, -1.5, ..., 2.0\}$。

| 步骤 | 操作 | 物理意义 |
|------|------|---------|
| 信号裁剪 | 去除 filtered 两端各 crop_each 个样本 | 消除 Step2 FIR 滤波的边界效应（群延迟导致的边缘失真） |
| 权重计算 | 在 9 个偏移量上求值 cubic(τ) | 三次 B 样条基函数离散采样，覆盖非零支撑区间 [−2, 2] |
| 权重归一化 | 每个权重除以总和 | 使 Σ w[k] = 1，保证平滑后信号幅度不变（直流增益为 1） |
| FIR 卷积 | 用归一化权重对裁剪信号卷积 | 加权移动平均，保留信号趋势的同时平滑残余噪声 |
| 参考准备 | 裁剪 target_waveform 相同范围 | 为 Step4 互相关峰值定位提供无噪声参考信号 |

**B 样条核与普通移动平均的区别**：三次 B 样条核（$C^2$ 连续）比简单矩形窗或三角窗具有更平滑的频率响应，能更好地保留信号的峰值特征。

#### 3.4.5 CPU/GPU 精度验证策略

| 验证维度 | CPU 实现 | GPU 实现 | 验证方法 |
|---------|---------|---------|---------|
| 信号裁剪 | `assign(begin, end)` | `vector(begin, end)` 构造 | 逐元素比较裁剪后的 cropped_input |
| 权重计算 | `cubic_cpu(offsets)` | `cubic_device(offsets, d_spline)` | 逐元素比较权重值 |
| 权重归一化 | `value /= sum` | `value /= weight_sum` | 逐元素比较归一化权重 |
| 平滑滤波 | `firfilter_typed_cpu` | `firfilter_device` | 逐元素比较 smoothed 输出 |

---

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按"语法结构—名称与类型—执行过程—任务语义—初学者易错点"讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

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

- `#pragma once`：要求编译器在同一翻译单元中只展开一次本头文件
- `#include "../task2_common.h"`：在预处理阶段将 `task2_common.h` 的声明展开到当前文件

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只建立头文件依赖和声明归属，不执行当前 step 的数值算法|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法|

**任务语义**

本块属于公共基础设施：建立头文件依赖，使后续声明能使用 `PipelineState`、`StepEvidence` 等共享类型。

**设计理由与替代方案**

`#pragma once` 是现代编译器广泛支持的 include guard 替代方案，比传统的 `#ifndef`/`#define`/`#endif` 更简洁且不会出现宏名冲突。

**初学者易错点**

- `#include` 是预处理指令，不是运行时函数调用；它在编译前展开，不产生可执行代码

### 4.2 Step3 入口函数声明

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`；符号 `run_step3`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step3(PipelineState& state);
```

**语法结构**

命名空间声明后跟函数声明。

**名称与类型**

- `namespace task2`：命名空间，将内部声明放入 `task2` 名称范围
- `StepEvidence`：返回类型，包含 step 名称、各阶段耗时和 metrics 指标
- `run_step3`：函数名，Step3 的入口函数
- `PipelineState& state`：参数类型为 `PipelineState` 的引用（`&`），非 const 引用表示函数可以修改 state 的内容

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step3`|Step3 流水线入口函数|无独立数学符号|接收 Step2 的 filtered，输出 smoothed 给 Step4|
|`state`|任务流水线共享状态|无单一数学符号|保存各 step 的数组与证据；Step2 写入 filtered，Step3 读取并写入 smoothed|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `namespace task2 {`：打开命名空间 `task2`
- `StepEvidence run_step3(PipelineState& state);`：函数声明，末尾分号结束声明

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法入口|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块声明 `run_step3` 函数签名，定义其接口契约|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明接口存在，但声明本身不执行算法|

**任务语义**

本块声明 Step3 的公开接口：接收流水线状态引用，返回步骤证据对象。`PipelineState&` 为非 const 引用，表明函数会修改 state 中的 `smoothed` 和 `reference_for_correlation` 字段。

**设计理由与替代方案**

引用传递（`&`）而非值传递，避免了拷贝整个 `PipelineState` 对象的开销。

**初学者易错点**

- 声明符中的 `&` 表示引用（现有对象的别名），不是取地址或按位与

### 4.3 命名空间闭合

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只关闭词法作用域，不产生新的业务数组或数值结果|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只管理作用域，不能冒充核心算法|

**任务语义**

本块只关闭命名空间作用域，没有独立的任务语义。

**设计理由与替代方案**

`// namespace task2` 注释是代码风格约定，帮助快速识别闭合的命名空间。

**初学者易错点**

- 花括号本身不产生运行时操作；它只是编译期作用域边界

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `文件级代码`；源码锚点 `#include "step3.h"`。

```cpp
#include "step3.h"

#include "bsplines/bsplines_typed.h"
#include "cuda_utils/device_array.h"
#include "filtering/filtering_typed.h"

#include <numeric>
```

**语法结构**

这个代码块由 5 条 `#include` 预处理指令构成。

**名称与类型**

- `"step3.h"`：包含 Step3 的头文件，引入 `run_step3` 声明和 `PipelineState`、`StepEvidence` 等类型
- `"bsplines/bsplines_typed.h"`：包含 `cubic_device` 等 B 样条基函数计算函数的声明
- `"cuda_utils/device_array.h"`：包含 `DeviceArray` 模板类声明，提供 GPU 显存管理和 H2D/D2H 传输
- `"filtering/filtering_typed.h"`：包含 `firfilter_device` 等 FIR 滤波函数的声明
- `<numeric>`：C++ 标准库，提供 `std::accumulate` 用于求和和归一化

**变量—公式—任务状态映射**

当前块只包含预处理指令，没有新出现的任务运行时变量。

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- 5 条 `#include` 指令在预处理阶段按顺序展开，将各自头文件的内容插入当前翻译单元
- `"..."` 表示项目内头文件，`<...>` 表示系统/标准库头文件

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只建立头文件依赖和声明归属，不执行当前 step 的数值算法|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法|

**任务语义**

本块属于公共基础设施：建立两个算子模块（bsplines、filtering）和 `DeviceArray`、标准库的依赖关系。与 Step2 不同，Step3 不需要 windows 和 filter_design 模块。

**设计理由与替代方案**

每个 `#include` 引入一个明确的依赖模块，使得编译依赖可审计。

**初学者易错点**

- `#include` 是预处理指令，不是运行时函数调用；`<numeric>` 的尖括号表示系统头文件搜索路径

### 4.5 命名空间和函数定义入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step3(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step3";
    const auto total_begin = Clock::now();
```

**语法结构**

命名空间、函数定义（含函数体开头）、多个局部变量声明与初始化。

**名称与类型**

- `namespace task2`：打开命名空间
- `run_step3(PipelineState& state)`：函数定义，`PipelineState&` 是非 const 引用参数
- `const auto& config`：`const auto&` 推导为 `const TaskConfig&`，只读引用避免拷贝
- `StepEvidence evidence`：默认构造的证据记录对象
- `evidence.name = "step3"`：字符串字面量赋值给 `std::string` 成员
- `const auto total_begin`：`auto` 推导为 `Clock::time_point`，记录总耗时起点

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置只读引用|无独立数学符号；包含 `step3_crop_each`、`samples`|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存各阶段耗时和 metrics，最终序列化|
|`total_begin`|总耗时计时起点|时间戳记作 $t_0$|与后续 `Clock::now()` 相减得 total_ms|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `namespace task2 {`：打开命名空间
- `PipelineState& state`：参数为引用，不复制对象
- `const auto& config = state.config`：声明只读引用 `config` 绑定到 `state.config`
- `StepEvidence evidence`：默认构造 `evidence` 对象
- `evidence.name = "step3"`：将字符串字面量 `"step3"` 赋值给 `evidence.name`
- `const auto total_begin = Clock::now()`：调用 `Clock::now()` 获取当前时间点

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法入口|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块定义 `run_step3` 函数体，初始化 config 引用和 evidence 对象|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块建立函数入口和计时起点，是后续算法执行的前置条件|

**任务语义**

本块是 Step3 函数体的入口：建立只读配置引用、创建证据记录对象、启动总计时器。

**设计理由与替代方案**

`const auto&` 避免了拷贝 `TaskConfig` 对象。`Clock::now()` 在函数体开头调用，使 `total_ms` 能覆盖完整的执行时间。

**初学者易错点**

- `const auto&` 中的 `&` 是引用声明，不是取地址；`const` 保证不能通过该引用修改原对象

### 4.6 契约验证、信号裁剪和偏移量定义

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    require(state.filtered.size() == static_cast<std::size_t>(config.samples),
        "step3 did not receive step2 output");
    const int crop_begin = config.step3_crop_each;
    const int crop_end = config.samples - config.step3_crop_each;
    std::vector<float> cropped(state.filtered.begin() + crop_begin,
                               state.filtered.begin() + crop_end);
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

变量声明与初始化、函数调用、赋值表达式。

**名称与类型**

- `auto begin`：`auto` 推导为 `Clock::time_point`，记录 prep 阶段计时起点
- `require(...)`：契约检查函数，第一个参数为 bool 条件
- `const int crop_begin`：`const int`，裁剪起始索引
- `const int crop_end`：`const int`，裁剪结束索引（半开区间）
- `std::vector<float> cropped(...)`：用迭代器区间构造裁剪后的信号
- `std::vector<float> offsets{...}`：用列表初始化构造 9 个偏移量

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|prep 阶段计时起点|时间戳 $t_p$|与 `Clock::now()` 相减得 prep_ms|
|`filtered`|Step2 输出的滤波信号|记作 $filtered[n]$|Step2 写入，Step3 读取并裁剪|
|`crop_begin`|裁剪起始索引|$i_{begin} = crop\_each$|从 config 读取|
|`crop_end`|裁剪结束索引|$i_{end} = N - crop\_each$|计算得出|
|`cropped`|裁剪后的信号|$x_{crop}[n] = filtered[n + crop\_each]$|传给 cubic 和 firfilter|
|`offsets`|三次 B 样条偏移量|$\tau_k \in \{-2.0, -1.5, ..., 2.0\}$|传给 cubic_device 计算核权重|

**数字常量逐项说明**

- `-2.0F`、`-1.5F`、`-1.0F`、`-0.5F`、`0.0F`、`0.5F`、`1.0F`、`1.5F`、`2.0F`：9 个 `float` 类型偏移量，步长 0.5。覆盖三次 B 样条基函数的非零支撑区间 $[-2, 2]$。步长 0.5 在精度和计算量间平衡；步长更小会提高精度但增加计算量。源码未记录选值推导依据。

**执行过程**

- `auto begin = Clock::now()`：记录 prep 阶段开始时间
- `require(...)`：运行时检查 `filtered.size() == config.samples`
- `const int crop_begin = config.step3_crop_each`：从配置读取裁剪起始索引
- `const int crop_end = config.samples - config.step3_crop_each`：计算裁剪结束索引
- `std::vector<float> cropped(...)`：用迭代器区间构造裁剪信号，不复制两端各 crop_each 个样本
- `std::vector<float> offsets{...}`：列表初始化 9 个偏移量
- `evidence.prep_ms = milliseconds(begin, Clock::now())`：计算 prep 阶段耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块验证 filtered 契约，裁剪信号两端，定义 B 样条偏移量|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入|

**任务语义**

本块完成 Step3 的 CPU 准备阶段：验证上游数据契约、裁剪信号两端以消除 FIR 滤波边界效应、定义三次 B 样条偏移量数组。裁剪范围由 `config.step3_crop_each` 控制。

**设计理由与替代方案**

信号两端裁剪是为了消除 Step2 FIR 滤波的边界效应（群延迟导致前 L−1 个输出样本含瞬态失真）。`offsets` 步长 0.5 覆盖了三次 B 样条支撑区间 $[-2, 2]$ 内的 9 个采样点，在精度和计算量间取得平衡。

**初学者易错点**

- `state.filtered.begin() + crop_begin` 是迭代器加法，不是指针算术；`begin()` 返回 `vector<float>::iterator`
- 列表初始化 `{...}` 中的 `F` 后缀表示 `float` 字面量（不加 `F` 默认为 `double`）

### 4.7 H2D 数据传输

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);
    auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
```

**语法结构**

赋值、变量声明与初始化、函数调用。

**名称与类型**

- `auto d_cropped`：`auto` 推导为 `cusignal::DeviceArray<float>`，封装 GPU 显存中的裁剪信号
- `auto d_offsets`：`auto` 推导为 `cusignal::DeviceArray<float>`，封装 GPU 显存中的偏移量数组
- `evidence.h2d_ms = ...`：覆盖赋值，记录 H2D 阶段耗时

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_cropped`|GPU 端裁剪信号|与 Host 端 `cropped` 数学含义相同|由 H2D 产生，传给 firfilter_device|
|`d_offsets`|GPU 端偏移量|与 Host 端 `offsets` 数学含义相同|由 H2D 产生，传给 cubic_device|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `begin = Clock::now()`：记录 H2D 阶段开始时间
- `DeviceArray<float>::from_host(cropped)`：在 GPU 上分配显存并将裁剪信号拷贝到 device
- `DeviceArray<float>::from_host(offsets)`：在 GPU 上分配显存并将偏移量拷贝到 device
- `evidence.h2d_ms = milliseconds(begin, Clock::now())`：计算 H2D 耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块将裁剪信号和偏移量从 Host 拷贝到 GPU 显存|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出|

**任务语义**

本块完成 H2D 数据传输：将裁剪信号（N_crop 个 float）和偏移量（9 个 float）从 Host 内存拷贝到 GPU 显存，为后续 GPU 算子链准备输入数据。

**设计理由与替代方案**

`evidence.h2d_ms =` 而非 `+=`，因为这是 Step3 的第一个 H2D 阶段。后续还有第二次 H2D（权重上传），使用 `+=` 累加。

**初学者易错点**

- `DeviceArray<float>::from_host` 是静态成员函数，`::` 是作用域解析运算符

### 4.8 三次 B 样条权重计算

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `cusignal::DeviceArray<float> d_spline(offsets.size());`。

```cpp
    cusignal::DeviceArray<float> d_spline(offsets.size());
    evidence.operator_ms["cubic"] = time_gpu([&] {
        cusignal::cubic_device(d_offsets, d_spline);
    });
```

**语法结构**

变量声明与初始化、lambda 表达式、函数调用。

**名称与类型**

- `cusignal::DeviceArray<float> d_spline(offsets.size())`：在 GPU 上分配 9 个 float 的显存
- `time_gpu([&] { ... })`：执行 lambda 并计量 GPU 耗时，返回毫秒值
- `cusignal::cubic_device(d_offsets, d_spline)`：GPU 版三次 B 样条基函数求值

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_spline`|GPU 端三次 B 样条权重|$cubic(\tau_k)$ = 分段三次多项式|由 cubic_device 填充，D2H 后归一化|
|`d_offsets`|GPU 端偏移量|$\tau_k$|传给 cubic_device 作为求值点|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `DeviceArray<float> d_spline(offsets.size())`：在 GPU 上分配 9 个 float 的显存
- `time_gpu([&] { ... })`：计量 GPU 耗时
- `cubic_device(d_offsets, d_spline)`：在 GPU 上对每个 $\tau_k$ 计算 $cubic(\tau_k)$，结果写入 `d_spline`
- `evidence.operator_ms["cubic"] = ...`：记录耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法（B 样条权重计算）|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块在 GPU 上计算三次 B 样条基函数在 9 个偏移量上的值|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明 B 样条权重已计算，但未归一化的权重不等于完整平滑|

**任务语义**

本块在 GPU 上计算三次 B 样条基函数 $cubic(\tau)$ 在 9 个偏移量 $\tau_k \in \{-2.0, -1.5, ..., 2.0\}$ 上的值。$cubic(\tau)$ 是 $C^2$ 连续的分段三次多项式，紧支撑于 $[-2, 2]$。

**设计理由与替代方案**

使用 `cubic_device` 而非 CPU 版本，因为后续 firfilter 也在 GPU 上执行，权重在 GPU 上计算可减少一次 H2D/D2H 往返。但当前实现仍将权重回传 CPU 做归一化，因为归一化需要计算总和（归约操作），在 GPU 上做归约需要额外的 kernel launch。

**初学者易错点**

- `offsets.size()` 返回 `std::size_t`，作为 `DeviceArray` 构造函数参数表示元素个数（9）
- `[&]` 是 lambda 捕获列表，`&` 表示按引用捕获

### 4.9 权重归一化（D2H + CPU 计算 + H2D）

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    auto weights = d_spline.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
    require(weight_sum > 0.0F, "step3 degenerate cubic window");
    for (float& value : weights) value /= weight_sum;
    begin = Clock::now();
    auto d_weights = cusignal::DeviceArray<float>::from_host(weights);
    evidence.h2d_ms += milliseconds(begin, Clock::now());
```

**语法结构**

赋值、变量声明与初始化、函数调用、范围 for 循环。

**名称与类型**

- `auto weights`：`auto` 推导为 `std::vector<float>`，D2H 后的权重
- `d_spline.to_host()`：将 GPU 显存内容拷贝回 Host
- `const float weight_sum`：`const float`，权重总和
- `std::accumulate(..., 0.0F)`：从 0.0F 开始累加，`0.0F` 是 float 字面量
- `require(weight_sum > 0.0F, ...)`：契约检查，防止归零导致的除零
- `for (float& value : weights)`：范围 for 循环，`&` 表示引用，允许原地修改
- `value /= weight_sum`：复合赋值除法，原地归一化

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`weights`|Host 端三次 B 样条权重|$cubic(\tau_k)$ → 归一化后 $w[k] = cubic(\tau_k) / \sum cubic(\tau_j)$|由 D2H 产生，归一化后 H2D 上传|
|`weight_sum`|权重总和|$\sum_{k=0}^{8} cubic(\tau_k)$|用于验证 > 0 和归一化分母|
|`d_weights`|GPU 端归一化权重|$w[k]$|H2D 上传，传给 firfilter_device|

**数字常量逐项说明**

- `0.0F`：`float` 类型加法单位元，作为 `std::accumulate` 的初始累加值。`F` 后缀确保是 `float` 而非 `double`，与 `weights` 的元素类型一致

**执行过程**

- `d_spline.to_host()`：将 GPU 权重拷贝回 Host（D2H）
- `evidence.d2h_ms += ...`：累加 D2H 耗时（`+=`）
- `std::accumulate(weights.begin(), weights.end(), 0.0F)`：从 0.0F 开始累加所有权重，得 `weight_sum`
- `require(weight_sum > 0.0F, ...)`：验证权重和不退化
- `for (float& value : weights) value /= weight_sum`：每个权重除以总和，归一化后 $\sum w[k] = 1$
- `DeviceArray<float>::from_host(weights)`：将归一化权重上传回 GPU（H2D）
- `evidence.h2d_ms += ...`：累加 H2D 耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法（权重归一化）|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块将权重回传 CPU，归一化使权重和为 1，再上传回 GPU|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明归一化已完成，权重和为 1 且不退化|

**任务语义**

本块完成权重归一化：D2H 回传 → 验证 sum > 0 → 归一化 w /= sum → H2D 上传回 GPU。归一化确保 $\sum w[k] = 1$，使平滑滤波的直流增益为 1（平滑后信号幅度不变）。`require(weight_sum > 0.0F)` 防止所有权重为零导致的除零错误。

**设计理由与替代方案**

在 CPU 上做归一化而非 GPU，因为归约求和（`std::accumulate`）在 CPU 上更简单直接。在 GPU 上做归约需要额外的 reduction kernel launch。`require` 在权重退化时立即终止并报告错误，避免静默产生全零输出。

**初学者易错点**

- `evidence.d2h_ms +=` 是复合赋值（累加），`evidence.h2d_ms +=` 同理；Step3 有两次 H2D 和两次 D2H
- `for (float& value : weights)` 中的 `&` 是引用声明，允许原地修改元素
- `value /= weight_sum` 等价于 `value = value / weight_sum`

### 4.10 FIR 卷积平滑滤波

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `cusignal::FirfilterOptions options;`。

```cpp
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(cropped.size())};
    options.axis = -1;
    cusignal::FirfilterDeviceResult smoothed;
    evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] {
        cusignal::firfilter_device(d_weights, d_cropped, smoothed, options);
    });
    evidence.compute_ms = evidence.operator_ms["cubic"] +
        evidence.operator_ms["firfilter_b_spline"];
```

**语法结构**

对象声明与字段赋值、lambda 表达式、函数调用、算术表达式。

**名称与类型**

- `cusignal::FirfilterOptions options`：FIR 滤波选项结构体
- `options.shape = {static_cast<int>(cropped.size())}`：列表初始化形状为 `{N_crop}`
- `options.axis = -1`：沿最后一维滤波
- `cusignal::FirfilterDeviceResult smoothed`：GPU 版 FIR 滤波结果结构体
- `firfilter_device(d_weights, d_cropped, smoothed, options)`：GPU 版 FIR 滤波
- `evidence.compute_ms = ... + ...`：两个算子耗时之和

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`options`|FIR 滤波选项|无独立数学符号|配置输出形状和滤波轴|
|`smoothed`|GPU 端平滑结果|$y[n] = \sum_{k=0}^{8} w[k] \times cropped[n-k]$|由 firfilter_device 填充，D2H 后传给 Step4|
|`d_weights`|GPU 端归一化权重|$w[k]$|firfilter 的 FIR 系数|
|`d_cropped`|GPU 端裁剪信号|$x[n]$|firfilter 的输入信号|

**数字常量逐项说明**

- `-1`：`int` 类型，表示最后一维（Python/NumPy 风格），一维数组中等价于 axis=0

**执行过程**

- `FirfilterOptions options`：默认构造滤波选项
- `options.shape = {static_cast<int>(cropped.size())}`：设置输出形状为 `{N_crop}`
- `options.axis = -1`：设置滤波轴为最后一维
- `FirfilterDeviceResult smoothed`：声明 GPU 滤波结果结构体
- `time_gpu([&] { ... })`：计量 GPU 耗时
- `firfilter_device(d_weights, d_cropped, smoothed, options)`：在 GPU 上执行 FIR 卷积平滑
- `evidence.operator_ms["firfilter_b_spline"] = ...`：记录耗时
- `evidence.compute_ms = cubic_ms + firfilter_ms`：汇总计算耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法（平滑执行）|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块在 GPU 上用归一化三次 B 样条权重对裁剪信号执行 FIR 卷积平滑|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明 B 样条平滑算法已执行，smoothed 是 Step3 的核心业务结果|

**任务语义**

本块是 Step3 的核心计算步骤：用归一化三次 B 样条权重作为 FIR 系数，对裁剪信号执行卷积平滑。卷积 $y[n] = \sum w[k] \times x[n-k]$ 等价于加权移动平均，B 样条核（$C^2$ 连续）比简单矩形窗或三角窗具有更平滑的频率响应。

**设计理由与替代方案**

使用 `firfilter_device` 而非专门的 B 样条平滑 kernel，因为 B 样条平滑本质上就是 FIR 滤波（权重为 FIR 系数）。`compute_ms` 手动求和而非使用 `std::accumulate`，因为只有两个算子，直接相加更简洁。

**初学者易错点**

- `FirfilterDeviceResult` 是 GPU 版本的结果结构体，与 CPU 版本的返回类型不同
- `smoothed` 是结构体变量（不是 `DeviceArray`），其 `.y` 成员是 `DeviceArray<float>`

### 4.11 平滑结果 D2H 和参考准备

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    state.smoothed = smoothed.y.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    begin = Clock::now();
    state.reference_for_correlation.assign(
        state.target_waveform.begin() + crop_begin,
        state.target_waveform.begin() + crop_end);
    evidence.post_ms = milliseconds(begin, Clock::now());
```

**语法结构**

赋值、成员访问、函数调用。

**名称与类型**

- `smoothed.y.to_host()`：访问 `smoothed` 结构体的 `.y` 成员，调用 `to_host()` 将 GPU 结果拷贝回 Host
- `state.smoothed = ...`：将平滑结果写入流水线状态
- `state.reference_for_correlation.assign(...)`：用迭代器区间赋值，准备互相关参考
- `state.target_waveform`：Step1 生成的原始 chirp 波形（无延迟、无干扰、无噪声）
- `evidence.post_ms = ...`：记录后处理耗时

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state.smoothed`|平滑后信号|$y[n]$（B 样条平滑输出）|Step3 写入，Step4 读取做特征提取|
|`state.reference_for_correlation`|互相关参考信号|裁剪后的原始 chirp 波形|Step3 写入，Step4 读取做互相关峰值定位|
|`state.target_waveform`|原始 chirp 波形|$chirp[n]$（无延迟/干扰/噪声）|Step1 生成，Step3 裁剪后用做参考|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `smoothed.y.to_host()`：将 GPU 平滑结果拷贝回 Host（D2H）
- `state.smoothed = ...`：写入流水线状态
- `evidence.d2h_ms += ...`：累加 D2H 耗时
- `state.reference_for_correlation.assign(...)`：将 target_waveform 的裁剪区间复制到参考信号
- `evidence.post_ms = milliseconds(...)`：记录后处理耗时

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法（输出交付与参考准备）|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块将平滑结果回传 Host，准备与原始 chirp 的互相关参考|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务输出已交付给流水线状态，互相关参考已准备|

**任务语义**

本块完成 Step3 的收尾工作：将平滑结果 D2H 回传 Host，准备互相关参考信号。`reference_for_correlation` 是原始 chirp 波形的裁剪版本（与 smoothed 相同的裁剪范围），供 Step4 做互相关峰值定位，用于估计目标延迟。

**设计理由与替代方案**

`reference_for_correlation` 使用原始 chirp 波形（无延迟/干扰/噪声）作为参考，因为它是发射波形的精确副本。互相关峰值位置对应目标延迟 $d$。`evidence.post_ms` 单独记录后处理耗时，与 D2H 和计算耗时分开。

**初学者易错点**

- `smoothed.y` 是成员访问，`smoothed` 是 `FirfilterDeviceResult` 结构体，`.y` 是其输出信号成员
- `assign(begin, end)` 是 `std::vector` 的成员函数，将迭代器区间内的元素复制到当前向量

### 4.12 总耗时和可视化捕获

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`。

```cpp
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.spline_weights = weights;
    return evidence;
```

**语法结构**

赋值、if 条件语句、return 语句。

**名称与类型**

- `evidence.total_ms = ...`：记录总耗时（从 total_begin 到此刻）
- `visualization_capture_enabled()`：全局函数，返回 bool
- `state.spline_weights = weights`：条件执行，保存归一化后的样条权重
- `return evidence`：返回 evidence 对象（NRVO 优化）

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`total_ms`|总耗时|$T_{total} = t_{now} - t_0$|从函数入口到此刻的 wall clock 时间|
|`spline_weights`|归一化三次 B 样条权重|$w[k] = cubic(\tau_k) / \sum cubic(\tau_j)$|仅在可视化模式下保存，供绘图分析|

**数字常量逐项说明**

当前源码块没有数值字面量。

**执行过程**

- `evidence.total_ms = milliseconds(total_begin, Clock::now())`：计算从函数入口到此刻的总耗时
- `if (visualization_capture_enabled())`：条件判断，仅在可视化模式启用时执行
- `state.spline_weights = weights`：将归一化权重保存到流水线状态
- `return evidence`：返回 evidence 对象

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法（证据交付）|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块记录总耗时，条件保存样条权重，返回完整证据|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明 Step3 流水线已完整执行，证据已交付|

**任务语义**

本块完成 Step3 的收尾工作：记录总耗时、条件保存可视化数据、返回 evidence。`visualization_capture_enabled()` 控制是否将归一化权重保存到 `state.spline_weights`，用于后续绘图分析 B 样条核的形状。

**设计理由与替代方案**

可视化数据捕获是条件执行的，避免在非可视化模式下产生不必要的内存开销。`return evidence` 利用 NRVO 避免拷贝。

**初学者易错点**

- `if` 是条件选择语句，条件为真时只执行其直接子语句
- `return` 是跳转语句，结束当前函数并将 evidence 返回给调用者

### 4.13 命名空间闭合

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只关闭词法作用域，不产生新的业务数组或数值结果|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只管理作用域，不能冒充核心算法|

**任务语义**

本块只关闭命名空间作用域，没有独立的任务语义。

**设计理由与替代方案**

`// namespace task2` 注释是代码风格约定。

**初学者易错点**

- 花括号本身不产生运行时操作；它只是编译期作用域边界

### 4.14 CPU 参考实现：裁剪和权重计算

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `Step3ValidationData collect_step3_validation(const PipelineState& state)`。

```cpp
Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data;
    data.cropped_input.assign(
        state.filtered.begin() + state.config.step3_crop_each,
        state.filtered.end() - state.config.step3_crop_each);
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    auto weights_cpu = cusignal::cubic_cpu(offsets);
    const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);
    for (float& value : weights_cpu) value /= sum;
```

**语法结构**

函数定义、局部变量声明与初始化、函数调用、范围 for 循环。

**名称与类型**

- `collect_step3_validation`：CPU 验证数据收集函数
- `const PipelineState& state`：const 引用参数（只读）
- `Step3ValidationData data`：返回数据结构，包含 `cropped_input`、`smoothed_cpu` 等字段
- `data.cropped_input.assign(...)`：用迭代器区间赋值裁剪信号
- `cusignal::cubic_cpu(offsets)`：CPU 版三次 B 样条基函数求值
- `const float sum`：`const float`，权重总和
- `for (float& value : weights_cpu) value /= sum`：归一化

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step3_validation`|CPU 验证数据收集函数|无独立数学符号|生成 CPU 参考基准，供 GPU/CPU 精度对比|
|`data.cropped_input`|CPU 版裁剪信号|$x_{crop}[n]$|从 filtered 裁剪，传给 firfilter|
|`weights_cpu`|CPU 版三次 B 样条权重|$cubic(\tau_k)$ → 归一化后 $w[k]$|由 cubic_cpu 计算，归一化后传给 firfilter|

**数字常量逐项说明**

- `-2.0F` ~ `2.0F`：9 个 `float` 偏移量，与 GPU 版本完全相同。步长 0.5，覆盖三次 B 样条支撑区间 $[-2, 2]$
- `0.0F`：`float` 类型加法单位元，作为 `std::accumulate` 初始值

**执行过程**

- `data.cropped_input.assign(...)`：将 filtered 的裁剪区间复制到 cropped_input（使用 `end() - crop_each` 而非 `begin() + crop_end`）
- `std::vector<float> offsets{...}`：列表初始化 9 个偏移量
- `cubic_cpu(offsets)`：在 CPU 上计算三次 B 样条基函数值
- `std::accumulate(...)`：求和
- `for (float& value : weights_cpu) value /= sum`：归一化

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块在 CPU 上裁剪信号、计算权重并归一化，作为 GPU 精度验证的基准|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径存在；不能替代核心实现|

**任务语义**

本块在 CPU 上完成信号裁剪和权重计算，作为 GPU 精度验证的基准。注意 CPU 版本使用 `end() - crop_each` 而非 `begin() + crop_end`，两者数学等价但写法不同。

**设计理由与替代方案**

CPU 和 GPU 版本使用相同的偏移量数组和数学公式，但 GPU 版本在 device 上计算权重，CPU 版本在 host 上计算。精度对比验证两者的数值一致性。

**初学者易错点**

- `state.filtered.end() - state.config.step3_crop_each` 是迭代器减法，等价于 `begin() + (samples - crop_each)`
- `cubic_cpu` 返回 `std::vector<float>`（CPU 版本），`cubic_device` 写入 `DeviceArray<float>`（GPU 版本）

### 4.15 CPU 参考实现：FIR 平滑滤波和返回

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `cusignal::FirfilterOptions options;`。

```cpp
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(data.cropped_input.size())};
    options.axis = -1;
    data.smoothed_cpu = cusignal::firfilter_typed_cpu(
        weights_cpu, data.cropped_input, options).y;
    return data;
```

**语法结构**

对象声明与字段赋值、函数调用、return 语句。

**名称与类型**

- `cusignal::FirfilterOptions options`：FIR 滤波选项结构体
- `options.shape = {static_cast<int>(data.cropped_input.size())}`：列表初始化形状
- `options.axis = -1`：沿最后一维滤波
- `data.smoothed_cpu`：CPU 版平滑结果
- `cusignal::firfilter_typed_cpu(...).y`：调用 CPU 版 FIR 滤波，`.y` 访问输出信号
- `return data`：返回验证数据集

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`smoothed_cpu`|CPU 版平滑结果|$y[n] = \sum_{k=0}^{8} w[k] \times cropped[n-k]$|与 GPU 版 smoothed 做精度对比|

**数字常量逐项说明**

- `-1`：`int` 类型，表示最后一维（Python/NumPy 风格），一维数组中等价于 axis=0

**执行过程**

- `FirfilterOptions options`：默认构造滤波选项
- `options.shape = {...}`：设置输出形状为 `{N_crop}`
- `options.axis = -1`：设置滤波轴
- `firfilter_typed_cpu(weights_cpu, data.cropped_input, options).y`：在 CPU 上执行 FIR 卷积平滑，`.y` 获取输出信号
- `data.smoothed_cpu = ...`：赋值给验证数据结构
- `return data`：返回完整验证数据集

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块在 CPU 上执行 B 样条平滑卷积，生成 smoothed_cpu 基准|
|业务输出|平滑序列 `smoothed` 及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径存在；不能替代核心实现|

**任务语义**

本块在 CPU 上执行 B 样条平滑卷积，生成 `smoothed_cpu` 作为 GPU 精度验证基准。与 GPU 版本使用相同的归一化权重、裁剪信号和滤波选项。

**设计理由与替代方案**

CPU 版本使用 `firfilter_typed_cpu`（而非 `firfilter_device`），两者内部实现可能不同但数学上等价。精度对比验证两者的数值一致性。

**初学者易错点**

- `.y` 是成员访问，`firfilter_typed_cpu` 返回的结构体包含 `.y`（输出信号）成员
- `return data` 利用 NRVO 优化，不拷贝 `Step3ValidationData` 对象

---

## 5. 自检问题

### 5.1 版本身份

1. Step3 当前学习版本绑定哪个 Git SHA？该提交时间是什么？
2. Step3 涉及哪些源码文件？每个文件的作用是什么？
3. Step3 的 CPU 参考实现和 GPU 实现分别在哪些文件中？

### 5.2 CPU/GPU 职责

4. `run_step3` 的几个处理阶段分别是什么？每个阶段做什么？
5. `collect_step3_validation` 的职责是什么？它生成哪些验证数据？
6. 为什么需要信号裁剪？裁剪范围如何确定？
7. 为什么权重归一化在 CPU 上执行而非 GPU 上？

### 5.3 数学映射

8. 三次 B 样条基函数 $cubic(\tau)$ 的分段定义是什么？
9. Step3 中 B 样条平滑的完整数学流程是什么？
10. 偏移量 $\tau_k$ 取哪些值？为什么取这些值？
11. 权重归一化的数学含义是什么？为什么需要归一化？

### 5.4 模板/类型

12. `FirfilterDeviceResult` 和普通 `DeviceArray` 有什么区别？
13. 为什么 Step3 使用 `firfilter_device` 而非 `firfilter_fp32_resident_device`（如 Step2）？

### 5.5 并行/同步

14. Step3 有几次 H2D 和 D2H 传输？为什么？
15. `time_gpu` 如何确保 GPU 操作完成后才停止计时？

### 5.6 精度/测试边界

16. CPU 和 GPU 版本之间如何做精度验证？
17. `reference_for_correlation` 的作用是什么？为什么需要它？
18. `require(weight_sum > 0.0F)` 防止什么错误？

---

## 6. 参考答案

### 6.1 版本身份

1. **SHA**：`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`，提交时间 `2026-08-18T20:22:35+08:00`。
2. **三个文件**：
   - `step3.h`：头文件，声明 `run_step3` 接口
   - `step3.cu`：实现文件，包含 `run_step3` 的 GPU 流水线编排
   - `step3_validation.cpp`：CPU 参考实现，包含 `collect_step3_validation` 用于精度验证
3. GPU 实现在 `step3.cu` 中，CPU 参考实现在 `step3_validation.cpp` 中。

### 6.2 CPU/GPU 职责

4. **六个阶段**：
   - 阶段1（CPU 准备）：验证 filtered 契约，裁剪信号，定义偏移量
   - 阶段2（H2D）：将裁剪信号和偏移量拷贝到 GPU
   - 阶段3（GPU 权重计算）：cubic_device 计算三次 B 样条基函数值
   - 阶段4（D2H + CPU 归一化 + H2D）：权重回传 → 归一化 → 上传
   - 阶段5（GPU 平滑滤波）：firfilter_device 执行 FIR 卷积平滑
   - 阶段6（D2H + CPU 参考）：平滑结果回传，准备互相关参考
5. `collect_step3_validation` 在 CPU 上生成 B 样条平滑的参考基准，输出 `smoothed_cpu`（平滑结果）和 `cropped_input`（裁剪信号），用于与 GPU 版本做精度对比。
6. 信号裁剪是为了消除 Step2 FIR 滤波的边界效应（群延迟导致前 L−1 个输出样本含瞬态失真）。裁剪范围由 `config.step3_crop_each` 控制，每端各裁剪 `crop_each` 个样本。
7. 权重归一化需要归约求和（`std::accumulate`），在 CPU 上执行比 GPU 上更简单直接（GPU 归约需要额外的 reduction kernel launch）。

### 6.3 数学映射

8. **三次 B 样条基函数**：

   $$cubic(\tau) = \begin{cases} \frac{(2-|\tau|)^3}{6} - \frac{2(1-|\tau|)^3}{3}, & 0 \le |\tau| < 1 \\[6pt] \frac{(2-|\tau|)^3}{6}, & 1 \le |\tau| < 2 \\[6pt] 0, & |\tau| \ge 2 \end{cases}$$

   具有 $C^2$ 连续性、紧支撑 $[-2, 2]$ 和单位积分 $\int cubic(\tau) d\tau = 1$ 的性质。

9. **完整流程**：(1) 裁剪信号 $x_{crop}[n] = filtered[n + crop\_each]$；(2) 在偏移量 $\tau_k$ 上求值 $cubic(\tau_k)$；(3) 归一化 $w[k] = cubic(\tau_k) / \sum cubic(\tau_j)$；(4) 卷积 $y[n] = \sum_{k=0}^{8} w[k] \times x_{crop}[n-k]$。

10. **偏移量**：$\tau_k \in \{-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0\}$，步长 0.5，共 9 个值。覆盖三次 B 样条的非零支撑区间 $[-2, 2]$，步长 0.5 在精度和计算量间平衡。

11. 归一化 $w[k] = cubic(\tau_k) / \sum cubic(\tau_j)$ 使 $\sum w[k] = 1$，保证平滑滤波的直流增益为 1（平滑后信号幅度不变）。

### 6.4 模板/类型

12. `FirfilterDeviceResult` 是 GPU 版 `firfilter_device` 的返回结构体，包含 `.y` 成员（`DeviceArray<float>` 类型的输出信号）。普通 `DeviceArray` 是单一的 GPU 显存缓冲区。结构体封装了 GPU 版本的完整输出。
13. Step2 使用 `firfilter_fp32_resident_device`（resident 版本，device 端分配），Step3 使用 `firfilter_device`（非 resident 版本，通过 `FirfilterDeviceResult` 管理输出）。这是因为 Step3 的输入信号（cropped）和权重（weights）均经过 H2D 传输，使用非 resident 版本更自然。

### 6.5 并行/同步

14. **两次 H2D**：第一次（cropped + offsets）、第二次（归一化后的 weights）。**两次 D2H**：第一次（cubic 权重）、第二次（smoothed 结果）。共 4 次传输，因为权重归一化需要在 CPU 上执行。
15. `time_gpu` 在执行 lambda 后调用 `cudaDeviceSynchronize()` 确保 GPU 操作完成，然后才计算耗时。

### 6.6 精度/测试边界

16. 通过 `collect_step3_validation` 生成 CPU 参考基准 `smoothed_cpu`，与 GPU 输出的 `state.smoothed` 逐元素比较。
17. `reference_for_correlation` 是原始 chirp 波形的裁剪版本（与 smoothed 相同范围），供 Step4 做互相关峰值定位，用于估计目标延迟。它是发射波形的精确副本，无延迟/干扰/噪声。
18. `require(weight_sum > 0.0F)` 防止所有权重为零时导致的除零错误（`value /= 0.0F` 会产生 inf 或 NaN）。这种情况在偏移量全在 $cubic(\tau)$ 的零值区域时可能发生，但由于 $\tau = 0$ 处 $cubic(0) > 0$，正常情况不会触发。