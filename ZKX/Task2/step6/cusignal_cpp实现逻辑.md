# Task2 Step6 cusignal_cpp 实现逻辑

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
|`ZKX/Task2/task2_gpu_cpu/step6/step6.h`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step6/step6.cu`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step6/step6.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#pragma once

#include "../task2_common.h"

namespace task2 {

// <学习注释：Step6 入口函数 —— Kalman 滤波参数估计流水线。
//
//  【函数总体处理方法】
//  run_step6 负责使用 Kalman 滤波对 Step5 关键点附近观测进行递推估计，输出目标距离归一化估计与协方差。
//  处理流程分五个阶段：
//    阶段1（CPU）：在 Host 端从 Step5 extrema 选出最强特征点 anchor，构造 Kalman 观测序列
//    阶段2（H2D）：将初始状态 x0、协方差 P0、F/Q/H/R/alpha 和观测序列拷贝到 GPU 显存
//    阶段3（GPU 递推）：调用 kalman_predict_update_scalar_sequence_device 完成标量 Kalman predict/update 序列
//    阶段4（D2H）：将最终状态估计 x̂ 和协方差 P 回传 Host，记录各阶段耗时证据
//    阶段5（证据）：写入 evidence.operator_ms["kalmanfilter_predict_update"] 和显存使用指标
//
//  【核心数学公式 —— 标量 Kalman 递推模型】
//    对观测序列 k（k = 1, 2, ..., K）：
//      预测：x̂_k^- = F · x̂_{k-1}
//            P_k^- = F · P_{k-1} · F^T + Q
//      更新：K_k  = P_k^- · H^T · (H · P_k^- · H^T + R)^{-1}
//            x̂_k  = x̂_k^- + K_k · (z_k - H · x̂_k^-)
//            P_k   = (I - K_k · H) · P_k^- · (I - K_k · H)^T + K_k · R · K_k^T
//
//    符号说明：
//      x̂   = estimate            ：当前状态估计（标量归一化目标位置）
//      P    = kalman_covariance  ：估计协方差（不确定度）
//      F    = kalman_f           ：状态转移系数（标量）
//      Q    = kalman_q           ：过程噪声协方差
//      H    = kalman_h           ：观测模型系数
//      R    = kalman_r           ：观测噪声协方差
//      α    = alpha              ：新息缩放因子
//      z_k  = observations[k]    ：第 k 次观测值（由特征点邻域加权形成）
//      K_k                       ：Kalman 增益
//
//  【物理意义】
//  模拟雷达参数估计中，对关键点附近观测序列进行 Kalman 递推，利用动态模型预测和带噪观测
//  的协方差加权融合，得到目标距离的最优估计和不确定度。这是 T2-R6（参数估计）条款的核心
//  实现，输出 estimate 将作为后续结果与可视化模块的输入。>
void prepare_step6_host_inputs(PipelineState& state);
StepEvidence run_step6_prepared(PipelineState& state);
StepEvidence run_step6(PipelineState& state);

}  // namespace task2
```

### 3.2 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#include "step6.h"

#include "cuda_utils/device_array.h"
#include "estimation/estimation_typed.h"

namespace task2 {
namespace {

// <学习注释：Host 观测序列构造 —— observations_from_feature_point。
//
//  【函数总体处理方法】
//  以 Step5 选出的 anchor 为中心，按 radius=1..K 在 feature_bundle 邻域内做加权质心计算，
//  产生 K 个归一化观测 z_k，作为 Kalman 滤波的输入观测序列。每个 radius 取 [a-r, a+r]
//  窗口内非负特征值作权重，将加权索引归一化到 [0,1] 后追加到 observations。
//
//  【核心数学公式 —— 加权质心观测模型】
//    对每个 radius r（r = 1, 2, ..., K）：
//      left  = max(0, a - r)
//      right = min(N - 1, a + r)
//      w_i   = max(0, bundle[i])         （非负权重，i ∈ [left, right]）
//      coordinate = (Σ w_i · i) / (Σ w_i)   （加权质心，weight=0 时回退为 anchor）
//      z_r  = coordinate / (N - 1)        （归一化到 [0,1]）
//
//  【物理意义】
//  将 Step5 检测到的关键点邻域特征强度作为位置似然，质心给出该 radius 尺度下目标距离的
//  归一化观测 z_k；不同 radius 形成多尺度观测序列，供 Kalman predict/update 递推。>
std::vector<float> observations_from_feature_point(
    const std::vector<float>& bundle, std::int64_t anchor, int observation_count)
{
    std::vector<float> observations;
    observations.reserve(observation_count);
    for (int radius = 1; radius <= observation_count; ++radius) {
        const int left = std::max(0, static_cast<int>(anchor) - radius);
        const int right = std::min(static_cast<int>(bundle.size()) - 1,
                                   static_cast<int>(anchor) + radius);
        double weighted_index = 0.0;
        double weight = 0.0;
        for (int index = left; index <= right; ++index) {
            const double local_weight = std::max(0.0F, bundle[index]);
            weighted_index += local_weight * index;
            weight += local_weight;
        }
        const double coordinate = weight > 0.0 ? weighted_index / weight : anchor;
        observations.push_back(static_cast<float>(coordinate / (bundle.size() - 1)));
    }
    return observations;
}

}  // namespace

// <学习注释：Host 输入准备 —— prepare_step6_host_inputs。
//
//  【函数总体处理方法】
//  从 Step5 extrema 中选出 feature_bundle 上幅值最强的特征点作为 Kalman anchor，
//  调用 observations_from_feature_point 生成观测序列，并由 bundle 长度和目标延迟
//  推算归一化真值 state.truth 用于后续误差评估。
//
//  【核心数学公式 —— anchor 与 truth】
//    anchor  = argmax_{n ∈ extrema} feature_bundle[n]
//    truth   = (0.5 · (N - 1) + target_delay) / (N - 1)
//    其中 N = feature_bundle.size()，target_delay 来自 config。
//
//  【物理意义】
//  anchor 是当前帧最强反射峰位置，作为观测序列的中心；truth 是该 demo 配置下目标
//  距离的归一化参考值，用于校验 Kalman 估计是否收敛到正确位置。>
void prepare_step6_host_inputs(PipelineState& state)
{
    const auto& config = state.config;
    require_task2_upstream(!state.extrema.empty(), "Step6", "Step5.extrema");
    require_task2_upstream(
        !state.feature_bundle.empty(), "Step6", "Step4.feature_bundle");
    const auto strongest = *std::max_element(
        state.extrema.begin(), state.extrema.end(),
        [&](std::int64_t left, std::int64_t right) {
            return state.feature_bundle[static_cast<std::size_t>(left)] <
                   state.feature_bundle[static_cast<std::size_t>(right)];
        });
    state.kalman_anchor = strongest;
    state.kalman_observations = observations_from_feature_point(
        state.feature_bundle, strongest, config.kalman_observation_count);
    state.truth = (0.5F * static_cast<float>(state.feature_bundle.size() - 1) +
                   static_cast<float>(config.target_delay_samples)) /
                  static_cast<float>(state.feature_bundle.size() - 1);
}

namespace {

// <学习注释：GPU Host wrapper —— run_step6_impl。
//
//  【函数总体处理方法】
//  串联 Step6 完整流水线，分六个阶段：
//    阶段1（输入准备）：可选调用 prepare_step6_host_inputs 构造 anchor/observations/truth
//    阶段2（参数构造）：layout(1,1,1)、initial_x=[0]、initial_p=[1]、F/Q/H/R/α=1 来自 config
//    阶段3（H2D）：将 initial_x、initial_p、f、q、alpha、h、r、observations 拷贝到 GPU
//    阶段4（GPU 递推）：initialize_kalman_device_state + kalman_predict_update_scalar_sequence_device
//    阶段5（D2H）：device_state.x/p 回传 Host，得到 estimate 与 kalman_covariance
//    阶段6（后处理）：shape 校验、写入 state.estimate、汇总 evidence 与 metrics
//
//  【核心数学公式 —— 标量 Kalman 递推】
//    预测：x̂_k^- = F · x̂_{k-1},   P_k^- = F · P_{k-1} · F^T + Q
//    更新：K_k  = P_k^- · H^T · (H · P_k^- · H^T + R)^{-1}
//          x̂_k  = x̂_k^- + K_k · (z_k - H · x̂_k^-)
//          P_k   = (I - K_k · H) · P_k^- · (I - K_k · H)^T + K_k · R · K_k^T
//    本任务为标量情形：F, H, α=1，R 与 Q 由 config 给出，初始 x0=0, P0=1。
//
//  【并行策略】
//  Host 端串行调度；GPU 上的 predict/update 序列由 cusignal 算子内部决定并行方式，
//  本函数只做 H2D、单次 kernel 调用和 D2H，不显式启动 grid/block。>
StepEvidence run_step6_impl(PipelineState& state, bool prepare_inputs)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step6";
    const auto total_begin = Clock::now();
    // <学习注释：阶段1 输入准备 —— prepare_inputs=true 时构造观测序列，否则校验已准备数据。>
    auto begin = Clock::now();
    if (prepare_inputs) prepare_step6_host_inputs(state);
    else require_task2_upstream(!state.kalman_observations.empty(),
            "Step6", "prepared.kalman_observations");
    // <学习注释：阶段2 参数构造 —— 标量 Kalman 初值与 F/Q/H/R/α。>
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F};
    const std::vector<float> initial_p{1.0F};
    const std::vector<float> f{config.kalman_f}, q{config.kalman_q}, alpha{1.0F};
    const std::vector<float> h{config.kalman_h}, r{config.kalman_r};
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段3 H2D —— initial_x/p、F/Q/α/H/R、observations 拷贝到 GPU 显存。>
    begin = Clock::now();
    auto d_initial_x = cusignal::DeviceArray<float>::from_host(initial_x);
    auto d_initial_p = cusignal::DeviceArray<float>::from_host(initial_p);
    auto d_f = cusignal::DeviceArray<float>::from_host(f);
    auto d_q = cusignal::DeviceArray<float>::from_host(q);
    auto d_alpha = cusignal::DeviceArray<float>::from_host(alpha);
    auto d_h = cusignal::DeviceArray<float>::from_host(h);
    auto d_r = cusignal::DeviceArray<float>::from_host(r);
    auto d_observations = cusignal::DeviceArray<float>::from_host(
        state.kalman_observations);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    // <学习注释：阶段4 GPU 递推 —— 初始化 device state，调用标量 Kalman predict/update 序列算子。>
    cusignal::KalmanDeviceState device_state(layout);
    cusignal::initialize_kalman_device_state(d_initial_x, d_initial_p, device_state);
    const float kalman_gpu_ms = time_gpu([&] {
        cusignal::kalman_predict_update_scalar_sequence_device(
            device_state, d_f, d_q, d_alpha, d_h, d_r, d_observations);
    });
    // <学习注释：记录显存峰值和算子耗时证据；sequence_kernel_count=1 表示单次 launch。>
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    evidence.operator_ms["kalmanfilter_predict_update"] = kalman_gpu_ms;
    evidence.compute_ms = kalman_gpu_ms;
    evidence.metrics["observation_h2d_batch_count"] = 1.0;
    evidence.metrics["sequence_kernel_count"] = 1.0;
    evidence.metrics["per_observation_event_sync_count"] = 0.0;
    // <学习注释：阶段5 D2H —— device_state.x/p 回传 Host。>
    begin = Clock::now();
    const auto estimated_state = device_state.x.to_host();
    state.kalman_covariance = device_state.p.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段6 后处理 —— 校验标量 shape，写入 state.estimate，汇总 evidence。>
    begin = Clock::now();
    require(estimated_state.size() == 1 && state.kalman_covariance.size() == 1,
            "step6 state shape mismatch");
    state.estimate = estimated_state.front();
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
}

}  // namespace

// <学习注释：已准备输入的 Step6 入口 —— run_step6_prepared。
//  调用 run_step6_impl(state, false)，跳过输入准备，要求 state 已含 kalman_observations。>
StepEvidence run_step6_prepared(PipelineState& state)
{
    return run_step6_impl(state, false);
}

// <学习注释：Step6 公开入口 —— run_step6。
//  调用 run_step6_impl(state, true)，自动执行 prepare_step6_host_inputs。>
StepEvidence run_step6(PipelineState& state)
{
    return run_step6_impl(state, true);
}

}  // namespace task2
```

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：CPU 参考实现 —— collect_step6_validation。
//
//  【函数总体处理方法】
//  使用与 GPU 相同的观测序列和参数，在 Host 端串行运行两遍标量 Kalman 递推，得到 CPU 参考
//  估计和协方差，以及扰动对照估计：
//    1) 主参考：以 config.kalman_r 作为 R，逐观测调用 kalman_predict_typed_cpu + kalman_update_typed_cpu
//    2) 扰动对照：以 config.kalman_measurement_noise_variant 作为 alternate R 重跑同一序列
//  两次递推使用相同的 F/Q/H/α、initial_x=[0]、initial_p=[1]。
//
//  【核心数学公式 —— CPU 标量 Kalman 递推（与 GPU 一致）】
//    预测：x̂_k^- = F · x̂_{k-1},   P_k^- = F · P_{k-1} · F^T + Q
//    更新：K_k  = P_k^- · H^T · (H · P_k^- · H^T + R)^{-1}
//          x̂_k  = x̂_k^- + K_k · (z_k - H · x̂_k^-)
//          P_k   = (I - K_k · H) · P_k^- · (I - K_k · H)^T + K_k · R · K_k^T
//    扰动对照使用 R' = kalman_measurement_noise_variant 替代 R，其余参数不变。
//
//  【物理意义】
//  主参考用于逐元素比较 GPU 输出，验证 device 实现的数值正确性；扰动对照用于检验 R 参数
//  变化对估计结果的可检测性，确保验证口径对参数敏感性可感知。>
Step6ValidationData collect_step6_validation(const PipelineState& state)
{
    Step6ValidationData data;
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F}, initial_p{1.0F};
    const std::vector<float> f{state.config.kalman_f}, q{state.config.kalman_q}, alpha{1.0F};
    // <学习注释：主参考递推 —— 使用 config.kalman_r 作为 R，逐观测执行 predict/update。>
    const std::vector<float> h{state.config.kalman_h}, r{state.config.kalman_r};
    auto cpu_state = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
        cusignal::kalman_predict_typed_cpu(cpu_state, f, q, alpha);
        cusignal::kalman_update_typed_cpu(cpu_state, h, r, z);
    }
    data.state_cpu = cpu_state.x;
    data.covariance_cpu = cpu_state.p;
    // <学习注释：扰动对照递推 —— 使用 kalman_measurement_noise_variant 作为 alternate R。>
    auto alternate = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
    const std::vector<float> alternate_r{state.config.kalman_measurement_noise_variant};
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
        cusignal::kalman_predict_typed_cpu(alternate, f, q, alpha);
        cusignal::kalman_update_typed_cpu(alternate, h, alternate_r, z);
    }
    data.alternate_estimate_cpu = alternate.x.front();
    return data;
}
```

### 3.4 函数总体处理方法

本节从方案设计的角度，概述 Step6 中所有函数的总体处理方法和它们之间的协作关系。

#### 3.4.1 总体流水线架构

Step6 的 Kalman 参数估计流水线由以下函数协作完成：

```
                    ┌─────────────────────────────────────────────────┐
                    │              run_step6 (公开入口)               │
                    │              ↓                                  │
                    │         run_step6_impl(state, true)             │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ 阶段1 prepare_step6_host_inputs         │    │
                    │  │   Step5 extrema → strongest anchor       │    │
                    │  │   → observations_from_feature_point      │    │
                    │  │   → kalman_observations, state.truth      │    │
                    │  └─────────────────────────────────────────┘    │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ 阶段2 参数构造: layout(1,1,1)            │    │
                    │  │   initial_x=[0], initial_p=[1]           │    │
                    │  │   F/Q/H/R/α=1 ← config                    │    │
                    │  └─────────────────────────────────────────┘    │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ 阶段3 H2D → 阶段4 GPU 递推               │    │
                    │  │   initialize_kalman_device_state         │    │
                    │  │   kalman_predict_update_scalar_sequence   │    │
                    │  │   _device  (predict + update × K)        │    │
                    │  └─────────────────────────────────────────┘    │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ 阶段5 D2H → 阶段6 后处理                 │    │
                    │  │   device_state.x → state.estimate         │    │
                    │  │   device_state.p → kalman_covariance      │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         ↑ 精度对比验证
                    ┌─────────────────────────────────────────────────┐
                    │      collect_step6_validation (CPU 参考)        │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  主参考: kalman_predict_typed_cpu         │   │
                    │  │          + kalman_update_typed_cpu (R)    │   │
                    │  │  扰动对照: 同序列 + alternate R'           │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### 3.4.2 各函数处理方法一览

| 函数 | 类型 | 职责 | 核心数学公式 | 并行策略 |
|------|------|------|-------------|----------|
| `run_step6` | GPU 入口 | 公开任务入口，自动准备输入并执行完整流水线 | 调用 `run_step6_impl(state, true)` | 串行调度 |
| `run_step6_prepared` | GPU 入口 | 消费已准备观测序列的入口，跳过输入准备 | 调用 `run_step6_impl(state, false)` | 串行调度 |
| `run_step6_impl` | GPU Host wrapper | 串联输入准备 → 参数构造 → H2D → GPU 递推 → D2H → 后处理 | 见 3.4.3 标量 Kalman 公式 | 串行编排，GPU kernel 内部并行 |
| `prepare_step6_host_inputs` | Host 输入准备 | 选 anchor、构造 observations、推算 truth | $anchor = \arg\max_n F[n]$；$z_r = \frac{\sum w_i \cdot i}{(N-1)\sum w_i}$ | 串行 |
| `observations_from_feature_point` | Host observation builder | 按 radius 在 anchor 邻域加权质心生成观测序列 | $z_r = \frac{\sum_{i \in [a-r,a+r]} \max(0,bundle[i]) \cdot i}{(N-1) \sum \max(0,bundle[i])}$ | 串行 for 循环 |
| `collect_step6_validation` | CPU 参考 | 串行运行主参考与扰动对照 Kalman 递推 | 与 GPU 一致，扰动使用 $R'$ | 串行 for 循环 |

#### 3.4.3 Kalman filter 数学与物理模型

Step6 使用标量 Kalman filter 对观测序列 $\{z_k\}_{k=1}^K$ 递推，得到目标距离的归一化估计 $\hat{x}$ 与不确定度 $P$：

**预测步（prior）**：
$$\hat{x}_k^- = F \cdot \hat{x}_{k-1}, \quad P_k^- = F \cdot P_{k-1} \cdot F^T + Q$$

**更新步（posterior）**：
$$K_k = \frac{P_k^- \cdot H^T}{H \cdot P_k^- \cdot H^T + R}, \quad \hat{x}_k = \hat{x}_k^- + K_k (z_k - H \cdot \hat{x}_k^-), \quad P_k = (I - K_k \cdot H) \cdot P_k^- \cdot (I - K_k \cdot H)^T + K_k \cdot R \cdot K_k^T$$

**任务实例化参数**：

| 参数 | 含义 | 取值/来源 |
|------|------|----------|
| $F$ | 状态转移系数 | `config.kalman_f`（标量） |
| $Q$ | 过程噪声协方差 | `config.kalman_q` |
| $H$ | 观测模型系数 | `config.kalman_h`（标量） |
| $R$ | 观测噪声协方差 | `config.kalman_r` |
| $\alpha$ | 新息缩放因子 | 固定 `1.0F` |
| $\hat{x}_0$ | 初始状态估计 | `0.0F`（无先验位置） |
| $P_0$ | 初始估计协方差 | `1.0F`（高不确定度） |
| $z_k$ | 第 k 次观测 | `observations_from_feature_point` 输出 |

##### 算子数学原理在当前任务中的实例化

- 学习入口：[数学物理原理](../../operators/estimation/kalmanfilter/数学物理原理.md)、[Python 源码算法](../../operators/estimation/kalmanfilter/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/estimation/kalmanfilter/cusignal_cpp_kalmanfilter复现逻辑.md)。
- 当前调用采用的数学关系：预测 $\hat x_k^-=F\hat x_{k-1}$、$P_k^-=FP_{k-1}F^T+Q$；更新 $K=P^-H^T(HP^-H^T+R)^{-1}$、$\hat x=\hat x^-+K(z-H\hat x^-)$。
- 任务变量与参数实例化：`f/q/h/r→F/Q/H/R`、`observations→z_k`、`estimate→x̂`、`kalman_covariance→P`；任务使用标量序列 predict/update。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 状态估计模型的物理含义

Step6 的核心是雷达目标距离的归一化估计：anchor 是 Step5 在归一化特征强度上找到的最强反射点，邻域加权质心给出多尺度观测 $z_k$，Kalman 递推将动态模型预测与带噪观测按协方差加权融合，输出目标距离的 MMSE 估计。

$$\hat{x}_{\text{final}} = \text{Kalman}(\{z_k\}_{k=1}^K; F, Q, H, R, \hat{x}_0=0, P_0=1)$$

| 分量 | 物理来源 | 实现方式 | 输出用途 |
|------|---------|---------|---------|
| 观测 $z_k$ | Step5 关键点邻域特征加权质心 | `observations_from_feature_point` 按 radius 多尺度构造 | Kalman update 输入 |
| 状态 $\hat{x}$ | 目标距离归一化估计 | Kalman predict + update 序列递推 | `state.estimate`，后续结果/可视化 |
| 协方差 $P$ | 估计不确定度 | Kalman 协方差传播 | `state.kalman_covariance`，置信度评估 |
| anchor | 最强反射点位置 | `std::max_element` on `feature_bundle` | `state.kalman_anchor`，观测中心 |
| truth | 目标距离归一化参考 | `(0.5(N-1)+delay)/(N-1)` | 误差评估的 ground truth |
| 扰动 $R'$ | 观测噪声扰动 | `kalman_measurement_noise_variant` | 验证参数敏感性 |

#### 3.4.5 CPU/GPU 精度验证策略

| 验证维度 | CPU 实现 | GPU 实现 | 验证方法 |
|---------|---------|---------|---------|
| Kalman 递推 | `kalman_predict_typed_cpu` + `kalman_update_typed_cpu` 串行 | `kalman_predict_update_scalar_sequence_device` 单次 launch | 逐元素比较 `state.estimate` 与 `data.state_cpu` |
| 协方差传播 | 同上 CPU 算子 | 同上 device 算子 | 比较 `kalman_covariance` 与 `data.covariance_cpu` |
| 参数敏感性 | 扰动对照使用 alternate $R'$ | GPU 主输出 | 比较 `data.alternate_estimate_cpu` 与 `state.estimate`，验证差异可检测 |
| 输入一致性 | 共用 `state.kalman_observations`、`config` | 同上 | 保证 CPU/GPU 使用相同输入参数 |
| 资源/计时 | 不适用 | `evidence.metrics`、`operator_ms` | 记录显存峰值、kernel 次数、H2D/D2H 耗时 |

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once

#include "../task2_common.h"
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `pragma`、`once`、`include`、`task2_common`、`h`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
- `#include "../task2_common.h"` 在预处理阶段引入 "../task2_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

void prepare_step6_host_inputs(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明、函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `void prepare_step6_host_inputs(PipelineState& state);` 中的 `prepare_step6_host_inputs` 是函数声明；名称前的 `void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `prepare_step6_host_inputs`；调用 `prepare_step6_host_inputs`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 run_step6_prepared：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.h`；符号 `run_step6_prepared`；源码锚点 `StepEvidence run_step6_prepared(PipelineState& state);`。

```cpp
StepEvidence run_step6_prepared(PipelineState& state);
StepEvidence run_step6(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step6_prepared`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`run_step6`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step6_prepared(PipelineState& state);` 中的 `run_step6_prepared` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。
- `StepEvidence run_step6(PipelineState& state);` 中的 `run_step6` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `run_step6_prepared`, `run_step6`；调用 `run_step6_prepared`, `run_step6`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6_prepared` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 run_step6_prepared：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.h`；符号 `run_step6_prepared`；源码锚点 `}  // namespace task2`。

```cpp

}  // namespace task2
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}  // namespace task2` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6_prepared` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_prepared`；源码锚点 `#include "step6.h"`。

```cpp
#include "step6.h"

#include "cuda_utils/device_array.h"
#include "estimation/estimation_typed.h"
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step6`、`h`、`cuda_utils`、`device_array`、`estimation`、`estimation_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step6.h"` 在预处理阶段引入 "step6.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "estimation/estimation_typed.h"` 在预处理阶段引入 "estimation/estimation_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 run_step6_prepared：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_prepared`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {
namespace {
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6_prepared` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.7 observations_from_feature_point：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `std::vector<float> observations_from_feature_point(`。

```cpp
std::vector<float> observations_from_feature_point(
    const std::vector<float>& bundle, std::int64_t anchor, int observation_count)
{
    std::vector<float> observations;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `bundle` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `anchor` 由 `std::int64_t` 声明：类型控制可表示值、可用操作和传参方式；
- `observation_count` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `observations` 由 `std::vector<float>` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|
|`anchor`|局部窗口或观测序列的中心索引|记作 $a$|通常来自最强峰位置，决定左右截取范围|
|`observation_count`|当前一次 Kalman 测量值|记作 $z_k$|逐项送入 update|
|`observations`|Kalman/定位观测序列|记作 $z_1,\ldots,z_K$|由特征点附近数据形成，供参数估计|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<float> observations_from_feature_point( const std::vector<float>& bundle, std::int64_t anchor, int observation_count) {` 中的 `observations_from_feature_point` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& bundle, std::int64_t anchor, int observation_count` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<float> observations;` 是对象声明：类型 `std::vector<float>` 应用于名称 `observations`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `observations_from_feature_point`；调用 `observations_from_feature_point`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.8 observations_from_feature_point：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `observations.reserve(observation_count);`。

```cpp
    observations.reserve(observation_count);
    for (int radius = 1; radius <= observation_count; ++radius) {
        const int left = std::max(0, static_cast<int>(anchor) - radius);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `radius` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `left` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`radius`|围绕 anchor 截取的半径，单位 sample|记作 $r$|形成区间 $[a-r,a+r]$ 并受数组边界裁剪|
|`left`|局部区间左端索引|记作 $\ell=\max(0,a-r)$|用于安全切片或循环起点|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `observations.reserve(observation_count);` 调用向量的 `reserve` 预留至少实参指定的容量，但不改变 `size()`；目的是减少后续 `push_back` 触发的重新分配。
- `for (int radius = 1; radius <= observation_count; ++radius) {` 是经典 `for`：先执行初始化 `int radius = 1`；每轮前检查 `radius <= observation_count`，为假即退出；每轮循环体结束后执行 `++radius`，再检查下一轮。
- `const int left = std::max(0, static_cast<int>(anchor) - radius);` 是声明并初始化：`const int left` 建立局部对象 `left`，右侧完整表达式 `std::max(0, static_cast<int>(anchor) - radius)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `reserve`, `std::max`；写入/初始化 `left`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.9 observations_from_feature_point：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `const int right = std::min(static_cast<int>(bundle.size()) - 1,`。

```cpp
        const int right = std::min(static_cast<int>(bundle.size()) - 1,
                                   static_cast<int>(anchor) + radius);
        double weighted_index = 0.0;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `right` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `weighted_index` 由 `double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`right`|局部区间右端索引|记作 $u=\min(N,a+r+1)$|用于安全切片或循环终点|
|`weighted_index`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `const int right = std::min(static_cast<int>(bundle.size()) - 1, static_cast<int>(anchor) + radius);` 是声明并初始化：`const int right` 建立局部对象 `right`，右侧完整表达式 `std::min(static_cast<int>(bundle.size()) - 1, static_cast<int>(anchor) + radius)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `double weighted_index = 0.0;` 是声明并初始化：`double weighted_index` 建立局部对象 `weighted_index`，右侧完整表达式 `0.0` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `std::min`, `size`；写入/初始化 `right`, `weighted_index`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 observations_from_feature_point：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `double weight = 0.0;`。

```cpp
        double weight = 0.0;
        for (int index = left; index <= right; ++index) {
            const double local_weight = std::max(0.0F, bundle[index]);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `weight` 由 `double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；
- `index` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `local_weight` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`local_weight`|当前局部样本的权重|记作 $w_i$|用于局部加权坐标或观测构造|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `double weight = 0.0;` 是声明并初始化：`double weight` 建立局部对象 `weight`，右侧完整表达式 `0.0` 产生初值。
- `for (int index = left; index <= right; ++index) {` 是经典 `for`：先执行初始化 `int index = left`；每轮前检查 `index <= right`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `const double local_weight = std::max(0.0F, bundle[index]);` 是声明并初始化：`const double local_weight` 建立局部对象 `local_weight`，右侧完整表达式 `std::max(0.0F, bundle[index])` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `std::max`；写入/初始化 `weight`, `local_weight`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.11 observations_from_feature_point：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `weighted_index += local_weight * index;`。

```cpp
            weighted_index += local_weight * index;
            weight += local_weight;
        }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `local_weight *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `weighted_index += local_weight * index;` 是复合赋值：读取 `weighted_index` 的旧值，与 `local_weight * index` 执行 `+` 运算，再把结果写回同一对象。
- `weight += local_weight;` 是复合赋值：读取 `weight` 的旧值，与 `local_weight` 执行 `+` 运算，再把结果写回同一对象。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过写入/初始化 `weighted_index`, `weight`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 observations_from_feature_point：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `const double coordinate = weight > 0.0 ? weighted_index / weight : anchor;`。

```cpp
        const double coordinate = weight > 0.0 ? weighted_index / weight : anchor;
        observations.push_back(static_cast<float>(coordinate / (bundle.size() - 1)));
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `coordinate` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`coordinate`|当前特征点的离散坐标/索引|记作 $n_i$|与权重组合形成局部观测|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const double coordinate = weight > 0.0 ? weighted_index / weight : anchor;` 是声明并初始化：`const double coordinate` 建立局部对象 `coordinate`，右侧完整表达式 `weight > 0.0 ? weighted_index / weight : anchor` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。
- `observations.push_back(static_cast<float>(coordinate / (bundle.size() - 1)));` 先求得实参值，再由 `push_back` 把一个新元素追加到向量末尾；必要时向量可能重新分配。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `push_back`, `size`；写入/初始化 `coordinate`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 observations_from_feature_point：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `return observations;`。

```cpp
    return observations;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `observations` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`observations`|Kalman/定位观测序列|记作 $z_1,\ldots,z_K$|由特征点附近数据形成，供参数估计|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return observations;` 是返回语句：先求值 `observations`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过返回 `observations`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.14 observations_from_feature_point：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `observations_from_feature_point`；源码锚点 `}  // namespace`。

```cpp

}  // namespace
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}  // namespace` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `observations_from_feature_point` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 prepare_step6_host_inputs：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `void prepare_step6_host_inputs(PipelineState& state)`。

```cpp
void prepare_step6_host_inputs(PipelineState& state)
{
    const auto& config = state.config;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `config` 由 `const auto&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `void prepare_step6_host_inputs(PipelineState& state) {` 中的 `prepare_step6_host_inputs` 是函数定义；名称前的 `void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `prepare_step6_host_inputs`；调用 `prepare_step6_host_inputs`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.16 prepare_step6_host_inputs：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `require_task2_upstream(!state.extrema.empty(), "Step6", "Step5.extrema");`。

```cpp
    require_task2_upstream(!state.extrema.empty(), "Step6", "Step5.extrema");
    require_task2_upstream(
        !state.feature_bundle.empty(), "Step6", "Step4.feature_bundle");
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `require_task2_upstream`、`state`、`extrema`、`empty`、`Step6`、`Step5`、`feature_bundle`、`Step4`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `require_task2_upstream(!state.extrema.empty(), "Step6", "Step5.extrema");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
- `require_task2_upstream( !state.feature_bundle.empty(), "Step6", "Step4.feature_bundle");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `require_task2_upstream`, `empty`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `prepare_step6_host_inputs` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.17 prepare_step6_host_inputs：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `const auto strongest = *std::max_element(`。

```cpp
    const auto strongest = *std::max_element(
        state.extrema.begin(), state.extrema.end(),
        [&](std::int64_t left, std::int64_t right) {
            return state.feature_bundle[static_cast<std::size_t>(left)] <
                   state.feature_bundle[static_cast<std::size_t>(right)];
        });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `strongest` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `left` 由 `std::int64_t` 声明：类型控制可表示值、可用操作和传参方式；
- `right` 由 `std::int64_t` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`strongest`|幅值或得分最大的特征点索引|记作 $n^*=\arg\max_n F[n]$|作为 Kalman anchor 或最终定位候选|
|`left`|局部区间左端索引|记作 $\ell=\max(0,a-r)$|用于安全切片或循环起点|
|`right`|局部区间右端索引|记作 $u=\min(N,a+r+1)$|用于安全切片或循环终点|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const auto strongest = *std::max_element( state.extrema.begin(), state.extrema.end(), [&](std::int64_t left, std::int64_t right) { return state.feature_bundle[static_cast<std::size_t>(left)] < state.feature_bundle[static_cast<std::size_t>(right)]; });` 是声明并初始化：`const auto strongest` 建立局部对象 `strongest`，右侧完整表达式 `*std::max_element( state.extrema.begin(), state.extrema.end(), [&](std::int64_t left, std::int64_t right) { return state.feature_bundle[static_cast<std::size_t>(left)] < state.feature_bundle[static_cast<std::size_t>(right)]; })` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `std::max_element`, `begin`, `end`；写入/初始化 `strongest`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `prepare_step6_host_inputs` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.18 prepare_step6_host_inputs：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `state.kalman_anchor = strongest;`。

```cpp
    state.kalman_anchor = strongest;
    state.kalman_observations = observations_from_feature_point(
        state.feature_bundle, strongest, config.kalman_observation_count);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`kalman_anchor`、`strongest`、`kalman_observations`、`observations_from_feature_point`、`feature_bundle`、`config`、`kalman_observation_count`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.kalman_anchor = strongest;` 是赋值：先求得右侧完整表达式 `strongest`，再把结果写入左侧可修改对象 `state.kalman_anchor`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.kalman_observations = observations_from_feature_point( state.feature_bundle, strongest, config.kalman_observation_count);` 是赋值：先求得右侧完整表达式 `observations_from_feature_point( state.feature_bundle, strongest, config.kalman_observation_count)`，再把结果写入左侧可修改对象 `state.kalman_observations`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `observations_from_feature_point`；写入/初始化 `kalman_anchor`, `kalman_observations`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.19 prepare_step6_host_inputs：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `state.truth = (0.5F * static_cast<float>(state.feature_bundle.size() - 1) +`。

```cpp
    state.truth = (0.5F * static_cast<float>(state.feature_bundle.size() - 1) +
                   static_cast<float>(config.target_delay_samples)) /
                  static_cast<float>(state.feature_bundle.size() - 1);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `state.truth = (0.5F * static_cast<float>(state.feature_bundle.size() - 1) + static_cast<float>(config.target_delay_samples)) / static_cast<float>(state.feature_bundle.size() - 1);` 是赋值：先求得右侧完整表达式 `(0.5F * static_cast<float>(state.feature_bundle.size() - 1) + static_cast<float>(config.target_delay_samples)) / static_cast<float>(state.feature_bundle.size() - 1)`，再把结果写入左侧可修改对象 `state.truth`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `size`；写入/初始化 `truth`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 prepare_step6_host_inputs：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `}`。

```cpp
}
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 ；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `prepare_step6_host_inputs` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.21 prepare_step6_host_inputs：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `prepare_step6_host_inputs`；源码锚点 `namespace {`。

```cpp

namespace {
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `prepare_step6_host_inputs` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 run_step6_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_impl`；源码锚点 `StepEvidence run_step6_impl(PipelineState& state, bool prepare_inputs)`。

```cpp
StepEvidence run_step6_impl(PipelineState& state, bool prepare_inputs)
{
    const auto& config = state.config;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `prepare_inputs` 由 `bool` 声明：类型控制可表示值、可用操作和传参方式；
- `config` 由 `const auto&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step6_impl`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`prepare_inputs`|当前函数或调用的参数名称，接收调用者绑定的输入 `prepare_inputs`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `StepEvidence run_step6_impl(PipelineState& state, bool prepare_inputs)`；值在 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step6_impl(PipelineState& state, bool prepare_inputs) {` 中的 `run_step6_impl` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state, bool prepare_inputs` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `run_step6_impl`；调用 `run_step6_impl`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.23 run_step6_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_impl`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step6";
    const auto total_begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `evidence` 由 `StepEvidence` 声明：类型控制可表示值、可用操作和传参方式；
- `total_begin` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `evidence.name = "step6";` 是赋值：先求得右侧完整表达式 `"step6"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `total_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6_impl` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 run_step6_impl：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_impl`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    if (prepare_inputs) prepare_step6_host_inputs(state);
    else require_task2_upstream(!state.kalman_observations.empty(),
            "Step6", "prepared.kalman_observations");
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`if` 条件语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `if (prepare_inputs) prepare_step6_host_inputs(state);` 是 `if` 条件语句：先把完整条件 `prepare_inputs` 求值并转换为布尔值。 条件为真时只执行其直接子语句 `prepare_step6_host_inputs(state);`；为假时跳过该子语句。
- `else require_task2_upstream(!state.kalman_observations.empty(), "Step6", "prepared.kalman_observations");` 是前一 `if` 的兜底分支，仅在此前条件均为假时执行。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过检查 `prepare_inputs`；调用 `Clock::now`, `prepare_step6_host_inputs`, `require_task2_upstream`, `empty`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.25 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `layout`；源码锚点 `const cusignal::KalmanLayout layout(1, 1, 1);`。

```cpp
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F};
    const std::vector<float> initial_p{1.0F};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`layout`|数组维度、步长或算子状态布局描述|无独立数学符号|保证 Host/GPU/算子对 shape 的解释一致|
|`initial_x`|初始状态估计|记作 $\hat x_0$|初始化 Kalman device/CPU 状态|
|`initial_p`|初始估计协方差|记作 $P_0$|初始化不确定度|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const cusignal::KalmanLayout layout(1, 1, 1);` 声明 `layout`，其静态类型是 `const cusignal::KalmanLayout`，并用 `1, 1, 1` 直接初始化/调用该类型的构造函数；这不是调用名为 `layout` 的函数，模板尖括号也不是比较或位移。
- `const std::vector<float> initial_x{0.0F};` 声明 `initial_x`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{0.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `const std::vector<float> initial_p{1.0F};` 声明 `initial_p`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{1.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `layout`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.26 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `layout`；源码锚点 `const std::vector<float> f{config.kalman_f}, q{config.kalman_q}, alpha{1.0F};`。

```cpp
    const std::vector<float> f{config.kalman_f}, q{config.kalman_q}, alpha{1.0F};
    const std::vector<float> h{config.kalman_h}, r{config.kalman_r};
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`f`|状态转移系数/矩阵|记作 $F$；预测为 $\hat x_k^-=F\hat x_{k-1}$|来自配置，传入 Kalman predict|
|`h`|观测模型系数/矩阵|记作 $H$；预测观测为 $H\hat x_k^-$|来自配置，传入 update|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const std::vector<float> f{config.kalman_f}, q{config.kalman_q}, alpha{1.0F};` 声明 `f`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{config.kalman_f}, q{config.kalman_q}, alpha{1.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `const std::vector<float> h{config.kalman_h}, r{config.kalman_r};` 声明 `h`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{config.kalman_h}, r{config.kalman_r}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|输入准备/数据契约|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.27 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `layout`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    auto d_initial_x = cusignal::DeviceArray<float>::from_host(initial_x);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_initial_x` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_initial_x`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 的 GPU 调用链|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_initial_x = cusignal::DeviceArray<float>::from_host(initial_x);` 是声明并初始化：`auto d_initial_x` 建立局部对象 `d_initial_x`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(initial_x)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`；写入/初始化 `begin`, `d_initial_x`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.28 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `layout`；源码锚点 `auto d_initial_p = cusignal::DeviceArray<float>::from_host(initial_p);`。

```cpp
    auto d_initial_p = cusignal::DeviceArray<float>::from_host(initial_p);
    auto d_f = cusignal::DeviceArray<float>::from_host(f);
    auto d_q = cusignal::DeviceArray<float>::from_host(q);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_initial_p` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_f` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_q` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_initial_p`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 的 GPU 调用链|
|`d_f`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 的 GPU 调用链|
|`d_q`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 的 GPU 调用链|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_initial_p = cusignal::DeviceArray<float>::from_host(initial_p);` 是声明并初始化：`auto d_initial_p` 建立局部对象 `d_initial_p`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(initial_p)` 产生初值。
- `auto d_f = cusignal::DeviceArray<float>::from_host(f);` 是声明并初始化：`auto d_f` 建立局部对象 `d_f`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(f)` 产生初值。
- `auto d_q = cusignal::DeviceArray<float>::from_host(q);` 是声明并初始化：`auto d_q` 建立局部对象 `d_q`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(q)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `from_host`；写入/初始化 `d_initial_p`, `d_f`, `d_q`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.29 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `layout`；源码锚点 `auto d_alpha = cusignal::DeviceArray<float>::from_host(alpha);`。

```cpp
    auto d_alpha = cusignal::DeviceArray<float>::from_host(alpha);
    auto d_h = cusignal::DeviceArray<float>::from_host(h);
    auto d_r = cusignal::DeviceArray<float>::from_host(r);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_alpha` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_h` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_r` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_alpha`|CA-CFAR 门限缩放系数|记作 $\alpha$|由 PFA 与训练单元数决定；门限为 $T=\alpha\hat P_n$|
|`d_h`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 的 GPU 调用链|
|`d_r`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 的 GPU 调用链|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_alpha = cusignal::DeviceArray<float>::from_host(alpha);` 是声明并初始化：`auto d_alpha` 建立局部对象 `d_alpha`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(alpha)` 产生初值。
- `auto d_h = cusignal::DeviceArray<float>::from_host(h);` 是声明并初始化：`auto d_h` 建立局部对象 `d_h`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(h)` 产生初值。
- `auto d_r = cusignal::DeviceArray<float>::from_host(r);` 是声明并初始化：`auto d_r` 建立局部对象 `d_r`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(r)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `from_host`；写入/初始化 `d_alpha`, `d_h`, `d_r`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.30 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `layout`；源码锚点 `auto d_observations = cusignal::DeviceArray<float>::from_host(`。

```cpp
    auto d_observations = cusignal::DeviceArray<float>::from_host(
        state.kalman_observations);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_observations` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_observations`|Kalman/定位观测序列|记作 $z_1,\ldots,z_K$|由特征点附近数据形成，供参数估计|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_observations = cusignal::DeviceArray<float>::from_host( state.kalman_observations);` 是声明并初始化：`auto d_observations` 建立局部对象 `d_observations`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host( state.kalman_observations)` 产生初值。
- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|公共基础设施|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`；写入/初始化 `d_observations`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.31 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `cusignal::KalmanDeviceState device_state(layout);`。

```cpp
    cusignal::KalmanDeviceState device_state(layout);
    cusignal::initialize_kalman_device_state(d_initial_x, d_initial_p, device_state);
    const float kalman_gpu_ms = time_gpu([&] {
        cusignal::kalman_predict_update_scalar_sequence_device(
            device_state, d_f, d_q, d_alpha, d_h, d_r, d_observations);
    });
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `kalman_gpu_ms` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`device_state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`kalman_gpu_ms`|以毫秒记录的工程计时字段|无独立算法符号；可记为 $t_{ms}$|由 wall clock 或 CUDA Event 产生，进入证据汇总|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::KalmanDeviceState device_state(layout);` 声明 `device_state`，其静态类型是 `cusignal::KalmanDeviceState`，并用 `layout` 直接初始化/调用该类型的构造函数；这不是调用名为 `device_state` 的函数，模板尖括号也不是比较或位移。
- `cusignal::initialize_kalman_device_state(d_initial_x, d_initial_p, device_state);` 调用 Kalman 设备状态初始化接口，把初始状态向量和协方差写入 `device_state`；对象的具体布局由前面构造的 `KalmanLayout` 约束。
- `const float kalman_gpu_ms = time_gpu([&] { cusignal::kalman_predict_update_scalar_sequence_device( device_state, d_f, d_q, d_alpha, d_h, d_r, d_observations); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `const float kalman_gpu_ms`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `device_state`, `cusignal::initialize_kalman_device_state`, `time_gpu`, `cusignal::kalman_predict_update_scalar_sequence_device`；写入/初始化 `kalman_gpu_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.32 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

```cpp
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `free_bytes` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`free_bytes`|设备当前空闲显存字节数|无独立数学符号|与 total_bytes 组合估计显存使用|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::size_t free_bytes = 0, total_bytes = 0;` 在同一条声明中建立两个 `std::size_t` 对象：`free_bytes` 由 `0` 初始化，`total_bytes` 由 `0` 初始化；逗号分隔两个声明符，不是逗号运算符。
- `CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));` 通过 `CUDA_CHECK` 包装 CUDA Runtime 调用；宏检查返回状态，失败时按项目统一错误路径传播，而实参调用负责实际查询/同步/复制。
- `evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);` 是赋值：先求得右侧完整表达式 `static_cast<double>(total_bytes - free_bytes)`，再把结果写入左侧可修改对象 `evidence.metrics["gpu_used_peak_bytes"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.33 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `evidence.operator_ms["kalmanfilter_predict_update"] = kalman_gpu_ms;`。

```cpp
    evidence.operator_ms["kalmanfilter_predict_update"] = kalman_gpu_ms;
    evidence.compute_ms = kalman_gpu_ms;
    evidence.metrics["observation_h2d_batch_count"] = 1.0;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `evidence.operator_ms["kalmanfilter_predict_update"] = kalman_gpu_ms;` 是赋值：先求得右侧完整表达式 `kalman_gpu_ms`，再把结果写入左侧可修改对象 `evidence.operator_ms["kalmanfilter_predict_update"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.compute_ms = kalman_gpu_ms;` 是赋值：先求得右侧完整表达式 `kalman_gpu_ms`，再把结果写入左侧可修改对象 `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.metrics["observation_h2d_batch_count"] = 1.0;` 是赋值：先求得右侧完整表达式 `1.0`，再把结果写入左侧可修改对象 `evidence.metrics["observation_h2d_batch_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过写入/初始化 `operator_ms["kalmanfilter_predict_update"]`, `compute_ms`, `metrics["observation_h2d_batch_count"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.34 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `evidence.metrics["sequence_kernel_count"] = 1.0;`。

```cpp
    evidence.metrics["sequence_kernel_count"] = 1.0;
    evidence.metrics["per_observation_event_sync_count"] = 0.0;
    begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `evidence.metrics["sequence_kernel_count"] = 1.0;` 是赋值：先求得右侧完整表达式 `1.0`，再把结果写入左侧可修改对象 `evidence.metrics["sequence_kernel_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.metrics["per_observation_event_sync_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.metrics["per_observation_event_sync_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `metrics["sequence_kernel_count"]`, `metrics["per_observation_event_sync_count"]`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于耗时或稳定性证据：它界定测量区间、迭代次数、构建 target 或资源采样，输出统计量而不是任务算法的新数学结果。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.35 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `const auto estimated_state = device_state.x.to_host();`。

```cpp
    const auto estimated_state = device_state.x.to_host();
    state.kalman_covariance = device_state.p.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `estimated_state` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`estimated_state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const auto estimated_state = device_state.x.to_host();` 是声明并初始化：`const auto estimated_state` 建立局部对象 `estimated_state`，右侧完整表达式 `device_state.x.to_host()` 产生初值。
- `state.kalman_covariance = device_state.p.to_host();` 是赋值：先求得右侧完整表达式 `device_state.p.to_host()`，再把结果写入左侧可修改对象 `state.kalman_covariance`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `to_host`, `milliseconds`, `Clock::now`；写入/初始化 `estimated_state`, `kalman_covariance`, `d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.36 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    require(estimated_state.size() == 1 && state.kalman_covariance.size() == 1,
            "step6 state shape mismatch");
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `require(estimated_state.size() == 1 && state.kalman_covariance.size() == 1, "step6 state shape mismatch");` 调用本文件的前置条件检查；第一个实参为真时继续，为假时按第二个实参给出的消息报告配置或数值退化错误。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `Clock::now`, `require`, `size`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于 Kalman 参数估计：观测与状态/协方差进入预测或更新，输出平滑后的估计序列。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.37 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `state.estimate = estimated_state.front();`。

```cpp
    state.estimate = estimated_state.front();
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`estimate`、`estimated_state`、`front`、`evidence`、`post_ms`、`milliseconds`、`begin`、`Clock`、`now`、`total_ms`、`total_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.estimate = estimated_state.front();` 是赋值：先求得右侧完整表达式 `estimated_state.front()`，再把结果写入左侧可修改对象 `state.estimate`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.post_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.post_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `front`, `milliseconds`, `Clock::now`；写入/初始化 `estimate`, `post_ms`, `total_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `device_state` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.38 device_state：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `return evidence;`。

```cpp
    return evidence;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `evidence` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `device_state` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.39 device_state：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `device_state`；源码锚点 `}  // namespace`。

```cpp

}  // namespace
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}  // namespace` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `device_state` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.40 run_step6_prepared：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_prepared`；源码锚点 `StepEvidence run_step6_prepared(PipelineState& state)`。

```cpp
StepEvidence run_step6_prepared(PipelineState& state)
{
    return run_step6_impl(state, false);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step6_prepared`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step6_prepared(PipelineState& state) {` 中的 `run_step6_prepared` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return run_step6_impl(state, false);` 是返回语句：先求值 `run_step6_impl(state, false)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `run_step6_prepared`, `run_step6_impl`；调用 `run_step6_prepared`, `run_step6_impl`；返回 `run_step6_impl(state, false)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6_prepared` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.41 run_step6_prepared：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6_prepared`；源码锚点 `}`。

```cpp
}
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 ；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6_prepared` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.42 run_step6：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6`；源码锚点 `StepEvidence run_step6(PipelineState& state)`。

```cpp

StepEvidence run_step6(PipelineState& state)
{
    return run_step6_impl(state, true);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step6`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step6(PipelineState& state) {` 中的 `run_step6` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return run_step6_impl(state, true);` 是返回语句：先求值 `run_step6_impl(state, true)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `run_step6`, `run_step6_impl`；调用 `run_step6`, `run_step6_impl`；返回 `run_step6_impl(state, true)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.43 run_step6：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6`；源码锚点 `}`。

```cpp
}
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 ；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.44 run_step6：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu`；符号 `run_step6`；源码锚点 `}  // namespace task2`。

```cpp

}  // namespace task2
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}  // namespace task2` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step6` 所在的 `ZKX/Task2/task2_gpu_cpu/step6/step6.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.45 collect_step6_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `collect_step6_validation`；源码锚点 `Step6ValidationData collect_step6_validation(const PipelineState& state)`。

```cpp
Step6ValidationData collect_step6_validation(const PipelineState& state)
{
    Step6ValidationData data;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `data` 由 `Step6ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step6_validation`|测试、收集或契约检查 helper|无独立数学变量；内部指标分别映射到误差/条件公式|消费运行结果并形成证据或失败路径|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step6ValidationData collect_step6_validation(const PipelineState& state) {` 中的 `collect_step6_validation` 是函数定义；名称前的 `Step6ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step6ValidationData data;` 是对象声明：类型 `Step6ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过定义/声明 `collect_step6_validation`；调用 `collect_step6_validation`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.46 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `const cusignal::KalmanLayout layout(1, 1, 1);`。

```cpp
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F}, initial_p{1.0F};
    const std::vector<float> f{state.config.kalman_f}, q{state.config.kalman_q}, alpha{1.0F};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`layout`|数组维度、步长或算子状态布局描述|无独立数学符号|保证 Host/GPU/算子对 shape 的解释一致|
|`initial_x`|初始状态估计|记作 $\hat x_0$|初始化 Kalman device/CPU 状态|
|`f`|状态转移系数/矩阵|记作 $F$；预测为 $\hat x_k^-=F\hat x_{k-1}$|来自配置，传入 Kalman predict|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const cusignal::KalmanLayout layout(1, 1, 1);` 声明 `layout`，其静态类型是 `const cusignal::KalmanLayout`，并用 `1, 1, 1` 直接初始化/调用该类型的构造函数；这不是调用名为 `layout` 的函数，模板尖括号也不是比较或位移。
- `const std::vector<float> initial_x{0.0F}, initial_p{1.0F};` 声明 `initial_x`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{0.0F}, initial_p{1.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `const std::vector<float> f{state.config.kalman_f}, q{state.config.kalman_q}, alpha{1.0F};` 声明 `f`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{state.config.kalman_f}, q{state.config.kalman_q}, alpha{1.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `layout`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.47 layout：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `const std::vector<float> h{state.config.kalman_h}, r{state.config.kalman_r};`。

```cpp
    const std::vector<float> h{state.config.kalman_h}, r{state.config.kalman_r};
    auto cpu_state = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：表达式/声明单元、声明初始化或赋值语句、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `cpu_state` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`h`|观测模型系数/矩阵|记作 $H$；预测观测为 $H\hat x_k^-$|来自配置，传入 update|
|`cpu_state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`observation`|当前一次 Kalman 测量值|记作 $z_k$|逐项送入 update|
|`z`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const std::vector<float> h{state.config.kalman_h}, r{state.config.kalman_r};` 声明 `h`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{state.config.kalman_h}, r{state.config.kalman_r}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `auto cpu_state = cusignal::make_kalman_host_state(initial_x, initial_p, layout);` 是声明并初始化：`auto cpu_state` 建立局部对象 `cpu_state`，右侧完整表达式 `cusignal::make_kalman_host_state(initial_x, initial_p, layout)` 产生初值。
- `for (float observation : state.kalman_observations) {` 是范围 `for`：从 `state.kalman_observations` 依次取得元素并绑定到 `float observation`，随后对每个元素执行循环体；声明中的 `&` 若存在表示引用，修改循环变量会修改原元素。
- `const std::vector<float> z{observation};` 声明 `z`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{observation}` 构造内容；花括号属于初始化，不是新的控制流作用域。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `cusignal::make_kalman_host_state`；写入/初始化 `cpu_state`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.48 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `cusignal::kalman_predict_typed_cpu(cpu_state, f, q, alpha);`。

```cpp
        cusignal::kalman_predict_typed_cpu(cpu_state, f, q, alpha);
        cusignal::kalman_update_typed_cpu(cpu_state, h, r, z);
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`kalman_predict_typed_cpu`、`cpu_state`、`f`、`q`、`alpha`、`kalman_update_typed_cpu`、`h`、`r`、`z`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::kalman_predict_typed_cpu(cpu_state, f, q, alpha);` 调用 CPU Kalman 预测接口，以当前状态及 `F/Q/alpha` 推进先验状态和协方差；函数通过第一个状态对象原地写回结果。
- `cusignal::kalman_update_typed_cpu(cpu_state, h, r, z);` 调用 CPU Kalman 更新接口，以 `H/R/z` 计算创新和增益，并通过第一个状态对象原地写回后验估计。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `cusignal::kalman_predict_typed_cpu`, `cusignal::kalman_update_typed_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.49 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `data.state_cpu = cpu_state.x;`。

```cpp
    data.state_cpu = cpu_state.x;
    data.covariance_cpu = cpu_state.p;
    auto alternate = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `alternate` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`alternate`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `alternate`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `auto alternate = cusignal::make_kalman_host_state(initial_x, initial_p, layout);`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.state_cpu = cpu_state.x;` 是赋值：先求得右侧完整表达式 `cpu_state.x`，再把结果写入左侧可修改对象 `data.state_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.covariance_cpu = cpu_state.p;` 是赋值：先求得右侧完整表达式 `cpu_state.p`，再把结果写入左侧可修改对象 `data.covariance_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto alternate = cusignal::make_kalman_host_state(initial_x, initial_p, layout);` 是声明并初始化：`auto alternate` 建立局部对象 `alternate`，右侧完整表达式 `cusignal::make_kalman_host_state(initial_x, initial_p, layout)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `cusignal::make_kalman_host_state`；写入/初始化 `state_cpu`, `covariance_cpu`, `alternate`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.50 layout：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `const std::vector<float> alternate_r{state.config.kalman_measurement_noise_variant};`。

```cpp
    const std::vector<float> alternate_r{state.config.kalman_measurement_noise_variant};
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `std`、`vector`、`float`、`alternate_r`、`state`、`config`、`kalman_measurement_noise_variant`、`observation`、`kalman_observations`、`z`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`alternate_r`|当前表达式读取或传递的工程名称 `alternate_r`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float> alternate_r{state.config.kalman_measurement_noise_variant};`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp` 当前作用域中产生或消费|
|`observation`|当前一次 Kalman 测量值|记作 $z_k$|逐项送入 update|
|`z`|当前标量观测|记作 $z_k$|进入 Kalman innovation $z_k-H\hat x_k^-$|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 Kalman 的 $F,Q,H,R$、初始状态/协方差和观测构造参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const std::vector<float> alternate_r{state.config.kalman_measurement_noise_variant};` 声明 `alternate_r`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{state.config.kalman_measurement_noise_variant}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `for (float observation : state.kalman_observations) {` 是范围 `for`：从 `state.kalman_observations` 依次取得元素并绑定到 `float observation`，随后对每个元素执行循环体；声明中的 `&` 若存在表示引用，修改循环变量会修改原元素。
- `const std::vector<float> z{observation};` 声明 `z`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{observation}` 构造内容；花括号属于初始化，不是新的控制流作用域。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过按源码所示完成当前作用域的声明、构造或资源状态变更，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.51 layout：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `cusignal::kalman_predict_typed_cpu(alternate, f, q, alpha);`。

```cpp
        cusignal::kalman_predict_typed_cpu(alternate, f, q, alpha);
        cusignal::kalman_update_typed_cpu(alternate, h, alternate_r, z);
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`kalman_predict_typed_cpu`、`alternate`、`f`、`q`、`alpha`、`kalman_update_typed_cpu`、`h`、`alternate_r`、`z`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::kalman_predict_typed_cpu(alternate, f, q, alpha);` 调用 CPU Kalman 预测接口，以当前状态及 `F/Q/alpha` 推进先验状态和协方差；函数通过第一个状态对象原地写回结果。
- `cusignal::kalman_update_typed_cpu(alternate, h, alternate_r, z);` 调用 CPU Kalman 更新接口，以 `H/R/z` 计算创新和增益，并通过第一个状态对象原地写回后验估计。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `cusignal::kalman_predict_typed_cpu`, `cusignal::kalman_update_typed_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- Kalman 递推把动态模型预测与带噪观测按协方差加权融合；直接使用原始峰值更简单，但不会利用时间连续性，也不能同时传播不确定度。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.52 layout：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step6_validation.cpp`；符号 `layout`；源码锚点 `data.alternate_estimate_cpu = alternate.x.front();`。

```cpp
    data.alternate_estimate_cpu = alternate.x.front();
    return data;
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `data` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.alternate_estimate_cpu = alternate.x.front();` 是赋值：先求得右侧完整表达式 `alternate.x.front()`，再把结果写入左侧可修改对象 `data.alternate_estimate_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R6：参数估计：使用 Kalman filter 对关键点附近观测进行递推估计|
|业务角色|测试证据|
|业务输入|T2-R5 的关键点、观测序列以及 F/Q/H/R 等模型参数|
|代码如何落实|本块通过调用 `front`；写入/初始化 `alternate_estimate_cpu`；返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|最终状态估计 `estimate` 与协方差 `kalman_covariance`|
|下一消费者|进入结果、扰动测试、正式证据和可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
