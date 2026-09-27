# Task2 Step1 cusignal_cpp 实现逻辑

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
|`ZKX/Task2/task2_gpu_cpu/step1/step1.h`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step1/step1.cu`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#pragma once

#include "../task2_common.h"

namespace task2 {

// <学习注释：Step1 入口函数 —— 雷达回波信号生成流水线。
//
//  【函数总体处理方法】
//  run_step1 负责生成四种预定义波形，并将其按权重叠加，模拟目标回波、干扰和噪声。
//  处理流程分五个阶段：
//    阶段1（CPU）：在 Host 端计算时间轴 t[n]、脉冲时间轴、相位数组
//    阶段2（H2D）：将数组拷贝到 GPU 显存
//    阶段3（GPU波形生成）：调用四个算子 kernel 分别生成 Chirp/Gausspulse/Sawtooth/Square
//    阶段4（GPU回波叠加）：在 mix_echo_kernel 中并行计算每个采样点的目标/干扰/噪声/回波
//    阶段5（D2H）：将结果回传 Host，记录各阶段耗时证据
//
//  【核心数学公式 —— 回波信号叠加模型】
//    对采样点 n（n = 0, 1, ..., N-1），最终回波信号为：
//      echo[n] = target[n] + interference[n] + noise[n]
//
//    其中各分量定义为：
//      target[n]       = w_t * chirp[n - d]    （n >= d 时；n < d 时为 0）
//      interference[n] = w_g * gausspulse[n] + w_s * sawtooth[n] + w_q * square[n]
//      noise[n]        = A_n * rand(n)          （幅度 A_n 的确定性伪随机噪声）
//
//    符号说明：
//      w_t = target_weight    ：目标回波幅度权重，模拟目标 RCS 和传播衰减
//      w_g = gaussian_weight  ：高斯脉冲干扰权重
//      w_s = sawtooth_weight  ：锯齿波干扰权重
//      w_q = square_weight    ：方波干扰权重
//      d   = delay (samples)  ：目标延迟，对应往返距离 R = c*d/(2*f_s)
//      A_n = noise_amplitude  ：噪声幅度
//
//  【物理意义】
//  模拟雷达接收机中，发射脉冲经目标反射（带延迟和衰减）后，与多种干扰信号
//  和环境噪声共同叠加形成的接收信号。这是 T2-R1（波形生成）条款的核心实现，
//  输出 echo 将作为 Step2 匹配滤波/脉冲压缩的输入。>
StepEvidence run_step1(PipelineState& state);

}  // namespace task2

```

### 3.2 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
#include "step1.h"

#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
#include "waveforms/waveforms_typed.h"

#include <cmath>
#include <numeric>

namespace task2 {
namespace {

// <学习注释：GPU kernel —— 回波信号叠加混合（mix_echo_kernel）。
//
//  【函数总体处理方法】
//  每个 CUDA 线程独立处理一个采样点 n，并行执行以下四步：
//    步骤1：计算全局线程索引 index = blockIdx.x * blockDim.x + threadIdx.x
//    步骤2：生成确定性伪随机噪声 noise_value（基于 PCG-LCG + Murmur-like avalanche）
//    步骤3：计算目标回波分量 target_value（带延迟的 chirp 信号）
//    步骤4：计算干扰分量 interference_value（高斯/锯齿/方波三种波形加权叠加）
//    步骤5：将三个分量分别写入输出数组，并叠加为 echo
//
//  【核心数学公式 —— 逐采样点回波模型】
//    对线程索引 n = index（n = 0, 1, ..., N-1）：
//
//    (a) 确定性噪声生成（PCG 混合 + Murmur-like 比特扩散）：
//        value = seed ⊕ (n × 747796405 + 2891336453)   [模 2^32 无符号运算]
//        value = finalize(value)                        [连续 × / ⊕ / >> 完成 avalanche]
//        noise_value = ( (value & 0xFFFF) / 32767.5 − 1.0 ) × noise_amplitude
//        → 将 16 位无符号整数 [0, 65535] 线性映射到 [−A_n, +A_n]
//
//    (b) 目标回波分量（延迟 chirp）：
//        target_value = { w_t × chirp[n − d]   if n ≥ d
//                       { 0.0                  if n < d
//        → 物理意义：雷达发射脉冲经距离 R 处目标反射，往返延迟 d = 2R·f_s/c 个样本
//          权重 w_t 模拟目标雷达截面积（RCS）和路径衰减
//
//    (c) 干扰分量（多波形加权叠加）：
//        interference_value = w_g × gausspulse[n] + w_s × sawtooth[n] + w_q × square[n]
//        → 物理意义：模拟多干扰源同时存在的复杂电磁环境
//
//    (d) 完整回波叠加（线性叠加原理）：
//        echo[n] = target_value + interference_value + noise_value
//        → 物理意义：雷达接收机前端收到的总信号 = 目标反射 + 干扰 + 噪声的线性叠加
//
//  【四种波形的物理意义】
//    - Chirp（线性调频脉冲）：脉压雷达的标准发射波形，频率随时间线性变化
//      f(t) = f_0 + β·t，兼顾距离分辨率与最大作用距离
//    - Gausspulse（高斯包络调制正弦脉冲）：s(t) = exp(−a·t²)·cos(2πf_c·t)
//      高斯包络具有最优时频集中性（Δf·Δt 最小），模拟窄带干扰或通信信号
//    - Sawtooth（锯齿波）：分段线性周期波形，含全部整数次谐波
//      模拟宽带扫频干扰源，如 FMCW 雷达中的互干扰
//    - Square（方波）：周期二电平信号，占空比决定高/低电平比例
//      模拟开关电源、数字时钟等周期性脉冲干扰
//
//  【并行策略】
//  使用一维 grid/block 启动，每个线程处理一个采样点。线程数向上取整到 block 的
//  整数倍，通过 if (index >= count) return 做越界保护。>
__global__ void mix_echo_kernel(
    const float* chirp, const float* gaussian, const float* sawtooth,
    const float* square,
    float* target, float* interference, float* noise, float* echo,
    int count, int delay, std::uint32_t seed, float target_weight,
    float gaussian_weight, float sawtooth_weight, float square_weight,
    float noise_amplitude)
{
    // <学习注释：计算全局一维线程索引。
    //  index = blockIdx.x * blockDim.x + threadIdx.x
    //  将 2D CUDA 启动参数（grid 中 block 数 + block 中线程数）展平为全局 ID。
    //  threadIdx.x ∈ [0, blockDim.x-1]：当前线程在 block 内的编号。
    //  blockIdx.x  ∈ [0, gridDim.x-1] ：当前 block 在 grid 中的编号。
    //  blockDim.x：每个 block 中的线程总数（通常为 256 或 512）。>
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // <学习注释：越界保护——当线程索引超出有效采样范围时直接返回。
    //  grid 启动的线程数向上取整到 block 的整数倍，可能超过实际 count。
    //  不执行此检查会导致越界访问 GPU 显存，产生未定义行为。>
    if (index >= count) return;
    // <学习注释：伪随机噪声种子混合——第1步：PCG 风格的 LCG 初始混合。
    //  value = seed ⊕ (index × 747796405 + 2891336453)  [无符号 32 位，模 2^32]
    //  ⊕ 是按位异或（C++ 中的 ^），不是乘方。
    //  747796405U = 0x2C9277B5：PCG 常用 32 位 LCG 乘数。
    //  2891336453U = 0xAC564B05：配套增量，防止 seed/index 零值直接保持零。
    //  目的：让相邻 index 不直接形成相邻整数状态，为后续 avalanche 做准备。>
    std::uint32_t value = seed ^
        (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);
    // <学习注释：Murmur-like 比特扩散（avalanche）——第2步：移位+异或。
    //  value ^= value >> 16：将高位 16 比特折叠到低位，通过异或混合。
    //  右移 16 位（无符号）：高位补 0，低位丢弃。>
    value ^= value >> 16;
    // <学习注释：avalanche 第3步：乘奇数扩散。
    //  value *= 2246822519U = 0x85EBCA77：奇数乘数，模 2^32 下可逆。
    //  乘法进位效应将低位变化传播到高位。>
    value *= 2246822519U;
    // <学习注释：avalanche 第4步：移位+异或，右移 13 位混合。
    //  16,13,16 的移位次序与 MurmurHash3 fmix32 相同，但乘数有微小差异。>
    value ^= value >> 13;
    // <学习注释：avalanche 第5步：乘奇数再次扩散。
    //  value *= 3266489917U = 0xC2B2AE3D：另一个奇数乘数。>
    value *= 3266489917U;
    // <学习注释：avalanche 第6步（最后一步）：移位+异或完成最终混合。
    //  经过 6 步操作后，value 的每个比特都受到 index 和 seed 所有比特的影响。>
    value ^= value >> 16;
    // <学习注释：将 16 位随机整数映射为 [-1, 1] 范围的浮点噪声样本。
    //  value & 0xFFFF：取低 16 位，得到 [0, 65535] 的整数。
    //  static_cast<float>(...)：转换为单精度浮点。
    //  / 32767.5F：除以 16 位范围的中点 65535/2 = 32767.5，得到约 [0, 2]。
    //  - 1.0F：平移使范围中心为 0，得到约 [-1, +1]。
    //  * noise_amplitude：缩放到目标噪声幅度 A_n。
    //  最终 noise_value ∈ [-A_n, +A_n]，均值为 0。
    //  数学公式：noise_value = A_n × ( (v & 0xFFFF) / 32767.5 − 1 )>
    const float noise_value =
        (static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;
    // <学习注释：目标回波分量——带延迟的 chirp 信号。
    //  三元运算符 ?: 语法：条件 ? 真值表达式 : 假值表达式。
    //  index >= delay：当样本索引达到或超过延迟时才产生回波。
    //  chirp[index - delay]：读取 delay 个样本之前的 chirp 值。
    //  target_weight * chirp[index - delay]：乘以幅度权重 w_t。
    //  物理意义：电磁波往返距离 R 对应的离散延迟 d = 2R·f_s/c。
    //  数学公式：target[n] = w_t × chirp[n − d]  （n ≥ d）；0（n < d）>
    const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;
    // <学习注释：干扰分量——三种干扰波形加权叠加。
    //  gaussian_weight * gaussian[index]：高斯脉冲干扰贡献。
    //  sawtooth_weight * sawtooth[index]：锯齿波干扰贡献。
    //  square_weight * square[index]：方波干扰贡献。
    //  加法从左到右结合（相同优先级），两次 + 运算产生三项之和。
    //  数学公式：interference[n] = w_g × g[n] + w_s × s[n] + w_q × q[n]>
    const float interference_value =
        gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] +
        square_weight * square[index];
    // <学习注释：将三个分量分别写入输出数组，供后续可视化/分析使用。
    //  target[index]       = target_value        —— 目标回波分量（纯 chirp 延迟版）
    //  interference[index] = interference_value  —— 干扰分量（三种波形加权叠加）
    //  noise[index]        = noise_value         —— 噪声分量（确定性伪随机）>
    target[index] = target_value;
    interference[index] = interference_value;
    noise[index] = noise_value;
    // <学习注释：完整回波信号叠加。
    //  echo[n] = target_value + interference_value + noise_value
    //  这是线性叠加原理的直接体现：接收机总信号 = 各独立信号源贡献之和。
    //  echo 数组将作为 Step2 匹配滤波（脉冲压缩）的输入。>
    echo[index] = target_value + interference_value + noise_value;
}

}  // namespace

StepEvidence run_step1(PipelineState& state)
{
    // <学习注释：获取只读配置引用，避免拷贝。
    //  config 包含所有外部参数：samples（采样数 N）、sample_rate_hz（采样率 f_s）、
    //  target_delay_samples（目标延迟 d）、noise_seed（噪声种子）、
    //  各波形权重（target_weight、gaussian_weight、sawtooth_weight、square_weight）、
    //  noise_amplitude（噪声幅度 A_n）。>
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step1";
    // <学习注释：记录总耗时起点，用于最终 evidence.total_ms。>
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    // <学习注释：阶段1（CPU）—— 在 Host 端计算时间轴和相位数组。
    //  时间轴：time[n] = n / f_s，其中 n = 0, 1, ..., N-1。
    //  物理意义：将离散样本索引 n 转换为物理时间 t = n·Δt，Δt = 1/f_s 为采样间隔。>
    state.time.resize(config.samples);
    std::vector<float> pulse_time(config.samples);
    std::vector<float> phase(config.samples);
    std::vector<float> square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        // <学习注释：time[n] = n / f_s —— 离散采样时刻（单位：秒）。>
        state.time[index] = static_cast<float>(index) / config.sample_rate_hz;
        // <学习注释：pulse_time[n] = time[n] - 0.45 —— 脉冲时间轴偏移。
        //  减去 0.45s 使高斯脉冲峰值对准 t=0 附近，而非采样起始。>
        pulse_time[index] = state.time[index] - 0.45F;
        // <学习注释：saw_cycles = (190.0 × n / f_s) mod 1 —— 锯齿波归一化周期。
        //  fmod(x, 1.0) 提取小数部分，得到 [0, 1) 范围的归一化相位。
        //  190.0 是锯齿波的频率（Hz），除以 f_s 得到每样本的周期增量。>
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        // <学习注释：square_cycles = (125.0 × n / f_s) mod 1 —— 方波归一化周期。
        //  125.0 是方波的频率（Hz）。>
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        // <学习注释：phase[n] = 2π × saw_cycles —— 锯齿波弧度相位。
        //  将 [0, 1) 的归一化周期映射到 [0, 2π) 弧度。
        //  数学公式：θ_saw[n] = 2π × (190.0 × n / f_s mod 1)>
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
        // <学习注释：square_phase[n] = 2π × square_cycles —— 方波弧度相位。
        //  数学公式：θ_sq[n] = 2π × (125.0 × n / f_s mod 1)>
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    // <学习注释：记录 CPU 准备阶段的耗时（毫秒）。>
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：阶段2（H2D）—— 将时间轴和相位数组从 Host 拷贝到 GPU 显存。
    //  DeviceArray::from_host 在 GPU 上分配显存并完成 H2D 传输。>
    begin = Clock::now();
    auto d_time = cusignal::DeviceArray<float>::from_host(state.time);
    auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);
    auto d_phase = cusignal::DeviceArray<float>::from_host(phase);
    auto d_square_phase = cusignal::DeviceArray<float>::from_host(square_phase);
    evidence.h2d_ms += milliseconds(begin, Clock::now());

    // <学习注释：阶段3（GPU 波形生成）—— 调用四个算子 kernel 在 GPU 上并行生成波形。
    //  每个算子都在 GPU 上独立执行，分配输出数组（DeviceArray 构造时分配显存）。>
    cusignal::DeviceArray<float> d_chirp(config.samples);
    cusignal::DeviceArray<float> d_gaussian(config.samples);
    cusignal::DeviceArray<double> d_sawtooth(config.samples);
    cusignal::DeviceArray<double> d_square(config.samples);
    // <学习注释：chirp_device —— 生成线性调频脉冲（Chirp）。
    //  参数：f0=20.0Hz（起始频率），t1=T（终止时刻），f1=70.0Hz（终止频率），phi=0.0°（初相位）。
    //  频率随时间线性变化：f(t) = 20 + (70-20)/T × t = 20 + 50·t/T Hz。
    //  数学公式：chirp(t) = cos(2π × ∫₀ᵗ f(τ)dτ) = cos(2π(20t + 25t²/T))。
    //  物理意义：模拟脉压雷达的发射脉冲，兼顾距离分辨率和作用距离。>
    evidence.operator_ms["chirp"] = time_gpu([&] {
        cusignal::chirp_device(
            d_time, d_chirp, 20.0F,
            std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),
            70.0F, 0.0F);
    });
    // <学习注释：gausspulse_device —— 生成高斯包络调制正弦脉冲。
    //  参数：fc=170.0Hz（中心频率），bw=0.20（相对带宽，即 20%）。
    //  绝对带宽 Δf = bw × fc = 0.20 × 170 = 34 Hz。
    //  高斯包络：a = -(π × bw × fc)² / (4·ln(10^(bwr/20)))，bwr=-6dB 时 a ≈ 4130。
    //  数学公式：gausspulse(t) = exp(−a·t²) × cos(2π·fc·t)。
    //  物理意义：高斯包络具有最优时频集中性，模拟窄带干扰或通信信号。>
    evidence.operator_ms["gausspulse"] = time_gpu([&] {
        cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F);
    });
    // <学习注释：sawtooth_device —— 生成锯齿波。
    //  参数：width=0.65（上升斜坡占周期 65%）。
    //  在一个 2π 周期内：前 65% 线性上升（−1→+1），后 35% 线性下降（+1→−1）。
    //  数学公式（分段线性）：
    //    y(t) = t_mod/(π·w) − 1           [0 ≤ t_mod < 2πw，上升段]
    //    y(t) = (π(w+1) − t_mod)/(π(1−w))  [2πw ≤ t_mod < 2π，下降段]
    //  其中 t_mod = t mod 2π，w = 0.65。
    //  物理意义：含全部整数次谐波，模拟宽带扫频干扰。>
    evidence.operator_ms["sawtooth"] = time_gpu([&] {
        cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65);
    });
    // <学习注释：square_device —— 生成方波（矩形波）。
    //  参数：duty=0.35（占空比 35%，高电平占周期的 35%）。
    //  在一个 2π 周期内：前 35% 输出 +1，后 65% 输出 −1。
    //  数学公式：
    //    y(t) = +1  [0 ≤ t_mod < 2π·D]
    //    y(t) = −1  [2π·D ≤ t_mod < 2π]
    //  其中 D = 0.35，t_mod = t mod 2π。
    //  物理意义：模拟开关电源、数字时钟等周期性脉冲干扰。>
    evidence.operator_ms["square"] = time_gpu([&] {
        cusignal::square_device(d_square_phase, d_square, 0.35);
    });

    // <学习注释：阶段3.5（D2H+D2H）—— 将锯齿波和方波从 GPU 传回 Host 做 fp64→fp32 窄化。
    //  sawtooth_device 和 square_device 输出 double 精度，需转回 float 再上传 GPU。>
    begin = Clock::now();
    state.target_waveform = d_chirp.to_host();
    const auto gaussian = d_gaussian.to_host();
    const auto sawtooth_double = d_sawtooth.to_host();
    const auto square_double = d_square.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    // <学习注释：窄化 fp64→fp32：double 精度（15-17 位有效数字）→ float（7 位）。
    //  在 Host 端执行，避免 GPU kernel 内的 double→float 类型转换开销。>
    const auto sawtooth =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double);
    const auto square =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double);
    begin = Clock::now();
    auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);
    auto d_square_float = cusignal::DeviceArray<float>::from_host(square);
    evidence.h2d_ms += milliseconds(begin, Clock::now());

    // <学习注释：阶段4（GPU 回波叠加）—— 分配目标/干扰/噪声/回波输出数组。
    //  每个数组大小 = config.samples（N 个 float 元素）。>
    cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);
    cusignal::DeviceArray<float> d_noise(config.samples), d_echo(config.samples);
    // <学习注释：记录当前 GPU 显存占用峰值，用于性能分析。>
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    // <学习注释：启动 mix_echo_kernel GPU kernel。
    //  launch_1d_kernel 自动计算 grid/block 配置，确保覆盖所有 N 个采样点。
    //  传入参数：四个波形数组（chirp/gaussian/sawtooth/square）、
    //  四个输出数组（target/interference/noise/echo）、
    //  以及 count=N、delay=d、noise_seed、五个权重和噪声幅度。
    //  GPU 上每个线程并行执行 mix_echo_kernel 的完整计算流程。>
    evidence.operator_ms["gpu_echo_superposition"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            mix_echo_kernel, static_cast<std::size_t>(config.samples),
            d_chirp.data(), d_gaussian.data(), d_saw_float.data(), d_square_float.data(),
            d_target.data(), d_interference.data(), d_noise.data(), d_echo.data(),
            config.samples, config.target_delay_samples, config.noise_seed,
            config.target_weight, config.gaussian_weight, config.sawtooth_weight,
            config.square_weight, config.noise_amplitude);
    });
    // <学习注释：汇总所有算子耗时（chirp + gausspulse + sawtooth + square + echo_superposition）。>
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });

    // <学习注释：阶段5（D2H）—— 将 GPU 计算结果传回 Host。
    //  target_component：目标回波分量（纯 chirp 延迟版）
    //  interference_component：干扰分量（三种波形加权叠加）
    //  noise_component：噪声分量（确定性伪随机）
    //  echo：完整回波信号（三者叠加，将作为 Step2 输入）>
    begin = Clock::now();
    state.target_component = d_target.to_host();
    state.interference_component = d_interference.to_host();
    state.noise_component = d_noise.to_host();
    state.echo = d_echo.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    // <学习注释：记录总耗时（从 total_begin 到此刻）。>
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    // <学习注释：可视化捕获：如果启用，保存中间波形供后续绘图分析。
    //  gaussian/sawtooth/square 波形数组仅在可视化模式下保留。>
    if (visualization_capture_enabled()) {
        state.gaussian_waveform = gaussian;
        state.sawtooth_waveform = sawtooth;
        state.square_waveform = square;
    }
    return evidence;
}

}  // namespace task2
```

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：CPU 参考实现 —— 确定性噪声比特生成（noise_bits）。
//  与 GPU kernel 中 mix_echo_kernel 的噪声生成逻辑完全一致，
//  用于在 CPU 端独立验证 GPU 结果的正确性。
//  数学公式：value = finalize(seed ⊕ (index × 747796405 + 2891336453))
//  其中 finalize 是 6 步 Murmur-like avalanche 操作。>
std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    // <学习注释：PCG 风格 LCG 初始混合：seed ⊕ (index × 747796405 + 2891336453)。
    //  ⊕ 是按位异或（^），不是乘方。747796405U 和 2891336453U 是 PCG 常用常数。>
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
    // <学习注释：avalanche 步骤1：value ^= value >> 16（高位折叠到低位）。>
    value ^= value >> 16;
    // <学习注释：avalanche 步骤2：value *= 2246822519U（奇数乘数扩散）。>
    value *= 2246822519U;
    // <学习注释：avalanche 步骤3：value ^= value >> 13。>
    value ^= value >> 13;
    // <学习注释：avalanche 步骤4：value *= 3266489917U。>
    value *= 3266489917U;
    // <学习注释：avalanche 步骤5（最后）：value ^= value >> 16，返回最终混合值。>
    return value ^ (value >> 16);
}

// <学习注释：CPU 参考实现 —— 确定性噪声样本生成（noise_sample）。
//  将 16 位随机整数映射为 [-amplitude, +amplitude] 的浮点噪声样本。
//  数学公式：noise = amplitude × ( (bits & 0xFFFF) / 32767.5 − 1.0 )
//  与 GPU kernel 中 noise_value 的计算公式完全相同，保证 CPU/GPU 可复现。>
float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * amplitude;
}

// <学习注释：CPU 参考实现 —— 回波信号合成（synthesize_cpu）。
//  这是 mix_echo_kernel 的 CPU 等价实现，逐采样点串行计算。
//  与 GPU 版本使用相同的确定性噪声生成，用于精度验证。
//
//  【函数总体处理方法】
//  对每个采样点 n（n = 0, 1, ..., N-1）串行执行：
//    1. target[n]       = w_t × chirp[n − d]  （n ≥ d 时）
//    2. interference[n] = w_g × g[n] + w_s × s[n] + w_q × q[n]
//    3. noise[n]        = noise_sample(n, seed, noise_amplitude)
//    4. echo[n]         = target[n] + interference[n] + noise[n]
//
//  【核心数学公式 —— 回波信号叠加模型（与 GPU 版本一致）】
//    echo[n] = w_t × chirp[n − d]  （目标回波，n ≥ d）
//            + w_g × gausspulse[n] + w_s × sawtooth[n] + w_q × square[n]  （干扰）
//            + A_n × rand(n)  （噪声）
//
//  【与 GPU 版本的差异】
//  - CPU 版本串行 for 循环，GPU 版本每个线程并行处理一个 n
//  - CPU 版本直接操作 std::vector，GPU 版本操作 device 指针
//  - 数学公式和噪声生成算法完全相同，保证精度验证的有效性>
std::vector<float> synthesize_cpu(
    const std::vector<float>& chirp, const std::vector<float>& gaussian,
    const std::vector<float>& sawtooth, const std::vector<float>& square,
    std::uint32_t seed, float target_weight, std::vector<float>* target,
    std::vector<float>* interference, std::vector<float>* noise,
    const TaskConfig& config)
{
    // <学习注释：初始化 echo 输出数组，大小与 chirp 相同（N 个元素）。>
    std::vector<float> echo(chirp.size());
    // <学习注释：将 target/interference/noise 数组清零，全部填充为 0.0F。>
    target->assign(chirp.size(), 0.0F);
    interference->assign(chirp.size(), 0.0F);
    noise->assign(chirp.size(), 0.0F);
    // <学习注释：逐采样点串行计算回波信号分量。
    //  与 GPU 版本每线程并行处理不同，CPU 版本使用 for 循环逐一处理。>
    for (std::size_t index = 0; index < chirp.size(); ++index) {
        // <学习注释：目标回波分量：带延迟的 chirp 信号。
        //  index >= delay：只有达到延迟后才产生回波（模拟电磁波往返时间）。
        //  target_weight × chirp[index − delay]：延迟 chirp 乘以幅度权重 w_t。>
        (*target)[index] = index >= static_cast<std::size_t>(config.target_delay_samples)
            ? target_weight * chirp[index - config.target_delay_samples] : 0.0F;
        // <学习注释：干扰分量：三种波形加权叠加。
        //  config.gaussian_weight × gaussian[index]：高斯脉冲干扰贡献。
        //  config.sawtooth_weight × sawtooth[index]：锯齿波干扰贡献。
        //  config.square_weight × square[index]：方波干扰贡献。>
        (*interference)[index] = config.gaussian_weight * gaussian[index] +
            config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index];
        // <学习注释：噪声分量：确定性伪随机噪声。
        //  noise_sample 调用 noise_bits 生成与 GPU 相同的伪随机序列。>
        (*noise)[index] = noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude);
        // <学习注释：回波信号叠加：echo[n] = target[n] + interference[n] + noise[n]。
        //  线性叠加原理：接收机总信号 = 各独立信号源贡献之和。>
        echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];
    }
    return echo;
}

}  // namespace

// <学习注释：CPU 参考实现 —— 完整 Step1 流水线（generate_step1_cpu_reference）。
//  这是 run_step1 的 CPU 等价实现，用于生成 GPU 精度验证的基准数据。
//
//  【函数总体处理方法】
//  阶段1：计算时间轴和相位数组（与 GPU 版本相同）
//  阶段2：调用四个算子的 CPU 版本生成波形
//         - chirp_typed_cpu：线性调频脉冲（f0=20→70Hz，线性扫频）
//         - gausspulse_typed_cpu：高斯调制脉冲（fc=170Hz，bw=20%）
//         - sawtooth_typed_cpu：锯齿波（width=0.65，fp64 输出）
//         - square_typed_cpu：方波（duty=0.35，fp64 输出）
//  阶段3：fp64→fp32 窄化（sawtooth/square 的 double→float 转换）
//  阶段4：调用 synthesize_cpu 完成回波信号叠加
//  阶段5：生成扰动对照组（seed+1，target_weight−0.05），用于验证
//         参数变化对输出的影响是否可检测
//
//  【与 GPU 版本的差异】
//  - 使用 CPU 版本的算子函数（_typed_cpu 后缀），而非 GPU 版本（_device 后缀）
//  - 串行执行，不使用 CUDA kernel
//  - 直接输出 std::vector<float>，不经过 DeviceArray
//  - 额外生成 changed_echo 用于验证参数敏感性>
Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)
{
    Step1ValidationData data;
    // <学习注释：分配时间轴和相位数组（与 GPU 版本相同的计算逻辑）。>
    std::vector<float> time(config.samples), pulse_time(config.samples);
    std::vector<float> phase(config.samples), square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        // <学习注释：time[n] = n / f_s —— 离散采样时刻。>
        time[index] = static_cast<float>(index) / config.sample_rate_hz;
        // <学习注释：pulse_time[n] = time[n] − 0.45 —— 脉冲时间轴偏移。>
        pulse_time[index] = time[index] - 0.45F;
        // <学习注释：saw_cycles = (190.0 × n / f_s) mod 1 —— 锯齿波归一化周期。>
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        // <学习注释：square_cycles = (125.0 × n / f_s) mod 1 —— 方波归一化周期。>
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        // <学习注释：phase[n] = 2π × saw_cycles —— 锯齿波弧度相位。>
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
        // <学习注释：square_phase[n] = 2π × square_cycles —— 方波弧度相位。>
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    // <学习注释：调用 CPU 版本的 chirp 算子。
    //  参数：time 时间轴、f0=20Hz、t1=T、f1=70Hz、phi=0°。
    //  数学公式：chirp(t) = cos(2π(20t + 25t²/T))。>
    data.chirp_cpu = cusignal::chirp_typed_cpu(
        time, 20.0F,
        std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),
        70.0F, 0.0F);
    // <学习注释：调用 CPU 版本的高斯脉冲算子。
    //  参数：pulse_time（偏移后的时间轴）、fc=170Hz、bw=0.20。
    //  数学公式：gausspulse(t) = exp(−a·t²) × cos(2π·fc·t)，a = -(π·bw·fc)²/(4·ln(10^(bwr/20)))，bwr=-6dB 时 a ≈ 4130。>
    const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(
        pulse_time, 170.0F, 0.20F);
    // <学习注释：调用 CPU 版本的锯齿波算子（fp64 输出），然后窄化为 fp32。
    //  参数：phase（弧度相位）、width=0.65。
    //  数学公式：y(t) = t_mod/(π·w) − 1（上升段）；y(t) = (π(w+1)−t_mod)/(π(1−w))（下降段）。>
    const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::sawtooth_typed_cpu(phase, 0.65));
    // <学习注释：调用 CPU 版本的方波算子（fp64 输出），然后窄化为 fp32。
    //  参数：square_phase（弧度相位）、duty=0.35。
    //  数学公式：y(t) = +1（0 ≤ t_mod < 2πD）；y(t) = −1（2πD ≤ t_mod < 2π）。>
    const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::square_typed_cpu(square_phase, 0.35));
    // <学习注释：调用 synthesize_cpu 完成回波信号叠加。
    //  传入：四个波形数组、noise_seed、target_weight、三个输出数组指针、config。
    //  返回 echo_cpu 作为 GPU 精度验证的基准。>
    data.echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed,
        config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config);
    // <学习注释：生成扰动对照组——改变 seed 和 target_weight 以验证参数敏感性。
    //  seed + 1：噪声序列完全不同，验证噪声生成的可复现性和敏感性。
    //  target_weight − 0.05：目标回波幅度略微减小，验证幅度检测能力。
    //  如果 changed_echo 与 echo 的差异可检测，则证明参数变化可被系统感知。>
    std::vector<float> changed_target, changed_interference, changed_noise;
    data.changed_echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U,
        std::max(0.0F, config.target_weight - 0.05F),
        &changed_target, &changed_interference, &changed_noise, config);
    return data;
}
```

### 3.4 函数总体处理方法

本节从方案设计的角度，概述 Step1 中所有函数的总体处理方法和它们之间的协作关系。

#### 3.4.1 总体流水线架构

Step1 的完整信号生成流水线由以下函数协作完成：

```
                    ┌─────────────────────────────────────────────────┐
                    │         generate_step1_cpu_reference            │
                    │  (CPU 参考实现，用于精度验证)                    │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  chirp_typed_cpu / gausspulse_typed_cpu │   │
                    │  │  sawtooth_typed_cpu / square_typed_cpu  │   │
                    │  │         ↓ (四个波形)                      │   │
                    │  │      synthesize_cpu                      │   │
                    │  │  (目标 + 干扰 + 噪声 → echo)              │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                         ↑ 精度对比验证
                    ┌─────────────────────────────────────────────────┐
                    │              run_step1 (GPU 实现)               │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │  chirp_device / gausspulse_device        │   │
                    │  │  sawtooth_device / square_device         │   │
                    │  │         ↓ (四个波形)                      │   │
                    │  │      mix_echo_kernel                     │   │
                    │  │  (目标 + 干扰 + 噪声 → echo)              │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### 3.4.2 各函数处理方法一览

| 函数 | 类型 | 职责 | 核心数学公式 | 并行策略 |
|------|------|------|-------------|----------|
| `run_step1` | GPU 入口 | 编排整个流水线：时间轴计算 → H2D → 波形生成 → 回波叠加 → D2H → 证据收集 | $echo[n] = \sum(目标 + 干扰 + 噪声)$ | 分阶段串行，各阶段内 GPU 并行 |
| `mix_echo_kernel` | GPU kernel | 每个线程并行处理一个采样点，计算目标回波、干扰、噪声和叠加 | $echo[n] = w_t·chirp[n-d] + \sum w_i·wave_i[n] + A_n·rand(n)$ | 一维 grid/block，每线程一个采样点 |
| `synthesize_cpu` | CPU 参考 | 串行等价实现，逐采样点计算回波分量，用于 GPU 精度验证 | 与 `mix_echo_kernel` 完全一致 | 串行 for 循环 |
| `generate_step1_cpu_reference` | CPU 参考入口 | 编排 CPU 流水线，生成基准数据和扰动对照组 | 与 `run_step1` 等价 | 分阶段串行 |
| `noise_bits` | CPU/GPU 共用 | 确定性伪随机数生成（PCG + Murmur-like avalanche） | $value = finalize(seed \oplus (n \times 747796405 + 2891336453))$ | 每采样点独立计算 |
| `noise_sample` | CPU 共用 | 将 16 位随机整数映射为 $[-A, +A]$ 浮点噪声 | $noise = A \times ((bits \& 0xFFFF) / 32767.5 - 1.0)$ | 每采样点独立计算 |

#### 3.4.3 四种信号生成函数的数学与物理模型

##### (a) Chirp（线性调频脉冲）

- **数学公式**：$x(t) = \cos\left(2\pi\int_0^t f(\tau)d\tau + \phi_0\right)$
- **本次参数**：$f_0 = 20\text{ Hz}$，$f_1 = 70\text{ Hz}$，线性扫频 $f(t) = 20 + 50t/T$
- **积分结果**：$x(t) = \cos(2\pi(20t + 25t^2/T))$
- **物理意义**：脉压雷达的标准发射波形，通过频率调制将长脉冲的能量分散到宽带中，接收端匹配滤波后压缩为窄脉冲，同时获得高距离分辨率和高信噪比
- **调用方式**：`chirp_device(d_time, d_chirp, 20.0F, T, 70.0F, 0.0F)`

##### (b) Gausspulse（高斯包络调制正弦脉冲）

- **数学公式**：$s(t) = \exp(-a t^2) \cdot \cos(2\pi f_c t)$，其中 $a = -\dfrac{(\pi \cdot f_c \cdot bw)^2}{4\ln(10^{bwr/20})}$，$bwr$ 为带宽参考电平（dB），默认 $-6$ dB
- **本次参数**：$f_c = 170\text{ Hz}$，$bw = 0.20$（20% 相对带宽）
- **绝对带宽**：$\Delta f = bw \times f_c = 34\text{ Hz}$
- **高斯衰减系数**：$bwr=-6$ dB 时，$reference=10^{-0.3}\approx 0.501$，$4\ln(reference)\approx -2.763$，$a = -(\pi \times 0.20 \times 170)^2 / (-2.763) \approx 4130$
- **物理意义**：高斯包络具有最优时频集中性（唯一使 $\Delta f \cdot \Delta t = 1/(4\pi)$ 的包络形状），模拟窄带干扰或通信信号；脉冲时间轴偏移 −0.45s 使峰值对准采样窗口中心
- **调用方式**：`gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F)`

##### (c) Sawtooth（锯齿波）

- **数学公式**（分段线性，周期 $2\pi$）：
  $$y(t) = \begin{cases} \dfrac{t_{\text{mod}}}{\pi w} - 1, & 0 \le t_{\text{mod}} < 2\pi w \quad\text{（上升段）} \\[8pt] \dfrac{\pi(w+1) - t_{\text{mod}}}{\pi(1-w)}, & 2\pi w \le t_{\text{mod}} < 2\pi \quad\text{（下降段）} \end{cases}$$
- **本次参数**：$w = 0.65$（上升斜坡占 65%），频率 $f = 190\text{ Hz}$
- **物理意义**：含全部整数次谐波（$f, 2f, 3f, \ldots$），频谱丰富；模拟 FMCW 雷达中的互干扰或宽带扫频干扰源
- **调用方式**：`sawtooth_device(d_phase, d_sawtooth, 0.65)`

##### (d) Square（方波/矩形波）

- **数学公式**（周期 $2\pi$，占空比 $D$）：
  $$y(t) = \begin{cases} +1, & 0 \le t_{\text{mod}} < 2\pi D \\ -1, & 2\pi D \le t_{\text{mod}} < 2\pi \end{cases}$$
- **本次参数**：$D = 0.35$（35% 占空比），频率 $f = 125\text{ Hz}$
- **物理意义**：模拟开关电源（SMPS）、数字时钟电路等产生的周期性脉冲干扰；占空比 35% 为非对称矩形波，含奇次和偶次谐波
- **调用方式**：`square_device(d_square_phase, d_square, 0.35)`

#### 3.4.4 回波信号叠加模型的雷达物理含义

Step1 的核心数学模型是雷达接收信号的线性叠加：

$$echo[n] = \underbrace{w_t \cdot chirp[n-d]}_{\text{目标回波}} + \underbrace{\sum_{k} w_k \cdot interfer_k[n]}_{\text{干扰分量}} + \underbrace{A_n \cdot rand(n)}_{\text{噪声分量}}$$

| 分量 | 物理来源 | 实现方式 | 输出用途 |
|------|---------|---------|---------|
| 目标回波 | 发射脉冲经目标反射 | chirp 延迟 $d$ 个样本并乘以权重 $w_t$ | Step2 匹配滤波目标检测 |
| 干扰分量 | 其他辐射源、反射体 | 三种波形加权叠加 | Step2 干扰抑制验证 |
| 噪声分量 | 接收机热噪声、环境噪声 | 确定性伪随机序列 | 信噪比评估 |
| 完整回波 | 接收机前端总信号 | 三者线性叠加 | Step2 脉冲压缩输入 |

**目标延迟 $d$ 与物理距离 $R$ 的关系**：
$$d = \frac{2R \cdot f_s}{c}$$

其中 $c \approx 3 \times 10^8\text{ m/s}$（光速），$f_s$ 为采样率。延迟 $d$ 个样本意味着电磁波往返 $2R$ 距离所需的时间。

#### 3.4.5 CPU/GPU 精度验证策略

| 验证维度 | CPU 实现 | GPU 实现 | 验证方法 |
|---------|---------|---------|---------|
| 波形生成 | `*_typed_cpu` 函数 | `*_device` 函数 | 逐元素比较输出数组 |
| 噪声生成 | `noise_bits` + `noise_sample` | `mix_echo_kernel` 内联 | 相同 seed/index 产生相同值 |
| 回波叠加 | `synthesize_cpu` 串行循环 | `mix_echo_kernel` 并行 kernel | 逐元素比较 echo 数组 |
| 参数敏感性 | 扰动对照组（seed+1, weight−0.05） | 与 GPU 对照组比较 | 验证差异可检测 |

---

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once

#include "../task2_common.h"
```

**语法结构**

这个代码块由预处理指令构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 0 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `pragma`、`once`、`include`、`task2_common`、`h`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `#pragma once`：`#pragma once` 要求编译器在同一翻译单元中只展开一次本头文件，避免重复声明或定义。
- 对源码锚点 `#include "../task2_common.h"`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step1(PipelineState& state);
```

**语法结构**

这个代码块由函数或方法定义、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step1`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `namespace task2 {`：`namespace` 建立命名空间 `task2`，把后续符号放进独立名称范围，调用者通常用 `task2::符号` 访问；末尾 `{` 同时打开该范围。
- 对源码锚点 `StepEvidence run_step1(PipelineState& state);`：函数调用语法：`run_step1(...)` 先准备括号内实参，再把控制权交给 `run_step1`，返回值回到调用点。 运算符：`&`：按位与；声明上下文中的单个 & 也可表示引用。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.3 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

```cpp

}  // namespace task2
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `}  // namespace task2`：运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `}  // namespace task2` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `#include "step1.h"`。

```cpp
#include "step1.h"

#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
#include "waveforms/waveforms_typed.h"
```

**语法结构**

这个代码块由预处理指令构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 0 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step1`、`h`、`cuda_utils`、`device_array`、`kernel_launch`、`waveforms`、`waveforms_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `#include "step1.h"`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。
- 对源码锚点 `#include "cuda_utils/device_array.h"`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。
- 对源码锚点 `#include "cuda_utils/kernel_launch.h"`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。
- 对源码锚点 `#include "waveforms/waveforms_typed.h"`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 若 `/` 两侧都是整数，C++ 执行截断的整数除法；要得到小数必须先把至少一侧转换为浮点类型；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `#include <cmath>`。

```cpp
#include <cmath>
#include <numeric>
```

**语法结构**

这个代码块由预处理指令构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 0 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`cmath`、`numeric`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `#include <cmath>`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。
- 对源码锚点 `#include <numeric>`：`#include` 是预处理指令，在编译前把尖括号中的系统头或双引号中的项目头展开到当前翻译单元；它本身不执行运行时算法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `#include <cmath>` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.6 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {
namespace {
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 0 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `namespace task2 {`：`namespace` 建立命名空间 `task2`，把后续符号放进独立名称范围，调用者通常用 `task2::符号` 访问；末尾 `{` 同时打开该范围。
- 对源码锚点 `namespace {`：空名称的匿名命名空间把其中符号限制在当前 C++ 翻译单元，避免与其他 `.cpp/.cu` 文件发生链接名称冲突；`{` 打开该范围。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.7 mix_echo_kernel：声明 CUDA kernel 并建立线程入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `__global__ void mix_echo_kernel(`。

```cpp
__global__ void mix_echo_kernel(
    const float* chirp, const float* gaussian, const float* sawtooth,
    const float* square,
    float* target, float* interference, float* noise, float* echo,
    int count, int delay, std::uint32_t seed, float target_weight,
    float gaussian_weight, float sawtooth_weight, float square_weight,
    float noise_amplitude)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
```

**语法结构**

这个代码块由函数或方法定义、变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `chirp` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `gaussian` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `sawtooth` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `square` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `target` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `interference` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `noise` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `echo` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `count` 由 `int` 声明：有符号整数类型；具体位宽由平台决定，本项目常见平台通常为 32 位；
- `delay` 由 `int` 声明：有符号整数类型；具体位宽由平台决定，本项目常见平台通常为 32 位；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `target_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `gaussian_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `sawtooth_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `square_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `noise_amplitude` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `index` 由 `const int` 声明：有符号整数类型；具体位宽由平台决定，本项目常见平台通常为 32 位；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`chirp`|当前函数或调用的参数名称，接收调用者绑定的输入 `chirp`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float* chirp, const float* gaussian, const float* sawtooth,`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`gaussian`|当前函数或调用的参数名称，接收调用者绑定的输入 `gaussian`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float* chirp, const float* gaussian, const float* sawtooth,`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`sawtooth`|当前函数或调用的参数名称，接收调用者绑定的输入 `sawtooth`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float* chirp, const float* gaussian, const float* sawtooth,`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`square`|当前函数或调用的参数名称，接收调用者绑定的输入 `square`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float* square,`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`target`|当前函数或调用的参数名称，接收调用者绑定的输入 `target`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `float* target, float* interference, float* noise, float* echo,`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`interference`|当前函数或调用的参数名称，接收调用者绑定的输入 `interference`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `float* target, float* interference, float* noise, float* echo,`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`delay`|目标离散延迟，单位 sample|记作 $d$|回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`target_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`gaussian_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`sawtooth_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`square_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`noise_amplitude`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `__global__ void mix_echo_kernel( const float* chirp, const float* gaussian, const float* sawtooth, const float* square, float* target, float* interference, float* noise, float* ...`：这是函数定义头而不是一次调用。`mix_echo_kernel` 是新函数名；它前面的 `__global__ void` 包含返回类型及可能的 CUDA 限定符。括号中的 ` const float* chirp, const float* gaussian, const float* sawtooth, const float* square, float* target, float* interference, float* noise, float* echo, int count, int delay, std::uint32_t seed, float target_weight, float gaussian_weight, float sawtooth_weight, float square_weight, float noise_amplitude` 是形参声明：调用者传入实参后按顺序绑定到这些局部名称；没有 `&` 或 `*` 的标量参数按值复制。末尾 `{` 打开函数体，因此此处只建立可执行代码的定义，真正计算要等调用发生。
- 对源码锚点 `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);`：先按括号和运算符优先级形成右侧 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 的值，再用它初始化或赋给 `const int index`。关键字语法：`const`：const 禁止通过当前名字修改对象；`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 CUDA 内建索引由运行时为每个线程提供；常见展平编号 `blockIdx.x * blockDim.x + threadIdx.x` 把块号、每块线程数和块内线程号合成全局线程号。 执行后，后续代码读取 `const int index` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`、`echo`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 使用 `blockIdx.x * blockDim.x + threadIdx.x` 是为了给每个一维输出元素分配唯一全局线程。这样同一 kernel 可适配不同 block 大小；随后必须用 count 做越界保护，因为 grid 往往向上取整。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.8 mix_echo_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `if (index >= count) return;`。

```cpp
    if (index >= count) return;
    std::uint32_t value = seed ^
        (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);
```

**语法结构**

这个代码块由变量声明与初始化、条件分支、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `value` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `747796405U` 是数值字面量；U 指定 unsigned int 起始类型；
- `2891336453U` 是数值字面量；U 指定 unsigned int 起始类型。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`747796405U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x2C9277B5`；这是 PCG 系列常用 32 位 LCG 乘数。此处用于让相邻 index 在进入后续 avalanche 前先分散；它不是物理常数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19） 当前上下文：在 `(static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2891336453U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xAC564B05`；这是与上述乘数配套使用的 32 位增量。此处平移状态，避免 seed/index 的零值直接保持零；不是雷达参数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19） 当前上下文：在 `(static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `if (index >= count) return;`：先按括号和运算符优先级形成右侧 `count) return` 的值，再用它初始化或赋给 `if (index >`。关键字语法：`if`：if 先把括号内表达式转换为真假，仅为真时进入分支；`return`：return 结束当前函数，并把表达式结果交给调用者。 运算符：`>=`：大于或等于比较。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `if (index >` 时得到的就是这次写入的新值。
- 对源码锚点 `std::uint32_t value = seed ^ (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);`：先按括号和运算符优先级形成右侧 `seed ^ (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U)` 的值，再用它初始化或赋给 `std::uint32_t value`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`747796405U` 是数值字面量；U 指定 unsigned int 起始类型；`2891336453U` 是数值字面量；U 指定 unsigned int 起始类型。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`^`：按位异或：把两个整数的同一二进制位比较，相同得 0、不同得 1；它不是乘方；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 本表达式的实际求值顺序是：先算括号中的乘法，再做加法，最后与括号外整数按位异或。参与运算的 `U` 常量会促使通常算术转换采用无符号类型；无符号 N 位结果按模 $2^N$ 回绕。这里把样本 `index`、随机 `seed` 和两个混合常量扩散成确定性的伪随机比特，后续噪声生成可在 CPU 与 GPU 上复现同一输入。 执行后，后续代码读取 `std::uint32_t value` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 先执行乘加再与 seed 组合，是为了让相邻 `index` 不直接形成相邻整数状态，并避免零输入停留在简单零模式；这一步采用 PCG 常见 LCG 常数作为低成本初始打散，后续再由 Murmur 风格 avalanche 完成扩散。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）。

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- C/C++/Python 中 `^` 是按位异或，不是乘方；乘方在 Python 写作 `**`，C++ 通常调用 `std::pow`；
- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.9 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `value ^= value >> 16;`。

```cpp
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 0 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2246822519U` 是数值字面量；U 指定 unsigned int 起始类型；
- `13` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `value ^= value >> 16;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2246822519U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x85EBCA77`。它是奇数，因此在模 $2^{32}$ 算术中可逆并能借助乘法进位扩散比特。它接近但不等于 MurmurHash3 `fmix32` 的 `0x85EBCA6B`；仓库源码未记录为何相差 `0x0C`，不能冒充原始 Murmur 常数。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：它直接参与表达式 `value *= 2246822519U;`，作用由同一表达式中的运算符决定。|
|`13`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`13` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `value ^= value >> 13;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `value ^= value >> 16;`：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`>>`：右移二进制位；无符号整数高位补 0；`^=`：复合按位异或赋值：先把左侧旧值与右侧逐位异或，再把结果写回同一对象。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `value *= 2246822519U;`：读取 `value` 的旧值，与 `2246822519U` 执行 `*` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`2246822519U` 是数值字面量；U 指定 unsigned int 起始类型。 运算符：`*=`：复合乘法赋值。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `value ^= value >> 13;`：读取 `value` 的旧值，与 `value >> 13` 执行 `^` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`13` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`>>`：右移二进制位；无符号整数高位补 0；`^=`：复合按位异或赋值：先把左侧旧值与右侧逐位异或，再把结果写回同一对象。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- C/C++/Python 中 `^` 是按位异或，不是乘方；乘方在 Python 写作 `**`，C++ 通常调用 `std::pow`；
- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断。

### 4.10 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `value *= 3266489917U;`。

```cpp
    value *= 3266489917U;
    value ^= value >> 16;
    const float noise_value =
        (static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;
```

**语法结构**

这个代码块由变量声明与初始化构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `noise_value` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `3266489917U` 是数值字面量；U 指定 unsigned int 起始类型；
- `16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0xffffU` 是数值字面量；U 指定 unsigned int 起始类型；
- `32767.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`noise_value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`3266489917U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xC2B2AE3D`。它同样是用于继续扩散的奇数乘数，但不等于 MurmurHash3 `fmix32` 的 `0xC2B2AE35`；仓库源码未记录为何相差 `0x08`。改变它会改变全部确定性噪声序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：它直接参与表达式 `value *= 3266489917U;`，作用由同一表达式中的运算符决定。|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `value ^= value >> 16;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`0xffffU`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。 当前上下文：在 `(static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`32767.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。 当前上下文：在 `(static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。 当前上下文：在 `(static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `value *= 3266489917U;`：读取 `value` 的旧值，与 `3266489917U` 执行 `*` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`3266489917U` 是数值字面量；U 指定 unsigned int 起始类型。 运算符：`*=`：复合乘法赋值。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `value ^= value >> 16;`：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`>>`：右移二进制位；无符号整数高位补 0；`^=`：复合按位异或赋值：先把左侧旧值与右侧逐位异或，再把结果写回同一对象。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `const float noise_value = (static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;`：先按括号和运算符优先级形成右侧 `(static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude` 的值，再用它初始化或赋给 `const float noise_value`。关键字语法：`const`：const 禁止通过当前名字修改对象；`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`0xffffU` 是数值字面量；U 指定 unsigned int 起始类型；`32767.5F` 是数值字面量；F 指定 float，而非默认 double；`1.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`-`：减法；位于单个操作数前时是一元负号；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const float noise_value` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 先用 `0xFFFF` 只保留低 16 位，是为了得到固定的 65536 个离散状态；随后除以半跨度 `32767.5` 并减 1，把它线性映射为以 0 为中心的 $[-1,1]$。最后乘 `noise_std` 才得到任务噪声幅度。可替代为 32 位到浮点的完整映射或标准 RNG，但那会改变历史 CPU/GPU 确定性序列和精度证据。

**初学者易错点**

- C/C++/Python 中 `^` 是按位异或，不是乘方；乘方在 Python 写作 `**`，C++ 通常调用 `std::pow`；
- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断。

### 4.11 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;`。

```cpp
    const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;
    const float interference_value =
        gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] +
        square_weight * square[index];
```

**语法结构**

这个代码块由变量声明与初始化构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `target_value` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `interference_value` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`target_value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`interference_value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;` 中，它为 `target_value` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;`：先按括号和运算符优先级形成右侧 `index >= delay ? target_weight * chirp[index - delay] : 0.0F` 的值，再用它初始化或赋给 `const float target_value`。关键字语法：`const`：const 禁止通过当前名字修改对象。 字面量与类型：`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`>=`：大于或等于比较；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`-`：减法；位于单个操作数前时是一元负号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员；`?`：条件运算符的起点，先判断条件，只计算冒号两侧中的一支；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const float target_value` 时得到的就是这次写入的新值。
- 对源码锚点 `const float interference_value = gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] + square_weight * square[index];`：先按括号和运算符优先级形成右侧 `gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] + square_weight * square[index]` 的值，再用它初始化或赋给 `const float interference_value`。关键字语法：`const`：const 禁止通过当前名字修改对象。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const float interference_value` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断。

### 4.12 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `target[index] = target_value;`。

```cpp
    target[index] = target_value;
    interference[index] = interference_value;
    noise[index] = noise_value;
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `target`、`index`、`target_value`、`interference`、`interference_value`、`noise`、`noise_value`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `target[index] = target_value;`：先按括号和运算符优先级形成右侧 `target_value` 的值，再用它初始化或赋给 `target[index]`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `target[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `interference[index] = interference_value;`：先按括号和运算符优先级形成右侧 `interference_value` 的值，再用它初始化或赋给 `interference[index]`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `interference[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `noise[index] = noise_value;`：先按括号和运算符优先级形成右侧 `noise_value` 的值，再用它初始化或赋给 `noise[index]`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `noise[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `target[index] = target_value;` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.13 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `echo[index] = target_value + interference_value + noise_value;`。

```cpp
    echo[index] = target_value + interference_value + noise_value;
}
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 1 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `echo`、`index`、`target_value`、`interference_value`、`noise_value`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `echo[index] = target_value + interference_value + noise_value;`：先按括号和运算符优先级形成右侧 `target_value + interference_value + noise_value` 的值，再用它初始化或赋给 `echo[index]`。运算符：`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `echo[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `echo[index] = target_value + interference_value + noise_value;` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.14 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `}  // namespace`。

```cpp

}  // namespace
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `}  // namespace`：运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `}  // namespace` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.15 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `StepEvidence run_step1(PipelineState& state)`。

```cpp
StepEvidence run_step1(PipelineState& state)
{
    const auto& config = state.config;
```

**语法结构**

这个代码块由函数或方法定义、变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `config` 由 `const auto&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step1`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `StepEvidence run_step1(PipelineState& state) {`：这是函数定义头而不是一次调用。`run_step1` 是新函数名；它前面的 `StepEvidence` 包含返回类型及可能的 CUDA 限定符。括号中的 `PipelineState& state` 是形参声明：调用者传入实参后按顺序绑定到这些局部名称；没有 `&` 或 `*` 的标量参数按值复制。末尾 `{` 打开函数体，因此此处只建立可执行代码的定义，真正计算要等调用发生。
- 对源码锚点 `const auto& config = state.config;`：先按括号和运算符优先级形成右侧 `state.config` 的值，再用它初始化或赋给 `const auto& config`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 运算符：`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto& config` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `config`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.16 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step1";
    const auto total_begin = Clock::now();
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 1 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

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

- 对源码锚点 `StepEvidence evidence;`：这是 C++ 局部变量声明：`StepEvidence` 是静态类型，`evidence` 是当前作用域中的变量名。没有显式初始化器；类类型会调用默认构造函数，而内置标量若不是静态存储期则值未确定，读取前必须先赋值。分号结束声明。
- 对源码锚点 `evidence.name = "step1";`：先按括号和运算符优先级形成右侧 `"step1"` 的值，再用它初始化或赋给 `evidence.name`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 执行后，后续代码读取 `evidence.name` 时得到的就是这次写入的新值。
- 对源码锚点 `const auto total_begin = Clock::now();`：先按括号和运算符优先级形成右侧 `Clock::now()` 的值，再用它初始化或赋给 `const auto total_begin`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto total_begin` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step1` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.17 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    state.time.resize(config.samples);
    std::vector<float> pulse_time(config.samples);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `auto begin = Clock::now();`：先按括号和运算符优先级形成右侧 `Clock::now()` 的值，再用它初始化或赋给 `auto begin`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto begin` 时得到的就是这次写入的新值。
- 对源码锚点 `state.time.resize(config.samples);`：函数调用语法：`state.time.resize(...)` 先准备括号内实参，再把控制权交给 `state.time.resize`，返回值回到调用点。 运算符：`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `std::vector<float> pulse_time(config.samples);`：函数调用语法：`pulse_time(...)` 先准备括号内实参，再把控制权交给 `pulse_time`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `time`、`samples`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.18 phase：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `std::vector<float> phase(config.samples);`。

```cpp
    std::vector<float> phase(config.samples);
    std::vector<float> square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        state.time[index] = static_cast<float>(index) / config.sample_rate_hz;
```

**语法结构**

这个代码块由变量声明与初始化、循环、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 6 个开放分隔符与 5 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `index` 由 `int` 声明：有符号整数类型；具体位宽由平台决定，本项目常见平台通常为 32 位；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `for (int index = 0; index < config.samples; ++index) {` 中，它参与迭代起点、终点或步长，直接决定循环执行次数。|

**执行过程**

- 对源码锚点 `std::vector<float> phase(config.samples);`：函数调用语法：`phase(...)` 先准备括号内实参，再把控制权交给 `phase`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `std::vector<float> square_phase(config.samples);`：函数调用语法：`square_phase(...)` 先准备括号内实参，再把控制权交给 `square_phase`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `for (int index = 0; index < config.samples; ++index) {`：先按括号和运算符优先级形成右侧 `0; index < config.samples; ++index) {` 的值，再用它初始化或赋给 `for (int index`。关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 字面量与类型：`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`++`：自增 1；前置形式先增后取值，后置形式先取旧值后增；`<`：小于比较；模板实参列表中也可作为左尖括号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `for (int index` 时得到的就是这次写入的新值。
- 对源码锚点 `state.time[index] = static_cast<float>(index) / config.sample_rate_hz;`：先按括号和运算符优先级形成右侧 `static_cast<float>(index) / config.sample_rate_hz` 的值，再用它初始化或赋给 `state.time[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `state.time[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`、`samples`、`time`、`sample_rate_hz`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.19 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `pulse_time[index] = state.time[index] - 0.45F;`。

```cpp
        pulse_time[index] = state.time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `saw_cycles` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `0.45F` 是数值字面量；F 指定 float，而非默认 double；
- `190.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`saw_cycles`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `saw_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const double saw_cycles = std::fmod(`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.45F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `pulse_time[index] = state.time[index] - 0.45F;`，作用由同一表达式中的运算符决定。|
|`190.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `pulse_time[index] = state.time[index] - 0.45F;`：先按括号和运算符优先级形成右侧 `state.time[index] - 0.45F` 的值，再用它初始化或赋给 `pulse_time[index]`。字面量与类型：`0.45F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`-`：减法；位于单个操作数前时是一元负号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `pulse_time[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `const double saw_cycles = std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`：先按括号和运算符优先级形成右侧 `std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 的值，再用它初始化或赋给 `const double saw_cycles`。关键字语法：`const`：const 禁止通过当前名字修改对象；`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`std::fmod(...)` 先准备括号内实参，再把控制权交给 `std::fmod`，返回值回到调用点。 字面量与类型：`190.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const double saw_cycles` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `time`、`sample_rate_hz`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.20 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `const double square_cycles = std::fmod(`。

```cpp
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `square_cycles` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `saw_cycles` 由 `kPi *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `125.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square_cycles`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `square_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const double square_cycles = std::fmod(`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`saw_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `saw_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`125.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `const double square_cycles = std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`：先按括号和运算符优先级形成右侧 `std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 的值，再用它初始化或赋给 `const double square_cycles`。关键字语法：`const`：const 禁止通过当前名字修改对象；`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`std::fmod(...)` 先准备括号内实参，再把控制权交给 `std::fmod`，返回值回到调用点。 字面量与类型：`125.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const double square_cycles` 时得到的就是这次写入的新值。
- 对源码锚点 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);`：先按括号和运算符优先级形成右侧 `static_cast<float>(2.0 * kPi * saw_cycles)` 的值，再用它初始化或赋给 `phase[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `phase[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`、`sample_rate_hz`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.21 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`。

```cpp
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 5 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `square_cycles` 由 `kPi *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`square_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `square_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`：先按括号和运算符优先级形成右侧 `static_cast<float>(2.0 * kPi * square_cycles)` 的值，再用它初始化或赋给 `square_phase[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `square_phase[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。
- 对源码锚点 `evidence.prep_ms = milliseconds(begin, Clock::now());`：先按括号和运算符优先级形成右侧 `milliseconds(begin, Clock::now())` 的值，再用它初始化或赋给 `evidence.prep_ms`。函数调用语法：`milliseconds(...)` 先准备括号内实参，再把控制权交给 `milliseconds`，返回值回到调用点；`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `evidence.prep_ms` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `phase` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.22 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    auto d_time = cusignal::DeviceArray<float>::from_host(state.time);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `d_time` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `begin = Clock::now();`：先按括号和运算符优先级形成右侧 `Clock::now()` 的值，再用它初始化或赋给 `begin`。函数调用语法：`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `begin` 时得到的就是这次写入的新值。
- 对源码锚点 `auto d_time = cusignal::DeviceArray<float>::from_host(state.time);`：先按括号和运算符优先级形成右侧 `cusignal::DeviceArray<float>::from_host(state.time)` 的值，再用它初始化或赋给 `auto d_time`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`from_host(...)` 先准备括号内实参，再把控制权交给 `from_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto d_time` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `time`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.23 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);`。

```cpp
    auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);
    auto d_phase = cusignal::DeviceArray<float>::from_host(phase);
    auto d_square_phase = cusignal::DeviceArray<float>::from_host(square_phase);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `d_pulse_time` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_phase` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_square_phase` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_pulse_time`|当前慢时间脉冲编号|记作 $p=\lfloor i/N_s\rfloor$|由展平索引整除每脉冲采样数得到|
|`d_phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`d_square_phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);`：先按括号和运算符优先级形成右侧 `cusignal::DeviceArray<float>::from_host(pulse_time)` 的值，再用它初始化或赋给 `auto d_pulse_time`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`from_host(...)` 先准备括号内实参，再把控制权交给 `from_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto d_pulse_time` 时得到的就是这次写入的新值。
- 对源码锚点 `auto d_phase = cusignal::DeviceArray<float>::from_host(phase);`：先按括号和运算符优先级形成右侧 `cusignal::DeviceArray<float>::from_host(phase)` 的值，再用它初始化或赋给 `auto d_phase`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`from_host(...)` 先准备括号内实参，再把控制权交给 `from_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto d_phase` 时得到的就是这次写入的新值。
- 对源码锚点 `auto d_square_phase = cusignal::DeviceArray<float>::from_host(square_phase);`：先按括号和运算符优先级形成右侧 `cusignal::DeviceArray<float>::from_host(square_phase)` 的值，再用它初始化或赋给 `auto d_square_phase`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`from_host(...)` 先准备括号内实参，再把控制权交给 `from_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto d_square_phase` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.24 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `evidence.h2d_ms += milliseconds(begin, Clock::now());`。

```cpp
    evidence.h2d_ms += milliseconds(begin, Clock::now());
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`h2d_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.h2d_ms += milliseconds(begin, Clock::now());`：读取 `evidence.h2d_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 对应运算，再把结果写回同一个 `evidence.h2d_ms` 对象。函数调用语法：`milliseconds(...)` 先准备括号内实参，再把控制权交给 `milliseconds`，返回值回到调用点；`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`+=`：复合加法赋值，等价于在只求值一次左侧地址的前提下执行左侧 = 左侧 + 右侧；`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `evidence.h2d_ms` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `phase` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.25 d_chirp：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_chirp`；源码锚点 `cusignal::DeviceArray<float> d_chirp(config.samples);`。

```cpp

    cusignal::DeviceArray<float> d_chirp(config.samples);
    cusignal::DeviceArray<float> d_gaussian(config.samples);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_chirp`、`config`、`samples`、`d_gaussian`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_chirp`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`d_gaussian`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `cusignal::DeviceArray<float> d_chirp(config.samples);`：函数调用语法：`d_chirp(...)` 先准备括号内实参，再把控制权交给 `d_chirp`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `cusignal::DeviceArray<float> d_gaussian(config.samples);`：函数调用语法：`d_gaussian(...)` 先准备括号内实参，再把控制权交给 `d_gaussian`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `samples`。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.26 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `cusignal::DeviceArray<double> d_sawtooth(config.samples);`。

```cpp
    cusignal::DeviceArray<double> d_sawtooth(config.samples);
    cusignal::DeviceArray<double> d_square(config.samples);
    evidence.operator_ms["chirp"] = time_gpu([&] {
        cusignal::chirp_device(
            d_time, d_chirp, 20.0F,
            std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),
            70.0F, 0.0F);
    });
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 9 个开放分隔符与 9 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `20.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `70.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_sawtooth`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`d_square`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`20.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `d_time, d_chirp, 20.0F,`，作用由同一表达式中的运算符决定。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`70.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `70.0F, 0.0F);`，作用由同一表达式中的运算符决定。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `d_time, d_chirp, 20.0F,`，作用由同一表达式中的运算符决定；它直接参与表达式 `70.0F, 0.0F);`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `cusignal::DeviceArray<double> d_sawtooth(config.samples);`：函数调用语法：`d_sawtooth(...)` 先准备括号内实参，再把控制权交给 `d_sawtooth`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `cusignal::DeviceArray<double> d_square(config.samples);`：函数调用语法：`d_square(...)` 先准备括号内实参，再把控制权交给 `d_square`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `evidence.operator_ms["chirp"] = time_gpu([&] { cusignal::chirp_device( d_time, d_chirp, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F,...`：先按括号和运算符优先级形成右侧 `time_gpu([&] { cusignal::chirp_device( d_time, d_chirp, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F, 0.0F); })` 的值，再用它初始化或赋给 `evidence.operator_ms["chirp"]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`time_gpu(...)` 先准备括号内实参，再把控制权交给 `time_gpu`，返回值回到调用点；`cusignal::chirp_device(...)` 先准备括号内实参，再把控制权交给 `cusignal::chirp_device`，返回值回到调用点；`std::max(...)` 先准备括号内实参，再把控制权交给 `std::max`，返回值回到调用点。 字面量与类型：`20.0F` 是数值字面量；F 指定 float，而非默认 double；`1.0F` 是数值字面量；F 指定 float，而非默认 double；`70.0F` 是数值字面量；F 指定 float，而非默认 double；`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `evidence.operator_ms["chirp"]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`samples`、`sample_rate_hz`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.27 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.operator_ms["gausspulse"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["gausspulse"] = time_gpu([&] {
        cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F);
    });
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 5 个开放分隔符与 5 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `170.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.20F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`170.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F);` 中，它作为 `cusignal::gausspulse_device` 的实参参与当前调用。|
|`0.20F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F);` 中，它作为 `cusignal::gausspulse_device` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `evidence.operator_ms["gausspulse"] = time_gpu([&] { cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F); });`：先按括号和运算符优先级形成右侧 `time_gpu([&] { cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F); })` 的值，再用它初始化或赋给 `evidence.operator_ms["gausspulse"]`。函数调用语法：`time_gpu(...)` 先准备括号内实参，再把控制权交给 `time_gpu`，返回值回到调用点；`cusignal::gausspulse_device(...)` 先准备括号内实参，再把控制权交给 `cusignal::gausspulse_device`，返回值回到调用点。 字面量与类型：`170.0F` 是数值字面量；F 指定 float，而非默认 double；`0.20F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `evidence.operator_ms["gausspulse"]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `gausspulse`。 本块直接出现 1 种波形符号：`gausspulse`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.28 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.operator_ms["sawtooth"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["sawtooth"] = time_gpu([&] {
        cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65);
    });
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 5 个开放分隔符与 5 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `0.65` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.65`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65);` 中，它作为 `cusignal::sawtooth_device` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `evidence.operator_ms["sawtooth"] = time_gpu([&] { cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65); });`：先按括号和运算符优先级形成右侧 `time_gpu([&] { cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65); })` 的值，再用它初始化或赋给 `evidence.operator_ms["sawtooth"]`。函数调用语法：`time_gpu(...)` 先准备括号内实参，再把控制权交给 `time_gpu`，返回值回到调用点；`cusignal::sawtooth_device(...)` 先准备括号内实参，再把控制权交给 `cusignal::sawtooth_device`，返回值回到调用点。 字面量与类型：`0.65` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `evidence.operator_ms["sawtooth"]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`。 本块直接出现 1 种波形符号：`sawtooth`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.29 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.operator_ms["square"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["square"] = time_gpu([&] {
        cusignal::square_device(d_square_phase, d_square, 0.35);
    });
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 5 个开放分隔符与 5 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.35`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `cusignal::square_device(d_square_phase, d_square, 0.35);` 中，它作为 `cusignal::square_device` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `evidence.operator_ms["square"] = time_gpu([&] { cusignal::square_device(d_square_phase, d_square, 0.35); });`：先按括号和运算符优先级形成右侧 `time_gpu([&] { cusignal::square_device(d_square_phase, d_square, 0.35); })` 的值，再用它初始化或赋给 `evidence.operator_ms["square"]`。函数调用语法：`time_gpu(...)` 先准备括号内实参，再把控制权交给 `time_gpu`，返回值回到调用点；`cusignal::square_device(...)` 先准备括号内实参，再把控制权交给 `cusignal::square_device`，返回值回到调用点。 字面量与类型：`0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `evidence.operator_ms["square"]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.30 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    state.target_waveform = d_chirp.to_host();
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`target_waveform`、`d_chirp`、`to_host`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `begin = Clock::now();`：先按括号和运算符优先级形成右侧 `Clock::now()` 的值，再用它初始化或赋给 `begin`。函数调用语法：`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `begin` 时得到的就是这次写入的新值。
- 对源码锚点 `state.target_waveform = d_chirp.to_host();`：先按括号和运算符优先级形成右侧 `d_chirp.to_host()` 的值，再用它初始化或赋给 `state.target_waveform`。函数调用语法：`d_chirp.to_host(...)` 先准备括号内实参，再把控制权交给 `d_chirp.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `state.target_waveform` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `target_waveform`。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.31 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `const auto gaussian = d_gaussian.to_host();`。

```cpp
    const auto gaussian = d_gaussian.to_host();
    const auto sawtooth_double = d_sawtooth.to_host();
    const auto square_double = d_square.to_host();
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `gaussian` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `sawtooth_double` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `square_double` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`gaussian`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `gaussian`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto gaussian = d_gaussian.to_host();`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`sawtooth_double`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `sawtooth_double`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto sawtooth_double = d_sawtooth.to_host();`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`square_double`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `square_double`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto square_double = d_square.to_host();`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `const auto gaussian = d_gaussian.to_host();`：先按括号和运算符优先级形成右侧 `d_gaussian.to_host()` 的值，再用它初始化或赋给 `const auto gaussian`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`d_gaussian.to_host(...)` 先准备括号内实参，再把控制权交给 `d_gaussian.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto gaussian` 时得到的就是这次写入的新值。
- 对源码锚点 `const auto sawtooth_double = d_sawtooth.to_host();`：先按括号和运算符优先级形成右侧 `d_sawtooth.to_host()` 的值，再用它初始化或赋给 `const auto sawtooth_double`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`d_sawtooth.to_host(...)` 先准备括号内实参，再把控制权交给 `d_sawtooth.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto sawtooth_double` 时得到的就是这次写入的新值。
- 对源码锚点 `const auto square_double = d_square.to_host();`：先按括号和运算符优先级形成右侧 `d_square.to_host()` 的值，再用它初始化或赋给 `const auto square_double`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`d_square.to_host(...)` 先准备括号内实参，再把控制权交给 `d_square.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto square_double` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`square`。 本块直接出现 2 种波形符号：`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.32 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.d2h_ms += milliseconds(begin, Clock::now());`。

```cpp
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    const auto sawtooth =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `sawtooth` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`sawtooth`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `sawtooth`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto sawtooth =`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.d2h_ms += milliseconds(begin, Clock::now());`：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 对应运算，再把结果写回同一个 `evidence.d2h_ms` 对象。函数调用语法：`milliseconds(...)` 先准备括号内实参，再把控制权交给 `milliseconds`，返回值回到调用点；`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`+=`：复合加法赋值，等价于在只求值一次左侧地址的前提下执行左侧 = 左侧 + 右侧；`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `evidence.d2h_ms` 时得到的就是这次写入的新值。
- 对源码锚点 `const auto sawtooth = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double);`：先按括号和运算符优先级形成右侧 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double)` 的值，再用它初始化或赋给 `const auto sawtooth`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(...)` 先准备括号内实参，再把控制权交给 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto sawtooth` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`。 本块直接出现 1 种波形符号：`sawtooth`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.33 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `const auto square =`。

```cpp
    const auto square =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double);
    begin = Clock::now();
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `square` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `square`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto square =`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `const auto square = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double);`：先按括号和运算符优先级形成右侧 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double)` 的值，再用它初始化或赋给 `const auto square`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(...)` 先准备括号内实参，再把控制权交给 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto square` 时得到的就是这次写入的新值。
- 对源码锚点 `begin = Clock::now();`：先按括号和运算符优先级形成右侧 `Clock::now()` 的值，再用它初始化或赋给 `begin`。函数调用语法：`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `begin` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.34 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);`。

```cpp
    auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);
    auto d_square_float = cusignal::DeviceArray<float>::from_host(square);
    evidence.h2d_ms += milliseconds(begin, Clock::now());
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `d_saw_float` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `d_square_float` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_saw_float`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`d_square_float`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);`：先按括号和运算符优先级形成右侧 `cusignal::DeviceArray<float>::from_host(sawtooth)` 的值，再用它初始化或赋给 `auto d_saw_float`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`from_host(...)` 先准备括号内实参，再把控制权交给 `from_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto d_saw_float` 时得到的就是这次写入的新值。
- 对源码锚点 `auto d_square_float = cusignal::DeviceArray<float>::from_host(square);`：先按括号和运算符优先级形成右侧 `cusignal::DeviceArray<float>::from_host(square)` 的值，再用它初始化或赋给 `auto d_square_float`。关键字语法：`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`from_host(...)` 先准备括号内实参，再把控制权交给 `from_host`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `auto d_square_float` 时得到的就是这次写入的新值。
- 对源码锚点 `evidence.h2d_ms += milliseconds(begin, Clock::now());`：读取 `evidence.h2d_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 对应运算，再把结果写回同一个 `evidence.h2d_ms` 对象。函数调用语法：`milliseconds(...)` 先准备括号内实参，再把控制权交给 `milliseconds`，返回值回到调用点；`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`+=`：复合加法赋值，等价于在只求值一次左侧地址的前提下执行左侧 = 左侧 + 右侧；`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `evidence.h2d_ms` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`square`。 本块直接出现 2 种波形符号：`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.35 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);`。

```cpp

    cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);
    cusignal::DeviceArray<float> d_noise(config.samples), d_echo(config.samples);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_target`、`config`、`samples`、`d_interference`、`d_noise`、`d_echo`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_target`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`d_noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);`：函数调用语法：`d_target(...)` 先准备括号内实参，再把控制权交给 `d_target`，返回值回到调用点；`d_interference(...)` 先准备括号内实参，再把控制权交给 `d_interference`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `cusignal::DeviceArray<float> d_noise(config.samples), d_echo(config.samples);`：函数调用语法：`d_noise(...)` 先准备括号内实参，再把控制权交给 `d_noise`，返回值回到调用点；`d_echo(...)` 先准备括号内实参，再把控制权交给 `d_echo`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `samples`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.36 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

```cpp
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

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
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `std::size_t free_bytes = 0, total_bytes = 0;` 中，它为 `free_bytes` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`：先按括号和运算符优先级形成右侧 `0, total_bytes = 0` 的值，再用它初始化或赋给 `std::size_t free_bytes`。字面量与类型：`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `std::size_t free_bytes` 时得到的就是这次写入的新值。
- 对源码锚点 `CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));`：函数调用语法：`CUDA_CHECK(...)` 先准备括号内实参，再把控制权交给 `CUDA_CHECK`，返回值回到调用点；`cudaMemGetInfo(...)` 先准备括号内实参，再把控制权交给 `cudaMemGetInfo`，返回值回到调用点。 运算符：`&`：按位与；声明上下文中的单个 & 也可表示引用。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);`：先按括号和运算符优先级形成右侧 `static_cast<double>(total_bytes - free_bytes)` 的值，再用它初始化或赋给 `evidence.metrics["gpu_used_peak_bytes"]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 运算符：`-`：减法；位于单个操作数前时是一元负号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `evidence.metrics["gpu_used_peak_bytes"]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.37 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.operator_ms["gpu_echo_superposition"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["gpu_echo_superposition"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            mix_echo_kernel, static_cast<std::size_t>(config.samples),
            d_chirp.data(), d_gaussian.data(), d_saw_float.data(), d_square_float.data(),
            d_target.data(), d_interference.data(), d_noise.data(), d_echo.data(),
            config.samples, config.target_delay_samples, config.noise_seed,
            config.target_weight, config.gaussian_weight, config.sawtooth_weight,
            config.square_weight, config.noise_amplitude);
    });
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 14 个开放分隔符与 14 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`gpu_echo_superposition`、`time_gpu`、`cusignal`、`cuda_utils`、`launch_1d_kernel`、`mix_echo_kernel`、`std`、`size_t`、`config`、`samples`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.operator_ms["gpu_echo_superposition"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( mix_echo_kernel, static_cast<std::size_t>(config.samples), d_chirp.data()...`：先按括号和运算符优先级形成右侧 `time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( mix_echo_kernel, static_cast<std::size_t>(config.samples), d_chirp.data(), d_gaussian.data(), d_saw_float.data(), d_square_float.data(), d_target.data(), d_interference.data(), d_noise.data(), d_echo.data(), config.samples, config.target_delay_samples, config.noise_seed, config.target_weight, config.gaussian_weight, config.sawtooth_weight, config.square_weight, config.noise_amplitude); })` 的值，再用它初始化或赋给 `evidence.operator_ms["gpu_echo_superposition"]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`time_gpu(...)` 先准备括号内实参，再把控制权交给 `time_gpu`，返回值回到调用点；`cusignal::cuda_utils::launch_1d_kernel(...)` 先准备括号内实参，再把控制权交给 `cusignal::cuda_utils::launch_1d_kernel`，返回值回到调用点；`d_chirp.data(...)` 先准备括号内实参，再把控制权交给 `d_chirp.data`，返回值回到调用点；`d_gaussian.data(...)` 先准备括号内实参，再把控制权交给 `d_gaussian.data`，返回值回到调用点；`d_saw_float.data(...)` 先准备括号内实参，再把控制权交给 `d_saw_float.data`，返回值回到调用点；`d_square_float.data(...)` 先准备括号内实参，再把控制权交给 `d_square_float.data`，返回值回到调用点；`d_target.data(...)` 先准备括号内实参，再把控制权交给 `d_target.data`，返回值回到调用点；`d_interference.data(...)` 先准备括号内实参，再把控制权交给 `d_interference.data`，返回值回到调用点；`d_noise.data(...)` 先准备括号内实参，再把控制权交给 `d_noise.data`，返回值回到调用点；`d_echo.data(...)` 先准备括号内实参，再把控制权交给 `d_echo.data`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `evidence.operator_ms["gpu_echo_superposition"]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`square`、`samples`、`target_delay_samples`、`noise_seed`、`target_weight`、`gaussian_weight`、`sawtooth_weight`、`square_weight`、`noise_amplitude`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 引号内文本是字符串数据而不是变量名或可执行表达式；只有格式化字符串占位、转义序列或后续解析器会赋予其中字符额外含义。

### 4.38 d_target：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.compute_ms = std::accumulate(`。

```cpp
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
```

**语法结构**

这个代码块由变量声明与初始化、lambda、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 6 个开放分隔符与 6 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `sum` 由 `double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；
- `item` 由 `const auto&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`sum`|循环或归约的累加器|通常记作 $S=\sum_i a_i$|每轮读取旧值并累加当前 item|
|`item`|当前循环/归约元素|记作 $a_i$|由容器迭代器产生并加入 sum|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,` 中，它作为 `evidence.operator_ms.begin` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `evidence.compute_ms = std::accumulate( evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0, [](double sum, const auto& item) { return sum + item.second; });`：先按括号和运算符优先级形成右侧 `std::accumulate( evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0, [](double sum, const auto& item) { return sum + item.second; })` 的值，再用它初始化或赋给 `evidence.compute_ms`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型；`return`：return 结束当前函数，并把表达式结果交给调用者。 函数调用语法：`std::accumulate(...)` 先准备括号内实参，再把控制权交给 `std::accumulate`，返回值回到调用点；`evidence.operator_ms.begin(...)` 先准备括号内实参，再把控制权交给 `evidence.operator_ms.begin`，返回值回到调用点；`evidence.operator_ms.end(...)` 先准备括号内实参，再把控制权交给 `evidence.operator_ms.end`，返回值回到调用点。 字面量与类型：`0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`+`：加法；也可能是一元正号；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `evidence.compute_ms` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.39 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    state.target_component = d_target.to_host();
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`target_component`、`d_target`、`to_host`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `begin = Clock::now();`：先按括号和运算符优先级形成右侧 `Clock::now()` 的值，再用它初始化或赋给 `begin`。函数调用语法：`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `begin` 时得到的就是这次写入的新值。
- 对源码锚点 `state.target_component = d_target.to_host();`：先按括号和运算符优先级形成右侧 `d_target.to_host()` 的值，再用它初始化或赋给 `state.target_component`。函数调用语法：`d_target.to_host(...)` 先准备括号内实参，再把控制权交给 `d_target.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `state.target_component` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `target_component`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.40 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `state.interference_component = d_interference.to_host();`。

```cpp
    state.interference_component = d_interference.to_host();
    state.noise_component = d_noise.to_host();
    state.echo = d_echo.to_host();
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`interference_component`、`d_interference`、`to_host`、`noise_component`、`d_noise`、`echo`、`d_echo`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `state.interference_component = d_interference.to_host();`：先按括号和运算符优先级形成右侧 `d_interference.to_host()` 的值，再用它初始化或赋给 `state.interference_component`。函数调用语法：`d_interference.to_host(...)` 先准备括号内实参，再把控制权交给 `d_interference.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `state.interference_component` 时得到的就是这次写入的新值。
- 对源码锚点 `state.noise_component = d_noise.to_host();`：先按括号和运算符优先级形成右侧 `d_noise.to_host()` 的值，再用它初始化或赋给 `state.noise_component`。函数调用语法：`d_noise.to_host(...)` 先准备括号内实参，再把控制权交给 `d_noise.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `state.noise_component` 时得到的就是这次写入的新值。
- 对源码锚点 `state.echo = d_echo.to_host();`：先按括号和运算符优先级形成右侧 `d_echo.to_host()` 的值，再用它初始化或赋给 `state.echo`。函数调用语法：`d_echo.to_host(...)` 先准备括号内实参，再把控制权交给 `d_echo.to_host`，返回值回到调用点。 运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `state.echo` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`、`interference_component`、`noise_component`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.41 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.d2h_ms += milliseconds(begin, Clock::now());`。

```cpp
    evidence.d2h_ms += milliseconds(begin, Clock::now());
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.d2h_ms += milliseconds(begin, Clock::now());`：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 对应运算，再把结果写回同一个 `evidence.d2h_ms` 对象。函数调用语法：`milliseconds(...)` 先准备括号内实参，再把控制权交给 `milliseconds`，返回值回到调用点；`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`+=`：复合加法赋值，等价于在只求值一次左侧地址的前提下执行左侧 = 左侧 + 右侧；`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `evidence.d2h_ms` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.42 d_target：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`。

```cpp

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled()) {
        state.gaussian_waveform = gaussian;
```

**语法结构**

这个代码块由条件分支、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 5 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`total_ms`、`milliseconds`、`total_begin`、`Clock`、`now`、`visualization_capture_enabled`、`state`、`gaussian_waveform`、`gaussian`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`：先按括号和运算符优先级形成右侧 `milliseconds(total_begin, Clock::now())` 的值，再用它初始化或赋给 `evidence.total_ms`。函数调用语法：`milliseconds(...)` 先准备括号内实参，再把控制权交给 `milliseconds`，返回值回到调用点；`Clock::now(...)` 先准备括号内实参，再把控制权交给 `Clock::now`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `evidence.total_ms` 时得到的就是这次写入的新值。
- 对源码锚点 `if (visualization_capture_enabled()) {`：关键字语法：`if`：if 先把括号内表达式转换为真假，仅为真时进入分支。 函数调用语法：`visualization_capture_enabled(...)` 先准备括号内实参，再把控制权交给 `visualization_capture_enabled`，返回值回到调用点。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `state.gaussian_waveform = gaussian;`：先按括号和运算符优先级形成右侧 `gaussian` 的值，再用它初始化或赋给 `state.gaussian_waveform`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 执行后，后续代码读取 `state.gaussian_waveform` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `gaussian_waveform`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于可视化数据链：它读取/导出数值结果，建立坐标、单位和图形对象；图片是解释性派生物，不能替代正式数值门禁。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.43 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `state.sawtooth_waveform = sawtooth;`。

```cpp
        state.sawtooth_waveform = sawtooth;
        state.square_waveform = square;
    }
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`sawtooth_waveform`、`sawtooth`、`square_waveform`、`square`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `state.sawtooth_waveform = sawtooth;`：先按括号和运算符优先级形成右侧 `sawtooth` 的值，再用它初始化或赋给 `state.sawtooth_waveform`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 执行后，后续代码读取 `state.sawtooth_waveform` 时得到的就是这次写入的新值。
- 对源码锚点 `state.square_waveform = square;`：先按括号和运算符优先级形成右侧 `square` 的值，再用它初始化或赋给 `state.square_waveform`。运算符：`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 执行后，后续代码读取 `state.square_waveform` 时得到的就是这次写入的新值。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`square`、`sawtooth_waveform`、`square_waveform`。 本块直接出现 2 种波形符号：`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `state.sawtooth_waveform = sawtooth;` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.44 d_target：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `return evidence;`。

```cpp
    return evidence;
}
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `evidence` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `return evidence;`：这是 C++ 局部变量声明：`return` 是静态类型，`evidence` 是当前作用域中的变量名。没有显式初始化器；类类型会调用默认构造函数，而内置标量若不是静态存储期则值未确定，读取前必须先赋值。分号结束声明。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `return evidence;` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.45 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `}  // namespace task2`。

```cpp

}  // namespace task2
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `}  // namespace task2`：运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `}  // namespace task2` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.46 noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_bits`；源码锚点 `std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)`。

```cpp
std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
```

**语法结构**

这个代码块由函数或方法定义、变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `index` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `value` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `747796405U` 是数值字面量；U 指定 unsigned int 起始类型；
- `2891336453U` 是数值字面量；U 指定 unsigned int 起始类型。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`747796405U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x2C9277B5`；这是 PCG 系列常用 32 位 LCG 乘数。此处用于让相邻 index 在进入后续 avalanche 前先分散；它不是物理常数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19） 当前上下文：在 `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 中，它为 `value` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|
|`2891336453U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xAC564B05`；这是与上述乘数配套使用的 32 位增量。此处平移状态，避免 seed/index 的零值直接保持零；不是雷达参数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19） 当前上下文：在 `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 中，它为 `value` 提供初始化/赋值；该变量的物理或工程含义见本块变量映射。|

**执行过程**

- 对源码锚点 `std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed) {`：这是函数定义头而不是一次调用。`noise_bits` 是新函数名；它前面的 `std::uint32_t` 包含返回类型及可能的 CUDA 限定符。括号中的 `std::uint32_t index, std::uint32_t seed` 是形参声明：调用者传入实参后按顺序绑定到这些局部名称；没有 `&` 或 `*` 的标量参数按值复制。末尾 `{` 打开函数体，因此此处只建立可执行代码的定义，真正计算要等调用发生。
- 对源码锚点 `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);`：先按括号和运算符优先级形成右侧 `seed ^ (index * 747796405U + 2891336453U)` 的值，再用它初始化或赋给 `std::uint32_t value`。字面量与类型：`747796405U` 是数值字面量；U 指定 unsigned int 起始类型；`2891336453U` 是数值字面量；U 指定 unsigned int 起始类型。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`^`：按位异或：把两个整数的同一二进制位比较，相同得 0、不同得 1；它不是乘方；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 本表达式的实际求值顺序是：先算括号中的乘法，再做加法，最后与括号外整数按位异或。参与运算的 `U` 常量会促使通常算术转换采用无符号类型；无符号 N 位结果按模 $2^N$ 回绕。这里把样本 `index`、随机 `seed` 和两个混合常量扩散成确定性的伪随机比特，后续噪声生成可在 CPU 与 GPU 上复现同一输入。 执行后，后续代码读取 `std::uint32_t value` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 先执行乘加再与 seed 组合，是为了让相邻 `index` 不直接形成相邻整数状态，并避免零输入停留在简单零模式；这一步采用 PCG 常见 LCG 常数作为低成本初始打散，后续再由 Murmur 风格 avalanche 完成扩散。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）。

**初学者易错点**

- C/C++/Python 中 `^` 是按位异或，不是乘方；乘方在 Python 写作 `**`，C++ 通常调用 `std::pow`；
- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.47 noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_bits`；源码锚点 `value ^= value >> 16;`。

```cpp
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 0 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2246822519U` 是数值字面量；U 指定 unsigned int 起始类型；
- `13` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `value ^= value >> 16;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2246822519U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x85EBCA77`。它是奇数，因此在模 $2^{32}$ 算术中可逆并能借助乘法进位扩散比特。它接近但不等于 MurmurHash3 `fmix32` 的 `0x85EBCA6B`；仓库源码未记录为何相差 `0x0C`，不能冒充原始 Murmur 常数。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：它直接参与表达式 `value *= 2246822519U;`，作用由同一表达式中的运算符决定。|
|`13`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`13` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `value ^= value >> 13;` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `value ^= value >> 16;`：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`>>`：右移二进制位；无符号整数高位补 0；`^=`：复合按位异或赋值：先把左侧旧值与右侧逐位异或，再把结果写回同一对象。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `value *= 2246822519U;`：读取 `value` 的旧值，与 `2246822519U` 执行 `*` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`2246822519U` 是数值字面量；U 指定 unsigned int 起始类型。 运算符：`*=`：复合乘法赋值。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `value ^= value >> 13;`：读取 `value` 的旧值，与 `value >> 13` 执行 `^` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`13` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`>>`：右移二进制位；无符号整数高位补 0；`^=`：复合按位异或赋值：先把左侧旧值与右侧逐位异或，再把结果写回同一对象。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- C/C++/Python 中 `^` 是按位异或，不是乘方；乘方在 Python 写作 `**`，C++ 通常调用 `std::pow`；
- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断。

### 4.48 noise_bits：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_bits`；源码锚点 `value *= 3266489917U;`。

```cpp
    value *= 3266489917U;
    return value ^ (value >> 16);
}
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 1 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `3266489917U` 是数值字面量；U 指定 unsigned int 起始类型；
- `16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`3266489917U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xC2B2AE3D`。它同样是用于继续扩散的奇数乘数，但不等于 MurmurHash3 `fmix32` 的 `0xC2B2AE35`；仓库源码未记录为何相差 `0x08`。改变它会改变全部确定性噪声序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：它直接参与表达式 `value *= 3266489917U;`，作用由同一表达式中的运算符决定。|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19） 当前上下文：在 `return value ^ (value >> 16);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `value *= 3266489917U;`：读取 `value` 的旧值，与 `3266489917U` 执行 `*` 对应运算，再把结果写回同一个 `value` 对象。字面量与类型：`3266489917U` 是数值字面量；U 指定 unsigned int 起始类型。 运算符：`*=`：复合乘法赋值。 执行后，后续代码读取 `value` 时得到的就是这次写入的新值。
- 对源码锚点 `return value ^ (value >> 16);`：关键字语法：`return`：return 结束当前函数，并把表达式结果交给调用者。 字面量与类型：`16` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`>>`：右移二进制位；无符号整数高位补 0；`^`：按位异或：把两个整数的同一二进制位比较，相同得 0、不同得 1；它不是乘方。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- C/C++/Python 中 `^` 是按位异或，不是乘方；乘方在 Python 写作 `**`，C++ 通常调用 `std::pow`；
- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断。

### 4.49 noise_sample：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_sample`；源码锚点 `float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)`。

```cpp

float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * amplitude;
```

**语法结构**

这个代码块由函数或方法定义、变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 5 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `index` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `amplitude` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `0xffffU` 是数值字面量；U 指定 unsigned int 起始类型；
- `32767.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`amplitude`|当前函数或调用的参数名称，接收调用者绑定的输入 `amplitude`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0xffffU`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。 当前上下文：在 `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`32767.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。 当前上下文：在 `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。 当前上下文：在 `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude) {`：这是函数定义头而不是一次调用。`noise_sample` 是新函数名；它前面的 `float` 包含返回类型及可能的 CUDA 限定符。括号中的 `std::uint32_t index, std::uint32_t seed, float amplitude` 是形参声明：调用者传入实参后按顺序绑定到这些局部名称；没有 `&` 或 `*` 的标量参数按值复制。末尾 `{` 打开函数体，因此此处只建立可执行代码的定义，真正计算要等调用发生。
- 对源码锚点 `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)`：关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换；`return`：return 结束当前函数，并把表达式结果交给调用者。 函数调用语法：`noise_bits(...)` 先准备括号内实参，再把控制权交给 `noise_bits`，返回值回到调用点。 字面量与类型：`0xffffU` 是数值字面量；U 指定 unsigned int 起始类型；`32767.5F` 是数值字面量；F 指定 float，而非默认 double；`1.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`-`：减法；位于单个操作数前时是一元负号；`&`：按位与；声明上下文中的单个 & 也可表示引用；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `* amplitude;`：这是源码注释；注释不参与执行。文字说明作者意图，后续解释仍以实际语句为准。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 先用 `0xFFFF` 只保留低 16 位，是为了得到固定的 65536 个离散状态；随后除以半跨度 `32767.5` 并减 1，把它线性映射为以 0 为中心的 $[-1,1]$。最后乘 `noise_std` 才得到任务噪声幅度。可替代为 32 位到浮点的完整映射或标准 RNG，但那会改变历史 CPU/GPU 确定性序列和精度证据。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.50 noise_sample：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_sample`；源码锚点 `}`。

```cpp
}
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 ；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `}` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.51 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `std::vector<float> synthesize_cpu(`。

```cpp

std::vector<float> synthesize_cpu(
    const std::vector<float>& chirp, const std::vector<float>& gaussian,
    const std::vector<float>& sawtooth, const std::vector<float>& square,
    std::uint32_t seed, float target_weight, std::vector<float>* target,
    std::vector<float>* interference, std::vector<float>* noise,
    const TaskConfig& config)
{
    std::vector<float> echo(chirp.size());
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `chirp` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `gaussian` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `sawtooth` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `square` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `target_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `target` 由 `std::vector<float>*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `interference` 由 `std::vector<float>*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `noise` 由 `std::vector<float>*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`chirp`|当前函数或调用的参数名称，接收调用者绑定的输入 `chirp`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& chirp, const std::vector<float>& gaussian,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`gaussian`|当前函数或调用的参数名称，接收调用者绑定的输入 `gaussian`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& chirp, const std::vector<float>& gaussian,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`sawtooth`|当前函数或调用的参数名称，接收调用者绑定的输入 `sawtooth`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& sawtooth, const std::vector<float>& square,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`square`|当前函数或调用的参数名称，接收调用者绑定的输入 `square`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& sawtooth, const std::vector<float>& square,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`target_weight`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`target`|当前函数或调用的参数名称，接收调用者绑定的输入 `target`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `std::uint32_t seed, float target_weight, std::vector<float>* target,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`interference`|当前函数或调用的参数名称，接收调用者绑定的输入 `interference`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `std::vector<float>* interference, std::vector<float>* noise,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `std::vector<float> synthesize_cpu( const std::vector<float>& chirp, const std::vector<float>& gaussian, const std::vector<float>& sawtooth, const std::vector<float>& square, std...`：这是函数定义头而不是一次调用。`synthesize_cpu` 是新函数名；它前面的 `std::vector<float>` 包含返回类型及可能的 CUDA 限定符。括号中的 ` const std::vector<float>& chirp, const std::vector<float>& gaussian, const std::vector<float>& sawtooth, const std::vector<float>& square, std::uint32_t seed, float target_weight, std::vector<float>* target, std::vector<float>* interference, std::vector<float>* noise, const TaskConfig& config` 是形参声明：调用者传入实参后按顺序绑定到这些局部名称；没有 `&` 或 `*` 的标量参数按值复制。末尾 `{` 打开函数体，因此此处只建立可执行代码的定义，真正计算要等调用发生。
- 对源码锚点 `std::vector<float> echo(chirp.size());`：函数调用语法：`echo(...)` 先准备括号内实参，再把控制权交给 `echo`，返回值回到调用点；`chirp.size(...)` 先准备括号内实参，再把控制权交给 `chirp.size`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`、`echo`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.52 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `target->assign(chirp.size(), 0.0F);`。

```cpp
    target->assign(chirp.size(), 0.0F);
    interference->assign(chirp.size(), 0.0F);
    noise->assign(chirp.size(), 0.0F);
```

**语法结构**

这个代码块由函数调用构成。代码块按原始顺序保留跨行结构；其中可见 6 个开放分隔符与 6 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `target->assign(chirp.size(), 0.0F);` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `interference->assign(chirp.size(), 0.0F);` 中，它是分支或边界比较值，决定哪条控制路径被执行；在 `noise->assign(chirp.size(), 0.0F);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `target->assign(chirp.size(), 0.0F);`：函数调用语法：`target->assign(...)` 先准备括号内实参，再把控制权交给 `target->assign`，返回值回到调用点；`chirp.size(...)` 先准备括号内实参，再把控制权交给 `chirp.size`，返回值回到调用点。 字面量与类型：`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`->`：通过指针访问对象成员，左侧必须产生指向对象的指针；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `interference->assign(chirp.size(), 0.0F);`：函数调用语法：`interference->assign(...)` 先准备括号内实参，再把控制权交给 `interference->assign`，返回值回到调用点；`chirp.size(...)` 先准备括号内实参，再把控制权交给 `chirp.size`，返回值回到调用点。 字面量与类型：`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`->`：通过指针访问对象成员，左侧必须产生指向对象的指针；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。
- 对源码锚点 `noise->assign(chirp.size(), 0.0F);`：函数调用语法：`noise->assign(...)` 先准备括号内实参，再把控制权交给 `noise->assign`，返回值回到调用点；`chirp.size(...)` 先准备括号内实参，再把控制权交给 `chirp.size`，返回值回到调用点。 字面量与类型：`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`->`：通过指针访问对象成员，左侧必须产生指向对象的指针；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.53 synthesize_cpu：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `for (std::size_t index = 0; index < chirp.size(); ++index) {`。

```cpp
    for (std::size_t index = 0; index < chirp.size(); ++index) {
        (*target)[index] = index >= static_cast<std::size_t>(config.target_delay_samples)
            ? target_weight * chirp[index - config.target_delay_samples] : 0.0F;
```

**语法结构**

这个代码块由变量声明与初始化、循环、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 7 个开放分隔符与 6 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `index` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `for (std::size_t index = 0; index < chirp.size(); ++index) {` 中，它参与迭代起点、终点或步长，直接决定循环执行次数；它直接参与表达式 `? target_weight * chirp[index - config.target_delay_samples] : 0.0F;`，作用由同一表达式中的运算符决定。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `? target_weight * chirp[index - config.target_delay_samples] : 0.0F;`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `for (std::size_t index = 0; index < chirp.size(); ++index) {`：先按括号和运算符优先级形成右侧 `0; index < chirp.size(); ++index) {` 的值，再用它初始化或赋给 `for (std::size_t index`。关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 函数调用语法：`chirp.size(...)` 先准备括号内实参，再把控制权交给 `chirp.size`，返回值回到调用点。 字面量与类型：`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`++`：自增 1；前置形式先增后取值，后置形式先取旧值后增；`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`<`：小于比较；模板实参列表中也可作为左尖括号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `for (std::size_t index` 时得到的就是这次写入的新值。
- 对源码锚点 `(*target)[index] = index >= static_cast<std::size_t>(config.target_delay_samples) ? target_weight * chirp[index - config.target_delay_samples] : 0.0F;`：先按括号和运算符优先级形成右侧 `index >= static_cast<std::size_t>(config.target_delay_samples) ? target_weight * chirp[index - config.target_delay_samples] : 0.0F` 的值，再用它初始化或赋给 `(*target)[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`>=`：大于或等于比较；`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`-`：减法；位于单个操作数前时是一元负号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员；`?`：条件运算符的起点，先判断条件，只计算冒号两侧中的一支；`:`：条件运算符的分支分隔符、标签或类型/字典分隔符，取决于上下文。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `(*target)[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`target_delay_samples`。 本块直接出现 1 种波形符号：`chirp`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.54 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `(*interference)[index] = config.gaussian_weight * gaussian[index] +`。

```cpp
        (*interference)[index] = config.gaussian_weight * gaussian[index] +
            config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index];
        (*noise)[index] = noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 9 个开放分隔符与 9 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `interference`、`index`、`config`、`gaussian_weight`、`gaussian`、`sawtooth_weight`、`sawtooth`、`square_weight`、`square`、`noise`、`noise_sample`、`std`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `(*interference)[index] = config.gaussian_weight * gaussian[index] + config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index];`：先按括号和运算符优先级形成右侧 `config.gaussian_weight * gaussian[index] + config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index]` 的值，再用它初始化或赋给 `(*interference)[index]`。运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `(*interference)[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `(*noise)[index] = noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude);`：先按括号和运算符优先级形成右侧 `noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude)` 的值，再用它初始化或赋给 `(*noise)[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`noise_sample(...)` 先准备括号内实参，再把控制权交给 `noise_sample`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `(*noise)[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sawtooth`、`square`、`gaussian_weight`、`sawtooth_weight`、`square_weight`、`noise_amplitude`。 本块直接出现 2 种波形符号：`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.55 synthesize_cpu：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];`。

```cpp
        echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];
    }
    return echo;
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 7 个开放分隔符与 8 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `echo` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];`：先按括号和运算符优先级形成右侧 `(*target)[index] + (*interference)[index] + (*noise)[index]` 的值，再用它初始化或赋给 `echo[index]`。运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`+`：加法；也可能是一元正号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `echo[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。
- 对源码锚点 `return echo;`：这是 C++ 局部变量声明：`return` 是静态类型，`echo` 是当前作用域中的变量名。没有显式初始化器；类类型会调用默认构造函数，而内置标量若不是静态存储期则值未确定，读取前必须先赋值。分号结束声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `echo`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断。

### 4.56 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `}`。

```cpp
}
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 ；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `}` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.57 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `}  // namespace`。

```cpp

}  // namespace
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `}  // namespace`：运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `}  // namespace` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

### 4.58 generate_step1_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `generate_step1_cpu_reference`；源码锚点 `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)`。

```cpp
Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)
{
    Step1ValidationData data;
```

**语法结构**

这个代码块由函数或方法定义、变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `data` 由 `Step1ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`generate_step1_cpu_reference`|生成当前名称所描述数据的 helper/结果|对应生成公式的输出端|由输入参数构造数据，供后续 step 或测试使用|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config) {`：这是函数定义头而不是一次调用。`generate_step1_cpu_reference` 是新函数名；它前面的 `Step1ValidationData` 包含返回类型及可能的 CUDA 限定符。括号中的 `const TaskConfig& config` 是形参声明：调用者传入实参后按顺序绑定到这些局部名称；没有 `&` 或 `*` 的标量参数按值复制。末尾 `{` 打开函数体，因此此处只建立可执行代码的定义，真正计算要等调用发生。
- 对源码锚点 `Step1ValidationData data;`：这是 C++ 局部变量声明：`Step1ValidationData` 是静态类型，`data` 是当前作用域中的变量名。没有显式初始化器；类类型会调用默认构造函数，而内置标量若不是静态存储期则值未确定，读取前必须先赋值。分号结束声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.59 time：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `std::vector<float> time(config.samples), pulse_time(config.samples);`。

```cpp
    std::vector<float> time(config.samples), pulse_time(config.samples);
    std::vector<float> phase(config.samples), square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        time[index] = static_cast<float>(index) / config.sample_rate_hz;
```

**语法结构**

这个代码块由变量声明与初始化、循环、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 8 个开放分隔符与 7 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `index` 由 `int` 声明：有符号整数类型；具体位宽由平台决定，本项目常见平台通常为 32 位；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `for (int index = 0; index < config.samples; ++index) {` 中，它参与迭代起点、终点或步长，直接决定循环执行次数。|

**执行过程**

- 对源码锚点 `std::vector<float> time(config.samples), pulse_time(config.samples);`：函数调用语法：`time(...)` 先准备括号内实参，再把控制权交给 `time`，返回值回到调用点；`pulse_time(...)` 先准备括号内实参，再把控制权交给 `pulse_time`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `std::vector<float> phase(config.samples), square_phase(config.samples);`：函数调用语法：`phase(...)` 先准备括号内实参，再把控制权交给 `phase`，返回值回到调用点；`square_phase(...)` 先准备括号内实参，再把控制权交给 `square_phase`，返回值回到调用点。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `for (int index = 0; index < config.samples; ++index) {`：先按括号和运算符优先级形成右侧 `0; index < config.samples; ++index) {` 的值，再用它初始化或赋给 `for (int index`。关键字语法：`for`：for 由初始化、继续条件和步进三部分控制重复执行。 字面量与类型：`0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`++`：自增 1；前置形式先增后取值，后置形式先取旧值后增；`<`：小于比较；模板实参列表中也可作为左尖括号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `for (int index` 时得到的就是这次写入的新值。
- 对源码锚点 `time[index] = static_cast<float>(index) / config.sample_rate_hz;`：先按括号和运算符优先级形成右侧 `static_cast<float>(index) / config.sample_rate_hz` 的值，再用它初始化或赋给 `time[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 运算符：`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `time[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`、`samples`、`sample_rate_hz`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出；
- 当前块含跨行开放符号；阅读时必须继续到后续匹配的闭合符号，不能把当前片段误判为完整调用、条件或初始化器。

### 4.60 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `pulse_time[index] = time[index] - 0.45F;`。

```cpp
        pulse_time[index] = time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `saw_cycles` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `0.45F` 是数值字面量；F 指定 float，而非默认 double；
- `190.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`saw_cycles`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `saw_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const double saw_cycles = std::fmod(`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.45F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `pulse_time[index] = time[index] - 0.45F;`，作用由同一表达式中的运算符决定。|
|`190.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `pulse_time[index] = time[index] - 0.45F;`：先按括号和运算符优先级形成右侧 `time[index] - 0.45F` 的值，再用它初始化或赋给 `pulse_time[index]`。字面量与类型：`0.45F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`-`：减法；位于单个操作数前时是一元负号；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 执行后，后续代码读取 `pulse_time[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `const double saw_cycles = std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`：先按括号和运算符优先级形成右侧 `std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 的值，再用它初始化或赋给 `const double saw_cycles`。关键字语法：`const`：const 禁止通过当前名字修改对象；`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`std::fmod(...)` 先准备括号内实参，再把控制权交给 `std::fmod`，返回值回到调用点。 字面量与类型：`190.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const double saw_cycles` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `sample_rate_hz`。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.61 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `const double square_cycles = std::fmod(`。

```cpp
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 4 个开放分隔符与 4 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `square_cycles` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `saw_cycles` 由 `kPi *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `125.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square_cycles`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `square_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const double square_cycles = std::fmod(`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`saw_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `saw_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`125.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`，作用由同一表达式中的运算符决定。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|

**执行过程**

- 对源码锚点 `const double square_cycles = std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);`：先按括号和运算符优先级形成右侧 `std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 的值，再用它初始化或赋给 `const double square_cycles`。关键字语法：`const`：const 禁止通过当前名字修改对象；`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`std::fmod(...)` 先准备括号内实参，再把控制权交给 `std::fmod`，返回值回到调用点。 字面量与类型：`125.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；`1.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const double square_cycles` 时得到的就是这次写入的新值。
- 对源码锚点 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);`：先按括号和运算符优先级形成右侧 `static_cast<float>(2.0 * kPi * saw_cycles)` 的值，再用它初始化或赋给 `phase[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `phase[index]` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `square`、`sample_rate_hz`。 本块直接出现 1 种波形符号：`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.62 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`。

```cpp
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    data.chirp_cpu = cusignal::chirp_typed_cpu(
        time, 20.0F,
        std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),
        70.0F, 0.0F);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 5 个开放分隔符与 6 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `square_cycles` 由 `kPi *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `20.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `70.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|
|`square_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `square_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。 当前上下文：在 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`20.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `time, 20.0F,`，作用由同一表达式中的运算符决定。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：在 `std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F),` 中，它是分支或边界比较值，决定哪条控制路径被执行。|
|`70.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `70.0F, 0.0F);`，作用由同一表达式中的运算符决定。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：它直接参与表达式 `time, 20.0F,`，作用由同一表达式中的运算符决定；它直接参与表达式 `70.0F, 0.0F);`，作用由同一表达式中的运算符决定。|

**执行过程**

- 对源码锚点 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`：先按括号和运算符优先级形成右侧 `static_cast<float>(2.0 * kPi * square_cycles)` 的值，再用它初始化或赋给 `square_phase[index]`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 字面量与类型：`2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`*`：乘法；若贴在类型后也可能声明指针，需由声明上下文判断；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 方括号在 `name[index]` 中执行下标访问；紧邻 lambda 参数前时则是捕获列表。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `square_phase[index]` 时得到的就是这次写入的新值。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。
- 对源码锚点 `data.chirp_cpu = cusignal::chirp_typed_cpu( time, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F, 0.0F);`：先按括号和运算符优先级形成右侧 `cusignal::chirp_typed_cpu( time, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F, 0.0F)` 的值，再用它初始化或赋给 `data.chirp_cpu`。关键字语法：`static_cast`：static_cast<T>(x) 请求编译期受检查的显式类型转换。 函数调用语法：`cusignal::chirp_typed_cpu(...)` 先准备括号内实参，再把控制权交给 `cusignal::chirp_typed_cpu`，返回值回到调用点；`std::max(...)` 先准备括号内实参，再把控制权交给 `std::max`，返回值回到调用点。 字面量与类型：`20.0F` 是数值字面量；F 指定 float，而非默认 double；`1.0F` 是数值字面量；F 指定 float，而非默认 double；`70.0F` 是数值字面量；F 指定 float，而非默认 double；`0.0F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`/`：除法；整数操作数执行整数除法，浮点操作数执行浮点除法；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `data.chirp_cpu` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`square`、`samples`、`sample_rate_hz`。 本块直接出现 2 种波形符号：`chirp`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 类型旁的 `*` 声明指针，两个数值之间的 `*` 才是乘法；两者不能只凭字符判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.63 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(`。

```cpp
    const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(
        pulse_time, 170.0F, 0.20F);
    const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::sawtooth_typed_cpu(phase, 0.65));
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `gaussian_cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `sawtooth_cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `170.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.20F` 是数值字面量；F 指定 float，而非默认 double；
- `0.65` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`gaussian_cpu`|CPU reference 对应量|与去掉 `_cpu` 后的任务量采用同一数学定义|供 GPU/CPU 精度 comparison 使用|
|`sawtooth_cpu`|CPU reference 对应量|与去掉 `_cpu` 后的任务量采用同一数学定义|供 GPU/CPU 精度 comparison 使用|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`170.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `pulse_time, 170.0F, 0.20F);`，作用由同一表达式中的运算符决定。|
|`0.20F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：它直接参与表达式 `pulse_time, 170.0F, 0.20F);`，作用由同一表达式中的运算符决定。|
|`0.65`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `cusignal::sawtooth_typed_cpu(phase, 0.65));` 中，它作为 `cusignal::sawtooth_typed_cpu` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `const auto gaussian_cpu = cusignal::gausspulse_typed_cpu( pulse_time, 170.0F, 0.20F);`：先按括号和运算符优先级形成右侧 `cusignal::gausspulse_typed_cpu( pulse_time, 170.0F, 0.20F)` 的值，再用它初始化或赋给 `const auto gaussian_cpu`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`cusignal::gausspulse_typed_cpu(...)` 先准备括号内实参，再把控制权交给 `cusignal::gausspulse_typed_cpu`，返回值回到调用点。 字面量与类型：`170.0F` 是数值字面量；F 指定 float，而非默认 double；`0.20F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto gaussian_cpu` 时得到的就是这次写入的新值。
- 对源码锚点 `const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::sawtooth_typed_cpu(phase, 0.65));`：先按括号和运算符优先级形成右侧 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::sawtooth_typed_cpu(phase, 0.65))` 的值，再用它初始化或赋给 `const auto sawtooth_cpu`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(...)` 先准备括号内实参，再把控制权交给 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`，返回值回到调用点；`cusignal::sawtooth_typed_cpu(...)` 先准备括号内实参，再把控制权交给 `cusignal::sawtooth_typed_cpu`，返回值回到调用点。 字面量与类型：`0.65` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto sawtooth_cpu` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `gausspulse`、`sawtooth`。 本块直接出现 2 种波形符号：`gausspulse`、`sawtooth`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.64 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(`。

```cpp
    const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::square_typed_cpu(square_phase, 0.35));
    data.echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed,
        config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 3 个开放分隔符与 3 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `square_cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square_cpu`|CPU reference 对应量|与去掉 `_cpu` 后的任务量采用同一数学定义|供 GPU/CPU 精度 comparison 使用|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.35`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `cusignal::square_typed_cpu(square_phase, 0.35));` 中，它作为 `cusignal::square_typed_cpu` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::square_typed_cpu(square_phase, 0.35));`：先按括号和运算符优先级形成右侧 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::square_typed_cpu(square_phase, 0.35))` 的值，再用它初始化或赋给 `const auto square_cpu`。关键字语法：`const`：const 禁止通过当前名字修改对象；`auto`：auto 让编译器从初始化表达式推导静态类型。 函数调用语法：`cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(...)` 先准备括号内实参，再把控制权交给 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`，返回值回到调用点；`cusignal::square_typed_cpu(...)` 先准备括号内实参，再把控制权交给 `cusignal::square_typed_cpu`，返回值回到调用点。 字面量与类型：`0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `const auto square_cpu` 时得到的就是这次写入的新值。
- 对源码锚点 `data.echo_cpu = synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed, config.target_weight, &data.target_cpu, &data.interference_cpu, &data....`：先按括号和运算符优先级形成右侧 `synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed, config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config)` 的值，再用它初始化或赋给 `data.echo_cpu`。函数调用语法：`synthesize_cpu(...)` 先准备括号内实参，再把控制权交给 `synthesize_cpu`，返回值回到调用点。 运算符：`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 执行后，后续代码读取 `data.echo_cpu` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`、`echo`、`noise_seed`、`target_weight`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- `auto` 不是动态类型；编译器仍在编译期确定唯一静态类型，引用和 `const` 是否保留要看声明写法；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.65 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `std::vector<float> changed_target, changed_interference, changed_noise;`。

```cpp
    std::vector<float> changed_target, changed_interference, changed_noise;
    data.changed_echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U,
        std::max(0.0F, config.target_weight - 0.05F),
        &changed_target, &changed_interference, &changed_noise, config);
```

**语法结构**

这个代码块由变量声明与初始化、函数调用构成。代码块按原始顺序保留跨行结构；其中可见 2 个开放分隔符与 2 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `changed_target` 由 `std::vector<float>` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `1U` 是数值字面量；U 指定 unsigned int 起始类型；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.05F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`changed_target`|当前表达式读取或传递的工程名称 `changed_target`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `std::vector<float> changed_target, changed_interference, changed_noise;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`config`|外部配置对象|无单一数学符号；其字段实例化 $N,f_s,B,PFA$ 等参数|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1U`|`U` 使整数字面量从 unsigned int 候选类型开始|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。 当前上下文：它直接参与表达式 `data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U,`，作用由同一表达式中的运算符决定。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。 当前上下文：在 `std::max(0.0F, config.target_weight - 0.05F),` 中，它作为 `std::max` 的实参参与当前调用。|
|`0.05F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed 或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。 当前上下文：在 `std::max(0.0F, config.target_weight - 0.05F),` 中，它作为 `std::max` 的实参参与当前调用。|

**执行过程**

- 对源码锚点 `std::vector<float> changed_target, changed_interference, changed_noise;`：运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。
- 对源码锚点 `data.changed_echo_cpu = synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U, std::max(0.0F, config.target_weight - 0.05F), &changed_ta...`：先按括号和运算符优先级形成右侧 `synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U, std::max(0.0F, config.target_weight - 0.05F), &changed_target, &changed_interference, &changed_noise, config)` 的值，再用它初始化或赋给 `data.changed_echo_cpu`。函数调用语法：`synthesize_cpu(...)` 先准备括号内实参，再把控制权交给 `synthesize_cpu`，返回值回到调用点；`std::max(...)` 先准备括号内实参，再把控制权交给 `std::max`，返回值回到调用点。 字面量与类型：`1U` 是数值字面量；U 指定 unsigned int 起始类型；`0.0F` 是数值字面量；F 指定 float，而非默认 double；`0.05F` 是数值字面量；F 指定 float，而非默认 double。 运算符：`::`：作用域解析，左侧指定命名空间、类型或类，右侧是其中的成员；`+`：加法；也可能是一元正号；`-`：减法；位于单个操作数前时是一元负号；`&`：按位与；声明上下文中的单个 & 也可表示引用；`=`：赋值或初始化：先得到右侧值，再写入左侧对象；`.`：通过对象本身访问成员。 圆括号若跟在函数名后表示实参表；出现在算术/逻辑表达式中则强制括号内部先形成结果；出现在控制关键字后则界定条件。 声明语法：类型部分决定对象的存储表示和可用操作，随后名称把该对象绑定到当前作用域；若有 `=` 或花括号初始化器，创建时立即取得初值。 执行后，后续代码读取 `data.changed_echo_cpu` 时得到的就是这次写入的新值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 `chirp`、`sawtooth`、`square`、`noise_seed`、`target_weight`。 本块直接出现 3 种波形符号：`chirp`、`sawtooth`、`square`；只有跨相关 Step1 块合计确认至少三种，单块不足时不能独立证明条款完成。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 整数后缀 `U` 参与通常算术转换；无符号结果按固定位宽回绕，不能按数学整数无限精度理解；
- 声明中的 `&` 表示引用，表达式中的 `&` 可能是取地址或按位与，必须结合左右操作数判断；
- 括号调用返回后，若结果既未赋值、未 return、也未通过引用/指针写出，返回值可能被丢弃；要结合当前调用签名判断真正输出。

### 4.66 time：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `return data;`。

```cpp
    return data;
}
```

**语法结构**

这个代码块由声明/表达式续接构成。代码块按原始顺序保留跨行结构；其中可见 0 个开放分隔符与 1 个闭合分隔符，圆括号用于参数/条件/分组，方括号用于下标、容器或 lambda 捕获，花括号用于 C++ 作用域或初始化器。续行不是独立语句：必须和前面的函数名、赋值号或开放括号合并阅读，直到对应闭合符号及语句终止符出现。

**名称与类型**

- `data` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- 对源码锚点 `return data;`：这是 C++ 局部变量声明：`return` 是静态类型，`data` 是当前作用域中的变量名。没有显式初始化器；类类型会调用默认构造函数，而内置标量若不是静态存储期则值未确定，读取前必须先赋值。分号结束声明。
- 对源码锚点 `}`：花括号建立或结束词法作用域；作用域决定局部变量的可见范围和自动对象的析构时机。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|当前块可见的业务锚点为 接口/资源/证据字段。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 当前锚点 `return data;` 没有引入额外的数值运算或所有权转移；它的主要阅读风险是脱离同一语义块误判角色，应结合本块的语法结构与执行过程判断它是声明、续接还是闭合作用域。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
