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
//  输出 echo 将作为 Step2 FIR 滤波去噪的输入。>
StepEvidence run_step1(PipelineState& state);

}  // namespace task2
```

### 3.2 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：文件级依赖与命名空间。`#include` 在预处理阶段引入 Step1 接口、设备数组、kernel 启动和波形算子声明；//  `<cmath>`/`<numeric>` 提供数学函数与累加算法。`namespace task2 { namespace {` 把入口放入 Task2 名称空间，并把私有 helper 限制在当前翻译单元。>
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
//  每个 CUDA 线程独立处理一个采样点 n，并行执行以下五步：
//    步骤1：计算全局线程索引 index = blockIdx.x * blockDim.x + threadIdx.x
//    步骤2：生成确定性伪随机噪声 noise_value（借用 PCG 32 位常数并采用 Murmur-like avalanche 结构的无状态整数混合（不是递推的完整 PCG））
//    步骤3：计算目标回波分量 target_value（带延迟的 chirp 信号）
//    步骤4：计算干扰分量 interference_value（高斯/锯齿/方波三种波形加权叠加）
//    步骤5：将三个分量分别写入输出数组，并叠加为 echo
//
//  【核心数学公式 —— 逐采样点回波模型】
//    对线程索引 n = index（n = 0, 1, ..., N-1）：
//
//    (a) 确定性噪声生成（借用 PCG 常数的无状态初始混合 + Murmur-like 比特扩散）：
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
    //  将一维 CUDA grid/block 的块号与块内线程号组合为全局 ID。
    //  threadIdx.x ∈ [0, blockDim.x-1]：当前线程在 block 内的编号。
    //  blockIdx.x  ∈ [0, gridDim.x-1] ：当前 block 在 grid 中的编号。
    //  blockDim.x：每个 block 的线程数；具体值由 `launch_1d_kernel` 封装决定，当前 kernel 源码没有写死。>
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // <学习注释：越界保护——当线程索引超出有效采样范围时直接返回。
    //  grid 启动的线程数向上取整到 block 的整数倍，可能超过实际 count。
    //  不执行此检查会导致越界访问 GPU 显存，产生未定义行为。>
    if (index >= count) return;
    // <学习注释：伪随机噪声种子混合——第1步：借用 PCG 常数的无状态初始混合。
    //  value = seed ⊕ (index × 747796405 + 2891336453)  [无符号 32 位，模 2^32]
    //  ⊕ 是按位异或（C++ 中的 ^），不是乘方。
    //  747796405U = 0x2C9277B5：PCG 32 位默认乘数；当前源码仅借用该数值，不执行 PCG 状态递推。
    //  2891336453U = 0xAC564B05：PCG 32 位默认增量；当前源码把它作为无状态乘加中的固定偏移，避免零输入直接保持零。
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
    //  echo 数组将作为 Step2 FIR 滤波去噪的输入。>
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
    //  高斯包络：a = (π × bw × fc)² = (π × 34)² ≈ 11411.66。
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
// <学习注释：由绑定 Git 快照 `bb1a48c2e18fe2e252e0203e8be717d0c40f9251` 的 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 补回此前漏摘的源码；以下解释只陈述语法和当前表达式可证明的作用。
//
//  【执行方法】
//  `#include "accuracy/validation/step1_validation.h"` 在预处理阶段引入 "accuracy/validation/step1_validation.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "waveforms/waveforms_typed.h"` 在预处理阶段引入 "waveforms/waveforms_typed.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include <cmath>` 在预处理阶段引入 <cmath> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task2::accuracy {` 打开命名空间 `task2::accuracy`；后续声明被放入该名称范围。
//  `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。
//
//  【任务作用】
//  本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
#include "accuracy/validation/step1_validation.h"

#include "waveforms/waveforms_typed.h"

#include <cmath>

namespace task2::accuracy {
namespace {

// <学习注释：CPU 参考实现 —— 确定性噪声比特生成（noise_bits）。
//  与 GPU kernel 中 mix_echo_kernel 的噪声生成逻辑完全一致，
//  用于在 CPU 端独立验证 GPU 结果的正确性。
//  数学公式：value = finalize(seed ⊕ (index × 747796405 + 2891336453))
//  其中 finalize 是 6 步 Murmur-like avalanche 操作。>
std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    // <学习注释：借用 PCG 常数的无状态初始混合：seed ⊕ (index × 747796405 + 2891336453)。
    //  ⊕ 是按位异或（^），不是乘方。747796405U 和 2891336453U 来自 PCG 32 位默认乘数/增量；本代码只借用常数，不构成完整 PCG。>
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
    //  数学公式：gausspulse(t) = exp(−(π·bw·fc·t)²) × cos(2π·fc·t)。>
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
// <学习注释：由绑定 Git 快照 `bb1a48c2e18fe2e252e0203e8be717d0c40f9251` 的 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 补回此前漏摘的源码；以下解释只陈述语法和当前表达式可证明的作用。
//
//  【执行方法】
//  `Step1ValidationData collect_step1_validation(const PipelineState& state) {` 中的
//  `collect_step1_validation` 是函数定义；名称前的 `Step1ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `return generate_step1_cpu_reference(state.config);` 是返回语句：先求值
//  `generate_step1_cpu_reference(state.config)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
//  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
//  `}  // namespace task2::accuracy` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。
//
//  【任务作用】
//  本块通过定义/声明 `collect_step1_validation`, `generate_step1_cpu_reference`；调用 `collect_step1_validation`,
//  `generate_step1_cpu_reference`；返回 `generate_step1_cpu_reference(state.config)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。
//
//  【初学者易错点】
//  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>

Step1ValidationData collect_step1_validation(const PipelineState& state)
{
    return generate_step1_cpu_reference(state.config);
}

}  // namespace task2::accuracy
```

### 3.4 函数总体处理方法

#### 3.4.1 总体业务流水线

`外部配置 → Host 时间轴/相位 → H2D → 四种正式波形算子 → GPU 回波叠加 kernel → D2H → CPU reference/扰动验证`。

Task2 Step1 生成 chirp、gausspulse、sawtooth 和 square 四种波形；延迟 chirp 构成目标分量，其余三种波形构成干扰，最后叠加确定性噪声得到 `echo`，交给 Step2 FIR 滤波。

#### 3.4.2 主要函数职责

|符号|职责|执行位置|
|---|---|---|
|`run_step1`|组织配置、时间轴、波形算子、H2D/D2H、回波 kernel 和证据字段|GPU Host wrapper/业务入口|
|`mix_echo_kernel`|每个有效线程处理一个采样点，计算目标、干扰、噪声及其和|GPU device kernel|
|`noise_bits` / `noise_sample`|按与 GPU 相同的无状态整数混合和幅度映射生成 CPU 噪声|CPU reference helper|
|`synthesize_cpu`|串行生成目标、干扰、噪声和总回波|CPU reference core|
|`generate_step1_cpu_reference`|生成基准结果和改变 seed/幅度后的扰动对照|CPU validation core|
|`collect_step1_validation`|以正式状态配置调用 CPU reference|验证入口|

#### 3.4.3 CPU/GPU 等价与边界

- CPU 与 GPU 使用相同的波形参数、目标延迟、权重、seed 和噪声幅度，但经独立 Host/device 路径计算。
- `if (index >= count) return;` 只保护多启动的 GPU 线程；有效线程与 CPU 循环使用同一展平采样索引。
- 相同输入下的 CPU/GPU 对照用于精度验证；运行结论仍必须绑定正式证据、配置和源码 SHA。

#### 3.4.4 正式算子数学原理入口

- `chirp`：[数学物理原理](../../operators/waveforms/chirp/数学物理原理.md)、[Python 源码算法](../../operators/waveforms/chirp/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/waveforms/chirp/cusignal_cpp_chirp复现逻辑.md)。当前调用以 `time` 为 $t$，使用起止频率和扫频时长生成 LFM。
- `gausspulse`：[数学物理原理](../../operators/waveforms/gausspulse/数学物理原理.md)、[Python 源码算法](../../operators/waveforms/gausspulse/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/waveforms/gausspulse/cusignal_cpp_gausspulse复现逻辑.md)。当前调用以 `pulse_time` 为时间变量，并实例化中心频率和分数带宽。
- `sawtooth`：[数学物理原理](../../operators/waveforms/sawtooth/数学物理原理.md)、[Python 源码算法](../../operators/waveforms/sawtooth/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/waveforms/sawtooth/cusignal_cpp_sawtooth复现逻辑.md)。当前调用输入相位 `phase` 和宽度参数 `0.65`。
- `square`：[数学物理原理](../../operators/waveforms/square/数学物理原理.md)、[Python 源码算法](../../operators/waveforms/square/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/waveforms/square/cusignal_cpp_square复现逻辑.md)。当前调用输入 `square_phase` 和占空比 `0.35`。

算子内部 Python/C++ 实现以以上三份学习文档为准；本任务文档只解释调用参数、前后数据和业务公式。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step1(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明、函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `StepEvidence run_step1(PipelineState& state);` 中的 `run_step1` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `run_step1`；调用 `run_step1`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `#include "step1.h"`。

```cpp
#include "step1.h"

#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
#include "waveforms/waveforms_typed.h"
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step1`、`h`、`cuda_utils`、`device_array`、`kernel_launch`、`waveforms`、`waveforms_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step1.h"` 在预处理阶段引入 "step1.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/kernel_launch.h"` 在预处理阶段引入 "cuda_utils/kernel_launch.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "waveforms/waveforms_typed.h"` 在预处理阶段引入 "waveforms/waveforms_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `#include <cmath>`。

```cpp
#include <cmath>
#include <numeric>
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`cmath`、`numeric`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include <cmath>` 在预处理阶段引入 <cmath> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

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

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `chirp` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `gaussian` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `sawtooth` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `square` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `target` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `interference` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `noise` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `echo` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `count` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `delay` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `target_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `gaussian_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `sawtooth_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `square_weight` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `noise_amplitude` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `index` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

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

- `__global__ void mix_echo_kernel( const float* chirp, const float* gaussian, const float* sawtooth, const float* square, float* target, float* interference, float* noise, float* echo, int count, int delay, std::uint32_t seed, float target_weight, float gaussian_weight, float sawtooth_weight, float square_weight, float noise_amplitude) {` 中的 `mix_echo_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const float* chirp, const float* gaussian, const float* sawtooth, const float* square, float* target, float* interference, float* noise, float* echo, int count, int delay, std::uint32_t seed, float target_weight, float gaussian_weight, float sawtooth_weight, float square_weight, float noise_amplitude` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index` 建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x` 把块号和块内线程号映射为一维全局线程索引。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `mix_echo_kernel`；调用 `mix_echo_kernel`；写入/初始化 `index`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 使用 `blockIdx.x * blockDim.x + threadIdx.x` 是为了给每个一维输出元素分配唯一全局线程。这样同一 kernel 可适配不同 block 大小；随后必须用 count 做越界保护，因为 grid 往往向上取整。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 mix_echo_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `if (index >= count) return;`。

```cpp
    if (index >= count) return;
    std::uint32_t value = seed ^
        (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 当前块中的 `if (...) return;` 必须先作为完整条件语句识别；比较运算符 `>=`/`<=` 中的 `=` 不是赋值。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`747796405U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x2C9277B5`；这是 PCG 系列常用 32 位 LCG 乘数。此处用于让相邻 index 在进入后续 avalanche 前先分散；它不是物理常数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）|
|`2891336453U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xAC564B05`；这是与上述乘数配套使用的 32 位增量。此处平移状态，避免 seed/index 的零值直接保持零；不是雷达参数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）|

**执行过程**

- `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前 CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
- `std::uint32_t value = seed ^ (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U);` 是声明并初始化：`std::uint32_t value` 建立局部对象 `value`，右侧完整表达式 `seed ^ (static_cast<std::uint32_t>(index) * 747796405U + 2891336453U)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过检查 `index >= count`；写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
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

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.9 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `value ^= value >> 16;`。

```cpp
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`2246822519U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x85EBCA77`。它是奇数，因此在模 $2^{32}$ 算术中可逆并能借助乘法进位扩散比特。它接近但不等于 MurmurHash3 `fmix32` 的 `0x85EBCA6B`；仓库源码未记录为何相差 `0x0C`，不能冒充原始 Murmur 常数。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`13`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`13` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|

**执行过程**

- `value ^= value >> 16;` 是复合赋值：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 运算，再把结果写回同一对象。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
- `value *= 2246822519U;` 是复合赋值：读取 `value` 的旧值，与 `2246822519U` 执行 `*` 运算，再把结果写回同一对象。
- `value ^= value >> 13;` 是复合赋值：读取 `value` 的旧值，与 `value >> 13` 执行 `^` 运算，再把结果写回同一对象。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.10 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `value *= 3266489917U;`。

```cpp
    value *= 3266489917U;
    value ^= value >> 16;
    const float noise_value =
        (static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`3266489917U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xC2B2AE3D`。它同样是用于继续扩散的奇数乘数，但不等于 MurmurHash3 `fmix32` 的 `0xC2B2AE35`；仓库源码未记录为何相差 `0x08`。改变它会改变全部确定性噪声序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`0xffffU`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。|
|`32767.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。|

**执行过程**

- `value *= 3266489917U;` 是复合赋值：读取 `value` 的旧值，与 `3266489917U` 执行 `*` 运算，再把结果写回同一对象。
- `value ^= value >> 16;` 是复合赋值：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 运算，再把结果写回同一对象。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
- `const float noise_value = (static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude;` 是声明并初始化：`const float noise_value` 建立局部对象 `noise_value`，右侧完整表达式 `(static_cast<float>(value & 0xffffU) / 32767.5F - 1.0F) * noise_amplitude` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 这里的 `&` 是逐位与掩码，用来保留掩码中置 1 的那些比特。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `value`, `noise_value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
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

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.11 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;`。

```cpp
    const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;
    const float interference_value =
        gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] +
        square_weight * square[index];
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `const float target_value = index >= delay ? target_weight * chirp[index - delay] : 0.0F;` 是声明并初始化：`const float target_value` 建立局部对象 `target_value`，右侧完整表达式 `index >= delay ? target_weight * chirp[index - delay] : 0.0F` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。
- `const float interference_value = gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] + square_weight * square[index];` 是声明并初始化：`const float interference_value` 建立局部对象 `interference_value`，右侧完整表达式 `gaussian_weight * gaussian[index] + sawtooth_weight * sawtooth[index] + square_weight * square[index]` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `target_value`, `interference_value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `target[index] = target_value;`。

```cpp
    target[index] = target_value;
    interference[index] = interference_value;
    noise[index] = noise_value;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `target[index] = target_value;` 是赋值：先求得右侧完整表达式 `target_value`，再把结果写入左侧可修改对象 `target[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `interference[index] = interference_value;` 是赋值：先求得右侧完整表达式 `interference_value`，再把结果写入左侧可修改对象 `interference[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `noise[index] = noise_value;` 是赋值：先求得右侧完整表达式 `noise_value`，再把结果写入左侧可修改对象 `noise[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `target[index]`, `interference[index]`, `noise[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `echo[index] = target_value + interference_value + noise_value;`。

```cpp
    echo[index] = target_value + interference_value + noise_value;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `echo[index] = target_value + interference_value + noise_value;` 是赋值：先求得右侧完整表达式 `target_value + interference_value + noise_value`，再把结果写入左侧可修改对象 `echo[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `echo[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.14 mix_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `mix_echo_kernel`；源码锚点 `}  // namespace`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `StepEvidence run_step1(PipelineState& state)`。

```cpp
StepEvidence run_step1(PipelineState& state)
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
|`run_step1`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step1(PipelineState& state) {` 中的 `run_step1` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `run_step1`；调用 `run_step1`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.16 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step1";
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
- `evidence.name = "step1";` 是赋值：先求得右侧完整表达式 `"step1"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `total_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step1` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.17 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    state.time.resize(config.samples);
    std::vector<float> pulse_time(config.samples);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `state.time.resize(config.samples);` 调用容器的 `resize` 把逻辑元素数改为实参指定的大小；增大时创建新元素，缩小时移除尾部元素。
- `std::vector<float> pulse_time(config.samples);` 声明 `pulse_time`，其静态类型是 `std::vector<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `pulse_time` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `Clock::now`, `resize`, `pulse_time`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 phase：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `std::vector<float> phase(config.samples);`。

```cpp
    std::vector<float> phase(config.samples);
    std::vector<float> square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        state.time[index] = static_cast<float>(index) / config.sample_rate_hz;
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> phase(config.samples);` 声明 `phase`，其静态类型是 `std::vector<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `phase` 的函数，模板尖括号也不是比较或位移。
- `std::vector<float> square_phase(config.samples);` 声明 `square_phase`，其静态类型是 `std::vector<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `square_phase` 的函数，模板尖括号也不是比较或位移。
- `for (int index = 0; index < config.samples; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index < config.samples`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `state.time[index] = static_cast<float>(index) / config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `static_cast<float>(index) / config.sample_rate_hz`，再把结果写入左侧可修改对象 `state.time[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `phase`, `square_phase`；写入/初始化 `time[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.19 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `pulse_time[index] = state.time[index] - 0.45F;`。

```cpp
        pulse_time[index] = state.time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.45F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`190.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `pulse_time[index] = state.time[index] - 0.45F;` 是赋值：先求得右侧完整表达式 `state.time[index] - 0.45F`，再把结果写入左侧可修改对象 `pulse_time[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const double saw_cycles = std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 是声明并初始化：`const double saw_cycles` 建立局部对象 `saw_cycles`，右侧完整表达式 `std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `std::fmod`；写入/初始化 `pulse_time[index]`, `saw_cycles`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.20 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `const double square_cycles = std::fmod(`。

```cpp
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`saw_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `saw_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`125.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。|

**执行过程**

- `const double square_cycles = std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 是声明并初始化：`const double square_cycles` 建立局部对象 `square_cycles`，右侧完整表达式 `std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);` 是赋值：先求得右侧完整表达式 `static_cast<float>(2.0 * kPi * saw_cycles)`，再把结果写入左侧可修改对象 `phase[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `std::fmod`；写入/初始化 `square_cycles`, `phase[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.21 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`。

```cpp
        square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);
    }
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。|

**执行过程**

- `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);` 是赋值：先求得右侧完整表达式 `static_cast<float>(2.0 * kPi * square_cycles)`，再把结果写入左侧可修改对象 `square_phase[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `square_phase[index]`, `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `phase` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    auto d_time = cusignal::DeviceArray<float>::from_host(state.time);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_time = cusignal::DeviceArray<float>::from_host(state.time);` 是声明并初始化：`auto d_time` 建立局部对象 `d_time`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.time)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`；写入/初始化 `begin`, `d_time`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.23 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);`。

```cpp
    auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);
    auto d_phase = cusignal::DeviceArray<float>::from_host(phase);
    auto d_square_phase = cusignal::DeviceArray<float>::from_host(square_phase);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `auto d_pulse_time = cusignal::DeviceArray<float>::from_host(pulse_time);` 是声明并初始化：`auto d_pulse_time` 建立局部对象 `d_pulse_time`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(pulse_time)` 产生初值。
- `auto d_phase = cusignal::DeviceArray<float>::from_host(phase);` 是声明并初始化：`auto d_phase` 建立局部对象 `d_phase`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(phase)` 产生初值。
- `auto d_square_phase = cusignal::DeviceArray<float>::from_host(square_phase);` 是声明并初始化：`auto d_square_phase` 建立局部对象 `d_square_phase`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(square_phase)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `from_host`；写入/初始化 `d_pulse_time`, `d_phase`, `d_square_phase`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 phase：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `phase`；源码锚点 `evidence.h2d_ms += milliseconds(begin, Clock::now());`。

```cpp
    evidence.h2d_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`h2d_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.h2d_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.h2d_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `evidence.h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `phase` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.25 d_chirp：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_chirp`；源码锚点 `cusignal::DeviceArray<float> d_chirp(config.samples);`。

```cpp

    cusignal::DeviceArray<float> d_chirp(config.samples);
    cusignal::DeviceArray<float> d_gaussian(config.samples);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_chirp`、`config`、`samples`、`d_gaussian`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_chirp`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`d_gaussian`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_chirp(config.samples);` 声明 `d_chirp`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_chirp` 的函数，模板尖括号也不是比较或位移。
- `cusignal::DeviceArray<float> d_gaussian(config.samples);` 声明 `d_gaussian`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_gaussian` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `d_chirp`, `d_gaussian`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

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

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`20.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`70.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `cusignal::DeviceArray<double> d_sawtooth(config.samples);` 声明 `d_sawtooth`，其静态类型是 `cusignal::DeviceArray<double>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_sawtooth` 的函数，模板尖括号也不是比较或位移。
- `cusignal::DeviceArray<double> d_square(config.samples);` 声明 `d_square`，其静态类型是 `cusignal::DeviceArray<double>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_square` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["chirp"] = time_gpu([&] { cusignal::chirp_device( d_time, d_chirp, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F, 0.0F); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["chirp"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `d_sawtooth`, `d_square`, `time_gpu`, `cusignal::chirp_device`, `std::max`；写入/初始化 `operator_ms["chirp"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.27 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.operator_ms["gausspulse"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["gausspulse"] = time_gpu([&] {
        cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`170.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`0.20F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `evidence.operator_ms["gausspulse"] = time_gpu([&] { cusignal::gausspulse_device(d_pulse_time, d_gaussian, 170.0F, 0.20F); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["gausspulse"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::gausspulse_device`；写入/初始化 `operator_ms["gausspulse"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.28 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.operator_ms["sawtooth"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["sawtooth"] = time_gpu([&] {
        cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.65` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.65`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `evidence.operator_ms["sawtooth"] = time_gpu([&] { cusignal::sawtooth_device(d_phase, d_sawtooth, 0.65); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["sawtooth"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::sawtooth_device`；写入/初始化 `operator_ms["sawtooth"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.29 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.operator_ms["square"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["square"] = time_gpu([&] {
        cusignal::square_device(d_square_phase, d_square, 0.35);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.35`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `evidence.operator_ms["square"] = time_gpu([&] { cusignal::square_device(d_square_phase, d_square, 0.35); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["square"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::square_device`；写入/初始化 `operator_ms["square"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.30 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    state.target_waveform = d_chirp.to_host();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.target_waveform = d_chirp.to_host();` 是赋值：先求得右侧完整表达式 `d_chirp.to_host()`，再把结果写入左侧可修改对象 `state.target_waveform`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`；写入/初始化 `begin`, `target_waveform`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.31 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `const auto gaussian = d_gaussian.to_host();`。

```cpp
    const auto gaussian = d_gaussian.to_host();
    const auto sawtooth_double = d_sawtooth.to_host();
    const auto square_double = d_square.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `const auto gaussian = d_gaussian.to_host();` 是声明并初始化：`const auto gaussian` 建立局部对象 `gaussian`，右侧完整表达式 `d_gaussian.to_host()` 产生初值。
- `const auto sawtooth_double = d_sawtooth.to_host();` 是声明并初始化：`const auto sawtooth_double` 建立局部对象 `sawtooth_double`，右侧完整表达式 `d_sawtooth.to_host()` 产生初值。
- `const auto square_double = d_square.to_host();` 是声明并初始化：`const auto square_double` 建立局部对象 `square_double`，右侧完整表达式 `d_square.to_host()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `to_host`；写入/初始化 `gaussian`, `sawtooth_double`, `square_double`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_sawtooth` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.32 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `evidence.d2h_ms += milliseconds(begin, Clock::now());`。

```cpp
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    const auto sawtooth =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。
- `const auto sawtooth = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double);` 是声明并初始化：`const auto sawtooth` 建立局部对象 `sawtooth`，右侧完整表达式 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(sawtooth_double)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`, `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`；写入/初始化 `evidence.d2h_ms`, `sawtooth`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.33 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `const auto square =`。

```cpp
    const auto square =
        cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double);
    begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `square` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `square`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto square =`；值在 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const auto square = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double);` 是声明并初始化：`const auto square` 建立局部对象 `square`，右侧完整表达式 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(square_double)` 产生初值。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`, `Clock::now`；写入/初始化 `square`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.34 d_sawtooth：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_sawtooth`；源码锚点 `auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);`。

```cpp
    auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);
    auto d_square_float = cusignal::DeviceArray<float>::from_host(square);
    evidence.h2d_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `auto d_saw_float = cusignal::DeviceArray<float>::from_host(sawtooth);` 是声明并初始化：`auto d_saw_float` 建立局部对象 `d_saw_float`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(sawtooth)` 产生初值。
- `auto d_square_float = cusignal::DeviceArray<float>::from_host(square);` 是声明并初始化：`auto d_square_float` 建立局部对象 `d_square_float`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(square)` 产生初值。
- `evidence.h2d_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.h2d_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`；写入/初始化 `d_saw_float`, `d_square_float`, `evidence.h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.35 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);`。

```cpp

    cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);
    cusignal::DeviceArray<float> d_noise(config.samples), d_echo(config.samples);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_target`、`config`、`samples`、`d_interference`、`d_noise`、`d_echo`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_target`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 的 GPU 调用链|
|`d_noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_target(config.samples), d_interference(config.samples);` 声明 `d_target`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `config.samples), d_interference(config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_target` 的函数，模板尖括号也不是比较或位移。
- `cusignal::DeviceArray<float> d_noise(config.samples), d_echo(config.samples);` 声明 `d_noise`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `config.samples), d_echo(config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_noise` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `d_target`, `d_interference`, `d_noise`, `d_echo`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.36 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

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

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`gpu_echo_superposition`、`time_gpu`、`cusignal`、`cuda_utils`、`launch_1d_kernel`、`mix_echo_kernel`、`std`、`size_t`、`config`、`samples`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.operator_ms["gpu_echo_superposition"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( mix_echo_kernel, static_cast<std::size_t>(config.samples), d_chirp.data(), d_gaussian.data(), d_saw_float.data(), d_square_float.data(), d_target.data(), d_interference.data(), d_noise.data(), d_echo.data(), config.samples, config.target_delay_samples, config.noise_seed, config.target_weight, config.gaussian_weight, config.sawtooth_weight, config.square_weight, config.noise_amplitude); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["gpu_echo_superposition"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::cuda_utils::launch_1d_kernel`, `data`；写入/初始化 `operator_ms["gpu_echo_superposition"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。

### 4.38 d_target：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.compute_ms = std::accumulate(`。

```cpp
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `evidence.compute_ms = std::accumulate( evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0, [](double sum, const auto& item) { return sum + item.second; });` 调用 `std::accumulate` 从初值 `0.0` 开始遍历证据映射；lambda 把累计值 `sum` 与当前键值项的 `second`（毫秒值）相加，最终总和写入左侧字段。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `compute_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.39 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    state.target_component = d_target.to_host();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.target_component = d_target.to_host();` 是赋值：先求得右侧完整表达式 `d_target.to_host()`，再把结果写入左侧可修改对象 `state.target_component`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`；写入/初始化 `begin`, `target_component`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.40 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `state.interference_component = d_interference.to_host();`。

```cpp
    state.interference_component = d_interference.to_host();
    state.noise_component = d_noise.to_host();
    state.echo = d_echo.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `state.interference_component = d_interference.to_host();` 是赋值：先求得右侧完整表达式 `d_interference.to_host()`，再把结果写入左侧可修改对象 `state.interference_component`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.noise_component = d_noise.to_host();` 是赋值：先求得右侧完整表达式 `d_noise.to_host()`，再把结果写入左侧可修改对象 `state.noise_component`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.echo = d_echo.to_host();` 是赋值：先求得右侧完整表达式 `d_echo.to_host()`，再把结果写入左侧可修改对象 `state.echo`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `to_host`；写入/初始化 `interference_component`, `noise_component`, `echo`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.41 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.d2h_ms += milliseconds(begin, Clock::now());`。

```cpp
    evidence.d2h_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `evidence.d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.42 d_target：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`。

```cpp

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled()) {
        state.gaussian_waveform = gaussian;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `if (visualization_capture_enabled()) {` 是 `if` 条件语句：先把完整条件 `visualization_capture_enabled()` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `state.gaussian_waveform = gaussian;` 是赋值：先求得右侧完整表达式 `gaussian`，再把结果写入左侧可修改对象 `state.gaussian_waveform`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过检查 `visualization_capture_enabled()`；调用 `milliseconds`, `Clock::now`, `visualization_capture_enabled`；写入/初始化 `total_ms`, `gaussian_waveform`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于可视化数据链：它读取/导出数值结果，建立坐标、单位和图形对象；图片是解释性派生物，不能替代正式数值门禁。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.43 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `state.sawtooth_waveform = sawtooth;`。

```cpp
        state.sawtooth_waveform = sawtooth;
        state.square_waveform = square;
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`sawtooth_waveform`、`sawtooth`、`square_waveform`、`square`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.sawtooth_waveform = sawtooth;` 是赋值：先求得右侧完整表达式 `sawtooth`，再把结果写入左侧可修改对象 `state.sawtooth_waveform`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.square_waveform = square;` 是赋值：先求得右侧完整表达式 `square`，再把结果写入左侧可修改对象 `state.square_waveform`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `sawtooth_waveform`, `square_waveform`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.44 d_target：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `return evidence;`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.45 d_target：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu`；符号 `d_target`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_target` 所在的 `ZKX/Task2/task2_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.46 validation 文件依赖与私有作用域

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `validation 文件依赖与私有作用域`；源码锚点 `#include "accuracy/validation/step1_validation.h"`。

```cpp
#include "accuracy/validation/step1_validation.h"

#include "waveforms/waveforms_typed.h"

#include <cmath>

namespace task2::accuracy {
namespace {
```

**语法结构**

该代码块按源码顺序包含 5 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

当前块没有任务运行时变量；`#include` 和命名空间只建立编译期依赖与名称可见性。

**变量—公式—任务状态映射**

当前块没有任务运行时变量；`#include` 和命名空间只建立编译期依赖与名称可见性。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "accuracy/validation/step1_validation.h"` 在预处理阶段引入 "accuracy/validation/step1_validation.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "waveforms/waveforms_typed.h"` 在预处理阶段引入 "waveforms/waveforms_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include <cmath>` 在预处理阶段引入 <cmath> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task2::accuracy {` 打开命名空间 `task2::accuracy`；后续声明被放入该名称范围。
- `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：生成四种波形并模拟目标、干扰和噪声回波|
|业务角色|CPU reference 与验证入口|
|业务输入|Task2 Step1 配置或流水线状态|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|CPU 基准数组或验证数据|
|下一消费者|精度、扰动和正式证据比较|
|证明边界|本块只证明接口/作用域或验证调用存在；数值正确性需结合 CPU 核心和 GPU 输出比较。|

**任务语义**

该块属于 Step1 验证文件的依赖、私有作用域或公开收集入口，不冒充波形/回波核心计算。

**设计理由与替代方案**

- 文件级依赖和薄验证入口把 CPU reference 与比较框架连接起来；数学核心仍位于同文件的 helper 中。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.47 noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_bits`；源码锚点 `std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)`。

```cpp
std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`747796405U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x2C9277B5`；这是 PCG 系列常用 32 位 LCG 乘数。此处用于让相邻 index 在进入后续 avalanche 前先分散；它不是物理常数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）|
|`2891336453U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xAC564B05`；这是与上述乘数配套使用的 32 位增量。此处平移状态，避免 seed/index 的零值直接保持零；不是雷达参数。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）|

**执行过程**

- `std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed) {` 中的 `noise_bits` 是函数定义；名称前的 `std::uint32_t` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 是声明并初始化：`std::uint32_t value` 建立局部对象 `value`，右侧完整表达式 `seed ^ (index * 747796405U + 2891336453U)` 产生初值。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `noise_bits`；调用 `noise_bits`；写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 先执行乘加再与 seed 组合，是为了让相邻 `index` 不直接形成相邻整数状态，并避免零输入停留在简单零模式；这一步采用 PCG 常见 LCG 常数作为低成本初始打散，后续再由 Murmur 风格 avalanche 完成扩散。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.48 noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_bits`；源码锚点 `value ^= value >> 16;`。

```cpp
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`2246822519U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0x85EBCA77`。它是奇数，因此在模 $2^{32}$ 算术中可逆并能借助乘法进位扩散比特。它接近但不等于 MurmurHash3 `fmix32` 的 `0x85EBCA6B`；仓库源码未记录为何相差 `0x0C`，不能冒充原始 Murmur 常数。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`13`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`13` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|

**执行过程**

- `value ^= value >> 16;` 是复合赋值：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 运算，再把结果写回同一对象。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
- `value *= 2246822519U;` 是复合赋值：读取 `value` 的旧值，与 `2246822519U` 执行 `*` 运算，再把结果写回同一对象。
- `value ^= value >> 13;` 是复合赋值：读取 `value` 的旧值，与 `value >> 13` 执行 `^` 运算，再把结果写回同一对象。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.49 noise_bits：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_bits`；源码锚点 `value *= 3266489917U;`。

```cpp
    value *= 3266489917U;
    return value ^ (value >> 16);
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`3266489917U`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制为 `0xC2B2AE3D`。它同样是用于继续扩散的奇数乘数，但不等于 MurmurHash3 `fmix32` 的 `0xC2B2AE35`；仓库源码未记录为何相差 `0x08`。改变它会改变全部确定性噪声序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|
|`16`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|这是整数混合器的右移位数。`16` 位把高位信息折回较低位，再通过异或改变输出；`16,13,16` 的移位次序与 MurmurHash3 `fmix32` 相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：[MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）|

**执行过程**

- `value *= 3266489917U;` 是复合赋值：读取 `value` 的旧值，与 `3266489917U` 执行 `*` 运算，再把结果写回同一对象。
- `return value ^ (value >> 16);` 是返回语句：先求值 `value ^ (value >> 16)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `value`；返回 `value ^ (value >> 16)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.50 noise_sample：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_sample`；源码锚点 `float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)`。

```cpp

float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * amplitude;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`0xffffU`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。|
|`32767.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。|

**执行过程**

- `float noise_sample(std::uint32_t index, std::uint32_t seed, float amplitude) {` 中的 `noise_sample` 是函数定义；名称前的 `float` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed, float amplitude` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * amplitude;` 是返回语句：先求值 `(static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * amplitude`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `noise_sample`；调用 `noise_sample`, `return`, `noise_bits`；返回 `(static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * amplitude`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 先用 `0xFFFF` 只保留低 16 位，是为了得到固定的 65536 个离散状态；随后除以半跨度 `32767.5` 并减 1，把它线性映射为以 0 为中心的 $[-1,1]$。最后乘 `noise_std` 才得到任务噪声幅度。可替代为 32 位到浮点的完整映射或标准 RNG，但那会改变历史 CPU/GPU 确定性序列和精度证据。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.51 noise_sample：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `noise_sample`；源码锚点 `}`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.52 synthesize_cpu：完成一个连续的数据处理动作

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

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<float> synthesize_cpu( const std::vector<float>& chirp, const std::vector<float>& gaussian, const std::vector<float>& sawtooth, const std::vector<float>& square, std::uint32_t seed, float target_weight, std::vector<float>* target, std::vector<float>* interference, std::vector<float>* noise, const TaskConfig& config) {` 中的 `synthesize_cpu` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& chirp, const std::vector<float>& gaussian, const std::vector<float>& sawtooth, const std::vector<float>& square, std::uint32_t seed, float target_weight, std::vector<float>* target, std::vector<float>* interference, std::vector<float>* noise, const TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<float> echo(chirp.size());` 声明 `echo`，其静态类型是 `std::vector<float>`，并用 `chirp.size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `echo` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `synthesize_cpu`；调用 `synthesize_cpu`, `echo`, `size`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.53 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `target->assign(chirp.size(), 0.0F);`。

```cpp
    target->assign(chirp.size(), 0.0F);
    interference->assign(chirp.size(), 0.0F);
    noise->assign(chirp.size(), 0.0F);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `target->assign(chirp.size(), 0.0F);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
- `interference->assign(chirp.size(), 0.0F);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
- `noise->assign(chirp.size(), 0.0F);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `assign`, `size`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.54 synthesize_cpu：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `for (std::size_t index = 0; index < chirp.size(); ++index) {`。

```cpp
    for (std::size_t index = 0; index < chirp.size(); ++index) {
        (*target)[index] = index >= static_cast<std::size_t>(config.target_delay_samples)
            ? target_weight * chirp[index - config.target_delay_samples] : 0.0F;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `for (std::size_t index = 0; index < chirp.size(); ++index) {` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index < chirp.size()`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `(*target)[index] = index >= static_cast<std::size_t>(config.target_delay_samples) ? target_weight * chirp[index - config.target_delay_samples] : 0.0F;` 是赋值：先求得右侧完整表达式 `index >= static_cast<std::size_t>(config.target_delay_samples) ? target_weight * chirp[index - config.target_delay_samples] : 0.0F`，再把结果写入左侧可修改对象 `(*target)[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `size`；写入/初始化 `(*target)[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.55 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `(*interference)[index] = config.gaussian_weight * gaussian[index] +`。

```cpp
        (*interference)[index] = config.gaussian_weight * gaussian[index] +
            config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index];
        (*noise)[index] = noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `interference`、`index`、`config`、`gaussian_weight`、`gaussian`、`sawtooth_weight`、`sawtooth`、`square_weight`、`square`、`noise`、`noise_sample`、`std`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `(*interference)[index] = config.gaussian_weight * gaussian[index] + config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index];` 是赋值：先求得右侧完整表达式 `config.gaussian_weight * gaussian[index] + config.sawtooth_weight * sawtooth[index] + config.square_weight * square[index]`，再把结果写入左侧可修改对象 `(*interference)[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `(*noise)[index] = noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude);` 是赋值：先求得右侧完整表达式 `noise_sample(static_cast<std::uint32_t>(index), seed, config.noise_amplitude)`，再把结果写入左侧可修改对象 `(*noise)[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `noise_sample`；写入/初始化 `(*interference)[index]`, `(*noise)[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.56 synthesize_cpu：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];`。

```cpp
        echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];
    }
    return echo;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `echo[index] = (*target)[index] + (*interference)[index] + (*noise)[index];` 是赋值：先求得右侧完整表达式 `(*target)[index] + (*interference)[index] + (*noise)[index]`，再把结果写入左侧可修改对象 `echo[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `return echo;` 是返回语句：先求值 `echo`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过写入/初始化 `echo[index]`；返回 `echo`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.57 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `}`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.58 synthesize_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `synthesize_cpu`；源码锚点 `}  // namespace`。

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
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.59 generate_step1_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `generate_step1_cpu_reference`；源码锚点 `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)`。

```cpp
Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)
{
    Step1ValidationData data;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `data` 由 `Step1ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`generate_step1_cpu_reference`|生成当前名称所描述数据的 helper/结果|对应生成公式的输出端|由输入参数构造数据，供后续 step 或测试使用|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config) {` 中的 `generate_step1_cpu_reference` 是函数定义；名称前的 `Step1ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step1ValidationData data;` 是对象声明：类型 `Step1ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过定义/声明 `generate_step1_cpu_reference`；调用 `generate_step1_cpu_reference`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.60 time：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `std::vector<float> time(config.samples), pulse_time(config.samples);`。

```cpp
    std::vector<float> time(config.samples), pulse_time(config.samples);
    std::vector<float> phase(config.samples), square_phase(config.samples);
    for (int index = 0; index < config.samples; ++index) {
        time[index] = static_cast<float>(index) / config.sample_rate_hz;
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> time(config.samples), pulse_time(config.samples);` 声明 `time`，其静态类型是 `std::vector<float>`，并用 `config.samples), pulse_time(config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `time` 的函数，模板尖括号也不是比较或位移。
- `std::vector<float> phase(config.samples), square_phase(config.samples);` 声明 `phase`，其静态类型是 `std::vector<float>`，并用 `config.samples), square_phase(config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `phase` 的函数，模板尖括号也不是比较或位移。
- `for (int index = 0; index < config.samples; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index < config.samples`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `time[index] = static_cast<float>(index) / config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `static_cast<float>(index) / config.sample_rate_hz`，再把结果写入左侧可修改对象 `time[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `time`, `pulse_time`, `phase`, `square_phase`；写入/初始化 `time[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.61 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `pulse_time[index] = time[index] - 0.45F;`。

```cpp
        pulse_time[index] = time[index] - 0.45F;
        const double saw_cycles = std::fmod(
            190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.45F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`190.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `pulse_time[index] = time[index] - 0.45F;` 是赋值：先求得右侧完整表达式 `time[index] - 0.45F`，再把结果写入左侧可修改对象 `pulse_time[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const double saw_cycles = std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 是声明并初始化：`const double saw_cycles` 建立局部对象 `saw_cycles`，右侧完整表达式 `std::fmod( 190.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `std::fmod`；写入/初始化 `pulse_time[index]`, `saw_cycles`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.62 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `const double square_cycles = std::fmod(`。

```cpp
        const double square_cycles = std::fmod(
            125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);
        phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`saw_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `saw_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`125.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。|

**执行过程**

- `const double square_cycles = std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0);` 是声明并初始化：`const double square_cycles` 建立局部对象 `square_cycles`，右侧完整表达式 `std::fmod( 125.0 * static_cast<double>(index) / config.sample_rate_hz, 1.0)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `phase[index] = static_cast<float>(2.0 * kPi * saw_cycles);` 是赋值：先求得右侧完整表达式 `static_cast<float>(2.0 * kPi * saw_cycles)`，再把结果写入左侧可修改对象 `phase[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `std::fmod`；写入/初始化 `square_cycles`, `phase[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.63 time：完成一个连续的数据处理动作

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

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|
|`square_cycles`|当前函数或调用的参数名称，接收调用者绑定的输入 `square_cycles`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。|
|`20.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`70.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `square_phase[index] = static_cast<float>(2.0 * kPi * square_cycles);` 是赋值：先求得右侧完整表达式 `static_cast<float>(2.0 * kPi * square_cycles)`，再把结果写入左侧可修改对象 `square_phase[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `data.chirp_cpu = cusignal::chirp_typed_cpu( time, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F, 0.0F);` 是赋值：先求得右侧完整表达式 `cusignal::chirp_typed_cpu( time, 20.0F, std::max(static_cast<float>(config.samples) / config.sample_rate_hz, 1.0F), 70.0F, 0.0F)`，再把结果写入左侧可修改对象 `data.chirp_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `cusignal::chirp_typed_cpu`, `std::max`；写入/初始化 `square_phase[index]`, `chirp_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.64 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(`。

```cpp
    const auto gaussian_cpu = cusignal::gausspulse_typed_cpu(
        pulse_time, 170.0F, 0.20F);
    const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::sawtooth_typed_cpu(phase, 0.65));
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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
|`170.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`0.20F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`0.65`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `const auto gaussian_cpu = cusignal::gausspulse_typed_cpu( pulse_time, 170.0F, 0.20F);` 是声明并初始化：`const auto gaussian_cpu` 建立局部对象 `gaussian_cpu`，右侧完整表达式 `cusignal::gausspulse_typed_cpu( pulse_time, 170.0F, 0.20F)` 产生初值。
- `const auto sawtooth_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::sawtooth_typed_cpu(phase, 0.65));` 是声明并初始化：`const auto sawtooth_cpu` 建立局部对象 `sawtooth_cpu`，右侧完整表达式 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::sawtooth_typed_cpu(phase, 0.65))` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `cusignal::gausspulse_typed_cpu`, `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`, `cusignal::sawtooth_typed_cpu`；写入/初始化 `gaussian_cpu`, `sawtooth_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.65 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(`。

```cpp
    const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::square_typed_cpu(square_phase, 0.35));
    data.echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed,
        config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `square_cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `0.35` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`square_cpu`|CPU reference 对应量|与去掉 `_cpu` 后的任务量采用同一数学定义|供 GPU/CPU 精度 comparison 使用|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.35`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `const auto square_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::square_typed_cpu(square_phase, 0.35));` 是声明并初始化：`const auto square_cpu` 建立局部对象 `square_cpu`，右侧完整表达式 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::square_typed_cpu(square_phase, 0.35))` 产生初值。
- `data.echo_cpu = synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed, config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config);` 是赋值：先求得右侧完整表达式 `synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed, config.target_weight, &data.target_cpu, &data.interference_cpu, &data.noise_cpu, config)`，再把结果写入左侧可修改对象 `data.echo_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`, `cusignal::square_typed_cpu`, `synthesize_cpu`；写入/初始化 `square_cpu`, `echo_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.66 time：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `std::vector<float> changed_target, changed_interference, changed_noise;`。

```cpp
    std::vector<float> changed_target, changed_interference, changed_noise;
    data.changed_echo_cpu = synthesize_cpu(
        data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U,
        std::max(0.0F, config.target_weight - 0.05F),
        &changed_target, &changed_interference, &changed_noise, config);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `changed_target` 由 `std::vector<float>` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `1U` 是数值字面量；U 指定 unsigned int 起始类型；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.05F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`changed_target`|当前表达式读取或传递的工程名称 `changed_target`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `std::vector<float> changed_target, changed_interference, changed_noise;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化样本数、采样率、四种波形参数/权重、目标延迟、噪声幅度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1U`|`U` 使整数字面量从 unsigned int 候选类型开始|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`0.05F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `std::vector<float> changed_target, changed_interference, changed_noise;` 是对象声明：类型 `std::vector<float>` 应用于名称 `changed_target`、`changed_interference`、`changed_noise`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `data.changed_echo_cpu = synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U, std::max(0.0F, config.target_weight - 0.05F), &changed_target, &changed_interference, &changed_noise, config);` 是赋值：先求得右侧完整表达式 `synthesize_cpu( data.chirp_cpu, gaussian_cpu, sawtooth_cpu, square_cpu, config.noise_seed + 1U, std::max(0.0F, config.target_weight - 0.05F), &changed_target, &changed_interference, &changed_noise, config)`，再把结果写入左侧可修改对象 `data.changed_echo_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过调用 `synthesize_cpu`, `std::max`；写入/初始化 `changed_echo_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.67 time：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `time`；源码锚点 `return data;`。

```cpp
    return data;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `data` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：波形生成：至少三种附录波形叠加，并模拟目标、干扰和噪声回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样数、采样率、各波形权重、目标延迟、干扰和噪声参数|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|组合参考波形、目标/干扰/噪声分量及 `echo`|
|下一消费者|`echo` 交给 T2-R2 滤波|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.68 collect_step1_validation：连接正式状态与 CPU reference

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step1_validation.cpp`；符号 `collect_step1_validation`；源码锚点 `Step1ValidationData collect_step1_validation(const PipelineState& state)`。

```cpp
Step1ValidationData collect_step1_validation(const PipelineState& state)
{
    return generate_step1_cpu_reference(state.config);
}

}  // namespace task2::accuracy
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：函数定义或声明、`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state`：只读流水线状态引用；本入口读取其中的 `config`。
- `collect_step1_validation`：验证收集入口，返回 `Step1ValidationData`。

**变量—公式—任务状态映射**

- `state`：只读流水线状态引用；本入口读取其中的 `config`。
- `collect_step1_validation`：验证收集入口，返回 `Step1ValidationData`。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step1ValidationData collect_step1_validation(const PipelineState& state) {` 中的 `collect_step1_validation` 是函数定义；名称前的 `Step1ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return generate_step1_cpu_reference(state.config);` 是返回语句：先求值 `generate_step1_cpu_reference(state.config)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `}  // namespace task2::accuracy` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R1：生成四种波形并模拟目标、干扰和噪声回波|
|业务角色|CPU reference 与验证入口|
|业务输入|Task2 Step1 配置或流水线状态|
|代码如何落实|本块通过定义/声明 `collect_step1_validation`, `generate_step1_cpu_reference`；调用 `collect_step1_validation`, `generate_step1_cpu_reference`；返回 `generate_step1_cpu_reference(state.config)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|CPU 基准数组或验证数据|
|下一消费者|精度、扰动和正式证据比较|
|证明边界|本块只证明接口/作用域或验证调用存在；数值正确性需结合 CPU 核心和 GPU 输出比较。|

**任务语义**

该块属于 Step1 验证文件的依赖、私有作用域或公开收集入口，不冒充波形/回波核心计算。

**设计理由与替代方案**

- 文件级依赖和薄验证入口把 CPU reference 与比较框架连接起来；数学核心仍位于同文件的 helper 中。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
