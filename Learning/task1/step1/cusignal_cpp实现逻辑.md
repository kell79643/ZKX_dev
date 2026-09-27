# Task1 Step1 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task1 Step1 的任务接口、LFM 生成、延迟—多普勒回波与确定性噪声模拟、GPU kernel/Host 调度，以及数学等价的 CPU reference；设备数组、通用 launch、计时与错误宏只解释本步调用边界。

- 本文件绑定的学习快照 SHA：`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`
- 2026-08-19 复核时工作区 HEAD：`54a4a93083dc137e685345e10715399cf9e56b1e`；本文件列出的源码 blob 与学习快照一致。
- 分支：`final-prep/benchmark-evidence-v1`
- 提交时间：`2026-08-18T20:22:35+08:00`
- 每段源码另绑定实际SHA。
- tracked源码状态：clean
- 本轮只静态阅读。

## 2. 源码覆盖与唯一归属

|源码|本文件负责的符号|SHA|
|---|---|---|
|`ZKX/Task1/task1_gpu_cpu/step1/step1.h`|`run_step1` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/step1/step1.cu`|`noise_bits`、`uniform_noise`、`generate_lfm_kernel`、`simulate_echo_kernel`、`run_step1`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`|`cpu_noise_bits`、`cpu_uniform_noise`、`waveform_cpu`、`echo_cpu`、`generate_step1_cpu_reference`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task1/task1_gpu_cpu/step1/step1.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
//  `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task1 { StepEvidence run_step1(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence
//  run_step1(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。
//
//  【变量—数学符号—数据流】
//  run_step1：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step1(PipelineState& state); }
```

### 3.2 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step1.h"` 在预处理阶段引入 "step1.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/kernel_launch.h"` 在预处理阶段引入 "cuda_utils/kernel_launch.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
#include "step1.h"

#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"

// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task1 {` 打开命名空间 `task1`；后续声明被放入该名称范围。
//  `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
#include <numeric>

namespace task1 {
namespace {

// <学习注释：语义块：noise_bits：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `__device__ std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed) {` 中的 `noise_bits`
//  是函数定义；名称前的 `__device__ std::uint32_t` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t
//  seed` 是形参声明。 `__device__` 表明该 helper 在 GPU device 代码中调用。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 是声明并初始化：`std::uint32_t value` 建立局部对象
//  `value`，右侧完整表达式 `seed ^ (index * 747796405U + 2891336453U)` 产生初值。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。
//
//  【变量—数学符号—数据流】
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
//  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
//
//  【数字常量】
//  747796405U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0x2C9277B5；这是 PCG 系列常用 32 位 LCG 乘数。此处用于让相邻 index 在进入后续
//  avalanche 前先分散；它不是物理常数。来源：PCG 原始技术报告（访问日期：2026-08-19）
//  2891336453U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0xAC564B05；这是与上述乘数配套使用的 32 位增量。此处平移状态，避免 seed/index
//  的零值直接保持零；不是雷达参数。来源：PCG 原始技术报告（访问日期：2026-08-19）
//
//  【数学/物理公式对照】
//  令 `index=n`、`seed=s`。初值 $v_0=s\oplus(747796405n+2891336453)\pmod{2^{32}}$；后续右移—异或和奇数乘法都在 `uint32_t` 的模 $2^{32}$ 算术中扩散比特，得到确定性的 32 位 $v$。这是无状态整数混合，不是完整 PCG 状态机，也不是物理噪声分布。
//  `U` 指定无符号整数字面量；`^` 是按位异或而非乘方，`>>` 对无符号值高位补零。移位把高位信息折叠进低位，乘法借助进位传播差异；删改常量或移位数会改变全部 CPU/GPU 可复现序列。
//
//  【为什么这样设计】
//  每个 CUDA 线程或 CPU 迭代仅凭 `(seed,index)` 即可重建相同结果，无需共享随机状态，因此适合逐元素精度比较。常量身份只按仓库和可靠资料可核实的范围表述，不把“借用常数”夸大成完整算法。
//
//  【初学者易错点】
//  在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。>
__device__ std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
    // <学习注释：语义块：noise_bits：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `value ^= value >> 16;` 是复合赋值：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 运算，再把结果写回同一对象。 `^`
    //  是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
    //  `value *= 2246822519U;` 是复合赋值：读取 `value` 的旧值，与 `2246822519U` 执行 `*` 运算，再把结果写回同一对象。
    //  `value ^= value >> 13;` 是复合赋值：读取 `value` 的旧值，与 `value >> 13` 执行 `^` 运算，再把结果写回同一对象。 `^`
    //  是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
    //
    //  【变量—数学符号—数据流】
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //
    //  【数字常量】
    //  16：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；这是整数混合器的右移位数。16 位把高位信息折回较低位，再通过异或改变输出；16,13,16 的移位次序与 MurmurHash3 fmix32
    //  相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //
    //  2246822519U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0x85EBCA77。它是奇数，因此在模 $2^{32}$ 算术中可逆并能借助乘法进位扩散比特。它接近但不等于
    //  MurmurHash3 fmix32 的 0x85EBCA6B；仓库源码未记录为何相差 0x0C，不能冒充原始 Murmur 常数。对照来源：MurmurHash3 原始 fmix32
    //  实现（访问日期：2026-08-19）
    //  13：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；这是整数混合器的右移位数。13 位把高位信息折回较低位，再通过异或改变输出；16,13,16 的移位次序与 MurmurHash3 fmix32
    //  相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //
    //
    //  【数学/物理公式对照】
    //  令 `index=n`、`seed=s`。初值 $v_0=s\oplus(747796405n+2891336453)\pmod{2^{32}}$；后续右移—异或和奇数乘法都在 `uint32_t` 的模 $2^{32}$ 算术中扩散比特，得到确定性的 32 位 $v$。这是无状态整数混合，不是完整 PCG 状态机，也不是物理噪声分布。
    //  `U` 指定无符号整数字面量；`^` 是按位异或而非乘方，`>>` 对无符号值高位补零。移位把高位信息折叠进低位，乘法借助进位传播差异；删改常量或移位数会改变全部 CPU/GPU 可复现序列。
    //
    //  【为什么这样设计】
    //  每个 CUDA 线程或 CPU 迭代仅凭 `(seed,index)` 即可重建相同结果，无需共享随机状态，因此适合逐元素精度比较。常量身份只按仓库和可靠资料可核实的范围表述，不把“借用常数”夸大成完整算法。
    //
    //  【初学者易错点】
    //  在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。>
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    // <学习注释：语义块：noise_bits：形成并返回当前结果。
    //
    //  【执行方法】
    //  `value *= 3266489917U;` 是复合赋值：读取 `value` 的旧值，与 `3266489917U` 执行 `*` 运算，再把结果写回同一对象。
    //  `return value ^ (value >> 16);` 是返回语句：先求值 `value ^ (value >> 16)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //
    //  【数字常量】
    //  3266489917U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0xC2B2AE3D。它同样是用于继续扩散的奇数乘数，但不等于 MurmurHash3 fmix32 的
    //  0xC2B2AE35；仓库源码未记录为何相差 0x08。改变它会改变全部确定性噪声序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //  16：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；这是整数混合器的右移位数。16 位把高位信息折回较低位，再通过异或改变输出；16,13,16 的移位次序与 MurmurHash3 fmix32
    //  相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //
    //
    //  【数学/物理公式对照】
    //  令 `index=n`、`seed=s`。初值 $v_0=s\oplus(747796405n+2891336453)\pmod{2^{32}}$；后续右移—异或和奇数乘法都在 `uint32_t` 的模 $2^{32}$ 算术中扩散比特，得到确定性的 32 位 $v$。这是无状态整数混合，不是完整 PCG 状态机，也不是物理噪声分布。
    //  `U` 指定无符号整数字面量；`^` 是按位异或而非乘方，`>>` 对无符号值高位补零。移位把高位信息折叠进低位，乘法借助进位传播差异；删改常量或移位数会改变全部 CPU/GPU 可复现序列。
    //
    //  【为什么这样设计】
    //  每个 CUDA 线程或 CPU 迭代仅凭 `(seed,index)` 即可重建相同结果，无需共享随机状态，因此适合逐元素精度比较。常量身份只按仓库和可靠资料可核实的范围表述，不把“借用常数”夸大成完整算法。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
    //  在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。>
    value *= 3266489917U;
    return value ^ (value >> 16);
}

// <学习注释：语义块：uniform_noise：形成并返回当前结果。
//
//  【执行方法】
//  `__device__ float uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std) {` 中的
//  `uniform_noise` 是函数定义；名称前的 `__device__ float` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index,
//  std::uint32_t seed, float noise_std` 是形参声明。 `__device__` 表明该 helper 在 GPU device 代码中调用。 末尾 `{`
//  打开函数体；这里定义函数但不会在定义时自动执行。
//  `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std;` 是返回语句：先求值
//  `(static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) *
//  noise_std`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
//
//  【变量—数学符号—数据流】
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
//  noise_std：噪声幅度尺度/标准差参数；记作 $\sigma$；缩放单位伪随机样本
//
//  【数字常量】
//  0xffffU：U 使整数字面量从 unsigned int 候选类型开始；十六进制 0xFFFF 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16
//  位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。
//  32767.5F：F 使浮点字面量为 float；不带后缀默认是 double；它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行
//  (u-32767.5)/32767.5 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。
//  1.0F：F 使浮点字面量为 float；不带后缀默认是 double；这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1
//  而不是 0。
//
//  【数学/物理公式对照】
//  令 $b=\texttt{noise_bits}(n,s)\mathbin{\&}65535\in[0,65535]$。`32767.5F=65535/2`，故 $u=b/32767.5-1\in[-1,1]$，返回 $w=\sigma u$，其中 `noise_std=σ`。先 `static_cast<float>` 再除可避免整数除法；仅凭本代码不能把 `noise_std` 解释成高斯标准差。
//
//  【为什么这样设计】
//  同一有界零中心映射同时用于 CPU/GPU，使比较不受随机库和状态推进顺序影响；它服务可复现模拟，不等价于经过统计检验的高斯热噪声发生器。
//
//  【初学者易错点】
//  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
//  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
__device__ float uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * noise_std;
// <学习注释：语义块：uniform_noise：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}

// <学习注释：语义块：generate_lfm_kernel：声明 CUDA kernel 并建立线程入口。
//
//  【执行方法】
//  `__global__ void generate_lfm_kernel( cusignal::TypedComplex<float>* waveform, int pulse_samples, float
//  sample_rate_hz, float bandwidth_hz) {` 中的 `generate_lfm_kernel` 是函数定义；名称前的 `__global__ void`
//  包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `cusignal::TypedComplex<float>* waveform, int pulse_samples, float
//  sample_rate_hz, float bandwidth_hz` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾
//  `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index`
//  建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。
//  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x`
//  把块号和块内线程号映射为一维全局线程索引。
//
//  【变量—数学符号—数据流】
//  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
//  pulse_samples：发射脉冲波形包含的快时间采样数；记作 $N_p$；由配置和采样率决定；限定 waveform 有效区间并形成脉宽 $T=N_p/f_s$
//  sample_rate_hz：采样率，单位 Hz；记作 $f_s$；把离散样本索引换算为秒
//  bandwidth_hz：扫频带宽，单位 Hz；记作 $B$；与脉宽共同决定 LFM 调频斜率 $k=B/T$
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//
//  【数学/物理公式对照】
//  `pulse_samples=L`、`sample_rate_hz=f_s`、`bandwidth_hz=B`、`index=n`。$T=L/f_s$，$t_n=n/f_s-T/2$，$K=B/T$，$\phi_n=\pi Kt_n^2$，`{cosf(phase),sinf(phase)}` 对应 $s[n]=e^{j\phi_n}$。每个有效线程只写一个复采样点；越界线程直接退出。
//
//  【为什么这样设计】
//  复指数基带 LFM 的瞬时频率为 $f(t)=\frac1{2\pi}d\phi/dt=Kt$；居中时间轴使扫频关于脉冲中心对称。逐采样点相互独立，适合一维 CUDA 并行。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
__global__ void generate_lfm_kernel(
    cusignal::TypedComplex<float>* waveform, int pulse_samples,
    float sample_rate_hz, float bandwidth_hz)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // <学习注释：语义块：generate_lfm_kernel：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (index >= pulse_samples) return;` 是 `if` 条件语句：先把完整条件 `index >= pulse_samples` 求值并转换为布尔值。 条件为真时执行
    //  `return`，即结束当前函数或当前 CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
    //  `const float pulse_width = static_cast<float>(pulse_samples) / sample_rate_hz;` 是声明并初始化：`const float
    //  pulse_width` 建立局部对象 `pulse_width`，右侧完整表达式 `static_cast<float>(pulse_samples) / sample_rate_hz` 产生初值。
    //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `const float time = static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width;` 是声明并初始化：`const float
    //  time` 建立局部对象 `time`，右侧完整表达式 `static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width` 产生初值。
    //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  pulse_width：脉冲持续时间，单位 s；记作 $T=N_p/f_s$；用于 LFM 相位和距离分辨率关系
    //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //
    //  【数字常量】
    //  0.5F：F 使浮点字面量为 float；不带后缀默认是 double；表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。
    //
    //  【数学/物理公式对照】
    //  `pulse_samples=L`、`sample_rate_hz=f_s`、`bandwidth_hz=B`、`index=n`。$T=L/f_s$，$t_n=n/f_s-T/2$，$K=B/T$，$\phi_n=\pi Kt_n^2$，`{cosf(phase),sinf(phase)}` 对应 $s[n]=e^{j\phi_n}$。每个有效线程只写一个复采样点；越界线程直接退出。
    //
    //  【为什么这样设计】
    //  复指数基带 LFM 的瞬时频率为 $f(t)=\frac1{2\pi}d\phi/dt=Kt$；居中时间轴使扫频关于脉冲中心对称。逐采样点相互独立，适合一维 CUDA 并行。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
    if (index >= pulse_samples) return;
    const float pulse_width = static_cast<float>(pulse_samples) / sample_rate_hz;
    const float time = static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width;
    // <学习注释：语义块：generate_lfm_kernel：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const float phase = kPi * (bandwidth_hz / pulse_width) * time * time;` 是声明并初始化：`const float phase` 建立局部对象
    //  `phase`，右侧完整表达式 `kPi * (bandwidth_hz / pulse_width) * time * time` 产生初值。
    //  `waveform[index] = {cosf(phase), sinf(phase)};` 是赋值：先求得右侧完整表达式 `{cosf(phase), sinf(phase)}`，再把结果写入左侧可修改对象
    //  `waveform[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  phase：复指数的相位，单位 rad；记作 $\phi$；通过 cos/sin 形成复波形或多普勒旋转
    //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //
    //  【数学/物理公式对照】
    //  `pulse_samples=L`、`sample_rate_hz=f_s`、`bandwidth_hz=B`、`index=n`。$T=L/f_s$，$t_n=n/f_s-T/2$，$K=B/T$，$\phi_n=\pi Kt_n^2$，`{cosf(phase),sinf(phase)}` 对应 $s[n]=e^{j\phi_n}$。每个有效线程只写一个复采样点；越界线程直接退出。
    //
    //  【为什么这样设计】
    //  复指数基带 LFM 的瞬时频率为 $f(t)=\frac1{2\pi}d\phi/dt=Kt$；居中时间轴使扫频关于脉冲中心对称。逐采样点相互独立，适合一维 CUDA 并行。
    //
    //  【初学者易错点】
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
    const float phase = kPi * (bandwidth_hz / pulse_width) * time * time;
    waveform[index] = {cosf(phase), sinf(phase)};
}

// <学习注释：语义块：simulate_echo_kernel：声明 CUDA kernel 并建立线程入口。
//
//  【执行方法】
//  `__global__ void simulate_echo_kernel( const cusignal::TypedComplex<float>* waveform,
//  cusignal::TypedComplex<float>* noiseless, cusignal::TypedComplex<float>* noise,
//  cusignal::TypedComplex<float>* echo, int count, int delay, float doppler_hz, std::uint32_t seed, int
//  samples_per_pulse, int pulse_samples, float prf_hz, float sample_rate_hz, float target_amplitude, float
//  noise_std) {` 中的 `simulate_echo_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的
//  `const cusignal::TypedComplex<float>* waveform, cusignal::TypedComplex<float>* noiseless,
//  cusignal::TypedComplex<float>* noise, cusignal::TypedComplex<float>* echo, int count, int delay, float
//  doppler_hz, std::uint32_t seed, int samples_per_pulse, int pulse_samples, float prf_hz, float
//  sample_rate_hz, float target_amplitude, float noise_std` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU
//  上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index`
//  建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。
//  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x`
//  把块号和块内线程号映射为一维全局线程索引。
//
//  【变量—数学符号—数据流】
//  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
//  noiseless：未加噪目标回波；记作 $x_0[p,n]$；用于区分信号与噪声贡献及测试证据
//  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
//  echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
//  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
//  delay：目标离散延迟，单位 sample；记作 $d$；回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$
//  doppler_hz：目标多普勒频移，单位 Hz；记作 $f_D$；进入 $e^{j2\pi f_Dt}$
//  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
//  samples_per_pulse：每个接收脉冲槽保存的快时间采样数；记作 $N_s$；用于二维 shape $P\times N_s$ 以及展平索引分解
//  pulse_samples：发射脉冲波形包含的快时间采样数；记作 $N_p$；由配置和采样率决定；限定 waveform 有效区间并形成脉宽 $T=N_p/f_s$
//  prf_hz：脉冲重复频率，单位 Hz；记作 $f_r$ 或 PRF；把脉冲编号换算为慢时间 $p/f_r$
//  sample_rate_hz：采样率，单位 Hz；记作 $f_s$；把离散样本索引换算为秒
//  target_amplitude：目标复回波幅度系数；记作 $A$；缩放无噪目标回波
//  noise_std：噪声幅度尺度/标准差参数；记作 $\sigma$；缩放单位伪随机样本
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//
//  【数学/物理公式对照】
//  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
//
//  【为什么这样设计】
//  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
__global__ void simulate_echo_kernel(
    const cusignal::TypedComplex<float>* waveform,
    cusignal::TypedComplex<float>* noiseless,
    cusignal::TypedComplex<float>* noise,
    cusignal::TypedComplex<float>* echo,
    int count, int delay, float doppler_hz, std::uint32_t seed,
    int samples_per_pulse, int pulse_samples, float prf_hz,
    float sample_rate_hz, float target_amplitude, float noise_std)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // <学习注释：语义块：simulate_echo_kernel：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前
    //  CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
    //  `const int pulse = index / samples_per_pulse;` 是声明并初始化：`const int pulse` 建立局部对象 `pulse`，右侧完整表达式 `index /
    //  samples_per_pulse` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
    //  `const int sample = index % samples_per_pulse;` 是声明并初始化：`const int sample` 建立局部对象 `sample`，右侧完整表达式 `index
    //  % samples_per_pulse` 产生初值。 `%` 取得整数除法余数，用来把展平索引还原为当前行/脉冲内的位置。
    //
    //  【变量—数学符号—数据流】
    //  pulse：当前慢时间脉冲编号；记作 $p=\lfloor i/N_s\rfloor$；由展平索引整除每脉冲采样数得到
    //  sample：当前脉冲内快时间采样编号；记作 $n=i\bmod N_s$；由展平索引取余得到
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。
    //  取余的分母不能为零；这里依赖每脉冲采样数/维度已通过配置校验。>
    if (index >= count) return;
    const int pulse = index / samples_per_pulse;
    const int sample = index % samples_per_pulse;
    // <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `float target_real = 0.0F;` 是声明并初始化：`float target_real` 建立局部对象 `target_real`，右侧完整表达式 `0.0F` 产生初值。
    //  `float target_imag = 0.0F;` 是声明并初始化：`float target_imag` 建立局部对象 `target_imag`，右侧完整表达式 `0.0F` 产生初值。
    //  `const int source = sample - delay;` 是声明并初始化：`const int source` 建立局部对象 `source`，右侧完整表达式 `sample - delay`
    //  产生初值。
    //
    //  【变量—数学符号—数据流】
    //  target_real：目标回波复数的实部；记作 $\Re\{x_0[p,n]\}$；由参考波形和多普勒旋转计算，写入 noiseless/echo
    //  target_imag：目标回波复数的虚部；记作 $\Im\{x_0[p,n]\}$；与 target_real 组成复回波
    //  source：施加延迟前回读的波形采样位置；记作 $n-d$；只有落在波形有效区间时才形成目标回波
    //
    //  【数字常量】
    //  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
    //
    //  【为什么这样设计】
    //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    float target_real = 0.0F;
    float target_imag = 0.0F;
    const int source = sample - delay;
    // <学习注释：语义块：simulate_echo_kernel：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (source >= 0 && source < pulse_samples) {` 是 `if` 条件语句：先把完整条件 `source >= 0 && source < pulse_samples`
    //  求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `const auto base = waveform[source];` 是声明并初始化：`const auto base` 建立局部对象 `base`，右侧完整表达式 `waveform[source]`
    //  产生初值。
    //  `const float time = static_cast<float>(pulse) / prf_hz + static_cast<float>(sample) / sample_rate_hz;`
    //  是声明并初始化：`const float time` 建立局部对象 `time`，右侧完整表达式 `static_cast<float>(pulse) / prf_hz +
    //  static_cast<float>(sample) / sample_rate_hz` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数
    //  `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  base：延迟读取后的参考复样本；记作 $s[n-d]$；乘目标幅度并施加多普勒相位
    //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    if (source >= 0 && source < pulse_samples) {
        const auto base = waveform[source];
        const float time = static_cast<float>(pulse) / prf_hz +
            static_cast<float>(sample) / sample_rate_hz;
        // <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `const float phase = 2.0F * kPi * doppler_hz * time;` 是声明并初始化：`const float phase` 建立局部对象 `phase`，右侧完整表达式
        //  `2.0F * kPi * doppler_hz * time` 产生初值。
        //  `const float cosine = cosf(phase);` 是声明并初始化：`const float cosine` 建立局部对象 `cosine`，右侧完整表达式 `cosf(phase)`
        //  产生初值。
        //  `const float sine = sinf(phase);` 是声明并初始化：`const float sine` 建立局部对象 `sine`，右侧完整表达式 `sinf(phase)` 产生初值。
        //
        //  【变量—数学符号—数据流】
        //  phase：复指数的相位，单位 rad；记作 $\phi$；通过 cos/sin 形成复波形或多普勒旋转
        //  cosine：当前相位的余弦值；记作 $\cos\phi$；与 sine 共同执行复数相位旋转
        //  sine：当前相位的正弦值；记作 $\sin\phi$；与 cosine 共同执行复数相位旋转
        //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
        //
        //  【数字常量】
        //  2.0F：F 使浮点字面量为 float；不带后缀默认是 double；来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。
        //
        //  【数学/物理公式对照】
        //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
        //
        //  【为什么这样设计】
        //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        const float phase = 2.0F * kPi * doppler_hz * time;
        const float cosine = cosf(phase);
        const float sine = sinf(phase);
        // <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `target_real = target_amplitude * (base.real * cosine - base.imag * sine);` 是赋值：先求得右侧完整表达式
        //  `target_amplitude * (base.real * cosine - base.imag * sine)`，再把结果写入左侧可修改对象 `target_real`；这里的 `=` 不属于
        //  `>=`、`<=`、`==` 或 `!=`。
        //  `target_imag = target_amplitude * (base.real * sine + base.imag * cosine);` 是赋值：先求得右侧完整表达式
        //  `target_amplitude * (base.real * sine + base.imag * cosine)`，再把结果写入左侧可修改对象 `target_imag`；这里的 `=` 不属于
        //  `>=`、`<=`、`==` 或 `!=`。
        //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
        //
        //  【变量—数学符号—数据流】
        //  target_real：目标回波复数的实部；记作 $\Re\{x_0[p,n]\}$；由参考波形和多普勒旋转计算，写入 noiseless/echo
        //  target_imag：目标回波复数的虚部；记作 $\Im\{x_0[p,n]\}$；与 target_real 组成复回波
        //  sine：当前相位的正弦值；记作 $\sin\phi$；与 cosine 共同执行复数相位旋转
        //  cosine：当前相位的余弦值；记作 $\cos\phi$；与 sine 共同执行复数相位旋转
        //
        //  【数学/物理公式对照】
        //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
        //
        //  【为什么这样设计】
        //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        target_real = target_amplitude * (base.real * cosine - base.imag * sine);
        target_imag = target_amplitude * (base.real * sine + base.imag * cosine);
    }
    // <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);` 是声明并初始化：`const std::uint32_t
    //  base_index` 建立局部对象 `base_index`，右侧完整表达式 `static_cast<std::uint32_t>(2 * index)` 产生初值。
    //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `const float noise_real = uniform_noise(base_index, seed, noise_std);` 是声明并初始化：`const float noise_real`
    //  建立局部对象 `noise_real`，右侧完整表达式 `uniform_noise(base_index, seed, noise_std)` 产生初值。
    //  `const float noise_imag = uniform_noise(base_index + 1U, seed, noise_std);` 是声明并初始化：`const float
    //  noise_imag` 建立局部对象 `noise_imag`，右侧完整表达式 `uniform_noise(base_index + 1U, seed, noise_std)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  base_index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  noise_real：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
    //  noise_imag：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
    //
    //  【数字常量】
    //  2：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //  1U：U 使整数字面量从 unsigned int 候选类型开始；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  令 $b=\texttt{noise_bits}(n,s)\mathbin{\&}65535\in[0,65535]$。`32767.5F=65535/2`，故 $u=b/32767.5-1\in[-1,1]$，返回 $w=\sigma u$，其中 `noise_std=σ`。先 `static_cast<float>` 再除可避免整数除法；仅凭本代码不能把 `noise_std` 解释成高斯标准差。
    //
    //  【为什么这样设计】
    //  同一有界零中心映射同时用于 CPU/GPU，使比较不受随机库和状态推进顺序影响；它服务可复现模拟，不等价于经过统计检验的高斯热噪声发生器。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);
    const float noise_real = uniform_noise(base_index, seed, noise_std);
    const float noise_imag = uniform_noise(base_index + 1U, seed, noise_std);
    // <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `noiseless[index] = {target_real, target_imag};` 是赋值：先求得右侧完整表达式 `{target_real, target_imag}`，再把结果写入左侧可修改对象
    //  `noiseless[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `noise[index] = {noise_real, noise_imag};` 是赋值：先求得右侧完整表达式 `{noise_real, noise_imag}`，再把结果写入左侧可修改对象
    //  `noise[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `echo[index] = {target_real + noise_real, target_imag + noise_imag};` 是赋值：先求得右侧完整表达式 `{target_real +
    //  noise_real, target_imag + noise_imag}`，再把结果写入左侧可修改对象 `echo[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
    //  echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
    //
    //  【数学/物理公式对照】
    //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
    //
    //  【为什么这样设计】
    //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    noiseless[index] = {target_real, target_imag};
    noise[index] = {noise_real, noise_imag};
    echo[index] = {target_real + noise_real, target_imag + noise_imag};
// <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}

// <学习注释：语义块：simulate_echo_kernel：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}  // namespace` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}  // namespace

// <学习注释：语义块：run_step1：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `StepEvidence run_step1(PipelineState& state) {` 中的 `run_step1` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  run_step1：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
StepEvidence run_step1(PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：run_step1：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.name = "step1";` 是赋值：先求得右侧完整表达式 `"step1"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `const auto formal_begin = Clock::now();` 是声明并初始化：`const auto formal_begin` 建立局部对象 `formal_begin`，右侧完整表达式
    //  `Clock::now()` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  formal_begin：正式端到端计时起点；无独立算法符号；时间戳可记作 $t_0$；与结束时间相减形成 formal_execution_ms
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    StepEvidence evidence;
    evidence.name = "step1";
    const auto formal_begin = Clock::now();
    // <学习注释：语义块：run_step1：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //  `const float doppler_hz = config.doppler_hz();` 是声明并初始化：`const float doppler_hz` 建立局部对象
    //  `doppler_hz`，右侧完整表达式 `config.doppler_hz()` 产生初值。
    //  `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象
    //  `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  doppler_hz：目标多普勒频移，单位 Hz；记作 $f_D$；进入 $e^{j2\pi f_Dt}$
    //  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
    //
    //  【为什么这样设计】
    //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    auto begin = Clock::now();
    const float doppler_hz = config.doppler_hz();
    const int count = config.num_pulses * config.samples_per_pulse;
    // <学习注释：语义块：run_step1：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_waveform(config.pulse_samples);` 声明
    //  `d_waveform`，其静态类型是 `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `config.pulse_samples`
    //  直接初始化/调用该类型的构造函数；这不是调用名为 `d_waveform` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["gpu_lfm_generation"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel(
    //  generate_lfm_kernel, static_cast<std::size_t>(config.pulse_samples), d_waveform.data(),
    //  config.pulse_samples, config.sample_rate_hz, config.bandwidth_hz); });` 先创建按引用捕获当前作用域对象的 lambda `[&]
    //  {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["gpu_lfm_generation"]`。lambda
    //  内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `pulse_samples=L`、`sample_rate_hz=f_s`、`bandwidth_hz=B`、`index=n`。$T=L/f_s$，$t_n=n/f_s-T/2$，$K=B/T$，$\phi_n=\pi Kt_n^2$，`{cosf(phase),sinf(phase)}` 对应 $s[n]=e^{j\phi_n}$。每个有效线程只写一个复采样点；越界线程直接退出。
    //
    //  【为什么这样设计】
    //  复指数基带 LFM 的瞬时频率为 $f(t)=\frac1{2\pi}d\phi/dt=Kt$；居中时间轴使扫频关于脉冲中心对称。逐采样点相互独立，适合一维 CUDA 并行。
    //
    //  【初学者易错点】
    //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
    //  kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。>
    evidence.prep_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_waveform(config.pulse_samples);
    evidence.operator_ms["gpu_lfm_generation"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            generate_lfm_kernel, static_cast<std::size_t>(config.pulse_samples),
            d_waveform.data(), config.pulse_samples,
            config.sample_rate_hz, config.bandwidth_hz);
    });
    // <学习注释：语义块：d_noiseless：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);` 声明 `d_noiseless`，其静态类型是
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_noiseless`
    //  的函数，模板尖括号也不是比较或位移。
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noise(count);` 声明 `d_noise`，其静态类型是
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_noise`
    //  的函数，模板尖括号也不是比较或位移。
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_echo(count);` 声明 `d_echo`，其静态类型是
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_echo`
    //  的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  d_noiseless：未加噪目标回波；记作 $x_0[p,n]$；用于区分信号与噪声贡献及测试证据
    //  d_noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
    //  d_echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。>
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noise(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_echo(count);
    // <学习注释：语义块：d_noiseless：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `std::size_t free_bytes = 0, total_bytes = 0;` 在同一条声明中建立两个 `std::size_t` 对象：`free_bytes` 由 `0`
    //  初始化，`total_bytes` 由 `0` 初始化；逗号分隔两个声明符，不是逗号运算符。
    //  `CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));` 通过 `CUDA_CHECK` 包装 CUDA Runtime
    //  调用；宏检查返回状态，失败时按项目统一错误路径传播，而实参调用负责实际查询/同步/复制。
    //  `evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);` 是赋值：先求得右侧完整表达式
    //  `static_cast<double>(total_bytes - free_bytes)`，再把结果写入左侧可修改对象
    //  `evidence.metrics["gpu_used_peak_bytes"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)`
    //  明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  free_bytes：设备当前空闲显存字节数；无独立数学符号；与 total_bytes 组合估计显存使用
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    // <学习注释：语义块：d_noiseless：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel(
    //  simulate_echo_kernel, static_cast<std::size_t>(count), d_waveform.data(), d_noiseless.data(),
    //  d_noise.data(), d_echo.data(), count, config.target_delay_samples, doppler_hz, config.noise_seed,
    //  config.samples_per_pulse, config.pulse_samples, config.prf_hz, config.sample_rate_hz,
    //  config.target_amplitude, config.noise_std); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的
    //  launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["gpu_delay_doppler_noise"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
    //
    //  【为什么这样设计】
    //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
    //
    //  【初学者易错点】
    //  kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。>
    evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            simulate_echo_kernel, static_cast<std::size_t>(count),
            d_waveform.data(), d_noiseless.data(), d_noise.data(), d_echo.data(),
            count, config.target_delay_samples, doppler_hz, config.noise_seed,
            config.samples_per_pulse, config.pulse_samples, config.prf_hz,
            config.sample_rate_hz, config.target_amplitude, config.noise_std);
    });
    // <学习注释：语义块：d_noiseless：形成并返回当前结果。
    //
    //  【执行方法】
    //  `evidence.compute_ms = std::accumulate( evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
    //  [](double sum, const auto& item) { return sum + item.second; });` 调用 `std::accumulate` 从初值 `0.0`
    //  开始遍历证据映射；lambda 把累计值 `sum` 与当前键值项的 `second`（毫秒值）相加，最终总和写入左侧字段。
    //
    //  【变量—数学符号—数据流】
    //  sum：循环或归约的累加器；通常记作 $S=\sum_i a_i$；每轮读取旧值并累加当前 item
    //  item：当前循环/归约元素；记作 $a_i$；由容器迭代器产生并加入 sum
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
    //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
    // <学习注释：语义块：d_noiseless：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `state.waveform = d_waveform.to_host();` 是赋值：先求得右侧完整表达式 `d_waveform.to_host()`，再把结果写入左侧可修改对象
    //  `state.waveform`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `state.noiseless_echo = d_noiseless.to_host();` 是赋值：先求得右侧完整表达式 `d_noiseless.to_host()`，再把结果写入左侧可修改对象
    //  `state.noiseless_echo`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    begin = Clock::now();
    state.waveform = d_waveform.to_host();
    state.noiseless_echo = d_noiseless.to_host();
    // <学习注释：语义块：d_noiseless：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.noise = d_noise.to_host();` 是赋值：先求得右侧完整表达式 `d_noise.to_host()`，再把结果写入左侧可修改对象 `state.noise`；这里的 `=`
    //  不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `state.echo = d_echo.to_host();` 是赋值：先求得右侧完整表达式 `d_echo.to_host()`，再把结果写入左侧可修改对象 `state.echo`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
    //  echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    state.noise = d_noise.to_host();
    state.echo = d_echo.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    // <学习注释：语义块：d_noiseless：形成并返回当前结果。
    //
    //  【执行方法】
    //  `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());` 是赋值：先求得右侧完整表达式
    //  `milliseconds(formal_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.formal_execution_ms`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

// <学习注释：语义块：d_noiseless：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}  // namespace task1` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}  // namespace task1
```

### 3.3 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`：`cpu_noise_bits`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：cpu_noise_bits：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::uint32_t cpu_noise_bits(std::uint32_t index, std::uint32_t seed) {` 中的 `cpu_noise_bits` 是函数定义；名称前的
//  `std::uint32_t` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed` 是形参声明。 末尾 `{`
//  打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 是声明并初始化：`std::uint32_t value` 建立局部对象
//  `value`，右侧完整表达式 `seed ^ (index * 747796405U + 2891336453U)` 产生初值。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。
//
//  【变量—数学符号—数据流】
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
//  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
//
//  【数字常量】
//  747796405U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0x2C9277B5；这是 PCG 系列常用 32 位 LCG 乘数。此处用于让相邻 index 在进入后续
//  avalanche 前先分散；它不是物理常数。来源：PCG 原始技术报告（访问日期：2026-08-19）
//  2891336453U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0xAC564B05；这是与上述乘数配套使用的 32 位增量。此处平移状态，避免 seed/index
//  的零值直接保持零；不是雷达参数。来源：PCG 原始技术报告（访问日期：2026-08-19）
//
//  【数学/物理公式对照】
//  令 `index=n`、`seed=s`。初值 $v_0=s\oplus(747796405n+2891336453)\pmod{2^{32}}$；后续右移—异或和奇数乘法都在 `uint32_t` 的模 $2^{32}$ 算术中扩散比特，得到确定性的 32 位 $v$。这是无状态整数混合，不是完整 PCG 状态机，也不是物理噪声分布。
//  `U` 指定无符号整数字面量；`^` 是按位异或而非乘方，`>>` 对无符号值高位补零。移位把高位信息折叠进低位，乘法借助进位传播差异；删改常量或移位数会改变全部 CPU/GPU 可复现序列。
//
//  【为什么这样设计】
//  每个 CUDA 线程或 CPU 迭代仅凭 `(seed,index)` 即可重建相同结果，无需共享随机状态，因此适合逐元素精度比较。常量身份只按仓库和可靠资料可核实的范围表述，不把“借用常数”夸大成完整算法。
//
//  【初学者易错点】
//  在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。>
std::uint32_t cpu_noise_bits(std::uint32_t index, std::uint32_t seed)
{
    std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);
    // <学习注释：语义块：cpu_noise_bits：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `value ^= value >> 16;` 是复合赋值：读取 `value` 的旧值，与 `value >> 16` 执行 `^` 运算，再把结果写回同一对象。 `^`
    //  是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
    //  `value *= 2246822519U;` 是复合赋值：读取 `value` 的旧值，与 `2246822519U` 执行 `*` 运算，再把结果写回同一对象。
    //  `value ^= value >> 13;` 是复合赋值：读取 `value` 的旧值，与 `value >> 13` 执行 `^` 运算，再把结果写回同一对象。 `^`
    //  是逐位异或，不是乘方；它把两个无符号整数的比特混合。 `>>` 在这里对无符号整数执行右移，高位补零；移出的低位信息通过后续异或反馈到结果。
    //
    //  【变量—数学符号—数据流】
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //
    //  【数字常量】
    //  16：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；这是整数混合器的右移位数。16 位把高位信息折回较低位，再通过异或改变输出；16,13,16 的移位次序与 MurmurHash3 fmix32
    //  相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //
    //  2246822519U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0x85EBCA77。它是奇数，因此在模 $2^{32}$ 算术中可逆并能借助乘法进位扩散比特。它接近但不等于
    //  MurmurHash3 fmix32 的 0x85EBCA6B；仓库源码未记录为何相差 0x0C，不能冒充原始 Murmur 常数。对照来源：MurmurHash3 原始 fmix32
    //  实现（访问日期：2026-08-19）
    //  13：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；这是整数混合器的右移位数。13 位把高位信息折回较低位，再通过异或改变输出；16,13,16 的移位次序与 MurmurHash3 fmix32
    //  相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //
    //
    //  【数学/物理公式对照】
    //  令 `index=n`、`seed=s`。初值 $v_0=s\oplus(747796405n+2891336453)\pmod{2^{32}}$；后续右移—异或和奇数乘法都在 `uint32_t` 的模 $2^{32}$ 算术中扩散比特，得到确定性的 32 位 $v$。这是无状态整数混合，不是完整 PCG 状态机，也不是物理噪声分布。
    //  `U` 指定无符号整数字面量；`^` 是按位异或而非乘方，`>>` 对无符号值高位补零。移位把高位信息折叠进低位，乘法借助进位传播差异；删改常量或移位数会改变全部 CPU/GPU 可复现序列。
    //
    //  【为什么这样设计】
    //  每个 CUDA 线程或 CPU 迭代仅凭 `(seed,index)` 即可重建相同结果，无需共享随机状态，因此适合逐元素精度比较。常量身份只按仓库和可靠资料可核实的范围表述，不把“借用常数”夸大成完整算法。
    //
    //  【初学者易错点】
    //  在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。>
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    // <学习注释：语义块：cpu_noise_bits：形成并返回当前结果。
    //
    //  【执行方法】
    //  `value *= 3266489917U;` 是复合赋值：读取 `value` 的旧值，与 `3266489917U` 执行 `*` 运算，再把结果写回同一对象。
    //  `return value ^ (value >> 16);` 是返回语句：先求值 `value ^ (value >> 16)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //
    //  【数字常量】
    //  3266489917U：U 使整数字面量从 unsigned int 候选类型开始；十六进制为 0xC2B2AE3D。它同样是用于继续扩散的奇数乘数，但不等于 MurmurHash3 fmix32 的
    //  0xC2B2AE35；仓库源码未记录为何相差 0x08。改变它会改变全部确定性噪声序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //  16：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；这是整数混合器的右移位数。16 位把高位信息折回较低位，再通过异或改变输出；16,13,16 的移位次序与 MurmurHash3 fmix32
    //  相同，但本仓库乘数不同，所以只能称为相似结构而非原算法。改变位数会改变比特扩散质量和 CPU/GPU 可复现序列。对照来源：MurmurHash3 原始 fmix32 实现（访问日期：2026-08-19）
    //
    //
    //  【数学/物理公式对照】
    //  令 `index=n`、`seed=s`。初值 $v_0=s\oplus(747796405n+2891336453)\pmod{2^{32}}$；后续右移—异或和奇数乘法都在 `uint32_t` 的模 $2^{32}$ 算术中扩散比特，得到确定性的 32 位 $v$。这是无状态整数混合，不是完整 PCG 状态机，也不是物理噪声分布。
    //  `U` 指定无符号整数字面量；`^` 是按位异或而非乘方，`>>` 对无符号值高位补零。移位把高位信息折叠进低位，乘法借助进位传播差异；删改常量或移位数会改变全部 CPU/GPU 可复现序列。
    //
    //  【为什么这样设计】
    //  每个 CUDA 线程或 CPU 迭代仅凭 `(seed,index)` 即可重建相同结果，无需共享随机状态，因此适合逐元素精度比较。常量身份只按仓库和可靠资料可核实的范围表述，不把“借用常数”夸大成完整算法。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
    //  在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。>
    value *= 3266489917U;
    return value ^ (value >> 16);
}
```

### 3.4 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`：`cpu_uniform_noise`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：cpu_uniform_noise：形成并返回当前结果。
//
//  【执行方法】
//  `float cpu_uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std) {` 中的
//  `cpu_uniform_noise` 是函数定义；名称前的 `float` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index,
//  std::uint32_t seed, float noise_std` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `return (static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std;`
//  是返回语句：先求值 `(static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) *
//  noise_std`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
//
//  【变量—数学符号—数据流】
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
//  noise_std：噪声幅度尺度/标准差参数；记作 $\sigma$；缩放单位伪随机样本
//
//  【数字常量】
//  0xffffU：U 使整数字面量从 unsigned int 候选类型开始；十六进制 0xFFFF 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16
//  位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。
//  32767.5F：F 使浮点字面量为 float；不带后缀默认是 double；它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行
//  (u-32767.5)/32767.5 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。
//  1.0F：F 使浮点字面量为 float；不带后缀默认是 double；这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1
//  而不是 0。
//
//  【数学/物理公式对照】
//  令 $b=\texttt{noise_bits}(n,s)\mathbin{\&}65535\in[0,65535]$。`32767.5F=65535/2`，故 $u=b/32767.5-1\in[-1,1]$，返回 $w=\sigma u$，其中 `noise_std=σ`。先 `static_cast<float>` 再除可避免整数除法；仅凭本代码不能把 `noise_std` 解释成高斯标准差。
//
//  【为什么这样设计】
//  同一有界零中心映射同时用于 CPU/GPU，使比较不受随机库和状态推进顺序影响；它服务可复现模拟，不等价于经过统计检验的高斯热噪声发生器。
//
//  【初学者易错点】
//  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
//  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
float cpu_uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)
{
    return (static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) /
            32767.5F - 1.0F) * noise_std;
// <学习注释：语义块：cpu_uniform_noise：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}
```

### 3.5 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`：`waveform_cpu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：waveform_cpu：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<cusignal::TypedComplex<float>> waveform_cpu(const TaskConfig& config) {` 中的 `waveform_cpu`
//  是函数定义；名称前的 `std::vector<cusignal::TypedComplex<float>>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::vector<cusignal::TypedComplex<float>> output(config.pulse_samples);` 声明 `output`，其静态类型是
//  `std::vector<cusignal::TypedComplex<float>>`，并用 `config.pulse_samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
//  的函数，模板尖括号也不是比较或位移。
//
//  【变量—数学符号—数据流】
//  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<cusignal::TypedComplex<float>> waveform_cpu(const TaskConfig& config)
{
    std::vector<cusignal::TypedComplex<float>> output(config.pulse_samples);
    // <学习注释：语义块：waveform_cpu：迭代处理数据范围。
    //
    //  【执行方法】
    //  `const double pulse_width = static_cast<double>(config.pulse_samples) / config.sample_rate_hz;`
    //  是声明并初始化：`const double pulse_width` 建立局部对象 `pulse_width`，右侧完整表达式 `static_cast<double>(config.pulse_samples)
    //  / config.sample_rate_hz` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `const double slope = static_cast<double>(config.bandwidth_hz) / pulse_width;` 是声明并初始化：`const double
    //  slope` 建立局部对象 `slope`，右侧完整表达式 `static_cast<double>(config.bandwidth_hz) / pulse_width` 产生初值。
    //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `for (int index = 0; index < config.pulse_samples; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查
    //  `index < config.pulse_samples`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //  `const double time = static_cast<double>(index) / config.sample_rate_hz - 0.5 * pulse_width;`
    //  是声明并初始化：`const double time` 建立局部对象 `time`，右侧完整表达式 `static_cast<double>(index) / config.sample_rate_hz -
    //  0.5 * pulse_width` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  pulse_width：脉冲持续时间，单位 s；记作 $T=N_p/f_s$；用于 LFM 相位和距离分辨率关系
    //  slope：LFM 调频斜率；记作 $k=B/T$，单位 Hz/s；进入二次相位 $\phi(t)=\pi kt^2$
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
    //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //  0.5：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。
    //
    //  【数学/物理公式对照】
    //  `pulse_samples=L`、`sample_rate_hz=f_s`、`bandwidth_hz=B`、`index=n`。$T=L/f_s$，$t_n=n/f_s-T/2$，$K=B/T$，$\phi_n=\pi Kt_n^2$，`{cosf(phase),sinf(phase)}` 对应 $s[n]=e^{j\phi_n}$。每个有效线程只写一个复采样点；越界线程直接退出。
    //
    //  【为什么这样设计】
    //  复指数基带 LFM 的瞬时频率为 $f(t)=\frac1{2\pi}d\phi/dt=Kt$；居中时间轴使扫频关于脉冲中心对称。逐采样点相互独立，适合一维 CUDA 并行。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
    const double pulse_width = static_cast<double>(config.pulse_samples) / config.sample_rate_hz;
    const double slope = static_cast<double>(config.bandwidth_hz) / pulse_width;
    for (int index = 0; index < config.pulse_samples; ++index) {
        const double time = static_cast<double>(index) / config.sample_rate_hz - 0.5 * pulse_width;
        // <学习注释：语义块：waveform_cpu：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `const double phase = 3.14159265358979323846 * slope * time * time;` 是声明并初始化：`const double phase` 建立局部对象
        //  `phase`，右侧完整表达式 `3.14159265358979323846 * slope * time * time` 产生初值。
        //  `output[index] = {static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};`
        //  是赋值：先求得右侧完整表达式 `{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))}`，再把结果写入左侧可修改对象
        //  `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数
        //  `T`，转换发生后才参与外层表达式。
        //
        //  【变量—数学符号—数据流】
        //  phase：复指数的相位，单位 rad；记作 $\phi$；通过 cos/sin 形成复波形或多普勒旋转
        //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
        //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
        //
        //  【数字常量】
        //  3.14159265358979323846：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
        //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
        //
        //  【为什么这样设计】
        //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        const double phase = 3.14159265358979323846 * slope * time * time;
        output[index] = {static_cast<float>(std::cos(phase)),
                         static_cast<float>(std::sin(phase))};
    // <学习注释：语义块：waveform_cpu：形成并返回当前结果。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  output：当前函数或步骤的输出容器；对应当前算法公式的左端结果，具体符号由所在 step 决定；由本块写入，返回或交给下一步骤
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    }
    return output;
}
```

### 3.6 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`：`echo_cpu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：echo_cpu：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<cusignal::TypedComplex<float>> echo_cpu( const std::vector<cusignal::TypedComplex<float>>&
//  waveform, int delay, float doppler_hz, std::uint32_t seed, const TaskConfig& config,
//  std::vector<cusignal::TypedComplex<float>>* noiseless, std::vector<cusignal::TypedComplex<float>>* noise)
//  {` 中的 `echo_cpu` 是函数定义；名称前的 `std::vector<cusignal::TypedComplex<float>>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的
//  `const std::vector<cusignal::TypedComplex<float>>& waveform, int delay, float doppler_hz, std::uint32_t
//  seed, const TaskConfig& config, std::vector<cusignal::TypedComplex<float>>* noiseless,
//  std::vector<cusignal::TypedComplex<float>>* noise` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象
//  `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。
//
//  【变量—数学符号—数据流】
//  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
//  delay：目标离散延迟，单位 sample；记作 $d$；回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$
//  doppler_hz：目标多普勒频移，单位 Hz；记作 $f_D$；进入 $e^{j2\pi f_Dt}$
//  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
//  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
//  noiseless：未加噪目标回波；记作 $x_0[p,n]$；用于区分信号与噪声贡献及测试证据
//  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
//  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
//
//  【数学/物理公式对照】
//  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
//
//  【为什么这样设计】
//  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
//
//  【初学者易错点】
//  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<cusignal::TypedComplex<float>> echo_cpu(
    const std::vector<cusignal::TypedComplex<float>>& waveform,
    int delay, float doppler_hz, std::uint32_t seed,
    const TaskConfig& config,
    std::vector<cusignal::TypedComplex<float>>* noiseless,
    std::vector<cusignal::TypedComplex<float>>* noise)
{
    const int count = config.num_pulses * config.samples_per_pulse;
    // <学习注释：语义块：echo_cpu：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `noiseless->assign(count, {0.0F, 0.0F});` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
    //  `noise->assign(count, {0.0F, 0.0F});` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
    //  `std::vector<cusignal::TypedComplex<float>> output(count);` 声明 `output`，其静态类型是
    //  `std::vector<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
    //  的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
    //
    //  【数字常量】
    //  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。>
    noiseless->assign(count, {0.0F, 0.0F});
    noise->assign(count, {0.0F, 0.0F});
    std::vector<cusignal::TypedComplex<float>> output(count);
    // <学习注释：语义块：echo_cpu：迭代处理数据范围。
    //
    //  【执行方法】
    //  `for (int index = 0; index < count; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index <
    //  count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //  `const int pulse = index / config.samples_per_pulse;` 是声明并初始化：`const int pulse` 建立局部对象 `pulse`，右侧完整表达式
    //  `index / config.samples_per_pulse` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
    //  `const int sample = index % config.samples_per_pulse;` 是声明并初始化：`const int sample` 建立局部对象 `sample`，右侧完整表达式
    //  `index % config.samples_per_pulse` 产生初值。 `%` 取得整数除法余数，用来把展平索引还原为当前行/脉冲内的位置。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  pulse：当前慢时间脉冲编号；记作 $p=\lfloor i/N_s\rfloor$；由展平索引整除每脉冲采样数得到
    //  sample：当前脉冲内快时间采样编号；记作 $n=i\bmod N_s$；由展平索引取余得到
    //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。
    //  取余的分母不能为零；这里依赖每脉冲采样数/维度已通过配置校验。>
    for (int index = 0; index < count; ++index) {
        const int pulse = index / config.samples_per_pulse;
        const int sample = index % config.samples_per_pulse;
        // <学习注释：语义块：echo_cpu：检查条件并选择执行路径。
        //
        //  【执行方法】
        //  `const int source = sample - delay;` 是声明并初始化：`const int source` 建立局部对象 `source`，右侧完整表达式 `sample - delay`
        //  产生初值。
        //  `if (source >= 0 && source < config.pulse_samples) {` 是 `if` 条件语句：先把完整条件 `source >= 0 && source <
        //  config.pulse_samples` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
        //  `const double time = static_cast<double>(pulse) / config.prf_hz + static_cast<double>(sample) /
        //  config.sample_rate_hz;` 是声明并初始化：`const double time` 建立局部对象 `time`，右侧完整表达式 `static_cast<double>(pulse) /
        //  config.prf_hz + static_cast<double>(sample) / config.sample_rate_hz` 产生初值。 `static_cast<T>(...)`
        //  明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
        //
        //  【变量—数学符号—数据流】
        //  source：施加延迟前回读的波形采样位置；记作 $n-d$；只有落在波形有效区间时才形成目标回波
        //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
        //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
        //
        //  【数字常量】
        //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
        //
        //  【为什么这样设计】
        //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
        //
        //  【初学者易错点】
        //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
        //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
        //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
        const int source = sample - delay;
        if (source >= 0 && source < config.pulse_samples) {
            const double time = static_cast<double>(pulse) / config.prf_hz +
                static_cast<double>(sample) / config.sample_rate_hz;
            // <学习注释：语义块：echo_cpu：完成一个连续的数据处理动作。
            //
            //  【执行方法】
            //  `const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;` 是声明并初始化：`const double phase`
            //  建立局部对象 `phase`，右侧完整表达式 `2.0 * 3.14159265358979323846 * doppler_hz * time` 产生初值。
            //  `const double cosine = std::cos(phase);` 是声明并初始化：`const double cosine` 建立局部对象 `cosine`，右侧完整表达式
            //  `std::cos(phase)` 产生初值。
            //  `const double sine = std::sin(phase);` 是声明并初始化：`const double sine` 建立局部对象 `sine`，右侧完整表达式 `std::sin(phase)`
            //  产生初值。
            //
            //  【变量—数学符号—数据流】
            //  phase：复指数的相位，单位 rad；记作 $\phi$；通过 cos/sin 形成复波形或多普勒旋转
            //  cosine：当前相位的余弦值；记作 $\cos\phi$；与 sine 共同执行复数相位旋转
            //  sine：当前相位的正弦值；记作 $\sin\phi$；与 cosine 共同执行复数相位旋转
            //  time：当前样本对应的物理时间，单位 s；记作 $t$；进入相位 $2\pi f_Dt$ 或 LFM 相位
            //
            //  【数字常量】
            //  2.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
            //
            //  3.14159265358979323846：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
            //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
            //
            //  【数学/物理公式对照】
            //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
            //
            //  【为什么这样设计】
            //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
            //
            //  【初学者易错点】
            //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
            const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            // <学习注释：语义块：echo_cpu：完成一个连续的数据处理动作。
            //
            //  【执行方法】
            //  `(*noiseless)[index] = { static_cast<float>(config.target_amplitude * (waveform[source].real * cosine -
            //  waveform[source].imag * sine)), static_cast<float>(config.target_amplitude * (waveform[source].real * sine
            //  + waveform[source].imag * cosine))};` 是赋值：先求得右侧完整表达式 `{ static_cast<float>(config.target_amplitude *
            //  (waveform[source].real * cosine - waveform[source].imag * sine)),
            //  static_cast<float>(config.target_amplitude * (waveform[source].real * sine + waveform[source].imag *
            //  cosine))}`，再把结果写入左侧可修改对象 `(*noiseless)[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)`
            //  明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
            //
            //  【变量—数学符号—数据流】
            //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
            //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
            //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
            //  sine：当前相位的正弦值；记作 $\sin\phi$；与 cosine 共同执行复数相位旋转
            //  cosine：当前相位的余弦值；记作 $\cos\phi$；与 sine 共同执行复数相位旋转
            //
            //  【数学/物理公式对照】
            //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
            //
            //  【为什么这样设计】
            //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
            //
            //  【初学者易错点】
            //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
            (*noiseless)[index] = {
                static_cast<float>(config.target_amplitude *
                    (waveform[source].real * cosine - waveform[source].imag * sine)),
                static_cast<float>(config.target_amplitude *
                    (waveform[source].real * sine + waveform[source].imag * cosine))};
        // <学习注释：语义块：echo_cpu：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
        //  `const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);` 是声明并初始化：`const std::uint32_t
        //  base_index` 建立局部对象 `base_index`，右侧完整表达式 `static_cast<std::uint32_t>(2 * index)` 产生初值。
        //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
        //  `(*noise)[index] = { cpu_uniform_noise(base_index, seed, config.noise_std), cpu_uniform_noise(base_index +
        //  1U, seed, config.noise_std)};` 是赋值：先求得右侧完整表达式 `{ cpu_uniform_noise(base_index, seed, config.noise_std),
        //  cpu_uniform_noise(base_index + 1U, seed, config.noise_std)}`，再把结果写入左侧可修改对象 `(*noise)[index]`；这里的 `=` 不属于
        //  `>=`、`<=`、`==` 或 `!=`。
        //
        //  【变量—数学符号—数据流】
        //  base_index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
        //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
        //  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
        //  seed：确定性伪随机种子；记作 $s_0$；来自配置；与索引共同生成可复现噪声
        //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
        //
        //  【数字常量】
        //  2：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
        //
        //  1U：U 使整数字面量从 unsigned int 候选类型开始；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
        //
        //  【数学/物理公式对照】
        //  令 $b=\texttt{noise_bits}(n,s)\mathbin{\&}65535\in[0,65535]$。`32767.5F=65535/2`，故 $u=b/32767.5-1\in[-1,1]$，返回 $w=\sigma u$，其中 `noise_std=σ`。先 `static_cast<float>` 再除可避免整数除法；仅凭本代码不能把 `noise_std` 解释成高斯标准差。
        //
        //  【为什么这样设计】
        //  同一有界零中心映射同时用于 CPU/GPU，使比较不受随机库和状态推进顺序影响；它服务可复现模拟，不等价于经过统计检验的高斯热噪声发生器。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        }
        const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);
        (*noise)[index] = {
            cpu_uniform_noise(base_index, seed, config.noise_std),
            cpu_uniform_noise(base_index + 1U, seed, config.noise_std)};
        // <学习注释：语义块：echo_cpu：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `output[index] = { (*noiseless)[index].real + (*noise)[index].real, (*noiseless)[index].imag +
        //  (*noise)[index].imag};` 是赋值：先求得右侧完整表达式 `{ (*noiseless)[index].real + (*noise)[index].real,
        //  (*noiseless)[index].imag + (*noise)[index].imag}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==`
        //  或 `!=`。
        //
        //  【变量—数学符号—数据流】
        //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
        //  noise：加性噪声数组；记作 $w[p,n]$；由 seed/index 确定性生成并叠加到 noiseless
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
        //
        //  【为什么这样设计】
        //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        output[index] = {
            (*noiseless)[index].real + (*noise)[index].real,
            (*noiseless)[index].imag + (*noise)[index].imag};
    // <学习注释：语义块：echo_cpu：形成并返回当前结果。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  output：当前函数或步骤的输出容器；对应当前算法公式的左端结果，具体符号由所在 step 决定；由本块写入，返回或交给下一步骤
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    }
    return output;
}
```

### 3.7 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`：`generate_step1_cpu_reference`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：generate_step1_cpu_reference：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config) {` 中的
//  `generate_step1_cpu_reference` 是函数定义；名称前的 `Step1ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `Step1ValidationData data;` 是对象声明：类型 `Step1ValidationData` 应用于名称
//  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  generate_step1_cpu_reference：生成当前名称所描述数据的 helper/结果；对应生成公式的输出端；由输入参数构造数据，供后续 step 或测试使用
//  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
//  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)
{
    Step1ValidationData data;
    // <学习注释：语义块：generate_step1_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.waveform_cpu = waveform_cpu(config);` 是赋值：先求得右侧完整表达式 `waveform_cpu(config)`，再把结果写入左侧可修改对象
    //  `data.waveform_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `const float doppler_hz = config.doppler_hz();` 是声明并初始化：`const float doppler_hz` 建立局部对象
    //  `doppler_hz`，右侧完整表达式 `config.doppler_hz()` 产生初值。
    //  `data.echo_cpu = echo_cpu( data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed,
    //  config, &data.noiseless_cpu, &data.noise_cpu);` 是赋值：先求得右侧完整表达式 `echo_cpu( data.waveform_cpu,
    //  config.target_delay_samples, doppler_hz, config.noise_seed, config, &data.noiseless_cpu,
    //  &data.noise_cpu)`，再把结果写入左侧可修改对象 `data.echo_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  doppler_hz：目标多普勒频移，单位 Hz；记作 $f_D$；进入 $e^{j2\pi f_Dt}$
    //  config：外部配置对象；`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  展平索引对应 $(p,n)$：$p=\lfloor index/M\rfloor$、$n=index\bmod M$，`samples_per_pulse=M`。`source=n-delay` 实现离散延迟 $d$；$t_{p,n}=p/PRF+n/f_s$，$\theta=2\pi f_Dt_{p,n}$。`target_real/imag` 是 $A\,s[n-d]e^{j\theta}$ 的实虚部展开；`echo=noiseless+noise`。`2*index` 与 `2*index+1` 分别索引复噪声实、虚部。
    //
    //  【为什么这样设计】
    //  展平矩阵让每线程独立写一个复采样点。延迟用索引移位、多普勒用复旋转；边界条件防止波形越界，去掉复旋转则无法模拟多普勒相位推进。
    //
    //  【初学者易错点】
    //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
    data.waveform_cpu = waveform_cpu(config);
    const float doppler_hz = config.doppler_hz();
    data.echo_cpu = echo_cpu(
        data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed, config,
        &data.noiseless_cpu, &data.noise_cpu);
    // <学习注释：语义块：generate_step1_cpu_reference：形成并返回当前结果。
    //
    //  【执行方法】
    //  `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    return data;
}
```

### 3.8 函数总体处理方法

#### 3.8.1 总体业务流水线

`外部配置→Host 时间/参数→GPU LFM 与回波模拟→D2H`。生成延迟、多普勒和噪声回波，输出 waveform/echo 给 Step2。

#### 3.8.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`noise_bits`|对 seed 与样本索引做无符号整数混合，产生可复现的伪随机位|GPU device helper|
|`uniform_noise`|把伪随机位缩放到以 0 为中心的均匀噪声样本|GPU device helper|
|`generate_lfm_kernel`|每个线程按一个快时间采样点计算 LFM 相位和复指数|GPU device kernel|
|`simulate_echo_kernel`|每个线程按一个脉冲—采样单元叠加延迟目标、多普勒相位与噪声|GPU device kernel|
|`run_step1`|校验配置，分配设备数组，启动波形/回波 kernel，回收输出并记录证据|GPU Host wrapper/业务入口|
|`cpu_noise_bits`|用与 device 相同的无符号整数规则生成 CPU 伪随机位|CPU reference helper|
|`cpu_uniform_noise`|把 CPU 伪随机位映射成与 GPU 同口径的均匀噪声|CPU reference helper|
|`waveform_cpu`|逐采样点按 LFM 公式生成 Host 参考波形|CPU reference core|
|`echo_cpu`|逐脉冲、逐采样点生成 Host 目标回波和噪声参考|CPU reference core|
|`generate_step1_cpu_reference`|汇总 Step1 CPU 波形、回波和验证元数据|CPU validation entry|

#### 3.8.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `chirp`

- 学习入口：[数学物理原理](../../operators/waveforms/chirp/数学物理原理.md)、[Python 源码算法](../../operators/waveforms/chirp/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/waveforms/chirp/cusignal_cpp_chirp复现逻辑.md)。
- 当前调用采用的数学关系：$s(t)=\exp\{j2\pi(f_0t+\tfrac12kt^2)+j\phi_0\}$，其中 $k=B/T$。
- 任务变量与参数实例化：任务变量 `time→t`、`bandwidth_hz→B`、`pulse_width→T`、`phase→2π(f_0t+kt²/2)+φ₀`；本 step 自有 kernel 生成等价 LFM，不复制算子内部代码。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.8.4 当前任务自有 LFM 与算子原理的对应边界

本 step **没有调用** `cusignal::chirp_*`；它以私有 `generate_lfm_kernel` 实现与线性 chirp 同属一个上位原理的复基带 LFM。实际离散式为
$T=L/f_s$、$t_n=n/f_s-T/2$、$K=B/T$、$s[n]=\exp(j\pi Kt_n^2)$，因此不能把算子文档中的其他 method、实余弦输出或默认参数直接套入本 kernel。

|代码变量|数学量|dtype/shape|来源与消费者|
|---|---|---|---|
|`pulse_samples`|$L$|`int` 标量|配置；决定波形长度与 $T$|
|`sample_rate_hz`|$f_s$，Hz|`float` 标量|配置；形成快时间|
|`bandwidth_hz`|$B$，Hz|`float` 标量|配置；形成 $K=B/T$|
|`waveform`|$s[n]$|device `TypedComplex<float>[L]`|LFM kernel 输出；回波 kernel 与 Step2 消费|
|`delay`、`doppler_hz`|$d$、$f_D$|`int` 样点、`float` Hz|配置换算；进入索引移位与 $e^{j2\pi f_Dt}$|
|`echo`|$A s[n-d]e^{j2\pi f_Dt}+w[p,n]$|展平 device/Host 复 `float[P×M]`|Step1 输出；Step2 输入|

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step1(PipelineState& state); }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
- `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task1 { StepEvidence run_step1(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence run_step1(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `#include "step1.h"`。

```cpp
#include "step1.h"

#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step1`、`h`、`cuda_utils`、`device_array`、`kernel_launch`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step1.h"` 在预处理阶段引入 "step1.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/kernel_launch.h"` 在预处理阶段引入 "cuda_utils/kernel_launch.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

```cpp
#include <numeric>

namespace task1 {
namespace {
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`numeric`、`namespace`、`task1`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task1 {` 打开命名空间 `task1`；后续声明被放入该名称范围。
- `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `noise_bits`；源码锚点 `__device__ std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)`。

```cpp
__device__ std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed)
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

- `__device__ std::uint32_t noise_bits(std::uint32_t index, std::uint32_t seed) {` 中的 `noise_bits` 是函数定义；名称前的 `__device__ std::uint32_t` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed` 是形参声明。 `__device__` 表明该 helper 在 GPU device 代码中调用。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 是声明并初始化：`std::uint32_t value` 建立局部对象 `value`，右侧完整表达式 `seed ^ (index * 747796405U + 2891336453U)` 产生初值。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `noise_bits`；调用 `noise_bits`；写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 先执行乘加再与 seed 组合，是为了让相邻 `index` 不直接形成相邻整数状态，并避免零输入停留在简单零模式；这一步采用 PCG 常见 LCG 常数作为低成本初始打散，后续再由 Murmur 风格 avalanche 完成扩散。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.5 noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `noise_bits`；源码锚点 `value ^= value >> 16;`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.6 noise_bits：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `noise_bits`；源码锚点 `value *= 3266489917U;`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `value`；返回 `value ^ (value >> 16)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.7 uniform_noise：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `uniform_noise`；源码锚点 `__device__ float uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)`。

```cpp

__device__ float uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)
{
    return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F)
        * noise_std;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `noise_std` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `0xffffU` 是数值字面量；U 指定 unsigned int 起始类型；
- `32767.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`noise_std`|噪声幅度尺度/标准差参数|记作 $\sigma$|缩放单位伪随机样本|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0xffffU`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。|
|`32767.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。|

**执行过程**

- `__device__ float uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std) {` 中的 `uniform_noise` 是函数定义；名称前的 `__device__ float` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed, float noise_std` 是形参声明。 `__device__` 表明该 helper 在 GPU device 代码中调用。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return (static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std;` 是返回语句：先求值 `(static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `uniform_noise`；调用 `uniform_noise`, `return`, `noise_bits`；返回 `(static_cast<float>(noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 先用 `0xFFFF` 只保留低 16 位，是为了得到固定的 65536 个离散状态；随后除以半跨度 `32767.5` 并减 1，把它线性映射为以 0 为中心的 $[-1,1]$。最后乘 `noise_std` 才得到任务噪声幅度。可替代为 32 位到浮点的完整映射或标准 RNG，但那会改变历史 CPU/GPU 确定性序列和精度证据。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.8 uniform_noise：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `uniform_noise`；源码锚点 `}`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.9 generate_lfm_kernel：声明 CUDA kernel 并建立线程入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `generate_lfm_kernel`；源码锚点 `__global__ void generate_lfm_kernel(`。

```cpp

__global__ void generate_lfm_kernel(
    cusignal::TypedComplex<float>* waveform, int pulse_samples,
    float sample_rate_hz, float bandwidth_hz)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `waveform` 由 `cusignal::TypedComplex<float>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `pulse_samples` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `sample_rate_hz` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `bandwidth_hz` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `index` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`pulse_samples`|发射脉冲波形包含的快时间采样数|记作 $N_p$|由配置和采样率决定；限定 waveform 有效区间并形成脉宽 $T=N_p/f_s$|
|`sample_rate_hz`|采样率，单位 Hz|记作 $f_s$|把离散样本索引换算为秒|
|`bandwidth_hz`|扫频带宽，单位 Hz|记作 $B$|与脉宽共同决定 LFM 调频斜率 $k=B/T$|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `__global__ void generate_lfm_kernel( cusignal::TypedComplex<float>* waveform, int pulse_samples, float sample_rate_hz, float bandwidth_hz) {` 中的 `generate_lfm_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `cusignal::TypedComplex<float>* waveform, int pulse_samples, float sample_rate_hz, float bandwidth_hz` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index` 建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x` 把块号和块内线程号映射为一维全局线程索引。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `generate_lfm_kernel`；调用 `generate_lfm_kernel`；写入/初始化 `index`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 使用 `blockIdx.x * blockDim.x + threadIdx.x` 是为了给每个一维输出元素分配唯一全局线程。这样同一 kernel 可适配不同 block 大小；随后必须用 count 做越界保护，因为 grid 往往向上取整。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 generate_lfm_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `generate_lfm_kernel`；源码锚点 `if (index >= pulse_samples) return;`。

```cpp
    if (index >= pulse_samples) return;
    const float pulse_width = static_cast<float>(pulse_samples) / sample_rate_hz;
    const float time = static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 当前块中的 `if (...) return;` 必须先作为完整条件语句识别；比较运算符 `>=`/`<=` 中的 `=` 不是赋值。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `pulse_width` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `time` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `0.5F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`pulse_width`|脉冲持续时间，单位 s|记作 $T=N_p/f_s$|用于 LFM 相位和距离分辨率关系|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|

**执行过程**

- `if (index >= pulse_samples) return;` 是 `if` 条件语句：先把完整条件 `index >= pulse_samples` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前 CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
- `const float pulse_width = static_cast<float>(pulse_samples) / sample_rate_hz;` 是声明并初始化：`const float pulse_width` 建立局部对象 `pulse_width`，右侧完整表达式 `static_cast<float>(pulse_samples) / sample_rate_hz` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `const float time = static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width;` 是声明并初始化：`const float time` 建立局部对象 `time`，右侧完整表达式 `static_cast<float>(index) / sample_rate_hz - 0.5F * pulse_width` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过检查 `index >= pulse_samples`；写入/初始化 `pulse_width`, `time`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `generate_lfm_kernel` 所在的 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.11 generate_lfm_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `generate_lfm_kernel`；源码锚点 `const float phase = kPi * (bandwidth_hz / pulse_width) * time * time;`。

```cpp
    const float phase = kPi * (bandwidth_hz / pulse_width) * time * time;
    waveform[index] = {cosf(phase), sinf(phase)};
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `phase` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `time` 由 `time *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const float phase = kPi * (bandwidth_hz / pulse_width) * time * time;` 是声明并初始化：`const float phase` 建立局部对象 `phase`，右侧完整表达式 `kPi * (bandwidth_hz / pulse_width) * time * time` 产生初值。
- `waveform[index] = {cosf(phase), sinf(phase)};` 是赋值：先求得右侧完整表达式 `{cosf(phase), sinf(phase)}`，再把结果写入左侧可修改对象 `waveform[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `cosf`, `sinf`；写入/初始化 `phase`, `waveform[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.12 simulate_echo_kernel：声明 CUDA kernel 并建立线程入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `__global__ void simulate_echo_kernel(`。

```cpp

__global__ void simulate_echo_kernel(
    const cusignal::TypedComplex<float>* waveform,
    cusignal::TypedComplex<float>* noiseless,
    cusignal::TypedComplex<float>* noise,
    cusignal::TypedComplex<float>* echo,
    int count, int delay, float doppler_hz, std::uint32_t seed,
    int samples_per_pulse, int pulse_samples, float prf_hz,
    float sample_rate_hz, float target_amplitude, float noise_std)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `waveform` 由 `const cusignal::TypedComplex<float>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `noiseless` 由 `cusignal::TypedComplex<float>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `noise` 由 `cusignal::TypedComplex<float>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `echo` 由 `cusignal::TypedComplex<float>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `count` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `delay` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `doppler_hz` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `samples_per_pulse` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `pulse_samples` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `prf_hz` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `sample_rate_hz` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `target_amplitude` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `noise_std` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `index` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`noiseless`|未加噪目标回波|记作 $x_0[p,n]$|用于区分信号与噪声贡献及测试证据|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`delay`|目标离散延迟，单位 sample|记作 $d$|回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$|
|`doppler_hz`|目标多普勒频移，单位 Hz|记作 $f_D$|进入 $e^{j2\pi f_Dt}$|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`samples_per_pulse`|每个接收脉冲槽保存的快时间采样数|记作 $N_s$|用于二维 shape $P\times N_s$ 以及展平索引分解|
|`pulse_samples`|发射脉冲波形包含的快时间采样数|记作 $N_p$|由配置和采样率决定；限定 waveform 有效区间并形成脉宽 $T=N_p/f_s$|
|`prf_hz`|脉冲重复频率，单位 Hz|记作 $f_r$ 或 PRF|把脉冲编号换算为慢时间 $p/f_r$|
|`sample_rate_hz`|采样率，单位 Hz|记作 $f_s$|把离散样本索引换算为秒|
|`target_amplitude`|目标复回波幅度系数|记作 $A$|缩放无噪目标回波|
|`noise_std`|噪声幅度尺度/标准差参数|记作 $\sigma$|缩放单位伪随机样本|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `__global__ void simulate_echo_kernel( const cusignal::TypedComplex<float>* waveform, cusignal::TypedComplex<float>* noiseless, cusignal::TypedComplex<float>* noise, cusignal::TypedComplex<float>* echo, int count, int delay, float doppler_hz, std::uint32_t seed, int samples_per_pulse, int pulse_samples, float prf_hz, float sample_rate_hz, float target_amplitude, float noise_std) {` 中的 `simulate_echo_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const cusignal::TypedComplex<float>* waveform, cusignal::TypedComplex<float>* noiseless, cusignal::TypedComplex<float>* noise, cusignal::TypedComplex<float>* echo, int count, int delay, float doppler_hz, std::uint32_t seed, int samples_per_pulse, int pulse_samples, float prf_hz, float sample_rate_hz, float target_amplitude, float noise_std` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index` 建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x` 把块号和块内线程号映射为一维全局线程索引。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `simulate_echo_kernel`；调用 `simulate_echo_kernel`；写入/初始化 `index`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 使用 `blockIdx.x * blockDim.x + threadIdx.x` 是为了给每个一维输出元素分配唯一全局线程。这样同一 kernel 可适配不同 block 大小；随后必须用 count 做越界保护，因为 grid 往往向上取整。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 simulate_echo_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `if (index >= count) return;`。

```cpp
    if (index >= count) return;
    const int pulse = index / samples_per_pulse;
    const int sample = index % samples_per_pulse;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 当前块中的 `if (...) return;` 必须先作为完整条件语句识别；比较运算符 `>=`/`<=` 中的 `=` 不是赋值。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `pulse` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `sample` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`pulse`|当前慢时间脉冲编号|记作 $p=\lfloor i/N_s\rfloor$|由展平索引整除每脉冲采样数得到|
|`sample`|当前脉冲内快时间采样编号|记作 $n=i\bmod N_s$|由展平索引取余得到|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前 CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
- `const int pulse = index / samples_per_pulse;` 是声明并初始化：`const int pulse` 建立局部对象 `pulse`，右侧完整表达式 `index / samples_per_pulse` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
- `const int sample = index % samples_per_pulse;` 是声明并初始化：`const int sample` 建立局部对象 `sample`，右侧完整表达式 `index % samples_per_pulse` 产生初值。 `%` 取得整数除法余数，用来把展平索引还原为当前行/脉冲内的位置。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过检查 `index >= count`；写入/初始化 `pulse`, `sample`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。
- 取余的分母不能为零；这里依赖每脉冲采样数/维度已通过配置校验。

### 4.14 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `float target_real = 0.0F;`。

```cpp
    float target_real = 0.0F;
    float target_imag = 0.0F;
    const int source = sample - delay;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `target_real` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `target_imag` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `source` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`target_real`|目标回波复数的实部|记作 $\Re\{x_0[p,n]\}$|由参考波形和多普勒旋转计算，写入 noiseless/echo|
|`target_imag`|目标回波复数的虚部|记作 $\Im\{x_0[p,n]\}$|与 target_real 组成复回波|
|`source`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `float target_real = 0.0F;` 是声明并初始化：`float target_real` 建立局部对象 `target_real`，右侧完整表达式 `0.0F` 产生初值。
- `float target_imag = 0.0F;` 是声明并初始化：`float target_imag` 建立局部对象 `target_imag`，右侧完整表达式 `0.0F` 产生初值。
- `const int source = sample - delay;` 是声明并初始化：`const int source` 建立局部对象 `source`，右侧完整表达式 `sample - delay` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `target_real`, `target_imag`, `source`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 simulate_echo_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `if (source >= 0 && source < pulse_samples) {`。

```cpp
    if (source >= 0 && source < pulse_samples) {
        const auto base = waveform[source];
        const float time = static_cast<float>(pulse) / prf_hz +
            static_cast<float>(sample) / sample_rate_hz;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `base` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `time` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`base`|延迟读取后的参考复样本|记作 $s[n-d]$|乘目标幅度并施加多普勒相位|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `if (source >= 0 && source < pulse_samples) {` 是 `if` 条件语句：先把完整条件 `source >= 0 && source < pulse_samples` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `const auto base = waveform[source];` 是声明并初始化：`const auto base` 建立局部对象 `base`，右侧完整表达式 `waveform[source]` 产生初值。
- `const float time = static_cast<float>(pulse) / prf_hz + static_cast<float>(sample) / sample_rate_hz;` 是声明并初始化：`const float time` 建立局部对象 `time`，右侧完整表达式 `static_cast<float>(pulse) / prf_hz + static_cast<float>(sample) / sample_rate_hz` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过检查 `source >= 0 && source < pulse_samples`；写入/初始化 `base`, `time`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.16 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `const float phase = 2.0F * kPi * doppler_hz * time;`。

```cpp
        const float phase = 2.0F * kPi * doppler_hz * time;
        const float cosine = cosf(phase);
        const float sine = sinf(phase);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `phase` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `time` 由 `doppler_hz *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `cosine` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `sine` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `2.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`cosine`|当前相位的余弦值|记作 $\cos\phi$|与 sine 共同执行复数相位旋转|
|`sine`|当前相位的正弦值|记作 $\sin\phi$|与 cosine 共同执行复数相位旋转|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|来自复指数相位 $e^{j2\pi ft}$ 的完整一周 $2\pi$；不是经验参数。改成 $\pi$ 会使频率/相位尺度减半。|

**执行过程**

- `const float phase = 2.0F * kPi * doppler_hz * time;` 是声明并初始化：`const float phase` 建立局部对象 `phase`，右侧完整表达式 `2.0F * kPi * doppler_hz * time` 产生初值。
- `const float cosine = cosf(phase);` 是声明并初始化：`const float cosine` 建立局部对象 `cosine`，右侧完整表达式 `cosf(phase)` 产生初值。
- `const float sine = sinf(phase);` 是声明并初始化：`const float sine` 建立局部对象 `sine`，右侧完整表达式 `sinf(phase)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `cosf`, `sinf`；写入/初始化 `phase`, `cosine`, `sine`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.17 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `target_real = target_amplitude * (base.real * cosine - base.imag * sine);`。

```cpp
        target_real = target_amplitude * (base.real * cosine - base.imag * sine);
        target_imag = target_amplitude * (base.real * sine + base.imag * cosine);
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `sine` 由 `imag *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `cosine` 由 `imag *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`target_real`|目标回波复数的实部|记作 $\Re\{x_0[p,n]\}$|由参考波形和多普勒旋转计算，写入 noiseless/echo|
|`target_imag`|目标回波复数的虚部|记作 $\Im\{x_0[p,n]\}$|与 target_real 组成复回波|
|`sine`|当前相位的正弦值|记作 $\sin\phi$|与 cosine 共同执行复数相位旋转|
|`cosine`|当前相位的余弦值|记作 $\cos\phi$|与 sine 共同执行复数相位旋转|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `target_real = target_amplitude * (base.real * cosine - base.imag * sine);` 是赋值：先求得右侧完整表达式 `target_amplitude * (base.real * cosine - base.imag * sine)`，再把结果写入左侧可修改对象 `target_real`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `target_imag = target_amplitude * (base.real * sine + base.imag * cosine);` 是赋值：先求得右侧完整表达式 `target_amplitude * (base.real * sine + base.imag * cosine)`，再把结果写入左侧可修改对象 `target_imag`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `target_real`, `target_imag`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);`。

```cpp
    const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);
    const float noise_real = uniform_noise(base_index, seed, noise_std);
    const float noise_imag = uniform_noise(base_index + 1U, seed, noise_std);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `base_index` 由 `const std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；`const` 禁止通过该名称修改对象；
- `noise_real` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `noise_imag` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1U` 是数值字面量；U 指定 unsigned int 起始类型。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`base_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`noise_real`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`noise_imag`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1U`|`U` 使整数字面量从 unsigned int 候选类型开始|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);` 是声明并初始化：`const std::uint32_t base_index` 建立局部对象 `base_index`，右侧完整表达式 `static_cast<std::uint32_t>(2 * index)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `const float noise_real = uniform_noise(base_index, seed, noise_std);` 是声明并初始化：`const float noise_real` 建立局部对象 `noise_real`，右侧完整表达式 `uniform_noise(base_index, seed, noise_std)` 产生初值。
- `const float noise_imag = uniform_noise(base_index + 1U, seed, noise_std);` 是声明并初始化：`const float noise_imag` 建立局部对象 `noise_imag`，右侧完整表达式 `uniform_noise(base_index + 1U, seed, noise_std)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `uniform_noise`；写入/初始化 `base_index`, `noise_real`, `noise_imag`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.19 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `noiseless[index] = {target_real, target_imag};`。

```cpp
    noiseless[index] = {target_real, target_imag};
    noise[index] = {noise_real, noise_imag};
    echo[index] = {target_real + noise_real, target_imag + noise_imag};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `noiseless`、`index`、`target_real`、`target_imag`、`noise`、`noise_real`、`noise_imag`、`echo`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `noiseless[index] = {target_real, target_imag};` 是赋值：先求得右侧完整表达式 `{target_real, target_imag}`，再把结果写入左侧可修改对象 `noiseless[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `noise[index] = {noise_real, noise_imag};` 是赋值：先求得右侧完整表达式 `{noise_real, noise_imag}`，再把结果写入左侧可修改对象 `noise[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `echo[index] = {target_real + noise_real, target_imag + noise_imag};` 是赋值：先求得右侧完整表达式 `{target_real + noise_real, target_imag + noise_imag}`，再把结果写入左侧可修改对象 `echo[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `noiseless[index]`, `noise[index]`, `echo[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `}`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.21 simulate_echo_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `simulate_echo_kernel`；源码锚点 `}  // namespace`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `StepEvidence run_step1(PipelineState& state)`。

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
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step1(PipelineState& state) {` 中的 `run_step1` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `run_step1`；调用 `run_step1`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.23 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step1";
    const auto formal_begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `evidence` 由 `StepEvidence` 声明：类型控制可表示值、可用操作和传参方式；
- `formal_begin` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`formal_begin`|正式端到端计时起点|无独立算法符号；时间戳可记作 $t_0$|与结束时间相减形成 formal_execution_ms|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `evidence.name = "step1";` 是赋值：先求得右侧完整表达式 `"step1"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto formal_begin = Clock::now();` 是声明并初始化：`const auto formal_begin` 建立局部对象 `formal_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `formal_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step1` 所在的 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    const float doppler_hz = config.doppler_hz();
    const int count = config.num_pulses * config.samples_per_pulse;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `doppler_hz` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`doppler_hz`|目标多普勒频移，单位 Hz|记作 $f_D$|进入 $e^{j2\pi f_Dt}$|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `const float doppler_hz = config.doppler_hz();` 是声明并初始化：`const float doppler_hz` 建立局部对象 `doppler_hz`，右侧完整表达式 `config.doppler_hz()` 产生初值。
- `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象 `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `Clock::now`, `doppler_hz`；写入/初始化 `begin`, `doppler_hz`, `count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.25 run_step1：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `run_step1`；源码锚点 `evidence.prep_ms = milliseconds(begin, Clock::now());`。

```cpp
    evidence.prep_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_waveform(config.pulse_samples);
    evidence.operator_ms["gpu_lfm_generation"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            generate_lfm_kernel, static_cast<std::size_t>(config.pulse_samples),
            d_waveform.data(), config.pulse_samples,
            config.sample_rate_hz, config.bandwidth_hz);
    });
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`prep_ms`、`milliseconds`、`begin`、`Clock`、`now`、`cusignal`、`DeviceArray`、`TypedComplex`、`float`、`d_waveform`、`config`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_waveform(config.pulse_samples);` 声明 `d_waveform`，其静态类型是 `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `config.pulse_samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_waveform` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["gpu_lfm_generation"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( generate_lfm_kernel, static_cast<std::size_t>(config.pulse_samples), d_waveform.data(), config.pulse_samples, config.sample_rate_hz, config.bandwidth_hz); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["gpu_lfm_generation"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`, `d_waveform`, `time_gpu`, `cusignal::cuda_utils::launch_1d_kernel`；写入/初始化 `prep_ms`, `operator_ms["gpu_lfm_generation"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
- kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。

### 4.26 d_noiseless：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);`。

```cpp
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noise(count);
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_echo(count);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`TypedComplex`、`float`、`d_noiseless`、`count`、`d_noise`、`d_echo`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_noiseless`|未加噪目标回波|记作 $x_0[p,n]$|用于区分信号与噪声贡献及测试证据|
|`d_noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`d_echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noiseless(count);` 声明 `d_noiseless`，其静态类型是 `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_noiseless` 的函数，模板尖括号也不是比较或位移。
- `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_noise(count);` 声明 `d_noise`，其静态类型是 `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_noise` 的函数，模板尖括号也不是比较或位移。
- `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_echo(count);` 声明 `d_echo`，其静态类型是 `cusignal::DeviceArray<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_echo` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|公共基础设施|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `d_noiseless`, `d_noise`, `d_echo`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.27 d_noiseless：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

```cpp
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.28 d_noiseless：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            simulate_echo_kernel, static_cast<std::size_t>(count),
            d_waveform.data(), d_noiseless.data(), d_noise.data(), d_echo.data(),
            count, config.target_delay_samples, doppler_hz, config.noise_seed,
            config.samples_per_pulse, config.pulse_samples, config.prf_hz,
            config.sample_rate_hz, config.target_amplitude, config.noise_std);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`gpu_delay_doppler_noise`、`time_gpu`、`cusignal`、`cuda_utils`、`launch_1d_kernel`、`simulate_echo_kernel`、`std`、`size_t`、`count`、`d_waveform`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.operator_ms["gpu_delay_doppler_noise"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( simulate_echo_kernel, static_cast<std::size_t>(count), d_waveform.data(), d_noiseless.data(), d_noise.data(), d_echo.data(), count, config.target_delay_samples, doppler_hz, config.noise_seed, config.samples_per_pulse, config.pulse_samples, config.prf_hz, config.sample_rate_hz, config.target_amplitude, config.noise_std); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["gpu_delay_doppler_noise"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|输入准备/数据契约|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::cuda_utils::launch_1d_kernel`, `data`；写入/初始化 `operator_ms["gpu_delay_doppler_noise"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。

### 4.29 d_noiseless：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `evidence.compute_ms = std::accumulate(`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `compute_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.30 d_noiseless：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    state.waveform = d_waveform.to_host();
    state.noiseless_echo = d_noiseless.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`waveform`、`d_waveform`、`to_host`、`noiseless_echo`、`d_noiseless`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.waveform = d_waveform.to_host();` 是赋值：先求得右侧完整表达式 `d_waveform.to_host()`，再把结果写入左侧可修改对象 `state.waveform`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.noiseless_echo = d_noiseless.to_host();` 是赋值：先求得右侧完整表达式 `d_noiseless.to_host()`，再把结果写入左侧可修改对象 `state.noiseless_echo`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`；写入/初始化 `begin`, `waveform`, `noiseless_echo`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.31 d_noiseless：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `state.noise = d_noise.to_host();`。

```cpp
    state.noise = d_noise.to_host();
    state.echo = d_echo.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`noise`、`d_noise`、`to_host`、`echo`、`d_echo`、`evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.noise = d_noise.to_host();` 是赋值：先求得右侧完整表达式 `d_noise.to_host()`，再把结果写入左侧可修改对象 `state.noise`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.echo = d_echo.to_host();` 是赋值：先求得右侧完整表达式 `d_echo.to_host()`，再把结果写入左侧可修改对象 `state.echo`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `to_host`, `milliseconds`, `Clock::now`；写入/初始化 `noise`, `echo`, `d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.32 d_noiseless：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());`。

```cpp
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `evidence` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(formal_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.formal_execution_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `formal_execution_ms`；返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.33 d_noiseless：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step1/step1.cu`；符号 `d_noiseless`；源码锚点 `}  // namespace task1`。

```cpp

}  // namespace task1
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task1`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}  // namespace task1` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|核心算法或本步数据准备|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.34 cpu_noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `cpu_noise_bits`；源码锚点 `std::uint32_t cpu_noise_bits(std::uint32_t index, std::uint32_t seed)`。

```cpp
std::uint32_t cpu_noise_bits(std::uint32_t index, std::uint32_t seed)
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

- `std::uint32_t cpu_noise_bits(std::uint32_t index, std::uint32_t seed) {` 中的 `cpu_noise_bits` 是函数定义；名称前的 `std::uint32_t` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::uint32_t value = seed ^ (index * 747796405U + 2891336453U);` 是声明并初始化：`std::uint32_t value` 建立局部对象 `value`，右侧完整表达式 `seed ^ (index * 747796405U + 2891336453U)` 产生初值。 `^` 是逐位异或，不是乘方；它把两个无符号整数的比特混合。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `cpu_noise_bits`；调用 `cpu_noise_bits`；写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 先执行乘加再与 seed 组合，是为了让相邻 `index` 不直接形成相邻整数状态，并避免零输入停留在简单零模式；这一步采用 PCG 常见 LCG 常数作为低成本初始打散，后续再由 Murmur 风格 avalanche 完成扩散。来源：[PCG 原始技术报告](https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.35 cpu_noise_bits：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `cpu_noise_bits`；源码锚点 `value ^= value >> 16;`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.36 cpu_noise_bits：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `cpu_noise_bits`；源码锚点 `value *= 3266489917U;`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `value`；返回 `value ^ (value >> 16)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 移位把高位比特搬到低位位置，异或把搬来的模式折叠进原状态；连续执行可形成 avalanche，使 index/seed 的小变化影响多个输出位。这里追求的是确定性、低成本和 CPU/GPU 一致的噪声种子混合，不是密码学安全。移位次序与 `fmix32` 相似，但仓库乘数不同，不能称为原始 MurmurHash3；对照见 [MurmurHash3 原始 `fmix32` 实现](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)（访问日期：2026-08-19）。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 在无符号整数混合代码里，`^` 是逐位异或、`>>` 是右移；二者都不是数学乘方。

### 4.37 cpu_uniform_noise：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `cpu_uniform_noise`；源码锚点 `float cpu_uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)`。

```cpp
float cpu_uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std)
{
    return (static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) /
            32767.5F - 1.0F) * noise_std;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `noise_std` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `0xffffU` 是数值字面量；U 指定 unsigned int 起始类型；
- `32767.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`noise_std`|噪声幅度尺度/标准差参数|记作 $\sigma$|缩放单位伪随机样本|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0xffffU`|`U` 使整数字面量从 unsigned int 候选类型开始|十六进制 `0xFFFF` 等于十进制 65535，二进制低 16 位全为 1。按位与把 32 位混合状态截取为低 16 位无符号量；改变掩码会改变保留位数、离散状态数和后续归一化范围。|
|`32767.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|它是 16 位无符号范围 $[0,65535]$ 的中点与半跨度：$65535/2=32767.5$。执行 `(u-32767.5)/32767.5` 可把端点线性映射到约 $[-1,1]$；改变它会引入偏置或改变噪声幅度。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|这里的 1 是归一化平移量：$u/32767.5$ 的范围约为 $[0,2]$，再减 1 得到 $[-1,1]$。去掉它会使噪声均值偏到约 1 而不是 0。|

**执行过程**

- `float cpu_uniform_noise(std::uint32_t index, std::uint32_t seed, float noise_std) {` 中的 `cpu_uniform_noise` 是函数定义；名称前的 `float` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `std::uint32_t index, std::uint32_t seed, float noise_std` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return (static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std;` 是返回语句：先求值 `(static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `cpu_uniform_noise`；调用 `cpu_uniform_noise`, `return`, `cpu_noise_bits`；返回 `(static_cast<float>(cpu_noise_bits(index, seed) & 0xffffU) / 32767.5F - 1.0F) * noise_std`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
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

### 4.38 cpu_uniform_noise：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `cpu_uniform_noise`；源码锚点 `}`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.39 waveform_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `waveform_cpu`；源码锚点 `std::vector<cusignal::TypedComplex<float>> waveform_cpu(const TaskConfig& config)`。

```cpp
std::vector<cusignal::TypedComplex<float>> waveform_cpu(const TaskConfig& config)
{
    std::vector<cusignal::TypedComplex<float>> output(config.pulse_samples);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<cusignal::TypedComplex<float>> waveform_cpu(const TaskConfig& config) {` 中的 `waveform_cpu` 是函数定义；名称前的 `std::vector<cusignal::TypedComplex<float>>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<cusignal::TypedComplex<float>> output(config.pulse_samples);` 声明 `output`，其静态类型是 `std::vector<cusignal::TypedComplex<float>>`，并用 `config.pulse_samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `waveform_cpu`；调用 `waveform_cpu`, `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.40 waveform_cpu：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `waveform_cpu`；源码锚点 `const double pulse_width = static_cast<double>(config.pulse_samples) / config.sample_rate_hz;`。

```cpp
    const double pulse_width = static_cast<double>(config.pulse_samples) / config.sample_rate_hz;
    const double slope = static_cast<double>(config.bandwidth_hz) / pulse_width;
    for (int index = 0; index < config.pulse_samples; ++index) {
        const double time = static_cast<double>(index) / config.sample_rate_hz - 0.5 * pulse_width;
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：声明初始化或赋值语句、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `pulse_width` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `slope` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `index` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `time` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0.5` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`pulse_width`|脉冲持续时间，单位 s|记作 $T=N_p/f_s$|用于 LFM 相位和距离分辨率关系|
|`slope`|LFM 调频斜率|记作 $k=B/T$，单位 Hz/s|进入二次相位 $\phi(t)=\pi kt^2$|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`0.5`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|

**执行过程**

- `const double pulse_width = static_cast<double>(config.pulse_samples) / config.sample_rate_hz;` 是声明并初始化：`const double pulse_width` 建立局部对象 `pulse_width`，右侧完整表达式 `static_cast<double>(config.pulse_samples) / config.sample_rate_hz` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `const double slope = static_cast<double>(config.bandwidth_hz) / pulse_width;` 是声明并初始化：`const double slope` 建立局部对象 `slope`，右侧完整表达式 `static_cast<double>(config.bandwidth_hz) / pulse_width` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `for (int index = 0; index < config.pulse_samples; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index < config.pulse_samples`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `const double time = static_cast<double>(index) / config.sample_rate_hz - 0.5 * pulse_width;` 是声明并初始化：`const double time` 建立局部对象 `time`，右侧完整表达式 `static_cast<double>(index) / config.sample_rate_hz - 0.5 * pulse_width` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `pulse_width`, `slope`, `time`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.41 waveform_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `waveform_cpu`；源码锚点 `const double phase = 3.14159265358979323846 * slope * time * time;`。

```cpp
        const double phase = 3.14159265358979323846 * slope * time * time;
        output[index] = {static_cast<float>(std::cos(phase)),
                         static_cast<float>(std::sin(phase))};
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `phase` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `time` 由 `time *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `3.14159265358979323846` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`3.14159265358979323846`|无后缀，因此为 `double` 字面量|这是圆周率 $\pi$ 的十进制近似，在当前代码中用于 LFM/Doppler 相位；改动会按比例改变相位和生成波形。|

**执行过程**

- `const double phase = 3.14159265358979323846 * slope * time * time;` 是声明并初始化：`const double phase` 建立局部对象 `phase`，右侧完整表达式 `3.14159265358979323846 * slope * time * time` 产生初值。
- `output[index] = {static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};` 是赋值：先求得右侧完整表达式 `{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `std::cos`, `std::sin`；写入/初始化 `phase`, `output[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.42 waveform_cpu：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `waveform_cpu`；源码锚点 `}`。

```cpp
    }
    return output;
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：作用域边界、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.43 echo_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `std::vector<cusignal::TypedComplex<float>> echo_cpu(`。

```cpp
std::vector<cusignal::TypedComplex<float>> echo_cpu(
    const std::vector<cusignal::TypedComplex<float>>& waveform,
    int delay, float doppler_hz, std::uint32_t seed,
    const TaskConfig& config,
    std::vector<cusignal::TypedComplex<float>>* noiseless,
    std::vector<cusignal::TypedComplex<float>>* noise)
{
    const int count = config.num_pulses * config.samples_per_pulse;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `waveform` 由 `const std::vector<cusignal::TypedComplex<float>>&` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `delay` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `doppler_hz` 由 `float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `seed` 由 `std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；
- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `noiseless` 由 `std::vector<cusignal::TypedComplex<float>>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `noise` 由 `std::vector<cusignal::TypedComplex<float>>*` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`delay`|目标离散延迟，单位 sample|记作 $d$|回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$|
|`doppler_hz`|目标多普勒频移，单位 Hz|记作 $f_D$|进入 $e^{j2\pi f_Dt}$|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|
|`noiseless`|未加噪目标回波|记作 $x_0[p,n]$|用于区分信号与噪声贡献及测试证据|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<cusignal::TypedComplex<float>> echo_cpu( const std::vector<cusignal::TypedComplex<float>>& waveform, int delay, float doppler_hz, std::uint32_t seed, const TaskConfig& config, std::vector<cusignal::TypedComplex<float>>* noiseless, std::vector<cusignal::TypedComplex<float>>* noise) {` 中的 `echo_cpu` 是函数定义；名称前的 `std::vector<cusignal::TypedComplex<float>>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<cusignal::TypedComplex<float>>& waveform, int delay, float doppler_hz, std::uint32_t seed, const TaskConfig& config, std::vector<cusignal::TypedComplex<float>>* noiseless, std::vector<cusignal::TypedComplex<float>>* noise` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象 `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `echo_cpu`；调用 `echo_cpu`；写入/初始化 `count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.44 echo_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `noiseless->assign(count, {0.0F, 0.0F});`。

```cpp
    noiseless->assign(count, {0.0F, 0.0F});
    noise->assign(count, {0.0F, 0.0F});
    std::vector<cusignal::TypedComplex<float>> output(count);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
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

- `noiseless->assign(count, {0.0F, 0.0F});` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
- `noise->assign(count, {0.0F, 0.0F});` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
- `std::vector<cusignal::TypedComplex<float>> output(count);` 声明 `output`，其静态类型是 `std::vector<cusignal::TypedComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `assign`, `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.45 echo_cpu：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `for (int index = 0; index < count; ++index) {`。

```cpp
    for (int index = 0; index < count; ++index) {
        const int pulse = index / config.samples_per_pulse;
        const int sample = index % config.samples_per_pulse;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `pulse` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `sample` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`pulse`|当前慢时间脉冲编号|记作 $p=\lfloor i/N_s\rfloor$|由展平索引整除每脉冲采样数得到|
|`sample`|当前脉冲内快时间采样编号|记作 $n=i\bmod N_s$|由展平索引取余得到|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `for (int index = 0; index < count; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index < count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `const int pulse = index / config.samples_per_pulse;` 是声明并初始化：`const int pulse` 建立局部对象 `pulse`，右侧完整表达式 `index / config.samples_per_pulse` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
- `const int sample = index % config.samples_per_pulse;` 是声明并初始化：`const int sample` 建立局部对象 `sample`，右侧完整表达式 `index % config.samples_per_pulse` 产生初值。 `%` 取得整数除法余数，用来把展平索引还原为当前行/脉冲内的位置。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `pulse`, `sample`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。
- 取余的分母不能为零；这里依赖每脉冲采样数/维度已通过配置校验。

### 4.46 echo_cpu：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `const int source = sample - delay;`。

```cpp
        const int source = sample - delay;
        if (source >= 0 && source < config.pulse_samples) {
            const double time = static_cast<double>(pulse) / config.prf_hz +
                static_cast<double>(sample) / config.sample_rate_hz;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `source` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `time` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`source`|施加延迟前回读的波形采样位置|记作 $n-d$|只有落在波形有效区间时才形成目标回波|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `const int source = sample - delay;` 是声明并初始化：`const int source` 建立局部对象 `source`，右侧完整表达式 `sample - delay` 产生初值。
- `if (source >= 0 && source < config.pulse_samples) {` 是 `if` 条件语句：先把完整条件 `source >= 0 && source < config.pulse_samples` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `const double time = static_cast<double>(pulse) / config.prf_hz + static_cast<double>(sample) / config.sample_rate_hz;` 是声明并初始化：`const double time` 建立局部对象 `time`，右侧完整表达式 `static_cast<double>(pulse) / config.prf_hz + static_cast<double>(sample) / config.sample_rate_hz` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过检查 `source >= 0 && source < config.pulse_samples`；写入/初始化 `source`, `time`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.47 echo_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;`。

```cpp
            const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `phase` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `time` 由 `doppler_hz *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `cosine` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `sine` 由 `const double` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；
- `2.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3.14159265358979323846` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`phase`|复指数的相位，单位 rad|记作 $\phi$|通过 cos/sin 形成复波形或多普勒旋转|
|`cosine`|当前相位的余弦值|记作 $\cos\phi$|与 sine 共同执行复数相位旋转|
|`sine`|当前相位的正弦值|记作 $\sin\phi$|与 cosine 共同执行复数相位旋转|
|`time`|当前样本对应的物理时间，单位 s|记作 $t$|进入相位 $2\pi f_Dt$ 或 LFM 相位|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`3.14159265358979323846`|无后缀，因此为 `double` 字面量|这是圆周率 $\pi$ 的十进制近似，在当前代码中用于 LFM/Doppler 相位；改动会按比例改变相位和生成波形。|

**执行过程**

- `const double phase = 2.0 * 3.14159265358979323846 * doppler_hz * time;` 是声明并初始化：`const double phase` 建立局部对象 `phase`，右侧完整表达式 `2.0 * 3.14159265358979323846 * doppler_hz * time` 产生初值。
- `const double cosine = std::cos(phase);` 是声明并初始化：`const double cosine` 建立局部对象 `cosine`，右侧完整表达式 `std::cos(phase)` 产生初值。
- `const double sine = std::sin(phase);` 是声明并初始化：`const double sine` 建立局部对象 `sine`，右侧完整表达式 `std::sin(phase)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `std::cos`, `std::sin`；写入/初始化 `phase`, `cosine`, `sine`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.48 echo_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `(*noiseless)[index] = {`。

```cpp
            (*noiseless)[index] = {
                static_cast<float>(config.target_amplitude *
                    (waveform[source].real * cosine - waveform[source].imag * sine)),
                static_cast<float>(config.target_amplitude *
                    (waveform[source].real * sine + waveform[source].imag * cosine))};
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `sine` 由 `imag *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `cosine` 由 `imag *` 声明：类型控制可表示值、可用操作和传参方式；`*` 说明变量保存地址，解引用前必须保证地址有效。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`sine`|当前相位的正弦值|记作 $\sin\phi$|与 cosine 共同执行复数相位旋转|
|`cosine`|当前相位的余弦值|记作 $\cos\phi$|与 sine 共同执行复数相位旋转|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `(*noiseless)[index] = { static_cast<float>(config.target_amplitude * (waveform[source].real * cosine - waveform[source].imag * sine)), static_cast<float>(config.target_amplitude * (waveform[source].real * sine + waveform[source].imag * cosine))};` 是赋值：先求得右侧完整表达式 `{ static_cast<float>(config.target_amplitude * (waveform[source].real * cosine - waveform[source].imag * sine)), static_cast<float>(config.target_amplitude * (waveform[source].real * sine + waveform[source].imag * cosine))}`，再把结果写入左侧可修改对象 `(*noiseless)[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `(*noiseless)[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.49 echo_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `}`。

```cpp
        }
        const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);
        (*noise)[index] = {
            cpu_uniform_noise(base_index, seed, config.noise_std),
            cpu_uniform_noise(base_index + 1U, seed, config.noise_std)};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：作用域边界、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `base_index` 由 `const std::uint32_t` 声明：固定宽度 32 位无符号整数，可表示 $0$ 到 $2^{32}-1$；算术按模 $2^{32}$ 回绕；`const` 禁止通过该名称修改对象；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1U` 是数值字面量；U 指定 unsigned int 起始类型。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`base_index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|
|`seed`|确定性伪随机种子|记作 $s_0$|来自配置；与索引共同生成可复现噪声|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1U`|`U` 使整数字面量从 unsigned int 候选类型开始|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `const std::uint32_t base_index = static_cast<std::uint32_t>(2 * index);` 是声明并初始化：`const std::uint32_t base_index` 建立局部对象 `base_index`，右侧完整表达式 `static_cast<std::uint32_t>(2 * index)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `(*noise)[index] = { cpu_uniform_noise(base_index, seed, config.noise_std), cpu_uniform_noise(base_index + 1U, seed, config.noise_std)};` 是赋值：先求得右侧完整表达式 `{ cpu_uniform_noise(base_index, seed, config.noise_std), cpu_uniform_noise(base_index + 1U, seed, config.noise_std)}`，再把结果写入左侧可修改对象 `(*noise)[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `cpu_uniform_noise`；写入/初始化 `base_index`, `(*noise)[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.50 echo_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `output[index] = {`。

```cpp
        output[index] = {
            (*noiseless)[index].real + (*noise)[index].real,
            (*noiseless)[index].imag + (*noise)[index].imag};
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `output`、`index`、`noiseless`、`real`、`noise`、`imag`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`noise`|加性噪声数组|记作 $w[p,n]$|由 seed/index 确定性生成并叠加到 noiseless|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `output[index] = { (*noiseless)[index].real + (*noise)[index].real, (*noiseless)[index].imag + (*noise)[index].imag};` 是赋值：先求得右侧完整表达式 `{ (*noiseless)[index].real + (*noise)[index].real, (*noiseless)[index].imag + (*noise)[index].imag}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过写入/初始化 `output[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.51 echo_cpu：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `echo_cpu`；源码锚点 `}`。

```cpp
    }
    return output;
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：作用域边界、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.52 generate_step1_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `generate_step1_cpu_reference`；源码锚点 `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config)`。

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
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config) {` 中的 `generate_step1_cpu_reference` 是函数定义；名称前的 `Step1ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step1ValidationData data;` 是对象声明：类型 `Step1ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过定义/声明 `generate_step1_cpu_reference`；调用 `generate_step1_cpu_reference`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.53 generate_step1_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `generate_step1_cpu_reference`；源码锚点 `data.waveform_cpu = waveform_cpu(config);`。

```cpp
    data.waveform_cpu = waveform_cpu(config);
    const float doppler_hz = config.doppler_hz();
    data.echo_cpu = echo_cpu(
        data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed, config,
        &data.noiseless_cpu, &data.noise_cpu);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `doppler_hz` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`doppler_hz`|目标多普勒频移，单位 Hz|记作 $f_D$|进入 $e^{j2\pi f_Dt}$|
|`config`|外部配置对象|`config` 在本 step 实例化脉冲数/每脉冲采样数、$f_s$、带宽 $B$、延迟 $d$、多普勒 $f_D$、目标幅度、噪声尺度和 seed。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.waveform_cpu = waveform_cpu(config);` 是赋值：先求得右侧完整表达式 `waveform_cpu(config)`，再把结果写入左侧可修改对象 `data.waveform_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const float doppler_hz = config.doppler_hz();` 是声明并初始化：`const float doppler_hz` 建立局部对象 `doppler_hz`，右侧完整表达式 `config.doppler_hz()` 产生初值。
- `data.echo_cpu = echo_cpu( data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed, config, &data.noiseless_cpu, &data.noise_cpu);` 是赋值：先求得右侧完整表达式 `echo_cpu( data.waveform_cpu, config.target_delay_samples, doppler_hz, config.noise_seed, config, &data.noiseless_cpu, &data.noise_cpu)`，再把结果写入左侧可修改对象 `data.echo_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过调用 `waveform_cpu`, `doppler_hz`, `echo_cpu`；写入/初始化 `waveform_cpu`, `doppler_hz`, `echo_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.54 generate_step1_cpu_reference：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step1_validation.cu`；符号 `generate_step1_cpu_reference`；源码锚点 `return data;`。

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
|赛题条款|T1-R1：接收信号模拟：构造包含目标延迟、多普勒频移和噪声的雷达回波|
|业务角色|测试证据|
|业务输入|外部配置中的采样率、带宽、脉冲数、目标延迟/速度、幅度、噪声尺度和 seed|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|发射参考波形 `waveform`、无噪回波 `noiseless`、噪声 `noise` 与叠加回波 `echo`|
|下一消费者|`echo` 与 `waveform` 交给 T1-R2 脉冲压缩|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
