# Task2 Step4 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task2 Step4 的复输入 adapter、FM/相关/谱/CWT 四路正式算子调用、谱与小波包络、归一化融合，以及对应 CPU reference。

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
|`ZKX/Task2/task2_gpu_cpu/step4/step4.h`|`run_step4` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step4/step4.cu`|`analytic_signal_kernel`、`spectral_envelope`、`wavelet_envelope`、`make_bundle`、`run_step4`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`|上述任务私有 helper 的 CPU 等价路径、`collect_step4_validation_impl`、`collect_step4_validation`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step4/step4.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
//  `#include "../task2_common.h"` 在预处理阶段引入 "../task2_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
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

#include "../task2_common.h"

// <学习注释：语义块：文件级代码：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
//  `StepEvidence run_step4(PipelineState& state);` 中的 `run_step4` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。
//
//  【变量—数学符号—数据流】
//  run_step4：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
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
namespace task2 {

StepEvidence run_step4(PipelineState& state);

// <学习注释：语义块：文件级代码：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}  // namespace task2` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}  // namespace task2

```

### 3.2 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step4.h"` 在预处理阶段引入 "step4.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "convolution/convolution_typed.h"` 在预处理阶段引入 "convolution/convolution_typed.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/kernel_launch.h"` 在预处理阶段引入 "cuda_utils/kernel_launch.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "demod/demod_typed.h"` 在预处理阶段引入 "demod/demod_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "spectral_analysis/spectral_analysis_typed.h"` 在预处理阶段引入
//  "spectral_analysis/spectral_analysis_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "wavelets/wavelets_typed.h"` 在预处理阶段引入 "wavelets/wavelets_typed.h"
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
#include "step4.h"

#include "convolution/convolution_typed.h"
#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
#include "demod/demod_typed.h"
#include "spectral_analysis/spectral_analysis_typed.h"
#include "wavelets/wavelets_typed.h"

// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
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

namespace task2 {
namespace {

// <学习注释：语义块：analytic_signal_kernel：声明 CUDA kernel 并建立线程入口。
//
//  【执行方法】
//  `__global__ void analytic_signal_kernel( const float* input, cusignal::DemodComplex<float>* output, int
//  count) {` 中的 `analytic_signal_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  float* input, cusignal::DemodComplex<float>* output, int count` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU
//  上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index`
//  建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。
//  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x`
//  把块号和块内线程号映射为一维全局线程索引。
//
//  【变量—数学符号—数据流】
//  input：当前函数或调用的参数名称，接收调用者绑定的输入 input；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const float* input,
//  cusignal::DemodComplex<float>* output, int count)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  output：当前函数或步骤的输出容器；对应当前算法公式的左端结果，具体符号由所在 step 决定；由本块写入，返回或交给下一步骤
//  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//
//  【数学/物理公式对照】
//  本 step 私有 adapter 构造 $z[n]=x[n]+j\frac{x[n+1]-x[n-1]}2$；端点复制最近样本。它是中心差分虚部的工程性复表示，不是 Hilbert 变换得到的严格解析信号。每线程读邻域并写一个 `DemodComplex<float>`。
//
//  【为什么这样设计】
//  `fm_demod` 需要复输入；该近似无需额外 FFT，但物理含义不同于 Hilbert 解析信号，文档必须保留这一差异。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
__global__ void analytic_signal_kernel(
    const float* input, cusignal::DemodComplex<float>* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // <学习注释：语义块：analytic_signal_kernel：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前
    //  CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
    //  `const float previous = index == 0 ? input[0] : input[index - 1];` 是声明并初始化：`const float previous` 建立局部对象
    //  `previous`，右侧完整表达式 `index == 0 ? input[0] : input[index - 1]` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。
    //  `const float next = index + 1 == count ? input[count - 1] : input[index + 1];` 是声明并初始化：`const float next`
    //  建立局部对象 `next`，右侧完整表达式 `index + 1 == count ? input[count - 1] : input[index + 1]` 产生初值。 `条件 ? 真分支 : 假分支`
    //  是条件运算符：只求值两个候选分支中的一个。
    //
    //  【变量—数学符号—数据流】
    //  previous：当前块的赋值左端，保存右侧表达式产生的中间值或结果 previous；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const float previous =
    //  index == 0 ? input[0] : input[index - 1];；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  next：当前块的赋值左端，保存右侧表达式产生的中间值或结果 next；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const float next = index + 1 ==
    //  count ? input[count - 1] : input[index + 1];；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    if (index >= count) return;
    const float previous = index == 0 ? input[0] : input[index - 1];
    const float next = index + 1 == count ? input[count - 1] : input[index + 1];
    // <学习注释：语义块：analytic_signal_kernel：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `output[index] = {input[index], 0.5F * (next - previous)};` 是赋值：先求得右侧完整表达式 `{input[index], 0.5F * (next -
    //  previous)}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //
    //  【数字常量】
    //  0.5F：F 使浮点字面量为 float；不带后缀默认是 double；表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。
    //
    //  【数学/物理公式对照】
    //  本 step 私有 adapter 构造 $z[n]=x[n]+j\frac{x[n+1]-x[n-1]}2$；端点复制最近样本。它是中心差分虚部的工程性复表示，不是 Hilbert 变换得到的严格解析信号。每线程读邻域并写一个 `DemodComplex<float>`。
    //
    //  【为什么这样设计】
    //  `fm_demod` 需要复输入；该近似无需额外 FFT，但物理含义不同于 Hilbert 解析信号，文档必须保留这一差异。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    output[index] = {input[index], 0.5F * (next - previous)};
}

// <学习注释：语义块：spectral_envelope：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<float> spectral_envelope( const std::vector<float>& flattened, int frequencies, int frames)
//  {` 中的 `spectral_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  std::vector<float>& flattened, int frequencies, int frames` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);` 声明 `output`，其静态类型是
//  `std::vector<float>`，并用 `static_cast<std::size_t>(frames), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
//  的函数，模板尖括号也不是比较或位移。
//
//  【变量—数学符号—数据流】
//  flattened：当前函数或调用的参数名称，接收调用者绑定的输入 flattened；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  flattened, int frequencies, int frames)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  frequencies：当前函数或调用的参数名称，接收调用者绑定的输入 frequencies；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
//  std::vector<float>& flattened, int frequencies, int frames)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu
//  当前作用域中产生或消费
//  frames：当前函数或调用的参数名称，接收调用者绑定的输入 frames；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  flattened, int frequencies, int frames)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//
//  【数字常量】
//  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
//
//  【数学/物理公式对照】
//  `output[frame]=max_frequency flattened[frequency*frames+frame]`，把二维谱 $S[k,m]$ 沿频率轴取最大值得一维 $e_s[m]$；它是任务私有降维，不是 spectrogram 算子本身。
//
//  【为什么这样设计】
//  最大包络保留每帧最强频率能量但丢失峰值所在频率；替代方案如求和/带宽统计会保留不同信息。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
    // <学习注释：语义块：spectral_envelope：迭代处理数据范围。
    //
    //  【执行方法】
    //  `for (int frequency = 0; frequency < frequencies; ++frequency) for (int frame = 0; frame < frames;
    //  ++frame) output[frame] = std::max(output[frame], flattened[static_cast<std::size_t>(frequency) * frames +
    //  frame]);` 是经典 `for`：先执行初始化 `int frequency = 0`；每轮前检查 `frequency < frequencies`，为假即退出；每轮循环体结束后执行
    //  `++frequency`，再检查下一轮。
    //
    //  【变量—数学符号—数据流】
    //  frequency：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frequency；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 for (int frequency = 0;
    //  frequency < frequencies; ++frequency)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  frame：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frame；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 for (int frame = 0; frame <
    //  frames; ++frame)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
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
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
    // <学习注释：语义块：spectral_envelope：形成并返回当前结果。
    //
    //  【执行方法】
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
    return output;
}

// <学习注释：语义块：wavelet_envelope：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<float> wavelet_envelope( const std::vector<double>& flattened, int widths, int samples) {` 中的
//  `wavelet_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  std::vector<double>& flattened, int widths, int samples` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);` 声明 `output`，其静态类型是
//  `std::vector<float>`，并用 `static_cast<std::size_t>(samples), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
//  的函数，模板尖括号也不是比较或位移。
//
//  【变量—数学符号—数据流】
//  flattened：当前函数或调用的参数名称，接收调用者绑定的输入 flattened；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<double>&
//  flattened, int widths, int samples)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  widths：当前函数或调用的参数名称，接收调用者绑定的输入 widths；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<double>&
//  flattened, int widths, int samples)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  samples：任务信号总采样点数；记作 $N$；配置决定数组 shape、计算量和证据规模
//
//  【数字常量】
//  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
//
//  【数学/物理公式对照】
//  `output[sample]=max_width |W(width,sample)|`，把 CWT 沿尺度轴取绝对值最大值；fp64 系数先取绝对值再窄化为 fp32。
//
//  【为什么这样设计】
//  最大尺度包络给每个时刻的最强小波响应，但丢失最佳尺度；窄化不会改变 shape，却限制后续有效精度。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
    // <学习注释：语义块：wavelet_envelope：迭代处理数据范围。
    //
    //  【执行方法】
    //  `for (int width = 0; width < widths; ++width) for (int sample = 0; sample < samples; ++sample)
    //  output[sample] = std::max(output[sample],
    //  cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(
    //  flattened[static_cast<std::size_t>(width) * samples + sample])));` 是经典 `for`：先执行初始化 `int width = 0`；每轮前检查
    //  `width < widths`，为假即退出；每轮循环体结束后执行 `++width`，再检查下一轮。
    //
    //  【变量—数学符号—数据流】
    //  width：当前块的赋值左端，保存右侧表达式产生的中间值或结果 width；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 for (int width = 0; width <
    //  widths; ++width)；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  sample：当前脉冲内快时间采样编号；记作 $n=i\bmod N_s$；由展平索引取余得到
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
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample],
                cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(
                    flattened[static_cast<std::size_t>(width) * samples + sample])));
    // <学习注释：语义块：wavelet_envelope：形成并返回当前结果。
    //
    //  【执行方法】
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
    return output;
}

// <学习注释：语义块：make_bundle：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<float> make_bundle( const std::vector<float>& fm, const std::vector<float>& correlation,
//  const std::vector<float>& spectral, const std::vector<float>& wavelet, std::size_t count, const
//  TaskConfig& config) {` 中的 `make_bundle` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的
//  `const std::vector<float>& fm, const std::vector<float>& correlation, const std::vector<float>& spectral,
//  const std::vector<float>& wavelet, std::size_t count, const TaskConfig& config` 是形参声明。 末尾 `{`
//  打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto fm_n = normalize_abs(resample_linear(fm, count));` 是声明并初始化：`const auto fm_n` 建立局部对象
//  `fm_n`，右侧完整表达式 `normalize_abs(resample_linear(fm, count))` 产生初值。
//
//  【变量—数学符号—数据流】
//  fm：当前函数或调用的参数名称，接收调用者绑定的输入 fm；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>& fm,；值在
//  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
//  spectral：当前函数或调用的参数名称，接收调用者绑定的输入 spectral；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  spectral,；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  wavelet：当前函数或调用的参数名称，接收调用者绑定的输入 wavelet；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  wavelet,；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
//  config：外部配置对象；`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。；入口加载，runner 和各 step 只读消费
//  fm_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 fm_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto fm_n =
//  normalize_abs(resample_linear(fm, count));；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//
//  【数学/物理公式对照】
//  四路特征先线性重采样到长度 $N=count$，再经 `normalize_abs` 得 $f_i[n]$，融合 $b[n]=w_f f_f[n]+w_c f_c[n]+w_s f_s[n]+w_w f_w[n]$。配置字段逐项对应四个权重，`bundle`/`feature_bundle` 是 Step5 输入。
//
//  【为什么这样设计】
//  先同长、同幅值口径再加权可避免某一路因长度或量纲主导；权重是否和为 1 由配置契约决定，当前代码没有在此强制。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<float> make_bundle(
    const std::vector<float>& fm,
    const std::vector<float>& correlation,
    const std::vector<float>& spectral,
    const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
    // <学习注释：语义块：make_bundle：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const auto corr_n = normalize_abs(resample_linear(correlation, count));` 是声明并初始化：`const auto corr_n`
    //  建立局部对象 `corr_n`，右侧完整表达式 `normalize_abs(resample_linear(correlation, count))` 产生初值。
    //  `const auto spec_n = normalize_abs(resample_linear(spectral, count));` 是声明并初始化：`const auto spec_n` 建立局部对象
    //  `spec_n`，右侧完整表达式 `normalize_abs(resample_linear(spectral, count))` 产生初值。
    //  `const auto wave_n = normalize_abs(resample_linear(wavelet, count));` 是声明并初始化：`const auto wave_n` 建立局部对象
    //  `wave_n`，右侧完整表达式 `normalize_abs(resample_linear(wavelet, count))` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  corr_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 corr_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto corr_n =
    //  normalize_abs(resample_linear(correlation, count));；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  spec_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 spec_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto spec_n =
    //  normalize_abs(resample_linear(spectral, count));；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  wave_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 wave_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto wave_n =
    //  normalize_abs(resample_linear(wavelet, count));；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
    // <学习注释：语义块：bundle：迭代处理数据范围。
    //
    //  【执行方法】
    //  `std::vector<float> bundle(count);` 声明 `bundle`，其静态类型是 `std::vector<float>`，并用 `count`
    //  直接初始化/调用该类型的构造函数；这不是调用名为 `bundle` 的函数，模板尖括号也不是比较或位移。
    //  `for (std::size_t index = 0; index < count; ++index) bundle[index] = config.fusion_weight_fm * fm_n[index]
    //  + config.fusion_weight_correlation * corr_n[index] + config.fusion_weight_spectral * spec_n[index] +
    //  config.fusion_weight_wavelet * wave_n[index];` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index <
    //  count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  config：外部配置对象；`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  四路特征先线性重采样到长度 $N=count$，再经 `normalize_abs` 得 $f_i[n]$，融合 $b[n]=w_f f_f[n]+w_c f_c[n]+w_s f_s[n]+w_w f_w[n]$。配置字段逐项对应四个权重，`bundle`/`feature_bundle` 是 Step5 输入。
    //
    //  【为什么这样设计】
    //  先同长、同幅值口径再加权可避免某一路因长度或量纲主导；权重是否和为 1 由配置契约决定，当前代码没有在此强制。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
    // <学习注释：语义块：bundle：形成并返回当前结果。
    //
    //  【执行方法】
    //  `return bundle;` 是返回语句：先求值 `bundle`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  bundle：成组保存多域特征的结构/数组集合；可记作 $\{f_m[n]\}_m$；由 Step4 形成，Step5/6 读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    return bundle;
}

// <学习注释：语义块：bundle：完成一个连续的数据处理动作。
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

// <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `StepEvidence run_step4(PipelineState& state) {` 中的 `run_step4` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  run_step4：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
StepEvidence run_step4(PipelineState& state)
{
    StepEvidence evidence;
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.name = "step4";` 是赋值：先求得右侧完整表达式 `"step4"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式
    //  `Clock::now()` 产生初值。
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  total_begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.name = "step4";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const int count = static_cast<int>(state.smoothed.size());` 是声明并初始化：`const int count` 建立局部对象
    //  `count`，右侧完整表达式 `static_cast<int>(state.smoothed.size())` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数
    //  `T`，转换发生后才参与外层表达式。
    //  `require_task2_upstream(count > 64, "Step4", "Step3.smoothed");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续
    //  step/字段名称报告缺失或非法输入。
    //  `require_task2_upstream( state.reference_for_correlation.size() == state.smoothed.size(), "Step4",
    //  "Step3.reference_for_correlation");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //
    //  【变量—数学符号—数据流】
    //  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数字常量】
    //  64：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const int count = static_cast<int>(state.smoothed.size());
    require_task2_upstream(count > 64, "Step4", "Step3.smoothed");
    require_task2_upstream(
        state.reference_for_correlation.size() == state.smoothed.size(),
        "Step4", "Step3.reference_for_correlation");
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `constexpr int frame = 64;` 是声明并初始化：`constexpr int frame` 建立局部对象 `frame`，右侧完整表达式 `64` 产生初值。 `constexpr`
    //  要求该初值可在编译期形成常量表达式。
    //  `constexpr int hop = 32;` 是声明并初始化：`constexpr int hop` 建立局部对象 `hop`，右侧完整表达式 `32` 产生初值。 `constexpr`
    //  要求该初值可在编译期形成常量表达式。
    //  `const int frames = 1 + (count - frame) / hop;` 是声明并初始化：`const int frames` 建立局部对象 `frames`，右侧完整表达式 `1 +
    //  (count - frame) / hop` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
    //
    //  【变量—数学符号—数据流】
    //  frame：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frame；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 constexpr int frame = 64;；值在
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  hop：当前块的赋值左端，保存右侧表达式产生的中间值或结果 hop；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 constexpr int hop = 32;；值在
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  frames：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frames；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const int frames = 1 + (count
    //  - frame) / hop;；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //
    //  【数字常量】
    //  64：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //  32：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
    constexpr int frame = 64;
    constexpr int hop = 32;
    const int frames = 1 + (count - frame) / hop;
    // <学习注释：语义块：run_step4：形成并返回当前结果。
    //
    //  【执行方法】
    //  `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};` 声明 `widths`，静态类型为 `const
    //  std::vector<float>`，并使用列表初始化器 `{2.0F, 4.0F, 8.0F, 12.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。 当前向量随后作为 CWT/Ricker
    //  的尺度参数：`2.0F、4.0F、8.0F、12.0F` 是四个离散尺度，源码固定采用这组值但没有证明其唯一最优。
    //  `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) { return
    //  cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width)); };` 声明 callable 对象 `ricker`，并用
    //  lambda 表达式初始化；`[]` 表示不捕获外部变量，`(int points, int width)` 声明调用参数，花括号内 `return
    //  cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));` 是每次调用时才执行的函数体。定义 lambda
    //  本身不会立即运行该函数体。
    //
    //  【变量—数学符号—数据流】
    //  widths：当前表达式读取或传递的工程名称 widths；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float> widths{2.0F,
    //  4.0F, 8.0F, 12.0F};；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  ricker：当前块的赋值左端，保存右侧表达式产生的中间值或结果 ricker；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
    //  cusignal::CwtRealWaveletCallable ricker =；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  points：当前函数或调用的参数名称，接收调用者绑定的输入 points；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 [](int points, int width) {；值在
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  width：当前函数或调用的参数名称，接收调用者绑定的输入 width；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 [](int points, int width) {；值在
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //
    //  【数字常量】
    //  2.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  4.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  8.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  12.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //
    //  【数学/物理公式对照】
    //  Ricker callable 为各宽度生成实小波；`widths={2,4,8,12}` 对应尺度 $a$。正式 CWT 计算 $W(a,b)=a^{-1/2}Σ_nx[n]ψ^*((n-b)/a)$，`smoothed/d_signal→x`，输出按 `width*samples+sample` 展平为 double。
    //
    //  【为什么这样设计】
    //  workspace 预生成/保存各尺度卷积所需数据；内部 Ricker 与 CWT 实现分别归 `Learning/operators/wavelets/ricker/`、`cwt/`。尺度值是仓库策略，源码未记录推导依据。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker =
        [](int points, int width) {
            return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
        };
        // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);` 是声明并初始化：`auto
        //  cwt_workspace` 建立局部对象 `cwt_workspace`，右侧完整表达式 `cusignal::prepare_cwt_workspace(count, widths, ricker)`
        //  产生初值。
        //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
        //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //
        //  【变量—数学符号—数据流】
        //  cwt_workspace：当前块的赋值左端，保存右侧表达式产生的中间值或结果 cwt_workspace；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 auto
        //  cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);；值在
        //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
        //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
        //
        //  【数学/物理公式对照】
        //  Ricker callable 为各宽度生成实小波；`widths={2,4,8,12}` 对应尺度 $a$。正式 CWT 计算 $W(a,b)=a^{-1/2}Σ_nx[n]ψ^*((n-b)/a)$，`smoothed/d_signal→x`，输出按 `width*samples+sample` 展平为 double。
        //
        //  【为什么这样设计】
        //  workspace 预生成/保存各尺度卷积所需数据；内部 Ricker 与 CWT 实现分别归 `Learning/operators/wavelets/ricker/`、`cwt/`。尺度值是仓库策略，源码未记录推导依据。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);` 是声明并初始化：`auto d_signal` 建立局部对象
    //  `d_signal`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.smoothed)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  d_signal：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
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
    auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);`
    //  是声明并初始化：`auto d_reference` 建立局部对象 `d_reference`，右侧完整表达式
    //  `cusignal::DeviceArray<float>::from_host(state.reference_for_correlation)` 产生初值。
    //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);` 声明 `d_analytic`，其静态类型是
    //  `cusignal::DeviceArray<cusignal::DemodComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_analytic`
    //  的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  d_reference：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  d_analytic：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。>
    auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.operator_ms["analytic_glue"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel(
    //  analytic_signal_kernel, static_cast<std::size_t>(count), d_signal.data(), d_analytic.data(), count); });`
    //  先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入
    //  `evidence.operator_ms["analytic_glue"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  本 step 私有 adapter 构造 $z[n]=x[n]+j\frac{x[n+1]-x[n-1]}2$；端点复制最近样本。它是中心差分虚部的工程性复表示，不是 Hilbert 变换得到的严格解析信号。每线程读邻域并写一个 `DemodComplex<float>`。
    //
    //  【为什么这样设计】
    //  `fm_demod` 需要复输入；该近似无需额外 FFT，但物理含义不同于 Hilbert 解析信号，文档必须保留这一差异。
    //
    //  【初学者易错点】
    //  kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。>
    evidence.operator_ms["analytic_glue"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            analytic_signal_kernel, static_cast<std::size_t>(count),
            d_signal.data(), d_analytic.data(), count);
    });
    // <学习注释：语义块：demod_workspace：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});` 声明
    //  `demod_workspace`，其静态类型是 `cusignal::FmDemodWorkspace`，并用 `cusignal::FmDemodOptions{{count}, -1}`
    //  直接初始化/调用该类型的构造函数；这不是调用名为 `demod_workspace` 的函数，模板尖括号也不是比较或位移。
    //  `cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());` 声明 `d_fm`，其静态类型是
    //  `cusignal::DeviceArray<float>`，并用 `demod_workspace.output_size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_fm`
    //  的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["fm_demod"] = time_gpu([&] { cusignal::fm_demod_device(d_analytic, d_fm,
    //  demod_workspace); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["fm_demod"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  demod_workspace：当前函数或调用的参数名称，接收调用者绑定的输入 demod_workspace；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据
    //  cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});；值在
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  d_fm：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  正式算子计算相邻复样本相位差 $d[n]=\arg(z[n]z^*[n-1])$。`d_analytic/analytic_cpu→z[n]`，输出 `d_fm/fm_cpu→d[n]`；workspace options shape 为 `{count}`、`axis=-1`，因此沿唯一/最后轴解调。
    //
    //  【为什么这样设计】
    //  相位差避免显式全局相位展开；算子内部边界与输出长度归 `Learning/operators/demod/fm_demod/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());
    evidence.operator_ms["fm_demod"] = time_gpu([&] {
        cusignal::fm_demod_device(d_analytic, d_fm, demod_workspace);
    });
    // <学习注释：语义块：d_correlation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<float> d_correlation(count);` 声明 `d_correlation`，其静态类型是
    //  `cusignal::DeviceArray<float>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_correlation` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["correlate"] = time_gpu([&] { cusignal::correlate_device(d_signal, d_reference,
    //  d_correlation, "same"); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["correlate"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子计算 $r_{xy}[k]=Σ_nx[n]y^*[n-k]$。`smoothed/d_signal→x`，`reference_for_correlation/d_reference→y`，mode=`same` 令输出 `correlation_feature` 长度与主输入 `count` 相同；CPU 验证指定 direct 方法。
    //
    //  【为什么这样设计】
    //  相关突出与参考形状相似的位置；内部 lag 对齐和 method 归 `Learning/operators/convolution/correlate/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<float> d_correlation(count);
    evidence.operator_ms["correlate"] = time_gpu([&] {
        cusignal::correlate_device(d_signal, d_reference, d_correlation, "same");
    });
    // <学习注释：语义块：d_spectral：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);` 声明
    //  `d_spectral`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `static_cast<std::size_t>(frame) * frames`
    //  直接初始化/调用该类型的构造函数；这不是调用名为 `d_spectral` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["spectrogram"] = time_gpu([&] { cusignal::spectrogram_device(d_signal, frame, hop,
    //  d_spectral); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["spectrogram"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_spectral：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子以 `frame=64`、`hop=32` 计算分帧谱，$S[m,k]=|Σ_{n=0}^{63}x[n+32m]w[n]e^{-j2πkn/64}|²$（精确缩放/窗以算子接口为准）。`frames=1+(count-64)/32`；展平输出按 `frequency*frames+frame` 读取。
    //
    //  【为什么这样设计】
    //  固定窗长/步长给出 50% 重叠；内部窗和归一化归 `Learning/operators/spectral_analysis/spectrogram/`。源码未记录为何选择 64/32，不能称为理论唯一值。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);
    evidence.operator_ms["spectrogram"] = time_gpu([&] {
        cusignal::spectrogram_device(d_signal, frame, hop, d_spectral);
    });
    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());` 声明 `d_cwt`，其静态类型是
    //  `cusignal::DeviceArray<double>`，并用 `cwt_workspace.output_size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_cwt`
    //  的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["cwt_ricker"] = time_gpu([&] { cusignal::cwt_device(d_signal, cwt_workspace, d_cwt);
    //  });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["cwt_ricker"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_cwt：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  Ricker callable 为各宽度生成实小波；`widths={2,4,8,12}` 对应尺度 $a$。正式 CWT 计算 $W(a,b)=a^{-1/2}Σ_nx[n]ψ^*((n-b)/a)$，`smoothed/d_signal→x`，输出按 `width*samples+sample` 展平为 double。
    //
    //  【为什么这样设计】
    //  workspace 预生成/保存各尺度卷积所需数据；内部 Ricker 与 CWT 实现分别归 `Learning/operators/wavelets/ricker/`、`cwt/`。尺度值是仓库策略，源码未记录推导依据。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());
    evidence.operator_ms["cwt_ricker"] = time_gpu([&] {
        cusignal::cwt_device(d_signal, cwt_workspace, d_cwt);
    });
    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
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
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    // <学习注释：语义块：d_cwt：形成并返回当前结果。
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

    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `state.fm_feature = d_fm.to_host();` 是赋值：先求得右侧完整表达式 `d_fm.to_host()`，再把结果写入左侧可修改对象 `state.fm_feature`；这里的
    //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
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
    state.fm_feature = d_fm.to_host();
    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.correlation_feature = d_correlation.to_host();` 是赋值：先求得右侧完整表达式
    //  `d_correlation.to_host()`，再把结果写入左侧可修改对象 `state.correlation_feature`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `const auto spectral_flat = d_spectral.to_host();` 是声明并初始化：`const auto spectral_flat` 建立局部对象
    //  `spectral_flat`，右侧完整表达式 `d_spectral.to_host()` 产生初值。
    //  `const auto cwt_flat = d_cwt.to_host();` 是声明并初始化：`const auto cwt_flat` 建立局部对象 `cwt_flat`，右侧完整表达式
    //  `d_cwt.to_host()` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  spectral_flat：当前块的赋值左端，保存右侧表达式产生的中间值或结果 spectral_flat；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto
    //  spectral_flat = d_spectral.to_host();；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  cwt_flat：当前块的赋值左端，保存右侧表达式产生的中间值或结果 cwt_flat；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto cwt_flat =
    //  d_cwt.to_host();；值在 ZKX/Task2/task2_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    state.correlation_feature = d_correlation.to_host();
    const auto spectral_flat = d_spectral.to_host();
    const auto cwt_flat = d_cwt.to_host();
    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);` 是赋值：先求得右侧完整表达式
    //  `spectral_envelope(spectral_flat, frame, frames)`，再把结果写入左侧可修改对象 `state.spectral_feature`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  `output[frame]=max_frequency flattened[frequency*frames+frame]`，把二维谱 $S[k,m]$ 沿频率轴取最大值得一维 $e_s[m]$；它是任务私有降维，不是 spectrogram 算子本身。
    //
    //  【为什么这样设计】
    //  最大包络保留每帧最强频率能量但丢失峰值所在频率；替代方案如求和/带宽统计会保留不同信息。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);
    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.wavelet_feature = wavelet_envelope( cwt_flat, static_cast<int>(widths.size()), count);`
    //  是赋值：先求得右侧完整表达式 `wavelet_envelope( cwt_flat, static_cast<int>(widths.size()), count)`，再把结果写入左侧可修改对象
    //  `state.wavelet_feature`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数
    //  `T`，转换发生后才参与外层表达式。
    //  `state.feature_bundle = make_bundle( state.fm_feature, state.correlation_feature, state.spectral_feature,
    //  state.wavelet_feature, count, state.config);` 是赋值：先求得右侧完整表达式 `make_bundle( state.fm_feature,
    //  state.correlation_feature, state.spectral_feature, state.wavelet_feature, count,
    //  state.config)`，再把结果写入左侧可修改对象 `state.feature_bundle`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `output[sample]=max_width |W(width,sample)|`，把 CWT 沿尺度轴取绝对值最大值；fp64 系数先取绝对值再窄化为 fp32。
    //
    //  【为什么这样设计】
    //  最大尺度包络给每个时刻的最强小波响应，但丢失最佳尺度；窄化不会改变 shape，却限制后续有效精度。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    state.wavelet_feature = wavelet_envelope(
        cwt_flat, static_cast<int>(widths.size()), count);
    state.feature_bundle = make_bundle(
        state.fm_feature, state.correlation_feature,
        state.spectral_feature, state.wavelet_feature, count, state.config);

    // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.post_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.post_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
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
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    // <学习注释：语义块：d_cwt：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (visualization_capture_enabled()) {` 是 `if` 条件语句：先把完整条件 `visualization_capture_enabled()` 求值并转换为布尔值。
    //  条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `state.spectrogram_grid = spectral_flat;` 是赋值：先求得右侧完整表达式 `spectral_flat`，再把结果写入左侧可修改对象
    //  `state.spectrogram_grid`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `state.cwt_grid = cwt_flat;` 是赋值：先求得右侧完整表达式 `cwt_flat`，再把结果写入左侧可修改对象 `state.cwt_grid`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    if (visualization_capture_enabled()) {
        state.spectrogram_grid = spectral_flat;
        state.cwt_grid = cwt_flat;
        // <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `state.spectrogram_frequency_bins = frame;` 是赋值：先求得右侧完整表达式 `frame`，再把结果写入左侧可修改对象
        //  `state.spectrogram_frequency_bins`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `state.spectrogram_frames = frames;` 是赋值：先求得右侧完整表达式 `frames`，再把结果写入左侧可修改对象 `state.spectrogram_frames`；这里的
        //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `state.cwt_width_count = static_cast<int>(widths.size());` 是赋值：先求得右侧完整表达式
        //  `static_cast<int>(widths.size())`，再把结果写入左侧可修改对象 `state.cwt_width_count`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
        //
        //  【变量—数学符号—数据流】
        //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
        //
        //  【为什么这样设计】
        //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        state.spectrogram_frequency_bins = frame;
        state.spectrogram_frames = frames;
        state.cwt_width_count = static_cast<int>(widths.size());
    // <学习注释：语义块：d_cwt：形成并返回当前结果。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
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
    return evidence;
}

// <学习注释：语义块：d_cwt：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}  // namespace task2` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}  // namespace task2
```

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`：CPU reference 相关符号

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：analytic_cpu：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input) {` 中的
//  `analytic_cpu` 是函数定义；名称前的 `std::vector<cusignal::DemodComplex<float>>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的
//  `const std::vector<float>& input` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::vector<cusignal::DemodComplex<float>> output(input.size());` 声明 `output`，其静态类型是
//  `std::vector<cusignal::DemodComplex<float>>`，并用 `input.size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
//  的函数，模板尖括号也不是比较或位移。
//
//  【变量—数学符号—数据流】
//  input：当前函数或调用的参数名称，接收调用者绑定的输入 input；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据
//  std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)；值在
//  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
//
//  【为什么这样设计】
//  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
//
//  【初学者易错点】
//  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)
{
    std::vector<cusignal::DemodComplex<float>> output(input.size());
    // <学习注释：语义块：analytic_cpu：迭代处理数据范围。
    //
    //  【执行方法】
    //  `for (std::size_t index = 0; index < input.size(); ++index) {` 是经典 `for`：先执行初始化 `std::size_t index =
    //  0`；每轮前检查 `index < input.size()`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //  `const float previous = index == 0 ? input.front() : input[index - 1];` 是声明并初始化：`const float previous`
    //  建立局部对象 `previous`，右侧完整表达式 `index == 0 ? input.front() : input[index - 1]` 产生初值。 `条件 ? 真分支 : 假分支`
    //  是条件运算符：只求值两个候选分支中的一个。
    //  `const float next = index + 1 == input.size() ? input.back() : input[index + 1];` 是声明并初始化：`const float
    //  next` 建立局部对象 `next`，右侧完整表达式 `index + 1 == input.size() ? input.back() : input[index + 1]` 产生初值。 `条件 ? 真分支
    //  : 假分支` 是条件运算符：只求值两个候选分支中的一个。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  previous：当前块的赋值左端，保存右侧表达式产生的中间值或结果 previous；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const float previous =
    //  index == 0 ? input.front() : input[index - 1];；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  next：当前块的赋值左端，保存右侧表达式产生的中间值或结果 next；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const float next = index + 1 ==
    //  input.size() ? input.back() : input[index + 1];；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float previous = index == 0 ? input.front() : input[index - 1];
        const float next = index + 1 == input.size() ? input.back() : input[index + 1];
        // <学习注释：语义块：analytic_cpu：形成并返回当前结果。
        //
        //  【执行方法】
        //  `output[index] = {input[index], 0.5F * (next - previous)};` 是赋值：先求得右侧完整表达式 `{input[index], 0.5F * (next -
        //  previous)}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
        //  `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
        //
        //  【变量—数学符号—数据流】
        //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
        //  output：当前函数或步骤的输出容器；对应当前算法公式的左端结果，具体符号由所在 step 决定；由本块写入，返回或交给下一步骤
        //
        //  【数字常量】
        //  0.5F：F 使浮点字面量为 float；不带后缀默认是 double；表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。
        //
        //  【数学/物理公式对照】
        //  本 step 私有 adapter 构造 $z[n]=x[n]+j\frac{x[n+1]-x[n-1]}2$；端点复制最近样本。它是中心差分虚部的工程性复表示，不是 Hilbert 变换得到的严格解析信号。每线程读邻域并写一个 `DemodComplex<float>`。
        //
        //  【为什么这样设计】
        //  `fm_demod` 需要复输入；该近似无需额外 FFT，但物理含义不同于 Hilbert 解析信号，文档必须保留这一差异。
        //
        //  【初学者易错点】
        //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
        output[index] = {input[index], 0.5F * (next - previous)};
    }
    return output;
// <学习注释：语义块：analytic_cpu：完成一个连续的数据处理动作。
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

// <学习注释：语义块：spectral_envelope：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<float> spectral_envelope( const std::vector<float>& flattened, int frequencies, int frames)
//  {` 中的 `spectral_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  std::vector<float>& flattened, int frequencies, int frames` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);` 声明 `output`，其静态类型是
//  `std::vector<float>`，并用 `static_cast<std::size_t>(frames), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
//  的函数，模板尖括号也不是比较或位移。
//
//  【变量—数学符号—数据流】
//  flattened：当前函数或调用的参数名称，接收调用者绑定的输入 flattened；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  flattened, int frequencies, int frames)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp
//  当前作用域中产生或消费
//  frequencies：当前函数或调用的参数名称，接收调用者绑定的输入 frequencies；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
//  std::vector<float>& flattened, int frequencies, int frames)；值在
//  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
//  frames：当前函数或调用的参数名称，接收调用者绑定的输入 frames；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  flattened, int frequencies, int frames)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp
//  当前作用域中产生或消费
//
//  【数字常量】
//  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
//
//  【数学/物理公式对照】
//  `output[frame]=max_frequency flattened[frequency*frames+frame]`，把二维谱 $S[k,m]$ 沿频率轴取最大值得一维 $e_s[m]$；它是任务私有降维，不是 spectrogram 算子本身。
//
//  【为什么这样设计】
//  最大包络保留每帧最强频率能量但丢失峰值所在频率；替代方案如求和/带宽统计会保留不同信息。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
    // <学习注释：语义块：spectral_envelope：迭代处理数据范围。
    //
    //  【执行方法】
    //  `for (int frequency = 0; frequency < frequencies; ++frequency) for (int frame = 0; frame < frames;
    //  ++frame) output[frame] = std::max(output[frame], flattened[static_cast<std::size_t>(frequency) * frames +
    //  frame]);` 是经典 `for`：先执行初始化 `int frequency = 0`；每轮前检查 `frequency < frequencies`，为假即退出；每轮循环体结束后执行
    //  `++frequency`，再检查下一轮。
    //
    //  【变量—数学符号—数据流】
    //  frequency：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frequency；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 for (int frequency = 0;
    //  frequency < frequencies; ++frequency)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp
    //  当前作用域中产生或消费
    //  frame：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frame；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 for (int frame = 0; frame <
    //  frames; ++frame)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
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
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
    // <学习注释：语义块：spectral_envelope：形成并返回当前结果。
    //
    //  【执行方法】
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
    return output;
}

// <学习注释：语义块：wavelet_envelope：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<float> wavelet_envelope( const std::vector<double>& flattened, int widths, int samples) {` 中的
//  `wavelet_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  std::vector<double>& flattened, int widths, int samples` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);` 声明 `output`，其静态类型是
//  `std::vector<float>`，并用 `static_cast<std::size_t>(samples), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output`
//  的函数，模板尖括号也不是比较或位移。
//
//  【变量—数学符号—数据流】
//  flattened：当前函数或调用的参数名称，接收调用者绑定的输入 flattened；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<double>&
//  flattened, int widths, int samples)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp
//  当前作用域中产生或消费
//  widths：当前函数或调用的参数名称，接收调用者绑定的输入 widths；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<double>&
//  flattened, int widths, int samples)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp
//  当前作用域中产生或消费
//  samples：任务信号总采样点数；记作 $N$；配置决定数组 shape、计算量和证据规模
//
//  【数字常量】
//  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
//
//  【数学/物理公式对照】
//  `output[sample]=max_width |W(width,sample)|`，把 CWT 沿尺度轴取绝对值最大值；fp64 系数先取绝对值再窄化为 fp32。
//
//  【为什么这样设计】
//  最大尺度包络给每个时刻的最强小波响应，但丢失最佳尺度；窄化不会改变 shape，却限制后续有效精度。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
    // <学习注释：语义块：wavelet_envelope：迭代处理数据范围。
    //
    //  【执行方法】
    //  `for (int width = 0; width < widths; ++width) for (int sample = 0; sample < samples; ++sample)
    //  output[sample] = std::max(output[sample], static_cast<float>(std::abs(
    //  flattened[static_cast<std::size_t>(width) * samples + sample])));` 是经典 `for`：先执行初始化 `int width = 0`；每轮前检查
    //  `width < widths`，为假即退出；每轮循环体结束后执行 `++width`，再检查下一轮。
    //
    //  【变量—数学符号—数据流】
    //  width：当前块的赋值左端，保存右侧表达式产生的中间值或结果 width；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 for (int width = 0; width <
    //  widths; ++width)；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  sample：当前脉冲内快时间采样编号；记作 $n=i\bmod N_s$；由展平索引取余得到
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
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample], static_cast<float>(std::abs(
                flattened[static_cast<std::size_t>(width) * samples + sample])));
    // <学习注释：语义块：wavelet_envelope：形成并返回当前结果。
    //
    //  【执行方法】
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
    return output;
}

// <学习注释：语义块：make_bundle：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `std::vector<float> make_bundle( const std::vector<float>& fm, const std::vector<float>& correlation,
//  const std::vector<float>& spectral, const std::vector<float>& wavelet, std::size_t count, const
//  TaskConfig& config) {` 中的 `make_bundle` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的
//  `const std::vector<float>& fm, const std::vector<float>& correlation, const std::vector<float>& spectral,
//  const std::vector<float>& wavelet, std::size_t count, const TaskConfig& config` 是形参声明。 末尾 `{`
//  打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto fm_n = normalize_abs(resample_linear(fm, count));` 是声明并初始化：`const auto fm_n` 建立局部对象
//  `fm_n`，右侧完整表达式 `normalize_abs(resample_linear(fm, count))` 产生初值。
//
//  【变量—数学符号—数据流】
//  fm：当前函数或调用的参数名称，接收调用者绑定的输入 fm；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>& fm, const
//  std::vector<float>& correlation,；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp
//  当前作用域中产生或消费
//  correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
//  spectral：当前函数或调用的参数名称，接收调用者绑定的输入 spectral；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  spectral, const std::vector<float>& wavelet,；值在
//  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
//  wavelet：当前函数或调用的参数名称，接收调用者绑定的输入 wavelet；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>&
//  spectral, const std::vector<float>& wavelet,；值在
//  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
//  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
//  config：外部配置对象；`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。；入口加载，runner 和各 step 只读消费
//  fm_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 fm_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto fm_n =
//  normalize_abs(resample_linear(fm, count));；值在
//  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
//
//  【数学/物理公式对照】
//  四路特征先线性重采样到长度 $N=count$，再经 `normalize_abs` 得 $f_i[n]$，融合 $b[n]=w_f f_f[n]+w_c f_c[n]+w_s f_s[n]+w_w f_w[n]$。配置字段逐项对应四个权重，`bundle`/`feature_bundle` 是 Step5 输入。
//
//  【为什么这样设计】
//  先同长、同幅值口径再加权可避免某一路因长度或量纲主导；权重是否和为 1 由配置契约决定，当前代码没有在此强制。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<float> make_bundle(
    const std::vector<float>& fm, const std::vector<float>& correlation,
    const std::vector<float>& spectral, const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
    // <学习注释：语义块：make_bundle：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const auto corr_n = normalize_abs(resample_linear(correlation, count));` 是声明并初始化：`const auto corr_n`
    //  建立局部对象 `corr_n`，右侧完整表达式 `normalize_abs(resample_linear(correlation, count))` 产生初值。
    //  `const auto spec_n = normalize_abs(resample_linear(spectral, count));` 是声明并初始化：`const auto spec_n` 建立局部对象
    //  `spec_n`，右侧完整表达式 `normalize_abs(resample_linear(spectral, count))` 产生初值。
    //  `const auto wave_n = normalize_abs(resample_linear(wavelet, count));` 是声明并初始化：`const auto wave_n` 建立局部对象
    //  `wave_n`，右侧完整表达式 `normalize_abs(resample_linear(wavelet, count))` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  corr_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 corr_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto corr_n =
    //  normalize_abs(resample_linear(correlation, count));；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  spec_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 spec_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto spec_n =
    //  normalize_abs(resample_linear(spectral, count));；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  wave_n：当前块的赋值左端，保存右侧表达式产生的中间值或结果 wave_n；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const auto wave_n =
    //  normalize_abs(resample_linear(wavelet, count));；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
    // <学习注释：语义块：bundle：迭代处理数据范围。
    //
    //  【执行方法】
    //  `std::vector<float> bundle(count);` 声明 `bundle`，其静态类型是 `std::vector<float>`，并用 `count`
    //  直接初始化/调用该类型的构造函数；这不是调用名为 `bundle` 的函数，模板尖括号也不是比较或位移。
    //  `for (std::size_t index = 0; index < count; ++index) bundle[index] = config.fusion_weight_fm * fm_n[index]
    //  + config.fusion_weight_correlation * corr_n[index] + config.fusion_weight_spectral * spec_n[index] +
    //  config.fusion_weight_wavelet * wave_n[index];` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index <
    //  count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  config：外部配置对象；`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  四路特征先线性重采样到长度 $N=count$，再经 `normalize_abs` 得 $f_i[n]$，融合 $b[n]=w_f f_f[n]+w_c f_c[n]+w_s f_s[n]+w_w f_w[n]$。配置字段逐项对应四个权重，`bundle`/`feature_bundle` 是 Step5 输入。
    //
    //  【为什么这样设计】
    //  先同长、同幅值口径再加权可避免某一路因长度或量纲主导；权重是否和为 1 由配置契约决定，当前代码没有在此强制。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
    // <学习注释：语义块：bundle：形成并返回当前结果。
    //
    //  【执行方法】
    //  `return bundle;` 是返回语句：先求值 `bundle`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【变量—数学符号—数据流】
    //  bundle：成组保存多域特征的结构/数组集合；可记作 $\{f_m[n]\}_m$；由 Step4 形成，Step5/6 读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    return bundle;
}

// <学习注释：语义块：bundle：完成一个连续的数据处理动作。
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

// <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step4ValidationData collect_step4_validation_impl( const PipelineState& state, std::vector<float>*
//  precomputed_correlation) {` 中的 `collect_step4_validation_impl` 是函数定义；名称前的 `Step4ValidationData`
//  包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state, std::vector<float>* precomputed_correlation`
//  是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `Step4ValidationData data;` 是对象声明：类型 `Step4ValidationData` 应用于名称
//  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  collect_step4_validation_impl：测试、收集或契约检查 helper；无独立数学变量；内部指标分别映射到误差/条件公式；消费运行结果并形成证据或失败路径
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  precomputed_correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
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
Step4ValidationData collect_step4_validation_impl(
    const PipelineState& state, std::vector<float>* precomputed_correlation)
{
    Step4ValidationData data;
    // <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const int count = static_cast<int>(state.smoothed.size());` 是声明并初始化：`const int count` 建立局部对象
    //  `count`，右侧完整表达式 `static_cast<int>(state.smoothed.size())` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数
    //  `T`，转换发生后才参与外层表达式。
    //  `constexpr int frame = 64;` 是声明并初始化：`constexpr int frame` 建立局部对象 `frame`，右侧完整表达式 `64` 产生初值。 `constexpr`
    //  要求该初值可在编译期形成常量表达式。
    //  `constexpr int hop = 32;` 是声明并初始化：`constexpr int hop` 建立局部对象 `hop`，右侧完整表达式 `32` 产生初值。 `constexpr`
    //  要求该初值可在编译期形成常量表达式。
    //
    //  【变量—数学符号—数据流】
    //  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  frame：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frame；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 constexpr int frame = 64;；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  hop：当前块的赋值左端，保存右侧表达式产生的中间值或结果 hop；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 constexpr int hop = 32;；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数字常量】
    //  64：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //  32：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const int count = static_cast<int>(state.smoothed.size());
    constexpr int frame = 64;
    constexpr int hop = 32;
    // <学习注释：语义块：collect_step4_validation_impl：形成并返回当前结果。
    //
    //  【执行方法】
    //  `const int frames = 1 + (count - frame) / hop;` 是声明并初始化：`const int frames` 建立局部对象 `frames`，右侧完整表达式 `1 +
    //  (count - frame) / hop` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
    //  `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};` 声明 `widths`，静态类型为 `const
    //  std::vector<float>`，并使用列表初始化器 `{2.0F, 4.0F, 8.0F, 12.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。 当前向量随后作为 CWT/Ricker
    //  的尺度参数：`2.0F、4.0F、8.0F、12.0F` 是四个离散尺度，源码固定采用这组值但没有证明其唯一最优。
    //  `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) { return
    //  cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width)); };` 声明 callable 对象 `ricker`，并用
    //  lambda 表达式初始化；`[]` 表示不捕获外部变量，`(int points, int width)` 声明调用参数，花括号内 `return
    //  cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));` 是每次调用时才执行的函数体。定义 lambda
    //  本身不会立即运行该函数体。
    //
    //  【变量—数学符号—数据流】
    //  frames：当前块的赋值左端，保存右侧表达式产生的中间值或结果 frames；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const int frames = 1 + (count
    //  - frame) / hop;；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  widths：当前表达式读取或传递的工程名称 widths；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float> widths{2.0F,
    //  4.0F, 8.0F, 12.0F};；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  ricker：当前块的赋值左端，保存右侧表达式产生的中间值或结果 ricker；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
    //  cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  points：当前函数或调用的参数名称，接收调用者绑定的输入 points；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
    //  cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  width：当前函数或调用的参数名称，接收调用者绑定的输入 width；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
    //  cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //  2.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  4.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  8.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  12.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //
    //  【数学/物理公式对照】
    //  Ricker callable 为各宽度生成实小波；`widths={2,4,8,12}` 对应尺度 $a$。正式 CWT 计算 $W(a,b)=a^{-1/2}Σ_nx[n]ψ^*((n-b)/a)$，`smoothed/d_signal→x`，输出按 `width*samples+sample` 展平为 double。
    //
    //  【为什么这样设计】
    //  workspace 预生成/保存各尺度卷积所需数据；内部 Ricker 与 CWT 实现分别归 `Learning/operators/wavelets/ricker/`、`cwt/`。尺度值是仓库策略，源码未记录推导依据。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
    //  整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。>
    const int frames = 1 + (count - frame) / hop;
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {
        return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
    };
    // <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});` 声明
    //  `demod_workspace`，其静态类型是 `cusignal::FmDemodWorkspace`，并用 `cusignal::FmDemodOptions{{count}, -1}`
    //  直接初始化/调用该类型的构造函数；这不是调用名为 `demod_workspace` 的函数，模板尖括号也不是比较或位移。
    //  `data.fm_cpu = cusignal::fm_demod_typed_cpu( analytic_cpu(state.smoothed), demod_workspace);`
    //  是赋值：先求得右侧完整表达式 `cusignal::fm_demod_typed_cpu( analytic_cpu(state.smoothed),
    //  demod_workspace)`，再把结果写入左侧可修改对象 `data.fm_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  demod_workspace：当前表达式读取或传递的工程名称 demod_workspace；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据
    //  cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  正式算子计算相邻复样本相位差 $d[n]=\arg(z[n]z^*[n-1])$。`d_analytic/analytic_cpu→z[n]`，输出 `d_fm/fm_cpu→d[n]`；workspace options shape 为 `{count}`、`axis=-1`，因此沿唯一/最后轴解调。
    //
    //  【为什么这样设计】
    //  相位差避免显式全局相位展开；算子内部边界与输出长度归 `Learning/operators/demod/fm_demod/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    data.fm_cpu = cusignal::fm_demod_typed_cpu(
        analytic_cpu(state.smoothed), demod_workspace);
    // <学习注释：语义块：collect_step4_validation_impl：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (precomputed_correlation) {` 是 `if` 条件语句：先把完整条件 `precomputed_correlation` 求值并转换为布尔值。
    //  条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `if (precomputed_correlation->size() != state.smoothed.size()) throw std::invalid_argument( "precomputed
    //  Task2 Step4 correlation size mismatch");` 是 `if` 条件语句：先把完整条件 `precomputed_correlation->size() !=
    //  state.smoothed.size()` 求值并转换为布尔值。 条件为真时构造并抛出 `std::invalid_argument( "precomputed Task2 Step4 correlation
    //  size mismatch")`，正常顺序执行立即中断并进入异常传播；条件为假时跳过抛出语句。
    //
    //  【变量—数学符号—数据流】
    //  Step4：当前函数或调用的参数名称，接收调用者绑定的输入 Step4；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 "precomputed Task2 Step4
    //  correlation size mismatch");；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp 当前作用域中产生或消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  correlation：输入与参考的离散相关序列；记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$；用于定位时延特征
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    if (precomputed_correlation) {
        if (precomputed_correlation->size() != state.smoothed.size())
            throw std::invalid_argument(
                "precomputed Task2 Step4 correlation size mismatch");
        // <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `data.correlation_cpu = std::move(*precomputed_correlation);` 是赋值：先求得右侧完整表达式
        //  `std::move(*precomputed_correlation)`，再把结果写入左侧可修改对象 `data.correlation_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
        //  `!=`。
        //  `} else {` 先用 `}` 关闭前一个 `if` 分支，再由 `else {` 打开互斥的假分支；只有前一个条件为假时才执行该分支。
        //  `data.correlation_cpu = cusignal::correlate_typed_cpu( state.smoothed, state.reference_for_correlation,
        //  "same", cusignal::CorrelateMethod::direct);` 是赋值：先求得右侧完整表达式 `cusignal::correlate_typed_cpu(
        //  state.smoothed, state.reference_for_correlation, "same", cusignal::CorrelateMethod::direct)`，再把结果写入左侧可修改对象
        //  `data.correlation_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //
        //  【变量—数学符号—数据流】
        //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
        //
        //  【数学/物理公式对照】
        //  正式算子计算 $r_{xy}[k]=Σ_nx[n]y^*[n-k]$。`smoothed/d_signal→x`，`reference_for_correlation/d_reference→y`，mode=`same` 令输出 `correlation_feature` 长度与主输入 `count` 相同；CPU 验证指定 direct 方法。
        //
        //  【为什么这样设计】
        //  相关突出与参考形状相似的位置；内部 lag 对齐和 method 归 `Learning/operators/convolution/correlate/`。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        data.correlation_cpu = std::move(*precomputed_correlation);
    } else {
        data.correlation_cpu = cusignal::correlate_typed_cpu(
            state.smoothed, state.reference_for_correlation, "same",
            cusignal::CorrelateMethod::direct);
    // <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `data.spectral_cpu = spectral_envelope( cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop),
    //  frame, frames);` 是赋值：先求得右侧完整表达式 `spectral_envelope( cusignal::spectrogram_typed_cpu(state.smoothed, frame,
    //  hop), frame, frames)`，再把结果写入左侧可修改对象 `data.spectral_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  正式算子以 `frame=64`、`hop=32` 计算分帧谱，$S[m,k]=|Σ_{n=0}^{63}x[n+32m]w[n]e^{-j2πkn/64}|²$（精确缩放/窗以算子接口为准）。`frames=1+(count-64)/32`；展平输出按 `frequency*frames+frame` 读取。
    //
    //  【为什么这样设计】
    //  固定窗长/步长给出 50% 重叠；内部窗和归一化归 `Learning/operators/spectral_analysis/spectrogram/`。源码未记录为何选择 64/32，不能称为理论唯一值。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    }
    data.spectral_cpu = spectral_envelope(
        cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames);
    // <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.wavelet_cpu = wavelet_envelope( cusignal::cwt_typed_cpu(state.smoothed, widths, ricker),
    //  static_cast<int>(widths.size()), count);` 是赋值：先求得右侧完整表达式 `wavelet_envelope(
    //  cusignal::cwt_typed_cpu(state.smoothed, widths, ricker), static_cast<int>(widths.size()),
    //  count)`，再把结果写入左侧可修改对象 `data.wavelet_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)`
    //  明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  Ricker callable 为各宽度生成实小波；`widths={2,4,8,12}` 对应尺度 $a$。正式 CWT 计算 $W(a,b)=a^{-1/2}Σ_nx[n]ψ^*((n-b)/a)$，`smoothed/d_signal→x`，输出按 `width*samples+sample` 展平为 double。
    //
    //  【为什么这样设计】
    //  workspace 预生成/保存各尺度卷积所需数据；内部 Ricker 与 CWT 实现分别归 `Learning/operators/wavelets/ricker/`、`cwt/`。尺度值是仓库策略，源码未记录推导依据。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.wavelet_cpu = wavelet_envelope(
        cusignal::cwt_typed_cpu(state.smoothed, widths, ricker),
        static_cast<int>(widths.size()), count);
    // <学习注释：语义块：collect_step4_validation_impl：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.bundle_cpu = make_bundle( data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu,
    //  count, state.config);` 是赋值：先求得右侧完整表达式 `make_bundle( data.fm_cpu, data.correlation_cpu, data.spectral_cpu,
    //  data.wavelet_cpu, count, state.config)`，再把结果写入左侧可修改对象 `data.bundle_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  四路特征先线性重采样到长度 $N=count$，再经 `normalize_abs` 得 $f_i[n]$，融合 $b[n]=w_f f_f[n]+w_c f_c[n]+w_s f_s[n]+w_w f_w[n]$。配置字段逐项对应四个权重，`bundle`/`feature_bundle` 是 Step5 输入。
    //
    //  【为什么这样设计】
    //  先同长、同幅值口径再加权可避免某一路因长度或量纲主导；权重是否和为 1 由配置契约决定，当前代码没有在此强制。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.bundle_cpu = make_bundle(
        data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu,
        count, state.config);
    // <学习注释：语义块：collect_step4_validation_impl：形成并返回当前结果。
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

// <学习注释：语义块：collect_step4_validation：形成并返回当前结果。
//
//  【执行方法】
//  `Step4ValidationData collect_step4_validation(const PipelineState& state) {` 中的 `collect_step4_validation`
//  是函数定义；名称前的 `Step4ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾
//  `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `return collect_step4_validation_impl(state, nullptr);` 是返回语句：先求值 `collect_step4_validation_impl(state,
//  nullptr)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
//
//  【变量—数学符号—数据流】
//  collect_step4_validation：测试、收集或契约检查 helper；无独立数学变量；内部指标分别映射到误差/条件公式；消费运行结果并形成证据或失败路径
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    return collect_step4_validation_impl(state, nullptr);
// <学习注释：语义块：collect_step4_validation：完成一个连续的数据处理动作。
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

### 3.4 函数总体处理方法

#### 3.4.1 总体业务流水线

`Step3 smoothed→FM/相关/谱/CWT 四路特征→融合→D2H`。输出 feature bundle/fused 给 Step5。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`analytic_signal_kernel`|每线程把一个实数平滑样本写成虚部为零的复解析输入|GPU device adapter kernel|
|`spectral_envelope`|沿频率轴对每帧取最大值，把二维谱压成一维谱包络|Host feature adapter|
|`wavelet_envelope`|沿尺度轴对每个样本取绝对值最大值，把 CWT 压成一维包络|Host feature adapter|
|`make_bundle`|重采样、幅值归一化并按配置权重融合四路特征|Host feature fusion|
|`run_step4`|调度 FM、相关、谱和 CWT 四路算子并融合特征|GPU Host wrapper/业务入口|
|`analytic_cpu`|构造与 device adapter 同口径的 CPU 复解析输入|CPU reference adapter|
|`collect_step4_validation_impl`|按指定扰动/输入路径计算 Step4 CPU/GPU 对照数据|CPU/GPU validation core|
|`collect_step4_validation`|以正式输入调用 Step4 验证实现并返回结果|CPU/GPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `fm_demod`

- 学习入口：[数学物理原理](../../operators/demod/fm_demod/数学物理原理.md)、[Python 源码算法](../../operators/demod/fm_demod/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/demod/fm_demod/cusignal_cpp_fm_demod复现逻辑.md)。
- 当前调用采用的数学关系：复解析信号的瞬时相位差可写为 $d[n]=\arg(x[n]x^*[n-1])$。
- 任务变量与参数实例化：输入 analytic signal，输出 FM 特征。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 四域正式调用与融合前后关系

|分支|正式调用与本次参数|输出 dtype/shape|任务私有后处理|
|---|---|---|---|
|FM|`fm_demod_device(d_analytic,d_fm, workspace{{count},-1})`；$d[n]=\arg(z[n]z^*[n-1])$|`float[output_size]`|线性重采样、绝对值归一化|
|相关|`correlate_device(d_signal,d_reference,d_correlation,"same")`；$r_{xy}[k]=\sum_nx[n]y^*[n-k]$|`float[count]`|绝对值归一化|
|谱|`spectrogram_device(d_signal,64,32,d_spectral)`|展平 `float[64×frames]`，$frames=1+(count-64)/32$|沿 64 个频率 bin 取最大值|
|小波|`cwt_device(d_signal,cwt_workspace,d_cwt)`；widths `{2,4,8,12}`，Ricker callable|展平 `double[4×count]`|沿尺度取 $|W|$ 最大值并窄化 fp32|

`analytic_signal_kernel` 形成的是 $z[n]=x[n]+j(x[n+1]-x[n-1])/2$ 的任务私有近似复输入，并非 Hilbert 解析信号。四路经重采样和 `normalize_abs` 后按
$b[n]=w_f f_f[n]+w_c f_c[n]+w_s f_s[n]+w_w f_w[n]$ 融合成 `feature_bundle[count]`；配置权重是否归一化由配置契约决定，本函数没有强制其和为 1。

##### `correlate`

- 学习入口：[数学物理原理](../../operators/convolution/correlate/数学物理原理.md)、[Python 源码算法](../../operators/convolution/correlate/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/convolution/correlate/cusignal_cpp_correlate复现逻辑.md)。
- 当前调用采用的数学关系：$r_{xy}[k]=\sum_nx[n]y^*[n-k]$。
- 任务变量与参数实例化：`signal/reference→x/y`，`correlation→r_xy`，mode=`same` 保持任务长度。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

##### `spectrogram`

- 学习入口：[数学物理原理](../../operators/spectral_analysis/spectrogram/数学物理原理.md)、[Python 源码算法](../../operators/spectral_analysis/spectrogram/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/spectral_analysis/spectrogram/cusignal_cpp_spectrogram复现逻辑.md)。
- 当前调用采用的数学关系：$S[m,k]=|\sum_nx[n+mH]w[n]e^{-j2\pi kn/N}|^2$（具体幅值/功率归一化依算子 options）。
- 任务变量与参数实例化：`frame/hop` 实例化窗长与步长，输出 spectrogram grid。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

##### `ricker`

- 学习入口：[数学物理原理](../../operators/wavelets/ricker/数学物理原理.md)、[Python 源码算法](../../operators/wavelets/ricker/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/wavelets/ricker/cusignal_cpp_ricker复现逻辑.md)。
- 当前调用采用的数学关系：Ricker 小波是高斯二阶导数形状，尺度参数控制宽度。
- 任务变量与参数实例化：作为 CWT 的 real wavelet callable。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

##### `cwt`

- 学习入口：[数学物理原理](../../operators/wavelets/cwt/数学物理原理.md)、[Python 源码算法](../../operators/wavelets/cwt/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/wavelets/cwt/cusignal_cpp_cwt复现逻辑.md)。
- 当前调用采用的数学关系：$W(a,b)=a^{-1/2}\sum_nx[n]\psi^*((n-b)/a)$。
- 任务变量与参数实例化：`widths→a`、平滑信号 `x`、输出 `cwt_grid→W`。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step4(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明、函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `StepEvidence run_step4(PipelineState& state);` 中的 `run_step4` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `run_step4`；调用 `run_step4`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `#include "step4.h"`。

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

该代码块按源码顺序包含 7 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step4`、`h`、`convolution`、`convolution_typed`、`cuda_utils`、`device_array`、`kernel_launch`、`demod`、`demod_typed`、`spectral_analysis`、`spectral_analysis_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step4.h"` 在预处理阶段引入 "step4.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "convolution/convolution_typed.h"` 在预处理阶段引入 "convolution/convolution_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/kernel_launch.h"` 在预处理阶段引入 "cuda_utils/kernel_launch.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "demod/demod_typed.h"` 在预处理阶段引入 "demod/demod_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "spectral_analysis/spectral_analysis_typed.h"` 在预处理阶段引入 "spectral_analysis/spectral_analysis_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "wavelets/wavelets_typed.h"` 在预处理阶段引入 "wavelets/wavelets_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

```cpp
#include <numeric>

namespace task2 {
namespace {
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`numeric`、`namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `namespace {` 打开命名空间 `匿名`；后续声明被放入该名称范围。 这是匿名命名空间，其中名称只在当前翻译单元可见。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 analytic_signal_kernel：声明 CUDA kernel 并建立线程入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `analytic_signal_kernel`；源码锚点 `__global__ void analytic_signal_kernel(`。

```cpp
__global__ void analytic_signal_kernel(
    const float* input, cusignal::DemodComplex<float>* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `input` 由 `const float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `output` 由 `cusignal::DemodComplex<float>*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `count` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `index` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`input`|当前函数或调用的参数名称，接收调用者绑定的输入 `input`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float* input, cusignal::DemodComplex<float>* output, int count)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `__global__ void analytic_signal_kernel( const float* input, cusignal::DemodComplex<float>* output, int count) {` 中的 `analytic_signal_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const float* input, cusignal::DemodComplex<float>* output, int count` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index` 建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x` 把块号和块内线程号映射为一维全局线程索引。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `analytic_signal_kernel`；调用 `analytic_signal_kernel`；写入/初始化 `index`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 使用 `blockIdx.x * blockDim.x + threadIdx.x` 是为了给每个一维输出元素分配唯一全局线程。这样同一 kernel 可适配不同 block 大小；随后必须用 count 做越界保护，因为 grid 往往向上取整。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.7 analytic_signal_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `analytic_signal_kernel`；源码锚点 `if (index >= count) return;`。

```cpp
    if (index >= count) return;
    const float previous = index == 0 ? input[0] : input[index - 1];
    const float next = index + 1 == count ? input[count - 1] : input[index + 1];
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 当前块中的 `if (...) return;` 必须先作为完整条件语句识别；比较运算符 `>=`/`<=` 中的 `=` 不是赋值。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `previous` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `next` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`previous`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `previous`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float previous = index == 0 ? input[0] : input[index - 1];`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`next`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `next`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float next = index + 1 == count ? input[count - 1] : input[index + 1];`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前 CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
- `const float previous = index == 0 ? input[0] : input[index - 1];` 是声明并初始化：`const float previous` 建立局部对象 `previous`，右侧完整表达式 `index == 0 ? input[0] : input[index - 1]` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。
- `const float next = index + 1 == count ? input[count - 1] : input[index + 1];` 是声明并初始化：`const float next` 建立局部对象 `next`，右侧完整表达式 `index + 1 == count ? input[count - 1] : input[index + 1]` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过检查 `index >= count`；写入/初始化 `previous`, `next`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `analytic_signal_kernel` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.8 analytic_signal_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `analytic_signal_kernel`；源码锚点 `output[index] = {input[index], 0.5F * (next - previous)};`。

```cpp
    output[index] = {input[index], 0.5F * (next - previous)};
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.5F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|

**执行过程**

- `output[index] = {input[index], 0.5F * (next - previous)};` 是赋值：先求得右侧完整表达式 `{input[index], 0.5F * (next - previous)}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过写入/初始化 `output[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `analytic_signal_kernel` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.9 spectral_envelope：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `spectral_envelope`；源码锚点 `std::vector<float> spectral_envelope(`。

```cpp

std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `flattened` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `frequencies` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `frames` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`flattened`|当前函数或调用的参数名称，接收调用者绑定的输入 `flattened`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& flattened, int frequencies, int frames)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`frequencies`|当前函数或调用的参数名称，接收调用者绑定的输入 `frequencies`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& flattened, int frequencies, int frames)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`frames`|当前函数或调用的参数名称，接收调用者绑定的输入 `frames`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& flattened, int frequencies, int frames)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> spectral_envelope( const std::vector<float>& flattened, int frequencies, int frames) {` 中的 `spectral_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& flattened, int frequencies, int frames` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);` 声明 `output`，其静态类型是 `std::vector<float>`，并用 `static_cast<std::size_t>(frames), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `spectral_envelope`；调用 `spectral_envelope`, `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `spectral_envelope` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.10 spectral_envelope：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `spectral_envelope`；源码锚点 `for (int frequency = 0; frequency < frequencies; ++frequency)`。

```cpp
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `frequency` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `frame` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`frequency`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frequency`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `for (int frequency = 0; frequency < frequencies; ++frequency)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`frame`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frame`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `for (int frame = 0; frame < frames; ++frame)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `for (int frequency = 0; frequency < frequencies; ++frequency) for (int frame = 0; frame < frames; ++frame) output[frame] = std::max(output[frame], flattened[static_cast<std::size_t>(frequency) * frames + frame]);` 是经典 `for`：先执行初始化 `int frequency = 0`；每轮前检查 `frequency < frequencies`，为假即退出；每轮循环体结束后执行 `++frequency`，再检查下一轮。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `std::max`；写入/初始化 `output[frame]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `spectral_envelope` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.11 spectral_envelope：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `spectral_envelope`；源码锚点 `return output;`。

```cpp
    return output;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `spectral_envelope` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.12 wavelet_envelope：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `wavelet_envelope`；源码锚点 `std::vector<float> wavelet_envelope(`。

```cpp

std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `flattened` 由 `const std::vector<double>&` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `widths` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `samples` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`flattened`|当前函数或调用的参数名称，接收调用者绑定的输入 `flattened`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<double>& flattened, int widths, int samples)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`widths`|当前函数或调用的参数名称，接收调用者绑定的输入 `widths`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<double>& flattened, int widths, int samples)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`samples`|任务信号总采样点数|记作 $N$|配置决定数组 shape、计算量和证据规模|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> wavelet_envelope( const std::vector<double>& flattened, int widths, int samples) {` 中的 `wavelet_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<double>& flattened, int widths, int samples` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);` 声明 `output`，其静态类型是 `std::vector<float>`，并用 `static_cast<std::size_t>(samples), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `wavelet_envelope`；调用 `wavelet_envelope`, `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.13 wavelet_envelope：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `wavelet_envelope`；源码锚点 `for (int width = 0; width < widths; ++width)`。

```cpp
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample],
                cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(
                    flattened[static_cast<std::size_t>(width) * samples + sample])));
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `width` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `sample` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`width`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `width`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `for (int width = 0; width < widths; ++width)`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`sample`|当前脉冲内快时间采样编号|记作 $n=i\bmod N_s$|由展平索引取余得到|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `for (int width = 0; width < widths; ++width) for (int sample = 0; sample < samples; ++sample) output[sample] = std::max(output[sample], cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs( flattened[static_cast<std::size_t>(width) * samples + sample])));` 是经典 `for`：先执行初始化 `int width = 0`；每轮前检查 `width < widths`，为假即退出；每轮循环体结束后执行 `++width`，再检查下一轮。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `std::max`, `cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host`, `std::abs`；写入/初始化 `output[sample]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.14 wavelet_envelope：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `wavelet_envelope`；源码锚点 `return output;`。

```cpp
    return output;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.15 make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `make_bundle`；源码锚点 `std::vector<float> make_bundle(`。

```cpp

std::vector<float> make_bundle(
    const std::vector<float>& fm,
    const std::vector<float>& correlation,
    const std::vector<float>& spectral,
    const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `fm` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `correlation` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `spectral` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `wavelet` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `count` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `fm_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`fm`|当前函数或调用的参数名称，接收调用者绑定的输入 `fm`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& fm,`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`spectral`|当前函数或调用的参数名称，接收调用者绑定的输入 `spectral`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& spectral,`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`wavelet`|当前函数或调用的参数名称，接收调用者绑定的输入 `wavelet`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& wavelet,`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。|入口加载，runner 和各 step 只读消费|
|`fm_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `fm_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto fm_n = normalize_abs(resample_linear(fm, count));`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<float> make_bundle( const std::vector<float>& fm, const std::vector<float>& correlation, const std::vector<float>& spectral, const std::vector<float>& wavelet, std::size_t count, const TaskConfig& config) {` 中的 `make_bundle` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& fm, const std::vector<float>& correlation, const std::vector<float>& spectral, const std::vector<float>& wavelet, std::size_t count, const TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto fm_n = normalize_abs(resample_linear(fm, count));` 是声明并初始化：`const auto fm_n` 建立局部对象 `fm_n`，右侧完整表达式 `normalize_abs(resample_linear(fm, count))` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `make_bundle`；调用 `make_bundle`, `normalize_abs`, `resample_linear`；写入/初始化 `fm_n`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.16 make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `make_bundle`；源码锚点 `const auto corr_n = normalize_abs(resample_linear(correlation, count));`。

```cpp
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `corr_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `spec_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `wave_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`corr_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `corr_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto corr_n = normalize_abs(resample_linear(correlation, count));`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`spec_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `spec_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto spec_n = normalize_abs(resample_linear(spectral, count));`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`wave_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `wave_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto wave_n = normalize_abs(resample_linear(wavelet, count));`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const auto corr_n = normalize_abs(resample_linear(correlation, count));` 是声明并初始化：`const auto corr_n` 建立局部对象 `corr_n`，右侧完整表达式 `normalize_abs(resample_linear(correlation, count))` 产生初值。
- `const auto spec_n = normalize_abs(resample_linear(spectral, count));` 是声明并初始化：`const auto spec_n` 建立局部对象 `spec_n`，右侧完整表达式 `normalize_abs(resample_linear(spectral, count))` 产生初值。
- `const auto wave_n = normalize_abs(resample_linear(wavelet, count));` 是声明并初始化：`const auto wave_n` 建立局部对象 `wave_n`，右侧完整表达式 `normalize_abs(resample_linear(wavelet, count))` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `normalize_abs`, `resample_linear`；写入/初始化 `corr_n`, `spec_n`, `wave_n`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.17 bundle：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `bundle`；源码锚点 `std::vector<float> bundle(count);`。

```cpp
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> bundle(count);` 声明 `bundle`，其静态类型是 `std::vector<float>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `bundle` 的函数，模板尖括号也不是比较或位移。
- `for (std::size_t index = 0; index < count; ++index) bundle[index] = config.fusion_weight_fm * fm_n[index] + config.fusion_weight_correlation * corr_n[index] + config.fusion_weight_spectral * spec_n[index] + config.fusion_weight_wavelet * wave_n[index];` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index < count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `bundle`；写入/初始化 `bundle[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.18 bundle：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `bundle`；源码锚点 `return bundle;`。

```cpp
    return bundle;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `bundle` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return bundle;` 是返回语句：先求值 `bundle`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `bundle`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `bundle` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.19 bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `bundle`；源码锚点 `}  // namespace`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `bundle` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `StepEvidence run_step4(PipelineState& state)`。

```cpp
StepEvidence run_step4(PipelineState& state)
{
    StepEvidence evidence;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `evidence` 由 `StepEvidence` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step4`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step4(PipelineState& state) {` 中的 `run_step4` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `run_step4`；调用 `run_step4`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step4` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.21 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `evidence.name = "step4";`。

```cpp
    evidence.name = "step4";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `total_begin` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`total_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.name = "step4";` 是赋值：先求得右侧完整表达式 `"step4"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `total_begin`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step4` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `const int count = static_cast<int>(state.smoothed.size());`。

```cpp
    const int count = static_cast<int>(state.smoothed.size());
    require_task2_upstream(count > 64, "Step4", "Step3.smoothed");
    require_task2_upstream(
        state.reference_for_correlation.size() == state.smoothed.size(),
        "Step4", "Step3.reference_for_correlation");
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `64` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`64`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|

**执行过程**

- `const int count = static_cast<int>(state.smoothed.size());` 是声明并初始化：`const int count` 建立局部对象 `count`，右侧完整表达式 `static_cast<int>(state.smoothed.size())` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `require_task2_upstream(count > 64, "Step4", "Step3.smoothed");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
- `require_task2_upstream( state.reference_for_correlation.size() == state.smoothed.size(), "Step4", "Step3.reference_for_correlation");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `size`, `require_task2_upstream`；写入/初始化 `count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.23 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `constexpr int frame = 64;`。

```cpp
    constexpr int frame = 64;
    constexpr int hop = 32;
    const int frames = 1 + (count - frame) / hop;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `frame` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `hop` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `frames` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `64` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `32` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`frame`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frame`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `constexpr int frame = 64;`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`hop`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `hop`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `constexpr int hop = 32;`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`frames`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frames`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const int frames = 1 + (count - frame) / hop;`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`64`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`32`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `constexpr int frame = 64;` 是声明并初始化：`constexpr int frame` 建立局部对象 `frame`，右侧完整表达式 `64` 产生初值。 `constexpr` 要求该初值可在编译期形成常量表达式。
- `constexpr int hop = 32;` 是声明并初始化：`constexpr int hop` 建立局部对象 `hop`，右侧完整表达式 `32` 产生初值。 `constexpr` 要求该初值可在编译期形成常量表达式。
- `const int frames = 1 + (count - frame) / hop;` 是声明并初始化：`const int frames` 建立局部对象 `frames`，右侧完整表达式 `1 + (count - frame) / hop` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过写入/初始化 `frame`, `hop`, `frames`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step4` 所在的 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.24 run_step4：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};`。

```cpp
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker =
        [](int points, int width) {
            return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
        };
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `ricker` 由 `const cusignal::CwtRealWaveletCallable` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `points` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `width` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `2.0F` 是数值字面量；F 指定 float，而非默认 double；
- `4.0F` 是数值字面量；F 指定 float，而非默认 double；
- `8.0F` 是数值字面量；F 指定 float，而非默认 double；
- `12.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`widths`|当前表达式读取或传递的工程名称 `widths`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`ricker`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `ricker`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const cusignal::CwtRealWaveletCallable ricker =`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`points`|当前函数或调用的参数名称，接收调用者绑定的输入 `points`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `[](int points, int width) {`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`width`|当前函数或调用的参数名称，接收调用者绑定的输入 `width`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `[](int points, int width) {`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`4.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`8.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`12.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};` 声明 `widths`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{2.0F, 4.0F, 8.0F, 12.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。 当前向量随后作为 CWT/Ricker 的尺度参数：`2.0F、4.0F、8.0F、12.0F` 是四个离散尺度，源码固定采用这组值但没有证明其唯一最优。
- `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) { return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width)); };` 声明 callable 对象 `ricker`，并用 lambda 表达式初始化；`[]` 表示不捕获外部变量，`(int points, int width)` 声明调用参数，花括号内 `return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));` 是每次调用时才执行的函数体。定义 lambda 本身不会立即运行该函数体。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过写入/初始化 `ricker`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.25 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `};`。

```cpp
    auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `cwt_workspace` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`cwt_workspace`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `cwt_workspace`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);` 是声明并初始化：`auto cwt_workspace` 建立局部对象 `cwt_workspace`，右侧完整表达式 `cusignal::prepare_cwt_workspace(count, widths, ricker)` 产生初值。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `cusignal::prepare_cwt_workspace`, `milliseconds`, `Clock::now`；写入/初始化 `cwt_workspace`, `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.26 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_signal` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_signal`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);` 是声明并初始化：`auto d_signal` 建立局部对象 `d_signal`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.smoothed)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`；写入/初始化 `begin`, `d_signal`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.27 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);`。

```cpp
    auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_reference` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_reference`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`d_analytic`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);` 是声明并初始化：`auto d_reference` 建立局部对象 `d_reference`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.reference_for_correlation)` 产生初值。
- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);` 声明 `d_analytic`，其静态类型是 `cusignal::DeviceArray<cusignal::DemodComplex<float>>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_analytic` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`, `d_analytic`；写入/初始化 `d_reference`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.28 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `evidence.operator_ms["analytic_glue"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["analytic_glue"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            analytic_signal_kernel, static_cast<std::size_t>(count),
            d_signal.data(), d_analytic.data(), count);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`analytic_glue`、`time_gpu`、`cusignal`、`cuda_utils`、`launch_1d_kernel`、`analytic_signal_kernel`、`std`、`size_t`、`count`、`d_signal`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.operator_ms["analytic_glue"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( analytic_signal_kernel, static_cast<std::size_t>(count), d_signal.data(), d_analytic.data(), count); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["analytic_glue"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::cuda_utils::launch_1d_kernel`, `data`；写入/初始化 `operator_ms["analytic_glue"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。

### 4.29 demod_workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `demod_workspace`；源码锚点 `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});`。

```cpp
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());
    evidence.operator_ms["fm_demod"] = time_gpu([&] {
        cusignal::fm_demod_device(d_analytic, d_fm, demod_workspace);
    });
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`demod_workspace`|当前函数或调用的参数名称，接收调用者绑定的输入 `demod_workspace`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`d_fm`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});` 声明 `demod_workspace`，其静态类型是 `cusignal::FmDemodWorkspace`，并用 `cusignal::FmDemodOptions{{count}, -1}` 直接初始化/调用该类型的构造函数；这不是调用名为 `demod_workspace` 的函数，模板尖括号也不是比较或位移。
- `cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());` 声明 `d_fm`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `demod_workspace.output_size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_fm` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["fm_demod"] = time_gpu([&] { cusignal::fm_demod_device(d_analytic, d_fm, demod_workspace); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["fm_demod"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `demod_workspace`, `d_fm`, `output_size`, `time_gpu`, `cusignal::fm_demod_device`；写入/初始化 `operator_ms["fm_demod"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.30 d_correlation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_correlation`；源码锚点 `cusignal::DeviceArray<float> d_correlation(count);`。

```cpp
    cusignal::DeviceArray<float> d_correlation(count);
    evidence.operator_ms["correlate"] = time_gpu([&] {
        cusignal::correlate_device(d_signal, d_reference, d_correlation, "same");
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_correlation`、`count`、`evidence`、`operator_ms`、`correlate`、`time_gpu`、`correlate_device`、`d_signal`、`d_reference`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_correlation(count);` 声明 `d_correlation`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_correlation` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["correlate"] = time_gpu([&] { cusignal::correlate_device(d_signal, d_reference, d_correlation, "same"); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["correlate"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `d_correlation`, `time_gpu`, `cusignal::correlate_device`；写入/初始化 `operator_ms["correlate"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.31 d_spectral：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_spectral`；源码锚点 `cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);`。

```cpp
    cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);
    evidence.operator_ms["spectrogram"] = time_gpu([&] {
        cusignal::spectrogram_device(d_signal, frame, hop, d_spectral);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_spectral`、`std`、`size_t`、`frame`、`frames`、`evidence`、`operator_ms`、`spectrogram`、`time_gpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_spectral`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);` 声明 `d_spectral`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `static_cast<std::size_t>(frame) * frames` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_spectral` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["spectrogram"] = time_gpu([&] { cusignal::spectrogram_device(d_signal, frame, hop, d_spectral); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["spectrogram"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `d_spectral`, `time_gpu`, `cusignal::spectrogram_device`；写入/初始化 `operator_ms["spectrogram"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.32 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());`。

```cpp
    cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());
    evidence.operator_ms["cwt_ricker"] = time_gpu([&] {
        cusignal::cwt_device(d_signal, cwt_workspace, d_cwt);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`double`、`d_cwt`、`cwt_workspace`、`output_size`、`evidence`、`operator_ms`、`cwt_ricker`、`time_gpu`、`cwt_device`、`d_signal`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_cwt`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());` 声明 `d_cwt`，其静态类型是 `cusignal::DeviceArray<double>`，并用 `cwt_workspace.output_size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_cwt` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["cwt_ricker"] = time_gpu([&] { cusignal::cwt_device(d_signal, cwt_workspace, d_cwt); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["cwt_ricker"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|公共基础设施|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `d_cwt`, `output_size`, `time_gpu`, `cusignal::cwt_device`；写入/初始化 `operator_ms["cwt_ricker"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.33 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.34 d_cwt：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `evidence.compute_ms = std::accumulate(`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `compute_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.35 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    state.fm_feature = d_fm.to_host();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`fm_feature`、`d_fm`、`to_host`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.fm_feature = d_fm.to_host();` 是赋值：先求得右侧完整表达式 `d_fm.to_host()`，再把结果写入左侧可修改对象 `state.fm_feature`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`；写入/初始化 `begin`, `fm_feature`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.36 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `state.correlation_feature = d_correlation.to_host();`。

```cpp
    state.correlation_feature = d_correlation.to_host();
    const auto spectral_flat = d_spectral.to_host();
    const auto cwt_flat = d_cwt.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `spectral_flat` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `cwt_flat` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`spectral_flat`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `spectral_flat`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto spectral_flat = d_spectral.to_host();`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`cwt_flat`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `cwt_flat`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto cwt_flat = d_cwt.to_host();`；值在 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.correlation_feature = d_correlation.to_host();` 是赋值：先求得右侧完整表达式 `d_correlation.to_host()`，再把结果写入左侧可修改对象 `state.correlation_feature`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto spectral_flat = d_spectral.to_host();` 是声明并初始化：`const auto spectral_flat` 建立局部对象 `spectral_flat`，右侧完整表达式 `d_spectral.to_host()` 产生初值。
- `const auto cwt_flat = d_cwt.to_host();` 是声明并初始化：`const auto cwt_flat` 建立局部对象 `cwt_flat`，右侧完整表达式 `d_cwt.to_host()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `to_host`；写入/初始化 `correlation_feature`, `spectral_flat`, `cwt_flat`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.37 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `evidence.d2h_ms = milliseconds(begin, Clock::now());`。

```cpp
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`、`state`、`spectral_feature`、`spectral_envelope`、`spectral_flat`、`frame`、`frames`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);` 是赋值：先求得右侧完整表达式 `spectral_envelope(spectral_flat, frame, frames)`，再把结果写入左侧可修改对象 `state.spectral_feature`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`, `spectral_envelope`；写入/初始化 `d2h_ms`, `begin`, `spectral_feature`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.38 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `state.wavelet_feature = wavelet_envelope(`。

```cpp
    state.wavelet_feature = wavelet_envelope(
        cwt_flat, static_cast<int>(widths.size()), count);
    state.feature_bundle = make_bundle(
        state.fm_feature, state.correlation_feature,
        state.spectral_feature, state.wavelet_feature, count, state.config);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`wavelet_feature`、`wavelet_envelope`、`cwt_flat`、`int`、`widths`、`size`、`count`、`feature_bundle`、`make_bundle`、`fm_feature`、`correlation_feature`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.wavelet_feature = wavelet_envelope( cwt_flat, static_cast<int>(widths.size()), count);` 是赋值：先求得右侧完整表达式 `wavelet_envelope( cwt_flat, static_cast<int>(widths.size()), count)`，再把结果写入左侧可修改对象 `state.wavelet_feature`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `state.feature_bundle = make_bundle( state.fm_feature, state.correlation_feature, state.spectral_feature, state.wavelet_feature, count, state.config);` 是赋值：先求得右侧完整表达式 `make_bundle( state.fm_feature, state.correlation_feature, state.spectral_feature, state.wavelet_feature, count, state.config)`，再把结果写入左侧可修改对象 `state.feature_bundle`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|输入准备/数据契约|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `wavelet_envelope`, `size`, `make_bundle`；写入/初始化 `wavelet_feature`, `feature_bundle`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.39 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `evidence.post_ms = milliseconds(begin, Clock::now());`。

```cpp

    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`post_ms`、`milliseconds`、`begin`、`Clock`、`now`、`total_ms`、`total_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.post_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.post_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `post_ms`, `total_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.40 d_cwt：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `if (visualization_capture_enabled()) {`。

```cpp
    if (visualization_capture_enabled()) {
        state.spectrogram_grid = spectral_flat;
        state.cwt_grid = cwt_flat;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `visualization_capture_enabled`、`state`、`spectrogram_grid`、`spectral_flat`、`cwt_grid`、`cwt_flat`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `if (visualization_capture_enabled()) {` 是 `if` 条件语句：先把完整条件 `visualization_capture_enabled()` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `state.spectrogram_grid = spectral_flat;` 是赋值：先求得右侧完整表达式 `spectral_flat`，再把结果写入左侧可修改对象 `state.spectrogram_grid`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.cwt_grid = cwt_flat;` 是赋值：先求得右侧完整表达式 `cwt_flat`，再把结果写入左侧可修改对象 `state.cwt_grid`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过检查 `visualization_capture_enabled()`；调用 `visualization_capture_enabled`；写入/初始化 `spectrogram_grid`, `cwt_grid`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于可视化数据链：它读取/导出数值结果，建立坐标、单位和图形对象；图片是解释性派生物，不能替代正式数值门禁。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.41 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `state.spectrogram_frequency_bins = frame;`。

```cpp
        state.spectrogram_frequency_bins = frame;
        state.spectrogram_frames = frames;
        state.cwt_width_count = static_cast<int>(widths.size());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`spectrogram_frequency_bins`、`frame`、`spectrogram_frames`、`frames`、`cwt_width_count`、`int`、`widths`、`size`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.spectrogram_frequency_bins = frame;` 是赋值：先求得右侧完整表达式 `frame`，再把结果写入左侧可修改对象 `state.spectrogram_frequency_bins`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.spectrogram_frames = frames;` 是赋值：先求得右侧完整表达式 `frames`，再把结果写入左侧可修改对象 `state.spectrogram_frames`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.cwt_width_count = static_cast<int>(widths.size());` 是赋值：先求得右侧完整表达式 `static_cast<int>(widths.size())`，再把结果写入左侧可修改对象 `state.cwt_width_count`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `size`；写入/初始化 `spectrogram_frequency_bins`, `spectrogram_frames`, `cwt_width_count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.42 d_cwt：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `}`。

```cpp
    }
    return evidence;
}
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：作用域边界、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `evidence` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.43 d_cwt：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step4/step4.cu`；符号 `d_cwt`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块提取调制、相关、频谱或时频特征；产生的特征数组随后被融合并交给峰值定位。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.44 analytic_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `analytic_cpu`；源码锚点 `std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)`。

```cpp
std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)
{
    std::vector<cusignal::DemodComplex<float>> output(input.size());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `input` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`input`|当前函数或调用的参数名称，接收调用者绑定的输入 `input`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input) {` 中的 `analytic_cpu` 是函数定义；名称前的 `std::vector<cusignal::DemodComplex<float>>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& input` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<cusignal::DemodComplex<float>> output(input.size());` 声明 `output`，其静态类型是 `std::vector<cusignal::DemodComplex<float>>`，并用 `input.size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `analytic_cpu`；调用 `analytic_cpu`, `output`, `size`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.45 analytic_cpu：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `analytic_cpu`；源码锚点 `for (std::size_t index = 0; index < input.size(); ++index) {`。

```cpp
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float previous = index == 0 ? input.front() : input[index - 1];
        const float next = index + 1 == input.size() ? input.back() : input[index + 1];
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `previous` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `next` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`previous`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `previous`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float previous = index == 0 ? input.front() : input[index - 1];`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`next`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `next`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float next = index + 1 == input.size() ? input.back() : input[index + 1];`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `for (std::size_t index = 0; index < input.size(); ++index) {` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index < input.size()`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `const float previous = index == 0 ? input.front() : input[index - 1];` 是声明并初始化：`const float previous` 建立局部对象 `previous`，右侧完整表达式 `index == 0 ? input.front() : input[index - 1]` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。
- `const float next = index + 1 == input.size() ? input.back() : input[index + 1];` 是声明并初始化：`const float next` 建立局部对象 `next`，右侧完整表达式 `index + 1 == input.size() ? input.back() : input[index + 1]` 产生初值。 `条件 ? 真分支 : 假分支` 是条件运算符：只求值两个候选分支中的一个。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `size`, `front`, `back`；写入/初始化 `previous`, `next`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.46 analytic_cpu：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `analytic_cpu`；源码锚点 `output[index] = {input[index], 0.5F * (next - previous)};`。

```cpp
        output[index] = {input[index], 0.5F * (next - previous)};
    }
    return output;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式；
- `0.5F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|

**执行过程**

- `output[index] = {input[index], 0.5F * (next - previous)};` 是赋值：先求得右侧完整表达式 `{input[index], 0.5F * (next - previous)}`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过写入/初始化 `output[index]`；返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.47 analytic_cpu：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `analytic_cpu`；源码锚点 `}`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.48 spectral_envelope：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `spectral_envelope`；源码锚点 `std::vector<float> spectral_envelope(`。

```cpp

std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `flattened` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `frequencies` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `frames` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`flattened`|当前函数或调用的参数名称，接收调用者绑定的输入 `flattened`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& flattened, int frequencies, int frames)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`frequencies`|当前函数或调用的参数名称，接收调用者绑定的输入 `frequencies`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& flattened, int frequencies, int frames)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`frames`|当前函数或调用的参数名称，接收调用者绑定的输入 `frames`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& flattened, int frequencies, int frames)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> spectral_envelope( const std::vector<float>& flattened, int frequencies, int frames) {` 中的 `spectral_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& flattened, int frequencies, int frames` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);` 声明 `output`，其静态类型是 `std::vector<float>`，并用 `static_cast<std::size_t>(frames), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `spectral_envelope`；调用 `spectral_envelope`, `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.49 spectral_envelope：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `spectral_envelope`；源码锚点 `for (int frequency = 0; frequency < frequencies; ++frequency)`。

```cpp
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `frequency` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `frame` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`frequency`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frequency`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `for (int frequency = 0; frequency < frequencies; ++frequency)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`frame`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frame`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `for (int frame = 0; frame < frames; ++frame)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `for (int frequency = 0; frequency < frequencies; ++frequency) for (int frame = 0; frame < frames; ++frame) output[frame] = std::max(output[frame], flattened[static_cast<std::size_t>(frequency) * frames + frame]);` 是经典 `for`：先执行初始化 `int frequency = 0`；每轮前检查 `frequency < frequencies`，为假即退出；每轮循环体结束后执行 `++frequency`，再检查下一轮。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `std::max`；写入/初始化 `output[frame]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.50 spectral_envelope：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `spectral_envelope`；源码锚点 `return output;`。

```cpp
    return output;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.51 wavelet_envelope：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `wavelet_envelope`；源码锚点 `std::vector<float> wavelet_envelope(`。

```cpp

std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `flattened` 由 `const std::vector<double>&` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `widths` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `samples` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`flattened`|当前函数或调用的参数名称，接收调用者绑定的输入 `flattened`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<double>& flattened, int widths, int samples)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`widths`|当前函数或调用的参数名称，接收调用者绑定的输入 `widths`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<double>& flattened, int widths, int samples)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`samples`|任务信号总采样点数|记作 $N$|配置决定数组 shape、计算量和证据规模|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> wavelet_envelope( const std::vector<double>& flattened, int widths, int samples) {` 中的 `wavelet_envelope` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<double>& flattened, int widths, int samples` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);` 声明 `output`，其静态类型是 `std::vector<float>`，并用 `static_cast<std::size_t>(samples), 0.0F` 直接初始化/调用该类型的构造函数；这不是调用名为 `output` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `wavelet_envelope`；调用 `wavelet_envelope`, `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.52 wavelet_envelope：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `wavelet_envelope`；源码锚点 `for (int width = 0; width < widths; ++width)`。

```cpp
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample], static_cast<float>(std::abs(
                flattened[static_cast<std::size_t>(width) * samples + sample])));
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `width` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `sample` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`width`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `width`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `for (int width = 0; width < widths; ++width)`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`sample`|当前脉冲内快时间采样编号|记作 $n=i\bmod N_s$|由展平索引取余得到|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `for (int width = 0; width < widths; ++width) for (int sample = 0; sample < samples; ++sample) output[sample] = std::max(output[sample], static_cast<float>(std::abs( flattened[static_cast<std::size_t>(width) * samples + sample])));` 是经典 `for`：先执行初始化 `int width = 0`；每轮前检查 `width < widths`，为假即退出；每轮循环体结束后执行 `++width`，再检查下一轮。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `std::max`, `std::abs`；写入/初始化 `output[sample]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.53 wavelet_envelope：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `wavelet_envelope`；源码锚点 `return output;`。

```cpp
    return output;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return output;` 是返回语句：先求值 `output`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `output`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.54 make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `make_bundle`；源码锚点 `std::vector<float> make_bundle(`。

```cpp

std::vector<float> make_bundle(
    const std::vector<float>& fm, const std::vector<float>& correlation,
    const std::vector<float>& spectral, const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `fm` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `correlation` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `spectral` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `wavelet` 由 `const std::vector<float>&` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `count` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `config` 由 `const TaskConfig&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `fm_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`fm`|当前函数或调用的参数名称，接收调用者绑定的输入 `fm`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& fm, const std::vector<float>& correlation,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`spectral`|当前函数或调用的参数名称，接收调用者绑定的输入 `spectral`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& spectral, const std::vector<float>& wavelet,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`wavelet`|当前函数或调用的参数名称，接收调用者绑定的输入 `wavelet`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float>& spectral, const std::vector<float>& wavelet,`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。|入口加载，runner 和各 step 只读消费|
|`fm_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `fm_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto fm_n = normalize_abs(resample_linear(fm, count));`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `std::vector<float> make_bundle( const std::vector<float>& fm, const std::vector<float>& correlation, const std::vector<float>& spectral, const std::vector<float>& wavelet, std::size_t count, const TaskConfig& config) {` 中的 `make_bundle` 是函数定义；名称前的 `std::vector<float>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const std::vector<float>& fm, const std::vector<float>& correlation, const std::vector<float>& spectral, const std::vector<float>& wavelet, std::size_t count, const TaskConfig& config` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto fm_n = normalize_abs(resample_linear(fm, count));` 是声明并初始化：`const auto fm_n` 建立局部对象 `fm_n`，右侧完整表达式 `normalize_abs(resample_linear(fm, count))` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `make_bundle`；调用 `make_bundle`, `normalize_abs`, `resample_linear`；写入/初始化 `fm_n`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.55 make_bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `make_bundle`；源码锚点 `const auto corr_n = normalize_abs(resample_linear(correlation, count));`。

```cpp
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `corr_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `spec_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `wave_n` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`corr_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `corr_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto corr_n = normalize_abs(resample_linear(correlation, count));`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`spec_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `spec_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto spec_n = normalize_abs(resample_linear(spectral, count));`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`wave_n`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `wave_n`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const auto wave_n = normalize_abs(resample_linear(wavelet, count));`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const auto corr_n = normalize_abs(resample_linear(correlation, count));` 是声明并初始化：`const auto corr_n` 建立局部对象 `corr_n`，右侧完整表达式 `normalize_abs(resample_linear(correlation, count))` 产生初值。
- `const auto spec_n = normalize_abs(resample_linear(spectral, count));` 是声明并初始化：`const auto spec_n` 建立局部对象 `spec_n`，右侧完整表达式 `normalize_abs(resample_linear(spectral, count))` 产生初值。
- `const auto wave_n = normalize_abs(resample_linear(wavelet, count));` 是声明并初始化：`const auto wave_n` 建立局部对象 `wave_n`，右侧完整表达式 `normalize_abs(resample_linear(wavelet, count))` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `normalize_abs`, `resample_linear`；写入/初始化 `corr_n`, `spec_n`, `wave_n`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.56 bundle：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `bundle`；源码锚点 `std::vector<float> bundle(count);`。

```cpp
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`config`|外部配置对象|`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> bundle(count);` 声明 `bundle`，其静态类型是 `std::vector<float>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `bundle` 的函数，模板尖括号也不是比较或位移。
- `for (std::size_t index = 0; index < count; ++index) bundle[index] = config.fusion_weight_fm * fm_n[index] + config.fusion_weight_correlation * corr_n[index] + config.fusion_weight_spectral * spec_n[index] + config.fusion_weight_wavelet * wave_n[index];` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index < count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `bundle`；写入/初始化 `bundle[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.57 bundle：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `bundle`；源码锚点 `return bundle;`。

```cpp
    return bundle;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `bundle` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `return bundle;` 是返回语句：先求值 `bundle`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `bundle`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.58 bundle：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `bundle`；源码锚点 `}  // namespace`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.59 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `Step4ValidationData collect_step4_validation_impl(`。

```cpp
Step4ValidationData collect_step4_validation_impl(
    const PipelineState& state, std::vector<float>* precomputed_correlation)
{
    Step4ValidationData data;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `precomputed_correlation` 由 `std::vector<float>*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `data` 由 `Step4ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step4_validation_impl`|测试、收集或契约检查 helper|无独立数学变量；内部指标分别映射到误差/条件公式|消费运行结果并形成证据或失败路径|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`precomputed_correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step4ValidationData collect_step4_validation_impl( const PipelineState& state, std::vector<float>* precomputed_correlation) {` 中的 `collect_step4_validation_impl` 是函数定义；名称前的 `Step4ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state, std::vector<float>* precomputed_correlation` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step4ValidationData data;` 是对象声明：类型 `Step4ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `collect_step4_validation_impl`；调用 `collect_step4_validation_impl`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.60 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `const int count = static_cast<int>(state.smoothed.size());`。

```cpp
    const int count = static_cast<int>(state.smoothed.size());
    constexpr int frame = 64;
    constexpr int hop = 32;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `frame` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `hop` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `64` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `32` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`frame`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frame`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `constexpr int frame = 64;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`hop`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `hop`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `constexpr int hop = 32;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`64`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`32`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|

**执行过程**

- `const int count = static_cast<int>(state.smoothed.size());` 是声明并初始化：`const int count` 建立局部对象 `count`，右侧完整表达式 `static_cast<int>(state.smoothed.size())` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `constexpr int frame = 64;` 是声明并初始化：`constexpr int frame` 建立局部对象 `frame`，右侧完整表达式 `64` 产生初值。 `constexpr` 要求该初值可在编译期形成常量表达式。
- `constexpr int hop = 32;` 是声明并初始化：`constexpr int hop` 建立局部对象 `hop`，右侧完整表达式 `32` 产生初值。 `constexpr` 要求该初值可在编译期形成常量表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `size`；写入/初始化 `count`, `frame`, `hop`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.61 collect_step4_validation_impl：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `const int frames = 1 + (count - frame) / hop;`。

```cpp
    const int frames = 1 + (count - frame) / hop;
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {
        return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
    };
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `frames` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `ricker` 由 `const cusignal::CwtRealWaveletCallable` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `points` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `width` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2.0F` 是数值字面量；F 指定 float，而非默认 double；
- `4.0F` 是数值字面量；F 指定 float，而非默认 double；
- `8.0F` 是数值字面量；F 指定 float，而非默认 double；
- `12.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`frames`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `frames`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const int frames = 1 + (count - frame) / hop;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`widths`|当前表达式读取或传递的工程名称 `widths`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`ricker`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `ricker`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`points`|当前函数或调用的参数名称，接收调用者绑定的输入 `points`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`width`|当前函数或调用的参数名称，接收调用者绑定的输入 `width`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`2.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`4.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`8.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`12.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `const int frames = 1 + (count - frame) / hop;` 是声明并初始化：`const int frames` 建立局部对象 `frames`，右侧完整表达式 `1 + (count - frame) / hop` 产生初值。 这里的除法操作数为整数类型，商按 C++ 整数除法规则向零截断；在本任务非负索引条件下等同于向下取整。
- `const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};` 声明 `widths`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{2.0F, 4.0F, 8.0F, 12.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。 当前向量随后作为 CWT/Ricker 的尺度参数：`2.0F、4.0F、8.0F、12.0F` 是四个离散尺度，源码固定采用这组值但没有证明其唯一最优。
- `const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) { return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width)); };` 声明 callable 对象 `ricker`，并用 lambda 表达式初始化；`[]` 表示不捕获外部变量，`(int points, int width)` 声明调用参数，花括号内 `return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));` 是每次调用时才执行的函数体。定义 lambda 本身不会立即运行该函数体。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过写入/初始化 `frames`, `ricker`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 整数除法会丢弃小数部分；索引换算成立还依赖分母为正和输入索引非负的契约。

### 4.62 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `};`。

```cpp
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    data.fm_cpu = cusignal::fm_demod_typed_cpu(
        analytic_cpu(state.smoothed), demod_workspace);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`demod_workspace`|当前表达式读取或传递的工程名称 `demod_workspace`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});` 声明 `demod_workspace`，其静态类型是 `cusignal::FmDemodWorkspace`，并用 `cusignal::FmDemodOptions{{count}, -1}` 直接初始化/调用该类型的构造函数；这不是调用名为 `demod_workspace` 的函数，模板尖括号也不是比较或位移。
- `data.fm_cpu = cusignal::fm_demod_typed_cpu( analytic_cpu(state.smoothed), demod_workspace);` 是赋值：先求得右侧完整表达式 `cusignal::fm_demod_typed_cpu( analytic_cpu(state.smoothed), demod_workspace)`，再把结果写入左侧可修改对象 `data.fm_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `demod_workspace`, `cusignal::fm_demod_typed_cpu`, `analytic_cpu`；写入/初始化 `fm_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.63 collect_step4_validation_impl：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `if (precomputed_correlation) {`。

```cpp
    if (precomputed_correlation) {
        if (precomputed_correlation->size() != state.smoothed.size())
            throw std::invalid_argument(
                "precomputed Task2 Step4 correlation size mismatch");
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `precomputed_correlation`、`size`、`state`、`smoothed`、`std`、`invalid_argument`、`precomputed`、`Task2`、`Step4`、`correlation`、`mismatch`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`Step4`|当前函数或调用的参数名称，接收调用者绑定的输入 `Step4`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `"precomputed Task2 Step4 correlation size mismatch");`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`correlation`|输入与参考的离散相关序列|记作 $r_{xy}[k]=\sum_n x[n]y^*[n-k]$|用于定位时延特征|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `if (precomputed_correlation) {` 是 `if` 条件语句：先把完整条件 `precomputed_correlation` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `if (precomputed_correlation->size() != state.smoothed.size()) throw std::invalid_argument( "precomputed Task2 Step4 correlation size mismatch");` 是 `if` 条件语句：先把完整条件 `precomputed_correlation->size() != state.smoothed.size()` 求值并转换为布尔值。 条件为真时构造并抛出 `std::invalid_argument( "precomputed Task2 Step4 correlation size mismatch")`，正常顺序执行立即中断并进入异常传播；条件为假时跳过抛出语句。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过检查 `precomputed_correlation`, `precomputed_correlation->size() != state.smoothed.size()`；调用 `size`, `std::invalid_argument`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.64 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `data.correlation_cpu = std::move(*precomputed_correlation);`。

```cpp
        data.correlation_cpu = std::move(*precomputed_correlation);
    } else {
        data.correlation_cpu = cusignal::correlate_typed_cpu(
            state.smoothed, state.reference_for_correlation, "same",
            cusignal::CorrelateMethod::direct);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`correlation_cpu`、`std`、`move`、`precomputed_correlation`、`cusignal`、`correlate_typed_cpu`、`state`、`smoothed`、`reference_for_correlation`、`same`、`CorrelateMethod`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.correlation_cpu = std::move(*precomputed_correlation);` 是赋值：先求得右侧完整表达式 `std::move(*precomputed_correlation)`，再把结果写入左侧可修改对象 `data.correlation_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `} else {` 先用 `}` 关闭前一个 `if` 分支，再由 `else {` 打开互斥的假分支；只有前一个条件为假时才执行该分支。
- `data.correlation_cpu = cusignal::correlate_typed_cpu( state.smoothed, state.reference_for_correlation, "same", cusignal::CorrelateMethod::direct);` 是赋值：先求得右侧完整表达式 `cusignal::correlate_typed_cpu( state.smoothed, state.reference_for_correlation, "same", cusignal::CorrelateMethod::direct)`，再把结果写入左侧可修改对象 `data.correlation_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `std::move`, `cusignal::correlate_typed_cpu`；写入/初始化 `correlation_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.65 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `}`。

```cpp
    }
    data.spectral_cpu = spectral_envelope(
        cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：作用域边界、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`spectral_cpu`、`spectral_envelope`、`cusignal`、`spectrogram_typed_cpu`、`state`、`smoothed`、`frame`、`hop`、`frames`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `data.spectral_cpu = spectral_envelope( cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames);` 是赋值：先求得右侧完整表达式 `spectral_envelope( cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames)`，再把结果写入左侧可修改对象 `data.spectral_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `spectral_envelope`, `cusignal::spectrogram_typed_cpu`；写入/初始化 `spectral_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.66 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `data.wavelet_cpu = wavelet_envelope(`。

```cpp
    data.wavelet_cpu = wavelet_envelope(
        cusignal::cwt_typed_cpu(state.smoothed, widths, ricker),
        static_cast<int>(widths.size()), count);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`wavelet_cpu`、`wavelet_envelope`、`cusignal`、`cwt_typed_cpu`、`state`、`smoothed`、`widths`、`ricker`、`int`、`size`、`count`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.wavelet_cpu = wavelet_envelope( cusignal::cwt_typed_cpu(state.smoothed, widths, ricker), static_cast<int>(widths.size()), count);` 是赋值：先求得右侧完整表达式 `wavelet_envelope( cusignal::cwt_typed_cpu(state.smoothed, widths, ricker), static_cast<int>(widths.size()), count)`，再把结果写入左侧可修改对象 `data.wavelet_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `wavelet_envelope`, `cusignal::cwt_typed_cpu`, `size`；写入/初始化 `wavelet_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.67 collect_step4_validation_impl：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `data.bundle_cpu = make_bundle(`。

```cpp
    data.bundle_cpu = make_bundle(
        data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu,
        count, state.config);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`bundle_cpu`、`make_bundle`、`fm_cpu`、`correlation_cpu`、`spectral_cpu`、`wavelet_cpu`、`count`、`state`、`config`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化谱帧/步长、CWT scales 以及 FM/相关/谱/小波四路融合权重。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.bundle_cpu = make_bundle( data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu, count, state.config);` 是赋值：先求得右侧完整表达式 `make_bundle( data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu, count, state.config)`，再把结果写入左侧可修改对象 `data.bundle_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过调用 `make_bundle`；写入/初始化 `bundle_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.68 collect_step4_validation_impl：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation_impl`；源码锚点 `return data;`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.69 collect_step4_validation：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation`；源码锚点 `Step4ValidationData collect_step4_validation(const PipelineState& state)`。

```cpp

Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    return collect_step4_validation_impl(state, nullptr);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step4_validation`|测试、收集或契约检查 helper|无独立数学变量；内部指标分别映射到误差/条件公式|消费运行结果并形成证据或失败路径|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step4ValidationData collect_step4_validation(const PipelineState& state) {` 中的 `collect_step4_validation` 是函数定义；名称前的 `Step4ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `return collect_step4_validation_impl(state, nullptr);` 是返回语句：先求值 `collect_step4_validation_impl(state, nullptr)`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块通过定义/声明 `collect_step4_validation`, `collect_step4_validation_impl`；调用 `collect_step4_validation`, `collect_step4_validation_impl`；返回 `collect_step4_validation_impl(state, nullptr)`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.70 collect_step4_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step4_validation.cpp`；符号 `collect_step4_validation`；源码锚点 `}`。

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
|赛题条款|T2-R4：多域特征提取：解调、相关、谱分析和小波变换四类大项各至少实现一种|
|业务角色|测试证据|
|业务输入|T2-R3 的平滑信号和参考波形/窗口/尺度配置|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|FM、相关、谱和小波特征，以及加权融合 `feature_bundle`/`fused`|
|下一消费者|融合特征交给 T2-R5 峰值定位|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
