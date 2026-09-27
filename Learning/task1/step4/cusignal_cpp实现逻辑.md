# Task1 Step4 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task1 Step4 的距离—多普勒功率 kernel、CFAR 训练窗计数、`cfar_alpha`/`ca_cfar` 正式调用、阈值与检测输出，以及数学等价的 CPU reference。

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
|`ZKX/Task1/task1_gpu_cpu/step4/step4.h`|`run_step4` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/step4/step4.cu`|`power_kernel`、`run_step4`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`|`generate_step4_cpu_reference`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task1/task1_gpu_cpu/step4/step4.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
//  `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task1 { StepEvidence run_step4(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence
//  run_step4(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。
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
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step4(PipelineState& state); }
```

### 3.2 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step4.h"` 在预处理阶段引入 "step4.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
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
#include "step4.h"

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

// <学习注释：语义块：power_kernel：声明 CUDA kernel 并建立线程入口。
//
//  【执行方法】
//  `__global__ void power_kernel( const ComplexFloat* input, float* output, int count) {` 中的 `power_kernel`
//  是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const ComplexFloat* input, float* output,
//  int count` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾 `{`
//  打开函数体；这里定义函数但不会在定义时自动执行。
//  `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index`
//  建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。
//  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x`
//  把块号和块内线程号映射为一维全局线程索引。
//
//  【变量—数学符号—数据流】
//  input：当前函数或调用的参数名称，接收调用者绑定的输入 input；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const ComplexFloat* input, float*
//  output, int count)；值在 ZKX/Task1/task1_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
//  output：当前函数或步骤的输出容器；对应当前算法公式的左端结果，具体符号由所在 step 决定；由本块写入，返回或交给下一步骤
//  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
//  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
//
//  【数学/物理公式对照】
//  `input[index]=z=a+jb`，输出 `a*a+b*b=|z|²`；dtype 从复 `float` 变为实 `float`，shape/展平索引不变。每个有效线程读一个复数并写一个功率值。
//
//  【为什么这样设计】
//  CFAR 比较非负功率；模平方避免平方根并保持排序。每线程一单元没有写冲突。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
__global__ void power_kernel(
    const ComplexFloat* input, float* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // <学习注释：语义块：power_kernel：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前
    //  CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
    //  `const auto value = input[index];` 是声明并初始化：`const auto value` 建立局部对象 `value`，右侧完整表达式 `input[index]` 产生初值。
    //  `output[index] = value.re * value.re + value.im * value.im;` 是赋值：先求得右侧完整表达式 `value.re * value.re +
    //  value.im * value.im`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //
    //  【数学/物理公式对照】
    //  `input[index]=z=a+jb`，输出 `a*a+b*b=|z|²`；dtype 从复 `float` 变为实 `float`，shape/展平索引不变。每个有效线程读一个复数并写一个功率值。
    //
    //  【为什么这样设计】
    //  CFAR 比较非负功率；模平方避免平方根并保持排序。每线程一单元没有写冲突。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    if (index >= count) return;
    const auto value = input[index];
    output[index] = value.re * value.re + value.im * value.im;
// <学习注释：语义块：power_kernel：完成一个连续的数据处理动作。
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

// <学习注释：语义块：power_kernel：完成一个连续的数据处理动作。
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
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  run_step4：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
StepEvidence run_step4(PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.name = "step4";` 是赋值：先求得右侧完整表达式 `"step4"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
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
    evidence.name = "step4";
    const auto formal_begin = Clock::now();
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //  `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象
    //  `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。
    //  `const bool has_resident_input = state.range_doppler_device.size() == static_cast<std::size_t>(count);`
    //  是声明并初始化：`const bool has_resident_input` 建立局部对象 `has_resident_input`，右侧完整表达式
    //  `state.range_doppler_device.size() == static_cast<std::size_t>(count)` 产生初值。 `static_cast<T>(...)`
    //  明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  has_resident_input：当前块的赋值左端，保存右侧表达式产生的中间值或结果 has_resident_input；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
    //  bool has_resident_input =；值在 ZKX/Task1/task1_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    auto begin = Clock::now();
    const int count = config.num_pulses * config.samples_per_pulse;
    const bool has_resident_input =
        state.range_doppler_device.size() == static_cast<std::size_t>(count);
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `require_task1_upstream(has_resident_input || state.range_doppler.size() ==
    //  static_cast<std::size_t>(count), "Step4", "Step3.range_doppler");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续
    //  step/字段名称报告缺失或非法输入。
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
    require_task1_upstream(has_resident_input ||
            state.range_doppler.size() == static_cast<std::size_t>(count),
            "Step4", "Step3.range_doppler");
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::CaCfarOptions options;` 是对象声明：类型 `cusignal::CaCfarOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};` 是赋值：先求得右侧完整表达式
    //  `{config.cfar_guard_doppler, config.cfar_guard_range}`，再把结果写入左侧可修改对象 `options.guard_cells`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};` 是赋值：先求得右侧完整表达式
    //  `{config.cfar_reference_doppler, config.cfar_reference_range}`，再把结果写入左侧可修改对象 `options.reference_cells`；这里的
    //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::CaCfarOptions options;；值在
    //  ZKX/Task1/task1_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::CaCfarOptions options;
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.pfa = config.pfa;` 是赋值：先求得右侧完整表达式 `config.pfa`，再把结果写入左侧可修改对象 `options.pfa`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `const int outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;` 是声明并初始化：`const int
    //  outer_doppler` 建立局部对象 `outer_doppler`，右侧完整表达式 `config.cfar_guard_doppler + config.cfar_reference_doppler`
    //  产生初值。
    //  `const int outer_range = config.cfar_guard_range + config.cfar_reference_range;` 是声明并初始化：`const int
    //  outer_range` 建立局部对象 `outer_range`，右侧完整表达式 `config.cfar_guard_range + config.cfar_reference_range` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  outer_doppler：当前块的赋值左端，保存右侧表达式产生的中间值或结果 outer_doppler；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const int
    //  outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;；值在
    //  ZKX/Task1/task1_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  outer_range：当前块的赋值左端，保存右侧表达式产生的中间值或结果 outer_range；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const int
    //  outer_range = config.cfar_guard_range + config.cfar_reference_range;；值在
    //  ZKX/Task1/task1_gpu_cpu/step4/step4.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  设保护半径 $G_d,G_r$、参考半径 $R_d,R_r$，外框 $O_d=G_d+R_d,O_r=G_r+R_r$；训练单元数 $N=(2O_d+1)(2O_r+1)-(2G_d+1)(2G_r+1)$。
    //
    //  【为什么这样设计】
    //  `cfar_alpha` 的 $N$ 必须与 CA-CFAR 训练环一致；漏减保护区或交换维度会使门限比例错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.pfa = config.pfa;
    const int outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int outer_range = config.cfar_guard_range + config.cfar_reference_range;
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const int reference_count = (2 * outer_doppler + 1) * (2 * outer_range + 1) - (2 *
    //  config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);` 是声明并初始化：`const int reference_count`
    //  建立局部对象 `reference_count`，右侧完整表达式 `(2 * outer_doppler + 1) * (2 * outer_range + 1) - (2 *
    //  config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1)` 产生初值。
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  reference_count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  2：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  设保护半径 $G_d,G_r$、参考半径 $R_d,R_r$，外框 $O_d=G_d+R_d,O_r=G_r+R_r$；训练单元数 $N=(2O_d+1)(2O_r+1)-(2G_d+1)(2G_r+1)$。
    //
    //  【为什么这样设计】
    //  `cfar_alpha` 的 $N$ 必须与 CA-CFAR 训练环一致；漏减保护区或交换维度会使门限比例错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const int reference_count = (2 * outer_doppler + 1) * (2 * outer_range + 1) -
        (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);
    evidence.prep_ms = milliseconds(begin, Clock::now());
    // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_range_doppler_fallback;` 是对象声明：类型
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>>` 应用于名称
    //  `d_range_doppler_fallback`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `const ComplexFloat* range_doppler_device = nullptr;` 是声明并初始化：`const ComplexFloat* range_doppler_device`
    //  建立局部对象 `range_doppler_device`，右侧完整表达式 `nullptr` 产生初值。 声明符中的 `*` 表示指针，不是乘法。
    //
    //  【变量—数学符号—数据流】
    //  d_range_doppler_fallback：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task1/task1_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  range_doppler_device：GPU 路径的对象、结果或计时字段；与同名 CPU/Host 量采用相同数学定义；由 device 调用产生，供 D2H、comparison 或证据消费
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。>
    begin = Clock::now();
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_range_doppler_fallback;
    const ComplexFloat* range_doppler_device = nullptr;
    // <学习注释：语义块：run_step4：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (has_resident_input) {` 是 `if` 条件语句：先把完整条件 `has_resident_input` 求值并转换为布尔值。
    //  条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `range_doppler_device = state.range_doppler_device.data();` 是赋值：先求得右侧完整表达式
    //  `state.range_doppler_device.data()`，再把结果写入左侧可修改对象 `range_doppler_device`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `evidence.h2d_ms = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //
    //  【变量—数学符号—数据流】
    //  range_doppler_device：GPU 路径的对象、结果或计时字段；与同名 CPU/Host 量采用相同数学定义；由 device 调用产生，供 D2H、comparison 或证据消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    if (has_resident_input) {
        range_doppler_device = state.range_doppler_device.data();
        evidence.h2d_ms = 0.0;
        // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `evidence.metrics["resident_input_reused"] = 1.0;` 是赋值：先求得右侧完整表达式 `1.0`，再把结果写入左侧可修改对象
        //  `evidence.metrics["resident_input_reused"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `} else {` 先用 `}` 关闭前一个 `if` 分支，再由 `else {` 打开互斥的假分支；只有前一个条件为假时才执行该分支。
        //  `static_assert(sizeof(cusignal::TypedComplex<float>) == sizeof(ComplexFloat));` 是编译期断言：编译器检查常量条件
        //  `sizeof(cusignal::TypedComplex<float>) ==
        //  sizeof(ComplexFloat)`；条件为假则编译失败。它不是运行时函数调用，本项目用它证明两种复数元素布局的字节大小相同。
        //
        //  【变量—数学符号—数据流】
        //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
        //
        //  【数字常量】
        //  1.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
        //
        //  【为什么这样设计】
        //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        evidence.metrics["resident_input_reused"] = 1.0;
    } else {
        static_assert(sizeof(cusignal::TypedComplex<float>) ==
            sizeof(ComplexFloat));
        // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `d_range_doppler_fallback = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(
        //  state.range_doppler);` 是赋值：先求得右侧完整表达式 `cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(
        //  state.range_doppler)`，再把结果写入左侧可修改对象 `d_range_doppler_fallback`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //
        //  【变量—数学符号—数据流】
        //  d_range_doppler_fallback：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
        //  ZKX/Task1/task1_gpu_cpu/step4/step4.cu 的 GPU 调用链
        //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
        //
        //  【为什么这样设计】
        //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
        //
        //  【初学者易错点】
        //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。>
        d_range_doppler_fallback =
            cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(
                state.range_doppler);
        // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `range_doppler_device = reinterpret_cast<const ComplexFloat*>( d_range_doppler_fallback.data());`
        //  是赋值：先求得右侧完整表达式 `reinterpret_cast<const ComplexFloat*>( d_range_doppler_fallback.data())`，再把结果写入左侧可修改对象
        //  `range_doppler_device`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //
        //  【变量—数学符号—数据流】
        //  range_doppler_device：GPU 路径的对象、结果或计时字段；与同名 CPU/Host 量采用相同数学定义；由 device 调用产生，供 D2H、comparison 或证据消费
        //
        //  【数学/物理公式对照】
        //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
        //
        //  【为什么这样设计】
        //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        range_doppler_device =
            reinterpret_cast<const ComplexFloat*>(
                d_range_doppler_fallback.data());
        // <学习注释：语义块：run_step4：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
        //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `evidence.metrics["resident_input_reused"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象
        //  `evidence.metrics["resident_input_reused"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
        //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
        //
        //  【变量—数学符号—数据流】
        //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
        //
        //  【数字常量】
        //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
        //
        //
        //  【数学/物理公式对照】
        //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
        //
        //  【为什么这样设计】
        //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        evidence.h2d_ms = milliseconds(begin, Clock::now());
        evidence.metrics["resident_input_reused"] = 0.0;
    }
    // <学习注释：语义块：d_power：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<float> d_power(count);` 声明 `d_power`，其静态类型是 `cusignal::DeviceArray<float>`，并用
    //  `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_power` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["range_doppler_power"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel(
    //  power_kernel, static_cast<std::size_t>(count), range_doppler_device, d_power.data(), count); });`
    //  先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入
    //  `evidence.operator_ms["range_doppler_power"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_power：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task1/task1_gpu_cpu/step4/step4.cu 的 GPU 调用链
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  `input[index]=z=a+jb`，输出 `a*a+b*b=|z|²`；dtype 从复 `float` 变为实 `float`，shape/展平索引不变。每个有效线程读一个复数并写一个功率值。
    //
    //  【为什么这样设计】
    //  CFAR 比较非负功率；模平方避免平方根并保持排序。每线程一单元没有写冲突。
    //
    //  【初学者易错点】
    //  kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。>
    cusignal::DeviceArray<float> d_power(count);
    evidence.operator_ms["range_doppler_power"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            power_kernel, static_cast<std::size_t>(count),
            range_doppler_device, d_power.data(), count);
    });
    // <学习注释：语义块：d_power：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<double> d_alpha;` 是对象声明：类型 `cusignal::DeviceArray<double>` 应用于名称
    //  `d_alpha`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.operator_ms["cfar_alpha"] = time_gpu([&] { d_alpha =
    //  cusignal::cfar_alpha_device<float>(config.pfa, reference_count); });` 先创建按引用捕获当前作用域对象的 lambda `[&]
    //  {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["cfar_alpha"]`。lambda
    //  内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_alpha：CA-CFAR 门限缩放系数；记作 $\alpha$；由 PFA 与训练单元数决定；门限为 $T=\alpha\hat P_n$
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  设保护半径 $G_d,G_r$、参考半径 $R_d,R_r$，外框 $O_d=G_d+R_d,O_r=G_r+R_r$；训练单元数 $N=(2O_d+1)(2O_r+1)-(2G_d+1)(2G_r+1)$。
    //
    //  【为什么这样设计】
    //  `cfar_alpha` 的 $N$ 必须与 CA-CFAR 训练环一致；漏减保护区或交换维度会使门限比例错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<double> d_alpha;
    evidence.operator_ms["cfar_alpha"] = time_gpu([&] {
        d_alpha = cusignal::cfar_alpha_device<float>(config.pfa, reference_count);
    });
    // <学习注释：语义块：d_power：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `thread_local cusignal::DeviceArray<float> d_cfar_threshold;` 是对象声明：类型 `cusignal::DeviceArray<float>`
    //  应用于名称 `d_cfar_threshold`。`thread_local` 使每个 Host 线程各自持有一份对象；类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `thread_local cusignal::DeviceArray<cusignal::CfarDetection> d_cfar_detections;` 是对象声明：类型
    //  `cusignal::DeviceArray<cusignal::CfarDetection>` 应用于名称 `d_cfar_detections`。`thread_local` 使每个 Host
    //  线程各自持有一份对象；类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `if (d_cfar_threshold.size() != static_cast<std::size_t>(count)) {` 是 `if` 条件语句：先把完整条件
    //  `d_cfar_threshold.size() != static_cast<std::size_t>(count)` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `d_cfar_threshold.reset(static_cast<std::size_t>(count));` 调用资源对象的
    //  `reset`，按给定元素数重新建立其持有的缓冲区；旧资源由对象的所有权逻辑释放。
    //
    //  【变量—数学符号—数据流】
    //  d_cfar_threshold：逐单元检测门限；记作 $T[i]$；与 CUT 功率比较产生 detection
    //  d_cfar_detections：检测布尔/索引结果；记作 $\mathcal D$；CFAR 输出，供统计与可视化
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    thread_local cusignal::DeviceArray<float> d_cfar_threshold;
    thread_local cusignal::DeviceArray<cusignal::CfarDetection> d_cfar_detections;
    if (d_cfar_threshold.size() != static_cast<std::size_t>(count)) {
        d_cfar_threshold.reset(static_cast<std::size_t>(count));
    // <学习注释：语义块：d_power：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `if (d_cfar_detections.size() != static_cast<std::size_t>(count)) {` 是 `if` 条件语句：先把完整条件
    //  `d_cfar_detections.size() != static_cast<std::size_t>(count)` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `d_cfar_detections.reset(static_cast<std::size_t>(count));` 调用资源对象的
    //  `reset`，按给定元素数重新建立其持有的缓冲区；旧资源由对象的所有权逻辑释放。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    }
    if (d_cfar_detections.size() != static_cast<std::size_t>(count)) {
        d_cfar_detections.reset(static_cast<std::size_t>(count));
    // <学习注释：语义块：d_power：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `evidence.operator_ms["ca_cfar"] = time_gpu([&] { cusignal::ca_cfar_device<float>( d_power,
    //  {config.num_pulses, config.samples_per_pulse}, options, d_cfar_threshold, d_cfar_detections); });`
    //  先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["ca_cfar"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `d_power→x[i,j]≥0`，shape `{num_pulses,samples_per_pulse}`；guard/reference 字段定义训练集合 $𝒯$。$\hat P=\frac1N\sum_{(u,v)∈𝒯}x[u,v]$，$T=α\hat P$，CUT 超过 $T$ 时检测。threshold 与 detections 均保持输入二维 shape。
    //
    //  【为什么这样设计】
    //  二维训练窗适应距离和 Doppler 背景，保护单元避免目标能量污染估计；算子内部 kernel/边界策略归 `Learning/operators/radartools/ca_cfar/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    }
    evidence.operator_ms["ca_cfar"] = time_gpu([&] {
        cusignal::ca_cfar_device<float>(
            d_power, {config.num_pulses, config.samples_per_pulse}, options,
            d_cfar_threshold, d_cfar_detections);
    });
    // <学习注释：语义块：d_power：形成并返回当前结果。
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
    // <学习注释：语义块：d_power：完成一个连续的数据处理动作。
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
    // <学习注释：语义块：d_power：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `state.power = d_power.to_host();` 是赋值：先求得右侧完整表达式 `d_power.to_host()`，再把结果写入左侧可修改对象 `state.power`；这里的 `=`
    //  不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `state.cfar_alpha = d_alpha.to_host().front();` 是赋值：先求得右侧完整表达式 `d_alpha.to_host().front()`，再把结果写入左侧可修改对象
    //  `state.cfar_alpha`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
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
    state.power = d_power.to_host();
    state.cfar_alpha = d_alpha.to_host().front();
    // <学习注释：语义块：d_power：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.cfar_threshold = d_cfar_threshold.to_host();` 是赋值：先求得右侧完整表达式
    //  `d_cfar_threshold.to_host()`，再把结果写入左侧可修改对象 `state.cfar_threshold`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `state.detections = d_cfar_detections.to_host();` 是赋值：先求得右侧完整表达式
    //  `d_cfar_detections.to_host()`，再把结果写入左侧可修改对象 `state.detections`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  detections：检测布尔/索引结果；记作 $\mathcal D$；CFAR 输出，供统计与可视化
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
    state.cfar_threshold = d_cfar_threshold.to_host();
    state.detections = d_cfar_detections.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    // <学习注释：语义块：d_power：形成并返回当前结果。
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

// <学习注释：语义块：d_power：完成一个连续的数据处理动作。
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

### 3.3 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`：`generate_step4_cpu_reference`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step4ValidationData generate_step4_cpu_reference(const PipelineState& state) {` 中的
//  `generate_step4_cpu_reference` 是函数定义；名称前的 `Step4ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  generate_step4_cpu_reference：生成当前名称所描述数据的 helper/结果；对应生成公式的输出端；由输入参数构造数据，供后续 step 或测试使用
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
Step4ValidationData generate_step4_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `Step4ValidationData data;` 是对象声明：类型 `Step4ValidationData` 应用于名称
    //  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象
    //  `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。
    //  `cusignal::CaCfarOptions options;` 是对象声明：类型 `cusignal::CaCfarOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //
    //  【变量—数学符号—数据流】
    //  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
    //  count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::CaCfarOptions options;；值在
    //  ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    Step4ValidationData data;
    const int count = config.num_pulses * config.samples_per_pulse;
    cusignal::CaCfarOptions options;
    // <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};` 是赋值：先求得右侧完整表达式
    //  `{config.cfar_guard_doppler, config.cfar_guard_range}`，再把结果写入左侧可修改对象 `options.guard_cells`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};` 是赋值：先求得右侧完整表达式
    //  `{config.cfar_reference_doppler, config.cfar_reference_range}`，再把结果写入左侧可修改对象 `options.reference_cells`；这里的
    //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.pfa = config.pfa;` 是赋值：先求得右侧完整表达式 `config.pfa`，再把结果写入左侧可修改对象 `options.pfa`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
    options.pfa = config.pfa;
    // <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;` 是声明并初始化：`const int od` 建立局部对象
    //  `od`，右侧完整表达式 `config.cfar_guard_doppler + config.cfar_reference_doppler` 产生初值。
    //  `const int orng = config.cfar_guard_range + config.cfar_reference_range;` 是声明并初始化：`const int orng` 建立局部对象
    //  `orng`，右侧完整表达式 `config.cfar_guard_range + config.cfar_reference_range` 产生初值。
    //  `const int reference_count = (2 * od + 1) * (2 * orng + 1) - (2 * config.cfar_guard_doppler + 1) * (2 *
    //  config.cfar_guard_range + 1);` 是声明并初始化：`const int reference_count` 建立局部对象 `reference_count`，右侧完整表达式 `(2 *
    //  od + 1) * (2 * orng + 1) - (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  od：当前块的赋值左端，保存右侧表达式产生的中间值或结果 od；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const int od =
    //  config.cfar_guard_doppler + config.cfar_reference_doppler;；值在
    //  ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu 当前作用域中产生或消费
    //  orng：当前块的赋值左端，保存右侧表达式产生的中间值或结果 orng；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const int orng =
    //  config.cfar_guard_range + config.cfar_reference_range;；值在
    //  ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu 当前作用域中产生或消费
    //  reference_count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  2：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  设保护半径 $G_d,G_r$、参考半径 $R_d,R_r$，外框 $O_d=G_d+R_d,O_r=G_r+R_r$；训练单元数 $N=(2O_d+1)(2O_r+1)-(2G_d+1)(2G_r+1)$。
    //
    //  【为什么这样设计】
    //  `cfar_alpha` 的 $N$ 必须与 CA-CFAR 训练环一致；漏减保护区或交换维度会使门限比例错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int orng = config.cfar_guard_range + config.cfar_reference_range;
    const int reference_count = (2 * od + 1) * (2 * orng + 1) -
        (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);
    // <学习注释：语义块：generate_step4_cpu_reference：迭代处理数据范围。
    //
    //  【执行方法】
    //  `data.power_cpu.resize(count);` 调用容器的 `resize` 把逻辑元素数改为实参指定的大小；增大时创建新元素，缩小时移除尾部元素。
    //  `for (int index = 0; index < count; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index <
    //  count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //  `const auto value = state.range_doppler[index];` 是声明并初始化：`const auto value` 建立局部对象 `value`，右侧完整表达式
    //  `state.range_doppler[index]` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
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
    data.power_cpu.resize(count);
    for (int index = 0; index < count; ++index) {
        const auto value = state.range_doppler[index];
        // <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
        //
        //  【执行方法】
        //  `data.power_cpu[index] = value.real * value.real + value.imag * value.imag;` 是赋值：先求得右侧完整表达式 `value.real *
        //  value.real + value.imag * value.imag`，再把结果写入左侧可修改对象 `data.power_cpu[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
        //  `!=`。
        //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
        //  `const auto cpu = cusignal::ca_cfar_typed_cpu<float>( data.power_cpu, {config.num_pulses,
        //  config.samples_per_pulse}, options);` 是声明并初始化：`const auto cpu` 建立局部对象 `cpu`，右侧完整表达式
        //  `cusignal::ca_cfar_typed_cpu<float>( data.power_cpu, {config.num_pulses, config.samples_per_pulse},
        //  options)` 产生初值。
        //
        //  【变量—数学符号—数据流】
        //  cpu：CPU/reference 路径的对象或结果；与任务正式公式相同，用作独立参考；由 Host 计算产生，供 GPU/CPU comparison
        //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
        //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
        //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
        //
        //  【数学/物理公式对照】
        //  `d_power→x[i,j]≥0`，shape `{num_pulses,samples_per_pulse}`；guard/reference 字段定义训练集合 $𝒯$。$\hat P=\frac1N\sum_{(u,v)∈𝒯}x[u,v]$，$T=α\hat P$，CUT 超过 $T$ 时检测。threshold 与 detections 均保持输入二维 shape。
        //
        //  【为什么这样设计】
        //  二维训练窗适应距离和 Doppler 背景，保护单元避免目标能量污染估计；算子内部 kernel/边界策略归 `Learning/operators/radartools/ca_cfar/`。
        //
        //  【初学者易错点】
        //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
        data.power_cpu[index] = value.real * value.real + value.imag * value.imag;
    }
    const auto cpu = cusignal::ca_cfar_typed_cpu<float>(
        data.power_cpu, {config.num_pulses, config.samples_per_pulse}, options);
    // <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.threshold_cpu = cpu.threshold;` 是赋值：先求得右侧完整表达式 `cpu.threshold`，再把结果写入左侧可修改对象
    //  `data.threshold_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `data.detections_cpu = cpu.detections;` 是赋值：先求得右侧完整表达式 `cpu.detections`，再把结果写入左侧可修改对象
    //  `data.detections_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `data.detections_same_power_cpu = cpu.detections;` 是赋值：先求得右侧完整表达式 `cpu.detections`，再把结果写入左侧可修改对象
    //  `data.detections_same_power_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  threshold：逐单元检测门限；记作 $T[i]$；与 CUT 功率比较产生 detection
    //  detections：检测布尔/索引结果；记作 $\mathcal D$；CFAR 输出，供统计与可视化
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.threshold_cpu = cpu.threshold;
    data.detections_cpu = cpu.detections;
    data.detections_same_power_cpu = cpu.detections;
    // <学习注释：语义块：generate_step4_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count);` 是赋值：先求得右侧完整表达式
    //  `cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count)`，再把结果写入左侧可修改对象 `data.alpha_cpu`；这里的
    //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  config：外部配置对象；`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  设保护半径 $G_d,G_r$、参考半径 $R_d,R_r$，外框 $O_d=G_d+R_d,O_r=G_r+R_r$；训练单元数 $N=(2O_d+1)(2O_r+1)-(2G_d+1)(2G_r+1)$。
    //
    //  【为什么这样设计】
    //  `cfar_alpha` 的 $N$ 必须与 CA-CFAR 训练环一致；漏减保护区或交换维度会使门限比例错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count);

    // <学习注释：语义块：generate_step4_cpu_reference：形成并返回当前结果。
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

### 3.4 函数总体处理方法

#### 3.4.1 总体业务流水线

`Step3 Doppler 功率→cfar_alpha→ca_cfar→D2H`。形成 threshold/detections，交给证据与可视化。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`power_kernel`|每线程计算一个复数距离—多普勒单元的模平方|GPU device kernel|
|`run_step4`|形成 Doppler 功率，配置并调用 CFAR 算子，回收阈值/检测结果|GPU Host wrapper/业务入口|
|`generate_step4_cpu_reference`|用同一参数生成功率、阈值和检测 CPU reference|CPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `cfar_alpha`

- 学习入口：[数学物理原理](../../operators/radartools/cfar_alpha/数学物理原理.md)、[Python 源码算法](../../operators/radartools/cfar_alpha/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/radartools/cfar_alpha/cusignal_cpp_cfar_alpha复现逻辑.md)。
- 当前调用采用的数学关系：指数噪声假设下 $\alpha=N(P_{FA}^{-1/N}-1)$。
- 任务变量与参数实例化：`pfa→P_FA`、训练单元总数 `N`、输出 `alpha→α`。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 两个正式算子的参数闭环

先由保护/参考半径计算训练单元数
$N=(2(G_d+R_d)+1)(2(G_r+R_r)+1)-(2G_d+1)(2G_r+1)$；随后
`cfar_alpha_device<float>(pfa,N)` 计算 $\alpha=N(P_{FA}^{-1/N}-1)$，
`ca_cfar_device<float>` 对功率图执行 $\hat P=\frac1N\sum_{\mathcal T}x$、$T=\alpha\hat P$。

|实参|数学量|dtype/shape 与去向|
|---|---|---|
|`d_power`|$x[i,j]=|D[i,j]|^2$|device `float[P×M]`，私有 power kernel 生成|
|`{num_pulses,samples_per_pulse}`|二维 shape $(P,M)$|决定 Doppler/距离轴顺序|
|`guard_cells={G_d,G_r}`|保护半径|排除 CUT 邻域目标泄漏|
|`reference_cells={R_d,R_r}`|训练环厚度|与 `reference_count=N` 必须一致|
|`pfa`|$P_{FA}$|配置标量；进入 $\alpha$|
|`d_cfar_threshold`|$T[i,j]$|device/Host `float[P×M]`|
|`d_cfar_detections`|检测判决|device/Host `CfarDetection[P×M]`|

##### `ca_cfar`

- 学习入口：[数学物理原理](../../operators/radartools/ca_cfar/数学物理原理.md)、[Python 源码算法](../../operators/radartools/ca_cfar/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/radartools/ca_cfar/cusignal_cpp_ca_cfar复现逻辑.md)。
- 当前调用采用的数学关系：$\hat P_n=\frac1N\sum_{i\in\mathcal T}x_i$，$T=\alpha\hat P_n$，CUT 超过 $T$ 时检测。
- 任务变量与参数实例化：`doppler power→x`、训练/保护单元定义 $\mathcal T$、`threshold→T`、`detections→𝒟`。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step4(PipelineState& state); }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
- `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task1 { StepEvidence run_step4(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence run_step4(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|公共基础设施|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `#include "step4.h"`。

```cpp
#include "step4.h"

#include "cuda_utils/device_array.h"
#include "cuda_utils/kernel_launch.h"
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step4`、`h`、`cuda_utils`、`device_array`、`kernel_launch`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step4.h"` 在预处理阶段引入 "step4.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/kernel_launch.h"` 在预处理阶段引入 "cuda_utils/kernel_launch.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|公共基础设施|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 power_kernel：声明 CUDA kernel 并建立线程入口

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `power_kernel`；源码锚点 `__global__ void power_kernel(`。

```cpp
__global__ void power_kernel(
    const ComplexFloat* input, float* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `input` 由 `const ComplexFloat*` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `output` 由 `float*` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`*` 说明变量保存地址，解引用前必须保证地址有效；
- `count` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `index` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`input`|当前函数或调用的参数名称，接收调用者绑定的输入 `input`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const ComplexFloat* input, float* output, int count)`；值在 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`output`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `__global__ void power_kernel( const ComplexFloat* input, float* output, int count) {` 中的 `power_kernel` 是函数定义；名称前的 `__global__ void` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const ComplexFloat* input, float* output, int count` 是形参声明。 `__global__` 表明该函数是由 Host 发起、在 GPU 上由许多线程并行执行的 CUDA kernel。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);` 是声明并初始化：`const int index` 建立局部对象 `index`，右侧完整表达式 `static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。 `blockIdx.x * blockDim.x + threadIdx.x` 把块号和块内线程号映射为一维全局线程索引。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过定义/声明 `power_kernel`；调用 `power_kernel`；写入/初始化 `index`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `power_kernel` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 使用 `blockIdx.x * blockDim.x + threadIdx.x` 是为了给每个一维输出元素分配唯一全局线程。这样同一 kernel 可适配不同 block 大小；随后必须用 count 做越界保护，因为 grid 往往向上取整。

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 power_kernel：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `power_kernel`；源码锚点 `if (index >= count) return;`。

```cpp
    if (index >= count) return;
    const auto value = input[index];
    output[index] = value.re * value.re + value.im * value.im;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 当前块中的 `if (...) return;` 必须先作为完整条件语句识别；比较运算符 `>=`/`<=` 中的 `=` 不是赋值。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `value` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `if (index >= count) return;` 是 `if` 条件语句：先把完整条件 `index >= count` 求值并转换为布尔值。 条件为真时执行 `return`，即结束当前函数或当前 CUDA 线程的 kernel 实例且不返回数值；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。
- `const auto value = input[index];` 是声明并初始化：`const auto value` 建立局部对象 `value`，右侧完整表达式 `input[index]` 产生初值。
- `output[index] = value.re * value.re + value.im * value.im;` 是赋值：先求得右侧完整表达式 `value.re * value.re + value.im * value.im`，再把结果写入左侧可修改对象 `output[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过检查 `index >= count`；写入/初始化 `value`, `output[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `power_kernel` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.6 power_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `power_kernel`；源码锚点 `}`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `power_kernel` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.7 power_kernel：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `power_kernel`；源码锚点 `}  // namespace`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `power_kernel` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `StepEvidence run_step4(PipelineState& state)`。

```cpp
StepEvidence run_step4(PipelineState& state)
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
|`run_step4`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step4(PipelineState& state) {` 中的 `run_step4` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过定义/声明 `run_step4`；调用 `run_step4`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.9 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step4";
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
- `evidence.name = "step4";` 是赋值：先求得右侧完整表达式 `"step4"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto formal_begin = Clock::now();` 是声明并初始化：`const auto formal_begin` 建立局部对象 `formal_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `formal_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step4` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    const int count = config.num_pulses * config.samples_per_pulse;
    const bool has_resident_input =
        state.range_doppler_device.size() == static_cast<std::size_t>(count);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `has_resident_input` 由 `const bool` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`has_resident_input`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `has_resident_input`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const bool has_resident_input =`；值在 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象 `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。
- `const bool has_resident_input = state.range_doppler_device.size() == static_cast<std::size_t>(count);` 是声明并初始化：`const bool has_resident_input` 建立局部对象 `has_resident_input`，右侧完整表达式 `state.range_doppler_device.size() == static_cast<std::size_t>(count)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `Clock::now`, `size`；写入/初始化 `begin`, `count`, `has_resident_input`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.11 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `require_task1_upstream(has_resident_input ||`。

```cpp
    require_task1_upstream(has_resident_input ||
            state.range_doppler.size() == static_cast<std::size_t>(count),
            "Step4", "Step3.range_doppler");
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `require_task1_upstream`、`has_resident_input`、`state`、`range_doppler`、`size`、`std`、`size_t`、`count`、`Step4`、`Step3`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `require_task1_upstream(has_resident_input || state.range_doppler.size() == static_cast<std::size_t>(count), "Step4", "Step3.range_doppler");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `require_task1_upstream`, `size`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `cusignal::CaCfarOptions options;`。

```cpp
    cusignal::CaCfarOptions options;
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `options` 由 `cusignal::CaCfarOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::CaCfarOptions options;`；值在 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::CaCfarOptions options;` 是对象声明：类型 `cusignal::CaCfarOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};` 是赋值：先求得右侧完整表达式 `{config.cfar_guard_doppler, config.cfar_guard_range}`，再把结果写入左侧可修改对象 `options.guard_cells`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};` 是赋值：先求得右侧完整表达式 `{config.cfar_reference_doppler, config.cfar_reference_range}`，再把结果写入左侧可修改对象 `options.reference_cells`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `guard_cells`, `reference_cells`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `options.pfa = config.pfa;`。

```cpp
    options.pfa = config.pfa;
    const int outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int outer_range = config.cfar_guard_range + config.cfar_reference_range;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `outer_doppler` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `outer_range` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`outer_doppler`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `outer_doppler`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const int outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;`；值在 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`outer_range`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `outer_range`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const int outer_range = config.cfar_guard_range + config.cfar_reference_range;`；值在 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.pfa = config.pfa;` 是赋值：先求得右侧完整表达式 `config.pfa`，再把结果写入左侧可修改对象 `options.pfa`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const int outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;` 是声明并初始化：`const int outer_doppler` 建立局部对象 `outer_doppler`，右侧完整表达式 `config.cfar_guard_doppler + config.cfar_reference_doppler` 产生初值。
- `const int outer_range = config.cfar_guard_range + config.cfar_reference_range;` 是声明并初始化：`const int outer_range` 建立局部对象 `outer_range`，右侧完整表达式 `config.cfar_guard_range + config.cfar_reference_range` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `pfa`, `outer_doppler`, `outer_range`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.14 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `const int reference_count = (2 * outer_doppler + 1) * (2 * outer_range + 1) -`。

```cpp
    const int reference_count = (2 * outer_doppler + 1) * (2 * outer_range + 1) -
        (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `reference_count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`reference_count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const int reference_count = (2 * outer_doppler + 1) * (2 * outer_range + 1) - (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);` 是声明并初始化：`const int reference_count` 建立局部对象 `reference_count`，右侧完整表达式 `(2 * outer_doppler + 1) * (2 * outer_range + 1) - (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1)` 产生初值。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `reference_count`, `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_range_doppler_fallback;
    const ComplexFloat* range_doppler_device = nullptr;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_range_doppler_fallback` 由 `cusignal::DeviceArray<cusignal::TypedComplex<float>>` 声明：由两个 float 分量表示的单精度复数；尾部 `*` 时形参保存 device/Host 缓冲区首元素地址而不是复制整段数组；
- `range_doppler_device` 由 `const ComplexFloat*` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`*` 说明变量保存地址，解引用前必须保证地址有效。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_range_doppler_fallback`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`range_doppler_device`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::DeviceArray<cusignal::TypedComplex<float>> d_range_doppler_fallback;` 是对象声明：类型 `cusignal::DeviceArray<cusignal::TypedComplex<float>>` 应用于名称 `d_range_doppler_fallback`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `const ComplexFloat* range_doppler_device = nullptr;` 是声明并初始化：`const ComplexFloat* range_doppler_device` 建立局部对象 `range_doppler_device`，右侧完整表达式 `nullptr` 产生初值。 声明符中的 `*` 表示指针，不是乘法。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|公共基础设施|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `begin`, `range_doppler_device`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.16 run_step4：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `if (has_resident_input) {`。

```cpp
    if (has_resident_input) {
        range_doppler_device = state.range_doppler_device.data();
        evidence.h2d_ms = 0.0;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`range_doppler_device`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `if (has_resident_input) {` 是 `if` 条件语句：先把完整条件 `has_resident_input` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `range_doppler_device = state.range_doppler_device.data();` 是赋值：先求得右侧完整表达式 `state.range_doppler_device.data()`，再把结果写入左侧可修改对象 `range_doppler_device`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.h2d_ms = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过检查 `has_resident_input`；调用 `data`；写入/初始化 `range_doppler_device`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.17 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `evidence.metrics["resident_input_reused"] = 1.0;`。

```cpp
        evidence.metrics["resident_input_reused"] = 1.0;
    } else {
        static_assert(sizeof(cusignal::TypedComplex<float>) ==
            sizeof(ComplexFloat));
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `evidence.metrics["resident_input_reused"] = 1.0;` 是赋值：先求得右侧完整表达式 `1.0`，再把结果写入左侧可修改对象 `evidence.metrics["resident_input_reused"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `} else {` 先用 `}` 关闭前一个 `if` 分支，再由 `else {` 打开互斥的假分支；只有前一个条件为假时才执行该分支。
- `static_assert(sizeof(cusignal::TypedComplex<float>) == sizeof(ComplexFloat));` 是编译期断言：编译器检查常量条件 `sizeof(cusignal::TypedComplex<float>) == sizeof(ComplexFloat)`；条件为假则编译失败。它不是运行时函数调用，本项目用它证明两种复数元素布局的字节大小相同。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `static_assert`；写入/初始化 `metrics["resident_input_reused"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step4` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `d_range_doppler_fallback =`。

```cpp
        d_range_doppler_fallback =
            cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(
                state.range_doppler);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `d_range_doppler_fallback`、`cusignal`、`DeviceArray`、`TypedComplex`、`float`、`from_host`、`state`、`range_doppler`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_range_doppler_fallback`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `d_range_doppler_fallback = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host( state.range_doppler);` 是赋值：先求得右侧完整表达式 `cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host( state.range_doppler)`，再把结果写入左侧可修改对象 `d_range_doppler_fallback`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|公共基础设施|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `from_host`；写入/初始化 `d_range_doppler_fallback`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.19 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `range_doppler_device =`。

```cpp
        range_doppler_device =
            reinterpret_cast<const ComplexFloat*>(
                d_range_doppler_fallback.data());
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `range_doppler_device`、`ComplexFloat`、`d_range_doppler_fallback`、`data`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`range_doppler_device`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `range_doppler_device = reinterpret_cast<const ComplexFloat*>( d_range_doppler_fallback.data());` 是赋值：先求得右侧完整表达式 `reinterpret_cast<const ComplexFloat*>( d_range_doppler_fallback.data())`，再把结果写入左侧可修改对象 `range_doppler_device`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `data`；写入/初始化 `range_doppler_device`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 run_step4：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `run_step4`；源码锚点 `evidence.h2d_ms = milliseconds(begin, Clock::now());`。

```cpp
        evidence.h2d_ms = milliseconds(begin, Clock::now());
        evidence.metrics["resident_input_reused"] = 0.0;
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.metrics["resident_input_reused"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.metrics["resident_input_reused"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `h2d_ms`, `metrics["resident_input_reused"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step4` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.21 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `cusignal::DeviceArray<float> d_power(count);`。

```cpp
    cusignal::DeviceArray<float> d_power(count);
    evidence.operator_ms["range_doppler_power"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            power_kernel, static_cast<std::size_t>(count),
            range_doppler_device, d_power.data(), count);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_power`、`count`、`evidence`、`operator_ms`、`range_doppler_power`、`time_gpu`、`cuda_utils`、`launch_1d_kernel`、`power_kernel`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_power`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_power(count);` 声明 `d_power`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_power` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["range_doppler_power"] = time_gpu([&] { cusignal::cuda_utils::launch_1d_kernel( power_kernel, static_cast<std::size_t>(count), range_doppler_device, d_power.data(), count); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 launch_1d_kernel，返回的毫秒值写入 `evidence.operator_ms["range_doppler_power"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|公共基础设施|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `d_power`, `time_gpu`, `cusignal::cuda_utils::launch_1d_kernel`, `data`；写入/初始化 `operator_ms["range_doppler_power"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- kernel 启动与其完成不是同一时刻；若要把耗时或 D2H 作为完成证据，必须确认封装或后续代码执行了同步。

### 4.22 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `cusignal::DeviceArray<double> d_alpha;`。

```cpp
    cusignal::DeviceArray<double> d_alpha;
    evidence.operator_ms["cfar_alpha"] = time_gpu([&] {
        d_alpha = cusignal::cfar_alpha_device<float>(config.pfa, reference_count);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_alpha` 由 `cusignal::DeviceArray<double>` 声明：通常为 IEEE 754 binary64 双精度浮点数；精度高于 float 但占用更多存储。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_alpha`|CA-CFAR 门限缩放系数|记作 $\alpha$|由 PFA 与训练单元数决定；门限为 $T=\alpha\hat P_n$|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<double> d_alpha;` 是对象声明：类型 `cusignal::DeviceArray<double>` 应用于名称 `d_alpha`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `evidence.operator_ms["cfar_alpha"] = time_gpu([&] { d_alpha = cusignal::cfar_alpha_device<float>(config.pfa, reference_count); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["cfar_alpha"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `time_gpu`；写入/初始化 `operator_ms["cfar_alpha"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.23 d_power：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `thread_local cusignal::DeviceArray<float> d_cfar_threshold;`。

```cpp
    thread_local cusignal::DeviceArray<float> d_cfar_threshold;
    thread_local cusignal::DeviceArray<cusignal::CfarDetection> d_cfar_detections;
    if (d_cfar_threshold.size() != static_cast<std::size_t>(count)) {
        d_cfar_threshold.reset(static_cast<std::size_t>(count));
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：表达式/声明单元、`if` 条件语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_cfar_threshold` 由 `cusignal::DeviceArray<float>` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；
- `d_cfar_detections` 由 `cusignal::DeviceArray<cusignal::CfarDetection>` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_cfar_threshold`|逐单元检测门限|记作 $T[i]$|与 CUT 功率比较产生 detection|
|`d_cfar_detections`|检测布尔/索引结果|记作 $\mathcal D$|CFAR 输出，供统计与可视化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `thread_local cusignal::DeviceArray<float> d_cfar_threshold;` 是对象声明：类型 `cusignal::DeviceArray<float>` 应用于名称 `d_cfar_threshold`。`thread_local` 使每个 Host 线程各自持有一份对象；类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `thread_local cusignal::DeviceArray<cusignal::CfarDetection> d_cfar_detections;` 是对象声明：类型 `cusignal::DeviceArray<cusignal::CfarDetection>` 应用于名称 `d_cfar_detections`。`thread_local` 使每个 Host 线程各自持有一份对象；类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `if (d_cfar_threshold.size() != static_cast<std::size_t>(count)) {` 是 `if` 条件语句：先把完整条件 `d_cfar_threshold.size() != static_cast<std::size_t>(count)` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `d_cfar_threshold.reset(static_cast<std::size_t>(count));` 调用资源对象的 `reset`，按给定元素数重新建立其持有的缓冲区；旧资源由对象的所有权逻辑释放。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|公共基础设施|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过检查 `d_cfar_threshold.size() != static_cast<std::size_t>(count)`；调用 `size`, `reset`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.24 d_power：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `}`。

```cpp
    }
    if (d_cfar_detections.size() != static_cast<std::size_t>(count)) {
        d_cfar_detections.reset(static_cast<std::size_t>(count));
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：作用域边界、`if` 条件语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `d_cfar_detections`、`size`、`std`、`size_t`、`count`、`reset`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `if (d_cfar_detections.size() != static_cast<std::size_t>(count)) {` 是 `if` 条件语句：先把完整条件 `d_cfar_detections.size() != static_cast<std::size_t>(count)` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `d_cfar_detections.reset(static_cast<std::size_t>(count));` 调用资源对象的 `reset`，按给定元素数重新建立其持有的缓冲区；旧资源由对象的所有权逻辑释放。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过检查 `d_cfar_detections.size() != static_cast<std::size_t>(count)`；调用 `size`, `reset`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块形成 CFAR 阈值或检测掩码；训练/保护单元、PFA 和输入功率共同决定检测结果。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.25 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `}`。

```cpp
    }
    evidence.operator_ms["ca_cfar"] = time_gpu([&] {
        cusignal::ca_cfar_device<float>(
            d_power, {config.num_pulses, config.samples_per_pulse}, options,
            d_cfar_threshold, d_cfar_detections);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：作用域边界、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`ca_cfar`、`time_gpu`、`cusignal`、`ca_cfar_device`、`float`、`d_power`、`config`、`num_pulses`、`samples_per_pulse`、`options`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `evidence.operator_ms["ca_cfar"] = time_gpu([&] { cusignal::ca_cfar_device<float>( d_power, {config.num_pulses, config.samples_per_pulse}, options, d_cfar_threshold, d_cfar_detections); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["ca_cfar"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|输入准备/数据契约|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `time_gpu`；写入/初始化 `operator_ms["ca_cfar"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.26 d_power：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `evidence.compute_ms = std::accumulate(`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `compute_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_power` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.27 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.28 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    state.power = d_power.to_host();
    state.cfar_alpha = d_alpha.to_host().front();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`power`、`d_power`、`to_host`、`cfar_alpha`、`d_alpha`、`front`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.power = d_power.to_host();` 是赋值：先求得右侧完整表达式 `d_power.to_host()`，再把结果写入左侧可修改对象 `state.power`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.cfar_alpha = d_alpha.to_host().front();` 是赋值：先求得右侧完整表达式 `d_alpha.to_host().front()`，再把结果写入左侧可修改对象 `state.cfar_alpha`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`, `front`；写入/初始化 `begin`, `power`, `cfar_alpha`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块形成 CFAR 阈值或检测掩码；训练/保护单元、PFA 和输入功率共同决定检测结果。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.29 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `state.cfar_threshold = d_cfar_threshold.to_host();`。

```cpp
    state.cfar_threshold = d_cfar_threshold.to_host();
    state.detections = d_cfar_detections.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`cfar_threshold`、`d_cfar_threshold`、`to_host`、`detections`、`d_cfar_detections`、`evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`detections`|检测布尔/索引结果|记作 $\mathcal D$|CFAR 输出，供统计与可视化|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.cfar_threshold = d_cfar_threshold.to_host();` 是赋值：先求得右侧完整表达式 `d_cfar_threshold.to_host()`，再把结果写入左侧可修改对象 `state.cfar_threshold`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.detections = d_cfar_detections.to_host();` 是赋值：先求得右侧完整表达式 `d_cfar_detections.to_host()`，再把结果写入左侧可修改对象 `state.detections`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `to_host`, `milliseconds`, `Clock::now`；写入/初始化 `cfar_threshold`, `detections`, `d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块形成 CFAR 阈值或检测掩码；训练/保护单元、PFA 和输入功率共同决定检测结果。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.30 d_power：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `formal_execution_ms`；返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_power` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.31 d_power：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu`；符号 `d_power`；源码锚点 `}  // namespace task1`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `d_power` 所在的 `ZKX/Task1/task1_gpu_cpu/step4/step4.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.32 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `Step4ValidationData generate_step4_cpu_reference(const PipelineState& state)`。

```cpp
Step4ValidationData generate_step4_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `config` 由 `const auto&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`generate_step4_cpu_reference`|生成当前名称所描述数据的 helper/结果|对应生成公式的输出端|由输入参数构造数据，供后续 step 或测试使用|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step4ValidationData generate_step4_cpu_reference(const PipelineState& state) {` 中的 `generate_step4_cpu_reference` 是函数定义；名称前的 `Step4ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过定义/声明 `generate_step4_cpu_reference`；调用 `generate_step4_cpu_reference`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.33 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `Step4ValidationData data;`。

```cpp
    Step4ValidationData data;
    const int count = config.num_pulses * config.samples_per_pulse;
    cusignal::CaCfarOptions options;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `data` 由 `Step4ValidationData` 声明：类型控制可表示值、可用操作和传参方式；
- `count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `options` 由 `cusignal::CaCfarOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|
|`count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::CaCfarOptions options;`；值在 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step4ValidationData data;` 是对象声明：类型 `Step4ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `const int count = config.num_pulses * config.samples_per_pulse;` 是声明并初始化：`const int count` 建立局部对象 `count`，右侧完整表达式 `config.num_pulses * config.samples_per_pulse` 产生初值。
- `cusignal::CaCfarOptions options;` 是对象声明：类型 `cusignal::CaCfarOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.34 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};`。

```cpp
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
    options.pfa = config.pfa;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `options`、`guard_cells`、`config`、`cfar_guard_doppler`、`cfar_guard_range`、`reference_cells`、`cfar_reference_doppler`、`cfar_reference_range`、`pfa`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};` 是赋值：先求得右侧完整表达式 `{config.cfar_guard_doppler, config.cfar_guard_range}`，再把结果写入左侧可修改对象 `options.guard_cells`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};` 是赋值：先求得右侧完整表达式 `{config.cfar_reference_doppler, config.cfar_reference_range}`，再把结果写入左侧可修改对象 `options.reference_cells`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.pfa = config.pfa;` 是赋值：先求得右侧完整表达式 `config.pfa`，再把结果写入左侧可修改对象 `options.pfa`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `guard_cells`, `reference_cells`, `pfa`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.35 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;`。

```cpp
    const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int orng = config.cfar_guard_range + config.cfar_reference_range;
    const int reference_count = (2 * od + 1) * (2 * orng + 1) -
        (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `od` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `orng` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `reference_count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `2` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`od`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `od`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;`；值在 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu` 当前作用域中产生或消费|
|`orng`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `orng`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const int orng = config.cfar_guard_range + config.cfar_reference_range;`；值在 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu` 当前作用域中产生或消费|
|`reference_count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;` 是声明并初始化：`const int od` 建立局部对象 `od`，右侧完整表达式 `config.cfar_guard_doppler + config.cfar_reference_doppler` 产生初值。
- `const int orng = config.cfar_guard_range + config.cfar_reference_range;` 是声明并初始化：`const int orng` 建立局部对象 `orng`，右侧完整表达式 `config.cfar_guard_range + config.cfar_reference_range` 产生初值。
- `const int reference_count = (2 * od + 1) * (2 * orng + 1) - (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);` 是声明并初始化：`const int reference_count` 建立局部对象 `reference_count`，右侧完整表达式 `(2 * od + 1) * (2 * orng + 1) - (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `od`, `orng`, `reference_count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.36 generate_step4_cpu_reference：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `data.power_cpu.resize(count);`。

```cpp
    data.power_cpu.resize(count);
    for (int index = 0; index < count; ++index) {
        const auto value = state.range_doppler[index];
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；
- `value` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `data.power_cpu.resize(count);` 调用容器的 `resize` 把逻辑元素数改为实参指定的大小；增大时创建新元素，缩小时移除尾部元素。
- `for (int index = 0; index < count; ++index) {` 是经典 `for`：先执行初始化 `int index = 0`；每轮前检查 `index < count`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `const auto value = state.range_doppler[index];` 是声明并初始化：`const auto value` 建立局部对象 `value`，右侧完整表达式 `state.range_doppler[index]` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过调用 `resize`；写入/初始化 `value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.37 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `data.power_cpu[index] = value.real * value.real + value.imag * value.imag;`。

```cpp
        data.power_cpu[index] = value.real * value.real + value.imag * value.imag;
    }
    const auto cpu = cusignal::ca_cfar_typed_cpu<float>(
        data.power_cpu, {config.num_pulses, config.samples_per_pulse}, options);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`cpu`|CPU/reference 路径的对象或结果|与任务正式公式相同，用作独立参考|由 Host 计算产生，供 GPU/CPU comparison|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.power_cpu[index] = value.real * value.real + value.imag * value.imag;` 是赋值：先求得右侧完整表达式 `value.real * value.real + value.imag * value.imag`，再把结果写入左侧可修改对象 `data.power_cpu[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `const auto cpu = cusignal::ca_cfar_typed_cpu<float>( data.power_cpu, {config.num_pulses, config.samples_per_pulse}, options);` 是声明并初始化：`const auto cpu` 建立局部对象 `cpu`，右侧完整表达式 `cusignal::ca_cfar_typed_cpu<float>( data.power_cpu, {config.num_pulses, config.samples_per_pulse}, options)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `power_cpu[index]`, `cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.38 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `data.threshold_cpu = cpu.threshold;`。

```cpp
    data.threshold_cpu = cpu.threshold;
    data.detections_cpu = cpu.detections;
    data.detections_same_power_cpu = cpu.detections;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`threshold_cpu`、`cpu`、`threshold`、`detections_cpu`、`detections`、`detections_same_power_cpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`threshold`|逐单元检测门限|记作 $T[i]$|与 CUT 功率比较产生 detection|
|`detections`|检测布尔/索引结果|记作 $\mathcal D$|CFAR 输出，供统计与可视化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.threshold_cpu = cpu.threshold;` 是赋值：先求得右侧完整表达式 `cpu.threshold`，再把结果写入左侧可修改对象 `data.threshold_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.detections_cpu = cpu.detections;` 是赋值：先求得右侧完整表达式 `cpu.detections`，再把结果写入左侧可修改对象 `data.detections_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.detections_same_power_cpu = cpu.detections;` 是赋值：先求得右侧完整表达式 `cpu.detections`，再把结果写入左侧可修改对象 `data.detections_same_power_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `threshold_cpu`, `detections_cpu`, `detections_same_power_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.39 generate_step4_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count);`。

```cpp
    data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`alpha_cpu`、`cusignal`、`cfar_alpha_typed_cpu`、`float`、`config`、`pfa`、`reference_count`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|`config` 在本 step 实例化 $P_{FA}$、训练单元数、保护单元数和 CFAR 边界/检测参数。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count);` 是赋值：先求得右侧完整表达式 `cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count)`，再把结果写入左侧可修改对象 `data.alpha_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过写入/初始化 `alpha_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- CFAR 用局部训练单元估计噪声并动态形成门限，是为了在噪声底变化时维持给定虚警率；固定全局阈值更简单，但对非平稳背景不稳健。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.40 generate_step4_cpu_reference：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step4_validation.cu`；符号 `generate_step4_cpu_reference`；源码锚点 `return data;`。

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
|赛题条款|T1-R4：恒虚警检测：用 `cfar_alpha` 与 `ca_cfar` 根据局部噪声动态生成门限并检测目标|
|业务角色|测试证据|
|业务输入|T1-R3 的距离—多普勒功率、PFA、训练单元和保护单元|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|门限 `threshold`、缩放系数 `alpha`、检测掩码/索引 `detections`|
|下一消费者|检测结果进入证据统计与可视化|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
