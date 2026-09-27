# Task2 Step2 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task2 Step2 的 Hamming 窗、低通 FIR taps 设计、resident FIR 滤波三段正式算子调用，以及相同参数下的 CPU reference 与前后数据关系。

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
|`ZKX/Task2/task2_gpu_cpu/step2/step2.h`|`run_step2` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step2/step2.cu`|`run_step2`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`|`collect_step2_validation`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`

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
//  `StepEvidence run_step2(PipelineState& state);` 中的 `run_step2` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。
//
//  【变量—数学符号—数据流】
//  run_step2：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
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

StepEvidence run_step2(PipelineState& state);

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

### 3.2 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step2.h"` 在预处理阶段引入 "step2.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "filter_design/filter_design_typed.h"` 在预处理阶段引入 "filter_design/filter_design_typed.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "filtering/filtering_typed.h"` 在预处理阶段引入 "filtering/filtering_typed.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "windows/windows_typed.h"` 在预处理阶段引入 "windows/windows_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
#include "step2.h"

#include "cuda_utils/device_array.h"
#include "filter_design/filter_design_typed.h"
#include "filtering/filtering_typed.h"
#include "windows/windows_typed.h"

// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
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

// <学习注释：语义块：run_step2：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `StepEvidence run_step2(PipelineState& state) {` 中的 `run_step2` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  run_step2：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
StepEvidence run_step2(PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：run_step2：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.name = "step2";` 是赋值：先求得右侧完整表达式 `"step2"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式
    //  `Clock::now()` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  total_begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    evidence.name = "step2";
    const auto total_begin = Clock::now();
    // <学习注释：语义块：run_step2：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //  `require_task2_upstream( state.echo.size() == static_cast<std::size_t>(config.samples), "Step2",
    //  "Step1.echo");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
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
    require_task2_upstream(
        state.echo.size() == static_cast<std::size_t>(config.samples),
        "Step2", "Step1.echo");
    // <学习注释：语义块：run_step2：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const std::vector<float> cutoff{config.filter_cutoff_hz};` 声明 `cutoff`，静态类型为 `const
    //  std::vector<float>`，并使用列表初始化器 `{config.filter_cutoff_hz}` 构造内容；花括号属于初始化，不是新的控制流作用域。
    //  `const int taps_count = config.filter_taps;` 是声明并初始化：`const int taps_count` 建立局部对象 `taps_count`，右侧完整表达式
    //  `config.filter_taps` 产生初值。
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  cutoff：当前表达式读取或传递的工程名称 cutoff；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>
    //  cutoff{config.filter_cutoff_hz};；值在 ZKX/Task2/task2_gpu_cpu/step2/step2.cu 当前作用域中产生或消费
    //  taps_count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
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
    const std::vector<float> cutoff{config.filter_cutoff_hz};
    const int taps_count = config.filter_taps;
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：语义块：run_step2：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);` 是声明并初始化：`auto d_echo` 建立局部对象
    //  `d_echo`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.echo)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  d_echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
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
    auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);
    // <学习注释：语义块：run_step2：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);` 是声明并初始化：`auto d_cutoff` 建立局部对象
    //  `d_cutoff`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(cutoff)` 产生初值。
    //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  d_cutoff：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step2/step2.cu 的 GPU 调用链
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
    auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);
    evidence.h2d_ms = milliseconds(begin, Clock::now());

    // <学习注释：语义块：d_window：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<float> d_window(taps_count);` 声明 `d_window`，其静态类型是
    //  `cusignal::DeviceArray<float>`，并用 `taps_count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_window` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["hamming"] = time_gpu([&] { cusignal::hamming_resident_device<float>(taps_count,
    //  d_window, true); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["hamming"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_window：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step2/step2.cu 的 GPU 调用链
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子生成 $L=\texttt{taps_count}$ 点对称 Hamming 窗 $w[n]=0.54-0.46\cos(2\pi n/(L-1))$；最后一个布尔实参为 `true`，按算子接口选择对称窗分支。GPU 输出 `d_window<float>`，CPU 输出 Host double/模板结果，随后作为 FIR 设计的显式窗。
    //
    //  【为什么这样设计】
    //  任务层绑定长度与对称模式；窗函数内部实现归 `Learning/operators/windows/hamming/`。Hamming 用较宽主瓣换取较低旁瓣。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<float> d_window(taps_count);
    evidence.operator_ms["hamming"] = time_gpu([&] {
        cusignal::hamming_resident_device<float>(taps_count, d_window, true);
    });
    // <学习注释：语义块：d_taps：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<float> d_taps(taps_count);` 声明 `d_taps`，其静态类型是 `cusignal::DeviceArray<float>`，并用
    //  `taps_count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_taps` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["firwin"] = time_gpu([&] { cusignal::firwin_lowpass_explicit_window_resident_device(
    //  taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true); });` 先创建按引用捕获当前作用域对象的 lambda `[&]
    //  {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["firwin"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  d_taps：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step2/step2.cu 的 GPU 调用链
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  正式算子以 `taps_count=L`、`cutoff/filter_cutoff_hz=f_c`、`fs=sample_rate_hz=f_s` 和显式 Hamming 窗设计低通 FIR。理想核为以中心 $m=(L-1)/2$ 平移的 sinc，实际 $h[n]=h_d[n]w[n]$；末尾 `true`/CPU options 选择缩放分支。输出 taps 长度为 $L$。
    //
    //  【为什么这样设计】
    //  显式传窗保证窗口选择可审计；设计内部的归一化和偶/奇长度边界归 `Learning/operators/filter_design/firwin/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<float> d_taps(taps_count);
    evidence.operator_ms["firwin"] = time_gpu([&] {
        cusignal::firwin_lowpass_explicit_window_resident_device(
            taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true);
    });
    // <学习注释：语义块：filtered：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::DeviceArray<float> filtered(config.samples);` 声明 `filtered`，其静态类型是
    //  `cusignal::DeviceArray<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `filtered` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.operator_ms["firfilter"] = time_gpu([&] { cusignal::firfilter_fp32_resident_device(d_taps,
    //  d_echo, filtered); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["firfilter"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  filtered：当前函数或调用的参数名称，接收调用者绑定的输入 filtered；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::DeviceArray<float>
    //  filtered(config.samples);；值在 ZKX/Task2/task2_gpu_cpu/step2/step2.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子执行 $y[n]=\sum_{m=0}^{L-1}h[m]x[n-m]$。`d_taps/taps_cpu→h[m]`，`d_echo/state.echo→x[n]`，`filtered/data.filtered_cpu→y[n]`，输入输出为一维 `float`，输出长度按本次算子接口/options 保持任务样本长度。
    //
    //  【为什么这样设计】
    //  FIR 是有限加权和，当前 resident GPU 调用避免算子内部重复 H2D/D2H；内部边界与卷积实现归 `Learning/operators/filtering/firfilter/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::DeviceArray<float> filtered(config.samples);
    evidence.operator_ms["firfilter"] = time_gpu([&] {
        cusignal::firfilter_fp32_resident_device(d_taps, d_echo, filtered);
    });
    // <学习注释：语义块：filtered：完成一个连续的数据处理动作。
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
    // <学习注释：语义块：filtered：形成并返回当前结果。
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
    // <学习注释：语义块：filtered：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `state.filtered = filtered.to_host();` 是赋值：先求得右侧完整表达式 `filtered.to_host()`，再把结果写入左侧可修改对象
    //  `state.filtered`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与
    //  `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    begin = Clock::now();
    state.filtered = filtered.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    // <学习注释：语义块：filtered：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.metrics["resident_internal_h2d_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象
    //  `evidence.metrics["resident_internal_h2d_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.metrics["resident_internal_d2h_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象
    //  `evidence.metrics["resident_internal_d2h_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.metrics["resident_internal_h2d_count"] = 0.0;
    evidence.metrics["resident_internal_d2h_count"] = 0.0;
    // <学习注释：语义块：filtered：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.metrics["resident_internal_allocation_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象
    //  `evidence.metrics["resident_internal_allocation_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.metrics["resident_kernel_count"] = 3.0;` 是赋值：先求得右侧完整表达式 `3.0`，再把结果写入左侧可修改对象
    //  `evidence.metrics["resident_kernel_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.metrics["resident_persistent_device_bytes"] = static_cast<double>(config.samples * sizeof(float)
    //  + sizeof(float) + taps_count * sizeof(float) * 3);` 声明 `sizeof`，其静态类型是
    //  `evidence.metrics["resident_persistent_device_bytes"] = static_cast<double>(config.samples *`，并用 `float) +
    //  sizeof(float) + taps_count * sizeof(float) * 3` 直接初始化/调用该类型的构造函数；这不是调用名为 `sizeof` 的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  3.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //  3：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
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
    evidence.metrics["resident_internal_allocation_count"] = 0.0;
    evidence.metrics["resident_kernel_count"] = 3.0;
    evidence.metrics["resident_persistent_device_bytes"] =
        static_cast<double>(config.samples * sizeof(float)
            + sizeof(float)
            + taps_count * sizeof(float) * 3);

    // <学习注释：语义块：filtered：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `if (visualization_capture_enabled()) state.filter_taps = d_taps.to_host();` 是 `if` 条件语句：先把完整条件
    //  `visualization_capture_enabled()` 求值并转换为布尔值。 条件为真时只执行其直接子语句 `state.filter_taps =
    //  d_taps.to_host();`；为假时跳过该子语句。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；H2D/D2H 或设备数组构造只改变驻留位置/所有权，不应改变元素值、dtype、逻辑 shape 或索引意义。
    //
    //  【为什么这样设计】
    //  显式设备数组和传输边界便于区分 Host/GPU 生命周期并分别计时；RAII 在作用域结束时回收设备资源。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.filter_taps = d_taps.to_host();
    // <学习注释：语义块：filtered：形成并返回当前结果。
    //
    //  【执行方法】
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
    return evidence;
}

// <学习注释：语义块：filtered：完成一个连续的数据处理动作。
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

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`：`collect_step2_validation`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step2ValidationData collect_step2_validation(const PipelineState& state) {` 中的 `collect_step2_validation`
//  是函数定义；名称前的 `Step2ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾
//  `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `Step2ValidationData data;` 是对象声明：类型 `Step2ValidationData` 应用于名称
//  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  collect_step2_validation：测试、收集或契约检查 helper；无独立数学变量；内部指标分别映射到误差/条件公式；消费运行结果并形成证据或失败路径
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
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
Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data;
    // <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const int taps_count = state.config.filter_taps;` 是声明并初始化：`const int taps_count` 建立局部对象
    //  `taps_count`，右侧完整表达式 `state.config.filter_taps` 产生初值。
    //  `const std::vector<float> cutoff{state.config.filter_cutoff_hz};` 声明 `cutoff`，静态类型为 `const
    //  std::vector<float>`，并使用列表初始化器 `{state.config.filter_cutoff_hz}` 构造内容；花括号属于初始化，不是新的控制流作用域。
    //  `const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);` 是声明并初始化：`const auto
    //  window_cpu` 建立局部对象 `window_cpu`，右侧完整表达式 `cusignal::hamming_typed_cpu<float>(taps_count, true)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  taps_count：当前一维缓冲区总元素数；记作 $N$；用于 kernel 越界保护和分配规模
    //  cutoff：当前表达式读取或传递的工程名称 cutoff；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const std::vector<float>
    //  cutoff{state.config.filter_cutoff_hz};；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp
    //  当前作用域中产生或消费
    //  window_cpu：CPU reference 对应量；与去掉 _cpu 后的任务量采用同一数学定义；供 GPU/CPU 精度 comparison 使用
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  正式算子生成 $L=\texttt{taps_count}$ 点对称 Hamming 窗 $w[n]=0.54-0.46\cos(2\pi n/(L-1))$；最后一个布尔实参为 `true`，按算子接口选择对称窗分支。GPU 输出 `d_window<float>`，CPU 输出 Host double/模板结果，随后作为 FIR 设计的显式窗。
    //
    //  【为什么这样设计】
    //  任务层绑定长度与对称模式；窗函数内部实现归 `Learning/operators/windows/hamming/`。Hamming 用较宽主瓣换取较低旁瓣。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const int taps_count = state.config.filter_taps;
    const std::vector<float> cutoff{state.config.filter_cutoff_hz};
    const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);
    // <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::FirwinOptions design_options;` 是对象声明：类型 `cusignal::FirwinOptions` 应用于名称
    //  `design_options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;` 是赋值：先求得右侧完整表达式
    //  `cusignal::FirwinWindowMode::explicit_values`，再把结果写入左侧可修改对象 `design_options.window_mode`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `design_options.window = window_cpu;` 是赋值：先求得右侧完整表达式 `window_cpu`，再把结果写入左侧可修改对象
    //  `design_options.window`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  design_options：当前表达式读取或传递的工程名称 design_options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::FirwinOptions
    //  design_options;；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp 当前作用域中产生或消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::FirwinOptions design_options;
    design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;
    design_options.window = window_cpu;
    // <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;` 是赋值：先求得右侧完整表达式
    //  `cusignal::FirwinPassZero::boolean_true`，再把结果写入左侧可修改对象 `design_options.pass_zero`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `design_options.scale = true;` 是赋值：先求得右侧完整表达式 `true`，再把结果写入左侧可修改对象 `design_options.scale`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `design_options.fs = state.config.sample_rate_hz;` 是赋值：先求得右侧完整表达式
    //  `state.config.sample_rate_hz`，再把结果写入左侧可修改对象 `design_options.fs`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;
    design_options.scale = true;
    design_options.fs = state.config.sample_rate_hz;
    // <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
    //  cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));` 是声明并初始化：`const auto taps_cpu` 建立局部对象
    //  `taps_cpu`，右侧完整表达式 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
    //  cusignal::firwin_typed_cpu(taps_count, cutoff, design_options))` 产生初值。
    //  `cusignal::FirfilterOptions filter_options;` 是对象声明：类型 `cusignal::FirfilterOptions` 应用于名称
    //  `filter_options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //
    //  【变量—数学符号—数据流】
    //  taps_cpu：CPU reference 对应量；与去掉 _cpu 后的任务量采用同一数学定义；供 GPU/CPU 精度 comparison 使用
    //  filter_options：当前表达式读取或传递的工程名称 filter_options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据
    //  cusignal::FirfilterOptions filter_options;；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp 当前作用域中产生或消费
    //
    //  【数学/物理公式对照】
    //  正式算子以 `taps_count=L`、`cutoff/filter_cutoff_hz=f_c`、`fs=sample_rate_hz=f_s` 和显式 Hamming 窗设计低通 FIR。理想核为以中心 $m=(L-1)/2$ 平移的 sinc，实际 $h[n]=h_d[n]w[n]$；末尾 `true`/CPU options 选择缩放分支。输出 taps 长度为 $L$。
    //
    //  【为什么这样设计】
    //  显式传窗保证窗口选择可审计；设计内部的归一化和偶/奇长度边界归 `Learning/operators/filter_design/firwin/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));
    cusignal::FirfilterOptions filter_options;
    // <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `filter_options.shape = {state.config.samples};` 是赋值：先求得右侧完整表达式 `{state.config.samples}`，再把结果写入左侧可修改对象
    //  `filter_options.shape`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `filter_options.axis = -1;` 是赋值：先求得右侧完整表达式 `-1`，再把结果写入左侧可修改对象 `filter_options.axis`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `data.filtered_cpu = cusignal::firfilter_typed_cpu( taps_cpu, state.echo, filter_options).y;`
    //  是赋值：先求得右侧完整表达式 `cusignal::firfilter_typed_cpu( taps_cpu, state.echo, filter_options).y`，再把结果写入左侧可修改对象
    //  `data.filtered_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。；入口加载，runner 和各 step 只读消费
    //  echo：目标回波、干扰与噪声叠加后的复数组；记作 $x[p,n]=x_0[p,n]+w[p,n]$；Step1 输出，后续压缩/滤波消费
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  正式算子执行 $y[n]=\sum_{m=0}^{L-1}h[m]x[n-m]$。`d_taps/taps_cpu→h[m]`，`d_echo/state.echo→x[n]`，`filtered/data.filtered_cpu→y[n]`，输入输出为一维 `float`，输出长度按本次算子接口/options 保持任务样本长度。
    //
    //  【为什么这样设计】
    //  FIR 是有限加权和，当前 resident GPU 调用避免算子内部重复 H2D/D2H；内部边界与卷积实现归 `Learning/operators/filtering/firfilter/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    filter_options.shape = {state.config.samples};
    filter_options.axis = -1;
    data.filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.echo, filter_options).y;
    // <学习注释：语义块：collect_step2_validation：形成并返回当前结果。
    //
    //  【执行方法】
    //  `data.target_filtered_cpu = cusignal::firfilter_typed_cpu( taps_cpu, state.target_component,
    //  filter_options).y;` 是赋值：先求得右侧完整表达式 `cusignal::firfilter_typed_cpu( taps_cpu, state.target_component,
    //  filter_options).y`，再把结果写入左侧可修改对象 `data.target_filtered_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
    //
    //  【数学/物理公式对照】
    //  正式算子执行 $y[n]=\sum_{m=0}^{L-1}h[m]x[n-m]$。`d_taps/taps_cpu→h[m]`，`d_echo/state.echo→x[n]`，`filtered/data.filtered_cpu→y[n]`，输入输出为一维 `float`，输出长度按本次算子接口/options 保持任务样本长度。
    //
    //  【为什么这样设计】
    //  FIR 是有限加权和，当前 resident GPU 调用避免算子内部重复 H2D/D2H；内部边界与卷积实现归 `Learning/operators/filtering/firfilter/`。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    data.target_filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.target_component, filter_options).y;
    return data;
// <学习注释：语义块：collect_step2_validation：完成一个连续的数据处理动作。
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

`Step1 echo→Hamming→firwin→firfilter→D2H`。去除噪声并输出 filtered 给 Step3。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`run_step2`|生成 Hamming 窗和 FIR taps，调用 firfilter 并回收 filtered|GPU Host wrapper/业务入口|
|`collect_step2_validation`|收集 Task2 Step2 GPU 输出与独立 CPU 滤波参考|CPU/GPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `hamming`

- 学习入口：[数学物理原理](../../operators/windows/hamming/数学物理原理.md)、[Python 源码算法](../../operators/windows/hamming/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/windows/hamming/cusignal_cpp_hamming复现逻辑.md)。
- 当前调用采用的数学关系：$w[n]=0.54-0.46\cos(2\pi n/(L-1))$。
- 任务变量与参数实例化：生成显式窗口并交给 firwin。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 三段正式调用的参数与组合公式

|调用|本次实参绑定|数学输出与数据关系|
|---|---|---|
|`hamming_resident_device<float>(L,d_window,true)`|$L=filter_taps$，对称窗|$w[n]=0.54-0.46\cos(2\pi n/(L-1))$，device `float[L]`|
|`firwin_lowpass_explicit_window_resident_device`|`cutoff={filter_cutoff_hz}`、`fs=sample_rate_hz`、显式 `d_window`、缩放 `true`|$h[n]=h_d[n]w[n]$，device `float[L]`|
|`firfilter_fp32_resident_device(d_taps,d_echo,filtered)`|taps $h$、Step1 回波 $x$|$y[n]=\sum_{m=0}^{L-1}h[m]x[n-m]$，`filtered` 为 device/Host `float[samples]`|

CPU reference 使用 `hamming_typed_cpu<float>`、`firwin_typed_cpu`、`firfilter_typed_cpu` 绑定相同 $L,f_c,f_s$ 与边界选项。GPU resident 版本不在算子内部往返 Host；这只改变数据驻留和计时边界，不改变 FIR 公式。

##### `firwin`

- 学习入口：[数学物理原理](../../operators/filter_design/firwin/数学物理原理.md)、[Python 源码算法](../../operators/filter_design/firwin/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/filter_design/firwin/cusignal_cpp_firwin复现逻辑.md)。
- 当前调用采用的数学关系：低通理想冲激响应为 sinc，实际 taps 为 $h[n]=h_d[n]w[n]$ 并按需要归一化。
- 任务变量与参数实例化：`filter_taps→L`、`filter_cutoff_hz/f_s` 决定归一化截止频率、Hamming 为 $w[n]$。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

##### `firfilter`

- 学习入口：[数学物理原理](../../operators/filtering/firfilter/数学物理原理.md)、[Python 源码算法](../../operators/filtering/firfilter/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/filtering/firfilter/cusignal_cpp_firfilter复现逻辑.md)。
- 当前调用采用的数学关系：$y[n]=\sum_{m=0}^{L-1}h[m]x[n-m]$。
- 任务变量与参数实例化：`echo→x`、`taps→h`、`filtered→y`；边界按算子 options 处理。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step2(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明、函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step2`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `StepEvidence run_step2(PipelineState& state);` 中的 `run_step2` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过定义/声明 `run_step2`；调用 `run_step2`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step2/step2.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step2/step2.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `文件级代码`；源码锚点 `#include "step2.h"`。

```cpp
#include "step2.h"

#include "cuda_utils/device_array.h"
#include "filter_design/filter_design_typed.h"
#include "filtering/filtering_typed.h"
#include "windows/windows_typed.h"
```

**语法结构**

该代码块按源码顺序包含 5 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step2`、`h`、`cuda_utils`、`device_array`、`filter_design`、`filter_design_typed`、`filtering`、`filtering_typed`、`windows`、`windows_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step2.h"` 在预处理阶段引入 "step2.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "filter_design/filter_design_typed.h"` 在预处理阶段引入 "filter_design/filter_design_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "filtering/filtering_typed.h"` 在预处理阶段引入 "filtering/filtering_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "windows/windows_typed.h"` 在预处理阶段引入 "windows/windows_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

```cpp
#include <numeric>

namespace task2 {
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`numeric`、`namespace`、`task2`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include <numeric>` 在预处理阶段引入 <numeric> 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `StepEvidence run_step2(PipelineState& state)`。

```cpp
StepEvidence run_step2(PipelineState& state)
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
|`run_step2`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step2(PipelineState& state) {` 中的 `run_step2` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过定义/声明 `run_step2`；调用 `run_step2`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.7 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step2";
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
- `evidence.name = "step2";` 是赋值：先求得右侧完整表达式 `"step2"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `total_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step2` 所在的 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    require_task2_upstream(
        state.echo.size() == static_cast<std::size_t>(config.samples),
        "Step2", "Step1.echo");
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `begin` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `require_task2_upstream( state.echo.size() == static_cast<std::size_t>(config.samples), "Step2", "Step1.echo");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `Clock::now`, `require_task2_upstream`, `size`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.9 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `const std::vector<float> cutoff{config.filter_cutoff_hz};`。

```cpp
    const std::vector<float> cutoff{config.filter_cutoff_hz};
    const int taps_count = config.filter_taps;
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `taps_count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`cutoff`|当前表达式读取或传递的工程名称 `cutoff`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float> cutoff{config.filter_cutoff_hz};`；值在 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 当前作用域中产生或消费|
|`taps_count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const std::vector<float> cutoff{config.filter_cutoff_hz};` 声明 `cutoff`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{config.filter_cutoff_hz}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `const int taps_count = config.filter_taps;` 是声明并初始化：`const int taps_count` 建立局部对象 `taps_count`，右侧完整表达式 `config.filter_taps` 产生初值。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `taps_count`, `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_echo` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);` 是声明并初始化：`auto d_echo` 建立局部对象 `d_echo`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.echo)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`；写入/初始化 `begin`, `d_echo`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.11 run_step2：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `run_step2`；源码锚点 `auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);`。

```cpp
    auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_cutoff` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_cutoff`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);` 是声明并初始化：`auto d_cutoff` 建立局部对象 `d_cutoff`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(cutoff)` 产生初值。
- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`；写入/初始化 `d_cutoff`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 d_window：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `d_window`；源码锚点 `cusignal::DeviceArray<float> d_window(taps_count);`。

```cpp

    cusignal::DeviceArray<float> d_window(taps_count);
    evidence.operator_ms["hamming"] = time_gpu([&] {
        cusignal::hamming_resident_device<float>(taps_count, d_window, true);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_window`、`taps_count`、`evidence`、`operator_ms`、`hamming`、`time_gpu`、`hamming_resident_device`、`true`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_window`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_window(taps_count);` 声明 `d_window`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `taps_count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_window` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["hamming"] = time_gpu([&] { cusignal::hamming_resident_device<float>(taps_count, d_window, true); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["hamming"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|公共基础设施|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `d_window`, `time_gpu`；写入/初始化 `operator_ms["hamming"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 d_taps：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `d_taps`；源码锚点 `cusignal::DeviceArray<float> d_taps(taps_count);`。

```cpp
    cusignal::DeviceArray<float> d_taps(taps_count);
    evidence.operator_ms["firwin"] = time_gpu([&] {
        cusignal::firwin_lowpass_explicit_window_resident_device(
            taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`d_taps`、`taps_count`、`evidence`、`operator_ms`、`firwin`、`time_gpu`、`firwin_lowpass_explicit_window_resident_device`、`d_cutoff`、`d_window`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_taps`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> d_taps(taps_count);` 声明 `d_taps`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `taps_count` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_taps` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["firwin"] = time_gpu([&] { cusignal::firwin_lowpass_explicit_window_resident_device( taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["firwin"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `d_taps`, `time_gpu`, `cusignal::firwin_lowpass_explicit_window_resident_device`；写入/初始化 `operator_ms["firwin"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.14 filtered：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `cusignal::DeviceArray<float> filtered(config.samples);`。

```cpp
    cusignal::DeviceArray<float> filtered(config.samples);
    evidence.operator_ms["firfilter"] = time_gpu([&] {
        cusignal::firfilter_fp32_resident_device(d_taps, d_echo, filtered);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`DeviceArray`、`float`、`filtered`、`config`、`samples`、`evidence`、`operator_ms`、`firfilter`、`time_gpu`、`firfilter_fp32_resident_device`、`d_taps`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`filtered`|当前函数或调用的参数名称，接收调用者绑定的输入 `filtered`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::DeviceArray<float> filtered(config.samples);`；值在 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::DeviceArray<float> filtered(config.samples);` 声明 `filtered`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `config.samples` 直接初始化/调用该类型的构造函数；这不是调用名为 `filtered` 的函数，模板尖括号也不是比较或位移。
- `evidence.operator_ms["firfilter"] = time_gpu([&] { cusignal::firfilter_fp32_resident_device(d_taps, d_echo, filtered); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["firfilter"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `filtered`, `time_gpu`, `cusignal::firfilter_fp32_resident_device`；写入/初始化 `operator_ms["firfilter"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 filtered：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.16 filtered：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `evidence.compute_ms = std::accumulate(`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `compute_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.17 filtered：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    state.filtered = filtered.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`filtered`、`to_host`、`evidence`、`d2h_ms`、`milliseconds`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.filtered = filtered.to_host();` 是赋值：先求得右侧完整表达式 `filtered.to_host()`，再把结果写入左侧可修改对象 `state.filtered`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`, `milliseconds`；写入/初始化 `begin`, `filtered`, `evidence.d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 filtered：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `evidence.metrics["resident_internal_h2d_count"] = 0.0;`。

```cpp

    evidence.metrics["resident_internal_h2d_count"] = 0.0;
    evidence.metrics["resident_internal_d2h_count"] = 0.0;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
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

- `evidence.metrics["resident_internal_h2d_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.metrics["resident_internal_h2d_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.metrics["resident_internal_d2h_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.metrics["resident_internal_d2h_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过写入/初始化 `metrics["resident_internal_h2d_count"]`, `metrics["resident_internal_d2h_count"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.19 filtered：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `evidence.metrics["resident_internal_allocation_count"] = 0.0;`。

```cpp
    evidence.metrics["resident_internal_allocation_count"] = 0.0;
    evidence.metrics["resident_kernel_count"] = 3.0;
    evidence.metrics["resident_persistent_device_bytes"] =
        static_cast<double>(config.samples * sizeof(float)
            + sizeof(float)
            + taps_count * sizeof(float) * 3);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `3` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|
|`3.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`3`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `evidence.metrics["resident_internal_allocation_count"] = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `evidence.metrics["resident_internal_allocation_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.metrics["resident_kernel_count"] = 3.0;` 是赋值：先求得右侧完整表达式 `3.0`，再把结果写入左侧可修改对象 `evidence.metrics["resident_kernel_count"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.metrics["resident_persistent_device_bytes"] = static_cast<double>(config.samples * sizeof(float) + sizeof(float) + taps_count * sizeof(float) * 3);` 声明 `sizeof`，其静态类型是 `evidence.metrics["resident_persistent_device_bytes"] = static_cast<double>(config.samples *`，并用 `float) + sizeof(float) + taps_count * sizeof(float) * 3` 直接初始化/调用该类型的构造函数；这不是调用名为 `sizeof` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过写入/初始化 `metrics["resident_internal_allocation_count"]`, `metrics["resident_kernel_count"]`, `metrics["resident_persistent_device_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 filtered：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`。

```cpp

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.filter_taps = d_taps.to_host();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`total_ms`、`milliseconds`、`total_begin`、`Clock`、`now`、`visualization_capture_enabled`、`state`、`filter_taps`、`d_taps`、`to_host`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `if (visualization_capture_enabled()) state.filter_taps = d_taps.to_host();` 是 `if` 条件语句：先把完整条件 `visualization_capture_enabled()` 求值并转换为布尔值。 条件为真时只执行其直接子语句 `state.filter_taps = d_taps.to_host();`；为假时跳过该子语句。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过检查 `visualization_capture_enabled()`；调用 `milliseconds`, `Clock::now`, `visualization_capture_enabled`, `to_host`；写入/初始化 `total_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于可视化数据链：它读取/导出数值结果，建立坐标、单位和图形对象；图片是解释性派生物，不能替代正式数值门禁。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.21 filtered：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `return evidence;`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.22 filtered：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step2/step2.cu`；符号 `filtered`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.23 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `Step2ValidationData collect_step2_validation(const PipelineState& state)`。

```cpp
Step2ValidationData collect_step2_validation(const PipelineState& state)
{
    Step2ValidationData data;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `data` 由 `Step2ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step2_validation`|测试、收集或契约检查 helper|无独立数学变量；内部指标分别映射到误差/条件公式|消费运行结果并形成证据或失败路径|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step2ValidationData collect_step2_validation(const PipelineState& state) {` 中的 `collect_step2_validation` 是函数定义；名称前的 `Step2ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step2ValidationData data;` 是对象声明：类型 `Step2ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过定义/声明 `collect_step2_validation`；调用 `collect_step2_validation`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `const int taps_count = state.config.filter_taps;`。

```cpp
    const int taps_count = state.config.filter_taps;
    const std::vector<float> cutoff{state.config.filter_cutoff_hz};
    const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `taps_count` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `window_cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`taps_count`|当前一维缓冲区总元素数|记作 $N$|用于 kernel 越界保护和分配规模|
|`cutoff`|当前表达式读取或传递的工程名称 `cutoff`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const std::vector<float> cutoff{state.config.filter_cutoff_hz};`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp` 当前作用域中产生或消费|
|`window_cpu`|CPU reference 对应量|与去掉 `_cpu` 后的任务量采用同一数学定义|供 GPU/CPU 精度 comparison 使用|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const int taps_count = state.config.filter_taps;` 是声明并初始化：`const int taps_count` 建立局部对象 `taps_count`，右侧完整表达式 `state.config.filter_taps` 产生初值。
- `const std::vector<float> cutoff{state.config.filter_cutoff_hz};` 声明 `cutoff`，静态类型为 `const std::vector<float>`，并使用列表初始化器 `{state.config.filter_cutoff_hz}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `const auto window_cpu = cusignal::hamming_typed_cpu<float>(taps_count, true);` 是声明并初始化：`const auto window_cpu` 建立局部对象 `window_cpu`，右侧完整表达式 `cusignal::hamming_typed_cpu<float>(taps_count, true)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过写入/初始化 `taps_count`, `window_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.25 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `cusignal::FirwinOptions design_options;`。

```cpp
    cusignal::FirwinOptions design_options;
    design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;
    design_options.window = window_cpu;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `design_options` 由 `cusignal::FirwinOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`design_options`|当前表达式读取或传递的工程名称 `design_options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FirwinOptions design_options;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::FirwinOptions design_options;` 是对象声明：类型 `cusignal::FirwinOptions` 应用于名称 `design_options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `design_options.window_mode = cusignal::FirwinWindowMode::explicit_values;` 是赋值：先求得右侧完整表达式 `cusignal::FirwinWindowMode::explicit_values`，再把结果写入左侧可修改对象 `design_options.window_mode`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `design_options.window = window_cpu;` 是赋值：先求得右侧完整表达式 `window_cpu`，再把结果写入左侧可修改对象 `design_options.window`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过写入/初始化 `window_mode`, `window`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.26 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;`。

```cpp
    design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;
    design_options.scale = true;
    design_options.fs = state.config.sample_rate_hz;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `design_options`、`pass_zero`、`cusignal`、`FirwinPassZero`、`boolean_true`、`scale`、`true`、`fs`、`state`、`config`、`sample_rate_hz`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `design_options.pass_zero = cusignal::FirwinPassZero::boolean_true;` 是赋值：先求得右侧完整表达式 `cusignal::FirwinPassZero::boolean_true`，再把结果写入左侧可修改对象 `design_options.pass_zero`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `design_options.scale = true;` 是赋值：先求得右侧完整表达式 `true`，再把结果写入左侧可修改对象 `design_options.scale`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `design_options.fs = state.config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `state.config.sample_rate_hz`，再把结果写入左侧可修改对象 `design_options.fs`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过写入/初始化 `pass_zero`, `scale`, `fs`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.27 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(`。

```cpp
    const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host(
        cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));
    cusignal::FirfilterOptions filter_options;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `taps_cpu` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；
- `filter_options` 由 `cusignal::FirfilterOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`taps_cpu`|CPU reference 对应量|与去掉 `_cpu` 后的任务量采用同一数学定义|供 GPU/CPU 精度 comparison 使用|
|`filter_options`|当前表达式读取或传递的工程名称 `filter_options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FirfilterOptions filter_options;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const auto taps_cpu = cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::firwin_typed_cpu(taps_count, cutoff, design_options));` 是声明并初始化：`const auto taps_cpu` 建立局部对象 `taps_cpu`，右侧完整表达式 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host( cusignal::firwin_typed_cpu(taps_count, cutoff, design_options))` 产生初值。
- `cusignal::FirfilterOptions filter_options;` 是对象声明：类型 `cusignal::FirfilterOptions` 应用于名称 `filter_options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `cusignal::cuda_utils::narrow_fp64_to_fp32_on_host`, `cusignal::firwin_typed_cpu`；写入/初始化 `taps_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.28 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `filter_options.shape = {state.config.samples};`。

```cpp
    filter_options.shape = {state.config.samples};
    filter_options.axis = -1;
    data.filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.echo, filter_options).y;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化 FIR taps 数、截止频率、采样率、Hamming 窗和滤波边界选项。|入口加载，runner 和各 step 只读消费|
|`echo`|目标回波、干扰与噪声叠加后的复数组|记作 $x[p,n]=x_0[p,n]+w[p,n]$|Step1 输出，后续压缩/滤波消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `filter_options.shape = {state.config.samples};` 是赋值：先求得右侧完整表达式 `{state.config.samples}`，再把结果写入左侧可修改对象 `filter_options.shape`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `filter_options.axis = -1;` 是赋值：先求得右侧完整表达式 `-1`，再把结果写入左侧可修改对象 `filter_options.axis`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.filtered_cpu = cusignal::firfilter_typed_cpu( taps_cpu, state.echo, filter_options).y;` 是赋值：先求得右侧完整表达式 `cusignal::firfilter_typed_cpu( taps_cpu, state.echo, filter_options).y`，再把结果写入左侧可修改对象 `data.filtered_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `cusignal::firfilter_typed_cpu`；写入/初始化 `shape`, `axis`, `filtered_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.29 collect_step2_validation：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `data.target_filtered_cpu = cusignal::firfilter_typed_cpu(`。

```cpp
    data.target_filtered_cpu = cusignal::firfilter_typed_cpu(
        taps_cpu, state.target_component, filter_options).y;
    return data;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `data` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.target_filtered_cpu = cusignal::firfilter_typed_cpu( taps_cpu, state.target_component, filter_options).y;` 是赋值：先求得右侧完整表达式 `cusignal::firfilter_typed_cpu( taps_cpu, state.target_component, filter_options).y`，再把结果写入左侧可修改对象 `data.target_filtered_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块通过调用 `cusignal::firfilter_typed_cpu`；写入/初始化 `target_filtered_cpu`；返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.30 collect_step2_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step2_validation.cpp`；符号 `collect_step2_validation`；源码锚点 `}`。

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
|赛题条款|T2-R2：滤波：选择窗口、滤波器设计与 filtering 算子去除特征噪声|
|业务角色|测试证据|
|业务输入|T2-R1 的 `echo`、窗口、FIR taps 与截止频率|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|滤波后序列 `filtered`|
|下一消费者|交给 T2-R3 B 样条平滑|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
