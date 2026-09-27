# Task2 Step3 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task2 Step3 的裁剪、三次 B 样条权重生成与归一化、FIR 平滑正式调用，以及同一数学路径的 CPU reference。

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
|`ZKX/Task2/task2_gpu_cpu/step3/step3.h`|`run_step3` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step3/step3.cu`|`run_step3`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`|`collect_step3_validation`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`

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
//  `StepEvidence run_step3(PipelineState& state);` 中的 `run_step3` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。
//
//  【变量—数学符号—数据流】
//  run_step3：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
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

StepEvidence run_step3(PipelineState& state);

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

### 3.2 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step3.h"` 在预处理阶段引入 "step3.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "bsplines/bsplines_typed.h"` 在预处理阶段引入 "bsplines/bsplines_typed.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "filtering/filtering_typed.h"` 在预处理阶段引入 "filtering/filtering_typed.h"
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
#include "step3.h"

#include "bsplines/bsplines_typed.h"
#include "cuda_utils/device_array.h"
#include "filtering/filtering_typed.h"

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

// <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `StepEvidence run_step3(PipelineState& state) {` 中的 `run_step3` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  run_step3：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
StepEvidence run_step3(PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.name = "step3";` 是赋值：先求得右侧完整表达式 `"step3"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
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
    evidence.name = "step3";
    const auto total_begin = Clock::now();
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //  `require_task2_upstream( state.filtered.size() == static_cast<std::size_t>(config.samples), "Step3",
    //  "Step2.filtered");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。；入口加载，runner 和各 step 只读消费
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
        state.filtered.size() == static_cast<std::size_t>(config.samples),
        "Step3", "Step2.filtered");
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `require_task2_upstream( state.target_waveform.size() == static_cast<std::size_t>(config.samples),
    //  "Step3", "Step1.target_waveform");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    require_task2_upstream(
        state.target_waveform.size() == static_cast<std::size_t>(config.samples),
        "Step3", "Step1.target_waveform");
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const int crop_begin = config.step3_crop_each;` 是声明并初始化：`const int crop_begin` 建立局部对象
    //  `crop_begin`，右侧完整表达式 `config.step3_crop_each` 产生初值。
    //  `const int crop_end = config.samples - config.step3_crop_each;` 是声明并初始化：`const int crop_end` 建立局部对象
    //  `crop_end`，右侧完整表达式 `config.samples - config.step3_crop_each` 产生初值。
    //  `std::vector<float> cropped(state.filtered.begin() + crop_begin, state.filtered.begin() + crop_end);` 声明
    //  `cropped`，其静态类型是 `std::vector<float>`，并用 `state.filtered.begin() + crop_begin, state.filtered.begin() +
    //  crop_end` 直接初始化/调用该类型的构造函数；这不是调用名为 `cropped` 的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  crop_begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  crop_end：计时或区间的终点状态；无独立算法符号；可记为 $t_1$ 或右边界；与起点相减/配对形成耗时或区间
    //  config：外部配置对象；`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。；入口加载，runner 和各 step 只读消费
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
    const int crop_begin = config.step3_crop_each;
    const int crop_end = config.samples - config.step3_crop_each;
    std::vector<float> cropped(state.filtered.begin() + crop_begin,
                               state.filtered.begin() + crop_end);
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F};` 声明 `offsets`，静态类型为
    //  `std::vector<float>`，并使用列表初始化器 `{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F}`
    //  构造内容；花括号属于初始化，不是新的控制流作用域。
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  offsets：相对起点的离散偏移；记作 $\Delta n$；参与索引换算或数据切片
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  2.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  1.5F：F 使浮点字面量为 float；不带后缀默认是 double；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //  1.0F：F 使浮点字面量为 float；不带后缀默认是 double；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //  0.5F：F 使浮点字面量为 float；不带后缀默认是 double；表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。
    //  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    evidence.prep_ms = milliseconds(begin, Clock::now());

    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);` 是声明并初始化：`auto d_cropped` 建立局部对象
    //  `d_cropped`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(cropped)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  d_cropped：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step3/step3.cu 的 GPU 调用链
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);` 是声明并初始化：`auto d_offsets` 建立局部对象
    //  `d_offsets`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(offsets)` 产生初值。
    //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `cusignal::DeviceArray<float> d_spline(offsets.size());` 声明 `d_spline`，其静态类型是
    //  `cusignal::DeviceArray<float>`，并用 `offsets.size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_spline` 的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  d_offsets：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step3/step3.cu 的 GPU 调用链
    //  d_spline：GPU device 缓冲区或 device 对象；与去掉 d_ 后的 Host 量数学含义相同；由 H2D/设备计算产生；作用域位于
    //  ZKX/Task2/task2_gpu_cpu/step3/step3.cu 的 GPU 调用链
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
    auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<float> d_spline(offsets.size());
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.operator_ms["cubic"] = time_gpu([&] { cusignal::cubic_device(d_offsets, d_spline); });`
    //  先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["cubic"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子在 `offsets={-2,-1.5,…,2}` 上计算三次 B 样条基函数 $B_3(x)$，输出 `d_spline/weights_cpu` 与 offsets 等长。随后 `weight_sum=Σ_m b[m]`，`weights[m]/=weight_sum` 形成单位直流增益 $\tilde b[m]=b[m]/Σb$。
    //
    //  【为什么这样设计】
    //  三次 B 样条具有局部支撑和平滑性；归一化避免平滑改变常量信号幅值。算子内部逐点公式归 `Learning/operators/bsplines/cubic/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.operator_ms["cubic"] = time_gpu([&] {
        cusignal::cubic_device(d_offsets, d_spline);
    });
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `auto weights = d_spline.to_host();` 是声明并初始化：`auto weights` 建立局部对象 `weights`，右侧完整表达式 `d_spline.to_host()`
    //  产生初值。
    //  `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与
    //  `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。
    //
    //  【变量—数学符号—数据流】
    //  weights：当前样本或特征的融合权重；记作 $w_i$；与对应特征/坐标相乘后进入加权和
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    auto weights = d_spline.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    // <学习注释：语义块：run_step3：迭代处理数据范围。
    //
    //  【执行方法】
    //  `const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);` 调用
    //  `std::accumulate`，从源码给出的初值开始遍历 `[begin, end)` 区间并累加元素；返回的总和用于初始化/写入左侧对象。
    //  `require(weight_sum > 0.0F, "step3 degenerate cubic window");`
    //  调用本文件的前置条件检查；第一个实参为真时继续，为假时按第二个实参给出的消息报告配置或数值退化错误。
    //  `for (float& value : weights) value /= weight_sum;` 是范围 `for`：从 `weights` 依次取得元素并绑定到 `float&
    //  value`，随后对每个元素执行循环体；声明中的 `&` 若存在表示引用，修改循环变量会修改原元素。 循环体 `value /= weight_sum;` 等价于 `value = value /
    //  (weight_sum)`；当前代码因此逐元素除以同一归一化因子。
    //
    //  【变量—数学符号—数据流】
    //  weight_sum：当前样本或特征的融合权重；记作 $w_i$；与对应特征/坐标相乘后进入加权和
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //
    //  【数字常量】
    //  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  `weight_sum=Σ_m weights[m]`；正值检查保证分母非零，循环执行 $weights[m]←weights[m]/weight_sum$，使 $Σ_m weights[m]=1$。
    //
    //  【为什么这样设计】
    //  单位和权重使 FIR 对 DC 的增益为 1；若不检查分母，退化窗口会产生除零和 NaN。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
    //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
    const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
    require(weight_sum > 0.0F, "step3 degenerate cubic window");
    for (float& value : weights) value /= weight_sum;
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `auto d_weights = cusignal::DeviceArray<float>::from_host(weights);` 是声明并初始化：`auto d_weights` 建立局部对象
    //  `d_weights`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(weights)` 产生初值。
    //  `evidence.h2d_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.h2d_ms` 的旧值，与
    //  `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。
    //
    //  【变量—数学符号—数据流】
    //  d_weights：当前样本或特征的融合权重；记作 $w_i$；与对应特征/坐标相乘后进入加权和
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    auto d_weights = cusignal::DeviceArray<float>::from_host(weights);
    evidence.h2d_ms += milliseconds(begin, Clock::now());
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::FirfilterOptions options;` 是对象声明：类型 `cusignal::FirfilterOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `options.shape = {static_cast<int>(cropped.size())};` 是赋值：先求得右侧完整表达式
    //  `{static_cast<int>(cropped.size())}`，再把结果写入左侧可修改对象 `options.shape`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `options.axis = -1;` 是赋值：先求得右侧完整表达式 `-1`，再把结果写入左侧可修改对象 `options.axis`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::FirfilterOptions options;；值在
    //  ZKX/Task2/task2_gpu_cpu/step3/step3.cu 当前作用域中产生或消费
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(cropped.size())};
    options.axis = -1;
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::FirfilterDeviceResult smoothed;` 是对象声明：类型 `cusignal::FirfilterDeviceResult` 应用于名称
    //  `smoothed`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] { cusignal::firfilter_device(d_weights,
    //  d_cropped, smoothed, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal
    //  算子，返回的毫秒值写入 `evidence.operator_ms["firfilter_b_spline"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  smoothed：当前函数或调用的参数名称，接收调用者绑定的输入 smoothed；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据
    //  cusignal::FirfilterDeviceResult smoothed;；值在 ZKX/Task2/task2_gpu_cpu/step3/step3.cu 当前作用域中产生或消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子用归一化样条权重平滑裁剪信号：$y[n]=Σ_m\tilde b[m]x[n-m]$。`cropped/d_cropped→x`，`weights/d_weights→\tilde b`，`smoothed.y/data.smoothed_cpu→y`；`axis=-1` 指最后/唯一轴，shape 为 `{cropped.size()}`。
    //
    //  【为什么这样设计】
    //  把样条样值作为 FIR taps 可落实局部加权平滑；卷积边界归 `Learning/operators/filtering/firfilter/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::FirfilterDeviceResult smoothed;
    evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] {
        cusignal::firfilter_device(d_weights, d_cropped, smoothed, options);
    });
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
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
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.compute_ms = evidence.operator_ms["cubic"] + evidence.operator_ms["firfilter_b_spline"];`
    //  是赋值：先求得右侧完整表达式 `evidence.operator_ms["cubic"] + evidence.operator_ms["firfilter_b_spline"]`，再把结果写入左侧可修改对象
    //  `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //
    //  【变量—数学符号—数据流】
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
    evidence.compute_ms = evidence.operator_ms["cubic"] +
        evidence.operator_ms["firfilter_b_spline"];
    begin = Clock::now();
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.smoothed = smoothed.y.to_host();` 是赋值：先求得右侧完整表达式 `smoothed.y.to_host()`，再把结果写入左侧可修改对象
    //  `state.smoothed`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与
    //  `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。
    //
    //  【变量—数学符号—数据流】
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
    state.smoothed = smoothed.y.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `state.reference_for_correlation.assign( state.target_waveform.begin() + crop_begin,
    //  state.target_waveform.begin() + crop_end);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    begin = Clock::now();
    state.reference_for_correlation.assign(
        state.target_waveform.begin() + crop_begin,
        state.target_waveform.begin() + crop_end);
    // <学习注释：语义块：run_step3：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `evidence.post_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.post_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `if (visualization_capture_enabled()) state.spline_weights = weights;` 是 `if` 条件语句：先把完整条件
    //  `visualization_capture_enabled()` 求值并转换为布尔值。 条件为真时只执行其直接子语句 `state.spline_weights = weights;`；为假时跳过该子语句。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.spline_weights = weights;
    // <学习注释：语义块：run_step3：形成并返回当前结果。
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

// <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
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

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`：`collect_step3_validation`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：collect_step3_validation：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step3ValidationData collect_step3_validation(const PipelineState& state) {` 中的 `collect_step3_validation`
//  是函数定义；名称前的 `Step3ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾
//  `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `Step3ValidationData data;` 是对象声明：类型 `Step3ValidationData` 应用于名称
//  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  collect_step3_validation：测试、收集或契约检查 helper；无独立数学变量；内部指标分别映射到误差/条件公式；消费运行结果并形成证据或失败路径
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
Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data;
    // <学习注释：语义块：collect_step3_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.cropped_input.assign( state.filtered.begin() + state.config.step3_crop_each, state.filtered.end() -
    //  state.config.step3_crop_each);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.cropped_input.assign(
        state.filtered.begin() + state.config.step3_crop_each,
        state.filtered.end() - state.config.step3_crop_each);
    // <学习注释：语义块：collect_step3_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.reference_for_correlation_cpu.assign( state.target_waveform.begin() + state.config.step3_crop_each,
    //  state.target_waveform.end() - state.config.step3_crop_each);` 通过对象/指针调用容器的
    //  `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
    //
    //  【为什么这样设计】
    //  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.reference_for_correlation_cpu.assign(
        state.target_waveform.begin() + state.config.step3_crop_each,
        state.target_waveform.end() - state.config.step3_crop_each);
    // <学习注释：语义块：collect_step3_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F};` 声明 `offsets`，静态类型为
    //  `std::vector<float>`，并使用列表初始化器 `{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F}`
    //  构造内容；花括号属于初始化，不是新的控制流作用域。
    //  `auto weights_cpu = cusignal::cubic_cpu(offsets);` 是声明并初始化：`auto weights_cpu` 建立局部对象 `weights_cpu`，右侧完整表达式
    //  `cusignal::cubic_cpu(offsets)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  offsets：相对起点的离散偏移；记作 $\Delta n$；参与索引换算或数据切片
    //  weights_cpu：当前样本或特征的融合权重；记作 $w_i$；与对应特征/坐标相乘后进入加权和
    //
    //  【数字常量】
    //  2.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //  1.5F：F 使浮点字面量为 float；不带后缀默认是 double；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //  1.0F：F 使浮点字面量为 float；不带后缀默认是 double；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //  0.5F：F 使浮点字面量为 float；不带后缀默认是 double；表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。
    //  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  正式算子在 `offsets={-2,-1.5,…,2}` 上计算三次 B 样条基函数 $B_3(x)$，输出 `d_spline/weights_cpu` 与 offsets 等长。随后 `weight_sum=Σ_m b[m]`，`weights[m]/=weight_sum` 形成单位直流增益 $\tilde b[m]=b[m]/Σb$。
    //
    //  【为什么这样设计】
    //  三次 B 样条具有局部支撑和平滑性；归一化避免平滑改变常量信号幅值。算子内部逐点公式归 `Learning/operators/bsplines/cubic/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    auto weights_cpu = cusignal::cubic_cpu(offsets);
    // <学习注释：语义块：collect_step3_validation：迭代处理数据范围。
    //
    //  【执行方法】
    //  `const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);` 调用
    //  `std::accumulate`，从源码给出的初值开始遍历 `[begin, end)` 区间并累加元素；返回的总和用于初始化/写入左侧对象。
    //  `for (float& value : weights_cpu) value /= sum;` 是范围 `for`：从 `weights_cpu` 依次取得元素并绑定到 `float&
    //  value`，随后对每个元素执行循环体；声明中的 `&` 若存在表示引用，修改循环变量会修改原元素。 循环体 `value /= sum;` 等价于 `value = value /
    //  (sum)`；当前代码因此逐元素除以同一归一化因子。
    //  `cusignal::FirfilterOptions options;` 是对象声明：类型 `cusignal::FirfilterOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //
    //  【变量—数学符号—数据流】
    //  sum：循环或归约的累加器；通常记作 $S=\sum_i a_i$；每轮读取旧值并累加当前 item
    //  value：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::FirfilterOptions options;；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp 当前作用域中产生或消费
    //
    //  【数字常量】
    //  0.0F：F 使浮点字面量为 float；不带后缀默认是 double；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  `weight_sum=Σ_m weights[m]`；正值检查保证分母非零，循环执行 $weights[m]←weights[m]/weight_sum$，使 $Σ_m weights[m]=1$。
    //
    //  【为什么这样设计】
    //  单位和权重使 FIR 对 DC 的增益为 1；若不检查分母，退化窗口会产生除零和 NaN。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
    //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
    const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);
    for (float& value : weights_cpu) value /= sum;
    cusignal::FirfilterOptions options;
    // <学习注释：语义块：collect_step3_validation：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.shape = {static_cast<int>(data.cropped_input.size())};` 是赋值：先求得右侧完整表达式
    //  `{static_cast<int>(data.cropped_input.size())}`，再把结果写入左侧可修改对象 `options.shape`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `options.axis = -1;` 是赋值：先求得右侧完整表达式 `-1`，再把结果写入左侧可修改对象 `options.axis`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `data.smoothed_cpu = cusignal::firfilter_typed_cpu( weights_cpu, data.cropped_input, options).y;`
    //  是赋值：先求得右侧完整表达式 `cusignal::firfilter_typed_cpu( weights_cpu, data.cropped_input, options).y`，再把结果写入左侧可修改对象
    //  `data.smoothed_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  正式算子用归一化样条权重平滑裁剪信号：$y[n]=Σ_m\tilde b[m]x[n-m]$。`cropped/d_cropped→x`，`weights/d_weights→\tilde b`，`smoothed.y/data.smoothed_cpu→y`；`axis=-1` 指最后/唯一轴，shape 为 `{cropped.size()}`。
    //
    //  【为什么这样设计】
    //  把样条样值作为 FIR taps 可落实局部加权平滑；卷积边界归 `Learning/operators/filtering/firfilter/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.shape = {static_cast<int>(data.cropped_input.size())};
    options.axis = -1;
    data.smoothed_cpu = cusignal::firfilter_typed_cpu(
        weights_cpu, data.cropped_input, options).y;
    // <学习注释：语义块：collect_step3_validation：形成并返回当前结果。
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

`Step2 filtered→cubic B 样条权重→firfilter→D2H`。输出 smoothed 给 Step4。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`run_step3`|生成 cubic B 样条权重，归一化后调用 firfilter 平滑|GPU Host wrapper/业务入口|
|`collect_step3_validation`|收集 Task2 Step3 GPU 输出与独立 B 样条平滑参考|CPU/GPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `cubic`

- 学习入口：[数学物理原理](../../operators/bsplines/cubic/数学物理原理.md)、[Python 源码算法](../../operators/bsplines/cubic/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/bsplines/cubic/cusignal_cpp_cubic复现逻辑.md)。
- 当前调用采用的数学关系：三次 B 样条基函数形成局部有限支撑平滑权重 $b[m]$。
- 任务变量与参数实例化：`cubic` 生成平滑 taps；其长度/尺度由本 step 配置实例化。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 样条权重到 FIR 平滑的闭环

`cubic_device(d_offsets,d_spline)` 在固定 offsets `{-2,-1.5,…,2}` 上取三次 B 样条 $b[m]=B_3(x_m)$；Host 端计算
$S=\sum_m b[m]$ 并写回 $\tilde b[m]=b[m]/S$，因此 $\sum_m\tilde b[m]=1$。随后
`firfilter_device(d_weights,d_cropped,smoothed,options)` 计算 $y[n]=\sum_m\tilde b[m]x[n-m]$。

|对象|dtype/shape|数学角色与来源|
|---|---|---|
|`cropped` / `d_cropped`|`float[N_c]`|Step2 `filtered` 去掉两端 `step3_crop_each` 后的 $x[n]$|
|`offsets`|`float[9]`|样条自变量 $x_m$；数值是仓库策略，源码未记录推导依据|
|`weights` / `d_weights`|`float[9]`|归一化 $\tilde b[m]$|
|`options.shape`|`{N_c}`|一维滤波布局|
|`options.axis`|`-1`|最后/唯一轴|
|`smoothed.y`|`float[N_c]`|Step4 的平滑输入|

##### `firfilter`

- 学习入口：[数学物理原理](../../operators/filtering/firfilter/数学物理原理.md)、[Python 源码算法](../../operators/filtering/firfilter/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/filtering/firfilter/cusignal_cpp_firfilter复现逻辑.md)。
- 当前调用采用的数学关系：$y[n]=\sum_m b[m]x[n-m]$。
- 任务变量与参数实例化：`filtered→x`、三次样条权重 `b`、`smoothed→y`；这里把样条权重落实为 FIR 平滑。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step3(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明、函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step3`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `StepEvidence run_step3(PipelineState& state);` 中的 `run_step3` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过定义/声明 `run_step3`；调用 `run_step3`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `文件级代码`；源码锚点 `#include "step3.h"`。

```cpp
#include "step3.h"

#include "bsplines/bsplines_typed.h"
#include "cuda_utils/device_array.h"
#include "filtering/filtering_typed.h"
```

**语法结构**

该代码块按源码顺序包含 4 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step3`、`h`、`bsplines`、`bsplines_typed`、`cuda_utils`、`device_array`、`filtering`、`filtering_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step3.h"` 在预处理阶段引入 "step3.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "bsplines/bsplines_typed.h"` 在预处理阶段引入 "bsplines/bsplines_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "filtering/filtering_typed.h"` 在预处理阶段引入 "filtering/filtering_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `StepEvidence run_step3(PipelineState& state)`。

```cpp
StepEvidence run_step3(PipelineState& state)
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
|`run_step3`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step3(PipelineState& state) {` 中的 `run_step3` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过定义/声明 `run_step3`；调用 `run_step3`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.7 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step3";
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
- `evidence.name = "step3";` 是赋值：先求得右侧完整表达式 `"step3"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `total_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    require_task2_upstream(
        state.filtered.size() == static_cast<std::size_t>(config.samples),
        "Step3", "Step2.filtered");
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
|`config`|外部配置对象|`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `require_task2_upstream( state.filtered.size() == static_cast<std::size_t>(config.samples), "Step3", "Step2.filtered");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`, `require_task2_upstream`, `size`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.9 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `require_task2_upstream(`。

```cpp
    require_task2_upstream(
        state.target_waveform.size() == static_cast<std::size_t>(config.samples),
        "Step3", "Step1.target_waveform");
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `require_task2_upstream`、`state`、`target_waveform`、`size`、`std`、`size_t`、`config`、`samples`、`Step3`、`Step1`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `require_task2_upstream( state.target_waveform.size() == static_cast<std::size_t>(config.samples), "Step3", "Step1.target_waveform");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `require_task2_upstream`, `size`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `const int crop_begin = config.step3_crop_each;`。

```cpp
    const int crop_begin = config.step3_crop_each;
    const int crop_end = config.samples - config.step3_crop_each;
    std::vector<float> cropped(state.filtered.begin() + crop_begin,
                               state.filtered.begin() + crop_end);
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `crop_begin` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象；
- `crop_end` 由 `const int` 声明：有符号整数类型；具体位宽由编译器与目标 ABI 决定，本块没有用 `sizeof(int)` 固定其位宽；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`crop_begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`crop_end`|计时或区间的终点状态|无独立算法符号；可记为 $t_1$ 或右边界|与起点相减/配对形成耗时或区间|
|`config`|外部配置对象|`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `const int crop_begin = config.step3_crop_each;` 是声明并初始化：`const int crop_begin` 建立局部对象 `crop_begin`，右侧完整表达式 `config.step3_crop_each` 产生初值。
- `const int crop_end = config.samples - config.step3_crop_each;` 是声明并初始化：`const int crop_end` 建立局部对象 `crop_end`，右侧完整表达式 `config.samples - config.step3_crop_each` 产生初值。
- `std::vector<float> cropped(state.filtered.begin() + crop_begin, state.filtered.begin() + crop_end);` 声明 `cropped`，其静态类型是 `std::vector<float>`，并用 `state.filtered.begin() + crop_begin, state.filtered.begin() + crop_end` 直接初始化/调用该类型的构造函数；这不是调用名为 `cropped` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|输入准备/数据契约|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `cropped`, `begin`；写入/初始化 `crop_begin`, `crop_end`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.11 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,`。

```cpp
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `2.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.5F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.5F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`offsets`|相对起点的离散偏移|记作 $\Delta n$|参与索引换算或数据切片|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F};` 声明 `offsets`，静态类型为 `std::vector<float>`，并使用列表初始化器 `{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_cropped` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_cropped`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 的 GPU 调用链|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);` 是声明并初始化：`auto d_cropped` 建立局部对象 `d_cropped`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(cropped)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`；写入/初始化 `begin`, `d_cropped`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);`。

```cpp
    auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<float> d_spline(offsets.size());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_offsets` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_offsets`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 的 GPU 调用链|
|`d_spline`|GPU device 缓冲区或 device 对象|与去掉 `d_` 后的 Host 量数学含义相同|由 H2D/设备计算产生；作用域位于 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 的 GPU 调用链|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);` 是声明并初始化：`auto d_offsets` 建立局部对象 `d_offsets`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(offsets)` 产生初值。
- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::DeviceArray<float> d_spline(offsets.size());` 声明 `d_spline`，其静态类型是 `cusignal::DeviceArray<float>`，并用 `offsets.size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `d_spline` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`, `d_spline`, `size`；写入/初始化 `d_offsets`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.14 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.operator_ms["cubic"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["cubic"] = time_gpu([&] {
        cusignal::cubic_device(d_offsets, d_spline);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`cubic`、`time_gpu`、`cusignal`、`cubic_device`、`d_offsets`、`d_spline`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.operator_ms["cubic"] = time_gpu([&] { cusignal::cubic_device(d_offsets, d_spline); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["cubic"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::cubic_device`；写入/初始化 `operator_ms["cubic"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    auto weights = d_spline.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `weights` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`weights`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto weights = d_spline.to_host();` 是声明并初始化：`auto weights` 建立局部对象 `weights`，右侧完整表达式 `d_spline.to_host()` 产生初值。
- `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`, `milliseconds`；写入/初始化 `begin`, `weights`, `evidence.d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.16 run_step3：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);`。

```cpp
    const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
    require(weight_sum > 0.0F, "step3 degenerate cubic window");
    for (float& value : weights) value /= weight_sum;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句、`for` 迭代语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `weight_sum` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`weight_sum`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);` 调用 `std::accumulate`，从源码给出的初值开始遍历 `[begin, end)` 区间并累加元素；返回的总和用于初始化/写入左侧对象。
- `require(weight_sum > 0.0F, "step3 degenerate cubic window");` 调用本文件的前置条件检查；第一个实参为真时继续，为假时按第二个实参给出的消息报告配置或数值退化错误。
- `for (float& value : weights) value /= weight_sum;` 是范围 `for`：从 `weights` 依次取得元素并绑定到 `float& value`，随后对每个元素执行循环体；声明中的 `&` 若存在表示引用，修改循环变量会修改原元素。 循环体 `value /= weight_sum;` 等价于 `value = value / (weight_sum)`；当前代码因此逐元素除以同一归一化因子。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`, `require`；写入/初始化 `weight_sum`, `for (float& value : weights) value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.17 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    auto d_weights = cusignal::DeviceArray<float>::from_host(weights);
    evidence.h2d_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_weights` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_weights`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_weights = cusignal::DeviceArray<float>::from_host(weights);` 是声明并初始化：`auto d_weights` 建立局部对象 `d_weights`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(weights)` 产生初值。
- `evidence.h2d_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.h2d_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|公共基础设施|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`, `milliseconds`；写入/初始化 `begin`, `d_weights`, `evidence.h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `cusignal::FirfilterOptions options;`。

```cpp
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(cropped.size())};
    options.axis = -1;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `options` 由 `cusignal::FirfilterOptions` 声明：类型控制可表示值、可用操作和传参方式；
- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FirfilterOptions options;`；值在 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `cusignal::FirfilterOptions options;` 是对象声明：类型 `cusignal::FirfilterOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `options.shape = {static_cast<int>(cropped.size())};` 是赋值：先求得右侧完整表达式 `{static_cast<int>(cropped.size())}`，再把结果写入左侧可修改对象 `options.shape`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `options.axis = -1;` 是赋值：先求得右侧完整表达式 `-1`，再把结果写入左侧可修改对象 `options.axis`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `size`；写入/初始化 `shape`, `axis`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.19 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `cusignal::FirfilterDeviceResult smoothed;`。

```cpp
    cusignal::FirfilterDeviceResult smoothed;
    evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] {
        cusignal::firfilter_device(d_weights, d_cropped, smoothed, options);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `smoothed` 由 `cusignal::FirfilterDeviceResult` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`smoothed`|当前函数或调用的参数名称，接收调用者绑定的输入 `smoothed`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FirfilterDeviceResult smoothed;`；值在 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 当前作用域中产生或消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::FirfilterDeviceResult smoothed;` 是对象声明：类型 `cusignal::FirfilterDeviceResult` 应用于名称 `smoothed`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] { cusignal::firfilter_device(d_weights, d_cropped, smoothed, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["firfilter_b_spline"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::firfilter_device`；写入/初始化 `operator_ms["firfilter_b_spline"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.21 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.compute_ms = evidence.operator_ms["cubic"] +`。

```cpp
    evidence.compute_ms = evidence.operator_ms["cubic"] +
        evidence.operator_ms["firfilter_b_spline"];
    begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`compute_ms`、`operator_ms`、`cubic`、`firfilter_b_spline`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.compute_ms = evidence.operator_ms["cubic"] + evidence.operator_ms["firfilter_b_spline"];` 是赋值：先求得右侧完整表达式 `evidence.operator_ms["cubic"] + evidence.operator_ms["firfilter_b_spline"]`，再把结果写入左侧可修改对象 `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `compute_ms`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于滤波或平滑阶段：输入样本和窗口/系数共同形成降噪后的序列，输出进入多域特征提取。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `state.smoothed = smoothed.y.to_host();`。

```cpp
    state.smoothed = smoothed.y.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`smoothed`、`y`、`to_host`、`evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.smoothed = smoothed.y.to_host();` 是赋值：先求得右侧完整表达式 `smoothed.y.to_host()`，再把结果写入左侧可修改对象 `state.smoothed`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms += milliseconds(begin, Clock::now());` 是复合赋值：读取 `evidence.d2h_ms` 的旧值，与 `milliseconds(begin, Clock::now())` 执行 `+` 运算，再把结果写回同一对象。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `to_host`, `milliseconds`, `Clock::now`；写入/初始化 `smoothed`, `evidence.d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.23 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp

    begin = Clock::now();
    state.reference_for_correlation.assign(
        state.target_waveform.begin() + crop_begin,
        state.target_waveform.begin() + crop_end);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `begin`、`Clock`、`now`、`state`、`reference_for_correlation`、`assign`、`target_waveform`、`crop_begin`、`crop_end`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.reference_for_correlation.assign( state.target_waveform.begin() + crop_begin, state.target_waveform.begin() + crop_end);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `Clock::now`, `assign`, `begin`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 run_step3：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.post_ms = milliseconds(begin, Clock::now());`。

```cpp
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.spline_weights = weights;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`post_ms`、`milliseconds`、`begin`、`Clock`、`now`、`total_ms`、`total_begin`、`visualization_capture_enabled`、`state`、`spline_weights`、`weights`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.post_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.post_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `if (visualization_capture_enabled()) state.spline_weights = weights;` 是 `if` 条件语句：先把完整条件 `visualization_capture_enabled()` 求值并转换为布尔值。 条件为真时只执行其直接子语句 `state.spline_weights = weights;`；为假时跳过该子语句。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过检查 `visualization_capture_enabled()`；调用 `milliseconds`, `Clock::now`, `visualization_capture_enabled`；写入/初始化 `post_ms`, `total_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块属于可视化数据链：它读取/导出数值结果，建立坐标、单位和图形对象；图片是解释性派生物，不能替代正式数值门禁。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.25 run_step3：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `return evidence;`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.26 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task2/task2_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.27 collect_step3_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `Step3ValidationData collect_step3_validation(const PipelineState& state)`。

```cpp
Step3ValidationData collect_step3_validation(const PipelineState& state)
{
    Step3ValidationData data;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `data` 由 `Step3ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step3_validation`|测试、收集或契约检查 helper|无独立数学变量；内部指标分别映射到误差/条件公式|消费运行结果并形成证据或失败路径|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step3ValidationData collect_step3_validation(const PipelineState& state) {` 中的 `collect_step3_validation` 是函数定义；名称前的 `Step3ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step3ValidationData data;` 是对象声明：类型 `Step3ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过定义/声明 `collect_step3_validation`；调用 `collect_step3_validation`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.28 collect_step3_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `data.cropped_input.assign(`。

```cpp
    data.cropped_input.assign(
        state.filtered.begin() + state.config.step3_crop_each,
        state.filtered.end() - state.config.step3_crop_each);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`cropped_input`、`assign`、`state`、`filtered`、`begin`、`config`、`step3_crop_each`、`end`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.cropped_input.assign( state.filtered.begin() + state.config.step3_crop_each, state.filtered.end() - state.config.step3_crop_each);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `assign`, `begin`, `end`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.29 collect_step3_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `data.reference_for_correlation_cpu.assign(`。

```cpp
    data.reference_for_correlation_cpu.assign(
        state.target_waveform.begin() + state.config.step3_crop_each,
        state.target_waveform.end() - state.config.step3_crop_each);
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`reference_for_correlation_cpu`、`assign`、`state`、`target_waveform`、`begin`、`config`、`step3_crop_each`、`end`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化裁剪宽度、三次 B 样条权重尺度和 FIR 平滑选项。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.reference_for_correlation_cpu.assign( state.target_waveform.begin() + state.config.step3_crop_each, state.target_waveform.end() - state.config.step3_crop_each);` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `assign`, `begin`, `end`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.30 collect_step3_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,`。

```cpp
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    auto weights_cpu = cusignal::cubic_cpu(offsets);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `weights_cpu` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `2.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.5F` 是数值字面量；F 指定 float，而非默认 double；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double；
- `0.5F` 是数值字面量；F 指定 float，而非默认 double；
- `1.0F` 是数值字面量；F 指定 float，而非默认 double；
- `1.5F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`offsets`|相对起点的离散偏移|记作 $\Delta n$|参与索引换算或数据切片|
|`weights_cpu`|当前样本或特征的融合权重|记作 $w_i$|与对应特征/坐标相乘后进入加权和|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|
|`1.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`1.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|
|`0.5F`|`F` 使浮点字面量为 float；不带后缀默认是 double|表示二分之一。若用于时间平移，则把脉冲中心移到零；若用于平均，则形成等权中点。当前具体角色由同一表达式决定。|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F};` 声明 `offsets`，静态类型为 `std::vector<float>`，并使用列表初始化器 `{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F}` 构造内容；花括号属于初始化，不是新的控制流作用域。
- `auto weights_cpu = cusignal::cubic_cpu(offsets);` 是声明并初始化：`auto weights_cpu` 建立局部对象 `weights_cpu`，右侧完整表达式 `cusignal::cubic_cpu(offsets)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `cusignal::cubic_cpu`；写入/初始化 `weights_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.31 collect_step3_validation：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);`。

```cpp
    const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);
    for (float& value : weights_cpu) value /= sum;
    cusignal::FirfilterOptions options;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、`for` 迭代语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `sum` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `options` 由 `cusignal::FirfilterOptions` 声明：类型控制可表示值、可用操作和传参方式；
- `0.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`sum`|循环或归约的累加器|通常记作 $S=\sum_i a_i$|每轮读取旧值并累加当前 item|
|`value`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::FirfilterOptions options;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp` 当前作用域中产生或消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `const float sum = std::accumulate(weights_cpu.begin(), weights_cpu.end(), 0.0F);` 调用 `std::accumulate`，从源码给出的初值开始遍历 `[begin, end)` 区间并累加元素；返回的总和用于初始化/写入左侧对象。
- `for (float& value : weights_cpu) value /= sum;` 是范围 `for`：从 `weights_cpu` 依次取得元素并绑定到 `float& value`，随后对每个元素执行循环体；声明中的 `&` 若存在表示引用，修改循环变量会修改原元素。 循环体 `value /= sum;` 等价于 `value = value / (sum)`；当前代码因此逐元素除以同一归一化因子。
- `cusignal::FirfilterOptions options;` 是对象声明：类型 `cusignal::FirfilterOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `sum`, `for (float& value : weights_cpu) value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.32 collect_step3_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `options.shape = {static_cast<int>(data.cropped_input.size())};`。

```cpp
    options.shape = {static_cast<int>(data.cropped_input.size())};
    options.axis = -1;
    data.smoothed_cpu = cusignal::firfilter_typed_cpu(
        weights_cpu, data.cropped_input, options).y;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `options.shape = {static_cast<int>(data.cropped_input.size())};` 是赋值：先求得右侧完整表达式 `{static_cast<int>(data.cropped_input.size())}`，再把结果写入左侧可修改对象 `options.shape`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `options.axis = -1;` 是赋值：先求得右侧完整表达式 `-1`，再把结果写入左侧可修改对象 `options.axis`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.smoothed_cpu = cusignal::firfilter_typed_cpu( weights_cpu, data.cropped_input, options).y;` 是赋值：先求得右侧完整表达式 `cusignal::firfilter_typed_cpu( weights_cpu, data.cropped_input, options).y`，再把结果写入左侧可修改对象 `data.smoothed_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过调用 `size`, `cusignal::firfilter_typed_cpu`；写入/初始化 `shape`, `axis`, `smoothed_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.33 collect_step3_validation：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step3_validation.cpp`；符号 `collect_step3_validation`；源码锚点 `return data;`。

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
|赛题条款|T2-R3：B 样条平滑：对截取信号进行 B 样条/三次样条平滑以弱化噪声|
|业务角色|测试证据|
|业务输入|T2-R2 的滤波序列、裁剪范围与样条权重|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|平滑序列及相关参考数据|
|下一消费者|交给 T2-R4 多域特征提取|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
