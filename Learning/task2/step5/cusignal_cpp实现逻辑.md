# Task2 Step5 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task2 Step5 的一维局部极大值 shape/workspace、`argrelextrema` 正式 GPU 调用、坐标回收与 CPU reference。

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
|`ZKX/Task2/task2_gpu_cpu/step5/step5.h`|`run_step5` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/step5/step5.cu`|`run_step5`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`|`collect_step5_cpu_validation`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task2/task2_gpu_cpu/step5/step5.h`

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
//  `StepEvidence run_step5(PipelineState& state);` 中的 `run_step5` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。
//
//  【变量—数学符号—数据流】
//  run_step5：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
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

StepEvidence run_step5(PipelineState& state);

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

### 3.2 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step5.h"` 在预处理阶段引入 "step5.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "peak_finding/peak_finding_typed.h"` 在预处理阶段引入 "peak_finding/peak_finding_typed.h"
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
#include "step5.h"

#include "cuda_utils/device_array.h"
#include "peak_finding/peak_finding_typed.h"

// <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
//  `StepEvidence run_step5(PipelineState& state) {` 中的 `run_step5` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  run_step5：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
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
namespace task2 {

StepEvidence run_step5(PipelineState& state)
{
    StepEvidence evidence;
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.name = "step5";` 是赋值：先求得右侧完整表达式 `"step5"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
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
    evidence.name = "step5";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `require_task2_upstream( !state.feature_bundle.empty(), "Step5", "Step4.feature_bundle");`
    //  调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //  `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});` 声明 `shape`，其静态类型是 `const
    //  cusignal::RelativeExtremaShape`，并用 `{state.feature_bundle.size()}` 直接初始化/调用该类型的构造函数；这不是调用名为 `shape`
    //  的函数，模板尖括号也不是比较或位移。
    //
    //  【变量—数学符号—数据流】
    //  shape：当前表达式读取或传递的工程名称 shape；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const cusignal::RelativeExtremaShape
    //  shape({state.feature_bundle.size()});；值在 ZKX/Task2/task2_gpu_cpu/step5/step5.cu 当前作用域中产生或消费
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
    require_task2_upstream(
        !state.feature_bundle.empty(), "Step5", "Step4.feature_bundle");
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
    // <学习注释：语义块：workspace：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());` 声明 `workspace`，其静态类型是
    //  `cusignal::RelativeExtremaDeviceWorkspace`，并用 `state.feature_bundle.size()` 直接初始化/调用该类型的构造函数；这不是调用名为
    //  `workspace` 的函数，模板尖括号也不是比较或位移。
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //
    //  【变量—数学符号—数据流】
    //  workspace：当前表达式读取或传递的工程名称 workspace；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据
    //  cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());；值在
    //  ZKX/Task2/task2_gpu_cpu/step5/step5.cu 当前作用域中产生或消费
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
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
    cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    // <学习注释：语义块：workspace：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto d_bundle = cusignal::DeviceArray<float>::from_host(state.feature_bundle);` 是声明并初始化：`auto d_bundle`
    //  建立局部对象 `d_bundle`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.feature_bundle)` 产生初值。
    //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `cusignal::RelativeExtremaDeviceResult result;` 是对象声明：类型 `cusignal::RelativeExtremaDeviceResult` 应用于名称
    //  `result`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //
    //  【变量—数学符号—数据流】
    //  d_bundle：成组保存多域特征的结构/数组集合；可记作 $\{f_m[n]\}_m$；由 Step4 形成，Step5/6 读取
    //  result：当前调用返回或聚合得到的结果对象；对应当前算法/证据的输出；被赋值后由返回、比较或序列化消费
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
    auto d_bundle = cusignal::DeviceArray<float>::from_host(state.feature_bundle);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::RelativeExtremaDeviceResult result;
    // <学习注释：语义块：workspace：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.operator_ms["argrelextrema"] = time_gpu([&] { result = cusignal::argrelextrema_device( d_bundle,
    //  shape, cusignal::RelativeExtremaComparator::greater, workspace, 0, state.config.argrelextrema_order,
    //  "clip"); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["argrelextrema"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  result：当前调用返回或聚合得到的结果对象；对应当前算法/证据的输出；被赋值后由返回、比较或序列化消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  正式算子在一维 shape `{feature_bundle.size()}` 上寻找局部极大值：索引 $i$ 必须在半径 `order` 内满足 `x[i]>x[i±r]`（精确比较/边界按算子定义）。本次 comparator=`greater`、axis=`0`、mode=`clip`；输出 rank 为 1 的坐标数组 `extrema`。
    //
    //  【为什么这样设计】
    //  邻域 order 抑制过密峰；clip 模式决定边界访问。内部比较和 workspace 归 `Learning/operators/peak_finding/argrelextrema/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.operator_ms["argrelextrema"] = time_gpu([&] {
        result = cusignal::argrelextrema_device(
            d_bundle, shape, cusignal::RelativeExtremaComparator::greater,
            workspace, 0, state.config.argrelextrema_order, "clip");
    });
    // <学习注释：语义块：workspace：完成一个连续的数据处理动作。
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
    // <学习注释：语义块：workspace：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.compute_ms = evidence.operator_ms["argrelextrema"];` 是赋值：先求得右侧完整表达式
    //  `evidence.operator_ms["argrelextrema"]`，再把结果写入左侧可修改对象 `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `require(result.coordinates.size() == 1, "step5 coordinate rank mismatch");`
    //  调用本文件的前置条件检查；第一个实参为真时继续，为假时按第二个实参给出的消息报告配置或数值退化错误。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  1：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.compute_ms = evidence.operator_ms["argrelextrema"];
    begin = Clock::now();
    require(result.coordinates.size() == 1, "step5 coordinate rank mismatch");
    // <学习注释：语义块：workspace：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.extrema = result.coordinates.front().to_host();` 是赋值：先求得右侧完整表达式
    //  `result.coordinates.front().to_host()`，再把结果写入左侧可修改对象 `state.extrema`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
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
    state.extrema = result.coordinates.front().to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());

    // <学习注释：语义块：workspace：形成并返回当前结果。
    //
    //  【执行方法】
    //  `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
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
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
// <学习注释：语义块：workspace：完成一个连续的数据处理动作。
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

// <学习注释：语义块：workspace：完成一个连续的数据处理动作。
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

### 3.3 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`：`collect_step5_cpu_validation`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "accuracy/validation/step5_validation.h"` 在预处理阶段引入 "accuracy/validation/step5_validation.h"
//  的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "peak_finding/peak_finding_typed.h"` 在预处理阶段引入 "peak_finding/peak_finding_typed.h"
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
#include "accuracy/validation/step5_validation.h"

#include "peak_finding/peak_finding_typed.h"

// <学习注释：语义块：collect_step5_cpu_validation：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `namespace task2::accuracy {` 打开命名空间 `task2::accuracy`；后续声明被放入该名称范围。
//  `Step5ValidationData collect_step5_cpu_validation(const PipelineState& state) {` 中的
//  `collect_step5_cpu_validation` 是函数定义；名称前的 `Step5ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `Step5ValidationData data;` 是对象声明：类型 `Step5ValidationData` 应用于名称
//  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
//
//  【变量—数学符号—数据流】
//  collect_step5_cpu_validation：测试、收集或契约检查 helper；无独立数学变量；内部指标分别映射到误差/条件公式；消费运行结果并形成证据或失败路径
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
namespace task2::accuracy {

Step5ValidationData collect_step5_cpu_validation(const PipelineState& state)
{
    Step5ValidationData data;
    // <学习注释：语义块：shape：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});` 声明 `shape`，其静态类型是 `const
    //  cusignal::RelativeExtremaShape`，并用 `{state.feature_bundle.size()}` 直接初始化/调用该类型的构造函数；这不是调用名为 `shape`
    //  的函数，模板尖括号也不是比较或位移。
    //  `data.expected = cusignal::argrelextrema_typed_cpu( state.feature_bundle, shape,
    //  cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order,
    //  "clip").coordinates.front();` 是赋值：先求得右侧完整表达式 `cusignal::argrelextrema_typed_cpu( state.feature_bundle,
    //  shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order,
    //  "clip").coordinates.front()`，再把结果写入左侧可修改对象 `data.expected`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  shape：当前函数或调用的参数名称，接收调用者绑定的输入 shape；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const
    //  cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp 当前作用域中产生或消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  正式算子在一维 shape `{feature_bundle.size()}` 上寻找局部极大值：索引 $i$ 必须在半径 `order` 内满足 `x[i]>x[i±r]`（精确比较/边界按算子定义）。本次 comparator=`greater`、axis=`0`、mode=`clip`；输出 rank 为 1 的坐标数组 `extrema`。
    //
    //  【为什么这样设计】
    //  邻域 order 抑制过密峰；clip 模式决定边界访问。内部比较和 workspace 归 `Learning/operators/peak_finding/argrelextrema/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
    data.expected = cusignal::argrelextrema_typed_cpu(
        state.feature_bundle, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
    // <学习注释：语义块：shape：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `data.corrupted_expected = data.expected;` 是赋值：先求得右侧完整表达式 `data.expected`，再把结果写入左侧可修改对象
    //  `data.corrupted_expected`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `if (!data.corrupted_expected.empty()) ++data.corrupted_expected.front();` 是 `if` 条件语句：先把完整条件
    //  `!data.corrupted_expected.empty()` 求值并转换为布尔值。 条件为真时只执行其直接子语句
    //  `++data.corrupted_expected.front();`；为假时跳过该子语句。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    data.corrupted_expected = data.expected;
    if (!data.corrupted_expected.empty()) ++data.corrupted_expected.front();

    // <学习注释：语义块：shape：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto injected = state.feature_bundle;` 是声明并初始化：`auto injected` 建立局部对象 `injected`，右侧完整表达式
    //  `state.feature_bundle` 产生初值。
    //  `const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;` 是声明并初始化：`const float
    //  high` 建立局部对象 `high`，右侧完整表达式 `*std::max_element(injected.begin(), injected.end()) + 10.0F` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  injected：当前块的赋值左端，保存右侧表达式产生的中间值或结果 injected；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 auto injected =
    //  state.feature_bundle;；值在 ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp 当前作用域中产生或消费
    //  high：当前块的赋值左端，保存右侧表达式产生的中间值或结果 high；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 const float high =
    //  *std::max_element(injected.begin(), injected.end()) + 10.0F;；值在
    //  ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp 当前作用域中产生或消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数字常量】
    //  10.0F：F 使浮点字面量为 float；不带后缀默认是 double；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
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
    auto injected = state.feature_bundle;
    const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;
    // <学习注释：语义块：shape：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `injected[50] = high;` 是赋值：先求得右侧完整表达式 `high`，再把结果写入左侧可修改对象 `injected[50]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `data.injected_cpu = cusignal::argrelextrema_typed_cpu( injected, shape,
    //  cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order,
    //  "clip").coordinates.front();` 是赋值：先求得右侧完整表达式 `cusignal::argrelextrema_typed_cpu( injected, shape,
    //  cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order,
    //  "clip").coordinates.front()`，再把结果写入左侧可修改对象 `data.injected_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  50：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  正式算子在一维 shape `{feature_bundle.size()}` 上寻找局部极大值：索引 $i$ 必须在半径 `order` 内满足 `x[i]>x[i±r]`（精确比较/边界按算子定义）。本次 comparator=`greater`、axis=`0`、mode=`clip`；输出 rank 为 1 的坐标数组 `extrema`。
    //
    //  【为什么这样设计】
    //  邻域 order 抑制过密峰；clip 模式决定边界访问。内部比较和 workspace 归 `Learning/operators/peak_finding/argrelextrema/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    injected[50] = high;
    data.injected_cpu = cusignal::argrelextrema_typed_cpu(
        injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();

    // <学习注释：语义块：shape：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `injected[50] = state.feature_bundle[50];` 是赋值：先求得右侧完整表达式 `state.feature_bundle[50]`，再把结果写入左侧可修改对象
    //  `injected[50]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `injected[70] = high;` 是赋值：先求得右侧完整表达式 `high`，再把结果写入左侧可修改对象 `injected[70]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数字常量】
    //  50：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
    //  或绘图参数，源码未记录推导依据；因此只能确认其当前作用，不能把它宣称为通用理论常数。改变它会通过所在表达式影响数组规模、阈值、迭代次数、误差门限或可视化样式。
    //  70：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值由当前仓库源码直接写入。若它是 demo 规模、阈值、容差、seed
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
    injected[50] = state.feature_bundle[50];
    injected[70] = high;
    // <学习注释：语义块：shape：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `data.moved_cpu = cusignal::argrelextrema_typed_cpu( injected, shape,
    //  cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order,
    //  "clip").coordinates.front();` 是赋值：先求得右侧完整表达式 `cusignal::argrelextrema_typed_cpu( injected, shape,
    //  cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order,
    //  "clip").coordinates.front()`，再把结果写入左侧可修改对象 `data.moved_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。；入口加载，runner 和各 step 只读消费
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  正式算子在一维 shape `{feature_bundle.size()}` 上寻找局部极大值：索引 $i$ 必须在半径 `order` 内满足 `x[i]>x[i±r]`（精确比较/边界按算子定义）。本次 comparator=`greater`、axis=`0`、mode=`clip`；输出 rank 为 1 的坐标数组 `extrema`。
    //
    //  【为什么这样设计】
    //  邻域 order 抑制过密峰；clip 模式决定边界访问。内部比较和 workspace 归 `Learning/operators/peak_finding/argrelextrema/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    data.moved_cpu = cusignal::argrelextrema_typed_cpu(
        injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
    // <学习注释：语义块：shape：形成并返回当前结果。
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

// <学习注释：语义块：shape：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `}  // namespace task2::accuracy` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
}  // namespace task2::accuracy
```

### 3.4 函数总体处理方法

#### 3.4.1 总体业务流水线

`Step4 fused feature→argrelextrema→D2H`。输出 extrema 与 strongest point 给 Step6。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`run_step5`|校验融合特征，调用局部极值算子并确定关键特征点|GPU Host wrapper/业务入口|
|`collect_step5_cpu_validation`|用同一 comparator/order 生成峰值定位 CPU 对照|CPU/GPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `argrelextrema`

- 学习入口：[数学物理原理](../../operators/peak_finding/argrelextrema/数学物理原理.md)、[Python 源码算法](../../operators/peak_finding/argrelextrema/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/peak_finding/argrelextrema/cusignal_cpp_argrelextrema复现逻辑.md)。
- 当前调用采用的数学关系：$i$ 为局部极大值当且仅当在指定 `order` 邻域内 $x[i]$ 满足 comparator 相对关系。
- 任务变量与参数实例化：`fused feature→x`、`order→邻域半径`、comparator=`greater`、输出 `extrema/peak_indices`。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 正式调用的比较关系与坐标输出

`argrelextrema_device(d_bundle, shape, greater, workspace, 0, order, "clip")` 的绑定为：

|实参|本次含义|
|---|---|
|`d_bundle`|Step4 `feature_bundle`，device `float[N]`|
|`shape({N})`|一维输入布局|
|`greater`|寻找局部极大值，而非极小值|
|`axis=0`|沿唯一轴比较|
|`order=argrelextrema_order`|左右比较邻域半径|
|`mode="clip"`|边界按算子定义的 clip 规则处理|
|`result.coordinates`|rank 必须为 1；第一个坐标数组 D2H 后写入 `state.extrema`|

数学判据可写为候选 $i$ 在算子规定邻域内满足 $x[i]>x[i\pm r]$；等号、边界与全部比较细节以链接的算子文档为准。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

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
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|公共基础设施|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.h`；符号 `文件级代码`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step5(PipelineState& state);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：命名空间声明、函数定义或声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step5`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `StepEvidence run_step5(PipelineState& state);` 中的 `run_step5` 是函数声明；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾分号结束声明；真正执行只会发生在其他位置调用该函数时。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过定义/声明 `run_step5`；调用 `run_step5`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 文件级代码：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.h`；符号 `文件级代码`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.h` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `文件级代码`；源码锚点 `#include "step5.h"`。

```cpp
#include "step5.h"

#include "cuda_utils/device_array.h"
#include "peak_finding/peak_finding_typed.h"
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step5`、`h`、`cuda_utils`、`device_array`、`peak_finding`、`peak_finding_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step5.h"` 在预处理阶段引入 "step5.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "peak_finding/peak_finding_typed.h"` 在预处理阶段引入 "peak_finding/peak_finding_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|公共基础设施|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `namespace task2 {`。

```cpp
namespace task2 {

StepEvidence run_step5(PipelineState& state)
{
    StepEvidence evidence;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：命名空间声明、函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `evidence` 由 `StepEvidence` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step5`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2 {` 打开命名空间 `task2`；后续声明被放入该名称范围。
- `StepEvidence run_step5(PipelineState& state) {` 中的 `run_step5` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过定义/声明 `run_step5`；调用 `run_step5`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `evidence.name = "step5";`。

```cpp
    evidence.name = "step5";
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

- `evidence.name = "step5";` 是赋值：先求得右侧完整表达式 `"step5"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto total_begin = Clock::now();` 是声明并初始化：`const auto total_begin` 建立局部对象 `total_begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `total_begin`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.7 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `require_task2_upstream(`。

```cpp
    require_task2_upstream(
        !state.feature_bundle.empty(), "Step5", "Step4.feature_bundle");
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `require_task2_upstream`、`state`、`feature_bundle`、`empty`、`Step5`、`Step4`、`cusignal`、`RelativeExtremaShape`、`shape`、`size`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`shape`|当前表达式读取或传递的工程名称 `shape`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});`；值在 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `require_task2_upstream( !state.feature_bundle.empty(), "Step5", "Step4.feature_bundle");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
- `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});` 声明 `shape`，其静态类型是 `const cusignal::RelativeExtremaShape`，并用 `{state.feature_bundle.size()}` 直接初始化/调用该类型的构造函数；这不是调用名为 `shape` 的函数，模板尖括号也不是比较或位移。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `require_task2_upstream`, `empty`, `shape`, `size`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());`。

```cpp
    cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `cusignal`、`RelativeExtremaDeviceWorkspace`、`workspace`、`state`、`feature_bundle`、`size`、`evidence`、`prep_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`workspace`|当前表达式读取或传递的工程名称 `workspace`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());`；值在 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 当前作用域中产生或消费|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());` 声明 `workspace`，其静态类型是 `cusignal::RelativeExtremaDeviceWorkspace`，并用 `state.feature_bundle.size()` 直接初始化/调用该类型的构造函数；这不是调用名为 `workspace` 的函数，模板尖括号也不是比较或位移。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `workspace`, `size`, `milliseconds`, `Clock::now`；写入/初始化 `prep_ms`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `workspace` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.9 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `auto d_bundle = cusignal::DeviceArray<float>::from_host(state.feature_bundle);`。

```cpp
    auto d_bundle = cusignal::DeviceArray<float>::from_host(state.feature_bundle);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::RelativeExtremaDeviceResult result;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_bundle` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `result` 由 `cusignal::RelativeExtremaDeviceResult` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_bundle`|成组保存多域特征的结构/数组集合|可记作 $\{f_m[n]\}_m$|由 Step4 形成，Step5/6 读取|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_bundle = cusignal::DeviceArray<float>::from_host(state.feature_bundle);` 是声明并初始化：`auto d_bundle` 建立局部对象 `d_bundle`，右侧完整表达式 `cusignal::DeviceArray<float>::from_host(state.feature_bundle)` 产生初值。
- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::RelativeExtremaDeviceResult result;` 是对象声明：类型 `cusignal::RelativeExtremaDeviceResult` 应用于名称 `result`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|公共基础设施|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`；写入/初始化 `d_bundle`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `evidence.operator_ms["argrelextrema"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["argrelextrema"] = time_gpu([&] {
        result = cusignal::argrelextrema_device(
            d_bundle, shape, cusignal::RelativeExtremaComparator::greater,
            workspace, 0, state.config.argrelextrema_order, "clip");
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `evidence.operator_ms["argrelextrema"] = time_gpu([&] { result = cusignal::argrelextrema_device( d_bundle, shape, cusignal::RelativeExtremaComparator::greater, workspace, 0, state.config.argrelextrema_order, "clip"); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["argrelextrema"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|输入准备/数据契约|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `time_gpu`, `cusignal::argrelextrema_device`；写入/初始化 `operator_ms["argrelextrema"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.11 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.12 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `evidence.compute_ms = evidence.operator_ms["argrelextrema"];`。

```cpp
    evidence.compute_ms = evidence.operator_ms["argrelextrema"];
    begin = Clock::now();
    require(result.coordinates.size() == 1, "step5 coordinate rank mismatch");
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `1` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`1`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|一在当前上下文通常表示乘法单位元、单步增量、首个元素或 true；若属于配置/测试值，源码未记录额外推导依据。|

**执行过程**

- `evidence.compute_ms = evidence.operator_ms["argrelextrema"];` 是赋值：先求得右侧完整表达式 `evidence.operator_ms["argrelextrema"]`，再把结果写入左侧可修改对象 `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `require(result.coordinates.size() == 1, "step5 coordinate rank mismatch");` 调用本文件的前置条件检查；第一个实参为真时继续，为假时按第二个实参给出的消息报告配置或数值退化错误。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `Clock::now`, `require`, `size`；写入/初始化 `compute_ms`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `state.extrema = result.coordinates.front().to_host();`。

```cpp
    state.extrema = result.coordinates.front().to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`extrema`、`result`、`coordinates`、`front`、`to_host`、`evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.extrema = result.coordinates.front().to_host();` 是赋值：先求得右侧完整表达式 `result.coordinates.front().to_host()`，再把结果写入左侧可修改对象 `state.extrema`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `front`, `to_host`, `milliseconds`, `Clock::now`；写入/初始化 `extrema`, `d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `workspace` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.14 workspace：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `evidence.total_ms = milliseconds(total_begin, Clock::now());`。

```cpp

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、`return` 跳转语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `evidence` 由 `return` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.total_ms = milliseconds(total_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(total_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.total_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `return evidence;` 是返回语句：先求值 `evidence`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `total_ms`；返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `workspace` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.15 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `}`。

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
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `workspace` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.16 workspace：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu`；符号 `workspace`；源码锚点 `}  // namespace task2`。

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
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|核心算法或本步数据准备|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `workspace` 所在的 `ZKX/Task2/task2_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.17 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `workspace`；源码锚点 `#include "accuracy/validation/step5_validation.h"`。

```cpp
#include "accuracy/validation/step5_validation.h"

#include "peak_finding/peak_finding_typed.h"
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`accuracy`、`validation`、`step5_validation`、`h`、`peak_finding`、`peak_finding_typed`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "accuracy/validation/step5_validation.h"` 在预处理阶段引入 "accuracy/validation/step5_validation.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "peak_finding/peak_finding_typed.h"` 在预处理阶段引入 "peak_finding/peak_finding_typed.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 collect_step5_cpu_validation：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `collect_step5_cpu_validation`；源码锚点 `namespace task2::accuracy {`。

```cpp
namespace task2::accuracy {

Step5ValidationData collect_step5_cpu_validation(const PipelineState& state)
{
    Step5ValidationData data;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：命名空间声明、函数定义或声明、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `const PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `data` 由 `Step5ValidationData` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`collect_step5_cpu_validation`|测试、收集或契约检查 helper|无独立数学变量；内部指标分别映射到误差/条件公式|消费运行结果并形成证据或失败路径|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task2::accuracy {` 打开命名空间 `task2::accuracy`；后续声明被放入该名称范围。
- `Step5ValidationData collect_step5_cpu_validation(const PipelineState& state) {` 中的 `collect_step5_cpu_validation` 是函数定义；名称前的 `Step5ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `Step5ValidationData data;` 是对象声明：类型 `Step5ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过定义/声明 `collect_step5_cpu_validation`；调用 `collect_step5_cpu_validation`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.19 shape：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});`。

```cpp
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
    data.expected = cusignal::argrelextrema_typed_cpu(
        state.feature_bundle, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：对象构造或函数调用语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`shape`|当前函数或调用的参数名称，接收调用者绑定的输入 `shape`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});` 声明 `shape`，其静态类型是 `const cusignal::RelativeExtremaShape`，并用 `{state.feature_bundle.size()}` 直接初始化/调用该类型的构造函数；这不是调用名为 `shape` 的函数，模板尖括号也不是比较或位移。
- `data.expected = cusignal::argrelextrema_typed_cpu( state.feature_bundle, shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order, "clip").coordinates.front();` 是赋值：先求得右侧完整表达式 `cusignal::argrelextrema_typed_cpu( state.feature_bundle, shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order, "clip").coordinates.front()`，再把结果写入左侧可修改对象 `data.expected`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `shape`, `size`, `cusignal::argrelextrema_typed_cpu`, `front`；写入/初始化 `expected`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 shape：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `data.corrupted_expected = data.expected;`。

```cpp
    data.corrupted_expected = data.expected;
    if (!data.corrupted_expected.empty()) ++data.corrupted_expected.front();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句、`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `data`、`corrupted_expected`、`expected`、`empty`、`front`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `data.corrupted_expected = data.expected;` 是赋值：先求得右侧完整表达式 `data.expected`，再把结果写入左侧可修改对象 `data.corrupted_expected`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `if (!data.corrupted_expected.empty()) ++data.corrupted_expected.front();` 是 `if` 条件语句：先把完整条件 `!data.corrupted_expected.empty()` 求值并转换为布尔值。 条件为真时只执行其直接子语句 `++data.corrupted_expected.front();`；为假时跳过该子语句。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过检查 `!data.corrupted_expected.empty()`；调用 `empty`, `front`；写入/初始化 `corrupted_expected`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.21 shape：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `auto injected = state.feature_bundle;`。

```cpp

    auto injected = state.feature_bundle;
    const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `injected` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式；
- `high` 由 `const float` 声明：通常为 IEEE 754 binary32 单精度浮点数；约 7 位十进制有效数字，运算会舍入；`const` 禁止通过该名称修改对象；
- `10.0F` 是数值字面量；F 指定 float，而非默认 double。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`injected`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `injected`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `auto injected = state.feature_bundle;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp` 当前作用域中产生或消费|
|`high`|当前块的赋值左端，保存右侧表达式产生的中间值或结果 `high`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;`；值在 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp` 当前作用域中产生或消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`10.0F`|`F` 使浮点字面量为 float；不带后缀默认是 double|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `auto injected = state.feature_bundle;` 是声明并初始化：`auto injected` 建立局部对象 `injected`，右侧完整表达式 `state.feature_bundle` 产生初值。
- `const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;` 是声明并初始化：`const float high` 建立局部对象 `high`，右侧完整表达式 `*std::max_element(injected.begin(), injected.end()) + 10.0F` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `std::max_element`, `begin`, `end`；写入/初始化 `injected`, `high`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 shape：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `injected[50] = high;`。

```cpp
    injected[50] = high;
    data.injected_cpu = cusignal::argrelextrema_typed_cpu(
        injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `50` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`50`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `injected[50] = high;` 是赋值：先求得右侧完整表达式 `high`，再把结果写入左侧可修改对象 `injected[50]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.injected_cpu = cusignal::argrelextrema_typed_cpu( injected, shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order, "clip").coordinates.front();` 是赋值：先求得右侧完整表达式 `cusignal::argrelextrema_typed_cpu( injected, shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order, "clip").coordinates.front()`，再把结果写入左侧可修改对象 `data.injected_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `cusignal::argrelextrema_typed_cpu`, `front`；写入/初始化 `injected[50]`, `injected_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.23 shape：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `injected[50] = state.feature_bundle[50];`。

```cpp

    injected[50] = state.feature_bundle[50];
    injected[70] = high;
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `50` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `50` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定；
- `70` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`50`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|
|`70`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该字面量由当前仓库源码固定使用。紧邻执行过程说明它参与的具体表达式；若源码、配置或正式证据未给出选值推导，文档明确保留“选值依据未知”，不把它泛化为理论常数。|

**执行过程**

- `injected[50] = state.feature_bundle[50];` 是赋值：先求得右侧完整表达式 `state.feature_bundle[50]`，再把结果写入左侧可修改对象 `injected[50]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `injected[70] = high;` 是赋值：先求得右侧完整表达式 `high`，再把结果写入左侧可修改对象 `injected[70]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过写入/初始化 `injected[50]`, `injected[70]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 shape：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `data.moved_cpu = cusignal::argrelextrema_typed_cpu(`。

```cpp
    data.moved_cpu = cusignal::argrelextrema_typed_cpu(
        injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化局部极值的 `order`、比较器和关键点选择口径。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `data.moved_cpu = cusignal::argrelextrema_typed_cpu( injected, shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order, "clip").coordinates.front();` 是赋值：先求得右侧完整表达式 `cusignal::argrelextrema_typed_cpu( injected, shape, cusignal::RelativeExtremaComparator::greater, 0, state.config.argrelextrema_order, "clip").coordinates.front()`，再把结果写入左侧可修改对象 `data.moved_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过调用 `cusignal::argrelextrema_typed_cpu`, `front`；写入/初始化 `moved_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.25 shape：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `return data;`。

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
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.26 shape：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task2/task2_gpu_cpu/accuracy/validation/step5_cpu_validation.cpp`；符号 `shape`；源码锚点 `}  // namespace task2::accuracy`。

```cpp

}  // namespace task2::accuracy
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `namespace`、`task2`、`accuracy`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}  // namespace task2::accuracy` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构；`//` 后面的文字仅标记所关闭的命名空间。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T2-R5：关键特征点定位：使用峰值/相对极值查找确定关键位置|
|业务角色|测试证据|
|业务输入|T2-R4 的融合特征和极值邻域 `order`|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|峰值索引 `extrema`/`peak_indices` 与最强特征点|
|下一消费者|作为 T2-R6 Kalman 观测锚点|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
