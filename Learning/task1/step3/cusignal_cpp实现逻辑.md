# Task1 Step3 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task1 Step3 的慢时间 Hamming 窗、多普勒 FFT 参数绑定、GPU 正式算子调用、距离—多普勒输出布局与独立 CPU reference；算子内部实现不在任务文档重复。

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
|`ZKX/Task1/task1_gpu_cpu/step3/step3.h`|`run_step3` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/step3/step3.cu`|`run_step3`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`|`generate_step3_cpu_reference`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task1/task1_gpu_cpu/step3/step3.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
//  `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task1 { StepEvidence run_step3(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence
//  run_step3(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。
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
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step3(PipelineState& state); }
```

### 3.2 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step3.h"` 在预处理阶段引入 "step3.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h"
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

#include "cuda_utils/device_array.h"

// <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `namespace task1 {` 打开命名空间 `task1`；后续声明被放入该名称范围。
//  `StepEvidence run_step3(PipelineState& state) {` 中的 `run_step3` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  run_step3：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
namespace task1 {

StepEvidence run_step3(PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.name = "step3";` 是赋值：先求得右侧完整表达式 `"step3"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
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
    evidence.name = "step3";
    const auto formal_begin = Clock::now();
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //  `require_task1_upstream(state.compressed.size() == static_cast<std::size_t>( config.num_pulses *
    //  config.samples_per_pulse), "Step3", "Step2.compressed");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  本块不做 Doppler 变换，只验证 `compressed.size()==P*M`，确保展平缓冲区可按 $y[p,n]$ 的 $P×M$ 布局沿 $p$ 维变换。
    //
    //  【为什么这样设计】
    //  shape 检查阻止错误步长造成跨距离单元混叠或越界。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    auto begin = Clock::now();
    require_task1_upstream(state.compressed.size() == static_cast<std::size_t>(
                config.num_pulses * config.samples_per_pulse),
            "Step3", "Step2.compressed");
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::PulseDopplerOptions options;` 是对象声明：类型 `cusignal::PulseDopplerOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `options.num_pulses = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象
    //  `options.num_pulses`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.samples_per_pulse = config.samples_per_pulse;` 是赋值：先求得右侧完整表达式
    //  `config.samples_per_pulse`，再把结果写入左侧可修改对象 `options.samples_per_pulse`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::PulseDopplerOptions
    //  options;；值在 ZKX/Task1/task1_gpu_cpu/step3/step3.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `P=num_pulses`、`M=samples_per_pulse` 规定输入布局，`nfft=P` 输出 $P$ 个 Doppler bin。Hamming 分支 $w[p]=0.54-0.46\cos(2\pi p/(P-1))$；none 分支 $w[p]=1$。
    //
    //  【为什么这样设计】
    //  Hamming 压低慢时间谱旁瓣但加宽主瓣；无窗分支用于对照。显式 options 使差异可审计。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::PulseDopplerOptions options;
    options.num_pulses = config.num_pulses;
    options.samples_per_pulse = config.samples_per_pulse;
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.nfft = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象 `options.nfft`；这里的
    //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.window = cusignal::RadarWindowKind::hamming;` 是赋值：先求得右侧完整表达式
    //  `cusignal::RadarWindowKind::hamming`，再把结果写入左侧可修改对象 `options.window`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  `P=num_pulses`、`M=samples_per_pulse` 规定输入布局，`nfft=P` 输出 $P$ 个 Doppler bin。Hamming 分支 $w[p]=0.54-0.46\cos(2\pi p/(P-1))$；none 分支 $w[p]=1$。
    //
    //  【为什么这样设计】
    //  Hamming 压低慢时间谱旁瓣但加宽主瓣；无窗分支用于对照。显式 options 使差异可审计。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.nfft = config.num_pulses;
    options.window = cusignal::RadarWindowKind::hamming;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `auto d_compressed = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed);`
    //  是声明并初始化：`auto d_compressed` 建立局部对象 `d_compressed`，右侧完整表达式
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed)` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  d_compressed：脉冲压缩输出；记作 $y[p,n]=(x[p,\cdot]\star s)[n]$；作为多普勒处理输入
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
    //  模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。>
    begin = Clock::now();
    auto d_compressed =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed);
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `const std::size_t output_count = static_cast<std::size_t>(config.num_pulses) *
    //  static_cast<std::size_t>(config.samples_per_pulse);` 是声明并初始化：`const std::size_t output_count` 建立局部对象
    //  `output_count`，右侧完整表达式 `static_cast<std::size_t>(config.num_pulses) *
    //  static_cast<std::size_t>(config.samples_per_pulse)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数
    //  `T`，转换发生后才参与外层表达式。
    //
    //  【变量—数学符号—数据流】
    //  output_count：当前函数或步骤的输出容器；对应当前算法公式的左端结果，具体符号由所在 step 决定；由本块写入，返回或交给下一步骤
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
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
    const std::size_t output_count = static_cast<std::size_t>(config.num_pulses) *
        static_cast<std::size_t>(config.samples_per_pulse);
    // <学习注释：语义块：run_step3：检查条件并选择执行路径。
    //
    //  【执行方法】
    //  `if (state.range_doppler_device.size() != output_count) {` 是 `if` 条件语句：先把完整条件
    //  `state.range_doppler_device.size() != output_count` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
    //  `state.range_doppler_device.reset(output_count);` 调用资源对象的 `reset`，按给定元素数重新建立其持有的缓冲区；旧资源由对象的所有权逻辑释放。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
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
    //  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。>
    if (state.range_doppler_device.size() != output_count) {
        state.range_doppler_device.reset(output_count);
    }
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.operator_ms["pulse_doppler"] = time_gpu([&] { cusignal::pulse_doppler_device<float>(
    //  d_compressed, options, state.range_doppler_device); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu`
    //  执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["pulse_doppler"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  正式算子对固定距离单元计算 $D[k,n]=\sum_{p=0}^{P-1}y[p,n]w[p]e^{-j2\pi kp/P}$。`compressed/d_compressed→y[p,n]∈ℂ^{P×M}`，输出 `range_doppler*→D[k,n]∈ℂ^{P×M}`；本次 `nfft=P`。正式 GPU 路径 `window=hamming`，对照分支可能为 `window=none`，必须分开理解。
    //
    //  【为什么这样设计】
    //  慢时间 DFT 把脉间相位推进转换为 Doppler bin；任务层只绑定 shape、FFT 长度和窗口，算子内部实现归 `Learning/operators/radartools/pulse_doppler/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.operator_ms["pulse_doppler"] = time_gpu([&] {
        cusignal::pulse_doppler_device<float>(
            d_compressed, options, state.range_doppler_device);
    });
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.compute_ms = evidence.operator_ms["pulse_doppler"];` 是赋值：先求得右侧完整表达式
    //  `evidence.operator_ms["pulse_doppler"]`，再把结果写入左侧可修改对象 `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `std::size_t free_bytes = 0, total_bytes = 0;` 在同一条声明中建立两个 `std::size_t` 对象：`free_bytes` 由 `0`
    //  初始化，`total_bytes` 由 `0` 初始化；逗号分隔两个声明符，不是逗号运算符。
    //  `CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));` 通过 `CUDA_CHECK` 包装 CUDA Runtime
    //  调用；宏检查返回状态，失败时按项目统一错误路径传播，而实参调用负责实际查询/同步/复制。
    //
    //  【变量—数学符号—数据流】
    //  free_bytes：设备当前空闲显存字节数；无独立数学符号；与 total_bytes 组合估计显存使用
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立信号处理变换。计时仅形成 $Δt=t_{end}-t_{begin}$，显存指标仅形成 `total_bytes-free_bytes`；这些证据量不参与算法输出。
    //
    //  【为什么这样设计】
    //  分开记录准备、传输、计算和端到端区间才能解释耗时边界；观测量不反馈改变数值路径。
    //
    //  【初学者易错点】
    //  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
    evidence.compute_ms = evidence.operator_ms["pulse_doppler"];
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);` 是赋值：先求得右侧完整表达式
    //  `static_cast<double>(total_bytes - free_bytes)`，再把结果写入左侧可修改对象
    //  `evidence.metrics["gpu_used_peak_bytes"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)`
    //  明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `const auto device_values = state.range_doppler_device.to_host();` 是声明并初始化：`const auto device_values`
    //  建立局部对象 `device_values`，右侧完整表达式 `state.range_doppler_device.to_host()` 产生初值。
    //
    //  【变量—数学符号—数据流】
    //  device_values：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
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
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    begin = Clock::now();
    const auto device_values = state.range_doppler_device.to_host();
    // <学习注释：语义块：run_step3：迭代处理数据范围。
    //
    //  【执行方法】
    //  `state.range_doppler.resize(device_values.size());` 调用容器的 `resize` 把逻辑元素数改为实参指定的大小；增大时创建新元素，缩小时移除尾部元素。
    //  `for (std::size_t index = 0; index < device_values.size(); ++index) {` 是经典 `for`：先执行初始化 `std::size_t index
    //  = 0`；每轮前检查 `index < device_values.size()`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
    //  `state.range_doppler[index] = { device_values[index].re, device_values[index].im};` 是赋值：先求得右侧完整表达式 `{
    //  device_values[index].re, device_values[index].im}`，再把结果写入左侧可修改对象 `state.range_doppler[index]`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  index：展平后的样本/元素索引；通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$；由 CUDA 线程号或循环产生；用于定位当前输出元素
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数字常量】
    //  0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  本块没有独立数学变换；条件只验证 shape、非空性或边界，使后续公式的索引集合有定义；条件为假时停止当前调用。
    //
    //  【为什么这样设计】
    //  尽早拒绝不满足契约的状态，可防止越界、除零或错误布局继续传播。
    //
    //  【初学者易错点】
    //  经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。>
    state.range_doppler.resize(device_values.size());
    for (std::size_t index = 0; index < device_values.size(); ++index) {
        state.range_doppler[index] = {
            device_values[index].re, device_values[index].im};
    // <学习注释：语义块：run_step3：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //  `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());` 是赋值：先求得右侧完整表达式
    //  `milliseconds(formal_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.formal_execution_ms`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
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
    }
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
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

### 3.3 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`：`generate_step3_cpu_reference`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：generate_step3_cpu_reference：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step3ValidationData generate_step3_cpu_reference(const PipelineState& state) {` 中的
//  `generate_step3_cpu_reference` 是函数定义；名称前的 `Step3ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  generate_step3_cpu_reference：生成当前名称所描述数据的 helper/结果；对应生成公式的输出端；由输入参数构造数据，供后续 step 或测试使用
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
Step3ValidationData generate_step3_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：generate_step3_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `Step3ValidationData data;` 是对象声明：类型 `Step3ValidationData` 应用于名称
    //  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `cusignal::PulseDopplerOptions options;` 是对象声明：类型 `cusignal::PulseDopplerOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `options.num_pulses = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象
    //  `options.num_pulses`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::PulseDopplerOptions
    //  options;；值在 ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `P=num_pulses`、`M=samples_per_pulse` 规定输入布局，`nfft=P` 输出 $P$ 个 Doppler bin。Hamming 分支 $w[p]=0.54-0.46\cos(2\pi p/(P-1))$；none 分支 $w[p]=1$。
    //
    //  【为什么这样设计】
    //  Hamming 压低慢时间谱旁瓣但加宽主瓣；无窗分支用于对照。显式 options 使差异可审计。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    Step3ValidationData data;
    cusignal::PulseDopplerOptions options;
    options.num_pulses = config.num_pulses;
    // <学习注释：语义块：generate_step3_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.samples_per_pulse = config.samples_per_pulse;` 是赋值：先求得右侧完整表达式
    //  `config.samples_per_pulse`，再把结果写入左侧可修改对象 `options.samples_per_pulse`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.nfft = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象 `options.nfft`；这里的
    //  `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.window = cusignal::RadarWindowKind::hamming;` 是赋值：先求得右侧完整表达式
    //  `cusignal::RadarWindowKind::hamming`，再把结果写入左侧可修改对象 `options.window`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  config：外部配置对象；`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `P=num_pulses`、`M=samples_per_pulse` 规定输入布局，`nfft=P` 输出 $P$ 个 Doppler bin。Hamming 分支 $w[p]=0.54-0.46\cos(2\pi p/(P-1))$；none 分支 $w[p]=1$。
    //
    //  【为什么这样设计】
    //  Hamming 压低慢时间谱旁瓣但加宽主瓣；无窗分支用于对照。显式 options 使差异可审计。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.samples_per_pulse = config.samples_per_pulse;
    options.nfft = config.num_pulses;
    options.window = cusignal::RadarWindowKind::hamming;
    // <学习注释：语义块：generate_step3_cpu_reference：形成并返回当前结果。
    //
    //  【执行方法】
    //  `data.range_doppler_cpu = radar_cpu_to_host( cusignal::pulse_doppler_typed_cpu<float>(state.compressed,
    //  options));` 是赋值：先求得右侧完整表达式 `radar_cpu_to_host( cusignal::pulse_doppler_typed_cpu<float>(state.compressed,
    //  options))`，再把结果写入左侧可修改对象 `data.range_doppler_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
    //
    //  【数学/物理公式对照】
    //  正式算子对固定距离单元计算 $D[k,n]=\sum_{p=0}^{P-1}y[p,n]w[p]e^{-j2\pi kp/P}$。`compressed/d_compressed→y[p,n]∈ℂ^{P×M}`，输出 `range_doppler*→D[k,n]∈ℂ^{P×M}`；本次 `nfft=P`。正式 GPU 路径 `window=hamming`，对照分支可能为 `window=none`，必须分开理解。
    //
    //  【为什么这样设计】
    //  慢时间 DFT 把脉间相位推进转换为 Doppler bin；任务层只绑定 shape、FFT 长度和窗口，算子内部实现归 `Learning/operators/radartools/pulse_doppler/`。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    data.range_doppler_cpu = radar_cpu_to_host(
        cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));
    return data;
// <学习注释：语义块：generate_step3_cpu_reference：完成一个连续的数据处理动作。
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

`Step2 compressed→H2D→pulse_doppler→D2H`。形成距离—多普勒结果，交给 Step4。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`run_step3`|校验 compressed，映射 Hamming/pulse_doppler 参数并回收距离—多普勒结果|GPU Host wrapper/业务入口|
|`generate_step3_cpu_reference`|用同一参数生成 Step3 独立 CPU reference|CPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `pulse_doppler`

- 学习入口：[数学物理原理](../../operators/radartools/pulse_doppler/数学物理原理.md)、[Python 源码算法](../../operators/radartools/pulse_doppler/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/radartools/pulse_doppler/cusignal_cpp_pulse_doppler复现逻辑.md)。
- 当前调用采用的数学关系：$D[k,n]=\sum_{p=0}^{P-1}y[p,n]w[p]e^{-j2\pi kp/P}$。
- 任务变量与参数实例化：`compressed→y`、`num_pulses→P`、窗口 `w[p]`、输出 `doppler_map→D`。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 正式调用的实参、shape 与具体公式

`pulse_doppler_device<float>(d_compressed, options, state.range_doppler_device)` 沿 $P$ 个脉冲的慢时间维计算
$D[k,n]=\sum_{p=0}^{P-1}y[p,n]w[p]e^{-j2\pi kp/P}$。

|实参/字段|本次绑定|数学与数据关系|
|---|---|---|
|`d_compressed`|device 复 `float`，长度 $PM$|$y[p,n]$，来自 Step2|
|`num_pulses`、`samples_per_pulse`|$P$、$M$|逻辑 shape $P\times M$|
|`nfft`|$P$|每个距离单元产生 $P$ 个 Doppler bin|
|`window`|正式路径 `hamming`|$w[p]=0.54-0.46\cos(2\pi p/(P-1))$|
|CPU 扰动分支|`window=none`|$w[p]=1$；它是对照，不能与正式结果混称|
|输出|device/Host 复 `float[P×M]`|`range_doppler_device` 供 Step4 resident 复用，Host 副本供证据/可视化|

##### `hamming`

- 学习入口：[数学物理原理](../../operators/windows/hamming/数学物理原理.md)、[Python 源码算法](../../operators/windows/hamming/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/windows/hamming/cusignal_cpp_hamming复现逻辑.md)。
- 当前调用采用的数学关系：$w[p]=0.54-0.46\cos(2\pi p/(P-1))$。
- 任务变量与参数实例化：通过 `RadarWindowKind::hamming` 在 pulse_doppler 中实例化；降低慢时间频谱旁瓣，但加宽主瓣。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step3(PipelineState& state); }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
- `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task1 { StepEvidence run_step3(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence run_step3(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|公共基础设施|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `文件级代码`；源码锚点 `#include "step3.h"`。

```cpp
#include "step3.h"

#include "cuda_utils/device_array.h"
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step3`、`h`、`cuda_utils`、`device_array`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step3.h"` 在预处理阶段引入 "step3.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|公共基础设施|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `namespace task1 {`。

```cpp
namespace task1 {

StepEvidence run_step3(PipelineState& state)
{
    const auto& config = state.config;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：命名空间声明、函数定义或声明、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `state` 由 `PipelineState&` 声明：类型控制可表示值、可用操作和传参方式；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `config` 由 `const auto&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`run_step3`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `namespace task1 {` 打开命名空间 `task1`；后续声明被放入该名称范围。
- `StepEvidence run_step3(PipelineState& state) {` 中的 `run_step3` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|输入准备/数据契约|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过定义/声明 `run_step3`；调用 `run_step3`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.4 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step3";
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
- `evidence.name = "step3";` 是赋值：先求得右侧完整表达式 `"step3"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto formal_begin = Clock::now();` 是声明并初始化：`const auto formal_begin` 建立局部对象 `formal_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `formal_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.5 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    require_task1_upstream(state.compressed.size() == static_cast<std::size_t>(
                config.num_pulses * config.samples_per_pulse),
            "Step3", "Step2.compressed");
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
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `require_task1_upstream(state.compressed.size() == static_cast<std::size_t>( config.num_pulses * config.samples_per_pulse), "Step3", "Step2.compressed");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|输入准备/数据契约|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `Clock::now`, `require_task1_upstream`, `size`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.6 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `cusignal::PulseDopplerOptions options;`。

```cpp
    cusignal::PulseDopplerOptions options;
    options.num_pulses = config.num_pulses;
    options.samples_per_pulse = config.samples_per_pulse;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `options` 由 `cusignal::PulseDopplerOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::PulseDopplerOptions options;`；值在 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::PulseDopplerOptions options;` 是对象声明：类型 `cusignal::PulseDopplerOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `options.num_pulses = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象 `options.num_pulses`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.samples_per_pulse = config.samples_per_pulse;` 是赋值：先求得右侧完整表达式 `config.samples_per_pulse`，再把结果写入左侧可修改对象 `options.samples_per_pulse`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|输入准备/数据契约|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过写入/初始化 `num_pulses`, `samples_per_pulse`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.7 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `options.nfft = config.num_pulses;`。

```cpp
    options.nfft = config.num_pulses;
    options.window = cusignal::RadarWindowKind::hamming;
    evidence.prep_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `options`、`nfft`、`config`、`num_pulses`、`window`、`cusignal`、`RadarWindowKind`、`hamming`、`evidence`、`prep_ms`、`milliseconds`、`begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.nfft = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象 `options.nfft`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.window = cusignal::RadarWindowKind::hamming;` 是赋值：先求得右侧完整表达式 `cusignal::RadarWindowKind::hamming`，再把结果写入左侧可修改对象 `options.window`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|输入准备/数据契约|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `nfft`, `window`, `prep_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    auto d_compressed =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_compressed` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_compressed`|脉冲压缩输出|记作 $y[p,n]=(x[p,\cdot]\star s)[n]$|作为多普勒处理输入|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `auto d_compressed = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed);` 是声明并初始化：`auto d_compressed` 建立局部对象 `d_compressed`，右侧完整表达式 `cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.compressed)` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|公共基础设施|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `Clock::now`, `from_host`；写入/初始化 `begin`, `d_compressed`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.9 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.h2d_ms = milliseconds(begin, Clock::now());`。

```cpp
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    const std::size_t output_count = static_cast<std::size_t>(config.num_pulses) *
        static_cast<std::size_t>(config.samples_per_pulse);
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `output_count` 由 `const std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`output_count`|当前函数或步骤的输出容器|对应当前算法公式的左端结果，具体符号由所在 step 决定|由本块写入，返回或交给下一步骤|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const std::size_t output_count = static_cast<std::size_t>(config.num_pulses) * static_cast<std::size_t>(config.samples_per_pulse);` 是声明并初始化：`const std::size_t output_count` 建立局部对象 `output_count`，右侧完整表达式 `static_cast<std::size_t>(config.num_pulses) * static_cast<std::size_t>(config.samples_per_pulse)` 产生初值。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|输入准备/数据契约|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `h2d_ms`, `output_count`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 run_step3：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `if (state.range_doppler_device.size() != output_count) {`。

```cpp
    if (state.range_doppler_device.size() != output_count) {
        state.range_doppler_device.reset(output_count);
    }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`if` 条件语句、对象构造或函数调用语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`range_doppler_device`、`size`、`output_count`、`reset`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `if (state.range_doppler_device.size() != output_count) {` 是 `if` 条件语句：先把完整条件 `state.range_doppler_device.size() != output_count` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `state.range_doppler_device.reset(output_count);` 调用资源对象的 `reset`，按给定元素数重新建立其持有的缓冲区；旧资源由对象的所有权逻辑释放。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过检查 `state.range_doppler_device.size() != output_count`；调用 `size`, `reset`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。

### 4.11 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.operator_ms["pulse_doppler"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["pulse_doppler"] = time_gpu([&] {
        cusignal::pulse_doppler_device<float>(
            d_compressed, options, state.range_doppler_device);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`pulse_doppler`、`time_gpu`、`cusignal`、`pulse_doppler_device`、`float`、`d_compressed`、`options`、`state`、`range_doppler_device`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.operator_ms["pulse_doppler"] = time_gpu([&] { cusignal::pulse_doppler_device<float>( d_compressed, options, state.range_doppler_device); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["pulse_doppler"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `time_gpu`；写入/初始化 `operator_ms["pulse_doppler"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.compute_ms = evidence.operator_ms["pulse_doppler"];`。

```cpp
    evidence.compute_ms = evidence.operator_ms["pulse_doppler"];
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
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

- `evidence.compute_ms = evidence.operator_ms["pulse_doppler"];` 是赋值：先求得右侧完整表达式 `evidence.operator_ms["pulse_doppler"]`，再把结果写入左侧可修改对象 `evidence.compute_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `std::size_t free_bytes = 0, total_bytes = 0;` 在同一条声明中建立两个 `std::size_t` 对象：`free_bytes` 由 `0` 初始化，`total_bytes` 由 `0` 初始化；逗号分隔两个声明符，不是逗号运算符。
- `CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));` 通过 `CUDA_CHECK` 包装 CUDA Runtime 调用；宏检查返回状态，失败时按项目统一错误路径传播，而实参调用负责实际查询/同步/复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `compute_ms`, `free_bytes`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.13 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);`。

```cpp
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    begin = Clock::now();
    const auto device_values = state.range_doppler_device.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `device_values` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`device_values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);` 是赋值：先求得右侧完整表达式 `static_cast<double>(total_bytes - free_bytes)`，再把结果写入左侧可修改对象 `evidence.metrics["gpu_used_peak_bytes"]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。 `static_cast<T>(...)` 明确要求编译器把括号内的值转换为模板参数 `T`，转换发生后才参与外层表达式。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto device_values = state.range_doppler_device.to_host();` 是声明并初始化：`const auto device_values` 建立局部对象 `device_values`，右侧完整表达式 `state.range_doppler_device.to_host()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `Clock::now`, `to_host`；写入/初始化 `metrics["gpu_used_peak_bytes"]`, `begin`, `device_values`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.14 run_step3：迭代处理数据范围

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `state.range_doppler.resize(device_values.size());`。

```cpp
    state.range_doppler.resize(device_values.size());
    for (std::size_t index = 0; index < device_values.size(); ++index) {
        state.range_doppler[index] = {
            device_values[index].re, device_values[index].im};
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：对象构造或函数调用语句、`for` 迭代语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `index` 由 `std::size_t` 声明：用于对象大小和下标的无符号整数类型，位宽足以表示当前平台最大对象大小；
- `0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`index`|展平后的样本/元素索引|通常记作 $i$；若二维数据按行展开，则 $i=pN_s+n$|由 CUDA 线程号或循环产生；用于定位当前输出元素|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `state.range_doppler.resize(device_values.size());` 调用容器的 `resize` 把逻辑元素数改为实参指定的大小；增大时创建新元素，缩小时移除尾部元素。
- `for (std::size_t index = 0; index < device_values.size(); ++index) {` 是经典 `for`：先执行初始化 `std::size_t index = 0`；每轮前检查 `index < device_values.size()`，为假即退出；每轮循环体结束后执行 `++index`，再检查下一轮。
- `state.range_doppler[index] = { device_values[index].re, device_values[index].im};` 是赋值：先求得右侧完整表达式 `{ device_values[index].re, device_values[index].im}`，再把结果写入左侧可修改对象 `state.range_doppler[index]`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `resize`, `size`；写入/初始化 `range_doppler[index]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 经典 `for` 与范围 `for` 的圆括号语法不同；范围 `for` 中的冒号不是条件运算符。

### 4.15 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `}`。

```cpp
    }
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：作用域边界、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`、`formal_execution_ms`、`formal_begin`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(formal_begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.formal_execution_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `d2h_ms`, `formal_execution_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.16 run_step3：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `return evidence;`。

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
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.17 run_step3：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu`；符号 `run_step3`；源码锚点 `}  // namespace task1`。

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
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step3` 所在的 `ZKX/Task1/task1_gpu_cpu/step3/step3.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 generate_step3_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`；符号 `generate_step3_cpu_reference`；源码锚点 `Step3ValidationData generate_step3_cpu_reference(const PipelineState& state)`。

```cpp
Step3ValidationData generate_step3_cpu_reference(const PipelineState& state)
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
|`generate_step3_cpu_reference`|生成当前名称所描述数据的 helper/结果|对应生成公式的输出端|由输入参数构造数据，供后续 step 或测试使用|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step3ValidationData generate_step3_cpu_reference(const PipelineState& state) {` 中的 `generate_step3_cpu_reference` 是函数定义；名称前的 `Step3ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|测试证据|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过定义/声明 `generate_step3_cpu_reference`；调用 `generate_step3_cpu_reference`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.19 generate_step3_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`；符号 `generate_step3_cpu_reference`；源码锚点 `Step3ValidationData data;`。

```cpp
    Step3ValidationData data;
    cusignal::PulseDopplerOptions options;
    options.num_pulses = config.num_pulses;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `data` 由 `Step3ValidationData` 声明：类型控制可表示值、可用操作和传参方式；
- `options` 由 `cusignal::PulseDopplerOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::PulseDopplerOptions options;`；值在 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step3ValidationData data;` 是对象声明：类型 `Step3ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `cusignal::PulseDopplerOptions options;` 是对象声明：类型 `cusignal::PulseDopplerOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `options.num_pulses = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象 `options.num_pulses`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|测试证据|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过写入/初始化 `num_pulses`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.20 generate_step3_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`；符号 `generate_step3_cpu_reference`；源码锚点 `options.samples_per_pulse = config.samples_per_pulse;`。

```cpp
    options.samples_per_pulse = config.samples_per_pulse;
    options.nfft = config.num_pulses;
    options.window = cusignal::RadarWindowKind::hamming;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `options`、`samples_per_pulse`、`config`、`nfft`、`num_pulses`、`window`、`cusignal`、`RadarWindowKind`、`hamming`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|`config` 在本 step 实例化慢时间脉冲数、距离单元数、Hamming 窗和 Doppler 输出布局。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.samples_per_pulse = config.samples_per_pulse;` 是赋值：先求得右侧完整表达式 `config.samples_per_pulse`，再把结果写入左侧可修改对象 `options.samples_per_pulse`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.nfft = config.num_pulses;` 是赋值：先求得右侧完整表达式 `config.num_pulses`，再把结果写入左侧可修改对象 `options.nfft`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.window = cusignal::RadarWindowKind::hamming;` 是赋值：先求得右侧完整表达式 `cusignal::RadarWindowKind::hamming`，再把结果写入左侧可修改对象 `options.window`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|测试证据|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过写入/初始化 `samples_per_pulse`, `nfft`, `window`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 窗化 FIR 通过有限冲激响应限制频带并保持线性相位可设计性；增加 taps 通常改善过渡带但增加卷积成本和群时延。IIR 可用更低阶数，但相位和稳定性分析不同。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.21 generate_step3_cpu_reference：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`；符号 `generate_step3_cpu_reference`；源码锚点 `data.range_doppler_cpu = radar_cpu_to_host(`。

```cpp
    data.range_doppler_cpu = radar_cpu_to_host(
        cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));
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

- `data.range_doppler_cpu = radar_cpu_to_host( cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options));` 是赋值：先求得右侧完整表达式 `radar_cpu_to_host( cusignal::pulse_doppler_typed_cpu<float>(state.compressed, options))`，再把结果写入左侧可修改对象 `data.range_doppler_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `return data;` 是返回语句：先求值 `data`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|测试证据|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块通过调用 `radar_cpu_to_host`；写入/初始化 `range_doppler_cpu`；返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.22 generate_step3_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step3_validation.cu`；符号 `generate_step3_cpu_reference`；源码锚点 `}`。

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
|赛题条款|T1-R3：多普勒处理：调用 `pulse_doppler` 提取目标多普勒频移|
|业务角色|测试证据|
|业务输入|T1-R2 的 `compressed`、脉冲数、每脉冲采样数和窗口选择|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|距离—多普勒结果 `doppler_map`/`doppler`|
|下一消费者|功率或幅值结果交给 T1-R4 CFAR|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
