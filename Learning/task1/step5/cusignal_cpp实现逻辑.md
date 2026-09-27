# Task1 Step5 cusignal_cpp 实现逻辑

## 1. 职责与版本

本文件负责 Task1 Step5 的二维模糊函数、零 Doppler/delay 切片、输出 dtype 适配、GPU 正式算子调用与独立 CPU reference；模糊函数算子内部算法归算子文档。

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
|`ZKX/Task1/task1_gpu_cpu/step5/step5.h`|`run_step5` 声明|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/step5/step5.cu`|`ambgfun_to_double`、`run_step5`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|
|`ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`|`to_double`、`generate_step5_cpu_reference`|`bb1a48c2e18fe2e252e0203e8be717d0c40f9251`|

正式算子内部归Learning/operators；其他文档只链接。

## 3. 完整相关源码

### 3.1 `ZKX/Task1/task1_gpu_cpu/step5/step5.h`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
//  `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
//  `namespace task1 { StepEvidence run_step5(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence
//  run_step5(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。
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
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step5(PipelineState& state); }
```

### 3.2 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp
// <学习注释：语义块：建立当前符号所需的依赖名称。
//
//  【执行方法】
//  `#include "step5.h"` 在预处理阶段引入 "step5.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
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
#include "step5.h"

#include "cuda_utils/device_array.h"

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

// <学习注释：语义块：ambgfun_to_double：检查条件并选择执行路径。
//
//  【执行方法】
//  `std::vector<double> ambgfun_to_double(const cusignal::AmbgfunDeviceResult& result) {` 中的
//  `ambgfun_to_double` 是函数定义；名称前的 `std::vector<double>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  cusignal::AmbgfunDeviceResult& result` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) {` 是 `if` 条件语句：先把完整条件 `result.dtype ==
//  cusignal::AmbgfunOutputDtype::fp32` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
//  `const auto values = result.fp32.to_host();` 是声明并初始化：`const auto values` 建立局部对象 `values`，右侧完整表达式
//  `result.fp32.to_host()` 产生初值。
//
//  【变量—数学符号—数据流】
//  result：当前调用返回或聚合得到的结果对象；对应当前算法/证据的输出；被赋值后由返回、比较或序列化消费
//  values：32 位整数混合器的中间状态；记作 $h$；由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散
//
//  【数字常量】
//  2：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
//
//
//  【数学/物理公式对照】
//  本块不改变 $χ$ 的定义，只把 fp32/fp64 结果统一为 Host `vector<double>`；fp32 扩宽为 double 不能恢复已舍入信息，shape 和元素顺序不变。
//
//  【为什么这样设计】
//  统一 Host dtype 简化验证/状态消费，同时显式保留实际输出分支；reinterpret 不能替代数值转换。
//
//  【初学者易错点】
//  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<double> ambgfun_to_double(const cusignal::AmbgfunDeviceResult& result)
{
    if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) {
        const auto values = result.fp32.to_host();
        // <学习注释：语义块：ambgfun_to_double：形成并返回当前结果。
        //
        //  【执行方法】
        //  `return std::vector<double>(values.begin(), values.end());` 是返回语句：先求值 `std::vector<double>(values.begin(),
        //  values.end())`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
        //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
        //  `return result.fp64.to_host();` 是返回语句：先求值 `result.fp64.to_host()`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
        //
        //  【数字常量】
        //  4：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
        //
        //
        //  【数学/物理公式对照】
        //  本块不改变 $χ$ 的定义，只把 fp32/fp64 结果统一为 Host `vector<double>`；fp32 扩宽为 double 不能恢复已舍入信息，shape 和元素顺序不变。
        //
        //  【为什么这样设计】
        //  统一 Host dtype 简化验证/状态消费，同时显式保留实际输出分支；reinterpret 不能替代数值转换。
        //
        //  【初学者易错点】
        //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
        return std::vector<double>(values.begin(), values.end());
    }
    return result.fp64.to_host();
// <学习注释：语义块：ambgfun_to_double：完成一个连续的数据处理动作。
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

// <学习注释：语义块：ambgfun_to_double：完成一个连续的数据处理动作。
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

// <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `StepEvidence run_step5(PipelineState& state) {` 中的 `run_step5` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的
//  CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  run_step5：运行当前 step 或流水线的入口函数/回调；无独立数学符号；组织配置、状态、算子调用和证据收尾
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
StepEvidence run_step5(PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `StepEvidence evidence;` 是对象声明：类型 `StepEvidence` 应用于名称 `evidence`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.name = "step5";` 是赋值：先求得右侧完整表达式 `"step5"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于
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
    evidence.name = "step5";
    const auto formal_begin = Clock::now();
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
    //  `require_task1_upstream( state.waveform.size() == static_cast<std::size_t>(config.pulse_samples), "Step5",
    //  "Step1.waveform");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //  config：外部配置对象；`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。；入口加载，runner 和各 step 只读消费
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
    require_task1_upstream(
            state.waveform.size() == static_cast<std::size_t>(config.pulse_samples),
            "Step5", "Step1.waveform");
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::AmbgfunOptions options;` 是对象声明：类型 `cusignal::AmbgfunOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `options.fs = config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `config.sample_rate_hz`，再把结果写入左侧可修改对象
    //  `options.fs`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.prf = config.prf_hz;` 是赋值：先求得右侧完整表达式 `config.prf_hz`，再把结果写入左侧可修改对象 `options.prf`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::AmbgfunOptions options;；值在
    //  ZKX/Task1/task1_gpu_cpu/step5/step5.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `fs=f_s` 给出延迟步长 $Δτ=1/f_s$，`prf` 给出 Doppler 轴语境；`cut` 选择二维面或固定变量的一维截面，`cut_value=0.0` 表示零切片。本块配置采样/模式，不执行积分或离散求和。
    //
    //  【为什么这样设计】
    //  复用同一 options 只切换 cut，可保持三种输出坐标定义一致；错误 `fs/prf` 会使物理轴错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::AmbgfunOptions options;
    options.fs = config.sample_rate_hz;
    options.prf = config.prf_hz;
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.cut = cusignal::AmbgfunCut::two_dimensional;` 是赋值：先求得右侧完整表达式
    //  `cusignal::AmbgfunCut::two_dimensional`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //
    //  【变量—数学符号—数据流】
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  `fs=f_s` 给出延迟步长 $Δτ=1/f_s$，`prf` 给出 Doppler 轴语境；`cut` 选择二维面或固定变量的一维截面，`cut_value=0.0` 表示零切片。本块配置采样/模式，不执行积分或离散求和。
    //
    //  【为什么这样设计】
    //  复用同一 options 只切换 cut，可保持三种输出坐标定义一致；错误 `fs/prf` 会使物理轴错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.cut = cusignal::AmbgfunCut::two_dimensional;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `auto d_waveform = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform);`
    //  是声明并初始化：`auto d_waveform` 建立局部对象 `d_waveform`，右侧完整表达式
    //  `cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform)` 产生初值。
    //  `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  d_waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
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
    auto d_waveform =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `cusignal::AmbgfunDeviceResult gpu_2d;` 是对象声明：类型 `cusignal::AmbgfunDeviceResult` 应用于名称
    //  `gpu_2d`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.operator_ms["ambgfun_2d"] = time_gpu([&] { cusignal::ambgfun_device<float>(d_waveform, gpu_2d,
    //  options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["ambgfun_2d"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  gpu_2d：GPU 路径的对象、结果或计时字段；与同名 CPU/Host 量采用相同数学定义；由 device 调用产生，供 D2H、comparison 或证据消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子计算离散自模糊函数，对应 $χ(τ,f_D)=\int s(t)s^*(t-τ)e^{-j2πf_Dt}dt$。`waveform/d_waveform→s[n]`，`fs→f_s`、`prf→PRF`。`two_dimensional` 返回二维面；`doppler,cut_value=0` 固定 $f_D=0$ 得 delay cut；`delay,cut_value=0` 固定 $τ=0$ 得 Doppler cut。结果 dtype 可为 fp32/fp64。
    //
    //  【为什么这样设计】
    //  二维面展示联合分辨能力，两条零切片分别检查 delay/Doppler 方向；内部相关、FFT 和归一化归 `Learning/operators/radartools/ambgfun/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    cusignal::AmbgfunDeviceResult gpu_2d;
    evidence.operator_ms["ambgfun_2d"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_2d, options);
    });
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.cut = cusignal::AmbgfunCut::doppler;` 是赋值：先求得右侧完整表达式
    //  `cusignal::AmbgfunCut::doppler`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.cut_value = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `options.cut_value`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `cusignal::AmbgfunDeviceResult gpu_delay;` 是对象声明：类型 `cusignal::AmbgfunDeviceResult` 应用于名称
    //  `gpu_delay`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //
    //  【变量—数学符号—数据流】
    //  gpu_delay：目标离散延迟，单位 sample；记作 $d$；回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$
    //
    //  【数字常量】
    //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  `fs=f_s` 给出延迟步长 $Δτ=1/f_s$，`prf` 给出 Doppler 轴语境；`cut` 选择二维面或固定变量的一维截面，`cut_value=0.0` 表示零切片。本块配置采样/模式，不执行积分或离散求和。
    //
    //  【为什么这样设计】
    //  复用同一 options 只切换 cut，可保持三种输出坐标定义一致；错误 `fs/prf` 会使物理轴错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.cut = cusignal::AmbgfunCut::doppler;
    options.cut_value = 0.0;
    cusignal::AmbgfunDeviceResult gpu_delay;
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `evidence.operator_ms["ambgfun_delay_cut"] = time_gpu([&] { cusignal::ambgfun_device<float>(d_waveform,
    //  gpu_delay, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["ambgfun_delay_cut"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子计算离散自模糊函数，对应 $χ(τ,f_D)=\int s(t)s^*(t-τ)e^{-j2πf_Dt}dt$。`waveform/d_waveform→s[n]`，`fs→f_s`、`prf→PRF`。`two_dimensional` 返回二维面；`doppler,cut_value=0` 固定 $f_D=0$ 得 delay cut；`delay,cut_value=0` 固定 $τ=0$ 得 Doppler cut。结果 dtype 可为 fp32/fp64。
    //
    //  【为什么这样设计】
    //  二维面展示联合分辨能力，两条零切片分别检查 delay/Doppler 方向；内部相关、FFT 和归一化归 `Learning/operators/radartools/ambgfun/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    evidence.operator_ms["ambgfun_delay_cut"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_delay, options);
    });
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.cut = cusignal::AmbgfunCut::delay;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::delay`，再把结果写入左侧可修改对象
    //  `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `cusignal::AmbgfunDeviceResult gpu_doppler;` 是对象声明：类型 `cusignal::AmbgfunDeviceResult` 应用于名称
    //  `gpu_doppler`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `evidence.operator_ms["ambgfun_doppler_cut"] = time_gpu([&] { cusignal::ambgfun_device<float>(d_waveform,
    //  gpu_doppler, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入
    //  `evidence.operator_ms["ambgfun_doppler_cut"]`。lambda 内部的赋值/调用才产生业务输出。
    //
    //  【变量—数学符号—数据流】
    //  gpu_doppler：GPU 路径的对象、结果或计时字段；与同名 CPU/Host 量采用相同数学定义；由 device 调用产生，供 D2H、comparison 或证据消费
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  正式算子计算离散自模糊函数，对应 $χ(τ,f_D)=\int s(t)s^*(t-τ)e^{-j2πf_Dt}dt$。`waveform/d_waveform→s[n]`，`fs→f_s`、`prf→PRF`。`two_dimensional` 返回二维面；`doppler,cut_value=0` 固定 $f_D=0$ 得 delay cut；`delay,cut_value=0` 固定 $τ=0$ 得 Doppler cut。结果 dtype 可为 fp32/fp64。
    //
    //  【为什么这样设计】
    //  二维面展示联合分辨能力，两条零切片分别检查 delay/Doppler 方向；内部相关、FFT 和归一化归 `Learning/operators/radartools/ambgfun/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.cut = cusignal::AmbgfunCut::delay;
    cusignal::AmbgfunDeviceResult gpu_doppler;
    evidence.operator_ms["ambgfun_doppler_cut"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_doppler, options);
    });
    // <学习注释：语义块：run_step5：形成并返回当前结果。
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
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
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
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `const auto gpu_2d_double = ambgfun_to_double(gpu_2d);` 是声明并初始化：`const auto gpu_2d_double` 建立局部对象
    //  `gpu_2d_double`，右侧完整表达式 `ambgfun_to_double(gpu_2d)` 产生初值。
    //  `state.ambiguity_2d.assign(gpu_2d_double.begin(), gpu_2d_double.end());` 通过对象/指针调用容器的
    //  `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。
    //
    //  【变量—数学符号—数据流】
    //  gpu_2d_double：GPU 路径的对象、结果或计时字段；与同名 CPU/Host 量采用相同数学定义；由 device 调用产生，供 D2H、comparison 或证据消费
    //  begin：当前计时子区间起点；无独立算法符号；时间戳可记作 $t_a$；与 Clock::now/Event 结束值相减
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //
    //  【数学/物理公式对照】
    //  本块不改变 $χ$ 的定义，只把 fp32/fp64 结果统一为 Host `vector<double>`；fp32 扩宽为 double 不能恢复已舍入信息，shape 和元素顺序不变。
    //
    //  【为什么这样设计】
    //  统一 Host dtype 简化验证/状态消费，同时显式保留实际输出分支；reinterpret 不能替代数值转换。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    begin = Clock::now();
    const auto gpu_2d_double = ambgfun_to_double(gpu_2d);
    state.ambiguity_2d.assign(gpu_2d_double.begin(), gpu_2d_double.end());
    // <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `state.ambiguity_delay = ambgfun_to_double(gpu_delay);` 是赋值：先求得右侧完整表达式
    //  `ambgfun_to_double(gpu_delay)`，再把结果写入左侧可修改对象 `state.ambiguity_delay`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `state.ambiguity_doppler = ambgfun_to_double(gpu_doppler);` 是赋值：先求得右侧完整表达式
    //  `ambgfun_to_double(gpu_doppler)`，再把结果写入左侧可修改对象 `state.ambiguity_doppler`；这里的 `=` 不属于 `>=`、`<=`、`==` 或
    //  `!=`。
    //  `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin,
    //  Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  evidence：当前步骤的证据记录对象；无独立数学符号；保存 prep/H2D/compute/D2H/指标并最终序列化
    //
    //  【数学/物理公式对照】
    //  本块不改变 $χ$ 的定义，只把 fp32/fp64 结果统一为 Host `vector<double>`；fp32 扩宽为 double 不能恢复已舍入信息，shape 和元素顺序不变。
    //
    //  【为什么这样设计】
    //  统一 Host dtype 简化验证/状态消费，同时显式保留实际输出分支；reinterpret 不能替代数值转换。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    state.ambiguity_delay = ambgfun_to_double(gpu_delay);
    state.ambiguity_doppler = ambgfun_to_double(gpu_doppler);
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    // <学习注释：语义块：run_step5：形成并返回当前结果。
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

// <学习注释：语义块：run_step5：完成一个连续的数据处理动作。
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

### 3.3 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`：`to_double`、`generate_step5_cpu_reference`

完整SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`。

```cpp

// <学习注释：语义块：to_double：检查条件并选择执行路径。
//
//  【执行方法】
//  `std::vector<double> to_double(const cusignal::AmbgfunCpuResult& result) {` 中的 `to_double` 是函数定义；名称前的
//  `std::vector<double>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const cusignal::AmbgfunCpuResult& result` 是形参声明。
//  末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) return std::vector<double>(result.fp32.begin(),
//  result.fp32.end());` 是 `if` 条件语句：先把完整条件 `result.dtype == cusignal::AmbgfunOutputDtype::fp32` 求值并转换为布尔值。
//  条件为真时执行 `return`，即求值 `std::vector<double>(result.fp32.begin(), result.fp32.end())`、把它作为返回值并结束当前函数；条件为假时跳过
//  `return` 并继续下一条语句。这不是赋值，也不是循环。
//
//  【变量—数学符号—数据流】
//  result：当前调用返回或聚合得到的结果对象；对应当前算法/证据的输出；被赋值后由返回、比较或序列化消费
//
//  【数字常量】
//  2：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
//
//
//  【数学/物理公式对照】
//  本块不改变 $χ$ 的定义，只把 fp32/fp64 结果统一为 Host `vector<double>`；fp32 扩宽为 double 不能恢复已舍入信息，shape 和元素顺序不变。
//
//  【为什么这样设计】
//  统一 Host dtype 简化验证/状态消费，同时显式保留实际输出分支；reinterpret 不能替代数值转换。
//
//  【初学者易错点】
//  `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
//  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
std::vector<double> to_double(const cusignal::AmbgfunCpuResult& result)
{
    if (result.dtype == cusignal::AmbgfunOutputDtype::fp32)
        return std::vector<double>(result.fp32.begin(), result.fp32.end());
    // <学习注释：语义块：to_double：形成并返回当前结果。
    //
    //  【执行方法】
    //  `return result.fp64;` 是返回语句：先求值 `result.fp64`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
    //  `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
    //
    //  【数字常量】
    //  4：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。
    //
    //
    //  【数学/物理公式对照】
    //  本块不改变 $χ$ 的定义，只把 fp32/fp64 结果统一为 Host `vector<double>`；fp32 扩宽为 double 不能恢复已舍入信息，shape 和元素顺序不变。
    //
    //  【为什么这样设计】
    //  统一 Host dtype 简化验证/状态消费，同时显式保留实际输出分支；reinterpret 不能替代数值转换。
    //
    //  【初学者易错点】
    //  `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。>
    return result.fp64;
}

// <学习注释：语义块：to_double：完成一个连续的数据处理动作。
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

// <学习注释：语义块：generate_step5_cpu_reference：完成一个连续的数据处理动作。
//
//  【执行方法】
//  `Step5ValidationData generate_step5_cpu_reference(const PipelineState& state) {` 中的
//  `generate_step5_cpu_reference` 是函数定义；名称前的 `Step5ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const
//  PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
//  `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config`
//  产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。
//
//  【变量—数学符号—数据流】
//  generate_step5_cpu_reference：生成当前名称所描述数据的 helper/结果；对应生成公式的输出端；由输入参数构造数据，供后续 step 或测试使用
//  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
//  config：外部配置对象；`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。；入口加载，runner 和各 step 只读消费
//
//  【数学/物理公式对照】
//  本块没有独立数学变换；它只建立声明、类型、作用域、资源所有权、检查或控制流边界。应按紧邻源码理解其如何准备输入或承接输出，不能把工程动作本身解释成一次信号处理运算。
//
//  【为什么这样设计】
//  显式接口、状态和资源边界使 dtype、shape、所有权、失败位置与数据依赖可审计；源码未记录其他动机时不额外猜测。
//
//  【初学者易错点】
//  声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。>
Step5ValidationData generate_step5_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
    // <学习注释：语义块：generate_step5_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `Step5ValidationData data;` 是对象声明：类型 `Step5ValidationData` 应用于名称
    //  `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `cusignal::AmbgfunOptions options;` 是对象声明：类型 `cusignal::AmbgfunOptions` 应用于名称
    //  `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
    //  `options.fs = config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `config.sample_rate_hz`，再把结果写入左侧可修改对象
    //  `options.fs`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  data：用于携带当前步骤输入/CPU reference/验证结果的数据结构；无单一数学符号；字段分别映射到算法数组；由生成函数填充并交给 comparison
    //  options：当前表达式读取或传递的工程名称 options；源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定；直接证据 cusignal::AmbgfunOptions options;；值在
    //  ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu 当前作用域中产生或消费
    //  config：外部配置对象；`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。；入口加载，runner 和各 step 只读消费
    //
    //  【数学/物理公式对照】
    //  `fs=f_s` 给出延迟步长 $Δτ=1/f_s$，`prf` 给出 Doppler 轴语境；`cut` 选择二维面或固定变量的一维截面，`cut_value=0.0` 表示零切片。本块配置采样/模式，不执行积分或离散求和。
    //
    //  【为什么这样设计】
    //  复用同一 options 只切换 cut，可保持三种输出坐标定义一致；错误 `fs/prf` 会使物理轴错误。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    Step5ValidationData data;
    cusignal::AmbgfunOptions options;
    options.fs = config.sample_rate_hz;
    // <学习注释：语义块：generate_step5_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.prf = config.prf_hz;` 是赋值：先求得右侧完整表达式 `config.prf_hz`，再把结果写入左侧可修改对象 `options.prf`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `options.cut = cusignal::AmbgfunCut::two_dimensional;` 是赋值：先求得右侧完整表达式
    //  `cusignal::AmbgfunCut::two_dimensional`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `data.ambiguity_2d_cpu = to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options));`
    //  是赋值：先求得右侧完整表达式 `to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options))`，再把结果写入左侧可修改对象
    //  `data.ambiguity_2d_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  config：外部配置对象；`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。；入口加载，runner 和各 step 只读消费
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //
    //  【数学/物理公式对照】
    //  正式算子计算离散自模糊函数，对应 $χ(τ,f_D)=\int s(t)s^*(t-τ)e^{-j2πf_Dt}dt$。`waveform/d_waveform→s[n]`，`fs→f_s`、`prf→PRF`。`two_dimensional` 返回二维面；`doppler,cut_value=0` 固定 $f_D=0$ 得 delay cut；`delay,cut_value=0` 固定 $τ=0$ 得 Doppler cut。结果 dtype 可为 fp32/fp64。
    //
    //  【为什么这样设计】
    //  二维面展示联合分辨能力，两条零切片分别检查 delay/Doppler 方向；内部相关、FFT 和归一化归 `Learning/operators/radartools/ambgfun/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.prf = config.prf_hz;
    options.cut = cusignal::AmbgfunCut::two_dimensional;
    data.ambiguity_2d_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
    // <学习注释：语义块：generate_step5_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.cut = cusignal::AmbgfunCut::doppler;` 是赋值：先求得右侧完整表达式
    //  `cusignal::AmbgfunCut::doppler`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `options.cut_value = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `options.cut_value`；这里的 `=` 不属于
    //  `>=`、`<=`、`==` 或 `!=`。
    //  `data.ambiguity_delay_cpu = to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options));`
    //  是赋值：先求得右侧完整表达式 `to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options))`，再把结果写入左侧可修改对象
    //  `data.ambiguity_delay_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //
    //  【数字常量】
    //  0.0：无显式后缀，类型按语言的整型/浮点字面量默认规则确定；零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。
    //
    //  【数学/物理公式对照】
    //  正式算子计算离散自模糊函数，对应 $χ(τ,f_D)=\int s(t)s^*(t-τ)e^{-j2πf_Dt}dt$。`waveform/d_waveform→s[n]`，`fs→f_s`、`prf→PRF`。`two_dimensional` 返回二维面；`doppler,cut_value=0` 固定 $f_D=0$ 得 delay cut；`delay,cut_value=0` 固定 $τ=0$ 得 Doppler cut。结果 dtype 可为 fp32/fp64。
    //
    //  【为什么这样设计】
    //  二维面展示联合分辨能力，两条零切片分别检查 delay/Doppler 方向；内部相关、FFT 和归一化归 `Learning/operators/radartools/ambgfun/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.cut = cusignal::AmbgfunCut::doppler;
    options.cut_value = 0.0;
    data.ambiguity_delay_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
    // <学习注释：语义块：generate_step5_cpu_reference：完成一个连续的数据处理动作。
    //
    //  【执行方法】
    //  `options.cut = cusignal::AmbgfunCut::delay;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::delay`，再把结果写入左侧可修改对象
    //  `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //  `data.ambiguity_doppler_cpu = to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options));`
    //  是赋值：先求得右侧完整表达式 `to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options))`，再把结果写入左侧可修改对象
    //  `data.ambiguity_doppler_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
    //
    //  【变量—数学符号—数据流】
    //  state：任务流水线共享状态；无单一数学符号；保存各 step 的数组与证据；前一步写入、后一步读取
    //  waveform：发射/参考复波形数组；记作 $s[n]$；Step1 生成，回波模拟和脉冲压缩消费
    //
    //  【数学/物理公式对照】
    //  正式算子计算离散自模糊函数，对应 $χ(τ,f_D)=\int s(t)s^*(t-τ)e^{-j2πf_Dt}dt$。`waveform/d_waveform→s[n]`，`fs→f_s`、`prf→PRF`。`two_dimensional` 返回二维面；`doppler,cut_value=0` 固定 $f_D=0$ 得 delay cut；`delay,cut_value=0` 固定 $τ=0$ 得 Doppler cut。结果 dtype 可为 fp32/fp64。
    //
    //  【为什么这样设计】
    //  二维面展示联合分辨能力，两条零切片分别检查 delay/Doppler 方向；内部相关、FFT 和归一化归 `Learning/operators/radartools/ambgfun/`。
    //
    //  【初学者易错点】
    //  不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。>
    options.cut = cusignal::AmbgfunCut::delay;
    data.ambiguity_doppler_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
    // <学习注释：语义块：generate_step5_cpu_reference：形成并返回当前结果。
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

`Step1 waveform→ambgfun 二维/切片→D2H`。形成模糊函数指标与图形数据。

#### 3.4.2 主要函数职责

|符号|处理职责|CPU/GPU/业务位置|
|---|---|---|
|`ambgfun_to_double`|按结果 dtype 选择 fp32/fp64 device buffer，并统一转换成 Host double 向量|Host dtype adapter|
|`run_step5`|配置并调用二维、delay cut、Doppler cut 模糊函数，回收结果与计时|GPU Host wrapper/业务入口|
|`to_double`|把算子不同输出 dtype 统一成验证侧 double 数组|CPU validation adapter|
|`generate_step5_cpu_reference`|用同一参数生成模糊函数 CPU reference|CPU validation entry|

#### 3.4.3 CPU/GPU 等价关系与证据边界

- CPU reference 必须使用与 GPU 相同的输入参数和数学定义，但通过独立 Host 路径计算，避免把 GPU 输出回读后冒充 reference。
- GPU Host wrapper 负责资源、shape、H2D/D2H、计时和 kernel/算子调度；device kernel 负责线程对应的数值运算。
- validation/comparison 只能证明验证方法和指标口径；历史运行结论仍须绑定正式证据、配置与源码 SHA。

#### 算子数学原理在当前任务中的实例化

##### `ambgfun`

- 学习入口：[数学物理原理](../../operators/radartools/ambgfun/数学物理原理.md)、[Python 源码算法](../../operators/radartools/ambgfun/python源码算法.md)、[cusignal_cpp 复现逻辑](../../operators/radartools/ambgfun/cusignal_cpp_ambgfun复现逻辑.md)。
- 当前调用采用的数学关系：$\chi(\tau,f_D)=\int s(t)s^*(t-\tau)e^{-j2\pi f_Dt}\,dt$；离散实现对 delay/Doppler 网格求相关。
- 任务变量与参数实例化：`waveform→s`、采样率决定 delay 轴、PRF/多普勒配置决定 Doppler 轴，输出 ambiguity surface/cut。
- 边界：这里只解释任务调用处的原理和变量映射；算子 Python/C++ 内部实现以链接文档为准，不在本任务文档重复。

#### 3.4.4 三次正式调用的模式与数据关系

三次 `ambgfun_*<float>` 使用同一复波形 $s[n]$ 和同一 `fs/prf`，对应离散自模糊函数
$\chi(\tau,f_D)=\int s(t)s^*(t-\tau)e^{-j2\pi f_Dt}dt$ 的不同输出模式。

|`options.cut`|`cut_value`|输出含义|任务状态|
|---|---:|---|---|
|`two_dimensional`|不使用|二维 $|\chi(\tau,f_D)|$|`ambiguity_2d`|
|`doppler`|`0.0`|固定 $f_D=0$ 的 delay cut|`ambiguity_delay`|
|`delay`|沿用 `0.0`|固定 $\tau=0$ 的 Doppler cut|`ambiguity_doppler`|

输入是长度 $L$ 的 `TypedComplex<float>`；算子结果可能为 fp32 或 fp64。`ambgfun_to_double`/`to_double` 只统一 Host dtype，不改变 shape、元素顺序或模糊函数定义；fp32 扩宽为 double 不能恢复已经舍入的精度。

## 4. 按源码顺序逐语义块深入解释

本章不使用行号。每个代码块均以完整 SHA、路径、符号名和源码原文作为锚点，并按“语法结构—名称与类型—执行过程—任务语义—初学者易错点”讲解。

### 4.1 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.h`；符号 `文件级代码`；源码锚点 `#pragma once`。

```cpp
#pragma once
#include "../task1_common.h"
namespace task1 { StepEvidence run_step5(PipelineState& state); }
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：预处理指令、命名空间声明。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

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

- `#pragma once` 是预处理指令，要求同一翻译单元只展开该头文件一次；它不产生运行时计算。
- `#include "../task1_common.h"` 在预处理阶段引入 "../task1_common.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `namespace task1 { StepEvidence run_step5(PipelineState& state); }` 在命名空间 `task1` 内声明 `StepEvidence run_step5(PipelineState& state);`，随后同一行的 `}` 立即关闭命名空间；这里只建立名称归属，不调用函数。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|公共基础设施|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.2 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `文件级代码`；源码锚点 `#include "step5.h"`。

```cpp
#include "step5.h"

#include "cuda_utils/device_array.h"
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：预处理指令。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `include`、`step5`、`h`、`cuda_utils`、`device_array`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `#include "step5.h"` 在预处理阶段引入 "step5.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。
- `#include "cuda_utils/device_array.h"` 在预处理阶段引入 "cuda_utils/device_array.h" 的声明或定义，使本文件能使用其中的符号；它本身不是运行时函数调用。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|公共基础设施|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.3 建立当前符号所需的依赖名称

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `文件级代码`；源码锚点 `#include <numeric>`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只建立头文件依赖、声明和命名空间归属，不执行当前 step 的数值算法。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `文件级代码` 所在的 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.4 ambgfun_to_double：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `ambgfun_to_double`；源码锚点 `std::vector<double> ambgfun_to_double(const cusignal::AmbgfunDeviceResult& result)`。

```cpp
std::vector<double> ambgfun_to_double(const cusignal::AmbgfunDeviceResult& result)
{
    if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) {
        const auto values = result.fp32.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：函数定义或声明、`if` 条件语句、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `result` 由 `const cusignal::AmbgfunDeviceResult&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权；
- `values` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|
|`values`|32 位整数混合器的中间状态|记作 $h$|由 seed 与 index 混合；连续移位、异或和乘法提高比特扩散|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|

**执行过程**

- `std::vector<double> ambgfun_to_double(const cusignal::AmbgfunDeviceResult& result) {` 中的 `ambgfun_to_double` 是函数定义；名称前的 `std::vector<double>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const cusignal::AmbgfunDeviceResult& result` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) {` 是 `if` 条件语句：先把完整条件 `result.dtype == cusignal::AmbgfunOutputDtype::fp32` 求值并转换为布尔值。 条件为真时进入随后花括号内的分支；为假时跳过该分支。这不是循环。
- `const auto values = result.fp32.to_host();` 是声明并初始化：`const auto values` 建立局部对象 `values`，右侧完整表达式 `result.fp32.to_host()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过定义/声明 `ambgfun_to_double`；检查 `result.dtype == cusignal::AmbgfunOutputDtype::fp32`；调用 `ambgfun_to_double`, `to_host`；写入/初始化 `values`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块生成模糊函数或切片，用于评价延迟与多普勒分辨能力，结果随后由证据或可视化层消费。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.5 ambgfun_to_double：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `ambgfun_to_double`；源码锚点 `return std::vector<double>(values.begin(), values.end());`。

```cpp
        return std::vector<double>(values.begin(), values.end());
    }
    return result.fp64.to_host();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `std`、`vector`、`double`、`values`、`begin`、`end`、`result`、`fp64`、`to_host`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|

**执行过程**

- `return std::vector<double>(values.begin(), values.end());` 是返回语句：先求值 `std::vector<double>(values.begin(), values.end())`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。
- `return result.fp64.to_host();` 是返回语句：先求值 `result.fp64.to_host()`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `begin`, `end`, `to_host`；返回 `std::vector<double>(values.begin(), values.end())`, `result.fp64.to_host()`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块生成模糊函数或切片，用于评价延迟与多普勒分辨能力，结果随后由证据或可视化层消费。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.6 ambgfun_to_double：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `ambgfun_to_double`；源码锚点 `}`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块生成模糊函数或切片，用于评价延迟与多普勒分辨能力，结果随后由证据或可视化层消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.7 ambgfun_to_double：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `ambgfun_to_double`；源码锚点 `}  // namespace`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块生成模糊函数或切片，用于评价延迟与多普勒分辨能力，结果随后由证据或可视化层消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.8 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `StepEvidence run_step5(PipelineState& state)`。

```cpp
StepEvidence run_step5(PipelineState& state)
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
|`run_step5`|运行当前 step 或流水线的入口函数/回调|无独立数学符号|组织配置、状态、算子调用和证据收尾|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `StepEvidence run_step5(PipelineState& state) {` 中的 `run_step5` 是函数定义；名称前的 `StepEvidence` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|输入准备/数据契约|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过定义/声明 `run_step5`；调用 `run_step5`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.9 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `StepEvidence evidence;`。

```cpp
    StepEvidence evidence;
    evidence.name = "step5";
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
- `evidence.name = "step5";` 是赋值：先求得右侧完整表达式 `"step5"`，再把结果写入左侧可修改对象 `evidence.name`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto formal_begin = Clock::now();` 是声明并初始化：`const auto formal_begin` 建立局部对象 `formal_begin`，右侧完整表达式 `Clock::now()` 产生初值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `Clock::now`；写入/初始化 `name`, `formal_begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.10 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `auto begin = Clock::now();`。

```cpp
    auto begin = Clock::now();
    require_task1_upstream(
            state.waveform.size() == static_cast<std::size_t>(config.pulse_samples),
            "Step5", "Step1.waveform");
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
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`config`|外部配置对象|`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto begin = Clock::now();` 是声明并初始化：`auto begin` 建立局部对象 `begin`，右侧完整表达式 `Clock::now()` 产生初值。
- `require_task1_upstream( state.waveform.size() == static_cast<std::size_t>(config.pulse_samples), "Step5", "Step1.waveform");` 调用任务上游契约检查；第一个实参必须为真，否则使用后续 step/字段名称报告缺失或非法输入。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|输入准备/数据契约|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `Clock::now`, `require_task1_upstream`, `size`；写入/初始化 `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.11 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `cusignal::AmbgfunOptions options;`。

```cpp
    cusignal::AmbgfunOptions options;
    options.fs = config.sample_rate_hz;
    options.prf = config.prf_hz;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `options` 由 `cusignal::AmbgfunOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::AmbgfunOptions options;`；值在 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::AmbgfunOptions options;` 是对象声明：类型 `cusignal::AmbgfunOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `options.fs = config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `config.sample_rate_hz`，再把结果写入左侧可修改对象 `options.fs`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.prf = config.prf_hz;` 是赋值：先求得右侧完整表达式 `config.prf_hz`，再把结果写入左侧可修改对象 `options.prf`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|输入准备/数据契约|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过写入/初始化 `fs`, `prf`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块证明业务参数可被加载、校验和传递；具体 demo 数值不等于赛题固定输入，也不能单独证明算法步骤完成。|

**任务语义**

本块属于输入配置与数据契约：它把外部字段转换成有类型的运行参数，后续 runner 或 step 读取这些值决定 shape、采样率、阈值和执行规模。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.12 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `options.cut = cusignal::AmbgfunCut::two_dimensional;`。

```cpp
    options.cut = cusignal::AmbgfunCut::two_dimensional;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `options`、`cut`、`cusignal`、`AmbgfunCut`、`two_dimensional`、`evidence`、`prep_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.cut = cusignal::AmbgfunCut::two_dimensional;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::two_dimensional`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.prep_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.prep_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `cut`, `prep_ms`, `begin`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块生成模糊函数或切片，用于评价延迟与多普勒分辨能力，结果随后由证据或可视化层消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.13 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `auto d_waveform =`。

```cpp
    auto d_waveform =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `d_waveform` 由 `auto` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`d_waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `auto d_waveform = cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform);` 是声明并初始化：`auto d_waveform` 建立局部对象 `d_waveform`，右侧完整表达式 `cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform)` 产生初值。
- `evidence.h2d_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.h2d_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|公共基础设施|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `from_host`, `milliseconds`, `Clock::now`；写入/初始化 `d_waveform`, `h2d_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块证明业务数据能安全搬运、同步、计时或持有；它没有独立数学业务输出，不能冒充核心算法。|

**任务语义**

本块属于公共基础设施：它管理 Host/device 缓冲区、复制、同步、错误或共享状态，为多个 step 提供资源与生命周期保证。

**设计理由与替代方案**

- 显式 H2D/D2H 和 RAII 缓冲区把数据驻留位置与所有权写清楚，便于分别统计传输和计算。统一托管内存是替代方案，但迁移时机更隐式，不利于当前证据口径。

**初学者易错点**

- 模板的 `<...>` 与嵌套闭合 `>>` 不是大小比较或右移，必须先由声明/类型上下文识别模板。

### 4.14 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `cusignal::AmbgfunDeviceResult gpu_2d;`。

```cpp
    cusignal::AmbgfunDeviceResult gpu_2d;
    evidence.operator_ms["ambgfun_2d"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_2d, options);
    });
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `gpu_2d` 由 `cusignal::AmbgfunDeviceResult` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`gpu_2d`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `cusignal::AmbgfunDeviceResult gpu_2d;` 是对象声明：类型 `cusignal::AmbgfunDeviceResult` 应用于名称 `gpu_2d`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `evidence.operator_ms["ambgfun_2d"] = time_gpu([&] { cusignal::ambgfun_device<float>(d_waveform, gpu_2d, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["ambgfun_2d"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `time_gpu`；写入/初始化 `operator_ms["ambgfun_2d"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.15 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `options.cut = cusignal::AmbgfunCut::doppler;`。

```cpp
    options.cut = cusignal::AmbgfunCut::doppler;
    options.cut_value = 0.0;
    cusignal::AmbgfunDeviceResult gpu_delay;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `gpu_delay` 由 `cusignal::AmbgfunDeviceResult` 声明：类型控制可表示值、可用操作和传参方式；
- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`gpu_delay`|目标离散延迟，单位 sample|记作 $d$|回波读取 $s[n-d]$；对应距离约 $R=cd/(2f_s)$|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `options.cut = cusignal::AmbgfunCut::doppler;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::doppler`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.cut_value = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `options.cut_value`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::AmbgfunDeviceResult gpu_delay;` 是对象声明：类型 `cusignal::AmbgfunDeviceResult` 应用于名称 `gpu_delay`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过写入/初始化 `cut`, `cut_value`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.16 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `evidence.operator_ms["ambgfun_delay_cut"] = time_gpu([&] {`。

```cpp
    evidence.operator_ms["ambgfun_delay_cut"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_delay, options);
    });
```

**语法结构**

该代码块按源码顺序包含 1 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `evidence`、`operator_ms`、`ambgfun_delay_cut`、`time_gpu`、`cusignal`、`ambgfun_device`、`float`、`d_waveform`、`gpu_delay`、`options`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `evidence.operator_ms["ambgfun_delay_cut"] = time_gpu([&] { cusignal::ambgfun_device<float>(d_waveform, gpu_delay, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["ambgfun_delay_cut"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `time_gpu`；写入/初始化 `operator_ms["ambgfun_delay_cut"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.17 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `options.cut = cusignal::AmbgfunCut::delay;`。

```cpp
    options.cut = cusignal::AmbgfunCut::delay;
    cusignal::AmbgfunDeviceResult gpu_doppler;
    evidence.operator_ms["ambgfun_doppler_cut"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_doppler, options);
    });
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、表达式/声明单元。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `gpu_doppler` 由 `cusignal::AmbgfunDeviceResult` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`gpu_doppler`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.cut = cusignal::AmbgfunCut::delay;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::delay`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `cusignal::AmbgfunDeviceResult gpu_doppler;` 是对象声明：类型 `cusignal::AmbgfunDeviceResult` 应用于名称 `gpu_doppler`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `evidence.operator_ms["ambgfun_doppler_cut"] = time_gpu([&] { cusignal::ambgfun_device<float>(d_waveform, gpu_doppler, options); });` 先创建按引用捕获当前作用域对象的 lambda `[&] {...}`；`time_gpu` 执行并计量其中的 cusignal 算子，返回的毫秒值写入 `evidence.operator_ms["ambgfun_doppler_cut"]`。lambda 内部的赋值/调用才产生业务输出。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `time_gpu`；写入/初始化 `cut`, `operator_ms["ambgfun_doppler_cut"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块参与波形或回波构造：配置/样本索引进入波形与噪声计算，输出复数样本供后续滤波、压缩或特征处理消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.18 run_step5：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `evidence.compute_ms = std::accumulate(`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `std::accumulate`, `begin`, `end`；写入/初始化 `compute_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.19 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `std::size_t free_bytes = 0, total_bytes = 0;`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `CUDA_CHECK`, `cudaMemGetInfo`；写入/初始化 `free_bytes`, `metrics["gpu_used_peak_bytes"]`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块从融合特征中定位离散关键点，输出索引或峰值集合供参数估计使用。

**设计理由与替代方案**

- 显式 `static_cast` 把类型转换写在源码中，避免依赖隐式转换并让有符号/无符号、整数/浮点边界可审查。替代的 C 风格强制转换可执行更多危险转换，因此这里更难审计。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.20 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `begin = Clock::now();`。

```cpp
    begin = Clock::now();
    const auto gpu_2d_double = ambgfun_to_double(gpu_2d);
    state.ambiguity_2d.assign(gpu_2d_double.begin(), gpu_2d_double.end());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句、对象构造或函数调用语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `gpu_2d_double` 由 `const auto` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`gpu_2d_double`|GPU 路径的对象、结果或计时字段|与同名 CPU/Host 量采用相同数学定义|由 device 调用产生，供 D2H、comparison 或证据消费|
|`begin`|当前计时子区间起点|无独立算法符号；时间戳可记作 $t_a$|与 Clock::now/Event 结束值相减|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `begin = Clock::now();` 是赋值：先求得右侧完整表达式 `Clock::now()`，再把结果写入左侧可修改对象 `begin`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `const auto gpu_2d_double = ambgfun_to_double(gpu_2d);` 是声明并初始化：`const auto gpu_2d_double` 建立局部对象 `gpu_2d_double`，右侧完整表达式 `ambgfun_to_double(gpu_2d)` 产生初值。
- `state.ambiguity_2d.assign(gpu_2d_double.begin(), gpu_2d_double.end());` 通过对象/指针调用容器的 `assign`，用给定数量/区间替换容器全部内容；旧元素被丢弃，容器长度随新范围确定。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `Clock::now`, `ambgfun_to_double`, `assign`, `begin`, `end`；写入/初始化 `begin`, `gpu_2d_double`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块生成模糊函数或切片，用于评价延迟与多普勒分辨能力，结果随后由证据或可视化层消费。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.21 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `state.ambiguity_delay = ambgfun_to_double(gpu_delay);`。

```cpp
    state.ambiguity_delay = ambgfun_to_double(gpu_delay);
    state.ambiguity_doppler = ambgfun_to_double(gpu_doppler);
    evidence.d2h_ms = milliseconds(begin, Clock::now());
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `state`、`ambiguity_delay`、`ambgfun_to_double`、`gpu_delay`、`ambiguity_doppler`、`gpu_doppler`、`evidence`、`d2h_ms`、`milliseconds`、`begin`、`Clock`、`now`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`evidence`|当前步骤的证据记录对象|无独立数学符号|保存 prep/H2D/compute/D2H/指标并最终序列化|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `state.ambiguity_delay = ambgfun_to_double(gpu_delay);` 是赋值：先求得右侧完整表达式 `ambgfun_to_double(gpu_delay)`，再把结果写入左侧可修改对象 `state.ambiguity_delay`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `state.ambiguity_doppler = ambgfun_to_double(gpu_doppler);` 是赋值：先求得右侧完整表达式 `ambgfun_to_double(gpu_doppler)`，再把结果写入左侧可修改对象 `state.ambiguity_doppler`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `evidence.d2h_ms = milliseconds(begin, Clock::now());` 是赋值：先求得右侧完整表达式 `milliseconds(begin, Clock::now())`，再把结果写入左侧可修改对象 `evidence.d2h_ms`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `ambgfun_to_double`, `milliseconds`, `Clock::now`；写入/初始化 `ambiguity_delay`, `ambiguity_doppler`, `d2h_ms`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块处理慢时间脉冲序列并形成多普勒谱；输出的距离–速度数据是 CFAR 或后续评价的输入。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.22 run_step5：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `milliseconds`, `Clock::now`；写入/初始化 `formal_execution_ms`；返回 `evidence`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.23 run_step5：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu`；符号 `run_step5`；源码锚点 `}  // namespace task1`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|核心算法或本步数据准备|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块可以证明所列调用、公式或状态写入确实落实了该条款的一部分；完整满足仍需结合本 step 其余块、前置输入契约和测试证据。|

**任务语义**

本块由 `run_step5` 所在的 `ZKX/Task1/task1_gpu_cpu/step5/step5.cu` 负责。它不单独定义新的数学算法，而是把当前对象、参数或控制状态连接到同一文档所述调用链；具体输出由紧随其后的赋值、返回或缓冲区写入确定。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.24 to_double：检查条件并选择执行路径

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `to_double`；源码锚点 `std::vector<double> to_double(const cusignal::AmbgfunCpuResult& result)`。

```cpp

std::vector<double> to_double(const cusignal::AmbgfunCpuResult& result)
{
    if (result.dtype == cusignal::AmbgfunOutputDtype::fp32)
        return std::vector<double>(result.fp32.begin(), result.fp32.end());
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：函数定义或声明、`if` 条件语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 当前块中的 `if (...) return;` 必须先作为完整条件语句识别；比较运算符 `>=`/`<=` 中的 `=` 不是赋值。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `result` 由 `const cusignal::AmbgfunCpuResult&` 声明：类型控制可表示值、可用操作和传参方式；`const` 禁止通过该名称修改对象；`&` 说明它是现有对象的别名，不复制对象也不取得所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`result`|当前调用返回或聚合得到的结果对象|对应当前算法/证据的输出|被赋值后由返回、比较或序列化消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`2`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|

**执行过程**

- `std::vector<double> to_double(const cusignal::AmbgfunCpuResult& result) {` 中的 `to_double` 是函数定义；名称前的 `std::vector<double>` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const cusignal::AmbgfunCpuResult& result` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) return std::vector<double>(result.fp32.begin(), result.fp32.end());` 是 `if` 条件语句：先把完整条件 `result.dtype == cusignal::AmbgfunOutputDtype::fp32` 求值并转换为布尔值。 条件为真时执行 `return`，即求值 `std::vector<double>(result.fp32.begin(), result.fp32.end())`、把它作为返回值并结束当前函数；条件为假时跳过 `return` 并继续下一条语句。这不是赋值，也不是循环。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过定义/声明 `to_double`；检查 `result.dtype == cusignal::AmbgfunOutputDtype::fp32`；调用 `to_double`, `begin`, `end`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 条件分支用于在访问数组、执行除法或继续流水线前验证边界/契约。删除它可能产生越界、非法 shape、错误证据或未定义行为；替代方案是调用前精确裁剪 grid/输入，但仍通常保留防御检查。

**初学者易错点**

- `if` 是条件选择，不是循环；条件为假时只跳过直接子语句。`if (...) return;` 的 `return` 只在条件为真时执行。
- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。
- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.25 to_double：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `to_double`；源码锚点 `return result.fp64;`。

```cpp
    return result.fp64;
}
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：`return` 跳转语句、作用域边界。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `result`、`fp64`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

当前块只含注释、闭合符号或构建命令，没有新出现的任务运行时变量；因此不存在可映射的数学变量。

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`4`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|该值恰为 2 的幂；但仅凭字面量不能断言它服务于 FFT 或线程配置。当前作用必须以紧邻源码表达式为准；源码没有证明它是唯一最优值。|

**执行过程**

- `return result.fp64;` 是返回语句：先求值 `result.fp64`，以它初始化调用者接收的返回结果，然后结束当前函数；它不是变量声明。
- `}` 的右花括号关闭此前打开的最近一层作用域；离开函数作用域时局部自动对象按逆构造顺序析构。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过返回 `result.fp64`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

### 4.26 to_double：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `to_double`；源码锚点 `}  // namespace`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块只关闭/建立词法作用域，不产生新的业务数组或数值结果。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.27 generate_step5_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `generate_step5_cpu_reference`；源码锚点 `Step5ValidationData generate_step5_cpu_reference(const PipelineState& state)`。

```cpp
Step5ValidationData generate_step5_cpu_reference(const PipelineState& state)
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
|`generate_step5_cpu_reference`|生成当前名称所描述数据的 helper/结果|对应生成公式的输出端|由输入参数构造数据，供后续 step 或测试使用|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`config`|外部配置对象|`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step5ValidationData generate_step5_cpu_reference(const PipelineState& state) {` 中的 `generate_step5_cpu_reference` 是函数定义；名称前的 `Step5ValidationData` 包含返回类型以及源码实际写出的 CUDA/存储限定符，圆括号中的 `const PipelineState& state` 是形参声明。 末尾 `{` 打开函数体；这里定义函数但不会在定义时自动执行。
- `const auto& config = state.config;` 是声明并初始化：`const auto& config` 建立局部对象 `config`，右侧完整表达式 `state.config` 产生初值。 声明符中的 `&` 表示引用别名，不是按位与；该名称绑定现有对象而不复制。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过定义/声明 `generate_step5_cpu_reference`；调用 `generate_step5_cpu_reference`；写入/初始化 `config`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 声明符中的 `&` 表示引用；只有在表达式操作数之间时才可能表示按位与或取地址。

### 4.28 generate_step5_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `generate_step5_cpu_reference`；源码锚点 `Step5ValidationData data;`。

```cpp
    Step5ValidationData data;
    cusignal::AmbgfunOptions options;
    options.fs = config.sample_rate_hz;
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：表达式/声明单元、声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `data` 由 `Step5ValidationData` 声明：类型控制可表示值、可用操作和传参方式；
- `options` 由 `cusignal::AmbgfunOptions` 声明：类型控制可表示值、可用操作和传参方式。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`data`|用于携带当前步骤输入/CPU reference/验证结果的数据结构|无单一数学符号；字段分别映射到算法数组|由生成函数填充并交给 comparison|
|`options`|当前表达式读取或传递的工程名称 `options`|源码没有为它规定独立数学符号；数学角色由同一源码锚点中的运算决定|直接证据 `cusignal::AmbgfunOptions options;`；值在 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu` 当前作用域中产生或消费|
|`config`|外部配置对象|`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。|入口加载，runner 和各 step 只读消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `Step5ValidationData data;` 是对象声明：类型 `Step5ValidationData` 应用于名称 `data`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `cusignal::AmbgfunOptions options;` 是对象声明：类型 `cusignal::AmbgfunOptions` 应用于名称 `options`。类类型在此执行默认构造；若是无初始化器的局部内置标量则值未确定，读取前必须赋值。
- `options.fs = config.sample_rate_hz;` 是赋值：先求得右侧完整表达式 `config.sample_rate_hz`，再把结果写入左侧可修改对象 `options.fs`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过写入/初始化 `fs`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.29 generate_step5_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `generate_step5_cpu_reference`；源码锚点 `options.prf = config.prf_hz;`。

```cpp
    options.prf = config.prf_hz;
    options.cut = cusignal::AmbgfunCut::two_dimensional;
    data.ambiguity_2d_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `options`、`prf`、`config`、`prf_hz`、`cut`、`cusignal`、`AmbgfunCut`、`two_dimensional`、`data`、`ambiguity_2d_cpu`、`to_double`、`ambgfun_typed_cpu`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`config`|外部配置对象|`config` 在本 step 实例化采样率、PRF、delay/Doppler cut、网格尺寸和模糊函数输出模式。|入口加载，runner 和各 step 只读消费|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.prf = config.prf_hz;` 是赋值：先求得右侧完整表达式 `config.prf_hz`，再把结果写入左侧可修改对象 `options.prf`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.cut = cusignal::AmbgfunCut::two_dimensional;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::two_dimensional`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.ambiguity_2d_cpu = to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options));` 是赋值：先求得右侧完整表达式 `to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options))`，再把结果写入左侧可修改对象 `data.ambiguity_2d_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `to_double`；写入/初始化 `prf`, `cut`, `ambiguity_2d_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.30 generate_step5_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `generate_step5_cpu_reference`；源码锚点 `options.cut = cusignal::AmbgfunCut::doppler;`。

```cpp
    options.cut = cusignal::AmbgfunCut::doppler;
    options.cut_value = 0.0;
    data.ambiguity_delay_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
```

**语法结构**

该代码块按源码顺序包含 3 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- `0.0` 是未带后缀的数值字面量，具体类型由写法和语言默认规则决定。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

|数字|类型/后缀|为什么使用、来源与改变后的影响|
|---:|---|---|
|`0.0`|无显式后缀，类型按语言的整型/浮点字面量默认规则确定|零在当前上下文通常表示加法单位元、空索引、无偏移、false/成功返回或清零初值；必须结合同一表达式判断。它不是统一的物理参数。|

**执行过程**

- `options.cut = cusignal::AmbgfunCut::doppler;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::doppler`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `options.cut_value = 0.0;` 是赋值：先求得右侧完整表达式 `0.0`，再把结果写入左侧可修改对象 `options.cut_value`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.ambiguity_delay_cpu = to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options));` 是赋值：先求得右侧完整表达式 `to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options))`，再把结果写入左侧可修改对象 `data.ambiguity_delay_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `to_double`；写入/初始化 `cut`, `cut_value`, `ambiguity_delay_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.31 generate_step5_cpu_reference：完成一个连续的数据处理动作

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `generate_step5_cpu_reference`；源码锚点 `options.cut = cusignal::AmbgfunCut::delay;`。

```cpp
    options.cut = cusignal::AmbgfunCut::delay;
    data.ambiguity_doppler_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
```

**语法结构**

该代码块按源码顺序包含 2 个完整语义单元：声明初始化或赋值语句。解析时先识别控制语句、声明、函数边界和 CUDA 扩展，再解释内部运算符。 跨行参数、模板实参和闭合括号属于同一语句，必须合并阅读。

**名称与类型**

- 当前块读取或传递的主要名称是 `options`、`cut`、`cusignal`、`AmbgfunCut`、`delay`、`data`、`ambiguity_doppler_cpu`、`to_double`、`ambgfun_typed_cpu`、`float`、`state`、`waveform`；它们按局部作用域、对象成员或命名空间解析，不会因出现在文档中而改变所有权。

**变量—公式—任务状态映射**

|变量|实际含义|公式/数学步骤|来源与消费者|
|---|---|---|---|
|`state`|任务流水线共享状态|无单一数学符号；保存各 step 的数组与证据|前一步写入、后一步读取|
|`waveform`|发射/参考复波形数组|记作 $s[n]$|Step1 生成，回波模拟和脉冲压缩消费|

**数字常量逐项说明**

当前源码块没有数值字面量，因此没有需要追溯的数字常量。

**执行过程**

- `options.cut = cusignal::AmbgfunCut::delay;` 是赋值：先求得右侧完整表达式 `cusignal::AmbgfunCut::delay`，再把结果写入左侧可修改对象 `options.cut`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。
- `data.ambiguity_doppler_cpu = to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options));` 是赋值：先求得右侧完整表达式 `to_double( cusignal::ambgfun_typed_cpu<float>(state.waveform, options))`，再把结果写入左侧可修改对象 `data.ambiguity_doppler_cpu`；这里的 `=` 不属于 `>=`、`<=`、`==` 或 `!=`。

**任务条款—业务逻辑—代码落实对应**

|对应维度|落实内容|
|---|---|
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过调用 `to_double`；写入/初始化 `cut`, `ambiguity_doppler_cpu`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- 不要把跨行续接单独求值；应从当前语句起点一直读到匹配的闭合符号和分号/花括号。

### 4.32 generate_step5_cpu_reference：形成并返回当前结果

完整 SHA `bb1a48c2e18fe2e252e0203e8be717d0c40f9251`；路径 `ZKX/Task1/task1_gpu_cpu/accuracy/validation/step5_validation.cu`；符号 `generate_step5_cpu_reference`；源码锚点 `return data;`。

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
|赛题条款|T1-R5：波形设计与模糊函数分析：用 `ambgfun` 检查延迟—多普勒主瓣和旁瓣|
|业务角色|测试证据|
|业务输入|T1-R1 的发射波形及采样率/PRF|
|代码如何落实|本块通过返回 `data`，落实其代码职责；这些名称和条件均直接来自紧邻源码。|
|业务输出|二维模糊函数或 delay/Doppler cut|
|下一消费者|由指标和可视化解释波形分辨能力|
|证明边界|本块只能证明验证口径、reference、误差/耗时/稳定性收集方式存在；它不能替代核心实现，也不能单独证明历史运行结果真实发生。|

**任务语义**

本块属于精度证据：CPU reference 与 GPU/任务输出在同一数据契约下比较，产生误差、语义一致性或失败原因。

**设计理由与替代方案**

- 当前块主要承担接口、声明、数据搬运、配置或证据连接职责，没有新增独立数学变换。采用显式中间变量/调用边界是为了让类型、所有权、失败位置和前后步骤数据依赖可审计；源码没有记录更具体的作者方案比较，因此不推测额外动机。

**初学者易错点**

- `return` 是跳转语句，不是类型名或变量声明；在普通函数/lambda 中结束当前调用，在 kernel 中只结束当前线程的 kernel 实例。

## 5. CPU/GPU边界

CPU reference从Host独立计算；GPU wrapper负责参数、显存和调度；私有kernel负责线程计算。comparison归测试文档，正式算子内部归Learning/operators。
