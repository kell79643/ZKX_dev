# cusignal_cpp_firwin 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1：当前学习版本](#v1当前学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | typed CPU reference、公开 GPU compatibility、resident 与融合 kernel |

## V1：当前学习版本（fdd55ac）

### 1. 版本身份与工作树门禁

| 项目 | 值 |
| --- | --- |
| 完整 Git SHA | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` |
| 分支 | `final-prep/benchmark-evidence-v1` |
| 提交时间 | `2026-08-15 23:45:58 +0800`（ISO：`2026-08-15T23:45:58+08:00`） |
| 提交主题 | `test(archive): 增加任务结果范围治理审计` |
| firwin 相关文件 dirty 状态 | 无；进入阶段三时 `git status --short -- <相关路径>` 无输出 |
| 读取方式 | `git -C ZKX show fdd55ac...:<路径>`，所有行号均属于该提交 |

本阶段只读 `ZKX/cusignal_cpp`，没有修改 C++/CUDA 源码，没有构建或运行测试，也没有连接 ZQ500。文中的测试结论分为“源码覆盖事实”和“仓库已有证据”，不声称本次重新执行。

### 2. 当前实现边界

公开 umbrella 头 `signal_processing.h` 导入 `filter_design_typed.h`。当前 `firwin` 实现由以下层次构成：

| 层次 | 文件与核心符号 | 职责 |
| --- | --- | --- |
| 公开接口与选项 | `filter_design_typed.h`：`FirwinOptions`、`firwin_typed_cpu`、`firwin_device`、resident API | 类型、shape、选项与契约 |
| Host 参数准备 | `filter_design_typed.cpp`：`prepare_firwin` | cutoff 校验、passband 配对、Kaiser/Hamming/显式窗 |
| CPU reference | 同文件：`firwin_typed_cpu<T>` | FP64 sinc、窗乘与归一化 |
| GPU compatibility wrapper | 同文件：`firwin_device<T>` | cutoff D2H、Host 准备、FP32 workspace H2D、kernel 路由 |
| Resident workspace | `FirwinDeviceWorkspace` 与 resident wrapper | 把参数准备、分配和传输移出稳定调用 |
| GPU launch wrapper | `filter_design_typed.cu` | shape 门禁与 kernel 启动 |
| Device 核心 | `filter_design_kernels.cuh` | sinc、多带叠加、归约、输出与 FP64 存储位拓宽 |
| 直接测试 | `e3_estimation_filter_design_type_smoke.cu` | 五种输入类型、CPU/GPU 对照、模式与异常 |

当前仓库不存在旧文档所称的 `filter_design.h/.cpp/filter_design_cuda.cu`。仍引用旧头文件的 D4/D6 示例不作为本 V1 的当前可构建证据；当前 CMake 明确注册的是 `e3_estimation_filter_design_type_smoke`。

### 3. 接口、数据结构与调用链

```text
signal_processing.h
  → filter_design_typed.h
  → firwin_typed_cpu<T>
       → prepare_firwin
       → double sinc / double window / double normalization
  → firwin_device<T>
       → cutoff.to_host()                         D2H
       → prepare_firwin                           Host
       → bands/window double→float
       → DeviceArray<float>::from_host            H2D
       ├─ N ≤ 1024: fused FP64-storage kernel
       └─ N > 1024: normalization kernel + FP64-storage output kernel

预准备高性能路径：
FirwinDeviceWorkspace 构造
  → prepare_firwin + 一次 H2D
  → firwin_resident_device / firwin_resident_fp64_storage_device
  → 稳定调用只发 kernel

特化低通路径：
device cutoff + device explicit window
  → firwin_lowpass_explicit_window_resident_device
  → 单个融合 FP32 kernel
```

### 4. 接口、参数准备与 CPU/GPU host 逻辑逐块解释

#### 公开 umbrella 导入

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/signal_processing.h:22`；符号：`公开导出`；提交内第 22–22 行。

```cpp
#include "filter_design/filter_design_typed.h"
```

这一行把 typed filter-design API 纳入总入口。

块内逐行说明：

- `#include "filter_design/filter_design_typed.h"`：引入 "filter_design/filter_design_typed.h" 所声明的类型或函数。

#### 头文件依赖与命名空间

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:1`；符号：`文件前导`；提交内第 1–8 行。

```cpp
#pragma once

#include "cuda_utils/device_array.h"
#include "cuda_utils/simple_signal_typed.h"

#include <vector>

namespace cusignal {
```

头文件防重复包含，并引入 DeviceArray、五类型策略和 vector。

块内逐行说明：

- `#pragma once`：头文件保护指令，防止同一翻译单元重复包含。
- `#include "cuda_utils/device_array.h"`：引入 "cuda_utils/device_array.h" 所声明的类型或函数。
- `#include "cuda_utils/simple_signal_typed.h"`：引入 "cuda_utils/simple_signal_typed.h" 所声明的类型或函数。
- `#include <vector>`：引入 <vector> 所声明的类型或函数。
- `namespace cusignal {`：进入命名空间 cusignal {。

#### pass_zero 与窗口模式枚举

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:10`；符号：`FirwinPassZero / FirwinWindowMode`；提交内第 10–19 行。

```cpp
enum class FirwinPassZero {
    boolean_true,
    boolean_false,
    lowpass,
    highpass,
    bandpass,
    bandstop
};

enum class FirwinWindowMode { hamming, none, explicit_values };
```

枚举完整表达 Python pass_zero 的六种取值，并限定 C++ 窗口模式。

块内逐行说明：

- `enum class FirwinPassZero {`：定义强类型枚举，枚举值不会隐式转换为整数。
- `boolean_true,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `boolean_false,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `lowpass,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `highpass,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `bandpass,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `bandstop`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `};`：开始或结束当前 C++ 作用域/类型定义。
- `enum class FirwinWindowMode { hamming, none, explicit_values };`：定义强类型枚举，枚举值不会隐式转换为整数。

#### 设计选项结构

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:21`；符号：`FirwinOptions`；提交内第 21–29 行。

```cpp
struct FirwinOptions {
    bool use_width = false;
    double width = 0.0;
    FirwinWindowMode window_mode = FirwinWindowMode::hamming;
    std::vector<double> window;
    FirwinPassZero pass_zero = FirwinPassZero::boolean_true;
    bool scale = true;
    double fs = 2.0;
};
```

结构体保存 width、窗口、pass_zero、scale 与 fs；默认值对齐 Python 常用路径。

块内逐行说明：

- `struct FirwinOptions {`：定义结构体，后续成员共同保存该阶段数据。
- `bool use_width = false;`：计算右侧表达式并初始化或更新左侧变量。
- `double width = 0.0;`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinWindowMode window_mode = FirwinWindowMode::hamming;`：计算右侧表达式并初始化或更新左侧变量。
- `std::vector<double> window;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinPassZero pass_zero = FirwinPassZero::boolean_true;`：计算右侧表达式并初始化或更新左侧变量。
- `bool scale = true;`：计算右侧表达式并初始化或更新左侧变量。
- `double fs = 2.0;`：计算右侧表达式并初始化或更新左侧变量。
- `};`：开始或结束当前 C++ 作用域/类型定义。

#### 预准备 workspace 契约

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:31`；符号：`FirwinDeviceWorkspace`；提交内第 31–76 行。

```cpp
/**
 * @brief `firwin` 的预准备 device-resident workspace。
 *
 * @details 构造阶段按五种业务类型解释并校验Host cutoff，准备bands/window后只上传一次。稳定执行阶段
 * 复用这些device数组，不再读取Host参数、不分配且不发生H2D/D2H。该workspace服务内部
 * FP32流水线；公开cuSignal兼容入口仍保持FP64输出存储。
 */
class FirwinDeviceWorkspace {
public:
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<float>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<__half>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<std::int32_t>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<std::int16_t>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<std::int8_t>& cutoff,
        const FirwinOptions& options = {});

    FirwinDeviceWorkspace(const FirwinDeviceWorkspace&) = delete;
    FirwinDeviceWorkspace& operator=(const FirwinDeviceWorkspace&) = delete;
    FirwinDeviceWorkspace(FirwinDeviceWorkspace&&) noexcept = default;
    FirwinDeviceWorkspace& operator=(FirwinDeviceWorkspace&&) noexcept = default;

    [[nodiscard]] int numtaps() const noexcept { return numtaps_; }

private:
    int numtaps_{0};
    bool scale_{true};
    DeviceArray<float> bands_;
    DeviceArray<float> window_;
    DeviceArray<float> normalization_;

    friend void firwin_resident_device(
        FirwinDeviceWorkspace&, DeviceArray<float>&);
    friend void firwin_resident_fp64_storage_device(
        FirwinDeviceWorkspace&, DeviceArray<double>&);
    friend void firwin_resident_stage_c_fp64_storage_device(
        FirwinDeviceWorkspace&, DeviceArray<double>&);
};
```

workspace 构造时准备并上传 bands/window，稳定调用复用 device 数组；禁止复制、允许移动。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief \`firwin\` 的预准备 device-resident workspace。`：`firwin` 的预准备 device-resident workspace。
- `*`：文档块中的空分隔行。
- `* @details 构造阶段按五种业务类型解释并校验Host cutoff，准备bands/window后只上传一次。稳定执行阶段`：构造阶段按五种业务类型解释并校验Host cutoff，准备bands/window后只上传一次。稳定执行阶段
- `* 复用这些device数组，不再读取Host参数、不分配且不发生H2D/D2H。该workspace服务内部`：复用这些device数组，不再读取Host参数、不分配且不发生H2D/D2H。该workspace服务内部
- `* FP32流水线；公开cuSignal兼容入口仍保持FP64输出存储。`：FP32流水线；公开cuSignal兼容入口仍保持FP64输出存储。
- `*/`：结束 Doxygen 文档块。
- `class FirwinDeviceWorkspace {`：定义类，封装可复用 device workspace 和所有权。
- `public:`：以下成员为公开接口。
- `FirwinDeviceWorkspace(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const std::vector<float>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const std::vector<__half>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const std::vector<std::int32_t>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const std::vector<std::int16_t>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const std::vector<std::int8_t>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace(const FirwinDeviceWorkspace&) = delete;`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace& operator=(const FirwinDeviceWorkspace&) = delete;`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace(FirwinDeviceWorkspace&&) noexcept = default;`：计算右侧表达式并初始化或更新左侧变量。
- `FirwinDeviceWorkspace& operator=(FirwinDeviceWorkspace&&) noexcept = default;`：计算右侧表达式并初始化或更新左侧变量。
- `[[nodiscard]] int numtaps() const noexcept { return numtaps_; }`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `private:`：以下成员仅类内部和 friend 可访问。
- `int numtaps_{0};`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `bool scale_{true};`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DeviceArray<float> bands_;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DeviceArray<float> window_;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DeviceArray<float> normalization_;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `friend void firwin_resident_device(`：声明 friend，使 wrapper 可访问 workspace 私有 device 数组。
- `FirwinDeviceWorkspace&, DeviceArray<float>&);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `friend void firwin_resident_fp64_storage_device(`：声明 friend，使 wrapper 可访问 workspace 私有 device 数组。
- `FirwinDeviceWorkspace&, DeviceArray<double>&);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `friend void firwin_resident_stage_c_fp64_storage_device(`：声明 friend，使 wrapper 可访问 workspace 私有 device 数组。
- `FirwinDeviceWorkspace&, DeviceArray<double>&);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `};`：开始或结束当前 C++ 作用域/类型定义。

#### FP32 resident 接口

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:78`；符号：`firwin_resident_device`；提交内第 78–86 行。

```cpp
/**
 * @brief 使用已准备workspace生成FP32 device-resident FIR系数。
 * @param workspace 构造阶段已完成参数校验、分配和上传的workspace。
 * @param out 调用者预分配的FP32 device输出 `[numtaps]`。
 * @details 稳定调用仅启动device计算；不分配、不执行H2D/D2H或Host后处理。该接口用于
 * Task流水线和resident benchmark，不能把其耗时冒充公开FP64兼容入口耗时。
 */
void firwin_resident_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<float>& out);
```

调用者提供已准备 workspace 和 FP32 输出，稳定调用只做 device 计算。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief 使用已准备workspace生成FP32 device-resident FIR系数。`：使用已准备workspace生成FP32 device-resident FIR系数。
- `* @param workspace 构造阶段已完成参数校验、分配和上传的workspace。`：@param workspace 构造阶段已完成参数校验、分配和上传的workspace。
- `* @param out 调用者预分配的FP32 device输出 \`[numtaps]\`。`：@param out 调用者预分配的FP32 device输出 `[numtaps]`。
- `* @details 稳定调用仅启动device计算；不分配、不执行H2D/D2H或Host后处理。该接口用于`：稳定调用仅启动device计算；不分配、不执行H2D/D2H或Host后处理。该接口用于
- `* Task流水线和resident benchmark，不能把其耗时冒充公开FP64兼容入口耗时。`：Task流水线和resident benchmark，不能把其耗时冒充公开FP64兼容入口耗时。
- `*/`：结束 Doxygen 文档块。
- `void firwin_resident_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinDeviceWorkspace& workspace, DeviceArray<float>& out);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。

#### FP64 存储 resident 接口

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:88`；符号：`firwin_resident_fp64_storage_device`；提交内第 88–97 行。

```cpp
/**
 * @brief 使用已准备workspace生成公开契约要求的FP64 device存储。
 * @param workspace 构造阶段已完成参数校验、分配和上传的workspace。
 * @param out 调用者预分配的FP64 device输出 `[numtaps]`。
 * @details 稳定调用只执行FP32计算，并以整数位操作写入等值IEEE-754 FP64存储；不执行
 * FP64算术，也不分配、不发生H2D/D2H或Host后处理。该接口用于五种业务类型完成setup后
 * 的同语义resident benchmark，不能跨compatibility scope计算加速比。
 */
void firwin_resident_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out);
```

主计算仍是 FP32，但按公开契约写入数值等值的 FP64 存储。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief 使用已准备workspace生成公开契约要求的FP64 device存储。`：使用已准备workspace生成公开契约要求的FP64 device存储。
- `* @param workspace 构造阶段已完成参数校验、分配和上传的workspace。`：@param workspace 构造阶段已完成参数校验、分配和上传的workspace。
- `* @param out 调用者预分配的FP64 device输出 \`[numtaps]\`。`：@param out 调用者预分配的FP64 device输出 `[numtaps]`。
- `* @details 稳定调用只执行FP32计算，并以整数位操作写入等值IEEE-754 FP64存储；不执行`：稳定调用只执行FP32计算，并以整数位操作写入等值IEEE-754 FP64存储；不执行
- `* FP64算术，也不分配、不发生H2D/D2H或Host后处理。该接口用于五种业务类型完成setup后`：FP64算术，也不分配、不发生H2D/D2H或Host后处理。该接口用于五种业务类型完成setup后
- `* 的同语义resident benchmark，不能跨compatibility scope计算加速比。`：的同语义resident benchmark，不能跨compatibility scope计算加速比。
- `*/`：结束 Doxygen 文档块。
- `void firwin_resident_fp64_storage_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinDeviceWorkspace& workspace, DeviceArray<double>& out);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。

#### 两 kernel 消融接口

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:99`；符号：`firwin_resident_stage_c_fp64_storage_device`；提交内第 99–105 行。

```cpp
/**
 * @brief 保留阶段C两kernel结构，用于与融合实现做同scope受控消融。
 * @details 该入口同样复用workspace并保持FP64存储语义，但分别启动归一化归约和输出kernel；
 * 不作为小规模正式快路径，大于融合上限时作为正确的通用回退。
 */
void firwin_resident_stage_c_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out);
```

保留归一化 kernel 加输出 kernel 的 Stage C 路径，供受控对比和大 N 回退。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief 保留阶段C两kernel结构，用于与融合实现做同scope受控消融。`：保留阶段C两kernel结构，用于与融合实现做同scope受控消融。
- `* @details 该入口同样复用workspace并保持FP64存储语义，但分别启动归一化归约和输出kernel；`：该入口同样复用workspace并保持FP64存储语义，但分别启动归一化归约和输出kernel；
- `* 不作为小规模正式快路径，大于融合上限时作为正确的通用回退。`：不作为小规模正式快路径，大于融合上限时作为正确的通用回退。
- `*/`：结束 Doxygen 文档块。
- `void firwin_resident_stage_c_fp64_storage_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinDeviceWorkspace& workspace, DeviceArray<double>& out);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。

#### 单 cutoff 显式窗 resident 特化

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:107`；符号：`firwin_lowpass_explicit_window_resident_device`；提交内第 107–117 行。

```cpp
/**
 * @brief device cutoff 与 FP32 显式窗口直连的低通 FIR resident 接口。
 * @tparam T 五种真实 cutoff 输入类型。
 * @details 仅覆盖单 cutoff、低通、显式窗口场景；输出由调用者预分配为 FP32。稳定调用只启动
 * 一个融合 kernel，不读取 Host、不分配、不传输。公开通用 `firwin_device` 语义不变。
 */
template <typename T>
void firwin_lowpass_explicit_window_resident_device(
    int numtaps, const DeviceArray<T>& cutoff,
    const DeviceArray<float>& window, DeviceArray<float>& out,
    float fs, bool scale = true);
```

这一接口只覆盖低通、单 cutoff、显式 device 窗和 FP32 输出。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief device cutoff 与 FP32 显式窗口直连的低通 FIR resident 接口。`：device cutoff 与 FP32 显式窗口直连的低通 FIR resident 接口。
- `* @tparam T 五种真实 cutoff 输入类型。`：@tparam T 五种真实 cutoff 输入类型。
- `* @details 仅覆盖单 cutoff、低通、显式窗口场景；输出由调用者预分配为 FP32。稳定调用只启动`：仅覆盖单 cutoff、低通、显式窗口场景；输出由调用者预分配为 FP32。稳定调用只启动
- `* 一个融合 kernel，不读取 Host、不分配、不传输。公开通用 \`firwin_device\` 语义不变。`：一个融合 kernel，不读取 Host、不分配、不传输。公开通用 `firwin_device` 语义不变。
- `*/`：结束 Doxygen 文档块。
- `template <typename T>`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `void firwin_lowpass_explicit_window_resident_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const DeviceArray<T>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const DeviceArray<float>& window, DeviceArray<float>& out,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `float fs, bool scale = true);`：计算右侧表达式并初始化或更新左侧变量。

#### CPU reference 声明

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:119`；符号：`firwin_typed_cpu`；提交内第 119–134 行。

```cpp
/**
 * @brief 在CPU上按完整firwin选项设计FIR，作为GPU reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 cutoff 输入；固定输出为FP64。
 * @param numtaps 正 tap 数。
 * @param cutoff host 一维严格递增数组 `[K]`，K>=1，每项位于 `(0,fs/2)`。
 * @param options width、window、完整pass_zero、scale和fs选项。
 * @return host `[numtaps]`、dtype FP64 的对称 FIR 系数；numtaps=1 使用窗值1。
 * @throws std::invalid_argument numtaps、cutoff、pass_zero、window、Nyquist或fs非法。
 * @details 支持low/high/band-pass/band-stop和多频带交替语义；width按cuSignal规则
 * 覆盖window并生成Kaiser。CPU sinc/window只作reference，无device或stream；临时bands和
 * window是workspace。
 */
template <typename T>
std::vector<double> firwin_typed_cpu(
    int numtaps, const std::vector<T>& cutoff,
    const FirwinOptions& options = {});
```

模板支持五种 cutoff 输入，固定返回 host FP64 系数。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief 在CPU上按完整firwin选项设计FIR，作为GPU reference。`：在CPU上按完整firwin选项设计FIR，作为GPU reference。
- `* @tparam T FP32、FP16、INT32、INT16 或 INT8 cutoff 输入；固定输出为FP64。`：@tparam T FP32、FP16、INT32、INT16 或 INT8 cutoff 输入；固定输出为FP64。
- `* @param numtaps 正 tap 数。`：@param numtaps 正 tap 数。
- `* @param cutoff host 一维严格递增数组 \`[K]\`，K>=1，每项位于 \`(0,fs/2)\`。`：@param cutoff host 一维严格递增数组 `[K]`，K>=1，每项位于 `(0,fs/2)`。
- `* @param options width、window、完整pass_zero、scale和fs选项。`：@param options width、window、完整pass_zero、scale和fs选项。
- `* @return host \`[numtaps]\`、dtype FP64 的对称 FIR 系数；numtaps=1 使用窗值1。`：@return host `[numtaps]`、dtype FP64 的对称 FIR 系数；numtaps=1 使用窗值1。
- `* @throws std::invalid_argument numtaps、cutoff、pass_zero、window、Nyquist或fs非法。`：@throws std::invalid_argument numtaps、cutoff、pass_zero、window、Nyquist或fs非法。
- `* @details 支持low/high/band-pass/band-stop和多频带交替语义；width按cuSignal规则`：支持low/high/band-pass/band-stop和多频带交替语义；width按cuSignal规则
- `* 覆盖window并生成Kaiser。CPU sinc/window只作reference，无device或stream；临时bands和`：覆盖window并生成Kaiser。CPU sinc/window只作reference，无device或stream；临时bands和
- `* window是workspace。`：window是workspace。
- `*/`：结束 Doxygen 文档块。
- `template <typename T>`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `std::vector<double> firwin_typed_cpu(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const std::vector<T>& cutoff,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。

#### GPU compatibility 声明

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.h:136`；符号：`firwin_device`；提交内第 136–153 行。

```cpp
/**
 * @brief 在GPU上以FP32完成主计算，并按cuSignal契约返回FP64存储。
 * @tparam T 五种业务cutoff输入类型；设备计算为FP32，正式输出为FP64。
 * @param numtaps 正 tap 数。
 * @param cutoff 调用者持有的 device 严格递增 `[K]` cutoff，K>=1。
 * @param out 调用者预分配的device FP64输出`[numtaps]`。
 * @param options 完整width/window/pass_zero/scale/fs选项。
 * @throws std::invalid_argument shape、cutoff、pass_zero、window、Nyquist或fs非法。
 * @details cutoff先D2H校验并在Host准备bands/window workspace；custom kernel执行
 * FP32计算后以整数位操作把FP32值写成等值IEEE-754 FP64存储，不执行FP64算术，也不再
 * D2H→Host拓宽→H2D。GPU kernel 使用默认stream；调用者在读取输出或跨stream复用
 * workspace 前负责同步。参数准备仍属于公开compatibility成本；高性能流水线应使用
 * `FirwinDeviceWorkspace` 和 `firwin_resident_device`。
 */
template <typename T>
void firwin_device(
    int numtaps, const DeviceArray<T>& cutoff, DeviceArray<double>& out,
    const FirwinOptions& options = {});
```

公开 GPU 入口以 FP32 计算、FP64 存储输出；参数准备仍计入 compatibility 成本。

块内逐行说明：

- `/**`：开始 Doxygen 文档块。
- `* @brief 在GPU上以FP32完成主计算，并按cuSignal契约返回FP64存储。`：在GPU上以FP32完成主计算，并按cuSignal契约返回FP64存储。
- `* @tparam T 五种业务cutoff输入类型；设备计算为FP32，正式输出为FP64。`：@tparam T 五种业务cutoff输入类型；设备计算为FP32，正式输出为FP64。
- `* @param numtaps 正 tap 数。`：@param numtaps 正 tap 数。
- `* @param cutoff 调用者持有的 device 严格递增 \`[K]\` cutoff，K>=1。`：@param cutoff 调用者持有的 device 严格递增 `[K]` cutoff，K>=1。
- `* @param out 调用者预分配的device FP64输出\`[numtaps]\`。`：@param out 调用者预分配的device FP64输出`[numtaps]`。
- `* @param options 完整width/window/pass_zero/scale/fs选项。`：@param options 完整width/window/pass_zero/scale/fs选项。
- `* @throws std::invalid_argument shape、cutoff、pass_zero、window、Nyquist或fs非法。`：@throws std::invalid_argument shape、cutoff、pass_zero、window、Nyquist或fs非法。
- `* @details cutoff先D2H校验并在Host准备bands/window workspace；custom kernel执行`：cutoff先D2H校验并在Host准备bands/window workspace；custom kernel执行
- `* FP32计算后以整数位操作把FP32值写成等值IEEE-754 FP64存储，不执行FP64算术，也不再`：FP32计算后以整数位操作把FP32值写成等值IEEE-754 FP64存储，不执行FP64算术，也不再
- `* D2H→Host拓宽→H2D。GPU kernel 使用默认stream；调用者在读取输出或跨stream复用`：D2H→Host拓宽→H2D。GPU kernel 使用默认stream；调用者在读取输出或跨stream复用
- `* workspace 前负责同步。参数准备仍属于公开compatibility成本；高性能流水线应使用`：workspace 前负责同步。参数准备仍属于公开compatibility成本；高性能流水线应使用
- `* \`FirwinDeviceWorkspace\` 和 \`firwin_resident_device\`。`：`FirwinDeviceWorkspace` 和 `firwin_resident_device`。
- `*/`：结束 Doxygen 文档块。
- `template <typename T>`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `void firwin_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int numtaps, const DeviceArray<T>& cutoff, DeviceArray<double>& out,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options = {});`：计算右侧表达式并初始化或更新左侧变量。

#### CPU 实现依赖与基础 helper

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:1`；符号：`load / sinc`；提交内第 1–16 行。

```cpp
#include "filter_design_typed.h"
#include "cuda_utils/host_output_finalize.h"
#include "windows/windows_typed.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cusignal {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
template <typename T> double load(T v) { return static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(v)); }
double sinc(double x) { return std::fabs(x) < 1.0e-15 ? 1.0 : std::sin(pi * x) / (pi * x); }
```

load 统一五种输入到 double；sinc 在中心用解析极限 1。

块内逐行说明：

- `#include "filter_design_typed.h"`：引入 "filter_design_typed.h" 所声明的类型或函数。
- `#include "cuda_utils/host_output_finalize.h"`：引入 "cuda_utils/host_output_finalize.h" 所声明的类型或函数。
- `#include "windows/windows_typed.h"`：引入 "windows/windows_typed.h" 所声明的类型或函数。
- `#include <algorithm>`：引入 <algorithm> 所声明的类型或函数。
- `#include <cmath>`：引入 <cmath> 所声明的类型或函数。
- `#include <complex>`：引入 <complex> 所声明的类型或函数。
- `#include <cstdint>`：引入 <cstdint> 所声明的类型或函数。
- `#include <limits>`：引入 <limits> 所声明的类型或函数。
- `#include <stdexcept>`：引入 <stdexcept> 所声明的类型或函数。
- `namespace cusignal {`：进入命名空间 cusignal {。
- `namespace {`：进入命名空间 {。
- `constexpr double pi = 3.141592653589793238462643383279502884;`：计算右侧表达式并初始化或更新左侧变量。
- `template <typename T> double load(T v) { return static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(v)); }`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `double sinc(double x) { return std::fabs(x) < 1.0e-15 ? 1.0 : std::sin(pi * x) / (pi * x); }`：定义归一化 sinc；|x|<1e-15 时直接返回极限 1，否则计算 sin(πx)/(πx)。

#### 准备结果结构

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:18`；符号：`FirwinPrepared`；提交内第 18–22 行。

```cpp
struct FirwinPrepared {
    std::vector<double> bands;
    std::vector<double> window;
    bool scale{true};
};
```

Host 准备结果只包含扁平通带边界、窗和 scale 标志。

块内逐行说明：

- `struct FirwinPrepared {`：定义结构体，后续成员共同保存该阶段数据。
- `std::vector<double> bands;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `std::vector<double> window;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `bool scale{true};`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `};`：开始或结束当前 C++ 作用域/类型定义。

#### 输入与 cutoff 归一化

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:24`；符号：`prepare_firwin`；提交内第 24–42 行。

```cpp
template <typename T>
FirwinPrepared prepare_firwin(
    int n, const std::vector<T>& cutoff, const FirwinOptions& options)
{
    if (n <= 0 || cutoff.empty() || !std::isfinite(options.fs) ||
        options.fs <= 0.0 ||
        options.fs > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::invalid_argument("firwin shape or fs");
    const double nyquist = options.fs * 0.5;
    std::vector<double> normalized(cutoff.size());
    double previous = 0.0;
    for (std::size_t index = 0; index < cutoff.size(); ++index) {
        const double value = load(cutoff[index]) / nyquist;
        if (!(value > 0.0 && value < 1.0) ||
            (index > 0 && !(value > previous)))
            throw std::invalid_argument("firwin cutoff");
        normalized[index] = value;
        previous = value;
    }
```

校验 N、fs、cutoff，再除以 Nyquist 得到严格递增的 0–1 边界。

块内逐行说明：

- `template <typename T>`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `FirwinPrepared prepare_firwin(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int n, const std::vector<T>& cutoff, const FirwinOptions& options)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `{`：开始或结束当前 C++ 作用域/类型定义。
- `if (n <= 0 || cutoff.empty() || !std::isfinite(options.fs) ||`：条件成立时执行随后语句或代码块。
- `options.fs <= 0.0 ||`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `options.fs > static_cast<double>(std::numeric_limits<float>::max()))`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `throw std::invalid_argument("firwin shape or fs");`：抛出 invalid_argument，拒绝当前非法输入。
- `const double nyquist = options.fs * 0.5;`：计算右侧表达式并初始化或更新左侧变量。
- `std::vector<double> normalized(cutoff.size());`：完成函数声明、调用或对象构造语句。
- `double previous = 0.0;`：计算右侧表达式并初始化或更新左侧变量。
- `for (std::size_t index = 0; index < cutoff.size(); ++index) {`：循环遍历对应索引或容器元素。
- `const double value = load(cutoff[index]) / nyquist;`：计算右侧表达式并初始化或更新左侧变量。
- `if (!(value > 0.0 && value < 1.0) ||`：条件成立时执行随后语句或代码块。
- `(index > 0 && !(value > previous)))`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `throw std::invalid_argument("firwin cutoff");`：抛出 invalid_argument，拒绝当前非法输入。
- `normalized[index] = value;`：计算右侧表达式并初始化或更新左侧变量。
- `previous = value;`：计算右侧表达式并初始化或更新左侧变量。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### pass_zero 归约与 Type II 门禁

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:43`；符号：`prepare_firwin`；提交内第 43–66 行。

```cpp
    bool pass_zero = true;
    switch (options.pass_zero) {
    case FirwinPassZero::boolean_true: pass_zero = true; break;
    case FirwinPassZero::boolean_false: pass_zero = false; break;
    case FirwinPassZero::lowpass:
        if (cutoff.size() != 1) throw std::invalid_argument("firwin lowpass cutoff");
        pass_zero = true;
        break;
    case FirwinPassZero::highpass:
        if (cutoff.size() != 1) throw std::invalid_argument("firwin highpass cutoff");
        pass_zero = false;
        break;
    case FirwinPassZero::bandpass:
        if (cutoff.size() <= 1) throw std::invalid_argument("firwin bandpass cutoff");
        pass_zero = false;
        break;
    case FirwinPassZero::bandstop:
        if (cutoff.size() <= 1) throw std::invalid_argument("firwin bandstop cutoff");
        pass_zero = true;
        break;
    }
    const bool pass_nyquist = static_cast<bool>(cutoff.size() & 1U) ^ pass_zero;
    if (pass_nyquist && n % 2 == 0)
        throw std::invalid_argument("even numtaps require zero response at Nyquist");
```

switch 把枚举归约成布尔初态，异或判断 Nyquist，偶数 taps 冲突时报错。

块内逐行说明：

- `bool pass_zero = true;`：计算右侧表达式并初始化或更新左侧变量。
- `switch (options.pass_zero) {`：按 pass_zero 枚举值选择语义分支。
- `case FirwinPassZero::boolean_true: pass_zero = true; break;`：处理这一枚举取值，设置 DC 通带初态并检查边界数量。
- `case FirwinPassZero::boolean_false: pass_zero = false; break;`：处理这一枚举取值，设置 DC 通带初态并检查边界数量。
- `case FirwinPassZero::lowpass:`：处理这一枚举取值，设置 DC 通带初态并检查边界数量。
- `if (cutoff.size() != 1) throw std::invalid_argument("firwin lowpass cutoff");`：条件成立时执行随后语句或代码块。
- `pass_zero = true;`：计算右侧表达式并初始化或更新左侧变量。
- `break;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `case FirwinPassZero::highpass:`：处理这一枚举取值，设置 DC 通带初态并检查边界数量。
- `if (cutoff.size() != 1) throw std::invalid_argument("firwin highpass cutoff");`：条件成立时执行随后语句或代码块。
- `pass_zero = false;`：计算右侧表达式并初始化或更新左侧变量。
- `break;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `case FirwinPassZero::bandpass:`：处理这一枚举取值，设置 DC 通带初态并检查边界数量。
- `if (cutoff.size() <= 1) throw std::invalid_argument("firwin bandpass cutoff");`：条件成立时执行随后语句或代码块。
- `pass_zero = false;`：计算右侧表达式并初始化或更新左侧变量。
- `break;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `case FirwinPassZero::bandstop:`：处理这一枚举取值，设置 DC 通带初态并检查边界数量。
- `if (cutoff.size() <= 1) throw std::invalid_argument("firwin bandstop cutoff");`：条件成立时执行随后语句或代码块。
- `pass_zero = true;`：计算右侧表达式并初始化或更新左侧变量。
- `break;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `const bool pass_nyquist = static_cast<bool>(cutoff.size() & 1U) ^ pass_zero;`：边界数奇偶与 DC 初态异或，得到 Nyquist 是否位于通带。
- `if (pass_nyquist && n % 2 == 0)`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("even numtaps require zero response at Nyquist");`：抛出 invalid_argument，拒绝当前非法输入。

#### 构造通带配对

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:67`；符号：`prepare_firwin`；提交内第 67–76 行。

```cpp
    std::vector<double> edges;
    if (pass_zero) edges.push_back(0.0);
    edges.insert(edges.end(), normalized.begin(), normalized.end());
    if (pass_nyquist) edges.push_back(1.0);
    if (edges.size() % 2 != 0)
        throw std::invalid_argument("firwin passband pairing");
    FirwinPrepared prepared;
    prepared.bands = std::move(edges);
    prepared.scale = options.scale;
    prepared.window.resize(n, 1.0);
```

按需补 0/1，使 edges 两两组成 [left,right]，再转入 prepared。

块内逐行说明：

- `std::vector<double> edges;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `if (pass_zero) edges.push_back(0.0);`：条件成立时执行随后语句或代码块。
- `edges.insert(edges.end(), normalized.begin(), normalized.end());`：完成函数声明、调用或对象构造语句。
- `if (pass_nyquist) edges.push_back(1.0);`：条件成立时执行随后语句或代码块。
- `if (edges.size() % 2 != 0)`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("firwin passband pairing");`：抛出 invalid_argument，拒绝当前非法输入。
- `FirwinPrepared prepared;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `prepared.bands = std::move(edges);`：计算右侧表达式并初始化或更新左侧变量。
- `prepared.scale = options.scale;`：计算右侧表达式并初始化或更新左侧变量。
- `prepared.window.resize(n, 1.0);`：完成函数声明、调用或对象构造语句。

#### Kaiser、Hamming 与显式窗

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:77`；符号：`prepare_firwin`；提交内第 77–102 行。

```cpp
    if (options.use_width) {
        const double attenuation =
            2.285 * (n - 1) * pi * (options.width / nyquist) + 7.95;
        const double beta = attenuation > 50.0
            ? 0.1102 * (attenuation - 8.7)
            : (attenuation > 21.0
                ? 0.5842 * std::pow(attenuation - 21.0, 0.4)
                    + 0.07886 * (attenuation - 21.0)
                : 0.0);
        const auto kaiser = kaiser_typed_cpu<float>(
            n, static_cast<float>(beta), true);
        for (int index = 0; index < n; ++index)
            prepared.window[index] = kaiser[index];
    } else if (options.window_mode == FirwinWindowMode::hamming) {
        for (int index = 0; index < n; ++index)
            prepared.window[index] = n == 1 ? 1.0
                : 0.54 - 0.46 * std::cos(2.0 * pi * index / (n - 1));
    } else if (options.window_mode == FirwinWindowMode::explicit_values) {
        if (options.window.size() != static_cast<std::size_t>(n))
            throw std::invalid_argument("firwin explicit window length");
        prepared.window = options.window;
    } else if (options.window_mode != FirwinWindowMode::none) {
        throw std::invalid_argument("firwin window mode");
    }
    return prepared;
}
```

width 优先并覆盖窗口模式；否则选择 Hamming、显式窗或 none。

块内逐行说明：

- `if (options.use_width) {`：条件成立时执行随后语句或代码块。
- `const double attenuation =`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `2.285 * (n - 1) * pi * (options.width / nyquist) + 7.95;`：Kaiser 衰减经验式，width/nyquist 是 Nyquist 归一化过渡宽度。
- `const double beta = attenuation > 50.0`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `? 0.1102 * (attenuation - 8.7)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `: (attenuation > 21.0`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `? 0.5842 * std::pow(attenuation - 21.0, 0.4)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `+ 0.07886 * (attenuation - 21.0)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `: 0.0);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `const auto kaiser = kaiser_typed_cpu<float>(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `n, static_cast<float>(beta), true);`：完成函数声明、调用或对象构造语句。
- `for (int index = 0; index < n; ++index)`：循环遍历对应索引或容器元素。
- `prepared.window[index] = kaiser[index];`：计算右侧表达式并初始化或更新左侧变量。
- `} else if (options.window_mode == FirwinWindowMode::hamming) {`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `for (int index = 0; index < n; ++index)`：循环遍历对应索引或容器元素。
- `prepared.window[index] = n == 1 ? 1.0`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `: 0.54 - 0.46 * std::cos(2.0 * pi * index / (n - 1));`：完成函数声明、调用或对象构造语句。
- `} else if (options.window_mode == FirwinWindowMode::explicit_values) {`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `if (options.window.size() != static_cast<std::size_t>(n))`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("firwin explicit window length");`：抛出 invalid_argument，拒绝当前非法输入。
- `prepared.window = options.window;`：计算右侧表达式并初始化或更新左侧变量。
- `} else if (options.window_mode != FirwinWindowMode::none) {`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `throw std::invalid_argument("firwin window mode");`：抛出 invalid_argument，拒绝当前非法输入。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `return prepared;`：结束当前函数或 lambda 并返回该表达式。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### GPU 计算函数前置声明

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:197`；符号：`firwin_*_compute_device`；提交内第 197–205 行。

```cpp
void firwin_fp32_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    DeviceArray<float>&, DeviceArray<float>&, bool);
void firwin_fp64_storage_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    DeviceArray<float>&, void*, std::size_t, bool);
void firwin_fused_fp64_storage_compute_device(
    int, const DeviceArray<float>&, const DeviceArray<float>&,
    void*, std::size_t, bool);
```

Host 翻译单元只声明 CUDA 翻译单元实现的三个 firwin 计算入口。

块内逐行说明：

- `void firwin_fp32_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int, const DeviceArray<float>&, const DeviceArray<float>&,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `DeviceArray<float>&, DeviceArray<float>&, bool);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `void firwin_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int, const DeviceArray<float>&, const DeviceArray<float>&,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `DeviceArray<float>&, void*, std::size_t, bool);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `void firwin_fused_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int, const DeviceArray<float>&, const DeviceArray<float>&,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `void*, std::size_t, bool);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。

#### 五类型 workspace 构造

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:210`；符号：`FirwinDeviceWorkspace constructors`；提交内第 210–228 行。

```cpp
#define DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(T) \
FirwinDeviceWorkspace::FirwinDeviceWorkspace( \
    int n, const std::vector<T>& cutoff, const FirwinOptions& options) \
    : numtaps_(n) \
{ \
    const auto prepared = prepare_firwin(n, cutoff, options); \
    scale_ = prepared.scale; \
    bands_ = DeviceArray<float>::from_host(std::vector<float>( \
        prepared.bands.begin(), prepared.bands.end())); \
    window_ = DeviceArray<float>::from_host(std::vector<float>( \
        prepared.window.begin(), prepared.window.end())); \
    normalization_.reset(1); \
}
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(float)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(__half)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int32_t)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int16_t)
DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int8_t)
#undef DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR
```

宏复用同一构造逻辑：prepare、double→float、H2D、分配归一化标量。

块内逐行说明：

- `#define DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(T) \`：开始预处理宏定义，反斜杠把定义延续到下一物理行。
- `FirwinDeviceWorkspace::FirwinDeviceWorkspace( \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `int n, const std::vector<T>& cutoff, const FirwinOptions& options) \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `: numtaps_(n) \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `{ \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `const auto prepared = prepare_firwin(n, cutoff, options); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `scale_ = prepared.scale; \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `bands_ = DeviceArray<float>::from_host(std::vector<float>( \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `prepared.bands.begin(), prepared.bands.end())); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `window_ = DeviceArray<float>::from_host(std::vector<float>( \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `prepared.window.begin(), prepared.window.end())); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `normalization_.reset(1); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(float)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(__half)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int32_t)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int16_t)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR(std::int8_t)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `#undef DEFINE_FIRWIN_WORKSPACE_CONSTRUCTOR`：取消宏定义，避免污染后续代码。

#### FP32 resident wrapper

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:230`；符号：`firwin_resident_device`；提交内第 230–238 行。

```cpp
void firwin_resident_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<float>& out)
{
    if (out.size() != static_cast<std::size_t>(workspace.numtaps_))
        throw std::invalid_argument("firwin resident output shape");
    firwin_fp32_compute_device(
        workspace.numtaps_, workspace.bands_, workspace.window_,
        workspace.normalization_, out, workspace.scale_);
}
```

检查输出长度后把 workspace 成员交给 FP32 launcher。

块内逐行说明：

- `void firwin_resident_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinDeviceWorkspace& workspace, DeviceArray<float>& out)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `{`：开始或结束当前 C++ 作用域/类型定义。
- `if (out.size() != static_cast<std::size_t>(workspace.numtaps_))`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("firwin resident output shape");`：抛出 invalid_argument，拒绝当前非法输入。
- `firwin_fp32_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `workspace.numtaps_, workspace.bands_, workspace.window_,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `workspace.normalization_, out, workspace.scale_);`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### FP64 resident 路由

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:240`；符号：`firwin_resident_fp64_storage_device`；提交内第 240–256 行。

```cpp
void firwin_resident_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out)
{
    if (out.size() != static_cast<std::size_t>(workspace.numtaps_))
        throw std::invalid_argument("firwin resident FP64 storage output shape");
    static_assert(sizeof(double) == sizeof(std::uint64_t),
        "FP64 storage must be 64-bit");
    if (workspace.numtaps_ <= 1024) {
        firwin_fused_fp64_storage_compute_device(
            workspace.numtaps_, workspace.bands_, workspace.window_,
            out.data(), out.size(), workspace.scale_);
    } else {
        firwin_fp64_storage_compute_device(
            workspace.numtaps_, workspace.bands_, workspace.window_,
            workspace.normalization_, out.data(), out.size(), workspace.scale_);
    }
}
```

N≤1024 使用融合 kernel；更大 N 回退到归一化和输出两 kernel。

块内逐行说明：

- `void firwin_resident_fp64_storage_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinDeviceWorkspace& workspace, DeviceArray<double>& out)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `{`：开始或结束当前 C++ 作用域/类型定义。
- `if (out.size() != static_cast<std::size_t>(workspace.numtaps_))`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("firwin resident FP64 storage output shape");`：抛出 invalid_argument，拒绝当前非法输入。
- `static_assert(sizeof(double) == sizeof(std::uint64_t),`：编译期断言：只允许约定的五种简单信号输入类型或存储宽度。
- `"FP64 storage must be 64-bit");`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `if (workspace.numtaps_ <= 1024) {`：条件成立时执行随后语句或代码块。
- `firwin_fused_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `workspace.numtaps_, workspace.bands_, workspace.window_,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `out.data(), out.size(), workspace.scale_);`：完成函数声明、调用或对象构造语句。
- `} else {`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `firwin_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `workspace.numtaps_, workspace.bands_, workspace.window_,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `workspace.normalization_, out.data(), out.size(), workspace.scale_);`：完成函数声明、调用或对象构造语句。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### Stage C wrapper

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:258`；符号：`firwin_resident_stage_c_fp64_storage_device`；提交内第 258–266 行。

```cpp
void firwin_resident_stage_c_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out)
{
    if (out.size() != static_cast<std::size_t>(workspace.numtaps_))
        throw std::invalid_argument("firwin stage C FP64 storage output shape");
    firwin_fp64_storage_compute_device(
        workspace.numtaps_, workspace.bands_, workspace.window_,
        workspace.normalization_, out.data(), out.size(), workspace.scale_);
}
```

无条件走两 kernel 路径，便于与融合实现同 scope 消融。

块内逐行说明：

- `void firwin_resident_stage_c_fp64_storage_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `FirwinDeviceWorkspace& workspace, DeviceArray<double>& out)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `{`：开始或结束当前 C++ 作用域/类型定义。
- `if (out.size() != static_cast<std::size_t>(workspace.numtaps_))`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("firwin stage C FP64 storage output shape");`：抛出 invalid_argument，拒绝当前非法输入。
- `firwin_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `workspace.numtaps_, workspace.bands_, workspace.window_,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `workspace.normalization_, out.data(), out.size(), workspace.scale_);`：完成函数声明、调用或对象构造语句。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### CPU sinc 多带核心

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:268`；符号：`firwin_typed_cpu`；提交内第 268–282 行。

```cpp
template <typename T>
std::vector<double> firwin_typed_cpu(
    int n, const std::vector<T>& cutoff, const FirwinOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin dtype");
    const auto prepared = prepare_firwin(n, cutoff, options);
    const auto coefficient = [&](double m) {
        double value = 0.0;
        for (std::size_t band = 0; band < prepared.bands.size(); band += 2) {
            const double left = prepared.bands[band];
            const double right = prepared.bands[band + 1];
            value += right * sinc(right * m) - left * sinc(left * m);
        }
        return value;
    };
```

lambda 对每个通带计算 right*sinc-right minus left*sinc-left，并线性累加。

块内逐行说明：

- `template <typename T>`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `std::vector<double> firwin_typed_cpu(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int n, const std::vector<T>& cutoff, const FirwinOptions& options)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `{`：开始或结束当前 C++ 作用域/类型定义。
- `static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin dtype");`：编译期断言：只允许约定的五种简单信号输入类型或存储宽度。
- `const auto prepared = prepare_firwin(n, cutoff, options);`：计算右侧表达式并初始化或更新左侧变量。
- `const auto coefficient = [&](double m) {`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `double value = 0.0;`：计算右侧表达式并初始化或更新左侧变量。
- `for (std::size_t band = 0; band < prepared.bands.size(); band += 2) {`：循环遍历对应索引或容器元素。
- `const double left = prepared.bands[band];`：计算右侧表达式并初始化或更新左侧变量。
- `const double right = prepared.bands[band + 1];`：计算右侧表达式并初始化或更新左侧变量。
- `value += right * sinc(right * m) - left * sinc(left * m);`：这一行就是多矩形通带逆 DTFT 的 sinc 差公式。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `return value;`：结束当前函数或 lambda 并返回该表达式。
- `};`：开始或结束当前 C++ 作用域/类型定义。

#### CPU 乘窗与归一化

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:283`；符号：`firwin_typed_cpu`；提交内第 283–300 行。

```cpp
    std::vector<double> work(n); const double mid = 0.5 * (n - 1);
    for (int i = 0; i < n; ++i) {
        const double m = i - mid;
        work[i] = coefficient(m) * prepared.window[i];
    }
    if (prepared.scale) {
        double normalization = 0.0;
        const double left = prepared.bands[0];
        const double right = prepared.bands[1];
        const double scale_frequency = left == 0.0 ? 0.0
            : (right == 1.0 ? 1.0 : 0.5 * (left + right));
        for (int i = 0; i < n; ++i) {
            normalization += work[i] * std::cos(pi * (i - mid) * scale_frequency);
        }
        for (double& value : work) value /= normalization;
    }
    return work;
}
```

先按中心坐标计算窗后系数，再在第一通带代表频率求余弦加权分母。

块内逐行说明：

- `std::vector<double> work(n); const double mid = 0.5 * (n - 1);`：计算右侧表达式并初始化或更新左侧变量。
- `for (int i = 0; i < n; ++i) {`：循环遍历对应索引或容器元素。
- `const double m = i - mid;`：计算右侧表达式并初始化或更新左侧变量。
- `work[i] = coefficient(m) * prepared.window[i];`：计算右侧表达式并初始化或更新左侧变量。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `if (prepared.scale) {`：条件成立时执行随后语句或代码块。
- `double normalization = 0.0;`：计算右侧表达式并初始化或更新左侧变量。
- `const double left = prepared.bands[0];`：计算右侧表达式并初始化或更新左侧变量。
- `const double right = prepared.bands[1];`：计算右侧表达式并初始化或更新左侧变量。
- `const double scale_frequency = left == 0.0 ? 0.0`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `: (right == 1.0 ? 1.0 : 0.5 * (left + right));`：计算右侧表达式并初始化或更新左侧变量。
- `for (int i = 0; i < n; ++i) {`：循环遍历对应索引或容器元素。
- `normalization += work[i] * std::cos(pi * (i - mid) * scale_frequency);`：把窗后系数乘参考频率余弦并累加成缩放分母。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `for (double& value : work) value /= normalization;`：范围 for 将每个系数除以同一 normalization。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `return work;`：结束当前函数或 lambda 并返回该表达式。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### 公开 GPU compatibility wrapper

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:302`；符号：`firwin_device`；提交内第 302–328 行。

```cpp
template <typename T>
void firwin_device(
    int n, const DeviceArray<T>& cutoff, DeviceArray<double>& out,
    const FirwinOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin dtype");
    if (out.size() != static_cast<std::size_t>(n > 0 ? n : 0))
        throw std::invalid_argument("firwin output shape");
    const auto prepared = prepare_firwin(n, cutoff.to_host(), options);
    std::vector<float> fp32_bands(
        prepared.bands.begin(), prepared.bands.end());
    std::vector<float> fp32_window(
        prepared.window.begin(), prepared.window.end());
    auto bands = DeviceArray<float>::from_host(fp32_bands);
    auto window = DeviceArray<float>::from_host(fp32_window);
    DeviceArray<float> normalization;
    if (n > 1024) normalization.reset(1);
    static_assert(sizeof(double) == sizeof(std::uint64_t),
        "FP64 storage must be 64-bit");
    if (n <= 1024) {
        firwin_fused_fp64_storage_compute_device(
            n, bands, window, out.data(), out.size(), prepared.scale);
    } else {
        firwin_fp64_storage_compute_device(
            n, bands, window, normalization, out.data(), out.size(), prepared.scale);
    }
}
```

cutoff D2H 后 Host prepare，再把 FP32 bands/window H2D，并按 N 路由融合或两 kernel。

块内逐行说明：

- `template <typename T>`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `void firwin_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `int n, const DeviceArray<T>& cutoff, DeviceArray<double>& out,`：这是跨行声明或调用的一个参数/元素，逗号表示仍有后续项。
- `const FirwinOptions& options)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `{`：开始或结束当前 C++ 作用域/类型定义。
- `static_assert(detail::is_simple_signal_input_v<T>, "unsupported firwin dtype");`：编译期断言：只允许约定的五种简单信号输入类型或存储宽度。
- `if (out.size() != static_cast<std::size_t>(n > 0 ? n : 0))`：条件成立时执行随后语句或代码块。
- `throw std::invalid_argument("firwin output shape");`：抛出 invalid_argument，拒绝当前非法输入。
- `const auto prepared = prepare_firwin(n, cutoff.to_host(), options);`：公开 GPU 入口先把 device cutoff 拷回 Host 进行完整校验和参数准备。
- `std::vector<float> fp32_bands(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `prepared.bands.begin(), prepared.bands.end());`：完成函数声明、调用或对象构造语句。
- `std::vector<float> fp32_window(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `prepared.window.begin(), prepared.window.end());`：完成函数声明、调用或对象构造语句。
- `auto bands = DeviceArray<float>::from_host(fp32_bands);`：把 FP32 通带边界从 Host 上传为 DeviceArray。
- `auto window = DeviceArray<float>::from_host(fp32_window);`：把 FP32 窗从 Host 上传为 DeviceArray。
- `DeviceArray<float> normalization;`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `if (n > 1024) normalization.reset(1);`：条件成立时执行随后语句或代码块。
- `static_assert(sizeof(double) == sizeof(std::uint64_t),`：编译期断言：只允许约定的五种简单信号输入类型或存储宽度。
- `"FP64 storage must be 64-bit");`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `if (n <= 1024) {`：条件成立时执行随后语句或代码块。
- `firwin_fused_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `n, bands, window, out.data(), out.size(), prepared.scale);`：完成函数声明、调用或对象构造语句。
- `} else {`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `firwin_fp64_storage_compute_device(`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `n, bands, window, normalization, out.data(), out.size(), prepared.scale);`：完成函数声明、调用或对象构造语句。
- `}`：开始或结束当前 C++ 作用域/类型定义。
- `}`：开始或结束当前 C++ 作用域/类型定义。

#### 五类型显式实例化

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cpp:402`；符号：`INST`；提交内第 402–414 行。

```cpp
#define INST(T) \
 template std::vector<double> firwin_typed_cpu( \
    int, const std::vector<T>&, const FirwinOptions&); \
 template void firwin_device( \
    int, const DeviceArray<T>&, DeviceArray<double>&, const FirwinOptions&); \
 template std::vector<double> firwin2_typed_cpu( \
    int, const std::vector<T>&, const std::vector<T>&, const Firwin2Options&); \
 template void firwin2_device( \
    int, const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<double>&, \
    const Firwin2Options&)
INST(float); INST(__half); INST(std::int32_t); INST(std::int16_t); INST(std::int8_t);
#undef INST
}  // namespace cusignal
```

共享宏同时实例化 firwin/firwin2；其中 firwin CPU/GPU 对五种输入类型生成实体。

块内逐行说明：

- `#define INST(T) \`：开始预处理宏定义，反斜杠把定义延续到下一物理行。
- `template std::vector<double> firwin_typed_cpu( \`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `int, const std::vector<T>&, const FirwinOptions&); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `template void firwin_device( \`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `int, const DeviceArray<T>&, DeviceArray<double>&, const FirwinOptions&); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `template std::vector<double> firwin2_typed_cpu( \`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `int, const std::vector<T>&, const std::vector<T>&, const Firwin2Options&); \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `template void firwin2_device( \`：声明模板参数，使同一实现适配多种 cutoff 输入类型。
- `int, const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<double>&, \`：宏定义的续行，末尾反斜杠把下一物理行并入同一宏。
- `const Firwin2Options&)`：该行是相邻声明、表达式或作用域的一部分，按 C++ 顺序参与当前语义块。
- `INST(float); INST(__half); INST(std::int32_t); INST(std::int16_t); INST(std::int8_t);`：完成函数声明、调用或对象构造语句。
- `#undef INST`：取消宏定义，避免污染后续代码。
- `}  // namespace cusignal`：结束所标注的命名空间。

### 5. GPU launcher、并行 kernel 与数值 helper 逐块解释

#### Neumaier 补偿累加器

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/cuda_utils/operator_compute_policy.h:90`；符号：`CompensatedFloatAccumulator`；提交内第 90–116 行。

```cpp
// Neumaier compensation keeps long FP32 reductions stable without requiring
// unsupported device FP64.  Call value() only after all terms have been added.
struct CompensatedFloatAccumulator {
    float sum{0.0F};
    float correction{0.0F};

    __host__ __device__ void add(float value)
    {
        const float next = sum + value;
        if (fabsf(sum) >= fabsf(value)) {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }

    __host__ __device__ void add_product(float left, float right)
    {
        add(fmaf(left, right, 0.0F));
    }

    __host__ __device__ float value() const
    {
        return sum + correction;
    }
};
```

归一化是长 FP32 归约；Neumaier correction 降低不同量级项相加的舍入损失。

块内逐行说明：

- `// Neumaier compensation keeps long FP32 reductions stable without requiring`：Neumaier compensation keeps long FP32 reductions stable without requiring
- `// unsupported device FP64.  Call value() only after all terms have been added.`：unsupported device FP64.  Call value() only after all terms have been added.
- `struct CompensatedFloatAccumulator {`：相邻声明、表达式或作用域的一部分。
- `float sum{0.0F};`：相邻声明、表达式或作用域的一部分。
- `float correction{0.0F};`：相邻声明、表达式或作用域的一部分。
- `__host__ __device__ void add(float value)`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `const float next = sum + value;`：初始化或更新变量。
- `if (fabsf(sum) >= fabsf(value)) {`：条件成立时执行后续语句或块。
- `correction += (sum - next) + value;`：初始化或更新变量。
- `} else {`：相邻声明、表达式或作用域的一部分。
- `correction += (value - next) + sum;`：初始化或更新变量。
- `}`：开始或结束当前作用域。
- `sum = next;`：初始化或更新变量。
- `}`：开始或结束当前作用域。
- `__host__ __device__ void add_product(float left, float right)`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `add(fmaf(left, right, 0.0F));`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `__host__ __device__ float value() const`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `return sum + correction;`：返回当前结果。
- `}`：开始或结束当前作用域。
- `};`：开始或结束当前作用域。

#### 一维 grid/block 配置

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/cuda_utils/kernel_launch.h:10`；符号：`make_1d_launch_config / launch_1d_kernel`；提交内第 10–56 行。

```cpp
template <typename IndexType>
constexpr IndexType div_up(IndexType value, IndexType divisor)
{
    return (value + divisor - 1) / divisor;
}

struct LaunchConfig1D {
    dim3 block;
    dim3 grid;
};

inline LaunchConfig1D make_1d_launch_config(
    std::size_t elements,
    unsigned int block_size = 256)
{
    return LaunchConfig1D{
        dim3(block_size),
        dim3(static_cast<unsigned int>(div_up(elements, static_cast<std::size_t>(block_size))))
    };
}

template <typename Kernel, typename... Args>
inline void launch_1d_kernel_with_config(
    Kernel kernel,
    std::size_t elements,
    cudaStream_t stream,
    unsigned int block_size,
    Args&&... args)
{
    LaunchConfig1D config = make_1d_launch_config(elements, block_size);
    kernel<<<config.grid, config.block, 0, stream>>>(std::forward<Args>(args)...);
    CUDA_KERNEL_CHECK();
}

template <typename Kernel, typename... Args>
inline void launch_1d_kernel(
    Kernel kernel,
    std::size_t elements,
    Args&&... args)
{
    launch_1d_kernel_with_config(
        kernel,
        elements,
        nullptr,
        256,
        std::forward<Args>(args)...);
}
```

默认 block 为 256 threads，grid 为 ceil(elements/256)，使用默认 stream 并在 launch 后检查错误。

块内逐行说明：

- `template <typename IndexType>`：声明模板参数。
- `constexpr IndexType div_up(IndexType value, IndexType divisor)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `return (value + divisor - 1) / divisor;`：返回当前结果。
- `}`：开始或结束当前作用域。
- `struct LaunchConfig1D {`：相邻声明、表达式或作用域的一部分。
- `dim3 block;`：相邻声明、表达式或作用域的一部分。
- `dim3 grid;`：相邻声明、表达式或作用域的一部分。
- `};`：开始或结束当前作用域。
- `inline LaunchConfig1D make_1d_launch_config(`：相邻声明、表达式或作用域的一部分。
- `std::size_t elements,`：跨行形参或实参的一项。
- `unsigned int block_size = 256)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `return LaunchConfig1D{`：返回当前结果。
- `dim3(block_size),`：跨行形参或实参的一项。
- `dim3(static_cast<unsigned int>(div_up(elements, static_cast<std::size_t>(block_size))))`：grid.x=ceil(elements/block_size)，保证覆盖全部一维元素。
- `};`：开始或结束当前作用域。
- `}`：开始或结束当前作用域。
- `template <typename Kernel, typename... Args>`：声明模板参数。
- `inline void launch_1d_kernel_with_config(`：调用统一一维 kernel launcher。
- `Kernel kernel,`：跨行形参或实参的一项。
- `std::size_t elements,`：跨行形参或实参的一项。
- `cudaStream_t stream,`：跨行形参或实参的一项。
- `unsigned int block_size,`：跨行形参或实参的一项。
- `Args&&... args)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `LaunchConfig1D config = make_1d_launch_config(elements, block_size);`：初始化或更新变量。
- `kernel<<<config.grid, config.block, 0, stream>>>(std::forward<Args>(args)...);`：按 grid/block、零动态共享内存和指定 stream 启动 kernel。
- `CUDA_KERNEL_CHECK();`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `template <typename Kernel, typename... Args>`：声明模板参数。
- `inline void launch_1d_kernel(`：调用统一一维 kernel launcher。
- `Kernel kernel,`：跨行形参或实参的一项。
- `std::size_t elements,`：跨行形参或实参的一项。
- `Args&&... args)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `launch_1d_kernel_with_config(`：调用统一一维 kernel launcher。
- `kernel,`：跨行形参或实参的一项。
- `elements,`：跨行形参或实参的一项。
- `nullptr,`：nullptr 表示 CUDA 默认 stream。
- `256,`：跨行形参或实参的一项。
- `std::forward<Args>(args)...);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。

#### FP32 两 kernel launcher

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cu:1`；符号：`firwin_fp32_compute_device`；提交内第 1–32 行。

```cpp
#include "filter_design_typed.h"
#include "filter_design_kernels.cuh"
#include "cuda_utils/cuda_utils.h"

#include <cmath>
#include <stdexcept>

namespace cusignal {
void firwin_fp32_compute_device(
    int n,
    const DeviceArray<float>& bands,
    const DeviceArray<float>& window,
    DeviceArray<float>& normalization,
    DeviceArray<float>& out,
    bool scale)
{
    if (n <= 0 || bands.empty() || bands.size() % 2 != 0 ||
        window.size() != static_cast<std::size_t>(n) ||
        normalization.size() != 1 ||
        out.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument("firwin compute shape");
    if (scale) {
        cuda_utils::launch_1d_kernel(
            filter_design_detail::firwin_prepared_normalization_kernel,
            256U, n, bands.data(), static_cast<int>(bands.size() / 2),
            window.data(), normalization.data());
    }
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_prepared_kernel, out.size(), n,
        bands.data(), static_cast<int>(bands.size() / 2), window.data(),
        normalization.data(), out.data(), scale);
}
```

scale=True 时先发单 block 归一化 kernel，再按 N 个输出元素发系数 kernel。

块内逐行说明：

- `#include "filter_design_typed.h"`：引入当前 helper 所需声明。
- `#include "filter_design_kernels.cuh"`：引入当前 helper 所需声明。
- `#include "cuda_utils/cuda_utils.h"`：引入当前 helper 所需声明。
- `#include <cmath>`：引入当前 helper 所需声明。
- `#include <stdexcept>`：引入当前 helper 所需声明。
- `namespace cusignal {`：进入命名空间。
- `void firwin_fp32_compute_device(`：相邻声明、表达式或作用域的一部分。
- `int n,`：跨行形参或实参的一项。
- `const DeviceArray<float>& bands,`：跨行形参或实参的一项。
- `const DeviceArray<float>& window,`：跨行形参或实参的一项。
- `DeviceArray<float>& normalization,`：跨行形参或实参的一项。
- `DeviceArray<float>& out,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `if (n <= 0 || bands.empty() || bands.size() % 2 != 0 ||`：条件成立时执行后续语句或块。
- `window.size() != static_cast<std::size_t>(n) ||`：相邻声明、表达式或作用域的一部分。
- `normalization.size() != 1 ||`：相邻声明、表达式或作用域的一部分。
- `out.size() != static_cast<std::size_t>(n))`：相邻声明、表达式或作用域的一部分。
- `throw std::invalid_argument("firwin compute shape");`：拒绝非法 shape、指针或 fs。
- `if (scale) {`：条件成立时执行后续语句或块。
- `cuda_utils::launch_1d_kernel(`：调用统一一维 kernel launcher。
- `filter_design_detail::firwin_prepared_normalization_kernel,`：跨行形参或实参的一项。
- `256U, n, bands.data(), static_cast<int>(bands.size() / 2),`：elements=256，因此归一化 kernel 为 grid=1、block=256。
- `window.data(), normalization.data());`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `cuda_utils::launch_1d_kernel(`：调用统一一维 kernel launcher。
- `filter_design_detail::firwin_prepared_kernel, out.size(), n,`：elements=N，因此输出 kernel 的 grid=ceil(N/256)。
- `bands.data(), static_cast<int>(bands.size() / 2), window.data(),`：跨行形参或实参的一项。
- `normalization.data(), out.data(), scale);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。

#### FP64 存储两 kernel launcher

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cu:34`；符号：`firwin_fp64_storage_compute_device`；提交内第 34–59 行。

```cpp
void firwin_fp64_storage_compute_device(
    int n,
    const DeviceArray<float>& bands,
    const DeviceArray<float>& window,
    DeviceArray<float>& normalization,
    void* out_storage,
    std::size_t out_size,
    bool scale)
{
    if (n <= 0 || bands.empty() || bands.size() % 2 != 0 ||
        window.size() != static_cast<std::size_t>(n) ||
        normalization.size() != 1 ||
        out_size != static_cast<std::size_t>(n) || out_storage == nullptr)
        throw std::invalid_argument("firwin storage compute shape");
    if (scale) {
        cuda_utils::launch_1d_kernel(
            filter_design_detail::firwin_prepared_normalization_kernel,
            256U, n, bands.data(), static_cast<int>(bands.size() / 2),
            window.data(), normalization.data());
    }
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_prepared_fp64_storage_kernel,
        out_size, n, bands.data(), static_cast<int>(bands.size() / 2),
        window.data(), normalization.data(),
        static_cast<std::uint64_t*>(out_storage), scale);
}
```

计算仍为 FP32；第二 kernel 把每个 FP32 数值按位写成等值 IEEE-754 double 存储。

块内逐行说明：

- `void firwin_fp64_storage_compute_device(`：相邻声明、表达式或作用域的一部分。
- `int n,`：跨行形参或实参的一项。
- `const DeviceArray<float>& bands,`：跨行形参或实参的一项。
- `const DeviceArray<float>& window,`：跨行形参或实参的一项。
- `DeviceArray<float>& normalization,`：跨行形参或实参的一项。
- `void* out_storage,`：跨行形参或实参的一项。
- `std::size_t out_size,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `if (n <= 0 || bands.empty() || bands.size() % 2 != 0 ||`：条件成立时执行后续语句或块。
- `window.size() != static_cast<std::size_t>(n) ||`：相邻声明、表达式或作用域的一部分。
- `normalization.size() != 1 ||`：相邻声明、表达式或作用域的一部分。
- `out_size != static_cast<std::size_t>(n) || out_storage == nullptr)`：相邻声明、表达式或作用域的一部分。
- `throw std::invalid_argument("firwin storage compute shape");`：拒绝非法 shape、指针或 fs。
- `if (scale) {`：条件成立时执行后续语句或块。
- `cuda_utils::launch_1d_kernel(`：调用统一一维 kernel launcher。
- `filter_design_detail::firwin_prepared_normalization_kernel,`：跨行形参或实参的一项。
- `256U, n, bands.data(), static_cast<int>(bands.size() / 2),`：跨行形参或实参的一项。
- `window.data(), normalization.data());`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `cuda_utils::launch_1d_kernel(`：调用统一一维 kernel launcher。
- `filter_design_detail::firwin_prepared_fp64_storage_kernel,`：跨行形参或实参的一项。
- `out_size, n, bands.data(), static_cast<int>(bands.size() / 2),`：跨行形参或实参的一项。
- `window.data(), normalization.data(),`：跨行形参或实参的一项。
- `static_cast<std::uint64_t*>(out_storage), scale);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。

#### FP64 存储融合 launcher

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cu:61`；符号：`firwin_fused_fp64_storage_compute_device`；提交内第 61–78 行。

```cpp
void firwin_fused_fp64_storage_compute_device(
    int n,
    const DeviceArray<float>& bands,
    const DeviceArray<float>& window,
    void* out_storage,
    std::size_t out_size,
    bool scale)
{
    if (n <= 0 || n > filter_design_detail::firwin_fused_max_taps ||
        bands.empty() || bands.size() % 2 != 0 ||
        window.size() != static_cast<std::size_t>(n) ||
        out_size != static_cast<std::size_t>(n) || out_storage == nullptr)
        throw std::invalid_argument("firwin fused storage compute shape");
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_prepared_fused_fp64_storage_kernel,
        256U, n, bands.data(), static_cast<int>(bands.size() / 2),
        window.data(), static_cast<std::uint64_t*>(out_storage), scale);
}
```

N≤1024 时一个 256-thread block 同时缓存未缩放值、归约并写 FP64 存储。

块内逐行说明：

- `void firwin_fused_fp64_storage_compute_device(`：相邻声明、表达式或作用域的一部分。
- `int n,`：跨行形参或实参的一项。
- `const DeviceArray<float>& bands,`：跨行形参或实参的一项。
- `const DeviceArray<float>& window,`：跨行形参或实参的一项。
- `void* out_storage,`：跨行形参或实参的一项。
- `std::size_t out_size,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `if (n <= 0 || n > filter_design_detail::firwin_fused_max_taps ||`：条件成立时执行后续语句或块。
- `bands.empty() || bands.size() % 2 != 0 ||`：相邻声明、表达式或作用域的一部分。
- `window.size() != static_cast<std::size_t>(n) ||`：相邻声明、表达式或作用域的一部分。
- `out_size != static_cast<std::size_t>(n) || out_storage == nullptr)`：相邻声明、表达式或作用域的一部分。
- `throw std::invalid_argument("firwin fused storage compute shape");`：拒绝非法 shape、指针或 fs。
- `cuda_utils::launch_1d_kernel(`：调用统一一维 kernel launcher。
- `filter_design_detail::firwin_prepared_fused_fp64_storage_kernel,`：跨行形参或实参的一项。
- `256U, n, bands.data(), static_cast<int>(bands.size() / 2),`：融合 kernel 固定单 block 256 threads。
- `window.data(), static_cast<std::uint64_t*>(out_storage), scale);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。

#### 低通显式窗 resident launcher

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cu:80`；符号：`firwin_lowpass_explicit_window_resident_device`；提交内第 80–99 行。

```cpp
template <typename T>
void firwin_lowpass_explicit_window_resident_device(
    int numtaps, const DeviceArray<T>& cutoff,
    const DeviceArray<float>& window, DeviceArray<float>& output,
    float fs, bool scale)
{
    static_assert(detail::is_simple_signal_input_v<T>,
        "unsupported firwin resident cutoff dtype");
    if (numtaps <= 0 ||
        numtaps > filter_design_detail::firwin_fused_max_taps ||
        cutoff.size() != 1 ||
        window.size() != static_cast<std::size_t>(numtaps) ||
        output.size() != static_cast<std::size_t>(numtaps) ||
        !std::isfinite(fs) || fs <= 0.0F) {
        throw std::invalid_argument("firwin lowpass resident shape or fs");
    }
    cuda_utils::launch_1d_kernel(
        filter_design_detail::firwin_lowpass_explicit_window_fused_fp32_kernel<T>,
        256, numtaps, cutoff.data(), window.data(), fs, output.data(), scale);
}
```

五类型 cutoff 保持在 device；shape/fs 门禁后只启动一个特化 FP32 kernel。

块内逐行说明：

- `template <typename T>`：声明模板参数。
- `void firwin_lowpass_explicit_window_resident_device(`：相邻声明、表达式或作用域的一部分。
- `int numtaps, const DeviceArray<T>& cutoff,`：跨行形参或实参的一项。
- `const DeviceArray<float>& window, DeviceArray<float>& output,`：跨行形参或实参的一项。
- `float fs, bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `static_assert(detail::is_simple_signal_input_v<T>,`：编译期限制支持类型或存储宽度。
- `"unsupported firwin resident cutoff dtype");`：相邻声明、表达式或作用域的一部分。
- `if (numtaps <= 0 ||`：条件成立时执行后续语句或块。
- `numtaps > filter_design_detail::firwin_fused_max_taps ||`：相邻声明、表达式或作用域的一部分。
- `cutoff.size() != 1 ||`：相邻声明、表达式或作用域的一部分。
- `window.size() != static_cast<std::size_t>(numtaps) ||`：相邻声明、表达式或作用域的一部分。
- `output.size() != static_cast<std::size_t>(numtaps) ||`：相邻声明、表达式或作用域的一部分。
- `!std::isfinite(fs) || fs <= 0.0F) {`：相邻声明、表达式或作用域的一部分。
- `throw std::invalid_argument("firwin lowpass resident shape or fs");`：拒绝非法 shape、指针或 fs。
- `}`：开始或结束当前作用域。
- `cuda_utils::launch_1d_kernel(`：调用统一一维 kernel launcher。
- `filter_design_detail::firwin_lowpass_explicit_window_fused_fp32_kernel<T>,`：跨行形参或实参的一项。
- `256, numtaps, cutoff.data(), window.data(), fs, output.data(), scale);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。

#### resident 五类型实例化

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_typed.cu:134`；符号：`INSTANTIATE_FIRWIN_LOWPASS_RESIDENT`；提交内第 134–145 行。

```cpp
#define INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(T) \
    template void firwin_lowpass_explicit_window_resident_device<T>( \
        int, const DeviceArray<T>&, const DeviceArray<float>&, \
        DeviceArray<float>&, float, bool)

INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(float);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(__half);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int32_t);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int16_t);
INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int8_t);

#undef INSTANTIATE_FIRWIN_LOWPASS_RESIDENT
```

宏为 float、half、int32、int16、int8 生成特化入口。

块内逐行说明：

- `#define INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(T) \`：开始宏定义。
- `template void firwin_lowpass_explicit_window_resident_device<T>( \`：声明模板参数。
- `int, const DeviceArray<T>&, const DeviceArray<float>&, \`：宏定义续行。
- `DeviceArray<float>&, float, bool)`：相邻声明、表达式或作用域的一部分。
- `INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(float);`：完成函数调用、声明或构造。
- `INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(__half);`：完成函数调用、声明或构造。
- `INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int32_t);`：完成函数调用、声明或构造。
- `INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int16_t);`：完成函数调用、声明或构造。
- `INSTANTIATE_FIRWIN_LOWPASS_RESIDENT(std::int8_t);`：完成函数调用、声明或构造。
- `#undef INSTANTIATE_FIRWIN_LOWPASS_RESIDENT`：取消宏定义。

#### device 类型加载、存储与 sinc

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:1`；符号：`load / store / sinc`；提交内第 1–34 行。

```cpp
#pragma once

#include "filter_design_typed.h"
#include "fft_interface/fft_interface.h"

#include <cmath>
#include <cstdint>
namespace cusignal {
namespace filter_design_detail {

template <class T>
using Accumulator = float;

template <class T>
__device__ Accumulator<T> load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <class T, class Value>
__device__ T store(Value value)
{
    return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));
}

template <class Value>
__device__ Value sinc(Value value)
{
    const Value epsilon = static_cast<Value>(1.0e-7);
    const Value pi = static_cast<Value>(3.14159265358979323846);
    return fabs(value) < epsilon
        ? static_cast<Value>(1)
        : sin(pi * value) / (pi * value);
}
```

所有输入先按类型策略加载到 FP32；sinc 在 |x|<1e-7 时使用极限 1。

块内逐行说明：

- `#pragma once`：防止头文件重复包含。
- `#include "filter_design_typed.h"`：引入当前 helper 所需声明。
- `#include "fft_interface/fft_interface.h"`：引入当前 helper 所需声明。
- `#include <cmath>`：引入当前 helper 所需声明。
- `#include <cstdint>`：引入当前 helper 所需声明。
- `namespace cusignal {`：进入命名空间。
- `namespace filter_design_detail {`：进入命名空间。
- `template <class T>`：声明模板参数。
- `using Accumulator = float;`：初始化或更新变量。
- `template <class T>`：声明模板参数。
- `__device__ Accumulator<T> load(T value)`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `return detail::SimpleSignalTypePolicy<T>::load(value);`：返回当前结果。
- `}`：开始或结束当前作用域。
- `template <class T, class Value>`：声明模板参数。
- `__device__ T store(Value value)`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `return detail::SimpleSignalTypePolicy<T>::store(static_cast<float>(value));`：返回当前结果。
- `}`：开始或结束当前作用域。
- `template <class Value>`：声明模板参数。
- `__device__ Value sinc(Value value)`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `const Value epsilon = static_cast<Value>(1.0e-7);`：初始化或更新变量。
- `const Value pi = static_cast<Value>(3.14159265358979323846);`：初始化或更新变量。
- `return fabs(value) < epsilon`：以 1e-7 为 FP32 可去奇点阈值。
- `? static_cast<Value>(1)`：相邻声明、表达式或作用域的一部分。
- `: sin(pi * value) / (pi * value);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。

#### 多带理想响应

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:36`；符号：`firwin_prepared_ideal`；提交内第 36–48 行。

```cpp
template <class Value>
__device__ Value firwin_prepared_ideal(
    const float* bands, int band_count, Value sample_offset)
{
    Value result = static_cast<Value>(0);
    for (int band = 0; band < band_count; ++band) {
        const Value left = static_cast<Value>(bands[2 * band]);
        const Value right = static_cast<Value>(bands[2 * band + 1]);
        result += right * sinc(right * sample_offset)
            - left * sinc(left * sample_offset);
    }
    return result;
}
```

每个线程对所有 [left,right] 通带串行累加 sinc 差。

块内逐行说明：

- `template <class Value>`：声明模板参数。
- `__device__ Value firwin_prepared_ideal(`：定义 Device helper。
- `const float* bands, int band_count, Value sample_offset)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `Value result = static_cast<Value>(0);`：初始化或更新变量。
- `for (int band = 0; band < band_count; ++band) {`：按给定范围循环。
- `const Value left = static_cast<Value>(bands[2 * band]);`：初始化或更新变量。
- `const Value right = static_cast<Value>(bands[2 * band + 1]);`：初始化或更新变量。
- `result += right * sinc(right * sample_offset)`：加入右边界 sinc 项。
- `- left * sinc(left * sample_offset);`：减左边界 sinc 项，完成一个通带。
- `}`：开始或结束当前作用域。
- `return result;`：返回当前结果。
- `}`：开始或结束当前作用域。

#### FP32 到 FP64 存储位拓宽

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:50`；符号：`fp32_to_fp64_storage_bits`；提交内第 50–82 行。

```cpp
__device__ std::uint64_t fp32_to_fp64_storage_bits(float value)
{
    union FloatStorageBits {
        float value;
        std::uint32_t bits;
    } source{value};
    const std::uint64_t sign =
        static_cast<std::uint64_t>(source.bits >> 31U) << 63U;
    const std::uint32_t exponent = (source.bits >> 23U) & 0xffU;
    std::uint32_t fraction = source.bits & 0x7fffffU;
    if (exponent == 0U) {
        if (fraction == 0U) return sign;
        int shift = 0;
        while ((fraction & 0x400000U) == 0U) {
            fraction <<= 1U;
            ++shift;
        }
        const std::uint64_t widened_exponent =
            static_cast<std::uint64_t>(896 - shift) << 52U;
        const std::uint64_t widened_fraction =
            static_cast<std::uint64_t>(fraction & 0x3fffffU) << 30U;
        return sign | widened_exponent | widened_fraction;
    }
    if (exponent == 0xffU) {
        return sign | (UINT64_C(0x7ff) << 52U) |
            (static_cast<std::uint64_t>(fraction) << 29U);
    }
    const std::uint64_t widened_exponent =
        static_cast<std::uint64_t>(exponent + 896U) << 52U;
    const std::uint64_t widened_fraction =
        static_cast<std::uint64_t>(fraction) << 29U;
    return sign | widened_exponent | widened_fraction;
}
```

不执行 device FP64 算术，而是拆解 sign/exponent/fraction，构造数值完全等值的 double 位模式。

块内逐行说明：

- `__device__ std::uint64_t fp32_to_fp64_storage_bits(float value)`：定义 Device helper。
- `{`：开始或结束当前作用域。
- `union FloatStorageBits {`：相邻声明、表达式或作用域的一部分。
- `float value;`：相邻声明、表达式或作用域的一部分。
- `std::uint32_t bits;`：相邻声明、表达式或作用域的一部分。
- `} source{value};`：相邻声明、表达式或作用域的一部分。
- `const std::uint64_t sign =`：相邻声明、表达式或作用域的一部分。
- `static_cast<std::uint64_t>(source.bits >> 31U) << 63U;`：把 float 符号位移动到 double 符号位。
- `const std::uint32_t exponent = (source.bits >> 23U) & 0xffU;`：初始化或更新变量。
- `std::uint32_t fraction = source.bits & 0x7fffffU;`：初始化或更新变量。
- `if (exponent == 0U) {`：条件成立时执行后续语句或块。
- `if (fraction == 0U) return sign;`：条件成立时执行后续语句或块。
- `int shift = 0;`：初始化或更新变量。
- `while ((fraction & 0x400000U) == 0U) {`：循环规格化次正规 fraction。
- `fraction <<= 1U;`：初始化或更新变量。
- `++shift;`：相邻声明、表达式或作用域的一部分。
- `}`：开始或结束当前作用域。
- `const std::uint64_t widened_exponent =`：相邻声明、表达式或作用域的一部分。
- `static_cast<std::uint64_t>(896 - shift) << 52U;`：规格化 float 次正规数并换算 double 指数。
- `const std::uint64_t widened_fraction =`：相邻声明、表达式或作用域的一部分。
- `static_cast<std::uint64_t>(fraction & 0x3fffffU) << 30U;`：完成函数调用、声明或构造。
- `return sign | widened_exponent | widened_fraction;`：返回当前结果。
- `}`：开始或结束当前作用域。
- `if (exponent == 0xffU) {`：条件成立时执行后续语句或块。
- `return sign | (UINT64_C(0x7ff) << 52U) |`：返回当前结果。
- `(static_cast<std::uint64_t>(fraction) << 29U);`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `const std::uint64_t widened_exponent =`：相邻声明、表达式或作用域的一部分。
- `static_cast<std::uint64_t>(exponent + 896U) << 52U;`：正常数指数加 896，把偏置 127 转成 1023。
- `const std::uint64_t widened_fraction =`：相邻声明、表达式或作用域的一部分。
- `static_cast<std::uint64_t>(fraction) << 29U;`：完成函数调用、声明或构造。
- `return sign | widened_exponent | widened_fraction;`：返回当前结果。
- `}`：开始或结束当前作用域。

#### 单抽头未缩放值

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:84`；符号：`firwin_prepared_unscaled_value`；提交内第 84–92 行。

```cpp
__device__ float firwin_prepared_unscaled_value(
    int tap, int numtaps, const float* bands, int band_count,
    const float* window)
{
    const float midpoint = static_cast<float>(0.5) * (numtaps - 1);
    const float offset = tap - midpoint;
    return firwin_prepared_ideal(
        bands, band_count, offset) * window[tap];
}
```

由 tap 算中心偏移，调用多带理想响应并乘 window[tap]。

块内逐行说明：

- `__device__ float firwin_prepared_unscaled_value(`：定义 Device helper。
- `int tap, int numtaps, const float* bands, int band_count,`：跨行形参或实参的一项。
- `const float* window)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `const float midpoint = static_cast<float>(0.5) * (numtaps - 1);`：初始化或更新变量。
- `const float offset = tap - midpoint;`：初始化或更新变量。
- `return firwin_prepared_ideal(`：返回当前结果。
- `bands, band_count, offset) * window[tap];`：相邻声明、表达式或作用域的一部分。
- `}`：开始或结束当前作用域。

#### 归一化归约 kernel

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:94`；符号：`firwin_prepared_normalization_kernel`；提交内第 94–126 行。

```cpp
__global__ void firwin_prepared_normalization_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    float* normalization)
{
    __shared__ float partial[256];
    const int lane = static_cast<int>(threadIdx.x);
    const float midpoint = static_cast<float>(0.5) * (numtaps - 1);
    const float pi = static_cast<float>(3.14159265358979323846);
    const float left = bands[0];
    const float right = bands[1];
    const float scale_frequency = left == static_cast<float>(0)
        ? static_cast<float>(0)
        : (right == static_cast<float>(1)
            ? static_cast<float>(1)
            : static_cast<float>(0.5) * (left + right));
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = lane; index < numtaps; index += 256) {
        const float sample_offset = index - midpoint;
        sum.add(firwin_prepared_unscaled_value(
            index, numtaps, bands, band_count, window)
            * cos(pi * sample_offset * scale_frequency));
    }
    partial[lane] = sum.value();
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) normalization[0] = partial[0];
}
```

固定 256 threads 的单 block；每 lane 跨步处理若干 taps，补偿累加后在 shared memory 做树形归约。

块内逐行说明：

- `__global__ void firwin_prepared_normalization_kernel(`：定义 Host 可启动的 CUDA global kernel。
- `int numtaps,`：跨行形参或实参的一项。
- `const float* bands,`：跨行形参或实参的一项。
- `int band_count,`：跨行形参或实参的一项。
- `const float* window,`：跨行形参或实参的一项。
- `float* normalization)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `__shared__ float partial[256];`：256 个部分和放在 block shared memory。
- `const int lane = static_cast<int>(threadIdx.x);`：初始化或更新变量。
- `const float midpoint = static_cast<float>(0.5) * (numtaps - 1);`：初始化或更新变量。
- `const float pi = static_cast<float>(3.14159265358979323846);`：初始化或更新变量。
- `const float left = bands[0];`：初始化或更新变量。
- `const float right = bands[1];`：初始化或更新变量。
- `const float scale_frequency = left == static_cast<float>(0)`：相邻声明、表达式或作用域的一部分。
- `? static_cast<float>(0)`：相邻声明、表达式或作用域的一部分。
- `: (right == static_cast<float>(1)`：相邻声明、表达式或作用域的一部分。
- `? static_cast<float>(1)`：相邻声明、表达式或作用域的一部分。
- `: static_cast<float>(0.5) * (left + right));`：完成函数调用、声明或构造。
- `compute_policy::CompensatedFloatAccumulator sum;`：相邻声明、表达式或作用域的一部分。
- `for (int index = lane; index < numtaps; index += 256) {`：每 lane 以步长 256 遍历 taps。
- `const float sample_offset = index - midpoint;`：初始化或更新变量。
- `sum.add(firwin_prepared_unscaled_value(`：相邻声明、表达式或作用域的一部分。
- `index, numtaps, bands, band_count, window)`：相邻声明、表达式或作用域的一部分。
- `* cos(pi * sample_offset * scale_frequency));`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `partial[lane] = sum.value();`：初始化或更新变量。
- `__syncthreads();`：block 屏障确保 partial 已写完。
- `for (int stride = 128; stride > 0; stride >>= 1) {`：树形归约从 stride=128 每轮减半。
- `if (lane < stride) partial[lane] += partial[lane + stride];`：条件成立时执行后续语句或块。
- `__syncthreads();`：block 级同步。
- `}`：开始或结束当前作用域。
- `if (lane == 0) normalization[0] = partial[0];`：条件成立时执行后续语句或块。
- `}`：开始或结束当前作用域。

#### FP32 输出 kernel

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:128`；符号：`firwin_prepared_kernel`；提交内第 128–142 行。

```cpp
__global__ void firwin_prepared_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    const float* normalization,
    float* output,
    bool scale)
{
    const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tap >= numtaps) return;
    const float divisor = scale ? normalization[0] : static_cast<float>(1);
    output[tap] = firwin_prepared_unscaled_value(
        tap, numtaps, bands, band_count, window) / divisor;
}
```

一线程负责一个 tap，越界线程退出；按 scale 选择分母后写 FP32。

块内逐行说明：

- `__global__ void firwin_prepared_kernel(`：定义 Host 可启动的 CUDA global kernel。
- `int numtaps,`：跨行形参或实参的一项。
- `const float* bands,`：跨行形参或实参的一项。
- `int band_count,`：跨行形参或实参的一项。
- `const float* window,`：跨行形参或实参的一项。
- `const float* normalization,`：跨行形参或实参的一项。
- `float* output,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);`：全局线程号决定一个线程负责的 tap。
- `if (tap >= numtaps) return;`：条件成立时执行后续语句或块。
- `const float divisor = scale ? normalization[0] : static_cast<float>(1);`：初始化或更新变量。
- `output[tap] = firwin_prepared_unscaled_value(`：相邻声明、表达式或作用域的一部分。
- `tap, numtaps, bands, band_count, window) / divisor;`：相邻声明、表达式或作用域的一部分。
- `}`：开始或结束当前作用域。

#### FP64 存储输出 kernel

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:144`；符号：`firwin_prepared_fp64_storage_kernel`；提交内第 144–159 行。

```cpp
__global__ void firwin_prepared_fp64_storage_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    const float* normalization,
    std::uint64_t* output,
    bool scale)
{
    const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tap >= numtaps) return;
    const float divisor = scale ? normalization[0] : static_cast<float>(1);
    output[tap] = fp32_to_fp64_storage_bits(
        firwin_prepared_unscaled_value(
            tap, numtaps, bands, band_count, window) / divisor);
}
```

一线程负责一个 tap，FP32 除法后通过位拓宽写入 uint64 视图。

块内逐行说明：

- `__global__ void firwin_prepared_fp64_storage_kernel(`：定义 Host 可启动的 CUDA global kernel。
- `int numtaps,`：跨行形参或实参的一项。
- `const float* bands,`：跨行形参或实参的一项。
- `int band_count,`：跨行形参或实参的一项。
- `const float* window,`：跨行形参或实参的一项。
- `const float* normalization,`：跨行形参或实参的一项。
- `std::uint64_t* output,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `const int tap = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);`：初始化或更新变量。
- `if (tap >= numtaps) return;`：条件成立时执行后续语句或块。
- `const float divisor = scale ? normalization[0] : static_cast<float>(1);`：初始化或更新变量。
- `output[tap] = fp32_to_fp64_storage_bits(`：相邻声明、表达式或作用域的一部分。
- `firwin_prepared_unscaled_value(`：相邻声明、表达式或作用域的一部分。
- `tap, numtaps, bands, band_count, window) / divisor);`：相邻声明、表达式或作用域的一部分。
- `}`：开始或结束当前作用域。

#### 融合 FP64 存储 kernel

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:161`；符号：`firwin_prepared_fused_fp64_storage_kernel`；提交内第 161–200 行。

```cpp
inline constexpr int firwin_fused_max_taps = 1024;

__global__ void firwin_prepared_fused_fp64_storage_kernel(
    int numtaps,
    const float* bands,
    int band_count,
    const float* window,
    std::uint64_t* output,
    bool scale)
{
    __shared__ float unscaled[firwin_fused_max_taps];
    __shared__ float partial[256];
    const int lane = static_cast<int>(threadIdx.x);
    const float midpoint = static_cast<float>(0.5) * (numtaps - 1);
    const float pi = static_cast<float>(3.14159265358979323846);
    const float left = bands[0];
    const float right = bands[1];
    const float scale_frequency = left == static_cast<float>(0)
        ? static_cast<float>(0)
        : (right == static_cast<float>(1)
            ? static_cast<float>(1)
            : static_cast<float>(0.5) * (left + right));
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = lane; index < numtaps; index += 256) {
        const float value = firwin_prepared_unscaled_value(
            index, numtaps, bands, band_count, window);
        unscaled[index] = value;
        if (scale)
            sum.add(value * cos(pi * (index - midpoint) * scale_frequency));
    }
    partial[lane] = sum.value();
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float divisor = scale ? partial[0] : static_cast<float>(1);
    for (int index = lane; index < numtaps; index += 256)
        output[index] = fp32_to_fp64_storage_bits(unscaled[index] / divisor);
}
```

一个 block 把最多 1024 个未缩放值缓存到 shared memory，归约一次，再由各 lane 跨步写输出。

块内逐行说明：

- `inline constexpr int firwin_fused_max_taps = 1024;`：初始化或更新变量。
- `__global__ void firwin_prepared_fused_fp64_storage_kernel(`：定义 Host 可启动的 CUDA global kernel。
- `int numtaps,`：跨行形参或实参的一项。
- `const float* bands,`：跨行形参或实参的一项。
- `int band_count,`：跨行形参或实参的一项。
- `const float* window,`：跨行形参或实参的一项。
- `std::uint64_t* output,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `__shared__ float unscaled[firwin_fused_max_taps];`：共享数组固定 1024 floats，形成融合上限。
- `__shared__ float partial[256];`：声明 block 内共享内存。
- `const int lane = static_cast<int>(threadIdx.x);`：初始化或更新变量。
- `const float midpoint = static_cast<float>(0.5) * (numtaps - 1);`：初始化或更新变量。
- `const float pi = static_cast<float>(3.14159265358979323846);`：初始化或更新变量。
- `const float left = bands[0];`：初始化或更新变量。
- `const float right = bands[1];`：初始化或更新变量。
- `const float scale_frequency = left == static_cast<float>(0)`：相邻声明、表达式或作用域的一部分。
- `? static_cast<float>(0)`：相邻声明、表达式或作用域的一部分。
- `: (right == static_cast<float>(1)`：相邻声明、表达式或作用域的一部分。
- `? static_cast<float>(1)`：相邻声明、表达式或作用域的一部分。
- `: static_cast<float>(0.5) * (left + right));`：完成函数调用、声明或构造。
- `compute_policy::CompensatedFloatAccumulator sum;`：相邻声明、表达式或作用域的一部分。
- `for (int index = lane; index < numtaps; index += 256) {`：按给定范围循环。
- `const float value = firwin_prepared_unscaled_value(`：相邻声明、表达式或作用域的一部分。
- `index, numtaps, bands, band_count, window);`：相邻声明、表达式或作用域的一部分。
- `unscaled[index] = value;`：初始化或更新变量。
- `if (scale)`：仅 scale=True 时累加归一化分母。
- `sum.add(value * cos(pi * (index - midpoint) * scale_frequency));`：完成函数调用、声明或构造。
- `}`：开始或结束当前作用域。
- `partial[lane] = sum.value();`：初始化或更新变量。
- `__syncthreads();`：block 级同步。
- `for (int stride = 128; stride > 0; stride >>= 1) {`：按给定范围循环。
- `if (lane < stride) partial[lane] += partial[lane + stride];`：条件成立时执行后续语句或块。
- `__syncthreads();`：block 级同步。
- `}`：开始或结束当前作用域。
- `const float divisor = scale ? partial[0] : static_cast<float>(1);`：初始化或更新变量。
- `for (int index = lane; index < numtaps; index += 256)`：每 lane 以步长 256 写若干输出。
- `output[index] = fp32_to_fp64_storage_bits(unscaled[index] / divisor);`：初始化或更新变量。
- `}`：开始或结束当前作用域。

#### 融合低通显式窗 FP32 kernel

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/src/filter_design/filter_design_kernels.cuh:202`；符号：`firwin_lowpass_explicit_window_fused_fp32_kernel`；提交内第 202–233 行。

```cpp
template <class T>
__global__ void firwin_lowpass_explicit_window_fused_fp32_kernel(
    int numtaps,
    const T* cutoff,
    const float* window,
    float fs,
    float* output,
    bool scale)
{
    __shared__ float unscaled[firwin_fused_max_taps];
    __shared__ float partial[256];
    const int lane = static_cast<int>(threadIdx.x);
    const float normalized_cutoff = load(cutoff[0]) / (0.5F * fs);
    const float midpoint = 0.5F * (numtaps - 1);
    compute_policy::CompensatedFloatAccumulator sum;
    for (int index = lane; index < numtaps; index += 256) {
        const float offset = index - midpoint;
        const float value = normalized_cutoff * sinc(normalized_cutoff * offset)
            * window[index];
        unscaled[index] = value;
        if (scale) sum.add(value);
    }
    partial[lane] = sum.value();
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float divisor = scale ? partial[0] : 1.0F;
    for (int index = lane; index < numtaps; index += 256)
        output[index] = unscaled[index] / divisor;
}
```

单 cutoff 低通把公式简化为 c*sinc(c*m)*window；scale 时 DC 分母就是系数和。

块内逐行说明：

- `template <class T>`：声明模板参数。
- `__global__ void firwin_lowpass_explicit_window_fused_fp32_kernel(`：定义 Host 可启动的 CUDA global kernel。
- `int numtaps,`：跨行形参或实参的一项。
- `const T* cutoff,`：跨行形参或实参的一项。
- `const float* window,`：跨行形参或实参的一项。
- `float fs,`：跨行形参或实参的一项。
- `float* output,`：跨行形参或实参的一项。
- `bool scale)`：相邻声明、表达式或作用域的一部分。
- `{`：开始或结束当前作用域。
- `__shared__ float unscaled[firwin_fused_max_taps];`：声明 block 内共享内存。
- `__shared__ float partial[256];`：声明 block 内共享内存。
- `const int lane = static_cast<int>(threadIdx.x);`：初始化或更新变量。
- `const float normalized_cutoff = load(cutoff[0]) / (0.5F * fs);`：device cutoff 经类型策略加载并除以 fs/2。
- `const float midpoint = 0.5F * (numtaps - 1);`：初始化或更新变量。
- `compute_policy::CompensatedFloatAccumulator sum;`：相邻声明、表达式或作用域的一部分。
- `for (int index = lane; index < numtaps; index += 256) {`：按给定范围循环。
- `const float offset = index - midpoint;`：初始化或更新变量。
- `const float value = normalized_cutoff * sinc(normalized_cutoff * offset)`：单低通理想响应 c*sinc(c*m)。
- `* window[index];`：相邻声明、表达式或作用域的一部分。
- `unscaled[index] = value;`：初始化或更新变量。
- `if (scale) sum.add(value);`：低通参考是 DC，所以分母是系数和。
- `}`：开始或结束当前作用域。
- `partial[lane] = sum.value();`：初始化或更新变量。
- `__syncthreads();`：block 级同步。
- `for (int stride = 128; stride > 0; stride >>= 1) {`：按给定范围循环。
- `if (lane < stride) partial[lane] += partial[lane + stride];`：条件成立时执行后续语句或块。
- `__syncthreads();`：block 级同步。
- `}`：开始或结束当前作用域。
- `const float divisor = scale ? partial[0] : 1.0F;`：初始化或更新变量。
- `for (int index = lane; index < numtaps; index += 256)`：按给定范围循环。
- `output[index] = unscaled[index] / divisor;`：初始化或更新变量。
- `}`：开始或结束当前作用域。

### 6. 直接测试与构建注册逐块解释

#### 测试依赖与数值比较 helper

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:1`；符号：`value / same / same_float`；提交内第 1–4 行。

```cpp
#include "estimation/estimation_typed.h"
#include "filter_design/filter_design_typed.h"
#include "windows/windows_typed.h"
#include "e_type_test_evidence.h"
```

直接包含 typed firwin 与窗口接口。

块内逐行说明：

- `#include "estimation/estimation_typed.h"`：引入测试所需当前 typed API 或证据 helper。
- `#include "filter_design/filter_design_typed.h"`：引入测试所需当前 typed API 或证据 helper。
- `#include "windows/windows_typed.h"`：引入测试所需当前 typed API 或证据 helper。
- `#include "e_type_test_evidence.h"`：引入测试所需当前 typed API 或证据 helper。

#### 测试类型转换与容差

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:16`；符号：`value / scalar / same / same_float`；提交内第 16–33 行。

```cpp
template <typename T> T value(float x) { return detail::SimpleSignalTypePolicy<T>::store(x); }
template <typename T> double scalar(T x) {
    if constexpr (std::is_same_v<T, double>) return x;
    return static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(x));
}

template <typename T>
bool same(const std::vector<T>& a, const std::vector<T>& b)
{
    return test_evidence::f_observe_real(a, b, 2.0e-3);
}

bool same_float(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 2.0e-4F)
{
    return test_evidence::f_observe_real(actual, expected, tolerance);
```

value 生成五类型输入，same 以 2e-3 比较 FP64 输出，resident FP32 另用 2e-4。

块内逐行说明：

- `template <typename T> T value(float x) { return detail::SimpleSignalTypePolicy<T>::store(x); }`：声明模板，使同一用例按五种输入类型实例化。
- `template <typename T> double scalar(T x) {`：声明模板，使同一用例按五种输入类型实例化。
- `if constexpr (std::is_same_v<T, double>) return x;`：条件不满足时报告失败或选择对应路径。
- `return static_cast<double>(detail::SimpleSignalTypePolicy<T>::load(x));`：返回比较结果或进程状态。
- `}`：开始或结束当前作用域/调用。
- `template <typename T>`：声明模板，使同一用例按五种输入类型实例化。
- `bool same(const std::vector<T>& a, const std::vector<T>& b)`：比较 GPU FP64 存储输出与 CPU reference。
- `{`：开始或结束当前作用域/调用。
- `return test_evidence::f_observe_real(a, b, 2.0e-3);`：返回比较结果或进程状态。
- `}`：开始或结束当前作用域/调用。
- `bool same_float(`：以 resident FP32 容差比较实际值与参考值。
- `const std::vector<float>& actual,`：跨行参数、初始化器或逻辑表达式的一项。
- `const std::vector<float>& expected,`：跨行参数、初始化器或逻辑表达式的一项。
- `float tolerance = 2.0e-4F)`：相邻测试表达式、声明或作用域的一部分。
- `{`：开始或结束当前作用域/调用。
- `return test_evidence::f_observe_real(actual, expected, tolerance);`：返回比较结果或进程状态。

#### 五类型测试函数上下文

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:101`；符号：`run_type<T>`；提交内第 101–103 行。

```cpp
template <typename T>
int run_type(const char* name)
{
```

firwin 用例位于同一 run_type<T>，main 会对五种 T 重复执行。

块内逐行说明：

- `template <typename T>`：声明模板，使同一用例按五种输入类型实例化。
- `int run_type(const char* name)`：相邻测试表达式、声明或作用域的一部分。
- `{`：开始或结束当前作用域/调用。

#### 低通与带通基线

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:300`；符号：`run_type<T> firwin setup`；提交内第 300–313 行。

```cpp
    test_evidence::f_begin_accuracy(type_dispatch::OperatorId::firwin);
    const std::vector<T> cutoff{value<T>(1)};
    FirwinOptions fw_options;
    fw_options.fs = 8.0;
    const auto fw_ref = firwin_typed_cpu<T>(17, cutoff, fw_options);
    auto dcut = DeviceArray<T>::from_host(cutoff);
    DeviceArray<double> dfw(17);
    const std::vector<T> band_cutoff{value<T>(1), value<T>(3)};
    FirwinOptions band_options = fw_options;
    band_options.pass_zero = FirwinPassZero::bandpass;
    const auto band_ref = firwin_typed_cpu<T>(16, band_cutoff, band_options);
    auto d_band_cutoff = DeviceArray<T>::from_host(band_cutoff);
    DeviceArray<double> d_band(16);
    firwin_device(16, d_band_cutoff, d_band, band_options);
```

构造 fs=8、cutoff=1 的 17-tap 低通 CPU reference，以及 16-tap 带通 CPU/GPU 对照。

块内逐行说明：

- `test_evidence::f_begin_accuracy(type_dispatch::OperatorId::firwin);`：完成函数调用或对象构造。
- `const std::vector<T> cutoff{value<T>(1)};`：完成函数调用或对象构造。
- `FirwinOptions fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `fw_options.fs = 8.0;`：初始化或更新测试变量。
- `const auto fw_ref = firwin_typed_cpu<T>(17, cutoff, fw_options);`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `auto dcut = DeviceArray<T>::from_host(cutoff);`：声明或由 Host 数据构造 device 数组。
- `DeviceArray<double> dfw(17);`：声明或由 Host 数据构造 device 数组。
- `const std::vector<T> band_cutoff{value<T>(1), value<T>(3)};`：完成函数调用或对象构造。
- `FirwinOptions band_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `band_options.pass_zero = FirwinPassZero::bandpass;`：设置当前通带/阻带起始语义。
- `const auto band_ref = firwin_typed_cpu<T>(16, band_cutoff, band_options);`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `auto d_band_cutoff = DeviceArray<T>::from_host(band_cutoff);`：声明或由 Host 数据构造 device 数组。
- `DeviceArray<double> d_band(16);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(16, d_band_cutoff, d_band, band_options);`：调用当前公开 GPU compatibility API。

#### 非法 Nyquist 与三条 resident 路径

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:314`；符号：`run_type<T> firwin validation/resident`；提交内第 314–332 行。

```cpp
    bool firwin_device_rejects_nyquist = false;
    bool firwin_cpu_rejects_nyquist = false;
    const std::vector<T> bad_cutoff{value<T>(4)};
    auto dbad_cutoff = DeviceArray<T>::from_host(bad_cutoff);
    try { firwin_device(17, dbad_cutoff, dfw, fw_options); }
    catch (const std::invalid_argument&) { firwin_device_rejects_nyquist = true; }
    try { (void)firwin_typed_cpu<T>(17, bad_cutoff, fw_options); }
    catch (const std::invalid_argument&) { firwin_cpu_rejects_nyquist = true; }
    firwin_device(17, dcut, dfw, fw_options);
    FirwinDeviceWorkspace resident_workspace(
        17, cutoff, fw_options);
    DeviceArray<float> resident_out(17);
    firwin_resident_device(resident_workspace, resident_out);
    DeviceArray<double> resident_fp64_storage_out(17);
    firwin_resident_fp64_storage_device(
        resident_workspace, resident_fp64_storage_out);
    DeviceArray<double> resident_stage_c_out(17);
    firwin_resident_stage_c_fp64_storage_device(
        resident_workspace, resident_stage_c_out);
```

验证 cutoff=Nyquist 被 CPU/GPU 拒绝，并执行 FP32 resident、融合 FP64 存储和 Stage C。

块内逐行说明：

- `bool firwin_device_rejects_nyquist = false;`：调用当前公开 GPU compatibility API。
- `bool firwin_cpu_rejects_nyquist = false;`：初始化或更新测试变量。
- `const std::vector<T> bad_cutoff{value<T>(4)};`：完成函数调用或对象构造。
- `auto dbad_cutoff = DeviceArray<T>::from_host(bad_cutoff);`：声明或由 Host 数据构造 device 数组。
- `try { firwin_device(17, dbad_cutoff, dfw, fw_options); }`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `catch (const std::invalid_argument&) { firwin_device_rejects_nyquist = true; }`：捕获 invalid_argument 并记录门禁按预期生效。
- `try { (void)firwin_typed_cpu<T>(17, bad_cutoff, fw_options); }`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `catch (const std::invalid_argument&) { firwin_cpu_rejects_nyquist = true; }`：捕获 invalid_argument 并记录门禁按预期生效。
- `firwin_device(17, dcut, dfw, fw_options);`：调用当前公开 GPU compatibility API。
- `FirwinDeviceWorkspace resident_workspace(`：相邻测试表达式、声明或作用域的一部分。
- `17, cutoff, fw_options);`：相邻测试表达式、声明或作用域的一部分。
- `DeviceArray<float> resident_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_resident_device(resident_workspace, resident_out);`：调用预准备 resident 路径。
- `DeviceArray<double> resident_fp64_storage_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_resident_fp64_storage_device(`：调用预准备 resident 路径。
- `resident_workspace, resident_fp64_storage_out);`：相邻测试表达式、声明或作用域的一部分。
- `DeviceArray<double> resident_stage_c_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_resident_stage_c_fp64_storage_device(`：调用预准备 resident 路径。
- `resident_workspace, resident_stage_c_out);`：相邻测试表达式、声明或作用域的一部分。

#### device 显式 Hamming 特化

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:333`；符号：`run_type<T> explicit resident`；提交内第 333–348 行。

```cpp
    std::vector<float> resident_ref(fw_ref.begin(), fw_ref.end());
    DeviceArray<float> resident_hamming(17);
    hamming_resident_device<T>(17, resident_hamming, true);
    const auto resident_hamming_host = resident_hamming.to_host();
    FirwinOptions resident_explicit_options = fw_options;
    resident_explicit_options.window_mode = FirwinWindowMode::explicit_values;
    resident_explicit_options.window.assign(
        resident_hamming_host.begin(), resident_hamming_host.end());
    const auto resident_explicit_ref = firwin_typed_cpu<T>(
        17, cutoff, resident_explicit_options);
    std::vector<float> resident_explicit_ref_fp32(
        resident_explicit_ref.begin(), resident_explicit_ref.end());
    DeviceArray<float> resident_explicit_out(17);
    firwin_lowpass_explicit_window_resident_device(
        17, dcut, resident_hamming, resident_explicit_out,
        static_cast<float>(fw_options.fs), true);
```

先在 device 生成 Hamming，再把相同窗送入 CPU reference 与特化 resident kernel。

块内逐行说明：

- `std::vector<float> resident_ref(fw_ref.begin(), fw_ref.end());`：完成函数调用或对象构造。
- `DeviceArray<float> resident_hamming(17);`：声明或由 Host 数据构造 device 数组。
- `hamming_resident_device<T>(17, resident_hamming, true);`：完成函数调用或对象构造。
- `const auto resident_hamming_host = resident_hamming.to_host();`：初始化或更新测试变量。
- `FirwinOptions resident_explicit_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `resident_explicit_options.window_mode = FirwinWindowMode::explicit_values;`：选择 Hamming、none 或显式窗分支。
- `resident_explicit_options.window.assign(`：相邻测试表达式、声明或作用域的一部分。
- `resident_hamming_host.begin(), resident_hamming_host.end());`：完成函数调用或对象构造。
- `const auto resident_explicit_ref = firwin_typed_cpu<T>(`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `17, cutoff, resident_explicit_options);`：相邻测试表达式、声明或作用域的一部分。
- `std::vector<float> resident_explicit_ref_fp32(`：相邻测试表达式、声明或作用域的一部分。
- `resident_explicit_ref.begin(), resident_explicit_ref.end());`：完成函数调用或对象构造。
- `DeviceArray<float> resident_explicit_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_lowpass_explicit_window_resident_device(`：相邻测试表达式、声明或作用域的一部分。
- `17, dcut, resident_hamming, resident_explicit_out,`：跨行参数、初始化器或逻辑表达式的一项。
- `static_cast<float>(fw_options.fs), true);`：完成函数调用或对象构造。

#### 单 tap、Kaiser、显式窗与 pass_zero 模式

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:349`；符号：`run_type<T> firwin modes`；提交内第 349–384 行。

```cpp
    DeviceArray<double> single_tap(1);
    firwin_device(1, dcut, single_tap, fw_options);
    FirwinOptions width_options = fw_options;
    width_options.use_width = true;
    width_options.width = 0.5;
    width_options.window_mode = FirwinWindowMode::explicit_values;
    // cuSignal ignores even an otherwise-invalid window specification when
    // width selects the Kaiser design path.
    width_options.window.clear();
    DeviceArray<double> width_out(17);
    firwin_device(17, dcut, width_out, width_options);
    FirwinOptions explicit_fw_options = fw_options;
    explicit_fw_options.window_mode = FirwinWindowMode::explicit_values;
    explicit_fw_options.window.assign(17, 0.5F);
    DeviceArray<double> explicit_fw_out(17);
    firwin_device(17, dcut, explicit_fw_out, explicit_fw_options);
    FirwinOptions none_fw_options = fw_options;
    none_fw_options.window_mode = FirwinWindowMode::none;
    DeviceArray<double> none_fw_out(17);
    firwin_device(17, dcut, none_fw_out, none_fw_options);
    FirwinOptions highpass_options = fw_options;
    highpass_options.pass_zero = FirwinPassZero::highpass;
    DeviceArray<double> highpass_out(17);
    firwin_device(17, dcut, highpass_out, highpass_options);
    FirwinOptions boolean_false_options = fw_options;
    boolean_false_options.pass_zero = FirwinPassZero::boolean_false;
    DeviceArray<double> boolean_false_out(17);
    firwin_device(17, dcut, boolean_false_out, boolean_false_options);
    FirwinOptions bandstop_options = fw_options;
    bandstop_options.pass_zero = FirwinPassZero::bandstop;
    DeviceArray<double> bandstop_out(17);
    firwin_device(17, d_band_cutoff, bandstop_out, bandstop_options);
    FirwinOptions unscaled_options = fw_options;
    unscaled_options.scale = false;
    DeviceArray<double> unscaled_out(17);
    firwin_device(17, dcut, unscaled_out, unscaled_options);
```

覆盖 N=1、width 覆盖 window、explicit/none、高通、boolean_false、bandstop 和 scale=false。

块内逐行说明：

- `DeviceArray<double> single_tap(1);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(1, dcut, single_tap, fw_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions width_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `width_options.use_width = true;`：启用并设置 Kaiser width 路径。
- `width_options.width = 0.5;`：启用并设置 Kaiser width 路径。
- `width_options.window_mode = FirwinWindowMode::explicit_values;`：选择 Hamming、none 或显式窗分支。
- `// cuSignal ignores even an otherwise-invalid window specification when`：cuSignal ignores even an otherwise-invalid window specification when
- `// width selects the Kaiser design path.`：width selects the Kaiser design path.
- `width_options.window.clear();`：完成函数调用或对象构造。
- `DeviceArray<double> width_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, dcut, width_out, width_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions explicit_fw_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `explicit_fw_options.window_mode = FirwinWindowMode::explicit_values;`：选择 Hamming、none 或显式窗分支。
- `explicit_fw_options.window.assign(17, 0.5F);`：完成函数调用或对象构造。
- `DeviceArray<double> explicit_fw_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, dcut, explicit_fw_out, explicit_fw_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions none_fw_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `none_fw_options.window_mode = FirwinWindowMode::none;`：选择 Hamming、none 或显式窗分支。
- `DeviceArray<double> none_fw_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, dcut, none_fw_out, none_fw_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions highpass_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `highpass_options.pass_zero = FirwinPassZero::highpass;`：设置当前通带/阻带起始语义。
- `DeviceArray<double> highpass_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, dcut, highpass_out, highpass_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions boolean_false_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `boolean_false_options.pass_zero = FirwinPassZero::boolean_false;`：设置当前通带/阻带起始语义。
- `DeviceArray<double> boolean_false_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, dcut, boolean_false_out, boolean_false_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions bandstop_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `bandstop_options.pass_zero = FirwinPassZero::bandstop;`：设置当前通带/阻带起始语义。
- `DeviceArray<double> bandstop_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, d_band_cutoff, bandstop_out, bandstop_options);`：调用当前公开 GPU compatibility API。
- `FirwinOptions unscaled_options = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `unscaled_options.scale = false;`：设置是否执行单位增益归一化。
- `DeviceArray<double> unscaled_out(17);`：声明或由 Host 数据构造 device 数组。
- `firwin_device(17, dcut, unscaled_out, unscaled_options);`：调用当前公开 GPU compatibility API。

#### 非法模式与长度门禁

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:385`；符号：`run_type<T> firwin rejection`；提交内第 385–410 行。

```cpp
    bool firwin_rejects_lowpass_multicutoff = false;
    FirwinOptions invalid_lowpass = fw_options;
    invalid_lowpass.pass_zero = FirwinPassZero::lowpass;
    try { (void)firwin_typed_cpu<T>(17, band_cutoff, invalid_lowpass); }
    catch (const std::invalid_argument&) {
        firwin_rejects_lowpass_multicutoff = true;
    }
    bool firwin_rejects_bandstop_single_cutoff = false;
    FirwinOptions invalid_bandstop = fw_options;
    invalid_bandstop.pass_zero = FirwinPassZero::bandstop;
    try { (void)firwin_typed_cpu<T>(17, cutoff, invalid_bandstop); }
    catch (const std::invalid_argument&) {
        firwin_rejects_bandstop_single_cutoff = true;
    }
    bool firwin_rejects_explicit_window_length = false;
    FirwinOptions invalid_window = fw_options;
    invalid_window.window_mode = FirwinWindowMode::explicit_values;
    try { (void)firwin_typed_cpu<T>(17, cutoff, invalid_window); }
    catch (const std::invalid_argument&) {
        firwin_rejects_explicit_window_length = true;
    }
    bool firwin_rejects_even_highpass = false;
    try { (void)firwin_typed_cpu<T>(16, cutoff, highpass_options); }
    catch (const std::invalid_argument&) {
        firwin_rejects_even_highpass = true;
    }
```

验证 lowpass 多边界、bandstop 单边界、显式窗长度错和偶数高通均抛 invalid_argument。

块内逐行说明：

- `bool firwin_rejects_lowpass_multicutoff = false;`：初始化或更新测试变量。
- `FirwinOptions invalid_lowpass = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `invalid_lowpass.pass_zero = FirwinPassZero::lowpass;`：设置当前通带/阻带起始语义。
- `try { (void)firwin_typed_cpu<T>(17, band_cutoff, invalid_lowpass); }`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `catch (const std::invalid_argument&) {`：捕获 invalid_argument 并记录门禁按预期生效。
- `firwin_rejects_lowpass_multicutoff = true;`：初始化或更新测试变量。
- `}`：开始或结束当前作用域/调用。
- `bool firwin_rejects_bandstop_single_cutoff = false;`：初始化或更新测试变量。
- `FirwinOptions invalid_bandstop = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `invalid_bandstop.pass_zero = FirwinPassZero::bandstop;`：设置当前通带/阻带起始语义。
- `try { (void)firwin_typed_cpu<T>(17, cutoff, invalid_bandstop); }`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `catch (const std::invalid_argument&) {`：捕获 invalid_argument 并记录门禁按预期生效。
- `firwin_rejects_bandstop_single_cutoff = true;`：初始化或更新测试变量。
- `}`：开始或结束当前作用域/调用。
- `bool firwin_rejects_explicit_window_length = false;`：初始化或更新测试变量。
- `FirwinOptions invalid_window = fw_options;`：创建或复制选项结构，随后只修改当前场景字段。
- `invalid_window.window_mode = FirwinWindowMode::explicit_values;`：选择 Hamming、none 或显式窗分支。
- `try { (void)firwin_typed_cpu<T>(17, cutoff, invalid_window); }`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `catch (const std::invalid_argument&) {`：捕获 invalid_argument 并记录门禁按预期生效。
- `firwin_rejects_explicit_window_length = true;`：初始化或更新测试变量。
- `}`：开始或结束当前作用域/调用。
- `bool firwin_rejects_even_highpass = false;`：初始化或更新测试变量。
- `try { (void)firwin_typed_cpu<T>(16, cutoff, highpass_options); }`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `catch (const std::invalid_argument&) {`：捕获 invalid_argument 并记录门禁按预期生效。
- `firwin_rejects_even_highpass = true;`：初始化或更新测试变量。
- `}`：开始或结束当前作用域/调用。

#### 聚合全部正确性判据

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:411`；符号：`firwin_ok`；提交内第 411–436 行。

```cpp
    const bool firwin_ok =
        firwin_device_rejects_nyquist && firwin_cpu_rejects_nyquist &&
        same(dfw.to_host(), fw_ref) &&
        same_float(resident_out.to_host(), resident_ref) &&
        same(resident_fp64_storage_out.to_host(), fw_ref) &&
        same(resident_stage_c_out.to_host(), fw_ref) &&
        same_float(resident_explicit_out.to_host(), resident_explicit_ref_fp32) &&
        same(d_band.to_host(), band_ref) &&
        same(single_tap.to_host(), firwin_typed_cpu<T>(1, cutoff, fw_options)) &&
        same(width_out.to_host(), firwin_typed_cpu<T>(17, cutoff, width_options)) &&
        same(explicit_fw_out.to_host(),
            firwin_typed_cpu<T>(17, cutoff, explicit_fw_options)) &&
        same(none_fw_out.to_host(),
            firwin_typed_cpu<T>(17, cutoff, none_fw_options)) &&
        same(highpass_out.to_host(),
            firwin_typed_cpu<T>(17, cutoff, highpass_options)) &&
        same(boolean_false_out.to_host(),
            firwin_typed_cpu<T>(17, cutoff, boolean_false_options)) &&
        same(bandstop_out.to_host(),
            firwin_typed_cpu<T>(17, band_cutoff, bandstop_options)) &&
        same(unscaled_out.to_host(),
            firwin_typed_cpu<T>(17, cutoff, unscaled_options)) &&
        firwin_rejects_lowpass_multicutoff &&
        firwin_rejects_bandstop_single_cutoff &&
        firwin_rejects_explicit_window_length &&
        firwin_rejects_even_highpass;
```

所有 CPU/GPU 数值对照和异常门禁必须同时成立。

块内逐行说明：

- `const bool firwin_ok =`：开始用逻辑与聚合全部 firwin 判据。
- `firwin_device_rejects_nyquist && firwin_cpu_rejects_nyquist &&`：调用当前公开 GPU compatibility API。
- `same(dfw.to_host(), fw_ref) &&`：比较 GPU FP64 存储输出与 CPU reference。
- `same_float(resident_out.to_host(), resident_ref) &&`：以 resident FP32 容差比较实际值与参考值。
- `same(resident_fp64_storage_out.to_host(), fw_ref) &&`：比较 GPU FP64 存储输出与 CPU reference。
- `same(resident_stage_c_out.to_host(), fw_ref) &&`：比较 GPU FP64 存储输出与 CPU reference。
- `same_float(resident_explicit_out.to_host(), resident_explicit_ref_fp32) &&`：以 resident FP32 容差比较实际值与参考值。
- `same(d_band.to_host(), band_ref) &&`：比较 GPU FP64 存储输出与 CPU reference。
- `same(single_tap.to_host(), firwin_typed_cpu<T>(1, cutoff, fw_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(width_out.to_host(), firwin_typed_cpu<T>(17, cutoff, width_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(explicit_fw_out.to_host(),`：比较 GPU FP64 存储输出与 CPU reference。
- `firwin_typed_cpu<T>(17, cutoff, explicit_fw_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(none_fw_out.to_host(),`：比较 GPU FP64 存储输出与 CPU reference。
- `firwin_typed_cpu<T>(17, cutoff, none_fw_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(highpass_out.to_host(),`：比较 GPU FP64 存储输出与 CPU reference。
- `firwin_typed_cpu<T>(17, cutoff, highpass_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(boolean_false_out.to_host(),`：比较 GPU FP64 存储输出与 CPU reference。
- `firwin_typed_cpu<T>(17, cutoff, boolean_false_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(bandstop_out.to_host(),`：比较 GPU FP64 存储输出与 CPU reference。
- `firwin_typed_cpu<T>(17, band_cutoff, bandstop_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `same(unscaled_out.to_host(),`：比较 GPU FP64 存储输出与 CPU reference。
- `firwin_typed_cpu<T>(17, cutoff, unscaled_options)) &&`：调用 CPU reference，得到当前类型输入对应的 FP64 基准。
- `firwin_rejects_lowpass_multicutoff &&`：相邻测试表达式、声明或作用域的一部分。
- `firwin_rejects_bandstop_single_cutoff &&`：相邻测试表达式、声明或作用域的一部分。
- `firwin_rejects_explicit_window_length &&`：相邻测试表达式、声明或作用域的一部分。
- `firwin_rejects_even_highpass;`：相邻测试表达式、声明或作用域的一部分。

#### main 五类型扫过

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/test/signal_processing/e3_estimation_filter_design_type_smoke.cu:580`；符号：`main`；提交内第 580–595 行。

```cpp
int main()
{
    try {
        int passed = 0;
        passed += run_type<float>("fp32");
        passed += run_type<__half>("fp16");
        passed += run_type<std::int32_t>("int32");
        passed += run_type<std::int16_t>("int16");
        passed += run_type<std::int8_t>("int8");
        std::cout << "[E3][estimation_filter_design] total=15 pass=" << passed
                  << " fail=" << (15 - passed) << '\n';
        return passed == 15 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "[E3][estimation_filter_design][FAIL] " << ex.what() << '\n';
        return 1;
    }
```

main 对 FP32、FP16、INT32、INT16、INT8 调用 run_type；本文件同时含三个算子，所以总判据为 15。

块内逐行说明：

- `int main()`：相邻测试表达式、声明或作用域的一部分。
- `{`：开始或结束当前作用域/调用。
- `try {`：调用预期失败的路径；若未抛异常，拒绝标志保持 false。
- `int passed = 0;`：初始化或更新测试变量。
- `passed += run_type<float>("fp32");`：为一种业务输入类型执行完整测试。
- `passed += run_type<__half>("fp16");`：为一种业务输入类型执行完整测试。
- `passed += run_type<std::int32_t>("int32");`：为一种业务输入类型执行完整测试。
- `passed += run_type<std::int16_t>("int16");`：为一种业务输入类型执行完整测试。
- `passed += run_type<std::int8_t>("int8");`：为一种业务输入类型执行完整测试。
- `std::cout << "[E3][estimation_filter_design] total=15 pass=" << passed`：相邻测试表达式、声明或作用域的一部分。
- `<< " fail=" << (15 - passed) << '\n';`：初始化或更新测试变量。
- `return passed == 15 ? 0 : 1;`：返回比较结果或进程状态。
- `} catch (const std::exception& ex) {`：相邻测试表达式、声明或作用域的一部分。
- `std::cerr << "[E3][estimation_filter_design][FAIL] " << ex.what() << '\n';`：完成函数调用或对象构造。
- `return 1;`：返回比较结果或进程状态。
- `}`：开始或结束当前作用域/调用。

#### CMake 源码注册

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/CMakeLists.txt:142`；符号：`SIGNAL_LIBRARY_SOURCES`；提交内第 142–142 行。

```cpp
    test/signal_processing/e3_estimation_filter_design_type_smoke.cu
```

把 E3 测试源码列入测试源集合。

块内逐行说明：

- `test/signal_processing/e3_estimation_filter_design_type_smoke.cu`：相邻测试表达式、声明或作用域的一部分。

#### CMake 可执行目标

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/CMakeLists.txt:186`；符号：`e3_estimation_filter_design_type_smoke target`；提交内第 186–188 行。

```cpp
add_executable(e3_estimation_filter_design_type_smoke test/signal_processing/e3_estimation_filter_design_type_smoke.cu)
configure_zq500_target(e3_estimation_filter_design_type_smoke)
target_link_libraries(e3_estimation_filter_design_type_smoke PRIVATE signal_lib fft_core)
```

建立 ZQ500 目标并链接 signal_lib 与 fft_core。

块内逐行说明：

- `add_executable(e3_estimation_filter_design_type_smoke test/signal_processing/e3_estimation_filter_design_type_smoke.cu)`：创建测试可执行目标。
- `configure_zq500_target(e3_estimation_filter_design_type_smoke)`：应用项目统一 ZQ500 编译配置。
- `target_link_libraries(e3_estimation_filter_design_type_smoke PRIVATE signal_lib fft_core)`：链接算子库和 FFT 核心库。

#### CMake 测试清单

Git `fdd55ac8415d70379eb38a2c299047f90bcf0a41`；`ZKX_dev/cusignal_cpp/CMakeLists.txt:216`；符号：`test registration list`；提交内第 216–216 行。

```cpp
    e3_estimation_filter_design_type_smoke
```

把该可执行文件加入统一测试目标列表。

块内逐行说明：

- `e3_estimation_filter_design_type_smoke`：相邻测试表达式、声明或作用域的一部分。

### 7. CPU、GPU、内存与同步职责

| 层次 | 计算精度 | 存储/搬运 | 同步与错误 |
| --- | --- | --- | --- |
| CPU reference | double sinc、窗、归一化 | Host vector<double> | 同步普通 C++；invalid_argument |
| compatibility GPU | Host prepare 为 double，Device 主算为 float | cutoff D2H；bands/window float H2D；输出 double storage | 默认 stream；launcher 只检查 launch error，读取前由调用者同步 |
| workspace resident FP32 | Device float | setup 一次上传；稳定调用无 H2D/D2H | 默认 stream；调用者负责跨 stream/读取同步 |
| resident FP64 storage | Device float 算术 | 用 uint64 位构造等值 double 存储 | N≤1024 融合；更大 N 两 kernel |
| 显式窗低通特化 | Device float | cutoff/window/output 全 resident | 一个 256-thread block |

GPU 没有调用 FFT、卷积库或线性代数库；核心是自定义 sinc/window/reduction kernel。临时缓冲为 bands、window、normalization，以及融合 kernel 的 shared `unscaled[1024]` 与 `partial[256]`。

### 8. 数学—Python—CPU—GPU 三方映射

| 原理或公式 | cuSignal Python | C++ CPU | C++ GPU | 差异 |
| --- | --- | --- | --- | --- |
| $m=n-(N-1)/2$ | 基准 Python `fir_filter_design.py:373–374` | SHA `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，`filter_design_typed.cpp:283–286` | 同 SHA，`filter_design_kernels.cuh:88–91` | CPU double；GPU float |
| $\sum_q[r_q\operatorname{sinc}(r_qm)-l_q\operatorname{sinc}(l_qm)]$ | Python `:376–378`；kernel `:107–113` | `filter_design_typed.cpp:274–282` | `filter_design_kernels.cuh:36–48` | 数学相同；可去奇点阈值不同 |
| $h_w=h_dw$ | Python `:380` / GPU `:115` | `filter_design_typed.cpp:284–287` | `filter_design_kernels.cuh:84–92` | 相同 |
| 第一通带参考频率 | Python `:385–393` | `filter_design_typed.cpp:288–297` | `filter_design_kernels.cuh:103–125` | GPU 用补偿和 shared tree reduction |
| Kaiser $A\to\beta$ | Python `:303–308` | `filter_design_typed.cpp:77–89` | Host 准备后上传窗 | GPU compatibility 不在 device 计算 Kaiser |
| Type II Nyquist 门禁 | Python `:344–350` | `filter_design_typed.cpp:64–66` | Host 先拒绝，不进 kernel | 语义一致 |
| 对称窗 | `get_window(...,fftbins=False)` | Hamming/Kaiser/显式对称窗 | 上传后逐点读取 | C++ 仅支持 hamming、none、explicit；width 可生成 Kaiser |

### 9. 与 Python 实现相同、等价替换和有意不同之处

- 相同：cutoff 归一化、pass_zero 交替频带、Type I/II 约束、sinc 差、乘窗、第一通带缩放。
- 等价替换：Python GPU 以 `ElementwiseKernel` 同时产生 `h/hc`；C++ 把归一化变成补偿归约 kernel，并提供融合或两 kernel 路径。
- 有意不同：C++ typed API 支持五种 cutoff 输入类型；Device 主算固定 FP32，正式输出只是“数值等值的 FP64 存储”，不提供 Python kernel 的真实 FP64 算术。
- 有意不同：Python `window` 接受通用窗口名；C++ 公开选项只直接支持 Hamming、none、显式数组，`width` 路径另生成 Kaiser。
- 有意不同：公开 `firwin_device` 会 D2H cutoff 做 Host 校验；resident API 把 setup 移到 workspace 构造，不能把两者耗时混为同一 compatibility scope。
- 工程优化：N≤1024 由 shared-memory 融合 kernel 一次完成；N>1024 使用通用两 kernel，数学原理不变。

### 10. 测试证据与未覆盖风险

当前提交的直接可构建证据是 CMake 注册的 `e3_estimation_filter_design_type_smoke`。源码覆盖五种输入类型、低通/高通/带通/带阻、width、三种窗口模式、scale、single tap、resident/compatibility、异常和 CPU/GPU 对照。本次没有重跑，不能把“源码存在”写成“本次实测通过”。

仍需保留的风险：

- GPU FP32 与 CPU double 的误差会随 N、Q、窗和截止位置变化；E3 容差不能替代全参数误差证明。
- 融合路径固定 shared `unscaled[1024]`，上限由代码门禁保证；极大 N 只覆盖两 kernel 回退。
- normalization 没有显式零分母门禁；异常频带/窗组合可能产生非有限输出。
- compatibility 入口的 D2H/H2D 与 Host prepare 是公开成本；resident benchmark 不能冒充端到端 compatibility 性能。
- 默认 stream 语义要求调用者在读取和跨 stream 复用前同步。
- 仓库中的旧 D4/D6 示例引用当前不存在的旧兼容头，不能作为本 SHA 的当前构建证据。

### 11. 阶段三自检问题与参考答案

1. **V1 绑定哪个版本，相关源码是否 dirty？**  
   答：完整 SHA 为 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`，进入阶段三时相关路径无 dirty 输出；所有定位都用 `git show` 读取该提交。

2. **CPU 与 GPU 的核心公式是否相同？**  
   答：相同。CPU 在 `filter_design_typed.cpp:274–286`，GPU 在 `filter_design_kernels.cuh:36–48,84–92` 实现同一 sinc 差和乘窗；差别是 CPU double、GPU float。

3. **公开 GPU compatibility 为什么不是纯 device-resident？**  
   答：`firwin_device` 在 `filter_design_typed.cpp:310–316` 把 cutoff D2H，Host prepare 后再把 bands/window H2D。纯稳定执行要用 `FirwinDeviceWorkspace` 或低通显式窗 resident API。

4. **N≤1024 与 N>1024 如何路由？**  
   答：`filter_design_typed.cpp:321–327`：前者走融合 FP64-storage kernel，后者先归一化再输出；1024 来自 `filter_design_kernels.cuh:161,171` 的 shared 数组上限。

5. **一个 GPU 线程负责什么？**  
   答：普通输出 kernel 的全局线程号对应一个 tap（`:137–141`）；融合 kernel 固定 256 lanes，每 lane 以 256 为步长负责多个 taps（`:184–199`）。

6. **为什么归一化使用补偿累加？**  
   答：通带余弦加权和可能包含不同量级和正负项；`CompensatedFloatAccumulator` 在 `operator_compute_policy.h:90–116` 用 Neumaier correction 减少 FP32 归约舍入。

7. **FP64 输出是否意味着 GPU 做了 FP64 算术？**  
   答：不是。GPU 先得到 FP32 数值，再由 `fp32_to_fp64_storage_bits` 构造完全等值的 double 位模式；头文件 `:137–148` 和 kernel `:50–82` 都明确这一点。

8. **模板与显式实例化覆盖哪些类型？**  
   答：float、`__half`、int32、int16、int8；Host/GPU 通用入口见 `filter_design_typed.cpp:402–413`，低通 resident 特化见 `filter_design_typed.cu:134–145`。

9. **同步责任在哪里？**  
   答：`launch_1d_kernel` 传默认 stream 并只做 launch error check（`kernel_launch.h:39–55`）；调用者在读取输出或跨 stream 复用 workspace 前负责同步。

10. **当前直接测试覆盖什么，什么没有证明？**  
    答：E3 覆盖五类型、主要模式、异常和 CPU/GPU 对照（测试 `:300–436,580–595`）。它没有证明所有 N/Q/cutoff/window 的误差上界，也没有在本次学习任务中重新执行。

### 12. 版本差异与原理不变量

当前只有 V1，尚无 V2 可比较。V1 建立后，未来优化必须追加版本而不能覆盖本节。

| 比较维度 | V1 当前实现 | 原理不变量 |
| --- | --- | --- |
| 算法结构 | Host prepare + CPU reference + compatibility/resident GPU | 理想矩形频带的 sinc 逆 DTFT |
| 并行方式 | 一 tap 一线程或 256 lanes 跨步；归约可融合 | 多带线性叠加、对称中心不变 |
| 精度 | CPU double；GPU float；double 仅存储拓宽 | 窗乘与参考频率单位增益目标不变 |
| 内存 | Host bands/window；Device workspace；shared 归约 | 输出长度始终为 numtaps |
| 优化 | 1024 taps 内融合；大 N 两 kernel | Type II Nyquist 约束不变 |
| 测试 | E3 五类型 CPU/GPU 对照 | 相同参数语义和异常边界 |

### 13. 阶段三代码阅读终点

本阶段按“公开入口 → 选项/workspace → Host prepare → CPU reference → GPU compatibility/resident wrapper → launcher → device helper/kernel → 类型实例化 → E3 测试/main”的顺序，已经解释到当前 `firwin` 相关测试调用链的最后一行 `e3_estimation_filter_design_type_smoke.cu:595`。没有越界进入 `firwin2` 核心实现。
