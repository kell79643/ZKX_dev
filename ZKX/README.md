# ZKX 科学计算算子与雷达信号处理工程

本工程面向中科芯 ZQ500 平台，实现 cuSignal 相关科学计算算子的 C++/GPU 适配，并完成
两个雷达信号处理任务的算子调用、数值验证、性能测试和结果可视化。工程由统一算子库、
固定版本参考源码、Task1、Task2 及综合测试入口组成。

## 工程结构

最终提交以正式源码、构建文件、运行入口和必要测试为范围。固定版本 cuSignal 与 CuPy
参考源码整体保留，但不展开其内部结构；具有相同组织方式的算子类别和任务步骤采用集合写法表示。
下列每个目录或文件条目后均给出用途；`{...}`、`step{1..N}` 和 `*.{h,cpp}` 等集合写法的说明适用于其展开后的每个目录或文件。

### 总体结构

```text
ZKX/                         工程根目录，集中管理构建、算子、任务、演示与测试代码
├── README.md                 工程结构、构建入口与功能说明
├── CMakeLists.txt            ZQ500 工程统一构建入口
├── CMakePresets.json         Thrust 与 DLFFT 后端构建预设
├── cmake/                    ZQ500 公共构建配置与 CMake 校验脚本
├── cusignal_cpp/             正式 C++/GPU 统一算子库
├── cusignal-23.08.00/        固定版本 cuSignal 参考源码，内部结构保持不变
├── cupy-13.6.0/              固定版本 CuPy 参考源码，内部结构保持不变
├── demo/                     面向评委的统一构建与交互演示体系
├── scripts/                  Task1/Task2 共用的稳定性运行入口
├── test_all/                 53 个算子及任务异常测试体系
├── Task1/                    雷达目标检测与多普勒分析任务
└── Task2/                    多域特征提取与定位任务
```

### `cmake/`

```text
cmake/                          ZQ500 公共构建与静态校验目录
├── VerifyProjectCMake.cmake  检查工程 CMake 目标、源文件和构建约束
└── ZQ500Build.cmake         集中定义 ZQ500 编译器、架构、头文件与链接规则
```

### `cusignal_cpp/`

```text
cusignal_cpp/                         正式 C++/GPU 算子库源码与构建目录
├── CMakeLists.txt                    组装算子、FFT/随机后端和运行时目标
├── cmake/                           算子库专用的合同与校验脚本
│   ├── contracts/                   算子源文件、类型和接口合同目录
│   │   ├── operator_contracts.cmake        声明算子实现与类型覆盖合同
│   │   └── verify_operator_contracts.cmake 在配置阶段验证算子合同完整性
│   └── validation/                  公开 API 文档与类型覆盖输出校验目录
│       ├── verify_public_api_docs.cmake       检查公开头文件的 API 文档约束
│       └── verify_type_coverage_output.cmake  检查算子数据类型覆盖报告
├── include/                         供 Task、测试和外部调用方使用的公开头文件
│   └── cusignal/                     cuSignal C++ 公开头文件根目录
│       ├── signal_processing.h          汇总导出常用信号处理算子接口
│       ├── operators/                   按功能分类的公开算子头文件
│       │   └── {bsplines,convolution,demod,estimation,
│       │       filter_design,filtering,peak_finding,radartools,
│       │       spectral_analysis,waveforms,wavelets,windows}/  十二类业务算子的公开接口目录
│       │       └── *_typed.h                 各算子的类型 C/C++ 调用声明
│       ├── backends/                    可替换计算后端的公开接口
│       │   ├── fft/                     FFT 后端统一接口目录
│       │   │   ├── fft_interface.h         GPU FFT 统一 C ABI 与计划接口
│       │   │   └── fft_interface_cpu.h     CPU FFT 参考接口
│       │   └── random/rand_interface.h   平台随机数后端公开接口
│       └── runtime/                     设备内存、类型派发与错误处理公共头文件
│           ├── errors.h                       统一运行时错误码和错误表示
│           ├── operator_abnormal_validation.h 算子异常输入与返回状态校验工具
│           ├── copy_utils.h                   主机与设备数据拷贝辅助函数
│           ├── cuda_error.h                   CUDA/ZQ500 调用错误检查宏
│           ├── cuda_utils.h                   设备调用与同步公共工具
│           ├── device_allocation_tracker.h   追踪设备分配量与资源释放
│           ├── device_array.h                封装设备数组所有权与寿命周期
│           ├── device_complex.h              定义设备端复数类型与运算辅助
│           ├── host_output_finalize.h        完成 GPU 输出回传与主机端收尾
│           ├── kernel_launch.h               统一 kernel 启动参数和错误检查
│           ├── operator_type_contract.h      定义算子支持的输入输出类型合同
│           ├── operator_type_dispatch.h      声明运行时数据类型派发逻辑
│           ├── resource_stability.h          采集并比较 CPU/GPU 资源稳定性指标
│           └── runtime_utils.h                其他通用运行时辅助函数
└── src/                             算子、后端与运行时的正式实现
    ├── operators/                   按功能分类的算子实现目录
    │   └── {bsplines,convolution,demod,estimation,
    │       filter_design,filtering,peak_finding,radartools,
    │       spectral_analysis,waveforms,wavelets,windows}/  十二类业务算子的实现目录
    │       ├── *_typed.cpp              各算子 CPU 参考或主机端类型实现
    │       ├── *_typed.cu               各算子 GPU 类型实现与调度入口
    │       ├── *_kernels.cuh            各算子的设备 kernel 声明与实现
    │       └── 模块私有头文件           仅在对应算子模块内共享的辅助声明
    ├── backends/                    FFT 与随机数后端实现目录
    │   ├── fft/                     统一 FFT 接口的多后端实现
    │   │   ├── fft_interface.cpp             GPU FFT 公开接口调度与生命周期管理
    │   │   ├── fft_interface_cpu.cpp         CPU FFT 参考实现入口
    │   │   ├── fft_cufft.cu                 ZQ500 dlFFT 后端实现
    │   │   ├── fft_dlfft.cu                 链接到fft_cufft.cu
    │   │   ├── fft_thrust.cu                Thrust 自研 FFT 后端实现
    │   │   └── fft_thrust_cufft_compat.cu   Thrust 后端的 dlFFT 兼容接口层
    │   └── random/rand_dlrand.cu       ZQ500 DLRAND 随机数后端实现
    └── runtime/                     数据类型、精度收尾与公共运行时实现
        ├── fp64_storage_finalize.{h,cu}  FP64 存储与输出收尾声明/实现
        ├── operator_type_dispatch.cu      算子运行时数据类型派发实现
        ├── operator_compute_policy.h     按类型选择计算精度与中间表示
        └── simple_signal_typed.h         简单信号算子的强类型通用模板
```

`include/cusignal/` 只保存 Task、测试或外部调用方需要包含的公开头文件；`src/` 保存实现、
kernel 和内部辅助头文件。算子合同数据与对应验证脚本统一归入 `cmake/contracts/`。

### `demo/`

```text
demo/                                      面向评委的统一构建、配置与交互演示体系
├── bin/{demo.sh,build_demo.sh}             启动演示和构建演示目标的 shell 入口
├── app/                                    交互界面与统一演示调度应用
│   ├── interactive_menu.py                    显示并处理终端交互菜单
│   ├── unified_demo_runner.py                 按配置统一调度算子、Task 和异常演示
│   ├── demo_capabilities.py                   定义可演示能力、目标与环境约束
│   ├── task_config.py                         解析并校验 Task 演示参数
│   └── terminal_tables.py                     统一格式化终端表格输出
├── execution/                              将菜单请求适配为具体可执行命令
│   ├── __init__.py                            声明 execution Python 包
│   ├── common.py                              提供子进程、路径和结果处理公共函数
│   ├── operator_adapter.py                    适配单算子正常演示入口
│   ├── task_adapter.py                        适配 Task1/Task2 流水线演示入口
│   ├── abnormal_adapter.py                    适配异常输入与合同演示入口
│   └── fft_library_swap_adapter.py            适配 FFT 后端切换对比演示
├── build/{build_demo.py,verify_demo_build.py} 构建演示目标并核验产物的 Python 脚本
├── config/                                  演示构建、运行与输入示例配置
│   ├── build/                                  构建计划与依赖审计配置
│   │   ├── demo_build_plan.json                    声明演示构建目标及顺序
│   │   ├── cmake_dependency_audit.json            记录演示目标的 CMake 依赖审计结果
│   │   └── task2_memory_compatibility.json         记录 Task2 内存演示兼容性约束
│   ├── runtime/                                演示能力、命令与用例的运行配置
│   │   ├── demo_profile.json                       定义当前演示配置组合
│   │   ├── demo_runtime.json                       定义运行时路径、超时与环境参数
│   │   ├── operator_demo_commands.json            映射算子演示项与执行命令
│   │   └── task_demo_cases.json                   声明 Task1/Task2 演示用例及参数
│   └── examples/                               用户自定义演示输入的 JSON 示例
│       ├── operator_demo_input_example.json         单算子演示输入格式示例
│       └── task_demo_input_example.json             Task 流水线演示输入格式示例
├── include/demo/config_file.h               为 C++ 演示目标提供配置文件读取接口
├── selection/                               从正式结果中选择最佳演示配置
│   ├── select_operator_best.ps1                选择算子最佳用例并生成配置
│   ├── select_task1_best.py                    选择 Task1 最佳演示 profile
│   └── select_task2_best.py                    选择 Task2 最佳演示 profile
├── validation/verify_abnormal_contract.py   校验演示异常用例与源码合同一致性
└── tools/generate_demo_metadata.ps1         从正式目标和结果生成演示元数据
```

### `scripts/`

```text
scripts/                              Task1/Task2 共用自动化脚本目录
└── stability/                      任务稳定性参数生成与运行入口
    ├── generate_stability_params.py  生成可复现的稳定性批次参数
    └── run_stability_tests.sh        按配置执行 Task1/Task2 稳定性测试
```

### `Task1/`

```text
Task1/                                                   任务一的流水线、评价、稳定性与可视化目录
├── CMakeLists.txt                                        定义 Task1 五步流水线及各类测试目标
├── cmake/verify_source_contract.cmake                    校验 Task1 源文件与构建合同
├── pipeline/                                             Task1 正式 C++/GPU 处理流水线
│   ├── include/task1/                                       Task1 对内对外头文件
│   │   ├── task1_common.h                                  声明公共数据结构与辅助函数
│   │   ├── task1_config.h                                  定义 Task1 输入规模、参数与配置结构
│   │   ├── task1_entry.h                                   声明 Task1 流水线对外调用入口
│   │   ├── task1_runner.h                                  声明流水线执行器与结果结构
│   │   ├── task1_abnormal_validation.h                     声明 Task1 异常输入校验逻辑
│   │   └── steps/step{1..5}.h                              声明五个独立处理步骤接口
│   ├── src/                                                 流水线公共逻辑与各步骤 GPU 实现
│   │   ├── task1_common.cu                                  实现数据初始化与公共 GPU 辅助逻辑
│   │   ├── task1_runner.cu                                  组装并执行五步 Task1 流水线
│   │   └── steps/step{1..5}/stepN.cu                       分别实现五个处理步骤
│   └── apps/                                                整体流水线与单步可执行入口
│       ├── pipeline/main.cu                                   运行完整 Task1 流水线
│       └── steps/step{1..5}/main.cu                          分别运行五个单步目标
├── accuracy/                                             Task1 CPU/GPU 数值精度比较与验证
│   ├── accuracy_types.h                                      定义精度指标、阈值与结果类型
│   ├── accuracy_utils.{h,cpp}                                提供误差统计与报告公共工具的声明/实现
│   ├── formal_accuracy.{h,cpp}                               实现正式精度口径与通过判定
│   ├── comparison/step{1..5}_comparison.{h,cpp}              分别比较五步 CPU/GPU 输出
│   └── validation/step{1..5}_validation.{h,cu}               分别生成并验证五步测试结果
├── performance/                                          Task1 分步与整体流水线性能评价
│   ├── cpp/                                                 C++/GPU 性能测量入口与规模配置
│   │   ├── pipeline_benchmark_main.cu                       测量完整流水线 CPU/GPU 配对耗时
│   │   ├── step{1..5}_benchmark_main.cu                    分别测量五个处理步骤耗时
│   │   └── task1_scale_config.{h,cpp}                      声明并解析 Task1 多尺度参数
│   ├── python/benchmark_main.py                             运行 Python/cuSignal 对照性能基准
│   └── scripts/                                             正式性能矩阵运行与核验脚本
│       ├── run_task1_formal_matrix.sh                        执行 Task1 正式多尺度用例矩阵
│       └── verify_task1_pipeline_formal_matrix.py            校验矩阵结果完整性与通过状态
├── stability/                                            Task1 内存与多尺度运行稳定性验证
│   ├── memory/memory_stability_main.cu                      重复执行并记录 CPU/GPU 资源变化
│   ├── scaled_pipeline/                                     使用扰动的多尺度稳定性流水线
│   │   ├── CMakeLists.txt                                  定义稳定性流水线构建目标
│   │   ├── app/stability_main.cu                          运行多尺度稳定性用例并写出证据
│   │   ├── include/                                       稳定性流水线专用头文件
│   │   └── src/                                           稳定性流水线的实现
│   │       ├── task1_common.cu                              实现稳定性用公共数据处理
│   │       ├── steps/step{1..5}.cu                         实现五步处理逻辑
│   │       └── accuracy/validation/                        存放稳定性用精度验证实现
│   └── python/{memory_metrics.py,memory_stability_main.py}   采集内存指标并调度 Python 稳定性测试
├── python_reference/                                     Task1 的 Python/cuSignal 数值对照实现
│   ├── main.py                                              运行完整 Python 参考流水线
│   ├── task1_common.py                                      提供参考流水线公共数据函数
│   ├── task1_runner.py                                      组装五步 Python 参考流水线
│   ├── steps/step{1..5}/{__init__.py,main.py,stepN.py}       定义包、单步入口与五步参考算法
│   └── utils/                                               参考实现共用的 I/O 与数值工具
├── visualization/                                      Task1 正式数据导出、本地评价与绘图
│   ├── cpp/pipeline_visualization_export.{h,cpp}            声明/实现 C++ 流水线 CSV 导出
│   ├── python/{main.py,visualization_common.py}              提供导出/绘图入口与公共可视化函数
│   ├── scripts/{plot_local.ps1,pull_data.ps1}                在 Windows 绘图并从服务器回收数据
│   └── requirements.txt                                     声明本地绘图所需 Python 依赖
└── evidence/                                           Task1 可审阅运行证据目录
    └── visualization/                                     可视化数据、评价、图表和运行摘要
        ├── data/                                              保存正式 pipeline 导出的 CSV 数据
        ├── evaluation/                                        保存数值评价报告
        ├── figures/                                           保存各步骤与总结图表
        ├── pipeline.log                                       记录正式导出流水线日志
        ├── run_summary.json                                   记录本次运行参数、版本与状态摘要
        └── task1_selected.conf                                记录选中的 Task1 演示配置
```

### `Task2/`

```text
Task2/                                                   任务二的流水线、评价、稳定性与可视化目录
├── CMakeLists.txt                                        定义 Task2 六步流水线及各类测试目标
├── pipeline/                                             Task2 正式 C++/GPU 处理流水线
│   ├── include/task2/                                       Task2 对内对外头文件
│   │   ├── task2_common.h                                  声明公共数据结构与辅助函数
│   │   ├── task2_config.h                                  定义 Task2 输入规模、参数与配置结构
│   │   ├── task2_entry.h                                   声明 Task2 流水线对外调用入口
│   │   ├── task2_runner.h                                  声明流水线执行器与结果结构
│   │   ├── task2_abnormal_validation.h                     声明 Task2 异常输入校验逻辑
│   │   └── steps/step{1..6}.h                              声明六个独立处理步骤接口
│   ├── src/                                                 流水线公共逻辑与各步骤 GPU 实现
│   │   ├── task2_common.cu                                  实现数据初始化与公共 GPU 辅助逻辑
│   │   ├── task2_runner.cu                                  组装并执行六步 Task2 流水线
│   │   └── steps/step{1..6}/stepN.cu                       分别实现六个处理步骤
│   └── apps/                                                整体流水线与单步可执行入口
│       ├── pipeline/main.cu                                   运行完整 Task2 流水线
│       └── steps/step{1..6}/main.cu                          分别运行六个单步目标
├── accuracy/                                             Task2 CPU/GPU 数值精度比较与验证
│   ├── accuracy_types.h                                      定义精度指标、阈值与结果类型
│   ├── accuracy_utils.{h,cpp}                                提供误差统计与报告工具的声明/实现
│   ├── formal_accuracy.{h,cpp}                               实现正式精度口径与通过判定
│   ├── comparison/step{1..6}_comparison.{h,cpp}              分别比较六步 CPU/GPU 输出
│   └── validation/{step{1..6}_validation.*,step5_cpu_validation.cpp}  验证六步结果并补充第五步 CPU 验证
├── performance/                                          Task2 分步与整体流水线性能评价
│   ├── cpp/                                                 C++/GPU 性能测量入口与规模配置
│   │   ├── pipeline_benchmark_main.cu                       测量完整流水线 CPU/GPU 配对耗时
│   │   ├── step{1..6}_benchmark_main.cu                    分别测量六个处理步骤耗时
│   │   └── task2_scale_config.{h,cpp}                      声明并解析 Task2 多尺度参数
│   ├── python/                                              Python/cuSignal 性能数据生成与对比
│   │   ├── benchmark_task2.py                              运行 Task2 Python 参考性能基准
│   │   ├── aggregate_task2_benchmarks.py                   汇总多批次 Task2 基准数据
│   │   └── compare_cpp_python_performance.py               比较 C++/GPU 与 Python/cuSignal 性能
│   └── scripts/{run_task2_formal_matrix.sh,verify_task2_formal_matrix.py}  运行并校验 Task2 正式性能矩阵
├── stability/                                            Task2 内存与多尺度运行稳定性验证
│   ├── memory/memory_stability_main.cu                      重复执行并记录 CPU/GPU 资源变化
│   ├── scaled_pipeline/                                     使用扰动的多尺度稳定性流水线
│   │   ├── CMakeLists.txt                                  定义稳定性流水线构建目标
│   │   ├── app/stability_main.cu                          运行多尺度稳定性用例并写出证据
│   │   ├── include/                                       稳定性流水线专用头文件
│   │   └── src/                                           稳定性流水线的实现
│   │       ├── task2_common.cu                              实现稳定性用公共数据处理
│   │       ├── steps/step{1..6}.cu                         实现六步处理逻辑
│   │       └── accuracy/validation/                        存放稳定性用精度验证实现
│   └── python/{memory_metrics.py,memory_stability_main.py}   采集内存指标并调度 Python 稳定性测试
├── python_reference/                                     Task2 的 Python/cuSignal 数值对照实现
│   ├── main.py                                              运行完整 Python 参考流水线
│   ├── task2_common.py                                      提供参考流水线公共数据函数
│   ├── task2_runner.py                                      组装六步 Python 参考流水线
│   ├── steps/step{1..6}/{__init__.py,main.py,stepN.py}       定义包、单步入口与六步参考算法
│   └── utils/                                               参考实现共用的 I/O 与数值工具
├── visualization/                                      Task2 正式数据导出、本地评价与绘图
│   ├── cpp/pipeline_visualization_export.{h,cpp}            声明/实现 C++ 流水线 CSV 导出
│   ├── python/{main.py,visualization_common.py}              提供导出/绘图入口与公共可视化函数
│   ├── scripts/{plot_local.ps1,pull_data.ps1}                在 Windows 绘图并从服务器回收数据
│   └── requirements.txt                                     声明本地绘图所需 Python 依赖
└── evidence/                                           Task2 可审阅运行证据目录
    └── visualization/                                     可视化数据、评价、图表和运行摘要
        ├── data/                                              保存正式 pipeline 导出的 CSV 数据
        ├── evaluation/                                        保存数值评价报告
        ├── figures/                                           保存各步骤与总结图表
        ├── pipeline.log                                       记录正式导出流水线日志
        ├── run_summary.json                                   记录本次运行参数、版本与状态摘要
        └── task2_selected.conf                                记录选中的 Task2 演示配置
```

Task 根目录直接展示 `accuracy/`、`performance/` 和 `stability/`，使精度、耗时与稳定性
测试入口可以被直接识别。`python_reference/` 只保存 cuSignal Python 对照 pipeline；其
性能和稳定性入口分别归入 `performance/python/` 与 `stability/python/`。

### `test_all/`

```text
test_all/                                                        53 个算子与任务异常、FFT 后端测试体系
├── support/                                                 测试配置解析、指标和稳定性公共支撑
│   ├── config/json_value.{h,cpp}                           声明/实现轻量 JSON 值与配置解析
│   ├── metrics/                                           精度、资源、统计与输出指标库
│   │   ├── accuracy.{h,cpp}                               计算数值误差与精度通过状态
│   │   ├── csv_writer.{h,cpp}                             按固定字段写出可审阅 CSV 结果
│   │   ├── identity.{h,cpp}                               记录用例、后端与代码版本身份
│   │   ├── memory.{h,cpp}                                 采集 CPU/GPU 内存与显存指标
│   │   ├── statistics.{h,cpp}                             计算耗时样本的统计摘要
│   │   └── terminal_table.{h,cpp}                         格式化终端测试结果表
│   └── stability/task_stability_evidence.h                  定义 Task 稳定性证据记录格式
├── config/                                                  算子、任务基准与稳定性用例配置
│   ├── task_benchmarks/                                   Task1/Task2 性能基准参数集
│   ├── stability/formal_1000/                            1000 轮正式稳定性用例配置
│   └── operators/                                         算子独立用例与组合 suite 配置
│       ├── cases/<模块>/<算子>/                            按模块和算子隔离的输入、期望值与阈值配置
│       └── suites/{radartools,waveforms,wavelets,task2_filter_chain,task2_features,task2_final}/  跨算子组合用例配置
├── operators/                                               算子独立、共享、组合、异常与 FFT 测试实现
│   ├── cases/<模块>/<算子>/                            53 个业务算子的独立构建与运行用例
│   ├── shared/{bsplines,filtering,radartools,waveforms,wavelets,windows}/  同类算子共享的夹具与辅助实现
│   ├── suites/{task2_features,task2_final}/                 Task2 特征与最终结果的跨算子组合测试
│   ├── abnormal/                                         算子参数、形状、类型与资源异常测试
│   └── fft_backend/                                      FFT 后端性能、稳定性与切换演示测试
│       ├── CMakeLists.txt                                  定义 FFT 后端各专项测试目标
│       ├── analysis/analyze_fft_backend_memory.py          分析 FFT 后端内存采样与泄漏分类
│       ├── scripts/run_fft_backend_memory_campaign.sh      执行 FFT 后端内存对照采样批次
│       └── src/                                            FFT 后端专项测试源码
│           ├── fft_backend_memory_unified.cu                    统一采集 FFT plan/execute 生命周期资源
│           ├── fft_backend_performance.cu                       比较 Thrust 与 DLFFT 后端性能
│           ├── fft_backend_stability.cu                         验证 FFT 后端重复执行稳定性
│           └── fft_library_swap_demo.cu                        演示同一接口下切换 FFT 库
└── tasks/                                                   Task1/Task2 任务级测试目录
    └── abnormal/                                              任务级异常合同用例与验证
        ├── task1/{CMakeLists.txt,fixtures/,task1_abnormal_case_catalog.csv,  Task1 异常目标、夹具与用例目录
        │          task1_abnormal_runner.cpp,verify_catalog.py,verify_results.py}  Task1 异常执行器与目录/结果校验脚本
        └── task2/    与 Task1 对称的 Task2 异常目标、夹具、执行器和校验脚本
```

算子测试以 `cases/<模块>/<算子>/` 为独立用例，公共夹具统一归入 `shared/`，跨算子组合
测试归入 `suites/`。

## 统一构建

进入项目运行环境后，可通过根目录的 `CMakeLists.txt` 完成统一配置、编译和测试。以下命令
均在工程根目录对应的运行环境中执行。

```sh
source /zq500/sdk/env.sh                         # 初始化编译器、头文件和平台库等 SDK 环境变量
cd /root/ZKX                                    # 进入容器内的工程根目录
cmake -E remove_directory build                 # 删除旧构建目录
cmake -S . -B build -DUSE_DLFFT=OFF -DUSE_THRUST=ON  # 选择项目 FFT 后端
cmake --build build --target test_all -j2       # 执行静态门禁并编译统一聚合目标
bash demo/bin/build_demo.sh --backend fft_thrust --jobs 2  # 发布 Thrust 后端演示程序
bash demo/bin/demo.sh                           # 在 /root/ZKX 启动统一交互演示
```

`test_all` 是构建聚合目标，本身不是可执行文件。因此完成统一构建后，需用
`build_demo.sh` 将演示程序发布到 `build/demo/`，再用 `demo.sh` 进入统一交互菜单。

如需使用平台 DLFFT 后端，配置为 `-DUSE_DLFFT=ON -DUSE_THRUST=OFF`。该模式的
Plan、Exec、Destroy 均直接调用供应商 `/zq500/sdk/lib/libdlfft.so`。
注：两个开关必须同时给出且恰好一个为 ON；切换后端时使用新的构建目录，避免复用旧 CMake cache。

Thrust 模式下 `fft_thrust.cu` 只由统一目标编译一次，产物为`libzkx_fft_thrust.so`。
算子、Task1、Task2 和测试均链接该共享库，不再把实现源文件加入各自目标；

CMake 在配置阶段完成公共编译规则加载、工程目标组织和各子目录装配；在构建阶段按照目标依赖
关系编译算子库与任务程序。`test_all` 是构建聚合入口，不执行程序，也不输出运行通过结论；
需要运行的算子与任务验证由各自目录中的正式脚本显式发起。其依赖关系如下。

| CMake 目标    | 功能                                         |
| ------------- | -------------------------------------------- |
| `task1_all` | 聚合 Task1 五步流水线、评价与稳定性构建目标  |
| `task2_all` | 聚合 Task2 六步流水线、评价与稳定性构建目标  |
| `test_all`  | 汇总静态门禁与两个任务的构建目标，不运行程序 |
