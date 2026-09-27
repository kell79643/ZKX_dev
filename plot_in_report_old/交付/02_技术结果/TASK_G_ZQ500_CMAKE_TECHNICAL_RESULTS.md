# TASK_G：ZQ500 CMake 构建闭环技术结果

## 1. 实现效果

TASK_G 没有重写 Task1、Task2 或算子算法源码，而是把已有正式实现收敛为一套可选择、
可复现、可统一验收的 ZQ500 构建运行体系：

1. 仓库根目录新增统一 CMake 入口，默认装配算子、Task1 和 Task2 的正式实现；
2. `cmake/ZQ500Build.cmake` 统一 `dlcc`、CXX 语言模式、`.cu` 的 `LANGUAGE CXX`、
   `-x cuda`、DLI_V2 架构参数、SDK include/lib 以及 `curt`/`dlfft` 链接规则；
3. 53 个算子均有可独立选择的 `g_operator_<name>` 构建运行目标，并按 12 个类别提供
   聚合目标；每个算子目标复用 F 的正式 correctness、accuracy、performance 实现；
4. Task1 暴露 5 个 step 和 1 个任务级入口，Task2 暴露 6 个 step 和 1 个任务级入口；
5. 根目标 `test_all` 统一复验 53 个算子、全部类别、Task1 和 Task2，避免测试与任务
   另建实现双轨；
6. 活动 CMake 门禁在配置期和 `test_all` 运行期执行，防止重新引入 NVIDIA CMake 规则；
7. 算子测试 CMake 的 88 处旧 double 源引用已切换到当前 typed 正式源码，并新增旧路径
   禁用规则，避免远端残留文件掩盖源码清单错误。

因此，“项目原本已有 CMake”与 G 的区别在于：原有文件只能说明存在构建脚本，G 交付的是
全目标可选择、同实现运行、ZQ500 规则一致、失败可阻断、证据可绑定的完整构建闭环。

## 2. ZQ500 平台适配结果

| 项目 | 最终结果 |
|---|---|
| 远端环境 | `usr_02@10.110.12.10`，容器 `gpu_02` |
| SDK | `/zq500/sdk`，`DLI_V2=ON` |
| CMake | 3.22.1，项目最低版本仍保持 3.16 |
| 编译器 | `/zq500/sdk/bin/dlcc`，Clang 15.0.6 |
| 项目语言 | `LANGUAGES CXX` |
| GPU 源码 | `.cu` 标记为 CXX，以 `-x cuda` 编译 |
| 架构参数 | `--offload-arch=dlgput64,dlgpux64` |
| 平台库 | `curt`，按真实依赖链接 `dlfft`/`dlrand` |
| 官方样例映射 | CUDA 基线参考 `cuda/vectorAdd`，FFT 参考 `dlfft`，RAND 参考 `dlrand` |
| 活动 CMake | 13 个，违规 0 |

最终 `compile_commands.json` 含 298 条编译记录和 132 个唯一编译源：298/298 使用
`dlcc`，293 条 GPU 编译记录全部包含 `-x cuda` 与 DLI_V2 架构参数；`nvcc`、
`sm_XX`、`compute_XX` 均为 0。

## 3. 最终构建运行结果

最终结果绑定提交 `db7763c5b7b5e942dcea43d3d31be017a283cb0d`。源码由该提交直接
生成 clean Git archive，并复制到全新远端/容器目录；22 个已归档旧算子源文件物理不存在。
合同门禁所需的两个固定上游参考树单独作为只读依赖复制，不参与项目源码编译。

| 门禁 | 结果 |
|---|---|
| 独立算子目标 | 53/53 build+run 通过 |
| 类别目标 | 12/12 build+run 通过 |
| 算子检查范围 | correctness + accuracy + performance |
| Task1 step | 5/5 通过 |
| Task1 整体 | 1/1 通过；CTest 6/6 |
| Task2 step | 6/6 通过 |
| Task2 整体 | 1/1 通过；CTest 7/7 |
| 统一 `test_all` | `build_run=passed` |
| 缺失/失败 | `missing=0`，`failed=0` |

原始证据位于
`semi_final_tasks/交付/04_原始证据/task_g/task_g_evidence_db7763c/`，其中完整构建运行日志约 1.9 MB，
同时保存 CMake cache、实际编译命令、132 项编译源闭包、目标清单、环境、官方样例路径
及 154 个相关源码/头文件的 SHA-256。旧 `task_g_evidence_0f8c4a4/` 已明确标为
`invalidated`，不计入最终结论。

## 4. 技术文档可用结论

TASK_G 对复赛技术文档有帮助，但应只作为“中科芯 ZQ500 平台适配与可复现构建体系”的
简短章节，不应包装成算法创新或性能提升。建议正文使用以下结论：

> 项目采用统一 ZQ500 CMake 构建体系，以 dlcc 的 CXX 模式编译 CUDA 源文件，并统一
> DLI_V2 架构参数和平台科学库链接。53 个算子、Task1 五步链路和 Task2 六步链路均复用
> 正式实现，通过统一 test_all 完成构建与运行闭环，最终缺失和失败项均为 0。

算法原理、精度数值和性能数据仍应分别引用 F、H、I 的技术结果，不能从 CMake 通过结果
推导算法精度或加速比。

## 5. 范围边界

按用户在 2026-07-15 对 TASK_G 的明确指令，`test_all/performance_optimization/stability/` 是未纳入本次复赛
稳定性验证的 NVIDIA CUDA 历史范围，本任务将其视为不存在：未修改、未扫描、未构建、
未运行，本文也不声称稳定性测试已经通过。后续是否执行稳定性测试由用户单独决定。
