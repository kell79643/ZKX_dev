# TASK_G ZQ500 clean 构建最终证据索引

## 证据身份

- 最终复验提交：`db7763c5b7b5e942dcea43d3d31be017a283cb0d`
- 源码状态：由该提交直接生成的 clean Git archive；不含未跟踪文件
- 测试时间：`2026-07-15T09:04:37+00:00`
- 远端环境：`usr_02@10.110.12.10` 的 `gpu_02`
- clean 源目录：`/tmp/ZKX_g_clean_db7763c`
- SDK：`/zq500/sdk`，`DLI_V2=ON`
- CMake：3.22.1；编译器：`dlcc`/Clang 15.0.6

## 为什么必须重新生成本证据

首次绑定 `0f8c4a4` 的运行虽然返回 0，但最终源码审计发现远端覆盖同步残留了 22 个已由
`efa4af8` 归档的旧 `*.cpp/*_cuda.cu`，旧 CMake 引用因此没有在配置期失败。`db7763c`
把 88 处测试源引用切换到当前 `*_typed.cpp/*_typed.cu`，并在活动 CMake 门禁中禁止旧路径。

本次使用 Git archive 新建远端和容器源码目录，旧文件物理不存在；只额外复制合同门禁要求的
固定只读参考树 `cusignal-23.08.00`、`cupy-13.6.0`。因此本目录取代
`task_g_evidence_0f8c4a4/`，后者只保留为明确失效的诊断历史。

## 最终门禁

```text
active_cmake=13
active_cmake_violations=0
compiled_sources=132
compiled_sources_missing=0
operator_targets_build_run=53/53
category_targets_build_run=all (12/12)
task1_step_targets_build_run=5/5
task1_task_target_build_run=true
task2_step_targets_build_run=6/6
task2_task_target_build_run=true
test_all_build_run=true
missing=0
failed=0
```

`test_all/stability/` 按用户对本次 TASK_G 的明确范围指令视为不存在：未扫描、未修改、
未构建、未运行，也不在上述通过结论内。

## 文件说明

| 文件 | 内容 |
|---|---|
| `task_g_db7763c_configure_final.log` | clean 源树最终配置原始日志 |
| `task_g_db7763c_build_run_final.log` | clean 源树统一 `test_all` 完整编译运行日志 |
| `gate-summary.txt` | Task1、Task2、53 算子和统一入口关键汇总行 |
| `compile_commands.json` | 298 条实际编译命令，绑定 132 个唯一编译源 |
| `compiled-sources.txt` | 从编译命令提取的 132 个唯一源路径 |
| `compiled-source-closure.txt` | `compiled_sources=132 missing=0` 回归门禁结果 |
| `CMakeCache.txt` | 最终配置 cache |
| `target-help.txt` | 53 个独立算子、12 个类别及 Task 目标清单 |
| `source-sha256.txt` | 154 个 Task1、Task2、算子 `.cu/.cpp/.h` 文件 SHA-256 |
| `active-cmake-scan.txt` | 当前活动 CMake 零违规结果 |
| `environment.txt` | commit、时间、SDK、CMake 和 dlcc 版本 |
| `official-sample-paths.txt` | vectorAdd、dlfft、dlrand 官方样例路径 |

编译命令复核结果：298/298 使用 `/zq500/sdk/bin/dlcc`；293 条 GPU 编译命令均包含
`-x cuda` 和 `--offload-arch=dlgput64,dlgpux64`；`nvcc`、`sm_XX`、`compute_XX`
命中均为 0。其余 5 条为无需 GPU 编译选项的纯 C++ 源文件。
