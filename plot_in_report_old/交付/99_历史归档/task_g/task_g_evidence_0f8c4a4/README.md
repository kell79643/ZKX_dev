# TASK_G ZQ500 CMake 最终证据索引

> **状态：`invalidated`，禁止用于最终交付。** 最终审计发现该次运行从远端残留目录编译了
> 22 个已归档旧算子源文件。有效替代证据为
> `semi_final_tasks/交付/04_原始证据/task_g/task_g_evidence_db7763c/README.md`；本目录只保留为诊断历史。

## 已失效证据身份

- 最终复验提交：`0f8c4a4e6acde4efd10d22ca462618e7d9ccf168`
- Task2 最新正式源码提交：`73c0ba1ec564943c12c2da99929de7d8c6b031d7`
- 测试时间：`2026-07-15T08:16:34+00:00`
- 远端环境：`usr_02@10.110.12.10` 的 `gpu_02`
- SDK：`/zq500/sdk`，`DLI_V2=ON`
- CMake：3.22.1；编译器：`dlcc`/Clang 15.0.6
- 工作树状态：`present-bound-in-g-manifest`

共享仓库在首次 G-final 全量运行期间新增了 Task2 正式源码提交 `73c0ba1`，因此首次绑定
`5f5b65e` 的日志没有作为最终证据。本目录只索引重新配置并完整运行当前 `0f8c4a4` 后的
有效结果。未提交改动不涉及本次编译的 Task1、Task2 或算子 C++/CUDA 源码；实际被测源码
另由 `source-sha256.txt` 逐文件绑定。

## 最终门禁

```text
active_cmake=13
active_cmake_violations=0
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
| `task_g_0f8c4a4_configure_final.log` | 最终配置原始日志，含活动 CMake、算子合同和 Task1 源码合同门禁 |
| `task_g_0f8c4a4_build_run_final.log` | 统一 `test_all` 完整编译与运行原始日志 |
| `gate-summary.txt` | Task1、Task2、53 算子和统一入口的关键汇总行 |
| `compile_commands.json` | 298 条实际编译命令，绑定 154 个唯一源文件 |
| `CMakeCache.txt` | 最终配置 cache |
| `target-help.txt` | 53 个独立算子、12 个类别及 Task 目标清单 |
| `source-sha256.txt` | 282 个 Task1、Task2、算子 `.cu/.cpp/.h` 文件 SHA-256 |
| `active-cmake-scan.txt` | 当前活动 CMake 零违规结果 |
| `environment.txt` | commit、时间、SDK、CMake 和 dlcc 版本 |
| `official-sample-paths.txt` | vectorAdd、dlrand 官方样例路径 |
| `official-dlfft-sample-paths.txt` | dlfft 官方样例路径 |

编译命令复核结果：298/298 使用 `/zq500/sdk/bin/dlcc`；293 条 GPU 编译命令均包含
`-x cuda` 和 `--offload-arch=dlgput64,dlgpux64`；`nvcc`、`sm_XX`、`compute_XX`
命中均为 0。其余 5 条为无需 GPU 编译选项的纯 C++ 源文件。
