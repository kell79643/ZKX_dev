# TASK_F 当前源码证据包

本目录保存 TASK_F 在中科芯 ZQ500 上的最终批次原始证据、结构化结果、派生指标和一次
`ZQ500_diPTI` 辅助采样。主证据绑定如下：

- 源码提交：`439fd41f78ada42f1f19a133ca987253d0be6ae2`；
- 分支：`feature/current-workflow`；
- 未提交状态指纹：`815fabcf1947a0a70d248f9d14c55c612217b3cd26bda2de22f4c5cee97e6971`；
- 批次：`task_f_final_439fd41_20260715`；
- 测试时间：`2026-07-15T04:11:46+00:00`；
- 远程环境：`usr_02@10.110.12.10` 的 `gpu_02` 容器，ZQ500-Q QUAD-2；
- 最终结果：correctness、accuracy、performance、complete 均为 265/265，严格聚合通过。

## 目录说明

- `test_all/operators/results/raw/task_f_final_439fd41_20260715/`：环境、Git 绑定、
  CMake、E8/EL6、框架门禁、九组正确性/精度日志、十组性能日志和最终驱动汇总；
- `test_all/operators/results/structured/task_f_final_439fd41_20260715/`：正式 265 行
  CSV/JSON、性能候选 CSV/JSON 和 `compile_commands.json`；
- `task_f_metrics_265.csv`：从同批原始记录按稳定测试 ID 聚合的 265 行技术文档数据源；
- `dlpti_correlate_fp32/`：对当前提交 `correlate/FP32` 的 profiler 辅助采样摘要；
- `SHA256SUMS.csv`：除清单自身外，本证据包全部文件的字节数和 SHA-256。

原始正确性/精度日志包含多个非退化、边界和非法参数 case，因此日志中存在 278 条
case 记录；正式验收维度是 53 算子 × 5 输入类型，即 265 个稳定测试 ID。聚合器要求
每个稳定 ID 的全部 case 均通过，不把 278 误写为算子类型组合数。

`ZQ500_diPTI` 只用于解释一次自编 benchmark 的活动/API 记录，不替代主批次耗时、
CPU/GPU 精度对照或最终性能表。其采样时 benchmark 均值与主批次均值不同，技术文档中
单独说明 profiler 开销和采样范围，不把该差异解释为算法误差。

## 完整性校验

在本目录中逐行读取 `SHA256SUMS.csv`，对对应相对路径重新计算 SHA-256。清单使用 UTF-8
CSV；路径分隔符统一为 `/`。任何文件内容变化都会导致校验失败，应生成新的证据版本，
不得静默覆盖本包。
