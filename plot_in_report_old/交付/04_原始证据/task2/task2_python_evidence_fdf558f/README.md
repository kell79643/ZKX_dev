# Task2 C++/cuSignal Python ZQ500 验收证据

本目录是将 ZQ500 兼容工具迁移到 `Task2/task2_python/utils`、恢复不可修改的原版
`cusignal-23.08.00` 并清除容器旧 `zq500_tools` 残留后，于 2026-07-16 重跑得到的正式证据。

- Task2 提交：`fdf558f9a344a29d53e4f77b07ef59a9a918f8d3`；
- 工作树：`dirty=true`，共享树的 cuSignal 合同更新另由逐文件 SHA-256 固定；
- 测试时间：`2026-07-16T10:46:25+00:00`。

关键内容：

- `task2_python_evidence/benchmark/task2_python_benchmark_summary.json`：Python 1+10 汇总；
- `task2_python_evidence/cpp_benchmark/run_1`～`run_10`：C++ 十轮完整 pipeline JSON；
- `task2_python_evidence/task2_cpp_python_performance.json`：同输入契约双方比较；
- `task2_python_evidence/step1`～`step6`、`pipeline/`：Python 独立入口和整链结果；
- `task2_python_evidence/logs/`：环境、Git 状态、源码 SHA-256、合同、显存和门禁日志；
- `task2_python_evidence_fdf558f.tar.gz`：从 `gpu_02` 导出的原始归档。

源码哈希清单只包含 `Task2/task2_python/utils`，不包含旧 `zq500_tools`。本批计时未使用
`ZQ500_diPTI`；`cusignal-23.08.00` 源码没有为 Task2 做任何修改。
