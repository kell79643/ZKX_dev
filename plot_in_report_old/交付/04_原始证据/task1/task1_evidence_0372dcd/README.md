# Task1 ZQ500 最终证据包

本目录是 Task1 采用“对应 CPU reference + L∞ 相对误差判定，同时保留最大绝对误差”后的
最终 ZQ500 验收证据。被测源码绑定真实 Git 提交：

```text
0372dcd5f8f512a6c2582c7fbf6b37fd90f41791
```

测试时间为 `2026-07-15T19:36:00+08:00`，环境为
`usr_02@10.110.12.10` 的 `gpu_02`，SDK 为 `/zq500/sdk`。

最终结论：

```text
CTest=6/6 passed
official_steps=5/5
cpu_references=5/5
relative_error_judgements=5/5
absolute_error_retained=5/5
accuracy_validation_files=5/5
accuracy_comparison_files=5/5
comparison_gpu_paths=0
dlpti_target_exitcodes=0/0/0
dlpti_exitcodes=0/0/0
status=passed
```

结构化最终数值源为
`task1_evidence/pipeline/task1_pipeline_evidence.json`；构建和 CTest 原始日志位于
`task1_evidence/logs/`；三轮 `ZQ500_diPTI` 辅助摘要位于 `task1_evidence/dlpti/`。

原始归档 `task1_evidence_0372dcd.tar.gz` 的 SHA-256：

```text
62a9979a695a4adc5269ba7bd198be484d42b276133a4f5a4689f7733ec9bac6
```

归档由容器中的 `task1_evidence/` 原样打包；远程宿主机和本地下载后哈希一致。
`logs/source-sha256.txt` 对应容器实际编译输入；本地复核结果见 Task1 技术结果文档。
