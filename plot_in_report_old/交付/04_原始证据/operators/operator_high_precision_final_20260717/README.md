# 高精度 CPU 参考最终证据

本目录冻结 2026-07-17 在中科芯 ZQ500 上完成的最终深度验证。结果为 7 个代表算子、
20 个场景、40/40 精度记录通过，以及 Task1 2/2 峰值业务记录通过。

## 先看结论

- [最终深度验证报告](FINAL_HIGH_PRECISION_CPU_VALIDATION.md)
- [复赛答辩讲述材料](复赛答辩_高精度CPU参考讲述材料.md)

## 可复核证据

- [ZQ500 原始结构化日志](raw_zq500_oracle.log)
- [40 条算子/任务链精度指标](oracle_metrics.csv)
- [2 条 Task1 峰值业务指标](task_metrics.csv)
- [运行环境、提交与 SHA-256 指纹](environment_and_fingerprint.md)

## 使用边界

高精度 CPU 用于精度风险和 dtype 决策，不用于 CPU/GPU 性能加速比。正式 Task1/Task2
保持 FP32；FP16 只可在具体非敏感路径完成业务指标复测后局部采用。
