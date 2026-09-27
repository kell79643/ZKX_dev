# 算子高精度参考实现第二批增量证据

本目录保存 2026-07-17 在 ZQ500 `gpu_02` 容器中运行
`sensitive_operator_oracle_test` 得到的第二批结构化结果。第二批增加 `firfilter`、
`upfirdn`、`pulse_compression`，与第一批合计覆盖 6 个算子的 FP32、FP16。

本次运行位于共享 dirty 工作区，结果属于增量候选证据；最终冻结仍需在目标提交上复跑，
补齐原始日志、源码 SHA-256 和 clean/已记录 dirty 指纹。

运行汇总：

```text
[ORACLE][SUMMARY] operators=6 floating_dtypes=2 records=12 contract_reference_retained=1 high_precision_oracle=1 integer_oracle_deferred=1 status=passed
```

`oracle_metrics.csv` 只保存第二批新增的 6 条记录；第一批记录继续保留在
`../operator_oracle_evidence_b34ab24/oracle_metrics.csv`，不覆盖历史证据。
