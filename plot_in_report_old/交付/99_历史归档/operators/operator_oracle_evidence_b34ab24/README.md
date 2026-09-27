# 算子高精度参考实现（Oracle）增量证据

本目录保存 2026-07-17 在 `usr_02@10.110.12.10` 的 `gpu_02` 容器中运行
`sensitive_operator_oracle_test` 得到的首批结构化结果。高精度测试实现绑定提交
`b34ab249`；正式算子与其他共享文件所在工作区为 dirty，因此本证据是增量候选证据，不能
替换 `task_f_evidence_439fd41/` 的 265 项冻结批次。最终 Word/PDF/PPT 冻结前应在目标提交的
clean/已记录 dirty 指纹下复跑并补齐原始日志和源码 SHA-256。

## 结果文件

- `oracle_metrics.csv`：3 个算子 × FP32/FP16 的 contract、高精度基准两项主误差和一项
  `diagnostic_compute` 定位数据；
- 主交付说明：`../OPERATOR_HIGH_PRECISION_ORACLE_RESULTS.md`；
- 测试源码：`test_all/operators/accuracy/oracle/`。

本批构建和运行命令：

```sh
cmake --build build_oracle --target sensitive_operator_oracle_test -j2
```

运行汇总为：

```text
[ORACLE][SUMMARY] operators=3 floating_dtypes=2 records=6 contract_reference_retained=1 high_precision_oracle=1 integer_oracle_deferred=1 status=passed
```

六条记录的 contract、diagnostic compute、high-precision reference mismatch 均为 0。高精度
参考实现耗时没有采集，也不
进入 CPU/GPU speedup。
