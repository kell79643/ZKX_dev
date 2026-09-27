# 图18任务显存正式重测证据

- 被测提交：`08e1e250940dadc04a120c2461ac6127b258d88e`
- 平台：`usr_02@10.110.12.10` / `gpu_02` / `ZQ500-Q QUAD-2`
- 后端：`DLFFT=OFF`、`THRUST=ON`
- 轮次：每组预热 5 轮、正式采集 20 轮
- 主口径：任务自身分配事件的同时活动字节峰值（`task-allocator-live-peak`）
- 旁证口径：warmup 后以 1 ms 轮询的设备全局水位，仅用于诊断

`formal_memory_summary.txt`保留四条正式摘要和两条C++资源趋势摘要；容器生成的
`memory_retest_20260719.tar.gz`包含四组完整运行日志、运行元数据和日志SHA-256。
图表规范化结果见`02_技术结果/图表数据/正式数据/task_memory_observations.csv`。
证据包SHA-256：`79E9BB65A6F8E2F96C5348E207FE33A449F26F8336E1E83FF796802E211701E9`。

| Task | 实现 | 分配基线/B | 分配峰值/B | 峰值增量/B | 结束活动字节/B |
| --- | --- | ---: | ---: | ---: | ---: |
| Task1 | cuSignal Python GPU | 0 | 596864 | 596864 | 0 |
| Task1 | C++ CUDA GPU | 338944 | 671236 | 332292 | 338944 |
| Task2 | cuSignal Python GPU | 0 | 53248 | 53248 | 0 |
| Task2 | C++ CUDA GPU | 7168 | 83724 | 76556 | 7168 |

四组`allocator_status`均为`pass`。warmup后的设备全局基线和峰值均为357777408 B，
正式阶段增长均为0；该字段只用于证明设备水位稳定，不参与图18柱高。
