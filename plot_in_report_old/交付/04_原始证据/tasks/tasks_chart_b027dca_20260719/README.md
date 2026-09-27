# Task 图表正式采样证据

- 代码提交：`b027dca36ba1332cc9be5a340123bb2cfd17b543`
- 平台：ZQ500 `gpu_02`
- FFT 后端：`DLFFT=OFF`、`THRUST=ON`
- Task1 pipeline：Python/C++ 各预热 5 次、正式 100 次。
- Task2 pipeline：Python/C++ 各 5 批，每批预热 5 次、正式 20 次；合计各 100 次正式样本。
- 历史设备显存：Task1/Task2 × Python/C++ 共4条，均为1 ms设备全局水位采样、正式20轮；
  该批显存数据已被`task_memory_retest_08e1e25_20260719/`的分配事件峰值正式替代。
- 证据包：`chart_tasks_b027dca_20260719.tar.gz`
- SHA256：`ACEE5F945C7D4158580FB188A5FC97D2ED6DFEE4887E1E300EE6AC4A08E14C9E`

证据包保留逐轮JSON、benchmark摘要和历史显存原始日志。Task1 C++的CPU live-heap
方向性启发式在20轮窗口内标记为`fail`，GPU全局净增长为0；该诊断不得改写或隐藏，
但本批warmup前共同48 MiB水位不再用于图18。图18现读取新版CSV的任务分配器峰值和
`allocator_status`。
