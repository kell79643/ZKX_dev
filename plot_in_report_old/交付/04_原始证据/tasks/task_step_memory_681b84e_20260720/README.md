# Task1/Task2逐Step显存正式测试证据

- 测试时间：2026-07-20 09:11（Asia/Shanghai）
- 源码提交：`681b84e`；同步工作区为`dirty`，未提交改动不涉及本次两个测量目标
- 平台：`gpu_02` / `ZQ500-Q QUAD-2`
- SDK：`/zq500/sdk/env.sh`
- CMake：3.22.1
- 后端：`DLFFT=OFF`、`THRUST=ON`
- 构建目录：`/tmp/ZKX_dev/build_step_memory_20260720`
- 测量目标：`task1_step_memory`、`task2_step_memory`
- 轮次：每个Step预热5轮、正式测量20轮
- 范围：`step-exclusive-task-allocator-live-peak`

每轮重新构造任务状态。目标Step之前的步骤只用于准备真实输入；前序完成并同步后读取
baseline、重置项目分配器峰值，再只运行目标Step并同步读取peak。表中保留20轮里delta
最大的同一轮baseline、peak、delta，不跨轮拼接字段。Task1五行、Task2六行均为
`status=pass`，且目标Step结束后的活动字节回到该轮baseline。

容器正式日志SHA-256：

- `task1_step_memory.log`：`85a58fde8fa685b070e87e6ad5b493144d9500cc158928dfe680c0171bd2aaaa`
- `task2_step_memory.log`：`5d227292770f007aa48dbc62ff2d8dda58a8b3dbcb97224f7ab05f5f9521d9f4`

正式规范化数据见
`02_技术结果/图表数据/正式数据/task_step_memory.csv`。测试结束后`dlsmi`显示设备占用
回到293 MiB、GPU利用率0%、无运行进程。
