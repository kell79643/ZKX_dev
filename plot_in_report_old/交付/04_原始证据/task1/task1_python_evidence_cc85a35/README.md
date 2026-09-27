# Task1 cuSignal Python ZQ500 正式证据

- 被测提交：`cc85a352c4eede97e54c54393bcdce6b9cda2545`
- 工作树标志：`dirty=true`（共享工作区存在其他任务未提交改动）
- 测试时间：`2026-07-18T15:50:00+08:00`
- 平台：`usr_02@10.110.12.10` / `gpu_02` / `ZQ500-Q QUAD-2`
- 环境：Python 3.12.11、NumPy 2.4.6、CuPy 13.6.0 私有 CUDA 11.7/ZQ500 构建、
  cuSignal 23.08.00、CUDA runtime/driver 11070
- 验收：五个单步、pipeline、6/6 JSON、20 轮内存、Python/C++ 双方 1+10、11 项结果
  语义比较及 benchmark gate 全部 pass
- diPTI：本轮未使用；正式耗时来自 CuPy Event/CudaEvent 与阶段墙钟

性能比的定义为：

```text
Speedup = T_(cuSignal Python GPU + 必需 ZQ500 adapter) / T_(C++/CUDA GPU)
```

分子是 Python 同一步骤 `formal_execution_ms`，包含真实 cuSignal 调用和全部必需 adapter；
分母是 C++/CUDA GPU 同一步骤 `formal_execution_ms`，不含 CPU reference、validation 或
accuracy comparison。逐步比值为 `6.086560`、`1.235301`、`0.545796`、`1.427742`、
`0.413465`。

原始解包内容位于 `task1_python_evidence/`。配套压缩包为相邻文件
`task1_python_evidence_cc85a35.tar.gz`，SHA-256：
`8c0c36aaa1a771f705cac49be97cccd6f84fb89b6fb2e7b3ef51455b55c9c0c7`。
