# 图9 Python 53 算子基线采集尝试

- 平台：ZQ500 `gpu_02`
- C++ 对照提交：`b027dca36ba1332cc9be5a340123bb2cfd17b543`
- 后端：`DLFFT=OFF`、`THRUST=ON`
- Python：3.12.11；CuPy 13.6.0 ZQ500 私有适配；cuSignal 23.08.00。
- 计划口径：与 TASK_F FP32 相同 shape/参数，每项预热 5 次、正式 20 次。
- 结果：未形成正式 CSV。stock cuSignal 路径出现 dlrtc/JIT 编译失败、
  `CUFFT_NOT_SUPPORTED`、cuSolver 未定义符号和 Bessel 内建函数 LLVM abort。
- 日志：`chart_python_operator_baseline_failed.log`
- 日志 SHA256：`86BFD2F44CD6F1A86F391EBC5252E9989FDA2420A084684BD71EDA42FF834371`

该失败证据用于约束图9状态，严禁以旧 RTX 4090 数据、C++ CPU 数据或推算值代替。
