# `firwin` P0 当前状态（评委同步版）

结论：`firwin` 的 FP32、FP16、INT32、INT16、INT8 五个 TASK_F 类型项已经按当前性能
规范正式完成；这是 53×5=265 项中的 5 项，不是 265 项全部优化完成，也不是 Task2
端到端加速。

## 原始问题与修改

优化前公开路径每次执行 cutoff D2H、Host 参数准备、bands/window H2D、临时分配、FP32
kernel、结果 D2H、Host FP32→FP64 扩宽和最终 H2D。原始 257 taps 五类型 CPU/GPU 只有
0.03399–0.03577×。

当前实现保持固定 FP64 输出存储和 FP32 算术，用 device 位操作删除无收益 Host FP64 收尾；
用 `FirwinDeviceWorkspace` 把参数准备、分配和上传移出稳定调用；用并行归约和单 kernel 融合
消除重复归一化。1024 taps 以上使用已验证的 Stage C 通用回退。

## 正式结果

- compatibility 257 taps：同口径 `S_opt(E)=2.780–2.878×`，每批 p95 中心改善
  2.699–2.923×；但 CPU/GPU 仍为 0.0978–0.1008×，没有隐藏这一不利结果。
- device-resident 1024 taps、20 批：Stage C→fused `S_opt(E)=1.190–1.206×`，
  CPU/GPU=1.572–1.592×，pooled p95 五类型均改善。
- crossover：33/257/513 taps 仍慢于 CPU；768/1024/2048/4096 taps 五类型均快于
  CPU，已测共同边界为 `513 < N* <= 768`。
- 正确性：E3 estimation/filter-design 15/15；resident 最大绝对误差
  `1.85423e-08`、RMSE `4.22869e-09`，mismatch、NaN/Inf、对称性误差均为 0。
- 首次五批抖动门禁 `pass=0` 和全部长尾样本均保留；追加 20 批后正式通过，不声称无抖动。

## 完整证据入口

本交付文件与源码仓库一起交付时，完整八项证据位于：

- `test_all/performance_optimization/operators/firwin/README.md`
- `test_all/performance_optimization/operators/firwin/EVALUATION_FINAL_20260718.md`
- `test_all/performance_optimization/operators/firwin/evidence_231cfe2/`

后续若把 resident 能力接入 Task2 Step2，必须另做同 Step 和完整 pipeline 的 before/after；
本页不会提前写入该收益。
