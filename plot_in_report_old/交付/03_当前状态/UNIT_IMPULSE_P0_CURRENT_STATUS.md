# `unit_impulse` P0 当前状态索引

本文用于把仓库内交付说明连接到当前性能优化证据，不覆盖
`task_f_evidence_439fd41/` 的历史 CSV、日志、提交绑定或当时结论。外层 `deliverables/`
中的 Word/PPT 由人工同步，不属于本文更新范围。

## 优化对象

本轮正式对象是 waveforms 模块的一维
`cusignal::unit_impulse_device<T>(DeviceArray<double>& out, int idx)`。FP32、FP16、
INT32、INT16、INT8 只表示五个 TASK_F 类型契约用例；该生成器没有业务数组输入，公开输出
固定为预分配 FP64 device 存储。优化后
由整数索引比较和 `uint64_t` 位模式写入直接生成 `0.0/1.0`，不执行 FP64 算术。

本轮没有把 tuple shape、字符串 `mid`、多维坐标、2D overload 或 Task1/Task2 流水线
纳入正式性能结论；当前两项 Task 的正式链路均不调用该算子。

## 当前结论（2026-07-18 正式闭环）

- `unit_impulse` P0 性能优化正式完成。
- benchmark contract 6/6、E5 5/5、E3 `unit_impulse` 20/20 通过。
- 五个 TASK_F 类型契约用例的零误差均通过路径独立、人工注错和独立位级复算三审。
- 固定版 cuSignal 23.08 的 docstring 宣称 dtype 可选，但实际源码未使用该参数、kernel 输出
  固定 FP64；源码自动审计及默认、FP64、FP32、FP16、INT32、INT16、INT8 七种请求的
  Python 黑盒均确认固定 FP64，见
  [dtype 契约审计](../../../test_all/performance_optimization/operators/unit_impulse/CUSIGNAL_DTYPE_CONTRACT_AUDIT_20260717.md)。
- clean `0f577cd...` 中以同一 executable、同一评价键完成 benchmark-only
  `legacy-host-finalize` 与公开 API `optimized-direct-storage` 的正式消融；优化前源码可由
  clean `e027da2...` 追溯。
- 五个 TASK_F 类型契约用例的正式 `S_opt(E)` 为 19.72–21.62×，优化后 CPU/GPU 中心
  加速比为 2.96–3.47×。
- 每个 dtype×implementation 保存五批共 100 个逐次 GPU 样本；优化后 pooled p95 为
  0.06203–0.105071 ms，相对优化前改善 12.34–20.90×，五类型均不回退。
- allocation、H2D、D2H、workspace、plan/handle 和 postprocess 全为 0。
- 二次审计 clean `d94509b...` 已补齐 20 批 before/after、kernel-only、资源字段和四规模
  crossover；共同 crossover=100K，异常样本和 1K/10K 慢区间未删除。
- 历史 `439fd41...` 的 20.71–22.01× 仍只作为历史参考比，不与正式 `S_opt(E)` 混写。

正式完成只针对本次一维独立算子 P0 切片，不表示 tuple/多维 API、Task1/Task2 或其他 P0
算子已经完成。当前 Task1/Task2 正式链路均不调用 `unit_impulse`，因此不报告 Task 收益。

## 三文档协作与原始数据

- 优化对象、固定场景和目标：
  [计划切片](../../../test_all/performance_optimization/operators/unit_impulse/PLAN_SLICE_20260717.md)
- 评价键、统计、三个加速比和 pass/fail：
  [指标判定](../../../test_all/performance_optimization/operators/unit_impulse/EVALUATION_20260717.md)
- 原始结构、修改、生效原理、失败历史和八项证据：
  [主证据文档](../../../test_all/performance_optimization/operators/unit_impulse/README.md)
- 2026-07-18 正式逐次样本：
  [日志](../../../test_all/performance_optimization/operators/unit_impulse/formal_ablation_0f577cd.log)
- 2026-07-18 正式汇总：
  [CSV](../../../test_all/performance_optimization/operators/unit_impulse/formal_ablation_0f577cd_summary.csv)
- 2026-07-18 20 批长稳原始样本与汇总：
  [日志](../../../test_all/performance_optimization/operators/unit_impulse/stability_0f577cd_20_batches.log)、
  [CSV](../../../test_all/performance_optimization/operators/unit_impulse/stability_0f577cd_20_batches_summary.csv)

对外引用时必须写明比较方向、评价键、单算子边界和 Task 不适用边界，不得只复制 19.72–21.62×。
