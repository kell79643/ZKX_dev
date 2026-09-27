# TASK_F 正式图表数据批次

- commit：`b027dca36ba1332cc9be5a340123bb2cfd17b543`
- FFT 后端：`USE_DLFFT=OFF`、`USE_THRUST=ON`
- 批次：`chart_final_b027dca_20260719_thrust`
- 性能采样：预热 5 次，每个算子×dtype 正式 100 次 GPU 样本和 100 次 CPU 样本
- 严格汇总：53×5=265 项完整通过；正确性、精度、性能均为 265 项；原始延迟记录
  53,000 条；MSE/RMSE 随精度记录保存
- E8/EL6：`done=265`，与本批次 commit、branch、dirty state 一致

目录中的 `chart_task_f_b027dca.tar.gz` 是从 `gpu_02` 取回的原始归档，SHA-256 为
`F8D6E2175AA166E77BFC9C687A1EE22A4B63AFF481CFAE3F39C1C3ABA1407AD1`。解包内容保持
容器内相对路径；规范化图表数据应读取 `test_all/operators/results/structured/` 下的
`task_f_results.csv`，原始样本与环境、构建、门禁日志位于对应 `raw/` 目录。
