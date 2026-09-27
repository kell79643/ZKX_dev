# CSV字段说明

> 阅读顺序：先读同目录`README.md`了解文件身份和用途，再用本文件查CSV行粒度、主键、关联键和全部列含义。

## 覆盖结论

- 本目录直接CSV：9个（含治理文件`folder_manifest.csv`）
- 本目录直接CSV已映射：9个
- 本目录直接CSV未映射：0个
- 本目录不同schema：9种
- 生成与校验时间：`2026-08-14T17:33:09+08:00`
- 校验规则：真实列名、列数、顺序及规范化表头SHA-256必须与schema一致。

## 直接CSV到schema映射

| CSV文件 | schema_id | 字段说明 | 表头SHA-256 | 列数 | 一行代表 | 主键 | 配对键 | 来源/生成方式 | 校验状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| [`12模块_性能汇总候选.csv`](./12模块_性能汇总候选.csv) | `SCHEMA_3ACCC959CC57` | [查看全部列](#schema_3accc959cc57) | `3accc959cc57dc1b4a7bd4d897690fd334ce009ae6ac0fa8bd8f16679182f9d1` | 13 | 一行对应一个由该CSV生成程序定义的记录对象。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`4类代表算子候选.csv`](./4类代表算子候选.csv) | `SCHEMA_657BBB94EDDF` | [查看全部列](#schema_657bbb94eddf) | `657bbb94eddf9abde1e49ebc8e7b101731b09d782311400ea9a83a7870972a1a` | 7 | 一行对应一个由该CSV生成程序定义的记录对象。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`53x5_最差精度候选.csv`](./53x5_最差精度候选.csv) | `SCHEMA_53562F8B9951` | [查看全部列](#schema_53562f8b9951) | `53562f8b995177314072e1adf310ba793c17cb1b41f4d39d421cd99f90712823` | 21 | 一行对应一个由该CSV生成程序定义的记录对象。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`53算子_12模块映射.csv`](./53算子_12模块映射.csv) | `SCHEMA_3B8D62FCD6D0` | [查看全部列](#schema_3b8d62fcd6d0) | `3b8d62fcd6d0bb51fabeb297405ee2aab60597cb53865e9b86fc95cca47c8a65` | 2 | 一行对应一个由该CSV生成程序定义的记录对象。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`53算子_全case性能区间候选.csv`](./53算子_全case性能区间候选.csv) | `SCHEMA_D383C06B6201` | [查看全部列](#schema_d383c06b6201) | `d383c06b6201a39a43f7fd80b4aa8d0031e8b8a29546a8740a38307e6ebd7784` | 30 | 一行对应一个由该CSV生成程序定义的记录对象。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`53算子_演示case性能候选.csv`](./53算子_演示case性能候选.csv) | `SCHEMA_071B901F8846` | [查看全部列](#schema_071b901f8846) | `071b901f8846e1ab09f3ac9cfe51c9edc6033310a30a0c0d63232fc1164bbe9d` | 23 | 一行对应一个case记录。 | `run_id + case_id + operator_name` | `run_id + case_id` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`全算子_逐case绘图候选.csv`](./全算子_逐case绘图候选.csv) | `SCHEMA_7ED5A58734DD` | [查看全部列](#schema_7ed5a58734dd) | `7ed5a58734dd95bb96c19f8cbb8be5ced297cb0b4414072f8c17247a4b011b9e` | 60 | 一行对应一个case记录。 | `stage + operator_name + case_id` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`窗函数_435case热力图候选.csv`](./窗函数_435case热力图候选.csv) | `SCHEMA_A7B6403ACEEB` | [查看全部列](#schema_a7b6403aceeb) | `a7b6403aceebfd63657bc0344d5748334f1ebfb49b34debc9e9e50acc398873a` | 25 | 一行对应一个case记录。 | `run_id + case_id + operator_name` | `run_id + case_id` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_071b901f8846"></a>
### SCHEMA_071B901F8846

- 适用文件：`53算子_演示case性能候选.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 3 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 4 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 5 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic_fp16_10e4` | NA |
| 6 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10000` | NA |
| 7 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[10000]` | NA |
| 8 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.07658552` | warmup_runs, measured_runs, timing_scope |
| 9 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.21139722` | warmup_runs, measured_runs, timing_scope |
| 10 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.3622825314353708` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 11 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.217279` | warmup_runs, measured_runs, timing_scope |
| 12 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.08293981931311126` | NA |
| 13 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 14 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 15 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 16 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 17 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 18 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 19 | `resource_gate_status` | string | NA | 样本未见空值 | `resource gate`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NOT_EVALUATED_IN_COMPARISON_INDEX` | NA |
| 20 | `leak_gate_status` | string | NA | 样本未见空值 | `leak gate`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NOT_EVALUATED_IN_COMPARISON_INDEX` | NA |
| 21 | `demo_status` | string | NA | 样本未见空值 | `demo`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PERFORMANCE_SHORTLIST_NOT_BEST_DEMO_CASE` | NA |
| 22 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110446Z_6132f71eb1e0_stage07_b06_formal_c6bea09d` | NA |
| 23 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage07_b06_formal_6132f71/formal/20260810T110446Z_6132f71eb1e0_stage07_...` | NA |

<a id="schema_3accc959cc57"></a>
### SCHEMA_3ACCC959CC57

- 适用文件：`12模块_性能汇总候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 2 | `operator_count` | integer | count | 样本未见空值 | `operator`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 3 | `total_cases` | integer | count | 样本未见空值 | 当前汇总范围内全部case数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `45` | NA |
| 4 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `45` | NA |
| 5 | `pass_rate` | integer | dimensionless | 样本未见空值 | `pass`占总数的比例。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 6 | `median_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的中位数。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.122668908006515` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 7 | `p25_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的第25百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.0647819756642247` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 8 | `p75_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的第75百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.149714532686372` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 9 | `min_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的最小值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.00689504332193441` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 10 | `max_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的最大值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.176760157366229` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 11 | `operators_median_below_1x` | integer | NA | 样本未见空值 | 算子中位加速比小于1的算子数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 12 | `worst_accuracy_ratio` | string | dimensionless | 样本未见空值 | 最差精度值相对其适用阈值的比值；阈值未绑定时必须为明确NA状态，不能臆造。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_THRESHOLD_SNAPSHOT_NOT_BOUND` | NA |
| 13 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_3b8d62fcd6d0"></a>
### SCHEMA_3B8D62FCD6D0

- 适用文件：`53算子_12模块映射.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |

<a id="schema_53562f8b9951"></a>
### SCHEMA_53562F8B9951

- 适用文件：`53x5_最差精度候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 3 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07` | NA |
| 4 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 5 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 6 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 7 | `worst_accuracy_ratio` | string | dimensionless | 样本未见空值 | 最差精度值相对其适用阈值的比值；阈值未绑定时必须为明确NA状态，不能臆造。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_THRESHOLD_SNAPSHOT_NOT_BOUND` | NA |
| 8 | `max_mse` | float | NA | 样本未见空值 | 当前聚合范围内`mse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 9 | `max_mse_case_id` | string | NA | 样本未见空值 | 产生`max_mse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 10 | `max_rmse` | float | NA | 样本未见空值 | 当前聚合范围内`rmse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 11 | `max_rmse_case_id` | string | NA | 样本未见空值 | 产生`max_rmse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 12 | `max_relative_l2` | float | NA | 样本未见空值 | 当前聚合范围内`relative l2`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 13 | `max_relative_l2_case_id` | string | NA | 样本未见空值 | 产生`max_relative_l2`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 14 | `max_relative_linf` | float | NA | 样本未见空值 | 当前聚合范围内`relative linf`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 15 | `max_relative_linf_case_id` | string | NA | 样本未见空值 | 产生`max_relative_linf`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 16 | `exact_match_false_count` | integer | count | 样本未见空值 | `exact match false`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 17 | `max_mismatch_count` | string | count | 允许 | 当前聚合范围内`mismatch count`的最大值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 18 | `max_mismatch_case_id` | string | NA | 允许 | 产生`max_mismatch`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 19 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 20 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 21 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_657bbb94eddf"></a>
### SCHEMA_657BBB94EDDF

- 适用文件：`4类代表算子候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `category` | string | NA | 样本未见空值 | 汇总、代表算子或图表对象所属的分类名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `最高性能收益` | NA |
| 2 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `spectrogram` | NA |
| 3 | `rule` | string | NA | 样本未见空值 | 产生当前记录或选择结果所采用的`rule`。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `全部case精度PASS后median_speedup最高` | NA |
| 4 | `overall_median_speedup` | float | dimensionless | 样本未见空值 | 当前全集内算子或case加速比的总体中位数，具体粒度由同表行定义确定。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.260742718902332` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 5 | `operator_median_speedup` | float | dimensionless | 样本未见空值 | 该算子全部纳入case的加速比中位数。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `10.4629706065054` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 6 | `cases_below_1x` | integer | NA | 样本未见空值 | 加速比小于1的case数量，即该case下GPU平均耗时高于CPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 7 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_7ed5a58734dd"></a>
### SCHEMA_7ED5A58734DD

- 适用文件：`全算子_逐case绘图候选.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 2 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 3 | `selection_status` | string | NA | 样本未见空值 | `selection`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PLOTTING_CANDIDATE_NEEDS_FINAL_QUALIFICATION` | NA |
| 4 | `formal_run_records_for_case` | integer | NA | 样本未见空值 | 同一case在去重前存在的正式运行记录数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 5 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07` | NA |
| 6 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07_b06_formal_6132f71` | NA |
| 7 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 8 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage07_b06_formal_6132f71/formal/20260810T110446Z_6132f71eb1e0_stage07_...` | NA |
| 9 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110446Z_6132f71eb1e0_stage07_b06_formal_fa81bf2c` | NA |
| 10 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 11 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110446Z` | NA |
| 12 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `6132f71eb1e0457ef1988f98e7e02226424d461c` | NA |
| 13 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 14 | `task_name` | string | NA | 样本未见空值 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `Task2` | NA |
| 15 | `step_name` | string | NA | 样本未见空值 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `Step3` | NA |
| 16 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 17 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp16_10e2__cb15388a9d7aeedc` | NA |
| 18 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic_fp16_10e2` | NA |
| 19 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 20 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `128` | NA |
| 21 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[128]` | NA |
| 22 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 23 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 24 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 25 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 26 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 27 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 28 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0007995200000000001` | warmup_runs, measured_runs, timing_scope |
| 29 | `cpu_p50_ms` | float | ms | 样本未见空值 | CPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0008` | warmup_runs, measured_runs, timing_scope |
| 30 | `cpu_p95_ms` | float | ms | 样本未见空值 | CPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0008` | warmup_runs, measured_runs, timing_scope |
| 31 | `cpu_p99_ms` | float | ms | 样本未见空值 | CPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0008107000000000004` | warmup_runs, measured_runs, timing_scope |
| 32 | `cpu_min_ms` | float | ms | 样本未见空值 | CPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.00079` | warmup_runs, measured_runs, timing_scope |
| 33 | `cpu_max_ms` | float | ms | 样本未见空值 | CPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.00088` | warmup_runs, measured_runs, timing_scope |
| 34 | `cpu_std_ms` | float | ms | 样本未见空值 | CPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `8.876350601457788e-06` | warmup_runs, measured_runs, timing_scope |
| 35 | `cpu_cv` | float | dimensionless | 样本未见空值 | CPU耗时变异系数，计算为cpu_std_ms/cpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.011102099511529151` | NA |
| 36 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.18717427` | warmup_runs, measured_runs, timing_scope |
| 37 | `gpu_p50_ms` | float | ms | 样本未见空值 | GPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.183809` | warmup_runs, measured_runs, timing_scope |
| 38 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.2024` | warmup_runs, measured_runs, timing_scope |
| 39 | `gpu_p99_ms` | float | ms | 样本未见空值 | GPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.24357071000000027` | warmup_runs, measured_runs, timing_scope |
| 40 | `gpu_min_ms` | float | ms | 样本未见空值 | GPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.178083` | warmup_runs, measured_runs, timing_scope |
| 41 | `gpu_max_ms` | float | ms | 样本未见空值 | GPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.297695` | warmup_runs, measured_runs, timing_scope |
| 42 | `gpu_std_ms` | float | ms | 样本未见空值 | GPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.014965651709066998` | warmup_runs, measured_runs, timing_scope |
| 43 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0799557103071218` | NA |
| 44 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.004271527277760988` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 45 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `typed_cpu_task2_reference` | NA |
| 46 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 47 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 48 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 49 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 50 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 51 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 52 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `output_count=128;output_dtype=FP16;task_gpu_call=present` | NA |
| 53 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 54 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 55 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 56 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 57 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 58 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 59 | `cpu_row_present` | boolean | NA | 样本未见空值 | `cpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |
| 60 | `gpu_row_present` | boolean | NA | 样本未见空值 | `gpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：`folder_manifest.csv`
- 一行代表：一行对应当前目录递归范围内的一个文件。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`relative_path`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 样本未见空值 | 相对于当前目录或表格约定根目录的文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 2 | `filename` | string | NA | 样本未见空值 | 文件名。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 3 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 4 | `file_size` | string | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 5 | `sha256` | string | NA | 样本未见空值 | 文件内容SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 6 | `source_host` | string | NA | 样本未见空值 | 证据来源主机。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 7 | `source_container` | string | NA | 样本未见空值 | 证据来源容器。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 8 | `source_path` | string | NA | 样本未见空值 | 证据在来源环境中的路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 9 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 10 | `test_object` | string | NA | 样本未见空值 | 测试对象的统一名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 11 | `task` | string | NA | 样本未见空值 | 任务名称或任务标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step` | string | NA | 样本未见空值 | 任务步骤标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 14 | `operator` | string | NA | 样本未见空值 | 算子标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 15 | `run_id` | string | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 16 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 17 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 18 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 19 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 20 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 21 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 22 | `git_dirty` | string | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 23 | `evidence_status` | string | NA | 样本未见空值 | 证据资格或归档状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NA` | NA |
| 24 | `formal_eligible` | string | NA | 样本未见空值 | 该证据是否满足正式使用资格。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 25 | `related_csv` | string | NA | 样本未见空值 | 与当前证据关联的CSV路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 26 | `related_log` | string | NA | 样本未见空值 | 与当前证据关联的完整日志路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 27 | `related_screenshot` | string | NA | 样本未见空值 | 与当前证据关联的截图路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 28 | `csv_schema_id` | string | NA | 样本未见空值 | CSV真实表头对应的稳定schema ID，由规范化表头SHA-256前缀生成。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 29 | `csv_column_document` | string | NA | 样本未见空值 | 该CSV逐列说明文件及schema锚点的相对路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 30 | `csv_header_sha256` | string | NA | 样本未见空值 | `csv_header`所指对象内容的SHA-256摘要。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 31 | `csv_column_count` | string | count | 样本未见空值 | `csv column`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 32 | `notes` | string | NA | 样本未见空值 | 补充说明、限制、NA规则或审计备注。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_a7b6403aceeb"></a>
### SCHEMA_A7B6403ACEEB

- 适用文件：`窗函数_435case热力图候选.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 2 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 3 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp16_10e2` | NA |
| 4 | `variant` | string | NA | 样本未见空值 | 输入或算子参数的语义变体名称，例如sym或periodic。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `periodic` | NA |
| 5 | `variant_key` | string | NA | 样本未见空值 | 由case参数和语义变体提取的稳定键，用于同算子、dtype和规模内区分不同输入变体。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1e7167366f8d98d0_periodic` | NA |
| 6 | `case_parameters_json` | JSON text | NA | 允许 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"length":100,"requested_dtype":"FP16","sym":false,"attenuation_db":80,"algor...` | NA |
| 7 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp16_10e2__1e7167366f8d98d0__periodic` | NA |
| 8 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 9 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 10 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.21930527000000008` | warmup_runs, measured_runs, timing_scope |
| 11 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.45719789` | warmup_runs, measured_runs, timing_scope |
| 12 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.15049793271386092` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 13 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.5123675000000001` | warmup_runs, measured_runs, timing_scope |
| 14 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.018416283922291041` | NA |
| 15 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `5.9750327471063565e-14` | accuracy_reference, accuracy_status |
| 16 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.4443880107516393e-07` | accuracy_reference, accuracy_status |
| 17 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `4.4652958722329837e-07` | accuracy_reference, accuracy_status |
| 18 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `4.888965245680027e-07` | accuracy_reference, accuracy_status |
| 19 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 20 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 21 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 22 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260807T052446Z_a97933046e9c_stage08_b01_formal_8a000107` | NA |
| 23 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b01_chebwin_fft_thrust/formal/20260807T052446Z_a97933046e9c_stag...` | NA |
| 24 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 25 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PLOTTING_CANDIDATE_NEEDS_FINAL_QUALIFICATION` | NA |

<a id="schema_d383c06b6201"></a>
### SCHEMA_D383C06B6201

- 适用文件：`53算子_全case性能区间候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 3 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07` | NA |
| 4 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 5 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 6 | `total_case_count` | integer | count | 样本未见空值 | `total case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 7 | `median_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的中位数。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.00689504332193441` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 8 | `p25_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的第25百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.00277444310276553` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 9 | `p75_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的第75百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.0359768450042551` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 10 | `min_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的最小值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.000774836200325786` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 11 | `max_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的最大值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.362282531435371` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 12 | `cases_below_1x` | integer | NA | 样本未见空值 | 加速比小于1的case数量，即该case下GPU平均耗时高于CPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 13 | `worst_accuracy_ratio` | string | dimensionless | 样本未见空值 | 最差精度值相对其适用阈值的比值；阈值未绑定时必须为明确NA状态，不能臆造。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_THRESHOLD_SNAPSHOT_NOT_BOUND` | NA |
| 14 | `worst_accuracy_case_id` | string | NA | 样本未见空值 | 产生`worst_accuracy`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_SEPARATE_METRICS_REPORTED` | NA |
| 15 | `max_mse` | float | NA | 样本未见空值 | 当前聚合范围内`mse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `8.67362e-17` | NA |
| 16 | `max_mse_case_id` | string | NA | 样本未见空值 | 产生`max_mse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 17 | `max_rmse` | float | NA | 样本未见空值 | 当前聚合范围内`rmse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `9.31323e-09` | NA |
| 18 | `max_rmse_case_id` | string | NA | 样本未见空值 | 产生`max_rmse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 19 | `max_relative_l2` | float | NA | 样本未见空值 | 当前聚合范围内`relative l2`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.01964e-08` | NA |
| 20 | `max_relative_l2_case_id` | string | NA | 样本未见空值 | 产生`max_relative_l2`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 21 | `max_relative_linf` | float | NA | 样本未见空值 | 当前聚合范围内`relative linf`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `8.94584e-08` | NA |
| 22 | `max_relative_linf_case_id` | string | NA | 样本未见空值 | 产生`max_relative_linf`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 23 | `max_mismatch_count` | integer | count | 允许 | 当前聚合范围内`mismatch count`的最大值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 24 | `max_gpu_p95_ms` | float | ms | 样本未见空值 | 当前聚合范围内`gpu p95 ms`的最大值。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.27393255` | warmup_runs, measured_runs, timing_scope |
| 25 | `max_gpu_p95_case_id` | string | NA | 样本未见空值 | 产生`max_gpu_p95`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e4__4ebb8a2d94032c64` | NA |
| 26 | `max_gpu_cv` | float | NA | 样本未见空值 | 当前聚合范围内`gpu cv`的最大值。 | 原始记录字段 | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.08293981931311126` | NA |
| 27 | `max_gpu_cv_case_id` | string | NA | 样本未见空值 | 产生`max_gpu_cv`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 28 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 29 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 30 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../README.md)
- [全局CSV字段语义总表](../../00_归档规范/CSV字段语义总表.md)
