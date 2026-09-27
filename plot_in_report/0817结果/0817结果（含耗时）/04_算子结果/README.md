# 算子结果

## 目录结构

```text
04_算子结果/
├─ 绘图数据/              # 主PPT、备份页和技术文档的短路径绘图候选表
├─ task_operators/       # 阶段07：20个任务涉及算子及展开数据
├─ remaining_operators/  # 阶段08：33个剩余算子及展开数据
├─ CSV可打开性审计.csv   # 每个CSV的编码、语法、列数、行数和路径风险
└─ CSV字段模式汇总.csv   # 35种CSV表头模式及样例路径
```

阶段07与阶段08集合应满足`overlap=0, union=53`。回收后按模块和算子继续建立叶子目录；每个新增目录同时建立README，列出正式结果、smoke、终端日志、精度、性能、资源和当前证据状态。

容器封存包中还包含旧运行、失败后保留目录和重复重跑。它们不会因进入本地归档自动成为正式证据，明确的历史结果转入`90_历史证据`，身份冲突结果才进入`99_隔离与待确认`。

当前阶段07/08已经完整展开到各自`data/`目录，阶段集合不再混放。`data/.../cpu_gpu_comparison.csv`是答辩、技术文档和绘图的首选入口；`runs/`保存未改写的逐次原始CSV、JSON、TXT与LOG。

## 展示结论的总原则

不建议“每个算子只展示性能最好的case”。最佳case可以用于现场演示，但不能作为算子整体性能结论，否则无法回答其余case是否更慢、是否全部通过精度门禁、是否存在GPU慢于CPU等问题。

> 主PPT展示全量case的统计分布，备份页展示53个算子明细，技术文档和CSV保留每个case的完整数据；最佳case只用于演示选择。

当前绘图候选层按以下确定性规则从回收数据中去重：

```text
source_partition=formal
run_type=formal
status=PASS
accuracy_status=PASS
CPU/GPU行均存在
同一 operator_name + case_id 取 test_time_utc 最新的正式记录
```

规则编号为`LATEST_FORMAL_PASS_PER_CASE_V1`。它只解决重复正式运行的确定性去重，不是“挑最快case”。当前结果为阶段07的302个唯一case、阶段08的810个唯一case，合计1,112个case。

所有`绘图数据/*候选.csv`仍标记为`CANDIDATE_NOT_FINAL`或`PLOTTING_CANDIDATE_NEEDS_FINAL_QUALIFICATION`。在最终资格清单、资源/泄漏门禁和精度阈值快照完成绑定前，不得改名为“正式绘图数据”。

## 所有建议图表的数据位置

| 使用位置或图表 | 直接使用的文件 | 关键列 | 当前边界 |
| --- | --- | --- | --- |
| 主PPT：12模块性能箱线图 | `绘图数据/53算子_全case性能区间候选.csv` | `module, median_speedup` | 一算子一个中位数，再按模块作箱线图，禁止把全部case直接混合 |
| 主PPT：模块数字摘要 | `绘图数据/12模块_性能汇总候选.csv` | `operator_count,total_cases,median_operator_speedup,operators_median_below_1x` | 是汇总表，不替代模块箱线图的算子级输入 |
| 备份页：53算子性能区间图 | `绘图数据/53算子_全case性能区间候选.csv` | `min_speedup,p25_speedup,median_speedup,p75_speedup,max_speedup,cases_below_1x` | 一行一个算子，可拆成两页 |
| 主PPT/备份页：53×5最差精度热力图 | `绘图数据/53x5_最差精度候选.csv` | `dtype,max_mse,max_rmse,max_relative_l2,max_relative_linf`及各自`*_case_id` | 当前未绑定阈值快照，必须画四张小热力图，不能合成门限比 |
| 窗口模块热力图 | `绘图数据/窗函数_435case热力图候选.csv` | `operator_name,dtype,scale_id,variant,variant_key,case_parameters_json,cpu_gpu_speedup,accuracy_status` | 含阶段07 hamming 15例及阶段08七个窗函数各60例 |
| 代表算子选择 | `绘图数据/4类代表算子候选.csv` | `category,operator_name,rule` | 规则化候选，不按人工审美挑选 |
| 代表算子多规模图 | `绘图数据/全算子_逐case绘图候选.csv` | `actual_elements,scale_id,dtype,cpu_mean_ms,gpu_mean_ms,gpu_p95_ms,gpu_cv` | 先按代表算子表筛选；三档规模只称多规模比较，不作连续趋势预测 |
| 演示case初筛 | `绘图数据/53算子_演示case性能候选.csv` | `case_id,speedup,gpu_p95_ms,gpu_cv`及精度列 | 只完成性能初筛，资源与泄漏门禁仍为`NOT_EVALUATED`，不能直接命名`BEST_DEMO_CASE` |
| 技术文档：12模块汇总 | `绘图数据/12模块_性能汇总候选.csv` | 全表 | `worst_accuracy_ratio`尚不可计算 |
| 技术文档：53算子汇总 | `绘图数据/53算子_全case性能区间候选.csv` | 全表 | 保留最差精度四指标各自数值和case，不伪造统一门限比 |
| 技术文档附录：完整case表 | `绘图数据/全算子_逐case绘图候选.csv` | 全表 | 1,112行，一行一个唯一正式case候选 |
| 原始逐轮耗时 | 阶段07/08的`runs/**/operator_timing_samples_*.csv` | `device,sample_index,latency_ms` | 只在复算分位数、画延迟分布或专家追溯时使用 |
| 原始内存轨迹 | 阶段07/08的`runs/**/*_memory_trace_*.csv` | `trace_phase,cpu_live_heap_bytes,rss_bytes,gpu_used_bytes` | 不能代替泄漏审计结论 |

绘图候选目录说明见`绘图数据/README.md`。

## 每个算子如何压缩15/60个case

`绘图数据/53算子_全case性能区间候选.csv`每个算子一行，所有case均参与统计：

| 字段 | 含义 |
| --- | --- |
| `case_count`、`total_case_count` | 该算子的唯一正式case数，当前普通算子多为15，七个阶段08窗函数为60，`firwin`为17 |
| `pass_count` | 同时满足功能与精度PASS的case数 |
| `median_speedup` | 全部case加速比中位数 |
| `p25_speedup`、`p75_speedup` | 中间50%的性能范围；采用排序后线性插值分位数 |
| `min_speedup`、`max_speedup` | 最差与最好case |
| `cases_below_1x` | GPU慢于CPU的case数量 |
| `worst_accuracy_ratio` | 当前为`NA_THRESHOLD_SNAPSHOT_NOT_BOUND`，表示尚未把对应阈值快照绑定进绘图表 |
| `max_mse`及`max_mse_case_id` | 该算子MSE最大值及对应case |
| `max_rmse`及`max_rmse_case_id` | 该算子RMSE最大值及对应case |
| `max_relative_l2`及对应case | 相对L2最差值及对应case |
| `max_relative_linf`及对应case | 相对L∞最差值及对应case |
| `max_gpu_p95_ms`及对应case | GPU长尾延迟最大的case |
| `max_gpu_cv`及对应case | GPU最不稳定case |
| `status` | 全量case汇总结论：`PASS/FAIL/PARTIAL` |

最佳case不能替代该表中的`median/P25/P75/min/max/below 1×`。

## 主PPT和备份页建议

### 主PPT：覆盖与精度

- 以`53x5_最差精度候选.csv`制作53×5 dtype热力图。
- 当前每个单元格分别提供MSE、RMSE、Relative L2、Relative L∞的最差值和最差case。
- 因阈值快照尚未绑定，当前应制作四张小热力图，不得强行计算统一`worst_accuracy_ratio`。
- 离散输出读取`exact_match_false_count`与`max_mismatch_count`；例如`argrelextrema`不应把数值误差`NA`解释成缺测。

### 主PPT：性能总体分布

- 使用12模块箱线图，横轴为`median_speedup`并采用对数坐标，参考线为`1×`。
- 每个算子先用自身全部case得到一个`median_speedup`，模块箱线图再使用模块内算子中位数集合。
- 禁止直接把模块内全部case混合，否则60-case窗函数会获得普通15-case算子的四倍权重。
- 同时标明中位数低于`1×`的算子数量，不隐藏GPU慢项。

### 备份页：53算子性能区间

一行一个算子：细线表示`min～max`，粗线表示`P25～P75`，圆点表示`median`，竖线表示`1×`。建议按模块排序拆成两页，并从`case_count`生成`n=15/n=60`图注。

### 窗口模块

- 主PPT仍只展示每算子统计分布。
- 备份页使用`窗函数_435case热力图候选.csv`，按`operator × dtype × scale × variant_key`展示speedup并叠加精度状态。
- `variant_key`从原始`case_id`的参数摘要和`sym/periodic`语义提取，在`operator + dtype + scale_id`内唯一，不是人工case序号。
- 210个较新的窗口case同时带有完整`case_parameters_json`；225个旧回收case未嵌入该字段。后者仍用`case_id + variant_key + input_shape`区分，但不得臆造缺失参数。
- 完整case表留在CSV或技术文档附录，不放主讲正文。

### 代表性算子

`4类代表算子候选.csv`按以下规则产生且避免重复：

1. 最高性能收益：全部case精度PASS后，`median_speedup`最高；
2. 典型对象：最接近53算子总体中位数；
3. 性能边界：低于`1×`case最多，并以中位加速比更低者优先；
4. 技术代表：`fft_thrust`算子中中位加速比较高且不与前三项重复。

代表算子图从`全算子_逐case绘图候选.csv`读取CPU/GPU mean、GPU P95、CV及四项精度。三档规模只能描述为“多规模比较”，不能外推为连续趋势。

## 最佳case只用于演示

最终演示配置应命名为`BEST_DEMO_CASE`，并同时满足功能、四项精度、资源、泄漏、speedup、GPU P95、CV和现场超时预算门禁。

`53算子_演示case性能候选.csv`目前只按“speedup高、GPU P95低、GPU CV低”完成性能初筛，其`resource_gate_status`和`leak_gate_status`明确为`NOT_EVALUATED_IN_COMPARISON_INDEX`。因此它只能用于后续检查，不能直接写入`demo_profile.json`或宣称为最终最佳演示case。

## CSV打不开的审计结论

已对阶段07/08、上层索引及绘图候选共9,126个CSV逐文件执行严格审计：

| 检查项 | 结果 |
| --- | ---: |
| UTF-8有效 | 9,126/9,126 |
| CSV语法可解析 | 9,126/9,126 |
| 各数据行列数与表头一致 | 9,126/9,126 |
| 空文件 | 0 |
| 只有表头无数据 | 0 |
| 解析失败 | 0 |
| 完整路径超过218字符 | 9,108 |
| 完整路径超过260字符 | 8,216 |
| 最长完整路径 | 342字符 |

结论：没有发现回收损坏；大量深层CSV“打不开”的主要风险是Windows/Office传统路径长度限制，而不是文件内容错误。逐文件结果见`CSV可打开性审计.csv`，35种表头结构见`CSV字段模式汇总.csv`。

正确使用方式：

1. 绘图优先打开`绘图数据/`下的短路径CSV，或阶段批次根部的`cpu_gpu_comparison.csv`；这些入口路径约116～121字符。
2. 不要从资源管理器或Excel直接双击`runs/`深层CSV。
3. 必须检查某个原始CSV时，根据`source_relative_path`定位后，把该单个文件复制到短目录再打开；复制品只用于查看，原件保持不变。
4. 也可使用支持长路径的PowerShell、编辑器或脚本读取原件。
5. Excel显示科学计数法属于显示格式，不表示原始数值丢失；绘图程序应直接按数值列解析。

## 首选数据表

| 文件 | 一行代表什么 | 主要用途 |
| --- | --- | --- |
| `cpu_gpu_comparison.csv` | 一个运行记录中某输入case的CPU/GPU成对结果 | 完整保留formal、smoke和重跑；直接读取四项精度、CPU/GPU耗时和加速比 |
| `measurement_rows.csv` | 一个输入case在一个device上的测量结果 | 分别检查CPU行、GPU行及原始统计量 |
| `operator_coverage.csv` | 一个算子的回收记录覆盖汇总 | 检查dtype、规模和形状；`case_count`含重跑记录，唯一case数应看`绘图数据/53算子_全case性能区间候选.csv` |
| `input_evidence_index.csv` | 一条dtype或输入内容证据 | 查看真实输入类型、生成方式、内容摘要和参数JSON |
| `file_catalog.csv` | 一个原始文件 | 从类型、阶段和相对路径定位原始证据 |
| `expanded_files.sha256` | 一个原始文件的SHA-256 | 校验展开后的`runs/`没有发生内容变化 |

## `cpu_gpu_comparison.csv`字段说明

### 身份与来源

| 列 | 含义与读取方法 |
| --- | --- |
| `stage` | `stage07`或`stage08` |
| `source_group` | 原始回收目录组；用于区分批次、重跑、输入预览和历史留存 |
| `source_partition` | 原路径中的`formal`、`smoke`或`other`；正式图表通常先筛`formal`，但仍需结合最终资格规则去重 |
| `source_relative_path` | 本行对应的原始`operator_main_results_*.csv`，相对当前批次README所在目录 |
| `run_id` | 唯一运行标识；答辩引用数据时必须保留 |
| `run_type` | 程序声明的`formal`或`smoke` |
| `test_time_utc` | 测试时间，UTC |
| `git_commit`、`git_dirty` | 运行时代码身份与工作区状态；dirty并不自动使证据失效 |
| `task_name`、`step_name` | 任务及步骤身份，阶段08部分算子可能为空 |

### 输入与实现类型

| 列 | 含义与读取方法 |
| --- | --- |
| `operator_name` | 算子名 |
| `case_id` | 输入case唯一标识；区分不同参数、规模、dtype和语义变体 |
| `scale_id` | 可读的规模标签 |
| `order_of_magnitude` | `10^2`、`10^3`、`10^4`等目标量级 |
| `actual_elements` | 实际参与计算的元素数，不一定严格等于量级标签 |
| `input_shape` | 实际输入形状；多输入算子用`name=[shape]|name=[shape]`表达 |
| `dtype` | 实测类型：`FP16`、`FP32`、`INT8`、`INT16`、`INT32` |
| `backend` | `fft_thrust`表示FFT/Thrust实现路径；`not_applicable`表示该算子不需要该子分类，不代表没有GPU结果 |
| `timing_scope` | 计时边界；当前常见值为`compatibility/end-to-end`，不同边界不可直接混算 |

### 耗时与稳定性

| 列 | 定义 |
| --- | --- |
| `warmup_runs` | 正式采样前预热次数，不计入统计 |
| `measured_runs` | 进入统计的重复测量次数 |
| `cpu_mean_ms`、`gpu_mean_ms` | CPU/GPU平均耗时，单位毫秒 |
| `cpu_p50_ms`、`gpu_p50_ms` | 中位耗时，50%样本不超过该值 |
| `cpu_p95_ms`、`gpu_p95_ms` | 95分位耗时，用于观察尾部波动 |
| `cpu_p99_ms`、`gpu_p99_ms` | 99分位耗时，用于观察极端尾延迟 |
| `cpu_min_ms`、`gpu_min_ms` | 最小耗时 |
| `cpu_max_ms`、`gpu_max_ms` | 最大耗时 |
| `cpu_std_ms`、`gpu_std_ms` | 耗时标准差，单位毫秒 |
| `cpu_cv`、`gpu_cv` | 变异系数，计算为`std_ms / mean_ms`；越小通常表示运行越稳定 |
| `cpu_gpu_speedup` | `CPU mean_ms / GPU mean_ms`；大于1表示GPU更快，等于1近似持平，小于1表示该case下GPU更慢 |

### 精度与语义一致性

设CPU参考结果为`y_ref`，GPU结果为`y_gpu`，差值为`e=y_gpu-y_ref`。

| 列 | 定义与判读 |
| --- | --- |
| `accuracy_reference` | 精度参考来源；当前核心结果通常为`typed_cpu_reference` |
| `mse` | `mean(abs(e)^2)`，均方误差，越小越好 |
| `rmse` | `sqrt(mse)`，与输出值同量纲，越小越好 |
| `relative_l2` | `norm(e,2) / norm(y_ref,2)`，衡量总体相对误差 |
| `relative_linf` | `max(abs(e)) / max(abs(y_ref))`，衡量最坏点相对误差 |
| `exact_match` | 离散、索引或坐标类输出是否完全一致 |
| `mismatch_count` | 不一致元素数量；精确匹配时应为0 |
| `semantic_check` | 形状、dtype、有限值、输出数量或任务调用闭包等语义门禁说明 |
| `accuracy_status` | 程序依据约定阈值或精确语义规则给出的精度结论 |

`mse/rmse/relative_l2/relative_linf=NA`不一定是缺失。对离散索引或坐标输出，应改读`exact_match`、`mismatch_count`与`semantic_check`。阶段07的15个`argrelextrema` case即采用这种精确匹配判据。

## 如何找到“不同类型输入”的证据

1. 在`operator_coverage.csv`按`operator_name`查看该算子覆盖的dtype、规模、形状及case数量。
2. 在`cpu_gpu_comparison.csv`按`operator_name + dtype + scale_id + input_shape`筛选所需输入类型，并读取同一行的四项精度、CPU/GPU耗时和加速比。
3. 需要输入内容、参数或dtype契约时，用同一个`run_id + case_id`到`input_evidence_index.csv`联查`typed_input_manifest`、`input_content_json`、`case_parameters_json`、`generator`和`preview_head4_tail2`。
4. 需要逐轮耗时而非汇总统计时，沿`source_relative_path`进入同一运行目录，读取同名case对应的`operator_timing_samples_*.csv`；其中`sample_index`和`latency_ms`是一轮一条的原始采样。
5. 需要完整审计链时，在同一运行目录继续查看`operator_*_report.json`、`operator_*_summary.txt`、`operator_*_full.log`和内存轨迹CSV。

## 使用边界

- `formal`与`smoke`均被完整保留，不能混合计算最终汇总值。
- 同一`case_id`可能存在重跑；做最终图表前应依据`run_id`、代码身份、状态和最终资格清单选择唯一结果，不能直接对全部回收行求平均。
- `cpu_gpu_speedup`必须与同一case、同一计时边界的CPU/GPU耗时一起展示。
- `status=PASS`说明运行成功，`accuracy_status=PASS`说明精度门禁通过；两者含义不同，建议同时检查。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`04_算子结果`
- 直接子目录：3个
- 直接文件：5个
- 递归文件：24955个
- 递归CSV：11672个
- README未覆盖文件：0个
- CSV字段说明未覆盖文件：0个

### 目录结构与子目录追溯

| 子目录 | 用途 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | 完整manifest | 状态 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `remaining_operators` | 该子目录的具体证据；先阅读其README。 | 19527 | 9416 | [README](./remaining_operators/README.md) | [CSV字段说明](./remaining_operators/CSV字段说明.md) | [folder manifest](./remaining_operators/folder_manifest.csv) | TRACEABLE |
| `task_operators` | 该子目录的具体证据；先阅读其README。 | 5412 | 2244 | [README](./task_operators/README.md) | [CSV字段说明](./task_operators/CSV字段说明.md) | [folder manifest](./task_operators/folder_manifest.csv) | TRACEABLE |
| `绘图数据` | 该子目录的具体证据；先阅读其README。 | 11 | 9 | [README](./绘图数据/README.md) | [CSV字段说明](./绘图数据/CSV字段说明.md) | [folder manifest](./绘图数据/folder_manifest.csv) | TRACEABLE |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | 来源与身份 | SHA-256位置 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [`CSV可打开性审计.csv`](./CSV可打开性审计.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_5e94889de383) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`CSV字段模式汇总.csv`](./CSV字段模式汇总.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_3c73f9c8b2b5) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 当前目录CSV逐文件schema映射和逐列语义说明。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 当前目录递归文件身份、来源、大小和SHA-256清单。 | [字段说明](./CSV字段说明.md#schema_7f4d2cbb2001) | 本地治理文件 | 由上级folder manifest记录 | TRACEABLE |
| [`README.md`](./README.md) | governance | 当前目录阅读入口、文件追溯与快速导航。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |

### 正确阅读顺序

1. 先读本`README.md`确认目录身份、文件用途和证据状态；
2. 遇到CSV，先读同目录`CSV字段说明.md`，按文件名定位schema，再读全部列定义；
3. 需要来源、大小和SHA-256时查`folder_manifest.csv`；
4. 需要跨目录身份时查`01_总清单与校验/`中的全局manifest和回收批次；
5. 需要原始日志、图表或文档引用时沿本README的相关文件和上级导航继续追溯。

### 快速导航

- 上级目录：[返回上级](../README.md)
- 归档根：[README](../README.md)
- 本目录CSV字段说明：[CSV字段说明.md](./CSV字段说明.md)
- 本目录完整文件清单：[folder_manifest.csv](./folder_manifest.csv)
- 全局artifact manifest：[../01_总清单与校验/artifact_manifest.csv](../01_总清单与校验/artifact_manifest.csv)
- 全局SHA-256总表：[../01_总清单与校验/sha256_manifest.csv](../01_总清单与校验/sha256_manifest.csv)
- 回收批次：[../01_总清单与校验/recovery_batches.csv](../01_总清单与校验/recovery_batches.csv)
- 当前缺失项：[../01_总清单与校验/missing_artifacts.csv](../01_总清单与校验/missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
