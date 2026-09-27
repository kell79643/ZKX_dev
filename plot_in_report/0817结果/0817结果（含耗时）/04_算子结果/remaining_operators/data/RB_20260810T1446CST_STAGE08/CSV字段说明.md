# CSV字段说明

> 阅读顺序：先读同目录`README.md`了解文件身份和用途，再用本文件查CSV行粒度、主键、关联键和全部列含义。

## 覆盖结论

- 本目录直接CSV：6个（含治理文件`folder_manifest.csv`）
- 本目录直接CSV已映射：6个
- 本目录直接CSV未映射：0个
- 本目录不同schema：6种
- 生成与校验时间：`2026-08-14T17:33:09+08:00`
- 校验规则：真实列名、列数、顺序及规范化表头SHA-256必须与schema一致。

## 直接CSV到schema映射

| CSV文件 | schema_id | 字段说明 | 表头SHA-256 | 列数 | 一行代表 | 主键 | 配对键 | 来源/生成方式 | 校验状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| [`cpu_gpu_comparison.csv`](./cpu_gpu_comparison.csv) | `SCHEMA_B8CA1C80348B` | [查看全部列](#schema_b8ca1c80348b) | `b8ca1c80348b336fc010c0d8485e6df1284dd375bc1433d63b92e45fb739c4b9` | 56 | 一行对应一个输入case的CPU/GPU成对比较。 | `stage + operator_name + case_id` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`file_catalog.csv`](./file_catalog.csv) | `SCHEMA_7FA1B27BF151` | [查看全部列](#schema_7fa1b27bf151) | `7fa1b27bf151f99c8bdf772dc8bbc95e5637335a280c0ef46dc349fb0608c4ff` | 7 | 一行对应一个文件或证据条目。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`input_evidence_index.csv`](./input_evidence_index.csv) | `SCHEMA_C6D6D1C6C971` | [查看全部列](#schema_c6d6d1c6c971) | `c6d6d1c6c971b4d98810266a230fcc7d5ad5239aa4a5ca41b57720fd3c3d217c` | 30 | 一行对应一个索引对象及其来源关系。 | `stage + operator_name + case_id` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`measurement_rows.csv`](./measurement_rows.csv) | `SCHEMA_2A341875B4AB` | [查看全部列](#schema_2a341875b4ab) | `2a341875b4ab86db2810425d2f615587fda4ea9bc41b773c797bb97cdaf4503f` | 49 | 一行对应一个case记录。 | `stage + operator_name + case_id` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_coverage.csv`](./operator_coverage.csv) | `SCHEMA_E7EC8ACA9A36` | [查看全部列](#schema_e7ec8aca9a36) | `e7ec8aca9a36dc28c5381a2550b47233673ebc6eafdca234dbc72ca7245bdd7d` | 15 | 一行对应一个测试对象的覆盖汇总。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明 | 已说明 | 未说明 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| [`runs`](./runs/README.md) | 9408 | [CSV字段说明](./runs/CSV字段说明.md) | 9408 | 0 | COVERED |

## schema完整定义

<a id="schema_2a341875b4ab"></a>
### SCHEMA_2A341875B4AB

- 适用文件：`measurement_rows.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_7200...` | NA |
| 5 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 6 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 7 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 8 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 9 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 10 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 26 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 27 | `mean_ms` | float | ms | 样本未见空值 | 耗时算术平均值，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7422645300000004` | warmup_runs, measured_runs, timing_scope |
| 28 | `p50_ms` | float | ms | 样本未见空值 | 耗时第50百分位，即中位数，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7423929999999999` | warmup_runs, measured_runs, timing_scope |
| 29 | `p95_ms` | float | ms | 样本未见空值 | 耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7533555000000001` | warmup_runs, measured_runs, timing_scope |
| 30 | `p99_ms` | float | ms | 样本未见空值 | 耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7555925000000001` | warmup_runs, measured_runs, timing_scope |
| 31 | `min_ms` | float | ms | 样本未见空值 | 最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7322329999999999` | warmup_runs, measured_runs, timing_scope |
| 32 | `max_ms` | float | ms | 样本未见空值 | 最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7585130000000002` | warmup_runs, measured_runs, timing_scope |
| 33 | `std_ms` | float | ms | 样本未见空值 | 耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0063154560871800988` | warmup_runs, measured_runs, timing_scope |
| 34 | `cv` | float | dimensionless | 样本未见空值 | 耗时变异系数，计算为std_ms/mean_ms，无量纲。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0023030076121723013` | NA |
| 35 | `cpu_gpu_speedup` | float | dimensionless | 允许 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `1.9409993884459864` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 36 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_typed_reference` | NA |
| 37 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 38 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 39 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 40 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 41 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 42 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 43 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `finite_fp64_window_shape_and_symmetry_policy_match_cpu:shape=[100];finite=tru...` | NA |
| 44 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 45 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 46 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 47 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 48 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 49 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

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

<a id="schema_7fa1b27bf151"></a>
### SCHEMA_7FA1B27BF151

- 适用文件：`file_catalog.csv`
- 一行代表：一行对应一个文件或证据条目。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_input_contract_audit_59e24c653eae.txt` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `other` | NA |
| 4 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `other` | NA |
| 5 | `extension` | string | NA | 样本未见空值 | 文件扩展名，用于文件类型统计和筛选。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `.txt` | NA |
| 6 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2165` | NA |
| 7 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_input_contract_audit_59e24c653eae.txt` | NA |

<a id="schema_b8ca1c80348b"></a>
### SCHEMA_B8CA1C80348B

- 适用文件：`cpu_gpu_comparison.csv`
- 一行代表：一行对应一个输入case的CPU/GPU成对比较。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_7200...` | NA |
| 5 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 6 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 7 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 8 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 9 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 10 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 11 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 13 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 14 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 15 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 16 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 17 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 18 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 19 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 20 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 21 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 22 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 23 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 24 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7422645300000004` | warmup_runs, measured_runs, timing_scope |
| 25 | `cpu_p50_ms` | float | ms | 样本未见空值 | CPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7423929999999999` | warmup_runs, measured_runs, timing_scope |
| 26 | `cpu_p95_ms` | float | ms | 样本未见空值 | CPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7533555000000001` | warmup_runs, measured_runs, timing_scope |
| 27 | `cpu_p99_ms` | float | ms | 样本未见空值 | CPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7555925000000001` | warmup_runs, measured_runs, timing_scope |
| 28 | `cpu_min_ms` | float | ms | 样本未见空值 | CPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7322329999999999` | warmup_runs, measured_runs, timing_scope |
| 29 | `cpu_max_ms` | float | ms | 样本未见空值 | CPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7585130000000002` | warmup_runs, measured_runs, timing_scope |
| 30 | `cpu_std_ms` | float | ms | 样本未见空值 | CPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0063154560871800988` | warmup_runs, measured_runs, timing_scope |
| 31 | `cpu_cv` | float | dimensionless | 样本未见空值 | CPU耗时变异系数，计算为cpu_std_ms/cpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0023030076121723013` | NA |
| 32 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.4128106099999997` | warmup_runs, measured_runs, timing_scope |
| 33 | `gpu_p50_ms` | float | ms | 样本未见空值 | GPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.3976614999999999` | warmup_runs, measured_runs, timing_scope |
| 34 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.462828` | warmup_runs, measured_runs, timing_scope |
| 35 | `gpu_p99_ms` | float | ms | 样本未见空值 | GPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.5004397900000028` | warmup_runs, measured_runs, timing_scope |
| 36 | `gpu_min_ms` | float | ms | 样本未见空值 | GPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.3832359999999999` | warmup_runs, measured_runs, timing_scope |
| 37 | `gpu_max_ms` | float | ms | 样本未见空值 | GPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.0542250000000002` | warmup_runs, measured_runs, timing_scope |
| 38 | `gpu_std_ms` | float | ms | 样本未见空值 | GPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.06816031129754252` | warmup_runs, measured_runs, timing_scope |
| 39 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.048244478640730573` | NA |
| 40 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `1.9409993884459864` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 41 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_typed_reference` | NA |
| 42 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.1434175650342245e-12` | accuracy_reference, accuracy_status |
| 43 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `1.4640415175240845e-06` | accuracy_reference, accuracy_status |
| 44 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.6867182611919047e-06` | accuracy_reference, accuracy_status |
| 45 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.7738614958593431e-06` | accuracy_reference, accuracy_status |
| 46 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 47 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 48 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `finite_fp64_window_shape_and_symmetry_policy_match_cpu:shape=[100];finite=tru...` | NA |
| 49 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 50 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 51 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 52 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 53 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 54 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 55 | `cpu_row_present` | boolean | NA | 样本未见空值 | `cpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |
| 56 | `gpu_row_present` | boolean | NA | 样本未见空值 | `gpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |

<a id="schema_c6d6d1c6c971"></a>
### SCHEMA_C6D6D1C6C971

- 适用文件：`input_evidence_index.csv`
- 一行代表：一行对应一个索引对象及其来源关系。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b02_strict_dtype` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `evidence_type` | string | NA | 样本未见空值 | 输入、dtype、运行、日志或其他证据的类型标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `dtype_and_input_contract` | NA |
| 5 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b02_strict_dtype/formal/20260807T161413Z_1c045bd7e750_stage08_b0...` | NA |
| 6 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000100` | NA |
| 7 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `parzen__parzen_fp32_10e2__7f7d081aa0af5dfa__sym` | NA |
| 8 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `parzen` | NA |
| 9 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `config_input_dtype` | string | NA | 允许 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 11 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `request_variant_no_typed_data_input` | NA |
| 12 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 13 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `length:INT32=100\|sym:BOOL=true` | NA |
| 14 | `typed_array_digest` | string | NA | 允许 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 15 | `input_content_json` | string | NA | 允许 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 16 | `case_parameters_json` | string | NA | 允许 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 17 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `7f7d081aa0af5dfac9cbd2114e6e1ae81fa1069a67d6310ffd6db02c7f813864` | NA |
| 18 | `compute_dtype` | string | NA | 允许 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 19 | `execution_dtype` | string | NA | 允许 | 执行路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 20 | `fft_dtype` | string | NA | 允许 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 21 | `output_dtype` | string | NA | 允许 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 22 | `backend` | string | NA | 允许 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 23 | `input_name` | string | NA | 允许 | 一个输入参数或输入数组的名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 24 | `shape` | string | NA | 允许 | 输入或输出数组的形状。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 25 | `dtype` | string | NA | 允许 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 26 | `element_count` | string | count | 允许 | 数组中的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 27 | `generator` | string | NA | 允许 | 输入数据生成器或生成规则。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 28 | `preview_head4_tail2` | string | NA | 允许 | 大型输入前4项与后2项的可读预览。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 29 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 30 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_e7ec8aca9a36"></a>
### SCHEMA_E7EC8ACA9A36

- 适用文件：`operator_coverage.csv`
- 一行代表：一行对应一个测试对象的覆盖汇总。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly` | NA |
| 3 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 4 | `formal_case_count` | integer | count | 样本未见空值 | `formal case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `46` | NA |
| 5 | `smoke_case_count` | integer | count | 允许 | `smoke case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 6 | `dtypes` | string | NA | 样本未见空值 | `dtypes`对应的数据类型或数据类型集合。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16\|FP32\|INT16\|INT32\|INT8` | NA |
| 7 | `backends` | string | NA | 样本未见空值 | 当前汇总对象覆盖的后端集合，通常使用竖线分隔。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 8 | `scale_ids` | string | NA | 样本未见空值 | `scale ids`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly_fp16_10e2\|channelize_poly_fp16_10e3\|channelize_poly_fp16_10e4...` | NA |
| 9 | `input_shapes` | string | NA | 样本未见空值 | `input shapes`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[1024];h=[64]\|x=[128];h=[32]\|x=[16384];h=[256]` | NA |
| 10 | `min_actual_elements` | integer | NA | 样本未见空值 | 当前聚合范围内`actual elements`的最小值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `160` | NA |
| 11 | `max_actual_elements` | integer | NA | 样本未见空值 | 当前聚合范围内`actual elements`的最大值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `16640` | NA |
| 12 | `status_pass_count` | integer | count | 样本未见空值 | `status pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 13 | `accuracy_pass_count` | integer | count | 样本未见空值 | `accuracy pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 14 | `numeric_error_case_count` | integer | count | 样本未见空值 | `numeric error case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 15 | `exact_match_case_count` | integer | count | 样本未见空值 | `exact match case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../../../README.md)
- [全局CSV字段语义总表](../../../../00_归档规范/CSV字段语义总表.md)
