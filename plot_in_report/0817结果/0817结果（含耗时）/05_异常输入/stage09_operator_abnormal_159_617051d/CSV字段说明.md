# CSV字段说明

> 阅读顺序：先读同目录`README.md`了解文件身份和用途，再用本文件查CSV行粒度、主键、关联键和全部列含义。

## 覆盖结论

- 本目录直接CSV：5个（含治理文件`folder_manifest.csv`）
- 本目录直接CSV已映射：5个
- 本目录直接CSV未映射：0个
- 本目录不同schema：5种
- 生成与校验时间：`2026-08-14T17:33:09+08:00`
- 校验规则：真实列名、列数、顺序及规范化表头SHA-256必须与schema一致。

## 直接CSV到schema映射

| CSV文件 | schema_id | 字段说明 | 表头SHA-256 | 列数 | 一行代表 | 主键 | 配对键 | 来源/生成方式 | 校验状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| [`abnormal_type_summary.csv`](./abnormal_type_summary.csv) | `SCHEMA_25DBF2F4D43D` | [查看全部列](#schema_25dbf2f4d43d) | `25dbf2f4d43d9d469418445bd1116060ea5f4bda403701c950a32a52aedd0e1e` | 7 | 一行对应一个异常输入case及其预期和实际处理结果。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`all_operator_abnormal_cases.csv`](./all_operator_abnormal_cases.csv) | `SCHEMA_BF0AE92FC547` | [查看全部列](#schema_bf0ae92fc547) | `bf0ae92fc547ff3d064ccddc056dd349a8cf3e9752419e726c46d281c9ac69dc` | 24 | 一行对应一个异常输入case及其预期和实际处理结果。 | `run_id + case_id + operator_name` | `run_id + case_id` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`file_catalog.csv`](./file_catalog.csv) | `SCHEMA_9C155BB8C8B9` | [查看全部列](#schema_9c155bb8c8b9) | `9c155bb8c8b96a33b4b49a3f282e0c5179fe94a52313ccb3abe54efb5c9bf960` | 4 | 一行对应一个文件或证据条目。 | `relative_path` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_abnormal_summary.csv`](./operator_abnormal_summary.csv) | `SCHEMA_4AD8B3491740` | [查看全部列](#schema_4ad8b3491740) | `4ad8b34917403c66e9f4b214fffecb57bf29d888a768f58b7a2a82712f693f21` | 12 | 一行对应一个异常输入case及其预期和实际处理结果。 | `未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。` | `NA` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明 | 已说明 | 未说明 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| [`documentation`](./documentation/README.md) | 1 | [CSV字段说明](./documentation/CSV字段说明.md) | 1 | 0 | COVERED |
| [`final_v1`](./final_v1/README.md) | 4 | [CSV字段说明](./final_v1/CSV字段说明.md) | 4 | 0 | COVERED |
| [`raw`](./raw/README.md) | 107 | [CSV字段说明](./raw/CSV字段说明.md) | 107 | 0 | COVERED |

## schema完整定义

<a id="schema_25dbf2f4d43d"></a>
### SCHEMA_25DBF2F4D43D

- 适用文件：`abnormal_type_summary.csv`
- 一行代表：一行对应一个异常输入case及其预期和实际处理结果。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `abnormal_type` | string | NA | 样本未见空值 | 单条异常输入的语义类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `backend_call_failure` | NA |
| 2 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |
| 3 | `expected_codes` | string | NA | 样本未见空值 | `expected codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `BACKEND_ERROR\|CUDA_RUNTIME_ERROR` | NA |
| 4 | `actual_codes` | string | NA | 样本未见空值 | `actual codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `BACKEND_ERROR\|CUDA_RUNTIME_ERROR` | NA |
| 5 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |
| 6 | `safe_exit_count` | integer | count | 样本未见空值 | `safe exit`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |
| 7 | `output_not_consumed_count` | integer | count | 样本未见空值 | `output not consumed`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |

<a id="schema_4ad8b3491740"></a>
### SCHEMA_4AD8B3491740

- 适用文件：`operator_abnormal_summary.csv`
- 一行代表：一行对应一个异常输入case及其预期和实际处理结果。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `module_name` | string | NA | 样本未见空值 | 算子所属模块名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 2 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 3 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 4 | `abnormal_types` | string | NA | 样本未见空值 | 一组记录覆盖的异常语义类别集合。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `backend_call_failure\|invalid_dtype\|invalid_shape` | NA |
| 5 | `backends` | string | NA | 样本未见空值 | 当前汇总对象覆盖的后端集合，通常使用竖线分隔。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cuda_runtime\|not_applicable` | NA |
| 6 | `expected_codes` | string | NA | 样本未见空值 | `expected codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR\|INVALID_DTYPE\|INVALID_SHAPE` | NA |
| 7 | `actual_codes` | string | NA | 样本未见空值 | `actual codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR\|INVALID_DTYPE\|INVALID_SHAPE` | NA |
| 8 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 9 | `safe_exit_count` | integer | count | 样本未见空值 | `safe exit`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 10 | `output_not_consumed_count` | integer | count | 样本未见空值 | `output not consumed`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 11 | `normal_run_ids` | datetime | NA | 样本未见空值 | `normal run ids`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996` | NA |
| 12 | `normal_case_ids` | string | NA | 样本未见空值 | `normal case ids`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |

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

<a id="schema_9c155bb8c8b9"></a>
### SCHEMA_9C155BB8C8B9

- 适用文件：`file_catalog.csv`
- 一行代表：一行对应一个文件或证据条目。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`relative_path`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `semantic_summary` | NA |
| 2 | `extension` | string | NA | 样本未见空值 | 文件扩展名，用于文件类型统计和筛选。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `.csv` | NA |
| 3 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `540` | NA |
| 4 | `relative_path` | string | NA | 样本未见空值 | 相对于当前目录或表格约定根目录的文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `abnormal_type_summary.csv` | NA |

<a id="schema_bf0ae92fc547"></a>
### SCHEMA_BF0AE92FC547

- 适用文件：`all_operator_abnormal_cases.csv`
- 一行代表：一行对应一个异常输入case及其预期和实际处理结果。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `timestamp` | datetime | datetime | 样本未见空值 | 该记录对应的时间戳；格式和时区以同表说明为准。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2026-08-11T11:35:00Z` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260811T113500Z_617051d_stage09_cubic_00000001` | NA |
| 3 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cuda_runtime` | NA |
| 4 | `test_object` | string | NA | 样本未见空值 | 测试对象的统一名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 5 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 6 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 7 | `module_name` | string | NA | 样本未见空值 | 算子所属模块名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 8 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 9 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines.cubic.backend_call_failure.inject_cuda_kernel_launch_failure_before_...` | NA |
| 10 | `normal_run_id` | datetime | NA | 样本未见空值 | 用于关联异常case的正常基准run ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996` | NA |
| 11 | `normal_case_id` | string | NA | 样本未见空值 | 用于关联异常case的正常基准case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 12 | `abnormal_type` | string | NA | 样本未见空值 | 单条异常输入的语义类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `backend_call_failure` | NA |
| 13 | `input_summary` | string | NA | 样本未见空值 | 实际输入shape、dtype、参数或内容的可读摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `mutation=inject CUDA kernel launch failure before output consumption` | NA |
| 14 | `expected_code` | string | NA | 样本未见空值 | 异常case预期返回的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 15 | `captured_code` | string | NA | 样本未见空值 | 测试框架捕获到的返回码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 16 | `returned_code` | string | NA | 样本未见空值 | 程序实际返回码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 17 | `actual_code` | string | NA | 样本未见空值 | 异常case实际返回的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 18 | `error_message` | string | NA | 样本未见空值 | 错误码对应的可读错误说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines::cubic捕获CUDA Runtime失败：mutation=inject CUDA kernel launch failure be...` | NA |
| 19 | `backend_error_code` | integer | NA | 允许 | `backend error code`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `101` | NA |
| 20 | `backend_error_name` | string | NA | 允许 | 后端错误码对应的符号名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cudaErrorInvalidDevice` | NA |
| 21 | `backend_error_message` | string | NA | 允许 | `backend error`对应的可读说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `invalid device ordinal` | NA |
| 22 | `safe_exit` | boolean | NA | 样本未见空值 | 异常输入是否按设计安全退出。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `true` | NA |
| 23 | `output_consumed` | boolean | NA | 样本未见空值 | 异常或失败输出是否被后续流程消费的布尔标记。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 24 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../README.md)
- [全局CSV字段语义总表](../../00_归档规范/CSV字段语义总表.md)
