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
| [`operator_bsplines_quadratic_memory_trace_not_applicable.csv`](./operator_bsplines_quadratic_memory_trace_not_applicable.csv) | `SCHEMA_620652473711` | [查看全部列](#schema_620652473711) | `620652473711789786d7516c8510d5c211b0f076783c0b258e367e2dbfa0ff52` | 30 | 一行对应一个资源轨迹采样点。 | `run_id + case_id + device` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_dtype_evidence_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv`](./operator_dtype_evidence_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv) | `SCHEMA_F790A782CFD0` | [查看全部列](#schema_f790a782cfd0) | `f790a782cfd084222f86da75ee259879c98f7bc4f74440212743eacae6bdc190` | 15 | 一行对应一个case记录。 | `run_id + case_id + operator_name` | `run_id + case_id` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_main_results_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv`](./operator_main_results_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv) | `SCHEMA_C185CB915CF7` | [查看全部列](#schema_c185cb915cf7) | `c185cb915cf74184eb0947423159e204169b4a0392519061701fc6861eddb032` | 66 | 一行对应一个输入case在一个device上的汇总测量。 | `run_id + case_id + device` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_timing_samples_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv`](./operator_timing_samples_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv) | `SCHEMA_8C46C287C375` | [查看全部列](#schema_8c46c287c375) | `8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca` | 31 | 一行对应一个device的一次原始计时样本。 | `run_id + case_id + device` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_620652473711"></a>
### SCHEMA_620652473711

- 适用文件：`operator_bsplines_quadratic_memory_trace_not_applicable.csv`
- 一行代表：一行对应一个资源轨迹采样点。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bdef73fce32b9234cec46609c420421a2b5cf2f6` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b08_quadratic_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `5c359dbfe6c22baef56e207c32f664cf92cec0bc796fb03500ef19b5c00b712f` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic__quadratic_int8_10e3__1948b336418b2c17` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic_int8_10e3` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^3` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1024` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[1024]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INT8` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `sample_index` | integer | count | 样本未见空值 | 逐轮测量样本序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 26 | `trace_phase` | string | NA | 样本未见空值 | 资源轨迹采样阶段。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `before` | NA |
| 27 | `phase_index` | integer | count | 样本未见空值 | 资源轨迹阶段内的序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 28 | `cpu_live_heap_bytes` | integer | bytes | 样本未见空值 | CPU当前存活堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1345280` | NA |
| 29 | `rss_bytes` | integer | bytes | 样本未见空值 | 进程常驻内存集大小，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `68947968` | NA |
| 30 | `gpu_used_bytes` | integer | bytes | 允许 | GPU当前已使用显存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `391544832` | NA |

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

<a id="schema_8c46c287c375"></a>
### SCHEMA_8C46C287C375

- 适用文件：`operator_timing_samples_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv`
- 一行代表：一行对应一个device的一次原始计时样本。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bdef73fce32b9234cec46609c420421a2b5cf2f6` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b08_quadratic_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `5c359dbfe6c22baef56e207c32f664cf92cec0bc796fb03500ef19b5c00b712f` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic__quadratic_int8_10e3__1948b336418b2c17` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic_int8_10e3` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^3` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1024` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[1024]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INT8` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 26 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 27 | `sample_index` | integer | count | 样本未见空值 | 逐轮测量样本序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 28 | `latency_ms` | float | ms | 样本未见空值 | 单轮延迟，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.023651` | warmup_runs, measured_runs, timing_scope |
| 29 | `synchronized` | boolean | NA | 样本未见空值 | 计时前后是否执行了设备同步。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `true` | NA |
| 30 | `input_digest` | string | NA | 样本未见空值 | 本次运行实际输入内容的摘要，用于CPU/GPU配对。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `1948b336418b2c1776d3f4d319a4cf5aa4689f866ae5891ec9417516e733f285` | NA |
| 31 | `output_digest` | string | NA | 样本未见空值 | 本次运行输出内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef` | NA |

<a id="schema_c185cb915cf7"></a>
### SCHEMA_C185CB915CF7

- 适用文件：`operator_main_results_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv`
- 一行代表：一行对应一个输入case在一个device上的汇总测量。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bdef73fce32b9234cec46609c420421a2b5cf2f6` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b08_quadratic_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `5c359dbfe6c22baef56e207c32f664cf92cec0bc796fb03500ef19b5c00b712f` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic__quadratic_int8_10e3__1948b336418b2c17` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic_int8_10e3` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^3` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1024` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[1024]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INT8` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 26 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 27 | `mean_ms` | float | ms | 样本未见空值 | 耗时算术平均值，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.02354749` | warmup_runs, measured_runs, timing_scope |
| 28 | `p50_ms` | float | ms | 样本未见空值 | 耗时第50百分位，即中位数，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.023542` | warmup_runs, measured_runs, timing_scope |
| 29 | `p95_ms` | float | ms | 样本未见空值 | 耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.023601` | warmup_runs, measured_runs, timing_scope |
| 30 | `p99_ms` | float | ms | 样本未见空值 | 耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.023641099999999998` | warmup_runs, measured_runs, timing_scope |
| 31 | `min_ms` | float | ms | 样本未见空值 | 最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.023461` | warmup_runs, measured_runs, timing_scope |
| 32 | `max_ms` | float | ms | 样本未见空值 | 最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.023651` | warmup_runs, measured_runs, timing_scope |
| 33 | `std_ms` | float | ms | 样本未见空值 | 耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `3.394068207918033e-05` | warmup_runs, measured_runs, timing_scope |
| 34 | `cv` | float | dimensionless | 样本未见空值 | 耗时变异系数，计算为std_ms/mean_ms，无量纲。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0014413715465716444` | NA |
| 35 | `cpu_gpu_speedup` | float | dimensionless | 允许 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.11742697893903414` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 36 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_quadratic_same_dtype` | NA |
| 37 | `mse` | integer | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 38 | `rmse` | integer | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 39 | `relative_l2` | integer | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 40 | `relative_linf` | integer | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 41 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 42 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 43 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `output_count=1024;output_dtype=INT8;order=-1` | NA |
| 44 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 45 | `cpu_heap_before_bytes` | integer | bytes | 样本未见空值 | 运行前CPU堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1345280` | NA |
| 46 | `cpu_heap_after_bytes` | integer | bytes | 样本未见空值 | 运行后CPU堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1357824` | NA |
| 47 | `cpu_heap_peak_bytes` | integer | bytes | 样本未见空值 | 运行期间CPU堆内存峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1357824` | NA |
| 48 | `cpu_heap_delta_bytes` | integer | bytes | 样本未见空值 | 运行后减运行前的CPU堆内存变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `12544` | NA |
| 49 | `rss_before_bytes` | integer | bytes | 样本未见空值 | 运行前RSS，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `68947968` | NA |
| 50 | `rss_after_bytes` | integer | bytes | 样本未见空值 | 运行后RSS，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `68947968` | NA |
| 51 | `rss_peak_bytes` | integer | bytes | 样本未见空值 | 运行期间RSS峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `68947968` | NA |
| 52 | `rss_delta_bytes` | integer | bytes | 样本未见空值 | 运行后减运行前的RSS变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `0` | NA |
| 53 | `gpu_before_bytes` | integer | bytes | 允许 | 运行前GPU显存占用，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `391544832` | NA |
| 54 | `gpu_after_bytes` | integer | bytes | 允许 | 运行后GPU显存占用，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `391544832` | NA |
| 55 | `gpu_peak_bytes` | integer | bytes | 允许 | 运行期间GPU显存峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `391544832` | NA |
| 56 | `gpu_delta_bytes` | integer | bytes | 允许 | 运行后减运行前的GPU显存变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `0` | NA |
| 57 | `stability_case_index` | string | count | 允许 | 稳定性扰动case序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 58 | `stability_total_cases` | string | NA | 允许 | 本组稳定性扰动case总数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 59 | `perturbation_file` | string | NA | 允许 | 稳定性扰动配置或输入文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 60 | `perturbation_sha256` | string | NA | 允许 | 稳定性扰动文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 61 | `perturbation_summary` | string | NA | 允许 | 稳定性扰动参数摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 62 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 63 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 64 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 65 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 66 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_f790a782cfd0"></a>
### SCHEMA_F790A782CFD0

- 适用文件：`operator_dtype_evidence_quadratic_not_applicable_20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T091357Z_bdef73fce32b_stage08_b08_formal_8772000d` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic__quadratic_int8_10e3__1948b336418b2c17` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `quadratic` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INT8` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_typed_x_native_same_dtype_output` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x:INT8:[1024]:bytes=1024:sha256=70a1fd085a62b6c8ea98365c2e70579a8b8fc6f8501fb...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `1948b336418b2c1776d3f4d319a4cf5aa4689f866ae5891ec9417516e733f285` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INT8` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INT8` | NA |
| 12 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 13 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 14 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"INT8","element_count":"1024","generator":"bspline_modular_or_piece...` | NA |
| 15 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"dtype":"INT8","elements":"1024","operator":"quadratic","order":"-1","seed":...` | NA |

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../../../../../../../README.md)
- [全局CSV字段语义总表](../../../../../../../../00_归档规范/CSV字段语义总表.md)
