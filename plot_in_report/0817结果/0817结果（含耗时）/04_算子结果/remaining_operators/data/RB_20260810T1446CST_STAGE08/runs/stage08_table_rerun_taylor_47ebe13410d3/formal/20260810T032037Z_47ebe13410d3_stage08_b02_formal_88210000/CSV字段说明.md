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
| [`operator_dtype_evidence_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`](./operator_dtype_evidence_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv) | `SCHEMA_298670123E40` | [查看全部列](#schema_298670123e40) | `298670123e40f5996033a2f9e3550d329b9af298fca89fe6a00bc70c647b91dc` | 12 | 一行对应一个case记录。 | `run_id + case_id + operator_name` | `run_id + case_id` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_main_results_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`](./operator_main_results_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv) | `SCHEMA_C185CB915CF7` | [查看全部列](#schema_c185cb915cf7) | `c185cb915cf74184eb0947423159e204169b4a0392519061701fc6861eddb032` | 66 | 一行对应一个输入case在一个device上的汇总测量。 | `run_id + case_id + device` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_timing_samples_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`](./operator_timing_samples_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv) | `SCHEMA_8C46C287C375` | [查看全部列](#schema_8c46c287c375) | `8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca` | 31 | 一行对应一个device的一次原始计时样本。 | `run_id + case_id + device` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`operator_windows_taylor_memory_trace_not_applicable.csv`](./operator_windows_taylor_memory_trace_not_applicable.csv) | `SCHEMA_620652473711` | [查看全部列](#schema_620652473711) | `620652473711789786d7516c8510d5c211b0f076783c0b258e367e2dbfa0ff52` | 30 | 一行对应一个资源轨迹采样点。 | `run_id + case_id + device` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`windows_batch02_module_summary_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`](./windows_batch02_module_summary_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv) | `SCHEMA_CB6B6E50B568` | [查看全部列](#schema_cb6b6e50b568) | `cb6b6e50b568088f256461889b29645473a3344adbb0bb9762a1784945926e94` | 20 | 一行对应一个汇总对象。 | `run_id + case_id + operator_name` | `operator_name + case_id + dtype + backend` | 原始或既有派生CSV；详见README和folder manifest | HEADER_MATCH |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_298670123e40"></a>
### SCHEMA_298670123E40

- 适用文件：`operator_dtype_evidence_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor__taylor_fp32_10e2__3137a27866a19941__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_typed_scalar_input` | NA |
| 7 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 8 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `length:INT32=100\|sym:BOOL=true\|sidelobe_level_db:FP32:bytes=0000f041` | NA |
| 9 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `3137a27866a19941acdc1cc33f184a349cd02d72175ccf8b3e87aca7a431cb7d` | NA |
| 10 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP64` | NA |
| 11 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 12 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_620652473711"></a>
### SCHEMA_620652473711

- 适用文件：`operator_windows_taylor_memory_trace_not_applicable.csv`
- 一行代表：一行对应一个资源轨迹采样点。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47ebe13410d3a97ff7b922a5622a21bff244b00a` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `true` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b02_taylor_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `1e7a268332169dbf26515052ca42f187fc360013dd724c8f10fd2a57331a5876` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor__taylor_fp32_10e2__3137a27866a19941__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor_fp32_10e2` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `sample_index` | integer | count | 样本未见空值 | 逐轮测量样本序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 26 | `trace_phase` | string | NA | 样本未见空值 | 资源轨迹采样阶段。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `before` | NA |
| 27 | `phase_index` | integer | count | 样本未见空值 | 资源轨迹阶段内的序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 28 | `cpu_live_heap_bytes` | integer | bytes | 样本未见空值 | CPU当前存活堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1446368` | NA |
| 29 | `rss_bytes` | integer | bytes | 样本未见空值 | 进程常驻内存集大小，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70500352` | NA |
| 30 | `gpu_used_bytes` | integer | bytes | 允许 | GPU当前已使用显存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |

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

- 适用文件：`operator_timing_samples_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`
- 一行代表：一行对应一个device的一次原始计时样本。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47ebe13410d3a97ff7b922a5622a21bff244b00a` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `true` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b02_taylor_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `1e7a268332169dbf26515052ca42f187fc360013dd724c8f10fd2a57331a5876` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor__taylor_fp32_10e2__3137a27866a19941__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor_fp32_10e2` | NA |
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
| 27 | `sample_index` | integer | count | 样本未见空值 | 逐轮测量样本序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 28 | `latency_ms` | float | ms | 样本未见空值 | 单轮延迟，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.01934` | warmup_runs, measured_runs, timing_scope |
| 29 | `synchronized` | boolean | NA | 样本未见空值 | 计时前后是否执行了设备同步。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `true` | NA |
| 30 | `input_digest` | string | NA | 样本未见空值 | 本次运行实际输入内容的摘要，用于CPU/GPU配对。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `3137a27866a19941acdc1cc33f184a349cd02d72175ccf8b3e87aca7a431cb7d` | NA |
| 31 | `output_digest` | string | NA | 样本未见空值 | 本次运行输出内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `f9df0d2136e1a4d024714d43cf01ab65d2173fe59184071c0f03b6162fd6cc0d` | NA |

<a id="schema_c185cb915cf7"></a>
### SCHEMA_C185CB915CF7

- 适用文件：`operator_main_results_taylor_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`
- 一行代表：一行对应一个输入case在一个device上的汇总测量。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47ebe13410d3a97ff7b922a5622a21bff244b00a` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `true` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b02_taylor_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `1e7a268332169dbf26515052ca42f187fc360013dd724c8f10fd2a57331a5876` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor__taylor_fp32_10e2__3137a27866a19941__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor_fp32_10e2` | NA |
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
| 27 | `mean_ms` | float | ms | 样本未见空值 | 耗时算术平均值，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.019180259999999994` | warmup_runs, measured_runs, timing_scope |
| 28 | `p50_ms` | float | ms | 样本未见空值 | 耗时第50百分位，即中位数，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.01908` | warmup_runs, measured_runs, timing_scope |
| 29 | `p95_ms` | float | ms | 样本未见空值 | 耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.01916` | warmup_runs, measured_runs, timing_scope |
| 30 | `p99_ms` | float | ms | 样本未见空值 | 耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.019433300000000046` | warmup_runs, measured_runs, timing_scope |
| 31 | `min_ms` | float | ms | 样本未见空值 | 最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.019030999999999999` | warmup_runs, measured_runs, timing_scope |
| 32 | `max_ms` | float | ms | 样本未见空值 | 最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.028670000000000001` | warmup_runs, measured_runs, timing_scope |
| 33 | `std_ms` | float | ms | 样本未见空值 | 耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.00095467978526833815` | warmup_runs, measured_runs, timing_scope |
| 34 | `cv` | float | dimensionless | 样本未见空值 | 耗时变异系数，计算为std_ms/mean_ms，无量纲。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.049774079458168891` | NA |
| 35 | `cpu_gpu_speedup` | float | dimensionless | 允许 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.048550067337625562` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 36 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_typed_reference` | NA |
| 37 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 38 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 39 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 40 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 41 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 42 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 43 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `finite_normalized_fp64_taylor_window_nbar_sll_and_symmetry_match_cpu:shape=[1...` | NA |
| 44 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 45 | `cpu_heap_before_bytes` | integer | bytes | 样本未见空值 | 运行前CPU堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1446368` | NA |
| 46 | `cpu_heap_after_bytes` | integer | bytes | 样本未见空值 | 运行后CPU堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1450224` | NA |
| 47 | `cpu_heap_peak_bytes` | integer | bytes | 样本未见空值 | 运行期间CPU堆内存峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1450224` | NA |
| 48 | `cpu_heap_delta_bytes` | integer | bytes | 样本未见空值 | 运行后减运行前的CPU堆内存变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `3856` | NA |
| 49 | `rss_before_bytes` | integer | bytes | 样本未见空值 | 运行前RSS，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70500352` | NA |
| 50 | `rss_after_bytes` | integer | bytes | 样本未见空值 | 运行后RSS，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70500352` | NA |
| 51 | `rss_peak_bytes` | integer | bytes | 样本未见空值 | 运行期间RSS峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70500352` | NA |
| 52 | `rss_delta_bytes` | integer | bytes | 样本未见空值 | 运行后减运行前的RSS变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `0` | NA |
| 53 | `gpu_before_bytes` | integer | bytes | 允许 | 运行前GPU显存占用，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |
| 54 | `gpu_after_bytes` | integer | bytes | 允许 | 运行后GPU显存占用，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |
| 55 | `gpu_peak_bytes` | integer | bytes | 允许 | 运行期间GPU显存峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |
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

<a id="schema_cb6b6e50b568"></a>
### SCHEMA_CB6B6E50B568

- 适用文件：`windows_batch02_module_summary_not_applicable_20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000.csv`
- 一行代表：一行对应一个汇总对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T032037Z_47ebe13410d3_stage08_b02_formal_88210000` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor__taylor_fp32_10e2__3137a27866a19941__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor` | NA |
| 4 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `taylor_fp32_10e2` | NA |
| 5 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `sym` | boolean | NA | 样本未见空值 | 窗函数是否使用对称形式的布尔参数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `true` | NA |
| 7 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 8 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 9 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 10 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.019180259999999994` | warmup_runs, measured_runs, timing_scope |
| 11 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.39506144999999998` | warmup_runs, measured_runs, timing_scope |
| 12 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.048550067337625562` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 13 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `4.8603990259821879e-15` | accuracy_reference, accuracy_status |
| 14 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `6.9716562063703258e-08` | accuracy_reference, accuracy_status |
| 15 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `1.0034733771186266e-07` | accuracy_reference, accuracy_status |
| 16 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `1.322186991520581e-07` | accuracy_reference, accuracy_status |
| 17 | `cuda_event_ms` | float | ms | 样本未见空值 | 使用CUDA兼容事件测得的设备时间，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.416485995054245` | warmup_runs, measured_runs, timing_scope |
| 18 | `cuda_event_source` | string | NA | 样本未见空值 | CUDA兼容事件计时值的来源或采集方式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cudaEvent` | NA |
| 19 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 20 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../../../../../../../README.md)
- [全局CSV字段语义总表](../../../../../../../../00_归档规范/CSV字段语义总表.md)
