# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：5个
- 不同schema：5种
- 生成与校验时间：`2026-08-17T16:59:16+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |
| `operator_main_results_triang_not_applicable_20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314.csv` | `SCHEMA_C185CB915CF7` | `c185cb915cf74184eb0947423159e204169b4a0392519061701fc6861eddb032` | 66 | [全部列](#schema_c185cb915cf7) | HEADER_MATCH |
| `operator_timing_samples_triang_not_applicable_20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314.csv` | `SCHEMA_8C46C287C375` | `8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca` | 31 | [全部列](#schema_8c46c287c375) | HEADER_MATCH |
| `operator_windows_triang_memory_trace_not_applicable.csv` | `SCHEMA_620652473711` | `620652473711789786d7516c8510d5c211b0f076783c0b258e367e2dbfa0ff52` | 30 | [全部列](#schema_620652473711) | HEADER_MATCH |
| `windows_batch02_module_summary_not_applicable_20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314.csv` | `SCHEMA_CB6B6E50B568` | `cb6b6e50b568088f256461889b29645473a3344adbb0bb9762a1784945926e94` | 20 | [全部列](#schema_cb6b6e50b568) | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_620652473711"></a>
### SCHEMA_620652473711

- 适用文件：``operator_windows_triang_memory_trace_not_applicable.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`620652473711789786d7516c8510d5c211b0f076783c0b258e367e2dbfa0ff52`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `ba0a42aa462da55e69c556c27f1e8036989708f0` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage08_b02_triang_scales_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `da852ccde79ce347ac1e3342ea92da9be9089cc06f8719a562bc2d05c431b83f` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `operator` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `triang__triang_int16_10e3__47864ce6053b1ce5__sym` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `triang_int16_10e3` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `1000` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `window_positions=[1000]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT16` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `CPU` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `sample_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 26 | `trace_phase` | string | NA | 允许按源schema使用NA | 原始记录字段 trace_phase。 | 结合行粒度、状态列和关联键判读。 | `before` |
| 27 | `phase_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 28 | `cpu_live_heap_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1449712` |
| 29 | `rss_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70361088` |
| 30 | `gpu_used_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：``folder_manifest.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 允许按源schema使用NA | 相对于当前清单约定根目录的文件路径。 | 结合行粒度、状态列和关联键判读。 | `CSV字段说明.md` |
| 2 | `filename` | string | NA | 允许按源schema使用NA | 文件名。 | 结合行粒度、状态列和关联键判读。 | `CSV字段说明.md` |
| 3 | `artifact_kind` | string | NA | 允许按源schema使用NA | 证据文件或治理文件类别。 | 结合行粒度、状态列和关联键判读。 | `governance` |
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `46679` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `a4992ea37fbf887f3bc35a61579d1507628d0213e3d93c387a9666e757ec843f` |
| 6 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `usr_02@10.110.12.10` |
| 7 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `gpu_02` |
| 8 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `/tmp/ZKX_dev/results/stage08_b02/formal/20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314/CSV字段说明.md` |
| 9 | `stage` | string | NA | 允许按源schema使用NA | 原始记录字段 stage。 | 结合行粒度、状态列和关联键判读。 | `stage08` |
| 10 | `test_object` | string | NA | 允许按源schema使用NA | 原始记录字段 test_object。 | 结合行粒度、状态列和关联键判读。 | `INFER_FROM_PATH` |
| 11 | `task` | string | NA | 允许按源schema使用NA | 原始记录字段 task。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 12 | `step` | string | NA | 允许按源schema使用NA | 原始记录字段 step。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 13 | `module` | string | NA | 允许按源schema使用NA | 原始记录字段 module。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 14 | `operator` | string | NA | 允许按源schema使用NA | 原始记录字段 operator。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 15 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 16 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 17 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 18 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 19 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 20 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 21 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 22 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 23 | `evidence_status` | string | NA | 允许按源schema使用NA | 证据资格或归档状态，不能仅凭非空推断PASS。 | 结合行粒度、状态列和关联键判读。 | `GOVERNANCE` |
| 24 | `formal_eligible` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NEEDS_REVIEW` |
| 25 | `related_csv` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 26 | `related_log` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 27 | `related_screenshot` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 28 | `csv_schema_id` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 29 | `csv_column_document` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 30 | `csv_header_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 31 | `csv_column_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 32 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `当前目录CSV逐文件schema映射和逐列语义说明。` |

<a id="schema_8c46c287c375"></a>
### SCHEMA_8C46C287C375

- 适用文件：``operator_timing_samples_triang_not_applicable_20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `ba0a42aa462da55e69c556c27f1e8036989708f0` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage08_b02_triang_scales_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `da852ccde79ce347ac1e3342ea92da9be9089cc06f8719a562bc2d05c431b83f` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `operator` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `triang__triang_int16_10e3__47864ce6053b1ce5__sym` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `triang_int16_10e3` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `1000` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `window_positions=[1000]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT16` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `CPU` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `sample_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 28 | `latency_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.010311000000000001` |
| 29 | `synchronized` | string | NA | 允许按源schema使用NA | 原始记录字段 synchronized。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 30 | `input_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `47864ce6053b1ce548212eb99f9471c3f7ff14a5cc6e4da5e96050bad1ea6578` |
| 31 | `output_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `17b07ba602b79c3a8074c20771cff3405993d6ef1e949e73e0e1ec98e6889d13` |

<a id="schema_c185cb915cf7"></a>
### SCHEMA_C185CB915CF7

- 适用文件：``operator_main_results_triang_not_applicable_20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`c185cb915cf74184eb0947423159e204169b4a0392519061701fc6861eddb032`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `ba0a42aa462da55e69c556c27f1e8036989708f0` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage08_b02_triang_scales_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `da852ccde79ce347ac1e3342ea92da9be9089cc06f8719a562bc2d05c431b83f` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `operator` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `triang__triang_int16_10e3__47864ce6053b1ce5__sym` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `triang_int16_10e3` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `1000` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `window_positions=[1000]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT16` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `CPU` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.010333660000000012` |
| 28 | `p50_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.010370000000000001` |
| 29 | `p95_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.01039` |
| 30 | `p99_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.0104` |
| 31 | `min_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.010019999999999999` |
| 32 | `max_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.0104` |
| 33 | `std_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `9.7707545256239022e-05` |
| 34 | `cv` | float | NA | 允许按源schema使用NA | 变异系数，std_ms除以mean_ms。 | 结合行粒度、状态列和关联键判读。 | `0.0094552699872299754` |
| 35 | `cpu_gpu_speedup` | float | NA | 允许按源schema使用NA | CPU mean_ms除以GPU mean_ms；大于1表示GPU更快。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 36 | `accuracy_reference` | string | NA | 允许按源schema使用NA | 精度指标使用的CPU/GPU参考关系。 | 结合行粒度、状态列和关联键判读。 | `cpu_typed_reference` |
| 37 | `mse` | float | NA | 允许按源schema使用NA | 相对精度参考计算的均方误差。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 38 | `rmse` | float | NA | 允许按源schema使用NA | 均方根误差，等于MSE平方根。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 39 | `relative_l2` | float | NA | 允许按源schema使用NA | 相对L2误差。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 40 | `relative_linf` | float | NA | 允许按源schema使用NA | 相对L∞误差。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 41 | `exact_match` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 42 | `mismatch_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 43 | `semantic_check` | string | NA | 允许按源schema使用NA | 与任务语义相符的精度判定规则及结论依据。 | 结合行粒度、状态列和关联键判读。 | `finite_fp64_triangular_window_shape_parity_and_symmetry_match_cpu:shape=[1000];finite=true;sym=true` |
| 44 | `accuracy_status` | string | NA | 允许按源schema使用NA | 本行精度门禁状态。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 45 | `cpu_heap_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1449712` |
| 46 | `cpu_heap_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1453056` |
| 47 | `cpu_heap_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1459952` |
| 48 | `cpu_heap_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `3344` |
| 49 | `rss_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70361088` |
| 50 | `rss_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70361088` |
| 51 | `rss_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70361088` |
| 52 | `rss_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 53 | `gpu_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 54 | `gpu_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 55 | `gpu_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 56 | `gpu_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 57 | `stability_case_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 58 | `stability_total_cases` | integer | count | 允许按源schema使用NA | 本次稳定性参数文件规定的配对case总数。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 59 | `perturbation_file` | string | NA | 允许按源schema使用NA | 本case读取的唯一稳定性参数JSON路径。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 60 | `perturbation_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 61 | `perturbation_summary` | string | NA | 允许按源schema使用NA | Task1与Task2实际扰动数值摘要，不是仅记录seed。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 62 | `input_mode` | string | NA | 允许按源schema使用NA | 输入生成或读取模式。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 63 | `input_source_file` | string | NA | 允许按源schema使用NA | 正式主输入CSV路径；NA表示由稳定性配置生成。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 64 | `input_source_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 65 | `selection_seed` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 66 | `profile_entry_id` | string | NA | 允许按源schema使用NA | 扰动配置中对应条目的稳定身份。 | 结合行粒度、状态列和关联键判读。 | `NA` |

<a id="schema_cb6b6e50b568"></a>
### SCHEMA_CB6B6E50B568

- 适用文件：``windows_batch02_module_summary_not_applicable_20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`cb6b6e50b568088f256461889b29645473a3344adbb0bb9762a1784945926e94`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314` |
| 2 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `triang__triang_int16_10e3__47864ce6053b1ce5__sym` |
| 3 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `triang` |
| 4 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `triang_int16_10e3` |
| 5 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT16` |
| 6 | `sym` | string | NA | 允许按源schema使用NA | 原始记录字段 sym。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 8 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 9 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 10 | `cpu_mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.010333660000000012` |
| 11 | `gpu_mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.25224022000000001` |
| 12 | `cpu_gpu_speedup` | float | NA | 允许按源schema使用NA | CPU mean_ms除以GPU mean_ms；大于1表示GPU更快。 | 结合行粒度、状态列和关联键判读。 | `0.040967534836434935` |
| 13 | `mse` | float | NA | 允许按源schema使用NA | 相对精度参考计算的均方误差。 | 结合行粒度、状态列和关联键判读。 | `1.6915186893922353e-16` |
| 14 | `rmse` | float | NA | 允许按源schema使用NA | 均方根误差，等于MSE平方根。 | 结合行粒度、状态列和关联键判读。 | `1.3005839801382437e-08` |
| 15 | `relative_l2` | float | NA | 允许按源schema使用NA | 相对L2误差。 | 结合行粒度、状态列和关联键判读。 | `2.2526786594492018e-08` |
| 16 | `relative_linf` | float | NA | 允许按源schema使用NA | 相对L∞误差。 | 结合行粒度、状态列和关联键判读。 | `2.9593497266780682e-08` |
| 17 | `cuda_event_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.2766839861869812` |
| 18 | `cuda_event_source` | string | NA | 允许按源schema使用NA | 原始记录字段 cuda_event_source。 | 结合行粒度、状态列和关联键判读。 | `cudaEvent` |
| 19 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 20 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
