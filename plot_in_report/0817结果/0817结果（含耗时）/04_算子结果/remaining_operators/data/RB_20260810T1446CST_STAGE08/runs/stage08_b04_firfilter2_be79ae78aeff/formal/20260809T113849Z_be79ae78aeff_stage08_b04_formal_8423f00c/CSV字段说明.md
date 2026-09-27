# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：5个
- 不同schema：5种
- 生成与校验时间：`2026-08-17T16:59:16+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |
| `operator_dtype_evidence_firfilter2_not_applicable_20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c.csv` | `SCHEMA_3AACC7DF0E1F` | `3aacc7df0e1f18d988cd49db995527b30e277777a54eca1b62c940a4a69946e2` | 13 | [全部列](#schema_3aacc7df0e1f) | HEADER_MATCH |
| `operator_filtering_firfilter2_memory_trace_not_applicable.csv` | `SCHEMA_620652473711` | `620652473711789786d7516c8510d5c211b0f076783c0b258e367e2dbfa0ff52` | 30 | [全部列](#schema_620652473711) | HEADER_MATCH |
| `operator_main_results_firfilter2_not_applicable_20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c.csv` | `SCHEMA_C185CB915CF7` | `c185cb915cf74184eb0947423159e204169b4a0392519061701fc6861eddb032` | 66 | [全部列](#schema_c185cb915cf7) | HEADER_MATCH |
| `operator_timing_samples_firfilter2_not_applicable_20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c.csv` | `SCHEMA_8C46C287C375` | `8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca` | 31 | [全部列](#schema_8c46c287c375) | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_3aacc7df0e1f"></a>
### SCHEMA_3AACC7DF0E1F

- 适用文件：``operator_dtype_evidence_firfilter2_not_applicable_20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`3aacc7df0e1f18d988cd49db995527b30e277777a54eca1b62c940a4a69946e2`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c` |
| 2 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2__firfilter2_int8_10e2__b813600a26be64d8` |
| 3 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 4 | `requested_dtype` | string | NA | 允许按源schema使用NA | 原始记录字段 requested_dtype。 | 结合行粒度、状态列和关联键判读。 | `INT8` |
| 5 | `dtype_semantics` | string | NA | 允许按源schema使用NA | 原始记录字段 dtype_semantics。 | 结合行粒度、状态列和关联键判读。 | `actual_same_dtype_x_b_arrays_fp32_bidirectional_direct` |
| 6 | `typed_input_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `2` |
| 7 | `typed_input_manifest` | string | NA | 允许按源schema使用NA | 原始记录字段 typed_input_manifest。 | 结合行粒度、状态列和关联键判读。 | `x:INT8:[10x10]:bytes=100:sha256=6a9cf3ded8ed69738b6c3aae545730a535b6150f6e262d13725afc90427ea2f3\|b:INT8:[3]:bytes=3:sha256=75c8fd04ad916aec3e3d5cb76a452b116b3d4d0912a0a485e9fb8e3d240e210c` |
| 8 | `actual_input_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `b813600a26be64d887e6ce07113d566f0a330296ff2797900a792df86074adf8` |
| 9 | `compute_dtype` | string | NA | 允许按源schema使用NA | 原始记录字段 compute_dtype。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 10 | `fft_dtype` | string | NA | 允许按源schema使用NA | 原始记录字段 fft_dtype。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 11 | `output_dtype` | string | NA | 允许按源schema使用NA | 原始记录字段 output_dtype。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 12 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 13 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |

<a id="schema_620652473711"></a>
### SCHEMA_620652473711

- 适用文件：``operator_filtering_firfilter2_memory_trace_not_applicable.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`620652473711789786d7516c8510d5c211b0f076783c0b258e367e2dbfa0ff52`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `be79ae78aeff2563251f30c57273a504021cb2f8` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage08_b04_firfilter2_scales_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `db7cf5052063d10fdd76bc34d664d61877de1512e467953b40bcded0ba2450b1` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `operator` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2__firfilter2_int8_10e2__b813600a26be64d8` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2_int8_10e2` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^2` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `103` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `x=[10x10];b=[3]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT8` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `CPU` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `sample_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 26 | `trace_phase` | string | NA | 允许按源schema使用NA | 原始记录字段 trace_phase。 | 结合行粒度、状态列和关联键判读。 | `before` |
| 27 | `phase_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 28 | `cpu_live_heap_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1426976` |
| 29 | `rss_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70340608` |
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
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `44724` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `c4e14f01d50fc6afcb68e869d8053e66067676c6bc276c38c9f2b5e8d9793506` |
| 6 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `usr_02@10.110.12.10` |
| 7 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `gpu_02` |
| 8 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `/tmp/ZKX_dev/results/stage08_b04_firfilter2_be79ae78aeff/formal/20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c/CSV字段说明.md` |
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

- 适用文件：``operator_timing_samples_firfilter2_not_applicable_20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `be79ae78aeff2563251f30c57273a504021cb2f8` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage08_b04_firfilter2_scales_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `db7cf5052063d10fdd76bc34d664d61877de1512e467953b40bcded0ba2450b1` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `operator` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2__firfilter2_int8_10e2__b813600a26be64d8` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2_int8_10e2` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^2` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `103` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `x=[10x10];b=[3]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT8` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `CPU` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `sample_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 28 | `latency_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.034889` |
| 29 | `synchronized` | string | NA | 允许按源schema使用NA | 原始记录字段 synchronized。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 30 | `input_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `b813600a26be64d887e6ce07113d566f0a330296ff2797900a792df86074adf8` |
| 31 | `output_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `6aeb74bb0d9ec04d27e349ea4db4450d1b6711851e0546a476c243dfd6a71eed` |

<a id="schema_c185cb915cf7"></a>
### SCHEMA_C185CB915CF7

- 适用文件：``operator_main_results_firfilter2_not_applicable_20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`c185cb915cf74184eb0947423159e204169b4a0392519061701fc6861eddb032`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z_be79ae78aeff_stage08_b04_formal_8423f00c` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `20260809T113849Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `be79ae78aeff2563251f30c57273a504021cb2f8` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage08_b04_firfilter2_scales_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `db7cf5052063d10fdd76bc34d664d61877de1512e467953b40bcded0ba2450b1` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `operator` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `firfilter2` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2__firfilter2_int8_10e2__b813600a26be64d8` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `firfilter2_int8_10e2` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^2` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `103` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `x=[10x10];b=[3]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `INT8` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `CPU` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `not_applicable` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.035369940000000002` |
| 28 | `p50_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.034749000000000002` |
| 29 | `p95_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.035021500000000004` |
| 30 | `p99_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.051845400000000035` |
| 31 | `min_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.034479999999999997` |
| 32 | `max_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.058319999999999997` |
| 33 | `std_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.0032288875261303231` |
| 34 | `cv` | float | NA | 允许按源schema使用NA | 变异系数，std_ms除以mean_ms。 | 结合行粒度、状态列和关联键判读。 | `0.091289030349792022` |
| 35 | `cpu_gpu_speedup` | float | NA | 允许按源schema使用NA | CPU mean_ms除以GPU mean_ms；大于1表示GPU更快。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 36 | `accuracy_reference` | string | NA | 允许按源schema使用NA | 精度指标使用的CPU/GPU参考关系。 | 结合行粒度、状态列和关联键判读。 | `cpu_independent_bidirectional_fir_reference` |
| 37 | `mse` | float | NA | 允许按源schema使用NA | 相对精度参考计算的均方误差。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 38 | `rmse` | float | NA | 允许按源schema使用NA | 均方根误差，等于MSE平方根。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 39 | `relative_l2` | float | NA | 允许按源schema使用NA | 相对L2误差。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 40 | `relative_linf` | float | NA | 允许按源schema使用NA | 相对L∞误差。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 41 | `exact_match` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 42 | `mismatch_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 43 | `semantic_check` | string | NA | 允许按源schema使用NA | 与任务语义相符的精度判定规则及结论依据。 | 结合行粒度、状态列和关联键判读。 | `shape_axis_padding_and_fp32_output_match_cpu:axis=1;padtype=odd;padlen=4` |
| 44 | `accuracy_status` | string | NA | 允许按源schema使用NA | 本行精度门禁状态。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 45 | `cpu_heap_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1426976` |
| 46 | `cpu_heap_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1430192` |
| 47 | `cpu_heap_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1430192` |
| 48 | `cpu_heap_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `3216` |
| 49 | `rss_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70340608` |
| 50 | `rss_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70340608` |
| 51 | `rss_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `70340608` |
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
