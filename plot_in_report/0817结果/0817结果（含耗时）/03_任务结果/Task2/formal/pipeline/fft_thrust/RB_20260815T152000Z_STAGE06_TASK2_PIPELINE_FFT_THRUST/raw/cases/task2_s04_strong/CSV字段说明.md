# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：6个
- 不同schema：6种
- 生成与校验时间：`2026-08-17T17:01:37+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |
| `task2_accuracy_details_fft_thrust_formal.csv` | `SCHEMA_EB3D6CA4887A` | `eb3d6ca4887a27034f53b26d2b71372821b309aa79f9f85874ce8e55c7b86074` | 29 | [全部列](#schema_eb3d6ca4887a) | HEADER_MATCH |
| `task2_benchmark_fft_thrust_formal.csv` | `SCHEMA_15EC33142824` | `15ec331428246fc1053d8882411ff049b5bf0a5c9d1e2dcc8eaef115acd49a4f` | 59 | [全部列](#schema_15ec33142824) | HEADER_MATCH |
| `task2_iteration_timing_fft_thrust_formal.csv` | `SCHEMA_8C46C287C375` | `8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca` | 31 | [全部列](#schema_8c46c287c375) | HEADER_MATCH |
| `task2_memory_trace_fft_thrust_formal.csv` | `SCHEMA_FCFDC9F55CEF` | `fcfdc9f55cef66edb82e85c0d1413f77f720f1db48f1a32f10f47f5c9e386cfe` | 34 | [全部列](#schema_fcfdc9f55cef) | HEADER_MATCH |
| `task2_pipeline_summary_fft_thrust_formal.csv` | `SCHEMA_8DB19D860D1A` | `8db19d860d1ae424af7e74e465dba866094398b8cfe5b8a323bc2691b6eb9500` | 35 | [全部列](#schema_8db19d860d1a) | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_15ec33142824"></a>
### SCHEMA_15EC33142824

- 适用文件：``task2_benchmark_fft_thrust_formal.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`15ec331428246fc1053d8882411ff049b5bf0a5c9d1e2dcc8eaef115acd49a4f`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260815T152000Z_b0478e7_fft_thrust_fft_thrust_task2_pipeline_benchmark_task2_s04_strong` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-15T15:05:00Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `b0478e7c5f1ad93e4cf41827d653d191c4a86804` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `task_scales_formal_v3` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `baaa5c346f0c31ac4bb1fd1f81069be745425af6b28982a50aa11d42ff41b232` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `task` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `Task2.pipeline` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `Task2` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `pipeline_total` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `chirp+gausspulse+sawtooth+square+hamming+firwin+firfilter+cubic+fm_demod+correlate+spectrogram+cwt+argrelextrema+kalman_filter;host:ricker+fusion+observation` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `task2_pipeline__task2_s04_strong__2843bbefb932f87d` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `task2_s04_strong` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `4096` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `[4096]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `cpu` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `81.814956570000007` |
| 28 | `p50_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `80.683043999999995` |
| 29 | `p95_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `87.043762600000008` |
| 30 | `p99_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `92.452107270000056` |
| 31 | `min_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `79.837383000000003` |
| 32 | `max_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `104.560824` |
| 33 | `std_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `3.3268415946248724` |
| 34 | `cv` | float | NA | 允许按源schema使用NA | 变异系数，std_ms除以mean_ms。 | 结合行粒度、状态列和关联键判读。 | `0.040663000190905949` |
| 35 | `cpu_gpu_speedup` | float | NA | 允许按源schema使用NA | CPU mean_ms除以GPU mean_ms；大于1表示GPU更快。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 36 | `accuracy_reference` | string | NA | 允许按源schema使用NA | 精度指标使用的CPU/GPU参考关系。 | 结合行粒度、状态列和关联键判读。 | `independent typed CPU six-step pipeline` |
| 37 | `mse` | float | NA | 允许按源schema使用NA | 相对精度参考计算的均方误差。 | 结合行粒度、状态列和关联键判读。 | `3.358737096161631e-10` |
| 38 | `rmse` | float | NA | 允许按源schema使用NA | 均方根误差，等于MSE平方根。 | 结合行粒度、状态列和关联键判读。 | `1.8326857603423535e-05` |
| 39 | `relative_l2` | float | NA | 允许按源schema使用NA | 相对L2误差。 | 结合行粒度、状态列和关联键判读。 | `1.138448581188131e-06` |
| 40 | `relative_linf` | float | NA | 允许按源schema使用NA | 相对L∞误差。 | 结合行粒度、状态列和关联键判读。 | `5.7077929638894572e-06` |
| 41 | `exact_match` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 42 | `mismatch_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 43 | `semantic_check` | string | NA | 允许按源schema使用NA | 与任务语义相符的精度判定规则及结论依据。 | 结合行粒度、状态列和关联键判读。 | `all_six_step_semantic_rules_pass` |
| 44 | `accuracy_status` | string | NA | 允许按源schema使用NA | 本行精度门禁状态。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 45 | `accuracy_threshold_set_id` | string | NA | 允许按源schema使用NA | 精度阈值集合身份或其SHA-256。 | 结合行粒度、状态列和关联键判读。 | `task_accuracy_formal_v7` |
| 46 | `accuracy_thresholds_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `b1ea03d79712ee0089bd3a3fb4fba74ce19c6ff5cf87e70b0ff62b590a36ae6f` |
| 47 | `cpu_heap_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `2101056` |
| 48 | `cpu_heap_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `2680512` |
| 49 | `cpu_heap_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `2680512` |
| 50 | `cpu_heap_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `579456` |
| 51 | `rss_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `76541952` |
| 52 | `rss_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `77807616` |
| 53 | `rss_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `77807616` |
| 54 | `rss_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `1265664` |
| 55 | `gpu_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 56 | `gpu_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 57 | `gpu_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 58 | `gpu_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 59 | `return_code` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |

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
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `44887` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `7f114ba6e65715ba8e34e39dedd395e5fa24e57d3fe869dbf2696564c632a347` |
| 6 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `LOCAL_WINDOWS` |
| 7 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 8 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `LOCAL_GOVERNANCE` |
| 9 | `stage` | string | NA | 允许按源schema使用NA | 原始记录字段 stage。 | 结合行粒度、状态列和关联键判读。 | `stage05` |
| 10 | `test_object` | string | NA | 允许按源schema使用NA | 原始记录字段 test_object。 | 结合行粒度、状态列和关联键判读。 | `task1_pipeline_benchmark` |
| 11 | `task` | string | NA | 允许按源schema使用NA | 原始记录字段 task。 | 结合行粒度、状态列和关联键判读。 | `Task1` |
| 12 | `step` | string | NA | 允许按源schema使用NA | 原始记录字段 step。 | 结合行粒度、状态列和关联键判读。 | `pipeline_total` |
| 13 | `module` | string | NA | 允许按源schema使用NA | 原始记录字段 module。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 14 | `operator` | string | NA | 允许按源schema使用NA | 原始记录字段 operator。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_OR_NA` |
| 15 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260810T123934Z_d186f2993f4a_source_isolated` |
| 16 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `aggregate` |
| 17 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 18 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `ComplexFP32` |
| 19 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `aggregate` |
| 20 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `governance` |
| 21 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `d186f2993f4a5211c7662571e948fb6b6d4e8aab` |
| 22 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 23 | `evidence_status` | string | NA | 允许按源schema使用NA | 证据资格或归档状态，不能仅凭非空推断PASS。 | 结合行粒度、状态列和关联键判读。 | `GOVERNANCE` |
| 24 | `formal_eligible` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 25 | `related_csv` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 26 | `related_log` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 27 | `related_screenshot` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 28 | `csv_schema_id` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 29 | `csv_column_document` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 30 | `csv_header_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 31 | `csv_column_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 32 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `stage05 Task1 fft_thrust pipeline正式110-case归档，三端逐文件SHA-256一致。` |

<a id="schema_8c46c287c375"></a>
### SCHEMA_8C46C287C375

- 适用文件：``task2_iteration_timing_fft_thrust_formal.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`8c46c287c375b5339ae4c31d42bef8ea0fabf7240f7c7719c0f1a0f54bcb97ca`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260815T152000Z_b0478e7_fft_thrust_fft_thrust_task2_pipeline_benchmark_task2_s04_strong` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-15T15:05:00Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `b0478e7c5f1ad93e4cf41827d653d191c4a86804` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `task_scales_formal_v3` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `baaa5c346f0c31ac4bb1fd1f81069be745425af6b28982a50aa11d42ff41b232` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `task` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `Task2.pipeline` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `Task2` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `pipeline_total` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `chirp+gausspulse+sawtooth+square+hamming+firwin+firfilter+cubic+fm_demod+correlate+spectrogram+cwt+argrelextrema+kalman_filter;host:ricker+fusion+observation` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `task2_pipeline__task2_s04_strong__2843bbefb932f87d` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `task2_s04_strong` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `4096` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `[4096]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `cpu` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `sample_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 28 | `latency_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `81.075209000000001` |
| 29 | `synchronized` | string | NA | 允许按源schema使用NA | 原始记录字段 synchronized。 | 结合行粒度、状态列和关联键判读。 | `true` |
| 30 | `input_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `2843bbefb932f87d5db962478aa04b7f31b0778b5279fc54585d5294adbbd973` |
| 31 | `output_digest` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `8bf7b58e7b462467a9a28b87930740f09244d92c5867e44fbbe9f64944d7a784` |

<a id="schema_8db19d860d1a"></a>
### SCHEMA_8DB19D860D1A

- 适用文件：``task2_pipeline_summary_fft_thrust_formal.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`8db19d860d1ae424af7e74e465dba866094398b8cfe5b8a323bc2691b6eb9500`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260815T152000Z_b0478e7_fft_thrust_fft_thrust_task2_pipeline_benchmark_task2_s04_strong` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-15T15:05:00Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `b0478e7c5f1ad93e4cf41827d653d191c4a86804` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `task_scales_formal_v3` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `baaa5c346f0c31ac4bb1fd1f81069be745425af6b28982a50aa11d42ff41b232` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `task` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `Task2.pipeline` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `Task2` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `pipeline_total` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `chirp+gausspulse+sawtooth+square+hamming+firwin+firfilter+cubic+fm_demod+correlate+spectrogram+cwt+argrelextrema+kalman_filter;host:ricker+fusion+observation` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `task2_pipeline__task2_s04_strong__2843bbefb932f87d` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `task2_s04_strong` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `4096` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `[4096]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `cpu` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `component` | string | NA | 允许按源schema使用NA | 原始记录字段 component。 | 结合行粒度、状态列和关联键判读。 | `step1` |
| 26 | `upstream` | string | NA | 允许按源schema使用NA | 原始记录字段 upstream。 | 结合行粒度、状态列和关联键判读。 | `config` |
| 27 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 28 | `mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `1.7711823600000007` |
| 29 | `p50_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `1.7625855000000001` |
| 30 | `p95_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `1.8045529999999999` |
| 31 | `p99_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `1.9304668` |
| 32 | `min_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `1.7435849999999999` |
| 33 | `max_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `1.940347` |
| 34 | `std_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0.033542495203106169` |
| 35 | `cv` | float | NA | 允许按源schema使用NA | 变异系数，std_ms除以mean_ms。 | 结合行粒度、状态列和关联键判读。 | `0.018937911736601847` |

<a id="schema_eb3d6ca4887a"></a>
### SCHEMA_EB3D6CA4887A

- 适用文件：``task2_accuracy_details_fft_thrust_formal.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`eb3d6ca4887a27034f53b26d2b71372821b309aa79f9f85874ce8e55c7b86074`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260815T152000Z_b0478e7_fft_thrust_fft_thrust_task2_pipeline_benchmark_task2_s04_strong` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `b0478e7c5f1ad93e4cf41827d653d191c4a86804` |
| 5 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `task2_pipeline__task2_s04_strong__2843bbefb932f87d` |
| 6 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `task2_s04_strong` |
| 7 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 8 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 9 | `threshold_set_id` | string | NA | 允许按源schema使用NA | 原始记录字段 threshold_set_id。 | 结合行粒度、状态列和关联键判读。 | `task_accuracy_formal_v7` |
| 10 | `thresholds_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `b1ea03d79712ee0089bd3a3fb4fba74ce19c6ff5cf87e70b0ff62b590a36ae6f` |
| 11 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `Task2.step1.echo` |
| 12 | `output_name` | string | NA | 允许按源schema使用NA | 原始记录字段 output_name。 | 结合行粒度、状态列和关联键判读。 | `echo` |
| 13 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 14 | `output_kind` | string | NA | 允许按源schema使用NA | 原始记录字段 output_kind。 | 结合行粒度、状态列和关联键判读。 | `floating` |
| 15 | `mse` | float | NA | 允许按源schema使用NA | 相对精度参考计算的均方误差。 | 结合行粒度、状态列和关联键判读。 | `1.3773035243227517e-15` |
| 16 | `mse_max` | string | NA | 允许按源schema使用NA | 原始记录字段 mse_max。 | 结合行粒度、状态列和关联键判读。 | `8.9999999999999999e-10` |
| 17 | `rmse` | float | NA | 允许按源schema使用NA | 均方根误差，等于MSE平方根。 | 结合行粒度、状态列和关联键判读。 | `3.7112040153065578e-08` |
| 18 | `rmse_max` | string | NA | 允许按源schema使用NA | 原始记录字段 rmse_max。 | 结合行粒度、状态列和关联键判读。 | `3.0000000000000001e-05` |
| 19 | `relative_l2` | float | NA | 允许按源schema使用NA | 相对L2误差。 | 结合行粒度、状态列和关联键判读。 | `3.8994537337248822e-08` |
| 20 | `relative_l2_max` | string | NA | 允许按源schema使用NA | 原始记录字段 relative_l2_max。 | 结合行粒度、状态列和关联键判读。 | `3.0000000000000001e-05` |
| 21 | `relative_linf` | float | NA | 允许按源schema使用NA | 相对L∞误差。 | 结合行粒度、状态列和关联键判读。 | `1.4477685290302563e-07` |
| 22 | `relative_linf_max` | string | NA | 允许按源schema使用NA | 原始记录字段 relative_linf_max。 | 结合行粒度、状态列和关联键判读。 | `3.0000000000000001e-05` |
| 23 | `relative_floor` | string | NA | 允许按源schema使用NA | 原始记录字段 relative_floor。 | 结合行粒度、状态列和关联键判读。 | `9.9999999999999998e-13` |
| 24 | `exact_match` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 25 | `mismatch_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 26 | `max_mismatch_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 27 | `semantic_rule` | string | NA | 允许按源schema使用NA | 原始记录字段 semantic_rule。 | 结合行粒度、状态列和关联键判读。 | `weighted_echo_sum_and_parameter_perturbation_pass` |
| 28 | `semantic_check` | string | NA | 允许按源schema使用NA | 与任务语义相符的精度判定规则及结论依据。 | 结合行粒度、状态列和关联键判读。 | `weighted_echo_sum_and_parameter_perturbation_pass:PASS` |
| 29 | `accuracy_status` | string | NA | 允许按源schema使用NA | 本行精度门禁状态。 | 结合行粒度、状态列和关联键判读。 | `PASS` |

<a id="schema_fcfdc9f55cef"></a>
### SCHEMA_FCFDC9F55CEF

- 适用文件：``task2_memory_trace_fft_thrust_formal.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`fcfdc9f55cef66edb82e85c0d1413f77f720f1db48f1a32f10f47f5c9e386cfe`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260815T152000Z_b0478e7_fft_thrust_fft_thrust_task2_pipeline_benchmark_task2_s04_strong` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `formal` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-15T15:05:00Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `b0478e7c5f1ad93e4cf41827d653d191c4a86804` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `task_scales_formal_v3` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `baaa5c346f0c31ac4bb1fd1f81069be745425af6b28982a50aa11d42ff41b232` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `task` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `Task2.pipeline` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `Task2` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `pipeline_total` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `chirp+gausspulse+sawtooth+square+hamming+firwin+firfilter+cubic+fm_demod+correlate+spectrogram+cwt+argrelextrema+kalman_filter;host:ricker+fusion+observation` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `task2_pipeline__task2_s04_strong__2843bbefb932f87d` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `task2_s04_strong` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^3` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `4096` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `[4096]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `cpu` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/end-to-end` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `20` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `100` |
| 27 | `sample_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 28 | `resource_scope` | string | NA | 允许按源schema使用NA | 原始记录字段 resource_scope。 | 结合行粒度、状态列和关联键判读。 | `step1` |
| 29 | `trace_phase` | string | NA | 允许按源schema使用NA | 原始记录字段 trace_phase。 | 结合行粒度、状态列和关联键判读。 | `before` |
| 30 | `phase_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 31 | `cpu_live_heap_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `2101056` |
| 32 | `rss_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `76541952` |
| 33 | `gpu_used_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 34 | `memory_source` | string | NA | 允许按源schema使用NA | 原始记录字段 memory_source。 | 结合行粒度、状态列和关联键判读。 | `same_pipeline_execution:mallinfo2+/proc/self/statm` |
