# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：2个
- 不同schema：2种
- 生成与校验时间：`2026-08-14T19:53:49+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |
| `task2_stability_results.csv` | `SCHEMA_29C746A81251` | `29c746a8125110b9851eaeae4b30f3e2a2c8b7e994a3fc61aa0703819cc12e5f` | 69 | [全部列](#schema_29c746a81251) | HEADER_MATCH |

## 子目录CSV说明索引

本目录没有子目录。

## schema完整定义

<a id="schema_29c746a81251"></a>
### SCHEMA_29C746A81251

- 适用文件：``task2_stability_results.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`29c746a8125110b9851eaeae4b30f3e2a2c8b7e994a3fc61aa0703819cc12e5f`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | string | NA | 允许按源schema使用NA | 结果CSV schema版本。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 2 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260814T021000Z_8841247_target_semantic_1000` |
| 3 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `stability` |
| 4 | `test_time_utc` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-14T02:09:29Z` |
| 5 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `88412479051830431ace7e6127a7d455cba72a9c` |
| 6 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 7 | `config_id` | string | NA | 允许按源schema使用NA | 运行配置的稳定标识。 | 结合行粒度、状态列和关联键判读。 | `stage10_stability_formal_1000_v1` |
| 8 | `config_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `9dea2ecdcfad71439b62ae91d497ee068efe8001ef76d77b75143611adb0098a` |
| 9 | `target_kind` | string | NA | 允许按源schema使用NA | 测试对象类别。 | 结合行粒度、状态列和关联键判读。 | `task` |
| 10 | `target` | string | NA | 允许按源schema使用NA | 任务或步骤的完整测试目标。 | 结合行粒度、状态列和关联键判读。 | `Task2.step1` |
| 11 | `task_name` | string | NA | 允许按源schema使用NA | 任务名称，Task1或Task2。 | 结合行粒度、状态列和关联键判读。 | `Task2` |
| 12 | `step_name` | string | NA | 允许按源schema使用NA | pipeline步骤名称或pipeline_total。 | 结合行粒度、状态列和关联键判读。 | `step1` |
| 13 | `operator_name` | string | NA | 允许按源schema使用NA | 该步骤对应的算法或算子名称。 | 结合行粒度、状态列和关联键判读。 | `chirp+gausspulse+sawtooth+square` |
| 14 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `stage10_pair_0506_745edc19` |
| 15 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `task2_s08_baseline` |
| 16 | `order_of_magnitude` | string | NA | 允许按源schema使用NA | 原始记录字段 order_of_magnitude。 | 结合行粒度、状态列和关联键判读。 | `10^4` |
| 17 | `actual_elements` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `16384` |
| 18 | `input_shape` | string | NA | 允许按源schema使用NA | 本行实际输入张量形状。 | 结合行粒度、状态列和关联键判读。 | `[16384]` |
| 19 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 20 | `device` | string | NA | 允许按源schema使用NA | 本行结果的实际计算设备，cpu或gpu。 | 结合行粒度、状态列和关联键判读。 | `cpu` |
| 21 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 22 | `timing_scope` | string | NA | 允许按源schema使用NA | 耗时覆盖的同步与步骤边界。 | 结合行粒度、状态列和关联键判读。 | `compatibility/step` |
| 23 | `status` | string | NA | 允许按源schema使用NA | 本行或本case执行状态，必须按枚举值判断。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 24 | `error_code` | string | NA | 允许按源schema使用NA | 程序记录的错误码；OK表示无错误。 | 结合行粒度、状态列和关联键判读。 | `OK` |
| 25 | `warmup_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 26 | `measured_runs` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 27 | `mean_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `6.61245100000000008e+00` |
| 28 | `p50_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `6.61245100000000008e+00` |
| 29 | `p95_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `6.61245100000000008e+00` |
| 30 | `p99_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `6.61245100000000008e+00` |
| 31 | `min_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `6.61245100000000008e+00` |
| 32 | `max_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `6.61245100000000008e+00` |
| 33 | `std_ms` | float | ms | 允许按源schema使用NA | 同步边界内的实测耗时统计。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 34 | `cv` | float | NA | 允许按源schema使用NA | 变异系数，std_ms除以mean_ms。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 35 | `cpu_gpu_speedup` | float | NA | 允许按源schema使用NA | CPU mean_ms除以GPU mean_ms；大于1表示GPU更快。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 36 | `accuracy_reference` | string | NA | 允许按源schema使用NA | 精度指标使用的CPU/GPU参考关系。 | 结合行粒度、状态列和关联键判读。 | `independent CPU pipeline versus GPU pipeline on one frozen paired perturbation` |
| 37 | `mse` | float | NA | 允许按源schema使用NA | 相对精度参考计算的均方误差。 | 结合行粒度、状态列和关联键判读。 | `4.84890839858861467e-16` |
| 38 | `rmse` | float | NA | 允许按源schema使用NA | 均方根误差，等于MSE平方根。 | 结合行粒度、状态列和关联键判读。 | `2.20202370527399513e-08` |
| 39 | `relative_l2` | float | NA | 允许按源schema使用NA | 相对L2误差。 | 结合行粒度、状态列和关联键判读。 | `4.19753534078952541e-08` |
| 40 | `relative_linf` | float | NA | 允许按源schema使用NA | 相对L∞误差。 | 结合行粒度、状态列和关联键判读。 | `1.23505521437841393e-07` |
| 41 | `exact_match` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 42 | `mismatch_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `1` |
| 43 | `semantic_check` | string | NA | 允许按源schema使用NA | 与任务语义相符的精度判定规则及结论依据。 | 结合行粒度、状态列和关联键判读。 | `finite relative_l2<=1e-3 and relative_linf<=1e-3` |
| 44 | `accuracy_status` | string | NA | 允许按源schema使用NA | 本行精度门禁状态。 | 结合行粒度、状态列和关联键判读。 | `PASS` |
| 45 | `accuracy_threshold_set_id` | string | NA | 允许按源schema使用NA | 精度阈值集合身份或其SHA-256。 | 结合行粒度、状态列和关联键判读。 | `stage10_task2_relative_v1` |
| 46 | `accuracy_thresholds_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 47 | `cpu_heap_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `3113568` |
| 48 | `cpu_heap_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `3572432` |
| 49 | `cpu_heap_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `3572432` |
| 50 | `cpu_heap_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `458864` |
| 51 | `rss_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `77656064` |
| 52 | `rss_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `77656064` |
| 53 | `rss_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `77656064` |
| 54 | `rss_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 55 | `gpu_before_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 56 | `gpu_after_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 57 | `gpu_peak_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 58 | `gpu_delta_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 59 | `return_code` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 60 | `stability_case_index` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `506` |
| 61 | `stability_total_cases` | integer | count | 允许按源schema使用NA | 本次稳定性参数文件规定的配对case总数。 | 结合行粒度、状态列和关联键判读。 | `1000` |
| 62 | `perturbation_file` | string | NA | 允许按源schema使用NA | 本case读取的唯一稳定性参数JSON路径。 | 结合行粒度、状态列和关联键判读。 | `/tmp/ZKX_dev/test_all/configs/stability/stage10_stability_formal_1000_v1/task_stability_params.json` |
| 63 | `perturbation_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `9dea2ecdcfad71439b62ae91d497ee068efe8001ef76d77b75143611adb0098a` |
| 64 | `perturbation_summary` | string | NA | 允许按源schema使用NA | Task1与Task2实际扰动数值摘要，不是仅记录seed。 | 结合行粒度、状态列和关联键判读。 | `Task1(delay=174,doppler_bin=8,amplitude=1.001894387,noise_std=0.033832556);Task2(delay=3251,target=0.815988405,gaussian=0.154900349,sawtooth=0.058751028,square=0.058156936,noise_amplitude=0.041142669,cutoff_hz=88.846658933,kalman_R=0.001965959)` |
| 65 | `input_mode` | string | NA | 允许按源schema使用NA | 输入生成或读取模式。 | 结合行粒度、状态列和关联键判读。 | `generated` |
| 66 | `input_source_file` | string | NA | 允许按源schema使用NA | 正式主输入CSV路径；NA表示由稳定性配置生成。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 67 | `input_source_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 68 | `selection_seed` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `1952373785` |
| 69 | `profile_entry_id` | string | NA | 允许按源schema使用NA | 扰动配置中对应条目的稳定身份。 | 结合行粒度、状态列和关联键判读。 | `stage10_pair_0506_745edc19` |

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：``folder_manifest.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 允许按源schema使用NA | 相对于当前清单约定根目录的文件路径。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 2 | `filename` | string | NA | 允许按源schema使用NA | 文件名。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 3 | `artifact_kind` | string | NA | 允许按源schema使用NA | 证据文件或治理文件类别。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 6 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 7 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 8 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 9 | `stage` | string | NA | 允许按源schema使用NA | 原始记录字段 stage。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 10 | `test_object` | string | NA | 允许按源schema使用NA | 原始记录字段 test_object。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 11 | `task` | string | NA | 允许按源schema使用NA | 原始记录字段 task。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 12 | `step` | string | NA | 允许按源schema使用NA | 原始记录字段 step。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 13 | `module` | string | NA | 允许按源schema使用NA | 原始记录字段 module。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 14 | `operator` | string | NA | 允许按源schema使用NA | 原始记录字段 operator。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 15 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 16 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 17 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 18 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 19 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 20 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 21 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 22 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 23 | `evidence_status` | string | NA | 允许按源schema使用NA | 证据资格或归档状态，不能仅凭非空推断PASS。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 24 | `formal_eligible` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 25 | `related_csv` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 26 | `related_log` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 27 | `related_screenshot` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 28 | `csv_schema_id` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 29 | `csv_column_document` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 30 | `csv_header_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 31 | `csv_column_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 32 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `NA` |
