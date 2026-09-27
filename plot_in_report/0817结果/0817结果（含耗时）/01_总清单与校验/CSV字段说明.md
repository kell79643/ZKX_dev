# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：5个
- 不同schema：5种
- 生成与校验时间：`2026-08-17T16:56:58+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `artifact_manifest.csv` | `SCHEMA_5629860C33F9` | `5629860c33f956ee61e946195691e8480b3783d1e510a53be332472669896ffd` | 31 | [全部列](#schema_5629860c33f9) | HEADER_MATCH |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |
| `missing_artifacts.csv` | `SCHEMA_1576D69B43BF` | `1576d69b43bfd265d4ce1e9bb68b325003edd067f9db6a23e92c494e539268ab` | 10 | [全部列](#schema_1576d69b43bf) | HEADER_MATCH |
| `recovery_batches.csv` | `SCHEMA_09421B1AA778` | `09421b1aa7787a300d376883bfdf05fd9e412b7731efa81e500df2d3176ea52c` | 21 | [全部列](#schema_09421b1aa778) | HEADER_MATCH |
| `sha256_manifest.csv` | `SCHEMA_0FE1A4609774` | `0fe1a4609774ed53004837dc9a11240bbdd4c1bba2117a799a30d72617d6d46e` | 10 | [全部列](#schema_0fe1a4609774) | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明入口 |
| --- | ---: | --- |
| `recovery_batches` | 12 | [CSV字段说明](./recovery_batches/CSV字段说明.md) |

## schema完整定义

<a id="schema_09421b1aa778"></a>
### SCHEMA_09421B1AA778

- 适用文件：``recovery_batches.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`09421b1aa7787a300d376883bfdf05fd9e412b7731efa81e500df2d3176ea52c`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `recovery_batch_id` | string | NA | 允许按源schema使用NA | 原始记录字段 recovery_batch_id。 | 结合行粒度、状态列和关联键判读。 | `RB_20260810T1446CST_STAGE07_08` |
| 2 | `recovery_timestamp` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-10T14:46:00+08:00` |
| 3 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `usr_02@10.110.12.10` |
| 4 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `gpu_02` |
| 5 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `/tmp/zkx_stage07_08_20260810T1446CST.tar.gz` |
| 6 | `remote_host_temp_path` | string | NA | 允许按源schema使用NA | 原始记录字段 remote_host_temp_path。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 7 | `target_path` | string | NA | 允许按源schema使用NA | 原始记录字段 target_path。 | 结合行粒度、状态列和关联键判读。 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/\|04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/` |
| 8 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_DECLARED` |
| 9 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_DECLARED` |
| 10 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_DECLARED` |
| 11 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_DECLARED` |
| 12 | `file_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `17301` |
| 13 | `total_bytes` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `13097773` |
| 14 | `source_manifest_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `01_总清单与校验/recovery_batches/RB_20260810T1446CST_STAGE07_08/zkx_stage07_08_20260810T1446CST_host_files.sha256` |
| 15 | `source_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `619370544e5a7c615ca88dd0b69e5bff018a7427973184363b8ff22658d9867e` |
| 16 | `target_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `619370544e5a7c615ca88dd0b69e5bff018a7427973184363b8ff22658d9867e` |
| 17 | `readme_coverage_status` | string | NA | 允许按源schema使用NA | 原始记录字段 readme_coverage_status。 | 结合行粒度、状态列和关联键判读。 | `COMPLETE` |
| 18 | `archive_file_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `0` |
| 19 | `recovery_status` | string | NA | 允许按源schema使用NA | 原始记录字段 recovery_status。 | 结合行粒度、状态列和关联键判读。 | `RECOVERED_EXPANDED_AND_INDEXED_VERIFIED` |
| 20 | `verification_status` | string | NA | 允许按源schema使用NA | 原始记录字段 verification_status。 | 结合行粒度、状态列和关联键判读。 | `CONTAINER_HOST_WINDOWS_AND_FINAL_EXPANDED_17301_FILES_VERIFIED` |
| 21 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `联合原始包只作历史来源说明；最终以展开文件和直接索引表呈现` |

<a id="schema_0fe1a4609774"></a>
### SCHEMA_0FE1A4609774

- 适用文件：``sha256_manifest.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`0fe1a4609774ed53004837dc9a11240bbdd4c1bba2117a799a30d72617d6d46e`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `artifact_id` | string | NA | 允许按源schema使用NA | 原始记录字段 artifact_id。 | 结合行粒度、状态列和关联键判读。 | `ART_RECOVERY_STAGE07_08_MANIFEST` |
| 2 | `archive_relative_path` | string | NA | 允许按源schema使用NA | Windows证据归档中的相对定位信息。 | 结合行粒度、状态列和关联键判读。 | `01_总清单与校验/recovery_batches/RB_20260810T1446CST_STAGE07_08/` |
| 3 | `filename` | string | NA | 允许按源schema使用NA | 文件名。 | 结合行粒度、状态列和关联键判读。 | `zkx_stage07_08_20260810T1446CST_host_files.sha256` |
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `4139686` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0` |
| 6 | `hash_algorithm` | string | NA | 允许按源schema使用NA | 原始记录字段 hash_algorithm。 | 结合行粒度、状态列和关联键判读。 | `SHA-256` |
| 7 | `source_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0` |
| 8 | `verification_status` | string | NA | 允许按源schema使用NA | 原始记录字段 verification_status。 | 结合行粒度、状态列和关联键判读。 | `CONTAINER_HOST_WINDOWS_MATCH` |
| 9 | `verified_timestamp` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-11T18:22:39+08:00` |
| 10 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `17301文件清单` |

<a id="schema_1576d69b43bf"></a>
### SCHEMA_1576D69B43BF

- 适用文件：``missing_artifacts.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`1576d69b43bfd265d4ce1e9bb68b325003edd067f9db6a23e92c494e539268ab`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `missing_id` | string | NA | 允许按源schema使用NA | 原始记录字段 missing_id。 | 结合行粒度、状态列和关联键判读。 | `MISS_STAGE10_WRONG_RECOVERY_ARCHIVES` |
| 2 | `test_object` | string | NA | 允许按源schema使用NA | 原始记录字段 test_object。 | 结合行粒度、状态列和关联键判读。 | `stage10_task_stability_1000` |
| 3 | `artifact_kind` | string | NA | 允许按源schema使用NA | 证据文件或治理文件类别。 | 结合行粒度、状态列和关联键判读。 | `compressed_transport_cleanup` |
| 4 | `expected_source_path` | string | NA | 允许按源schema使用NA | 原始记录字段 expected_source_path。 | 结合行粒度、状态列和关联键判读。 | `/tmp/stage10_stability_evidence_20260814_8841247.tar.gz\|/home/usr_02/stage10_stability_evidence_20260814_8841247.tar.gz\|recovered_results/stage10_stability_20260814/` |
| 5 | `planned_archive_path` | string | NA | 允许按源schema使用NA | 原始记录字段 planned_archive_path。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 6 | `current_status` | string | NA | 允许按源schema使用NA | 原始记录字段 current_status。 | 结合行粒度、状态列和关联键判读。 | `CLOSED_VERIFIED` |
| 7 | `priority` | string | NA | 允许按源schema使用NA | 原始记录字段 priority。 | 结合行粒度、状态列和关联键判读。 | `HIGH` |
| 8 | `impact` | string | NA | 允许按源schema使用NA | 原始记录字段 impact。 | 结合行粒度、状态列和关联键判读。 | `已闭环且不再影响阶段10归档资格` |
| 9 | `completion_criterion` | string | NA | 允许按源schema使用NA | 原始记录字段 completion_criterion。 | 结合行粒度、状态列和关联键判读。 | `正式普通文件树9004项三端校验通过；用户明确授权后删除容器和宿主机压缩包及Windows错误目录；三处复扫为0` |
| 10 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `2026-08-14获得用户明确授权并执行删除；正确归档、容器正式结果和宿主机普通文件树均保留` |

<a id="schema_5629860c33f9"></a>
### SCHEMA_5629860C33F9

- 适用文件：``artifact_manifest.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`5629860c33f956ee61e946195691e8480b3783d1e510a53be332472669896ffd`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `artifact_id` | string | NA | 允许按源schema使用NA | 原始记录字段 artifact_id。 | 结合行粒度、状态列和关联键判读。 | `ART_RECOVERY_STAGE07_08_MANIFEST` |
| 2 | `artifact_kind` | string | NA | 允许按源schema使用NA | 证据文件或治理文件类别。 | 结合行粒度、状态列和关联键判读。 | `checksum_manifest` |
| 3 | `test_object` | string | NA | 允许按源schema使用NA | 原始记录字段 test_object。 | 结合行粒度、状态列和关联键判读。 | `stage07_stage08` |
| 4 | `task` | string | NA | 允许按源schema使用NA | 原始记录字段 task。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 5 | `step` | string | NA | 允许按源schema使用NA | 原始记录字段 step。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 6 | `module` | string | NA | 允许按源schema使用NA | 原始记录字段 module。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 7 | `operator` | string | NA | 允许按源schema使用NA | 原始记录字段 operator。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 8 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_DECLARED` |
| 9 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 10 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 11 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 12 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 13 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `recovery` |
| 14 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 15 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 16 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `usr_02@10.110.12.10` |
| 17 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `gpu_02` |
| 18 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `/tmp/zkx_stage07_08_20260810T1446CST_files.sha256` |
| 19 | `archive_relative_path` | string | NA | 允许按源schema使用NA | Windows证据归档中的相对定位信息。 | 结合行粒度、状态列和关联键判读。 | `01_总清单与校验/recovery_batches/RB_20260810T1446CST_STAGE07_08/` |
| 20 | `filename` | string | NA | 允许按源schema使用NA | 文件名。 | 结合行粒度、状态列和关联键判读。 | `zkx_stage07_08_20260810T1446CST_host_files.sha256` |
| 21 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `4139686` |
| 22 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0` |
| 23 | `generated_timestamp` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-10T14:46:00+08:00` |
| 24 | `recovered_timestamp` | string | NA | 允许按源schema使用NA | 带时区或UTC的时间戳。 | 结合行粒度、状态列和关联键判读。 | `2026-08-11T18:22:39+08:00` |
| 25 | `evidence_status` | string | NA | 允许按源schema使用NA | 证据资格或归档状态，不能仅凭非空推断PASS。 | 结合行粒度、状态列和关联键判读。 | `RECOVERED_VERIFIED` |
| 26 | `formal_eligible` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `false` |
| 27 | `related_csv` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 28 | `related_log` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 29 | `related_screenshot` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 30 | `related_score_id` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 31 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `联合包17301文件原始校验清单` |

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：``folder_manifest.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 允许按源schema使用NA | 相对于当前清单约定根目录的文件路径。 | 结合行粒度、状态列和关联键判读。 | `整改闭环记录_20260814.md` |
| 2 | `filename` | string | NA | 允许按源schema使用NA | 文件名。 | 结合行粒度、状态列和关联键判读。 | `整改闭环记录_20260814.md` |
| 3 | `artifact_kind` | string | NA | 允许按源schema使用NA | 证据文件或治理文件类别。 | 结合行粒度、状态列和关联键判读。 | `md` |
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `3144` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `acacaf35924023956d432ff8a46132405695d0648533bcec85888ec25ccefc66` |
| 6 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `LOCAL_WINDOWS` |
| 7 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 8 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `LOCAL_GOVERNANCE` |
| 9 | `stage` | string | NA | 允许按源schema使用NA | 原始记录字段 stage。 | 结合行粒度、状态列和关联键判读。 | `stage06` |
| 10 | `test_object` | string | NA | 允许按源schema使用NA | 原始记录字段 test_object。 | 结合行粒度、状态列和关联键判读。 | `task2_pipeline_benchmark` |
| 11 | `task` | string | NA | 允许按源schema使用NA | 原始记录字段 task。 | 结合行粒度、状态列和关联键判读。 | `Task2` |
| 12 | `step` | string | NA | 允许按源schema使用NA | 原始记录字段 step。 | 结合行粒度、状态列和关联键判读。 | `pipeline_total` |
| 13 | `module` | string | NA | 允许按源schema使用NA | 原始记录字段 module。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 14 | `operator` | string | NA | 允许按源schema使用NA | 原始记录字段 operator。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_OR_NA` |
| 15 | `run_id` | string | NA | 允许按源schema使用NA | 一次正式运行的唯一标识。 | 结合行粒度、状态列和关联键判读。 | `20260815T152000Z_b0478e7_fft_thrust` |
| 16 | `case_id` | string | NA | 允许按源schema使用NA | 共享扰动参数确定的配对case唯一标识。 | 结合行粒度、状态列和关联键判读。 | `aggregate` |
| 17 | `backend` | string | NA | 允许按源schema使用NA | 本行实际使用的实现或FFT后端。 | 结合行粒度、状态列和关联键判读。 | `fft_thrust` |
| 18 | `dtype` | string | NA | 允许按源schema使用NA | 本case实际数据类型。 | 结合行粒度、状态列和关联键判读。 | `FP32` |
| 19 | `scale_id` | string | NA | 允许按源schema使用NA | 输入规模的可读标识。 | 结合行粒度、状态列和关联键判读。 | `aggregate` |
| 20 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `governance` |
| 21 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `b0478e7c5f1ad93e4cf41827d653d191c4a86804` |
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
| 32 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `stage06 Task2 fft_thrust pipeline正式110-case归档，三端逐文件SHA-256一致。` |
