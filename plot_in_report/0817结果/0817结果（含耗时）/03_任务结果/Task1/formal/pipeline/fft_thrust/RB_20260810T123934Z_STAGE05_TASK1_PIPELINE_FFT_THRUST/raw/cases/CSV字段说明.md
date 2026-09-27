# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：1个
- 不同schema：1种
- 生成与校验时间：`2026-08-17T16:53:36+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明入口 |
| --- | ---: | --- |
| `task1_s01_baseline` | 6 | [CSV字段说明](./task1_s01_baseline/CSV字段说明.md) |
| `task1_s01_doppler_neg` | 6 | [CSV字段说明](./task1_s01_doppler_neg/CSV字段说明.md) |
| `task1_s01_doppler_pos` | 6 | [CSV字段说明](./task1_s01_doppler_pos/CSV字段说明.md) |
| `task1_s01_far` | 6 | [CSV字段说明](./task1_s01_far/CSV字段说明.md) |
| `task1_s01_near` | 6 | [CSV字段说明](./task1_s01_near/CSV字段说明.md) |
| `task1_s01_noiseless` | 6 | [CSV字段说明](./task1_s01_noiseless/CSV字段说明.md) |
| `task1_s01_noisy` | 6 | [CSV字段说明](./task1_s01_noisy/CSV字段说明.md) |
| `task1_s01_quiet` | 6 | [CSV字段说明](./task1_s01_quiet/CSV字段说明.md) |
| `task1_s01_strong` | 6 | [CSV字段说明](./task1_s01_strong/CSV字段说明.md) |
| `task1_s01_weak` | 6 | [CSV字段说明](./task1_s01_weak/CSV字段说明.md) |
| `task1_s02_baseline` | 6 | [CSV字段说明](./task1_s02_baseline/CSV字段说明.md) |
| `task1_s02_doppler_neg` | 6 | [CSV字段说明](./task1_s02_doppler_neg/CSV字段说明.md) |
| `task1_s02_doppler_pos` | 6 | [CSV字段说明](./task1_s02_doppler_pos/CSV字段说明.md) |
| `task1_s02_far` | 6 | [CSV字段说明](./task1_s02_far/CSV字段说明.md) |
| `task1_s02_near` | 6 | [CSV字段说明](./task1_s02_near/CSV字段说明.md) |
| `task1_s02_noiseless` | 6 | [CSV字段说明](./task1_s02_noiseless/CSV字段说明.md) |
| `task1_s02_noisy` | 6 | [CSV字段说明](./task1_s02_noisy/CSV字段说明.md) |
| `task1_s02_quiet` | 6 | [CSV字段说明](./task1_s02_quiet/CSV字段说明.md) |
| `task1_s02_strong` | 6 | [CSV字段说明](./task1_s02_strong/CSV字段说明.md) |
| `task1_s02_weak` | 6 | [CSV字段说明](./task1_s02_weak/CSV字段说明.md) |
| `task1_s03_baseline` | 6 | [CSV字段说明](./task1_s03_baseline/CSV字段说明.md) |
| `task1_s03_doppler_neg` | 6 | [CSV字段说明](./task1_s03_doppler_neg/CSV字段说明.md) |
| `task1_s03_doppler_pos` | 6 | [CSV字段说明](./task1_s03_doppler_pos/CSV字段说明.md) |
| `task1_s03_far` | 6 | [CSV字段说明](./task1_s03_far/CSV字段说明.md) |
| `task1_s03_near` | 6 | [CSV字段说明](./task1_s03_near/CSV字段说明.md) |
| `task1_s03_noiseless` | 6 | [CSV字段说明](./task1_s03_noiseless/CSV字段说明.md) |
| `task1_s03_noisy` | 6 | [CSV字段说明](./task1_s03_noisy/CSV字段说明.md) |
| `task1_s03_quiet` | 6 | [CSV字段说明](./task1_s03_quiet/CSV字段说明.md) |
| `task1_s03_strong` | 6 | [CSV字段说明](./task1_s03_strong/CSV字段说明.md) |
| `task1_s03_weak` | 6 | [CSV字段说明](./task1_s03_weak/CSV字段说明.md) |
| `task1_s04_baseline` | 6 | [CSV字段说明](./task1_s04_baseline/CSV字段说明.md) |
| `task1_s04_doppler_neg` | 6 | [CSV字段说明](./task1_s04_doppler_neg/CSV字段说明.md) |
| `task1_s04_doppler_pos` | 6 | [CSV字段说明](./task1_s04_doppler_pos/CSV字段说明.md) |
| `task1_s04_far` | 6 | [CSV字段说明](./task1_s04_far/CSV字段说明.md) |
| `task1_s04_near` | 6 | [CSV字段说明](./task1_s04_near/CSV字段说明.md) |
| `task1_s04_noiseless` | 6 | [CSV字段说明](./task1_s04_noiseless/CSV字段说明.md) |
| `task1_s04_noisy` | 6 | [CSV字段说明](./task1_s04_noisy/CSV字段说明.md) |
| `task1_s04_quiet` | 6 | [CSV字段说明](./task1_s04_quiet/CSV字段说明.md) |
| `task1_s04_strong` | 6 | [CSV字段说明](./task1_s04_strong/CSV字段说明.md) |
| `task1_s04_weak` | 6 | [CSV字段说明](./task1_s04_weak/CSV字段说明.md) |
| `task1_s05_baseline` | 6 | [CSV字段说明](./task1_s05_baseline/CSV字段说明.md) |
| `task1_s05_doppler_neg` | 6 | [CSV字段说明](./task1_s05_doppler_neg/CSV字段说明.md) |
| `task1_s05_doppler_pos` | 6 | [CSV字段说明](./task1_s05_doppler_pos/CSV字段说明.md) |
| `task1_s05_far` | 6 | [CSV字段说明](./task1_s05_far/CSV字段说明.md) |
| `task1_s05_near` | 6 | [CSV字段说明](./task1_s05_near/CSV字段说明.md) |
| `task1_s05_noiseless` | 6 | [CSV字段说明](./task1_s05_noiseless/CSV字段说明.md) |
| `task1_s05_noisy` | 6 | [CSV字段说明](./task1_s05_noisy/CSV字段说明.md) |
| `task1_s05_quiet` | 6 | [CSV字段说明](./task1_s05_quiet/CSV字段说明.md) |
| `task1_s05_strong` | 6 | [CSV字段说明](./task1_s05_strong/CSV字段说明.md) |
| `task1_s05_weak` | 6 | [CSV字段说明](./task1_s05_weak/CSV字段说明.md) |
| `task1_s06_baseline` | 6 | [CSV字段说明](./task1_s06_baseline/CSV字段说明.md) |
| `task1_s06_doppler_neg` | 6 | [CSV字段说明](./task1_s06_doppler_neg/CSV字段说明.md) |
| `task1_s06_doppler_pos` | 6 | [CSV字段说明](./task1_s06_doppler_pos/CSV字段说明.md) |
| `task1_s06_far` | 6 | [CSV字段说明](./task1_s06_far/CSV字段说明.md) |
| `task1_s06_near` | 6 | [CSV字段说明](./task1_s06_near/CSV字段说明.md) |
| `task1_s06_noiseless` | 6 | [CSV字段说明](./task1_s06_noiseless/CSV字段说明.md) |
| `task1_s06_noisy` | 6 | [CSV字段说明](./task1_s06_noisy/CSV字段说明.md) |
| `task1_s06_quiet` | 6 | [CSV字段说明](./task1_s06_quiet/CSV字段说明.md) |
| `task1_s06_strong` | 6 | [CSV字段说明](./task1_s06_strong/CSV字段说明.md) |
| `task1_s06_weak` | 6 | [CSV字段说明](./task1_s06_weak/CSV字段说明.md) |
| `task1_s07_baseline` | 6 | [CSV字段说明](./task1_s07_baseline/CSV字段说明.md) |
| `task1_s07_doppler_neg` | 6 | [CSV字段说明](./task1_s07_doppler_neg/CSV字段说明.md) |
| `task1_s07_doppler_pos` | 6 | [CSV字段说明](./task1_s07_doppler_pos/CSV字段说明.md) |
| `task1_s07_far` | 6 | [CSV字段说明](./task1_s07_far/CSV字段说明.md) |
| `task1_s07_near` | 6 | [CSV字段说明](./task1_s07_near/CSV字段说明.md) |
| `task1_s07_noiseless` | 6 | [CSV字段说明](./task1_s07_noiseless/CSV字段说明.md) |
| `task1_s07_noisy` | 6 | [CSV字段说明](./task1_s07_noisy/CSV字段说明.md) |
| `task1_s07_quiet` | 6 | [CSV字段说明](./task1_s07_quiet/CSV字段说明.md) |
| `task1_s07_strong` | 6 | [CSV字段说明](./task1_s07_strong/CSV字段说明.md) |
| `task1_s07_weak` | 6 | [CSV字段说明](./task1_s07_weak/CSV字段说明.md) |
| `task1_s08_baseline` | 6 | [CSV字段说明](./task1_s08_baseline/CSV字段说明.md) |
| `task1_s08_doppler_neg` | 6 | [CSV字段说明](./task1_s08_doppler_neg/CSV字段说明.md) |
| `task1_s08_doppler_pos` | 6 | [CSV字段说明](./task1_s08_doppler_pos/CSV字段说明.md) |
| `task1_s08_far` | 6 | [CSV字段说明](./task1_s08_far/CSV字段说明.md) |
| `task1_s08_near` | 6 | [CSV字段说明](./task1_s08_near/CSV字段说明.md) |
| `task1_s08_noiseless` | 6 | [CSV字段说明](./task1_s08_noiseless/CSV字段说明.md) |
| `task1_s08_noisy` | 6 | [CSV字段说明](./task1_s08_noisy/CSV字段说明.md) |
| `task1_s08_quiet` | 6 | [CSV字段说明](./task1_s08_quiet/CSV字段说明.md) |
| `task1_s08_strong` | 6 | [CSV字段说明](./task1_s08_strong/CSV字段说明.md) |
| `task1_s08_weak` | 6 | [CSV字段说明](./task1_s08_weak/CSV字段说明.md) |
| `task1_s09_baseline` | 6 | [CSV字段说明](./task1_s09_baseline/CSV字段说明.md) |
| `task1_s09_doppler_neg` | 6 | [CSV字段说明](./task1_s09_doppler_neg/CSV字段说明.md) |
| `task1_s09_doppler_pos` | 6 | [CSV字段说明](./task1_s09_doppler_pos/CSV字段说明.md) |
| `task1_s09_far` | 6 | [CSV字段说明](./task1_s09_far/CSV字段说明.md) |
| `task1_s09_near` | 6 | [CSV字段说明](./task1_s09_near/CSV字段说明.md) |
| `task1_s09_noiseless` | 6 | [CSV字段说明](./task1_s09_noiseless/CSV字段说明.md) |
| `task1_s09_noisy` | 6 | [CSV字段说明](./task1_s09_noisy/CSV字段说明.md) |
| `task1_s09_quiet` | 6 | [CSV字段说明](./task1_s09_quiet/CSV字段说明.md) |
| `task1_s09_strong` | 6 | [CSV字段说明](./task1_s09_strong/CSV字段说明.md) |
| `task1_s09_weak` | 6 | [CSV字段说明](./task1_s09_weak/CSV字段说明.md) |
| `task1_s10_baseline` | 6 | [CSV字段说明](./task1_s10_baseline/CSV字段说明.md) |
| `task1_s10_doppler_neg` | 6 | [CSV字段说明](./task1_s10_doppler_neg/CSV字段说明.md) |
| `task1_s10_doppler_pos` | 6 | [CSV字段说明](./task1_s10_doppler_pos/CSV字段说明.md) |
| `task1_s10_far` | 6 | [CSV字段说明](./task1_s10_far/CSV字段说明.md) |
| `task1_s10_near` | 6 | [CSV字段说明](./task1_s10_near/CSV字段说明.md) |
| `task1_s10_noiseless` | 6 | [CSV字段说明](./task1_s10_noiseless/CSV字段说明.md) |
| `task1_s10_noisy` | 6 | [CSV字段说明](./task1_s10_noisy/CSV字段说明.md) |
| `task1_s10_quiet` | 6 | [CSV字段说明](./task1_s10_quiet/CSV字段说明.md) |
| `task1_s10_strong` | 6 | [CSV字段说明](./task1_s10_strong/CSV字段说明.md) |
| `task1_s10_weak` | 6 | [CSV字段说明](./task1_s10_weak/CSV字段说明.md) |
| `task1_s11_baseline` | 6 | [CSV字段说明](./task1_s11_baseline/CSV字段说明.md) |
| `task1_s11_doppler_neg` | 6 | [CSV字段说明](./task1_s11_doppler_neg/CSV字段说明.md) |
| `task1_s11_doppler_pos` | 6 | [CSV字段说明](./task1_s11_doppler_pos/CSV字段说明.md) |
| `task1_s11_far` | 6 | [CSV字段说明](./task1_s11_far/CSV字段说明.md) |
| `task1_s11_near` | 6 | [CSV字段说明](./task1_s11_near/CSV字段说明.md) |
| `task1_s11_noiseless` | 6 | [CSV字段说明](./task1_s11_noiseless/CSV字段说明.md) |
| `task1_s11_noisy` | 6 | [CSV字段说明](./task1_s11_noisy/CSV字段说明.md) |
| `task1_s11_quiet` | 6 | [CSV字段说明](./task1_s11_quiet/CSV字段说明.md) |
| `task1_s11_strong` | 6 | [CSV字段说明](./task1_s11_strong/CSV字段说明.md) |
| `task1_s11_weak` | 6 | [CSV字段说明](./task1_s11_weak/CSV字段说明.md) |

## schema完整定义

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
