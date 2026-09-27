# CSV字段说明

> 先读同目录README，再按本文件逐文件映射定位schema；列名、列数和顺序以真实表头为准。

- 本目录直接CSV：1个
- 不同schema：1种
- 生成与校验时间：`2026-08-18T10:39:23+08:00`

## 直接CSV到schema映射

| CSV文件 | schema_id | 表头SHA-256 | 列数 | 字段说明 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `folder_manifest.csv` | `SCHEMA_7F4D2CBB2001` | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | [全部列](#schema_7f4d2cbb2001) | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明入口 |
| --- | ---: | --- |
| `00_归档规范` | 1 | [CSV字段说明](./00_归档规范/CSV字段说明.md) |
| `01_总清单与校验` | 21 | [CSV字段说明](./01_总清单与校验/CSV字段说明.md) |
| `02_环境与版本` | 1 | [CSV字段说明](./02_环境与版本/CSV字段说明.md) |
| `03_任务结果` | 1339 | [CSV字段说明](./03_任务结果/CSV字段说明.md) |
| `04_算子结果` | 11672 | [CSV字段说明](./04_算子结果/CSV字段说明.md) |
| `05_异常输入` | 128 | [CSV字段说明](./05_异常输入/CSV字段说明.md) |
| `06_稳定性` | 5008 | [CSV字段说明](./06_稳定性/CSV字段说明.md) |
| `07_内存显存泄漏` | 6 | [CSV字段说明](./07_内存显存泄漏/CSV字段说明.md) |
| `08_FFT后端` | 1 | [CSV字段说明](./08_FFT后端/CSV字段说明.md) |
| `09_性能优化` | 1 | [CSV字段说明](./09_性能优化/CSV字段说明.md) |
| `10_演示系统` | 1 | [CSV字段说明](./10_演示系统/CSV字段说明.md) |
| `11_规范化数据` | 1 | [CSV字段说明](./11_规范化数据/CSV字段说明.md) |
| `12_绘图数据` | 1 | [CSV字段说明](./12_绘图数据/CSV字段说明.md) |
| `13_图表` | 1 | [CSV字段说明](./13_图表/CSV字段说明.md) |
| `14_截图` | 1 | [CSV字段说明](./14_截图/CSV字段说明.md) |
| `15_PPT引用` | 1 | [CSV字段说明](./15_PPT引用/CSV字段说明.md) |
| `16_技术文档引用` | 1 | [CSV字段说明](./16_技术文档引用/CSV字段说明.md) |
| `17_完整日志` | 1 | [CSV字段说明](./17_完整日志/CSV字段说明.md) |
| `90_历史证据` | 4 | [CSV字段说明](./90_历史证据/CSV字段说明.md) |
| `99_隔离与待确认` | 1 | [CSV字段说明](./99_隔离与待确认/CSV字段说明.md) |

## schema完整定义

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：``folder_manifest.csv``
- 编码与格式：UTF-8、RFC 4180逗号分隔；一行对应文件自身约定的最小记录粒度。
- 表头SHA-256：`7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8`

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 判读方法 | 真实示例 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 允许按源schema使用NA | 相对于当前清单约定根目录的文件路径。 | 结合行粒度、状态列和关联键判读。 | `00_归档规范/CSV字段说明.md` |
| 2 | `filename` | string | NA | 允许按源schema使用NA | 文件名。 | 结合行粒度、状态列和关联键判读。 | `CSV字段说明.md` |
| 3 | `artifact_kind` | string | NA | 允许按源schema使用NA | 证据文件或治理文件类别。 | 结合行粒度、状态列和关联键判读。 | `governance` |
| 4 | `file_size` | integer | bytes | 允许按源schema使用NA | 字节数或文件大小。 | 结合行粒度、状态列和关联键判读。 | `8424` |
| 5 | `sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `e3b91657cbbe067172977e7a3752531b258fe55dada09931fa9c125c6b698f92` |
| 6 | `source_host` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `LOCAL_WINDOWS` |
| 7 | `source_container` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 8 | `source_path` | string | NA | 允许按源schema使用NA | 证据来源环境、路径或来源摘要。 | 结合行粒度、状态列和关联键判读。 | `LOCAL_GOVERNANCE` |
| 9 | `stage` | string | NA | 允许按源schema使用NA | 原始记录字段 stage。 | 结合行粒度、状态列和关联键判读。 | `MULTIPLE_OR_NOT_APPLICABLE` |
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
| 20 | `run_type` | string | NA | 允许按源schema使用NA | 运行类型，例如formal、smoke或stability。 | 结合行粒度、状态列和关联键判读。 | `governance` |
| 21 | `git_commit` | string | NA | 允许按源schema使用NA | 运行时源码Git提交SHA。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 22 | `git_dirty` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 23 | `evidence_status` | string | NA | 允许按源schema使用NA | 证据资格或归档状态，不能仅凭非空推断PASS。 | 结合行粒度、状态列和关联键判读。 | `GOVERNANCE` |
| 24 | `formal_eligible` | boolean | NA | 允许按源schema使用NA | 布尔状态；按true或false判读。 | 结合行粒度、状态列和关联键判读。 | `NA` |
| 25 | `related_csv` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 26 | `related_log` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 27 | `related_screenshot` | string | NA | 允许按源schema使用NA | 关联CSV、日志、截图或评分记录。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 28 | `csv_schema_id` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 29 | `csv_column_document` | string | NA | 允许按源schema使用NA | CSV schema、表头或字段说明的追溯信息。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 30 | `csv_header_sha256` | string | NA | 允许按源schema使用NA | SHA-256内容身份摘要，用于一致性核验。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 31 | `csv_column_count` | integer | count | 允许按源schema使用NA | 计数、索引、种子或进程返回码。 | 结合行粒度、状态列和关联键判读。 | `空值` |
| 32 | `notes` | string | NA | 允许按源schema使用NA | 补充说明、限制或NA规则。 | 结合行粒度、状态列和关联键判读。 | `本地生成的归档治理文件。` |
