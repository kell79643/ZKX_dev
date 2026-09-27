# CSV字段说明

> 阅读顺序：先读同目录`README.md`了解文件身份和用途，再用本文件查CSV行粒度、主键、关联键和全部列含义。

## 覆盖结论

- 本目录直接CSV：1个（含治理文件`folder_manifest.csv`）
- 本目录直接CSV已映射：1个
- 本目录直接CSV未映射：0个
- 本目录不同schema：1种
- 生成与校验时间：`2026-08-14T17:33:09+08:00`
- 校验规则：真实列名、列数、顺序及规范化表头SHA-256必须与schema一致。

## 直接CSV到schema映射

| CSV文件 | schema_id | 字段说明 | 表头SHA-256 | 列数 | 一行代表 | 主键 | 配对键 | 来源/生成方式 | 校验状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明 | 已说明 | 未说明 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| [`20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200000`](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200000/README.md) | 6 | [CSV字段说明](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200000/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200001`](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200001/README.md) | 6 | [CSV字段说明](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200001/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200002`](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200002/README.md) | 6 | [CSV字段说明](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200002/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200003`](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200003/README.md) | 6 | [CSV字段说明](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200003/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200004`](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200004/README.md) | 6 | [CSV字段说明](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200004/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200005`](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200005/README.md) | 6 | [CSV字段说明](./20260810T032030Z_47ebe13410d3_stage08_b02_formal_88200005/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200006`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200006/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200006/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200007`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200007/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200007/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200008`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200008/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200008/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200009`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200009/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_88200009/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000a`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000a/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000a/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000b`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000b/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000b/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000c`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000c/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000c/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000d`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000d/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000d/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000e`](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000e/README.md) | 6 | [CSV字段说明](./20260810T032031Z_47ebe13410d3_stage08_b02_formal_8820000e/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_8820000f`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_8820000f/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_8820000f/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200010`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200010/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200010/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200011`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200011/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200011/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200012`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200012/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200012/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200013`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200013/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200013/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200014`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200014/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200014/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200015`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200015/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200015/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200016`](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200016/README.md) | 6 | [CSV字段说明](./20260810T032032Z_47ebe13410d3_stage08_b02_formal_88200016/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200017`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200017/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200017/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200018`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200018/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200018/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200019`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200019/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_88200019/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001a`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001a/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001a/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001b`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001b/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001b/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001c`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001c/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001c/CSV字段说明.md) | 6 | 0 | COVERED |
| [`20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001d`](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001d/README.md) | 6 | [CSV字段说明](./20260810T032033Z_47ebe13410d3_stage08_b02_formal_8820001d/CSV字段说明.md) | 6 | 0 | COVERED |

## schema完整定义

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

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../../../../../../README.md)
- [全局CSV字段语义总表](../../../../../../../00_归档规范/CSV字段语义总表.md)
