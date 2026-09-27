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
| [`20260810T103323Z_e6d8c17f237e_stage07_b04_formal_2bd3cd51`](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_2bd3cd51/README.md) | 5 | [CSV字段说明](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_2bd3cd51/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103323Z_e6d8c17f237e_stage07_b04_formal_362bdabd`](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_362bdabd/README.md) | 5 | [CSV字段说明](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_362bdabd/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103323Z_e6d8c17f237e_stage07_b04_formal_888e3b50`](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_888e3b50/README.md) | 5 | [CSV字段说明](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_888e3b50/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103323Z_e6d8c17f237e_stage07_b04_formal_a0b6f16b`](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_a0b6f16b/README.md) | 5 | [CSV字段说明](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_a0b6f16b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103323Z_e6d8c17f237e_stage07_b04_formal_c1185937`](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_c1185937/README.md) | 5 | [CSV字段说明](./20260810T103323Z_e6d8c17f237e_stage07_b04_formal_c1185937/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_084f9a3c`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_084f9a3c/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_084f9a3c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_48b60234`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_48b60234/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_48b60234/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_56a4ba34`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_56a4ba34/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_56a4ba34/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_60807da7`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_60807da7/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_60807da7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_a1098143`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_a1098143/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_a1098143/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_a67d3bf5`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_a67d3bf5/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_a67d3bf5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103324Z_e6d8c17f237e_stage07_b04_formal_f1683fe7`](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_f1683fe7/README.md) | 5 | [CSV字段说明](./20260810T103324Z_e6d8c17f237e_stage07_b04_formal_f1683fe7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103325Z_e6d8c17f237e_stage07_b04_formal_424a433c`](./20260810T103325Z_e6d8c17f237e_stage07_b04_formal_424a433c/README.md) | 5 | [CSV字段说明](./20260810T103325Z_e6d8c17f237e_stage07_b04_formal_424a433c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103325Z_e6d8c17f237e_stage07_b04_formal_78929cf8`](./20260810T103325Z_e6d8c17f237e_stage07_b04_formal_78929cf8/README.md) | 5 | [CSV字段说明](./20260810T103325Z_e6d8c17f237e_stage07_b04_formal_78929cf8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103325Z_e6d8c17f237e_stage07_b04_formal_a7932a7e`](./20260810T103325Z_e6d8c17f237e_stage07_b04_formal_a7932a7e/README.md) | 5 | [CSV字段说明](./20260810T103325Z_e6d8c17f237e_stage07_b04_formal_a7932a7e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103334Z_e6d8c17f237e_stage07_b04_formal_1e57815b`](./20260810T103334Z_e6d8c17f237e_stage07_b04_formal_1e57815b/README.md) | 5 | [CSV字段说明](./20260810T103334Z_e6d8c17f237e_stage07_b04_formal_1e57815b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103334Z_e6d8c17f237e_stage07_b04_formal_32594043`](./20260810T103334Z_e6d8c17f237e_stage07_b04_formal_32594043/README.md) | 5 | [CSV字段说明](./20260810T103334Z_e6d8c17f237e_stage07_b04_formal_32594043/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103334Z_e6d8c17f237e_stage07_b04_formal_6d827fad`](./20260810T103334Z_e6d8c17f237e_stage07_b04_formal_6d827fad/README.md) | 5 | [CSV字段说明](./20260810T103334Z_e6d8c17f237e_stage07_b04_formal_6d827fad/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103335Z_e6d8c17f237e_stage07_b04_formal_37b79289`](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_37b79289/README.md) | 5 | [CSV字段说明](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_37b79289/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103335Z_e6d8c17f237e_stage07_b04_formal_419234b5`](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_419234b5/README.md) | 5 | [CSV字段说明](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_419234b5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103335Z_e6d8c17f237e_stage07_b04_formal_a1543c64`](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_a1543c64/README.md) | 5 | [CSV字段说明](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_a1543c64/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103335Z_e6d8c17f237e_stage07_b04_formal_d5fdf9f1`](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_d5fdf9f1/README.md) | 5 | [CSV字段说明](./20260810T103335Z_e6d8c17f237e_stage07_b04_formal_d5fdf9f1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103336Z_e6d8c17f237e_stage07_b04_formal_17ff2a6b`](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_17ff2a6b/README.md) | 5 | [CSV字段说明](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_17ff2a6b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103336Z_e6d8c17f237e_stage07_b04_formal_3a8d019a`](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_3a8d019a/README.md) | 5 | [CSV字段说明](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_3a8d019a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103336Z_e6d8c17f237e_stage07_b04_formal_529834a9`](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_529834a9/README.md) | 5 | [CSV字段说明](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_529834a9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103336Z_e6d8c17f237e_stage07_b04_formal_5ac9578e`](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_5ac9578e/README.md) | 5 | [CSV字段说明](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_5ac9578e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103336Z_e6d8c17f237e_stage07_b04_formal_7ca7dc6f`](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_7ca7dc6f/README.md) | 5 | [CSV字段说明](./20260810T103336Z_e6d8c17f237e_stage07_b04_formal_7ca7dc6f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103337Z_e6d8c17f237e_stage07_b04_formal_1a58898d`](./20260810T103337Z_e6d8c17f237e_stage07_b04_formal_1a58898d/README.md) | 5 | [CSV字段说明](./20260810T103337Z_e6d8c17f237e_stage07_b04_formal_1a58898d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103337Z_e6d8c17f237e_stage07_b04_formal_63521468`](./20260810T103337Z_e6d8c17f237e_stage07_b04_formal_63521468/README.md) | 5 | [CSV字段说明](./20260810T103337Z_e6d8c17f237e_stage07_b04_formal_63521468/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103337Z_e6d8c17f237e_stage07_b04_formal_8deda8ce`](./20260810T103337Z_e6d8c17f237e_stage07_b04_formal_8deda8ce/README.md) | 5 | [CSV字段说明](./20260810T103337Z_e6d8c17f237e_stage07_b04_formal_8deda8ce/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103350Z_e6d8c17f237e_stage07_b04_formal_26ff5c14`](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_26ff5c14/README.md) | 5 | [CSV字段说明](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_26ff5c14/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103350Z_e6d8c17f237e_stage07_b04_formal_35ca6ed8`](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_35ca6ed8/README.md) | 5 | [CSV字段说明](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_35ca6ed8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103350Z_e6d8c17f237e_stage07_b04_formal_61eba7b8`](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_61eba7b8/README.md) | 5 | [CSV字段说明](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_61eba7b8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103350Z_e6d8c17f237e_stage07_b04_formal_6e55b278`](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_6e55b278/README.md) | 5 | [CSV字段说明](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_6e55b278/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103350Z_e6d8c17f237e_stage07_b04_formal_a6c4b841`](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_a6c4b841/README.md) | 5 | [CSV字段说明](./20260810T103350Z_e6d8c17f237e_stage07_b04_formal_a6c4b841/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103351Z_e6d8c17f237e_stage07_b04_formal_1e3999bd`](./20260810T103351Z_e6d8c17f237e_stage07_b04_formal_1e3999bd/README.md) | 5 | [CSV字段说明](./20260810T103351Z_e6d8c17f237e_stage07_b04_formal_1e3999bd/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103351Z_e6d8c17f237e_stage07_b04_formal_36d8a750`](./20260810T103351Z_e6d8c17f237e_stage07_b04_formal_36d8a750/README.md) | 5 | [CSV字段说明](./20260810T103351Z_e6d8c17f237e_stage07_b04_formal_36d8a750/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103352Z_e6d8c17f237e_stage07_b04_formal_3ea4bbe9`](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_3ea4bbe9/README.md) | 5 | [CSV字段说明](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_3ea4bbe9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103352Z_e6d8c17f237e_stage07_b04_formal_438ea714`](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_438ea714/README.md) | 5 | [CSV字段说明](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_438ea714/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103352Z_e6d8c17f237e_stage07_b04_formal_cd88a90f`](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_cd88a90f/README.md) | 5 | [CSV字段说明](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_cd88a90f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103352Z_e6d8c17f237e_stage07_b04_formal_f0d3e7ca`](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_f0d3e7ca/README.md) | 5 | [CSV字段说明](./20260810T103352Z_e6d8c17f237e_stage07_b04_formal_f0d3e7ca/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103353Z_e6d8c17f237e_stage07_b04_formal_14f10313`](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_14f10313/README.md) | 5 | [CSV字段说明](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_14f10313/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103353Z_e6d8c17f237e_stage07_b04_formal_436e8297`](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_436e8297/README.md) | 5 | [CSV字段说明](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_436e8297/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103353Z_e6d8c17f237e_stage07_b04_formal_89c8178a`](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_89c8178a/README.md) | 5 | [CSV字段说明](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_89c8178a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T103353Z_e6d8c17f237e_stage07_b04_formal_e6623fdc`](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_e6623fdc/README.md) | 5 | [CSV字段说明](./20260810T103353Z_e6d8c17f237e_stage07_b04_formal_e6623fdc/CSV字段说明.md) | 5 | 0 | COVERED |

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
