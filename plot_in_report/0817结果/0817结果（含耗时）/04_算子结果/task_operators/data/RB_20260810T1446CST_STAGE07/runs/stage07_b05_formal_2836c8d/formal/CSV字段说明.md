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
| [`20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8`](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8/README.md) | 5 | [CSV字段说明](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105057Z_2836c8dc200d_stage07_b05_formal_c7d48895`](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_c7d48895/README.md) | 5 | [CSV字段说明](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_c7d48895/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105057Z_2836c8dc200d_stage07_b05_formal_d783f494`](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_d783f494/README.md) | 5 | [CSV字段说明](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_d783f494/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105057Z_2836c8dc200d_stage07_b05_formal_dcf21822`](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_dcf21822/README.md) | 5 | [CSV字段说明](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_dcf21822/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105057Z_2836c8dc200d_stage07_b05_formal_f8f457e5`](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_f8f457e5/README.md) | 5 | [CSV字段说明](./20260810T105057Z_2836c8dc200d_stage07_b05_formal_f8f457e5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105058Z_2836c8dc200d_stage07_b05_formal_33458f79`](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_33458f79/README.md) | 5 | [CSV字段说明](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_33458f79/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105058Z_2836c8dc200d_stage07_b05_formal_4b6dc00b`](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_4b6dc00b/README.md) | 5 | [CSV字段说明](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_4b6dc00b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105058Z_2836c8dc200d_stage07_b05_formal_76051d68`](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_76051d68/README.md) | 5 | [CSV字段说明](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_76051d68/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105058Z_2836c8dc200d_stage07_b05_formal_cf0a3e3d`](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_cf0a3e3d/README.md) | 5 | [CSV字段说明](./20260810T105058Z_2836c8dc200d_stage07_b05_formal_cf0a3e3d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105059Z_2836c8dc200d_stage07_b05_formal_4b8bcc09`](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_4b8bcc09/README.md) | 5 | [CSV字段说明](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_4b8bcc09/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105059Z_2836c8dc200d_stage07_b05_formal_5e32c370`](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_5e32c370/README.md) | 5 | [CSV字段说明](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_5e32c370/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105059Z_2836c8dc200d_stage07_b05_formal_8fb1cf30`](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_8fb1cf30/README.md) | 5 | [CSV字段说明](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_8fb1cf30/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105059Z_2836c8dc200d_stage07_b05_formal_96dba455`](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_96dba455/README.md) | 5 | [CSV字段说明](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_96dba455/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105059Z_2836c8dc200d_stage07_b05_formal_ca0e566f`](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_ca0e566f/README.md) | 5 | [CSV字段说明](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_ca0e566f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105059Z_2836c8dc200d_stage07_b05_formal_f64e35de`](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_f64e35de/README.md) | 5 | [CSV字段说明](./20260810T105059Z_2836c8dc200d_stage07_b05_formal_f64e35de/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105108Z_2836c8dc200d_stage07_b05_formal_9bdb78f7`](./20260810T105108Z_2836c8dc200d_stage07_b05_formal_9bdb78f7/README.md) | 5 | [CSV字段说明](./20260810T105108Z_2836c8dc200d_stage07_b05_formal_9bdb78f7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105108Z_2836c8dc200d_stage07_b05_formal_d077f0b2`](./20260810T105108Z_2836c8dc200d_stage07_b05_formal_d077f0b2/README.md) | 5 | [CSV字段说明](./20260810T105108Z_2836c8dc200d_stage07_b05_formal_d077f0b2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105108Z_2836c8dc200d_stage07_b05_formal_fd6000dd`](./20260810T105108Z_2836c8dc200d_stage07_b05_formal_fd6000dd/README.md) | 5 | [CSV字段说明](./20260810T105108Z_2836c8dc200d_stage07_b05_formal_fd6000dd/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105109Z_2836c8dc200d_stage07_b05_formal_bebe1bb2`](./20260810T105109Z_2836c8dc200d_stage07_b05_formal_bebe1bb2/README.md) | 5 | [CSV字段说明](./20260810T105109Z_2836c8dc200d_stage07_b05_formal_bebe1bb2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105109Z_2836c8dc200d_stage07_b05_formal_d6498d18`](./20260810T105109Z_2836c8dc200d_stage07_b05_formal_d6498d18/README.md) | 5 | [CSV字段说明](./20260810T105109Z_2836c8dc200d_stage07_b05_formal_d6498d18/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105110Z_2836c8dc200d_stage07_b05_formal_62618a74`](./20260810T105110Z_2836c8dc200d_stage07_b05_formal_62618a74/README.md) | 5 | [CSV字段说明](./20260810T105110Z_2836c8dc200d_stage07_b05_formal_62618a74/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105112Z_2836c8dc200d_stage07_b05_formal_08865e73`](./20260810T105112Z_2836c8dc200d_stage07_b05_formal_08865e73/README.md) | 5 | [CSV字段说明](./20260810T105112Z_2836c8dc200d_stage07_b05_formal_08865e73/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105112Z_2836c8dc200d_stage07_b05_formal_1334d7a0`](./20260810T105112Z_2836c8dc200d_stage07_b05_formal_1334d7a0/README.md) | 5 | [CSV字段说明](./20260810T105112Z_2836c8dc200d_stage07_b05_formal_1334d7a0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105112Z_2836c8dc200d_stage07_b05_formal_8d36c685`](./20260810T105112Z_2836c8dc200d_stage07_b05_formal_8d36c685/README.md) | 5 | [CSV字段说明](./20260810T105112Z_2836c8dc200d_stage07_b05_formal_8d36c685/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105113Z_2836c8dc200d_stage07_b05_formal_82308464`](./20260810T105113Z_2836c8dc200d_stage07_b05_formal_82308464/README.md) | 5 | [CSV字段说明](./20260810T105113Z_2836c8dc200d_stage07_b05_formal_82308464/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105113Z_2836c8dc200d_stage07_b05_formal_ca36b547`](./20260810T105113Z_2836c8dc200d_stage07_b05_formal_ca36b547/README.md) | 5 | [CSV字段说明](./20260810T105113Z_2836c8dc200d_stage07_b05_formal_ca36b547/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105114Z_2836c8dc200d_stage07_b05_formal_c4160d69`](./20260810T105114Z_2836c8dc200d_stage07_b05_formal_c4160d69/README.md) | 5 | [CSV字段说明](./20260810T105114Z_2836c8dc200d_stage07_b05_formal_c4160d69/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105115Z_2836c8dc200d_stage07_b05_formal_a9cb60f5`](./20260810T105115Z_2836c8dc200d_stage07_b05_formal_a9cb60f5/README.md) | 5 | [CSV字段说明](./20260810T105115Z_2836c8dc200d_stage07_b05_formal_a9cb60f5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105115Z_2836c8dc200d_stage07_b05_formal_beb352ef`](./20260810T105115Z_2836c8dc200d_stage07_b05_formal_beb352ef/README.md) | 5 | [CSV字段说明](./20260810T105115Z_2836c8dc200d_stage07_b05_formal_beb352ef/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105115Z_2836c8dc200d_stage07_b05_formal_e6348320`](./20260810T105115Z_2836c8dc200d_stage07_b05_formal_e6348320/README.md) | 5 | [CSV字段说明](./20260810T105115Z_2836c8dc200d_stage07_b05_formal_e6348320/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105125Z_2836c8dc200d_stage07_b05_formal_1c6b7986`](./20260810T105125Z_2836c8dc200d_stage07_b05_formal_1c6b7986/README.md) | 5 | [CSV字段说明](./20260810T105125Z_2836c8dc200d_stage07_b05_formal_1c6b7986/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105125Z_2836c8dc200d_stage07_b05_formal_b188221d`](./20260810T105125Z_2836c8dc200d_stage07_b05_formal_b188221d/README.md) | 5 | [CSV字段说明](./20260810T105125Z_2836c8dc200d_stage07_b05_formal_b188221d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105126Z_2836c8dc200d_stage07_b05_formal_e3c27483`](./20260810T105126Z_2836c8dc200d_stage07_b05_formal_e3c27483/README.md) | 5 | [CSV字段说明](./20260810T105126Z_2836c8dc200d_stage07_b05_formal_e3c27483/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105134Z_2836c8dc200d_stage07_b05_formal_178dffe3`](./20260810T105134Z_2836c8dc200d_stage07_b05_formal_178dffe3/README.md) | 5 | [CSV字段说明](./20260810T105134Z_2836c8dc200d_stage07_b05_formal_178dffe3/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105134Z_2836c8dc200d_stage07_b05_formal_c51cc5da`](./20260810T105134Z_2836c8dc200d_stage07_b05_formal_c51cc5da/README.md) | 5 | [CSV字段说明](./20260810T105134Z_2836c8dc200d_stage07_b05_formal_c51cc5da/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105135Z_2836c8dc200d_stage07_b05_formal_0752c922`](./20260810T105135Z_2836c8dc200d_stage07_b05_formal_0752c922/README.md) | 5 | [CSV字段说明](./20260810T105135Z_2836c8dc200d_stage07_b05_formal_0752c922/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105146Z_2836c8dc200d_stage07_b05_formal_50b827d9`](./20260810T105146Z_2836c8dc200d_stage07_b05_formal_50b827d9/README.md) | 5 | [CSV字段说明](./20260810T105146Z_2836c8dc200d_stage07_b05_formal_50b827d9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105146Z_2836c8dc200d_stage07_b05_formal_b57f0fe6`](./20260810T105146Z_2836c8dc200d_stage07_b05_formal_b57f0fe6/README.md) | 5 | [CSV字段说明](./20260810T105146Z_2836c8dc200d_stage07_b05_formal_b57f0fe6/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105147Z_2836c8dc200d_stage07_b05_formal_452df5cc`](./20260810T105147Z_2836c8dc200d_stage07_b05_formal_452df5cc/README.md) | 5 | [CSV字段说明](./20260810T105147Z_2836c8dc200d_stage07_b05_formal_452df5cc/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105155Z_2836c8dc200d_stage07_b05_formal_2956e124`](./20260810T105155Z_2836c8dc200d_stage07_b05_formal_2956e124/README.md) | 5 | [CSV字段说明](./20260810T105155Z_2836c8dc200d_stage07_b05_formal_2956e124/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105155Z_2836c8dc200d_stage07_b05_formal_b83dc473`](./20260810T105155Z_2836c8dc200d_stage07_b05_formal_b83dc473/README.md) | 5 | [CSV字段说明](./20260810T105155Z_2836c8dc200d_stage07_b05_formal_b83dc473/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105156Z_2836c8dc200d_stage07_b05_formal_5446b8ba`](./20260810T105156Z_2836c8dc200d_stage07_b05_formal_5446b8ba/README.md) | 5 | [CSV字段说明](./20260810T105156Z_2836c8dc200d_stage07_b05_formal_5446b8ba/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105204Z_2836c8dc200d_stage07_b05_formal_0d628023`](./20260810T105204Z_2836c8dc200d_stage07_b05_formal_0d628023/README.md) | 5 | [CSV字段说明](./20260810T105204Z_2836c8dc200d_stage07_b05_formal_0d628023/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105204Z_2836c8dc200d_stage07_b05_formal_e7fea20a`](./20260810T105204Z_2836c8dc200d_stage07_b05_formal_e7fea20a/README.md) | 5 | [CSV字段说明](./20260810T105204Z_2836c8dc200d_stage07_b05_formal_e7fea20a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T105205Z_2836c8dc200d_stage07_b05_formal_f9c94431`](./20260810T105205Z_2836c8dc200d_stage07_b05_formal_f9c94431/README.md) | 5 | [CSV字段说明](./20260810T105205Z_2836c8dc200d_stage07_b05_formal_f9c94431/CSV字段说明.md) | 5 | 0 | COVERED |

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
