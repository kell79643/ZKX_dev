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
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000100`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000100/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000100/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000101`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000101/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000101/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000102`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000102/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000102/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000103`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000103/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000103/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000104`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000104/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000104/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000105`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000105/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000105/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000106`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000106/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000106/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000107`](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000107/README.md) | 5 | [CSV字段说明](./20260807T103305Z_ba0a42aa462d_stage08_b02_formal_8b000107/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000108`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000108/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000108/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000109`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000109/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000109/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010a`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010a/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010b`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010b/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010c`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010c/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010d`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010d/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010e`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010e/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010f`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010f/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b00010f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000110`](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000110/README.md) | 5 | [CSV字段说明](./20260807T103306Z_ba0a42aa462d_stage08_b02_formal_8b000110/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000111`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000111/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000111/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000112`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000112/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000112/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000113`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000113/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000113/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000114`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000114/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000114/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000115`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000115/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000115/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000116`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000116/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000116/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000117`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000117/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000117/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000118`](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000118/README.md) | 5 | [CSV字段说明](./20260807T103307Z_ba0a42aa462d_stage08_b02_formal_8b000118/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000119`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000119/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000119/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011a`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011a/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011b`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011b/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011c`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011c/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011d`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011d/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b00011d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000200`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000200/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000200/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000201`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000201/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000201/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000202`](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000202/README.md) | 5 | [CSV字段说明](./20260807T103308Z_ba0a42aa462d_stage08_b02_formal_8b000202/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000203`](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000203/README.md) | 5 | [CSV字段说明](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000203/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000204`](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000204/README.md) | 5 | [CSV字段说明](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000204/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000205`](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000205/README.md) | 5 | [CSV字段说明](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000205/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000206`](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000206/README.md) | 5 | [CSV字段说明](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000206/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000207`](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000207/README.md) | 5 | [CSV字段说明](./20260807T103309Z_ba0a42aa462d_stage08_b02_formal_8b000207/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b000208`](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b000208/README.md) | 5 | [CSV字段说明](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b000208/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b000209`](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b000209/README.md) | 5 | [CSV字段说明](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b000209/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b00020a`](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b00020a/README.md) | 5 | [CSV字段说明](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b00020a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b00020b`](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b00020b/README.md) | 5 | [CSV字段说明](./20260807T103310Z_ba0a42aa462d_stage08_b02_formal_8b00020b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020c`](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020c/README.md) | 5 | [CSV字段说明](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020d`](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020d/README.md) | 5 | [CSV字段说明](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020e`](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020e/README.md) | 5 | [CSV字段说明](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020f`](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020f/README.md) | 5 | [CSV字段说明](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b00020f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b000210`](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b000210/README.md) | 5 | [CSV字段说明](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b000210/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b000211`](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b000211/README.md) | 5 | [CSV字段说明](./20260807T103311Z_ba0a42aa462d_stage08_b02_formal_8b000211/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000212`](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000212/README.md) | 5 | [CSV字段说明](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000212/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000213`](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000213/README.md) | 5 | [CSV字段说明](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000213/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000214`](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000214/README.md) | 5 | [CSV字段说明](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000214/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000215`](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000215/README.md) | 5 | [CSV字段说明](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000215/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000216`](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000216/README.md) | 5 | [CSV字段说明](./20260807T103312Z_ba0a42aa462d_stage08_b02_formal_8b000216/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000217`](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000217/README.md) | 5 | [CSV字段说明](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000217/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000218`](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000218/README.md) | 5 | [CSV字段说明](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000218/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000219`](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000219/README.md) | 5 | [CSV字段说明](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b000219/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021a`](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021a/README.md) | 5 | [CSV字段说明](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021b`](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021b/README.md) | 5 | [CSV字段说明](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021c`](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021c/README.md) | 5 | [CSV字段说明](./20260807T103313Z_ba0a42aa462d_stage08_b02_formal_8b00021c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b00021d`](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b00021d/README.md) | 5 | [CSV字段说明](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b00021d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000300`](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000300/README.md) | 5 | [CSV字段说明](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000300/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000301`](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000301/README.md) | 5 | [CSV字段说明](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000301/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000302`](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000302/README.md) | 5 | [CSV字段说明](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000302/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000303`](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000303/README.md) | 5 | [CSV字段说明](./20260807T103314Z_ba0a42aa462d_stage08_b02_formal_8b000303/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000304`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000304/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000304/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000305`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000305/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000305/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000306`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000306/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000306/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000307`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000307/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000307/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000308`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000308/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000308/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000309`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000309/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b000309/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b00030a`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b00030a/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b00030a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b00030b`](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b00030b/README.md) | 5 | [CSV字段说明](./20260807T103315Z_ba0a42aa462d_stage08_b02_formal_8b00030b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030c`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030c/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030d`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030d/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030e`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030e/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030f`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030f/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b00030f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000310`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000310/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000310/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000311`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000311/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000311/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000312`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000312/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000312/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000313`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000313/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000313/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314`](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314/README.md) | 5 | [CSV字段说明](./20260807T103316Z_ba0a42aa462d_stage08_b02_formal_8b000314/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000315`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000315/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000315/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000316`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000316/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000316/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000317`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000317/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000317/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000318`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000318/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000318/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000319`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000319/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b000319/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031a`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031a/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031b`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031b/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031c`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031c/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031d`](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031d/README.md) | 5 | [CSV字段说明](./20260807T103317Z_ba0a42aa462d_stage08_b02_formal_8b00031d/CSV字段说明.md) | 5 | 0 | COVERED |

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
