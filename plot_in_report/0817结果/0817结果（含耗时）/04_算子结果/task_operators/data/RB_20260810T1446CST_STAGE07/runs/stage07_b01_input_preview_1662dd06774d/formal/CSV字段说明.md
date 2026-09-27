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
| [`20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18`](./20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18/README.md) | 5 | [CSV字段说明](./20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080820Z_1662dd06774d_stage07_b01_formal_7b80cbb4`](./20260810T080820Z_1662dd06774d_stage07_b01_formal_7b80cbb4/README.md) | 5 | [CSV字段说明](./20260810T080820Z_1662dd06774d_stage07_b01_formal_7b80cbb4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080820Z_1662dd06774d_stage07_b01_formal_8e5e5ded`](./20260810T080820Z_1662dd06774d_stage07_b01_formal_8e5e5ded/README.md) | 5 | [CSV字段说明](./20260810T080820Z_1662dd06774d_stage07_b01_formal_8e5e5ded/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080825Z_1662dd06774d_stage07_b01_formal_2a8442f8`](./20260810T080825Z_1662dd06774d_stage07_b01_formal_2a8442f8/README.md) | 5 | [CSV字段说明](./20260810T080825Z_1662dd06774d_stage07_b01_formal_2a8442f8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080825Z_1662dd06774d_stage07_b01_formal_444065a9`](./20260810T080825Z_1662dd06774d_stage07_b01_formal_444065a9/README.md) | 5 | [CSV字段说明](./20260810T080825Z_1662dd06774d_stage07_b01_formal_444065a9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080826Z_1662dd06774d_stage07_b01_formal_2d27644c`](./20260810T080826Z_1662dd06774d_stage07_b01_formal_2d27644c/README.md) | 5 | [CSV字段说明](./20260810T080826Z_1662dd06774d_stage07_b01_formal_2d27644c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080838Z_1662dd06774d_stage07_b01_formal_422397e3`](./20260810T080838Z_1662dd06774d_stage07_b01_formal_422397e3/README.md) | 5 | [CSV字段说明](./20260810T080838Z_1662dd06774d_stage07_b01_formal_422397e3/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080838Z_1662dd06774d_stage07_b01_formal_ab40795a`](./20260810T080838Z_1662dd06774d_stage07_b01_formal_ab40795a/README.md) | 5 | [CSV字段说明](./20260810T080838Z_1662dd06774d_stage07_b01_formal_ab40795a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080839Z_1662dd06774d_stage07_b01_formal_2b1f319d`](./20260810T080839Z_1662dd06774d_stage07_b01_formal_2b1f319d/README.md) | 5 | [CSV字段说明](./20260810T080839Z_1662dd06774d_stage07_b01_formal_2b1f319d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080843Z_1662dd06774d_stage07_b01_formal_cec35362`](./20260810T080843Z_1662dd06774d_stage07_b01_formal_cec35362/README.md) | 5 | [CSV字段说明](./20260810T080843Z_1662dd06774d_stage07_b01_formal_cec35362/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080844Z_1662dd06774d_stage07_b01_formal_98951c62`](./20260810T080844Z_1662dd06774d_stage07_b01_formal_98951c62/README.md) | 5 | [CSV字段说明](./20260810T080844Z_1662dd06774d_stage07_b01_formal_98951c62/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080844Z_1662dd06774d_stage07_b01_formal_c507704b`](./20260810T080844Z_1662dd06774d_stage07_b01_formal_c507704b/README.md) | 5 | [CSV字段说明](./20260810T080844Z_1662dd06774d_stage07_b01_formal_c507704b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080849Z_1662dd06774d_stage07_b01_formal_58bf76ad`](./20260810T080849Z_1662dd06774d_stage07_b01_formal_58bf76ad/README.md) | 5 | [CSV字段说明](./20260810T080849Z_1662dd06774d_stage07_b01_formal_58bf76ad/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080849Z_1662dd06774d_stage07_b01_formal_ea37fb48`](./20260810T080849Z_1662dd06774d_stage07_b01_formal_ea37fb48/README.md) | 5 | [CSV字段说明](./20260810T080849Z_1662dd06774d_stage07_b01_formal_ea37fb48/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T080850Z_1662dd06774d_stage07_b01_formal_ff8ba0a1`](./20260810T080850Z_1662dd06774d_stage07_b01_formal_ff8ba0a1/README.md) | 5 | [CSV字段说明](./20260810T080850Z_1662dd06774d_stage07_b01_formal_ff8ba0a1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081006Z_1662dd06774d_stage07_b01_formal_29e3eb18`](./20260810T081006Z_1662dd06774d_stage07_b01_formal_29e3eb18/README.md) | 5 | [CSV字段说明](./20260810T081006Z_1662dd06774d_stage07_b01_formal_29e3eb18/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081006Z_1662dd06774d_stage07_b01_formal_8e5e5ded`](./20260810T081006Z_1662dd06774d_stage07_b01_formal_8e5e5ded/README.md) | 5 | [CSV字段说明](./20260810T081006Z_1662dd06774d_stage07_b01_formal_8e5e5ded/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081007Z_1662dd06774d_stage07_b01_formal_7b80cbb4`](./20260810T081007Z_1662dd06774d_stage07_b01_formal_7b80cbb4/README.md) | 5 | [CSV字段说明](./20260810T081007Z_1662dd06774d_stage07_b01_formal_7b80cbb4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081012Z_1662dd06774d_stage07_b01_formal_2a8442f8`](./20260810T081012Z_1662dd06774d_stage07_b01_formal_2a8442f8/README.md) | 5 | [CSV字段说明](./20260810T081012Z_1662dd06774d_stage07_b01_formal_2a8442f8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081012Z_1662dd06774d_stage07_b01_formal_444065a9`](./20260810T081012Z_1662dd06774d_stage07_b01_formal_444065a9/README.md) | 5 | [CSV字段说明](./20260810T081012Z_1662dd06774d_stage07_b01_formal_444065a9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081013Z_1662dd06774d_stage07_b01_formal_2d27644c`](./20260810T081013Z_1662dd06774d_stage07_b01_formal_2d27644c/README.md) | 5 | [CSV字段说明](./20260810T081013Z_1662dd06774d_stage07_b01_formal_2d27644c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081024Z_1662dd06774d_stage07_b01_formal_ab40795a`](./20260810T081024Z_1662dd06774d_stage07_b01_formal_ab40795a/README.md) | 5 | [CSV字段说明](./20260810T081024Z_1662dd06774d_stage07_b01_formal_ab40795a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081025Z_1662dd06774d_stage07_b01_formal_2b1f319d`](./20260810T081025Z_1662dd06774d_stage07_b01_formal_2b1f319d/README.md) | 5 | [CSV字段说明](./20260810T081025Z_1662dd06774d_stage07_b01_formal_2b1f319d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081025Z_1662dd06774d_stage07_b01_formal_422397e3`](./20260810T081025Z_1662dd06774d_stage07_b01_formal_422397e3/README.md) | 5 | [CSV字段说明](./20260810T081025Z_1662dd06774d_stage07_b01_formal_422397e3/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081030Z_1662dd06774d_stage07_b01_formal_c507704b`](./20260810T081030Z_1662dd06774d_stage07_b01_formal_c507704b/README.md) | 5 | [CSV字段说明](./20260810T081030Z_1662dd06774d_stage07_b01_formal_c507704b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081030Z_1662dd06774d_stage07_b01_formal_cec35362`](./20260810T081030Z_1662dd06774d_stage07_b01_formal_cec35362/README.md) | 5 | [CSV字段说明](./20260810T081030Z_1662dd06774d_stage07_b01_formal_cec35362/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081031Z_1662dd06774d_stage07_b01_formal_98951c62`](./20260810T081031Z_1662dd06774d_stage07_b01_formal_98951c62/README.md) | 5 | [CSV字段说明](./20260810T081031Z_1662dd06774d_stage07_b01_formal_98951c62/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081036Z_1662dd06774d_stage07_b01_formal_58bf76ad`](./20260810T081036Z_1662dd06774d_stage07_b01_formal_58bf76ad/README.md) | 5 | [CSV字段说明](./20260810T081036Z_1662dd06774d_stage07_b01_formal_58bf76ad/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081036Z_1662dd06774d_stage07_b01_formal_ea37fb48`](./20260810T081036Z_1662dd06774d_stage07_b01_formal_ea37fb48/README.md) | 5 | [CSV字段说明](./20260810T081036Z_1662dd06774d_stage07_b01_formal_ea37fb48/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081036Z_1662dd06774d_stage07_b01_formal_ff8ba0a1`](./20260810T081036Z_1662dd06774d_stage07_b01_formal_ff8ba0a1/README.md) | 5 | [CSV字段说明](./20260810T081036Z_1662dd06774d_stage07_b01_formal_ff8ba0a1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081041Z_1662dd06774d_stage07_b01_formal_e663f54b`](./20260810T081041Z_1662dd06774d_stage07_b01_formal_e663f54b/README.md) | 5 | [CSV字段说明](./20260810T081041Z_1662dd06774d_stage07_b01_formal_e663f54b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081042Z_1662dd06774d_stage07_b01_formal_269661da`](./20260810T081042Z_1662dd06774d_stage07_b01_formal_269661da/README.md) | 5 | [CSV字段说明](./20260810T081042Z_1662dd06774d_stage07_b01_formal_269661da/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081042Z_1662dd06774d_stage07_b01_formal_326294ba`](./20260810T081042Z_1662dd06774d_stage07_b01_formal_326294ba/README.md) | 5 | [CSV字段说明](./20260810T081042Z_1662dd06774d_stage07_b01_formal_326294ba/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081057Z_1662dd06774d_stage07_b01_formal_393e7bd8`](./20260810T081057Z_1662dd06774d_stage07_b01_formal_393e7bd8/README.md) | 5 | [CSV字段说明](./20260810T081057Z_1662dd06774d_stage07_b01_formal_393e7bd8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081057Z_1662dd06774d_stage07_b01_formal_9b9c7a9a`](./20260810T081057Z_1662dd06774d_stage07_b01_formal_9b9c7a9a/README.md) | 5 | [CSV字段说明](./20260810T081057Z_1662dd06774d_stage07_b01_formal_9b9c7a9a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081058Z_1662dd06774d_stage07_b01_formal_61d2c1e9`](./20260810T081058Z_1662dd06774d_stage07_b01_formal_61d2c1e9/README.md) | 5 | [CSV字段说明](./20260810T081058Z_1662dd06774d_stage07_b01_formal_61d2c1e9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081125Z_1662dd06774d_stage07_b01_formal_8baaa211`](./20260810T081125Z_1662dd06774d_stage07_b01_formal_8baaa211/README.md) | 5 | [CSV字段说明](./20260810T081125Z_1662dd06774d_stage07_b01_formal_8baaa211/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081125Z_1662dd06774d_stage07_b01_formal_e55b7d82`](./20260810T081125Z_1662dd06774d_stage07_b01_formal_e55b7d82/README.md) | 5 | [CSV字段说明](./20260810T081125Z_1662dd06774d_stage07_b01_formal_e55b7d82/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081126Z_1662dd06774d_stage07_b01_formal_8f133ff4`](./20260810T081126Z_1662dd06774d_stage07_b01_formal_8f133ff4/README.md) | 5 | [CSV字段说明](./20260810T081126Z_1662dd06774d_stage07_b01_formal_8f133ff4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081141Z_1662dd06774d_stage07_b01_formal_2f0c42d0`](./20260810T081141Z_1662dd06774d_stage07_b01_formal_2f0c42d0/README.md) | 5 | [CSV字段说明](./20260810T081141Z_1662dd06774d_stage07_b01_formal_2f0c42d0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081141Z_1662dd06774d_stage07_b01_formal_446abe7b`](./20260810T081141Z_1662dd06774d_stage07_b01_formal_446abe7b/README.md) | 5 | [CSV字段说明](./20260810T081141Z_1662dd06774d_stage07_b01_formal_446abe7b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081141Z_1662dd06774d_stage07_b01_formal_9bbb2712`](./20260810T081141Z_1662dd06774d_stage07_b01_formal_9bbb2712/README.md) | 5 | [CSV字段说明](./20260810T081141Z_1662dd06774d_stage07_b01_formal_9bbb2712/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081157Z_1662dd06774d_stage07_b01_formal_74a94911`](./20260810T081157Z_1662dd06774d_stage07_b01_formal_74a94911/README.md) | 5 | [CSV字段说明](./20260810T081157Z_1662dd06774d_stage07_b01_formal_74a94911/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081157Z_1662dd06774d_stage07_b01_formal_7fe111e2`](./20260810T081157Z_1662dd06774d_stage07_b01_formal_7fe111e2/README.md) | 5 | [CSV字段说明](./20260810T081157Z_1662dd06774d_stage07_b01_formal_7fe111e2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081158Z_1662dd06774d_stage07_b01_formal_f381cb24`](./20260810T081158Z_1662dd06774d_stage07_b01_formal_f381cb24/README.md) | 5 | [CSV字段说明](./20260810T081158Z_1662dd06774d_stage07_b01_formal_f381cb24/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081214Z_1662dd06774d_stage07_b01_formal_8c5d72a0`](./20260810T081214Z_1662dd06774d_stage07_b01_formal_8c5d72a0/README.md) | 5 | [CSV字段说明](./20260810T081214Z_1662dd06774d_stage07_b01_formal_8c5d72a0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081214Z_1662dd06774d_stage07_b01_formal_f1b6b137`](./20260810T081214Z_1662dd06774d_stage07_b01_formal_f1b6b137/README.md) | 5 | [CSV字段说明](./20260810T081214Z_1662dd06774d_stage07_b01_formal_f1b6b137/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081215Z_1662dd06774d_stage07_b01_formal_eab6b0f5`](./20260810T081215Z_1662dd06774d_stage07_b01_formal_eab6b0f5/README.md) | 5 | [CSV字段说明](./20260810T081215Z_1662dd06774d_stage07_b01_formal_eab6b0f5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081221Z_1662dd06774d_stage07_b01_formal_0c6290f1`](./20260810T081221Z_1662dd06774d_stage07_b01_formal_0c6290f1/README.md) | 5 | [CSV字段说明](./20260810T081221Z_1662dd06774d_stage07_b01_formal_0c6290f1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081221Z_1662dd06774d_stage07_b01_formal_17fae2a2`](./20260810T081221Z_1662dd06774d_stage07_b01_formal_17fae2a2/README.md) | 5 | [CSV字段说明](./20260810T081221Z_1662dd06774d_stage07_b01_formal_17fae2a2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081222Z_1662dd06774d_stage07_b01_formal_0271eef5`](./20260810T081222Z_1662dd06774d_stage07_b01_formal_0271eef5/README.md) | 5 | [CSV字段说明](./20260810T081222Z_1662dd06774d_stage07_b01_formal_0271eef5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081229Z_1662dd06774d_stage07_b01_formal_d8d7005c`](./20260810T081229Z_1662dd06774d_stage07_b01_formal_d8d7005c/README.md) | 5 | [CSV字段说明](./20260810T081229Z_1662dd06774d_stage07_b01_formal_d8d7005c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081229Z_1662dd06774d_stage07_b01_formal_e1d69246`](./20260810T081229Z_1662dd06774d_stage07_b01_formal_e1d69246/README.md) | 5 | [CSV字段说明](./20260810T081229Z_1662dd06774d_stage07_b01_formal_e1d69246/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081230Z_1662dd06774d_stage07_b01_formal_d83feca6`](./20260810T081230Z_1662dd06774d_stage07_b01_formal_d83feca6/README.md) | 5 | [CSV字段说明](./20260810T081230Z_1662dd06774d_stage07_b01_formal_d83feca6/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081236Z_1662dd06774d_stage07_b01_formal_ef3e4066`](./20260810T081236Z_1662dd06774d_stage07_b01_formal_ef3e4066/README.md) | 5 | [CSV字段说明](./20260810T081236Z_1662dd06774d_stage07_b01_formal_ef3e4066/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081237Z_1662dd06774d_stage07_b01_formal_ab42a7e0`](./20260810T081237Z_1662dd06774d_stage07_b01_formal_ab42a7e0/README.md) | 5 | [CSV字段说明](./20260810T081237Z_1662dd06774d_stage07_b01_formal_ab42a7e0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081238Z_1662dd06774d_stage07_b01_formal_4cdf74c7`](./20260810T081238Z_1662dd06774d_stage07_b01_formal_4cdf74c7/README.md) | 5 | [CSV字段说明](./20260810T081238Z_1662dd06774d_stage07_b01_formal_4cdf74c7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081244Z_1662dd06774d_stage07_b01_formal_694ab2d0`](./20260810T081244Z_1662dd06774d_stage07_b01_formal_694ab2d0/README.md) | 5 | [CSV字段说明](./20260810T081244Z_1662dd06774d_stage07_b01_formal_694ab2d0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081244Z_1662dd06774d_stage07_b01_formal_e93b848a`](./20260810T081244Z_1662dd06774d_stage07_b01_formal_e93b848a/README.md) | 5 | [CSV字段说明](./20260810T081244Z_1662dd06774d_stage07_b01_formal_e93b848a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081245Z_1662dd06774d_stage07_b01_formal_ad409739`](./20260810T081245Z_1662dd06774d_stage07_b01_formal_ad409739/README.md) | 5 | [CSV字段说明](./20260810T081245Z_1662dd06774d_stage07_b01_formal_ad409739/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081252Z_1662dd06774d_stage07_b01_formal_0fc26f06`](./20260810T081252Z_1662dd06774d_stage07_b01_formal_0fc26f06/README.md) | 5 | [CSV字段说明](./20260810T081252Z_1662dd06774d_stage07_b01_formal_0fc26f06/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081252Z_1662dd06774d_stage07_b01_formal_36254ed8`](./20260810T081252Z_1662dd06774d_stage07_b01_formal_36254ed8/README.md) | 5 | [CSV字段说明](./20260810T081252Z_1662dd06774d_stage07_b01_formal_36254ed8/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081252Z_1662dd06774d_stage07_b01_formal_a24d94ee`](./20260810T081252Z_1662dd06774d_stage07_b01_formal_a24d94ee/README.md) | 5 | [CSV字段说明](./20260810T081252Z_1662dd06774d_stage07_b01_formal_a24d94ee/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081252Z_1662dd06774d_stage07_b01_formal_cb265736`](./20260810T081252Z_1662dd06774d_stage07_b01_formal_cb265736/README.md) | 5 | [CSV字段说明](./20260810T081252Z_1662dd06774d_stage07_b01_formal_cb265736/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081252Z_1662dd06774d_stage07_b01_formal_cf37b5ee`](./20260810T081252Z_1662dd06774d_stage07_b01_formal_cf37b5ee/README.md) | 5 | [CSV字段说明](./20260810T081252Z_1662dd06774d_stage07_b01_formal_cf37b5ee/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081253Z_1662dd06774d_stage07_b01_formal_182342b7`](./20260810T081253Z_1662dd06774d_stage07_b01_formal_182342b7/README.md) | 5 | [CSV字段说明](./20260810T081253Z_1662dd06774d_stage07_b01_formal_182342b7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081253Z_1662dd06774d_stage07_b01_formal_6664266c`](./20260810T081253Z_1662dd06774d_stage07_b01_formal_6664266c/README.md) | 5 | [CSV字段说明](./20260810T081253Z_1662dd06774d_stage07_b01_formal_6664266c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081253Z_1662dd06774d_stage07_b01_formal_e953b908`](./20260810T081253Z_1662dd06774d_stage07_b01_formal_e953b908/README.md) | 5 | [CSV字段说明](./20260810T081253Z_1662dd06774d_stage07_b01_formal_e953b908/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081254Z_1662dd06774d_stage07_b01_formal_065b6cb9`](./20260810T081254Z_1662dd06774d_stage07_b01_formal_065b6cb9/README.md) | 5 | [CSV字段说明](./20260810T081254Z_1662dd06774d_stage07_b01_formal_065b6cb9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081254Z_1662dd06774d_stage07_b01_formal_18a0df11`](./20260810T081254Z_1662dd06774d_stage07_b01_formal_18a0df11/README.md) | 5 | [CSV字段说明](./20260810T081254Z_1662dd06774d_stage07_b01_formal_18a0df11/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081254Z_1662dd06774d_stage07_b01_formal_3a2ddbaf`](./20260810T081254Z_1662dd06774d_stage07_b01_formal_3a2ddbaf/README.md) | 5 | [CSV字段说明](./20260810T081254Z_1662dd06774d_stage07_b01_formal_3a2ddbaf/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081254Z_1662dd06774d_stage07_b01_formal_f6f1fd79`](./20260810T081254Z_1662dd06774d_stage07_b01_formal_f6f1fd79/README.md) | 5 | [CSV字段说明](./20260810T081254Z_1662dd06774d_stage07_b01_formal_f6f1fd79/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081255Z_1662dd06774d_stage07_b01_formal_3e75f231`](./20260810T081255Z_1662dd06774d_stage07_b01_formal_3e75f231/README.md) | 5 | [CSV字段说明](./20260810T081255Z_1662dd06774d_stage07_b01_formal_3e75f231/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081255Z_1662dd06774d_stage07_b01_formal_9b299815`](./20260810T081255Z_1662dd06774d_stage07_b01_formal_9b299815/README.md) | 5 | [CSV字段说明](./20260810T081255Z_1662dd06774d_stage07_b01_formal_9b299815/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081255Z_1662dd06774d_stage07_b01_formal_e20ffdb5`](./20260810T081255Z_1662dd06774d_stage07_b01_formal_e20ffdb5/README.md) | 5 | [CSV字段说明](./20260810T081255Z_1662dd06774d_stage07_b01_formal_e20ffdb5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_0cc48783`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_0cc48783/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_0cc48783/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_4c44f50a`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_4c44f50a/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_4c44f50a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_6d875982`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_6d875982/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_6d875982/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_7d427de9`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_7d427de9/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_7d427de9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_cbb0bf8a`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_cbb0bf8a/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_cbb0bf8a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_e0bfebb2`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_e0bfebb2/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_e0bfebb2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081256Z_1662dd06774d_stage07_b01_formal_fb9e378f`](./20260810T081256Z_1662dd06774d_stage07_b01_formal_fb9e378f/README.md) | 5 | [CSV字段说明](./20260810T081256Z_1662dd06774d_stage07_b01_formal_fb9e378f/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_05b15ed7`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_05b15ed7/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_05b15ed7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_13288dd7`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_13288dd7/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_13288dd7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_1b469801`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_1b469801/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_1b469801/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_769fddcb`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_769fddcb/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_769fddcb/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_970a3312`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_970a3312/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_970a3312/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_ec50f2c4`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_ec50f2c4/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_ec50f2c4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081257Z_1662dd06774d_stage07_b01_formal_eeed66c6`](./20260810T081257Z_1662dd06774d_stage07_b01_formal_eeed66c6/README.md) | 5 | [CSV字段说明](./20260810T081257Z_1662dd06774d_stage07_b01_formal_eeed66c6/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T081258Z_1662dd06774d_stage07_b01_formal_3f599314`](./20260810T081258Z_1662dd06774d_stage07_b01_formal_3f599314/README.md) | 5 | [CSV字段说明](./20260810T081258Z_1662dd06774d_stage07_b01_formal_3f599314/CSV字段说明.md) | 5 | 0 | COVERED |

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
