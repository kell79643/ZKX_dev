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
| [`20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2`](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2/README.md) | 5 | [CSV字段说明](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_4b613850`](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_4b613850/README.md) | 5 | [CSV字段说明](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_4b613850/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_776b788a`](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_776b788a/README.md) | 5 | [CSV字段说明](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_776b788a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_9336fb23`](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_9336fb23/README.md) | 5 | [CSV字段说明](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_9336fb23/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_a8241bb1`](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_a8241bb1/README.md) | 5 | [CSV字段说明](./20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_a8241bb1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_16eaf3b4`](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_16eaf3b4/README.md) | 5 | [CSV字段说明](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_16eaf3b4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_24943170`](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_24943170/README.md) | 5 | [CSV字段说明](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_24943170/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_3942ce5c`](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_3942ce5c/README.md) | 5 | [CSV字段说明](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_3942ce5c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_7fa114e5`](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_7fa114e5/README.md) | 5 | [CSV字段说明](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_7fa114e5/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_ac27e616`](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_ac27e616/README.md) | 5 | [CSV字段说明](./20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_ac27e616/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_9d00750e`](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_9d00750e/README.md) | 5 | [CSV字段说明](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_9d00750e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_a3ccd21c`](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_a3ccd21c/README.md) | 5 | [CSV字段说明](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_a3ccd21c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_a7a5ca3a`](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_a7a5ca3a/README.md) | 5 | [CSV字段说明](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_a7a5ca3a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_c3852a4b`](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_c3852a4b/README.md) | 5 | [CSV字段说明](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_c3852a4b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_e21f8895`](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_e21f8895/README.md) | 5 | [CSV字段说明](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_e21f8895/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_fa191094`](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_fa191094/README.md) | 5 | [CSV字段说明](./20260810T092020Z_76c5b6f8b36d_stage07_b02_formal_fa191094/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_04902b88`](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_04902b88/README.md) | 5 | [CSV字段说明](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_04902b88/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_1ad817f1`](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_1ad817f1/README.md) | 5 | [CSV字段说明](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_1ad817f1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_320faeca`](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_320faeca/README.md) | 5 | [CSV字段说明](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_320faeca/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_af317efa`](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_af317efa/README.md) | 5 | [CSV字段说明](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_af317efa/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_c9292f7b`](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_c9292f7b/README.md) | 5 | [CSV字段说明](./20260810T092021Z_76c5b6f8b36d_stage07_b02_formal_c9292f7b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_0537e8f1`](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_0537e8f1/README.md) | 5 | [CSV字段说明](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_0537e8f1/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_a9089da7`](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_a9089da7/README.md) | 5 | [CSV字段说明](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_a9089da7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_a9712040`](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_a9712040/README.md) | 5 | [CSV字段说明](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_a9712040/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_b450d2b6`](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_b450d2b6/README.md) | 5 | [CSV字段说明](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_b450d2b6/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_c7487d97`](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_c7487d97/README.md) | 5 | [CSV字段说明](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_c7487d97/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_ec6f70c6`](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_ec6f70c6/README.md) | 5 | [CSV字段说明](./20260810T092022Z_76c5b6f8b36d_stage07_b02_formal_ec6f70c6/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_051841b4`](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_051841b4/README.md) | 5 | [CSV字段说明](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_051841b4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_082e14c2`](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_082e14c2/README.md) | 5 | [CSV字段说明](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_082e14c2/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_7a994adb`](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_7a994adb/README.md) | 5 | [CSV字段说明](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_7a994adb/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_c489c3db`](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_c489c3db/README.md) | 5 | [CSV字段说明](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_c489c3db/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_d26c69e0`](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_d26c69e0/README.md) | 5 | [CSV字段说明](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_d26c69e0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_e93f6118`](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_e93f6118/README.md) | 5 | [CSV字段说明](./20260810T092023Z_76c5b6f8b36d_stage07_b02_formal_e93f6118/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_018c69b7`](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_018c69b7/README.md) | 5 | [CSV字段说明](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_018c69b7/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_3bb7173b`](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_3bb7173b/README.md) | 5 | [CSV字段说明](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_3bb7173b/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_5a2af18a`](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_5a2af18a/README.md) | 5 | [CSV字段说明](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_5a2af18a/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_b7693de4`](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_b7693de4/README.md) | 5 | [CSV字段说明](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_b7693de4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_f49261c4`](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_f49261c4/README.md) | 5 | [CSV字段说明](./20260810T092024Z_76c5b6f8b36d_stage07_b02_formal_f49261c4/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_039771ff`](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_039771ff/README.md) | 5 | [CSV字段说明](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_039771ff/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_2666dbaf`](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_2666dbaf/README.md) | 5 | [CSV字段说明](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_2666dbaf/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_63263178`](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_63263178/README.md) | 5 | [CSV字段说明](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_63263178/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_66c7d721`](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_66c7d721/README.md) | 5 | [CSV字段说明](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_66c7d721/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_db5f5240`](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_db5f5240/README.md) | 5 | [CSV字段说明](./20260810T092025Z_76c5b6f8b36d_stage07_b02_formal_db5f5240/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_0c4db795`](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_0c4db795/README.md) | 5 | [CSV字段说明](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_0c4db795/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_112ecb88`](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_112ecb88/README.md) | 5 | [CSV字段说明](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_112ecb88/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_42c25fa3`](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_42c25fa3/README.md) | 5 | [CSV字段说明](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_42c25fa3/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_475bf893`](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_475bf893/README.md) | 5 | [CSV字段说明](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_475bf893/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_ad68e021`](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_ad68e021/README.md) | 5 | [CSV字段说明](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_ad68e021/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_adcaccd0`](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_adcaccd0/README.md) | 5 | [CSV字段说明](./20260810T092026Z_76c5b6f8b36d_stage07_b02_formal_adcaccd0/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_4c57a1e3`](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_4c57a1e3/README.md) | 5 | [CSV字段说明](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_4c57a1e3/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_cbfdd652`](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_cbfdd652/README.md) | 5 | [CSV字段说明](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_cbfdd652/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_cd1d3033`](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_cd1d3033/README.md) | 5 | [CSV字段说明](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_cd1d3033/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_eed5563c`](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_eed5563c/README.md) | 5 | [CSV字段说明](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_eed5563c/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_f9ee4e6e`](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_f9ee4e6e/README.md) | 5 | [CSV字段说明](./20260810T092027Z_76c5b6f8b36d_stage07_b02_formal_f9ee4e6e/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_09f25ff3`](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_09f25ff3/README.md) | 5 | [CSV字段说明](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_09f25ff3/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_253e39a9`](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_253e39a9/README.md) | 5 | [CSV字段说明](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_253e39a9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_64c27d0d`](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_64c27d0d/README.md) | 5 | [CSV字段说明](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_64c27d0d/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_b95b2d92`](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_b95b2d92/README.md) | 5 | [CSV字段说明](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_b95b2d92/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_e2b190f9`](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_e2b190f9/README.md) | 5 | [CSV字段说明](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_e2b190f9/CSV字段说明.md) | 5 | 0 | COVERED |
| [`20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_ee922b85`](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_ee922b85/README.md) | 5 | [CSV字段说明](./20260810T092028Z_76c5b6f8b36d_stage07_b02_formal_ee922b85/CSV字段说明.md) | 5 | 0 | COVERED |

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
