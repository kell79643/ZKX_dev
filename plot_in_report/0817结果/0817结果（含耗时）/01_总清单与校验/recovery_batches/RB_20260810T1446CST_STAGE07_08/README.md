# 阶段07/08历史回收批次记录

- 批次号：`RB_20260810T1446CST_STAGE07_08`
- 历史来源标识：`gpu_02:/tmp/zkx_stage07_08_20260810T1446CST.tar.gz`；只用于说明2026-08-10批次来源，不作为现行回收方式
- 文件总数：17,301
- 历史运输文件SHA-256：`619370544e5a7c615ca88dd0b69e5bff018a7427973184363b8ff22658d9867e`
- 逐文件清单SHA-256：`9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0`
- 校验结论：历史回收三端身份一致；最终展开后的17,301个文件逐项一致

本地历史压缩文件已经在确认17,301个展开文件和逐文件SHA-256清单完整后于2026-08-14删除。本目录只保留逐文件校验清单和回收说明。最终可直接浏览的数据分别位于：

- 阶段07：`../../../04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/`
- 阶段08：`../../../04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/`

当前全归档压缩包扫描数量为0；后续批次必须按照`AGENT.md`逐文件回收，不再创建运输压缩包。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`01_总清单与校验/recovery_batches/RB_20260810T1446CST_STAGE07_08`
- 直接子目录：0个
- 直接文件：4个
- 递归文件：4个
- 递归CSV：1个
- README未覆盖文件：0个
- CSV字段说明未覆盖文件：0个

### 目录结构与子目录追溯

本目录为叶子目录，没有直接子目录。

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | 来源与身份 | SHA-256位置 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 当前目录CSV逐文件schema映射和逐列语义说明。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 当前目录递归文件身份、来源、大小和SHA-256清单。 | [字段说明](./CSV字段说明.md#schema_7f4d2cbb2001) | 本地治理文件 | 由上级folder manifest记录 | TRACEABLE |
| [`README.md`](./README.md) | governance | 当前目录阅读入口、文件追溯与快速导航。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`zkx_stage07_08_20260810T1446CST_host_files.sha256`](./zkx_stage07_08_20260810T1446CST_host_files.sha256) | sha256 | 归档文件；具体用途结合文件名和同目录说明。 | NA | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |

### 正确阅读顺序

1. 先读本`README.md`确认目录身份、文件用途和证据状态；
2. 遇到CSV，先读同目录`CSV字段说明.md`，按文件名定位schema，再读全部列定义；
3. 需要来源、大小和SHA-256时查`folder_manifest.csv`；
4. 需要跨目录身份时查`01_总清单与校验/`中的全局manifest和回收批次；
5. 需要原始日志、图表或文档引用时沿本README的相关文件和上级导航继续追溯。

### 快速导航

- 上级目录：[返回上级](../README.md)
- 归档根：[README](../../../README.md)
- 本目录CSV字段说明：[CSV字段说明.md](./CSV字段说明.md)
- 本目录完整文件清单：[folder_manifest.csv](./folder_manifest.csv)
- 全局artifact manifest：[../../artifact_manifest.csv](../../artifact_manifest.csv)
- 全局SHA-256总表：[../../sha256_manifest.csv](../../sha256_manifest.csv)
- 回收批次：[../../recovery_batches.csv](../../recovery_batches.csv)
- 当前缺失项：[../../missing_artifacts.csv](../../missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
