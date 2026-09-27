# 技术文档引用

本目录维护技术文档章节、技术主张、表格、图号、代码位置和证据文件之间的映射。由于原始Word不提供给AI，本目录只形成可供用户整合的内容与引用建议。

当前尚未建立最终章节级引用表。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`16_技术文档引用`
- 直接子目录：0个
- 直接文件：3个
- 递归文件：3个
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

### 正确阅读顺序

1. 先读本`README.md`确认目录身份、文件用途和证据状态；
2. 遇到CSV，先读同目录`CSV字段说明.md`，按文件名定位schema，再读全部列定义；
3. 需要来源、大小和SHA-256时查`folder_manifest.csv`；
4. 需要跨目录身份时查`01_总清单与校验/`中的全局manifest和回收批次；
5. 需要原始日志、图表或文档引用时沿本README的相关文件和上级导航继续追溯。

### 快速导航

- 上级目录：[返回上级](../README.md)
- 归档根：[README](../README.md)
- 本目录CSV字段说明：[CSV字段说明.md](./CSV字段说明.md)
- 本目录完整文件清单：[folder_manifest.csv](./folder_manifest.csv)
- 全局artifact manifest：[../01_总清单与校验/artifact_manifest.csv](../01_总清单与校验/artifact_manifest.csv)
- 全局SHA-256总表：[../01_总清单与校验/sha256_manifest.csv](../01_总清单与校验/sha256_manifest.csv)
- 回收批次：[../01_总清单与校验/recovery_batches.csv](../01_总清单与校验/recovery_batches.csv)
- 当前缺失项：[../01_总清单与校验/missing_artifacts.csv](../01_总清单与校验/missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
