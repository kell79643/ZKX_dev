# 任务涉及算子

- 对应开发阶段：阶段07
- 目标集合：Task1/Task2调用链冻结得到的20个唯一算子
- 当前文档状态：20/20 `OBJECT_PASS`
- 当前直接证据状态：`EXPANDED_AND_INDEXED`
- 计划内容：按`<module>/<operator>/formal|smoke|terminal_logs`组织真实文件
- 正式使用条件：原始CSV、逐轮耗时、资源轨迹、输入证据、审计日志和SHA-256均完成本地校验

阶段07的最终浏览入口为`data/RB_20260810T1446CST_STAGE07/README.md`。共3,885个原始文件已展开，另提供CPU/GPU对比表、逐设备测量表、输入证据索引、算子覆盖表和文件目录表，不需要解压即可查看。

阶段07派生压缩包已在最终展开校验通过后移除；评委、PPT和技术文档统一从`data/`进入。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`04_算子结果/task_operators`
- 直接子目录：1个
- 直接文件：3个
- 递归文件：5412个
- 递归CSV：2244个
- README未覆盖文件：0个
- CSV字段说明未覆盖文件：0个

### 目录结构与子目录追溯

| 子目录 | 用途 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | 完整manifest | 状态 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `data` | 该子目录的具体证据；先阅读其README。 | 5409 | 2243 | [README](./data/README.md) | [CSV字段说明](./data/CSV字段说明.md) | [folder manifest](./data/folder_manifest.csv) | TRACEABLE |

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
- 归档根：[README](../../README.md)
- 本目录CSV字段说明：[CSV字段说明.md](./CSV字段说明.md)
- 本目录完整文件清单：[folder_manifest.csv](./folder_manifest.csv)
- 全局artifact manifest：[../../01_总清单与校验/artifact_manifest.csv](../../01_总清单与校验/artifact_manifest.csv)
- 全局SHA-256总表：[../../01_总清单与校验/sha256_manifest.csv](../../01_总清单与校验/sha256_manifest.csv)
- 回收批次：[../../01_总清单与校验/recovery_batches.csv](../../01_总清单与校验/recovery_batches.csv)
- 当前缺失项：[../../01_总清单与校验/missing_artifacts.csv](../../01_总清单与校验/missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
