# 归档规范

## 证据分层

```text
原始证据 -> 校验后的规范数据 -> 汇总数据 -> 绘图数据 -> 图表 -> PPT/技术文档引用
```

- 原始CSV、JSON、日志、配置和截图只读保存，不因绘图需要就地修改。
- 每个派生文件记录输入文件、输入SHA-256、处理入口、筛选条件、输入/输出行数和输出SHA-256。
- PPT与技术文档只能引用可由`artifact_manifest.csv`定位的证据。
- 不同commit、run_id、backend、dtype、scale、计时口径和运行类型不得静默混合。

## 当前证据状态

允许使用：

`AVAILABLE_FORMAL`、`AVAILABLE_NEEDS_REVIEW`、`PARTIAL`、`SMOKE_ONLY`、`OBJECT_PASS_ONLY`、`REMOTE_NOT_ARCHIVED`、`OLD_COMMIT`、`DIRTY_RUN`、`INVALID_FOR_FINAL`、`MISSING`、`CONFLICTED`。

`git_dirty=true`不自动降低证据资格。完整dirty结果按其真实身份归档，不因dirty单独进入`99_隔离与待确认`。smoke、失败后保留结果和明确的旧结果进入`90_历史证据`；来源或身份无法确认的文件才进入`99_隔离与待确认`。

## 正式资格最低要求

- 文件真实存在且可读取；
- 具有run_id、case_id、commit和真实dirty状态；
- 记录backend、dtype、规模和运行次数；
- CPU/GPU使用相同输入digest；
- 正式性能结果为预热20次、测量100次；
- 计时、精度、资源和状态字段完整；
- 能从CSV反查配置、日志和源码版本；
- 回收前后SHA-256一致。

## 使用边界

- 文档中的PASS不自动等于正式证据已经归档。
- 代码中存在入口不自动等于已经运行通过。
- 计划测试不得写成当前PASS。
- 不删除失败、长尾、GPU慢于CPU或后端不支持的结果。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`00_归档规范`
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
| [`CSV字段语义总表.md`](./CSV字段语义总表.md) | md | 规范、说明、审计或引用文档。 | NA | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
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
