# 回收批次

每个子目录保存一次回收批次的来源身份、逐文件校验清单和闭环说明。批次状态、来源与目标记录在上一级`recovery_batches.csv`中。

本目录不得保存压缩包。阶段07与阶段08已分别完整展开；阶段10正式稳定性批次采用容器普通文件树→宿主机普通文件树→Windows普通文件树的逐文件回收方式，9004个文件三端SHA-256一致。各批次来源依靠逐文件SHA-256清单、回收批次记录及各级folder manifest追溯。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-17T16:56:58+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`01_总清单与校验/recovery_batches`
- 直接子目录：6个
- 直接文件：3个
- 递归文件：29 个
- 递归CSV：12 个

### 子目录追溯

| 子目录 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `RB_20260810T123934Z_STAGE05_TASK1_PIPELINE_FFT_THRUST` | 5 | 2 | [README](./RB_20260810T123934Z_STAGE05_TASK1_PIPELINE_FFT_THRUST/README.md) | [字段说明](./RB_20260810T123934Z_STAGE05_TASK1_PIPELINE_FFT_THRUST/CSV字段说明.md) | [清单](./RB_20260810T123934Z_STAGE05_TASK1_PIPELINE_FFT_THRUST/folder_manifest.csv) |
| `RB_20260810T1446CST_STAGE07_08` | 4 | 1 | [README](./RB_20260810T1446CST_STAGE07_08/README.md) | [字段说明](./RB_20260810T1446CST_STAGE07_08/CSV字段说明.md) | [清单](./RB_20260810T1446CST_STAGE07_08/folder_manifest.csv) |
| `RB_20260814T021000Z_STAGE10_FORMAL_1000` | 4 | 2 | [README](./RB_20260814T021000Z_STAGE10_FORMAL_1000/README.md) | [字段说明](./RB_20260814T021000Z_STAGE10_FORMAL_1000/CSV字段说明.md) | [清单](./RB_20260814T021000Z_STAGE10_FORMAL_1000/folder_manifest.csv) |
| `RB_20260815_STAGE09_TASK_ABNORMAL` | 4 | 2 | [README](./RB_20260815_STAGE09_TASK_ABNORMAL/README.md) | [字段说明](./RB_20260815_STAGE09_TASK_ABNORMAL/CSV字段说明.md) | [清单](./RB_20260815_STAGE09_TASK_ABNORMAL/folder_manifest.csv) |
| `RB_20260815_STAGE09_TASK_ABNORMAL_FFT_THRUST` | 4 | 2 | [README](./RB_20260815_STAGE09_TASK_ABNORMAL_FFT_THRUST/README.md) | [字段说明](./RB_20260815_STAGE09_TASK_ABNORMAL_FFT_THRUST/CSV字段说明.md) | [清单](./RB_20260815_STAGE09_TASK_ABNORMAL_FFT_THRUST/folder_manifest.csv) |
| `RB_20260815T152000Z_STAGE06_TASK2_PIPELINE_FFT_THRUST` | 5 | 2 | [README](./RB_20260815T152000Z_STAGE06_TASK2_PIPELINE_FFT_THRUST/README.md) | [字段说明](./RB_20260815T152000Z_STAGE06_TASK2_PIPELINE_FFT_THRUST/CSV字段说明.md) | [清单](./RB_20260815T152000Z_STAGE06_TASK2_PIPELINE_FFT_THRUST/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256位置 |
| --- | --- | --- | --- | --- |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`README.md`](./README.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |

### 阅读顺序

1. 先读本README；2. CSV先查同目录字段说明；3. 文件身份查folder manifest；4. 跨目录身份查全局回收批次。
<!-- AUTO_ARCHIVE_INDEX_END -->
