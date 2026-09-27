# 任务结果

本目录用于Task1、Task2两个后端完整case矩阵的整体结果和逐Step结果，包括精度、耗时、输入身份、状态、调用链和完整运行外键。

正式结果应至少能按`task_name、run_id、case_id、backend、dtype、scale_id、step_name`检索，并明确CPU/GPU比较口径。算子独立结果归入`04_算子结果`，异常输入归入`05_异常输入`。

当前目录尚未回收Task1/Task2最终完整矩阵，不得用阶段07/08算子结果替代任务结果。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-17T16:56:58+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`03_任务结果`
- 直接子目录：2个
- 直接文件：3个
- 递归文件：2703 个
- 递归CSV：1339 个

### 子目录追溯

| 子目录 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `Task1` | 1351 | 669 | [README](./Task1/README.md) | [字段说明](./Task1/CSV字段说明.md) | [清单](./Task1/folder_manifest.csv) |
| `Task2` | 1349 | 669 | [README](./Task2/README.md) | [字段说明](./Task2/CSV字段说明.md) | [清单](./Task2/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256位置 |
| --- | --- | --- | --- | --- |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`README.md`](./README.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |

### 阅读顺序

1. 先读本README；2. CSV先查同目录字段说明；3. 文件身份查folder manifest；4. 跨目录身份查全局回收批次。
<!-- AUTO_ARCHIVE_INDEX_END -->
