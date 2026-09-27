# 阶段09任务级异常回收批次

- 批次号：`RB_20260815_STAGE09_TASK_ABNORMAL`。
- 来源主机/容器：`usr_02@10.110.12.10` / `gpu_02`。
- 容器原件：Task1、Task2 两个结果目录，共 4 个普通文件、10,888 bytes。
- 宿主机临时普通文件树：曾使用`/home/usr_02/zkx_recovery/RB_20260815_STAGE09_TASK_ABNORMAL/`；三端校验完成后，经用户明确授权于2026-08-15删除。
- Windows入口：`../../../05_异常输入/stage09_task_abnormal_12_cases/`。
- 源清单：`source_manifest.csv`，由容器根据原件生成。
- 校验结论：4/4 文件三端大小与 SHA-256 一致；源端、宿主机和 Windows 目标均无压缩文件。
- 保留策略：容器原件和Windows正式归档保留；宿主机传输中转副本已删除，无证据损失。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-15T21:01:08+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`01_总清单与校验/recovery_batches/RB_20260815_STAGE09_TASK_ABNORMAL`
- 直接子目录：0个
- 直接文件：4个
- 递归文件：4 个
- 递归CSV：2 个

### 子目录追溯

本目录为叶子目录。

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256位置 |
| --- | --- | --- | --- | --- |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`README.md`](./README.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`source_manifest.csv`](./source_manifest.csv) | csv | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |

### 阅读顺序

1. 先读本README；2. CSV先查同目录字段说明；3. 文件身份查folder manifest；4. 跨目录身份查全局回收批次。
<!-- AUTO_ARCHIVE_INDEX_END -->
