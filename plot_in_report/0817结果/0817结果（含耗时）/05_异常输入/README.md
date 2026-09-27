# 异常输入与错误码

本目录保存阶段09异常输入证据，任务级和算子级结果分开组织。

## 当前入口

| 范围 | backend | 文字入口 | 目标CSV | 状态 |
| --- | --- | --- | --- | --- |
| 53个算子 | 按算子真实后端分类 | [算子异常README](./stage09_operator_abnormal_159_617051d/README.md) | `all_operator_abnormal_cases.csv` | 159/159 PASS |
| Task1 | `dlfft` | [DLFFT任务异常README](./stage09_task_abnormal_12_cases/README.md) | `Task1/task_abnormal_cases_dlfft.csv` | 6/6 PASS |
| Task2 | `dlfft` | [DLFFT任务异常README](./stage09_task_abnormal_12_cases/README.md) | `Task2/task_abnormal_cases_dlfft.csv` | 6/6 PASS |
| Task1 | `fft_thrust` | [fft_thrust任务异常README](./stage09_task_abnormal_12_cases_fft_thrust/README.md) | `Task1/task_abnormal_cases_fft_thrust.csv` | 6/6 PASS |
| Task2 | `fft_thrust` | [fft_thrust任务异常README](./stage09_task_abnormal_12_cases_fft_thrust/README.md) | `Task2/task_abnormal_cases_fft_thrust.csv` | 6/6 PASS |

两套任务级结果平行保留，不能把一个后端的CSV改名充当另一个后端。算子逐项明细和终端输出位于算子入口的`raw/`。本地数据不使用压缩包呈现。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-15T20:56:08+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`05_异常输入`
- 直接子目录：3个
- 直接文件：3个
- 递归文件：317 个
- 递归CSV：128 个

### 子目录追溯

| 子目录 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `stage09_operator_abnormal_159_617051d` | 288 | 117 | [README](./stage09_operator_abnormal_159_617051d/README.md) | [字段说明](./stage09_operator_abnormal_159_617051d/CSV字段说明.md) | [清单](./stage09_operator_abnormal_159_617051d/folder_manifest.csv) |
| `stage09_task_abnormal_12_cases` | 13 | 5 | [README](./stage09_task_abnormal_12_cases/README.md) | [字段说明](./stage09_task_abnormal_12_cases/CSV字段说明.md) | [清单](./stage09_task_abnormal_12_cases/folder_manifest.csv) |
| `stage09_task_abnormal_12_cases_fft_thrust` | 13 | 5 | [README](./stage09_task_abnormal_12_cases_fft_thrust/README.md) | [字段说明](./stage09_task_abnormal_12_cases_fft_thrust/CSV字段说明.md) | [清单](./stage09_task_abnormal_12_cases_fft_thrust/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256位置 |
| --- | --- | --- | --- | --- |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`README.md`](./README.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |

### 阅读顺序

1. 先读本README；2. CSV先查同目录字段说明；3. 文件身份查folder manifest；4. 跨目录身份查全局回收批次。
<!-- AUTO_ARCHIVE_INDEX_END -->
