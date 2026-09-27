# 阶段09：Task1/Task2任务级异常（DLFFT）

## 结论与范围

本目录保存 Task1 与 Task2 在 `dlfft` 编译配置下的任务级异常正式复验结果。每个任务覆盖 6 类失败语义，共 12 条：配置缺失、上游输入缺失、非法任务参数、不支持 dtype、受控 OOM/超大输入、非法 backend。`fft_thrust` 平行结果位于相邻目录 `stage09_task_abnormal_12_cases_fft_thrust/`。

- Task1：6/6 PASS，`safe_exit=true` 6/6，`output_consumed=false` 6/6。
- Task2：6/6 PASS，`safe_exit=true` 6/6，`output_consumed=false` 6/6。
- backend：`dlfft`，CSV文件名为 `task_abnormal_cases_dlfft.csv`。
- 原件：每个任务各一份逐 case CSV 和一份终端简表日志。
- 回收批次：`RB_20260815_STAGE09_TASK_ABNORMAL`。
- 完整性：容器、宿主机、Windows 三端逐文件大小与 SHA-256 一致；本目录压缩文件为 0。

## 判读约束

CSV 的 `run_id`、`case_id`、`task_name` 和 `step_name` 是逐 case 关联键。`status=PASS` 还必须同时满足捕获/返回错误码与期望一致、安全退出、失败后未消费输出；不能只凭进程退出判断。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-15T20:56:13+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`05_异常输入/stage09_task_abnormal_12_cases`
- 直接子目录：2个
- 直接文件：3个
- 递归文件：13 个
- 递归CSV：5 个

### 子目录追溯

| 子目录 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `Task1` | 5 | 2 | [README](./Task1/README.md) | [字段说明](./Task1/CSV字段说明.md) | [清单](./Task1/folder_manifest.csv) |
| `Task2` | 5 | 2 | [README](./Task2/README.md) | [字段说明](./Task2/CSV字段说明.md) | [清单](./Task2/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256位置 |
| --- | --- | --- | --- | --- |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`README.md`](./README.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |

### 阅读顺序

1. 先读本README；2. CSV先查同目录字段说明；3. 文件身份查folder manifest；4. 跨目录身份查全局回收批次。
<!-- AUTO_ARCHIVE_INDEX_END -->
