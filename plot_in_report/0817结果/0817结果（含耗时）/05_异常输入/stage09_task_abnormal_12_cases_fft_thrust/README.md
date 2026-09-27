# 阶段09：Task1/Task2任务级异常（fft_thrust）

## 结论与范围

本目录保存 Task1 与 Task2 在真实 `fft_thrust` 编译配置下的任务级异常正式结果。构建使用互斥选项 `USE_DLFFT=OFF`、`USE_THRUST=ON`，CMake明确输出 `ZQ500 FFT backend: THRUST`。

- Task1：6/6 PASS，`safe_exit=true` 6/6，`output_consumed=false` 6/6。
- Task2：6/6 PASS，`safe_exit=true` 6/6，`output_consumed=false` 6/6。
- backend：`fft_thrust`，CSV文件名为 `task_abnormal_cases_fft_thrust.csv`。
- 源码提交：`5fcc1a8bc321546dbe3e4f8f591cd5e56fd368ca`，同步前工作树 clean。
- 回收批次：`RB_20260815_STAGE09_TASK_ABNORMAL_FFT_THRUST`。
- 完整性：容器、宿主机、Windows三端逐文件大小与SHA-256一致；DLFFT平行结果保留在相邻目录，未覆盖或删除。

## 判读约束

这些case验证任务级前置校验和非法backend拒绝路径，不应解释为真实FFT计算失败。CSV的 `backend=fft_thrust` 表示编译与运行上下文；非法backend消息进一步记录 `compiled=fft_thrust`。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-15T20:56:14+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`05_异常输入/stage09_task_abnormal_12_cases_fft_thrust`
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
