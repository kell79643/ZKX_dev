# 阶段09：53算子异常输入与错误码

## 结论与范围

本目录是阶段09“算子异常输入部分”的最终本地结果：53个GPU算子，每算子3类不同失败语义，共159条异常case。

```text
operators=53
cases=159
pass=159
fail=0
safe_exit=159
output_consumed=0
normal_links=159/159
```

这一定义只覆盖算子部分，不代替Task1、Task2任务级异常结果。

## 从哪里开始看

| 优先级 | 文件或目录 | 内容 |
| --- | --- | --- |
| 1 | `all_operator_abnormal_cases.csv` | 159行统一总表，答辩、筛选和截图选材首选入口 |
| 2 | `operator_abnormal_summary.csv` | 53行，一行一个算子，核对每算子3类异常和3/3 PASS |
| 3 | `abnormal_type_summary.csv` | 5类异常语义的数量、错误码、安全退出汇总 |
| 4 | `final_v1/` | 按后端拆分的三份正式原始CSV |
| 5 | `raw/` | 53个算子目录，每个包含3行CSV和完整`terminal.log` |
| 6 | `file_catalog.csv` | 展开文件的类型、大小和相对路径 |
| 7 | `expanded_files.sha256` | 展开文件逐项SHA-256清单 |
| 8 | `documentation/` | 源阶段结论、远程证据和错误码规范 |

## 正式后端拆分

| 文件 | 行数 | 说明 |
| --- | ---: | --- |
| `final_v1/operator_abnormal_cases_cuda_runtime.csv` | 32 | CUDA Runtime后端选择结果 |
| `final_v1/operator_abnormal_cases_not_applicable.csv` | 113 | 不需要特定后端子分类的结果 |
| `final_v1/operator_abnormal_cases_selected_backend.csv` | 14 | FFT等显式选择后端结果 |
| 合计 | 159 | `case_id`全部唯一 |

## 异常语义分布

| `abnormal_type` | 数量 | 主要项目错误码 |
| --- | ---: | --- |
| `invalid_shape` | 53 | `INVALID_SHAPE` |
| `invalid_dtype` | 8 | `INVALID_DTYPE` |
| `invalid_parameter` | 47 | `INVALID_ARGUMENT` |
| `unsupported_operation` | 7 | `UNSUPPORTED_OPERATION` |
| `backend_call_failure` | 44 | `CUDA_RUNTIME_ERROR / BACKEND_ERROR` |

## 总表字段怎么读

| 列 | 含义 |
| --- | --- |
| `timestamp` | case执行时间 |
| `run_id` | 本次异常运行的唯一身份 |
| `backend` | 本case使用或选择的后端 |
| `test_object` | 当前为`operator` |
| `task_name、step_name` | 算子独立测试为`NA`；任务级异常才使用这些列 |
| `module_name、operator_name` | 模块与算子，用于筛选53个对象 |
| `case_id` | 异常case唯一标识 |
| `normal_run_id、normal_case_id` | 对应正常PASS证据的外键，不是随意占位 |
| `abnormal_type` | 失败语义类别，同一算子的3条必须互不相同 |
| `input_summary` | 注入的异常输入或变异摘要 |
| `expected_code` | 设计时要求返回的稳定项目错误码 |
| `captured_code` | 测试入口实际捕获到的项目错误码 |
| `returned_code` | 算子返回路径携带的项目错误码 |
| `actual_code` | 用于最终判定的实际错误码；应与`expected_code`一致 |
| `error_message` | 明确自然语言报错，不允许只有数字状态 |
| `backend_error_code` | CUDA、FFT或内部数值后端原始/受控等价错误码；输入前置校验可为`NA` |
| `backend_error_name` | 后端错误名称 |
| `backend_error_message` | 后端错误描述 |
| `safe_exit` | `true`表示未崩溃、未挂起并安全返回 |
| `output_consumed` | 异常后是否继续消费输出；本批次159条均为`false` |
| `status` | 本case合同判定；正式159条均为`PASS` |

## 判读规则

一条异常case只有同时满足以下条件才算PASS：

1. `expected_code == actual_code`；
2. `safe_exit == true`；
3. `output_consumed == false`；
4. `error_message`明确；
5. 后端失败时三元组完整，输入校验失败时允许为`NA`；
6. `normal_run_id + normal_case_id`可关联正常PASS证据。

本批次上述门禁均为159/159通过；53个算子的case数、不同异常语义数、安全退出数和PASS数均为3。

## 失败尝试边界

`lfilter_zi`首次运行的后端错误映射为2/3，通过修复后使用`filtering_lfilter_zi_retry1`取得3/3 PASS。首次失败文件已移至`90_历史证据/stage09_failed_attempts/`，没有混入本目录三份正式合并CSV。

## 来源与校验

- 远程来源：`/home/usr_02/ZKX_dev/results/stage09_operator_abnormal_159_617051d.tar.gz`
- 来源包大小：32,015 bytes
- 来源包SHA-256：`d0ba30d6a0e3e599e6e64053f3d08dba2389f6b14f1c515be5b117111e820857`
- 本地呈现：完全展开的CSV、LOG和Markdown；不保留阶段09本地压缩包
- 展开校验：见`expanded_files.sha256`

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`05_异常输入/stage09_operator_abnormal_159_617051d`
- 直接子目录：3个
- 直接文件：8个
- 递归文件：288个
- 递归CSV：117个
- README未覆盖文件：0个
- CSV字段说明未覆盖文件：0个

### 目录结构与子目录追溯

| 子目录 | 用途 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | 完整manifest | 状态 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `documentation` | 该子目录的具体证据；先阅读其README。 | 6 | 1 | [README](./documentation/README.md) | [CSV字段说明](./documentation/CSV字段说明.md) | [folder manifest](./documentation/folder_manifest.csv) | TRACEABLE |
| `final_v1` | 该子目录的具体证据；先阅读其README。 | 6 | 4 | [README](./final_v1/README.md) | [CSV字段说明](./final_v1/CSV字段说明.md) | [folder manifest](./final_v1/folder_manifest.csv) | TRACEABLE |
| `raw` | 该子目录的具体证据；先阅读其README。 | 268 | 107 | [README](./raw/README.md) | [CSV字段说明](./raw/CSV字段说明.md) | [folder manifest](./raw/folder_manifest.csv) | TRACEABLE |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | 来源与身份 | SHA-256位置 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [`abnormal_type_summary.csv`](./abnormal_type_summary.csv) | csv | 异常输入、错误码和安全退出结果。 | [字段说明](./CSV字段说明.md#schema_25dbf2f4d43d) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`all_operator_abnormal_cases.csv`](./all_operator_abnormal_cases.csv) | csv | 异常输入、错误码和安全退出结果。 | [字段说明](./CSV字段说明.md#schema_bf0ae92fc547) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 当前目录CSV逐文件schema映射和逐列语义说明。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`expanded_files.sha256`](./expanded_files.sha256) | sha256 | 归档文件；具体用途结合文件名和同目录说明。 | NA | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`file_catalog.csv`](./file_catalog.csv) | csv | 证据清单、目录或关联索引。 | [字段说明](./CSV字段说明.md#schema_9c155bb8c8b9) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 当前目录递归文件身份、来源、大小和SHA-256清单。 | [字段说明](./CSV字段说明.md#schema_7f4d2cbb2001) | 本地治理文件 | 由上级folder manifest记录 | TRACEABLE |
| [`operator_abnormal_summary.csv`](./operator_abnormal_summary.csv) | csv | 异常输入、错误码和安全退出结果。 | [字段说明](./CSV字段说明.md#schema_4ad8b3491740) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
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
