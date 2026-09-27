# 阶段07任务涉及算子：直接数据入口

## 数据范围

- 算子：20个
- 回收CPU/GPU成对记录：433条，其中`formal=422`、`smoke=11`
- 唯一正式测试case：302个；正式记录中的其余120条为同一case的重跑
- CPU/GPU测量行：866行；433条回收记录的CPU行和GPU行均齐全
- dtype：`FP16 / FP32 / INT8 / INT16 / INT32`
- 后端标签：`fft_thrust / not_applicable`
- 原始文件：3,885个，包括1,732个CSV、873个JSON、846个LOG、434个TXT
- 当前回收行：`status=PASS` 433/433，`accuracy_status=PASS` 433/433

## 从哪里开始看

| 优先级 | 文件或目录 | 内容 |
| --- | --- | --- |
| 1 | `cpu_gpu_comparison.csv` | 433行，一行一个运行记录对应的输入case；含formal、smoke和重跑，直接含四项精度、CPU/GPU各类耗时和加速比 |
| 2 | `operator_coverage.csv` | 20行，一行一个算子；查看回收记录中的dtype、规模和形状覆盖，`case_count`含重跑 |
| 3 | `input_evidence_index.csv` | 433行；查看输入dtype契约、输入内容JSON、case参数与摘要 |
| 4 | `measurement_rows.csv` | 866行；一行一个CPU或GPU测量对象 |
| 5 | `file_catalog.csv` | 3,885行；定位每个原始文件 |
| 6 | `runs/` | 完整展开的原始CSV、JSON、TXT与LOG |
| 7 | `expanded_files.sha256` | 3,885个原始文件的SHA-256清单 |

## 特别说明

阶段07中`argrelextrema`的15个case输出为离散坐标，因此四个数值误差字段为`NA`；这些case均为`exact_match=true`、`mismatch_count=0`、`accuracy_status=PASS`。这属于正确的语义判据切换，不是精度数据丢失。

完整列定义、公式、加速比方向、逐轮耗时位置和联表方法见上级`../../../README.md`。

无重复的一行一case绘图候选位于`../../../绘图数据/全算子_逐case绘图候选.csv`，其中阶段07共302行。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07`
- 直接子目录：1个
- 直接文件：9个
- 递归文件：5406个
- 递归CSV：2242个
- README未覆盖文件：0个
- CSV字段说明未覆盖文件：0个

### 目录结构与子目录追溯

| 子目录 | 用途 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | 完整manifest | 状态 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `runs` | 该子目录的具体证据；先阅读其README。 | 5397 | 2236 | [README](./runs/README.md) | [CSV字段说明](./runs/CSV字段说明.md) | [folder manifest](./runs/folder_manifest.csv) | TRACEABLE |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | 来源与身份 | SHA-256位置 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [`cpu_gpu_comparison.csv`](./cpu_gpu_comparison.csv) | csv | 精度、耗时、稳定性和加速比结果。 | [字段说明](./CSV字段说明.md#schema_b8ca1c80348b) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 当前目录CSV逐文件schema映射和逐列语义说明。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`expanded_files.sha256`](./expanded_files.sha256) | sha256 | 归档文件；具体用途结合文件名和同目录说明。 | NA | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`file_catalog.csv`](./file_catalog.csv) | csv | 证据清单、目录或关联索引。 | [字段说明](./CSV字段说明.md#schema_7fa1b27bf151) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 当前目录递归文件身份、来源、大小和SHA-256清单。 | [字段说明](./CSV字段说明.md#schema_7f4d2cbb2001) | 本地治理文件 | 由上级folder manifest记录 | TRACEABLE |
| [`input_evidence_index.csv`](./input_evidence_index.csv) | csv | 证据清单、目录或关联索引。 | [字段说明](./CSV字段说明.md#schema_c6d6d1c6c971) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`measurement_rows.csv`](./measurement_rows.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_2a341875b4ab) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`operator_coverage.csv`](./operator_coverage.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_e7ec8aca9a36) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`README.md`](./README.md) | governance | 当前目录阅读入口、文件追溯与快速导航。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |

### 正确阅读顺序

1. 先读本`README.md`确认目录身份、文件用途和证据状态；
2. 遇到CSV，先读同目录`CSV字段说明.md`，按文件名定位schema，再读全部列定义；
3. 需要来源、大小和SHA-256时查`folder_manifest.csv`；
4. 需要跨目录身份时查`01_总清单与校验/`中的全局manifest和回收批次；
5. 需要原始日志、图表或文档引用时沿本README的相关文件和上级导航继续追溯。

### 快速导航

- 上级目录：[返回上级](../README.md)
- 归档根：[README](../../../../README.md)
- 本目录CSV字段说明：[CSV字段说明.md](./CSV字段说明.md)
- 本目录完整文件清单：[folder_manifest.csv](./folder_manifest.csv)
- 全局artifact manifest：[../../../../01_总清单与校验/artifact_manifest.csv](../../../../01_总清单与校验/artifact_manifest.csv)
- 全局SHA-256总表：[../../../../01_总清单与校验/sha256_manifest.csv](../../../../01_总清单与校验/sha256_manifest.csv)
- 回收批次：[../../../../01_总清单与校验/recovery_batches.csv](../../../../01_总清单与校验/recovery_batches.csv)
- 当前缺失项：[../../../../01_总清单与校验/missing_artifacts.csv](../../../../01_总清单与校验/missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
