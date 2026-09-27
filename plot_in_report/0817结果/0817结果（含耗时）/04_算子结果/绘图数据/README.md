# 绘图候选数据

本目录把阶段07和阶段08的完整回收结果压缩成短路径、可直接读取的绘图候选表。这里的“候选”表示已完成正式PASS筛选和确定性去重，但尚未绑定最终资格清单、精度阈值快照、资源门禁与泄漏门禁。

## 去重规则

规则编号：`LATEST_FORMAL_PASS_PER_CASE_V1`。

```text
仅保留 formal + status PASS + accuracy_status PASS + CPU/GPU均存在
同一 operator_name + case_id 按 test_time_utc、run_id 降序取第一条
```

该规则从2,034条正式PASS运行记录中选出1,112个唯一case；它不按speedup选择，不会为了图表挑更快结果。

`全算子_逐case绘图候选.csv`中的`source_relative_path`是相对原阶段批次目录的路径：

- `stage=stage07`时，以`../task_operators/data/RB_20260810T1446CST_STAGE07/`为根；
- `stage=stage08`时，以`../remaining_operators/data/RB_20260810T1446CST_STAGE08/`为根。

## 文件导航

| 文件 | 行数 | 直接用途 |
| --- | ---: | --- |
| `全算子_逐case绘图候选.csv` | 1,112 | 完整case绘图源、代表算子多规模图、技术文档附录 |
| `53算子_全case性能区间候选.csv` | 53 | 53算子区间图、12模块箱线图输入、算子汇总表 |
| `12模块_性能汇总候选.csv` | 12 | 模块数字摘要和技术文档模块表 |
| `53x5_最差精度候选.csv` | 265 | 四项精度各自的53×5最差值热力图 |
| `窗函数_435case热力图候选.csv` | 435 | 8个windows模块算子的dtype×scale×`variant_key`热力图；210例带完整参数JSON，225例保留原始唯一case键但参数JSON为空 |
| `4类代表算子候选.csv` | 4 | 高收益、典型、性能边界、FFT技术代表的规则化选择 |
| `53算子_演示case性能候选.csv` | 53 | 每算子性能演示初筛；资源和泄漏门禁尚未完成 |
| `53算子_12模块映射.csv` | 53 | 算子到12个模块的固定映射 |

## 当前不能直接宣称的内容

- `worst_accuracy_ratio`：当前绘图表未绑定每个算子、dtype、输出对应的阈值快照，字段明确写为`NA_THRESHOLD_SNAPSHOT_NOT_BOUND`。应先用四项最差值分别绘图。
- `BEST_DEMO_CASE`：当前只完成性能初筛，不能跳过资源、泄漏和现场时长检查。
- 正式PPT图：在最终资格清单签字确认前，本目录文件仍保留“候选”后缀。

完整字段解释、图表数据位置和CSV长路径问题见上级`../README.md`。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 本节由归档整改生成器于`2026-08-14T17:33:09+08:00`生成。任何文件新增、移动、删除或修改后必须重新生成。

### 本目录身份

- 归档相对路径：`04_算子结果/绘图数据`
- 直接子目录：0个
- 直接文件：11个
- 递归文件：11个
- 递归CSV：9个
- README未覆盖文件：0个
- CSV字段说明未覆盖文件：0个

### 目录结构与子目录追溯

本目录为叶子目录，没有直接子目录。

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | 来源与身份 | SHA-256位置 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [`12模块_性能汇总候选.csv`](./12模块_性能汇总候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_3accc959cc57) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`4类代表算子候选.csv`](./4类代表算子候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_657bbb94eddf) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`53x5_最差精度候选.csv`](./53x5_最差精度候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_53562f8b9951) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`53算子_12模块映射.csv`](./53算子_12模块映射.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_3b8d62fcd6d0) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`53算子_全case性能区间候选.csv`](./53算子_全case性能区间候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_d383c06b6201) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`53算子_演示case性能候选.csv`](./53算子_演示case性能候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_071b901f8846) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 当前目录CSV逐文件schema映射和逐列语义说明。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 当前目录递归文件身份、来源、大小和SHA-256清单。 | [字段说明](./CSV字段说明.md#schema_7f4d2cbb2001) | 本地治理文件 | 由上级folder manifest记录 | TRACEABLE |
| [`README.md`](./README.md) | governance | 当前目录阅读入口、文件追溯与快速导航。 | NA | 本地治理文件 | 本目录folder_manifest.csv | TRACEABLE |
| [`全算子_逐case绘图候选.csv`](./全算子_逐case绘图候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_7ed5a58734dd) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |
| [`窗函数_435case热力图候选.csv`](./窗函数_435case热力图候选.csv) | csv | 结构化表格证据；列含义见CSV字段说明。 | [字段说明](./CSV字段说明.md#schema_a7b6403aceeb) | 详见folder_manifest.csv | 本目录folder_manifest.csv | TRACEABLE |

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
