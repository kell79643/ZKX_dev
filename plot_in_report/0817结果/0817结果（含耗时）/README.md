# 全国总决赛证据归档阅读指南

本文件是整个归档的唯一第一入口。任何开发者、答辩人员、PPT或技术文档制作者以及后续AI，都必须从这里开始，不能直接随机打开深层CSV得出结论。

## 必须遵守的阅读顺序

1. **先完整阅读本`README.md`**：确认归档分层、正确入口和当前证据边界。
2. **再阅读`AGENT.md`**：了解全阶段回收、零压缩包、逐目录README、逐文件追溯和CSV字段说明硬规则。
3. **再阅读根目录`CSV字段说明.md`**：从根级索引逐层定位任意CSV的字段说明。
4. **需要核对文件身份时阅读`folder_manifest.csv`**：查看相对路径、大小、SHA-256、来源和证据状态。
5. **再进入`00_归档规范/`**：理解formal、smoke、history、候选、正式资格和派生数据边界。
6. **再进入`01_总清单与校验/`**：查看artifact、SHA-256、回收批次和当前缺失项。
7. **最后按问题进入具体证据目录**：Task进入`03_任务结果/`，算子进入`04_算子结果/`，异常进入`05_异常输入/`，其余阶段按一级目录名称进入。

## 阅读任何子目录的固定方法

进入任意子目录后重复以下顺序：

1. 先读该目录`README.md`；
2. 若要读CSV，先在同目录`CSV字段说明.md`按文件名找到schema；
3. 完整理解行粒度、主键、配对键、列类型、单位、NA规则和公式后，再打开CSV；
4. 用`folder_manifest.csv`核对文件来源、大小和SHA-256；
5. 沿README中的相对链接反查日志、输入证据、原始结果、图表及文档引用；
6. 不跨formal、smoke、history、commit、backend、dtype、scale或timing scope静默合并。

## CSV阅读硬规则

- 不允许只根据列名猜含义；
- 不允许跳过`CSV字段说明.md`直接把数值用于PPT；
- `cpu_gpu_speedup = CPU mean_ms / GPU mean_ms`，大于1才表示GPU更快；
- MSE、RMSE、Relative L2和Relative L∞必须结合`accuracy_reference`与阈值来源；
- 离散输出的数值误差可能为`NA`，此时读取`exact_match、mismatch_count、semantic_check`；
- P50/P95/P99、min/max、std和CV必须结合warmup、measured runs、同步方式和timing scope；
- 路径较深的原始CSV优先由支持长路径的工具读取，不修改原件。

## 当前范围边界

- 已存在的证据按实际状态整理，不把计划数据写成当前PASS；
- Task1/Task2的`fft_thrust` pipeline正式110-case矩阵已回收到`03_任务结果/`；任务级异常证据已回收到`05_异常输入/`；其他backend、逐Step和长周期证据仍必须以实际目录、回收批次和`formal_eligible`状态为准；
- `missing_artifacts.csv`在本次本地归档整改全部通过后保持表头、无活动数据行；后续新缺口必须重新登记；
- 本归档禁止任何压缩包，最终应满足压缩包扫描数量为0。

## 常用入口

| 需要回答的问题 | 先读哪里 |
| --- | --- |
| 全阶段归档规则 | `AGENT.md` |
| 任意CSV列含义 | 当前目录`CSV字段说明.md`，从根目录逐级进入 |
| 任意文件来源和SHA-256 | 当前目录`folder_manifest.csv` |
| 证据资格与分层 | `00_归档规范/README.md` |
| artifact、SHA-256和回收批次 | `01_总清单与校验/README.md` |
| Task1/Task2 | `03_任务结果/README.md` |
| 53个算子及绘图候选 | `04_算子结果/README.md` |
| 异常输入 | `05_异常输入/README.md` |
| 稳定性、泄漏、FFT和优化 | `06_稳定性/`至`09_性能优化/`各自README |
| 演示与离线备用 | `10_演示系统/README.md` |
| 绘图数据与图表 | `12_绘图数据/README.md`、`13_图表/README.md` |
| 截图和文档引用 | `14_截图/`至`16_技术文档引用/` |
| 完整日志 | `17_完整日志/README.md` |
| 历史或冲突证据 | `90_历史证据/README.md`、`99_隔离与待确认/README.md` |

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 当前目录真实状态与导航

> 同步时间：`2026-08-17T22:14:16+08:00`。本节仅从当前文件树和`folder_manifest.csv`生成；任何文件增删改后必须重新同步。

### 目录身份

- 归档相对路径：`.`
- 用途：当前路径下证据、治理文件和子目录的可追溯入口。
- 直接子目录：20个；直接文件：4个
- 递归文件：46131个；递归CSV：18192个
- manifest非治理证据行：28466；`formal_eligible=true`行：11017

### 实际证据边界

- stage：MULTIPLE_OR_NOT_APPLICABLE, stage05, stage06, stage07, stage08, stage09, stage10, stage11
- task：NA, Task1, Task1+Task2, Task2
- backend：cpu+gpu, cuda_runtime, dlfft, fft_thrust, not_applicable, selected_backend
- dtype：ComplexFP32, FP16, FP32, INDEX_INT64, INT16, INT32, INT8, mixed …（共9种）
- run_type：developer_formal, formal, smoke
- evidence_status：ARCHIVED_NEEDS_CLASSIFICATION, AVAILABLE_FORMAL, FORMAL_CANDIDATE_NEEDS_QUALIFICATION, HISTORY, RECOVERED_VERIFIED, RECOVERED_VERIFIED_FAIL, RECOVERED_VERIFIED_PASS, SMOKE …（共9种）
- 是否可用于正式结论，必须继续核对``formal_eligible``、run_id、commit、case数、采样次数及同目录摘要；“有文件”不等于“完整PASS”。

### 子目录追溯

| 子目录 | 递归文件 | 递归CSV | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `00_归档规范` | 4 | 1 | [README](./00_归档规范/README.md) | [字段说明](./00_归档规范/CSV字段说明.md) | [清单](./00_归档规范/folder_manifest.csv) |
| `01_总清单与校验` | 47 | 21 | [README](./01_总清单与校验/README.md) | [字段说明](./01_总清单与校验/CSV字段说明.md) | [清单](./01_总清单与校验/folder_manifest.csv) |
| `02_环境与版本` | 3 | 1 | [README](./02_环境与版本/README.md) | [字段说明](./02_环境与版本/CSV字段说明.md) | [清单](./02_环境与版本/folder_manifest.csv) |
| `03_任务结果` | 2703 | 1339 | [README](./03_任务结果/README.md) | [字段说明](./03_任务结果/CSV字段说明.md) | [清单](./03_任务结果/folder_manifest.csv) |
| `04_算子结果` | 24955 | 11672 | [README](./04_算子结果/README.md) | [字段说明](./04_算子结果/CSV字段说明.md) | [清单](./04_算子结果/folder_manifest.csv) |
| `05_异常输入` | 317 | 128 | [README](./05_异常输入/README.md) | [字段说明](./05_异常输入/CSV字段说明.md) | [清单](./05_异常输入/folder_manifest.csv) |
| `06_稳定性` | 18024 | 5008 | [README](./06_稳定性/README.md) | [字段说明](./06_稳定性/CSV字段说明.md) | [清单](./06_稳定性/folder_manifest.csv) |
| `07_内存显存泄漏` | 30 | 6 | [README](./07_内存显存泄漏/README.md) | [字段说明](./07_内存显存泄漏/CSV字段说明.md) | [清单](./07_内存显存泄漏/folder_manifest.csv) |
| `08_FFT后端` | 3 | 1 | [README](./08_FFT后端/README.md) | [字段说明](./08_FFT后端/CSV字段说明.md) | [清单](./08_FFT后端/folder_manifest.csv) |
| `09_性能优化` | 3 | 1 | [README](./09_性能优化/README.md) | [字段说明](./09_性能优化/CSV字段说明.md) | [清单](./09_性能优化/folder_manifest.csv) |
| `10_演示系统` | 3 | 1 | [README](./10_演示系统/README.md) | [字段说明](./10_演示系统/CSV字段说明.md) | [清单](./10_演示系统/folder_manifest.csv) |
| `11_规范化数据` | 3 | 1 | [README](./11_规范化数据/README.md) | [字段说明](./11_规范化数据/CSV字段说明.md) | [清单](./11_规范化数据/folder_manifest.csv) |
| `12_绘图数据` | 3 | 1 | [README](./12_绘图数据/README.md) | [字段说明](./12_绘图数据/CSV字段说明.md) | [清单](./12_绘图数据/folder_manifest.csv) |
| `13_图表` | 3 | 1 | [README](./13_图表/README.md) | [字段说明](./13_图表/CSV字段说明.md) | [清单](./13_图表/folder_manifest.csv) |
| `14_截图` | 3 | 1 | [README](./14_截图/README.md) | [字段说明](./14_截图/CSV字段说明.md) | [清单](./14_截图/folder_manifest.csv) |
| `15_PPT引用` | 3 | 1 | [README](./15_PPT引用/README.md) | [字段说明](./15_PPT引用/CSV字段说明.md) | [清单](./15_PPT引用/folder_manifest.csv) |
| `16_技术文档引用` | 3 | 1 | [README](./16_技术文档引用/README.md) | [字段说明](./16_技术文档引用/CSV字段说明.md) | [清单](./16_技术文档引用/folder_manifest.csv) |
| `17_完整日志` | 3 | 1 | [README](./17_完整日志/README.md) | [字段说明](./17_完整日志/CSV字段说明.md) | [清单](./17_完整日志/folder_manifest.csv) |
| `90_历史证据` | 11 | 4 | [README](./90_历史证据/README.md) | [字段说明](./90_历史证据/CSV字段说明.md) | [清单](./90_历史证据/folder_manifest.csv) |
| `99_隔离与待确认` | 3 | 1 | [README](./99_隔离与待确认/README.md) | [字段说明](./99_隔离与待确认/CSV字段说明.md) | [清单](./99_隔离与待确认/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256与来源 |
| --- | --- | --- | --- | --- |
| [`AGENT.md`](./AGENT.md) | md | 原始证据或归档治理文件。 | NA | [`folder_manifest.csv`](./folder_manifest.csv) |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 当前目录CSV逐文件schema、逐列语义和子目录索引。 | NA | [`folder_manifest.csv`](./folder_manifest.csv) |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 当前目录递归文件身份、来源、大小和SHA-256清单。 | [manifest schema](./CSV字段说明.md) | [`folder_manifest.csv`](./folder_manifest.csv) |
| [`README.md`](./README.md) | governance | 当前目录阅读入口、证据边界和导航。 | NA | [`folder_manifest.csv`](./folder_manifest.csv) |

### 正确使用方法

1. 先读本README的“实际证据边界”，不把目录存在解读为阶段PASS；
2. 读任何CSV前，先在同目录``CSV字段说明.md``按文件名定位schema；
3. 用``folder_manifest.csv``核对来源路径、run_id、commit、backend、dtype、证据状态、大小和SHA-256；
4. 耗时结论必须区分CPU/GPU、warmup/measured、timing scope和竞争环境；
5. 不跨formal/smoke/history、commit、backend、dtype或scale静默合并；需派生数据时写入``11_``/``12_``，不改原始证据。

### 快速导航

- [归档根README](./README.md)
- [当前目录CSV字段说明](./CSV字段说明.md)
- [当前目录完整文件清单](./folder_manifest.csv)
- [全局artifact manifest](./01_总清单与校验/artifact_manifest.csv)
- [全局SHA-256总表](./01_总清单与校验/sha256_manifest.csv)
- [回收批次](./01_总清单与校验/recovery_batches.csv)
- [当前缺失项](./01_总清单与校验/missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
