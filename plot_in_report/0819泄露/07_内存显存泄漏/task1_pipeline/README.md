# task1_pipeline

长周期CPU内存、GPU显存、增长斜率和泄漏门禁。 本README按当前文件树自动生成；证据资格以状态字段和全局manifest为准。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 当前目录真实状态与导航

> 同步时间：`2026-08-18T12:10:34+08:00`。本节仅从当前文件树和`folder_manifest.csv`生成；任何文件增删改后必须重新同步。

### 目录身份

- 归档相对路径：`07_内存显存泄漏/task1_pipeline`
- 用途：长周期CPU内存、GPU显存、增长斜率和泄漏门禁。
- 直接子目录：1个；直接文件：3个
- 递归文件：17个；递归CSV：3个
- manifest非治理证据行：8；`formal_eligible=true`行：8

### 实际证据边界

- stage：stage11
- task：Task1
- backend：fft_thrust
- dtype：ComplexFP32
- run_type：formal
- evidence_status：RECOVERED_VERIFIED_PASS
- 是否可用于正式结论，必须继续核对``formal_eligible``、run_id、commit、case数、采样次数及同目录摘要；“有文件”不等于“完整PASS”。

### 子目录追溯

| 子目录 | 递归文件 | 递归CSV | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `RB_20260817T123624Z_STAGE11_TASK1_PIPELINE_FFT_THRUST_PASS` | 14 | 2 | [README](./RB_20260817T123624Z_STAGE11_TASK1_PIPELINE_FFT_THRUST_PASS/README.md) | [字段说明](./RB_20260817T123624Z_STAGE11_TASK1_PIPELINE_FFT_THRUST_PASS/CSV字段说明.md) | [清单](./RB_20260817T123624Z_STAGE11_TASK1_PIPELINE_FFT_THRUST_PASS/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256与来源 |
| --- | --- | --- | --- | --- |
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

- [返回上级](../README.md)
- [归档根README](../../README.md)
- [当前目录CSV字段说明](./CSV字段说明.md)
- [当前目录完整文件清单](./folder_manifest.csv)
- [全局artifact manifest](../../01_总清单与校验/artifact_manifest.csv)
- [全局SHA-256总表](../../01_总清单与校验/sha256_manifest.csv)
- [回收批次](../../01_总清单与校验/recovery_batches.csv)
- [当前缺失项](../../01_总清单与校验/missing_artifacts.csv)
<!-- AUTO_ARCHIVE_INDEX_END -->
