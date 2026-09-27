# 总清单与校验

## 当前文件

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| `artifact_manifest.csv` | 主要归档文件的身份、来源和关联关系 | 已新增阶段10源清单、配对case清单、正式摘要和原始SHA清单入口 |
| `sha256_manifest.csv` | 主要入口文件SHA-256和校验状态 | 阶段10主要入口均已登记容器、宿主机和Windows一致状态 |
| `recovery_batches.csv` | 每次回收的来源、目标、数量、大小和校验状态 | 阶段10正式9004文件逐项回收通过，最终关闭待错误压缩载体清理授权 |
| `missing_artifacts.csv` | 当前活动缺失项 | 已登记阶段10错误回收压缩载体的授权清理项，不静默删除 |

## 阶段07/08回收信息

- 文件数：17,301
- 历史容器导出来源：`/tmp/zkx_stage07_08_20260810T1446CST.tar.gz`；仅记录历史来源身份，现行规则禁止继续采用压缩回收
- 历史运输文件大小：13,097,773 bytes
- 历史运输文件SHA-256：`619370544e5a7c615ca88dd0b69e5bff018a7427973184363b8ff22658d9867e`
- 逐文件清单：`/tmp/zkx_stage07_08_20260810T1446CST_files.sha256`
- 清单SHA-256：`9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0`
- 历史回收批次记录：`recovery_batches/RB_20260810T1446CST_STAGE07_08/`；本地压缩文件已于2026-08-14受控删除
- 阶段07展开数据：`../04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/`
- 阶段08展开数据：`../04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/`
- 阶段07文件数：3,885
- 阶段08文件数：13,416
- 当前状态：`RECOVERED_EXPANDED_AND_INDEXED_VERIFIED`

历史回收时容器、宿主机和Windows运输文件SHA-256一致；17,301个文件已在最终展开目录逐项校验通过。根据现行`AGENT.md`，最终入口是可直接读取的CSV、JSON、TXT、LOG和README。本地历史`tar.gz`已在确认展开文件与逐文件SHA-256清单完整后于2026-08-14删除，当前全归档压缩包扫描数量为0。

## 阶段09算子异常输入

- 远程来源：`/home/usr_02/ZKX_dev/results/stage09_operator_abnormal_159_617051d.tar.gz`
- 来源SHA-256：`d0ba30d6a0e3e599e6e64053f3d08dba2389f6b14f1c515be5b117111e820857`
- 最终本地入口：`../05_异常输入/stage09_operator_abnormal_159_617051d/`
- 范围：53个算子、159条异常case，159 PASS、159安全退出、0条消费异常输出
- 当前状态：`RECOVERED_EXPANDED_VERIFIED_NO_LOCAL_ARCHIVE`

阶段09本地结果只保留展开CSV、LOG、Markdown和SHA-256清单，不保留阶段09压缩包。首次`lfilter_zi`失败尝试已分流到`90_历史证据`，没有混入正式159条总表。

<!-- AUTO_ARCHIVE_INDEX_BEGIN -->
## 自动生成的完整追溯索引

> 生成时间：`2026-08-17T16:56:58+08:00`；新增、移动、删除或修改文件后必须重新生成。

- 归档相对路径：`01_总清单与校验`
- 直接子目录：1个
- 直接文件：8个
- 递归文件：37 个
- 递归CSV：17 个

### 子目录追溯

| 子目录 | 递归文件数 | 递归CSV数 | README | CSV字段说明 | manifest |
| --- | ---: | ---: | --- | --- | --- |
| `recovery_batches` | 29 | 12 | [README](./recovery_batches/README.md) | [字段说明](./recovery_batches/CSV字段说明.md) | [清单](./recovery_batches/folder_manifest.csv) |

### 本目录直接文件

| 文件 | 类型 | 作用 | CSV字段说明 | SHA-256位置 |
| --- | --- | --- | --- | --- |
| [`整改闭环记录_20260814.md`](./整改闭环记录_20260814.md) | md | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`artifact_manifest.csv`](./artifact_manifest.csv) | csv | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`CSV字段说明.md`](./CSV字段说明.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`folder_manifest.csv`](./folder_manifest.csv) | governance | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`missing_artifacts.csv`](./missing_artifacts.csv) | csv | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`README.md`](./README.md) | governance | 本目录原始证据或治理文件。 | NA | `folder_manifest.csv` |
| [`recovery_batches.csv`](./recovery_batches.csv) | csv | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |
| [`sha256_manifest.csv`](./sha256_manifest.csv) | csv | 本目录原始证据或治理文件。 | [字段说明](./CSV字段说明.md) | `folder_manifest.csv` |

### 阅读顺序

1. 先读本README；2. CSV先查同目录字段说明；3. 文件身份查folder manifest；4. 跨目录身份查全局回收批次。
<!-- AUTO_ARCHIVE_INDEX_END -->
