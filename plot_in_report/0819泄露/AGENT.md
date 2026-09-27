# 全国总决赛全阶段证据回收、目录组织与追溯规则

本文件适用于阶段00至阶段15及后续展示补强、复测、重跑、图表、截图、PPT引用和技术文档引用。它不是阶段07、阶段08专用规则。

本文件约束所有后续AI和人工会话对以下归档根目录执行的操作：

```text
D:\05_Virgo\Desktop\ZKX_dev\全国总决赛证据归档\
```

核心要求只有四条，任何阶段都不能例外：

1. 容器内真实数据必须按规定目录结构逐文件回收到远程宿主机，再逐文件回收到Windows归档根；
2. 回收链路、宿主机临时目录和最终归档中不得创建、保留或交付任何压缩包；
3. 每个已创建子文件夹都必须有`README.md`，且该README必须能够追溯该子文件夹内的全部文件，不能只写一句用途说明。
4. 每个CSV都必须有明确的逐列说明；统一使用UTF-8 Markdown文件`CSV字段说明.md`作为AI和人工阅读的主格式，并在各级目录建立可递归追溯链。

## 1. 适用范围

本规则覆盖：

- 阶段00：人工基线冻结；
- 阶段01：只读仓库审查；
- 阶段02：任务流程真值冻结；
- 阶段03：测试规范与配置设计；
- 阶段04：公共测试基础设施；
- 阶段05：任务一；
- 阶段06：任务二；
- 阶段07：任务涉及算子；
- 阶段08：剩余算子；
- 阶段09：异常输入与错误码；
- 阶段10：随机扰动稳定性；
- 阶段11：内存和显存泄漏；
- 阶段12：FFT后端；
- 阶段13：性能优化；
- 阶段14：最佳规模与演示系统；
- 阶段15：最终回归、截图和证据闭环；
- 工作包之外的展示补强、补测、重跑和答辩备用证据；
- 规范化数据、绘图数据、图表、截图、PPT引用和技术文档引用。

不得因为某阶段当前没有数据，就把该阶段从目标目录、清单或缺失项管理中删除。没有真实文件时记录为计划或缺失；不得创建虚假结果填充目录。

## 2. 必须遵守的上级环境规则

只要涉及ZQ500同步、构建、运行、测试、容器或远程环境判断，必须先完整阅读：

1. `ZKX_dev/代码规范.txt`；
2. 活跃源码仓库中的`ZKX/AGENTS.md`；
3. `ZKX/docs/development/ZQ500_RUNTIME.md`的“标准操作流程”。

远程环境基准：

- 主机：`usr_02@10.110.12.10`；
- 远程宿主机项目目录：`/home/usr_02/ZKX_dev`；
- 容器：`gpu_02`；
- 容器项目目录：`/tmp/ZKX_dev`；
- SDK初始化：`source /zq500/sdk/env.sh`。

不得自行发明新的`sudo`、Docker、shell包装、反向源码同步或owner修复方式。不得索取、等待、代输或记录sudo密码。若现有标准入口不支持“无压缩、逐文件导出”，必须停止并向用户报告缺少合规导出入口，不能退回压缩包方案。

## 3. 规定的归档目录结构

只有确定存在真实内容或已经明确需要时才创建目录，避免无意义空目录；但一旦创建，必须同时创建合格的`README.md`。

```text
全国总决赛证据归档/
├─ AGENT.md
├─ README.md
├─ 00_归档规范/
│  └─ README.md
├─ 01_总清单与校验/
│  ├─ README.md
│  ├─ artifact_manifest.csv
│  ├─ sha256_manifest.csv
│  ├─ recovery_batches.csv
│  └─ missing_artifacts.csv
├─ 02_环境与版本/
│  ├─ README.md
│  ├─ hardware/README.md
│  ├─ software/README.md
│  ├─ git/README.md
│  └─ build/README.md
├─ 03_任务结果/
│  ├─ README.md
│  ├─ Task1/
│  │  ├─ README.md
│  │  ├─ formal/README.md
│  │  ├─ smoke/README.md
│  │  └─ terminal_logs/README.md
│  └─ Task2/
│     ├─ README.md
│     ├─ formal/README.md
│     ├─ smoke/README.md
│     └─ terminal_logs/README.md
├─ 04_算子结果/
│  ├─ README.md
│  ├─ task_operators/README.md
│  ├─ remaining_operators/README.md
│  └─ modules/
│     ├─ README.md
│     └─ <module_name>/
│        ├─ README.md
│        └─ <operator_name>/
│           ├─ README.md
│           ├─ formal/README.md
│           └─ terminal_logs/README.md
├─ 05_异常输入/README.md
├─ 06_稳定性/README.md
├─ 07_内存显存泄漏/README.md
├─ 08_FFT后端/
│  ├─ README.md
│  ├─ fft_thrust/README.md
│  ├─ dlfft/README.md
│  └─ paired_comparisons/README.md
├─ 09_性能优化/
│  ├─ README.md
│  ├─ baselines/README.md
│  ├─ optimized/README.md
│  ├─ ablations/README.md
│  └─ paired_comparisons/README.md
├─ 10_演示系统/
│  ├─ README.md
│  ├─ capabilities/README.md
│  ├─ live_run_logs/README.md
│  └─ offline_fallback/README.md
├─ 11_规范化数据/
│  ├─ README.md
│  ├─ validated/README.md
│  └─ rejected/README.md
├─ 12_绘图数据/README.md
├─ 13_图表/
│  ├─ README.md
│  ├─ ppt_main/README.md
│  ├─ ppt_backup/README.md
│  ├─ technical_document/README.md
│  └─ editable_source/README.md
├─ 14_截图/
│  ├─ README.md
│  └─ screenshot_index.csv
├─ 15_PPT引用/
│  ├─ README.md
│  └─ page_evidence_index.csv
├─ 16_技术文档引用/
│  ├─ README.md
│  └─ section_evidence_index.csv
├─ 17_完整日志/README.md
├─ 90_历史证据/README.md
└─ 99_隔离与待确认/README.md
```

实际阶段目录可以在上述一级目录下继续按`stage、task、module、operator、run_id、case_id、backend、dtype`建立子目录，但不得另起一个脱离此骨架的归档根。

## 4. 全阶段到目录的默认路由

一个阶段可以写入多个证据目录，但每个文件必须有唯一主归档位置，其他位置只能通过清单或引用文件关联，不能复制出多份身份不明的“原件”。

| 阶段或产物 | 主归档目录 | 典型证据 |
| --- | --- | --- |
| 阶段00～04 | `00_归档规范/`、`01_总清单与校验/`、`02_环境与版本/`、`17_完整日志/` | 基线、环境、版本、配置、构建、审计日志 |
| 阶段05 | `03_任务结果/Task1/` | Task1正式、smoke、分Step和pipeline结果及日志 |
| 阶段06 | `03_任务结果/Task2/` | Task2正式、smoke、分Step和pipeline结果及日志 |
| 阶段07 | `04_算子结果/task_operators/`及`modules/` | 任务涉及算子完整case |
| 阶段08 | `04_算子结果/remaining_operators/`及`modules/` | 剩余算子完整case |
| 阶段09 | `05_异常输入/` | 异常case、错误码、安全退出和日志 |
| 阶段10 | `06_稳定性/` | 随机扰动、重复运行、分布和失败样本 |
| 阶段11 | `07_内存显存泄漏/` | 内存、显存、长循环、泄漏判定和轨迹 |
| 阶段12 | `08_FFT后端/` | `fft_thrust`、`dlfft`及同case配对比较 |
| 阶段13 | `09_性能优化/` | baseline、optimized、ablation及配对结果 |
| 阶段14 | `10_演示系统/` | 能力清单、现场运行日志和离线备用证据 |
| 阶段15 | 按对象写入`02`至`10`，截图进`14_截图/`，完整日志进`17_完整日志/` | 最终回归、截图、构建与运行闭环 |
| 规范化处理 | `11_规范化数据/` | 校验通过和拒绝数据 |
| 绘图输入 | `12_绘图数据/` | 可复算的窄表、宽表和图表输入 |
| 图表成品 | `13_图表/` | 主PPT、备份页、技术文档和可编辑源 |
| 截图 | `14_截图/` | 截图文件和截图索引 |
| PPT引用 | `15_PPT引用/` | 页面到证据的映射 |
| 技术文档引用 | `16_技术文档引用/` | 章节到证据的映射 |
| 历史结果 | `90_历史证据/` | smoke、旧结果、失败后保留结果和被替代重跑 |
| 身份冲突 | `99_隔离与待确认/` | 来源、commit、run_id或内容无法确认的文件 |

`git_dirty=true`不自动进入`99_隔离与待确认`。dirty但完整且身份明确的数据按真实阶段正常归档。最终证据也不要求绑定同一个最终commit，但每条证据必须保留自己的真实commit和dirty状态。

## 5. 绝对禁止压缩包

本规则中的“不得有压缩包”适用于整个回收链路，而不只是最终展示目录。

禁止创建、传输、保留或交付：

- `.zip`；
- `.7z`；
- `.rar`；
- `.tar`；
- `.tar.gz`；
- `.tgz`；
- `.gz`；
- `.bz2`；
- `.xz`；
- 其他归档、打包或压缩容器；
- 仅修改扩展名、实际内容仍为归档格式的文件。

必须采用“目录树和普通文件逐项传输”：

```text
gpu_02容器普通文件树
→ 远程宿主机普通文件树
→ Windows归档普通文件树
```

要求：

- 容器到宿主机不得先打包；
- 宿主机到Windows不得先打包；
- 不得把压缩包称为“内部底稿”“运输载体”或“追溯包”后保留；
- 大文件或大量小文件不能成为改用压缩包的理由；
- 若传输工具效率不足，应改进逐文件同步、断点续传或批次清单，不得改变无压缩要求；
- 每批回收结束必须扫描容器导出目录、宿主机导出目录和Windows目标目录，压缩包数量必须为0；
- 若发现历史压缩包，必须先确认其内容已完整展开并通过逐文件校验，再经用户授权删除；在删除前批次不得标记为最终合规。

## 6. 无压缩回收标准流程

### 6.1 回收前盘点

在容器内对待回收目录生成普通文本或CSV清单，不得生成压缩包。至少记录：

- 相对路径；
- 文件名；
- 文件类型；
- 文件大小；
- SHA-256；
- 修改时间；
- stage、test object、run_id、case_id；
- backend、dtype、scale；
- git commit、git dirty；
- formal、smoke、history或待确认状态。

清单文件本身也是证据，必须在其父目录README中登记。

### 6.2 容器到远程宿主机

- 先确定容器来源绝对路径；
- 先确定经项目规则批准的无压缩导出方式；
- 先确定远程宿主机临时导出目录、owner和权限；
- 保持原目录层次逐文件复制；
- 不得反向覆盖`/home/usr_02/ZKX_dev`源码树；
- 不得把root-owned文件混入远程可写源码目录；
- 不得使用源码同步脚本反向同步`results/`；
- 每个文件复制后按源清单校验大小和SHA-256；
- 失败、权限错误、文件数不一致或哈希不一致时立即停止。

若当前sudoers或标准远程入口不允许合规的无压缩导出，停止操作并向用户说明需要新增哪一个受控入口。不得试探其他sudo或Docker命令。

### 6.3 远程宿主机到Windows

- 从宿主机普通文件树逐文件复制到规定的Windows目标目录；
- 支持断点续传时仍以源清单为准，不以终端输出判断完整性；
- 保持相对目录结构，不能把不同阶段、run或证据状态扁平化混合；
- Windows落盘后逐文件重算SHA-256；
- 容器、宿主机和Windows三处的相对路径、文件大小、SHA-256必须一致；
- Windows最终文件数必须等于源清单文件数；
- 只有逐文件核验全部通过，才能关闭missing项。

### 6.4 最终分流

- 按本文件第3、4节的目录结构分流；
- 原始证据不得为了适应目录或绘图而修改内容；
- formal、smoke、历史和冲突文件不得静默混放；
- 文件移动到最终目录后再次核对README覆盖、清单路径、文件数和SHA-256；
- 临时导出目录只有在Windows最终归档校验通过并确认不再需要后才能清理；
- 未经用户明确批准，不删除容器原件或远程源结果。

## 7. 每个子文件夹都必须有README.md

“每个子文件夹”包括：

- 一级目录；
- 阶段目录；
- task、module、operator目录；
- formal、smoke、history目录；
- run_id目录；
- case目录；
- backend、dtype、scale目录；
- 日志、截图、图表和引用目录；
- 任何脚本或回收流程新建的中间层目录。

创建目录和创建README必须是同一个操作切片。不得先批量生成目录，之后再承诺补README。

若目录直接或递归包含CSV，创建目录、创建README和创建`CSV字段说明.md`必须属于同一个操作切片。

### 7.1 README必须追溯本目录全部文件

每个README至少必须回答：

1. 本目录是什么阶段、对象和证据层；
2. 本目录直接包含哪些子目录；
3. 本目录直接包含哪些文件；
4. 本目录递归共包含多少个文件；
5. 每个文件的来源路径、大小、SHA-256、身份和证据状态在哪里查；
6. 哪些文件是原始证据，哪些是规范数据、派生数据或引用；
7. 正式结果、日志、输入证据、manifest和缺失项分别在哪里；
8. 如何从本目录文件反查容器源、run_id、case_id、commit和相关日志；
9. 当前目录是否存在缺失、冲突、失败或不具备正式资格的文件。

README不能只写“本目录用于保存某类结果”。

### 7.2 非叶子目录README

非叶子目录README必须包含：

#### 目录结构

只展示当前目录和直接子目录，不用复制整个全局目录树。

#### 子目录追溯表

至少包含：

| 字段 | 要求 |
| --- | --- |
| `subdirectory` | 直接子目录名称和相对链接 |
| `purpose` | 子目录用途 |
| `stage_or_object` | 对应阶段或测试对象 |
| `direct_file_count` | 子目录直接文件数 |
| `recursive_file_count` | 子目录递归文件数 |
| `readme` | 子目录README相对路径 |
| `folder_manifest` | 子目录完整文件清单相对路径 |
| `evidence_status` | 当前证据状态 |
| `notes` | 缺失、冲突、历史或使用边界 |

#### 本目录直接文件表

README自身以外的每个直接文件必须逐项列出，不能使用`*.csv`等通配概括。

若直接文件是CSV，直接文件表必须增加`column_document`和`schema_id`，分别指向本目录`CSV字段说明.md`中的具体锚点及该CSV绑定的字段模式。

### 7.3 叶子目录README

叶子目录README必须逐文件列出本目录全部文件，至少包含：

| 字段 | 含义 |
| --- | --- |
| `filename` | 文件名 |
| `artifact_kind` | CSV、JSON、日志、截图、配置、图表等 |
| `purpose` | 文件证明什么 |
| `source_host` | 来源主机 |
| `source_container` | 来源容器 |
| `source_path` | 容器内原始路径 |
| `run_id` | 运行ID |
| `case_id` | case ID |
| `git_commit` | 运行commit |
| `git_dirty` | 真实dirty状态 |
| `backend` | 后端 |
| `dtype` | 数据类型 |
| `scale_id` | 输入规模 |
| `file_size` | 字节数 |
| `sha256` | Windows最终文件SHA-256 |
| `evidence_status` | formal、smoke、history、conflicted等 |
| `formal_eligible` | 是否可用于正式结论 |
| `related_files` | 关联CSV、日志、输入证据、截图或图表 |
| `column_document` | CSV专用；对应`CSV字段说明.md`及具体schema锚点 |
| `schema_id` | CSV专用；与真实表头绑定的字段模式ID |
| `notes` | NA规则、失败信息或使用边界 |

### 7.4 每个目录的完整文件清单

文件较多时，README不必把数千行全部嵌入正文，但必须在README中提供该目录的完整追溯入口：

```text
folder_manifest.csv
```

`folder_manifest.csv`必须递归覆盖该目录下的所有证据文件，并至少包含：

```text
relative_path,filename,artifact_kind,file_size,sha256,
source_host,source_container,source_path,
stage,test_object,task,step,module,operator,
run_id,case_id,backend,dtype,scale_id,run_type,
git_commit,git_dirty,evidence_status,formal_eligible,
related_csv,related_log,related_screenshot,notes
```

对于CSV文件，`folder_manifest.csv`还必须增加：

```text
csv_schema_id,csv_column_document,csv_header_sha256,csv_column_count
```

规则：

- README必须写明`folder_manifest.csv`的行数、覆盖文件数和生成/校验时间；
- `folder_manifest.csv`记录该目录全部后代证据文件；
- README和`folder_manifest.csv`属于治理文件，必须在README的“治理文件”小节单独登记；
- 为避免自哈希循环，`folder_manifest.csv`不记录自己的SHA-256；它的SHA-256写入上级目录manifest和`01_总清单与校验/sha256_manifest.csv`；
- 任一文件新增、移动、删除、重命名或内容变化后，必须同步更新本目录README、folder manifest以及所有上级目录的递归计数；
- `folder_manifest.csv`覆盖数必须等于实际递归证据文件数；
- README未覆盖文件数必须为0；
- README指向不存在文件数必须为0。

### 7.5 README快速导航

每个README必须提供：

- 上级目录；
- 根目录；
- 本目录完整manifest；
- 全局artifact manifest；
- SHA-256总表；
- recovery batch记录；
- missing项；
- 正式结果位置；
- 完整日志位置；
- 输入证据位置；
- 相关规范化数据、绘图数据、图表、截图、PPT页或技术文档章节。

### 7.6 所有CSV必须有逐列说明

统一选择UTF-8 Markdown文件作为列说明主格式，固定文件名为：

```text
CSV字段说明.md
```

选择Markdown的原因：

- AI可以直接读取，不需要打开Excel或执行转换；
- 中文语义、公式、单位、NA规则和判读注意事项表达清楚；
- 可以使用锚点让每个CSV精确链接到对应schema；
- 相同表头可以复用同一个schema定义，同时保留逐文件映射；
- Git diff、人工审阅和长文本检索均较方便。

禁止只依靠代码、CSV表头、口头说明或上级通用术语表推测列含义。CSV列名看似直观也必须登记。

#### 7.6.1 叶子目录要求

只要某目录直接包含一个或多个CSV，该目录必须有`CSV字段说明.md`，并包含：

1. 本目录CSV总数；
2. 已绑定字段说明的CSV数量；
3. 未绑定数量，合格值必须为0；
4. CSV文件到schema的逐文件映射表；
5. 每个schema的完整字段说明；
6. 表头校验时间和校验结论；
7. 相关原始数据、生成入口和使用边界。

逐文件映射表至少包含：

| 字段 | 要求 |
| --- | --- |
| `csv_file` | CSV文件名或相对路径，不允许通配符 |
| `schema_id` | 本文档内唯一字段模式ID |
| `schema_anchor` | 对应字段说明章节锚点 |
| `header_sha256` | 规范化后真实表头的SHA-256 |
| `column_count` | 实际列数 |
| `row_granularity` | 一行代表什么 |
| `primary_key` | 主键；没有则写明原因 |
| `pairing_key` | CPU/GPU或前后优化等配对键；不适用写`NA` |
| `source_or_generator` | 原始来源或派生生成入口 |
| `verification_status` | `HEADER_MATCH`等校验状态 |

同一目录中多个CSV的列名和顺序完全一致时，可以共用一个schema定义，但每个CSV仍必须在逐文件映射表中单独占一行。文件名相似不等于schema相同，必须按真实表头判断。

#### 7.6.2 每个schema的字段说明

每个schema必须先说明：

- schema ID和适用文件；
- CSV用途；
- 一行代表什么；
- 编码、分隔符和引号规则；
- 主键和唯一性约束；
- CPU/GPU配对键或其他关联键；
- 数据来源；
- 原始或派生身份；
- 正式证据使用边界。

然后按真实列顺序逐列说明：

| 字段 | 含义 |
| --- | --- |
| `column_order` | 从1开始的列序号 |
| `column_name` | 与CSV真实表头完全一致的列名 |
| `data_type` | string、integer、float、boolean、datetime、JSON text等 |
| `unit` | ms、bytes、count、ratio等；无单位写`dimensionless`或`NA` |
| `nullable` | 是否允许空值 |
| `na_values` | `NA`、空字符串等合法缺失表达及含义 |
| `description` | 该列准确业务含义 |
| `source_or_formula` | 原始来源或计算公式 |
| `interpretation` | 如何正确判读，越大/越小/阈值或枚举含义 |
| `example` | 来自真实数据的短示例；不得虚构正式数值 |
| `related_columns` | 相关列和联读方法 |

以下类型必须额外写清：

- 时间列：时区、格式和计时边界；
- 耗时列：单位、同步方式、warmup/measured runs；
- 分位数：计算对象及P50/P95/P99定义；
- CV：`std/mean`公式；
- speedup：分子、分母及`>1/<1`方向；
- 精度：参考输出、公式、阈值来源和NA规则；
- bytes：CPU heap、RSS、GPU显存的口径；
- 状态和错误码：所有枚举值及含义；
- digest和SHA-256：摘要对象；
- JSON文本列：JSON内部字段或其说明文件；
- 路径列：相对哪个目录解析；
- 派生统计列：输入文件、筛选、去重、聚合和排序规则。

#### 7.6.3 各级目录的递归CSV说明

“各级”要求如下：

- 直接包含CSV的目录：必须逐文件映射并逐列说明；
- 不直接含CSV、但子目录含CSV的目录：也必须有`CSV字段说明.md`；
- 上级`CSV字段说明.md`不重复粘贴所有子级字段定义，而是建立完整子目录索引；
- 子目录索引必须让读者从根目录逐级进入，最终定位到每个CSV及其schema锚点；
- 任一级不得出现“该目录下有CSV，但不知道字段说明在哪”的断链。

上级目录的子级CSV说明索引至少包含：

| 字段 | 要求 |
| --- | --- |
| `subdirectory` | 直接子目录 |
| `recursive_csv_count` | 该子目录递归CSV数量 |
| `csv_field_document` | 子目录`CSV字段说明.md`相对链接 |
| `documented_csv_count` | 已有字段说明的CSV数 |
| `undocumented_csv_count` | 未说明CSV数，合格值为0 |
| `schema_count` | 不同真实表头数量 |
| `verification_status` | 覆盖和表头核验结论 |

根目录、一级目录、阶段目录、对象目录、run目录和叶子目录由此形成连续的CSV说明链。

#### 7.6.4 表头一致性校验

每次生成或更新`CSV字段说明.md`后必须自动核对：

- CSV真实列名与schema列名完全一致；
- 列顺序完全一致；
- 列数一致；
- 没有schema声明但CSV不存在的列；
- 没有CSV存在但schema遗漏的列；
- 映射中的header SHA-256与真实表头一致；
- CSV文件数等于逐文件映射行数；
- 所有上级递归CSV计数等于实际数量；
- 所有schema锚点和相对链接有效。

任一项不一致时，状态必须标记为`CSV_COLUMN_DOCUMENT_MISMATCH`，不得把目录或回收批次标记为完成。

### 7.7 README与CSV字段说明的职责边界

- `README.md`负责追溯目录结构、全部文件、来源、身份、状态和快速导航；
- `CSV字段说明.md`负责逐个CSV的行粒度、主键、关联键和逐列语义；
- README中的每个CSV条目必须链接`CSV字段说明.md`的schema锚点；
- `CSV字段说明.md`中的每个CSV条目必须反向链接README和CSV文件；
- `folder_manifest.csv`负责机器可读的文件身份和哈希；
- 三者必须相互引用，不能相互替代，也不能出现孤立CSV。

## 8. 全局manifest与回收批次

### 8.1 `artifact_manifest.csv`

至少包含：

```text
artifact_id,artifact_kind,test_object,task,step,module,operator,
run_id,case_id,backend,dtype,scale_id,run_type,
git_commit,git_dirty,source_host,source_container,source_path,
archive_relative_path,filename,file_size,sha256,
generated_timestamp,recovered_timestamp,evidence_status,formal_eligible,
related_csv,related_log,related_screenshot,related_score_id,notes
```

PPT、技术文档和图表不得依赖无法在manifest中定位的孤立文件。

### 8.2 `sha256_manifest.csv`

登记所有主要入口、每个目录的`folder_manifest.csv`和需要独立引用的文件。源文件和Windows目标文件必须有明确校验状态。

### 8.3 `recovery_batches.csv`

每次回收必须记录：

- recovery batch ID和时间；
- 来源主机、容器和目录；
- 远程宿主机无压缩临时目录；
- Windows最终目录；
- stage、对象、run_id、commit、backend、dtype；
- 文件数、总字节数；
- 源逐文件清单和目标逐文件校验状态；
- README覆盖状态；
- 压缩包扫描数量；
- 回收状态、失败停止原因和备注。

### 8.4 `missing_artifacts.csv`

缺文件、缺字段、缺CPU/GPU配对、缺README、缺`CSV字段说明.md`、CSV列说明与真实表头不一致、缺manifest、哈希不一致和发现压缩包都必须登记。问题解决后保留原记录并更新为已闭环，不得静默删除。

## 9. 各类证据的最低内容

### 9.1 Task1和Task2

必须保留：

- 完整pipeline和每个Step；
- CPU与GPU成对结果；
- 相同输入digest；
- 分Step和整体耗时；
- 分Step和整体精度；
- 多规模、dtype、backend和输入来源；
- run ID、case ID、commit、dirty、配置SHA-256；
- 完整终端日志和退出码；
- formal与smoke分离。

任务一、任务二宣称完整运行时，必须覆盖两个后端的全部case矩阵。演示只运行一条pipeline不能替代完整离线证据。

### 9.2 算子

每个算子必须能够按不同输入类型和case提供：

- `MSE`；
- `RMSE`；
- `relative_l2`；
- `relative_linf`；
- CPU/GPU的`mean_ms`；
- CPU/GPU的`p50_ms、p95_ms、p99_ms`；
- CPU/GPU的`min_ms、max_ms、std_ms、cv`；
- `cpu_gpu_speedup`；
- warmup、measured runs和timing scope；
- 输入shape、dtype、规模、参数、内容摘要和digest；
- 原始逐轮`sample_index、latency_ms`；
- 状态、错误码、完整日志和资源轨迹。

加速比定义固定为：

```text
cpu_gpu_speedup = CPU mean_ms / GPU mean_ms
```

离散、索引和坐标输出允许数值误差为`NA`，但必须提供`exact_match、mismatch_count、semantic_check`。

### 9.3 异常、稳定性、泄漏、FFT和优化

- 异常：输入类型、预期错误、实际错误码、安全退出、CPU/GPU一致性和日志；
- 稳定性：随机种子、扰动参数、每轮状态、失败样本、统计分布和正式轮数；
- 泄漏：初始、峰值、结束值、增长斜率、循环数、判定规则和轨迹；
- FFT：同case、同dtype、同规模下`fft_thrust`和`dlfft`配对；
- 优化：baseline和optimized必须同case配对，保留消融和不利结果；
- 演示：能力清单、单目标命令、超时预算、现场日志和离线备用证据。

## 10. 原始、规范、派生和引用分层

```text
原始证据
→ 校验后的规范数据
→ 汇总数据
→ 绘图数据
→ 图表
→ PPT引用和技术文档引用
```

- 原始CSV、JSON、TXT、LOG、配置和截图只读保存；
- 不得修改原始CSV适应绘图；
- 清洗、筛选、配对和汇总必须生成新文件；
- 每个派生文件记录输入相对路径、输入SHA-256、处理入口、筛选规则、输入输出行数和输出SHA-256；
- 不同commit、run ID、backend、dtype、scale、timing scope和run type不得静默混合；
- 不删除失败、长尾、GPU慢于CPU、CV高或精度边界结果；
- 当前证据与计划证据必须分开，不得用计划值冒充实测值。

## 11. Windows路径和CSV可用性

- 面向开发者和评委的入口CSV放在对应阶段或对象目录较浅位置；
- 原始目录结构必须保留，但不能要求用户直接双击超长路径；
- README必须说明短路径入口和深层原始文件定位方法；
- 查看深层文件时可以复制单个查看副本到短目录，但归档原件和manifest路径不变；
- 每次回收后审计UTF-8、CSV语法、表头、行列一致性、空文件、只有表头无数据和路径长度；
- Excel科学计数法只是显示格式，不能据此认定数据丢失。

## 12. 禁止事项

- 禁止任何阶段使用或保留压缩包；
- 禁止只在Windows有文件而宿主机没有逐文件中间回收记录；
- 禁止只记录远程路径不回收真实文件；
- 禁止把容器结果反向覆盖源码树；
- 禁止把结果写进`ZKX/`源码仓库；
- 禁止创建没有README的子目录；
- 禁止README不列文件、无manifest或无法反查来源；
- 禁止README、folder manifest和实际目录三者数量不一致；
- 禁止存在没有`CSV字段说明.md`映射和逐列定义的CSV；
- 禁止只解释部分重要列而省略其他真实列；
- 禁止用文件名通配符替代逐个CSV到schema的映射；
- 禁止字段说明列名、列数或顺序与真实CSV表头不一致；
- 禁止上级目录无法递归定位后代CSV的字段说明；
- 禁止只提供压缩包、截图或汇总表而缺少原始CSV和日志；
- 禁止把smoke写成正式性能结论；
- 禁止把dirty自动隔离或自动判为无效；
- 禁止只挑最佳case代表全部case；
- 禁止隐藏GPU慢于CPU、失败或长尾结果；
- 禁止在本地运行依赖ZQ500硬件、CUDA、Python或MATLAB的项目测试。

## 12.1 README与目录manifest的强制同步入口

`ZKX/remote_scripts/sync_archive_readmes.ps1`是本归档的长期维护工具，不是一次性
整改脚本，使用后不得删除。

只要本归档中有文件新增、删除、移动、内容修改，或有目录新建、删除、重命名，
不论变化是原始证据、派生数据、全局清单、字段说明还是人工README前言，都必须
在该切片的其他归档写入完成后执行：

```powershell
Set-Location D:\05_Virgo\Desktop\ZKX_dev\ZKX
.\remote_scripts\sync_archive_readmes.ps1 `
  -ArchiveRoot 'D:\05_Virgo\Desktop\ZKX_dev\全国总决赛证据归档'
```

执行顺序固定为：

1. 先完成原始证据回收、三端SHA-256校验和最终路径分流；
2. 再更新`artifact_manifest.csv`、`sha256_manifest.csv`、`recovery_batches.csv`、
   `missing_artifacts.csv`和所有人工证据身份/语义说明；
3. 再更新直接包含CSV和上级递归索引所需的`CSV字段说明.md`；
4. 最后运行本脚本，同步全部README的当前证据边界、目录统计、子目录导航和
   直接文件表，并自底向上重算全部`folder_manifest.csv`的文件大小与SHA-256。

脚本会保留`<!-- AUTO_ARCHIVE_INDEX_BEGIN -->`之前的人工README前言，重建标记区域；
因此人工前言中的阶段状态仍必须在运行前更新，不得指望脚本猜测人工语义。

脚本不负责创建或修改任务/算子原始结果，不负责补写全局artifact和回收批次，
不负责猜测CSV字段语义，也不将“目录中存在文件”自动提升为“阶段PASS”。

当前归档规模下完整执行通常需要约5至10分钟。运行期间暂无终端输出属于正常现象，
必须等待唯一进程输出：

```text
[ARCHIVE][README-SYNC] directories=<actual> generated_at=<timestamp> status=PASS
```

不得因静默而重复启动脚本。该PASS只说明生成过程完成；交付前仍必须独立核对：

- 实际目录数=README数；
- README断链=0，递归文件/CSV空统计=0；
- 根`folder_manifest.csv`行数=除根manifest自身外的实际文件数；
- 根manifest逐文件大小与SHA-256不一致=0；
- 失效schema、缺字段说明CSV和压缩文件数均为0。

## 13. 回收完成判据

只有下列条件全部满足，才能把任一阶段或批次标记为回收完成：

- [ ] 容器源目录和文件身份已经确认；
- [ ] 容器源逐文件manifest已经生成；
- [ ] 容器到宿主机采用普通目录和文件逐项导出，没有压缩包；
- [ ] 宿主机文件数、相对路径、大小和SHA-256与容器一致；
- [ ] 宿主机到Windows采用普通目录和文件逐项回收，没有压缩包；
- [ ] Windows文件数、相对路径、大小和SHA-256与宿主机及容器一致；
- [ ] 文件已按第3、4节规定目录分流；
- [ ] 每个已创建子目录都有README；
- [ ] 每个README都能通过直接文件表和`folder_manifest.csv`追溯目录内全部文件；
- [ ] 每个目录`folder_manifest.csv`覆盖数等于实际递归证据文件数；
- [ ] README未覆盖文件数为0；
- [ ] README失效链接数为0；
- [ ] 每个直接或递归包含CSV的目录都有`CSV字段说明.md`；
- [ ] 每个CSV都在所在目录的逐文件映射表中绑定schema和锚点；
- [ ] 每个schema按真实列顺序说明全部列，而不是只说明部分指标列；
- [ ] CSV真实表头与字段说明的列名、列数和顺序完全一致；
- [ ] 各级`CSV字段说明.md`形成从根目录到叶子CSV的无断链追溯；
- [ ] 无字段说明CSV数为0，表头不一致数为0，失效schema引用数为0；
- [ ] 全归档压缩包扫描数量为0；
- [ ] artifact、SHA-256、recovery batch和missing清单均已更新；
- [ ] formal、smoke、历史和冲突结果边界明确；
- [ ] 原始证据、规范数据、绘图数据和引用数据没有就地混改；
- [ ] 临时目录已按规则处理，容器原件未被擅自删除；
- [ ] 根README已更新实际目录树和常用入口。

完成报告必须明确给出以下数字，不能只写“已完成”：

```text
source_file_count
host_file_count
windows_file_count
sha256_match_count
sha256_mismatch_count
archive_file_count
subdirectory_count
subdirectory_with_readme_count
subdirectory_missing_readme_count
folder_manifest_covered_file_count
folder_manifest_uncovered_file_count
readme_broken_reference_count
csv_file_count
csv_with_column_document_count
csv_without_column_document_count
csv_schema_count
csv_header_document_mismatch_count
csv_broken_schema_reference_count
```

合格值必须满足：

```text
source_file_count = host_file_count = windows_file_count
sha256_mismatch_count = 0
archive_file_count = 0
subdirectory_missing_readme_count = 0
folder_manifest_uncovered_file_count = 0
readme_broken_reference_count = 0
csv_file_count = csv_with_column_document_count
csv_without_column_document_count = 0
csv_header_document_mismatch_count = 0
csv_broken_schema_reference_count = 0
```

## 14. 现有归档的规则适用方式

本文件发布前已经回收的数据也必须逐步整改到本规则，不享有永久例外：

- 已存在的压缩包在确认内容已经完整展开、逐文件校验通过并获得用户授权后删除；
- 已存在但缺README的目录必须补建README和`folder_manifest.csv`；
- 已存在的README必须补齐直接文件表、子目录追溯表、递归文件数和完整manifest入口；
- 已存在CSV的目录必须补建`CSV字段说明.md`，逐文件绑定schema并说明全部真实列；
- 所有上级目录必须补建递归CSV说明索引，直到归档根目录；
- 历史CSV字段文档若只说明部分列或与真实表头不一致，仍视为未完成；
- 整改完成前，应在`missing_artifacts.csv`或专项审计表中登记为`NOT_COMPLIANT_WITH_CURRENT_ARCHIVE_RULES`；
- 不得因为历史数据量大就降低“每目录README、全文件可追溯、每CSV全部列有说明、零压缩包”的完成标准。
