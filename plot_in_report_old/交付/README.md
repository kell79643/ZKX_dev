# 复赛交付材料与证据索引

本目录是复赛答辩、技术结果和阶段证据的开发工作区，不属于最终源码上传白名单。正式
源码上传范围以 `docs/development/DELIVERY_MANIFEST.md` 为准；最终复验后仍需随源码
交付的证据，应迁入 `Task1/`、`Task2/`、`cusignal_cpp/` 或 `test_all/` 对应的
`evidence/` 目录，并同步更新引用。

## 推荐阅读顺序

1. 答辩准备先看 `01_答辩材料/`。
2. 核对正式结论看 `02_技术结果/`；绘制技术文档图表先看
   `02_技术结果/图表数据/技术文档图表数据总表.md`。
3. 核对晚于冻结批次的优化进度看 `03_当前状态/`。
4. 复核日志、指标、环境和源码指纹看 `04_原始证据/`。
5. `99_历史归档/` 只用于追溯，不得作为当前源码的正式结论。

## 目录职责

| 目录 | 用途 | 使用边界 |
| --- | --- | --- |
| `01_答辩材料/` | PPT 逐页讲稿和技术文档素材 | 属于表达材料，不替代正式证据 |
| `02_技术结果/` | Task1、Task2、TASK_F、TASK_G、高精度参考结果和图表数据入口 | 面向技术文档、PPT 和评委阅读 |
| `03_当前状态/` | 冻结证据批次之后继续推进的专项状态 | 必须与对应历史批次区分 |
| `04_原始证据/` | 当前有效或仍被正式文档引用的环境、日志、指标和归档 | 保留原批次名、commit 和哈希 |
| `99_历史归档/` | 已失效、被替代或仅用于增量追溯的证据 | 不得冒充当前正式证据 |

## 当前证据矩阵

| 对象 | 技术结果 | 当前原始证据 | 状态 |
| --- | --- | --- | --- |
| Task1 C++/GPU | `02_技术结果/TASK1_SEMIFINAL_TECHNICAL_RESULTS.md` | `04_原始证据/task1/task1_evidence_0372dcd/` | 冻结正式批次 |
| Task1 cuSignal Python | 同上 | `04_原始证据/task1/task1_python_evidence_cc85a35/` | 冻结正式批次 |
| Task2 C++/cuSignal Python | `02_技术结果/TASK2_SEMIFINAL_TECHNICAL_RESULTS.md` | `04_原始证据/task2/task2_python_evidence_fdf558f/` | 当前正式批次 |
| TASK_F 53×5 算子批次 | `02_技术结果/TASK_F_SEMIFINAL_TECHNICAL_RESULTS.md` | `04_原始证据/task_f/task_f_evidence_439fd41/` | 2026-07-15 冻结批次 |
| TASK_G CMake | `02_技术结果/TASK_G_ZQ500_CMAKE_TECHNICAL_RESULTS.md` | `04_原始证据/task_g/task_g_evidence_db7763c/` | 当前有效；旧批次已归档 |
| 高精度 CPU 参考 | `02_技术结果/OPERATOR_HIGH_PRECISION_ORACLE_RESULTS.md` | `04_原始证据/operators/operator_high_precision_final_20260717/` | 当前最终批次 |
| `firwin` P0 | `03_当前状态/FIRWIN_P0_CURRENT_STATUS.md` | 以状态文档链接的优化证据为准 | 晚于 TASK_F 冻结批次 |
| `unit_impulse` P0 | `03_当前状态/UNIT_IMPULSE_P0_CURRENT_STATUS.md` | 以状态文档链接的优化证据为准 | 晚于 TASK_F 冻结批次 |
| 技术文档图表 | `02_技术结果/图表数据/技术文档图表数据总表.md` | 规范化结果写入同目录 `正式数据/`，原始证据仍在 `04_原始证据/` | 19 张图、4 张表的唯一入口；采集实现与正式数值状态分开标注 |

## 历史归档

- `99_历史归档/operators/`：高精度参考最终批次之前的增量证据。
- `99_历史归档/task_g/task_g_evidence_0f8c4a4/`：README 已明确标记失效的
  TASK_G 诊断批次。
- `99_历史归档/待确认/task2_python_evidence_2bc8d8d/`：当前为空，仅保留目录身份，
  后续确认无外部材料依赖后再决定是否移除。

## 维护规则

- 新证据必须使用稳定的对象名并携带 commit 短哈希或日期，不使用 `new`、`latest2` 等名称。
- 新批次确认有效后，先更新本索引和技术结果，再把被替代批次迁入 `99_历史归档/`。
- 移动文件时必须用 `rg` 检查并同步更新仓库内全部路径引用。
- 原始日志、CSV、JSON、压缩包和哈希文件不做内容改写；目录整理只改变其上层归类。
