# CSV字段语义总表

> 本文件是全归档schema总入口。实际阅读某个CSV时仍应先读该CSV同目录的`CSV字段说明.md`，由逐文件映射定位本文件中的同一schema。

- 生成时间：`2026-08-14T17:33:09+08:00`
- 全局schema数量：46
- 全局CSV文件数量：11818
- 表头摘要：按真实列顺序使用U+001F连接列名后计算SHA-256。

## schema索引

| schema_id | 文件数 | 列数 | 样例文件 |
| --- | ---: | ---: | --- |
| [`SCHEMA_0031B1D01972`](#schema_0031b1d01972) | 47 | 17 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b03_channelize_poly_0b2008e7f76a/formal/20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000/filtering_batch03_module_summary_fft_thrust_20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000.csv` |
| [`SCHEMA_071B901F8846`](#schema_071b901f8846) | 1 | 23 | `04_算子结果/绘图数据/53算子_演示case性能候选.csv` |
| [`SCHEMA_0FE1A4609774`](#schema_0fe1a4609774) | 1 | 10 | `01_总清单与校验/sha256_manifest.csv` |
| [`SCHEMA_1576D69B43BF`](#schema_1576d69b43bf) | 1 | 10 | `01_总清单与校验/missing_artifacts.csv` |
| [`SCHEMA_25DBF2F4D43D`](#schema_25dbf2f4d43d) | 1 | 7 | `05_异常输入/stage09_operator_abnormal_159_617051d/abnormal_type_summary.csv` |
| [`SCHEMA_298670123E40`](#schema_298670123e40) | 211 | 12 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_csvread_windows_smoke_47ebe13410d3/smoke/20260810T040000Z_47ebe13410d3_stage08_b01_smoke_88000004/operator_dtype_evidence_general_cosine_not_applicable_20260810T040000Z_47ebe13410d3_stage08_b01_smoke_88000004.csv` |
| [`SCHEMA_2A341875B4AB`](#schema_2a341875b4ab) | 2 | 49 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/measurement_rows.csv` |
| [`SCHEMA_2F43D6693DE5`](#schema_2f43d6693de5) | 211 | 11 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_input_preview_b01_chebwin_8b745f9a6864/formal/20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001/operator_input_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001.csv` |
| [`SCHEMA_3AACC7DF0E1F`](#schema_3aacc7df0e1f) | 487 | 13 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b03_channelize_poly_0b2008e7f76a/formal/20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000/operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000.csv` |
| [`SCHEMA_3ACCC959CC57`](#schema_3accc959cc57) | 1 | 13 | `04_算子结果/绘图数据/12模块_性能汇总候选.csv` |
| [`SCHEMA_3B8D62FCD6D0`](#schema_3b8d62fcd6d0) | 1 | 2 | `04_算子结果/绘图数据/53算子_12模块映射.csv` |
| [`SCHEMA_3C73F9C8B2B5`](#schema_3c73f9c8b2b5) | 1 | 5 | `04_算子结果/CSV字段模式汇总.csv` |
| [`SCHEMA_43F03932A81C`](#schema_43f03932a81c) | 15 | 16 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_input_preview_b04_firwin2_db927ed96663/formal/20260810T083702Z_db927ed96663_stage08_b04_formal_b4020001/operator_dtype_evidence_firwin2_fft_thrust_20260810T083702Z_db927ed96663_stage08_b04_formal_b4020001.csv` |
| [`SCHEMA_4AD8B3491740`](#schema_4ad8b3491740) | 1 | 12 | `05_异常输入/stage09_operator_abnormal_159_617051d/operator_abnormal_summary.csv` |
| [`SCHEMA_4B1EA6CE9919`](#schema_4b1ea6ce9919) | 16 | 16 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_input_preview_b04_correlate2d_db927ed96663/formal/20260810T083651Z_db927ed96663_stage08_b04_formal_b4010001/operator_dtype_evidence_correlate2d_not_applicable_20260810T083651Z_db927ed96663_stage08_b04_formal_b4010001.csv` |
| [`SCHEMA_52DC1450C17A`](#schema_52dc1450c17a) | 92 | 11 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b01_5e66669d177a/formal/20260810T024215Z_5e66669d177a_stage07_b01_formal_c65ae676/operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024215Z_5e66669d177a_stage07_b01_formal_c65ae676.csv` |
| [`SCHEMA_53562F8B9951`](#schema_53562f8b9951) | 1 | 21 | `04_算子结果/绘图数据/53x5_最差精度候选.csv` |
| [`SCHEMA_5629860C33F9`](#schema_5629860c33f9) | 1 | 31 | `01_总清单与校验/artifact_manifest.csv` |
| [`SCHEMA_5DBF65A8D3C7`](#schema_5dbf65a8d3c7) | 31 | 16 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b03_formal_20e76c9b0ef/formal/20260810T100929Z_20e76c9b0ef0_stage07_b03_formal_711701b6/operator_dtype_evidence_cwt_not_applicable_20260810T100929Z_20e76c9b0ef0_stage07_b03_formal_711701b6.csv` |
| [`SCHEMA_5E94889DE383`](#schema_5e94889de383) | 1 | 15 | `04_算子结果/CSV可打开性审计.csv` |
| [`SCHEMA_60A0F3BAE71C`](#schema_60a0f3bae71c) | 16 | 14 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b04_correlate2d_2c86271438dd/smoke/20260809T110300Z_2c86271438dd_stage08_b04_smoke_84000001/operator_dtype_evidence_correlate2d_not_applicable_20260809T110300Z_2c86271438dd_stage08_b04_smoke_84000001.csv` |
| [`SCHEMA_620652473711`](#schema_620652473711) | 2084 | 30 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000/operator_windows_chebwin_memory_trace_not_applicable.csv` |
| [`SCHEMA_657BBB94EDDF`](#schema_657bbb94eddf) | 1 | 7 | `04_算子结果/绘图数据/4类代表算子候选.csv` |
| [`SCHEMA_715A16F6F7F0`](#schema_715a16f6f7f0) | 64 | 16 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b04_firwin_batchfix_smoke_d5013d1/smoke/20260810T102454Z_d5013d1285d1_stage07_b04_smoke_2d8e7d36/operator_dtype_evidence_firwin_not_applicable_20260810T102454Z_d5013d1285d1_stage07_b04_smoke_2d8e7d36.csv` |
| [`SCHEMA_7599DF084A43`](#schema_7599df084a43) | 211 | 14 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_input_preview_b01_chebwin_8b745f9a6864/formal/20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001/operator_dtype_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001.csv` |
| [`SCHEMA_7ED5A58734DD`](#schema_7ed5a58734dd) | 1 | 60 | `04_算子结果/绘图数据/全算子_逐case绘图候选.csv` |
| [`SCHEMA_7F4D2CBB2001`](#schema_7f4d2cbb2001) | 2625 | 32 | `folder_manifest.csv` |
| [`SCHEMA_7FA1B27BF151`](#schema_7fa1b27bf151) | 2 | 7 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/file_catalog.csv` |
| [`SCHEMA_8C1017E6E31D`](#schema_8c1017e6e31d) | 1 | 17 | `01_总清单与校验/recovery_batches.csv` |
| [`SCHEMA_8C46C287C375`](#schema_8c46c287c375) | 2084 | 31 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000/operator_timing_samples_chebwin_not_applicable_20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000.csv` |
| [`SCHEMA_917242DCFB95`](#schema_917242dcfb95) | 62 | 14 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b02_formal_76c5b6f8b36d/formal/20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2/operator_dtype_evidence_chirp_not_applicable_20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2.csv` |
| [`SCHEMA_9C155BB8C8B9`](#schema_9c155bb8c8b9) | 1 | 4 | `05_异常输入/stage09_operator_abnormal_159_617051d/file_catalog.csv` |
| [`SCHEMA_A7B6403ACEEB`](#schema_a7b6403aceeb) | 1 | 25 | `04_算子结果/绘图数据/窗函数_435case热力图候选.csv` |
| [`SCHEMA_AC3D0A8CA003`](#schema_ac3d0a8ca003) | 46 | 15 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b05_formal_2836c8d/formal/20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8/operator_dtype_evidence_fm_demod_not_applicable_20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8.csv` |
| [`SCHEMA_B8CA1C80348B`](#schema_b8ca1c80348b) | 2 | 56 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/cpu_gpu_comparison.csv` |
| [`SCHEMA_BF0AE92FC547`](#schema_bf0ae92fc547) | 58 | 24 | `05_异常输入/stage09_operator_abnormal_159_617051d/all_operator_abnormal_cases.csv` |
| [`SCHEMA_C09B6D1032ED`](#schema_c09b6d1032ed) | 92 | 10 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b02_strict_dtype/formal/20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000100/operator_dtype_evidence_parzen_not_applicable_20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000100.csv` |
| [`SCHEMA_C185CB915CF7`](#schema_c185cb915cf7) | 2084 | 66 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000/operator_main_results_chebwin_not_applicable_20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000.csv` |
| [`SCHEMA_C5A345C78C5D`](#schema_c5a345c78c5d) | 16 | 14 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b04_firwin2_6e5af3b6102d/smoke/20260809T111800Z_6e5af3b6102d_stage08_b04_smoke_84100001/operator_dtype_evidence_firwin2_fft_thrust_20260809T111800Z_6e5af3b6102d_stage08_b04_smoke_84100001.csv` |
| [`SCHEMA_C6D6D1C6C971`](#schema_c6d6d1c6c971) | 2 | 30 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/input_evidence_index.csv` |
| [`SCHEMA_C74B9B1781E2`](#schema_c74b9b1781e2) | 92 | 13 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b01_input_preview_1662dd06774d/formal/20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18/operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18.csv` |
| [`SCHEMA_C9856F6503C1`](#schema_c9856f6503c1) | 46 | 16 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/runs/stage07_b06_formal_6132f71/formal/20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996/operator_dtype_evidence_cubic_not_applicable_20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996.csv` |
| [`SCHEMA_CB6B6E50B568`](#schema_cb6b6e50b568) | 736 | 20 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000/windows_batch01_module_summary_20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000.csv` |
| [`SCHEMA_D383C06B6201`](#schema_d383c06b6201) | 1 | 30 | `04_算子结果/绘图数据/53算子_全case性能区间候选.csv` |
| [`SCHEMA_E7EC8ACA9A36`](#schema_e7ec8aca9a36) | 2 | 15 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/operator_coverage.csv` |
| [`SCHEMA_F790A782CFD0`](#schema_f790a782cfd0) | 365 | 15 | `04_算子结果/remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_input_preview_b03_channelize_poly_5bf09d215ce4/formal/20260810T073905Z_5bf09d215ce4_stage08_b03_formal_99010001/operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073905Z_5bf09d215ce4_stage08_b03_formal_99010001.csv` |

## schema完整定义

<a id="schema_0031b1d01972"></a>
### SCHEMA_0031B1D01972

- 适用文件：`filtering_batch03_module_summary_fft_thrust_20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f001.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091034Z_0b2008e7f76a_stage08_b03_formal_8303f002.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091043Z_0b2008e7f76a_stage08_b03_formal_8303f003.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091043Z_0b2008e7f76a_stage08_b03_formal_8303f004.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091044Z_0b2008e7f76a_stage08_b03_formal_8303f005.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091111Z_0b2008e7f76a_stage08_b03_formal_8303f006.csv`、`filtering_batch03_module_summary_fft_thrust_20260809T091111Z_0b2008e7f76a_stage08_b03_formal_8303f007.csv`等47个文件
- 一行代表：一行对应一个汇总对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly__channelize_poly_fp32_10e2__b6e2f353c8853cdd` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly` | NA |
| 4 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly_fp32_10e2` | NA |
| 5 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 7 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 8 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 9 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.088647360000000008` | warmup_runs, measured_runs, timing_scope |
| 10 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.39801488000000002` | warmup_runs, measured_runs, timing_scope |
| 11 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.22272373334383883` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 12 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `7.5317356518223022e-15` | accuracy_reference, accuracy_status |
| 13 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `8.6785572832253072e-08` | accuracy_reference, accuracy_status |
| 14 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.1076189520663169e-08` | accuracy_reference, accuracy_status |
| 15 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.1017899634618443e-08` | accuracy_reference, accuracy_status |
| 16 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 17 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_071b901f8846"></a>
### SCHEMA_071B901F8846

- 适用文件：`53算子_演示case性能候选.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 3 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 4 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 5 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic_fp16_10e4` | NA |
| 6 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10000` | NA |
| 7 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[10000]` | NA |
| 8 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.07658552` | warmup_runs, measured_runs, timing_scope |
| 9 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.21139722` | warmup_runs, measured_runs, timing_scope |
| 10 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.3622825314353708` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 11 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.217279` | warmup_runs, measured_runs, timing_scope |
| 12 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.08293981931311126` | NA |
| 13 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 14 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 15 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 16 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 17 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 18 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 19 | `resource_gate_status` | string | NA | 样本未见空值 | `resource gate`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NOT_EVALUATED_IN_COMPARISON_INDEX` | NA |
| 20 | `leak_gate_status` | string | NA | 样本未见空值 | `leak gate`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NOT_EVALUATED_IN_COMPARISON_INDEX` | NA |
| 21 | `demo_status` | string | NA | 样本未见空值 | `demo`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PERFORMANCE_SHORTLIST_NOT_BEST_DEMO_CASE` | NA |
| 22 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110446Z_6132f71eb1e0_stage07_b06_formal_c6bea09d` | NA |
| 23 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage07_b06_formal_6132f71/formal/20260810T110446Z_6132f71eb1e0_stage07_...` | NA |

<a id="schema_0fe1a4609774"></a>
### SCHEMA_0FE1A4609774

- 适用文件：`sha256_manifest.csv`
- 一行代表：一行对应一个文件或证据条目。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`artifact_id`
- 配对或关联键：`artifact_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `artifact_id` | string | NA | 样本未见空值 | 归档证据的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ART_RECOVERY_STAGE07_08_MANIFEST` | NA |
| 2 | `archive_relative_path` | string | NA | 样本未见空值 | 相对于全国总决赛证据归档根的目标路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `01_总清单与校验/recovery_batches/RB_20260810T1446CST_STAGE07_08/` | NA |
| 3 | `filename` | string | NA | 样本未见空值 | 文件名。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `zkx_stage07_08_20260810T1446CST_host_files.sha256` | NA |
| 4 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `4139686` | NA |
| 5 | `sha256` | string | NA | 样本未见空值 | 文件内容SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0` | NA |
| 6 | `hash_algorithm` | string | NA | 样本未见空值 | 摘要算法名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `SHA-256` | NA |
| 7 | `source_sha256` | string | NA | 允许 | 来源文件或来源清单记录的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0` | NA |
| 8 | `verification_status` | string | NA | 样本未见空值 | 文件、表头、清单或回收校验状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CONTAINER_HOST_WINDOWS_MATCH` | NA |
| 9 | `verified_timestamp` | datetime | datetime | 样本未见空值 | 完成校验的时间。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2026-08-11T18:22:39+08:00` | NA |
| 10 | `notes` | string | NA | 样本未见空值 | 补充说明、限制、NA规则或审计备注。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `17301文件清单` | NA |

<a id="schema_1576d69b43bf"></a>
### SCHEMA_1576D69B43BF

- 适用文件：`missing_artifacts.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`missing_id`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `missing_id` | string | NA | 样本未见空值 | 缺失项或整改项唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 2 | `test_object` | string | NA | 样本未见空值 | 测试对象的统一名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 3 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 4 | `expected_source_path` | string | NA | 样本未见空值 | 预期证据来源路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 5 | `planned_archive_path` | string | NA | 样本未见空值 | 计划归档目标路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 6 | `current_status` | string | NA | 样本未见空值 | 缺失项当前处理状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NA` | NA |
| 7 | `priority` | string | NA | 样本未见空值 | 缺失项处理优先级。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 8 | `impact` | string | NA | 样本未见空值 | 缺失或不合规对结论、图表或演示的影响。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 9 | `completion_criterion` | string | NA | 样本未见空值 | 关闭缺失项必须满足的条件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 10 | `notes` | string | NA | 样本未见空值 | 补充说明、限制、NA规则或审计备注。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_25dbf2f4d43d"></a>
### SCHEMA_25DBF2F4D43D

- 适用文件：`abnormal_type_summary.csv`
- 一行代表：一行对应一个异常输入case及其预期和实际处理结果。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `abnormal_type` | string | NA | 样本未见空值 | 单条异常输入的语义类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `backend_call_failure` | NA |
| 2 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |
| 3 | `expected_codes` | string | NA | 样本未见空值 | `expected codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `BACKEND_ERROR\|CUDA_RUNTIME_ERROR` | NA |
| 4 | `actual_codes` | string | NA | 样本未见空值 | `actual codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `BACKEND_ERROR\|CUDA_RUNTIME_ERROR` | NA |
| 5 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |
| 6 | `safe_exit_count` | integer | count | 样本未见空值 | `safe exit`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |
| 7 | `output_not_consumed_count` | integer | count | 样本未见空值 | `output not consumed`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `44` | NA |

<a id="schema_298670123e40"></a>
### SCHEMA_298670123E40

- 适用文件：`operator_dtype_evidence_general_cosine_not_applicable_20260810T040000Z_47ebe13410d3_stage08_b01_smoke_88000004.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031835Z_47ebe13410d3_stage08_b01_formal_88100000.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031835Z_47ebe13410d3_stage08_b01_formal_88100001.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031835Z_47ebe13410d3_stage08_b01_formal_88100002.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031836Z_47ebe13410d3_stage08_b01_formal_88100003.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031836Z_47ebe13410d3_stage08_b01_formal_88100004.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031843Z_47ebe13410d3_stage08_b01_formal_88100005.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T031849Z_47ebe13410d3_stage08_b01_formal_88100006.csv`等211个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T040000Z_47ebe13410d3_stage08_b01_smoke_88000004` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `general_cosine__general_cosine_fp32_10e2__7aea9ce1217a0525__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `general_cosine` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_typed_array_input` | NA |
| 7 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 8 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `length:INT32=100\|sym:BOOL=true\|coefficient_values:FP32:bytes=0000803f;0000004...` | NA |
| 9 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `7aea9ce1217a052519180c6753897bccd990fec0c5b4981f1dfe2709bb981540` | NA |
| 10 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP64` | NA |
| 11 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 12 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_2a341875b4ab"></a>
### SCHEMA_2A341875B4AB

- 适用文件：`measurement_rows.csv`、`measurement_rows.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_7200...` | NA |
| 5 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 6 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 7 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 8 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 9 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 10 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 26 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 27 | `mean_ms` | float | ms | 样本未见空值 | 耗时算术平均值，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7422645300000004` | warmup_runs, measured_runs, timing_scope |
| 28 | `p50_ms` | float | ms | 样本未见空值 | 耗时第50百分位，即中位数，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7423929999999999` | warmup_runs, measured_runs, timing_scope |
| 29 | `p95_ms` | float | ms | 样本未见空值 | 耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7533555000000001` | warmup_runs, measured_runs, timing_scope |
| 30 | `p99_ms` | float | ms | 样本未见空值 | 耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7555925000000001` | warmup_runs, measured_runs, timing_scope |
| 31 | `min_ms` | float | ms | 样本未见空值 | 最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7322329999999999` | warmup_runs, measured_runs, timing_scope |
| 32 | `max_ms` | float | ms | 样本未见空值 | 最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7585130000000002` | warmup_runs, measured_runs, timing_scope |
| 33 | `std_ms` | float | ms | 样本未见空值 | 耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0063154560871800988` | warmup_runs, measured_runs, timing_scope |
| 34 | `cv` | float | dimensionless | 样本未见空值 | 耗时变异系数，计算为std_ms/mean_ms，无量纲。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0023030076121723013` | NA |
| 35 | `cpu_gpu_speedup` | float | dimensionless | 允许 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `1.9409993884459864` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 36 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_typed_reference` | NA |
| 37 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 38 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 39 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 40 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 41 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 42 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 43 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `finite_fp64_window_shape_and_symmetry_policy_match_cpu:shape=[100];finite=tru...` | NA |
| 44 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 45 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 46 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 47 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 48 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 49 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_2f43d6693de5"></a>
### SCHEMA_2F43D6693DE5

- 适用文件：`operator_input_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010002.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010003.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082001Z_8b745f9a6864_stage08_b01_formal_a1010004.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082001Z_8b745f9a6864_stage08_b01_formal_a1010005.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082008Z_8b745f9a6864_stage08_b01_formal_a1010006.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082014Z_8b745f9a6864_stage08_b01_formal_a1010007.csv`、`operator_input_evidence_chebwin_fft_thrust_20260810T082014Z_8b745f9a6864_stage08_b01_formal_a1010008.csv`等211个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__37f607f8470837ce__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 4 | `input_name` | string | NA | 样本未见空值 | 一个输入参数或输入数组的名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions` | NA |
| 5 | `shape` | string | NA | 样本未见空值 | 输入或输出数组的形状。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[100]` | NA |
| 6 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `INDEX_INT64` | NA |
| 7 | `element_count` | integer | count | 样本未见空值 | 数组中的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 8 | `generator` | string | NA | 样本未见空值 | 输入数据生成器或生成规则。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `implicit_window_index_domain(no_materialized_array)` | NA |
| 9 | `preview_head4_tail2` | string | NA | 样本未见空值 | 大型输入前4项与后2项的可读预览。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `i=0:position=0;i=1:position=1;i=2:position=2;i=3:position=3;i=98:position=98;...` | NA |
| 10 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 11 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_3aacc7df0e1f"></a>
### SCHEMA_3AACC7DF0E1F

- 适用文件：`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f001.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091034Z_0b2008e7f76a_stage08_b03_formal_8303f002.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091043Z_0b2008e7f76a_stage08_b03_formal_8303f003.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091043Z_0b2008e7f76a_stage08_b03_formal_8303f004.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091044Z_0b2008e7f76a_stage08_b03_formal_8303f005.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091111Z_0b2008e7f76a_stage08_b03_formal_8303f006.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260809T091111Z_0b2008e7f76a_stage08_b03_formal_8303f007.csv`等487个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260809T091033Z_0b2008e7f76a_stage08_b03_formal_8303f000` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly__channelize_poly_fp32_10e2__b6e2f353c8853cdd` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_typed_array_inputs_fp32_compute_extension` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x:FP32:[128]:bytes=512:sha256=a77bf42e2c86562f8cfd734074f488b4350e3e3126ca245...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `b6e2f353c8853cddcf85e3c02d052280d158379ab9b9c415f42db9ed5e61d864` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 12 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 13 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_3accc959cc57"></a>
### SCHEMA_3ACCC959CC57

- 适用文件：`12模块_性能汇总候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 2 | `operator_count` | integer | count | 样本未见空值 | `operator`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 3 | `total_cases` | integer | count | 样本未见空值 | 当前汇总范围内全部case数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `45` | NA |
| 4 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `45` | NA |
| 5 | `pass_rate` | integer | dimensionless | 样本未见空值 | `pass`占总数的比例。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 6 | `median_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的中位数。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.122668908006515` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 7 | `p25_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的第25百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.0647819756642247` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 8 | `p75_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的第75百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.149714532686372` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 9 | `min_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的最小值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.00689504332193441` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 10 | `max_operator_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`operator speedup`的最大值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.176760157366229` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 11 | `operators_median_below_1x` | integer | NA | 样本未见空值 | 算子中位加速比小于1的算子数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 12 | `worst_accuracy_ratio` | string | dimensionless | 样本未见空值 | 最差精度值相对其适用阈值的比值；阈值未绑定时必须为明确NA状态，不能臆造。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_THRESHOLD_SNAPSHOT_NOT_BOUND` | NA |
| 13 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_3b8d62fcd6d0"></a>
### SCHEMA_3B8D62FCD6D0

- 适用文件：`53算子_12模块映射.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |

<a id="schema_3c73f9c8b2b5"></a>
### SCHEMA_3C73F9C8B2B5

- 适用文件：`CSV字段模式汇总.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `file_count` | integer | count | 样本未见空值 | `file`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2084` | NA |
| 2 | `header_columns` | integer | count | 样本未见空值 | CSV表头列数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `66` | NA |
| 3 | `parse_statuses` | string | NA | 样本未见空值 | 同一schema样本的解析状态集合。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `PASS` | NA |
| 4 | `header` | string | NA | 样本未见空值 | CSV真实表头的逗号分隔文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `schema_version,run_id,run_type,test_time_utc,git_commit,git_dirty,config_id,c...` | NA |
| 5 | `sample_relative_path` | string | NA | 样本未见空值 | 该字段模式的样例CSV相对路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `remaining_operators/data/RB_20260810T1446CST_STAGE08/runs/stage08_b01/formal/...` | NA |

<a id="schema_43f03932a81c"></a>
### SCHEMA_43F03932A81C

- 适用文件：`operator_dtype_evidence_firwin2_fft_thrust_20260810T083702Z_db927ed96663_stage08_b04_formal_b4020001.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083703Z_db927ed96663_stage08_b04_formal_b4020002.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083703Z_db927ed96663_stage08_b04_formal_b4020003.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083704Z_db927ed96663_stage08_b04_formal_b4020004.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083704Z_db927ed96663_stage08_b04_formal_b4020005.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083704Z_db927ed96663_stage08_b04_formal_b4020006.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083705Z_db927ed96663_stage08_b04_formal_b4020007.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260810T083705Z_db927ed96663_stage08_b04_formal_b4020008.csv`等15个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T083702Z_db927ed96663_stage08_b04_formal_b4020001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `firwin2__firwin2_fp32_10e2__a26a252576b841ba` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `firwin2` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_same_dtype_freq_gain_arrays` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `freq:FP32:[3]:bytes=12:sha256=ec59471fa91d47300ee1ceb1c4d6a5baa520636059cff91...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `a26a252576b841baf1b94619cf5f52ed4bc780e063cb61ea0a43dba89c157c78` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP64` | NA |
| 12 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 13 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 14 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 15 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"3","generator":"fixed_frequency_breakpoints...` | NA |
| 16 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"antisymmetric":"false","dtype":"FP32","fs":"16","nfreqs":"33","numtaps":"17...` | NA |

<a id="schema_4ad8b3491740"></a>
### SCHEMA_4AD8B3491740

- 适用文件：`operator_abnormal_summary.csv`
- 一行代表：一行对应一个异常输入case及其预期和实际处理结果。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `module_name` | string | NA | 样本未见空值 | 算子所属模块名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 2 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 3 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 4 | `abnormal_types` | string | NA | 样本未见空值 | 一组记录覆盖的异常语义类别集合。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `backend_call_failure\|invalid_dtype\|invalid_shape` | NA |
| 5 | `backends` | string | NA | 样本未见空值 | 当前汇总对象覆盖的后端集合，通常使用竖线分隔。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cuda_runtime\|not_applicable` | NA |
| 6 | `expected_codes` | string | NA | 样本未见空值 | `expected codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR\|INVALID_DTYPE\|INVALID_SHAPE` | NA |
| 7 | `actual_codes` | string | NA | 样本未见空值 | `actual codes`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR\|INVALID_DTYPE\|INVALID_SHAPE` | NA |
| 8 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 9 | `safe_exit_count` | integer | count | 样本未见空值 | `safe exit`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 10 | `output_not_consumed_count` | integer | count | 样本未见空值 | `output not consumed`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 11 | `normal_run_ids` | datetime | NA | 样本未见空值 | `normal run ids`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996` | NA |
| 12 | `normal_case_ids` | string | NA | 样本未见空值 | `normal case ids`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |

<a id="schema_4b1ea6ce9919"></a>
### SCHEMA_4B1EA6CE9919

- 适用文件：`operator_dtype_evidence_correlate2d_not_applicable_20260810T083651Z_db927ed96663_stage08_b04_formal_b4010001.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083651Z_db927ed96663_stage08_b04_formal_b4010002.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083652Z_db927ed96663_stage08_b04_formal_b4010003.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083653Z_db927ed96663_stage08_b04_formal_b4010004.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083653Z_db927ed96663_stage08_b04_formal_b4010005.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083653Z_db927ed96663_stage08_b04_formal_b4010006.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083656Z_db927ed96663_stage08_b04_formal_b4010007.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260810T083656Z_db927ed96663_stage08_b04_formal_b4010008.csv`等16个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T083651Z_db927ed96663_stage08_b04_formal_b4010001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `correlate2d__correlate2d_fp32_10e2__11d622e248c4eb6f` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `correlate2d` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_same_dtype_arrays_native_direct` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `in1:FP32:[10x10]:bytes=400:sha256=69e4595433fdcbb3baf657e00e7ab3a034d6144f697...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `11d622e248c4eb6feb2d50c956b6ad821146c38d461e2f6ccc615e349c801118` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 12 | `native_capability` | string | NA | 样本未见空值 | 平台或后端原生能力说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `native` | NA |
| 13 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 14 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 15 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"100","generator":"correlate_modular(seed=84...` | NA |
| 16 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"boundary":"fill","cols1":"10","cols2":"3","dtype":"FP32","fillvalue":"1","m...` | NA |

<a id="schema_52dc1450c17a"></a>
### SCHEMA_52DC1450C17A

- 适用文件：`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024215Z_5e66669d177a_stage07_b01_formal_c65ae676.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024216Z_5e66669d177a_stage07_b01_formal_3384702f.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024216Z_5e66669d177a_stage07_b01_formal_61e75083.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024221Z_5e66669d177a_stage07_b01_formal_0c44de32.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024221Z_5e66669d177a_stage07_b01_formal_6280f963.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024222Z_5e66669d177a_stage07_b01_formal_6523dfd7.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024234Z_5e66669d177a_stage07_b01_formal_0a272c78.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T024234Z_5e66669d177a_stage07_b01_formal_631b8a06.csv`等92个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T024215Z_5e66669d177a_stage07_b01_formal_c65ae676` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `pulse_compression__pc_fp32_10e2__d7f9111bf9056c73` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `pulse_compression` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 6 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `echo:FP32:bytes=2048:sha256=c9f90ed4f45dd0c7f5c20f18ae23d6190e7ca08d44774775c...` | NA |
| 7 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `d7f9111bf9056c7317f2c9ca923cbcfe21f245291dcbc35d070d49bbc56ccbad` | NA |
| 8 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 9 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 10 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 11 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_53562f8b9951"></a>
### SCHEMA_53562F8B9951

- 适用文件：`53x5_最差精度候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 3 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07` | NA |
| 4 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 5 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 6 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 7 | `worst_accuracy_ratio` | string | dimensionless | 样本未见空值 | 最差精度值相对其适用阈值的比值；阈值未绑定时必须为明确NA状态，不能臆造。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_THRESHOLD_SNAPSHOT_NOT_BOUND` | NA |
| 8 | `max_mse` | float | NA | 样本未见空值 | 当前聚合范围内`mse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 9 | `max_mse_case_id` | string | NA | 样本未见空值 | 产生`max_mse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 10 | `max_rmse` | float | NA | 样本未见空值 | 当前聚合范围内`rmse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 11 | `max_rmse_case_id` | string | NA | 样本未见空值 | 产生`max_rmse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 12 | `max_relative_l2` | float | NA | 样本未见空值 | 当前聚合范围内`relative l2`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 13 | `max_relative_l2_case_id` | string | NA | 样本未见空值 | 产生`max_relative_l2`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 14 | `max_relative_linf` | float | NA | 样本未见空值 | 当前聚合范围内`relative linf`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | NA |
| 15 | `max_relative_linf_case_id` | string | NA | 样本未见空值 | 产生`max_relative_linf`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 16 | `exact_match_false_count` | integer | count | 样本未见空值 | `exact match false`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 17 | `max_mismatch_count` | string | count | 允许 | 当前聚合范围内`mismatch count`的最大值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 18 | `max_mismatch_case_id` | string | NA | 允许 | 产生`max_mismatch`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 19 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 20 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 21 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_5629860c33f9"></a>
### SCHEMA_5629860C33F9

- 适用文件：`artifact_manifest.csv`
- 一行代表：一行对应一个文件或证据条目。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`artifact_id`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `artifact_id` | string | NA | 样本未见空值 | 归档证据的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ART_RECOVERY_STAGE07_08_MANIFEST` | NA |
| 2 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `checksum_manifest` | NA |
| 3 | `test_object` | string | NA | 样本未见空值 | 测试对象的统一名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07_stage08` | NA |
| 4 | `task` | string | NA | 允许 | 任务名称或任务标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 5 | `step` | string | NA | 允许 | 任务步骤标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 6 | `module` | string | NA | 允许 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 7 | `operator` | string | NA | 允许 | 算子标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 8 | `run_id` | string | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 9 | `case_id` | string | NA | 允许 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 10 | `backend` | string | NA | 允许 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 11 | `dtype` | string | NA | 允许 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 12 | `scale_id` | string | NA | 允许 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 13 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `recovery` | NA |
| 14 | `git_commit` | string | NA | 允许 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 15 | `git_dirty` | string | NA | 允许 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 16 | `source_host` | string | NA | 样本未见空值 | 证据来源主机。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `usr_02@10.110.12.10` | NA |
| 17 | `source_container` | string | NA | 样本未见空值 | 证据来源容器。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `gpu_02` | NA |
| 18 | `source_path` | string | NA | 样本未见空值 | 证据在来源环境中的路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `/tmp/zkx_stage07_08_20260810T1446CST_files.sha256` | NA |
| 19 | `archive_relative_path` | string | NA | 样本未见空值 | 相对于全国总决赛证据归档根的目标路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `01_总清单与校验/recovery_batches/RB_20260810T1446CST_STAGE07_08/` | NA |
| 20 | `filename` | string | NA | 样本未见空值 | 文件名。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `zkx_stage07_08_20260810T1446CST_host_files.sha256` | NA |
| 21 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `4139686` | NA |
| 22 | `sha256` | string | NA | 样本未见空值 | 文件内容SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `9eda3a4054c4af9ea000effef9b77d041274b8b126cd52e7695f54e891157fd0` | NA |
| 23 | `generated_timestamp` | datetime | datetime | 样本未见空值 | 文件生成时间。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2026-08-10T14:46:00+08:00` | NA |
| 24 | `recovered_timestamp` | datetime | datetime | 样本未见空值 | 文件回收到本地归档的时间。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2026-08-11T18:22:39+08:00` | NA |
| 25 | `evidence_status` | string | NA | 样本未见空值 | 证据资格或归档状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `RECOVERED_VERIFIED` | NA |
| 26 | `formal_eligible` | boolean | NA | 样本未见空值 | 该证据是否满足正式使用资格。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 27 | `related_csv` | string | NA | 允许 | 与当前证据关联的CSV路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 28 | `related_log` | string | NA | 允许 | 与当前证据关联的完整日志路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 29 | `related_screenshot` | string | NA | 允许 | 与当前证据关联的截图路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 30 | `related_score_id` | string | NA | 允许 | 与当前证据关联的评分点ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 31 | `notes` | string | NA | 样本未见空值 | 补充说明、限制、NA规则或审计备注。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `联合包17301文件原始校验清单` | NA |

<a id="schema_5dbf65a8d3c7"></a>
### SCHEMA_5DBF65A8D3C7

- 适用文件：`operator_dtype_evidence_cwt_not_applicable_20260810T100929Z_20e76c9b0ef0_stage07_b03_formal_711701b6.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100929Z_20e76c9b0ef0_stage07_b03_formal_fd07f4d1.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100930Z_20e76c9b0ef0_stage07_b03_formal_4e4ad6e8.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100934Z_20e76c9b0ef0_stage07_b03_formal_c0698a93.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100935Z_20e76c9b0ef0_stage07_b03_formal_1d9f1c9c.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100936Z_20e76c9b0ef0_stage07_b03_formal_a4169d2c.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100949Z_20e76c9b0ef0_stage07_b03_formal_958598c9.csv`、`operator_dtype_evidence_cwt_not_applicable_20260810T100949Z_20e76c9b0ef0_stage07_b03_formal_d763bb18.csv`等31个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T100929Z_20e76c9b0ef0_stage07_b03_formal_711701b6` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cwt__cwt_task_fp32_10e3__b1fe60496c6f9086` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cwt` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `signal:FP32:bytes=4096:sha256=d7682c0511fb9eea278eb168948149d78f76a741cad9784...` | NA |
| 7 | `typed_array_digest` | string | NA | 样本未见空值 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `d7682c0511fb9eea278eb168948149d78f76a741cad97841d2dacac3340e201a` | NA |
| 8 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"1024","generator":"sinusoid_mixture_plus_tr...` | NA |
| 9 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"backend_truth":"not_applicable_direct_custom_convolution","expected_output_...` | NA |
| 10 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `b1fe60496c6f90864463d3bb73892e7c06912a234cdccd8b5baa1cbfc02f6152` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP64` | NA |
| 12 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 13 | `task_gpu_call_status` | string | NA | 样本未见空值 | 任务GPU调用闭包检查状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `proved_gpu_cpu` | NA |
| 14 | `fft_metadata_note` | string | NA | 样本未见空值 | FFT调用、补零或后端元数据说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `task_call_chain_uses_fft_true_conflicts_with_direct_non_fft_implementation` | NA |
| 15 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 16 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_5e94889de383"></a>
### SCHEMA_5E94889DE383

- 适用文件：`CSV可打开性审计.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`relative_path`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 样本未见空值 | 相对于当前目录或表格约定根目录的文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `remaining_operators/data/RB_20260810T1446CST_STAGE08/cpu_gpu_comparison.csv` | NA |
| 2 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1963493` | NA |
| 3 | `path_length` | integer | count | 样本未见空值 | 完整路径字符数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `121` | NA |
| 4 | `over_218_chars` | boolean | NA | 样本未见空值 | 完整路径是否超过218字符的布尔标记。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `False` | NA |
| 5 | `over_260_chars` | boolean | NA | 样本未见空值 | 完整路径是否超过260字符的布尔标记。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `False` | NA |
| 6 | `utf8_status` | string | NA | 样本未见空值 | UTF-8编码校验状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 7 | `parse_status` | string | NA | 样本未见空值 | CSV语法解析状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 8 | `header_columns` | integer | count | 样本未见空值 | CSV表头列数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `56` | NA |
| 9 | `data_rows` | integer | count | 样本未见空值 | CSV数据行数量，不含表头。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1651` | NA |
| 10 | `min_row_fields` | integer | NA | 样本未见空值 | 数据行中最少字段数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `56` | NA |
| 11 | `max_row_fields` | integer | NA | 样本未见空值 | 数据行中最多字段数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `56` | NA |
| 12 | `blank_header_count` | integer | count | 样本未见空值 | 空表头列数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 13 | `duplicate_header_name_count` | integer | count | 样本未见空值 | 重复表头名称数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 14 | `header` | string | NA | 样本未见空值 | CSV真实表头的逗号分隔文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage,source_group,source_partition,source_relative_path,run_id,run_type,test...` | NA |
| 15 | `error` | string | NA | 允许 | CSV审计或处理过程中捕获的错误文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_60a0f3bae71c"></a>
### SCHEMA_60A0F3BAE71C

- 适用文件：`operator_dtype_evidence_correlate2d_not_applicable_20260809T110300Z_2c86271438dd_stage08_b04_smoke_84000001.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110640Z_e458c69972d3_stage08_b04_formal_8403f000.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110641Z_e458c69972d3_stage08_b04_formal_8403f001.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110641Z_e458c69972d3_stage08_b04_formal_8403f002.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110642Z_e458c69972d3_stage08_b04_formal_8403f003.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110642Z_e458c69972d3_stage08_b04_formal_8403f004.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110642Z_e458c69972d3_stage08_b04_formal_8403f005.csv`、`operator_dtype_evidence_correlate2d_not_applicable_20260809T110645Z_e458c69972d3_stage08_b04_formal_8403f006.csv`等16个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260809T110300Z_2c86271438dd_stage08_b04_smoke_84000001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `correlate2d__correlate2d_fp32_10e2__11d622e248c4eb6f` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `correlate2d` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_same_dtype_arrays_native_direct` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `in1:FP32:[10x10]:bytes=400:sha256=69e4595433fdcbb3baf657e00e7ab3a034d6144f697...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `11d622e248c4eb6feb2d50c956b6ad821146c38d461e2f6ccc615e349c801118` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 12 | `native_capability` | string | NA | 样本未见空值 | 平台或后端原生能力说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `native` | NA |
| 13 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 14 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_620652473711"></a>
### SCHEMA_620652473711

- 适用文件：`operator_windows_chebwin_memory_trace_not_applicable.csv`、`operator_windows_chebwin_memory_trace_not_applicable.csv`、`operator_windows_chebwin_memory_trace_not_applicable.csv`、`operator_windows_chebwin_memory_trace_not_applicable.csv`、`operator_windows_general_cosine_memory_trace_not_applicable.csv`、`operator_windows_general_cosine_memory_trace_not_applicable.csv`、`operator_windows_general_cosine_memory_trace_not_applicable.csv`、`operator_windows_general_cosine_memory_trace_not_applicable.csv`等2084个文件
- 一行代表：一行对应一个资源轨迹采样点。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01_chebwin_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `7acfa10f463087d911cfc45d3dbcf0d9fbf434a48d44ef73332618d8fc85f1f0` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `sample_index` | integer | count | 样本未见空值 | 逐轮测量样本序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 26 | `trace_phase` | string | NA | 样本未见空值 | 资源轨迹采样阶段。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `before` | NA |
| 27 | `phase_index` | integer | count | 样本未见空值 | 资源轨迹阶段内的序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 28 | `cpu_live_heap_bytes` | integer | bytes | 样本未见空值 | CPU当前存活堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1581632` | NA |
| 29 | `rss_bytes` | integer | bytes | 样本未见空值 | 进程常驻内存集大小，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70709248` | NA |
| 30 | `gpu_used_bytes` | integer | bytes | 允许 | GPU当前已使用显存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |

<a id="schema_657bbb94eddf"></a>
### SCHEMA_657BBB94EDDF

- 适用文件：`4类代表算子候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `category` | string | NA | 样本未见空值 | 汇总、代表算子或图表对象所属的分类名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `最高性能收益` | NA |
| 2 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `spectrogram` | NA |
| 3 | `rule` | string | NA | 样本未见空值 | 产生当前记录或选择结果所采用的`rule`。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `全部case精度PASS后median_speedup最高` | NA |
| 4 | `overall_median_speedup` | float | dimensionless | 样本未见空值 | 当前全集内算子或case加速比的总体中位数，具体粒度由同表行定义确定。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.260742718902332` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 5 | `operator_median_speedup` | float | dimensionless | 样本未见空值 | 该算子全部纳入case的加速比中位数。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `10.4629706065054` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 6 | `cases_below_1x` | integer | NA | 样本未见空值 | 加速比小于1的case数量，即该case下GPU平均耗时高于CPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 7 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_715a16f6f7f0"></a>
### SCHEMA_715A16F6F7F0

- 适用文件：`operator_dtype_evidence_firwin_not_applicable_20260810T102454Z_d5013d1285d1_stage07_b04_smoke_2d8e7d36.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_0727f4d3.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_4eb196b4.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_60429611.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_66895c8f.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_8e89afd0.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_edec60b5.csv`、`operator_dtype_evidence_hamming_not_applicable_20260810T101628Z_20e76c9b0ef0_stage07_b04_formal_f0147759.csv`等64个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T102454Z_d5013d1285d1_stage07_b04_smoke_2d8e7d36` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `firwin__firwin_fp32_10e4__3a39c5a92af24753` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `firwin` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cutoff:FP32:sha256=fb7614b89815f3ebaaba34a1cbd1fe490c839c39571d6a45e3a3cff5fc...` | NA |
| 7 | `typed_array_digest` | string | NA | 样本未见空值 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `fb7614b89815f3ebaaba34a1cbd1fe490c839c39571d6a45e3a3cff5fcb1383c` | NA |
| 8 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"1","generator":"configured_cutoff_scalar","...` | NA |
| 9 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"batch_count":10,"cutoff_hz":60,"fs_hz":1000,"numtaps":1023,"output_dtype":"...` | NA |
| 10 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `3a39c5a92af24753ffc1113b8353fc5ddb14ada0d1ad4b483a756fd8010334b5` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 12 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 13 | `task_gpu_call_status` | string | NA | 样本未见空值 | 任务GPU调用闭包检查状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `proved_gpu_cpu` | NA |
| 14 | `link_closure_note` | string | NA | 样本未见空值 | 任务、算子、CSV和日志关联闭包说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust_linked_for_shared_translation_unit_not_executed` | NA |
| 15 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 16 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_7599df084a43"></a>
### SCHEMA_7599DF084A43

- 适用文件：`operator_dtype_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010002.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010003.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082001Z_8b745f9a6864_stage08_b01_formal_a1010004.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082001Z_8b745f9a6864_stage08_b01_formal_a1010005.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082008Z_8b745f9a6864_stage08_b01_formal_a1010006.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082014Z_8b745f9a6864_stage08_b01_formal_a1010007.csv`、`operator_dtype_evidence_chebwin_fft_thrust_20260810T082014Z_8b745f9a6864_stage08_b01_formal_a1010008.csv`等211个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T082000Z_8b745f9a6864_stage08_b01_formal_a1010001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__37f607f8470837ce__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_typed_scalar_input` | NA |
| 7 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 8 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `length:INT32=100\|sym:BOOL=true\|attenuation_db:FP32:bytes=0000a042` | NA |
| 9 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"input_name":"window_positions","shape":"[100]","dtype":"INDEX_INT64","elem...` | NA |
| 10 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"length":100,"requested_dtype":"FP32","sym":true,"attenuation_db":80,"algori...` | NA |
| 11 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `37f607f8470837cec7654ff6bc53e232897f3290174c1e7eb004ee8f2413f934` | NA |
| 12 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP64` | NA |
| 13 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 14 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_7ed5a58734dd"></a>
### SCHEMA_7ED5A58734DD

- 适用文件：`全算子_逐case绘图候选.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 2 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 3 | `selection_status` | string | NA | 样本未见空值 | `selection`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PLOTTING_CANDIDATE_NEEDS_FINAL_QUALIFICATION` | NA |
| 4 | `formal_run_records_for_case` | integer | NA | 样本未见空值 | 同一case在去重前存在的正式运行记录数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 5 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07` | NA |
| 6 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07_b06_formal_6132f71` | NA |
| 7 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 8 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage07_b06_formal_6132f71/formal/20260810T110446Z_6132f71eb1e0_stage07_...` | NA |
| 9 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110446Z_6132f71eb1e0_stage07_b06_formal_fa81bf2c` | NA |
| 10 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 11 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110446Z` | NA |
| 12 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `6132f71eb1e0457ef1988f98e7e02226424d461c` | NA |
| 13 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 14 | `task_name` | string | NA | 样本未见空值 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `Task2` | NA |
| 15 | `step_name` | string | NA | 样本未见空值 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `Step3` | NA |
| 16 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 17 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp16_10e2__cb15388a9d7aeedc` | NA |
| 18 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic_fp16_10e2` | NA |
| 19 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 20 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `128` | NA |
| 21 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[128]` | NA |
| 22 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 23 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 24 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 25 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 26 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 27 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 28 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0007995200000000001` | warmup_runs, measured_runs, timing_scope |
| 29 | `cpu_p50_ms` | float | ms | 样本未见空值 | CPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0008` | warmup_runs, measured_runs, timing_scope |
| 30 | `cpu_p95_ms` | float | ms | 样本未见空值 | CPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0008` | warmup_runs, measured_runs, timing_scope |
| 31 | `cpu_p99_ms` | float | ms | 样本未见空值 | CPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0008107000000000004` | warmup_runs, measured_runs, timing_scope |
| 32 | `cpu_min_ms` | float | ms | 样本未见空值 | CPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.00079` | warmup_runs, measured_runs, timing_scope |
| 33 | `cpu_max_ms` | float | ms | 样本未见空值 | CPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.00088` | warmup_runs, measured_runs, timing_scope |
| 34 | `cpu_std_ms` | float | ms | 样本未见空值 | CPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `8.876350601457788e-06` | warmup_runs, measured_runs, timing_scope |
| 35 | `cpu_cv` | float | dimensionless | 样本未见空值 | CPU耗时变异系数，计算为cpu_std_ms/cpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.011102099511529151` | NA |
| 36 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.18717427` | warmup_runs, measured_runs, timing_scope |
| 37 | `gpu_p50_ms` | float | ms | 样本未见空值 | GPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.183809` | warmup_runs, measured_runs, timing_scope |
| 38 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.2024` | warmup_runs, measured_runs, timing_scope |
| 39 | `gpu_p99_ms` | float | ms | 样本未见空值 | GPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.24357071000000027` | warmup_runs, measured_runs, timing_scope |
| 40 | `gpu_min_ms` | float | ms | 样本未见空值 | GPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.178083` | warmup_runs, measured_runs, timing_scope |
| 41 | `gpu_max_ms` | float | ms | 样本未见空值 | GPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.297695` | warmup_runs, measured_runs, timing_scope |
| 42 | `gpu_std_ms` | float | ms | 样本未见空值 | GPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.014965651709066998` | warmup_runs, measured_runs, timing_scope |
| 43 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0799557103071218` | NA |
| 44 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.004271527277760988` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 45 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `typed_cpu_task2_reference` | NA |
| 46 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 47 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 48 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 49 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 50 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 51 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 52 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `output_count=128;output_dtype=FP16;task_gpu_call=present` | NA |
| 53 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 54 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 55 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 56 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 57 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 58 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 59 | `cpu_row_present` | boolean | NA | 样本未见空值 | `cpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |
| 60 | `gpu_row_present` | boolean | NA | 样本未见空值 | `gpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：`folder_manifest.csv`、`folder_manifest.csv`、`folder_manifest.csv`、`folder_manifest.csv`、`folder_manifest.csv`、`folder_manifest.csv`、`folder_manifest.csv`、`folder_manifest.csv`等2625个文件
- 一行代表：一行对应当前目录递归范围内的一个文件。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`relative_path`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `relative_path` | string | NA | 样本未见空值 | 相对于当前目录或表格约定根目录的文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 2 | `filename` | string | NA | 样本未见空值 | 文件名。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 3 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 4 | `file_size` | string | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 5 | `sha256` | string | NA | 样本未见空值 | 文件内容SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 6 | `source_host` | string | NA | 样本未见空值 | 证据来源主机。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 7 | `source_container` | string | NA | 样本未见空值 | 证据来源容器。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 8 | `source_path` | string | NA | 样本未见空值 | 证据在来源环境中的路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 9 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 10 | `test_object` | string | NA | 样本未见空值 | 测试对象的统一名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 11 | `task` | string | NA | 样本未见空值 | 任务名称或任务标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step` | string | NA | 样本未见空值 | 任务步骤标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 14 | `operator` | string | NA | 样本未见空值 | 算子标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 15 | `run_id` | string | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 16 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 17 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 18 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 19 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 20 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 21 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 22 | `git_dirty` | string | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 23 | `evidence_status` | string | NA | 样本未见空值 | 证据资格或归档状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `NA` | NA |
| 24 | `formal_eligible` | string | NA | 样本未见空值 | 该证据是否满足正式使用资格。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 25 | `related_csv` | string | NA | 样本未见空值 | 与当前证据关联的CSV路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 26 | `related_log` | string | NA | 样本未见空值 | 与当前证据关联的完整日志路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 27 | `related_screenshot` | string | NA | 样本未见空值 | 与当前证据关联的截图路径或ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 28 | `csv_schema_id` | string | NA | 样本未见空值 | CSV真实表头对应的稳定schema ID，由规范化表头SHA-256前缀生成。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 29 | `csv_column_document` | string | NA | 样本未见空值 | 该CSV逐列说明文件及schema锚点的相对路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 30 | `csv_header_sha256` | string | NA | 样本未见空值 | `csv_header`所指对象内容的SHA-256摘要。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 31 | `csv_column_count` | string | count | 样本未见空值 | `csv column`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 32 | `notes` | string | NA | 样本未见空值 | 补充说明、限制、NA规则或审计备注。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_7fa1b27bf151"></a>
### SCHEMA_7FA1B27BF151

- 适用文件：`file_catalog.csv`、`file_catalog.csv`
- 一行代表：一行对应一个文件或证据条目。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_input_contract_audit_59e24c653eae.txt` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `other` | NA |
| 4 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `other` | NA |
| 5 | `extension` | string | NA | 样本未见空值 | 文件扩展名，用于文件类型统计和筛选。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `.txt` | NA |
| 6 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2165` | NA |
| 7 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_input_contract_audit_59e24c653eae.txt` | NA |

<a id="schema_8c1017e6e31d"></a>
### SCHEMA_8C1017E6E31D

- 适用文件：`recovery_batches.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`recovery_batch_id`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `recovery_batch_id` | string | NA | 样本未见空值 | 一次证据回收批次的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `RB_20260810T1446CST_STAGE07_08` | NA |
| 2 | `recovery_timestamp` | datetime | datetime | 样本未见空值 | 回收批次时间。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2026-08-10T14:46:00+08:00` | NA |
| 3 | `source_host` | string | NA | 样本未见空值 | 证据来源主机。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `usr_02@10.110.12.10` | NA |
| 4 | `source_container` | string | NA | 样本未见空值 | 证据来源容器。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `gpu_02` | NA |
| 5 | `source_path` | string | NA | 样本未见空值 | 证据在来源环境中的路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `/tmp/zkx_stage07_08_20260810T1446CST.tar.gz` | NA |
| 6 | `target_path` | string | NA | 样本未见空值 | 回收、复制或生成的目标路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/\|04_算子结果/remaining_op...` | NA |
| 7 | `run_id` | string | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 8 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 9 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 10 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `MULTIPLE_DECLARED` | NA |
| 11 | `file_count` | integer | count | 样本未见空值 | `file`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `17301` | NA |
| 12 | `total_bytes` | integer | bytes | 样本未见空值 | 文件集合总字节数。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `13097773` | NA |
| 13 | `source_sha256` | string | NA | 样本未见空值 | 来源文件或来源清单记录的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `619370544e5a7c615ca88dd0b69e5bff018a7427973184363b8ff22658d9867e` | NA |
| 14 | `target_sha256` | string | NA | 样本未见空值 | 目标文件SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `619370544e5a7c615ca88dd0b69e5bff018a7427973184363b8ff22658d9867e` | NA |
| 15 | `recovery_status` | string | NA | 样本未见空值 | 回收批次当前状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `RECOVERED_EXPANDED_AND_INDEXED_VERIFIED` | NA |
| 16 | `verification_status` | string | NA | 样本未见空值 | 文件、表头、清单或回收校验状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CONTAINER_HOST_WINDOWS_AND_FINAL_EXPANDED_17301_FILES_VERIFIED` | NA |
| 17 | `notes` | string | NA | 样本未见空值 | 补充说明、限制、NA规则或审计备注。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `联合原始包只作追溯底稿；阶段07和阶段08最终以展开文件和直接索引表呈现` | NA |

<a id="schema_8c46c287c375"></a>
### SCHEMA_8C46C287C375

- 适用文件：`operator_timing_samples_chebwin_not_applicable_20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000.csv`、`operator_timing_samples_chebwin_not_applicable_20260806T103701Z_9ad02548cf6c_stage08_b01_formal_72000001.csv`、`operator_timing_samples_chebwin_not_applicable_20260806T103701Z_9ad02548cf6c_stage08_b01_formal_72000002.csv`、`operator_timing_samples_chebwin_not_applicable_20260806T103742Z_9ad02548cf6c_stage08_b01_formal_72000003.csv`、`operator_timing_samples_general_cosine_not_applicable_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000000.csv`、`operator_timing_samples_general_cosine_not_applicable_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000001.csv`、`operator_timing_samples_general_cosine_not_applicable_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000002.csv`、`operator_timing_samples_general_cosine_not_applicable_20260806T115717Z_9ad02548cf6c_stage08_b01_formal_73000003.csv`等2084个文件
- 一行代表：一行对应一个device的一次原始计时样本。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01_chebwin_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `7acfa10f463087d911cfc45d3dbcf0d9fbf434a48d44ef73332618d8fc85f1f0` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 26 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 27 | `sample_index` | integer | count | 样本未见空值 | 逐轮测量样本序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 28 | `latency_ms` | float | ms | 样本未见空值 | 单轮延迟，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7504529999999998` | warmup_runs, measured_runs, timing_scope |
| 29 | `synchronized` | boolean | NA | 样本未见空值 | 计时前后是否执行了设备同步。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `true` | NA |
| 30 | `input_digest` | string | NA | 样本未见空值 | 本次运行实际输入内容的摘要，用于CPU/GPU配对。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `3cc7913e3c61fc1fd039d8483b30938147f61a68863001bab9eadb1f4294e0a5` | NA |
| 31 | `output_digest` | string | NA | 样本未见空值 | 本次运行输出内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `7b406a9bb69c1457d872f180a26cc44825acbe82dec536e7055b33cbb5977690` | NA |

<a id="schema_917242dcfb95"></a>
### SCHEMA_917242DCFB95

- 适用文件：`operator_dtype_evidence_chirp_not_applicable_20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_4b613850.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_776b788a.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_9336fb23.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_a8241bb1.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_16eaf3b4.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_24943170.csv`、`operator_dtype_evidence_chirp_not_applicable_20260810T092019Z_76c5b6f8b36d_stage07_b02_formal_3942ce5c.csv`等62个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T092018Z_76c5b6f8b36d_stage07_b02_formal_3bf4e4a2` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chirp__chirp_fp32_10e2__bd6ca618747dbd1d` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chirp` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `input:FP32:count=128:bytes=512:sha256=aad0570dbadb76247a0ed63609083a90d570b75...` | NA |
| 7 | `typed_array_digest` | string | NA | 样本未见空值 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `aad0570dbadb76247a0ed63609083a90d570b7510812fd7727c55b30a3fbd68a` | NA |
| 8 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"128","generator":"uniform_time_0_to_31(31*i...` | NA |
| 9 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"f0":0.02,"f1":0.07,"method":"linear","phase_degrees":0,"t1":31}` | NA |
| 10 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `bd6ca618747dbd1d5917fc5e5e2f9ca3bdcdd8661936688e3aa04a56fb4db921` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 12 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 13 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 14 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_9c155bb8c8b9"></a>
### SCHEMA_9C155BB8C8B9

- 适用文件：`file_catalog.csv`
- 一行代表：一行对应一个文件或证据条目。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`relative_path`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `artifact_kind` | string | NA | 样本未见空值 | 证据文件类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `semantic_summary` | NA |
| 2 | `extension` | string | NA | 样本未见空值 | 文件扩展名，用于文件类型统计和筛选。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `.csv` | NA |
| 3 | `file_size` | integer | bytes | 样本未见空值 | 文件大小，单位字节。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `540` | NA |
| 4 | `relative_path` | string | NA | 样本未见空值 | 相对于当前目录或表格约定根目录的文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `abnormal_type_summary.csv` | NA |

<a id="schema_a7b6403aceeb"></a>
### SCHEMA_A7B6403ACEEB

- 适用文件：`窗函数_435case热力图候选.csv`
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 2 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 3 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp16_10e2` | NA |
| 4 | `variant` | string | NA | 样本未见空值 | 输入或算子参数的语义变体名称，例如sym或periodic。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `periodic` | NA |
| 5 | `variant_key` | string | NA | 样本未见空值 | 由case参数和语义变体提取的稳定键，用于同算子、dtype和规模内区分不同输入变体。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1e7167366f8d98d0_periodic` | NA |
| 6 | `case_parameters_json` | JSON text | NA | 允许 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"length":100,"requested_dtype":"FP16","sym":false,"attenuation_db":80,"algor...` | NA |
| 7 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp16_10e2__1e7167366f8d98d0__periodic` | NA |
| 8 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 9 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 10 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.21930527000000008` | warmup_runs, measured_runs, timing_scope |
| 11 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.45719789` | warmup_runs, measured_runs, timing_scope |
| 12 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.15049793271386092` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 13 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.5123675000000001` | warmup_runs, measured_runs, timing_scope |
| 14 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.018416283922291041` | NA |
| 15 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `5.9750327471063565e-14` | accuracy_reference, accuracy_status |
| 16 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.4443880107516393e-07` | accuracy_reference, accuracy_status |
| 17 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `4.4652958722329837e-07` | accuracy_reference, accuracy_status |
| 18 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `4.888965245680027e-07` | accuracy_reference, accuracy_status |
| 19 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 20 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 21 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 22 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260807T052446Z_a97933046e9c_stage08_b01_formal_8a000107` | NA |
| 23 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b01_chebwin_fft_thrust/formal/20260807T052446Z_a97933046e9c_stag...` | NA |
| 24 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 25 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PLOTTING_CANDIDATE_NEEDS_FINAL_QUALIFICATION` | NA |

<a id="schema_ac3d0a8ca003"></a>
### SCHEMA_AC3D0A8CA003

- 适用文件：`operator_dtype_evidence_fm_demod_not_applicable_20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105057Z_2836c8dc200d_stage07_b05_formal_c7d48895.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105057Z_2836c8dc200d_stage07_b05_formal_d783f494.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105057Z_2836c8dc200d_stage07_b05_formal_dcf21822.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105057Z_2836c8dc200d_stage07_b05_formal_f8f457e5.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105058Z_2836c8dc200d_stage07_b05_formal_33458f79.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105058Z_2836c8dc200d_stage07_b05_formal_4b6dc00b.csv`、`operator_dtype_evidence_fm_demod_not_applicable_20260810T105058Z_2836c8dc200d_stage07_b05_formal_76051d68.csv`等46个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T105057Z_2836c8dc200d_stage07_b05_formal_a23c73a8` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fm_demod__fm_demod_fp16_10e2__8f40eac5c2bc7520` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fm_demod` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP16` | NA |
| 6 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `analytic:ComplexFP16:[128]:sha256=01fe27cee1b50b607200b1d7d882b6e3b10eefeb03f...` | NA |
| 7 | `typed_array_digest` | string | NA | 样本未见空值 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `01fe27cee1b50b607200b1d7d882b6e3b10eefeb03fa6f1739f01b435fd3ad52` | NA |
| 8 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"ComplexFP16","element_count":"128","generator":"deterministic_chir...` | NA |
| 9 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"count":128,"output_count":127}` | NA |
| 10 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `8f40eac5c2bc7520eda627facc770b095f1c9df0db1eddd265cbecf831c19e81` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 12 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 13 | `task_gpu_call_status` | string | NA | 样本未见空值 | 任务GPU调用闭包检查状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `proved_gpu_cpu` | NA |
| 14 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 15 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_b8ca1c80348b"></a>
### SCHEMA_B8CA1C80348B

- 适用文件：`cpu_gpu_comparison.csv`、`cpu_gpu_comparison.csv`
- 一行代表：一行对应一个输入case的CPU/GPU成对比较。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b01/formal/20260806T103700Z_9ad02548cf6c_stage08_b01_formal_7200...` | NA |
| 5 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 6 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 7 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 8 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 9 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 10 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 11 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 13 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 14 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 15 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 16 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 17 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 18 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 19 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 20 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 21 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 22 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 23 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 24 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7422645300000004` | warmup_runs, measured_runs, timing_scope |
| 25 | `cpu_p50_ms` | float | ms | 样本未见空值 | CPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7423929999999999` | warmup_runs, measured_runs, timing_scope |
| 26 | `cpu_p95_ms` | float | ms | 样本未见空值 | CPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7533555000000001` | warmup_runs, measured_runs, timing_scope |
| 27 | `cpu_p99_ms` | float | ms | 样本未见空值 | CPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7555925000000001` | warmup_runs, measured_runs, timing_scope |
| 28 | `cpu_min_ms` | float | ms | 样本未见空值 | CPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7322329999999999` | warmup_runs, measured_runs, timing_scope |
| 29 | `cpu_max_ms` | float | ms | 样本未见空值 | CPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7585130000000002` | warmup_runs, measured_runs, timing_scope |
| 30 | `cpu_std_ms` | float | ms | 样本未见空值 | CPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0063154560871800988` | warmup_runs, measured_runs, timing_scope |
| 31 | `cpu_cv` | float | dimensionless | 样本未见空值 | CPU耗时变异系数，计算为cpu_std_ms/cpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0023030076121723013` | NA |
| 32 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.4128106099999997` | warmup_runs, measured_runs, timing_scope |
| 33 | `gpu_p50_ms` | float | ms | 样本未见空值 | GPU耗时第50百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.3976614999999999` | warmup_runs, measured_runs, timing_scope |
| 34 | `gpu_p95_ms` | float | ms | 样本未见空值 | GPU耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.462828` | warmup_runs, measured_runs, timing_scope |
| 35 | `gpu_p99_ms` | float | ms | 样本未见空值 | GPU耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.5004397900000028` | warmup_runs, measured_runs, timing_scope |
| 36 | `gpu_min_ms` | float | ms | 样本未见空值 | GPU最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.3832359999999999` | warmup_runs, measured_runs, timing_scope |
| 37 | `gpu_max_ms` | float | ms | 样本未见空值 | GPU最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.0542250000000002` | warmup_runs, measured_runs, timing_scope |
| 38 | `gpu_std_ms` | float | ms | 样本未见空值 | GPU耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.06816031129754252` | warmup_runs, measured_runs, timing_scope |
| 39 | `gpu_cv` | float | dimensionless | 样本未见空值 | GPU耗时变异系数，计算为gpu_std_ms/gpu_mean_ms。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.048244478640730573` | NA |
| 40 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `1.9409993884459864` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 41 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_typed_reference` | NA |
| 42 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.1434175650342245e-12` | accuracy_reference, accuracy_status |
| 43 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `1.4640415175240845e-06` | accuracy_reference, accuracy_status |
| 44 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.6867182611919047e-06` | accuracy_reference, accuracy_status |
| 45 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.7738614958593431e-06` | accuracy_reference, accuracy_status |
| 46 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 47 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 48 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `finite_fp64_window_shape_and_symmetry_policy_match_cpu:shape=[100];finite=tru...` | NA |
| 49 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 50 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 51 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 52 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 53 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 54 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 55 | `cpu_row_present` | boolean | NA | 样本未见空值 | `cpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |
| 56 | `gpu_row_present` | boolean | NA | 样本未见空值 | `gpu row`是否存在的布尔标记。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `True` | NA |

<a id="schema_bf0ae92fc547"></a>
### SCHEMA_BF0AE92FC547

- 适用文件：`all_operator_abnormal_cases.csv`、`operator_abnormal_cases_cuda_runtime.csv`、`operator_abnormal_cases_not_applicable.csv`、`operator_abnormal_cases_selected_backend.csv`、`operator_abnormal_cases_mixed.csv`、`operator_abnormal_cases_mixed.csv`、`operator_abnormal_cases_mixed.csv`、`operator_abnormal_cases_mixed.csv`等58个文件
- 一行代表：一行对应一个异常输入case及其预期和实际处理结果。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `timestamp` | datetime | datetime | 样本未见空值 | 该记录对应的时间戳；格式和时区以同表说明为准。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2026-08-11T11:35:00Z` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260811T113500Z_617051d_stage09_cubic_00000001` | NA |
| 3 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cuda_runtime` | NA |
| 4 | `test_object` | string | NA | 样本未见空值 | 测试对象的统一名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 5 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 6 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 7 | `module_name` | string | NA | 样本未见空值 | 算子所属模块名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 8 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 9 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines.cubic.backend_call_failure.inject_cuda_kernel_launch_failure_before_...` | NA |
| 10 | `normal_run_id` | datetime | NA | 样本未见空值 | 用于关联异常case的正常基准run ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996` | NA |
| 11 | `normal_case_id` | string | NA | 样本未见空值 | 用于关联异常case的正常基准case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 12 | `abnormal_type` | string | NA | 样本未见空值 | 单条异常输入的语义类别。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `backend_call_failure` | NA |
| 13 | `input_summary` | string | NA | 样本未见空值 | 实际输入shape、dtype、参数或内容的可读摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `mutation=inject CUDA kernel launch failure before output consumption` | NA |
| 14 | `expected_code` | string | NA | 样本未见空值 | 异常case预期返回的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 15 | `captured_code` | string | NA | 样本未见空值 | 测试框架捕获到的返回码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 16 | `returned_code` | string | NA | 样本未见空值 | 程序实际返回码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 17 | `actual_code` | string | NA | 样本未见空值 | 异常case实际返回的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CUDA_RUNTIME_ERROR` | NA |
| 18 | `error_message` | string | NA | 样本未见空值 | 错误码对应的可读错误说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines::cubic捕获CUDA Runtime失败：mutation=inject CUDA kernel launch failure be...` | NA |
| 19 | `backend_error_code` | integer | NA | 允许 | `backend error code`错误码或返回码值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `101` | NA |
| 20 | `backend_error_name` | string | NA | 允许 | 后端错误码对应的符号名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cudaErrorInvalidDevice` | NA |
| 21 | `backend_error_message` | string | NA | 允许 | `backend error`对应的可读说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `invalid device ordinal` | NA |
| 22 | `safe_exit` | boolean | NA | 样本未见空值 | 异常输入是否按设计安全退出。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `true` | NA |
| 23 | `output_consumed` | boolean | NA | 样本未见空值 | 异常或失败输出是否被后续流程消费的布尔标记。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 24 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |

<a id="schema_c09b6d1032ed"></a>
### SCHEMA_C09B6D1032ED

- 适用文件：`operator_dtype_evidence_parzen_not_applicable_20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000100.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000101.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000102.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000103.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000104.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161414Z_1c045bd7e750_stage08_b02_formal_8d000105.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161414Z_1c045bd7e750_stage08_b02_formal_8d000106.csv`、`operator_dtype_evidence_parzen_not_applicable_20260807T161414Z_1c045bd7e750_stage08_b02_formal_8d000107.csv`等92个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000100` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `parzen__parzen_fp32_10e2__7f7d081aa0af5dfa__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `parzen` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `request_variant_no_typed_data_input` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `length:INT32=100\|sym:BOOL=true` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `7f7d081aa0af5dfac9cbd2114e6e1ae81fa1069a67d6310ffd6db02c7f813864` | NA |
| 9 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 10 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_c185cb915cf7"></a>
### SCHEMA_C185CB915CF7

- 适用文件：`operator_main_results_chebwin_not_applicable_20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000.csv`、`operator_main_results_chebwin_not_applicable_20260806T103701Z_9ad02548cf6c_stage08_b01_formal_72000001.csv`、`operator_main_results_chebwin_not_applicable_20260806T103701Z_9ad02548cf6c_stage08_b01_formal_72000002.csv`、`operator_main_results_chebwin_not_applicable_20260806T103742Z_9ad02548cf6c_stage08_b01_formal_72000003.csv`、`operator_main_results_general_cosine_not_applicable_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000000.csv`、`operator_main_results_general_cosine_not_applicable_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000001.csv`、`operator_main_results_general_cosine_not_applicable_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000002.csv`、`operator_main_results_general_cosine_not_applicable_20260806T115717Z_9ad02548cf6c_stage08_b01_formal_73000003.csv`等2084个文件
- 一行代表：一行对应一个输入case在一个device上的汇总测量。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + device`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `schema_version` | integer | NA | 样本未见空值 | CSV结构版本号，用于判断字段契约。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `1` | NA |
| 2 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 3 | `run_type` | string | NA | 样本未见空值 | 运行类型，例如formal、smoke或recovery。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `test_time_utc` | datetime | datetime | 样本未见空值 | 测试发生时间，UTC时区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z` | NA |
| 5 | `git_commit` | string | NA | 样本未见空值 | 运行或生成数据时的Git提交SHA。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `9ad02548cf6c74fae677c6162b7c4a2db2ac0e88` | NA |
| 6 | `git_dirty` | boolean | NA | 样本未见空值 | 运行时工作区是否存在未提交改动的真实状态。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `false` | NA |
| 7 | `config_id` | string | NA | 样本未见空值 | 测试配置的逻辑标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b01_chebwin_scales_v1` | NA |
| 8 | `config_sha256` | string | NA | 样本未见空值 | 测试配置内容的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `7acfa10f463087d911cfc45d3dbcf0d9fbf434a48d44ef73332618d8fc85f1f0` | NA |
| 9 | `target_kind` | string | NA | 样本未见空值 | 被测目标类别，例如task、step或operator。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `operator` | NA |
| 10 | `target` | string | NA | 样本未见空值 | 被测目标名称或入口。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 11 | `task_name` | string | NA | 允许 | 任务名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 12 | `step_name` | string | NA | 允许 | 任务步骤名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 13 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 14 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 15 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 16 | `order_of_magnitude` | string | NA | 样本未见空值 | 目标输入数量级，例如10^2、10^3。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `10^2` | NA |
| 17 | `actual_elements` | integer | count | 样本未见空值 | 本case实际参与计算的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 18 | `input_shape` | string | NA | 样本未见空值 | 实际输入张量形状；多输入算子可用名称与形状组合表示。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `window_positions=[100]` | NA |
| 19 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 20 | `device` | string | NA | 样本未见空值 | 执行设备，例如CPU或GPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `CPU` | NA |
| 21 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 22 | `timing_scope` | string | NA | 样本未见空值 | 计时边界，只有相同边界的数据才能直接比较。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `compatibility/end-to-end` | NA |
| 23 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 24 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 25 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 26 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 27 | `mean_ms` | float | ms | 样本未见空值 | 耗时算术平均值，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7422645300000004` | warmup_runs, measured_runs, timing_scope |
| 28 | `p50_ms` | float | ms | 样本未见空值 | 耗时第50百分位，即中位数，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7423929999999999` | warmup_runs, measured_runs, timing_scope |
| 29 | `p95_ms` | float | ms | 样本未见空值 | 耗时第95百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7533555000000001` | warmup_runs, measured_runs, timing_scope |
| 30 | `p99_ms` | float | ms | 样本未见空值 | 耗时第99百分位，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7555925000000001` | warmup_runs, measured_runs, timing_scope |
| 31 | `min_ms` | float | ms | 样本未见空值 | 最小耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7322329999999999` | warmup_runs, measured_runs, timing_scope |
| 32 | `max_ms` | float | ms | 样本未见空值 | 最大耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7585130000000002` | warmup_runs, measured_runs, timing_scope |
| 33 | `std_ms` | float | ms | 样本未见空值 | 耗时标准差，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.0063154560871800988` | warmup_runs, measured_runs, timing_scope |
| 34 | `cv` | float | dimensionless | 样本未见空值 | 耗时变异系数，计算为std_ms/mean_ms，无量纲。 | std_ms / mean_ms（对应device） | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.0023030076121723013` | NA |
| 35 | `cpu_gpu_speedup` | float | dimensionless | 允许 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `1.9409993884459864` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 36 | `accuracy_reference` | string | NA | 样本未见空值 | 精度比较使用的参考输出来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cpu_typed_reference` | NA |
| 37 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 38 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 39 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 40 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `0` | accuracy_reference, accuracy_status |
| 41 | `exact_match` | string | NA | 允许 | 离散、索引或坐标输出是否逐项完全一致。 | 原始记录字段 | 布尔值；必须结合对应检查对象和状态列。 | `NA` | NA |
| 42 | `mismatch_count` | string | count | 允许 | 与参考输出不一致的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 43 | `semantic_check` | string | NA | 样本未见空值 | 形状、dtype、有限值、输出数量或业务语义门禁摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `finite_fp64_window_shape_and_symmetry_policy_match_cpu:shape=[100];finite=tru...` | NA |
| 44 | `accuracy_status` | string | NA | 样本未见空值 | 依据精度阈值或精确匹配规则得到的精度结论。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 45 | `cpu_heap_before_bytes` | integer | bytes | 样本未见空值 | 运行前CPU堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1581632` | NA |
| 46 | `cpu_heap_after_bytes` | integer | bytes | 样本未见空值 | 运行后CPU堆内存，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1583808` | NA |
| 47 | `cpu_heap_peak_bytes` | integer | bytes | 样本未见空值 | 运行期间CPU堆内存峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `1583808` | NA |
| 48 | `cpu_heap_delta_bytes` | integer | bytes | 样本未见空值 | 运行后减运行前的CPU堆内存变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `2176` | NA |
| 49 | `rss_before_bytes` | integer | bytes | 样本未见空值 | 运行前RSS，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70709248` | NA |
| 50 | `rss_after_bytes` | integer | bytes | 样本未见空值 | 运行后RSS，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70709248` | NA |
| 51 | `rss_peak_bytes` | integer | bytes | 样本未见空值 | 运行期间RSS峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `70709248` | NA |
| 52 | `rss_delta_bytes` | integer | bytes | 样本未见空值 | 运行后减运行前的RSS变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `0` | NA |
| 53 | `gpu_before_bytes` | integer | bytes | 允许 | 运行前GPU显存占用，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |
| 54 | `gpu_after_bytes` | integer | bytes | 允许 | 运行后GPU显存占用，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |
| 55 | `gpu_peak_bytes` | integer | bytes | 允许 | 运行期间GPU显存峰值，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `341000192` | NA |
| 56 | `gpu_delta_bytes` | integer | bytes | 允许 | 运行后减运行前的GPU显存变化，单位字节。 | 原始记录字段 | 单位为字节；正负变化和峰值必须结合采样阶段判读。 | `0` | NA |
| 57 | `stability_case_index` | string | count | 允许 | 稳定性扰动case序号。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 58 | `stability_total_cases` | string | NA | 允许 | 本组稳定性扰动case总数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 59 | `perturbation_file` | string | NA | 允许 | 稳定性扰动配置或输入文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 60 | `perturbation_sha256` | string | NA | 允许 | 稳定性扰动文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 61 | `perturbation_summary` | string | NA | 允许 | 稳定性扰动参数摘要。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 62 | `input_mode` | string | NA | 允许 | 输入生成、选择或加载模式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 63 | `input_source_file` | string | NA | 允许 | 输入数据来源文件。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 64 | `input_source_sha256` | string | NA | 允许 | 输入来源文件的SHA-256。 | 对说明对象内容计算SHA-256 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 65 | `selection_seed` | string | NA | 允许 | 确定性选择输入或case时使用的随机种子。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 66 | `profile_entry_id` | string | NA | 允许 | 性能配置或profile条目的标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |

<a id="schema_c5a345c78c5d"></a>
### SCHEMA_C5A345C78C5D

- 适用文件：`operator_dtype_evidence_firwin2_fft_thrust_20260809T111800Z_6e5af3b6102d_stage08_b04_smoke_84100001.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112202Z_ba26c5229701_stage08_b04_formal_8413f000.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112202Z_ba26c5229701_stage08_b04_formal_8413f001.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112202Z_ba26c5229701_stage08_b04_formal_8413f002.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112203Z_ba26c5229701_stage08_b04_formal_8413f003.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112203Z_ba26c5229701_stage08_b04_formal_8413f004.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112204Z_ba26c5229701_stage08_b04_formal_8413f005.csv`、`operator_dtype_evidence_firwin2_fft_thrust_20260809T112204Z_ba26c5229701_stage08_b04_formal_8413f006.csv`等16个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260809T111800Z_6e5af3b6102d_stage08_b04_smoke_84100001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `firwin2__firwin2_fp16_10e2__09117d6346d84820` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `firwin2` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_same_dtype_freq_gain_arrays` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `freq:FP16:[3]:bytes=6:sha256=7a4f57ca7425bdc7f8ec06318f9d2388ff2223c589a45ef7...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `09117d6346d848201f91fc3a2cec37480ccee8e7d902ea44e2190da2290e7463` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP64` | NA |
| 12 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 13 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 14 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_c6d6d1c6c971"></a>
### SCHEMA_C6D6D1C6C971

- 适用文件：`input_evidence_index.csv`、`input_evidence_index.csv`
- 一行代表：一行对应一个索引对象及其来源关系。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`stage + operator_name + case_id`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `source_group` | string | NA | 样本未见空值 | 原始回收目录组或运行批次组，用于区分重跑与来源。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08_b02_strict_dtype` | NA |
| 3 | `source_partition` | string | NA | 样本未见空值 | 原路径中的formal、smoke、history或其他证据分区。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `formal` | NA |
| 4 | `evidence_type` | string | NA | 样本未见空值 | 输入、dtype、运行、日志或其他证据的类型标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `dtype_and_input_contract` | NA |
| 5 | `source_relative_path` | string | NA | 样本未见空值 | 相对于当前批次根的原始证据文件路径。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `runs/stage08_b02_strict_dtype/formal/20260807T161413Z_1c045bd7e750_stage08_b0...` | NA |
| 6 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260807T161413Z_1c045bd7e750_stage08_b02_formal_8d000100` | NA |
| 7 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `parzen__parzen_fp32_10e2__7f7d081aa0af5dfa__sym` | NA |
| 8 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `parzen` | NA |
| 9 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `config_input_dtype` | string | NA | 允许 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 11 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `request_variant_no_typed_data_input` | NA |
| 12 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 13 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `length:INT32=100\|sym:BOOL=true` | NA |
| 14 | `typed_array_digest` | string | NA | 允许 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `NA` | NA |
| 15 | `input_content_json` | string | NA | 允许 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 16 | `case_parameters_json` | string | NA | 允许 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 17 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `7f7d081aa0af5dfac9cbd2114e6e1ae81fa1069a67d6310ffd6db02c7f813864` | NA |
| 18 | `compute_dtype` | string | NA | 允许 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 19 | `execution_dtype` | string | NA | 允许 | 执行路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 20 | `fft_dtype` | string | NA | 允许 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 21 | `output_dtype` | string | NA | 允许 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 22 | `backend` | string | NA | 允许 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 23 | `input_name` | string | NA | 允许 | 一个输入参数或输入数组的名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 24 | `shape` | string | NA | 允许 | 输入或输出数组的形状。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 25 | `dtype` | string | NA | 允许 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 26 | `element_count` | string | count | 允许 | 数组中的元素数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 27 | `generator` | string | NA | 允许 | 输入数据生成器或生成规则。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 28 | `preview_head4_tail2` | string | NA | 允许 | 大型输入前4项与后2项的可读预览。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA` | NA |
| 29 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 30 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_c74b9b1781e2"></a>
### SCHEMA_C74B9B1781E2

- 适用文件：`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080820Z_1662dd06774d_stage07_b01_formal_7b80cbb4.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080820Z_1662dd06774d_stage07_b01_formal_8e5e5ded.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080825Z_1662dd06774d_stage07_b01_formal_2a8442f8.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080825Z_1662dd06774d_stage07_b01_formal_444065a9.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080826Z_1662dd06774d_stage07_b01_formal_2d27644c.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080838Z_1662dd06774d_stage07_b01_formal_422397e3.csv`、`operator_dtype_evidence_pulse_compression_fft_thrust_20260810T080838Z_1662dd06774d_stage07_b01_formal_ab40795a.csv`等92个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T080820Z_1662dd06774d_stage07_b01_formal_29e3eb18` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `pulse_compression__pc_fp32_10e3__b072f67b4953fb9f` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `pulse_compression` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 6 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `echo:FP32:bytes=8192:sha256=9843ab277c09b77ec85b6335204580212f7ca5c2f208d7118...` | NA |
| 7 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"ComplexFP32","element_count":"1024","generator":"complex_modular(s...` | NA |
| 8 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"nfft":256,"normalize":false,"num_pulses":4,"pulse_samples":64,"samples_per_...` | NA |
| 9 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `b072f67b4953fb9f05cc39a488cdb77ccb10817bcf3026fe5f0a186c010ec383` | NA |
| 10 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 11 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 12 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 13 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_c9856f6503c1"></a>
### SCHEMA_C9856F6503C1

- 适用文件：`operator_dtype_evidence_cubic_not_applicable_20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4bb912c8.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110445Z_6132f71eb1e0_stage07_b06_formal_aa0f2ca5.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110446Z_6132f71eb1e0_stage07_b06_formal_057f9050.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110446Z_6132f71eb1e0_stage07_b06_formal_588ecc4c.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110446Z_6132f71eb1e0_stage07_b06_formal_951c5aa8.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110446Z_6132f71eb1e0_stage07_b06_formal_a0fe3bdd.csv`、`operator_dtype_evidence_cubic_not_applicable_20260810T110446Z_6132f71eb1e0_stage07_b06_formal_ba7c2680.csv`等46个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T110445Z_6132f71eb1e0_stage07_b06_formal_4589f996` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `config_input_dtype` | string | NA | 样本未见空值 | 配置文件声明的输入dtype。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `execution_dtype` | string | NA | 样本未见空值 | 执行路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x:FP32:[128]:sha256=43e38285a44bebb006325d572fbc120149872a34f0981d8e644e56e6f...` | NA |
| 8 | `typed_array_digest` | string | NA | 样本未见空值 | 带类型数组内容的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `43e38285a44bebb006325d572fbc120149872a34f0981d8e644e56e6f943985f` | NA |
| 9 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"128","generator":"deterministic_cubic_domai...` | NA |
| 10 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"count":128,"output_count":128}` | NA |
| 11 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `5407de070298d52f158fc0772fb1d3ed2c6ea8226fdc463773983dee4f113ca5` | NA |
| 12 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 13 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 14 | `task_gpu_call_status` | string | NA | 样本未见空值 | 任务GPU调用闭包检查状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `proved_gpu_cpu` | NA |
| 15 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 16 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_cb6b6e50b568"></a>
### SCHEMA_CB6B6E50B568

- 适用文件：`windows_batch01_module_summary_20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000.csv`、`windows_batch01_module_summary_20260806T103701Z_9ad02548cf6c_stage08_b01_formal_72000001.csv`、`windows_batch01_module_summary_20260806T103701Z_9ad02548cf6c_stage08_b01_formal_72000002.csv`、`windows_batch01_module_summary_20260806T103742Z_9ad02548cf6c_stage08_b01_formal_72000003.csv`、`windows_batch01_module_summary_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000000.csv`、`windows_batch01_module_summary_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000001.csv`、`windows_batch01_module_summary_20260806T115716Z_9ad02548cf6c_stage08_b01_formal_73000002.csv`、`windows_batch01_module_summary_20260806T115717Z_9ad02548cf6c_stage08_b01_formal_73000003.csv`等736个文件
- 一行代表：一行对应一个汇总对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`operator_name + case_id + dtype + backend`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260806T103700Z_9ad02548cf6c_stage08_b01_formal_72000000` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin__chebwin_fp32_10e2__3cc7913e3c61fc1f__sym` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin` | NA |
| 4 | `scale_id` | string | NA | 样本未见空值 | 输入规模的可读标签。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `chebwin_fp32_10e2` | NA |
| 5 | `dtype` | string | NA | 样本未见空值 | 本case实际测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 6 | `sym` | boolean | NA | 样本未见空值 | 窗函数是否使用对称形式的布尔参数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `true` | NA |
| 7 | `backend` | string | NA | 样本未见空值 | 实现后端；not_applicable表示不需要该子分类，不代表没有GPU结果。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `not_applicable` | NA |
| 8 | `warmup_runs` | integer | count | 样本未见空值 | 正式计时前的预热次数，不计入性能统计。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20` | NA |
| 9 | `measured_runs` | integer | count | 样本未见空值 | 进入性能统计的重复测量次数。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `100` | NA |
| 10 | `cpu_mean_ms` | float | ms | 样本未见空值 | 同一case的CPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `2.7422645300000004` | warmup_runs, measured_runs, timing_scope |
| 11 | `gpu_mean_ms` | float | ms | 样本未见空值 | 同一case的GPU平均耗时，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.4128106099999997` | warmup_runs, measured_runs, timing_scope |
| 12 | `cpu_gpu_speedup` | float | dimensionless | 样本未见空值 | CPU平均耗时除以GPU平均耗时；大于1表示GPU更快。 | cpu_mean_ms / gpu_mean_ms | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `1.9409993884459864` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 13 | `mse` | float | dimensionless | 样本未见空值 | CPU参考与GPU输出差值绝对值平方的平均值。 | mean(abs(y_gpu-y_ref)^2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.1434175650342245e-12` | accuracy_reference, accuracy_status |
| 14 | `rmse` | float | dimensionless | 样本未见空值 | MSE的平方根，与输出值同量纲。 | sqrt(mse) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `1.4640415175240845e-06` | accuracy_reference, accuracy_status |
| 15 | `relative_l2` | float | dimensionless | 样本未见空值 | 误差L2范数除以参考输出L2范数。 | norm(y_gpu-y_ref,2) / norm(y_ref,2) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `2.6867182611919047e-06` | accuracy_reference, accuracy_status |
| 16 | `relative_linf` | float | dimensionless | 样本未见空值 | 误差最大绝对值除以参考输出最大绝对值。 | max(abs(y_gpu-y_ref)) / max(abs(y_ref)) | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.7738614958593431e-06` | accuracy_reference, accuracy_status |
| 17 | `cuda_event_ms` | float | ms | 样本未见空值 | 使用CUDA兼容事件测得的设备时间，单位毫秒。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `1.403596043586731` | warmup_runs, measured_runs, timing_scope |
| 18 | `cuda_event_source` | string | NA | 样本未见空值 | CUDA兼容事件计时值的来源或采集方式。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cudaEvent` | NA |
| 19 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 20 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |

<a id="schema_d383c06b6201"></a>
### SCHEMA_D383C06B6201

- 适用文件：`53算子_全case性能区间候选.csv`
- 一行代表：一行对应一个由该CSV生成程序定义的记录对象。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic` | NA |
| 2 | `module` | string | NA | 样本未见空值 | 算子所属模块标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `bsplines` | NA |
| 3 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage07` | NA |
| 4 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 5 | `pass_count` | integer | count | 样本未见空值 | `pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 6 | `total_case_count` | integer | count | 样本未见空值 | `total case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 7 | `median_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的中位数。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.00689504332193441` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 8 | `p25_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的第25百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.00277444310276553` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 9 | `p75_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的第75百分位。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.0359768450042551` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 10 | `min_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的最小值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.000774836200325786` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 11 | `max_speedup` | float | dimensionless | 样本未见空值 | 当前聚合范围内`speedup`的最大值。 | 原始记录字段 | 大于1表示GPU更快；小于1表示GPU更慢；必须与同case计时边界联读。 | `0.362282531435371` | cpu_mean_ms, gpu_mean_ms, timing_scope |
| 12 | `cases_below_1x` | integer | NA | 样本未见空值 | 加速比小于1的case数量，即该case下GPU平均耗时高于CPU。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `15` | NA |
| 13 | `worst_accuracy_ratio` | string | dimensionless | 样本未见空值 | 最差精度值相对其适用阈值的比值；阈值未绑定时必须为明确NA状态，不能臆造。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_THRESHOLD_SNAPSHOT_NOT_BOUND` | NA |
| 14 | `worst_accuracy_case_id` | string | NA | 样本未见空值 | 产生`worst_accuracy`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `NA_SEPARATE_METRICS_REPORTED` | NA |
| 15 | `max_mse` | float | NA | 样本未见空值 | 当前聚合范围内`mse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `8.67362e-17` | NA |
| 16 | `max_mse_case_id` | string | NA | 样本未见空值 | 产生`max_mse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 17 | `max_rmse` | float | NA | 样本未见空值 | 当前聚合范围内`rmse`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `9.31323e-09` | NA |
| 18 | `max_rmse_case_id` | string | NA | 样本未见空值 | 产生`max_rmse`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 19 | `max_relative_l2` | float | NA | 样本未见空值 | 当前聚合范围内`relative l2`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `3.01964e-08` | NA |
| 20 | `max_relative_l2_case_id` | string | NA | 样本未见空值 | 产生`max_relative_l2`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 21 | `max_relative_linf` | float | NA | 样本未见空值 | 当前聚合范围内`relative linf`的最大值。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `8.94584e-08` | NA |
| 22 | `max_relative_linf_case_id` | string | NA | 样本未见空值 | 产生`max_relative_linf`指标或结论的case ID。 | 原始记录字段 | 数值越小表示与参考结果越接近；阈值必须来自对应配置或规则快照。 | `cubic__cubic_fp32_10e2__5407de070298d52f` | NA |
| 23 | `max_mismatch_count` | integer | count | 允许 | 当前聚合范围内`mismatch count`的最大值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |
| 24 | `max_gpu_p95_ms` | float | ms | 样本未见空值 | 当前聚合范围内`gpu p95 ms`的最大值。 | 原始记录字段 | 单位为毫秒；不同timing_scope或同步方式不可直接混合。 | `0.27393255` | warmup_runs, measured_runs, timing_scope |
| 25 | `max_gpu_p95_case_id` | string | NA | 样本未见空值 | 产生`max_gpu_p95`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp32_10e4__4ebb8a2d94032c64` | NA |
| 26 | `max_gpu_cv` | float | NA | 样本未见空值 | 当前聚合范围内`gpu cv`的最大值。 | 原始记录字段 | 越小通常表示耗时越稳定；必须结合样本数和计时边界。 | `0.08293981931311126` | NA |
| 27 | `max_gpu_cv_case_id` | string | NA | 样本未见空值 | 产生`max_gpu_cv`指标或结论的case ID。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `cubic__cubic_fp16_10e4__1b20c6a49d276caf` | NA |
| 28 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 29 | `selection_rule_id` | string | NA | 样本未见空值 | 筛选、去重或代表对象选择规则的版本化标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `LATEST_FORMAL_PASS_PER_CASE_V1` | NA |
| 30 | `qualification_status` | string | NA | 样本未见空值 | `qualification`的状态枚举值；具体允许值以生成程序和同表记录为准。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `CANDIDATE_NOT_FINAL` | NA |

<a id="schema_e7ec8aca9a36"></a>
### SCHEMA_E7EC8ACA9A36

- 适用文件：`operator_coverage.csv`、`operator_coverage.csv`
- 一行代表：一行对应一个测试对象的覆盖汇总。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`未声明；使用完整行及来源路径识别，正式派生前需检查唯一性。`
- 配对或关联键：`NA`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `stage` | string | NA | 样本未见空值 | 证据所属开发阶段或归档阶段标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `stage08` | NA |
| 2 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly` | NA |
| 3 | `case_count` | integer | count | 样本未见空值 | `case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 4 | `formal_case_count` | integer | count | 样本未见空值 | `formal case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `46` | NA |
| 5 | `smoke_case_count` | integer | count | 允许 | `smoke case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `3` | NA |
| 6 | `dtypes` | string | NA | 样本未见空值 | `dtypes`对应的数据类型或数据类型集合。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP16\|FP32\|INT16\|INT32\|INT8` | NA |
| 7 | `backends` | string | NA | 样本未见空值 | 当前汇总对象覆盖的后端集合，通常使用竖线分隔。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `fft_thrust` | NA |
| 8 | `scale_ids` | string | NA | 样本未见空值 | `scale ids`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly_fp16_10e2\|channelize_poly_fp16_10e3\|channelize_poly_fp16_10e4...` | NA |
| 9 | `input_shapes` | string | NA | 样本未见空值 | `input shapes`取值集合，通常使用分隔符连接。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x=[1024];h=[64]\|x=[128];h=[32]\|x=[16384];h=[256]` | NA |
| 10 | `min_actual_elements` | integer | NA | 样本未见空值 | 当前聚合范围内`actual elements`的最小值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `160` | NA |
| 11 | `max_actual_elements` | integer | NA | 样本未见空值 | 当前聚合范围内`actual elements`的最大值。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `16640` | NA |
| 12 | `status_pass_count` | integer | count | 样本未见空值 | `status pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 13 | `accuracy_pass_count` | integer | count | 样本未见空值 | `accuracy pass`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 14 | `numeric_error_case_count` | integer | count | 样本未见空值 | `numeric error case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `47` | NA |
| 15 | `exact_match_case_count` | integer | count | 样本未见空值 | `exact match case`对象或事件的数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `0` | NA |

<a id="schema_f790a782cfd0"></a>
### SCHEMA_F790A782CFD0

- 适用文件：`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073905Z_5bf09d215ce4_stage08_b03_formal_99010001.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073905Z_5bf09d215ce4_stage08_b03_formal_99010002.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073905Z_5bf09d215ce4_stage08_b03_formal_99010003.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073915Z_5bf09d215ce4_stage08_b03_formal_99010004.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073915Z_5bf09d215ce4_stage08_b03_formal_99010005.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073916Z_5bf09d215ce4_stage08_b03_formal_99010006.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073942Z_5bf09d215ce4_stage08_b03_formal_99010007.csv`、`operator_dtype_evidence_channelize_poly_fft_thrust_20260810T073943Z_5bf09d215ce4_stage08_b03_formal_99010008.csv`等365个文件
- 一行代表：一行对应一个case记录。
- 编码与格式：UTF-8；逗号分隔；遵循RFC 4180引号转义。
- 主键：`run_id + case_id + operator_name`
- 配对或关联键：`run_id + case_id`
- 表头摘要算法：按真实列顺序使用U+001F连接列名后计算SHA-256。

| 序号 | 列名 | 类型 | 单位 | 可空/NA | 含义 | 来源或公式 | 判读方法 | 真实示例 | 相关列 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `run_id` | datetime | NA | 样本未见空值 | 一次运行的唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `20260810T073905Z_5bf09d215ce4_stage08_b03_formal_99010001` | NA |
| 2 | `case_id` | string | NA | 样本未见空值 | 输入参数、dtype、规模和语义变体共同确定的case唯一标识。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly__channelize_poly_fp32_10e2__b6e2f353c8853cdd` | NA |
| 3 | `operator_name` | string | NA | 样本未见空值 | 算子名称。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `channelize_poly` | NA |
| 4 | `requested_dtype` | string | NA | 样本未见空值 | 配置请求测试的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 5 | `dtype_semantics` | string | NA | 样本未见空值 | 该算子对输入、计算和输出dtype的业务语义说明。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `actual_typed_array_inputs_fp32_compute_extension` | NA |
| 6 | `typed_input_count` | integer | count | 样本未见空值 | 带明确dtype的输入对象数量。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `2` | NA |
| 7 | `typed_input_manifest` | string | NA | 样本未见空值 | 各输入名称、shape和dtype组成的清单。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `x:FP32:[128]:bytes=512:sha256=a77bf42e2c86562f8cfd734074f488b4350e3e3126ca245...` | NA |
| 8 | `actual_input_digest` | string | NA | 样本未见空值 | 程序实际消费输入的摘要。 | 原始记录字段 | 用于身份和一致性核验，不代表性能或精度结论。 | `b6e2f353c8853cddcf85e3c02d052280d158379ab9b9c415f42db9ed5e61d864` | NA |
| 9 | `compute_dtype` | string | NA | 样本未见空值 | 核心计算实际使用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `FP32` | NA |
| 10 | `fft_dtype` | string | NA | 样本未见空值 | FFT子路径实际采用的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 11 | `output_dtype` | string | NA | 样本未见空值 | 输出结果的数据类型。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `ComplexFP32` | NA |
| 12 | `status` | string | NA | 样本未见空值 | 该记录的执行或验收状态。 | 原始记录字段 | 按枚举值判断，不得仅凭非空推断为PASS。 | `PASS` | NA |
| 13 | `error_code` | string | NA | 样本未见空值 | 程序返回或证据门禁使用的错误码。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `OK` | NA |
| 14 | `input_content_json` | JSON text | NA | 样本未见空值 | 实际输入内容或摘要的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `[{"dtype":"FP32","element_count":"128","generator":"modular_signal(seed=83010...` | NA |
| 15 | `case_parameters_json` | JSON text | NA | 样本未见空值 | 本case业务参数的JSON文本。 | 原始记录字段 | 按字段说明、行粒度、主键和相关列共同判读。 | `{"dtype":"FP32","h-count":"32","n-chans":"8","seed":"83010001","x-count":"128"}` | NA |
