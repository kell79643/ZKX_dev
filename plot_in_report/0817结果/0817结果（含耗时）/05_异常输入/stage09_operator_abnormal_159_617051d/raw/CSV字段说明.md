# CSV字段说明

> 阅读顺序：先读同目录`README.md`了解文件身份和用途，再用本文件查CSV行粒度、主键、关联键和全部列含义。

## 覆盖结论

- 本目录直接CSV：1个（含治理文件`folder_manifest.csv`）
- 本目录直接CSV已映射：1个
- 本目录直接CSV未映射：0个
- 本目录不同schema：1种
- 生成与校验时间：`2026-08-14T17:33:09+08:00`
- 校验规则：真实列名、列数、顺序及规范化表头SHA-256必须与schema一致。

## 直接CSV到schema映射

| CSV文件 | schema_id | 字段说明 | 表头SHA-256 | 列数 | 一行代表 | 主键 | 配对键 | 来源/生成方式 | 校验状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| [`folder_manifest.csv`](./folder_manifest.csv) | `SCHEMA_7F4D2CBB2001` | [查看全部列](#schema_7f4d2cbb2001) | `7f4d2cbb20011c229afba04411d0db1e9aff71bad328b4fbfbb73d6fa572eaf8` | 32 | 一行对应当前目录递归范围内的一个文件。 | `relative_path` | `run_id + case_id` | 归档治理生成器 | HEADER_MATCH |

## 子目录CSV说明索引

| 子目录 | 递归CSV数 | 字段说明 | 已说明 | 未说明 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| [`bsplines_cubic`](./bsplines_cubic/README.md) | 2 | [CSV字段说明](./bsplines_cubic/CSV字段说明.md) | 2 | 0 | COVERED |
| [`bsplines_gauss_spline`](./bsplines_gauss_spline/README.md) | 2 | [CSV字段说明](./bsplines_gauss_spline/CSV字段说明.md) | 2 | 0 | COVERED |
| [`bsplines_quadratic`](./bsplines_quadratic/README.md) | 2 | [CSV字段说明](./bsplines_quadratic/CSV字段说明.md) | 2 | 0 | COVERED |
| [`convolution_correlate`](./convolution_correlate/README.md) | 2 | [CSV字段说明](./convolution_correlate/CSV字段说明.md) | 2 | 0 | COVERED |
| [`convolution_correlate2d`](./convolution_correlate2d/README.md) | 2 | [CSV字段说明](./convolution_correlate2d/CSV字段说明.md) | 2 | 0 | COVERED |
| [`demod_fm_demod`](./demod_fm_demod/README.md) | 2 | [CSV字段说明](./demod_fm_demod/CSV字段说明.md) | 2 | 0 | COVERED |
| [`estimation_kalman_filter`](./estimation_kalman_filter/README.md) | 2 | [CSV字段说明](./estimation_kalman_filter/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filter_design_firwin`](./filter_design_firwin/README.md) | 2 | [CSV字段说明](./filter_design_firwin/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filter_design_firwin2`](./filter_design_firwin2/README.md) | 2 | [CSV字段说明](./filter_design_firwin2/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_channelize_poly`](./filtering_channelize_poly/README.md) | 2 | [CSV字段说明](./filtering_channelize_poly/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_decimate`](./filtering_decimate/README.md) | 2 | [CSV字段说明](./filtering_decimate/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_detrend`](./filtering_detrend/README.md) | 2 | [CSV字段说明](./filtering_detrend/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_firfilter`](./filtering_firfilter/README.md) | 2 | [CSV字段说明](./filtering_firfilter/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_firfilter2`](./filtering_firfilter2/README.md) | 2 | [CSV字段说明](./filtering_firfilter2/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_freq_shift`](./filtering_freq_shift/README.md) | 2 | [CSV字段说明](./filtering_freq_shift/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_hilbert`](./filtering_hilbert/README.md) | 2 | [CSV字段说明](./filtering_hilbert/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_hilbert2`](./filtering_hilbert2/README.md) | 2 | [CSV字段说明](./filtering_hilbert2/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_lfilter_zi_retry1`](./filtering_lfilter_zi_retry1/README.md) | 2 | [CSV字段说明](./filtering_lfilter_zi_retry1/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_resample`](./filtering_resample/README.md) | 2 | [CSV字段说明](./filtering_resample/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_resample_poly`](./filtering_resample_poly/README.md) | 2 | [CSV字段说明](./filtering_resample_poly/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_sosfilt`](./filtering_sosfilt/README.md) | 2 | [CSV字段说明](./filtering_sosfilt/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_upfirdn`](./filtering_upfirdn/README.md) | 2 | [CSV字段说明](./filtering_upfirdn/CSV字段说明.md) | 2 | 0 | COVERED |
| [`filtering_wiener`](./filtering_wiener/README.md) | 2 | [CSV字段说明](./filtering_wiener/CSV字段说明.md) | 2 | 0 | COVERED |
| [`peak_finding_argrelextrema`](./peak_finding_argrelextrema/README.md) | 2 | [CSV字段说明](./peak_finding_argrelextrema/CSV字段说明.md) | 2 | 0 | COVERED |
| [`radartools_ambgfun`](./radartools_ambgfun/README.md) | 2 | [CSV字段说明](./radartools_ambgfun/CSV字段说明.md) | 2 | 0 | COVERED |
| [`radartools_ca_cfar`](./radartools_ca_cfar/README.md) | 2 | [CSV字段说明](./radartools_ca_cfar/CSV字段说明.md) | 2 | 0 | COVERED |
| [`radartools_cfar_alpha`](./radartools_cfar_alpha/README.md) | 2 | [CSV字段说明](./radartools_cfar_alpha/CSV字段说明.md) | 2 | 0 | COVERED |
| [`radartools_pulse_compression`](./radartools_pulse_compression/README.md) | 2 | [CSV字段说明](./radartools_pulse_compression/CSV字段说明.md) | 2 | 0 | COVERED |
| [`radartools_pulse_doppler`](./radartools_pulse_doppler/README.md) | 2 | [CSV字段说明](./radartools_pulse_doppler/CSV字段说明.md) | 2 | 0 | COVERED |
| [`spectral_analysis_csd`](./spectral_analysis_csd/README.md) | 2 | [CSV字段说明](./spectral_analysis_csd/CSV字段说明.md) | 2 | 0 | COVERED |
| [`spectral_analysis_istft`](./spectral_analysis_istft/README.md) | 2 | [CSV字段说明](./spectral_analysis_istft/CSV字段说明.md) | 2 | 0 | COVERED |
| [`spectral_analysis_lombscargle`](./spectral_analysis_lombscargle/README.md) | 2 | [CSV字段说明](./spectral_analysis_lombscargle/CSV字段说明.md) | 2 | 0 | COVERED |
| [`spectral_analysis_spectrogram`](./spectral_analysis_spectrogram/README.md) | 2 | [CSV字段说明](./spectral_analysis_spectrogram/CSV字段说明.md) | 2 | 0 | COVERED |
| [`spectral_analysis_stft`](./spectral_analysis_stft/README.md) | 2 | [CSV字段说明](./spectral_analysis_stft/CSV字段说明.md) | 2 | 0 | COVERED |
| [`spectral_analysis_vectorstrength`](./spectral_analysis_vectorstrength/README.md) | 2 | [CSV字段说明](./spectral_analysis_vectorstrength/CSV字段说明.md) | 2 | 0 | COVERED |
| [`waveforms_chirp`](./waveforms_chirp/README.md) | 2 | [CSV字段说明](./waveforms_chirp/CSV字段说明.md) | 2 | 0 | COVERED |
| [`waveforms_gausspulse`](./waveforms_gausspulse/README.md) | 2 | [CSV字段说明](./waveforms_gausspulse/CSV字段说明.md) | 2 | 0 | COVERED |
| [`waveforms_sawtooth`](./waveforms_sawtooth/README.md) | 2 | [CSV字段说明](./waveforms_sawtooth/CSV字段说明.md) | 2 | 0 | COVERED |
| [`waveforms_square`](./waveforms_square/README.md) | 2 | [CSV字段说明](./waveforms_square/CSV字段说明.md) | 2 | 0 | COVERED |
| [`waveforms_unit_impulse`](./waveforms_unit_impulse/README.md) | 2 | [CSV字段说明](./waveforms_unit_impulse/CSV字段说明.md) | 2 | 0 | COVERED |
| [`wavelets_cwt`](./wavelets_cwt/README.md) | 2 | [CSV字段说明](./wavelets_cwt/CSV字段说明.md) | 2 | 0 | COVERED |
| [`wavelets_morlet`](./wavelets_morlet/README.md) | 2 | [CSV字段说明](./wavelets_morlet/CSV字段说明.md) | 2 | 0 | COVERED |
| [`wavelets_morlet2`](./wavelets_morlet2/README.md) | 2 | [CSV字段说明](./wavelets_morlet2/CSV字段说明.md) | 2 | 0 | COVERED |
| [`wavelets_qmf`](./wavelets_qmf/README.md) | 2 | [CSV字段说明](./wavelets_qmf/CSV字段说明.md) | 2 | 0 | COVERED |
| [`wavelets_ricker`](./wavelets_ricker/README.md) | 2 | [CSV字段说明](./wavelets_ricker/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_chebwin`](./windows_chebwin/README.md) | 2 | [CSV字段说明](./windows_chebwin/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_general_cosine`](./windows_general_cosine/README.md) | 2 | [CSV字段说明](./windows_general_cosine/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_general_gaussian`](./windows_general_gaussian/README.md) | 2 | [CSV字段说明](./windows_general_gaussian/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_hamming`](./windows_hamming/README.md) | 2 | [CSV字段说明](./windows_hamming/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_kaiser`](./windows_kaiser/README.md) | 2 | [CSV字段说明](./windows_kaiser/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_parzen`](./windows_parzen/README.md) | 2 | [CSV字段说明](./windows_parzen/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_taylor`](./windows_taylor/README.md) | 2 | [CSV字段说明](./windows_taylor/CSV字段说明.md) | 2 | 0 | COVERED |
| [`windows_triang`](./windows_triang/README.md) | 2 | [CSV字段说明](./windows_triang/CSV字段说明.md) | 2 | 0 | COVERED |

## schema完整定义

<a id="schema_7f4d2cbb2001"></a>
### SCHEMA_7F4D2CBB2001

- 适用文件：`folder_manifest.csv`
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

## 快速导航

- [同目录README](./README.md)
- [同目录完整文件清单](./folder_manifest.csv)
- [归档根README](../../../README.md)
- [全局CSV字段语义总表](../../../00_归档规范/CSV字段语义总表.md)
