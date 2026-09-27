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
| [`stage08_b01`](./stage08_b01/README.md) | 501 | [CSV字段说明](./stage08_b01/CSV字段说明.md) | 501 | 0 | COVERED |
| [`stage08_b01_chebwin_fft_thrust`](./stage08_b01_chebwin_fft_thrust/README.md) | 161 | [CSV字段说明](./stage08_b01_chebwin_fft_thrust/CSV字段说明.md) | 161 | 0 | COVERED |
| [`stage08_b02`](./stage08_b02/README.md) | 465 | [CSV字段说明](./stage08_b02/CSV字段说明.md) | 465 | 0 | COVERED |
| [`stage08_b02_strict_dtype`](./stage08_b02_strict_dtype/README.md) | 551 | [CSV字段说明](./stage08_b02_strict_dtype/CSV字段说明.md) | 551 | 0 | COVERED |
| [`stage08_b03_channelize_poly_0b2008e7f76a`](./stage08_b03_channelize_poly_0b2008e7f76a/README.md) | 94 | [CSV字段说明](./stage08_b03_channelize_poly_0b2008e7f76a/CSV字段说明.md) | 94 | 0 | COVERED |
| [`stage08_b03_channelize_poly_dev`](./stage08_b03_channelize_poly_dev/README.md) | 8 | [CSV字段说明](./stage08_b03_channelize_poly_dev/CSV字段说明.md) | 8 | 0 | COVERED |
| [`stage08_b03_decimate_02b9734201c9`](./stage08_b03_decimate_02b9734201c9/README.md) | 79 | [CSV字段说明](./stage08_b03_decimate_02b9734201c9/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b03_decimate_dev`](./stage08_b03_decimate_dev/README.md) | 7 | [CSV字段说明](./stage08_b03_decimate_dev/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b03_hilbert2_0b2008e7f76a`](./stage08_b03_hilbert2_0b2008e7f76a/README.md) | 85 | [CSV字段说明](./stage08_b03_hilbert2_0b2008e7f76a/CSV字段说明.md) | 85 | 0 | COVERED |
| [`stage08_b03_hilbert_0b2008e7f76a`](./stage08_b03_hilbert_0b2008e7f76a/README.md) | 79 | [CSV字段说明](./stage08_b03_hilbert_0b2008e7f76a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b03_hilbert_dev`](./stage08_b03_hilbert_dev/README.md) | 7 | [CSV字段说明](./stage08_b03_hilbert_dev/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b03_resample_35406beea576`](./stage08_b03_resample_35406beea576/README.md) | 79 | [CSV字段说明](./stage08_b03_resample_35406beea576/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b03_resample_dev`](./stage08_b03_resample_dev/README.md) | 7 | [CSV字段说明](./stage08_b03_resample_dev/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b04_correlate2d_2c86271438dd`](./stage08_b04_correlate2d_2c86271438dd/README.md) | 7 | [CSV字段说明](./stage08_b04_correlate2d_2c86271438dd/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b04_correlate2d_e458c69972d3`](./stage08_b04_correlate2d_e458c69972d3/README.md) | 79 | [CSV字段说明](./stage08_b04_correlate2d_e458c69972d3/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b04_firfilter2_be79ae78aeff`](./stage08_b04_firfilter2_be79ae78aeff/README.md) | 79 | [CSV字段说明](./stage08_b04_firfilter2_be79ae78aeff/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b04_firfilter2_cdf69665a35d`](./stage08_b04_firfilter2_cdf69665a35d/README.md) | 7 | [CSV字段说明](./stage08_b04_firfilter2_cdf69665a35d/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b04_firwin2_6e5af3b6102d`](./stage08_b04_firwin2_6e5af3b6102d/README.md) | 7 | [CSV字段说明](./stage08_b04_firwin2_6e5af3b6102d/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b04_firwin2_ba26c5229701`](./stage08_b04_firwin2_ba26c5229701/README.md) | 79 | [CSV字段说明](./stage08_b04_firwin2_ba26c5229701/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b04_resample_poly_6d43b8e8ac93`](./stage08_b04_resample_poly_6d43b8e8ac93/README.md) | 7 | [CSV字段说明](./stage08_b04_resample_poly_6d43b8e8ac93/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b04_resample_poly_a6a399dddc61`](./stage08_b04_resample_poly_a6a399dddc61/README.md) | 79 | [CSV字段说明](./stage08_b04_resample_poly_a6a399dddc61/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b04_upfirdn_1a48b0bcf8cd`](./stage08_b04_upfirdn_1a48b0bcf8cd/README.md) | 7 | [CSV字段说明](./stage08_b04_upfirdn_1a48b0bcf8cd/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b04_upfirdn_8c1bc3832603`](./stage08_b04_upfirdn_8c1bc3832603/README.md) | 79 | [CSV字段说明](./stage08_b04_upfirdn_8c1bc3832603/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b05_detrend_2140b2b84ec4`](./stage08_b05_detrend_2140b2b84ec4/README.md) | 79 | [CSV字段说明](./stage08_b05_detrend_2140b2b84ec4/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b05_detrend_9f9c471697c5`](./stage08_b05_detrend_9f9c471697c5/README.md) | 7 | [CSV字段说明](./stage08_b05_detrend_9f9c471697c5/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b05_detrend_c39404c04ba1`](./stage08_b05_detrend_c39404c04ba1/README.md) | 6 | [CSV字段说明](./stage08_b05_detrend_c39404c04ba1/CSV字段说明.md) | 6 | 0 | COVERED |
| [`stage08_b05_freq_shift_5aa8f9beb422`](./stage08_b05_freq_shift_5aa8f9beb422/README.md) | 7 | [CSV字段说明](./stage08_b05_freq_shift_5aa8f9beb422/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b05_freq_shift_99442e69c882`](./stage08_b05_freq_shift_99442e69c882/README.md) | 79 | [CSV字段说明](./stage08_b05_freq_shift_99442e69c882/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b05_lfilter_zi_01967914167d`](./stage08_b05_lfilter_zi_01967914167d/README.md) | 7 | [CSV字段说明](./stage08_b05_lfilter_zi_01967914167d/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b05_lfilter_zi_bb1a0e0e2788`](./stage08_b05_lfilter_zi_bb1a0e0e2788/README.md) | 79 | [CSV字段说明](./stage08_b05_lfilter_zi_bb1a0e0e2788/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b05_sosfilt_019b253f23f3`](./stage08_b05_sosfilt_019b253f23f3/README.md) | 7 | [CSV字段说明](./stage08_b05_sosfilt_019b253f23f3/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b05_sosfilt_1681b603eadb`](./stage08_b05_sosfilt_1681b603eadb/README.md) | 79 | [CSV字段说明](./stage08_b05_sosfilt_1681b603eadb/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b05_sosfilt_b12500640dda`](./stage08_b05_sosfilt_b12500640dda/README.md) | 7 | [CSV字段说明](./stage08_b05_sosfilt_b12500640dda/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b05_wiener_c7605a98cd9f`](./stage08_b05_wiener_c7605a98cd9f/README.md) | 7 | [CSV字段说明](./stage08_b05_wiener_c7605a98cd9f/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b05_wiener_fd04f1143cb8`](./stage08_b05_wiener_fd04f1143cb8/README.md) | 79 | [CSV字段说明](./stage08_b05_wiener_fd04f1143cb8/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b06_csd_336d860e7e54`](./stage08_b06_csd_336d860e7e54/README.md) | 79 | [CSV字段说明](./stage08_b06_csd_336d860e7e54/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b06_csd_94da2510d955`](./stage08_b06_csd_94da2510d955/README.md) | 7 | [CSV字段说明](./stage08_b06_csd_94da2510d955/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_b06_istft_365e0a0ca281`](./stage08_b06_istft_365e0a0ca281/README.md) | 85 | [CSV字段说明](./stage08_b06_istft_365e0a0ca281/CSV字段说明.md) | 85 | 0 | COVERED |
| [`stage08_b06_lombscargle_ab7484afd8df`](./stage08_b06_lombscargle_ab7484afd8df/README.md) | 85 | [CSV字段说明](./stage08_b06_lombscargle_ab7484afd8df/CSV字段说明.md) | 85 | 0 | COVERED |
| [`stage08_b06_stft_81f248cc9e2d`](./stage08_b06_stft_81f248cc9e2d/README.md) | 85 | [CSV字段说明](./stage08_b06_stft_81f248cc9e2d/CSV字段说明.md) | 85 | 0 | COVERED |
| [`stage08_b06_vectorstrength_89aacbf48da8`](./stage08_b06_vectorstrength_89aacbf48da8/README.md) | 85 | [CSV字段说明](./stage08_b06_vectorstrength_89aacbf48da8/CSV字段说明.md) | 85 | 0 | COVERED |
| [`stage08_b07_morlet2_90c7905b71ff`](./stage08_b07_morlet2_90c7905b71ff/README.md) | 79 | [CSV字段说明](./stage08_b07_morlet2_90c7905b71ff/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b07_morlet_167aadba1745`](./stage08_b07_morlet_167aadba1745/README.md) | 85 | [CSV字段说明](./stage08_b07_morlet_167aadba1745/CSV字段说明.md) | 85 | 0 | COVERED |
| [`stage08_b07_qmf_4948b3e7dd01`](./stage08_b07_qmf_4948b3e7dd01/README.md) | 79 | [CSV字段说明](./stage08_b07_qmf_4948b3e7dd01/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b08_gauss_spline_a1467e3b776a`](./stage08_b08_gauss_spline_a1467e3b776a/README.md) | 79 | [CSV字段说明](./stage08_b08_gauss_spline_a1467e3b776a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b08_quadratic_a1467e3b776a`](./stage08_b08_quadratic_a1467e3b776a/README.md) | 79 | [CSV字段说明](./stage08_b08_quadratic_a1467e3b776a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b08_rerun_gauss_spline_a1467e3b776a`](./stage08_b08_rerun_gauss_spline_a1467e3b776a/README.md) | 79 | [CSV字段说明](./stage08_b08_rerun_gauss_spline_a1467e3b776a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b08_rerun_quadratic_a1467e3b776a`](./stage08_b08_rerun_quadratic_a1467e3b776a/README.md) | 79 | [CSV字段说明](./stage08_b08_rerun_quadratic_a1467e3b776a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b08_rerun_unit_impulse_a1467e3b776a`](./stage08_b08_rerun_unit_impulse_a1467e3b776a/README.md) | 79 | [CSV字段说明](./stage08_b08_rerun_unit_impulse_a1467e3b776a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_b08_unit_impulse_a1467e3b776a`](./stage08_b08_unit_impulse_a1467e3b776a/README.md) | 79 | [CSV字段说明](./stage08_b08_unit_impulse_a1467e3b776a/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_csvread_windows_smoke_47ebe13410d3`](./stage08_csvread_windows_smoke_47ebe13410d3/README.md) | 8 | [CSV字段说明](./stage08_csvread_windows_smoke_47ebe13410d3/CSV字段说明.md) | 8 | 0 | COVERED |
| [`stage08_input_preview_b01_chebwin_8b745f9a6864`](./stage08_input_preview_b01_chebwin_8b745f9a6864/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b01_chebwin_8b745f9a6864/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b01_general_cosine_8b745f9a6864`](./stage08_input_preview_b01_general_cosine_8b745f9a6864/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b01_general_cosine_8b745f9a6864/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b01_general_gaussian_8b745f9a6864`](./stage08_input_preview_b01_general_gaussian_8b745f9a6864/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b01_general_gaussian_8b745f9a6864/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b01_kaiser_8b745f9a6864`](./stage08_input_preview_b01_kaiser_8b745f9a6864/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b01_kaiser_8b745f9a6864/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b02_parzen_a64a65577ccc`](./stage08_input_preview_b02_parzen_a64a65577ccc/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b02_parzen_a64a65577ccc/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b02_taylor_a64a65577ccc`](./stage08_input_preview_b02_taylor_a64a65577ccc/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b02_taylor_a64a65577ccc/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b02_triang_a64a65577ccc`](./stage08_input_preview_b02_triang_a64a65577ccc/README.md) | 214 | [CSV字段说明](./stage08_input_preview_b02_triang_a64a65577ccc/CSV字段说明.md) | 214 | 0 | COVERED |
| [`stage08_input_preview_b03_channelize_poly_5bf09d215ce4`](./stage08_input_preview_b03_channelize_poly_5bf09d215ce4/README.md) | 94 | [CSV字段说明](./stage08_input_preview_b03_channelize_poly_5bf09d215ce4/CSV字段说明.md) | 94 | 0 | COVERED |
| [`stage08_input_preview_b03_decimate_5bf09d215ce4`](./stage08_input_preview_b03_decimate_5bf09d215ce4/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b03_decimate_5bf09d215ce4/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b03_hilbert2_5bf09d215ce4`](./stage08_input_preview_b03_hilbert2_5bf09d215ce4/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b03_hilbert2_5bf09d215ce4/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b03_hilbert_5bf09d215ce4`](./stage08_input_preview_b03_hilbert_5bf09d215ce4/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b03_hilbert_5bf09d215ce4/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b03_resample_5bf09d215ce4`](./stage08_input_preview_b03_resample_5bf09d215ce4/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b03_resample_5bf09d215ce4/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b04_correlate2d_db927ed96663`](./stage08_input_preview_b04_correlate2d_db927ed96663/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b04_correlate2d_db927ed96663/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b04_firfilter2_db927ed96663`](./stage08_input_preview_b04_firfilter2_db927ed96663/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b04_firfilter2_db927ed96663/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b04_firwin2_db927ed96663`](./stage08_input_preview_b04_firwin2_db927ed96663/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b04_firwin2_db927ed96663/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b04_resample_poly_db927ed96663`](./stage08_input_preview_b04_resample_poly_db927ed96663/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b04_resample_poly_db927ed96663/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b04_upfirdn_db927ed96663`](./stage08_input_preview_b04_upfirdn_db927ed96663/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b04_upfirdn_db927ed96663/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b05_detrend_652061b694fc`](./stage08_input_preview_b05_detrend_652061b694fc/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b05_detrend_652061b694fc/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b05_freq_shift_652061b694fc`](./stage08_input_preview_b05_freq_shift_652061b694fc/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b05_freq_shift_652061b694fc/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b05_lfilter_zi_652061b694fc`](./stage08_input_preview_b05_lfilter_zi_652061b694fc/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b05_lfilter_zi_652061b694fc/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b05_smoke_20260810T0845Z`](./stage08_input_preview_b05_smoke_20260810T0845Z/README.md) | 7 | [CSV字段说明](./stage08_input_preview_b05_smoke_20260810T0845Z/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_input_preview_b05_sosfilt_652061b694fc`](./stage08_input_preview_b05_sosfilt_652061b694fc/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b05_sosfilt_652061b694fc/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b05_wiener_652061b694fc`](./stage08_input_preview_b05_wiener_652061b694fc/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b05_wiener_652061b694fc/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b06_csd_9b01dadd36f6`](./stage08_input_preview_b06_csd_9b01dadd36f6/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b06_csd_9b01dadd36f6/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b06_istft_9b01dadd36f6`](./stage08_input_preview_b06_istft_9b01dadd36f6/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b06_istft_9b01dadd36f6/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b06_lombscargle_9b01dadd36f6`](./stage08_input_preview_b06_lombscargle_9b01dadd36f6/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b06_lombscargle_9b01dadd36f6/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b06_smoke_20260810T0900Z`](./stage08_input_preview_b06_smoke_20260810T0900Z/README.md) | 7 | [CSV字段说明](./stage08_input_preview_b06_smoke_20260810T0900Z/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_input_preview_b06_stft_9b01dadd36f6`](./stage08_input_preview_b06_stft_9b01dadd36f6/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b06_stft_9b01dadd36f6/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b06_vectorstrength_9b01dadd36f6`](./stage08_input_preview_b06_vectorstrength_9b01dadd36f6/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b06_vectorstrength_9b01dadd36f6/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b07_morlet2_ed9b9505e41d`](./stage08_input_preview_b07_morlet2_ed9b9505e41d/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b07_morlet2_ed9b9505e41d/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b07_morlet_ed9b9505e41d`](./stage08_input_preview_b07_morlet_ed9b9505e41d/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b07_morlet_ed9b9505e41d/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b07_qmf_ed9b9505e41d`](./stage08_input_preview_b07_qmf_ed9b9505e41d/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b07_qmf_ed9b9505e41d/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b07_smoke_20260810T0920Z`](./stage08_input_preview_b07_smoke_20260810T0920Z/README.md) | 7 | [CSV字段说明](./stage08_input_preview_b07_smoke_20260810T0920Z/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_input_preview_b08_gauss_spline_bdef73fce32b`](./stage08_input_preview_b08_gauss_spline_bdef73fce32b/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b08_gauss_spline_bdef73fce32b/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b08_quadratic_bdef73fce32b`](./stage08_input_preview_b08_quadratic_bdef73fce32b/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b08_quadratic_bdef73fce32b/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_b08_smoke_20260810T0935Z`](./stage08_input_preview_b08_smoke_20260810T0935Z/README.md) | 7 | [CSV字段说明](./stage08_input_preview_b08_smoke_20260810T0935Z/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_input_preview_b08_unit_impulse_bdef73fce32b`](./stage08_input_preview_b08_unit_impulse_bdef73fce32b/README.md) | 79 | [CSV字段说明](./stage08_input_preview_b08_unit_impulse_bdef73fce32b/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_input_preview_probe_channelize_poly_47ebe13410d3`](./stage08_input_preview_probe_channelize_poly_47ebe13410d3/README.md) | 8 | [CSV字段说明](./stage08_input_preview_probe_channelize_poly_47ebe13410d3/CSV字段说明.md) | 8 | 0 | COVERED |
| [`stage08_input_preview_probe_correlate2d_6767e484d941`](./stage08_input_preview_probe_correlate2d_6767e484d941/README.md) | 7 | [CSV字段说明](./stage08_input_preview_probe_correlate2d_6767e484d941/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_input_preview_probe_general_cosine_eaf30f1cad58`](./stage08_input_preview_probe_general_cosine_eaf30f1cad58/README.md) | 9 | [CSV字段说明](./stage08_input_preview_probe_general_cosine_eaf30f1cad58/CSV字段说明.md) | 9 | 0 | COVERED |
| [`stage08_table_rerun_channelize_poly_47ebe13410d3`](./stage08_table_rerun_channelize_poly_47ebe13410d3/README.md) | 94 | [CSV字段说明](./stage08_table_rerun_channelize_poly_47ebe13410d3/CSV字段说明.md) | 94 | 0 | COVERED |
| [`stage08_table_rerun_chebwin_47ebe13410d3`](./stage08_table_rerun_chebwin_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_chebwin_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_table_rerun_general_cosine_47ebe13410d3`](./stage08_table_rerun_general_cosine_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_general_cosine_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_table_rerun_general_gaussian_47ebe13410d3`](./stage08_table_rerun_general_gaussian_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_general_gaussian_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_table_rerun_hilbert2_47ebe13410d3`](./stage08_table_rerun_hilbert2_47ebe13410d3/README.md) | 79 | [CSV字段说明](./stage08_table_rerun_hilbert2_47ebe13410d3/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_table_rerun_hilbert_47ebe13410d3`](./stage08_table_rerun_hilbert_47ebe13410d3/README.md) | 79 | [CSV字段说明](./stage08_table_rerun_hilbert_47ebe13410d3/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_table_rerun_kaiser_47ebe13410d3`](./stage08_table_rerun_kaiser_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_kaiser_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_table_rerun_parzen_47ebe13410d3`](./stage08_table_rerun_parzen_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_parzen_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_table_rerun_resample_47ebe13410d3`](./stage08_table_rerun_resample_47ebe13410d3/README.md) | 79 | [CSV字段说明](./stage08_table_rerun_resample_47ebe13410d3/CSV字段说明.md) | 79 | 0 | COVERED |
| [`stage08_table_rerun_taylor_47ebe13410d3`](./stage08_table_rerun_taylor_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_taylor_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_table_rerun_triang_47ebe13410d3`](./stage08_table_rerun_triang_47ebe13410d3/README.md) | 184 | [CSV字段说明](./stage08_table_rerun_triang_47ebe13410d3/CSV字段说明.md) | 184 | 0 | COVERED |
| [`stage08_terminal_table_smoke_54336d803079`](./stage08_terminal_table_smoke_54336d803079/README.md) | 7 | [CSV字段说明](./stage08_terminal_table_smoke_54336d803079/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_terminal_table_smoke_6a39977a8c1f`](./stage08_terminal_table_smoke_6a39977a8c1f/README.md) | 7 | [CSV字段说明](./stage08_terminal_table_smoke_6a39977a8c1f/CSV字段说明.md) | 7 | 0 | COVERED |
| [`stage08_terminal_windows_smoke_6a39977a8c1f`](./stage08_terminal_windows_smoke_6a39977a8c1f/README.md) | 8 | [CSV字段说明](./stage08_terminal_windows_smoke_6a39977a8c1f/CSV字段说明.md) | 8 | 0 | COVERED |

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
- [归档根README](../../../../../README.md)
- [全局CSV字段语义总表](../../../../../00_归档规范/CSV字段语义总表.md)
