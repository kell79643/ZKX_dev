# 阶段10正式稳定性1000次回收审计

## 结论

阶段10正式run `20260814T021000Z_8841247_target_semantic_1000` 的9004个原始文件已按“容器普通文件树→远程宿主机普通文件树→Windows归档普通文件树”逐项回收。容器、宿主机和Windows的相对路径、文件大小与SHA-256全部一致；1000个Task1/Task2配对case全部完成并通过。

正式证据和治理索引已完成。上一轮错误回收在归档根外遗留的压缩载体及Windows错误目录已取得用户明确授权并删除，容器、宿主机和Windows复扫均为0；批次最终合规关闭。`missing_artifacts.csv`保留闭环记录。

## 三端逐文件核验

```text
source_file_count = 9004
host_file_count = 9004
windows_file_count = 9004
sha256_match_count = 9004
sha256_mismatch_count = 0
source_total_bytes = 40247879
host_total_bytes = 40247879
windows_total_bytes = 40247879
archive_file_count = 0
```

`archive_file_count`覆盖本批次容器正式源目录、宿主机无压缩临时导出目录和Windows正式目标目录；错误流程产生且位于这些受控目录之外的历史压缩载体另行登记，未隐藏为0。

## 新增归档范围治理审计

以下数字覆盖阶段10正式证据树和对应回收批次记录目录；治理审计报告自身加入后，范围内文件总数为18,025。

```text
subdirectory_count = 3006
subdirectory_with_readme_count = 3006
subdirectory_missing_readme_count = 0
folder_manifest_covered_file_count = 18025
folder_manifest_uncovered_file_count = 0
readme_broken_reference_count = 0
csv_file_count = 5009
csv_with_column_document_count = 5009
csv_without_column_document_count = 0
csv_schema_count = 4
csv_header_document_mismatch_count = 0
csv_broken_schema_reference_count = 0
```

归档根`folder_manifest.csv`在本报告加入并完成祖先清单回刷后应覆盖43,349个文件；最终复核必须再次确认实际行数与该值一致。

## 证据边界

- 正式范围：1000个配对case，1000 PASS、0 FAIL；
- Task1和Task2共享每个case的`case_id`与`stability_case_index`；
- 每个Task结果CSV保留正常任务完整表头，包含Step与`pipeline_total`的耗时、精度、资源和状态；
- 实际扰动数值保存在每行`perturbation_summary`及唯一参数文件身份字段中，不只保存seed；
- 旧失败、诊断与候选run未混入本正式目录；
- 容器原件与宿主机普通文件临时树均未删除。
