# 最终运行环境与指纹

## 运行绑定

| 项目 | 实际值 |
|---|---|
| 日期 | 2026-07-17 |
| 远程主机 | `usr_02@10.110.12.10` |
| 容器 | `gpu_02` |
| 容器源码目录 | `/tmp/ZKX_dev` |
| 平台 | 中科芯 ZQ500，GPU 正式路径使用平台 runtime/科学库 |
| SDK 初始化 | `source /zq500/sdk/env.sh` |
| 测试源提交 | `da63935183f1b4c2bdc0ecd45c42e1418b498689` |
| 最终标志 | `operators=7 operator_scenarios=20 task_chains=2 records=40 status=passed` |

构建、运行和日志读取均通过项目规定的同步、精确 `docker cp` 和持久交互容器入口完成。
原始日志只保留测试程序输出的 40 条 `ACCURACY`、2 条 `TASK` 和 1 条 `SUMMARY`。

## 工作树状态说明

运行时和归档时共享本地工作树为 dirty，且存在另一位开发者的并行修改。本任务没有清理、
覆盖或提交那些修改。高精度测试源码已经绑定上述提交，并用下列 SHA-256 冻结。因此本批
证据的准确表述是“测试源提交 + 文件指纹 + dirty 状态已披露”，不是 clean worktree。

## SHA-256

```text
6282e1221e581a38612a33dc4049e3c559c23c4390e087abafcbb24009f2bb27  test_all/operators/accuracy/oracle/sensitive_operator_oracle_tests.cu
2d51a5358fb513532e05ce45f78f2f7b13f0b1057cbd2ac0feaee5dace8f397a  test_all/operators/accuracy/oracle/radar_oracle.cpp
0bcca95aabdee853bf24d538d7b11b75c2d527d46cd126757fe9ea1e7eb11d3c  test_all/operators/accuracy/oracle/radar_oracle.h
37be761708e590da5852936c3802c2a14fc409c8273fa58b65c12a5599b5c3da  test_all/operators/accuracy/oracle/filtering_oracle.cpp
5d5b36245b8a3ed80b70647329317ab85590ad961db42abdc70b0109ee65720f  test_all/operators/accuracy/oracle/filtering_oracle.h
d175a7c0af4e57ef551aaf2d6ca62e2232216fcf306afd3633d02e4449eac438  test_all/operators/accuracy/oracle/convolution_oracle.cpp
e4a5b1120913d4643d10957ee50ed1c50ff5f19d3fce7d13e4b07b8bf377bc01  test_all/operators/accuracy/oracle/convolution_oracle.h
0f9acc4a13bcdfc92397a2d1bc9d6c76140ad28ba564eb18732f4422a6e68638  test_all/operators/accuracy/oracle/spectral_oracle.cpp
fca5fb6a38717e0bfcc8a4a83bd296c8be5b6c57e147664d2904b43634e999dd  test_all/operators/accuracy/oracle/spectral_oracle.h
0b26de9018ca3bad4cdc19e9b343c377642421a5d7c376ea269df12b50da0dc2  test_all/operators/accuracy/oracle/high_precision_common.h
7a315e7f8e0211876101e5cc935bc5aaf52ad7b27bb12ba56f8fb89d81570f68  raw_zq500_oracle.log
ee823c35eec0ccf6f3438ef96edaa1267b93a9cc3a4b1617a1b47570aa530b59  oracle_metrics.csv
352ae4c48ac643f57a8ee497c1f5219bbcfc862bae393a4ec134b47bbbc37cce  task_metrics.csv
```
