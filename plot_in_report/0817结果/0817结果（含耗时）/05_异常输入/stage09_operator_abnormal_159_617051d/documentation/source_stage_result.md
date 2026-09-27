# 阶段09：53算子异常输入与错误码（PASS）

## 结论

阶段09已在ZQ500 `gpu_02`完成53个GPU算子、每算子3类不同失败语义、共159条异常输入检测。
最终深检结果为：

```text
RESULT_PASS operators=53 cases=159 pass=159 fail=0 safe_exit=159 output_consumed=0 normal_links=53
```

异常测试全部位于`test_all/operators/abnormal/`，本轮对既有正常用例目录
`test_all/operators/cases/`零修改。每条结果均可按`module_name/operator_name`过滤，并保存
真实正常测试外键`normal_run_id/normal_case_id`。

## 源码错误合同

- 稳定项目错误码和统一异常结构位于`cusignal_cpp/src/common/errors.h`，不依赖某个test入口。
- 53算子注册及异常前置检测位于
  `cusignal_cpp/src/common/operator_abnormal_validation.h`；`test_all`只负责选择fixture、捕获、
  打印和落盘。
- 每条结果固定保存`expected_code/captured_code/returned_code/actual_code`、明确自然语言消息、
  `backend_error_code/backend_error_name/backend_error_message`、`safe_exit`和
  `output_consumed`。
- 输入校验失败的后端三元组为`NA/NA/NA`；后端失败必须有非空原始或受控等价状态三元组。
- 后端失败通过异常立即返回，执行动作未正常结束且`output_consumed=false`才可能判PASS。

## 159条语义分布

| 异常语义 | 项目错误码 | 数量 |
|---|---|---:|
| `invalid_shape` | `INVALID_SHAPE` | 53 |
| `invalid_dtype` | `INVALID_DTYPE` | 8 |
| `invalid_parameter` | `INVALID_ARGUMENT` | 47 |
| `unsupported_operation` | `UNSUPPORTED_OPERATION` | 7 |
| `backend_call_failure` | `CUDA_RUNTIME_ERROR/BACKEND_ERROR` | 44 |
| 合计 | — | 159 |

同一算子的三条清单行具有三种不同`abnormal_type`，静态门禁输出：

```text
CATALOG_PASS operators=53 cases=159 semantics_per_operator=3 implemented=159 developer_pending=0
```

## 后端失败处理

- 29条CUDA Runtime后端失败使用真实`cudaSetDevice(-1)`失败，并由
  `cudaGetErrorName/cudaGetErrorString`取得
  `101/cudaErrorInvalidDevice/invalid device ordinal`，映射为`CUDA_RUNTIME_ERROR`。
- FFT路径原尝试用零长度`cufftPlan1d`制造真实失败；ZQ500供应商dlfft在内部断言并以RC=139
  终止进程，不满足`safe_exit`。正式实现不再调用该危险入口，而注入其公开兼容状态
  `8/CUFFT_INVALID_SIZE/controlled dlfft-compatible status injection: invalid FFT size`，映射为
  `BACKEND_ERROR`，且不产生或消费plan输出。
- `lfilter_zi`内部线性求解失败没有外部后端查询API，使用稳定受控状态
  `SINGULAR/LINEAR_SOLVER_SINGULAR/controlled linear-solver failure injection: singular coefficient matrix`。
- 既有CUDA Driver代表路径仍使用真实`cuGetErrorName/cuGetErrorString`，结果为
  `101/CUDA_ERROR_INVALID_DEVICE/invalid device ordinal`。

## 终端与结果

通用runner `stage09_operator_abnormal`支持
`--module/--operator/--case-id/--backend`选择器。终端逐条打印：异常输入、case ID、期望码、
捕获码、返回码、自然语言消息、后端三元组、`safe_exit`和PASS/FAIL；每个算子的完整输出同时
写入独立`terminal.log`。

最终容器结果根目录：

```text
/tmp/ZKX_dev/results/stage09_operator_abnormal_159_617051d/
```

- `raw/<module>_<operator>/terminal.log`：53个算子的逐条终端明细；
- `raw/<module>_<operator>/operator_abnormal_cases_mixed.csv`：逐算子三条结果；
- `final_v1/operator_abnormal_cases_cuda_runtime.csv`：32条；
- `final_v1/operator_abnormal_cases_not_applicable.csv`：113条；
- `final_v1/operator_abnormal_cases_selected_backend.csv`：14条。

上述三份最终CSV合计159个唯一`case_id`。`lfilter_zi`首次失败目录未纳入合并，正式结果只采用
`filtering_lfilter_zi_retry1`。

完整结果已回收到远程宿主机
`/home/usr_02/ZKX_dev/results/stage09_operator_abnormal_159_617051d.tar.gz`，SHA-256为
`d0ba30d6a0e3e599e6e64053f3d08dba2389f6b14f1c515be5b117111e820857`。

## 提交

- `0636b60 feat(errors): 增加53算子异常检测统一runner`
- `a507321 test(errors): 补齐异常简表与正常结果外键解析`
- `7d40ec6 feat(errors): 落地159条算子异常检测合同`
- `8d8e3bb test(errors): 增加159条异常结果深检门禁`
- `196a456 fix(errors): 避免dlfft非法plan导致进程崩溃`
- `617051d test(errors): 增加159条异常结果按后端合并`
- `21bd96d fix(errors): 支持内部线性求解后端异常`
