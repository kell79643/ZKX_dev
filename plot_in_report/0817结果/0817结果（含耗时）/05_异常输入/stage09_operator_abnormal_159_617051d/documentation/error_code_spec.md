# 统一错误码与异常输入规范

## 统一错误码

| code | 含义 | 是否可继续下游 |
|---|---|---|
| `SUCCESS(0)` | 成功 | 是 |
| `INVALID_ARGUMENT(1001)` | 业务参数或范围无效 | 否 |
| `INVALID_SHAPE(1002)` | 输入shape无效 | 否 |
| `INVALID_DTYPE(1003)` | 输入dtype无效 | 否 |
| `UNSUPPORTED_OPERATION(1004)` | 当前backend/dtype/API明确不支持 | 否；不得切换后端 |
| `CONFIG_ERROR(1005)` | JSON/Schema/交叉配置无效 | 否 |
| `CUDA_DRIVER_ERROR(2001)` | CUDA Driver API返回失败 | 否 |
| `CUDA_RUNTIME_ERROR(2002)` | CUDA Runtime API返回失败 | 否 |
| `OUT_OF_MEMORY(2003)` | Host/Device分配失败 | 否 |
| `BACKEND_ERROR(2099)` | 非CUDA后端返回非成功原始错误 | 否 |
| `EXECUTION_FAILED(2100)` | CPU/GPU业务执行失败 | 否 |
| `SYNC_FAILED(2101)` | GPU同步失败 | 否 |
| `ACCURACY_FAILED(3001)` | 执行完成但精度/语义门禁失败 | 否 |
| `RESOURCE_FAILED(3002)` | 内存/显存/泄漏门禁失败 | 否 |
| `IO_ERROR(4001)` | 配置/证据文件读写失败 | 否 |
| `INTERNAL_ERROR(9001)` | 未分类内部不变量失败 | 否 |
| `UPSTREAM_STEP_FAILED(9002)` | 冻结上游失败导致本Step未执行 | 否 |

优先返回最具体错误。Task链上游失败后，下游必须写`UPSTREAM_STEP_FAILED`或不产生业务行并由任务汇总明确标记，不能使用旧中间结果继续。

数值是稳定项目合同，不能随枚举声明顺序变化。阶段03/04名称`OK`、`INVALID_CONFIG`、
`UNSUPPORTED`、`ALLOCATION_FAILED`仅保留为源码兼容别名；新CSV、JSON和终端输出统一写规范名称。

## 后端原始错误三元组

每条失败固定保留：

- `backend_error_code`：后端原始整数/字符串代码；无后端错误写`NA`。
- `backend_error_name`：SDK定义的符号名；无法解析时写`UNKNOWN_BACKEND_ERROR`，不可留空。
- `backend_error_message`：原始或官方映射消息，不含密码和个人路径。

统一code与原始三元组同时存在：例如SDK分配错误映射为`ALLOCATION_FAILED`，三元组仍保存原值。不得把原始错误丢失为统一的“failed”。后续实现参考官方CUDA 11.7 Error Handling表，但阶段03不伪造ZQ500未实测枚举。

## 异常case规则

- Task1、Task2分别至少3类任务级异常，必须含上游Step失败/中间输入缺失。
- 每个正式算子至少3类适用异常；不适用类别须以设计理由排除，不能减少总覆盖。
- 每case记录`expected_code,actual_code,error_message,safe_exit`及后端三元组。
- PASS要求实际码等于期望码、消息非空、safe_exit=true且无结果冒充成功。
- `UNSUPPORTED`必须明确backend、dtype、接口和原因，禁止静默切换`dlfft/fft_thrust`。
