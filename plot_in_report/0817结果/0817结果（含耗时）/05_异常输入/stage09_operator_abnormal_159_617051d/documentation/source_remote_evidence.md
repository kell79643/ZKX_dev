# 阶段09：53算子159条异常输入远程证据

## 环境与标准流程

本轮在本地`ZKX/`核对Git状态后，使用标准同步命令取得RC=0，输出同时包含：

```text
/home/usr_02/ZKX_dev
Sync done
```

随后精确执行：

```powershell
.\remote_scripts\remote-run.ps1 --no-sync "sudo docker cp /home/usr_02/ZKX_dev/. gpu_02:/tmp/ZKX_dev"
```

复制成功后，通过独占socket `/tmp/gpu02_stage09_196.sock`连接标准持久交互容器会话。环境为
ZQ500-Q、Driver `2.3.11`、CMake `3.22.1`，工作目录`/tmp/ZKX_dev`，SDK已初始化。

## 构建与清单门禁

通用目标构建成功：

```sh
cmake --build build/stage09_operator_abnormal_8d8 \
  --target stage09_operator_abnormal -j2
```

清单门禁：

```text
CATALOG_PASS operators=53 cases=159 semantics_per_operator=3 implemented=159 developer_pending=0
```

`resolve_normal_case_links.py`从既有正常PASS CSV生成53行真实外键：

```text
NORMAL_LINK_PASS operators=53 output=/tmp/ZKX_dev/results/stage09_normal_links_8d8.csv
```

## 逐算子执行

每个算子使用独立命令和输出目录，选择器固定为：

```text
--module <module> --operator <operator> --case-id all --backend all
```

每次执行三条不同失败语义并打印完整输入和错误简表，成功标志均为：

```text
[STAGE09][SUMMARY] total=3 pass=3 fail=0 status=PASS
```

代表CUDA Runtime后端失败输出：

```text
filtering/sosfilt                 backend_call_failure    CUDA_RUNTIME_ERROR       CUDA_RUNTIME_ERROR       CUDA_RUNTIME_ERROR       true       PASS
  case_id: filtering.sosfilt.backend_call_failure.inject_cuda_kernel_launch_failure_before_output_
  abnormal_input: mutation=inject CUDA kernel launch failure before output consumption
  error_message: filtering::sosfilt捕获CUDA Runtime失败：...；cudaErrorInvalidDevice（invalid device ordinal）
  backend_error: code=101 name=cudaErrorInvalidDevice message=invalid device ordinal
```

代表内部数值后端失败输出：

```text
filtering/lfilter_zi              backend_call_failure    BACKEND_ERROR             BACKEND_ERROR             BACKEND_ERROR             true       PASS
  abnormal_input: mutation=inject singular linear-system backend failure
  backend_error: code=SINGULAR name=LINEAR_SOLVER_SINGULAR message=controlled linear-solver failure injection: singular coefficient matrix
```

`lfilter_zi`首次运行把`not_applicable`误判为配置错误，结果为2/3；AI立即停止后续算子，修复并
只重跑该算子，`retry1`为3/3。首次失败文件未纳入正式合并。

FFT失败最初曾用零长度`cufftPlan1d`尝试真实失败，供应商dlfft内部断言并以RC=139终止，且未
生成CSV。随后改用公开`CUFFT_INVALID_SIZE`兼容状态受控注入，避免调用会崩溃的plan入口；
代表`convolution::correlate`复验为3/3 PASS、`safe_exit=true`。

## 合并与深检

53份通过的逐算子CSV由`merge_results.py`合并，输出：

```text
MERGE_PASS cases=159 files=cuda_runtime:32:.../operator_abnormal_cases_cuda_runtime.csv|not_applicable:113:.../operator_abnormal_cases_not_applicable.csv|selected_backend:14:.../operator_abnormal_cases_selected_backend.csv
```

最终深检逐行核对清单合同、53个正常外键、四个项目码字段、消息、后端三元组、
`safe_exit/output_consumed/status`及唯一case ID：

```text
RESULT_PASS operators=53 cases=159 pass=159 fail=0 safe_exit=159 output_consumed=0 normal_links=53
```

正式结果位于：

```text
/tmp/ZKX_dev/results/stage09_operator_abnormal_159_617051d/final_v1/
```

逐算子终端明细和原始CSV位于同一结果根目录的`raw/`下。

为避免证据只保留在容器临时目录，已将上述结果根目录和53行正常外键打包回收到远程宿主机：

```text
/home/usr_02/ZKX_dev/results/stage09_operator_abnormal_159_617051d.tar.gz
bytes=32015
sha256=d0ba30d6a0e3e599e6e64053f3d08dba2389f6b14f1c515be5b117111e820857
```

容器生成包与宿主机回收包的SHA-256一致。
