# Task1 复赛 ZQ500 技术结果与证据说明

## 1. 文档定位和结论边界

本文是任务一“雷达目标检测与多普勒分析”的复赛技术文档数据源，内容按
`LATEST_SOURCE_PLATFORMIZATION_REQUIREMENTS.md`、`TARGET_EFFECT_ZQ500_PLATFORMIZATION.md`
和 `TASK_K_DELIVERY_EVIDENCE.md` 要求整理。所有当前数值均来自 ZQ500 `gpu_02`
容器的本次实测，不使用旧 NVIDIA 结果冒充当前证据。

任务一 H 门禁结论为 `done`：五个正式 step、五个正式业务算子、五步自编计时、
五步 CPU/GPU 精度对照和任务语义指标均通过，CTest `6/6`。这个结论只针对
Task1；整个复赛交付仍须等 E/F/I/G/J/K 各自门禁完成，不在本文扩大声明为
“整个项目已可交付”。

## 2. 被测版本、环境和证据身份

| 字段 | 当前值 | 来源 |
|---|---|---|
| 被测本地 commit | `0372dcd5f8f512a6c2582c7fbf6b37fd90f41791` | CMake 证据绑定字段与 pipeline JSON |
| 本地工作树 | `git_dirty=true` | 验收时共享工作树存在 Task2/文档无关改动；Task1 被测源由 SHA-256 清单另行绑定 |
| 远程同步树 | `remote_sync_tree_dirty=false` | `logs/environment.txt` |
| 远程主机 | `usr_02@10.110.12.10` | pipeline JSON |
| 容器/主机名 | `gpu_02` / `supervisor-X7850H0` | pipeline JSON、`logs/environment.txt` |
| SDK | `/zq500/sdk` | pipeline JSON、验收命令 |
| CMake | `3.22.1` | `logs/environment.txt` |
| dlcc/Clang | `15.0.6` | `logs/environment.txt`；实际 CUDA 编译器为 SDK `dlcc` |
| 测试时间 | `2026-07-15T19:36:00+08:00` | pipeline JSON、CMake 编译宏 |
| 证据包 SHA-256 | `62a9979a695a4adc5269ba7bd198be484d42b276133a4f5a4689f7733ec9bac6` | 远程宿主机与本地下载后双重 `sha256sum/Get-FileHash` |

`remote-run.ps1` 不同步 `.git/`，所以容器中旧 HEAD 不被冒充为被测提交。证据同时
保存调用方 commit/dirty、实际编译输入 SHA-256、编译命令和远程状态，用于共享工作树下的
可追溯性。源码清单不把测试后回填的 README/证据说明作为编译输入；最终数据以绑定
`0372dcd` 的 JSON、日志和源哈希为准。

对 `logs/source-sha256.txt` 的 51 个实际编译输入逐项与当前本地文件复核，原始字节哈希
`51/51` 一致；不需要依赖 CRLF/LF 归一化解释，代码、参数和 CMake 均与被测输入相同。

## 3. 目录治理与总体架构

唯一活动实现为 `Task1/task1_gpu_cpu`，直接在初赛 Task1 源码上完成 ZQ500 复赛迁移，
没有再建一套复赛副本。旧 `Task1/task1_cpu`、`Task1/Task1_targetwave_form`、
NVIDIA/cuFFT/cuRAND/FP64 演示、私有 CFAR/ambgfun wrapper 和失效入口已删除；
`verify_task1_source_contract.cmake` 在配置时禁止这些路径回归。

| 模块 | 位置 | 职责 |
|---|---|---|
| 共享状态/计时 | `task1_common.h/.cu` | `PipelineState`、CudaEvent 和 step 证据结构 |
| 正式 GPU Step | `stepN/stepN.cu` | 只运行正式 GPU 链路、生成输出和统计正式执行时间 |
| Validation | `accuracy/validation/stepN_validation.cu/.h` | 收集 GPU 输出，独立生成 Host/typed CPU reference；不算误差、不判定通过 |
| Comparison | `accuracy/comparison/stepN_comparison.cpp/.h` | 纯 Host C++；以相对误差判定 GPU/CPU 数值 pass/fail，同时保留绝对误差、失败位置和语义判据 |
| 精度公共层 | `accuracy/accuracy_types.h`、`accuracy_utils.*` | 共享 validation 数据、`AccuracyEvidence` 和全元素比较工具 |
| 编排/结果 | `task1_runner.h/.cu` | 五步连续调用，打印并写入 JSON |
| 入口/测试 | `CMakeLists.txt`、`main.cu`、`stepN/main.cu` | 5 个 step 入口+1 个 pipeline，CTest 串行取证 |

数据连续性由同一 `PipelineState` 强制：step2 只读 step1 的 `echo/waveform`，step3
只读 step2 的 `compressed`，step4 只读 step3 的 `range_doppler`，step5 复用
step1 的同一 `waveform`。单步入口也从 step1 连续运行到自身，不接受与前步无关的
演示输入。JSON 中四个 `stepN_to_stepM_data_match=1` 证明实际交接通过。

调用顺序固定为 `run_stepN → collect_stepN_validation → compare_stepN_accuracy → 合并证据`。
各层通过内存结构传递，不写中间数据文件。配置期门禁逐文件扫描，实测
`accuracy_validation_files=5`、`accuracy_comparison_files=5`、
`comparison_gpu_paths=0`；CMake 显式登记全部源文件，并把 comparison 编译为 Host C++。

## 4. 固定场景和数据契约

| 项目 | 值 |
|---|---:|
| 业务输入/主要中间量 | `ComplexFP32` |
| 回波 shape/布局 | `[32,256]`，row-major，脉冲×快时间 |
| 采样率 | `2,560,000 Hz` |
| 脉冲数/PRF/PRI | `32` / `8,000 Hz` / `125 us` |
| LFM 脉冲点数/脉宽/带宽 | `64` / `25 us` / `640 kHz` |
| 目标延迟 | `80` samples，对应 `4,684.257156 m` |
| 目标多普勒 | `1,250 Hz`，对应未移位 bin `5` |
| 目标幅度 | `1.0` |
| 复噪声标准差/seed | `0.035` / `20260715` |
| 实测噪声 RMS/SNR | `0.0286246` / `24.8446 dB` |

主算法 GPU 存储使用 `DeviceArray`，步骤间为 Host `PipelineState` 并显式 H2D/D2H。
`cfar_alpha_device<float>` 的标量输出与 ambgfun CPU 参考容器按正式接口可使用
`double` 存储，但 Task1 业务输入和 GPU 主计算路径均为 FP32/ComplexFP32；源码门禁禁止
`cuDoubleComplex`，不存在 FP64 GPU 业务主线。

## 5. 选用的正式算子及边界

Task1 选用 5 个不重复的正式业务算子，均来自当前固定 53 算子集：

| step | 算子 | 实际 GPU API | CPU reference | 选择理由/业务作用 |
|---|---|---|---|---|
| 2 | `pulse_compression` | `pulse_compression_device<float>` | `pulse_compression_typed_cpu<float>` | 官方指定的匹配滤波/距离压缩 |
| 3 | `pulse_doppler` | `pulse_doppler_device<float>` | `pulse_doppler_typed_cpu<float>` | 官方指定的慢时间 Doppler FFT |
| 4 | `cfar_alpha` | `cfar_alpha_device<float>` | `cfar_alpha_typed_cpu<float>` | 按 PFA 和参考单元数独立计算阈值因子 |
| 4 | `ca_cfar` | `ca_cfar_device<float>` | `ca_cfar_typed_cpu<float>` | 二维 CA-CFAR 阈值和检测 mask |
| 5 | `ambgfun` | `ambgfun_device<float>` | `ambgfun_typed_cpu<float>` | 二维模糊函数与两种 cut，评估波形主瓣/旁瓣 |

`gpu_lfm_generation`、`gpu_delay_doppler_noise` 和 `range_doppler_power` 是任务建模/数据
glue 的私有 GPU helper，有独立 CudaEvent 时间和 CPU 对照，但不统计为新的正式业务
算子，也不用它们替代任一官方指定函数。

## 6. CUDA API、科学库和 ZQ500 CMake 路径

- 内存与传输通过 `DeviceArray` 统一进入 `cudaMalloc/cudaMemcpy/cudaFree`语义；
- kernel 由项目 `launch_1d_kernel` 进入 SDK CUDA runtime，同步与计时使用
  `cudaEventCreate/Record/Synchronize/ElapsedTime/Destroy`；
- `pulse_compression`、`pulse_doppler`、`ambgfun` 经 `FFTInterface` 进入 ZQ500 `dlfft`，
  FFT plan/workspace 在当前 convenience API 调用内创建；
- `cfar_alpha`、`ca_cfar`、step1 建模和功率转换为 custom kernel，不声称使用了未实际
  调用的科学库；
- 噪声由固定 seed 的可复现 hash 公式在 GPU/CPU 分别实现，没有使用 ZQ500 受限
  RAND 类型，也没有残留 cuRAND 依赖；
- CMake 从 `cusignal_cpp` 根目录以 `BUILD_TASK1=ON` 登记，使用项目
  `configure_zq500_target`、`signal_lib` 和 `fft_core`，配置输出确认 `DLI_V2`；
- 本轮配置契约实测为算子 `53/53 approved`、dtype `265/265 approved`、
  Task1 `official_steps=5`、`formal_apis=5`、`legacy_paths=0`、`nvidia_only_path=0`。

以上“使用了什么”由 `radartools_typed.cu`、`DeviceArray`、根 CMake、Task1 CMake、
`compile_commands.json` 和清理构建日志共同证明，不是仅依据文档推断。

## 7. 自编计时方法和口径

`task1_common.h::time_gpu` 在默认 stream 上记录开始/结束 CudaEvent，等待结束 event 并用
`cudaEventElapsedTime` 得到 `operators_ms`。这是 Task1 自编计时，不依赖 diPTI。

每个 step 另用 `std::chrono::steady_clock` 记录：

- `prepare`：Host 参数/shape 准备；
- `h2d`：函数外显式 Host-to-Device；
- `compute`：本 step 的 CudaEvent 调用时间之和；
- `d2h`：显式 Device-to-Host；
- `formal_execution`：进入 `run_stepN` 到正式输出 D2H 完成，不包含 accuracy 层；
- `postprocess`：validation、comparison、语义指标和负向检查；
- `total`：进入 step 到全部自编验证结束，保留拆分前的完整墙钟口径。

FFT plan/workspace 在正式公开调用内创建，因而计入 `compute/operators_ms`。本表是固定
场景的单次业务验收运行；`formal_execution` 可表达不含 CPU reference 的正式链路，
`total` 包含完整验证。两者都包含本轮初次 plan/workspace 成本，不是 warmup+多轮统计
benchmark，不能直接解释为纯 kernel 稳态吞吐。

## 8. 当前 ZQ500 分步耗时

以最终 pipeline 同一轮 JSON 为数值源，单位均为 ms。JSON 使用 17 位有效数字，避免
seed、细小误差和时间被默认 6 位有效数字舍入；控制台日志仍以 6 位小数便于阅读。

| step | prepare | H2D | compute | D2H | formal execution | postprocess | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 回波模拟 | 0.000030 | 0 | 3.742130 | 5.693819 | 112.322282 | 6.742875 | 119.263870 |
| 2 脉冲压缩 | 0.000600 | 0.129043 | 6906.898438 | 0.281124 | 6907.639273 | 33.307322 | 6941.065607 |
| 3 多普勒处理 | 0.000250 | 0.092841 | 3186.307617 | 0.431027 | 3187.014949 | 36.013724 | 3223.107464 |
| 4 CA-CFAR | 0.000760 | 0.075012 | 0.486020 | 0.283134 | 1.023036 | 3.557835 | 4.858296 |
| 5 模糊函数 | 0.000380 | 0.054881 | 8068.766113 | 0.427227 | 8069.767695 | 60.209693 | 8130.205761 |

`pipeline_total_ms=18418.539927999998`，五个 step `total` 之和为
`18418.500998 ms`，两者之差是
编排/记录开销。关键 CudaEvent 调用分解为：

| step | 调用 | GPU ms |
|---|---|---:|
| 1 | `gpu_lfm_generation` / `gpu_delay_doppler_noise` | 3.638972 / 0.103158 |
| 2 | `pulse_compression` | 6906.898438 |
| 3 | `pulse_doppler` | 3186.307617 |
| 4 | `range_doppler_power` / `cfar_alpha` / `ca_cfar` | 0.073278 / 0.237926 / 0.174816 |
| 5 | `ambgfun_2d` / `ambgfun_delay_cut` / `ambgfun_doppler_cut` | 3217.892090 / 1564.330566 / 3286.543457 |

Step2、3、5 是当前主要耗时来源，与其 dlFFT plan/workspace 在便捷接口内重复建立有关。

## 9. 误差的对比对象、方法和结果

每一种数值输出先计算全元素最大绝对误差，再以 CPU reference 的 L∞ 范数归一化得到
相对误差；pass/fail 只使用相对误差，绝对误差作为量级证据继续保留：

```text
max_abs_error = max_i |GPU_output[i] - CPU_reference[i]|
max_rel_error = max_abs_error / max(max_i |CPU_reference[i]|, relative_floor)
pass = (max_rel_error <= relative_tolerance)
```

其中 `relative_floor=1e-12`，只用于 CPU reference 全零时避免除零。复数使用复差模；
复合 step 对各输出分别计算相对误差和绝对误差，再分别取最大值。先校验 shape/元素数，
再比较全部元素，不抽样。误差不是“GPU 输出自己和自己比”，而是如下独立路径：

### 9.1 公式的源码实现和判定位置

上述公式不是只写在文档中：

| 内容 | 源码位置 | 实际职责 |
|---|---|---|
| `max_abs_error` | `Task1/task1_gpu_cpu/accuracy/accuracy_utils.cpp` | 对复数、FP32、FP64 容器计算全元素最大绝对误差 |
| `max_rel_error` | 同文件 `maximum_relative_difference` / `max_relative_error` | 先计算 CPU reference 的 L∞ 范数，再执行 `max_abs_error / max(reference_scale, relative_floor)` |
| 首个失败位置 | 同文件 `first_relative_failure_index` | 使用同一个相对误差分母和相对容差查找失败元素 |
| pass/fail | `accuracy/comparison/step1_comparison.cpp` 至 `step5_comparison.cpp` | 五步均执行 `require(evidence.max_rel_error <= evidence.relative_tolerance, ...)` |
| JSON 输出 | `Task1/task1_gpu_cpu/task1_runner.cu` | 同时写出 `max_abs_error`、`max_rel_error`、`relative_tolerance` 和 `relative_floor` |

因此，最大绝对误差只作为保留数值；数值精度通过与否由最大相对误差决定。

### 9.2 CPU reference 是否调用算子库

| step | CUDA 被测路径 | CPU reference 调用点 | CPU 算子库声明/实现 | 结论 |
|---|---|---|---|---|
| 1 | `gpu_lfm_generation`、`gpu_delay_doppler_noise` | `accuracy/validation/step1_validation.cu` 的 `waveform_cpu()`、`echo_cpu()` | 不调用正式 radartools 算子；按同一场景参数独立实现 Host LFM、延迟、多普勒和噪声公式 | 对应的是 Task1 私有建模 helper，独立公式比复用 GPU/helper 更适合作为 reference |
| 2 | `pulse_compression_device<float>` | `accuracy/validation/step2_validation.cu` 调用 `pulse_compression_typed_cpu<float>` | `cusignal_cpp/src/radartools/radartools_typed.h/.cpp` | 已调用算子库 CPU 实现 |
| 3 | `pulse_doppler_device<float>` | `accuracy/validation/step3_validation.cu` 调用 `pulse_doppler_typed_cpu<float>` | `cusignal_cpp/src/radartools/radartools_typed.h/.cpp` | 已调用算子库 CPU 实现 |
| 4 | `cfar_alpha_device<float>`、`ca_cfar_device<float>` | `accuracy/validation/step4_validation.cu` 调用 `cfar_alpha_typed_cpu<float>`、`ca_cfar_typed_cpu<float>`；功率另由 Host FP32 公式计算 | `cusignal_cpp/src/radartools/radartools_typed.h/.cpp` | 两个正式算子均调用对应 CPU 库实现，检测 mask 另做全元素精确比较 |
| 5 | `ambgfun_device<float>` | `accuracy/validation/step5_validation.cu` 调用 `ambgfun_typed_cpu<float>` 生成二维结果及两个 cut | `cusignal_cpp/src/radartools/radartools_typed.h/.cpp` | 已调用算子库 CPU 实现 |

证据分为四层：validation 源码证明实际调用；`radartools_typed.h/.cpp` 证明 API 声明和 CPU
实现存在；`logs/configure.log` 的 `cpu_references=5 relative_error_judgements=5
absolute_error_retained=5 status=passed` 证明配置门禁通过；最终 pipeline JSON 的每步
`reference_source`、`comparison_method` 和误差字段证明这些路径在 ZQ500 验收中实际运行。

### 9.3 当前实测结果

| step | GPU 被测输出 | 对应 CPU reference 和方法 | 元素数 | max abs error | max rel error | rel tolerance | 结果 |
|---|---|---|---:|---:|---:|---:|---|
| 1 | LFM、无噪回波、噪声、最终回波 | Host `waveform_cpu/echo_cpu` 按相同场景参数独立循环/相位公式复算，四组分别归一化后取最大值 | 24,640 | `5.1013358737896945e-6` | `5.1013355983833819e-6` | `2e-5` | pass |
| 2 | `[32,256]` 脉冲压缩复数结果 | `pulse_compression_typed_cpu<float>`，同一回波、模板、`nfft=256`、归一化和 circular policy | 8,192 | `1.9660499694908431e-6` | `2.4510527921028506e-7` | `2e-3` | pass |
| 3 | `[32,256]` 距离-多普勒复数结果 | `pulse_doppler_typed_cpu<float>`，同一 step2 输入、慢时间 axis、`nfft=32`、Hamming、无归一化 | 8,192 | `5.340576171875e-5` | `3.9757297706299202e-7` | `1e-2` | pass |
| 4 | 功率、alpha、阈值、检测 mask | Host FP32 功率公式 + `cfar_alpha_typed_cpu<float>` + `ca_cfar_typed_cpu<float>`；三种数值分别归一化，mask 完整 vector 精确相等 | 24,577 | `3.5846605896949768e-3` | `5.5684892729678185e-7` | `2e-2` | pass |
| 5 | 二维模糊图、delay cut、Doppler cut | `ambgfun_typed_cpu<float>` 按同一波形/采样率/cut 参数计算，三组分别归一化后取最大值 | 16,511 | `6.7173750246257713e-6` | `6.7173750246257823e-6` | `3e-3` | pass |

任务一自编程序直接完成上述对比、相对容差和 pass/fail；diPTI 不参与任一误差计算。
每个 step 的 JSON 都独立保存 `reference_source`、`comparison_method`、
`compared_elements`、`max_abs_error`、`max_rel_error`、`relative_tolerance`、
`relative_floor` 和 `first_failure_index`；本轮五步失败位置均为 `-1`，即没有元素超过相对容差。

## 10. 任务效果指标及零值审查

| step | 任务语义结果 | 判定来源 |
|---|---|---|
| 1 | 目标回波、延迟真值、Doppler 真值、噪声均存在；SNR `24.8446 dB` | GPU 分量回传 + 独立 Host 模型 + 统计量 |
| 2 | 峰值 bin `80`，距离误差 `0 m`；半功率宽度 `3` 点，小于压缩前 `64` 点 | step1 延迟真值 + 压缩后幅度峰/宽度 |
| 3 | 峰值 Doppler bin `5`，`1250 Hz`，频率误差 `0 Hz`；距离 bin 仍为 `80` | step1 真值 + `PRF/N=250 Hz` 网格 + 全图峰值 |
| 4 | 目标命中 `1`、漏报 `0`、误报 `52`，可评估区域误报率 `0.008333 < 0.02`；纯噪声误报 `0` | GPU/CPU mask 精确对照 + 目标坐标 + 可评估区域计数 |
| 5 | 归一化峰 `1`，归一化误差 `2.38419e-7`；零延迟/零 Doppler 峰值；主瓣宽 `3 samples/1 bin`；PSLR `-14.3336/-13.457 dB` | 模糊图坐标、归一化峰、主瓣边界和最大旁瓣 |

`range_error_m=0`、`doppler_error_hz=0`、峰值延迟/多普勒为零等是“真值恰好落在离散 bin”的
语义结果，不是 GPU/CPU 数组完全相同的声明。零值已按三项审查：

1. GPU 与 CPU 为独立执行路径，全元素比较的 `max_abs_error` 和 `max_rel_error` 实际均为非零；
2. step1 分别扰动延迟 `+3`、Doppler `+250 Hz`、seed `+1`，step2 扰动模板，
   step3 扰动 window，step4 扰动干扰/PFA，step5 扰动波形，均要求结果变化；
3. CPU 相位/坐标公式在 Host 独立复算，GPU 主路径不读 CPU 结果，峰值位置又与独立场景真值相互校验。

因此 Task1 本轮 `unreviewed_zero_metrics=0`。“纯噪声误报 0”是计数指标，不等于数值误差为零；
其非退化性由人工干扰能改变检测行为、降低 PFA 会提高目标位阈值的负向检查补充。

## 11. `ZQ500_diPTI` 使用情况和辅助范围

本轮使用了 `ZQ500_diPTI`，但只是辅助分析：

- 工具：`/zq500/sdk/bin/dlpti_tools`，版本 `0.9.0`，commit `4a5e8fe`，tag
  `V2_SOFTWARE_dlcu_dev_202512302023`；
- activity mask：`cmd,cu,curt`；每轮目标超时 60 s，capture 超时 90 s；
- 三轮 Task1 pipeline 的目标程序和 diPTI 退出码均为 `0`；
- 三轮 activities 为 `12,721/12,720/12,720`；第二、三轮 kind `49` 比第一轮少 1，
  其余 kind 计数一致；
- 原始 capture 分别为 `2,505,838/2,505,364/2,505,736 bytes`，保留在远程
  `/tmp/zkx-dlpti-task1-pipeline/task1-pipeline-r{1,2,3}-.../capture.json`；
- 本地交付证据保存三轮小型 summary、metadata、远程目录和双退出码。

diPTI 的作用限于确认完整 pipeline 确实产生 SDK/API、kernel、传输/同步类 activity，
并帮助后续定位耗时分布。它不产生本文正式耗时，不替代 CudaEvent/墙钟计时，不计算
精度，不决定 pass/fail，也不用 activity 条数推导算法加速比。

## 12. 构建、测试和证据门禁

实际在 `gpu_02` 执行：

```bash
source /zq500/sdk/env.sh
cd /tmp/ZKX_dev/Task1/task1_gpu_cpu
bash run_zq500_acceptance.sh \
  --commit 0372dcd5f8f512a6c2582c7fbf6b37fd90f41791 \
  --dirty true \
  --test-time 2026-07-15T19:36:00+08:00 \
  --with-dlpti
```

验收脚本删除 `cusignal_cpp/build_task1`，重新配置 `USE_DLFFT=ON`、`BUILD_TASK1=ON`，
构建六个入口，再执行 CTest 和 JSON 门禁。最终输出：

```text
100% tests passed, 0 tests failed out of 6
Total Test time (real) = 65.79 sec
[TASK1][EVIDENCE_GATE] json=6/6 formal_business_operators=5/5
  steps=5/5 version_binding=passed finite_values=passed status=passed
[TASK1][CTEST_GATE] step_targets_build_run=5/5
  task1_target_build_run=true tests=6/6 failed=0 status=passed
[TASK1][ACCEPTANCE] ... status=passed
```

源码契约同时验证五个 step/正式 API 存在，五步对应 CPU reference、validation/comparison
各 5 个、五步相对误差判定、五步保留绝对误差且 comparison GPU 路径为 0；旧路径、
NVIDIA-only API、私有替代 wrapper 不存在。运行证据门禁还验证 JSON 中五组
`max_abs_error/max_rel_error` 字段和两处精确 seed `20260715`。

## 13. 证据位置和数值源

本地可持久证据根：
`semi_final_tasks/交付/04_原始证据/task1/task1_evidence_0372dcd/task1_evidence/`。

| 证据 | 位置/用途 |
|---|---|
| 最终结构化数值源 | `pipeline/task1_pipeline_evidence.json` |
| 同轮完整打印精度 | `logs/build-and-ctest.log` |
| 单步结果 | `step1` 至 `step5` 的 `task1_stepN_evidence.json` |
| 配置证据 | `logs/configure.log` |
| 构建与 CTest 原始日志 | `logs/build-and-ctest.log` |
| 实际编译源和命令 | `logs/compile_commands.json` |
| 环境/远程状态 | `logs/environment.txt`、`logs/remote-git-status.txt` |
| 被测编译输入指纹 | `logs/source-sha256.txt` |
| diPTI 辅助证据 | `dlpti/summary-r*.json`、`metadata-r*.txt`、`*-exitcode`、`remote-run-dir-r*.txt` |
| 原始证据归档 | 同级 `task1_evidence_0372dcd.tar.gz`，SHA-256 见第 2 节 |

容器中原始构建证据根为
`/tmp/ZKX_dev/cusignal_cpp/build_task1/task1_evidence`；原始 diPTI capture 路径由
`remote-run-dir-r*.txt` 逐轮精确记录。

## 14. 算法优点、限制和改进方向

优点：

- 从回波建模到脉冲压缩、Doppler、CFAR 和模糊函数共享真实数据契约，不是五个无关 demo；
- 正式算子路径和 Task glue 边界清晰，CPU 只用于 reference/后处理，不存在 CPU fallback；
- 固定 seed、全元素对照、语义真值和负向扰动同时存在，避免只验证“程序不崩溃”；
- CudaEvent、step 墙钟和 diPTI 的口径分离，耗时可追溯。

限制与改进：

- 当前 FFT 便捷接口每次创建 plan/workspace，是 step2/3/5 主要耗时；应引入可复用
  plan/workspace，再用 warmup+多轮均值/最小/最大/标准差 benchmark 量化改进；
- 当前只有单目标、固定 SNR 和固定网格；可扩展多目标、off-bin Doppler、不同 SNR/杂波/干扰强度的曲线；
- `pfa=0.001` 与当前小样本单次误报率不应简单解释为理论虚警概率的统计估计；
  需多 seed/多帧 Monte Carlo 后才能给出置信区间；
- `formal_execution` 已排除 CPU reference；新增的同进程 1 次预热 + 10 次统计 benchmark
  见第 16 节，后续优化应沿用同一输入和计时口径复测；
- 工作树在验收时因共享 Task2/文档改动为 dirty，虽已用源指纹绑定 Task1，最终 J/K 冻结时
  仍应在单一干净交付候选上重收证据。

## 15. Task1 H 最终门禁摘要

```text
official_task1_requirements=5/5
formal_steps=5/5
step_data_continuity_pass=true
target_echo_delay_doppler_noise=true
pulse_compression_called=true
pulse_doppler_called=true
cfar_alpha_called=true
ca_cfar_called=true
ambgfun_called=true
gpu_operator_mapping=5/5
cpu_fallback=0
temporary_implementation=0
nvidia_only_path=0
unresolved_api=0
unresolved_scilib=0
step_targets_build_run=5/5
task1_target_build_run=true
self_timed_steps=5/5
accuracy_validation_files=5/5
accuracy_comparison_files=5/5
comparison_gpu_paths=0
cpu_references=5/5
relative_error_judgements=5/5
absolute_error_retained=5/5
step_accuracy_pass=5/5
task_semantic_metrics_pass=5/5
key_operator_performance_linked=true
comparison_negative_tests_pass=true
json_precision_gate=passed
unreviewed_zero_metrics=0
task_test_path_match=true
stale_step_evidence=0
missing=0
failed=0
```

## 16. cuSignal Python ZQ500 对照结果（2026-07-18 重测）

### 16.1 结论、版本和证据边界

为满足“用 cuSignal Python GPU 完整执行 Task1，并与正式 C++/CUDA GPU 版本比较性能”的新增要求，
`Task1/task1_python` 已在 ZQ500 完成五步、完整 pipeline、内存稳定性和双方同进程 1+10
benchmark。它是 Python 对照实现，不替代第 1～15 节的正式 C++/CUDA 交付，也不重复建立
CPU reference、accuracy validation 或 accuracy comparison。

本节后续比较双方均为 **ZQ500 GPU 实现**：`cuSignal Python GPU` 指基于 CuPy/cuSignal 的
`Task1/task1_python`，`C++/CUDA GPU` 指基于 `cusignal_cpp` 的
`Task1/task1_gpu_cpu` 正式 GPU 链路。这里不存在 Python CPU 与 C++ CPU 的性能比较；
以下统一写作“cuSignal Python GPU + 必需 ZQ500 adapter / C++/CUDA GPU”。加速比固定为：

```text
Speedup = T_(cuSignal Python GPU + 必需 ZQ500 adapter) / T_(C++/CUDA GPU)
```

分子是 Python 同一步骤 `formal_execution_ms`，包含真实 cuSignal 调用、任务脚手架、H2D/D2H
和 `task1_python/utils` 全部必需 adapter；分母是 C++/CUDA GPU 同一步骤
`formal_execution_ms`，不含 CPU reference、validation 或 accuracy comparison。

| 字段 | 新结果 | 来源 |
|---|---|---|
| 被测提交 | `cc85a352c4eede97e54c54393bcdce6b9cda2545`，dirty=`true` | cuSignal Python GPU benchmark、pipeline 和 C++/CUDA GPU benchmark JSON |
| 测试时间 | `2026-07-18T15:50:00+08:00` | 同上 JSON 绑定字段 |
| 设备 | `ZQ500-Q QUAD-2`，device count=1 | `logs/preflight.log`、benchmark summary |
| Python/依赖 | Python 3.12.11、NumPy 2.4.6、CuPy 13.6.0、cuSignal 23.08.00 | `benchmark/task1_python_benchmark_summary.json` |
| CUDA 版本 | runtime=11070、driver=11070，即 CUDA 11.7 | preflight、benchmark runtime 字段 |
| cuSignal 不可修改门禁 | 152 个文件；tree SHA-256=`ca9e760bdebb0a481e18171546881b2e75214a422482f7442a0814871476c163`；`radartools.py` SHA-256=`438fdde707ea86e9249cc38cb5ad8addca3e65da7f38e7d06218809236f2f650` | runtime 的 `cusignal_immutable_source`、`logs/source-contract.log` |
| 功能门禁 | 五个单步、pipeline、运行证据 `6/6` 全部 pass | `logs/step*.log`、`pipeline/*.json`、`logs/runtime-evidence-gate.log` |
| 内存门禁 | 20 轮；显存池 baseline=max=859648 bytes，growth=0 | `logs/memory-stability.log` |
| 结果语义门禁 | Step2～Step5 共 11 个 C++/Python GPU 任务指标比较全部 pass | `task1_cpp_python_performance.json/result_semantic_comparison` |
| 原始证据目录 | `task1_python_evidence_cc85a35/task1_python_evidence/` | 本地交付目录 |
| 压缩包 | `task1_python_evidence_cc85a35.tar.gz`，SHA-256=`8c0c36aaa1a771f705cac49be97cccd6f84fb89b6fb2e7b3ef51455b55c9c0c7` | 本地 `Get-FileHash` |

这里有两组不同的提交身份，不能混写：第 1～15 节的 C++ 五步 CPU 精度正式证据仍是
`0372dcd...`；本节为了公平性能比较，在 `cc85a35...` 同一源码快照重新构建 C++
`task1_benchmark`，并与 cuSignal Python GPU benchmark 绑定到相同 commit/dirty/test-time。当前 C++/CUDA GPU
benchmark 的每次任务执行仍运行现有 CPU validation/comparison，但本节主比较只读取排除
CPU reference 的逐步 `formal_execution_ms`。

### 16.2 Python 五步选用的算子和结果

| Step | Python 正式实现 | ZQ500 实测结果 |
|---|---|---|
| 1 接收模拟 | 公开 `cusignal.chirp`；`utils` 仅适配 NumPy 类型别名和 ZQ500 clang 显式 FP32 常量窄化；CuPy 延迟/多普勒/确定性噪声任务脚手架 | 回波 8192 元素；目标 delay=80 samples、Doppler=1250 Hz；峰值幅度 1.048340；pass |
| 2 脉冲压缩 | `cusignal.pulse_compression` | `complex64`；距离峰值 bin=80；峰值能量 2039.342651；半峰宽 3 bins；pass |
| 3 多普勒处理 | `utils` 同公式 FP32 Hamming dtype adapter + `cusignal.pulse_doppler` | `complex64`；峰值 Doppler bin=5、range bin=80、1250 Hz；峰值功率 18044.406250；pass |
| 4 恒虚警检测 | `cusignal.radartools.cfar_alpha` + `cusignal.radartools.ca_cfar` | alpha=7.142329；检测数 2014；目标被检出；阈值有限比例 1；pass |
| 5 模糊函数 | 三次原版 `cusignal.ambgfun(cut="2d")` 分别生成二维结果，再由 adapter 取得两个网格对齐零切片 | shape=`[127,128]`；峰值坐标=`[63,64]`；delay/doppler 峰值索引=`63/64`；主瓣宽 3/1；PSLR=-14.333704/-13.456973 dB；pass |

结果来源是 `pipeline/task1_pipeline_evidence.json` 和各 `stepN/task1_stepN_evidence.json`。
这些任务指标证明 Python 五步执行了对应 cuSignal/CuPy GPU 路径并得到有限、语义正确的结果，
但它们不是“Python 与 CPU 的误差”。本轮另将这些 Python GPU 任务指标与相同输入、相同提交
的 C++/CUDA GPU 指标比较：除 PSLR 使用 `1e-3 dB` 绝对容差外，其余峰值、频率、检测和
主瓣宽指标要求精确相等；11 项均 pass。按约定，Python 目录没有 CPU 精度文件；正式数值精度
仍由第 9 节 C++/CUDA 输出与独立 CPU reference 的全元素相对误差判定，绝对误差同时保留。
本轮也没有把 C++/CUDA GPU 与 cuSignal Python GPU 的任务指标差值冒充新的精度指标。

`cusignal-23.08.00` 本体没有任何平台补丁。`Task1/task1_python/utils/source_guard.py` 在 import
前核对整棵上游源码树，`utils/waveforms.py`、`utils/windows.py`、`utils/radartools.py` 只负责
Task1 私有兼容边界。Step1 仍由公开 `cusignal.chirp` 发起调用，waveforms adapter 只给官方
FP32 kernel 的常量添加 ZQ500 clang 要求的显式 cast。Step3 的 FP32 Hamming adapter 保持
`cusignal.hamming(sym=True)` 公式，避免 FP64 窗把输入提升到不支持的 Z2Z FFT。之所以不直接调用上游
delay/doppler 一维分支，是其内部会进入 ComplexFP64 或非 2 次幂 FFT；当前实现保持原版 2D
cuSignal 计算，再做 GPU 行/列切片。负向回归测试只篡改 `/tmp` 临时副本，门禁按预期拒绝；
正式上游摘要在测试前后保持不变。

### 16.3 CUDA 11.7/ZQ500 编译适配

容器原有 CuPy 的 NVIDIA CUDA 11.8 构建没有被当作交付运行时。验收脚本从项目
`cupy-13.6.0` 源码建立 `/tmp/task1_cupy_zq500_13_6_0_cuda_11_7` 私有副本，并按 ZQ500
SDK 的 runtime/driver 11070 编译和加载。普通 CUDA `.cu` 经 wrapper 进入
`dlcc -x cuda --cuda-gpu-arch=dlgpuc64`；FFT plan 进入 ZQ500 的 dlfft/cuFFT 兼容实现。

NVIDIA CUB、Thrust 和新随机分布扩展包含 PTX/NVVM、libdevice 或 NVIDIA nvcc 专属假设，
没有继续使用 NVIDIA `nv` 参数强行编译：

- CUB 保留 CuPy ABI stub，但 `available=false`；reduction 使用 CuPy 通用 kernel；
- Task1 不调用排序，因此不构建、不加载 Thrust 扩展；
- 不构建 NVIDIA 新 Generator 随机分布 `.cu`，Step1 使用可复现的 CuPy uint32 GPU 噪声；
- NVRTC 去除 NVIDIA arch/ftz 选项，并使用 CUDA 11.7 兼容头；
- Task1 全部 FFT 输入输出固定为 `complex64`，避免 ZQ500 不支持的 Z2Z 双精度 FFT。

支持边界及官方三份 Excel 的映射说明见
`Task1/task1_python/OFFICIAL_API_COMPATIBILITY.md`；实际兼容性不是由表格推测，而由
`logs/preflight.log` 中 Event、generic reduction、FFT、pulse compression、pulse Doppler、
CA-CFAR RawKernel 和 Task1 wrapper 的三种 ambiguity 输出的 ZQ500 实测共同确认；正式 Step5
执行三次原版 `ambgfun(cut="2d")`，再对后两次二维结果取切片，不表示上游一维 cut 分支已执行。

### 16.4 最新同口径100轮正式采样性能

双方输入完全一致：ComplexFP32、shape=`[32,256]`、fs=2.56 MHz、PRF=8 kHz、脉冲 64
samples、delay=80、Doppler bin=5/1250 Hz、带宽 640 kHz、相同目标幅度、噪声标准差和 seed。
cuSignal Python GPU 预热5次后正式运行100次，C++/CUDA GPU提供100次正式证据。cuSignal Python GPU 使用
`cupy.cuda.Event` 统计 GPU 算子，并用 `time.perf_counter` 统计阶段；C++/CUDA GPU 使用
CudaEvent/墙钟。主口径是逐步
`formal_execution_ms`，包含正式 step 和所需 D2H，不包含 CPU reference/comparison。

| Step | 分子：cuSignal Python GPU + adapter 均值 ms | 分母：C++/CUDA GPU 均值 ms | Speedup=分子/分母 | 结果 |
|---|---:|---:|---:|---|
| 1 | 2.568627 | 0.567933 | 4.522766 | C++/CUDA GPU 较快 |
| 2 | 1.128780 | 1.091368 | 1.034280 | C++/CUDA GPU 较快 |
| 3 | 0.630182 | 1.211668 | 0.520094 | cuSignal Python GPU + adapter 较快 |
| 4 | 0.914943 | 0.941896 | 0.971385 | cuSignal Python GPU + adapter 较快 |
| 5 | 2.699767 | 5.828495 | 0.463201 | cuSignal Python GPU + adapter 较快 |

同口径业务Pipeline按每轮五个匹配Step的 `formal_execution` 求和：cuSignal Python GPU
均值 **7.942299 ms**，C++/CUDA GPU均值 **9.641359 ms**，正式比值
`T_(cuSignal Python GPU + 必需 ZQ500 adapter) / T_(C++/CUDA GPU)` 为 **0.823774**。
双方各自的 `pipeline_total_ms` 因C++侧包含CPU reference/精度比较而职责不同，只保留在
原始JSON中作单实现诊断，不进入本Pipeline性能表、图17、图19或加速比。

数值来源为
`04_原始证据/tasks/tasks_chart_b027dca_20260719/chart_tasks_b027dca_20260719.tar.gz`
内的100轮Python/C++逐轮JSON和 `task1_cpp_python_performance_100.json`；公平性契约记录
Python warmup=5、双方 measured=100、输入/提交一致且 comparison passed。

### 16.5 `ZQ500_diPTI` 使用范围

本轮“cuSignal Python GPU + 必需 ZQ500 adapter / C++/CUDA GPU”100轮正式性能比较没有使用 `ZQ500_diPTI`，JSON 明确记录
`dlpti_used=false`。所有表格数字来自项目自编 Event/墙钟 benchmark。diPTI 在本节的辅助
范围为“可选的 kernel、传输和同步分析”，实际未采集、未参与计时、未计算精度、未决定
pass/fail。第 11 节的三轮 diPTI 证据属于此前正式 C++/CUDA GPU pipeline 验收，不能移作
本轮 cuSignal Python GPU benchmark 的 profiler 证据。

### 16.6 图18的Task1 C++状态说明

图18已于2026-07-19改用任务自身分配事件峰值，状态列对应`allocator_status`。在
`DLFFT=OFF、THRUST=ON`、5轮预热+20轮正式运行下，Task1 C++的任务分配基线为
338944 B、峰值671236 B、峰值增量332292 B，正式轮次结束后回到338944 B，故状态为
`pass`。同批1 ms设备全局水位在warmup后保持357777408 B、增长0，仅作诊断旁证。

同一程序附带的CPU live heap短窗方向判据本轮为`fail`；它既不是任务显存分配失败，
也不是GPU趋势、Task1功能或精度失败，因此只保留在`cpu_status/resource_status`列。

2026-07-19按当前源码重新执行1024次预热和1024次正式采样。Thrust FFT后端Task1实测
CPU净增1984 B、尾窗净增1440 B、GPU净增0，CPU、GPU及联合门禁均通过；完整日志及
SHA-256见`04_原始证据/task1/task1_memory_recheck_1254cb4/README.md`。此前独立长测
`docs/development/MEMORY_FFT_FINAL_ACCEPTANCE_20260718.md`也得到联合门禁通过，作为
交叉佐证。图表CSV同时保留`allocator_status`、`cpu_status`、`gpu_status`和
`resource_status`，不再混用四种状态。本次四组重测证据见
`04_原始证据/tasks/task_memory_retest_08e1e25_20260719/`。
