# TASK_F 复赛技术结果与证据说明

> 当前优化结果索引：本文及 `task_f_evidence_439fd41/` 冻结的是 2026-07-15 的 TASK_F
> 本文主体是历史批次，不因后续优化而覆盖。`unit_impulse` P0 已于 2026-07-18 补齐正式
> before/after、pooled p95 与 20 批长稳并标记完成；当前结果和证据边界见
> [unit_impulse P0 当前状态](../03_当前状态/UNIT_IMPULSE_P0_CURRENT_STATUS.md)。
> `firwin` 五个类型项也已于 2026-07-18 完成 compatibility、resident Stage C/fused、
> 20 批长稳和七规模 crossover；评委版同步见
> [firwin P0 当前状态](../03_当前状态/FIRWIN_P0_CURRENT_STATUS.md)。两项新增结论都不能覆盖本文主体的历史
> 265 项批次，也不能冒充 Task2 端到端收益。

## 1. 结论、版本和适用范围

TASK_F 已在中科芯 ZQ500 上完成固定 53 个算子的五类型正确性、精度和性能闭环：
`correctness=265`、`accuracy=265`、`performance=265`、`complete=265`，缺失、额外、
重复、绑定和字段错误均为 0。这里的 265 是 53 算子 × FP16、FP32、INT8、INT16、
INT32 五种业务输入类型，不是只做类型转换或代表算子 smoke。

数值结果绑定源码提交 `439fd41f78ada42f1f19a133ca987253d0be6ae2`、分支
`feature/current-workflow`、未提交状态指纹
`815fabcf1947a0a70d248f9d14c55c612217b3cd26bda2de22f4c5cee97e6971` 和批次
`task_f_final_439fd41_20260715`。证据归档提交为 `2f37b46`；该提交只增加证据文件，
不改变被测算子、测试、CMake 或数值结果。后续文档提交同理不能冒充新的硬件复验。

验收口径来自 `semi_final_tasks/LATEST_SOURCE_PLATFORMIZATION_REQUIREMENTS.md` 第 12—17
章；目标效果映射来自 `semi_final_tasks/TARGET_EFFECT_ZQ500_PLATFORMIZATION.md`；最终
Word/PDF、图表和同源数据要求来自 `semi_final_tasks/TASK_K_DELIVERY_EVIDENCE.md` K2—K5。
本文件是后续 K 技术文档可直接引用的 TASK_F 数据源，不表示 Task1、Task2、G、J、K
等后续大任务已完成。

## 2. 被测平台与构建链

| 项目 | 实测值 | 来源 |
| --- | --- | --- |
| 主机/容器 | `supervisor-X7850H0` / `gpu_02` | `batch_metadata.txt` |
| 加速卡 | ZQ500-Q QUAD-2，16 GB | `zq500_environment.txt` 的 `dlsmi` 输出 |
| DLSMI/驱动 | 12.0 / 2.3.11 | 同上 |
| CMake | 3.22.1；项目最低版本仍保持 3.16 | 同上及项目 CMake 约束 |
| 编译器 | 中科芯 `dlcc`/Clang 15.0.6 | 同上 |
| CUDA 工具版本 | 11.7 | 同上 |
| SDK | `/zq500/sdk`，先执行 `source /zq500/sdk/env.sh` | 批次环境与项目远程规则 |
| 测试时间 | 2026-07-15 04:11:46 UTC（北京时间 12:11:46） | `batch_metadata.txt` |

构建使用 ZQ500 官方模式：C++ 工程、`$DLICC_PATH` 编译器、`.cu` 按 C++/CUDA 前端
处理并链接中科芯 runtime/科学库；实际编译命令保存在 `compile_commands.json`，配置和
构建原文分别在 `cmake_configure.txt`、`cmake_build.txt`。

## 3. 固定 53 个算子

- B-spline（3）：`cubic`、`gauss_spline`、`quadratic`；
- 卷积（2）：`correlate`、`correlate2d`；解调（1）：`fm_demod`；估计（1）：
  `KalmanFilter`；滤波器设计（2）：`firwin`、`firwin2`；
- 滤波（14）：`channelize_poly`、`detrend`、`firfilter`、`firfilter2`、`freq_shift`、
  `hilbert`、`hilbert2`、`lfilter_zi`、`sosfilt`、`wiener`、`decimate`、`resample`、
  `resample_poly`、`upfirdn`；
- 峰值/雷达（6）：`argrelextrema`、`ambgfun`、`ca_cfar`、`cfar_alpha`、
  `pulse_compression`、`pulse_doppler`；
- 频谱（6）：`csd`、`istft`、`lombscargle`、`spectrogram`、`stft`、
  `vectorstrength`；
- 波形（5）：`chirp`、`sawtooth`、`square`、`gausspulse`、`unit_impulse`；
- 小波（5）：`cwt`、`morlet`、`morlet2`、`qmf`、`ricker`；
- 窗函数（8）：`chebwin`、`general_cosine`、`general_gaussian`、`hamming`、
  `kaiser`、`parzen`、`taylor`、`triang`。

E8/EL6 同批前置门禁给出：唯一 GPU API 53、最新 wrapper 53、逐函数中文声明 53、
CPU reference 53、typed 契约 265、真实 GPU 运行 265、未解决原语 0，且
`done=265`、其他状态均为 0。来源为 `e8_el6_gate.txt`；这保证 TASK_F 测的是最新正式
接口，而非 registry、转换 helper 或旧二进制。

## 4. 五类型、语义与 CPU reference

五类型是实际输入契约。每条记录保存输入和输出 dtype，允许内部采用更安全的
`ComputeT`/平台库类型后再按算子契约回转输出；整数路径不是只把 FP32 测试入口改名，
FP16 也不是 FP32 结果的事后截断。输出 shape、返回值数量、axis、padding、窗口、
归一化、复数布局等以本仓库冻结的 cuSignal 语义契约为参照；ZQ500 平台能力造成的
实现差异必须保持外部语义一致。

每个 GPU 路径对应独立 C/C++ typed CPU reference。二者共享测试输入和语义参数，但不
调用同一实现、不共用输出缓冲；GPU 结果从 ZQ500 正式 API 取得，CPU 结果由 host 参考
算法独立生成，再逐元素比较。CPU reference 的角色是对比基准，不是宣称 cuSignal 在
ZQ500 上运行。路径、类型链、后端、shape、参数和命令均在原始 F3 记录中。

## 5. “误差”与比较方法

本文误差是“ZQ500 GPU 正式算子输出相对于同一输入、同一参数下独立 C/C++ CPU
reference 输出”的差异。对第 i 个元素：

```text
abs_error[i] = abs(gpu[i] - cpu[i])
rel_error[i] = abs_error[i] / abs(cpu[i])      （abs(cpu[i]) > 1e-30）
             = abs_error[i]                    （参考值近零）
通过条件：abs_error[i] <= atol + rtol * abs(cpu[i])
```

记录全元素 `compared`、最大绝对/相对误差、mismatch、容差、finite/domain 和首个失败
位置；复数按分量/布局约定比较，索引和 mask 还审查离散结果。总计比较 12,865 个聚合
元素，265 项 mismatch 总和为 0。容差随 dtype 和算子数值特性变化，不能脱离记录中的
`atol + rtol*abs(ref)` 单独解释误差。

原始日志有 278 条正确性记录和 278 条精度记录，因为 13 个稳定 ID 额外保留了非退化、
边界或异常 case；正式矩阵仍是 265 个稳定 ID。聚合器要求一个 ID 的全部 case 通过，
不会挑选最好的一次结果。

### 5.1 精度结果和特殊值解释

- 265/265 均通过，mismatch=0；89/265 的聚合最大绝对误差显示为精确 0。
- 最大绝对误差出现在 `taylor`：`6.63525437971657e-06`；对应最大相对误差约
  `4.587e-07`，未超过各 dtype 的记录容差。
- 数值最大的“最大相对误差”出现在 `firwin`：约 `1.6872787077368557e9`，但最大绝对
  误差仅 `1.8467756668361091e-07`，mismatch=0。原因是参考值接近零，极小分母会放大
  比值；因此不能用最大相对误差单项断言结果错误，判定采用绝对项与相对项组合容差。
- 示例 `correlate/FP32`：比较 28 个元素，最大绝对误差
  `4.76837158203125e-07`、最大相对误差 `2.38418579101563e-07`、mismatch=0，
  容差为 `0.06 + 1e-4*abs(ref)`。

所有 89 个零指标 ID 都通过三审：确认 CPU/GPU 路径、输入/输出缓冲和全元素比较独立；
增加不同非空输入/shape/参数并注入超容差扰动，比较器确实拒绝；以科学计数法高精度输出
并逐元素复算。框架还对超容差、NaN、Inf、shape、index、mask、复数符号和复数布局做
8/8 负向测试，避免“比较器总通过”。

### 5.2 高精度 CPU 参考最终深度结论

2026-07-17 的最终增量不改写第 1 节 2026-07-15 的历史 265 项冻结批次。它针对数值敏感
风险增加独立 Host `long double` 高精度参考，最终覆盖 `correlate`、`sosfilt`、
`lombscargle`、`firfilter`、`upfirdn`、`pulse_compression`、
`pulse_doppler` 7 个代表算子，以及 Task1 脉冲压缩→脉冲多普勒链。

baseline、long、sensitive 和 Task 参数共形成 20 个场景、FP32/FP16 40 条精度记录，
40/40 通过；Task1 两条峰值记录 2/2 通过。FP32 在单算子场景中的高精度基准最大绝对
误差约不超过 `5.87e-6`，Task1 链为 `2.61e-5`；FP16 对场景明显更敏感，
`correlate` 敏感场景达到 `1.45e-2`，Task1 链为 `4.11e-3`。同 dtype GPU/typed
CPU 仍保持一致，说明误差增长主要来自 FP16 表示和有限精度，而不是 GPU 偏离当前 dtype
算法契约。

Task1 当前单目标场景中，FP32、FP16 均定位到 range bin 80、Doppler bin 5；但这不能
推广为 FP16 在弱目标、多目标或近阈值检测中普遍安全。因此 Task1/Task2 正式路径保持
FP32，FP16 只可在具体非敏感步骤完成业务指标复测后局部使用。高精度 CPU 耗时不进入
性能 speedup。

完整日志、CSV、源码指纹、停止条件和答辩材料见
[高精度 CPU 最终证据](../04_原始证据/operators/operator_high_precision_final_20260717/README.md)。
## 6. 异常输入和明确报错

用户指出的风险已纳入最终门禁：每个稳定 ID 都有 `invalid_case_review=1`、
`error_behavior=standard-exception-or-explicit-rejection`、`process_continued=1`。非法 shape、
axis、参数或不支持组合必须被明确拒绝，可表现为 C++ 标准异常（例如
`std::invalid_argument`）、自定义错误码或明确错误信息；不要求固定报错文字，也不要求
退出整个程序。测试捕获该拒绝后继续运行后续 case，265/265 均证明“错误被识别且进程
继续”。这满足复赛要求，也保留初赛“不必因单个异常输入退出程序”的行为。

`ambgfun` 曾暴露的是 CPU/GPU 对应位置同时为 NaN/Inf 时比较器语义处理问题，而不是
其非法参数路径没有报错；最终比较器只接受语义匹配的特殊值，单边 NaN/Inf 仍失败，
`cut`、`fs` 等非法输入路径由显式拒绝/异常证据覆盖。

## 7. 自编性能 benchmark 与结果

主性能数据来自项目自编 benchmark，不来自 profiler。每项先 warmup 5 次，再运行 20 次，
每次迭代后同步；记录 total、mean、min、max、标准差、工作量分母、吞吐量、CPU mean、
speedup 以及 allocation、H2D、D2H、workspace、plan/handle、后处理是否计入。加速比定义为
`CPU mean / GPU mean`，CPU baseline 是同一 typed CPU reference、同一测试规模和参数。

| 输入类型 | GPU 均值的跨算子均值/中位数 (ms) | 最小/最大 (ms) | speedup 均值/中位数 | speedup>1 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 1.647537 / 0.336212 | 0.054078 / 63.9410 | 16.8178 / 1.1773 | 30/53 |
| FP32 | 1.685558 / 0.336206 | 0.049418 / 66.0050 | 7.9663 / 1.0016 | 27/53 |
| INT8 | 1.645952 / 0.298320 | 0.054255 / 64.0684 | 7.2200 / 1.0830 | 27/53 |
| INT16 | 1.657689 / 0.317001 | 0.053910 / 64.1716 | 8.4805 / 0.8647 | 26/53 |
| INT32 | 1.677417 / 0.336511 | 0.054916 / 64.1323 | 6.9512 / 1.0450 | 27/53 |
| 全部 265 | 1.662830 / 0.334827 | 0.049418 / 66.0050 | 9.4871 / 1.1221 | 137/265 |

关键样例：最快均值是 `gausspulse/FP32` 0.0494184 ms，CPU 0.135099 ms，2.73377×；
最慢是 `chebwin/FP32` 66.005 ms，CPU 281.64 ms，4.26695×；最大加速比是
`correlate/FP16` 358.173×（GPU 0.0773667 ms，CPU 27.7107 ms）；最小加速比是标量
`cfar_alpha/INT16` 0.000856751×（GPU 0.242778 ms，CPU 0.000208 ms），说明小工作量下
启动/同步开销可超过计算收益，不能声称所有算子都加速。

`correlate/FP32` 主批次为 total 1.42874 ms、mean 0.0714371 ms、min 0.068381 ms、
max 0.076851 ms、标准差 0.00210968 ms、工作量 4,224、吞吐量 59,128,900 item/s、
CPU 10.3715 ms、加速比 145.184×。该项的 allocation/H2D/D2H/workspace/plan/后处理
标志均为 0，即统计稳定执行段；其他项范围必须读取各自行字段。全表中计入 allocation、
H2D、D2H、workspace、plan/handle、后处理的记录数分别是 215、205、210、115、55、
205。由于工作量定义和计时范围不同，吞吐量只用于同一算子/同一口径比较，不能跨算子
直接排名。

## 8. `ZQ500_diPTI` 是否使用及辅助边界

本次使用了 `ZQ500_diPTI`，但仅辅助采样当前提交的 `correlate/FP32` 性能入口。工具为
`dlpti_tools 0.9.0`（commit `4a5e8fe`），捕获和目标退出码均为 0；摘要含 308 条 activity、
1 个 profile target、1 条 profiler 信息，kind 计数完整保存在 `summary.json`。

采样运行中的 benchmark mean 为 0.075686 ms，主批次为 0.0714371 ms，前者高约 5.95%。
两者不是数值精度“误差”，而是不同运行中 profiler 注入、启动/同步和采样扰动导致的耗时
差异。当前最小保留摘要只证明 activity/API 类别被捕获，没有可靠的 kernel 名称级时间
拆分，因此本文不虚构 kernel 热点，也不把 profiler 时间替代自编 benchmark。主性能表、
CPU baseline、speedup 和 265 项完成判定全部来自自编测试。

## 9. 证据索引与数据来源

证据根目录为 `semi_final_tasks/交付/04_原始证据/task_f/task_f_evidence_439fd41/`：

- `task_f_metrics_265.csv`：本文全部 265 行精度/性能统计的直接数据源；
- `test_all/operators/results/structured/task_f_final_439fd41_20260715/task_f_results.csv`
  和 `.json`：严格聚合正式结果；
- 同级 `performance_candidate.csv/.json`：完整自编性能记录；
- `test_all/operators/results/raw/task_f_final_439fd41_20260715/`：环境、源码绑定、
  CMake、E8/EL6、F4、九组正确性/精度和十组性能原始日志；
- `task_f_final_driver.txt`：最终 F7/F6 汇总，其中 complete=265、pass=1；
- `dlpti_correlate_fp32/`：diPTI 命令、元数据、退出码、日志和摘要；
- `SHA256SUMS.csv`：44 个证据文件的字节数与 SHA-256，可检查归档后未被改写。

## 10. 可复现入口

按项目规则从宿主机同步源码并复制进容器，交互进入 `gpu_02` 后执行：

```bash
source /zq500/sdk/env.sh
cd /tmp/ZKX_dev/test_all/operators
export TASK_F_GIT_COMMIT=439fd41f78ada42f1f19a133ca987253d0be6ae2
export TASK_F_GIT_BRANCH=feature/current-workflow
export TASK_F_DIRTY_STATE=815fabcf1947a0a70d248f9d14c55c612217b3cd26bda2de22f4c5cee97e6971
./run_task_e8_el6_gate.sh build_f_task_f | tee /tmp/e8_el6_final_gate_439.txt
export TASK_F_E8_EL6_DONE=265
export TASK_F_E8_EL6_LOG=/tmp/e8_el6_final_gate_439.txt
./run_task_f_full_batch.sh final task_f_final_439fd41_20260715
```

无 `.git` 的容器副本还必须注入同一快照导出的 status、worktree patch 和 index patch；
脚本会重新计算 dirty 指纹并拒绝不一致绑定。最终模式会干净配置/构建、运行比较器负向
门禁、正确性、精度、性能和严格 collector；任一缺失、重复、字段/绑定错误或异常输入未被
拒绝都会非零退出。

## 11. 使用限制与后续交付

本批数据来自同一 ZQ500 环境和源码快照，适合做 265 项当前状态表、误差/精度气泡图及
耗时/加速比气泡图。图表应直接读取 CSV/JSON，并保留 commit、dtype、shape、计时范围和
容差；89 个零值只有在标注三审通过后才能使用。不同算子 workload、CPU 算法复杂度和
计时范围不同，跨算子平均值只描述本批集合，不是硬件理论峰值或公平微基准排名。

若算子、测试、类型契约、CMake 或计时实现发生变化，本证据必须标记待复核并产生新批次，
不能把 `439fd41` 的结果继续写成“最新源码实测”。最终 Word/PDF/PPT 应经 J/K 统一 evidence
ID 冻结并补入 Task1/Task2 step 级结果；本文件不替代该最终冻结流程。
