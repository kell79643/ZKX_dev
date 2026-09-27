# Task2 复赛技术结果与证据说明

## 1. 文档定位

本文汇总 Task2 六步 GPU 链路中可供复赛技术文档、Word/PDF、PPT、图表和截图索引使用的
实现方法、参数、计时口径、精度对比方法、当前 ZQ500 实测结果及证据来源。本文只记录精度
模块最终拆分后的正式结果，所有拆分前提交号、耗时和 profiler 数字均已覆盖，不作历史混用。

内容字段主要来自以下要求：

- `LATEST_SOURCE_PLATFORMIZATION_REQUIREMENTS.md` 第 13～17 章：精度指标、零值三审、
  自编性能、Task step 时间/精度和统一证据字段；
- `TARGET_EFFECT_ZQ500_PLATFORMIZATION.md` 第 6～8 节及追加门禁：Task 每步调用正式 GPU
  算子、每步独立计时、每步与 CPU reference/批准判据对比；
- `TASK_K_DELIVERY_EVIDENCE.md` K1～K5：冻结版本、技术文档内容、图表数据来源、截图索引
  和跨成品数字一致性。

## 2. 当前快照和证据状态

| 字段 | 当前值 | 来源/说明 |
|---|---|---|
| 日期 | 2026-07-16 | 本轮远程验收日期，文档时区 Asia/Shanghai；证据 JSON 时间为 UTC |
| 分支 | `snapshot/all-current-20260716` | 文档更新时 `git branch --show-current` |
| 被测 Task2 合规快照 | `fdf558f9a344a29d53e4f77b07ef59a9a918f8d3` | 原版 cuSignal 恢复后，ZQ500 兼容工具全部位于 `Task2/task2_python/utils`；双方 JSON 的 `git_commit` |
| 最新100轮图表性能提交 | `b027dca36ba1332cc9be5a340123bb2cfd17b543` | 2026-07-19 两种实现各5批，每批预热5轮、正式20轮；正式比较只使用匹配六步 `formal_execution` 之和 |
| 精度模块最终拆分提交 | `ca56c61` | 与 Task1 一致：逐 step 的 `validation/`、`comparison/`，共享 `accuracy_utils.*`，runner 统一编排 |
| 初赛遗留文档归档提交 | `4caf2f9` | 旧 NVIDIA 分析移至 `Archiving/Task2_NVIDIA_HISTORY/task2_gpu_cpu_docs/`，不参与当前结论 |
| 工作树说明 | `dirty=true`，共享树存在其他任务改动 | 被测 Task2 版本以完整 SHA、逐文件 SHA-256、远程目录和 JSON 四重绑定，未把其他任务改动计入结果 |
| 远程主机/容器 | `usr_02@10.110.12.10` / `gpu_02` | 项目固定 ZQ500 环境 |
| SDK | `/zq500/sdk`，已执行 `source /zq500/sdk/env.sh` | 2026-07-16 验收环境日志 |
| 编译器 | `/zq500/sdk/bin/dlcc`，Clang 15.0.6 | CMake configure 输出 |
| CMake | 容器实测 3.22.1 | 项目环境记录；最低版本仍保持 3.16 |
| GPU | `ZQ500-Q QUAD-2`；CUDA runtime/driver API `11070/11070` | 2026-07-16 Python ZQ500 预检 JSON |
| 契约门禁 | 算子 `53/53 approved`，dtype `265/265 approved`，证据 110 条 | 2026-07-16 当前快照 CMake 配置输出 |
| 当前 Task2 运行状态 | `current-zq500-verified` | 当前 C++ target 构建通过；Python 源码门禁、预检、六步、整链、20轮显存和双方最新100轮性能证据门禁通过 |
| 最新运行时门禁 | `7/7`，失败 0 | 六个 Python step JSON 加完整 pipeline JSON；不是 CTest 墙钟 |
| 稳态 benchmark | 两个 ZQ500 GPU 实现均为5批×20正式轮 | 正式中心值：C++/CUDA GPU 11.027838 ms；cuSignal Python GPU 28.937357 ms；均为匹配六步 `formal_execution` 之和 |
| diPTI | 本次 `fdf558f...` 未使用 | 正式时间完全来自代码内 CUDA Event/Host 墙钟；`ca56c61...` 三轮 L2 只作为历史辅助分析，不与本次性能表混用 |

> 本文第7、8节的精度证据来自 `fdf558f...` 的2026-07-16验收；第13.3节正式性能已由
> `b027dca...` 的2026-07-19同口径100轮证据覆盖。第9节保留的 `ca56c61...` 历史diPTI
> 仅用于说明profiler范围，不参与本次计时、精度或通过判定。

> **性能比较对象说明：**本文简称的“Python”是调用项目内 cuSignal 23.08.00、由私有
> CuPy 13.6.0 驱动 ZQ500 的 **cuSignal Python GPU 实现**；“C++”是调用平台化算子库的
> **C++/CUDA ZQ500 GPU 实现**。第 13 节比较的是两个 GPU 实现，不是 Python 与 CPU 的性能
> 对比。CPU 只在 C++/CUDA 精度验证中作为 reference；相关相对误差和最大绝对误差见第 6、7 节。

## 3. 拆分后的总体架构

| 模块 | 路径 | 职责 |
|---|---|---|
| 共享状态与计时工具 | `Task2/task2_gpu_cpu/task2_common.h/.cu` | `PipelineState`、`StepEvidence`、CudaEvent 和通用任务计算；不保存逐步 reference |
| 正式 GPU 六步 | `Task2/task2_gpu_cpu/step1/step1.cu`～`step6/step6.cu` | 只执行正式算法、数据衔接和 GPU/阶段计时，不调用 CPU reference 或精度比较 |
| 精度交换结构/共享工具 | `Task2/task2_gpu_cpu/accuracy/accuracy_types.h`、`accuracy_utils.*` | 保存六步 validation 数据类型，统一最大误差、失败索引和证据合并 |
| 六步精度验证 | `accuracy/validation/step1_validation.cpp`～`step4_validation.cpp`、`step5_cpu_validation.cpp`、`step6_validation.cpp` | 逐步生成 CPU reference、扰动输入和独立复算结果，不判 pass/fail；Step5 CUDA 扰动输出另在 `.cu` |
| 六步精度对比 | `Task2/task2_gpu_cpu/accuracy/comparison/step1_comparison.cpp`～`step6_comparison.cpp` | 逐步计算误差、容差、比较元素数、失败索引、语义指标和 pass/fail |
| 统一编排器 | `Task2/task2_gpu_cpu/task2_runner.h/.cu` | 按 Task1 同款顺序执行“正式 step→validation→comparison”，打印并写 JSON；失败步留证并停止下游 |
| 构建和测试入口 | `Task2/task2_gpu_cpu/CMakeLists.txt` | `task2_step1`～`task2_step6`、`task2_pipeline`、CTest 和 `task2_all` |

六个 step 共享同一个 `PipelineState`，因此 step2 消费 step1 的 `echo`，step3 消费
step2 的 `filtered`，step4 消费 step3 的 `smoothed`，step5 消费融合后的
`feature_bundle`，step6 消费 step5 的 `extrema`。各步骤入口运行到自身时仍从 step1
开始，不使用六个互不相关的演示输入。

## 4. 固定输入、正式函数和参数来源

### 4.1 公共输入

- 样本数：512；采样率：512 Hz；目标延迟：96 点；噪声 seed：`20260714`；
- 业务输入和主要 GPU 中间量为 FP32；`sawtooth`、`square`、窗口、FIR taps、CWT 等
  按正式算子契约可能使用 FP64 存储输出，再在 Host 合法收尾或显式转为任务 FP32；
- 参数常量来源：`Task2/task2_gpu_cpu/task2_common.h:20-24`。

### 4.2 六步选择

| step | 正式函数和主要参数 | 选择/参数来源 |
|---|---|---|
| 1 | `chirp(20→70 Hz, t1=1 s)`、`gausspulse(fc=170 Hz,bw=0.20)`、`sawtooth(width=0.65)`、`square(duty=0.35)` | `SELECTION_GUIDE.md` 推荐 Chirp、Gaussian、Square；额外加入 sawtooth 形成四波形组合 |
| 1 回波 | 目标权重 0.80、延迟 96；干扰权重 0.16/0.06/0.06；噪声幅值约 ±0.04 | `step1/step1.cu` 的 `mix_echo_kernel` 与 `accuracy/validation/step1_validation.cu` 的独立 Host `synthesize_cpu` |
| 2 | `hamming(sym=true)`、33 taps、截止 90 Hz、`firwin(pass_zero=true,scale=true,fs=512)`、`firfilter(axis=-1)` | Hamming 是指南中的通用雷达/实时首选 |
| 3 | 截取 `[16,496)`；`cubic` offsets `[-2,-1.5,...,2]`；权重归一化后 `firfilter` | 三次 B 样条是指南中的雷达目标检测首选 |
| 4 | `fm_demod(axis=-1)`、`correlate(mode=same)`、`spectrogram(frame=64,hop=32)`、`cwt`+`ricker(widths=2,4,8,12)` | 官方四类要求全部覆盖 |
| 4 融合 | FM/相关/谱/小波权重 `0.10/0.70/0.10/0.10` | 指南雷达场景以相关性优先；各分支先取绝对值、归一化并重采样到 480 点 |
| 5 | `argrelextrema(greater,axis=0,order=3,mode=clip)` | 官方指定峰值定位函数 |
| 6 | 一维状态/观测；`x0=0,P0=1,F=1,Q=1e-4,alpha=1,H=1,R=2e-3`；8 次观测 | 官方指定 Kalman；观测来自 step5 最强关键点周围半径 1～8 的局部加权质心 |

### 4.3 正式算子选型总表

Task2 共选取 15 个不重复的官方附件函数。`firfilter` 在 Step2 和 Step3 复用，但只按一个
正式算子计数；`kalmanfilter` 在 C++ 正式接口中展开为 predict/update 两个阶段，仍对应
附件中的一个估计算子。

| step | 官方类别 | 选取函数 | 实际 GPU API | CPU reference/对比依据 | 输出及后续用途 | 选择原因 |
|---|---|---|---|---|---|---|
| 1 | `waveforms` | `chirp` | `chirp_device` | `chirp_typed_cpu` | FP32 LFM，延迟后构成目标主体 | 通用雷达首选，兼顾距离分辨率和杂波抑制 |
| 1 | `waveforms` | `gausspulse` | `gausspulse_device` | `gausspulse_typed_cpu` | FP32 Gaussian 脉冲，作为干扰分量 | 提供时域集中的高分辨脉冲成分 |
| 1 | `waveforms` | `sawtooth` | `sawtooth_device` | `sawtooth_typed_cpu` | 正式 FP64 存储后显式转为任务 FP32 | 增加非正弦周期谐波干扰 |
| 1 | `waveforms` | `square` | `square_device` | `square_typed_cpu` | 正式 FP64 存储后显式转为任务 FP32 | 指南推荐波形，边沿和谐波形成特征噪声 |
| 2 | `windows` | `hamming` | `hamming_device` | `hamming_typed_cpu` | 33 点窗口，交给 `firwin` | 通用雷达/实时滤波首选，复杂度和阻带衰减平衡 |
| 2 | `filter_design` | `firwin` | `firwin_device` | `firwin_typed_cpu` | 33 taps、90 Hz 低通 FIR 系数 | 参数直观，可把正式窗口显式纳入设计 |
| 2、3 | `filtering` | `firfilter` | `firfilter_device` | `firfilter_typed_cpu` | Step2 产生滤波回波，Step3 执行 B 样条平滑 | 同一正式 FIR 路径可承接低通 taps 和样条权重 |
| 3 | `bsplines` | `cubic` | `cubic_device` | `cubic_cpu` | 9 点三次 B 样条权重 | 指南中雷达目标检测和高精度频率分析首选 |
| 4 | `demod` | `fm_demod` | `fm_demod_device` | `fm_demod_typed_cpu` | 调制域特征，融合权重 0.10 | 官方解调类别指定函数，表示瞬时相位变化 |
| 4 | `correlate` | `correlate` | `correlate_device(...,"same")` | `correlate_typed_cpu(method=direct)` | 480 点时域相关特征，权重 0.70 | 雷达目标检测优先级最高，反映延迟匹配位置 |
| 4 | `spectral` | `spectrogram` | `spectrogram_device` | `spectrogram_typed_cpu` | `64×14` 双边 PSD，再提取 envelope | 提供时变频域能量，比单一全局频谱更适合回波 |
| 4 | `wavelets` | `ricker` | `ricker_typed_cpu` 生成显式 wavelet bank | callable 返回值和 workspace shape | CWT 尺度 2、4、8、12 的实小波 | 适合峰形和瞬态结构，wavelet bank 可审查 |
| 4 | `wavelets` | `cwt` | `cwt_device` | `cwt_typed_cpu` | CWT `[4,480]`，取 envelope 后权重 0.10 | 提供时频联合特征，补充相关和谱分析 |
| 5 | `peak_finding` | `argrelextrema` | `argrelextrema_device` | `argrelextrema_typed_cpu` 精确 INT64 坐标 | 关键点索引，生成 Step6 观测锚点 | 官方指定函数，覆盖 comparator/order/axis/mode |
| 6 | `estimation` | `kalmanfilter` | `kalman_predict_device` + `kalman_update_device` | 对应两个 `*_typed_cpu` | FP32 状态/协方差及归一化位置估计 | 官方指定估计方法，稳定含噪关键点观测 |

按官方类别统计：

```text
waveforms=4
windows=1
filter_design=1
filtering=1
bsplines=1
demod=1
correlate=1
spectral=1
wavelets=2
peak_finding=1
estimation=1
unique_official_functions=15
```

### 4.4 任务 glue 与正式算子的边界

Task2 还包含两个任务级 custom kernel，但它们不属于附件候选函数，不能计入上述 15 个
正式算子：

- `mix_echo_kernel`：把正式波形输出按目标、干扰、噪声模型叠加为接收回波；CPU 对照为
  `accuracy/validation/step1_validation.cu` 中的 `synthesize_cpu`；
- `analytic_signal_kernel`：为 `fm_demod` 准备 `DemodComplex<float>` 输入，只承担数据
  组织，不替代 `fm_demod_device`。

技术文档应把二者写成 Task glue/数据编排，并说明输入、输出和验证方式；不能把它们宣传为
新增 cuSignal 算子，也不能用它们替代任一官方类别要求。

## 5. 自编计时方法及统计范围

### 5.1 关键 GPU 计算时间

`task2_common.h::time_gpu` 使用 `cudaEventCreate/Record/ElapsedTime`：在默认 stream 上记录
开始 event，调用被测 GPU 函数，记录结束 event，并用 `cudaEventSynchronize(end)` 等待完成。
因此 `operator_ms` 是两个 event 之间的 GPU 调用时间，不含函数外显式 H2D/D2H，也不依赖
`ZQ500_diPTI`。

需要注意：部分 convenience API 会在函数内部创建 workspace/plan，例如当前
`spectrogram_device(x,frame,hop,out)`；其 event 时间包含该公开调用边界内的 plan/workspace
成本，不能解释为单一 kernel 稳态时间。

### 5.2 Step 自编端到端时间

每个 `run_stepN` 使用 `std::chrono::steady_clock` 记录正式算法阶段；`task2_runner.cu` 再按
Task1 同款 `run_verified_step` 包装 validation 和 comparison：

- `prepare`：Host 输入、参数、shape 或可复用 workspace 的准备；
- `h2d`：显式 `DeviceArray::from_host`；
- `compute`：本 step 的 CudaEvent 关键调用时间之和；
- `d2h`：显式 `to_host`；
- `postprocess`：正式 step 的 Host 收尾，加上对应 `validation/stepN_validation.cu` 的 CPU
  reference/扰动数据生成和 `comparison/stepN_comparison.cpp` 的误差、语义、负向检查；
- `formal_execution`：只覆盖 `run_stepN` 返回前的正式算法执行，不含 accuracy 模块；
- `total`：runner 从进入正式 step 到 validation/comparison 全部结束的墙钟时间。

`total` 因而是“包含验证的 step 端到端时间”，不是纯业务推理延迟。正式技术文档若需要
纯业务延迟，应另建不执行 CPU reference 的 benchmark，不能直接删减或推算。

精度拆分只改变职责和证据组织，不改变六步算法、输入、容差或负向判据。按用户决定，不再
输出 `accuracy_validation_ms` 和 `accuracy_comparison_ms` 两个独立字段；物理分工由两个目录
体现，两类工作合并进入 `postprocess`。当前十轮 Step6 H2D 均值为 0.649968 ms，包含八次
观测 `z` 的显式 H2D 累计。

稳态统计采用同一可执行文件先预热 1 次，再完整运行 10 次。均值、最小值、最大值和样本
标准差由 10 份项目自编 JSON 汇总；精度与任务指标在每轮均执行，不因 benchmark 而跳过。
本输入规模固定为 512 点、主要任务 dtype 为 FP32。本文不报告吞吐率或 CPU 加速比，因为当前
Task2 没有“只跑 CPU pipeline 且排除验证逻辑”的同口径性能入口；CPU reference 的职责是
正确性对照，不能拿 `postprocess` 时间反推 CPU baseline。

### 5.3 `ZQ500_diPTI` 边界

本节所有正式 Task2 时间均来自项目自编代码，diPTI 不参与精度或性能 pass/fail。性能数据
提交 `ca56c61...` 额外执行三轮完整 pipeline L2 采集，activity mask 为 `cmd,cu,curt`，
用于确认调用结构、
runtime/driver activity、传输和同步记录的可重复性。由于该层只做 activity 汇总，且 profiler
本身会引入开销，不能把 activity 时间当作 CudaEvent 时间，也不能用三轮 profiler 运行替代
十轮自编 benchmark。详细结果见第 9 节。

## 6. 精度对比对象、方法和容差

### 6.0 CUDA/CPU 统一误差公式

Step1～Step4、Step6 均将正式 CUDA 输出记为 \(y_{CUDA}\)，将对应 C/C++ CPU reference
记为 \(y_{CPU}\)。主要判据为相对 L2 误差：

\[
E_{rel}=\frac{\lVert y_{CUDA}-y_{CPU}\rVert_2}
{\max(\lVert y_{CPU}\rVert_2,10^{-12})}
\]

同时保留最大绝对误差数值：

\[
E_{abs}=\max_i\left|y_{CUDA,i}-y_{CPU,i}\right|
\]

代码实现位于 `Task2/task2_gpu_cpu/accuracy/accuracy_utils.cpp`：`relative_l2_error()` 对应
\(E_{rel}\)，`max_abs_error()` 对应 \(E_{abs}\)。实现中的平方参考量下限 `1.0e-24` 在开方后
等价于公式分母的 `1.0e-12`。每个 `stepN_comparison.cpp` 调用这两个函数；runner 将结果分别
写入 JSON 的 `accuracy.relative_error`、`accuracy.relative_tolerance` 和
`accuracy.max_abs_error` 字段。

连续数值步骤以 \(E_{rel}\) 为主要 pass/fail 判据；复合 step 取所有被比较输出中的最大相对
L2 误差。\(E_{abs}\) 继续完整记录，首先要求两侧元素数完全相同，然后比较全部元素，不抽样，
但不再作为主要通过判据。Step5 是 INT64 索引输出，改用相对索引错配率作为主要判据，并
保留最大绝对索引差，详见第 6.5 节。

### 6.1 Step1

- GPU 被测量：四个正式波形输出、目标分量、干扰分量、噪声分量和最终回波；
- reference：`chirp_typed_cpu`、`gausspulse_typed_cpu`、`sawtooth_typed_cpu`、
  `square_typed_cpu`；任务叠加由 Host `synthesize_cpu` 按同一公开参数独立循环复算；
- 相对误差实际取 chirp、target、interference、noise、echo 五组相对 L2 误差的最大值；
  最大绝对误差仍取五组最大绝对误差中的最大值；
  Gaussian/sawtooth/square 通过其进入叠加后的分量和回波间接覆盖；
- 容差：`3.0e-5`；
- 扰动：seed `+1`、目标权重 `0.80→0.75`，要求回波变化量大于 `1e-3`。

### 6.2 Step2

- GPU 被测量：`hamming_device → firwin_device → firfilter_device` 的最终 512 点滤波输出；
- reference：`hamming_typed_cpu → firwin_typed_cpu → firfilter_typed_cpu`，使用相同 33 taps、
  90 Hz、fs、scale、axis 和完整 step1 回波；
- 主指标：最终输出相对 L2 误差；相对容差 `5.0e-4`；同时保留最大绝对误差；
- 任务效果：分别对目标分量和“回波减目标”的残差执行同一 CPU FIR，计算
  `SNR=20*log10(RMS(target)/RMS(residual))`，要求滤波后至少提升 1 dB。

### 6.3 Step3

- GPU 被测量：`cubic_device` 生成并归一化 9 点权重，再由 `firfilter_device` 平滑；
- reference：`cubic_cpu` 和 `firfilter_typed_cpu`；
- 主指标：480 点平滑输出相对 L2 误差；相对容差 `5.0e-5`；同时保留最大绝对误差；
- 粗糙度：二阶差分的 RMS，要求平滑后小于平滑前；
- 结构保持：`peak_after/peak_before`，并要求平滑后峰值绝对值大于原峰值的 20%。

### 6.4 Step4

- GPU 被测量：FM、same 相关、spectrogram envelope、CWT envelope 和最终融合 bundle；
- reference：Host `analytic_cpu` 后调用 `fm_demod_typed_cpu`，以及
  `correlate_typed_cpu(method=direct)`、`spectrogram_typed_cpu`、`cwt_typed_cpu`；随后使用
  相同 envelope、重采样、归一化和融合公式；
- 主指标：上述五组输出相对 L2 误差的最大值；相对容差 `1.0e-2`；最大绝对误差保留；
- 非退化判据：融合 bundle 最大值必须大于 0.1。

### 6.5 Step5

- GPU 被测量：`argrelextrema_device` 返回的一维 INT64 坐标数组；
- reference：`argrelextrema_typed_cpu`；
- 主指标：三组 CUDA/CPU 坐标 vector 的相对错配率，要求为 0；同时记录最大绝对索引差，
  并继续要求坐标数量、顺序和值精确相等；
- 负向测试：在索引 50 注入高峰，随后恢复并把高峰移到 70，要求 GPU/CPU 检测索引均随之
  移动且逐项相等；再人工把期望坐标加 1，要求比较器判不等；
- 零值审查：GPU/CPU 路径独立、输入扰动、比较器负向、INT64 独立整数复算四项门禁均为 1。
  因输出是离散坐标，高精度浮点复算不适用；最终表述为“INT64 索引精确匹配”，不宣传为
  浮点意义的“绝对零误差”。

### 6.6 Step6

- GPU 被测量：8 次 `kalman_predict_device + kalman_update_device` 后的状态 `x` 和协方差 `P`；
- reference：同一观测序列上的 `kalman_predict_typed_cpu + kalman_update_typed_cpu`；
- 主指标：最终 `x` 和 `P` 两组相对 L2 误差的最大值；相对容差 `2.0e-5`；两组最大绝对
  误差中的最大值继续保留；
- 独立任务真值：目标延迟映射到 480 点融合坐标后的归一化位置；要求估计与真值绝对误差
  不大于 0.18；
- 扰动：CPU reference 将测量噪声 `R` 从 `2e-3` 改为 `0.5`，要求结果变化且保持有限值。

## 7. 当前 ZQ500 精度与任务效果

以下精度结果来自提交 `fdf558f...` 的 2026-07-16 ZQ500 C++ 十轮完整 pipeline JSON。
每轮均在固定输入和 seed 下重新生成 CUDA 输出、对应 C/C++ CPU reference 和任务判据；六步
十轮全部 `pass`，表中数值在十轮内一致。CPU 验证实现位于独立
`accuracy/validation/*.cpp`，误差计算和判定位于独立 `accuracy/comparison/*.cpp`，没有与
正式 `.cu` step 混写。

| step | CUDA 与谁比较 | 主要相对误差 | 相对容差 | 最大绝对误差（保留） | 元素数 | 结果 |
|---|---|---:|---:|---:|---:|---|
| 1 | typed CPU 四波形 + Host `synthesize_cpu` | `1.061341108e-5` | `3.0e-5` | `1.799315214e-6` | 2560 | pass |
| 2 | CPU Hamming→FIR 设计→FIR 滤波链 | `1.197660815e-7` | `5.0e-4` | `2.384185791e-7` | 512 | pass |
| 3 | `cubic_cpu` + CPU FIR | `5.931607227e-8` | `5.0e-5` | `5.960464478e-8` | 480 | pass |
| 4 | 四域 typed CPU reference + Host 融合 | `7.774341946e-7` | `1.0e-2` | `2.288818359e-5` | 1933 | pass |
| 5 | `argrelextrema_typed_cpu` | `0`（索引错配率） | `0` | `0`（最大绝对索引差） | 187 | pass，精确匹配 |
| 6 | CPU Kalman predict/update | `6.831205823e-8` | `2.0e-5` | `2.910383046e-11` | 2 | pass |

Step5 的 `0` 不表示浮点无限精度，而表示两个 INT64 坐标 vector 完全相等。三审证据为：

1. GPU `argrelextrema_device` 与独立 C++ CPU `argrelextrema_typed_cpu` 分路径计算；
2. 将峰从索引 50 移到 70 后，GPU 与 CPU 重新计算结果仍精确相等，检测位置随输入移动；
3. 人工把期望坐标加 1 后，精确比较器必须判不等；INT64 输出不适用更高浮点精度复算，采用
   独立整数复算代替。十轮 JSON 中四个 `zero_review_*` 门禁均为 1。

任务效果不是 CPU/GPU 数值误差的替代，而是额外验证算法确实完成任务：

| 指标 | 当前值 | 计算对象/公式 | 判定 |
|---|---:|---|---|
| Step1 波形数量 | 4 | `chirp`、`gausspulse`、`sawtooth`、`square` | ≥3，pass |
| Step1 扰动变化量 | 0.118545 | 更换 seed 并改变目标权重后的回波差异 | >`1e-3`，pass |
| Step2 SNR | 16.7468→25.4972 dB | 同一 CPU FIR 后目标 RMS 与残差 RMS，`20log10` | 提升 8.75032 dB，pass |
| Step3 粗糙度 | 0.135166→0.105550 | 二阶差分 RMS | 下降，pass |
| Step3 峰值保持比 | 0.952412 | 平滑后/平滑前绝对峰值 | >0.20，pass |
| Step4 特征覆盖 | 4/4，bundle 480 点 | 四类非退化特征归一化、对齐并融合 | pass |
| Step5 关键点 | 62 | 融合 bundle 上 `greater/order=3/mode=clip` | 注入/移动峰跟随，pass |
| Step6 估计 | 0.743048 | 8 个局部质心观测的 Kalman 结果 | 有限，pass |
| Step6 仿真真值 | 0.700418 | 目标延迟映射到 480 点归一化坐标 | 绝对误差 0.042631≤0.18，pass |
| Step6 噪声响应 | 0.043490 | CPU reference 将 `R` 从 `2e-3` 改为 `0.5` | 结果变化且有限，pass |

## 8. ZQ500 构建、合规复验与性能结果

### 8.0 最新严格合规复验

2026-07-16 按项目标准远程流程同步提交
`fdf558f9a344a29d53e4f77b07ef59a9a918f8d3`，复制到 `gpu_02:/tmp/ZKX_dev`，初始化
`/zq500/sdk/env.sh` 后配置并构建 `task2_pipeline`。配置门禁为证据 110 条、算子
`53/53 approved`、dtype `265/265 approved`，target 构建到 `100%`。随后完整验收执行
C++ 预热 1 次和正式 10 次，十个 JSON 顶层均为 `requested_steps=6`、
`completed_steps=6`、`status=pass`，完整 SHA 与本节一致；每步 `accuracy` 对象均包含：

- `relative_error`、`relative_tolerance`、`max_abs_error`；
- `compared_elements`、绝对诊断容差、`first_failure_index`；
- `reference_source`、`comparison_method`、`status`；
- `failure_reason`、`input_shape`、`input_dtype`、`failure_action`。

当比较判据失败或正式步骤、validation、comparison 抛出异常时，runner 将其转换为失败步的
结构化证据、返回非零状态并停止后续步骤，避免只打印异常、JSON 丢失或下游继续运行。
本次十轮六步全部通过，故六步 `first_failure_index=-1`、`failure_reason` 为空；该值表示当前
固定输入没有失败元素，不表示未实现失败定位。完整验收还单独执行 Python 六步、整链、20 轮
显存观察和运行时证据门禁，详见第 13 节。

### 8.1 十轮量化数据的构建快照

实际构建与验收命令的核心部分：

```bash
source /zq500/sdk/env.sh
cd /tmp/ZKX_dev/cusignal_cpp
cmake -S . -B build_task2_python_compare \
  -DUSE_DLFFT=ON \
  -DBUILD_TASK2=ON \
  -DTASK2_EVIDENCE_GIT_COMMIT=fdf558f9a344a29d53e4f77b07ef59a9a918f8d3
cmake --build build_task2_python_compare --target task2_pipeline -j2
cd /tmp/ZKX_dev/Task2/task2_python
export TASK2_CPP_EXECUTABLE=/tmp/ZKX_dev/cusignal_cpp/build_task2_python_compare/Task2/task2_gpu_cpu/task2_pipeline
./run_zq500_acceptance.sh \
  --commit fdf558f9a344a29d53e4f77b07ef59a9a918f8d3 \
  --dirty true --test-time 2026-07-16T10:46:25+00:00
```

配置阶段通过，并输出：

```text
[CONTRACT][EVIDENCE] total=110
[CONTRACT][OPERATORS] total=53 approved=53 reviewed=0 draft=0 blocked=0
[CONTRACT][DTYPES] total=265 approved=265 reviewed=0 draft=0 blocked=0
Using ZQ500 DLI_V2 architecture options
Configuring done
Generating done
```

构建确认 `accuracy_utils.cpp`、6 个 validation、6 个 comparison、`task2_common.cu`、
`task2_runner.cu` 和 Step1～Step6 均作为独立编译单元进入 dlcc。验收脚本随后以同一输入契约
运行 Python 六个“从 Step1 跑到指定步骤”的独立入口、一次 Python 完整 pipeline、Python
1+10 和 C++ 1+10。独立入口的冷启动/JIT 时间不作为稳态 pipeline 延迟，正式结果只取各自
benchmark JSON。

### 8.2 十轮 step 分阶段统计

统计条件：固定 512 点、主要 dtype FP32、先预热 1 次、正式 10 次；表中各时间列为均值，
`total 范围/σ` 给出最小值、最大值和样本标准差。单位均为 ms。

| step | prepare | H2D | CudaEvent compute | D2H | postprocess | formal execution | total 均值 | total 范围/σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.026197 | 104.192200 | 4.169771 | 0.388798 | 0.386589 | 109.102300 | 110.116900 | 108.382～112.061 / 1.132706 |
| 2 | 0.001653 | 0.152537 | 1.308283 | 0.148716 | 0.562185 | 1.729832 | 2.533050 | 2.45085～2.65056 / 0.051868 |
| 3 | 0.002153 | 0.150914 | 0.279351 | 0.101955 | 0.125736 | 0.618893 | 0.942892 | 0.923356～0.978666 / 0.019474 |
| 4 | 5.409320 | 0.097725 | 3232.049000 | 0.202168 | 8.465478 | 3238.345000 | 3247.002000 | 3237.68～3258.55 / 6.292554 |
| 5 | 0.011723 | 0.055783 | 0.576497 | 0.060276 | 1.205397 | 0.747334 | 2.147859 | 2.01437～2.80460 / 0.245129 |
| 6 | 0.010728 | 0.766927 | 3.308477 | 0.092628 | 0.066513 | 5.101606 | 6.185613 | 5.99799～6.36826 / 0.115972 |

完整 C++/CUDA ZQ500 GPU pipeline 墙钟：均值 **3368.947000 ms**，最小 3360.100000 ms，最大
3381.130000 ms，样本标准差 6.629156 ms。六步 `total` 之和：均值
**3368.928000 ms**，最小 3360.090000 ms，最大 3381.110000 ms，样本标准差
6.627674 ms。step `total` 包含 validation/comparison，属于“带自检
端到端时间”；纯正式算法边界看 `formal execution`，纯 GPU 关键调用看 `CudaEvent compute`。

### 8.3 十轮关键 GPU 调用统计

下表来自每轮 JSON 的 `operators_ms`，均为 CudaEvent 公开调用边界；单位 ms，标准差为样本
标准差。`gpu_echo_superposition` 和 `analytic_glue` 是 Task glue，不计入 15 个正式附件函数。

| step/调用 | 均值 | 最小 | 最大 | σ |
|---|---:|---:|---:|---:|
| 1 `chirp` | 3.316732 | 3.248270 | 3.360300 | 0.035763 |
| 1 `gausspulse` | 0.129636 | 0.123698 | 0.140692 | 0.006286 |
| 1 `sawtooth` | 0.348105 | 0.320862 | 0.380700 | 0.018979 |
| 1 `square` | 0.284900 | 0.276530 | 0.310808 | 0.010726 |
| 1 `gpu_echo_superposition` | 0.090399 | 0.087446 | 0.094818 | 0.002494 |
| 2 `hamming` | 0.335779 | 0.298928 | 0.447580 | 0.042525 |
| 2 `firwin` | 0.561285 | 0.541258 | 0.578450 | 0.011797 |
| 2 `firfilter` | 0.411218 | 0.389064 | 0.436482 | 0.014890 |
| 3 `cubic` | 0.087630 | 0.082378 | 0.091662 | 0.003243 |
| 3 `firfilter_b_spline` | 0.191721 | 0.182860 | 0.201300 | 0.005601 |
| 4 `analytic_glue` | 0.073995 | 0.070834 | 0.076490 | 0.001849 |
| 4 `fm_demod` | 0.089821 | 0.084822 | 0.102742 | 0.005256 |
| 4 `correlate` | 0.147464 | 0.143290 | 0.154056 | 0.002979 |
| 4 `spectrogram` | 3231.349000 | 3222.000000 | 3242.870000 | 6.302325 |
| 4 `cwt_ricker` | 0.390306 | 0.368584 | 0.412726 | 0.016897 |
| 5 `argrelextrema` | 0.576497 | 0.513264 | 0.998872 | 0.148797 |
| 6 `kalmanfilter_predict_update`（8 次累计） | 3.308477 | 3.172890 | 3.447650 | 0.074937 |

Step4 占六步总时间的绝大部分，`spectrogram` 又占 Step4 compute 的绝大部分。这里测量的是
`spectrogram_device(x,frame,hop,out)` 公开 convenience 调用，包含其内部 plan/workspace 和
多个 kernel，不应写成“单个 FFT kernel 需要 3.21 s”。优化优先级是复用 plan/workspace，
然后按相同预热和十轮口径复测。当前没有同口径 CPU-only pipeline，所以不报告 CPU 加速比。

## 9. 历史 ZQ500_diPTI 辅助分析（不属于本次性能结果）

### 9.1 使用情况和命令

历史提交 `ca56c61...` 使用了 ZQ500 SDK 实际提供的 `dlpti_tools 0.9.0`（commit
`4a5e8fe`）做三轮完整
pipeline L2 采集：

```bash
for r in 1 2 3; do
  bash tools/dlpti/dlpti_capture.sh \
    --case task2-ca56c61-r${r} \
    --output-dir /tmp/zkx-dlpti-task2-ca56c61 \
    --activity-mask cmd,cu,curt \
    --timeout-seconds 60 --capture-timeout-seconds 90 \
    -- ./build_task2_accuracy/Task2/task2_gpu_cpu/task2_pipeline
done
```

| run | DB 字节 | JSON 字节 | activities | target/dlPTI 退出码 |
|---|---:|---:|---:|---|
| `task2-ca56c61-r1-20260715T043641-138977` | 585728 | 1237364 | 5812 | 0/0 |
| `task2-ca56c61-r2-20260715T043646-139067` | 585728 | 1236699 | 5812 | 0/0 |
| `task2-ca56c61-r3-20260715T043651-139157` | 585728 | 1236640 | 5812 | 0/0 |

三轮 kind 分布完全一致：`1:2073`、`2:1886`、`17:1`、`18:1`、`19:1`、`20:37`、
`49:342`、`50:329`、`56:188`、`57:858`、`60:48`、`61:48`，合计 5812。

### 9.2 辅助范围和不能推出的结论

diPTI 在本任务中用于：

- 确认三轮 pipeline 的 activity 数量和类别分布可重复；
- 辅助定位 runtime/driver 调用、kernel、内存传输和同步活动；
- 为 Step4 `spectrogram` 热点的后续逐 range/kprof 分析提供入口。

diPTI 不用于：

- 生成 CPU/GPU 误差或决定精度 pass/fail；
- 替代 CudaEvent 的关键 GPU 调用时间；
- 替代 step `steady_clock` 端到端时间；
- 直接给出 pipeline 加速比或把 profiler activity 时间解释成纯 kernel 时间。

正式性能表以第 8、13 节 `fdf558f...` 的项目自编 benchmark 为准。本次验收没有执行 diPTI；
三轮历史数据只证明 `ca56c61...` 的完整 pipeline 可重复采集，不能与本次正式数据混算。若要
证明某个 kernel 的 active cycles 或优化前后变化，仍需单独的 kprof/range 实验，不能从本节
L2 kind count 猜测。

## 10. 四类多域特征融合与后续链路合规检查

结论：**符合任务要求，四类特征先融合，再统一进入峰值检测和 Kalman 参数估计，没有绕过
融合 bundle 的旁路。**

源码数据流为：

1. Step4 从同一 Step3 `smoothed` 输入分别生成调制域 `fm_feature`、相关域
   `correlation_feature`、频域 `spectral_feature` 和时频域 `wavelet_feature`；
2. `make_bundle` 把四类特征分别线性重采样到 480 点、取绝对值并各自按最大值归一化；
3. 统一融合公式为
   `bundle[i]=0.10*FM+0.70*Correlation+0.10*Spectral+0.10*Wavelet`，四权重之和为 1；
4. Step5 明确要求 `feature_bundle` 非空，并只把该融合结果上传给
   `argrelextrema_device(greater,axis=0,order=3,mode=clip)`，得到统一关键点 `extrema`；
5. Step6 先在 `extrema` 中按 `feature_bundle` 幅值选择最强关键点，再由该点附近半径 1～8 的
   融合特征局部加权质心生成 8 个观测，最后统一执行 Kalman predict/update。

对应源码证据：`step4/step4.cu` 的 `make_bundle/run_step4`、`step5/step5.cu` 的
`run_step5`、`step6/step6.cu` 的 `observations_from_feature_point/run_step6`，以及
`task2_runner.cu` 的顺序编排。运行证据为
`feature_families=4`、`feature_bundle_size=480`、`step4_to_step5_data_match=1`、
`step5_to_step6_data_match=1`，十轮均保持一致。

这种“先融合再检测/估计”的实现满足 `TASK_I_TASK2_PIPELINE.md` 对 Step4 输出组织为 Step5 可
消费特征集合、Step5 从四类特征或明确融合结果定位、Step6 根据关键点和前述链路生成观测的
要求。需要说明的是，0.70 的相关域权重是基于 `SELECTION_GUIDE.md` 的雷达定位优先级做出的
工程选择，不是官方强制权重；后续可用验证集调参，但不能删除任一特征类别。

## 11. 当前可陈述结论、限制与优化方向

可以陈述：

1. 最新严格合规提交 `fdf558f...` 已在 ZQ500 上完成 C++/CUDA GPU 构建及两个 GPU 实现的全量验收；
2. 六步正式主路径调用 ZQ500 typed GPU 算子，CPU 实现仅用于 reference/postprocess；
3. 当前已有 step 分阶段时间、关键调用 CudaEvent、十轮统计及每步精度和任务效果；
4. Step5 的离散零值已完成路径独立、输入扰动、比较器负向和独立整数复算审查；
5. 四类多域特征先融合，再统一进入峰值检测和 Kalman，链路连续；
6. 本次正式性能未使用 diPTI；`ca56c61...` 的三轮历史 L2 辅助采集已与正式结果分开；
7. 任一步失败会记录输入形状、dtype、原因、首个失败索引和处置路径，并停止下游步骤。

仍不能陈述：

1. “Task2 相对 CPU 加速多少”——没有同口径 CPU-only pipeline benchmark；
2. “spectrogram 单个 kernel 耗时 3.21 s”——当前测的是包含 plan/workspace 的公开调用；
3. “diPTI 证明了 Task2 性能”——正式性能证据是项目自编 CudaEvent/墙钟统计；
4. “所有可能输入都通过”——当前结果绑定 512 点固定仿真输入和指定参数；
5. “共享仓库所有任务都已冻结”——本文件只闭环 Task2，被测源码用 SHA 单独绑定。

优化优先级：

- 将 `spectrogram` convenience 路径改为显式可复用 plan/workspace，再按同一十轮口径比较；
- 继续区分进程冷启动和同进程稳态；当前正式表已经使用同进程预热 1 次后采样 10 次；
- 为 Step6 保存完整状态序列，便于绘制收敛曲线，而不仅记录最终状态；
- 扩展 Step5 的近邻峰、平台峰和边界峰测试；
- 若交付材料需要加速比，新增独立 CPU-only pipeline benchmark，保持相同输入、输出和统计
  范围，不能把 CPU reference 的 postprocess 时间当 baseline。

## 12. 数据源和成品使用规则

当前证据来源：

| 证据 | 位置/来源 | 用途 |
|---|---|---|
| 最新合规源码 | Git commit `fdf558f9a344a29d53e4f77b07ef59a9a918f8d3` | 固定 C++ accuracy 分工、Python 同链路、Task2 本地 `utils` 和原版 cuSignal 不可修改边界 |
| 最新构建 | 容器 `/tmp/ZKX_dev/cusignal_cpp/build_task2_python_compare` | 配置门禁、`task2_pipeline` 构建成功 |
| C++ 十轮 JSON | `task2_python_evidence_fdf558f/task2_python_evidence/cpp_benchmark/run_1`～`run_10` | 六步相对误差、最大绝对误差、语义指标、分阶段和算子时间 |
| Python 十轮汇总 | `task2_python_evidence_fdf558f/task2_python_evidence/benchmark/task2_python_benchmark_summary.json` | Python 均值、min、max、样本标准差和算子统计 |
| 两个 GPU 实现比较 JSON | `task2_python_evidence_fdf558f/task2_python_evidence/task2_cpp_python_performance.json` | cuSignal Python GPU 与 C++/CUDA GPU 在同输入契约下的逐步、逐算子和 pipeline 比值 |
| 验收日志和哈希 | `task2_python_evidence_fdf558f/task2_python_evidence/logs/` | 环境、Git 状态、源码 SHA-256、六步、显存、合同和门禁日志 |
| diPTI 原始证据 | 容器 `/tmp/zkx-dlpti-task2-ca56c61/<run>/capture.db/json` | L2 activity 辅助分析 |
| diPTI 摘要 | 同目录各轮 `summary.json` | activities、kind count、工具版本 |
| 精度验证实现 | `Task2/task2_gpu_cpu/accuracy/validation/*_validation.cpp` | CPU reference、扰动输入和独立复算数据；Step5 CUDA 扰动收集单独为 `.cu` |
| 精度对比实现 | `Task2/task2_gpu_cpu/accuracy/comparison/step1_comparison.cpp`～`step6_comparison.cpp` | 误差、容差、失败索引、任务判据和负向测试 |

Word/PDF/PPT 和图表应统一使用第 7、8、13 节 `fdf558f...` 的同批精度与双方十轮性能，
标注 commit、dirty、512 点、FP32、预热 1 次、正式 10 次和计时边界。建议图表
包括六步时间分解、Step4 调用热点、六步相对误差与保留的最大绝对误差、
SNR/粗糙度/峰值保持/Kalman 真值误差以及四类融合数据流。Step5 可展示为“INT64 索引精确
匹配”，不要在对数误差轴上伪造一个非零值。

本次原始 JSON、日志和 tar 归档已经进入交付目录，不再依赖远程 `/tmp` 持久保存。后续图表
应直接由同一机器可读数据生成，避免手工抄表造成 Word、PDF、PPT 数字不一致。

## 13. cuSignal Python GPU 结果与两个 ZQ500 GPU 实现比较

### 13.1 cuSignal Python GPU 实现边界

`Task2/task2_python` 完全复现 Step1～Step6，但不建立 CPU accuracy 目录。其正式调用为：

1. Step1：`chirp`、`gausspulse`、`sawtooth`、`square`；
2. Step2：`hamming`、`firwin`、`firfilter`；
3. Step3：`cubic`、`firfilter`；
4. Step4：`fm_demod`、`correlate`、`spectrogram`、`cwt/ricker`，四域归一化后融合；
5. Step5：`argrelextrema`，只消费融合结果；
6. Step6：`KalmanFilter.predict/update`，消费峰值观测。

项目内 `cusignal-23.08.00` 源码未修改；ZQ500 差异由
`Task2/task2_python/utils/cusignal_source_adapter.py` 在运行时处理。预检记录 Python
`3.12.11`、NumPy `2.4.6`、CuPy `13.6.0`、cuSignal `23.08.00`、设备
`ZQ500-Q QUAD-2`、runtime/driver `11070/11070` 及
`vendored_source_modified=false`。

### 13.2 cuSignal Python ZQ500 GPU 十轮分步结果

| Step | total 均值 | formal execution 均值 | CUDA Event compute 均值 |
|---|---:|---:|---:|
| 1 | 8.454345 | 3.948806 | 2.864965 |
| 2 | 2.852834 | 2.698683 | 2.128414 |
| 3 | 1.026927 | 0.941533 | 0.455492 |
| 4 | 10.232242 | 10.153875 | 9.139943 |
| 5 | 6.759917 | 2.279668 | 2.013597 |
| 6 | 24.842551 | 13.038280 | 9.868312 |

cuSignal Python ZQ500 GPU pipeline 均值 **54.254358 ms**，最小 **47.870163 ms**，最大 **55.718086 ms**，
样本标准差 **2.299816 ms**。六步效果检查全部 pass；20 轮显存观察为
`baseline=max=growth=0 bytes`（这是历史CuPy池保留量观察，不是图18峰值）；运行时证据
门禁`7/7`、Python CPU accuracy声明`0`。图18已于2026-07-19按分配事件口径重测：Task2
Python GPU基线0 B、峰值53248 B、增量53248 B；Task2 C++ CUDA GPU基线7168 B、
峰值83724 B、增量76556 B。两组结束时均回到基线，`allocator_status=pass`；完整日志见
`04_原始证据/tasks/task_memory_retest_08e1e25_20260719/`。

### 13.3 cuSignal Python GPU 与 C++/CUDA GPU 同输入比较

本节所有 `Python GPU/C++ GPU` 比值均采用同一正式加速比定义：

\[
Speedup =
\frac{T_{\text{cuSignal Python GPU + 必需 ZQ500 adapter}}}
     {T_{\text{C++/CUDA GPU}}}
\]

分子为未修改 cuSignal 23.08 Python/GPU 算子加上 `Task2/task2_python/utils` 中仅为 ZQ500
不兼容点提供的 adapter；adapter 的设备执行、分配和同步开销均计入，预热阶段首次 JIT 不计入
正式样本。分母为相同输入、参数和六步职责下 C++/CUDA GPU 的 `formal_execution_ms`，不含
CPU validation、CPU comparison 或语义后检查。因而表中比值大于 1 表示 C++/CUDA GPU 更快；
Pipeline正式比较也必须使用六个匹配Step的 `formal_execution_ms` 逐轮求和；双方各自包含
不同验证/语义职责的 `pipeline_total_ms` 只作单实现诊断，不计算跨实现比值。

| Step | cuSignal Python GPU formal 均值 | C++/CUDA GPU formal 均值 | Speedup = T<sub>cuSignal Python GPU + 必需 ZQ500 adapter</sub> / T<sub>C++/CUDA GPU</sub> | 较快实现 |
|---|---:|---:|---:|---|
| 1 | 3.424752 | 1.582380 | 2.164305 | C++ GPU |
| 2 | 2.450149 | 0.466692 | 5.249999 | C++ GPU |
| 3 | 0.829742 | 0.664074 | 1.249471 | C++ GPU |
| 4 | 8.940632 | 1.971777 | 4.534303 | C++ GPU |
| 5 | 2.004896 | 0.457192 | 4.385234 | C++ GPU |
| 6 | 11.287856 | 5.882796 | 1.918791 | C++ GPU |

同口径业务Pipeline以六个匹配Step的 `formal_execution_ms` 逐轮之和计算。按五批、每批
预热5轮并正式20轮的既定策略，五个批均值的中位数为：cuSignal Python GPU
**28.937357 ms**、C++/CUDA GPU **11.027838 ms**，正式比值为 **2.624028**，即C++/CUDA
GPU更快。两者均使用FP32 `[512]`、采样率512 Hz、目标延迟96、seed `20260714`、相同融合
权重和峰值参数。双方不同职责的 `pipeline_total_ms` 不进入本表、图17、图19或加速比；
逐算子比值因高层分配与预分配边界不同，只作诊断。该表不是Python/CPU性能比较，C++路径中
用于精度判定的CPU reference也不是这里的被比较实现。100轮数据来源为
`04_原始证据/tasks/tasks_chart_b027dca_20260719/chart_tasks_b027dca_20260719.tar.gz`。

本批验收未使用 diPTI。正式时间由自编 CUDA Event 和 Host 墙钟产生，精度由 C++ 独立 CPU
validation/comparison 产生；两者均不依赖 profiler 判定。
