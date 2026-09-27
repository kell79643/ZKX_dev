# 性能评价指标规范

本文统一规定算子与 Task1/Task2 的正确性门禁、计时作用域、三个加速比、统计方法和通过标准。评价顺序固定为：

```text
确认结果身份和评价键
  → 正确性硬门禁
  → 确认计时层级与 scope
  → 采集性能统计
  → 在同一评价键内计算加速比
  → 检查收益是否传递到 Task
```

正确性和性能不做加权评分。只有正确性硬门禁通过后，耗时、吞吐量和加速比数据才有效。

## 文档职责与协作关系

| 文档 | 唯一权威职责 | 接收的输入 | 必须产出的结果 |
|---|---|---|---|
| [性能评价指标规范](PERFORMANCE_EVALUATION_METRICS.md)（本文） | 定义“测什么、怎么算、什么条件下可比、怎样才算通过” | 优化计划给出的对象、固定用例和目标层级 | 评价键、正确性结论、计时 scope、统计量、三个加速比和 pass/fail |
| [性能优化实施计划](OPTIMIZATION_PLAN.md) | 定义“优化什么、为什么优化、优先级、实施顺序和阶段目标” | 当前基线、瓶颈数据和本文的评价结论 | P0～P4 优化切片、固定场景、预期收益和 Task 传递要求 |
| [优化说明与证据留档规范](OPTIMIZATION_EXPLANATION_GUIDE.md) | 定义“哪些过程和证据必须长期保留、怎样才能称为完成” | 优化前后代码、本文生成的指标结果、计划中的目标 | 八项证据索引、主证据文档、原始数据定位、失败历史和完成状态 |

本文是指标名称、公式、统计参数、计时 scope 和性能门禁的唯一权威来源，不负责选择下一个优化对象，也不规定证据文档的留档结构。优化对象和顺序必须回到 [性能优化实施计划](OPTIMIZATION_PLAN.md)；实验结果如何归档、同步和判定“已完成”必须回到 [优化说明与证据留档规范](OPTIMIZATION_EXPLANATION_GUIDE.md)。

三份文档形成以下闭环：

```text
OPTIMIZATION_PLAN 选择优化切片和固定场景
  → PERFORMANCE_EVALUATION_METRICS 生成同口径结果并判定是否通过
  → OPTIMIZATION_EXPLANATION_GUIDE 固化原始问题、修改、结果和边界
  → 证据结论回填 OPTIMIZATION_PLAN，决定下一优先级
```

## 一、评价对象、结果身份与比较基准

### 1. Python、CPU、GPU 四种结果身份

项目中的结果必须明确写出实现和用途，不能只使用含糊的 `python`、`cpu`、`gpu`：

| 结果身份 | 正式名称 | 主要用途 | 性能可比条件 |
|---|---|---|---|
| Python 语义结果 | 上游 Python cuSignal 23.08.00 语义参考 | 公开语义、shape、dtype 和数值正确性参考 | 若运行在 NVIDIA 或不同环境，只能作为正确性或跨平台参考 |
| Python GPU 性能结果 | ZQ500 cuSignal Python GPU | 与同一 ZQ500 上 C++/CUDA GPU 做 Task 同步骤性能比较 | 必须同输入、同 Step、同 `formal_execution_ms` 边界 |
| CPU 结果 | 同服务器 C++ CPU reference | GPU 正确性参考及独立算子 CPU/GPU 加速比 | 必须与 GPU 使用同一算子、输入、参数和输出范围 |
| GPU 结果 | ZQ500 C++/CUDA GPU | 当前正式算子或任务实现 | 必须声明 kernel-only、device-resident、compatibility 或 Task scope |

### 2. 四种比较基准的分工

- 固定版本 `cusignal-23.08.00` 的上游 Python 实现：用于公开语义和正确性参考。这里的
  “参考”表示验证角色，不代表另有一份库或版本。
- 同服务器 C++ CPU reference：用于主要 CPU/GPU 性能加速比。
- 优化前 clean GPU commit：用于证明具体优化带来的收益。
- 语义相同的平台库函数：FFT 等算子可与 ZQ500 dlFFT 等平台库比较实现差距。

如果 Python cuSignal 运行在 NVIDIA、C++ GPU 运行在 ZQ500，就不能写成“同硬件公平加速比”。可以保留对比，但必须标出硬件、软件版本和测试环境，将其称为“跨平台参考”，不能作为主要性能结论。

### 3. 统一评价键

三个加速比不是三个可以任意组合的数字。每条性能记录和每个加速比都必须绑定评价键 `E`：

```text
E = {
  evaluation_level, operator_or_task, task_step,
  input_shape, dtype, parameters,
  timing_scope, synchronization,
  warmup, iterations, hardware, software_environment
}
```

除“实现身份”或“优化前后版本”这一被比较变量外，分子和分母的评价键必须一致。

### 4. 同一算子在不同任务中的耗时

同一算子在独立 benchmark、Task1 和 Task2 中的耗时不要求相同。输入规模、前后算子、缓存、预分配、workspace/plan 复用、stream、同步及 H2D/D2H 都可能不同。因此每个场景必须产生独立记录：

```text
operator/standalone/device-resident
operator/standalone/compatibility-end-to-end
Task1/StepN/formal_execution
Task2/StepN/formal_execution
Task/pipeline/end-to-end
```

只允许在相同记录层级内部计算加速比。即使算子名称相同，也禁止跨 Task、跨 Step、跨输入规模或跨计时作用域相除。

独立算子 benchmark 是评价算子本身的主证据；Task1/Task2 用于回答优化是否传递到真实流水线。两类结果必须同时保留，不能互相替代。单算子优化完成后，应先报告独立算子收益；只有该算子确实位于任务链路，并完成相同 Task 层级的优化前后复测，才能继续报告 Task 净收益。

## 二、正确性硬门禁

### 1. 通用通过条件

性能数据只有在以下条件全部通过后才有效：

- shape、dtype、输出数量和参数语义通过。
- 全元素 `atol + rtol × abs(reference)` 比较通过。
- mismatch=0。
- NaN/Inf、边界和异常行为通过。
- 零误差项完成路径独立、人工扰动、独立复算三审。

正确性不与性能做加权评分。

### 2. 硬指标计算定义

当前项目以这些硬指标为主：

```text
abs_error  = abs(gpu - reference)
tolerance  = atol + rtol * abs(reference)
mismatch   = count(abs_error > tolerance)
```

同时报告：

- `max_abs_error`
- `max_rel_error`
- RMSE 或归一化 RMSE
- mismatch 数量和比例
- NaN/Inf
- shape、dtype、输出数量
- 算子特定语义结果

### 3. 按算子类型选用质量指标

#### 3.1 波形、滤波与重采样类

如 `firfilter`、`lfilter`、`sosfilt`、`detrend`、`decimate`、`upfirdn`：

- 主指标：全元素容差、最大绝对/相对误差、RMSE。
- 辅助指标：Pearson、余弦相似度。
- PSNR 只有在明确统一峰值定义后才使用。

#### 3.2 FFT、频谱、相关、脉冲压缩类

- 主指标：复数全元素误差、相对 L2 误差、幅值误差、相位误差。
- 辅助指标：频谱幅值的余弦相似度或 Pearson。
- PSNR 可用于固定范围的 spectrogram，但不能替代复数误差。

#### 3.3 CFAR、峰值检测、极值索引类

Pearson、余弦相似度和 PSNR 都不适合作为主要指标。应报告：

- 检测 mask 完全一致性
- mismatch 数量
- 峰值索引偏差
- TP、FP、FN
- 阈值误差

#### 3.4 `unit_impulse`、窗口和滤波器系数生成

- `unit_impulse` 应检查 shape、dtype、冲激位置、0/1 全元素一致性。
- 窗口和 FIR 系数应使用全元素误差、对称性、系数和等性质。
- 相似度接近 1 在这里没有足够证明力。

#### 3.5 复数输出

不应直接把复数数组随意转成实数计算 Pearson。必须明确采用：

- 实部、虚部分别计算；或
- 幅值和相位分别计算；或
- 明确定义复数相关系数。

Pearson、余弦相似度和 PSNR 是数值正确性或结果质量的辅助指标，不是性能指标，也不能覆盖全元素硬门禁失败。

## 三、计时层级与 Scope

### 1. 三种算子计时口径和任务计时

每个重点算子至少区分三种算子口径；进入任务流水线后，再单独报告 Task 口径：

| 层级 | 定义 | 用途 |
|---|---|---|
| `kernel-only` | CudaEvent 记录 kernel | 只用于分析 kernel |
| `device-resident API` | device 输入、device 输出，包含 launch 和完成同步 | 单算子主指标 |
| `end-to-end/compatibility` | 包含必要的 H2D/D2H 和最终 Host 输出 | 公开兼容入口，单独报告 |
| `Task Step/formal_execution` | 同一 Task Step 的正式执行边界 | Task 分步比较 |
| `Task pipeline/end-to-end` | 完整任务流水线总耗时 | 验证单算子收益是否传递到业务链路 |

这样老师质疑“为什么还有很多 H2D/D2H”时，可以直接从 scope 字段回答，而不是靠解释。

### 2. Scope 必记字段

每条记录必须明确是否包含：

- allocation
- H2D
- D2H
- workspace creation
- plan/handle creation
- postprocess
- stream 与同步位置

历史 265 项中，不同算子的 wrapper 可能包含内部 allocation、H2D、D2H、plan 或 postprocess。必须按每条记录的 scope 标注为 device-resident 或 compatibility end-to-end，不能把历史 265 项全部称为 resident。

## 四、单算子性能指标

每个 `operator × dtype × input size` 必须记录：

| 指标 | 定义 | 用途 |
|---|---|---|
| GPU resident mean | device 输入、device 输出，完成同步后的平均耗时 | 最主要的 API 性能 |
| GPU p95 | 正式迭代的 95 分位耗时 | 尾延迟 |
| GPU median/p99 | 正式迭代的中位数和 99 分位耗时 | 中心值与极端尾延迟 |
| GPU min/max/stddev | 正式迭代的最小值、最大值和标准差 | 范围与离散程度 |
| CV | `stddev / mean` | 抖动 |
| p95/p50、失败率 | 尾延迟相对中心值及运行失败比例 | 稳定性 |
| CPU mean | 同机器、同输入、同参数的 C++ CPU reference | 主要对比基准 |
| CPU/GPU speedup | `CPU mean / GPU resident mean` | 是否真正快于 CPU |
| GPU 优化加速比 | `旧 GPU mean / 新 GPU mean` | 证明本次优化收益 |
| 吞吐量 | `work items / GPU mean` | 同算子、同规模比较；必须明确分母是 elements、samples 还是 frames |
| 资源开销 | 峰值显存、分配次数、传输字节、kernel 数 | 解释为什么变快 |
| profiler | kernel/API/传输/同步占比、硬件计数器 | 只解释瓶颈，不替代正式 benchmark |
| Scope | allocation、H2D、D2H、workspace、plan、postprocess | 判断到底测了什么 |

## 五、三个加速比

### 1. 统一公式与评价情境

```text
CPU/GPU 加速比 S_cpu_gpu(E)
    = T_cpp_cpu_reference(E) / T_cpp_cuda_gpu(E)

GPU 优化加速比 S_opt(E)
    = T_gpu_before(E) / T_gpu_after(E)

cuSignal Python GPU + 必需 ZQ500 adapter / C++/CUDA GPU 加速比 S_py_cpp(E)
    = T_cusignal_python_gpu_plus_required_zq500_adapter(E) / T_cpp_cuda_gpu(E)
```

对应简写为：

```text
CPU/GPU 加速比                      = CPU_reference_mean / 当前_GPU_mean
GPU 优化加速比                      = 优化前_GPU_mean / 优化后_GPU_mean
cuSignal Python GPU + adapter / C++/CUDA GPU = Python_GPU_with_adapter_formal_mean / C++_CUDA_GPU_formal_mean
```

这里的分子是未修改 cuSignal 23.08 的公开 Python/GPU 算子执行，加上为 ZQ500 后端不兼容点
提供的必需 adapter；adapter 的实际设备执行、分配和同步开销必须计入分子，只有与双方一致的
预热/JIT 阶段可以排除。分母是相同输入、参数和职责范围内的 C++/CUDA GPU 正式执行区间，
不得混入 CPU validation、CPU comparison 或只存在于一方的语义后检查。该比值大于 1 表示
C++/CUDA GPU 更快，小于 1 表示 cuSignal Python GPU 加必需 adapter 更快。

| 加速比 | 正式使用情境 | 不得替代的结论 |
|---|---|---|
| CPU/GPU 加速比 | 同一独立算子、同一输入规模、同一参数、同一输出范围和明确计时作用域下，比较 C++ CPU reference 与当前 C++/CUDA GPU | 不能用 Task Step 耗时替代独立算子 CPU/GPU，也不能把不同 scope 的 CPU、GPU 耗时相除 |
| GPU 优化加速比 | 同一独立算子、同一 Task Step 或同一 pipeline 层级中，以完全相同计时边界比较优化前和优化后 GPU | Python GPU/C++ GPU 不是优化前后；单算子收益也不是 Task 收益 |
| cuSignal Python GPU + 必需 ZQ500 adapter / C++/CUDA GPU 加速比 | 当前主要用于同一 ZQ500 上 Task1/Task2 相同步骤的逐步 `formal_execution_ms` 比较；分子包含必需 adapter，分母不含 CPU 验证/比较 | 不能外推为 CPU/GPU、kernel-only 或优化前后加速比 |

三个加速比必须同时保留、明确命名，不能混写。`CPU/GPU 加速比`和`GPU 优化加速比`都不是只能在 Task1/Task2 场景中使用。

### 2. CPU/GPU 加速比

该指标回答当前 C++/CUDA GPU 是否真正快于同服务器 C++ CPU reference。正式主结论使用同一独立算子、同一输入规模、同一参数和相同输出范围；GPU 计时必须写明是 device-resident 还是 compatibility end-to-end。

官方代表性规模下要求 `CPU/GPU speedup > 1`；竞赛展示目标建议至少 `≥1.2×`，避免轻微抖动后跌回 CPU 以下。

### 3. GPU 优化加速比

该指标回答本次优化相对优化前 clean GPU 基线产生了多少收益。它可以用于独立算子、Task Step 或完整 pipeline，但前后必须保持相同评价层级、输入、参数、硬件、统计方案和计时边界。

优化前后 GPU mean 应改善至少 10%，且收益大于跨批波动；p95 不回退。单算子收益只有进入正式 Task 链路并完成同 Task、同 Step、同边界的 before/after 后，才能计入 Task 加速。

### 4. cuSignal Python GPU + 必需 ZQ500 adapter / C++/CUDA GPU 加速比

`semi_final_tasks/交付` 中存在同一 ZQ500 GPU 上两个 GPU 实现之间的第三种加速比。其名称、方向和公式冻结为：

```text
cuSignal Python GPU + 必需 ZQ500 adapter / C++/CUDA GPU 加速比
    = (cuSignal Python GPU + 必需 ZQ500 adapter) formal_execution_ms mean
    / C++/CUDA GPU formal_execution_ms mean
```

- 分子：同一步骤的 Python `formal_execution_ms mean`，必须包含真实 cuSignal API、任务必须的
  GPU 辅助计算、H2D/D2H，以及为绕开 ZQ500 不支持 dtype/FFT 路径而位于任务私有 `utils/`
  中的全部必需 adapter；不得只截取纯 cuSignal API 时间。
- 分母：同一步骤 C++/CUDA GPU 的 `formal_execution_ms mean`，不包含 CPU reference、CPU
  validation 或 CPU/GPU accuracy comparison。

- 比值 `> 1`：`C++/CUDA GPU` 较快，数值就是其相对“cuSignal Python GPU + 必需
  ZQ500 adapter”的加速倍数。
- 比值 `= 1`：两者耗时相同。
- 比值 `< 1`：“cuSignal Python GPU + 必需 ZQ500 adapter”较快；其加速倍数为该比值的倒数。不得在不说明求倒数和比较方向的情况下，把原比值直接称为 `C++/CUDA GPU` 加速比。

该指标只有在以下条件同时成立时才可作为正式性能结论：

- 双方均在同一 ZQ500 GPU 上运行，而不是 NVIDIA Python GPU 与 ZQ500 C++/CUDA GPU 的跨平台比较。
- 输入、shape、dtype、参数、seed、预热次数和正式采样次数一致。
- 双方计时边界一致，且正确性或任务效果门禁均通过。
- Task1/Task2 的主比较项使用逐步 `formal_execution_ms`。
- pipeline 只有在双方职责和验证开销一致时才能计算正式端到端加速比；职责不一致时只能作为补充记录。
- 逐算子比较只有在高层分配与预分配边界一致时才能作为正式加速比，否则只作诊断。

现有同硬件证据与口径来源：

- [Task1 cuSignal Python ZQ500 对照结果](../../../semi_final_tasks/交付/TASK1_SEMIFINAL_TECHNICAL_RESULTS.md#164-1-次预热--10-次正式采样性能)
- [Task2 cuSignal Python GPU 与 C++/CUDA GPU 同输入比较](../../../semi_final_tasks/交付/TASK2_SEMIFINAL_TECHNICAL_RESULTS.md#133-cusignal-python-gpu-与-ccuda-gpu-同输入比较)

### 5. 三方结果不要求每次都同时参与性能比较

- 独立算子 CPU/GPU 记录必须有 C++ CPU 和 C++/CUDA GPU；固定版本
  `cusignal-23.08.00` 的 Python 结果通常只作上游语义参考。
- Task Python GPU/C++ GPU 记录必须有双方 GPU 的同步骤耗时；CPU validation 即使存在，也不能自动进入该加速比。
- GPU 优化记录必须有同层级的优化前和优化后 GPU；Python 和 CPU 只作为正确性或外部基准，不能代替历史 GPU 基线。

## 六、正式统计方案

### 1. 上游方法与历史建议

cuSignal 23.08 官方 benchmark 使用 SciPy CPU 与 cuSignal GPU 对照，采用 pytest-benchmark、同步 GPU、比较 mean，并按输入规模和参数组织测试。这个方法可以继承，但 NVIDIA V100/A100 的绝对耗时不能作为 ZQ500 的性能验收线。[cuSignal 官方仓库与 benchmark 说明](https://github.com/rapidsai/cusignal#benchmarking)

本地上游配置为 warmup 10 次、最少正式轮次 25 次、关闭 GC，并主要按 mean 排序。计划文档旧稿还曾建议单算子 warmup 10 次、至少正式测量 30 次，亚毫秒算子每个样本内部循环 100～1000 次并扣除循环空开销，Task1/Task2 warmup 1～3 次且至少正式运行 10 次。这些内容保留为方法背景和历史证据解释，不再构成另一套正式合同。

### 2. 当前项目统一正式合同

为避免三份文档出现互相冲突的采样参数，当前正式统计方案冻结为：

- 每批 warmup 5 次。
- 每批正式运行 20 次。
- 至少 5 个独立批次，共 100 个正式样本。
- 主中心值：五批 mean 的中位数。
- 同时报告：批间最小/最大 mean、pooled p95、最大 CV。
- 不删除异常点；异常批次单独解释。
- 原始迭代或至少每批完整统计必须保存。

对于小于约 0.1 ms 的算子，20 次可能过少，应增加迭代数，但仍保持“每次调用后同步”的正式 API 口径。CV 优先控制在 0.1 以下，超过时必须披露并补长稳测试。

既有 Task1/Task2 `1+10` 或其他历史证据可以保留，但必须明确标注为既有证据口径，不能与当前统一正式批次直接合并。若官方验收程序强制其他轮次，以官方口径为准，并在评价键中记录偏离原因。

跨多个算子汇总加速比使用几何平均，不能对不同算子的加速比直接取算术平均。

## 七、输入规模与 crossover

重点算子采用 1K、10K、100K、1M 四档长度，或采用符合该算子实际输入结构的等价规模档位。当前 53×5 benchmark 主要是固定代表性用例，尚不能替代多规模 crossover 测试。

建议记录：

```text
N* = 最小的、GPU p95 < CPU mean 的测试规模
```

最终不只回答“GPU 是否快”，还回答“从多大规模开始稳定快于 CPU”。

### 允许设置“展示重点规模”

完成全部预定规模的 crossover 测试后，可以从正确性、计时 scope 和稳定性门禁均通过的
规模中，选择一个 C++/CUDA GPU 相对 CPU reference 或 cuSignal Python GPU 优势较大的
数据规模，作为技术文档、PPT 和答辩中的**展示重点规模**。这样做的直接目的，是让核心优化
收益更突出、对比数据和图表更直观，也就是使展示数据“更好看”、更容易让评委快速看清 C++
版本的性能优势。

“重点讲述”不等于“只讲述”。使用展示重点规模时必须同时满足：

- 先按预先约定的 1K、10K、100K、1M 或等价档位完成全规模测试，再选择重点规模；不能
  测到某个最好结果后反向删除其他规模。
- 同一张表或相邻材料保留全部规模的 mean、p95、CV、吞吐量和适用加速比，并标出 crossover
  `N*`；正文可以重点分析其中一个规模，但其他规模数据必须可查。
- 明确写出选择理由，例如“该规模已越过 kernel launch crossover，能够体现 device-resident
  C++/CUDA 路径的并行和带宽优势”，不能只写“选取最优结果”。
- 重点规模必须使用与其他规模相同的评价键规则、正确性门禁、预热、正式轮次、同步和统计
  方法，不得为重点规模单独改变输入内容、计时边界或剔除异常批次。
- 同时披露小规模下可能因 launch、同步和固定开销而优势较小甚至慢于 CPU 的事实，并说明
  适用范围；不能把重点规模的加速比外推到全部输入规模。
- 如果赛题、官方验收或真实 Task 已规定代表性规模，官方规模仍是主验收结果；另选的优势
  规模只能作为补充亮点，不能替代官方规模。

推荐展示结构为“全规模曲线或表格 + 重点规模放大说明 + crossover 结论”。这样既能突出
C++/CUDA 版本最好、最有竞争力的性能区间，又保留完整数据，避免只挑优势点造成不公平比较。

## 八、Task1/Task2 指标

重点算子的优化如果进入正式任务，还必须报告：

- `formal_execution_ms`。
- Task 总耗时 median/p95。
- 各 Step 耗时及占比。
- H2D、compute、D2H 分项。
- GPU 显存峰值。
- 优化前后端到端加速比。
- 单算子收益是否真实传递到 Task。

若算子不在任务链路中，只报告单算子收益，不计入 Task 加速。若算子进入 Task1/Task2，端到端必须观察到可解释的净收益。

## 九、性能通过标准

后续优化采用以下门禁：

- 正确性和语义门禁全部通过。
- device-resident 稳定阶段内部 H2D=0、D2H=0；不变量场景 allocation/plan/workspace creation=0。
- 官方代表性规模下 `CPU/GPU speedup > 1`；竞赛展示目标建议至少 `≥1.2×`。
- 优化前后 GPU mean 改善至少 10%，且收益大于跨批波动。
- p95 不回退；CV 优先控制在 0.1 以下，超过时必须披露并补长稳测试。
- 若算子进入 Task1/Task2，端到端必须观察到可解释的净收益。
