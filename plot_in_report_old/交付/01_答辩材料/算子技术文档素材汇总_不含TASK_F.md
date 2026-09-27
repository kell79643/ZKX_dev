# 复赛技术文档算子素材汇总（不含 TASK_F）

> 用途：为复赛技术文档、Word/PDF、PPT、图表和答辩说明提供算子相关素材。
>
> 边界：本文不收录 TASK_F 生成的 53×5 正确性、精度和性能正式结果。TASK_F 的原始日志、265 项误差、265 项性能和零指标三审应在其完成后另写专门文档。本文只整理算子范围、源码架构、接口与类型设计、cuSignal 语义依据、CPU reference、ZQ500 平台适配，以及 TASK_F 之前已有且能够说明技术路线的局部结果。
>
> 版本提醒：本文是“技术文档素材”，不是最终唯一证据索引。最终引用前必须通过 TASK_J/K 将结果重新绑定到冻结的 Git commit、未提交状态、实际构建源码、ZQ500 环境、命令、日期和原始输出。
>
> 当前性能补充：`unit_impulse` P0 的实现、指标和证据状态见
> [unit_impulse P0 当前状态](../03_当前状态/UNIT_IMPULSE_P0_CURRENT_STATUS.md)；本文后续接口表只承担素材
> 索引职责，不重复性能结论。

## 1. 编制依据和来源等级

### 1.1 权威要求

本文字段和判定边界来自：

1. ZKX/semi_final_tasks/LATEST_SOURCE_PLATFORMIZATION_REQUIREMENTS.md：最新源码验收、53 个算子、五类型、cuSignal 对齐、CPU reference、精度、性能、证据和技术文档要求。
2. ZKX/semi_final_tasks/TARGET_EFFECT_ZQ500_PLATFORMIZATION.md：ZQ500 平台迁移目标、CUDA API/科学库/CMake、算子封装、template、五类型、独立测试和交付证据要求。
3. ZKX/semi_final_tasks/TASK_K_DELIVERY_EVIDENCE.md：交付冻结、Word/PDF、图表、PPT、渲染检查和最终平台门禁。

### 1.2 来源等级

| 等级 | 含义 | 技术文档引用方式 |
| --- | --- | --- |
| A：当前源码事实 | 可从当前 cusignal_cpp/src、CMake 和契约文件直接复核 | 可陈述“当前实现/当前接口”，最终仍需绑定冻结 commit |
| B：当前 ZQ500 实测 | 在 gpu_02 初始化 SDK 后，对明确源码快照完成清理构建和运行 | 可作为当前结果，但必须附 commit、命令、日志和日期 |
| C：历史局部结果 | 对应旧接口、旧 commit、旧参数域或阶段性 smoke | 只能说明演进、问题定位或方法，不得冒充最终结果 |
| D：要求或设计 | 需求文档、静态设计、计划、未运行矩阵 | 只能写“要求/设计为”，不能写“已经验证” |

来源冲突时，采用“当前源码与当前 ZQ500 原始输出优先于历史文档状态”的原则。接口、算法、kernel、dtype、shape、布局、科学库、CMake、CPU reference、测试或容差变化后，受影响旧结果进入 needs-revalidation。

## 2. 本次整理的源码快照

首次整理时读取到：

- 分支：feature/current-workflow；
- 整理前 HEAD：4f11a287fb295a8e11a8c5f05fdcffd8286ed87c；
- 日期：2026-07-15；
- 工作区存在 TASK_F、TASK_H 和其他开发者的并行未提交修改，因此该 HEAD 不是最终交付冻结版本。

最终文档应由 TASK_J/K 将这里替换为冻结候选 commit、未提交 diff 标识和源码 manifest。

来源：本地只读 git branch --show-current、git rev-parse HEAD、git status --short；环境要求来自 ZKX/AGENTS.md。

## 3. 算子范围和源码架构

### 3.1 固定范围

当前正式范围为 53 个业务算子、12 个类别。五种要求输入类型为 FP32、FP16、INT32、INT16、INT8，形成 53×5=265 个输入类型契约。

| 类别 | 算子数 | 算子 |
| --- | ---: | --- |
| bsplines | 3 | cubic、gauss_spline、quadratic |
| convolution | 2 | correlate、correlate2d |
| demod | 1 | fm_demod |
| estimation | 1 | KalmanFilter |
| filter_design | 2 | firwin、firwin2 |
| filtering | 14 | channelize_poly、detrend、firfilter、firfilter2、freq_shift、hilbert、hilbert2、lfilter_zi、sosfilt、wiener、decimate、resample、resample_poly、upfirdn |
| peak_finding | 1 | argrelextrema（含 argrelmin/argrelmax 正式便捷入口） |
| radartools | 5 | ambgfun、ca_cfar、cfar_alpha、pulse_compression、pulse_doppler |
| spectral_analysis | 6 | csd、istft、lombscargle、spectrogram、stft、vectorstrength |
| waveforms | 5 | chirp、sawtooth、square、gausspulse、unit_impulse |
| wavelets | 5 | cwt、morlet、morlet2、qmf、ricker |
| windows | 8 | chebwin、general_cosine、general_gaussian、hamming、kaiser、parzen、taylor、triang |
| 合计 | 53 | 固定业务集合 |

来源：ZKX/cusignal_cpp/contracts/operator_contracts.cmake 的 CUSIGNAL_OPERATOR_CONTRACTS。本文整理时逐行解析得到 53 行；该文件同时固定 cuSignal 23.08.00、CuPy 13.6.0 和五种要求输入类型。

### 3.2 统一物理结构

12 个业务类别采用统一组织：

~~~text
cusignal_cpp/src/<module>/
  <module>_typed.h       正式公开 GPU API 和 CPU reference 声明
  <module>_typed.cpp     host wrapper、CPU reference、参数/shape/返回结构处理
  <module>_typed.cu      GPU 实现、显式实例化和设备侧类型路径
  <module>_kernels.cuh   kernel 与设备辅助实现
~~~

公共设施：

- src/fft_interface：FFT 抽象及 ZQ500 dlfft 后端；
- src/rand_interface：独立 RAND 封装；53 个业务算子不直接依赖随机库；
- src/cuda_utils：设备数组、缓冲、拷贝、错误检查、launch、shape、类型契约、dispatch 和 host 输出收尾；
- src/signal_processing.h：聚合包含类别公开头文件，不承载旧 double 算法主体。

来源：当前 ZKX/cusignal_cpp/src 文件清单；命名和封装要求来自两份总体要求文档。

### 3.3 正式封装

调用者只依赖公开头文件，不直接包含 cpp/cu。GPU 正式入口以 *_device 为主，CPU reference 以 *_typed_cpu 为主；B-spline CPU reference 为 cubic_cpu、gauss_spline_cpu、quadratic_cpu。多个 overload 或阶段不增加业务算子数，例如 KalmanFilter 的 predict/update 共同构成一个业务算子。

2026-07-13 的公开声明门禁曾输出：

~~~text
[E7][SUMMARY] operators=53 gpu_declarations=65 cpu_declarations=54 documented=119 missing=0
~~~

这只验证对应快照的声明清单和中文 Doxygen 字段，不验证数值或性能。当前源码增删 overload 后必须重跑。

来源：ZKX/cusignal_cpp/docs/E7_PUBLIC_API_TYPE_DOCUMENTATION_EVIDENCE.md；脚本 ZKX/cusignal_cpp/cmake/verify_e7_public_docs.cmake；目标 e7_public_api_docs_gate。来源等级：B/C。

## 4. 五类型、输出类型和计算类型

### 4.1 准确定义

“五类型支持”是指正式业务入口接收 FP32、FP16、INT32、INT16、INT8 五种真实输入。输出 dtype、shape、返回值数量和数学行为按固定版 cuSignal 的可观察语义确定，不能统一规定输出等于输入。中间计算类型不要求与输入或输出一致。

统一类型链：

~~~text
InputT
  -> 显式 load/convert
  -> ComputeT
  -> 数学函数、custom kernel 或 ZQ500 科学库
  -> ComputeResultT
  -> 显式 store/host finalize
  -> cuSignalContractOutputT
~~~

来源：LATEST_SOURCE_PLATFORMIZATION_REQUIREMENTS.md 第九、十章；OPERATOR_PLATFORMIZATION_E3_E8_MATH_TYPE_POLICY.md；operator_contracts.cmake。

### 4.2 典型输出规则

| 场景 | 输入示例 | 输出 | 语义来源 |
| --- | --- | --- | --- |
| 同型输出 | cubic 五类型输入 | 对应 FP32/FP16/INT32/INT16/INT8 | 契约 EV-BS-CUBIC |
| 连续 FP32 输出 | fm_demod 五类型输入 | FP32 | EV-DEMOD、EV-CUPY-ANGLE/UNWRAP/DIFF |
| 复数 FP32 输出 | hilbert 的 FP32/FP16 输入 | ComplexFP32 | EV-HILBERT、EV-CUPY-FFT |
| FP64 输出 | firwin 五类型输入 | FP64 系数 | EV-FIRWIN-DTYPE |
| ComplexFP64 输出 | freq_shift 五种实输入 | ComplexFP64 | EV-FREQ-SHIFT |
| 逻辑复合输出 | ca_cfar | FP32 threshold + BOOL detection | EV-CA-CFAR 系列 |
| 坐标集合 | argrelextrema | 与 rank 等长的 INT64 坐标数组集合 | EV-CUPY-NONZERO |
| 固定整数输出 | qmf 五类型输入 | INT64 | EV-QMF |

完整 265 行 InputT/ComputeT/OutputT/shape/return/finalize 规则位于 operator_contracts.cmake 的 CUSIGNAL_DTYPE_CONTRACTS。本文解析到 265 行，全部属于上述 53 个算子。

### 4.3 template 与等价复用

算子通过 template、traits、显式实例化或等价 dispatch 避免复制五套算法。静态设计分类：

| 方案 | 算子数 | 含义 |
| --- | ---: | --- |
| T | 35 | 通用 template/traits 与显式实例化 |
| FD | 14 | typed 入口加统一 FFT 类型转换/dispatch |
| ET | 2 | 扩展已有 template 机制 |
| S | 2 | 明确专用 dispatch，避免错误的统一算术模板 |
| 合计 | 53 | 每个算子均有非空复用方案 |

来源：ZKX/cusignal_cpp/docs/OPERATOR_PLATFORMIZATION_E2_TYPE_REUSE_AUDIT.md。该表属于设计/静态审计；实际实现状态由当前源码和 ZQ500 测试确认。

## 5. FP64 输出与设备端 FP64 限制

ZQ500 限制针对 GPU 设备端计算：正式 kernel 不能使用 double、ComplexDouble 或依赖 FP64 科学库完成算法。它不禁止接口按 cuSignal 规则返回 FP64/ComplexFP64。

项目策略：

~~~text
五类型输入
  -> GPU 端 FP32/ComplexFP32 计算
  -> 同步并 D2H
  -> CPU 端 FP32 -> FP64/ComplexFP64 拓宽
  -> 返回 host 结果，或接口明确要求 device 结果时可选 H2D
~~~

转换必须记录同步点、临时缓冲、结果可读时机、D2H/H2D和性能计时范围，不能隐藏 host 拓宽成本。

当前 265 行契约中，本文解析到 99 行固定 FP64 输出、23 行固定 ComplexFP64 输出，另有按参数或分支动态返回 FP64/ComplexFP64 的规则。这是契约统计，不是精度或性能结果。

来源：总体要求中的 FP64 条款；operator_contracts.cmake；ZKX/cusignal_cpp/src/cuda_utils/host_output_finalize.h；ZKX/cusignal_cpp/CMakeLists.txt 对正式 GPU 源的 FP64 token 门禁。

## 6. cuSignal/CuPy 语义追踪

### 6.1 固定参考

- cuSignal：23.08.00，本地参考仓位于 ZKX/ 同级的 cusignal-23.08.00；
- CuPy：契约固定版本 13.6.0；
- CUSIGNAL_CONTRACT_EVIDENCE 为每条证据记录来源项目、版本、symbol、源码路径、行号、hash、原行为、项目解释和对应 CPU/GPU 符号。

来源：operator_contracts.cmake。

### 6.2 为什么继续追踪 CuPy

cuSignal 将 dtype、索引、FFT、result_type、nonzero和数学ufunc等行为委托给CuPy，只读cuSignal表层代码不足以确定最终输出。例如：

- argrelextrema -> cupy.nonzero：确定 rank 个 INT64 坐标数组、row-major 顺序和空结果；
- firfilter/firfilter2 -> cupy.result_type：确定类型提升；
- hilbert/stft/resample -> cupy.fft：确定 FFT 精度与复数类型；
- detrend -> cupy.mean：确定 FP16、FP32和整数规约dtype；
- fm_demod -> angle/unwrap/diff：确定相位边界、axis和输出shape。

技术文档中的“与cuSignal一致”应表述为：

> 以固定版cuSignal公开函数的可观察行为为主，继续追踪其调用的CuPy dtype、shape、索引和FFT规则；对cuSignal已确认缺陷，只有获得明确批准才对齐CuPy正确语义，并记录偏离原因。

### 6.3 可引用追踪案例

| 算子/规则 | cuSignal 源码 | CuPy 源码（适用时） | 项目结论 |
| --- | --- | --- | --- |
| argrelextrema | python/cusignal/peak_finding/peak_finding.py:21-228 | cupy/_core/_routines_indexing.pyx:54-120 | 返回 rank 个 INT64 坐标数组，不再把二维坐标编码为 flat index |
| ambgfun | python/cusignal/radartools/radartools.py:146-240 | norm、true_divide、absolute | 保留三种 cut 的 shape/dtype；已批准修复不等长 y 缺陷 |
| KalmanFilter | python/cusignal/estimation/filters.py:197-430 | 不适用 | 保留 predict/update、状态shape、Joseph协方差和错误语义 |
| firwin | python/cusignal/filter_design/fir_filter_design.py:96-396 | 不适用 | 五类型输入、FP64输出，GPU FP32计算后host拓宽 |
| detrend | python/cusignal/filtering/filtering.py:986-1093 | routines_statistics.pyx:639-655 | constant/linear 分支具有不同 dtype 提升 |
| qmf | python/cusignal/wavelets/wavelets.py:19-41 | 不适用 | 固定版忽略输入值，按长度/索引生成 INT64 |

精确hash和完整描述见契约中的EV-*行；最终引用时对冻结参考仓复核行号。

## 7. CPU reference 与误差来源

### 7.1 误差跟谁比较

GPU误差的主要比较对象是项目中的独立C/C++ CPU reference，而不是一句含糊的“与cuSignal比较”。cuSignal/CuPy用于冻结接口和算法语义，CPU reference按相同参数、dtype、shape和返回结构计算期望值，正式GPU API产生实际值，再比较两者。

~~~text
cuSignal/CuPy源码
  -> 冻结dtype、shape、返回结构、参数和算法语义
  -> 独立C/C++ CPU reference生成期望结果

同一业务输入
  -> 正式GPU API生成实际结果

CPU期望值 vs GPU实际值
  -> 返回数量/dtype/shape/finite
  -> 全元素数值误差或exact结构比较
~~~

### 7.2 独立性

CPU reference位于 *_typed.cpp 或公开头文件中的CPU模板，GPU计算位于 *_typed.cu/*_kernels.cuh。CPU reference不得调用GPU wrapper、读取GPU结果或把GPU输出转型后作为期望值。CPU/GPU可共享类型policy、参数结构和安全helper，但关键计算独立。

历史静态审计曾映射53个CPU reference，并确认 *_typed.cpp 未出现 cudaMalloc/cudaMemcpy/kernel launch/DeviceArray/__global__。来源：E4_TYPED_CPU_REFERENCE_EVIDENCE.md。该结论需由冻结源码复核。

### 7.3 指标定义

连续实数/复数输出通常使用：

~~~text
max_abs = max_i |gpu_i - cpu_i|
max_rel = max_i |gpu_i - cpu_i| / max(|cpu_i|, relative_floor)
~~~

复数差值取模，不能只比较实部。离散结构不得伪装成浮点误差：

- index、coordinate、mask、bool、one-hot、符号结构：mismatches/compared；
- dtype、shape、返回值数量：exact；
- ca_cfar：threshold做浮点比较，detection mask做exact；
- argrelextrema：逐维INT64坐标、数量和顺序做exact。

任意结果必须同时写明：GPU API、CPU reference/独立公式、输入输出dtype、case、shape、比较元素数、finite、指标、容差、commit、命令、日期和ZQ500日志。

### 7.4 非TASK_F容差设计基线

历史E5契约记录：

| 类别 | FP32/整数浮点输出 atol | FP16 atol | FP32/整数 rtol | FP16 rtol |
| --- | ---: | ---: | ---: | ---: |
| scalar | 1e-4 | 1.5e-4 | 1e-4 | 1e-3 |
| pointwise | 3e-2 | 4.5e-2 | 1e-4 | 1e-3 |
| filtering | 4e-2 | 6e-2 | 1e-4 | 1e-3 |
| fft | 6e-2 | 9e-2 | 1e-4 | 1e-3 |
| exact | 0 | 0 | 0 | 0 |

来源：ZKX/cusignal_cpp/docs/E5_TYPE_TOLERANCE_CONTRACT.md。这里只能解释精度预算设计；最终265项阈值、结果和零指标三审属于TASK_F专文。

### 7.5 零误差边界

浮点指标为零时必须排除共享结果缓冲、未比较全元素、全零/identity退化输入、只比较首元素或总和、打印精度不足、CPU/GPU同路径逐位复制等弱测试。正式零指标要求路径独立性、更换输入并人工注错、高精度逐元素独立复算三次审查。本文不收录TASK_F三审结果。

## 8. ZQ500 API、科学库与CMake

### 8.1 CUDA运行时

常用能力集中在 src/cuda_utils：

- cuda_error.h：错误检查；
- copy_utils.h：H2D、D2H、D2D和stream同步；
- kernel_launch.h：常规1D launch及launch后检查；
- runtime_utils.h：设备信息和event计时；
- device_array.h/device_buffer.h：资源生命周期；
- shape_utils.h：shape、axis和元素数安全处理。

复杂block/grid、shared memory、常量符号、FFT workspace允许直接调用，但要说明原因和参数范围。

来源：CUDA_OPERATOR_API_PLATFORMIZATION_MATRIX.md。该矩阵含历史路径，最终图表应按当前typed源码复核。

### 8.2 FFT

当前基线调用链：

~~~text
业务算子
  -> FFTInterface / FFTInterface2D（适用时）
  -> fft_dlfft.cu
  -> ZQ500 dlfft
~~~

非FP32输入显式提升到FP32/ComplexFP32。FFT长度、补零、batch、stride、workspace、in-place/out-of-place和plan生命周期按算子记录。

历史D5曾对15个FFT依赖算子运行统一smoke；当前correlate2d已改为boundary-aware direct实现，不再属于FFT依赖，因此旧D5截图不能证明当前correlate2d。

来源：OPERATOR_PLATFORMIZATION_D5_SCILIB_SMOKE.md。来源等级：C，用于迁移过程和后端设计，不是最终结果。

### 8.3 其他科学库

历史审计结论：53个业务算子不直接依赖BLAS、DNN、SOLVER、SPARSE；RAND保留为独立封装，不属于53个算子直接依赖。不能为了展示库覆盖强迫无依赖算子调用科学库。

来源：D5文档和NON_FFT_SCI_LIB_C5_MATRIX.md；最终由当前源码扫描复核。

### 8.4 CMake模式

ZQ500活动目标应：

- 在gpu_02执行source /zq500/sdk/env.sh；
- 容器实测CMake 3.22.1，最低版本保持3.16；
- project(... LANGUAGES CXX)；
- 从$DLICC_PATH设置C++编译器；
- .cu标记LANGUAGE CXX并使用-x cuda；
- 链接curt和实际需要的dlfft等平台库；
- 禁止LANGUAGES CUDA、nvcc、CMAKE_CUDA_*、sm_XX/compute_XX和/usr/local/cuda。

来源：三份权威要求、ZKX/AGENTS.md、TASK_G_ZQ500_CMAKE.md。最终零违规结论属于TASK_G/J/K。

## 9. 非TASK_F局部结果及引用边界

### 9.1 数学/device原语

2026-07-12记录：

~~~text
[E3][MATH][SUMMARY] total=5 pass=5 fail=0
~~~

覆盖FP16显式load、cosf/sinf/expf/powf/acoshf/logf/sqrtf/atan2f/hypotf、定义域、FP32 Bessel I0、float atomic、shared FP32 reduction、FP32 complex和FP16回转。只证明对应快照的基础原语，不证明53×5算子。

来源：OPERATOR_PLATFORMIZATION_E3_E8_MATH_TYPE_POLICY.md。来源等级：B/C。

### 9.2 公开API文档门禁

记录为53算子、65个GPU声明、54个CPU声明、119个中文说明块、missing=0。比较对象是registry/应有声明清单与公开头文件的实际声明及Doxygen字段，不产生算法误差。

来源：E7_PUBLIC_API_TYPE_DOCUMENTATION_EVIDENCE.md。来源等级：B/C。

### 9.3 历史D6 FP32

2026-07-10 D10历史结果：

~~~text
[D6][SUMMARY] total=53 pass=53 in-progress=0 fail=0
[D6][PASS] spectral_analysis::lombscargle
max_abs=1.42098e-17
max_rel=3.4874e-16
metric_source=device_vs_long_double_reference
~~~

lombscargle误差来自GPU结果与独立long-double reference的逐元素比较。连续数值输出记录max_abs/max_rel，离散结构记录mismatches/compared；D10禁止numeric双零。

来源：OPERATOR_PLATFORMIZATION_D6_FP32_ACCURACY.md:211-237和test_all/operators/d6_operator_fp32_accuracy.cu。它是旧FP32/旧接口局部证据，不能替代当前typed API或TASK_F五类型结果。来源等级：C。

### 9.4 历史D9边界算子

| 算子 | max_abs | max_rel | 比较说明 |
| --- | ---: | ---: | --- |
| correlate2d | 4.44089e-16 | 9.78709e-17 | 当时GPU实现 vs CPU/独立参考 |
| KalmanFilter | 6.44189e-08 | 7.61314e-08 | FP32 device vs double参考 |
| channelize_poly | 8.62265e-15 | 1.34198e-16 | 当时GPU实现 vs CPU/独立参考 |
| sosfilt | 1.11022e-16 | 1.91163e-16 | device vs 独立long-double生成器 |
| csd | 1.70185e-08 | 4.5249e-08 | 当时GPU实现 vs CPU/独立参考 |
| stft | 1.04257e-08 | 3.86053e-08 | 当时GPU实现 vs CPU/独立参考 |
| chirp | 1.11022e-16 | 1.11022e-16 | 当时GPU实现 vs CPU/独立参考 |
| cwt | 2.22045e-16 | 1.01404e-16 | 当时GPU实现 vs CPU/独立参考 |
| chebwin | 7.77156e-16 | 7.77156e-16 | 当时GPU实现 vs CPU/独立参考 |

~~~text
[D9][SUMMARY] total=9 pass=9 in-progress=0 blocked=0 fail=0
~~~

来源：TASK_D_OPERATOR_FP32_CLOSURE.md:811-834。该文档说明KalmanFilter采用FP32 device对double参考，sosfilt采用独立long-double参考。旧chebwin曾含后来禁止的device double路径，所以这些数字只能用于整改过程，不能证明当前正式GPU路径无device FP64。来源等级：C。

### 9.5 历史typed能力

2026-07-12九组typed业务smoke曾合计：

~~~text
15 + 10 + 5 + 20 + 15 + 70 + 30 + 30 + 70 = 265
~~~

每个业务入口实例化五种类型并与CPU reference比较。之后正式路径合并和接口整改使部分证据失效。

来源：E3_53_OPERATOR_IMPLEMENTATION_EVIDENCE.md。本文不将它写成当前最终265/265；正式结果归TASK_F专文。

## 10. 技术亮点素材

1. 统一typed架构：每类采用typed.h/cpp/cu和kernels.cuh。
2. 真实五类型入口：业务API直接接收五种类型，不以registry或转换helper冒充。
3. 输入/计算/输出解耦：逐算子契约驱动InputT、ComputeT和OutputT。
4. FP16/整数显式提升：进入FP32数学和FFT路径，避免隐式转换和device FP64。
5. cuSignal+CuPy追踪：追踪result_type、FFT、nonzero、mean和ufunc。
6. host拓宽保持语义：GPU FP32计算，host生成FP64/ComplexFP64输出。
7. 动态shape和结构返回：极值坐标tuple、CFAR threshold+mask等。
8. 独立CPU reference：先校验dtype/shape/finite，再做全元素数值或exact比较。
9. FFT统一后端：经FFTInterface进入ZQ500 dlfft，集中处理plan/workspace。
10. 证据防伪：numeric与exact分开，零误差三审，源码变化传播失效。

## 11. 建议图表及数据来源

| 图表 | 当前素材 | TASK_F后补充 |
| --- | --- | --- |
| 53算子类别分布 | 第3.1节12类计数 | 无 |
| typed源码架构图 | 第3.2节和当前src | 冻结commit |
| 五类型处理流程图 | 第4、5节 | 代表算子实际case |
| 输出dtype案例表 | 契约和第4.2节 | 265项实际输出 |
| ZQ500 FFT链路图 | 第8.2节 | 当前构建日志 |
| CPU/GPU比较流程 | 第7节 | 265项误差与三审 |
| 误差分布图 | 不用D6/D9冒充最终图 | TASK_F正式结果 |
| 性能/加速比图 | 本文不提供 | TASK_F正式benchmark |
| 整改案例 | chebwin、lombscargle历史过程 | 当前正式路径复验 |

## 12. 53个正式入口索引

下表来源于当前operator_contracts.cmake；路径相对ZKX/cusignal_cpp。

| 算子 | 类别 | GPU符号 | CPU reference | 公开头文件 | cuSignal符号 |
| --- | --- | --- | --- | --- | --- |
| cubic | bsplines | cubic_device | cubic_cpu | src/bsplines/bsplines_typed.h | cusignal.bsplines.cubic |
| gauss_spline | bsplines | gauss_spline_device | gauss_spline_cpu | src/bsplines/bsplines_typed.h | cusignal.bsplines.gauss_spline |
| quadratic | bsplines | quadratic_device | quadratic_cpu | src/bsplines/bsplines_typed.h | cusignal.bsplines.quadratic |
| correlate | convolution | correlate_device | correlate_typed_cpu | src/convolution/convolution_typed.h | cusignal.convolution.correlate |
| correlate2d | convolution | correlate2d_device | correlate2d_typed_cpu | src/convolution/convolution_typed.h | cusignal.convolution.correlate2d |
| fm_demod | demod | fm_demod_device | fm_demod_typed_cpu | src/demod/demod_typed.h | cusignal.demod.fm_demod |
| kalman_filter | estimation | kalman_predict_device、kalman_update_device | kalman_predict_typed_cpu、kalman_update_typed_cpu | src/estimation/estimation_typed.h | cusignal.estimation.KalmanFilter |
| firwin | filter_design | firwin_device | firwin_typed_cpu | src/filter_design/filter_design_typed.h | cusignal.filter_design.firwin |
| firwin2 | filter_design | firwin2_device | firwin2_typed_cpu | src/filter_design/filter_design_typed.h | cusignal.filter_design.firwin2 |
| channelize_poly | filtering | channelize_poly_device | channelize_poly_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.channelize_poly |
| detrend | filtering | detrend_device | detrend_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.detrend |
| firfilter | filtering | firfilter_device | firfilter_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.firfilter |
| firfilter2 | filtering | firfilter2_device | firfilter2_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.firfilter2 |
| freq_shift | filtering | freq_shift_device | freq_shift_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.freq_shift |
| hilbert | filtering | hilbert_device | hilbert_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.hilbert |
| hilbert2 | filtering | hilbert2_device | hilbert2_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.hilbert2 |
| lfilter_zi | filtering | lfilter_zi_device | lfilter_zi_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.lfilter_zi |
| sosfilt | filtering | sosfilt_device | sosfilt_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.sosfilt |
| wiener | filtering | wiener_device | wiener_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.wiener |
| decimate | filtering | decimate_device | decimate_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.decimate |
| resample | filtering | resample_device | resample_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.resample |
| resample_poly | filtering | resample_poly_device | resample_poly_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.resample_poly |
| upfirdn | filtering | upfirdn_device | upfirdn_typed_cpu | src/filtering/filtering_typed.h | cusignal.filtering.upfirdn |
| argrelextrema | peak_finding | argrelextrema_device、argrelmin_device、argrelmax_device | argrelextrema_typed_cpu、argrelmin_typed_cpu、argrelmax_typed_cpu | src/peak_finding/peak_finding_typed.h | cusignal.peak_finding.argrelextrema/argrelmin/argrelmax |
| ambgfun | radartools | ambgfun_device | ambgfun_typed_cpu | src/radartools/radartools_typed.h | cusignal.radartools.ambgfun |
| ca_cfar | radartools | ca_cfar_device | ca_cfar_typed_cpu | src/radartools/radartools_typed.h | cusignal.radartools.ca_cfar |
| cfar_alpha | radartools | cfar_alpha_device | cfar_alpha_typed_cpu | src/radartools/radartools_typed.h | cusignal.radartools.cfar_alpha |
| pulse_compression | radartools | pulse_compression_device | pulse_compression_typed_cpu | src/radartools/radartools_typed.h | cusignal.radartools.pulse_compression |
| pulse_doppler | radartools | pulse_doppler_device | pulse_doppler_typed_cpu | src/radartools/radartools_typed.h | cusignal.radartools.pulse_doppler |
| csd | spectral_analysis | csd_device | csd_typed_cpu | src/spectral_analysis/spectral_analysis_typed.h | cusignal.spectral_analysis.csd |
| istft | spectral_analysis | istft_device | istft_typed_cpu | src/spectral_analysis/spectral_analysis_typed.h | cusignal.spectral_analysis.istft |
| lombscargle | spectral_analysis | lombscargle_device | lombscargle_typed_cpu | src/spectral_analysis/spectral_analysis_typed.h | cusignal.spectral_analysis.lombscargle |
| spectrogram | spectral_analysis | spectrogram_device | spectrogram_typed_cpu | src/spectral_analysis/spectral_analysis_typed.h | cusignal.spectral_analysis.spectrogram |
| stft | spectral_analysis | stft_device | stft_typed_cpu | src/spectral_analysis/spectral_analysis_typed.h | cusignal.spectral_analysis.stft |
| vectorstrength | spectral_analysis | vectorstrength_device | vectorstrength_typed_cpu | src/spectral_analysis/spectral_analysis_typed.h | cusignal.spectral_analysis.vectorstrength |
| chirp | waveforms | chirp_device、chirp_complex_device | chirp_typed_cpu、chirp_complex_typed_cpu | src/waveforms/waveforms_typed.h | cusignal.waveforms.chirp |
| sawtooth | waveforms | sawtooth_device | sawtooth_typed_cpu | src/waveforms/waveforms_typed.h | cusignal.waveforms.sawtooth |
| square | waveforms | square_device | square_typed_cpu | src/waveforms/waveforms_typed.h | cusignal.waveforms.square |
| gausspulse | waveforms | gausspulse_device | gausspulse_typed_cpu、gausspulse_cutoff | src/waveforms/waveforms_typed.h | cusignal.waveforms.gausspulse |
| unit_impulse | waveforms | unit_impulse_device | unit_impulse_typed_cpu | src/waveforms/waveforms_typed.h | cusignal.waveforms.unit_impulse |
| cwt | wavelets | prepare_cwt_workspace、cwt_device | cwt_typed_cpu | src/wavelets/wavelets_typed.h | cusignal.wavelets.cwt |
| morlet | wavelets | morlet_device | morlet_typed_cpu | src/wavelets/wavelets_typed.h | cusignal.wavelets.morlet |
| morlet2 | wavelets | morlet2_device | morlet2_typed_cpu | src/wavelets/wavelets_typed.h | cusignal.wavelets.morlet2 |
| qmf | wavelets | qmf_device | qmf_typed_cpu | src/wavelets/wavelets_typed.h | cusignal.wavelets.qmf |
| ricker | wavelets | ricker_device | ricker_typed_cpu | src/wavelets/wavelets_typed.h | cusignal.wavelets.ricker |
| chebwin | windows | chebwin_device | chebwin_typed_cpu | src/windows/windows_typed.h | cusignal.windows.chebwin |
| general_cosine | windows | general_cosine_device | general_cosine_typed_cpu | src/windows/windows_typed.h | cusignal.windows.general_cosine |
| general_gaussian | windows | general_gaussian_device | general_gaussian_typed_cpu | src/windows/windows_typed.h | cusignal.windows.general_gaussian |
| hamming | windows | hamming_device | hamming_typed_cpu | src/windows/windows_typed.h | cusignal.windows.hamming |
| kaiser | windows | kaiser_device | kaiser_typed_cpu | src/windows/windows_typed.h | cusignal.windows.kaiser |
| parzen | windows | parzen_device | parzen_typed_cpu | src/windows/windows_typed.h | cusignal.windows.parzen |
| taylor | windows | taylor_device | taylor_typed_cpu | src/windows/windows_typed.h | cusignal.windows.taylor |
| triang | windows | triang_device | triang_typed_cpu | src/windows/windows_typed.h | cusignal.windows.triang |

## 13. 最终引用检查表

- [ ] 使用冻结交付候选的最新源码，而非本文整理前HEAD；
- [ ] 当前结论绑定commit、未提交状态、ZQ500环境、命令、日期和原始日志；
- [ ] 不把D6/D9/E3历史局部结果写成TASK_F五类型最终结果；
- [ ] 每个误差写明GPU结果与哪个CPU reference/独立公式比较；
- [ ] 写明输入输出dtype、shape、case、元素数、指标和容差；
- [ ] 离散结构使用mismatch，不写成浮点零误差；
- [ ] 零浮点指标引用TASK_F三审证据；
- [ ] FP64输出说明host拓宽、D2H/H2D和计时边界；
- [ ] 科学库图只画实际依赖；
- [ ] 环境写gpu_02实测CMake 3.22.1、最低兼容3.16；
- [ ] 性能图、误差分布和265项表引用TASK_F专文；
- [ ] Word/PDF/PPT数字与TASK_J结构化索引一致；
- [ ] 渲染后检查中文字体、表格、分页、图片清晰度和引用可读性。

## 14. TASK_F专文预留引用

~~~text
TASK_F文档路径：待填写
TASK_F冻结commit：待填写
correctness_pass：待填写/265
accuracy_pass：待填写/265
performance_present：待填写/265
unreviewed_zero_metrics：待填写
missing/extra/duplicate/failed/stale：待填写
原始日志目录：待填写
结构化CSV/JSON目录：待填写
~~~

最终Word/PDF可引用本文的架构、接口、类型、cuSignal追踪和平台适配章节，再从TASK_F专文引用正式误差和性能，避免技术路线与最终测试数据混在同一证据层级。
