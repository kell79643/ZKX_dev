# 图9：ZQ500 FP32 算子 GPU 相对 cuSignal Python GPU 加速比测试报告

生成日期：2026-07-20

用途：作为图9“GPU 相对 cuSignal Python GPU 加速比气泡图”的唯一直接数据来源。

## 1. 当前结论

本报告不再假定 53 个 FP32 算子均能在 ZQ500 上通过 cuSignal Python GPU 路径。对 Task1/Task2 之外的 33 项完成逐项补测后，可确认 **34 个算子实际执行成功**；其中 **32 个算子具有独立 Python/C++ GPU 成对均值，可生成图9气泡**。另外，`cfar_alpha` 只有 Host 标量计时，`ricker` 仅作为 `cwt` 的 wavelet callback 被耦合计时，因此二者不单独计算加速比。

其余 19 个算子未能在 ZQ500 cuSignal Python GPU 路径完成正式采样，保留 `Failed` 状态、实际尝试配置和直接原因，不填 0，不参与加速比、模块均值或气泡大小计算。旧 RTX 4090/NVIDIA CUDA 12.4 报告仅用于版式参考，其数值未进入本报告。

本次因 C++ 后端由 dlfft 切换为 Thrust，已在 ZQ500 上重新构建并覆盖旧报告结果。可绘图数据的总体范围为 **0.4467x～53.1495x**，中位数为 **2.8938x**；32 个气泡中 ZQ500 C++ GPU 更快 29 个，cuSignal Python GPU 更快 3 个。加速比大于 1 表示 ZQ500 C++ GPU 更快。

## 2. 数据来源与可追溯性

### 2.1 Thrust 重测成功子集

- 测试代码提交：`a3c8387b201e9e253483f6e62153dcf4deb6c54b`，测试时工作区 `dirty=true`；Task1 证据协议解析修正提交为 `5847532`。
- 构建目录：容器 `/tmp/ZKX_dev/build_figure9_thrust`；CMake 配置明确为 `USE_DLFFT=OFF`、`USE_THRUST=ON`，配置输出为 `ZQ500 FFT backend: THRUST`。
- Task1 原始证据：容器 `/tmp/ZKX_dev/Task1/task1_python/figure9_thrust_evidence/`；关键汇总为 `task1_cpp_python_performance.json` 和 `benchmark/task1_python_benchmark_summary.json`。
- Task2 原始证据：容器 `/tmp/ZKX_dev/Task2/task2_python/figure9_thrust_evidence/`；关键汇总为 `task2_cpp_python_performance.json` 和 `benchmark/task2_python_benchmark_summary.json`。
- 四个关键汇总已导出至远程宿主机 `/home/usr_02/figure9_thrust_export/`，可由算子名、批次、输入、参数、原始样本数和计时字段回溯。
- 重测时间：2026-07-20；Task1 为 1 次预热 + 10 次测量，Task2 为 5 批、每批 5 次预热 + 20 次测量。

### 2.2 其余 33 算子逐项补测

- Task1/Task2 已覆盖的 20 项未重复运行；其余 33 项使用 TASK_F FP32 固定 shape 和参数补测。
- 批量正式证据：远程宿主机 `/home/usr_02/figure9_remaining33_safe_suite.json`，SHA256 `a60f07388a133b1d3a39cfa391173eb267f4eb1ee90b31d4eac6315a621440fc`。
- 人类可读基线：`/home/usr_02/figure9_remaining33_safe_suite_baseline.md`，SHA256 `5fc29a97d06b20354b44236def8092374044eb9af6644b6a42558652d9d3d668`。
- 执行入口：`remote_scripts/benchmark_cusignal_zq500_fp32_isolated.py`；每项预热 5 次、正式测量 20 次，wall-clock 包围公开 Python API，并在调用前后同步设备。
- `resample_poly`、`kaiser` 使用独立进程确认，均在预热阶段触发 `LLVM ERROR: unsupported function call legalization: __nv_cyl_bessel_i0` 并 abort；没有有效样本。
- 补测结果：14 项 `Success`、19 项 `Failed`。普通 Python 异常逐项捕获；进程级 abort 独立隔离，不会阻止其他算子测试。

### 2.2 53 算子独立基线失败证据

- 目录：`交付/04_原始证据/operators/python_baseline_attempt_b027dca/`
- 日志：`chart_python_operator_baseline_failed.log`
- 日志 SHA256：`86BFD2F44CD6F1A86F391EBC5252E9989FDA2420A084684BD71EDA42FF834371`
- C++ 对照提交：`b027dca36ba1332cc9be5a340123bb2cfd17b543`
- 计划口径：TASK_F FP32 相同 shape/参数，预热 5 次、正式 20 次
- 结果：采集在 `resample_poly` 处因 LLVM Bessel 内建函数错误中止，未形成正式 Python CSV；此前还出现 dlrtc/JIT、`CUDA_ERROR_FILE_NOT_FOUND`、`CUFFT_NOT_SUPPORTED`、cuSolver 未定义符号等错误。

空的旧 `正式数据/operator_python_cpp.csv` 和 `正式数据/task_operator_python_cpp.csv` 不能视为性能结果。本报告的 32 个数值由 18 个 Task Thrust 重测结果和 14 个本次逐项补测结果组成，均按两侧算术平均值相除；不从空表、旧 NVIDIA 报告、旧 dlfft 批次或 C++ CPU 数据推算。

## 3. 测试平台、后端与口径

| 字段 | 配置 |
| --- | --- |
| 测试平台 | 远程主机 `usr_02@10.110.12.10`，容器 `gpu_02`，设备 `ZQ500-Q QUAD-2`，device 0 |
| Python 后端 | Python 3.12.11、cuSignal 23.08.00、CuPy 13.6.0 ZQ500 私有适配、CUDA runtime/driver 11.7（11070） |
| ZQ500 C++ 后端 | ZQ500 DLI_V2；`USE_DLFFT=OFF`、`USE_THRUST=ON`；FFT 相关实现使用 Thrust 路径 |
| 数据类型 | FP32；Task1 雷达复数链路为 ComplexFP32；CWT 实现内部允许 FP64 输出/中间量，但输入为 FP32 |
| 预热次数 | Task1：Python/C++ 各 1 次；Task2：各 5 批 × 5 次，共 25 次；其余成功项：Python 5 次，C++ TASK_F 5 次 |
| 采样次数 | Task1：各 10 次；Task2：各 100 次；其余成功项：Python 20 次、C++ TASK_F 100 次；各侧均取全部有效样本的算术平均值 |
| Python 计时器 | `cupy.cuda.Event`，每个公开算子调用同步后读取 elapsed time；`cfar_alpha` 例外为 Host wall time |
| C++ 计时器 | 同步 CUDA event 包围公开 C++ GPU 算子调用 |
| Timing Scope | Task1/Task2 为 CUDA event 包围公开算子 API，不含任务级 H2D/D2H；新增 Python 项为同步 wall-clock 公开 API（包含脚本 `checksum` 所需的同步/归约/回读），C++ 侧严格沿用 TASK_F 每项记录的 `device-resident` 或 `compatibility` 边界。逐行明确列出，不解释为纯 kernel 对比 |
| 公平性边界 | 同一 Task、同一步、同输入和参数；Task1 两侧协议均为 1+10，Task2 两侧协议均为 5×(5+20)。Python 高层 API 与 C++ wrapper 的 allocation/plan 边界按真实公开调用计入，因此本报告不解释为纯 kernel 消融 |

### 3.1 加速比定义

```text
Speedup = cuSignal Python GPU 平均耗时 / ZQ500 GPU 平均耗时
```

- `Speedup > 1`：ZQ500 C++ GPU 更快。
- `Speedup < 1`：cuSignal Python GPU 更快。
- `Status != Success` 或没有独立成对计时：加速比写 `—`，不写 0，不参与绘图。
- `ambgfun` 采用 Task1 的等价三调用工作负载：2D、delay cut、doppler cut 的三个公开调用分别求和；仅计 cuSignal/C++ API，不把 Python cut adapter 计入算子时间。
- `cwt` 行采用 `cwt(..., ricker, widths=[2,4,8,12])` 的耦合公开调用；`ricker` 行只记录“成功但耦合”，不重复制造第二个气泡。
- Task2 既有验收脚本另有“五批均值中位数”汇总字段；该规则来自项目脚本，只作稳定性旁证，**不用于图9**。图9严格使用全部 100 个有效样本的算术平均值；C++ 五批样本数相同，故五个批均值的算术平均等于全部样本均值。

## 4. 图9简明清单

以下 32 行可直接作为图9气泡数据；建议气泡面积按加速比的对数或截断尺度映射，标签仍显示原始值，避免 `firwin` 等大值压缩其他点的视觉差异。

```text
bsplines - cubic - 3.0934x
bsplines - gauss_spline - 5.7503x
bsplines - quadratic - 2.4486x
convolution - correlate - 2.6231x
demod - fm_demod - 33.4381x
estimation - KalmanFilter - 3.6123x
filter_design - firwin - 53.1495x
filtering - firfilter - 6.2499x
filtering - resample - 0.4467x
peak_finding - argrelextrema - 8.2294x
radartools - ambgfun - 0.5495x
radartools - ca_cfar - 5.8321x
radartools - pulse_compression - 1.7814x
radartools - pulse_doppler - 0.4591x
spectral_analysis - spectrogram - 2.8969x
spectral_analysis - csd - 2.4688x
spectral_analysis - istft - 6.7237x
spectral_analysis - stft - 2.8906x
spectral_analysis - vectorstrength - 1.7078x
waveforms - chirp - 3.0628x
waveforms - gausspulse - 7.5339x
waveforms - sawtooth - 1.0638x
waveforms - square - 6.7844x
waveforms - unit_impulse - 2.8294x
wavelets - cwt (ricker callback) - 9.3100x
wavelets - morlet - 2.8289x
wavelets - morlet2 - 2.8804x
wavelets - qmf - 13.2550x
windows - hamming - 7.3752x
windows - general_cosine - 2.8681x
windows - general_gaussian - 1.5622x
windows - taylor - 2.6044x
```

## 5. 可计算全量明细表

表中 `Input Shape` 是该条加速比实际使用的输入，不是 TASK_F 的另一组独立固定用例。`参数配置` 仅省略不会改变算子语义的任务脚手架参数。

| 模块 | 算子 | 来源 | 测试平台 | 后端配置 | Input Shape | 输入规模 | 参数配置 | 数据类型 | 预热 | 采样 | Timing Scope | Python mean (ms) | ZQ500 mean (ms) | Speedup | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| bsplines | cubic | Task2 Step3 | ZQ500 `gpu_02` | cuSignal/CuPy vs C++ DLI_V2 Thrust | `[9]` | 9 offsets | `-2:0.5:2` | FP32 input | 25 | 100 | operator API execution only | 0.079216 | 0.025608 | 3.0934x | Success |
| bsplines | gauss_spline | 33项补测/TASK_F | 同上 | 同上 | `[4096]` | 4096 elements | `n=5` | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ device-resident | 0.328859 | 0.057190 | 5.7503x | Success |
| bsplines | quadratic | 33项补测/TASK_F | 同上 | 同上 | `[4096]` | 4096 elements | default | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ device-resident | 0.137841 | 0.056293 | 2.4486x | Success |
| convolution | correlate | Task2 Step4 | 同上 | 同上 | `[480]` × `[480]` | 480×480 same-correlation | `mode=same, method=auto` | FP32 | 25 | 100 | operator API execution only | 0.214577 | 0.081803 | 2.6231x | Success |
| demod | fm_demod | Task2 Step4 | 同上 | 同上 | `[480]` | 480 samples | `axis=-1` | ComplexFP32 | 25 | 100 | operator API execution only | 0.911733 | 0.027266 | 33.4381x | Success |
| estimation | KalmanFilter | Task2 Step6 | 同上 | 同上 | observations `[8]`, state/covariance `[1]` | 8 predict/update | `dim_x=dim_z=points=1,F=H=1,Q=0.0001,R=0.002` | FP32 | 25 | 100 | object/state setup excluded; predict/update execution only | 8.260163 | 2.286685 | 3.6123x | Success |
| filter_design | firwin | Task2 Step2 | 同上 | 同上 | cutoff `[1]`, output `[33]` | 33 taps | `cutoff=90 Hz,window=hamming,fs=512,gpupath=true` | implementation-selected floating | 25 | 100 | operator API execution only | 1.501283 | 0.028246 | 53.1495x | Success |
| filtering | firfilter | Task2 Step2 | 同上 | 同上 | signal `[512]`, taps `[33]` | 512×33 | `axis=-1,taps=33` | FP32 | 25 | 100 | operator API execution only | 0.225353 | 0.036057 | 6.2499x | Success |
| filtering | resample | 33项补测/TASK_F | 同上 | 同上 | `[256]` | 256→192 samples | `num=192,axis=0,window=None,domain=time` | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.716590 | 1.604067 | 0.4467x | Success |
| peak_finding | argrelextrema | Task2 Step5 | 同上 | 同上 | `[480]` | 480 samples | `greater,axis=0,order=3,mode=clip` | FP32 input/INT64 indices | 25 | 100 | operator API execution only | 1.736727 | 0.211040 | 8.2294x | Success |
| radartools | ambgfun | Task1 Step5 | ZQ500 `gpu_02` | cuSignal/CuPy vs C++ DLI_V2 Thrust | waveform `[64]` | 3 calls: 2D + 2 cuts | `fs=2.56 MHz,prf=8 kHz`; three-call API sum | ComplexFP32 | 1 | 10 | three operator API calls summed; Python cut adapter excluded | 2.396458 | 4.360915 | 0.5495x | Success |
| radartools | ca_cfar | Task1 Step4 | 同上 | 同上 | range-Doppler power `[32,256]` | 8192 cells | `guard=(1,2),reference=(2,6),pfa=1e-3,alpha=7.142329` | FP32 | 1 | 10 | operator API execution only | 0.413530 | 0.070906 | 5.8321x | Success |
| radartools | pulse_compression | Task1 Step2 | 同上 | 同上 | pulses `[32,256]`, template `[64]` | 8192 complex samples | `normalize=true,nfft=256,circular padding` | ComplexFP32 | 1 | 10 | operator API execution only | 0.937262 | 0.526143 | 1.7814x | Success |
| radartools | pulse_doppler | Task1 Step3 | 同上 | 同上 | `[32,256]` | 32 pulses × 256 range bins | `nfft=32`, Hamming window supplied by matched step | ComplexFP32 | 1 | 10 | operator API execution only; Python Hamming adapter excluded | 0.270391 | 0.588907 | 0.4591x | Success |
| spectral_analysis | spectrogram | Task2 Step4 | ZQ500 `gpu_02` | cuSignal/CuPy vs C++ DLI_V2 Thrust | `[480]` | 14 frames × 64 bins | `fs=512,boxcar,nperseg=64,noverlap=32,nfft=64,two-sided,spectrum,magnitude` | FP32 | 25 | 100 | public API; internal plan/workspace included | 1.059389 | 0.365694 | 2.8969x | Success |
| spectral_analysis | csd | 33项补测/TASK_F | 同上 | 同上 | `[512]` × `[512]` | two 512-sample signals | `fs=100,hann,nperseg=64,noverlap=32,nfft=64,density,mean` | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 1.759220 | 0.712593 | 2.4688x | Success |
| spectral_analysis | istft | 33项补测/TASK_F | 同上 | 同上 | `Zxx=[64,15]` | 64 bins × 15 frames | `fs=100,hann,nperseg=64,noverlap=32,nfft=64` | ComplexFP32 input | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 2.664226 | 0.396247 | 6.7237x | Success |
| spectral_analysis | stft | 33项补测/TASK_F | 同上 | 同上 | `[512]` | 512 samples | `fs=100,hann,nperseg=64,noverlap=32,nfft=64` | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 1.635507 | 0.565799 | 2.8906x | Success |
| spectral_analysis | vectorstrength | 33项补测/TASK_F | 同上 | 同上 | events `[64]`, periods `[3]` | 64 events × 3 periods | `periods=[8,16,32]` | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.705540 | 0.413131 | 1.7078x | Success |
| waveforms | chirp | Task2 Step1 | 同上 | 同上 | `[512]` | 512 samples | `f0=20,t1=1,f1=70,phi=0` | FP32 | 25 | 100 | operator API execution only | 0.084456 | 0.027575 | 3.0628x | Success |
| waveforms | gausspulse | Task2 Step1 | 同上 | 同上 | `[512]` | 512 samples | `fc=170,bw=0.20` | FP32 | 25 | 100 | operator API execution only | 0.198377 | 0.026331 | 7.5339x | Success |
| waveforms | sawtooth | Task2 Step1 | 同上 | 同上 | `[512]` | 512 samples | `width=0.65` | FP32 | 25 | 100 | operator API execution only | 0.210812 | 0.198163 | 1.0638x | Success |
| waveforms | square | Task2 Step1 | 同上 | 同上 | `[512]` | 512 samples | `duty=0.35` | FP32 | 25 | 100 | operator API execution only | 0.180343 | 0.026582 | 6.7844x | Success |
| waveforms | unit_impulse | 33项补测/TASK_F | 同上 | 同上 | output `[4096]` | 4096 elements | `idx=mid,dtype=FP32` | FP32 output | 5/5 | 20/100 | Python synchronized API+checksum; C++ device-resident | 0.179336 | 0.063383 | 2.8294x | Success |
| wavelets | cwt | Task2 Step4 | 同上 | 同上 | `[480]`, widths `[4]` | 4×480 coefficients | `wavelet=ricker,widths=[2,4,8,12]` | FP32 input/implementation FP64 output | 25 | 100 | coupled `cwt(ricker)` operator API execution only | 2.394297 | 0.257174 | 9.3100x | Success |
| wavelets | morlet | 33项补测/TASK_F | 同上 | 同上 | output `[4096]` | 4096 elements | `w=5,s=2,complete=true` | FP32 case / complex output | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.845856 | 0.299002 | 2.8289x | Success |
| wavelets | morlet2 | 33项补测/TASK_F | 同上 | 同上 | output `[4096]` | 4096 elements | `s=2,w=5` | FP32 case / complex output | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.868425 | 0.301490 | 2.8804x | Success |
| wavelets | qmf | 33项补测/TASK_F | 同上 | 同上 | `[65536]` | 65536 taps | default | FP32 | 5/5 | 20/100 | Python synchronized API+checksum; C++ device-resident | 0.853524 | 0.064393 | 13.2550x | Success |
| windows | hamming | Task2 Step2 | 同上 | 同上 | `[33]` | 33 points | `M=33,sym=true` | implementation-selected floating | 25 | 100 | operator API execution only | 0.187656 | 0.025444 | 7.3752x | Success |
| windows | general_cosine | 33项补测/TASK_F | 同上 | 同上 | output `[1024]`, coefficients `[3]` | 1024 points | `a=[1,2,1],sym=true` | FP32 coefficients | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.593492 | 0.206929 | 2.8681x | Success |
| windows | general_gaussian | 33项补测/TASK_F | 同上 | 同上 | output `[1024]` | 1024 points | `p=2,sig=32,sym=true` | FP32 case | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.324019 | 0.207410 | 1.5622x | Success |
| windows | taylor | 33项补测/TASK_F | 同上 | 同上 | output `[1024]` | 1024 points | `nbar=4,sll=30,norm=true,sym=true` | FP32 case | 5/5 | 20/100 | Python synchronized API+checksum; C++ compatibility end-to-end | 0.893661 | 0.343137 | 2.6044x | Success |

## 6. 53 个 FP32 算子全量状态表

状态说明：

- `Success`：有独立成对算术平均值，可参与图9；Task1 为 10 次，Task2 为 100 次。
- `Success-Coupled`：实际运行成功，但计时与另一个算子不可分；保留测试结果，不单独画气泡。
- `Unsupported`：当前 Python 路径不是 GPU 算子计时，不能作为分子。
- `Failed`：Python GPU 路径未形成有效正式样本；原因列给出本次逐项补测观察到的直接错误。

| 模块 | 算子 | Status | 图9 | 结果或原因 |
| --- | --- | --- | --- | --- |
| bsplines | cubic | Success | 3.0934x | Task2 Thrust 重测 100 次成对结果 |
| bsplines | gauss_spline | Success | 5.7503x | 独立补测完成 5 次预热、20 次 Python 测量；匹配 TASK_F C++ 数据 |
| bsplines | quadratic | Success | 2.4486x | 同上 |
| convolution | correlate | Success | 2.6231x | Task2 Thrust 重测 100 次成对结果；独立 stock 路径另见 `CUDA_ERROR_FILE_NOT_FOUND` |
| convolution | correlate2d | Failed | — | `CUDA_ERROR_FILE_NOT_FOUND` |
| demod | fm_demod | Success | 33.4381x | Task2 Thrust 重测 100 次成对结果 |
| estimation | KalmanFilter | Success | 3.6123x | Task2 ZQ500 adapter Thrust 重测 100 次结果；stock 路径 dlrtc 编译失败 |
| filter_design | firwin | Success | 53.1495x | Task2 Thrust 重测 100 次成对结果 |
| filter_design | firwin2 | Failed | — | `CUFFT_NOT_SUPPORTED` |
| filtering | channelize_poly | Failed | — | `CUDA_ERROR_FILE_NOT_FOUND` |
| filtering | detrend | Failed | — | dlrtc/LLVM 编译失败，`math_constants.h` 宏与生成 kernel 相关错误 |
| filtering | firfilter | Success | 6.2499x | Task2 Thrust 重测 100 次成对结果 |
| filtering | firfilter2 | Failed | — | cuSolver `cusolverDnSXgels_bufferSize` 未定义符号 |
| filtering | freq_shift | Failed | — | dlrtc/LLVM 编译失败 |
| filtering | hilbert | Failed | — | dlrtc/LLVM 编译失败 |
| filtering | hilbert2 | Failed | — | `CUFFT_NOT_SUPPORTED` |
| filtering | lfilter_zi | Failed | — | cuSolver `cusolverDnSXgels_bufferSize` 未定义符号 |
| filtering | sosfilt | Failed | — | `CUDA_ERROR_FILE_NOT_FOUND` |
| filtering | wiener | Failed | — | `Direct method is only implemented for 1D` |
| filtering | decimate | Failed | — | `CUDA_ERROR_FILE_NOT_FOUND` |
| filtering | resample | Success | 0.4467x | 独立补测完成正式采样 |
| filtering | resample_poly | Failed | — | LLVM abort：`unsupported function call legalization: __nv_cyl_bessel_i0` |
| filtering | upfirdn | Failed | — | `CUDA_ERROR_FILE_NOT_FOUND` |
| peak_finding | argrelextrema | Success | 8.2294x | Task2 ZQ500 adapter Thrust 重测 100 次结果 |
| radartools | ambgfun | Success | 0.5495x | Task1 Thrust 重测三种 cut 等价工作负载 10 次结果 |
| radartools | ca_cfar | Success | 5.8321x | Task1 Thrust 重测 10 次成对结果 |
| radartools | cfar_alpha | Unsupported | — | Task1 执行成功，但 Python 侧为 Host wall time，不是 cuSignal Python GPU 时间 |
| radartools | pulse_compression | Success | 1.7814x | Task1 Thrust 重测 10 次成对结果 |
| radartools | pulse_doppler | Success | 0.4591x | Task1 Thrust 重测 10 次成对结果 |
| spectral_analysis | csd | Success | 2.4688x | 独立补测完成正式采样 |
| spectral_analysis | istft | Success | 6.7237x | 独立补测完成正式采样 |
| spectral_analysis | lombscargle | Failed | — | 同上 |
| spectral_analysis | spectrogram | Success | 2.8969x | Task2 Thrust 重测 100 次成对结果；内部 plan/workspace 按真实调用计入 |
| spectral_analysis | stft | Success | 2.8906x | 独立补测完成正式采样 |
| spectral_analysis | vectorstrength | Success | 1.7078x | 独立补测完成正式采样 |
| waveforms | chirp | Success | 3.0628x | Task2 Thrust 重测 100 次成对结果；Task1 也有成功调用证据 |
| waveforms | sawtooth | Success | 1.0638x | Task2 Thrust 重测 100 次成对结果 |
| waveforms | square | Success | 6.7844x | Task2 Thrust 重测 100 次成对结果 |
| waveforms | gausspulse | Success | 7.5339x | Task2 Thrust 重测 100 次成对结果 |
| waveforms | unit_impulse | Success | 2.8294x | 独立补测完成正式采样 |
| wavelets | cwt | Success | 9.3100x | Task2 `cwt(ricker)` Thrust 重测 100 次耦合调用结果 |
| wavelets | morlet | Success | 2.8289x | 独立补测完成正式采样 |
| wavelets | morlet2 | Success | 2.8804x | 独立补测完成正式采样 |
| wavelets | qmf | Success | 13.2550x | 独立补测完成正式采样 |
| wavelets | ricker | Success-Coupled | — | Task2 中作为 `cwt` callback 实际执行成功；无独立计时，不重复计算气泡 |
| windows | chebwin | Failed | — | dlrtc/LLVM kernel 编译失败（`math_constants.h` 相关诊断） |
| windows | general_cosine | Success | 2.8681x | 独立补测完成正式采样 |
| windows | general_gaussian | Success | 1.5622x | 独立补测完成正式采样 |
| windows | hamming | Success | 7.3752x | Task2 Thrust 重测 100 次成对结果；Task1 另有 adapter 调用证据 |
| windows | kaiser | Failed | — | 独立进程 LLVM abort：`unsupported function call legalization: __nv_cyl_bessel_i0` |
| windows | parzen | Failed | — | dlrtc/LLVM kernel 编译失败（`math_constants.h` 相关诊断） |
| windows | taylor | Success | 2.6044x | 独立补测完成正式采样 |
| windows | triang | Failed | — | dlrtc/LLVM kernel 编译失败（`math_constants.h` 相关诊断） |

### 6.1 未进入图9算子的实际测试配置

下表补齐所有 FP32 非气泡项的 `Input Shape` 和 `Timing Scope`。`Failed` 项的预热/采样写成“计划 5/20，实际 0 valid”，表示公开 API 在预热或首次执行阶段失败；不能把失败前的启动时间当作算子耗时。

| 模块 | 算子 | Input Shape | 参数配置 | 数据类型 | 预热/采样 | Timing Scope | Status / 原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| convolution | correlate2d | in1 `[128,128]`, in2 `[5,5]` | `mode=same,boundary=fill,fillvalue=0` | FP32 | 计划 5/20，0 valid | synchronized Python public API；失败，无有效计时 | Failed：`CUDA_ERROR_FILE_NOT_FOUND` |
| filter_design | firwin2 | output `[257]`, freq/gain `[3]` | `nfreqs=513,window=hamming,fs=16,gpupath=true` | FP32 case | 同上 | 同上 | Failed：`CUFFT_NOT_SUPPORTED` |
| filtering | channelize_poly | x `[256]`, h `[32]` | `n_chans=8` | FP32 | 同上 | 同上 | Failed：`CUDA_ERROR_FILE_NOT_FOUND` |
| filtering | detrend | `[16,16]` | `axis=1,type=linear,bp=0` | FP32 | 同上 | 同上 | Failed：dlrtc/LLVM 编译失败 |
| filtering | firfilter2 | b `[5]`, x `[256]` | `axis=-1,padtype=odd,padlen=12,method=pad` | FP32 | 同上 | 同上 | Failed：cuSolver `cusolverDnSXgels_bufferSize` 未定义符号 |
| filtering | freq_shift | `[256]` | `freq=1,fs=32` | ComplexFP32 | 同上 | 同上 | Failed：dlrtc/LLVM 编译失败 |
| filtering | hilbert | `[16,16]` | `N=16,axis=1` | FP32 | 同上 | 同上 | Failed：dlrtc/LLVM 编译失败 |
| filtering | hilbert2 | `[16,16]` | `N=None` | FP32 | 同上 | 同上 | Failed：`CUFFT_NOT_SUPPORTED` |
| filtering | lfilter_zi | b `[5]`, a `[5]` | fixed FIR/IIR coefficients | FP32 | 同上 | 同上 | Failed：cuSolver `cusolverDnSXgels_bufferSize` 未定义符号 |
| filtering | sosfilt | sos `[2,6]`, x `[256]` | `axis=-1` | FP32 | 同上 | 同上 | Failed：`CUDA_ERROR_FILE_NOT_FOUND` |
| filtering | wiener | `[16,16]` | `mysize=[3,3],noise=None` | FP32 | 同上 | 同上 | Failed：Direct method 仅实现 1D |
| filtering | decimate | `[256]` | `q=2,n=32,axis=-1,zero_phase=true,gpupath=true` | FP32 | 同上 | 同上 | Failed：`CUDA_ERROR_FILE_NOT_FOUND` |
| filtering | resample_poly | `[256]` | `up=3,down=2,axis=0` | FP32 | 同上 | 同上 | Failed：LLVM Bessel abort |
| filtering | upfirdn | h `[5]`, x `[256]` | `up=3,down=2,axis=-1` | FP32 | 同上 | 同上 | Failed：`CUDA_ERROR_FILE_NOT_FOUND` |
| radartools | cfar_alpha | scalar | `pfa=1e-3,N=32` | FP32 scalar contract | Task1 1/10 | Host wall time only；不是 GPU Timing Scope | Unsupported：无 cuSignal Python GPU 分子 |
| spectral_analysis | lombscargle | x/y `[64]`, freqs `[32]` | `precenter=true,normalize=true` | FP32 | 计划 5/20，0 valid | synchronized Python public API；失败，无有效计时 | Failed：`CUDA_ERROR_FILE_NOT_FOUND` |
| wavelets | ricker | points `480` within CWT, widths `[2,4,8,12]` | `cwt(...,wavelet=ricker)` | FP32 input / implementation FP64 output | Task2 25/100 | coupled inside CWT public API；不可独立拆分 | Success-Coupled |
| windows | chebwin | output `[1024]` | `at=80,sym=true` | FP32 case | 计划 5/20，0 valid | synchronized Python public API；失败，无有效计时 | Failed：dlrtc/LLVM 编译失败 |
| windows | kaiser | output `[1024]` | `beta=8,sym=true` | FP32 case | 同上 | 同上 | Failed：LLVM Bessel abort |
| windows | parzen | output `[1024]` | `sym=true` | FP32 case | 同上 | 同上 | Failed：dlrtc/LLVM 编译失败 |
| windows | triang | output `[1024]` | `sym=true` | FP32 case | 同上 | 同上 | Failed：dlrtc/LLVM 编译失败 |

## 7. 模块汇总

| 模块 | FP32 算子总数 | 实际执行成功 | 可计算气泡 | Speedup 范围 | ZQ500 更快 | Python 更快 | 未形成独立气泡 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| bsplines | 3 | 3 | 3 | 2.4486x～5.7503x | 3 | 0 | 0 |
| convolution | 2 | 1 | 1 | 2.6231x | 1 | 0 | 1 |
| demod | 1 | 1 | 1 | 33.4381x | 1 | 0 | 0 |
| estimation | 1 | 1 | 1 | 3.6123x | 1 | 0 | 0 |
| filter_design | 2 | 1 | 1 | 53.1495x | 1 | 0 | 1 |
| filtering | 14 | 2 | 2 | 0.4467x～6.2499x | 1 | 1 | 12 |
| peak_finding | 1 | 1 | 1 | 8.2294x | 1 | 0 | 0 |
| radartools | 5 | 5 | 4 | 0.4591x～5.8321x | 2 | 2 | 1 |
| spectral_analysis | 6 | 5 | 5 | 1.7078x～6.7237x | 5 | 0 | 1 |
| waveforms | 5 | 5 | 5 | 1.0638x～7.5339x | 5 | 0 | 0 |
| wavelets | 5 | 5 | 4 | 2.8289x～13.2550x | 4 | 0 | 1 |
| windows | 8 | 4 | 4 | 1.5622x～7.3752x | 4 | 0 | 4 |
| **总计** | **53** | **34** | **32** | **0.4467x～53.1495x** | **29** | **3** | **21** |

“实际执行成功”只表示当前证据中真实调用成功，不等价于 stock cuSignal 23.08 在所有参数域均受支持；Task2 的 `KalmanFilter`、`argrelextrema` 等依赖必需 ZQ500 adapter，vendored cuSignal 源码未修改。

## 8. 异常与边界分析

### 8.1 后端切换的显著影响

旧报告中 `spectrogram` 和 `chirp` 分别为 0.0004x、0.0286x；Thrust 重测后分别为 2.8969x、3.0628x，证明后端切换对结果有实质影响，因此旧 dlfft 数值已全部废弃并被本报告覆盖。`firwin` 为本次最大值 53.1495x，来自 Python 1.501283 ms 与 ZQ500 0.028246 ms 的 100 次算术平均；它是当前公开 API 边界下的结果，不外推为所有输入规模的峰值性能。

### 8.2 三个低于 1 的算子

`resample`、`ambgfun`、`pulse_doppler` 分别为 0.4467x、0.5495x、0.4591x，表示各自记录的公开调用边界中 cuSignal Python GPU 更快。三者均保留真实值，不截断为 1，也不从模块汇总中剔除。`resample` 使用 TASK_F compatibility C++ 边界；后两项使用 Task1 operator API 边界，图表必须保留逐行 Timing Scope。

### 8.3 `cfar_alpha` 不参与气泡

Task1 已验证 `cfar_alpha` 业务调用成功，Python 证据均值约 0.003155 ms，但证据明确标记为 Host wall time。把它除以 C++ GPU event 时间会混合 Host 与 GPU 口径，因此本报告标记 `Unsupported`，不计算加速比。

### 8.4 `ricker` 不重复计算

Task2 的计时键为 `cwt_ricker`，表示 `cusignal.cwt` 在内部调用 `cusignal.ricker`。现有证据不能把总时间可靠拆成两个独立均值。图9只为 `cwt` 生成一个气泡；`ricker` 保留 `Success-Coupled` 状态，避免同一时间被重复使用两次。

### 8.5 全量逐项补测与进程隔离

旧全量脚本会被 `resample_poly` 或 `kaiser` 的 LLVM abort 直接终止，因此旧日志不能判断后续算子。此次对 33 个非任务算子重新补测：普通异常逐项捕获，完成项即时写入 JSON，两个 Bessel abort 项单独进程确认。由此新增 14 个正式成功算子，并为其余 19 项取得直接失败原因；不再使用“采集未到达”作为最终状态依据。

## 9. 后续补测与图9绘制规则

1. 当前图9只能绘制第4、5节的 32 个气泡，标题或图注明确写“ZQ500 实际成功且具备独立成对计时的 FP32 算子（32项）”，不能写“53/53”。
2. X 轴使用算子名，Y 轴使用模块名，气泡原始值使用 `Speedup`；建议颜色区分 `>1` 与 `<1`，并以 `1x` 为视觉基准。
3. `Failed`、`Unsupported`、`Success-Coupled` 不生成 0 大小气泡；可在图外附状态计数 `32 Success / 1 Success-Coupled / 1 Unsupported / 19 Failed`。
4. 新增结果必须使用相同 input shape、参数、FP32 dtype、同步和 Timing Scope；若改用 TASK_F 独立输入或再次切换后端，应重新测量 Python 与 C++ 两侧，不能与本报告 Task 场景或旧 dlfft 批次交叉相除。
5. 后续若修复失败项，必须以独立进程重新完成 5 次预热和至少 20 次有效 Python GPU 采样，并保存原始样本、mean、平台、后端、提交和错误文本；不能把“API 返回一次”直接升级为 `Success`。
