# 核心必做算子 API 接口对照表

口径说明：本表按任务1、任务2核心流程算子整理，基准为 Python `cusignal-23.08.00`。  
“一致”表示 host 接口的函数名、参数语义、默认行为和输出形态与官方库基本同形；“语义对齐”表示 C++/CUDA 为表达动态类型、多返回值、`None`、预分配输出或 workspace 而采用等价接口形态。

| 任务 | 模块/环节 | 算子函数 | 核心输入参数 | 输出结果 | 与 Python cuSignal 对齐情况 |
|---|---|---|---|---|---|
| 任务1 | LFM 脉冲生成 | `generate_lfm_pulse_kernel`（对照 `chirp` 语义） | `fs, bandwidth, pulse_width, samples_per_pulse` | 复数 LFM 发射脉冲 | 不完全一致：任务自定义 kernel，数学语义接近 Chirp/LFM，但函数名和参数不与 Python cuSignal 同形 |
| 任务1 | 脉冲压缩 | `pulse_compression` / `pulse_compression_device` | `x, template, normalize, window, nfft` | 复数压缩结果，1D 或 2D | 语义对齐：`window=None/nfft=None` 以空字符串或 `-1` 表达；device 使用 params/workspace |
| 任务1 | 脉冲多普勒 | `pulse_doppler` / `pulse_doppler_device` | `x, window, nfft` | 复数距离-多普勒矩阵 | 语义对齐：host 参数语义对齐；device 使用 params/workspace |
| 任务1 | CFAR 检测 | `ca_cfar` / `ca_cfar_device` | `array, guard_cells, reference_cells, pfa` | 检测索引或检测 mask/threshold | 一致：host 接口对齐；device 输出拆为预分配 mask/threshold |
| 任务1 | CFAR 阈值系数 | `cfar_alpha` / `cfar_alpha_device` | `pfa, N` | 阈值乘数 `alpha` | 一致：host 接口与 Python cuSignal 对齐；device 输出为预分配标量 buffer |
| 任务1 | 模糊函数分析 | `ambgfun` / `ambgfun_device` | `x, fs, prf, y, cut, cutValue` | 2D 模糊函数或切片结果 | 语义对齐：`y=None` 以空数组表达；device 使用 workspace |
| 任务2 | Chirp 波形 | `chirp` / `chirp_device` | `t, f0, t1, f1, method, phi, vertex_zero, type` | 实数或复数 Chirp 波形 | 语义对齐：host 接口对齐；device 采用预分配输出 |
| 任务2 | 高斯脉冲 | `gausspulse` / `gausspulse_device` | `t, fc, bw, bwr, tpr, retquad, retenv` | `y` 或 `(y, yq)` 或 `(y, yq, ye)` | 语义对齐：host 用 variant 表达多返回；device 固定多输出 buffer |
| 任务2 | 锯齿波 | `sawtooth` / `sawtooth_device` | `t, width` | 实数锯齿波序列 | 一致：host 参数和默认值对齐；device 采用预分配输出 |
| 任务2 | 方波 | `square` / `square_device` | `t, duty` | 实数方波序列 | 一致：host 参数和默认值对齐；device 采用预分配输出 |
| 任务2 | FIR 滤波器设计 | `firwin` / `firwin_device` | `numtaps, cutoff, width, window, pass_zero, scale, nyq, fs, gpupath` | FIR 滤波器系数 | 语义对齐：`width/nyq/fs=None` 以 `0` 或 `-1` 哨兵值表达 |
| 任务2 | FIR 滤波应用 | `firfilter` / `firfilter_device` | `b, x, axis, zi` | 滤波后信号，带 `zi` 时返回 `y, zf` | 语义对齐：host 用重载/pair 表达 `zi=None` 与多返回；device 使用预分配输出 |
| 任务2 | B 样条平滑 | `cubic` / `cubic_device` | `x` | 三次 B 样条平滑序列 | 一致：host 接口与输出形态对齐；device 采用预分配输出 |
| 任务2 | B 样条平滑 | `quadratic` / `quadratic_device` | `x` | 二次 B 样条平滑序列 | 一致：host 接口与输出形态对齐；device 采用预分配输出 |
| 任务2 | B 样条平滑 | `gauss_spline` / `gauss_spline_device` | `x, n` | 高斯样条平滑序列 | 一致：host 接口与输出形态对齐；device 采用预分配输出 |
| 任务2 | FM 解调 | `fm_demod` / `fm_demod_device` | `x, axis` | 瞬时频率/相位差分序列 | 一致：host 参数语义对齐；device 对 1D/2D 显式传 shape/axis |
| 任务2 | 相关分析 | `correlate` / `correlate_device` | `in1, in2, mode, method` | 相关序列 | 语义对齐：host 参数对齐；device 可使用 workspace 加速 |
| 任务2 | 谱分析 | `lombscargle` / `lombscargle_device` | `x, y, freqs, precenter, normalize` | 周期图功率谱 | 一致：参数语义和输出形态对齐；device 采用预分配输出 |
| 任务2 | 小波变换 | `cwt` / `cwt_device` | `data, wavelet, widths` | 尺度 x 时间的 CWT 复数矩阵 | 语义对齐：host 支持函数/字符串小波；device 使用小波名称和扁平输出 |
| 任务2 | Morlet 小波 | `morlet` / `morlet_device` | `M, w, s, complete` | 复数 Morlet 小波母函数 | 一致：host 参数与默认值对齐；device 采用预分配输出 |
| 任务2 | Ricker 小波 | `ricker` / `ricker_device` | `points, a` | 实数 Ricker 小波母函数 | 一致：host 参数和输出形态对齐；device 采用预分配输出 |
| 任务2 | 卡尔曼滤波 | `KalmanFilter` / `kalman_predict_device` / `kalman_update_device` | `dim_x, dim_z, x, P, Q, F, H, R, z` | 更新后的状态估计 `x` 和协方差 `P`（原地修改） | 语义对齐：host 构造同形但缺少 dtype 及 predict/update 可选覆盖；device 拆为两独立函数并采用预分配 DeviceArray |
| 任务2 | 相对极值检测 | `argrelextrema` / `argrelextrema_device` | `data, comparator, axis, order, mode` | 极值点索引数组（多维返回 tuple） | 语义对齐：host 参数默认值对齐且 comparator 支持双形态；device 以 find_max 布尔替代 comparator 且输出拆为预分配 buffer |

## 简要统计

| 统计项 | 数量 | 说明 |
|---|---:|---|
| 表内核心算子条目 | 23 | 按 `ZKX/Task1`、`ZKX/Task2` 实际实现/调用口径列出，`chirp` 同时服务任务1和任务2 |
| 接口基本一致 | 11 | host 接口与 Python cuSignal 基本同形，device 仅输出形态适配 |
| 语义对齐但形态不完全一致 | 11 | 主要涉及 `None` 哨兵值、多返回值、`DeviceArray`、workspace、预分配输出或类/方法形态差异 |
| 任务自定义实现 | 1 | 任务1 LFM 脉冲生成使用自定义 kernel，不是 Python cuSignal 同形 API |
| 已有 host/Python 输出对比基础 | 235/235 | 来自现有 reference 对比记录 |
| GPU device 可调用与计时覆盖 | 53/53 | 仍需补完整逐元素 CPU reference vs device 矩阵 |
