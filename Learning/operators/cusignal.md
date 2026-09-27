全算子学习顺序总表

> **编号说明：**下表共整理本地 `ZKX/cusignal-23.08.00` 中的 91 个公开算子或 API；其中序号加粗的 **53 项**，才是《2026 集创赛中科芯赛题》附录明确指定的竞赛算子。官方的“53 个算子”是任务范围，不是 cuSignal 全部 API 的数量。
>
> 本地条目多于 53 项，主要是因为统计口径更宽：还纳入了卷积、声学、波束形成等未被赛题选中的模块，以及 `get_window`、`choose_conv_method`、`correlation_lags`、`argrelmax` 等调度器、辅助函数或便捷封装。当前源码版本为 cuSignal 23.08.00（2023-08-09），但赛题 PDF 没有给出其选取算子时所依据的精确 cuSignal 版本，因此不能据此断定“多出的 38 项是因为本地版本更新”。更直接的反证是：本地 `CHANGELOG.md` 的 cuSignal 0.1（2019-11-04）记录中已经包含 `convolve`、`fftconvolve`、`choose_conv_method`、`periodogram`、`welch`、`argrelmax` 等多项赛题未选 API。可见数量差异主要来自范围筛选，而不是版本新旧；即使存在版本差异，也只可能影响少量条目。竞赛实现与验收仍应以 PDF 指定的 53 项为边界。

| 序号         | 算子                        | 所属模块             | 核心原理                                                            |
| ------------ | --------------------------- | -------------------- | ------------------------------------------------------------------- |
| **1**  | **sawtooth**          | waveforms            | 锯齿波/三角波，`x=2(t/T-⌊t/T+1-w⌋)`，width 控制上升占比         |
| **2**  | **square**            | waveforms            | 方波，`sign(sin(2πft)-2duty+1)`，含奇次谐波                      |
| **3**  | **unit_impulse**      | waveforms            | Kronecker δ[n-idx]，离散脉冲激励                                   |
| **4**  | **chirp**             | waveforms            | 瞬时频率 f(t)=f₀+βt，LFM 调频信号，时宽带宽积决定分辨力           |
| **5**  | **gausspulse**        | waveforms            | 高斯包络调制正弦`exp(-at²)·cos(2πf_c·t)`，时频同时集中        |
| **6**  | **quadratic**         | bsplines             | 二次 B 样条，分段二次，C¹ 连续插值                                 |
| **7**  | **cubic**             | bsplines             | 三次 B 样条，C² 连续，信号重建权重                                 |
| **8**  | **gauss_spline**      | bsplines             | B 样条高斯极限近似`1/√(2πσ²)·exp(-x²/2σ²)`                |
| 9            | boxcar                      | windows              | 矩形窗 w[n]=1，旁瓣 -13 dB                                          |
| **10** | **triang**            | windows              | 三角窗 `1-                                                          |
| 11           | bartlett                    | windows              | Bartlett 窗，端点为零的三角窗，频域 sinc²                          |
| 12           | cosine                      | windows              | 余弦窗`sin(π(n+0.5)/N)`，半余弦                                  |
| 13           | hann                        | windows              | 升余弦窗`0.5-0.5cos(2πn/(N-1))`，旁瓣 -32 dB                     |
| **14** | **hamming**           | windows              | Hamming 窗`0.54-0.46cos(…)`，旁瓣 -43 dB                         |
| 15           | general_hamming             | windows              | 广义 Hamming 族`α-(1-α)cos(…)`，α=0.54→Hamming，0.5→Hann    |
| **16** | **general_cosine**    | windows              | 统一余弦窗框架`Σaₖcos(2πkn/(N-1))`                             |
| 17           | blackman                    | windows              | Blackman 窗，3 项余弦，旁瓣 -58 dB                                  |
| 18           | blackmanharris              | windows              | 4 项 Blackman-Harris 窗，旁瓣 -92 dB                                |
| 19           | nuttall                     | windows              | Nuttall4c 窗，4 项余弦，与 Blackman-Harris 类似                     |
| 20           | flattop                     | windows              | 平顶窗，5 项余弦，幅度测量精度最高                                  |
| 21           | barthann                    | windows              | 修正 Bartlett-Hann 窗                                               |
| 22           | bohman                      | windows              | Bohman 窗，旁瓣 -46 dB                                              |
| **23** | **parzen**            | windows              | Parzen 窗，三角窗卷积得到，旁瓣 -53 dB                              |
| 24           | tukey                       | windows              | 锥化余弦窗，α=0 矩形，α=1 Hann，SAR 常用                          |
| 25           | gaussian                    | windows              | 高斯窗`exp(-½(n/σ)²)`，时频最优不确定性                        |
| **26** | **general_gaussian**  | windows              | 广义高斯窗 `exp(-½                                                 |
| 27           | exponential                 | windows              | 指数窗 `exp(-                                                       |
| **28** | **kaiser**            | windows              | Kaiser 窗`I₀(β√(1-(…))²)/I₀(β)`，β 可调主瓣/旁瓣          |
| **29** | **taylor**            | windows              | Taylor 窗，近似等旁瓣 Chebyshev，雷达天线加权                       |
| **30** | **chebwin**           | windows              | Dolph-Chebyshev 等纹波最优窗，Chebyshev 多项式逼近                  |
| 31           | get_window                  | windows              | 窗名→窗函数调度器                                                  |
| 32           | kaiser_beta                 | filter_design        | 由阻带衰减 a 计算 Kaiser β 参数                                    |
| 33           | kaiser_atten                | filter_design        | 由 Kaiser FIR 阶数和过渡带宽计算衰减                                |
| **34** | **firwin**            | filter_design        | 理想低通 sinc→加窗截断→FIR 系数                                   |
| **35** | **firwin2**           | filter_design        | 任意目标频响→采样→IFFT→加窗→系数                                |
| 36           | cmplx_sort                  | filter_design        | 按模值排序复数根，极零点排序辅助                                    |
| **37** | **detrend**           | filtering            | 去除均值或最小二乘线性趋势                                          |
| **38** | **freq_shift**        | filtering            | 频域平移`x[n]·exp(-j2πfn/fs)`，载波搬移                         |
| **39** | **firfilter**         | filtering            | FIR 卷积`y[n]=Σbₖx[n-k]`，线性时不变                            |
| 40           | lfilter                     | filtering            | 差分方程`a₀y[n]=Σbₖx[n-k]-Σaₖy[n-k]`（目前仅 FIR）           |
| **41** | **lfilter_zi**        | filtering            | 解 A·zi+B 求 IIR 稳态初始条件                                      |
| 42           | firfilter_zi                | filtering            | FIR 稳态初态，lfilter_zi 简化版                                     |
| **43** | **firfilter2**        | filtering            | 前向 FIR→反转→反向 FIR→反转，零相位双向                          |
| 44           | filtfilt                    | filtering            | 零相位滤波通用接口（a=[1]降级 FIR）                                 |
| **45** | **sosfilt**           | filtering            | Direct Form II Transposed SOS 递推，IIR 数值稳定实现                |
| **46** | **wiener**            | filtering            | Wiener 滤波`ŷ=μ+(σ²-ν²)/σ²·(x-μ)`，最小均方误差降噪     |
| **47** | **hilbert**           | filtering            | FFT→负频置零+正频加倍→IFFT，解析信号构造                          |
| **48** | **hilbert2**          | filtering            | 2D FFT→频域 mask→IFFT，二维解析信号                               |
| **49** | **channelize_poly**   | filtering            | 多相分解+batched FFT→多窄带子信道，频分复用解调                    |
| **50** | **decimate**          | filtering (resample) | 低通 FIR 抗混叠→下采样                                             |
| **51** | **upfirdn**           | filtering (resample) | 上采样(插零)→FIR→下采样，多速率原子操作                           |
| **52** | **resample**          | filtering (resample) | FFT→频域补零/截断→IFFT，频域重采样                                |
| **53** | **resample_poly**     | filtering (resample) | 多相分解 FIR+upfirdn，时域高效重采样                                |
| 54           | convolve                    | convolution          | 一维/N 维卷积`Σaₖb[n-k]`，direct 或 FFT                         |
| 55           | fftconvolve                 | convolution          | FFT(A)·FFT(B)→IFFT，O(NlogN) 快速卷积                             |
| 56           | convolve2d                  | convolution          | 二维滑动卷积，图像滤波                                              |
| **57** | **correlate**         | convolution          | 互相关`Σx[n+m]·y*[n]`，匹配检测与时延估计                       |
| **58** | **correlate2d**       | convolution          | 二维互相关，模板匹配                                                |
| 59           | correlation_lags            | convolution          | 1D 互相关滞后索引数组                                               |
| 60           | choose_conv_method          | convolution          | 根据输入大小自动选择 direct 或 FFT 卷积                             |
| 61           | convolve1d2o                | convolution          | 1D 信号与 2 阶滤波器卷积                                            |
| 62           | convolve1d3o                | convolution          | 1D 信号与 3 阶滤波器卷积                                            |
| **63** | **fm_demod**          | demod                | angle→unwrap→diff 提取瞬时频率，鉴频器                            |
| 64           | periodogram                 | spectral_analysis    | 周期图法 `                                                          |
| 65           | welch                       | spectral_analysis    | 分帧→加窗→batched FFT→模平方→平均，降低方差                     |
| **66** | **spectrogram**       | spectral_analysis    |                                                                     |
| **67** | **stft**              | spectral_analysis    | 分帧→加窗→batched FFT，短时 Fourier 变换                          |
| **68** | **istft**             | spectral_analysis    | 重叠相加 IFFT→求和→归一化，逆 STFT                                |
| **69** | **csd**               | spectral_analysis    | 互谱密度 conj(STFT_x)·STFT_y→平均                                 |
| 70           | coherence                   | spectral_analysis    | 幅度平方相干 `                                                      |
| **71** | **lombscargle**       | spectral_analysis    | 非均匀采样周期图，不规则数据周期检测                                |
| **72** | **vectorstrength**    | spectral_analysis    | 向量强度 `                                                          |
| **73** | **ricker**            | wavelets             | Ricker 小波，高斯二阶导`(1-t²/a²)exp(-t²/2a²)`                |
| **74** | **morlet**            | wavelets             | 复 Morlet 小波`exp(jω₀x)·exp(-x²/2)`，频域局部化              |
| **75** | **morlet2**           | wavelets             | 归一化 Morlet2，能量归一化，与 cwt 兼容                             |
| **76** | **qmf**               | wavelets             | 正交镜像滤波器`g[k]=(-1)^k·h[L-1-k]`                             |
| **77** | **cwt**               | wavelets             | 连续小波变换`Σx[n]·ψ*[(n-τ)/a]`，尺度-平移分析                |
| **78** | **argrelextrema**     | peak_finding         | 通用比较器，与邻域比较检测极值                                      |
| 79           | argrelmax                   | peak_finding         | 局部极大值便捷入口                                                  |
| 80           | argrelmin                   | peak_finding         | 局部极小值便捷入口                                                  |
| **81** | **KalmanFilter**      | estimation           | predict-update 递推`x⁻=Fx⁺, K=P⁻Hᵀ/(HP⁻Hᵀ+R)`，最小方差估计 |
| 82           | real_cepstrum               | acoustics            | 实倒谱 `IFFT(log                                                    |
| 83           | complex_cepstrum            | acoustics            | 复倒谱 `IFFT(log                                                    |
| 84           | inverse_complex_cepstrum    | acoustics            | 复倒谱逆变换`IFFT(exp(FFT(ceps)))`，信号重建                      |
| 85           | minimum_phase               | acoustics            | 实倒谱→窗口截取因果部分→最小相位重构                              |
| **86** | **pulse_compression** | radartools           | 频域匹配滤波 FFT(回波)·conj(FFT(模板))→IFFT，距离压缩             |
| **87** | **pulse_doppler**     | radartools           | 慢时间轴 FFT，Doppler 频移→径向速度                                |
| **88** | **cfar_alpha**        | radartools           | CFAR 阈值乘子`α=N(Pfa^{-1/N}-1)`，恒虚警保证                     |
| **89** | **ca_cfar**           | radartools           | CA-CFAR 二维滑动窗口，参考单元均值自适应检测                        |
| **90** | **ambgfun**           | radartools           | 模糊函数`χ(τ,f)=∫x(t)y*(t-τ)exp(j2πft)dt`，波形分辨力        |
| 91           | mvdr                        | beamformers          | MVDR 波束形成`w=R⁻¹sv/(svᴴR⁻¹sv)`，最小方差无失真响应        |
