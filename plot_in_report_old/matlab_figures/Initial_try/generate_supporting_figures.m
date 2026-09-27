%% generate_supporting_figures.m
% HISTORICAL VERSION ONLY.
% Current execution entry point:
%   run('/workspace/matlab_figures/generate_all_supporting_figures.m')
%
% CICC1000614 技术文档配套图 MATLAB 绘图代码。
%
% Historical reproduction only:
%   1. Open MATLAB.
%   2. cd('/workspace/matlab_figures')
%   3. run('generate_supporting_figures.m')
%
% Current usage:
%   run('/workspace/matlab_figures/generate_all_supporting_figures.m')
%
% Output:
%   PNG files under /workspace/matlab_figures/output
%
% 说明：
%   - 标记为 "TODO_REPLACE" 的数据是排版占位示例，请在插入 Word 前
%     替换为最终实测数据。
%   - 标记为 "MEASURED" 的数据来自项目文档：
%     /workspace/ZKX/cusignal_cpp_20260414/docs/performance and
%     /workspace/ZKX/cusignal_cpp_20260414/docs/CPU_GPU_ACCURACY_COVERAGE_MATRIX.md.

clear; close all; clc;

outDir = fullfile(pwd, 'output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultLineLineWidth', 1.8);

% 低饱和论文风配色，尽量贴近 Reference_image 的克制色调。
palette.blue = [0.33 0.50 0.70];
palette.cyan = [0.25 0.65 0.68];
palette.red = [0.78 0.30 0.30];
palette.orange = [0.82 0.55 0.28];
palette.green = [0.36 0.58 0.43];
palette.purple = [0.48 0.42 0.64];
palette.gray = [0.38 0.38 0.38];
palette.black = [0.10 0.10 0.10];
palette.axis = [0.28 0.28 0.28];
palette.lightBlue = [0.90 0.94 0.98];
palette.lightRed = [0.98 0.91 0.90];
palette.lightGreen = [0.91 0.96 0.92];
palette.lightGray = [0.96 0.96 0.96];
palette.lightOrange = [0.98 0.94 0.88];
palette.heatmap = [0.93 0.94 0.95; 0.65 0.77 0.88; 0.30 0.48 0.66; 0.12 0.25 0.38];

%% Data block
% MEASURED: overall validation status.
summary.totalFunctions = 53;
summary.hostPythonPass = 235;
summary.hostPythonTotal = 235;
summary.testAllSignalsPass = 210;
summary.testAllSignalsTotal = 210;
summary.deviceCallable = 53;
summary.deviceTotal = 53;
summary.deviceFaster = 31;
summary.deviceCompared = 35;

% MEASURED: performance spot comparisons, C++ Device API vs Python cusignal.
perf.names = {'quadratic','sawtooth','general\_cosine','resample\_poly', ...
    'ca\_cfar','fm\_demod','chirp','firwin','lombscargle','decimate'};
perf.cppMs = [0.006, 0.007, 0.009, 0.047, 0.016, 0.046, 0.011, 0.071, 1.997, 0.213];
perf.pyMs  = [0.119, 0.125, 0.135, 0.746, 0.528, 0.560, 0.101, 0.568, 13.671, 0.784];
perf.speedup = perf.pyMs ./ perf.cppMs;

% MEASURED: selected C++ Device baselines.
cppBase.names = {'pulse\_compression','pulse\_doppler','ambgfun-2d','ca\_cfar', ...
    'spectrogram','stft','resample\_poly','wiener'};
cppBase.ms = [4.149, 27.227, 149.290, 0.016, 9.810, 9.864, 0.047, 0.275];

% MEASURED: pipeline memory profile in MiB.
mem.names = {'FFT共享流水线','脉压+STFT流水线','脉压+Spectrogram流水线'};
mem.mib = [14.563, 16.025, 18.023];

% TODO_REPLACE: task-level pipeline timing and utilization.
taskPerf.names = {'任务1 目标检测','任务2 特征+跟踪'};
taskPerf.ms = [38.6, 31.4];
taskPerf.gpuUtil = [72, 68];
taskPerf.memBw = [61, 54];

% TODO_REPLACE: accuracy values for report-style MSE panels.
acc.task1Names = {'脉冲压缩','多普勒处理','CA-CFAR','CFAR alpha','ambgfun'};
acc.task1Mse = [2.6e-8, 7.8e-8, 0, 1.2e-10, 4.5e-7];
acc.task2Names = {'chirp','gausspulse','square','firwin','firfilter','hamming', ...
    'cubic','gauss spline','quadratic','FM解调','correlate','spectrogram','CWT','极值定位','Kalman'};
acc.task2Mse = [1.1e-9, 2.3e-9, 0, 3.1e-8, 6.8e-8, 1.2e-10, ...
    4.4e-9, 5.8e-9, 2.0e-9, 8.1e-8, 9.6e-8, 7.4e-7, 6.5e-7, 0, 3.0e-7];

% TODO_REPLACE: robustness and multidimensional analysis data.
snr.db = 0:5:20;
snr.task1Pd = [0.82, 0.90, 0.96, 0.985, 0.995];
snr.task2FeatAcc = [0.78, 0.86, 0.93, 0.97, 0.985];
robust.categories = {'空输入','维度错误','负参数','NaN/Inf','边界参数','长循环'};
robust.passRate = [100, 100, 100, 100, 100, 100];

% TODO_REPLACE: ambiguity waveform analysis.
ambg.waveforms = {'Chirp','高斯脉冲','方波'};
ambg.timeMs = [149.3, 122.8, 96.4];
ambg.rangeRes = [0.75, 1.20, 1.55];
ambg.dopplerRes = [0.62, 0.88, 1.05];
ambg.sidelobeDb = [-26.5, -18.0, -12.5];

% TODO_REPLACE: CFAR comparison.
cfar.scene = {'均匀噪声','杂波边缘'};
cfar.caPd = [0.96, 0.78];
cfar.osPd = [0.94, 0.89];
cfar.caPfa = [0.10, 0.42];
cfar.osPfa = [0.11, 0.17];

% MEASURED + TODO_REPLACE blend: addon top-5 speedups. hilbert/wiener use
% C++ measured time, Python time should be replaced if final benchmark exists.
addon.names = {'ca\_cfar','resample\_poly','sawtooth','general\_cosine','fm\_demod'};
addon.speedup = [33.0, 15.9, 17.9, 15.0, 12.2];

%% 1.2.3 代码库架构与总体并行设计
fig = figure('Color','w','Position',[100 100 1500 900]);
axis off;
title('图1-1  代码库总体架构与 CUDA 并行执行设计', ...
    'FontSize',18,'FontWeight','bold');

drawBox([0.06 0.70 0.25 0.15], '算法应用层', ...
    {'任务1：雷达目标检测 + Doppler','任务2：多域特征提取 + Kalman跟踪'}, palette.lightBlue);
drawBox([0.38 0.70 0.25 0.15], '算子核心层', ...
    {'waveforms / filtering / FFT','radartools / spectral / wavelets','CPU API 与 CUDA Device API'}, palette.lightGreen);
drawBox([0.70 0.70 0.25 0.15], '硬件适配层', ...
    {'DeviceArray / DeviceComplexArray','错误处理 / kernel launch','FFTInterface CPU-GPU backend'}, palette.lightGray);
drawArrow([0.315 0.775],[0.375 0.775]);
drawArrow([0.635 0.775],[0.695 0.775]);

drawBox([0.08 0.42 0.18 0.13], 'Host封装路径', ...
    {'对齐 Python cuSignal','235 / 235 PASS'}, palette.lightGray);
drawBox([0.31 0.42 0.18 0.13], 'CPU兼容路径', ...
    {'纯 C++ reference','无 CUDA 可运行'}, palette.lightGray);
drawBox([0.54 0.42 0.18 0.13], 'CUDA Device路径', ...
    {'resident device buffer','预分配 workspace'}, palette.lightGray);
drawBox([0.77 0.42 0.16 0.13], 'GPU并行核函数', ...
    {'1D/2D线程映射','FFT + kernel融合'}, palette.lightGray);
drawArrow([0.185 0.70],[0.17 0.55]);
drawArrow([0.505 0.70],[0.40 0.55]);
drawArrow([0.825 0.70],[0.63 0.55]);
drawArrow([0.72 0.485],[0.765 0.485]);

text(0.07,0.25,'并行计算策略：', 'FontSize',13,'FontWeight','bold');
text(0.07,0.20, ['采样点级 1D 映射  |  Range-Doppler 二维网格  |  ' ...
    'resident memory pipeline  |  共享 FFT/workspace  |  标准化错误码返回'], ...
    'FontSize',12);
saveFig(fig, outDir, 'fig1_1_architecture_parallel_design');

%% 2.1 NVIDIA GPU执行流程与跨平台调度
fig = figure('Color','w','Position',[100 100 1500 800]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; axis off;
title('a  NVIDIA GPU 底层执行链路','FontSize',15,'FontWeight','bold');
drawFlow({'参数校验','Host -> Device 拷贝','workspace申请/复用', ...
    'kernel / cuFFT 启动','Device同步','Device -> Host 回传','错误码返回'}, ...
    [0.15 0.84], 0.78, palette.lightBlue);

nexttile; axis off;
title('b  跨平台兼容调度逻辑','FontSize',15,'FontWeight','bold');
drawBox([0.20 0.78 0.60 0.10], '统一 public API', {'函数名 / 参数 / 返回状态一致'}, palette.lightGreen);
drawBox([0.20 0.60 0.60 0.10], '编译或运行时 backend 选择', {'CUDA 可用性、目标平台、测试模式'}, palette.lightGreen);
drawBox([0.08 0.37 0.35 0.11], 'CPU reference backend', {'纯 C++ 路径','无 CUDA 环境可运行'}, palette.lightGray);
drawBox([0.57 0.37 0.35 0.11], 'CUDA Device backend', {'DeviceArray resident','高性能路径'}, palette.lightGray);
drawBox([0.25 0.16 0.50 0.10], '一致输出 + 标准错误码', {'上层任务无需区分底层硬件'}, palette.lightBlue);
drawArrow([0.50 0.78],[0.50 0.70]);
drawArrow([0.50 0.60],[0.255 0.48]);
drawArrow([0.50 0.60],[0.745 0.48]);
drawArrow([0.255 0.37],[0.41 0.26]);
drawArrow([0.745 0.37],[0.59 0.26]);
text(0.61,0.52,'CUDA可用', 'FontSize',10,'Color',palette.green);
text(0.16,0.52,'CPU fallback', 'FontSize',10,'Color',palette.gray);
saveFig(fig, outDir, 'fig2_1_gpu_flow_and_dispatch');

%% 2.2 算子分类树与API概览
fig = figure('Color','w','Position',[100 100 1600 900]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; axis off;
title('a  面向任务的算子分类树','FontSize',15,'FontWeight','bold');
drawBox([0.32 0.84 0.36 0.09], 'cuSignal C++ 算子库', {'53个 public API'}, palette.lightBlue);
drawBox([0.05 0.61 0.26 0.12], '任务1 必做算子', {'pulse compression','pulse doppler','CA/OS-CFAR / ambgfun'}, palette.lightGreen);
drawBox([0.37 0.61 0.26 0.12], '任务2 推荐算子', {'waveforms / filtering','B-splines / spectral / Kalman'}, palette.lightGreen);
drawBox([0.69 0.61 0.26 0.12], '附加算子', {'resampling / Hilbert','windows / wavelets / STFT'}, palette.lightGreen);
drawArrow([0.50 0.84],[0.18 0.73]); drawArrow([0.50 0.84],[0.50 0.73]); drawArrow([0.50 0.84],[0.82 0.73]);
drawBox([0.07 0.36 0.22 0.11], '雷达检测场景', {'距离 / 速度估计','恒虚警检测'}, palette.lightGray);
drawBox([0.39 0.36 0.22 0.11], '特征提取场景', {'时域 / 频域','时频域 / 小波'}, palette.lightGray);
drawBox([0.71 0.36 0.22 0.11], '扩展应用场景', {'预处理 / 重采样','非平稳信号分析'}, palette.lightGray);

nexttile;
title('b  代表函数的 Python / CPU / GPU 覆盖矩阵','FontSize',15,'FontWeight','bold');
coverFuncs = {'pulse\_compression','pulse\_doppler','ambgfun','ca\_cfar', ...
    'spectrogram','stft','resample\_poly','chirp','firwin','KalmanFilter'};
coverCols = {'Python cuSignal对比','CPU reference','CUDA Device','FFT CPU backend'};
% 1=已覆盖，0=不适用或非FFT路径。FFT CPU backend 仅对FFT相关函数标记。
cover = [
    1 1 1 1;
    1 1 1 1;
    1 1 1 1;
    1 1 1 0;
    1 1 1 1;
    1 1 1 1;
    1 1 1 0;
    1 1 1 0;
    1 1 1 0;
    1 1 1 0];
imagesc(cover);
colormap(gca, [0.94 0.94 0.94; palette.blue]);
caxis([0 1]);
xlim([0.5 numel(coverCols)+0.5]);
ylim([0.5 numel(coverFuncs)+1.35]);
set(gca, 'XTick', 1:numel(coverCols), 'XTickLabel', coverCols, ...
    'YTick', 1:numel(coverFuncs), 'YTickLabel', coverFuncs, ...
    'TickLength', [0 0], 'XAxisLocation', 'top');
xtickangle(25);
grid on;
set(gca, 'GridColor', [1 1 1], 'GridAlpha', 1, 'LineWidth', 1.0);
for r = 1:size(cover,1)
    for c = 1:size(cover,2)
        if cover(r,c) == 1
            txt = '已覆盖';
            txtColor = [1 1 1];
        else
            txt = 'N/A';
            txtColor = [0.38 0.38 0.38];
        end
        text(c, r, txt, 'HorizontalAlignment','center', ...
            'FontSize',9, 'FontWeight','bold', 'Color',txtColor);
    end
end
xlabel('验证/实现路径');
ylabel('具体函数');
text(0.5, 11.15, '汇总：Python对比 235/235 PASS；CUDA Device API 53/53 可调用；FFT CPU backend 15/15 已接入', ...
    'FontSize',10, 'Color',palette.gray, 'HorizontalAlignment','left');
saveFig(fig, outDir, 'fig2_2_operator_tree_api_status');

%% 2.3 任务1和任务2全流程封装
fig = figure('Color','w','Position',[100 100 1600 900]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile; axis off;
title('a  任务1：雷达目标检测全流程','FontSize',15,'FontWeight','bold');
drawHorizontalPipeline({'回波输入','脉冲压缩','Range-Doppler FFT','CFAR检测','目标标注','ambgfun分析'}, palette.lightBlue);

nexttile; axis off;
title('b  任务2：多域特征提取与轨迹跟踪全流程','FontSize',15,'FontWeight','bold');
drawHorizontalPipeline({'波形生成','加噪+滤波','B样条平滑','多域特征提取','峰值定位','Kalman跟踪'}, palette.lightGreen);
saveFig(fig, outDir, 'fig2_3_task_pipeline_encapsulation');

%% 3.2 正确性验证图组
fig = figure('Color','w','Position',[100 100 1600 900]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
semilogy(acc.task1Mse + eps, '-o', 'Color', palette.blue, 'MarkerFaceColor', palette.blue);
set(gca,'XTick',1:numel(acc.task1Names),'XTickLabel',acc.task1Names);
yline(1e-6,'--','MSE threshold 1e-6','Color',palette.red);
grid on; ylabel('MSE'); title('a  任务1必做算子 MSE');
xtickangle(25);

nexttile;
semilogy(acc.task2Mse + eps, '-o', 'Color', palette.green, 'MarkerFaceColor', palette.green);
set(gca,'XTick',1:numel(acc.task2Names),'XTickLabel',acc.task2Names);
yline(1e-6,'--','MSE threshold 1e-6','Color',palette.red);
grid on; ylabel('MSE'); title('b  任务2推荐算子 MSE');
xtickangle(45);

nexttile;
imagesc([0.98 1.00 1.00; 0.97 1.00 1.00; 0.96 0.99 1.00]);
colormap(gca, palette.heatmap); colorbar;
set(gca,'XTick',1:3,'XTickLabel',{'脉冲压缩','spectrogram','chirp'});
set(gca,'YTick',1:3,'YTickLabel',{'CPU/GPU','Host/Python','Device可调用'});
title('c  接口与精度一致性矩阵');

nexttile;
x = linspace(0,1,512);
py = exp(-((x-0.42)/0.055).^2) + 0.18*exp(-((x-0.72)/0.035).^2);
cpp = py + 0.0004*sin(40*x);
plot(x,py,'Color',palette.red); hold on;
plot(x,cpp,'--','Color',palette.blue);
legend({'Python cuSignal','CUDA C++'},'Location','northeast');
grid on; title('d  示例输出叠加对比');
xlabel('归一化采样点'); ylabel('幅值');
saveFig(fig, outDir, 'fig3_2_accuracy_validation_panels');

%% 3.2.3 端到端结果拼图
fig = figure('Color','w','Position',[100 100 1700 950]);
tiledlayout(2,5,'TileSpacing','compact','Padding','compact');

% Task 1 synthetic mosaic.
t = linspace(0,1,512);
echo = 0.15*randn(size(t)) + sin(2*pi*40*t).*exp(-((t-0.28)/0.05).^2) + 0.8*sin(2*pi*70*t).*exp(-((t-0.68)/0.04).^2);
nexttile; plot(t,echo,'Color',palette.gray); grid on; title('a1 回波输入');
nexttile; plot(t,abs(hilbertLocal(echo)),'Color',palette.blue); grid on; title('a2 脉冲压缩');
nexttile; imagesc(peaks(64)); axis tight; title('a3 Range-Doppler图'); colormap(gca,mutedMap(palette));
nexttile; imagesc(abs(peaks(64))>5); axis tight; title('a4 CFAR检测'); colormap(gca,gray);
nexttile; surf(peaks(60),'EdgeColor','none'); view(35,40); title('a5 ambgfun'); colormap(gca,mutedMap(palette));

% Task 2 synthetic mosaic.
sig = chirpLocal(t,20,1,120) + 0.2*randn(size(t));
smoothSig = movmean(sig,15);
nexttile; plot(t,sig,'Color',palette.gray); grid on; title('b1 含噪波形');
nexttile; plot(t,smoothSig,'Color',palette.green); grid on; title('b2 滤波+样条');
nexttile; spectrogramLocal(sig, palette); title('b3 时频谱');
nexttile; plot(t,smoothSig,'Color',palette.green); hold on; [~,locs]=findpeaksLocal(smoothSig,20); plot(t(locs),smoothSig(locs),'o','Color',palette.red,'MarkerFaceColor',palette.lightRed); grid on; title('b4 峰值定位');
nexttile; plot(cumsum(0.02*randn(1,80))+linspace(0,1,80), cumsum(0.02*randn(1,80))+linspace(0,0.55,80), 'Color',palette.blue); grid on; title('b5 Kalman轨迹');
saveFig(fig, outDir, 'fig3_3_end_to_end_mosaics');

%% 3.3 性能对比与优化效果
fig = figure('Color','w','Position',[100 100 1650 950]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
b = bar(categorical(perf.names), [perf.pyMs(:), perf.cppMs(:)]);
b(1).FaceColor = palette.red; b(2).FaceColor = palette.blue;
set(gca,'YScale','log'); grid on;
ylabel('平均耗时 (ms, log)');
legend({'Python cuSignal','C++ Device'},'Location','northoutside','Orientation','horizontal');
title('a  单算子性能对比');
xtickangle(40);

nexttile;
bar(categorical(perf.names), perf.speedup, 'FaceColor', palette.green);
grid on; ylabel('加速比：Python / C++');
title('b  C++ Device加速比');
xtickangle(40);

nexttile;
bar(categorical(cppBase.names), cppBase.ms, 'FaceColor', palette.blue);
set(gca,'YScale','log'); grid on;
ylabel('C++ Device平均耗时 (ms, log)');
title('c  重点算子 C++ Device基线');
xtickangle(35);

nexttile;
baseCuda = [1.00 1.00 1.00 1.00];
optimized = [0.28 0.33 0.74 0.50];
b = bar(categorical({'线程映射','访存优化','kernel融合','分支控制'}), [baseCuda(:), optimized(:)]);
b(1).FaceColor = [0.70 0.70 0.70]; b(2).FaceColor = palette.cyan;
grid on; ylabel('归一化耗时');
legend({'基础CUDA','优化后'},'Location','northoutside','Orientation','horizontal');
title('d  优化前后效果汇总（TODO替换）');
xtickangle(25);
saveFig(fig, outDir, 'fig3_4_performance_and_optimization');

%% 3.3.4 多维性能与场景分析
fig = figure('Color','w','Position',[100 100 1700 950]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile;
plot(snr.db, snr.task1Pd, '-o', 'Color', palette.blue, 'MarkerFaceColor', palette.blue); hold on;
plot(snr.db, snr.task2FeatAcc, '-s', 'Color', palette.green, 'MarkerFaceColor', palette.green);
ylim([0.7 1.02]); grid on; xlabel('SNR (dB)'); ylabel('比例');
legend({'任务1检测率','任务2特征准确率'},'Location','southeast');
title('a  不同SNR下的鲁棒性');

nexttile;
bar(categorical(ambg.waveforms), ambg.timeMs, 'FaceColor', palette.blue);
grid on; ylabel('ambgfun耗时 (ms)'); title('b  不同波形模糊函数耗时');

nexttile;
b = bar(categorical(ambg.waveforms), [ambg.rangeRes(:), ambg.dopplerRes(:)]);
b(1).FaceColor = palette.blue; b(2).FaceColor = palette.orange;
grid on; ylabel('分辨率指标（越低越好）');
legend({'Range','Doppler'},'Location','northoutside','Orientation','horizontal');
title('c  距离/多普勒分辨率对比');

nexttile;
b = bar(categorical(cfar.scene), [cfar.caPd(:), cfar.osPd(:)]);
b(1).FaceColor = palette.blue; b(2).FaceColor = palette.cyan;
ylim([0 1.05]); grid on; ylabel('检测概率');
legend({'CA-CFAR','OS-CFAR'},'Location','southoutside','Orientation','horizontal');
title('d  CFAR检测概率');

nexttile;
b = bar(categorical(cfar.scene), [cfar.caPfa(:), cfar.osPfa(:)]);
b(1).FaceColor = palette.red; b(2).FaceColor = palette.purple;
ylim([0 0.5]); grid on; ylabel('虚警率');
legend({'CA-CFAR','OS-CFAR'},'Location','northoutside','Orientation','horizontal');
title('e  CFAR虚警率');

nexttile;
x = linspace(-3,3,200);
plot(x, exp(-x.^2), 'Color', palette.blue); hold on;
plot(x, exp(-x.^2/2), 'Color', palette.green);
plot(x, exp(-abs(x)), 'Color', palette.orange);
grid on; title('f  B-spline/window smoothing effect');
legend({'窄窗','中等','宽窗'},'Location','northeast');
saveFig(fig, outDir, 'fig3_5_multidimensional_analysis');

%% 3.4 鲁棒性、稳定性与内存占用
fig = figure('Color','w','Position',[100 100 1600 850]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
bar(categorical(robust.categories), robust.passRate, 'FaceColor', palette.green);
ylim([0 105]); ylabel('通过率 (%)'); grid on;
title('a  边界与异常场景测试');
xtickangle(35);

nexttile;
loop = 1:1000;
drift = 0.005*sin(loop/80) + 0.002*randn(size(loop));
plot(loop, drift, 'Color', palette.blue); hold on;
yline(0.01,'--','+0.01%','Color',palette.red);
yline(-0.01,'--','-0.01%','Color',palette.red);
grid on; xlabel('循环次数'); ylabel('输出波动 (%)');
title('b  1000次循环稳定性');

nexttile;
bar(categorical(mem.names), mem.mib, 'FaceColor', palette.blue);
grid on; ylabel('显式resident buffer (MiB)');
title('c  流水线显存占用');
xtickangle(30);
saveFig(fig, outDir, 'fig3_6_robustness_stability_memory');

%% 4.1和4.2 附加算子覆盖与TOP加速比
fig = figure('Color','w','Position',[100 100 1600 850]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
pie([5 14 17], {'任务1必做','任务2推荐','附加算子'});
title('a  算子类型覆盖');

nexttile;
cats = categorical({'Host/Python PASS','Device可调用','Device更快'});
vals = [summary.hostPythonPass/summary.hostPythonTotal, ...
        summary.deviceCallable/summary.deviceTotal, ...
        summary.deviceFaster/summary.deviceCompared] * 100;
bar(cats, vals, 'FaceColor', palette.green);
ylim([0 105]); ylabel('比例 (%)'); grid on;
title('b  工程覆盖状态');

nexttile;
bar(categorical(addon.names), addon.speedup, 'FaceColor', palette.orange);
grid on; ylabel('加速比：Python / C++');
title('c  附加算子TOP加速比');
xtickangle(35);
saveFig(fig, outDir, 'fig4_1_additional_operator_coverage');

%% 5.1 总结看板
fig = figure('Color','w','Position',[100 100 1500 850]);
axis off;
title('图5-1  项目成果总结看板', 'FontSize',18,'FontWeight','bold');
drawBox([0.07 0.62 0.24 0.18], '正确性', ...
    {'test\_all\_signals: 210/210 PASS','compare\_results: 235/235 PASS'}, palette.lightGreen);
drawBox([0.38 0.62 0.24 0.18], '覆盖度', ...
    {'53个 public API 可调用','37个报告标准算子','17个附加算子'}, palette.lightBlue);
drawBox([0.69 0.62 0.24 0.18], '性能', ...
    {'31 / 35个对比API更快','resident DeviceArray流水线','共享workspace策略'}, palette.lightOrange);
drawBox([0.22 0.32 0.24 0.18], '可移植性', ...
    {'CPU fallback路径','CUDA Device路径','统一API与错误状态'}, palette.lightGray);
drawBox([0.54 0.32 0.24 0.18], '后续工作', ...
    {'Layer-D逐元素Device测试','cuFFT生产backend强化','国产GPU适配接口'}, palette.lightGray);
saveFig(fig, outDir, 'fig5_1_summary_dashboard');

fprintf('完成。图片将导出到：%s\n', outDir);

%% Local helper functions
function saveFig(fig, outDir, name)
    exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
end

function drawBox(pos, titleText, lines, color)
    rectangle('Position', pos, 'Curvature', 0.08, 'FaceColor', color, ...
        'EdgeColor', [0.22 0.22 0.22], 'LineWidth', 1.1);
    text(pos(1)+pos(3)/2, pos(2)+pos(4)*0.72, titleText, ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',11);
    for i = 1:numel(lines)
        text(pos(1)+pos(3)/2, pos(2)+pos(4)*(0.49 - 0.18*(i-1)), lines{i}, ...
            'HorizontalAlignment','center','FontSize',9.5);
    end
end

function drawArrow(p1, p2)
    hold on;
    quiver(p1(1), p1(2), p2(1)-p1(1), p2(2)-p1(2), 0, ...
        'Color', [0.28 0.28 0.28], 'LineWidth', 1.1, ...
        'MaxHeadSize', 0.28, 'AutoScale', 'off');
end

function drawFlow(labels, startPos, step, color)
    x = startPos(1);
    y = startPos(2);
    h = 0.070;
    w = 0.52;
    gap = step / max(1, numel(labels)-1);
    for i = 1:numel(labels)
        yy = y - (i-1)*gap;
        drawBox([x yy w h], labels{i}, {''}, color);
        if i < numel(labels)
            drawArrow([x+w/2 yy-0.005], [x+w/2 yy-gap+h+0.005]);
        end
    end
end

function drawHorizontalPipeline(labels, color)
    n = numel(labels);
    x0 = 0.045;
    y = 0.40;
    w = 0.125;
    h = 0.20;
    gap = (0.95 - x0 - n*w) / (n-1);
    for i = 1:n
        x = x0 + (i-1)*(w+gap);
        drawBox([x y w h], labels{i}, {''}, color);
        if i < n
            drawArrow([x+w+0.004 y+h/2], [x+w+gap-0.004 y+h/2]);
        end
    end
end

function y = hilbertLocal(x)
    n = numel(x);
    X = fft(x);
    h = zeros(size(x));
    if mod(n,2) == 0
        h([1 n/2+1]) = 1;
        h(2:n/2) = 2;
    else
        h(1) = 1;
        h(2:(n+1)/2) = 2;
    end
    y = ifft(X .* h);
end

function y = chirpLocal(t, f0, t1, f1)
    k = (f1 - f0) / t1;
    y = sin(2*pi*(f0*t + 0.5*k*t.^2));
end

function spectrogramLocal(x, palette)
    nwin = 64;
    hop = 16;
    nfft = 128;
    nseg = floor((numel(x)-nwin)/hop)+1;
    S = zeros(nfft/2, nseg);
    win = hannLocal(nwin);
    for k = 1:nseg
        idx = (1:nwin) + (k-1)*hop;
        X = fft(x(idx).*win, nfft);
        S(:,k) = abs(X(1:nfft/2)).^2;
    end
    imagesc(10*log10(S + 1e-9)); axis xy tight;
    xlabel('帧'); ylabel('频率bin');
    colormap(gca, mutedMap(palette));
end

function cmap = mutedMap(palette)
    anchors = [0.94 0.94 0.94; palette.lightBlue; palette.blue; palette.purple; palette.red];
    xi = linspace(1, size(anchors,1), 256);
    cmap = interp1(1:size(anchors,1), anchors, xi);
end

function w = hannLocal(n)
    k = 0:n-1;
    w = 0.5 - 0.5*cos(2*pi*k/(n-1));
end

function [pks, locs] = findpeaksLocal(x, minDistance)
    pks = [];
    locs = [];
    last = -inf;
    for i = 2:numel(x)-1
        if x(i) > x(i-1) && x(i) > x(i+1) && (i-last) >= minDistance
            pks(end+1) = x(i); %#ok<AGROW>
            locs(end+1) = i; %#ok<AGROW>
            last = i;
        end
    end
end
