function fig3_5_operator_latency_bar_errorbar()
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

%% ===================== 算子名称 =====================
task2Names = {'chirp','gausspulse','sawtooth','square','firwin','firfilter', ...
    'hamming','cubic','fm_demod','correlate','spectrogram','cwt','ricker', ...
    'argrelextrema','KalmanFilter'};
task1Names = {'pulse_compression','pulse_doppler','ca_cfar','cfar_alpha','ambgfun'};
allNames = [task2Names, task1Names];
numOps = numel(allNames);

%% ===================== 从 04_算子结果 读取真实耗时 =====================
% 数据来源：04_算子结果/task_operators/data/RB_20260810T1446CST_STAGE07/cpu_gpu_comparison.csv
% 筛选规则：source_partition=formal & run_type=formal & status=PASS
% 选取规则：每个算子在所有 case 中取 gpu_mean_ms 最小的一行
%   opMean(i) = 该 case 的 gpu_mean_ms（单位 ms）
%   opStd(i)  = 该 case 的 gpu_std_ms（单位 ms）
% 注：算子名顺序与 allNames 一一对应；KalmanFilter 在 csv 中写作 kalman_filter
opMean = [ ...
    0.16789680;  % chirp
    0.16808258;  % gausspulse
    0.29242296;  % sawtooth
    0.16833156;  % square
    0.27575765;  % firwin
    0.43966900;  % firfilter
    0.10113762;  % hamming
    0.18570084;  % cubic
    0.19215198;  % fm_demod
    0.28321222;  % correlate
    0.56421960;  % spectrogram
    0.59264276;  % cwt
    0.24970015;  % ricker
    0.44861558;  % argrelextrema
    0.95542265;  % KalmanFilter
    0.70488469;  % pulse_compression
    0.34475853;  % pulse_doppler
    0.32025264;  % ca_cfar
    0.17334826;  % cfar_alpha
    0.49969187]; % ambgfun

opStd = [ ...
    0.006440929673580;  % chirp
    0.005805783313525;  % gausspulse
    0.014036673667874;  % sawtooth
    0.003413415358025;  % square
    0.003755722610564;  % firwin
    0.006398229770804;  % firfilter
    0.002356695431234;  % hamming
    0.010421002849745;  % cubic
    0.004738040008231;  % fm_demod
    0.012301533663393;  % correlate
    0.018554679239480;  % spectrogram
    0.015879596723544;  % cwt
    0.019908574939646;  % ricker
    0.009153579968712;  % argrelextrema
    0.035855122781375;  % KalmanFilter
    0.024813497500633;  % pulse_compression
    0.007546806241656;  % pulse_doppler
    0.010721694441197;  % ca_cfar
    0.008108267028928;  % cfar_alpha
    0.034843484792901]; % ambgfun

% 按均值降序排序
[sortedMean, sortIdx] = sort(opMean, 'descend');
sortedStd  = opStd(sortIdx);
sortedNames = allNames(sortIdx);

%% ===================== 配色方案：绿色由浅到深 =====================
% 按排序位置渐变：position 1（最慢）= 浅色，position numOps（最快）= 深色
colorLight = [0.80 0.90 0.80];  % 浅绿（慢端）
colorDeep  = [0.20 0.55 0.35];  % 深绿（快端）
barColors = zeros(numOps, 3);
for i = 1:numOps
    t = (i - 1) / (numOps - 1);  % t: 0(最慢) → 1(最快)
    barColors(i, :) = colorLight * (1 - t) + colorDeep * t;
end

%% ===================== 创建图形 =====================
fig = figure('Color','w');
set(fig, 'Units', 'centimeters', 'Position', [3 3 16 14]);

ax = axes('Parent', fig);
hold(ax, 'on');

yPos = 1:numOps;
barHeight = 0.55;

% 绘制横向条形（均值），使用渐变颜色
for i = 1:numOps
    barh(ax, yPos(i), sortedMean(i), ...
        'FaceColor', barColors(i, :), ...
        'EdgeColor', 'none', ...
        'BarWidth', barHeight);
end

% 绘制水平误差棒（均值 ± 标准差）
capSize = 0.15;  % 误差棒端盖长度
for i = 1:numOps
    xCenter = sortedMean(i);
    xLeft   = max(xCenter - sortedStd(i), 0);
    xRight  = xCenter + sortedStd(i);
    yC      = yPos(i);
    
    % 误差主线
    plot([xLeft, xRight], [yC, yC], 'k-', 'LineWidth', 1.2);
    % 左端盖
    plot([xLeft, xLeft], [yC - capSize, yC + capSize], 'k-', 'LineWidth', 1.2);
    % 右端盖
    plot([xRight, xRight], [yC - capSize, yC + capSize], 'k-', 'LineWidth', 1.2);
end

% 均值线
meanVal = mean(sortedMean);
xline(ax, meanVal, '--', ...
    'LineWidth', 1.2, ...
    'Color', [0.35 0.35 0.35], ...
    'Label', sprintf('  均值 %.1f ms', meanVal), ...
    'LabelVerticalAlignment', 'bottom', ...
    'LabelOrientation', 'horizontal');

% 条形末端数值标注（均值 ± 标准差）
maxVal = max(sortedMean + sortedStd);
offset = maxVal * 0.02;
for i = 1:numOps
    text(ax, sortedMean(i) + sortedStd(i) + offset, yPos(i), ...
        sprintf('%.3f ± %.3f ms', sortedMean(i), sortedStd(i)), ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 8);
end

% 坐标轴设置
ax.YTick = yPos;
ax.YTickLabel = sortedNames;
ax.TickLabelInterpreter = 'none';
ax.XLabel.String = '算子耗时 (ms)';
ax.XLabel.FontSize = 10;
ax.XLabel.FontWeight = 'bold';
ax.YLabel.String = '算子名称';
ax.YLabel.FontSize = 10;
ax.YLabel.FontWeight = 'bold';
ax.Title.String = '算子耗时对比（GPU mean ± std）';
ax.Title.FontSize = 11;
ax.Title.FontWeight = 'bold';
ax.FontSize = 9;
ax.XLim = [0, maxVal * 1.25];
ax.YLim = [0.3, numOps + 0.7];
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.XMinorGrid = 'on';

set(ax, 'FontName', 'Microsoft YaHei');
end
