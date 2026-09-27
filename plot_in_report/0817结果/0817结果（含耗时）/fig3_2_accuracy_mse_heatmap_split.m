function fig3_2_accuracy_mse_heatmap_split()
% 算子精度RMSE热力图（拆分为两张子图） - 基于53算子最新数据
% 第一张：信号生成/窗/小波/样条/卷积/滤波器设计（25算子）
% 第二张：滤波/频谱/峰值/雷达/解调/估计（28算子）
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

%% 数据读取（最新数据：53算子 x 5数据类型 最差精度候选）
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, '04_算子结果', '绘图数据');
accuracyFile = fullfile(dataDir, '53x5_最差精度候选.csv');
T = readtable(accuracyFile);

% 修复BOM导致的列名问题
if ~isprop(T, 'operator_name')
    T.Properties.VariableNames{1} = 'operator_name';
end

%% 算子按模块组织（模块顺序：信号生成→滤波→频谱→雷达→其他）
moduleNames = {'waveforms', 'windows', 'wavelets', 'bsplines', 'convolution', ...
    'filter_design', 'filtering', 'spectral_analysis', 'peak_finding', ...
    'radartools', 'demod', 'estimation'};

moduleOps = {
    {'chirp', 'gausspulse', 'sawtooth', 'square', 'unit_impulse'}, ...
    {'chebwin', 'general_cosine', 'general_gaussian', 'hamming', 'kaiser', 'parzen', 'taylor', 'triang'}, ...
    {'cwt', 'morlet', 'morlet2', 'qmf', 'ricker'}, ...
    {'cubic', 'gauss_spline', 'quadratic'}, ...
    {'correlate', 'correlate2d'}, ...
    {'firwin', 'firwin2'}, ...
    {'channelize_poly', 'decimate', 'detrend', 'firfilter', 'firfilter2', ...
     'freq_shift', 'hilbert', 'hilbert2', 'lfilter_zi', 'resample', ...
     'resample_poly', 'sosfilt', 'upfirdn', 'wiener'}, ...
    {'csd', 'istft', 'lombscargle', 'spectrogram', 'stft', 'vectorstrength'}, ...
    {'argrelextrema'}, ...
    {'ambgfun', 'ca_cfar', 'cfar_alpha', 'pulse_compression', 'pulse_doppler'}, ...
    {'fm_demod'}, ...
    {'kalman_filter'} ...
};

% 构建算子名称列表
allNames = {};
for m = 1:numel(moduleNames)
    allNames = [allNames, moduleOps{m}];
end

dtypeOrder = {'FP16', 'FP32', 'INT16', 'INT32', 'INT8'};
numDtypes = numel(dtypeOrder);

%% 构建 RMSE 矩阵（使用 max_rmse 列，缺失记录记为 NaN）
numOps = numel(allNames);
rmseMatrix = NaN(numOps, numDtypes);
for i = 1:numOps
    for j = 1:numDtypes
        idx = strcmp(T.operator_name, allNames{i}) & strcmp(T.dtype, dtypeOrder{j});
        if any(idx)
            rmseMatrix(i, j) = T.max_rmse(idx);
        end
    end
end

%% 莫兰迪配色（最右档改为浅黄绿色；NaN单独使用浅灰色）
morandiColors = [ ...
    0.15 0.50 0.25; ...   % 0：深绿
    0.25 0.60 0.35; ...   % [0, 1e-9)：中绿
    0.35 0.70 0.45; ...   % [1e-9, 1e-7)：浅绿
    0.50 0.80 0.60; ...   % [1e-7, 1e-5)：浅绿
    0.75 0.92 0.78; ...   % >= 1e-5：最浅绿
    0.85 0.85 0.80; ...   % NaN：浅灰
];

labels = {'0', '[0, 1\times10^{-9})', '[1\times10^{-9}, 1\times10^{-7})', ...
    '[1\times10^{-7}, 1\times10^{-5})', '\geq1\times10^{-5}', 'NaN'};

%% 拆分两组：第一组前6个模块，第二组后6个模块
% 计算每个模块累计算子数，确定切分点
moduleCounts = arrayfun(@(m) numel(moduleOps{m}), 1:numel(moduleOps));
splitIdx = 6;  % 前6个模块作为第一张图
part1Count = sum(moduleCounts(1:splitIdx));   % 25
part2Count = sum(moduleCounts(splitIdx+1:end)); % 28

% 第一张图：算子 1..part1Count
names1 = allNames(1:part1Count);
rmse1 = rmseMatrix(1:part1Count, :);
modules1 = moduleNames(1:splitIdx);
ops1 = moduleOps(1:splitIdx);

% 第二张图：算子 part1Count+1..end
names2 = allNames(part1Count+1:end);
rmse2 = rmseMatrix(part1Count+1:end, :);
modules2 = moduleNames(splitIdx+1:end);
ops2 = moduleOps(splitIdx+1:end);

%% 绘制第一张图
drawHeatmap(names1, rmse1, modules1, ops1, dtypeOrder, ...
    morandiColors, labels, '算子精度RMSE对比（ZQ500 GPU）- 第一部分', 1);

%% 绘制第二张图
drawHeatmap(names2, rmse2, modules2, ops2, dtypeOrder, ...
    morandiColors, labels, '算子精度RMSE对比（ZQ500 GPU）- 第二部分', 2);

end

%% ===================== 子函数：绘制单张热力图 =====================
function drawHeatmap(opNames, rmseMat, modNames, modOps, dtypeOrder, ...
    morandiColors, labels, figTitle, figIndex)

numOps = numel(opNames);
numDtypes = numel(dtypeOrder);

%% 创建图窗（宽度固定14cm，高度按算子数动态调整）
figH = max(18, min(30, numOps * 0.95));  % 厘米
fig = figure('Color', 'w');
set(fig, 'Units', 'centimeters', 'Position', [3 3 14 figH]);

% axes：左侧0.22给算子名，右侧0.20给模块标注，底部0.08给图例空间
ax = axes('Parent', fig, 'Position', [0.22 0.08 0.56 0.86]);
hold(ax, 'on');

%% 绘制热力图
for i = 1:numOps
    for j = 1:numDtypes
        mseVal = rmseMat(i, j);
        if isnan(mseVal)
            colorIdx = 6;
        elseif mseVal == 0
            colorIdx = 1;
        elseif mseVal < 1e-9
            colorIdx = 2;
        elseif mseVal < 1e-7
            colorIdx = 3;
        elseif mseVal < 1e-5
            colorIdx = 4;
        else
            colorIdx = 5;
        end

        x = j;
        y = numOps - i + 1;

        patch(ax, ...
            [x-0.5, x+0.5, x+0.5, x-0.5], ...
            [y-0.5, y-0.5, y+0.5, y+0.5], ...
            morandiColors(colorIdx, :), ...
            'EdgeColor', 'none');

        if isnan(mseVal)
            text(ax, x, y, 'NaN', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 10, 'Interpreter', 'none');
        else
            txt = formatScientific(mseVal);
            % 数字根据单元格宽度自适应缩放：长科学计数法缩小字号
            txtLen = length(txt);
            if txtLen <= 12
                fsz = 11;
            elseif txtLen <= 16
                fsz = 10;
            else
                fsz = 9;
            end
            text(ax, x, y, txt, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', fsz, 'Interpreter', 'tex');
        end
    end
end

%% 模块分隔线
cumCount = 0;
for m = 1:numel(modNames)-1
    cumCount = cumCount + numel(modOps{m});
    yLine = numOps - cumCount + 0.5;
    plot(ax, [0.5, numDtypes+0.5], [yLine, yLine], ...
        'Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 0.5);
end

ax.XLim = [0.5, numDtypes + 0.5];
ax.YLim = [0.5, numOps + 0.5];

ax.XTick = 1:numDtypes;
ax.XTickLabel = dtypeOrder;
ax.YTick = 1:numOps;
ax.YTickLabel = opNames;
ax.XAxisLocation = 'top';
ax.TickLabelInterpreter = 'none';

% 标签字号
set(ax, 'FontSize', 11);
xlabel(ax, '输入数据类型', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax, '算子名称', 'FontSize', 12, 'FontWeight', 'bold');
title(ax, figTitle, 'FontSize', 12, 'FontWeight', 'bold');

%% 模块分组标注
cumCount = 0;
for m = 1:numel(modNames)
    cnt = numel(modOps{m});
    center = numOps - (cumCount + (cnt + 1) / 2) + 1;
    cumCount = cumCount + cnt;

    % 雷达模块用蓝色，其他用绿色
    if strcmp(modNames{m}, 'radartools')
        color = [0.40 0.50 0.65];
    else
        color = [0.40 0.55 0.45];
    end

    text(ax, ax.XLim(2) + 0.15, center, modNames{m}, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 10, 'FontWeight', 'bold', ...
        'Color', color, 'Interpreter', 'none');
end

%% 图例
legendAxes = axes('Parent', fig, 'Position', [0.12 0.015 0.76 0.045], 'Visible', 'off');
hold(legendAxes, 'on');
numLegends = size(morandiColors, 1);
legendPositions = linspace(0.06, 0.94, numLegends);
for i = 1:numLegends
    x0 = legendPositions(i);
    rectangle('Parent', legendAxes, ...
        'Position', [x0-0.07 0.20 0.14 0.55], ...
        'FaceColor', morandiColors(i, :), ...
        'EdgeColor', 'none');
    text(legendAxes, x0, 0.90, labels{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Interpreter', 'tex');
end
legendAxes.XLim = [0 1];
legendAxes.YLim = [0 1];

% 标记图序号，便于区分
set(fig, 'Name', sprintf('Figure %d', figIndex), 'NumberTitle', 'off');

end

%% ===================== 子函数：科学计数法格式化 =====================
function str = formatScientific(num)
    if num == 0
        str = '0';
        return;
    end
    exp = floor(log10(num));
    coeff = num / 10^exp;
    if abs(coeff - round(coeff)) < 1e-6
        coeff = round(coeff);
        str = sprintf('%d\\times10^{%d}', coeff, exp);
    else
        str = sprintf('%.1f\\times10^{%d}', coeff, exp);
    end
end
