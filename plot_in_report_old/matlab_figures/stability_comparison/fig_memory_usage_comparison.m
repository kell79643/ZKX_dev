function fig_memory_usage_comparison()
% 各算子内存显存占用对比分析图
% 基于加速比汇总报告第7章的深度分析

close all; clc;

%% 初始化
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultLineLineWidth', 1.8);
set(groot, 'defaultFigureColor', [1 1 1]);
set(groot, 'defaultAxesColor', [1 1 1]);

% 调色板
palette.blue = [0.35 0.55 0.80];
palette.cyan = [0.25 0.65 0.70];
palette.red = [0.75 0.35 0.40];
palette.orange = [0.80 0.50 0.35];
palette.green = [0.30 0.65 0.45];
palette.purple = [0.55 0.40 0.70];
palette.gray = [0.50 0.50 0.50];

baseDir = fileparts(mfilename('fullpath'));
outDir = fullfile(baseDir, 'output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% ============ 数据定义 ============

moduleNames = {'BSplines', 'Convolution', 'Demod', 'Radartools'};

gpuMemoryKB = [160, 300, 240, 250];

overheadRate = [0, 80, 0, 50];

memoryEfficiency = [100, 55, 100, 67];

operators = {
    {'cubic', 'quadratic', 'gauss_spline'}, ...
    {'correlate(FFT)', 'correlate(Direct)'}, ...
    {'fm_demod_1D', 'fm_demod_2D'}, ...
    {'CA-CFAR', 'OS-CFAR', 'ambgfun', 'pulse\_compress', 'pulse\_doppler'} ...
};

scaleLabels = {'1K', '10K', '100K', '1M'};
bsplinesMemory = [0.016, 0.16, 1.6, 16];
demodMemory = [0.024, 0.24, 2.4, 24];
convolutionMemory = [0.030, 0.30, 3.0, 30];
radartoolsMemory = [0.025, 0.25, 2.5, 25];

%% ============ 图1: 显存静态指标分析 (2×2布局) ============
fig1 = figure('Color', 'w', 'Position', [100 100 900 800]);
tiledlayout(2, 2, 'TileSpacing', 'normal', 'Padding', 'normal');

x = categorical(moduleNames);
x = reordercats(x, moduleNames);

%% 子图1: N=10K时显存占用分组柱状图
nexttile;
b1 = bar(x, gpuMemoryKB, 'FaceColor', 'flat');
b1.CData = [palette.blue; palette.orange; palette.green; palette.purple];
hold on;
for i = 1:numel(gpuMemoryKB)
    text(i, gpuMemoryKB(i) + 8, sprintf('%.0f KB', gpuMemoryKB(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end
ylabel('GPU显存 (KB)', 'FontSize', 12);
title('a) N=10,000时各模块GPU显存占用', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0 400]);
grid on;

%% 子图2: 额外开销率对比
nexttile;
b2 = bar(x, overheadRate, 'FaceColor', 'flat');
colors2 = zeros(4, 3);
for i = 1:4
    if overheadRate(i) == 0
        colors2(i,:) = palette.green;
    elseif overheadRate(i) <= 50
        colors2(i,:) = palette.orange;
    else
        colors2(i,:) = palette.red;
    end
end
b2.CData = colors2;
hold on;
for i = 1:numel(overheadRate)
    text(i, overheadRate(i) + 3, sprintf('%.0f%%', overheadRate(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end
ylabel('额外开销率 (%)', 'FontSize', 12);
title('b) 各模块显存额外开销率', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0 100]);
grid on;
legendItems = [
    patch(NaN, NaN, palette.green),
    patch(NaN, NaN, palette.orange),
    patch(NaN, NaN, palette.red)
];
legend(legendItems, {'0%: 最优', '≤50%: 中等', '>50%: 较高'}, ...
    'Location', 'northeast', 'FontSize', 10);

%% 子图3: 显存有效利用率对比
nexttile;
b3 = bar(x, memoryEfficiency, 'FaceColor', 'flat');
colors3 = zeros(4, 3);
for i = 1:4
    if memoryEfficiency(i) >= 90
        colors3(i,:) = palette.green;
    elseif memoryEfficiency(i) >= 60
        colors3(i,:) = palette.orange;
    else
        colors3(i,:) = palette.red;
    end
end
b3.CData = colors3;
hold on;
for i = 1:numel(memoryEfficiency)
    text(i, memoryEfficiency(i) + 3, sprintf('%.0f%%', memoryEfficiency(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end
ylabel('显存有效利用率 (%)', 'FontSize', 12);
title('c) 各模块显存有效利用率', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0 120]);
grid on;
yline(100, '--', '理论最优', 'Color', palette.green, 'LineWidth', 1.0);
legendItems3 = [
    patch(NaN, NaN, palette.green),
    patch(NaN, NaN, palette.orange),
    patch(NaN, NaN, palette.red)
];
legend(legendItems3, {'≥90%: 优秀', '60-89%: 良好', '<60%: 一般'}, ...
    'Location', 'northeast', 'FontSize', 10);

%% 子图4: 空占位（保持2×2布局平衡）
nexttile;
axis off;

sgtitle('CUDA算子模块显存静态指标分析', 'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(fig1, fullfile(outDir, 'fig_memory_usage_comparison_1.png'), 'Resolution', 300);
saveas(fig1, fullfile(outDir, 'fig_memory_usage_comparison_1.fig'));

%% ============ 图2: 显存扩展性分析 ============
fig2 = figure('Color', 'w', 'Position', [100 100 900 600]);

nScale = numel(scaleLabels);
x4 = 1:nScale;
plot(x4, bsplinesMemory, '-s', 'Color', palette.blue, 'MarkerFaceColor', palette.blue, ...
    'MarkerSize', 10, 'DisplayName', 'BSplines', 'LineWidth', 2);
hold on;
plot(x4, demodMemory, '-o', 'Color', palette.green, 'MarkerFaceColor', palette.green, ...
    'MarkerSize', 10, 'DisplayName', 'Demod', 'LineWidth', 2);
plot(x4, convolutionMemory, '-^', 'Color', palette.orange, 'MarkerFaceColor', palette.orange, ...
    'MarkerSize', 10, 'DisplayName', 'Convolution', 'LineWidth', 2);
plot(x4, radartoolsMemory, '-d', 'Color', palette.purple, 'MarkerFaceColor', palette.purple, ...
    'MarkerSize', 10, 'DisplayName', 'Radartools(CFAR)', 'LineWidth', 2);
set(gca, 'YScale', 'log');
ylim([0.01 100]);
ylabel('GPU显存 (MB, 对数坐标)', 'FontSize', 12);
xlabel('数据规模', 'FontSize', 12);
title('各模块显存扩展性对比（线性~O(N)，Convolution近线性）', 'FontSize', 14, 'FontWeight', 'bold');
xticks(x4);
xticklabels(scaleLabels);
legend('Location', 'northwest', 'FontSize', 11);
grid on;

exportgraphics(fig2, fullfile(outDir, 'fig_memory_usage_comparison_2.png'), 'Resolution', 300);
saveas(fig2, fullfile(outDir, 'fig_memory_usage_comparison_2.fig'));

fprintf('图表已保存至: %s\\\n', outDir);
end