function fig_task12_operator_distribution()
% 任务一/任务二核心算子占比分布饼状图
% Task1: 5个核心算子 (雷达目标检测与多普勒分析)
% Task2: 17个核心算子 (多域特征提取)
% 总计: 53个算子

close all; clc;

%% 初始化
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultFigureColor', [1 1 1]);

% 调色板
palette.blue = [0.55 0.70 0.85];
palette.orange = [0.85 0.60 0.55];
palette.green = [0.55 0.75 0.65];
palette.purple = [0.68 0.60 0.78];
palette.red = [0.80 0.58 0.60];
palette.gray = [0.65 0.63 0.60];

baseDir = fileparts(mfilename('fullpath'));
outDir = fullfile(baseDir, 'output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 算子数量定义
task1_core = 5;      % 任务一核心算子
task2_core = 17;     % 任务二核心算子
total_ops = 53;       % 总算子数
other_ops = total_ops - task1_core - task2_core;  % 附加算子

%% 创建数据
data = [task1_core, task2_core, other_ops];
labels = {'任务一核心算子', '任务二核心算子', '附加算子'};
colors = [palette.blue; palette.orange; palette.gray];

%% 创建图表
fig = figure('Color', 'w', 'Position', [100 100 900 600]);

%% 子图1: 饼状图
subplot(1, 2, 1);
pie(data);
colormap(colors);
title('a) 算子数量分布', 'FontWeight', 'bold', 'FontSize', 12);
legend({'任务一核心算子 (5)', '任务二核心算子 (17)', '附加算子 (31)'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');

%% 子图2: 条形对比图
subplot(1, 2, 2);
categories = {'任务一核心', '任务二核心', '附加算子'};
x = categorical(categories);
x = reordercats(x, categories);
b = bar(x, data, 'FaceColor', 'flat');
b.CData = colors;
hold on;
for i = 1:numel(data)
    text(i, data(i) + 1, sprintf('%d', data(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end
ylabel('算子数量');
title('b) 算子数量对比', 'FontWeight', 'bold', 'FontSize', 12);
ylim([0 40]);
grid on;

%% 总标题
sgtitle('CUDA算子模块占比分析 (总计53个算子)', 'FontSize', 14, 'FontWeight', 'bold');

%% 保存图表
exportgraphics(fig, fullfile(outDir, 'fig_task12_operator_distribution.png'), 'Resolution', 300);
fprintf('图表已保存至: %s\\fig_task12_operator_distribution.png\n', outDir);
end
