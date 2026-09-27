function fig_accuracy_step_linechart()
% =========================================================================
% 绘制任务一/任务二各 step 精度指标折线图
% 四个精度指标：MSE, RMSE, relative_l2, relative_linf
% X轴：Pipeline 步骤 (Step)
% 每个step：全部case × 该step全部输出（排除NA/discrete）取中位数
% 布局：上下两个子图，Task1 在上，Task2 在下
% 每个子图：左右双纵坐标（yyaxis）—— 左轴 MSE/RMSE，右轴 rel_l2/rel_linf
% 0值/NA值：替换为floor值，空心标记，线条连贯
% 数据来源：03_任务结果/Task{1,2}/.../accuracy_details_fft_thrust_formal.csv
% =========================================================================
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName',  'Microsoft YaHei');

%% ===================== 路径配置 =====================
baseDir = 'c:\Users\asus\Desktop\0817结果（含耗时）\03_任务结果';
task1Root = fullfile(baseDir, 'Task1', 'formal', 'pipeline', 'fft_thrust', ...
    'RB_20260810T123934Z_STAGE05_TASK1_PIPELINE_FFT_THRUST', 'raw', 'cases');
task2Root = fullfile(baseDir, 'Task2', 'formal', 'pipeline', 'fft_thrust', ...
    'RB_20260815T152000Z_STAGE06_TASK2_PIPELINE_FFT_THRUST', 'raw', 'cases');

%% ===================== 读取并汇总 Task1 / Task2 =====================
metricNames = {'mse', 'rmse', 'relative_l2', 'relative_linf'};
nStepTask1 = 5;
nStepTask2 = 6;

fprintf('开始读取 Task1 精度数据...\n');
[t1Median, t1Count] = collect_task_metrics(task1Root, 'task1', nStepTask1, metricNames);
fprintf('Task1 读取完成。\n\n');

fprintf('开始读取 Task2 精度数据...\n');
[t2Median, t2Count] = collect_task_metrics(task2Root, 'task2', nStepTask2, metricNames);
fprintf('Task2 读取完成。\n\n');

%% ===================== 打印统计信息便于核对 =====================
fprintf('===== Task1 各 step 中位数 =====\n');
fprintf('step  count    MSE         RMSE        rel_l2      rel_linf\n');
for s = 1:nStepTask1
    fprintf('S%-2d  %-5d  %s  %s  %s  %s\n', s, t1Count(s), ...
        fmt_med(t1Median(s,1)), fmt_med(t1Median(s,2)), ...
        fmt_med(t1Median(s,3)), fmt_med(t1Median(s,4)));
end
fprintf('\n===== Task2 各 step 中位数 =====\n');
fprintf('step  count    MSE         RMSE        rel_l2      rel_linf\n');
for s = 1:nStepTask2
    fprintf('S%-2d  %-5d  %s  %s  %s  %s\n', s, t2Count(s), ...
        fmt_med(t2Median(s,1)), fmt_med(t2Median(s,2)), ...
        fmt_med(t2Median(s,3)), fmt_med(t2Median(s,4)));
end

%% ===================== 配色方案（低饱和度、区分度好） =====================
colors = [
    0.22 0.42 0.68;   % 1 MSE  - 深蓝
    0.82 0.48 0.22;   % 2 RMSE - 橙棕
    0.28 0.58 0.40;   % 3 rel_l2   - 墨绿
    0.55 0.35 0.65];  % 4 rel_linf - 深紫
markerStyles = {'o', 's', '^', 'd'};  % 圆、方、三角、菱形
lineWidth    = 1.8;
markerSize   = 7;
displayNames = {'MSE', 'RMSE', '相对误差 L2', '相对误差 Linf'};

%% ===================== 创建图形 =====================
fig = figure('Color','w');
set(fig, 'Units', 'centimeters', 'Position', [2 2 21 25]);

% 子图位置 [left bottom width height]
% 宽度收窄至0.75，右侧留出空间给右轴标签
pos1 = [0.08 0.56 0.75 0.36];   % 上方 Task1
pos2 = [0.08 0.14 0.75 0.36];   % 下方 Task2

%% -------- 上方子图：Task1 --------
ax1 = axes('Parent', fig, 'Position', pos1);
hold(ax1, 'on');

x1 = 1:nStepTask1;
[t1Plot, t1ZeroMask, t1NaNMask] = prepare_for_log(t1Median);
t1Hollow = t1ZeroMask | t1NaNMask;   % 0 和 NA 都用空心标记

% 左轴：MSE, RMSE
yyaxis(ax1, 'left')
h_t1_mse  = plot_metric(ax1, x1, t1Plot(:,1), t1Hollow(:,1), ...
    colors(1,:), markerStyles{1}, lineWidth, markerSize, displayNames{1});
h_t1_rmse = plot_metric(ax1, x1, t1Plot(:,2), t1Hollow(:,2), ...
    colors(2,:), markerStyles{2}, lineWidth, markerSize, displayNames{2});

% 右轴：rel_l2, rel_linf
yyaxis(ax1, 'right')
h_t1_rl2  = plot_metric(ax1, x1, t1Plot(:,3), t1Hollow(:,3), ...
    colors(3,:), markerStyles{3}, lineWidth, markerSize, displayNames{3});
h_t1_rli  = plot_metric(ax1, x1, t1Plot(:,4), t1Hollow(:,4), ...
    colors(4,:), markerStyles{4}, lineWidth, markerSize, displayNames{4});

format_yyaxis(ax1, x1, nStepTask1, '任务一', t1Plot);

%% -------- 下方子图：Task2 --------
ax2 = axes('Parent', fig, 'Position', pos2);
hold(ax2, 'on');

x2 = 1:nStepTask2;
[t2Plot, t2ZeroMask, t2NaNMask] = prepare_for_log(t2Median);
t2Hollow = t2ZeroMask | t2NaNMask;

% 左轴：MSE, RMSE
yyaxis(ax2, 'left')
h_t2_mse  = plot_metric(ax2, x2, t2Plot(:,1), t2Hollow(:,1), ...
    colors(1,:), markerStyles{1}, lineWidth, markerSize, displayNames{1});
h_t2_rmse = plot_metric(ax2, x2, t2Plot(:,2), t2Hollow(:,2), ...
    colors(2,:), markerStyles{2}, lineWidth, markerSize, displayNames{2});

% 右轴：rel_l2, rel_linf
yyaxis(ax2, 'right')
h_t2_rl2  = plot_metric(ax2, x2, t2Plot(:,3), t2Hollow(:,3), ...
    colors(3,:), markerStyles{3}, lineWidth, markerSize, displayNames{3});
h_t2_rli  = plot_metric(ax2, x2, t2Plot(:,4), t2Hollow(:,4), ...
    colors(4,:), markerStyles{4}, lineWidth, markerSize, displayNames{4});

format_yyaxis(ax2, x2, nStepTask2, '任务二', t2Plot);

%% -------- 全局图例 --------
hLegends = [h_t1_mse, h_t1_rmse, h_t1_rl2, h_t1_rli];
legend(hLegends, displayNames, ...
    'Orientation','horizontal', ...
    'Location','northoutside', ...
    'NumColumns',4, ...
    'Box','off', ...
    'FontSize',10, ...
    'Interpreter','none');

%% -------- 脚注：说明空心标记含义 --------
annotation(fig, 'textbox', [0.10 0.015 0.85 0.05], ...
    'String', {'※ 空心标记 = 误差为0（完全匹配）', ...
               '   Task2 Step5: 离散输出，exact_match=true → 误差为0   Step6: 中位数为0（完全匹配）'}, ...
    'FontSize', 8, 'EdgeColor', 'none', 'Interpreter', 'none', ...
    'FontName', 'Microsoft YaHei');

%% -------- 保存 --------
set(fig, 'Color','w');
drawnow;

scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir, '03_任务结果');
if ~isfolder(outDir)
    outDir = scriptDir;
end
fprintf('\n输出目录：%s\n', outDir);

figName = 'accuracy_step_linechart';
save_figure(fig, outDir, figName);
end


%% =========================================================================
% 辅助：绘制一条折线（含填充/空心标记分层）
%   hollowMask 为 true 的点用空心标记，其余用填充标记
%   返回主线句柄（用于 legend）
% =========================================================================
function h = plot_metric(ax, x, y, hollowMask, color, markerStyle, lw, ms, displayName)
    % 主线（无标记，连贯连接所有点）
    h = plot(ax, x, y, 'Color', color, 'LineStyle', '-', 'LineWidth', lw, ...
        'Marker', 'none', 'DisplayName', displayName);

    % 填充标记：非零/非NA的点
    idxFill = ~hollowMask;
    if any(idxFill)
        plot(ax, x(idxFill), y(idxFill), 'Color', color, 'LineStyle', 'none', ...
            'Marker', markerStyle, 'MarkerSize', ms, ...
            'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', ...
            'HandleVisibility', 'off');
    end

    % 空心标记：零值/NA的点
    idxHollow = hollowMask;
    if any(idxHollow)
        plot(ax, x(idxHollow), y(idxHollow), 'Color', color, 'LineStyle', 'none', ...
            'Marker', markerStyle, 'MarkerSize', ms, ...
            'MarkerFaceColor', 'none', 'MarkerEdgeColor', color, ...
            'LineWidth', 1.5, ...
            'HandleVisibility', 'off');
    end
end


%% =========================================================================
% 辅助：格式化 yyaxis 子图（对数刻度、标签、标题）
% =========================================================================
function format_yyaxis(ax, xVals, nStep, subplotTitle, plotData)
    set(ax, 'XLim', [0.5, nStep + 0.5]);
    set(ax, 'XTick', xVals);
    xtickLabels = cell(1, nStep);
    for s = 1:nStep
        xtickLabels{s} = sprintf('Step %d', s);
    end
    ax.XTickLabel = xtickLabels;
    ax.TickLabelInterpreter = 'none';
    ax.FontSize = 10;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    ax.Box = 'off';

    % 左轴：对数刻度 + 范围
    yyaxis(ax, 'left');
    ax.YScale = 'log';
    allLeft = plotData(:, [1 2]);
    allLeft = allLeft(isfinite(allLeft) & allLeft > 0);
    if ~isempty(allLeft)
        ax.YLim = [min(allLeft) * 0.3, max(allLeft) * 5];
    else
        ax.YLim = [1e-18, 1e-10];
    end
    ax.YLabel.String = 'MSE / RMSE';
    ax.YLabel.FontSize = 11;
    ax.YLabel.FontWeight = 'bold';
    ax.YLabel.Rotation = 90;

    % 右轴：对数刻度 + 范围
    yyaxis(ax, 'right');
    ax.YScale = 'log';
    allRight = plotData(:, [3 4]);
    allRight = allRight(isfinite(allRight) & allRight > 0);
    if ~isempty(allRight)
        ax.YLim = [min(allRight) * 0.3, max(allRight) * 5];
    else
        ax.YLim = [1e-18, 1e-10];
    end
    ax.YLabel.String = '相对误差 L2 / Linf';
    ax.YLabel.FontSize = 11;
    ax.YLabel.FontWeight = 'bold';
    ax.YLabel.Color = [0.15 0.15 0.15];
    ax.YLabel.Rotation = 90;

    % 标题
    title(ax, subplotTitle, 'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', [0.15 0.15 0.15]);
end


%% =========================================================================
% 辅助：将真实 median 做 log 坐标预处理
%   0 和 NaN 都替换为 floor 值（最小正值 ×0.01），使线条连贯
%   返回：plotData(处理后的值)、zeroMask(原值为0)、nanMask(原值为NaN)
% =========================================================================
function [plotData, zeroMask, nanMask] = prepare_for_log(medians)
    nanMask  = isnan(medians);
    zeroMask = (medians == 0) & ~nanMask;
    refs = medians(~nanMask & ~zeroMask);
    if isempty(refs)
        floorVal = 1e-18;
    else
        floorVal = min(refs(:)) * 0.01;
    end
    plotData = medians;
    plotData(zeroMask) = floorVal;
    plotData(nanMask)  = floorVal;   % NA 也替换为 floor，保证连线
end


%% =========================================================================
% 辅助：保存图形为多种格式
% =========================================================================
function save_figure(fig, outDir, figName)
    try
        figFile = fullfile(outDir, [figName, '.fig']);
        saveas(fig, figFile);
        fprintf('已保存 MATLAB FIG: %s\n', figFile);
    catch ME
        fprintf('保存 FIG 失败：%s\n', ME.message);
    end
    try
        pngFile = fullfile(outDir, [figName, '.png']);
        exportgraphics(fig, pngFile, 'Resolution', 300);
        fprintf('已保存 PNG (300dpi): %s\n', pngFile);
    catch ME
        fprintf('exportgraphics 不可用，回退至 print：%s\n', ME.message);
        try
            pngFile = fullfile(outDir, [figName, '.png']);
            print(fig, pngFile, '-dpng', '-r300');
            fprintf('已保存 PNG: %s\n', pngFile);
        catch ME2
            fprintf('保存 PNG 失败：%s\n', ME2.message);
        end
    end
    try
        epsFile = fullfile(outDir, [figName, '.eps']);
        exportgraphics(fig, epsFile, 'ContentType', 'vector');
        fprintf('已保存 EPS (矢量): %s\n', epsFile);
    catch ME
        fprintf('保存 EPS 失败：%s\n', ME.message);
    end
end


%% =========================================================================
% 子函数：收集单个任务的所有case、所有step的指标中位数
% =========================================================================
function [stepMedians, stepCounts] = collect_task_metrics(casesRoot, taskPrefix, nStep, metricNames)
stepBuckets = cell(nStep, numel(metricNames));
for s = 1:nStep
    for m = 1:numel(metricNames)
        stepBuckets{s,m} = [];
    end
end

caseDirInfo = dir(fullfile(casesRoot, [taskPrefix, '_s*']));
caseDirs    = {caseDirInfo([caseDirInfo.isdir]).name};
fprintf('  找到 %d 个 case 目录\n', numel(caseDirs));

nUsedCases = 0;
for i = 1:numel(caseDirs)
    caseName   = caseDirs{i};
    csvFile    = fullfile(casesRoot, caseName, ...
        sprintf('%s_accuracy_details_fft_thrust_formal.csv', taskPrefix));
    if ~isfile(csvFile)
        continue;
    end
    nUsedCases = nUsedCases + 1;

    try
        T = readtable(csvFile, 'ReadVariableNames', true);
    catch ME
        warning('读取失败：%s\n  原因：%s', csvFile, ME.message);
        continue;
    end

    neededCols = {'target', 'output_kind', metricNames{:}};
    if ~all(ismember(neededCols, T.Properties.VariableNames))
        warning('列不全，跳过：%s', csvFile);
        continue;
    end
    hasExactMatch = ismember('exact_match', T.Properties.VariableNames);

    targetStrs = T.target;
    stepIds   = nan(height(T), 1);
    tok = regexp(targetStrs, '\.step(\d+)\.', 'tokens', 'once');
    for r = 1:height(T)
        if ~isempty(tok{r})
            stepIds(r) = str2double(tok{r}{1});
        end
    end

    for r = 1:height(T)
        sid = stepIds(r);
        if isnan(sid) || sid < 1 || sid > nStep
            continue;
        end
        kindChar = char(T.output_kind(r));
        if strcmpi(kindChar, 'discrete')
            % 离散输出：检查 exact_match，完全匹配则四项误差为0
            if hasExactMatch
                emVal = T.exact_match(r);
                emStr = char(emVal);
                if strcmpi(emStr, 'true') || ...
                   (isnumeric(emVal) && emVal == 1) || ...
                   (islogical(emVal) && emVal)
                    for m = 1:numel(metricNames)
                        stepBuckets{sid, m} = [stepBuckets{sid, m}; 0];
                    end
                end
            end
            continue;
        end
        for m = 1:numel(metricNames)
            valRaw = T.(metricNames{m})(r);
            if isnumeric(valRaw) && ~isnan(valRaw) && isfinite(valRaw) && valRaw >= 0
                stepBuckets{sid, m} = [stepBuckets{sid, m}; valRaw];
            end
        end
    end
end
fprintf('  实际读取 %d 个有 accuracy 文件的 case\n', nUsedCases);

stepMedians = nan(nStep, numel(metricNames));
stepCounts  = zeros(nStep, 1);
for s = 1:nStep
    stepCounts(s) = numel(stepBuckets{s, 1});
    for m = 1:numel(metricNames)
        bucket = stepBuckets{s, m};
        if ~isempty(bucket)
            stepMedians(s, m) = median(bucket);
        end
    end
end
end


%% =========================================================================
% 辅助：fprintf 用的统一打印格式（处理 NA / 0）
% =========================================================================
function s = fmt_med(v)
    if isnan(v)
        s = 'NA            ';
    elseif v == 0
        s = '0.0000e+00    ';
    else
        s = sprintf('%.4e    ', v);
    end
end
