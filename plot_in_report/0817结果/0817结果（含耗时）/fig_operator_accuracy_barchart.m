function fig_operator_accuracy_barchart()
% =========================================================================
% 绘制53个算子四种精度指标的散点图（横向布局）
% 四个精度指标：MSE, RMSE, relative_l2, relative_linf
% 布局：两张 figure，每张上下双子图
%   上子图：MSE / RMSE（对数X轴）
%   下子图：相对误差 L2 / Linf（对数X轴）
% Y轴：算子名称（竖排），按所属模块分组排列
% 每张图约27个算子，4色散点 + 同指标横向连线
% 数值聚合：每个算子取所有 case × dtype 的中位数
% 数据源：04_算子结果/绘图数据/全算子_逐case绘图候选.csv
%         04_算子结果/绘图数据/53算子_12模块映射.csv
% =========================================================================
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName',  'Microsoft YaHei');

%% ===================== 路径配置 =====================
baseDir = 'c:\Users\asus\Desktop\0817结果（含耗时）\04_算子结果\绘图数据';
csvAllCases = fullfile(baseDir, '全算子_逐case绘图候选.csv');
csvMapping  = fullfile(baseDir, '53算子_12模块映射.csv');

%% ===================== 读取算子 → 模块映射 =====================
fprintf('读取算子-模块映射...\n');
Tmap = readtable(csvMapping, 'ReadVariableNames', true);
opOrder  = Tmap.operator_name;   % cell array of 53
modOrder = Tmap.module;           % cell array of 53
nOp = numel(opOrder);
fprintf('  共 %d 个算子，涵盖 %d 个模块\n', nOp, numel(unique(modOrder)));

%% ===================== 读取逐case数据，计算每算子中位数 =====================
fprintf('读取逐case绘图候选数据...\n');
T = readtable(csvAllCases, 'ReadVariableNames', true);
fprintf('  共 %d 条 case 记录\n', height(T));

metricCols = {'mse', 'rmse', 'relative_l2', 'relative_linf'};
opMedians  = nan(nOp, 4);
opCounts   = zeros(nOp, 1);

for i = 1:nOp
    opName = opOrder{i};
    idx = strcmp(T.operator_name, opName);
    if ~any(idx)
        warning('算子 %s 在逐case表中未找到数据，留为NaN', opName);
        continue;
    end
    opCounts(i) = sum(idx);
    for m = 1:4
        col = metricCols{m};
        vals = T.(col)(idx);
        mask = isfinite(vals) & vals >= 0;
        if any(mask)
            opMedians(i, m) = median(vals(mask));
        end
    end
end
fprintf('  中位数计算完成。\n\n');

%% ===================== 打印核对表 =====================
fprintf('===== 53算子精度中位数（按模块排序）=====\n');
fprintf('No.  module          operator            count    MSE         RMSE        rel_L2      rel_Linf\n');
for i = 1:nOp
    fprintf('%-3d  %-15s %-18s %-5d  %s  %s  %s  %s\n', i, ...
        modOrder{i}, opOrder{i}, opCounts(i), ...
        fmt_med(opMedians(i,1)), fmt_med(opMedians(i,2)), ...
        fmt_med(opMedians(i,3)), fmt_med(opMedians(i,4)));
end

%% ===================== 配色方案（低饱和度、4色区分） =====================
colors = [
    0.22 0.42 0.68;   % 1 MSE      - 深蓝
    0.82 0.48 0.22;   % 2 RMSE     - 橙棕
    0.28 0.58 0.40;   % 3 rel_L2   - 墨绿
    0.55 0.35 0.65];  % 4 rel_Linf - 深紫
markerStyles = {'o', 's', '^', 'd'};
displayNames = {'MSE', 'RMSE', '相对误差 L2', '相对误差 Linf'};
lineWidth    = 1.5;
markerSize   = 7;

%% ===================== 对数坐标 floor 值准备 =====================
[medPlot, zeroMask, nanMask] = prepare_for_log(opMedians);
hollowMask = zeroMask | nanMask;

%% ===================== 分两页：每页约27个算子 =====================
nPerPage = ceil(nOp / 2);   % 27
pages = {1:nPerPage, nPerPage+1:nOp};

for pg = 1:2
    idxPage = pages{pg};
    nPage = numel(idxPage);
    opNamesPage  = opOrder(idxPage);
    modNamesPage = modOrder(idxPage);
    medPage      = medPlot(idxPage, :);
    hollowPage   = hollowMask(idxPage, :);

    %% ===================== 创建图形 =====================
    fig = figure('Color','w');
    set(fig, 'Units', 'centimeters', 'Position', [2 2 24 28]);

    % 上下双子图位置 [left bottom width height]
    posTop = [0.22 0.55 0.73 0.38];
    posBot = [0.22 0.10 0.73 0.38];

    %% -------- 上子图：MSE / RMSE --------
    ax1 = axes('Parent', fig, 'Position', posTop);
    hold(ax1, 'on');
    yVals = 1:nPage;  % Y轴位置（递增），通过 YDir=reverse 实现第一个算子在顶部

    % MSE（指标1）和 RMSE（指标2）
    for m = 1:2
        xData = medPage(:, m);
        yData = yVals;
        % 填充散点（非0/非NA）
        idxFill = ~hollowPage(:, m);
        if any(idxFill)
            scatter(ax1, xData(idxFill), yData(idxFill), markerSize^2, ...
                colors(m,:), markerStyles{m}, 'filled', ...
                'MarkerEdgeColor', 'w', 'HandleVisibility', 'off');
        end
        % 空心散点（0/NA）
        idxHollow = hollowPage(:, m);
        if any(idxHollow)
            scatter(ax1, xData(idxHollow), yData(idxHollow), markerSize^2, ...
                colors(m,:), markerStyles{m}, ...
                'MarkerEdgeColor', colors(m,:), 'LineWidth', 1.5, ...
                'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
        end
    end

    % 创建图例句柄（只取上子图）
    hLegends = gobjects(2,1);
    for m = 1:2
        hLegends(m) = plot(ax1, NaN, NaN, 'Color', colors(m,:), 'LineWidth', lineWidth, ...
            'Marker', markerStyles{m}, 'MarkerSize', markerSize, ...
            'MarkerFaceColor', colors(m,:), 'MarkerEdgeColor', 'w', ...
            'DisplayName', displayNames{m});
    end

    format_scatter_axes(ax1, yVals, opNamesPage, modNamesPage, nPage, 'MSE / RMSE');

    %% -------- 下子图：相对误差 L2 / Linf --------
    ax2 = axes('Parent', fig, 'Position', posBot);
    hold(ax2, 'on');

    for m = 3:4
        xData = medPage(:, m);
        yData = yVals;
        idxFill = ~hollowPage(:, m);
        if any(idxFill)
            scatter(ax2, xData(idxFill), yData(idxFill), markerSize^2, ...
                colors(m,:), markerStyles{m}, 'filled', ...
                'MarkerEdgeColor', 'w', 'HandleVisibility', 'off');
        end
        idxHollow = hollowPage(:, m);
        if any(idxHollow)
            scatter(ax2, xData(idxHollow), yData(idxHollow), markerSize^2, ...
                colors(m,:), markerStyles{m}, ...
                'MarkerEdgeColor', colors(m,:), 'LineWidth', 1.5, ...
                'MarkerFaceColor', 'none', 'HandleVisibility', 'off');
        end
    end

    % 下子图图例句柄
    hLegends2 = gobjects(2,1);
    for m = 3:4
        hLegends2(m-2) = plot(ax2, NaN, NaN, 'Color', colors(m,:), 'LineWidth', lineWidth, ...
            'Marker', markerStyles{m}, 'MarkerSize', markerSize, ...
            'MarkerFaceColor', colors(m,:), 'MarkerEdgeColor', 'w', ...
            'DisplayName', displayNames{m});
    end

    format_scatter_axes(ax2, yVals, opNamesPage, modNamesPage, nPage, '相对误差 L2 / Linf');

    %% -------- 全局图例 --------
    legend([hLegends; hLegends2], displayNames, ...
        'Orientation', 'horizontal', ...
        'Location', 'northoutside', ...
        'NumColumns', 4, ...
        'Box', 'off', ...
        'FontSize', 10, ...
        'Interpreter', 'none');

    %% -------- 脚注 --------
    annotation(fig, 'textbox', [0.05 0.005 0.90 0.04], ...
        'String', {'※ 空心标记 = 中位数为 0（完全匹配）或无有效数据（NA）；X轴为对数刻度；算子按所属模块分组排列'}, ...
        'FontSize', 8, 'EdgeColor', 'none', 'Interpreter', 'none', ...
        'FontName', 'Microsoft YaHei');

    set(fig, 'Color','w');
    drawnow;

    fprintf('图 %d（算子 %d-%d）绘制完成。\n', pg, idxPage(1), idxPage(end));
end

fprintf('\n全部绘图完成。未自动保存，请在图窗中手动保存。\n');
end


%% =========================================================================
% 辅助：格式化散点图坐标轴
% =========================================================================
function format_scatter_axes(ax, yVals, opNames, modNames, nOp, xLabelStr)
    % Y轴：算子名
    ax.YTick = yVals;
    ax.YTickLabel = opNames;
    ax.YDir = 'reverse';          % 第一个算子显示在最上面
    ax.TickLabelInterpreter = 'none';
    ax.FontSize = 9;
    ax.YLim = [0.3, nOp + 0.7];

    % X轴：对数刻度
    ax.XScale = 'log';
    ax.XLabel.String = xLabelStr;
    ax.XLabel.FontSize = 11;
    ax.XLabel.FontWeight = 'bold';

    % 网格
    ax.XGrid = 'on';
    ax.YGrid = 'off';
    ax.GridAlpha = 0.3;
    ax.Box = 'off';
    ax.Layer = 'bottom';

    % 模块分隔线（水平虚线）
    modChangeIdx = find(~strcmp(modNames(1:end-1), modNames(2:end)));
    xl = ax.XLim;
    xBot = xl(1);
    xTop = xl(2);
    for ci = modChangeIdx(:)'
        yLine = yVals(ci) - 0.5;   % 两算子之间的Y位置
        line(ax, [xBot; xTop], [yLine; yLine], ...
            'Color', [0.75 0.75 0.75], 'LineStyle', '--', 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
    end
end


%% =========================================================================
% 辅助：将中位数做 log 坐标预处理（0/NaN → floor 值，保持图形连贯）
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
    plotData(nanMask)  = floorVal;
end


%% =========================================================================
% 辅助：fprintf 用的统一打印格式
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
