function fig_task_operator_classification_tree()
% 生成“面向任务的算子分类树”。
[outDir, palette] = TaskOperatorCoverageCommon.init();
spec = TaskOperatorCoverageCommon.operatorSpec();

fig = figure('Color', 'w', 'Position', [80 80 1700 980]);
ax = axes(fig);
axis(ax, [0 1 0 1]);
axis(ax, 'off');
hold(ax, 'on');

title(ax, '面向任务的算子分类树', 'FontSize', 18, 'FontWeight', 'bold');
text(0.5, 0.955, '按赛题任务、处理场景和函数粒度组织，共覆盖 22 个目标算子', ...
    'HorizontalAlignment', 'center', 'FontSize', 11.5, 'Color', palette.gray);

TaskOperatorCoverageCommon.drawBox([0.36 0.865 0.28 0.060], ...
    '项目目标算子', '任务一 5 个 / 任务二 17 个', palette.lightBlue);

task1Pos = [0.06 0.735 0.38 0.070];
task2Pos = [0.56 0.735 0.38 0.070];
TaskOperatorCoverageCommon.drawBox(task1Pos, '任务一：雷达信号处理链路', '脉压、多普勒、CFAR、模糊函数', palette.lightGreen);
TaskOperatorCoverageCommon.drawBox(task2Pos, '任务二：通用信号处理算子集', '波形、滤波、样条、谱分析、小波与状态估计', palette.lightGreen);
TaskOperatorCoverageCommon.drawArrow([0.45 0.865], [0.25 0.805]);
TaskOperatorCoverageCommon.drawArrow([0.55 0.865], [0.75 0.805]);

drawTaskGroups(spec, '任务一', 0.045, 0.680, 0.425, palette);
drawTaskGroups(spec, '任务二', 0.525, 0.680, 0.430, palette);

legendY = 0.040;
rectangle('Position', [0.055 legendY 0.020 0.020], 'FaceColor', palette.lightGreen, 'EdgeColor', [0.35 0.35 0.35]);
text(0.083, legendY + 0.010, '任务层', 'VerticalAlignment', 'middle', 'FontSize', 9.5);
rectangle('Position', [0.160 legendY 0.020 0.020], 'FaceColor', palette.lightOrange, 'EdgeColor', [0.35 0.35 0.35]);
text(0.188, legendY + 0.010, '处理场景 / 算子类别', 'VerticalAlignment', 'middle', 'FontSize', 9.5);
rectangle('Position', [0.330 legendY 0.020 0.020], 'FaceColor', palette.lightGray, 'EdgeColor', [0.35 0.35 0.35]);
text(0.358, legendY + 0.010, '具体函数算子', 'VerticalAlignment', 'middle', 'FontSize', 9.5);

TaskOperatorCoverageCommon.saveFig(fig, outDir, 'fig_task_operator_classification_tree');
end

function drawTaskGroups(spec, taskName, x0, topY, width, palette)
taskRows = spec(strcmp(spec.task, taskName), :);
groups = unique(taskRows.group, 'stable');
nGroups = numel(groups);
if strcmp(taskName, "任务一")
    nCols = 3;
    panelBottom = 0.235;
else
    nCols = 3;
    panelBottom = 0.105;
end
nRows = ceil(nGroups / nCols);
gapX = 0.020;
gapY = 0.030;
groupW = (width - gapX*(nCols-1)) / nCols;
rowH = (topY - panelBottom - gapY*(nRows-1)) / nRows;

for g = 1:nGroups
    row = floor((g-1) / nCols);
    col = mod(g-1, nCols);
    gx = x0 + col*(groupW + gapX);
    rowTop = topY - row*(rowH + gapY);
    members = taskRows(strcmp(taskRows.group, groups(g)), :);
    groupH = min(0.065, rowH * 0.24);
    TaskOperatorCoverageCommon.drawBox([gx rowTop-groupH groupW groupH], ...
        char(groups(g)), sprintf('%d 个算子', height(members)), palette.lightOrange);

    listTop = rowTop - groupH - 0.018;
    rowBottom = rowTop - rowH;
    itemGap = 0.010;
    itemH = min(0.047, (listTop - rowBottom - itemGap*(height(members)-1)) / max(height(members), 1));
    itemH = max(itemH, 0.026);
    for i = 1:height(members)
        y = listTop - i*itemH - (i-1)*itemGap;
        TaskOperatorCoverageCommon.drawArrow([gx + groupW/2, rowTop-groupH], [gx + groupW/2, y + itemH]);
        titleText = char(members.label(i));
        subText = char(members.operator(i));
        TaskOperatorCoverageCommon.drawBox([gx y groupW itemH], titleText, subText, palette.lightGray, [0.55 0.55 0.55]);
    end
end
end
