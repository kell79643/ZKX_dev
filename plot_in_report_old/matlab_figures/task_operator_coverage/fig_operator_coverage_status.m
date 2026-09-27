function fig_operator_coverage_status()
% 生成“项目算子覆盖状态”总览图。
[outDir, palette] = TaskOperatorCoverageCommon.init();
C = TaskOperatorCoverageCommon.coverageTable();

fig = figure('Color', 'w', 'Position', [80 80 1780 1080]);
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, '项目算子覆盖状态与一致性验证', 'FontSize', 16, 'FontWeight', 'bold');
annotation(fig, 'textbox', [0.18 0.943 0.64 0.030], ...
    'String', '覆盖指标：Python 参考、CPU 输出、GPU 输出、CPU-Python 通过、GPU-CPU 通过；通过判据为 MSE <= 阈值', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', 'FontSize', 10.5, 'Color', palette.gray);

ax1 = nexttile(tl, [2 2]);
status = double([C.pythonRef, C.cpuOutput, C.gpuOutput, C.cpuPyPass, C.gpuCpuPass]);
imagesc(ax1, status);
colormap(ax1, [0.94 0.94 0.94; palette.green]);
caxis(ax1, [0 1]);
set(ax1, 'XTick', 1:5, 'XTickLabel', {'Python参考','CPU输出','GPU输出','CPU-Python','GPU-CPU'}, ...
    'YTick', 1:height(C), 'YTickLabel', operatorLabels(C), ...
    'XAxisLocation', 'top', 'TickLength', [0 0], 'FontSize', 9.5);
xtickangle(ax1, 28);
grid(ax1, 'on');
set(ax1, 'GridColor', [1 1 1], 'GridAlpha', 1, 'LineWidth', 1.0);
xlabel(ax1, '覆盖与验证链路');
ylabel(ax1, '目标算子');
title(ax1, 'a  全量算子覆盖矩阵', 'FontSize', 13, 'FontWeight', 'bold');

for r = 1:size(status,1)
    for c = 1:size(status,2)
        if status(r,c) == 1
            txt = '通过';
            color = [1 1 1];
        else
            txt = '缺失';
            color = palette.gray;
        end
        if c <= 3 && status(r,c) == 1
            txt = '有';
        end
        text(ax1, c, r, txt, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 8.5, 'FontWeight', 'bold', ...
            'Color', color);
    end
end

taskBoundary = find(strcmp(C.task, "任务一"), 1, 'last') + 0.5;
hold(ax1, 'on');
plot(ax1, [0.5 5.5], [taskBoundary taskBoundary], '-', 'Color', [0.20 0.20 0.20], 'LineWidth', 1.3);
text(ax1, 5.55, 3, '任务一', 'FontSize', 9.5, 'Color', palette.gray, 'HorizontalAlignment', 'left');
text(ax1, 5.55, taskBoundary + 8.5, '任务二', 'FontSize', 9.5, 'Color', palette.gray, 'HorizontalAlignment', 'left');

ax2 = nexttile(tl);
comparisons = {'CPU-Python','GPU-CPU','综合覆盖'};
vals = [mean(C.cpuPyPass), mean(C.gpuCpuPass), mean(C.fullyCovered)] * 100;
b = bar(ax2, vals, 0.62, 'FaceColor', 'flat');
b.CData = [palette.blue; palette.green; palette.purple];
set(ax2, 'XTick', 1:3, 'XTickLabel', comparisons, 'FontSize', 10);
ylim(ax2, [0 105]);
ylabel(ax2, '比例 (%)');
title(ax2, 'b  验证通过率', 'FontSize', 13, 'FontWeight', 'bold');
grid(ax2, 'on');
for i = 1:numel(vals)
    text(ax2, i, vals(i) + 3, sprintf('%.1f%%', vals(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', [0.18 0.18 0.18]);
end

ax3 = nexttile(tl);
mse = C.maxMse;
valid = ~isnan(mse);
barh(ax3, find(valid), log10(mse(valid) + eps), 0.72, 'FaceColor', palette.cyan, 'EdgeColor', 'none');
set(ax3, 'YTick', 1:height(C), 'YTickLabel', operatorLabels(C), 'YDir', 'reverse', 'FontSize', 8.8);
xlabel(ax3, 'log_{10}(最大 MSE + eps)');
title(ax3, 'c  最大误差量级', 'FontSize', 13, 'FontWeight', 'bold');
grid(ax3, 'on');
thresholdLine = log10(1e-6);
xline(ax3, thresholdLine, '--', '阈值 1e-6', 'Color', palette.red, ...
    'LineWidth', 1.2, 'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');

summaryPath = fullfile(outDir, 'operator_coverage_summary.csv');
writetable(C, summaryPath);
TaskOperatorCoverageCommon.saveFig(fig, outDir, 'fig_operator_coverage_status');
end

function labels = operatorLabels(C)
labels = strings(height(C), 1);
for i = 1:height(C)
    labels(i) = C.label(i) + " / " + C.operator(i);
end
labels = cellstr(labels);
end
