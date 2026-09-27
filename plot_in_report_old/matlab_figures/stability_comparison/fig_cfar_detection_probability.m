function fig_cfar_detection_probability()
[outDir, palette] = CFARComparisonCommon.init();
d = CFARComparisonCommon.data();

fig = figure('Color','w','Position',[100 100 1200 700]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
x = [1, 2];
pdData = d.detectionProbability * 100;
caPd = pdData(:, 1);
osPd = pdData(:, 2);
barWidth = 0.35;
b1 = bar(x - barWidth/2, pdData(:, 1), barWidth, 'FaceColor', palette.blue, 'DisplayName', 'CA-CFAR');
hold on;
b2 = bar(x + barWidth/2, pdData(:, 2), barWidth, 'FaceColor', palette.orange, 'DisplayName', 'OS-CFAR');
caValues = [100.00, 99.70];
osValues = [100.00, 100.00];
for i = 1:2
    text(x(i) - barWidth/2, caValues(i) + 0.5, sprintf('%.2f%%', caValues(i)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold','Color',palette.blue);
    text(x(i) + barWidth/2, osValues(i) + 0.5, sprintf('%.2f%%', osValues(i)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold','Color',palette.orange);
end
ylim([95 102]);
ylabel('检测概率 (%)');
set(gca, 'XTick', x, 'XTickLabel', d.scenarioNames);
title('a  各场景检测概率对比');
legend('Location', 'southoutside', 'Orientation', 'horizontal');
grid on;
ax = gca;
ax.YAxis.MinorTick = 'on';
ax.YMinorGrid = 'on';

nexttile;
pdDiff = (d.detectionProbability(:, 2) - d.detectionProbability(:, 1)) * 100;
colors = arrayfun(@(x) ternary(x >= 0, palette.green, palette.red), pdDiff, 'UniformOutput', false);
b3 = bar(x, pdDiff, 0.5, 'FaceColor', [0.36 0.58 0.43], 'DisplayName', '差异 (OS-CFAR - CA-CFAR)');
hold on;
for i = 1:2
    val = pdDiff(i);
    text(i, val + 0.1, sprintf('%+0.2f%%', val), ...
        'HorizontalAlignment','center','FontSize',11,'FontWeight','bold','Color', palette.green);
end
ylim([-0.5 1]);
ylabel('检测概率差异 (%)');
set(gca, 'XTick', x, 'XTickLabel', d.scenarioNames);
title('b  OS-CFAR相对CA-CFAR检测概率提升');
grid on;
plot([0.5 2.5], [0 0], 'k--', 'LineWidth', 0.8);

sgtitle('CA-CFAR与OS-CFAR检测概率性能对比', 'FontSize', 16, 'FontWeight', 'bold');

CFARComparisonCommon.saveFig(fig, outDir, 'fig_cfar_detection_probability');
end