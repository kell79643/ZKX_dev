function fig_stability_4_performance_compare()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 1500 700]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
moduleColors = {palette.blue, palette.orange, palette.green, palette.purple};
cats = categorical(d.perfNames, d.perfNames);
b = bar(cats, d.perfTimeMs, 'FaceColor', 'flat');
for i = 1:numel(d.perfTimeMs)
    mc = moduleColors{d.perfModule(i)};
    b.CData(i,:) = mc;
end
set(gca, 'YScale', 'log');
grid on; ylabel('执行耗时 (ms, 对数坐标)');
title('a  各算子执行耗时对比');
xtickangle(45);
legend(d.moduleNames, 'Location', 'northeast');

nexttile;
demodScales = [d.demodSpeedup1DScale, d.demodSpeedup2DScale];
demodSpeedups = [d.demodSpeedup1D, d.demodSpeedup2D];
demodTypes = [ones(1,4), 2*ones(1,3)];
cats2 = categorical(demodScales, demodScales);
b2 = bar(cats2, demodSpeedups, 'FaceColor', 'flat');
for i = 1:numel(demodSpeedups)
    if demodTypes(i) == 1
        b2.CData(i,:) = palette.blue;
    else
        b2.CData(i,:) = palette.green;
    end
end
hold on;
for i = 1:numel(demodSpeedups)
    text(i, demodSpeedups(i)+3, sprintf('%.1fx', demodSpeedups(i)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
ylim([0 140]); ylabel('GPU加速比'); grid on;
title('b  FM解调算子GPU加速比');
p1 = plot(NaN, NaN, 's', 'Color', palette.blue, 'MarkerFaceColor', palette.blue, 'MarkerSize', 8);
p2 = plot(NaN, NaN, 's', 'Color', palette.green, 'MarkerFaceColor', palette.green, 'MarkerSize', 8);
legend([p1, p2], {'一维数据','二维数据'}, 'Location', 'northwest');

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_4_performance_compare');
end
