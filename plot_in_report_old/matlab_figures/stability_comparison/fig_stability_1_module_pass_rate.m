function fig_stability_1_module_pass_rate()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 1400 700]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
cats = categorical(d.moduleNames, d.moduleNames);
b = bar(cats, d.modulePassRate, 'FaceColor', palette.blue);
hold on;
for i = 1:numel(d.modulePassRate)
    text(i, d.modulePassRate(i)+2.5, ...
        sprintf('%.1f%%', d.modulePassRate(i)), ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',12);
end
yline(100, '--', '满分线', 'Color', palette.green, 'LineWidth', 1.2);
ylim([0 115]); ylabel('通过率 (%)'); grid on;
title('a  各模块总体通过率');

nexttile;
b2 = bar(cats, [d.modulePassTests(:), d.moduleFailTests(:)]);
b2(1).FaceColor = palette.green;
b2(2).FaceColor = palette.red;
hold on;
for i = 1:numel(d.modulePassTests)
    text(i, d.modulePassTests(i)+1.5, num2str(d.modulePassTests(i)), ...
        'HorizontalAlignment','center','FontSize',10,'Color',palette.green);
    if d.moduleFailTests(i) > 0
        text(i, d.modulePassTests(i)+d.moduleFailTests(i)+1.5, num2str(d.moduleFailTests(i)), ...
            'HorizontalAlignment','center','FontSize',10,'Color',palette.red);
    end
end
ylabel('测试项数'); grid on;
legend({'通过','失败'},'Location','northoutside','Orientation','horizontal');
title('b  各模块通过/失败项数');

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_1_module_pass_rate');
end
