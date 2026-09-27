function fig_stability_3_mse_comparison()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 1600 750]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
validIdx = d.operatorMSE > 0 & d.operatorMSE < 1;
validNames = d.operatorNames(validIdx);
validMSE = d.operatorMSE(validIdx);
validModule = d.operatorModule(validIdx);

moduleColors = {palette.blue, palette.orange, palette.green, palette.purple};
moduleNames = {'BSplines', 'Convolution', 'Demod', 'Radartools'};

semilogy(validMSE + eps, '-o', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
set(gca,'XTick',1:numel(validNames),'XTickLabel',validNames);
hold on;
yline(1e-6, '--', 'MSE阈值 1e-6', 'Color', palette.red, 'LineWidth', 1.3);
for i = 1:numel(validMSE)
    mc = moduleColors{validModule(i)};
    plot(i, validMSE(i)+eps, 'o', 'Color', mc, 'MarkerFaceColor', mc, 'MarkerSize', 10);
end
validIdxList = find(validIdx);
for i = 1:numel(validMSE)
    mc = moduleColors{validModule(i)};
    text(i, validMSE(i)*3+eps, d.operatorMSELabel{validIdxList(i)}, ...
        'HorizontalAlignment','center','FontSize',7.5,'Color',mc,'Rotation',55);
end
grid on; ylabel('MSE (对数坐标)');
title('a  各算子MSE数值一致性');
xtickangle(45);

hLegends = gobjects(1,4);
for mi = 1:4
    hLegends(mi) = plot(NaN, NaN, 'o', 'Color', moduleColors{mi}, ...
        'MarkerFaceColor', moduleColors{mi}, 'MarkerSize', 8);
end
legend([hLegends, plot(NaN,NaN,'--','Color',palette.red)], ...
    [moduleNames, 'MSE阈值'], 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 3);

nexttile;
allPass = d.operatorPassRate;
barColors = zeros(numel(allPass), 3);
for i = 1:numel(allPass)
    if allPass(i) >= 100
        barColors(i,:) = palette.green;
    else
        barColors(i,:) = palette.red;
    end
end
cats2 = categorical(d.operatorNames, d.operatorNames);
b = bar(cats2, allPass, 'FaceColor', 'flat');
b.CData = barColors;
hold on;
yline(100, '--', '满分线', 'Color', palette.green, 'LineWidth', 1.0);
ylim([0 115]); ylabel('通过率 (%)'); grid on;
title('b  各算子测试通过率');
xtickangle(55);

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_3_mse_comparison');
end
