function fig_stability_5_error_stability()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 1600 750]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
covData = d.errorCodeCoverage;
covDisplay = covData;
covDisplay(isnan(covDisplay)) = 0;
cats = categorical(d.moduleNames, d.moduleNames);
b = bar(cats, covDisplay, 'FaceColor', 'flat');
for i = 1:numel(covData)
    if isnan(covData(i))
        b.CData(i,:) = palette.gray;
    elseif covData(i) >= 80
        b.CData(i,:) = palette.green;
    elseif covData(i) >= 40
        b.CData(i,:) = palette.orange;
    else
        b.CData(i,:) = palette.red;
    end
end
hold on;
for i = 1:numel(covData)
    if isnan(covData(i))
        text(i, 5, '未测试', 'HorizontalAlignment','center','FontWeight','bold','FontSize',10,'Color',palette.gray);
    else
        text(i, covData(i)+3, sprintf('%.1f%%', covData(i)), ...
            'HorizontalAlignment','center','FontWeight','bold','FontSize',11);
    end
end
yline(100, '--', '满分线', 'Color', palette.green, 'LineWidth', 1.0);
ylim([0 115]); ylabel('错误码匹配度 (%)'); grid on;
title('a  标准化错误码匹配度');

nexttile;
stabilityData = [d.stabilityPass; ...
                 100, 0, 100, 100; ...
                 d.exceptionPass];
stabilityCats = categorical({'长时间运行', '资源占用', '异常容错'}, ...
    {'长时间运行', '资源占用', '异常容错'});
b2 = bar(stabilityCats, stabilityData');
b2(1).FaceColor = palette.blue;
b2(2).FaceColor = palette.orange;
b2(3).FaceColor = palette.green;
b2(4).FaceColor = palette.purple;
ylim([0 115]); ylabel('通过率 (%)'); grid on;
legend(d.moduleNames, 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2);
title('b  稳定性指标对比');

nexttile;
axis off;
title('c  失败项汇总', 'FontSize', 13, 'FontWeight', 'bold');

yStart = 0.92;
text(0.05, yStart, 'BSplines (10项失败):', 'FontWeight','bold','FontSize',11,'Color',palette.blue);
bsplinesFails = {'F007: 二次B样条理论值偏差', ...
    'B009/B010: 超大域值边界', ...
    'E004: 负数阶数未拒绝', ...
    'E007-E012: NaN/Inf传播异常'};
for j = 1:numel(bsplinesFails)
    text(0.08, yStart - j*0.07, bsplinesFails{j}, 'FontSize',9);
end
yStart = yStart - numel(bsplinesFails)*0.07 - 0.04;

text(0.05, yStart, 'Convolution (4项失败):', 'FontWeight','bold','FontSize',11,'Color',palette.orange);
convFails = {'correlate1d\_complex\_direct: MSE=7.97', ...
    'exception\_size\_mismatch: 异常未抛出', ...
    'boundary\_large\_size: 大尺寸异常', ...
    'resource\_memory: 显存超出预期'};
for j = 1:numel(convFails)
    text(0.08, yStart - j*0.07, convFails{j}, 'FontSize',9);
end
yStart = yStart - numel(convFails)*0.07 - 0.04;

text(0.05, yStart, 'Demod: 全部通过', 'FontWeight','bold','FontSize',11,'Color',palette.green);
yStart = yStart - 0.07;
text(0.05, yStart, 'Radartools: 全部通过', 'FontWeight','bold','FontSize',11,'Color',palette.purple);

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_5_error_stability');
end
