function fig_stability_7_radar_chart()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 900 900]);

dims = {'功能正确性', '边界场景', '异常处理', '稳定性', '性能'};
nDims = numel(dims);

data = [d.functionalPass; d.boundaryPass; d.exceptionPass; d.stabilityPass; d.performancePass];

angles = linspace(0, 2*pi, nDims+1);
angles = angles(1:end-1);

moduleColors = {palette.blue, palette.orange, palette.green, palette.purple};
moduleNames = d.moduleNames;

ax = axes('NextPlot','add','Visible','off','DataAspectRatio',[1 1 1], ...
    'XLim',[-1.4 1.4],'YLim',[-1.4 1.4],'Parent',fig, ...
    'Color',[1 1 1],'XColor',[0 0 0],'YColor',[0 0 0]);
axis(ax, 'off');

for r = [0.25, 0.5, 0.75, 1.0]
    xRing = r * cos([angles, angles(1)]);
    yRing = r * sin([angles, angles(1)]);
    plot(xRing, yRing, '-', 'Color', [0.82 0.82 0.82], 'LineWidth', 0.8);
    text(0.02, r+0.03, sprintf('%d%%', round(r*100)), 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
end

for i = 1:nDims
    xLine = [0, cos(angles(i))];
    yLine = [0, sin(angles(i))];
    plot(xLine, yLine, '-', 'Color', [0.82 0.82 0.82], 'LineWidth', 0.8);
    text(1.25*cos(angles(i)), 1.25*sin(angles(i)), dims{i}, ...
        'HorizontalAlignment','center','FontSize',12,'FontWeight','bold','Color',[0 0 0]);
end

patchHandles = [];
for mi = 1:4
    vals = data(:, mi) / 100;
    xPoly = vals' .* cos(angles);
    yPoly = vals' .* sin(angles);
    xPoly = [xPoly, xPoly(1)];
    yPoly = [yPoly, yPoly(1)];
    fillColor = moduleColors{mi};
    ph = fill(xPoly, yPoly, fillColor, 'FaceAlpha', 0.15, 'EdgeColor', fillColor, 'LineWidth', 2);
    patchHandles = [patchHandles, ph];
    for di = 1:nDims
        plot(vals(di)*cos(angles(di)), vals(di)*sin(angles(di)), ...
            'o', 'Color', fillColor, 'MarkerFaceColor', fillColor, 'MarkerSize', 6);
    end
end

lgd = legend(patchHandles, moduleNames, 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2);
lgd.Color = [1 1 1];
lgd.EdgeColor = [0.6 0.6 0.6];
lgd.TextColor = [0 0 0];
title('各模块全维度测试指标雷达图', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0 0 0]);

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_7_radar_chart');
end
