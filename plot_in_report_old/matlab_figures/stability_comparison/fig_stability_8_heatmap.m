function fig_stability_8_heatmap()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 1200 600]);

dims = {'功能正确性', '边界场景', '异常处理', '稳定性', '性能'};
data = [d.functionalPass; d.boundaryPass; d.exceptionPass; d.stabilityPass; d.performancePass]';

cmap = [0.78 0.30 0.30; 0.82 0.55 0.28; 0.90 0.80 0.40; 0.60 0.75 0.45; 0.36 0.58 0.43];

ax = axes('Position', [0.12 0.18 0.65 0.70], 'Color', [1 1 1], ...
    'XColor', [0 0 0], 'YColor', [0 0 0]);
imagesc(data);
colormap(ax, cmap);
cb = colorbar;
cb.Label.String = '通过率 (%)';
cb.Label.Color = [0 0 0];
cb.Ticks = [0, 25, 50, 75, 100];
cb.TickLabels = {'0%', '25%', '50%', '75%', '100%'};
set(ax, 'YTick', 1:4, 'YTickLabel', d.moduleNames);
set(ax, 'XTick', 1:5, 'XTickLabel', dims);
xtickangle(15);

for mi = 1:4
    for di = 1:5
        val = data(mi, di);
        if val >= 80
            txtColor = [1 1 1];
        else
            txtColor = [0 0 0];
        end
        text(di, mi, sprintf('%.1f%%', val), ...
            'HorizontalAlignment','center','FontSize',12,'FontWeight','bold','Color',txtColor);
    end
end
clim([0 100]);

title('各模块全维度测试通过率热力图', 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0 0 0]);

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_8_heatmap');
end
