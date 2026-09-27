function fig3_5_bspline_smoothing()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1000 650]);
x = linspace(-3,3,200);
plot(x, exp(-x.^2), 'Color', palette.blue); hold on;
plot(x, exp(-x.^2/2), 'Color', palette.green);
plot(x, exp(-abs(x)), 'Color', palette.orange);
grid on; title('图3-5-4  B-spline/window smoothing effect');
legend({'窄窗','中等','宽窗'},'Location','northeast');
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_5_4_bspline_smoothing');
end
