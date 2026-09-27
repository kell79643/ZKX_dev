function fig3_2_accuracy_task1_mse()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1200 650]);
semilogy(d.acc.task1Mse + eps, '-o', 'Color', palette.blue, 'MarkerFaceColor', palette.blue);
set(gca,'XTick',1:numel(d.acc.task1Names),'XTickLabel',d.acc.task1Names);
yline(1e-6,'--','MSE threshold 1e-6','Color',palette.red);
grid on; ylabel('MSE'); title('图3-2-1  任务1必做算子 MSE');
xtickangle(25);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_2_1_accuracy_task1_mse');
end
