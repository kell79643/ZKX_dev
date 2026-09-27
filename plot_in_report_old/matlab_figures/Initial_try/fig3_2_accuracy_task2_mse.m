function fig3_2_accuracy_task2_mse()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1400 650]);
semilogy(d.acc.task2Mse + eps, '-o', 'Color', palette.green, 'MarkerFaceColor', palette.green);
set(gca,'XTick',1:numel(d.acc.task2Names),'XTickLabel',d.acc.task2Names);
yline(1e-6,'--','MSE threshold 1e-6','Color',palette.red);
grid on; ylabel('MSE'); title('图3-2-2  任务2推荐算子 MSE');
xtickangle(45);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_2_2_accuracy_task2_mse');
end
