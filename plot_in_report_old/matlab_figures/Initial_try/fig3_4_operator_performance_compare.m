function fig3_4_operator_performance_compare()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1350 700]);
b = bar(categorical(d.perf.names), [d.perf.pyMs(:), d.perf.cppMs(:)]);
b(1).FaceColor = palette.red; b(2).FaceColor = palette.blue;
set(gca,'YScale','log'); grid on;
ylabel('平均耗时 (ms, log)');
legend({'Python cuSignal','C++ Device'},'Location','northoutside','Orientation','horizontal');
title('图3-4-1  单算子性能对比');
xtickangle(40);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_4_1_operator_performance_compare');
end
