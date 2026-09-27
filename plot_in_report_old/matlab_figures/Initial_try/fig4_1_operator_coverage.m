function fig4_1_operator_coverage()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1300 650]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
pie([5 14 17], {'任务1必做','任务2推荐','附加算子'});
title('a  算子类型覆盖');
nexttile;
cats = categorical({'Host/Python PASS','Device可调用','Device更快'});
vals = [d.summary.hostPythonPass/d.summary.hostPythonTotal, d.summary.deviceCallable/d.summary.deviceTotal, d.summary.deviceFaster/d.summary.deviceCompared] * 100;
bar(cats, vals, 'FaceColor', palette.green);
ylim([0 105]); ylabel('比例 (%)'); grid on;
title('b  工程覆盖状态');
SupportFiguresCommon.saveFig(fig, outDir, 'fig4_1_operator_coverage');
end
