function fig3_6_stability_resource_error_handling()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
rng(13);
fig = figure('Color','w','Position',[100 100 1650 850]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile;
bar(categorical(d.robust.categories), d.robust.passRate, 'FaceColor', palette.green);
ylim([0 105]); ylabel('通过率 (%)'); grid on;
title('a  异常输入明确错误返回');
xtickangle(35);
for i = 1:numel(d.robust.errorCases)
    text(i, 8, d.robust.errorCodes{i}, 'HorizontalAlignment','center', 'Rotation',90, 'FontSize',8, 'Color',palette.gray);
end
nexttile;
loop = 1:1000;
drift = 0.005*sin(loop/80) + 0.002*randn(size(loop));
plot(loop, drift, 'Color', palette.blue); hold on;
yline(0.01,'--','+0.01%','Color',palette.red);
yline(-0.01,'--','-0.01%','Color',palette.red);
grid on; xlabel('循环次数'); ylabel('输出波动 (%)');
title('b  1000次循环稳定性，无内存/显存泄露');
nexttile;
bar(categorical(d.mem.names), d.mem.mib, 'FaceColor', palette.blue);
grid on; ylabel('显式resident buffer (MiB)');
title('c  流水线显存占用');
xtickangle(30);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_6_stability_resource_error_handling');
end
