function fig_stability_2_dimensional_pass_rate()
[outDir, palette] = StabilityTestCommon.init();
d = StabilityTestCommon.data();

fig = figure('Color','w','Position',[100 100 1500 750]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
dims = {'功能正确性', '边界场景', '异常处理', '稳定性', '性能'};
cats = categorical(dims, dims);
passData = [d.functionalPass; d.boundaryPass; d.exceptionPass; d.stabilityPass; d.performancePass]';
b = bar(cats, passData);
b(1).FaceColor = palette.blue;
b(2).FaceColor = palette.orange;
b(3).FaceColor = palette.green;
b(4).FaceColor = palette.purple;
legend(d.moduleNames, 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2);
ylim([0 115]); ylabel('通过率 (%)'); grid on;
title('a  各模块分维度通过率对比');
xtickangle(20);

nexttile;
failData = [d.bsplinesFails; d.convolutionFails; d.demodFails; d.radartoolsFails]';
b2 = bar(cats, failData);
b2(1).FaceColor = palette.blue;
b2(2).FaceColor = palette.orange;
b2(3).FaceColor = palette.green;
b2(4).FaceColor = palette.purple;
legend(d.moduleNames, 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 2);
ylabel('失败项数'); grid on;
title('b  各模块分维度失败项数');
xtickangle(20);

StabilityTestCommon.saveFig(fig, outDir, 'fig_stability_2_dimensional_pass_rate');
end
