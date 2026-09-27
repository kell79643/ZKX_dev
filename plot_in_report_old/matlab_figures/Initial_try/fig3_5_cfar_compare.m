function fig3_5_cfar_compare()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1300 650]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
b = bar(categorical(d.cfar.scene), [d.cfar.caPd(:), d.cfar.osPd(:)]);
b(1).FaceColor = palette.blue; b(2).FaceColor = palette.cyan;
ylim([0 1.05]); grid on; ylabel('检测概率');
legend({'CA-CFAR','OS-CFAR'},'Location','southoutside','Orientation','horizontal');
title('a  CFAR检测概率');
nexttile;
b = bar(categorical(d.cfar.scene), [d.cfar.caPfa(:), d.cfar.osPfa(:)]);
b(1).FaceColor = palette.red; b(2).FaceColor = palette.purple;
ylim([0 0.5]); grid on; ylabel('虚警率');
legend({'CA-CFAR','OS-CFAR'},'Location','northoutside','Orientation','horizontal');
title('b  CFAR虚警率');
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_5_3_cfar_compare');
end
