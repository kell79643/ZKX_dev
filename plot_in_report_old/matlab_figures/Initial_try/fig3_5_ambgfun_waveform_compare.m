function fig3_5_ambgfun_waveform_compare()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1300 650]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
bar(categorical(d.ambg.waveforms), d.ambg.timeMs, 'FaceColor', palette.blue);
grid on; ylabel('ambgfun耗时 (ms)'); title('a  不同波形模糊函数耗时');
nexttile;
b = bar(categorical(d.ambg.waveforms), [d.ambg.rangeRes(:), d.ambg.dopplerRes(:)]);
b(1).FaceColor = palette.blue; b(2).FaceColor = palette.orange;
grid on; ylabel('分辨率指标（越低越好）');
legend({'Range','Doppler'},'Location','northoutside','Orientation','horizontal');
title('b  距离/多普勒分辨率对比');
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_5_2_ambgfun_waveform_compare');
end
