function fig3_5_snr_robustness()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1000 650]);
plot(d.snr.db, d.snr.task1Pd, '-o', 'Color', palette.blue, 'MarkerFaceColor', palette.blue); hold on;
plot(d.snr.db, d.snr.task2FeatAcc, '-s', 'Color', palette.green, 'MarkerFaceColor', palette.green);
ylim([0.7 1.02]); grid on; xlabel('SNR (dB)'); ylabel('比例');
legend({'任务1检测率','任务2特征准确率'},'Location','southeast');
title('图3-5-1  不同 SNR 下的鲁棒性（TODO替换实测）');
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_5_1_snr_robustness');
end
