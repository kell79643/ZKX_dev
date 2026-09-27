function fig5_1_summary_dashboard()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1500 850]);
axis off;
title('图5-1  项目成果总结看板', 'FontSize',18,'FontWeight','bold');
SupportFiguresCommon.drawBox([0.07 0.62 0.24 0.18], '正确性', {'C++回归测试: 210/210通过','Python reference对比: 235/235通过'}, palette.lightGreen);
SupportFiguresCommon.drawBox([0.38 0.62 0.24 0.18], '覆盖度', {'53个GPU Device API已调用并计时','任务1/任务2核心算子纳入一致性验证','Host API保留, Device API扩展'}, palette.lightBlue);
SupportFiguresCommon.drawBox([0.69 0.62 0.24 0.18], '性能', {'35个可比项中31项更快','显存常驻数据链路','复用中间显存与FFT plan'}, palette.lightOrange);
SupportFiguresCommon.drawBox([0.22 0.32 0.24 0.18], '可移植性', {'无CUDA: CPU兼容路径','有GPU: *\_device高性能路径','统一调用入口, 支持后端切换'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.54 0.32 0.24 0.18], '后续工作', {'补齐53个Device API逐元素正确性测试','修复18个Python benchmark不可比项','优化correlate2d与sosfilt慢项'}, palette.lightGray);
SupportFiguresCommon.saveFig(fig, outDir, 'fig5_1_summary_dashboard');
end
