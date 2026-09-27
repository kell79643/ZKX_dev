function fig3_4_speedup_compare()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1250 650]);
bar(categorical(d.perf.names), d.perf.speedup, 'FaceColor', palette.green);
grid on; ylabel('加速比：Python / C++');
title('图3-4-2  C++ Device 加速比');
xtickangle(40);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_4_2_speedup_compare');
end
