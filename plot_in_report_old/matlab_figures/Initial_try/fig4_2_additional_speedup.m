function fig4_2_additional_speedup()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1000 650]);
bar(categorical(d.addon.names), d.addon.speedup, 'FaceColor', palette.orange);
grid on; ylabel('加速比：Python / C++');
title('图4-2  附加算子 TOP 加速比');
xtickangle(35);
SupportFiguresCommon.saveFig(fig, outDir, 'fig4_2_additional_speedup');
end
