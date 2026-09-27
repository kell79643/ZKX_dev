function fig3_4_cpp_device_baseline()
[outDir, palette] = SupportFiguresCommon.init();
d = SupportFiguresCommon.data();
fig = figure('Color','w','Position',[100 100 1250 650]);
bar(categorical(d.cppBase.names), d.cppBase.ms, 'FaceColor', palette.blue);
set(gca,'YScale','log'); grid on;
ylabel('C++ Device平均耗时 (ms, log)');
title('图3-4-3  重点算子 C++ Device 基线');
xtickangle(35);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_4_3_cpp_device_baseline');
end
