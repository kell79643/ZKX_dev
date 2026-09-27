function fig3_2_output_overlay_example()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1000 650]);
x = linspace(0,1,512);
py = exp(-((x-0.42)/0.055).^2) + 0.18*exp(-((x-0.72)/0.035).^2);
cpp = py + 0.0004*sin(40*x);
plot(x,py,'Color',palette.red); hold on;
plot(x,cpp,'--','Color',palette.blue);
legend({'Python cuSignal','CUDA C++'},'Location','northeast');
grid on; title('图3-2-4  示例输出叠加对比');
xlabel('归一化采样点'); ylabel('幅值');
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_2_4_output_overlay_example');
end
