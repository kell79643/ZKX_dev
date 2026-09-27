function fig3_4_cuda_optimization_before_after()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1200 650]);
baseCuda = [1.00 1.00 1.00 1.00];
optimized = [0.28 0.33 0.74 0.50];
b = bar(categorical({'线程映射','访存优化','kernel融合','分支控制'}), [baseCuda(:), optimized(:)]);
b(1).FaceColor = [0.70 0.70 0.70]; b(2).FaceColor = palette.cyan;
grid on; ylabel('归一化耗时');
legend({'基础CUDA','优化后'},'Location','northoutside','Orientation','horizontal');
title('图3-4-4  CUDA 优化前后对比（TODO替换实测）');
xtickangle(25);
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_4_4_cuda_optimization_before_after');
end
