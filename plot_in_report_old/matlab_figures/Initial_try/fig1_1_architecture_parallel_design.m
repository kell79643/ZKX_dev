function fig1_1_architecture_parallel_design()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1500 900]);
axis off;
title('图1-1  代码库总体架构与 CUDA 并行执行设计', 'FontSize',18,'FontWeight','bold');
SupportFiguresCommon.drawBox([0.06 0.70 0.25 0.15], '算法应用层', {'任务1：雷达目标检测 + Doppler','任务2：多域特征提取 + Kalman跟踪'}, palette.lightBlue);
SupportFiguresCommon.drawBox([0.38 0.70 0.25 0.15], '算子核心层', {'waveforms / filtering / FFT','radartools / spectral / wavelets','CPU API 与 CUDA Device API'}, palette.lightGreen);
SupportFiguresCommon.drawBox([0.70 0.70 0.25 0.15], '硬件适配层', {'DeviceArray / DeviceComplexArray','错误处理 / kernel launch','FFTInterface CPU-GPU backend'}, palette.lightGray);
SupportFiguresCommon.drawArrow([0.315 0.775],[0.375 0.775]);
SupportFiguresCommon.drawArrow([0.635 0.775],[0.695 0.775]);
SupportFiguresCommon.drawBox([0.08 0.42 0.18 0.13], 'Host封装路径', {'对齐 Python cuSignal','235 / 235 PASS'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.31 0.42 0.18 0.13], 'CPU兼容路径', {'纯 C++ reference','无 CUDA 可运行'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.54 0.42 0.18 0.13], 'CUDA Device路径', {'resident device buffer','预分配 workspace'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.77 0.42 0.16 0.13], 'GPU并行核函数', {'1D/2D线程映射','FFT + kernel融合'}, palette.lightGray);
SupportFiguresCommon.drawArrow([0.185 0.70],[0.17 0.55]);
SupportFiguresCommon.drawArrow([0.505 0.70],[0.40 0.55]);
SupportFiguresCommon.drawArrow([0.825 0.70],[0.63 0.55]);
SupportFiguresCommon.drawArrow([0.72 0.485],[0.765 0.485]);
text(0.07,0.25,'并行计算策略：', 'FontSize',13,'FontWeight','bold');
text(0.07,0.20, '采样点级 1D 映射  |  Range-Doppler 二维网格  |  resident memory pipeline  |  共享 FFT/workspace  |  标准化错误码返回', 'FontSize',12);
SupportFiguresCommon.saveFig(fig, outDir, 'fig1_1_architecture_parallel_design');
end
