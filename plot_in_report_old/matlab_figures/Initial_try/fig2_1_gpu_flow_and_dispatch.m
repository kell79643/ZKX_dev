function fig2_1_gpu_flow_and_dispatch()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1500 800]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; axis off;
title('a  NVIDIA GPU 底层执行链路','FontSize',15,'FontWeight','bold');
labels = {'参数校验','Host -> Device 拷贝','workspace申请/复用','kernel / cuFFT 启动','Device同步','Device -> Host 回传','错误码返回'};
for i = 1:numel(labels)
    y = 0.84 - (i-1)*0.13;
    SupportFiguresCommon.drawBox([0.15 y 0.52 0.070], labels{i}, {''}, palette.lightBlue);
    if i < numel(labels)
        SupportFiguresCommon.drawArrow([0.41 y-0.005], [0.41 y-0.055]);
    end
end
nexttile; axis off;
title('b  跨平台兼容调度逻辑','FontSize',15,'FontWeight','bold');
SupportFiguresCommon.drawBox([0.20 0.78 0.60 0.10], '统一 public API', {'函数名 / 参数 / 返回状态一致'}, palette.lightGreen);
SupportFiguresCommon.drawBox([0.20 0.60 0.60 0.10], '编译或运行时 backend 选择', {'CUDA 可用性、目标平台、测试模式'}, palette.lightGreen);
SupportFiguresCommon.drawBox([0.08 0.37 0.35 0.11], 'CPU reference backend', {'纯 C++ 路径','无 CUDA 环境可运行'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.57 0.37 0.35 0.11], 'CUDA Device backend', {'DeviceArray resident','高性能路径'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.25 0.16 0.50 0.10], '一致输出 + 标准错误码', {'上层任务无需区分底层硬件'}, palette.lightBlue);
SupportFiguresCommon.drawArrow([0.50 0.78],[0.50 0.70]);
SupportFiguresCommon.drawArrow([0.50 0.60],[0.255 0.48]);
SupportFiguresCommon.drawArrow([0.50 0.60],[0.745 0.48]);
SupportFiguresCommon.drawArrow([0.255 0.37],[0.41 0.26]);
SupportFiguresCommon.drawArrow([0.745 0.37],[0.59 0.26]);
text(0.61,0.52,'CUDA可用', 'FontSize',10,'Color',palette.green);
text(0.16,0.52,'CPU fallback', 'FontSize',10,'Color',palette.gray);
SupportFiguresCommon.saveFig(fig, outDir, 'fig2_1_gpu_flow_and_dispatch');
end
