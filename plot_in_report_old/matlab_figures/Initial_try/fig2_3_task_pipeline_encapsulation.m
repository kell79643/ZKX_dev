function fig2_3_task_pipeline_encapsulation()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1700 920]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile; axis off;
title('a  任务1：雷达目标检测与多普勒分析算法','FontSize',15,'FontWeight','bold');
drawOfficialPipeline({ ...
    {'接收信号模拟', {'目标回波延迟','多普勒频移 + 噪声'}}, ...
    {'脉冲压缩', {'pulse\_compression','距离分辨率提升'}}, ...
    {'多普勒处理', {'pulse\_doppler','频移信息提取'}}, ...
    {'恒虚警率检测', {'ca\_cfar','cfar\_alpha'}}, ...
    {'波形与模糊函数分析', {'ambgfun','主瓣窄、旁瓣低'}} ...
}, palette.lightBlue);

nexttile; axis off;
title('b  任务2：多域特征提取定位算法','FontSize',15,'FontWeight','bold');
drawOfficialPipeline({ ...
    {'波形生成', {'waveforms 至少三种','目标 + 干扰 + 噪声'}}, ...
    {'滤波', {'filter\_design','filtering + Windows'}}, ...
    {'B 样条平滑', {'bsplines','弱化噪声'}}, ...
    {'多域特征提取', {'demod / correlate','spectral / wavelets'}}, ...
    {'关键特征点定位', {'峰值查找','定位关键特征点'}}, ...
    {'参数估计', {'kalmanfilter','状态/参数估计'}} ...
}, palette.lightGreen);

SupportFiguresCommon.saveFig(fig, outDir, 'fig2_3_task_pipeline_encapsulation');
end

function drawOfficialPipeline(nodes, color)
n = numel(nodes);
x0 = 0.035;
y = 0.33;
w = 0.135;
h = 0.28;
gap = (0.96 - x0 - n*w) / (n-1);
for i = 1:n
    x = x0 + (i-1)*(w+gap);
    node = nodes{i};
    SupportFiguresCommon.drawBox([x y w h], node{1}, node{2}, color);
    if i < n
        SupportFiguresCommon.drawArrow([x+w+0.005 y+h/2], [x+w+gap-0.005 y+h/2]);
    end
end
end
