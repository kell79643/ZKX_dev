function fig2_2_operator_tree_api_status()
[outDir, palette] = SupportFiguresCommon.init();
fig = figure('Color','w','Position',[100 100 1600 900]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; axis off;
title('a  面向任务的算子分类树','FontSize',15,'FontWeight','bold');
SupportFiguresCommon.drawBox([0.32 0.84 0.36 0.09], 'cuSignal C++ 算子库', {'53个 public API'}, palette.lightBlue);
SupportFiguresCommon.drawBox([0.05 0.61 0.26 0.12], '任务1 必做算子', {'pulse compression','pulse doppler','CA/OS-CFAR / ambgfun'}, palette.lightGreen);
SupportFiguresCommon.drawBox([0.37 0.61 0.26 0.12], '任务2 推荐算子', {'waveforms / filtering','B-splines / spectral / Kalman'}, palette.lightGreen);
SupportFiguresCommon.drawBox([0.69 0.61 0.26 0.12], '附加算子', {'resampling / Hilbert','windows / wavelets / STFT'}, palette.lightGreen);
SupportFiguresCommon.drawArrow([0.50 0.84],[0.18 0.73]);
SupportFiguresCommon.drawArrow([0.50 0.84],[0.50 0.73]);
SupportFiguresCommon.drawArrow([0.50 0.84],[0.82 0.73]);
SupportFiguresCommon.drawBox([0.07 0.36 0.22 0.11], '雷达检测场景', {'距离 / 速度估计','恒虚警检测'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.39 0.36 0.22 0.11], '特征提取场景', {'时域 / 频域','时频域 / 小波'}, palette.lightGray);
SupportFiguresCommon.drawBox([0.71 0.36 0.22 0.11], '扩展应用场景', {'预处理 / 重采样','非平稳信号分析'}, palette.lightGray);
nexttile;
title('b  代表函数的 Python / CPU / GPU 覆盖矩阵','FontSize',15,'FontWeight','bold');
coverFuncs = {'pulse\_compression','pulse\_doppler','ambgfun','ca\_cfar','spectrogram','stft','resample\_poly','chirp','firwin','KalmanFilter'};
coverCols = {'Python cuSignal对比','CPU reference','CUDA Device','FFT CPU backend'};
cover = [1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 0; 1 1 1 1; 1 1 1 1; 1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0];
imagesc(cover);
colormap(gca, [0.94 0.94 0.94; palette.blue]);
caxis([0 1]);
set(gca, 'XTick', 1:numel(coverCols), 'XTickLabel', coverCols, 'YTick', 1:numel(coverFuncs), 'YTickLabel', coverFuncs, 'TickLength', [0 0], 'XAxisLocation', 'top');
xtickangle(25); grid on;
set(gca, 'GridColor', [1 1 1], 'GridAlpha', 1, 'LineWidth', 1.0);
for r = 1:size(cover,1)
    for c = 1:size(cover,2)
        if cover(r,c) == 1
            text(c, r, '已覆盖', 'HorizontalAlignment','center', 'FontSize',9, 'FontWeight','bold', 'Color',[1 1 1]);
        else
            text(c, r, 'N/A', 'HorizontalAlignment','center', 'FontSize',9, 'FontWeight','bold', 'Color',palette.gray);
        end
    end
end
xlabel('验证/实现路径'); ylabel('具体函数');
SupportFiguresCommon.saveFig(fig, outDir, 'fig2_2_operator_tree_api_status');
end
