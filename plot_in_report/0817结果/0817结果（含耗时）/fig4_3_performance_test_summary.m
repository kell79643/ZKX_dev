function fig4_3_performance_test_summary()
% 初始化输出目录和调色板（内联实现，不依赖 SupportFiguresCommon）
outDir = fullfile(fileparts(mfilename('fullpath')), 'figures_output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
palette = [];  % 未使用

% 数据来源: 03_任务结果/Task1|Task2 各case的memory_trace CSV
% gpu_used_bytes (cudaMemGetInfo测量) = 357777408 bytes ≈ 341.2 MB
% GPU显存在流水线启动时一次性分配，各步骤间恒定不变
gpuPeakBytes = 357777408;
task1StepNames = {'Step5','Step4','Step3','Step2','Step1'};
task1StepMem = gpuPeakBytes * ones(1, 5) / 1024 / 1024;
task2StepNames = {'Step6','Step5','Step4','Step3','Step2','Step1'};
task2StepMem = gpuPeakBytes * ones(1, 6) / 1024 / 1024;

colorTask1 = [0.58 0.68 0.76];
colorTask2 = [0.78 0.65 0.68];

fig = figure('Color','w','Units','centimeters','Position',[2 2 18 12]);

nTask1 = numel(task1StepNames);
nTask2 = numel(task2StepNames);
totalSteps = nTask1 + nTask2;

allY = 1:nTask2+nTask1+1;
allMem = [task2StepMem(:); 0; task1StepMem(:)];
allNames = cat(1, task2StepNames(:), {''}, task1StepNames(:));

maxVal = max(allMem(:));

ax = axes('Parent', fig, 'Position', [0.15 0.18 0.80 0.72]);

barWidth = 0.65;
yTask2 = 1:nTask2;
yTask1 = nTask2+2:nTask2+nTask1+1;

hold(ax, 'on');

b2 = barh(ax, yTask2, task2StepMem(:), barWidth, 'FaceColor', colorTask2, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 1.0);
b1 = barh(ax, yTask1, task1StepMem(:), barWidth, 'FaceColor', colorTask1, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 1.0);


for i = 1:nTask2
    yPos = i;
    text(ax, allMem(yPos)*1.08, yPos, sprintf('%.1f', task2StepMem(i)), ...
        'FontSize', 12, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'Color', colorTask2);
end

for i = 1:nTask1
    yPos = nTask2 + 1 + i;
    text(ax, allMem(yPos)*1.08, yPos, sprintf('%.1f', task1StepMem(i)), ...
        'FontSize', 12, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'Color', colorTask1);
end

set(ax, 'YTick', allY, 'YTickLabel', allNames);
grid(ax, 'on'); grid(ax, 'minor');
box(ax, 'on');
xlabel(ax, '显存 (MB)', 'FontSize', 16);
ylabel(ax, '步骤', 'FontSize', 14);
xlim(ax, [1e-2 maxVal*2.5]);
ax.XScale = 'log';
set(ax, 'FontSize', 13);

lgd = legend(ax, [b1, b2], {'任务一', '任务二'}, 'Location', 'best', 'FontSize', 14);
lgd.Box = 'on';

% 保存图像（内联实现：PNG + FIG + EPS/PDF 多格式）
baseName = 'fig4_3_performance_test_summary';
figPath = fullfile(outDir, [baseName '.fig']);
pngPath = fullfile(outDir, [baseName '.png']);
print(fig, pngPath, '-dpng', '-r300');
savefig(fig, figPath);
try
    epsPath = fullfile(outDir, [baseName '.eps']);
    print(fig, epsPath, '-depsc', '-r300');
catch ME
    warning('保存EPS失败: %s', ME.message);
end
fprintf('图像已保存至目录: %s\n', outDir);
end