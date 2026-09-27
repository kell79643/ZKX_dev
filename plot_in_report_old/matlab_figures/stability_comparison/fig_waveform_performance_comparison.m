function fig_waveform_performance_comparison()
[outDir, palette] = initStyle();
data = getWaveformData();

labels = {'LFM Chirp', 'Gaussian Pulse', 'Square Wave'};
x = categorical(labels);
x = reordercats(x, labels);

%% ============ 图1: 分辨率与旁瓣性能 ============
fig1 = figure('Color', 'w', 'Position', [100, 100, 1200, 600]);
tiledlayout(1, 2, 'TileSpacing', 'normal', 'Padding', 'normal');

%% 子图1: Range & Doppler Resolution
nexttile;
metrics1 = [data.rangeResolution; data.dopplerResolution];
b1 = bar(x, metrics1', 'Grouped');
b1(1).FaceColor = palette.blue;
b1(2).FaceColor = palette.purple;
ylabel('分辨率', 'FontWeight', 'bold', 'FontSize', 12);
title('a) 距离与多普勒分辨率', 'FontWeight', 'bold', 'FontSize', 12);
legend({'距离分辨率 (m)', '多普勒分辨率 (m/s)'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
ylim([0 max(metrics1(:))*1.25]);
for i = 1:3
    text(i-0.2, metrics1(1,i)+max(metrics1(:))*0.03, sprintf('%.2f', metrics1(1,i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    text(i+0.2, metrics1(2,i)+max(metrics1(:))*0.03, sprintf('%.2f', metrics1(2,i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

%% 子图2: Peak-to-SideLobe Ratio
nexttile;
pslrData = [abs(data.rangePSLR); abs(data.velocityPSLR)];
b2 = bar(x, pslrData', 'Grouped');
b2(1).FaceColor = palette.green;
b2(2).FaceColor = palette.orange;
ylabel('PSLR (dB, |值|)', 'FontWeight', 'bold', 'FontSize', 12);
title('b) 峰值旁瓣比 (越高越好)', 'FontWeight', 'bold', 'FontSize', 12);
legend({'距离PSLR', '速度PSLR'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
ylim([0 max(pslrData(:))*1.25]);
for i = 1:3
    text(i-0.2, pslrData(1,i)+max(pslrData(:))*0.03, sprintf('%.1f', pslrData(1,i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    text(i+0.2, pslrData(2,i)+max(pslrData(:))*0.03, sprintf('%.1f', pslrData(2,i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

sgtitle('雷达波形性能：分辨率与旁瓣特性', ...
    'FontSize', 14, 'FontWeight', 'bold');

saveFig(fig1, outDir, 'fig_waveform_performance_comparison_1');

%% ============ 图2: PAPR与综合评分 ============
fig2 = figure('Color', 'w', 'Position', [100, 100, 1200, 600]);
tiledlayout(1, 2, 'TileSpacing', 'normal', 'Padding', 'normal');

%% 子图3: Peak-to-Average Power Ratio
nexttile;
paprData = [data.papr];
colors = [palette.blue; palette.green; palette.orange];
b3 = bar(x, paprData, 'FaceColor', 'flat');
for i = 1:3
    b3.CData(i,:) = colors(i,:);
end
ylabel('PAPR (dB, 越低越好)', 'FontWeight', 'bold', 'FontSize', 12);
title('a) 峰值平均功率比', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
ylim([0 max(paprData)*1.25]);
for i = 1:3
    text(i, paprData(i)+max(paprData)*0.03, sprintf('%.1f', paprData(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

%% 子图4: Comprehensive Performance Score
nexttile;
scoreData = [data.rangeScore; data.velocityScore; data.paprScore; data.couplingScore];
b4 = bar(x, scoreData', 'Grouped');
b4(1).FaceColor = palette.blue;
b4(2).FaceColor = palette.purple;
b4(3).FaceColor = palette.green;
b4(4).FaceColor = palette.orange;
ylabel('评分 (越高越好)', 'FontWeight', 'bold', 'FontSize', 12);
title('b) 综合性能评分', 'FontWeight', 'bold', 'FontSize', 12);
legend({'距离分辨率', '速度分辨率', 'PAPR', '耦合性'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
ylim([0 12]);

sgtitle('雷达波形性能：功率效率与综合评分', ...
    'FontSize', 14, 'FontWeight', 'bold');

saveFig(fig2, outDir, 'fig_waveform_performance_comparison_2');

%% ============ 图3: 结论文字 ============
fig3 = figure('Color', 'w', 'Position', [100, 100, 900, 600]);
axes('Position', [0.08 0.08 0.84 0.84]);
axis off;

totalScores = sum(scoreData, 1);
bestIdx = find(totalScores == max(totalScores));

summaryText = {
    '雷达波形性能综合对比分析结论'
    ''
    '一、单项指标分析:'
    sprintf('  距离分辨率: Gaussian Pulse最优 (%.2f m), LFM Chirp/Square Wave较差 (%.2f m)', ...
        data.rangeResolution(2), data.rangeResolution(1))
    sprintf('  多普勒分辨率: LFM Chirp/Square Wave最优 (%.2f m/s), Gaussian Pulse较差 (%.2f m/s)', ...
        data.dopplerResolution(1), data.dopplerResolution(2))
    sprintf('  距离旁瓣比: LFM Chirp/Square Wave最优 (%.1f dB), Gaussian Pulse一般 (%.1f dB)', ...
        abs(data.rangePSLR(1)), abs(data.rangePSLR(2)))
    sprintf('  速度旁瓣比: Gaussian Pulse最优 (%.1f dB), Square Wave最差 (%.1f dB)', ...
        abs(data.velocityPSLR(2)), abs(data.velocityPSLR(3)))
    sprintf('  PAPR: Square Wave最优 (%.1f dB), LFM Chirp较差 (%.1f dB)', ...
        data.papr(3), data.papr(1))
    ''
    '二、综合评分:'
    sprintf('  LFM Chirp:      距离(7) + 速度(9) + PAPR(2) + 耦合(3) = %.0f分', totalScores(1))
    sprintf('  Gaussian Pulse: 距离(10) + 速度(5) + PAPR(3) + 耦合(10) = %.0f分', totalScores(2))
    sprintf('  Square Wave:    距离(7) + 速度(9) + PAPR(10) + 耦合(5) = %.0f分', totalScores(3))
    ''
    '三、结论:'
    sprintf('  综合评分最高: %s (%.0f分)', labels{bestIdx}, totalScores(bestIdx))
    ''
    '  LFM Chirp: 速度分辨率优异，但PAPR较高，适合高速目标检测场景'
    '  Gaussian Pulse: 距离分辨率和耦合性优异，但速度分辨率较差，适合近距离高精度成像'
    '  Square Wave: PAPR最优，但旁瓣性能较差，适合对功耗敏感的低成本应用'
    ''
    '四、应用建议:'
    '  • 高速目标跟踪: 推荐 LFM Chirp'
    '  • 高精度成像:    推荐 Gaussian Pulse'
    '  • 低功耗应用:    推荐 Square Wave'
};

text(0.05, 0.95, summaryText, 'Units', 'normalized', 'FontSize', 11, ...
    'VerticalAlignment', 'top', 'FontName', 'Microsoft YaHei');
title('雷达波形性能综合对比分析结论', 'FontSize', 14, 'FontWeight', 'bold');

saveFig(fig3, outDir, 'fig_waveform_performance_comparison_summary');

fprintf('Waveform comparison figures saved to %s\n', outDir);
end

function [outDir, palette] = initStyle()
baseDir = fileparts(mfilename('fullpath'));
outDir = fullfile(baseDir, 'output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultFigureColor', [1 1 1]);
palette.blue = [0.45 0.60 0.80];
palette.cyan = [0.35 0.70 0.75];
palette.red = [0.70 0.40 0.45];
palette.orange = [0.75 0.55 0.35];
palette.green = [0.40 0.70 0.50];
palette.purple = [0.55 0.45 0.75];
end

function data = getWaveformData()
data.waveforms = {'LFM Chirp', 'Gaussian Pulse', 'Square Wave'};
data.rangeResolution = [8.99, 0.30, 8.99];
data.dopplerResolution = [0.54, 6.60, 0.53];
data.rangePSLR = [-17.84, -13.20, -17.84];
data.velocityPSLR = [-5.93, -13.20, -0.02];
data.papr = [33.98, 30.0, 0.00];
data.computeTime = [40, 40, 40];

data.rangeScore = [7, 10, 7];
data.velocityScore = [9, 5, 9];
data.paprScore = [2, 3, 10];
data.couplingScore = [3, 10, 5];
end

function saveFig(fig, outDir, name)
exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
saveas(fig, fullfile(outDir, [name '.fig']));
end