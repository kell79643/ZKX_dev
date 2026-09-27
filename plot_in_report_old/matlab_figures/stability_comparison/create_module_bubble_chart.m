clc; clear; close all;

%% =========================
% 1. 原始数据：GPU/Python 加速比
% =========================
rawData = [
    "bsplines - cubic - 15.12x"
    "bsplines - gauss_spline - 42.25x"
    "bsplines - quadratic - 38.50x"
    "convolution - correlate - 4.11x"
    "convolution - correlate2d - 0.0006x"
    "demod - fm_demod - 16.96x"
    "estimation - KalmanFilter - 4.95x"
    "filter_design - firwin - 18.65x"
    "filter_design - firwin2 - 14.36x"
    "filtering - channelize_poly - 0.0069x"
    "filtering - detrend - 2.67x"
    "filtering - firfilter - 65.00x"
    "filtering - firfilter2 - 3.88x"
    "filtering - freq_shift - 35.44x"
    "filtering - hilbert - 3.45x"
    "filtering - hilbert2 - 0.09x"
    "filtering - lfilter_zi - 4.75x"
    "filtering - sosfilt - 0.64x"
    "filtering - wiener - 8.07x"
    "filtering - decimate - 7.59x"
    "filtering - resample - 0.51x"
    "filtering - resample_poly - 35.34x"
    "filtering - upfirdn - 7.95x"
    "peak_finding - argrelextrema - 8.01x"
    "radartools - ambgfun - 0.05x"
    "radartools - ca_cfar - 61.00x"
    "radartools - cfar_alpha - 0.50x"
    "radartools - pulse_compression - 0.14x"
    "radartools - pulse_doppler - 0.0105x"
    "spectral_analysis - csd - 0.07x"
    "spectral_analysis - istft - 1.58x"
    "spectral_analysis - lombscargle - 7.72x"
    "spectral_analysis - spectrogram - 0.13x"
    "spectral_analysis - stft - 0.22x"
    "spectral_analysis - vectorstrength - 2.04x"
    "waveforms - chirp - 24.18x"
    "waveforms - sawtooth - 30.14x"
    "waveforms - square - 54.83x"
    "waveforms - gausspulse - 19.33x"
    "waveforms - unit_impulse - 34.22x"
    "wavelets - cwt - 66.10x"
    "wavelets - morlet - 31.36x"
    "wavelets - morlet2 - 27.70x"
    "wavelets - qmf - 54.60x"
    "wavelets - ricker - 30.88x"
    "windows - chebwin - 2.90x"
    "windows - general_cosine - 33.33x"
    "windows - general_gaussian - 19.92x"
    "windows - hamming - 47.00x"
    "windows - kaiser - 16.08x"
    "windows - parzen - 38.86x"
    "windows - taylor - 20.65x"
    "windows - triang - 44.33x"
];

%% =========================
% 2. 解析数据
% =========================
parts = split(rawData, " - ");

moduleName = parts(:, 1);
funcName   = parts(:, 2);
valueStr   = parts(:, 3);

isNA = strcmpi(valueStr, "N/A");

speedup = nan(size(valueStr));
speedup(~isNA) = str2double(erase(valueStr(~isNA), "x"));

% 保持模块出现顺序
moduleList = unique(moduleName, "stable");
nModule = numel(moduleList);

% 纵轴：模块
y = zeros(size(moduleName));

% 横轴：每个模块内部的函数序号 1, 2, 3, ...
x = zeros(size(moduleName));

for i = 1:nModule
    idx = find(moduleName == moduleList(i));
    y(idx) = i;
    x(idx) = 1:numel(idx);
end

maxFuncNum = max(x);

%% =========================
% 3. 气泡大小与颜色映射
%    使用 log10 映射，避免大值压制小值
% =========================
validValue = speedup(~isNA);
logValue = log10(validValue);

scaleMin = 0.01;
scaleMax = 100;
logMin = log10(scaleMin);
logMax = log10(scaleMax);

bubbleMin = 60;
bubbleMax = 760;

bubbleSize = nan(size(speedup));
bubbleSize(~isNA) = bubbleMin + ...
    (log10(min(max(speedup(~isNA), scaleMin), scaleMax)) - logMin) ./ (logMax - logMin) ...
    * (bubbleMax - bubbleMin);

colorValue = nan(size(speedup));
colorValue(~isNA) = log10(min(max(speedup(~isNA), scaleMin), scaleMax));

%% =========================
% 4. 绘图
% =========================
figure("Color", "w", "Position", [100, 100, 1800, 850]);
hold on;

% 有效数据点
idxValid = ~isNA;

scatter( ...
    x(idxValid), y(idxValid), ...
    bubbleSize(idxValid), ...
    colorValue(idxValid), ...
    "filled", ...
    "MarkerFaceAlpha", 0.78, ...
    "MarkerEdgeColor", [0.2 0.2 0.2], ...
    "LineWidth", 0.5);

% N/A 数据点：灰色空心圆 + 叉号
idxNA = isNA;

scatter( ...
    x(idxNA), y(idxNA), ...
    180, ...
    "o", ...
    "MarkerFaceColor", "none", ...
    "MarkerEdgeColor", [0.45 0.45 0.45], ...
    "LineWidth", 1.3);

scatter( ...
    x(idxNA), y(idxNA), ...
    80, ...
    "x", ...
    "MarkerEdgeColor", [0.45 0.45 0.45], ...
    "LineWidth", 1.5);

%% =========================
% 5. 添加函数名与数值标签
% =========================
for i = 1:numel(funcName)
    if isNA(i)
        labelText = sprintf("%s\nN/A", funcName(i));
    elseif speedup(i) < 0.01
        labelText = sprintf("%s\n%.4fx", funcName(i), speedup(i));
    elseif speedup(i) < 0.1
        labelText = sprintf("%s\n%.3fx", funcName(i), speedup(i));
    else
        labelText = sprintf("%s\n%.2fx", funcName(i), speedup(i));
    end

    text( ...
        x(i) + 0.18, y(i), ...
        labelText, ...
        "FontSize", 8.5, ...
        "HorizontalAlignment", "left", ...
        "VerticalAlignment", "middle", ...
        "Interpreter", "none");
end

%% =========================
% 6. 坐标轴设置
% =========================
set(gca, ...
    "YTick", 1:nModule, ...
    "YTickLabel", moduleList, ...
    "XTick", 1:maxFuncNum, ...
    "XTickLabel", repmat("", 1, maxFuncNum), ...
    "FontName", "Arial", ...
    "FontSize", 11, ...
    "LineWidth", 1);

set(gca, "YDir", "reverse");

xlabel("函数", "FontSize", 14, "FontWeight", "bold");
ylabel("模块", "FontSize", 14, "FontWeight", "bold");
title("信号处理函数 GPU/Python 加速比气泡图", "FontSize", 17, "FontWeight", "bold");

xlim([0.6, maxFuncNum + 2.2]);
ylim([0.4, nModule + 0.6]);

grid on;
box off;

set(gca, ...
    "GridLineStyle", "--", ...
    "GridAlpha", 0.20);

%% =========================
% 7. 颜色条
% =========================
blue = [0.12 0.33 0.78];
mid = [0.96 0.96 0.96];
red = [0.82 0.18 0.15];
nColor = 256;
halfN = nColor / 2;
cmapCold = [linspace(blue(1), mid(1), halfN)', ...
    linspace(blue(2), mid(2), halfN)', ...
    linspace(blue(3), mid(3), halfN)'];
cmapWarm = [linspace(mid(1), red(1), halfN)', ...
    linspace(mid(2), red(2), halfN)', ...
    linspace(mid(3), red(3), halfN)'];
colormap([cmapCold; cmapWarm]);

caxis([logMin, logMax]);

cb = colorbar;
cb.Label.String = "GPU/Python 加速比（颜色，log_{10}尺度）";
cb.Label.FontSize = 12;

tickValue = [0.01, 0.1, 1, 10, 100];

cb.Ticks = log10(tickValue);
cb.TickLabels = compose("%.2gx", tickValue);

%% =========================
% 8. 气泡大小图例
% =========================
legendValue = [0.01, 0.1, 1, 10, 100];

legendValue = legendValue( ...
    legendValue >= scaleMin & legendValue <= scaleMax);

legendSize = bubbleMin + ...
    (log10(legendValue) - logMin) ./ (logMax - logMin) ...
    * (bubbleMax - bubbleMin);

hLegend = gobjects(numel(legendValue), 1);

for i = 1:numel(legendValue)
    hLegend(i) = scatter( ...
        nan, nan, ...
        legendSize(i), ...
        "o", ...
        "MarkerFaceColor", [0.85 0.85 0.85], ...
        "MarkerEdgeColor", [0.45 0.45 0.45], ...
        "LineWidth", 0.8);
end

lgd = legend( ...
    hLegend, ...
    compose("%gx", legendValue), ...
    "Title", "GPU/Python 加速比（气泡大小）", ...
    "Location", "eastoutside");

lgd.Box = "off";
lgd.FontSize = 10;

%% =========================
% 9. 导出图片
% =========================
% exportgraphics(gcf, "speedup_bubble_chart.png", "Resolution", 300);
