clc;
clear;
close all;

%% =========================
% 0. 全局设置
% =========================

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLegendFontName','Arial');
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');

config = plot_common();

sObj = settings;
desktopZoom = sObj.matlab.desktop.Zoom.PersonalValue;
sObj.matlab.desktop.Zoom.PersonalValue = 100;
zoomCleanup = onCleanup(@() restore_desktop_zoom(sObj,desktopZoom));

%% =========================
% 1. 数据读取与汇总
% =========================

accuracyPath = fullfile(config.data_dir,'operator_accuracy.csv');
cpuPerformancePath = fullfile(config.data_dir,'operator_fp32_multiscale.csv');
cusignalPerformancePath = fullfile(config.data_dir,'operator_multiscale_performance.csv');

accuracyData = readtable(accuracyPath,'VariableNamingRule','preserve');
cpuPerformanceData = readtable(cpuPerformancePath,'VariableNamingRule','preserve');
cusignalPerformanceData = readtable(cusignalPerformancePath,'VariableNamingRule','preserve');

idxAccuracyFP32 = string(accuracyData.input_dtype) == "FP32";
accuracyOperatorRaw = string(accuracyData.operator(idxAccuracyFP32));
accuracyErrorRaw = accuracyData.rmse(idxAccuracyFP32);

idxValidCpuPerformance = strcmpi(string(cpuPerformanceData.correctness_status),"pass") & ...
    isfinite(cpuPerformanceData.CPU_GPU_speedup) & ...
    cpuPerformanceData.CPU_GPU_speedup > 0;
cpuOperatorRaw = string(cpuPerformanceData.operator(idxValidCpuPerformance));
cpuSpeedupRaw = cpuPerformanceData.CPU_GPU_speedup(idxValidCpuPerformance);

idxValidCusignalPerformance = strcmpi(string(cusignalPerformanceData.status),"Success") & ...
    isfinite(cusignalPerformanceData.speedup) & ...
    cusignalPerformanceData.speedup > 0;
cusignalOperatorRaw = string(cusignalPerformanceData.operator(idxValidCusignalPerformance));
cusignalSpeedupRaw = cusignalPerformanceData.speedup(idxValidCusignalPerformance);

operator = unique([accuracyOperatorRaw; cpuOperatorRaw; cusignalOperatorRaw],'stable');
errorValue = max_by_operator(operator,accuracyOperatorRaw,accuracyErrorRaw);
cpuSpeedup = max_by_operator(operator,cpuOperatorRaw,cpuSpeedupRaw);
cusignalSpeedup = max_by_operator(operator,cusignalOperatorRaw,cusignalSpeedupRaw);

[operator,errorValue,cpuSpeedup,cusignalSpeedup] = apply_reference_order(...
    operator,errorValue,cpuSpeedup,cusignalSpeedup);

%% =========================
% 2. 摘要缩略图
% =========================

create_combined_preview(operator,errorValue,cpuSpeedup,cusignalSpeedup,config);

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
clear zoomCleanup;

%% =========================
% 3. 本地函数
% =========================

function create_combined_preview(operator,errorValue,cpuSpeedup,cusignalSpeedup,config)

fig = figure(...
    'Color','w',...
    'Units','centimeters',...
    'Position',[2 2 7 5],...
    'PaperUnits','centimeters',...
    'PaperPosition',[0 0 7 5],...
    'PaperSize',[7 5],...
    'Visible','off');

plot_accuracy_axes(fig,[0.305 0.720 0.655 0.240],operator,errorValue,false);
plot_speedup_axes(fig,[0.305 0.400 0.655 0.240],operator,cpuSpeedup,...
    {'加速比','(vs. CPU)'},[0.33 0.52 0.78],false,10.^(-2:2:4));
plot_speedup_axes(fig,[0.305 0.080 0.655 0.240],operator,cusignalSpeedup,...
    {'加速比','(vs. cuSignal)'},[0.88 0.47 0.16],true,10.^(-1:2:3));

exportgraphics(fig,...
    fullfile(config.output_dir,'operator_metrics_preview.jpg'),...
    'Resolution',300);
close(fig);

fprintf('Saved: %s\n',fullfile(config.output_dir,'operator_metrics_preview.jpg'));

end

function plot_accuracy_axes(fig,position,operator,errorValue,showXLabel)

idx = isfinite(errorValue) & errorValue > 0;
x = find(idx);
y = errorValue(idx);

ax = axes(fig,'Position',position);
scatter(ax,x,y,14,...
    'filled',...
    'MarkerFaceColor',[0.33 0.31 0.83],...
    'MarkerFaceAlpha',0.90,...
    'MarkerEdgeColor',[0.88 0.88 0.88],...
    'LineWidth',0.25);

set(ax,...
    'YScale','log',...
    'XTick',[],...
    'XTickLabel',{},...
    'FontName','Arial',...
    'FontSize',6.5,...
    'LineWidth',0.55,...
    'TickDir','out',...
    'TickLength',[0.004 0.004],...
    'TickLabelInterpreter','tex');

yMin = floor(log10(min(y))) - 1;
yMax = ceil(log10(max(y))) + 1;
ylim(ax,[10^yMin 10^yMax]);
xlim(ax,[0.2 length(operator)+0.8]);
tickExponent = uniform_log_tick_exponents(yMin,yMax);
set_log_ticks(ax,10.^tickExponent);

ylabel(ax,'');
if showXLabel
    xlabel(ax,'算子','FontName','Arial','FontSize',6.5);
else
    xlabel(ax,'');
end
grid(ax,'off');
box(ax,'on');

end

function plot_speedup_axes(fig,position,operator,speedup,~,barColor,...
    showXLabel,tickValue)

idx = isfinite(speedup) & speedup > 0;
x = find(idx);
y = speedup(idx);

ax = axes(fig,'Position',position);
bar(ax,x,y,0.72,...
    'FaceColor',barColor,...
    'EdgeColor',barColor,...
    'LineWidth',0.2);

set(ax,...
    'YScale','log',...
    'XTick',[],...
    'XTickLabel',{},...
    'FontName','Arial',...
    'FontSize',6.5,...
    'LineWidth',0.55,...
    'TickDir','out',...
    'TickLength',[0.004 0.004],...
    'TickLabelInterpreter','tex');

yMinExp = floor(log10(min(y))) - 1;
yMaxExp = ceil(log10(max(y)));
ylim(ax,[10^yMinExp 10^yMaxExp]);
xlim(ax,[0.2 length(operator)+0.8]);
tickValue = tickValue(tickValue >= 10^yMinExp & tickValue <= 10^yMaxExp);
set_log_ticks(ax,tickValue);

ylabel(ax,'');
if showXLabel
    xlabel(ax,'算子','FontName','Arial','FontSize',6.5);
else
    xlabel(ax,'');
end

grid(ax,'off');
box(ax,'on');

end

function value = max_by_operator(operator,operatorRaw,valueRaw)

value = nan(size(operator));

for i = 1:length(operator)
    idx = operatorRaw == operator(i);
    candidate = valueRaw(idx);
    candidate = candidate(isfinite(candidate) & candidate > 0);
    if ~isempty(candidate)
        value(i) = max(candidate);
    end
end

end

function tickExponent = uniform_log_tick_exponents(minExponent,maxExponent)

span = maxExponent - minExponent;
step = max(1,ceil(span/3));
tickExponent = minExponent:step:maxExponent;

end

function set_log_ticks(ax,tickValue)

tickValue = unique(tickValue,'stable');
set(ax,...
    'YTick',tickValue,...
    'YTickLabel',format_log_tick_labels(tickValue),...
    'YTickMode','manual',...
    'YTickLabelMode','manual');

end

function tickLabel = format_log_tick_labels(tickValue)

tickLabel = strings(size(tickValue));

for i = 1:numel(tickValue)
    exponent = log10(tickValue(i));
    if abs(exponent - round(exponent)) < 1e-10
        exponent = round(exponent);
        tickLabel(i) = sprintf('10^{%d}',exponent);
    else
        tickLabel(i) = sprintf('%.3g',tickValue(i));
    end
end

end

function restore_desktop_zoom(sObj,desktopZoom)

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;

end

function [operator,varargout] = apply_reference_order(operator,varargin)

referenceOrder = [
    "bsplines - cubic"
    "bsplines - gauss_spline"
    "bsplines - quadratic"
    "convolution - correlate"
    "convolution - correlate2d"
    "demod - fm_demod"
    "estimation - KalmanFilter"
    "filter_design - firwin"
    "filter_design - firwin2"
    "filtering - channelize_poly"
    "filtering - detrend"
    "filtering - firfilter"
    "filtering - firfilter2"
    "filtering - freq_shift"
    "filtering - hilbert"
    "filtering - hilbert2"
    "filtering - lfilter_zi"
    "filtering - sosfilt"
    "filtering - wiener"
    "filtering - decimate"
    "filtering - resample"
    "filtering - resample_poly"
    "filtering - upfirdn"
    "peak_finding - argrelextrema"
    "radartools - ambgfun"
    "radartools - ca_cfar"
    "radartools - cfar_alpha"
    "radartools - pulse_compression"
    "radartools - pulse_doppler"
    "spectral_analysis - csd"
    "spectral_analysis - istft"
    "spectral_analysis - lombscargle"
    "spectral_analysis - spectrogram"
    "spectral_analysis - stft"
    "spectral_analysis - vectorstrength"
    "waveforms - chirp"
    "waveforms - sawtooth"
    "waveforms - square"
    "waveforms - gausspulse"
    "waveforms - unit_impulse"
    "wavelets - cwt"
    "wavelets - morlet"
    "wavelets - morlet2"
    "wavelets - qmf"
    "wavelets - ricker"
    "windows - chebwin"
    "windows - general_cosine"
    "windows - general_gaussian"
    "windows - hamming"
    "windows - kaiser"
    "windows - parzen"
    "windows - taylor"
    "windows - triang"
];

referenceParts = split(referenceOrder," - ");
referenceOperator = referenceParts(:,2);

orderedIdx = zeros(0,1);
for i = 1:length(referenceOperator)
    idx = find(operator == referenceOperator(i),1,'first');
    if ~isempty(idx)
        orderedIdx(end+1,1) = idx; %#ok<AGROW>
    end
end

extraIdx = find(~ismember(operator,referenceOperator));
sortIdx = [orderedIdx; extraIdx];
operator = operator(sortIdx);
varargout = cell(size(varargin));

for i = 1:nargin-1
    varargout{i} = varargin{i}(sortIdx);
end

end
