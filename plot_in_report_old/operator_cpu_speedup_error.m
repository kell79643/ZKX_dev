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

%% =========================
% 1. 数据读取
% =========================

accuracy_path = fullfile(config.data_dir,...
    'operator_accuracy.csv');

performance_path = fullfile(config.data_dir,...
    'operator_fp32_multiscale.csv');


accuracy_data = readtable(accuracy_path,...
    'VariableNamingRule','preserve');


performance_data = readtable(performance_path,...
    'VariableNamingRule','preserve');



%% =========================
% 2. 合并数据
% =========================


% 精度数据：仅使用原始精度文件中 FP32 输入时的 RMSE。

idxAccuracyFP32 = string(accuracy_data.input_dtype) == "FP32";

accuracy_table = table(...
    string(accuracy_data.operator(idxAccuracyFP32)),...
    accuracy_data.rmse(idxAccuracyFP32),...
    'VariableNames',...
    {'operator','error'});



% 性能数据：只使用有效通过记录，后续按算子取最高 CPU 加速比。

idxValid = strcmpi(string(performance_data.correctness_status),"pass") & ...
    isfinite(performance_data.CPU_GPU_speedup) & ...
    performance_data.CPU_GPU_speedup > 0;

performanceOperatorRaw = string(performance_data.operator(idxValid));
performanceSpeedupRaw = performance_data.CPU_GPU_speedup(idxValid);

performanceOperator = unique(performanceOperatorRaw,'stable');
performanceSpeedup = nan(size(performanceOperator));

for i = 1:length(performanceOperator)
    idx = performanceOperatorRaw == performanceOperator(i);
    performanceSpeedup(i) = max(performanceSpeedupRaw(idx));
end

performance_table = table(...
    performanceOperator,...
    performanceSpeedup,...
    'VariableNames',...
    {'operator','speedup'});



data = outerjoin(...
    accuracy_table,...
    performance_table,...
    'Keys','operator',...
    'MergeKeys',true);



operatorRaw = string(data.operator);
errorRaw = data.error;
speedupRaw = data.speedup;

operator = unique(operatorRaw,'stable');
errorValue = nan(size(operator));
speedup = nan(size(operator));

for i = 1:length(operator)
    idx = operatorRaw == operator(i);

    err = errorRaw(idx);
    err = err(isfinite(err));
    if ~isempty(err)
        errorValue(i) = max(err);
    end

    spd = speedupRaw(idx);
    spd = spd(isfinite(spd) & spd > 0);
    if ~isempty(spd)
        speedup(i) = max(spd);
    end
end

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
referenceModule = referenceParts(:,1);
referenceOperator = referenceParts(:,2);

orderedIdx = zeros(0,1);
for i = 1:length(referenceOperator)
    idx = find(operator == referenceOperator(i),1,'first');
    if ~isempty(idx)
        orderedIdx(end+1,1) = idx; %#ok<SAGROW>
    end
end

extraIdx = find(~ismember(operator,referenceOperator));
operator = operator([orderedIdx; extraIdx]);
errorValue = errorValue([orderedIdx; extraIdx]);
speedup = speedup([orderedIdx; extraIdx]);



%% =========================
% 3. 手动模块映射
% =========================


moduleName = strings(size(operator));


for i = 1:length(operator)

    refIdx = find(referenceOperator == operator(i),1,'first');

    if ~isempty(refIdx)
        moduleName(i) = referenceModule(refIdx);
        continue;
    end

    opKey = lower(operator(i));

    if ismember(opKey,["cubic","gauss_spline","quadratic"])
        moduleName(i)="bsplines";
    elseif ismember(opKey,["correlate","correlate2d"])
        moduleName(i)="convolution";
    elseif ismember(opKey,"fm_demod")
        moduleName(i)="demod";
    elseif ismember(opKey,"kalmanfilter")
        moduleName(i)="estimation";
    elseif ismember(opKey,["firwin","firwin2"])
        moduleName(i)="filter_design";
    elseif ismember(opKey,["channelize_poly","detrend","firfilter",...
            "firfilter2","freq_shift","hilbert","hilbert2","lfilter_zi",...
            "sosfilt","wiener","decimate","resample","resample_poly","upfirdn"])
        moduleName(i)="filtering";
    elseif ismember(opKey,"argrelextrema")
        moduleName(i)="peak_finding";
    elseif ismember(opKey,["ambgfun","ca_cfar","cfar_alpha",...
            "pulse_compression","pulse_doppler"])
        moduleName(i)="radartools";
    elseif ismember(opKey,["csd","istft","lombscargle","spectrogram",...
            "stft","vectorstrength"])
        moduleName(i)="spectral_analysis";
    elseif ismember(opKey,["chirp","sawtooth","square","gausspulse","unit_impulse"])
        moduleName(i)="waveforms";
    elseif ismember(opKey,["cwt","morlet","morlet2","qmf","ricker"])
        moduleName(i)="wavelets";
    elseif ismember(opKey,["chebwin","general_cosine","general_gaussian",...
            "hamming","hann","kaiser","parzen","taylor","triang"])
        moduleName(i)="windows";
    else
        moduleName(i)="others";
    end

end

filteringSecondRow = [
    "lfilter_zi"
    "sosfilt"
    "wiener"
    "decimate"
    "resample"
    "resample_poly"
    "upfirdn"
];

idxFiltering = moduleName == "filtering";
idxFilteringSecondRow = idxFiltering & ismember(operator,filteringSecondRow);

moduleName(idxFiltering & ~idxFilteringSecondRow) = "filtering I";
moduleName(idxFilteringSecondRow) = "filtering II";



%% =========================
% 4. 坐标计算
% =========================


moduleList = unique(moduleName,'stable');
moduleTickLabel = replace(moduleList,"_","\_\newline");

nModule = numel(moduleList);


x=zeros(size(operator));
y=zeros(size(operator));


for i=1:nModule

    idx=find(moduleName==moduleList(i));

    y(idx)=i;

    x(idx)=1:length(idx);

end


maxFuncNum=max(x);



%% =========================
% 5. speedup bubble size
% =========================


idxSpeedup = isfinite(speedup) & speedup > 0;


validSpeedup=speedup(idxSpeedup);


logSpeedup=log10(validSpeedup);


logMin=min(logSpeedup);
logMax=max(logSpeedup);


bubbleMin=14;
bubbleMax=240;


bubbleSize=nan(size(speedup));


if logMax > logMin
    sizeNorm = (log10(speedup(idxSpeedup))-logMin)...
        /(logMax-logMin);
    sizeNorm = sizeNorm .^ 1.75;

    bubbleSize(idxSpeedup)=...
        bubbleMin + ...
        sizeNorm*(bubbleMax-bubbleMin);
else
    bubbleSize(idxSpeedup)=bubbleMin;
end



% 防止极小值不可见

bubbleSize(idxSpeedup)=...
    max(bubbleSize(idxSpeedup),bubbleMin);

if logMax > logMin
    oneNorm = (log10(1)-logMin)/(logMax-logMin);
    oneNorm = max(0,min(1,oneNorm)) .^ 1.75;
    bubbleSizeOne = bubbleMin + oneNorm*(bubbleMax-bubbleMin);
else
    bubbleSizeOne = bubbleMin;
end


%% =========================
% 6. speedup颜色
% =========================


idxError = isfinite(errorValue);
rmseMedian = median(errorValue(idxError),'omitnan');
rmseMean = mean(errorValue(idxError),'omitnan');
rmseMax = max(errorValue(idxError));

speedupMedian = median(validSpeedup,'omitnan');
speedupMean = mean(validSpeedup,'omitnan');
speedupMax = max(validSpeedup);

rmseMedianText = format_sci_tex(rmseMedian,3);
rmseMeanText = format_sci_tex(rmseMean,3);
rmseMaxText = format_sci_tex(rmseMax,3);
speedupMedianText = sprintf('%.3g',speedupMedian);
speedupMeanText = sprintf('%.3g',speedupMean);
speedupMaxText = format_sci_tex(speedupMax,3);

colorValue = nan(size(speedup));
colorValue(idxSpeedup) = log10(speedup(idxSpeedup));

colorCenter = log10(0.85);
colorLim = [
    min(colorValue(idxSpeedup))
    max(colorValue(idxSpeedup))
];
colorLim(1) = min(colorLim(1),colorCenter);
colorLim(2) = max(colorLim(2),colorCenter);



%% =========================
% 7. 绘图
% =========================

sObj = settings;
zoomSetting = sObj.matlab.desktop.Zoom;
hadPersonalZoom = hasPersonalValue(zoomSetting);
if hadPersonalZoom
    desktopZoom = zoomSetting.PersonalValue;
else
    desktopZoom = [];
end
zoomSetting.PersonalValue = 100;
zoomCleanup = onCleanup(@() restore_desktop_zoom( ...
    zoomSetting,desktopZoom,hadPersonalZoom));

fig=figure(...
    'Color','w',...
    'Units','centimeters',...
    'Position',[2 2 15 12],...
    'PaperUnits','centimeters',...
    'PaperPosition',[0 0 15 12],...
    'PaperSize',[15 12]);


ax=axes(fig);

hold(ax,'on');



%% 有speedup：实心圆，统一浅灰细边框表示已有精度验证

scatter(ax,...
    x(idxSpeedup),...
    y(idxSpeedup),...
    bubbleSize(idxSpeedup),...
    colorValue(idxSpeedup),...
    'filled',...
    'MarkerFaceAlpha',0.82,...
    'MarkerEdgeColor',[0.68 0.68 0.68],...
    'LineWidth',0.35);


%% =========================
% 8. 算子标签
% =========================


for i=1:length(operator)

    text(ax,...
        x(i),...
        y(i)+0.25,...
        wrap_operator_label(operator(i),9),...
        'HorizontalAlignment','center',...
        'VerticalAlignment','top',...
        'Rotation',0,...
        'FontSize',5,...
        'FontName','Arial',...
        'Interpreter','none');

end



%% =========================
% 9. 坐标轴
% =========================


set(ax,...
    'YTick',1:nModule,...
    'YTickLabel',cellstr(moduleTickLabel),...
    'XTick',1:maxFuncNum,...
    'XTickLabel',repmat({''},1,maxFuncNum),...
    'FontName','Arial',...
    'FontSize',8,...
    'LineWidth',0.55,...
    'TickLabelInterpreter','tex');



set(ax,'YDir','reverse');


xlabel(ax,'算子','FontSize',9,'FontName','Arial');

ylabel(ax,'模块','FontSize',9,'FontName','Arial');


title(ax,...
    'ZQ500 全算子性能分析',...
    'FontSize',11,...
    'FontWeight','bold',...
    'FontName','Arial');



xlim(ax,[0.2 maxFuncNum+0.8]);

ylim(ax,[0.5 nModule+1]);


grid(ax,'on');

box(ax,'on');

set(ax,...
    'GridLineStyle','--',...
    'GridAlpha',0.5,...
    'GridColor',[0.8 0.8 0.8]);


ax.Position=[
    0.16
    0.12
    0.58
    0.80];



%% =========================
% 10. speedup colorbar
% =========================


clim(ax,colorLim);


colormap(ax,coolwarm_centered_biased(256,colorLim,colorCenter));


cb=colorbar(ax);

cb.Location='eastoutside';
cb.Position=[0.76 0.12 0.020 0.80];

cb.Label.String='加速比 (vs. CPU)';


cb.FontName='Arial';
cb.FontSize=8;
cb.Label.FontName='Arial';
cb.Label.FontSize=8;
cb.AxisLocation='in';

tickSpeed = [
    0.1
    1
    10
    100
    1000
];

tickSpeed = tickSpeed(tickSpeed >= min(validSpeedup) & ...
    tickSpeed <= max(validSpeedup));

if isempty(tickSpeed)
    tickSpeed = median(validSpeedup);
end

cb.Ticks = log10(tickSpeed);
cb.TickLabels = compose('%g',tickSpeed);



%% =========================
% 11. speedup 图例
% =========================


legendSpeed=[
    1
    10
    100
    1000
];

legendSpeed = legendSpeed(legendSpeed >= min(validSpeedup) & ...
    legendSpeed <= max(validSpeedup));

if isempty(legendSpeed)
    legendSpeed = median(validSpeedup);
end

if logMax > logMin
    legendNorm = (log10(legendSpeed)-logMin)...
        /(logMax-logMin);
    legendNorm = max(0,min(1,legendNorm)) .^ 1.75;
    legendSize=bubbleMin + legendNorm*(bubbleMax-bubbleMin);
else
    legendSize = repmat(bubbleMin,size(legendSpeed));
end


axLeg=axes(...
    'Position',[0.84 0.24 0.15 0.40]);


hold(axLeg,'on');

axis(axLeg,'off');
xlim(axLeg,[0 1]);
ylim(axLeg,[0 1]);


ypos=linspace(0.90,0.43,length(legendSpeed));


for i=1:length(legendSpeed)

    scatter(axLeg,...
        0.3,...
        ypos(i),...
        legendSize(i),...
        'o',...
        'MarkerFaceColor',[0.88 0.88 0.88],...
        'MarkerEdgeColor',[0.68 0.68 0.68],...
        'LineWidth',0.35);


    text(axLeg,...
        0.6,...
        ypos(i),...
        sprintf('%g',legendSpeed(i)),...
        'HorizontalAlignment','left',...
        'VerticalAlignment','middle',...
        'FontSize',8,...
        'FontName','Arial');

end

text(ax,...
    0.70,...
    0.98,...
    sprintf(['RMSE (vs. CPU)\n' ...
        '中位数 %s\n' ...
        '平均值 %s\n' ...
        '最大值 %s\n' ...
        '\n' ...
        '加速比 (vs. CPU)\n' ...
        '中位数 %s\n' ...
        '平均值 %s\n' ...
        '最大值 %s'],...
        rmseMedianText,rmseMeanText,rmseMaxText,...
        speedupMedianText,speedupMeanText,speedupMaxText),...
    'Units','normalized',...
    'HorizontalAlignment','left',...
    'VerticalAlignment','top',...
    'FontName','Arial',...
    'FontSize',7,...
    'BackgroundColor','w',...
    'Margin',2.0,...
    'Interpreter','tex');



%% =========================
% 12. 输出
% =========================

outputPath = fullfile(config.output_dir,'operator_cpu_accuracy_speedup_bubble');
exportgraphics(fig,[outputPath '.png'],'Resolution',600);
savefig(fig,[outputPath '.fig']);
fprintf('Saved: %s\n',outputPath);

clear zoomCleanup;




%% =========================
% 13. colormap函数
% =========================

function map = coolwarm_centered_biased(N,clim,centerValue)

if nargin < 1 || isempty(N)
    N = 256;
end

v = linspace(clim(1),clim(2),N);
t = zeros(size(v));

if centerValue <= clim(1)
    t = 0.5 + 0.5*(v-clim(1))/(clim(2)-clim(1));
elseif centerValue >= clim(2)
    t = 0.5*(v-clim(1))/(clim(2)-clim(1));
else
    idxBlue = v <= centerValue;
    idxRed = v > centerValue;

    redGamma = 0.55;

    t(idxBlue) = 0.5*(v(idxBlue)-clim(1))/(centerValue-clim(1));
    t(idxRed) = 0.5 + 0.5*...
        ((v(idxRed)-centerValue)/(clim(2)-centerValue)).^redGamma;
end

t = max(0,min(1,t));
baseMap = coolwarm(256);
baseX = linspace(0,1,size(baseMap,1));
map = interp1(baseX,baseMap,t,'linear');

map = max(0,min(1,map));

end

function label = wrap_operator_label(name,maxChars)

name = char(name);

if strlength(string(name)) <= maxChars
    label = string(name);
    return;
end

underscorePos = strfind(name,'_');

if ~isempty(underscorePos)
    [~,idx] = min(abs(underscorePos - length(name)/2));
    pos = underscorePos(idx);
else
    pos = ceil(length(name)/2);
end

label = string({name(1:pos); name(pos+1:end)});

end

function txt = format_sci_tex(value,nSig)

if ~isfinite(value)
    txt = 'N/A';
    return;
end

if value == 0
    txt = '0';
    return;
end

exponent = floor(log10(abs(value)));
mantissa = value/(10^exponent);
fmt = sprintf('%%.%df×10^{%%d}',max(nSig-1,0));
txt = sprintf(fmt,mantissa,exponent);

end

function restore_desktop_zoom(zoomSetting,desktopZoom,hadPersonalZoom)

if hadPersonalZoom
    zoomSetting.PersonalValue = desktopZoom;
else
    clearPersonalValue(zoomSetting);
end

end
