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
script_dir = fileparts(mfilename('fullpath'));
config.data_dir = fullfile(script_dir,'04_算子结果','绘图数据');
config.output_dir = fullfile(script_dir,'output');
if ~exist(config.output_dir,'dir')
    mkdir(config.output_dir);
end

% 旧数据：cuSignal (Python GPU) 耗时，仅含FP32
cusignal_data_candidates = {
    fullfile(script_dir,'04_算子结果','绘图数据', ...
        'operator_multiscale_performance.csv')
    fullfile(script_dir,'..','..','交付','02_技术结果', ...
        '图表数据','正式数据','operator_multiscale_performance.csv')
    fullfile(script_dir,'..','..','..','plot_in_report_old','交付', ...
        '02_技术结果','图表数据','正式数据', ...
        'operator_multiscale_performance.csv')
};

cusignal_data_path = '';
for iCandidate = 1:numel(cusignal_data_candidates)
    candidate_path = char(cusignal_data_candidates{iCandidate});
    if isfile(candidate_path)
        cusignal_data_path = candidate_path;
        break;
    end
end

if isempty(cusignal_data_path)
    error('operator_cusignal_speedup_error:MissingCuSignalData', ...
        ['找不到 cuSignal 基准数据 operator_multiscale_performance.csv。' ...
         '请将该文件放入脚本目录下的 04_算子结果/绘图数据。']);
end

%% =========================
% 1. 数据读取
% =========================

% 新数据：精度+GPU性能（全dtype）
data_path = fullfile(config.data_dir,...
    '全算子_逐case绘图候选.csv');

raw_data = readtable(data_path,...
    'VariableNamingRule','preserve');

% 旧数据：cuSignal耗时
cusignal_data = readtable(cusignal_data_path,...
    'VariableNamingRule','preserve');



%% =========================
% 2. 合并数据
% =========================


% 精度数据：全部dtype，仅保留精度PASS且RMSE有效记录。

idxAccuracyValid = strcmpi(string(raw_data.accuracy_status),"pass") & ...
    isfinite(raw_data.rmse);

accuracy_table = table(...
    string(raw_data.operator_name(idxAccuracyValid)),...
    raw_data.rmse(idxAccuracyValid),...
    'VariableNames',...
    {'operator','error'});



% 性能数据：旧cuSignal耗时 vs 新GPU耗时，按算子+规模匹配。

% 旧cuSignal数据：仅保留Success且python_gpu_mean_ms有效
idxCusignal = strcmpi(string(cusignal_data.status),"Success") & ...
    isfinite(cusignal_data.python_gpu_mean_ms);

cusignalOperator = string(cusignal_data.operator(idxCusignal));
cusignalScale = string(cusignal_data.input_scale_order(idxCusignal));
cusignalTime = cusignal_data.python_gpu_mean_ms(idxCusignal);

% 标准化算子名（KalmanFilter -> kalman_filter）
cusignalOperator = lower(cusignalOperator);
cusignalOperator(cusignalOperator == "kalmanfilter") = "kalman_filter";

% 新GPU数据：FP32, PASS, gpu_mean_ms有效
idxNewFP32 = strcmpi(string(raw_data.dtype),"FP32") & ...
    strcmpi(string(raw_data.status),"pass") & ...
    isfinite(raw_data.gpu_mean_ms);

newOperator = string(raw_data.operator_name(idxNewFP32));
newScale = string(raw_data.order_of_magnitude(idxNewFP32));
newGpuTime = raw_data.gpu_mean_ms(idxNewFP32);

% 按算子+规模匹配，计算speedup = cusignal_time / gpu_time
matchedOperator = unique(cusignalOperator,'stable');
matchedSpeedup = nan(size(matchedOperator));

for i = 1:length(matchedOperator)
    idxOld = cusignalOperator == matchedOperator(i);
    oldScales = cusignalScale(idxOld);
    oldTimes = cusignalTime(idxOld);

    idxNew = newOperator == matchedOperator(i);
    newScales = newScale(idxNew);
    newTimes = newGpuTime(idxNew);

    caseSpeedups = [];
    for j = 1:length(oldScales)
        idxMatch = newScales == oldScales(j);
        if any(idxMatch)
            caseSpeedups(end+1,1) = oldTimes(j) / min(newTimes(idxMatch)); %#ok<SAGROW>
        end
    end

    if ~isempty(caseSpeedups)
        matchedSpeedup(i) = max(caseSpeedups);
    end
end

performance_table = table(...
    matchedOperator,...
    matchedSpeedup,...
    'VariableNames',...
    {'operator','speedup'});



% 左连接：
% 保证所有精度测试算子存在

data = outerjoin(...
    accuracy_table,...
    performance_table,...
    'Keys','operator',...
    'MergeKeys',true);



operator = string(data.operator);

errorValue = data.error;

speedup = data.speedup;

% 绘图层按算子汇总，避免 accuracy/performance 按 dtype 外连接后产生重复点。
operatorRaw = operator;
errorRaw = errorValue;
speedupRaw = speedup;

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
    "estimation - kalman_filter"
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
    elseif ismember(opKey,"kalman_filter")
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

% 右上角统计必须与实心气泡使用同一批有效加速比数据。
validSpeedup = speedup(idxSpeedup);
assert(~isempty(validSpeedup), ...
    'operator_cusignal_speedup_error:NoValidSpeedup', ...
    '没有可用于绘制气泡图的有效 cuSignal 加速比数据。');


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


speedupMean = mean(validSpeedup);
speedupMax = max(validSpeedup);

speedupMeanText = sprintf('%.3g',speedupMean);
speedupMaxText = sprintf('%.3g',speedupMax);

colorValue = nan(size(speedup));
colorValue(idxSpeedup) = log10(speedup(idxSpeedup));

colorCenter = log10(0.85);
colorLim = [
    min(colorValue(idxSpeedup))
    max(colorValue(idxSpeedup))
];
colorLim(1) = min(colorLim(1),colorCenter);
colorLim(2) = max(colorLim(2),colorCenter);

% 将蓝色低值区限制为颜色条长度的 22%（小于 25%）。
% 更低的离群值在 caxis 下限处使用最深蓝色饱和显示。
blueFractionMax = 0.22;
blueLowerLimit = colorCenter - ...
    blueFractionMax/(1-blueFractionMax)*(colorLim(2)-colorCenter);
colorLim(1) = max(colorLim(1),blueLowerLimit);



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
try
    ax.Toolbar.Visible = 'off';
    drawnow;
catch
end

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


%% 无speedup：无填充圆 + 中心 x

idxNoSpeedup = ~idxSpeedup;

scatter(ax,...
    x(idxNoSpeedup),...
    y(idxNoSpeedup),...
    bubbleSizeOne,...
    'o',...
    'MarkerFaceColor','none',...
    'MarkerEdgeColor',[0.62 0.62 0.62],...
    'LineWidth',0.50);

scatter(ax,...
    x(idxNoSpeedup),...
    y(idxNoSpeedup),...
    bubbleSizeOne*0.42,...
    'x',...
    'MarkerEdgeColor',[0.62 0.62 0.62],...
    'LineWidth',0.45);



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
cb.Position=[0.76 0.12 0.018 0.80];

cb.Label.String='加速比 (vs. cuSignal)';


cb.FontName='Arial';
cb.FontSize=8;
cb.Label.FontName='Arial';
cb.Label.FontSize=8;
cb.AxisLocation='out';

tickSpeed = [
    1
    10
    100
];

tickSpeed = tickSpeed(log10(tickSpeed) >= colorLim(1) & ...
    log10(tickSpeed) <= colorLim(2));

if isempty(tickSpeed)
    fallbackTick = log10(median(validSpeedup));
    fallbackTick = min(max(fallbackTick,colorLim(1)),colorLim(2));
    tickSpeed = 10^fallbackTick;
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
ylim(axLeg,[-0.25 1]);


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

scatter(axLeg,...
    0.3,...
    0.20,...
    bubbleSizeOne,...
    'o',...
    'MarkerFaceColor','none',...
    'MarkerEdgeColor',[0.62 0.62 0.62],...
    'LineWidth',0.50);

scatter(axLeg,...
    0.3,...
    0.20,...
    bubbleSizeOne*0.42,...
    'x',...
    'MarkerEdgeColor',[0.62 0.62 0.62],...
    'LineWidth',0.45);

text(axLeg,...
    0.6,...
    0.20,...
    'N/A',...
    'HorizontalAlignment','left',...
    'VerticalAlignment','middle',...
    'FontSize',8,...
    'FontName','Arial');

text(axLeg,...
    -0.1,...
    0.03,...
    sprintf('N/A: Python cuSignal\n部分算子在 ZQ500\n平台无法直接运行'),...
    'HorizontalAlignment','left',...
    'VerticalAlignment','top',...
    'FontSize',7,...
    'FontName','Arial');

text(ax,...
    0.94,...
    0.94,...
    sprintf(['加速比 (vs. cuSignal)\n' ...
        '平均值 %s\n' ...
        '最大值 %s'],...
        speedupMeanText,speedupMaxText),...
    'Units','normalized',...
    'HorizontalAlignment','right',...
    'VerticalAlignment','top',...
    'FontName','Arial',...
    'FontSize',7,...
    'BackgroundColor','w',...
    'Margin',2.0,...
    'Interpreter','tex');



%% =========================
% 12. 输出
% =========================

outputPath = fullfile(config.output_dir,'operator_accuracy_speedup_bubble_recomputed_stats');
exportgraphics(fig,[outputPath '.png'],'Resolution',600);
if isgraphics(fig,'figure')
    savefig(fig,[outputPath '.fig']);
else
    warning('PNG 已导出，但图窗已关闭，跳过 FIG 保存：%s.fig', outputPath);
end
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
blueFloor = 0.20;

if centerValue <= clim(1)
    t = 0.5 + 0.5*(v-clim(1))/(clim(2)-clim(1));
elseif centerValue >= clim(2)
    t = blueFloor + (0.5-blueFloor)*...
        (v-clim(1))/(clim(2)-clim(1));
else
    idxBlue = v <= centerValue;
    idxRed = v > centerValue;

    redGamma = 0.55;

    t(idxBlue) = blueFloor + (0.5-blueFloor)*...
        (v(idxBlue)-clim(1))/(centerValue-clim(1));
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
