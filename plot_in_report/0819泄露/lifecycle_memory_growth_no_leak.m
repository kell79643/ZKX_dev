clc;
clear;
close all;


%% =========================
% 0. 全局设置
% =========================
% 字体与解释器：标签 interpreter 设为 none，避免下划线被识别为下标

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLegendFontName','Arial');

set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');

% 内联配置（不依赖外部 plot_common.m，保证脚本自包含）
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

config = struct();
config.data_dir   = scriptDir;
config.output_dir = fullfile(scriptDir,'output');

if ~exist(config.output_dir,'dir')
    mkdir(config.output_dir);
end


%% =========================
% 1. 数据读取
% =========================
% 数据来源：07_内存显存泄漏 下 task1 / task2 pipeline 的 PASS run 的
% raw/*_memory_stability.log 文件，每轮一条 [MEMORY][SAMPLE] key=value 行。
% 本次未运行 dlfft 后端，因此只读 fft_thrust 的两条 run（task1 + task2）。

logFiles = {
    fullfile(config.data_dir,'07_内存显存泄漏','task1_pipeline',...
        'RB_20260817T123624Z_STAGE11_TASK1_PIPELINE_FFT_THRUST_PASS',...
        'raw','task1_memory_stability.log'), 'task1';
    fullfile(config.data_dir,'07_内存显存泄漏','task2_pipeline',...
        'RB_20260817T094347Z_STAGE11_TASK2_PIPELINE_FFT_THRUST_PASS',...
        'raw','task2_memory_stability.log'), 'task2'
};

data = table();

for i = 1:size(logFiles,1)
    filePath    = logFiles{i,1};
    runIdValue  = logFiles{i,2};

    if ~exist(filePath,'file')
        error('Log file not found: %s',filePath);
    end

    % 读取全部行，仅保留 [MEMORY][SAMPLE] 行
    fileText   = fileread(filePath);
    allLines   = strtrim(splitlines(fileText));
    sampleMask = startsWith(allLines,'[MEMORY][SAMPLE]');
    sampleLines = allLines(sampleMask);

    nSamples = numel(sampleLines);

    iterations = zeros(nSamples,1);
    cpuRss      = zeros(nSamples,1);
    cpuHeap     = zeros(nSamples,1);
    gpuUsed     = zeros(nSamples,1);

    for j = 1:nSamples
        line         = sampleLines{j};
        iterations(j)= parse_kv_int(line,'iteration');
        cpuRss(j)    = parse_kv_int(line,'cpu_rss_bytes');
        cpuHeap(j)   = parse_kv_int(line,'cpu_live_heap_bytes');
        gpuUsed(j)   = parse_kv_int(line,'gpu_used_bytes');
    end

    % baseline = 该 run 的第一轮值；增长量 = current - baseline
    baselineRss  = cpuRss(1);
    baselineHeap = cpuHeap(1);
    baselineGpu  = gpuUsed(1);

    oneData = table();
    oneData.iteration = iterations;
    oneData.backend   = repmat("fft_thrust",nSamples,1);
    oneData.run_id    = repmat(string(runIdValue),nSamples,1);
    oneData.source_file = repmat(string(filePath),nSamples,1);

    oneData.cpu_rss_growth_from_baseline_bytes       = cpuRss  - baselineRss;
    oneData.cpu_live_heap_growth_from_baseline_bytes = cpuHeap - baselineHeap;
    oneData.gpu_growth_from_baseline_bytes           = gpuUsed - baselineGpu;

    data = [data; oneData]; %#ok<AGROW>
end

requiredColumns = [
    "iteration"
    "backend"
    "cpu_rss_growth_from_baseline_bytes"
    "cpu_live_heap_growth_from_baseline_bytes"
    "gpu_growth_from_baseline_bytes"
];

missingColumns = setdiff(requiredColumns,string(data.Properties.VariableNames));
if ~isempty(missingColumns)
    error('Missing required columns: %s',strjoin(missingColumns,', '));
end


%% =========================
% 2. 数据筛选与排序
% =========================
% 只保留 fft_thrust；按 backend、run_id、iteration 排序。
% 随后跨 task1/task2 在每个 iteration 上取 mean，得到一条 1024 点的 fft_thrust 曲线。

backendOrder = ["fft_thrust"];

idxBackend = ismember(string(data.backend),backendOrder);
data = data(idxBackend,:);

if isempty(data)
    error('No fft_thrust rows found in memory logs.');
end

data.backend = categorical(string(data.backend),backendOrder,'Ordinal',true);
data = sortrows(data,{'backend','run_id','iteration'});

[uniqueIters,~,iterGroups] = unique(data.iteration);
nIters = numel(uniqueIters);

dataAvg = table();
dataAvg.iteration = uniqueIters;
dataAvg.backend   = repmat("fft_thrust",nIters,1);
dataAvg.cpu_rss_growth_from_baseline_bytes = ...
    accumarray(iterGroups,data.cpu_rss_growth_from_baseline_bytes,[nIters 1],@mean);
dataAvg.cpu_live_heap_growth_from_baseline_bytes = ...
    accumarray(iterGroups,data.cpu_live_heap_growth_from_baseline_bytes,[nIters 1],@mean);
dataAvg.gpu_growth_from_baseline_bytes = ...
    accumarray(iterGroups,data.gpu_growth_from_baseline_bytes,[nIters 1],@mean);
dataAvg = sortrows(dataAvg,'iteration');

pointCountText = compose_point_count_text(data,dataAvg,backendOrder);

rssText  = compose_summary_text(dataAvg,...
    "cpu_rss_growth_from_baseline_bytes",backendOrder);
heapText = compose_summary_text(dataAvg,...
    "cpu_live_heap_growth_from_baseline_bytes",backendOrder);
gpuText  = compose_summary_text(dataAvg,...
    "gpu_growth_from_baseline_bytes",backendOrder);


%% =========================
% 3. 绘图
% =========================
% 左轴：CPU RSS（实线）+ CPU live heap（点线）
% 右轴：GPU（虚线）
% 三条曲线均使用 fft_thrust 的基色，靠 lighten 与线型区分。

sObj = settings;
desktopZoom = sObj.matlab.desktop.Zoom.PersonalValue;
sObj.matlab.desktop.Zoom.PersonalValue = 100;
zoomCleanup = onCleanup(@() restore_desktop_zoom(sObj,desktopZoom));

fig = figure(...
    'Color','w',...
    'Units','centimeters',...
    'Position',[2 2 15 9.5],...
    'PaperUnits','centimeters',...
    'PaperPosition',[0 0 15 9.5],...
    'PaperSize',[15 9.5]);

palette = struct();
palette.fft_thrust = [250 127 111] ./ 255 ./ 1.05;   % 低饱和珊瑚色
palette.rssLighten  = 0.00;   % 最深
palette.heapLighten = 0.30;   % 中等
palette.gpuLighten  = 0.55;   % 最浅

ax = axes(fig);
hold(ax,'on');

yyaxis(ax,'left');
cpuRssLine = plot_growth_axis(ax,dataAvg,...
    "cpu_rss_growth_from_baseline_bytes",...
    backendOrder,palette,palette.rssLighten,'-');
cpuHeapLine = plot_growth_axis(ax,dataAvg,...
    "cpu_live_heap_growth_from_baseline_bytes",...
    backendOrder,palette,palette.heapLighten,':');

ylabel(ax,'内存占用增长量 (MB)',...
    'FontSize',10,...
    'FontName','Arial');

ax.YColor = [0.18 0.18 0.18];
ylim(ax,[-0.02 0.12]);
yticks(ax,-0.02:0.02:0.12);

yyaxis(ax,'right');
gpuLine = plot_growth_axis(ax,dataAvg,...
    "gpu_growth_from_baseline_bytes",...
    backendOrder,palette,palette.gpuLighten,'--');

ylabel(ax,'显存占用增长量 (MB)',...
    'FontSize',10,...
    'FontName','Arial');

ax.YColor = [0.38 0.38 0.38];
ylim(ax,[-0.30 0.05]);
yticks(ax,-0.30:0.05:0.05);

yyaxis(ax,'left');

xlabel(ax,'循环次数',...
    'FontSize',10,...
    'FontName','Arial');

title(ax,...
    'fft_thrust 资源占用',...
    'FontSize',11,...
    'FontWeight','bold',...
    'FontName','Arial');

% 为 title / ylabel 留出间距，避免与刻度或曲线文字重叠
titleH = get(ax,'Title');
titleH.Units = 'normalized';
titleH.Position(2) = 1.08;   % 把标题往上推开，避免与顶边刻度贴在一起

grid(ax,'on');
box(ax,'on');

set(ax,...
    'FontName','Arial',...
    'FontSize',8,...
    'LineWidth',0.55,...
    'GridLineStyle','--',...
    'GridAlpha',0.45,...
    'GridColor',[0.82 0.82 0.82]);

xlim(ax,[0 max(dataAvg.iteration)]);
xticks(ax,0:200:1024);
% 主绘图区恢复正常宽度，右上角覆盖式图例不占用画布外部
ax.Position = [0.12 0.16 0.80 0.70];

draw_manual_legend(fig,palette);


%% =========================
% 4. 输出
% =========================

outputPath = fullfile(config.output_dir,...
    'lifecycle_memory_growth_no_leak');

exportgraphics(fig,[outputPath '.png'],'Resolution',600);
savefig(fig,[outputPath '.fig']);
fprintf('Saved: %s\n',outputPath);
fprintf('%s\n%s\n%s\n%s\n',pointCountText,rssText,heapText,gpuText);

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
clear zoomCleanup;


%% =========================
% 5. 局部函数
% =========================

function lineHandles = plot_growth_axis(ax,data,columnName,...
    backendOrder,palette,lightenAmount,lineStyle)
% 在当前 y 轴上，为指定后端绘制一条增长量曲线（单位转换为 MB）

lineHandles = gobjects(numel(backendOrder),1);

lineWidth = 1;

for i = 1:numel(backendOrder)
    backend = backendOrder(i);
    idx = string(data.backend) == backend;
    oneData = data(idx,:);

    if isempty(oneData)
        continue;
    end

    colorValue = lighten_color(palette.(char(backend)),lightenAmount);

    lineHandles(i) = plot(ax,...
        oneData.iteration,...
        oneData.(columnName) ./ 1e6,...
        'LineStyle',lineStyle,...
        'Color',colorValue,...
        'LineWidth',lineWidth);
end

end

function draw_manual_legend(fig,palette)
% 图例覆盖在主图右上角，不占用画布外部空间
% 主图 Position=[0.12 0.16 0.80 0.70]，右上角落在 (0.92, 0.86) 附近

axLegend = axes(fig,...
    'Units','normalized',...
    'Position',[0.61 0.66 0.30 0.20],...
    'Color','w',...
    'XColor','none',...
    'YColor','none',...
    'Box','on',...
    'LineWidth',0.5);

hold(axLegend,'on');
axis(axLegend,'off');
xlim(axLegend,[0 1]);
ylim(axLegend,[0 1]);

labels = {
    '内存 (RSS)';
    '堆内存';
    '显存'
};

colors = [
    lighten_color(palette.fft_thrust,0.00)
    lighten_color(palette.fft_thrust,0.30)
    lighten_color(palette.fft_thrust,0.55)
];

styles = {'-'; ':'; '--'};

xLineStart = 0.08;
xLineEnd   = 0.36;
xText      = 0.42;
yRows      = [0.78 0.50 0.22];

for r = 1:3
    plot(axLegend,...
        [xLineStart xLineEnd],...
        [yRows(r) yRows(r)],...
        'LineStyle',styles{r},...
        'Color',colors(r,:),...
        'LineWidth',1.6);

    text(axLegend,...
        xText,yRows(r),...
        labels{r},...
        'HorizontalAlignment','left',...
        'VerticalAlignment','middle',...
        'FontName','Arial',...
        'FontSize',8);
end

end

function summaryText = compose_summary_text(data,columnName,backendOrder)
% 汇总每个后端曲线的 final / max / min（单位 MB）

parts = strings(numel(backendOrder),1);

for i = 1:numel(backendOrder)
    backend = backendOrder(i);
    idx = string(data.backend) == backend;
    oneData = data(idx,:);

    if isempty(oneData)
        parts(i) = sprintf('%s: N/A',backend);
        continue;
    end

    [~,lastIdx] = max(oneData.iteration);
    lastValue = oneData.(columnName)(lastIdx) ./ 1e6;
    maxValue  = max(oneData.(columnName),[],'omitnan') ./ 1e6;
    minValue  = min(oneData.(columnName),[],'omitnan') ./ 1e6;

    parts(i) = sprintf('%s %s: final %.6g, max %.6g, min %.6g MB',...
        erase(columnName,"_growth_from_baseline_bytes"),...
        backend,lastValue,maxValue,minValue);
end

summaryText = strjoin(parts,newline);

end

function pointCountText = compose_point_count_text(data,dataAvg,backendOrder)
% 报告原始点数（task1+task2 合并）和平均后点数

parts = strings(numel(backendOrder),1);

for i = 1:numel(backendOrder)
    backend = backendOrder(i);
    idx = string(data.backend) == backend;
    oneData = data(idx,:);

    if isempty(oneData)
        parts(i) = sprintf('%s: 0 points',backend);
        continue;
    end

    parts(i) = sprintf('%s: %d raw points (task1+task2), iteration %d-%d; averaged to %d points',...
        backend,height(oneData),...
        min(oneData.iteration),max(oneData.iteration),...
        height(dataAvg));
end

pointCountText = strjoin(parts,newline);

end

function out = lighten_color(in,amount)
% 颜色按比例向白色靠拢，构造同色系不同明度

out = in + (1-in)*amount;
out = max(0,min(1,out));

end

function restore_desktop_zoom(sObj,desktopZoom)

try
    sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
catch
end

end

function value = parse_kv_int(line,key)
% 从 "key=value" 文本中解析整数值（用于 [MEMORY][SAMPLE] 行）

tokens = regexp(line,[key '=(\d+)'],'tokens');
if isempty(tokens)
    value = NaN;
else
    value = str2double(tokens{1}{1});
end

end
