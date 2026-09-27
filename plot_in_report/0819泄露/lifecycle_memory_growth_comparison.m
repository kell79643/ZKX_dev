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

memoryPattern = fullfile(config.data_dir,...
    'memory',...
    'lifecycle_run_1*.csv');

memoryFiles = dir(memoryPattern);

if isempty(memoryFiles)
    error('No lifecycle memory CSV files found: %s',memoryPattern);
end

data = table();

for i = 1:numel(memoryFiles)
    filePath = fullfile(memoryFiles(i).folder,memoryFiles(i).name);

    oneData = readtable(filePath,...
        'VariableNamingRule','preserve');

    oneData.source_file = repmat(string(memoryFiles(i).name),height(oneData),1);

    data = [data; oneData]; %#ok<AGROW>
end

requiredColumns = [
    "iteration"
    "backend"
    "cpu_after_free_growth_from_baseline_bytes"
    "gpu_after_free_growth_from_baseline_bytes"
];

missingColumns = setdiff(requiredColumns,string(data.Properties.VariableNames));
if ~isempty(missingColumns)
    error('Missing required columns: %s',strjoin(missingColumns,', '));
end


%% =========================
% 2. 数据筛选与排序
% =========================

backendOrder = [
    "dlfft"
    "fft_thrust"
];

idxBackend = ismember(string(data.backend),backendOrder);
data = data(idxBackend,:);

if isempty(data)
    error('No dlfft or fft_thrust rows found in: %s',memoryPattern);
end

data.backend = categorical(string(data.backend),backendOrder,'Ordinal',true);
data = sortrows(data,{'backend','run_id','iteration'});

pointCountText = compose_point_count_text(data,backendOrder);

cpuLastText = compose_summary_text(data,...
    "cpu_after_free_growth_from_baseline_bytes",...
    backendOrder);

gpuLastText = compose_summary_text(data,...
    "gpu_after_free_growth_from_baseline_bytes",...
    backendOrder);


%% =========================
% 3. 绘图
% =========================

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
mainColors = [130 176 210; 250 127 111] ./ 255 ./ 1.05;
palette.dlfft = mainColors(1,:);
palette.fft_thrust = mainColors(2,:);
palette.cpuLighten = 0.00;
palette.gpuLighten = 0.42;

ax = axes(fig);
hold(ax,'on');

yyaxis(ax,'left');
cpuLines = plot_growth_axis(ax,data,...
    "cpu_after_free_growth_from_baseline_bytes",...
    backendOrder,palette,palette.cpuLighten,'-');

ylabel(ax,'内存占用增长量 (MB)',...
    'FontSize',10,...
    'FontName','Arial');

ax.YColor = [0.18 0.18 0.18];
ylim(ax,[-0.05 0.70]);
yticks(ax,0 : 0.1 : 0.7);

yyaxis(ax,'right');
gpuLines = plot_growth_axis(ax,data,...
    "gpu_after_free_growth_from_baseline_bytes",...
    backendOrder,palette,palette.gpuLighten,'--');

ylabel(ax,'显存占用增长量 (MB)',...
    'FontSize',10,...
    'FontName','Arial');

ax.YColor = [0.38 0.38 0.38];
ylim(ax,[-0.01 0.30]);
yticks(ax,0:0.05:0.30);

yyaxis(ax,'left');

xlabel(ax,'循环次数',...
    'FontSize',10,...
    'FontName','Arial');

title(ax,...
    'FFT 资源占用',...
    'FontSize',11,...
    'FontWeight','bold',...
    'FontName','Arial');

grid(ax,'on');
box(ax,'on');

set(ax,...
    'FontName','Arial',...
    'FontSize',8,...
    'LineWidth',0.55,...
    'GridLineStyle','--',...
    'GridAlpha',0.45,...
    'GridColor',[0.82 0.82 0.82]);

xlim(ax,[0 max(data.iteration)]);
xticks(ax,0:400:2000);
ax.Position = [0.12 0.15 0.72 0.74];

draw_manual_legend(fig,palette);


%% =========================
% 4. 输出
% =========================

outputPath = fullfile(config.output_dir,...
    'lifecycle_after_free_growth_comparison');

exportgraphics(fig,[outputPath '.png'],'Resolution',600);
savefig(fig,[outputPath '.fig']);
fprintf('Saved: %s\n',outputPath);
fprintf('%s\n%s\n%s\n',pointCountText,cpuLastText,gpuLastText);

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
clear zoomCleanup;


%% =========================
% 5. 局部函数
% =========================

function lineHandles = plot_growth_axis(ax,data,columnName,...
    backendOrder,palette,lightenAmount,lineStyle)

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

axLegend = axes(fig,...
    'Position',[0.155 0.765 0.38 0.105],...
    'Color','none',...
    'XColor','none',...
    'YColor','none');

hold(axLegend,'on');
axis(axLegend,'off');
xlim(axLegend,[0 1]);
ylim(axLegend,[0 1]);

labels = {
    '内存 dlfft', '内存 fft-thrust';
    '显存 dlfft', '显存 fft-thrust'
};

colors = [
    lighten_color(palette.dlfft,palette.cpuLighten)
    lighten_color(palette.fft_thrust,palette.cpuLighten)
    lighten_color(palette.dlfft,palette.gpuLighten)
    lighten_color(palette.fft_thrust,palette.gpuLighten)
];

styles = {
    '-', '-';
    '--', '--'
};

xLineStart = [0.02 0.52];
xLineEnd = [0.17 0.67];
xText = [0.20 0.70];
yRow = [0.68 0.28];

idx = 0;
for r = 1:2
    for c = 1:2
        idx = idx + 1;

        plot(axLegend,...
            [xLineStart(c) xLineEnd(c)],...
            [yRow(r) yRow(r)],...
            'LineStyle',styles{r,c},...
            'Color',colors(idx,:),...
            'LineWidth',1.55);

        text(axLegend,...
            xText(c),...
            yRow(r),...
            labels{r,c},...
            'HorizontalAlignment','left',...
            'VerticalAlignment','middle',...
            'FontName','Arial',...
            'FontSize',8);
    end
end

end

function summaryText = compose_summary_text(data,columnName,backendOrder)

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
    maxValue = max(oneData.(columnName),[],'omitnan') ./ 1e6;

    parts(i) = sprintf('%s %s: final %.6g, max %.6g MB',...
        erase(columnName,"_after_free_growth_from_baseline_bytes"),...
        backend,lastValue,maxValue);
end

summaryText = strjoin(parts,newline);

end

function pointCountText = compose_point_count_text(data,backendOrder)

parts = strings(numel(backendOrder),1);

for i = 1:numel(backendOrder)
    backend = backendOrder(i);
    idx = string(data.backend) == backend;
    oneData = data(idx,:);

    if isempty(oneData)
        parts(i) = sprintf('%s: 0 points',backend);
        continue;
    end

    parts(i) = sprintf('%s: %d points, iteration %d-%d',...
        backend,height(oneData),min(oneData.iteration),max(oneData.iteration));
end

pointCountText = strjoin(parts,newline);

end

function out = lighten_color(in,amount)

out = in + (1-in)*amount;
out = max(0,min(1,out));

end

function restore_desktop_zoom(sObj,desktopZoom)

try
    sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
catch
end

end
