clc;
close all;

%% =========================
% 0. 全局设置
% =========================

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLegendFontName','Arial');

set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');

config = plot_common();
config.data_dir = fullfile(rootDir,config.data_dir);
config.output_dir = fullfile(rootDir,config.output_dir);

if ~exist(config.output_dir,'dir')
    mkdir(config.output_dir);
end

%% =========================
% 1. 步骤顺序
% =========================

taskOrder = [
    "Task1"
    "Task1"
    "Task1"
    "Task1"
    "Task1"
    "Task2"
    "Task2"
    "Task2"
    "Task2"
    "Task2"
    "Task2"
];

stepOrder = [
    "step1"
    "step2"
    "step3"
    "step4"
    "step5"
    "step1"
    "step2"
    "step3"
    "step4"
    "step5"
    "step6"
];

stepLabel = [
    "T1-S1"
    "T1-S2"
    "T1-S3"
    "T1-S4"
    "T1-S5"
    "T2-S1"
    "T2-S2"
    "T2-S3"
    "T2-S4"
    "T2-S5"
    "T2-S6"
];

%% =========================
% 2. 数据读取
% =========================

csvPath = fullfile(config.data_dir,'task_stability_accuracy.csv');
matPath = fullfile(config.data_dir,'task_step_rmse_stability.mat');

if exist(matPath,'file')
    rmseSamples = read_rmse_samples_from_mat(matPath,numel(stepOrder));
elseif exist(csvPath,'file')
    rmseSamples = read_rmse_samples_from_csv(csvPath,taskOrder,stepOrder);
else
    rmseSamples = generate_placeholder_samples(config,taskOrder,stepOrder);
end

rmseSamples = double(rmseSamples);
rmseSamples(~isfinite(rmseSamples) | rmseSamples < 0) = NaN;

idxPositive = isfinite(rmseSamples) & rmseSamples > 0;
rmseFloor = min(rmseSamples(idxPositive),[],'all')/10;
rmseSamples(isfinite(rmseSamples) & rmseSamples == 0) = rmseFloor;

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
    'Position',[2 2 15 10],...
    'PaperUnits','centimeters',...
    'PaperPosition',[0 0 15 10],...
    'PaperSize',[15 10]);

ax = axes(fig);
hold(ax,'on');

colorMap = coolwarm(256);
boxColor = colorMap(52,:);

nStep = numel(stepOrder);
nRun = size(rmseSamples,2);
group = repmat((1:nStep)',1,nRun);

boxchart(ax,group(:),rmseSamples(:),...
    'BoxFaceColor',boxColor,...
    'BoxFaceAlpha',0.70,...
    'BoxEdgeColor',boxColor*0.70,...
    'WhiskerLineColor',boxColor*0.70,...
    'MarkerColor',boxColor*0.85,...
    'MarkerStyle','.',...
    'MarkerSize',2,...
    'LineWidth',0.65);

set(ax,...
    'XTick',1:nStep,...
    'XTickLabel',cellstr(stepLabel),...
    'YScale','log',...
    'FontName','Arial',...
    'FontSize',8,...
    'LineWidth',0.55,...
    'TickLabelInterpreter','none');

xlabel(ax,'信号处理任务步骤','FontSize',9,'FontName','Arial');
ylabel(ax,'RMSE','FontSize',9,'FontName','Arial');
title(ax,'ZQ500 算法稳定性分析',...
    'FontSize',11,...
    'FontWeight','bold',...
    'FontName','Arial');

xlim(ax,[0.35 nStep+0.65]);
ylim(ax,[rmseFloor/2 max(rmseSamples,[],'all')*4]);

grid(ax,'off');
box(ax,'on');

yl = ylim(ax);
plot(ax,[5.5 5.5],yl,...
    '--',...
    'Color',[0.48 0.48 0.48],...
    'LineWidth',0.7,...
    'HandleVisibility','off');

text(ax,3,yl(2)/1.35,'任务1',...
    'HorizontalAlignment','center',...
    'VerticalAlignment','top',...
    'FontSize',8,...
    'FontName','Arial',...
    'Color',[0.25 0.25 0.25]);

text(ax,8.5,yl(2)/1.35,'任务2',...
    'HorizontalAlignment','center',...
    'VerticalAlignment','top',...
    'FontSize',8,...
    'FontName','Arial',...
    'Color',[0.25 0.25 0.25]);

ax.Position = [0.10 0.18 0.82 0.70];

%% =========================
% 4. 输出
% =========================

outputPath = fullfile(config.output_dir,'task_step_rmse_stability_boxplot');
exportgraphics(fig,[outputPath '.png'],'Resolution',600);
savefig(fig,[outputPath '.fig']);
fprintf('Saved: %s\n',outputPath);

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
clear zoomCleanup;

function rmseSamples = read_rmse_samples_from_mat(matPath,nStep)

data = load(matPath);
fieldName = string(fieldnames(data));

for i = 1:numel(fieldName)
    value = data.(fieldName(i));
    if isnumeric(value) && ismatrix(value)
        if size(value,1) == nStep
            rmseSamples = value;
            return;
        elseif size(value,2) == nStep
            rmseSamples = value';
            return;
        end
    end
end

error('No numeric %d-by-N RMSE matrix found in %s.',nStep,matPath);

end

function rmseSamples = read_rmse_samples_from_csv(csvPath,taskOrder,stepOrder)

data = readtable(csvPath,'VariableNamingRule','preserve');
names = string(data.Properties.VariableNames);
lowerNames = lower(names);

idxTask = find(lowerNames == "task",1);
idxStep = find(lowerNames == "step",1);
idxRmse = find(lowerNames == "rmse",1);

if ~isempty(idxTask) && ~isempty(idxStep) && ~isempty(idxRmse)
    rmseCell = cell(numel(stepOrder),1);
    dataTask = string(data.(names(idxTask)));
    dataStep = normalize_step_name(string(data.(names(idxStep))));

    for i = 1:numel(stepOrder)
        idx = strcmpi(dataTask,taskOrder(i)) & ...
            strcmpi(dataStep,stepOrder(i));
        rmseCell{i} = data.(names(idxRmse))(idx);
    end

    nRun = max(cellfun(@numel,rmseCell));
    rmseSamples = NaN(numel(stepOrder),nRun);

    for i = 1:numel(stepOrder)
        values = rmseCell{i};
        rmseSamples(i,1:numel(values)) = values(:)';
    end

    return;
end

raw = readmatrix(csvPath);

if size(raw,1) == numel(stepOrder)
    rmseSamples = raw;
elseif size(raw,2) == numel(stepOrder)
    rmseSamples = raw';
else
    error('CSV must be long format with task/step/rmse columns or an 11-by-N RMSE matrix.');
end

end

function normalizedStep = normalize_step_name(rawStep)

normalizedStep = lower(rawStep);

for i = 1:numel(rawStep)
    token = regexp(char(rawStep(i)),'(?i)step\s*([0-9]+)','tokens','once');

    if ~isempty(token)
        normalizedStep(i) = "step" + string(token{1});
    end
end

end

function rmseSamples = generate_placeholder_samples(config,taskOrder,stepOrder)

accuracyPath = fullfile(config.data_dir,'task_step_accuracy.csv');
accuracyData = readtable(accuracyPath,...
    'VariableNamingRule','preserve');

nStep = numel(stepOrder);
nRun = 2000;
baseRmse = nan(nStep,1);

for i = 1:nStep
    idx = strcmpi(string(accuracyData.task),taskOrder(i)) & ...
        strcmpi(string(accuracyData.step),stepOrder(i));

    if any(idx)
        baseRmse(i) = accuracyData.rmse(find(idx,1,'first'));
    end
end

idxPositive = isfinite(baseRmse) & baseRmse > 0;
baseFloor = min(baseRmse(idxPositive))/10;
baseRmse(~isfinite(baseRmse) | baseRmse <= 0) = baseFloor;

rng(20260722,'twister');
sigma = 0.18 + 0.08*rand(nStep,1);
noise = exp(sigma.*randn(nStep,nRun));
drift = 1 + 0.035*sin(linspace(0,6*pi,nRun));
rmseSamples = baseRmse .* noise .* drift;

end

function restore_desktop_zoom(sObj,desktopZoom)

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;

end
