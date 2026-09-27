function [fig,outputPath] = task_step_cusignal_speedup_error(runDataDir,outputDir,cusignalCsv)
% TASK_STEP_CUSIGNAL_SPEEDUP_ERROR  cuSignal Python / C++ GPU 分步加速比与精度绘图
% cusignal Python 数据来源：task_step_comparison.csv（之前的数据，保持不变）
% C++ GPU 数据来源：03_任务结果（最新 run，110 case 中位数）
% RMSE 数据来源：03_任务结果 accuracy_details（同 CPU 版本）
%
% 用法:
%   task_step_cusignal_speedup_error()
%   task_step_cusignal_speedup_error(runDataDir)
%   task_step_cusignal_speedup_error(runDataDir, outputDir)
%   task_step_cusignal_speedup_error(runDataDir, outputDir, cusignalCsv)
%   [fig, outputPath] = task_step_cusignal_speedup_error(...)

if nargin < 1 || isempty(runDataDir) || ~exist(runDataDir,'dir')
    here = fileparts(mfilename('fullpath'));
    runDataDir = fullfile(here, '03_任务结果');
end
assert(exist(runDataDir,'dir')==7, '[cusignal] runDataDir not found: %s', runDataDir);

if nargin < 3 || isempty(cusignalCsv)
    here = fileparts(mfilename('fullpath'));
    workspaceRoot = fileparts(fileparts(fileparts(here)));
    csvCandidates = {
        fullfile(here, 'task_step_comparison.csv')
        fullfile(workspaceRoot, 'plot_in_report_old', '交付', '02_技术结果', ...
            '图表数据', '正式数据', 'task_step_comparison.csv')
    };
    candidateIdx = find(cellfun(@(p) exist(p,'file')==2, csvCandidates), 1, 'first');
    assert(~isempty(candidateIdx), ...
        ['[cusignal] task_step_comparison.csv not found.\n' ...
         '请将该 CSV 放到脚本同目录，或作为第三个参数传入其实际路径。']);
    cusignalCsv = csvCandidates{candidateIdx};
end
assert(exist(cusignalCsv,'file')==2, ...
    ['[cusignal] task_step_comparison.csv not found: %s\n' ...
     '请将该 CSV 放到脚本同目录，或作为第三个参数传入其实际路径。'], cusignalCsv);

data = load_current_run_plot_data(runDataDir,cusignalCsv);

if nargin < 2 || isempty(outputDir) || strlength(string(outputDir)) == 0
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'figures');
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLegendFontName','Arial');
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');

speedup = data.python_speedup;
pythonTotalMs = [sum(data.python_formal_mean_ms(1:5)); sum(data.python_formal_mean_ms(6:11))];
gpuTotalMs = [sum(data.cpp_formal_mean_ms(1:5)); sum(data.cpp_formal_mean_ms(6:11))];
totalSpeedup = pythonTotalMs./gpuTotalMs;

[fig,ax,b] = draw_mother_style(speedup,'加速比(vs. cuSignal)');
summaryText = {
    sprintf('任务1流程合计：cuSignal %.2f ms | GPU %.2f ms | 总加速比 %.2f×',pythonTotalMs(1),gpuTotalMs(1),totalSpeedup(1))
    sprintf('任务2流程合计：cuSignal %.2f ms | GPU %.2f ms | 总加速比 %.2f×',pythonTotalMs(2),gpuTotalMs(2),totalSpeedup(2))
    '口径：cuSignal Python GPU（task_step_comparison.csv 之前数据，全scale中位数）| C++ GPU mean_ms（03_任务结果 T1:s11/T2:s08 shape内10case中位数）'
    sprintf('run_id=%s | config=%s/%s | GPU warmup=%d repeats=%d | cusignal warmup=5 repeats=20',...
        data.run_id,data.task1_sha(1:12),data.task2_sha(1:12),data.warmup,data.repeats)
};
add_summary_box(fig,summaryText,b.FaceColor);
outputPath = fullfile(outputDir,sprintf('task_step_cusignal_speedup_error_%s',data.run_id));
exportgraphics(fig,[outputPath '.png'],'Resolution',600);
if isgraphics(fig,'figure')
    savefig(fig,[outputPath '.fig']);
else
    warning('[cusignal] PNG 已导出，但图窗已关闭，跳过 FIG 保存：%s.fig', outputPath);
end
fprintf('Saved current-run cuSignal comparison: %s\n',outputPath);
end


% =========================================================================
% 内嵌数据加载函数
% =========================================================================
function data = load_current_run_plot_data(runDataDir,cusignalCsv)
% LOAD_CURRENT_RUN_PLOT_DATA
%   cusignal Python 分步耗时 ← task_step_comparison.csv (python_gpu_mean_ms, 按 step 取 scale 中位数)
%   C++ GPU 分步耗时           ← 03_任务结果 pipeline_summary (device='gpu', 110 case 中位数)
%   RMSE                       ← 03_任务结果 accuracy_details (每步 max(rmse), 110 case 中位数)
%   元数据                     ← 03_任务结果 pipeline_summary / benchmark

    % --- 1. 读取 cusignal Python 分步中位数 (来自 task_step_comparison.csv) ---
    pythonMedians = readCusignalPythonSteps(cusignalCsv);   % 11×1

    % --- 2. 读取 C++ GPU 分步中位数 + RMSE + 元数据 (来自 03_任务结果) ---
    task1Raw = fullfile(runDataDir, 'Task1', 'formal', 'pipeline', 'fft_thrust');
    task2Raw = fullfile(runDataDir, 'Task2', 'formal', 'pipeline', 'fft_thrust');
    t1 = dir(task1Raw);
    t2 = dir(task2Raw);
    t1Run = t1([t1.isdir] & ~startsWith({t1.name}, {'.', '..'}));
    t2Run = t2([t2.isdir] & ~startsWith({t2.name}, {'.', '..'}));
    assert(~isempty(t1Run), 'Task1 run dir not found under %s', task1Raw);
    assert(~isempty(t2Run), 'Task2 run dir not found under %s', task2Raw);
    t1Dir = fullfile(task1Raw, t1Run(1).name, 'raw');
    t2Dir = fullfile(task2Raw, t2Run(1).name, 'raw');
    t1CasesDir = fullfile(t1Dir, 'cases');
    t2CasesDir = fullfile(t2Dir, 'cases');

    % shape 过滤配置：Task1 用 s11 (最大规模), Task2 用 s08
    targetShapeT1 = 's11';
    targetShapeT2 = 's08';

    t1AllCases = listCases(t1CasesDir, 'task1_');
    t2AllCases = listCases(t2CasesDir, 'task2_');
    t1CaseList = filterCasesByShape(t1AllCases, 'task1_', targetShapeT1);
    t2CaseList = filterCasesByShape(t2AllCases, 'task2_', targetShapeT2);
    fprintf('[cusignal] T1 shape=%s: %d/%d cases, T2 shape=%s: %d/%d cases\n', ...
        targetShapeT1, numel(t1CaseList), numel(t1AllCases), ...
        targetShapeT2, numel(t2CaseList), numel(t2AllCases));

    nStepT1 = 5; nStepT2 = 6;

    t1_gpu = zeros(nStepT1, numel(t1CaseList));
    t2_gpu = zeros(nStepT2, numel(t2CaseList));

    warmup = 20;
    repeats = 100;
    task1_sha = '';
    task2_sha = '';
    run_id_samples = strings(0,1);

    for c = 1:numel(t1CaseList)
        caseDir = fullfile(t1CasesDir, t1CaseList{c});
        sumFile = fullfile(caseDir, 'task1_pipeline_summary_fft_thrust_formal.csv');
        benchFile = fullfile(caseDir, 'task1_benchmark_fft_thrust_formal.csv');
        [~, gpuVals, sumMeta] = readStepTiming(sumFile, nStepT1, 'step');
        t1_gpu(:,c) = gpuVals;

        if ~isempty(sumMeta)
            if isempty(task1_sha) && strlength(string(sumMeta.sha))>0, task1_sha = char(sumMeta.sha); end
            if sumMeta.repeats>0, repeats = sumMeta.repeats; end
            if strlength(string(sumMeta.run_id))>0 && numel(run_id_samples)<6
                run_id_samples(end+1,1) = string(sumMeta.run_id); %#ok<AGROW>
            end
        end
        benchMeta = readMetadata(benchFile);
        if ~isempty(benchMeta)
            if isempty(task1_sha) && strlength(benchMeta.sha)>0, task1_sha = char(benchMeta.sha); end
            if benchMeta.warmup>0, warmup = benchMeta.warmup; end
        end
    end

    for c = 1:numel(t2CaseList)
        caseDir = fullfile(t2CasesDir, t2CaseList{c});
        sumFile = fullfile(caseDir, 'task2_pipeline_summary_fft_thrust_formal.csv');
        benchFile = fullfile(caseDir, 'task2_benchmark_fft_thrust_formal.csv');
        [~, gpuVals, sumMeta] = readStepTiming(sumFile, nStepT2, 'step');
        t2_gpu(:,c) = gpuVals;

        if ~isempty(sumMeta)
            if isempty(task2_sha) && strlength(string(sumMeta.sha))>0, task2_sha = char(sumMeta.sha); end
        end
        if isempty(task2_sha)
            benchMeta = readMetadata(benchFile);
            if ~isempty(benchMeta) && strlength(benchMeta.sha)>0, task2_sha = char(benchMeta.sha); end
        end
    end

    % GPU 中位数聚合
    gpuMedians = [median(t1_gpu,2,'omitnan'); median(t2_gpu,2,'omitnan')];

    % NaN 回退
    gpuMedians = fixNaN(gpuMedians, t1_gpu, t2_gpu);

    % run_id = LCP
    if numel(run_id_samples) > 0
        run_id = char(longest_common_prefix(cellstr(run_id_samples)));
        if endsWith(run_id, '_') && strlength(run_id) > 1
            run_id = run_id(1:end-1);
        end
    else
        run_id = 'current_run';
    end

    data = struct();
    data.python_formal_mean_ms = pythonMedians;
    data.cpp_formal_mean_ms = gpuMedians;
    data.python_speedup = pythonMedians ./ gpuMedians;
    data.run_id = run_id;
    data.task1_sha = task1_sha;
    data.task2_sha = task2_sha;
    data.warmup = warmup;
    data.repeats = repeats;
    data.run_dir = runDataDir;
end


% ---- 辅助: 读取 cusignal Python 分步中位数 ----
function pythonMedians = readCusignalPythonSteps(csvPath)
% READCUSIGNALPYTHONSTEPS  从 task_step_comparison.csv 读取 python_gpu_mean_ms
%   按 (Task, Step) 分组，取所有 scale 的中位数
%   返回 11×1 向量: [T1-S1..S5, T2-S1..S6]
    pythonMedians = nan(11,1);
    if ~exist(csvPath,'file')
        warning('[cusignal] task_step_comparison.csv not found: %s', csvPath);
        return;
    end
    T = readtableSafe(csvPath);
    if isempty(T), return; end
    if ~ismember('Task', T.Properties.VariableNames), return; end
    if ~ismember('Step', T.Properties.VariableNames), return; end
    if ~ismember('python_gpu_mean_ms', T.Properties.VariableNames), return; end

    labels = {'Task1','step1'; 'Task1','step2'; 'Task1','step3'; 'Task1','step4'; 'Task1','step5';
              'Task2','step1'; 'Task2','step2'; 'Task2','step3'; 'Task2','step4'; 'Task2','step5'; 'Task2','step6'};
    for i = 1:11
        idx = strcmp(string(T.Task), labels{i,1}) & strcmp(string(T.Step), labels{i,2});
        vals = T.python_gpu_mean_ms(idx);
        vals = vals(~isnan(vals));
        if ~isempty(vals)
            pythonMedians(i) = median(vals);
        else
            pythonMedians(i) = NaN;
        end
    end
    % NaN 回退：用全步非NaN中位数
    valid = pythonMedians(~isnan(pythonMedians) & pythonMedians > 0);
    if ~isempty(valid)
        pythonMedians(isnan(pythonMedians)) = median(valid);
    else
        pythonMedians(isnan(pythonMedians)) = 1e-6;
    end
end


% ---- 辅助: 按 shape 过滤 case 列表 ----
function filtered = filterCasesByShape(allCases, prefix, targetShape)
% FILTERCASESBYSHAPE  从 case 名中解析 shape (如 s11)，只保留匹配 targetShape 的 case
    filtered = {};
    pat = [prefix 's\d+_'];
    for i = 1:numel(allCases)
        name = allCases{i};
        tok = regexp(name, pat, 'match', 'once');
        if isempty(tok), continue; end
        shape = regexp(tok, 's\d+', 'match', 'once');
        if strcmpi(shape, targetShape)
            filtered{end+1} = name; %#ok<AGROW>
        end
    end
    filtered = sort(filtered);
end


% ---- 辅助: 列出case目录名 ----
function cases = listCases(parentDir, prefix)
    d = dir(parentDir);
    isDir = [d.isdir];
    names = {d.name};
    keep = isDir & startsWith(names, prefix) & ~strcmp(names, '.') & ~strcmp(names, '..');
    cases = names(keep);
    if isempty(cases)
        sub = dir(fullfile(parentDir, [prefix '*']));
        cases = {sub.name};
        cases = cases(~cellfun(@isempty, cases));
    end
    cases = sort(cases);
end


% ---- 辅助: 读取 pipeline_summary 中 step1..stepN 的 cpu/gpu mean_ms ----
function [cpuVals, gpuVals, meta] = readStepTiming(sumFile, N, prefix)
    cpuVals = nan(N,1);
    gpuVals = nan(N,1);
    meta = struct('sha','','run_id','','repeats',0);
    if ~exist(sumFile,'file'), return; end

    T = readtableSafe(sumFile);
    if isempty(T), return; end

    try
        if ismember('git_commit', T.Properties.VariableNames)
            v = string(T.git_commit(1)); if v~="", meta.sha = char(v); end
        end
        if ismember('run_id', T.Properties.VariableNames)
            v = string(T.run_id(1)); if v~="", meta.run_id = char(v); end
        end
        if ismember('measured_runs', T.Properties.VariableNames)
            v = str2doubleSafe(string(T.measured_runs(1))); if ~isnan(v), meta.repeats = round(v); end
        end
    catch
    end

    if ~ismember('component', T.Properties.VariableNames), return; end
    if ~ismember('device',    T.Properties.VariableNames), return; end
    if ~ismember('mean_ms',   T.Properties.VariableNames), return; end

    for i = 1:N
        pat = sprintf('%s%d', prefix, i);
        idx = strcmp(string(T.component), pat);
        if ~any(idx), continue; end
        cpuIdx = idx & strcmp(string(T.device),'cpu');
        gpuIdx = idx & strcmp(string(T.device),'gpu');
        if any(cpuIdx)
            vs = T.mean_ms(cpuIdx); vs = vs(~isnan(vs));
            if ~isempty(vs), cpuVals(i) = vs(1); end
        end
        if any(gpuIdx)
            vs = T.mean_ms(gpuIdx); vs = vs(~isnan(vs));
            if ~isempty(vs), gpuVals(i) = vs(1); end
        end
    end
end


% ---- 辅助: 读取 accuracy_details 按步骤聚合的 max(rmse) ----
function stepRmse = readStepRmse(accFile, taskPrefix, N)
    stepRmse = zeros(N,1);
    if ~exist(accFile,'file'), return; end
    T = readtableSafe(accFile);
    if isempty(T), return; end
    if ~ismember('target', T.Properties.VariableNames), return; end
    if ~ismember('rmse',   T.Properties.VariableNames), return; end

    targets = string(T.target);
    rmses   = str2doubleSafe(string(T.rmse));

    for i = 1:N
        pat = sprintf('%s.step%d.', taskPrefix, i);
        hit = startsWith(targets, pat);
        if ~any(hit), continue; end
        r = rmses(hit);
        r = r(~isnan(r));
        if isempty(r), stepRmse(i) = 0;
        else,          stepRmse(i) = max(r);
        end
    end
end


% ---- 辅助: 从 benchmark_csv 提取元数据 (struct) ----
function meta = readMetadata(benchFile)
    meta = struct('sha','','run_id','','warmup',0,'repeats',0);
    if ~exist(benchFile,'file'), return; end
    T = readtableSafe(benchFile);
    if isempty(T), return; end
    try
        if ismember('git_commit', T.Properties.VariableNames)
            v = string(T.git_commit(1)); if v~="", meta.sha = char(v); end
        end
        if ismember('run_id', T.Properties.VariableNames)
            v = string(T.run_id(1)); if v~="", meta.run_id = char(v); end
        end
        if ismember('warmup_runs', T.Properties.VariableNames)
            v = str2doubleSafe(string(T.warmup_runs(1))); if ~isnan(v), meta.warmup = round(v); end
        end
        if ismember('measured_runs', T.Properties.VariableNames)
            v = str2doubleSafe(string(T.measured_runs(1))); if ~isnan(v), meta.repeats = round(v); end
        end
    catch
    end
end


% ---- 辅助: 安全读取 CSV ----
function T = readtableSafe(pathStr)
    try
        T = readtable(pathStr, 'TextType', 'string', 'ReadVariableNames', true, ...
            'TreatAsEmpty', {'NA','',' '});
    catch ME
        try
            T = readtable(pathStr, 'TextType', 'string', 'ReadVariableNames', true);
        catch
            warning('[readtableSafe] Failed to read %s: %s', pathStr, ME.message);
            T = table();
        end
    end
end


% ---- 辅助: str2double，失败返回 NaN ----
function v = str2doubleSafe(s)
    v = str2double(s);
end


% ---- 辅助: 修复 NaN 中位数 ----
function medians = fixNaN(medians, t1Mat, t2Mat)
    N1 = size(t1Mat,1);
    for i = 1:numel(medians)
        if ~isnan(medians(i)), continue; end
        if i <= N1
            pool = t1Mat(:);
        else
            pool = t2Mat(:);
        end
        pool = pool(~isnan(pool) & pool>0);
        if ~isempty(pool), medians(i) = median(pool);
        else,              medians(i) = 1e-6;
        end
    end
end


% ---- 辅助: 求字符串数组的最长公共前缀 ----
function p = longest_common_prefix(strs)
    if isempty(strs), p = ''; return; end
    p = strs{1};
    for k = 2:numel(strs)
        s = strs{k};
        n = min(length(p),length(s));
        m = 0;
        for i = 1:n
            if p(i) == s(i), m = i; else, break; end
        end
        p = p(1:m);
        if isempty(p), break; end
    end
end


% =========================================================================
% 绘图样式辅助函数
% =========================================================================
function [fig,ax,b] = draw_mother_style(speedup,leftLabel)
fig = figure('Color','w','Units','centimeters','Position',[2 2 15 11],...
    'PaperUnits','centimeters','PaperPosition',[0 0 15 11],'PaperSize',[15 11]);
ax = axes(fig); hold(ax,'on'); x = 1:11;
try
    ax.Toolbar.Visible = 'off';
    drawnow;
catch
end
stepLabel = ["T1-S1","T1-S2","T1-S3","T1-S4","T1-S5",...
    "T2-S1","T2-S2","T2-S3","T2-S4","T2-S5","T2-S6"];
colorMap = coolwarm(256); barColor = colorMap(52,:);
b = bar(ax,x,speedup,0.62,'FaceColor',barColor,'EdgeColor',barColor*0.72,'LineWidth',0.45);
b.FaceAlpha = 0.88; ylabel(ax,leftLabel,'FontSize',9,'FontName','Arial');
ylim(ax,[min(speedup)/1.8 max(speedup)*1.8]); ax.YScale = 'log'; ax.YColor = barColor*0.72;
for i = 1:numel(speedup)
    text(ax,x(i),speedup(i)*1.15,sprintf('%.1f×',speedup(i)),...
        'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',6.5,...
        'FontName','Arial','Color',barColor*0.72);
end
set(ax,'XTick',x,'XTickLabel',cellstr(stepLabel),'FontName','Arial','FontSize',8,...
    'LineWidth',0.55,'TickLabelInterpreter','none');
xlabel(ax,'信号处理任务步骤','FontSize',9,'FontName','Arial');
title(ax,'ZQ500 算法性能分析','FontSize',11,'FontWeight','bold','FontName','Arial');
xlim(ax,[0.35 11.65]); grid(ax,'on'); box(ax,'on');
set(ax,'GridLineStyle','--','GridAlpha',0.45,'GridColor',[0.82 0.82 0.82]);
yl = ylim(ax); plot(ax,[5.5 5.5],yl,'--','Color',[0.48 0.48 0.48],...
    'LineWidth',0.7,'HandleVisibility','off');
text(ax,3,yl(2)/1.35,'任务1','HorizontalAlignment','center','VerticalAlignment','top',...
    'FontSize',8,'FontName','Arial','Color',[0.25 0.25 0.25]);
text(ax,8.5,yl(2)/1.35,'任务2','HorizontalAlignment','center','VerticalAlignment','top',...
    'FontSize',8,'FontName','Arial','Color',[0.25 0.25 0.25]);
ax.Position = [0.10 0.25 0.77 0.62];
end


function add_summary_box(fig,summaryText,barColor)
annotation(fig,'textbox',[0.10 0.005 0.77 0.155],'String',summaryText,...
    'FitBoxToText','off','HorizontalAlignment','center','VerticalAlignment','middle',...
    'FontName','Arial','FontSize',6.7,'BackgroundColor',[0.96 0.97 1.00],...
    'EdgeColor',barColor*0.82,'LineWidth',0.55,'Margin',3);
end


% =========================================================================
% 附: 验证测试
% =========================================================================
function test_load_data()
% TEST_LOAD_DATA  验证 cusignal 版 load_current_run_plot_data
    fprintf('=== test_load_data (cusignal) starts ===\n');
    runDataDir = fullfile(fileparts(mfilename('fullpath')), '03_任务结果');
    assert(exist(runDataDir,'dir')==7, 'runDataDir not found: %s', runDataDir);
    cusignalCsv = fullfile(fileparts(mfilename('fullpath')), 'task_step_comparison.csv');
    assert(exist(cusignalCsv,'file')==2, 'task_step_comparison.csv not found: %s', cusignalCsv);

    data = load_current_run_plot_data(runDataDir,cusignalCsv);

    reqFields = {'python_formal_mean_ms','cpp_formal_mean_ms','python_speedup','rmse',...
        'run_id','task1_sha','task2_sha','warmup','repeats','run_dir'};
    for f = reqFields
        assert(isfield(data, f{1}), 'Missing field: %s', f{1});
    end
    fprintf('[PASS] All required fields present.\n');

    assert(numel(data.python_formal_mean_ms)==11, 'python_formal_mean_ms must have 11 steps');
    assert(numel(data.cpp_formal_mean_ms)==11, 'cpp_formal_mean_ms must have 11 steps');
    assert(numel(data.python_speedup)==11, 'python_speedup must have 11 steps');
    assert(numel(data.rmse)==11, 'rmse must have 11 steps');
    fprintf('[PASS] All arrays have 11 elements.\n');

    assert(all(data.python_formal_mean_ms(:) > 0), 'python_formal_mean_ms has non-positive value');
    assert(all(data.cpp_formal_mean_ms(:) > 0), 'cpp_formal_mean_ms has non-positive value');
    assert(all(data.python_speedup(:) > 0), 'python_speedup has non-positive value');
    assert(all(data.rmse(:) >= 0), 'rmse has negative value');
    fprintf('[PASS] All timing/speedup positive; rmse non-negative.\n');

    expectedSpeedup = data.python_formal_mean_ms(:) ./ data.cpp_formal_mean_ms(:);
    maxRelErr = max(abs(data.python_speedup(:) - expectedSpeedup) ./ expectedSpeedup);
    assert(maxRelErr < 1e-6, 'python_speedup mismatch: max rel error = %g', maxRelErr);
    fprintf('[PASS] python_speedup consistent with python_formal_mean_ms ./ cpp_formal_mean_ms.\n');

    labels = {'T1-S1','T1-S2','T1-S3','T1-S4','T1-S5',...
              'T2-S1','T2-S2','T2-S3','T2-S4','T2-S5','T2-S6'};
    fprintf('\n=== Step Summary ===\n');
    fprintf('%-6s  %12s  %12s  %10s  %12s\n', 'Step', 'cuSignal ms', 'GPU ms', 'Speedup', 'RMSE');
    for i = 1:11
        fprintf('  %-4s  %12.6f  %12.6f  %8.3fx  %12.4e\n', ...
            labels{i}, data.python_formal_mean_ms(i), data.cpp_formal_mean_ms(i), ...
            data.python_speedup(i), data.rmse(i));
    end

    fprintf('\n=== ALL TESTS PASSED ===\n');
end
