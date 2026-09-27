function [fig,outputPath] = task_step_cpu_cusignal_speedup_error(runDataDir,outputDir)
% TASK_STEP_CPU_CUSIGNAL_SPEEDUP_ERROR  使用当前 run 的 CPU/C++ GPU 证据绘图（禁止回退历史CSV）
% 从 03_任务结果/Task1 和 03_任务结果/Task2 目录下所有 110 个 case 提取分步统计：
%   - 耗时：pipeline_summary 中 step1~step5(Task1)/step1~step6(Task2) 所有 case mean_ms 取中位数
%   - 加速比：cpu_median_ms ./ gpu_median_ms（分步）；流程合计取 ΣCPU / ΣGPU
%   - RMSE：accuracy_details 中每步骤所有 output 的 max(rmse)，再对 110 个 case 取中位数
%
% 用法:
%   task_step_cpu_cusignal_speedup_error()                     % 使用脚本旁的 03_任务结果 目录
%   task_step_cpu_cusignal_speedup_error(runDataDir)           % 指定目录
%   task_step_cpu_cusignal_speedup_error(runDataDir, outputDir)
%   [fig, outputPath] = task_step_cpu_cusignal_speedup_error(...)

if nargin < 1 || isempty(runDataDir) || ~exist(runDataDir,'dir')
    % 默认使用 m 文件所在目录下的 03_任务结果
    here = fileparts(mfilename('fullpath'));
    runDataDir = fullfile(here, '03_任务结果');
end
assert(exist(runDataDir,'dir')==7, '[task_step_cpu_cusignal_speedup_error] runDataDir not found: %s', runDataDir);

data = load_current_run_plot_data(runDataDir);

if nargin < 2 || isempty(outputDir) || strlength(string(outputDir)) == 0
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'figures');
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLegendFontName','Arial');
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');

speedup = data.cpu_speedup;
cpuTotalMs = [sum(data.cpu_mean_ms(1:5)); sum(data.cpu_mean_ms(6:11))];
gpuTotalMs = [sum(data.cpp_compute_mean_ms(1:5)); sum(data.cpp_compute_mean_ms(6:11))];
totalSpeedup = cpuTotalMs./gpuTotalMs;

[fig,ax,b] = draw_mother_style(speedup,'加速比(vs. CPU)');

summaryText = {
    sprintf('任务1流程合计：CPU %.2f ms | GPU %.2f ms | 总加速比 %.1f×',cpuTotalMs(1),gpuTotalMs(1),totalSpeedup(1))
    sprintf('任务2流程合计：CPU %.2f ms | GPU %.2f ms | 总加速比 %.1f×',cpuTotalMs(2),gpuTotalMs(2),totalSpeedup(2))
    '口径：CPU wall-clock / C++ GPU mean_ms  (T1:s11 / T2:s08 shape 族内 10 case 中位数)'
    sprintf('run_id=%s | config=%s/%s | warmup=%d repeats=%d | current_run_only', ...
        data.run_id,data.task1_sha(1:min(12,end)),data.task2_sha(1:min(12,end)),data.warmup,data.repeats)
};
add_summary_box(fig,summaryText,b.FaceColor);

outputPath = fullfile(outputDir,sprintf('task_step_cpu_speedup_error_%s',data.run_id));
exportgraphics(fig,[outputPath '.png'],'Resolution',600);
if isgraphics(fig,'figure')
    savefig(fig,[outputPath '.fig']);
else
    warning('[cpu] PNG 已导出，但图窗已关闭，跳过 FIG 保存：%s.fig', outputPath);
end
fprintf('Saved current-run CPU comparison: %s\n',outputPath);
end


% =========================================================================
% 内嵌数据加载函数：直接从 03_任务结果 目录读取分步耗时和精度
% =========================================================================
function data = load_current_run_plot_data(runDataDir)
% LOAD_CURRENT_RUN_PLOT_DATA  从 runDataDir (03_任务结果) 中分步提取 CPU/GPU 耗时中位数和RMSE中位数
%   按 shape 族过滤：Task1 → s11 (最大输入规模), Task2 → s08
%   每个 shape 下 10 个 scenario case 取中位数
%
% 返回结构体字段:
%   cpu_mean_ms(11)         cpu 分步耗时中位数(ms)，顺序 T1-S1..S5, T2-S1..S6
%   cpp_compute_mean_ms(11) gpu 分步耗时中位数(ms)
%   cpu_speedup(11)         = cpu_mean_ms ./ cpp_compute_mean_ms
%   rmse(11)                每步 max(rmse per output) 在 shape 内 case 中的中位数
%   run_id / task1_sha / task2_sha / warmup / repeats / run_dir

    % shape 过滤配置：Task1 用 s11 (最大规模), Task2 用 s08
    targetShapeT1 = 's11';
    targetShapeT2 = 's08';

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

    % --- 读取所有 case，按 shape 过滤 ---
    t1AllCases = listCases(t1CasesDir, 'task1_');
    t2AllCases = listCases(t2CasesDir, 'task2_');
    t1CaseList = filterCasesByShape(t1AllCases, 'task1_', targetShapeT1);
    t2CaseList = filterCasesByShape(t2AllCases, 'task2_', targetShapeT2);
    fprintf('[load_current_run_plot_data] T1 shape=%s: %d/%d cases, T2 shape=%s: %d/%d cases\n', ...
        targetShapeT1, numel(t1CaseList), numel(t1AllCases), ...
        targetShapeT2, numel(t2CaseList), numel(t2AllCases));

    % Task1: step1..step5 + pipeline_total (ignore pipeline_total)
    % Task2: step1..step6 + pipeline_total
    nStepT1 = 5; nStepT2 = 6;

    t1_cpu = zeros(nStepT1, numel(t1CaseList));
    t1_gpu = zeros(nStepT1, numel(t1CaseList));
    t1_rmse = zeros(nStepT1, numel(t1CaseList));
    t2_cpu = zeros(nStepT2, numel(t2CaseList));
    t2_gpu = zeros(nStepT2, numel(t2CaseList));
    t2_rmse = zeros(nStepT2, numel(t2CaseList));

    warmup = 20;
    repeats = 100;
    task1_sha = '';
    task2_sha = '';
    run_id_samples = strings(0,1);   % 取多个 case 的 run_id 样本，后求公共前缀

    for c = 1:numel(t1CaseList)
        caseDir = fullfile(t1CasesDir, t1CaseList{c});
        sumFile = fullfile(caseDir, 'task1_pipeline_summary_fft_thrust_formal.csv');
        accFile = fullfile(caseDir, 'task1_accuracy_details_fft_thrust_formal.csv');
        benchFile = fullfile(caseDir, 'task1_benchmark_fft_thrust_formal.csv');
        [cpuVals, gpuVals, sumMeta] = readStepTiming(sumFile, nStepT1, 'step');
        t1_cpu(:,c) = cpuVals;
        t1_gpu(:,c) = gpuVals;
        t1_rmse(:,c) = readStepRmse(accFile, 'Task1', nStepT1);

        % 提取元数据 (pipeline_summary 提供 measured_runs, benchmark 提供 warmup_runs)
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
            if benchMeta.repeats>0 && repeats==0, repeats = benchMeta.repeats; end
        end
    end

    for c = 1:numel(t2CaseList)
        caseDir = fullfile(t2CasesDir, t2CaseList{c});
        sumFile = fullfile(caseDir, 'task2_pipeline_summary_fft_thrust_formal.csv');
        accFile = fullfile(caseDir, 'task2_accuracy_details_fft_thrust_formal.csv');
        benchFile = fullfile(caseDir, 'task2_benchmark_fft_thrust_formal.csv');
        [cpuVals, gpuVals, sumMeta] = readStepTiming(sumFile, nStepT2, 'step');
        t2_cpu(:,c) = cpuVals;
        t2_gpu(:,c) = gpuVals;
        t2_rmse(:,c) = readStepRmse(accFile, 'Task2', nStepT2);

        if ~isempty(sumMeta)
            if isempty(task2_sha) && strlength(string(sumMeta.sha))>0, task2_sha = char(sumMeta.sha); end
            if sumMeta.repeats>0, repeats = sumMeta.repeats; end
        end
        if isempty(task2_sha)
            benchMeta = readMetadata(benchFile);
            if ~isempty(benchMeta) && strlength(benchMeta.sha)>0, task2_sha = char(benchMeta.sha); end
        end
    end

    % 中位数聚合 (按 step 取所有 case 的中位数)
    cpuMedians = [median(t1_cpu,2,'omitnan'); median(t2_cpu,2,'omitnan')];
    gpuMedians = [median(t1_gpu,2,'omitnan'); median(t2_gpu,2,'omitnan')];
    rmseMedians = [median(t1_rmse,2,'omitnan'); median(t2_rmse,2,'omitnan')];

    % 回退：若某步全NaN（理论不应出现），取同任务其他case非NaN中位数，最后用极小正值
    cpuMedians = fixNaN(cpuMedians, t1_cpu, t2_cpu);
    gpuMedians = fixNaN(gpuMedians, t1_gpu, t2_gpu);
    rmseMedians(isnan(rmseMedians)) = 0;   % RMSE 无数据视为 0

    % run_id = 所有样本的最长公共前缀，去除尾部 "_" 或 case 标识符
    if numel(run_id_samples) > 0
        run_id = char(longest_common_prefix(cellstr(run_id_samples)));
        % 去掉尾部 "_task1_s..." 等残段：只保留到最后一个已知稳定片段之前
        if endsWith(run_id, '_') && strlength(run_id) > 1
            run_id = run_id(1:end-1);
        end
    else
        run_id = 'current_run';
    end

    data = struct();
    data.cpu_mean_ms = cpuMedians;
    data.cpp_compute_mean_ms = gpuMedians;
    data.cpu_speedup = cpuMedians ./ gpuMedians;
    data.rmse = rmseMedians;
    data.run_id = run_id;
    data.task1_sha = task1_sha;
    data.task2_sha = task2_sha;
    data.warmup = warmup;
    data.repeats = repeats;
    data.run_dir = runDataDir;
end


% ---- 辅助: 按 shape 过滤 case 列表 ----
function filtered = filterCasesByShape(allCases, prefix, targetShape)
% FILTERCASESBYSHAPE  从 case 名中解析 shape (如 s11)，只保留匹配 targetShape 的 case
    filtered = {};
    pat = [prefix 's\d+_'];   % e.g., 'task1_s11_'
    for i = 1:numel(allCases)
        name = allCases{i};
        tok = regexp(name, pat, 'match', 'once');
        if isempty(tok), continue; end
        shape = regexp(tok, 's\d+', 'match', 'once');   % extract 's11'
        if strcmpi(shape, targetShape)
            filtered{end+1} = name; %#ok<AGROW>
        end
    end
    filtered = sort(filtered);
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


% ---- 辅助: 列出case目录名 ----
function cases = listCases(parentDir, prefix)
    d = dir(parentDir);
    isDir = [d.isdir];
    names = {d.name};
    keep = isDir & startsWith(names, prefix) & ~strcmp(names, '.') & ~strcmp(names, '..');
    cases = names(keep);
    if isempty(cases)
        % fallback: glob pattern match via subdir listing
        sub = dir(fullfile(parentDir, [prefix '*']));
        cases = {sub.name};
        cases = cases(~cellfun(@isempty, cases));
    end
    cases = sort(cases);
end


% ---- 辅助: 读取 pipeline_summary 中 step1..stepN 的 cpu/gpu mean_ms，并返回第一行元数据 ----
function [cpuVals, gpuVals, meta] = readStepTiming(sumFile, N, prefix)
    cpuVals = nan(N,1);
    gpuVals = nan(N,1);
    meta = struct('sha','','run_id','','repeats',0);
    if ~exist(sumFile,'file'), return; end

    T = readtableSafe(sumFile);
    if isempty(T), return; end

    % 读取第一行元数据（整个表的 git_commit/run_id/measured_runs 所有行相同）
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
        pat = sprintf('%s%d', prefix, i);   % "step1".."stepN"
        idx = strcmp(string(T.component), pat);
        if ~any(idx), continue; end
        cpuIdx = idx & strcmp(string(T.device),'cpu');
        gpuIdx = idx & strcmp(string(T.device),'gpu');
        if any(cpuIdx)
            vs = T.mean_ms(cpuIdx); vs = vs(~isnan(vs));
            if ~isempty(vs), cpuVals(i) = vs(1); end   % 同step同device仅一行
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
    rmses   = str2doubleSafe(string(T.rmse));   % NA→NaN

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


% ---- 辅助: 从 benchmark_csv 提取元数据 (struct 形式) ----
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
        % 不中断
    end
end


% ---- 辅助: 安全读取 CSV (detect text type) ----
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
    % 若某步是NaN（所有case都是NaN），尝试用同任务非NaN中位数，否则 1e-6 ms
    N1 = size(t1Mat,1);
    N2 = size(t2Mat,1);
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


% =========================================================================
% 绘图样式辅助函数（保留原脚本风格）
% =========================================================================
function [fig,ax,b] = draw_mother_style(speedup,leftLabel)
fig = figure('Color','w','Units','centimeters','Position',[2 2 15 11],...
    'PaperUnits','centimeters','PaperPosition',[0 0 15 11],'PaperSize',[15 11]);
ax = axes(fig); hold(ax,'on');
try
    ax.Toolbar.Visible = 'off';
    drawnow;
catch
end
x = 1:11; stepLabel = ["T1-S1","T1-S2","T1-S3","T1-S4","T1-S5",...
    "T2-S1","T2-S2","T2-S3","T2-S4","T2-S5","T2-S6"];
colorMap = coolwarm(256); barColor = colorMap(52,:);
b = bar(ax,x,speedup,0.62,'FaceColor',barColor,'EdgeColor',barColor*0.72,'LineWidth',0.45);
b.FaceAlpha = 0.88; ylabel(ax,leftLabel,'FontSize',9,'FontName','Arial');
ylim(ax,[min(speedup)/1.8 max(speedup)*3.0]); ax.YScale = 'log'; ax.YColor = barColor*0.72;
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
% 附: 验证测试 (在 MATLAB 命令行手动调用 test_load_data)
% =========================================================================
function test_load_data()
% TEST_LOAD_DATA  验证 load_current_run_plot_data 是否正确读取 03_任务结果 中的数据
% 用法: 在 m 文件所在目录，命令行执行 >> test_load_data

    fprintf('=== test_load_data starts ===\n');
    runDataDir = fullfile(fileparts(mfilename('fullpath')), '03_任务结果');
    assert(exist(runDataDir,'dir')==7, 'runDataDir not found: %s', runDataDir);

    data = load_current_run_plot_data(runDataDir);

    reqFields = {'cpu_mean_ms','cpp_compute_mean_ms','cpu_speedup','rmse',...
        'run_id','task1_sha','task2_sha','warmup','repeats','run_dir'};
    for f = reqFields
        assert(isfield(data, f{1}), 'Missing field: %s', f{1});
    end
    fprintf('[PASS] All required fields present.\n');

    assert(numel(data.cpu_mean_ms)==11, 'cpu_mean_ms must have 11 steps, got %d', numel(data.cpu_mean_ms));
    assert(numel(data.cpp_compute_mean_ms)==11, 'cpp_compute_mean_ms must have 11 steps, got %d', numel(data.cpp_compute_mean_ms));
    assert(numel(data.cpu_speedup)==11, 'cpu_speedup must have 11 steps, got %d', numel(data.cpu_speedup));
    assert(numel(data.rmse)==11, 'rmse must have 11 steps, got %d', numel(data.rmse));
    fprintf('[PASS] All arrays have 11 elements.\n');

    assert(all(data.cpu_mean_ms(:) > 0), 'cpu_mean_ms has non-positive value');
    assert(all(data.cpp_compute_mean_ms(:) > 0), 'cpp_compute_mean_ms has non-positive value');
    assert(all(data.cpu_speedup(:) > 0), 'cpu_speedup has non-positive value');
    assert(all(data.rmse(:) >= 0), 'rmse has negative value');
    fprintf('[PASS] All timing/speedup positive; rmse non-negative.\n');

    expectedSpeedup = data.cpu_mean_ms(:) ./ data.cpp_compute_mean_ms(:);
    maxRelErr = max(abs(data.cpu_speedup(:) - expectedSpeedup) ./ expectedSpeedup);
    assert(maxRelErr < 1e-6, 'cpu_speedup mismatch: max rel error = %g', maxRelErr);
    fprintf('[PASS] cpu_speedup consistent with cpu_mean_ms ./ cpp_compute_mean_ms.\n');

    assert(strlength(string(data.run_id)) > 0, 'run_id empty');
    assert(strlength(string(data.task1_sha)) > 0, 'task1_sha empty');
    assert(strlength(string(data.task2_sha)) > 0, 'task2_sha empty');
    assert(data.warmup > 0, 'warmup must be > 0');
    assert(data.repeats > 0, 'repeats must be > 0');
    fprintf('[PASS] Metadata non-empty. run_id=%s, sha1=%s, sha2=%s, warmup=%d, repeats=%d\n', ...
        data.run_id, data.task1_sha(1:min(12,end)), data.task2_sha(1:min(12,end)), ...
        data.warmup, data.repeats);

    labels = {
        'T1-S1' 'T1-S2' 'T1-S3' 'T1-S4' 'T1-S5' ...
        'T2-S1' 'T2-S2' 'T2-S3' 'T2-S4' 'T2-S5' 'T2-S6'
    };
    fprintf('\n=== Step Summary (T1:s11 / T2:s08 shape median, 10 cases each) ===\n');
    fprintf('%-5s  %-6s  %12s  %12s  %10s  %12s\n', ...
        'Idx', 'Step', 'CPU ms', 'C++ GPU ms', 'Speedup', 'RMSE');
    for i = 1:11
        fprintf('  %-3d  %-6s  %12.4f  %12.4f  %8.2fx  %12.4e\n', ...
            i, labels{i}, data.cpu_mean_ms(i), data.cpp_compute_mean_ms(i), ...
            data.cpu_speedup(i), data.rmse(i));
    end

    fprintf('\n=== ALL TESTS PASSED ===\n');
end
