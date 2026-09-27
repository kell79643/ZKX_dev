function data = load_current_run_plot_data(runDataDir)
% 严格读取一个 run bundle；不搜索、不回退、不按最大值挑选重复记录。

runDir = char(java.io.File(runDataDir).getCanonicalPath());
manifestPath = fullfile(runDir,'manifest.json');
chartDir = fullfile(runDir,'chart_data');
required = {manifestPath,...
    fullfile(chartDir,'task_step_accuracy.csv'),...
    fullfile(chartDir,'task_cpu_gpu_performance_comparison.csv'),...
    fullfile(chartDir,'task_step_comparison.csv'),...
    fullfile(chartDir,'task_effect_metrics.csv'),...
    fullfile(chartDir,'task_demo_summary.json')};
for i = 1:numel(required)
    if ~isfile(required{i}), error('当前 run bundle 缺少文件：%s',required{i}); end
end
manifest = jsondecode(fileread(manifestPath));
if ~strcmp(string(manifest.status),"pass"), error('manifest 状态不是 pass。'); end
if ~isfield(manifest,'artifacts_sha256'), error('manifest 缺少 artifacts_sha256。'); end

taskOrder = [repmat("Task1",5,1); repmat("Task2",6,1)];
stepOrder = ["step1";"step2";"step3";"step4";"step5";...
    "step1";"step2";"step3";"step4";"step5";"step6"];
keys = ["T1-S1";"T1-S2";"T1-S3";"T1-S4";"T1-S5";...
    "T2-S1";"T2-S2";"T2-S3";"T2-S4";"T2-S5";"T2-S6"];
task1Sha = string(manifest.configs.Task1.sha256);
task2Sha = string(manifest.configs.Task2.sha256);
expectedSha = [repmat(task1Sha,5,1); repmat(task2Sha,6,1)];

accuracy = readtable(required{2},'VariableNamingRule','preserve');
cpu = readtable(required{3},'VariableNamingRule','preserve');
comparison = readtable(required{4},'VariableNamingRule','preserve');
effect = readtable(required{5},'VariableNamingRule','preserve');
rmse = zeros(11,1); cpuSpeedup = zeros(11,1); pythonSpeedup = zeros(11,1);
cpuMean = zeros(11,1); cppCompute = zeros(11,1); pythonFormal = zeros(11,1); cppFormal = zeros(11,1);
for i = 1:11
    ai = exact_row(accuracy,taskOrder(i),stepOrder(i),keys(i),expectedSha(i),'accuracy');
    ci = exact_row(cpu,taskOrder(i),stepOrder(i),keys(i),expectedSha(i),'CPU comparison');
    pi = exact_row(comparison,taskOrder(i),stepOrder(i),keys(i),expectedSha(i),'Python comparison');
    ei = strcmpi(string(effect.("Task")),taskOrder(i)) & ...
        strcmpi(string(effect.("Step")),stepOrder(i)) & ...
        strcmp(string(effect.("EvaluationKey")),keys(i)) & ...
        strcmp(string(effect.("config_sha256")),expectedSha(i));
    if nnz(ei) < 1, error('%s 缺少本次业务指标。',keys(i)); end
    rmse(i) = accuracy.('RMSE')(ai);
    cpuSpeedup(i) = cpu.('CPU_CPP_GPU_speedup')(ci);
    pythonSpeedup(i) = comparison.('Python_CPP_GPU_speedup')(pi);
    cpuMean(i) = cpu.('CPU_reference_mean_ms')(ci);
    cppCompute(i) = cpu.('CPP_GPU_compute_mean_ms')(ci);
    pythonFormal(i) = comparison.('Python_GPU_formal_mean_ms')(pi);
    cppFormal(i) = comparison.('CPP_GPU_formal_mean_ms')(pi);
end
allValues = [rmse;cpuSpeedup;pythonSpeedup;cpuMean;cppCompute;pythonFormal;cppFormal];
if any(~isfinite(allValues)), error('本次图表数据包含 NaN/Inf。'); end
if any(rmse < 0) || any(cpuSpeedup <= 0) || any(pythonSpeedup <= 0) || ...
        any(cpuMean <= 0) || any(cppCompute <= 0) || any(pythonFormal <= 0) || any(cppFormal <= 0)
    error('本次图表数据违反非负误差/正耗时/正加速比约束。');
end
data = struct('run_dir',runDir,'run_id',char(manifest.run_id),...
    'task1_sha',char(task1Sha),'task2_sha',char(task2Sha),...
    'warmup',manifest.warmup_runs,'repeats',manifest.measured_runs,...
    'rmse',rmse,'cpu_speedup',cpuSpeedup,'python_speedup',pythonSpeedup,...
    'cpu_mean_ms',cpuMean,'cpp_compute_mean_ms',cppCompute,...
    'python_formal_mean_ms',pythonFormal,'cpp_formal_mean_ms',cppFormal);
end


function index = exact_row(tableData,task,step,key,configSha,label)
idx = strcmpi(string(tableData.('Task')),task) & ...
    strcmpi(string(tableData.('Step')),step) & ...
    strcmp(string(tableData.('EvaluationKey')),key) & ...
    strcmp(string(tableData.('config_sha256')),configSha);
if nnz(idx) ~= 1
    error('%s 的 %s 必须唯一，实际 %d 行。',key,label,nnz(idx));
end
index = find(idx,1);
end
