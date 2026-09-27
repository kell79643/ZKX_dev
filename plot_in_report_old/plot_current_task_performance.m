function outputs = plot_current_task_performance(runDataDir,outputDir)
% 生成本次 run_id 的左右单图和组合图。
if nargin < 2 || strlength(string(outputDir)) == 0
    outputDir = fullfile(runDataDir,'figures');
end
[leftFig,leftBase] = plot_task_step_cpu_speedup_error(runDataDir,outputDir);
[rightFig,rightBase] = plot_task_step_cusignal_speedup_error(runDataDir,outputDir);
data = load_current_run_plot_data(runDataDir);
leftImage = imread([leftBase '.png']); rightImage = imread([rightBase '.png']);
combo = figure('Color','w','Units','centimeters','Position',[1 1 30 11],...
    'PaperUnits','centimeters','PaperPosition',[0 0 30 11],'PaperSize',[30 11]);
layout = tiledlayout(combo,1,2,'TileSpacing','compact','Padding','compact');
nexttile(layout); image(leftImage); axis image off; title('与 CPU reference 对比','FontName','Arial');
nexttile(layout); image(rightImage); axis image off; title('与 cuSignal Python GPU 对比','FontName','Arial');
comboBase = fullfile(outputDir,sprintf('task_performance_combo_%s',data.run_id));
exportgraphics(combo,[comboBase '.png'],'Resolution',600); savefig(combo,[comboBase '.fig']);
outputs = struct('cpu',leftBase,'cusignal',rightBase,'combo',comboBase,...
    'figures',[leftFig rightFig combo]);
fprintf('Saved current-run combined figure: %s\n',comboBase);
end
