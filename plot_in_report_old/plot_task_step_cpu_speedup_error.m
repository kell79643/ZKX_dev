function [fig,outputPath] = plot_task_step_cpu_speedup_error(runDataDir,outputDir)
% 对外入口；绘图实现唯一保留在原 CPU 母版文件中。
if nargin < 2
    [fig,outputPath] = task_step_cpu_cusignal_speedup_error(runDataDir);
else
    [fig,outputPath] = task_step_cpu_cusignal_speedup_error(runDataDir,outputDir);
end
end
