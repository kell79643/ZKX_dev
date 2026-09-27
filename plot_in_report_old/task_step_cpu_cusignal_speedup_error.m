function [fig,outputPath] = task_step_cpu_cusignal_speedup_error(runDataDir,outputDir)
% 使用指定 run_id 的本次 CPU/C++ GPU 证据绘图，禁止回退历史正式 CSV。

data = load_current_run_plot_data(runDataDir);
if nargin < 2 || strlength(string(outputDir)) == 0
    outputDir = fullfile(data.run_dir,'figures');
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLegendFontName','Arial');
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');

speedup = data.cpu_speedup;
rmse = data.rmse;
cpuTotalMs = [sum(data.cpu_mean_ms(1:5)); sum(data.cpu_mean_ms(6:11))];
gpuTotalMs = [sum(data.cpp_compute_mean_ms(1:5)); sum(data.cpp_compute_mean_ms(6:11))];
totalSpeedup = cpuTotalMs./gpuTotalMs;

[fig,ax,b,p,rmseFloor] = draw_mother_style(speedup,rmse,'加速比(vs. CPU)');
lgd = legend(ax,[b p],{'加速比','RMSE'},'Location','none','Orientation','vertical',...
    'Box','on','FontSize',8,'FontName','Arial');
lgd.Units = 'normalized';
lgd.Position = [0.15 0.735 0.105 0.090];

summaryText = {
    sprintf('任务1流程合计：CPU %.2f ms | GPU %.2f ms | 总加速比 %.1f×',cpuTotalMs(1),gpuTotalMs(1),totalSpeedup(1))
    sprintf('任务2流程合计：CPU %.2f ms | GPU %.2f ms | 总加速比 %.1f×',cpuTotalMs(2),gpuTotalMs(2),totalSpeedup(2))
    '口径：CPU reference algorithm wall-clock / C++ GPU compute_ms'
    sprintf('run_id=%s | config=%s/%s | warmup=%d repeats=%d | current_run_only', ...
        data.run_id,data.task1_sha(1:12),data.task2_sha(1:12),data.warmup,data.repeats)
};
add_summary_box(fig,summaryText,b.FaceColor);
mark_zero_rmse(ax,rmse,rmseFloor);

outputPath = fullfile(outputDir,sprintf('task_step_cpu_speedup_error_%s',data.run_id));
exportgraphics(fig,[outputPath '.png'],'Resolution',600);
savefig(fig,[outputPath '.fig']);
fprintf('Saved current-run CPU comparison: %s\n',outputPath);
end


function [fig,ax,b,p,rmseFloor] = draw_mother_style(speedup,rmse,leftLabel)
sObj = settings;
desktopZoom = sObj.matlab.desktop.Zoom.PersonalValue;
sObj.matlab.desktop.Zoom.PersonalValue = 100;
zoomCleanup = onCleanup(@() restore_desktop_zoom(sObj,desktopZoom)); %#ok<NASGU>
fig = figure('Color','w','Units','centimeters','Position',[2 2 15 11],...
    'PaperUnits','centimeters','PaperPosition',[0 0 15 11],'PaperSize',[15 11]);
ax = axes(fig); hold(ax,'on');
x = 1:11; stepLabel = ["T1-S1","T1-S2","T1-S3","T1-S4","T1-S5",...
    "T2-S1","T2-S2","T2-S3","T2-S4","T2-S5","T2-S6"];
colorMap = coolwarm(256); barColor = colorMap(52,:); lineColor = colorMap(205,:);
yyaxis(ax,'left');
b = bar(ax,x,speedup,0.62,'FaceColor',barColor,'EdgeColor',barColor*0.72,'LineWidth',0.45);
b.FaceAlpha = 0.88; ylabel(ax,leftLabel,'FontSize',9,'FontName','Arial');
ylim(ax,[min(speedup)/1.8 max(speedup)*3.0]); ax.YAxis(1).Scale = 'log'; ax.YColor = barColor*0.72;
for i = 1:numel(speedup)
    text(ax,x(i),speedup(i)*1.15,sprintf('%.1f×',speedup(i)),...
        'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',6.5,...
        'FontName','Arial','Color',barColor*0.72);
end
yyaxis(ax,'right');
positiveRmse = rmse(rmse > 0);
if isempty(positiveRmse), rmseFloor = 1e-12; else, rmseFloor = min(positiveRmse)/10; end
rmsePlot = rmse; rmsePlot(rmse == 0) = rmseFloor;
p = plot(ax,x,rmsePlot,'-o','Color',lineColor,'MarkerFaceColor','w',...
    'MarkerEdgeColor',lineColor,'LineWidth',1.35,'MarkerSize',4.5);
ylabel(ax,'RMSE','FontSize',9,'FontName','Arial');
ylim(ax,[rmseFloor/2 max(rmsePlot)*4]); ax.YAxis(2).Scale = 'log'; ax.YColor = lineColor*0.78;
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


function mark_zero_rmse(ax,rmse,rmseFloor)
yyaxis(ax,'right');
for i = find(rmse == 0)'
    text(ax,i,rmseFloor*1.35,'0','HorizontalAlignment','center',...
        'VerticalAlignment','bottom','FontSize',6.5,'FontName','Arial');
end
end


function add_summary_box(fig,summaryText,barColor)
annotation(fig,'textbox',[0.10 0.005 0.77 0.155],'String',summaryText,...
    'FitBoxToText','off','HorizontalAlignment','center','VerticalAlignment','middle',...
    'FontName','Arial','FontSize',6.7,'BackgroundColor',[0.96 0.97 1.00],...
    'EdgeColor',barColor*0.82,'LineWidth',0.55,'Margin',3);
end


function restore_desktop_zoom(sObj,desktopZoom)
sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
end
