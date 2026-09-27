function fig5_mse_bar()

close all;
sObj = settings;
desktopZoom = sObj.matlab.desktop.Zoom.PersonalValue;
sObj.matlab.desktop.Zoom.PersonalValue = 100;

config = plot_common();

data_path = fullfile(config.data_dir, 'operator_accuracy.csv');
accuracy_data = readtable(data_path);

fp32_data = accuracy_data(strcmp(accuracy_data.input_dtype, 'FP32'), :);
operators = fp32_data.operator;
rmse_values = fp32_data.rmse;

fig = figure('Units', 'centimeters', 'Position', ...
    [1, 1, 15, 12], 'Color', 'white');

ax = axes('Parent', fig);
bar(rmse_values, config.bar_width, 'FaceColor', config.colors.gpu, 'EdgeColor', 'black');

set(ax, 'YScale', 'log');
set(ax, 'XTick', 1:length(operators));
set(ax, 'XTickLabel', operators);
set(ax, 'XTickLabelRotation', 90);
set(ax, 'FontSize', config.font_size_small);
set(ax, 'LineWidth', config.line_width);

xlabel('cuSignal 算子', 'FontSize', config.font_size_large, 'FontName', config.font_name);
ylabel('RMSE', 'FontSize', config.font_size_large, 'FontName', config.font_name);

apply_style(ax, config);

% annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
%     'String', '测试环境: ZQ500 GPU | dtype: FP32 | MSE = mean((x_gpu - x_ref)^2) | 数据来源: operator_accuracy.csv', ...
%     'FontSize', config.font_size_small, 'FontName', config.font_name, ...
%     'EdgeColor', 'none', 'HorizontalAlignment', 'center');
save_figure(fig, 'rmse_bar', config);

sObj.matlab.desktop.Zoom.PersonalValue = desktopZoom;
end