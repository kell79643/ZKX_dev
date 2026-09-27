function fig8_speedup_bar()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_performance.csv');
    perf_data = readtable(data_path);
    
    fp32_data = perf_data(strcmp(perf_data.input_dtype, 'FP32'), :);
    operators = fp32_data.operator;
    speedup = fp32_data.cpu_gpu_speedup;
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    bar(speedup, config.bar_width, 'FaceColor', config.colors.gpu, 'EdgeColor', 'black');
    
    set(ax, 'XTick', 1:length(operators));
    set(ax, 'XTickLabel', operators);
    set(ax, 'XTickLabelRotation', 90);
    set(ax, 'FontSize', config.font_size_small);
    set(ax, 'LineWidth', config.line_width);
    
    xlabel('算子名称', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('CPU/GPU加速比', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    yline(1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 加速比 = CPU延迟/GPU延迟 | 数据来源: operator_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end