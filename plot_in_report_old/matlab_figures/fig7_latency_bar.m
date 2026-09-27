function fig7_latency_bar()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_performance.csv');
    perf_data = readtable(data_path);
    
    fp32_data = perf_data(strcmp(perf_data.input_dtype, 'FP32'), :);
    operators = fp32_data.operator;
    gpu_latency = fp32_data.gpu_mean_ms;
    cpu_latency = fp32_data.cpu_mean_ms;
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    x = 1:length(operators);
    bar_width = 0.35;
    
    bar(ax, x - bar_width/2, gpu_latency, bar_width, 'FaceColor', config.colors.gpu, 'EdgeColor', 'black');
    bar(ax, x + bar_width/2, cpu_latency, bar_width, 'FaceColor', config.colors.cpu, 'EdgeColor', 'black');
    
    set(ax, 'XTick', x);
    set(ax, 'XTickLabel', operators);
    set(ax, 'XTickLabelRotation', 90);
    set(ax, 'FontSize', config.font_size_small);
    set(ax, 'LineWidth', config.line_width);
    set(ax, 'YScale', 'log');
    
    xlabel('算子名称', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('平均延迟 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    legend({'GPU延迟', 'CPU延迟'}, 'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'Location', 'best');
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 统计次数: 100 | 数据来源: operator_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end