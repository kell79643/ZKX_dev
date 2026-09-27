function fig19_task_latency_boxplot()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'task_pipeline_performance.csv');
    pipeline_data = readtable(data_path);
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    
    python_data = pipeline_data(strcmp(pipeline_data.implementation, 'cuSignal Python GPU'), :);
    cpp_data = pipeline_data(strcmp(pipeline_data.implementation, 'C++ CUDA GPU'), :);
    
    python_mean = python_data.mean_ms;
    cpp_mean = cpp_data.mean_ms;
    
    x = [1, 2];
    b = bar(ax, x, [python_mean, cpp_mean], config.bar_width);
    b(1).FaceColor = config.colors.python;
    b(2).FaceColor = config.colors.cpp;
    
    set(ax, 'XTick', x);
    set(ax, 'XTickLabel', {'Python', 'C++'});
    set(ax, 'FontSize', config.font_size);
    set(ax, 'LineWidth', config.line_width);
    set(ax, 'YScale', 'log');
    
    xlabel('实现方式', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('端到端延迟 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 统计次数: 100 | 数据来源: task_pipeline_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end