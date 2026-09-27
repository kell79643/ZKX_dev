function fig16_scatter()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'task_step_comparison.csv');
    step_data = readtable(data_path);
    
    python_times = step_data.python_mean_ms;
    cpp_times = step_data.cpp_mean_ms;
    
    valid_idx = ~isnan(python_times) & ~isnan(cpp_times);
    python_times = python_times(valid_idx);
    cpp_times = cpp_times(valid_idx);
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    scatter(python_times, cpp_times, 100, config.colors.gpu, 'filled');
    
    max_val = max([python_times; cpp_times]) * 1.1;
    plot(ax, [0 max_val], [0 max_val], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    
    set(ax, 'FontSize', config.font_size);
    set(ax, 'LineWidth', config.line_width);
    set(ax, 'XScale', 'log');
    set(ax, 'YScale', 'log');
    
    xlabel('Python耗时 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('C++耗时 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 对角线表示性能相当 | 数据来源: task_step_comparison.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end