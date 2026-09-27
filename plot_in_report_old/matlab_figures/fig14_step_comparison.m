function fig14_step_comparison()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'task_step_comparison.csv');
    step_data = readtable(data_path);
    
    task1_data = step_data(strcmp(step_data.task, 'Task1'), :);
    task2_data = step_data(strcmp(step_data.task, 'Task2'), :);
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax1 = subplot(1, 2, 1);
    steps = task1_data.step;
    x = 1:length(steps);
    bar_width = 0.35;
    
    python_vals = task1_data.python_mean_ms;
    cpp_vals = task1_data.cpp_mean_ms;
    
    valid_idx = ~isnan(python_vals) & ~isnan(cpp_vals);
    x_valid = x(valid_idx);
    python_vals = python_vals(valid_idx);
    cpp_vals = cpp_vals(valid_idx);
    steps_valid = steps(valid_idx);
    
    bar(ax1, x_valid - bar_width/2, python_vals, bar_width, 'FaceColor', config.colors.python, 'EdgeColor', 'black');
    bar(ax1, x_valid + bar_width/2, cpp_vals, bar_width, 'FaceColor', config.colors.cpp, 'EdgeColor', 'black');
    
    set(ax1, 'XTick', x_valid);
    set(ax1, 'XTickLabel', steps_valid);
    set(ax1, 'XTickLabelRotation', 45);
    set(ax1, 'FontSize', config.font_size_small);
    set(ax1, 'LineWidth', config.line_width);
    set(ax1, 'YScale', 'log');
    
    xlabel('Task1步骤', 'FontSize', config.font_size, 'FontName', config.font_name);
    ylabel('耗时 (ms)', 'FontSize', config.font_size, 'FontName', config.font_name);
    title('Task1步骤耗时对比', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    legend({'Python', 'C++'}, 'FontSize', config.font_size_small, 'FontName', config.font_name, 'Location', 'best');
    
    apply_style(ax1, config);
    
    ax2 = subplot(1, 2, 2);
    steps2 = task2_data.step;
    x2 = 1:length(steps2);
    
    bar(ax2, x2 - bar_width/2, task2_data.python_mean_ms, bar_width, 'FaceColor', config.colors.python, 'EdgeColor', 'black');
    bar(ax2, x2 + bar_width/2, task2_data.cpp_mean_ms, bar_width, 'FaceColor', config.colors.cpp, 'EdgeColor', 'black');
    
    set(ax2, 'XTick', x2);
    set(ax2, 'XTickLabel', steps2);
    set(ax2, 'XTickLabelRotation', 45);
    set(ax2, 'FontSize', config.font_size_small);
    set(ax2, 'LineWidth', config.line_width);
    set(ax2, 'YScale', 'log');
    
    xlabel('Task2步骤', 'FontSize', config.font_size, 'FontName', config.font_name);
    ylabel('耗时 (ms)', 'FontSize', config.font_size, 'FontName', config.font_name);
    title('Task2步骤耗时对比', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    legend({'Python', 'C++'}, 'FontSize', config.font_size_small, 'FontName', config.font_name, 'Location', 'best');
    
    apply_style(ax2, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 统计次数: 100 | 数据来源: task_step_comparison.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end