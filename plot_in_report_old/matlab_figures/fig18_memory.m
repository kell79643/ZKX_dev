function fig18_memory()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'task_memory_observations.csv');
    memory_data = readtable(data_path);
    
    tasks = unique(memory_data.task);
    
    python_memory = zeros(1, length(tasks));
    cpp_memory = zeros(1, length(tasks));
    
    for i = 1:length(tasks)
        task_data = memory_data(strcmp(memory_data.task, tasks{i}), :);
        python_idx = contains(task_data.implementation, 'python');
        cpp_idx = contains(task_data.implementation, 'cpp');
        
        if any(python_idx)
            python_memory(i) = task_data.peak_bytes(python_idx) / 1e6;
        end
        if any(cpp_idx)
            cpp_memory(i) = task_data.peak_bytes(cpp_idx) / 1e6;
        end
    end
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    x = 1:length(tasks);
    bar_width = 0.35;
    
    bar(ax, x - bar_width/2, python_memory, bar_width, 'FaceColor', config.colors.python, 'EdgeColor', 'black');
    bar(ax, x + bar_width/2, cpp_memory, bar_width, 'FaceColor', config.colors.cpp, 'EdgeColor', 'black');
    
    set(ax, 'XTick', x);
    set(ax, 'XTickLabel', tasks);
    set(ax, 'FontSize', config.font_size);
    set(ax, 'LineWidth', config.line_width);
    
    xlabel('任务', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('GPU内存峰值 (MB)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    legend({'Python实现', 'C++实现'}, 'FontSize', config.font_size_small, 'FontName', config.font_name, 'Location', 'best');
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 数据来源: task_memory_observations.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end