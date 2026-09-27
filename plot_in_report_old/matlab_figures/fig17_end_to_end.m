function fig17_end_to_end()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'task_pipeline_performance.csv');
    pipeline_data = readtable(data_path);
    
    tasks = unique(pipeline_data.task);
    python_data = pipeline_data(contains(pipeline_data.implementation, 'Python'), :);
    cpp_data = pipeline_data(contains(pipeline_data.implementation, 'C++'), :);
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    x = 1:length(tasks);
    bar_width = 0.35;
    
    bar(ax, x - bar_width/2, python_data.mean_ms, bar_width, 'FaceColor', config.colors.python, 'EdgeColor', 'black');
    bar(ax, x + bar_width/2, cpp_data.mean_ms, bar_width, 'FaceColor', config.colors.cpp, 'EdgeColor', 'black');
    
    set(ax, 'XTick', x);
    set(ax, 'XTickLabel', tasks);
    set(ax, 'FontSize', config.font_size);
    set(ax, 'LineWidth', config.line_width);
    
    xlabel('任务', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('端到端总耗时 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    legend({'Python实现', 'C++实现'}, 'FontSize', config.font_size_small, 'FontName', config.font_name, 'Location', 'best');
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 数据来源: task_pipeline_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end