function fig15_step_composition()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'task_step_composition.csv');
    comp_data = readtable(data_path);
    
    task1_python = comp_data(contains(comp_data.implementation, 'Task1-cuSignal-Python'), :);
    task2_python = comp_data(contains(comp_data.implementation, 'Task2-cuSignal-Python'), :);
    task1_cpp = comp_data(contains(comp_data.implementation, 'Task1-C++-CUDA'), :);
    task2_cpp = comp_data(contains(comp_data.implementation, 'Task2-C++-CUDA'), :);
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    
    stages = {'prepare', 'h2d', 'compute', 'd2h', 'postprocess'};
    colors = {config.colors.prepare, config.colors.h2d, config.colors.compute, config.colors.d2h, config.colors.postprocess};
    
    task1_py_vals = [sum(task1_python.prepare_ms), sum(task1_python.h2d_ms), sum(task1_python.compute_ms), sum(task1_python.d2h_ms), sum(task1_python.postprocess_ms)];
    task2_py_vals = [sum(task2_python.prepare_ms), sum(task2_python.h2d_ms), sum(task2_python.compute_ms), sum(task2_python.d2h_ms), sum(task2_python.postprocess_ms)];
    task1_cpp_vals = [sum(task1_cpp.prepare_ms), sum(task1_cpp.h2d_ms), sum(task1_cpp.compute_ms), sum(task1_cpp.d2h_ms), sum(task1_cpp.postprocess_ms)];
    task2_cpp_vals = [sum(task2_cpp.prepare_ms), sum(task2_cpp.h2d_ms), sum(task2_cpp.compute_ms), sum(task2_cpp.d2h_ms), sum(task2_cpp.postprocess_ms)];
    
    hold(ax, 'on');
    bottom = zeros(1, 4);
    
    for i = 1:length(stages)
        vals = [task1_py_vals(i), task2_py_vals(i), task1_cpp_vals(i), task2_cpp_vals(i)];
        b = bar(ax, [1 2 3 4], vals, config.bar_width, 'BaseValue', bottom, 'EdgeColor', 'black');
        b.FaceColor = colors{i};
        bottom = bottom + vals;
    end
    
    set(ax, 'XTick', [1 2 3 4]);
    set(ax, 'XTickLabel', {'Task1-Python', 'Task2-Python', 'Task1-C++', 'Task2-C++'});
    set(ax, 'FontSize', config.font_size_small);
    set(ax, 'LineWidth', config.line_width);
    
    xlabel('任务与实现方式', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('耗时 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    legend(stages, 'FontSize', config.font_size_small, 'FontName', config.font_name, 'Location', 'best');
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 数据来源: task_step_composition.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end