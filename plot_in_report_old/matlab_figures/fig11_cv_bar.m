function fig11_cv_bar()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_performance.csv');
    perf_data = readtable(data_path);
    
    fp32_data = perf_data(strcmp(perf_data.input_dtype, 'FP32'), :);
    operators = fp32_data.operator;
    cv_values = fp32_data.gpu_cv * 100;
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    bar(cv_values, config.bar_width, 'FaceColor', config.colors.cpu, 'EdgeColor', 'black');
    
    set(ax, 'XTick', 1:length(operators));
    set(ax, 'XTickLabel', operators);
    set(ax, 'XTickLabelRotation', 90);
    set(ax, 'FontSize', config.font_size_small);
    set(ax, 'LineWidth', config.line_width);
    
    xlabel('算子名称', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('稳定性 CV (%)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    yline(5, '--', 'Color', [0.85 0.33 0.10], 'LineWidth', 1);
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | CV = std/mean × 100 | 数据来源: operator_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end