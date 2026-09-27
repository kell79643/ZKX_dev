function fig12_scale_performance()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_scale_performance.csv');
    scale_data = readtable(data_path);
    
    operators = {'unit_impulse', 'firwin'};
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    
    line_colors = [config.colors.gpu; config.colors.cpu; config.colors.python; config.colors.cpp; ...
                   [0.58 0.40 0.74]; [0.85 0.33 0.10]];
    
    for i = 1:length(operators)
        op_data = scale_data(strcmp(scale_data.operator, operators{i}), :);
        plot(ax, op_data.scale, op_data.gpu_mean_ms, ...
            'LineWidth', config.line_width_thick, ...
            'Color', line_colors(i,:), ...
            'Marker', 'o', ...
            'MarkerSize', config.marker_size, ...
            'DisplayName', operators{i});
    end
    
    hold(ax, 'on');
    xlabel('输入规模', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('GPU平均延迟 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    set(ax, 'FontSize', config.font_size);
    set(ax, 'LineWidth', config.line_width);
    set(ax, 'YScale', 'log');
    
    legend('FontSize', config.font_size_small, 'FontName', config.font_name, 'Location', 'best');
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 统计次数: 100 | 数据来源: operator_scale_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end