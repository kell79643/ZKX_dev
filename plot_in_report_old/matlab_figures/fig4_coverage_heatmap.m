function fig4_coverage_heatmap()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_coverage.csv');
    coverage_data = readtable(data_path);
    
    operators = unique(coverage_data.operator);
    dtypes = {'FP32', 'FP16', 'INT32', 'INT16', 'INT8'};
    num_ops = length(operators);
    num_dtypes = length(dtypes);
    
    coverage_matrix = zeros(num_ops, num_dtypes);
    
    for i = 1:num_ops
        for j = 1:num_dtypes
            idx = find(strcmp(coverage_data.operator, operators{i}) & ...
                      strcmp(coverage_data.input_dtype, dtypes{j}));
            if ~isempty(idx)
                coverage_matrix(i, j) = coverage_data.value(idx);
            end
        end
    end
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height_tall*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    imagesc(coverage_matrix);
    
    colormap(ax, [0.85 0.33 0.10; 0.15 0.68 0.38]);
    colorbar('Ticks', [0.5 1.5], 'TickLabels', {'未通过', '通过'}, ...
        'FontSize', config.font_size, 'FontName', config.font_name);
    
    set(ax, 'XTick', 1:num_dtypes);
    set(ax, 'XTickLabel', dtypes);
    set(ax, 'YTick', 1:num_ops);
    set(ax, 'YTickLabel', operators);
    set(ax, 'FontSize', config.font_size_small);
    set(ax, 'LineWidth', config.line_width);
    
    xlabel('数据类型', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('算子名称', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | 判定标准: 满足规定误差阈值则通过 | 数据来源: operator_coverage.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end