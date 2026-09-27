function fig3_task2_flow()
    config = plot_common();
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    set(ax, 'Visible', 'off');
    set(ax, 'XLim', [0 100]);
    set(ax, 'YLim', [0 100]);
    
    box_width = 15;
    box_height = 8;
    
    x_center = 50;
    y_top = 75;
    y_middle = 50;
    y_bottom = 25;
    
    draw_box(ax, [x_center - box_width/2, y_top, box_width, box_height], [0.25 0.41 0.88], '特征提取');
    
    y_feature = y_top - 18;
    x_features = [15, 35, 55, 75];
    feature_labels = {'时域特征', '频域特征', '空间域特征', '其他特征'};
    feature_colors = [0.58 0.40 0.74; 0.15 0.68 0.38; 0.93 0.69 0.13; 0.85 0.33 0.10];
    
    for i = 1:length(feature_labels)
        pos = [x_features(i), y_feature, box_width, box_height];
        draw_box(ax, pos, feature_colors(i,:), feature_labels{i});
    end
    
    draw_box(ax, [x_center - box_width/2, y_middle - box_height/2 - 10, box_width, box_height], [0.25 0.41 0.88], '特征融合');
    
    draw_box(ax, [x_center - box_width/2, y_bottom, box_width, box_height], [0.15 0.68 0.38], '定位结果');
    
    draw_arrow(fig, x_center, y_top - 5, x_center, y_feature + box_height);
    
    for i = 1:length(feature_labels)
        draw_arrow(fig, x_features(i) + box_width/2, y_feature, x_features(i) + box_width/2, y_middle - box_height/2 - 10 + box_height);
    end
    
    draw_arrow(fig, x_center, y_middle - box_height/2 - 10 - 5, x_center, y_bottom + box_height);
    
    detail_labels = {'均值/方差/峰值', '频谱特征', '空间相关性', '统计特征', '特征加权融合', '坐标与置信度'};
    detail_positions = [x_features, x_center, x_center];
    detail_y_positions = [y_feature - 6, y_feature - 6, y_feature - 6, y_feature - 6, y_middle - box_height/2 - 24, y_bottom - 6];
    
    for i = 1:length(detail_labels)
        if i <= 4
            text_x = detail_positions(i) + box_width/2;
        else
            text_x = detail_positions(i);
        end
        text(text_x, detail_y_positions(i), detail_labels{i}, ...
            'Parent', ax, ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 10, ...
            'FontName', 'Microsoft YaHei', ...
            'Color', [0.5 0.5 0.5]);
    end
    
    annotation(fig, 'textbox', [0.05, 0.05, 0.9, 0.05], ...
        'String', '测试环境: ZQ500 GPU | Task2: 多域特征提取定位 | 统计次数: 100', ...
        'FontSize', config.font_size_small, ...
        'FontName', config.font_name, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center');
end

function draw_box(ax, pos, color, text_str)
    x = pos(1);
    y = pos(2);
    w = pos(3);
    h = pos(4);
    
    rectangle('Parent', ax, ...
        'Position', [x, y, w, h], ...
        'FaceColor', color, ...
        'EdgeColor', 'black', ...
        'LineWidth', 1.5);
    
    text(x + w/2, y + h/2, text_str, ...
        'Parent', ax, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 12, ...
        'FontName', 'Microsoft YaHei', ...
        'Color', 'white');
end

function draw_arrow(fig, x1, y1, x2, y2)
    annotation(fig, 'arrow', ...
        [x1/100, x2/100], ...
        [y1/100, y2/100], ...
        'LineWidth', 1.5, ...
        'Color', [0.3 0.3 0.3]);
end