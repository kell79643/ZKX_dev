function fig1_framework()
    config = plot_common();
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    set(ax, 'Visible', 'off');
    set(ax, 'XLim', [0 100]);
    set(ax, 'YLim', [0 100]);
    
    box_width = 18;
    box_height = 10;
    
    x_start = 10;
    y_start = 45;
    
    boxes = struct();
    boxes.cusignal = [x_start, y_start, box_width, box_height];
    boxes.analysis = [x_start + 25, y_start, box_width, box_height];
    boxes.kernel = [x_start + 50, y_start, box_width, box_height];
    boxes.library = [x_start + 75, y_start, box_width, box_height];
    boxes.task = [x_start + 42.5, y_start - 25, box_width, box_height];
    
    colors = [0.25 0.41 0.88; 0.58 0.40 0.74; 0.15 0.68 0.38; 0.93 0.69 0.13; 0.85 0.33 0.10];
    
    draw_box(ax, boxes.cusignal, colors(1,:), strcat('cuSignal Python', char(10), '算法'));
    draw_box(ax, boxes.analysis, colors(2,:), strcat('算子分析', char(10), '与抽象'));
    draw_box(ax, boxes.kernel, colors(3,:), strcat('CUDA Kernel', char(10), '实现'));
    draw_box(ax, boxes.library, colors(4,:), 'GPU算子库');
    draw_box(ax, boxes.task, colors(5,:), strcat('Task1/Task2', char(10), '任务执行'));
    
    draw_arrow(fig, boxes.cusignal(1)+box_width, y_start+box_height/2, boxes.analysis(1), y_start+box_height/2);
    draw_arrow(fig, boxes.analysis(1)+box_width, y_start+box_height/2, boxes.kernel(1), y_start+box_height/2);
    draw_arrow(fig, boxes.kernel(1)+box_width, y_start+box_height/2, boxes.library(1), y_start+box_height/2);
    
    draw_arrow(fig, boxes.library(1)+box_width/2, y_start, boxes.library(1)+box_width/2, y_start-25+box_height);
    draw_arrow(fig, boxes.cusignal(1)+box_width/2, y_start, boxes.cusignal(1)+box_width/2, y_start-25+box_height);
    
    annotation(fig, 'textbox', [0.05, 0.05, 0.9, 0.05], ...
        'String', '测试环境: ZQ500 GPU | 输入规模: 标准测试集 | dtype: FP32/FP16/INT32/INT16/INT8 | 统计次数: 100', ...
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