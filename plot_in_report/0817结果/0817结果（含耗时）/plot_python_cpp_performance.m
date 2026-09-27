function plot_python_cpp_performance()
    fig = figure('Position', [100, 100, 800, 650], 'Color', 'white');
    
    task1_names = {
        'S1', ...
        'S2', ...
        'S3', ...
        'S4', ...
        'S5'
    };
    
    task1_cpp_time = [1.46958163, 2.24002196, 1.79690902, 1.49078984, 6.00420832];
    task1_python_time = [4.86855489, 271.60766000, 696.12803812, 7.20071167, 416.70572792];
    
    task2_names = {
        'S1', ...
        'S2', ...
        'S3', ...
        'S4', ...
        'S5', ...
        'S6'
    };
    
    task2_cpp_time = [4.10444440, 0.48716535, 0.56718621, 11.71301968, 0.36969646, 10.10762143];
    task2_python_time = [7.29229230, 55.74835963, 2.25876521, 238.52005609, 3.78376704, 12.16222690];
    
    cpp_times = [task1_cpp_time, task2_cpp_time];
    python_times = [task1_python_time, task2_python_time];
    all_names = [task1_names, task2_names];
    
    min_time = min([cpp_times, python_times]);
    max_time = max([cpp_times, python_times]);
    
    log_min = floor(log10(min_time));
    log_max = ceil(log10(max_time));
    
    x_vals = 10.^(log_min:0.1:log_max);
    y_eq = x_vals;
    
    hold on;
    
    plot(x_vals, y_eq, 'k--', 'LineWidth', 2.5, 'DisplayName', 'Equal Performance Line');
    
    scatter(task1_python_time, task1_cpp_time, 180, [0.487, 0.664, 0.796], 'filled', 'MarkerEdgeColor', [0.35, 0.48, 0.58], 'LineWidth', 2, 'DisplayName', 'Task1 (雷达信号处理)');
    
    scatter(task2_python_time, task2_cpp_time, 180, [0.930, 0.477, 0.416], 'filled', 'MarkerEdgeColor', [0.68, 0.35, 0.30], 'LineWidth', 2, 'DisplayName', 'Task2 (多域特征提取定位)');
    
    n_pts = length(all_names);
    log_px = log10(python_times);
    log_cy = log10(cpp_times);

    n_task1 = length(task1_names);
    label_colors = zeros(n_pts, 3);
    label_colors(1:n_task1, :) = repmat([0.35, 0.48, 0.58], n_task1, 1);
    label_colors(n_task1+1:end, :) = repmat([0.68, 0.35, 0.30], n_pts-n_task1, 1);

    d_step = 0.06;
    dir_offsets = [
        0, d_step;
        0, -d_step;
        d_step, 0;
        -d_step, 0;
        d_step*0.85, d_step*0.85;
        -d_step*0.85, d_step*0.85;
        d_step*0.85, -d_step*0.85;
        -d_step*0.85, -d_step*0.85;
    ];

    label_x = zeros(1, n_pts);
    label_y = zeros(1, n_pts);
    align_h = cell(1, n_pts);
    align_v = cell(1, n_pts);
    
    for i = 1:n_pts
        best_score = -inf;
        best_d = 1;
        
        for d = 1:size(dir_offsets, 1)
            cx = log_px(i) + dir_offsets(d, 1);
            cy = log_cy(i) + dir_offsets(d, 2);
            
            min_d = inf;
            for j = 1:n_pts
                if i == j
                    continue;
                end
                dist = sqrt((cx - log_px(j))^2 + (cy - log_cy(j))^2);
                if dist < min_d
                    min_d = dist;
                end
            end
            
            if min_d > best_score
                best_score = min_d;
                best_d = d;
            end
        end
        
        ox = dir_offsets(best_d, 1);
        oy = dir_offsets(best_d, 2);
        
        label_x(i) = python_times(i) * 10^ox;
        label_y(i) = cpp_times(i) * 10^oy;
        
        if ox > 0
            align_h{i} = 'left';
        elseif ox < 0
            align_h{i} = 'right';
        else
            align_h{i} = 'center';
        end

        if oy > 0
            align_v{i} = 'bottom';
        elseif oy < 0
            align_v{i} = 'top';
        else
            align_v{i} = 'middle';
        end
    end
    
    min_lbl_dist = 0.12;
    for pass = 1:3
        for i = 1:n_pts
            for j = i+1:n_pts
                lx1 = log10(label_x(i));
                ly1 = log10(label_y(i));
                lx2 = log10(label_x(j));
                ly2 = log10(label_y(j));
                dist = sqrt((lx1-lx2)^2 + (ly1-ly2)^2);
                
                if dist < min_lbl_dist
                    for d = 1:size(dir_offsets, 1)
                        cx = log_px(j) + dir_offsets(d, 1);
                        cy = log_cy(j) + dir_offsets(d, 2);
                        
                        new_dist = sqrt((cx-lx1)^2 + (cy-ly1)^2);
                        
                        min_data_d = inf;
                        for k = 1:n_pts
                            if k == j
                                continue;
                            end
                            dk = sqrt((cx-log_px(k))^2 + (cy-log_cy(k))^2);
                            if dk < min_data_d
                                min_data_d = dk;
                            end
                        end
                        
                        if new_dist > min_lbl_dist && min_data_d > 0.06
                            ox = dir_offsets(d, 1);
                            oy = dir_offsets(d, 2);
                            label_x(j) = python_times(j) * 10^ox;
                            label_y(j) = cpp_times(j) * 10^oy;
                            
                            if ox > 0
                                align_h{j} = 'left';
                            elseif ox < 0
                                align_h{j} = 'right';
                            else
                                align_h{j} = 'center';
                            end

                            if oy > 0
                                align_v{j} = 'bottom';
                            elseif oy < 0
                                align_v{j} = 'top';
                            else
                                align_v{j} = 'middle';
                            end
                            break;
                        end
                    end
                end
            end
        end
    end
    
    for i = 1:n_pts
        plot([python_times(i), label_x(i)], [cpp_times(i), label_y(i)], ...
            'Color', [label_colors(i, :), 0.35], 'LineStyle', ':', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    end

    for i = 1:n_pts
        text(label_x(i), label_y(i), all_names{i}, ...
            'FontSize', 10, 'FontWeight', 'bold', 'FontName', 'SimHei', ...
            'Color', label_colors(i, :), ...
            'HorizontalAlignment', align_h{i}, 'VerticalAlignment', align_v{i});
    end
    
    hold off;
    
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log');
    
    xlim([10^log_min, 10^log_max]);
    ylim([10^log_min, 10^log_max]);
    
    xlabel('C++ CPU Time (ms)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('C++ GPU Time (ms)', 'FontSize', 14, 'FontWeight', 'bold');
    title('Performance: C++ GPU vs C++ CPU', 'FontSize', 18, 'FontWeight', 'bold');
    
    legend('Location', 'southeast', 'FontSize', 12);
    
    grid on;
    set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.15);
    
    set(gca, 'FontSize', 12);
    
    set(fig, 'PaperPositionMode', 'auto');
    saveas(fig, 'python_cpp_performance_comparison.png');
    disp('图表已保存: python_cpp_performance_comparison.png');
end