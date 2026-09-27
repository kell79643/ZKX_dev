function plot_python_cpp_performance()
    fig = figure('Position', [100, 100, 800, 650], 'Color', 'white');
    
    task1_names = {
        'Step1 脉冲压缩', ...
        'Step2 脉冲多普勒', ...
        'Step3 CFAR检测', ...
        'Step4 模糊函数'
    };
    
    task1_cpp_time = [28.90, 123.80, 261.05, 3771.14];
    task1_python_time = [1465.55, 1556.94, 6909.25, 15609.44];
    
    task2_names = {
        'Step1 波形生成', ...
        'Step2 滤波去噪', ...
        'Step3 B样条平滑', ...
        'Step4 特征提取', ...
        'Step5 峰值查找', ...
        'Step6 Kalman估计'
    };
    
    task2_cpp_time = [2.78, 0.45, 0.60, 15+64+244, 17.84, 50.86];
    task2_python_time = [357.69, 289.14, 10.81, 118.46, 17.41, 148.59];
    
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
    
    plot(x_vals, y_eq, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Equal Performance Line');
    
    scatter(task1_python_time, task1_cpp_time, 180, [0.29, 0.55, 0.86], 'filled', 'MarkerEdgeColor', [0.15, 0.35, 0.60], 'LineWidth', 2, 'DisplayName', 'Task1 (雷达信号处理)');
    
    scatter(task2_python_time, task2_cpp_time, 180, [0.74, 0.37, 0.22], 'filled', 'MarkerEdgeColor', [0.55, 0.25, 0.15], 'LineWidth', 2, 'DisplayName', 'Task2 (后续处理)');
    
    for i = 1:length(all_names)
        text(python_times(i) * 1.18, cpp_times(i) * 1.18, all_names{i}, ...
            'FontSize', 10, 'FontWeight', 'bold', 'FontName', 'SimHei', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    end
    
    hold off;
    
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log');
    
    xlim([10^log_min, 10^log_max]);
    ylim([10^log_min, 10^log_max]);
    
    xlabel('Python cusignal GPU Time (ms)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('C++ GPU Time (ms)', 'FontSize', 14, 'FontWeight', 'bold');
    title('GPU Performance: C++ GPU vs Python cusignal', 'FontSize', 18, 'FontWeight', 'bold');
    
    legend('Location', 'southeast', 'FontSize', 12);
    
    grid on;
    set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.15);
    
    set(gca, 'FontSize', 12);
    
    set(fig, 'PaperPositionMode', 'auto');
    saveas(fig, 'python_cpp_performance_comparison.png');
    disp('图表已保存: python_cpp_performance_comparison.png');
end