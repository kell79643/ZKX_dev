function fig10_latency_boxplot()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_performance.csv');
    perf_data = readtable(data_path);
    
    fp32_data = perf_data(strcmp(perf_data.input_dtype, 'FP32'), :);
    
    select_ops = {'stft', 'spectrogram', 'hilbert', 'hilbert2', 'kaiser', 'pulse_compression', 'cwt', 'resample', 'firwin2', 'ambgfun'};
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    ax = axes('Parent', fig);
    
    means = zeros(1, length(select_ops));
    stds = zeros(1, length(select_ops));
    labels = {};
    
    idx = 1;
    for i = 1:length(select_ops)
        op_data = fp32_data(strcmp(fp32_data.operator, select_ops{i}), :);
        if ~isempty(op_data)
            means(idx) = op_data.gpu_mean_ms;
            stds(idx) = op_data.gpu_stddev_ms;
            labels{idx} = select_ops{i};
            idx = idx + 1;
        end
    end
    
    means = means(1:idx-1);
    stds = stds(1:idx-1);
    
    x = 1:length(means);
    bar(ax, x, means, config.bar_width, 'FaceColor', config.colors.gpu, 'EdgeColor', 'black');
    
    hold(ax, 'on');
    errorbar(x, means, stds, 'Color', 'black', 'LineStyle', 'none', 'CapSize', 5);
    
    set(ax, 'XTick', x);
    set(ax, 'XTickLabel', labels);
    set(ax, 'XTickLabelRotation', 45);
    set(ax, 'FontSize', config.font_size_small);
    set(ax, 'LineWidth', config.line_width);
    set(ax, 'YScale', 'log');
    
    xlabel('算子名称', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    ylabel('GPU延迟 (ms)', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    
    apply_style(ax, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 统计次数: 100 | 数据来源: operator_performance.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end