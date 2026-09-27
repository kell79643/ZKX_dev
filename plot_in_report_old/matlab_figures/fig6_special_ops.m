function fig6_special_ops()
    config = plot_common();
    
    data_path = fullfile(config.data_dir, 'operator_accuracy.csv');
    accuracy_data = readtable(data_path);
    
    fft_ops = {'stft', 'istft', 'spectrogram', 'hilbert', 'hilbert2', 'csd'};
    cfar_ops = {'ca_cfar', 'cfar_alpha', 'pulse_doppler', 'pulse_compression'};
    window_ops = {'hamming', 'kaiser', 'chebwin', 'firwin', 'firwin2', 'taylor'};
    
    fig = figure('Position', [100, 100, config.figure_width*100, config.figure_height*100]);
    set(fig, 'Color', 'white');
    
    fft_data = accuracy_data(ismember(accuracy_data.operator, fft_ops), :);
    cfar_data = accuracy_data(ismember(accuracy_data.operator, cfar_ops), :);
    window_data = accuracy_data(ismember(accuracy_data.operator, window_ops), :);
    
    ax1 = subplot(1, 3, 1);
    bar(ax1, log10(fft_data.max_abs_error + eps), 'FaceColor', config.colors.gpu, 'EdgeColor', 'black');
    set(ax1, 'XTick', 1:length(fft_ops));
    set(ax1, 'XTickLabel', fft_ops);
    set(ax1, 'XTickLabelRotation', 45);
    xlabel('FFT/频谱算子', 'FontSize', config.font_size, 'FontName', config.font_name);
    ylabel('log_{10}(最大绝对误差)', 'FontSize', config.font_size, 'FontName', config.font_name);
    title('FFT/频谱类算子', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    apply_style(ax1, config);
    
    ax2 = subplot(1, 3, 2);
    bar(ax2, cfar_data.mismatch_count, 'FaceColor', config.colors.cpu, 'EdgeColor', 'black');
    set(ax2, 'XTick', 1:length(cfar_ops));
    set(ax2, 'XTickLabel', cfar_ops);
    set(ax2, 'XTickLabelRotation', 45);
    xlabel('CFAR检测算子', 'FontSize', config.font_size, 'FontName', config.font_name);
    ylabel('Mismatch数量', 'FontSize', config.font_size, 'FontName', config.font_name);
    title('CFAR检测类算子', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    apply_style(ax2, config);
    
    ax3 = subplot(1, 3, 3);
    bar(ax3, log10(window_data.max_abs_error + eps), 'FaceColor', config.colors.python, 'EdgeColor', 'black');
    set(ax3, 'XTick', 1:length(window_ops));
    set(ax3, 'XTickLabel', window_ops);
    set(ax3, 'XTickLabelRotation', 45);
    xlabel('窗函数/滤波系数算子', 'FontSize', config.font_size, 'FontName', config.font_name);
    ylabel('log_{10}(最大绝对误差)', 'FontSize', config.font_size, 'FontName', config.font_name);
    title('窗函数/滤波系数类', 'FontSize', config.font_size_large, 'FontName', config.font_name);
    apply_style(ax3, config);
    
    annotation(fig, 'textbox', [0.05, 0.02, 0.9, 0.03], ...
        'String', '测试环境: ZQ500 GPU | dtype: FP32 | 数据来源: operator_accuracy.csv', ...
        'FontSize', config.font_size_small, 'FontName', config.font_name, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end