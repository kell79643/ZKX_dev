function config = plot_common()
    config = struct();
    
    config.font_size = 14;
    config.font_size_small = 12;
    config.font_size_large = 16;
    config.font_name = 'Microsoft YaHei';
    
    config.line_width = 1.5;
    config.line_width_thick = 2.0;
    config.marker_size = 8;
    config.bar_width = 0.8;
    
    config.colors = struct();
    config.colors.gpu = [0.25 0.41 0.88];
    config.colors.cpu = [0.85 0.33 0.10];
    config.colors.python = [0.93 0.69 0.13];
    config.colors.cpp = [0.15 0.68 0.38];
    config.colors.h2d = [0.25 0.41 0.88];
    config.colors.compute = [0.15 0.68 0.38];
    config.colors.d2h = [0.93 0.69 0.13];
    config.colors.prepare = [0.55 0.27 0.07];
    config.colors.postprocess = [0.58 0.40 0.74];
    
    config.colors.dtypes = struct();
    config.colors.dtypes.FP32 = [0.25 0.41 0.88];
    config.colors.dtypes.FP16 = [0.58 0.40 0.74];
    config.colors.dtypes.INT32 = [0.15 0.68 0.38];
    config.colors.dtypes.INT16 = [0.93 0.69 0.13];
    config.colors.dtypes.INT8 = [0.85 0.33 0.10];
    
    config.figure_width = 12;
    config.figure_height = 8;
    config.figure_height_tall = 10;
    config.figure_height_wide = 6;
    
    config.margin_left = 0.15;
    config.margin_right = 0.05;
    config.margin_bottom = 0.15;
    config.margin_top = 0.05;
    
    config.output_dir = 'output';
    config.data_dir = '交付(gengxin1)/02_技术结果/图表数据/正式数据';
    
    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end
    
    set(0, 'DefaultTextFontName', config.font_name);
    set(0, 'DefaultAxesFontName', config.font_name);
    set(0, 'DefaultLegendFontName', config.font_name);
    set(0, 'DefaultTextFontSize', config.font_size);
    set(0, 'DefaultAxesFontSize', config.font_size);
    set(0, 'DefaultLegendFontSize', config.font_size_small);
    set(0, 'DefaultLineLineWidth', config.line_width);
    set(0, 'DefaultLineMarkerSize', config.marker_size);
end

function apply_style(ax, config)
    set(ax, 'FontName', config.font_name);
    set(ax, 'LineWidth', config.line_width);
    box(ax, 'on');
end

function save_figure(fig, filename, config)
    output_path = fullfile(config.output_dir, [filename, '.png']);
    print(fig, output_path, '-dpng', '-r300');
end