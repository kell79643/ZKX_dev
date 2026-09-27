function apply_style(ax, config)
    set(ax, 'FontName', config.font_name);
    set(ax, 'FontSize', config.font_size);
    set(ax, 'LineWidth', config.line_width);
    set(ax, 'TickLength', [0.02 0.02]);
    
    xlabel_obj = get(ax, 'XLabel');
    if ~isempty(xlabel_obj)
        set(xlabel_obj, 'FontSize', config.font_size_large);
    end
    
    ylabel_obj = get(ax, 'YLabel');
    if ~isempty(ylabel_obj)
        set(ylabel_obj, 'FontSize', config.font_size_large);
    end
    
    title_obj = get(ax, 'Title');
    if ~isempty(title_obj)
        set(title_obj, 'FontSize', config.font_size_large);
    end
end