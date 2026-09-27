function save_figure(fig, filename, config)
    output_path = fullfile(config.output_dir, filename);
    saveas(fig, output_path, 'png');
    saveas(fig, output_path, 'fig');
    fprintf('Saved: %s\n', output_path);
end