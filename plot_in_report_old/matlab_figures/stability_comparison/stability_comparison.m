modules = {'BSplines', 'Convolution', 'Demod', 'Radartools'};
pass_rates = [78.72, 78.9, 100.0, 100.0];
total_tests = [47, 19, 47, 28];
passed_tests = [37, 15, 47, 28];

palette.green = [0.30 0.65 0.45];
palette.red = [0.75 0.35 0.40];
palette.orange = [0.80 0.50 0.35];

fig1 = figure('Color', 'w', 'Position', [100, 100, 900, 600]);
bar_width = 0.6;

colors = [palette.red; palette.red; palette.green; palette.green];

hBars = bar(1:4, pass_rates, bar_width, 'FaceColor', 'flat');
for i = 1:4
    if pass_rates(i) < 100
        hBars.CData(i,:) = palette.red;
    else
        hBars.CData(i,:) = palette.green;
    end
end

hold on;

for i = 1:4
    text(i, pass_rates(i) + 2, sprintf('%d/%d', passed_tests(i), total_tests(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

ylim([0 115]);
ylabel('Pass Rate (%)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Module', 'FontSize', 14, 'FontWeight', 'bold');
title('CUDA Modules Stability Comparison (Overall Pass Rate)', ...
    'FontSize', 16, 'FontWeight', 'bold');

set(gca, 'XTick', 1:4, 'XTickLabel', modules, 'FontSize', 12);
grid on;

threshold_line = yline(80, '--', 'DisplayName', '80% Threshold');
threshold_line.Color = palette.orange;
threshold_line.LineWidth = 2;

legend_objects = [patch(NaN, NaN, 'FaceColor', palette.green), ...
                  patch(NaN, NaN, 'FaceColor', palette.red), ...
                  threshold_line];
legend(legend_objects, {'Pass (100%)', 'Fail (<100%)', 'Threshold (80%)'}, ...
    'Location', 'northeast', 'FontSize', 11);

for i = 1:4
    if pass_rates(i) >= 100
        y_value = 50;
        text(i, y_value, '100%', 'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', palette.green);
    end
end

hold off;

baseDir = fileparts(mfilename('fullpath'));
outDir = fullfile(baseDir, 'output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

saveas(fig1, fullfile(outDir, 'stability_overall.fig'));
exportgraphics(fig1, fullfile(outDir, 'stability_overall.png'), 'Resolution', 300);
close(fig1);

fprintf('Stability chart saved to %s\\stability_overall.png\n', outDir);