modules = {'BSplines', 'Convolution', 'Demod', 'Radartools'};

boundary_rates = [83.3, 66.7, 100.0, 100.0];
exception_rates = [46.2, 66.7, 100.0, 100.0];
stability_rates = [100.0, 100.0, 100.0, 100.0];
resource_rates = [100.0, 0.0, 100.0, 100.0];

fig2 = figure('Position', [100, 100, 1000, 650]);

x = 1:4;
bar_width = 0.18;
gap = 0.02;

b1 = bar(x - 1.5*bar_width - gap, boundary_rates, bar_width, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;
b2 = bar(x - 0.5*bar_width, exception_rates, bar_width, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 0.5);
b3 = bar(x + 0.5*bar_width + gap, stability_rates, bar_width, 'FaceColor', [0.4 0.8 0.4], 'EdgeColor', 'k', 'LineWidth', 0.5);
b4 = bar(x + 1.5*bar_width + 2*gap, resource_rates, bar_width, 'FaceColor', [0.6 0.4 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);

for i = 1:4
    text(x(i) - 1.5*bar_width - gap, boundary_rates(i) + 2, sprintf('%.1f%%', boundary_rates(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(x(i) - 0.5*bar_width, exception_rates(i) + 2, sprintf('%.1f%%', exception_rates(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(x(i) + 0.5*bar_width + gap, stability_rates(i) + 2, sprintf('%.1f%%', stability_rates(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(x(i) + 1.5*bar_width + 2*gap, resource_rates(i) + 2, sprintf('%.1f%%', resource_rates(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

ylim([0 120]);
ylabel('Pass Rate (%)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Module', 'FontSize', 14, 'FontWeight', 'bold');
title('CUDA Modules Robustness Comparison (Stability & Error Handling)', ...
    'FontSize', 16, 'FontWeight', 'bold');

set(gca, 'XTick', x, 'XTickLabel', modules, 'FontSize', 12);
grid on;

threshold_line = yline(80, '--r', 'DisplayName', '80% Threshold');
threshold_line.LineWidth = 2;

legend([b1, b2, b3, b4], ...
    {'Boundary Tests', 'Exception Handling', 'Stability Tests', 'Resource Usage'}, ...
    'Location', 'northeast', 'FontSize', 11);

hold off;

saveas(fig2, '/workspace/matlab_figures/stability_comparison/robustness_comparison.png');
print(fig2, '/workspace/matlab_figures/stability_comparison/robustness_comparison', '-dpng', '-r300');
close(fig2);

fprintf('Robustness chart saved to /workspace/matlab_figures/stability_comparison/robustness_comparison.png\n');