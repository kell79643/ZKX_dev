% 核心必做算子 API 接口对照表
% 运行本脚本可生成普通绘图对象表格，并导出 PNG。

clear; clc;

outDir = fileparts(mfilename('fullpath'));
csvPath = fullfile(outDir, 'core_required_operator_api_table.csv');

T = readtable(csvPath, 'TextType', 'string');

headers = ["任务", "模块/环节", "算子函数", "核心输入参数", "输出结果", "对齐情况"];
cols = [T.Task, T.Stage, T.Operator, T.CoreInputs, T.Output, T.Alignment];

fig = figure('Color', 'w', 'Position', [100 100 2400 1400]);
ax = axes(fig, 'Position', [0 0 1 1]);
axis(ax, 'off');
xlim(ax, [0 1]);
ylim(ax, [0 1]);

text(ax, 0.02, 0.965, "核心必做算子 API 接口对照表", ...
    'FontSize', 22, 'FontWeight', 'bold', 'Interpreter', 'none');
text(ax, 0.02, 0.935, ...
    "基准：Python cuSignal 23.08.00；一致=host接口基本同形，语义对齐=CUDA/C++实现形态等价映射", ...
    'FontSize', 11, 'Color', [0.25 0.25 0.25], 'Interpreter', 'none');

left = 0.02;
right = 0.98;
top = 0.895;
bottom = 0.065;
width = right - left;
nRows = height(T) + 1;
rowH = (top - bottom) / nRows;
colW = [0.055, 0.105, 0.17, 0.255, 0.19, 0.225];
colX = left + [0, cumsum(colW(1:end-1))] * width;
colAbsW = colW * width;

rectangle(ax, 'Position', [left bottom width top-bottom], ...
    'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 1.0);

for r = 1:nRows
    y = top - r * rowH;
    if r == 1
        face = [0.90 0.93 0.97];
    elseif mod(r, 2) == 0
        face = [1 1 1];
    else
        face = [0.97 0.98 0.99];
    end
    rectangle(ax, 'Position', [left y width rowH], ...
        'FaceColor', face, 'EdgeColor', [0.82 0.82 0.82], 'LineWidth', 0.5);
end

for c = 1:numel(headers)
    x = colX(c);
    rectangle(ax, 'Position', [x bottom colAbsW(c) top-bottom], ...
        'EdgeColor', [0.82 0.82 0.82], 'LineWidth', 0.5);
    text(ax, x + 0.004, top - rowH * 0.62, char(headers(c)), ...
        'FontSize', 10.5, 'FontWeight', 'bold', 'Interpreter', 'none', ...
        'VerticalAlignment', 'middle');
end

for r = 1:height(T)
    y = top - (r + 1) * rowH + rowH * 0.5;
    for c = 1:numel(headers)
        value = char(cols(r, c));
        value = wrapText(value, max(8, floor(colAbsW(c) * 95)));
        text(ax, colX(c) + 0.004, y, value, ...
            'FontSize', 8.6, 'Interpreter', 'none', ...
            'VerticalAlignment', 'middle', ...
            'HorizontalAlignment', 'left');
    end
end

note = "注：GPU device 路径已覆盖可调用和计时，完整逐元素 CPU reference vs device 一致性矩阵仍需补齐。";
text(ax, 0.02, 0.028, note, ...
    'FontSize', 10, 'Color', [0.35 0.35 0.35], 'Interpreter', 'none');

pngPath = fullfile(outDir, 'fig_core_required_operator_api_table.png');
if exist('exportgraphics', 'file')
    exportgraphics(fig, pngPath, 'Resolution', 200);
else
    print(fig, pngPath, '-dpng', '-r200');
end

function out = wrapText(s, maxChars)
    s = string(s);
    if strlength(s) <= maxChars
        out = char(s);
        return;
    end
    parts = split(s, ["；", "; ", "/ "]);
    lines = strings(0);
    cur = "";
    for i = 1:numel(parts)
        piece = strtrim(parts(i));
        if piece == ""
            continue;
        end
        if cur == ""
            cur = piece;
        elseif strlength(cur + "；" + piece) <= maxChars
            cur = cur + "；" + piece;
        else
            lines(end + 1) = cur; %#ok<AGROW>
            cur = piece;
        end
    end
    if cur ~= ""
        lines(end + 1) = cur;
    end
    out = char(join(lines, newline));
end
