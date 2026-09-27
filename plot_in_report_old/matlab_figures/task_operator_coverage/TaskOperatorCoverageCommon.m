classdef TaskOperatorCoverageCommon
    methods(Static)
        function [outDir, palette] = init()
            baseDir = fileparts(mfilename('fullpath'));
            outDir = fullfile(baseDir, 'output');
            if ~exist(outDir, 'dir')
                mkdir(outDir);
            end

            fontName = TaskOperatorCoverageCommon.pickChineseFont();
            set(groot, 'defaultAxesFontName', fontName);
            set(groot, 'defaultTextFontName', fontName);
            set(groot, 'defaultAxesFontSize', 10.5);
            set(groot, 'defaultLineLineWidth', 1.8);
            set(groot, 'defaultFigureColor', [1 1 1]);
            set(groot, 'defaultAxesColor', [1 1 1]);
            set(groot, 'defaultAxesXColor', [0.12 0.12 0.12]);
            set(groot, 'defaultAxesYColor', [0.12 0.12 0.12]);

            palette.blue = [0.33 0.50 0.70];
            palette.cyan = [0.25 0.65 0.68];
            palette.red = [0.78 0.30 0.30];
            palette.orange = [0.82 0.55 0.28];
            palette.green = [0.36 0.58 0.43];
            palette.purple = [0.48 0.42 0.64];
            palette.gray = [0.38 0.38 0.38];
            palette.dark = [0.12 0.18 0.24];
            palette.lightBlue = [0.90 0.94 0.98];
            palette.lightRed = [0.98 0.91 0.90];
            palette.lightGreen = [0.91 0.96 0.92];
            palette.lightGray = [0.96 0.96 0.96];
            palette.lightOrange = [0.98 0.94 0.88];
            palette.lightPurple = [0.94 0.92 0.97];
            palette.matrix = [0.94 0.94 0.94; palette.green; palette.orange; palette.red];
        end

        function fontName = pickChineseFont()
            fonts = listfonts;
            candidates = {'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', ...
                'Source Han Sans SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS'};
            fontName = 'Helvetica';
            for i = 1:numel(candidates)
                if any(strcmpi(fonts, candidates{i}))
                    fontName = candidates{i};
                    return;
                end
            end
        end

        function spec = operatorSpec()
            ops = {'pulse_compression','pulse_doppler','ca_cfar','cfar_alpha','ambgfun', ...
                'chirp','gausspulse','sawtooth','square','firwin','firfilter','cubic', ...
                'quadratic','gauss_spline','fm_demod','correlate','lombscargle','cwt', ...
                'morlet','ricker','kalmanfilter','argrelextrema'}';
            tasks = {'任务一','任务一','任务一','任务一','任务一', ...
                '任务二','任务二','任务二','任务二','任务二','任务二','任务二', ...
                '任务二','任务二','任务二','任务二','任务二','任务二', ...
                '任务二','任务二','任务二','任务二'}';
            groups = {'雷达脉冲处理','雷达脉冲处理','恒虚警检测','恒虚警检测','模糊函数分析', ...
                '波形生成','波形生成','波形生成','波形生成','滤波处理','滤波处理', ...
                '样条与插值','样条与插值','样条与插值','解调与相关','解调与相关', ...
                '谱分析与小波','谱分析与小波','谱分析与小波','谱分析与小波','状态与特征','状态与特征'}';
            labels = {'脉冲压缩','脉冲多普勒','CA-CFAR','CFAR 系数','模糊函数', ...
                '线性调频','高斯脉冲','锯齿波','方波','FIR 窗函数','FIR 滤波', ...
                '三次样条','二次样条','高斯样条','FM 解调','相关运算', ...
                'Lomb-Scargle','连续小波变换','Morlet 小波','Ricker 小波','Kalman 滤波','相对极值'}';
            spec = table(string(ops), string(tasks), string(groups), string(labels), ...
                'VariableNames', {'operator','task','group','label'});
        end

        function T = readConsistency()
            baseDir = fileparts(mfilename('fullpath'));
            matlabFiguresDir = fileparts(baseDir);
            projectRoot = fileparts(matlabFiguresDir);
            candidates = { ...
                fullfile(baseDir, 'operator_consistency_results.csv'), ...
                fullfile(pwd, 'operator_consistency_results.csv'), ...
                fullfile(projectRoot, 'CONSISTENCY', 'operator_consistency_results.csv'), ...
                fullfile(projectRoot, 'CONSISTENCY', 'consistency_figure', 'operator_consistency_results.csv'), ...
                '/workspace/CONSISTENCY/consistency_figure/operator_consistency_results.csv', ...
                '/workspace/CONSISTENCY/operator_consistency_results.csv'};
            csvPath = '';
            for i = 1:numel(candidates)
                if exist(candidates{i}, 'file') == 2
                    csvPath = candidates{i};
                    break;
                end
            end
            if isempty(csvPath)
                error('未找到 operator_consistency_results.csv。请将实测 CSV 放在当前目录，或保留 /workspace/CONSISTENCY 下的结果文件。');
            end
            try
                T = readtable(csvPath, 'TextType', 'string', 'VariableNamingRule', 'preserve');
            catch
                T = readtable(csvPath, 'TextType', 'string');
            end
        end

        function C = coverageTable()
            spec = TaskOperatorCoverageCommon.operatorSpec();
            T = TaskOperatorCoverageCommon.readConsistency();
            n = height(spec);

            pythonRef = false(n,1);
            cpuOutput = false(n,1);
            gpuOutput = false(n,1);
            cpuPyPass = false(n,1);
            gpuCpuPass = false(n,1);
            cpuPyMse = nan(n,1);
            gpuCpuMse = nan(n,1);
            maxMse = nan(n,1);
            avgMse = nan(n,1);

            for i = 1:n
                op = spec.operator(i);
                rows = strcmp(string(T.operator), op);
                cpuPyRows = rows & strcmp(string(T.comparison), "cpu_vs_python");
                gpuCpuRows = rows & strcmp(string(T.comparison), "gpu_vs_cpu");

                pythonRef(i) = TaskOperatorCoverageCommon.hasPath(T, cpuPyRows, 'reference_path');
                cpuOutput(i) = TaskOperatorCoverageCommon.hasPath(T, cpuPyRows, 'actual_path') || ...
                    TaskOperatorCoverageCommon.hasPath(T, gpuCpuRows, 'reference_path');
                gpuOutput(i) = TaskOperatorCoverageCommon.hasPath(T, gpuCpuRows, 'actual_path');

                cpuPyMse(i) = TaskOperatorCoverageCommon.firstNumber(T, cpuPyRows, 'mse');
                gpuCpuMse(i) = TaskOperatorCoverageCommon.firstNumber(T, gpuCpuRows, 'mse');
                cpuPyPass(i) = TaskOperatorCoverageCommon.comparisonPass(T, cpuPyRows);
                gpuCpuPass(i) = TaskOperatorCoverageCommon.comparisonPass(T, gpuCpuRows);

                mseVals = [cpuPyMse(i), gpuCpuMse(i)];
                mseVals = mseVals(~isnan(mseVals));
                if ~isempty(mseVals)
                    maxMse(i) = max(mseVals);
                    avgMse(i) = mean(mseVals);
                end
            end

            status = [pythonRef, cpuOutput, gpuOutput, cpuPyPass, gpuCpuPass];
            coverageScore = mean(status, 2) * 100;
            fullyCovered = all(status, 2);
            C = [spec table(pythonRef, cpuOutput, gpuOutput, cpuPyPass, gpuCpuPass, ...
                coverageScore, fullyCovered, cpuPyMse, gpuCpuMse, avgMse, maxMse)];
        end

        function tf = hasPath(T, rows, varName)
            tf = false;
            if ~any(rows) || ~ismember(varName, T.Properties.VariableNames)
                return;
            end
            vals = string(T{rows, varName});
            vals = vals(vals ~= "" & ~ismissing(vals));
            tf = ~isempty(vals);
        end

        function val = firstNumber(T, rows, varName)
            val = nan;
            if ~any(rows) || ~ismember(varName, T.Properties.VariableNames)
                return;
            end
            raw = T{rows, varName};
            if isnumeric(raw)
                nums = raw;
            else
                nums = str2double(string(raw));
            end
            nums = nums(~isnan(nums));
            if ~isempty(nums)
                val = nums(1);
            end
        end

        function tf = comparisonPass(T, rows)
            tf = false;
            if ~any(rows)
                return;
            end
            mse = TaskOperatorCoverageCommon.firstNumber(T, rows, 'mse');
            threshold = TaskOperatorCoverageCommon.firstNumber(T, rows, 'threshold');
            if isnan(threshold)
                threshold = 1e-6;
            end
            tf = ~isnan(mse) && mse <= threshold;
        end

        function cmap = continuousMap(palette, n)
            anchors = [0.96 0.96 0.96; palette.lightBlue; palette.blue; palette.purple; palette.red];
            xi = linspace(1, size(anchors,1), n);
            cmap = interp1(1:size(anchors,1), anchors, xi);
        end

        function saveFig(fig, outDir, name)
            exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
            savefig(fig, fullfile(outDir, [name '.fig']));
        end

        function drawBox(pos, titleText, subText, faceColor, edgeColor)
            if nargin < 5
                edgeColor = [0.22 0.22 0.22];
            end
            rectangle('Position', pos, 'Curvature', 0.06, 'FaceColor', faceColor, ...
                'EdgeColor', edgeColor, 'LineWidth', 1.0);
            text(pos(1) + pos(3)/2, pos(2) + pos(4)*0.62, titleText, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.12 0.12 0.12], ...
                'Interpreter', 'none');
            if strlength(string(subText)) > 0
                subFontSize = 8.8;
                if strlength(string(subText)) > 16
                    subFontSize = 7.6;
                end
                text(pos(1) + pos(3)/2, pos(2) + pos(4)*0.31, subText, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', subFontSize, 'Color', [0.24 0.24 0.24], 'Interpreter', 'none');
            end
        end

        function drawArrow(p1, p2)
            hold on;
            quiver(p1(1), p1(2), p2(1)-p1(1), p2(2)-p1(2), 0, ...
                'Color', [0.30 0.30 0.30], 'LineWidth', 1.0, ...
                'MaxHeadSize', 0.18, 'AutoScale', 'off');
        end
    end
end
