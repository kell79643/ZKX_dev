classdef StabilityTestCommon
    methods(Static)
        function [outDir, palette] = init()
            baseDir = fileparts(mfilename('fullpath'));
            outDir = fullfile(baseDir, 'output');
            if ~exist(outDir, 'dir')
                mkdir(outDir);
            end
            set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
            set(groot, 'defaultTextFontName', 'Microsoft YaHei');
            set(groot, 'defaultAxesFontSize', 11);
            set(groot, 'defaultLineLineWidth', 1.8);
            set(groot, 'defaultFigureColor', [1 1 1]);
            set(groot, 'defaultAxesColor', [1 1 1]);
            set(groot, 'defaultAxesXColor', [0 0 0]);
            set(groot, 'defaultAxesYColor', [0 0 0]);
            set(groot, 'defaultAxesZColor', [0 0 0]);
            palette.blue = [0.33 0.50 0.70];
            palette.cyan = [0.25 0.65 0.68];
            palette.red = [0.78 0.30 0.30];
            palette.orange = [0.82 0.55 0.28];
            palette.green = [0.36 0.58 0.43];
            palette.purple = [0.48 0.42 0.64];
            palette.gray = [0.38 0.38 0.38];
            palette.lightBlue = [0.90 0.94 0.98];
            palette.lightRed = [0.98 0.91 0.90];
            palette.lightGreen = [0.91 0.96 0.92];
            palette.lightGray = [0.96 0.96 0.96];
            palette.lightOrange = [0.98 0.94 0.88];
        end

        function data = data()
            data.moduleNames = {'BSplines', 'Convolution', 'Demod', 'Radartools'};

            data.moduleTotalTests = [47, 19, 47, 28];
            data.modulePassTests = [37, 15, 47, 28];
            data.moduleFailTests = [10, 4, 0, 0];
            data.modulePassRate = data.modulePassTests ./ data.moduleTotalTests * 100;

            data.functionalPass = [91.7, 88.9, 100, 100];
            data.boundaryPass = [83.3, 66.7, 100, 100];
            data.exceptionPass = [46.2, 66.7, 100, 100];
            data.stabilityPass = [100, 50, 100, 100];
            data.performancePass = [100, 100, 100, 100];

            data.operatorNames = {'cubic', 'quadratic', 'gauss\_spline', ...
                'correlate1d\_real\_direct', 'correlate1d\_real\_fft', ...
                'correlate1d\_complex\_direct', 'correlate1d\_complex\_fft', ...
                'correlate2d\_real', 'correlate2d\_complex', ...
                'fm\_demod\_1D', 'fm\_demod\_2D', ...
                'CA-CFAR', 'OS-CFAR', 'ambgfun', 'pulse\_compress', 'pulse\_doppler'};
            data.operatorModule = [1,1,1, 2,2,2,2,2,2, 3,3, 4,4,4,4,4];

            data.operatorMSE = [1.01e-32, 1.81e-34, 4.75e-35, ...
                0, 2.13e-14, 7.97, 3.73e-16, 0, 1.18e-16, ...
                1.19e-33, 1.89e-33, ...
                0, 0, 0, 0, 0];

            data.operatorMSELabel = {'1.01e-32', '1.81e-34', '4.75e-35', ...
                '<1e-6', '2.13e-14', '7.97(失败)', '3.73e-16', '<1e-6', '1.18e-16', ...
                '1.19e-33', '1.89e-33', ...
                '<1e-6', '<1e-6', '<1e-6', '<1e-6', '<1e-6'};

            data.operatorPassRate = [100, 100, 100, ...
                100, 100, 0, 100, 100, 100, ...
                100, 100, ...
                100, 100, 100, 100, 100];

            data.perfNames = {'cubic', 'quadratic', 'gauss\_spline', ...
                'correlate', 'fm\_demod\_1D', 'fm\_demod\_2D', ...
                'CA-CFAR', 'OS-CFAR', 'ambgfun', 'pulse\_compress', 'pulse\_doppler'};
            data.perfTimeMs = [0.126, 0.095, 0.121, ...
                0.296, 0.082, 0.092, ...
                0.016, 0.016, 149.29, 4.149, 27.227];
            data.perfModule = [1,1,1, 2, 3,3, 4,4,4,4,4];

            data.demodSpeedup1D = [7.66, 26.30, 13.00, 42.35];
            data.demodSpeedup1DScale = {'1K', '4K', '16K', '65K'};
            data.demodSpeedup2D = [4.89, 24.60, 118.41];
            data.demodSpeedup2DScale = {'64x128', '128x256', '256x512'};

            data.errorCodeCoverage = [44.4, NaN, 100, NaN];

            data.failCategories = {'功能正确性', '边界场景', '异常处理', '稳定性', '性能'};
            data.bsplinesFails = [1, 2, 7, 0, 0];
            data.convolutionFails = [1, 1, 1, 1, 0];
            data.demodFails = [0, 0, 0, 0, 0];
            data.radartoolsFails = [0, 0, 0, 0, 0];
        end

        function saveFig(fig, outDir, name)
            exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
        end
    end
end
