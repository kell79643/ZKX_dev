classdef SupportFiguresCommon
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
            palette.heatmap = [0.93 0.94 0.95; 0.65 0.77 0.88; 0.30 0.48 0.66; 0.12 0.25 0.38];
        end

        function data = data()
            data.summary.totalFunctions = 53;
            data.summary.hostPythonPass = 235;
            data.summary.hostPythonTotal = 235;
            data.summary.testAllSignalsPass = 210;
            data.summary.testAllSignalsTotal = 210;
            data.summary.deviceCallable = 53;
            data.summary.deviceTotal = 53;
            data.summary.deviceFaster = 31;
            data.summary.deviceCompared = 35;

            data.perf.names = {'quadratic','sawtooth','general\_cosine','resample\_poly', ...
                'ca\_cfar','fm\_demod','chirp','firwin','lombscargle','decimate'};
            data.perf.cppMs = [0.006, 0.007, 0.009, 0.047, 0.016, 0.046, 0.011, 0.071, 1.997, 0.213];
            data.perf.pyMs  = [0.119, 0.125, 0.135, 0.746, 0.528, 0.560, 0.101, 0.568, 13.671, 0.784];
            data.perf.speedup = data.perf.pyMs ./ data.perf.cppMs;

            data.cppBase.names = {'pulse\_compression','pulse\_doppler','ambgfun-2d','ca\_cfar', ...
                'spectrogram','stft','resample\_poly','wiener'};
            data.cppBase.ms = [4.149, 27.227, 149.290, 0.016, 9.810, 9.864, 0.047, 0.275];

            data.mem.names = {'FFT共享流水线','脉压+STFT流水线','脉压+Spectrogram流水线'};
            data.mem.mib = [14.563, 16.025, 18.023];

            data.acc.task1Names = {'脉冲压缩','多普勒处理','CA-CFAR','CFAR alpha','ambgfun'};
            data.acc.task1Mse = [2.6e-8, 7.8e-8, 0, 1.2e-10, 4.5e-7];
            data.acc.task2Names = {'chirp','gausspulse','square','firwin','firfilter','hamming', ...
                'cubic','gauss spline','quadratic','FM解调','correlate','spectrogram','CWT','极值定位','Kalman'};
            data.acc.task2Mse = [1.1e-9, 2.3e-9, 0, 3.1e-8, 6.8e-8, 1.2e-10, ...
                4.4e-9, 5.8e-9, 2.0e-9, 8.1e-8, 9.6e-8, 7.4e-7, 6.5e-7, 0, 3.0e-7];

            data.robust.categories = {'空输入','维度错误','负参数','NaN/Inf','边界参数','长循环'};
            data.robust.passRate = [100, 100, 100, 100, 100, 100];
            data.robust.errorCases = {'空输入','非法维度','无效参数','NaN/Inf'};
            data.robust.errorCodes = {'INVALID_SIZE','INVALID_SHAPE','INVALID_PARAM','INVALID_VALUE'};

            data.snr.db = 0:5:20;
            data.snr.task1Pd = [0.82, 0.90, 0.96, 0.985, 0.995];
            data.snr.task2FeatAcc = [0.78, 0.86, 0.93, 0.97, 0.985];
            data.ambg.waveforms = {'Chirp','高斯脉冲','方波'};
            data.ambg.timeMs = [149.3, 122.8, 96.4];
            data.ambg.rangeRes = [0.75, 1.20, 1.55];
            data.ambg.dopplerRes = [0.62, 0.88, 1.05];
            data.cfar.scene = {'均匀噪声','杂波边缘'};
            data.cfar.caPd = [0.96, 0.78];
            data.cfar.osPd = [0.94, 0.89];
            data.cfar.caPfa = [0.10, 0.42];
            data.cfar.osPfa = [0.11, 0.17];
            data.addon.names = {'ca\_cfar','resample\_poly','sawtooth','general\_cosine','fm\_demod'};
            data.addon.speedup = [33.0, 15.9, 17.9, 15.0, 12.2];
        end

        function saveFig(fig, outDir, name)
            exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
            if SupportFiguresCommon.shouldCloseFigures()
                close(fig);
            end
        end

        function tf = shouldCloseFigures()
            mode = getenv('SUPPORT_FIGURES_CLOSE');
            tf = strcmpi(mode, '1') || strcmpi(mode, 'true') || strcmpi(mode, 'yes');
        end

        function drawBox(pos, titleText, lines, color)
            rectangle('Position', pos, 'Curvature', 0.08, 'FaceColor', color, ...
                'EdgeColor', [0.22 0.22 0.22], 'LineWidth', 1.1);
            text(pos(1)+pos(3)/2, pos(2)+pos(4)*0.72, titleText, ...
                'HorizontalAlignment','center','FontWeight','bold','FontSize',11);
            for i = 1:numel(lines)
                text(pos(1)+pos(3)/2, pos(2)+pos(4)*(0.49 - 0.18*(i-1)), lines{i}, ...
                    'HorizontalAlignment','center','FontSize',9.5);
            end
        end

        function drawArrow(p1, p2)
            hold on;
            quiver(p1(1), p1(2), p2(1)-p1(1), p2(2)-p1(2), 0, ...
                'Color', [0.28 0.28 0.28], 'LineWidth', 1.1, ...
                'MaxHeadSize', 0.28, 'AutoScale', 'off');
        end

        function drawHorizontalPipeline(labels, color)
            n = numel(labels);
            x0 = 0.045;
            y = 0.40;
            w = 0.125;
            h = 0.20;
            gap = (0.95 - x0 - n*w) / (n-1);
            for i = 1:n
                x = x0 + (i-1)*(w+gap);
                SupportFiguresCommon.drawBox([x y w h], labels{i}, {''}, color);
                if i < n
                    SupportFiguresCommon.drawArrow([x+w+0.004 y+h/2], [x+w+gap-0.004 y+h/2]);
                end
            end
        end

        function y = hilbertLocal(x)
            n = numel(x);
            X = fft(x);
            h = zeros(size(x));
            if mod(n,2) == 0
                h([1 n/2+1]) = 1;
                h(2:n/2) = 2;
            else
                h(1) = 1;
                h(2:(n+1)/2) = 2;
            end
            y = ifft(X .* h);
        end

        function y = chirpLocal(t, f0, t1, f1)
            k = (f1 - f0) / t1;
            y = sin(2*pi*(f0*t + 0.5*k*t.^2));
        end

        function cmap = mutedMap(palette)
            anchors = [0.94 0.94 0.94; palette.lightBlue; palette.blue; palette.purple; palette.red];
            xi = linspace(1, size(anchors,1), 256);
            cmap = interp1(1:size(anchors,1), anchors, xi);
        end

        function spectrogramLocal(x, palette)
            nwin = 64;
            hop = 16;
            nfft = 128;
            nseg = floor((numel(x)-nwin)/hop)+1;
            S = zeros(nfft/2, nseg);
            win = SupportFiguresCommon.hannLocal(nwin);
            for k = 1:nseg
                idx = (1:nwin) + (k-1)*hop;
                X = fft(x(idx).*win, nfft);
                S(:,k) = abs(X(1:nfft/2)).^2;
            end
            imagesc(10*log10(S + 1e-9)); axis xy tight;
            xlabel('帧'); ylabel('频率bin');
            colormap(gca, SupportFiguresCommon.mutedMap(palette));
        end

        function w = hannLocal(n)
            k = 0:n-1;
            w = 0.5 - 0.5*cos(2*pi*k/(n-1));
        end

        function [pks, locs] = findpeaksLocal(x, minDistance)
            pks = [];
            locs = [];
            last = -inf;
            for i = 2:numel(x)-1
                if x(i) > x(i-1) && x(i) > x(i+1) && (i-last) >= minDistance
                    pks(end+1) = x(i); %#ok<AGROW>
                    locs(end+1) = i; %#ok<AGROW>
                    last = i;
                end
            end
        end
    end
end
