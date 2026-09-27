function fig3_3_task1_end_to_end_mosaic()
[outDir, palette] = SupportFiguresCommon.init();
rng(7);
fig = figure('Color','w','Position',[100 100 1600 700]);
tiledlayout(1,5,'TileSpacing','compact','Padding','compact');
t = linspace(0,1,512);
echo = 0.15*randn(size(t)) + sin(2*pi*40*t).*exp(-((t-0.28)/0.05).^2) + 0.8*sin(2*pi*70*t).*exp(-((t-0.68)/0.04).^2);
nexttile; plot(t,echo,'Color',palette.gray); grid on; title('a1 回波输入');
nexttile; plot(t,abs(SupportFiguresCommon.hilbertLocal(echo)),'Color',palette.blue); grid on; title('a2 脉冲压缩');
nexttile; imagesc(peaks(64)); axis tight; title('a3 Range-Doppler图'); colormap(gca,SupportFiguresCommon.mutedMap(palette));
nexttile; imagesc(abs(peaks(64))>5); axis tight; title('a4 CFAR检测'); colormap(gca,gray);
nexttile; surf(peaks(60),'EdgeColor','none'); view(35,40); title('a5 ambgfun'); colormap(gca,SupportFiguresCommon.mutedMap(palette));
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_3_1_task1_end_to_end_mosaic');
end
