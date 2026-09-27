function fig3_3_task2_end_to_end_mosaic()
[outDir, palette] = SupportFiguresCommon.init();
rng(11);
fig = figure('Color','w','Position',[100 100 1600 700]);
tiledlayout(1,5,'TileSpacing','compact','Padding','compact');
t = linspace(0,1,512);
sig = SupportFiguresCommon.chirpLocal(t,20,1,120) + 0.2*randn(size(t));
smoothSig = movmean(sig,15);
nexttile; plot(t,sig,'Color',palette.gray); grid on; title('b1 含噪波形');
nexttile; plot(t,smoothSig,'Color',palette.green); grid on; title('b2 滤波+样条');
nexttile; SupportFiguresCommon.spectrogramLocal(sig, palette); title('b3 时频谱');
nexttile; plot(t,smoothSig,'Color',palette.green); hold on; [~,locs]=SupportFiguresCommon.findpeaksLocal(smoothSig,20); plot(t(locs),smoothSig(locs),'o','Color',palette.red,'MarkerFaceColor',palette.lightRed); grid on; title('b4 峰值定位');
nexttile; plot(cumsum(0.02*randn(1,80))+linspace(0,1,80), cumsum(0.02*randn(1,80))+linspace(0,0.55,80), 'Color',palette.blue); grid on; title('b5 Kalman轨迹');
SupportFiguresCommon.saveFig(fig, outDir, 'fig3_3_2_task2_end_to_end_mosaic');
end
