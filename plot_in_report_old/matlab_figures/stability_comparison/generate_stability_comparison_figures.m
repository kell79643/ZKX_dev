function generate_stability_comparison_figures()
    fprintf('=== 生成全维度测试指标横向对比图表 ===\n\n');

    fprintf('[1/8] 各模块总体通过率对比...\n');
    fig_stability_1_module_pass_rate();

    fprintf('[2/8] 各模块分维度通过率对比...\n');
    fig_stability_2_dimensional_pass_rate();

    fprintf('[3/8] 各算子MSE与通过率对比...\n');
    fig_stability_3_mse_comparison();

    fprintf('[4/8] 各算子性能对比...\n');
    fig_stability_4_performance_compare();

    fprintf('[5/8] 错误码与稳定性对比...\n');
    fig_stability_5_error_stability();

    fprintf('[6/8] 综合看板...\n');
    fig_stability_6_summary_dashboard();

    fprintf('[7/8] 雷达图...\n');
    fig_stability_7_radar_chart();

    fprintf('[8/8] 热力图...\n');
    fig_stability_8_heatmap();

    fprintf('\n=== 所有图表已生成完毕 ===\n');
    [~, outDir] = StabilityTestCommon.init();
    fprintf('输出目录: %s\n', outDir);
end
