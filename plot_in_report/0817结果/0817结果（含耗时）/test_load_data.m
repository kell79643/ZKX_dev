function test_load_data()
% test_load_data  RED-phase test: validate load_current_run_plot_data against 03_任务结果
% 验证维度: cpu_mean_ms, cpp_compute_mean_ms, cpu_speedup, rmse 均为 11x1，数值合理

    fprintf('=== test_load_data starts ===\n');
    runDataDir = fullfile(fileparts(mfilename('fullpath')), '03_任务结果');
    assert(exist(runDataDir,'dir')==7, 'runDataDir not found: %s', runDataDir);
    
    data = load_current_run_plot_data(runDataDir);
    
    % 1. 检查关键字段存在
    reqFields = {'cpu_mean_ms','cpp_compute_mean_ms','cpu_speedup','rmse',...
        'run_id','task1_sha','task2_sha','warmup','repeats','run_dir'};
    for f = reqFields
        assert(isfield(data, f{1}), 'Missing field: %s', f{1});
    end
    fprintf('[PASS] All required fields present.\n');
    
    % 2. 维度检查 (11 steps: T1 S1-S5 + T2 S1-S6)
    assert(numel(data.cpu_mean_ms)==11, 'cpu_mean_ms must have 11 steps, got %d', numel(data.cpu_mean_ms));
    assert(numel(data.cpp_compute_mean_ms)==11, 'cpp_compute_mean_ms must have 11 steps, got %d', numel(data.cpp_compute_mean_ms));
    assert(numel(data.cpu_speedup)==11, 'cpu_speedup must have 11 steps, got %d', numel(data.cpu_speedup));
    assert(numel(data.rmse)==11, 'rmse must have 11 steps, got %d', numel(data.rmse));
    fprintf('[PASS] All arrays have 11 elements.\n');
    
    % 3. 数值合理检查
    assert(all(data.cpu_mean_ms(:) > 0), 'cpu_mean_ms has non-positive value');
    assert(all(data.cpp_compute_mean_ms(:) > 0), 'cpp_compute_mean_ms has non-positive value');
    assert(all(data.cpu_speedup(:) > 0), 'cpu_speedup has non-positive value');
    assert(all(data.rmse(:) >= 0), 'rmse has negative value');
    fprintf('[PASS] All timing/speedup positive; rmse non-negative.\n');
    
    % 4. cpu_speedup 一致性: cpu_speedup ≈ cpu_mean_ms ./ cpp_compute_mean_ms
    expectedSpeedup = data.cpu_mean_ms(:) ./ data.cpp_compute_mean_ms(:);
    maxRelErr = max(abs(data.cpu_speedup(:) - expectedSpeedup) ./ expectedSpeedup);
    assert(maxRelErr < 1e-6, 'cpu_speedup mismatch: max rel error = %g', maxRelErr);
    fprintf('[PASS] cpu_speedup consistent with cpu_mean_ms ./ cpp_compute_mean_ms.\n');
    
    % 5. 元数据非空
    assert(strlength(string(data.run_id)) > 0, 'run_id empty');
    assert(strlength(string(data.task1_sha)) > 0, 'task1_sha empty');
    assert(strlength(string(data.task2_sha)) > 0, 'task2_sha empty');
    assert(data.warmup > 0, 'warmup must be > 0');
    assert(data.repeats > 0, 'repeats must be > 0');
    fprintf('[PASS] Metadata non-empty. run_id=%s, sha1=%s, sha2=%s, warmup=%d, repeats=%d\n', ...
        data.run_id, data.task1_sha(1:min(12,end)), data.task2_sha(1:min(12,end)), ...
        data.warmup, data.repeats);
    
    % 6. 分步摘要打印
    labels = {
        'T1-S1 (pulse_compression)'
        'T1-S2 (pulse_doppler)'
        'T1-S3 (cfar_alpha+ca_cfar)'
        'T1-S4 (step4 final)'
        'T1-S5 (ambiguity function)'
        'T2-S1 (signal generation)'
        'T2-S2 (firwin+firfilter)'
        'T2-S3 (cubic interp)'
        'T2-S4 (correlation + spectral + wavelet + FM demod)'
        'T2-S5 (argrelextrema)'
        'T2-S6 (Kalman filter)'
    };
    fprintf('\n=== Step Summary (medians across all 110 cases) ===\n');
    fprintf('%-5s  %-18s  %12s  %12s  %10s  %12s\n', ...
        'Idx', 'Step', 'CPU ms', 'C++ GPU ms', 'Speedup', 'RMSE');
    for i = 1:11
        fprintf('  %-3d  %-18s  %12.4f  %12.4f  %8.2fx  %12.4e\n', ...
            i, labels{i}, data.cpu_mean_ms(i), data.cpp_compute_mean_ms(i), ...
            data.cpu_speedup(i), data.rmse(i));
    end
    
    fprintf('\n=== ALL TESTS PASSED ===\n');
end
