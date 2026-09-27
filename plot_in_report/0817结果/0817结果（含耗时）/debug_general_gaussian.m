function debug_general_gaussian()
% 调试 general_gaussian 为何显示 NaN
dataDir = 'c:\Users\asus\Desktop\0817结果（含耗时）\04_算子结果\绘图数据';
accuracyFile = fullfile(dataDir, '53x5_最差精度候选.csv');
T = readtable(accuracyFile);

if ~isprop(T, 'operator_name')
    T.Properties.VariableNames{1} = 'operator_name';
end

fprintf('max_rmse class: %s\n', class(T.max_rmse));
fprintf('dtype class: %s\n', class(T.dtype));

idx = strcmp(T.operator_name, 'general_gaussian');
fprintf('rows: %d\n', sum(idx));
fprintf('dtypes: '); disp(T.dtype(idx));
fprintf('rmse: '); disp(T.max_rmse(idx));

if iscell(T.max_rmse)
    nNA = sum(strcmp(T.max_rmse, 'NA') | strcmp(T.max_rmse, 'NaN'));
    fprintf('NA count in max_rmse: %d / %d\n', nNA, height(T));
else
    fprintf('NaN count: %d / %d\n', sum(isnan(T.max_rmse)), height(T));
end

fprintf('unique dtypes:\n');
ud = unique(T.dtype);
for k = 1:numel(ud)
    fprintf('  [%s] %d\n', char(ud(k)), sum(strcmp(T.dtype, ud(k))));
end
end
