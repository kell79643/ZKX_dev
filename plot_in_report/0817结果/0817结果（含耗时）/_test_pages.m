% 诊断脚本：检查分页逻辑
baseDir = 'c:\Users\asus\Desktop\0817结果（含耗时）\04_算子结果\绘图数据';
csvMapping  = fullfile(baseDir, '53算子_12模块映射.csv');
Tmap = readtable(csvMapping, 'ReadVariableNames', true);
opOrder  = Tmap.operator_name;
nOp = numel(opOrder);
fprintf('nOp = %d\n', nOp);
nPerPage = ceil(nOp / 2);
fprintf('nPerPage = %d\n', nPerPage);
pages = {1:nPerPage, nPerPage+1:nOp};
fprintf('pages{1} = 1:%d (长度 %d)\n', nPerPage, numel(pages{1}));
fprintf('pages{2} = %d:%d (长度 %d)\n', nPerPage+1, nOp, numel(pages{2}));
fprintf('循环将执行 %d 次\n', numel(pages));
