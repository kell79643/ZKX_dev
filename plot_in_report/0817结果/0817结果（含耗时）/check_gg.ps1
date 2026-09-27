$csv = Import-Csv 'c:\Users\asus\Desktop\0817结果（含耗时）\04_算子结果\绘图数据\53x5_最差精度候选.csv'
$gg = $csv | Where-Object { $_.operator_name -eq 'general_gaussian' }
$gg | ForEach-Object { Write-Output ('{0} max_rmse=[{1}]' -f $_.dtype, $_.max_rmse) }
