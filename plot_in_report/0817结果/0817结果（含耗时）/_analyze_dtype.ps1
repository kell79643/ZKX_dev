$csv = Import-Csv "c:\Users\asus\Desktop\0817结果（含耗时）\04_算子结果\绘图数据\全算子_逐case绘图候选.csv"
function Get-Median($arr) {
  $s = $arr | Sort-Object
  $n = $s.Count
  if ($n % 2 -eq 1) { return [double]$s[[int](($n-1)/2)] }
  else { return (([double]$s[$n/2-1] + [double]$s[$n/2]) / 2) }
}
Write-Host "=== Unique timing_scope ==="
$csv | Select-Object -ExpandProperty timing_scope -Unique
Write-Host "`n=== Per-dtype ==="
$csv | Group-Object dtype | ForEach-Object {
  $sp = $_.Group | ForEach-Object { [double]$_.cpu_gpu_speedup }
  $avg = ($sp | Measure-Object -Average).Average
  $med = Get-Median $sp
  $b1 = ($sp | Where-Object { $_ -lt 1 }).Count
  $a1 = ($sp | Where-Object { $_ -gt 1 }).Count
  "{0,-6} cnt={1,-5} avg={2,-10:N4} med={3,-10:N4} <1x:{4} >1x:{5}" -f $_.Name,$_.Count,$avg,$med,$b1,$a1
}
Write-Host "`n=== FP vs INT pooled ==="
$fpsp=@(); $intsp=@()
foreach($r in $csv){ if($r.dtype -like 'FP*'){ $fpsp+=[double]$r.cpu_gpu_speedup } else { $intsp+=[double]$r.cpu_gpu_speedup } }
"FP(16+32)    cnt={0} avg={1:N4} med={2:N4}" -f $fpsp.Count, (($fpsp|Measure-Object -Average).Average), (Get-Median $fpsp)
"INT(8+16+32) cnt={0} avg={1:N4} med={2:N4}" -f $intsp.Count, (($intsp|Measure-Object -Average).Average), (Get-Median $intsp)
Write-Host "`n=== Per scale (order_of_magnitude) ==="
$csv | Group-Object order_of_magnitude | ForEach-Object {
  $sp = $_.Group | ForEach-Object { [double]$_.cpu_gpu_speedup }
  $avg = ($sp | Measure-Object -Average).Average
  $med = Get-Median $sp
  $b1 = ($sp | Where-Object { $_ -lt 1 }).Count
  "{0,-8} cnt={1,-5} avg={2,-10:N4} med={3,-10:N4} <1x:{4}" -f $_.Name,$_.Count,$avg,$med,$b1
}
Write-Host "`n=== dtype x scale avg_sp ==="
$csv | Group-Object dtype, order_of_magnitude | Sort-Object Name | ForEach-Object {
  $sp = $_.Group | ForEach-Object { [double]$_.cpu_gpu_speedup }
  $avg = ($sp | Measure-Object -Average).Average
  "{0,-24} cnt={1,-4} avg={2,-10:N4}" -f $_.Name, $_.Count, $avg
}
Write-Host "`n=== overall ==="
$allsp = $csv | ForEach-Object { [double]$_.cpu_gpu_speedup }
"ALL cnt={0} avg={1:N4} med={2:N4}" -f $allsp.Count, (($allsp|Measure-Object -Average).Average), (Get-Median $allsp)
