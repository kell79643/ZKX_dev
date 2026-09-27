param(
    [string]$ArchiveRoot,
    [string]$ProfilePath
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$demoRoot = Split-Path -Parent $scriptRoot
$repositoryRoot = Split-Path -Parent $demoRoot
$workspaceRoot = Split-Path -Parent $repositoryRoot
if ([string]::IsNullOrWhiteSpace($ArchiveRoot)) {
    $ArchiveRoot = Join-Path $workspaceRoot '全国总决赛证据归档'
}
if ([string]::IsNullOrWhiteSpace($ProfilePath)) {
    $ProfilePath = Join-Path $demoRoot 'config\runtime\demo_profile.json'
}

$candidatePath = Join-Path $ArchiveRoot '04_算子结果\绘图数据\全算子_逐case绘图候选.csv'
$stageRoots = @{
    stage07 = Join-Path $ArchiveRoot '04_算子结果\task_operators\data\RB_20260810T1446CST_STAGE07'
    stage08 = Join-Path $ArchiveRoot '04_算子结果\remaining_operators\data\RB_20260810T1446CST_STAGE08'
}
$dtypeOrder = @('FP32', 'FP16', 'INT32', 'INT16', 'INT8')

function Convert-RequiredDouble([string]$Value, [string]$Field, [string]$CaseId) {
    $parsed = 0.0
    if (-not [double]::TryParse(
            $Value,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$parsed)) {
        throw "invalid numeric $Field for $CaseId`: $Value"
    }
    return $parsed
}

function Convert-AccuracySortValue([string]$Value, [string]$Field, [string]$CaseId) {
    if ($Value -eq 'NA' -or [string]::IsNullOrWhiteSpace($Value)) {
        return 0.0
    }
    return Convert-RequiredDouble $Value $Field $CaseId
}

function Archive-RelativePath([string]$AbsolutePath) {
    $archive = [IO.Path]::GetFullPath($ArchiveRoot).TrimEnd('\') + '\'
    $target = [IO.Path]::GetFullPath($AbsolutePath)
    $baseUri = [Uri]$archive
    $targetUri = [Uri]$target
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
}

if (-not (Test-Path -LiteralPath $candidatePath -PathType Leaf)) {
    throw "operator candidate table does not exist: $candidatePath"
}
if (-not (Test-Path -LiteralPath $ProfilePath -PathType Leaf)) {
    throw "demo profile does not exist: $ProfilePath"
}

$rows = @(Import-Csv -LiteralPath $candidatePath -Encoding UTF8)
if ($rows.Count -ne 1112) {
    throw "operator candidate row count must be 1112, got $($rows.Count)"
}
$invalid = @($rows | Where-Object {
    $_.run_type -ne 'formal' -or $_.status -ne 'PASS' -or
    $_.accuracy_status -ne 'PASS' -or $_.cpu_row_present -ne 'True' -or
    $_.gpu_row_present -ne 'True'
})
if ($invalid.Count -ne 0) {
    throw "operator candidate admission failed for $($invalid.Count) rows"
}
$operators = @($rows.operator_name | Sort-Object -Unique)
if ($operators.Count -ne 53) {
    throw "operator count must be 53, got $($operators.Count)"
}
$groups = @($rows | Group-Object operator_name, dtype)
if ($groups.Count -ne 265) {
    throw "operator/dtype group count must be 265, got $($groups.Count)"
}

$candidateTableSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidatePath).Hash.ToLowerInvariant()
$operatorEntries = [Collections.Generic.List[object]]::new()
foreach ($operator in ($operators | Sort-Object)) {
    foreach ($dtype in $dtypeOrder) {
        $candidates = @($rows | Where-Object {
            $_.operator_name -eq $operator -and $_.dtype -eq $dtype
        })
        if ($candidates.Count -lt 1) {
            throw "missing candidates for $operator/$dtype"
        }
        $backends = @($candidates.backend | Sort-Object -Unique)
        if ($backends.Count -ne 1 -or $backends[0] -notin @('fft_thrust', 'not_applicable')) {
            throw "invalid backend set for $operator/$dtype`: $($backends -join ',')"
        }
        $ranked = @($candidates | Sort-Object `
            @{Expression={-(Convert-RequiredDouble $_.cpu_gpu_speedup 'cpu_gpu_speedup' $_.case_id)}}, `
            @{Expression={Convert-AccuracySortValue $_.mse 'mse' $_.case_id}}, `
            @{Expression={Convert-AccuracySortValue $_.rmse 'rmse' $_.case_id}}, `
            @{Expression={Convert-AccuracySortValue $_.relative_l2 'relative_l2' $_.case_id}}, `
            @{Expression={Convert-AccuracySortValue $_.relative_linf 'relative_linf' $_.case_id}}, `
            @{Expression={Convert-RequiredDouble $_.gpu_p95_ms 'gpu_p95_ms' $_.case_id}}, `
            @{Expression={$_.case_id}})
        $best = $ranked[0]
        $stageRoot = $stageRoots[$best.stage]
        if ($null -eq $stageRoot) {
            throw "unknown source stage for $($best.case_id): $($best.stage)"
        }
        $sourceCsv = Join-Path $stageRoot ($best.source_relative_path -replace '/', '\')
        if (-not (Test-Path -LiteralPath $sourceCsv -PathType Leaf)) {
            throw "selected source CSV does not exist: $sourceCsv"
        }
        $sourceSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceCsv).Hash.ToLowerInvariant()
        $candidateSet = @($ranked | ForEach-Object {
            [ordered]@{
                case_id = $_.case_id
                scale_id = $_.scale_id
                cpu_gpu_speedup = Convert-RequiredDouble $_.cpu_gpu_speedup 'cpu_gpu_speedup' $_.case_id
                mse = $_.mse
                rmse = $_.rmse
                relative_l2 = $_.relative_l2
                relative_linf = $_.relative_linf
                gpu_p95_ms = Convert-RequiredDouble $_.gpu_p95_ms 'gpu_p95_ms' $_.case_id
                source_relative_path = $_.source_relative_path
            }
        })
        $module = $best.module
        $profileId = 'operator_{0}_{1}_{2}_{3}' -f `
            $module, $operator, $best.backend, $dtype.ToLowerInvariant()
        $operatorEntries.Add([ordered]@{
            profile_entry_id = $profileId
            target_kind = 'operator'
            target = $operator
            module = $module
            backend = $best.backend
            dtype = $dtype
            best_scale_id = $best.scale_id
            case_id = $best.case_id
            run_id = $best.run_id
            source_csv = ('全国总决赛证据归档/{0}' -f (Archive-RelativePath $sourceCsv).Replace('\','/'))
            source_csv_sha256 = $sourceSha
            source_candidate_table = '全国总决赛证据归档/04_算子结果/绘图数据/全算子_逐case绘图候选.csv'
            source_candidate_table_sha256 = $candidateTableSha
            selection_metric = 'cpu_gpu_speedup_desc_then_accuracy_asc_then_gpu_p95_asc'
            selection_value = Convert-RequiredDouble $best.cpu_gpu_speedup 'cpu_gpu_speedup' $best.case_id
            selected_metrics = [ordered]@{
                cpu_mean_ms = Convert-RequiredDouble $best.cpu_mean_ms 'cpu_mean_ms' $best.case_id
                gpu_mean_ms = Convert-RequiredDouble $best.gpu_mean_ms 'gpu_mean_ms' $best.case_id
                cpu_gpu_speedup = Convert-RequiredDouble $best.cpu_gpu_speedup 'cpu_gpu_speedup' $best.case_id
                gpu_p95_ms = Convert-RequiredDouble $best.gpu_p95_ms 'gpu_p95_ms' $best.case_id
                mse = $best.mse
                rmse = $best.rmse
                relative_l2 = $best.relative_l2
                relative_linf = $best.relative_linf
            }
            candidate_case_ids = @($ranked.case_id)
            candidate_set = $candidateSet
            selection_reason = '完整候选集均为formal、正常退出、CPU/GPU结果齐全且accuracy_status=PASS；按加速比降序，MSE、RMSE、相对L2、相对Linf升序，GPU p95升序及case_id稳定兜底排序。离散输出的NA误差项不参与区分。'
        })
    }
}

$profile = Get-Content -Raw -Encoding UTF8 -LiteralPath $ProfilePath | ConvertFrom-Json
$taskEntries = @($profile.entries | Where-Object target_kind -eq 'task')
if ($taskEntries.Count -ne 2) {
    throw "expected two admitted task entries, got $($taskEntries.Count)"
}
$output = [ordered]@{
    schema_version = 2
    profile_id = 'demo_task_and_operator_best_v2'
    coverage = 'task1_task2_and_53x5_operator_best'
    operator_selection_source_sha256 = $candidateTableSha
    entries = @($taskEntries) + @($operatorEntries)
}
$json = $output | ConvertTo-Json -Depth 12
[IO.File]::WriteAllText($ProfilePath, $json + "`n", [Text.UTF8Encoding]::new($false))
Write-Output (
    '[DEMO][OPERATOR_BEST] candidates={0} operators={1} dtypes=5 entries={2} status=PASS output={3}' -f
    $rows.Count, $operators.Count, $operatorEntries.Count, $ProfilePath
)
