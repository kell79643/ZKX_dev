param(
    [string]$ArchiveRoot
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$demoRoot = Split-Path -Parent $scriptRoot
$repositoryRoot = Split-Path -Parent $demoRoot
$workspaceRoot = Split-Path -Parent $repositoryRoot
if ([string]::IsNullOrWhiteSpace($ArchiveRoot)) {
    $ArchiveRoot = Join-Path $workspaceRoot '全国总决赛证据归档'
}
$configBuildRoot = Join-Path $demoRoot 'config\build'
$configRuntimeRoot = Join-Path $demoRoot 'config\runtime'
$profilePath = Join-Path $configRuntimeRoot 'demo_profile.json'
$runtimePath = Join-Path $configRuntimeRoot 'demo_runtime.json'

function Write-Json([string]$Path, [object]$Value, [int]$Depth = 12) {
    $json = $Value | ConvertTo-Json -Depth $Depth
    [IO.File]::WriteAllText($Path, $json + "`n", [Text.UTF8Encoding]::new($false))
}

function Relative-RepositoryPath([string]$Path) {
    $base = [Uri]([IO.Path]::GetFullPath($repositoryRoot).TrimEnd('\') + '\')
    $target = [Uri][IO.Path]::GetFullPath($Path)
    return [Uri]::UnescapeDataString($base.MakeRelativeUri($target).ToString())
}

function Parse-CaseTarget([string]$CMakePath) {
    $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $CMakePath
    $match = [regex]::Match($text, '(?m)^\s*add_executable\(\s*([A-Za-z0-9_]+)')
    if (-not $match.Success) {
        $match = [regex]::Match(
            $text,
            '(?m)^\s*(?:operator_case_add_[A-Za-z0-9_]+_target|zkx_add_remaining_operator_case_window_operator)\(\s*([A-Za-z0-9_]+)'
        )
    }
    if (-not $match.Success) {
        throw "cannot resolve executable target from $CMakePath"
    }
    return $match.Groups[1].Value
}

function Backend-Options([string]$Backend) {
    if ($Backend -eq 'dlfft') {
        return @('-DUSE_DLFFT=ON', '-DUSE_THRUST=OFF')
    }
    return @('-DUSE_DLFFT=OFF', '-DUSE_THRUST=ON')
}

function Resolve-ShellRunnerContractPath([string]$RunnerPath) {
    $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $RunnerPath
    $patterns = @(
        '\$\{0%/\*\}/([^"'']+)',
        '\$\(dirname "\$0"\)/([^"'']+)'
    )
    foreach ($pattern in $patterns) {
        $match = [regex]::Match($text, $pattern)
        if ($match.Success) {
            $candidate = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $RunnerPath) $match.Groups[1].Value))
            if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                throw "shell runner delegate does not exist: $candidate"
            }
            return $candidate
        }
    }
    return $RunnerPath
}

function Test-RunnerBackendArgument([string]$RunnerKind, [string]$RunnerRelativePath) {
    if ($RunnerKind -eq 'windows_binary') {
        return $false
    }
    $runner = Join-Path $repositoryRoot $RunnerRelativePath
    if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
        throw "runner does not exist: $RunnerRelativePath"
    }
    if ($RunnerKind -eq 'shell_wrapper') {
        $runner = Resolve-ShellRunnerContractPath $runner
        $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $runner
        return $text.Contains('BACKEND') -or $text.Contains('--backend')
    }
    $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $runner
    $parseIndex = $text.IndexOf('parse_args')
    $parserText = if ($parseIndex -ge 0) {$text.Substring(0, $parseIndex)} else {$text}
    return $parserText.Contains('"--backend"') -or
        $parserText.Contains("'--backend'") -or
        $parserText.Contains('"backend"') -or
        $parserText.Contains("'backend'")
}

if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) {
    throw "missing demo profile: $profilePath"
}
$profile = Get-Content -Raw -Encoding UTF8 -LiteralPath $profilePath | ConvertFrom-Json
$runtime = Get-Content -Raw -Encoding UTF8 -LiteralPath $runtimePath | ConvertFrom-Json
$supportedBackends = @('fft_thrust', 'dlfft')
$sharedBackend = 'not_applicable'
$defaultInputProfileBackend = [string]$runtime.default_input_profile_backend
if ($defaultInputProfileBackend -notin $supportedBackends) {
    throw 'default_input_profile_backend must be one of the supported backends'
}
$operatorProfiles = @($profile.entries | Where-Object target_kind -eq 'operator')
if ($operatorProfiles.Count -ne 265) {
    throw "operator profile count must be 265, got $($operatorProfiles.Count)"
}

# Build a unique operator -> config/threshold mapping from real JSON files.
$configIndex = @{}
$configFiles = Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'test_all\config\operators') `
    -Recurse -File -Filter operator_scales.json
foreach ($configFile in $configFiles) {
    $payload = Get-Content -Raw -Encoding UTF8 -LiteralPath $configFile.FullName | ConvertFrom-Json
    foreach ($operator in @($payload.operators.operator_name)) {
        if ($configIndex.ContainsKey($operator)) {
            throw "duplicate operator config mapping for $operator"
        }
        $thresholds = Join-Path $configFile.DirectoryName 'accuracy_thresholds.json'
        if (-not (Test-Path -LiteralPath $thresholds -PathType Leaf)) {
            throw "missing thresholds for $operator"
        }
        $configIndex[$operator] = [ordered]@{
            config = Relative-RepositoryPath $configFile.FullName
            thresholds = Relative-RepositoryPath $thresholds
        }
    }
}

$operatorCommands = [Collections.Generic.List[object]]::new()
$operatorBuildGroups = [Collections.Generic.List[object]]::new()
$operatorNames = @($operatorProfiles.target | Sort-Object -Unique)
if ($operatorNames.Count -ne 53) {
    throw "operator profile must cover 53 operators"
}
foreach ($operator in $operatorNames) {
    $entries = @($operatorProfiles | Where-Object target -eq $operator)
    $moduleValues = @($entries.module | Sort-Object -Unique)
    $backendValues = @($entries.backend | Sort-Object -Unique)
    if ($entries.Count -ne 5 -or $moduleValues.Count -ne 1 -or $backendValues.Count -ne 1) {
        throw "invalid profile coverage for $operator"
    }
    $module = $moduleValues[0]
    $logicalBackend = $backendValues[0]
    $caseDirectory = Join-Path $repositoryRoot "test_all\operators\cases\$module\$operator"
    $cmakePath = Join-Path $caseDirectory 'CMakeLists.txt'
    if (-not (Test-Path -LiteralPath $cmakePath -PathType Leaf)) {
        throw "missing case CMake for $module/$operator"
    }
    $target = Parse-CaseTarget $cmakePath
    $shellRunner = Join-Path $caseDirectory 'run_operator.sh'
    $pythonRunner = Join-Path $caseDirectory 'run_operator.py'
    if (Test-Path -LiteralPath $shellRunner -PathType Leaf) {
        $runnerKind = 'shell_wrapper'
        $runnerPath = Relative-RepositoryPath $shellRunner
    } elseif (Test-Path -LiteralPath $pythonRunner -PathType Leaf) {
        $runnerKind = 'python_wrapper'
        $runnerPath = Relative-RepositoryPath $pythonRunner
    } elseif ($module -eq 'bsplines' -and $operator -in @('gauss_spline','quadratic')) {
        $runnerKind = 'python_wrapper'
        $runnerPath = 'test_all/operators/shared/bsplines/run_operator.py'
    } elseif ($module -eq 'windows') {
        $runnerKind = 'windows_binary'
        $runnerPath = $null
    } else {
        throw "no supported single-case runner for $module/$operator"
    }
    if (-not $configIndex.ContainsKey($operator)) {
        throw "missing config mapping for $operator"
    }
    $tree = if ($logicalBackend -eq $sharedBackend) {$sharedBackend} else {$logicalBackend}
    $selectableBackends = @($supportedBackends)
    $binaryTrees = [ordered]@{}
    foreach ($backend in $supportedBackends) {
        $binaryTrees[$backend] = if ($logicalBackend -eq $sharedBackend) {$sharedBackend} else {$backend}
    }
    $stableName = "operator_${module}_${operator}"
    $operatorCommands.Add([ordered]@{
        module = $module
        operator = $operator
        profile_backend = $logicalBackend
        selectable_backends = $selectableBackends
        dtypes = @('FP32','FP16','INT32','INT16','INT8')
        runner_kind = $runnerKind
        runner_path = $runnerPath
        runner_backend_argument = Test-RunnerBackendArgument $runnerKind $runnerPath
        config = $configIndex[$operator].config
        thresholds = $configIndex[$operator].thresholds
        binary_tree = $tree
        binary_tree_by_backend = $binaryTrees
        binary = "bin/$stableName"
        run_type = 'smoke'
        effective_warmup_runs = 1
        effective_measured_runs = 1
    })
    $buildBackends = if ($logicalBackend -eq $sharedBackend) {@($sharedBackend)} else {@($supportedBackends)}
    foreach ($buildBackend in $buildBackends) {
        $configuredBackend = if ($buildBackend -eq $sharedBackend) {$defaultInputProfileBackend} else {$buildBackend}
        $operatorBuildGroups.Add([ordered]@{
            build_group_id = "operator_${module}_${operator}"
            kind = 'operator'
            logical_backend = $buildBackend
            configured_backend = $configuredBackend
            source = Relative-RepositoryPath $caseDirectory
            work = "$buildBackend/work/operators/$module/$operator"
            cmake_options = Backend-Options $configuredBackend
            targets = @([ordered]@{
                cmake_target = $target
                internal_binary = "bin/$target"
                published_binary = "$buildBackend/bin/$stableName"
            })
        })
    }
}

$taskGroups = [Collections.Generic.List[object]]::new()
foreach ($backend in $supportedBackends) {
    $taskGroups.Add([ordered]@{
        build_group_id = 'task_pipelines'
        kind = 'task'
        logical_backend = $backend
        configured_backend = $backend
        source = '.'
        work = "$backend/work/root"
        cmake_options = Backend-Options $backend
        targets = @(
            [ordered]@{cmake_target='task1_pipeline_benchmark';internal_binary='bin/task1_pipeline_benchmark';published_binary="$backend/bin/task1_pipeline_benchmark"},
            [ordered]@{cmake_target='task2_pipeline_benchmark';internal_binary='bin/task2_pipeline_benchmark';published_binary="$backend/bin/task2_pipeline_benchmark"}
        )
    })
}

$abnormalSources = [ordered]@{
    operator = [ordered]@{source='test_all/operators/abnormal';target='abnormal_input_operator_abnormal';published='operator_abnormal_input'}
    task1 = [ordered]@{source='test_all/tasks/abnormal/task1';target='task1_task_abnormal';published='task1_abnormal_input'}
    task2 = [ordered]@{source='test_all/tasks/abnormal/task2';target='task2_task_abnormal';published='task2_abnormal_input'}
}
$abnormalGroups = [Collections.Generic.List[object]]::new()
foreach ($backend in $supportedBackends) {
    foreach ($name in $abnormalSources.Keys) {
        $item = $abnormalSources[$name]
        $abnormalGroups.Add([ordered]@{
            build_group_id = "${name}_abnormal_input"
            kind = 'abnormal'
            logical_backend = $backend
            configured_backend = $backend
            source = $item.source
            work = "$backend/work/abnormal/$name"
            cmake_options = Backend-Options $backend
            targets = @([ordered]@{
                cmake_target = $item.target
                internal_binary = "bin/$($item.target)"
                published_binary = "$backend/bin/$($item.published)"
            })
        })
    }
}

$buildPlan = [ordered]@{
    schema_version = 2
    build_root = 'build/demo'
    supported_backends = @($supportedBackends)
    shared_backend = $sharedBackend
    public_directories = @($supportedBackends | ForEach-Object {"$_/bin"; "$_/lib"}) + @("$sharedBackend/bin","$sharedBackend/lib")
    forbidden_public_name_patterns = @('task_f','cusignal_task_f','stage[0-9]+','[0-9a-f]{7,40}')
    groups = @($taskGroups) + @($operatorBuildGroups) + @($abnormalGroups)
}
Write-Json (Join-Path $configBuildRoot 'demo_build_plan.json') $buildPlan 10

$commands = [ordered]@{
    schema_version = 2
    operator_count = 53
    dtype_count = 5
    entries = @($operatorCommands)
}
Write-Json (Join-Path $configRuntimeRoot 'operator_demo_commands.json') $commands 8

# Freeze Task formal case_id -> scale_id mappings so --case-id does not depend on Windows at runtime.
$taskCases = [Collections.Generic.List[object]]::new()
foreach ($task in @('Task1','Task2')) {
    $taskRoot = Join-Path $ArchiveRoot "03_任务结果\$task\formal\pipeline\$defaultInputProfileBackend"
    $files = @(Get-ChildItem -LiteralPath $taskRoot -Recurse -File -Filter "$($task.ToLower())_benchmark_${defaultInputProfileBackend}_formal.csv")
    $byScale = @{}
    foreach ($file in $files) {
        $rows = @(Import-Csv -LiteralPath $file.FullName -Encoding UTF8)
        $cpu = @($rows | Where-Object device -eq 'cpu')
        if ($cpu.Count -ne 1) { continue }
        $row = $cpu[0]
        if ($row.status -ne 'PASS' -or $row.accuracy_status -ne 'PASS') { continue }
        if ($byScale.ContainsKey($row.scale_id)) {
            throw "duplicate admitted Task case for $task/$($row.scale_id)"
        }
        $byScale[$row.scale_id] = [ordered]@{
            target = $task.ToLower()
            backend = $defaultInputProfileBackend
            dtype = $row.dtype
            case_id = $row.case_id
            scale_id = $row.scale_id
            source_csv_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        }
    }
    if ($byScale.Count -ne 110) {
        throw "Task case mapping must contain 110 rows for $task, got $($byScale.Count)"
    }
    foreach ($key in ($byScale.Keys | Sort-Object)) { $taskCases.Add($byScale[$key]) }
}
Write-Json (Join-Path $configRuntimeRoot 'task_demo_cases.json') ([ordered]@{
    schema_version=1; entries=@($taskCases)
}) 6

# Record every CMake file, including historical files, while marking only active files build-eligible.
$cmakeRows = [Collections.Generic.List[object]]::new()
$cmakeFiles = Get-ChildItem -LiteralPath $repositoryRoot -Recurse -File -Force | Where-Object {
    $_.Name -eq 'CMakeLists.txt' -or $_.Extension -eq '.cmake'
} | Sort-Object FullName
foreach ($file in $cmakeFiles) {
    $relative = Relative-RepositoryPath $file.FullName
    $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $file.FullName
    $projects = @([regex]::Matches($text,'(?m)^\s*project\s*\(\s*([^\s\)]+)') | ForEach-Object {$_.Groups[1].Value})
    $executables = @([regex]::Matches($text,'(?m)^\s*add_executable\s*\(\s*([^\s\)]+)') | ForEach-Object {$_.Groups[1].Value})
    $custom = @([regex]::Matches($text,'(?m)^\s*add_custom_target\s*\(\s*([^\s\)]+)') | ForEach-Object {$_.Groups[1].Value})
    $includes = @([regex]::Matches($text,'(?m)^\s*include\s*\(([^\r\n\)]+)') | ForEach-Object {$_.Groups[1].Value.Trim(' ','"')})
    $subdirectories = @([regex]::Matches($text,'(?m)^\s*add_subdirectory\s*\(([^\r\n\)]+)') | ForEach-Object {$_.Groups[1].Value.Trim()})
    $cmakeRows.Add([ordered]@{
        path = $relative
        status = if ($relative.StartsWith('Archiving/')) {'historical_excluded'} else {'active'}
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        projects = $projects
        executable_targets = $executables
        custom_targets = $custom
        includes = $includes
        subdirectories = $subdirectories
    })
}
if ($cmakeRows.Count -ne 77 -or @($cmakeRows | Where-Object status -eq 'active').Count -ne 77) {
    throw "CMake inventory changed: total=$($cmakeRows.Count) active=$(@($cmakeRows | Where-Object status -eq 'active').Count)"
}
Write-Json (Join-Path $configBuildRoot 'cmake_dependency_audit.json') ([ordered]@{
    schema_version=1
    total_cmake_files=$cmakeRows.Count
    active_cmake_files=@($cmakeRows | Where-Object status -eq 'active').Count
    historical_cmake_files=@($cmakeRows | Where-Object status -eq 'historical_excluded').Count
    operator_case_cmake_files=53
    files=@($cmakeRows)
}) 10

Write-Output (
    '[DEMO][METADATA] cmake=77 active=77 operators=53 operator_profiles=265 task_cases=220 build_groups={0} status=PASS' -f
    $buildPlan.groups.Count
)
