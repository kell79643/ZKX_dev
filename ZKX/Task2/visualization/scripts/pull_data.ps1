param(
    [Parameter(Mandatory = $true)][string]$RemoteRun
)

$ErrorActionPreference = "Stop"

$scriptDirectory = $PSScriptRoot
$repositoryRoot = (Resolve-Path (Join-Path $scriptDirectory "..\..\..")).Path
$remoteRun = (Resolve-Path -LiteralPath $RemoteRun).Path
$sshKey = Join-Path $env:USERPROFILE ".ssh\id_ed25519_zkx"
$remoteHost = "usr_02@10.110.12.10"
$transferId = "task2-visualization-data-$PID"
$containerArchive = "/tmp/$transferId.tar.gz"
$localArchive = Join-Path $env:TEMP "$transferId.tar.gz"
$stagingRoot = Join-Path $env:TEMP "$transferId-extracted"
$helper = "env ZKX_GPU02_INTERACTIVE_SOCKET=/tmp/zkx-gpu02-interactive-$transferId.sock python3 remote_scripts/gpu02_interactive.py"

if (-not (Test-Path -LiteralPath $remoteRun -PathType Leaf)) {
    throw "remote runner not found: $remoteRun"
}
if (-not (Test-Path -LiteralPath $sshKey -PathType Leaf)) {
    throw "SSH key not found: $sshKey"
}

function Invoke-RemoteStep {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$SuccessMarker
    )
    $output = @(& $remoteRun --no-sync $Command 2>&1)
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        throw "remote step failed with exit code $exitCode"
    }
    if (($output -join "`n") -notmatch [regex]::Escape($SuccessMarker)) {
        throw "remote step did not report success marker: $SuccessMarker"
    }
}

Push-Location $repositoryRoot
try {
    Invoke-RemoteStep "$helper start --timeout 30" "status=prompt"
    Invoke-RemoteStep "$helper send 'source /zq500/sdk/env.sh && echo TASK2_SDK_READY' --timeout 30" "TASK2_SDK_READY"
    Invoke-RemoteStep "$helper send 'cd /tmp/ZKX_dev && test -f Task2/evidence/visualization/data/task2_manifest.csv && tar -czf $containerArchive Task2/evidence/visualization/data Task2/evidence/visualization/task2_selected.conf Task2/evidence/visualization/pipeline.log Task2/evidence/visualization/run_summary.json && echo TASK2_DATA_PACKAGE_READY' --timeout 180" "TASK2_DATA_PACKAGE_READY"
    Invoke-RemoteStep "$helper fetch $containerArchive $containerArchive --timeout 600" "status=ok"

    & scp -i $sshKey -o IdentitiesOnly=yes -o BatchMode=yes "${remoteHost}:$containerArchive" $localArchive
    if ($LASTEXITCODE -ne 0) {
        throw "SCP failed with exit code $LASTEXITCODE"
    }

    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    & tar -xzf $localArchive -C $stagingRoot
    if ($LASTEXITCODE -ne 0) {
        throw "local staging extraction failed with exit code $LASTEXITCODE"
    }
    $stagedResults = Join-Path $stagingRoot "Task2\evidence\visualization"
    $stagedManifest = Join-Path $stagedResults "data\task2_manifest.csv"
    if (-not (Test-Path -LiteralPath $stagedManifest -PathType Leaf)) {
        throw "Task2 manifest is missing from the downloaded package"
    }

    $resultsRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot "Task2\evidence\visualization"))
    $dataRoot = [IO.Path]::GetFullPath((Join-Path $resultsRoot "data"))
    if ([IO.Path]::GetDirectoryName($dataRoot) -ne $resultsRoot) {
        throw "unsafe local data directory: $dataRoot"
    }
    New-Item -ItemType Directory -Path $resultsRoot -Force | Out-Null
    if (Test-Path -LiteralPath $dataRoot) {
        Remove-Item -LiteralPath $dataRoot -Recurse -Force
    }
    Move-Item -LiteralPath (Join-Path $stagedResults "data") -Destination $dataRoot
    foreach ($name in @("task2_selected.conf", "pipeline.log", "run_summary.json")) {
        Copy-Item -LiteralPath (Join-Path $stagedResults $name) -Destination (Join-Path $resultsRoot $name) -Force
    }
    if (-not (Test-Path -LiteralPath (
            Join-Path $dataRoot "task2_manifest.csv") -PathType Leaf)) {
        throw "Task2 manifest is missing after extraction"
    }

    Write-Host "[TASK2][VISUALIZATION][PULL] data=$dataRoot status=PASS"
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $localArchive) {
        Remove-Item -LiteralPath $localArchive -Force
    }
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
