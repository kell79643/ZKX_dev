param([switch]$NoOpen)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$entry = Join-Path $repositoryRoot "Task1\visualization\python\main.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Local visualization environment is missing. From the repository root run: python -m venv .venv"
}

Push-Location $repositoryRoot
try {
    & $python $entry --plot-only
    if ($LASTEXITCODE -ne 0) {
        throw "Task1 local plotting failed with exit code $LASTEXITCODE"
    }
    if (-not $NoOpen) {
        Invoke-Item (Join-Path $repositoryRoot "Task1\evidence\visualization\figures")
    }
}
finally {
    Pop-Location
}
