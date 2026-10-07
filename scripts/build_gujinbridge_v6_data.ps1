[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$InputDir = "data/gujinbridge/processed_v5_final",
    [string]$OutputDir = "data/gujinbridge/processed_v6_targeted"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) { return $PathValue }
    return Join-Path $repoRoot $PathValue
}

if (-not $PythonExe) {
    $gujinBridgePython = "E:\Anaconda\envs\GujinBridge\python.exe"
    if (Test-Path -LiteralPath $gujinBridgePython -PathType Leaf) {
        $PythonExe = $gujinBridgePython
    }
    else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}

$resolvedInput = Resolve-ProjectPath $InputDir
$resolvedOutput = Resolve-ProjectPath $OutputDir
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repoRoot
try {
    & $PythonExe "tools/build_gujinbridge_v6_curriculum.py" `
        --input-dir $resolvedInput `
        --output-dir $resolvedOutput
    if ($LASTEXITCODE -ne 0) {
        throw "V6 data construction exited with code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "V6 targeted data is ready: $(Join-Path $resolvedOutput 'V6_DATA_REPORT.md')"
