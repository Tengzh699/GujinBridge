[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$ReferencesFile = "data/gujinbridge/eval_v5/references.jsonl",
    [string]$V5Predictions = "data/gujinbridge/eval_v5/predictions_v5.jsonl",
    [string]$DpoPredictions = "data/gujinbridge/eval_dpo_v1/predictions_dpo.jsonl",
    [string]$OutputDir = "data/gujinbridge/eval_dpo_v1",
    [int]$ReviewLimit = 300
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
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

$resolvedReferences = Resolve-ProjectPath $ReferencesFile
$resolvedV5 = Resolve-ProjectPath $V5Predictions
$resolvedDpo = Resolve-ProjectPath $DpoPredictions
$resolvedOutput = Resolve-ProjectPath $OutputDir

foreach ($path in @($resolvedReferences, $resolvedV5, $resolvedDpo)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required evaluation file does not exist: $path"
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $repoRoot
try {
    & $PythonExe "tools/analyze_gujinbridge_errors.py" `
        --references $resolvedReferences `
        --v5-predictions $resolvedV5 `
        --dpo-predictions $resolvedDpo `
        --output-dir $resolvedOutput `
        --review-limit $ReviewLimit
    if ($LASTEXITCODE -ne 0) {
        throw "Error analysis exited with code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Error analysis complete. Open: $(Join-Path $resolvedOutput 'ERROR_ANALYSIS.md')"
