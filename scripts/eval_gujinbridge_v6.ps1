[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$BaseModel = "outputs-gujinbridge-v5-merged",
    [string]$V6Adapter = "outputs-gujinbridge-sft-v6-targeted-qwen3-1.7b",
    [string]$TokenizerPath = "outputs-gujinbridge-v5-merged",
    [string]$PromptsFile = "data/gujinbridge/eval_v5/prompts.txt",
    [string]$ReferencesFile = "data/gujinbridge/eval_v5/references.jsonl",
    [string]$V5Predictions = "data/gujinbridge/eval_v5/predictions_v5.jsonl",
    [string]$DpoPredictions = "data/gujinbridge/eval_dpo_v1/predictions_dpo.jsonl",
    [string]$OutputDir = "data/gujinbridge/eval_v6",
    [int]$BatchSize = 32,
    [int]$MaxNewTokens = 512,
    [string]$CudaDevices = "0",
    [switch]$SmokeTest,
    [switch]$Force,
    [switch]$DryRun
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

$resolvedBase = Resolve-ProjectPath $BaseModel
$resolvedAdapter = Resolve-ProjectPath $V6Adapter
$resolvedTokenizer = Resolve-ProjectPath $TokenizerPath
$resolvedPrompts = Resolve-ProjectPath $PromptsFile
$resolvedReferences = Resolve-ProjectPath $ReferencesFile
$resolvedV5 = Resolve-ProjectPath $V5Predictions
$resolvedDpo = Resolve-ProjectPath $DpoPredictions
$resolvedOutput = Resolve-ProjectPath $OutputDir
if ($SmokeTest) { $resolvedOutput = "$resolvedOutput-smoke" }

foreach ($path in @($resolvedBase, $resolvedAdapter, $resolvedTokenizer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Required model directory does not exist: $path"
    }
}
foreach ($path in @($resolvedPrompts, $resolvedReferences, $resolvedV5, $resolvedDpo)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required evaluation file does not exist: $path"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedAdapter "adapter_model.safetensors") -PathType Leaf)) {
    throw "V6 adapter weights are missing: $resolvedAdapter"
}

$predictions = Join-Path $resolvedOutput "predictions_v6.jsonl"
$metrics = Join-Path $resolvedOutput "metrics_v6.json"
$v5Comparison = Join-Path $resolvedOutput "comparison_v5_vs_v6.json"
$v5Report = Join-Path $resolvedOutput "REPORT_V5_VS_V6.md"
$dpoComparison = Join-Path $resolvedOutput "comparison_dpo_vs_v6.json"
$dpoReport = Join-Path $resolvedOutput "REPORT_DPO_VS_V6.md"
$outputs = @($predictions, $metrics, $v5Comparison, $v5Report, $dpoComparison, $dpoReport)
$existing = @($outputs | Where-Object { Test-Path -LiteralPath $_ })
if ($existing.Count -gt 0 -and -not $Force) {
    throw "Evaluation outputs already exist. Use -Force to overwrite: $($existing -join ', ')"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$maxSamples = if ($SmokeTest) { "64" } else { "-1" }

$inferenceArgs = @(
    "demo/inference.py",
    "--base_model", $resolvedBase,
    "--lora_model", $resolvedAdapter,
    "--tokenizer_path", $resolvedTokenizer,
    "--system_prompt_file", (Join-Path $repoRoot "configs\gujinbridge_system_prompt.txt"),
    "--data_file", $resolvedPrompts,
    "--output_file", $predictions,
    "--temperature", "0",
    "--repetition_penalty", "1.0",
    "--max_new_tokens", $MaxNewTokens.ToString(),
    "--eval_batch_size", $BatchSize.ToString(),
    "--max_samples", $maxSamples,
    "--disable_thinking",
    "--quiet",
    "--local_files_only",
    "--cache_dir", (Join-Path $repoRoot "cache")
)
$evaluateArgs = @(
    "tools/evaluate_gujinbridge_predictions.py",
    "--references", $resolvedReferences,
    "--predictions", $predictions,
    "--output", $metrics
)
$v5CompareArgs = @(
    "tools/compare_gujinbridge_evals.py",
    "--references", $resolvedReferences,
    "--baseline-predictions", $resolvedV5,
    "--candidate-predictions", $predictions,
    "--output-json", $v5Comparison,
    "--output-markdown", $v5Report,
    "--baseline-name", "V5 SFT",
    "--candidate-name", "V6 targeted SFT"
)
$dpoCompareArgs = @(
    "tools/compare_gujinbridge_evals.py",
    "--references", $resolvedReferences,
    "--baseline-predictions", $resolvedDpo,
    "--candidate-predictions", $predictions,
    "--output-json", $dpoComparison,
    "--output-markdown", $dpoReport,
    "--baseline-name", "DPO V1",
    "--candidate-name", "V6 targeted SFT"
)
if ($SmokeTest) {
    $v5CompareArgs += "--allow-partial"
    $dpoCompareArgs += "--allow-partial"
}

Write-Host "GujinBridge V6 frozen-test evaluation:"
Write-Host "  Base: $resolvedBase"
Write-Host "  V6 adapter: $resolvedAdapter"
Write-Host "  Output: $resolvedOutput"
Write-Host "  Samples: $(if ($SmokeTest) { 64 } else { 'all 1440' })"

if ($DryRun) {
    Write-Host "Dry run passed. Evaluation was not started."
    Write-Host "$PythonExe $($inferenceArgs -join ' ')"
    return
}

New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
Push-Location $repoRoot
try {
    & $PythonExe @inferenceArgs
    if ($LASTEXITCODE -ne 0) { throw "V6 inference exited with code: $LASTEXITCODE" }
    & $PythonExe @evaluateArgs
    if ($LASTEXITCODE -ne 0) { throw "V6 metric evaluation exited with code: $LASTEXITCODE" }
    & $PythonExe @v5CompareArgs
    if ($LASTEXITCODE -ne 0) { throw "V5/V6 comparison exited with code: $LASTEXITCODE" }
    & $PythonExe @dpoCompareArgs
    if ($LASTEXITCODE -ne 0) { throw "DPO/V6 comparison exited with code: $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "V6 evaluation complete. Open: $v5Report"
