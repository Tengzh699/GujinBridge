[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$BaseModel = "outputs-gujinbridge-v5-merged",
    [string]$DpoAdapter = "outputs-gujinbridge-dpo-v1-qwen3-1.7b",
    [string]$TokenizerPath = "outputs-gujinbridge-v5-merged",
    [string]$PromptsFile = "data/gujinbridge/eval_v5/prompts.txt",
    [string]$ReferencesFile = "data/gujinbridge/eval_v5/references.jsonl",
    [string]$BaselinePredictions = "data/gujinbridge/eval_v5/predictions_v5.jsonl",
    [string]$OutputDir = "data/gujinbridge/eval_dpo_v1",
    [int]$BatchSize = 32,
    [int]$MaxNewTokens = 512,
    [string]$CudaDevices = "0",
    [switch]$SmokeTest,
    [switch]$AllowDownload,
    [switch]$Force,
    [switch]$DryRun
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

$resolvedBaseModel = Resolve-ProjectPath $BaseModel
$resolvedDpoAdapter = Resolve-ProjectPath $DpoAdapter
$resolvedTokenizerPath = Resolve-ProjectPath $TokenizerPath
$resolvedPrompts = Resolve-ProjectPath $PromptsFile
$resolvedReferences = Resolve-ProjectPath $ReferencesFile
$resolvedBaseline = Resolve-ProjectPath $BaselinePredictions
$resolvedOutputDir = Resolve-ProjectPath $OutputDir
if ($SmokeTest) {
    $resolvedOutputDir = "$resolvedOutputDir-smoke"
}

foreach ($path in @($resolvedBaseModel, $resolvedDpoAdapter, $resolvedTokenizerPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Required model directory does not exist: $path"
    }
}
foreach ($path in @($resolvedPrompts, $resolvedReferences, $resolvedBaseline)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required evaluation file does not exist: $path"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedDpoAdapter "adapter_model.safetensors") -PathType Leaf)) {
    throw "DPO adapter weights are missing: $resolvedDpoAdapter"
}

$predictions = Join-Path $resolvedOutputDir "predictions_dpo.jsonl"
$metrics = Join-Path $resolvedOutputDir "metrics_dpo.json"
$comparison = Join-Path $resolvedOutputDir "comparison_v5_vs_dpo.json"
$report = Join-Path $resolvedOutputDir "REPORT.md"
$outputs = @($predictions, $metrics, $comparison, $report)
$existing = @($outputs | Where-Object { Test-Path -LiteralPath $_ })
if ($existing.Count -gt 0 -and -not $Force) {
    throw "Evaluation outputs already exist. Use -Force to overwrite: $($existing -join ', ')"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $AllowDownload) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}

$maxSamples = if ($SmokeTest) { "64" } else { "-1" }
$inferenceArgs = @(
    "demo/inference.py",
    "--base_model", $resolvedBaseModel,
    "--lora_model", $resolvedDpoAdapter,
    "--tokenizer_path", $resolvedTokenizerPath,
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
    "--cache_dir", (Join-Path $repoRoot "cache")
)
if (-not $AllowDownload) {
    $inferenceArgs += "--local_files_only"
}

$evaluateArgs = @(
    "tools/evaluate_gujinbridge_predictions.py",
    "--references", $resolvedReferences,
    "--predictions", $predictions,
    "--output", $metrics
)
$compareArgs = @(
    "tools/compare_gujinbridge_evals.py",
    "--references", $resolvedReferences,
    "--baseline-predictions", $resolvedBaseline,
    "--candidate-predictions", $predictions,
    "--output-json", $comparison,
    "--output-markdown", $report,
    "--baseline-name", "V5 SFT",
    "--candidate-name", "DPO V1"
)
if ($SmokeTest) {
    $compareArgs += "--allow-partial"
}

Write-Host "GujinBridge DPO A/B evaluation:"
Write-Host "  Base (merged V5): $resolvedBaseModel"
Write-Host "  DPO adapter: $resolvedDpoAdapter"
Write-Host "  Prompts: $resolvedPrompts"
Write-Host "  Baseline predictions: $resolvedBaseline"
Write-Host "  Output: $resolvedOutputDir"
Write-Host "  Samples: $(if ($SmokeTest) { 64 } else { 'all' })"
Write-Host "  Batch size: $BatchSize"

if ($DryRun) {
    Write-Host "Dry run passed. Evaluation was not started."
    Write-Host "$PythonExe $($inferenceArgs -join ' ')"
    Write-Host "$PythonExe $($evaluateArgs -join ' ')"
    Write-Host "$PythonExe $($compareArgs -join ' ')"
    return
}

New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
Push-Location $repoRoot
try {
    & $PythonExe @inferenceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DPO inference exited with code: $LASTEXITCODE"
    }
    & $PythonExe @evaluateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Metric evaluation exited with code: $LASTEXITCODE"
    }
    & $PythonExe @compareArgs
    if ($LASTEXITCODE -ne 0) {
        throw "A/B comparison exited with code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Evaluation complete. Open: $report"
