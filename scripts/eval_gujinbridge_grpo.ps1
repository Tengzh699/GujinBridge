[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$BaseModel = "outputs-gujinbridge-v5-merged",
    [string]$DpoAdapter = "outputs-gujinbridge-dpo-v1-qwen3-1.7b",
    [string]$GrpoOutput = "outputs-gujinbridge-grpo-v1-qwen3-1.7b",
    [string]$TokenizerPath = "outputs-gujinbridge-v5-merged",
    [string]$AllReferences = "data/gujinbridge/eval_v5/references.jsonl",
    [string]$OutputDir = "data/gujinbridge/eval_grpo_v1",
    [int[]]$Checkpoints = @(50, 75, 100),
    [int]$BatchSize = 32,
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
    $preferred = "E:\Anaconda\envs\GujinBridge\python.exe"
    $PythonExe = if (Test-Path -LiteralPath $preferred -PathType Leaf) { $preferred } else { (Get-Command python -ErrorAction Stop).Source }
}

$base = Resolve-ProjectPath $BaseModel
$dpo = Resolve-ProjectPath $DpoAdapter
$grpo = Resolve-ProjectPath $GrpoOutput
$tokenizer = Resolve-ProjectPath $TokenizerPath
$allRefs = Resolve-ProjectPath $AllReferences
$output = Resolve-ProjectPath $OutputDir
if ($SmokeTest) { $output = "$output-smoke"; $Checkpoints = @($Checkpoints[-1]) }

foreach ($path in @($base, $dpo, $grpo, $tokenizer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Required model directory does not exist: $path" }
}
foreach ($path in @($allRefs)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required evaluation file does not exist: $path" }
}

$prompts = Join-Path $output "prompts_punctuate.txt"
$references = Join-Path $output "references_punctuate.jsonl"
$systemPrompt = Join-Path $repoRoot "configs\gujinbridge_punctuation_grpo_system_prompt.txt"
$maxSamples = if ($SmokeTest) { "32" } else { "-1" }
$dpoControlDir = Join-Path $output "dpo_control"
$dpoControlPredictions = Join-Path $dpoControlDir "predictions_dpo_control.jsonl"
$dpoControlMetrics = Join-Path $dpoControlDir "metrics_dpo_control.json"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "GujinBridge GRPO checkpoint evaluation:"
Write-Host "  Reconstruction: V5 merged -> merge DPO -> GRPO adapter"
Write-Host "  Checkpoints: $($Checkpoints -join ', ')"
Write-Host "  Samples: $(if ($SmokeTest) { 32 } else { 361 })"
if ($DryRun) {
    foreach ($step in $Checkpoints) {
        $adapter = Join-Path $grpo "checkpoint-$step"
        if (-not (Test-Path -LiteralPath $adapter -PathType Container)) { throw "Checkpoint does not exist: $adapter" }
        Write-Host "Would evaluate: $adapter"
    }
    Write-Host "Dry run passed. Evaluation was not started."
    return
}

New-Item -ItemType Directory -Path $output -Force | Out-Null
Push-Location $repoRoot
try {
    & $PythonExe "tools/export_gujinbridge_task_eval.py" `
        --references $allRefs --task punctuate `
        --output-references $references --output-prompts $prompts
    if ($LASTEXITCODE -ne 0) { throw "Punctuation export exited with code: $LASTEXITCODE" }

    $dpoExisting = @(@($dpoControlPredictions, $dpoControlMetrics) | Where-Object { Test-Path -LiteralPath $_ })
    if ($dpoExisting.Count -gt 0 -and -not $Force) {
        throw "DPO control outputs already exist; use -Force to overwrite."
    }
    New-Item -ItemType Directory -Path $dpoControlDir -Force | Out-Null
    $dpoInferenceArgs = @(
        "demo/inference.py",
        "--base_model", $base,
        "--lora_model", $dpo,
        "--tokenizer_path", $tokenizer,
        "--system_prompt_file", $systemPrompt,
        "--data_file", $prompts,
        "--output_file", $dpoControlPredictions,
        "--temperature", "0",
        "--repetition_penalty", "1.0",
        "--max_new_tokens", "384",
        "--eval_batch_size", $BatchSize.ToString(),
        "--max_samples", $maxSamples,
        "--disable_thinking", "--quiet", "--local_files_only",
        "--cache_dir", (Join-Path $repoRoot "cache")
    )
    & $PythonExe @dpoInferenceArgs
    if ($LASTEXITCODE -ne 0) { throw "DPO control inference failed: $LASTEXITCODE" }
    & $PythonExe "tools/evaluate_gujinbridge_predictions.py" `
        --references $references --predictions $dpoControlPredictions --output $dpoControlMetrics
    if ($LASTEXITCODE -ne 0) { throw "DPO control metrics failed: $LASTEXITCODE" }

    foreach ($step in $Checkpoints) {
        $adapter = Join-Path $grpo "checkpoint-$step"
        if (-not (Test-Path -LiteralPath (Join-Path $adapter "adapter_model.safetensors") -PathType Leaf)) {
            throw "GRPO checkpoint is incomplete: $adapter"
        }
        $stepDir = Join-Path $output "checkpoint-$step"
        $predictions = Join-Path $stepDir "predictions_grpo.jsonl"
        $metrics = Join-Path $stepDir "metrics_grpo.json"
        $comparison = Join-Path $stepDir "comparison_dpo_vs_grpo.json"
        $report = Join-Path $stepDir "REPORT_DPO_VS_GRPO.md"
        $existing = @(@($predictions, $metrics, $comparison, $report) | Where-Object { Test-Path -LiteralPath $_ })
        if ($existing.Count -gt 0 -and -not $Force) {
            throw "Outputs already exist for checkpoint-$step; use -Force to overwrite."
        }
        New-Item -ItemType Directory -Path $stepDir -Force | Out-Null

        $inferenceArgs = @(
            "demo/inference.py",
            "--base_model", $base,
            "--merge_lora_model", $dpo,
            "--lora_model", $adapter,
            "--tokenizer_path", $tokenizer,
            "--system_prompt_file", $systemPrompt,
            "--data_file", $prompts,
            "--output_file", $predictions,
            "--temperature", "0",
            "--repetition_penalty", "1.0",
            "--max_new_tokens", "384",
            "--eval_batch_size", $BatchSize.ToString(),
            "--max_samples", $maxSamples,
            "--disable_thinking", "--quiet", "--local_files_only",
            "--cache_dir", (Join-Path $repoRoot "cache")
        )
        & $PythonExe @inferenceArgs
        if ($LASTEXITCODE -ne 0) { throw "Inference failed for checkpoint-${step}: $LASTEXITCODE" }
        & $PythonExe "tools/evaluate_gujinbridge_predictions.py" `
            --references $references --predictions $predictions --output $metrics
        if ($LASTEXITCODE -ne 0) { throw "Metrics failed for checkpoint-${step}: $LASTEXITCODE" }
        $compareArgs = @(
            "tools/compare_gujinbridge_evals.py",
            "--references", $references,
            "--baseline-predictions", $dpoControlPredictions,
            "--candidate-predictions", $predictions,
            "--output-json", $comparison,
            "--output-markdown", $report,
            "--baseline-name", "DPO V1 controlled",
            "--candidate-name", "GRPO checkpoint-$step"
        )
        if ($SmokeTest) { $compareArgs += "--allow-partial" }
        & $PythonExe @compareArgs
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed for checkpoint-${step}: $LASTEXITCODE" }
    }
}
finally { Pop-Location }

Write-Host "GRPO evaluation complete: $output"
