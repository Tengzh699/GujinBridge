[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$ModelPath = "outputs-gujinbridge-v5-merged",
    [string]$TokenizerPath = "",
    [string]$TrainDir = "data/gujinbridge/dpo_v1_final/train",
    [string]$ValidationDir = "data/gujinbridge/dpo_v1_final/validation",
    [string]$OutputDir = "outputs-gujinbridge-dpo-v1-qwen3-1.7b",
    [int]$MaxSteps = 100,
    [string]$CudaDevices = "0",
    [switch]$SmokeTest,
    [switch]$Resume,
    [string]$ResumeFrom = "",
    [switch]$AllowDownload,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

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

$resolvedModelPath = Resolve-ProjectPath $ModelPath
if (-not $TokenizerPath) {
    $TokenizerPath = $ModelPath
}
$resolvedTokenizerPath = Resolve-ProjectPath $TokenizerPath
$resolvedTrainDir = Resolve-ProjectPath $TrainDir
$resolvedValidationDir = Resolve-ProjectPath $ValidationDir
$resolvedOutputDir = Resolve-ProjectPath $OutputDir

if ($SmokeTest) {
    $resolvedOutputDir = "$resolvedOutputDir-smoke"
    $MaxSteps = 20
}

foreach ($path in @($resolvedModelPath, $resolvedTokenizerPath, $resolvedTrainDir, $resolvedValidationDir)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Required directory does not exist: $path"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $resolvedModelPath "config.json") -PathType Leaf)) {
    throw "Model config is missing: $resolvedModelPath\config.json"
}
if (-not ((Test-Path -LiteralPath (Join-Path $resolvedModelPath "model.safetensors") -PathType Leaf) -or
          (Test-Path -LiteralPath (Join-Path $resolvedModelPath "model.safetensors.index.json") -PathType Leaf) -or
          (Test-Path -LiteralPath (Join-Path $resolvedModelPath "pytorch_model.bin") -PathType Leaf))) {
    throw "Full model weights are missing under: $resolvedModelPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedTokenizerPath "tokenizer_config.json") -PathType Leaf)) {
    throw "Tokenizer config is missing under: $resolvedTokenizerPath"
}

$trainFiles = @(Get-ChildItem -LiteralPath $resolvedTrainDir -Recurse -File -Filter "*.jsonl")
$validationFiles = @(Get-ChildItem -LiteralPath $resolvedValidationDir -Recurse -File -Filter "*.jsonl")
if ($trainFiles.Count -eq 0) {
    throw "No JSONL training data found under: $resolvedTrainDir"
}
if ($validationFiles.Count -eq 0) {
    throw "No JSONL validation data found under: $resolvedValidationDir"
}

$resolvedResumeCheckpoint = $null
if ($ResumeFrom) {
    $resolvedResumeCheckpoint = Resolve-ProjectPath $ResumeFrom
}
elseif ($Resume) {
    if (-not (Test-Path -LiteralPath $resolvedOutputDir -PathType Container)) {
        throw "Cannot resume because the output directory does not exist: $resolvedOutputDir"
    }
    $latestCheckpoint = Get-ChildItem -LiteralPath $resolvedOutputDir -Directory -Filter "checkpoint-*" |
        Where-Object {
            $_.Name -match '^checkpoint-(\d+)$' -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "trainer_state.json")) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "adapter_model.safetensors")) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "optimizer.pt")) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "scheduler.pt"))
        } |
        Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } -Descending |
        Select-Object -First 1
    if (-not $latestCheckpoint) {
        throw "No complete checkpoint was found under: $resolvedOutputDir"
    }
    $resolvedResumeCheckpoint = $latestCheckpoint.FullName
}

if ($resolvedResumeCheckpoint) {
    foreach ($requiredFile in @("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt")) {
        $requiredPath = Join-Path $resolvedResumeCheckpoint $requiredFile
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Resume checkpoint is incomplete; missing: $requiredPath"
        }
    }
}
elseif ((Test-Path -LiteralPath $resolvedOutputDir) -and
        (Get-ChildItem -LiteralPath $resolvedOutputDir -Force | Select-Object -First 1)) {
    throw "Output directory is not empty; refusing to overwrite: $resolvedOutputDir"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $AllowDownload) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}

$maxTrainSamples = if ($SmokeTest) { "128" } else { "-1" }
$maxEvalSamples = if ($SmokeTest) { "31" } else { "-1" }
$evalSteps = if ($SmokeTest) { "10" } else { "25" }
$saveSteps = if ($SmokeTest) { "20" } else { "50" }
$warmupSteps = if ($SmokeTest) { "2" } else { "10" }

$trainingArgs = @(
    "training/dpo_training.py",
    "--model_name_or_path", $resolvedModelPath,
    "--tokenizer_name_or_path", $resolvedTokenizerPath,
    "--train_file_dir", $resolvedTrainDir,
    "--validation_file_dir", $resolvedValidationDir,
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", "8",
    "--do_train",
    "--do_eval",
    "--use_peft", "True",
    "--max_train_samples", $maxTrainSamples,
    "--max_eval_samples", $maxEvalSamples,
    "--max_steps", $MaxSteps.ToString(),
    "--learning_rate", "5e-6",
    "--warmup_steps", $warmupSteps,
    "--weight_decay", "0.05",
    "--logging_steps", "1",
    "--eval_strategy", "steps",
    "--eval_steps", $evalSteps,
    "--save_steps", $saveSteps,
    "--max_source_length", "640",
    "--max_target_length", "384",
    "--output_dir", $resolvedOutputDir,
    "--target_modules", "all",
    "--lora_rank", "16",
    "--lora_alpha", "32",
    "--lora_dropout", "0.05",
    "--torch_dtype", "bfloat16",
    "--bf16", "True",
    "--fp16", "False",
    "--report_to", "tensorboard",
    "--remove_unused_columns", "False",
    "--gradient_checkpointing", "True",
    # On Windows, datasets multiprocessing may fail to create worker pipes.
    "--preprocessing_num_workers", "1",
    "--cache_dir", (Join-Path $repoRoot "cache")
)
if ($resolvedResumeCheckpoint) {
    $trainingArgs += @("--resume_from_checkpoint", $resolvedResumeCheckpoint)
}

Write-Host "GujinBridge DPO V1 configuration:"
Write-Host "  Model: $resolvedModelPath"
Write-Host "  Tokenizer: $resolvedTokenizerPath"
Write-Host "  Train: $resolvedTrainDir"
Write-Host "  Validation: $resolvedValidationDir"
Write-Host "  Output: $resolvedOutputDir"
Write-Host "  Max steps: $MaxSteps"
Write-Host "  Smoke test: $SmokeTest"
Write-Host "  Resume checkpoint: $(if ($resolvedResumeCheckpoint) { $resolvedResumeCheckpoint } else { 'none' })"

if ($DryRun) {
    Write-Host "Dry run passed. Training was not started."
    Write-Host "$PythonExe $($trainingArgs -join ' ')"
    return
}

Push-Location $repoRoot
try {
    & $PythonExe @trainingArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DPO training process exited with code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
