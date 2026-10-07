[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$BaseModel = "Qwen/Qwen3-1.7B",
    [string]$TokenizerPath = "outputs-gujinbridge-sft-qwen3-1.7b",
    [string]$TrainDir = "data/gujinbridge/processed_v5_final/train",
    [string]$ValidationDir = "data/gujinbridge/processed_v5_final/validation",
    [string]$OutputDir = "outputs-gujinbridge-sft-v5-qwen3-1.7b",
    [int]$MaxSteps = -1,
    [int]$NumEpochs = 2,
    [string]$CudaDevices = "0",
    [switch]$SmokeTest,
    [switch]$Resume,
    [string]$ResumeFrom = "",
    [switch]$AllowDownload,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $PythonExe) {
    $gujinBridgePython = "E:\Anaconda\envs\GujinBridge\python.exe"
    if (Test-Path -LiteralPath $gujinBridgePython) {
        $PythonExe = $gujinBridgePython
    }
    else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}

# In offline mode, prefer a complete model snapshot from the default HF cache.
# The project cache may contain only config/tokenizer files and no model weights.
if (-not $AllowDownload -and $BaseModel -eq "Qwen/Qwen3-1.7B") {
    $defaultSnapshotRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3-1.7B\snapshots"
    if (Test-Path -LiteralPath $defaultSnapshotRoot) {
        $completeSnapshot = Get-ChildItem -LiteralPath $defaultSnapshotRoot -Directory |
            Where-Object {
                (Test-Path -LiteralPath (Join-Path $_.FullName "model.safetensors.index.json")) -or
                (Test-Path -LiteralPath (Join-Path $_.FullName "model.safetensors")) -or
                (Test-Path -LiteralPath (Join-Path $_.FullName "pytorch_model.bin"))
            } |
            Select-Object -First 1
        if ($completeSnapshot) {
            $BaseModel = $completeSnapshot.FullName
        }
    }
}

$resolvedTrainDir = Join-Path $repoRoot $TrainDir
$resolvedValidationDir = Join-Path $repoRoot $ValidationDir
$resolvedTokenizerPath = Join-Path $repoRoot $TokenizerPath
$resolvedOutputDir = Join-Path $repoRoot $OutputDir
if ($SmokeTest) {
    $resolvedOutputDir = "$resolvedOutputDir-smoke"
    $MaxSteps = 20
}

foreach ($path in @($resolvedTrainDir, $resolvedValidationDir, $resolvedTokenizerPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

$resolvedResumeCheckpoint = $null
if ($ResumeFrom) {
    if ([System.IO.Path]::IsPathRooted($ResumeFrom)) {
        $resolvedResumeCheckpoint = $ResumeFrom
    }
    else {
        $resolvedResumeCheckpoint = Join-Path $repoRoot $ResumeFrom
    }
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
    if (-not (Test-Path -LiteralPath $resolvedResumeCheckpoint -PathType Container)) {
        throw "Resume checkpoint does not exist: $resolvedResumeCheckpoint"
    }
    foreach ($requiredFile in @("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt")) {
        $requiredPath = Join-Path $resolvedResumeCheckpoint $requiredFile
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Resume checkpoint is incomplete; missing: $requiredPath"
        }
    }
}
elseif ((Test-Path -LiteralPath $resolvedOutputDir) -and (Get-ChildItem -LiteralPath $resolvedOutputDir -Force | Select-Object -First 1)) {
    throw "Output directory is not empty; refusing to overwrite: $resolvedOutputDir"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
if (-not $AllowDownload) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}

$maxTrainSamples = if ($SmokeTest) { "500" } else { "-1" }
$maxEvalSamples = if ($SmokeTest) { "100" } else { "1000" }
$evalSteps = if ($SmokeTest) { "10" } else { "500" }
$saveSteps = if ($SmokeTest) { "20" } else { "500" }
$trainingArgs = @(
    "training/supervised_finetuning.py",
    "--model_name_or_path", $BaseModel,
    "--tokenizer_name_or_path", $resolvedTokenizerPath,
    "--train_file_dir", $resolvedTrainDir,
    "--validation_file_dir", $resolvedValidationDir,
    "--per_device_train_batch_size", "2",
    "--per_device_eval_batch_size", "1",
    "--do_train",
    "--do_eval",
    "--use_peft", "True",
    "--max_train_samples", $maxTrainSamples,
    "--max_eval_samples", $maxEvalSamples,
    "--model_max_length", "1024",
    "--num_train_epochs", $NumEpochs.ToString(),
    "--max_steps", $MaxSteps.ToString(),
    "--learning_rate", "2e-5",
    "--warmup_steps", "20",
    "--weight_decay", "0.05",
    "--logging_strategy", "steps",
    "--logging_steps", "10",
    "--eval_strategy", "steps",
    "--eval_steps", $evalSteps,
    "--save_strategy", "steps",
    "--save_steps", $saveSteps,
    "--save_total_limit", "3",
    "--gradient_accumulation_steps", "8",
    "--preprocessing_num_workers", "4",
    "--output_dir", $resolvedOutputDir,
    "--logging_first_step", "True",
    "--target_modules", "all",
    "--lora_rank", "16",
    "--lora_alpha", "32",
    "--lora_dropout", "0.05",
    "--torch_dtype", "bfloat16",
    "--bf16",
    "--report_to", "tensorboard",
    "--gradient_checkpointing", "True",
    "--cache_dir", (Join-Path $repoRoot "cache")
)
if ($resolvedResumeCheckpoint) {
    $trainingArgs += @("--resume_from_checkpoint", $resolvedResumeCheckpoint)
}

Write-Host "GujinBridge V5 SFT configuration:"
Write-Host "  Base: $BaseModel"
Write-Host "  Train: $resolvedTrainDir"
Write-Host "  Validation: $resolvedValidationDir"
Write-Host "  Output: $resolvedOutputDir"
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
        throw "Training process exited with code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
