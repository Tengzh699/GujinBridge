[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$BaseModel = "outputs-gujinbridge-v5-merged",
    [string]$TokenizerPath = "outputs-gujinbridge-v5-merged",
    [string]$TrainDir = "data/gujinbridge/processed_v6_targeted/train",
    [string]$ValidationDir = "data/gujinbridge/processed_v6_targeted/validation",
    [string]$OutputDir = "outputs-gujinbridge-sft-v6-targeted-qwen3-1.7b",
    [int]$MaxSteps = -1,
    [int]$NumEpochs = 1,
    [double]$LearningRate = 1e-5,
    [string]$CudaDevices = "0",
    [switch]$SmokeTest,
    [switch]$Resume,
    [string]$ResumeFrom = "",
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

$resolvedBaseModel = Resolve-ProjectPath $BaseModel
$resolvedTokenizer = Resolve-ProjectPath $TokenizerPath
$resolvedTrain = Resolve-ProjectPath $TrainDir
$resolvedValidation = Resolve-ProjectPath $ValidationDir
$resolvedOutput = Resolve-ProjectPath $OutputDir
if ($SmokeTest) {
    $resolvedOutput = "$resolvedOutput-smoke"
    $MaxSteps = 20
}

foreach ($path in @($resolvedBaseModel, $resolvedTokenizer, $resolvedTrain, $resolvedValidation)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Required directory does not exist: $path"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedBaseModel "config.json") -PathType Leaf)) {
    throw "Base model config is missing: $resolvedBaseModel"
}
if (-not (
    (Test-Path -LiteralPath (Join-Path $resolvedBaseModel "model.safetensors") -PathType Leaf) -or
    (Test-Path -LiteralPath (Join-Path $resolvedBaseModel "model.safetensors.index.json") -PathType Leaf)
)) {
    throw "Base model weights are missing: $resolvedBaseModel"
}

$resolvedResumeCheckpoint = $null
if ($ResumeFrom) {
    $resolvedResumeCheckpoint = Resolve-ProjectPath $ResumeFrom
}
elseif ($Resume) {
    if (-not (Test-Path -LiteralPath $resolvedOutput -PathType Container)) {
        throw "Cannot resume because the output directory does not exist: $resolvedOutput"
    }
    $latestCheckpoint = Get-ChildItem -LiteralPath $resolvedOutput -Directory -Filter "checkpoint-*" |
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
        throw "No complete checkpoint was found under: $resolvedOutput"
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
elseif ((Test-Path -LiteralPath $resolvedOutput) -and (Get-ChildItem -LiteralPath $resolvedOutput -Force | Select-Object -First 1)) {
    throw "Output directory is not empty; refusing to overwrite: $resolvedOutput"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$maxTrainSamples = if ($SmokeTest) { "500" } else { "-1" }
$maxEvalSamples = if ($SmokeTest) { "100" } else { "-1" }
$evalSteps = if ($SmokeTest) { "10" } else { "100" }
$saveSteps = if ($SmokeTest) { "20" } else { "100" }
$trainingArgs = @(
    "training/supervised_finetuning.py",
    "--model_name_or_path", $resolvedBaseModel,
    "--tokenizer_name_or_path", $resolvedTokenizer,
    "--train_file_dir", $resolvedTrain,
    "--validation_file_dir", $resolvedValidation,
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
    "--learning_rate", $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
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
    "--preprocessing_num_workers", "1",
    "--output_dir", $resolvedOutput,
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

Write-Host "GujinBridge V6 targeted SFT configuration:"
Write-Host "  Base (merged V5): $resolvedBaseModel"
Write-Host "  Train: $resolvedTrain"
Write-Host "  Validation: $resolvedValidation"
Write-Host "  Output: $resolvedOutput"
Write-Host "  Epochs: $NumEpochs"
Write-Host "  Learning rate: $LearningRate"
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
        throw "V6 training process exited with code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
