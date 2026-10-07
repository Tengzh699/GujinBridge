[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$BaseModel = "outputs-gujinbridge-v5-merged",
    [string]$InitialAdapter = "outputs-gujinbridge-dpo-v1-qwen3-1.7b",
    [string]$TokenizerPath = "outputs-gujinbridge-v5-merged",
    [string]$TrainFile = "data/gujinbridge/grpo_v1/train/gujinbridge_grpo_train.jsonl",
    [string]$ValidationFile = "data/gujinbridge/grpo_v1/validation/gujinbridge_grpo_validation.jsonl",
    [string]$OutputDir = "outputs-gujinbridge-grpo-v1-qwen3-1.7b",
    [int]$MaxSteps = 100,
    [string]$CudaDevices = "0",
    [switch]$SmokeTest,
    [switch]$Resume,
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
$adapter = Resolve-ProjectPath $InitialAdapter
$tokenizer = Resolve-ProjectPath $TokenizerPath
$train = Resolve-ProjectPath $TrainFile
$validation = Resolve-ProjectPath $ValidationFile
$output = Resolve-ProjectPath $OutputDir
if ($SmokeTest) { $output = "$output-smoke"; $MaxSteps = 3 }

foreach ($path in @($base, $adapter, $tokenizer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Required model directory does not exist: $path" }
}
foreach ($path in @($train, $validation)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required GRPO data file does not exist: $path" }
}
if (-not $Resume -and (Test-Path -LiteralPath $output) -and (Get-ChildItem -LiteralPath $output -Force | Select-Object -First 1)) {
    throw "Output directory is not empty; use -Resume or choose another OutputDir: $output"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$datasetsCache = Join-Path $repoRoot "cache\datasets"
New-Item -ItemType Directory -Path $datasetsCache -Force | Out-Null
$env:HF_DATASETS_CACHE = $datasetsCache
$trainSamples = if ($SmokeTest) { "16" } else { "-1" }
$saveSteps = if ($SmokeTest) { "3" } else { "25" }
$argsList = @(
    "training/gujinbridge_grpo.py",
    "--model_name_or_path", $base,
    "--tokenizer_name_or_path", $tokenizer,
    "--initial_adapter_path", $adapter,
    "--train_file", $train,
    "--validation_file", $validation,
    "--train_samples", $trainSamples,
    "--preprocessing_num_workers", "1",
    "--output_dir", $output,
    "--max_steps", $MaxSteps.ToString(),
    "--num_train_epochs", "1",
    "--learning_rate", "5e-7",
    "--lr_scheduler_type", "cosine",
    "--warmup_steps", "3",
    "--beta", "0.01",
    "--loss_type", "dapo",
    "--epsilon", "0.2",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--num_generations", "4",
    "--max_completion_length", "384",
    "--temperature", "0.9",
    "--top_p", "0.95",
    "--repetition_penalty", "1.0",
    "--use_vllm", "False",
    "--use_peft", "True",
    "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "--lora_r", "16",
    "--lora_alpha", "32",
    "--lora_dropout", "0.05",
    "--dtype", "bfloat16",
    "--bf16", "True",
    "--gradient_checkpointing", "False",
    "--eval_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", $saveSteps,
    "--save_total_limit", "3",
    "--logging_steps", "1",
    "--logging_first_step", "True",
    "--report_to", "tensorboard",
    "--remove_unused_columns", "False",
    "--log_completions", "True",
    "--num_completions_to_print", "2"
)
# The Python entry point discovers the latest complete checkpoint automatically.
# -Resume only relaxes the non-empty-output guard above.

Write-Host "GujinBridge punctuation GRPO:"
Write-Host "  Base: $base"
Write-Host "  Initial DPO adapter: $adapter"
Write-Host "  Train: $train"
Write-Host "  Output: $output"
Write-Host "  Max steps: $MaxSteps"
Write-Host "  Smoke test: $SmokeTest"
if ($DryRun) {
    Write-Host "Dry run passed. GRPO was not started."
    Write-Host "$PythonExe $($argsList -join ' ')"
    return
}

Push-Location $repoRoot
try {
    & $PythonExe @argsList
    if ($LASTEXITCODE -ne 0) { throw "GRPO training exited with code: $LASTEXITCODE" }
}
finally { Pop-Location }
