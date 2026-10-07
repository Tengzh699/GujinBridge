[CmdletBinding()]
param(
    [string]$BaseModel = "Qwen/Qwen3.5-0.8B",
    [string]$TrainDir = "data/gujinbridge/processed/train",
    [string]$ValidationDir = "data/gujinbridge/processed/validation",
    [string]$OutputDir = "outputs-gujinbridge-sft-qwen3.5-0.8b",
    [int]$MaxSteps = -1,
    [string]$CudaDevices = "0"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TrainDir -PathType Container)) {
    throw "训练目录不存在：$TrainDir。请先准备完整数据，或传入 -TrainDir data/gujinbridge/demo/train 做冒烟测试。"
}
if (-not (Test-Path -LiteralPath $ValidationDir -PathType Container)) {
    throw "验证目录不存在：$ValidationDir"
}

$env:CUDA_VISIBLE_DEVICES = $CudaDevices
$trainingArgs = @(
    "training/supervised_finetuning.py",
    "--model_name_or_path", $BaseModel,
    "--train_file_dir", $TrainDir,
    "--validation_file_dir", $ValidationDir,
    "--per_device_train_batch_size", "2",
    "--per_device_eval_batch_size", "1",
    "--do_train",
    "--do_eval",
    "--use_peft", "True",
    "--max_train_samples", "-1",
    "--max_eval_samples", "1000",
    "--model_max_length", "1024",
    "--num_train_epochs", "2",
    "--max_steps", $MaxSteps.ToString(),
    "--learning_rate", "2e-5",
    "--warmup_steps", "20",
    "--weight_decay", "0.05",
    "--logging_strategy", "steps",
    "--logging_steps", "10",
    "--eval_strategy", "steps",
    "--eval_steps", "500",
    "--save_strategy", "steps",
    "--save_steps", "500",
    "--save_total_limit", "3",
    "--gradient_accumulation_steps", "8",
    "--preprocessing_num_workers", "4",
    "--output_dir", $OutputDir,
    "--logging_first_step", "True",
    "--target_modules", "all",
    "--lora_rank", "16",
    "--lora_alpha", "32",
    "--lora_dropout", "0.05",
    "--torch_dtype", "bfloat16",
    "--bf16",
    "--report_to", "tensorboard",
    "--gradient_checkpointing", "True",
    "--cache_dir", "./cache"
)

Write-Host "GujinBridge SFT: $BaseModel -> $OutputDir"
& python @trainingArgs
if ($LASTEXITCODE -ne 0) {
    throw "训练进程退出，代码：$LASTEXITCODE"
}
