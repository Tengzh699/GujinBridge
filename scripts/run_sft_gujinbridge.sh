#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_DIR="${TRAIN_DIR:-data/gujinbridge/processed/train}"
VALIDATION_DIR="${VALIDATION_DIR:-data/gujinbridge/processed/validation}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-gujinbridge-sft-qwen3.5-0.8b}"
MAX_STEPS="${MAX_STEPS:--1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "训练目录不存在：$TRAIN_DIR" >&2
  echo "请先准备完整数据，或设置 TRAIN_DIR=data/gujinbridge/demo/train 做冒烟测试。" >&2
  exit 1
fi

if [[ ! -d "$VALIDATION_DIR" ]]; then
  echo "验证目录不存在：$VALIDATION_DIR" >&2
  exit 1
fi

python training/supervised_finetuning.py \
  --model_name_or_path "$BASE_MODEL" \
  --train_file_dir "$TRAIN_DIR" \
  --validation_file_dir "$VALIDATION_DIR" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples -1 \
  --max_eval_samples 1000 \
  --model_max_length 1024 \
  --num_train_epochs 2 \
  --max_steps "$MAX_STEPS" \
  --learning_rate 2e-5 \
  --warmup_steps 20 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache
