#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO training for GujinBridge punctuation with verifiable rewards."""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser


os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SPACE = re.compile(r"\s+")
CONTENT = re.compile(r"[^\w]", flags=re.UNICODE)
ALLOWED_PUNCTUATION = set("，。！？；：、‘’“”（）《》〈〉【】〔〕—…·,.!?;:'\"()[]-")
SYSTEM_PROMPT = (
    "你是古今桥的古文标点模块。请为用户给出的古文添加恰当标点。"
    "不得增删、替换或调整任何原文字词，只输出加标点后的原文；"
    "不要解释，不要输出思考过程，不要使用 Markdown 或答案标签。"
)


@dataclass
class ScriptArguments:
    train_file: str = field(metadata={"help": "GRPO training JSONL file"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Optional validation JSONL file"})
    tokenizer_name_or_path: Optional[str] = field(default=None)
    initial_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional adapter to merge into the base before creating the GRPO adapter"},
    )
    train_samples: int = field(default=-1)
    validation_samples: int = field(default=-1)
    preprocessing_num_workers: int = field(default=1)
    reward_version: str = field(
        default="v1",
        metadata={"help": "Reward recipe: v1 reproduces the original run; v2 uses dense punctuation rewards"},
    )


def normalize(value: object) -> str:
    return SPACE.sub("", str(value or "")).strip()


def content_core(value: object) -> str:
    return CONTENT.sub("", normalize(value))


def completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


def character_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, content_core(left), content_core(right)).ratio()


def punctuation_boundaries(value: object) -> Counter[tuple[int, str]]:
    boundaries: Counter[tuple[int, str]] = Counter()
    content_index = 0
    for char in normalize(value):
        if char in ALLOWED_PUNCTUATION:
            boundaries[(content_index, char)] += 1
        elif re.match(r"\w", char, flags=re.UNICODE):
            content_index += 1
    return boundaries


def boundary_f1(prediction: str, reference: str) -> float:
    predicted = punctuation_boundaries(prediction)
    expected = punctuation_boundaries(reference)
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def position_boundaries(value: object) -> Counter[int]:
    """Return punctuation positions while ignoring the punctuation mark type."""
    typed = punctuation_boundaries(value)
    positions: Counter[int] = Counter()
    for (content_index, _mark), count in typed.items():
        positions[content_index] += count
    return positions


def position_boundary_f1(prediction: str, reference: str) -> float:
    predicted = position_boundaries(prediction)
    expected = position_boundaries(reference)
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def preservation_reward(completions, source_text, **kwargs) -> List[float]:
    rewards = []
    for completion, source in zip(completions, source_text):
        output = completion_text(completion)
        similarity = character_similarity(output, str(source))
        rewards.append(1.0 if content_core(output) == content_core(source) else 0.5 * similarity)
    return rewards


def punctuation_reward(completions, answer, source_text, **kwargs) -> List[float]:
    rewards = []
    for completion, reference, source in zip(completions, answer, source_text):
        output = completion_text(completion)
        # Incorrect source text cannot obtain a high punctuation reward.
        rewards.append(boundary_f1(output, str(reference)) * character_similarity(output, str(source)))
    return rewards


def exact_reward(completions, answer, **kwargs) -> List[float]:
    return [
        1.0 if normalize(completion_text(completion)) == normalize(reference) else 0.0
        for completion, reference in zip(completions, answer)
    ]


def direct_format_reward(completions, **kwargs) -> List[float]:
    forbidden = ("<think>", "<answer>", "```", "答案：", "解释：")
    rewards = []
    for completion in completions:
        output = completion_text(completion).strip()
        valid = bool(output) and not any(marker in output for marker in forbidden)
        rewards.append(1.0 if valid else 0.0)
    return rewards


def preservation_guard_reward(completions, source_text, **kwargs) -> List[float]:
    """V2 hard guard: any source-character change receives a negative reward."""
    return [
        1.0 if content_core(completion_text(completion)) == content_core(source) else -1.0
        for completion, source in zip(completions, source_text)
    ]


def typed_boundary_reward_v2(completions, answer, source_text, **kwargs) -> List[float]:
    """Dense reward for putting the correct punctuation type at the correct boundary."""
    rewards = []
    for completion, reference, source in zip(completions, answer, source_text):
        output = completion_text(completion)
        preserved = content_core(output) == content_core(source)
        rewards.append(boundary_f1(output, str(reference)) if preserved else 0.0)
    return rewards


def position_boundary_reward_v2(completions, answer, source_text, **kwargs) -> List[float]:
    """Partial credit for a correct punctuation boundary even when its type is wrong."""
    rewards = []
    for completion, reference, source in zip(completions, answer, source_text):
        output = completion_text(completion)
        preserved = content_core(output) == content_core(source)
        rewards.append(position_boundary_f1(output, str(reference)) if preserved else 0.0)
    return rewards


def prepare_dataset(path: str, tokenizer, limit: int, workers: int):
    dataset = load_dataset("json", data_files=path, split="train")
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    num_proc = workers if workers and workers > 1 else None
    return dataset.map(
        lambda row: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ]
        },
        num_proc=num_proc,
        desc="Preparing GujinBridge GRPO prompts",
    )


def target_modules(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig) -> None:
    tokenizer_path = script_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(
        script_args.train_file, tokenizer, script_args.train_samples, script_args.preprocessing_num_workers
    )
    eval_dataset = None
    if script_args.validation_file and training_args.eval_strategy != "no":
        eval_dataset = prepare_dataset(
            script_args.validation_file,
            tokenizer,
            script_args.validation_samples,
            script_args.preprocessing_num_workers,
        )

    dtype = model_args.dtype if model_args.dtype in ("auto", None) else getattr(torch, model_args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
        local_files_only=True,
    )
    if script_args.initial_adapter_path:
        logger.info(f"Merging initial adapter: {script_args.initial_adapter_path}")
        model = PeftModel.from_pretrained(model, script_args.initial_adapter_path).merge_and_unload()
    model.config.use_cache = False

    modules = target_modules(model_args.lora_target_modules)
    peft_config = None
    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=modules,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
    reward_version = script_args.reward_version.lower().strip()
    if reward_version == "v1":
        reward_funcs = [preservation_reward, punctuation_reward, exact_reward, direct_format_reward]
        default_reward_weights = [0.45, 0.35, 0.15, 0.05]
    elif reward_version == "v2":
        reward_funcs = [preservation_guard_reward, typed_boundary_reward_v2, position_boundary_reward_v2]
        default_reward_weights = [0.45, 0.40, 0.15]
    else:
        raise ValueError(f"Unknown reward_version={script_args.reward_version!r}; expected 'v1' or 'v2'")
    if training_args.reward_weights is None:
        training_args.reward_weights = default_reward_weights
    logger.info(
        f"Using GujinBridge GRPO reward recipe {reward_version}: "
        f"{[func.__name__ for func in reward_funcs]} with weights {training_args.reward_weights}"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        reward_processing_classes=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    checkpoint = training_args.resume_from_checkpoint
    if checkpoint is None and Path(training_args.output_dir).is_dir():
        checkpoint = get_last_checkpoint(training_args.output_dir)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"GRPO model saved to {training_args.output_dir}")


def main() -> None:
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
