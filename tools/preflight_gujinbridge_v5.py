#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run final data and tokenizer checks before GujinBridge V5 retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple

from transformers import AutoConfig, AutoTokenizer


SPLITS = ("train", "validation", "test")
EXPECTED_TASKS = {"c2m", "m2c", "punctuate"}


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            yield record


def split_path(root: Path, split: str) -> Path:
    return root / split / f"gujinbridge_{split}.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: Sequence[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[max(index, 0)]


def extract_messages(record: Mapping[str, object]) -> Tuple[List[Dict[str, str]], str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{record.get('id')} 缺少 conversations")
    messages: List[Dict[str, str]] = []
    answer = ""
    counts: Counter[str] = Counter()
    for item in conversations:
        if not isinstance(item, dict):
            raise ValueError(f"{record.get('id')} 含非法 conversation 项")
        source_role = str(item.get("from", ""))
        value = str(item.get("value", "")).strip()
        if not value:
            raise ValueError(f"{record.get('id')} 含空消息")
        role = {"human": "user", "gpt": "assistant"}.get(source_role, source_role)
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{record.get('id')} 含未知角色：{source_role}")
        counts[role] += 1
        if role == "assistant":
            answer = value
        else:
            messages.append({"role": role, "content": value})
    if counts != Counter({"system": 1, "user": 1, "assistant": 1}):
        raise ValueError(f"{record.get('id')} 的消息结构不是 system/user/assistant 各一条：{dict(counts)}")
    return messages, answer


def validate_splits(data_dir: Path) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
    splits: Dict[str, List[Dict[str, object]]] = {}
    all_ids: Set[str] = set()
    source_sets: Dict[str, Set[str]] = {}
    summary: Dict[str, object] = {}
    for split in SPLITS:
        path = split_path(data_dir, split)
        if not path.is_file():
            raise FileNotFoundError(f"缺少切分文件：{path}")
        records = list(iter_jsonl(path))
        split_ids = [str(record.get("id", "")).strip() for record in records]
        if not all(split_ids) or len(split_ids) != len(set(split_ids)):
            raise ValueError(f"{split} 存在空 ID 或重复 ID")
        overlap = all_ids & set(split_ids)
        if overlap:
            raise ValueError(f"ID 跨切分重复：{sorted(overlap)[0]}")
        all_ids.update(split_ids)
        tasks = Counter(str(record.get("task", "")) for record in records)
        if set(tasks) != EXPECTED_TASKS:
            raise ValueError(f"{split} 任务集合不完整：{sorted(tasks)}")
        source_sets[split] = {str(record.get("source", "")).strip() for record in records}
        if "" in source_sets[split]:
            raise ValueError(f"{split} 存在空 source")
        for record in records:
            extract_messages(record)
        splits[split] = records
        summary[split] = {
            "records": len(records),
            "tasks": dict(sorted(tasks.items())),
            "sources": len(source_sets[split]),
            "sha256": file_sha256(path),
        }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = source_sets[left] & source_sets[right]
            if overlap:
                raise ValueError(f"来源泄漏：{left}/{right} 共享 {len(overlap)} 个来源")
    return splits, summary


def validate_audit_resolution(
    flagged_path: Path,
    reviewed_path: Path,
) -> Tuple[int, int]:
    flagged_ids = {str(record.get("id", "")) for record in iter_jsonl(flagged_path)}
    approved_ids = {
        str(record.get("id", ""))
        for record in iter_jsonl(reviewed_path)
        if str(record.get("review_status", "")) == "approved"
    }
    unresolved = flagged_ids - approved_ids
    if unresolved:
        raise ValueError(f"最终审计仍有 {len(unresolved)} 条未批准风险：{sorted(unresolved)[0]}")
    return len(flagged_ids), len(unresolved)


def validate_gold(
    gold_path: Path,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
) -> int:
    gold_ids = {str(record.get("id", "")) for record in iter_jsonl(gold_path)}
    locations: Dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            record_id = str(record.get("id", ""))
            if record_id in gold_ids:
                locations[record_id] = split
    missing = gold_ids - set(locations)
    if missing:
        raise ValueError(f"黄金评测集缺少 {len(missing)} 条：{sorted(missing)[0]}")
    wrong_split = {record_id for record_id, split in locations.items() if split != "test"}
    if wrong_split:
        raise ValueError(f"黄金评测记录不在 test：{sorted(wrong_split)[0]}")
    return len(gold_ids)


def token_statistics(
    records: Sequence[Mapping[str, object]],
    tokenizer,
    max_length: int,
    progress_label: str,
) -> Dict[str, object]:
    lengths: List[int] = []
    truncated = 0
    for index, record in enumerate(records, 1):
        messages, answer = extract_messages(record)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        source_ids = tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = tokenizer.encode(answer, add_special_tokens=False)
        length = len(source_ids) + len(target_ids) + 1
        lengths.append(length)
        if length > max_length:
            truncated += 1
        if index % 20000 == 0:
            print(f"  {progress_label}: 已检查 {index}/{len(records)} 条", flush=True)
    ordered = sorted(lengths)
    return {
        "records": len(lengths),
        "min": ordered[0] if ordered else 0,
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else 0,
        "over_max_length": truncated,
        "over_max_length_rate": round(truncated / len(lengths), 6) if lengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GujinBridge V5 重训前检查")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--audit-flagged", required=True, type=Path)
    parser.add_argument("--reviewed", required=True, type=Path)
    parser.add_argument("--gold-benchmark", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required_files = (args.audit_flagged, args.reviewed, args.gold_benchmark)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"报告已存在；如需覆盖请添加 --overwrite：{args.report}")
    if args.max_length < 64:
        raise ValueError("--max-length 不能小于 64")

    splits, split_summary = validate_splits(args.data_dir)
    final_flagged, unresolved = validate_audit_resolution(args.audit_flagged, args.reviewed)
    gold_records = validate_gold(args.gold_benchmark, splits)

    AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer 缺少 chat_template")

    token_stats = {
        split: token_statistics(records, tokenizer, args.max_length, split)
        for split, records in splits.items()
    }
    report = {
        "project": "GujinBridge",
        "version": "V5 final",
        "status": "ready_for_training",
        "base_model": args.base_model,
        "tokenizer_path": args.tokenizer_path,
        "model_max_length": args.max_length,
        "records": sum(len(records) for records in splits.values()),
        "splits": split_summary,
        "token_lengths": token_stats,
        "final_audit_flags": final_flagged,
        "unresolved_audit_flags": unresolved,
        "gold_benchmark_records": gold_records,
        "checks": {
            "json_schema": "passed",
            "global_id_uniqueness": "passed",
            "source_isolation": "passed",
            "task_coverage": "passed",
            "gold_benchmark_test_only": "passed",
            "all_remaining_audit_flags_explicitly_approved": "passed",
            "base_model_available_offline": "passed",
            "tokenizer_chat_template": "passed",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
