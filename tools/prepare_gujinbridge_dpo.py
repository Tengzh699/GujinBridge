#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare reviewed DPO seed pairs and a post-SFT generation queue for GujinBridge.

Only the original SFT *training* split is read. Held-out validation/test and gold
benchmark records are deliberately excluded to avoid evaluation contamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple


ASSISTANT_ROLES = {"assistant", "gpt"}
USER_ROLES = {"human", "user"}


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield item


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def extract_prompt_and_answer(record: Mapping[str, object]) -> Tuple[List[Dict[str, str]], str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no conversations list")

    messages: List[Dict[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", "")).strip()
        value = str(message.get("value", "")).strip()
        if role and value:
            messages.append({"from": role, "value": value})

    answer_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["from"] in ASSISTANT_ROLES:
            answer_index = index
            break
    if answer_index < 0:
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no assistant answer")

    prompt = messages[:answer_index]
    answer = messages[answer_index]["value"]
    if not prompt or not any(message["from"] in USER_ROLES for message in prompt):
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no user prompt")
    return prompt, answer


def parse_task_limits(value: str) -> Dict[str, int]:
    limits: Dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        try:
            task, raw_limit = part.split("=", 1)
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"Invalid task limit: {part!r}; expected task=number") from exc
        if limit < 0:
            raise ValueError(f"Task limit must be non-negative: {part!r}")
        limits[task.strip()] = limit
    if not limits:
        raise ValueError("At least one task limit is required")
    return limits


def load_training_records(train_dir: Path) -> List[Dict[str, object]]:
    files = sorted(train_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under training directory: {train_dir}")
    records: List[Dict[str, object]] = []
    seen_ids = set()
    for path in files:
        for record in iter_jsonl(path):
            record_id = str(record.get("id", "")).strip()
            if not record_id:
                raise ValueError(f"Training record in {path} is missing id")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate training id: {record_id}")
            seen_ids.add(record_id)
            records.append(record)
    return records


def build_reviewed_pairs(
    records_by_id: Mapping[str, Mapping[str, object]],
    corrections: Iterable[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Counter]:
    pairs: List[Dict[str, object]] = []
    skipped: Counter = Counter()
    seen = set()
    for correction in corrections:
        record_id = str(correction.get("id", "")).strip()
        if not record_id or record_id in seen:
            skipped["missing_or_duplicate_id"] += 1
            continue
        if str(correction.get("split", "train")) != "train":
            skipped["not_training_split"] += 1
            continue
        if str(correction.get("confidence", "")).lower() != "high":
            skipped["not_high_confidence"] += 1
            continue
        record = records_by_id.get(record_id)
        if record is None:
            skipped["id_not_found_in_training"] += 1
            continue

        chosen = str(correction.get("approved_reference", "")).strip()
        rejected = str(correction.get("original_reference", "")).strip()
        if not chosen or not rejected or normalize_text(chosen) == normalize_text(rejected):
            skipped["empty_or_identical_pair"] += 1
            continue

        conversations, _ = extract_prompt_and_answer(record)
        pairs.append(
            {
                "id": f"{record_id}#review-correction",
                "source_id": record_id,
                "task": record.get("task", correction.get("task", "unknown")),
                "source": record.get("source", correction.get("source", "未知出处")),
                "category": record.get("category", ""),
                "conversations": conversations,
                "chosen": chosen,
                "rejected": rejected,
                "pair_source": "review_correction",
                "negative_type": "original_incorrect_reference",
                "preference_confidence": "high",
                "reviewer": correction.get("reviewer", ""),
                "review_notes": correction.get("review_notes", ""),
                "audit_flags": correction.get("audit_flags", []),
            }
        )
        seen.add(record_id)
    return pairs, skipped


def reservoir_candidates(
    records: Sequence[Mapping[str, object]],
    limits: Mapping[str, int],
    excluded_ids: set,
    seed: int,
) -> Dict[str, List[Mapping[str, object]]]:
    rng = random.Random(seed)
    selected: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    seen: Counter = Counter()
    for record in records:
        task = str(record.get("task", "unknown"))
        limit = limits.get(task, 0)
        record_id = str(record.get("id", ""))
        if limit <= 0 or record_id in excluded_ids:
            continue
        try:
            extract_prompt_and_answer(record)
        except ValueError:
            continue
        seen[task] += 1
        bucket = selected[task]
        if len(bucket) < limit:
            bucket.append(record)
        else:
            replacement = rng.randrange(seen[task])
            if replacement < limit:
                bucket[replacement] = record
    for task in selected:
        rng.shuffle(selected[task])
    return selected


def make_generation_queue(
    selected: Mapping[str, Sequence[Mapping[str, object]]]
) -> List[Dict[str, object]]:
    queue: List[Dict[str, object]] = []
    for task in sorted(selected):
        for record in selected[task]:
            conversations, reference = extract_prompt_and_answer(record)
            record_id = str(record["id"])
            queue.append(
                {
                    "id": f"{record_id}#v5-candidates",
                    "source_id": record_id,
                    "task": task,
                    "source": record.get("source", "未知出处"),
                    "category": record.get("category", ""),
                    "conversations": conversations,
                    "reference": reference,
                    "candidate_outputs": [],
                    "review_status": "pending_generation",
                    "chosen": "",
                    "rejected": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return queue


def prompt_group_key(pair: Mapping[str, object]) -> str:
    payload = json.dumps(pair.get("conversations", []), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_pairs(
    pairs: Sequence[Dict[str, object]], validation_ratio: float, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    groups: MutableMapping[str, List[Dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        groups[prompt_group_key(pair)].append(pair)
    keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    validation_group_count = int(round(len(keys) * validation_ratio))
    if validation_ratio > 0 and len(keys) > 1:
        validation_group_count = max(1, min(validation_group_count, len(keys) - 1))
    validation_keys = set(keys[:validation_group_count])
    train, validation = [], []
    for key in keys:
        target = validation if key in validation_keys else train
        target.extend(groups[key])
    return train, validation


def ensure_writable_output(output_dir: Path, overwrite: bool) -> None:
    known_outputs = [
        output_dir / "reviewed" / "train" / "gujinbridge_dpo_train.jsonl",
        output_dir / "reviewed" / "validation" / "gujinbridge_dpo_validation.jsonl",
        output_dir / "generation_candidates.jsonl",
        output_dir / "manifest.json",
    ]
    existing = [path for path in known_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(map(str, existing)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GujinBridge DPO seed data and generation queue")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/gujinbridge/processed_v5_final/train"),
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=Path("data/gujinbridge/processed_v5_reviewed/corrected_by_review.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/gujinbridge/dpo_v1"))
    parser.add_argument(
        "--candidate-per-task",
        default="c2m=1000,m2c=1000,punctuate=1000",
        help="Balanced post-SFT generation queue sizes, e.g. c2m=1000,m2c=1000,punctuate=1000",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.train_dir.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {args.train_dir}")
    if not args.corrections.is_file():
        raise FileNotFoundError(f"Correction file does not exist: {args.corrections}")
    ensure_writable_output(args.output_dir, args.overwrite)

    records = load_training_records(args.train_dir)
    records_by_id = {str(record["id"]): record for record in records}
    corrections = list(iter_jsonl(args.corrections))
    reviewed_pairs, skipped = build_reviewed_pairs(records_by_id, corrections)
    reviewed_train, reviewed_validation = split_pairs(
        reviewed_pairs, args.validation_ratio, args.seed
    )

    limits = parse_task_limits(args.candidate_per_task)
    excluded_ids = {str(pair["source_id"]) for pair in reviewed_pairs}
    selected = reservoir_candidates(records, limits, excluded_ids, args.seed)
    generation_queue = make_generation_queue(selected)

    train_path = args.output_dir / "reviewed" / "train" / "gujinbridge_dpo_train.jsonl"
    validation_path = (
        args.output_dir / "reviewed" / "validation" / "gujinbridge_dpo_validation.jsonl"
    )
    queue_path = args.output_dir / "generation_candidates.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(train_path, reviewed_train)
    write_jsonl(validation_path, reviewed_validation)
    write_jsonl(queue_path, generation_queue)

    manifest = {
        "project": "GujinBridge",
        "version": "DPO v1 preparation",
        "status": "reviewed_seed_ready; model_candidates_pending_generation",
        "data_policy": {
            "source_split": "processed_v5_final/train only",
            "held_out_validation_test_used": False,
            "gold_benchmark_used": False,
            "reviewed_seed": "high-confidence corrected references only",
            "generation_queue": "must receive V5 outputs and preference review before final DPO training",
        },
        "inputs": {
            "train_dir": str(args.train_dir),
            "corrections": str(args.corrections),
        },
        "reviewed_pairs": {
            "total": len(reviewed_pairs),
            "train": len(reviewed_train),
            "validation": len(reviewed_validation),
            "tasks": dict(Counter(str(pair["task"]) for pair in reviewed_pairs)),
            "skipped": dict(skipped),
        },
        "generation_queue": {
            "total": len(generation_queue),
            "tasks": dict(Counter(str(item["task"]) for item in generation_queue)),
            "requested_limits": limits,
        },
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Reviewed DPO pairs: {len(reviewed_pairs)}")
    print(f"  train={len(reviewed_train)}, validation={len(reviewed_validation)}")
    print(f"Post-SFT generation queue: {len(generation_queue)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
