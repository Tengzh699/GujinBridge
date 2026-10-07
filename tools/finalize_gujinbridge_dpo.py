#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize reviewed GujinBridge preference records into DPO train/validation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple


APPROVED_STATUSES = {"approved", "accepted", "ready"}


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
                raise ValueError(f"{path}:{line_number} must contain an object")
            yield item


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize(value: object) -> str:
    return " ".join(str(value or "").split())


def validate_pair(record: Mapping[str, object]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return "missing_conversations"
    chosen = normalize(record.get("chosen"))
    rejected = normalize(record.get("rejected"))
    if not chosen or not rejected:
        return "missing_preference"
    if chosen == rejected:
        return "identical_preference"
    return "accepted"


def pair_key(record: Mapping[str, object]) -> str:
    payload = {
        "conversations": record.get("conversations", []),
        "chosen": normalize(record.get("chosen")),
        "rejected": normalize(record.get("rejected")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_key(record: Mapping[str, object]) -> str:
    encoded = json.dumps(
        record.get("conversations", []), ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_pairs(seed_files: Sequence[Path], review_files: Sequence[Path]) -> Tuple[List[Dict[str, object]], Counter]:
    pairs: List[Dict[str, object]] = []
    skipped: Counter = Counter()
    seen = set()

    for path in seed_files:
        for record in iter_jsonl(path):
            reason = validate_pair(record)
            if reason != "accepted":
                skipped[f"seed_{reason}"] += 1
                continue
            key = pair_key(record)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            pairs.append(record)
            seen.add(key)

    for path in review_files:
        for record in iter_jsonl(path):
            if str(record.get("review_status", "")).lower() not in APPROVED_STATUSES:
                skipped["review_not_approved"] += 1
                continue
            reason = validate_pair(record)
            if reason != "accepted":
                skipped[f"review_{reason}"] += 1
                continue
            key = pair_key(record)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            output = dict(record)
            output.setdefault("pair_source", "reviewed_model_candidates")
            output.setdefault("preference_confidence", "reviewed")
            pairs.append(output)
            seen.add(key)
    return pairs, skipped


def split_by_prompt(
    pairs: Sequence[Dict[str, object]], validation_ratio: float, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    groups: MutableMapping[str, List[Dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        groups[prompt_key(pair)].append(pair)
    keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    validation_count = int(round(len(keys) * validation_ratio))
    if validation_ratio > 0 and len(keys) > 1:
        validation_count = max(1, min(validation_count, len(keys) - 1))
    validation_keys = set(keys[:validation_count])
    train, validation = [], []
    for key in keys:
        (validation if key in validation_keys else train).extend(groups[key])
    return train, validation


def cap_pairs_by_task(
    pairs: Sequence[Dict[str, object]], task_caps: Mapping[str, int], seed: int
) -> Tuple[List[Dict[str, object]], Counter]:
    """Deterministically cap selected tasks while preserving original order."""
    selected_keys = set()
    removed: Counter = Counter()
    grouped: MutableMapping[str, List[Dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair.get("task", "unknown"))].append(pair)

    for task, records in grouped.items():
        cap = task_caps.get(task)
        if cap is None or len(records) <= cap:
            selected_keys.update(pair_key(record) for record in records)
            continue
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{seed}:{task}:{pair_key(record)}".encode("utf-8")
            ).hexdigest(),
        )
        selected_keys.update(pair_key(record) for record in ranked[:cap])
        removed[task] = len(records) - cap

    return [pair for pair in pairs if pair_key(pair) in selected_keys], removed


def parse_task_caps(values: Sequence[str]) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid task cap {value!r}; expected TASK=COUNT")
        task, raw_count = value.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Invalid task cap {value!r}; task is empty")
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"Invalid task cap {value!r}; count must be positive")
        caps[task] = count
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize reviewed GujinBridge DPO data")
    parser.add_argument("--seed-file", type=Path, action="append", default=[])
    parser.add_argument("--reviews", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("data/gujinbridge/dpo_v1_final"))
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--status", default="ready_for_dpo_training")
    parser.add_argument(
        "--max-per-task",
        action="append",
        default=[],
        metavar="TASK=COUNT",
        help="Deterministically cap a task before train/validation splitting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.seed_file and not args.reviews:
        raise ValueError("At least one --seed-file or --reviews file is required")
    for path in [*args.seed_file, *args.reviews]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_path = args.output_dir / "train" / "gujinbridge_dpo_train.jsonl"
    validation_path = args.output_dir / "validation" / "gujinbridge_dpo_validation.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in (train_path, validation_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(map(str, existing)))

    pairs, skipped = collect_pairs(args.seed_file, args.reviews)
    if not pairs:
        raise ValueError("No approved valid DPO pairs were found")
    task_counts_before_caps = Counter(str(pair.get("task", "unknown")) for pair in pairs)
    task_caps = parse_task_caps(args.max_per_task)
    pairs, capped = cap_pairs_by_task(pairs, task_caps, args.seed)
    train, validation = split_by_prompt(pairs, args.validation_ratio, args.seed)
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "project": "GujinBridge",
        "version": "DPO v1 final",
        "status": args.status,
        "total": len(pairs),
        "train": len(train),
        "validation": len(validation),
        "tasks": dict(Counter(str(pair.get("task", "unknown")) for pair in pairs)),
        "tasks_before_caps": dict(task_counts_before_caps),
        "task_caps": task_caps,
        "removed_by_task_caps": dict(capped),
        "pair_sources": dict(Counter(str(pair.get("pair_source", "unknown")) for pair in pairs)),
        "skipped": dict(skipped),
        "inputs": {
            "seed_files": [str(path) for path in args.seed_file],
            "review_files": [str(path) for path in args.reviews],
        },
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Final DPO pairs: {len(pairs)}")
    print(f"  train={len(train)}, validation={len(validation)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
