#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a deterministic, stratified sample of flagged GujinBridge records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Set, Tuple


DEFAULT_TASKS = ("c2m", "m2c")
DEFAULT_FLAGS = (
    "number_or_date_mismatch",
    "low_character_overlap",
    "conflicting_duplicate_prompt",
)


def parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("至少指定一个值")
    return items


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


def stable_hash(seed: int, task: str, flag: str, record_id: str) -> str:
    material = f"{seed}:{task}:{flag}:{record_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def to_candidate(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        "id": record.get("id"),
        "task": record.get("task"),
        "source": record.get("source"),
        "category": (record.get("record") or {}).get("category", "未分类"),
        "prompt": record.get("input", ""),
        "reference": record.get("output", ""),
        "audit_flags": record.get("audit_flags", []),
        "audit_metrics": record.get("audit_metrics", {}),
        "review_status": "pending",
        "approved_reference": "",
        "review_notes": "",
    }


def select_candidates(
    records: Iterable[Mapping[str, object]],
    tasks: Sequence[str],
    flags: Sequence[str],
    per_task_flag: int,
    seed: int,
    allow_short_buckets: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    pools: MutableMapping[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    task_set = set(tasks)
    flag_set = set(flags)
    for record in records:
        task = str(record.get("task", ""))
        if task not in task_set:
            continue
        record_flags = {str(flag) for flag in record.get("audit_flags", [])}
        for flag in flags:
            if flag in record_flags:
                pools[(task, flag)].append(record)

    selected: List[Dict[str, object]] = []
    selected_ids: Set[str] = set()
    bucket_counts: Dict[str, Dict[str, int]] = {}
    for task in tasks:
        bucket_counts[task] = {}
        for flag in flags:
            ordered = sorted(
                pools[(task, flag)],
                key=lambda record: stable_hash(seed, task, flag, str(record.get("id", ""))),
            )
            bucket = []
            for record in ordered:
                record_id = str(record.get("id", ""))
                if not record_id or record_id in selected_ids:
                    continue
                bucket.append(to_candidate(record))
                selected_ids.add(record_id)
                if len(bucket) == per_task_flag:
                    break
            if len(bucket) < per_task_flag and not allow_short_buckets:
                raise ValueError(
                    f"{task}/{flag} 仅有 {len(bucket)} 条可用记录，少于目标 {per_task_flag}"
                )
            selected.extend(bucket)
            bucket_counts[task][flag] = len(bucket)

    report = {
        "records": len(selected),
        "tasks": {task: sum(bucket_counts[task].values()) for task in tasks},
        "buckets": bucket_counts,
    }
    return selected, report


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 GujinBridge 审计记录的分层复核候选")
    parser.add_argument("--audit-flags", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tasks", type=parse_csv, default=DEFAULT_TASKS)
    parser.add_argument("--flags", type=parse_csv, default=DEFAULT_FLAGS)
    parser.add_argument("--per-task-flag", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-short-buckets", action="store_true", help="样本不足的分层桶保留全部可用记录")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.per_task_flag <= 0:
        raise ValueError("--per-task-flag 必须为正整数")
    if not args.audit_flags.is_file():
        raise FileNotFoundError(f"审计记录不存在：{args.audit_flags}")
    outputs = [args.output]
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    outputs.append(manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请添加 --overwrite：" + "、".join(map(str, existing)))

    selected, report = select_candidates(
        iter_jsonl(args.audit_flags),
        tasks=args.tasks,
        flags=args.flags,
        per_task_flag=args.per_task_flag,
        seed=args.seed,
        allow_short_buckets=args.allow_short_buckets,
    )
    write_jsonl(args.output, selected)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "audit_flags": str(args.audit_flags),
                "tasks": list(args.tasks),
                "flags": list(args.flags),
                "per_task_flag": args.per_task_flag,
                "seed": args.seed,
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
