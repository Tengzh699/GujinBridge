#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a source-diverse, task-balanced candidate set for human gold review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            yield value


def extract_pair(record: Mapping[str, object]) -> tuple:
    prompt = ""
    reference = ""
    conversations = record.get("conversations", [])
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if message.get("from") in {"human", "user"}:
                prompt = str(message.get("value", "")).strip()
            elif message.get("from") in {"gpt", "assistant"}:
                reference = str(message.get("value", "")).strip()
    return prompt, reference


def stable_hash(seed: int, *parts: object) -> str:
    material = ":".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_candidates(
    records: Sequence[Dict[str, object]],
    per_task: int,
    seed: int,
    excluded_ids: Iterable[str],
) -> List[Dict[str, object]]:
    excluded = set(excluded_ids)
    by_task_source: MutableMapping[str, MutableMapping[str, List[Dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        record_id = str(record.get("id", ""))
        if record_id in excluded:
            continue
        task = str(record.get("task", "unknown"))
        source = str(record.get("source", "未知出处"))
        by_task_source[task][source].append(record)

    selected: List[Dict[str, object]] = []
    for task in sorted(by_task_source):
        queues: Dict[str, Deque[Dict[str, object]]] = {}
        for source, items in by_task_source[task].items():
            ordered = sorted(items, key=lambda item: stable_hash(seed, task, source, item.get("id", "")))
            queues[source] = deque(ordered)
        source_order = sorted(queues, key=lambda source: stable_hash(seed, task, source))
        task_selected = 0
        while task_selected < per_task and any(queues[source] for source in source_order):
            for source in source_order:
                if task_selected >= per_task:
                    break
                if not queues[source]:
                    continue
                record = queues[source].popleft()
                prompt, reference = extract_pair(record)
                if not prompt or not reference:
                    continue
                selected.append({
                    "id": record.get("id"),
                    "task": task,
                    "source": source,
                    "category": record.get("category", "未分类"),
                    "prompt": prompt,
                    "reference": reference,
                    "review_status": "pending",
                    "approved_reference": "",
                    "review_notes": "",
                })
                task_selected += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="导出按任务均衡、来源多样的人工黄金集候选")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-flags", type=Path, help="可选：跳过审计工具已标记的样本")
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.per_task <= 0:
        raise ValueError("--per-task 必须大于 0")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    existing = [path for path in (args.output, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请添加 --overwrite：" + "、".join(map(str, existing)))

    excluded_ids = set()
    if args.audit_flags and args.audit_flags.exists():
        excluded_ids = {str(item.get("id", "")) for item in iter_jsonl(args.audit_flags)}
    records = list(iter_jsonl(args.dataset))
    selected = select_candidates(records, args.per_task, args.seed, excluded_ids)
    dataset_ids = {str(record.get("id", "")) for record in records}
    excluded_in_dataset = len(dataset_ids & excluded_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = Counter(str(item["task"]) for item in selected)
    sources: MutableMapping[str, set] = defaultdict(set)
    for item in selected:
        sources[str(item["task"])].add(str(item["source"]))
    manifest = {
        "dataset": str(args.dataset),
        "audit_flags": str(args.audit_flags) if args.audit_flags else None,
        "audit_flag_catalog_size": len(excluded_ids),
        "excluded_flagged_ids_in_dataset": excluded_in_dataset,
        "seed": args.seed,
        "requested_per_task": args.per_task,
        "selected": len(selected),
        "tasks": {
            task: {"records": count, "sources": len(sources[task])}
            for task, count in sorted(counts.items())
        },
        "status": "candidate_only; every row requires human review before becoming a gold benchmark",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已导出 {len(selected)} 条黄金集候选：{args.output}")
    print(json.dumps(manifest["tasks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
