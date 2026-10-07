#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a conservative, training-ready GujinBridge V5 dataset.

All currently flagged records are removed unless an explicit final review approved
the record. Gold benchmark IDs are verified to remain present and are never removed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Set


SPLITS = ("train", "validation", "test")


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


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique_map(records: Iterable[Mapping[str, object]], label: str) -> Dict[str, Mapping[str, object]]:
    result: Dict[str, Mapping[str, object]] = {}
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            raise ValueError(f"{label} 中存在空 ID")
        if record_id in result:
            raise ValueError(f"{label} 中 ID 重复：{record_id}")
        result[record_id] = record
    return result


def split_path(root: Path, split: str) -> Path:
    return root / split / f"gujinbridge_{split}.jsonl"


def task_counts(records: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(record.get("task", "unknown")) for record in records).items()))


def verify_source_isolation(splits: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    sources = {
        split: {str(record.get("source", "")) for record in records}
        for split, records in splits.items()
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = sources[left] & sources[right]
            if overlap:
                raise ValueError(f"来源泄漏：{left}/{right} 共享 {len(overlap)} 个来源")


def finalize_records(
    records: Iterable[Mapping[str, object]],
    removal_ids: Set[str],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    for source_record in records:
        record = dict(source_record)
        if str(record.get("id", "")) in removal_ids:
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="生成保守清洗后的 GujinBridge V5 最终训练数据")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--audit-flags", required=True, type=Path)
    parser.add_argument("--reviewed", required=True, type=Path)
    parser.add_argument("--gold-benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.audit_flags, args.reviewed, args.gold_benchmark):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{args.input_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空；如需覆盖请添加 --overwrite：{args.output_dir}")

    flagged = unique_map(iter_jsonl(args.audit_flags), "审计文件")
    reviews = unique_map(iter_jsonl(args.reviewed), "审核文件")
    gold = unique_map(iter_jsonl(args.gold_benchmark), "黄金评测集")
    approved_ids = {
        record_id for record_id, review in reviews.items()
        if str(review.get("review_status", "")) == "approved"
    }
    rejected_or_pending = {
        record_id for record_id, review in reviews.items()
        if str(review.get("review_status", "")) != "approved"
    }
    exception_ids = set(flagged) & approved_ids
    removal_ids = set(flagged) - exception_ids
    if removal_ids & set(gold):
        preview = sorted(removal_ids & set(gold))[0]
        raise ValueError(f"黄金评测记录将被删除：{preview}")
    if exception_ids & rejected_or_pending:
        raise AssertionError("内部错误：拒绝或待定记录进入审核豁免")

    output_splits: Dict[str, List[Dict[str, object]]] = {}
    removed_ids_found: Set[str] = set()
    all_output_ids: Set[str] = set()
    removed_log: List[Dict[str, object]] = []
    split_summary: Dict[str, object] = {}
    for split in SPLITS:
        path = split_path(args.input_dir, split)
        if not path.is_file():
            raise FileNotFoundError(f"缺少切分文件：{path}")
        original = list(iter_jsonl(path))
        kept, removed = finalize_records(original, removal_ids)
        for record in removed:
            record_id = str(record["id"])
            removed_ids_found.add(record_id)
            audit = flagged[record_id]
            removed_log.append({
                "id": record_id,
                "task": record.get("task"),
                "source": record.get("source"),
                "split": split,
                "audit_flags": audit.get("audit_flags", []),
                "audit_metrics": audit.get("audit_metrics", {}),
                "removal_reason": "unreviewed audit risk removed by conservative V5 finalization policy",
            })
        kept_ids = {str(record.get("id", "")) for record in kept}
        if len(kept_ids) != len(kept):
            raise ValueError(f"{split} 存在空 ID 或重复 ID")
        if all_output_ids & kept_ids:
            raise ValueError(f"数据 ID 跨切分重复：{sorted(all_output_ids & kept_ids)[0]}")
        all_output_ids.update(kept_ids)
        output_splits[split] = kept
        split_summary[split] = {
            "before": len(original),
            "removed_unreviewed_risks": len(removed),
            "after": len(kept),
            "removed_tasks": task_counts(removed),
            "remaining_tasks": task_counts(kept),
            "remaining_sources": len({str(record.get("source", "")) for record in kept}),
        }

    missing_removals = removal_ids - removed_ids_found
    if missing_removals:
        raise ValueError(f"有 {len(missing_removals)} 条待删除 ID 未找到：{sorted(missing_removals)[0]}")
    missing_gold = set(gold) - all_output_ids
    if missing_gold:
        raise ValueError(f"有 {len(missing_gold)} 条黄金评测记录缺失：{sorted(missing_gold)[0]}")
    verify_source_isolation(output_splits)

    for split, records in output_splits.items():
        write_jsonl(split_path(args.output_dir, split), records)
    write_jsonl(args.output_dir / "removed_unreviewed_risks.jsonl", removed_log)
    exceptions = [
        {
            "id": record_id,
            "task": flagged[record_id].get("task"),
            "source": flagged[record_id].get("source"),
            "audit_flags": flagged[record_id].get("audit_flags", []),
            "review_status": reviews[record_id].get("review_status"),
            "reviewer": reviews[record_id].get("reviewer"),
            "confidence": reviews[record_id].get("confidence"),
            "review_notes": reviews[record_id].get("review_notes"),
        }
        for record_id in sorted(exception_ids)
    ]
    write_jsonl(args.output_dir / "reviewed_audit_exceptions.jsonl", exceptions)

    manifest = {
        "project": "GujinBridge",
        "version": "V5 final",
        "status": "ready_for_training",
        "derived_from": str(args.input_dir),
        "audit_flags": str(args.audit_flags),
        "reviewed_decisions": str(args.reviewed),
        "gold_benchmark": str(args.gold_benchmark),
        "policy": "Remove every currently flagged record unless an explicit final review approved it.",
        "flagged_before": len(flagged),
        "reviewed_exceptions_kept": len(exception_ids),
        "unreviewed_risks_removed": len(removal_ids),
        "gold_benchmark_records_preserved": len(gold),
        "remaining_records": sum(len(records) for records in output_splits.values()),
        "splits": split_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
