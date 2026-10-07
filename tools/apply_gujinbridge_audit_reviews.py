#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply reviewed audit decisions to GujinBridge dataset splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Set, Tuple


SPLITS = ("train", "validation", "test")
VALID_STATUSES = {"approved", "rejected"}


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


def read_review_map(path: Path) -> Dict[str, Dict[str, object]]:
    reviews: Dict[str, Dict[str, object]] = {}
    for review in iter_jsonl(path):
        record_id = str(review.get("id", "")).strip()
        status = str(review.get("review_status", "")).strip()
        if not record_id:
            raise ValueError(f"审核文件存在空 ID：{path}")
        if record_id in reviews:
            raise ValueError(f"审核文件 ID 重复：{record_id}")
        if status not in VALID_STATUSES:
            raise ValueError(f"{record_id} 的审核状态不是最终状态：{status}")
        reviews[record_id] = review
    if not reviews:
        raise ValueError(f"审核文件为空：{path}")
    return reviews


def get_assistant_message(record: Mapping[str, object]) -> MutableMapping[str, object]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{record.get('id')} 缺少 conversations")
    matches = [
        item for item in conversations
        if isinstance(item, dict) and str(item.get("from", "")) in {"gpt", "assistant"}
    ]
    if len(matches) != 1:
        raise ValueError(f"{record.get('id')} 的助手消息数量不是 1：{len(matches)}")
    return matches[0]


def apply_reviews(
    records: Iterable[Mapping[str, object]],
    review_map: Mapping[str, Mapping[str, object]],
    split: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Set[str]]:
    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    corrected: List[Dict[str, object]] = []
    found_ids: Set[str] = set()

    for source_record in records:
        record = dict(source_record)
        record_id = str(record.get("id", "")).strip()
        review = review_map.get(record_id)
        if review is None:
            kept.append(record)
            continue
        if record_id in found_ids:
            raise ValueError(f"数据集中 ID 重复：{record_id}")
        found_ids.add(record_id)

        assistant = get_assistant_message(record)
        current_reference = str(assistant.get("value", "")).strip()
        reviewed_reference = str(review.get("reference", "")).strip()
        if reviewed_reference and current_reference != reviewed_reference:
            raise ValueError(f"{record_id} 的当前参考答案与审核候选不一致")

        status = str(review["review_status"])
        change_base = {
            "id": record_id,
            "task": record.get("task"),
            "source": record.get("source"),
            "split": split,
            "audit_flags": review.get("audit_flags", []),
            "reviewer": review.get("reviewer", ""),
            "confidence": review.get("confidence", ""),
            "review_notes": review.get("review_notes", ""),
        }
        if status == "rejected":
            removed.append({**change_base, "original_reference": current_reference})
            continue

        approved_reference = str(review.get("approved_reference", "")).strip()
        if approved_reference:
            # conversations 内部对象来自深层结构，需要复制后再修改，避免污染调用者数据。
            record = json.loads(json.dumps(record, ensure_ascii=False))
            assistant = get_assistant_message(record)
            assistant["value"] = approved_reference
            corrected.append({
                **change_base,
                "original_reference": current_reference,
                "approved_reference": approved_reference,
            })
        kept.append(record)
    return kept, removed, corrected, found_ids


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
                raise ValueError(f"来源泄漏：{left} 与 {right} 共享 {len(overlap)} 个来源")


def main() -> None:
    parser = argparse.ArgumentParser(description="将最终审核结论应用到 GujinBridge 数据切分")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path, help="summarize 工具生成的完整审核 JSONL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{args.input_dir}")
    if not args.reviews.is_file():
        raise FileNotFoundError(f"审核文件不存在：{args.reviews}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空；如需覆盖请添加 --overwrite：{args.output_dir}")

    review_map = read_review_map(args.reviews)
    output_splits: Dict[str, List[Dict[str, object]]] = {}
    removed_all: List[Dict[str, object]] = []
    corrected_all: List[Dict[str, object]] = []
    found_all: Set[str] = set()
    split_summary: Dict[str, object] = {}

    for split in SPLITS:
        path = split_path(args.input_dir, split)
        if not path.is_file():
            raise FileNotFoundError(f"缺少切分文件：{path}")
        original = list(iter_jsonl(path))
        kept, removed, corrected, found = apply_reviews(original, review_map, split)
        duplicate_found = found_all & found
        if duplicate_found:
            raise ValueError(f"审核 ID 跨切分重复：{sorted(duplicate_found)[0]}")
        found_all.update(found)
        removed_all.extend(removed)
        corrected_all.extend(corrected)
        output_splits[split] = kept
        split_summary[split] = {
            "before": len(original),
            "removed": len(removed),
            "corrected": len(corrected),
            "after": len(kept),
            "removed_tasks": task_counts(removed),
            "remaining_tasks": task_counts(kept),
        }

    missing = set(review_map) - found_all
    if missing:
        raise ValueError(f"有 {len(missing)} 条审核 ID 未在输入数据中找到：{sorted(missing)[0]}")
    verify_source_isolation(output_splits)

    for split, records in output_splits.items():
        write_jsonl(split_path(args.output_dir, split), records)
    write_jsonl(args.output_dir / "removed_by_review.jsonl", removed_all)
    write_jsonl(args.output_dir / "corrected_by_review.jsonl", corrected_all)

    manifest = {
        "project": "GujinBridge",
        "stage": "V5 reviewed intermediate",
        "derived_from": str(args.input_dir),
        "review_file": str(args.reviews),
        "reviewed_records": len(review_map),
        "removed_records": len(removed_all),
        "corrected_records": len(corrected_all),
        "remaining_records": sum(len(records) for records in output_splits.values()),
        "splits": split_summary,
        "policy": "Only explicitly rejected sampled records are removed; only non-empty approved_reference values are applied.",
        "warning": "Ambiguous records outside the reviewed sample remain and require further review before final V5 training.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
