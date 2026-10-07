#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate GPT/human review decisions and build an approved GujinBridge benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple


VALID_STATUSES = {"approved", "rejected", "needs_review"}
VALID_CONFIDENCE = {"high", "medium", "low"}


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
            value["_review_file"] = str(path)
            value["_review_line"] = line_number
            yield value


def merge_reviews(
    candidates: Sequence[Mapping[str, object]],
    reviews: Iterable[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    candidate_ids = [str(item.get("id", "")) for item in candidates]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("候选集存在空 ID 或重复 ID")
    candidate_map = {str(item["id"]): item for item in candidates}

    review_map: Dict[str, Mapping[str, object]] = {}
    for review in reviews:
        review_id = str(review.get("id", ""))
        location = f"{review.get('_review_file', '<memory>')}:{review.get('_review_line', '?')}"
        if review_id not in candidate_map:
            raise ValueError(f"审核结果包含未知 ID：{review_id}（{location}）")
        if review_id in review_map:
            raise ValueError(f"审核结果 ID 重复：{review_id}（{location}）")
        status = str(review.get("review_status", ""))
        confidence = str(review.get("confidence", ""))
        if status not in VALID_STATUSES:
            raise ValueError(f"{review_id} 的 review_status 非法：{status}")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(f"{review_id} 的 confidence 非法：{confidence}")
        if str(review.get("task", "")) != str(candidate_map[review_id].get("task", "")):
            raise ValueError(f"{review_id} 的 task 与候选集不一致")
        if status != "approved" and not str(review.get("review_notes", "")).strip():
            raise ValueError(f"{review_id} 为 {status}，但 review_notes 为空")
        review_map[review_id] = review

    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in review_map]
    if missing:
        preview = "、".join(missing[:10])
        raise ValueError(f"缺少 {len(missing)} 条审核结果：{preview}")

    merged: List[Dict[str, object]] = []
    approved: List[Dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    approved_confidence_counts: Counter[str] = Counter()
    task_status: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    corrected = 0
    for candidate in candidates:
        review = review_map[str(candidate["id"])]
        status = str(review["review_status"])
        task = str(candidate.get("task", "unknown"))
        approved_reference = str(review.get("approved_reference", "")).strip()
        merged_item = dict(candidate)
        merged_item.update({
            "review_status": status,
            "approved_reference": approved_reference,
            "review_notes": str(review.get("review_notes", "")).strip(),
            "reviewer": str(review.get("reviewer", "")),
            "confidence": str(review.get("confidence", "")),
        })
        merged.append(merged_item)
        status_counts[status] += 1
        confidence_counts[str(review["confidence"])] += 1
        task_status[task][status] += 1
        if status == "approved":
            approved_confidence_counts[str(review["confidence"])] += 1
            final_reference = approved_reference or str(candidate.get("reference", ""))
            if approved_reference:
                corrected += 1
            approved.append({
                "id": candidate.get("id"),
                "task": task,
                "source": candidate.get("source"),
                "category": candidate.get("category", "未分类"),
                "prompt": candidate.get("prompt", ""),
                "reference": final_reference,
                "original_reference": candidate.get("reference", ""),
                "reference_corrected": bool(approved_reference),
                "reviewer": str(review.get("reviewer", "")),
                "review_confidence": str(review.get("confidence", "")),
                "review_notes": str(review.get("review_notes", "")).strip(),
            })

    report: Dict[str, object] = {
        "candidates": len(candidates),
        "reviewed": len(merged),
        "approved_benchmark_records": len(approved),
        "corrected_references": corrected,
        "statuses": dict(sorted(status_counts.items())),
        "confidence": dict(sorted(confidence_counts.items())),
        "approved_confidence": dict(sorted(approved_confidence_counts.items())),
        "approved_high_confidence_records": approved_confidence_counts["high"],
        "tasks": {task: dict(sorted(counts.items())) for task, counts in sorted(task_status.items())},
        "warning": "GPT review is a first pass. Source-critical and needs_review items still require human verification.",
    }
    return merged, approved, report


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="合并并验证 GujinBridge 黄金集审核结果")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--reviews", required=True, nargs="+", type=Path)
    parser.add_argument("--reviewed-output", required=True, type=Path)
    parser.add_argument("--approved-output", required=True, type=Path)
    parser.add_argument("--high-confidence-output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [args.reviewed_output, args.approved_output, args.report]
    if args.high_confidence_output:
        outputs.append(args.high_confidence_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请添加 --overwrite：" + "、".join(map(str, existing)))
    missing_files = [path for path in (args.candidates, *args.reviews) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("输入文件不存在：" + "、".join(map(str, missing_files)))

    candidates = list(iter_jsonl(args.candidates))
    for candidate in candidates:
        candidate.pop("_review_file", None)
        candidate.pop("_review_line", None)
    reviews: List[Dict[str, object]] = []
    for path in args.reviews:
        reviews.extend(iter_jsonl(path))
    merged, approved, report = merge_reviews(candidates, reviews)
    report["candidate_file"] = str(args.candidates)
    report["review_files"] = [str(path) for path in args.reviews]

    write_jsonl(args.reviewed_output, merged)
    write_jsonl(args.approved_output, approved)
    if args.high_confidence_output:
        high_confidence = [
            record for record in approved
            if record.get("review_confidence") == "high"
        ]
        write_jsonl(args.high_confidence_output, high_confidence)
        report["high_confidence_output"] = str(args.high_confidence_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
