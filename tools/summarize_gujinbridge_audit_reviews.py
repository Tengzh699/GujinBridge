#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and summarize reviewed GujinBridge audit samples."""

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
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            record["_file"] = str(path)
            record["_line"] = line_number
            yield record


def summarize_reviews(
    candidates: Sequence[Mapping[str, object]],
    reviews: Iterable[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    candidate_ids = [str(candidate.get("id", "")) for candidate in candidates]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("候选集存在空 ID 或重复 ID")
    candidate_map = {str(candidate["id"]): candidate for candidate in candidates}
    review_map: Dict[str, Mapping[str, object]] = {}

    for review in reviews:
        record_id = str(review.get("id", ""))
        location = f"{review.get('_file', '<memory>')}:{review.get('_line', '?')}"
        candidate = candidate_map.get(record_id)
        if candidate is None:
            raise ValueError(f"审核结果包含未知 ID：{record_id}（{location}）")
        if record_id in review_map:
            raise ValueError(f"审核结果 ID 重复：{record_id}（{location}）")
        if str(review.get("task", "")) != str(candidate.get("task", "")):
            raise ValueError(f"{record_id} 的 task 与候选不一致")
        if set(map(str, review.get("audit_flags", []))) != set(map(str, candidate.get("audit_flags", []))):
            raise ValueError(f"{record_id} 的 audit_flags 与候选不一致")
        status = str(review.get("review_status", ""))
        confidence = str(review.get("confidence", ""))
        if status not in VALID_STATUSES:
            raise ValueError(f"{record_id} 的 review_status 非法：{status}")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(f"{record_id} 的 confidence 非法：{confidence}")
        if status != "approved" and not str(review.get("review_notes", "")).strip():
            raise ValueError(f"{record_id} 为 {status}，但 review_notes 为空")
        review_map[record_id] = review

    missing = [record_id for record_id in candidate_ids if record_id not in review_map]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 条审核结果：{'、'.join(missing[:10])}")

    merged: List[Dict[str, object]] = []
    statuses: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    tasks: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    flags: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        record_id = str(candidate["id"])
        review = review_map[record_id]
        status = str(review["review_status"])
        task = str(candidate.get("task", "unknown"))
        merged_record = dict(candidate)
        merged_record.update({
            "review_status": status,
            "approved_reference": str(review.get("approved_reference", "")).strip(),
            "review_notes": str(review.get("review_notes", "")).strip(),
            "reviewer": str(review.get("reviewer", "")),
            "confidence": str(review.get("confidence", "")),
        })
        merged.append(merged_record)
        statuses[status] += 1
        confidence[str(review["confidence"])] += 1
        tasks[task][status] += 1
        for flag in candidate.get("audit_flags", []):
            flags[str(flag)]["records"] += 1
            flags[str(flag)][status] += 1

    report = {
        "candidates": len(candidates),
        "reviewed": len(merged),
        "statuses": dict(sorted(statuses.items())),
        "confidence": dict(sorted(confidence.items())),
        "tasks": {task: dict(sorted(counts.items())) for task, counts in sorted(tasks.items())},
        "flags": {flag: dict(sorted(counts.items())) for flag, counts in sorted(flags.items())},
        "warning": "This is a stratified audit sample, not a benchmark. Use it to estimate flag precision before filtering all records.",
    }
    return merged, report


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 GujinBridge 审计样本的人工或模型复核结果")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--reviews", required=True, nargs="+", type=Path)
    parser.add_argument("--reviewed-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [args.candidates, *args.reviews]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("输入文件不存在：" + "、".join(map(str, missing)))
    outputs = [args.reviewed_output, args.report]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请添加 --overwrite：" + "、".join(map(str, existing)))

    candidates = list(iter_jsonl(args.candidates))
    for candidate in candidates:
        candidate.pop("_file", None)
        candidate.pop("_line", None)
    reviews: List[Dict[str, object]] = []
    for path in args.reviews:
        reviews.extend(iter_jsonl(path))
    merged, report = summarize_reviews(candidates, reviews)
    report["candidate_file"] = str(args.candidates)
    report["review_files"] = [str(path) for path in args.reviews]
    write_jsonl(args.reviewed_output, merged)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
