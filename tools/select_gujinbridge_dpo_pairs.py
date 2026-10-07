#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select high-confidence GujinBridge DPO pairs and prioritize manual review.

The selector is deliberately conservative. It auto-approves only violations that
make the model output unambiguously worse than the reviewed reference. Ambiguous
translation differences remain in the manual-review queue.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


PUNCTUATION_RE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
FORMAT_LEAK_RE = re.compile(r"<\/?(?:think|answer)>|作为(?:一个)?AI|无法回答|抱歉", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+|[一二三四五六七八九十百千万亿两〇零]+(?:年|月|日|岁|人|军|里|丈|枚|次)?")


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
    return SPACE_RE.sub("", str(value or "")).strip()


def text_core(value: object) -> str:
    return PUNCTUATION_RE.sub("", normalize(value))


def extract_source_text(record: Mapping[str, object]) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    user_text = ""
    for message in conversations:
        if isinstance(message, dict) and message.get("from") in {"human", "user"}:
            user_text = str(message.get("value", ""))
    parts = re.split(r"\r?\n\s*\r?\n", user_text, maxsplit=1)
    return parts[-1].strip() if parts else user_text.strip()


def character_f1(prediction: str, reference: str) -> float:
    pred_counter = Counter(normalize(prediction))
    ref_counter = Counter(normalize(reference))
    overlap = sum((pred_counter & ref_counter).values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def has_adjacent_repetition(text: str, minimum: int = 10) -> bool:
    compact = normalize(text)
    if len(compact) < minimum * 2:
        return False
    max_chunk = min(80, len(compact) // 2)
    for size in range(max_chunk, minimum - 1, -1):
        for start in range(0, len(compact) - size * 2 + 1):
            if compact[start : start + size] == compact[start + size : start + size * 2]:
                return True
    return False


def numeric_tokens(text: str) -> set:
    return set(NUMBER_RE.findall(text))


def classify_candidate(
    record: Mapping[str, object], candidate: str
) -> Tuple[List[str], List[str], float, Dict[str, object]]:
    task = str(record.get("task", "unknown"))
    reference = str(record.get("reference", "")).strip()
    source = extract_source_text(record)
    candidate_norm = normalize(candidate)
    reference_norm = normalize(reference)
    source_core = text_core(source)
    candidate_core = text_core(candidate)

    auto_reasons: List[str] = []
    review_flags: List[str] = []
    reference_length = max(1, len(reference_norm))
    length_ratio = len(candidate_norm) / reference_length
    f1 = character_f1(candidate, reference)

    if not candidate_norm:
        auto_reasons.append("empty_output")
    if FORMAT_LEAK_RE.search(candidate):
        auto_reasons.append("format_or_refusal_leak")
    if has_adjacent_repetition(candidate):
        auto_reasons.append("adjacent_repetition")

    if task == "punctuate":
        reference_core = text_core(reference)
        if source_core and reference_core != source_core:
            review_flags.append("reference_text_changed")
        elif source_core and candidate_core != source_core:
            auto_reasons.append("punctuation_text_changed")
    else:
        if task == "c2m" and source_core and candidate_core == source_core:
            review_flags.append("source_echo_without_translation")
        if length_ratio < 0.30:
            review_flags.append("severe_omission")
        if length_ratio > 2.75:
            review_flags.append("severe_expansion")

    source_numbers = numeric_tokens(source)
    reference_numbers = numeric_tokens(reference)
    candidate_numbers = numeric_tokens(candidate)
    numeric_mismatch = bool(
        source_numbers
        and reference_numbers == source_numbers
        and candidate_numbers != reference_numbers
    )

    priority = 0.0
    priority += max(0.0, 0.75 - f1) * 4.0
    priority += abs(1.0 - min(length_ratio, 2.0))
    priority += 1.5 if numeric_mismatch else 0.0
    priority += 2.0 if auto_reasons else 0.0
    priority += 1.0 if review_flags else 0.0
    metrics = {
        "candidate_char_f1": round(f1, 6),
        "candidate_reference_length_ratio": round(length_ratio, 6),
        "numeric_mismatch": numeric_mismatch,
    }
    return auto_reasons, review_flags, round(priority, 6), metrics


def select_pairs(records: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Counter]:
    approved: List[Dict[str, object]] = []
    manual: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    counts: Counter = Counter()

    for source_record in records:
        reference = str(source_record.get("reference", "")).strip()
        candidates = source_record.get("candidate_outputs", [])
        if not isinstance(candidates, list) or not candidates:
            item = dict(source_record)
            item["selection_reason"] = "no_candidates"
            skipped.append(item)
            counts["no_candidates"] += 1
            continue

        unique_candidates = []
        seen = set()
        for value in candidates:
            candidate = str(value).strip()
            key = normalize(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)

        usable = [candidate for candidate in unique_candidates if normalize(candidate) != normalize(reference)]
        if not usable:
            item = dict(source_record)
            item["selection_reason"] = "all_candidates_equivalent_to_reference"
            skipped.append(item)
            counts["equivalent"] += 1
            continue

        classified = []
        for candidate in usable:
            auto_reasons, review_flags, priority, metrics = classify_candidate(
                source_record, candidate
            )
            classified.append((candidate, auto_reasons, review_flags, priority, metrics))
        classified.sort(key=lambda item: item[3], reverse=True)
        candidate, auto_reasons, review_flags, priority, metrics = classified[0]

        output = dict(source_record)
        output["candidate_outputs"] = unique_candidates
        output["chosen"] = reference
        output["rejected"] = candidate
        output["selection_reasons"] = auto_reasons
        output["review_flags"] = review_flags
        output["selection_metrics"] = metrics
        output["review_priority"] = priority
        if auto_reasons:
            output["review_status"] = "approved"
            output["pair_source"] = "v5_auto_verified_failure"
            output["preference_confidence"] = "high"
            approved.append(output)
            counts["auto_approved"] += 1
            for reason in auto_reasons:
                counts[f"reason:{reason}"] += 1
        else:
            output["review_status"] = "pending_review"
            output["pair_source"] = "v5_model_candidate"
            output["preference_confidence"] = "pending"
            manual.append(output)
            counts["manual_review"] += 1
            for flag in review_flags:
                counts[f"review_flag:{flag}"] += 1

    manual.sort(key=lambda item: float(item.get("review_priority", 0.0)), reverse=True)
    return approved, manual, skipped, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and prioritize GujinBridge DPO pairs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-review-per-task", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    paths = {
        "approved": args.output_dir / "auto_approved.jsonl",
        "manual": args.output_dir / "manual_review.jsonl",
        "translation_top": args.output_dir / "translation_review_top.jsonl",
        "skipped": args.output_dir / "skipped.jsonl",
        "manifest": args.output_dir / "manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(map(str, existing)))

    records = list(iter_jsonl(args.input))
    approved, manual, skipped, counts = select_pairs(records)
    translation_top: List[Dict[str, object]] = []
    for task in ("c2m", "m2c"):
        task_records = [record for record in manual if record.get("task") == task]
        translation_top.extend(task_records[: args.translation_review_per_task])
    translation_top.sort(
        key=lambda item: float(item.get("review_priority", 0.0)), reverse=True
    )
    write_jsonl(paths["approved"], approved)
    write_jsonl(paths["manual"], manual)
    write_jsonl(paths["translation_top"], translation_top)
    write_jsonl(paths["skipped"], skipped)
    manifest = {
        "project": "GujinBridge",
        "version": "DPO v1 candidate selection",
        "status": "auto_verified_pairs_ready; ambiguous_pairs_require_review",
        "input": str(args.input),
        "total": len(records),
        "auto_approved": len(approved),
        "manual_review": len(manual),
        "translation_review_top": len(translation_top),
        "translation_review_per_task": args.translation_review_per_task,
        "skipped": len(skipped),
        "counts": dict(counts),
        "policy": "Only unambiguous structural failures are auto-approved; semantic preferences require review.",
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
