#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two GujinBridge prediction files on the same reference set."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Tuple


def iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
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


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def character_f1(prediction: object, reference: object) -> float:
    prediction_chars = Counter(normalize_text(prediction))
    reference_chars = Counter(normalize_text(reference))
    overlap = sum((prediction_chars & reference_chars).values())
    prediction_total = sum(prediction_chars.values())
    reference_total = sum(reference_chars.values())
    precision = overlap / prediction_total if prediction_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def content_core(value: object) -> str:
    return re.sub(r"[^\w]", "", str(value or ""), flags=re.UNICODE)


def prompt_content(value: object) -> str:
    text = str(value or "")
    positions = [position for mark in ("：", ":") if (position := text.find(mark)) >= 0]
    return text[min(positions) + 1 :].strip() if positions else text.strip()


PUNCTUATION = set(
    "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001"
    "\u2018\u2019\u201c\u201d\uff08\uff09\u300a\u300b\u3008\u3009\u3010\u3011\u2014\u2026"
    ",.!?;:'\"()[]-"
)


def punctuation_boundaries(value: object, include_type: bool = True) -> Counter:
    boundaries: Counter = Counter()
    content_index = 0
    for char in normalize_text(value):
        if char in PUNCTUATION:
            key = (content_index, char) if include_type else content_index
            boundaries[key] += 1
        elif re.match(r"\w", char, flags=re.UNICODE):
            content_index += 1
    return boundaries


def counter_f1(predicted: Counter, expected: Counter) -> float:
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def punctuation_f1(prediction: object, reference: object, include_type: bool = True) -> float:
    return counter_f1(
        punctuation_boundaries(prediction, include_type=include_type),
        punctuation_boundaries(reference, include_type=include_type),
    )


def load_predictions(path: Path) -> Tuple[MutableMapping[str, Deque[str]], int]:
    by_input: MutableMapping[str, Deque[str]] = defaultdict(deque)
    count = 0
    for row in iter_jsonl(path):
        input_text = row.get("Input", row.get("input", ""))
        output_text = row.get("Output", row.get("output", ""))
        key = normalize_text(input_text)
        if not key:
            raise ValueError(f"Prediction {count + 1} in {path} has no input")
        by_input[key].append(str(output_text or ""))
        count += 1
    return by_input, count


def aggregate(rows: Iterable[Mapping[str, object]], prefix: str) -> Dict[str, object]:
    materialized = list(rows)
    if not materialized:
        return {
            "count": 0,
            "exact_match": 0.0,
            "char_f1": 0.0,
            "mean_length_ratio": 0.0,
            "empty_outputs": 0,
            "tag_leaks": 0,
        }
    return {
        "count": len(materialized),
        "exact_match": sum(float(row[f"{prefix}_exact"]) for row in materialized) / len(materialized),
        "char_f1": sum(float(row[f"{prefix}_f1"]) for row in materialized) / len(materialized),
        "mean_length_ratio": sum(float(row[f"{prefix}_length_ratio"]) for row in materialized)
        / len(materialized),
        "empty_outputs": sum(not str(row[f"{prefix}_output"]).strip() for row in materialized),
        "tag_leaks": sum(
            "<think>" in str(row[f"{prefix}_output"]) or "<answer>" in str(row[f"{prefix}_output"])
            for row in materialized
        ),
    }


def compare(
    references: Iterable[Mapping[str, object]],
    baseline_predictions: MutableMapping[str, Deque[str]],
    candidate_predictions: MutableMapping[str, Deque[str]],
    allow_partial: bool = False,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    missing_baseline = 0
    missing_candidate = 0
    for reference in references:
        key = normalize_text(reference.get("input", ""))
        baseline_queue = baseline_predictions.get(key)
        candidate_queue = candidate_predictions.get(key)
        if not baseline_queue:
            missing_baseline += 1
            continue
        if not candidate_queue:
            missing_candidate += 1
            if allow_partial:
                continue
            continue
        baseline_output = baseline_queue.popleft()
        candidate_output = candidate_queue.popleft()
        reference_text = str(reference.get("reference", ""))
        reference_length = max(1, len(normalize_text(reference_text)))
        baseline_f1 = character_f1(baseline_output, reference_text)
        candidate_f1 = character_f1(candidate_output, reference_text)
        rows.append(
            {
                "id": reference.get("id"),
                "task": str(reference.get("task", "unknown")),
                "input": str(reference.get("input", "")),
                "baseline_output": baseline_output,
                "candidate_output": candidate_output,
                "baseline_exact": normalize_text(baseline_output) == normalize_text(reference_text),
                "candidate_exact": normalize_text(candidate_output) == normalize_text(reference_text),
                "baseline_f1": baseline_f1,
                "candidate_f1": candidate_f1,
                "baseline_length_ratio": len(normalize_text(baseline_output)) / reference_length,
                "candidate_length_ratio": len(normalize_text(candidate_output)) / reference_length,
                "baseline_boundary_f1": punctuation_f1(baseline_output, reference_text, True),
                "candidate_boundary_f1": punctuation_f1(candidate_output, reference_text, True),
                "baseline_position_f1": punctuation_f1(baseline_output, reference_text, False),
                "candidate_position_f1": punctuation_f1(candidate_output, reference_text, False),
            }
        )
    if not allow_partial and (missing_baseline or missing_candidate):
        raise ValueError(
            f"Incomplete predictions: missing baseline={missing_baseline}, missing candidate={missing_candidate}"
        )
    if not rows:
        raise RuntimeError("No common predictions were found")

    by_task: MutableMapping[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)

    def section(section_rows: List[Mapping[str, object]]) -> Dict[str, object]:
        baseline = aggregate(section_rows, "baseline")
        candidate = aggregate(section_rows, "candidate")
        wins = sum(float(row["candidate_f1"]) > float(row["baseline_f1"]) + 1e-12 for row in section_rows)
        losses = sum(float(row["baseline_f1"]) > float(row["candidate_f1"]) + 1e-12 for row in section_rows)
        ties = len(section_rows) - wins - losses
        return {
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                "exact_match": float(candidate["exact_match"]) - float(baseline["exact_match"]),
                "char_f1": float(candidate["char_f1"]) - float(baseline["char_f1"]),
                "mean_length_ratio": float(candidate["mean_length_ratio"])
                - float(baseline["mean_length_ratio"]),
            },
            "pairwise_char_f1": {"wins": wins, "ties": ties, "losses": losses},
        }

    punct_rows = by_task.get("punctuate", [])
    punctuation = {}
    punctuation_quality = {}
    if punct_rows:
        baseline_preserved = sum(
            content_core(prompt_content(row["input"])) == content_core(row["baseline_output"])
            for row in punct_rows
        )
        candidate_preserved = sum(
            content_core(prompt_content(row["input"])) == content_core(row["candidate_output"])
            for row in punct_rows
        )
        punctuation = {
            "count": len(punct_rows),
            "baseline_preserved": baseline_preserved,
            "baseline_preservation_rate": baseline_preserved / len(punct_rows),
            "candidate_preserved": candidate_preserved,
            "candidate_preservation_rate": candidate_preserved / len(punct_rows),
        }
        baseline_boundary = sum(float(row["baseline_boundary_f1"]) for row in punct_rows) / len(punct_rows)
        candidate_boundary = sum(float(row["candidate_boundary_f1"]) for row in punct_rows) / len(punct_rows)
        baseline_position = sum(float(row["baseline_position_f1"]) for row in punct_rows) / len(punct_rows)
        candidate_position = sum(float(row["candidate_position_f1"]) for row in punct_rows) / len(punct_rows)
        boundary_wins = sum(
            float(row["candidate_boundary_f1"]) > float(row["baseline_boundary_f1"]) + 1e-12
            for row in punct_rows
        )
        boundary_losses = sum(
            float(row["baseline_boundary_f1"]) > float(row["candidate_boundary_f1"]) + 1e-12
            for row in punct_rows
        )
        punctuation_quality = {
            "count": len(punct_rows),
            "baseline_boundary_f1": baseline_boundary,
            "candidate_boundary_f1": candidate_boundary,
            "boundary_f1_delta": candidate_boundary - baseline_boundary,
            "baseline_position_f1": baseline_position,
            "candidate_position_f1": candidate_position,
            "position_f1_delta": candidate_position - baseline_position,
            "pairwise_boundary_f1": {
                "wins": boundary_wins,
                "ties": len(punct_rows) - boundary_wins - boundary_losses,
                "losses": boundary_losses,
            },
        }

    return {
        "matching": {
            "scored": len(rows),
            "missing_baseline": missing_baseline,
            "missing_candidate": missing_candidate,
        },
        "overall": section(rows),
        "by_task": {task: section(task_rows) for task, task_rows in sorted(by_task.items())},
        "punctuation_text_preservation": punctuation,
        "punctuation_quality": punctuation_quality,
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: Mapping[str, object], baseline_name: str, candidate_name: str) -> str:
    lines = [
        f"# GujinBridge {baseline_name} 与 {candidate_name} A/B 评测",
        "",
        f"- Baseline：{baseline_name}",
        f"- Candidate：{candidate_name}",
        f"- 成功配对：{report['matching']['scored']}",
        "",
        "## 字符级回归指标",
        "",
        f"| 任务 | 数量 | {baseline_name} F1 | {candidate_name} F1 | 差值 | "
        f"{candidate_name}胜/平/负 | {baseline_name}长度比 | {candidate_name}长度比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sections = [("overall", "全部"), ("c2m", "c2m"), ("m2c", "m2c"), ("punctuate", "punctuate")]
    for key, label in sections:
        section = report["overall"] if key == "overall" else report["by_task"].get(key)
        if not section:
            continue
        baseline = section["baseline"]
        candidate = section["candidate"]
        pairwise = section["pairwise_char_f1"]
        lines.append(
            f"| {label} | {baseline['count']} | {percent(baseline['char_f1'])} | "
            f"{percent(candidate['char_f1'])} | {section['delta']['char_f1']:+.4f} | "
            f"{pairwise['wins']}/{pairwise['ties']}/{pairwise['losses']} | "
            f"{baseline['mean_length_ratio']:.3f} | {candidate['mean_length_ratio']:.3f} |"
        )
    punctuation = report.get("punctuation_text_preservation") or {}
    if punctuation:
        lines.extend(
            [
                "",
                "## 标点原文字保留",
                "",
                f"- {baseline_name}：{punctuation['baseline_preserved']}/{punctuation['count']} "
                f"({percent(punctuation['baseline_preservation_rate'])})",
                f"- {candidate_name}：{punctuation['candidate_preserved']}/{punctuation['count']} "
                f"({percent(punctuation['candidate_preservation_rate'])})",
            ]
        )
    punctuation_quality = report.get("punctuation_quality") or {}
    if punctuation_quality:
        pairwise = punctuation_quality["pairwise_boundary_f1"]
        lines.extend(
            [
                "",
                "## Punctuation boundary metrics",
                "",
                f"- Typed boundary F1: {baseline_name} "
                f"{percent(punctuation_quality['baseline_boundary_f1'])} -> {candidate_name} "
                f"{percent(punctuation_quality['candidate_boundary_f1'])} "
                f"({punctuation_quality['boundary_f1_delta']:+.4f})",
                f"- Position-only F1: {baseline_name} "
                f"{percent(punctuation_quality['baseline_position_f1'])} -> {candidate_name} "
                f"{percent(punctuation_quality['candidate_position_f1'])} "
                f"({punctuation_quality['position_f1_delta']:+.4f})",
                f"- Typed-boundary wins/ties/losses: "
                f"{pairwise['wins']}/{pairwise['ties']}/{pairwise['losses']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "字符 F1 只适合做同测试集回归比较，不能替代翻译语义人工审核。",
            f"{candidate_name}胜/平/负表示同一样本上 {candidate_name} 字符 F1 "
            f"高于、等于或低于 {baseline_name}。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GujinBridge V5 and DPO predictions")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--baseline-name", default="V5 SFT")
    parser.add_argument("--candidate-name", default="DPO V1")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    baseline, baseline_count = load_predictions(args.baseline_predictions)
    candidate, candidate_count = load_predictions(args.candidate_predictions)
    report = compare(iter_jsonl(args.references), baseline, candidate, args.allow_partial)
    report["prediction_counts"] = {"baseline": baseline_count, "candidate": candidate_count}
    report["models"] = {"baseline": args.baseline_name, "candidate": args.candidate_name}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(
        render_markdown(report, args.baseline_name, args.candidate_name), encoding="utf-8"
    )
    print(f"Compared {report['matching']['scored']} predictions")
    print(f"JSON: {args.output_json}")
    print(f"Report: {args.output_markdown}")


if __name__ == "__main__":
    main()
