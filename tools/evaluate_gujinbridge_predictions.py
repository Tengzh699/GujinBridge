#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute lightweight exact-match and character-F1 metrics for GujinBridge."""

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
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            yield value


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", "", str(text)).strip()


def character_prf(prediction: str, reference: str) -> Tuple[float, float, float]:
    prediction_chars = Counter(normalize_text(prediction))
    reference_chars = Counter(normalize_text(reference))
    overlap = sum((prediction_chars & reference_chars).values())
    prediction_total = sum(prediction_chars.values())
    reference_total = sum(reference_chars.values())
    precision = overlap / prediction_total if prediction_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


PUNCTUATION = set(
    "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001"
    "\u2018\u2019\u201c\u201d\uff08\uff09\u300a\u300b\u3008\u3009\u3010\u3011\u2014\u2026"
    ",.!?;:'\"()[]-"
)


def content_core(text: object) -> str:
    return re.sub(r"[^\w]", "", normalize_text(text), flags=re.UNICODE)


def prompt_content(text: object) -> str:
    value = str(text or "")
    positions = [position for mark in ("\uff1a", ":") if (position := value.find(mark)) >= 0]
    return value[min(positions) + 1 :].strip() if positions else value.strip()


def punctuation_boundaries(text: object, include_type: bool = True) -> Counter:
    boundaries: Counter = Counter()
    content_index = 0
    for char in normalize_text(text):
        if char in PUNCTUATION:
            key = (content_index, char) if include_type else content_index
            boundaries[key] += 1
        elif re.match(r"\w", char, flags=re.UNICODE):
            content_index += 1
    return boundaries


def counter_prf(predicted: Counter, expected: Counter) -> Tuple[float, float, float]:
    overlap = sum((predicted & expected).values())
    predicted_count = sum(predicted.values())
    expected_count = sum(expected.values())
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def punctuation_prf(prediction: str, reference: str, include_type: bool = True) -> Tuple[float, float, float]:
    return counter_prf(
        punctuation_boundaries(prediction, include_type=include_type),
        punctuation_boundaries(reference, include_type=include_type),
    )


def aggregate_punctuation(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {"count": 0}
    keys = (
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "position_precision",
        "position_recall",
        "position_f1",
        "text_preserved",
        "punctuation_count_abs_error",
    )
    result: Dict[str, float] = {"count": len(materialized)}
    result.update({key: sum(row[key] for row in materialized) / len(materialized) for key in keys})
    result["text_preserved_count"] = int(sum(row["text_preserved"] for row in materialized))
    return result


def aggregate(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {"count": 0, "exact_match": 0.0, "char_precision": 0.0, "char_recall": 0.0, "char_f1": 0.0}
    keys = ("exact_match", "char_precision", "char_recall", "char_f1")
    result: Dict[str, float] = {"count": len(materialized)}
    result.update({key: sum(row[key] for row in materialized) / len(materialized) for key in keys})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 GujinBridge 批量推理结果")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="可选：保存 JSON 指标")
    args = parser.parse_args()

    references_by_input: MutableMapping[str, Deque[Mapping[str, object]]] = defaultdict(deque)
    reference_count = 0
    for reference in iter_jsonl(args.references):
        key = normalize_text(reference.get("input", ""))
        if not key:
            raise ValueError(f"参考样本 {reference.get('id', '<unknown>')} 缺少 input")
        references_by_input[key].append(reference)
        reference_count += 1

    scored: List[Dict[str, float]] = []
    by_task: MutableMapping[str, List[Dict[str, float]]] = defaultdict(list)
    punctuation_rows: List[Dict[str, float]] = []
    unmatched_predictions = 0
    prediction_count = 0
    for prediction in iter_jsonl(args.predictions):
        prediction_count += 1
        input_text = prediction.get("Input", prediction.get("input", ""))
        output_text = prediction.get("Output", prediction.get("output", ""))
        queue = references_by_input.get(normalize_text(input_text))
        if not queue:
            unmatched_predictions += 1
            continue
        reference = queue.popleft()
        reference_text = str(reference.get("reference", ""))
        precision, recall, f1 = character_prf(str(output_text), reference_text)
        row = {
            "exact_match": float(normalize_text(output_text) == normalize_text(reference_text)),
            "char_precision": precision,
            "char_recall": recall,
            "char_f1": f1,
        }
        scored.append(row)
        task = str(reference.get("task", "unknown"))
        by_task[task].append(row)
        if task == "punctuate":
            boundary_precision, boundary_recall, boundary_f1 = punctuation_prf(
                str(output_text), reference_text, include_type=True
            )
            position_precision, position_recall, position_f1 = punctuation_prf(
                str(output_text), reference_text, include_type=False
            )
            punctuation_rows.append(
                {
                    "boundary_precision": boundary_precision,
                    "boundary_recall": boundary_recall,
                    "boundary_f1": boundary_f1,
                    "position_precision": position_precision,
                    "position_recall": position_recall,
                    "position_f1": position_f1,
                    "text_preserved": float(
                        content_core(output_text) == content_core(prompt_content(reference.get("input", "")))
                    ),
                    "punctuation_count_abs_error": float(
                        abs(
                            sum(punctuation_boundaries(output_text).values())
                            - sum(punctuation_boundaries(reference_text).values())
                        )
                    ),
                }
            )

    unmatched_references = sum(len(queue) for queue in references_by_input.values())
    if not scored:
        raise RuntimeError("没有匹配到可评分的预测；请确认预测文件来自对应的 prompts.txt")
    report: Dict[str, object] = {
        "overall": aggregate(scored),
        "by_task": {task: aggregate(rows) for task, rows in sorted(by_task.items())},
        "matching": {
            "references": reference_count,
            "predictions": prediction_count,
            "scored": len(scored),
            "unmatched_references": unmatched_references,
            "unmatched_predictions": unmatched_predictions,
        },
        "punctuation": aggregate_punctuation(punctuation_rows),
        "note": "这些是去除空白后的严格匹配和字符集合 F1，只适合回归测试，不代表完整翻译质量。",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"指标已保存：{args.output}")


if __name__ == "__main__":
    main()
