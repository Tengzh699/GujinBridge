#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a heuristic error analysis and a human-review queue for GujinBridge.

The automatic labels in this module are triage signals, not semantic judgments.
They are intended to make a smaller, high-value sample for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple


PUNCTUATION = re.compile(r"[^\w]", flags=re.UNICODE)
SPACE = re.compile(r"\s+")
NUMBER_MARKER = re.compile(
    r"(?:\d+(?:\.\d+)?(?:年|月|日|人|里|丈|尺|亩|次|国|岁)?|"
    r"[〇零一二三四五六七八九十百千万亿两]+(?:年|月|日|人|里|丈|尺|亩|次|国|岁)|"
    r"正月|初[一二三四五六七八九十]|"
    r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
)
REVIEW_FIELDS = (
    "human_preferred",
    "human_error_labels_v5",
    "human_error_labels_dpo",
    "human_severity_v5",
    "human_severity_dpo",
    "human_notes",
)


def iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            yield value


def normalize(value: object) -> str:
    return SPACE.sub("", str(value or "")).strip()


def content_core(value: object) -> str:
    return PUNCTUATION.sub("", normalize(value))


def prompt_content(value: object) -> str:
    text = str(value or "")
    positions = [index for mark in ("：", ":") if (index := text.find(mark)) >= 0]
    return text[min(positions) + 1 :].strip() if positions else text.strip()


def character_scores(prediction: object, reference: object) -> Tuple[float, float, float]:
    predicted = Counter(normalize(prediction))
    expected = Counter(normalize(reference))
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_predictions(path: Path) -> Tuple[MutableMapping[str, Deque[str]], int]:
    result: MutableMapping[str, Deque[str]] = defaultdict(deque)
    count = 0
    for row in iter_jsonl(path):
        input_text = row.get("Input", row.get("input", ""))
        output_text = row.get("Output", row.get("output", ""))
        key = normalize(input_text)
        if not key:
            raise ValueError(f"Prediction {count + 1} in {path} has no input")
        result[key].append(str(output_text or ""))
        count += 1
    return result, count


def number_markers(value: object) -> List[str]:
    return NUMBER_MARKER.findall(str(value or ""))


def classify_output(task: str, source: str, reference: str, output: str) -> Dict[str, object]:
    precision, recall, f1 = character_scores(output, reference)
    reference_length = max(1, len(normalize(reference)))
    length_ratio = len(normalize(output)) / reference_length
    flags: List[str] = []

    if not normalize(output):
        flags.append("empty_output")

    if task == "punctuate":
        if content_core(output) != content_core(source):
            flags.append("punctuation_text_changed")
    else:
        if content_core(output) == content_core(source) and content_core(source):
            flags.append("source_echo")
        # A short output is a useful omission signal. Low character recall alone is
        # not: a valid paraphrase can use very different wording from one reference.
        if length_ratio < 0.65:
            flags.append("omission_risk")
        if length_ratio > 1.55:
            flags.append("over_translation_risk")
        if f1 < 0.40:
            flags.append("low_reference_overlap")
        expected_markers = number_markers(reference)
        if expected_markers and number_markers(output) != expected_markers:
            flags.append("number_or_date_mismatch_risk")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "length_ratio": length_ratio,
        "flags": flags,
    }


def priority_score(row: Mapping[str, object]) -> float:
    baseline = row["v5_metrics"]
    candidate = row["dpo_metrics"]
    flags = set(candidate["flags"])
    score = 0.0
    score += max(0.0, -float(row["f1_delta"])) * 8.0
    score += 3.0 if "source_echo" in flags else 0.0
    score += 2.5 if "punctuation_text_changed" in flags else 0.0
    score += 2.0 if "omission_risk" in flags else 0.0
    score += 1.5 if "low_reference_overlap" in flags else 0.0
    score += 1.0 if "number_or_date_mismatch_risk" in flags else 0.0
    score += 0.5 if candidate["flags"] != baseline["flags"] else 0.0
    return round(score, 6)


def analyze(
    references: Iterable[Mapping[str, object]],
    baseline_predictions: MutableMapping[str, Deque[str]],
    candidate_predictions: MutableMapping[str, Deque[str]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    missing_baseline = 0
    missing_candidate = 0

    for reference_row in references:
        input_text = str(reference_row.get("input", ""))
        key = normalize(input_text)
        baseline_queue = baseline_predictions.get(key)
        candidate_queue = candidate_predictions.get(key)
        if not baseline_queue:
            missing_baseline += 1
            continue
        if not candidate_queue:
            missing_candidate += 1
            continue

        task = str(reference_row.get("task", "unknown"))
        reference = str(reference_row.get("reference", ""))
        source_text = prompt_content(input_text)
        baseline_output = baseline_queue.popleft()
        candidate_output = candidate_queue.popleft()
        baseline_metrics = classify_output(task, source_text, reference, baseline_output)
        candidate_metrics = classify_output(task, source_text, reference, candidate_output)
        row: Dict[str, object] = {
            "id": reference_row.get("id"),
            "task": task,
            "source": reference_row.get("source", ""),
            "input": input_text,
            "source_text": source_text,
            "reference": reference,
            "v5_output": baseline_output,
            "dpo_output": candidate_output,
            "v5_metrics": baseline_metrics,
            "dpo_metrics": candidate_metrics,
            "f1_delta": float(candidate_metrics["f1"]) - float(baseline_metrics["f1"]),
        }
        row["comparison_flags"] = []
        if row["f1_delta"] < -0.05:
            row["comparison_flags"].append("dpo_regression")
        elif row["f1_delta"] > 0.05:
            row["comparison_flags"].append("dpo_improvement")
        new_flags = sorted(set(candidate_metrics["flags"]) - set(baseline_metrics["flags"]))
        if new_flags:
            row["comparison_flags"].extend(f"new_{flag}" for flag in new_flags)
        row["priority_score"] = priority_score(row)
        rows.append(row)

    if missing_baseline or missing_candidate:
        raise ValueError(
            f"Incomplete predictions: missing V5={missing_baseline}, missing DPO={missing_candidate}"
        )
    if not rows:
        raise RuntimeError("No common predictions were found")

    def summarize(items: Sequence[Mapping[str, object]], model_key: str) -> Dict[str, object]:
        flag_counts: Counter[str] = Counter()
        for item in items:
            flag_counts.update(item[model_key]["flags"])
        return {"count": len(items), "flag_counts": dict(sorted(flag_counts.items()))}

    by_task: Dict[str, object] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        by_task[task] = {
            "v5": summarize(task_rows, "v5_metrics"),
            "dpo": summarize(task_rows, "dpo_metrics"),
            "dpo_regressions": sum("dpo_regression" in row["comparison_flags"] for row in task_rows),
            "dpo_improvements": sum("dpo_improvement" in row["comparison_flags"] for row in task_rows),
        }

    summary = {
        "method": "heuristic_triage_v1",
        "warning": "自动标签只用于筛选人工审核样本，不能替代语义判断。",
        "sample_count": len(rows),
        "by_task": by_task,
    }
    return summary, rows


def select_review_queue(rows: Sequence[Mapping[str, object]], limit: int) -> List[Dict[str, object]]:
    candidates = [
        row
        for row in rows
        if row["priority_score"] > 0
        or row["v5_metrics"]["flags"]
        or row["dpo_metrics"]["flags"]
    ]
    candidates.sort(
        key=lambda row: (float(row["priority_score"]), -float(row["f1_delta"])), reverse=True
    )
    selected = candidates[:limit] if limit > 0 else candidates
    result: List[Dict[str, object]] = []
    for rank, row in enumerate(selected, 1):
        review = dict(row)
        review["review_rank"] = rank
        review["review_status"] = "pending"
        review.update({field: "" for field in REVIEW_FIELDS})
        result.append(review)
    return result


def render_markdown(summary: Mapping[str, object], queue: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# GujinBridge V5 / DPO 错误归因报告",
        "",
        f"- 冻结测试样本：{summary['sample_count']}",
        f"- 人工审核队列：{len(queue)}",
        "- 注意：下列自动标签是高召回风险提示，不是语义错误的最终结论。",
        "",
        "## 自动风险统计",
        "",
        "| 任务 | 模型 | 原文复述 | 疑似漏译 | 过度扩写 | 低重合 | 数字/日期风险 | 标点改字 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, task_summary in summary["by_task"].items():
        for model in ("v5", "dpo"):
            flags = task_summary[model]["flag_counts"]
            lines.append(
                f"| {task} | {model.upper()} | {flags.get('source_echo', 0)} | "
                f"{flags.get('omission_risk', 0)} | {flags.get('over_translation_risk', 0)} | "
                f"{flags.get('low_reference_overlap', 0)} | "
                f"{flags.get('number_or_date_mismatch_risk', 0)} | "
                f"{flags.get('punctuation_text_changed', 0)} |"
            )
    lines.extend(
        [
            "",
            "## DPO 明显变化",
            "",
            "这里把字符 F1 变化超过 0.05 作为回归/改善候选，仅用于抽样。",
            "",
            "| 任务 | 改善候选 | 回归候选 |",
            "|---|---:|---:|",
        ]
    )
    for task, task_summary in summary["by_task"].items():
        lines.append(
            f"| {task} | {task_summary['dpo_improvements']} | {task_summary['dpo_regressions']} |"
        )
    lines.extend(
        [
            "",
            "## 人工审核方法",
            "",
            "在 `error_review_queue.csv` 中填写以下字段：",
            "",
            "- `human_preferred`：`v5`、`dpo`、`tie` 或 `both_bad`。",
            "- `human_error_labels_v5/dpo`：可多选 `source_echo;omission;hallucination;relation_error;entity_error;over_translation;punctuation_damage;acceptable_variant`。",
            "- `human_severity_v5/dpo`：建议使用 `0`（无错）、`1`（轻微）、`2`（明显）、`3`（严重）。",
            "- `human_notes`：记录判断依据，尤其是主客体、否定、时间和实体问题。",
            "",
            "审核队列按回退幅度及高风险规则排序；不要根据字符 F1 直接决定译文优劣。",
            "",
        ]
    )
    return "\n".join(lines)


def flatten_for_csv(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "review_rank": row["review_rank"],
        "id": row["id"],
        "task": row["task"],
        "source": row["source"],
        "priority_score": row["priority_score"],
        "f1_delta": round(float(row["f1_delta"]), 6),
        "v5_f1": round(float(row["v5_metrics"]["f1"]), 6),
        "dpo_f1": round(float(row["dpo_metrics"]["f1"]), 6),
        "v5_length_ratio": round(float(row["v5_metrics"]["length_ratio"]), 6),
        "dpo_length_ratio": round(float(row["dpo_metrics"]["length_ratio"]), 6),
        "v5_auto_flags": ";".join(row["v5_metrics"]["flags"]),
        "dpo_auto_flags": ";".join(row["dpo_metrics"]["flags"]),
        "comparison_flags": ";".join(row["comparison_flags"]),
        "input": row["input"],
        "reference": row["reference"],
        "v5_output": row["v5_output"],
        "dpo_output": row["dpo_output"],
        "human_preferred": row["human_preferred"],
        "human_error_labels_v5": row["human_error_labels_v5"],
        "human_error_labels_dpo": row["human_error_labels_dpo"],
        "human_severity_v5": row["human_severity_v5"],
        "human_severity_dpo": row["human_severity_dpo"],
        "human_notes": row["human_notes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GujinBridge V5/DPO errors")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--v5-predictions", type=Path, required=True)
    parser.add_argument("--dpo-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-limit", type=int, default=300)
    args = parser.parse_args()

    v5_predictions, v5_count = load_predictions(args.v5_predictions)
    dpo_predictions, dpo_count = load_predictions(args.dpo_predictions)
    summary, rows = analyze(iter_jsonl(args.references), v5_predictions, dpo_predictions)
    summary["prediction_counts"] = {"v5": v5_count, "dpo": dpo_count}
    queue = select_review_queue(rows, args.review_limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "error_analysis.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "error_review_queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_rows = [flatten_for_csv(row) for row in queue]
    with (args.output_dir / "error_review_queue.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)
    (args.output_dir / "ERROR_ANALYSIS.md").write_text(
        render_markdown(summary, queue), encoding="utf-8"
    )

    print(f"Analyzed {summary['sample_count']} paired samples")
    print(f"Review queue: {len(queue)}")
    print(f"Report: {args.output_dir / 'ERROR_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
