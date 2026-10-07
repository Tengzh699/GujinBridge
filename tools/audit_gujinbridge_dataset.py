#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flag suspicious GujinBridge examples for human review without deleting data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CJK_RE = re.compile(r"[\u3400-\u9fff]")
DATE_OR_NUMBER_RE = re.compile(r"\d+|[〇零一二三四五六七八九十百千]+[年月日]")
COMMON_CHARS = set("之其而以于为者也矣乎焉乃则所与及不无有是曰云人将中上下来")
MEANINGFUL_PUNCTUATION = set("，。！？；：、,.!?;:")
SENTENCE_END_PUNCTUATION = set("。！？.!?")
SENTENCE_BOUNDARY_PUNCTUATION = set("。！？；：.!?;:")
CLOSING_QUOTE_OR_BRACKET = set("”’」』》\"")
QUOTE_PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』"}
MIN_PUNCTUATION_DENSITY = 0.05


def quotes_are_balanced(value: str) -> bool:
    closing_to_opening = {closing: opening for opening, closing in QUOTE_PAIRS.items()}
    stack: List[str] = []
    for char in value:
        if char in QUOTE_PAIRS:
            stack.append(char)
        elif char in closing_to_opening:
            if not stack or stack[-1] != closing_to_opening[char]:
                return False
            stack.pop()
    return not stack


def discover_jsonl(inputs: Sequence[Path]) -> List[Path]:
    files = set()
    for path in inputs:
        if path.is_file() and path.suffix.lower() == ".jsonl":
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.jsonl"))
        else:
            raise FileNotFoundError(f"输入不存在或不是 JSONL：{path}")
    if not files:
        raise FileNotFoundError("未找到 JSONL 文件")
    return sorted(files, key=lambda item: str(item).lower())


def iter_jsonl(files: Iterable[Path]) -> Iterator[Dict[str, object]]:
    for path in files:
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
                value["_audit_file"] = str(path)
                value["_audit_line"] = line_number
                yield value


def extract_pair(record: Mapping[str, object]) -> Tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return "", ""
    prompt = ""
    output = ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        value = str(message.get("value", "")).strip()
        if role in {"human", "user"}:
            prompt = value
        elif role in {"gpt", "assistant"}:
            output = value
    input_text = prompt.rsplit("\n\n", 1)[-1].strip() if prompt else ""
    return input_text, output


def normalize_text(value: str) -> str:
    return re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", value).lower()


def cjk_set(value: str) -> set:
    return {char for char in CJK_RE.findall(value) if char not in COMMON_CHARS}


def audit_pair(input_text: str, output_text: str, task: Optional[str] = None) -> Tuple[List[str], Dict[str, float]]:
    flags: List[str] = []
    normalized_input = normalize_text(input_text)
    normalized_output = normalize_text(output_text)
    input_length = len(normalized_input)
    output_length = len(normalized_output)
    ratio = output_length / max(input_length, 1)
    punctuation_count = sum(char in MEANINGFUL_PUNCTUATION for char in output_text)
    punctuation_density = punctuation_count / max(output_length, 1)

    if not input_text or not output_text:
        flags.append("missing_pair")
    if task != "punctuate" and normalized_input and normalized_input == normalized_output:
        flags.append("identical_input_output")
    if task == "punctuate":
        if normalized_input and normalized_output and normalized_input != normalized_output:
            flags.append("punctuation_text_mismatch")
        if input_length >= 20 and punctuation_density < MIN_PUNCTUATION_DENSITY:
            flags.append("insufficient_punctuation")
        if input_length >= 20 and not any(
            char in SENTENCE_BOUNDARY_PUNCTUATION for char in output_text
        ):
            flags.append("missing_sentence_boundary_punctuation")
        if not quotes_are_balanced(output_text):
            flags.append("unbalanced_quotes")
        stripped_input = input_text.lstrip()
        stripped_output = output_text.lstrip()
        if (
            stripped_output
            and stripped_output[0] in SENTENCE_END_PUNCTUATION
            and (not stripped_input or stripped_input[0] not in CLOSING_QUOTE_OR_BRACKET)
        ):
            flags.append("leading_boundary_punctuation")
    if output_length < 2:
        flags.append("output_too_short")
    if input_length >= 8 and (ratio < 0.20 or ratio > 4.0):
        flags.append("extreme_length_ratio")
    if len(CJK_RE.findall(output_text)) < 2:
        flags.append("low_cjk_output")

    input_chars = cjk_set(input_text)
    output_chars = cjk_set(output_text)
    union = input_chars | output_chars
    overlap = len(input_chars & output_chars) / len(union) if union else 0.0
    if min(len(input_chars), len(output_chars)) >= 6 and overlap < 0.05:
        flags.append("low_character_overlap")

    input_numbers = set(DATE_OR_NUMBER_RE.findall(input_text))
    output_numbers = set(DATE_OR_NUMBER_RE.findall(output_text))
    if input_numbers and not input_numbers & output_numbers:
        flags.append("number_or_date_mismatch")

    return flags, {
        "input_length": float(input_length),
        "output_length": float(output_length),
        "length_ratio": round(ratio, 4),
        "character_overlap": round(overlap, 4),
        "punctuation_count": float(punctuation_count),
        "punctuation_density": round(punctuation_density, 4),
    }


def audit_records(records: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    prompt_outputs: MutableMapping[str, set] = defaultdict(set)
    extracted: List[Tuple[str, str]] = []
    for record in records:
        input_text, output_text = extract_pair(record)
        extracted.append((input_text, output_text))
        if input_text:
            prompt_outputs[normalize_text(input_text)].add(normalize_text(output_text))
    conflicting_prompts = {prompt for prompt, outputs in prompt_outputs.items() if len(outputs) > 1}

    flagged: List[Dict[str, object]] = []
    flag_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    task_flagged: Counter[str] = Counter()
    for record, (input_text, output_text) in zip(records, extracted):
        task = str(record.get("task", "unknown"))
        task_counts[task] += 1
        flags, metrics = audit_pair(input_text, output_text, task=task)
        if normalize_text(input_text) in conflicting_prompts:
            flags.append("conflicting_duplicate_prompt")
        if not flags:
            continue
        flag_counts.update(flags)
        task_flagged[task] += 1
        clean_record = {key: value for key, value in record.items() if not key.startswith("_audit_")}
        flagged.append({
            "id": record.get("id"),
            "task": task,
            "source": record.get("source"),
            "audit_flags": sorted(set(flags)),
            "audit_metrics": metrics,
            "input": input_text,
            "output": output_text,
            "record": clean_record,
            "location": {"file": record.get("_audit_file"), "line": record.get("_audit_line")},
        })

    report: Dict[str, object] = {
        "records": len(records),
        "flagged_records": len(flagged),
        "flagged_rate": round(len(flagged) / len(records), 6) if records else 0.0,
        "flag_counts": dict(sorted(flag_counts.items())),
        "tasks": {
            task: {
                "records": count,
                "flagged": task_flagged[task],
                "flagged_rate": round(task_flagged[task] / count, 6) if count else 0.0,
            }
            for task, count in sorted(task_counts.items())
        },
        "policy": "Flags are review candidates, not automatic deletion decisions.",
    }
    return flagged, report


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 GujinBridge 数据中的可疑错配和异常样本")
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--flagged", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    existing = [path for path in (args.report, args.flagged) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请添加 --overwrite：" + "、".join(map(str, existing)))

    files = discover_jsonl(args.input)
    records = list(iter_jsonl(files))
    flagged, report = audit_records(records)
    report["input_files"] = [str(path) for path in files]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.flagged.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.flagged.open("w", encoding="utf-8", newline="\n") as handle:
        for record in flagged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"审计完成：{len(records)} 条，标记 {len(flagged)} 条（{report['flagged_rate']:.2%}）")
    print(f"报告：{args.report}")
    print(f"待复核样本：{args.flagged}")


if __name__ == "__main__":
    main()
