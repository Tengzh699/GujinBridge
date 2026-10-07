#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a train-only targeted SFT curriculum for GujinBridge V6.

The V6 curriculum intentionally reweights difficult, already-approved training
examples. Validation examples remain validation-only and the frozen test split is
used solely for leakage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


SPACE = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w]", flags=re.UNICODE)
DATE_OR_NUMBER = re.compile(
    r"(?:\d+|[〇零一二三四五六七八九十百千万亿两]+(?:年|月|日|人|军|里|丈|尺|亩|岁)|"
    r"正月|初[一二三四五六七八九十]|"
    r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
)
NEGATION = re.compile(r"[不未无非莫勿弗毋]|尚未|未尝")
RELATION = re.compile(r"故|是以|以故|因|由是|于是|遂|乃|则|虽|然|而|若|苟|盖|既|会")
ENTITY = re.compile(r"帝|王|后|太子|将军|太守|刺史|尚书|丞相|使者|诏|敕|京|郡|州|县|军")
CLAUSE_PUNCT = re.compile(r"[，；：。！？、]")


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
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
    return PUNCT.sub("", normalize(value))


def conversation_text(record: Mapping[str, object], role: str) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    aliases = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    values = [
        str(turn.get("value", ""))
        for turn in conversations
        if isinstance(turn, dict) and str(turn.get("from", "")) in aliases
    ]
    return values[-1] if values else ""


def prompt_content(prompt: str) -> str:
    positions = [index for mark in ("：", ":") if (index := prompt.find(mark)) >= 0]
    return prompt[min(positions) + 1 :].strip() if positions else prompt.strip()


def stable_key(seed: int, namespace: str, record: Mapping[str, object]) -> str:
    material = f"{seed}\0{namespace}\0{record.get('id', '')}\0{record.get('source', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def record_signature(record: Mapping[str, object]) -> str:
    material = content_core(conversation_text(record, "human")) + "\0" + content_core(
        conversation_text(record, "gpt")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def c2m_difficulty(record: Mapping[str, object]) -> Tuple[int, List[str]]:
    source = prompt_content(conversation_text(record, "human"))
    target = conversation_text(record, "gpt")
    source_length = max(1, len(normalize(source)))
    target_length = len(normalize(target))
    score = 0
    reasons: List[str] = []

    if source_length >= 80:
        score += 4
        reasons.append("very_long_source")
    elif source_length >= 40:
        score += 2
        reasons.append("long_source")
    if target_length / source_length >= 1.45:
        score += 2
        reasons.append("expanded_translation")
    if DATE_OR_NUMBER.search(source):
        score += 2
        reasons.append("date_or_number")
    if NEGATION.search(source):
        score += 2
        reasons.append("negation")
    if RELATION.search(source):
        score += 2
        reasons.append("logical_relation")
    if ENTITY.search(source):
        score += 1
        reasons.append("entity_or_title")
    if len(CLAUSE_PUNCT.findall(source)) >= 4:
        score += 2
        reasons.append("multi_clause")
    return score, reasons


def annotate(record: Mapping[str, object], bucket: str, score: int, reasons: Sequence[str]) -> Dict[str, object]:
    result = dict(record)
    result["v6_selection"] = {
        "bucket": bucket,
        "difficulty_score": score,
        "reasons": list(reasons),
    }
    return result


def deterministic_sample(
    records: Sequence[Mapping[str, object]], limit: int, seed: int, namespace: str
) -> List[Mapping[str, object]]:
    ordered = sorted(records, key=lambda row: stable_key(seed, namespace, row))
    return ordered[: min(limit, len(ordered))]


def select_curriculum(
    records: Sequence[Mapping[str, object]],
    c2m_hard: int,
    c2m_general: int,
    m2c_replay: int,
    punctuate_replay: int,
    seed: int,
) -> List[Dict[str, object]]:
    by_task: Dict[str, List[Mapping[str, object]]] = {"c2m": [], "m2c": [], "punctuate": []}
    for record in records:
        task = str(record.get("task", ""))
        if task in by_task:
            by_task[task].append(record)

    scored = []
    for record in by_task["c2m"]:
        score, reasons = c2m_difficulty(record)
        scored.append((score, stable_key(seed, "c2m-hard", record), record, reasons))
    scored.sort(key=lambda item: (-item[0], item[1]))
    hard_items = scored[: min(c2m_hard, len(scored))]
    hard_ids = {str(item[2].get("id", "")) for item in hard_items}
    general_pool = [record for record in by_task["c2m"] if str(record.get("id", "")) not in hard_ids]
    general_items = deterministic_sample(general_pool, c2m_general, seed, "c2m-general")

    selected: List[Dict[str, object]] = [
        annotate(record, "c2m_hard", score, reasons) for score, _, record, reasons in hard_items
    ]
    selected.extend(annotate(record, "c2m_general", 0, ["general_replay"]) for record in general_items)
    selected.extend(
        annotate(record, "m2c_replay", 0, ["capability_replay"])
        for record in deterministic_sample(by_task["m2c"], m2c_replay, seed, "m2c-replay")
    )
    selected.extend(
        annotate(record, "punctuate_replay", 0, ["capability_replay"])
        for record in deterministic_sample(
            by_task["punctuate"], punctuate_replay, seed, "punctuate-replay"
        )
    )
    selected.sort(key=lambda row: stable_key(seed, "final-order", row))
    return selected


def validate_isolation(
    train: Sequence[Mapping[str, object]],
    validation: Sequence[Mapping[str, object]],
    test: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    splits = {"train": train, "validation": validation, "test": test}
    ids = {name: {str(row.get("id", "")) for row in rows} for name, rows in splits.items()}
    sources = {name: {str(row.get("source", "")) for row in rows} for name, rows in splits.items()}
    signatures = {name: {record_signature(row) for row in rows} for name, rows in splits.items()}
    overlaps: Dict[str, Dict[str, int]] = {}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = {
                "id_overlap": len(ids[left] & ids[right]),
                "source_overlap": len(sources[left] & sources[right]),
                "prompt_answer_signature_overlap": len(signatures[left] & signatures[right]),
            }
    passed = all(value == 0 for pair in overlaps.values() for value in pair.values())
    return {"passed": passed, "overlaps": overlaps}


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    task_counts = Counter(str(row.get("task", "unknown")) for row in rows)
    bucket_counts = Counter(
        str(row.get("v6_selection", {}).get("bucket", "unknown")) for row in rows
    )
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row.get("v6_selection", {}).get("reasons", []))
    return {
        "records": len(rows),
        "tasks": dict(sorted(task_counts.items())),
        "buckets": dict(sorted(bucket_counts.items())),
        "selection_reasons": dict(sorted(reason_counts.items())),
        "sources": len({str(row.get("source", "")) for row in rows}),
    }


def render_report(manifest: Mapping[str, object]) -> str:
    lines = [
        "# GujinBridge V6 定向 SFT 数据报告",
        "",
        f"- 状态：`{manifest['status']}`",
        f"- 随机种子：{manifest['seed']}",
        "- 策略：从 V5 的训练/验证池中确定性重采样；冻结测试集只参与泄漏检测。",
        "",
        "## 数据规模",
        "",
        "| Split | 总数 | c2m | m2c | punctuate | 来源数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation"):
        section = manifest["splits"][split]
        tasks = section["tasks"]
        lines.append(
            f"| {split} | {section['records']} | {tasks.get('c2m', 0)} | "
            f"{tasks.get('m2c', 0)} | {tasks.get('punctuate', 0)} | {section['sources']} |"
        )
    lines.extend(["", "## 训练集选择桶", ""])
    for bucket, count in manifest["splits"]["train"]["buckets"].items():
        lines.append(f"- `{bucket}`：{count}")
    lines.extend(
        [
            "",
            "## 防泄漏检查",
            "",
            f"- 结果：`{'passed' if manifest['isolation']['passed'] else 'failed'}`",
        ]
    )
    for pair, values in manifest["isolation"]["overlaps"].items():
        lines.append(
            f"- `{pair}`：ID={values['id_overlap']}，来源={values['source_overlap']}，"
            f"问答签名={values['prompt_answer_signature_overlap']}"
        )
    lines.extend(
        [
            "",
            "## 使用原则",
            "",
            "该数据集用于从 V5 模型进行短程修复性 SFT，不代表新增知识数据。",
            "训练后仍必须在原始 1,440 条冻结测试集上进行 A/B 回归评测。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GujinBridge V6 targeted SFT curriculum")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--train-c2m-hard", type=int, default=3000)
    parser.add_argument("--train-c2m-general", type=int, default=1000)
    parser.add_argument("--train-m2c-replay", type=int, default=1000)
    parser.add_argument("--train-punctuate-replay", type=int, default=1000)
    parser.add_argument("--validation-c2m-hard", type=int, default=300)
    parser.add_argument("--validation-c2m-general", type=int, default=100)
    parser.add_argument("--validation-m2c-replay", type=int, default=100)
    parser.add_argument("--validation-punctuate-replay", type=int, default=100)
    args = parser.parse_args()

    def split_file(split: str) -> Path:
        return args.input_dir / split / f"gujinbridge_{split}.jsonl"

    original_train = list(iter_jsonl(split_file("train")))
    original_validation = list(iter_jsonl(split_file("validation")))
    frozen_test = list(iter_jsonl(split_file("test")))
    train = select_curriculum(
        original_train,
        args.train_c2m_hard,
        args.train_c2m_general,
        args.train_m2c_replay,
        args.train_punctuate_replay,
        args.seed,
    )
    validation = select_curriculum(
        original_validation,
        args.validation_c2m_hard,
        args.validation_c2m_general,
        args.validation_m2c_replay,
        args.validation_punctuate_replay,
        args.seed + 1,
    )
    isolation = validate_isolation(train, validation, frozen_test)
    if not isolation["passed"]:
        raise ValueError(f"Data leakage detected: {isolation['overlaps']}")

    train_path = args.output_dir / "train" / "gujinbridge_train.jsonl"
    validation_path = args.output_dir / "validation" / "gujinbridge_validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "project": "GujinBridge",
        "version": "V6 targeted SFT curriculum v1",
        "status": "ready_for_smoke_test",
        "derived_from": str(args.input_dir),
        "frozen_test_records": len(frozen_test),
        "seed": args.seed,
        "policy": "Train/validation-only deterministic hard-example reweighting with replay.",
        "splits": {"train": summarize(train), "validation": summarize(validation)},
        "isolation": isolation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "V6_DATA_REPORT.md").write_text(
        render_report(manifest), encoding="utf-8"
    )
    print(f"Train: {len(train)} -> {train_path}")
    print(f"Validation: {len(validation)} -> {validation_path}")
    print(f"Leakage check: passed")
    print(f"Report: {args.output_dir / 'V6_DATA_REPORT.md'}")


if __name__ == "__main__":
    main()
