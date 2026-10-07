#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare source-isolated punctuation data for GujinBridge GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence


SPACE = re.compile(r"\s+")
CONTENT = re.compile(r"[^\w]", flags=re.UNICODE)


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
                raise ValueError(f"{path}:{line_number} must be an object")
            yield value


def normalize(value: object) -> str:
    return SPACE.sub("", str(value or "")).strip()


def content_core(value: object) -> str:
    return CONTENT.sub("", normalize(value))


def conversation_text(record: Mapping[str, object], role: str) -> str:
    aliases = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    values = [
        str(turn.get("value", ""))
        for turn in record.get("conversations", [])
        if isinstance(turn, dict) and str(turn.get("from", "")) in aliases
    ]
    return values[-1] if values else ""


def prompt_content(prompt: str) -> str:
    positions = [index for mark in ("：", ":") if (index := prompt.find(mark)) >= 0]
    return prompt[min(positions) + 1 :].strip() if positions else prompt.strip()


def stable_key(seed: int, record: Mapping[str, object]) -> str:
    material = f"{seed}\0{record.get('id', '')}\0{record.get('source', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def convert_record(record: Mapping[str, object], max_source_chars: int) -> tuple[Dict[str, object] | None, str]:
    if str(record.get("task", "")) != "punctuate":
        return None, "not_punctuation"
    prompt = conversation_text(record, "human")
    answer = conversation_text(record, "gpt")
    source_text = prompt_content(prompt)
    source_length = len(normalize(source_text))
    if not source_text or not answer:
        return None, "empty_text"
    if source_length < 8:
        return None, "source_too_short"
    if source_length > max_source_chars:
        return None, "source_too_long"
    if content_core(source_text) != content_core(answer):
        return None, "reference_changes_source"
    return {
        "id": record.get("id"),
        "source": record.get("source", ""),
        "question": f"为下列古文添加恰当标点，只输出添加标点后的原文：\n\n{source_text}",
        "source_text": source_text,
        "answer": answer,
    }, "accepted"


def select(records: Iterable[Mapping[str, object]], limit: int, seed: int, max_source_chars: int):
    accepted: List[Dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        converted, reason = convert_record(record, max_source_chars)
        reasons[reason] += 1
        if converted is not None:
            accepted.append(converted)
    accepted.sort(key=lambda row: stable_key(seed, row))
    if limit > 0:
        accepted = accepted[:limit]
    return accepted, dict(sorted(reasons.items()))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GujinBridge punctuation GRPO data")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=800)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--max-source-chars", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    def source_path(split: str) -> Path:
        return args.input_dir / split / f"gujinbridge_{split}.jsonl"

    train, train_reasons = select(
        iter_jsonl(source_path("train")), args.train_limit, args.seed, args.max_source_chars
    )
    validation, validation_reasons = select(
        iter_jsonl(source_path("validation")),
        args.validation_limit,
        args.seed + 1,
        args.max_source_chars,
    )
    train_sources = {str(row["source"]) for row in train}
    validation_sources = {str(row["source"]) for row in validation}
    source_overlap = train_sources & validation_sources
    if source_overlap:
        raise ValueError(f"Train/validation source leakage: {sorted(source_overlap)[:5]}")

    train_path = args.output_dir / "train" / "gujinbridge_grpo_train.jsonl"
    validation_path = args.output_dir / "validation" / "gujinbridge_grpo_validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "project": "GujinBridge",
        "version": "GRPO punctuation v1",
        "status": "ready_for_smoke_test",
        "policy": "Only source-preserving punctuation examples from existing train/validation splits.",
        "seed": args.seed,
        "max_source_chars": args.max_source_chars,
        "splits": {
            "train": {"records": len(train), "sources": len(train_sources), "filter_counts": train_reasons},
            "validation": {
                "records": len(validation),
                "sources": len(validation_sources),
                "filter_counts": validation_reasons,
            },
        },
        "checks": {"train_validation_source_overlap": 0},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Train: {len(train)} -> {train_path}")
    print(f"Validation: {len(validation)} -> {validation_path}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
