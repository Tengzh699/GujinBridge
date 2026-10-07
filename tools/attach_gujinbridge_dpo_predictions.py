#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attach one or more ordered batch-inference outputs to the DPO review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Mapping


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


def normalize(value: object) -> str:
    return " ".join(str(value or "").split())


def prediction_input(record: Mapping[str, object]) -> str:
    for key in ("Input", "input", "prompt"):
        if key in record:
            return normalize(record[key])
    return ""


def prediction_output(record: Mapping[str, object]) -> str:
    for key in ("Output", "output", "prediction", "response"):
        if key in record:
            return str(record[key]).strip()
    return ""


def attach_predictions(
    queue: List[Dict[str, object]], prediction_sets: List[List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    for predictions in prediction_sets:
        if len(predictions) != len(queue):
            raise ValueError(
                f"Prediction count {len(predictions)} does not match queue count {len(queue)}"
            )

    output_records: List[Dict[str, object]] = []
    for index, source in enumerate(queue):
        record = dict(source)
        expected_prompt = ""
        conversations = record.get("conversations", [])
        if isinstance(conversations, list):
            expected_prompt = normalize(
                "\n".join(
                    str(message.get("value", ""))
                    for message in conversations
                    if isinstance(message, dict) and message.get("from") in {"human", "user"}
                )
            )

        candidates = []
        for predictions in prediction_sets:
            prediction = predictions[index]
            actual_prompt = prediction_input(prediction)
            if actual_prompt and expected_prompt and actual_prompt != expected_prompt:
                raise ValueError(
                    f"Prompt mismatch at row {index + 1}: queue id={record.get('id')}"
                )
            candidate = prediction_output(prediction)
            if candidate and normalize(candidate) not in {normalize(value) for value in candidates}:
                candidates.append(candidate)
        record["candidate_outputs"] = candidates
        record["review_status"] = "pending_review" if candidates else "generation_failed"
        output_records.append(record)
    return output_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach model outputs to GujinBridge DPO candidates")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.queue.is_file():
        raise FileNotFoundError(args.queue)
    for path in args.predictions:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")

    queue = list(iter_jsonl(args.queue))
    prediction_sets = [list(iter_jsonl(path)) for path in args.predictions]
    records = attach_predictions(queue, prediction_sets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    generated = sum(bool(record.get("candidate_outputs")) for record in records)
    print(f"Review candidates: {len(records)}, with outputs: {generated}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
