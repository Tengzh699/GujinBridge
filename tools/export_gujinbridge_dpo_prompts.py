#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a GujinBridge DPO generation queue for demo/inference.py."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple


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


def extract_system_and_prompt(record: Mapping[str, object]) -> Tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no conversations")
    system = ""
    user_parts = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", ""))
        value = str(message.get("value", "")).strip()
        if role == "system":
            system = value
        elif role in {"human", "user"}:
            user_parts.append(value)
    prompt = "\n".join(part for part in user_parts if part).strip()
    if not prompt:
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no user prompt")
    return system, prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GujinBridge DPO generation prompts")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.queue.is_file():
        raise FileNotFoundError(args.queue)
    outputs = [args.prompts_file, args.mapping_file, args.system_prompt_file]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(map(str, existing)))

    records = list(iter_jsonl(args.queue))
    systems = set()
    exported = []
    for index, record in enumerate(records):
        system, prompt = extract_system_and_prompt(record)
        if system:
            systems.add(system)
        single_line_prompt = re.sub(r"[\r\n]+", " ", prompt).strip()
        exported.append((record, single_line_prompt))
    if len(systems) > 1:
        raise ValueError("Generation queue contains multiple system prompts; batch inference requires one")

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.prompts_file.write_text(
        "".join(prompt + "\n" for _, prompt in exported), encoding="utf-8"
    )
    with args.mapping_file.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (record, prompt) in enumerate(exported):
            mapping = {
                "index": index,
                "id": record.get("id"),
                "source_id": record.get("source_id"),
                "task": record.get("task"),
                "prompt": prompt,
            }
            handle.write(json.dumps(mapping, ensure_ascii=False) + "\n")
    args.system_prompt_file.write_text(next(iter(systems), "") + "\n", encoding="utf-8")
    print(f"Exported {len(exported)} prompts: {args.prompts_file}")


if __name__ == "__main__":
    main()
