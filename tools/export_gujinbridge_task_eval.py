#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export one task from a GujinBridge reference JSONL for focused evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a GujinBridge task evaluation subset")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-references", type=Path, required=True)
    parser.add_argument("--output-prompts", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with args.references.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{args.references}:{line_number} must be an object")
            if str(row.get("task", "")) == args.task:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found for task: {args.task}")

    args.output_references.parent.mkdir(parents=True, exist_ok=True)
    args.output_prompts.parent.mkdir(parents=True, exist_ok=True)
    with args.output_references.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output_prompts.write_text(
        "\n".join(str(row.get("input", "")).replace("\r", " ").replace("\n", " ") for row in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(rows)} {args.task} examples")


if __name__ == "__main__":
    main()
