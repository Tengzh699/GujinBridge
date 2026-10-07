#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export GujinBridge ShareGPT test data for batch inference and scoring."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple


def iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            yield item


def extract_pair(record: Mapping[str, object]) -> Tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"样本 {record.get('id', '<unknown>')} 缺少 conversations")
    prompt = ""
    reference = ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        value = str(message.get("value", "")).strip()
        if role in {"human", "user"}:
            prompt = value
        elif role in {"gpt", "assistant"}:
            reference = value
    if not prompt or not reference:
        raise ValueError(f"样本 {record.get('id', '<unknown>')} 缺少 human/gpt 文本")
    return prompt, reference


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 GujinBridge 测试提示词和参考答案")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--references-file", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(f"测试集不存在：{args.dataset}")
    existing = [path for path in (args.prompts_file, args.references_file) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请添加 --overwrite：" + "、".join(map(str, existing)))

    args.prompts_file.parent.mkdir(parents=True, exist_ok=True)
    args.references_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.prompts_file.open("w", encoding="utf-8", newline="\n") as prompts, args.references_file.open(
        "w", encoding="utf-8", newline="\n"
    ) as references:
        for record in iter_jsonl(args.dataset):
            prompt, reference = extract_pair(record)
            exported_prompt = re.sub(r"\s+", " ", prompt).strip()
            prompts.write(exported_prompt + "\n")
            entry: Dict[str, object] = {
                "id": record.get("id", f"eval-{count:06d}"),
                "input": exported_prompt,
                "reference": reference,
                "task": record.get("task", "unknown"),
                "source": record.get("source", "未知出处"),
            }
            references.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"已导出 {count} 条提示词：{args.prompts_file}")
    print(f"参考答案：{args.references_file}")


if __name__ == "__main__":
    main()
