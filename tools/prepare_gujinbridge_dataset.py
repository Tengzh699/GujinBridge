#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare GujinBridge SFT data from the Chinese Classical Corpus JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_LIMITS = {"c2m": 60000, "m2c": 40000, "punctuate": 20000}
DEFAULT_SYSTEM_PROMPT = (
    "你是古今桥（GujinBridge），一个专注于中国古典文献的智能助手。你能够进行文言文与现代汉语互译、"
    "断句标点、释义和基于典籍的问答。回答应忠实原文，区分原文、译文与推断；如果无法确认，应明确说明，"
    "不得虚构典籍出处、作者或原文。"
)
UPSTREAM_NAME = "gujilab/chinese-classical-corpus"
UPSTREAM_LICENSE = "CC0-1.0 (instruction data); see upstream for source-data terms"
PUNCTUATION_MARKS = set("，。！？；：、,.!?;:")
SENTENCE_END_PUNCTUATION = set("。！？.!?")
SENTENCE_BOUNDARY_PUNCTUATION = set("。！？；：.!?;:")
CLOSING_QUOTE_OR_BRACKET = set("”’」』》\"")
QUOTE_PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』"}
MIN_PUNCTUATION_DENSITY = 0.05


def parse_task_limits(value: str) -> Dict[str, int]:
    """Parse a comma-separated task=limit specification."""
    limits: Dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"任务上限格式应为 task=数量，收到：{item}")
        task, raw_limit = (part.strip() for part in item.split("=", 1))
        if not task:
            raise argparse.ArgumentTypeError("任务名不能为空")
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{task} 的数量不是整数：{raw_limit}") from exc
        if limit < -1:
            raise argparse.ArgumentTypeError(f"{task} 的数量必须为 -1 或非负整数")
        limits[task] = limit
    if not limits:
        raise argparse.ArgumentTypeError("至少要提供一个 task=数量")
    return limits


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_jsonl_files(inputs: Sequence[Path], output_dir: Path) -> List[Path]:
    """Resolve input files while ensuring generated output cannot be read as input."""
    output_resolved = output_dir.resolve()
    files = set()
    for input_path in inputs:
        resolved = input_path.resolve()
        if resolved == output_resolved or _is_relative_to(output_resolved, resolved):
            raise ValueError(f"输出目录不能位于输入目录内部：{output_resolved}")
        if _is_relative_to(resolved, output_resolved):
            raise ValueError(f"输入不能位于输出目录内部：{resolved}")
        if input_path.is_file():
            if input_path.suffix.lower() != ".jsonl":
                raise ValueError(f"输入文件不是 JSONL：{input_path}")
            files.add(resolved)
        elif input_path.is_dir():
            files.update(path.resolve() for path in input_path.rglob("*.jsonl"))
        else:
            raise FileNotFoundError(f"输入不存在：{input_path}")
    result = sorted(files, key=lambda path: str(path).lower())
    if not result:
        raise FileNotFoundError("没有在输入路径中找到 .jsonl 文件")
    return result


def iter_jsonl(files: Sequence[Path], stats: MutableMapping[str, int]) -> Iterator[Mapping[str, object]]:
    for path in files:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                stats["raw_lines"] += 1
                if not line.strip():
                    stats["blank_lines"] += 1
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue
                if not isinstance(item, dict):
                    stats["non_object"] += 1
                    continue
                yield item


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _stable_id(record: Mapping[str, object]) -> str:
    material = "\u241f".join(
        str(record.get(field, "")) for field in ("task", "source", "instruction", "input", "output")
    )
    return "gujinbridge-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def normalize_content(value: str) -> str:
    """Remove formatting and punctuation while preserving content characters."""
    return "".join(char.lower() for char in value if char.isalnum())


def quotes_are_balanced(value: str) -> bool:
    """Validate paired quote order, including nested Chinese quotation marks."""
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


def punctuation_quality_reason(input_text: str, output_text: str) -> Optional[str]:
    """Return a deterministic rejection reason for structurally bad punctuation pairs."""
    normalized_input = normalize_content(input_text)
    normalized_output = normalize_content(output_text)
    if normalized_input and normalized_output and normalized_input != normalized_output:
        return "punctuation_text_mismatch"

    punctuation_count = sum(char in PUNCTUATION_MARKS for char in output_text)
    punctuation_density = punctuation_count / max(len(normalized_output), 1)
    if len(normalized_input) >= 20 and punctuation_density < MIN_PUNCTUATION_DENSITY:
        return "insufficient_punctuation"
    if len(normalized_input) >= 20 and not any(
        char in SENTENCE_BOUNDARY_PUNCTUATION for char in output_text
    ):
        return "missing_sentence_boundary_punctuation"
    if not quotes_are_balanced(output_text):
        return "unbalanced_quotes"

    stripped_input = input_text.lstrip()
    stripped_output = output_text.lstrip()
    if (
        stripped_output
        and stripped_output[0] in SENTENCE_END_PUNCTUATION
        and (not stripped_input or stripped_input[0] not in CLOSING_QUOTE_OR_BRACKET)
    ):
        return "leading_boundary_punctuation"
    return None


def convert_record(
    record: Mapping[str, object],
    system_prompt: str,
    supported_tasks: Iterable[str],
    min_input_chars: int = 4,
    max_input_chars: int = 500,
    keep_box: bool = False,
    filter_punctuation_quality: bool = True,
) -> Tuple[Optional[Dict[str, object]], str]:
    """Validate and convert one upstream row into the ShareGPT structure."""
    task = str(record.get("task", "")).strip()
    if task not in supported_tasks:
        return None, "unsupported_task"
    if not keep_box and _truthy(record.get("_has_box", False)):
        return None, "has_box"

    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()
    if not instruction or not input_text or not output_text:
        return None, "missing_text"
    if len(input_text) < min_input_chars:
        return None, "input_too_short"
    if max_input_chars > 0 and len(input_text) > max_input_chars:
        return None, "input_too_long"
    if task == "punctuate" and filter_punctuation_quality:
        quality_reason = punctuation_quality_reason(input_text, output_text)
        if quality_reason:
            return None, quality_reason

    record_id = str(record.get("id", "")).strip() or _stable_id(record)
    source = str(record.get("source", "")).strip() or f"未知出处/{record_id}"
    category = str(record.get("category", "")).strip() or "未分类"
    converted: Dict[str, object] = {
        "id": record_id,
        "task": task,
        "source": source,
        "category": category,
        "upstream": UPSTREAM_NAME,
        "license": UPSTREAM_LICENSE,
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": f"{instruction}\n\n{input_text}"},
            {"from": "gpt", "value": output_text},
        ],
    }
    return converted, "accepted"


def reservoir_sample(
    records: Iterable[Mapping[str, object]],
    limits: Mapping[str, int],
    system_prompt: str,
    seed: int,
    min_input_chars: int,
    max_input_chars: int,
    keep_box: bool,
    stats: MutableMapping[str, int],
    filter_punctuation_quality: bool = True,
) -> List[Dict[str, object]]:
    """Keep a deterministic bounded sample per task without loading all raw rows."""
    reservoirs: Dict[str, List[Dict[str, object]]] = {task: [] for task in limits}
    accepted_seen: Counter[str] = Counter()
    rngs = {
        task: random.Random(seed + int(hashlib.sha256(task.encode("utf-8")).hexdigest()[:8], 16))
        for task in limits
    }

    for record in records:
        converted, reason = convert_record(
            record,
            system_prompt=system_prompt,
            supported_tasks=limits,
            min_input_chars=min_input_chars,
            max_input_chars=max_input_chars,
            keep_box=keep_box,
            filter_punctuation_quality=filter_punctuation_quality,
        )
        stats[reason] += 1
        if converted is None:
            continue
        task = str(converted["task"])
        limit = limits[task]
        accepted_seen[task] += 1
        if limit == 0:
            continue
        if limit == -1 or len(reservoirs[task]) < limit:
            reservoirs[task].append(converted)
            continue
        replacement = rngs[task].randrange(accepted_seen[task])
        if replacement < limit:
            reservoirs[task][replacement] = converted

    sampled: List[Dict[str, object]] = []
    for task in limits:
        stats[f"accepted_{task}"] = accepted_seen[task]
        stats[f"sampled_{task}"] = len(reservoirs[task])
        sampled.extend(reservoirs[task])
    return sampled


def split_by_source(
    records: Sequence[Dict[str, object]],
    seed: int = 42,
    train_ratio: float = 0.98,
    validation_ratio: float = 0.01,
    test_ratio: float = 0.01,
    min_task_sources_per_eval_split: int = 0,
    task_min_sources_per_eval_split: Optional[Mapping[str, int]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Split complete source groups while guaranteeing task coverage when feasible."""
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9 or min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("train/validation/test 比例必须为非负数且总和为 1")
    if min_task_sources_per_eval_split < 0:
        raise ValueError("min_task_sources_per_eval_split 不能为负数")
    task_minimums = dict(task_min_sources_per_eval_split or {})
    if any(value < 0 for value in task_minimums.values()):
        raise ValueError("task_min_sources_per_eval_split 不能包含负数")

    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        source = str(record.get("source", "")).strip() or str(record.get("id", "未知出处"))
        groups[source].append(record)

    split_names = ("train", "validation", "test")
    ratios = (train_ratio, validation_ratio, test_ratio)
    result: Dict[str, List[Dict[str, object]]] = {name: [] for name in split_names}
    source_tasks: Dict[str, set] = {
        source: {str(item.get("task", "unknown")) for item in items}
        for source, items in groups.items()
    }
    task_sources: Dict[str, set] = defaultdict(set)
    for source, tasks in source_tasks.items():
        for task in tasks:
            task_sources[task].add(source)

    def stable_key(*parts: object) -> str:
        material = ":".join(str(part) for part in (seed, *parts))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    assignment: Dict[str, str] = {}
    protected_sources = set()
    tasks_by_rarity = sorted(task_sources, key=lambda task: (len(task_sources[task]), task))

    # Reserve at least one source per task for training before allocating evaluation groups.
    if train_ratio > 0:
        for task in tasks_by_rarity:
            if any(assignment.get(source) == "train" for source in task_sources[task]):
                continue
            candidates = [source for source in task_sources[task] if source not in assignment]
            if candidates:
                anchor = min(candidates, key=lambda source: stable_key("train-anchor", task, source))
                assignment[anchor] = "train"
                protected_sources.add(anchor)

    eval_splits = [name for name, ratio in zip(split_names[1:], ratios[1:]) if ratio > 0]
    effective_minimums: Dict[str, int] = {}
    for task in tasks_by_rarity:
        non_train_sources = sum(assignment.get(source) != "train" for source in task_sources[task])
        effective_minimums[task] = min(
            task_minimums.get(task, min_task_sources_per_eval_split),
            non_train_sources // len(eval_splits) if eval_splits else 0,
        )

    # Allocate rare tasks first. A source is global: it can never cross splits even if it has several tasks.
    for split_index, split in enumerate(eval_splits):
        future_splits = len(eval_splits) - split_index - 1
        for task in tasks_by_rarity:
            required = effective_minimums[task]
            current = sum(
                assignment.get(source) == split
                for source in task_sources[task]
            )
            while current < required:
                candidates = [source for source in task_sources[task] if source not in assignment]
                reserve = future_splits * required
                if len(candidates) <= reserve:
                    break
                chosen = min(candidates, key=lambda source: stable_key("required", split, task, source))
                assignment[chosen] = split
                protected_sources.add(chosen)
                current += 1

    for source in sorted(groups):
        if source in assignment:
            continue
        digest = hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()
        point = int.from_bytes(digest[:8], "big") / float(2**64)
        if point < train_ratio:
            split = "train"
        elif point < train_ratio + validation_ratio:
            split = "validation"
        else:
            split = "test"
        assignment[source] = split

    assigned_sources: Dict[str, List[str]] = {
        split: [source for source, assigned_split in assignment.items() if assigned_split == split]
        for split in split_names
    }

    requested = [name for name, ratio in zip(split_names, ratios) if ratio > 0]
    if len(groups) >= len(requested):
        for empty_split in [name for name in requested if not assigned_sources[name]]:
            donors = [
                name for name in requested
                if any(source not in protected_sources for source in assigned_sources[name])
            ]
            if not donors:
                break
            donor = max(donors, key=lambda name: (len(assigned_sources[name]), ratios[split_names.index(name)]))
            moved_source = min(
                [source for source in assigned_sources[donor] if source not in protected_sources],
                key=lambda source: (len(groups[source]), hashlib.sha256(source.encode("utf-8")).hexdigest()),
            )
            assigned_sources[donor].remove(moved_source)
            assigned_sources[empty_split].append(moved_source)

    for split, sources in assigned_sources.items():
        for source in sources:
            result[split].extend(groups[source])
        random.Random(seed + split_names.index(split)).shuffle(result[split])

    for task, required in effective_minimums.items():
        for split in eval_splits:
            actual = len({str(item.get("source", "")) for item in result[split] if item.get("task") == task})
            if actual < required:
                raise AssertionError(f"内部错误：{split} 的 {task} 仅有 {actual} 个来源，要求至少 {required} 个")
    return result


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(
    output_dir: Path,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    manifest: MutableMapping[str, object],
    overwrite: bool,
) -> Dict[str, Path]:
    paths = {
        split: output_dir / split / f"gujinbridge_{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in [*paths.values(), manifest_path] if path.exists()]
    if existing and not overwrite:
        joined = "、".join(str(path) for path in existing)
        raise FileExistsError(f"输出已存在；如需覆盖请添加 --overwrite：{joined}")

    for split, path in paths.items():
        _write_jsonl(path, splits[split])
    outputs = {}
    for split, path in paths.items():
        task_counts = Counter(str(item.get("task", "unknown")) for item in splits[split])
        task_sources: Dict[str, set] = defaultdict(set)
        for item in splits[split]:
            task_sources[str(item.get("task", "unknown"))].add(str(item.get("source", "")))
        outputs[split] = {
            "path": str(path),
            "records": len(splits[split]),
            "sha256": _sha256(path),
            "sources": len({str(item.get("source", "")) for item in splits[split]}),
            "tasks": dict(sorted(task_counts.items())),
            "task_sources": {task: len(sources) for task, sources in sorted(task_sources.items())},
        }
    manifest["outputs"] = outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="清洗并转换 GujinBridge 古籍指令数据")
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="一个或多个 JSONL 文件/目录")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task-limits",
        type=parse_task_limits,
        default=DEFAULT_LIMITS,
        help="每类任务的抽样上限，-1 表示不限；默认 c2m=60000,m2c=40000,punctuate=20000",
    )
    parser.add_argument("--system-prompt-file", type=Path, default=Path("configs/gujinbridge_system_prompt.txt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument(
        "--min-task-sources-per-eval-split",
        type=int,
        default=3,
        help="验证集和测试集为每种任务保留的最少来源数；来源不足时自动降低",
    )
    parser.add_argument(
        "--task-min-sources-per-eval-split",
        type=parse_task_limits,
        default={},
        help="按任务覆盖全局最少来源数，例如 punctuate=8",
    )
    parser.add_argument("--min-input-chars", type=int, default=4)
    parser.add_argument("--max-input-chars", type=int, default=500, help="0 表示不限制")
    parser.add_argument("--keep-box", action="store_true", help="保留上游标记为 _has_box 的异常字符样本")
    parser.add_argument(
        "--allow-low-quality-punctuation",
        action="store_true",
        help="保留正文不一致、标点密度过低或切片边界异常的断句样本",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_input_chars < 0 or args.max_input_chars < 0:
        raise ValueError("字符长度限制不能为负数")
    if args.max_input_chars and args.max_input_chars < args.min_input_chars:
        raise ValueError("--max-input-chars 不能小于 --min-input-chars")
    if any(value < 0 for value in args.task_min_sources_per_eval_split.values()):
        raise ValueError("--task-min-sources-per-eval-split 不能包含负数")

    prompt_path = args.system_prompt_file
    system_prompt = prompt_path.read_text(encoding="utf-8-sig").strip() if prompt_path.exists() else DEFAULT_SYSTEM_PROMPT
    if not system_prompt:
        raise ValueError("系统提示词不能为空")

    files = discover_jsonl_files(args.input, args.output_dir)
    stats: Counter[str] = Counter()
    sampled = reservoir_sample(
        iter_jsonl(files, stats),
        limits=args.task_limits,
        system_prompt=system_prompt,
        seed=args.seed,
        min_input_chars=args.min_input_chars,
        max_input_chars=args.max_input_chars,
        keep_box=args.keep_box,
        stats=stats,
        filter_punctuation_quality=not args.allow_low_quality_punctuation,
    )
    if not sampled:
        raise RuntimeError("清洗后没有可用样本，请检查输入字段、任务名和过滤参数")
    splits = split_by_source(
        sampled,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        min_task_sources_per_eval_split=args.min_task_sources_per_eval_split,
        task_min_sources_per_eval_split=args.task_min_sources_per_eval_split,
    )
    source_sets = [{str(item["source"]) for item in splits[name]} for name in ("train", "validation", "test")]
    if source_sets[0] & source_sets[1] or source_sets[0] & source_sets[2] or source_sets[1] & source_sets[2]:
        raise AssertionError("内部错误：数据切分出现 source 泄漏")

    manifest: Dict[str, object] = {
        "project": "GujinBridge",
        "upstream": UPSTREAM_NAME,
        "input_files": [str(path) for path in files],
        "seed": args.seed,
        "task_limits": args.task_limits,
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "min_task_sources_per_eval_split": args.min_task_sources_per_eval_split,
        "task_min_sources_per_eval_split": args.task_min_sources_per_eval_split,
        "filters": {
            "min_input_chars": args.min_input_chars,
            "max_input_chars": args.max_input_chars,
            "keep_box": args.keep_box,
            "filter_punctuation_quality": not args.allow_low_quality_punctuation,
        },
        "stats": dict(sorted(stats.items())),
    }
    paths = write_outputs(args.output_dir, splits, manifest, overwrite=args.overwrite)
    print(f"已处理 {stats['raw_lines']} 行，抽样 {len(sampled)} 条。")
    for split in ("train", "validation", "test"):
        print(f"{split}: {len(splits[split])} 条 -> {paths[split]}")
    print(f"清单：{args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
