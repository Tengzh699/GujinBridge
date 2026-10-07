#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a derived GujinBridge split after removing deterministic audit failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Set, Tuple


SPLITS = ("train", "validation", "test")
DEFAULT_REMOVABLE_FLAGS = {
    "missing_pair",
    "output_too_short",
    "low_cjk_output",
    "identical_input_output",
    "extreme_length_ratio",
}


def parse_flags(value: str) -> Set[str]:
    flags = {item.strip() for item in value.split(",") if item.strip()}
    if not flags:
        raise argparse.ArgumentTypeError("至少指定一个审计标记")
    return flags


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            yield record


def read_flag_map(path: Path) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    for record in iter_jsonl(path):
        record_id = str(record.get("id", "")).strip()
        flags = record.get("audit_flags", [])
        if not record_id or not isinstance(flags, list):
            raise ValueError(f"审计记录字段不完整：{path}")
        if record_id in result:
            raise ValueError(f"审计记录 ID 重复：{record_id}")
        result[record_id] = {str(flag) for flag in flags}
    return result


def read_protected_ids(path: Path | None) -> Set[str]:
    if path is None:
        return set()
    protected_ids = {str(record.get("id", "")).strip() for record in iter_jsonl(path)}
    protected_ids.discard("")
    if not protected_ids:
        raise ValueError(f"保护文件中未找到有效 ID：{path}")
    return protected_ids


def filter_records(
    records: Iterable[Mapping[str, object]],
    flag_map: Mapping[str, Set[str]],
    removable_flags: Set[str],
    protected_ids: Set[str] | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Counter[str]]:
    protected = protected_ids or set()
    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    removed_flag_counts: Counter[str] = Counter()

    for record in records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            raise ValueError("输入数据含有空 ID")
        matching_flags = flag_map.get(record_id, set()) & removable_flags
        if not matching_flags:
            kept.append(dict(record))
            continue
        if record_id in protected:
            raise ValueError(f"保护的评测记录将被删除：{record_id}")
        removed.append(dict(record))
        removed_flag_counts.update(matching_flags)
    return kept, removed, removed_flag_counts


def split_path(input_dir: Path, split: str) -> Path:
    return input_dir / split / f"gujinbridge_{split}.jsonl"


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def task_counts(records: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(record.get("task", "unknown")) for record in records).items()))


def verify_source_isolation(splits: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    source_sets = {
        split: {str(record.get("source", "")) for record in records}
        for split, records in splits.items()
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = source_sets[left] & source_sets[right]
            if overlap:
                preview = "、".join(sorted(overlap)[:5])
                raise ValueError(f"来源泄漏：{left}/{right} 共享 {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="依据审计标记过滤 GujinBridge 已切分数据")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--audit-flags", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protected-ids", type=Path, help="必须保留的 JSONL 记录 ID，例如黄金集候选")
    parser.add_argument(
        "--remove-flags",
        type=parse_flags,
        default=DEFAULT_REMOVABLE_FLAGS,
        help="逗号分隔的确定性坏样本标记",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{args.input_dir}")
    if not args.audit_flags.is_file():
        raise FileNotFoundError(f"审计记录不存在：{args.audit_flags}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空；如需覆盖请添加 --overwrite：{args.output_dir}")

    input_paths = {split: split_path(args.input_dir, split) for split in SPLITS}
    missing = [path for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少切分文件：" + "、".join(map(str, missing)))

    flag_map = read_flag_map(args.audit_flags)
    protected_ids = read_protected_ids(args.protected_ids)
    filtered_splits: Dict[str, List[Dict[str, object]]] = {}
    removal_summary: MutableMapping[str, Dict[str, object]] = {}
    removed_ids: Set[str] = set()

    for split, path in input_paths.items():
        original = list(iter_jsonl(path))
        kept, removed, removed_flags = filter_records(
            original,
            flag_map,
            args.remove_flags,
            protected_ids,
        )
        filtered_splits[split] = kept
        removed_ids.update(str(record["id"]) for record in removed)
        removal_summary[split] = {
            "before": len(original),
            "removed": len(removed),
            "after": len(kept),
            "removed_tasks": task_counts(removed),
            "remaining_tasks": task_counts(kept),
            "removed_flags": dict(sorted(removed_flags.items())),
        }

    verify_source_isolation(filtered_splits)
    if removed_ids & protected_ids:
        raise AssertionError("内部错误：删除了受保护的评测记录")

    for split, records in filtered_splits.items():
        write_jsonl(split_path(args.output_dir, split), records)

    source_manifest_path = args.input_dir / "manifest.json"
    source_manifest: Dict[str, object] = {}
    if source_manifest_path.is_file():
        value = json.loads(source_manifest_path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            source_manifest = value
    manifest = {
        "project": "GujinBridge",
        "derived_from": str(args.input_dir),
        "source_manifest": str(source_manifest_path) if source_manifest else None,
        "audit_flags": str(args.audit_flags),
        "protected_ids": str(args.protected_ids) if args.protected_ids else None,
        "removal_policy": {
            "remove_flags": sorted(args.remove_flags),
            "note": "Only deterministic audit failures are removed; ambiguous flags remain for review.",
        },
        "source_split_policy": source_manifest.get("split_ratios"),
        "source_task_minimums": source_manifest.get("task_min_sources_per_eval_split"),
        "splits": removal_summary,
        "protected_records": len(protected_ids),
        "removed_records": len(removed_ids),
        "remaining_records": sum(len(records) for records in filtered_splits.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
