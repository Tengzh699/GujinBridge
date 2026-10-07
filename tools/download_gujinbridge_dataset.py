#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the GujinBridge upstream dataset from Hugging Face Hub."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 GujinBridge 推荐古籍语料")
    parser.add_argument("--output-dir", type=Path, default=Path("data/gujinbridge/raw"))
    parser.add_argument("--repo-id", default="gujilab/chinese-classical-corpus")
    parser.add_argument("--revision", default=None, help="可选：固定分支、标签或 commit")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("缺少 huggingface_hub，请先执行：pip install huggingface_hub") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.output_dir),
        allow_patterns=["**/*.jsonl", "README*", "**/README*", "LICENSE*", "**/LICENSE*"],
    )
    print(f"数据已下载到：{downloaded}")
    print("下一步运行：python tools/prepare_gujinbridge_dataset.py --input data/gujinbridge/raw --output-dir data/gujinbridge/processed")


if __name__ == "__main__":
    main()
