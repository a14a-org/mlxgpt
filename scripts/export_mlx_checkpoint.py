#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nanochat_mlx.checkpoint import resolve_resume_checkpoint
from nanochat_mlx.config import NanoChatConfig
from nanochat_mlx.model import NanoChatMLX


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DP MLX checkpoint into inference-ready config/weights files")
    parser.add_argument("--model-tag", required=True, help="Training checkpoint tag to export")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint dir or latest.json path; 'latest' resolves to the newest checkpoint for the tag",
    )
    parser.add_argument("--checkpoint-root", default=".mlx-checkpoints", help="Root directory for native MLX checkpoints")
    parser.add_argument("--export-root", default="converted", help="Root directory for exported inference-ready models")
    parser.add_argument("--tokenizer-dir", default="", help="Optional tokenizer directory override")
    return parser


def copy_tokenizer(tokenizer_src: Path | None, export_dir: Path) -> None:
    if tokenizer_src is None or not tokenizer_src.exists():
        return
    tokenizer_dst = export_dir / "tokenizer"
    if tokenizer_dst.exists():
        shutil.rmtree(tokenizer_dst)
    shutil.copytree(tokenizer_src, tokenizer_dst)


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_ref = None if args.checkpoint == "latest" else args.checkpoint
    checkpoint_dir = resolve_resume_checkpoint(args.checkpoint_root, args.model_tag, checkpoint_ref)

    model_config = NanoChatConfig.from_json(checkpoint_dir / "model_config.json")
    model = NanoChatMLX(model_config)
    model.load_weights(str(checkpoint_dir / "weights_rank0.safetensors"))

    export_dir = Path(args.export_root) / args.model_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    model_config.to_json(export_dir / "config.json")
    model.save_weights(str(export_dir / "weights.safetensors"))

    tokenizer_src = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    if tokenizer_src is None:
        train_config_path = checkpoint_dir / "train_config.json"
        if train_config_path.exists():
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
            tokenizer_value = train_config.get("tokenizer_dir")
            if tokenizer_value:
                tokenizer_src = Path(tokenizer_value)
    copy_tokenizer(tokenizer_src, export_dir)

    print(
        json.dumps(
            {
                "event": "export_complete",
                "model_tag": args.model_tag,
                "checkpoint_dir": str(checkpoint_dir),
                "export_dir": str(export_dir),
                "weights_size": (export_dir / "weights.safetensors").stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
