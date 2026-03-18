#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from nanochat_mlx.config import NanoChatConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a nanochat checkpoint to MLX weights")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory that contains model_*.pt and meta_*.json")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load. Defaults to the largest step in checkpoint-dir")
    parser.add_argument("--model-file", type=Path, help="Explicit path to model_XXXXXX.pt")
    parser.add_argument("--meta-file", type=Path, help="Explicit path to meta_XXXXXX.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write config.json and weights.safetensors")
    parser.add_argument("--tokenizer-dir", type=Path, help="Optional tokenizer directory to copy into the output dir")
    return parser


def patch_model_config(model_config: dict) -> dict:
    patched = dict(model_config)
    patched.setdefault("window_pattern", "L")
    return patched


def find_last_step(checkpoint_dir: Path) -> int:
    model_files = sorted(checkpoint_dir.glob("model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files found in {checkpoint_dir}")
    return max(int(path.stem.split("_")[-1]) for path in model_files)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.model_file and args.meta_file:
        return args.model_file, args.meta_file
    if not args.checkpoint_dir:
        raise ValueError("Provide either --model-file/--meta-file or --checkpoint-dir")
    step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
    return (
        args.checkpoint_dir / f"model_{step:06d}.pt",
        args.checkpoint_dir / f"meta_{step:06d}.json",
    )


def tensor_to_numpy(tensor, torch_module) -> np.ndarray:
    if tensor.dtype == torch_module.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def map_weight_name(name: str) -> str:
    if name.startswith("transformer.wte."):
        return name.replace("transformer.wte.", "wte.", 1)
    if name.startswith("transformer.h."):
        return name.replace("transformer.h.", "layers.", 1)
    if name.startswith("value_embeds."):
        parts = name.split(".")
        parts[1] = f"layer_{parts[1]}"
        return ".".join(parts)
    return name


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for conversion. Install with: pip install '.[convert]'") from exc

    globals()["torch"] = torch
    model_path, meta_path = resolve_paths(args)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    model_data = torch.load(model_path, map_location="cpu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_config = patch_model_config(meta["model_config"])
    model_data = {name.removeprefix("_orig_mod."): tensor for name, tensor in model_data.items()}
    model_data.setdefault("resid_lambdas", torch.ones(model_config["n_layer"]))
    model_data.setdefault("x0_lambdas", torch.zeros(model_config["n_layer"]))

    weights: dict[str, np.ndarray] = {}
    for name, tensor in model_data.items():
        if name in {"cos", "sin"}:
            continue
        weights[map_weight_name(name)] = tensor_to_numpy(tensor, torch)

    padded_vocab_size = int(weights["lm_head.weight"].shape[0])
    config = NanoChatConfig.from_dict({**model_config, "padded_vocab_size": padded_vocab_size})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(args.output_dir / "config.json")
    save_file(weights, str(args.output_dir / "weights.safetensors"))
    (args.output_dir / "source_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.tokenizer_dir:
        target = args.output_dir / "tokenizer"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(args.tokenizer_dir, target)

    print(f"Converted {len(weights)} tensors from {model_path.name} into {args.output_dir}")


if __name__ == "__main__":
    main()
