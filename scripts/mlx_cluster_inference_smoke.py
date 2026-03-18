#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import mlx.core as mx
import numpy as np

from nanochat_mlx.io import load_model
from nanochat_mlx.tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal distributed inference smoke test for converted nanochat MLX models")
    parser.add_argument("--backend", default="any", choices=["any", "ring", "jaccl", "mpi", "nccl"], help="Distributed backend passed to mlx.distributed.init()")
    parser.add_argument("--model-dir", required=True, help="Directory with config.json and weights.safetensors")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory with tokenizer.pkl or tokenizer.json")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt used for smoke generation")
    parser.add_argument("--max-tokens", type=int, default=16, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Keep 0 for deterministic smoke tests")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff when temperature > 0")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--prepend-bos", action="store_true", help="Prefix the prompt with the tokenizer BOS token")
    parser.add_argument("--output-dir", default="", help="Optional directory to write one JSON result per rank")
    return parser


def checksum(tokens: list[int]) -> int:
    total = 0
    for idx, token in enumerate(tokens, start=1):
        total = (total + idx * int(token)) % 2_147_483_647
    return total


def main() -> None:
    args = build_parser().parse_args()
    world = mx.distributed.init(backend=args.backend)
    rank = world.rank()
    size = world.size()
    host = socket.gethostname()

    model = load_model(args.model_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    prompt_tokens = tokenizer.encode(args.prompt)
    if args.prepend_bos:
        prompt_tokens = [tokenizer.get_bos_token_id()] + prompt_tokens
    if not prompt_tokens:
        raise SystemExit("Prompt produced zero tokens")
    if max(prompt_tokens) >= model.config.vocab_size:
        raise SystemExit("Prompt token ids exceed model vocabulary; use the tokenizer that matches the converted checkpoint")

    generated_tokens = model.generate(
        prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    generated_text = tokenizer.decode(prompt_tokens + generated_tokens)
    token_checksum = checksum(generated_tokens)

    checksum_tensor = mx.array([token_checksum], dtype=mx.int64)
    rank_tensor = mx.array([rank], dtype=mx.int32)
    all_checksums = mx.distributed.all_gather(checksum_tensor)
    all_ranks = mx.distributed.all_gather(rank_tensor)
    mx.eval(all_checksums, all_ranks)

    gathered_checksums = [int(x) for x in np.asarray(all_checksums).reshape(-1)]
    gathered_ranks = [int(x) for x in np.asarray(all_ranks).reshape(-1)]

    result = {
        "backend": args.backend,
        "host": host,
        "rank": int(rank),
        "size": int(size),
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "checksum": token_checksum,
        "all_checksums": gathered_checksums,
        "all_ranks": gathered_ranks,
    }

    print(json.dumps(result, sort_keys=True))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.joinpath(f"rank_{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if sorted(gathered_ranks) != list(range(size)):
        raise SystemExit(f"all_gather ranks mismatch on rank {rank}: got {gathered_ranks}")
    if len(set(gathered_checksums)) != 1:
        raise SystemExit(f"Generated token checksums diverged across ranks: {gathered_checksums}")


if __name__ == "__main__":
    main()
