#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import mlx.core as mx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MLX distributed smoke test")
    parser.add_argument("--backend", default="ring", choices=["any", "ring", "jaccl", "mpi", "nccl"], help="Distributed backend")
    parser.add_argument("--length", type=int, default=8, help="Length of the test vector")
    parser.add_argument("--output-dir", default="", help="Optional directory to write one JSON result per rank")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world = mx.distributed.init(backend=args.backend)
    rank = world.rank()
    size = world.size()
    host = socket.gethostname()

    x = mx.ones((args.length,), dtype=mx.float32) * (rank + 1)
    summed = mx.distributed.all_sum(x)
    gathered = mx.distributed.all_gather(mx.array([rank], dtype=mx.int32))
    mx.eval(summed, gathered)

    result = {
        "backend": args.backend,
        "host": host,
        "rank": int(rank),
        "size": int(size),
        "input": x.tolist(),
        "all_sum": summed.tolist(),
        "all_gather": gathered.tolist(),
        "expected_sum_value": int(size * (size + 1) / 2),
    }

    print(json.dumps(result, sort_keys=True))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.joinpath(f"rank_{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    expected = result["expected_sum_value"]
    if any(value != expected for value in result["all_sum"]):
        raise SystemExit(f"all_sum mismatch on rank {rank}: expected {expected}, got {result['all_sum']}")
    if sorted(result["all_gather"]) != list(range(size)):
        raise SystemExit(f"all_gather mismatch on rank {rank}: got {result['all_gather']}")


if __name__ == "__main__":
    main()
