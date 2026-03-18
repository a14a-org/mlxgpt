#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nanochat_mlx.tokenizer import load_tokenizer
from nanochat_mlx.training import (
    ClusterTrainer,
    ParallelConfig,
    TrainConfig,
    build_model_config,
    create_model,
    create_optimizer,
    maybe_resume,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLX-native single-node or distributed cluster trainer for nanochat-compatible models")
    parser.add_argument("--parallelism", choices=["dp", "tp"], required=True, help="Distributed training mode")
    parser.add_argument("--backend", default="any", help="Backend passed to mlx.distributed.init()")
    parser.add_argument("--model-tag", required=True, help="Checkpoint/export tag")
    parser.add_argument("--depth", type=int, required=True, help="Number of transformer blocks")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Sequence length used for packed LM training")
    parser.add_argument("--device-batch-size", type=int, required=True, help="Per-rank microbatch size")
    parser.add_argument("--total-batch-size", type=int, required=True, help="Target global batch size in tokens")
    parser.add_argument("--num-iterations", type=int, required=True, help="Number of optimizer steps to run")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save a training checkpoint every N optimizer steps")
    parser.add_argument("--resume", nargs="?", const="latest", default="", help="Resume from latest or from an explicit checkpoint dir/latest.json")
    parser.add_argument("--export-final", action="store_true", help="Export an inference-ready MLX model into converted/<tag> at the end")
    parser.add_argument("--base-dir", default=".nanochat-cache", help="Base dir that holds tokenizer and dataset artifacts")
    parser.add_argument("--tokenizer-dir", default="", help="Override tokenizer directory")
    parser.add_argument("--fallback-tokenizer", default="gpt2", help="Fallback tokenizer name for synthetic/local smoke runs")
    parser.add_argument("--dataset-dir", default="", help="Override parquet dataset directory")
    parser.add_argument("--checkpoint-root", default=".mlx-checkpoints", help="Root directory for native MLX training checkpoints")
    parser.add_argument("--export-root", default="converted", help="Root directory for inference exports")
    parser.add_argument("--run-dir", default="", help="Directory for rank metrics and run summaries")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="Model width heuristic: n_embd ~= depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--window-pattern", default="L", help="Window pattern copied into NanoChatConfig")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="MLX RNG seed")
    parser.add_argument("--data-mode", choices=["real", "synthetic"], default="real", help="Real parquet training data or a synthetic smoke dataset")
    parser.add_argument("--synthetic-documents", type=int, default=2048, help="Synthetic documents available when --data-mode synthetic")
    parser.add_argument("--tokenizer-threads", type=int, default=4, help="Threads used for batched tokenizer encoding")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128, help="Documents per tokenizer encode batch")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Best-fit document buffer size")
    parser.add_argument("--log-every", type=int, default=10, help="Rank-0 logging cadence in optimizer steps")
    parser.add_argument("--verify-every", type=int, default=50, help="Parameter checksum verification cadence")
    parser.add_argument("--eval-every", type=int, default=250, help="Run validation every N optimizer steps; <=0 disables validation")
    parser.add_argument("--eval-batches", type=int, default=8, help="Number of validation batches to average per validation event")
    parser.add_argument("--sample-every", type=int, default=500, help="Generate a sample on rank 0 every N optimizer steps; <=0 disables samples")
    parser.add_argument("--sample-prompt", default="The capital of France is", help="Prompt used for deterministic in-training sample generation")
    parser.add_argument("--sample-max-tokens", type=int, default=32, help="Maximum number of tokens to generate for sample events")
    parser.add_argument("--sample-temperature", type=float, default=0.0, help="Sampling temperature for in-training sample generation")
    parser.add_argument(
        "--sample-prepend-bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix the sample prompt with the tokenizer BOS token",
    )
    parser.add_argument("--early-stop-min-step", type=int, default=0, help="Do not consider early stopping before this step")
    parser.add_argument("--early-stop-patience-evals", type=int, default=0, help="Number of non-improving eval events before early-stop can trigger")
    parser.add_argument("--early-stop-degrade-ratio", type=float, default=1.0, help="Stop when val loss exceeds best val by this ratio after patience")
    parser.add_argument("--early-stop-vs-champion-ratio", type=float, default=1.0, help="Stop when val loss is clearly worse than the seeded champion by this ratio")
    parser.add_argument("--champion-best-val", type=float, default=None, help="Seeded champion best validation loss used for non-competitive early stop")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir)
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else base_dir / "tokenizer"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else base_dir / "base_data_climbmix"
    checkpoint_root = Path(args.checkpoint_root)
    export_root = Path(args.export_root)
    run_dir = Path(args.run_dir) if args.run_dir else Path("build") / "mlx-train" / args.model_tag

    world = mx.distributed.init(backend=args.backend)
    parallel_config = ParallelConfig.from_world(
        mode=args.parallelism,
        backend=args.backend,
        rank=world.rank(),
        world_size=world.size(),
    )

    mx.random.seed(args.seed)
    tokenizer = load_tokenizer(tokenizer_dir if tokenizer_dir.exists() else None, fallback=args.fallback_tokenizer)
    train_config = TrainConfig(
        model_tag=args.model_tag,
        depth=args.depth,
        max_seq_len=args.max_seq_len,
        device_batch_size=args.device_batch_size,
        total_batch_size=args.total_batch_size,
        num_iterations=args.num_iterations,
        checkpoint_every=args.checkpoint_every,
        base_dir=str(base_dir),
        checkpoint_root=str(checkpoint_root),
        export_root=str(export_root),
        tokenizer_dir=str(tokenizer_dir),
        dataset_dir=str(dataset_dir),
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        window_pattern=args.window_pattern,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_mode=args.data_mode,
        tokenizer_threads=args.tokenizer_threads,
        tokenizer_batch_size=args.tokenizer_batch_size,
        buffer_size=args.buffer_size,
        synthetic_documents=args.synthetic_documents,
        log_every=args.log_every,
        verify_every=args.verify_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        sample_max_tokens=args.sample_max_tokens,
        sample_temperature=args.sample_temperature,
        sample_prepend_bos=args.sample_prepend_bos,
        early_stop_min_step=args.early_stop_min_step,
        early_stop_patience_evals=args.early_stop_patience_evals,
        early_stop_degrade_ratio=args.early_stop_degrade_ratio,
        early_stop_vs_champion_ratio=args.early_stop_vs_champion_ratio,
        champion_best_val=args.champion_best_val,
        run_dir=str(run_dir),
        export_final=args.export_final,
    )

    model_config = build_model_config(train_config, tokenizer.get_vocab_size(), parallel_config)
    model = create_model(model_config, parallel_config)
    optimizer = create_optimizer(train_config)

    resume = args.resume
    if resume == "latest":
        resume = str(checkpoint_root / args.model_tag / "latest.json")
    checkpoint_state = maybe_resume(train_config, parallel_config, model, optimizer, resume or None)

    trainer = ClusterTrainer(
        train_config=train_config,
        parallel_config=parallel_config,
        model_config=model_config,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        checkpoint_state=checkpoint_state,
    )

    if parallel_config.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "train_config": train_config.to_dict(),
                    "parallel_config": parallel_config.to_dict(),
                    "model_config": model_config.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    result = trainer.train()
    print(json.dumps({"event": "train_complete", "rank": parallel_config.rank, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
