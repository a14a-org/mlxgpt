#!/usr/bin/env python3
from __future__ import annotations

import argparse

from nanochat_mlx.io import load_model
from nanochat_mlx.tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with a converted nanochat MLX model")
    parser.add_argument("--model-dir", required=True, help="Directory with config.json and weights.safetensors")
    parser.add_argument("--tokenizer-dir", help="Directory with tokenizer.pkl or tokenizer.json")
    parser.add_argument("--fallback-tokenizer", default="gpt2", help="Fallback tiktoken name if tokenizer-dir is omitted")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--prepend-bos", action="store_true", help="Prefix the prompt with the tokenizer BOS token")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.model_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir, fallback=args.fallback_tokenizer)

    prompt_tokens = tokenizer.encode(args.prompt)
    if args.prepend_bos:
        prompt_tokens = [tokenizer.get_bos_token_id()] + prompt_tokens
    if not prompt_tokens:
        raise SystemExit("Prompt produced zero tokens")
    if max(prompt_tokens) >= model.config.vocab_size:
        raise SystemExit(
            "Prompt token ids exceed the model vocabulary. Use the tokenizer that was trained with nanochat."
        )

    generated = model.generate(
        prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(tokenizer.decode(prompt_tokens + generated))


if __name__ == "__main__":
    main()
