#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from nanochat_mlx.config import NanoChatConfig
from nanochat_mlx.io import load_model
from nanochat_mlx.model import NanoChatMLX


def main() -> None:
    config = NanoChatConfig(
        sequence_len=32,
        vocab_size=256,
        padded_vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
    )
    model = NanoChatMLX(config)
    ids = mx.array(np.random.default_rng(42).integers(0, config.vocab_size, size=(1, 8), dtype=np.int32))
    logits = model(ids)
    assert logits.shape == (1, 8, config.vocab_size), logits.shape
    generated = model.generate(ids.tolist()[0], max_tokens=8, temperature=0.0, seed=7)
    assert len(generated) == 8, generated

    out_dir = Path("build/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(out_dir / "config.json")
    model.save_weights(str(out_dir / "weights.safetensors"))
    reloaded = load_model(out_dir)
    reloaded_logits = reloaded(ids)
    assert reloaded_logits.shape == logits.shape, reloaded_logits.shape

    print("MLX smoke test passed")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"generated tokens: {generated}")


if __name__ == "__main__":
    main()
