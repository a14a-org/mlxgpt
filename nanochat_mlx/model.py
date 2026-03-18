from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import NanoChatConfig


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: mx.array, offset: int, base: float) -> mx.array:
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    channels = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv_freq = 1.0 / (base ** (channels / head_dim))
    positions = mx.arange(offset, offset + x.shape[2], dtype=mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs)[None, None, :, :].astype(x.dtype)
    sin = mx.sin(freqs)[None, None, :, :].astype(x.dtype)
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    y1 = x1 * cos + x2 * sin
    y2 = x2 * cos - x1 * sin
    return mx.concatenate([y1, y2], axis=-1)


def build_attention_mask(t_q: int, t_k: int, offset: int, window: int) -> str | mx.array:
    if window < 0 or window >= t_k:
        return "causal"
    query_positions = offset + mx.arange(t_q, dtype=mx.int32)[:, None]
    key_positions = mx.arange(t_k, dtype=mx.int32)[None, :]
    causal = key_positions <= query_positions
    in_window = (query_positions - key_positions) <= window
    return mx.expand_dims(mx.expand_dims(mx.logical_and(causal, in_window), axis=0), axis=0)


def sample_token(logits: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int | None) -> int:
    if temperature <= 0:
        return int(logits.argmax())

    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        indices = np.argpartition(logits, -top_k)[-top_k:]
        candidate_logits = logits[indices] / temperature
        candidate_logits = candidate_logits - candidate_logits.max()
        probs = np.exp(candidate_logits)
        probs = probs / probs.sum()
        choice = int(rng.choice(indices, p=probs))
        return choice

    scaled = logits / temperature
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


class KVCache:
    def __init__(self, n_layers: int):
        self.layers: list[dict[str, mx.array | None]] = [{"k": None, "v": None} for _ in range(n_layers)]

    def get(self, layer_idx: int) -> tuple[mx.array | None, mx.array | None]:
        layer = self.layers[layer_idx]
        return layer["k"], layer["v"]

    def update(self, layer_idx: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array, int]:
        past_k, past_v = self.get(layer_idx)
        offset = 0 if past_k is None else int(past_k.shape[2])
        if past_k is None:
            full_k = k
            full_v = v
        else:
            full_k = mx.concatenate([past_k, k], axis=2)
            full_v = mx.concatenate([past_v, v], axis=2)
        self.layers[layer_idx]["k"] = full_k
        self.layers[layer_idx]["v"] = full_v
        return full_k, full_v, offset

    def reset(self) -> None:
        for layer in self.layers:
            layer["k"] = None
            layer["v"] = None


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))
        return self.c_proj(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NanoChatConfig, layer_idx: int, window_size: tuple[int, int]):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.window_size = window_size
        self.ve_gate_channels = config.ve_gate_channels
        self.rope_base = config.rope_base
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def __call__(self, x: mx.array, ve: mx.array | None, cache: KVCache | None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            gate = 3.0 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            gate = gate.transpose(0, 2, 1)[..., None]
            v = v + gate * ve

        offset = 0
        if cache is not None:
            past_k, past_v = cache.get(self.layer_idx)
            offset = 0 if past_k is None else int(past_k.shape[2])
        else:
            past_k, past_v = None, None

        q = apply_rotary_emb(q, offset=offset, base=self.rope_base)
        k = apply_rotary_emb(k, offset=offset, base=self.rope_base)
        q = rms_norm(q) * 1.15
        k = rms_norm(k) * 1.15
        if past_k is None:
            full_k = k
            full_v = v
        else:
            full_k = mx.concatenate([past_k, k], axis=2)
            full_v = mx.concatenate([past_v, v], axis=2)
        if cache is not None:
            cache.layers[self.layer_idx]["k"] = full_k
            cache.layers[self.layer_idx]["v"] = full_v

        mask = build_attention_mask(seq_len, int(full_k.shape[2]), offset, self.window_size[0])
        y = mx.fast.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            scale=self.head_dim ** -0.5,
            mask=mask,
        )
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_embd)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, config: NanoChatConfig, layer_idx: int, window_size: tuple[int, int]):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, window_size)
        self.mlp = MLP(config.n_embd)

    def __call__(self, x: mx.array, ve: mx.array | None, cache: KVCache | None) -> mx.array:
        x = x + self.attn(rms_norm(x), ve, cache)
        x = x + self.mlp(rms_norm(x))
        return x


class NanoChatMLX(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.effective_padded_vocab_size, config.n_embd)
        self.layers = [Block(config, layer_idx, self.window_sizes[layer_idx]) for layer_idx in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.effective_padded_vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((config.n_layer,), 0.1, dtype=mx.float32)
        self.value_embeds = {
            f"layer_{i}": nn.Embedding(config.effective_padded_vocab_size, config.n_kv_head * config.head_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }

    def _compute_window_sizes(self, config: NanoChatConfig) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()
        if any(ch not in "SL" for ch in pattern):
            raise ValueError(f"Invalid window_pattern={config.window_pattern!r}")
        long_window = config.sequence_len
        short_window = -(-long_window // 3 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[layer_idx % len(pattern)]] for layer_idx in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def __call__(self, idx: mx.array, cache: KVCache | None = None) -> mx.array:
        if idx.ndim == 1:
            idx = idx[None, :]
        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x
        for i, block in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = f"layer_{i}"
            ve = self.value_embeds[ve_key](idx).astype(x.dtype) if ve_key in self.value_embeds else None
            x = block(x, ve, cache)
        x = rms_norm(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size]
        logits = logits.astype(mx.float32)
        return self.config.softcap * mx.tanh(logits / self.config.softcap)

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        seed: int = 42,
        stop_tokens: set[int] | None = None,
    ) -> list[int]:
        stop_tokens = stop_tokens or set()
        rng = np.random.default_rng(seed)
        cache = KVCache(self.config.n_layer)
        logits = self(mx.array([prompt_tokens], dtype=mx.int32), cache=cache)
        logits_np = np.asarray(logits[0, -1])
        generated: list[int] = []
        for _ in range(max_tokens):
            token = sample_token(logits_np, rng, temperature=temperature, top_k=top_k)
            generated.append(token)
            if token in stop_tokens:
                break
            logits = self(mx.array([[token]], dtype=mx.int32), cache=cache)
            logits_np = np.asarray(logits[0, -1])
        return generated

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
