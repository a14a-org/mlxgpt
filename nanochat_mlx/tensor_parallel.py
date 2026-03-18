from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import NanoChatConfig
from .model import apply_rotary_emb, build_attention_mask, has_ve, rms_norm


def _concat_gather(x: mx.array, axis: int, world_size: int) -> mx.array:
    if world_size == 1:
        return x
    ndim = x.ndim
    axis = axis if axis >= 0 else ndim + axis
    permutation = [axis] + [idx for idx in range(ndim) if idx != axis]
    inverse = [permutation.index(idx) for idx in range(ndim)]
    moved = mx.transpose(x, permutation)
    gathered = mx.distributed.all_gather(moved)
    return mx.transpose(gathered, inverse)


def _local_range(total_size: int, world_size: int, rank: int) -> tuple[int, int]:
    if total_size % world_size != 0:
        raise ValueError(f"Cannot shard size {total_size} across world_size={world_size}")
    shard_size = total_size // world_size
    start = rank * shard_size
    return start, start + shard_size


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, rank: int, world_size: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.start, self.end = _local_range(vocab_size, world_size, rank)
        self.embedding = nn.Embedding(self.end - self.start, embedding_dim)

    def __call__(self, idx: mx.array) -> mx.array:
        mask = mx.logical_and(idx >= self.start, idx < self.end)
        local_idx = mx.where(mask, idx - self.start, 0)
        embedded = self.embedding(local_idx)
        embedded = embedded * mask[..., None].astype(embedded.dtype)
        return mx.distributed.all_sum(embedded) if self.world_size > 1 else embedded

    def export_weight(self) -> mx.array:
        return _concat_gather(self.embedding.weight, axis=0, world_size=self.world_size)


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, world_size: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.start, self.end = _local_range(output_dim, world_size, rank)
        self.linear = nn.Linear(input_dim, self.end - self.start, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)

    def export_weight(self) -> mx.array:
        return _concat_gather(self.linear.weight, axis=0, world_size=self.world_size)


class RowParallelLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, world_size: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.start, self.end = _local_range(input_dim, world_size, rank)
        self.linear = nn.Linear(self.end - self.start, output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        projected = self.linear(x)
        return mx.distributed.all_sum(projected) if self.world_size > 1 else projected

    def export_weight(self) -> mx.array:
        return _concat_gather(self.linear.weight, axis=1, world_size=self.world_size)


class VocabParallelLMHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, rank: int, world_size: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.start, self.end = _local_range(vocab_size, world_size, rank)
        self.linear = nn.Linear(input_dim, self.end - self.start, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)

    def export_weight(self) -> mx.array:
        return _concat_gather(self.linear.weight, axis=0, world_size=self.world_size)


class TensorParallelMLP(nn.Module):
    def __init__(self, n_embd: int, rank: int, world_size: int):
        super().__init__()
        self.c_fc = ColumnParallelLinear(n_embd, 4 * n_embd, rank=rank, world_size=world_size)
        self.c_proj = RowParallelLinear(4 * n_embd, n_embd, rank=rank, world_size=world_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))
        return self.c_proj(x)


class TensorParallelCausalSelfAttention(nn.Module):
    def __init__(self, config: NanoChatConfig, layer_idx: int, window_size: tuple[int, int], rank: int, world_size: int):
        super().__init__()
        if config.n_head % world_size != 0 or config.n_kv_head % world_size != 0 or config.n_embd % world_size != 0:
            raise ValueError(
                "Tensor parallel mode requires n_head, n_kv_head, and n_embd to be divisible by world_size"
            )
        self.layer_idx = layer_idx
        self.world_size = world_size
        self.n_head_local = config.n_head // world_size
        self.n_kv_head_local = config.n_kv_head // world_size
        self.n_embd_local = config.n_embd // world_size
        self.head_dim = config.head_dim
        self.window_size = window_size
        self.ve_gate_channels = config.ve_gate_channels
        self.rope_base = config.rope_base
        self.c_q = ColumnParallelLinear(config.n_embd, config.n_head * self.head_dim, rank=rank, world_size=world_size)
        self.c_k = ColumnParallelLinear(config.n_embd, config.n_kv_head * self.head_dim, rank=rank, world_size=world_size)
        self.c_v = ColumnParallelLinear(config.n_embd, config.n_kv_head * self.head_dim, rank=rank, world_size=world_size)
        self.c_proj = RowParallelLinear(config.n_embd, config.n_embd, rank=rank, world_size=world_size)
        self.ve_gate = (
            ColumnParallelLinear(config.ve_gate_channels, config.n_kv_head, rank=rank, world_size=world_size)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def __call__(self, x: mx.array, ve: mx.array | None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head_local, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head_local, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head_local, self.head_dim).transpose(0, 2, 1, 3)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head_local, self.head_dim).transpose(0, 2, 1, 3)
            gate = 3.0 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            gate = gate.transpose(0, 2, 1)[..., None]
            v = v + gate * ve

        q = apply_rotary_emb(q, offset=0, base=self.rope_base)
        k = apply_rotary_emb(k, offset=0, base=self.rope_base)
        q = rms_norm(q) * 1.15
        k = rms_norm(k) * 1.15
        mask = build_attention_mask(seq_len, seq_len, offset=0, window=self.window_size[0])
        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.head_dim ** -0.5,
            mask=mask,
        )
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_embd_local)
        return self.c_proj(y)


class TensorParallelBlock(nn.Module):
    def __init__(self, config: NanoChatConfig, layer_idx: int, window_size: tuple[int, int], rank: int, world_size: int):
        super().__init__()
        self.attn = TensorParallelCausalSelfAttention(
            config,
            layer_idx=layer_idx,
            window_size=window_size,
            rank=rank,
            world_size=world_size,
        )
        self.mlp = TensorParallelMLP(config.n_embd, rank=rank, world_size=world_size)

    def __call__(self, x: mx.array, ve: mx.array | None) -> mx.array:
        x = x + self.attn(rms_norm(x), ve)
        x = x + self.mlp(rms_norm(x))
        return x


class TensorParallelNanoChatMLX(nn.Module):
    def __init__(self, config: NanoChatConfig, rank: int, world_size: int):
        super().__init__()
        if config.effective_padded_vocab_size % world_size != 0:
            raise ValueError(
                "Tensor parallel mode requires padded vocab size to be divisible by world_size. "
                f"Got {config.effective_padded_vocab_size} and world_size={world_size}."
            )
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = VocabParallelEmbedding(config.effective_padded_vocab_size, config.n_embd, rank=rank, world_size=world_size)
        self.layers = [
            TensorParallelBlock(config, layer_idx, self.window_sizes[layer_idx], rank=rank, world_size=world_size)
            for layer_idx in range(config.n_layer)
        ]
        self.lm_head = VocabParallelLMHead(config.n_embd, config.effective_padded_vocab_size, rank=rank, world_size=world_size)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((config.n_layer,), 0.1, dtype=mx.float32)
        self.value_embeds = {
            f"layer_{i}": VocabParallelEmbedding(
                config.effective_padded_vocab_size,
                (config.n_kv_head // world_size) * config.head_dim,
                rank=rank,
                world_size=world_size,
            )
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

    def __call__(self, idx: mx.array, gather_logits: bool = False) -> mx.array:
        if idx.ndim == 1:
            idx = idx[None, :]
        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x
        for i, block in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = f"layer_{i}"
            ve = self.value_embeds[ve_key](idx).astype(x.dtype) if ve_key in self.value_embeds else None
            x = block(x, ve)
        x = rms_norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        if gather_logits:
            logits = _concat_gather(logits, axis=-1, world_size=self.world_size)
        logits = logits[..., : self.config.vocab_size]
        return self.config.softcap * mx.tanh(logits / self.config.softcap)

    def export_full_weights(self) -> dict[str, mx.array]:
        weights: dict[str, mx.array] = {
            "wte.weight": self.wte.export_weight(),
            "lm_head.weight": self.lm_head.export_weight(),
            "resid_lambdas": self.resid_lambdas,
            "x0_lambdas": self.x0_lambdas,
        }
        for layer_idx, layer in enumerate(self.layers):
            prefix = f"layers.{layer_idx}"
            weights[f"{prefix}.attn.c_q.weight"] = layer.attn.c_q.export_weight()
            weights[f"{prefix}.attn.c_k.weight"] = layer.attn.c_k.export_weight()
            weights[f"{prefix}.attn.c_v.weight"] = layer.attn.c_v.export_weight()
            weights[f"{prefix}.attn.c_proj.weight"] = layer.attn.c_proj.export_weight()
            weights[f"{prefix}.mlp.c_fc.weight"] = layer.mlp.c_fc.export_weight()
            weights[f"{prefix}.mlp.c_proj.weight"] = layer.mlp.c_proj.export_weight()
            if layer.attn.ve_gate is not None:
                weights[f"{prefix}.attn.ve_gate.weight"] = layer.attn.ve_gate.export_weight()

        for key, embed in self.value_embeds.items():
            weights[f"value_embeds.{key}.weight"] = embed.export_weight()
        return weights

    def config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()
