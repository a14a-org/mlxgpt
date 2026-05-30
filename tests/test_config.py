"""Unit tests for nanochat_mlx.config.NanoChatConfig.

These exercise the pure configuration logic (validation, derived dimensions,
vocab padding math, and JSON round-tripping) with no MLX dependency.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nanochat_mlx.config import NanoChatConfig


class NanoChatConfigValidationTests(unittest.TestCase):
    def test_defaults_are_consistent(self) -> None:
        config = NanoChatConfig()
        # n_embd (768) must be divisible by n_head (6) for the default to be valid.
        self.assertEqual(config.n_embd % config.n_head, 0)
        self.assertEqual(config.n_head % config.n_kv_head, 0)

    def test_rejects_embd_not_divisible_by_head(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            NanoChatConfig(n_embd=100, n_head=6)
        self.assertIn("must be divisible", str(ctx.exception))

    def test_rejects_head_not_divisible_by_kv_head(self) -> None:
        # n_embd divisible by n_head, but n_head (6) not divisible by n_kv_head (4).
        with self.assertRaises(ValueError):
            NanoChatConfig(n_embd=96, n_head=6, n_kv_head=4)

    def test_grouped_query_attention_is_allowed(self) -> None:
        # n_head divisible by n_kv_head (8 % 2 == 0) is a valid GQA setup.
        config = NanoChatConfig(n_embd=256, n_head=8, n_kv_head=2)
        self.assertEqual(config.n_head, 8)
        self.assertEqual(config.n_kv_head, 2)


class NanoChatConfigDerivedTests(unittest.TestCase):
    def test_head_dim(self) -> None:
        config = NanoChatConfig(n_embd=768, n_head=6)
        self.assertEqual(config.head_dim, 128)

    def test_effective_padded_vocab_rounds_up(self) -> None:
        # 32768 is already a multiple of 64, so it is unchanged.
        self.assertEqual(NanoChatConfig(vocab_size=32768, pad_vocab_size_to=64).effective_padded_vocab_size, 32768)
        # 100 padded up to a multiple of 64 -> 128.
        self.assertEqual(NanoChatConfig(vocab_size=100, pad_vocab_size_to=64).effective_padded_vocab_size, 128)
        # Exact multiple stays put.
        self.assertEqual(NanoChatConfig(vocab_size=128, pad_vocab_size_to=64).effective_padded_vocab_size, 128)

    def test_effective_padded_vocab_respects_explicit_override(self) -> None:
        config = NanoChatConfig(vocab_size=100, pad_vocab_size_to=64, padded_vocab_size=200)
        # An explicit padded_vocab_size short-circuits the rounding.
        self.assertEqual(config.effective_padded_vocab_size, 200)

    def test_padding_with_non_power_of_two_multiple(self) -> None:
        # 130 rounded up to the next multiple of 48 -> 144.
        self.assertEqual(NanoChatConfig(vocab_size=130, pad_vocab_size_to=48).effective_padded_vocab_size, 144)


class NanoChatConfigSerializationTests(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = NanoChatConfig(n_embd=256, n_head=8, n_kv_head=2, n_layer=4, sequence_len=512)
        restored = NanoChatConfig.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_to_dict_is_json_serializable(self) -> None:
        data = NanoChatConfig().to_dict()
        # asdict output must survive a JSON round-trip unchanged.
        self.assertEqual(json.loads(json.dumps(data)), data)

    def test_json_file_roundtrip(self) -> None:
        original = NanoChatConfig(n_embd=384, n_head=6, n_layer=3, vocab_size=1000)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original.to_json(path)
            self.assertTrue(path.exists())
            restored = NanoChatConfig.from_json(path)
        self.assertEqual(restored, original)
        self.assertEqual(restored.head_dim, original.head_dim)


if __name__ == "__main__":
    unittest.main()
