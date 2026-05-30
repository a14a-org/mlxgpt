"""Unit tests for nanochat_mlx.data.

Covers the loader-state serialization, the deterministic synthetic document
source, and the causal packing iterator. These are pure NumPy/dataclass logic
with no MLX dependency, so a tiny fake tokenizer is enough to drive the packer.
"""

from __future__ import annotations

import unittest

import numpy as np

from nanochat_mlx.data import (
    LoaderState,
    PackedCausalBatchIterator,
    SyntheticDocumentBatchSource,
)


class _FakeTokenizer:
    """Minimal tokenizer that maps each document to a fixed-length token list.

    Mirrors the interface PackedCausalBatchIterator relies on:
    get_bos_token_id() and a batched encode(prepend=...) returning list[list[int]].
    """

    def __init__(self, doc_token_len: int, bos: int = 1) -> None:
        self.doc_token_len = doc_token_len
        self.bos = bos

    def get_bos_token_id(self) -> int:
        return self.bos

    def encode(self, texts, prepend=None, num_threads=1):  # noqa: ANN001
        out: list[list[int]] = []
        for i, _ in enumerate(texts):
            body = [(i % 50) + 10] * self.doc_token_len
            out.append(([prepend] if prepend is not None else []) + body)
        return out


class LoaderStateTests(unittest.TestCase):
    def test_from_none_returns_defaults(self) -> None:
        state = LoaderState.from_dict(None)
        self.assertEqual(state.pq_idx, 0)
        self.assertIsNone(state.rg_idx)
        self.assertEqual(state.epoch, 1)

    def test_roundtrip_preserves_values(self) -> None:
        state = LoaderState(pq_idx=7, rg_idx=3, epoch=4)
        restored = LoaderState.from_dict(state.to_dict())
        self.assertEqual(restored, state)

    def test_from_dict_coerces_types_and_preserves_none_rg(self) -> None:
        state = LoaderState.from_dict({"pq_idx": "5", "rg_idx": None, "epoch": "2"})
        self.assertEqual(state.pq_idx, 5)
        self.assertIsNone(state.rg_idx)
        self.assertEqual(state.epoch, 2)


class SyntheticDocumentBatchSourceTests(unittest.TestCase):
    def test_batch_size_and_determinism(self) -> None:
        src_a = SyntheticDocumentBatchSource("train", rank=0, world_size=1, tokenizer_batch_size=8, num_documents=64)
        src_b = SyntheticDocumentBatchSource("train", rank=0, world_size=1, tokenizer_batch_size=8, num_documents=64)
        batch_a, state_a = next(src_a)
        batch_b, state_b = next(src_b)
        self.assertEqual(len(batch_a), 8)
        # Two sources built with the same params must yield identical batches.
        self.assertEqual(batch_a, batch_b)
        self.assertEqual(state_a, state_b)

    def test_split_changes_document_text(self) -> None:
        train_batch, _ = next(SyntheticDocumentBatchSource("train", 0, 1, num_documents=16))
        val_batch, _ = next(SyntheticDocumentBatchSource("val", 0, 1, num_documents=16))
        self.assertTrue(train_batch[0].startswith("train"))
        self.assertTrue(val_batch[0].startswith("val"))

    def test_ranks_read_disjoint_documents(self) -> None:
        # With world_size=2 the two ranks should not start on the same document.
        b0, _ = next(SyntheticDocumentBatchSource("train", rank=0, world_size=2, tokenizer_batch_size=1, num_documents=64))
        b1, _ = next(SyntheticDocumentBatchSource("train", rank=1, world_size=2, tokenizer_batch_size=1, num_documents=64))
        self.assertNotEqual(b0[0], b1[0])

    def test_position_advances_state(self) -> None:
        src = SyntheticDocumentBatchSource("train", 0, 1, tokenizer_batch_size=4, num_documents=64)
        _, first = next(src)
        _, second = next(src)
        self.assertEqual(first.pq_idx, 1)
        self.assertEqual(second.pq_idx, 2)


class PackedCausalBatchIteratorTests(unittest.TestCase):
    def _make_iter(self, seq_len: int, batch_size: int, doc_len: int) -> PackedCausalBatchIterator:
        return PackedCausalBatchIterator(
            tokenizer=_FakeTokenizer(doc_token_len=doc_len),
            batch_size=batch_size,
            sequence_len=seq_len,
            split="train",
            rank=0,
            world_size=1,
            data_mode="synthetic",
            tokenizer_batch_size=16,
            buffer_size=64,
            synthetic_documents=512,
        )

    def test_batch_shapes_and_causal_shift(self) -> None:
        seq_len, batch_size = 16, 4
        it = self._make_iter(seq_len=seq_len, batch_size=batch_size, doc_len=5)
        inputs, targets, state = it.next_batch()
        # row_capacity = seq_len + 1, then split into inputs[:-1] and targets[1:].
        self.assertEqual(inputs.shape, (batch_size, seq_len))
        self.assertEqual(targets.shape, (batch_size, seq_len))
        self.assertEqual(inputs.dtype, np.int32)
        # Causal language modeling: target at position t is the input at t+1.
        self.assertTrue(np.array_equal(inputs[:, 1:], targets[:, :-1]))

    def test_rows_are_densely_packed(self) -> None:
        # Documents shorter than the row should be packed back-to-back; with a
        # BOS token (id 1) the rows are fully filled, leaving no trailing zeros.
        it = self._make_iter(seq_len=24, batch_size=3, doc_len=4)
        inputs, targets, _ = it.next_batch()
        full = np.concatenate([inputs, targets[:, -1:]], axis=1)
        self.assertEqual(full.shape, (3, 25))
        self.assertFalse(np.any(full == 0), "packed rows should contain no padding zeros")

    def test_state_dict_returned(self) -> None:
        it = self._make_iter(seq_len=16, batch_size=2, doc_len=6)
        _, _, state = it.next_batch()
        self.assertIn("pq_idx", state)
        self.assertIn("epoch", state)

    def test_real_mode_requires_data_dir(self) -> None:
        with self.assertRaises(ValueError):
            PackedCausalBatchIterator(
                tokenizer=_FakeTokenizer(4),
                batch_size=1,
                sequence_len=8,
                split="train",
                rank=0,
                world_size=1,
                data_mode="real",
                data_dir=None,
            )


if __name__ == "__main__":
    unittest.main()
