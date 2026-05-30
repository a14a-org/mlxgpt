"""Unit tests for the pure configuration helpers in nanochat_mlx.training.

Covers model-dimension rounding, model-config derivation (including
tensor-parallel vocab padding), ParallelConfig.from_world group layout, and
CheckpointState serialization. No training is actually run.
"""

from __future__ import annotations

import math
import unittest

from nanochat_mlx.training import (
    CheckpointState,
    ParallelConfig,
    TrainConfig,
    build_model_config,
    rounded_model_dim,
)


def _train_config(**overrides) -> TrainConfig:
    base = dict(
        model_tag="t",
        depth=12,
        max_seq_len=512,
        device_batch_size=2,
        total_batch_size=64,
        num_iterations=1,
        checkpoint_every=1,
        base_dir="",
        checkpoint_root="",
        export_root="",
        tokenizer_dir="",
        dataset_dir="",
    )
    base.update(overrides)
    return TrainConfig(**base)


class RoundedModelDimTests(unittest.TestCase):
    def test_rounds_up_to_head_dim_multiple(self) -> None:
        # depth*aspect_ratio = 12*64 = 768, already a multiple of head_dim 64.
        self.assertEqual(rounded_model_dim(depth=12, aspect_ratio=64, head_dim=64), 768)

    def test_rounds_up_when_not_a_multiple(self) -> None:
        # 10*16 = 160, head_dim 16 -> 160 (exact).
        self.assertEqual(rounded_model_dim(depth=10, aspect_ratio=16, head_dim=16), 160)
        # 5*16 = 80, head_dim 24 -> next multiple of 24 above 80 is 96.
        self.assertEqual(rounded_model_dim(depth=5, aspect_ratio=16, head_dim=24), 96)

    def test_result_is_always_divisible_by_head_dim(self) -> None:
        for depth in (1, 3, 7, 12, 20):
            dim = rounded_model_dim(depth=depth, aspect_ratio=37, head_dim=64)
            self.assertEqual(dim % 64, 0)


class BuildModelConfigTests(unittest.TestCase):
    def test_derives_consistent_head_count(self) -> None:
        pc = ParallelConfig.from_world("dp", "any", rank=0, world_size=1)
        cfg = build_model_config(_train_config(depth=12, aspect_ratio=64, head_dim=64), vocab_size=1000, parallel_config=pc)
        self.assertEqual(cfg.n_embd, 768)
        self.assertEqual(cfg.n_head, 12)
        self.assertEqual(cfg.n_head, cfg.n_kv_head)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.n_layer, 12)
        self.assertEqual(cfg.sequence_len, 512)

    def test_pad_multiple_is_lcm_of_64_and_tp_size(self) -> None:
        pc = ParallelConfig.from_world("tp", "any", rank=0, world_size=3)
        cfg = build_model_config(_train_config(), vocab_size=1000, parallel_config=pc)
        # lcm(64, 3) = 192; padded vocab must be a multiple of that.
        self.assertEqual(cfg.pad_vocab_size_to, math.lcm(64, 3))
        self.assertEqual(cfg.effective_padded_vocab_size % math.lcm(64, 3), 0)

    def test_produced_config_is_self_consistent(self) -> None:
        pc = ParallelConfig.from_world("dp", "any", rank=0, world_size=2)
        cfg = build_model_config(_train_config(depth=4, aspect_ratio=16, head_dim=16), vocab_size=500, parallel_config=pc)
        # The NanoChatConfig __post_init__ validation must accept the derived dims.
        self.assertEqual(cfg.n_embd % cfg.n_head, 0)


class ParallelConfigTests(unittest.TestCase):
    def test_dp_layout(self) -> None:
        pc = ParallelConfig.from_world("dp", "ring", rank=1, world_size=4)
        self.assertEqual(pc.dp_size, 4)
        self.assertEqual(pc.tp_size, 1)
        self.assertEqual(pc.dp_group, (0, 1, 2, 3))
        self.assertEqual(pc.tp_group, (1,))

    def test_tp_layout(self) -> None:
        pc = ParallelConfig.from_world("tp", "ring", rank=2, world_size=4)
        self.assertEqual(pc.tp_size, 4)
        self.assertEqual(pc.dp_size, 1)
        self.assertEqual(pc.tp_group, (0, 1, 2, 3))
        self.assertEqual(pc.dp_group, (2,))

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            ParallelConfig.from_world("pp", "ring", rank=0, world_size=2)

    def test_to_dict_roundtrips_through_constructor(self) -> None:
        pc = ParallelConfig.from_world("dp", "any", rank=0, world_size=2)
        self.assertEqual(pc.to_dict()["dp_group"], (0, 1))


class CheckpointStateTests(unittest.TestCase):
    def test_defaults(self) -> None:
        state = CheckpointState()
        self.assertEqual(state.step, 0)
        self.assertEqual(state.rng_seed, 42)
        self.assertFalse(state.stopped_early)

    def test_roundtrip(self) -> None:
        state = CheckpointState(
            step=5,
            tokens_seen=1024,
            best_val_loss=1.23,
            best_val_step=4,
            stopped_early=True,
            stop_reason="patience",
            loader_state={"pq_idx": 2},
        )
        restored = CheckpointState.from_dict(state.to_dict())
        self.assertEqual(restored, state)

    def test_from_dict_coerces_numeric_strings(self) -> None:
        state = CheckpointState.from_dict({"step": "9", "tokens_seen": "2048", "best_val_step": "3"})
        self.assertEqual(state.step, 9)
        self.assertEqual(state.tokens_seen, 2048)
        self.assertEqual(state.best_val_step, 3)

    def test_from_dict_tolerates_missing_keys(self) -> None:
        state = CheckpointState.from_dict({})
        self.assertEqual(state, CheckpointState())


if __name__ == "__main__":
    unittest.main()
