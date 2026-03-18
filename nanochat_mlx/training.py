from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .checkpoint import (
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
    write_latest_pointer,
)
from .config import NanoChatConfig
from .data import PackedCausalBatchIterator
from .model import NanoChatMLX
from .tensor_parallel import TensorParallelNanoChatMLX


@dataclass
class TrainConfig:
    model_tag: str
    depth: int
    max_seq_len: int
    device_batch_size: int
    total_batch_size: int
    num_iterations: int
    checkpoint_every: int
    base_dir: str
    checkpoint_root: str
    export_root: str
    tokenizer_dir: str
    dataset_dir: str
    aspect_ratio: int = 64
    head_dim: int = 64
    window_pattern: str = "L"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 42
    data_mode: str = "real"
    tokenizer_threads: int = 4
    tokenizer_batch_size: int = 128
    buffer_size: int = 1000
    synthetic_documents: int = 2048
    log_every: int = 10
    verify_every: int = 50
    eval_every: int = 250
    eval_batches: int = 8
    sample_every: int = 500
    sample_prompt: str = "The capital of France is"
    sample_max_tokens: int = 32
    sample_temperature: float = 0.0
    sample_prepend_bos: bool = True
    early_stop_min_step: int = 0
    early_stop_patience_evals: int = 0
    early_stop_degrade_ratio: float = 1.0
    early_stop_vs_champion_ratio: float = 1.0
    champion_best_val: float | None = None
    run_dir: str = ""
    export_final: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelConfig:
    mode: str
    backend: str
    rank: int
    world_size: int
    dp_size: int
    tp_size: int
    dp_group: tuple[int, ...] = field(default_factory=tuple)
    tp_group: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_world(cls, mode: str, backend: str, rank: int, world_size: int) -> "ParallelConfig":
        if mode not in {"dp", "tp"}:
            raise ValueError(f"Unsupported parallel mode: {mode}")
        dp_size = world_size if mode == "dp" else 1
        tp_size = world_size if mode == "tp" else 1
        return cls(
            mode=mode,
            backend=backend,
            rank=rank,
            world_size=world_size,
            dp_size=dp_size,
            tp_size=tp_size,
            dp_group=tuple(range(world_size)) if mode == "dp" else (rank,),
            tp_group=tuple(range(world_size)) if mode == "tp" else (rank,),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointState:
    step: int = 0
    tokens_seen: int = 0
    rng_seed: int = 42
    loader_state: dict[str, Any] | None = None
    val_loader_state: dict[str, Any] | None = None
    last_loss: float | None = None
    last_val_loss: float | None = None
    best_val_loss: float | None = None
    best_val_step: int = 0
    last_sample_text: str | None = None
    last_sample_tokens: list[int] | None = None
    last_tokens_per_second: float | None = None
    no_improvement_evals: int = 0
    stopped_early: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        return cls(
            step=int(data.get("step", 0)),
            tokens_seen=int(data.get("tokens_seen", 0)),
            rng_seed=int(data.get("rng_seed", 42)),
            loader_state=data.get("loader_state"),
            val_loader_state=data.get("val_loader_state"),
            last_loss=data.get("last_loss"),
            last_val_loss=data.get("last_val_loss"),
            best_val_loss=data.get("best_val_loss"),
            best_val_step=int(data.get("best_val_step", 0)),
            last_sample_text=data.get("last_sample_text"),
            last_sample_tokens=data.get("last_sample_tokens"),
            last_tokens_per_second=data.get("last_tokens_per_second"),
            no_improvement_evals=int(data.get("no_improvement_evals", 0)),
            stopped_early=bool(data.get("stopped_early", False)),
            stop_reason=data.get("stop_reason"),
        )


def rounded_model_dim(depth: int, aspect_ratio: int, head_dim: int) -> int:
    base_dim = depth * aspect_ratio
    return ((base_dim + head_dim - 1) // head_dim) * head_dim


def build_model_config(train_config: TrainConfig, vocab_size: int, parallel_config: ParallelConfig) -> NanoChatConfig:
    n_embd = rounded_model_dim(train_config.depth, train_config.aspect_ratio, train_config.head_dim)
    n_head = n_embd // train_config.head_dim
    pad_multiple = math.lcm(64, parallel_config.tp_size)
    return NanoChatConfig(
        sequence_len=train_config.max_seq_len,
        vocab_size=vocab_size,
        n_layer=train_config.depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern=train_config.window_pattern,
        pad_vocab_size_to=pad_multiple,
    )


def _tree_checksum(tree: dict[str, Any]) -> float:
    checksum = 0.0
    for index, (_, value) in enumerate(tree_flatten(tree), start=1):
        checksum += float(np.asarray(mx.sum(value.astype(mx.float32)))) * index
    return checksum


def _sync_dp_gradients(grads: dict[str, Any], parallel_config: ParallelConfig) -> dict[str, Any]:
    if parallel_config.mode != "dp" or parallel_config.dp_size == 1:
        return grads
    return tree_map(lambda g: mx.distributed.all_sum(g) / parallel_config.dp_size, grads)


class ClusterTrainer:
    def __init__(
        self,
        train_config: TrainConfig,
        parallel_config: ParallelConfig,
        model_config: NanoChatConfig,
        model,
        optimizer,
        tokenizer,
        checkpoint_state: CheckpointState,
    ) -> None:
        self.train_config = train_config
        self.parallel_config = parallel_config
        self.model_config = model_config
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.state = checkpoint_state
        self.run_dir = Path(train_config.run_dir or Path("build") / "mlx-train" / train_config.model_tag)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / f"metrics_rank{parallel_config.rank}.jsonl"
        data_rank = parallel_config.rank if parallel_config.mode == "dp" else 0
        data_world = parallel_config.dp_size if parallel_config.mode == "dp" else 1
        self.batch_iterator = PackedCausalBatchIterator(
            tokenizer=tokenizer,
            batch_size=train_config.device_batch_size,
            sequence_len=train_config.max_seq_len,
            split="train",
            rank=data_rank,
            world_size=data_world,
            state=checkpoint_state.loader_state,
            data_dir=train_config.dataset_dir,
            data_mode=train_config.data_mode,
            tokenizer_threads=train_config.tokenizer_threads,
            tokenizer_batch_size=train_config.tokenizer_batch_size,
            buffer_size=train_config.buffer_size,
            synthetic_documents=train_config.synthetic_documents,
        )
        self.val_batch_iterator = None
        if train_config.eval_every > 0 and train_config.eval_batches > 0:
            self.val_batch_iterator = PackedCausalBatchIterator(
                tokenizer=tokenizer,
                batch_size=train_config.device_batch_size,
                sequence_len=train_config.max_seq_len,
                split="val",
                rank=data_rank,
                world_size=data_world,
                state=checkpoint_state.val_loader_state,
                data_dir=train_config.dataset_dir,
                data_mode=train_config.data_mode,
                tokenizer_threads=train_config.tokenizer_threads,
                tokenizer_batch_size=train_config.tokenizer_batch_size,
                buffer_size=train_config.buffer_size,
                synthetic_documents=max(256, train_config.synthetic_documents // 4),
            )
        tokens_per_microbatch = train_config.device_batch_size * train_config.max_seq_len * max(1, data_world)
        self.grad_accum_steps = max(1, math.ceil(train_config.total_batch_size / max(1, tokens_per_microbatch)))
        self.loss_and_grad = nn.value_and_grad(self.model, self._loss_fn)

    def _write_event(self, event: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _loss_fn(self, model, inputs: mx.array, targets: mx.array) -> mx.array:
        logits = model(inputs, gather_logits=True) if self.parallel_config.mode == "tp" else model(inputs)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def _forward_loss(self, inputs: mx.array, targets: mx.array) -> mx.array:
        logits = self.model(inputs, gather_logits=True) if self.parallel_config.mode == "tp" else self.model(inputs)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def _parameter_checksum(self) -> float:
        return _tree_checksum(self.model.parameters())

    def _verify_sync(self) -> None:
        checksum = self._parameter_checksum()
        checksum_tensor = mx.array([checksum], dtype=mx.float32)
        gathered = mx.distributed.all_gather(checksum_tensor) if self.parallel_config.world_size > 1 else checksum_tensor
        values = [float(x) for x in np.asarray(gathered).reshape(-1)]
        if len({round(value, 5) for value in values}) != 1:
            raise RuntimeError(f"Parameter checksums diverged across ranks: {values}")

    def _checkpoint_dir(self, step: int) -> Path:
        return Path(self.train_config.checkpoint_root) / self.train_config.model_tag / f"step_{step:06d}"

    def _save_checkpoint(self) -> None:
        checkpoint_dir = self._checkpoint_dir(self.state.step)
        save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            rank=self.parallel_config.rank,
            model=self.model,
            optimizer=self.optimizer,
            state=self.state.to_dict(),
            model_config=self.model_config.to_dict(),
            train_config=self.train_config.to_dict(),
            parallel_config=self.parallel_config.to_dict(),
        )
        if self.parallel_config.rank == 0:
            write_latest_pointer(self.train_config.checkpoint_root, self.train_config.model_tag, self.state.step)
        self._write_event(
            {
                "event": "checkpoint",
                "rank": self.parallel_config.rank,
                "step": self.state.step,
                "checkpoint_dir": str(checkpoint_dir),
            }
        )

    def _run_validation(self, step_idx: int) -> dict[str, Any] | None:
        if self.val_batch_iterator is None:
            return None
        total_loss = mx.array(0.0, dtype=mx.float32)
        last_loader_state = self.state.val_loader_state
        for _ in range(self.train_config.eval_batches):
            batch_inputs, batch_targets, loader_state = self.val_batch_iterator.next_batch()
            last_loader_state = loader_state
            inputs = mx.array(batch_inputs, dtype=mx.int32)
            targets = mx.array(batch_targets, dtype=mx.int32)
            total_loss = total_loss + self._forward_loss(inputs, targets)
        mean_loss = total_loss / self.train_config.eval_batches
        if self.parallel_config.world_size > 1:
            mean_loss = mx.distributed.all_sum(mean_loss) / self.parallel_config.world_size
        mx.eval(mean_loss)

        self.state.val_loader_state = last_loader_state
        self.state.last_val_loss = float(mean_loss.item())
        improved = self.state.best_val_loss is None or self.state.last_val_loss < self.state.best_val_loss
        if improved:
            self.state.best_val_loss = self.state.last_val_loss
            self.state.best_val_step = step_idx
            self.state.no_improvement_evals = 0
        else:
            self.state.no_improvement_evals += 1

        event = {
            "event": "val_step",
            "mode": self.parallel_config.mode,
            "rank": self.parallel_config.rank,
            "step": step_idx,
            "loss": self.state.last_val_loss,
            "best_val_loss": self.state.best_val_loss,
            "best_val_step": self.state.best_val_step,
            "no_improvement_evals": self.state.no_improvement_evals,
        }
        self._write_event(event)
        if self.parallel_config.rank == 0:
            print(json.dumps(event, sort_keys=True))
        return event

    def _maybe_early_stop(self, step_idx: int, val_event: dict[str, Any] | None) -> str | None:
        if val_event is None:
            return None
        if step_idx < self.train_config.early_stop_min_step:
            return None
        if self.state.best_val_loss is None or self.state.last_val_loss is None:
            return None

        champion = self.train_config.champion_best_val
        if champion is not None and champion > 0:
            champion_limit = champion * self.train_config.early_stop_vs_champion_ratio
            if self.state.last_val_loss > champion_limit:
                return (
                    f"validation non-competitive vs champion {champion:.6f} at "
                    f"{self.train_config.early_stop_vs_champion_ratio:.3f}x threshold"
                )

        if self.train_config.early_stop_patience_evals <= 0:
            return None
        if self.state.no_improvement_evals < self.train_config.early_stop_patience_evals:
            return None

        degrade_limit = self.state.best_val_loss * self.train_config.early_stop_degrade_ratio
        if self.state.last_val_loss > degrade_limit:
            return (
                f"validation degraded past {self.train_config.early_stop_degrade_ratio:.3f}x best "
                f"after {self.state.no_improvement_evals} evals"
            )
        return None

    def _emit_early_stop(self, step_idx: int, reason: str) -> None:
        self.state.stopped_early = True
        self.state.stop_reason = reason
        event = {
            "event": "early_stop",
            "mode": self.parallel_config.mode,
            "rank": self.parallel_config.rank,
            "step": step_idx,
            "reason": reason,
            "best_val_loss": self.state.best_val_loss,
            "best_val_step": self.state.best_val_step,
            "last_val_loss": self.state.last_val_loss,
        }
        self._write_event(event)
        if self.parallel_config.rank == 0:
            print(json.dumps(event, sort_keys=True))

    def _run_sample(self, step_idx: int) -> dict[str, Any] | None:
        if self.parallel_config.rank != 0 or self.train_config.sample_every <= 0 or not hasattr(self.model, "generate"):
            return None

        prompt_tokens = self.tokenizer.encode(self.train_config.sample_prompt)
        if self.train_config.sample_prepend_bos:
            prompt_tokens = [self.tokenizer.get_bos_token_id()] + prompt_tokens
        if not prompt_tokens:
            raise ValueError("Sample prompt produced zero tokens")
        if max(prompt_tokens) >= self.model.config.vocab_size:
            raise ValueError("Sample prompt token ids exceed model vocabulary")

        generated_tokens = self.model.generate(
            prompt_tokens,
            max_tokens=self.train_config.sample_max_tokens,
            temperature=self.train_config.sample_temperature,
            seed=self.train_config.seed + step_idx,
        )
        text = self.tokenizer.decode(prompt_tokens + generated_tokens)
        self.state.last_sample_text = text
        self.state.last_sample_tokens = generated_tokens
        event = {
            "event": "sample",
            "mode": self.parallel_config.mode,
            "rank": self.parallel_config.rank,
            "step": step_idx,
            "prompt": self.train_config.sample_prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "text": text,
            "temperature": self.train_config.sample_temperature,
            "max_tokens": self.train_config.sample_max_tokens,
        }
        self._write_event(event)
        print(json.dumps(event, sort_keys=True))
        return event

    def _write_summary(self, export_dir: str | Path | None = None) -> dict[str, Any]:
        summary = {
            "event": "summary",
            "mode": self.parallel_config.mode,
            "step": self.state.step,
            "final_train_loss": self.state.last_loss,
            "last_val_loss": self.state.last_val_loss,
            "best_val_loss": self.state.best_val_loss,
            "best_val_step": self.state.best_val_step,
            "last_sample_text": self.state.last_sample_text,
            "last_sample_tokens": self.state.last_sample_tokens,
            "final_tokens_per_second": self.state.last_tokens_per_second,
            "tokens_seen": self.state.tokens_seen,
            "no_improvement_evals": self.state.no_improvement_evals,
            "stopped_early": self.state.stopped_early,
            "stop_reason": self.state.stop_reason,
            "export_dir": None if export_dir is None else str(export_dir),
        }
        if self.parallel_config.rank == 0:
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def export_model(self, export_dir: str | Path | None = None) -> Path | None:
        export_dir = Path(export_dir or Path(self.train_config.export_root) / self.train_config.model_tag)
        if self.parallel_config.rank != 0:
            return None

        export_dir.mkdir(parents=True, exist_ok=True)
        self.model_config.to_json(export_dir / "config.json")
        if self.parallel_config.mode == "tp":
            mx.save_safetensors(str(export_dir / "weights.safetensors"), self.model.export_full_weights())
        else:
            self.model.save_weights(str(export_dir / "weights.safetensors"))

        tokenizer_src = Path(self.train_config.tokenizer_dir)
        tokenizer_dst = export_dir / "tokenizer"
        if tokenizer_src.exists():
            if tokenizer_dst.exists():
                shutil.rmtree(tokenizer_dst)
            shutil.copytree(tokenizer_src, tokenizer_dst)

        summary = {
            "event": "export",
            "mode": self.parallel_config.mode,
            "export_dir": str(export_dir),
            "step": self.state.step,
        }
        self._write_event(summary)
        return export_dir

    def train(self) -> dict[str, Any]:
        stopped_early = False
        for step_idx in range(self.state.step + 1, self.train_config.num_iterations + 1):
            start_time = time.time()
            accumulated_grads = None
            accumulated_loss = mx.array(0.0, dtype=mx.float32)
            last_loader_state = self.state.loader_state

            for _ in range(self.grad_accum_steps):
                batch_inputs, batch_targets, loader_state = self.batch_iterator.next_batch()
                last_loader_state = loader_state
                inputs = mx.array(batch_inputs, dtype=mx.int32)
                targets = mx.array(batch_targets, dtype=mx.int32)
                loss, grads = self.loss_and_grad(self.model, inputs, targets)
                accumulated_loss = accumulated_loss + loss
                accumulated_grads = grads if accumulated_grads is None else tree_map(lambda a, b: a + b, accumulated_grads, grads)

            accumulated_grads = tree_map(lambda g: g / self.grad_accum_steps, accumulated_grads)
            accumulated_grads = _sync_dp_gradients(accumulated_grads, self.parallel_config)
            mean_loss = accumulated_loss / self.grad_accum_steps
            if self.parallel_config.world_size > 1:
                mean_loss = mx.distributed.all_sum(mean_loss) / self.parallel_config.world_size

            self.optimizer.update(self.model, accumulated_grads)
            mx.eval(self.model.parameters(), self.optimizer.state, mean_loss)

            self.state.step = step_idx
            self.state.loader_state = last_loader_state
            self.state.tokens_seen += self.train_config.device_batch_size * self.train_config.max_seq_len * self.grad_accum_steps
            self.state.last_loss = float(mean_loss.item())

            elapsed = time.time() - start_time
            tokens_per_second = (
                self.train_config.device_batch_size
                * self.train_config.max_seq_len
                * self.grad_accum_steps
                * max(1, self.parallel_config.dp_size)
            ) / max(elapsed, 1e-6)
            self.state.last_tokens_per_second = tokens_per_second

            event = {
                "event": "train_step",
                "mode": self.parallel_config.mode,
                "rank": self.parallel_config.rank,
                "step": step_idx,
                "loss": self.state.last_loss,
                "tokens_seen": self.state.tokens_seen,
                "tokens_per_second": tokens_per_second,
                "grad_accum_steps": self.grad_accum_steps,
            }
            self._write_event(event)

            if self.parallel_config.rank == 0 and (
                step_idx == 1 or step_idx % self.train_config.log_every == 0 or step_idx == self.train_config.num_iterations
            ):
                print(json.dumps(event, sort_keys=True))

            if step_idx % self.train_config.verify_every == 0:
                self._verify_sync()

            if self.train_config.eval_every > 0 and self.train_config.eval_batches > 0 and step_idx % self.train_config.eval_every == 0:
                val_event = self._run_validation(step_idx)
                stop_reason = self._maybe_early_stop(step_idx, val_event)
                if stop_reason is not None:
                    self._emit_early_stop(step_idx, stop_reason)
                    self._save_checkpoint()
                    stopped_early = True
                    break

            if self.train_config.sample_every > 0 and step_idx % self.train_config.sample_every == 0:
                self._run_sample(step_idx)

            if step_idx % self.train_config.checkpoint_every == 0 or step_idx == self.train_config.num_iterations:
                self._save_checkpoint()

        if stopped_early and self.parallel_config.rank == 0:
            print(json.dumps({"event": "train_stopped_early", "step": self.state.step, "reason": self.state.stop_reason}, sort_keys=True))

        export_dir = self.export_model() if self.train_config.export_final else None
        summary = self._write_summary(export_dir)
        result = {
            "step": self.state.step,
            "last_loss": self.state.last_loss,
            "last_val_loss": self.state.last_val_loss,
            "best_val_loss": self.state.best_val_loss,
            "best_val_step": self.state.best_val_step,
            "last_sample_text": self.state.last_sample_text,
            "stopped_early": self.state.stopped_early,
            "stop_reason": self.state.stop_reason,
            "checkpoint_dir": str(self._checkpoint_dir(self.state.step)),
            "export_dir": None if export_dir is None else str(export_dir),
            "mode": self.parallel_config.mode,
            "world_size": self.parallel_config.world_size,
            "summary": summary,
        }
        if self.parallel_config.rank == 0:
            (self.run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def create_model(model_config: NanoChatConfig, parallel_config: ParallelConfig):
    if parallel_config.mode == "tp":
        return TensorParallelNanoChatMLX(model_config, rank=parallel_config.rank, world_size=parallel_config.world_size)
    return NanoChatMLX(model_config)


def create_optimizer(train_config: TrainConfig):
    return optim.AdamW(train_config.learning_rate, weight_decay=train_config.weight_decay)


def maybe_resume(
    train_config: TrainConfig,
    parallel_config: ParallelConfig,
    model,
    optimizer,
    resume: str | Path | None,
) -> CheckpointState:
    if not resume:
        return CheckpointState(rng_seed=train_config.seed)
    checkpoint_dir = resolve_resume_checkpoint(train_config.checkpoint_root, train_config.model_tag, resume)
    state_dict = load_training_checkpoint(checkpoint_dir=checkpoint_dir, rank=parallel_config.rank, model=model, optimizer=optimizer)
    return CheckpointState.from_dict(state_dict)
