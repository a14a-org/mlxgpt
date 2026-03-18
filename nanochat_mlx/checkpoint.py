from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def _flatten_tree(tree: dict[str, Any]) -> dict[str, mx.array]:
    flat = tree_flatten(tree)
    return {key: value for key, value in flat}


def _load_npz_tree(path: Path) -> dict[str, Any]:
    loaded = mx.load(str(path))
    items = sorted(loaded.items(), key=lambda item: item[0])
    return tree_unflatten(items)


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    rank: int,
    model,
    optimizer,
    state: dict[str, Any],
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    parallel_config: dict[str, Any],
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"weights_rank{rank}.safetensors"
    optimizer_path = checkpoint_dir / f"optimizer_rank{rank}.npz"
    model.save_weights(str(model_path))
    mx.savez(str(optimizer_path), **_flatten_tree(optimizer.state))

    (checkpoint_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (checkpoint_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (checkpoint_dir / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    (checkpoint_dir / "parallel_config.json").write_text(json.dumps(parallel_config, indent=2), encoding="utf-8")


def load_training_checkpoint(checkpoint_dir: str | Path, rank: int, model, optimizer) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / f"weights_rank{rank}.safetensors"
    optimizer_path = checkpoint_dir / f"optimizer_rank{rank}.npz"
    if rank != 0 and not model_path.exists():
        fallback_model_path = checkpoint_dir / "weights_rank0.safetensors"
        if fallback_model_path.exists():
            model_path = fallback_model_path
    if rank != 0 and not optimizer_path.exists():
        fallback_optimizer_path = checkpoint_dir / "optimizer_rank0.npz"
        if fallback_optimizer_path.exists():
            optimizer_path = fallback_optimizer_path
    model.load_weights(str(model_path))
    optimizer.state = _load_npz_tree(optimizer_path)
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_latest_pointer(checkpoint_root: str | Path, tag: str, step: int) -> Path:
    checkpoint_root = Path(checkpoint_root)
    model_root = checkpoint_root / tag
    model_root.mkdir(parents=True, exist_ok=True)
    latest_path = model_root / "latest.json"
    latest_path.write_text(json.dumps({"step": step, "path": str(model_root / f"step_{step:06d}")}, indent=2), encoding="utf-8")
    return latest_path


def resolve_resume_checkpoint(checkpoint_root: str | Path, tag: str, resume: str | Path | None) -> Path:
    checkpoint_root = Path(checkpoint_root)
    model_root = checkpoint_root / tag

    def _latest_step_dir(root: Path) -> Path:
        step_dirs = sorted(
            (path for path in root.glob("step_*") if path.is_dir()),
            key=lambda path: path.name,
        )
        if not step_dirs:
            raise FileNotFoundError(f"No checkpoint directories found for tag {tag} in {root}")
        return step_dirs[-1]

    if resume:
        resume_path = Path(resume)
        if resume_path.is_dir():
            return resume_path
        if not resume_path.exists():
            if resume_path.name == "latest.json" and resume_path.parent == model_root:
                return _latest_step_dir(model_root)
            raise FileNotFoundError(f"Resume pointer not found: {resume_path}")
        resume_data = json.loads(resume_path.read_text(encoding="utf-8"))
        return Path(resume_data["path"])
    latest_path = model_root / "latest.json"
    if not latest_path.exists():
        return _latest_step_dir(model_root)
    latest_data = json.loads(latest_path.read_text(encoding="utf-8"))
    return Path(latest_data["path"])
