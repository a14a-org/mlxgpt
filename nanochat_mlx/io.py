from __future__ import annotations

from pathlib import Path

from .config import NanoChatConfig
from .model import NanoChatMLX


def load_model(model_dir: str | Path) -> NanoChatMLX:
    model_dir = Path(model_dir)
    config = NanoChatConfig.from_json(model_dir / "config.json")
    model = NanoChatMLX(config)
    model.load_weights(str(model_dir / "weights.safetensors"))
    return model
