from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class NanoChatConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    padded_vocab_size: int | None = None
    pad_vocab_size_to: int = 64
    rope_base: float = 100000.0
    softcap: float = 15.0
    ve_gate_channels: int = 12

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head={self.n_head} must be divisible by n_kv_head={self.n_kv_head}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def effective_padded_vocab_size(self) -> int:
        if self.padded_vocab_size is not None:
            return self.padded_vocab_size
        multiple = self.pad_vocab_size_to
        return ((self.vocab_size + multiple - 1) // multiple) * multiple

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "NanoChatConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "NanoChatConfig":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
