from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _list_parquet_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.iterdir() if path.suffix == ".parquet" and not path.name.endswith(".tmp"))


@dataclass
class LoaderState:
    pq_idx: int = 0
    rg_idx: int | None = None
    epoch: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LoaderState":
        if data is None:
            return cls()
        return cls(
            pq_idx=int(data.get("pq_idx", 0)),
            rg_idx=None if data.get("rg_idx") is None else int(data["rg_idx"]),
            epoch=int(data.get("epoch", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"pq_idx": self.pq_idx, "rg_idx": self.rg_idx, "epoch": self.epoch}


class DocumentBatchSource:
    def __iter__(self):
        return self

    def __next__(self) -> tuple[list[str], LoaderState]:
        raise NotImplementedError


class SyntheticDocumentBatchSource(DocumentBatchSource):
    def __init__(
        self,
        split: str,
        rank: int,
        world_size: int,
        state: LoaderState | None = None,
        tokenizer_batch_size: int = 128,
        num_documents: int = 2048,
    ) -> None:
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.documents = self._build_documents(split, num_documents)
        self.state = state or LoaderState()
        self.position = self.state.pq_idx

    def _build_documents(self, split: str, num_documents: int) -> list[str]:
        suffix = "train" if split == "train" else "val"
        return [
            (
                f"{suffix} synthetic document {i}. "
                "The quick brown fox jumps over the lazy dog. "
                "Apple Silicon MLX cluster training smoke data."
            )
            for i in range(num_documents)
        ]

    def __next__(self) -> tuple[list[str], LoaderState]:
        batch: list[str] = []
        start = self.position * self.world_size + self.rank
        for offset in range(self.tokenizer_batch_size):
            doc_idx = (start + offset * self.world_size) % len(self.documents)
            batch.append(self.documents[doc_idx])
        self.position += 1
        self.state = LoaderState(pq_idx=self.position, rg_idx=0, epoch=1 + (start // max(1, len(self.documents))))
        return batch, self.state


class ParquetDocumentBatchSource(DocumentBatchSource):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        rank: int,
        world_size: int,
        state: LoaderState | None = None,
        tokenizer_batch_size: int = 128,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for real-data MLX training. Install it with: pip install '.[train]'"
            ) from exc

        self.pq = pq
        self.rank = rank
        self.world_size = world_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.state = state or LoaderState()
        parquet_paths = _list_parquet_files(Path(data_dir))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")
        self.parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        if not self.parquet_paths:
            raise FileNotFoundError(f"Split {split!r} produced no parquet shards in {data_dir}")
        self.first_pass = True

    def __next__(self) -> tuple[list[str], LoaderState]:
        while True:
            pq_idx = self.state.pq_idx if self.first_pass else 0
            while pq_idx < len(self.parquet_paths):
                filepath = self.parquet_paths[pq_idx]
                parquet_file = self.pq.ParquetFile(filepath)
                if self.first_pass and self.state.rg_idx is not None and pq_idx == self.state.pq_idx:
                    base_idx = self.state.rg_idx // self.world_size
                    base_idx += 1
                    rg_idx = base_idx * self.world_size + self.rank
                    self.state = LoaderState(pq_idx=self.state.pq_idx, rg_idx=None, epoch=self.state.epoch)
                    if rg_idx >= parquet_file.num_row_groups:
                        pq_idx += 1
                        continue
                else:
                    rg_idx = self.rank

                while rg_idx < parquet_file.num_row_groups:
                    row_group = parquet_file.read_row_group(rg_idx)
                    batch = row_group.column("text").to_pylist()
                    for start in range(0, len(batch), self.tokenizer_batch_size):
                        self.state = LoaderState(pq_idx=pq_idx, rg_idx=rg_idx, epoch=self.state.epoch)
                        return batch[start : start + self.tokenizer_batch_size], self.state
                    rg_idx += self.world_size
                pq_idx += 1

            self.first_pass = False
            self.state = LoaderState(pq_idx=0, rg_idx=None, epoch=self.state.epoch + 1)


class PackedCausalBatchIterator:
    def __init__(
        self,
        tokenizer,
        batch_size: int,
        sequence_len: int,
        split: str,
        rank: int,
        world_size: int,
        state: dict[str, Any] | None = None,
        data_dir: str | Path | None = None,
        data_mode: str = "real",
        tokenizer_threads: int = 4,
        tokenizer_batch_size: int = 128,
        buffer_size: int = 1000,
        synthetic_documents: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.row_capacity = sequence_len + 1
        self.buffer_size = buffer_size
        self.tokenizer_threads = tokenizer_threads
        self.bos_token = tokenizer.get_bos_token_id()
        self.doc_buffer: list[list[int]] = []
        self.state = LoaderState.from_dict(state)

        if data_mode == "synthetic":
            self.source = SyntheticDocumentBatchSource(
                split=split,
                rank=rank,
                world_size=world_size,
                state=self.state,
                tokenizer_batch_size=tokenizer_batch_size,
                num_documents=synthetic_documents,
            )
        else:
            if data_dir is None:
                raise ValueError("data_dir is required when data_mode='real'")
            self.source = ParquetDocumentBatchSource(
                data_dir=data_dir,
                split=split,
                rank=rank,
                world_size=world_size,
                state=self.state,
                tokenizer_batch_size=tokenizer_batch_size,
            )

    def _refill_buffer(self) -> None:
        texts, state = next(self.source)
        token_lists = self.tokenizer.encode(
            texts,
            prepend=self.bos_token,
            num_threads=self.tokenizer_threads,
        )
        self.state = state
        self.doc_buffer.extend(token_lists)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = np.zeros((self.batch_size, self.row_capacity), dtype=np.int32)

        for row_idx in range(self.batch_size):
            pos = 0
            while pos < self.row_capacity:
                while len(self.doc_buffer) < self.buffer_size:
                    self._refill_buffer()

                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = idx
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    rows[row_idx, pos : pos + len(doc)] = np.asarray(doc, dtype=np.int32)
                    pos += len(doc)
                    continue

                shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                doc = self.doc_buffer.pop(shortest_idx)
                rows[row_idx, pos : pos + remaining] = np.asarray(doc[:remaining], dtype=np.int32)
                pos += remaining

        return rows[:, :-1], rows[:, 1:], self.state.to_dict()
