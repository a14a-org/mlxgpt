from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer as HFTokenizer


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


@dataclass
class TiktokenTokenizer:
    enc: tiktoken.Encoding
    bos_token: str

    @classmethod
    def from_directory(cls, tokenizer_dir: str | Path) -> "TiktokenTokenizer":
        tokenizer_dir = Path(tokenizer_dir)
        with tokenizer_dir.joinpath("tokenizer.pkl").open("rb") as handle:
            enc = pickle.load(handle)
        bos_token = "<|bos|>" if "<|bos|>" in enc.special_tokens_set else "<|endoftext|>"
        return cls(enc=enc, bos_token=bos_token)

    @classmethod
    def from_pretrained(cls, name: str) -> "TiktokenTokenizer":
        return cls(enc=tiktoken.get_encoding(name), bos_token="<|endoftext|>")

    def encode_special(self, text: str) -> int:
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self) -> int:
        return self.encode_special(self.bos_token)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(
        self,
        text: str | list[str],
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:
        prepend_id = None if prepend is None else (prepend if isinstance(prepend, int) else self.encode_special(prepend))
        append_id = None if append is None else (append if isinstance(append, int) else self.encode_special(append))

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
            return ids

        ids_batch = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        if prepend_id is not None:
            for ids in ids_batch:
                ids.insert(0, prepend_id)
        if append_id is not None:
            for ids in ids_batch:
                ids.append(append_id)
        return ids_batch

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


@dataclass
class HuggingFaceTokenizerWrapper:
    tokenizer: HFTokenizer

    @classmethod
    def from_directory(cls, tokenizer_dir: str | Path) -> "HuggingFaceTokenizerWrapper":
        tokenizer_dir = Path(tokenizer_dir)
        return cls(tokenizer=HFTokenizer.from_file(str(tokenizer_dir / "tokenizer.json")))

    def encode_special(self, text: str) -> int:
        token_id = self.tokenizer.token_to_id(text)
        if token_id is None:
            raise KeyError(f"Special token {text!r} is not present in tokenizer")
        return token_id

    def get_bos_token_id(self) -> int:
        for token in ("<|bos|>", "<|endoftext|>"):
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                return token_id
        raise KeyError("No BOS token found in tokenizer")

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str | list[str],
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:
        del num_threads
        prepend_id = None if prepend is None else (prepend if isinstance(prepend, int) else self.encode_special(prepend))
        append_id = None if append is None else (append if isinstance(append, int) else self.encode_special(append))

        def encode_one(value: str) -> list[int]:
            ids = self.tokenizer.encode(value, add_special_tokens=False).ids
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
            return ids

        if isinstance(text, str):
            return encode_one(text)

        return [encode_one(item) for item in text]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def load_tokenizer(tokenizer_dir: str | Path | None = None, fallback: str = "gpt2"):
    if tokenizer_dir is None:
        return TiktokenTokenizer.from_pretrained(fallback)

    tokenizer_dir = Path(tokenizer_dir)
    if tokenizer_dir.is_file():
        tokenizer_dir = tokenizer_dir.parent

    if (tokenizer_dir / "tokenizer.pkl").exists():
        return TiktokenTokenizer.from_directory(tokenizer_dir)
    if (tokenizer_dir / "tokenizer.json").exists():
        return HuggingFaceTokenizerWrapper.from_directory(tokenizer_dir)

    raise FileNotFoundError(f"No tokenizer.pkl or tokenizer.json found in {tokenizer_dir}")
