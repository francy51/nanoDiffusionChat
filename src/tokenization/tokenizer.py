from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


@dataclass
class CharacterTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    pad_token_id: int = 0
    mask_token_id: int = 1
    unk_token_id: int = 2

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.unk_token_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(idx, "") for idx in ids if idx >= 3)

    def token_to_id(self, token: str) -> int | None:
        return self.stoi.get(token)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def save(self, path: str) -> None:
        payload = {
            "stoi": self.stoi,
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str) -> CharacterTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {key: int(value) for key, value in payload["stoi"].items()}
        itos = {value: key for key, value in stoi.items()}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_token_id=payload["pad_token_id"],
            mask_token_id=payload["mask_token_id"],
            unk_token_id=payload["unk_token_id"],
        )

    @classmethod
    def from_text(cls, text: str) -> CharacterTokenizer:
        chars = sorted(set(text))
        stoi = {"<pad>": 0, "<mask>": 1, "<unk>": 2}
        for index, char in enumerate(chars, start=3):
            stoi[char] = index
        itos = {value: key for key, value in stoi.items()}
        return cls(stoi=stoi, itos=itos)


class Tokenizer:
    def __init__(self, tokenizer_name: str = "char", corpus_text: str | None = None):
        self.tokenizer_name = tokenizer_name
        self.corpus_text = corpus_text
        self._tokenizer: CharacterTokenizer | HFTokenizer | None = None

    def load(self) -> CharacterTokenizer | HFTokenizer:
        if self._tokenizer is None:
            if self.tokenizer_name == "char":
                if self.corpus_text is None:
                    raise ValueError(
                        "Character tokenizer requires corpus_text to build"
                    )
                self._tokenizer = CharacterTokenizer.from_text(self.corpus_text)
            else:
                self._tokenizer = HFTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.load().get_vocab_size()

    @property
    def mask_token_id(self) -> int:
        tokenizer = self.load()
        if isinstance(tokenizer, CharacterTokenizer):
            return tokenizer.mask_token_id
        token_id = tokenizer.token_to_id("<mask>")
        if token_id is not None:
            return token_id
        fallback = tokenizer.token_to_id("<|endoftext|>")
        if fallback is None:
            raise ValueError("Tokenizer does not expose a mask token policy")
        return fallback

    @property
    def pad_token_id(self) -> int:
        tokenizer = self.load()
        if isinstance(tokenizer, CharacterTokenizer):
            return tokenizer.pad_token_id
        token_id = tokenizer.token_to_id("<pad>")
        if token_id is not None:
            return token_id
        fallback = tokenizer.token_to_id("<|endoftext|>")
        if fallback is None:
            raise ValueError("Tokenizer does not expose a pad token policy")
        return fallback

    def encode(self, text: str) -> list[int]:
        tokenizer = self.load()
        if isinstance(tokenizer, CharacterTokenizer):
            return tokenizer.encode(text)
        return tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        tokenizer = self.load()
        if isinstance(tokenizer, CharacterTokenizer):
            return tokenizer.decode(ids)
        return tokenizer.decode(ids)

    def save(self, path: Path) -> None:
        self.load().save(str(path))

    @classmethod
    def from_file(cls, path: Path) -> Tokenizer:
        instance = cls()
        if path.suffix == ".json":
            try:
                instance._tokenizer = CharacterTokenizer.from_file(str(path))
            except Exception:
                instance._tokenizer = HFTokenizer.from_file(str(path))
        else:
            instance._tokenizer = HFTokenizer.from_file(str(path))
        return instance
