from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        self._tokenizer: HFTokenizer | None = None

    def load(self) -> HFTokenizer:
        if self._tokenizer is None:
            self._tokenizer = HFTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.load().get_vocab_size()

    @property
    def mask_token_id(self) -> int:
        return self.load().token_to_id("<|endoftext|>")

    @property
    def pad_token_id(self) -> int:
        return self.load().token_to_id("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return self.load().encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.load().decode(ids)

    def save(self, path: Path) -> None:
        self.load().save(str(path))

    @classmethod
    def from_file(cls, path: Path) -> "Tokenizer":
        instance = cls()
        instance._tokenizer = HFTokenizer.from_file(str(path))
        return instance
