from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        seq_len: int = 512,
        mask_token_id: int = 50256,
        pad_token_id: int = 50257,
    ):
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.sequences = self._load_sequences(data_path)

    def _load_sequences(self, path: Path) -> list[Tensor]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        data = torch.load(path)
        if isinstance(data, Tensor):
            if data.dim() != 2:
                raise ValueError(f"Expected [num_sequences, seq_len], got {data.shape}")
            return [row.clone() for row in data]
        if not isinstance(data, list):
            raise ValueError(f"Expected list of tensors or 2D tensor, got {type(data)}")

        return data

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tensor:
        seq = self.sequences[idx]

        if len(seq) < self.seq_len:
            padding = torch.full(
                (self.seq_len - len(seq),),
                self.pad_token_id,
                dtype=torch.long,
            )
            seq = torch.cat([seq, padding])
        elif len(seq) > self.seq_len:
            seq = seq[: self.seq_len]

        return seq


TokenDataset = DiffusionDataset
