from __future__ import annotations

import torch
from torch import Tensor


def collate_token_batches(batch: list[Tensor]) -> Tensor:
    """Stack fixed-length token sequences into a batch tensor."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    for item in batch:
        if item.dim() != 1:
            raise ValueError(
                f"Expected 1D token tensors, got shape {tuple(item.shape)}"
            )
    return torch.stack(batch, dim=0)
