from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected 3D logits tensor, got {logits.dim()}D")
    if targets.shape != mask.shape:
        raise ValueError("targets and mask must have the same shape")
    batch_size, seq_len, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction="none",
    ).view(batch_size, seq_len)
    masked_loss = loss * mask.float()
    if reduction == "mean":
        return masked_loss.sum() / mask.float().sum().clamp(min=1.0)
    if reduction == "sum":
        return masked_loss.sum()
    if reduction == "none":
        return masked_loss
    raise ValueError(f"Unknown reduction: {reduction}")
