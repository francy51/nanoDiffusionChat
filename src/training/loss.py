import torch.nn.functional as F
from torch import Tensor


def masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute cross-entropy loss only at masked positions.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        mask: Boolean mask where True indicates positions to compute loss
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    batch_size, seq_len, vocab_size = logits.shape

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    loss = loss * mask_flat.float()

    if reduction == "mean":
        return loss.sum() / mask_flat.sum().clamp(min=1)
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss.view(batch_size, seq_len)
