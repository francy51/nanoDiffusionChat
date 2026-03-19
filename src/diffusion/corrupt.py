import torch
from torch import Tensor


def corrupt_tokens(
    tokens: Tensor,
    mask_prob: Tensor | float,
    mask_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Apply masking corruption to tokens.

    Args:
        tokens: Input token IDs [batch, seq_len]
        mask_prob: Probability of masking each token (scalar or per-sample)
        mask_token_id: ID of the mask token

    Returns:
        Tuple of (corrupted_tokens, mask) where mask is True at masked positions
    """
    assert tokens.dim() == 2, f"Expected 2D tensor, got {tokens.dim()}D"

    batch_size, seq_len = tokens.shape

    if isinstance(mask_prob, float):
        mask_prob_tensor = torch.full((batch_size,), mask_prob, device=tokens.device)
    else:
        mask_prob_tensor = mask_prob

    mask = torch.rand(
        batch_size, seq_len, device=tokens.device
    ) < mask_prob_tensor.unsqueeze(1)

    corrupted = tokens.clone()
    corrupted[mask] = mask_token_id

    return corrupted, mask
