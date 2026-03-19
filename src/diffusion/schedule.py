from typing import Literal

import torch
from torch import Tensor


def get_mask_probability(
    t: Tensor,
    schedule: Literal["linear", "cosine", "uniform"] = "uniform",
) -> Tensor:
    if schedule == "linear":
        return t
    elif schedule == "cosine":
        return 0.5 * (1 - torch.cos(torch.pi * t))
    elif schedule == "uniform":
        return torch.rand_like(t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def sample_timesteps(batch_size: int, num_steps: int, device: str = "cpu") -> Tensor:
    return torch.randint(0, num_steps, (batch_size,), device=device)


def corrupt_tokens(
    tokens: Tensor,
    mask_prob: float | Tensor,
    mask_token_id: int,
) -> tuple[Tensor, Tensor]:
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
