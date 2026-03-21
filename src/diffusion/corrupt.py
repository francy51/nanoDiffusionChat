from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.diffusion.protocols import CorruptionPolicy
from src.diffusion.schedule import (
    ScheduleName,
    get_mask_probability,
    normalize_timesteps,
)


@dataclass
class MaskedDiscreteCorruptionPolicy(CorruptionPolicy):
    num_steps: int
    schedule_name: ScheduleName
    mask_token_id: int

    def corrupt(self, tokens: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.dim() != 2:
            raise ValueError(f"Expected 2D tokens tensor, got {tokens.dim()}D")
        if timesteps.dim() != 1:
            raise ValueError(f"Expected 1D timesteps tensor, got {timesteps.dim()}D")
        if timesteps.shape[0] != tokens.shape[0]:
            raise ValueError(
                "timesteps length must match batch size, got "
                f"{timesteps.shape[0]} and {tokens.shape[0]}"
            )
        mask_prob = get_mask_probability(
            normalize_timesteps(timesteps, self.num_steps),
            self.schedule_name,
        )
        mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob.unsqueeze(1)
        corrupted = tokens.clone()
        corrupted[mask] = self.mask_token_id
        return corrupted, mask


def corrupt_tokens(
    tokens: Tensor,
    mask_prob: Tensor | float,
    mask_token_id: int,
) -> tuple[Tensor, Tensor]:
    if tokens.dim() != 2:
        raise ValueError(f"Expected 2D tokens tensor, got {tokens.dim()}D")
    if isinstance(mask_prob, float):
        mask_prob_tensor = torch.full(
            (tokens.shape[0],), mask_prob, device=tokens.device
        )
    else:
        mask_prob_tensor = mask_prob.to(tokens.device)
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob_tensor.unsqueeze(
        1
    )
    corrupted = tokens.clone()
    corrupted[mask] = mask_token_id
    return corrupted, mask
