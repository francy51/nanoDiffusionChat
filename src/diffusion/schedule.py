from typing import Literal

import torch
from torch import Tensor

ScheduleName = Literal["uniform", "linear", "cosine"]


def sample_timesteps(batch_size: int, num_steps: int, device: str = "cpu") -> Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return torch.randint(0, num_steps, (batch_size,), device=device)


def normalize_timesteps(timesteps: Tensor, num_steps: int) -> Tensor:
    if num_steps <= 1:
        return torch.zeros_like(timesteps, dtype=torch.float32)
    return timesteps.float() / float(num_steps - 1)


def get_mask_probability(
    t: Tensor,
    schedule: ScheduleName = "uniform",
) -> Tensor:
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 0.5 * (1 - torch.cos(torch.pi * t))
    if schedule == "uniform":
        return torch.rand_like(t)
    raise ValueError(f"Unknown schedule: {schedule}")
