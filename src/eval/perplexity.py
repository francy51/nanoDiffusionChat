from __future__ import annotations

import math

from torch import Tensor

from src.config.schema import ExperimentConfig
from src.diffusion.corrupt import MaskedDiscreteCorruptionPolicy
from src.diffusion.objectives import masked_cross_entropy
from src.diffusion.schedule import sample_timesteps
from src.models.denoiser import Denoiser


def compute_masked_reconstruction_ppl(
    model: Denoiser,
    tokens: Tensor,
    config: ExperimentConfig,
    device: str = "cpu",
) -> float:
    model.eval()
    policy = MaskedDiscreteCorruptionPolicy(
        num_steps=config.diffusion.num_steps,
        schedule_name=config.diffusion.schedule_name,
        mask_token_id=config.diffusion.mask_token_id,
    )
    tokens = tokens.to(device)
    timesteps = sample_timesteps(
        tokens.shape[0],
        config.diffusion.num_steps,
        device=device,
    )
    corrupted, mask = policy.corrupt(tokens, timesteps)
    logits = model.to(device)(corrupted, timesteps)
    loss = masked_cross_entropy(logits, tokens, mask)
    return float(math.exp(float(loss.item())))


def compute_perplexity_proxy(
    model: Denoiser,
    tokens: Tensor,
    config: ExperimentConfig,
    device: str = "cpu",
) -> float:
    return compute_masked_reconstruction_ppl(model, tokens, config, device)
