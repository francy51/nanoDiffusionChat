from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from src.models.denoiser import Denoiser


def sample_step(
    model: "Denoiser",
    tokens: Tensor,
    timestep: int,
    num_steps: int,
    temperature: float = 1.0,
    mask_token_id: int = 50256,
) -> Tensor:
    """Perform one denoising step.

    Args:
        model: The denoising model
        tokens: Current token sequence [batch, seq_len]
        timestep: Current diffusion timestep
        num_steps: Total number of diffusion steps
        temperature: Sampling temperature
        mask_token_id: ID of the mask token

    Returns:
        Updated token sequence with some tokens unmasked
    """
    t_tensor = torch.full(
        (tokens.shape[0],), timestep, device=tokens.device, dtype=torch.long
    )

    with torch.no_grad():
        logits = model(tokens, t_tensor)

    mask = tokens == mask_token_id

    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        new_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(
            tokens.shape
        )
    else:
        new_tokens = logits.argmax(dim=-1)

    result = tokens.clone()
    result[mask] = new_tokens[mask]

    return result
