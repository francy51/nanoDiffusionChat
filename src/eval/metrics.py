import torch
from torch import Tensor

from src.models.denoiser import Denoiser


@torch.no_grad()
def compute_perplexity(
    model: Denoiser,
    tokens: Tensor,
    num_steps: int = 256,
    mask_token_id: int = 50256,
    device: str = "cpu",
) -> float:
    """Compute perplexity using the diffusion model.

    Uses uniform masking schedule and computes loss over all positions.
    """
    model.eval()
    model = model.to(device)
    tokens = tokens.to(device)

    total_loss = 0.0
    num_samples = 0

    for i in range(tokens.shape[0]):
        batch = tokens[i : i + 1]

        t = torch.rand(1, device=device)
        mask_prob = t

        mask = torch.rand_like(batch.float()) < mask_prob
        corrupted = batch.clone()
        corrupted[mask] = mask_token_id

        timestep = (t * num_steps).long()
        logits = model(corrupted, timestep)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.view(-1),
            reduction="none",
        )

        loss = loss.view(batch.shape)
        loss = loss[mask].mean()

        total_loss += loss.item()
        num_samples += 1

    avg_loss = total_loss / max(num_samples, 1)
    return torch.exp(torch.tensor(avg_loss)).item()
