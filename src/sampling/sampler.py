from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from src.config.base import Config
from src.models.denoiser import Denoiser


class DiffusionSampler:
    def __init__(
        self,
        model: Denoiser,
        config: Config,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        prompt_tokens: Tensor | None = None,
        num_tokens: int = 64,
        temperature: float = 1.0,
    ) -> Iterator[tuple[Tensor, int]]:
        """Generate tokens using iterative denoising.

        Yields (current_tokens, step) at each denoising step.
        """
        if prompt_tokens is None:
            tokens = torch.full(
                (1, num_tokens),
                self.config.diffusion.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
        else:
            prompt_len = prompt_tokens.shape[-1]
            tokens = torch.full(
                (1, prompt_len + num_tokens),
                self.config.diffusion.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            tokens[:, :prompt_len] = prompt_tokens.to(self.device)

        yield tokens.clone(), 0

        for step in range(self.config.diffusion.num_steps - 1, -1, -1):
            t = torch.tensor([step], device=self.device)

            mask = tokens == self.config.diffusion.mask_token_id
            if not mask.any():
                break

            logits = self.model(tokens, t)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                new_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(
                    tokens.shape
                )
            else:
                new_tokens = logits.argmax(dim=-1)

            tokens[mask] = new_tokens[mask]

            yield tokens.clone(), self.config.diffusion.num_steps - step

        yield tokens, self.config.diffusion.num_steps

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: Config,
        device: str = "cpu",
    ) -> "DiffusionSampler":
        model = Denoiser(config.model, config.diffusion.num_steps)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, config, device)
