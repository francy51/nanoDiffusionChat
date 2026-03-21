from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.diffusion.protocols import ReverseSampler, SampleStep


@dataclass
class FullRefreshSampler(ReverseSampler):
    mask_token_id: int
    device: str = "cpu"

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        *,
        prompt_tokens: Tensor | None,
        num_new_tokens: int,
        temperature: float,
        num_steps: int,
    ) -> Iterator[SampleStep]:
        if prompt_tokens is None:
            tokens = torch.full(
                (1, num_new_tokens),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
        else:
            prompt_tokens = prompt_tokens.to(self.device)
            prompt_len = prompt_tokens.shape[-1]
            tokens = torch.full(
                (1, prompt_len + num_new_tokens),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            tokens[:, :prompt_len] = prompt_tokens

        yield SampleStep(
            step_index=0,
            timestep=num_steps,
            tokens=tokens.clone(),
            newly_filled_mask=None,
        )

        for step_index, timestep in enumerate(range(num_steps - 1, -1, -1), start=1):
            mask = tokens == self.mask_token_id
            if not mask.any():
                break
            timestep_tensor = torch.full(
                (tokens.shape[0],),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            logits = model(tokens, timestep_tensor)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(
                    tokens.shape
                )
            else:
                sampled = logits.argmax(dim=-1)
            tokens[mask] = sampled[mask]
            yield SampleStep(
                step_index=step_index,
                timestep=timestep,
                tokens=tokens.clone(),
                newly_filled_mask=mask.clone(),
            )
