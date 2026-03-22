from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.diffusion.protocols import ReverseSampler, SampleStep


def _initialize_tokens(
    mask_token_id: int,
    prompt_tokens: Tensor | None,
    num_new_tokens: int,
    device: str,
) -> tuple[Tensor, int]:
    if prompt_tokens is None:
        tokens = torch.full(
            (1, num_new_tokens),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        return tokens, 0

    prompt_tokens = prompt_tokens.to(device)
    prompt_len = int(prompt_tokens.shape[-1])
    tokens = torch.full(
        (1, prompt_len + num_new_tokens),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    tokens[:, :prompt_len] = prompt_tokens
    return tokens, prompt_len


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
        tokens, _ = _initialize_tokens(
            self.mask_token_id,
            prompt_tokens,
            num_new_tokens,
            self.device,
        )

        yield SampleStep(
            step_index=0,
            timestep=num_steps,
            tokens=tokens.clone(),
            newly_filled_mask=None,
            num_masked_remaining=int((tokens == self.mask_token_id).sum().item()),
            num_revealed=0,
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
                num_masked_remaining=int((tokens == self.mask_token_id).sum().item()),
                num_revealed=int(mask.sum().item()),
            )


@dataclass
class ConfidenceIterativeSampler(ReverseSampler):
    mask_token_id: int
    reveal_ratio_min: float = 0.1
    reveal_ratio_max: float = 0.35
    device: str = "cpu"

    def _reveal_count(self, remaining: int, step_index: int, total_steps: int) -> int:
        if step_index >= total_steps - 1:
            return remaining
        progress = step_index / max(total_steps - 1, 1)
        reveal_ratio = (
            self.reveal_ratio_min
            + (self.reveal_ratio_max - self.reveal_ratio_min) * progress
        )
        return max(1, min(remaining, int(round(remaining * reveal_ratio))))

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
        tokens, prompt_len = _initialize_tokens(
            self.mask_token_id,
            prompt_tokens,
            num_new_tokens,
            self.device,
        )
        generation_mask = torch.zeros_like(tokens, dtype=torch.bool)
        generation_mask[:, prompt_len:] = True

        yield SampleStep(
            step_index=0,
            timestep=num_steps,
            tokens=tokens.clone(),
            newly_filled_mask=None,
            num_masked_remaining=int((tokens == self.mask_token_id).sum().item()),
            num_revealed=0,
        )

        for step_index, timestep in enumerate(range(num_steps - 1, -1, -1), start=1):
            active_mask = (tokens == self.mask_token_id) & generation_mask
            remaining = int(active_mask.sum().item())
            if remaining == 0:
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
                probs = torch.softmax(logits, dim=-1)
                sampled = logits.argmax(dim=-1)

            confidence = probs.max(dim=-1).values
            confidence = confidence.masked_fill(~active_mask, -1.0)
            reveal_count = self._reveal_count(remaining, step_index, num_steps)

            flat_confidence = confidence.view(-1)
            topk = torch.topk(flat_confidence, k=reveal_count)
            reveal_mask = torch.zeros_like(flat_confidence, dtype=torch.bool)
            reveal_mask[topk.indices] = True
            reveal_mask = reveal_mask.view_as(tokens) & active_mask

            tokens[reveal_mask] = sampled[reveal_mask]
            yield SampleStep(
                step_index=step_index,
                timestep=timestep,
                tokens=tokens.clone(),
                newly_filled_mask=reveal_mask.clone(),
                num_masked_remaining=int((tokens == self.mask_token_id).sum().item()),
                num_revealed=int(reveal_mask.sum().item()),
            )
