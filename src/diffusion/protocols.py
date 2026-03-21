from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor, nn


class CorruptionPolicy(Protocol):
    def corrupt(self, tokens: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclass
class SampleStep:
    step_index: int
    timestep: int
    tokens: Tensor
    newly_filled_mask: Tensor | None


class ReverseSampler(Protocol):
    def sample(
        self,
        model: nn.Module,
        *,
        prompt_tokens: Tensor | None,
        num_new_tokens: int,
        temperature: float,
        num_steps: int,
    ) -> Iterator[SampleStep]: ...
