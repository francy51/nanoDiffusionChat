from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainStepMetrics:
    step: int
    loss: float
    lr: float
    tokens_per_sec: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass
class EvalMetrics:
    step: int
    masked_loss: float
    perplexity_proxy: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)
