from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    source_name: str = "tinystories"
    tokenizer_name: str = "char"
    seq_len: int = 256
    format_name: Literal["plain_text", "chat_transcript"] = "plain_text"
    chat_template_name: str | None = None
    train_split: float = 0.9
    val_split: float = 0.1
    text_column: str | None = None
    shard_size: int | None = None

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if not 0 < self.train_split < 1:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if not 0 <= self.val_split < 1:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if self.train_split + self.val_split > 1:
            raise ValueError(
                "train_split + val_split must be <= 1, got "
                f"{self.train_split + self.val_split}"
            )


@dataclass
class ModelConfig:
    vocab_size: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

    @property
    def num_params_estimate(self) -> int:
        return (
            self.vocab_size * self.hidden_dim
            + self.max_seq_len * self.hidden_dim
            + self.num_layers
            * (
                4 * self.hidden_dim * self.hidden_dim
                + 2 * self.hidden_dim * self.hidden_dim
                + 4 * self.hidden_dim
            )
            + self.vocab_size * self.hidden_dim
        )


@dataclass
class DiffusionConfig:
    algorithm: Literal["masked_discrete"] = "masked_discrete"
    num_steps: int = 128
    schedule_name: Literal["uniform", "linear", "cosine"] = "uniform"
    mask_token_id: int = 1
    objective_name: Literal["masked_ce"] = "masked_ce"
    sampler_name: Literal["full_refresh", "confidence_iterative"] = (
        "confidence_iterative"
    )
    reveal_ratio_min: float = 0.1
    reveal_ratio_max: float = 0.35

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.mask_token_id < 0:
            raise ValueError(
                f"mask_token_id must be non-negative, got {self.mask_token_id}"
            )
        if not 0 < self.reveal_ratio_min <= 1:
            raise ValueError(
                f"reveal_ratio_min must be in (0, 1], got {self.reveal_ratio_min}"
            )
        if not 0 < self.reveal_ratio_max <= 1:
            raise ValueError(
                f"reveal_ratio_max must be in (0, 1], got {self.reveal_ratio_max}"
            )
        if self.reveal_ratio_min > self.reveal_ratio_max:
            raise ValueError(
                "reveal_ratio_min must be <= reveal_ratio_max, got "
                f"{self.reveal_ratio_min} > {self.reveal_ratio_max}"
            )


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: int = 500
    resume_from_checkpoint: str | None = None
    fine_tune_learning_rate: float | None = None
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 100
    checkpoint_interval: int = 100
    log_interval: int = 10
    num_workers: int = 0
    seed: int = 7
    freeze_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if (
            self.fine_tune_learning_rate is not None
            and self.fine_tune_learning_rate <= 0
        ):
            raise ValueError(
                "fine_tune_learning_rate must be positive, got "
                f"{self.fine_tune_learning_rate}"
            )


@dataclass
class EvalConfig:
    num_eval_batches: int = 4
    num_qualitative_samples: int = 4
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.8, 1.0])

    def __post_init__(self) -> None:
        if self.num_eval_batches <= 0:
            raise ValueError(
                f"num_eval_batches must be positive, got {self.num_eval_batches}"
            )
        if self.num_qualitative_samples <= 0:
            raise ValueError(
                "num_qualitative_samples must be positive, got "
                f"{self.num_qualitative_samples}"
            )
        if not self.temperatures:
            raise ValueError("temperatures must not be empty")


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        if self.dataset.seq_len > self.model.max_seq_len:
            raise ValueError(
                "dataset.seq_len must be <= model.max_seq_len, got "
                f"{self.dataset.seq_len} > {self.model.max_seq_len}"
            )
        if self.diffusion.mask_token_id >= self.model.vocab_size:
            raise ValueError(
                "mask_token_id must be < vocab_size, got "
                f"{self.diffusion.mask_token_id} >= {self.model.vocab_size}"
            )
