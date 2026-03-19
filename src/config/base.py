from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @property
    def num_params(self) -> int:
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
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_steps: int = 100000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 500
    checkpoint_interval: int = 5000
    log_interval: int = 100
    num_workers: int = 4

    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)


@dataclass
class DiffusionConfig:
    num_steps: int = 256
    schedule: Literal["linear", "cosine", "uniform"] = "uniform"
    mask_token_id: int = 50256

    def __post_init__(self):
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")


@dataclass
class DataConfig:
    dataset_name: str = "tinystories"
    seq_len: int = 512
    train_split: float = 0.95
    data_dir: Path = field(default_factory=lambda: Path("data"))
    tokenizer_name: str = "gpt2"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def tiny(cls) -> "Config":
        return cls(
            model=ModelConfig(
                hidden_dim=512,
                num_layers=6,
                num_heads=8,
                max_seq_len=512,
            ),
            diffusion=DiffusionConfig(num_steps=256),
        )

    @classmethod
    def small(cls) -> "Config":
        return cls(
            model=ModelConfig(
                hidden_dim=768,
                num_layers=12,
                num_heads=12,
                max_seq_len=512,
            ),
            diffusion=DiffusionConfig(num_steps=256),
        )
