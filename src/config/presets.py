from __future__ import annotations

from src.config.schema import (
    DatasetConfig,
    DiffusionConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


def build_experiment_config(name: str) -> ExperimentConfig:
    presets = {
        "debug": ExperimentConfig(
            dataset=DatasetConfig(seq_len=64),
            model=ModelConfig(
                vocab_size=256,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                max_seq_len=64,
            ),
            diffusion=DiffusionConfig(num_steps=32, mask_token_id=1),
            training=TrainingConfig(
                batch_size=8,
                max_steps=40,
                warmup_steps=5,
                eval_interval=10,
                checkpoint_interval=20,
                log_interval=5,
            ),
            eval=EvalConfig(num_eval_batches=2, num_qualitative_samples=2),
        ),
        "tiny": ExperimentConfig(
            dataset=DatasetConfig(seq_len=128),
            model=ModelConfig(
                vocab_size=256,
                hidden_dim=256,
                num_layers=4,
                num_heads=4,
                max_seq_len=128,
            ),
            diffusion=DiffusionConfig(num_steps=64, mask_token_id=1),
            training=TrainingConfig(
                batch_size=16,
                max_steps=200,
                warmup_steps=20,
                eval_interval=50,
                checkpoint_interval=50,
                log_interval=10,
            ),
        ),
        "small": ExperimentConfig(
            dataset=DatasetConfig(seq_len=256),
            model=ModelConfig(
                vocab_size=256,
                hidden_dim=384,
                num_layers=6,
                num_heads=6,
                max_seq_len=256,
            ),
            diffusion=DiffusionConfig(num_steps=128, mask_token_id=1),
            training=TrainingConfig(
                batch_size=16,
                max_steps=500,
                warmup_steps=50,
                eval_interval=100,
                checkpoint_interval=100,
                log_interval=20,
            ),
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}")
    return presets[name]


def list_presets() -> list[str]:
    return ["debug", "tiny", "small"]
