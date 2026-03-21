from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config.schema import (
    DatasetConfig,
    DiffusionConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from src.utils.serialization import load_json, save_json


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(**payload["dataset"]),
        model=ModelConfig(**payload["model"]),
        diffusion=DiffusionConfig(**payload["diffusion"]),
        training=TrainingConfig(**payload["training"]),
        eval=EvalConfig(**payload["eval"]),
    )


def save_experiment_config(config: ExperimentConfig, path: Path) -> None:
    save_json(path, experiment_config_to_dict(config))


def load_experiment_config(path: Path) -> ExperimentConfig:
    return experiment_config_from_dict(load_json(path))
