from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.config.io import experiment_config_from_dict, experiment_config_to_dict
from src.config.schema import ExperimentConfig


def save_checkpoint(
    path: Path,
    *,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    config: ExperimentConfig,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "config": experiment_config_to_dict(config),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def validate_checkpoint_config(path: Path, config: ExperimentConfig) -> None:
    checkpoint = load_checkpoint(path, device="cpu")
    checkpoint_config = experiment_config_from_dict(checkpoint["config"])
    if checkpoint_config != config:
        raise ValueError(
            f"Checkpoint config at {path} does not match the current run config"
        )


def restore_optimizer_state(
    checkpoint: dict[str, Any],
    optimizer: Optimizer,
    scheduler: LambdaLR,
) -> tuple[int, float]:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["step"]), float(checkpoint.get("best_val_loss", float("inf")))
