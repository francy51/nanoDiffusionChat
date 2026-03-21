from __future__ import annotations

from src.config.schema import ExperimentConfig, ModelConfig
from src.models.denoiser import Denoiser


def build_denoiser(config: ModelConfig, num_diffusion_steps: int) -> Denoiser:
    return Denoiser(config, num_diffusion_steps=num_diffusion_steps)


def build_model_from_experiment(config: ExperimentConfig) -> Denoiser:
    return build_denoiser(config.model, config.diffusion.num_steps)
