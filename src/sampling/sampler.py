from pathlib import Path

import torch
from torch import Tensor

from src.config.io import load_experiment_config
from src.config.schema import ExperimentConfig
from src.diffusion.samplers import FullRefreshSampler
from src.models.factory import build_model_from_experiment


class DiffusionSampler:
    def __init__(self, model, config: ExperimentConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.sampler = FullRefreshSampler(
            mask_token_id=config.diffusion.mask_token_id,
            device=device,
        )

    def sample(
        self,
        prompt_tokens: Tensor | None = None,
        num_tokens: int = 64,
        temperature: float = 1.0,
    ):
        return self.sampler.sample(
            self.model,
            prompt_tokens=prompt_tokens,
            num_new_tokens=num_tokens,
            temperature=temperature,
            num_steps=self.config.diffusion.num_steps,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cpu",
    ) -> "DiffusionSampler":
        run_dir = checkpoint_path.parent.parent
        config = load_experiment_config(run_dir / "config.json")
        model = build_model_from_experiment(config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, config, device)
