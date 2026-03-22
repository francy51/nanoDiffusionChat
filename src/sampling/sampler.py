from pathlib import Path

import torch
from torch import Tensor

from src.config.io import load_experiment_config
from src.config.schema import ExperimentConfig
from src.diffusion.samplers import ConfidenceIterativeSampler, FullRefreshSampler
from src.models.factory import build_model_from_experiment


class DiffusionSampler:
    def __init__(self, model, config: ExperimentConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.sampler = self._build_sampler(config.diffusion.sampler_name)

    def _build_sampler(self, sampler_name: str):
        if sampler_name == "full_refresh":
            self.sampler = FullRefreshSampler(
                mask_token_id=self.config.diffusion.mask_token_id,
                device=self.device,
            )
        else:
            self.sampler = ConfidenceIterativeSampler(
                mask_token_id=self.config.diffusion.mask_token_id,
                reveal_ratio_min=self.config.diffusion.reveal_ratio_min,
                reveal_ratio_max=self.config.diffusion.reveal_ratio_max,
                device=self.device,
            )
        return self.sampler

    def sample(
        self,
        prompt_tokens: Tensor | None = None,
        num_tokens: int = 64,
        temperature: float = 1.0,
        sampler_name: str | None = None,
    ):
        sampler = (
            self.sampler if sampler_name is None else self._build_sampler(sampler_name)
        )
        return sampler.sample(
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
