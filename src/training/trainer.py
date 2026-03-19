import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.base import Config
from src.diffusion.corrupt import corrupt_tokens
from src.diffusion.schedule import get_mask_probability, sample_timesteps
from src.models.denoiser import Denoiser
from src.training.loss import masked_cross_entropy


class Trainer:
    def __init__(
        self,
        config: Config,
        model: Denoiser,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cpu",
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = self._create_scheduler(config.training.warmup_steps)

        self.step = 0
        self.best_val_loss = float("inf")

        self.log_file: Path | None = None
        self.logs: list[dict[str, Any]] = []

    def _create_scheduler(self, warmup_steps: int) -> LambdaLR:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Tensor) -> dict[str, float]:
        self.model.train()
        batch = batch.to(self.device)

        timesteps = sample_timesteps(
            batch.shape[0],
            self.config.diffusion.num_steps,
            device=self.device,
        )

        t_normalized = timesteps.float() / self.config.diffusion.num_steps
        mask_prob = get_mask_probability(t_normalized, self.config.diffusion.schedule)

        corrupted, mask = corrupt_tokens(
            batch, mask_prob, self.config.diffusion.mask_token_id
        )

        logits = self.model(corrupted, timesteps)

        loss = masked_cross_entropy(logits, batch, mask)

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        return {
            "loss": float(loss.item()),
            "lr": float(self.scheduler.get_last_lr()[0]),
            "step": self.step,
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            timesteps = sample_timesteps(
                batch.shape[0],
                self.config.diffusion.num_steps,
                device=self.device,
            )

            t_normalized = timesteps.float() / self.config.diffusion.num_steps
            mask_prob = get_mask_probability(
                t_normalized, self.config.diffusion.schedule
            )

            corrupted, mask = corrupt_tokens(
                batch, mask_prob, self.config.diffusion.mask_token_id
            )

            logits = self.model(corrupted, timesteps)
            loss = masked_cross_entropy(logits, batch, mask)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: Path | None = None) -> Path:
        if path is None:
            self.config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.training.checkpoint_dir / f"step_{self.step}.pt"

        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": asdict(self.config),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

        return path

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    def log(self, metrics: dict[str, Any]) -> None:
        if self.log_file is None:
            self.config.training.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.config.training.log_dir / "training_log.jsonl"

        self.logs.append(metrics)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
