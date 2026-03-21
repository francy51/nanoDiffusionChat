from __future__ import annotations

import math
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.schema import ExperimentConfig
from src.diffusion.corrupt import MaskedDiscreteCorruptionPolicy
from src.diffusion.objectives import masked_cross_entropy
from src.diffusion.schedule import sample_timesteps
from src.models.denoiser import Denoiser
from src.store.run_store import RunStore
from src.training.checkpoint import (
    load_checkpoint,
    restore_optimizer_state,
    save_checkpoint,
)
from src.training.metrics import EvalMetrics, TrainStepMetrics


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model: Denoiser,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cpu",
        run_id: str | None = None,
        run_store: RunStore | None = None,
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_id = run_id
        self.run_store = run_store

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = self._create_scheduler(config.training.warmup_steps)
        self.corruption_policy = MaskedDiscreteCorruptionPolicy(
            num_steps=config.diffusion.num_steps,
            schedule_name=config.diffusion.schedule_name,
            mask_token_id=config.diffusion.mask_token_id,
        )
        self.step = 0
        self.best_val_loss = float("inf")

    def _create_scheduler(self, warmup_steps: int) -> LambdaLR:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Tensor) -> TrainStepMetrics:
        self.model.train()
        batch = batch.to(self.device)
        start = time.perf_counter()
        timesteps = sample_timesteps(
            batch.shape[0],
            self.config.diffusion.num_steps,
            device=self.device,
        )
        corrupted, mask = self.corruption_policy.corrupt(batch, timesteps)
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

        elapsed = max(time.perf_counter() - start, 1e-8)
        tokens_per_sec = float(batch.numel() / elapsed)
        metrics = TrainStepMetrics(
            step=self.step,
            loss=float(loss.item()),
            lr=float(self.scheduler.get_last_lr()[0]),
            tokens_per_sec=tokens_per_sec,
        )
        if self.run_id and self.run_store:
            self.run_store.append_metric(self.run_id, "train", metrics.to_dict())
        return metrics

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Tensor] | None = None) -> EvalMetrics:
        self.model.eval()
        if batches is None:
            if self.val_loader is None:
                raise ValueError("No validation batches available")
            batches = self.val_loader
        total_loss = 0.0
        count = 0
        for batch in batches:
            batch = batch.to(self.device)
            timesteps = sample_timesteps(
                batch.shape[0],
                self.config.diffusion.num_steps,
                device=self.device,
            )
            corrupted, mask = self.corruption_policy.corrupt(batch, timesteps)
            logits = self.model(corrupted, timesteps)
            loss = masked_cross_entropy(logits, batch, mask)
            total_loss += float(loss.item())
            count += 1
            if count >= self.config.eval.num_eval_batches:
                break
        mean_loss = total_loss / max(1, count)
        perplexity_proxy = float(math.exp(mean_loss)) if mean_loss < 20 else None
        metrics = EvalMetrics(
            step=self.step,
            masked_loss=mean_loss,
            perplexity_proxy=perplexity_proxy,
        )
        if self.run_id and self.run_store:
            self.run_store.append_metric(self.run_id, "eval", metrics.to_dict())
            if mean_loss < self.best_val_loss:
                self.best_val_loss = mean_loss
        return metrics

    def save_checkpoint(self, tag: str | None = None) -> Path:
        if not self.run_id or not self.run_store:
            raise ValueError(
                "Trainer requires a run_store and run_id to save checkpoints"
            )
        filename = f"step_{self.step:06d}.pt" if tag is None else f"{tag}.pt"
        path = self.run_store.checkpoint_path(self.run_id, filename)
        save_checkpoint(
            path,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            config=self.config,
            step=self.step,
            best_val_loss=self.best_val_loss,
        )
        last_path = self.run_store.checkpoint_path(self.run_id, "last.pt")
        save_checkpoint(
            last_path,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            config=self.config,
            step=self.step,
            best_val_loss=self.best_val_loss,
        )
        best_path = None
        if self.best_val_loss < float("inf") and tag == "best":
            best_path = path
        self.run_store.update_status(
            self.run_id,
            {
                "state": "checkpointed",
                "step": self.step,
                "latest_checkpoint": str(path),
                "best_checkpoint": str(best_path) if best_path else None,
            },
        )
        return path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.step, self.best_val_loss = restore_optimizer_state(
            checkpoint, self.optimizer, self.scheduler
        )
