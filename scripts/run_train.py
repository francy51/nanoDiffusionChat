from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from src.config.presets import build_experiment_config
from src.data.batching import collate_token_batches
from src.data.dataset import TokenDataset
from src.models.factory import build_model_from_experiment
from src.store import DatasetStore, RunStore
from src.training.loop import Trainer
from src.utils.device import get_device
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    config = build_experiment_config(args.preset)
    dataset_store = DatasetStore()
    run_store = RunStore()
    artifact = dataset_store.get(args.dataset_id)
    train_loader = DataLoader(
        TokenDataset(artifact.train_path),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_token_batches,
    )
    val_loader = DataLoader(
        TokenDataset(artifact.val_path),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_token_batches,
    )
    set_seed(config.training.seed)
    run = run_store.create_run(config, artifact.dataset_id, preset_name=args.preset)
    trainer = Trainer(
        config=config,
        model=build_model_from_experiment(config),
        train_loader=train_loader,
        val_loader=val_loader,
        device=get_device(),
        run_id=run.run_id,
        run_store=run_store,
    )
    train_iter = iter(train_loader)
    for _ in range(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        print(trainer.train_step(batch))
    print(trainer.evaluate())
    print(trainer.save_checkpoint(tag="last"))


if __name__ == "__main__":
    main()
