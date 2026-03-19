import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from src.config.base import Config
    from src.models.denoiser import Denoiser
    from src.data.dataset import DiffusionDataset
    from src.training.trainer import Trainer
    from src.utils.helpers import get_device
    import torch
    from torch.utils.data import DataLoader
    import marimo as mo

    config = Config.tiny()
    device = get_device()
    print(f"Using device: {device}")
    return


@app.cell
def _(config):
    data_path = Path("data") / f"train_{config.data.seq_len}.pt"

    if not data_path.exists():
        print("Run notebook 01_data.py first to prepare data")
    else:
        dataset = DiffusionDataset(
            data_path,
            seq_len=config.data.seq_len,
            mask_token_id=config.diffusion.mask_token_id,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
        )
        print(f"Dataset size: {len(dataset)}")
        print(f"Batches: {len(loader)}")
    return


@app.cell
def _(config, data_path, device):
    val_path = Path("data") / f"val_{config.data.seq_len}.pt"

    val_dataset = (
        DiffusionDataset(
            val_path,
            seq_len=config.data.seq_len,
            mask_token_id=config.diffusion.mask_token_id,
        )
        if val_path.exists()
        else None
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )
        if val_dataset
        else None
    )

    model = Denoiser(config.model, config.diffusion.num_steps)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        device=device,
    )
    print("Trainer initialized")
    return


@app.cell
def _(mo, trainer):
    steps_input = mo.ui.number(value=100, start=1, stop=10000, label="Steps to train")
    steps_input
    return


@app.cell
def _(steps_input, trainer, loader, mo):
    import time

    steps = int(steps_input.value)
    losses = []

    batch_iter = iter(loader)

    progress = mo.ui.progress(steps)

    with progress:
        for i in range(steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)

            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

            if (i + 1) % 10 == 0:
                avg_loss = sum(losses[-10:]) / 10
                progress.set_progress(i + 1, f"Step {i + 1}, Loss: {avg_loss:.4f}")

    print(f"Training complete. Final loss: {losses[-1]:.4f}")
    return


@app.cell
def _(losses, mo):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame({"step": range(len(losses)), "loss": losses})

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="step", y="loss")
        .properties(width=600, height=300)
    )

    mo.ui.altair_chart(chart)
    return


@app.cell
def _(trainer):
    checkpoint_path = trainer.save_checkpoint()
    print(f"Checkpoint saved: {checkpoint_path}")
    return


if __name__ == "__main__":
    app.run()
