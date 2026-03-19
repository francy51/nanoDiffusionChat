import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from src.config.base import Config
    from src.models.denoiser import Denoiser
    from src.eval.metrics import compute_perplexity
    from src.data.dataset import DiffusionDataset
    from src.utils.helpers import get_device, load_json_log
    import torch
    import marimo as mo

    config = Config.tiny()
    device = get_device()
    print(f"Device: {device}")
    return


@app.cell
def _(config, device):
    checkpoints = list(Path("checkpoints").glob("*.pt"))

    if not checkpoints:
        print("No checkpoints found. Run notebook 03_train.py first.")
    else:
        latest = sorted(checkpoints)[-1]
        print(f"Loading: {latest}")

        model = Denoiser(config.model, config.diffusion.num_steps)
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        print(f"Checkpoint step: {checkpoint['step']}")
    return


@app.cell
def _(config):
    val_path = Path("data") / f"val_{config.data.seq_len}.pt"

    if val_path.exists():
        val_dataset = DiffusionDataset(
            val_path,
            seq_len=config.data.seq_len,
            mask_token_id=config.diffusion.mask_token_id,
        )

        val_tokens = torch.stack(
            [val_dataset[i] for i in range(min(100, len(val_dataset)))]
        )
        print(f"Validation samples: {val_tokens.shape}")
    else:
        print("Validation data not found")
    return


@app.cell
def _(model, val_tokens, device):
    ppl = compute_perplexity(model, val_tokens[:10], device=device)
    print(f"Perplexity: {ppl:.2f}")
    return


@app.cell
def _():
    log_path = Path("logs") / "training_log.jsonl"

    if log_path.exists():
        logs = load_json_log(log_path)
        print(f"Log entries: {len(logs)}")

        if logs:
            import pandas as pd
            import altair as alt

            df = pd.DataFrame(logs)

            chart = (
                alt.Chart(df)
                .mark_line()
                .encode(x="step", y="loss")
                .properties(width=600, height=300, title="Training Loss")
            )

            mo.ui.altair_chart(chart)
    else:
        print("No training logs found")
    return


if __name__ == "__main__":
    app.run()
