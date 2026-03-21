import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    from torch.utils.data import DataLoader

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from src.config.io import load_experiment_config
    from src.data.batching import collate_token_batches
    from src.data.dataset import TokenDataset
    from src.eval.perplexity import compute_perplexity_proxy
    from src.models.factory import build_model_from_experiment
    from src.store import DatasetStore, RunStore
    from src.training.checkpoint import load_checkpoint
    from src.utils.device import get_device

    run_store = RunStore()
    dataset_store = DatasetStore()
    run_options = [run.run_id for run in run_store.list_runs()] or ["none"]
    run_id = mo.ui.dropdown(options=run_options, value=run_options[0], label="Run")
    action = mo.ui.run_button(label="Evaluate")
    rows = []
    if action.value and run_id.value != "none":
        run = run_store.get(run_id.value)
        checkpoint = run.latest_checkpoint or run.best_checkpoint
        if checkpoint is not None:
            config = load_experiment_config(run.config_path)
            model = build_model_from_experiment(config)
            checkpoint_data = load_checkpoint(checkpoint, get_device())
            model.load_state_dict(checkpoint_data["model_state_dict"])
            artifact = dataset_store.get(run.dataset_id)
            loader = DataLoader(
                TokenDataset(artifact.val_path),
                batch_size=config.training.batch_size,
                shuffle=False,
                collate_fn=collate_token_batches,
            )
            batch = next(iter(loader))
            rows.append(
                {
                    "run_id": run.run_id,
                    "perplexity_proxy": compute_perplexity_proxy(
                        model, batch, config, device=get_device()
                    ),
                    "latest_checkpoint": str(checkpoint),
                }
            )
    mo.vstack([run_id, action, mo.ui.table(rows)])
    return


if __name__ == "__main__":
    app.run()
