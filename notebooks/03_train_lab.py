import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from src.config.presets import list_presets
    from src.store import DatasetStore as _DatasetStore
    from src.store import RunStore as _RunStore

    dataset_options = [item.dataset_id for item in _DatasetStore().list_datasets()] or [
        "none"
    ]
    resumable_runs = [
        run
        for run in _RunStore().list_runs()
        if run.latest_checkpoint or run.best_checkpoint
    ]
    resume_run_options = [run.run_id for run in resumable_runs] or ["none"]
    preset = mo.ui.dropdown(options=list_presets(), value="debug", label="Preset")
    dataset_id = mo.ui.dropdown(
        options=dataset_options,
        value=dataset_options[0],
        label="Dataset",
    )
    training_mode = mo.ui.dropdown(
        options=["new", "resume"],
        value="new",
        label="Training mode",
    )
    resume_run_id = mo.ui.dropdown(
        options=resume_run_options,
        value=resume_run_options[0],
        label="Resume run",
    )
    checkpoint_source = mo.ui.dropdown(
        options=["latest", "best"],
        value="latest",
        label="Checkpoint",
    )
    steps = mo.ui.number(value=10, start=1, stop=1_000_000, label="Training steps")
    train_action = mo.ui.run_button(label="Start training")
    mo.vstack(
        [
            mo.md("## Training controls"),
            training_mode,
            preset,
            dataset_id,
            resume_run_id,
            checkpoint_source,
            steps,
            train_action,
        ]
    )
    return (
        checkpoint_source,
        dataset_id,
        mo,
        preset,
        resume_run_id,
        steps,
        train_action,
        training_mode,
    )


@app.cell
def _(
    checkpoint_source,
    dataset_id,
    mo,
    preset,
    resume_run_id,
    steps,
    train_action,
    training_mode,
):
    from pathlib import Path as _Path

    from torch.utils.data import DataLoader

    from src.config.io import load_experiment_config
    from src.config.presets import build_experiment_config
    from src.data.batching import collate_token_batches
    from src.data.dataset import TokenDataset
    from src.models.factory import build_model_from_experiment
    from src.store import DatasetStore as _DatasetStore
    from src.store import RunStore as _RunStore
    from src.training.loop import Trainer
    from src.utils.device import get_device
    from src.utils.seed import set_seed

    result_rows: list[dict[str, float | int | None]] = []
    summary_lines = ["Press `Start training` to begin."]

    if (
        train_action.value
        and training_mode.value == "new"
        and dataset_id.value != "none"
    ):
        dataset_store = _DatasetStore()
        run_store = _RunStore()
        config = build_experiment_config(preset.value)
        artifact = dataset_store.get(dataset_id.value)
        config.model.vocab_size = artifact.vocab_size
        config.model.max_seq_len = artifact.seq_len
        config.dataset.seq_len = artifact.seq_len
        config.diffusion.mask_token_id = artifact.mask_token_id

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
        run = run_store.create_run(
            config,
            artifact.dataset_id,
            preset_name=preset.value,
        )
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
        step_count = int(steps.value)
        progress = mo.status.progress_bar(
            range(step_count),
            title="Training run",
            subtitle=f"{run.run_id} on {artifact.dataset_id}",
            completion_title="Training complete",
            total=step_count,
        )
        for _step in progress:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            result_rows.append(trainer.train_step(batch).to_dict())

        eval_metrics = trainer.evaluate().to_dict()
        result_rows.append(eval_metrics)
        checkpoint_path = trainer.save_checkpoint(tag="last")
        summary_lines = [
            "Mode: `new`",
            f"Run: `{run.run_id}`",
            f"Dataset: `{artifact.dataset_id}`",
            f"Preset: `{preset.value}`",
            f"Steps: `{step_count}`",
            f"Eval masked loss: `{eval_metrics['masked_loss']:.4f}`",
            f"Checkpoint: `{checkpoint_path}`",
        ]
    elif (
        train_action.value
        and training_mode.value == "resume"
        and resume_run_id.value != "none"
    ):
        dataset_store = _DatasetStore()
        run_store = _RunStore()
        run = run_store.get(resume_run_id.value)
        checkpoint_map = {
            "latest": run.latest_checkpoint,
            "best": run.best_checkpoint,
        }
        checkpoint_path = checkpoint_map[checkpoint_source.value]
        if checkpoint_path is None:
            available = ", ".join(
                name for name, value in checkpoint_map.items() if value is not None
            )
            summary_lines = [
                "Run "
                f"`{run.run_id}` does not have a "
                f"`{checkpoint_source.value}` checkpoint.",
                f"Available checkpoints: `{available or 'none'}`",
            ]
        else:
            checkpoint_path = _Path(checkpoint_path)
            resume_checkpoint = checkpoint_path
            config = load_experiment_config(run.config_path)
            config.training.resume_from_checkpoint = str(resume_checkpoint)
            artifact = dataset_store.get(run.dataset_id)
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
            trainer = Trainer(
                config=config,
                model=build_model_from_experiment(config),
                train_loader=train_loader,
                val_loader=val_loader,
                device=get_device(),
                run_id=run.run_id,
                run_store=run_store,
            )
            trainer.load_checkpoint(resume_checkpoint)

            step_count = int(steps.value)
            starting_step = trainer.step
            target_step = starting_step + step_count
            train_iter = iter(train_loader)
            progress = mo.status.progress_bar(
                range(step_count),
                title="Resuming training run",
                subtitle=f"{run.run_id} from step {starting_step}",
                completion_title="Training complete",
                total=step_count,
            )
            for _step in progress:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                result_rows.append(trainer.train_step(batch).to_dict())

            eval_metrics = trainer.evaluate().to_dict()
            result_rows.append(eval_metrics)
            saved_checkpoint = trainer.save_checkpoint(tag="last")
            summary_lines = [
                "Mode: `resume`",
                f"Run: `{run.run_id}`",
                f"Dataset: `{artifact.dataset_id}`",
                f"Resumed from: `{resume_checkpoint}`",
                f"Start step: `{starting_step}`",
                f"End step: `{target_step}`",
                f"Added steps: `{step_count}`",
                f"Eval masked loss: `{eval_metrics['masked_loss']:.4f}`",
                f"Checkpoint: `{saved_checkpoint}`",
            ]
    elif (
        train_action.value
        and training_mode.value == "new"
        and dataset_id.value == "none"
    ):
        summary_lines = [
            "No prepared dataset is available. Build one in `01_data_lab.py`."
        ]
    elif train_action.value and training_mode.value == "resume":
        summary_lines = [
            "No resumable run is available. Train a model first to create a checkpoint."
        ]
    return result_rows, summary_lines


@app.cell
def _(mo, result_rows: list[dict[str, float | int | None]], summary_lines):
    summary = mo.md("\n".join(f"- {line}" for line in summary_lines))
    table = (
        mo.md("No step metrics yet.") if not result_rows else mo.ui.table(result_rows)
    )
    mo.vstack([mo.md("## Training summary"), summary, table])
    return


@app.cell
def _(mo, result_rows: list[dict[str, float | int | None]]):
    import altair as alt
    import polars as pl

    result_df = pl.DataFrame(result_rows)

    # Create an Altair chart
    _chart = (
        alt.Chart(result_df)
        .mark_point()
        .encode(
            x="step",
            y="loss",
            color="tokens_per_sec",
        )
        .properties(height=450)
    )

    # Make it reactive ⚡
    chart = mo.ui.altair_chart(_chart)
    chart  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
