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

    from src.config.presets import build_experiment_config, list_presets
    from src.data.batching import collate_token_batches
    from src.data.dataset import TokenDataset
    from src.models.factory import build_model_from_experiment
    from src.store import DatasetStore, RunStore
    from src.training.loop import Trainer
    from src.utils.device import get_device
    from src.utils.seed import set_seed

    dataset_store = DatasetStore()
    run_store = RunStore()
    dataset_options = [item.dataset_id for item in dataset_store.list_datasets()] or [
        "none"
    ]
    preset = mo.ui.dropdown(options=list_presets(), value="debug", label="Preset")
    dataset_id = mo.ui.dropdown(
        options=dataset_options,
        value=dataset_options[0],
        label="Dataset",
    )
    steps = mo.ui.number(value=10, start=1, stop=5000, label="Steps")
    create_and_train = mo.ui.run_button(label="Create run and train")

    result_rows = []
    if create_and_train.value and dataset_id.value != "none":
        config = build_experiment_config(preset.value)
        artifact = dataset_store.get(dataset_id.value)
        config.model.vocab_size = artifact.vocab_size
        config.model.max_seq_len = artifact.seq_len
        config.dataset.seq_len = artifact.seq_len
        config.diffusion.mask_token_id = artifact.mask_token_id
        dataset = TokenDataset(artifact.train_path)
        val_dataset = TokenDataset(artifact.val_path)
        train_loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_token_batches,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_token_batches,
        )
        set_seed(config.training.seed)
        model = build_model_from_experiment(config)
        run = run_store.create_run(
            config,
            artifact.dataset_id,
            preset_name=preset.value,
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=get_device(),
            run_id=run.run_id,
            run_store=run_store,
        )
        train_iter = iter(train_loader)
        for _ in range(int(steps.value)):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            metrics = trainer.train_step(batch)
            result_rows.append(metrics.to_dict())
        eval_metrics = trainer.evaluate()
        trainer.save_checkpoint(tag="last")
        result_rows.append(eval_metrics.to_dict())

    mo.vstack(
        [
            preset,
            dataset_id,
            steps,
            create_and_train,
            mo.ui.table(result_rows),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
