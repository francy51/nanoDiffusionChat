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

    source_name = mo.ui.dropdown(
        options=["tinystories"],
        value="tinystories",
        label="Source",
    )
    tokenizer_name = mo.ui.dropdown(
        options=["char", "gpt2"],
        value="char",
        label="Tokenizer",
    )
    seq_len = mo.ui.number(value=128, start=32, stop=2048, label="Sequence length")
    train_split = mo.ui.slider(
        start=0.5,
        stop=0.99,
        step=0.01,
        value=0.9,
        label="Train split",
    )
    mo.vstack([source_name, tokenizer_name, seq_len, train_split])
    return mo, seq_len, source_name, tokenizer_name, train_split


@app.cell
def _(mo, seq_len, source_name, tokenizer_name, train_split):
    from src.config.schema import DatasetConfig

    config = DatasetConfig(
        source_name=source_name.value,
        tokenizer_name=tokenizer_name.value,
        seq_len=int(seq_len.value),
        train_split=float(train_split.value),
        val_split=round(1 - float(train_split.value), 2),
    )
    config_preview = mo.md(
        f"""
        ## Prepared dataset configuration
        - Dataset ID: `{config.source_name}_seq{config.seq_len}_{config.tokenizer_name}`
        - Train split: `{config.train_split:.2f}`
        - Val split: `{config.val_split:.2f}`
        """
    )
    config_preview
    return (config,)


@app.cell
def _(mo):
    from src.store import DatasetStore

    action = mo.ui.run_button(label="Prepare dataset")
    action
    return (action,)


@app.cell
def _(action, config, dataset_store, mo):
    dataset = dataset_store.create_prepared_dataset(config) if action.value else None
    content = (
        mo.md("Press `Prepare dataset` to build an artifact.")
        if dataset is None
        else mo.vstack(
            [
                mo.md("## Prepared artifact"),
                mo.ui.table(
                    [
                        {
                            "dataset_id": dataset.dataset_id,
                            "train_path": str(dataset.train_path),
                            "val_path": str(dataset.val_path),
                            "stats_path": str(dataset.stats_path),
                        }
                    ]
                ),
            ]
        )
    )
    content
    return


if __name__ == "__main__":
    app.run()
