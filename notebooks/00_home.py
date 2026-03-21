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

    from src.store import DatasetStore, RunStore
    from src.store.paths import ensure_store_roots
    from src.utils.device import get_device

    ensure_store_roots()
    dataset_store = DatasetStore()
    run_store = RunStore()
    device = get_device()
    datasets = dataset_store.list_datasets()
    runs = run_store.list_runs()

    status = mo.md(
        f"""
        # nanoDiffusionChat

        - Device: `{device}`
        - Dataset artifacts: `{len(datasets)}`
        - Runs: `{len(runs)}`
        - Artifact root: `artifacts/`
        """
    )
    recent_runs = mo.ui.table(
        [
            {
                "run_id": run.run_id,
                "config_path": str(run.config_path),
                "latest_checkpoint": str(run.latest_checkpoint)
                if run.latest_checkpoint
                else "none",
            }
            for run in runs[:10]
        ]
    )
    recent_datasets = mo.ui.table(
        [
            {
                "dataset_id": dataset.dataset_id,
                "source_name": dataset.source_name,
                "seq_len": dataset.seq_len,
                "train_path": str(dataset.train_path),
            }
            for dataset in datasets[:10]
        ]
    )
    mo.vstack(
        [
            status,
            mo.md("## Recent datasets"),
            recent_datasets,
            mo.md("## Recent runs"),
            recent_runs,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
