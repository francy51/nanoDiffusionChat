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

    from src.eval.compare import summarize_run
    from src.store import RunStore
    from src.utils.serialization import load_jsonl

    run_store = RunStore()
    summaries = [summarize_run(run.run_dir) for run in run_store.list_runs()]
    metric_rows = []
    for run in run_store.list_runs()[:5]:
        metric_rows.extend(load_jsonl(run.run_dir / "metrics" / "train.jsonl")[-5:])
    mo.vstack(
        [
            mo.md("# Runs dashboard"),
            mo.ui.table(summaries),
            mo.md("## Recent metrics"),
            mo.ui.table(metric_rows),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
