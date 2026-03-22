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

    from src.store import RunStore

    run_options = [run.run_id for run in RunStore().list_runs()] or ["none"]
    run_id = mo.ui.dropdown(options=run_options, value=run_options[0], label="Run")
    prompt = mo.ui.text(value="Once upon a time", label="Prompt")
    num_tokens = mo.ui.number(value=32, start=1, stop=256, label="New tokens")
    temperature = mo.ui.slider(
        start=0.0,
        stop=1.5,
        step=0.1,
        value=0.8,
        label="Temperature",
    )
    action = mo.ui.run_button(label="Generate")
    mo.vstack([run_id, prompt, num_tokens, temperature, action])
    return action, num_tokens, prompt, run_id, temperature


@app.cell
def _(action, num_tokens, prompt, run_id, temperature):
    from pathlib import Path as _Path

    import marimo as _mo
    import torch

    from src.config.io import load_experiment_config
    from src.sampling.sampler import DiffusionSampler
    from src.store import RunStore as _RunStore
    from src.tokenization.tokenizer import Tokenizer

    rows = []
    if action.value and run_id.value != "none":
        run = _RunStore().get(run_id.value)
        checkpoint = run.latest_checkpoint or run.best_checkpoint
        if checkpoint is not None:
            sampler = DiffusionSampler.from_checkpoint(checkpoint)
            config = load_experiment_config(run.config_path)
            tokenizer = Tokenizer.from_file(
                _Path("artifacts")
                / "datasets"
                / config.dataset.source_name
                / "prepared"
                / run.dataset_id
                / "tokenizer.json"
            )
            prompt_tokens = torch.tensor(
                [tokenizer.encode(prompt.value)],
                dtype=torch.long,
            )
            for step in sampler.sample(
                prompt_tokens=prompt_tokens,
                num_tokens=int(num_tokens.value),
                temperature=float(temperature.value),
            ):
                rows.append(
                    {
                        "step_index": step.step_index,
                        "timestep": step.timestep,
                        "preview": tokenizer.decode(step.tokens[0].tolist())[:160],
                    }
                )

    output = (
        _mo.md("Press `Generate` to sample from the selected run.")
        if not rows
        else _mo.ui.table(rows)
    )
    output  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
