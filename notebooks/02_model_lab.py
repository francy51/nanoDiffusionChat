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

    preset = mo.ui.dropdown(options=list_presets(), value="tiny", label="Preset")
    preset
    return mo, preset


@app.cell
def _(mo, preset):
    import torch



    from src.config.presets import build_experiment_config
    from src.models.factory import build_model_from_experiment

    config = build_experiment_config(preset.value)
    model = build_model_from_experiment(config)
    sample_len = min(config.dataset.seq_len, 32)
    tokens = torch.randint(0, config.model.vocab_size, (2, sample_len))
    timesteps = torch.randint(0, config.diffusion.num_steps, (2,))
    logits = model(tokens, timesteps)
    mo.ui.table(
        [
            {
                "preset": preset.value,
                "hidden_dim": config.model.hidden_dim,
                "layers": config.model.num_layers,
                "heads": config.model.num_heads,
                "vocab_size": config.model.vocab_size,
                "estimated_params": config.model.num_params_estimate,
                "forward_shape": tuple(logits.shape),
            }
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
