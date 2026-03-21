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

    from src.config.presets import build_experiment_config
    from src.diffusion.samplers import FullRefreshSampler
    from src.models.factory import build_model_from_experiment

    config = build_experiment_config("debug")
    model = build_model_from_experiment(config)
    sampler = FullRefreshSampler(mask_token_id=config.diffusion.mask_token_id)
    rows = [
        {
            "step_index": step.step_index,
            "timestep": step.timestep,
            "shape": tuple(step.tokens.shape),
        }
        for step in sampler.sample(
            model,
            prompt_tokens=None,
            num_new_tokens=16,
            temperature=0.0,
            num_steps=config.diffusion.num_steps,
        )
    ]
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
