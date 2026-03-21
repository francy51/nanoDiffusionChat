import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import torch

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from src.diffusion.corrupt import MaskedDiscreteCorruptionPolicy

    policy = MaskedDiscreteCorruptionPolicy(
        num_steps=16,
        schedule_name="linear",
        mask_token_id=1,
    )
    tokens = torch.arange(32).view(1, 32)
    timesteps = torch.tensor([8])
    corrupted, mask = policy.corrupt(tokens, timesteps)
    rows = [
        {"clean": int(clean), "corrupted": int(noisy), "masked": bool(flag)}
        for clean, noisy, flag in zip(
            tokens[0].tolist(),
            corrupted[0].tolist(),
            mask[0].tolist(),
            strict=True,
        )
    ]
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
