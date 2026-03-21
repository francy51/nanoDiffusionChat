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

    from src.diffusion.schedule import get_mask_probability

    schedule = mo.ui.dropdown(
        options=["uniform", "linear", "cosine"],
        value="linear",
        label="Schedule",
    )
    t = torch.linspace(0.0, 1.0, 8)
    rows = [
        {"t": float(value), "mask_prob": float(prob)}
        for value, prob in zip(
            t.tolist(),
            get_mask_probability(t, schedule.value).tolist(),
            strict=True,
        )
    ]
    mo.vstack([schedule, mo.ui.table(rows)])
    return


if __name__ == "__main__":
    app.run()
