import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from src.config.base import Config
    from src.models.denoiser import Denoiser
    from src.sampling.sampler import DiffusionSampler
    from src.tokenization.tokenizer import Tokenizer
    from src.utils.helpers import get_device
    import torch
    import marimo as mo

    config = Config.tiny()
    device = get_device()
    tokenizer = Tokenizer("gpt2")
    print(f"Device: {device}")
    return


@app.cell
def _(config, device):
    checkpoints = list(Path("checkpoints").glob("*.pt"))

    if not checkpoints:
        print("No checkpoints found. Run notebook 03_train.py first.")
        checkpoint_path = None
    else:
        checkpoint_path = sorted(checkpoints)[-1]
        print(f"Using checkpoint: {checkpoint_path}")
    return


@app.cell
def _(checkpoint_path, config, device):
    model = Denoiser(config.model, config.diffusion.num_steps)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        print("Model loaded")

    sampler = DiffusionSampler(model, config, device)
    return


@app.cell
def _(mo):
    prompt_input = mo.ui.text(value="Once upon a time", label="Prompt")
    num_tokens = mo.ui.slider(start=16, stop=128, value=64, label="Tokens to generate")
    temperature = mo.ui.slider(
        start=0.0, stop=2.0, value=1.0, step=0.1, label="Temperature"
    )

    mo.vstack([prompt_input, num_tokens, temperature])
    return


@app.cell
def _(prompt_input, tokenizer, num_tokens, sampler, temperature, mo):
    prompt = prompt_input.value
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)])

    generated = None
    steps = []

    for tokens, step in sampler.sample(
        prompt_tokens=prompt_tokens,
        num_tokens=int(num_tokens.value),
        temperature=float(temperature.value),
    ):
        if step % 32 == 0 or step == sampler.config.diffusion.num_steps:
            decoded = tokenizer.decode(tokens[0].tolist())
            steps.append((step, decoded))

    final = tokenizer.decode(
        generated[0].tolist() if generated is not None else tokens[0].tolist()
    )

    print(f"Prompt: {prompt}")
    print(f"\nGenerated:\n{final}")
    return


@app.cell
def _(steps, mo):
    for step, text in steps:
        print(f"\n--- Step {step} ---")
        print(text[:200] + "..." if len(text) > 200 else text)
    return


if __name__ == "__main__":
    app.run()
