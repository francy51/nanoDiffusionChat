import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    from src.config.base import Config
    from src.models.denoiser import Denoiser
    import torch
    import marimo as mo

    config = Config.tiny()
    print(f"Config: {config.model}")
    return


@app.cell
def _(config):
    model = Denoiser(config.model, config.diffusion.num_steps)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.1f} MB (float32)")
    return


@app.cell
def _(model):
    print(model)
    return


@app.cell
def _(config, model):
    batch_size = 2
    seq_len = 64

    tokens = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    timesteps = torch.randint(0, config.diffusion.num_steps, (batch_size,))

    print(f"Input shape: {tokens.shape}")
    print(f"Timesteps: {timesteps}")

    logits = model(tokens, timesteps)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.model.vocab_size})")
    return


@app.cell
def _(config, model):
    from src.diffusion.corrupt import corrupt_tokens

    clean = torch.randint(0, 100, (1, 32))
    corrupted, mask = corrupt_tokens(clean, 0.5, config.diffusion.mask_token_id)

    print(f"Clean: {clean[0, :10]}")
    print(f"Corrupted: {corrupted[0, :10]}")
    print(f"Mask: {mask[0, :10]}")
    return


if __name__ == "__main__":
    app.run()
