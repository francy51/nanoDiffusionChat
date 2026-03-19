import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch

    print("Environment Check")
    print(f"Python version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return


@app.cell
def _():
    from src.tokenization.tokenizer import Tokenizer

    tokenizer = Tokenizer("gpt2")
    tok = tokenizer.load()
    print(f"Tokenizer loaded: {tokenizer.tokenizer_name}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Mask token ID: {tokenizer.mask_token_id}")

    sample = tokenizer.encode("Hello, world!")
    print(f"Encoded sample: {sample}")
    print(f"Decoded: {tokenizer.decode(sample)}")
    return


@app.cell
def _():
    from src.config.base import Config

    config = Config.tiny()
    print("Tiny config loaded")
    print(f"Model hidden dim: {config.model.hidden_dim}")
    print(f"Model layers: {config.model.num_layers}")
    print(f"Diffusion steps: {config.diffusion.num_steps}")
    return


if __name__ == "__main__":
    app.run()
