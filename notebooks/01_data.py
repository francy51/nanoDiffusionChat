import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from src.config.base import Config, DataConfig
    from src.data.prepare import prepare_data, download_tinystories
    from src.tokenization.tokenizer import Tokenizer
    import marimo as mo

    data_dir = Path("data")
    config = Config.tiny()
    return


@app.cell
def _(data_dir):
    raw_path = data_dir / "tinystories_raw.txt"

    if not raw_path.exists():
        print("Downloading TinyStories dataset...")
        download_tinystories(data_dir)
    else:
        print(f"TinyStories already downloaded: {raw_path}")
        print(f"File size: {raw_path.stat().st_size / 1024 / 1024:.1f} MB")
    return


@app.cell
def _(config, data_dir):
    train_path, val_path = prepare_data(
        data_dir=data_dir,
        tokenizer_name=config.data.tokenizer_name,
        seq_len=config.data.seq_len,
        train_split=config.data.train_split,
    )
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    return


@app.cell
def _(train_path, val_path):
    import torch

    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")
    print(f"Sample sequence shape: {train_data[0].shape}")
    return


@app.cell
def _():
    from src.data.dataset import DiffusionDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionDataset(
        train_path,
        seq_len=512,
        mask_token_id=50256,
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    return


if __name__ == "__main__":
    app.run()
