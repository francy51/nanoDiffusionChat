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

    from src.config.schema import DatasetConfig
    from src.data.prepare import dataset_id_from_config
    from src.store import DatasetStore
    from src.tokenization.tokenizer import CharacterTokenizer, Tokenizer
    from src.utils.serialization import load_json

    def token_label(tokenizer: Tokenizer, token_id: int) -> str:
        inner = tokenizer.load()
        if isinstance(inner, CharacterTokenizer):
            return inner.itos.get(token_id, f"<id:{token_id}>")

        if hasattr(inner, "id_to_token"):
            label = inner.id_to_token(token_id)
            if label is not None:
                return label

        decoded = tokenizer.decode([token_id])
        return decoded if decoded else f"<id:{token_id}>"

    def summarize_split(
        tokens: torch.Tensor,
        pad_token_id: int,
    ) -> dict[str, float | int]:
        assert tokens.dim() == 2, (
            f"Expected [num_sequences, seq_len], got {tokens.shape}"
        )
        num_sequences, seq_len = tokens.shape
        total_tokens = int(tokens.numel())
        unique_tokens = int(torch.unique(tokens).numel())
        pad_tokens = int((tokens == pad_token_id).sum().item())
        non_pad_tokens = total_tokens - pad_tokens
        sampled_unique_sequences = torch.unique(
            tokens[: min(256, num_sequences)],
            dim=0,
        ).shape[0]
        return {
            "num_sequences": int(num_sequences),
            "seq_len": int(seq_len),
            "total_tokens": total_tokens,
            "unique_token_ids": unique_tokens,
            "pad_tokens": pad_tokens,
            "non_pad_fraction": round(non_pad_tokens / max(total_tokens, 1), 4),
            "sampled_unique_sequences": int(sampled_unique_sequences),
        }

    def preview_rows(
        tokens: torch.Tensor,
        tokenizer: Tokenizer,
        *,
        split_name: str,
        row_count: int,
    ) -> list[dict[str, str | int]]:
        rows: list[dict[str, str | int]] = []
        limit = min(int(tokens.shape[0]), row_count)
        for index in range(limit):
            token_ids = tokens[index].tolist()
            decoded = tokenizer.decode(token_ids).replace("\n", "\\n")
            rows.append(
                {
                    "split": split_name,
                    "row": index,
                    "chars": len(decoded),
                    "unique_token_ids": len(set(token_ids)),
                    "token_ids_preview": str(token_ids[:24]),
                    "decoded_preview": decoded[:240],
                }
            )
        return rows

    def top_token_rows(
        tokens: torch.Tensor,
        tokenizer: Tokenizer,
        *,
        top_k: int,
    ) -> list[dict[str, str | int | float]]:
        flat = tokens.reshape(-1)
        counts = torch.bincount(flat, minlength=int(flat.max().item()) + 1)
        total = max(int(counts.sum().item()), 1)
        values, indices = torch.topk(counts, k=min(top_k, int(counts.numel())))
        rows: list[dict[str, str | int | float]] = []
        for token_count, token_id in zip(
            values.tolist(),
            indices.tolist(),
            strict=False,
        ):
            rows.append(
                {
                    "token_id": int(token_id),
                    "token": token_label(tokenizer, int(token_id)),
                    "count": int(token_count),
                    "share": round(float(token_count) / total, 4),
                }
            )
        return rows

    return (
        DatasetConfig,
        DatasetStore,
        Tokenizer,
        dataset_id_from_config,
        load_json,
        mo,
        preview_rows,
        summarize_split,
        top_token_rows,
        torch,
    )


@app.cell
def _(mo):
    source_name = mo.ui.dropdown(
        options=["tinystories"],
        value="tinystories",
        label="Source",
    )
    tokenizer_name = mo.ui.dropdown(
        options=["char", "gpt2"],
        value="char",
        label="Tokenizer",
    )
    seq_len = mo.ui.number(value=128, start=32, stop=2048, label="Sequence length")
    train_split = mo.ui.slider(
        start=0.5,
        stop=0.99,
        step=0.01,
        value=0.9,
        label="Train split",
    )
    preview_count = mo.ui.number(value=5, start=1, stop=20, label="Preview rows")
    top_k = mo.ui.number(value=12, start=3, stop=50, label="Top tokens")
    controls = mo.vstack(
        [source_name, tokenizer_name, seq_len, train_split, preview_count, top_k]
    )
    return (
        controls,
        preview_count,
        seq_len,
        source_name,
        tokenizer_name,
        top_k,
        train_split,
    )


@app.cell
def _(
    DatasetConfig,
    dataset_id_from_config,
    mo,
    seq_len,
    source_name,
    tokenizer_name,
    train_split,
):
    config = DatasetConfig(
        source_name=source_name.value,
        tokenizer_name=tokenizer_name.value,
        seq_len=int(seq_len.value),
        train_split=float(train_split.value),
        val_split=round(1 - float(train_split.value), 2),
    )
    dataset_id = dataset_id_from_config(config)
    config_preview = mo.md(
        f"""
        ## Prepared dataset configuration
        - Dataset ID: `{dataset_id}`
        - Train split: `{config.train_split:.2f}`
        - Val split: `{config.val_split:.2f}`
        """
    )
    return config, config_preview, dataset_id


@app.cell
def _(DatasetStore, mo):
    dataset_store = DatasetStore()
    action = mo.ui.run_button(label="Prepare dataset")
    return action, dataset_store


@app.cell
def _(action, config, dataset_id, dataset_store):
    dataset = dataset_store.create_prepared_dataset(config) if action.value else None
    if dataset is None:
        try:
            dataset = dataset_store.get(dataset_id)
        except FileNotFoundError:
            dataset = None
    return (dataset,)


@app.cell
def _(action, config_preview, controls, dataset, mo):
    artifact_summary = (
        mo.callout(
            "No prepared artifact matches this configuration yet. "
            "Press `Prepare dataset` to build one.",
            kind="warn",
        )
        if dataset is None
        else mo.vstack(
            [
                mo.md("## Prepared artifact"),
                mo.ui.table(
                    [
                        {
                            "field": "dataset_id",
                            "value": dataset.dataset_id,
                        },
                        {
                            "field": "train_path",
                            "value": str(dataset.train_path),
                        },
                        {
                            "field": "val_path",
                            "value": str(dataset.val_path),
                        },
                        {
                            "field": "stats_path",
                            "value": str(dataset.stats_path),
                        },
                        {
                            "field": "vocab_size",
                            "value": dataset.vocab_size,
                        },
                        {
                            "field": "mask_token_id",
                            "value": dataset.mask_token_id,
                        },
                        {
                            "field": "pad_token_id",
                            "value": dataset.pad_token_id,
                        },
                    ]
                ),
            ]
        )
    )
    mo.vstack(
        [
            mo.md("# Data Lab"),
            controls,
            action,
            config_preview,
            artifact_summary,
        ],
        gap=1.25,
    )
    return


@app.cell
def _(Tokenizer, dataset, load_json, torch):
    if dataset is None:
        stats = None
        tokenizer = None
        train_tokens = None
        val_tokens = None
    else:
        train_tokens = torch.load(dataset.train_path, map_location="cpu")
        val_tokens = torch.load(dataset.val_path, map_location="cpu")
        stats = load_json(dataset.stats_path)
        tokenizer = Tokenizer.from_file(dataset.stats_path.parent / "tokenizer.json")
    return stats, tokenizer, train_tokens, val_tokens


@app.cell
def _(dataset, mo, stats, summarize_split, train_tokens, val_tokens):
    mo.stop(
        dataset is None or stats is None or train_tokens is None or val_tokens is None,
        mo.md("Prepare or load a dataset artifact to inspect it."),
    )

    train_summary = summarize_split(train_tokens, dataset.pad_token_id)
    val_summary = summarize_split(val_tokens, dataset.pad_token_id)
    mo.vstack(
        [
            mo.md(
                f"""
                ## Dataset overview
                - Total sequences in corpus: `{stats["num_sequences"]}`
                - Total tokens before chunking: `{stats["num_tokens"]}`
                - Declared vocab size: `{stats["vocab_size"]}`
                - Train/val split counts:
                  `{stats["num_train_sequences"]}` / `{stats["num_val_sequences"]}`
                """
            ),
            mo.ui.table(
                [
                    {
                        "metric": "dataset_id",
                        "train": dataset.dataset_id,
                        "val": dataset.dataset_id,
                    },
                    {
                        "metric": "source_name",
                        "train": stats["source_name"],
                        "val": stats["source_name"],
                    },
                    {
                        "metric": "seq_len",
                        "train": train_summary["seq_len"],
                        "val": val_summary["seq_len"],
                    },
                    {
                        "metric": "num_sequences",
                        "train": train_summary["num_sequences"],
                        "val": val_summary["num_sequences"],
                    },
                    {
                        "metric": "total_tokens",
                        "train": train_summary["total_tokens"],
                        "val": val_summary["total_tokens"],
                    },
                    {
                        "metric": "unique_token_ids",
                        "train": train_summary["unique_token_ids"],
                        "val": val_summary["unique_token_ids"],
                    },
                    {
                        "metric": "non_pad_fraction",
                        "train": train_summary["non_pad_fraction"],
                        "val": val_summary["non_pad_fraction"],
                    },
                    {
                        "metric": "file_size_mb",
                        "train": round(
                            dataset.train_path.stat().st_size / (1024 * 1024),
                            2,
                        ),
                        "val": round(
                            dataset.val_path.stat().st_size / (1024 * 1024),
                            2,
                        ),
                    },
                ]
            ),
        ]
    )
    return


@app.cell
def _(
    dataset,
    mo,
    preview_count,
    preview_rows,
    tokenizer,
    train_tokens,
    val_tokens,
):
    mo.stop(
        dataset is None
        or tokenizer is None
        or train_tokens is None
        or val_tokens is None,
        mo.md(""),
    )

    mo.vstack(
        [
            mo.md("## Visual inspection"),
            mo.accordion(
                {
                    "Train samples": mo.ui.table(
                        preview_rows(
                            train_tokens,
                            tokenizer,
                            split_name="train",
                            row_count=int(preview_count.value),
                        )
                    ),
                    "Validation samples": mo.ui.table(
                        preview_rows(
                            val_tokens,
                            tokenizer,
                            split_name="val",
                            row_count=int(preview_count.value),
                        )
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(dataset, mo, tokenizer, top_k, top_token_rows, train_tokens):
    mo.stop(
        dataset is None or tokenizer is None or train_tokens is None,
        mo.md(""),
    )
    mo.vstack(
        [
            mo.md("## Token frequency snapshot"),
            mo.ui.table(
                top_token_rows(
                    train_tokens,
                    tokenizer,
                    top_k=int(top_k.value),
                )
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
