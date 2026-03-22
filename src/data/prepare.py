from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from src.config.schema import DatasetConfig
from src.data.instruction import (
    instruction_example_from_dict,
    serialize_instruction_example,
)
from src.store.manifests import DatasetManifest, dataset_manifest_to_dict, save_manifest
from src.store.paths import prepared_dataset_dir
from src.tokenization.tokenizer import Tokenizer
from src.utils.serialization import save_json


def dataset_id_from_config(config: DatasetConfig) -> str:
    """Build a stable prepared-dataset identifier from the dataset config."""
    return f"{config.source_name}_seq{config.seq_len}_{config.tokenizer_name}"


def _raw_source_dir(source_name: str) -> Path:
    return Path("artifacts") / "datasets" / source_name / "raw"


def _load_plain_text_documents(config: DatasetConfig) -> list[str]:
    raw_dir = _raw_source_dir(config.source_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    text_paths = sorted(raw_dir.glob("*.txt"))
    if not text_paths:
        raise FileNotFoundError(f"No raw text files found in {raw_dir}")

    documents = [
        path.read_text(encoding="utf-8").strip()
        for path in text_paths
        if path.read_text(encoding="utf-8").strip()
    ]
    if not documents:
        raise ValueError(f"No non-empty documents found in {raw_dir}")
    return documents


def _load_chat_transcript_documents(config: DatasetConfig) -> list[str]:
    raw_dir = _raw_source_dir(config.source_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    jsonl_paths = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL instruction files found in {raw_dir}")

    documents: list[str] = []
    for path in jsonl_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            example = instruction_example_from_dict(payload)
            documents.append(serialize_instruction_example(example))

    if not documents:
        raise ValueError(f"No instruction examples found in {raw_dir}")
    return documents


def _load_documents(config: DatasetConfig) -> list[str]:
    if config.format_name == "chat_transcript":
        return _load_chat_transcript_documents(config)
    return _load_plain_text_documents(config)


def _chunk_and_pad(
    token_ids: list[int],
    seq_len: int,
    pad_token_id: int,
) -> list[list[int]]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if not token_ids:
        return []

    chunks: list[list[int]] = []
    for start in range(0, len(token_ids), seq_len):
        chunk = token_ids[start : start + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [pad_token_id] * (seq_len - len(chunk))
        chunks.append(chunk)
    return chunks


def _build_sequences(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    seq_len: int,
) -> tuple[Tensor, int]:
    sequences: list[list[int]] = []
    total_tokens = 0
    for document in documents:
        if not document.strip():
            continue
        encoded = tokenizer.encode(document)
        total_tokens += len(encoded)
        sequences.extend(_chunk_and_pad(encoded, seq_len, tokenizer.pad_token_id))

    if not sequences:
        raise ValueError("Dataset preparation produced no token sequences")

    return torch.tensor(sequences, dtype=torch.long), total_tokens


def _split_sequences(
    sequences: Tensor,
    train_split: float,
    val_split: float,
) -> tuple[Tensor, Tensor]:
    assert sequences.dim() == 2, (
        f"Expected [num_sequences, seq_len], got {sequences.shape}"
    )

    num_sequences = int(sequences.shape[0])
    train_count = int(num_sequences * train_split)
    val_count = int(num_sequences * val_split)

    if num_sequences == 1:
        return sequences.clone(), sequences.new_empty((0, sequences.shape[1]))

    train_count = max(1, min(train_count, num_sequences - 1))
    val_count = min(val_count, num_sequences - train_count)

    if val_split > 0 and val_count == 0:
        val_count = 1
        train_count = max(1, num_sequences - val_count)

    train = sequences[:train_count].clone()
    val = sequences[train_count : train_count + val_count].clone()
    return train, val


def prepare_dataset(config: DatasetConfig) -> DatasetManifest:
    """Prepare a dataset artifact and return its manifest.

    Args:
        config: Dataset preparation settings. Raw inputs are read from
            `artifacts/datasets/<source_name>/raw`.

    Returns:
        Dataset manifest describing the prepared train/validation tensors and
        tokenizer metadata.
    """

    documents = _load_documents(config)
    corpus_text = "\n".join(documents) if config.tokenizer_name == "char" else None
    tokenizer = Tokenizer(config.tokenizer_name, corpus_text=corpus_text)
    tokenizer.load()

    dataset_id = dataset_id_from_config(config)
    output_dir = prepared_dataset_dir(config.source_name, dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.pt"
    val_path = output_dir / "val.pt"
    stats_path = output_dir / "stats.json"
    manifest_path = output_dir / "dataset_manifest.json"
    tokenizer_path = output_dir / "tokenizer.json"

    sequences, total_tokens = _build_sequences(documents, tokenizer, config.seq_len)
    train_sequences, val_sequences = _split_sequences(
        sequences,
        train_split=config.train_split,
        val_split=config.val_split,
    )

    torch.save(train_sequences, train_path)
    torch.save(val_sequences, val_path)
    tokenizer.save(tokenizer_path)

    stats = {
        "dataset_id": dataset_id,
        "source_name": config.source_name,
        "format_name": config.format_name,
        "tokenizer_name": config.tokenizer_name,
        "seq_len": config.seq_len,
        "vocab_size": tokenizer.vocab_size,
        "mask_token_id": tokenizer.mask_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "num_documents": len(documents),
        "num_sequences": int(sequences.shape[0]),
        "num_tokens": total_tokens,
        "num_train_sequences": int(train_sequences.shape[0]),
        "num_val_sequences": int(val_sequences.shape[0]),
    }
    save_json(stats_path, stats)

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        source_name=config.source_name,
        tokenizer_name=config.tokenizer_name,
        format_name=config.format_name,
        seq_len=config.seq_len,
        train_path=str(train_path),
        val_path=str(val_path),
        stats_path=str(stats_path),
        manifest_path=str(manifest_path),
        vocab_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_train_sequences=int(train_sequences.shape[0]),
        num_val_sequences=int(val_sequences.shape[0]),
    )
    save_manifest(manifest_path, dataset_manifest_to_dict(manifest))
    return manifest


def prepare_data(
    data_dir: Path,
    tokenizer_name: str = "char",
    seq_len: int = 512,
) -> tuple[Path, Path]:
    """Backward-compatible helper that prepares a TinyStories artifact.

    Args:
        data_dir: Unused output directory kept for API compatibility.
        tokenizer_name: Tokenizer name to build.
        seq_len: Fixed sequence length.

    Returns:
        Paths to the prepared train and validation tensors.
    """

    del data_dir
    manifest = prepare_dataset(
        DatasetConfig(
            source_name="tinystories",
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
        )
    )
    return Path(manifest.train_path), Path(manifest.val_path)
