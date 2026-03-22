from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.serialization import load_json, save_json


@dataclass
class DatasetManifest:
    dataset_id: str
    source_name: str
    tokenizer_name: str
    format_name: str
    seq_len: int
    train_path: str
    val_path: str
    stats_path: str
    manifest_path: str
    vocab_size: int
    mask_token_id: int
    pad_token_id: int
    num_train_sequences: int
    num_val_sequences: int


@dataclass
class RunManifest:
    run_id: str
    dataset_id: str
    preset_name: str
    config_path: str
    run_dir: str


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def load_manifest(path: Path) -> dict[str, Any]:
    return load_json(path)


def dataset_manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    return asdict(manifest)


def dataset_manifest_from_dict(payload: dict[str, Any]) -> DatasetManifest:
    upgraded_payload = dict(payload)
    upgraded_payload.setdefault("format_name", "plain_text")
    return DatasetManifest(**upgraded_payload)


def run_manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return asdict(manifest)


def run_manifest_from_dict(payload: dict[str, Any]) -> RunManifest:
    return RunManifest(**payload)
