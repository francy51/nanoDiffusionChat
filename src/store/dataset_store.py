from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.schema import DatasetConfig
from src.data.prepare import prepare_dataset
from src.store.manifests import (
    DatasetManifest,
    dataset_manifest_from_dict,
    load_manifest,
)
from src.store.paths import DATASETS_ROOT


@dataclass
class DatasetArtifact:
    dataset_id: str
    source_name: str
    tokenizer_name: str
    seq_len: int
    vocab_size: int
    mask_token_id: int
    pad_token_id: int
    train_path: Path
    val_path: Path
    stats_path: Path
    manifest_path: Path


class DatasetStore:
    def __init__(self, root: Path = DATASETS_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[DatasetArtifact]:
        artifacts: list[DatasetArtifact] = []
        for manifest_path in self.root.glob("*/prepared/*/dataset_manifest.json"):
            manifest = dataset_manifest_from_dict(load_manifest(manifest_path))
            artifacts.append(self._artifact_from_manifest(manifest))
        return sorted(artifacts, key=lambda item: item.dataset_id)

    def get(self, dataset_id: str) -> DatasetArtifact:
        matches = list(self.root.glob(f"*/prepared/{dataset_id}/dataset_manifest.json"))
        if not matches:
            raise FileNotFoundError(f"Unknown dataset artifact: {dataset_id}")
        manifest = dataset_manifest_from_dict(load_manifest(matches[0]))
        return self._artifact_from_manifest(manifest)

    def create_prepared_dataset(self, config: DatasetConfig) -> DatasetArtifact:
        manifest = prepare_dataset(config)
        return self._artifact_from_manifest(manifest)

    def _artifact_from_manifest(self, manifest: DatasetManifest) -> DatasetArtifact:
        return DatasetArtifact(
            dataset_id=manifest.dataset_id,
            source_name=manifest.source_name,
            tokenizer_name=manifest.tokenizer_name,
            seq_len=manifest.seq_len,
            vocab_size=manifest.vocab_size,
            mask_token_id=manifest.mask_token_id,
            pad_token_id=manifest.pad_token_id,
            train_path=Path(manifest.train_path),
            val_path=Path(manifest.val_path),
            stats_path=Path(manifest.stats_path),
            manifest_path=Path(manifest.manifest_path),
        )
