from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.io import save_experiment_config
from src.config.schema import ExperimentConfig
from src.store.manifests import (
    RunManifest,
    load_manifest,
    run_manifest_from_dict,
    save_manifest,
)
from src.store.paths import RUNS_ROOT, run_dir
from src.utils.serialization import append_jsonl, load_json, save_json


@dataclass
class RunRecord:
    run_id: str
    dataset_id: str
    run_dir: Path
    config_path: Path
    status_path: Path
    latest_checkpoint: Path | None
    best_checkpoint: Path | None


class RunStore:
    def __init__(self, root: Path = RUNS_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        config: ExperimentConfig,
        dataset_id: str,
        preset_name: str = "custom",
    ) -> RunRecord:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = (
            f"{timestamp}_{config.dataset.source_name}_{config.diffusion.algorithm}"
            f"_{preset_name}"
        )
        directory = run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        for child in ("checkpoints", "metrics", "samples", "exports"):
            (directory / child).mkdir(parents=True, exist_ok=True)

        config_path = directory / "config.json"
        run_path = directory / "run.json"
        status_path = directory / "status.json"
        save_experiment_config(config, config_path)
        save_manifest(
            run_path,
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "preset_name": preset_name,
                "config_path": str(config_path),
                "run_dir": str(directory),
            },
        )
        save_json(
            status_path,
            {
                "run_id": run_id,
                "state": "created",
                "step": 0,
                "latest_checkpoint": None,
                "best_checkpoint": None,
            },
        )
        return self.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for run_path in self.root.glob("*/run.json"):
            manifest = run_manifest_from_dict(load_manifest(run_path))
            records.append(self._record_from_manifest(manifest))
        return sorted(records, key=lambda item: item.run_id, reverse=True)

    def get(self, run_id: str) -> RunRecord:
        path = run_dir(run_id) / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        manifest = run_manifest_from_dict(load_manifest(path))
        return self._record_from_manifest(manifest)

    def resume_target(self, run_id: str) -> RunRecord:
        return self.get(run_id)

    def append_metric(self, run_id: str, stream: str, payload: dict[str, Any]) -> None:
        append_jsonl(run_dir(run_id) / "metrics" / f"{stream}.jsonl", payload)

    def update_status(self, run_id: str, payload: dict[str, Any]) -> None:
        status_path = run_dir(run_id) / "status.json"
        current = load_json(status_path) if status_path.exists() else {"run_id": run_id}
        current.update(payload)
        save_json(status_path, current)

    def checkpoint_path(self, run_id: str, name: str) -> Path:
        return run_dir(run_id) / "checkpoints" / name

    def _record_from_manifest(self, manifest: RunManifest) -> RunRecord:
        directory = Path(manifest.run_dir)
        status = load_json(directory / "status.json")
        latest = status.get("latest_checkpoint")
        best = status.get("best_checkpoint")
        return RunRecord(
            run_id=manifest.run_id,
            dataset_id=manifest.dataset_id,
            run_dir=directory,
            config_path=Path(manifest.config_path),
            status_path=directory / "status.json",
            latest_checkpoint=Path(latest) if latest else None,
            best_checkpoint=Path(best) if best else None,
        )
