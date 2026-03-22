from __future__ import annotations

from src.store.paths import ARTIFACTS_ROOT, DATASETS_ROOT, EXPORTS_ROOT, RUNS_ROOT

__all__ = [
    "ARTIFACTS_ROOT",
    "DATASETS_ROOT",
    "DatasetArtifact",
    "DatasetStore",
    "EXPORTS_ROOT",
    "RUNS_ROOT",
    "RunRecord",
    "RunStore",
]


def __getattr__(name: str):
    if name in {"DatasetArtifact", "DatasetStore"}:
        from src.store.dataset_store import DatasetArtifact, DatasetStore

        return {
            "DatasetArtifact": DatasetArtifact,
            "DatasetStore": DatasetStore,
        }[name]
    if name in {"RunRecord", "RunStore"}:
        from src.store.run_store import RunRecord, RunStore

        return {
            "RunRecord": RunRecord,
            "RunStore": RunStore,
        }[name]
    raise AttributeError(f"module 'src.store' has no attribute {name!r}")
