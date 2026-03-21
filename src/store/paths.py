from pathlib import Path

ARTIFACTS_ROOT = Path("artifacts")
DATASETS_ROOT = ARTIFACTS_ROOT / "datasets"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
EXPORTS_ROOT = ARTIFACTS_ROOT / "exports"


def ensure_store_roots() -> None:
    for path in (ARTIFACTS_ROOT, DATASETS_ROOT, RUNS_ROOT, EXPORTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def dataset_root(source_name: str) -> Path:
    return DATASETS_ROOT / source_name


def prepared_dataset_dir(source_name: str, dataset_id: str) -> Path:
    return dataset_root(source_name) / "prepared" / dataset_id


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id
