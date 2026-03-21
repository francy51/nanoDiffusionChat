from pathlib import Path

import pytest
import torch

from src.config.presets import build_experiment_config
from src.store.paths import ensure_store_roots


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_store_roots()
    return tmp_path


@pytest.fixture
def debug_config():
    return build_experiment_config("debug")


@pytest.fixture
def sample_tokens():
    return torch.randint(2, 32, (4, 32))


@pytest.fixture
def tinystories_raw_text(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "artifacts" / "datasets" / "tinystories" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "tinystories.txt"
    raw_path.write_text(
        "tiny stories are useful for diffusion experiments\n" * 64,
        encoding="utf-8",
    )
    return raw_path
